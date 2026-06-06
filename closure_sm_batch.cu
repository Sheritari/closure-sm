#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#if defined(_MSC_VER)
#include <intrin.h>
#endif

#define CUDA_CHECK(x)                                                                                                  \
    do {                                                                                                               \
        cudaError_t e = (x);                                                                                           \
        if (e != cudaSuccess) {                                                                                        \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__);               \
            std::exit(1);                                                                                              \
        }                                                                                                              \
    } while (0)

using u32 = std::uint32_t;
__device__ u32* g_d_mu_lut_dev = nullptr;

namespace {

constexpr int K = 5;
constexpr int K2 = K * K;
constexpr u32 NN = 1u << K2;
constexpr int BS_WORDS = static_cast<int>(NN / 64u);
constexpr u32 MASK_FULL = NN - 1u;

constexpr int THREADS = 256;
constexpr int THREADS_TURBO = 512;
constexpr int MAX_PARALLEL_SEEDS = 16;
constexpr int MAX_BFS_INNER_ROUNDS = 50000;
constexpr int MAX_OUTER_LAYERS_DEFAULT = 100000;
// Orbit60 batch: эмпирически fixpoint на outer layer 2 (кластеры A/B); cap 3 — запас, не замена all_stuck.
constexpr int MAX_OUTER_LAYERS_ORBIT60_DEFAULT = 3;
constexpr int CLUSTER_A_MAX = 65544;
constexpr int WS_BIT_U64 = 7 * BS_WORDS;
constexpr int WS_LIST_INTS = 2 * BS_WORDS + 2;
constexpr int WS_STRIDE_U64 = WS_BIT_U64 + (WS_LIST_INTS * static_cast<int>(sizeof(int)) + 7) / 8;
constexpr std::size_t MU_LUT_BYTES = static_cast<std::size_t>(NN) * sizeof(u32);
constexpr std::size_t WS_BYTES = static_cast<std::size_t>(WS_STRIDE_U64) * sizeof(std::uint64_t);

u32* h_mu_lut = nullptr;

__host__ __device__ __forceinline__ u32 sheffer_mu_loop(u32 m) {
    u32 t = 0;
    for (int i = 0; i < K; ++i)
        for (int j = 0; j < K; ++j)
            if ((m >> (K * j + i)) & 1u) t |= 1u << (K * i + j);
    return t & MASK_FULL;
}

__host__ bool mu_lut_load_file(const char* path) {
    if (h_mu_lut) {
        std::free(h_mu_lut);
        h_mu_lut = nullptr;
    }
    FILE* f = std::fopen(path, "rb");
    if (!f) return false;
    if (std::fseek(f, 0, SEEK_END) != 0) {
        std::fclose(f);
        return false;
    }
    if (std::ftell(f) != static_cast<long>(MU_LUT_BYTES)) {
        std::fclose(f);
        return false;
    }
    std::rewind(f);
    h_mu_lut = static_cast<u32*>(std::malloc(MU_LUT_BYTES));
    if (!h_mu_lut) {
        std::fclose(f);
        return false;
    }
    if (std::fread(h_mu_lut, 1, MU_LUT_BYTES, f) != MU_LUT_BYTES) {
        std::free(h_mu_lut);
        h_mu_lut = nullptr;
        std::fclose(f);
        return false;
    }
    std::fclose(f);
    return true;
}

__host__ bool mu_lut_upload_device() {
    if (!h_mu_lut) return false;
    u32* p = nullptr;
    if (cudaMalloc(&p, MU_LUT_BYTES) != cudaSuccess) return false;
    if (cudaMemcpy(p, h_mu_lut, MU_LUT_BYTES, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(p);
        return false;
    }
    if (cudaMemcpyToSymbol(g_d_mu_lut_dev, &p, sizeof(u32*)) != cudaSuccess) {
        cudaFree(p);
        return false;
    }
    return true;
}

__host__ void mu_lut_free_all() {
    u32* p = nullptr;
    if (cudaMemcpyFromSymbol(&p, g_d_mu_lut_dev, sizeof(u32*)) == cudaSuccess && p) cudaFree(p);
    u32* nullp = nullptr;
    cudaMemcpyToSymbol(g_d_mu_lut_dev, &nullp, sizeof(u32*));
    if (h_mu_lut) {
        std::free(h_mu_lut);
        h_mu_lut = nullptr;
    }
}

__host__ __device__ __forceinline__ u32 sheffer_mu(u32 m) {
    m &= MASK_FULL;
#ifdef __CUDA_ARCH__
    if (g_d_mu_lut_dev) return g_d_mu_lut_dev[m];
#else
    if (h_mu_lut) return h_mu_lut[m];
#endif
    return sheffer_mu_loop(m);
}

__host__ __device__ __forceinline__ u32 d_e_const() {
    u32 e = 0;
    for (int i = 0; i < K; ++i) e |= 1u << (K * i + i);
    return e;
}

__host__ __device__ __forceinline__ u32 sheffer_star(u32 a, u32 b) {
    u32 c = 0;
    for (int j = 0; j < K; ++j) {
        for (int i = 0; i < K; ++i) {
            int v = 0;
            for (int kk = 0; kk < K; ++kk)
                v |= static_cast<int>(((a >> (K * kk + i)) & 1u) & ((b >> (K * j + kk)) & 1u));
            if (v) c |= 1u << (K * j + i);
        }
    }
    return c & MASK_FULL;
}

__host__ __device__ __forceinline__ u32 sheffer_cap(u32 a, u32 b) { return (a & b) & MASK_FULL; }

__device__ __forceinline__ bool d_test_bit(const std::uint64_t* bs, int x) {
    return (bs[x >> 6] >> (x & 63)) & 1ULL;
}

__device__ __forceinline__ void d_set_bit(std::uint64_t* bs, int x) { bs[x >> 6] |= (1ULL << (x & 63)); }

__device__ __forceinline__ bool d_try_set_bit(std::uint64_t* bs, int x) {
    const std::uint64_t mask = 1ULL << (x & 63);
    auto* ptr = reinterpret_cast<unsigned long long*>(&bs[x >> 6]);
    const unsigned long long old = atomicOr(ptr, static_cast<unsigned long long>(mask));
    return (old & mask) == 0ULL;
}

__device__ void sheffer_insert_S(std::uint64_t* visited, std::uint64_t* new_front, std::uint64_t* mu_done,
                                 std::uint64_t* from_mu, unsigned int* any_new, int* vis_count, int* over_limit,
                                 int limit_count, int v) {
    v &= MASK_FULL;
    if (d_test_bit(visited, v)) return;
    if (!d_try_set_bit(visited, v)) return;
    if (vis_count) {
        const int now = atomicAdd(vis_count, 1) + 1;
        if (over_limit && limit_count > 0 && now > limit_count) atomicExch(over_limit, 1);
    }
    d_try_set_bit(new_front, v);
    atomicOr(any_new, 1u);

    if (d_test_bit(from_mu, v)) return;
    if (d_test_bit(mu_done, v)) return;
    d_set_bit(mu_done, v);

    const int t = static_cast<int>(sheffer_mu(static_cast<u32>(v)));
    if (d_test_bit(visited, t)) return;
    if (!d_try_set_bit(visited, t)) return;
    if (vis_count) {
        const int now = atomicAdd(vis_count, 1) + 1;
        if (over_limit && limit_count > 0 && now > limit_count) atomicExch(over_limit, 1);
    }
    d_set_bit(from_mu, t);
    d_try_set_bit(new_front, t);
    atomicOr(any_new, 1u);
}

__device__ void sheffer_expand_generator_a(u32 a, const std::uint64_t* snap, std::uint64_t* visited,
                                           std::uint64_t* new_front, std::uint64_t* mu_done, std::uint64_t* from_mu,
                                           unsigned int* any_new, int* vis_count, int* over_limit, int limit_count,
                                           int tid, int T) {
    if (tid == 0) {
        const u32 mu_a = sheffer_mu(a);
        if (!d_test_bit(visited, static_cast<int>(mu_a)))
            sheffer_insert_S(visited, new_front, mu_done, from_mu, any_new, vis_count, over_limit, limit_count,
                             static_cast<int>(mu_a));
    }
    __syncthreads();

    for (int wb = tid; wb < BS_WORDS; wb += T) {
        if (over_limit && *over_limit) break;
        std::uint64_t vb = snap[wb];
        while (vb) {
            if (over_limit && *over_limit) break;
            const int bb = __ffsll(static_cast<long long>(vb)) - 1;
            vb &= vb - 1;
            const u32 b = static_cast<u32>((wb << 6) + bb);
            const int ab = static_cast<int>(sheffer_star(a, b));
            if (!d_test_bit(visited, ab))
                sheffer_insert_S(visited, new_front, mu_done, from_mu, any_new, vis_count, over_limit, limit_count, ab);
            const int ba = static_cast<int>(sheffer_star(b, a));
            if (!d_test_bit(visited, ba))
                sheffer_insert_S(visited, new_front, mu_done, from_mu, any_new, vis_count, over_limit, limit_count, ba);
            const int cab = static_cast<int>(sheffer_cap(a, b));
            if (!d_test_bit(visited, cab))
                sheffer_insert_S(visited, new_front, mu_done, from_mu, any_new, vis_count, over_limit, limit_count,
                                 cab);
        }
    }
    __syncthreads();
}

__host__ __device__ __forceinline__ std::uint64_t* ws_visited(std::uint64_t* ws) { return ws; }
__host__ __device__ __forceinline__ std::uint64_t* ws_frontier(std::uint64_t* ws) { return ws + BS_WORDS; }
__host__ __device__ __forceinline__ std::uint64_t* ws_new_front(std::uint64_t* ws) { return ws + 2 * BS_WORDS; }
__host__ __device__ __forceinline__ std::uint64_t* ws_snap(std::uint64_t* ws) { return ws + 3 * BS_WORDS; }
__host__ __device__ __forceinline__ std::uint64_t* ws_mu_done(std::uint64_t* ws) { return ws + 4 * BS_WORDS; }
__host__ __device__ __forceinline__ std::uint64_t* ws_from_mu(std::uint64_t* ws) { return ws + 5 * BS_WORDS; }
__host__ __device__ __forceinline__ std::uint64_t* ws_expanded(std::uint64_t* ws) { return ws + 6 * BS_WORDS; }
__host__ __device__ __forceinline__ int* ws_fr_w(std::uint64_t* ws) {
    return reinterpret_cast<int*>(ws + WS_BIT_U64);
}
__host__ __device__ __forceinline__ int* ws_vis_w(std::uint64_t* ws) { return ws_fr_w(ws) + BS_WORDS; }
__host__ __device__ __forceinline__ int* ws_fr_n(std::uint64_t* ws) { return ws_vis_w(ws) + BS_WORDS; }

__global__ void k5_sm_init_seeds(std::uint64_t* __restrict__ ws_base, int ws_stride_u64,
                                 const u32* __restrict__ seeds) {
    std::uint64_t* ws = ws_base + static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(ws_stride_u64);
    const u32 f = seeds[blockIdx.x];
    const int tid = threadIdx.x;
    const int T = blockDim.x;
    for (int i = tid; i < ws_stride_u64; i += T) ws[i] = 0ULL;
    __syncthreads();
    if (tid != 0) return;

    std::uint64_t* visited = ws_visited(ws);
    std::uint64_t* frontier = ws_frontier(ws);
    std::uint64_t* mu_done = ws_mu_done(ws);
    std::uint64_t* from_mu = ws_from_mu(ws);
    const u32 S0[4] = {0u, d_e_const(), MASK_FULL, f};
    for (int k = 0; k < 4; ++k) {
        const int v = static_cast<int>(S0[k]);
        d_set_bit(visited, v);
        d_set_bit(frontier, v);
        d_set_bit(mu_done, v);
        const int t = static_cast<int>(sheffer_mu(S0[k]));
        if (!d_test_bit(visited, t)) {
            d_set_bit(visited, t);
            d_set_bit(frontier, t);
            d_set_bit(from_mu, t);
        }
    }
}

__device__ void dev_bfs_prepare(std::uint64_t* ws, int tid, int T) {
    std::uint64_t* visited = ws_visited(ws);
    std::uint64_t* frontier = ws_frontier(ws);
    std::uint64_t* new_front = ws_new_front(ws);
    std::uint64_t* snap = ws_snap(ws);
    std::uint64_t* expanded = ws_expanded(ws);
    int* fr_w = ws_fr_w(ws);
    int* fr_n = ws_fr_n(ws);

    if (tid == 0) *fr_n = 0;
    __syncthreads();
    for (int w = tid; w < BS_WORDS; w += T) {
        snap[w] = visited[w];
        const std::uint64_t fu = visited[w] & ~expanded[w];
        frontier[w] = fu;
        if (fu) {
            const int idx = atomicAdd(fr_n, 1);
            if (idx < BS_WORDS) fr_w[idx] = w;
        }
    }
    __syncthreads();
    if (tid == 0) {
        int n = *fr_n;
        if (n > BS_WORDS) n = BS_WORDS;
        *fr_n = n;
    }
    __syncthreads();
    for (int i = tid; i < BS_WORDS; i += T) new_front[i] = 0ULL;
}

__device__ void dev_bfs_expand_all_range(std::uint64_t* ws, int fr_n_cap, unsigned int* any_new, int* vis_count,
                                         int* over_limit, int limit_count, int tid, int T, int block_id,
                                         int block_count) {
    std::uint64_t* visited = ws_visited(ws);
    std::uint64_t* frontier = ws_frontier(ws);
    std::uint64_t* new_front = ws_new_front(ws);
    const std::uint64_t* snap = ws_snap(ws);
    std::uint64_t* mu_done = ws_mu_done(ws);
    std::uint64_t* from_mu = ws_from_mu(ws);
    std::uint64_t* expanded = ws_expanded(ws);
    const int* fr_w = ws_fr_w(ws);

    for (int fi = block_id; fi < fr_n_cap; fi += block_count) {
        if (over_limit && *over_limit) break;
        const int wa = fr_w[fi];
        std::uint64_t fa = frontier[wa];
        while (fa) {
            if (over_limit && *over_limit) break;
            const int ba = __ffsll(static_cast<long long>(fa)) - 1;
            fa &= fa - 1;
            const u32 a = static_cast<u32>((wa << 6) + ba);
            sheffer_expand_generator_a(a, snap, visited, new_front, mu_done, from_mu, any_new, vis_count, over_limit,
                                       limit_count, tid, T);
            auto* exp_ptr = reinterpret_cast<unsigned long long*>(&expanded[static_cast<int>(a) >> 6]);
            atomicOr(exp_ptr, 1ULL << (a & 63));
            __syncthreads();
        }
    }
}

__device__ void dev_bfs_expand_all(std::uint64_t* ws, int fr_n_cap, unsigned int* any_new, int* vis_count, int* over_limit,
                                   int limit_count, int tid, int T) {
    dev_bfs_expand_all_range(ws, fr_n_cap, any_new, vis_count, over_limit, limit_count, tid, T, 0, 1);
}

// Внутренний BFS на device: prepare + expand до пустого frontier (один sync на host).
__global__ void k5_sm_inner_drain(std::uint64_t* __restrict__ ws_base, int ws_stride_u64,
                                  const unsigned char* __restrict__ d_active, unsigned int* __restrict__ d_any_per_block,
                                  int* __restrict__ d_live_counts, int* __restrict__ d_over_limit, int limit_count,
                                  int max_inner_rounds, int one_inner_only) {
    if (d_active && d_active[blockIdx.x] == 0) return;
    std::uint64_t* ws = ws_base + static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(ws_stride_u64);
    const int tid = threadIdx.x;
    const int T = blockDim.x;
    unsigned int* const any_new = d_any_per_block ? &d_any_per_block[blockIdx.x] : nullptr;
    int* const vis_count = d_live_counts ? &d_live_counts[blockIdx.x] : nullptr;
    int* const over_limit = d_over_limit ? &d_over_limit[blockIdx.x] : nullptr;

    for (int inner = 0; inner < max_inner_rounds; ++inner) {
        if (over_limit && *over_limit) break;
        dev_bfs_prepare(ws, tid, T);

        __shared__ int s_fr_n;
        if (tid == 0) s_fr_n = *ws_fr_n(ws);
        __syncthreads();
        if (s_fr_n == 0) break;

        if (any_new && tid == 0) *any_new = 0u;
        __syncthreads();

        dev_bfs_expand_all(ws, s_fr_n, any_new, vis_count, over_limit, limit_count, tid, T);

        if (one_inner_only) break;
        __syncthreads();
    }
}

// Single-seed turbo: host-chunked multi-block expand (как probe_k5_closure_hybrid).

__global__ void k5_sm_single_prepare(std::uint64_t* __restrict__ ws) {
    dev_bfs_prepare(ws, threadIdx.x, blockDim.x);
}

__global__ void k5_sm_single_expand(std::uint64_t* __restrict__ ws, int fi_begin, int fi_end,
                                    unsigned int* __restrict__ d_any) {
    std::uint64_t* visited = ws_visited(ws);
    std::uint64_t* frontier = ws_frontier(ws);
    std::uint64_t* new_front = ws_new_front(ws);
    const std::uint64_t* snap = ws_snap(ws);
    std::uint64_t* mu_done = ws_mu_done(ws);
    std::uint64_t* from_mu = ws_from_mu(ws);
    std::uint64_t* expanded = ws_expanded(ws);
    const int* fr_w = ws_fr_w(ws);
    const int tid = threadIdx.x;
    const int T = blockDim.x;
    const int bid = blockIdx.x;
    const int nblocks = gridDim.x;

    __shared__ int s_fr_n;
    if (tid == 0) {
        s_fr_n = *ws_fr_n(ws);
        if (s_fr_n > BS_WORDS) s_fr_n = BS_WORDS;
    }
    __syncthreads();
    if (fi_end > s_fr_n) fi_end = s_fr_n;
    if (fi_begin < 0) fi_begin = 0;

    for (int fi = fi_begin + bid; fi < fi_end; fi += nblocks) {
        const int wa = fr_w[fi];
        std::uint64_t fa = frontier[wa];
        while (fa) {
            const int ba = __ffsll(static_cast<long long>(fa)) - 1;
            fa &= fa - 1;
            const u32 a = static_cast<u32>((wa << 6) + ba);
            sheffer_expand_generator_a(a, snap, visited, new_front, mu_done, from_mu, d_any, nullptr, nullptr, 0, tid,
                                       T);
            auto* exp_ptr = reinterpret_cast<unsigned long long*>(&expanded[static_cast<int>(a) >> 6]);
            atomicOr(exp_ptr, 1ULL << (a & 63));
            __syncthreads();
        }
    }
}

__global__ void k5_sm_popcount_visited(std::uint64_t* __restrict__ ws_base, int ws_stride_u64,
                                         int* __restrict__ out_counts) {
    std::uint64_t* ws = ws_base + static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(ws_stride_u64);
    const int tid = threadIdx.x;
    const int T = blockDim.x;
    __shared__ int warp_sums[32];
    int local = 0;
    const std::uint64_t* visited = ws_visited(ws);
    for (int i = tid; i < BS_WORDS; i += T) local += __popcll(visited[i]);
    for (int off = 16; off > 0; off >>= 1) local += __shfl_xor_sync(0xFFFFFFFFu, local, off);
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    if (lane == 0) warp_sums[warp_id] = local;
    __syncthreads();
    if (tid == 0) {
        int total = 0;
        const int nwarps = (T + 31) >> 5;
        for (int w = 0; w < nwarps; ++w) total += warp_sums[w];
        out_counts[blockIdx.x] = total;
    }
}

// --- Host helpers ---

static void host_get_counts(const std::uint64_t* d_ws, int grid_seeds, int* d_counts, std::vector<int>* out) {
    out->resize(static_cast<std::size_t>(grid_seeds));
    CUDA_CHECK(cudaMemcpy(out->data(), d_counts, static_cast<std::size_t>(grid_seeds) * sizeof(int),
                          cudaMemcpyDeviceToHost));
}

// Один «внешний слой»: drain inner на device, один cudaDeviceSynchronize.
static void run_one_outer_layer(std::uint64_t* d_ws, int grid_seeds, unsigned int* d_any, int* d_counts,
                                unsigned char* d_active, int* d_live_counts, int* d_over_limit, int limit_count,
                                bool one_inner_only) {
    const int max_inner = one_inner_only ? 1 : MAX_BFS_INNER_ROUNDS;
    k5_sm_inner_drain<<<grid_seeds, THREADS>>>(d_ws, WS_STRIDE_U64, d_active, d_any, d_live_counts, d_over_limit,
                                               limit_count, max_inner, one_inner_only ? 1 : 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    k5_sm_popcount_visited<<<grid_seeds, THREADS>>>(d_ws, WS_STRIDE_U64, d_counts);
    CUDA_CHECK(cudaDeviceSynchronize());
}

static void run_closure_layers(std::uint64_t* d_ws, int grid_seeds, unsigned int* d_any, int* d_counts,
                               unsigned char* d_active, int* d_live_counts, int* d_over_limit, int max_outer_layers,
                               bool print_layers, bool one_inner_only, bool skip_over_65544) {
    std::vector<int> counts(static_cast<std::size_t>(grid_seeds), 0);
    std::vector<unsigned char> active(static_cast<std::size_t>(grid_seeds), 1);
    if (d_active) {
        CUDA_CHECK(cudaMemcpy(d_active, active.data(), static_cast<std::size_t>(grid_seeds) * sizeof(unsigned char),
                              cudaMemcpyHostToDevice));
    }
    k5_sm_popcount_visited<<<grid_seeds, THREADS>>>(d_ws, WS_STRIDE_U64, d_counts);
    CUDA_CHECK(cudaDeviceSynchronize());
    host_get_counts(d_ws, grid_seeds, d_counts, &counts);
    if (d_live_counts) {
        CUDA_CHECK(cudaMemcpy(d_live_counts, counts.data(), static_cast<std::size_t>(grid_seeds) * sizeof(int),
                              cudaMemcpyHostToDevice));
    }
    if (d_over_limit) {
        std::vector<int> zero_over(static_cast<std::size_t>(grid_seeds), 0);
        CUDA_CHECK(cudaMemcpy(d_over_limit, zero_over.data(), static_cast<std::size_t>(grid_seeds) * sizeof(int),
                              cudaMemcpyHostToDevice));
    }
    if (print_layers && grid_seeds == 1) {
        std::printf("layer 0 |S|=%d\n", counts[0]);
        std::fflush(stdout);
    }

    for (int L = 1; L <= max_outer_layers; ++L) {
        std::vector<int> before = counts;
        run_one_outer_layer(d_ws, grid_seeds, d_any, d_counts, d_active, d_live_counts, d_over_limit,
                            skip_over_65544 ? CLUSTER_A_MAX : 0, one_inner_only);
        host_get_counts(d_ws, grid_seeds, d_counts, &counts);

        if (skip_over_65544) {
            bool need_sync = false;
            std::vector<int> over(static_cast<std::size_t>(grid_seeds), 0);
            if (d_over_limit) {
                CUDA_CHECK(cudaMemcpy(over.data(), d_over_limit, static_cast<std::size_t>(grid_seeds) * sizeof(int),
                                      cudaMemcpyDeviceToHost));
            }
            for (int b = 0; b < grid_seeds; ++b) {
                if (!active[b]) continue;
                if (counts[b] > CLUSTER_A_MAX || over[b]) {
                    active[b] = 0;
                    need_sync = true;
                }
            }
            if (need_sync && d_active) {
                CUDA_CHECK(cudaMemcpy(d_active, active.data(),
                                      static_cast<std::size_t>(grid_seeds) * sizeof(unsigned char),
                                      cudaMemcpyHostToDevice));
            }
        }

        if (print_layers) {
            if (grid_seeds == 1) {
                std::printf("layer %d |S|=%d (+%d)%s\n", L, counts[0], counts[0] - before[0],
                            one_inner_only ? " [one-inner]" : "");
            } else {
                std::printf("layer %d", L);
                for (int b = 0; b < grid_seeds; ++b)
                    std::printf("  b%d|S|=%d(+%d)", b, counts[b], counts[b] - before[b]);
                std::printf("%s\n", one_inner_only ? " [one-inner]" : "");
            }
            std::fflush(stdout);
        }

        bool all_stuck = true;
        bool all_full = true;
        int active_count = 0;
        for (int b = 0; b < grid_seeds; ++b) {
            if (!active[b]) continue;
            ++active_count;
            if (counts[b] != before[b]) all_stuck = false;
            if (static_cast<u32>(counts[b]) < NN) all_full = false;
        }
        if (active_count == 0) break;
        if (all_stuck || all_full) break;
    }
}

static double now_sec() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static int max_parallel_for_vram() {
    std::size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) return 4;
    const std::size_t need = WS_BYTES + sizeof(int) + 4096u;
    int n = static_cast<int>(free_b / need);
    if (n < 1) n = 1;
    if (n > MAX_PARALLEL_SEEDS) n = MAX_PARALLEL_SEEDS;
    return n;
}

static constexpr std::size_t OFF_FR_N =
    static_cast<std::size_t>(WS_BIT_U64) * sizeof(std::uint64_t) + 2u * static_cast<std::size_t>(BS_WORDS) * sizeof(int);

static int turbo_read_fr_n(const std::uint64_t* d_ws) {
    int h_fr = 0;
    CUDA_CHECK(cudaMemcpy(&h_fr, reinterpret_cast<const char*>(d_ws) + OFF_FR_N, sizeof(int), cudaMemcpyDeviceToHost));
    return h_fr;
}

static int turbo_device_popcount(std::uint64_t* d_ws, int* d_count) {
    k5_sm_popcount_visited<<<1, THREADS>>>(d_ws, WS_STRIDE_U64, d_count);
    CUDA_CHECK(cudaDeviceSynchronize());
    int n = 0;
    CUDA_CHECK(cudaMemcpy(&n, d_count, sizeof(int), cudaMemcpyDeviceToHost));
    return n;
}

static int turbo_expand_grid(int fr_words) {
    static int cached = 0;
    if (cached <= 0) {
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
            cached = 64;
        } else {
            cached = prop.multiProcessorCount * 4;
            if (cached < 32) cached = 32;
            if (cached > 512) cached = 512;
        }
    }
    if (fr_words < 1) return 1;
    return std::min(cached, fr_words);
}

static void turbo_launch_expand(std::uint64_t* d_ws, unsigned int* d_any, int fi0, int fi1) {
    const int fr_range = fi1 - fi0;
    if (fr_range <= 0) return;
    const int grid = turbo_expand_grid(fr_range);
    CUDA_CHECK(cudaMemset(d_any, 0, sizeof(unsigned int)));
    k5_sm_single_expand<<<grid, THREADS_TURBO>>>(d_ws, fi0, fi1, d_any);
    CUDA_CHECK(cudaDeviceSynchronize());
}

static int turbo_inner_drain(std::uint64_t* d_ws, unsigned int* d_any, int* d_count, int max_rounds,
                             bool one_inner_only) {
    for (int inner = 0; inner < max_rounds; ++inner) {
        k5_sm_single_prepare<<<1, THREADS_TURBO>>>(d_ws);
        CUDA_CHECK(cudaDeviceSynchronize());

        int h_fr = turbo_read_fr_n(d_ws);
        if (h_fr == 0) break;
        if (h_fr > BS_WORDS) h_fr = BS_WORDS;

        turbo_launch_expand(d_ws, d_any, 0, h_fr);

        if (one_inner_only) break;
    }
    return turbo_device_popcount(d_ws, d_count);
}

static void turbo_run_closure_layers(std::uint64_t* d_ws, unsigned int* d_any, int* d_counts, int max_outer_layers,
                                     bool print_layers, bool one_inner_only) {
    int h_vis = turbo_device_popcount(d_ws, d_counts);
    if (print_layers) {
        std::printf("layer 0 |S|=%d\n", h_vis);
        std::fflush(stdout);
    }

    const int max_inner = one_inner_only ? 1 : MAX_BFS_INNER_ROUNDS;
    for (int L = 1; L <= max_outer_layers; ++L) {
        const int before = h_vis;
        h_vis = turbo_inner_drain(d_ws, d_any, d_counts, max_inner, one_inner_only);
        if (print_layers) {
            std::printf("layer %d |S|=%d (+%d)%s\n", L, h_vis, h_vis - before, one_inner_only ? " [one-inner]" : "");
            std::fflush(stdout);
        }
        if (h_vis == before) break;
        if (static_cast<u32>(h_vis) >= NN) break;
    }
}

static void run_chunk_closure(std::uint64_t* d_ws, u32* d_seeds, unsigned int* d_any, int* d_counts, const u32* h_seeds,
                              unsigned char* d_active, int* d_live_counts, int* d_over_limit, int chunk_n, int max_layers, bool one_inner_only,
                              bool skip_over_65544) {
    CUDA_CHECK(cudaMemcpy(d_seeds, h_seeds, static_cast<std::size_t>(chunk_n) * sizeof(u32), cudaMemcpyHostToDevice));
    k5_sm_init_seeds<<<chunk_n, THREADS>>>(d_ws, WS_STRIDE_U64, d_seeds);
    CUDA_CHECK(cudaDeviceSynchronize());
    run_closure_layers(d_ws, chunk_n, d_any, d_counts, d_active, d_live_counts, d_over_limit, max_layers, false,
                       one_inner_only, skip_over_65544);
}

static double bench_sequential(std::uint64_t* d_ws, u32* d_seeds, unsigned int* d_any, int* d_counts,
                               unsigned char* d_active, int* d_live_counts, int* d_over_limit,
                               const std::vector<u32>& seeds, int max_layers,
                               bool one_inner_only) {
    const auto t0 = now_sec();
    for (u32 s : seeds) {
        run_chunk_closure(d_ws, d_seeds, d_any, d_counts, &s, d_active, d_live_counts, d_over_limit, 1, max_layers,
                          one_inner_only, false);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    return now_sec() - t0;
}

static double bench_parallel(std::uint64_t* d_ws, u32* d_seeds, unsigned int* d_any, int* d_counts,
                             unsigned char* d_active, int* d_live_counts, int* d_over_limit,
                             const std::vector<u32>& seeds, int parallel, int max_layers, bool one_inner_only) {
    const auto t0 = now_sec();
    for (std::size_t off = 0; off < seeds.size();) {
        const int chunk = static_cast<int>(std::min<std::size_t>(seeds.size() - off, static_cast<std::size_t>(parallel)));
        run_chunk_closure(d_ws, d_seeds, d_any, d_counts, seeds.data() + off, d_active, d_live_counts, d_over_limit,
                          chunk, max_layers, one_inner_only, false);
        off += static_cast<std::size_t>(chunk);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    return now_sec() - t0;
}

// --- CLI ---

struct SeedRec {
    u32 hex = 0;
    int line_no = 0;
};

static bool parse_orbit60_line(const char* line, SeedRec* out) {
    // Строки #DONE ... и заголовок # ... — уже посчитаны, пропускаем.
    if (!line || line[0] == '#' || line[0] == '\n') return false;
    int a = 0, b = 0, c = 0, orbit = 0, cnt = 0;
    unsigned int hex = 0;
    if (std::sscanf(line, "%d %d %d 0x%x %d %d", &a, &b, &c, &hex, &orbit, &cnt) < 4) return false;
    out->hex = hex;
    return true;
}

static bool load_orbit60_seeds(const char* path, int limit, std::vector<SeedRec>* seeds) {
    FILE* f = std::fopen(path, "r");
    if (!f) return false;
    char buf[512];
    int line_no = 0;
    while (std::fgets(buf, sizeof(buf), f)) {
        ++line_no;
        SeedRec r{};
        r.line_no = line_no;
        if (!parse_orbit60_line(buf, &r)) continue;
        seeds->push_back(r);
        if (limit > 0 && static_cast<int>(seeds->size()) >= limit) break;
    }
    std::fclose(f);
    return !seeds->empty();
}

static void usage(const char* prog) {
    std::fprintf(stderr,
                "Usage:\n"
                "  %s <seed_hex> [--layers N] [--one-inner] [--no-mu-lut] [--no-turbo]\n"
                "  %s --orbit60 <file> [--limit N] [--parallel P] [--bench] [--one-inner]\n"
                "      (без --layers: max outer layers=%d; строки #DONE пропускаются)\n"
                "\n"
                "  single seed: --turbo (default) = probe-style 512-thread expand по всем SM.\n"
                "  --no-turbo: <<<1,256>>> device inner drain (медленнее, но проще).\n"
                "  --parallel P: grid=P блоков, blockIdx.x * WS_STRIDE — свой seed на SM.\n"
                "  --bench: сравнить sequential (P=1) vs parallel на том же наборе seed.\n"
                "  --one-inner: один prepare+expand за внешний слой (быстрый прогон, != snap-BFS).\n"
                "  --layers N: cap внешних слоев; fixpoint раньше — all_stuck (|S| не растет) / all_full.\n"
                "  --skip-over-65544: в batch отключать seed после первого |S|>65544 (ускорение, не strict |S|∞).\n",
                prog, prog, MAX_OUTER_LAYERS_ORBIT60_DEFAULT);
}

static int run_single_seed(u32 seed, int max_layers, bool use_lut, bool one_inner_only, bool use_turbo) {
    if (use_lut && mu_lut_load_file("mu_lut_k5.bin")) (void)mu_lut_upload_device();

    std::uint64_t* d_ws = nullptr;
    u32* d_seeds = nullptr;
    unsigned int* d_any = nullptr;
    int* d_counts = nullptr;
    unsigned char* d_active = nullptr;
    int* d_live_counts = nullptr;
    int* d_over_limit = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ws, WS_BYTES));
    CUDA_CHECK(cudaMalloc(&d_seeds, sizeof(u32)));
    CUDA_CHECK(cudaMalloc(&d_any, sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_counts, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_active, sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_live_counts, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_over_limit, sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_seeds, &seed, sizeof(u32), cudaMemcpyHostToDevice));

    int gpu_n = 0;
    if (use_turbo) {
        const int grid = turbo_expand_grid(BS_WORDS);
        std::printf("[turbo] probe-style: %d expand blocks x %d threads\n", grid, THREADS_TURBO);
        std::fflush(stdout);
        k5_sm_init_seeds<<<1, THREADS>>>(d_ws, WS_STRIDE_U64, d_seeds);
        CUDA_CHECK(cudaDeviceSynchronize());
        turbo_run_closure_layers(d_ws, d_any, d_counts, max_layers, true, one_inner_only);
        CUDA_CHECK(cudaMemcpy(&gpu_n, d_counts, sizeof(int), cudaMemcpyDeviceToHost));
    } else {
        k5_sm_init_seeds<<<1, THREADS>>>(d_ws, WS_STRIDE_U64, d_seeds);
        CUDA_CHECK(cudaDeviceSynchronize());
        run_closure_layers(d_ws, 1, d_any, d_counts, d_active, d_live_counts, d_over_limit, max_layers, true,
                           one_inner_only, false);
        CUDA_CHECK(cudaMemcpy(&gpu_n, d_counts, sizeof(int), cudaMemcpyDeviceToHost));
    }
    std::printf("seed=0x%x  |S|=%d  NN=%u  sheffer=%s\n", seed, gpu_n, NN, gpu_n == int(NN) ? "YES" : "NO");

    cudaFree(d_ws);
    cudaFree(d_seeds);
    cudaFree(d_any);
    cudaFree(d_counts);
    cudaFree(d_active);
    cudaFree(d_live_counts);
    cudaFree(d_over_limit);
    mu_lut_free_all();
    return (gpu_n >= int(NN)) ? 0 : 1;
}

static int run_orbit60_batch(const char* path, int limit, int max_layers, bool use_lut, bool one_inner_only,
                             int parallel, bool do_bench, bool skip_over_65544) {
    std::vector<SeedRec> seed_recs;
    if (!load_orbit60_seeds(path, limit, &seed_recs)) {
        std::fprintf(stderr, "Failed to read pending seeds from %s (all #DONE?)\n", path);
        return 1;
    }
    std::printf("orbit60 pending seeds to run: %zu (lines #DONE skipped)  max_outer_layers=%d\n", seed_recs.size(),
                max_layers);
    std::fflush(stdout);
    std::vector<u32> seeds;
    seeds.reserve(seed_recs.size());
    for (const auto& r : seed_recs) seeds.push_back(r.hex);

    if (use_lut && mu_lut_load_file("mu_lut_k5.bin")) (void)mu_lut_upload_device();

    const int vram_max = max_parallel_for_vram();
    if (parallel < 1) parallel = 1;
    if (parallel > vram_max) {
        std::printf("[warn] --parallel %d > vram_max %d, clamping\n", parallel, vram_max);
        parallel = vram_max;
    }
    const int alloc_grid = do_bench ? vram_max : parallel;

    std::uint64_t* d_ws = nullptr;
    u32* d_seeds = nullptr;
    unsigned int* d_any = nullptr;
    int* d_counts = nullptr;
    unsigned char* d_active = nullptr;
    int* d_live_counts = nullptr;
    int* d_over_limit = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ws, WS_BYTES * static_cast<std::size_t>(alloc_grid)));
    CUDA_CHECK(cudaMalloc(&d_seeds, static_cast<std::size_t>(alloc_grid) * sizeof(u32)));
    CUDA_CHECK(cudaMalloc(&d_any, static_cast<std::size_t>(alloc_grid) * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_counts, static_cast<std::size_t>(alloc_grid) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_active, static_cast<std::size_t>(alloc_grid) * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_live_counts, static_cast<std::size_t>(alloc_grid) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_over_limit, static_cast<std::size_t>(alloc_grid) * sizeof(int)));

    if (do_bench) {
        std::printf("[bench] seeds=%zu  layers=%d  one_inner=%d  ws_per_seed=%.1f MiB  vram_max_parallel=%d\n",
                    seeds.size(), max_layers, one_inner_only ? 1 : 0, WS_BYTES / (1024.0 * 1024.0), vram_max);
        std::fflush(stdout);

        run_chunk_closure(d_ws, d_seeds, d_any, d_counts, seeds.data(), d_active, d_live_counts, d_over_limit, 1, 1,
                          one_inner_only, false);

        const double t_seq = bench_sequential(d_ws, d_seeds, d_any, d_counts, d_active, d_live_counts, d_over_limit,
                                              seeds, max_layers, one_inner_only);
        std::printf("[bench] sequential (grid=1): %.3f s  (%.4f s/seed)\n", t_seq, t_seq / seeds.size());

        std::vector<int> try_p = {2, 4, 8};
        if (parallel > 1) try_p.push_back(parallel);
        std::sort(try_p.begin(), try_p.end());
        try_p.erase(std::unique(try_p.begin(), try_p.end()), try_p.end());

        for (int p : try_p) {
            if (p < 2 || p > vram_max) continue;
            if (p > static_cast<int>(seeds.size())) continue;
            const double t_par = bench_parallel(d_ws, d_seeds, d_any, d_counts, d_active, d_live_counts, d_over_limit,
                                                seeds, p, max_layers, one_inner_only);
            std::printf("[bench] parallel grid=%d: %.3f s  (%.4f s/seed)  speedup=%.2fx\n", p, t_par,
                        t_par / seeds.size(), t_seq / t_par);
        }
    } else {
        const auto t0 = now_sec();
        if (parallel <= 1) {
            for (std::size_t i = 0; i < seeds.size(); ++i) {
                run_chunk_closure(d_ws, d_seeds, d_any, d_counts, &seeds[i], d_active, d_live_counts, d_over_limit, 1,
                                  max_layers, one_inner_only, skip_over_65544);
                int n = 0;
                CUDA_CHECK(cudaMemcpy(&n, d_counts, sizeof(int), cudaMemcpyDeviceToHost));
                if (skip_over_65544 && n > CLUSTER_A_MAX) {
                    std::printf("[%zu] seed=0x%x  |S|>%d  [skipped]\n", i + 1, seeds[i], CLUSTER_A_MAX);
                } else {
                    std::printf("[%zu] seed=0x%x  |S|=%d\n", i + 1, seeds[i], n);
                }
            }
        } else {
            for (std::size_t off = 0; off < seeds.size();) {
                const int chunk =
                    static_cast<int>(std::min<std::size_t>(seeds.size() - off, static_cast<std::size_t>(parallel)));
                run_chunk_closure(d_ws, d_seeds, d_any, d_counts, seeds.data() + off, d_active, d_live_counts,
                                  d_over_limit, chunk, max_layers, one_inner_only, skip_over_65544);
                std::vector<int> ns(static_cast<std::size_t>(chunk));
                CUDA_CHECK(cudaMemcpy(ns.data(), d_counts, static_cast<std::size_t>(chunk) * sizeof(int),
                                      cudaMemcpyDeviceToHost));
                for (int b = 0; b < chunk; ++b) {
                    if (skip_over_65544 && ns[b] > CLUSTER_A_MAX) {
                        std::printf("[%zu] seed=0x%x  |S|>%d  [skipped]  (block %d)\n", off + b + 1, seeds[off + b],
                                    CLUSTER_A_MAX, b);
                    } else {
                        std::printf("[%zu] seed=0x%x  |S|=%d  (block %d)\n", off + b + 1, seeds[off + b], ns[b], b);
                    }
                }
                off += static_cast<std::size_t>(chunk);
            }
        }
        const double elapsed = now_sec() - t0;
        std::printf("done %zu seeds  parallel=%d  elapsed=%.3f s  (%.4f s/seed)\n", seeds.size(), parallel, elapsed,
                    elapsed / seeds.size());
    }

    cudaFree(d_ws);
    cudaFree(d_seeds);
    cudaFree(d_any);
    cudaFree(d_counts);
    cudaFree(d_active);
    cudaFree(d_live_counts);
    cudaFree(d_over_limit);
    mu_lut_free_all();
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc < 2) {
        usage(argv[0]);
        return 1;
    }

    bool orbit60 = false;
    bool layers_explicit = false;
    bool one_inner_only = false;
    bool use_lut = true;
    bool do_bench = false;
    bool skip_over_65544 = false;
    bool use_turbo = true;
    int max_layers = MAX_OUTER_LAYERS_DEFAULT;
    int limit = 0;
    int parallel = 1;
    const char* orbit_path = "k5_ge36000_orbit60.txt";
    u32 seed = 0;
    bool have_seed = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--orbit60") == 0 && i + 1 < argc) {
            orbit60 = true;
            orbit_path = argv[++i];
        } else if (std::strcmp(argv[i], "--limit") == 0 && i + 1 < argc) {
            limit = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--layers") == 0 && i + 1 < argc) {
            max_layers = std::atoi(argv[++i]);
            layers_explicit = true;
        } else if (std::strcmp(argv[i], "--parallel") == 0 && i + 1 < argc) {
            parallel = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--bench") == 0) {
            do_bench = true;
        } else if (std::strcmp(argv[i], "--one-inner") == 0) {
            one_inner_only = true;
        } else if (std::strcmp(argv[i], "--no-mu-lut") == 0) {
            use_lut = false;
        } else if (std::strcmp(argv[i], "--skip-over-65544") == 0) {
            skip_over_65544 = true;
        } else if (std::strcmp(argv[i], "--no-turbo") == 0) {
            use_turbo = false;
        } else if (argv[i][0] != '-') {
            seed = static_cast<u32>(std::strtoul(argv[i], nullptr, 0));
            have_seed = true;
        } else {
            std::fprintf(stderr, "Unknown option: %s\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }

    if (orbit60) {
        if (!layers_explicit) max_layers = MAX_OUTER_LAYERS_ORBIT60_DEFAULT;
        return run_orbit60_batch(orbit_path, limit, max_layers, use_lut, one_inner_only, parallel, do_bench,
                                 skip_over_65544);
    }
    if (!have_seed) {
        usage(argv[0]);
        return 1;
    }
    return run_single_seed(seed, max_layers, use_lut, one_inner_only, use_turbo);
}
