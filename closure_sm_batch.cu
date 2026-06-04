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

using MatrixMask = std::uint32_t;
__device__ MatrixMask* device_mu_lookup_table = nullptr;

namespace {

constexpr int matrix_order = 5;
constexpr int matrix_bit_count = matrix_order * matrix_order;
constexpr MatrixMask universe_size = 1u << matrix_bit_count;
constexpr int bitmap_word_count = static_cast<int>(universe_size / 64u);
constexpr MatrixMask full_matrix_mask = universe_size - 1u;

constexpr int block_thread_count = 256;
constexpr int max_parallel_seed_count = 16;
constexpr int max_inner_bfs_rounds = 50000;
constexpr int default_max_outer_layers = 100000;

constexpr int orbit60_default_max_outer_layers = 3;
constexpr int cluster_a_max_closure_size = 65544;
constexpr int workspace_bitmap_u64_count = 7 * bitmap_word_count;
constexpr int workspace_list_int_count = 2 * bitmap_word_count + 2;
constexpr int workspace_stride_u64 = workspace_bitmap_u64_count + (workspace_list_int_count * static_cast<int>(sizeof(int)) + 7) / 8;
constexpr std::size_t mu_lookup_table_bytes = static_cast<std::size_t>(universe_size) * sizeof(MatrixMask);
constexpr std::size_t workspace_bytes_per_seed = static_cast<std::size_t>(workspace_stride_u64) * sizeof(std::uint64_t);

MatrixMask* host_mu_lookup_table = nullptr;

__host__ __device__ __forceinline__ MatrixMask compute_transpose_without_lookup(MatrixMask m) {
    MatrixMask t = 0;
    for (int i = 0; i < matrix_order; ++i)
        for (int j = 0; j < matrix_order; ++j)
            if ((m >> (matrix_order * j + i)) & 1u) t |= 1u << (matrix_order * i + j);
    return t & full_matrix_mask;
}

__host__ bool load_mu_lookup_from_file(const char* path) {
    if (host_mu_lookup_table) {
        std::free(host_mu_lookup_table);
        host_mu_lookup_table = nullptr;
    }
    FILE* f = std::fopen(path, "rb");
    if (!f) return false;
    if (std::fseek(f, 0, SEEK_END) != 0) {
        std::fclose(f);
        return false;
    }
    if (std::ftell(f) != static_cast<long>(mu_lookup_table_bytes)) {
        std::fclose(f);
        return false;
    }
    std::rewind(f);
    host_mu_lookup_table = static_cast<MatrixMask*>(std::malloc(mu_lookup_table_bytes));
    if (!host_mu_lookup_table) {
        std::fclose(f);
        return false;
    }
    if (std::fread(host_mu_lookup_table, 1, mu_lookup_table_bytes, f) != mu_lookup_table_bytes) {
        std::free(host_mu_lookup_table);
        host_mu_lookup_table = nullptr;
        std::fclose(f);
        return false;
    }
    std::fclose(f);
    return true;
}

__host__ bool upload_mu_lookup_to_device() {
    if (!host_mu_lookup_table) return false;
    MatrixMask* p = nullptr;
    if (cudaMalloc(&p, mu_lookup_table_bytes) != cudaSuccess) return false;
    if (cudaMemcpy(p, host_mu_lookup_table, mu_lookup_table_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(p);
        return false;
    }
    if (cudaMemcpyToSymbol(device_mu_lookup_table, &p, sizeof(MatrixMask*)) != cudaSuccess) {
        cudaFree(p);
        return false;
    }
    return true;
}

__host__ void free_mu_lookup_tables() {
    MatrixMask* p = nullptr;
    if (cudaMemcpyFromSymbol(&p, device_mu_lookup_table, sizeof(MatrixMask*)) == cudaSuccess && p) cudaFree(p);
    MatrixMask* nullp = nullptr;
    cudaMemcpyToSymbol(device_mu_lookup_table, &nullp, sizeof(MatrixMask*));
    if (host_mu_lookup_table) {
        std::free(host_mu_lookup_table);
        host_mu_lookup_table = nullptr;
    }
}

__host__ __device__ __forceinline__ MatrixMask compute_transpose(MatrixMask m) {
    m &= full_matrix_mask;
#ifdef __CUDA_ARCH__
    if (device_mu_lookup_table) return device_mu_lookup_table[m];
#else
    if (host_mu_lookup_table) return host_mu_lookup_table[m];
#endif
    return compute_transpose_without_lookup(m);
}

__host__ __device__ __forceinline__ MatrixMask identity_matrix_mask() {
    MatrixMask e = 0;
    for (int i = 0; i < matrix_order; ++i) e |= 1u << (matrix_order * i + i);
    return e;
}

__host__ __device__ __forceinline__ MatrixMask sheffer_star_product(MatrixMask a, MatrixMask b) {
    MatrixMask c = 0;
    for (int j = 0; j < matrix_order; ++j) {
        for (int i = 0; i < matrix_order; ++i) {
            int v = 0;
            for (int kk = 0; kk < matrix_order; ++kk)
                v |= static_cast<int>(((a >> (matrix_order * kk + i)) & 1u) & ((b >> (matrix_order * j + kk)) & 1u));
            if (v) c |= 1u << (matrix_order * j + i);
        }
    }
    return c & full_matrix_mask;
}

__host__ __device__ __forceinline__ MatrixMask sheffer_cap_product(MatrixMask a, MatrixMask b) { return (a & b) & full_matrix_mask; }

__device__ __forceinline__ bool bitmap_test_bit(const std::uint64_t* bs, int x) {
    return (bs[x >> 6] >> (x & 63)) & 1ULL;
}

__device__ __forceinline__ void bitmap_set_bit(std::uint64_t* bs, int x) { bs[x >> 6] |= (1ULL << (x & 63)); }

__device__ __forceinline__ bool bitmap_try_set_bit(std::uint64_t* bs, int x) {
    const std::uint64_t mask = 1ULL << (x & 63);
    auto* ptr = reinterpret_cast<unsigned long long*>(&bs[x >> 6]);
    const unsigned long long old = atomicOr(ptr, static_cast<unsigned long long>(mask));
    return (old & mask) == 0ULL;
}

__device__ void insert_into_closure_set(std::uint64_t* visited, std::uint64_t* next_frontier, std::uint64_t* transpose_already_applied,
                                 std::uint64_t* came_from_transpose, unsigned int* any_new_elements_flag, int* live_visit_count, int* over_size_limit_flag,
                                 int max_closure_size_limit, int v) {
    v &= full_matrix_mask;
    if (bitmap_test_bit(visited, v)) return;
    if (!bitmap_try_set_bit(visited, v)) return;
    if (live_visit_count) {
        const int now = atomicAdd(live_visit_count, 1) + 1;
        if (over_size_limit_flag && max_closure_size_limit > 0 && now > max_closure_size_limit) atomicExch(over_size_limit_flag, 1);
    }
    bitmap_try_set_bit(next_frontier, v);
    atomicOr(any_new_elements_flag, 1u);

    if (bitmap_test_bit(came_from_transpose, v)) return;
    if (bitmap_test_bit(transpose_already_applied, v)) return;
    bitmap_set_bit(transpose_already_applied, v);

    const int t = static_cast<int>(compute_transpose(static_cast<MatrixMask>(v)));
    if (bitmap_test_bit(visited, t)) return;
    if (!bitmap_try_set_bit(visited, t)) return;
    if (live_visit_count) {
        const int now = atomicAdd(live_visit_count, 1) + 1;
        if (over_size_limit_flag && max_closure_size_limit > 0 && now > max_closure_size_limit) atomicExch(over_size_limit_flag, 1);
    }
    bitmap_set_bit(came_from_transpose, t);
    bitmap_try_set_bit(next_frontier, t);
    atomicOr(any_new_elements_flag, 1u);
}

__device__ void expand_generator_vertex(MatrixMask a, const std::uint64_t* snap, std::uint64_t* visited,
                                           std::uint64_t* next_frontier, std::uint64_t* transpose_already_applied, std::uint64_t* came_from_transpose,
                                           unsigned int* any_new_elements_flag, int* live_visit_count, int* over_size_limit_flag, int max_closure_size_limit,
                                           int thread_index, int threads_per_block) {
    if (thread_index == 0) {
        const MatrixMask mu_a = compute_transpose(a);
        if (!bitmap_test_bit(visited, static_cast<int>(mu_a)))
            insert_into_closure_set(visited, next_frontier, transpose_already_applied, came_from_transpose, any_new_elements_flag, live_visit_count, over_size_limit_flag, max_closure_size_limit,
                             static_cast<int>(mu_a));
    }
    __syncthreads();

    for (int wb = thread_index; wb < bitmap_word_count; wb += threads_per_block) {
        if (over_size_limit_flag && *over_size_limit_flag) break;
        std::uint64_t vb = snap[wb];
        while (vb) {
            if (over_size_limit_flag && *over_size_limit_flag) break;
            const int bb = __ffsll(static_cast<long long>(vb)) - 1;
            vb &= vb - 1;
            const MatrixMask b = static_cast<MatrixMask>((wb << 6) + bb);
            insert_into_closure_set(visited, next_frontier, transpose_already_applied, came_from_transpose, any_new_elements_flag, live_visit_count, over_size_limit_flag, max_closure_size_limit,
                             static_cast<int>(sheffer_star_product(a, b)));
            insert_into_closure_set(visited, next_frontier, transpose_already_applied, came_from_transpose, any_new_elements_flag, live_visit_count, over_size_limit_flag, max_closure_size_limit,
                             static_cast<int>(sheffer_star_product(b, a)));
            insert_into_closure_set(visited, next_frontier, transpose_already_applied, came_from_transpose, any_new_elements_flag, live_visit_count, over_size_limit_flag, max_closure_size_limit,
                             static_cast<int>(sheffer_cap_product(a, b)));
        }
    }
    __syncthreads();
}

__host__ __device__ __forceinline__ std::uint64_t* workspace_visited_bitmap(std::uint64_t* ws) { return ws; }
__host__ __device__ __forceinline__ std::uint64_t* workspace_frontier_bitmap(std::uint64_t* ws) { return ws + bitmap_word_count; }
__host__ __device__ __forceinline__ std::uint64_t* workspace_next_frontier_bitmap(std::uint64_t* ws) { return ws + 2 * bitmap_word_count; }
__host__ __device__ __forceinline__ std::uint64_t* workspace_snapshot_bitmap(std::uint64_t* ws) { return ws + 3 * bitmap_word_count; }
__host__ __device__ __forceinline__ std::uint64_t* workspace_transpose_done_bitmap(std::uint64_t* ws) { return ws + 4 * bitmap_word_count; }
__host__ __device__ __forceinline__ std::uint64_t* workspace_from_transpose_bitmap(std::uint64_t* ws) { return ws + 5 * bitmap_word_count; }
__host__ __device__ __forceinline__ std::uint64_t* workspace_expanded_bitmap(std::uint64_t* ws) { return ws + 6 * bitmap_word_count; }
__host__ __device__ __forceinline__ int* workspace_frontier_word_indices(std::uint64_t* ws) {
    return reinterpret_cast<int*>(ws + workspace_bitmap_u64_count);
}
__host__ __device__ __forceinline__ int* workspace_unused_vis_words(std::uint64_t* ws) { return workspace_frontier_word_indices(ws) + bitmap_word_count; }
__host__ __device__ __forceinline__ int* workspace_frontier_word_count_ptr(std::uint64_t* ws) { return workspace_unused_vis_words(ws) + bitmap_word_count; }

__global__ void init_seed_workspaces_kernel(std::uint64_t* __restrict__ workspace_base, int workspace_stride_u64,
                                 const MatrixMask* __restrict__ seeds) {
    std::uint64_t* ws = workspace_base + static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(workspace_stride_u64);
    const MatrixMask seed_mask = seeds[blockIdx.x];
    const int thread_index = threadIdx.x;
    const int threads_per_block = blockDim.x;
    for (int i = thread_index; i < workspace_stride_u64; i += threads_per_block) ws[i] = 0ULL;
    __syncthreads();
    if (thread_index != 0) return;

    std::uint64_t* visited = workspace_visited_bitmap(ws);
    std::uint64_t* frontier = workspace_frontier_bitmap(ws);
    std::uint64_t* transpose_already_applied = workspace_transpose_done_bitmap(ws);
    std::uint64_t* came_from_transpose = workspace_from_transpose_bitmap(ws);
    const MatrixMask initial_generators[4] = {0u, identity_matrix_mask(), full_matrix_mask, seed_mask};
    for (int generator_index = 0; generator_index < 4; ++generator_index) {
        const int vertex_index = static_cast<int>(initial_generators[generator_index]);
        bitmap_set_bit(visited, vertex_index);
        bitmap_set_bit(frontier, vertex_index);
        bitmap_set_bit(transpose_already_applied, vertex_index);
        const int transposed_vertex = static_cast<int>(compute_transpose(initial_generators[generator_index]));
        if (!bitmap_test_bit(visited, transposed_vertex)) {
            bitmap_set_bit(visited, transposed_vertex);
            bitmap_set_bit(frontier, transposed_vertex);
            bitmap_set_bit(came_from_transpose, transposed_vertex);
        }
    }
}

__device__ void bfs_prepare_round(std::uint64_t* ws, int thread_index, int threads_per_block) {
    std::uint64_t* visited = workspace_visited_bitmap(ws);
    std::uint64_t* frontier = workspace_frontier_bitmap(ws);
    std::uint64_t* next_frontier = workspace_next_frontier_bitmap(ws);
    std::uint64_t* snap = workspace_snapshot_bitmap(ws);
    std::uint64_t* expanded = workspace_expanded_bitmap(ws);
    int* frontier_word_indices = workspace_frontier_word_indices(ws);
    int* frontier_word_count = workspace_frontier_word_count_ptr(ws);

    if (thread_index == 0) *frontier_word_count = 0;
    __syncthreads();
    for (int w = thread_index; w < bitmap_word_count; w += threads_per_block) {
        snap[w] = visited[w];
        const std::uint64_t fu = visited[w] & ~expanded[w];
        frontier[w] = fu;
        if (fu) {
            const int idx = atomicAdd(frontier_word_count, 1);
            if (idx < bitmap_word_count) frontier_word_indices[idx] = w;
        }
    }
    __syncthreads();
    if (thread_index == 0) {
        int n = *frontier_word_count;
        if (n > bitmap_word_count) n = bitmap_word_count;
        *frontier_word_count = n;
    }
    __syncthreads();
    for (int i = thread_index; i < bitmap_word_count; i += threads_per_block) next_frontier[i] = 0ULL;
}

__device__ void bfs_expand_round(std::uint64_t* ws, int frontier_word_limit, unsigned int* any_new_elements_flag, int* live_visit_count, int* over_size_limit_flag,
                                   int max_closure_size_limit, int thread_index, int threads_per_block) {
    std::uint64_t* visited = workspace_visited_bitmap(ws);
    std::uint64_t* frontier = workspace_frontier_bitmap(ws);
    std::uint64_t* next_frontier = workspace_next_frontier_bitmap(ws);
    const std::uint64_t* snap = workspace_snapshot_bitmap(ws);
    std::uint64_t* transpose_already_applied = workspace_transpose_done_bitmap(ws);
    std::uint64_t* came_from_transpose = workspace_from_transpose_bitmap(ws);
    std::uint64_t* expanded = workspace_expanded_bitmap(ws);
    const int* frontier_word_indices = workspace_frontier_word_indices(ws);

    for (int fi = 0; fi < frontier_word_limit; ++fi) {
        if (over_size_limit_flag && *over_size_limit_flag) break;
        const int wa = frontier_word_indices[fi];
        std::uint64_t fa = frontier[wa];
        while (fa) {
            if (over_size_limit_flag && *over_size_limit_flag) break;
            const int ba = __ffsll(static_cast<long long>(fa)) - 1;
            fa &= fa - 1;
            const MatrixMask a = static_cast<MatrixMask>((wa << 6) + ba);
            expand_generator_vertex(a, snap, visited, next_frontier, transpose_already_applied, came_from_transpose, any_new_elements_flag, live_visit_count, over_size_limit_flag,
                                       max_closure_size_limit, thread_index, threads_per_block);
            auto* exp_ptr = reinterpret_cast<unsigned long long*>(&expanded[static_cast<int>(a) >> 6]);
            atomicOr(exp_ptr, 1ULL << (a & 63));
            __syncthreads();
        }
    }
}

__global__ void inner_bfs_drain_kernel(std::uint64_t* __restrict__ workspace_base, int workspace_stride_u64,
                                  const unsigned char* __restrict__ device_seed_active_flags, unsigned int* __restrict__ d_any_per_block,
                                  int* __restrict__ device_live_visit_counts, int* __restrict__ device_over_size_limit_flags, int max_closure_size_limit,
                                  int max_inner_rounds, int single_inner_round_only) {
    if (device_seed_active_flags && device_seed_active_flags[blockIdx.x] == 0) return;
    std::uint64_t* ws = workspace_base + static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(workspace_stride_u64);
    const int thread_index = threadIdx.x;
    const int threads_per_block = blockDim.x;
    unsigned int* const any_new_elements_flag = d_any_per_block ? &d_any_per_block[blockIdx.x] : nullptr;
    int* const live_visit_count = device_live_visit_counts ? &device_live_visit_counts[blockIdx.x] : nullptr;
    int* const over_size_limit_flag = device_over_size_limit_flags ? &device_over_size_limit_flags[blockIdx.x] : nullptr;

    for (int inner = 0; inner < max_inner_rounds; ++inner) {
        if (over_size_limit_flag && *over_size_limit_flag) break;
        bfs_prepare_round(ws, thread_index, threads_per_block);

        __shared__ int shared_frontier_word_count;
        if (thread_index == 0) shared_frontier_word_count = *workspace_frontier_word_count_ptr(ws);
        __syncthreads();
        if (shared_frontier_word_count == 0) break;

        if (any_new_elements_flag && thread_index == 0) *any_new_elements_flag = 0u;
        __syncthreads();

        bfs_expand_round(ws, shared_frontier_word_count, any_new_elements_flag, live_visit_count, over_size_limit_flag, max_closure_size_limit, thread_index, threads_per_block);

        if (single_inner_round_only) break;
        __syncthreads();
    }
}

__global__ void popcount_visited_kernel(std::uint64_t* __restrict__ workspace_base, int workspace_stride_u64,
                                         int* __restrict__ out_counts) {
    std::uint64_t* ws = workspace_base + static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(workspace_stride_u64);
    const int thread_index = threadIdx.x;
    const int threads_per_block = blockDim.x;
    __shared__ int warp_partial_sums[32];
    int local = 0;
    const std::uint64_t* visited = workspace_visited_bitmap(ws);
    for (int i = thread_index; i < bitmap_word_count; i += threads_per_block) local += __popcll(visited[i]);
    for (int off = 16; off > 0; off >>= 1) local += __shfl_xor_sync(0xFFFFFFFFu, local, off);
    const int warp_index = thread_index >> 5;
    const int lane = thread_index & 31;
    if (lane == 0) warp_partial_sums[warp_index] = local;
    __syncthreads();
    if (thread_index == 0) {
        int total = 0;
        const int warp_count = (threads_per_block + 31) >> 5;
        for (int w = 0; w < warp_count; ++w) total += warp_partial_sums[w];
        out_counts[blockIdx.x] = total;
    }
}

static void copy_closure_sizes_from_device(const std::uint64_t* device_workspaces, int parallel_seed_count, int* device_closure_sizes, std::vector<int>* out) {
    out->resize(static_cast<std::size_t>(parallel_seed_count));
    CUDA_CHECK(cudaMemcpy(out->data(), device_closure_sizes, static_cast<std::size_t>(parallel_seed_count) * sizeof(int),
                          cudaMemcpyDeviceToHost));
}

static void run_single_outer_layer(std::uint64_t* device_workspaces, int parallel_seed_count, unsigned int* device_any_new_flags, int* device_closure_sizes,
                                unsigned char* device_seed_active_flags, int* device_live_visit_counts, int* device_over_size_limit_flags, int max_closure_size_limit,
                                bool single_inner_round_only) {
    const int max_inner = single_inner_round_only ? 1 : max_inner_bfs_rounds;
    inner_bfs_drain_kernel<<<parallel_seed_count, block_thread_count>>>(device_workspaces, workspace_stride_u64, device_seed_active_flags, device_any_new_flags, device_live_visit_counts, device_over_size_limit_flags,
                                               max_closure_size_limit, max_inner, single_inner_round_only ? 1 : 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    popcount_visited_kernel<<<parallel_seed_count, block_thread_count>>>(device_workspaces, workspace_stride_u64, device_closure_sizes);
    CUDA_CHECK(cudaDeviceSynchronize());
}

static void run_closure_until_fixpoint(std::uint64_t* device_workspaces, int parallel_seed_count, unsigned int* device_any_new_flags, int* device_closure_sizes,
                               unsigned char* device_seed_active_flags, int* device_live_visit_counts, int* device_over_size_limit_flags, int max_outer_layer_count,
                               bool print_layer_program_nameress, bool single_inner_round_only, bool skip_seeds_over_cluster_a) {
    std::vector<int> counts(static_cast<std::size_t>(parallel_seed_count), 0);
    std::vector<unsigned char> active(static_cast<std::size_t>(parallel_seed_count), 1);
    if (device_seed_active_flags) {
        CUDA_CHECK(cudaMemcpy(device_seed_active_flags, active.data(), static_cast<std::size_t>(parallel_seed_count) * sizeof(unsigned char),
                              cudaMemcpyHostToDevice));
    }
    popcount_visited_kernel<<<parallel_seed_count, block_thread_count>>>(device_workspaces, workspace_stride_u64, device_closure_sizes);
    CUDA_CHECK(cudaDeviceSynchronize());
    copy_closure_sizes_from_device(device_workspaces, parallel_seed_count, device_closure_sizes, &counts);
    if (device_live_visit_counts) {
        CUDA_CHECK(cudaMemcpy(device_live_visit_counts, counts.data(), static_cast<std::size_t>(parallel_seed_count) * sizeof(int),
                              cudaMemcpyHostToDevice));
    }
    if (device_over_size_limit_flags) {
        std::vector<int> cleared_over_limit_flags(static_cast<std::size_t>(parallel_seed_count), 0);
        CUDA_CHECK(cudaMemcpy(device_over_size_limit_flags, cleared_over_limit_flags.data(), static_cast<std::size_t>(parallel_seed_count) * sizeof(int),
                              cudaMemcpyHostToDevice));
    }
    if (print_layer_program_nameress && parallel_seed_count == 1) {
        std::printf("layer 0 |S|=%d\n", counts[0]);
        std::fflush(stdout);
    }

    for (int L = 1; L <= max_outer_layer_count; ++L) {
        std::vector<int> before = counts;
        run_single_outer_layer(device_workspaces, parallel_seed_count, device_any_new_flags, device_closure_sizes, device_seed_active_flags, device_live_visit_counts, device_over_size_limit_flags,
                            skip_seeds_over_cluster_a ? cluster_a_max_closure_size : 0, single_inner_round_only);
        copy_closure_sizes_from_device(device_workspaces, parallel_seed_count, device_closure_sizes, &counts);

        if (skip_seeds_over_cluster_a) {
            bool active_flags_need_upload = false;
            std::vector<int> over(static_cast<std::size_t>(parallel_seed_count), 0);
            if (device_over_size_limit_flags) {
                CUDA_CHECK(cudaMemcpy(over.data(), device_over_size_limit_flags, static_cast<std::size_t>(parallel_seed_count) * sizeof(int),
                                      cudaMemcpyDeviceToHost));
            }
            for (int b = 0; b < parallel_seed_count; ++b) {
                if (!active[b]) continue;
                if (counts[b] > cluster_a_max_closure_size || over[b]) {
                    active[b] = 0;
                    active_flags_need_upload = true;
                }
            }
            if (active_flags_need_upload && device_seed_active_flags) {
                CUDA_CHECK(cudaMemcpy(device_seed_active_flags, active.data(),
                                      static_cast<std::size_t>(parallel_seed_count) * sizeof(unsigned char),
                                      cudaMemcpyHostToDevice));
            }
        }

        if (print_layer_program_nameress) {
            if (parallel_seed_count == 1) {
                std::printf("layer %d |S|=%d (+%d)%s\n", L, counts[0], counts[0] - before[0],
                            single_inner_round_only ? " [one-inner]" : "");
            } else {
                std::printf("layer %d", L);
                for (int b = 0; b < parallel_seed_count; ++b)
                    std::printf("  b%d|S|=%d(+%d)", b, counts[b], counts[b] - before[b]);
                std::printf("%s\n", single_inner_round_only ? " [one-inner]" : "");
            }
            std::fflush(stdout);
        }

        bool all_seeds_unchanged = true;
        bool all_seeds_sheffer_full = true;
        int active_seed_count = 0;
        for (int b = 0; b < parallel_seed_count; ++b) {
            if (!active[b]) continue;
            ++active_seed_count;
            if (counts[b] != before[b]) all_seeds_unchanged = false;
            if (static_cast<MatrixMask>(counts[b]) < universe_size) all_seeds_sheffer_full = false;
        }
        if (active_seed_count == 0) break;
        if (all_seeds_unchanged || all_seeds_sheffer_full) break;
    }
}

static double current_time_seconds() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static int max_parallel_seeds_for_vram() {
    std::size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) return 4;
    const std::size_t need = workspace_bytes_per_seed + sizeof(int) + 4096u;
    int n = static_cast<int>(free_b / need);
    if (n < 1) n = 1;
    if (n > max_parallel_seed_count) n = max_parallel_seed_count;
    return n;
}

static void run_seed_chunk_closure(std::uint64_t* device_workspaces, MatrixMask* device_seed_masks, unsigned int* device_any_new_flags, int* device_closure_sizes, const MatrixMask* host_seed_masks,
                              unsigned char* device_seed_active_flags, int* device_live_visit_counts, int* device_over_size_limit_flags, int chunk_seed_count, int max_outer_layer_count, bool single_inner_round_only,
                              bool skip_seeds_over_cluster_a) {
    CUDA_CHECK(cudaMemcpy(device_seed_masks, host_seed_masks, static_cast<std::size_t>(chunk_seed_count) * sizeof(MatrixMask), cudaMemcpyHostToDevice));
    init_seed_workspaces_kernel<<<chunk_seed_count, block_thread_count>>>(device_workspaces, workspace_stride_u64, device_seed_masks);
    CUDA_CHECK(cudaDeviceSynchronize());
    run_closure_until_fixpoint(device_workspaces, chunk_seed_count, device_any_new_flags, device_closure_sizes, device_seed_active_flags, device_live_visit_counts, device_over_size_limit_flags, max_outer_layer_count, false,
                       single_inner_round_only, skip_seeds_over_cluster_a);
}

static double benchmark_sequential(std::uint64_t* device_workspaces, MatrixMask* device_seed_masks, unsigned int* device_any_new_flags, int* device_closure_sizes,
                               unsigned char* device_seed_active_flags, int* device_live_visit_counts, int* device_over_size_limit_flags,
                               const std::vector<MatrixMask>& seeds, int max_outer_layer_count,
                               bool single_inner_round_only) {
    const auto t0 = current_time_seconds();
    for (MatrixMask s : seeds) {
        run_seed_chunk_closure(device_workspaces, device_seed_masks, device_any_new_flags, device_closure_sizes, &s, device_seed_active_flags, device_live_visit_counts, device_over_size_limit_flags, 1, max_outer_layer_count,
                          single_inner_round_only, false);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    return current_time_seconds() - t0;
}

static double benchmark_parallel(std::uint64_t* device_workspaces, MatrixMask* device_seed_masks, unsigned int* device_any_new_flags, int* device_closure_sizes,
                             unsigned char* device_seed_active_flags, int* device_live_visit_counts, int* device_over_size_limit_flags,
                             const std::vector<MatrixMask>& seeds, int parallel, int max_outer_layer_count, bool single_inner_round_only) {
    const auto t0 = current_time_seconds();
    for (std::size_t off = 0; off < seeds.size();) {
        const int chunk = static_cast<int>(std::min<std::size_t>(seeds.size() - off, static_cast<std::size_t>(parallel)));
        run_seed_chunk_closure(device_workspaces, device_seed_masks, device_any_new_flags, device_closure_sizes, seeds.data() + off, device_seed_active_flags, device_live_visit_counts, device_over_size_limit_flags,
                          chunk, max_outer_layer_count, single_inner_round_only, false);
        off += static_cast<std::size_t>(chunk);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    return current_time_seconds() - t0;
}

struct OrbitSeedRecord {
    MatrixMask hex = 0;
    int line_no = 0;
};

static bool parse_orbit60_input_line(const char* line, OrbitSeedRecord* out) {

    if (!line || line[0] == '#' || line[0] == '\n') return false;
    int a = 0, b = 0, c = 0, orbit = 0, cnt = 0;
    unsigned int hex = 0;
    if (std::sscanf(line, "%d %d %d 0x%x %d %d", &a, &b, &c, &hex, &orbit, &cnt) < 4) return false;
    out->hex = hex;
    return true;
}

static bool load_pending_orbit60_seeds(const char* path, int limit, std::vector<OrbitSeedRecord>* seeds) {
    FILE* f = std::fopen(path, "r");
    if (!f) return false;
    char buf[512];
    int line_no = 0;
    while (std::fgets(buf, sizeof(buf), f)) {
        ++line_no;
        OrbitSeedRecord r{};
        r.line_no = line_no;
        if (!parse_orbit60_input_line(buf, &r)) continue;
        seeds->push_back(r);
        if (limit > 0 && static_cast<int>(seeds->size()) >= limit) break;
    }
    std::fclose(f);
    return !seeds->empty();
}

static void usage(const char* program_name) {
    std::fprintf(stderr,
                "Usage:\n"
                "  %s <seed_hex> [--layers N] [--one-inner] [--no-mu-lut]\n"
                "  %s --orbit60 <file> [--limit N] [--parallel P] [--bench] [--one-inner]\n"
                "      (без --layers: max outer layers=%d; строки #DONE пропускаются)\n"
                "\n"
                "  --parallel P: grid=P блоков, blockIdx.x * WS_STRIDE — свой seed на SM.\n"
                "  --bench: сравнить sequential (P=1) vs parallel на том же наборе seed.\n"
                "  --one-inner: один prepare+expand за внешний слой (быстрый прогон, != snap-BFS).\n"
                "  --layers N: cap внешних слоев; fixpoint раньше — all_seeds_unchanged (|S| не растет) / all_seeds_sheffer_full.\n"
                "  --skip-over-65544: в batch отключать seed после первого |S|>65544 (ускорение, не strict |S|∞).\n",
                program_name, program_name, orbit60_default_max_outer_layers);
}

static int run_single_seed_closure(MatrixMask seed, int max_outer_layer_count, bool use_mu_lookup_table, bool single_inner_round_only) {
    if (use_mu_lookup_table && load_mu_lookup_from_file("mu_lut_k5.bin")) (void)upload_mu_lookup_to_device();

    std::uint64_t* device_workspaces = nullptr;
    MatrixMask* device_seed_masks = nullptr;
    unsigned int* device_any_new_flags = nullptr;
    int* device_closure_sizes = nullptr;
    unsigned char* device_seed_active_flags = nullptr;
    int* device_live_visit_counts = nullptr;
    int* device_over_size_limit_flags = nullptr;
    CUDA_CHECK(cudaMalloc(&device_workspaces, workspace_bytes_per_seed));
    CUDA_CHECK(cudaMalloc(&device_seed_masks, sizeof(MatrixMask)));
    CUDA_CHECK(cudaMalloc(&device_any_new_flags, sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&device_closure_sizes, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&device_seed_active_flags, sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&device_live_visit_counts, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&device_over_size_limit_flags, sizeof(int)));

    CUDA_CHECK(cudaMemcpy(device_seed_masks, &seed, sizeof(MatrixMask), cudaMemcpyHostToDevice));
    init_seed_workspaces_kernel<<<1, block_thread_count>>>(device_workspaces, workspace_stride_u64, device_seed_masks);
    CUDA_CHECK(cudaDeviceSynchronize());
    run_closure_until_fixpoint(device_workspaces, 1, device_any_new_flags, device_closure_sizes, device_seed_active_flags, device_live_visit_counts, device_over_size_limit_flags, max_outer_layer_count, true,
                       single_inner_round_only, false);

    int final_closure_size = 0;
    CUDA_CHECK(cudaMemcpy(&final_closure_size, device_closure_sizes, sizeof(int), cudaMemcpyDeviceToHost));
    std::printf("seed=0x%x  |S|=%d  universe_size=%u  sheffer=%s\n", seed, final_closure_size, universe_size, final_closure_size == int(universe_size) ? "YES" : "NO");

    cudaFree(device_workspaces);
    cudaFree(device_seed_masks);
    cudaFree(device_any_new_flags);
    cudaFree(device_closure_sizes);
    cudaFree(device_seed_active_flags);
    cudaFree(device_live_visit_counts);
    cudaFree(device_over_size_limit_flags);
    free_mu_lookup_tables();
    return (final_closure_size >= int(universe_size)) ? 0 : 1;
}

static int run_orbit60_batch_closure(const char* path, int limit, int max_outer_layer_count, bool use_mu_lookup_table, bool single_inner_round_only,
                             int parallel, bool run_benchmark, bool skip_seeds_over_cluster_a) {
    std::vector<OrbitSeedRecord> orbit_seed_records;
    if (!load_pending_orbit60_seeds(path, limit, &orbit_seed_records)) {
        std::fprintf(stderr, "Failed to read pending seeds from %s (all #DONE?)\n", path);
        return 1;
    }
    std::printf("orbit60 pending seeds to run: %zu (lines #DONE skipped)  max_outer_layer_count=%d\n", orbit_seed_records.size(),
                max_outer_layer_count);
    std::fflush(stdout);
    std::vector<MatrixMask> seeds;
    seeds.reserve(orbit_seed_records.size());
    for (const auto& r : orbit_seed_records) seeds.push_back(r.hex);

    if (use_mu_lookup_table && load_mu_lookup_from_file("mu_lut_k5.bin")) (void)upload_mu_lookup_to_device();

    const int max_parallel_by_vram = max_parallel_seeds_for_vram();
    if (parallel < 1) parallel = 1;
    if (parallel > max_parallel_by_vram) {
        std::printf("[warn] --parallel %d > max_parallel_by_vram %d, clamping\n", parallel, max_parallel_by_vram);
        parallel = max_parallel_by_vram;
    }
    const int allocated_grid_size = run_benchmark ? max_parallel_by_vram : parallel;

    std::uint64_t* device_workspaces = nullptr;
    MatrixMask* device_seed_masks = nullptr;
    unsigned int* device_any_new_flags = nullptr;
    int* device_closure_sizes = nullptr;
    unsigned char* device_seed_active_flags = nullptr;
    int* device_live_visit_counts = nullptr;
    int* device_over_size_limit_flags = nullptr;
    CUDA_CHECK(cudaMalloc(&device_workspaces, workspace_bytes_per_seed * static_cast<std::size_t>(allocated_grid_size)));
    CUDA_CHECK(cudaMalloc(&device_seed_masks, static_cast<std::size_t>(allocated_grid_size) * sizeof(MatrixMask)));
    CUDA_CHECK(cudaMalloc(&device_any_new_flags, static_cast<std::size_t>(allocated_grid_size) * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&device_closure_sizes, static_cast<std::size_t>(allocated_grid_size) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&device_seed_active_flags, static_cast<std::size_t>(allocated_grid_size) * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&device_live_visit_counts, static_cast<std::size_t>(allocated_grid_size) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&device_over_size_limit_flags, static_cast<std::size_t>(allocated_grid_size) * sizeof(int)));

    if (run_benchmark) {
        std::printf("[bench] seeds=%zu  layers=%d  one_inner=%d  ws_per_seed=%.1f MiB  vram_max_parallel=%d\n",
                    seeds.size(), max_outer_layer_count, single_inner_round_only ? 1 : 0, workspace_bytes_per_seed / (1024.0 * 1024.0), max_parallel_by_vram);
        std::fflush(stdout);

        run_seed_chunk_closure(device_workspaces, device_seed_masks, device_any_new_flags, device_closure_sizes, seeds.data(), device_seed_active_flags, device_live_visit_counts, device_over_size_limit_flags, 1, 1,
                          single_inner_round_only, false);

        const double sequential_seconds = benchmark_sequential(device_workspaces, device_seed_masks, device_any_new_flags, device_closure_sizes, device_seed_active_flags, device_live_visit_counts, device_over_size_limit_flags,
                                              seeds, max_outer_layer_count, single_inner_round_only);
        std::printf("[bench] sequential (grid=1): %.3f s  (%.4f s/seed)\n", sequential_seconds, sequential_seconds / seeds.size());

        std::vector<int> parallel_counts_to_try = {2, 4, 8};
        if (parallel > 1) parallel_counts_to_try.push_back(parallel);
        std::sort(parallel_counts_to_try.begin(), parallel_counts_to_try.end());
        parallel_counts_to_try.erase(std::unique(parallel_counts_to_try.begin(), parallel_counts_to_try.end()), parallel_counts_to_try.end());

        for (int p : parallel_counts_to_try) {
            if (p < 2 || p > max_parallel_by_vram) continue;
            if (p > static_cast<int>(seeds.size())) continue;
            const double parallel_seconds = benchmark_parallel(device_workspaces, device_seed_masks, device_any_new_flags, device_closure_sizes, device_seed_active_flags, device_live_visit_counts, device_over_size_limit_flags,
                                                seeds, p, max_outer_layer_count, single_inner_round_only);
            std::printf("[bench] parallel grid=%d: %.3f s  (%.4f s/seed)  speedup=%.2fx\n", p, parallel_seconds,
                        parallel_seconds / seeds.size(), sequential_seconds / parallel_seconds);
        }
    } else {
        const auto t0 = current_time_seconds();
        if (parallel <= 1) {
            for (std::size_t i = 0; i < seeds.size(); ++i) {
                run_seed_chunk_closure(device_workspaces, device_seed_masks, device_any_new_flags, device_closure_sizes, &seeds[i], device_seed_active_flags, device_live_visit_counts, device_over_size_limit_flags, 1,
                                  max_outer_layer_count, single_inner_round_only, skip_seeds_over_cluster_a);
                int n = 0;
                CUDA_CHECK(cudaMemcpy(&n, device_closure_sizes, sizeof(int), cudaMemcpyDeviceToHost));
                if (skip_seeds_over_cluster_a && n > cluster_a_max_closure_size) {
                    std::printf("[%zu] seed=0x%x  |S|>%d  [skipped]\n", i + 1, seeds[i], cluster_a_max_closure_size);
                } else {
                    std::printf("[%zu] seed=0x%x  |S|=%d\n", i + 1, seeds[i], n);
                }
            }
        } else {
            for (std::size_t off = 0; off < seeds.size();) {
                const int chunk =
                    static_cast<int>(std::min<std::size_t>(seeds.size() - off, static_cast<std::size_t>(parallel)));
                run_seed_chunk_closure(device_workspaces, device_seed_masks, device_any_new_flags, device_closure_sizes, seeds.data() + off, device_seed_active_flags, device_live_visit_counts,
                                  device_over_size_limit_flags, chunk, max_outer_layer_count, single_inner_round_only, skip_seeds_over_cluster_a);
                std::vector<int> ns(static_cast<std::size_t>(chunk));
                CUDA_CHECK(cudaMemcpy(ns.data(), device_closure_sizes, static_cast<std::size_t>(chunk) * sizeof(int),
                                      cudaMemcpyDeviceToHost));
                for (int b = 0; b < chunk; ++b) {
                    if (skip_seeds_over_cluster_a && ns[b] > cluster_a_max_closure_size) {
                        std::printf("[%zu] seed=0x%x  |S|>%d  [skipped]  (block %d)\n", off + b + 1, seeds[off + b],
                                    cluster_a_max_closure_size, b);
                    } else {
                        std::printf("[%zu] seed=0x%x  |S|=%d  (block %d)\n", off + b + 1, seeds[off + b], ns[b], b);
                    }
                }
                off += static_cast<std::size_t>(chunk);
            }
        }
        const double elapsed = current_time_seconds() - t0;
        std::printf("done %zu seeds  parallel=%d  elapsed=%.3f s  (%.4f s/seed)\n", seeds.size(), parallel, elapsed,
                    elapsed / seeds.size());
    }

    cudaFree(device_workspaces);
    cudaFree(device_seed_masks);
    cudaFree(device_any_new_flags);
    cudaFree(device_closure_sizes);
    cudaFree(device_seed_active_flags);
    cudaFree(device_live_visit_counts);
    cudaFree(device_over_size_limit_flags);
    free_mu_lookup_tables();
    return 0;
}

}

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc < 2) {
        usage(argv[0]);
        return 1;
    }

    bool use_orbit60_mode = false;
    bool outer_layers_explicit = false;
    bool single_inner_round_only = false;
    bool use_mu_lookup_table = true;
    bool run_benchmark = false;
    bool skip_seeds_over_cluster_a = false;
    int max_outer_layer_count = default_max_outer_layers;
    int limit = 0;
    int parallel_seed_count_cli = 1;
    const char* orbit60_file_path = "k5_ge36000_orbit60.txt";
    MatrixMask seed = 0;
    bool seed_argument_provided = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--orbit60") == 0 && i + 1 < argc) {
            use_orbit60_mode = true;
            orbit60_file_path = argv[++i];
        } else if (std::strcmp(argv[i], "--limit") == 0 && i + 1 < argc) {
            limit = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--layers") == 0 && i + 1 < argc) {
            max_outer_layer_count = std::atoi(argv[++i]);
            outer_layers_explicit = true;
        } else if (std::strcmp(argv[i], "--parallel") == 0 && i + 1 < argc) {
            parallel_seed_count_cli = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--bench") == 0) {
            run_benchmark = true;
        } else if (std::strcmp(argv[i], "--one-inner") == 0) {
            single_inner_round_only = true;
        } else if (std::strcmp(argv[i], "--no-mu-lut") == 0) {
            use_mu_lookup_table = false;
        } else if (std::strcmp(argv[i], "--skip-over-65544") == 0) {
            skip_seeds_over_cluster_a = true;
        } else if (argv[i][0] != '-') {
            seed = static_cast<MatrixMask>(std::strtoul(argv[i], nullptr, 0));
            seed_argument_provided = true;
        } else {
            std::fprintf(stderr, "Unknown option: %s\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }

    if (use_orbit60_mode) {
        if (!outer_layers_explicit) max_outer_layer_count = orbit60_default_max_outer_layers;
        return run_orbit60_batch_closure(orbit60_file_path, limit, max_outer_layer_count, use_mu_lookup_table, single_inner_round_only, parallel_seed_count_cli, run_benchmark,
                                 skip_seeds_over_cluster_a);
    }
    if (!seed_argument_provided) {
        usage(argv[0]);
        return 1;
    }
    return run_single_seed_closure(seed, max_outer_layer_count, use_mu_lookup_table, single_inner_round_only);
}
