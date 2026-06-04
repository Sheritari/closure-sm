
#include <cstdint>
#include <cstdio>
#include <cstring>

static constexpr int matrix_order = 5;
static constexpr std::uint32_t universe_size = 1u << 25;
static constexpr std::uint32_t full_matrix_mask = universe_size - 1u;

static std::uint32_t transpose_matrix_mask(std::uint32_t m) {
    m &= full_matrix_mask;
    std::uint32_t t = 0;
    for (int i = 0; i < matrix_order; ++i)
        for (int j = 0; j < matrix_order; ++j)
            if ((m >> (matrix_order * j + i)) & 1u) t |= 1u << (matrix_order * i + j);
    return t;
}

int main(int argc, char** argv) {
    const char* output_file_path = (argc >= 2) ? argv[1] : "mu_lut_k5.bin";
    FILE* f = std::fopen(output_file_path, "wb");
    if (!f) {
        std::perror("fopen");
        return 1;
    }

    std::uint32_t write_chunk[1 << 16];
    std::size_t entries_written = 0;
    for (std::uint32_t m = 0; m < universe_size; ++m) {
        write_chunk[m & 0xFFFFu] = transpose_matrix_mask(m);
        if ((m & 0xFFFFu) == 0xFFFFu || m == universe_size - 1) {
            const std::size_t chunk = (m == universe_size - 1) ? ((m & 0xFFFFu) + 1) : (sizeof write_chunk / sizeof write_chunk[0]);
            if (std::fwrite(write_chunk, sizeof(std::uint32_t), chunk, f) != chunk) {
                std::perror("fwrite");
                std::fclose(f);
                return 1;
            }
            entries_written += chunk;
            if ((m & 0x3FFFFu) == 0) {
                std::fprintf(stderr, "\r%u / %u", static_cast<unsigned>(m + 1), static_cast<unsigned>(universe_size));
                std::fflush(stderr);
            }
        }
    }
    std::fprintf(stderr, "\rdone: %zu entries -> %s\n", entries_written, output_file_path);
    std::fclose(f);
    return 0;
}
