// Drives the V2 F1 decoder of `llvq_f1rank_v2.cuh` on the CPU, one word at a
// time: the 24 values as floats, and the per-block dot against a set of
// activation vectors.
//
// Reads a fixture on stdin, writes results on stdout, both little-endian
// binary. The Rust side builds the fixture and checks the answers against
// `llvq_bench::f1::rank::decode_word` and an f64 dot, so this file holds no
// expectations — it is a harness, not a reference.
//
//   in : u32 n
//        u32[4096] rows                — the 16 KiB rank table, nothing else:
//                                        V2 reads no prefix, branch or suffix
//                                        table, and the three pointers are
//                                        passed as NULL so a read would crash
//        u64[n] words                  — bits 0..47 the label; bits 48..63
//                                        whatever the Rust side put there
//        u32 nx
//        f32[nx*24] x                  — activation blocks
//   out: u32[12] columns               — the twelve immediates this text was
//                                        compiled with, for the Rust side to
//                                        compare to the derived ones
//        f32[n*24] y                   — `f1r_decode_v2_f`, trio order
//        f32[n*nx] dots                — `f1r_dot_v2(word i, x k)` at [i*nx + k]
//
// Both entry points are scalar register arithmetic — no warp primitive, no
// shared memory — so they are *executed* here, exactly as host_f1rank.cpp
// executes `f1r_decode`. The kernel of f1rank_v2.cu is not included: the tile
// loop is the card's business and `bin/cuhcheck` parses it.

#include "host_shim.h"
// In the order the bench concatenates for NVRTC: llvq_slot.cuh gives `u32`
// and LLVQ_DIM, llvq_f1rank.cuh gives `F1rTables`, `F1R_N0_MIXED` and
// `f1r_val`, the V2 header comes after both.
#include "../kernels/llvq_slot.cuh"
#include "../kernels/llvq_f1rank.cuh"
#include "../kernels/llvq_f1rank_v2.cuh"

#include <cstdio>
#include <cstdlib>
#include <vector>

// Nothing here launches, but the shim declares these `extern` and a header
// that so much as names a block index needs them defined.
Dim3 blockIdx{0, 0, 0};
Dim3 threadIdx{0, 0, 0};
Dim3 blockDim{1, 1, 1};

template <typename T>
static std::vector<T> read_n(std::FILE* f, std::size_t n, const char* what) {
    std::vector<T> v(n);
    if (n && std::fread(v.data(), sizeof(T), n, f) != n) {
        std::fprintf(stderr, "fixture tronquee: %s\n", what);
        std::exit(2);
    }
    return v;
}

int main() {
    std::FILE* in = stdin;
    const unsigned n = read_n<unsigned>(in, 1, "n")[0];
    const auto rows = read_n<unsigned>(in, 4096, "rows");
    const auto words = read_n<unsigned long long>(in, n, "words");
    const unsigned nx = read_n<unsigned>(in, 1, "nx")[0];
    const auto x = read_n<float>(in, static_cast<std::size_t>(nx) * LLVQ_DIM, "x");

    // The rank rows only. Null for the three small tables: V2 must not touch them.
    const F1rTables t{rows.data(), nullptr, nullptr, nullptr};

    const u32 cols[12] = {F1R_V2_COL_S8_0, F1R_V2_COL_S8_1, F1R_V2_COL_S8_2, F1R_V2_COL_S8_3,
                          F1R_V2_COL_S8_4, F1R_V2_COL_S8_5, F1R_V2_COL_B1,   F1R_V2_COL_B2_0,
                          F1R_V2_COL_B2_1, F1R_V2_COL_B2_2, F1R_V2_COL_B2_3, F1R_V2_COL_B3};
    std::fwrite(cols, sizeof(u32), 12, stdout);

    std::vector<float> y(static_cast<std::size_t>(n) * LLVQ_DIM, 0.0f);
    for (unsigned i = 0; i < n; ++i) {
        const unsigned long long w = words[i];
        f1r_decode_v2_f(static_cast<u32>(w), static_cast<u32>(w >> 32), t,
                        y.data() + static_cast<std::size_t>(i) * LLVQ_DIM);
    }
    std::fwrite(y.data(), sizeof(float), y.size(), stdout);

    std::vector<float> dots(static_cast<std::size_t>(n) * nx, 0.0f);
    for (unsigned i = 0; i < n; ++i) {
        const unsigned long long w = words[i];
        for (unsigned k = 0; k < nx; ++k) {
            dots[static_cast<std::size_t>(i) * nx + k] =
                f1r_dot_v2(static_cast<u32>(w), static_cast<u32>(w >> 32), t,
                           x.data() + static_cast<std::size_t>(k) * LLVQ_DIM);
        }
    }
    std::fwrite(dots.data(), sizeof(float), dots.size(), stdout);
    return 0;
}
