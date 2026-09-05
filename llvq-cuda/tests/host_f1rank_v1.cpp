// Drives the V1 decoder of `llvq_f1rank_v1.cuh` on the CPU: the 24 floats of
// a word, the per-block product, the chained product, and one section alone.
//
// Reads a fixture on stdin, writes results on stdout, both little-endian
// binary. The Rust side builds the fixture and checks the answers against
// `llvq_bench::f1::rank::{decode_word, val}` and its own f64 / f32 sums, so
// this file holds no expectations — it is a harness, not a reference.
//
//   in : u32 n, u32 nx, u32 nsec
//        u32[4096] rows, u8[128] prefixes, u16[1024] branches, u8[128] suffixes
//        u64[n]      words — bits 0..47 the label, bits 48..63 whatever the
//                    Rust side put there (`f1r_v1_lanes` masks every field)
//        f32[nx*24]  x, one activation block per column of the dot checks
//        u32[nsec*3] (p, c, row) triples for the section check
//   out: f32[n*24]   f1r_decode_v1_f, trio order
//        f32[nx*n]   f1r_dot_v1(word i, x m) at [m*n + i]         from zero
//        f32[nx*n]   f1r_dot_v1_acc chained over i = 0..n for each m, the
//                    running value after word i at [m*n + i]       from zero
//        f32[nsec*8] f1r_section_v1_f
//
// Everything driven here is scalar register arithmetic — no warp primitive,
// no shared memory — so it is *executed*, exactly as host_f1rank.cpp executes
// `f1r_decode`. `tv_f1r_v1` of f1rank_v1.cu is not included: the tile loop is
// the card's business and `bin/cuhcheck` parses it.

#include "host_shim.h"
// In the order the bench concatenates for NVRTC: llvq_slot.cuh gives `u32`
// and LLVQ_DIM, the original decoder header comes next (`F1rTables`,
// `F1R_N0_MIXED`), the V1 header after it.
#include "../kernels/llvq_slot.cuh"
#include "../kernels/llvq_f1rank.cuh"
#include "../kernels/llvq_f1rank_v1.cuh"

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
    const auto hdr = read_n<unsigned>(in, 3, "en-tete");
    const unsigned n = hdr[0], nx = hdr[1], nsec = hdr[2];
    const auto rows = read_n<unsigned>(in, 4096, "rows");
    const auto prefixes = read_n<unsigned char>(in, 128, "prefixes");
    const auto branches = read_n<unsigned short>(in, 1024, "branches");
    const auto suffixes = read_n<unsigned char>(in, 128, "suffixes");
    const auto words = read_n<unsigned long long>(in, n, "words");
    const auto x = read_n<float>(in, static_cast<std::size_t>(nx) * LLVQ_DIM, "x");
    const auto sec = read_n<unsigned>(in, static_cast<std::size_t>(nsec) * 3, "sections");

    const F1rTables t{rows.data(), prefixes.data(), branches.data(), suffixes.data()};

    std::vector<float> y(static_cast<std::size_t>(n) * LLVQ_DIM, 0.0f);
    for (unsigned i = 0; i < n; ++i) {
        const unsigned long long w = words[i];
        f1r_decode_v1_f(static_cast<u32>(w), static_cast<u32>(w >> 32), t,
                        y.data() + static_cast<std::size_t>(i) * LLVQ_DIM);
    }

    std::vector<float> dot(static_cast<std::size_t>(nx) * n, 0.0f);
    std::vector<float> chain(static_cast<std::size_t>(nx) * n, 0.0f);
    for (unsigned m = 0; m < nx; ++m) {
        const float* xb = x.data() + static_cast<std::size_t>(m) * LLVQ_DIM;
        float acc = 0.0f;
        for (unsigned i = 0; i < n; ++i) {
            const unsigned long long w = words[i];
            const u32 lo = static_cast<u32>(w), hi16 = static_cast<u32>(w >> 32);
            dot[static_cast<std::size_t>(m) * n + i] = f1r_dot_v1(lo, hi16, t, xb);
            acc = f1r_dot_v1_acc(lo, hi16, t, xb, acc);
            chain[static_cast<std::size_t>(m) * n + i] = acc;
        }
    }

    std::vector<float> ys(static_cast<std::size_t>(nsec) * 8, 0.0f);
    for (unsigned i = 0; i < nsec; ++i) {
        f1r_section_v1_f(sec[3 * i], sec[3 * i + 1], sec[3 * i + 2],
                         ys.data() + static_cast<std::size_t>(i) * 8);
    }

    std::fwrite(y.data(), sizeof(float), y.size(), stdout);
    std::fwrite(dot.data(), sizeof(float), dot.size(), stdout);
    std::fwrite(chain.data(), sizeof(float), chain.size(), stdout);
    std::fwrite(ys.data(), sizeof(float), ys.size(), stdout);
    return 0;
}
