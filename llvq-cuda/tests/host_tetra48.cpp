// Drives the served Tetra decode of `llvq_tetra48.cuh` on the CPU, one word
// at a time.
//
// Reads a fixture on stdin, writes results on stdout, both little-endian
// binary. The Rust side builds the fixture and checks the answers against
// `llvq_search::tetra::Tetra::decode` and `llvq_quant::reconstruct_shape_gain`
// — the two functions the encoder and the artifact reader actually use — so
// this file holds no expectations. It is a harness, not a reference.
//
// What it buys: `tetra48_dot` is the one function in the served path where a
// wrong answer is silent. A swapped permutation, an unsigned `__dp4a`, a
// missing origin entry — none of them crash, none of them are visible in a
// throughput number, and all three would reach a $8 job as a quietly wrong
// model. Two seconds of `clang++` here against fifty minutes of image rebuild
// and a rented card.
//
//   in : u32 n
//        u32[4096] rows, u8[128] prefixes, u16[1024] branches, u8[128] suffixes
//        f32[2] gscale       the two gain centroids of the matrix
//        f32[32] invnorm     1/sqrt(16 m), entry 0 = 0
//        u64[n] words        bits 0..47 the word, gain bit at 47; the upper
//                            half is whatever the Rust side put there, since
//                            the decoder reads only the low 16 bits of `hi16`
//                            past bit 47
//        u32 nx
//        f32[nx*24] x        activations, NATURAL coordinate order
//   out: f32[n*24]  `tetra48_decode_f`, natural order, both scales applied
//        f32[n*nx]  `tetra48_dot(word i, x k)` at index i*nx + k
//        u32[n]     `tetra48_n2` — the raw shell sum, so the Rust side can
//                   assert `n2 % 16 == 0` and `n2/16 <= 27` on every word
//                   rather than trusting the header's comment
//
// Every function executed here is scalar register arithmetic — no warp
// primitive, no shared memory, no barrier — so this is the real code, not a
// stand-in. The kernel that will wrap it is the card's business.

#include "host_shim.h"
// In the order the bench concatenates for NVRTC: llvq_slot.cuh gives `u32`
// and LLVQ_DIM, the base decoder header gives `F1rTables` and
// `F1R_N0_MIXED`, the v3 variant next, the served header last.
#include "../kernels/llvq_slot.cuh"
#include "../kernels/llvq_f1rank.cuh"
#include "../kernels/llvq_f1rank_v3.cuh"
#include "../kernels/llvq_tetra48.cuh"

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
    const auto prefixes = read_n<unsigned char>(in, 128, "prefixes");
    const auto branches = read_n<unsigned short>(in, 1024, "branches");
    const auto suffixes = read_n<unsigned char>(in, 128, "suffixes");
    const auto gscale = read_n<float>(in, 2, "gscale");
    const auto invnorm = read_n<float>(in, TETRA48_SHELLS, "invnorm");
    const auto words = read_n<unsigned long long>(in, n, "words");
    const unsigned nx = read_n<unsigned>(in, 1, "nx")[0];
    const auto x = read_n<float>(in, static_cast<std::size_t>(nx) * LLVQ_DIM, "x");

    const F1rTables t{rows.data(), prefixes.data(), branches.data(), suffixes.data()};
    std::vector<float> y(static_cast<std::size_t>(n) * LLVQ_DIM, 0.0f);
    std::vector<float> dot(static_cast<std::size_t>(n) * nx, 0.0f);
    std::vector<unsigned> n2(n, 0u);

    for (unsigned i = 0; i < n; ++i) {
        const unsigned long long w = words[i];
        const u32 lo = static_cast<u32>(w), hi16 = static_cast<u32>(w >> 32);
        tetra48_decode_f(lo, hi16, t, gscale.data(), invnorm.data(),
                         y.data() + static_cast<std::size_t>(i) * LLVQ_DIM);
        for (unsigned k = 0; k < nx; ++k) {
            dot[static_cast<std::size_t>(i) * nx + k] =
                tetra48_dot(lo, hi16, t, x.data() + static_cast<std::size_t>(k) * LLVQ_DIM,
                            gscale.data(), invnorm.data());
        }
        // The quads again, only to publish `n2`: the header keeps it private
        // to the dot, and a bound the test cannot read is a bound nobody
        // checks.
        u32 q[6];
        f1r_v3_quads(lo, hi16, t, q);
        n2[i] = tetra48_n2(q);
    }

    std::fwrite(y.data(), sizeof(float), y.size(), stdout);
    std::fwrite(dot.data(), sizeof(float), dot.size(), stdout);
    std::fwrite(n2.data(), sizeof(unsigned), n2.size(), stdout);
    return 0;
}
