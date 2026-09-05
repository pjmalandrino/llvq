// Drives the V3 decoder of `llvq_f1rank_v3.cuh` on the CPU, one word at a
// time — and, in its second mode, the shim's `__byte_perm` alone.
//
// Reads a fixture on stdin, writes results on stdout, both little-endian
// binary. The Rust side builds the fixture and checks the answers against
// `llvq_bench::f1::rank::decode_word` and against an f64 dot, so this file
// holds no expectations — it is a harness, not a reference.
//
// Default mode (no argument):
//   in : u32 n
//        u32[4096] rows, u8[128] prefixes, u16[1024] branches, u8[128] suffixes
//        u64[n] words — bits 0..47 the label; bits 48..63 whatever the Rust
//                       side put there, since the decoder is told to ignore
//                       the upper half of `hi16`
//        u32 nx
//        f32[nx*24] x
//   out: f32[n*24]  the 24 values of every word, `f1r_decode_v3_f`
//        f32[n*nx]  `f1r_dot_v3(word i, x k)` at index i·nx + k
//
// Mode `prmt` (first argument):
//   in : u32 n, then n × (u32 a, u32 b, u32 s)
//   out: u32[n] `__byte_perm(a, b, s)` of the shim — the function
//        `llvq_f1rank_v3.cuh` executes here in place of the instruction, so
//        the Rust side checks it against 16 hand-computed cases of PTX
//        `prmt.b32` before it reads a single decode.
//
// Both `f1r_dot_v3` and `f1r_decode_v3_f` are scalar register arithmetic —
// no warp primitive, no shared memory — so they are *executed* here, exactly
// as host_f1rank.cpp executes `f1r_decode`. The kernel of f1rank_v3.cu is not
// included: the tile loop is the card's business and `bin/cuhcheck` parses it.
//
// Compiled twice by the Rust test: as is, and with
// `-DLLVQ_F1R_V3_NO_SIGN_PRMT`, the mask without the msb mode of `prmt`.

#include "host_shim.h"
// In the order the bench concatenates for NVRTC: llvq_slot.cuh gives `u32`
// and LLVQ_DIM, the base decoder header gives `F1rTables` and
// `F1R_N0_MIXED`, the variant comes after both.
#include "../kernels/llvq_slot.cuh"
#include "../kernels/llvq_f1rank.cuh"
#include "../kernels/llvq_f1rank_v3.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
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

static int run_prmt(std::FILE* in) {
    const unsigned n = read_n<unsigned>(in, 1, "n")[0];
    const auto abs_ = read_n<unsigned>(in, static_cast<std::size_t>(n) * 3, "cas");
    std::vector<unsigned> out(n);
    for (unsigned i = 0; i < n; ++i) {
        out[i] = __byte_perm(abs_[3 * i], abs_[3 * i + 1], abs_[3 * i + 2]);
    }
    std::fwrite(out.data(), sizeof(unsigned), out.size(), stdout);
    return 0;
}

static int run_decode(std::FILE* in) {
    const unsigned n = read_n<unsigned>(in, 1, "n")[0];
    const auto rows = read_n<unsigned>(in, 4096, "rows");
    const auto prefixes = read_n<unsigned char>(in, 128, "prefixes");
    const auto branches = read_n<unsigned short>(in, 1024, "branches");
    const auto suffixes = read_n<unsigned char>(in, 128, "suffixes");
    const auto words = read_n<unsigned long long>(in, n, "words");
    const unsigned nx = read_n<unsigned>(in, 1, "nx")[0];
    const auto x = read_n<float>(in, static_cast<std::size_t>(nx) * LLVQ_DIM, "x");

    const F1rTables t{rows.data(), prefixes.data(), branches.data(), suffixes.data()};
    std::vector<float> y(static_cast<std::size_t>(n) * LLVQ_DIM, 0.0f);
    std::vector<float> dot(static_cast<std::size_t>(n) * nx, 0.0f);
    for (unsigned i = 0; i < n; ++i) {
        const unsigned long long w = words[i];
        const u32 lo = static_cast<u32>(w), hi16 = static_cast<u32>(w >> 32);
        f1r_decode_v3_f(lo, hi16, t, y.data() + static_cast<std::size_t>(i) * LLVQ_DIM);
        for (unsigned k = 0; k < nx; ++k) {
            dot[static_cast<std::size_t>(i) * nx + k] =
                f1r_dot_v3(lo, hi16, t, x.data() + static_cast<std::size_t>(k) * LLVQ_DIM);
        }
    }
    std::fwrite(y.data(), sizeof(float), y.size(), stdout);
    std::fwrite(dot.data(), sizeof(float), dot.size(), stdout);
    return 0;
}

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "prmt") == 0) return run_prmt(stdin);
    return run_decode(stdin);
}
