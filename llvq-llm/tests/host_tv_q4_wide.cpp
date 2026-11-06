// Executes tv_q4_hh (the served text) and tv_q4_hs (tv_q4_wide.cu) on the CPU,
// lane by lane, and returns each lane's partial sum before the butterfly.
//
// `host_tv_q4_h.cpp` cannot run the int4 kernels: one warp a row needs a
// 32-lane shuffle, so it executes `q4_deq` and a hand-written mirror of the
// lane order. This file executes the KERNELS themselves, entry points
// included, with one substitution: `warp_sum` records the lane's argument
// instead of reducing it. Both kernels call the same `warp_sum` on the card
// (matvec.cu's butterfly), so if every lane hands it the same float in both
// kernels, the two outputs are the same bits. That is the claim this file
// tests, at any width, with no shared-memory ceiling: the host has none, so
// tv_q4_hh runs here at 25,600 too, which no 99 KiB card can do.
//
// The staging: each kernel stages `x` with a block-strided loop and then
// computes. Run thread by thread in one pass, thread 0 would compute before
// thread 1 staged its share. So every block is run TWICE: the first pass
// stages (the staging is idempotent: every thread writes the same values on
// the second pass), the second pass computes on a complete stage and is the
// one recorded.
//
//   in : u32 d_out, u32 d_in, u32 gpr, u32 x_off,
//        u32[d_out*d_in/8] words, u16[d_out*gpr] scales, u16[d_out*gpr] biases,
//        u16[x_off + d_in] x (f16 bits; the activation starts at x_off)
//   out: f32[d_out*32] hh lane partials, f32[d_out*32] hs lane partials,
//        u16[d_out] y: hs's partials through warp_sum's butterfly, then f2h

#include "../../llvq-cuda/tests/host_shim.h"
#include "../../llvq-cuda/kernels/llvq_slot.cuh"  // u32, LLVQ_DIM

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define TILE_COLS (128u * LLVQ_DIM)  // skip matvec.cu in tv_q4_h.cu

// -ffp-contract=off makes these one rounding each, as the device intrinsics.
static inline float __fmul_rn(float a, float b) { return a * b; }
static inline float __fadd_rn(float a, float b) { return a + b; }

// Exact conversions through clang's IEEE binary16, as host_tv_q4_h.cpp.
static inline float h2f(unsigned short h) {
    _Float16 v;
    std::memcpy(&v, &h, 2);
    return (float)v;
}
static inline unsigned short f2h(float f) {
    _Float16 v = (_Float16)f;
    unsigned short r;
    std::memcpy(&r, &v, 2);
    return r;
}

// The substitution: record the lane's partial sum.
static float lane_acc[32];
static inline float warp_sum(float v) {
    lane_acc[threadIdx.x & 31u] = v;
    return v;
}

#include "../kernels/tv_q4_h.cu"
#include "../kernels/tv_q4_wide.cu"

// The `extern __shared__` stand-ins: 32,768 f32 and 65,536 halves, past the
// widest fixture (31,744).
float xs[1 << 15];
unsigned short xs16[1 << 16];
Dim3 blockIdx{0, 0, 0};
Dim3 threadIdx{0, 0, 0};
Dim3 blockDim{256, 1, 1};

template <typename T>
static std::vector<T> read_n(std::FILE* f, std::size_t n, const char* what) {
    std::vector<T> v(n);
    if (n && std::fread(v.data(), sizeof(T), n, f) != n) {
        std::fprintf(stderr, "fixture truncated: %s\n", what);
        std::exit(2);
    }
    return v;
}

int main() {
    auto hdr = read_n<unsigned>(stdin, 4, "header");
    unsigned d_out = hdr[0], d_in = hdr[1], gpr = hdr[2], x_off = hdr[3];
    if (d_in > (1u << 15) || d_in % 8u != 0u || d_out % 8u != 0u) {
        std::fprintf(stderr, "fixture out of range: d_out=%u d_in=%u\n", d_out, d_in);
        return 2;
    }
    auto words = read_n<unsigned>(stdin, (std::size_t)d_out * (d_in >> 3), "words");
    auto scales = read_n<unsigned short>(stdin, (std::size_t)d_out * gpr, "scales");
    auto biases = read_n<unsigned short>(stdin, (std::size_t)d_out * gpr, "biases");
    auto x = read_n<unsigned short>(stdin, (std::size_t)x_off + d_in, "x");

    const unsigned THREADS = 256u;
    std::vector<float> hh((std::size_t)d_out * 32), hs((std::size_t)d_out * 32);
    std::vector<unsigned short> y_scratch(d_out);
    blockDim.x = THREADS;

    for (unsigned b = 0; b < d_out / (THREADS / 32u); ++b) {
        blockIdx.x = b;
        for (int pass = 0; pass < 2; ++pass)
            for (unsigned t = 0; t < THREADS; ++t) {
                threadIdx.x = t;
                // tv_q4_hh reads from the base pointer: hand it the activation.
                tv_q4_hh(words.data(), scales.data(), biases.data(), x.data() + x_off,
                         y_scratch.data(), d_in, gpr);
                if (pass == 1) hh[(std::size_t)(b * THREADS + t)] = lane_acc[t & 31u];
            }
        for (int pass = 0; pass < 2; ++pass)
            for (unsigned t = 0; t < THREADS; ++t) {
                threadIdx.x = t;
                tv_q4_hs(words.data(), scales.data(), biases.data(), x.data(),
                         y_scratch.data(), d_in, gpr, x_off);
                if (pass == 1) hs[(std::size_t)(b * THREADS + t)] = lane_acc[t & 31u];
            }
    }

    // warp_sum's butterfly, in its exact order, on hs's partials.
    std::vector<unsigned short> y(d_out);
    for (unsigned r = 0; r < d_out; ++r) {
        float lanes[32];
        std::memcpy(lanes, &hs[(std::size_t)r * 32], sizeof lanes);
        for (int k = 16; k > 0; k >>= 1) {
            float nv[32];
            for (int i = 0; i < 32; ++i) nv[i] = lanes[i] + lanes[i ^ k];
            std::memcpy(lanes, nv, sizeof nv);
        }
        y[r] = f2h(lanes[0]);
    }

    std::fwrite(hh.data(), 4, hh.size(), stdout);
    std::fwrite(hs.data(), 4, hs.size(), stdout);
    std::fwrite(y.data(), 2, y.size(), stdout);
    return 0;
}
