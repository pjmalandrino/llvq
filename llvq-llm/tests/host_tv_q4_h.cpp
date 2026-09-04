// Drives tv_q4_h on the CPU.
//
// Reads a fixture on stdin, writes results on stdout, both little-endian
// binary. The Rust side builds the fixture and checks the answers against
// llvq-artifact's own dequantizer, so this file holds no expectations — it is
// a harness, not a reference (same pattern as host_embq8.cpp).
//
//   in : u32 d_out, u32 d_in, u32 gpr
//        u32[d_out*d_in/8] packed words, u16[d_out*gpr] scales,
//        u16[d_out*gpr] biases, f32[d_in] x
//   out: f32[d_out*d_in] dequant of every weight — q4_deq, executed,
//        f32[d_out]      per-row dot against x — the kernel's lane order and
//                        its warp butterfly, reproduced exactly.
//
// `tv_q4_h` needs a real warp (a 32-lane stride and a shuffle reduction), so
// like `tv_q8_h` it is compile-checked rather than executed. What IS executed
// is `q4_deq`, the arithmetic that decides whether a served row equals the row
// the file decodes to; the accumulation around it is mirrored below, word by
// word, in the kernel's order.
//
// matvec.cu is deliberately NOT included: under the host shim its `f2h` is a
// stub returning 0, which would nullify the executed mirror. TILE_COLS is
// pre-defined so tv_q4_h.cu's include guard skips it, and real conversions are
// supplied via _Float16 — clang's native IEEE binary16, exact widening and
// round-to-nearest-even narrowing, i.e. precisely what cvt.f32.f16 /
// cvt.rn.f16.f32 do on the device. Same choice, and same reason, as
// host_embq8.cpp.

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
static inline float warp_sum(float v) { return v; }  // compile-only path

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

#include "../kernels/tv_q4_h.cu"

// The extern __shared__ array and thread indices the (never-executed) kernel
// references.
float xs[1 << 15];
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
    auto hdr = read_n<unsigned>(in, 3, "en-tete");
    unsigned d_out = hdr[0], d_in = hdr[1], gpr = hdr[2];
    unsigned nwords = d_in >> 3;
    auto words = read_n<unsigned>(in, (std::size_t)d_out * nwords, "words");
    auto scales = read_n<unsigned short>(in, (std::size_t)d_out * gpr, "scales");
    auto biases = read_n<unsigned short>(in, (std::size_t)d_out * gpr, "biases");
    auto x = read_n<float>(in, d_in, "x");

    // ---- q4_deq over every weight, in the kernel's own addressing ----
    std::vector<float> deq((std::size_t)d_out * d_in);
    for (unsigned r = 0; r < d_out; ++r)
        for (unsigned c = 0; c < d_in; ++c) {
            unsigned q = (words[(std::size_t)r * nwords + (c >> 3)]
                          >> ((c & 7u) * 4u)) & 0xfu;
            unsigned g = r * gpr + (c >> 7);
            deq[(std::size_t)r * d_in + c] = q4_deq(q, scales[g], biases[g]);
        }

    // ---- tv_q4_h's accumulation, mirrored lane by lane ----
    std::vector<float> dot(d_out);
    for (unsigned r = 0; r < d_out; ++r) {
        float lanes[32] = {0};
        for (unsigned wi = 0; wi < nwords; ++wi) {
            unsigned p = words[(std::size_t)r * nwords + wi];
            unsigned c = wi << 3;
            unsigned short sb = scales[r * gpr + (c >> 7)];
            unsigned short bb = biases[r * gpr + (c >> 7)];
            float& acc = lanes[wi & 31u];
            for (unsigned k = 0; k < 8u; ++k)
                acc = __fmaf_rn(q4_deq((p >> (4u * k)) & 0xfu, sb, bb), x[c + k], acc);
        }
        // The butterfly of warp_sum, in its exact order.
        for (int k = 16; k > 0; k >>= 1) {
            float nv[32];
            for (int i = 0; i < 32; ++i) nv[i] = lanes[i] + lanes[i ^ k];
            std::memcpy(lanes, nv, sizeof nv);
        }
        dot[r] = lanes[0];
    }

    std::fwrite(deq.data(), 4, deq.size(), stdout);
    std::fwrite(dot.data(), 4, dot.size(), stdout);
    return 0;
}
