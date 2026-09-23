// Drives the q4 embedding CUDA kernels on the CPU.
//
// host_embq8.cpp at half the weight width; see that file for the method. Reads
// a fixture on stdin, writes results on stdout, little-endian binary. The Rust
// side (tests/embed_q4.rs) builds the fixture and checks the answers against
// llvq-artifact's own dequantizer, so this file holds no expectations.
//
//   in : u32 rows, u32 d, u32 gpr, u32 ntok
//        u32[rows*d/8] packed words, u16[rows*gpr] scales, u16[rows*gpr] biases,
//        u32[ntok] ids, u16[d] x (f16 bits)
//   out: u16[ntok*d]  gather output (f16 bits), emb_q4_gather executed,
//        u32[rows*d]  dequant of every weight (f32 bits), e4_deq executed
//                     through the kernel's own word and nibble addressing,
//        f32[rows]    per-row dot against x, tv_emb_q4_h's accumulation
//                     mirrored lane by lane, butterfly included.
//
// `tv_emb_q4_h` needs a real warp, so it is compile-checked only; its dequant
// IS `e4_deq` and its accumulation is mirrored below in its lane order.
// matvec.cu is not included, for host_embq8.cpp's reason: real conversions
// come from _Float16, clang's own binary16.

#include "../../llvq-cuda/tests/host_shim.h"
#include "../../llvq-cuda/kernels/llvq_slot.cuh"  // u32, LLVQ_DIM

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define TILE_COLS (128u * LLVQ_DIM)  // skip matvec.cu in emb_q4.cu

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

#include "../kernels/emb_q4.cu"

float xs[1 << 15];
Dim3 blockIdx{0, 0, 0};
Dim3 threadIdx{0, 0, 0};
Dim3 blockDim{1, 1, 1};

template <typename T>
static std::vector<T> read_n(std::FILE* f, std::size_t n, const char* what) {
    std::vector<T> v(n);
    if (n && std::fread(v.data(), sizeof(T), n, f) != n) {
        std::fprintf(stderr, "truncated fixture: %s\n", what);
        std::exit(2);
    }
    return v;
}

int main() {
    std::FILE* in = stdin;
    auto hdr = read_n<unsigned>(in, 4, "header");
    unsigned rows = hdr[0], d = hdr[1], gpr = hdr[2], ntok = hdr[3];
    if (d % 8u != 0u) {
        std::fprintf(stderr, "d = %u is not a multiple of 8\n", d);
        return 2;
    }
    auto words = read_n<unsigned>(in, (std::size_t)rows * d / 8, "words");
    auto scales = read_n<unsigned short>(in, (std::size_t)rows * gpr, "scales");
    auto biases = read_n<unsigned short>(in, (std::size_t)rows * gpr, "biases");
    auto ids = read_n<unsigned>(in, ntok, "ids");
    auto x = read_n<unsigned short>(in, d, "x");

    // ---- emb_q4_gather, executed: one block per token, one thread ----
    std::vector<unsigned short> y((std::size_t)ntok * d);
    for (unsigned t = 0; t < ntok; ++t) {
        blockIdx.x = t;
        emb_q4_gather(words.data(), scales.data(), biases.data(), ids.data(),
                      y.data(), d, gpr, 0);
    }

    // ---- e4_deq over every weight, with the matvec's addressing ----
    std::vector<float> deq((std::size_t)rows * d);
    for (unsigned r = 0; r < rows; ++r)
        for (unsigned wi = 0; wi < d / 8; ++wi) {
            unsigned p = words[(std::size_t)r * (d / 8) + wi];
            unsigned c = wi << 3;
            unsigned g = r * gpr + (c >> 6);
            for (unsigned k = 0; k < 8; ++k)
                deq[(std::size_t)r * d + c + k] =
                    e4_deq((p >> (4u * k)) & 0xfu, scales[g], biases[g]);
        }

    // ---- tv_emb_q4_h's accumulation, mirrored lane by lane ----
    std::vector<float> xstage(d);
    for (unsigned i = 0; i < d; ++i) xstage[i] = h2f(x[i]);
    std::vector<float> dot(rows);
    for (unsigned r = 0; r < rows; ++r) {
        float lanes[32] = {0};
        for (unsigned wi = 0; wi < d / 8; ++wi) {
            unsigned p = words[(std::size_t)r * (d / 8) + wi];
            unsigned c = wi << 3;
            unsigned short sb = scales[r * gpr + (c >> 6)];
            unsigned short bb = biases[r * gpr + (c >> 6)];
            float& acc = lanes[wi & 31u];
            for (unsigned k = 0; k < 8; ++k)
                acc = __fmaf_rn(e4_deq((p >> (4u * k)) & 0xfu, sb, bb), xstage[c + k], acc);
        }
        // The butterfly of warp_sum, in its exact order.
        for (int k = 16; k > 0; k >>= 1) {
            float nv[32];
            for (int i = 0; i < 32; ++i) nv[i] = lanes[i] + lanes[i ^ k];
            std::memcpy(lanes, nv, sizeof nv);
        }
        dot[r] = lanes[0];
    }

    std::fwrite(y.data(), 2, y.size(), stdout);
    std::fwrite(deq.data(), 4, deq.size(), stdout);
    std::fwrite(dot.data(), 4, dot.size(), stdout);
    return 0;
}
