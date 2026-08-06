// Drives the q8 embedding CUDA kernels on the CPU.
//
// Reads a fixture on stdin, writes results on stdout, both little-endian
// binary. The Rust side builds the fixture and checks the answers against
// llvq-artifact's own dequantizer, so this file holds no expectations — it is
// a harness, not a reference (same pattern as llvq-cuda/tests/host_planes.cpp).
//
//   in : u32 rows, u32 d, u32 gpr, u32 ntok
//        u32[rows*d/4] packed words, u16[rows*gpr] scales, u16[rows*gpr] biases,
//        u32[ntok] ids, u16[d] x (f16 bits)
//   out: u16[ntok*d]  gather output (f16 bits) — emb_q8_gather, executed,
//        u32[rows*d]  dequant of every weight (f32 bits) — q8_deq, executed,
//        f32[rows]    per-row dot against x — driver loop over q8_deq +
//                     __fmaf_rn in the kernel's lane order, warp butterfly
//                     reproduced exactly.
//
// `q8_deq` and `emb_q8_gather` are scalar register arithmetic, so they are
// *executed* here — gather with blockDim.x = 1, which makes its stride-loop
// cover every column. `tv_q8_h` needs a real warp (a 32-lane stride and a
// shuffle reduction) so it is compile-checked only; its dequant IS `q8_deq`
// and its accumulation is mirrored by the driver below, lane by lane.
//
// matvec.cu is deliberately NOT included: under the host shim its h2f/f2h
// stubs return 0, which would nullify the executed mirror. TILE_COLS is
// pre-defined so emb_q8.cu's include guard skips it, and real conversions are
// supplied via _Float16 — clang's native IEEE binary16, exact widening and
// round-to-nearest-even narrowing, i.e. precisely what cvt.f32.f16 /
// cvt.rn.f16.f32 do on the device.

#include "../../llvq-cuda/tests/host_shim.h"
#include "../../llvq-cuda/kernels/llvq_slot.cuh"  // u32, LLVQ_DIM

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define TILE_COLS (128u * LLVQ_DIM)  // skip matvec.cu in emb_q8.cu

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

#include "../kernels/emb_q8.cu"

// The extern __shared__ array and thread indices the (never-executed) matvec
// kernel references; the gather uses none of them but the driver sets the
// indices it does read.
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
    auto hdr = read_n<unsigned>(in, 4, "en-tete");
    unsigned rows = hdr[0], d = hdr[1], gpr = hdr[2], ntok = hdr[3];
    auto words = read_n<unsigned>(in, (std::size_t)rows * d / 4, "words");
    auto scales = read_n<unsigned short>(in, (std::size_t)rows * gpr, "scales");
    auto biases = read_n<unsigned short>(in, (std::size_t)rows * gpr, "biases");
    auto ids = read_n<unsigned>(in, ntok, "ids");
    auto x = read_n<unsigned short>(in, d, "x");

    // ---- emb_q8_gather, executed: one block per token, one thread ----
    std::vector<unsigned short> y((std::size_t)ntok * d);
    for (unsigned t = 0; t < ntok; ++t) {
        blockIdx.x = t;
        emb_q8_gather(words.data(), scales.data(), biases.data(), ids.data(),
                      y.data(), d, gpr, 0);
    }

    // ---- q8_deq over every weight, the dequant both kernels share ----
    std::vector<float> deq((std::size_t)rows * d);
    for (unsigned r = 0; r < rows; ++r)
        for (unsigned c = 0; c < d; ++c) {
            unsigned q = (words[(std::size_t)r * (d / 4) + (c >> 2)]
                          >> ((c & 3u) * 8u)) & 0xffu;
            unsigned g = r * gpr + (c >> 6);
            deq[(std::size_t)r * d + c] = q8_deq(q, scales[g], biases[g]);
        }

    // ---- tv_q8_h's accumulation, mirrored lane by lane ----
    std::vector<float> xstage(d);
    for (unsigned i = 0; i < d; ++i) xstage[i] = h2f(x[i]);
    std::vector<float> dot(rows);
    for (unsigned r = 0; r < rows; ++r) {
        float lanes[32] = {0};
        for (unsigned wi = 0; wi < d / 4; ++wi) {
            unsigned p = words[(std::size_t)r * (d / 4) + wi];
            unsigned c = wi << 2;
            unsigned short sb = scales[r * gpr + (c >> 6)];
            unsigned short bb = biases[r * gpr + (c >> 6)];
            float& acc = lanes[wi & 31u];
            acc = __fmaf_rn(q8_deq(p & 0xffu, sb, bb), xstage[c], acc);
            acc = __fmaf_rn(q8_deq((p >> 8) & 0xffu, sb, bb), xstage[c + 1], acc);
            acc = __fmaf_rn(q8_deq((p >> 16) & 0xffu, sb, bb), xstage[c + 2], acc);
            acc = __fmaf_rn(q8_deq(p >> 24, sb, bb), xstage[c + 3], acc);
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
