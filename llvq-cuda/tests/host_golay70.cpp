// Drives the Golay70 CUDA decoder AND its exception correction on the CPU.
//
// Reads a fixture on stdin, writes results on stdout, both little-endian
// binary. The Rust side builds the fixture and checks the answers against its
// own decoders and an f64 reference, so this file holds no expectations — it
// is a harness, not a reference.
//
//   in : u32 d_out, u32 nblocks, u32 n_exc, u32 nwords (main stream),
//        u32 newords (exception records), u32 ntab (ClassRec words),
//        u32 ngtab (GolayClassRec words), u32 ncw (codewords)
//        u32[nwords] words, u32[n_exc] exc_idx, u32[newords] exc_words,
//        u32[ntab] tab, u32[ngtab] gtab, u32[ncw] cw,
//        f32[2] gscale, f32[d_out] rscale, f32[nblocks*24] x
//   out: f32[d_out*nblocks*24] slot values (main stream, sign applied,
//        no gscale/rscale — golay70_slot_value verbatim),
//        u32[d_out*nblocks*3] cls (class id, gain, golay rank),
//        f32[d_out] y (row pass + exception corrections)
//
// `golay70_fields`, `golay70_slot_value`, `golay70_dot`, `planes12x_locate`,
// `golay70_exc_lane_term` and `golay70_exc_combine` are scalar register
// arithmetic and are *executed* here. The warp of the correction pass is
// emulated by a serial loop over the 32 lanes accumulating what warp_sum
// would reduce; atomicAdd by a plain `+=` — a single thread is the one
// schedule where that is equivalent, which is why the true atomicity can
// only be proven on the card. tv_golay70 itself is compile-only, but
// including golay70.cu still catches every syntax and type error in it.

#include "host_shim.h"

// golay70.cu calls atomicAdd, which host_shim.h does not provide: define the
// serial equivalent before the kernel text is included, so the TU type-checks
// the exact call the device makes.
static inline float atomicAdd(float* addr, float v) {
    float old = *addr;
    *addr += v;
    return old;
}

// Deliberately in the order the host concatenates for NVRTC, so every
// `#ifndef` guard takes the same branch here as on the device: llvq_slot.cuh,
// matvec.cu, llvq_planes.cuh, planes.cu, llvq_planes12.cuh, planes12.cu,
// llvq_golay.cuh, golay70.cu.
#include "../kernels/llvq_slot.cuh"
#define TILE_BLOCKS 128u
#include "../kernels/matvec.cu"
#include "../kernels/llvq_planes.cuh"
#include "../kernels/planes.cu"
#include "../kernels/llvq_planes12.cuh"
#include "../kernels/planes12.cu"
#include "../kernels/llvq_golay.cuh"
#include "../kernels/golay70.cu"

#include <cstdio>
#include <cstdlib>
#include <vector>

// The `extern __shared__` arrays and thread indices the (never-executed)
// kernels reference. `__shared__` expands to nothing under the shim.
float xs[TILE_BLOCKS * LLVQ_DIM];
Dim3 blockIdx{0, 0, 0};
Dim3 threadIdx{0, 0, 0};
Dim3 blockDim{256, 1, 1};

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
    auto hdr = read_n<unsigned>(in, 8, "en-tete");
    unsigned d_out = hdr[0], nblocks = hdr[1], n_exc = hdr[2];
    unsigned nwords = hdr[3], newords = hdr[4], ntab = hdr[5];
    unsigned ngtab = hdr[6], ncw = hdr[7];
    std::size_t ntotal = static_cast<std::size_t>(d_out) * nblocks;

    auto words = read_n<unsigned>(in, nwords, "words");
    auto exc_idx = read_n<unsigned>(in, n_exc, "exc_idx");
    auto exc_words = read_n<unsigned>(in, newords, "exc_words");
    auto tab_raw = read_n<unsigned>(in, ntab, "tab");
    auto gtab_raw = read_n<unsigned>(in, ngtab, "gtab");
    auto cw = read_n<unsigned>(in, ncw, "cw");
    auto gscale = read_n<float>(in, 2, "gscale");
    auto rscale = read_n<float>(in, d_out, "rscale");
    auto x = read_n<float>(in, static_cast<std::size_t>(nblocks) * LLVQ_DIM, "x");

    const ClassRec* tab = reinterpret_cast<const ClassRec*>(tab_raw.data());
    const GolayClassRec* gtab = reinterpret_cast<const GolayClassRec*>(gtab_raw.data());

    std::vector<float> slots(ntotal * LLVQ_DIM, 0.0f);
    std::vector<unsigned> cls(ntotal * 3, 0);
    std::vector<float> y(d_out, 0.0f);

    // ---- the main-stream fields and per-slot values, block by block ----
    for (std::size_t b = 0; b < ntotal; ++b) {
        Golay70Fields f = golay70_fields(words.data(), static_cast<u32>(b));
        const GolayClassRec r = gtab[f.id];
        u32 c = cw[f.g];
        cls[b * 3]     = f.id;
        cls[b * 3 + 1] = f.gain;
        cls[b * 3 + 2] = f.g;
        for (u32 j = 0; j < LLVQ_DIM; ++j)
            slots[b * LLVQ_DIM + j] = golay70_slot_value(r, c, f, j);
    }

    // ---- the row pass: what tv_golay70's row CTAs compute ----
    // Serial sum per row (the kernel's four-chain association differs; the
    // Rust side compares against f64 with a tolerance, as the bench does).
    for (unsigned row = 0; row < d_out; ++row) {
        float acc = 0.0f;
        for (unsigned j = 0; j < nblocks; ++j)
            acc += golay70_dot(words.data(), cw.data(), gtab, gscale.data(),
                               row * nblocks + j,
                               x.data() + static_cast<std::size_t>(j) * LLVQ_DIM);
        y[row] += acc * rscale[row];  // += : the correction pass adds on top
    }

    // ---- the correction pass: the kernel's own pieces, lane by lane ----
    for (unsigned e = 0; e < n_exc; ++e) {
        Planes12xExc loc = planes12x_locate(exc_idx.data(), e, nblocks);
        PlanesFields fe = planes_fields(exc_words.data(), e);
        const ClassRec re = tab[fe.id];
        float ve = 0.0f;
        for (u32 lane = 0; lane < 32u; ++lane)  // warp_sum, serially
            ve += golay70_exc_lane_term(
                fe, re, lane,
                x.data() + static_cast<std::size_t>(loc.col) * LLVQ_DIM);
        y[loc.row] += golay70_exc_combine(ve, gscale.data(), fe.gain, rscale[loc.row]);
    }

    std::fwrite(slots.data(), sizeof(float), slots.size(), stdout);
    std::fwrite(cls.data(), sizeof(unsigned), cls.size(), stdout);
    std::fwrite(y.data(), sizeof(float), y.size(), stdout);
    return 0;
}
