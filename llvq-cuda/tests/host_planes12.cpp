// Drives the Planes12x CUDA decoder AND its overlay correction on the CPU.
//
// Reads a fixture on stdin, writes results on stdout, both little-endian
// binary. The Rust side builds the fixture and checks the answers against its
// own decoders and an f64 reference, so this file holds no expectations — it
// is a harness, not a reference.
//
//   in : u32 d_out, u32 nblocks, u32 n_exc, u32 nwords (main stream),
//        u32 newords (exception records), u32 ntab
//        u32[nwords] words, u32[n_exc] exc_idx, u32[newords] exc_words,
//        u32[ntab] tab, f32[2] gscale, f32[d_out] rscale,
//        f32[nblocks*24] x
//   out: u32[d_out*nblocks*24] slots (level | sign << 3, main stream),
//        u32[d_out*nblocks*2] cls (class id, gain, main stream),
//        f32[d_out] y (row pass + exception corrections — the overlay)
//
// `planes12_fields`, `planes12_dot`, `planes12x_locate`, `planes12x_lane_terms`
// and `planes12x_combine` are scalar register arithmetic and are *executed*
// here, exactly as host_planes.cpp executes planes_dot. The warp of the
// correction pass is emulated by a serial loop over the 32 lanes accumulating
// what warp_sum would reduce; atomicAdd by a plain `+=` — a single thread is
// the one schedule where that is equivalent, which is why the true atomicity
// can only be proven on the card. tv_planes12x itself is compile-only, but
// including planes12.cu still catches every syntax and type error in it.

#include "host_shim.h"

// planes12.cu calls atomicAdd, which host_shim.h does not provide: define the
// serial equivalent before the kernel text is included, so the TU type-checks
// the exact call the device makes.
static inline float atomicAdd(float* addr, float v) {
    float old = *addr;
    *addr += v;
    return old;
}

// Deliberately in the order the host concatenates for NVRTC, so every
// `#ifndef` guard takes the same branch here as on the device: llvq_slot.cuh,
// matvec.cu, llvq_planes.cuh, planes.cu, llvq_planes12.cuh, planes12.cu.
#include "../kernels/llvq_slot.cuh"
#define TILE_BLOCKS 128u
#include "../kernels/matvec.cu"
#include "../kernels/llvq_planes.cuh"
#include "../kernels/planes.cu"
#include "../kernels/llvq_planes12.cuh"
#include "../kernels/planes12.cu"

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
    auto hdr = read_n<unsigned>(in, 6, "en-tete");
    unsigned d_out = hdr[0], nblocks = hdr[1], n_exc = hdr[2];
    unsigned nwords = hdr[3], newords = hdr[4], ntab = hdr[5];
    std::size_t ntotal = static_cast<std::size_t>(d_out) * nblocks;

    auto words = read_n<unsigned>(in, nwords, "words");
    auto exc_idx = read_n<unsigned>(in, n_exc, "exc_idx");
    auto exc_words = read_n<unsigned>(in, newords, "exc_words");
    auto tab_raw = read_n<unsigned>(in, ntab, "tab");
    auto gscale = read_n<float>(in, 2, "gscale");
    auto rscale = read_n<float>(in, d_out, "rscale");
    auto x = read_n<float>(in, static_cast<std::size_t>(nblocks) * LLVQ_DIM, "x");

    const ClassRec* tab = reinterpret_cast<const ClassRec*>(tab_raw.data());

    std::vector<unsigned> slots(ntotal * LLVQ_DIM, 0);
    std::vector<unsigned> cls(ntotal * 2, 0);
    std::vector<float> y(d_out, 0.0f);

    // ---- the main-stream fields, block by block ----
    for (std::size_t b = 0; b < ntotal; ++b) {
        Planes12Fields f = planes12_fields(words.data(), static_cast<u32>(b));
        cls[b * 2]     = f.id;
        cls[b * 2 + 1] = f.gain;
        for (unsigned j = 0; j < LLVQ_DIM; ++j) {
            unsigned lev = ((f.p0 >> j) & 1u) | ((f.p1 >> j) & 1u) << 1;
            unsigned sign = (f.smask >> j) & 1u;
            slots[b * LLVQ_DIM + j] = lev | sign << 3;
        }
    }

    // ---- the row pass: what tv_planes12x's row CTAs compute ----
    // Serial sum per row (the kernel's four-chain association differs; the
    // Rust side compares against f64 with a tolerance, as the bench does).
    for (unsigned row = 0; row < d_out; ++row) {
        float acc = 0.0f;
        for (unsigned j = 0; j < nblocks; ++j)
            acc += planes12_dot(words.data(), tab, gscale.data(),
                                row * nblocks + j,
                                x.data() + static_cast<std::size_t>(j) * LLVQ_DIM);
        y[row] += acc * rscale[row];  // += : the correction pass adds on top
    }

    // ---- the correction pass: the kernel's own pieces, lane by lane ----
    for (unsigned e = 0; e < n_exc; ++e) {
        Planes12xExc loc = planes12x_locate(exc_idx.data(), e, nblocks);
        PlanesFields fe = planes_fields(exc_words.data(), e);
        Planes12Fields fa = planes12_fields(words.data(), loc.b);
        const ClassRec re = tab[fe.id];
        const ClassRec ra = tab[fa.id];
        float ve = 0.0f, va = 0.0f;
        for (u32 lane = 0; lane < 32u; ++lane) {  // warp_sum, serially
            float te, ta;
            planes12x_lane_terms(fe, fa, re, ra, lane,
                                 x.data() + static_cast<std::size_t>(loc.col) * LLVQ_DIM,
                                 &te, &ta);
            ve += te;
            va += ta;
        }
        y[loc.row] += planes12x_combine(ve, va, gscale.data(), fe.gain, fa.gain,
                                        rscale[loc.row]);
    }

    std::fwrite(slots.data(), sizeof(unsigned), slots.size(), stdout);
    std::fwrite(cls.data(), sizeof(unsigned), cls.size(), stdout);
    std::fwrite(y.data(), sizeof(float), y.size(), stdout);
    return 0;
}
