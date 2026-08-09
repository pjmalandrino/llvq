// Drives the inference-side Planes12x overlay on the CPU.
//
// Reads a fixture on stdin, writes results on stdout, both little-endian
// binary. The Rust side builds the fixture and checks the answers against its
// own decoders and an f64 reference, so this file holds no expectations — it
// is a harness, not a reference.
//
//   in : u32 d_out, u32 nblocks, u32 n_exc, u32 nwords (main stream),
//        u32 newords (exception records), u32 ntab
//        u32[nwords] words, u32[n_exc] exc_idx, u32[newords] exc_words,
//        u32[d_out+1] row_exc, u32[ntab] tab,
//        f32[2] gscale, f32[d_out] rscale, f32[nblocks*24] x
//   out: f32[d_out] y    (row pass + corrections, times rscale — no tail)
//        f32[d_out] corr (the correction term alone, before rscale)
//
// ## What this file proves, and what it cannot
//
// `planes12_dot` and — the point of this harness — `planes12x_row_correction`
// are scalar register arithmetic: no barrier, no shuffle, no atomic. They are
// **executed** here, over all 32 lanes of each row, exactly as
// llvq-cuda/tests/host_planes12.cpp executes the bench arm's pieces. So the
// arithmetic that makes the overlay exact — the exception's exact record
// minus the main stream's swapped approximation — is checked on this machine.
//
// `tv_planes12x_h` itself is **compile-only**: a single-threaded driver
// reproduces neither `__syncthreads` nor `warp_sum`, and `f2h` is a PTX
// instruction the shim stubs to zero. Including the kernel still costs two
// seconds to catch every syntax and type error in it, which is the difference
// between finding a typo here and finding it in a billed job.
//
// The translation unit is assembled in the order fused_cuda concatenates for
// NVRTC (llvq_slot.cuh, matvec.cu, llvq_planes.cuh, planes.cu, tv_planes_h.cu,
// llvq_planes12.cuh, tv_planes12x_h.cu) minus the rotation pair, which shares
// no symbol with any of this. Every `#ifndef` guard therefore takes the same
// branch here as on the device.

#include "../../llvq-cuda/tests/host_shim.h"

#include "../../llvq-cuda/kernels/llvq_slot.cuh"
#define TILE_BLOCKS 128u
#include "../../llvq-cuda/kernels/matvec.cu"
#include "../../llvq-cuda/kernels/llvq_planes.cuh"
#include "../../llvq-cuda/kernels/planes.cu"
#include "../kernels/tv_planes_h.cu"
#include "../../llvq-cuda/kernels/llvq_planes12.cuh"
#include "../kernels/tv_planes12x_h.cu"

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

    auto words = read_n<unsigned>(in, nwords, "words");
    auto exc_idx = read_n<unsigned>(in, n_exc, "exc_idx");
    auto exc_words = read_n<unsigned>(in, newords, "exc_words");
    auto row_exc = read_n<unsigned>(in, d_out + 1u, "row_exc");
    auto tab_raw = read_n<unsigned>(in, ntab, "tab");
    auto gscale = read_n<float>(in, 2, "gscale");
    auto rscale = read_n<float>(in, d_out, "rscale");
    auto x = read_n<float>(in, static_cast<std::size_t>(nblocks) * LLVQ_DIM, "x");

    const ClassRec* tab = reinterpret_cast<const ClassRec*>(tab_raw.data());

    std::vector<float> y(d_out, 0.0f);
    std::vector<float> corr(d_out, 0.0f);

    for (unsigned row = 0; row < d_out; ++row) {
        u32 b0r = row * nblocks;

        // The row pass. Serial rather than the kernel's 32-lane stride: the
        // association differs, which is why the Rust side compares against an
        // f64 reference with a tolerance instead of demanding bit equality.
        float acc = 0.0f;
        for (unsigned j = 0; j < nblocks; ++j)
            acc += planes12_dot(words.data(), tab, gscale.data(), b0r + j,
                                x.data() + static_cast<std::size_t>(j) * LLVQ_DIM);

        // The correction, through the kernel's own device function, once per
        // lane — the serial stand-in for the `warp_sum` that collects the 32
        // partials on the card. This is the executed part.
        float c = 0.0f;
        for (u32 lane = 0; lane < 32u; ++lane)
            c += planes12x_row_correction(words.data(), exc_idx.data(), exc_words.data(),
                                          tab, gscale.data(), x.data(),
                                          row_exc[row], row_exc[row + 1], b0r, lane);

        corr[row] = c;
        y[row] = (acc + c) * rscale[row];
    }

    std::fwrite(y.data(), sizeof(float), y.size(), stdout);
    std::fwrite(corr.data(), sizeof(float), corr.size(), stdout);
    return 0;
}
