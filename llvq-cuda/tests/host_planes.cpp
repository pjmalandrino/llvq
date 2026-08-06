// Drives the Planes14 CUDA decoder on the CPU, one block at a time.
//
// Reads a fixture on stdin, writes results on stdout, both little-endian
// binary. The Rust side builds the fixture and checks the answers against its
// own decoders, so this file holds no expectations — it is a harness, not a
// reference.
//
//   in : u32 nblocks, u32 nwords, u32 ntab, u32 nx
//        u32[nwords] words (the Planes14 stream, padded and packed),
//        u32[ntab] tab, f32[2] gscale, f32[nx] x   (nx = nblocks * 24)
//   out: u32[nblocks*24] slots (level | sign << 3),
//        u32[nblocks*2] cls (class id, gain),
//        f32[nblocks] dot
//
// `planes_fields` and `planes_dot` are scalar register arithmetic — no warp
// primitive, no shared memory — so they are *executed* here, exactly as
// host_probe.cpp executes slot_dot. `tv_planes` itself is compile-only (a
// single-threaded driver reproduces neither `__syncthreads` nor a shuffle);
// including planes.cu still catches every syntax and type error in it before
// one costs a billed job.

#include "host_shim.h"
// Deliberately in the order the host concatenates for NVRTC, so every
// `#ifndef` guard takes the same branch here as on the device: llvq_slot.cuh,
// matvec.cu (which needs TILE_BLOCKS and defines TILE_COLS and warp_sum),
// llvq_planes.cuh, planes.cu.
#include "../kernels/llvq_slot.cuh"
#define TILE_BLOCKS 128u
#include "../kernels/matvec.cu"
#include "../kernels/llvq_planes.cuh"
#include "../kernels/planes.cu"

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
    auto hdr = read_n<unsigned>(in, 4, "en-tete");
    unsigned nblocks = hdr[0], nwords = hdr[1], ntab = hdr[2], nx = hdr[3];

    auto words = read_n<unsigned>(in, nwords, "words");
    auto tab_raw = read_n<unsigned>(in, ntab, "tab");
    auto gscale = read_n<float>(in, 2, "gscale");
    auto x = read_n<float>(in, nx, "x");

    const ClassRec* tab = reinterpret_cast<const ClassRec*>(tab_raw.data());

    std::vector<unsigned> slots(static_cast<std::size_t>(nblocks) * LLVQ_DIM, 0);
    std::vector<unsigned> cls(static_cast<std::size_t>(nblocks) * 2, 0);
    std::vector<float> dot(nblocks, 0.0f);

    for (unsigned b = 0; b < nblocks; ++b) {
        PlanesFields f = planes_fields(words.data(), b);
        cls[b * 2]     = f.id;
        cls[b * 2 + 1] = f.gain;
        for (unsigned j = 0; j < LLVQ_DIM; ++j) {
            unsigned lev = ((f.p0 >> j) & 1u) | ((f.p1 >> j) & 1u) << 1
                         | ((f.p2 >> j) & 1u) << 2;
            unsigned sign = (f.smask >> j) & 1u;
            slots[static_cast<std::size_t>(b) * LLVQ_DIM + j] = lev | sign << 3;
        }
        dot[b] = planes_dot(words.data(), tab, gscale.data(), b,
                            x.data() + static_cast<std::size_t>(b) * LLVQ_DIM);
    }

    std::fwrite(slots.data(), sizeof(unsigned), slots.size(), stdout);
    std::fwrite(cls.data(), sizeof(unsigned), cls.size(), stdout);
    std::fwrite(dot.data(), sizeof(float), dot.size(), stdout);
    return 0;
}
