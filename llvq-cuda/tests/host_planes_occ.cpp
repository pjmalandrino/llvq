// Drives the two scalar helpers of the A3 occupancy kernels on the CPU, and
// compiles the seven kernels themselves.
//
// Reads a fixture on stdin, writes results on stdout, both little-endian
// binary. The Rust side (`tests/planes_occ_matches_rust.rs`) builds the
// fixture and checks the answers against `llvq_cuda::occ`, so this file
// holds no expectations — it is a harness, not a reference.
//
//   in : u32 n_idx, u32 n_slice
//        u32[n_idx] i                        (staged float indices to map)
//        u32[n_slice * 3] (nblocks, nsplit, s)
//   out: u32[n_idx] xs_index<24>, u32[n_idx] xs_index<28>,
//        u32[n_slice * 2] (klo, khi)
//
// `occ_xs_index` and `occ_slice` are scalar register arithmetic — no warp
// primitive, no shared memory, no atomic — so they are *executed* here,
// exactly as host_planes.cpp executes planes_dot. The seven kernels
// (`tv_planes_pad`, `_mr2`, `_mr4`, `_mr2p`, `_pers`, `_sk`, `_persall`)
// are compile-only: a single-threaded driver reproduces neither
// `__syncthreads` nor a shuffle nor the ticket atomic. Including
// planes_occ.cu still catches every syntax and type error in them before one
// costs a billed job — and it instantiates every template the device build
// will, so a template that only fails at instantiation fails here.

#include "host_shim.h"
// Deliberately in the order the host concatenates for NVRTC, so every
// `#ifndef` guard takes the same branch here as on the device.
#include "../kernels/llvq_slot.cuh"
#define TILE_BLOCKS 128u
#include "../kernels/matvec.cu"
#include "../kernels/llvq_planes.cuh"
#include "../kernels/planes.cu"
#include "../kernels/planes_seg.cu"
#include "../kernels/planes_occ.cu"

#include <cstdio>
#include <cstdlib>
#include <vector>

// The `extern __shared__` array and the thread indices the (never-executed)
// kernels reference. Sized for the padded stride, the widest staging.
float xs[TILE_BLOCKS * LLVQ_XS_PAD];
Dim3 blockIdx{0, 0, 0};
Dim3 threadIdx{0, 0, 0};
Dim3 blockDim{256, 1, 1};
Dim3 gridDim{1, 1, 1};

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
    auto hdr = read_n<unsigned>(in, 2, "en-tete");
    unsigned n_idx = hdr[0], n_slice = hdr[1];
    auto idx = read_n<unsigned>(in, n_idx, "indices");
    auto sl = read_n<unsigned>(in, static_cast<std::size_t>(n_slice) * 3, "slices");

    std::vector<unsigned> out;
    out.reserve(static_cast<std::size_t>(n_idx) * 2 + static_cast<std::size_t>(n_slice) * 2);
    for (unsigned i = 0; i < n_idx; ++i) out.push_back(occ_xs_index<LLVQ_DIM>(idx[i]));
    for (unsigned i = 0; i < n_idx; ++i) out.push_back(occ_xs_index<LLVQ_XS_PAD>(idx[i]));
    for (unsigned i = 0; i < n_slice; ++i) {
        unsigned klo = 0, khi = 0;
        occ_slice(sl[i * 3], sl[i * 3 + 1], sl[i * 3 + 2], &klo, &khi);
        out.push_back(klo);
        out.push_back(khi);
    }
    std::fwrite(out.data(), sizeof(unsigned), out.size(), stdout);
    return 0;
}
