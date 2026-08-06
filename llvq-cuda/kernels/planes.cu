// tv_planes: the fused matvec over a Planes14 stream.
//
// Same grid (one warp per row, 8 rows per 256-thread block), same tiling,
// same two barriers, same shared staging of the activation and the same tail
// epilogue as tv_slot in matvec.cu — only the block decode differs, so a
// millisecond delta between the two arms can only come from the layout.
//
// Like tv_slot, there is deliberately no `if (row >= d_out) return;`: a
// return before `__syncthreads()` deadlocks, and it would break the full-warp
// mask of `warp_sum`. The host asserts `d_out % 8 == 0` instead.
//
// This file cannot `#include` tv_slot / tv_f16 under NVRTC: the loader has no
// filesystem, so the host concatenates the sources (lib.rs explains why — the
// string handed to NVRTC must be closed under its own hash). The guards below
// therefore key on macros the earlier parts define — TILE_COLS comes from
// matvec.cu — and only resolve from disk under a host clang++ syntax check.
// Order is the caller's contract: llvq_slot.cuh, matvec.cu, llvq_planes.cuh,
// then this file.

#ifndef TILE_COLS
#include "matvec.cu"
#endif
#ifndef LLVQ_PLANES_CUH
#include "llvq_planes.cuh"
#endif

extern "C" __global__ void tv_planes(const u32* __restrict__ words,
                                     const ClassRec* __restrict__ tab,
                                     const float* __restrict__ gscale,
                                     const float* __restrict__ rscale,
                                     const float* __restrict__ tail,
                                     const float* __restrict__ x,
                                     float* __restrict__ y,
                                     u32 nblocks,
                                     u32 tail_w)
{
    extern __shared__ float xs[];
    u32 lane = threadIdx.x & 31u;
    u32 row  = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    u32 b0r  = row * nblocks;
    float acc = 0.0f;

    u32 ntiles = (nblocks + TILE_BLOCKS - 1u) / TILE_BLOCKS;
    for (u32 t = 0; t < ntiles; ++t) {
        u32 jlo = t * TILE_BLOCKS;
        u32 jhi = jlo + TILE_BLOCKS < nblocks ? jlo + TILE_BLOCKS : nblocks;
        u32 n   = (jhi - jlo) * LLVQ_DIM;
        __syncthreads();
        for (u32 i = threadIdx.x; i < n; i += blockDim.x) xs[i] = x[jlo * LLVQ_DIM + i];
        __syncthreads();

        for (u32 j = jlo + lane; j < jhi; j += 32u)
            acc += planes_dot(words, tab, gscale, b0r + j, xs + (j - jlo) * LLVQ_DIM);
    }

    acc = warp_sum(acc);
    if (lane == 0) {
        float tv = 0.0f;
        u32 tc0 = nblocks * LLVQ_DIM;
        for (u32 i = 0; i < tail_w; ++i) tv += tail[row * tail_w + i] * x[tc0 + i];
        y[row] = acc * rscale[row] + tv;
    }
}
