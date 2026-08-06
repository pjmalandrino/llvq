// tv_planes_h: tv_planes storing f16 — the shape the fused inference path needs.
//
// The exact counterpart of matvec.cu's tv_slot_h: same grid (one warp per row,
// 8 rows per 256-thread block), same tiling, same two barriers, same shared
// staging, same tail epilogue, f32 accumulation throughout — only the block
// decode is planes_dot instead of slot_dot, and only the final store narrows
// to f16, exactly where candle's conversion kernel would have narrowed it.
//
// This file lives in llvq-llm, not llvq-cuda, on purpose: the committed
// Planes14 kernels (llvq_planes.cuh + planes.cu) belong to the C1 lot and are
// reused verbatim; the half-storing entry point is a need of the inference
// runtime alone, the same way tv_slot_h is a separate kernel so tv_slot's
// register allocation cannot move under a bench's feet.
//
// Like every kernel of this family there is deliberately no
// `if (row >= d_out) return;`: a return before `__syncthreads()` deadlocks,
// and it would break the full-warp mask of `warp_sum`. The host asserts
// `d_out % 8 == 0` instead.
//
// NVRTC has no filesystem, so the host concatenates the sources; the guards
// below only resolve from disk under a host clang++ syntax check. Order is the
// caller's contract: llvq_slot.cuh, matvec.cu (TILE_COLS, warp_sum, f2h),
// llvq_planes.cuh (planes_dot), planes.cu, then this file.

#ifndef TILE_COLS
#include "../../llvq-cuda/kernels/matvec.cu"
#endif
#ifndef LLVQ_PLANES_CUH
#include "../../llvq-cuda/kernels/llvq_planes.cuh"
#endif

extern "C" __global__ void tv_planes_h(const u32* __restrict__ words,
                                       const ClassRec* __restrict__ tab,
                                       const float* __restrict__ gscale,
                                       const float* __restrict__ rscale,
                                       const float* __restrict__ tail,
                                       const float* __restrict__ x,
                                       unsigned short* __restrict__ y,
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
        y[row] = f2h(acc * rscale[row] + tv);
    }
}
