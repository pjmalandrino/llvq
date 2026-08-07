// tv_golay70: the fused matvec over a Golay70 stream — main pass and
// exception correction in ONE launch, the exact motif of tv_planes12x.
//
// The grid has two regions, split by `row_cta`:
//
//   * CTAs [0, row_cta) are tv_planes verbatim — same tiling, same two
//     barriers, same shared staging of the activation — with golay70_dot
//     instead of planes_dot, and the final store replaced by an atomicAdd:
//     the correction CTAs of the same launch target the same y[row], no
//     intra-launch ordering exists between CTAs, so BOTH sides accumulate
//     into a y the host has zeroed beforehand (the memset is timed inside
//     the arm — part of what the layout costs).
//
//   * CTAs [row_cta, gridDim.x) handle the exceptions, one per warp: warp w
//     of CTA c owns exception e = (c - row_cta)·8 + w. Lanes 0..23 each hold
//     one slot of the exact 14-byte record — the same Planes14 read path as
//     Planes12x's corrections — times x[col·24 + lane] read from GLOBAL
//     memory (an exception touches 24 floats once; staging would cost a
//     barrier for nothing). One warp reduction, then lane 0 adds
//     ve·gscale·rscale[row] to y[row] atomically. Unlike Planes12x there is
//     no approximate term to subtract: the main stream carries the ORIGIN at
//     every exception, whose contribution is exactly zero.
//
// The exception path has no __syncthreads and no early return before one, so
// the whole-warp `if (e >= n_exc) return` is safe (e is warp-uniform), and
// the full-warp shuffle mask of warp_sum holds. The row region keeps
// tv_planes' no-return rule for exactly the reasons matvec.cu documents.
//
// Same NVRTC concatenation contract as planes12.cu; order is the caller's:
// llvq_slot.cuh, matvec.cu, llvq_planes.cuh, planes.cu, llvq_planes12.cuh,
// planes12.cu, llvq_golay.cuh, then this file.

#ifndef TILE_COLS
#include "matvec.cu"
#endif
#ifndef LLVQ_GOLAY_CUH
#include "llvq_golay.cuh"
#endif

extern "C" __global__ void tv_golay70(const u32* __restrict__ words,
                                      const u32* __restrict__ exc_idx,
                                      const u32* __restrict__ exc_words,
                                      const u32* __restrict__ cwtab,
                                      const GolayClassRec* __restrict__ gtab,
                                      const ClassRec* __restrict__ tab,
                                      const float* __restrict__ gscale,
                                      const float* __restrict__ rscale,
                                      const float* __restrict__ tail,
                                      const float* __restrict__ x,
                                      float* __restrict__ y,
                                      u32 nblocks,
                                      u32 tail_w,
                                      u32 row_cta,
                                      u32 n_exc)
{
    extern __shared__ float xs[];
    u32 lane = threadIdx.x & 31u;

    if (blockIdx.x < row_cta) {
        // ---- row region: tv_planes with the Golay70 decoder ----
        u32 row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
        u32 b0r = row * nblocks;
        float acc = 0.0f;

        u32 ntiles = (nblocks + TILE_BLOCKS - 1u) / TILE_BLOCKS;
        for (u32 t = 0; t < ntiles; ++t) {
            u32 jlo = t * TILE_BLOCKS;
            u32 jhi = jlo + TILE_BLOCKS < nblocks ? jlo + TILE_BLOCKS : nblocks;
            u32 n   = (jhi - jlo) * LLVQ_DIM;
            __syncthreads();
            for (u32 i = threadIdx.x; i < n; i += blockDim.x)
                xs[i] = x[jlo * LLVQ_DIM + i];
            __syncthreads();

            for (u32 j = jlo + lane; j < jhi; j += 32u)
                acc += golay70_dot(words, cwtab, gtab, gscale, b0r + j,
                                   xs + (j - jlo) * LLVQ_DIM);
        }

        acc = warp_sum(acc);
        if (lane == 0) {
            float tv = 0.0f;
            u32 tc0 = nblocks * LLVQ_DIM;
            for (u32 i = 0; i < tail_w; ++i) tv += tail[row * tail_w + i] * x[tc0 + i];
            atomicAdd(&y[row], acc * rscale[row] + tv);
        }
    } else {
        // ---- correction region: one exception per warp, exact term only ----
        u32 e = (blockIdx.x - row_cta) * (blockDim.x >> 5) + (threadIdx.x >> 5);
        if (e >= n_exc) return;
        Planes12xExc loc = planes12x_locate(exc_idx, e, nblocks);
        PlanesFields  fe = planes_fields(exc_words, e);
        const ClassRec re = tab[fe.id];
        float ve = golay70_exc_lane_term(fe, re, lane, x + loc.col * LLVQ_DIM);
        ve = warp_sum(ve);
        if (lane == 0)
            atomicAdd(&y[loc.row],
                      golay70_exc_combine(ve, gscale, fe.gain, rscale[loc.row]));
    }
}
