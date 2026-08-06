// tv_planes12x: the fused matvec over a Planes12x stream — main pass and
// exception correction in ONE launch.
//
// The grid has two regions, split by `row_cta`:
//
//   * CTAs [0, row_cta) are tv_planes verbatim — same tiling, same two
//     barriers, same shared staging of the activation — with planes12_dot
//     instead of planes_dot, and the final store replaced by an atomicAdd:
//     the correction CTAs of the same launch target the same y[row], and no
//     intra-launch ordering exists between CTAs, so BOTH sides must
//     accumulate into a y the host has zeroed beforehand. (One extra
//     memset per launch; it is part of what the layout costs and the bench
//     times it inside the arm.)
//
//   * CTAs [row_cta, gridDim.x) handle the exceptions, one per warp: warp w
//     of CTA c owns exception e = (c - row_cta)·8 + w. Lanes 0..23 each hold
//     one slot — exact value from the 14-byte record (3-plane tree), approx
//     value re-read from the main stream at the block's index (2-plane
//     tree) — times x[col·24 + lane] read from GLOBAL memory (the staged xs
//     belongs to the row CTAs; an exception touches 24 floats once, staging
//     would cost a barrier for nothing). Two warp reductions, then lane 0
//     adds (ve·gscale − va·gscale)·rscale[row] to y[row] atomically.
//
// The exception path has no __syncthreads and no early return before one, so
// the whole-warp `if (e >= n_exc) return` is safe (e is warp-uniform), and
// the full-warp shuffle mask of warp_sum holds. The row region keeps
// tv_planes' no-return rule for exactly the reasons matvec.cu documents.
//
// Same NVRTC concatenation contract as planes.cu; order is the caller's:
// llvq_slot.cuh, matvec.cu, llvq_planes.cuh, planes.cu, llvq_planes12.cuh,
// then this file.

#ifndef TILE_COLS
#include "matvec.cu"
#endif
#ifndef LLVQ_PLANES12_CUH
#include "llvq_planes12.cuh"
#endif

extern "C" __global__ void tv_planes12x(const u32* __restrict__ words,
                                        const u32* __restrict__ exc_idx,
                                        const u32* __restrict__ exc_words,
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
        // ---- row region: tv_planes with the 2-plane decoder ----
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
                acc += planes12_dot(words, tab, gscale, b0r + j,
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
        // ---- correction region: one exception per warp ----
        u32 e = (blockIdx.x - row_cta) * (blockDim.x >> 5) + (threadIdx.x >> 5);
        if (e >= n_exc) return;
        Planes12xExc  loc = planes12x_locate(exc_idx, e, nblocks);
        PlanesFields   fe = planes_fields(exc_words, e);
        Planes12Fields fa = planes12_fields(words, loc.b);
        const ClassRec re = tab[fe.id];
        const ClassRec ra = tab[fa.id];
        float ve, va;
        planes12x_lane_terms(fe, fa, re, ra, lane, x + loc.col * LLVQ_DIM, &ve, &va);
        ve = warp_sum(ve);
        va = warp_sum(va);
        if (lane == 0)
            atomicAdd(&y[loc.row],
                      planes12x_combine(ve, va, gscale, fe.gain, fa.gain,
                                        rscale[loc.row]));
    }
}
