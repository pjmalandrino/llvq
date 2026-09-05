// tv_f1r_v1: `tv_f1r` with the V1 value construction — the same word stream,
// the same 16 KiB table, the same three trellis tables, the same 24 FMAs in
// the same order; only the path from three rows to 24 floats differs
// (`llvq_f1rank_v1.cuh`: byte lanes, a LOP3 mux for the sign, and the
// 2²³-bias trick instead of an I2F per coordinate).
//
// Everything else is `f1rank.cu`'s `tv_f1r`, copied rather than shared: the
// signature, the grid (one warp per row, 256 threads = 8 rows per block), the
// TILE_BLOCKS tile of x staged in shared, the two barriers, `warp_sum`, the
// tail epilogue and the store of `y`. The shell is what must not vary between
// arms (`docs/format-noyau.md` §6); the bench launches this kernel with the
// argument list of `tv_f1r`, in the same order.
//
// The chain is `tv_f1r`'s: `acc = __fmaf_rn(y_s, xb[s], acc)` for s = 0..23,
// across every block of the row, with `y_s` bit for bit the float `tv_f1r`
// converts from its `i8`. So the row this arm writes equals the row `tv_f1r`
// writes exactly, and the bench's equality control between the two is a
// comparison of identical floats, not a tolerance.
//
// Same assembly contract as `f1rank.cu`: NVRTC has no file system, the host
// concatenates `llvq_slot.cuh`, `matvec.cu`, `llvq_f1rank.cuh`, `f1rank.cu`,
// `llvq_f1rank_v1.cuh`, this file, `nullk.cu`; the guards below only resolve
// from disk under a host check (`bin/cuhcheck`). Nothing here is named in
// `f1rank.cu`, so the two files concatenate in either order.

#ifndef TILE_COLS
#include "matvec.cu"
#endif
#ifndef LLVQ_F1RANK_CUH
#include "llvq_f1rank.cuh"
#endif
#ifndef LLVQ_F1RANK_V1_CUH
#include "llvq_f1rank_v1.cuh"
#endif

extern "C" __global__ void tv_f1r_v1(const u32* __restrict__ words,
                                     u32 row_stride_u32,
                                     const u32* __restrict__ rows,
                                     const unsigned char* __restrict__ prefixes,
                                     const unsigned short* __restrict__ branches,
                                     const unsigned char* __restrict__ suffixes,
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
    const u32* wrow = words + row * row_stride_u32;
    F1rTables tab = { rows, prefixes, branches, suffixes };
    float acc = 0.0f;

    u32 ntiles = (nblocks + TILE_BLOCKS - 1u) / TILE_BLOCKS;
    for (u32 t = 0; t < ntiles; ++t) {
        u32 jlo = t * TILE_BLOCKS;
        u32 jhi = jlo + TILE_BLOCKS < nblocks ? jlo + TILE_BLOCKS : nblocks;
        u32 n   = (jhi - jlo) * LLVQ_DIM;
        __syncthreads();
        for (u32 i = threadIdx.x; i < n; i += blockDim.x) xs[i] = x[jlo * LLVQ_DIM + i];
        __syncthreads();

        for (u32 j = jlo + lane; j < jhi; j += 32u) {
            const float* xb = xs + (j - jlo) * LLVQ_DIM;
            u32 lo, hi16;
            f1r_load(wrow, j, lo, hi16);
            acc = f1r_dot_v1_acc(lo, hi16, tab, xb, acc);
        }
    }

    acc = warp_sum(acc);
    if (lane == 0) {
        float tv = 0.0f;
        u32 tc0 = nblocks * LLVQ_DIM;
        for (u32 i = 0; i < tail_w; ++i) tv += tail[row * tail_w + i] * x[tc0 + i];
        y[row] = acc * rscale[row] + tv;
    }
}
