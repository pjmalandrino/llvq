// tv_f1r_v2: the F1 floor arm of `f1rank.cu` with the trellis by F₂ algebra —
// `tv_f1r`, same signature, same tile loop, same barriers, same epilogue, the
// decode+FMA of the block replaced by `f1r_dot_v2` (`llvq_f1rank_v2.cuh`).
//
// What this arm changes, and nothing else: the three chained small-table
// reads per block (`prefixes` → `branches` → `suffixes`) become twelve masked
// XORs on the word's own bits, so the only loads of a block are its 6-byte
// word and the three 16 KiB-table rows, none of which depends on another
// load. The 24 values are `f1r_val`'s, converted at the FMA exactly as
// `tv_f1r` converts them; the accumulation differs by ONE rounding per block
// (`f1r_dot_v2` sums the 24 products from zero, then the block's dot joins
// the running sum), which is inside the bench's y-equality control.
//
// The three table pointers `prefixes`, `branches`, `suffixes` are in the
// signature so the bench launches every `f1r` arm with one argument list;
// this arm never dereferences them.
//
// Measured against `tv_f1r` in ONE process by `bin/f1rankfloor`, round by
// round: `Du_v2 = t(f1r_v2) − t(word)`, `T_v2 = t(f1r_v2) − t(nullk)`. Read
// `f1rank.cu` for the ladder, the 36 stream copies, and the trap of a load
// that feeds nothing — the decoded values ARE the multipliers here too.
//
// Same assembly contract as `f1rank.cu`: NVRTC has no file system, the host
// concatenates `llvq_slot.cuh`, `matvec.cu`, `llvq_f1rank.cuh`,
// `llvq_f1rank_v2.cuh`, this file; the guards below only resolve from disk
// under a host check (`bin/cuhcheck`).

#ifndef TILE_COLS
#include "matvec.cu"
#endif
#ifndef LLVQ_F1RANK_CUH
#include "llvq_f1rank.cuh"
#endif
#ifndef LLVQ_F1RANK_V2_CUH
#include "llvq_f1rank_v2.cuh"
#endif

// ---------------------------------------------------------------------------
// Rung 2, variant 2 — the stream plus the decode, trellis by algebra. The 3
// row reads go through global memory; the pattern bytes come out of
// `f1r_v2_patterns` in registers; the 24 values are the multipliers.
// ---------------------------------------------------------------------------
extern "C" __global__ void tv_f1r_v2(const u32* __restrict__ words,
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
    // Carried whole for the one argument list; only `rows` is read.
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
            acc += f1r_dot_v2(lo, hi16, tab, xb);
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
