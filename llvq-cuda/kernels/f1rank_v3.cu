// tv_f1r_v3: the F1 floor arm of `f1rank.cu` with the values by byte tables
// and `prmt` — `tv_f1r`, same signature, same tile loop, same barriers, same
// epilogue, the decode+FMA of the block replaced by `f1r_dot_v3`
// (`llvq_f1rank_v3.cuh`).
//
// What this arm changes, and nothing else: after the six loads of a block —
// the same word, the same three trellis bytes, the same three 16 KiB-table
// rows as `tv_f1r` — the 24 values are looked up four at a time in register
// byte tables by `prmt`, chosen by a byte mask from the pattern bits, and
// turned into floats by placing each byte under `0x4b0000__` and one FADD.
// No int→float conversion, no per-coordinate select chain, no `i8` array.
// The 24 products are summed in `tv_f1r`'s order; the accumulation differs
// by ONE rounding per block (`f1r_dot_v3` sums from zero, then the block's
// dot joins the running sum), which is inside the bench's y-equality control.
//
// The three table pointers `prefixes`, `branches`, `suffixes` are in the
// signature and read, exactly as `tv_f1r` reads them, so the bench launches
// every `f1r` arm with one argument list.
//
// Measured against `tv_f1r` in ONE process by `bin/f1rankfloor`, round by
// round: `Du_v3 = t(f1r_v3) − t(word)`, `T_v3 = t(f1r_v3) − t(nullk)`. Read
// `f1rank.cu` for the ladder, the 36 stream copies, and the trap of a load
// that feeds nothing — the decoded values ARE the multipliers here too.
//
// Same assembly contract as `f1rank.cu`: NVRTC has no file system, the host
// concatenates `llvq_slot.cuh`, `matvec.cu`, `llvq_f1rank.cuh`,
// `llvq_f1rank_v3.cuh`, this file; the guards below only resolve from disk
// under a host check (`bin/cuhcheck`).

#ifndef TILE_COLS
#include "matvec.cu"
#endif
#ifndef LLVQ_F1RANK_CUH
#include "llvq_f1rank.cuh"
#endif
#ifndef LLVQ_F1RANK_V3_CUH
#include "llvq_f1rank_v3.cuh"
#endif

// ---------------------------------------------------------------------------
// Rung 2, variant 3 — the stream plus the decode, values by byte tables. The
// 3 row reads and the 3 pattern reads go through global memory; the 24
// values never leave registers, six packed bytes-quads per block, and each
// one is the multiplier of its slot inside `f1r_dot_v3`.
// ---------------------------------------------------------------------------
extern "C" __global__ void tv_f1r_v3(const u32* __restrict__ words,
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
            acc += f1r_dot_v3(lo, hi16, tab, xb);
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
