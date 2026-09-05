// tv_f1r_*: the COMPILED FLOOR of the F1 universal-table decoder — the word
// stream and the full decode, in `tv_nullk`'s geometry, nothing else.
//
// The table floor of 2026-09-05 priced the lookups of an F1 decoder swept by
// footprint, with the indices manufactured by a hash and no weight stream; it
// said a 16 KiB table costs 0.34 to 0.66 ms (`docs/mesures/f1-plancher-table-
// 2026-09-05.txt`). The universal 16 KiB table exists and its quality is
// measured (−0.6 pp against exact F1, `llvq-bench/examples/f1rankbench.rs`).
// What nobody has measured is the decode itself, COMPILED: read six bytes a
// block, three table rows and three pattern bytes, unroll 24 coordinates,
// multiply. That is where E1v died — 79 registers, 0.25× f16 — and the
// question is registers, local bytes and milliseconds over the 252 launches.
// `proofs/preregistration-f1-rang-plancher-2026-09-05.md`.
//
// ## The ladder, and why no arm is read on its own
//
//     nullk   the same pass without one byte of weights      (the floor, in THIS process)
//     word    nullk + the 6-byte word per block, folded into a float, no decode
//     f1r     word + the full decode: 3 rows, 3 pattern bytes, 24 coordinates, 24 FMAs
//
//     S  = t(word) − t(nullk)    the F1 stream in our geometry
//     Du = t(f1r)  − t(word)     table + arithmetic decode
//     T  = t(f1r)  − t(nullk)    what F1 spends on stream AND decode
//
// Every arm runs `tv_nullk`'s grid in `tv_nullk`'s process: one warp per row,
// 256 threads = 8 rows per block, the same TILE_BLOCKS tile of x staged in
// shared, the same two barriers, the same `warp_sum`, the same tail epilogue,
// the same store of `y`. The loop skeleton below is `f1floor.cu`'s, copied
// rather than shared, because the shell is what must not vary
// (`docs/format-noyau.md` §6).
//
// ## The word stream, and why there are 36 copies of it
//
// Row-major; row stride in bytes = round_up(nblocks·6, 8); block j of a row
// starts at byte 6j; lane j reads the two aligned u32 covering it
// (`f1r_load`, llvq_f1rank.cuh, with the byte arithmetic written out). A warp
// reads 192 contiguous bytes per iteration — coalesced, and exactly the
// pattern an F1 kernel would have, since the word served is the word stored.
//
// ⚠️ The bench allocates **36 DISTINCT copies** of the stream, one per
// "layer", ~0.91 GB on the device at 6 bytes a block (the prereg's 0.98 GB is
// the 2.16 b/weight disk figure). With one buffer per shape the 25 MB of words
// a layer touches would sit in the L40S's 96 MiB L2 from the second layer on,
// and `S` would measure the L2, not the DRAM. The words are generated ON THE
// DEVICE (`f1r_fill`): uploading 0.9 GB would need that much host RAM in a
// container whose limit nobody here knows — the same reasoning as
// `f1floorbench`'s 4 GiB point — and the host replays the mixer to check the
// decode instead.
//
// ## The trap, and what defuses it
//
// 🚨 A kernel whose loads feed nothing is a kernel the compiler deletes.
// `tv_f1r_word` folds the two loaded words into a float that multiplies every
// one of the 24 slots, as `f1floor.cu`'s `f1_fold` does; `tv_f1r`'s decoded
// coordinates ARE the multipliers. The FMA count is identical to `nullk`'s in
// both, so the arms stay comparable. `bin/f1rankfloor` asserts that `f1r`'s
// output differs from `word`'s and from `nullk`'s, and `word`'s from
// `nullk`'s; a deleted load shows up as that check failing.
//
// Unlike `nullk` and the table floor, `tv_f1r` HAS a reference: `tv_f1r_dump`
// writes decoded blocks back and the bench compares every coordinate to
// `llvq_bench::f1::rank::decode_word` on the same words before it prints a
// time. It is still a floor and not a cost — no gain scale, uniform labels
// rather than a model's, no Planes14 in the process.
//
// Same assembly contract as `f1floor.cu` and `nullk.cu`: NVRTC has no file
// system, the host concatenates `llvq_slot.cuh`, `matvec.cu`,
// `llvq_f1rank.cuh`, this file, `nullk.cu`; the guards below only resolve from
// disk under a host check (`bin/cuhcheck`).

#ifndef TILE_COLS
#include "matvec.cu"
#endif
#ifndef LLVQ_F1RANK_CUH
#include "llvq_f1rank.cuh"
#endif

// The two loaded words folded into a multiplier, the low byte only, exactly
// as `f1_fold`: one instruction, and dependent on both loads so neither can
// be dropped.
__device__ __forceinline__ float f1r_fold(u32 lo, u32 hi16)
{
    return (float)((lo ^ hi16) & 0xffu);
}

// ---------------------------------------------------------------------------
// Rung 1 — the stream. nullk plus the 6-byte word per block, no decode.
// ---------------------------------------------------------------------------
extern "C" __global__ void tv_f1r_word(const u32* __restrict__ words,
                                       u32 row_stride_u32,
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
            float f = f1r_fold(lo, hi16);
#pragma unroll
            for (u32 s = 0; s < LLVQ_DIM; ++s) acc = __fmaf_rn(f, xb[s], acc);
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

// ---------------------------------------------------------------------------
// Rung 2 — the stream plus the decode. The 3 row reads and the 3 pattern
// reads go through global memory; the 24 coordinates come out of `f1r_decode`
// in registers and each one is the multiplier of its slot.
// ---------------------------------------------------------------------------
extern "C" __global__ void tv_f1r(const u32* __restrict__ words,
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
            signed char yv[LLVQ_DIM];
            f1r_decode(lo, hi16, tab, yv);
#pragma unroll
            for (u32 s = 0; s < LLVQ_DIM; ++s) acc = __fmaf_rn((float)yv[s], xb[s], acc);
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

// ---------------------------------------------------------------------------
// The on-card correctness control. Thread j < ndump decodes block j of the
// stream in row-major block order — row j / nblocks, block j % nblocks, so
// the row stride is exercised as the timed kernels exercise it — and writes
// its 24 coordinates to `out[24j + s]`, which the bench compares to the Rust
// reference on the words the host replays. Trivial grid: no shared memory,
// no barrier, so the early return is legal here.
// ---------------------------------------------------------------------------
extern "C" __global__ void tv_f1r_dump(const u32* __restrict__ words,
                                       u32 row_stride_u32,
                                       const u32* __restrict__ rows,
                                       const unsigned char* __restrict__ prefixes,
                                       const unsigned short* __restrict__ branches,
                                       const unsigned char* __restrict__ suffixes,
                                       signed char* __restrict__ out,
                                       u32 nblocks,
                                       u32 ndump)
{
    u32 j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= ndump) return;
    u32 row = j / nblocks;
    u32 jb  = j - row * nblocks;
    F1rTables tab = { rows, prefixes, branches, suffixes };
    u32 lo, hi16;
    f1r_load(words + row * row_stride_u32, jb, lo, hi16);
    signed char yv[LLVQ_DIM];
    f1r_decode(lo, hi16, tab, yv);
#pragma unroll
    for (u32 s = 0; s < LLVQ_DIM; ++s) out[j * LLVQ_DIM + s] = yv[s];
}

// ---------------------------------------------------------------------------
// The stream generator: `buf[i] = f1r_mix32(seed + i)` for i < n, one word
// per thread, the host launching ceil(n / 256) blocks. Replayed in Rust by
// `bin/f1rankfloor::mix32` for the dump check.
// ---------------------------------------------------------------------------
extern "C" __global__ void f1r_fill(u32* __restrict__ buf, u32 n, u32 seed)
{
    u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) buf[i] = f1r_mix32(seed + i);
}
