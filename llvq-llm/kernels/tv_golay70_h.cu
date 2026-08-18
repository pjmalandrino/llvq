// tv_golay70_h: the fused matvec over a Golay70 stream, storing f16 — the
// shape the inference path needs, and a *different grid* from the bench's.
//
// ## What Golay70 is, in one paragraph
//
// A 9-byte main stream whose per-slot decode goes through the rank of the
// block's Golay codeword — the residue-mod-4 plane of every block IS a
// codeword, so 12 bits of rank replace what the bit-plane layouts spend on
// explicit level indices — plus a per-matrix exception table carrying the
// exact 14-byte Planes14 record of every block the 2-bit in-class code cannot
// hold (7.4357 % of the published 4B, 7.4116 % of the 8B — counted, not
// assumed). The main stream stores the zero-contribution ORIGIN at every
// exception, so the correction ADDS the exact contribution and subtracts
// nothing — simpler than Planes12x, whose main stream carries an
// approximation that must be cancelled.
//
// ## Why this is not golay70.cu with an f16 store
//
// Same reasoning, word for word, as tv_planes12x_h.cu (read its header — the
// design decision lives there): the bench kernel's two-region grid forces a
// memset of y before every launch and an atomic on the store, neither of
// which the inference path can want. Here `exc_idx` is ascending and blocks
// are row-major, so each output row's exceptions are a contiguous slice of
// the exception table; the host ships the `d_out + 1` prefix offsets
// (`row_exc`, built and *checked* by `fused::row_offsets`) and every row
// applies its own corrections inside its own warp. No CTA touches a row
// another CTA touches: no atomic, no memset, `y` stays uninitialised, f32
// accumulation runs end to end and the only narrowing to binary16 is the
// final store.
//
// The serial-corrections bound is looser than Planes12x's and worth stating:
// at 7.44 % of blocks a row holds `0.0744·nblocks` exceptions — **7.9** on
// the 4B's narrowest projections (nblocks = 106), **30.1** on its `down_proj`
// (405). Still under 32, so no lane takes more than one exception on the 4B —
// but `down_proj` sits one model size from the edge: the 8B's 512-block rows
// hold **38.1**, where lanes start taking two. That is a second serial pass,
// not a cliff, and it lands on the layout's *heaviest-exception* rows; the
// number to watch in any 8B fusedrun, exactly as tv_planes12x_h.cu says of
// its own bound.
//
// ## 🚨 What is NOT established on the machine that wrote this
//
// There is no CUDA card here. `golay70_fields`, `golay70_prologue`,
// `golay70_slot_value`, `golay70_dot` (the v2 decoder, proven against the
// Rust reference over the whole sealed 4B) and `golay70_row_correction`
// below are scalar register arithmetic and are **executed** by
// `tests/host_golay70_h.cpp` against an f64 reference. The kernel body — the
// grid, the two barriers, `warp_sum`, the f16 store — is **compile-only**
// there. Nothing on this machine establishes that this kernel is correct.
// The first run on a card must diff its greedy tokens against the dense arm.
//
// NVRTC has no filesystem, so the host concatenates the sources; the guards
// below only resolve from disk under a host clang++ syntax check. Order is
// the caller's contract (`fused::planes_source_names`): llvq_slot.cuh,
// matvec.cu (TILE_COLS, warp_sum, f2h, tail_dot_h), llvq_rot.cuh, rotate.cu,
// llvq_planes.cuh (planes_fields), planes.cu, tv_planes_h.cu,
// llvq_planes12.cuh, tv_planes12x_h.cu, llvq_golay.cuh (golay70_dot v2 and
// the exception helpers), then this file.

#ifndef TILE_COLS
#include "../../llvq-cuda/kernels/matvec.cu"
#endif
#ifndef LLVQ_GOLAY_CUH
#include "../../llvq-cuda/kernels/llvq_golay.cuh"
#endif

// One lane's share of one row's exception corrections, WITHOUT the row scale.
//
// `[e0, e1)` is the row's slice of the exception table and `b0r = row·nblocks`
// its first block. Lane `l` takes exceptions e0+l, e0+l+32, … and computes
// each one *serially over its 24 slots* through `golay70_exc_lane_term` — the
// same device function the bench kernel spreads across a warp. Same text,
// different assignment of work to lanes (the tv_planes12x_h.cu pattern).
//
// Unlike `planes12x_row_correction` there is NO approximate term to read
// back from the main stream and subtract: the transcoder holed every
// exception block to the origin, whose contribution is exactly 0.0f, and the
// CPU reference decoder asserts that invariant. One record read, one exact
// term, added under its own gain centroid.
//
// `xb` is read from GLOBAL memory — an exception's block column need not be
// in the tile the row pass last staged, it is touched exactly once, and
// staging it would cost a barrier for nothing. The return value carries
// `gscale` (inside `golay70_exc_combine`) but not `rscale` — the caller
// applies that once to `acc + corr`, which is why `rs = 1`.
__device__ __forceinline__ float golay70_row_correction(
        const u32* __restrict__ exc_idx,
        const u32* __restrict__ exc_words,
        const ClassRec* __restrict__ tab,
        const float* __restrict__ gscale,
        const float* __restrict__ x,
        u32 e0,
        u32 e1,
        u32 b0r,
        u32 lane)
{
    float corr = 0.0f;
    for (u32 e = e0 + lane; e < e1; e += 32u) {
        u32 b = exc_idx[e];
        const float* xb = x + (b - b0r) * LLVQ_DIM;
        PlanesFields fe = planes_fields(exc_words, e);
        const ClassRec re = tab[fe.id];
        float ve = 0.0f;
        for (u32 j = 0; j < LLVQ_DIM; ++j)
            ve += golay70_exc_lane_term(fe, re, j, xb);
        corr += golay70_exc_combine(ve, gscale, fe.gain, 1.0f);
    }
    return corr;
}

// One warp per output row, 8 rows per 256-thread block — `tv_planes_h`'s grid
// exactly, with `golay70_dot` (the v2 block-prologue decoder) for the main
// stream and one extra term for the row's exceptions.
//
// Like every kernel of this family there is deliberately no
// `if (row >= d_out) return;`: a return before `__syncthreads()` deadlocks,
// and it would break the full-warp mask of `warp_sum`. The host asserts
// `d_out % 8 == 0` instead.
extern "C" __global__ void tv_golay70_h(const u32* __restrict__ words,
                                        const u32* __restrict__ exc_idx,
                                        const u32* __restrict__ exc_words,
                                        const u32* __restrict__ row_exc,
                                        const u32* __restrict__ cwtab,
                                        const GolayClassRec* __restrict__ gtab,
                                        const ClassRec* __restrict__ tab,
                                        const float* __restrict__ gscale,
                                        const float* __restrict__ rscale,
                                        const unsigned short* __restrict__ tail,
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
            acc += golay70_dot(words, cwtab, gtab, gscale, b0r + j,
                               xs + (j - jlo) * LLVQ_DIM);
    }

    // After the last barrier, and reading `x` from global: the correction
    // introduces no barrier, so its data-dependent trip count — lanes of the
    // same warp take different numbers of exceptions — cannot deadlock
    // anything. `row_exc[row]` and `row_exc[row + 1]` are warp-uniform.
    acc += golay70_row_correction(exc_idx, exc_words, tab, gscale, x,
                                  row_exc[row], row_exc[row + 1], b0r, lane);

    acc = warp_sum(acc);
    if (lane == 0) {
        float tv = tail_dot_h(tail, x + nblocks * LLVQ_DIM, row, tail_w);
        y[row] = f2h(acc * rscale[row] + tv);
    }
}
