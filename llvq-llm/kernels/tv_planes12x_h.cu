// tv_planes12x_h: the fused matvec over a Planes12x stream, storing f16 —
// the shape the inference path needs, and a *different grid* from the bench's.
//
// ## What Planes12x is, in one paragraph
//
// A 12-byte main stream whose every record decodes through at most four
// magnitude levels, plus a per-matrix exception table carrying the exact
// 14-byte Planes14 record of every 5-level block (5 096 688 of the published
// 4B's 150 681 600, 3.3824 %). The main stream stores an *approximation* of
// those blocks — their best L <= 4 direction, gain copied — so any consumer
// owes them a correction of (exact − approx)·x before its result is the
// Planes14 result. That is the whole layout: 4.342 b/weight at the bench
// against 4.804, at bit-identical reconstruction.
//
// ## Why this is not planes12.cu with an f16 store
//
// `tv_planes12x` (llvq-cuda, the M2 bench arm) splits the grid in two
// regions — row CTAs and exception CTAs — which are unordered with respect to
// one another and both accumulate into the same `y[row]`. That forces two
// things this path cannot want:
//
//   * **a memset of `y` before every launch.** `FusedOp::cuda_fwd` allocates
//     its output uninitialised precisely because `tv_planes_h` writes every
//     row exactly once; an overlay that *adds* into `y` would turn that into
//     252 extra memsets per token, on a decode where launch overhead was
//     measured at ~48 % of the token.
//   * **an atomic on the store.** The model wants halves, so the choice would
//     be `atomicAdd` on `__half` — which rounds to binary16 once per partial
//     sum instead of once at the end, and is therefore *not* the arithmetic
//     the Planes14 arm it replaces performs — or an f32 buffer plus a
//     narrowing pass, i.e. a third launch per projection.
//
// This kernel dissolves the question instead of answering it. `exc_idx` is
// ascending and blocks are row-major (`b = row·nblocks + col`), so **each
// output row's exceptions are a contiguous slice of the exception table**.
// The host ships the `d_out + 1` prefix offsets of those slices (`row_exc`,
// built and *checked* by `fused::row_offsets`), and every row applies its own
// corrections inside its own warp. Rows then partition the output exactly as
// in `tv_planes_h`:
//
//   * no CTA touches a row another CTA touches, so there is no atomic, and
//     `y` may stay uninitialised — the grid is exact and every row is stored
//     once, which is the invariant `tv_planes_h` already relies on;
//   * the correction is folded into `acc` *before* the single `warp_sum` and
//     the single `rscale`, so f32 accumulation runs end to end and the only
//     narrowing to binary16 is the final store — the same rounding structure
//     as the Planes14 arm, at the same place candle would have narrowed.
//
// The price is that a row's corrections are serial within its own warp rather
// than parallel across extra CTAs. It is bounded, and the bound is worth
// writing down because it is what makes the trade obviously right at these
// shapes. At the measured 3.3824 % of blocks, a row holds `0.0338·nblocks`
// exceptions: **3.6** on the 4B's narrowest projections (nblocks = 106),
// **13.7** on its `down_proj` (405), **17.3** on the 8B's (512). All under
// 32, so with `e += 32` **no lane ever takes more than one exception** and
// the whole correction is one extra pass over 24 slots in a warp that has
// just done `nblocks/32` blocks' worth. A layout or a model that pushed a row
// past 32 exceptions would start serialising, and that is the number to watch
// — not the percentage.
//
// ## 🚨 What is NOT established on the machine that wrote this
//
// There is no CUDA card here. `planes12_dot` and `planes12x_row_correction`
// are scalar register arithmetic and are **executed** by
// `tests/host_planes12x_h.cpp`, lane by lane, against an f64 reference — that
// part is checked, and so is `tail_dot_h`, by `tests/host_tail_h.cpp`. The
// kernel body below — the grid, the two barriers,
// `warp_sum`, the f16 store — is **compile-only** there, exactly as
// `llvq-cuda/tests/host_matvec.cpp` says of `tv_slot`: "a single-threaded
// driver cannot reproduce a barrier or a warp shuffle". Nothing on this
// machine establishes that this kernel is correct. The first run on a card
// must diff its logits against the `Planes14` arm.
//
// NVRTC has no filesystem, so the host concatenates the sources; the guards
// below only resolve from disk under a host clang++ syntax check. Order is
// the caller's contract (`fused_cuda::planes_source_names`): llvq_slot.cuh,
// matvec.cu (TILE_COLS, warp_sum, f2h, tail_dot_h), llvq_rot.cuh, rotate.cu,
// llvq_planes.cuh (planes_fields), planes.cu, llvq_planes12.cuh
// (planes12_fields, planes12_dot, planes12x_lane_terms, planes12x_combine),
// then this file.

#ifndef TILE_COLS
#include "../../llvq-cuda/kernels/matvec.cu"
#endif
#ifndef LLVQ_PLANES12_CUH
#include "../../llvq-cuda/kernels/llvq_planes12.cuh"
#endif

// One lane's share of one row's exception corrections, WITHOUT the row scale.
//
// `[e0, e1)` is the row's slice of the exception table and `b0r = row·nblocks`
// its first block. Lane `l` takes exceptions e0+l, e0+l+32, … and computes
// each one *serially over its 24 slots* through `planes12x_lane_terms` — the
// same device function the bench kernel spreads across a warp. Same text,
// different assignment of work to lanes: there, one warp per exception and
// one slot per lane, closed by two `warp_sum`s; here, one lane per exception
// and 24 slots in sequence, closed by the caller's existing `warp_sum` over
// `acc`. That is what removes the per-exception reduction *and* the atomic.
//
// The column is `b − b0r`, a subtraction, not the division `planes12x_locate`
// pays to recover a row this kernel already owns. What makes that legitimate
// is entirely host-side: `fused::row_offsets` refuses to build a `row_exc`
// whose slice for row r holds a block outside [r·nblocks, (r+1)·nblocks).
//
// `xb` is read from GLOBAL memory. An exception's block column need not be in
// the tile the row pass last staged, it is touched exactly once, and staging
// it would cost a barrier for nothing.
//
// The return value carries `gscale` (per block, applied inside
// `planes12x_combine`) but not `rscale` — the caller applies that once to
// `acc + corr`, which is why `planes12x_combine` is handed `rs = 1`.
//
// ⚠️ The cancellation is exact in ℝ, not in f32. `planes12_dot` sums a block
// over four interleaved FMA chains; the `va` term below sums the same 24
// products serially, so the approximation the row pass added and the one this
// subtracts differ in the last ulp. The residue is ~1e-7 of one block's
// contribution on 3.4 % of blocks — three orders under the binary16 store
// that follows — and it is the same residue the bench arm carries, which is
// why both are checked against an f64 reference at 1e-5·Σ|w·x| rather than
// asserted bit-equal to Planes14.
__device__ __forceinline__ float planes12x_row_correction(
        const u32* __restrict__ words,
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
        PlanesFields   fe = planes_fields(exc_words, e);
        Planes12Fields fa = planes12_fields(words, b);
        const ClassRec re = tab[fe.id];
        const ClassRec ra = tab[fa.id];
        float ve = 0.0f, va = 0.0f;
        for (u32 j = 0; j < LLVQ_DIM; ++j) {
            float te, ta;
            planes12x_lane_terms(fe, fa, re, ra, j, xb, &te, &ta);
            ve += te;
            va += ta;
        }
        corr += planes12x_combine(ve, va, gscale, fe.gain, fa.gain, 1.0f);
    }
    return corr;
}

// One warp per output row, 8 rows per 256-thread block — `tv_planes_h`'s grid
// exactly, with `planes12_dot` for the main stream and one extra term.
//
// Like every kernel of this family there is deliberately no
// `if (row >= d_out) return;`: a return before `__syncthreads()` deadlocks,
// and it would break the full-warp mask of `warp_sum`. The host asserts
// `d_out % 8 == 0` instead.
extern "C" __global__ void tv_planes12x_h(const u32* __restrict__ words,
                                          const u32* __restrict__ exc_idx,
                                          const u32* __restrict__ exc_words,
                                          const u32* __restrict__ row_exc,
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
            acc += planes12_dot(words, tab, gscale, b0r + j, xs + (j - jlo) * LLVQ_DIM);
    }

    // After the last barrier, and reading `x` from global: the correction
    // introduces no barrier, so its data-dependent trip count — lanes of the
    // same warp take different numbers of exceptions — cannot deadlock
    // anything. `row_exc[row]` and `row_exc[row + 1]` are warp-uniform.
    acc += planes12x_row_correction(words, exc_idx, exc_words, tab, gscale, x,
                                    row_exc[row], row_exc[row + 1], b0r, lane);

    acc = warp_sum(acc);
    if (lane == 0) {
        float tv = tail_dot_h(tail, x + nblocks * LLVQ_DIM, row, tail_w);
        y[row] = f2h(acc * rscale[row] + tv);
    }
}
