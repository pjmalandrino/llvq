// The FROZEN v1 Golay70 decoder — the witness arm of the v2 campaign.
//
// This file is a snapshot of the decode path that produced every published
// Golay70 number (e2-golay70-bench-2026-08-07, six-arm-awq-2026-08-10,
// baseline-head-2026-08-10): the per-slot coset resolution that llvq_golay.cuh
// replaced with the v2 block-level prologue on 2026-08-11. It exists so the
// v2/v1 ratio can be formed ROUND BY ROUND in one process — the §4 rule of
// proofs/preregistration-2026-08-10.md forbids citing a ratio against a set
// of arms that did not produce it, and the published v1 milliseconds came
// from another job.
//
// ## The freeze contract
//
// Exactly THREE symbols are renamed from the published text, nothing else may
// differ by one character:
//
//   golay70_slot_value  ->  golay70_slot_value_v1
//   golay70_dot         ->  golay70_dot_v1        (+ its call site)
//   tv_golay70          ->  tv_golay70_v1         (+ its call to the dot)
//
// Everything both versions share IS shared, not duplicated: `GolayClassRec`,
// `Golay70Fields`, `golay70_fields`, the exception helpers and the stream
// format are character-identical between v1 and v2 (not one stored byte
// moved), so this file uses the definitions from llvq_golay.cuh — both arms
// therefore read the SAME device buffers, and the v1 arm adds zero VRAM.
//
// The identity v1 ≡ v2 is asserted bit for bit by the host harness
// (tests/host_golay70.cpp), which executes both slot paths and both row
// passes on the same fixture — so a drift in EITHER copy fails the suite on
// the development machine, before any card.
//
// Same NVRTC concatenation contract as golay70.cu; this file is appended
// LAST (after awq_gemv.cu) so adding the witness arm never reorders the
// fragments of the published arms.

#ifndef TILE_COLS
#include "matvec.cu"
#endif
#ifndef LLVQ_GOLAY_CUH
#include "llvq_golay.cuh"
#endif

// One slot's signed value — the unified even/odd formula, fully predicated.
//
// The low select bit is A on both cosets; the high bit is the codeword bit
// (even: residue) or the B bit (odd: level high bit). The sign is the B bit
// (even: smask) or `cbit ^ flag(selected value)` (odd: the forced-sign rule).
// `flags >> sel` is a register shift of a scalar — no memory, no divergence.
__device__ __forceinline__ float golay70_slot_value_v1(const GolayClassRec& r,
                                                       u32 cw,
                                                       const Golay70Fields& f,
                                                       u32 j)
{
    u32 bj   = 1u << j;
    u32 cbit = (cw & bj) ? 1u : 0u;
    u32 abit = (f.a & bj) ? 1u : 0u;
    u32 bbit = (f.bm & bj) ? 1u : 0u;
    bool odd = r.is_odd != 0u;

    u32 hi   = odd ? bbit : cbit;
    float v  = hi ? (abit ? r.v[3] : r.v[2]) : (abit ? r.v[1] : r.v[0]);
    u32 sel  = (hi << 1) | abit;
    u32 flag = (r.flags >> sel) & 1u;
    u32 neg  = odd ? (cbit ^ flag) : bbit;
    return neg ? -v : v;
}

// One main-stream block's contribution to a dot product — the Golay70 twin of
// planes12_dot. Same four independent FMA chains, same accumulator selection
// by residue mod 4, same final parenthesisation, same gscale epilogue, so the
// arm is comparable against the shared f64 reference.
//
// An exception block carries the origin record (class 0): every slot selects
// a zero from the all-zero table entry, the contribution is exactly 0.0f (at
// worst -0.0f, additively neutral), and the correction pass of the same
// launch adds the exact contribution — nothing to subtract.
__device__ __forceinline__ float golay70_dot_v1(const u32* __restrict__ words,
                                                const u32* __restrict__ cwtab,
                                                const GolayClassRec* __restrict__ gtab,
                                                const float* __restrict__ gscale,
                                                u32 b,
                                                const float* xb)
{
    Golay70Fields f = golay70_fields(words, b);
    const GolayClassRec r = gtab[f.id];
    u32 cw = cwtab[f.g];

    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
#pragma unroll
    for (u32 i = 0; i < LLVQ_DIM; i += 4) {
#pragma unroll
        for (u32 k = 0; k < 4; ++k) {
            u32 j   = i + k;
            float v = golay70_slot_value_v1(r, cw, f, j);
            if (k == 0)      d0 = __fmaf_rn(v, xb[j], d0);
            else if (k == 1) d1 = __fmaf_rn(v, xb[j], d1);
            else if (k == 2) d2 = __fmaf_rn(v, xb[j], d2);
            else             d3 = __fmaf_rn(v, xb[j], d3);
        }
    }
    return ((d0 + d1) + (d2 + d3)) * gscale[f.gain];
}

extern "C" __global__ void tv_golay70_v1(const u32* __restrict__ words,
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
                acc += golay70_dot_v1(words, cwtab, gtab, gscale, b0r + j,
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
