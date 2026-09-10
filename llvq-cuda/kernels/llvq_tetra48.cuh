// The served Tetra decode: `f1r_dot_v3` plus the four things it leaves out.
//
// `tv_f1r_v3` (`f1rank_v3.cu`) is already a whole matvec — one warp per row,
// 128-block tiles in shared memory, `warp_sum`, the row scale and the f32
// tail. What it is *not* is a served kernel, and the gap is exactly four
// items, none of them structural:
//
//   1. **the gain bit**, bit 47 of the word. `llvq_f1rank.cuh:143` says it in
//      as many words: *"the gain bit. Not read here."* The floor applies no
//      scale at all, so it measures a decode's speed and not a model's output;
//   2. **the magnitude**. `reconstruct_shape_gain`
//      (`llvq-quant/src/quantizer.rs:925`) is `y_j · centroids[g] · rscale /
//      √(16 m)` with `m` the shell index. Since `‖y‖² = 16 m` by definition of
//      the shell, that divisor *is* `‖y‖`: the point is normalised, then
//      scaled. The floor skips it;
//   3. **the trio permutation**. `f1r_dot_v3` dots in trio order; the
//      activation staged in `xs` is in natural order. Dotting the two
//      together is a silent wrong answer, not a crash — and the floor's own
//      check (`f1rankfloor.rs`) compares trio against trio, so it structurally
//      cannot see the bug;
//   4. **the origin**. Word 0 is a legal code — `quantizer.rs:770` writes
//      `BlockCode { point: [0; DIM], gain: 0 }` for a zero block, and
//      `reconstruct_shape_gain` returns zeros for it. `1/‖y‖` is a division by
//      zero there. It is handled by the table's entry 0, not by a branch.
//
// **The shell bound is 27**, not a guess: `llvq-bench/examples/tetrashell.rs`
// sweeps all 4,096 rows of the table against both parities and both residues
// and reports `max |y_j| = 10`, worst section `Σy² = 144`, hence `n2 ≤ 432`
// and `m ≤ 27`. So the inverse-norm table is **32 floats, 128 bytes** — small
// enough that it is not a table in any interesting sense, and the whole
// magnitude costs six `__dp4a` and one load.
//
// The header adds no global memory traffic over v3 beyond those two tiny
// arrays, and no dynamic indexing: `q` stays in registers under the same
// `#pragma unroll` rule as `llvq_f1rank.cuh` §"No dynamic indexing".

#ifndef LLVQ_TETRA48_CUH
#define LLVQ_TETRA48_CUH

// Guarded, and the guard is not decoration: NVRTC has no filesystem, so the
// host concatenates the parts and an unconditional `#include` is a
// **catastrophic error** on the card while `clang++` — which does have one —
// resolves it happily and says nothing. That is exactly how this file reached
// a billed job on 2026-09-08 and died in NVRTC at line 2043.
#ifndef LLVQ_F1RANK_V3_CUH
#include "llvq_f1rank_v3.cuh"
#endif

// Entries of the inverse-norm table: `m` runs 0..27 on this codebook and 32 is
// that bound rounded up. The host builder asserts the real maximum rather than
// trusting this constant, and `tetra48_dot` masks with it so that a corrupt
// word reads a wrong scale instead of reading off the end of the table.
#define TETRA48_SHELLS 32u

// Trio position `k` carries natural coordinate `TETRA48_ORDER[k]`, so the dot
// `Σ_j y_nat[j]·x[j]` is `Σ_k y_trio[k]·x[ORDER[k]]`. Every index is constant
// inside the two unrolled loops below, so this costs no instruction: it is an
// address computed at compile time, not a gather. Derived from
// `llvq_search::tetra::Tetra::order()` and asserted against it host-side —
// a permutation that drifts from the encoder is a wrong model, not a slow one.
static constexpr unsigned char TETRA48_ORDER[24] = {
    0,  1,  2,  3,  4,  7,  10, 12, 6,  11, 13, 14,
    16, 17, 18, 19, 5,  8,  9,  15, 20, 21, 22, 23,
};

// `‖y‖²` of a block, from the six packed byte-quads `f1r_v3_quads` produced.
//
// Those bytes are `val(o, ρ) + 128` (`llvq_f1rank_v3.cuh:89-96`). XOR by
// `0x80` maps that back onto the signed byte `val` exactly — for `b < 128`,
// `b ^ 0x80 = b + 128` reads as `b − 128` in two's complement; for `b ≥ 128`
// it is `b − 128` outright — so `__dp4a` of the biased quad with itself,
// signed, accumulates `Σ val²` with no conversion and no unpacking. Six
// instructions plus six XOR, against 24 sign-extensions and 24 multiplies.
//
// The result is exact and small: `|val| ≤ 10` over the whole table, so
// `n2 ≤ 432` and the `int` accumulator cannot overflow.
__device__ __forceinline__ u32 tetra48_n2(const u32 q[6])
{
    int acc = 0;
#pragma unroll
    for (u32 i = 0; i < 6u; ++i) {
        int s = (int) (q[i] ^ 0x80808080u);
        acc = __dp4a(s, s, acc);
    }
    return (u32) acc;
}

// The served block dot: `(Σ_j y_nat[j]·xb[j]) · centroids[g] / √(16 m)`.
//
// The two scales are folded **per block**, inside the accumulation the caller
// runs over a row — the same shape as `planes_dot`, and for the same reason:
// the gain is a property of the block, the row scale a property of the row,
// and only the second may be folded once at the end.
//
// `invnorm[m]` is `1/√(16 m)` for `m > 0` and **`0` for `m = 0`**, which is
// how the origin is reconstructed as a zero block without a branch and
// without a division. That entry is the whole handling of item 4 above.
__device__ __forceinline__ float tetra48_dot(u32 lo,
                                             u32 hi16,
                                             const F1rTables& t,
                                             const float* __restrict__ xb,
                                             const float* __restrict__ gscale,
                                             const float* __restrict__ invnorm)
{
    u32 q[6];
    f1r_v3_quads(lo, hi16, t, q);

    float acc = 0.0f;
#pragma unroll
    for (u32 i = 0; i < 6u; ++i) {
#pragma unroll
        for (u32 j = 0; j < 4u; ++j) {
            acc = __fmaf_rn(f1r_v3_float(q[i], j), xb[TETRA48_ORDER[4u * i + j]], acc);
        }
    }

    // `‖y‖² = 16 m` exactly, by the definition of the shell index, so the
    // shift is not a rounding — a remainder here would mean the decode left
    // the lattice. The host probe asserts `n2 % 16 == 0` on every word it
    // sweeps; the mask is the runtime's cheap guard against a corrupt read.
    u32 m = (tetra48_n2(q) >> 4) & (TETRA48_SHELLS - 1u);
    u32 g = (hi16 >> 15) & 1u;
    return acc * gscale[g] * invnorm[m];
}

// The same block dot, against R activation rows at once.
//
// ## Why it exists
//
// `tetra48_dot` reads ONE activation row, so a prompt of N tokens decodes the
// whole weight stream N times: 0.97 GB re-read per token on the served 4B, and
// one launch per row per matrix. A 5-shot MMLU question is several hundred
// tokens, which is both 200,000 launches and 776 GB of reads — and the host
// refuses past 256 rows outright rather than look like a hang.
//
// The weight is what costs. The activation is 96 bytes a block. So the word is
// decoded ONCE and applied to R rows, and the stream is read R times less.
// This is what every quantized-inference stack does for its prefill, and the
// reason they all carry two kernels: one row for decoding a token, many rows
// for swallowing a prompt.
//
// ## What must not change, and how it is held
//
// **The result is bit-identical to R calls of `tetra48_dot`, and that is a
// requirement rather than a consequence.** Two things would break it and both
// are one keystroke away:
//
//   * the 24 FMAs of a row must keep their order. They do: the same two
//     unrolled loops, the same `TETRA48_ORDER`, per row.
//   * the two scales must keep their association. `tetra48_dot` returns
//     `acc * gscale[g] * invnorm[m]`, which is `(acc · g) · inv` — NOT
//     `acc · (g · inv)`. Hoisting `gscale[g] * invnorm[m]` out of the row loop
//     would be the obvious optimisation and it is wrong: the two differ by one
//     ULP on roughly one word in a thousand. The repository has already paid
//     for that association once, in a test that modelled it the other way.
//
// `tests/host_tetra48.cpp` runs both routes on the same fixture and the Rust
// side compares them bit for bit, so this is checked on a machine with no
// card rather than asserted in this comment.
//
// `row_stride` is in floats and separates the rows inside the caller's shared
// staging; the caller owns that layout.
template <unsigned R>
__device__ __forceinline__ void tetra48_dot_rows(u32 lo,
                                                 u32 hi16,
                                                 const F1rTables& t,
                                                 const float* __restrict__ xb,
                                                 u32 row_stride,
                                                 const float* __restrict__ gscale,
                                                 const float* __restrict__ invnorm,
                                                 float* acc)
{
    // Decoded once. This is the whole point: the six quads, the shell sum and
    // the gain bit do not depend on the activation.
    u32 q[6];
    f1r_v3_quads(lo, hi16, t, q);
    u32 m = (tetra48_n2(q) >> 4) & (TETRA48_SHELLS - 1u);
    u32 g = (hi16 >> 15) & 1u;
    const float gs = gscale[g];
    const float iv = invnorm[m];

#pragma unroll
    for (unsigned r = 0; r < R; ++r) {
        const float* __restrict__ xr = xb + r * row_stride;
        float a = 0.0f;
#pragma unroll
        for (u32 i = 0; i < 6u; ++i) {
#pragma unroll
            for (u32 j = 0; j < 4u; ++j) {
                a = __fmaf_rn(f1r_v3_float(q[i], j), xr[TETRA48_ORDER[4u * i + j]], a);
            }
        }
        // `(a · gs) · iv`, left to right, exactly as `tetra48_dot` returns it.
        acc[r] += a * gs * iv;
    }
}

// The 24 reconstructed values in **natural** order, for the host check.
//
// The same bytes, the same float construction and the same two scales as the
// dot, without the FMA chain — so a disagreement between this and the Rust
// reference is a decode bug, and a disagreement between the dot and the sum
// of these is an association difference. Keeping the two separable is what
// makes the host probe able to say which.
__device__ __forceinline__ void tetra48_decode_f(u32 lo,
                                                 u32 hi16,
                                                 const F1rTables& t,
                                                 const float* __restrict__ gscale,
                                                 const float* __restrict__ invnorm,
                                                 float y[24])
{
    u32 q[6];
    f1r_v3_quads(lo, hi16, t, q);
    u32 m = (tetra48_n2(q) >> 4) & (TETRA48_SHELLS - 1u);
    u32 g = (hi16 >> 15) & 1u;
    float s = gscale[g] * invnorm[m];
#pragma unroll
    for (u32 i = 0; i < 6u; ++i) {
#pragma unroll
        for (u32 j = 0; j < 4u; ++j) {
            y[TETRA48_ORDER[4u * i + j]] = f1r_v3_float(q[i], j) * s;
        }
    }
}

#endif
