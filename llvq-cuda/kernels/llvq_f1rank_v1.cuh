// V1 of the F1 universal-table decoder — "no I2F": the same word, the same
// 16 KiB table, the same three trellis tables and the same three row reads as
// `llvq_f1rank.cuh`, but the 24 coordinates are built as FLOATS without ever
// touching the int→float pipe, and the sign is resolved eight lanes at a time.
//
// The reference is unchanged: `llvq_bench::f1::rank::decode_word`.
// `tests/f1rank_v1_matches_rust.rs` compiles this text as host C++ through
// `tests/host_shim.h` and requires `f1r_decode_v1_f` to return, bit for bit,
// `(float)decode_word` on 10,000 random words; the per-block product is
// checked against an f64 sum, and the chained form against the very f32 FMA
// chain `tv_f1r` runs, bit for bit.
//
// ## Why the value construction, and nothing else, changes
//
// The floor of 2026-09-05 put ~2.0 ms of `tv_f1r`'s 2.7 ms decode in
// ARITHMETIC (`docs/mesures/f1-rang-plancher-2026-09-05.txt`). Per block the
// original pays, per coordinate, a chain of selects over `(o, ρ)` — about 14
// ALU instructions — then `(float)yv[s]`, an I2F that issues at 1/8 of the
// FFMA rate on sm_89 (24 of them per block). This file keeps the tables and
// the decode structure (fields, six loads, the N0 split, `r3`) and replaces
// only what happens after the three rows are in registers.
//
// ## The value as one signed formula
//
// Read `val(o, ρ)` of the reference with `o = p + 2c`, `p` shared by the block:
//
//     u = 2ρ + p          a = 2 − 2p          s = (ρ ⊕ c ⊕ p) & 1
//     v = s ? (u + a) : −u
//
// Against the four progressions: p=0,c=0 → 0, +4, −4, +8, −8 (ρ=0: s=0, −0;
// ρ=1: s=1, 2+2); p=0,c=1 → +2, −2, +6, −6, +10 (ρ=0: s=1, 0+2); p=1,c=0 →
// +1, −3, +5, −7, +9 (ρ=0: s=1, u=1); p=1,c=1 → −1, +3, −5, +7, −9 (ρ=0: s=0,
// −1). The formula is exact for EVERY nibble value 0..15, not only the ranks
// 0..4 the table holds — `f1r_section_v1_f` is checked over all sixteen.
//
// ## Byte lanes: eight coordinates in two registers
//
// A section's eight ranks are the eight nibbles of `row`. They are split into
// two registers of four byte lanes — `ga` the even coordinates 0,2,4,6 in
// lanes 0..3, `gb` the odd ones 1,3,5,7 — and everything up to the signed
// value is computed on all four lanes of a register at once:
//
//     r2   = 2ρ per lane          (row << 1) & 0x1e1e1e1e  /  (row >> 3) & 0x1e1e1e1e
//     t1   = r2 + (K + 2 − p)     the `u + a + K` branch     (K a bias, below)
//     t2   = (K − p) − r2         the `K − u` branch
//     s    = ρ ⊕ c ⊕ p at bit 1   (row << 1  or  row >> 3) ⊕ cp, masked 0x02020202
//     mask = s · 0x7f             0xfe per lane where s = 1
//     g    = mask ? t1 : t2       one LOP3, and g = K + v in every lane
//
// The mux mask covers bits 1..7 only. That is enough: bit 0 of `t1` and of
// `t2` is `(K + p) & 1` in both branches, because `r2` and `a` are even.
// `K = 64` keeps both branches inside a byte with no borrow and no carry for
// any nibble: `t2 ≥ 64 − 1 − 30 = 33`, `t1 ≤ 30 + 66 = 96`. Neither lane
// arithmetic crosses a byte, so the four lanes never see each other. (At
// this K bit 7 is zero in both branches too, so a mask of 0x3f · s would
// decide the same — a mutation the harness cannot kill, by construction;
// 0x7f is kept so the mux stays right for any K up to 222.)
//
// `cp` puts `c_j ⊕ p` at bit 1 of lane j/2. With `c' = c ⊕ (p ? 0xff : 0)`:
//
//     even j : ((c' & 0x55) · 0x82082) & 0x02020202      bit 2k → bit 8k + 1
//     odd  j : ((c' & 0xaa) · 0x41041) & 0x02020202      bit 2k+1 → bit 8k + 1
//
// `0x41041 = 1 + 2⁶ + 2¹² + 2¹⁸` and `0x82082` is its double: bit i of the
// masked byte lands at i + 6k' (odd half) or i + 1 + 6k' (even half) for
// k' = 0..3 — the same thirteen odd positions 1, 3, …, 25 in both halves, and
// the wanted landing 8k + 1 has exactly one product per lane. Two products
// meet at bits 7, 13 and 19 (c'₀ with c'₆ in the even half, c'₁ with c'₇ in
// the odd) and carry one bit up, into 8, 14 and 20, where no product lands;
// the carry stops there, below the next lane's bit 1, which stays single.
// The masks 0x55 / 0xaa are what keeps every column at most two deep:
// without them a column of three would carry twice and reach a wanted bit.
// `f1r_section_v1_f` is run on all 256 pattern bytes, both parities, so this
// argument is also checked, not only made.
//
// ## From a byte lane to a float, on the FMA pipe
//
// For 0 ≤ m < 2²³, `__uint_as_float(0x4b000000 | m)` is exactly 2²³ + m, and
// one FADD of −2²³ gives `(float)m` with no I2F. Here the lane already holds
// `K + v` with `v` signed, so the constant is `0x4b400000` and the FADD
// subtracts `2²³ + 2²² + K = 12582976.0f`:
//
//     y = __uint_as_float(0x4b400000 | lane) − 12582976.0f        exact, ±0 → +0.0f
//
// The FADD is exact (both operands and the result are integers below 2²⁴),
// so `y` is bit for bit the float `tv_f1r` obtains from `(float)yv[s]`, and
// the chained product `acc = __fmaf_rn(y, x, acc)` in the same order IS
// `tv_f1r`'s arithmetic — the bench's equality control is met with Δ = 0,
// not within a tolerance.
//
// ## Instruction budget, per block (read off this text, sm_89 pipes)
//
//     fields, indices, six loads        ~25 ALU + 6 LD      unchanged
//     per section (×3)                  ~21 ALU             c' (1), cp ×2 (3+3),
//                                                           row shifts (2), r2 ×2 (2),
//                                                           t1/t2 ×2 (4), s ×2 (2),
//                                                           mask ×2 (2 IMAD), mux ×2 (2)
//     per coordinate (×24)              1–2 ALU             byte extract + OR of the
//                                                           bias (a PRMT if ptxas folds it)
//                                       1 FADD + 1 FFMA     FMA pipe, full rate
//     total                             ≈ 130 ALU, 48 FMA-pipe, 0 I2F
//
// against ≈ 24 × 14 ALU + 24 I2F + 24 FFMA before. The IMADs (mask, cp) issue
// on the FMA-heavy datapath, not the INT one, which is the datapath the
// original saturates. Expected registers: ~40 to 48 (six lane registers
// replace 24 `i8` values), 0 local bytes — `g[6]` is indexed by constants
// under a full unroll, the same contract as `yv[24]` in the original.
//
// Host-compilable on purpose: shifts, masks, `?:`, `__uint_as_float` — all in
// `host_shim.h`; no `__byte_perm`, no funnel shift.

#ifndef LLVQ_F1RANK_V1_CUH
#define LLVQ_F1RANK_V1_CUH

#ifndef LLVQ_F1RANK_CUH
#include "llvq_f1rank.cuh"
#endif

// The lane bias: every byte lane holds K + v, K − 31 ≥ 0 and K + 33 ≤ 255.
#define F1R_V1_K 64u
// 0x4b400000 = 2²³ + 2²², the float whose low 22 mantissa bits receive the lane.
#define F1R_V1_BIAS 0x4b400000u
// (float)(F1R_V1_BIAS) + K: 12582912 + 64, exactly representable.
#define F1R_V1_FSUB 12582976.0f

// What every section of a block shares: the parity folded into the pattern
// byte, and the two branch constants replicated on the four lanes.
struct F1rV1Block {
    u32 pff;   // p ? 0xff : 0
    u32 kp;    // (K + 2 − p) · 0x01010101
    u32 km;    // (K − p) · 0x01010101
};

__device__ __forceinline__ F1rV1Block f1r_v1_block(u32 p)
{
    F1rV1Block b;
    b.pff = p * 0xffu;
    b.kp  = (F1R_V1_K + 2u - p) * 0x01010101u;
    b.km  = (F1R_V1_K - p) * 0x01010101u;
    return b;
}

// One section into two lane registers: `ga` the even coordinates (0,2,4,6 in
// lanes 0..3), `gb` the odd ones (1,3,5,7). Every lane holds K + v.
__device__ __forceinline__ void f1r_v1_section(const F1rV1Block& b, u32 c, u32 row, u32& ga, u32& gb)
{
    u32 cp  = c ^ b.pff;
    u32 cpa = ((cp & 0x55u) * 0x82082u) & 0x02020202u;
    u32 cpb = ((cp & 0xaau) * 0x41041u) & 0x02020202u;
    u32 rowa = row << 1;   // ρ_{2k}   at lane k, bits 1..4
    u32 rowb = row >> 3;   // ρ_{2k+1} at lane k, bits 1..4
    u32 r2a = rowa & 0x1e1e1e1eu;
    u32 r2b = rowb & 0x1e1e1e1eu;
    u32 sa  = (rowa ^ cpa) & 0x02020202u;
    u32 sb  = (rowb ^ cpb) & 0x02020202u;
    u32 ma  = sa * 0x7fu;
    u32 mb  = sb * 0x7fu;
    u32 t1a = r2a + b.kp, t2a = b.km - r2a;
    u32 t1b = r2b + b.kp, t2b = b.km - r2b;
    ga = t2a ^ ((t1a ^ t2a) & ma);
    gb = t2b ^ ((t1b ^ t2b) & mb);
}

// Lane `k` of a lane register, as the float it encodes. `k` is a constant at
// every call site, so the shift and the mask fold; ptxas may fuse the
// extract and the OR into one PRMT.
__device__ __forceinline__ float f1r_v1_lane(u32 g, u32 k)
{
    u32 bits = F1R_V1_BIAS | ((g >> (8u * k)) & 0xffu);
    return __uint_as_float(bits) - F1R_V1_FSUB;
}

// The fields, the six loads and the three sections into `g[0..6]`:
// `g[2·sec]` the even coordinates of section `sec`, `g[2·sec + 1]` the odd.
//
// The first half is `f1r_decode`'s, line for line — the same fields, the
// same dependent chain `s8 → br → s16 → c3`, the same N0 split, the same
// `r3`, the same mask on `s16` (a corrupted upload must not read past the
// 128 suffix bytes). Copied, not shared, because `f1r_decode` returns `i8`
// and this path never materialises an integer coordinate; the harness
// compares both to the same reference, so the copy cannot drift unnoticed.
__device__ __forceinline__ void f1r_v1_lanes(u32 lo, u32 hi16, const F1rTables& t, u32 g[6])
{
    u32 p  = lo & 1u;
    u32 r  = (lo >> 1) & 1u;
    u32 s8 = (lo >> 2) & 63u;
    u32 b1 = (lo >> 8) & 1u;
    u32 i1 = (lo >> 9) & 0x7ffu;
    u32 b2 = (lo >> 20) & 15u;
    u32 i2 = ((lo >> 24) | ((hi16 & 7u) << 8)) & 0x7ffu;
    u32 b3 = (hi16 >> 3) & 1u;
    u32 i3 = (hi16 >> 4) & 0x7ffu;

    u32 c1  = t.prefixes[2u * s8 + b1];
    u32 br  = t.branches[16u * s8 + b2];
    u32 c2  = br & 0xffu;
    u32 s16 = (br >> 8) & 63u;
    u32 c3  = t.suffixes[2u * s16 + b3];

    u32 row1 = t.rows[2048u * r + i1];
    bool mid  = i2 < F1R_N0_MIXED;
    u32 idx2  = mid ? i2 : (2048u - F1R_N0_MIXED) + i2;
    u32 row2  = t.rows[idx2];
    u32 delta = mid ? 0u : 1u;
    u32 r3    = (p ^ r ^ delta) & 1u;
    u32 row3  = t.rows[2048u * r3 + i3];

    F1rV1Block b = f1r_v1_block(p);
    f1r_v1_section(b, c1, row1, g[0], g[1]);
    f1r_v1_section(b, c2, row2, g[2], g[3]);
    f1r_v1_section(b, c3, row3, g[4], g[5]);
}

// Coordinate `s` of a block (trio order) out of the six lane registers:
// section s >> 3, coordinate j = s & 7, register 2·sec + (j & 1), lane j >> 1.
// Only ever called with a constant `s` under `#pragma unroll`.
__device__ __forceinline__ float f1r_v1_coord(const u32 g[6], u32 s)
{
    u32 j = s & 7u;
    return f1r_v1_lane(g[2u * (s >> 3) + (j & 1u)], j >> 1);
}

// The 24 coordinates as floats, for the host check: bit for bit
// `(float)decode_word` (exactly representable small integers, +0.0f at 0).
__device__ __forceinline__ void f1r_decode_v1_f(u32 lo, u32 hi16, const F1rTables& t, float y[24])
{
    u32 g[6];
    f1r_v1_lanes(lo, hi16, t, g);
#pragma unroll
    for (u32 s = 0; s < LLVQ_DIM; ++s) y[s] = f1r_v1_coord(g, s);
}

// One section on its own, for the exhaustive host check over every pattern
// byte, both parities and every nibble value — the argument about carries in
// the spreading products, run rather than trusted.
__device__ __forceinline__ void f1r_section_v1_f(u32 p, u32 c, u32 row, float y[8])
{
    F1rV1Block b = f1r_v1_block(p & 1u);
    u32 ga, gb;
    f1r_v1_section(b, c & 0xffu, row, ga, gb);
#pragma unroll
    for (u32 k = 0; k < 4u; ++k) {
        y[2u * k]      = f1r_v1_lane(ga, k);
        y[2u * k + 1u] = f1r_v1_lane(gb, k);
    }
}

// The block's 24 products chained into `acc`, coordinate 0 first — the order
// of `tv_f1r`'s loop, `acc = __fmaf_rn((float)yv[s], xb[s], acc)`, so a row
// accumulated through this function is bit for bit the row `tv_f1r` writes.
__device__ __forceinline__ float f1r_dot_v1_acc(u32 lo, u32 hi16, const F1rTables& t, const float* xb, float acc)
{
    u32 g[6];
    f1r_v1_lanes(lo, hi16, t, g);
#pragma unroll
    for (u32 s = 0; s < LLVQ_DIM; ++s) acc = __fmaf_rn(f1r_v1_coord(g, s), xb[s], acc);
    return acc;
}

// The per-block product Σ_s y_s · xb[s], from zero, the same chain.
__device__ __forceinline__ float f1r_dot_v1(u32 lo, u32 hi16, const F1rTables& t, const float* xb)
{
    return f1r_dot_v1_acc(lo, hi16, t, xb, 0.0f);
}

#endif  // LLVQ_F1RANK_V1_CUH
