// V3 of the F1 universal-table decoder: the 24 values by byte tables in
// registers, moved into place by `prmt` — no int→float conversion, no
// per-coordinate select chain.
//
// `llvq_f1rank.cuh` builds each coordinate as an integer: extract the rank
// nibble, extract the pattern bit, `f1r_val(o, ρ)` as a chain of selects and
// a negation, store an `i8`, then `(float)` it at the FMA — 24 int→float
// conversions per block, on a pipe that issues at a fraction of the FFMA
// rate on sm_89. This file reads the SAME word, the SAME three trellis bytes
// and the SAME three 16 KiB-table rows, and replaces everything after the
// loads. `tests/f1rank_v3_matches_rust.rs` diffs the 24 floats bit for bit
// against `(float)llvq_bench::f1::rank::decode_word` on the development
// machine; the arm `tv_f1r_v3` (`f1rank_v3.cu`) is compared to `tv_f1r` on
// the card by `bin/f1rankfloor`.
//
// ## The idea: a value is one byte, and `prmt` is an 8-entry byte table
//
// `prmt.b32 d, a, b, s` (`__byte_perm`) fills each of the 4 bytes of `d`
// with one of the 8 bytes of `{b, a}`, the choice being the low 3 bits of
// the matching control nibble of `s`. So a pair of registers is an 8-entry
// table of bytes, and one `prmt` performs FOUR lookups in it. The table row
// is already the selector: rank `j` sits at bits `4j..4j+3`, the ranks of
// coordinates 0..3 are the low 16 bits and those of 4..7 the high 16 —
// `row` and `row >> 16` are the two control words; the ISA reads only
// `s[15:0]` and the intrinsic ANDs the selector with 0x7777 on the way (one
// LOP the compiler inserts, harmless here: ranks are ≤ 4, bit 3 is clear). Ranks are ≤ 4 on this
// table (`MAX_RANK`, asserted by the builder), so bit 3 of every nibble is
// clear and no lookup sign-replicates; the tables still carry ranks 5..7,
// which is what the 8 entries are.
//
// The value depends on `o = p + 2·c_j`; `p` is one bit per block, `c_j` one
// bit per coordinate. Per block, `p` selects the pair of tables `(o = p, o =
// p + 2)`; per quad of coordinates, BOTH tables are read (2 `prmt`) and the
// bytes are chosen by a byte mask built from the four pattern bits: one LOP3.
// The mask itself is three instructions: `c4 · 0x00204081` places bit `j`
// of `c4` at bit `8j`, the lsb of byte `j` — the four shifted copies of a
// 4-bit value at 0, 7, 14 and 21 cannot overlap, so no carry — an AND keeps
// those lsbs, and `· 0xff` fills each byte. (A single `prmt` in its
// sign-replication mode would do it, but `__byte_perm` cannot reach that
// mode: see `f1r_v3_bytemask`.)
//
// ## From a byte to the float, without the conversion pipe
//
// The bytes are BIASED: `val(o, ρ) + 128`, in `[112, 144]` for ranks 0..7,
// so the sign is inside the table and costs nothing per coordinate. One
// `prmt` places byte `j` of the quad under `0x4b0000__`, which as a float is
// `2^23 + byte` exactly (the ulp at 2^23 is 1), and one FADD of
// `−(2^23 + 128)` leaves `val(o, ρ)` — an exact small integer, `+0.0f` for
// zero (`x − x` is `+0` in round-to-nearest), so the result is bit for bit
// `(float)` of the reference's integer. Per coordinate: PRMT, FADD, FFMA.
//
// ## Count, per block (read the code; the pipes are sm_86+'s)
//
//     fields + 6 loads + index arithmetic   ~26  ALU/LSU     (as `f1r_decode`)
//     4 table selects by p                    4  ALU (SEL)
//     per section (×3):
//       row >> 16                             1  ALU
//       4 table prmt                          4  ALU (PRMT)
//       c & 0xf, c >> 4                       2  ALU
//       2 × (IMAD, PRMT) masks                2  FMA-pipe (IMAD) + 2 ALU
//       2 LOP3 byte selects                   2  ALU
//       8 × (PRMT, FADD, FFMA)                8  ALU + 16 FMA-pipe
//     ------------------------------------------------------------------
//     ≈ 30 + 3 × 37 = 141 instructions: 54 on the FMA pipe (24 FFMA, 24
//     FADD, 6 IMAD), ~85 on the ALU pipe (42 PRMT, the rest LOP3/SHF/SEL and
//     the address arithmetic), 6 loads, and NO I2F. Against ~380 + 24 I2F
//     for `f1r_decode` + its FMA loop.
//
// The 24 products are summed in the order `tv_f1r` sums them, coordinate 0
// to 23, one `__fmaf_rn` chain — from zero here, so a block's dot joins the
// running sum with one rounding `tv_f1r` does not have; that is inside the
// bench's y-equality control and the harness checks the chain itself bit
// for bit against the same chain in Rust.
//
// Host-compilable under the same rule as `llvq_f1rank.cuh`, plus one
// intrinsic: `__byte_perm`, which `host_shim.h` defines as the ISA's
// `prmt.b32` and the harness checks on 16 hand-computed cases. Includes
// nothing; the host concatenates `llvq_slot.cuh`, `matvec.cu`,
// `llvq_f1rank.cuh` before it, and this file uses `F1rTables` and
// `F1R_N0_MIXED` from there.

#ifndef LLVQ_F1RANK_V3_CUH
#define LLVQ_F1RANK_V3_CUH

// The four progressions as biased bytes, `val(o, ρ) + 128`, rank 0 in the
// low byte of `LO`, rank 4 in the low byte of `HI`. Derived from
// `llvq_bench::f1::rank::val` (the harness reproduces every one of the 20
// entries ranks 0..4 reach, and the 12 beyond on synthetic rows).
#define F1R_V3_T0_LO 0x887c8480u   // o = 0 :  0, +4, −4, +8
#define F1R_V3_T0_HI 0x90748c78u   //         −8, +12, −12, +16
#define F1R_V3_T2_LO 0x7a867e82u   // o = 2 : +2, −2, +6, −6
#define F1R_V3_T2_HI 0x728e768au   //         +10, −10, +14, −14
#define F1R_V3_T1_LO 0x79857d81u   // o = 1 : +1, −3, +5, −7
#define F1R_V3_T1_HI 0x718d7589u   //         +9, −11, +13, −15
#define F1R_V3_T3_LO 0x877b837fu   // o = 3 : −1, +3, −5, +7
#define F1R_V3_T3_HI 0x8f738b77u   //         −9, +11, −13, +15

// `2^23 + 128`: what a biased byte under `0x4b0000__` reads as, minus the
// value. Exactly representable, so the FADD is exact.
#define F1R_V3_BIAS 8388736.0f

// The two tables of a block: `o = p` for pattern bit 0, `o = p + 2` for 1.
struct F1rV3Tab {
    u32 c0lo, c0hi;   // o = p
    u32 c1lo, c1hi;   // o = p + 2
};

__device__ __forceinline__ F1rV3Tab f1r_v3_tables(u32 p)
{
    F1rV3Tab t;
    t.c0lo = p ? F1R_V3_T1_LO : F1R_V3_T0_LO;
    t.c0hi = p ? F1R_V3_T1_HI : F1R_V3_T0_HI;
    t.c1lo = p ? F1R_V3_T3_LO : F1R_V3_T2_LO;
    t.c1hi = p ? F1R_V3_T3_HI : F1R_V3_T2_HI;
    return t;
}

// `0xff` in byte `j` where bit `j` of `c4` is set, `0x00` elsewhere, `j < 4`;
// bits 4..31 of `c4` must be clear. `c4 · 0x00204081` spreads the four bits
// to the four byte lsbs without a carry (copies at shifts 0, 7, 14, 21 of a
// 4-bit value never overlap), the AND keeps them, `· 0xff` fills each byte.
// Two IMAD and one LOP per quad.
//
// ⚠️ NOT the one-`prmt` form. `prmt.b32` has a sign-replication mode (bit 3
// of each control nibble) that would turn a byte's msb into 0x00/0xff in a
// single instruction — but the CUDA intrinsic `__byte_perm` masks the
// selector with 0x7777 before the instruction (a `LOP.AND 0x7777` in the
// SASS whenever the selector is not an immediate; NVIDIA forums #17822 and
// #22447, CUDA.jl issue #1424), so `0xba98` reaches the card as `0x3210`, the
// identity, and every coordinate with `c_j = 1` decodes wrong. Found by the
// adversarial review of 2026-09-05 before any card time. The one-`prmt`
// form is kept under `LLVQ_F1R_V3_SIGN_PRMT` for a build that emits the raw
// PTX itself; it is what the host shim's `__byte_perm` models, and it is
// NOT what the intrinsic does.
__device__ __forceinline__ u32 f1r_v3_bytemask(u32 c4)
{
#ifdef LLVQ_F1R_V3_SIGN_PRMT
    return __byte_perm(c4 * 0x10204080u, 0u, 0xba98u);
#else
    return ((c4 * 0x00204081u) & 0x01010101u) * 0xffu;
#endif
}

// Four biased value bytes: the four coordinates whose rank nibbles are the
// low 16 bits of `sel` and whose pattern bits are the low 4 bits of `c4`.
__device__ __forceinline__ u32 f1r_v3_quad(const F1rV3Tab& t, u32 sel, u32 c4)
{
    u32 m0 = __byte_perm(t.c0lo, t.c0hi, sel);
    u32 m1 = __byte_perm(t.c1lo, t.c1hi, sel);
    u32 k  = f1r_v3_bytemask(c4);
    return (m0 & ~k) | (m1 & k);   // one LOP3
}

// One section: the two quads of a pattern byte `c` and a table row `row`.
// `row` goes to `prmt` whole: the ISA reads `s[15:0]`, the intrinsic masks
// each nibble to 3 bits, and ranks ≤ 4 never set the fourth.
__device__ __forceinline__ void f1r_v3_section(const F1rV3Tab& t, u32 c, u32 row, u32& qa, u32& qb)
{
    qa = f1r_v3_quad(t, row, c & 0xfu);
    qb = f1r_v3_quad(t, row >> 16, c >> 4);
}

// Byte `j` of a quad as the value it encodes. Control `0x744j`: byte 0 from
// the quad, bytes 1 and 2 the low byte of `0x4b000000` (zero), byte 3 its
// top byte. `j` is a constant after unrolling, so the control is an
// immediate.
__device__ __forceinline__ float f1r_v3_float(u32 quad, u32 j)
{
    return __uint_as_float(__byte_perm(quad, 0x4b000000u, 0x7440u | j)) - F1R_V3_BIAS;
}

// What a block's loads yield: `p`, the three pattern bytes, the three rows.
struct F1rV3Block {
    u32 p;
    u32 c1, c2, c3;
    u32 row1, row2, row3;
};

// The fields and the six loads, exactly as `f1r_decode` performs them —
// the same chain `s8 → br → s16 → c3`, the same `N0` split, the same `r3`.
__device__ __forceinline__ F1rV3Block f1r_v3_fetch(u32 lo, u32 hi16, const F1rTables& t)
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

    F1rV3Block b;
    b.p  = p;
    b.c1 = t.prefixes[2u * s8 + b1];
    u32 br  = t.branches[16u * s8 + b2];
    b.c2 = br & 0xffu;
    u32 s16 = (br >> 8) & 63u;
    b.c3 = t.suffixes[2u * s16 + b3];

    b.row1 = t.rows[2048u * r + i1];
    bool mid  = i2 < F1R_N0_MIXED;
    u32 idx2  = mid ? i2 : (2048u - F1R_N0_MIXED) + i2;
    b.row2 = t.rows[idx2];
    u32 delta = mid ? 0u : 1u;
    u32 r3    = (p ^ r ^ delta) & 1u;
    b.row3 = t.rows[2048u * r3 + i3];
    return b;
}

// The six quads of a block, in coordinate order: section 1 low, high, …
__device__ __forceinline__ void f1r_v3_quads(u32 lo, u32 hi16, const F1rTables& t, u32 q[6])
{
    F1rV3Block b = f1r_v3_fetch(lo, hi16, t);
    F1rV3Tab   v = f1r_v3_tables(b.p);
    f1r_v3_section(v, b.c1, b.row1, q[0], q[1]);
    f1r_v3_section(v, b.c2, b.row2, q[2], q[3]);
    f1r_v3_section(v, b.c3, b.row3, q[4], q[5]);
}

// The block's dot product, `Σ_s y_s · xb[s]`, one `__fmaf_rn` chain in
// coordinate order from zero. Fully unrolled with constant indices, so `q`
// stays in registers (the rule of `llvq_f1rank.cuh` §"No dynamic indexing").
__device__ __forceinline__ float f1r_dot_v3(u32 lo, u32 hi16, const F1rTables& t, const float* xb)
{
    u32 q[6];
    f1r_v3_quads(lo, hi16, t, q);
    float acc = 0.0f;
#pragma unroll
    for (u32 i = 0; i < 6u; ++i) {
#pragma unroll
        for (u32 j = 0; j < 4u; ++j) acc = __fmaf_rn(f1r_v3_float(q[i], j), xb[4u * i + j], acc);
    }
    return acc;
}

// The 24 values as floats, for the host check — the same bytes and the same
// float construction as the dot, without the FMA.
__device__ __forceinline__ void f1r_decode_v3_f(u32 lo, u32 hi16, const F1rTables& t, float y[24])
{
    u32 q[6];
    f1r_v3_quads(lo, hi16, t, q);
#pragma unroll
    for (u32 i = 0; i < 6u; ++i) {
#pragma unroll
        for (u32 j = 0; j < 4u; ++j) y[4u * i + j] = f1r_v3_float(q[i], j);
    }
}

#endif  // LLVQ_F1RANK_V3_CUH
