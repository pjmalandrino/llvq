// V2 of the F1 universal-table decoder: the trellis by F₂ algebra, no
// dependent loads.
//
// `llvq_f1rank.cuh` resolves the three pattern bytes of a block through a
// chain of three small-table reads, each addressed by the previous one:
// `prefixes[2·s8 + b1]`, then `branches[16·s8 + b2]` giving `(c2, s16)`, then
// `suffixes[2·s16 + b3]` — three L1 latencies in series per block, after the
// word is in hand. This file replaces the chain by twelve masked XORs and
// reads nothing but the 16 KiB rank rows. Same word, same rows, same
// `f1r_val`, same 24 values: `tests/f1rank_v2_matches_rust.rs` diffs the
// output bit for bit against `llvq_bench::f1::rank::decode_word` on the
// development machine, and the arm `tv_f1r_v2` (`f1rank_v2.cu`) is compared
// to `tv_f1r` on the card by `bin/f1rankfloor`.
//
// ## Why algebra is possible, and under WHICH numbering
//
// The three maps are linear in the Golay cosets by construction; the question
// was whether they are linear in the NUMBERS the word stores, since
// `Trellis::new` numbers the 64 states at each cut by SORTING the smallest
// member of every coset, and the 16 branches of a state by sorting their
// middle bytes. They are — decided by code, not by argument
// (`llvq-bench/examples/f1linear.rs`, 2026-09-05, and pinned by the tests of
// `llvq_bench::f1::rank::TrellisLinear`): on the whole domain, 128 + 1,024 +
// 128 inputs, each map is `M·x` with a ZERO constant, and their composition
// `(s8, b1, b2, b3) ↦ (c1, c2, c3)` is linear on all 64 × 2 × 16 × 2 paths.
// The reason, for the reader: the smallest member of a coset of a subspace V
// is its member with zeros at V's pivot bits, which is a linear reduction, so
// the 64 representatives form a subspace, and a subspace listed in numeric
// order is enumerated by a linear coordinate (index bit `i` ↔ the reduced
// basis vector with the i-th lowest pivot). The sixteen middle bytes of a
// state are a coset of one fixed 4-dimensional subspace (`examples/f1mids.rs`),
// so the same argument gives the branch index. `s16` is composed away.
//
// **The word's bits keep their meaning.** No relabelling, no change to the
// format: this header reads the word `llvq_f1rank.cuh` reads.
//
// ## The map
//
// Input `x`, twelve bits in the order the word lays them: `s8` (bits 2..7 of
// `lo`), `b1` (bit 8), `b2` (bits 20..23), `b3` (bit 3 of `hi16`). Output
// `cc = c1 | c2 << 8 | c3 << 16 = ⊕_i x_i · COL_i`. The twelve columns are
// `llvq_bench::f1::rank::LINEAR_COLUMNS`, verbatim; the host harness writes
// the twelve it compiled with and the Rust test compares them to the derived
// ones, so a mistyped immediate fails before any card. `b1` and `b3` are the
// complementary prefix and suffix: their columns are the all-ones bytes.
//
// Per input bit: an all-ones mask from the bit (`0 − (bit)`: SHF, LOP3,
// IADD — or two shifts if the compiler prefers), then one LOP3 for
// `cc ^= mask & COL`. Twelve bits, no memory. The three row reads stay, and
// they depend only on the word; `row3`'s address depends on `δ` and `r3`,
// computed, as before.
//
// Host-compilable under the same rule as `llvq_f1rank.cuh`: plain shifts,
// masks and `?:`, nothing `host_shim.h` does not define. Includes nothing; the
// host concatenates `llvq_slot.cuh`, `matvec.cu`, `llvq_f1rank.cuh` before it,
// and this file uses `F1rTables`, `F1R_N0_MIXED` and `f1r_val` from there.

#ifndef LLVQ_F1RANK_V2_CUH
#define LLVQ_F1RANK_V2_CUH

// The twelve columns, `c1 | c2 << 8 | c3 << 16` per input bit. Mirrors
// `llvq_bench::f1::rank::LINEAR_COLUMNS` (a test compares them).
#define F1R_V2_COL_S8_0 0x2d002eu
#define F1R_V2_COL_S8_1 0x3a005au
#define F1R_V2_COL_S8_2 0x740033u
#define F1R_V2_COL_S8_3 0x03061eu
#define F1R_V2_COL_S8_4 0x050963u
#define F1R_V2_COL_S8_5 0x090578u
#define F1R_V2_COL_B1   0x0000ffu
#define F1R_V2_COL_B2_0 0x2d1d00u
#define F1R_V2_COL_B2_1 0x3a2b00u
#define F1R_V2_COL_B2_2 0x744700u
#define F1R_V2_COL_B2_3 0x638e00u
#define F1R_V2_COL_B3   0xff0000u

// All ones when bit `bit` of `w` is set, zero otherwise. Unsigned throughout,
// so it is defined on every compiler; the device has no compare and no select
// in it.
__device__ __forceinline__ u32 f1r_v2_fill(u32 w, u32 bit)
{
    return 0u - ((w >> bit) & 1u);
}

// The three pattern bytes of the word, packed `c1 | c2 << 8 | c3 << 16`, by
// the linear map above. Twelve masked XORs, no load.
__device__ __forceinline__ u32 f1r_v2_patterns(u32 lo, u32 hi16)
{
    u32 cc = 0u;
    cc ^= f1r_v2_fill(lo, 2u) & F1R_V2_COL_S8_0;
    cc ^= f1r_v2_fill(lo, 3u) & F1R_V2_COL_S8_1;
    cc ^= f1r_v2_fill(lo, 4u) & F1R_V2_COL_S8_2;
    cc ^= f1r_v2_fill(lo, 5u) & F1R_V2_COL_S8_3;
    cc ^= f1r_v2_fill(lo, 6u) & F1R_V2_COL_S8_4;
    cc ^= f1r_v2_fill(lo, 7u) & F1R_V2_COL_S8_5;
    cc ^= f1r_v2_fill(lo, 8u) & F1R_V2_COL_B1;
    cc ^= f1r_v2_fill(lo, 20u) & F1R_V2_COL_B2_0;
    cc ^= f1r_v2_fill(lo, 21u) & F1R_V2_COL_B2_1;
    cc ^= f1r_v2_fill(lo, 22u) & F1R_V2_COL_B2_2;
    cc ^= f1r_v2_fill(lo, 23u) & F1R_V2_COL_B2_3;
    cc ^= f1r_v2_fill(hi16, 3u) & F1R_V2_COL_B3;
    return cc;
}

// One section into floats: pattern byte `c`, table row `row`, `y[0..8]`.
// The value construction is `llvq_f1rank.cuh`'s `f1r_val` and the same
// int→float conversion `tv_f1r` performs at its FMA — on purpose, so what the
// bench measures between `tv_f1r` and `tv_f1r_v2` is the trellis alone.
__device__ __forceinline__ void f1r_section_v2_f(u32 p, u32 c, u32 row, float* y)
{
#pragma unroll
    for (u32 j = 0; j < 8u; ++j) {
        u32 o   = p + 2u * ((c >> j) & 1u);
        u32 rho = (row >> (4u * j)) & 15u;
        y[j] = (float)f1r_val(o, rho);
    }
}

// The decode, as `f1r_decode` but with `f1r_v2_patterns` in place of the three
// small-table reads. `t.rows` is the only table touched; the other three
// pointers of `t` are never dereferenced (the host harness passes null for
// them, so a read would crash the test). `lo` and `hi16` as in `f1r_decode`:
// the upper half of `hi16` is ignored. The 24 values are the exactly
// representable small integers `decode_word` returns, as floats.
__device__ __forceinline__ void f1r_decode_v2_f(u32 lo, u32 hi16, const F1rTables& t, float y[24])
{
    u32 p  = lo & 1u;
    u32 r  = (lo >> 1) & 1u;
    u32 i1 = (lo >> 9) & 0x7ffu;
    // i2 = bits 24..34: eight from the top of `lo`, three from the bottom of `hi16`.
    u32 i2 = ((lo >> 24) | ((hi16 & 7u) << 8)) & 0x7ffu;
    u32 i3 = (hi16 >> 4) & 0x7ffu;
    // bit 47, `hi16 >> 15`: the gain bit. Not read here.

    u32 cc = f1r_v2_patterns(lo, hi16);

    u32 row1 = t.rows[2048u * r + i1];
    bool mid  = i2 < F1R_N0_MIXED;
    u32 idx2  = mid ? i2 : (2048u - F1R_N0_MIXED) + i2;   // 2048 + i2 − N0
    u32 row2  = t.rows[idx2];
    u32 delta = mid ? 0u : 1u;
    u32 r3    = (p ^ r ^ delta) & 1u;
    u32 row3  = t.rows[2048u * r3 + i3];

    f1r_section_v2_f(p, cc & 0xffu, row1, y);
    f1r_section_v2_f(p, (cc >> 8) & 0xffu, row2, y + 8);
    f1r_section_v2_f(p, (cc >> 16) & 0xffu, row3, y + 16);
}

// The per-block product `Σ_s y_s · xb[s]`: the 24 values above, one
// `__fmaf_rn` chain from zero in coordinate order. `tv_f1r` folds the same 24
// products into its running accumulator instead; the two differ by the
// rounding of one addition per block, within the bench's control.
__device__ __forceinline__ float f1r_dot_v2(u32 lo, u32 hi16, const F1rTables& t, const float* xb)
{
    float y[LLVQ_DIM];
    f1r_decode_v2_f(lo, hi16, t, y);
    float acc = 0.0f;
#pragma unroll
    for (u32 s = 0; s < LLVQ_DIM; ++s) acc = __fmaf_rn(y[s], xb[s], acc);
    return acc;
}

#endif  // LLVQ_F1RANK_V2_CUH
