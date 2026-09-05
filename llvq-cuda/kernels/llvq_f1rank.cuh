// The F1 universal-table block decoder: one 48-bit word → 24 coordinates,
// through a 16 KiB table every one of the 528 section regions shares.
//
// The reference is `llvq_bench::f1::rank::decode_word`, and this file must
// agree with it bit for bit: `tests/f1rank_matches_rust.rs` compiles this text
// as host C++ through `tests/host_shim.h` and diffs 10,000 random words, and
// `bin/f1rankfloor` decodes blocks on the card (`tv_f1r_dump`) and compares
// them to the same reference before it prints a millisecond.
//
// ## The object
//
// Λ₂₄ in trio coordinate order is a three-section code: three sections of 8
// coordinates, each `y_j = p + 2·c_j + 4·k_j` with `p ∈ {0,1}` shared by the
// block, `c` the section's Golay pattern byte and `k ∈ Z^8`. The word stores,
// per section, not the point but its RANK VECTOR `ρ ∈ {0..7}^8`: coordinate j
// is the `ρ_j`-th value of the progression `o_j + 4Z` listed outward from
// zero, with `o_j = p + 2·c_j ∈ {0,1,2,3}`:
//
//     o = 0 : ρ = 0 → 0 ; else m = 4·((ρ+1) >> 1), y = (ρ & 1) ? +m : −m   (0, +4, −4, +8, −8, …)
//     o = 2 : m = 2 + 4·(ρ >> 1),                y = (ρ & 1) ? −m : +m   (+2, −2, +6, −6, +10, …)
//     o = 1 : m = 2ρ + 1,                        y = (ρ & 1) ? −m : +m   (+1, −3, +5, −7, +9, …)
//     o = 3 : m = 2ρ + 1,                        y = (ρ & 1) ? +m : −m   (−1, +3, −5, +7, −9, …)
//
// The class of a rank vector, `cls(ρ) = Σ_j [ρ_j ∈ {1,2}] mod 2`, equals the
// parity of `Σk` for every pattern, which is what lets one table serve every
// region.
//
// ## The table (`F1rTables::rows`, 4,096 × u32, 16 KiB)
//
//     rows    0..2047 : the 2,048 lowest-cost rank vectors of class 0,
//                       sorted by (cost, ρ lexicographic) ascending
//     rows 2048..4095 : the 2,048 lowest-cost of class 1, same order
//     cost(ρ) = Σ_j (2ρ_j + 1)²   ;   rank j at bits 4j..4j+3 of the row
//
// The middle section's 2,048 rows are "the 2,048 lowest-cost overall": class-0
// rows 0..N0−1 followed by class-1 rows 0..(2048−N0)−1, with N0 = 1240
// (`F1R_N0_MIXED`, asserted by the Rust builder and by the bench against it).
//
// The three small tables are the trellis: `prefixes[2·s8 + b]` (u8, 128) the
// two prefix bytes of Golay state s8 at cut 8; `branches[16·s8 + b]` (u16,
// 1,024) the 16 middle bytes leaving s8, low byte = pattern byte, high byte =
// the state s16 reached, sorted by byte; `suffixes[2·s16 + b]` (u8, 128).
//
// ## The 48-bit word (bit 0 = least significant; a block is 6 bytes, LE)
//
//     bit 0        p
//     bit 1        r        k-parity class of section 1
//     bits 2..7    s8       Golay state at cut 8 (0..63)
//     bit 8        b1       which of the 2 prefix bytes of s8
//     bits 9..19   i1       row of section 1 in class r            (11 bits)
//     bits 20..23  b2       branch out of s8 (0..15)
//     bits 24..34  i2       row of section 2 in the MIXED order    (11 bits)
//     bit 35       b3       which of the 2 suffix bytes of s16
//     bits 36..46  i3       row of section 3 in class r3           (11 bits)
//     bit 47       g        gain bit — read by the served kernel, IGNORED by
//                           this floor, which applies no scale
//
// Every field range is a power of two, so any 48-bit word is a valid label
// and the decoded point is in Λ₂₄ — which is why the bench may stream
// pseudo-random words and still check every decode against the reference.
//
// ## No dynamic indexing
//
// Every coordinate is a chain of selects over `(o, ρ)`; the only indexed
// reads are the six table loads, all from global memory. `llvq_planes.cuh`
// says why: a computed index into a local array is a local-memory spill on
// the hottest path, and `local_size_bytes() == 0` is the contract the bench
// prints. The `y[24]` output is written under a fully unrolled loop with
// constant indices, which is what keeps it in registers.
//
// Host-compilable on purpose (`__device__ __forceinline__` is defined away by
// the shim): plain shifts, masks and `?:` — no `__byte_perm`, no
// `__funnelshift_*`, nothing `host_shim.h` does not define.

#ifndef LLVQ_F1RANK_CUH
#define LLVQ_F1RANK_CUH

#ifndef LLVQ_SLOT_CUH
#include "llvq_slot.cuh"
#endif

// Class-0 rows in the middle section's mixed order. Mirrors
// `RankTable::n0_mixed`, which the Rust builder asserts equals 1240; the
// bench asserts the uploaded table agrees before any launch.
#define F1R_N0_MIXED 1240u

struct F1rTables {
    const u32*            rows;       // 4096 × u32, two classes of 2048
    const unsigned char*  prefixes;   // 128: [s8][b1]
    const unsigned short* branches;   // 1024: [s8][b2], byte | s16 << 8
    const unsigned char*  suffixes;   // 128: [s16][b3]
};

// `val(o, ρ)` of the table above, as selects.
//
//   m : o odd (1, 3)      → 2ρ + 1
//       o = 2             → 2 + 4·(ρ >> 1)
//       o = 0             → 4·((ρ + 1) >> 1), which is 0 at ρ = 0, so the
//                           "ρ = 0 → 0" case needs no branch of its own
//   sign : +m when (ρ odd) XOR (o ∈ {1, 2}), −m otherwise — check against the
//          four progressions: o=0 ρ=1 → +4, o=2 ρ=0 → +2, o=1 ρ=1 → −3,
//          o=3 ρ=0 → −1. `(o + 1) >> 1` is 1 exactly for o ∈ {1, 2}.
__device__ __forceinline__ int f1r_val(u32 o, u32 rho)
{
    int m_odd = 2 * (int)rho + 1;
    int m_two = 2 + 4 * (int)(rho >> 1);
    int m_zero = 4 * (int)((rho + 1u) >> 1);
    int m = (o & 1u) ? m_odd : ((o & 2u) ? m_two : m_zero);
    u32 plus = (rho + ((o + 1u) >> 1)) & 1u;
    return plus ? m : -m;
}

// One section: pattern byte `c`, table row `row`, into `y[0..8]`.
__device__ __forceinline__ void f1r_section(u32 p, u32 c, u32 row, signed char* y)
{
#pragma unroll
    for (u32 j = 0; j < 8u; ++j) {
        u32 o   = p + 2u * ((c >> j) & 1u);
        u32 rho = (row >> (4u * j)) & 15u;
        y[j] = (signed char)f1r_val(o, rho);
    }
}

// The decode. `lo` = bits 0..31 of the word, `hi16` = bits 32..47 in its low
// half; the upper bits of `hi16` are ignored (every field read from it is
// masked), so a caller may pass a whole u32 of stream.
//
// The three small-table reads are a dependent chain, `s8 → br → s16 → c3`;
// the three row reads depend only on the word. The middle row index is one
// select and one load, not two loads.
__device__ __forceinline__ void f1r_decode(u32 lo, u32 hi16, const F1rTables& t, signed char y[24])
{
    u32 p  = lo & 1u;
    u32 r  = (lo >> 1) & 1u;
    u32 s8 = (lo >> 2) & 63u;
    u32 b1 = (lo >> 8) & 1u;
    u32 i1 = (lo >> 9) & 0x7ffu;
    u32 b2 = (lo >> 20) & 15u;
    // i2 = bits 24..34: eight from the top of `lo`, three from the bottom of `hi16`.
    u32 i2 = ((lo >> 24) | ((hi16 & 7u) << 8)) & 0x7ffu;
    u32 b3 = (hi16 >> 3) & 1u;
    u32 i3 = (hi16 >> 4) & 0x7ffu;
    // bit 47, `hi16 >> 15`: the gain bit. Not read here.

    u32 c1  = t.prefixes[2u * s8 + b1];
    u32 br  = t.branches[16u * s8 + b2];
    u32 c2  = br & 0xffu;
    // Masked to the 64 Golay states: a corrupted upload must not read past the
    // 128 suffix bytes — an out-of-bounds read kills the context of a billed job.
    u32 s16 = (br >> 8) & 63u;
    u32 c3  = t.suffixes[2u * s16 + b3];

    u32 row1 = t.rows[2048u * r + i1];
    bool mid  = i2 < F1R_N0_MIXED;
    u32 idx2  = mid ? i2 : (2048u - F1R_N0_MIXED) + i2;   // 2048 + i2 − N0
    u32 row2  = t.rows[idx2];
    u32 delta = mid ? 0u : 1u;
    u32 r3    = (p ^ r ^ delta) & 1u;
    u32 row3  = t.rows[2048u * r3 + i3];

    f1r_section(p, c1, row1, y);
    f1r_section(p, c2, row2, y + 8);
    f1r_section(p, c3, row3, y + 16);
}

// Where block `j` of a row sits, and the two aligned u32 that cover it.
//
// A row's stream is packed: block j occupies bytes [6j, 6j+6). Its byte offset
// is ≡ 0 or 2 (mod 4), so two aligned words always cover it: `w0` at u32
// index 6j/4 = 3j >> 1, `w1` the next, in-word shift 16·(j & 1). Written out:
//
//     j = 0 : bytes  0..6   w0 = row[0] (bytes  0..4)  w1 = row[1] (bytes  4..8)   sh = 0
//             lo = w0 (bytes 0..4)                hi16 = w1 & 0xffff (bytes 4..6)
//     j = 1 : bytes  6..12  w0 = row[1] (bytes  4..8)  w1 = row[2] (bytes  8..12)  sh = 16
//             lo = w0>>16 | w1<<16 (bytes 6..10)  hi16 = w1 >> 16 (bytes 10..12)
//     j = 2 : bytes 12..18  w0 = row[3] (bytes 12..16) w1 = row[4] (bytes 16..20)  sh = 0
//             lo = w0 (bytes 12..16)              hi16 = w1 & 0xffff (bytes 16..18)
//     j = 3 : bytes 18..24  w0 = row[4] (bytes 16..20) w1 = row[5] (bytes 20..24)  sh = 16
//             lo = w0>>16 | w1<<16 (bytes 18..22) hi16 = w1 >> 16 (bytes 22..24)
//
// The shift is 0 or 16, never 32, so no expression degenerates — the two
// cases are selected, not shifted by a variable. The window of an even j
// reaches 2 bytes past 6j+6; the host pads every row to
// round_up(6·nblocks, 8) bytes, which covers it (`bin/f1rankfloor` asserts
// the inequality), so `w1` of the last block of the last row stays in bounds.
__device__ __forceinline__ void f1r_load(const u32* row, u32 j, u32& lo, u32& hi16)
{
    u32 w   = (3u * j) >> 1;
    u32 w0  = row[w];
    u32 w1  = row[w + 1u];
    u32 odd = j & 1u;
    lo   = odd ? ((w0 >> 16) | (w1 << 16)) : w0;
    hi16 = odd ? (w1 >> 16) : (w1 & 0xffffu);
}

// The stream generator, so the words are made ON THE DEVICE and the host
// can reproduce any of them for the dump check. A 32-bit finaliser applied
// to `seed + i` (wrapping), word i of a buffer:
//
//     h ^= h >> 16;  h *= 0x7feb352d;  h ^= h >> 15;  h *= 0x846ca68b;  h ^= h >> 16;
//
// Mirrored bit for bit by `mix32` in `bin/f1rankfloor.rs`. Not a quality
// claim: what the bench needs is that consecutive words share nothing, so
// the table rows are drawn uniformly, and that the host can replay it.
__device__ __forceinline__ u32 f1r_mix32(u32 h)
{
    h ^= h >> 16;
    h *= 0x7feb352du;
    h ^= h >> 15;
    h *= 0x846ca68bu;
    h ^= h >> 16;
    return h;
}

#endif  // LLVQ_F1RANK_CUH
