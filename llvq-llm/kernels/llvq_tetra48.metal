// The Tetra decoder in Metal Shading Language.
//
// A faithful port of `llvq-cuda/kernels/llvq_f1rank.cuh`,
// `llvq_f1rank_v3.cuh` and `llvq_tetra48.cuh`. The arithmetic is the same and
// is meant to be BIT-IDENTICAL: `tests/tetra48_metal_matches_rust.rs` decodes
// the same words here and in `llvq_search::tetra::Tetra` and demands equality,
// not a tolerance. A lattice decode that is merely close is wrong.
//
// ## What could not be ported, and what replaced it
//
// Three CUDA intrinsics have no MSL equivalent. Each is replaced by an
// expression proved equal rather than by something that looks similar.
//
//   __byte_perm(a, b, s)   PRMT. Written out below as `prmt`, including the
//                          sign-replication mode bit 3 of each selector nibble
//                          selects. The CUDA side relies on the plain mode
//                          only, but a port that silently dropped the other
//                          would be right until a table changed.
//
//   __dp4a(s, s, acc)      Four-way signed byte dot product. Written out as
//                          four multiplies. The operands are lattice
//                          coordinates bounded by 16, so the squares sum well
//                          inside an int and there is nothing to saturate.
//
//   f1r_v3_float           CUDA builds `0x4B0000XX` with a PRMT and subtracts
//                          8388736.0f, which yields `XX - 128` without an
//                          integer-to-float instruction. That is a scheduling
//                          trick, not arithmetic: 2^23 = 8388608 and the ulp
//                          there is exactly 1, so the bit pattern IS
//                          8388608 + XX. Metal converts directly and lands on
//                          the same value, exactly, for every XX in 0..255.
//
// ## Address spaces
//
// The four decoder tables total 18,688 bytes: rows 4096 u32, branches 1024
// u16, prefixes 128 u8, suffixes 128 u8. On NVIDIA they live in global memory
// and are read through L1, which the activation tile competes with; that
// competition is worth 19.1 % on sm_89 (`docs/mesures/tuile-l40s-2026-09-20.txt`).
// An M3 Max has 32,768 B of threadgroup memory, so the tables and a tile of 64
// blocks (6,144 B) both fit and the tables could be PINNED. They are left in
// `device` here on purpose: this file's job is to be correct first, and moving
// them is a measurement, not a guess.

#include <metal_stdlib>
using namespace metal;

// Every multiply-add in this file is EXACTLY what is written.
//
// Metal is clang, and clang contracts `a * b + c` into an `fma` unless told
// not to. `fma` rounds once where the written form rounds twice, so the two
// are different numbers. That is invisible in a benchmark and fatal here: the
// gate is equality against a host reference, and on 2026-09-20 the contraction
// put the matvec one to two ulp off on every row, which is the size of error a
// tolerance would have hidden and a real defect would also have produced.
//
// Turning Metal's fast math off is necessary and NOT sufficient: it stops the
// reassociation and leaves the contraction. This stops the contraction.
//
// Where fusion IS wanted the code calls `fma` by name, which is what the CUDA
// original does with `__fmaf_rn`. One decision, written down, on both sides.
#pragma clang fp contract(off)

// Class-0 rows in the middle section's mixed order. Mirrors
// `RankTable::n0_mixed`, which the Rust builder asserts equals 1240.
#define F1R_N0_MIXED 1240u
#define TETRA48_SHELLS 32u

// The v3 quad tables: eight biased bytes a class, `value + 128`.
//   o = 0 :  0, +4, -4, +8, -8, +12, -12, +16
//   o = 2 : +2, -2, +6, -6, +10, -10, +14, -14
//   o = 1 : +1, -3, +5, -7, +9, -11, +13, -15
//   o = 3 : -1, +3, -5, +7, -9, +11, -13, +15
#define F1R_V3_T0_LO 0x887c8480u
#define F1R_V3_T0_HI 0x90748c78u
#define F1R_V3_T2_LO 0x7a867e82u
#define F1R_V3_T2_HI 0x728e768au
#define F1R_V3_T1_LO 0x79857d81u
#define F1R_V3_T1_HI 0x718d7589u
#define F1R_V3_T3_LO 0x877b837fu
#define F1R_V3_T3_HI 0x8f738b77u

/// The 24 coordinates come out of the quads in this order.
constant uchar TETRA48_ORDER[24] = {
    0,  1,  2,  3,  4,  7,  10, 12, 6,  11, 13, 14,
    16, 17, 18, 19, 5,  8,  9,  15, 20, 21, 22, 23,
};

struct F1rTables {
    const device uint*   rows;       // 4096, two classes of 2048
    const device uchar*  prefixes;   // 128: [s8][b1]
    const device ushort* branches;   // 1024: [s8][b2], byte | s16 << 8
    const device uchar*  suffixes;   // 128: [s16][b3]
};

struct F1rV3Tab {
    uint c0lo, c0hi;   // o = p
    uint c1lo, c1hi;   // o = p + 2
};

/// CUDA's PRMT, default mode, written out.
///
/// The eight source bytes are `a` then `b`, little-endian within each. Each
/// nibble of `s` names one of them; its bit 3 asks for the SIGN of that byte
/// replicated over the output byte instead of the byte itself.
inline uint prmt(uint a, uint b, uint s)
{
    uint out = 0u;
    for (uint i = 0u; i < 4u; ++i) {
        uint sel = (s >> (4u * i)) & 0xfu;
        uint idx = sel & 7u;
        uint src = idx < 4u ? a : b;
        uint byte = (src >> (8u * (idx & 3u))) & 0xffu;
        uint v = (sel & 8u) ? ((byte & 0x80u) ? 0xffu : 0x00u) : byte;
        out |= v << (8u * i);
    }
    return out;
}

inline F1rV3Tab f1r_v3_tables(uint p)
{
    F1rV3Tab t;
    t.c0lo = p ? F1R_V3_T1_LO : F1R_V3_T0_LO;
    t.c0hi = p ? F1R_V3_T1_HI : F1R_V3_T0_HI;
    t.c1lo = p ? F1R_V3_T3_LO : F1R_V3_T2_LO;
    t.c1hi = p ? F1R_V3_T3_HI : F1R_V3_T2_HI;
    return t;
}

/// The four low bits of `c4`, one per output byte, as 0x00 or 0xff.
inline uint f1r_v3_bytemask(uint c4)
{
    return ((c4 * 0x00204081u) & 0x01010101u) * 0xffu;
}

inline uint f1r_v3_quad(F1rV3Tab t, uint sel, uint c4)
{
    uint m0 = prmt(t.c0lo, t.c0hi, sel);
    uint m1 = prmt(t.c1lo, t.c1hi, sel);
    uint k = f1r_v3_bytemask(c4);
    return (m0 & ~k) | (m1 & k);
}

inline void f1r_v3_section(F1rV3Tab t, uint c, uint row, thread uint& qa, thread uint& qb)
{
    qa = f1r_v3_quad(t, row, c & 0xfu);
    qb = f1r_v3_quad(t, row >> 16, c >> 4);
}

/// Coordinate `j` of a quad. See the header: exactly `byte - 128`.
inline float f1r_v3_float(uint quad, uint j)
{
    return float((quad >> (8u * j)) & 0xffu) - 128.0f;
}

struct F1rV3Block {
    uint p;
    uint c1, c2, c3;
    uint row1, row2, row3;
};

inline F1rV3Block f1r_v3_fetch(uint lo, uint hi16, F1rTables t)
{
    uint p  = lo & 1u;
    uint r  = (lo >> 1) & 1u;
    uint s8 = (lo >> 2) & 63u;
    uint b1 = (lo >> 8) & 1u;
    uint i1 = (lo >> 9) & 0x7ffu;
    uint b2 = (lo >> 20) & 15u;
    // i2 = bits 24..34: eight from the top of `lo`, three from the bottom of `hi16`.
    uint i2 = ((lo >> 24) | ((hi16 & 7u) << 8)) & 0x7ffu;
    uint b3 = (hi16 >> 3) & 1u;
    uint i3 = (hi16 >> 4) & 0x7ffu;
    // bit 47, `hi16 >> 15`: the gain bit. Not read here.

    F1rV3Block b;
    b.p  = p;
    b.c1 = t.prefixes[2u * s8 + b1];
    uint br = t.branches[16u * s8 + b2];
    b.c2 = br & 0xffu;
    // Masked to the 64 Golay states: a corrupted upload must not read past the
    // 128 suffix bytes.
    uint s16 = (br >> 8) & 63u;
    b.c3 = t.suffixes[2u * s16 + b3];

    b.row1 = t.rows[2048u * r + i1];
    bool mid = i2 < F1R_N0_MIXED;
    uint idx2 = mid ? i2 : (2048u - F1R_N0_MIXED) + i2;
    b.row2 = t.rows[idx2];
    uint delta = mid ? 0u : 1u;
    uint r3 = (p ^ r ^ delta) & 1u;
    b.row3 = t.rows[2048u * r3 + i3];
    return b;
}

inline void f1r_v3_quads(uint lo, uint hi16, F1rTables t, thread uint q[6])
{
    F1rV3Block b = f1r_v3_fetch(lo, hi16, t);
    F1rV3Tab   v = f1r_v3_tables(b.p);
    f1r_v3_section(v, b.c1, b.row1, q[0], q[1]);
    f1r_v3_section(v, b.c2, b.row2, q[2], q[3]);
    f1r_v3_section(v, b.c3, b.row3, q[4], q[5]);
}

/// The squared norm, from the biased bytes. CUDA reaches for `__dp4a`; the
/// operands are bounded by 16 so four multiplies carry no risk of overflow.
inline uint tetra48_n2(thread const uint q[6])
{
    int acc = 0;
    for (uint i = 0u; i < 6u; ++i) {
        uint x = q[i] ^ 0x80808080u;
        for (uint j = 0u; j < 4u; ++j) {
            int s = int(char((x >> (8u * j)) & 0xffu));
            acc += s * s;
        }
    }
    return uint(acc);
}

/// Where block `j` of a row sits, and the two aligned words that cover it.
///
/// A row's stream is packed: block j occupies bytes [6j, 6j+6). Its byte
/// offset is 0 or 2 mod 4, so two aligned words always cover it. The shift is
/// 0 or 16, never 32, so no expression degenerates. The window of an even `j`
/// reaches two bytes past 6j+6; the host pads every row to
/// round_up(6 * nblocks, 8) bytes, which covers it.
inline void f1r_load(const device uint* row, uint j, thread uint& lo, thread uint& hi16)
{
    uint w = (3u * j) >> 1;
    uint w0 = row[w];
    uint w1 = row[w + 1u];
    uint odd = j & 1u;
    lo   = odd ? ((w0 >> 16) | (w1 << 16)) : w0;
    hi16 = odd ? (w1 >> 16) : (w1 & 0xffffu);
}

/// One block against 24 activations, scaled by its gain and its shell.
inline float tetra48_dot(uint lo,
                         uint hi16,
                         F1rTables t,
                         const threadgroup float* xb,
                         const device float* gscale,
                         const device float* invnorm)
{
    uint q[6];
    f1r_v3_quads(lo, hi16, t, q);
    float acc = 0.0f;
    for (uint i = 0u; i < 6u; ++i) {
        for (uint j = 0u; j < 4u; ++j) {
            acc = fma(f1r_v3_float(q[i], j), xb[TETRA48_ORDER[4u * i + j]], acc);
        }
    }
    uint m = (tetra48_n2(q) >> 4) & (TETRA48_SHELLS - 1u);
    uint g = (hi16 >> 15) & 1u;
    return acc * gscale[g] * invnorm[m];
}

/// The lattice point of one block, unscaled, in artifact coordinate order.
///
/// The gate of the whole port. `tetra48_probe` writes these and a Rust test
/// compares them against `llvq_search::tetra::Tetra`, exactly.
inline void tetra48_point(uint lo, uint hi16, F1rTables t, thread float y[24])
{
    uint q[6];
    f1r_v3_quads(lo, hi16, t, q);
    for (uint i = 0u; i < 6u; ++i) {
        for (uint j = 0u; j < 4u; ++j) {
            y[TETRA48_ORDER[4u * i + j]] = f1r_v3_float(q[i], j);
        }
    }
}

// ---------------------------------------------------------------------------
// The probe. One thread a block, no tile, no reduction: it exists so the
// decoder can be judged on its own, before any matvec is written.
// ---------------------------------------------------------------------------

kernel void tetra48_probe(const device uint*   words    [[buffer(0)]],
                          const device uint*   rows     [[buffer(1)]],
                          const device uchar*  prefixes [[buffer(2)]],
                          const device ushort* branches [[buffer(3)]],
                          const device uchar*  suffixes [[buffer(4)]],
                          device float*        out      [[buffer(5)]],
                          device uint*         shell    [[buffer(6)]],
                          constant uint&       row_stride_u32 [[buffer(7)]],
                          constant uint&       nblocks  [[buffer(8)]],
                          uint gid [[thread_position_in_grid]])
{
    uint row = gid / nblocks;
    uint j   = gid % nblocks;
    F1rTables t = { rows, prefixes, branches, suffixes };

    uint lo, hi16;
    f1r_load(words + row * row_stride_u32, j, lo, hi16);

    float y[24];
    tetra48_point(lo, hi16, t, y);
    for (uint i = 0u; i < 24u; ++i) {
        out[gid * 24u + i] = y[i];
    }

    uint q[6];
    f1r_v3_quads(lo, hi16, t, q);
    // The shell index and the gain bit, the two things a point alone does not
    // carry and that the scale depends on.
    shell[gid * 2u + 0u] = (tetra48_n2(q) >> 4) & (TETRA48_SHELLS - 1u);
    shell[gid * 2u + 1u] = (hi16 >> 15) & 1u;
}

// ---------------------------------------------------------------------------
// The matvec: one SIMD-group a row, the activation staged in threadgroup
// memory. The Metal twin of `llvq-llm/kernels/tv_tetra48_h.cu`.
// ---------------------------------------------------------------------------

// Blocks of the activation one threadgroup stages.
//
// Host-injected by prepending a `#define`, the way the CUDA side injects it
// through NVRTC. The default is 64, which is the measured optimum on sm_89
// (`docs/mesures/tuile-l40s-2026-09-20.txt`). NOTHING is measured on Apple:
// the mechanism there is different, because 18,688 B of tables and a tile of
// 64 both fit in the 32,768 B of threadgroup memory, so the eviction that
// costs sm_89 19.1 % need not happen at all. Treat this number as a
// placeholder with a provenance, not as a tuned value.
#ifndef LLVQ_TILE_BLOCKS
#define LLVQ_TILE_BLOCKS 64u
#endif

/// The butterfly, written out rather than `simd_sum`.
///
/// `simd_sum` does not specify its reduction order, and floating-point
/// addition is not associative, so a kernel built on it cannot promise the
/// same bits twice across drivers. CUDA's `warp_sum` is an explicit
/// `__shfl_xor_sync` butterfly; this is the same one, lane for lane, which is
/// what lets the gate demand equality against a host reference.
inline float warp_sum(float v)
{
    for (ushort k = 16; k > 0; k >>= 1) {
        v += simd_shuffle_xor(v, k);
    }
    return v;
}

kernel void tv_tetra48_metal(const device uint*   words          [[buffer(0)]],
                             constant uint&       row_stride_u32 [[buffer(1)]],
                             const device uint*   rows           [[buffer(2)]],
                             const device uchar*  prefixes       [[buffer(3)]],
                             const device ushort* branches       [[buffer(4)]],
                             const device uchar*  suffixes       [[buffer(5)]],
                             const device float*  gscale         [[buffer(6)]],
                             const device float*  invnorm        [[buffer(7)]],
                             const device float*  rscale         [[buffer(8)]],
                             const device half*   tail           [[buffer(9)]],
                             const device float*  x              [[buffer(10)]],
                             device float*        y              [[buffer(11)]],
                             constant uint&       nblocks        [[buffer(12)]],
                             constant uint&       tail_w         [[buffer(13)]],
                             threadgroup float*   xs             [[threadgroup(0)]],
                             uint tid  [[thread_position_in_threadgroup]],
                             uint gid  [[thread_position_in_grid]],
                             uint tgs  [[threads_per_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]])
{
    uint row = gid >> 5;
    const device uint* wrow = words + row * row_stride_u32;
    F1rTables tab = { rows, prefixes, branches, suffixes };
    float acc = 0.0f;

    uint ntiles = (nblocks + LLVQ_TILE_BLOCKS - 1u) / LLVQ_TILE_BLOCKS;
    for (uint t = 0u; t < ntiles; ++t) {
        uint jlo = t * LLVQ_TILE_BLOCKS;
        uint jhi = min(jlo + LLVQ_TILE_BLOCKS, nblocks);
        uint n = (jhi - jlo) * 24u;
        // Two barriers, not one, for the reason matvec.cu gives: the second
        // orders the fill against the readers, the first stops the next fill
        // from racing a straggler still reading the previous tile. `ntiles`
        // depends only on `nblocks`, so both stay uniform across the group.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < n; i += tgs) {
            xs[i] = x[jlo * 24u + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint j = jlo + lane; j < jhi; j += 32u) {
            uint lo, hi16;
            f1r_load(wrow, j, lo, hi16);
            acc += tetra48_dot(lo, hi16, tab, xs + (j - jlo) * 24u, gscale, invnorm);
        }
    }

    acc = warp_sum(acc);
    if (lane == 0u) {
        // Multiply-then-add, not `fma`: this is the association the CUDA
        // epilogue has, and the only thing that may differ between the two
        // paths is the stored width of the weight, never the arithmetic.
        float tv = 0.0f;
        const device float* xt = x + nblocks * 24u;
        for (uint i = 0u; i < tail_w; ++i) {
            tv += float(tail[row * tail_w + i]) * xt[i];
        }
        y[row] = acc * rscale[row] + tv;
    }
}
