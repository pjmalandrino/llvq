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
//   __byte_perm(a, b, s)   PRMT. Written out below as `prmt`. The sign mode
//                          bit 3 selects is implemented for completeness and
//                          is UNREACHABLE from the CUDA side, which ANDs its
//                          selector with 0x7777 before the instruction. So it
//                          is dead code that documents the instruction, not
//                          the insurance an earlier draft of this comment
//                          claimed it was.
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
// memory. The Metal counterpart of `llvq-llm/kernels/tv_tetra48_h.cu`.
//
// NOT its twin, and an audit of 2026-09-21 made the word precise. `y` here is
// `device float*`; the CUDA one is `unsigned short*` written through `f2h`.
// The arithmetic up to the store is the same, including the two `fma` in the
// epilogue, but this kernel does not narrow. Storing f16 is what an inference
// runtime wants and is a later lot; narrowing here without a gate that
// compares the two widths would be a claim rather than a change.
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
        // `fma` at BOTH sites, and this was wrong until an audit caught it.
        //
        // The CUDA twin writes `tv += h2f(...) * xt[i]` and
        // `y[row] = f2h(acc * rscale[row] + tv)` as plain expressions, and
        // `llvq-cuda/src/gpu.rs:14` records that NVRTC compiles with
        // `--fmad=true`, "NVRTC's default anyway". So on the card both
        // contract into an FFMA and round ONCE. An earlier version of this
        // file copied the SOURCE and, under `contract(off)`, rounded twice.
        //
        // Measured: over two million random (acc, rscale, tv) triples in the
        // ranges this kernel sees, the two forms give a different f32 on
        // 27 % of them. The two host references could not see it, because
        // both were written from this shader rather than from the card.
        float tv = 0.0f;
        const device float* xt = x + nblocks * 24u;
        for (uint i = 0u; i < tail_w; ++i) {
            tv = fma(float(tail[row * tail_w + i]), xt[i], tv);
        }
        y[row] = fma(acc, rscale[row], tv);
    }
}

// ---------------------------------------------------------------------------
// The same matvec with the decoder tables PINNED in threadgroup memory.
//
// ## The question it asks
//
// `tv_tetra48_metal` reads its four tables from `device` memory, three
// DEPENDENT lookups a block, 106 blocks a row. On NVIDIA those tables sit in
// L1 and the activation tile competes with them for the same 102,400 B of
// SRAM; that competition is worth 19.1 % on sm_89
// (`docs/mesures/tuile-l40s-2026-09-20.txt`). On Apple they have no residency
// at all, and the measured matvec runs at 49 GB/s against the card's 146.
//
// The tables are 18,688 B and a tile of 64 is 6,144. Together 24,832 against
// Apple's 32,768, so they FIT. This variant copies them in once a threadgroup
// and decodes from there.
//
// ## Why it might lose
//
// Every threadgroup re-reads 18,688 B. A 2,560-row matrix launches 320 of
// them, so that is 6.0 MB of copies against 1.6 MB of weight stream. If the
// copies do not come out of cache, the cure is worse. And pinning 18,688 of
// 32,768 caps the threadgroups resident on a core, trading latency for
// occupancy.
//
// Which wins is a measurement. `llvq-metal/examples/metalsplit.rs` makes it.
//
// ## Why the decode is duplicated rather than shared
//
// MSL has no address-space-generic pointer, so a function that reads
// `device const uint*` cannot read `threadgroup const uint*`. The three
// fetch steps are repeated below with the other qualifier and nothing else
// changed. A diff of the two bodies should show only the word `threadgroup`.
// ---------------------------------------------------------------------------

struct F1rTablesTG {
    const threadgroup uint*   rows;
    const threadgroup uchar*  prefixes;
    const threadgroup ushort* branches;
    const threadgroup uchar*  suffixes;
};

inline void f1r_v3_quads_tg(uint lo, uint hi16, F1rTablesTG t, thread uint q[6])
{
    uint p  = lo & 1u;
    uint r  = (lo >> 1) & 1u;
    uint s8 = (lo >> 2) & 63u;
    uint b1 = (lo >> 8) & 1u;
    uint i1 = (lo >> 9) & 0x7ffu;
    uint b2 = (lo >> 20) & 15u;
    uint i2 = ((lo >> 24) | ((hi16 & 7u) << 8)) & 0x7ffu;
    uint b3 = (hi16 >> 3) & 1u;
    uint i3 = (hi16 >> 4) & 0x7ffu;

    uint c1 = t.prefixes[2u * s8 + b1];
    uint br = t.branches[16u * s8 + b2];
    uint c2 = br & 0xffu;
    uint s16 = (br >> 8) & 63u;
    uint c3 = t.suffixes[2u * s16 + b3];

    uint row1 = t.rows[2048u * r + i1];
    bool mid = i2 < F1R_N0_MIXED;
    uint idx2 = mid ? i2 : (2048u - F1R_N0_MIXED) + i2;
    uint row2 = t.rows[idx2];
    uint delta = mid ? 0u : 1u;
    uint r3 = (p ^ r ^ delta) & 1u;
    uint row3 = t.rows[2048u * r3 + i3];

    F1rV3Tab v = f1r_v3_tables(p);
    f1r_v3_section(v, c1, row1, q[0], q[1]);
    f1r_v3_section(v, c2, row2, q[2], q[3]);
    f1r_v3_section(v, c3, row3, q[4], q[5]);
}

inline float tetra48_dot_tg(uint lo,
                            uint hi16,
                            F1rTablesTG t,
                            const threadgroup float* xb,
                            const device float* gscale,
                            const device float* invnorm)
{
    uint q[6];
    f1r_v3_quads_tg(lo, hi16, t, q);
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

kernel void tv_tetra48_metal_tg(const device uint*   words          [[buffer(0)]],
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
                                threadgroup uint*    t_rows         [[threadgroup(1)]],
                                threadgroup ushort*  t_bran         [[threadgroup(2)]],
                                threadgroup uchar*   t_pref         [[threadgroup(3)]],
                                threadgroup uchar*   t_suff         [[threadgroup(4)]],
                                uint tid  [[thread_position_in_threadgroup]],
                                uint gid  [[thread_position_in_grid]],
                                uint tgs  [[threads_per_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]])
{
    // The copy, once a threadgroup, before anything reads a table.
    for (uint i = tid; i < 4096u; i += tgs) t_rows[i] = rows[i];
    for (uint i = tid; i < 1024u; i += tgs) t_bran[i] = branches[i];
    for (uint i = tid; i < 128u; i += tgs) { t_pref[i] = prefixes[i]; t_suff[i] = suffixes[i]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint row = gid >> 5;
    const device uint* wrow = words + row * row_stride_u32;
    F1rTablesTG tab = { t_rows, t_pref, t_bran, t_suff };
    float acc = 0.0f;

    uint ntiles = (nblocks + LLVQ_TILE_BLOCKS - 1u) / LLVQ_TILE_BLOCKS;
    for (uint t = 0u; t < ntiles; ++t) {
        uint jlo = t * LLVQ_TILE_BLOCKS;
        uint jhi = min(jlo + LLVQ_TILE_BLOCKS, nblocks);
        uint n = (jhi - jlo) * 24u;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < n; i += tgs) {
            xs[i] = x[jlo * 24u + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint j = jlo + lane; j < jhi; j += 32u) {
            uint lo, hi16;
            f1r_load(wrow, j, lo, hi16);
            acc += tetra48_dot_tg(lo, hi16, tab, xs + (j - jlo) * 24u, gscale, invnorm);
        }
    }

    acc = warp_sum(acc);
    if (lane == 0u) {
        float tv = 0.0f;
        const device float* xt = x + nblocks * 24u;
        for (uint i = 0u; i < tail_w; ++i) {
            tv = fma(float(tail[row * tail_w + i]), xt[i], tv);
        }
        y[row] = fma(acc, rscale[row], tv);
    }
}

// ---------------------------------------------------------------------------
// The same matvec with the value computed instead of looked up.
//
// ## Why the v3 tables exist at all, and why that reason does not cross
//
// `llvq_f1rank_v3.cuh` packs the lattice coordinates into four byte tables and
// gathers them with `prmt.b32`, ONE instruction. Its own header says so: the
// tables are there because PRMT makes a byte gather cheaper than the
// arithmetic.
//
// Metal has no PRMT. The `prmt` above emulates it with a four-trip loop of
// shifts, masks and selects, and it is called TWELVE times a block. That is
// roughly 240 instructions a block bought to avoid about 8.
//
// So this variant drops the v3 representation and computes `val(o, rho)`
// directly, which is what the SCALAR decoder `f1r_val` in `llvq_f1rank.cuh`
// does and has always done. Same values, no table, no emulation.
//
//   m : o odd (1, 3) -> 2*rho + 1
//       o = 2        -> 2 + 4*(rho >> 1)
//       o = 0        -> 4*((rho + 1) >> 1), zero at rho = 0
//   sign : plus when (rho + ((o + 1) >> 1)) is odd
//
// Whether it wins is a measurement, and porting an optimisation whose premise
// does not hold is the mistake this file is correcting.
// ---------------------------------------------------------------------------

/// `val(o, rho)`, branchless. A transcription of `f1r_val`.
inline int f1r_val_m(uint o, uint rho)
{
    int m_odd = 2 * int(rho) + 1;
    int m_two = 2 + 4 * int(rho >> 1);
    int m_zero = 4 * int((rho + 1u) >> 1);
    int m = (o & 1u) ? m_odd : ((o & 2u) ? m_two : m_zero);
    uint plus = (rho + ((o + 1u) >> 1)) & 1u;
    return plus ? m : -m;
}

inline float tetra48_dot_arith(uint lo,
                               uint hi16,
                               F1rTables t,
                               const threadgroup float* xb,
                               const device float* gscale,
                               const device float* invnorm)
{
    F1rV3Block b = f1r_v3_fetch(lo, hi16, t);
    uint cs[3] = { b.c1, b.c2, b.c3 };
    uint rs[3] = { b.row1, b.row2, b.row3 };
    float acc = 0.0f;
    int n2 = 0;
    for (uint sec = 0u; sec < 3u; ++sec) {
        uint c = cs[sec];
        uint row = rs[sec];
        for (uint j = 0u; j < 8u; ++j) {
            uint o = b.p + 2u * ((c >> j) & 1u);
            uint rho = (row >> (4u * j)) & 15u;
            int v = f1r_val_m(o, rho);
            n2 += v * v;
            acc = fma(float(v), xb[TETRA48_ORDER[sec * 8u + j]], acc);
        }
    }
    uint m = (uint(n2) >> 4) & (TETRA48_SHELLS - 1u);
    uint g = (hi16 >> 15) & 1u;
    return acc * gscale[g] * invnorm[m];
}

kernel void tv_tetra48_metal_ar(const device uint*   words          [[buffer(0)]],
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
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < n; i += tgs) {
            xs[i] = x[jlo * 24u + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint j = jlo + lane; j < jhi; j += 32u) {
            uint lo, hi16;
            f1r_load(wrow, j, lo, hi16);
            acc += tetra48_dot_arith(lo, hi16, tab, xs + (j - jlo) * 24u, gscale, invnorm);
        }
    }

    acc = warp_sum(acc);
    if (lane == 0u) {
        float tv = 0.0f;
        const device float* xt = x + nblocks * 24u;
        for (uint i = 0u; i < tail_w; ++i) {
            tv = fma(float(tail[row * tail_w + i]), xt[i], tv);
        }
        y[row] = fma(acc, rscale[row], tv);
    }
}

// ---------------------------------------------------------------------------
// `tv_tetra48_metal_ar` storing f16, which is what the CUDA twin does.
//
// The model runs in f16. With an f32 store the adapter narrows afterwards,
// one candle launch a projection: 290 of the path's 688 launches a token,
// for no arithmetic. This ends in `half(...)` instead and the adapter keeps
// what it is handed.
// ---------------------------------------------------------------------------
kernel void tv_tetra48_metal_arh(const device uint*   words          [[buffer(0)]],
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
                                device half*         y              [[buffer(11)]],
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
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < n; i += tgs) {
            xs[i] = x[jlo * 24u + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint j = jlo + lane; j < jhi; j += 32u) {
            uint lo, hi16;
            f1r_load(wrow, j, lo, hi16);
            acc += tetra48_dot_arith(lo, hi16, tab, xs + (j - jlo) * 24u, gscale, invnorm);
        }
    }

    acc = warp_sum(acc);
    if (lane == 0u) {
        float tv = 0.0f;
        const device float* xt = x + nblocks * 24u;
        for (uint i = 0u; i < tail_w; ++i) {
            tv = fma(float(tail[row * tail_w + i]), xt[i], tv);
        }
        // Narrowed IN the kernel, as `tv_tetra48_h.cu` does with `f2h`.
        // MSL's float-to-half is IEEE round-to-nearest-even, the same rule
        // `f2h` implements and the same one candle's `to_dtype` applies, so
        // the value is what the adapter used to produce. What it saves is
        // the LAUNCH: 290 of this path's 688 a token were conversions.
        y[row] = half(fma(acc, rscale[row], tv));
    }
}

// ---------------------------------------------------------------------------
// The same value, looked up in 64 floats instead of computed.
//
// ## Why a table again, after removing one
//
// The v3 table was removed because REACHING it cost a PRMT emulation, twenty
// instructions to gather one byte out of a packed word. This table is not
// packed: `val(o, rho)` has 4 x 16 = 64 entries, one float each, indexed by
// `(o << 4) | rho` with a single load. There is nothing to gather.
//
// So the arithmetic version trades about ten operations a coordinate for one
// indexed load from `constant` memory, 24 times a block. 256 bytes total,
// which every thread of every threadgroup reads and which is small enough to
// be resident wherever Apple keeps `constant`.
//
// The values are `f1r_val`'s, enumerated. Their range is [-31, 32], integers,
// so the float form is exact and the shell sum below is unaffected.
// ---------------------------------------------------------------------------

constant float F1R_VAL[64] = {
        0.0f,     4.0f,    -4.0f,     8.0f,    -8.0f,    12.0f,   -12.0f,    16.0f,   -16.0f,    20.0f,   -20.0f,    24.0f,   -24.0f,    28.0f,   -28.0f,    32.0f,
        1.0f,    -3.0f,     5.0f,    -7.0f,     9.0f,   -11.0f,    13.0f,   -15.0f,    17.0f,   -19.0f,    21.0f,   -23.0f,    25.0f,   -27.0f,    29.0f,   -31.0f,
        2.0f,    -2.0f,     6.0f,    -6.0f,    10.0f,   -10.0f,    14.0f,   -14.0f,    18.0f,   -18.0f,    22.0f,   -22.0f,    26.0f,   -26.0f,    30.0f,   -30.0f,
       -1.0f,     3.0f,    -5.0f,     7.0f,    -9.0f,    11.0f,   -13.0f,    15.0f,   -17.0f,    19.0f,   -21.0f,    23.0f,   -25.0f,    27.0f,   -29.0f,    31.0f,
};

inline float tetra48_dot_lut(uint lo,
                             uint hi16,
                             F1rTables t,
                             const threadgroup float* xb,
                             const device float* gscale,
                             const device float* invnorm)
{
    F1rV3Block b = f1r_v3_fetch(lo, hi16, t);
    uint cs[3] = { b.c1, b.c2, b.c3 };
    uint rs[3] = { b.row1, b.row2, b.row3 };
    float acc = 0.0f;
    float n2 = 0.0f;
    for (uint sec = 0u; sec < 3u; ++sec) {
        uint c = cs[sec];
        uint row = rs[sec];
        for (uint j = 0u; j < 8u; ++j) {
            uint o = b.p + 2u * ((c >> j) & 1u);
            uint rho = (row >> (4u * j)) & 15u;
            float v = F1R_VAL[(o << 4) | rho];
            n2 = fma(v, v, n2);
            acc = fma(v, xb[TETRA48_ORDER[sec * 8u + j]], acc);
        }
    }
    // The squares are integers bounded by 24 * 32^2 = 24,576, so the f32 sum
    // is exact and the truncation below is the integer one.
    uint m = (uint(n2) >> 4) & (TETRA48_SHELLS - 1u);
    uint g = (hi16 >> 15) & 1u;
    return acc * gscale[g] * invnorm[m];
}

kernel void tv_tetra48_metal_lut(const device uint*   words          [[buffer(0)]],
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
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < n; i += tgs) {
            xs[i] = x[jlo * 24u + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint j = jlo + lane; j < jhi; j += 32u) {
            uint lo, hi16;
            f1r_load(wrow, j, lo, hi16);
            acc += tetra48_dot_lut(lo, hi16, tab, xs + (j - jlo) * 24u, gscale, invnorm);
        }
    }

    acc = warp_sum(acc);
    if (lane == 0u) {
        float tv = 0.0f;
        const device float* xt = x + nblocks * 24u;
        for (uint i = 0u; i < tail_w; ++i) {
            tv = fma(float(tail[row * tail_w + i]), xt[i], tv);
        }
        y[row] = fma(acc, rscale[row], tv);
    }
}

// ---------------------------------------------------------------------------
// `tv_tetra48_metal_ar` storing f16, which is what the CUDA twin does.
//
// The model runs in f16. With an f32 store the adapter narrows afterwards,
// one candle launch a projection: 290 of the path's 688 launches a token,
// for no arithmetic. This ends in `half(...)` instead and the adapter keeps
// what it is handed.
// ---------------------------------------------------------------------------
