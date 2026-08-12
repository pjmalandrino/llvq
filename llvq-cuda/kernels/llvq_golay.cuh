// The Golay70 block decoder — E2 of docs/pistes-format-vram-2026-08-05.md:
// the Golay/GF(2) stage. A 9-byte AoS main stream whose per-slot decode leans
// on the algebra of the Leech construction itself — the residue-mod-4 plane of
// every block IS a Golay codeword, so 12 bits of codeword rank replace what
// the bit-plane layouts spend on explicit level indices — plus a per-matrix
// exception table carrying the exact 14-byte Planes14 record of every block
// the 2-bit in-class code cannot hold.
//
// Main-stream record, from bit 0, LSB-first within a byte, bytes ascending —
// the same convention as Slot32/Planes14/Planes12x:
//
//     [ class : 9 ][ gain : 1 ][ golay : 12 ][ A : 24 ][ B : 24 ][ 0 : 2 ]
//
// `golay` is the RANK of a codeword in the canonical 4096-entry table
// (`Golay::codewords()` order — the order frozen by format v1); the kernel
// resolves it through a 16 KiB table of 24-bit words passed as a device
// array, resident in L1 after first touch. No XOR re-encoding from the 12
// generator combinations in-kernel: the table is the verified infrastructure.
//
// ## The two cosets, one predicated decode
//
// * EVEN class: the codeword is `c = {i : |x_i| ≡ 2 (mod 4)}` — established
//   to be a Golay codeword by the Λ₂₄ construction (−2 ≡ 2). Its bit fixes
//   the slot's residue; bit A picks which of the ≤ 2 values of that residue
//   (classes with a single value in a residue duplicate it in the table, so
//   the pick is branchless and the transcoder writes A = 0 canonically); bit
//   B is the sign, exact Slot32 `smask` semantics (zero slots written 0).
// * ODD class: the codeword is `c = {i : x_i ≡ 3 (mod 4)}` — the rank the
//   v1 index already carries. A and B are two level bit-planes
//   (`level = A | B<<1`, a direct index into the ≤ 4 class values); the
//   sign is COMPUTED, not stored: `x_i > 0  ⇔  c_i == flag(value)` with
//   `flag = 1` iff the value ≡ 3 (mod 4) — the "signs carry no information"
//   rule of generic.rs / index.rs, so `neg = c_i ^ flag`.
//
// Both formulas are evaluated as predicated selections and chosen by the
// class's `is_odd` flag: the memory traffic is identical on both cosets
// (window words + one codeword + one class record), so warps mixing cosets
// 50/50 diverge in ALU predication only — the assumed cost of E2, and what
// the bench's 5th arm exists to measure.
//
// ## v2 (2026-08-11): the coset logic is hoisted to block level
//
// The E2 bench answered: the assumed cost was the whole story. The v1
// decoder resolved the coset per SLOT — two `odd ?` selects plus the odd
// sign chain `flag = (flags >> ((hi<<1)|abit)) & 1; neg = cbit ^ flag`, a
// serial ~14 integer ops per slot against ~9 for a Planes14 slot — and the
// measured 198 GB/s (30 % of the byte bound, ×1.61 the Planes14 time, vs a
// ~1.56 instruction-count ratio) is attributed to exactly that count
// (docs/projections-golay70-2026-08-11.md §3.1).
//
// Everything that distinguishes the cosets is a function of the block's
// three 24-bit words, so it hoists into a per-block prologue:
//
//   fw = flag[(bbit<<1)|abit] per bit — three mask muxes over m0..m3, the
//        24-bit broadcasts of the class's flag bits (even classes have
//        flags = 0, so fw = 0 with no branch);
//   hw = odd ? B : c        — the high select bit of every slot;
//   nw = odd ? (c ^ fw) : B — the sign bit of every slot.
//
// after which a slot is exactly a Planes14 slot with three masks instead of
// four: three immediate mask tests, the same predicated value tree, one
// negation — no variable shift, no coset select left on the per-slot path.
// Identity with v1, coset by coset:
//
//   even: hi  = cbit = c_j  = hw_j ;  neg = bbit = B_j = nw_j.
//   odd:  hi  = bbit = B_j  = hw_j ;  neg = cbit ^ flag[(bbit<<1)|abit]
//                                         = (c ^ fw)_j.
//
// The identity is asserted slot by slot by the host probe, block by block
// by the Rust reference (tests/golay70_decoder_matches_rust.rs), and class
// by class by the hand-packed record probe — the format itself is
// unchanged: not one stored byte moves.
//
// ## The read window
//
// The byte offset 9·b is ≡ 0, 1, 2 or 3 (mod 4), so the in-word shift is
// sh ∈ {0, 8, 16, 24} and the last payload bit sits at sh + 69 <= 93 < 96:
// THREE words suffice, structurally. No shift degenerates:
// the header shift `sh` is at most 24; the golay-rank shift `sh + 10` lies in
// [10, 34]; the plane shift `fs = sh + 22` lies in [22, 46], so `64 - fs`
// lies in [18, 42] — never 0, never 64. The host pads the stream by 4 bytes
// and word-aligns it so the last block's three-word window stays in bounds.
//
// ## No dynamic indexing
//
// As in the other decoders: values come from a predicated tree over two bits,
// never `v[idx]` with a computed idx, and the flag lookup is a register shift
// of a scalar (`flags >> sel`), not an array access. A dynamically indexed
// local array is local memory on the hottest path — `local_size_bytes() == 0`
// is the detector, read at bench startup.

#ifndef LLVQ_GOLAY_CUH
#define LLVQ_GOLAY_CUH

#ifndef LLVQ_PLANES12_CUH
#include "llvq_planes12.cuh"
#endif

#define LLVQ_GOLAY70_STRIDE 9u

// What the Golay70 decoder knows about a class before reading a block.
//
// 32 bytes; the table is allocated at 512 entries so the 9-bit class field
// cannot index out of bounds even from a truncated stream (same reasoning as
// ClassRec). Values are already divided by sqrt(16 * shell).
//
//   * EVEN class: v = { r0[0], r0[1], r2[0], r2[1] } — the residue pairs in
//     canonical level order, single-value residues duplicated. flags = 0.
//   * ODD class:  v = the class's level values, canonical order (<= 4);
//     flags bit k = 1 iff value k ≡ 3 (mod 4).
//   * Exception classes and the origin: all-zero entries — the main stream
//     never names an exception class (it carries the origin record instead),
//     and the origin decodes to zero through them.
struct GolayClassRec {
    float v[4];
    u32   flags;
    u32   is_odd;
    u32   pad0, pad1;
};

struct Golay70Fields {
    u32 id, gain, g, a, bm;
};

// The three-word window, and every field off it.
__device__ __forceinline__ Golay70Fields golay70_fields(const u32* __restrict__ words,
                                                        u32 b)
{
    u32 byte = LLVQ_GOLAY70_STRIDE * b;
    u32 w  = byte >> 2;
    u32 w0 = words[w], w1 = words[w + 1], w2 = words[w + 2];
    u32 sh = (byte & 3u) * 8u;  // 0, 8, 16 or 24
    u64 lo = ((u64)w1 << 32) | (u64)w0;

    Golay70Fields f;
    u32 hdr = (u32)(lo >> sh) & 0x3ffu;
    f.id   = hdr & 0x1ffu;
    f.gain = hdr >> 9;
    f.g    = (u32)(lo >> (sh + 10u)) & 0xfffu;  // sh+10 in [10, 34]
    u32 fs = sh + 22u;                          // [22, 46]; 64-fs in [18, 42]
    u64 pay = (lo >> fs) | ((u64)w2 << (64u - fs));
    f.a  = (u32)pay & 0xffffffu;
    f.bm = (u32)(pay >> 24) & 0xffffffu;
    return f;
}

// The per-block decode plan — the v2 prologue (see the header comment).
//
// Four value registers plus three 24-bit words: after this, no slot needs
// to know which coset it is on. ~12 integer ops, amortized over 24 slots.
struct Golay70Dec {
    float v0, v1, v2, v3;
    u32 hw, aw, nw;
};

__device__ __forceinline__ Golay70Dec golay70_prologue(const GolayClassRec& r,
                                                       u32 cw,
                                                       const Golay70Fields& f)
{
    // m_k: 24-bit broadcast of flag bit k. Even classes have flags = 0, so
    // every mask — and fw with them — collapses to zero without a branch.
    u32 m0 = (r.flags & 1u) ? 0xffffffu : 0u;
    u32 m1 = (r.flags & 2u) ? 0xffffffu : 0u;
    u32 m2 = (r.flags & 4u) ? 0xffffffu : 0u;
    u32 m3 = (r.flags & 8u) ? 0xffffffu : 0u;
    u32 t_lo = (m1 & f.a) | (m0 & ~f.a);      // flag[(0<<1)|abit], per bit
    u32 t_hi = (m3 & f.a) | (m2 & ~f.a);      // flag[(1<<1)|abit], per bit
    u32 fw   = (t_hi & f.bm) | (t_lo & ~f.bm); // flag[sel_j] at every j

    bool odd = r.is_odd != 0u;
    Golay70Dec d;
    d.v0 = r.v[0];
    d.v1 = r.v[1];
    d.v2 = r.v[2];
    d.v3 = r.v[3];
    d.hw = odd ? f.bm : cw;
    d.aw = f.a;
    d.nw = odd ? (cw ^ fw) : f.bm;
    return d;
}

// One slot's signed value — three immediate mask tests, the Planes14
// predicated tree, one negation. No memory access, no variable shift, no
// coset select: the per-slot path is coset-blind by construction.
__device__ __forceinline__ float golay70_slot_value(const Golay70Dec& d, u32 j)
{
    u32 bj  = 1u << j;
    float v = (d.hw & bj) ? ((d.aw & bj) ? d.v3 : d.v2)
                          : ((d.aw & bj) ? d.v1 : d.v0);
    return (d.nw & bj) ? -v : v;
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
__device__ __forceinline__ float golay70_dot(const u32* __restrict__ words,
                                             const u32* __restrict__ cwtab,
                                             const GolayClassRec* __restrict__ gtab,
                                             const float* __restrict__ gscale,
                                             u32 b,
                                             const float* xb)
{
    Golay70Fields f = golay70_fields(words, b);
    const GolayClassRec r = gtab[f.id];
    Golay70Dec dec = golay70_prologue(r, cwtab[f.g], f);

    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
#pragma unroll
    for (u32 i = 0; i < LLVQ_DIM; i += 4) {
#pragma unroll
        for (u32 k = 0; k < 4; ++k) {
            u32 j   = i + k;
            float v = golay70_slot_value(dec, j);
            if (k == 0)      d0 = __fmaf_rn(v, xb[j], d0);
            else if (k == 1) d1 = __fmaf_rn(v, xb[j], d1);
            else if (k == 2) d2 = __fmaf_rn(v, xb[j], d2);
            else             d3 = __fmaf_rn(v, xb[j], d3);
        }
    }
    return ((d0 + d1) + (d2 + d3)) * gscale[f.gain];
}

// ---------------------------------------------------------------------------
// The correction pass's pieces — factored out of tv_golay70 so the host
// harness (tests/host_golay70.cpp) executes the *same text* the kernel runs,
// lane by lane; warp_sum and atomicAdd are the only parts it cannot.
//
// Exceptions reuse Planes12x's machinery verbatim: a u32 block index plus the
// exact 14-byte Planes14 record, located by planes12x_locate and read by
// planes_fields. The one difference from Planes12x is what the main stream
// holds at an exception: the ORIGIN, not an approximation — so the correction
// adds the exact contribution and re-reads nothing from the main stream.
// ---------------------------------------------------------------------------

// One lane's product for exception e: the exact record's slot value (3-plane
// tree over the Planes14 record) times x[lane]. Lanes 24..31 contribute zero.
// The sign is applied here; gain and rscale after the warp reduction.
__device__ __forceinline__ float golay70_exc_lane_term(const PlanesFields& fe,
                                                       const ClassRec& re,
                                                       u32 lane,
                                                       const float* __restrict__ xb)
{
    float t = 0.0f;
    if (lane < LLVQ_DIM) {
        u32 bj = 1u << lane;
        float vlo = (fe.p1 & bj) ? ((fe.p0 & bj) ? re.vals[3] : re.vals[2])
                                 : ((fe.p0 & bj) ? re.vals[1] : re.vals[0]);
        float v = (fe.p2 & bj) ? re.vals[4] : vlo;
        v = (fe.smask & bj) ? -v : v;
        t = v * xb[lane];
    }
    return t;
}

// The value the correction adds to y[row]: the exact contribution under its
// own gain centroid, times the row scale. No approximate term to subtract —
// the main stream carries the origin at every exception by construction, and
// the CPU reference decoder asserts exactly that invariant.
__device__ __forceinline__ float golay70_exc_combine(float ve,
                                                     const float* __restrict__ gscale,
                                                     u32 ge,
                                                     float rs)
{
    return ve * gscale[ge] * rs;
}

#endif  // LLVQ_GOLAY_CUH
