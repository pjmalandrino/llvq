// The E1c block decoder — the bit-sliced twin of Planes14 and Planes12x,
// item X3 of docs/archive/spec-memoire-extreme-2026-08-12.md, arms `e1c14` and
// `e1c12` of proofs/preregistration-p4-2026-08-14.md §2.5.
//
// Same content as the Planes streams — same class ids, same gains, same signs,
// same per-slot levels — with the per-slot bits **transposed over a group of
// 32 blocks**, so that the record's padding disappears:
//
//     Planes14   112 bits/block (106 payload + 6 padding)   4.6667 b/w
//     E1c14      106 bits/block, zero padding               4.4167 b/w
//     Planes12x   96 bits/block ( 82 payload + 14 padding)  4.0000 b/w
//     E1c12       82 bits/block, zero padding               3.4167 b/w
//
// ## The word map, and why the group is 32
//
//     word  0 ..  9        headers, 10 bits per block, block l at bit 10·l
//     word 10 + j·F + f    slot j, field f, bit l = block l's bit
//                          F = 1 + PLANES  (smask, then plane 0 .. n-1)
//
// The group is the **warp**. Bit `l` of a word is lane `l`, so every lane of
// the warp reads the *same word address* — a broadcast out of L1 rather than
// 32 strided reads — and a lane's own bit is `(w >> lane) & 1`: one shift, no
// offset arithmetic.
//
// Slot-major, not field-major, and the reason is this file: a lane walking
// slots 0..24 reads a slot's whole field set from **contiguous** words, four
// for E1c14 and three for E1c12. Field-major would scatter those four 24 words
// apart and turn one load into four.
//
// ## What this trades, and what nothing here can prove
//
// A lane reads 82 or 106 words per group where Planes14 reads four per lane.
// **Whether the extra loads cost less than the bytes they save is the whole
// question, and it is the bench's, not this file's.** The exactness half is
// already settled and not by this file either: `llvq-artifact`'s E1c sweep ran
// the two repacks over the 150 681 600 blocks of the sealed 4B against the
// archive decoder (docs/mesures/e1c-sweep-4b-2026-08-12.txt). This kernel has
// to reproduce that contract, not establish it.
//
// ## What has been checked, and where
//
// 🚨 **Nothing in this file has been compiled.** The development machine is a
// Mac: no nvcc, no NVRTC, no card. What *has* been checked is the arithmetic
// below, transcribed by hand into Rust and run against `E1c14Blocks::
// decode_block` on every class plus the origin —
// `llvq-artifact/tests/e1c_cuda_mirror.rs`. That catches a wrong word map, a
// wrong shift, a wrong lane bit: the class of defect that cost two runs on
// 2026-08-15. It cannot catch a syntax error, a launch configuration, a
// register spill or a race. Those surface at job start, which is what
// `preflight` is for.
//
// ⚠️ And the mirror is a **transcription**, not a shared implementation. A
// drift between the two is not detected mechanically; the test says so in its
// own header rather than letting a reader assume otherwise.
//
// ## No dynamic indexing
//
// Same rule as llvq_planes.cuh: values are selected by a predicated tree over
// the plane bits, never by `vals[idx]` with a computed idx. A dynamically
// indexed array is local memory on the hottest path, and `local_size_bytes()
// == 0` is a contract this crate refuses to break.

#ifndef LLVQ_E1C_CUH
#define LLVQ_E1C_CUH

#ifndef LLVQ_SLOT_CUH
#include "llvq_slot.cuh"
#endif

// Blocks per group — the warp, and the reason the transposition exists.
#define LLVQ_E1C_GROUP 32u
// Header bits per block: class 9 + gain 1. Same field split as every Planes
// record, so a header that moved there would have to move here.
#define LLVQ_E1C_HDR_BITS 10u
// Words the header region of one group takes: 10 bits × 32 blocks = 320 = 10
// words, with nothing left over — a consequence of the group being the word
// width, not a coincidence of the constants.
#define LLVQ_E1C_HDR_WORDS ((LLVQ_E1C_HDR_BITS * LLVQ_E1C_GROUP) / 32u)

// Words one group occupies, for a variant with `planes` bit-planes.
__device__ __forceinline__ u32 e1c_group_words(u32 planes) {
    return LLVQ_E1C_HDR_WORDS + 24u * (1u + planes);
}

struct E1cFields {
    u32 id, gain, smask, p0, p1, p2;
};

// One block's fields, out of its group's words.
//
// `PLANES` is a compile-time constant so the two plane branches fold away and
// E1c12 pays for nothing it does not carry. The header window may straddle two
// words (bit 10·l with l up to 31 gives bit 310, word 9, shift 22, so 22+10 =
// 32 exactly — never past); reading `grp[w + 1]` is in range regardless
// because word 10 is the first slot word, and the extra bits are masked off.
template <u32 PLANES>
__device__ __forceinline__ E1cFields e1c_fields(const u32* __restrict__ words,
                                                u32 g,
                                                u32 l)
{
    const u32* grp = words + g * (LLVQ_E1C_HDR_WORDS + 24u * (1u + PLANES));

    u32 bit = LLVQ_E1C_HDR_BITS * l;
    u32 w   = bit >> 5;
    u32 sh  = bit & 31u;
    u64 win = (u64)grp[w] | ((u64)grp[w + 1u] << 32);
    u32 hdr = (u32)(win >> sh) & 0x3ffu;

    E1cFields f;
    f.id    = hdr & 0x1ffu;
    f.gain  = hdr >> 9;
    f.smask = 0u;
    f.p0    = 0u;
    f.p1    = 0u;
    f.p2    = 0u;

    const u32* slots = grp + LLVQ_E1C_HDR_WORDS;
#pragma unroll
    for (u32 j = 0; j < 24u; ++j) {
        const u32* fw = slots + j * (1u + PLANES);
        f.smask |= ((fw[0] >> l) & 1u) << j;
        f.p0    |= ((fw[1] >> l) & 1u) << j;
        if (PLANES > 1u) f.p1 |= ((fw[2] >> l) & 1u) << j;
        if (PLANES > 2u) f.p2 |= ((fw[3] >> l) & 1u) << j;
    }
    return f;
}

// One block's contribution to a dot product — the E1c twin of `planes_dot`.
//
// Four independent FMA chains, accumulator chosen by residue mod 4, same final
// parenthesisation and the same gscale epilogue as `planes_dot`, so the two
// arms' outputs are comparable against the shared f64 reference. They are
// **not** asserted bit-equal to each other: two kernels, and the compiler may
// contract differently.
template <u32 PLANES>
__device__ __forceinline__ float e1c_dot(const u32* __restrict__ words,
                                         const ClassRec* __restrict__ tab,
                                         const float* __restrict__ gscale,
                                         u32 b,
                                         const float* xb)
{
    E1cFields f = e1c_fields<PLANES>(words, b / LLVQ_E1C_GROUP, b % LLVQ_E1C_GROUP);
    const ClassRec r = tab[f.id];
    float v0 = r.vals[0], v1 = r.vals[1], v2 = r.vals[2],
          v3 = r.vals[3], v4 = r.vals[4];

    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
#pragma unroll
    for (u32 j = 0; j < 24u; ++j) {
        u32 b0 = (f.p0 >> j) & 1u;
        u32 b1 = (PLANES > 1u) ? ((f.p1 >> j) & 1u) : 0u;
        u32 b2 = (PLANES > 2u) ? ((f.p2 >> j) & 1u) : 0u;
        // Predicated tree over (b2, b1, b0). ClassRec.len <= 5, so a set b2
        // implies the two low bits are zero and the tree needs no 5..7 arms;
        // `vals` past `len` is zero in the uploaded table, so even a corrupt
        // record selects a zero rather than reading out of bounds.
        float v = b2 ? v4 : (b1 ? (b0 ? v3 : v2) : (b0 ? v1 : v0));
        // Sign by XOR of the float's sign bit — a sign on a zero value gives
        // -0.0f, additively neutral, exactly as in the Planes decoders.
        u32 neg = (f.smask >> j) & 1u;
        float wv = __int_as_float(__float_as_int(v) ^ (neg << 31));
        float xv = xb[j];
        if ((j & 3u) == 0u) d0 = fmaf(wv, xv, d0);
        else if ((j & 3u) == 1u) d1 = fmaf(wv, xv, d1);
        else if ((j & 3u) == 2u) d2 = fmaf(wv, xv, d2);
        else d3 = fmaf(wv, xv, d3);
    }
    return ((d0 + d1) + (d2 + d3)) * gscale[f.gain];
}

#endif // LLVQ_E1C_CUH
