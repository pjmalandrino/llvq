// The Planes14 block decoder — layout E1a of docs/pistes-format-vram-2026-08-05.md.
//
// One record per block, uniform 14-byte stride, offset = 14·b, **no bases
// array**. From bit 0, LSB-first within a byte, bytes ascending — the same
// convention as Slot32, so a little-endian `const u32*` read gives the words
// directly:
//
//     [ class : 9 ][ gain : 1 ][ smask : 24 ][ plane0 : 24 ][ plane1 : 24 ][ plane2 : 24 ][ 0 : 6 ]
//
// The level of slot j is `plane0[j] | plane1[j]<<1 | plane2[j]<<2`, an
// explicit 0..len-1 index into the same ClassRec.vals[] Slot32 uses. Where
// Slot32 leaves level 0 implicit (the complement of the mask union), Planes14
// encodes it as three clear bits — so an origin block (len = 1) is a header, a
// zero sign mask and three zero planes. `smask` bit j is the sign of slot j,
// exactly as in Slot32: zero slots carry a written-0 bit, and a sign on a
// zero value would at worst produce -0.0f, additively neutral.
//
// 112 bits per block = 4.6667 b/weight of payload, and the stream is a pure
// bijection of the Slot32 content — same classes, same gains, same signs,
// same levels — so the arms decode identical weights.
//
// ## The read window
//
// The byte offset 14·b is ≡ 0 or 2 (mod 4), so the in-word shift is 0 or 16
// bits, and 16 + 106 payload bits = 122 <= 128: **four** words suffice, where
// Slot32 needs five. The bound is structural — a constant of the layout —
// unlike Slot32's five-word inequality, which depends on the class table and
// must be re-asserted host-side per table. The host still pads the stream so
// the last block's four-word window stays in bounds (up to 2 bytes past
// 14·nblocks).
//
// ## No dynamic indexing
//
// Values are selected by a predicated tree over the three plane bits, never
// by `vals[idx]` with a computed idx: a dynamically indexed array is local
// memory on the hottest path, the exact failure `local_size_bytes() == 0`
// guards against (see llvq_slot.cuh on `#pragma unroll`).

#ifndef LLVQ_PLANES_CUH
#define LLVQ_PLANES_CUH

#ifndef LLVQ_SLOT_CUH
#include "llvq_slot.cuh"
#endif

#define LLVQ_PLANES_STRIDE 14u

struct PlanesFields {
    u32 id, gain, smask, p0, p1, p2;
};

// The four-word window, and every field off it.
//
// `fs = sh + 10` is 10 or 26 — never 0, never 64 — so no shift degenerates.
// `pay_hi` keeps whatever the next record put in the top bits; `ext24` (from
// llvq_slot.cuh) masks it off. Bits 106..111 are zero by construction and
// never read.
__device__ __forceinline__ PlanesFields planes_fields(const u32* __restrict__ words,
                                                      u32 b)
{
    u32 byte = LLVQ_PLANES_STRIDE * b;
    u32 w  = byte >> 2;
    u32 w0 = words[w], w1 = words[w + 1], w2 = words[w + 2], w3 = words[w + 3];
    u32 sh = (byte & 3u) * 8u;  // 0 or 16: 14·b mod 4 ∈ {0, 2}
    u64 lo = ((u64)w1 << 32) | (u64)w0;
    u64 hi = ((u64)w3 << 32) | (u64)w2;

    PlanesFields f;
    u32 hdr = (u32)(lo >> sh) & 0x3ffu;
    f.id    = hdr & 0x1ffu;
    f.gain  = hdr >> 9;
    u32 fs  = sh + 10u;
    u64 pay_lo = (lo >> fs) | (hi << (64u - fs));
    u64 pay_hi = hi >> fs;
    f.smask = (u32)pay_lo & 0xffffffu;
    f.p0 = ext24<24u>(pay_lo, pay_hi);
    f.p1 = ext24<48u>(pay_lo, pay_hi);
    f.p2 = ext24<72u>(pay_lo, pay_hi);
    return f;
}

// One block's contribution to a dot product — the Planes14 twin of slot_dot.
//
// Same four independent FMA chains, same accumulator selection by residue
// mod 4, same final parenthesisation, same gscale epilogue — so the two
// kernels' outputs are comparable, each against the shared f64 reference.
// (They are *not* asserted bit-equal against each other: two kernels, and the
// compiler may schedule contractions differently.)
//
// The value select is a tree over (p2, p1, p0). ClassRec.len <= 5, so levels
// stop at 4 = 100b: when the p2 bit is set the two low bits are zero by
// construction and the tree needs no arms for 5..7. vals[] of slots past
// `len` are zero in the uploaded table (512 entries, 5 floats each), so even
// a corrupt record selects a zero rather than reading out of bounds.
//
// The table's `len` field is deliberately not read: the level index is
// explicit in the planes, which is the entire point of this layout — no
// `nlev` gating, one fewer serial dependency before the selects.
__device__ __forceinline__ float planes_dot(const u32* __restrict__ words,
                                            const ClassRec* __restrict__ tab,
                                            const float* __restrict__ gscale,
                                            u32 b,
                                            const float* xb)
{
    PlanesFields f = planes_fields(words, b);
    const ClassRec r = tab[f.id];
    float v0 = r.vals[0], v1 = r.vals[1], v2 = r.vals[2],
          v3 = r.vals[3], v4 = r.vals[4];

    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
#pragma unroll
    for (u32 i = 0; i < LLVQ_DIM; i += 4) {
#pragma unroll
        for (u32 k = 0; k < 4; ++k) {
            u32 j  = i + k;
            u32 bj = 1u << j;
            float vlo = (f.p1 & bj) ? ((f.p0 & bj) ? v3 : v2)
                                    : ((f.p0 & bj) ? v1 : v0);
            float v = (f.p2 & bj) ? v4 : vlo;
            v = (f.smask & bj) ? -v : v;
            if (k == 0)      d0 = __fmaf_rn(v, xb[j], d0);
            else if (k == 1) d1 = __fmaf_rn(v, xb[j], d1);
            else if (k == 2) d2 = __fmaf_rn(v, xb[j], d2);
            else             d3 = __fmaf_rn(v, xb[j], d3);
        }
    }
    return ((d0 + d1) + (d2 + d3)) * gscale[f.gain];
}

#endif  // LLVQ_PLANES_CUH
