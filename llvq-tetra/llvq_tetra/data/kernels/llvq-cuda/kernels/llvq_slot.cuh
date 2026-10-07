// The Slot32 block decoder, ported from `llvq_metal::PAYLOAD_MSL::slot_dot`.
//
// One block is 24 weights. Its record, from bit 0:
//
//     [ class : 9 ][ gain : 1 ][ smask : 24 ][ m1 : 24 ] … [ m_{L-1} : 24 ]
//
// Level 0 is implicit — the complement of the union of the other masks — and
// the masks are disjoint by construction of the encoder. `smask` bit i is the
// sign of *slot* i, not of the i-th nonzero. Every field sits at a fixed
// offset, which is the entire point of this layout: no popcount chain, no
// serial state, no per-level loop, zero divergence.
//
// The bit order is LSB-first within a byte, bytes ascending, so reading the
// byte stream as `const unsigned*` on a little-endian host gives exactly the
// words the Metal shader reads. That is the one place in this port where a
// mistake would pass every CPU test and break only on the GPU.
//
// No CUDA headers are included on purpose: `nvidia/cuda:*-runtime` ships no
// `-dev` package, so `/usr/local/cuda/include` is empty and `<cuda_fp16.h>`
// would not resolve. Nothing here needs it.

#ifndef LLVQ_SLOT_CUH
#define LLVQ_SLOT_CUH

#define LLVQ_DIM 24u

typedef unsigned int       u32;
typedef unsigned long long u64;

// What the decoder knows about a class before reading a block.
//
// The Metal record carries eight more fields; they serve the *other* layouts'
// decoders, which this port deliberately leaves behind. 24 bytes, and the
// table is allocated at 512 entries rather than 384 so that the 9-bit class
// field cannot index out of bounds even from a truncated file — on Metal an
// over-read is benign, here it is an illegal address that kills the context
// in the middle of a billed job.
struct ClassRec {
    float vals[5];   // level values already divided by sqrt(16 * shell)
    u32   len;       // levels in the class; 1 for the origin
};

// 24 bits at a fixed offset of a 128-bit register pair.
//
// The offset is a template parameter, not an argument. The Metal original
// takes it as a value and relies on the compiler folding a `?:` over literals;
// that is a property of the optimiser, not of the language. Here the branch is
// resolved by `if constexpr`, so the shift that would be out of range is never
// instantiated. Called only at 24, 48, 72 and 96.
template <u32 OFF>
__device__ __forceinline__ u32 ext24(u64 lo, u64 hi)
{
    if constexpr (OFF < 64u) {
        return (u32)((lo >> OFF) | (hi << (64u - OFF))) & 0xffffffu;
    } else {
        return (u32)(hi >> (OFF - 64u)) & 0xffffffu;
    }
}

// Where block `b` starts, in bytes.
//
// Blocks are grouped by 32 in flat order. `bases` has ngroups+1 entries, the
// last equal to the stream length, and every base is a multiple of 32, so the
// byte shift below is at most 24.
//
// A warp does NOT cover a group: nblocks is not a multiple of 32 (2560/24 =
// 106, 9728/24 = 405), so for most rows the 32 blocks a warp holds straddle
// two groups with two bases and two strides. Any optimisation that hoists
// `bases[g]` or `stride` out of the loop assuming uniformity is wrong.
__device__ __forceinline__ u32 slot_byte(const u32* __restrict__ bases, u32 b)
{
    u32 g      = b >> 5;
    u32 base   = bases[g];
    u32 stride = (bases[g + 1] - base) >> 5;
    return base + (b & 31u) * stride;
}

// The five-word window, and the header read in place.
//
// Worst case is a 24-bit alignment shift plus a 130-bit payload = 154 <= 160.
// That inequality is asserted host-side (`ClassTable::worst_width_slot`); it
// is not re-derivable here, and it is what makes the fifth word enough.
//
// `fs = sh + 10` lies in [10, 34], so `64 - fs` lies in [30, 54] — never 0,
// never 64. The shifts cannot degenerate.
struct SlotFields {
    u32 id, gain, smask, m1, m2, m3, m4;
};

__device__ __forceinline__ SlotFields slot_fields(const u32* __restrict__ words,
                                                  const ClassRec* __restrict__ tab,
                                                  u32 byte,
                                                  u32* out_len)
{
    u32 w  = byte >> 2;
    u32 w0 = words[w], w1 = words[w + 1], w2 = words[w + 2],
        w3 = words[w + 3], w4 = words[w + 4];
    u32 sh = (byte & 3u) * 8u;
    u64 lo = ((u64)w1 << 32) | (u64)w0;
    u64 hi = ((u64)w3 << 32) | (u64)w2;

    SlotFields f;
    u32 hdr = (u32)(lo >> sh) & 0x3ffu;
    f.id    = hdr & 0x1ffu;
    f.gain  = hdr >> 9;
    u32 fs  = sh + 10u;
    u64 pay_lo = (lo >> fs) | (hi << (64u - fs));
    u64 pay_hi = (hi >> fs) | ((u64)w4 << (64u - fs));

    u32 nlev = tab[f.id].len;
    *out_len = nlev;
    f.smask  = (u32)pay_lo & 0xffffffu;
    f.m1 = (nlev > 1) ? ext24<24u>(pay_lo, pay_hi) : 0u;
    f.m2 = (nlev > 2) ? ext24<48u>(pay_lo, pay_hi) : 0u;
    f.m3 = (nlev > 3) ? ext24<72u>(pay_lo, pay_hi) : 0u;
    f.m4 = (nlev > 4) ? ext24<96u>(pay_lo, pay_hi) : 0u;
    return f;
}

// Which level slot `j` belongs to. Level 0 is the default, so a class with
// fewer levels simply has zero masks and every slot falls through to it.
__device__ __forceinline__ u32 slot_level(const SlotFields& f, u32 j)
{
    u32 bj = 1u << j;
    return (f.m1 & bj) ? 1u : (f.m2 & bj) ? 2u
         : (f.m3 & bj) ? 3u : (f.m4 & bj) ? 4u : 0u;
}

// One block's contribution to a dot product, activations in `xb`.
//
// The origin needs no special case: entry 0 of the table has len = 1 and
// vals = {0,…}, so all four masks are forced to zero, every slot takes 0.0f,
// and the sign bit produces -0.0f, which is additively neutral. The Rust
// decoder returns early instead; the divergence is apparent, not real.
//
// Four independent FMA chains, the accumulator chosen by the slot's residue
// mod 4, and an explicit final parenthesisation — the same shape and the same
// association as the Metal original, so the two return values are comparable.
// `#pragma unroll` is not decoration: without it `d[k]` becomes a dynamically
// indexed array, hence local memory, on the hottest path. `local_size_bytes()`
// is read at startup as the detector.
__device__ __forceinline__ float slot_dot(const u32* __restrict__ words,
                                          const u32* __restrict__ bases,
                                          const ClassRec* __restrict__ tab,
                                          const float* __restrict__ gscale,
                                          u32 b,
                                          const float* xb)
{
    u32 len;
    SlotFields f = slot_fields(words, tab, slot_byte(bases, b), &len);
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
            float v = (f.m1 & bj) ? v1 : (f.m2 & bj) ? v2 : (f.m3 & bj) ? v3
                    : (f.m4 & bj) ? v4 : v0;
            v = (f.smask & bj) ? -v : v;
            if (k == 0)      d0 = __fmaf_rn(v, xb[j], d0);
            else if (k == 1) d1 = __fmaf_rn(v, xb[j], d1);
            else if (k == 2) d2 = __fmaf_rn(v, xb[j], d2);
            else             d3 = __fmaf_rn(v, xb[j], d3);
        }
    }
    return ((d0 + d1) + (d2 + d3)) * gscale[f.gain];
}

#endif  // LLVQ_SLOT_CUH
