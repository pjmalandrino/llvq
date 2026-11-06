// attn_dec.metal: split-KV grouped decode and verify attention for Qwen3
// (head_dim 128) on Metal, the twin of `llvq-cuda/kernels/attn_dec.cu`.
//
// SELF-CONTAINED: its own translation unit, compiled by the host with
// MTLMathMode::Safe and precise functions (`attn_metal.rs`), never with the
// fast-math defaults the matvec units take. Fast math would allow an
// approximate division and would let the compiler assume no infinities, and
// the arithmetic below tests `m == -inf`.
//
// THE ARITHMETIC IS CANONICAL (bit-relevant): `llvq-cuda/src/attn_ref.rs`,
// which `llvq-llm` includes as `attn::canon`. This text must return its bits,
// and `llvq-llm/tests/attn_metal.rs` demands it. Summarised:
//   - 16 lanes own one (row, query head); lane l owns dims 8l..8l+7; its
//     partial dot is an fma chain over them from +0;
//   - the score is the xor butterfly 8, 4, 2, 1 over the 16 partials, then
//     one RN product by c = scale x log2(e), rounded to f32 on the host;
//   - a tile is 8 positions aligned to absolute positions: tile maximum,
//     carry ex2(m_old - m_new), p = ex2(s - m_new), their sum as a left fold
//     from +0, d <- d*carry + ts, each accumulator <- acc*carry then an fma
//     chain over the tile's positions in order; positions at or past the
//     row's length do no arithmetic;
//   - a unit is 64 positions, tiles in order; a split is 256 positions
//     aligned to multiples of 256, units folded left to right; a row folds
//     its splits left to right; a merge is two RN products and one RN sum
//     per term;
//   - the output is f16 round-to-nearest-even of acc / d (IEEE division);
//   - exp2 is ex2_fixed, a fixed polynomial with an exponent add.
//
// THE MAPPING IS NOT (bit-neutral), and it is not the CUDA kernel's: Metal
// has no grid-wide fence pattern to lean on, so the combine is a second
// dispatch instead of a last-CTA ticket.
//   - attn_split_metal: one threadgroup per (split, query head, row), 64
//     threads = 4 groups of 16 lanes, group g computing unit g of the split.
//     Every group walks its 8 tiles and 8 positions a tile with uniform
//     control flow, so the butterfly's shuffles never read an inactive lane;
//     a position past the row's length reads no memory and changes no state.
//     Group 0 folds the four unit states in order and writes the split state.
//   - attn_merge_metal: one threadgroup of 16 threads per (query head, row),
//     the split states folded in order, then the f16 output.
// A row's bits depend on its length alone.

#include <metal_stdlib>
using namespace metal;

#pragma clang fp contract(off)

#define ATTN_HD 128u
#define ATTN_DPL 8u
#define ATTN_TT 8u
#define ATTN_SPLIT 256u
#define ATTN_UNIT 64u
#define ATTN_UPS 4u
#define ATTN_PSTRIDE 130u

// EX2_COEF_BITS of attn_ref.rs, as bits: no decimal round trip.
constant uint ATTN_EX2C[6] = { 0x3f317217u, 0x3e75feb3u, 0x3d634139u, 0x3c1eab8fu, 0x3aa228dbu, 0x39665cf6u };

/// 2^x for x <= 0, `ex2_fixed` of attn_ref.rs.
inline float attn_ex2(float x)
{
    if (isnan(x)) {
        return x;
    }
    if (!(x > -126.0f)) {
        return 0.0f;
    }
    float fl = floor(x);
    float f = x - fl;
    float p = as_type<float>(ATTN_EX2C[5]);
    p = fma(p, f, as_type<float>(ATTN_EX2C[4]));
    p = fma(p, f, as_type<float>(ATTN_EX2C[3]));
    p = fma(p, f, as_type<float>(ATTN_EX2C[2]));
    p = fma(p, f, as_type<float>(ATTN_EX2C[1]));
    p = fma(p, f, as_type<float>(ATTN_EX2C[0]));
    p = fma(p, f, 1.0f);
    int e = int(fl);
    return as_type<float>(as_type<uint>(p) + (uint(e) << 23));
}

inline float attn_h2f(ushort h)
{
    return float(as_type<half>(h));
}

/// f32 to binary16 bits, round to nearest even: `f2h` of attn_ref.rs, in
/// integers, so no conversion instruction's rounding is assumed.
inline ushort attn_f2h(float x)
{
    uint b = as_type<uint>(x);
    uint sign = (b >> 16) & 0x8000u;
    int exp = int((b >> 23) & 0xffu);
    uint man = b & 0x7fffffu;
    if (exp == 255) {
        return ushort(sign | 0x7c00u | (man != 0u ? 0x200u : 0u));
    }
    int e16 = exp - 127 + 15;
    if (e16 >= 31) {
        return ushort(sign | 0x7c00u);
    }
    if (e16 <= 0) {
        if (e16 < -10) {
            return ushort(sign);
        }
        uint m = man | 0x800000u;
        uint shift = uint(14 - e16);
        uint half_ = 1u << (shift - 1u);
        uint v = m >> shift;
        uint rem = m & ((1u << shift) - 1u);
        if (rem > half_ || (rem == half_ && (v & 1u) == 1u)) {
            v += 1u;
        }
        return ushort(sign | v);
    }
    uint v = sign | (uint(e16) << 10) | (man >> 13);
    uint rem = man & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (v & 1u) == 1u)) {
        v += 1u;
    }
    return ushort(v);
}

struct AttnArgs {
    uint n_heads;
    uint groups;
    uint rows;
    uint pos0;
    uint nsplit;
    uint k_hs;   // K head stride, elements
    uint k_ps;   // K position stride, elements
    uint v_hs;
    uint v_ps;
    float c;
};

kernel void attn_split_metal(const device ushort* q    [[buffer(0)]],
                             const device ushort* k    [[buffer(1)]],
                             const device ushort* v    [[buffer(2)]],
                             device float*        part [[buffer(3)]],
                             constant AttnArgs&   a    [[buffer(4)]],
                             uint3 tg  [[threadgroup_position_in_grid]],
                             uint3 tid3 [[thread_position_in_threadgroup]])
{
    uint tid = tid3.x;
    threadgroup float um[ATTN_UPS];
    threadgroup float ud[ATTN_UPS];
    threadgroup float ua[ATTN_UPS][ATTN_HD];

    uint s = tg.x;
    uint hq = tg.y;
    uint r = tg.z;
    uint g = tid >> 4;
    uint lane = tid & 15u;
    uint len = a.pos0 + r + 1u;
    uint h = hq / a.groups;
    uint d0 = lane * ATTN_DPL;

    float qf[ATTN_DPL];
    const device ushort* qrow = q + (hq * a.rows + r) * ATTN_HD + d0;
    for (uint i = 0u; i < ATTN_DPL; ++i) {
        qf[i] = attn_h2f(qrow[i]);
    }
    const device ushort* kh = k + h * a.k_hs + d0;
    const device ushort* vh = v + h * a.v_hs + d0;

    float m = -INFINITY;
    float d = 0.0f;
    float acc[ATTN_DPL];
    for (uint i = 0u; i < ATTN_DPL; ++i) {
        acc[i] = 0.0f;
    }

    uint u0 = s * ATTN_SPLIT + g * ATTN_UNIT;
    for (uint t = 0u; t < ATTN_UNIT / ATTN_TT; ++t) {
        uint p0 = u0 + t * ATTN_TT;
        float sc[ATTN_TT];
        for (uint j = 0u; j < ATTN_TT; ++j) {
            uint p = p0 + j;
            float part_dot = 0.0f;
            if (p < len) {
                const device ushort* kr = kh + p * a.k_ps;
                for (uint i = 0u; i < ATTN_DPL; ++i) {
                    part_dot = fma(qf[i], attn_h2f(kr[i]), part_dot);
                }
            }
            // Uniform across the simdgroup: every lane shuffles every time.
            part_dot = part_dot + simd_shuffle_xor(part_dot, ushort(8));
            part_dot = part_dot + simd_shuffle_xor(part_dot, ushort(4));
            part_dot = part_dot + simd_shuffle_xor(part_dot, ushort(2));
            part_dot = part_dot + simd_shuffle_xor(part_dot, ushort(1));
            sc[j] = part_dot * a.c;
        }
        if (p0 < len) {
            uint nv = min(ATTN_TT, len - p0);
            float mt = sc[0];
            for (uint j = 1u; j < nv; ++j) {
                mt = max(mt, sc[j]);
            }
            float m_new;
            float carry;
            if (m == -INFINITY) {
                m_new = mt;
                carry = 0.0f;
            } else {
                m_new = max(m, mt);
                carry = attn_ex2(m - m_new);
            }
            float pj[ATTN_TT];
            float ts = 0.0f;
            for (uint j = 0u; j < nv; ++j) {
                pj[j] = attn_ex2(sc[j] - m_new);
                ts = ts + pj[j];
            }
            d = d * carry + ts;
            for (uint i = 0u; i < ATTN_DPL; ++i) {
                float x = acc[i] * carry;
                for (uint j = 0u; j < nv; ++j) {
                    x = fma(pj[j], attn_h2f(vh[(p0 + j) * a.v_ps + i]), x);
                }
                acc[i] = x;
            }
            m = m_new;
        }
    }

    if (lane == 0u) {
        um[g] = m;
        ud[g] = d;
    }
    for (uint i = 0u; i < ATTN_DPL; ++i) {
        ua[g][d0 + i] = acc[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (g == 0u) {
        float sm = -INFINITY;
        float sd = 0.0f;
        float sa[ATTN_DPL];
        for (uint i = 0u; i < ATTN_DPL; ++i) {
            sa[i] = 0.0f;
        }
        for (uint u = 0u; u < ATTN_UPS; ++u) {
            float bm = um[u];
            if (bm == -INFINITY) {
                continue;
            }
            if (sm == -INFINITY) {
                sm = bm;
                sd = ud[u];
                for (uint i = 0u; i < ATTN_DPL; ++i) {
                    sa[i] = ua[u][d0 + i];
                }
                continue;
            }
            float mm = max(sm, bm);
            float fa = attn_ex2(sm - mm);
            float fb = attn_ex2(bm - mm);
            sd = sd * fa + ud[u] * fb;
            for (uint i = 0u; i < ATTN_DPL; ++i) {
                sa[i] = sa[i] * fa + ua[u][d0 + i] * fb;
            }
            sm = mm;
        }
        device float* out = part + ((hq * a.rows + r) * a.nsplit + s) * ATTN_PSTRIDE;
        for (uint i = 0u; i < ATTN_DPL; ++i) {
            out[d0 + i] = sa[i];
        }
        if (lane == 0u) {
            out[ATTN_HD] = sm;
            out[ATTN_HD + 1u] = sd;
        }
    }
}

kernel void attn_merge_metal(const device float* part [[buffer(0)]],
                             device ushort*      o    [[buffer(1)]],
                             constant AttnArgs&  a    [[buffer(2)]],
                             uint2 tg   [[threadgroup_position_in_grid]],
                             uint2 lane2 [[thread_position_in_threadgroup]])
{
    uint lane = lane2.x;
    uint hq = tg.x;
    uint r = tg.y;
    uint d0 = lane * ATTN_DPL;
    float tm = -INFINITY;
    float td = 0.0f;
    float ta[ATTN_DPL];
    for (uint i = 0u; i < ATTN_DPL; ++i) {
        ta[i] = 0.0f;
    }
    const device float* rowp = part + (hq * a.rows + r) * a.nsplit * ATTN_PSTRIDE;
    for (uint s = 0u; s < a.nsplit; ++s) {
        const device float* b = rowp + s * ATTN_PSTRIDE;
        float bm = b[ATTN_HD];
        if (bm == -INFINITY) {
            continue;
        }
        if (tm == -INFINITY) {
            tm = bm;
            td = b[ATTN_HD + 1u];
            for (uint i = 0u; i < ATTN_DPL; ++i) {
                ta[i] = b[d0 + i];
            }
            continue;
        }
        float mm = max(tm, bm);
        float fa = attn_ex2(tm - mm);
        float fb = attn_ex2(bm - mm);
        td = td * fa + b[ATTN_HD + 1u] * fb;
        for (uint i = 0u; i < ATTN_DPL; ++i) {
            ta[i] = ta[i] * fa + b[d0 + i] * fb;
        }
        tm = mm;
    }
    device ushort* orow = o + (hq * a.rows + r) * ATTN_HD + d0;
    for (uint i = 0u; i < ATTN_DPL; ++i) {
        orow[i] = attn_f2h(precise::divide(ta[i], td));
    }
}
