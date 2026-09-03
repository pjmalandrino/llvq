// =============================================================================
// binomial_block_flat.metal — the `marche-bloc-plat` arm of P1b, amendment É1
//
// Pre-registration: proofs/preregistration-p1b-2026-08-15.md, §7 É1.
//
// Same output as `binomial_block.metal`, same record, same 12-byte stride. What
// changes is the per-slot state: that arm keeps `kinds_on[24]` and
// `kinds_off[24]` — 48 bytes indexed **by the slot**, i.e. by data — and
// `cascade_uniform.metal` §1(d) had already written why that is a mistake here:
//
//   « No dynamic indexing of registers. […] On a GPU a dynamically indexed
//     register array is spilled to thread-local memory; this arm would be
//     measuring a spill. »
//
// This file deposits each kind straight from its `taken` mask, the way
// `decode_walk` does, and computes the two sign ranks by popcount instead of
// reading them out of an array:
//
//   * rank of a slot WITHIN the support   = popcount(cw & ((1<<s) − 1))
//   * rank among the NONZERO off-support  = popcount(nz & ((1<<s) − 1))
//
// 🚨 **It is a REDUCTION, not a suppression, and É1 says so before the
// measurement.** The off-support sign rank counts the nonzero slots *before* a
// slot, so it needs the complete `nz` mask, so it needs every off kind walked
// before the first off deposit. One array survives — the `taken` masks,
// MAX_KINDS = 8 words — but it is indexed by the **loop counter**, not by a
// slot. 48 data-indexed bytes become 8 counter-indexed words.
//
// ⚠️ The two sign fields sit at the END of the record, after both arrangements,
// so a deposit made during the walk has not read them yet. They are peeked at
// an offset the record itself gives (`10 + wb_golay + Σ wb_on + Σ wb_off`). The
// format does not change by one bit — the width is the same, so P5's C1 and the
// packed stream are untouched.
//
// 🚨 It does NOT replace `binomial_block.metal`. That arm has a measured
// verdict, 0.6735 ns/block, and both run in the same round so the gap between
// them is what the spill cost. Deleting the first would delete the comparison.
//
// No timing, no threshold, no claim.
// =============================================================================

#include <metal_stdlib>
using namespace metal;

constant uint DIM          = 24;
constant uint MAX_KINDS    = 8;
constant uint BINOM_STRIDE = DIM + 1;

struct Cursor { ulong lo; uint hi; };

static inline uint take(thread Cursor &c, uint width) {
    if (width == 0) return 0u;
    uint v = uint(c.lo & ((1ul << width) - 1ul));
    c.lo = (c.lo >> width) | (ulong(c.hi) << (64u - width));
    c.hi = c.hi >> width;
    return v;
}

// Mirror of `p1host::GpuBlockRec`, 108 bytes, alignment 4. One per class, plus the origin at
// entry 0 — the id convention of every table in this crate.
struct BlockRec {
    float vals_on[MAX_KINDS];   // support values (even) / class values (odd),
                                // ALREADY divided by sqrt(16·m)
    float vals_off[MAX_KINDS];  // off-support values; the zero run's is 0.0f
    uchar cnt_on[MAX_KINDS];    // support counts; sum = w (even) or 24 (odd)
    uchar cnt_off[MAX_KINDS];   // off-support counts, zero run included
    uchar wb_on[MAX_KINDS];     // ⌈log₂ radix⌉ per kind, 0 for the last
    uchar wb_off[MAX_KINDS];
    uchar k_on, k_off, w, geom; // geom: p_req | odd<<1
    uint  golay_base;           // offset of the weight bucket in the flat table
    uchar wb_golay, wb_sw, wb_sf, pad;
};

// One kind's slots, by colex unranking over the free positions, walked in SLOT
// space. Identical in shape to `binomial_walk.metal`'s inner scan — same
// snapshot discipline, same transposed row, same compare-and-subtract, no
// division. Returns the mask of slots this kind takes.
static inline uint walk_kind(uint rank,
                             uint cnt,
                             uint freem,
                             constant uint* binom)
{
    constant uint* row = binom + cnt * BINOM_STRIDE;
    uint taken = 0u;
    uint p = popcount(freem);
    uint m = freem;
    while (m != 0u) {
        uint s = 31u - clz(m);
        m ^= 1u << s;
        --p;
        uint b = row[p];
        bool hit = (rank >= b) && (cnt != 0u);
        uint one = hit ? 1u : 0u;
        rank  -= hit ? b : 0u;
        row   -= one * BINOM_STRIDE;
        cnt   -= one;
        taken |= hit ? (1u << s) : 0u;
    }
    return taken;
}

// Read `width` bits at `off` out of the 96-bit record, without moving the
// cursor. `off` is always at least 10 — the header precedes every field — so
// the `64 - off` shift below is always defined.
static inline uint peek(ulong lo, uint hi, uint off, uint width) {
    if (width == 0u) return 0u;
    ulong v;
    if (off >= 64u) {
        v = ulong(hi) >> (off - 64u);
    } else {
        v = (lo >> off) | (ulong(hi) << (64u - off));
    }
    return uint(v & ((1ul << width) - 1ul));
}

// ---------------------------------------------------------------------------
// The arm. Same buffers as `decode_block`, same record, same stride.
// ---------------------------------------------------------------------------
kernel void decode_block_flat(device const uint*   words  [[buffer(0)]],
                              constant BlockRec*   tab    [[buffer(1)]],
                              constant uint*       binom  [[buffer(2)]],
                              device const uint*   golay  [[buffer(3)]],
                              constant float*      gscale [[buffer(4)]],
                              device const float*  x      [[buffer(5)]],
                              device float*        out    [[buffer(6)]],
                              uint gid [[thread_position_in_grid]],
                              uint tid [[thread_index_in_threadgroup]])
{
    threadgroup float xs[DIM];
    if (tid < DIM) xs[tid] = x[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint w0 = words[3u * gid], w1 = words[3u * gid + 1u], w2 = words[3u * gid + 2u];
    ulong lo = (ulong(w1) << 32) | ulong(w0);
    Cursor cur = { lo, w2 };

    uint id   = take(cur, 9u);
    uint gain = take(cur, 1u);
    constant BlockRec &r = tab[id];

    uint w     = r.w;
    bool odd   = ((r.geom >> 1) & 1u) != 0u;
    uint p_req = r.geom & 1u;

    // ---- the two sign fields, peeked ahead --------------------------------
    uint pre = uint(r.wb_golay);
    for (uint j = 0u; j < MAX_KINDS; ++j) pre += uint(r.wb_on[j]) + uint(r.wb_off[j]);
    uint r_sw = peek(lo, w2, 10u + pre, r.wb_sw);
    uint r_sf = peek(lo, w2, 10u + pre + uint(r.wb_sw), r.wb_sf);
    // The parity repair is the XOR of every free support bit, and `s_w` is
    // exactly `2^(w−1)`, so `r_sw` holds those bits and nothing else: the
    // accumulator the other arm carries across the loop is one popcount here.
    uint par = uint(popcount(r_sw)) & 1u;

    uint gi = take(cur, r.wb_golay);
    uint cw = golay[r.golay_base + gi] & 0xffffffu;

    float dot = 0.0f;

    // ---- the support (all 24 slots on the odd coset) ----------------------
    {
        uint freem = odd ? 0xffffffu : cw;
        for (uint j = 0u; j < r.k_on; ++j) {
            uint taken;
            if (j + 1u == r.k_on) {
                taken = freem;
                freem = 0u;
            } else {
                uint rank = take(cur, r.wb_on[j]);
                uint c    = r.cnt_on[j];
                if (c == 0u) continue;
                taken = walk_kind(rank, c, freem, binom);
                freem ^= taken;
            }
            float v = r.vals_on[j];
            float mag = fabs(v);
            uint  flag = as_type<uint>(v) >> 31u;   // bit 1 of |value|, odd coset
            uint m = taken;
            while (m != 0u) {
                uint s = ctz(m);
                m &= m - 1u;
                uint neg;
                if (odd) {
                    neg = ((cw >> s) & 1u) ^ flag;
                } else {
                    uint s_on = uint(popcount(cw & ((1u << s) - 1u)));
                    neg = (s_on + 1u == w) ? (par ^ p_req) : ((r_sw >> s_on) & 1u);
                }
                float wv = as_type<float>(as_type<uint>(mag) ^ (neg << 31u));
                dot = fma(wv, xs[s], dot);
            }
        }
    }

    // ---- the off-support, even coset only ---------------------------------
    //
    // Two loops, and the split is the whole point: the first walks every kind
    // and unions the nonzero ones into `nz`, the second deposits. A single loop
    // cannot, because a slot's sign rank counts the nonzero slots BEFORE it and
    // a later kind can own one of those.
    if (!odd) {
        uint tk[MAX_KINDS];
        uint nz = 0u;
        uint freem = (~cw) & 0xffffffu;
        for (uint j = 0u; j < MAX_KINDS; ++j) tk[j] = 0u;
        for (uint j = 0u; j < r.k_off; ++j) {
            uint taken;
            if (j + 1u == r.k_off) {
                taken = freem;
                freem = 0u;
            } else {
                uint rank = take(cur, r.wb_off[j]);
                uint c    = r.cnt_off[j];
                if (c == 0u) continue;
                taken = walk_kind(rank, c, freem, binom);
                freem ^= taken;
            }
            tk[j] = taken;
            // The zero run is marked by its VALUE, never by its index.
            if (r.vals_off[j] != 0.0f) nz |= taken;
        }
        for (uint j = 0u; j < r.k_off; ++j) {
            float v = r.vals_off[j];
            if (v == 0.0f) continue;      // the zero run deposits nothing
            uint m = tk[j];
            while (m != 0u) {
                uint s = ctz(m);
                m &= m - 1u;
                uint fbit = uint(popcount(nz & ((1u << s) - 1u)));
                uint neg  = (r_sf >> fbit) & 1u;
                float wv  = as_type<float>(as_type<uint>(v) ^ (neg << 31u));
                dot = fma(wv, xs[s], dot);
            }
        }
    }

    out[gid] = dot * gscale[gain];
}
