// =============================================================================
// binomial_block.metal — the `marche-bloc` arm of P1b
//
// Pre-registration: proofs/preregistration-p1b-2026-08-15.md
// CPU reference:    llvq-search/src/cns.rs::cns_decode, already swept against
//                   FastDecoder::decode on the 150,681,600 blocks of the sealed
//                   4B (P5 C2, zero discrepancy). Not a new yardstick — the one
//                   P5 proved.
//
// -----------------------------------------------------------------------------
// 1. WHAT THIS DECODES THAT `binomial_walk.metal` DOES NOT
// -----------------------------------------------------------------------------
// `decode_walk` decodes ONE 24-slot walk. Its own §9 said so and called the
// rest ABSENT. A real E1v block of the **even** coset is:
//
//   * a Golay codeword, looked up by class weight and rank;
//   * TWO walks — the support (w slots) and its complement (24 − w);
//   * three sign rules, one per situation, selected by arithmetic;
//   * the parity repair on the last support slot.
//
// The odd coset stays one 24-slot walk, but even there this file does what
// `decode_walk` does not: it looks the codeword up and derives each sign from
// it, instead of reading a 24-bit sign mask handed over by the host.
//
// 🚨 So this arm does **strictly more work per block**, and the journal must
// never quote it as the cost of a walk. It is the cost of a BLOCK, which is the
// quantity P1's thresholds name and the quantity nobody had measured.
//
// -----------------------------------------------------------------------------
// 2. WHAT IS HELD IDENTICAL TO THE WALK ARM, ON PURPOSE
// -----------------------------------------------------------------------------
//   * the record stride — three aligned `uint`, 96 bits, `decode_f96`'s
//     addressing. The widest CNS record is 56 bits (class 323, 8 fields), so
//     the payload fits with room to spare and the two arms pay the same
//     address bill;
//   * the binomial table, transposed, `constant`, `bt[c·25 + p] = C(p, c)`;
//   * the activation staged once per threadgroup, one float out per block;
//   * the cursor discipline (`take`), including the legal zero-width field —
//     a radix-1 kind reads nothing, and a shift by 64 on a `ulong` would be
//     undefined.
//
// An arm that changed any of those would be measuring the change.
//
// -----------------------------------------------------------------------------
// 3. THE RECORD, AND WHY THE HOST PACKS THE WIDTHS
// -----------------------------------------------------------------------------
//   [class 9][gain 1][golay ⌈log₂ γ(w)⌉][on: one field per kind but the last]
//   [off: idem][word signs w−1][free signs free_n]
//
// The field widths are a function of the class, so the shader reads them out of
// the class record rather than recomputing `⌈log₂⌉` per block — the same choice
// `decode_walk` makes with `wbits`. What the host may not do is pack the
// *values* differently from what the shader unpacks, which is why the mirror
// below is byte-for-byte and asserted on the Rust side.
//
// -----------------------------------------------------------------------------
// 4. NO TIMING, NO THRESHOLD, NO CLAIM
// -----------------------------------------------------------------------------
// Nothing here predicts a nanosecond. Operation counts read off this source
// have already been wrong by a factor of 2 on this project's kernels, twice.
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

// One arrangement: every kind but the last placed by a walk, the last taking
// what is left. Writes the kind index of each slot into `kinds`.
static inline void walk_all(thread Cursor &cur,
                            constant uchar* wb,
                            constant uchar* cnt,
                            uint k,
                            uint freem,
                            constant uint* binom,
                            thread uchar* kinds)
{
    for (uint j = 0u; j + 1u < k; ++j) {
        uint rank = take(cur, wb[j]);   // width 0 for a radix-1 kind: reads nothing
        uint c    = cnt[j];
        if (c == 0u) continue;
        uint taken = walk_kind(rank, c, freem, binom);
        freem ^= taken;
        while (taken != 0u) {
            uint s = ctz(taken);
            taken &= taken - 1u;
            kinds[s] = uchar(j);
        }
    }
    while (freem != 0u) {
        uint s = ctz(freem);
        freem &= freem - 1u;
        kinds[s] = uchar(k - 1u);
    }
}

// ---------------------------------------------------------------------------
// The arm. One block per lane, one float out.
//
//   buffer(0) words   device const uint*    three words per block
//   buffer(1) tab     constant BlockRec*    384 records
//   buffer(2) binom   constant uint*        C(p,c) transposed, 25x25
//   buffer(3) golay   device const uint*    the 4096 codewords, flat
//   buffer(4) gscale  constant float*       gain centroids
//   buffer(5) x       device const float*   24 activations
//   buffer(6) out     device float*         one float per block
// ---------------------------------------------------------------------------
kernel void decode_block(device const uint*   words  [[buffer(0)]],
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
    Cursor cur = { (ulong(w1) << 32) | ulong(w0), w2 };

    uint id   = take(cur, 9u);
    uint gain = take(cur, 1u);
    constant BlockRec &r = tab[id];

    uint w    = r.w;
    bool odd  = ((r.geom >> 1) & 1u) != 0u;
    uint p_req = r.geom & 1u;

    uint gi = take(cur, r.wb_golay);
    uint cw = golay[r.golay_base + gi] & 0xffffffu;

    // ---- the two arrangements -------------------------------------------
    //
    // The support is the codeword's set bits, its complement the rest. On the
    // odd coset `w` is 24 and the second walk sees an empty mask, so the same
    // two calls serve both cosets without a branch on the coset itself.
    uchar kinds_on[DIM], kinds_off[DIM];
    for (uint s = 0u; s < DIM; ++s) { kinds_on[s] = 0u; kinds_off[s] = 0u; }
    uint mask_on  = odd ? 0xffffffu : cw;
    uint mask_off = odd ? 0u : ((~cw) & 0xffffffu);
    walk_all(cur, r.wb_on,  r.cnt_on,  r.k_on,  mask_on,  binom, kinds_on);
    if (!odd) {
        walk_all(cur, r.wb_off, r.cnt_off, r.k_off, mask_off, binom, kinds_off);
    }

    uint r_sw = take(cur, r.wb_sw);
    uint r_sf = take(cur, r.wb_sf);

    // ---- the deposit, and the three sign rules ---------------------------
    float dot = 0.0f;
    uint s_on = 0u, par = 0u;
    for (uint s = 0u; s < DIM; ++s) {
        bool on = ((cw >> s) & 1u) != 0u;
        float v_on  = r.vals_on[kinds_on[s]];
        float v_off = r.vals_off[kinds_off[s]];

        // odd: forced by the codeword and the value's residue mod 4. The host
        // stores the parity of |value| in the sign bit of `vals_on` so the rule
        // is one XOR here too — see p1host, which asserts it.
        uint sign_odd = (((cw >> s) & 1u) ^ (as_type<uint>(v_on) >> 31u)) << 31u;

        // even, support: a free bit, except the last slot, which carries the
        // parity repair — the bit the class does not spend.
        bool last_on = (s_on + 1u == w);
        uint bfree   = (r_sw >> s_on) & 1u;
        uint sign_a  = (last_on ? (par ^ p_req) : bfree) << 31u;
        // 🚨 `par` accumulates over SUPPORT slots only. The first version
        // XORed it on every slot, and since `s_on` does not advance off the
        // support, an off-support slot folded the SAME bit in a second time and
        // flipped the parity. V0 caught it on 4,149,448 blocks of 16,777,216 —
        // half the even ones, which is exactly the shape of the defect.
        par ^= (on && !last_on) ? bfree : 0u;

        // even, off-support: the next free bit, consumed only by a nonzero
        // value. The zero run is marked by its VALUE, never by its index.
        bool nzv    = (v_off != 0.0f);
        uint sign_b = ((r_sf & 1u) & uint(nzv)) << 31u;
        r_sf >>= (!on && nzv) ? 1u : 0u;

        // ⚠️ On the odd coset EVERY slot's magnitude comes from `vals_on`: its
        // single walk spans all 24, and `cw` says which residue a slot carries,
        // not which table it lives in. Reading `vals_off` off the codeword there
        // would silently zero every off-codeword coordinate — half of them.
        float mag  = (odd || on) ? fabs(v_on) : fabs(v_off);
        uint  sign = odd ? sign_odd : (on ? sign_a : sign_b);
        float wv   = as_type<float>(as_type<uint>(mag) ^ sign);
        dot = fma(wv, xs[s], dot);
        s_on += on ? 1u : 0u;
    }

    out[gid] = dot * gscale[gain];
}
