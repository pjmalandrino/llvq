// =============================================================================
// e1v_flux.metal — the `e1v-flux` arm of P1c
//
// Pre-registration: proofs/preregistration-p1c-2026-08-15.md
// CPU reference:    llvq-search/src/cns.rs::cns_decode, already swept against
//                   FastDecoder::decode on the 150,681,600 blocks of the sealed
//                   4B (P5 C2, zero discrepancy). Not a new yardstick — the one
//                   P5 proved.
//
// -----------------------------------------------------------------------------
// 1. WHAT THIS PAYS THAT `binomial_block.metal` DOES NOT
// -----------------------------------------------------------------------------
// Nothing, in the decode. The decode body of this file is BYTE-IDENTICAL to
// that arm's, and `llvq-metal/tests/e1v_flux.rs` asserts it on both files'
// text: if the two ever drift, the gap between them stops being the price of
// addressing and becomes the price of two different decoders.
//
// What this file adds is the ADDRESSING of the real E1v stream, which
// `marche-bloc` replaces with a fixed 12-byte stride:
//
//   * one load of the group's base word, `bases[gid / 32]`;
//   * the lane's own header at a FIXED stride, `10 bits × lane`. It is the only
//     field readable before anything else is known: a record's width depends on
//     its class, its class lives in its header, and a header at a variable
//     offset would have to be found before it could be read;
//   * a WARP-SCAN over the 32 payload widths of the group — every earlier lane
//     contributes its own, and each of those widths is a table lookup on that
//     lane's header;
//   * a read at an ARBITRARY bit offset, up to 46 bits wide, hence a window
//     straddling three words. (46, not the 56 the pre-registration divulges:
//     that 56 is a RECORD, header included, and the header is read elsewhere.
//     The scan sums payloads, so ten bits separate the two quantities and
//     nothing but their names keeps them apart.)
//
// 🚨 So this arm is `marche-bloc` PLUS an address bill, and 0.6704 ns/block is a
// LOWER BOUND on it rather than an estimate of it. The pre-registration says so
// before the measurement, and this run can only widen the gap.
//
// -----------------------------------------------------------------------------
// 2. THE MAP, RESTATED FROM `llvq_artifact::e1v` — THE WRITER
// -----------------------------------------------------------------------------
//   group of 32 blocks, starting at word `bases[g]`:
//     [32 headers, 10 bits each: class 9 | gain 1]   fixed stride, 320 bits
//     [32 payloads, variable width, in lane order]
//     [pad to a whole word]
//
// The header's class field is the CNS's own id — 0..382 for a class, 511 for
// the origin. The record tables of this crate use another convention (entry 0
// the origin, entry 1+ci class ci), so the kernel converts. That conversion is
// not an artefact of the bench: any served kernel reading this stream pays it,
// and it is part of what this arm prices.
//
// -----------------------------------------------------------------------------
// 3. 🚨 THE RESERVE, AND IT BELONGS AT THE TOP
// -----------------------------------------------------------------------------
// Here `gid` IS the block index, so a SIMD group of 32 consecutive lanes IS
// exactly one E1v group: the alignment is true BY CONSTRUCTION. The served
// matvec puts one warp per ROW, and `nblocks mod 32` is 10 or 21 on the five
// shapes of the 4B — no warp there reads a single group. This file therefore
// measures the BEST CASE of E1v addressing; the served case reads two base
// words and scans two header regions.
//
// -----------------------------------------------------------------------------
// 4. NO TIMING, NO THRESHOLD, NO CLAIM
// -----------------------------------------------------------------------------
// Nothing here predicts a nanosecond. Operation counts read off this project's
// kernel sources have been wrong by a factor of ~2 three times — Golay70, E1c,
// and the walk-versus-block count of P1b.
// =============================================================================

#include <metal_stdlib>
using namespace metal;

constant uint DIM          = 24;
constant uint MAX_KINDS    = 8;
constant uint BINOM_STRIDE = DIM + 1;

// E1v's own constants, restated from `llvq_artifact::e1v` because MSL cannot
// import them. The host asserts the agreement (`p1host::e1v_payload_bits`).
constant uint E1V_GROUP       = 32u;
constant uint E1V_HEADER_BITS = 10u;
constant uint E1V_ORIGIN_ID   = 511u;

// A field of at most 32 bits at an arbitrary bit offset: TWO words, because a
// window that starts anywhere in a word ends inside the next one. Used for the
// header alone — 10 bits, at a fixed stride.
static inline uint read_small(device const uint* data, ulong bit, uint width) {
    uint  w   = uint(bit >> 5);
    uint  o   = uint(bit & 31ul);
    ulong two = (ulong(data[w + 1u]) << 32) | ulong(data[w]);
    return uint((two >> o) & ((1ul << width) - 1ul));
}

// The payload window: 96 bits from `bit`, shifted so that `bit` becomes bit 0
// of the pair `(lo, hi)`. THREE words, and the third is not a margin — the
// widest payload is 46 bits (class 323, 8 fields: a 56-bit record less its
// 10-bit header), and 46 bits starting at offset 31 of a word end in the third
// one. The host asserts the bound the other way round: 96 − 31 = 65 bits are
// readable at every offset, and no class asks for more.
//
// The buffer carries two words of padding past the last group so this read
// cannot run off the end. Per the repo's convention that padding is not counted
// in the arm's bytes (`thesis.rs:731`).
static inline void window96(device const uint* data,
                            ulong bit,
                            thread ulong &lo,
                            thread uint  &hi)
{
    uint w = uint(bit >> 5);
    uint o = uint(bit & 31ul);
    lo = (ulong(data[w + 1u]) << 32) | ulong(data[w]);
    hi = data[w + 2u];
    // A shift by 64 is undefined, so the already-aligned case is named rather
    // than reached by arithmetic.
    if (o != 0u) {
        lo = (lo >> o) | (ulong(hi) << (64u - o));
        hi = hi >> o;
    }
}

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
//   buffer(0) data    device const uint*    the E1v stream, bit-packed
//   buffer(1) bases   device const uint*    word offset of each group's start
//   buffer(2) tab     constant BlockRec*    384 records, origin at entry 0
//   buffer(3) pay     constant uint*        payload bits, indexed by the
//                                           header's own 9-bit class field
//   buffer(4) binom   constant uint*        C(p,c) transposed, 25x25
//   buffer(5) golay   device const uint*    the 4096 codewords, flat
//   buffer(6) gscale  constant float*       gain centroids
//   buffer(7) x       device const float*   24 activations
//   buffer(8) out     device float*         one float per block
// ---------------------------------------------------------------------------
kernel void decode_e1v(device const uint*   data   [[buffer(0)]],
                       device const uint*   bases  [[buffer(1)]],
                       constant BlockRec*   tab    [[buffer(2)]],
                       constant uint*       pay    [[buffer(3)]],
                       constant uint*       binom  [[buffer(4)]],
                       device const uint*   golay  [[buffer(5)]],
                       constant float*      gscale [[buffer(6)]],
                       device const float*  x      [[buffer(7)]],
                       device float*        out    [[buffer(8)]],
                       uint gid [[thread_position_in_grid]],
                       uint tid [[thread_index_in_threadgroup]])
{
    threadgroup float xs[DIM];
    if (tid < DIM) xs[tid] = x[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- the addressing, which is the whole reason this file exists -------
    uint  lane = gid & (E1V_GROUP - 1u);
    ulong gbit = ulong(bases[gid / E1V_GROUP]) * 32ul;

    // The header, at its fixed stride.
    uint hdr  = read_small(data, gbit + ulong(E1V_HEADER_BITS * lane), E1V_HEADER_BITS);
    uint id   = hdr & 511u;
    uint gain = (hdr >> 9) & 1u;

    // The warp-scan. Every lane reads its own width out of `pay`, indexed by
    // the header's id so the load does not wait on the table conversion below,
    // and the exclusive prefix sum over the 32 lanes says where this lane's
    // payload begins.
    //
    // 🚨 It is a SIMD-GROUP scan, so it is right only if lane `l` of the SIMD
    // group is block `32·g + l`. That holds for a 1-D dispatch whose
    // threadgroup is a multiple of 32, and it is NOT assumed: V0 reads all
    // 2^24 blocks back, and a mis-scan moves every payload after the first lane
    // whose width differs — which no tolerance absorbs.
    uint off = simd_prefix_exclusive_sum(pay[id]);

    ulong lo; uint hi;
    window96(data, gbit + ulong(E1V_HEADER_BITS * E1V_GROUP) + ulong(off), lo, hi);
    Cursor cur = { lo, hi };

    // The stream's id convention is not the table's: 0..382 for a class and 511
    // for the origin, against entry 0 for the origin and 1+ci for class ci.
    constant BlockRec &r = tab[(id == E1V_ORIGIN_ID) ? 0u : id + 1u];

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
