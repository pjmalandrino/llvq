// =============================================================================
// cascade_uniform.metal — the `cascade-uniformisée` arm of P1
//
// Pre-registration: proofs/preregistration-p1-2026-08-13.md (§2, arm 4).
// CPU reference:    llvq-search/src/rankdec.rs::cascade_uniform  (commit 8a1d192)
// Conventions:      llvq-metal/src/lib.rs::PAYLOAD_MSL (struct mirrors, cursor
//                   discipline, one float per block, activation staged once).
//
// This file decodes **an archive index** — the 48-bit v1 index of a real block
// of the sealed 4B — into a dot product against 24 activations, one block per
// lane, and writes ONE float per block. It never writes the 24 decoded weights:
// that was defect no. 1 of the 2026-07-31 bench, which measured its own
// uncoalesced stores and called the result a decoder.
//
// It contains no timing, no threshold and no claim. Nothing here predicts a
// nanosecond, and the operation counts a reader can make from this source have
// already been wrong by a factor of 2 on this kernel (Golay70, 2026-08-11).
//
// -----------------------------------------------------------------------------
// 1. WHAT IS UNIFORM, AND WHY EACH ONE IS
// -----------------------------------------------------------------------------
//
// (a) **Twenty-four outer iterations, for every block, in both cosets.** The
//     odd coset is one arrangement problem over 24 slots. The even coset is
//     two — the support (w slots) and the off-support (24 − w) — and
//     w + (24 − w) = 24 exactly. So the second problem is not a second loop:
//     it is a *state swap* inside the first, taken at s == w, selected
//     arithmetically like everything else. A lane decoding an odd block and a
//     lane decoding an even block of any weight run the same 24 steps in
//     lockstep. This is the property the arm exists to buy, and it is the one
//     `unrank_multiset` (fastdec.rs:167) cannot have: its outer trip count is
//     `out.len()`, which is 24, w, or 24 − w depending on the block.
//
// (b) **Eight candidates evaluated at every step, always, in order.** No
//     `break` when the slot is decided, no `continue` when a kind is spent. A
//     kind with count 0 yields mj = 0, an empty interval [acc, acc), and
//     `inside` is false for free — the skip is arithmetic, not control flow.
//     MAX_KINDS is 8 because the *even off-support* problem needs 8 (the zero
//     run is its own kind, fastdec.rs:40-47): a shader dimensioned on the old
//     4-kind `decode_rank` (llvq-metal/src/bin/decode.rs:123) would decode half
//     the file wrong, and the even coset is ~50% of it.
//
// (c) **Selection by mask arithmetic** (`sel_ul`/`sel_u`), the exact shape of
//     rankdec.rs::sel. Two lanes that choose different kinds do not diverge;
//     they execute the same instructions on different bits.
//
// (d) **No dynamic indexing of registers.** The CPU reference has exactly one
//     (`counts[chosen] -= 1`, rankdec.rs:141). Here the eight counts live as
//     eight bytes of one `ulong`, so that decrement is
//     `counts -= 1ul << (8*chosen)` — a variable shift, not an address — and
//     the same trick reads the chosen kind's |value|. On a GPU a dynamically
//     indexed register array is spilled to thread-local memory; this arm would
//     be measuring a spill.
//
// (e) **A fixed nine-step class search.** The archive stores only the index, so
//     the class must be found (`FastDecoder::decode` does it with
//     `partition_point`, fastdec.rs:329). The search here is branchless over an
//     `ends` table the host pads to 512 with ULONG_MAX, so the trip count is 9
//     for every lane and no probe is ever out of range.
//     ⚠️ Its cost is INSIDE this arm. A journal that compares this arm to a
//     layout whose payload carries an explicit 9-bit class field (Slot32,
//     Planes14, Flat32 — all of PAYLOAD_MSL) is comparing two different jobs,
//     and must say so.
//
// (f) **One divisor table, indexed by stage, not by class.** The cascade's
//     divisor is the number of slots left — 24, 23, … 1 — whatever the class.
//     §2 of the pre-registration describes "réciproques magiques par (classe,
//     étage) en table"; rankdec.rs:52-55 observes that such a table would hold
//     383 identical copies of the same 24 pairs, and keeps 25 entries. This
//     file follows rankdec.rs. Same arithmetic, same results, smaller table —
//     but it is a narrowing of the arm as written, and the journal should say
//     which of the two it ran.
//
// -----------------------------------------------------------------------------
// 2. WHAT IS NOT UNIFORM, AND CANNOT BE
// -----------------------------------------------------------------------------
//
//   * `xs[slot]` — threadgroup memory addressed by the slot being filled. Every
//     decoder in this repository pays it, PAYLOAD_MSL included.
//   * `classes[ci]`, `golay[base + gi]`, `dv.magic64[n]` — three loads whose
//     *address* differs across lanes. Their trip counts do not.
//   * The per-class mixed-radix peel needs a division by a per-class 64-bit
//     divisor (a_on, a_off, m_arr). That one IS per class, and it is the only
//     magic reciprocal here that has to be.
//
// Nothing else in this file varies with the data. In particular the number of
// multiplies, of divisions, of FMAs and of memory transactions per block is the
// same for every block of the file.
//
// -----------------------------------------------------------------------------
// 3. THE DIVISION — WHAT I CHOSE, AND WHY
// -----------------------------------------------------------------------------
//
// The CPU reference divides by multiplying: `magic_div` (rankdec.rs:78) does
// `(x as u128 * mul) >> shift` with `mul = 2^shift/n + 1` and shift = 64+⌈log₂n⌉.
// **MSL has no u128**, and worse, that multiplier does not fit 64 bits either
// (n = 3 gives mul ≈ 2.46e19 > 2^64). The constants cannot be carried over; the
// technique can. Three derivations were considered.
//
// **CHOSEN (default, `LLVQ_CASCADE_DIV == 0`): 64-bit round-up magic with a
// hand-written mulhi64.** Take M(n) = ⌊2^64/n⌋ + 1 for n ≥ 2, which fits u64
// (n = 2 gives 2^63 + 1). Then mulhi64(x, M) = ⌊x/n⌋ whenever x·n < 2^64.
// Proof: M = (2^64 + e)/n exactly with 1 ≤ e ≤ n, so x·M/2^64 = x/n + x·e/(n·2^64);
// writing x = qn + r, the floor is q iff r + x·e/2^64 < n, and x·e ≤ x·n < 2^64
// gives it. Our x is m·c with c ≤ 24 and m ≤ the class cardinality, which is
// ≤ N(13) = 280,974,212,784,720 < 2^48 — so x < 2^53 and the condition holds by
// a factor of 2^11 on the a-priori bound alone. Enumerated over the 383 classes
// (harness of §6), the largest m0 of any arrangement problem is in fact
// 5,354,228,880 < 2^33, giving m0·24 = 128,501,493,120 < 2^37: eight octaves of
// margin, not one. n = 1 has no representable multiplier and is handled by a
// select (the quotient is x). This path is **exact for every input it can
// receive**, needs no divisibility hypothesis, and is the one the
// pre-registration describes; that is why it is the default. It costs the
// mulhi64 below: four 32×32 multiplies and a carry column, per candidate.
//
// **NOT chosen but written (`== 1`): exact division by a modular inverse.**
// n divides m·c at every step — that is the invariant of the recurrence
// (fastdec.rs:154-156: m·c_j/n = (n−1)!/((c_j−1)!∏c_i!), an integer for every
// j, and 0 for c_j = 0). For an exactly divisible x, ⌊x/n⌋ = (x >> tz(n)) ·
// inv(n >> tz(n)) mod 2^64, which needs only the LOW half of one 64-bit
// product — roughly a third of the work of the mulhi. It is not the default for
// one reason: it is silent when its hypothesis fails. The magic path is wrong
// nowhere; this one is wrong everywhere the divisibility breaks, and returns a
// plausible number while doing it. It is here so the bench can dispatch it as
// an **appended** arm (§1.3: a new arm never reorders the existing ones) and so
// the full sweep can require the two paths to agree — the pattern of
// `both_factorizations_agree`.
//
// **NOT chosen but written (`== 2`): the q/r split.** With q = ⌊m/n⌋ and
// r = m − qn, m·c/n = q·c + (r·c)/n, and the second term is exact for the same
// reason (n | m·c and n | qnc ⇒ n | rc). It moves the 64-bit division out of
// the candidate loop — one per slot instead of eight — leaving a 32-bit
// multiply and a 32-bit magic (r·c < 24·24 = 576) per candidate. Same results,
// materially less arithmetic. It is a **different arm**, not a faster spelling
// of this one: it needs its own V0.
//
// 🚨 Changing `LLVQ_CASCADE_DIV`'s default is a change of what the arm is, and
// therefore a §7bis entry in the pre-registration — not an edit.
//
// -----------------------------------------------------------------------------
// 4. SCALES — THE DOUBLE-NORMALISATION TRAP
// -----------------------------------------------------------------------------
//
// `gpu_class_table` (llvq-metal/src/lib.rs:380) hands PAYLOAD_MSL `vals`
// **already divided by √(16m)**. This file does NOT use that table: it reads
// the class's *integer* |values| (bytes) and one `inv_norm = 1/√(16m)` per
// class, and applies inv_norm exactly once, at the end, together with the gain
// centroid. A reference that fed this kernel the pre-divided values and then
// divided again would be off by 1/√(16m) — and since the tolerance is relative
// to the expected dot, the error would read as a decode divergence rather than
// as a scale bug.
//
// Applying the scale once at the end (rather than per coordinate) also makes
// the accumulation exact in the integers up to the final multiply. The CPU
// reference of §3 of the pre-registration — `FastDecoder::decode` → point →
// /√(16m) → ×gscale — is the etalon; the two differ by rounding only, which is
// why V0 compares with a tolerance relative to Σ|wᵢxᵢ| and NOT with the
// absolute 2e-3 floor of `decreal::verify` (decreal.rs:129). A floor that
// coarse cannot see a flipped sign bit on a small dot: "a fire alarm is not a
// guard" (thesis.rs:84).
//
// -----------------------------------------------------------------------------
// 5. WHAT THE HOST MUST ASSERT — none of it is checked here
// -----------------------------------------------------------------------------
//
//   1. `fd.n_classes() == N_CLASSES` (383) and `ends` padded to CLASS_PAD (512)
//      entries with ULONG_MAX. The search reads ends[0..511] unconditionally.
//   2. every `idx[b] <= N13`. An index past the table produces an out-of-range
//      `gi`, i.e. a read past the Golay table. `ci` is clamped here so the class
//      record itself is never read out of range, but that clamp is a fault
//      guard, not a correctness path.
//   3. every `gains[b] < gscale_len`. Unlike every other shader in the repo,
//      this one does NOT hardcode a gain width — the rank arrives in its own
//      array — but it trusts it.
//   4. `CascadeRec` is 88 bytes and `DivTab` 600, field for field, in the order
//      declared below; and unused kinds have BOTH their count byte and their
//      value byte zero.
//   5. `out` is read back and compared before any timing is believed. A kernel
//      whose output nobody reads is a kernel the compiler may delete, and a
//      dead kernel benchmarks beautifully.
//
// -----------------------------------------------------------------------------
// 6. PROVENANCE — what has been run, and what has not
// -----------------------------------------------------------------------------
//
// ✅ **The algorithm has been executed and is right.** Every line below was
// transcribed register for register into Rust and run against
// `FastDecoder::decode`: **49,410 blocks decoded identically** — 43 probes per
// class (both class boundaries, the midpoint, and 40 uniform draws) over all
// **383 classes**, in all three `LLVQ_CASCADE_DIV` modes, plus the origin. The
// comparison is an **equality on the 24 signed integer coordinates**, not a
// tolerance on a dot product, so a flipped sign or a permuted arrangement
// cannot hide in it. That harness lives beside this file (`sim/src/main.rs`),
// and it is deliberately a slavish copy: it is a transcription check, NOT the
// independently-written CPU reference of the pre-registration's §3 point 1, and
// it does not discharge it — nor point 2's synthetic fixture, nor point 3's
// 150,681,600-block sweep.
//
// ❌ **This file has never been through a Metal compiler.** `xcrun metal` is not
// installed on the machine it was written on (Xcode command-line tools only), so
// every MSL-specific claim here — that `as_type`, `ctz`, `min`, the address-space
// qualifiers and the `ulong` arithmetic compile as written — rests on
// PAYLOAD_MSL using the same constructs, not on an observed compile. First run
// is a syntax check; treat a compile error as expected work, not as a surprise.
//
// ❌ No count of registers, no occupancy, no spill report, and above all **no
// nanoseconds**. The whole point of P1 is that this file does not get to have an
// opinion about its own speed.
//
// =============================================================================

#include <metal_stdlib>
using namespace metal;

constant uint DIM       = 24;
// Widest single unranking problem: the even off-support side, zero run
// included (llvq-search/src/fastdec.rs:47).
constant uint MAX_KINDS = 8;
// Classes of the cap-13 ball (`FastDecoder::n_classes()`), and the padded
// length of the `ends` table — 512, the power of two above it, so the
// branchless search is exactly 9 steps and every probe is in range.
constant uint N_CLASSES = 383;
constant uint CLASS_TOP = 256;   // CLASS_PAD / 2, the search's first stride

// 0 = round-up magic (the pre-registered arm), 1 = modular inverse,
// 2 = q/r split. See §3 of the header before touching this.
#ifndef LLVQ_CASCADE_DIV
#define LLVQ_CASCADE_DIV 0
#endif

// ---------------------------------------------------------------------------
// Division tables, indexed by the STAGE (slots left, 1..24) — never by class.
// Mirror of the Rust-side `#[repr(C)]` record, 600 bytes:
//   magic64[n] = floor(2^64 / n) + 1          (n >= 2; entries 0,1 zero)
//   modinv[n]  = (n >> ctz(n))^-1 mod 2^64    (odd part's inverse)
//   magic32[n] = floor(2^32 / n) + 1          (n >= 2; entries 0,1 zero)
//   tz[n]      = ctz(n)
// The host builds all four in Rust, where the identities are unit-testable —
// no constant in this file was written by hand.
// ---------------------------------------------------------------------------
struct DivTab {
    ulong magic64[25];
    ulong modinv[25];
    uint  magic32[25];
    uint  tz[25];
};

// ---------------------------------------------------------------------------
// One class of format v1, flattened for the cascade. Mirror of the Rust-side
// `#[repr(C)]` record, 88 bytes, built from `FastDecoder::cascade_class`
// (llvq-search/src/fastdec.rs:311) — the same data `FastDecoder::decode` runs
// on, so there is no second derivation to drift from.
//
// The odd and even cosets are folded into ONE shape, which is what lets the
// kernel have one code path:
//
//   even : problem A = on-support  (w slots,      m0 = a_on = d_on)
//          problem B = off-support (24 − w slots, m0 = a_off = d_off)
//          peel order: s_f (shift), s_w (shift), a_off, a_on, then the codeword
//   odd  : problem A = the whole block (24 slots, m0 = m_arr = d_on)
//          problem B unused; set w = 24, sf_shift = sw_shift = 0, d_off = 1
//          so the peel degenerates to `r_on = local % m_arr, gi = local / m_arr`
//          without a single branch.
//
// `d_on` and `d_off` are each both a radix of the peel AND the m0 of a
// problem — they are the same number, which is why neither is stored twice.
// ---------------------------------------------------------------------------
struct CascadeRec {
    ulong offset;      // local rank = (idx - 1) - offset
    ulong d_on;        // a_on   (even) / m_arr (odd)   — A's m0, and the last radix
    ulong d_off;       // a_off  (even) / 1     (odd)   — B's m0, and the first radix
    ulong magic_on;    // Granlund-Montgomery multiplier for d_on
    ulong magic_off;   // ditto for d_off
    ulong counts_a;    // 8 kinds x 8 bits: word_counts (even) / counts (odd)
    ulong counts_b;    // 8 kinds x 8 bits: off_counts, zero run included (even)
    ulong vals_a;      // 8 kinds x 8 bits: |value| per kind, 0 for unused kinds
    ulong vals_b;      // ditto; the zero run's byte is 0, which is what marks it
    float inv_norm;    // 1 / sqrt(16 m)
    uint  golay_base;  // offset into the 4096-word table: 0 (odd), offsets[w] (even)
    uint  geom;        // w | p_req<<8 | sf_shift<<16 | sw_shift<<24
    uint  dflags;      // shift_off | shift_on<<8 | one_off<<16 | one_on<<17 | odd<<18
};

// ---------------------------------------------------------------------------
// Branch-free selects — `rankdec.rs::sel` (:86), transcribed. `p ? a : b`
// written as arithmetic, so two lanes disagreeing on `p` cost nothing.
// ---------------------------------------------------------------------------
static inline ulong sel_ul(bool p, ulong a, ulong b) {
    ulong m = ~(ulong(p) - 1ul);            // 0 or all-ones
    return (a & m) | (b & ~m);
}

static inline uint sel_u(bool p, uint a, uint b) {
    uint m = ~(uint(p) - 1u);
    return (a & m) | (b & ~m);
}

// ---------------------------------------------------------------------------
// High 64 bits of a 64x64 product, by hand.
//
// MSL has no u128 and no 128-bit intermediate. `mulhi` may or may not be
// declared for `ulong` on a given Metal version; this file does not find out at
// the user's expense — the four 32x32 partial products and their carry column
// are written out, which compiles everywhere and lowers to exactly the
// multiplies an Apple GPU's 32-bit ALU would issue anyway.
//
//   a*b = p11*2^64 + (p01 + p10)*2^32 + p00
// and the high half is p11 plus the two high halves plus the carry out of the
// 2^32 column.
// ---------------------------------------------------------------------------
static inline ulong mulhi64(ulong a, ulong b) {
    ulong a0 = a & 0xfffffffful, a1 = a >> 32;
    ulong b0 = b & 0xfffffffful, b1 = b >> 32;

    ulong p00 = a0 * b0;
    ulong p01 = a0 * b1;
    ulong p10 = a1 * b0;
    ulong p11 = a1 * b1;

    ulong mid = (p00 >> 32) + (p01 & 0xfffffffful) + (p10 & 0xfffffffful);
    return p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
}

// ---------------------------------------------------------------------------
// x / d for a per-CLASS 64-bit divisor: Granlund-Montgomery with the add-back
// step, exact over the whole 64-bit range.
//
//   magic = floor(2^(64+b) / d) + 1 - 2^64,  b = ceil(log2 d),  shift = b - 1
//   q = mulhi64(x, magic); t = ((x - q) >> 1) + q; quot = t >> shift
//
// The `((x - q) >> 1) + q` is `(x + q) / 2` written so it cannot overflow.
// d = 1 has no b, so the host sets `one` and the result is x — the select keeps
// the shift defined rather than shifting by -1.
//
// This is the only reciprocal here that is per class, and it has to be: the
// peel's radices are the class's arrangement counts.
// ---------------------------------------------------------------------------
static inline ulong div_big(ulong x, ulong magic, uint shift, bool one) {
    ulong q = mulhi64(x, magic);
    ulong t = ((x - q) >> 1) + q;
    return sel_ul(one, x, t >> shift);
}

// ---------------------------------------------------------------------------
// One block: archive index in, dot product out.
//
// Everything below runs the same number of times for every block. The only
// per-block quantities are the values in registers and the addresses of three
// loads.
// ---------------------------------------------------------------------------
static inline float cascade_block(ulong idx,
                                  uint  gain,
                                  device const ulong*      ends,
                                  device const CascadeRec* classes,
                                  device const uint*       golay,
                                  constant DivTab&         dv,
                                  constant float*          gscale,
                                  threadgroup const float* xs)
{
    // ---- 0. the class, in nine fixed steps -------------------------------
    //
    // `FastDecoder::decode` takes i = idx - 1 and the first class whose end is
    // strictly above it. The origin (idx == 0) decodes to the zero point, so it
    // is not special-cased with a branch: i is clamped to 0, some class is read,
    // and the result is multiplied by zero at the end. Real draws from the 4B
    // hold no origin block (docs/mesures/shell-distribution-4b-2026-08-10.txt),
    // but the §1.6 synthetic fixture does, and it must go through this kernel.
    bool  live = (idx != 0ul);
    ulong i    = sel_ul(live, idx - 1ul, 0ul);

    uint ci = 0u;
    for (uint b = CLASS_TOP; b != 0u; b >>= 1) {
        ci = sel_u(ends[ci + b - 1u] <= i, ci + b, ci);
    }
    ci = min(ci, N_CLASSES - 1u);      // fault guard, never a correctness path
    device const CascadeRec& r = classes[ci];

    // ---- 1. the mixed-radix peel -----------------------------------------
    ulong local    = i - r.offset;
    uint  geom     = r.geom;
    uint  w        = geom & 0xffu;             // 24 for the odd coset
    uint  p_req    = (geom >> 8) & 1u;
    uint  sf_shift = (geom >> 16) & 0xffu;     // log2 s_f, 0 for the odd coset
    uint  sw_shift = (geom >> 24) & 0xffu;     // log2 s_w, 0 for the odd coset

    // The two sign fields are powers of two, so they peel with a mask and a
    // shift. Both are at most 24 bits, hence both fit a uint.
    uint r_sf = uint(local & ((1ul << sf_shift) - 1ul));
    local >>= sf_shift;
    uint r_sw = uint(local & ((1ul << sw_shift) - 1ul));
    local >>= sw_shift;

    uint  dflags = r.dflags;
    ulong q      = div_big(local, r.magic_off, dflags & 0xffu, ((dflags >> 16) & 1u) != 0u);
    ulong r_off  = local - q * r.d_off;
    local        = q;
    q            = div_big(local, r.magic_on, (dflags >> 8) & 0xffu, ((dflags >> 17) & 1u) != 0u);
    ulong r_on   = local - q * r.d_on;

    bool odd = ((dflags >> 18) & 1u) != 0u;
    uint cw  = golay[r.golay_base + uint(q)] & 0xffffffu;

    // ---- 2. twenty-four identical steps ----------------------------------
    //
    // State of the problem in flight. For the even coset the second problem is
    // swapped in at s == w; for the odd coset w is 24 and the swap never fires.
    ulong rank   = r_on;
    ulong m      = r.d_on;
    ulong counts = r.counts_a;
    ulong vals   = r.vals_a;
    // Slots still to visit, in increasing coordinate order: the codeword's
    // support for problem A, its complement for problem B, all 24 for the odd
    // coset (whose single problem is indexed by the coordinate itself).
    uint  mask   = sel_u(odd, 0xffffffu, cw);
    uint  maskb  = (~cw) & 0xffffffu;

    float dot = 0.0f;
    uint  par = 0u;      // running parity of the support's free sign bits

    for (uint s = 0; s < DIM; ++s) {
        bool swap = (s == w);
        rank   = sel_ul(swap, r_off,      rank);
        m      = sel_ul(swap, r.d_off,    m);
        counts = sel_ul(swap, r.counts_b, counts);
        vals   = sel_ul(swap, r.vals_b,   vals);
        mask   = sel_u (swap, maskb,      mask);
        bool in_b = (s >= w);

        // Slots left in the problem in flight, in closed form rather than as a
        // decrementing register: problem A has w - s left, problem B has
        // (24 - w) - (s - w) = 24 - s, and the odd coset (w = 24) is the second
        // case with s never reaching w. Always in 1..24, never 0.
        uint n = sel_u(in_b, DIM - s, w - s);

        // The stage's divisor is `n`, the slots left — one table, no class.
#if LLVQ_CASCADE_DIV == 0
        ulong M   = dv.magic64[n];
        bool  n1  = (n == 1u);
#elif LLVQ_CASCADE_DIV == 1
        uint  tz  = dv.tz[n];
        ulong inv = dv.modinv[n];
#else
        ulong M   = dv.magic64[n];
        bool  n1  = (n == 1u);
        ulong qs  = sel_ul(n1, m, mulhi64(m, M));   // m / n
        ulong rs  = m - qs * ulong(n);              // m % n, < 24
        uint  M32 = dv.magic32[n];
#endif

        // Eight candidates, always eight, in order. `acc` walks the low edge of
        // the intervals; the eight `mj` are independent of each other, which is
        // where whatever instruction-level parallelism this loop has lives.
        ulong acc = 0ul, base = 0ul, new_m = m;
        uint  chosen = 0u;
        for (uint j = 0; j < MAX_KINDS; ++j) {
            ulong c = (counts >> (8u * j)) & 0xfful;
#if LLVQ_CASCADE_DIV == 0
            ulong x  = m * c;
            ulong mj = sel_ul(n1, x, mulhi64(x, M));
#elif LLVQ_CASCADE_DIV == 1
            // Exact only because n divides m*c — see §3 of the header.
            ulong mj = ((m * c) >> tz) * inv;
#else
            uint  rc = uint(rs) * uint(c);          // < 24*24 = 576
            ulong mj = qs * c + ((ulong(rc) * ulong(M32)) >> 32);
#endif
            ulong lo = acc, hi = acc + mj;
            bool inside = (rank >= lo) && (rank < hi);
            chosen = sel_u (inside, j,  chosen);
            new_m  = sel_ul(inside, mj, new_m);
            base   = sel_ul(inside, lo, base);
            acc    = hi;
        }
        rank  -= base;
        m      = new_m;
        counts -= (1ul << (8u * chosen));           // the one dynamic index, as a shift

        // ---- the slot this step just decided ----
        //
        // `mask` holds exactly the slots not yet visited by the problem in
        // flight; w + (24 - w) = 24 means it is never empty at a step, so `ctz`
        // never sees zero.
        uint slot = ctz(mask);
        mask &= mask - 1u;

        uint  v   = uint((vals >> (8u * chosen)) & 0xfful);
        float fv  = float(v);
        bool  nzv = (v != 0u);       // only the even off-support zero run is 0

        // Three sign rules, one per situation, all evaluated, one selected.
        //
        //   odd    : forced by the codeword and the value mod 4 — a value
        //            ≡ 3 (mod 4) is positive on the support, negative off it,
        //            and ≡ 1 (mod 4) the opposite (fastdec.rs:343-346). For odd
        //            values, `v % 4 == 3` is bit 1 of v, so the whole rule is
        //            one XOR.
        //   even A : bit s of r_sw, except on the last support slot, which
        //            carries the parity repair `par ^ p_req` (fastdec.rs:371-378).
        //   even B : the next bit of r_sf, consumed only by nonzero values.
        uint cbit    = (cw >> slot) & 1u;
        uint neg_odd = cbit ^ ((v >> 1) & 1u);

        bool last_on = (s + 1u == w);
        uint bfree   = (r_sw >> s) & 1u;
        uint neg_a   = sel_u(last_on, par ^ p_req, bfree);
        par ^= sel_u(last_on, 0u, bfree);   // only ever read on the A side

        uint neg_b = (r_sf & 1u) & uint(nzv);
        r_sf >>= sel_u(in_b && nzv, 1u, 0u);

        uint neg = sel_u(odd, neg_odd, sel_u(in_b, neg_b, neg_a));

        // Sign by XOR of the float's sign bit: one integer op, and -0.0 for the
        // zero run is harmless in an accumulate.
        float wv = as_type<float>(as_type<uint>(fv) ^ (neg << 31u));
        dot = fma(wv, xs[slot], dot);
    }

    // One scale for the whole block: the shell norm folded with the gain
    // centroid. The row scale is the caller's, as in PAYLOAD_MSL. The origin
    // has no shell, so its scale is zero rather than `inv_norm`.
    float scale = r.inv_norm * gscale[gain];
    return dot * (live ? scale : 0.0f);
}

// ---------------------------------------------------------------------------
// The arm. One block per lane, structure-of-arrays inputs, one float out.
//
//   buffer(0) idx      device const ulong*      archive index per block
//   buffer(1) gains    device const uchar*      gain rank per block
//   buffer(2) ends     device const ulong*      cumulative class ends, 512 padded
//   buffer(3) classes  device const CascadeRec* 383 records
//   buffer(4) golay    device const uint*       the 4096 codewords, `Golay::codewords`
//   buffer(5) dv       constant DivTab&         stage reciprocals
//   buffer(6) gscale   constant float*          gain centroids
//   buffer(7) x        device const float*      24 activations
//   buffer(8) out      device float*            one float per block
//
// `classes` sits in `device` and not `constant` on purpose: every lane reads a
// different record, so the access is divergent rather than broadcast, and the
// table is 33.7 KB against PAYLOAD_MSL's 12 KB. Which address space wins is a
// one-word question for the bench, not a claim to make here.
// ---------------------------------------------------------------------------
kernel void cascade_uniform(device const ulong*      idx     [[buffer(0)]],
                            device const uchar*      gains   [[buffer(1)]],
                            device const ulong*      ends    [[buffer(2)]],
                            device const CascadeRec* classes [[buffer(3)]],
                            device const uint*       golay   [[buffer(4)]],
                            constant DivTab&         dv      [[buffer(5)]],
                            constant float*          gscale  [[buffer(6)]],
                            device const float*      x       [[buffer(7)]],
                            device float*            out     [[buffer(8)]],
                            uint gid [[thread_position_in_grid]],
                            uint tid [[thread_index_in_threadgroup]])
{
    // The activation is staged once per threadgroup and never re-read from
    // device memory inside the loop — defect no. 2 of the 2026-07-31 bench, which
    // made the floor kernel bandwidth-bound and hid what it was measuring.
    threadgroup float xs[DIM];
    if (tid < DIM) xs[tid] = x[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    out[gid] = cascade_block(idx[gid], uint(gains[gid]),
                             ends, classes, golay, dv, gscale, xs);
}
