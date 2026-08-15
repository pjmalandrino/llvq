// ===========================================================================
// binomial_walk.metal — the `marche-binomiale` arm of P1, in MSL.
//
// This file contains ONE kernel and NO number. It decodes; the nanoseconds are
// the bench's business. No count of source-level operations appears below as a
// prediction: on this project such a count has already been wrong by a factor
// of two (Golay70, v2 promised 1.9–2.4× and the card returned 1.77×), and
// `proofs/preregistration-p1-2026-08-13.md` §5 forbids fabricating a range for
// a decoder that has never run.
//
// ---------------------------------------------------------------------------
// 1. What it is a port of, and what "port" means here
// ---------------------------------------------------------------------------
// The single source of truth is `llvq-search/src/rankdec.rs::binomial_walk`
// (commit 8a1d192). Same class records, same colex unranking, same
// arrangement out. Where this file departs from the Rust in *shape*, it is
// noted below and the departure is provably outcome-neutral; where it departs
// in *outcome*, it is a bug in this file and V0 must fail.
//
// Its V0 standard is NOT equality with `FastDecoder::decode` — amendment É2 of
// the pre-registration settles that. The walk decodes its own combinatorial
// order; relating it to the archive's multiset-permutation order IS the CNS
// re-bijection, which is P5, which this bench gates. Its standard is a round
// trip on its own bijection plus the proof that it is one. What this shader
// therefore has to be checked against, block by block, before a single
// millisecond is believed, is `binomial_walk` itself run on the same records —
// a third independent implementation agreeing with the Rust, exactly as
// `decreal` checks its two kernels against the CPU decoder.
//
// ---------------------------------------------------------------------------
// 2. How it plugs into the bench
// ---------------------------------------------------------------------------
// The repo's pattern is `format!("{}{}", llvq_metal::PAYLOAD_MSL, SRC)`
// (`decreal.rs:184`). This file is written to compile **standalone** so it can
// be read and checked on its own; when it is appended to `PAYLOAD_MSL`,
// delete the fenced PRELUDE below — `DIM`, `Cursor` and `take` are already
// defined there, verbatim as reproduced here, and nothing else in this file
// collides. Reusing `PAYLOAD_MSL`'s `take` rather than writing a second bit
// cursor is deliberate: that function carries two traps (a legal width of 0,
// which would otherwise shift a ulong by 64, and the funnel refill from `hi`)
// and the repo should own exactly one copy of them.
//
// ---------------------------------------------------------------------------
// 3. The per-block record, and why it is 12 bytes at a fixed stride
// ---------------------------------------------------------------------------
// Three aligned u32 read at `words[3*gid ..]` — the same addressing as the
// `sol` and `Fixed96` anchors (`decreal.rs:51` and `:74`). That is the point:
// this arm is priced against those two, so it must not pay a different
// addressing bill. A variable-width packing would fold E1v's addressing into
// what is supposed to be a decoder number, and E1v's addressing is a separate
// question.
//
// Field order, LSB-first inside the 96-bit little-endian window, mirroring
// Slot32's header so a reader of this repo already knows it:
//
//     [class 9][gain 1][smask 24][rank_0 wbits[0]] … [rank_{k-2} wbits[k-2]]
//
// The final kind has no field at all — see §6.
//
// ⚠️ CONTRACT THE RUST SIDE MUST ASSERT, because the shader cannot:
//   * `9 + 1 + 24 + Σ_j wbits[j] ≤ 96` for every class. 34 bits of header and
//     signs leave 62, which is exactly the cursor's remaining capacity: after
//     34 bits are consumed, `hi` is empty and the surviving 62 live in `lo`.
//     CALCULATED, and it fits with ZERO SLACK: the maximum of `Σ wbits` over
//     *every* ordered composition of 24 into at most 8 positive parts is
//     exactly **62**, attained at counts [2,2,3,2,4,3,4,4] (exhaustive
//     enumeration, 2026-08-15). That range is a strict superset of the real
//     classes — a zero count contributes a radix of 1, hence width 0, and
//     leaves `free` untouched, so it can never raise the sum — therefore
//     `9+1+24+62 = 96` holds for every class the codebook can contain.
//     ⚠️ Zero slack is a hazard, not a comfort. A SECOND GAIN BIT WOULD
//     OVERFLOW THIS RECORD BY EXACTLY ONE BIT on that worst class. So would
//     any extra header field. The bound is also a bound over shapes, not a
//     reading of the codebook: the bench must still assert per class, because
//     the widths it writes come from `walk_radices` on the real
//     `FastDecoder`, and a disagreement there is a bug this bound cannot see.
//     If the assertion ever fires, the record grows to four words and the
//     journal must say so, because the arm would then read 16 bytes where
//     `sol` reads 12; the honest repair is to widen `sol` too, not to leave
//     the confound in.
//   * `wbits[j] = ceil(log2 walk_radices(counts,k)[j])`, and `wbits[k-1] = 0`.
//   * ranks are drawn in `0..radix_j` (pre-registration §3: uniformly, inside
//     the *real* class of each real block). A rank outside its radix is
//     malformed input; see the clamp in §7.
//   * `k ≥ 1` and `Σ_{j<k} counts[j] = 24`. Both are structural for a class
//     record and neither is checked at runtime — MSL has no bounds checks.
//   * `id ≤ 383`. Nine bits admit 512; the table has 384 entries. Inherited
//     exposure, identical to `PAYLOAD_MSL`'s `tab[id]`.
//
// Unlike `slot_dot`, `flat_dot` and `cursor_g32`, this kernel reads **no
// window past its own record** — word `3*gid+2` is the block's last word. It
// therefore needs none of the +16/+20 byte padding those three require
// (`decreal.rs:193`, `matvec.rs:487`, `thesis.rs:731`), and there is no
// padding to accidentally count as bytes read. On CUDA that same overrun is a
// fault that kills the context; this arm does not have the problem to port.
//
// ---------------------------------------------------------------------------
// 4. DELICATE POINT ONE — where C(n,k) lives, and why
// ---------------------------------------------------------------------------
// The walk reads `C(p, c)` for `p ≤ 23`, `c ≤ 24`. Three facts decide the
// placement, and all three are arithmetic rather than measured:
//
//   (a) Every entry fits in 32 bits. The largest is C(24,12) = 2 704 156, so
//       the whole table is `uint`, not `ulong`, and 25×25×4 = 2 500 bytes.
//       The Rust `BINOM` is `[[u64; 25]; 25]` (5 KB); the bench narrows it
//       with a checked cast so there is still exactly one table in the
//       project. Writing the 625 literals into this shader would create the
//       second copy that §3.1 of the pre-registration exists to forbid.
//   (b) Every *rank* also fits in 32 bits, since a rank is bounded by a
//       radix which is one of those entries. The entire walk is 32-bit: no
//       64-bit compare-and-subtract chain anywhere.
//   (c) The table is transposed relative to the Rust. Here `bt[c*25 + p]` is
//       `C(p, c)`, zero when `c > p`. The inner scan holds `c` fixed and walks
//       `p` downward, so the transpose turns a stride of 25 words into a
//       stride of one, and lets the row pointer be *decremented* by 25 when a
//       unit is placed instead of a multiply per step. This is a claim about
//       addresses, not about time.
//
// It is bound as a `constant` buffer, which is what the repo already does for
// its read-only tables (`constant ClassRec* tab`, `constant float* gscale`).
// Two alternatives exist and neither is predicted here:
//   * `device const uint*` — a one-word change to the signature. The classic
//     justification for `constant` is *uniform* access, and that justification
//     does not hold here: each lane sits on a different class, so each lane
//     reads a different (c,p). Whether that costs anything on an M3 is a
//     measurement this file does not make.
//   * staging into `threadgroup` — 2 500 bytes plus one barrier, competing for
//     the same 32 KB the activations already use 96 of. Also a measurement.
// The zeros for `c > p` are load-bearing, not a guard: when a kind must take
// every remaining position, `C(p,c) = 0` makes `rank >= b` true at every step,
// which is precisely how the walk places them. Do not "optimise" them away.
//
// ---------------------------------------------------------------------------
// 5. DELICATE POINT TWO — free position → slot, and the bug that cannot be
//    written in this form
// ---------------------------------------------------------------------------
// The CPU unranks in *free-position* space and then maps back: it snapshots
// `let before = free_mask;` and calls `nth_set_bit(before, want)`, clearing
// bits only afterwards. The comment there earns its keep — mapping against a
// mask one is simultaneously clearing shifts every later index by one, and the
// arrangement comes out permuted without any test on ranks noticing. That bug
// was written (in the instrumented twin, `rankdec.rs:546`) and caught.
//
// A literal port would be doubly wrong on a GPU. `nth_set_bit` is a loop of
// `x &= x-1` run `want` times, and `want` is a free-position index that
// depends on the *rank* — so the literal port would make the step count
// rank-dependent, destroying the one property this arm is built on. There is
// no PDEP on Metal to rescue it either.
//
// So this shader never leaves slot space:
//
//   * the scan iterates the free slots from the top (`31 - clz(m)`), and the
//     free-position index `p` is simply a counter initialised to
//     `popcount(freem)` and decremented once per step. That counter IS the
//     mapping — no search, no `nth_set_bit`, nothing rank-dependent;
//   * the scan consumes a *snapshot* `m`, and accumulates what it takes into
//     `taken`, in slot bits. `freem` is not touched until the scan is over:
//     `freem ^= taken;` on the line after the loop. Read-before-clear is not a
//     discipline to remember here, it is the shape of the code — there is no
//     expression in this kernel that could read a partially cleared mask.
//
// The correspondence with the Rust is exact: the highest free slot is the
// (n_free−1)-th set bit counted from the LSB, which is the free position the
// CPU labels `n_free−1`, and so on downward.
//
// ---------------------------------------------------------------------------
// 6. DELICATE POINT THREE — the last kind reads no rank
// ---------------------------------------------------------------------------
// Its radix is 1: it takes every slot still free, and storing a rank for it
// would be storing a constant. Three consequences, all visible below:
//   * the outer loop runs `j + 1 < k`, so the final kind never enters it;
//   * `wbits[k-1] = 0`, so even a bench that walked the field list uniformly
//     would consume nothing — `take(c, 0)` is legal and returns 0, which is
//     the trap `PAYLOAD_MSL` already guards;
//   * the final kind is deposited by one `ctz` walk over the residual `freem`,
//     of length `counts[k-1]` — a class constant.
// A zero-count kind is handled by the same arithmetic: its radix is also 1,
// hence width 0, hence the `take` before the `continue` consumes nothing. The
// `take` is placed before the skip on purpose, so the cursor advance has no
// data-dependent shape.
// The origin record (k = 1, counts [24,0…], vals all zero) falls out: the
// outer loop is empty and the deposit walk multiplies 24 zeros. It decodes to
// a dot of 0, which is what the origin means.
//
// ---------------------------------------------------------------------------
// 7. Fixed step count, and the one departure from the Rust
// ---------------------------------------------------------------------------
// The Rust scan carries `if c == 0 { break; }`. This shader drops it, and the
// drop is outcome-neutral by a two-line argument: colex unranking maintains
// `r < C(p+1, c)`, so when `c` reaches 0 we have `r < C(p+1,0) = 1`, i.e.
// `r = 0`; and `C(p,0) = 1` for every p, so `r >= b` is `0 >= 1`, false, at
// every remaining step. The break is an optimisation, never a decision.
//
// Dropping it is what makes the arm what it claims to be. The instrumented
// twin `steps_of` (`rankdec.rs:517`) counts `n_free` per kind — the *full*
// scan — while the Rust actually executes fewer. This kernel executes exactly
// what the twin counts, so the step count is a function of the class alone,
// with no rank in the expression. `the_step_count_depends_only_on_the_class`
// is a statement about this shader, not merely about its reference.
//
// The scan body is branch-free for the same reason: the state updates are
// selects, and the FMA is not in the scan at all but in a separate deposit
// walk, so the arithmetic that varies with the rank is a handful of ALU ops
// and never a divergent block. Total FMAs: 24 per block, one per slot,
// exactly as `slot_dot` does. The alternative — branch around the body and do
// fewer selects — trades uniform work for less work, is a three-line change,
// and which one wins is a measurement, not a paragraph.
//
// The divergence that remains IS the measurement. Lanes carry different
// classes, so different `k`, different counts, different scan lengths. That
// is the whole reason P1 runs on real blocks: the 2026-07-31 bench put four
// uniform magnitudes in every block and therefore tested no divergence at all
// (pre-registration §1.5).
//
// The `(cnt != 0)` term in the hit predicate is NOT part of that argument. It
// can never change an outcome for a rank inside its radix — the paragraph
// above proves the compare already fails — and it is there so that a
// malformed feed produces a wrong answer instead of walking the row pointer
// below the table and reading outside a `constant` buffer. A wrong answer is
// caught by V0; an out-of-bounds constant read is a mystery.
//
// ---------------------------------------------------------------------------
// 8. Traps inherited from this repo, named so they are not rediscovered
// ---------------------------------------------------------------------------
// * ONE GAIN BIT IS HARDCODED, like every other MSL in the repo
//   (`PAYLOAD_MSL` `take(c,1)`, `slot_dot`'s `hdr >> 9`). The Rust side must
//   carry `decreal.rs:154-159`'s assertion — centroid count is a power of two
//   with exponent 1 — or a two-gain-bit artifact will silently decode wrong
//   classes, every field after the gain being shifted by one.
// * `vals` ARE ALREADY DIVIDED BY THE SHELL NORM sqrt(16·m), exactly as
//   `gpu_class_table` does (`lib.rs:392-397`). This shader never divides by a
//   norm. A CPU reference that started from the *integer* point and also
//   divided would be right; one that read these `vals` and divided again
//   would be off by 1/sqrt(16m) — and since `decreal`'s tolerance is relative
//   to the expected value, that shows up as a decode divergence rather than as
//   the scale error it is.
// * The signs here are a 24-bit slot-major mask, like `slot_dot`'s `smask`.
//   That is a STAND-IN, declared as such: a real E1v decoder does not obtain
//   signs this way (the archive forces them from the codeword on the odd
//   side). It is included rather than omitted because the bias of a bench must
//   run against the arm it measures, never for it — an arm that skipped signs
//   would look cheaper than `masques`, which applies them, for a reason that
//   is not decoding.
// * Every lane writes `out[gid]`, and the bench must read the whole buffer
//   back and verify it before believing any timing. A kernel nobody reads is
//   a kernel the compiler may delete, and a dead kernel benchmarks
//   beautifully.
// * Do not time this with `Kernel::time` / `time_many`: they run warmup+reps
//   of a single arm before yielding, which is what produced minima from rounds
//   that never coexisted. The conforming template is the manual loop of
//   `thesis.rs:871-901`, with every arm dispatched every round in a fixed
//   order, the first two rounds discarded by the `if`, and the submission
//   overhead measured PER ROUND with its dispersion printed
//   (pre-registration §1.2, §1.3).
//
// ---------------------------------------------------------------------------
// 9. What this file does not settle
// ---------------------------------------------------------------------------
// * ABSENT — E1v's actual field layout. The 12-byte record above is the
//   bench's feed, chosen to match the anchors' addressing; it is not a claim
//   about how `e1v-séparé` packs its 53.332 bits/block.
// * ABSENT — how an even class composes. The archive's even side runs two
//   unrankings (support of w slots against the Golay codeword, non-support of
//   24−w) plus a parity repair. This kernel decodes ONE 24-slot walk, which is
//   what `binomial_walk` is and what the Rust tests exercise (they skip
//   `n_slots != DIM`). If the bench feeds one walk per block, the ns it
//   reports is the cost of one walk — and the pre-registration's thresholds
//   are per BLOCK. Whichever the bench does, the journal has to say it, and
//   the conservative reading is the one that costs the arm more.
// * No speed claim of any kind, for any hardware.
//
// ---------------------------------------------------------------------------
// 10. What has actually been run on this file — and what has not
// ---------------------------------------------------------------------------
// MEASURED, 2026-08-15, Apple M3 Max, via `MTLDevice.makeLibrary(source:)` —
// the same runtime path as `Kernel::new`:
//   * it compiles, and `decode_walk` comes out of the library;
//   * 20 000 blocks over 10 synthetic class shapes — including the origin
//     shape (k=1), the 62-bit worst case [2,2,3,2,4,3,4,4], a zero count in
//     the middle [8,0,8,8], and three radix-1-mid-walk shapes — agree with a
//     LITERAL port of `binomial_walk` (free-position space, `nth_set_bit`,
//     read-before-clear), worst relative error 5.0e-8, i.e. f32 rounding;
//   * six mutants killed: scanning from the bottom (`ctz` for `31-clz`), an
//     off-by-one on the free-position counter, reading the table untransposed,
//     decrementing the row pointer unconditionally, the final kind taking
//     `vals[0]`, and signs dropped. The check can fail, which is the only
//     reason its passing means anything (§5 of CLAUDE.md).
//
// The §5 claim was verified rather than asserted: clearing `freem` inside the
// scan instead of after it is **bit-identical** — same worst error, same
// block. The scan consumes a snapshot and a counter and never reads `freem`,
// so the read-before-clear bug really is unwritable in this form. (The first
// attempt at that mutant XORed the mask twice and failed; a mutant that fails
// unexpectedly is a broken mutant until proven otherwise.)
//
// NOT RUN, and none of it is optional — this check is a smoke test, not V0:
//   * the real 383 classes of `FastDecoder`, with the real `walk_radices`
//     widths. Everything above used synthetic shapes;
//   * the round-trip standard of pre-registration §3 (`rank → arrangement →
//     rank` over every rank of the small classes, plus pairwise distinctness
//     and a count equal to the cardinality) — that is the arm's actual V0;
//   * the fixture for the origin and the shell-13 classes, which no real draw
//     reaches (98 of 384 table entries);
//   * the integral sweep of the 150 681 600 blocks;
//   * real blocks from `~/llvq-q4b.llvq`, and therefore the real class mix and
//     the real inter-lane divergence — the whole reason P1 exists;
//   * one nanosecond of timing.
// ===========================================================================

// ---- PRELUDE: delete these lines when appending to llvq_metal::PAYLOAD_MSL,
// ---- which already defines DIM, Cursor and take verbatim.
#include <metal_stdlib>
using namespace metal;

constant uint DIM = 24;

struct Cursor { ulong lo; uint hi; };

static inline uint take(thread Cursor &c, uint width) {
    if (width == 0) return 0u;
    uint v = uint(c.lo & ((1ul << width) - 1ul));
    c.lo = (c.lo >> width) | (ulong(c.hi) << (64u - width));
    c.hi = c.hi >> width;
    return v;
}
// ---- end PRELUDE ----

constant uint MAX_KINDS    = 8;        // fastdec::MAX_KINDS; the even side's
                                       // non-support problem needs all eight
constant uint BINOM_STRIDE = DIM + 1;  // 25 — one row per kind count

// Per-class record for the walk. 52 bytes, alignment 4, padding explicit.
// Rust mirror to be written alongside `GpuClassRec`, in the same crate as this
// source so the two cannot drift:
//
//     #[repr(C)]
//     pub struct GpuWalkRec {
//         pub vals:   [f32; 8],  // level value per kind, ALREADY / sqrt(16·m)
//         pub counts: [u8; 8],   // slots this kind takes; Σ = 24
//         pub wbits:  [u8; 8],   // ceil(log2 radix_j); 0 for the final kind
//         pub k:      u8,        // kinds in use, ≥ 1
//         pub _pad:   [u8; 3],
//     }
//
// Table convention identical to `gpu_class_table`: entry 0 is the origin,
// entry 1+ci is class ci, 384 entries, 19 968 bytes.
struct WalkRec {
    float vals[MAX_KINDS];
    uchar counts[MAX_KINDS];
    uchar wbits[MAX_KINDS];
    uchar k;
    uchar pad[3];
};

// One kind's contribution, once its slots are known. Same shape as
// PAYLOAD_MSL's `lrun`, with slot-major signs instead of a level-major shift:
// the mask comes off its word with ctz, and slots this kind does not own are
// never touched. Exactly `popcount(mask)` iterations, and summed over the
// kinds that is 24 — one FMA per slot, no more.
static inline float dep_slots(uint mask,
                              float v,
                              uint smask,
                              threadgroup const float* xs)
{
    float d = 0.0f;
    while (mask != 0u) {
        uint s = ctz(mask);
        mask &= mask - 1u;
        d = fma(((smask >> s) & 1u) ? -v : v, xs[s], d);
    }
    return d;
}

// ---------------------------------------------------------------------------
// The arm. One block per lane, one float out, activations staged once per
// threadgroup. No cross-lane operation: this prices a decode, not a matvec.
// ---------------------------------------------------------------------------
kernel void decode_walk(device const uint*   words  [[buffer(0)]],
                        constant WalkRec*    tab    [[buffer(1)]],
                        constant uint*       binom  [[buffer(2)]],
                        constant float*      gscale [[buffer(3)]],
                        device const float*  x      [[buffer(4)]],
                        device float*        out    [[buffer(5)]],
                        uint gid [[thread_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]])
{
    threadgroup float xs[DIM];
    if (tid < DIM) xs[tid] = x[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Three aligned words, fixed stride — `decode_f96`'s addressing.
    uint w0 = words[3u * gid], w1 = words[3u * gid + 1u], w2 = words[3u * gid + 2u];
    Cursor cur = { (ulong(w1) << 32) | ulong(w0), w2 };

    uint id    = take(cur, 9u);
    uint gain  = take(cur, 1u);        // 1 gain bit — see §8
    uint smask = take(cur, DIM);       // slot-major signs, stand-in — see §8
    constant WalkRec &r = tab[id];

    uint k     = r.k;                  // ≥ 1 by the record's contract
    uint freem = (1u << DIM) - 1u;     // bit i set ⇒ slot i still free
    float d    = 0.0f;

    for (uint j = 0u; j + 1u < k; ++j) {
        // Read the field first, unconditionally: a zero-count kind has radix 1
        // hence width 0, so this consumes nothing and the cursor's advance
        // keeps no data-dependent shape.
        uint rank = take(cur, r.wbits[j]);
        uint cnt  = r.counts[j];
        if (cnt == 0u) continue;

        // Row of the transposed table: row[p] == C(p, cnt), zero when cnt > p.
        // Placing a unit decrements cnt, which is one pointer subtraction —
        // no multiply in the scan.
        constant uint* row = binom + cnt * BINOM_STRIDE;

        // Colex unranking, walked in SLOT space. `m` is a snapshot; `freem`
        // is not touched until the scan ends (§5). `p` is the free-position
        // index of the slot under the cursor, a plain counter — this is the
        // mapping the CPU pays `nth_set_bit` for.
        uint taken = 0u;
        uint p = popcount(freem);
        uint m = freem;
        while (m != 0u) {
            uint s = 31u - clz(m);         // highest free slot
            m ^= 1u << s;
            --p;

            uint b = row[p];               // compare-and-subtract; no division
            bool hit = (rank >= b) && (cnt != 0u);
            uint one = hit ? 1u : 0u;

            rank  -= hit ? b : 0u;
            row   -= one * BINOM_STRIDE;
            cnt   -= one;
            taken |= hit ? (1u << s) : 0u;
        }

        // Cleared only now, and in one operation. There is no window in which
        // a partially cleared mask could be read.
        freem ^= taken;
        d += dep_slots(taken, r.vals[j], smask, xs);
    }

    // The final kind takes what is left and reads no rank at all (§6).
    d += dep_slots(freem, r.vals[k - 1u], smask, xs);

    out[gid] = d * gscale[gain];
}
