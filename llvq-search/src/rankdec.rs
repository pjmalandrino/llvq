//! The two candidate rank decoders P1 puts on the bench, as CPU references.
//!
//! Both answer the same shape of question — *which kind sits in each of the 24
//! slots?* — and they differ in what they are allowed to cost. The incumbent,
//! [`crate::fastdec::unrank_multiset`], spends **one multiply and one divide
//! per candidate** and has a data-dependent inner trip count. That divide, and
//! that variable trip count, are what a GPU pays for twice.
//!
//! ## The two arms, and why their standards differ
//!
//! [`cascade_uniform`] decodes **the archive's order**: same rank in, same
//! arrangement out as `unrank_multiset`. It is verified by equality against it,
//! and through it against `Indexer::decode`. What it changes is the *shape* of
//! the work — 24 identical iterations for every class, every candidate
//! evaluated, no early exit, no `continue`, selection by arithmetic rather than
//! by branch, and the divide replaced by a per-stage magic reciprocal.
//!
//! [`binomial_walk`] decodes **its own order**, and this is not a detail. The
//! archive rank is a *multiset permutation rank*; a binomial walk unranks in
//! *combinatorial* order over subsets. Relating the two **is** the CNS
//! re-bijection — that is P5, which this bench gates. Requiring equality with
//! `fd.decode` from this arm would demand the transcoder its own verdict
//! authorises: a circularity, and a V0 it could never pass. Its standard is
//! therefore a **round trip on its own bijection** plus a **proof that it is
//! one** — the arrangements it produces over `0..cardinality` are pairwise
//! distinct and exactly as many. Fixed in `proofs/preregistration-p1-2026-08-13.md`,
//! amendment É2.
//!
//! ## What "zero division" buys, and what it costs
//!
//! `binomial_walk` reads a table of `C(n, k)` for `n ≤ 24` — 5 KB, L1-resident
//! — and spends compare-and-subtract. It never divides. The price is paid in
//! the *format*: its per-kind ranks are **separate bit fields**, not one packed
//! integer, because peeling a packed mixed-radix integer needs the division
//! this arm exists to avoid. That is the `e1v-séparé` variant, and it is why it
//! costs 53,332 bits/block where the packed one costs 52,975.
//!
//! Neither of these is a speed claim. This module produces arrangements; the
//! nanoseconds are the bench's business, and no count of source-level
//! operations has ever survived contact with a card in this repository.

use crate::fastdec::MAX_KINDS;
use llvq_core::DIM;

// ---------------------------------------------------------------------------
// Exact division by a small known divisor, without dividing
// ---------------------------------------------------------------------------

/// `(multiplier, shift)` per divisor `n ∈ 1..=24`, so that
/// `x / n == ((x as u128 * mul) >> shift) as u64` for every `x < 2^63`.
///
/// The cascade's divisor is the number of slots left, which runs 24, 23, … 1
/// **regardless of the class**. One table of 25 entries therefore covers every
/// class — the per-class table the bench brief sketched would hold 383 copies
/// of these same 24 pairs.
static MAGIC: [(u128, u32); 25] = build_magic();

const fn build_magic() -> [(u128, u32); 25] {
    let mut t = [(0u128, 0u32); 25];
    let mut n = 1usize;
    while n <= 24 {
        // shift = 64 + ceil(log2 n) is enough for every x < 2^63; the +1 in the
        // multiplier is the standard round-up that makes the identity exact.
        let mut bits = 0u32;
        while (1usize << bits) < n {
            bits += 1;
        }
        let shift = 64 + bits;
        t[n] = (((1u128 << shift) / n as u128) + 1, shift);
        n += 1;
    }
    t
}

/// `x / n`, by multiply-high. Exact for `x < 2^63` and `1 ≤ n ≤ 24`, pinned by
/// `magic_division_is_exact`.
#[inline]
fn magic_div(x: u64, n: usize) -> u64 {
    let (mul, sh) = MAGIC[n];
    ((x as u128 * mul) >> sh) as u64
}

/// Branch-free select: `if p { a } else { b }`, written as arithmetic so the
/// shape carries to a shader where a branch would diverge across lanes.
#[inline]
fn sel(p: bool, a: u64, b: u64) -> u64 {
    let m = (p as u64).wrapping_neg(); // 0 or all-ones
    (a & m) | (b & !m)
}

// ---------------------------------------------------------------------------
// Arm 1 — the uniformised cascade
// ---------------------------------------------------------------------------

/// The archive's arrangement, in a fixed number of identical steps.
///
/// Same contract as [`crate::fastdec::unrank_multiset`]: same `rank`, same
/// `counts`, same `m0`, same slots out. `out.len()` slots are filled, and
/// `counts` is consumed.
///
/// **What is uniform.** The outer loop always runs `out.len()` times. The inner
/// loop always runs `MAX_KINDS` times — a kind with zero left contributes a
/// zero-width interval and is skipped by arithmetic, not by `continue`. There
/// is no early exit when the slot is decided: the remaining candidates are
/// evaluated and discarded. Nothing indexes a table with a value derived from
/// the data except the final `counts[chosen] -= 1`.
///
/// **What that costs on a CPU** is obvious and irrelevant: it does strictly
/// more work than the incumbent. The bet is on a machine where 32 lanes must
/// agree on a trip count, and it is the bench that settles it.
pub fn cascade_uniform(rank: u64, counts: &mut [u8; MAX_KINDS], k: usize, m0: u64, out: &mut [u8]) {
    debug_assert!(k <= MAX_KINDS);
    let mut m = m0;
    let mut rank = rank;
    let n_slots = out.len();

    for (slot, o) in out.iter_mut().enumerate() {
        let n = n_slots - slot;
        let mut acc = 0u64; // running low edge of the current candidate interval
        let mut chosen = 0u64;
        let mut new_m = m;
        let mut base = 0u64;

        for (j, &c) in counts.iter().enumerate().take(MAX_KINDS) {
            let c = c as u64;
            // Exact: n divides m·c at every step (the invariant the module
            // header of `fastdec` states). A count of zero gives mj = 0, so the
            // interval is empty and `inside` is false for free.
            let mj = magic_div(m * c, n);
            let lo = acc;
            let hi = acc + mj;
            let inside = rank >= lo && rank < hi;
            chosen = sel(inside, j as u64, chosen);
            new_m = sel(inside, mj, new_m);
            base = sel(inside, lo, base);
            acc = hi;
        }

        rank -= base;
        m = new_m;
        counts[chosen as usize] -= 1;
        *o = chosen as u8;
    }
}

// ---------------------------------------------------------------------------
// Arm 2 — the binomial walk
// ---------------------------------------------------------------------------

/// `C(n, k)` for `n ≤ 24`, the whole table a walk ever reads. 5 KB — L1.
static BINOM: [[u64; DIM + 1]; DIM + 1] = build_binom();

const fn build_binom() -> [[u64; DIM + 1]; DIM + 1] {
    let mut t = [[0u64; DIM + 1]; DIM + 1];
    let mut n = 0usize;
    while n <= DIM {
        t[n][0] = 1;
        let mut k = 1usize;
        while k <= n {
            t[n][k] = t[n - 1][k - 1] + if k < n { t[n - 1][k] } else { 0 };
            k += 1;
        }
        n += 1;
    }
    t
}

/// `C(n, k)`, zero when `k > n`.
#[inline]
pub fn binom(n: usize, k: usize) -> u64 {
    if k > n {
        0
    } else {
        BINOM[n][k]
    }
}

/// How many arrangements a class holds — the product of the walk's own radices.
///
/// Equal to the multiset permutation count, and therefore to the archive's
/// cardinality for the same class: the two orders differ, the *set* does not.
/// That equality is what makes a re-bijection possible at all, and
/// `the_walk_spans_exactly_the_class` checks it rather than assuming it.
pub fn walk_cardinality(counts: &[u8; MAX_KINDS], k: usize, n_slots: usize) -> u64 {
    let mut free = n_slots;
    let mut total = 1u64;
    for &c in counts.iter().take(k) {
        total *= binom(free, c as usize);
        free -= c as usize;
    }
    total
}

/// The radix of kind `j` — the number of distinct choices it has once the
/// kinds before it have taken their slots. The **separate** field width the
/// `e1v-séparé` format stores.
///
/// The last kind has radix 1: it takes what is left, and storing a rank for it
/// would be storing a constant.
pub fn walk_radices(counts: &[u8; MAX_KINDS], k: usize, n_slots: usize) -> [u64; MAX_KINDS] {
    let mut r = [1u64; MAX_KINDS];
    let mut free = n_slots;
    for j in 0..k {
        let c = counts[j] as usize;
        r[j] = binom(free, c);
        free -= c;
    }
    r
}

/// Decode a **separated** rank: one colex subset rank per kind, in its own
/// field. No division anywhere — only table lookups, compares and subtracts.
///
/// `ranks[j]` must be `< walk_radices(counts, k)[j]`; the last kind's field is
/// ignored, since it has no choice left.
///
/// Fixed-count by construction: kind `j` scans exactly the number of slots free
/// when it runs, so the total step count is a function of the **class**, never
/// of the rank. `the_step_count_depends_only_on_the_class` pins that, because
/// a walk whose length varied with the data would diverge across lanes exactly
/// where the cascade does.
pub fn binomial_walk(
    ranks: &[u64; MAX_KINDS],
    counts: &[u8; MAX_KINDS],
    k: usize,
    out: &mut [u8],
) {
    let mut steps = 0u32;
    binomial_walk_counted(ranks, counts, k, out, &mut steps);
}

/// [`binomial_walk`], plus the count of **outer scan iterations** it took.
///
/// This is P5's C3.1 instrument, and it is the *same code path* as the walk —
/// not a twin. `rankdec`'s previous instrumented twin, `steps_of`, was written
/// with the read-before-clear bug its original does not have, and the lesson
/// recorded then was that *a twin that drifts from its original tests the
/// twin*. So there is no twin: one implementation, two entry points, and the
/// counter cannot describe a walk that did not happen.
///
/// A "step" is one iteration of the scan over free positions, summed over every
/// kind of the arrangement. It is a function of the **class** and never of the
/// rank — the property [`crate::cns`] turns into an opposable criterion.
pub fn binomial_walk_counted(
    ranks: &[u64; MAX_KINDS],
    counts: &[u8; MAX_KINDS],
    k: usize,
    out: &mut [u8],
    steps: &mut u32,
) {
    debug_assert!(k <= MAX_KINDS);
    let n_slots = out.len();
    debug_assert!(n_slots <= DIM);
    debug_assert_eq!(
        counts.iter().take(k).map(|&c| c as usize).sum::<usize>(),
        n_slots,
        "the counts must fill exactly the slots on offer"
    );
    // Bit `i` set ⇒ slot `i` still free.
    let mut free_mask: u32 = if n_slots == 32 { u32::MAX } else { (1u32 << n_slots) - 1 };
    out.fill(0);

    for j in 0..k {
        let c_total = counts[j] as usize;
        if c_total == 0 {
            continue;
        }
        let n_free = free_mask.count_ones() as usize;
        if j + 1 == k {
            // The last kind takes every slot still free: no rank is read, and
            // none is stored. This is the radix-1 field of `walk_radices`.
            let mut mask = free_mask;
            while mask != 0 {
                let i = mask.trailing_zeros() as usize;
                out[i] = j as u8;
                mask &= mask - 1;
            }
            free_mask = 0;
            continue;
        }

        // Colex unranking of a `c_total`-subset of the `n_free` free positions,
        // scanned from the top. `C(p, c)` is the number of subsets whose
        // largest element is below `p`.
        let mut r = ranks[j];
        let mut c = c_total;
        let mut taken: u32 = 0;
        // ⚠️ NO early exit. The whole justification of this arm is that its
        // step count is a function of the class and never of the rank — a walk
        // that stopped as soon as its last unit was placed would diverge across
        // lanes exactly where the cascade already does, which is what it exists
        // to avoid. When `c` reaches zero the threshold is set out of reach
        // instead, so the remaining positions are scanned and never taken.
        //
        // 🕳️ **The `u64::MAX` itself is inert on well-formed input, and a
        // surviving mutant is what says so.** Replacing it with plain
        // `binom(p, c)` passes the integral sweep — 150 681 600 real blocks,
        // `llvq-artifact/tests/p1_rank_sweep.rs` — because colex unranking
        // consumes its rank exactly: when `c` reaches zero the residual `r` is
        // zero, and `binom(p, 0) == 1 > 0` is already out of reach. So this is
        // not the line that buys the fixed step count; the absence of a `break`
        // is, and the mutant preserves that too.
        //
        // It is kept because it is **not** inert on malformed input, which the
        // *format* admits: a rank field is `ceil(log2 radix)` bits wide, so
        // every value in `[radix, 2^wbits)` is representable in `e1v-séparé`
        // and none of them leaves `r` at zero. Without the guard `c` would
        // underflow and the walk would run off the class. `binomial_walk.metal`
        // carries the same guard as `&& (cnt != 0u)`, for the same reason and
        // at the same cost — one compare and one AND per step, in the arm's hot
        // loop. Whether the timed arm should pay it is a question for the bench
        // brief, not something to settle by deleting the line here.
        for p in (0..n_free).rev() {
            *steps += 1;
            let b = if c > 0 { binom(p, c) } else { u64::MAX };
            let take = r >= b;
            if take {
                r -= b;
                taken |= 1 << p;
                c -= 1;
            }
        }
        debug_assert_eq!(c, 0, "the walk must place every unit of the kind");

        // `taken` indexes *free positions*; map them back to slot indices.
        //
        // ⚠️ The mapping reads `free_mask` **as it was before this kind ran**,
        // and clears bits only after: taking the n-th free slot from a mask one
        // is simultaneously clearing would shift every later index by one, and
        // the arrangement would come out permuted without any test on ranks
        // seeing it.
        let before = free_mask;
        let mut mask = taken;
        while mask != 0 {
            let want = mask.trailing_zeros() as usize;
            let slot = nth_set_bit(before, want);
            out[slot] = j as u8;
            free_mask &= !(1u32 << slot);
            mask &= mask - 1;
        }
    }
}

/// Index of the `n`-th set bit of `x`, counting from zero.
#[inline]
fn nth_set_bit(mut x: u32, n: usize) -> usize {
    for _ in 0..n {
        x &= x - 1;
    }
    x.trailing_zeros() as usize
}

/// The inverse of [`binomial_walk`] — the ranks an arrangement corresponds to.
///
/// This is the half a transcoder would run, and the half that makes the round
/// trip a real test rather than a restatement: `walk → rank → walk` closing on
/// itself proves nothing, `rank → walk → rank` over every rank of a class
/// proves the map is injective on that class.
pub fn binomial_rank(
    arrangement: &[u8],
    counts: &[u8; MAX_KINDS],
    k: usize,
) -> [u64; MAX_KINDS] {
    let n_slots = arrangement.len();
    let mut ranks = [0u64; MAX_KINDS];
    let mut free_mask: u32 = if n_slots == 32 { u32::MAX } else { (1u32 << n_slots) - 1 };

    for j in 0..k.saturating_sub(1) {
        let c_total = counts[j] as usize;
        if c_total == 0 {
            continue;
        }
        // Positions of this kind, expressed among the currently free slots.
        let mut r = 0u64;
        let mut c = c_total;
        let n_free = free_mask.count_ones() as usize;
        for p in (0..n_free).rev() {
            let slot = nth_set_bit(free_mask, p);
            if arrangement[slot] == j as u8 {
                r += binom(p, c);
                c -= 1;
            }
        }
        debug_assert_eq!(c, 0);
        ranks[j] = r;
        for (slot, &a) in arrangement.iter().enumerate() {
            if a == j as u8 {
                free_mask &= !(1u32 << slot);
            }
        }
    }
    ranks
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fastdec::{unrank_multiset, FastDecoder};
    use llvq_core::SplitMix64;

    #[test]
    fn magic_division_is_exact() {
        let mut rng = SplitMix64::new(0x0_D1_D0);
        for n in 1..=24usize {
            for _ in 0..20_000 {
                let x = rng.next() >> 1; // < 2^63
                assert_eq!(magic_div(x, n), x / n as u64, "x={x} n={n}");
            }
            for x in [0u64, 1, n as u64 - 1, n as u64, u64::MAX >> 1] {
                assert_eq!(magic_div(x, n), x / n as u64, "edge x={x} n={n}");
            }
        }
    }

    /// A class's counts, from the real codebook rather than invented.
    fn real_classes() -> Vec<([u8; MAX_KINDS], usize, u64, usize)> {
        let fd = FastDecoder::new();
        let mut out = Vec::new();
        for ci in 0..fd.n_classes() {
            let c = fd.cascade_class(ci);
            if c.odd {
                out.push((c.counts, c.n_kinds, c.m_arr, DIM));
            } else {
                let mut wc = [0u8; MAX_KINDS];
                wc[..4].copy_from_slice(&c.word_counts);
                out.push((wc, c.n_word, c.a_on, c.w as usize));
                out.push((c.off_counts, c.n_off, c.a_off, DIM - c.w as usize));
            }
        }
        out
    }

    /// Arm 1 must be indistinguishable from the incumbent — that is the whole
    /// point of it being an *arm* and not a format.
    #[test]
    #[cfg_attr(debug_assertions, ignore = "sweeps every class, run in release")]
    fn cascade_uniform_matches_the_archive() {
        let mut rng = SplitMix64::new(0x0_CA5C);
        for (counts, k, m0, n_slots) in real_classes() {
            if m0 == 0 || n_slots == 0 {
                continue;
            }
            let probes: Vec<u64> = [0, m0 - 1, m0 / 2]
                .into_iter()
                .chain((0..40).map(|_| rng.next() % m0))
                .collect();
            for r in probes {
                let (mut a, mut b) = (counts, counts);
                let (mut oa, mut ob) = ([0u8; DIM], [0u8; DIM]);
                unrank_multiset(r, &mut a, k, m0, &mut oa[..n_slots]);
                cascade_uniform(r, &mut b, k, m0, &mut ob[..n_slots]);
                assert_eq!(oa, ob, "rank {r}, counts {counts:?}, k {k}");
            }
        }
    }

    /// Arm 2 spans exactly the class: as many arrangements as the archive has,
    /// and the same *set* of them — in another order.
    #[test]
    #[cfg_attr(debug_assertions, ignore = "enumerates small classes fully, run in release")]
    fn the_walk_spans_exactly_the_class() {
        use std::collections::HashSet;
        for (counts, k, m0, n_slots) in real_classes() {
            if m0 == 0 || m0 > 200_000 {
                continue; // full enumeration only where it is affordable
            }
            assert_eq!(
                walk_cardinality(&counts, k, n_slots),
                m0,
                "the walk must span the same set as the archive: counts {counts:?}"
            );
            let radices = walk_radices(&counts, k, n_slots);
            let mut seen = HashSet::new();
            let mut idx = [0u64; MAX_KINDS];
            let mut n = 0u64;
            loop {
                let mut out = [0u8; DIM];
                binomial_walk(&idx, &counts, k, &mut out[..n_slots]);
                assert!(
                    seen.insert(out),
                    "collision: the walk is not injective"
                );
                // Also: the arrangement must realise the class's multiset.
                for (j, &c) in counts.iter().enumerate().take(k) {
                    assert_eq!(
                        out[..n_slots].iter().filter(|&&x| x == j as u8).count(),
                        c as usize,
                        "kind {j} misplaced"
                    );
                }
                n += 1;
                // odometer over the separated fields
                let mut j = 0;
                loop {
                    if j >= k {
                        break;
                    }
                    idx[j] += 1;
                    if idx[j] < radices[j] {
                        break;
                    }
                    idx[j] = 0;
                    j += 1;
                }
                if j >= k {
                    break;
                }
            }
            assert_eq!(n, m0, "the walk must produce exactly {m0} arrangements");
        }
    }

    /// The round trip on arm 2's **own** bijection — its standard, per É2.
    #[test]
    #[cfg_attr(debug_assertions, ignore = "many draws over every class, run in release")]
    fn the_walk_round_trips_on_its_own_bijection() {
        let mut rng = SplitMix64::new(0x0_B1_A0);
        for (counts, k, m0, n_slots) in real_classes() {
            if m0 == 0 {
                continue;
            }
            let radices = walk_radices(&counts, k, n_slots);
            for _ in 0..200 {
                let mut idx = [0u64; MAX_KINDS];
                for j in 0..k {
                    idx[j] = if radices[j] > 1 { rng.next() % radices[j] } else { 0 };
                }
                let mut out = [0u8; DIM];
                binomial_walk(&idx, &counts, k, &mut out[..n_slots]);

                // 🚨 The output must BE an arrangement of the class — every
                // kind present in its exact count, the LAST ONE INCLUDED, and
                // no symbol outside `0..k`.
                //
                // This lives here, not in the enumeration test, because the
                // round trip cannot see it: `binomial_rank` walks `0..k-1`,
                // the last kind having no rank of its own. A mutation that
                // wrote `j + 1` for the last kind survived all five tests
                // before this assertion existed — it produced arrangements
                // where the last kind was absent and 17 to 19 slots carried a
                // symbol out of range, on 0,11 % of the codebook.
                for (j, &c) in counts.iter().enumerate().take(k) {
                    assert_eq!(
                        out[..n_slots].iter().filter(|&&x| x == j as u8).count(),
                        c as usize,
                        "kind {j} misplaced: counts {counts:?}, idx {idx:?}"
                    );
                }
                assert!(
                    out[..n_slots].iter().all(|&x| (x as usize) < k),
                    "a slot carries a symbol outside 0..{k}: counts {counts:?}"
                );

                let back = binomial_rank(&out[..n_slots], &counts, k);
                for j in 0..k.saturating_sub(1) {
                    assert_eq!(back[j], idx[j], "kind {j}: counts {counts:?}, idx {idx:?}");
                }
            }
        }
    }

    /// A walk whose length varied with the rank would diverge across lanes
    /// exactly where the cascade already does — which would make the arm
    /// pointless. The count is a function of the class alone.
    #[test]
    #[cfg_attr(debug_assertions, ignore = "counts steps over every class, run in release")]
    fn the_step_count_depends_only_on_the_class() {
        let mut rng = SplitMix64::new(0x0_57E7);
        for (counts, k, m0, n_slots) in real_classes() {
            if m0 == 0 {
                continue;
            }
            // Steps the walk takes: for each non-final kind, the number of free
            // slots it scans. No rank enters this expression — that IS the test.
            let mut want = 0usize;
            let mut free = n_slots;
            for &c in counts.iter().take(k.saturating_sub(1)) {
                if c > 0 {
                    want += free;
                    free -= c as usize;
                }
            }
            let radices = walk_radices(&counts, k, n_slots);
            for _ in 0..20 {
                let mut idx = [0u64; MAX_KINDS];
                for j in 0..k {
                    idx[j] = if radices[j] > 1 { rng.next() % radices[j] } else { 0 };
                }
                // 🚨 Counted by the WALK ITSELF, not by a twin. The first
                // version of this test called an instrumented copy
                // (`steps_of`), which was written with a bug the original does
                // not have and had to be debugged against it — so the test
                // measured the copy. `binomial_walk_counted` is the same code
                // path as `binomial_walk`; there is nothing left to drift.
                let mut out = [0u8; DIM];
                let mut steps = 0u32;
                binomial_walk_counted(&idx, &counts, k, &mut out[..n_slots], &mut steps);
                assert_eq!(
                    steps as usize, want,
                    "step count moved with the rank: counts {counts:?}"
                );
            }
        }
    }

}
