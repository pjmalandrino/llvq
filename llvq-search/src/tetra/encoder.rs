//! The production Tetra encoder: the bench's rank-region rule at a few
//! hundred microseconds a block on one core — `llvq-bench/examples/f1enclazy.rs`
//! brought into the dependency-free crate and written against [`Tetra`]
//! (roadmap §2.2 quater, step 1).
//!
//! ## The rule it reproduces
//!
//! The yardstick is the bench's `Codebook<RankRegion>` (`examples/f1rankbench.rs`,
//! held verbatim in `llvq_bench::f1::rankbook`). At one scale `s` it solves
//! the 1,536 (section, `p`, pattern, k-parity) problems with one candidate
//! rule — round the target `x/s` on the offset lattice `o + 4Z⁸`, repair the
//! parity of `Σk` on the least-decided coordinate, try the sixteen
//! two-coordinate re-roundings, at seven shrinks of the target for the end
//! sections and eight (the last one zero) for the middle; keep the nearest
//! candidate that is a member of the section's row set, replacing on strict
//! `<` only; the end sections are seeded with the lowest-norm member of their
//! region, the fallback — then takes the best path through the trellis,
//! `e1[r][s8] + e2[b][δ] + e3[p ⊕ r ⊕ δ][s16]`, over both block parities.
//! Distances are always to the unshrunk target.
//!
//! ## What makes it cheap
//!
//! Membership is the closed form of [`Bound`] on a packed key, no set. The
//! shrink-1.0 base of a pattern is the exact optimum of its parity coset, so
//! when it is a member the solve is finished, and when it is not its
//! distance is a LOWER BOUND on the pattern's answer. The join first runs
//! over bounds: at most two seed joins resolve the sections of the best
//! bound path and give a feasible cost `U`; one pass over the 8,192 paths
//! gives every entry the best path through it; only the entries whose best
//! path is under `U` are solved in full, the others cannot win and are
//! dropped. On Gaussian blocks 13.6 % of the entry slots are solved in
//! full, against 45.3 % with a bound of zero, which prunes nothing —
//! `the_join_solves_a_small_fraction_of_the_entries` in this file pins the
//! rate, so that mutant fails a test rather than a stopwatch.
//!
//! Nothing that could beat `U` is pruned, so the answer is the rule's
//! answer **up to a tie**: the rule keeps the first candidate of its own
//! enumeration among several at exactly equal distance, and this
//! enumeration is not the bench's, so on an exact f64 tie the two can end
//! on different points of the same cost. Checked point for point against
//! the yardstick on the 2,000 evaluation blocks at three scales — 6,000 of
//! 6,000 pairs at the bench's own point on 2026-09-05, no tie observed, but
//! the test accepts one at equal cost (`llvq-bench/tests/tetra_encoder.rs`).
//!
//! ## The two scales
//!
//! The rule's answer depends on `s`, and the best `s` moves from block to
//! block beyond what `‖x‖` predicts. [`Encoder::encode`] runs two adaptive
//! scales, `ALPHA·‖x‖/√24` and `RATIO` times it, and keeps the point of larger
//! `t = ⟨x, y⟩/‖y‖`. The pair is pinned on the 4,000 TRAINING blocks of the
//! F1b seed by `llvq-bench/examples/tetrascales.rs` ([`Encoder::ALPHA`]); the
//! prototype had fixed it on the evaluation blocks.
//!
//! ## Orders and the word
//!
//! `x` comes in the repository's NATURAL order; the encoder permutes it to
//! trio order, works there, and returns the point in natural order together
//! with its word. The point is assembled by the decoder's own formula from
//! the `(p, pattern, row)` it chose — `val(p + 2·c_j, ρ_j)` — and the word
//! from the same three rows, so `Tetra::decode(word) == point` holds by
//! construction; `Tetra::encode(&point) == Some(word)` is what the tests pin.
//! The origin carries `t = −∞` (the bench's `t_of`), so a scale that lands
//! there never wins; `encode` reaches it only for `x = 0`.

use super::trellis::{BRANCHES, GOLAY_STATES};
use super::{pack, rank_class, rank_of, unpack, val, Bound, Fields, Tetra, CLASS_BOUNDS, CLASS_ROWS, MIXED_BOUND, N0_MIXED, SECTION};
use llvq_core::DIM;

/// The shrinks of the candidate rule, in the order the bench applies them.
const SHRINKS: [f64; 8] = [1.0, 0.85, 0.7, 0.55, 0.4, 0.25, 0.1, 0.0];

/// The end sections' rule stops before the zero shrink: their fallback
/// already covers "nothing near the target is a member".
const END_SHRINKS: usize = 7;

/// The middle section runs all eight: the zero shrink is what makes every
/// `(pattern, δ)` feasible, since the origin-nearest point of a parity coset
/// costs at most 16 and every such row is in the mixed set.
const MID_SHRINKS: usize = 8;

/// Seed joins before the pruning pass.
const WARM_JOINS: usize = 2;

/// Distinct sets of sixteen middle bytes over the 64 states — a counted
/// fact of the trellis, asserted by [`Encoder::new`].
const MSETS: usize = 8;

/// End-section entries: `[r][state]`.
const END_ENTRIES: usize = 2 * GOLAY_STATES;

/// Middle entries: `[mset][byte]`, each with two parities.
const MID_ENTRIES: usize = MSETS * BRANCHES;

/// Rank of `o + 4k` for `k` in `−8..=8`; outside, the rank is past the table.
const K_REACH: i32 = 8;

/// A rank no row holds: `(2·15 + 1)² = 961` fails every bound on its own.
const RANK_NONE: u8 = 15;

/// How many shrinks each section runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rule {
    pub end: usize,
    pub mid: usize,
}

impl Rule {
    /// The bench's rule: what the yardstick test compares against.
    pub const BENCH: Rule = Rule { end: END_SHRINKS, mid: MID_SHRINKS };
}

/// A [`Bound`] on packed keys: `cost < C`, or `cost == C` and the key is at
/// or below the cut in lexicographic order of the ranks.
#[derive(Clone, Copy)]
struct Closed {
    cost: u32,
    cut_lex: u32,
}

impl Closed {
    fn of(b: &Bound) -> Self {
        Self { cost: b.cost, cut_lex: lex(pack(&b.cut)) }
    }

    #[inline(always)]
    fn holds(&self, cost: u32, key: u32) -> bool {
        cost < self.cost || (cost == self.cost && lex(key) <= self.cut_lex)
    }
}

/// The eight nibbles of a packed key reversed, so that the integer order of
/// the result is the lexicographic order of `(ρ_0, …, ρ_7)` — `pack` puts
/// `ρ_0` in the LOW nibble, the cut compares from `ρ_0` down.
#[inline(always)]
fn lex(key: u32) -> u32 {
    let swapped = ((key & 0x0f0f_0f0f) << 4) | ((key >> 4) & 0x0f0f_0f0f);
    swapped.swap_bytes()
}

/// One coordinate under one offset at one shrink: the rounding of the shrunk
/// target, and the five values `k0 − 2 ..= k0 + 2` with their rank, weight
/// `(2ρ + 1)²`, and squared distance to the UNSHRUNK target coordinate.
#[derive(Clone, Copy, Default)]
struct Coord {
    k0: i32,
    dir: i32,
    regret: f64,
    rank: [u8; 5],
    w: [u16; 5],
    d: [f64; 5],
}

/// Per (section target, `p`): a [`Coord`] for every shrink, coordinate and
/// offset bit `c_j`. Rebuilt per block parity; ~9 KiB.
struct Tables {
    t: [[[Coord; 2]; SECTION]; 8],
}

impl Tables {
    const fn empty() -> Self {
        Self { t: [[[Coord { k0: 0, dir: 0, regret: 0.0, rank: [0; 5], w: [0; 5], d: [0.0; 5] }; 2]; SECTION]; 8] }
    }

    fn build(&mut self, target: &[f64; SECTION], p: u32, ranks: &[[u8; 2 * K_REACH as usize + 1]; 4]) {
        for (layer, &shrink) in self.t.iter_mut().zip(SHRINKS.iter()) {
            for (j, coord) in layer.iter_mut().enumerate() {
                for (cbit, e) in coord.iter_mut().enumerate() {
                    let o = p + 2 * cbit as u32;
                    // A target no block ever produces still must not overflow
                    // the integer arithmetic below.
                    let z = ((target[j] * shrink - o as f64) / 4.0).clamp(-1e6, 1e6);
                    let k0 = z.round() as i32;
                    e.k0 = k0;
                    e.dir = if z > k0 as f64 { 1 } else { -1 };
                    e.regret = (z - k0 as f64).abs();
                    for (m, slot) in (-2i32..=2).enumerate() {
                        let k = k0 + slot;
                        let r = if (-K_REACH..=K_REACH).contains(&k) { ranks[o as usize][(k + K_REACH) as usize] } else { RANK_NONE };
                        e.rank[m] = r;
                        e.w[m] = ((2 * r as u32 + 1) * (2 * r as u32 + 1)) as u16;
                        e.d[m] = ((o as i32 + 4 * k) as f64 - target[j]).powi(2);
                    }
                }
            }
        }
    }
}

/// One section entry as the join sees it: a distance that is exact or a
/// lower bound, and the `(pattern byte, packed row)` of the point when exact.
#[derive(Clone, Copy)]
struct Entry {
    value: f64,
    exact: bool,
    byte: u8,
    key: u32,
}

const INFEASIBLE: Entry = Entry { value: f64::INFINITY, exact: true, byte: 0, key: 0 };

/// The lowest-norm member of an end region, its byte and row, and its point
/// in trio order for the distance.
#[derive(Clone, Copy)]
struct Fallback {
    byte: u8,
    key: u32,
    y: [i32; SECTION],
}

/// Entries the lazy pass had to take from a lower bound to an exact value,
/// and the parity passes that offered them — the pruning of the module doc
/// as a number rather than as a claim.
///
/// Test-only: the encoder must not pay a counter per entry in production,
/// and the three `Scratch::count_*` below are empty functions outside
/// `cfg(test)`.
#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Solved {
    /// `end_resolve` calls, over the `2·END_ENTRIES` end slots of a pass.
    end: u64,
    /// `mid_resolve` calls, over the `2·MID_ENTRIES` middle slots of a pass.
    mid: u64,
    /// Parity passes, two per scale and four per block.
    passes: u64,
}

/// Slots one parity pass offers the solver: `e1` and `e3` end entries, and
/// both `δ` of every middle entry.
#[cfg(test)]
const SLOTS_PER_PASS: u64 = (2 * END_ENTRIES + 2 * MID_ENTRIES) as u64;

/// Per-thread workspace: the coordinate tables of the three sections and the
/// entries of the join. No allocation per block.
pub struct Scratch {
    tabs: [Tables; 3],
    e1: [Entry; END_ENTRIES],
    e2: [[Entry; 2]; MID_ENTRIES],
    e3: [Entry; END_ENTRIES],
    b1: [f64; END_ENTRIES],
    b2: [[f64; 2]; MID_ENTRIES],
    b3: [f64; END_ENTRIES],
    #[cfg(test)]
    solved: Solved,
}

impl Scratch {
    pub fn new() -> Self {
        Self {
            tabs: [Tables::empty(), Tables::empty(), Tables::empty()],
            e1: [INFEASIBLE; END_ENTRIES],
            e2: [[INFEASIBLE; 2]; MID_ENTRIES],
            e3: [INFEASIBLE; END_ENTRIES],
            b1: [0.0; END_ENTRIES],
            b2: [[0.0; 2]; MID_ENTRIES],
            b3: [0.0; END_ENTRIES],
            #[cfg(test)]
            solved: Solved::default(),
        }
    }

    /// One end entry resolved in full. Nothing outside `cfg(test)`.
    #[inline(always)]
    fn count_end(&mut self) {
        #[cfg(test)]
        {
            self.solved.end += 1;
        }
    }

    /// One middle entry resolved in full.
    #[inline(always)]
    fn count_mid(&mut self) {
        #[cfg(test)]
        {
            self.solved.mid += 1;
        }
    }

    /// One parity pass started.
    #[inline(always)]
    fn count_pass(&mut self) {
        #[cfg(test)]
        {
            self.solved.passes += 1;
        }
    }

    /// The counters since the last call, then zero. The unit is the *slot*:
    /// `passes · SLOTS_PER_PASS` is what the solver would have paid with no
    /// pruning at all.
    #[cfg(test)]
    fn take_solved(&mut self) -> Solved {
        core::mem::take(&mut self.solved)
    }
}

impl Default for Scratch {
    fn default() -> Self {
        Self::new()
    }
}

/// A block's code: the word (gain bit 0), the point in natural order, and
/// `t = ⟨x, y⟩/‖y‖` — `−∞` on the origin.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TetraCode {
    pub word: u64,
    pub point: [i32; DIM],
    pub t: f64,
}

/// The best path of a join: its cost and the entries it runs through.
#[derive(Clone, Copy)]
struct Path {
    cost: f64,
    i1: usize,
    i2: usize,
    delta: usize,
    i3: usize,
}

/// The winning path of a block at one scale, with what the word needs.
#[derive(Clone, Copy)]
struct Winner {
    cost: f64,
    p: u32,
    r: u32,
    s8: usize,
    s16: usize,
    delta: u32,
    sections: [(u8, u32); 3],
}

/// The encoder: the trellis in the form the join walks it, the closed-form
/// bounds, the fallbacks, and the row index that turns a chosen row into
/// its field. Built once; `Sync`, so one instance serves every thread with
/// its own [`Scratch`].
pub struct Encoder {
    order: [u32; DIM],
    prefixes: [[u8; 2]; GOLAY_STATES],
    suffixes: [[u8; 2]; GOLAY_STATES],
    branches: [[(u8, u8); BRANCHES]; GOLAY_STATES],
    /// The eight distinct middle-byte sets, each ascending; middle entry
    /// `m·16 + k` is byte `k` of set `m`, and `branch_ix[s8][b2]` names the
    /// entry of branch `b2` out of `s8`.
    msets: [[u8; BRANCHES]; MSETS],
    branch_ix: [[u16; BRANCHES]; GOLAY_STATES],
    class: [Closed; 2],
    mixed: Closed,
    /// `ranks[o][k + K_REACH]`: the rank of `o + 4k`, `RANK_NONE` past rank 7.
    ranks: [[u8; 2 * K_REACH as usize + 1]; 4],
    /// `[((kind · 2 + p) · 2 + r) · 64 + state]`, kind 0 prefixes, 1 suffixes.
    fallback: Vec<Fallback>,
    /// `(packed row, position in the table)`, sorted by row.
    index: Vec<(u32, u16)>,
    rule: Rule,
}

impl Encoder {
    /// Lower adaptive scale, `s₀ = ALPHA·‖x‖/√24`. Pinned on the 4,000
    /// TRAINING blocks of the F1b seed by `llvq-bench/examples/tetrascales.rs`
    /// on 2026-09-05, before the evaluation blocks were read (the raw output
    /// is `tetrascales-2026-09-05.txt` in the session's scratchpad):
    ///
    /// ```text
    /// (a) grille 18 points 0,10·1,14^i, encodeur du banc : rétention 88.73 %
    ///     α = s_gagnante·√24/‖x‖ : quartiles 0.285 / 0.322 / 0.352, déciles 0.234 / 0.379
    ///     →   α* = 0.3218 (médiane)
    /// paires notées sur les MÊMES blocs d'entraînement, règle F1b :
    ///   (α*, 1,14·α*)        banc 88.58 %   production 88.58 %   Δ(a) -0.14 pp   0 blocs à point différent
    ///   (α*/1,07, 1,07·α*)   banc 88.36 %   production 88.36 %   Δ(a) -0.37 pp   0 blocs à point différent
    ///   (α*/1,14, α*)        banc 87.75 %   production 87.75 %   Δ(a) -0.98 pp   0 blocs à point différent
    ///   gagnante : (α*, 1,14·α*) à 88.58 %   →   ALPHA = 0.3218, RATIO = 1.1400
    /// les 2000 blocs d'ÉVALUATION, lus une fois :
    ///   témoin boule-12 + 1 bit de gain : rétention 92.00 %
    ///   Tetra, paire gagnante (α*, 1,14·α*) : rétention 88.89 %   Δ témoin -3.11 pp
    /// ```
    ///
    /// Gaussian retention at 2.000 b/dim on fixed blocks, not a quality
    /// claim; `llvq-bench/tests/tetra_encoder.rs` guards the 88.89 as a
    /// non-regression. The prototype's eval-fixed 0.321 lands on the same
    /// 88.89 (same output, last line): the choice is flat around `α*`.
    pub const ALPHA: f64 = 0.3218;

    /// Upper scale over lower: `s₁ = RATIO·s₀`, pinned with [`Self::ALPHA`].
    pub const RATIO: f64 = 1.14;

    /// The bench's rule.
    pub fn new(tetra: &Tetra) -> Self {
        Self::with_rule(tetra, Rule::BENCH)
    }

    /// The rule truncated to `n` shrinks on every section — the prototype's
    /// fast configuration. A middle entry the truncated rule leaves without
    /// a member is re-solved under the full rule, so every `(pattern, δ)`
    /// stays feasible and the origin stays unreachable for `x ≠ 0`.
    pub fn with_shrinks(tetra: &Tetra, n: usize) -> Self {
        assert!(n >= 1, "the rule needs its first shrink");
        Self::with_rule(tetra, Rule { end: n.min(END_SHRINKS), mid: n.min(MID_SHRINKS) })
    }

    fn with_rule(tetra: &Tetra, rule: Rule) -> Self {
        let (prefixes, suffixes, branches) = (*tetra.prefixes(), *tetra.suffixes(), *tetra.branches());

        // The eight middle-byte sets, in order of first appearance.
        let mut msets: Vec<[u8; BRANCHES]> = Vec::new();
        let mut branch_ix = [[0u16; BRANCHES]; GOLAY_STATES];
        for s8 in 0..GOLAY_STATES {
            let mut m: [u8; BRANCHES] = core::array::from_fn(|b| branches[s8][b].0);
            m.sort_unstable();
            let mi = match msets.iter().position(|x| *x == m) {
                Some(i) => i,
                None => {
                    msets.push(m);
                    msets.len() - 1
                }
            };
            for b2 in 0..BRANCHES {
                let k = m.iter().position(|&x| x == branches[s8][b2].0).expect("a branch byte is in its set");
                branch_ix[s8][b2] = (mi * BRANCHES + k) as u16;
            }
        }
        assert_eq!(msets.len(), MSETS, "the trellis has {} middle-byte sets", msets.len());
        let msets: [[u8; BRANCHES]; MSETS] = msets.try_into().expect("eight sets");

        let ranks: [[u8; 2 * K_REACH as usize + 1]; 4] =
            core::array::from_fn(|o| core::array::from_fn(|i| rank_of(o as u32, o as i32 + 4 * (i as i32 - K_REACH)).map_or(RANK_NONE, |r| r as u8)));

        let rows = tetra.rows();
        let mut index: Vec<(u32, u16)> = rows.iter().enumerate().map(|(i, &w)| (w, i as u16)).collect();
        index.sort_unstable();

        // The fallback of an end region: over its two patterns and the 2,048
        // rows of its class, the point of least norm, ties to the smaller
        // point — the bench's `min_by_key(|y| (norm, y))`.
        let mut fallback = Vec::with_capacity(2 * 2 * 2 * GOLAY_STATES);
        for kind in 0..2usize {
            for p in 0..2u32 {
                for r in 0..2usize {
                    for state in 0..GOLAY_STATES {
                        let patterns = if kind == 0 { prefixes[state] } else { suffixes[state] };
                        let mut best: Option<(i64, [i32; SECTION], u8, u32)> = None;
                        for &c in &patterns {
                            for &row in &rows[CLASS_ROWS * r..CLASS_ROWS * (r + 1)] {
                                let y = section_point(p, c, row);
                                let n: i64 = y.iter().map(|&v| (v as i64) * (v as i64)).sum();
                                if best.is_none_or(|(bn, by, _, _)| (n, y) < (bn, by)) {
                                    best = Some((n, y, c, row));
                                }
                            }
                        }
                        let (_, y, byte, key) = best.expect("a region has 4,096 members");
                        fallback.push(Fallback { byte, key, y });
                    }
                }
            }
        }

        Self {
            order: *tetra.order(),
            prefixes,
            suffixes,
            branches,
            msets,
            branch_ix,
            class: [Closed::of(&CLASS_BOUNDS[0]), Closed::of(&CLASS_BOUNDS[1])],
            mixed: Closed::of(&MIXED_BOUND),
            ranks,
            fallback,
            index,
            rule,
        }
    }

    /// The rule in force.
    pub fn rule(&self) -> Rule {
        self.rule
    }

    #[inline]
    fn fb(&self, kind: usize, p: u32, r: usize, state: usize) -> &Fallback {
        &self.fallback[((kind * 2 + p as usize) * 2 + r) * GOLAY_STATES + state]
    }

    /// Position of a packed row in the table. Every key the solver accepts
    /// is a row: the closed form is the table (`rank.rs`, pinned).
    fn position(&self, key: u32) -> usize {
        let i = self.index.binary_search_by_key(&key, |&(w, _)| w).expect("a member key is a row");
        self.index[i].1 as usize
    }

    /// Both parities' shrink-1.0 bases of one pattern from a single gather:
    /// `(distance, member, key)` for k-parity 0 and 1. The unrepaired
    /// rounding serves one parity; the other is the same with the least
    /// decided coordinate stepped toward the target, a three-term delta.
    #[inline(never)]
    fn base_pair(tab: &Tables, pattern: u8, bounds: [&Closed; 2]) -> [(f64, bool, u32); 2] {
        let layer = &tab.t[0];
        let e: [&Coord; SECTION] = core::array::from_fn(|j| &layer[j][(pattern >> j & 1) as usize]);
        let mut j0 = 0usize;
        for (j, c) in e.iter().enumerate().skip(1) {
            if c.regret > e[j0].regret {
                j0 = j;
            }
        }
        let (mut cost, mut key, mut dist, mut ksum) = (0u32, 0u32, 0.0f64, 0i32);
        for (j, c) in e.iter().enumerate() {
            cost += c.w[2] as u32;
            key |= (c.rank[2] as u32) << (4 * j);
            dist += c.d[2];
            ksum += c.k0;
        }
        let par0 = ksum.rem_euclid(2) as usize;
        let f0 = (2 + e[j0].dir) as usize;
        let cost1 = cost - e[j0].w[2] as u32 + e[j0].w[f0] as u32;
        let key1 = (key & !(0xfu32 << (4 * j0))) | ((e[j0].rank[f0] as u32) << (4 * j0));
        let dist1 = dist - e[j0].d[2] + e[j0].d[f0];
        let mut out = [(0.0f64, false, 0u32); 2];
        out[par0] = (dist, bounds[par0].holds(cost, key), key);
        out[par0 ^ 1] = (dist1, bounds[par0 ^ 1].holds(cost1, key1), key1);
        out
    }

    /// The bench's candidate rule for one pattern and one k-parity, without
    /// the fallback, over the first `n_shrinks` layers: the nearest member
    /// `(distance, packed row)`, if any candidate is one.
    #[inline(never)]
    #[allow(clippy::needless_range_loop)] // the index loops are the rule as the bench states it
    fn solve(tab: &Tables, pattern: u8, want: u32, bound: &Closed, n_shrinks: usize) -> Option<(f64, u32)> {
        let mut best: Option<(f64, u32)> = None;
        let mut best_d = f64::INFINITY;
        let cbits: [usize; SECTION] = core::array::from_fn(|j| (pattern >> j & 1) as usize);
        for (si, layer) in tab.t.iter().enumerate().take(n_shrinks) {
            let e: [&Coord; SECTION] = core::array::from_fn(|j| &layer[j][cbits[j]]);
            // Top-two regrets, ties to the lower index (the bench's stable sort).
            let (mut j0, mut j1) = (0usize, 1usize);
            if e[1].regret > e[0].regret {
                (j0, j1) = (1, 0);
            }
            for j in 2..SECTION {
                if e[j].regret > e[j0].regret {
                    j1 = j0;
                    j0 = j;
                } else if e[j].regret > e[j1].regret {
                    j1 = j;
                }
            }
            // The base: k = k0 everywhere, then the parity repair at j0.
            // `f` indexes the five-entry tables; 2 is k0.
            let mut f = [2usize; SECTION];
            let ksum: i32 = e.iter().map(|c| c.k0).sum();
            if ksum.rem_euclid(2) != want as i32 {
                f[j0] = (2 + e[j0].dir) as usize;
            }
            let (mut cost, mut key, mut dist) = (0u32, 0u32, 0.0f64);
            for j in 0..SECTION {
                cost += e[j].w[f[j]] as u32;
                key |= (e[j].rank[f[j]] as u32) << (4 * j);
                dist += e[j].d[f[j]];
            }
            if bound.holds(cost, key) && dist < best_d {
                best_d = dist;
                best = Some((dist, key));
                if si == 0 {
                    // The parity coset's optimum is a member: nothing else
                    // can be nearer, and the bench keeps it, replacing on
                    // strict `<` only.
                    break;
                }
            }
            // The sixteen two-coordinate re-roundings: ±1 at j, and at j2 a
            // step toward the target from the FIXED k — back to k0 when j2
            // was repaired, toward z otherwise. Both steps together keep
            // the parity.
            for j in 0..SECTION {
                let j2 = if j == j0 { j1 } else { j0 };
                let s2: i32 = if f[j2] == 2 { e[j2].dir } else { 2 - f[j2] as i32 };
                let f2 = (f[j2] as i32 + s2) as usize;
                for step in [-1i32, 1] {
                    let fj = (f[j] as i32 + step) as usize;
                    let d2 = dist - e[j].d[f[j]] - e[j2].d[f[j2]] + e[j].d[fj] + e[j2].d[f2];
                    if d2 >= best_d {
                        continue;
                    }
                    let c2 = cost - e[j].w[f[j]] as u32 - e[j2].w[f[j2]] as u32 + e[j].w[fj] as u32 + e[j2].w[f2] as u32;
                    let mask = !((0xfu32 << (4 * j)) | (0xfu32 << (4 * j2)));
                    let k2 = (key & mask) | ((e[j].rank[fj] as u32) << (4 * j)) | ((e[j2].rank[f2] as u32) << (4 * j2));
                    if bound.holds(c2, k2) {
                        best_d = d2;
                        best = Some((d2, k2));
                    }
                }
            }
        }
        best
    }

    /// One end state resolved in full: the fallback, then the full rule on
    /// both patterns, replacing on strict `<` as the bench's `nearest` does.
    fn end_resolve(&self, tab: &Tables, target: &[f64; SECTION], kind: usize, p: u32, r: usize, state: usize) -> Entry {
        let patterns = if kind == 0 { self.prefixes[state] } else { self.suffixes[state] };
        let fb = self.fb(kind, p, r, state);
        let mut e = Entry { value: dist_to(&fb.y, target), exact: true, byte: fb.byte, key: fb.key };
        for &c in &patterns {
            if let Some((d, key)) = Self::solve(tab, c, r as u32, &self.class[r], self.rule.end) {
                if d < e.value {
                    e = Entry { value: d, exact: true, byte: c, key };
                }
            }
        }
        e
    }

    /// One middle `(byte, δ)` resolved in full. Under a truncated rule a
    /// pattern can be left without a member near the target; the full rule
    /// then finds one, its zero shrink never failing.
    fn mid_resolve(&self, tab: &Tables, byte: u8, delta: u32) -> Entry {
        let mut pick = Self::solve(tab, byte, delta, &self.mixed, self.rule.mid);
        if pick.is_none() && self.rule.mid < MID_SHRINKS {
            pick = Self::solve(tab, byte, delta, &self.mixed, MID_SHRINKS);
        }
        match pick {
            Some((d, key)) => Entry { value: d, exact: true, byte, key },
            None => INFEASIBLE,
        }
    }

    /// The best path by the entries' current values.
    fn join(&self, p: u32, sc: &Scratch) -> Path {
        let mut bp = Path { cost: f64::INFINITY, i1: 0, i2: 0, delta: 0, i3: 0 };
        for r in 0..2usize {
            for s8 in 0..GOLAY_STATES {
                let i1 = r * GOLAY_STATES + s8;
                let a = sc.e1[i1].value;
                if a >= bp.cost {
                    continue;
                }
                for (b2, &(_, s16)) in self.branches[s8].iter().enumerate() {
                    let i2 = self.branch_ix[s8][b2] as usize;
                    for delta in 0..2usize {
                        let r_out = (p as usize ^ r ^ delta) & 1;
                        let i3 = r_out * GOLAY_STATES + s16 as usize;
                        let c = a + sc.e2[i2][delta].value + sc.e3[i3].value;
                        if c < bp.cost {
                            bp = Path { cost: c, i1, i2, delta, i3 };
                        }
                    }
                }
            }
        }
        bp
    }

    /// The rule's answer at one block parity, lazily: bases, seed joins,
    /// the pruning pass, the survivors, the final join. `u0` is the best cost
    /// already found at the other parity: a path that cannot beat it cannot
    /// win either, so it seeds `U`.
    fn solve_parity(&self, p: u32, targets: &[[f64; SECTION]; 3], sc: &mut Scratch, u0: f64) -> Option<Winner> {
        sc.count_pass();
        for (tab, t) in sc.tabs.iter_mut().zip(targets) {
            tab.build(t, p, &self.ranks);
        }
        // End entries: the fallback, then the bases of both patterns — a
        // member replaces on strict `<`, a non-member lowers the bound.
        for kind in 0..2usize {
            for st in 0..GOLAY_STATES {
                let patterns = if kind == 0 { self.prefixes[st] } else { self.suffixes[st] };
                let tab = &sc.tabs[2 * kind];
                let target = &targets[2 * kind];
                let mut ent: [Entry; 2] = core::array::from_fn(|r| {
                    let fb = self.fb(kind, p, r, st);
                    Entry { value: dist_to(&fb.y, target), exact: true, byte: fb.byte, key: fb.key }
                });
                let mut lb = [f64::INFINITY; 2];
                let mut unresolved = [false; 2];
                for &c in &patterns {
                    let pair = Self::base_pair(tab, c, [&self.class[0], &self.class[1]]);
                    for r in 0..2usize {
                        let (d, m, key) = pair[r];
                        if m {
                            if d < ent[r].value {
                                ent[r] = Entry { value: d, exact: true, byte: c, key };
                            }
                        } else {
                            lb[r] = lb[r].min(d);
                            unresolved[r] = true;
                        }
                    }
                }
                let out = if kind == 0 { &mut sc.e1 } else { &mut sc.e3 };
                for r in 0..2usize {
                    out[r * GOLAY_STATES + st] = if unresolved[r] && lb[r] < ent[r].value { Entry { value: lb[r], exact: false, ..ent[r] } } else { ent[r] };
                }
            }
        }
        // Middle entries: the base of each byte for both parities.
        for (m, set) in self.msets.iter().enumerate() {
            for (k, &b) in set.iter().enumerate() {
                let pair = Self::base_pair(&sc.tabs[1], b, [&self.mixed; 2]);
                sc.e2[m * BRANCHES + k] = core::array::from_fn(|d| {
                    let (dist, mem, key) = pair[d];
                    Entry { value: dist, exact: mem, byte: b, key }
                });
            }
        }

        // 1. Seed joins: resolve the sections of the best bound path until
        //    it is all exact, at most WARM_JOINS times.
        let mut u = u0;
        for _ in 0..WARM_JOINS {
            let path = self.join(p, sc);
            let mut all = true;
            if !sc.e1[path.i1].exact {
                sc.e1[path.i1] = self.end_resolve(&sc.tabs[0], &targets[0], 0, p, path.i1 / GOLAY_STATES, path.i1 % GOLAY_STATES);
                sc.count_end();
                all = false;
            }
            if !sc.e2[path.i2][path.delta].exact {
                sc.e2[path.i2][path.delta] = self.mid_resolve(&sc.tabs[1], self.msets[path.i2 / BRANCHES][path.i2 % BRANCHES], path.delta as u32);
                sc.count_mid();
                all = false;
            }
            if !sc.e3[path.i3].exact {
                sc.e3[path.i3] = self.end_resolve(&sc.tabs[2], &targets[2], 1, p, path.i3 / GOLAY_STATES, path.i3 % GOLAY_STATES);
                sc.count_end();
                all = false;
            }
            if all {
                u = u.min(path.cost);
                break;
            }
        }
        // 2. One pass over the 8,192 paths: the best path through every
        //    entry by current values, and the best all-exact path U.
        sc.b1.fill(f64::INFINITY);
        sc.b2.fill([f64::INFINITY; 2]);
        sc.b3.fill(f64::INFINITY);
        for r in 0..2usize {
            for s8 in 0..GOLAY_STATES {
                let i1 = r * GOLAY_STATES + s8;
                let a = sc.e1[i1];
                for (b2, &(_, s16)) in self.branches[s8].iter().enumerate() {
                    let i2 = self.branch_ix[s8][b2] as usize;
                    for delta in 0..2usize {
                        let r_out = (p as usize ^ r ^ delta) & 1;
                        let i3 = r_out * GOLAY_STATES + s16 as usize;
                        let (v2, v3) = (sc.e2[i2][delta], sc.e3[i3]);
                        let c = a.value + v2.value + v3.value;
                        sc.b1[i1] = sc.b1[i1].min(c);
                        sc.b2[i2][delta] = sc.b2[i2][delta].min(c);
                        sc.b3[i3] = sc.b3[i3].min(c);
                        if a.exact && v2.exact && v3.exact && c < u {
                            u = c;
                        }
                    }
                }
            }
        }
        // 3. Resolve every unresolved entry whose best path is under U.
        for i1 in 0..END_ENTRIES {
            if !sc.e1[i1].exact && sc.b1[i1] < u {
                sc.e1[i1] = self.end_resolve(&sc.tabs[0], &targets[0], 0, p, i1 / GOLAY_STATES, i1 % GOLAY_STATES);
                sc.count_end();
            }
        }
        for i2 in 0..MID_ENTRIES {
            for d in 0..2usize {
                if !sc.e2[i2][d].exact && sc.b2[i2][d] < u {
                    sc.e2[i2][d] = self.mid_resolve(&sc.tabs[1], self.msets[i2 / BRANCHES][i2 % BRANCHES], d as u32);
                    sc.count_mid();
                }
            }
        }
        for i3 in 0..END_ENTRIES {
            if !sc.e3[i3].exact && sc.b3[i3] < u {
                sc.e3[i3] = self.end_resolve(&sc.tabs[2], &targets[2], 1, p, i3 / GOLAY_STATES, i3 % GOLAY_STATES);
                sc.count_end();
            }
        }
        // 4. What is still unresolved cannot beat U: invisible to the join.
        for e in sc.e1.iter_mut().chain(sc.e3.iter_mut()).chain(sc.e2.iter_mut().flatten()) {
            if !e.exact {
                *e = INFEASIBLE;
            }
        }
        let path = self.join(p, sc);
        if !path.cost.is_finite() {
            return None;
        }
        let (r, s8) = (path.i1 / GOLAY_STATES, path.i1 % GOLAY_STATES);
        let mid = sc.e2[path.i2][path.delta];
        let b2 = self.branches[s8].iter().position(|&(b, _)| b == mid.byte).expect("the middle byte is a branch of s8");
        let s16 = self.branches[s8][b2].1 as usize;
        let (e1, e3) = (sc.e1[path.i1], sc.e3[path.i3]);
        Some(Winner {
            cost: path.cost,
            p,
            r: r as u32,
            s8,
            s16,
            delta: path.delta as u32,
            sections: [(e1.byte, e1.key), (mid.byte, mid.key), (e3.byte, e3.key)],
        })
    }

    /// The word and the point of a winner: the point by the decoder's own
    /// formula from `(p, byte, row)`, the word from the rows' positions.
    fn code_of(&self, x: &[f64; DIM], w: &Winner) -> TetraCode {
        let [(c1, k1), (c2, k2), (c3, k3)] = w.sections;
        let b1 = self.prefixes[w.s8].iter().position(|&b| b == c1).expect("a prefix byte of s8");
        let b2 = self.branches[w.s8].iter().position(|&(b, _)| b == c2).expect("a branch byte of s8");
        let b3 = self.suffixes[w.s16].iter().position(|&b| b == c3).expect("a suffix byte of s16");
        let pos1 = self.position(k1);
        let pos2 = self.position(k2);
        let pos3 = self.position(k3);
        let r3 = (w.p ^ w.r ^ w.delta) & 1;
        debug_assert_eq!(pos1 / CLASS_ROWS, w.r as usize, "section 1's row is not of class r");
        debug_assert_eq!(pos2 / CLASS_ROWS, w.delta as usize, "section 2's row is not of class δ");
        debug_assert_eq!(pos3 / CLASS_ROWS, r3 as usize, "section 3's row is not of class p ⊕ r ⊕ δ");
        debug_assert_eq!(rank_class(&unpack(k2)), w.delta, "δ is not the middle row's class");
        let i2 = pos2 % CLASS_ROWS + w.delta as usize * N0_MIXED;
        debug_assert!(pos2 % CLASS_ROWS < if w.delta == 0 { N0_MIXED } else { CLASS_ROWS - N0_MIXED }, "the middle row is not in the mixed set");
        let word = Fields {
            p: w.p as u8,
            r: w.r as u8,
            s8: w.s8 as u8,
            b1: b1 as u8,
            i1: (pos1 % CLASS_ROWS) as u16,
            b2: b2 as u8,
            i2: i2 as u16,
            b3: b3 as u8,
            i3: (pos3 % CLASS_ROWS) as u16,
            gain: 0,
        }
        .join();

        let mut point = [0i32; DIM];
        for (k, &(c, key)) in w.sections.iter().enumerate() {
            let y = section_point(w.p, c, key);
            for (j, &v) in y.iter().enumerate() {
                point[self.order[SECTION * k + j] as usize] = v;
            }
        }
        TetraCode { word, point, t: t_of(x, &point) }
    }

    /// The rule's answer at scale `s`: the nearest Tetra point to `x/s` over
    /// both block parities, its word and its `t`. For tests and for the
    /// scale study; `encode` is the production call.
    pub fn encode_at_scale(&self, x: &[f64; DIM], s: f64, scratch: &mut Scratch) -> TetraCode {
        assert!(s > 0.0 && s.is_finite(), "scale {s} is not a positive finite number");
        let targets: [[f64; SECTION]; 3] = core::array::from_fn(|k| core::array::from_fn(|j| x[self.order[SECTION * k + j] as usize] / s));
        let mut best: Option<Winner> = None;
        for p in 0..2u32 {
            let u0 = best.map_or(f64::INFINITY, |b| b.cost);
            if let Some(w) = self.solve_parity(p, &targets, scratch, u0) {
                if best.is_none_or(|b| w.cost < b.cost) {
                    best = Some(w);
                }
            }
        }
        // Every end entry has a fallback and every middle byte a member, so
        // a path exists at both parities.
        let w = best.expect("a feasible path");
        self.code_of(x, &w)
    }

    /// The production call: two adaptive scales, `ALPHA·‖x‖/√24` and `RATIO`
    /// times it, the larger `t` wins and the lower scale keeps ties. A zero
    /// block is the origin, word 0; no other block is. The norm is taken
    /// relative to the largest coordinate so that it neither underflows nor
    /// overflows, and a block so small that its scale would not be a normal
    /// number is encoded as its unit-max multiple — the code is a function
    /// of the direction, `t` scales back.
    pub fn encode(&self, x: &[f64; DIM], scratch: &mut Scratch) -> TetraCode {
        let (m, s0) = Self::lower_scale(x);
        if m == 0.0 {
            return TetraCode { word: 0, point: [0; DIM], t: f64::NEG_INFINITY };
        }
        if s0.is_normal() {
            return self.two_scales(x, s0, scratch);
        }
        let xs: [f64; DIM] = core::array::from_fn(|j| x[j] / m);
        let mut code = self.two_scales(&xs, Self::lower_scale(&xs).1, scratch);
        code.t *= m;
        code
    }

    /// `(max_j |x_j|, ALPHA·‖x‖/√24)`: the block's largest coordinate and the
    /// lower scale `encode` runs, computed as it computes it.
    pub fn lower_scale(x: &[f64; DIM]) -> (f64, f64) {
        let m = x.iter().fold(0.0f64, |a, &v| a.max(v.abs()));
        if m == 0.0 {
            return (0.0, 0.0);
        }
        let unit_norm = x.iter().map(|&v| (v / m) * (v / m)).sum::<f64>().sqrt();
        (m, Self::ALPHA * m * unit_norm / (DIM as f64).sqrt())
    }

    fn two_scales(&self, x: &[f64; DIM], s0: f64, scratch: &mut Scratch) -> TetraCode {
        let a = self.encode_at_scale(x, s0, scratch);
        let b = self.encode_at_scale(x, s0 * Self::RATIO, scratch);
        if b.t > a.t {
            b
        } else {
            a
        }
    }
}

/// One section's point from its parity, pattern byte and packed row — the
/// decoder's formula.
fn section_point(p: u32, c: u8, row: u32) -> [i32; SECTION] {
    core::array::from_fn(|j| val(p + 2 * ((c >> j) & 1) as u32, (row >> (4 * j)) & 15))
}

#[inline]
fn dist_to(y: &[i32; SECTION], target: &[f64; SECTION]) -> f64 {
    y.iter().zip(target).map(|(&v, &x)| (v as f64 - x).powi(2)).sum()
}

/// `⟨x, y⟩/‖y‖`, `−∞` on the origin: the bench's `t_of`.
pub fn t_of(x: &[f64; DIM], y: &[i32; DIM]) -> f64 {
    let dot: f64 = x.iter().zip(y).map(|(&a, &b)| a * b as f64).sum();
    let nn: f64 = y.iter().map(|&b| (b as f64) * (b as f64)).sum();
    if nn > 0.0 {
        dot / nn.sqrt()
    } else {
        f64::NEG_INFINITY
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::cost;
    use llvq_core::SplitMix64;

    /// Every `ρ ∈ {0..4}⁸`, flat.
    fn all_to_four() -> impl Iterator<Item = [u32; SECTION]> {
        (0..5u32.pow(8)).map(|code| {
            let mut c = code;
            core::array::from_fn(|_| {
                let r = c % 5;
                c /= 5;
                r
            })
        })
    }

    /// The closed form on packed keys is `Bound::contains` on the ranks, for
    /// the three bounds and all 390,625 vectors up to rank 4 — the nibble
    /// reversal is what makes the integer comparison lexicographic.
    #[test]
    fn the_packed_closed_form_is_the_bound() {
        let bounds = [(Closed::of(&CLASS_BOUNDS[0]), CLASS_BOUNDS[0]), (Closed::of(&CLASS_BOUNDS[1]), CLASS_BOUNDS[1]), (Closed::of(&MIXED_BOUND), MIXED_BOUND)];
        let mut on = 0usize;
        for rho in all_to_four() {
            let (c, key) = (cost(&rho), pack(&rho));
            for (closed, bound) in &bounds {
                assert_eq!(closed.holds(c, key), bound.contains(&rho), "{rho:?}");
                on += usize::from(c == bound.cost);
            }
        }
        assert!(on > 1_000, "the boundary shells were visited {on} times");
        // Reversal is its own inverse and moves ρ_0 to the top nibble.
        assert_eq!(lex(lex(0x1234_5678)), 0x1234_5678);
        assert_eq!(lex(pack(&[1, 0, 0, 0, 0, 0, 0, 0])), 0x1000_0000);
        assert_eq!(lex(pack(&[0, 0, 0, 0, 0, 0, 0, 1])), 0x0000_0001);
    }

    /// The rank table of the coordinate builder inverts `val` on the whole
    /// reach and refuses beyond rank 7 — so a Coord's `(rank, value)` pairs
    /// are the decoder's.
    #[test]
    fn the_rank_table_inverts_val() {
        let e = Encoder::new(&Tetra::new());
        for o in 0..4u32 {
            for k in -K_REACH..=K_REACH {
                let r = e.ranks[o as usize][(k + K_REACH) as usize];
                match rank_of(o, o as i32 + 4 * k) {
                    Some(want) => {
                        assert_eq!(r as u32, want);
                        assert_eq!(val(o, r as u32), o as i32 + 4 * k);
                    }
                    None => assert_eq!(r, RANK_NONE, "o={o} k={k}"),
                }
            }
            assert!(e.ranks[o as usize].iter().filter(|&&r| r != RANK_NONE).count() == 8, "o={o}: eight ranks");
        }
    }

    /// The fallbacks are members of their region at the least norm: the bench's
    /// rule, checked by enumerating the members of a few regions in full.
    #[test]
    fn the_fallbacks_are_the_least_norm_members() {
        let tetra = Tetra::new();
        let e = Encoder::new(&tetra);
        let rows = tetra.rows();
        for (kind, p, r, state) in [(0usize, 0u32, 0usize, 0usize), (0, 1, 1, 17), (1, 0, 1, 63), (1, 1, 0, 5)] {
            let fb = e.fb(kind, p, r, state);
            let patterns = if kind == 0 { e.prefixes[state] } else { e.suffixes[state] };
            assert!(patterns.contains(&fb.byte));
            assert_eq!(e.position(fb.key) / CLASS_ROWS, r);
            assert_eq!(section_point(p, fb.byte, fb.key), fb.y);
            let n = |y: &[i32; SECTION]| y.iter().map(|&v| (v as i64) * (v as i64)).sum::<i64>();
            for &c in &patterns {
                for &row in &rows[CLASS_ROWS * r..CLASS_ROWS * (r + 1)] {
                    let y = section_point(p, c, row);
                    assert!((n(&y), y) >= (n(&fb.y), fb.y), "kind {kind} p={p} r={r} state {state}: {y:?} is below the fallback");
                }
            }
        }
        // The origin's own region: the zero section.
        assert_eq!(e.fb(0, 0, 0, 0).y, [0; SECTION]);
    }

    /// The pruning is the whole reason the join is cheap, and it is pinned
    /// here as a **rate**, not as a duration: over 200 Gaussian blocks the
    /// lazy pass takes 13.59 % of the entry slots from a lower bound to an
    /// exact value, the rest never being solved at all.
    ///
    /// Both directions are held. A bound that prunes nothing — the `< u`
    /// tests replaced by `true`, or `b1`/`b2`/`b3` filled with `0.0`
    /// instead of `INFINITY` — pays for every entry a base left
    /// unresolved and blows the ceiling; a bound that prunes everything —
    /// the tests replaced by `false` — never solves anything and falls
    /// under the floor. Without the floor the second family would only be
    /// visible in `llvq-bench/tests/tetra_encoder.rs`, which is release-only
    /// and needs the bench in the same process.
    ///
    /// Measured on 2026-09-05, and deterministic — the same figure in debug
    /// and in release: 13.59 % of the 409,600 slots, 6.37 % end and 7.22 %
    /// middle. The mutants give 45.34 % (`b1`/`b2`/`b3` filled with `0.0`)
    /// and 0.42 % (`< u` replaced by `< 0.0`, only the seed joins left). So
    /// the ceiling of 20 % sits 1.47× above the recorded rate and 2.27×
    /// below the first mutant, and the floor of 5 % 2.7× below the recorded
    /// rate and 11.9× above the second.
    #[test]
    fn the_join_solves_a_small_fraction_of_the_entries() {
        const BLOCKS: usize = 200;
        let enc = Encoder::new(&Tetra::new());
        let mut sc = Scratch::new();
        let mut rng = SplitMix64::new(0x0F1B_5017);
        for _ in 0..BLOCKS {
            let x: [f64; DIM] = core::array::from_fn(|_| rng.next_gaussian());
            let code = enc.encode(&x, &mut sc);
            assert!(code.t.is_finite(), "a Gaussian block is not the origin");
        }
        let s = sc.take_solved();
        // Four parity passes a block: two scales, two block parities.
        assert_eq!(s.passes, 4 * BLOCKS as u64, "encode runs two scales over two parities");
        let slots = s.passes * SLOTS_PER_PASS;
        let pct = |n: u64| 100.0 * n as f64 / slots as f64;
        let (end, mid, all) = (pct(s.end), pct(s.mid), pct(s.end + s.mid));
        println!("{BLOCKS} blocks, {slots} entry slots: {all:.2} % solved in full ({end:.2} % end, {mid:.2} % middle)");
        assert!(all < 20.0, "the pruning has stopped pruning: {all:.2} % of the slots solved in full, against 13.59 % recorded");
        assert!(all > 5.0, "only {all:.2} % of the slots solved: the bound test is pruning entries the rule needs");
    }
}
