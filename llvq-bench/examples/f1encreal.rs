//! The F1 encoder prototype on REAL blocks — the compensated, rotated GPTQ
//! residues of Qwen3-0.6B dumped by `llvq-llm/examples/f1recdump.rs` — timed
//! against a Gaussian control of the same size in the same process, one core.
//!
//! `cargo run --release -p llvq-bench --example f1encreal -- <blocks.f64> [rowscale.f64|-] [meta.csv|-] [n]`
//!
//! ## The question
//!
//! `f1enclazy.rs` measured 290-296 µs/block/core at two adaptive scales on
//! N(0,1) blocks, pruning all but ~188 of the 1,536 section solves per scale.
//! That pruning rests on the shrink-1.0 base being a member of its region
//! (58 % of entries at the adaptive scale on a Gaussian). Whether it holds on
//! the blocks the encoder sees in production — GPTQ residues in a rotated
//! basis, heavier-tailed than a Gaussian — was never measured, and the
//! exhaustive two-scale variant (962 µs) sits above F1c's 656 µs gate.
//!
//! ## What is copied, and what is added
//!
//! The encoder (`Fast`, its solver, its tables) is copied from
//! `f1enclazy.rs` (2026-09-05) and not modified except for two counters in
//! [`Stats`]: `base_entries` and `base_members`, incremented where the lazy
//! pass reads a base's membership. Nothing here touches `f1.rs`, a format or
//! a served path.
//!
//! Blocks are divided by the row scale the served loop announced for them
//! (`rowscale.f64`), which is how the served gain code sees them; the
//! direction code is scale-equivariant, so timing is unaffected. Retention
//! is `retention_pct(mse / P, 2.0)` with `P` the mean squared weight of the
//! blocks: on an N(0,1) source `P = 1` and the number is exactly f1bench's.

// A prototype: the index loops mirror the bench's code they are checked
// against, and the solver's signature carries its counters.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

use llvq_bench::f1::{point_to_natural, SectionSet, Trellis, BRANCHES, GOLAY_STATES, SECTION};
use llvq_bench::{gauss_block, lloyd_max, nearest_centroid, precompute13, retention_pct};
use llvq_core::leech::DIM;
use llvq_core::SplitMix64;
use llvq_search::Searcher;
use std::time::Instant;

// ---------------------------------------------------------------------------
// The closed-form solver — copied from f1enclazy.rs (2026-09-05)
// ---------------------------------------------------------------------------

/// The region in closed form: `cost < c` or `cost == c && key <= cut`, where
/// `key` packs `ρ_0` in the top nibble so the integer order is the
/// lexicographic order of the eight ranks.
#[derive(Clone, Copy)]
struct Bound {
    c: u32,
    cut: u32,
}

const BOUND_CLASS: [Bound; 2] = [Bound { c: 88, cut: 0x1103_0011 }, Bound { c: 96, cut: 0x0022_2011 }];
const BOUND_MIXED: Bound = Bound { c: 72, cut: 0x3000_1010 };

#[inline(always)]
fn member(b: Bound, cost: u32, key: u32) -> bool {
    cost < b.c || (cost == b.c && key <= b.cut)
}

/// Rank of `o + 4k` in the progression `o + 4Z` listed outward from zero —
/// `val` inverted in closed form.
#[inline(always)]
fn rank_from_k(o: u32, k: i32) -> u32 {
    match o {
        0 => {
            if k == 0 {
                0
            } else if k > 0 {
                (2 * k - 1) as u32
            } else {
                (-2 * k) as u32
            }
        }
        2 => {
            if k >= 0 {
                (2 * k) as u32
            } else {
                (-2 * k - 1) as u32
            }
        }
        1 => {
            if k >= 0 {
                (2 * k) as u32
            } else {
                (-2 * k - 1) as u32
            }
        }
        _ => {
            if k >= 0 {
                (2 * k + 1) as u32
            } else {
                (-2 * k - 2) as u32
            }
        }
    }
}

/// One coordinate under one offset, at one shrink: the rounding of the shrunk
/// target, and the five values `k0 − 2 ..= k0 + 2` with their rank, weight,
/// and squared distance to the ORIGINAL target coordinate.
#[derive(Clone, Copy, Default)]
struct Coord {
    k0: i32,
    dir: i32,
    regret: f64,
    /// Rank capped at 15 so the nibble never overflows; anything ≥ 5 is
    /// outside every region and its weight (≥ 121) says so.
    rank: [u8; 5],
    w: [u16; 5],
    d: [f64; 5],
}

/// Per (section target, parity p): for every shrink, coordinate and offset bit
/// `c_j`, a [`Coord`].
struct Tables {
    p: u32,
    t: [[[Coord; 2]; SECTION]; 8],
}

const SHRINKS: [f64; 8] = [1.0, 0.85, 0.7, 0.55, 0.4, 0.25, 0.1, 0.0];

impl Tables {
    fn build(target: &[f64; SECTION], p: u32) -> Self {
        let mut t = [[[Coord::default(); 2]; SECTION]; 8];
        for (si, &shrink) in SHRINKS.iter().enumerate() {
            for j in 0..SECTION {
                for cbit in 0..2u32 {
                    let o = p + 2 * cbit;
                    let u = target[j] * shrink;
                    let z = (u - o as f64) / 4.0;
                    let k0 = z.round() as i32;
                    let mut e = Coord { k0, dir: if z > k0 as f64 { 1 } else { -1 }, regret: (z - k0 as f64).abs(), ..Default::default() };
                    for (m, slot) in (-2i32..=2).enumerate() {
                        let k = k0 + slot;
                        let v = o as i32 + 4 * k;
                        let r = rank_from_k(o, k).min(15);
                        e.rank[m] = r as u8;
                        e.w[m] = ((2 * r + 1) * (2 * r + 1)).min(u16::MAX as u32) as u16;
                        e.d[m] = (v as f64 - target[j]).powi(2);
                    }
                    t[si][j][cbit as usize] = e;
                }
            }
        }
        Self { p, t }
    }
}

/// Both parities' shrink-1.0 bases of one pattern from a single gather:
/// `(distance, member, point)` for k-parity 0 and 1. The unrepaired rounding
/// serves one parity; the other is the same with the least decided
/// coordinate stepped toward the target, a three-term delta.
#[inline(never)]
fn base_pair(tab: &Tables, pattern: u8, bounds: [Bound; 2]) -> [(f64, bool, [i32; SECTION]); 2] {
    let layer = &tab.t[0];
    let cbits: [usize; SECTION] = core::array::from_fn(|j| (pattern >> j & 1) as usize);
    let e: [&Coord; SECTION] = core::array::from_fn(|j| &layer[j][cbits[j]]);
    let mut j0 = 0usize;
    for j in 1..SECTION {
        if e[j].regret > e[j0].regret {
            j0 = j;
        }
    }
    let mut cost = 0u32;
    let mut key = 0u32;
    let mut dist = 0.0f64;
    let mut ksum = 0i32;
    for j in 0..SECTION {
        cost += e[j].w[2] as u32;
        key |= (e[j].rank[2] as u32) << (4 * (7 - j));
        dist += e[j].d[2];
        ksum += e[j].k0;
    }
    let par0 = ksum.rem_euclid(2) as usize; // parity of the unrepaired base
    let f0 = (2 + e[j0].dir) as usize;
    let cost1 = cost - e[j0].w[2] as u32 + e[j0].w[f0] as u32;
    let key1 = (key & !(0xfu32 << (4 * (7 - j0)))) | ((e[j0].rank[f0] as u32) << (4 * (7 - j0)));
    let dist1 = dist - e[j0].d[2] + e[j0].d[f0];
    let point = |fix: Option<usize>| -> [i32; SECTION] {
        core::array::from_fn(|j| {
            let o = tab.p as i32 + 2 * cbits[j] as i32;
            let slot = if Some(j) == fix { f0 as i32 } else { 2 };
            o + 4 * (e[j].k0 + slot - 2)
        })
    };
    let mut out = [(0.0f64, false, [0i32; SECTION]); 2];
    let m0 = member(bounds[par0], cost, key);
    out[par0] = (dist, m0, if m0 { point(None) } else { [0; SECTION] });
    let m1 = member(bounds[par0 ^ 1], cost1, key1);
    out[par0 ^ 1] = (dist1, m1, if m1 { point(Some(j0)) } else { [0; SECTION] });
    out
}

/// The solver's answer: the point, its distance, and where it was found.
#[derive(Clone, Copy, Debug)]
struct Pick {
    y: [i32; SECTION],
    dist: f64,
}

struct Solver {
    /// How many shrinks to run; 8 is the bench's rule.
    n_shrinks: usize,
}

impl Solver {
    /// The bench's `nearest_constrained` for one pattern and one k-parity,
    /// without the fallback. `bound` is the region; `base_only` stops after
    /// the shrink-1.0 base candidate (member or not).
    #[inline(never)]
    fn solve(&self, tab: &Tables, pattern: u8, want: u32, bound: Bound, base_only: bool, base_closed: &mut bool, base_dist: &mut f64) -> Option<Pick> {
        let mut best: Option<(f64, [u8; SECTION], [i32; SECTION])> = None;
        let mut best_d = f64::INFINITY;
        let cbits: [usize; SECTION] = core::array::from_fn(|j| (pattern >> j & 1) as usize);
        for (si, layer) in tab.t.iter().enumerate().take(self.n_shrinks) {
            // Gather the eight coordinates of this pattern.
            let e: [&Coord; SECTION] = core::array::from_fn(|j| &layer[j][cbits[j]]);
            // Top-two regrets, ties to the lower index (the bench's stable sort).
            let (mut j0, mut j1) = (0usize, 1usize);
            if e[1].regret > e[0].regret {
                j0 = 1;
                j1 = 0;
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
            let mut f = [2usize; SECTION]; // slot index into the 5-entry tables: 2 = k0
            let ksum: i32 = e.iter().map(|c| c.k0).sum();
            let fixed = ksum.rem_euclid(2) != want as i32;
            if fixed {
                f[j0] = (2 + e[j0].dir) as usize;
            }
            let mut cost = 0u32;
            let mut key = 0u32;
            let mut dist = 0.0f64;
            for j in 0..SECTION {
                cost += e[j].w[f[j]] as u32;
                key |= (e[j].rank[f[j]] as u32) << (4 * (7 - j));
                dist += e[j].d[f[j]];
            }
            if si == 0 {
                *base_dist = dist;
            }
            if member(bound, cost, key) && dist < best_d {
                best_d = dist;
                best = Some((dist, core::array::from_fn(|j| f[j] as u8), core::array::from_fn(|j| e[j].k0)));
                if si == 0 {
                    *base_closed = true;
                    // The unconstrained coset-parity optimum is a member:
                    // nothing else can be nearer. Identical to the bench,
                    // whose later candidates only replace on strict `<`.
                    break;
                }
            }
            if base_only {
                break;
            }
            // The sixteen two-coordinate re-roundings.
            for j in 0..SECTION {
                let j2 = if j == j0 { j1 } else { j0 };
                // Step at j2: toward z from the FIXED k, i.e. back to k0 when
                // j2 was repaired, toward z otherwise.
                let s2: i32 = if f[j2] == 2 { e[j2].dir } else { 2 - f[j2] as i32 };
                let f2 = (f[j2] as i32 + s2) as usize;
                for step in [-1i32, 1] {
                    let fj = (f[j] as i32 + step) as usize;
                    let d2 = dist - e[j].d[f[j]] - e[j2].d[f[j2]] + e[j].d[fj] + e[j2].d[f2];
                    if d2 >= best_d {
                        continue;
                    }
                    let c2 = cost - e[j].w[f[j]] as u32 - e[j2].w[f[j2]] as u32 + e[j].w[fj] as u32 + e[j2].w[f2] as u32;
                    let mask = !((0xfu32 << (4 * (7 - j))) | (0xfu32 << (4 * (7 - j2))));
                    let k2 = (key & mask) | ((e[j].rank[fj] as u32) << (4 * (7 - j))) | ((e[j2].rank[f2] as u32) << (4 * (7 - j2)));
                    if member(bound, c2, k2) {
                        best_d = d2;
                        let mut fs = f;
                        fs[j] = fj;
                        fs[j2] = f2;
                        best = Some((d2, core::array::from_fn(|j| fs[j] as u8), core::array::from_fn(|j| e[j].k0)));
                    }
                }
            }
        }
        best.map(|(dist, fs, k0)| {
            let y: [i32; SECTION] = core::array::from_fn(|j| {
                let o = tab.p as i32 + 2 * cbits[j] as i32;
                o + 4 * (k0[j] + fs[j] as i32 - 2)
            });
            Pick { y, dist }
        })
    }
}

// ---------------------------------------------------------------------------
// The production encoder: bounds first, full solves only where they can win
// ---------------------------------------------------------------------------

/// Class of a packed rank key: the number of nibbles in {1, 2}, mod 2.
#[inline(always)]
fn class_of_key(key: u32) -> u32 {
    let mut c = 0u32;
    for j in 0..SECTION {
        let r = (key >> (4 * (7 - j))) & 0xf;
        c ^= (r == 1 || r == 2) as u32;
    }
    c
}

/// One section entry as the join sees it: a value that is either exact or a
/// lower bound, and the point when exact.
#[derive(Clone, Copy)]
struct Entry {
    value: f64,
    exact: bool,
    y: [i32; SECTION],
}

const INFEASIBLE: Entry = Entry { value: f64::INFINITY, exact: true, y: [0; SECTION] };

fn msets_of(trellis: &Trellis) -> (Vec<Vec<u8>>, Vec<usize>) {
    let mut msets: Vec<Vec<u8>> = Vec::new();
    let mut mset_of = vec![0usize; GOLAY_STATES];
    for (s8, slot) in mset_of.iter_mut().enumerate() {
        let mut m: Vec<u8> = trellis.branches[s8].iter().map(|&(b, _)| b).collect();
        m.sort_unstable();
        *slot = match msets.iter().position(|s| *s == m) {
            Some(i) => i,
            None => {
                msets.push(m);
                msets.len() - 1
            }
        };
    }
    assert_eq!(msets.len(), 8);
    (msets, mset_of)
}

pub struct Fast {
    trellis: Trellis,
    msets: Vec<Vec<u8>>,
    /// Fallback (lowest-norm member) of each end region, `[(kind, p, r, state)]`
    /// flattened as `((kind * 2 + p) * 2 + r) * 64 + state`, kind 0 = prefix,
    /// 1 = suffix.
    fallback: Vec<[i32; SECTION]>,
    /// `branch_ix[s8][bi]`: index of the middle entry of branch `bi` of `s8`.
    branch_ix: Vec<[usize; BRANCHES]>,
    solver: Solver,
    /// Shrinks of the end sections' full rule (the bench: 7) and warm-up joins.
    shrinks_end: usize,
    warm: usize,
}

/// Counters of one `encode_at_scale` call.
///
/// `base_entries` / `base_members` are the addition to f1enclazy's copy: how
/// many section entries the lazy pass looked at, and how many of them the
/// shrink-1.0 base closed on its own — the rate the pruning rests on.
#[derive(Default, Clone, Copy)]
pub struct Stats {
    pub full_solves: usize,
    pub joins: usize,
    pub base_entries: usize,
    pub base_members: usize,
    /// Seconds in: tables, bases, warm joins + their solves, the per-entry pass, survivor solves, final join.
    pub t: [f64; 6],
}

impl Stats {
    fn add(&mut self, o: &Stats) {
        self.full_solves += o.full_solves;
        self.joins += o.joins;
        self.base_entries += o.base_entries;
        self.base_members += o.base_members;
        for (a, b) in self.t.iter_mut().zip(&o.t) {
            *a += b;
        }
    }
}

impl Fast {
    fn build() -> Self {
        let trellis = Trellis::new();
        let (msets, mset_of) = msets_of(&trellis);
        let mut fallback = Vec::with_capacity(2 * 2 * 2 * GOLAY_STATES);
        for kind in 0..2usize {
            for p in 0..2u32 {
                for r in 0..2u32 {
                    for s in 0..GOLAY_STATES {
                        let patterns = if kind == 0 { trellis.prefixes[s] } else { trellis.suffixes[s] };
                        let set = SectionSet { patterns: patterns.to_vec(), p, k_parity: Some(r) };
                        let bound = BOUND_CLASS[r as usize];
                        let mut t = 8usize;
                        let y = loop {
                            let pts = set.enumerate_below(t);
                            let found = pts
                                .into_iter()
                                .filter(|y| {
                                    let (cost, key) = cost_key_of(y, p);
                                    class_of_key(key) == r && member(bound, cost, key)
                                })
                                .min_by_key(|y| (y.iter().map(|&v| (v as i64) * (v as i64)).sum::<i64>(), *y));
                            if let Some(y) = found {
                                break y;
                            }
                            t *= 2;
                            assert!(t < 4096, "no member found");
                        };
                        fallback.push(y);
                    }
                }
            }
        }
        let branch_ix: Vec<[usize; BRANCHES]> = (0..GOLAY_STATES)
            .map(|s8| {
                let m = mset_of[s8];
                core::array::from_fn(|bi| m * BRANCHES + msets[m].iter().position(|&x| x == trellis.branches[s8][bi].0).expect("byte in its set"))
            })
            .collect();
        Self { trellis, msets, fallback, branch_ix, solver: Solver { n_shrinks: 8 }, shrinks_end: 7, warm: 2 }
    }

    /// The retained configuration of f1enclazy: the rule truncated to three
    /// shrinks on every section.
    fn truncated() -> Self {
        let mut f = Self::build();
        f.shrinks_end = 3;
        f.solver.n_shrinks = 3;
        f
    }

    fn fb(&self, kind: usize, p: u32, r: u32, state: usize) -> &[i32; SECTION] {
        &self.fallback[((kind * 2 + p as usize) * 2 + r as usize) * GOLAY_STATES + state]
    }

    /// One end state's entry: the fallback (exact), then the two patterns —
    /// base-only when `lazy`, the full 7-shrink rule otherwise. Mirrors the
    /// bench's `nearest`: the fallback seeds the minimum, replacements need a
    /// strictly smaller distance.
    fn end_entry(&self, tab: &Tables, target: &[f64; SECTION], patterns: [u8; 2], r: u32, fallback: &[i32; SECTION], lazy: bool, stats: &mut Stats) -> Entry {
        let d_fb: f64 = fallback.iter().zip(target).map(|(&v, &x)| (v as f64 - x).powi(2)).sum();
        let mut e = Entry { value: d_fb, exact: true, y: *fallback };
        let mut lb = f64::INFINITY;
        let mut unresolved = false;
        let bound = BOUND_CLASS[r as usize];
        let solver = Solver { n_shrinks: self.shrinks_end };
        for &c in &patterns {
            let mut closed = false;
            let mut base_dist = f64::INFINITY;
            let pick = solver.solve(tab, c, r, bound, lazy, &mut closed, &mut base_dist);
            if !lazy {
                stats.full_solves += 1;
            }
            if closed || !lazy {
                if let Some(pk) = pick {
                    if pk.dist < e.value {
                        e = Entry { value: pk.dist, exact: true, y: pk.y };
                    }
                }
            } else {
                // A base that is not a member: its distance — the coset-parity
                // optimum — bounds the pattern's answer from below.
                lb = lb.min(base_dist);
                unresolved = true;
            }
        }
        if unresolved && lb < e.value {
            Entry { value: lb, exact: false, y: e.y }
        } else {
            e
        }
    }

    /// The same state, resolved in full: the fallback and the full rule on
    /// both patterns.
    fn end_resolve(&self, tab: &Tables, target: &[f64; SECTION], patterns: [u8; 2], r: u32, fallback: &[i32; SECTION], stats: &mut Stats) -> Entry {
        self.end_entry(tab, target, patterns, r, fallback, false, stats)
    }

    fn mid_entry(&self, tab: &Tables, b: u8, delta: u32, lazy: bool, stats: &mut Stats) -> Entry {
        let mut closed = false;
        let mut base_dist = f64::INFINITY;
        let pick = self.solver.solve(tab, b, delta, BOUND_MIXED, lazy, &mut closed, &mut base_dist);
        if !lazy {
            stats.full_solves += 1;
        }
        if lazy && !closed {
            return Entry { value: base_dist, exact: false, y: [0; SECTION] };
        }
        match pick {
            Some(pk) => Entry { value: pk.dist, exact: true, y: pk.y },
            None => INFEASIBLE,
        }
    }

    /// The bench's `encode_at_scale`, lazy or not. Returns the point and the
    /// join's cost.
    pub fn encode_at_scale(&self, x: &[f64; 24], s: f64, lazy: bool, stats: &mut Stats) -> ([i32; 24], f64) {
        let part = |lo: usize| -> [f64; SECTION] { core::array::from_fn(|j| x[lo + j] / s) };
        let (t1, t2, t3) = (part(0), part(8), part(16));
        let mut best: (f64, [i32; 24]) = (f64::INFINITY, [0; 24]);
        for p in 0..2u32 {
            let clock = Instant::now();
            let tabs = [Tables::build(&t1, p), Tables::build(&t2, p), Tables::build(&t3, p)];
            stats.t[0] += clock.elapsed().as_secs_f64();
            let clock = Instant::now();
            // Entries: e1[r][s8], e2[b][δ] (b indexed by its position in the
            // 128 middle bytes, via mset and offset), e3[r_out][s16].
            let mut e1: Vec<Entry> = Vec::with_capacity(2 * GOLAY_STATES);
            let mut e3: Vec<Entry> = Vec::with_capacity(2 * GOLAY_STATES);
            let mut e2: Vec<[Entry; 2]> = Vec::with_capacity(8 * BRANCHES);
            if lazy {
                // Bases only, one gather per pattern for both parities.
                for (kind, tab, target, out) in [(0usize, &tabs[0], &t1, &mut e1), (1, &tabs[2], &t3, &mut e3)] {
                    out.resize(2 * GOLAY_STATES, INFEASIBLE);
                    for st in 0..GOLAY_STATES {
                        let patterns = if kind == 0 { self.trellis.prefixes[st] } else { self.trellis.suffixes[st] };
                        let mut ent = [Entry { value: 0.0, exact: true, y: [0; SECTION] }; 2];
                        let mut lb = [f64::INFINITY; 2];
                        let mut unresolved = [false; 2];
                        for r in 0..2usize {
                            let fb = self.fb(kind, p, r as u32, st);
                            ent[r] = Entry { value: fb.iter().zip(target).map(|(&v, &x)| (v as f64 - x).powi(2)).sum(), exact: true, y: *fb };
                        }
                        for &c in &patterns {
                            let pair = base_pair(tab, c, BOUND_CLASS);
                            for r in 0..2usize {
                                let (d, m, y) = pair[r];
                                stats.base_entries += 1;
                                if m {
                                    stats.base_members += 1;
                                    if d < ent[r].value {
                                        ent[r] = Entry { value: d, exact: true, y };
                                    }
                                } else {
                                    lb[r] = lb[r].min(d);
                                    unresolved[r] = true;
                                }
                            }
                        }
                        for r in 0..2usize {
                            out[r * GOLAY_STATES + st] = if unresolved[r] && lb[r] < ent[r].value { Entry { value: lb[r], exact: false, y: ent[r].y } } else { ent[r] };
                        }
                    }
                }
                for m in 0..8usize {
                    for &b in &self.msets[m] {
                        let pair = base_pair(&tabs[1], b, [BOUND_MIXED; 2]);
                        stats.base_entries += 2;
                        stats.base_members += pair[0].1 as usize + pair[1].1 as usize;
                        e2.push(core::array::from_fn(|d| {
                            let (dist, mem, y) = pair[d];
                            Entry { value: dist, exact: mem, y }
                        }));
                    }
                }
            } else {
                for r in 0..2u32 {
                    for s8 in 0..GOLAY_STATES {
                        e1.push(self.end_entry(&tabs[0], &t1, self.trellis.prefixes[s8], r, self.fb(0, p, r, s8), false, stats));
                    }
                }
                for r in 0..2u32 {
                    for s16 in 0..GOLAY_STATES {
                        e3.push(self.end_entry(&tabs[2], &t3, self.trellis.suffixes[s16], r, self.fb(1, p, r, s16), false, stats));
                    }
                }
                for m in 0..8usize {
                    for &b in &self.msets[m] {
                        e2.push([self.mid_entry(&tabs[1], b, 0, false, stats), self.mid_entry(&tabs[1], b, 1, false, stats)]);
                    }
                }
            }

            // The join: best path by current values, with the indices of its
            // three entries.
            let join = |e1: &[Entry], e2: &[[Entry; 2]], e3: &[Entry], stats: &mut Stats| -> (f64, usize, usize, u32, usize) {
                stats.joins += 1;
                let mut bp = (f64::INFINITY, 0usize, 0usize, 0u32, 0usize);
                for r in 0..2u32 {
                    for s8 in 0..GOLAY_STATES {
                        let i1 = r as usize * GOLAY_STATES + s8;
                        let a = e1[i1].value;
                        if a >= bp.0 {
                            continue;
                        }
                        for (bi, &(_, s16)) in self.trellis.branches[s8].iter().enumerate() {
                            let i2 = self.branch_ix[s8][bi];
                            for delta in 0..2u32 {
                                let r_out = (p ^ r ^ delta) & 1;
                                let i3 = r_out as usize * GOLAY_STATES + s16 as usize;
                                let c = a + e2[i2][delta as usize].value + e3[i3].value;
                                if c < bp.0 {
                                    bp = (c, i1, i2, delta, i3);
                                }
                            }
                        }
                    }
                }
                bp
            };

            stats.t[1] += clock.elapsed().as_secs_f64();
            if lazy {
                // 1. A feasible path close to the optimum: resolve the sections
                //    of the best bound path, at most `warm` times.
                let clock = Instant::now();
                let warm = self.warm;
                let mut u = best.0;
                for _ in 0..warm {
                    let (c, i1, i2, d, i3) = join(&e1, &e2, &e3, stats);
                    let mut all = true;
                    if !e1[i1].exact {
                        let (r, s8) = (i1 / GOLAY_STATES, i1 % GOLAY_STATES);
                        e1[i1] = self.end_resolve(&tabs[0], &t1, self.trellis.prefixes[s8], r as u32, self.fb(0, p, r as u32, s8), stats);
                        all = false;
                    }
                    if !e2[i2][d as usize].exact {
                        let (m, k) = (i2 / BRANCHES, i2 % BRANCHES);
                        e2[i2][d as usize] = self.mid_entry(&tabs[1], self.msets[m][k], d, false, stats);
                        all = false;
                    }
                    if !e3[i3].exact {
                        let (r, s16) = (i3 / GOLAY_STATES, i3 % GOLAY_STATES);
                        e3[i3] = self.end_resolve(&tabs[2], &t3, self.trellis.suffixes[s16], r as u32, self.fb(1, p, r as u32, s16), stats);
                        all = false;
                    }
                    if all {
                        u = c;
                        break;
                    }
                }
                stats.t[2] += clock.elapsed().as_secs_f64();
                let clock = Instant::now();
                // 2. One pass over the 8,192 paths: the best path through every
                //    entry (with current values), and the best all-exact path U.
                stats.joins += 1;
                let mut b1 = vec![f64::INFINITY; e1.len()];
                let mut b2 = vec![[f64::INFINITY; 2]; e2.len()];
                let mut b3 = vec![f64::INFINITY; e3.len()];
                for r in 0..2u32 {
                    for s8 in 0..GOLAY_STATES {
                        let i1 = r as usize * GOLAY_STATES + s8;
                        let a = e1[i1];
                        for (bi, &(_, s16)) in self.trellis.branches[s8].iter().enumerate() {
                            let i2 = self.branch_ix[s8][bi];
                            for delta in 0..2u32 {
                                let r_out = (p ^ r ^ delta) & 1;
                                let i3 = r_out as usize * GOLAY_STATES + s16 as usize;
                                let (v2, v3) = (e2[i2][delta as usize], e3[i3]);
                                let c = a.value + v2.value + v3.value;
                                if c < b1[i1] {
                                    b1[i1] = c;
                                }
                                if c < b2[i2][delta as usize] {
                                    b2[i2][delta as usize] = c;
                                }
                                if c < b3[i3] {
                                    b3[i3] = c;
                                }
                                if a.exact && v2.exact && v3.exact && c < u {
                                    u = c;
                                }
                            }
                        }
                    }
                }
                stats.t[3] += clock.elapsed().as_secs_f64();
                let clock = Instant::now();
                // 3. Resolve every unresolved entry whose best path is under U.
                for i1 in 0..e1.len() {
                    if !e1[i1].exact && b1[i1] < u {
                        let (r, s8) = (i1 / GOLAY_STATES, i1 % GOLAY_STATES);
                        e1[i1] = self.end_resolve(&tabs[0], &t1, self.trellis.prefixes[s8], r as u32, self.fb(0, p, r as u32, s8), stats);
                    }
                }
                for i2 in 0..e2.len() {
                    for d in 0..2u32 {
                        if !e2[i2][d as usize].exact && b2[i2][d as usize] < u {
                            let (m, k) = (i2 / BRANCHES, i2 % BRANCHES);
                            e2[i2][d as usize] = self.mid_entry(&tabs[1], self.msets[m][k], d, false, stats);
                        }
                    }
                }
                for i3 in 0..e3.len() {
                    if !e3[i3].exact && b3[i3] < u {
                        let (r, s16) = (i3 / GOLAY_STATES, i3 % GOLAY_STATES);
                        e3[i3] = self.end_resolve(&tabs[2], &t3, self.trellis.suffixes[s16], r as u32, self.fb(1, p, r as u32, s16), stats);
                    }
                }
                // 4. What is still unresolved cannot beat U: invisible to the
                //    final join.
                for e in e1.iter_mut().chain(e3.iter_mut()).chain(e2.iter_mut().flatten()) {
                    if !e.exact {
                        *e = INFEASIBLE;
                    }
                }
                stats.t[4] += clock.elapsed().as_secs_f64();
            }
            let clock = Instant::now();
            let (c, i1, i2, d, i3) = join(&e1, &e2, &e3, stats);
            stats.t[5] += clock.elapsed().as_secs_f64();
            if c < best.0 {
                let mut y = [0i32; 24];
                y[..8].copy_from_slice(&e1[i1].y);
                y[8..16].copy_from_slice(&e2[i2][d as usize].y);
                y[16..].copy_from_slice(&e3[i3].y);
                best = (c, y);
            }
        }
        (best.1, best.0)
    }

    pub fn t_of(x: &[f64; 24], y: &[i32; 24]) -> f64 {
        let dot: f64 = x.iter().zip(y).map(|(&a, &b)| a * b as f64).sum();
        let nn: f64 = y.iter().map(|&b| (b as f64) * (b as f64)).sum();
        if nn > 0.0 { dot / nn.sqrt() } else { f64::NEG_INFINITY }
    }
}

/// `cost` and lexicographic key of a point of a section with parity `p`,
/// read back from its values (the decoder's inverse).
fn cost_key_of(y: &[i32; SECTION], p: u32) -> (u32, u32) {
    let mut cost = 0u32;
    let mut key = 0u32;
    for j in 0..SECTION {
        let d = y[j] - p as i32;
        let c = (d.div_euclid(2)).rem_euclid(2) as u32;
        let o = p + 2 * c;
        let k = (y[j] - o as i32).div_euclid(4);
        let r = rank_from_k(o, k).min(15);
        cost += (2 * r + 1) * (2 * r + 1);
        key |= r << (4 * (7 - j));
    }
    (cost, key)
}

// ---------------------------------------------------------------------------
// The measurement
// ---------------------------------------------------------------------------

/// The adaptive scale of f1enclazy: `α · ‖x‖ / √24`, α chosen on Gaussian
/// evaluation blocks (the journal's caveat stands: not refitted here).
const ALPHA: f64 = 0.321;
/// The retained pair of scales: `(s₀, 1.14·s₀)`.
const RATIO: f64 = 1.14;
/// F1c's gate, µs per block per core (docs/ROADMAP.md §2.2).
const GATE_US: f64 = 656.0;
/// Shells 2..=12 of the ball, the control's codebook.
const BALL12_SHELLS: usize = 11;

fn read_f64_le(path: &str) -> Vec<f64> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    assert!(bytes.len().is_multiple_of(8), "{path}: {} bytes is not a whole number of f64", bytes.len());
    bytes.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().expect("8 bytes"))).collect()
}

/// `(matrix, column block)` per block of the dump, from its sidecar.
fn read_meta(path: &str) -> Vec<(String, usize)> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let mut lines = text.lines();
    assert_eq!(lines.next(), Some("matrix,colblock"), "{path}: unexpected header");
    lines
        .map(|l| {
            let (m, s) = l.split_once(',').unwrap_or_else(|| panic!("{path}: bad line {l:?}"));
            (m.to_string(), s.parse().unwrap_or_else(|e| panic!("{path}: {l:?}: {e}")))
        })
        .collect()
}

/// What one timed pass produced: wall time, counters, and the best `t` per block.
struct Pass {
    us_per_block: f64,
    n_scales: usize,
    stats: Stats,
    ts: Vec<f64>,
    /// Per-block wall time, seconds — filled only when asked (the retained pass).
    per_block: Vec<f64>,
}

fn run_pass(f: &Fast, lazy: bool, xs: &[[f64; 24]], scales_of: &dyn Fn(&[f64; 24]) -> Vec<f64>, per_block: bool) -> Pass {
    let t0 = Instant::now();
    let mut stats = Stats::default();
    let mut n_scales = 0usize;
    let mut ts = Vec::with_capacity(xs.len());
    let mut pb = Vec::with_capacity(if per_block { xs.len() } else { 0 });
    for x in xs {
        let tb = Instant::now();
        let mut best = (f64::NEG_INFINITY, [0i32; 24]);
        for s in scales_of(x) {
            let (y, _) = f.encode_at_scale(x, s, lazy, &mut stats);
            let t = Fast::t_of(x, &y);
            if t > best.0 {
                best = (t, y);
            }
            n_scales += 1;
        }
        ts.push(best.0);
        if per_block {
            pb.push(tb.elapsed().as_secs_f64());
        }
    }
    let el = t0.elapsed().as_secs_f64();
    Pass { us_per_block: 1e6 * el / xs.len() as f64, n_scales, stats, ts, per_block: pb }
}

fn describe(label: &str, p: &Pass, n: usize) -> String {
    let per_scale = p.n_scales as f64;
    let members = if p.stats.base_entries > 0 { 100.0 * p.stats.base_members as f64 / p.stats.base_entries as f64 } else { f64::NAN };
    let ph: Vec<String> = p.stats.t.iter().map(|v| format!("{:.0}", 1e6 * v / per_scale)).collect();
    format!(
        "{label:<44} {:>8.1} µs/bloc  {:>7.1} µs/(bloc,éch.)   {:>6.1} résolutions complètes/éch.   base membre {:>5.1} %   phases [{}]   ({} blocs, {:.1} éch./bloc)",
        p.us_per_block,
        p.us_per_block * n as f64 / per_scale,
        p.stats.full_solves as f64 / per_scale,
        members,
        ph.join("/"),
        n,
        per_scale / n as f64
    )
}

/// `t` of the ball-12 control from the shell maxima.
fn t_ball12(d: &[f64]) -> f64 {
    d[..BALL12_SHELLS]
        .iter()
        .enumerate()
        .map(|(i, &v)| v / ((16 * (i + 2)) as f64).sqrt())
        .fold(f64::NEG_INFINITY, f64::max)
}

/// Shape–gain MSE per weight with a 1-bit gain on the block norm, as f1bench.
fn mse_shape_gain(xx: &[f64], t: &[f64], centroids: &[f64]) -> f64 {
    xx.iter()
        .zip(t)
        .map(|(&x2, &tv)| {
            let g = centroids[nearest_centroid(centroids, x2.sqrt())];
            x2 - 2.0 * g * tv + g * g
        })
        .sum::<f64>()
        / (DIM * xx.len()) as f64
}

struct Retention {
    f1: f64,
    ctrl: f64,
    /// Paired per-weight error difference F1 − control: mean and standard error.
    paired: (f64, f64),
    power: f64,
}

fn retention(xs: &[[f64; 24]], t_f1: &[f64], t_ctrl: &[f64]) -> Retention {
    let n = xs.len();
    let xx: Vec<f64> = xs.iter().map(|x| x.iter().map(|v| v * v).sum()).collect();
    let norms: Vec<f64> = xx.iter().map(|v| v.sqrt()).collect();
    let centroids = lloyd_max(&norms, 1, 60);
    let power = xx.iter().sum::<f64>() / (DIM * n) as f64;
    let m_f1 = mse_shape_gain(&xx, t_f1, &centroids);
    let m_ctrl = mse_shape_gain(&xx, t_ctrl, &centroids);
    let err = |i: usize, t: f64| {
        let g = centroids[nearest_centroid(&centroids, norms[i])];
        xx[i] - 2.0 * g * t + g * g
    };
    let d: Vec<f64> = (0..n).map(|i| (err(i, t_f1[i]) - err(i, t_ctrl[i])) / (DIM as f64 * power)).collect();
    let m = d.iter().sum::<f64>() / n as f64;
    let sd = (d.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (n as f64 - 1.0)).sqrt();
    Retention {
        f1: retention_pct(m_f1 / power, 2.0),
        ctrl: retention_pct(m_ctrl / power, 2.0),
        paired: (m, sd / (n as f64).sqrt()),
        power,
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let blocks_path = args.get(1).expect("usage: f1encreal <blocks.f64> [rowscale.f64|-] [meta.csv|-] [n]");
    let scale_path = args.get(2).filter(|s| s.as_str() != "-");
    let meta_path = args.get(3).filter(|s| s.as_str() != "-");
    let n_max: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(usize::MAX);

    // ---- the real blocks ----
    let raw = read_f64_le(blocks_path);
    assert!(raw.len().is_multiple_of(DIM), "{blocks_path}: {} values is not a whole number of blocks", raw.len());
    let n_file = raw.len() / DIM;
    let scales: Option<Vec<f64>> = scale_path.map(|p| {
        let s = read_f64_le(p);
        assert_eq!(s.len(), n_file, "{p}: one row scale per block expected");
        s
    });
    let meta: Option<Vec<(String, usize)>> = meta_path.map(|p| {
        let m = read_meta(p);
        assert_eq!(m.len(), n_file, "{p}: one line per block expected");
        m
    });
    let mut real: Vec<[f64; 24]> = Vec::with_capacity(n_file);
    let mut keep: Vec<usize> = Vec::with_capacity(n_file);
    let mut zero = 0usize;
    for i in 0..n_file.min(n_max) {
        let mut x: [f64; 24] = raw[i * DIM..(i + 1) * DIM].try_into().expect("24");
        if let Some(s) = &scales {
            if s[i] > 0.0 {
                for v in x.iter_mut() {
                    *v /= s[i];
                }
            }
        }
        if x.iter().all(|&v| v == 0.0) {
            zero += 1;
            continue;
        }
        real.push(x);
        keep.push(i);
    }
    let n = real.len();
    let sqrt_dim = (DIM as f64).sqrt();
    let norms: Vec<f64> = real.iter().map(|x| x.iter().map(|v| v * v).sum::<f64>().sqrt()).collect();
    let kurt = {
        let vals: Vec<f64> = real.iter().flat_map(|x| x.iter().copied()).collect();
        let m2 = vals.iter().map(|v| v * v).sum::<f64>() / vals.len() as f64;
        let m4 = vals.iter().map(|v| v.powi(4)).sum::<f64>() / vals.len() as f64;
        m4 / (m2 * m2)
    };
    println!(
        "blocs réels : {n_file} dans {blocks_path}, {n} encodés ({zero} nuls écartés), {}",
        if scales.is_some() { "divisés par l'échelle de ligne servie" } else { "bruts" }
    );
    println!(
        "  ‖x‖ : min {:.3}  médiane {:.3}  max {:.3}   kurtosis des coordonnées {kurt:.2} (gaussienne : 3,00)",
        norms.iter().cloned().fold(f64::INFINITY, f64::min),
        {
            let mut s = norms.clone();
            s.sort_by(f64::total_cmp);
            s[s.len() / 2]
        },
        norms.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    );

    // ---- the Gaussian control: the bench's stream, past the 4,000 training blocks ----
    let mut rng = SplitMix64::new(0x0f1b_2026_0904);
    for _ in 0..4_000 {
        gauss_block(&mut rng);
    }
    let gauss: Vec<[f64; 24]> = (0..n).map(|_| gauss_block(&mut rng)).collect();

    let t0 = Instant::now();
    let full = Fast::build();
    let trunc = Fast::truncated();
    println!("encodeurs préparés en {:.2} s\n", t0.elapsed().as_secs_f64());
    let leech = llvq_core::Leech::new();

    let two = |x: &[f64; 24]| -> Vec<f64> {
        let s0 = ALPHA * x.iter().map(|v| v * v).sum::<f64>().sqrt() / sqrt_dim;
        vec![s0, s0 * RATIO]
    };

    // ---- 1. exactness on the real blocks: lazy = exhaustive, same rule ----
    let t0 = Instant::now();
    let (mut same, mut tie, mut outside, mut checked) = (0usize, 0usize, 0usize, 0usize);
    for x in &real {
        for s in two(x) {
            let mut st = Stats::default();
            let (ye, ce) = trunc.encode_at_scale(x, s, false, &mut st);
            let (yl, cl) = trunc.encode_at_scale(x, s, true, &mut st);
            checked += 1;
            assert!((cl - ce).abs() <= 1e-9 * ce.max(1.0), "bloc: coût paresseux {cl} contre exhaustif {ce} à s={s:.5}");
            if yl == ye { same += 1 } else { tie += 1 }
            if !leech.contains(&point_to_natural(&yl, &trunc.trellis.code.order)) {
                outside += 1;
            }
        }
    }
    println!(
        "exactitude (règle tronquée, 2 échelles) : {checked} (bloc, échelle) — {same} mêmes points, {tie} à coût égal, {outside} hors Λ₂₄   ({:.1} s)\n",
        t0.elapsed().as_secs_f64()
    );

    // ---- 2. cost, one core, real against Gaussian in the same process ----
    println!("coût par bloc, un cœur, 2 échelles adaptatives (s₀, {RATIO}·s₀), α = {ALPHA} :");
    let mut retained: Option<Pass> = None;
    let mut retained_gauss: Option<Pass> = None;
    for (set_label, xs) in [("réel", &real), ("gaussien", &gauss)] {
        let p = run_pass(&trunc, true, xs, &two, true);
        println!("  {set_label:<9} {}", describe("paresseux, règle tronquée [retenu]", &p, n));
        let p2 = run_pass(&full, true, xs, &two, false);
        println!("  {set_label:<9} {}", describe("paresseux, règle complète", &p2, n));
        let p3 = run_pass(&trunc, false, xs, &two, false);
        println!("  {set_label:<9} {}", describe("exhaustif, règle tronquée", &p3, n));
        let p4 = run_pass(&full, false, xs, &two, false);
        println!("  {set_label:<9} {}", describe("exhaustif, règle complète", &p4, n));
        if set_label == "réel" {
            retained = Some(p);
        } else {
            retained_gauss = Some(p);
        }
    }
    let retained = retained.expect("real pass ran");
    let retained_gauss = retained_gauss.expect("gaussian pass ran");
    let rate = |p: &Pass| 100.0 * p.stats.base_members as f64 / p.stats.base_entries.max(1) as f64;
    println!(
        "\nverdict : paresseux 2 échelles tronqué sur blocs réels = {:.1} µs/bloc, porte {GATE_US:.0} → {} ; ×{:.2} le gaussien ({:.1} µs) ; base membre {:.1} % contre {:.1} % ; résolutions complètes/éch. {:.1} contre {:.1}",
        retained.us_per_block,
        if retained.us_per_block < GATE_US { "SOUS la porte" } else { "AU-DESSUS de la porte" },
        retained.us_per_block / retained_gauss.us_per_block,
        retained_gauss.us_per_block,
        rate(&retained),
        rate(&retained_gauss),
        retained.stats.full_solves as f64 / retained.n_scales as f64,
        retained_gauss.stats.full_solves as f64 / retained_gauss.n_scales as f64
    );

    // ---- 3. by matrix and by column position, retained pass ----
    if let Some(meta) = &meta {
        println!("\npar matrice et par tiers de position de colonne (règle tronquée, 2 échelles) :");
        let mut names: Vec<String> = meta.iter().map(|(m, _)| m.clone()).collect();
        names.sort();
        names.dedup();
        for name in &names {
            let max_col = meta.iter().filter(|(m, _)| m == name).map(|(_, c)| *c).max().unwrap_or(0);
            for tier in 0..3usize {
                let (lo, hi) = (tier * (max_col + 1) / 3, (tier + 1) * (max_col + 1) / 3);
                let idx: Vec<usize> = keep
                    .iter()
                    .enumerate()
                    .filter(|(_, &i)| meta[i].0 == *name && (lo..hi).contains(&meta[i].1))
                    .map(|(k, _)| k)
                    .collect();
                if idx.is_empty() {
                    continue;
                }
                let sub: Vec<[f64; 24]> = idx.iter().map(|&k| real[k]).collect();
                let secs: f64 = idx.iter().map(|&k| retained.per_block[k]).sum();
                let mut st = Stats::default();
                for x in &sub {
                    let mut s1 = Stats::default();
                    for s in two(x) {
                        trunc.encode_at_scale(x, s, true, &mut s1);
                    }
                    st.add(&s1);
                }
                let per_scale = (2 * sub.len()) as f64;
                println!(
                    "  {name:<20} colonnes {lo:>3}..{hi:<3} {:>6} blocs  {:>7.1} µs/bloc  base membre {:>5.1} %  résolutions complètes/éch. {:>6.1}",
                    sub.len(),
                    1e6 * secs / sub.len() as f64,
                    100.0 * st.base_members as f64 / st.base_entries.max(1) as f64,
                    st.full_solves as f64 / per_scale
                );
            }
        }
    }

    // ---- 4. retention: F1 rank codebook against the ball-12 control ----
    println!("\nrétention (règle F1b : 1 bit de gain sur la norme, centroïdes ajustés sur les mêmes blocs, MSE / puissance moyenne) :");
    let searcher = Searcher::new();
    for (set_label, xs, pass) in [("réel", &real, &retained), ("gaussien", &gauss, &retained_gauss)] {
        let t0 = Instant::now();
        let dots = precompute13(&searcher, xs);
        let t_ctrl: Vec<f64> = dots.iter().map(|d| t_ball12(&d.d)).collect();
        let el = t0.elapsed().as_secs_f64();
        let r = retention(xs, &pass.ts, &t_ctrl);
        println!(
            "  {set_label:<9} témoin boule-12 {:6.2} %   F1 rang (paresseux, 2 éch.) {:6.2} %   Δ {:+.2} pp   écart apparié {:+.5} ± {:.5}   (puissance {:.4}, témoin {el:.1} s sur tous les cœurs)",
            r.ctrl,
            r.f1,
            r.f1 - r.ctrl,
            r.paired.0,
            r.paired.1,
            r.power
        );
    }
}
