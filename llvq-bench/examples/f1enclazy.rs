//! The production F1 encoder, prototyped: closed-form section solver, base
//! lower bounds, and a trellis join that decides which sections need the full
//! search — checked against the bench codebook block for block, and timed.
//!
//! `cargo run --release -p llvq-bench --example f1enclazy -- [n_blocks]`
//!
//! ## The design
//!
//! At one scale the bench solves 1,536 (section, p, pattern, parity) problems
//! then joins them (`f1encprof.rs`). `f1encsolve.rs` reproduces each solve in
//! closed form at 252 ns, which still puts one scale at ~390 µs and three
//! scales past F1c's 656 µs gate. This prototype adds the step that removes
//! most of the solves: the shrink-1.0 base candidate costs 45 ns, is the exact
//! coset-parity optimum, and so is a **lower bound** on the region-constrained
//! answer — and the answer itself when it is a member (half the time at the
//! useful scale). A join over lower bounds then names the only entries that
//! can still matter: after one feasible path of cost `U` is found, an entry
//! whose bound plus the best bounds of the other two sections reaches `U`
//! cannot be on a better path and is never solved in full.
//!
//! The result is the bench's answer — the same candidate rule, the same
//! strict-`<` tie-breaking, pruning only what cannot win — checked here
//! against `Codebook::encode_at_scale` of `f1rankbench.rs` (copied, not
//! modified), point for point.
//!
//! Nothing here touches `f1.rs`, a format or a served path.

// A prototype: the index loops mirror the bench's code they are checked
// against, and the solver's signature carries its counters.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

use llvq_bench::f1::{point_to_natural, SectionSet, Trellis, BRANCHES, GOLAY_STATES, SECTION};
use llvq_bench::gauss_block;
use llvq_core::leech::DIM;
use llvq_core::SplitMix64;
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

// ---------------------------------------------------------------------------
// The bench's rank-table codebook, copied from f1rankbench.rs (2026-09-05)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// The rank-space table — same construction as f1rankbench.rs (2026-09-05)
// ---------------------------------------------------------------------------

fn val(o: u32, rho: u32) -> i32 {
    match o {
        0 => {
            if rho == 0 {
                0
            } else {
                let m = 4 * rho.div_ceil(2) as i32;
                if rho.is_multiple_of(2) { -m } else { m }
            }
        }
        2 => {
            let m = (2 + 4 * (rho / 2)) as i32;
            if rho.is_multiple_of(2) { m } else { -m }
        }
        1 => {
            let m = (2 * rho + 1) as i32;
            if rho.is_multiple_of(2) { m } else { -m }
        }
        3 => {
            let m = (2 * rho + 1) as i32;
            if rho.is_multiple_of(2) { -m } else { m }
        }
        _ => unreachable!(),
    }
}

fn rank_of(o: u32, y: i32) -> Option<u32> {
    if y.rem_euclid(4) != o as i32 {
        return None;
    }
    (0..8u32).find(|&r| val(o, r) == y)
}

fn rank_class(rho: &[u32; SECTION]) -> u32 {
    rho.iter().filter(|&&r| r == 1 || r == 2).count() as u32 & 1
}

fn pack(rho: &[u32; SECTION]) -> u32 {
    rho.iter().enumerate().fold(0u32, |a, (j, &r)| a | (r << (4 * j)))
}

pub struct RankTable {
    class: [HashSet<u32>; 2],
    mixed: HashSet<u32>,
}

fn rank_table() -> RankTable {
    fn walk(j: usize, acc: i64, cap: i64, rho: &mut [u32; SECTION], out: &mut Vec<([u32; SECTION], i64)>) {
        if j == SECTION {
            out.push((*rho, acc));
            return;
        }
        for r in 0..8u32 {
            let c = (2 * r as i64 + 1).pow(2);
            if acc + c > cap {
                break;
            }
            rho[j] = r;
            walk(j + 1, acc + c, cap, rho, out);
        }
    }
    let mut cap = 64i64;
    loop {
        let mut all = Vec::new();
        walk(0, 0, cap, &mut [0; SECTION], &mut all);
        all.sort_by_key(|&(r, c)| (c, r));
        let c0: Vec<_> = all.iter().filter(|(r, _)| rank_class(r) == 0).take(2048).map(|&(r, _)| r).collect();
        let c1: Vec<_> = all.iter().filter(|(r, _)| rank_class(r) == 1).take(2048).map(|&(r, _)| r).collect();
        if c0.len() == 2048 && c1.len() == 2048 && all.len() >= 4096 {
            let cost = |r: &[u32; SECTION]| r.iter().map(|&x| (2 * x as i64 + 1).pow(2)).sum::<i64>();
            let kept = all.iter().take(4096).map(|&(_, c)| c).max().unwrap();
            if kept < cap && cost(&c0[2047]) < cap && cost(&c1[2047]) < cap {
                let mixed: Vec<_> = all.iter().take(2048).map(|&(r, _)| r).collect();
                return RankTable {
                    class: [c0.iter().map(pack).collect(), c1.iter().map(pack).collect()],
                    mixed: mixed.iter().map(pack).collect(),
                };
            }
        }
        cap *= 2;
    }
}

#[derive(Clone, Copy)]
enum Mode {
    End(u32),
    Mid,
}

pub struct RankRegion {
    set: SectionSet,
    table: Arc<RankTable>,
    mode: Mode,
    fallback: [i32; SECTION],
}

impl RankRegion {
    fn new(set: SectionSet, table: Arc<RankTable>, mode: Mode) -> Self {
        let mut r = Self { set, table, mode, fallback: [0; SECTION] };
        let mut t = 8usize;
        loop {
            let pts = r.set.enumerate_below(t);
            if let Some(y) = pts
                .into_iter()
                .filter(|y| r.contains(y))
                .min_by_key(|y| (y.iter().map(|&v| (v as i64) * (v as i64)).sum::<i64>(), *y))
            {
                r.fallback = y;
                return r;
            }
            t *= 2;
            assert!(t < 4096, "no member found");
        }
    }

    fn rho_of(&self, y: &[i32; SECTION]) -> Option<[u32; SECTION]> {
        let p = self.set.p as i32;
        let mut rho = [0u32; SECTION];
        for j in 0..SECTION {
            let d = y[j] - p;
            if d.rem_euclid(2) != 0 {
                return None;
            }
            let cj = (d.div_euclid(2)).rem_euclid(2) as u32;
            let o = self.set.p + 2 * cj;
            rho[j] = rank_of(o, y[j])?;
        }
        Some(rho)
    }

    pub fn contains(&self, y: &[i32; SECTION]) -> bool {
        if !self.set.contains(y) {
            return false;
        }
        let Some(rho) = self.rho_of(y) else { return false };
        let key = pack(&rho);
        match self.mode {
            Mode::End(r) => self.table.class[r as usize].contains(&key),
            Mode::Mid => self.table.mixed.contains(&key),
        }
    }

    fn k_parity_of(&self, pattern: u8, y: &[i32; SECTION]) -> u32 {
        (0..SECTION)
            .map(|j| {
                let base = self.set.p as i32 + 2 * ((pattern >> j & 1) as i32);
                ((y[j] - base).div_euclid(4)).rem_euclid(2) as u32
            })
            .fold(0, |a, b| a ^ b)
    }

    fn candidates_with(&self, pattern: u8, target: &[f64; SECTION], want_parity: Option<u32>) -> Vec<[i32; SECTION]> {
        let base: Vec<i32> = (0..SECTION).map(|j| self.set.p as i32 + 2 * ((pattern >> j & 1) as i32)).collect();
        let z: Vec<f64> = (0..SECTION).map(|j| (target[j] - base[j] as f64) / 4.0).collect();
        let mut k: Vec<i32> = z.iter().map(|v| v.round() as i32).collect();
        let mut regret: Vec<(f64, usize)> = (0..SECTION).map(|j| ((z[j] - k[j] as f64).abs(), j)).collect();
        regret.sort_by(|a, b| b.0.total_cmp(&a.0));
        if let Some(r) = want_parity {
            let sum: i32 = k.iter().sum();
            if sum.rem_euclid(2) != r as i32 {
                let j = regret[0].1;
                k[j] += if z[j] > k[j] as f64 { 1 } else { -1 };
            }
        }
        let build = |k: &[i32]| {
            let mut y = [0i32; SECTION];
            for j in 0..SECTION {
                y[j] = base[j] + 4 * k[j];
            }
            y
        };
        let mut out = vec![build(&k)];
        for &(_, j) in &regret {
            for step in [-1i32, 1] {
                let mut k2 = k.clone();
                k2[j] += step;
                if want_parity.is_some() {
                    let j2 = regret.iter().map(|&(_, x)| x).find(|&x| x != j).expect("8 > 1");
                    k2[j2] += if z[j2] > k[j2] as f64 { 1 } else { -1 };
                }
                out.push(build(&k2));
            }
        }
        out
    }

    fn nearest(&self, target: &[f64; SECTION]) -> ([i32; SECTION], f64) {
        let dist = |y: &[i32; SECTION]| -> f64 { y.iter().zip(target).map(|(&v, &x)| (v as f64 - x).powi(2)).sum() };
        let mut best = (self.fallback, dist(&self.fallback));
        for shrink in [1.0f64, 0.85, 0.7, 0.55, 0.4, 0.25, 0.1] {
            let mut scaled = [0.0f64; SECTION];
            for (j, v) in scaled.iter_mut().enumerate() {
                *v = target[j] * shrink;
            }
            for &pattern in &self.set.patterns {
                for cand in self.candidates_with(pattern, &scaled, self.set.k_parity) {
                    if !self.contains(&cand) {
                        continue;
                    }
                    let d = dist(&cand);
                    if d < best.1 {
                        best = (cand, d);
                    }
                }
            }
        }
        best
    }

    fn nearest_constrained(&self, target: &[f64; SECTION], pattern: u8, kparity: u32) -> Option<([i32; SECTION], f64)> {
        let dist = |y: &[i32; SECTION]| -> f64 { y.iter().zip(target).map(|(&v, &x)| (v as f64 - x).powi(2)).sum() };
        let mut best: Option<([i32; SECTION], f64)> = None;
        for shrink in [1.0f64, 0.85, 0.7, 0.55, 0.4, 0.25, 0.1, 0.0] {
            let mut scaled = [0.0f64; SECTION];
            for (j, v) in scaled.iter_mut().enumerate() {
                *v = target[j] * shrink;
            }
            for cand in self.candidates_with(pattern, &scaled, Some(kparity)) {
                if !self.contains(&cand) || self.k_parity_of(pattern, &cand) != kparity {
                    continue;
                }
                let d = dist(&cand);
                if best.as_ref().is_none_or(|&(_, bd)| d < bd) {
                    best = Some((cand, d));
                }
            }
        }
        best
    }
}

// ---------------------------------------------------------------------------
// The codebook of f1rankbench.rs, on rank regions only
// ---------------------------------------------------------------------------

type MidPick = [Option<([i32; SECTION], f64)>; 2];

pub struct Codebook {
    pub trellis: Trellis,
    sec1: Vec<RankRegion>,
    sec2: Vec<RankRegion>,
    sec3: Vec<RankRegion>,
    mset_of: Vec<usize>,
    msets: Vec<Vec<u8>>,
}

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

impl Codebook {
    fn build(table: Arc<RankTable>) -> Self {
        let trellis = Trellis::new();
        let (msets, mset_of) = msets_of(&trellis);
        let mut sec1 = Vec::new();
        let mut sec3 = Vec::new();
        for p in 0..2u32 {
            for r in 0..2u32 {
                for s in 0..GOLAY_STATES {
                    sec1.push(RankRegion::new(trellis.section1(s, p, r), table.clone(), Mode::End(r)));
                    sec3.push(RankRegion::new(
                        SectionSet { patterns: trellis.suffixes[s].to_vec(), p, k_parity: Some(r) },
                        table.clone(),
                        Mode::End(r),
                    ));
                }
            }
        }
        let mut sec2 = Vec::new();
        for p in 0..2u32 {
            for m in &msets {
                sec2.push(RankRegion::new(SectionSet { patterns: m.clone(), p, k_parity: None }, table.clone(), Mode::Mid));
            }
        }
        Self { trellis, sec1, sec2, sec3, mset_of, msets }
    }

    fn s1(&self, p: u32, r: u32, s8: usize) -> &RankRegion {
        &self.sec1[((p * 2 + r) as usize) * GOLAY_STATES + s8]
    }
    fn s3(&self, p: u32, r_out: u32, s16: usize) -> &RankRegion {
        &self.sec3[((p * 2 + r_out) as usize) * GOLAY_STATES + s16]
    }
    fn s2(&self, p: u32, mset: usize) -> &RankRegion {
        &self.sec2[p as usize * 8 + mset]
    }

    pub fn encode_at_scale(&self, x: &[f64; 24], s: f64) -> [i32; 24] {
        let part = |lo: usize| -> [f64; SECTION] {
            let mut t = [0.0f64; SECTION];
            for (j, v) in t.iter_mut().enumerate() {
                *v = x[lo + j] / s;
            }
            t
        };
        let (t1, t2, t3) = (part(0), part(8), part(16));
        let mut best: Option<(f64, [i32; 24])> = None;
        for p in 0..2u32 {
            let mut mid: Vec<MidPick> = Vec::new();
            for m in 0..8usize {
                for &b in &self.msets[m] {
                    mid.push([self.s2(p, m).nearest_constrained(&t2, b, 0), self.s2(p, m).nearest_constrained(&t2, b, 1)]);
                }
            }
            let mid_at = |m: usize, b: u8| -> &MidPick {
                let k = self.msets[m].iter().position(|&x| x == b).expect("byte in its set");
                &mid[m * BRANCHES + k]
            };
            let mut end: Vec<([i32; SECTION], f64)> = Vec::with_capacity(2 * GOLAY_STATES);
            for r_out in 0..2u32 {
                for s16 in 0..GOLAY_STATES {
                    end.push(self.s3(p, r_out, s16).nearest(&t3));
                }
            }
            for r in 0..2u32 {
                for s8 in 0..GOLAY_STATES {
                    let (y1, d1) = self.s1(p, r, s8).nearest(&t1);
                    let m = self.mset_of[s8];
                    for &(b, s16) in &self.trellis.branches[s8] {
                        for delta in 0..2u32 {
                            let Some((y2, d2)) = mid_at(m, b)[delta as usize] else { continue };
                            let r_out = (p ^ r ^ delta) & 1;
                            let (y3, d3) = end[r_out as usize * GOLAY_STATES + s16 as usize];
                            let cost = d1 + d2 + d3;
                            if best.as_ref().is_none_or(|&(bc, _)| cost < bc) {
                                let mut y = [0i32; 24];
                                y[..8].copy_from_slice(&y1);
                                y[8..16].copy_from_slice(&y2);
                                y[16..].copy_from_slice(&y3);
                                best = Some((cost, y));
                            }
                        }
                    }
                }
            }
        }
        best.expect("never empty").1
    }

    /// `t = ⟨x, y⟩/‖y‖` of the codeword found at scale `s`, and the codeword.
    pub fn t_at_scale(&self, x: &[f64; 24], s: f64) -> (f64, [i32; 24]) {
        let y = self.encode_at_scale(x, s);
        let dot: f64 = x.iter().zip(&y).map(|(&a, &b)| a * b as f64).sum();
        let nn: f64 = y.iter().map(|&b| (b as f64) * (b as f64)).sum();
        (if nn > 0.0 { dot / nn.sqrt() } else { f64::NEG_INFINITY }, y)
    }
}

// ---------------------------------------------------------------------------
// The closed-form solver
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
#[derive(Default, Clone, Copy)]
pub struct Stats {
    pub full_solves: usize,
    pub joins: usize,
    /// Seconds in: tables, bases, warm joins + their solves, the per-entry pass, survivor solves, final join.
    pub t: [f64; 6],
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
                                if m {
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

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(200);
    let table = Arc::new(rank_table());
    let bench = Codebook::build(table);
    let t0 = Instant::now();
    let fast = Fast::build();
    println!("encodeur rapide préparé en {:.2} s (replis des 512 régions d'extrémité)", t0.elapsed().as_secs_f64());
    let leech = llvq_core::Leech::new();

    // The bench's evaluation blocks.
    let mut rng = SplitMix64::new(0x0f1b_2026_0904);
    for _ in 0..4_000 {
        gauss_block(&mut rng);
    }
    let xs: Vec<[f64; 24]> = (0..n).map(|_| gauss_block(&mut rng)).collect();
    let norms: Vec<f64> = xs.iter().map(|x| x.iter().map(|v| v * v).sum::<f64>().sqrt()).collect();
    let sqrt_dim = (DIM as f64).sqrt();
    let alpha = 0.321f64;

    // ---- 1. exactness against the bench, full and lazy, at three scales per block ----
    let (mut same_full, mut same_lazy, mut tie_full, mut tie_lazy, mut checked, mut outside) = (0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
    let t0 = Instant::now();
    for (i, x) in xs.iter().enumerate() {
        let s0 = alpha * norms[i] / sqrt_dim;
        for s in [s0 / 1.14, s0, s0 * 1.14] {
            let yb = bench.encode_at_scale(x, s);
            let cb: f64 = (0..24).map(|j| (yb[j] as f64 - x[j] / s).powi(2)).sum();
            let mut st = Stats::default();
            let (yf, cf) = fast.encode_at_scale(x, s, false, &mut st);
            let (yl, cl) = fast.encode_at_scale(x, s, true, &mut st);
            checked += 1;
            assert!((cf - cb).abs() <= 1e-9 * cb.max(1.0), "bloc {i} s={s:.4}: coût complet {cf} contre banc {cb}");
            assert!((cl - cb).abs() <= 1e-9 * cb.max(1.0), "bloc {i} s={s:.4}: coût paresseux {cl} contre banc {cb}");
            if yf == yb { same_full += 1 } else { tie_full += 1 }
            if yl == yb { same_lazy += 1 } else { tie_lazy += 1 }
            if !leech.contains(&point_to_natural(&yl, &fast.trellis.code.order)) {
                outside += 1;
            }
        }
    }
    println!(
        "exactitude : {checked} (bloc, échelle) — complet : {same_full} mêmes points, {tie_full} à coût égal ; paresseux : {same_lazy} mêmes points, {tie_lazy} à coût égal ; {outside} hors Λ₂₄   ({:.1} s, banc compris)",
        t0.elapsed().as_secs_f64()
    );

    // ---- 2. cost, single thread ----
    let time = |label: &str, fast: &Fast, lazy: bool, scales_of: &dyn Fn(usize) -> Vec<f64>| {
        let t0 = Instant::now();
        let mut st = Stats::default();
        let mut n_scales = 0usize;
        let mut tsum = 0.0;
        for (i, x) in xs.iter().enumerate() {
            let mut best = (f64::NEG_INFINITY, [0i32; 24]);
            for s in scales_of(i) {
                let (y, _) = fast.encode_at_scale(x, s, lazy, &mut st);
                let t = Fast::t_of(x, &y);
                if t > best.0 {
                    best = (t, y);
                }
                n_scales += 1;
            }
            tsum += best.0;
        }
        let el = t0.elapsed().as_secs_f64();
        let ph: Vec<String> = st.t.iter().map(|v| format!("{:.0}", 1e6 * v / n_scales as f64)).collect();
        println!(
            "{label:<48} {:>8.1} µs/bloc  {:>7.1} µs par (bloc, échelle)   {:.0} résolutions complètes, {:.1} jonctions   phases µs [tables/bases/amorce/passe/survivants/finale] = [{}]   [t moyen {:.4}]",
            1e6 * el / n as f64,
            1e6 * el / n_scales as f64,
            st.full_solves as f64 / n_scales as f64,
            st.joins as f64 / n_scales as f64,
            ph.join("/"),
            tsum / n as f64
        );
    };
    let one = |i: usize| vec![alpha * norms[i] / sqrt_dim];
    let three = |i: usize| {
        let s0 = alpha * norms[i] / sqrt_dim;
        vec![s0 / 1.14, s0, s0 * 1.14]
    };
    let grid = |_: usize| (0..18).map(|k| 0.10 * 1.14f64.powi(k)).collect::<Vec<_>>();
    let two = |i: usize| {
        let s0 = alpha * norms[i] / sqrt_dim;
        vec![s0 / 1.07, s0 * 1.07]
    };
    time("complet, 1 échelle adaptative", &fast, false, &one);
    time("paresseux, 1 échelle adaptative", &fast, true, &one);
    time("paresseux, 2 échelles adaptatives (s0/1,07, s0·1,07)", &fast, true, &two);
    time("paresseux, 3 échelles adaptatives (ratio 1,14)", &fast, true, &three);
    time("paresseux, grille 18 points du banc", &fast, true, &grid);
    for warm in [0usize, 1] {
        let mut f = Fast::build();
        f.warm = warm;
        time(&format!("paresseux, 1 échelle, amorce = {warm}"), &f, true, &one);
    }
    {
        let mut f = Fast::build();
        f.shrinks_end = 3;
        f.solver.n_shrinks = 3;
        time("paresseux, 1 échelle, règle tronquée à 3 rétrécissements", &f, true, &one);
        let mut diff = 0usize;
        for (i, x) in xs.iter().enumerate() {
            let s0 = alpha * norms[i] / sqrt_dim;
            let mut st = Stats::default();
            let (y8, _) = fast.encode_at_scale(x, s0, true, &mut st);
            let (y3, _) = f.encode_at_scale(x, s0, true, &mut st);
            diff += (y8 != y3) as usize;
        }
        println!("    règle tronquée : {diff} blocs sur {n} changent de point à l'échelle centrale");
        let two_up = |i: usize| {
            let s0 = alpha * norms[i] / sqrt_dim;
            vec![s0, s0 * 1.14]
        };
        time("paresseux, 2 échelles (s0, s0·1,14), règle tronquée", &f, true, &two_up);
        time("paresseux, 3 échelles (ratio 1,14), règle tronquée", &f, true, &three);
        time("paresseux, 2 échelles (s0, s0·1,14), règle complète", &fast, true, &two_up);
    }

    // The line search of f1encrankscales.rs (f): centre, two neighbours, walk.
    let t0 = Instant::now();
    let mut st = Stats::default();
    let mut passes_total = 0usize;
    for (i, x) in xs.iter().enumerate() {
        let s0 = alpha * norms[i] / sqrt_dim;
        let mut passes = 0usize;
        let at = |e: i32, st: &mut Stats, passes: &mut usize| {
            *passes += 1;
            let (y, _) = fast.encode_at_scale(x, s0 * 1.14f64.powi(e), true, st);
            (Fast::t_of(x, &y), y)
        };
        let c = at(0, &mut st, &mut passes);
        let lo = at(-1, &mut st, &mut passes);
        let hi = at(1, &mut st, &mut passes);
        let (mut best, dir, mut e) = if lo.0 > c.0 && lo.0 >= hi.0 { (lo, -1i32, -1i32) } else if hi.0 > c.0 { (hi, 1, 1) } else { (c, 0, 0) };
        while dir != 0 && passes < 6 {
            e += dir;
            let nx = at(e, &mut st, &mut passes);
            if nx.0 > best.0 {
                best = nx;
            } else {
                break;
            }
        }
        passes_total += passes;
    }
    let el = t0.elapsed().as_secs_f64();
    println!(
        "{:<48} {:>8.1} µs/bloc  {:>7.1} µs par (bloc, échelle)   {:.2} passes en moyenne, {:.0} résolutions complètes par passe",
        "paresseux, recherche en ligne (≤ 6 passes)",
        1e6 * el / n as f64,
        1e6 * el / passes_total as f64,
        passes_total as f64 / n as f64,
        st.full_solves as f64 / passes_total as f64
    );

    // ---- 3. retention on the bench's 2,000 blocks, F1b rule, with the fast encoder ----
    let n_eval = 2_000usize;
    let mut rng = SplitMix64::new(0x0f1b_2026_0904);
    let train: Vec<[f64; 24]> = (0..4_000).map(|_| gauss_block(&mut rng)).collect();
    let eval: Vec<[f64; 24]> = (0..n_eval).map(|_| gauss_block(&mut rng)).collect();
    let norms_train: Vec<f64> = train.iter().map(|x| x.iter().map(|v| v * v).sum::<f64>().sqrt()).collect();
    let centroids = llvq_bench::lloyd_max(&norms_train, 1, 60);
    let xx: Vec<f64> = eval.iter().map(|x| x.iter().map(|v| v * v).sum()).collect();
    let nrm: Vec<f64> = xx.iter().map(|v| v.sqrt()).collect();
    let g: Vec<f64> = nrm.iter().map(|&v| centroids[llvq_bench::nearest_centroid(&centroids, v)]).collect();
    let err = |i: usize, t: f64| xx[i] - 2.0 * g[i] * t + g[i] * g[i];
    let retention = |ts: &[f64]| llvq_bench::retention_pct(ts.iter().enumerate().map(|(i, &t)| err(i, t)).sum::<f64>() / (DIM * n_eval) as f64, 2.0);
    println!("\nrétention sur les {n_eval} blocs du banc (règle F1b, témoin 92,00, grille 18 points 89,05 selon f1encrankscales) :");
    let run = |f: &Fast, scales_of: &dyn Fn(usize) -> Vec<f64>| -> (Vec<f64>, f64) {
        let t0 = Instant::now();
        let ts: Vec<f64> = eval
            .iter()
            .enumerate()
            .map(|(i, x)| {
                let mut st = Stats::default();
                scales_of(i).into_iter().map(|s| Fast::t_of(x, &f.encode_at_scale(x, s, true, &mut st).0)).fold(f64::NEG_INFINITY, f64::max)
            })
            .collect();
        (ts, t0.elapsed().as_secs_f64())
    };
    let s_of = |i: usize| alpha * nrm[i] / sqrt_dim;
    let (t_ref, el) = run(&fast, &|i| vec![s_of(i) / 1.14, s_of(i), s_of(i) * 1.14]);
    println!("  {:<52} rétention {:6.2} %   ({el:.1} s, 1 fil)", "3 échelles adaptatives, règle complète (= (e) du banc)", retention(&t_ref));
    let paired = |ts: &[f64]| {
        let d: Vec<f64> = (0..n_eval).map(|i| err(i, ts[i]) - err(i, t_ref[i])).collect();
        let m = d.iter().sum::<f64>() / n_eval as f64;
        let sd = (d.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (n_eval as f64 - 1.0)).sqrt();
        (m, sd / (n_eval as f64).sqrt())
    };
    let mut f3 = Fast::build();
    f3.shrinks_end = 3;
    f3.solver.n_shrinks = 3;
    for (label, f, sc) in [
        ("3 échelles adaptatives, règle tronquée à 3 rétrécissements", &f3, 3usize),
        ("2 échelles adaptatives (s0/1,07, s0·1,07), règle complète", &fast, 2),
        ("2 échelles adaptatives (s0, s0·1,14), règle complète", &fast, 4),
        ("2 échelles adaptatives (s0, s0·1,14), règle tronquée", &f3, 4),
        ("2 échelles adaptatives (s0/1,14, s0), règle complète", &fast, 5),
        ("1 échelle adaptative, règle complète (= (d) du banc)", &fast, 1),
    ] {
        let (ts, el) = run(f, &|i| {
            let s0 = s_of(i);
            match sc {
                1 => vec![s0],
                2 => vec![s0 / 1.07, s0 * 1.07],
                4 => vec![s0, s0 * 1.14],
                5 => vec![s0 / 1.14, s0],
                _ => vec![s0 / 1.14, s0, s0 * 1.14],
            }
        });
        let (m, se) = paired(&ts);
        println!("  {label:<52} rétention {:6.2} %   Δ(3 adaptatives) {:+.2} pp   écart apparié {m:+.4} ± {se:.4}   ({el:.1} s)", retention(&ts), retention(&ts) - retention(&t_ref));
    }
}
