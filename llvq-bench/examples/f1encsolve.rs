//! The fast F1 section solver on rank regions: closed-form membership,
//! per-coordinate tables, the bench's candidate rule reproduced exactly —
//! checked against `RankRegion` of `f1rankbench.rs` point for point, and timed.
//!
//! `cargo run --release -p llvq-bench --example f1encsolve -- [n_targets]`
//!
//! ## What this is
//!
//! The bench encoder decides membership of a candidate by hashing its rank
//! vector into a `HashSet` of 2,048 rows, after rebuilding the ranks from the
//! point (`RankRegion::contains`, ~69 ns per candidate, `f1encprof.rs`). The
//! 2,048 lowest-cost rows of a class are exactly
//! `{ρ : cost(ρ) < C_r} ∪ {cost(ρ) = C_r, ρ ≤ cut_r lexicographically}` with
//! `cost(ρ) = Σ(2ρ_j + 1)²` — verified by enumeration on 2026-09-05:
//! class 0 `C = 88`, cut `(1,1,0,3,0,0,1,1)`; class 1 `C = 96`, cut
//! `(0,0,2,2,2,0,1,1)`; the mixed middle set `C = 72`, cut `(3,0,0,0,1,0,1,0)`,
//! 1,240 rows of class 0. So membership is one integer compare, and a rank
//! vector is eight nibbles.
//!
//! The candidate rule of the bench (`candidates_with`: the rounded coset point,
//! the D₈ parity repair on the least decided coordinate, then sixteen
//! two-coordinate re-roundings, all of it at eight radial shrinks of the
//! target) is kept **unchanged**, so the answers are the bench's answers and
//! the retention already measured applies. What changes is the cost: the
//! per-coordinate rounding is done once per (section, parity, shrink) for the
//! two offsets a coordinate can take and shared by every pattern; a neighbour
//! is a two-nibble edit of the base with O(1) distance, cost and key deltas;
//! and a candidate whose distance cannot beat the incumbent is never tested.
//!
//! ## What is measured
//!
//! 1. Equivalence: on random section targets at the useful scale, for every
//!    pattern and both parities, the solver's point equals
//!    `RankRegion::nearest_constrained`'s point (or both are `None`), and its
//!    distance matches to 1e-9. Ties in distance between distinct points are
//!    the only admissible difference and are counted.
//! 2. Cost: ns per solve for the base-only step and for the full rule, and
//!    the share of solves the base step closes (member at shrink 1.0 = the
//!    exact coset optimum, nothing can beat it).
//! 3. The trellis join alone: 8,192 three-term sums with the argmin.

use llvq_bench::f1::{SectionSet, Trellis, GOLAY_STATES, SECTION};
use llvq_bench::gauss_block;
use llvq_core::SplitMix64;
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

// ---------------------------------------------------------------------------
// The bench's RankRegion, copied from f1rankbench.rs (2026-09-05) as the oracle
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

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    End(u32),
    Mid,
}

pub struct RankRegion {
    set: SectionSet,
    table: Arc<RankTable>,
    mode: Mode,
}

impl RankRegion {
    fn new(set: SectionSet, table: Arc<RankTable>, mode: Mode) -> Self {
        Self { set, table, mode }
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
    fn solve(&self, tab: &Tables, pattern: u8, want: u32, bound: Bound, base_only: bool, base_closed: &mut bool) -> Option<Pick> {
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

fn section_target(rng: &mut SplitMix64, s: f64) -> [f64; SECTION] {
    let b = gauss_block(rng);
    core::array::from_fn(|j| b[j] / s)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n_targets: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(300);
    let table = Arc::new(rank_table());
    let trellis = Trellis::new();
    let mut rng = SplitMix64::new(0x0f1e_2026_0905);

    // Section sets: the 128 prefix bytes and the 128 middle bytes, each under
    // both parities p. The suffixes are the prefixes' twins structurally.
    let mut prefix_bytes: Vec<u8> = trellis.prefixes.iter().flatten().copied().collect();
    prefix_bytes.sort_unstable();
    prefix_bytes.dedup();
    let middle_bytes = trellis.middle_bytes();
    assert_eq!(prefix_bytes.len(), 128);
    assert_eq!(middle_bytes.len(), 128);
    // Every pattern byte has even weight (self-dual code, octad support), which
    // is what makes Σk mod 2 equal to the rank class for every pattern.
    for &b in prefix_bytes.iter().chain(&middle_bytes) {
        assert!(b.count_ones().is_multiple_of(2), "pattern {b:#04x} has odd weight");
    }

    let solver = Solver { n_shrinks: 8 };
    let scale = 0.344f64; // the f1scale.rs optimum: targets sit at the region boundary
    println!("résolveur en forme close contre RankRegion — {n_targets} cibles de section à s = {scale}, 128 préfixes + 128 octets de milieu, 2 parités p, 2 parités k\n");

    // ---- 1. equivalence ----
    let mut checked = 0usize;
    let mut ties = 0usize;
    let mut base_closed_n = 0usize;
    let t0 = Instant::now();
    for _ in 0..n_targets {
        let target = section_target(&mut rng, scale);
        for p in 0..2u32 {
            let tab = Tables::build(&target, p);
            for (bytes, mode) in [(&prefix_bytes, None), (&middle_bytes, Some(Mode::Mid))] {
                for &pattern in bytes.iter() {
                    for r in 0..2u32 {
                        let (region, bound) = match mode {
                            None => (RankRegion::new(SectionSet { patterns: vec![pattern], p, k_parity: Some(r) }, table.clone(), Mode::End(r)), BOUND_CLASS[r as usize]),
                            Some(m) => (RankRegion::new(SectionSet { patterns: vec![pattern], p, k_parity: None }, table.clone(), m), BOUND_MIXED),
                        };
                        let oracle = region.nearest_constrained(&target, pattern, r);
                        let mut closed = false;
                        let got = solver.solve(&tab, pattern, r, bound, false, &mut closed);
                        base_closed_n += closed as usize;
                        checked += 1;
                        match (oracle, got) {
                            (None, None) => {}
                            (Some((y, d)), Some(g)) => {
                                assert!((g.dist - d).abs() <= 1e-9, "distance {} against oracle {d} (p={p} pattern={pattern:#04x} r={r})", g.dist);
                                if g.y != y {
                                    ties += 1;
                                }
                            }
                            (o, g) => panic!("p={p} pattern={pattern:#04x} r={r}: oracle {o:?} against solver {g:?}"),
                        }
                    }
                }
            }
        }
    }
    println!(
        "équivalence : {checked} résolutions (cible, p, motif, parité k), 0 désaccord de distance, {ties} points différents à distance égale ; base membre à 1,0 sur {:.1} %   ({:.1} s, oracle compris)",
        100.0 * base_closed_n as f64 / checked as f64,
        t0.elapsed().as_secs_f64()
    );

    // ---- 2. cost ----
    let targets: Vec<[f64; SECTION]> = (0..n_targets).map(|_| section_target(&mut rng, scale)).collect();
    // Tables: per (target, p), eight shrinks.
    let t0 = Instant::now();
    let mut sink = 0i32;
    for target in &targets {
        for p in 0..2u32 {
            let tab = Tables::build(target, p);
            sink = sink.wrapping_add(tab.t[7][7][1].k0);
        }
    }
    let per_tab = t0.elapsed().as_secs_f64() / (2 * n_targets) as f64;
    println!("tables par (section, p), 8 rétrécissements × 8 coordonnées × 2 décalages : {:.2} µs   [{sink}]", 1e6 * per_tab);

    let time_solves = |label: &str, base_only: bool, n_shrinks: usize| {
        let solver = Solver { n_shrinks };
        let t0 = Instant::now();
        let mut n = 0usize;
        let mut closed_n = 0usize;
        let mut acc = 0.0f64;
        for target in &targets {
            for p in 0..2u32 {
                let tab = Tables::build(target, p);
                for (bytes, bound_of) in [(&prefix_bytes, 0u8), (&middle_bytes, 1u8)] {
                    for &pattern in bytes.iter() {
                        for r in 0..2u32 {
                            let bound = if bound_of == 0 { BOUND_CLASS[r as usize] } else { BOUND_MIXED };
                            let mut closed = false;
                            if let Some(g) = solver.solve(&tab, pattern, r, bound, base_only, &mut closed) {
                                acc += g.dist;
                            }
                            closed_n += closed as usize;
                            n += 1;
                        }
                    }
                }
            }
        }
        let el = t0.elapsed().as_secs_f64() - per_tab * (2 * n_targets) as f64;
        println!("{label:<52} {:>7.1} ns par résolution, base membre {:.1} %   [Σd {acc:.3}]", 1e9 * el / n as f64, 100.0 * closed_n as f64 / n as f64);
    };
    time_solves("base seule (rétrécissement 1,0, premier candidat)", true, 1);
    time_solves("règle complète, 8 rétrécissements (= le banc)", false, 8);
    time_solves("règle tronquée à 3 rétrécissements (1,0 / 0,85 / 0,7)", false, 3);

    // ---- 3. the join ----
    // d1[p][r][s8], d2[p][b][δ] for the 128 middle bytes, d3[p][r_out][s16];
    // the branches (b, s16) of each s8 from the trellis. 8,192 sums.
    let mut d1 = vec![0.0f64; 2 * 2 * GOLAY_STATES];
    let mut d2 = vec![0.0f64; 2 * 256 * 2];
    let mut d3 = vec![0.0f64; 2 * 2 * GOLAY_STATES];
    for v in d1.iter_mut().chain(d2.iter_mut()).chain(d3.iter_mut()) {
        *v = 10.0 + 10.0 * rng.next_f64();
    }
    let branches: Vec<[(u8, u8); 16]> = (0..GOLAY_STATES).map(|s| core::array::from_fn(|i| trellis.branches[s][i])).collect();
    let t0 = Instant::now();
    let reps = 2000;
    let mut best_sum = 0.0;
    for _ in 0..reps {
        let mut best = (f64::INFINITY, 0u32);
        for p in 0..2usize {
            for r in 0..2usize {
                for s8 in 0..GOLAY_STATES {
                    let a = d1[(p * 2 + r) * GOLAY_STATES + s8];
                    for (bi, &(b, s16)) in branches[s8].iter().enumerate() {
                        for delta in 0..2usize {
                            let r_out = (p ^ r ^ delta) & 1;
                            let c = a + d2[(p * 256 + b as usize) * 2 + delta] + d3[(p * 2 + r_out) * GOLAY_STATES + s16 as usize];
                            if c < best.0 {
                                best = (c, ((p * 2 + r) * 64 + s8) as u32 * 32 + (bi * 2 + delta) as u32);
                            }
                        }
                    }
                }
            }
        }
        best_sum += best.0;
        d1[0] += 1e-9; // keep the optimizer honest
    }
    println!("jonction : 8 192 sommes de trois termes avec argmin : {:.2} µs   [{best_sum:.3}]", 1e6 * t0.elapsed().as_secs_f64() / reps as f64);
}
