//! F1 with a UNIVERSAL 16 KiB decoder table, measured against the exact
//! 12/15/12 codebook and the ball-12 control — same blocks, same process.
//!
//! `cargo run --release -p llvq-bench --example f1rankbench -- [n_eval] [threads]`
//!
//! ## What is measured
//!
//! The F1 table floor of 2026-09-05 says a table ≤ 16 KiB costs 0.663 ms and
//! the 3,336 KiB exact table sits on the L2 plateau at 4.5 ms. `f1shrink.rs`
//! shows the only decoder under 16 KiB is one where every section reads ONE
//! shared table: a rank vector `ρ ∈ {0..4}^8`, decoded per coordinate as the
//! `ρ_j`-th value of the progression `o_j + 4Z` listed outward from zero,
//! `o_j = p + 2c_j`. Two parity classes of 2048 rows (Σ[ρ_j ∈ {1,2}] mod 2 is
//! pattern-independent), 4 bytes a row: 16 KiB. The ends read the 2048 rows
//! of their parity `r`; the middle reads the 2048 lowest-cost rows overall,
//! its outgoing parity being the class of the row.
//!
//! For p = 1 that region IS the exact lowest-norm region (every coordinate has
//! the same norm profile). For p = 0 it is a compromise between the o = 0 and
//! o = 2 profiles, and `f1shrink.rs` prices it at +0.45 dB (ends) and +0.53 dB
//! (middle) of second moment. What that costs in Gaussian retention is what
//! this measures — the second-moment model has already been caught at a
//! factor 2.2 on the one split it was checked against (13/13/13).
//!
//! ## Signed before the run (2026-09-05)
//!
//! Loss against the exact codebook on the same blocks: between 2.0 and 4.5 pp,
//! central 2.5. Kill the route if the loss exceeds 4.5 pp; keep it without
//! further question if under 2.0; in between, the decision belongs to F1c.
//!
//! The encoder's exactness on these regions is checked by `examples/f1rankenc.rs`
//! (99.8 % exact optima against exhaustive search, 2026-09-05).
//!
//! ## Also measured: the real access distribution of the EXACT decoder
//!
//! Nobody had computed it. For every block the exact codebook encodes, each
//! section's chosen point is located in its region: which parity `p`, and the
//! index quantile (points of lower norm / 2^w). That says how much of a table
//! is actually hot, and whether "regions equiprobable" holds for `p`.

use llvq_bench::f1::{Prepared, SectionSet, Trellis, BRANCHES, GOLAY_STATES, SECTION};
use llvq_bench::{gauss_block, lloyd_max, nearest_centroid, precompute13, retention_pct};
use llvq_core::leech::DIM;
use llvq_core::SplitMix64;
use llvq_search::Searcher;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

const BALL12_SHELLS: usize = 11;

fn t_ball12(d: &[f64; 12]) -> f64 {
    d[..BALL12_SHELLS]
        .iter()
        .enumerate()
        .map(|(i, &v)| v / ((16 * (i + 2)) as f64).sqrt())
        .fold(f64::NEG_INFINITY, f64::max)
}

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

// ---------------------------------------------------------------------------
// The rank-space table (same construction as f1shrink.rs)
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

/// Rank of `y` in the progression `o + 4Z`, or None if `y` is not in it or
/// beyond rank 7.
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
    /// Per parity class: the 2048 lowest-cost rank vectors, as a membership set.
    class: [HashSet<u32>; 2],
    /// The 2048 lowest-cost overall.
    mixed: HashSet<u32>,
    n0_mixed: usize,
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
                let n0_mixed = mixed.iter().filter(|r| rank_class(r) == 0).count();
                return RankTable {
                    class: [c0.iter().map(pack).collect(), c1.iter().map(pack).collect()],
                    mixed: mixed.iter().map(pack).collect(),
                    n0_mixed,
                };
            }
        }
        cap *= 2;
    }
}

// ---------------------------------------------------------------------------
// A section region that decodes through the universal table
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
enum Mode {
    /// End section, parity `r`: the 2048 rows of class `r`.
    End(u32),
    /// Middle section: the 2048 lowest-cost rows, both classes.
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
        // Lowest-norm member, found by growing a radius.
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

    /// Same candidate list as `Prepared::candidates_with`, copied so the two
    /// encoders differ only by their membership test.
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
}

/// What the generic codebook needs from a section region.
pub trait Region: Sync {
    fn nearest(&self, target: &[f64; SECTION]) -> ([i32; SECTION], f64);
    fn nearest_constrained(&self, target: &[f64; SECTION], pattern: u8, kparity: u32) -> Option<([i32; SECTION], f64)>;
}

impl Region for Prepared {
    fn nearest(&self, target: &[f64; SECTION]) -> ([i32; SECTION], f64) {
        Prepared::nearest(self, target)
    }
    fn nearest_constrained(&self, target: &[f64; SECTION], pattern: u8, kparity: u32) -> Option<([i32; SECTION], f64)> {
        Prepared::nearest_constrained(self, target, pattern, kparity)
    }
}

impl Region for RankRegion {
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
// The codebook, generic over the region — `f1::Codebook` with the type opened
// ---------------------------------------------------------------------------

/// One section's pick for each outgoing parity.
type MidPick = [Option<([i32; SECTION], f64)>; 2];
/// Cumulative norm histogram of a region, and its boundary shell.
type CumHist = (Vec<u64>, usize);

pub struct Codebook<R: Region> {
    pub trellis: Trellis,
    sec1: Vec<R>,
    sec2: Vec<R>,
    sec3: Vec<R>,
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

impl<R: Region> Codebook<R> {
    fn build(w: [u32; 3], mk1: impl Fn(SectionSet, u32) -> R, mk2: impl Fn(SectionSet, u32) -> R, mk3: impl Fn(SectionSet, u32) -> R) -> Self {
        let trellis = Trellis::new();
        let (msets, mset_of) = msets_of(&trellis);
        let mut sec1 = Vec::new();
        let mut sec3 = Vec::new();
        for p in 0..2u32 {
            for r in 0..2u32 {
                for s in 0..GOLAY_STATES {
                    sec1.push(mk1(trellis.section1(s, p, r), w[0]));
                    sec3.push(mk3(SectionSet { patterns: trellis.suffixes[s].to_vec(), p, k_parity: Some(r) }, w[2]));
                }
            }
        }
        let mut sec2 = Vec::new();
        for p in 0..2u32 {
            for m in &msets {
                sec2.push(mk2(SectionSet { patterns: m.clone(), p, k_parity: None }, w[1]));
            }
        }
        Self { trellis, sec1, sec2, sec3, mset_of, msets }
    }

    fn s1(&self, p: u32, r: u32, s8: usize) -> &R {
        &self.sec1[((p * 2 + r) as usize) * GOLAY_STATES + s8]
    }
    fn s3(&self, p: u32, r_out: u32, s16: usize) -> &R {
        &self.sec3[((p * 2 + r_out) as usize) * GOLAY_STATES + s16]
    }
    fn s2(&self, p: u32, mset: usize) -> &R {
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

    pub fn best_t(&self, x: &[f64; 24], scales: &[f64]) -> (f64, [i32; 24]) {
        let mut best = (f64::NEG_INFINITY, [0i32; 24]);
        for &s in scales {
            let y = self.encode_at_scale(x, s);
            let dot: f64 = x.iter().zip(&y).map(|(&a, &b)| a * b as f64).sum();
            let nn: f64 = y.iter().map(|&b| (b as f64) * (b as f64)).sum();
            if nn > 0.0 {
                let t = dot / nn.sqrt();
                if t > best.0 {
                    best = (t, y);
                }
            }
        }
        best
    }
}

fn encode_all<R: Region>(cb: &Codebook<R>, xs: &[[f64; 24]], scales: &[f64], threads: usize) -> Vec<(f64, [i32; 24])> {
    let chunk = xs.len().div_ceil(threads);
    let mut out = Vec::with_capacity(xs.len());
    std::thread::scope(|sc| {
        let handles: Vec<_> = xs.chunks(chunk).map(|c| sc.spawn(move || c.iter().map(|x| cb.best_t(x, scales)).collect::<Vec<_>>())).collect();
        for h in handles {
            out.extend(h.join().expect("thread"));
        }
    });
    out
}

// ---------------------------------------------------------------------------
// Locating a section point in its exact region: the real access distribution
// ---------------------------------------------------------------------------

struct Locator {
    trellis: Trellis,
    msets: Vec<Vec<u8>>,
    mset_of: Vec<usize>,
    /// Cumulative norm histogram per (section kind, p, r-or-2, state) at its w.
    cum: HashMap<(u8, u32, u32, usize), CumHist>,
}

impl Locator {
    fn new() -> Self {
        let trellis = Trellis::new();
        let (msets, mset_of) = msets_of(&trellis);
        Self { trellis, msets, mset_of, cum: HashMap::new() }
    }

    /// `(p, index quantile in [0,1), section kind)` for one section of a block.
    fn locate(&mut self, y: &[i32; 24], kind: u8, w: u32) -> (u32, f64) {
        let lo = 8 * kind as usize;
        let sec: [i32; SECTION] = y[lo..lo + 8].try_into().unwrap();
        let p = sec[0].rem_euclid(2) as u32;
        let pattern: u8 = (0..SECTION).fold(0u8, |a, j| a | ((((sec[j] - p as i32).div_euclid(2)).rem_euclid(2) as u8) << j));
        let kpar: u32 = (0..SECTION)
            .map(|j| ((sec[j] - p as i32 - 2 * ((pattern >> j) & 1) as i32).div_euclid(4)).rem_euclid(2) as u32)
            .fold(0, |a, b| a ^ b);
        let (state, r) = match kind {
            0 => (self.trellis.prefixes.iter().position(|pr| pr.contains(&pattern)).expect("prefix"), kpar),
            1 => (
                self.mset_of[self.trellis.branches.iter().position(|br| br.iter().any(|&(b, _)| b == pattern)).expect("middle")],
                2,
            ),
            _ => (self.trellis.suffixes.iter().position(|sf| sf.contains(&pattern)).expect("suffix"), kpar),
        };
        let key = (kind, p, r, state);
        if !self.cum.contains_key(&key) {
            let set = match kind {
                0 => self.trellis.section1(state, p, r),
                1 => SectionSet { patterns: self.msets[state].clone(), p, k_parity: None },
                _ => SectionSet { patterns: self.trellis.suffixes[state].to_vec(), p, k_parity: Some(r) },
            };
            let reg = set.region(w);
            let h = set.norm_histogram(reg.rho2);
            let mut c = Vec::with_capacity(h.len());
            let mut acc = 0u64;
            for v in h {
                c.push(acc);
                acc += v;
            }
            self.cum.insert(key, (c, reg.rho2));
        }
        let (c, rho2) = &self.cum[&key];
        let n: usize = sec.iter().map(|&v| (v * v) as usize).sum();
        assert!(n <= *rho2, "a section point outside its region: norm {n} > rho2 {rho2}");
        (p, c[n] as f64 / (1u64 << w) as f64)
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n_eval: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(2_000);
    let threads: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(8);
    let n_train = 4_000usize;
    let rate: f64 = 48.0 / DIM as f64;
    let w = [12u32, 15, 12];

    // The prereg's fixed seed, as in bin/f1bench: both arms see these blocks.
    let mut rng = SplitMix64::new(0x0f1b_2026_0904);
    let train: Vec<_> = (0..n_train).map(|_| gauss_block(&mut rng)).collect();
    let eval: Vec<_> = (0..n_eval).map(|_| gauss_block(&mut rng)).collect();
    println!("F1 table universelle — {n_train} blocs d'entraînement, {n_eval} d'évaluation, {threads} fils, débit {rate:.3} b/dim\n");

    let xx_train: Vec<f64> = train.iter().map(|x| x.iter().map(|v| v * v).sum()).collect();
    let xx_eval: Vec<f64> = eval.iter().map(|x| x.iter().map(|v| v * v).sum()).collect();
    let norms: Vec<f64> = xx_train.iter().map(|v| v.sqrt()).collect();
    let centroids = lloyd_max(&norms, 1, 60);

    // ---- control ----
    let s = Searcher::new();
    let t0 = std::time::Instant::now();
    let dots = precompute13(&s, &eval);
    let t_ctrl: Vec<f64> = dots.iter().map(|d| t_ball12(&d.d)).collect();
    let mse_ctrl = mse_shape_gain(&xx_eval, &t_ctrl, &centroids);
    let ret_ctrl = retention_pct(mse_ctrl, rate);
    println!("témoin boule-12 + 1 bit de gain : MSE {mse_ctrl:.6}  rétention {ret_ctrl:.2} %   ({:.1} s)", t0.elapsed().as_secs_f64());

    let scales: Vec<f64> = (0..18).map(|i| 0.10 * 1.14f64.powi(i)).collect();

    // ---- exact F1, through the generic codebook, checked against f1::Codebook ----
    let cb_exact: Codebook<Prepared> = Codebook::build(w, Prepared::new, Prepared::new, Prepared::new);
    {
        let reference = llvq_bench::f1::Codebook::new(w);
        for x in eval.iter().take(5) {
            let a = cb_exact.best_t(x, &scales);
            let b = reference.best_t(x, &scales);
            assert_eq!(a.1, b.1, "the generic codebook does not reproduce f1::Codebook");
        }
        println!("codebook générique = f1::Codebook sur 5 blocs (mêmes points)");
    }
    let t1 = std::time::Instant::now();
    let enc_exact = encode_all(&cb_exact, &eval, &scales, threads);
    let t_exact: Vec<f64> = enc_exact.iter().map(|e| e.0).collect();
    let mse_exact = mse_shape_gain(&xx_eval, &t_exact, &centroids);
    let ret_exact = retention_pct(mse_exact, rate);
    println!(
        "F1 exact 12/15/12 : MSE {mse_exact:.6}  rétention {ret_exact:.2} %   Δ témoin {:+.2} pp   ({:.1} min)",
        ret_exact - ret_ctrl,
        t1.elapsed().as_secs_f64() / 60.0
    );

    // ---- F1 with the universal table ----
    let table = Arc::new(rank_table());
    println!(
        "table universelle : 2 classes × 2048 rangs (16 Kio à 4 o), milieu mixte = {} de classe 0 + {} de classe 1",
        table.n0_mixed,
        2048 - table.n0_mixed
    );
    let tb = table.clone();
    let cb_rank: Codebook<RankRegion> = Codebook::build(
        w,
        |set, _| {
            let r = set.k_parity.unwrap();
            RankRegion::new(set, tb.clone(), Mode::End(r))
        },
        |set, _| RankRegion::new(set, tb.clone(), Mode::Mid),
        |set, _| {
            let r = set.k_parity.unwrap();
            RankRegion::new(set, tb.clone(), Mode::End(r))
        },
    );
    // Every region holds exactly 2^w members: the bijection, at the level of one section.
    {
        let t = Trellis::new();
        for (set, mode, wbits) in [
            (t.section1(5, 0, 1), Mode::End(1), 12u32),
            (t.section1(5, 1, 0), Mode::End(0), 12),
            (SectionSet { patterns: t.branches[3].iter().map(|&(b, _)| b).collect(), p: 0, k_parity: None }, Mode::Mid, 15),
            (SectionSet { patterns: t.branches[3].iter().map(|&(b, _)| b).collect(), p: 1, k_parity: None }, Mode::Mid, 15),
        ] {
            let reg = RankRegion::new(set.clone(), table.clone(), mode);
            let n = set.enumerate_below(200).iter().filter(|y| reg.contains(y)).count();
            assert_eq!(n, 1 << wbits, "a rank region is not 2^w");
        }
        println!("régions de rangs : exactement 2^w membres (4 régions sondées)");
    }
    let t2 = std::time::Instant::now();
    let enc_rank = encode_all(&cb_rank, &eval, &scales, threads);
    let leech = llvq_core::Leech::new();
    let bad = enc_rank.iter().filter(|e| !leech.contains(&llvq_bench::f1::point_to_natural(&e.1, &cb_rank.trellis.code.order))).count();
    assert_eq!(bad, 0, "the universal decoder produced points outside Λ₂₄");
    let t_rank: Vec<f64> = enc_rank.iter().map(|e| e.0).collect();
    let mse_rank = mse_shape_gain(&xx_eval, &t_rank, &centroids);
    let ret_rank = retention_pct(mse_rank, rate);
    println!(
        "F1 table universelle 16 Kio : MSE {mse_rank:.6}  rétention {ret_rank:.2} %   Δ témoin {:+.2} pp   Δ F1 exact {:+.2} pp   ({:.1} min, {bad} blocs hors Λ₂₄)",
        ret_rank - ret_ctrl,
        ret_rank - ret_exact,
        t2.elapsed().as_secs_f64() / 60.0
    );
    // Per-block paired difference of the score, for a range on the delta.
    let per_block: Vec<f64> = (0..n_eval)
        .map(|i| {
            let g = centroids[nearest_centroid(&centroids, xx_eval[i].sqrt())];
            (xx_eval[i] - 2.0 * g * t_rank[i] + g * g) - (xx_eval[i] - 2.0 * g * t_exact[i] + g * g)
        })
        .collect();
    let mean = per_block.iter().sum::<f64>() / n_eval as f64;
    let sd = (per_block.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / (n_eval as f64 - 1.0)).sqrt();
    let se = sd / (n_eval as f64).sqrt();
    println!(
        "  écart apparié de l'erreur par bloc : {mean:+.4} ± {se:.4} (1 σ) sur {n_eval} blocs ; la perte de MSE est {:.1} ± {:.1} %",
        100.0 * mean / (DIM as f64 * mse_exact),
        100.0 * se / (DIM as f64 * mse_exact)
    );

    // ---- the real access distribution of the exact decoder ----
    let mut loc = Locator::new();
    let mut p_count = [0usize; 2];
    let mut dec = [[0usize; 10]; 2]; // [kind ends/mid][decile of table index]
    for e in &enc_exact {
        for (kind, wb) in [(0u8, 12u32), (1, 15), (2, 12)] {
            let (p, q) = loc.locate(&e.1, kind, wb);
            if kind == 0 {
                p_count[p as usize] += 1;
            }
            let k = if kind == 1 { 1 } else { 0 };
            dec[k][((q * 10.0) as usize).min(9)] += 1;
        }
    }
    println!("\naccès réel du décodeur exact sur les {n_eval} blocs :");
    println!("  parité p des blocs : p=0 {:.1} %, p=1 {:.1} %", 100.0 * p_count[0] as f64 / n_eval as f64, 100.0 * p_count[1] as f64 / n_eval as f64);
    for (k, name, n) in [(0usize, "extrêmes (2 par bloc)", 2 * n_eval), (1, "milieu", n_eval)] {
        let row: Vec<String> = dec[k].iter().map(|&c| format!("{:.1}", 100.0 * c as f64 / n as f64)).collect();
        let top_half: usize = dec[k][5..].iter().sum();
        let top_quarter: usize = dec[k][8..].iter().sum();
        println!(
            "  {name:<22} déciles d'index (norme croissante) : [{}] ; moitié externe {:.1} %, cinquième externe {:.1} %",
            row.join(" "),
            100.0 * top_half as f64 / n as f64,
            100.0 * top_quarter as f64 / n as f64
        );
    }
}
