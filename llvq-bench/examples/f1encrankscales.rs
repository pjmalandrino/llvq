//! How many scales the F1 production encoder needs — measured on the
//! universal-table (rank-region) encoder, not argued.
//!
//! `cargo run --release -p llvq-bench --example f1encrankscales -- [n_eval] [threads]`
//!
//! ## The question
//!
//! The bench encoder sweeps 18 scales `0.10·1.14^i` and, at each, runs a full
//! per-section search joined through the trellis: 13.3 ms per (block, scale)
//! on one core (`examples/f1encprof.rs`, 2026-09-05), 240 ms per block. F1c's
//! encoding-cost gate is 656 µs per block per core. Every scale a production
//! encoder keeps costs a full section solve plus a join, so the number of
//! scales is the first design decision, and it must come from the same
//! measurement the retention came from.
//!
//! ## What is measured
//!
//! Same blocks as `f1rankbench.rs` — seed `0x0f1b_2026_0904`, 4,000 training
//! blocks for the gain centroids drawn first, then the evaluation blocks — and
//! the same rank-table codebook, copied from that file (not modified). Arms,
//! all in one process, all paired against (a) block by block:
//!
//! * (a) the 18-point grid, which must reproduce the 89.05% of 2026-09-05;
//!   the per-block `t` at **every** scale is kept, so any sub-grid of the 18
//!   is scored exactly from that matrix without re-encoding;
//! * (b) 5 grid points and (c) 3 grid points: every window of consecutive
//!   grid indices is scored, and the pre-specified windows around the
//!   `f1scale.rs` optimum (s ≈ 0.344) are reported alongside the best;
//! * (d) one scale per block chosen from the block norm, `s = α·‖x‖/√24`,
//!   for α on a bracket around the median α the 18-point winners imply;
//! * (e) three points centred on that adaptive scale, ratio 1.14 and 1.30.
//!
//! Score: the F1b rule — `t = ⟨x, y⟩/‖y‖`, gain from the 1-bit Lloyd–Max
//! centroids on the block norm, `e² = ‖x‖² − 2gt + g²`, retention at
//! 2.000 b/dim. The ball-12 control is computed in the same process.
//!
//! Nothing here touches `f1.rs`, `f1rankbench.rs`, a format or a served path.

use llvq_bench::f1::{point_to_natural, SectionSet, Trellis, BRANCHES, GOLAY_STATES, SECTION};
use llvq_bench::{gauss_block, lloyd_max, nearest_centroid, precompute13, retention_pct};
use llvq_core::leech::DIM;
use llvq_core::SplitMix64;
use llvq_search::Searcher;
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

const BALL12_SHELLS: usize = 11;

fn t_ball12(d: &[f64; 12]) -> f64 {
    d[..BALL12_SHELLS]
        .iter()
        .enumerate()
        .map(|(i, &v)| v / ((16 * (i + 2)) as f64).sqrt())
        .fold(f64::NEG_INFINITY, f64::max)
}

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
// Running the arms
// ---------------------------------------------------------------------------

/// One block's result on one arm: best `t`, its codeword, index of the
/// winning scale within the arm's list, and the `t` at every scale of the arm.
struct Enc {
    t: f64,
    y: [i32; 24],
    win: usize,
    per_scale: Vec<f64>,
}

/// Encode every block on its own scale list, `threads` threads, blocks
/// interleaved so every thread sees the same mix.
fn encode_all(cb: &Codebook, xs: &[[f64; 24]], scales_of: &(dyn Fn(usize, &[f64; 24]) -> Vec<f64> + Sync), threads: usize) -> Vec<Enc> {
    let n = xs.len();
    let mut out: Vec<Option<Enc>> = (0..n).map(|_| None).collect();
    std::thread::scope(|sc| {
        let handles: Vec<_> = (0..threads)
            .map(|k| {
                sc.spawn(move || {
                    let mut local = Vec::new();
                    let mut i = k;
                    while i < n {
                        let x = &xs[i];
                        let scales = scales_of(i, x);
                        let mut best = (f64::NEG_INFINITY, [0i32; 24], 0usize);
                        let mut per = Vec::with_capacity(scales.len());
                        for (si, &s) in scales.iter().enumerate() {
                            let (t, y) = cb.t_at_scale(x, s);
                            per.push(t);
                            if t > best.0 {
                                best = (t, y, si);
                            }
                        }
                        local.push((i, Enc { t: best.0, y: best.1, win: best.2, per_scale: per }));
                        i += threads;
                    }
                    local
                })
            })
            .collect();
        for h in handles {
            for (i, e) in h.join().expect("thread") {
                out[i] = Some(e);
            }
        }
    });
    out.into_iter().map(|e| e.expect("every block encoded")).collect()
}

struct Scorer {
    xx: Vec<f64>,
    g: Vec<f64>,
    rate: f64,
}

impl Scorer {
    fn err(&self, i: usize, t: f64) -> f64 {
        self.xx[i] - 2.0 * self.g[i] * t + self.g[i] * self.g[i]
    }
    fn mse(&self, ts: &[f64]) -> f64 {
        ts.iter().enumerate().map(|(i, &t)| self.err(i, t)).sum::<f64>() / (DIM * ts.len()) as f64
    }
    fn retention(&self, ts: &[f64]) -> f64 {
        retention_pct(self.mse(ts), self.rate)
    }
    /// Paired against a reference: retention, Δ pp, per-block error
    /// difference mean ± 1 σ SE as % of the reference MSE, and the share of
    /// blocks where the arm found the reference's `t` (same codeword or one
    /// scoring identically).
    fn paired(&self, ts: &[f64], reference: &[f64]) -> (f64, f64, f64, f64, f64) {
        let n = ts.len();
        let ret = self.retention(ts);
        let ret_ref = self.retention(reference);
        let diff: Vec<f64> = (0..n).map(|i| self.err(i, ts[i]) - self.err(i, reference[i])).collect();
        let mean = diff.iter().sum::<f64>() / n as f64;
        let sd = (diff.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0)).sqrt();
        let se = sd / (n as f64).sqrt();
        let denom = DIM as f64 * self.mse(reference);
        let same = (0..n).filter(|&i| (ts[i] - reference[i]).abs() <= 1e-9).count() as f64 / n as f64;
        (ret, ret - ret_ref, 100.0 * mean / denom, 100.0 * se / denom, 100.0 * same)
    }
}

fn line(label: &str, sc: &Scorer, ts: &[f64], reference: &[f64], extra: &str) {
    let (ret, d, m, se, same) = sc.paired(ts, reference);
    println!("  {label:<46} rétention {ret:6.2} %   Δ(a) {d:+6.2} pp   perte MSE {m:+5.2} ± {se:4.2} %   même t que (a) sur {same:5.1} % des blocs{extra}");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n_eval: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(2_000);
    let threads: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(8);
    let n_train = 4_000usize;
    let rate: f64 = 48.0 / DIM as f64;

    let mut rng = SplitMix64::new(0x0f1b_2026_0904);
    let train: Vec<_> = (0..n_train).map(|_| gauss_block(&mut rng)).collect();
    let eval: Vec<[f64; 24]> = (0..n_eval).map(|_| gauss_block(&mut rng)).collect();
    println!("F1 encodeur de rangs, étude d'échelles — {n_train} blocs d'entraînement, {n_eval} d'évaluation, {threads} fils, débit {rate:.3} b/dim\n");

    let xx_train: Vec<f64> = train.iter().map(|x| x.iter().map(|v| v * v).sum()).collect();
    let norms_train: Vec<f64> = xx_train.iter().map(|v| v.sqrt()).collect();
    let centroids = lloyd_max(&norms_train, 1, 60);
    let xx: Vec<f64> = eval.iter().map(|x| x.iter().map(|v| v * v).sum()).collect();
    let norms: Vec<f64> = xx.iter().map(|v| v.sqrt()).collect();
    let g: Vec<f64> = norms.iter().map(|&n| centroids[nearest_centroid(&centroids, n)]).collect();
    let sc = Scorer { xx, g, rate };
    println!("centroïdes de gain (1 bit, sur les normes) : {centroids:?}");

    // ---- control, in chunks of `threads` blocks so precompute13 never spawns more ----
    let s = Searcher::new();
    let t0 = Instant::now();
    let mut t_ctrl = Vec::with_capacity(n_eval);
    for ch in eval.chunks(threads) {
        t_ctrl.extend(precompute13(&s, ch).iter().map(|d| t_ball12(&d.d)));
    }
    println!("témoin boule-12 + 1 bit de gain : MSE {:.6}  rétention {:.2} %   ({:.1} s)\n", sc.mse(&t_ctrl), sc.retention(&t_ctrl), t0.elapsed().as_secs_f64());

    let table = Arc::new(rank_table());
    println!("table universelle : 2 classes × 2048 rangs, milieu mixte = {} de classe 0", table.n0_mixed);
    let cb = Codebook::build(table);
    let leech = llvq_core::Leech::new();
    let outside = |enc: &[Enc]| enc.iter().filter(|e| !leech.contains(&point_to_natural(&e.y, &cb.trellis.code.order))).count();

    // ---- (a) the 18-point grid ----
    let grid: Vec<f64> = (0..18).map(|i| 0.10 * 1.14f64.powi(i)).collect();
    let t0 = Instant::now();
    let enc_a = encode_all(&cb, &eval, &|_, _| grid.clone(), threads);
    let wall_a = t0.elapsed().as_secs_f64();
    let t_a: Vec<f64> = enc_a.iter().map(|e| e.t).collect();
    println!(
        "(a) grille 18 points 0,10·1,14^i : MSE {:.6}  rétention {:.2} %   Δ témoin {:+.2} pp   ({:.1} min, {:.2} ms par (bloc, échelle) par fil, {} blocs hors Λ₂₄)",
        sc.mse(&t_a),
        sc.retention(&t_a),
        sc.retention(&t_a) - sc.retention(&t_ctrl),
        wall_a / 60.0,
        1e3 * wall_a * threads as f64 / (n_eval * grid.len()) as f64,
        outside(&enc_a)
    );
    let mut hist = vec![0usize; grid.len()];
    for e in &enc_a {
        hist[e.win] += 1;
    }
    let row: Vec<String> = hist.iter().map(|c| format!("{:.1}", 100.0 * *c as f64 / n_eval as f64)).collect();
    println!("    échelle gagnante, % des blocs par indice 0..17 : [{}]", row.join(" "));
    // α implied by the winners: s_win = α·‖x‖/√24.
    let mut alphas: Vec<f64> = enc_a.iter().enumerate().map(|(i, e)| grid[e.win] * (DIM as f64).sqrt() / norms[i]).collect();
    alphas.sort_by(f64::total_cmp);
    let q = |p: f64| alphas[((p * (n_eval as f64 - 1.0)).round() as usize).min(n_eval - 1)];
    let alpha_star = q(0.5);
    println!("    α = s_gagnante·√24/‖x‖ : quartiles {:.3} / {:.3} / {:.3}, déciles {:.3} / {:.3}\n", q(0.25), alpha_star, q(0.75), q(0.1), q(0.9));

    // ---- sub-grids of (a), scored from the kept matrix ----
    let sub = |idx: &[usize]| -> Vec<f64> {
        enc_a.iter().map(|e| idx.iter().map(|&i| e.per_scale[i]).fold(f64::NEG_INFINITY, f64::max)).collect()
    };
    println!("sous-grilles de (a), dérivées de la matrice t[bloc][échelle] sans ré-encodage :");
    let mut best_single = (f64::NEG_INFINITY, 0usize);
    for i in 0..grid.len() {
        let r = sc.retention(&sub(&[i]));
        if r > best_single.0 {
            best_single = (r, i);
        }
    }
    line(&format!("1 point fixe, le meilleur : s = {:.3} (i = {})", grid[best_single.1], best_single.1), &sc, &sub(&[best_single.1]), &t_a, "");
    for (w, label) in [(3usize, "(c)"), (5, "(b)"), (7, "7 pts"), (9, "9 pts")] {
        let mut best = (f64::NEG_INFINITY, 0usize);
        for start in 0..=(grid.len() - w) {
            let idx: Vec<usize> = (start..start + w).collect();
            let r = sc.retention(&sub(&idx));
            if r > best.0 {
                best = (r, start);
            }
        }
        let idx: Vec<usize> = (best.1..best.1 + w).collect();
        line(&format!("{label} {w} points consécutifs, la meilleure fenêtre : i = {}..{} (s = {:.3}..{:.3})", best.1, best.1 + w - 1, grid[best.1], grid[best.1 + w - 1]), &sc, &sub(&idx), &t_a, "");
    }
    // Pre-specified windows around the f1scale.rs optimum (s ≈ 0.344, between i = 9 and 10).
    line("(c) pré-spécifiée i = 8..10 (s = 0,285 / 0,325 / 0,371)", &sc, &sub(&[8, 9, 10]), &t_a, "");
    line("(b) pré-spécifiée i = 7..11 (s = 0,250 .. 0,423)", &sc, &sub(&[7, 8, 9, 10, 11]), &t_a, "");
    line("un point sur deux (9 pts, i pairs)", &sc, &sub(&[0, 2, 4, 6, 8, 10, 12, 14, 16]), &t_a, "");
    line("un point sur trois (6 pts, i = 1,4,..,16)", &sc, &sub(&[1, 4, 7, 10, 13, 16]), &t_a, "");
    line("les 12 premiers points (i = 0..11, s ≤ 0,423)", &sc, &sub(&(0..12).collect::<Vec<_>>()), &t_a, "");
    println!();

    // ---- (d) one adaptive scale per block, s = α·‖x‖/√24 ----
    let sqrt_dim = (DIM as f64).sqrt();
    let alphas_d: Vec<f64> = [-2i32, -1, 0, 1, 2].iter().map(|&k| alpha_star * 1.14f64.powi(k)).collect();
    println!("(d) une échelle par bloc, s = α·‖x‖/√24, α = α*·1,14^k autour de la médiane α* = {alpha_star:.3} :");
    let mut best_d = (f64::NEG_INFINITY, alpha_star);
    for &alpha in &alphas_d {
        let t0 = Instant::now();
        let enc = encode_all(&cb, &eval, &|i, _| vec![alpha * norms[i] / sqrt_dim], threads);
        let ts: Vec<f64> = enc.iter().map(|e| e.t).collect();
        let r = sc.retention(&ts);
        line(&format!("α = {alpha:.3}"), &sc, &ts, &t_a, &format!("   ({:.1} s, {} hors Λ₂₄)", t0.elapsed().as_secs_f64(), outside(&enc)));
        if r > best_d.0 {
            best_d = (r, alpha);
        }
    }
    println!();

    // ---- (e) three points centred on the adaptive scale ----
    println!("(e) trois échelles par bloc centrées sur s = α·‖x‖/√24 :");
    let mut done: Vec<(f64, f64)> = Vec::new();
    for (alpha, ratio) in [(alpha_star, 1.14f64), (best_d.1, 1.14), (alpha_star, 1.30), (best_d.1, 1.30)] {
        if done.contains(&(alpha, ratio)) {
            continue; // α* is also the best single α: the row would repeat
        }
        done.push((alpha, ratio));
        let t0 = Instant::now();
        let enc = encode_all(&cb, &eval, &|i, _| {
            let s0 = alpha * norms[i] / sqrt_dim;
            vec![s0 / ratio, s0, s0 * ratio]
        }, threads);
        let ts: Vec<f64> = enc.iter().map(|e| e.t).collect();
        let mut h = [0usize; 3];
        for e in &enc {
            h[e.win] += 1;
        }
        line(
            &format!("α = {alpha:.3}, ratio {ratio:.2} (s/r, s, s·r)"),
            &sc,
            &ts,
            &t_a,
            &format!("   gagnant bas/centre/haut {:.0}/{:.0}/{:.0} %   ({:.1} s, {} hors Λ₂₄)", 100.0 * h[0] as f64 / n_eval as f64, 100.0 * h[1] as f64 / n_eval as f64, 100.0 * h[2] as f64 / n_eval as f64, t0.elapsed().as_secs_f64(), outside(&enc)),
        );
    }
    // Five points centred on the adaptive scale, ratio 1.14.
    {
        let alpha = alpha_star;
        let t0 = Instant::now();
        let enc = encode_all(&cb, &eval, &|i, _| {
            let s0 = alpha * norms[i] / sqrt_dim;
            (-2i32..=2).map(|k| s0 * 1.14f64.powi(k)).collect()
        }, threads);
        let ts: Vec<f64> = enc.iter().map(|e| e.t).collect();
        let mut h = [0usize; 5];
        for e in &enc {
            h[e.win] += 1;
        }
        let row: Vec<String> = h.iter().map(|c| format!("{:.0}", 100.0 * *c as f64 / n_eval as f64)).collect();
        line(&format!("α = {alpha:.3}, 5 points ratio 1.14 (k = −2..2)"), &sc, &ts, &t_a, &format!("   gagnant [{}] %   ({:.1} s, {} hors Λ₂₄)", row.join("/"), t0.elapsed().as_secs_f64(), outside(&enc)));
    }

    // ---- (f) a line search in scale from the adaptive centre ----
    // Evaluate the centre, then its two neighbours at ratio r; walk in the
    // improving direction while t improves, at most `max_passes` encodes.
    println!("\n(f) recherche en ligne depuis s = α*·‖x‖/√24, ratio 1.14 : centre, ses deux voisins, puis marche tant que t monte :");
    for max_passes in [4usize, 6, 9] {
        let t0 = Instant::now();
        let alpha = alpha_star;
        let ratio = 1.14f64;
        let n = eval.len();
        let results: Vec<(f64, [i32; 24], usize)> = {
            let mut out: Vec<Option<(f64, [i32; 24], usize)>> = (0..n).map(|_| None).collect();
            std::thread::scope(|scp| {
                let handles: Vec<_> = (0..threads)
                    .map(|k| {
                        let cb = &cb;
                        let eval = &eval;
                        let norms = &norms;
                        scp.spawn(move || {
                            let mut local = Vec::new();
                            let mut i = k;
                            while i < n {
                                let x = &eval[i];
                                let s0 = alpha * norms[i] / sqrt_dim;
                                let mut passes = 0usize;
                                let at = |e: i32, passes: &mut usize| {
                                    *passes += 1;
                                    cb.t_at_scale(x, s0 * ratio.powi(e))
                                };
                                let c = at(0, &mut passes);
                                let lo = at(-1, &mut passes);
                                let hi = at(1, &mut passes);
                                let (mut best, dir, mut e) = if lo.0 > c.0 && lo.0 >= hi.0 {
                                    (lo, -1i32, -1i32)
                                } else if hi.0 > c.0 {
                                    (hi, 1, 1)
                                } else {
                                    (c, 0, 0)
                                };
                                while dir != 0 && passes < max_passes {
                                    e += dir;
                                    let nx = at(e, &mut passes);
                                    if nx.0 > best.0 {
                                        best = nx;
                                    } else {
                                        break;
                                    }
                                }
                                local.push((i, (best.0, best.1, passes)));
                                i += threads;
                            }
                            local
                        })
                    })
                    .collect();
                for h in handles {
                    for (i, r) in h.join().expect("thread") {
                        out[i] = Some(r);
                    }
                }
            });
            out.into_iter().map(|r| r.expect("encoded")).collect()
        };
        let ts: Vec<f64> = results.iter().map(|r| r.0).collect();
        let passes: f64 = results.iter().map(|r| r.2 as f64).sum::<f64>() / n as f64;
        let bad = results.iter().filter(|r| !leech.contains(&point_to_natural(&r.1, &cb.trellis.code.order))).count();
        line(&format!("au plus {max_passes} passes"), &sc, &ts, &t_a, &format!("   {passes:.2} passes en moyenne   ({:.1} s, {bad} hors Λ₂₄)", t0.elapsed().as_secs_f64()));
    }
    println!("\nmeilleure α à une échelle : {:.3} (rétention {:.2} %)", best_d.1, best_d.0);
}
