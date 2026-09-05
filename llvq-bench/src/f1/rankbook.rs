//! The bench's rank-region codebook — the yardstick of the production Trio
//! encoder — held here so that `tests/trio_encoder.rs` and
//! `examples/trioscales.rs` read one copy.
//!
//! Copied on 2026-09-05 from `examples/f1rankbench.rs` (`Membership`, `Mode`,
//! `RankRegion`, `Region`, `Codebook<R>`, `msets_of`, the ball-12 control and
//! the F1b score) without a change to any rule: the same candidate list as
//! `Prepared::candidates_with`, the same strict-`<` replacement, the same
//! fallback, the same join. The example keeps its own copy and is not
//! modified; what it measured (−0.6 pp of retention against exact F1,
//! `docs/mesures/f1-plancher-table-2026-09-05.txt`) is what this reproduces.
//! Slow by design — every candidate is a `HashSet` lookup — and that is the
//! point: `llvq_search::trio::Encoder` must return these points faster, not
//! other points.

use super::rank::{pack, rank_of, RankTable};
use super::{SectionSet, Trellis, BRANCHES, GOLAY_STATES, SECTION};
use crate::nearest_centroid;
use llvq_core::leech::DIM;
use std::collections::HashSet;
use std::sync::Arc;

/// Shells of the ball-12 control: `m = 2..=12`, the first eleven of `d`.
pub const BALL12_SHELLS: usize = 11;

/// `t = max_m d_m/√(16m)` over the ball-12 directions, from `BlockDots13::d`.
pub fn t_ball12(d: &[f64; 12]) -> f64 {
    d[..BALL12_SHELLS]
        .iter()
        .enumerate()
        .map(|(i, &v)| v / ((16 * (i + 2)) as f64).sqrt())
        .fold(f64::NEG_INFINITY, f64::max)
}

/// The F1b score: one gain bit on the block norm, `‖x − g·v̂‖² = ‖x‖² − 2gt + g²`
/// averaged per weight.
pub fn mse_shape_gain(xx: &[f64], t: &[f64], centroids: &[f64]) -> f64 {
    xx.iter()
        .zip(t)
        .map(|(&x2, &tv)| {
            let g = centroids[nearest_centroid(centroids, x2.sqrt())];
            x2 - 2.0 * g * tv + g * g
        })
        .sum::<f64>()
        / (DIM * xx.len()) as f64
}

/// The rows of [`RankTable`] as the sets the membership test needs.
pub struct Membership {
    /// Per parity class: the 2048 lowest-cost rank vectors.
    class: [HashSet<u32>; 2],
    /// The 2048 lowest-cost overall.
    mixed: HashSet<u32>,
}

impl Membership {
    pub fn new(t: &RankTable) -> Self {
        Self {
            class: [t.class_rows(0).iter().copied().collect(), t.class_rows(1).iter().copied().collect()],
            mixed: t.mixed_rows().into_iter().collect(),
        }
    }
}

#[derive(Clone, Copy)]
pub enum Mode {
    /// End section, parity `r`: the 2048 rows of class `r`.
    End(u32),
    /// Middle section: the 2048 lowest-cost rows, both classes.
    Mid,
}

/// A section region that decodes through the universal table.
pub struct RankRegion {
    set: SectionSet,
    table: Arc<Membership>,
    mode: Mode,
    fallback: [i32; SECTION],
}

impl RankRegion {
    pub fn new(set: SectionSet, table: Arc<Membership>, mode: Mode) -> Self {
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

/// One section's pick for each outgoing parity.
type MidPick = [Option<([i32; SECTION], f64)>; 2];

/// The codebook, generic over the region — `f1::Codebook` with the type opened.
pub struct Codebook<R: Region> {
    pub trellis: Trellis,
    sec1: Vec<R>,
    sec2: Vec<R>,
    sec3: Vec<R>,
    mset_of: Vec<usize>,
    msets: Vec<Vec<u8>>,
}

/// The eight distinct sets of sixteen middle bytes, and each state's.
pub fn msets_of(trellis: &Trellis) -> (Vec<Vec<u8>>, Vec<usize>) {
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
    pub fn build(w: [u32; 3], mk1: impl Fn(SectionSet, u32) -> R, mk2: impl Fn(SectionSet, u32) -> R, mk3: impl Fn(SectionSet, u32) -> R) -> Self {
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

    /// The nearest codeword to `x/s`, `x` in TRIO order, returned in trio order.
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

    /// The largest `t = ⟨x, y⟩/‖y‖` over `scales`, and its codeword.
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

impl Codebook<RankRegion> {
    /// The codebook on the universal table, as `f1rankbench.rs` builds it.
    pub fn rank(table: &RankTable) -> Self {
        let tb = Arc::new(Membership::new(table));
        let w = [12u32, 15, 12];
        let mk_end = |set: SectionSet, _: u32| {
            let r = set.k_parity.expect("an end section has a k-parity");
            RankRegion::new(set, tb.clone(), Mode::End(r))
        };
        Codebook::build(w, mk_end, |set, _| RankRegion::new(set, tb.clone(), Mode::Mid), mk_end)
    }
}

/// `best_t` over `scales` on every block, `threads` wide; blocks in trio order.
pub fn encode_all<R: Region>(cb: &Codebook<R>, xs: &[[f64; 24]], scales: &[f64], threads: usize) -> Vec<(f64, [i32; 24])> {
    let chunk = xs.len().div_ceil(threads.max(1));
    let mut out = Vec::with_capacity(xs.len());
    std::thread::scope(|sc| {
        let handles: Vec<_> = xs.chunks(chunk).map(|c| sc.spawn(move || c.iter().map(|x| cb.best_t(x, scales)).collect::<Vec<_>>())).collect();
        for h in handles {
            out.extend(h.join().expect("thread"));
        }
    });
    out
}
