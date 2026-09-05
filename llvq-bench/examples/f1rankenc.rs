//! Encoder exactness on the universal-table (rank-space) regions — the control
//! `examples/f1enc.rs` runs on the exact regions, run here on the rank regions.
//! Same code as `f1rankbench.rs` (kept in one file there); this binary exists so
//! the check can run while the bench binary is busy.
//!
//! `cargo run --release -p llvq-bench --example f1rankenc`
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
//! ## Also measured: the real access distribution of the EXACT decoder
//!
//! Nobody had computed it. For every block the exact codebook encodes, each
//! section's chosen point is located in its region: which parity `p`, and the
//! index quantile (points of lower norm / 2^w). That says how much of a table
//! is actually hot, and whether "regions equiprobable" holds for `p`.

use llvq_bench::f1::rank::{pack, rank_of, RankTable};
use llvq_bench::f1::{Prepared, SectionSet, Trellis, SECTION};
use llvq_core::SplitMix64;
use std::collections::HashSet;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// The rank-space table: `f1::rank`, held here as membership sets
// ---------------------------------------------------------------------------

/// The rows of [`RankTable`] as the sets the encoder's membership test needs.
/// The construction lives in `llvq_bench::f1::rank`, where the CUDA decoder is
/// checked against it; this is only its rows, hashed.
pub struct Membership {
    /// Per parity class: the 2048 lowest-cost rank vectors.
    class: [HashSet<u32>; 2],
    /// The 2048 lowest-cost overall.
    mixed: HashSet<u32>,
}

impl Membership {
    fn new(t: &RankTable) -> Self {
        Self {
            class: [t.class_rows(0).iter().copied().collect(), t.class_rows(1).iter().copied().collect()],
            mixed: t.mixed_rows().into_iter().collect(),
        }
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
    table: Arc<Membership>,
    mode: Mode,
    fallback: [i32; SECTION],
}

impl RankRegion {
    fn new(set: SectionSet, table: Arc<Membership>, mode: Mode) -> Self {
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

/// `-- enc`: how exact the candidate search is on a rank region, against
/// exhaustive search over the region — the same control `examples/f1enc.rs`
/// runs on the exact regions. Where it is not exact it can only return a point
/// farther away, so the bench's loss for the universal table is an upper bound.
fn encoder_exactness(table: Arc<Membership>) {
    let t = Trellis::new();
    let mut rng = SplitMix64::new(0x5f1b_2026_0904);
    println!("{:<26} {:>7} {:>9} {:>10}   (exact region, same targets)", "région de rangs", "exact", "excès moy", "excès max");
    for (label, set, mode, w) in [
        ("extrême p0 r0 (w=12)", t.section1(2, 0, 0), Mode::End(0), 12u32),
        ("extrême p1 r1 (w=12)", t.section1(31, 1, 1), Mode::End(1), 12),
        ("milieu p0 (w=15)", t.section2(2, 0), Mode::Mid, 15),
        ("milieu p1 (w=15)", t.section2(2, 1), Mode::Mid, 15),
    ] {
        let reg = RankRegion::new(set.clone(), table.clone(), mode);
        let region: Vec<[i32; SECTION]> = set.enumerate_below(200).into_iter().filter(|y| reg.contains(y)).collect();
        assert_eq!(region.len(), 1 << w);
        let prep = Prepared::new(set.clone(), w);
        let exact_region = set.region_points(w);
        let n = 400usize;
        let mut stats = [(0usize, 0.0f64, 0.0f64); 2];
        for _ in 0..n {
            let mut x = [0.0f64; SECTION];
            for v in x.iter_mut() {
                let (u, z): (f64, f64) = (rng.next_f64(), rng.next_f64());
                *v = 3.0 * (-2.0f64 * (u + 1e-12).ln()).sqrt() * (std::f64::consts::TAU * z).cos();
            }
            let dist = |y: &[i32; SECTION]| y.iter().zip(&x).map(|(&v, &e)| (v as f64 - e).powi(2)).sum::<f64>();
            for (k, (pts, got)) in [(&region, Region::nearest(&reg, &x).1), (&exact_region, Region::nearest(&prep, &x).1)].into_iter().enumerate() {
                let truth = pts.iter().map(dist).fold(f64::INFINITY, f64::min);
                assert!(got >= truth - 1e-9, "found a point closer than the region's best");
                if got <= truth + 1e-9 {
                    stats[k].0 += 1;
                } else {
                    let e = got / truth - 1.0;
                    stats[k].1 += e;
                    stats[k].2 = stats[k].2.max(e);
                }
            }
        }
        let fmt = |s: &(usize, f64, f64)| {
            let miss = n - s.0;
            format!(
                "{:>6.1}% {:>9} {:>10}",
                100.0 * s.0 as f64 / n as f64,
                if miss == 0 { "—".into() } else { format!("{:.4}", s.1 / miss as f64) },
                if miss == 0 { "—".into() } else { format!("{:.4}", s.2) }
            )
        };
        println!("{label:<26} {}   ({})", fmt(&stats[0]), fmt(&stats[1]));
    }
}

fn main() {
    encoder_exactness(Arc::new(Membership::new(&RankTable::build())));
}
