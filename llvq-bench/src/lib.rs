//! # llvq-bench — Gaussian-source evaluation (paper §4, gate G4)
//!
//! Quantizes i.i.d. `N(0, 1)` blocks of 24 weights with the LLVQ codebooks
//! available so far (union of shells m ≤ 3, ≈ 1.0 bit/dim) and measures
//! MSE / SQNR / Shannon retention, following the protocol of §4 of the
//! paper (arXiv:2603.11021): `MSE*(R) = 2^(−2R)`, `SQNR_bits = −½·log2 MSE`,
//! `Ret = SQNR_bits / R`.
//!
//! The paper's Table 3 reference numbers (89.14 % / 92.11 % retention) are
//! at 2 bits/dim, which requires shells up to m = 13 (Phase 2b); this crate
//! produces the **preliminary point at ≈ 1 bit/dim** — the first quality
//! figure comparable to the paper's Figure 1, with no LLM involved.
//!
//! Per-block search results are scale-independent: for each block the two
//! per-shell max dot products `(d₂, d₃)` are computed once, then
//!
//! * **spherical shaping** at scale β minimizes
//!   `‖x‖² + β²·16m − 2β·d_m` over {origin, shell 2, shell 3} — the β sweep
//!   costs nothing beyond the initial search pass;
//! * **shape–gain** uses the angular winner `t = max(d₂/√32, d₃/√48)`
//!   (the projection `⟨x, v̂⟩`), and the paper's scale-correction (App. E)
//!   makes `t` exactly the scalar the gain quantizer must encode:
//!   `min_g ‖x − g·v̂‖²` is attained at `g = t`, so
//!   `MSE = ‖x‖² − 2ĝt + ĝ²` with `ĝ = Q_gain(t)` (Lloyd–Max centroids).

#![forbid(unsafe_code)]

use llvq_core::{SplitMix64, DIM};
use llvq_search::{Searcher, Workspace};

/// Norms of the two shells in the integer embedding.
const N2: f64 = 32.0;
const N3: f64 = 48.0;

/// Scale-independent per-block search summary.
#[derive(Clone, Copy, Debug)]
pub struct BlockDots {
    /// `‖x‖²`.
    pub xx: f64,
    /// `max_{v ∈ Shell(2)} ⟨x, v⟩`.
    pub d2: f64,
    /// `max_{v ∈ Shell(3)} ⟨x, v⟩`.
    pub d3: f64,
}

impl BlockDots {
    /// Best projection `⟨x, v̂⟩` over the union's unit directions.
    #[inline]
    pub fn t(&self) -> f64 {
        (self.d2 / N2.sqrt()).max(self.d3 / N3.sqrt())
    }

    /// Squared quantization error of spherical shaping at scale `beta`
    /// (codebook `β·(Λ₂₄(3) ∪ {0})`).
    #[inline]
    pub fn spherical_err2(&self, beta: f64) -> f64 {
        let e2 = self.xx + beta * beta * N2 - 2.0 * beta * self.d2;
        let e3 = self.xx + beta * beta * N3 - 2.0 * beta * self.d3;
        e2.min(e3).min(self.xx) // ‖x − 0‖² = ‖x‖² for the origin codeword
    }
}

/// Draw a Gaussian block.
pub fn gauss_block(rng: &mut SplitMix64) -> [f64; DIM] {
    core::array::from_fn(|_| rng.next_gaussian())
}

/// One exact search pass per block (both shells), parallelized across
/// threads with one [`Workspace`] each.
pub fn precompute(s: &Searcher, xs: &[[f64; DIM]]) -> Vec<BlockDots> {
    let nt = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(xs.len().max(1));
    let chunk = xs.len().div_ceil(nt);
    let mut out = Vec::with_capacity(xs.len());
    std::thread::scope(|sc| {
        let handles: Vec<_> = xs
            .chunks(chunk)
            .map(|ch| {
                sc.spawn(move || {
                    let mut ws = Workspace::new();
                    ch.iter()
                        .map(|x| {
                            let f2 = s.nearest_in_shell2_with(&mut ws, x);
                            let f3 = s.nearest_in_shell3_with(&mut ws, x);
                            BlockDots {
                                xx: x.iter().map(|v| v * v).sum(),
                                d2: f2.dot,
                                d3: f3.dot,
                            }
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        for h in handles {
            out.extend(h.join().expect("search thread panicked"));
        }
    });
    out
}

/// Mean squared error **per weight** of spherical shaping at scale `beta`.
pub fn spherical_mse(dots: &[BlockDots], beta: f64) -> f64 {
    dots.iter().map(|d| d.spherical_err2(beta)).sum::<f64>() / (DIM * dots.len()) as f64
}

/// Pick the best scale on a training set by grid sweep (the per-block search
/// is scale-independent, so this is a pure post-processing loop).
pub fn optimize_beta(train: &[BlockDots], lo: f64, hi: f64, steps: usize) -> f64 {
    let mut best = (f64::INFINITY, lo);
    for k in 0..=steps {
        let beta = lo + (hi - lo) * k as f64 / steps as f64;
        let mse = spherical_mse(train, beta);
        if mse < best.0 {
            best = (mse, beta);
        }
    }
    best.1
}

/// Lloyd–Max (1-D k-means) on scalar samples; `k_bits = 0` yields the mean.
/// Returns sorted centroids.
pub fn lloyd_max(samples: &[f64], k_bits: u32, iters: usize) -> Vec<f64> {
    assert!(!samples.is_empty());
    let k = 1usize << k_bits;
    let mut sorted = samples.to_vec();
    sorted.sort_unstable_by(f64::total_cmp);
    // Quantile initialization.
    let mut centroids: Vec<f64> = (0..k)
        .map(|j| sorted[(2 * j + 1) * sorted.len() / (2 * k)])
        .collect();
    for _ in 0..iters {
        let mut sums = vec![0.0f64; k];
        let mut counts = vec![0u64; k];
        for &t in samples {
            let j = nearest_centroid(&centroids, t);
            sums[j] += t;
            counts[j] += 1;
        }
        for j in 0..k {
            if counts[j] > 0 {
                centroids[j] = sums[j] / counts[j] as f64;
            }
        }
        centroids.sort_unstable_by(f64::total_cmp);
    }
    centroids
}

/// Index of the nearest centroid (centroids sorted; k ≤ 4 → linear scan).
#[inline]
pub fn nearest_centroid(centroids: &[f64], t: f64) -> usize {
    let mut best = (f64::INFINITY, 0usize);
    for (j, &c) in centroids.iter().enumerate() {
        let d = (t - c).abs();
        if d < best.0 {
            best = (d, j);
        }
    }
    best.1
}

/// Mean squared error per weight of shape–gain with the given gain
/// centroids: `‖x − ĝ·v̂‖² = ‖x‖² − 2ĝt + ĝ²`.
pub fn shape_gain_mse(dots: &[BlockDots], centroids: &[f64]) -> f64 {
    dots.iter()
        .map(|d| {
            let t = d.t();
            let g = centroids[nearest_centroid(centroids, t)];
            d.xx - 2.0 * g * t + g * g
        })
        .sum::<f64>()
        / (DIM * dots.len()) as f64
}

/// Bits/dim of the spherical-shaping codebook (union incl. origin).
pub fn rate_spherical() -> f64 {
    let n = llvq_core::leech::THETA[0].1 + llvq_core::leech::THETA[1].1 + 1;
    (n as f64).log2() / DIM as f64
}

/// Bits/dim of shape–gain: union directions + `k_bits` of gain per block.
pub fn rate_shape_gain(k_bits: u32) -> f64 {
    let n = llvq_core::leech::THETA[0].1 + llvq_core::leech::THETA[1].1;
    ((n as f64).log2() + k_bits as f64) / DIM as f64
}

/// `SQNR` in bits for a unit-variance source: `−½·log2(MSE)`.
pub fn sqnr_bits(mse: f64) -> f64 {
    -0.5 * mse.log2()
}

/// Shannon retention in percent: `100 · SQNR_bits / R`.
pub fn retention_pct(mse: f64, rate: f64) -> f64 {
    100.0 * sqnr_bits(mse) / rate
}

/// Optimal 1-bit scalar quantizer MSE for `N(0,1)` (Lloyd–Max levels
/// `±√(2/π)`): `1 − 2/π` — the analytic scalar baseline at 1 bit/dim.
pub fn lloyd_max_1bit_scalar_mse() -> f64 {
    1.0 - core::f64::consts::FRAC_2_PI
}

// ---------------------------------------------------------------------------
// Full ball-cut Λ₂₄(13) — the 2 bit/dim regime (gate G4, paper Table 3)
// ---------------------------------------------------------------------------

use llvq_search::generic::{BallSearcher, NSHELLS};

/// Scale-independent per-block search summary over all shells m = 2..=13.
#[derive(Clone, Copy, Debug)]
pub struct BlockDots13 {
    pub xx: f64,
    /// `max_{v ∈ Shell(m)} ⟨x, v⟩` for m = 2..=13 (index m − 2).
    pub d: [f64; NSHELLS],
}

impl BlockDots13 {
    /// Best projection `⟨x, v̂⟩` over the ball's unit directions.
    #[inline]
    pub fn t(&self) -> f64 {
        self.d
            .iter()
            .enumerate()
            .map(|(i, &d)| d / ((16 * (i + 2)) as f64).sqrt())
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// Squared error of spherical shaping at scale `beta`
    /// (codebook `β·(Λ₂₄(13) ∪ {0})`).
    #[inline]
    pub fn spherical_err2(&self, beta: f64) -> f64 {
        let mut e = self.xx; // origin codeword
        for (i, &d) in self.d.iter().enumerate() {
            let n = (16 * (i + 2)) as f64;
            e = e.min(self.xx + beta * beta * n - 2.0 * beta * d);
        }
        e
    }
}

/// One generic search pass per block over all 12 shells, parallelized.
pub fn precompute13(s: &Searcher, xs: &[[f64; DIM]]) -> Vec<BlockDots13> {
    let nt = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(xs.len().max(1));
    let chunk = xs.len().div_ceil(nt);
    let mut out = Vec::with_capacity(xs.len());
    std::thread::scope(|sc| {
        let handles: Vec<_> = xs
            .chunks(chunk)
            .map(|ch| {
                sc.spawn(move || {
                    let mut ws = Workspace::new();
                    let mut ball = BallSearcher::new();
                    ch.iter()
                        .map(|x| {
                            let bests = ball.shell_bests(s, &mut ws, x);
                            BlockDots13 {
                                xx: x.iter().map(|v| v * v).sum(),
                                d: core::array::from_fn(|i| bests[i].dot),
                            }
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        for h in handles {
            out.extend(h.join().expect("search thread panicked"));
        }
    });
    out
}

/// Mean squared error per weight of ball-13 spherical shaping at `beta`.
pub fn spherical_mse13(dots: &[BlockDots13], beta: f64) -> f64 {
    dots.iter().map(|d| d.spherical_err2(beta)).sum::<f64>() / (DIM * dots.len()) as f64
}

/// Best scale by grid sweep on a training set.
pub fn optimize_beta13(train: &[BlockDots13], lo: f64, hi: f64, steps: usize) -> f64 {
    let mut best = (f64::INFINITY, lo);
    for k in 0..=steps {
        let beta = lo + (hi - lo) * k as f64 / steps as f64;
        let mse = spherical_mse13(train, beta);
        if mse < best.0 {
            best = (mse, beta);
        }
    }
    best.1
}

/// Shape–gain MSE per weight over the ball's directions.
pub fn shape_gain_mse13(dots: &[BlockDots13], centroids: &[f64]) -> f64 {
    dots.iter()
        .map(|d| {
            let t = d.t();
            let g = centroids[nearest_centroid(centroids, t)];
            d.xx - 2.0 * g * t + g * g
        })
        .sum::<f64>()
        / (DIM * dots.len()) as f64
}

/// Bits/dim of the ball-13 spherical codebook (incl. origin): ≈ 2.0000.
pub fn rate_spherical13() -> f64 {
    ((llvq_core::leech::N_SHELL_13_CUMULATIVE + 1) as f64).log2() / DIM as f64
}

/// Bits/dim of ball-13 shape–gain with `k_bits` of gain per block.
pub fn rate_shape_gain13(k_bits: u32) -> f64 {
    ((llvq_core::leech::N_SHELL_13_CUMULATIVE as f64).log2() + k_bits as f64) / DIM as f64
}
