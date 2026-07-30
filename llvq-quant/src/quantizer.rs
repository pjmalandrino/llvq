//! The block quantizers the GPTQ loop calls.
//!
//! One block is 24 consecutive **input channels** of one output row — the
//! `v ← W̃_{i,Q}` of Algorithm 1. A quantizer turns that block into its
//! reconstruction; everything about shells, classes and indices stays inside
//! [`llvq_search`].

use llvq_core::DIM;
use llvq_search::generic::BallSearcher;
use llvq_search::Searcher;

/// Reconstructs a block of weights from its quantized representation.
///
/// The contract is deliberately narrow — in, out, same length — so the GPTQ
/// loop is testable against quantizers that have nothing to do with the Leech
/// lattice (identity, coarse scalar rounding), which is how its error
/// feedback gets pinned independently of the codebook.
pub trait BlockQuantizer {
    /// Block length this quantizer accepts, in weights.
    fn block_len(&self) -> usize;

    /// Announce the scale that the following blocks are relative to.
    ///
    /// Block magnitudes vary by orders of magnitude *across* the rows of a
    /// weight matrix but are homogeneous *within* one, so a gain code with a
    /// handful of levels is only workable relative to a per-row scale. That
    /// scale is one float per row — `16/d_in` bits per weight, i.e. 0.006 on
    /// a 2560-wide layer — against `16/24 = 0.67` for an unquantized
    /// magnitude per block. Ignored by quantizers that carry no gain.
    fn set_row_scale(&mut self, _scale: f64) {}

    /// Write the reconstruction of `v` into `out` (same length as `v`).
    fn quantize(&mut self, v: &[f64], out: &mut [f64]);
}

/// Shape only, with the block magnitude kept in **full precision**.
///
/// `Q(v) = ‖v‖₂ · Q_dir(v/‖v‖₂)`, where `Q_dir` is the exact angular search
/// over the normalized ball `Λ₂₄(13)`. Norm-preserving by construction, so
/// the spherical retraction of Algorithm 3 is a no-op on top of it.
///
/// ⚠️ **This is not a 2 bit/weight quantizer, despite the direction costing
/// exactly 48 bits per 24 weights.** The magnitude it keeps is an
/// unquantized float per block — another 16 bits if stored in half
/// precision, which is `16/24 = 0.67` bits per weight. "Zero gain bits" in
/// the paper means a *single constant* for the whole tensor, not a free
/// float per block. Use [`LeechShapeGain`] for an honest rate; this type is
/// the upper bound on what the direction code alone can achieve, and is
/// useful as a control.
pub struct LeechDirection {
    searcher: Searcher,
    ball: BallSearcher,
}

impl LeechDirection {
    pub fn new() -> Self {
        Self {
            searcher: Searcher::new(),
            ball: BallSearcher::new(),
        }
    }
}

impl Default for LeechDirection {
    fn default() -> Self {
        Self::new()
    }
}

impl BlockQuantizer for LeechDirection {
    fn block_len(&self) -> usize {
        DIM
    }

    fn quantize(&mut self, v: &[f64], out: &mut [f64]) {
        let x: &[f64; DIM] = v.try_into().expect("block must be 24 weights");
        let norm = x.iter().map(|a| a * a).sum::<f64>().sqrt();
        if norm == 0.0 {
            out.fill(0.0);
            return;
        }
        let f = self.ball.nearest_angular(&self.searcher, x);
        // f.point lies on shell `f.shell`, so ‖f.point‖ = √(16·shell).
        let scale = norm / ((16 * f.shell) as f64).sqrt();
        for (o, &p) in out.iter_mut().zip(f.point.iter()) {
            *o = p as f64 * scale;
        }
    }
}

/// Spherical shaping: the nearest point of the ball `β·(Λ₂₄(13) ∪ {0})`
/// under the Euclidean metric, magnitude included.
///
/// This is the variant the paper's Table 6 shows **losing to QTIP** on
/// Qwen3-4B (21.80 vs 17.04 perplexity); it is kept because Appendix I's
/// ablation needs it — it is the code whose radial drift Spherical GPTQ is
/// there to fix, and the contrast only shows up against it.
pub struct LeechBall {
    searcher: Searcher,
    ball: BallSearcher,
    beta: f64,
}

impl LeechBall {
    pub fn new(beta: f64) -> Self {
        assert!(beta > 0.0, "scale must be positive");
        Self {
            searcher: Searcher::new(),
            ball: BallSearcher::new(),
            beta,
        }
    }
}

impl BlockQuantizer for LeechBall {
    fn block_len(&self) -> usize {
        DIM
    }

    fn quantize(&mut self, v: &[f64], out: &mut [f64]) {
        let x: &[f64; DIM] = v.try_into().expect("block must be 24 weights");
        let f = self.ball.nearest_scaled(&self.searcher, x, self.beta);
        for (o, &p) in out.iter_mut().zip(f.point.iter()) {
            *o = p as f64 * self.beta;
        }
    }
}

/// Reconstructs its input exactly. Only useful to assert that the GPTQ loop
/// is a no-op when nothing is lost — which catches sign and index errors in
/// the error feedback that no approximate quantizer would reveal.
pub struct Identity {
    pub block: usize,
}

impl BlockQuantizer for Identity {
    fn block_len(&self) -> usize {
        self.block
    }

    fn quantize(&mut self, v: &[f64], out: &mut [f64]) {
        out.copy_from_slice(v);
    }
}

/// Round-to-nearest on a fixed grid. The baseline GPTQ has to beat.
pub struct ScalarGrid {
    pub block: usize,
    pub step: f64,
}

impl BlockQuantizer for ScalarGrid {
    fn block_len(&self) -> usize {
        self.block
    }

    fn quantize(&mut self, v: &[f64], out: &mut [f64]) {
        for (o, &a) in out.iter_mut().zip(v.iter()) {
            *o = (a / self.step).round() * self.step;
        }
    }
}

/// Shape–gain: the direction from the Leech ball, the magnitude from a
/// `k`-bit scalar code relative to a per-row scale.
///
/// This is the configuration the paper's Table 8 measures as its best on a
/// Gaussian source — `norm(Λ₂₄(12))` with **one** gain bit reaches 92.14 %
/// retention at exactly 2.000 bits/dim, ahead of the zero-gain-bit variant
/// (89.12 %). High-resolution theory suggests spending `1/n` of the budget
/// on the radius, which for n = 24 would be two bits; the paper's own sweep
/// finds one to be better, and says so.
///
/// Rate: `(48 + k)` bits per 24 weights, plus one row scale per output row.
pub struct LeechShapeGain {
    searcher: Searcher,
    ball: BallSearcher,
    /// Gain levels, relative to the row scale, ascending.
    centroids: Vec<f64>,
    row_scale: f64,
}

impl LeechShapeGain {
    /// `centroids` are gains **relative to the row scale**, as fitted by
    /// [`fit_gain_centroids`].
    pub fn new(centroids: Vec<f64>) -> Self {
        Self::with_shell_cap(centroids, llvq_search::classes::MAX_SHELL)
    }

    /// Restrict the direction code to `Λ₂₄(cap)`.
    ///
    /// `cap = 12` drops the index from 48 to 47 bits, which pays for a gain
    /// bit at the same total rate — the paper's Table 8 best configuration,
    /// `norm(Λ₂₄(12))` + 1 gain bit.
    pub fn with_shell_cap(centroids: Vec<f64>, cap: u32) -> Self {
        assert!(!centroids.is_empty(), "need at least one gain level");
        let mut ball = BallSearcher::new();
        ball.set_shell_cap(cap);
        Self {
            searcher: Searcher::new(),
            ball,
            centroids,
            row_scale: 1.0,
        }
    }

    /// Bits spent per block on the gain.
    pub fn gain_bits(&self) -> u32 {
        self.centroids.len().next_power_of_two().trailing_zeros()
    }
}

impl BlockQuantizer for LeechShapeGain {
    fn block_len(&self) -> usize {
        DIM
    }

    fn set_row_scale(&mut self, scale: f64) {
        self.row_scale = if scale > 0.0 { scale } else { 1.0 };
    }

    fn quantize(&mut self, v: &[f64], out: &mut [f64]) {
        let x: &[f64; DIM] = v.try_into().expect("block must be 24 weights");
        let norm = x.iter().map(|a| a * a).sum::<f64>().sqrt();
        if norm == 0.0 {
            out.fill(0.0);
            return;
        }
        let f = self.ball.nearest_angular(&self.searcher, x);
        // The gain code sees the block norm relative to its row, which is
        // what makes a two-level code meaningful at all.
        let g = norm / self.row_scale;
        let picked = nearest_level(&self.centroids, g) * self.row_scale;
        let scale = picked / ((16 * f.shell) as f64).sqrt();
        for (o, &p) in out.iter_mut().zip(f.point.iter()) {
            *o = p as f64 * scale;
        }
    }
}

#[inline]
fn nearest_level(levels: &[f64], g: f64) -> f64 {
    let mut best = (f64::INFINITY, levels[0]);
    for &c in levels {
        let d = (g - c).abs();
        if d < best.0 {
            best = (d, c);
        }
    }
    best.1
}

/// Lloyd–Max on the relative block gains of a weight matrix.
///
/// `k_bits = 0` collapses to a single level — the paper's true "zero gain
/// bits" — and larger `k` splits it. Fitting on the matrix being quantized
/// rather than on an assumed χ distribution costs one cheap scan and copes
/// with weight distributions that are not Gaussian.
pub fn fit_gain_centroids(
    w: &[f64],
    d_out: usize,
    d_in: usize,
    block: usize,
    k_bits: u32,
    iters: usize,
) -> Vec<f64> {
    let mut g = Vec::with_capacity(d_out * (d_in / block));
    for i in 0..d_out {
        let row = &w[i * d_in..(i + 1) * d_in];
        let scale = row_scale(row);
        if scale <= 0.0 {
            continue;
        }
        for b in row.chunks_exact(block) {
            let n = b.iter().map(|a| a * a).sum::<f64>().sqrt();
            if n > 0.0 {
                g.push(n / scale);
            }
        }
    }
    if g.is_empty() {
        return vec![1.0];
    }
    let k = 1usize << k_bits;
    g.sort_unstable_by(f64::total_cmp);
    let mut c: Vec<f64> = (0..k).map(|j| g[(2 * j + 1) * g.len() / (2 * k)]).collect();
    for _ in 0..iters {
        let mut sums = vec![0.0f64; k];
        let mut counts = vec![0u64; k];
        for &v in &g {
            let j = (0..k)
                .min_by(|&a, &b| (v - c[a]).abs().total_cmp(&(v - c[b]).abs()))
                .expect("k > 0");
            sums[j] += v;
            counts[j] += 1;
        }
        for j in 0..k {
            if counts[j] > 0 {
                c[j] = sums[j] / counts[j] as f64;
            }
        }
        c.sort_unstable_by(f64::total_cmp);
    }
    c
}

/// The scale a row's gains are expressed against: the RMS block norm.
pub fn row_scale(row: &[f64]) -> f64 {
    let n = row.len();
    if n == 0 {
        return 0.0;
    }
    (row.iter().map(|a| a * a).sum::<f64>() / (n / DIM).max(1) as f64).sqrt()
}

/// Bits needed to index `Λ₂₄(cap) ∪ {0}` — `⌈log₂(N(cap) + 1)⌉`.
///
/// 48 for the full ball, 47 for `cap = 12`. That one bit is what makes a
/// gain bit free at 2.000 bits/dim.
pub fn index_bits(cap: u32) -> u32 {
    let cs = llvq_search::classes::enumerate_classes(cap);
    let n: u64 = (2..=cap).map(|m| cs.shell_cardinality(m)).sum();
    (128 - (n as u128).leading_zeros()).max(1)
}
