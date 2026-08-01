//! The block quantizers the GPTQ loop calls.
//!
//! One block is 24 consecutive **input channels** of one output row — the
//! `v ← W̃_{i,Q}` of Algorithm 1. A quantizer turns that block into its
//! reconstruction; everything about shells, classes and indices stays inside
//! [`llvq_search`].

use llvq_core::DIM;
use llvq_search::generic::BallSearcher;
use llvq_search::Searcher;

/// What a quantizer emitted for the block it just reconstructed — everything
/// an artifact needs to rebuild that block, and nothing else.
///
/// The lattice point rather than its 48-bit index: bijective indexing costs a
/// multiset-permutation rank per block, which has no business running inside
/// the quantization loop. A layer's points are encoded once, when the layer is
/// written out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockCode {
    /// Direction, as a point of `√8·Λ₂₄ ⊂ Z²⁴`.
    pub point: llvq_core::Point,
    /// Rank of the gain level within the matrix's fitted centroids.
    pub gain: u32,
}

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

    /// The code the last [`Self::quantize`] call emitted, for quantizers that
    /// have one — the lattice point and the gain level rank.
    ///
    /// This is what an artifact writes. Quantizers with no discrete code
    /// (identity, scalar rounding) return `None` and simply cannot be
    /// serialized at a compressed rate.
    ///
    /// **The code describes what `quantize` produced, not necessarily what the
    /// layer loop keeps.** A retraction that moves the block off the code's
    /// own magnitude invalidates it; nothing here can detect that, so the
    /// guarantee comes from the round-trip test instead — decode the artifact
    /// and demand the evaluated weights back, bit for bit.
    fn last_code(&self) -> Option<BlockCode> {
        None
    }

    /// The sphere the retraction of Algorithm 3 should put this block back on,
    /// given the block's norm before quantization.
    ///
    /// **Which sphere is not a detail.** Rescaling to the block's own
    /// pre-quantization norm hands back a free float per block, which silently
    /// cancels whatever the gain code just chose: two disjoint gain codebooks
    /// then produce bit-identical weights, and the reconstruction costs 16
    /// bits per block that no rate accounting charges. A quantizer that codes
    /// its magnitude must name a sphere **its code can express**.
    ///
    /// The default is the input norm, which is exactly right for quantizers
    /// whose magnitude is free by construction ([`LeechDirection`]) or which
    /// carry no notion of block magnitude at all ([`Identity`],
    /// [`ScalarGrid`]).
    ///
    /// `None` means *"my output is already on the sphere it should be on —
    /// do not touch it."* That is not the same as retracting to the norm the
    /// quantizer just produced: rescaling by a computed `k ≈ 1` perturbs the
    /// last bits, and a decoder rebuilding the block from its code alone
    /// cannot reproduce that perturbation. An artifact can only be bit-exact
    /// if the loop leaves such blocks alone.
    fn retraction_target(&self, norm_before: f64) -> Option<f64> {
        Some(norm_before)
    }
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
///
/// ## An unresolved tension with the spherical retraction
///
/// Eq. 17 as written retracts a block back to `‖W_{i,B}‖₂` — its exact
/// pre-quantization norm. Composed with a gain code, that **cancels the code
/// outright**: the level chosen here is overwritten, and the stored magnitude
/// is a free float per block (16 bits nothing charges for). `g5_retraction.rs`
/// pins it — two disjoint codebooks then give bit-identical weights.
///
/// The paper's own resolution is that magnitudes are held "by the norm
/// constraint during GPTQ **and then by a closed-form solve at the end of the
/// layer**" — Algorithm 3's `refine_group_scales`, which we keep disabled
/// because it degraded perplexity across 28 blocks.
///
/// So there are three coherent designs and we have not measured which is
/// right, only which is *honest*:
///
/// | | retraction | rate/block | gain code |
/// |---|---|---|---|
/// | [`Self::retract_to_level`] (default) | to the nearest code level | 47 + k | load-bearing |
/// | [`Self::with_free_magnitude`] | to the exact block norm | 47 + 16 | inert |
/// | Algorithm 3 as written | exact norm, then a closed-form solve | to be determined | deferred to the solve |
///
/// The default is the honest one: what the code claims to store is what the
/// reconstruction uses. The alternative is kept so the two can be A/B'd
/// rather than argued about — the choice is a measurement, not a preference.
pub struct LeechShapeGain {
    searcher: Searcher,
    ball: BallSearcher,
    /// Gain levels, relative to the row scale, ascending.
    centroids: Vec<f64>,
    row_scale: f64,
    /// Retract to a level the gain code can express, rather than to the
    /// block's own norm.
    retract_to_level: bool,
    /// Code emitted by the most recent `quantize`, for artifact writing.
    last: Option<BlockCode>,
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
        Self::with_caps(centroids, cap, llvq_search::generic::MAX_LEVELS_ANY)
    }

    /// Restrict the direction code by shell **and** by the number of distinct
    /// magnitudes a block may hold, zero included.
    ///
    /// The level cap is a memory knob, not a quality one: the fused kernel's
    /// runtime layout spends `34 + 24(L−1)` bits per block, so L *is* what it
    /// costs in RAM. Capping at 4 drops a bit per weight for 0.25 points of
    /// retention on a Gaussian source; capping at 3 drops two bits — under
    /// 4-bit quantization's 4.50 — for 2.6 points (`llvq-bench --bin lcap`).
    /// Whether real weights behave like that Gaussian is the whole question,
    /// and only a run answers it.
    pub fn with_caps(centroids: Vec<f64>, cap: u32, level_cap: usize) -> Self {
        assert!(!centroids.is_empty(), "need at least one gain level");
        let mut ball = BallSearcher::with_level_cap(level_cap);
        ball.set_shell_cap(cap);
        Self {
            searcher: Searcher::new(),
            ball,
            centroids,
            row_scale: 1.0,
            retract_to_level: true,
            last: None,
        }
    }

    /// Let the retraction restore the block's exact norm, as Eq. 17 is
    /// written — which makes the gain code inert and the true rate `47 + 16`
    /// bits per block rather than `47 + k`.
    ///
    /// This is what every run before 2026-07-31 did, unknowingly. Kept only so
    /// the two can be measured against each other; see the type's note.
    pub fn with_free_magnitude(mut self) -> Self {
        self.retract_to_level = false;
        self
    }

    /// Bits spent per block on the gain.
    pub fn gain_bits(&self) -> u32 {
        self.centroids.len().next_power_of_two().trailing_zeros()
    }

    /// Rebuild a block from its code alone — the decoder side of an artifact.
    ///
    /// This must mirror [`BlockQuantizer::quantize`] operation for operation,
    /// not merely agree with it mathematically: a round-trip that is only
    /// correct to a few ulps is not a round-trip, and the difference between
    /// the two is exactly the kind of thing that shows up as an unexplained
    /// perplexity three hours into a run.
    pub fn reconstruct(&self, code: &BlockCode, row_scale: f64, out: &mut [f64]) {
        assert_eq!(out.len(), DIM);
        match llvq_core::Leech::shell_index(&code.point) {
            Some(m) if m > 0 => {
                let picked = self.centroids[code.gain as usize] * row_scale;
                let scale = picked / ((16 * m) as f64).sqrt();
                for (o, &p) in out.iter_mut().zip(code.point.iter()) {
                    *o = p as f64 * scale;
                }
            }
            // The origin: a zero block, representable and reconstructed as is.
            _ => out.fill(0.0),
        }
    }

    /// Bits per block this quantizer actually costs, given how it retracts.
    ///
    /// The point of having this on the quantizer rather than in a table: a
    /// configuration cannot claim a rate its reconstruction does not honour.
    pub fn block_bits(&self) -> u32 {
        let index = index_bits(self.ball.shell_cap());
        if self.retract_to_level {
            index + self.gain_bits()
        } else {
            index + 16
        }
    }
}

impl BlockQuantizer for LeechShapeGain {
    fn block_len(&self) -> usize {
        DIM
    }

    fn set_row_scale(&mut self, scale: f64) {
        self.row_scale = if scale > 0.0 { scale } else { 1.0 };
    }

    /// The retraction must land on a magnitude this code can express, so it
    /// targets the nearest gain level rather than the block's own norm.
    ///
    /// Retracting to `norm_before` instead would restore a free float per
    /// block and cancel the gain code outright — see the trait's note and the
    /// type's note on the three coherent designs.
    fn retraction_target(&self, norm_before: f64) -> Option<f64> {
        if self.retract_to_level {
            // `quantize` already placed the block on the nearest level's
            // sphere. Rescaling it to a recomputed value of that same norm
            // would only cost a rounding error the decoder cannot mirror.
            debug_assert!(
                {
                    let want = nearest_level(&self.centroids, norm_before / self.row_scale);
                    want.is_finite()
                },
                "gain levels must be finite"
            );
            None
        } else {
            Some(norm_before)
        }
    }

    fn quantize(&mut self, v: &[f64], out: &mut [f64]) {
        let x: &[f64; DIM] = v.try_into().expect("block must be 24 weights");
        let norm = x.iter().map(|a| a * a).sum::<f64>().sqrt();
        if norm == 0.0 {
            out.fill(0.0);
            // The origin is a codebook point (index 0); a zero block is
            // representable, not an absence of code.
            self.last = Some(BlockCode {
                point: [0; DIM],
                gain: 0,
            });
            return;
        }
        let f = self.ball.nearest_angular(&self.searcher, x);
        // The gain code sees the block norm relative to its row, which is
        // what makes a two-level code meaningful at all.
        let g = norm / self.row_scale;
        let level = nearest_level_index(&self.centroids, g);
        let picked = self.centroids[level] * self.row_scale;
        let scale = picked / ((16 * f.shell) as f64).sqrt();
        for (o, &p) in out.iter_mut().zip(f.point.iter()) {
            *o = p as f64 * scale;
        }
        self.last = Some(BlockCode {
            point: f.point,
            gain: level as u32,
        });
    }

    fn last_code(&self) -> Option<BlockCode> {
        self.last
    }
}

#[inline]
fn nearest_level(levels: &[f64], g: f64) -> f64 {
    levels[nearest_level_index(levels, g)]
}

/// The **rank** of the nearest level, which is what gets written to disk.
///
/// The value alone cannot be stored: an artifact carries the level's index
/// and looks the value up in the matrix's fitted centroids.
#[inline]
fn nearest_level_index(levels: &[f64], g: f64) -> usize {
    let mut best = (f64::INFINITY, 0usize);
    for (i, &c) in levels.iter().enumerate() {
        let d = (g - c).abs();
        if d < best.0 {
            best = (d, i);
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
