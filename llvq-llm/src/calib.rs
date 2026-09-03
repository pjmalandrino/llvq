//! Layer Hessians, and the sequential block loop that consumes them.
//!
//! ## Why the loop is sequential
//!
//! GPTQ quantizes block `t` against the activations that actually reach it —
//! which means the activations that have already flowed through blocks
//! `0..t` **in their quantized form**. Calibrating every layer against the
//! FP16 model instead is simpler and measurably worse: each layer would be
//! corrected for an input distribution the deployed model never sees.
//!
//! So each block costs two forward passes over the calibration set: one with
//! the original weights to collect `H`, and one with the quantized weights to
//! produce the input of the next block.
//!
//! ## Where the Hessians come from
//!
//! `H = AᵀA/N` is accumulated as a matmul on the device — `n × n` for
//! `n` up to 3072, which is a GEMM, not something to loop in scalar Rust.
//! Accumulation is in f32 (as every GPTQ implementation does) with each
//! window pre-scaled by `1/N` so the running sum stays O(1) instead of
//! growing with the sample count; the conversion to f64 happens once, at the
//! factorization.
//!
//! ## What survives a factorization, and what does not
//!
//! `H` itself — as opposed to its factor — is read in exactly one place: the
//! end-of-layer closed-form scale solve. See [`needs_dense_hessian`].

use crate::model::{Act, Capture, Qwen3};
use candle_core::{DType, Device, Tensor};
use llvq_quant::gptq::{GptqConfig, Weights};
use llvq_quant::linalg::GptqFactor;
use llvq_quant::quantizer::{
    fit_gain_centroids, BlockQuantizer, Identity, LeechDirection, LeechShapeGain, ScalarGrid,
    ScalarGroupwise,
};
use llvq_quant::rotation::Rotation;
use std::collections::HashMap;

/// Running `AᵀA/N` for one activation.
pub struct Hessian {
    sum: Tensor,
    scale: f64,
}

impl Hessian {
    pub fn new(width: usize, device: &Device, total_rows: usize) -> candle_core::Result<Self> {
        Ok(Self {
            sum: Tensor::zeros((width, width), DType::F32, device)?,
            scale: 1.0 / total_rows as f64,
        })
    }

    /// `x` is `(batch, seq, width)`.
    pub fn accumulate(&mut self, x: &Tensor) -> candle_core::Result<()> {
        let w = x.dim(x.rank() - 1)?;
        let a = x.reshape(((), w))?.to_dtype(DType::F32)?.contiguous()?;
        let g = (a.t()?.contiguous()?.matmul(&a)? * self.scale)?;
        self.sum = (&self.sum + g)?;
        Ok(())
    }

    /// Dense row-major `n × n`, in f64, ready for [`GptqFactor`].
    pub fn to_f64(&self) -> candle_core::Result<Vec<f64>> {
        Ok(self
            .sum
            .flatten_all()?
            .to_vec1::<f32>()?
            .into_iter()
            .map(|v| v as f64)
            .collect())
    }
}

/// Collects the four Hessians of a single block.
struct BlockCapture {
    target: usize,
    acc: HashMap<Act, Hessian>,
}

impl Capture for BlockCapture {
    fn on_activation(&mut self, layer: usize, act: Act, x: &Tensor) -> candle_core::Result<()> {
        if layer == self.target {
            if let Some(h) = self.acc.get_mut(&act) {
                h.accumulate(x)?;
            }
        }
        Ok(())
    }
}

/// Where a run's wall clock went, in seconds per phase.
///
/// ## Why this exists
///
/// The first remote run made the point sharply. On this Mac, with Metal doing
/// the forward passes, the Leech encoder is ~59 % of a run. On a CPU-only x86
/// job — same code, same model — it is **12 %**, and the forward passes
/// dominate. Two opposite optimization targets, and no way to tell which one
/// applies to a given flavor without measuring it there.
///
/// That matters directly in money: hardware is billed by the minute, and the
/// gap between the right and the wrong flavor for a 32B run is a factor of
/// four. `CLAUDE.md` already carries the warning that the profiler has never
/// been used on this project; this is the cheap half of fixing that.
#[derive(Debug, Default, Clone)]
pub struct Phases {
    /// Pass 1 — forward through the block, accumulating `H = AᵀA/N`.
    pub capture: f64,
    /// `GptqFactor::new`, four per block. Cubic in the activation width, and
    /// measured at 1.5 % of a 4B run since `faer` — not the bottleneck it was.
    pub factor: f64,
    /// Weights out to `f64` and reconstructions back — the host↔device
    /// traffic, which a GPU run pays and a CPU run does not.
    pub transfer: f64,
    /// The GPTQ loop itself: Leech search, error feedback, retraction.
    pub quantize: f64,
    /// Packing lattice indices into the artifact.
    pub write: f64,
    /// Pass 2 — advance the activations through the *quantized* block.
    pub advance: f64,
}

impl Phases {
    pub fn total(&self) -> f64 {
        self.capture + self.factor + self.transfer + self.quantize + self.write + self.advance
    }

    /// `(name, seconds, percent)`, largest first — what to optimize, in order.
    pub fn ranked(&self) -> Vec<(&'static str, f64, f64)> {
        let total = self.total().max(1e-9);
        let mut v = vec![
            ("capture (pass 1)", self.capture, 0.0),
            ("factorization", self.factor, 0.0),
            ("f64 transfer", self.transfer, 0.0),
            ("quantization", self.quantize, 0.0),
            ("artifact write", self.write, 0.0),
            ("advance (pass 2)", self.advance, 0.0),
        ];
        for e in v.iter_mut() {
            e.2 = 100.0 * e.1 / total;
        }
        v.sort_by(|a, b| b.1.total_cmp(&a.1));
        v
    }
}

/// What a quantization run reports back.
#[derive(Debug, Default, Clone)]
pub struct Report {
    pub matrices: usize,
    pub weights: u64,
    /// Weights that stayed at full precision because they fell in a tail.
    pub tail_weights: u64,
    /// Output rows, each carrying one f16 scale.
    pub rows: u64,
    /// Bits per quantized block, gain included.
    pub block_bits: f64,
    /// Weights one block code covers — [`Codebook::block_len`]. Set from the
    /// codebook at the end of the run; [`Report::bits_per_weight`] refuses to
    /// compute a rate without it, because a wrong divisor here would report a
    /// rate no file has and nothing downstream would notice.
    pub block_len: usize,
    pub seconds: f64,
    /// Where the time went. See [`Phases`].
    pub phases: Phases,
    /// Bytes of **dense Hessian** held past its own factorization, for one
    /// transformer block — the largest such total over the blocks of the run.
    ///
    /// Zero on the published path, and that is the point: see
    /// [`needs_dense_hessian`]. It is reported rather than merely asserted
    /// because it is the one memory figure in this loop that is *exact* —
    /// `8·n²` per activation, no allocator, no estimate — and because a run
    /// that starts keeping them again should say so in its own log rather
    /// than be discovered by a machine that runs out of RAM.
    pub dense_hessian_bytes: u64,
}

impl Report {
    /// Effective bits per weight — **everything that would be written to
    /// disk**: the block code (index + gain), 16 bits for each full-precision
    /// tail weight, and one f16 scale per output row.
    ///
    /// Counting only the index is how a 2.73 bit/weight run got reported as
    /// 2.07; the magnitude has to be in here.
    pub fn bits_per_weight(&self) -> f64 {
        // `Report` derives `Default`, so this field can be zero — and a zero
        // divisor would hand back an infinite rate that reads as a bug
        // somewhere else entirely. It is a programming error, not a user one,
        // so it says which field and where it comes from.
        assert!(
            self.block_len > 0,
            "Report::block_len is zero: a rate cannot be computed without \
             knowing how many weights a block code covers. It is set from \
             Codebook::block_len at the end of a run; a Report built by hand \
             has to set it too."
        );
        let quantized = self.weights - self.tail_weights;
        let blocks = quantized as f64 / self.block_len as f64;
        (blocks * self.block_bits + self.tail_weights as f64 * 16.0 + self.rows as f64 * 16.0)
            / self.weights as f64
    }
}

/// Which codebook to quantize with.
///
/// An enum rather than a factory closure because one of them —
/// [`Codebook::ShapeGain`] — has to be **fitted to each matrix**: its gain
/// levels come from that matrix's own block magnitudes. A closure built once
/// for the whole run cannot do that.
#[derive(Clone, Copy, Debug)]
pub enum Codebook {
    /// Lossless. The control that proves the pipeline is a no-op when the
    /// codebook is.
    Identity,
    /// Round-to-nearest on a fixed grid.
    Grid { step: f64 },
    /// Affine INT-`bits` with a scale and a zero point per group of `group`
    /// input channels — the quantizer deployed pipelines ship, and the
    /// baseline a lattice code has to beat. See
    /// [`llvq_quant::quantizer::ScalarGroupwise`].
    ///
    /// It exists for one reason: a result shown only on Λ₂₄ is a statement
    /// about *our codebook*, and the same result shown on the field's own
    /// scalar quantizer is a statement about **GPTQ**.
    ///
    /// ⚠️ Its reconstruction is not describable by a `BlockCode`, so a run
    /// using it **cannot write an artifact** — `quantize_model_capturing`
    /// refuses it. Its perplexity is read off the in-memory model, which is
    /// why every arm it is compared against must be read the same way.
    ///
    /// ⚠️ [`Report::bits_per_weight`] charges one f16 per output row to every
    /// codebook, and this one carries no per-row scale — so its reported rate
    /// is high by `16/d_in` (≈ 0.006 b/weight at `d_in` = 2560). Declared
    /// rather than special-cased: what an A/B on this arm measures is a
    /// *variance*, not a rate.
    ScalarGroup { bits: u32, group: usize },
    /// Leech direction with the block magnitude left in full precision.
    /// **Not** a 2 bit/weight code — see `LeechDirection`.
    Direction,
    /// Leech direction plus a `gain_bits`-bit magnitude code, relative to a
    /// per-row scale. The honest 2 bit/weight configuration.
    ///
    /// `max_shell` restricts the direction code: 13 is the full ball (48-bit
    /// index), 12 costs 47 bits and pays for the gain bit at the same rate.
    ///
    /// `free_magnitude` restores the pre-2026-07-31 behaviour, where the
    /// spherical retraction put each block back on its exact norm and thereby
    /// cancelled the gain code — a free f16 per block. It is charged as such
    /// here. Kept only to A/B the honest configuration against it.
    ShapeGain {
        gain_bits: u32,
        max_shell: u32,
        free_magnitude: bool,
        /// Distinct magnitudes a block may hold, zero included. 5 excludes
        /// nothing; lower values shrink the codebook and, more to the point,
        /// the width of the runtime layout the fused kernel reads.
        level_cap: usize,
    },
}

impl Codebook {
    /// Bits per 24-weight block, gain included. The per-row scale is one f16
    /// per output row and is counted separately by [`Report`].
    pub fn block_bits(&self) -> f64 {
        match self {
            Codebook::Identity | Codebook::Grid { .. } => 24.0 * 16.0,
            Codebook::Direction => 48.0 + 16.0,
            // `bits` per weight, plus what a deployed INT-k file actually
            // stores for the group: an **f16 scale** and a zero point packed
            // at **`bits` width**, not a second f16. The zero is an integer in
            // `0..=2^bits−1` by construction (see `ScalarGroupwise`), which is
            // exactly why it packs that narrow — and charging it 16 bits, as
            // this line did before the upstream source was read, overstates
            // the rate by `(16−bits)/group`: 0.54 b/weight at `int3g24`.
            Codebook::ScalarGroup { bits, group } => {
                *group as f64 * *bits as f64 + 16.0 + *bits as f64
            }
            // A free per-block magnitude is an f16, and has to be charged as
            // one — claiming `gain_bits` while the retraction hands back a
            // float is exactly the accounting error of 2026-07-31.
            Codebook::ShapeGain {
                gain_bits,
                max_shell,
                free_magnitude,
                // The level cap shrinks the codebook — Λ₂₄(13) at L ≤ 3 needs
                // 46 index bits, not 48 — but the artifact still indexes over
                // the full ball, so the file pays 48. The saving is real and
                // claimable only once the indexer is rebuilt over the same
                // filter; charging it here would report a rate the file does
                // not have.
                level_cap: _,
            } => {
                let magnitude = if *free_magnitude { 16 } else { *gain_bits };
                (llvq_quant::quantizer::index_bits(*max_shell) + magnitude) as f64
            }
        }
    }

    /// Input channels one block code covers, and therefore what
    /// [`llvq_quant::gptq::GptqConfig::block`] must be set to.
    ///
    /// Λ₂₄ fixes it at 24 — a Leech block *is* 24 weights. The scalar arm
    /// carries its own group size instead, and the two numbers are one on
    /// purpose: the group and the error-feedback granularity have to move
    /// together, or an A/B between families changes two things at once.
    ///
    /// **One source of truth on purpose**, the same discipline as
    /// [`effective_rotation_seed`]: the site that configures the GPTQ loop
    /// and the site that divides weights into blocks to report a rate must
    /// not be able to disagree. They did not disagree while every codebook
    /// was 24 wide; the first one that is not would have made
    /// [`Report::bits_per_weight`] divide by the wrong number and report a
    /// rate no file has.
    pub fn block_len(&self) -> usize {
        match self {
            Codebook::ScalarGroup { group, .. } => *group,
            Codebook::Identity | Codebook::Grid { .. } | Codebook::Direction => llvq_core::DIM,
            Codebook::ShapeGain { .. } => llvq_core::DIM,
        }
    }
}

/// How to run [`quantize_model`].
pub struct RunConfig {
    pub gptq: GptqConfig,
    /// Hessian damping, relative to `mean(diag H)`.
    pub damping: f64,
    /// Off-diagonal shrinkage of the Hessian **estimate**, `ρ ∈ [0, 1]`:
    /// `H ← ρ·H + (1 − ρ)·diag(H)`, applied in the natural basis before the
    /// rotation. `1.0` is the published path — and not a multiply by one: the
    /// call is skipped, so the bytes of every shipped artifact are untouched.
    ///
    /// The hypothesis it exists to test (`docs/ROADMAP.md`, M1):
    /// the σ = 5.2 % between three calibration draws of the 4B (F5) comes
    /// from the **off-diagonal** terms of `H`, estimated at 13.5 samples per
    /// dimension on `down_proj`, through which the error feedback passes. If
    /// so, shrinking them towards the (stable) diagonal should halve the
    /// inter-seed spread at 28 blocks of the 0.6B before it hurts the median.
    /// The diagonal is left exact at every ρ — it is the part that is
    /// well-estimated, and the damping already handles its conditioning.
    ///
    /// Natural basis, deliberately: the rotation is a change of coordinates
    /// applied to an *estimate*, and shrinking after it would target
    /// `diag(Q H Qᵀ)`, which a Hadamard flattens to almost a constant — that
    /// variant is a large relative damping under another name, and is not
    /// what M1 sweeps.
    pub h_shrink: f64,
    pub codebook: Codebook,
    pub threads: usize,
    /// First block to quantize. Blocks below it are **advanced only** — they
    /// are assumed to already hold their quantized weights, which is what
    /// makes resuming a killed run possible (see the module header of
    /// [`crate::artifact2`]). `0` for a run that starts from scratch, which is
    /// every run that does not resume.
    pub start: usize,
    /// One past the last block to quantize; later ones are left untouched but
    /// still advanced. `usize::MAX` for a real run.
    ///
    /// ⚠️ An **absolute bound**, not a count. It has always been one — the
    /// loop tests `t >= limit` — and with [`RunConfig::start`] the difference
    /// becomes visible: a segment resuming at block 18 of a 36-block model
    /// runs with `start = 18, limit = 36`, not `limit = 18`.
    pub limit: usize,
    /// Seed for the incoherence rotation applied to each linear's **input**
    /// basis, or `None` to quantize in the natural basis.
    ///
    /// The rotation is per (block, activation), so the matrices that share an
    /// input — q/k/v, and gate/up — necessarily share it too, which is what
    /// makes rotating the single shared Hessian legitimate.
    pub rotation_seed: Option<u64>,
}

/// Quantize every block of `model` in place, sequentially.
///
/// `hidden` holds one `(1, seq, hidden_size)` tensor per calibration window,
/// entering block 0; it is advanced in place as the loop proceeds.
/// The rotation seed actually used for one (block, activation) pair.
///
/// **One source of truth on purpose.** This value is needed twice — to build
/// the rotation, and to store it in the artifact so a decoder can undo it —
/// and the two must agree exactly. Written out twice, they can drift: storing
/// the run's base seed instead un-rotates every matrix with block 0's
/// transform, which decodes to plausible garbage rather than failing. A
/// mutation test caught that nothing forbade it; this makes it unwriteable.
pub fn effective_rotation_seed(base: u64, block: usize, act: Act) -> u64 {
    base ^ ((block as u64) << 32) ^ (act.index() << 16)
}

/// What one transformer block contributes to the artifact stream: each
/// projection with the activation it was quantized against, **in the exact
/// order [`quantize_model_capturing`] pushes them**.
///
/// **One source of truth on purpose**, the third instance of the discipline
/// behind [`effective_rotation_seed`] and [`needs_dense_hessian`]. The loop
/// that *writes* the stream and the resume path that *validates* a shard
/// against it must not be able to disagree about the order. A shard whose
/// records sit in a different order is not a prefix of anything, and a resume
/// that accepted it would happily append the rest of the model behind matrices
/// the decoder then hands to the wrong projections — a file that opens, runs,
/// and is wrong.
///
/// The activation comes back alongside the name because it is what the
/// rotation seed is derived from: checking a shard's stored seed means
/// recomputing `effective_rotation_seed(base, block, act)`, and recovering
/// `act` from the name would be a second source of truth.
///
/// Block-independent by construction — every block emits the same seven
/// projections — so a caller pairs it with [`crate::artifact::key`].
pub fn block_matrix_plan() -> Vec<(Act, &'static str)> {
    Act::ALL
        .iter()
        .flat_map(|a| a.consumers().iter().map(move |n| (*a, *n)))
        .collect()
}

/// How many matrices one transformer block contributes — 7 on Qwen3: q/k/v,
/// o, gate/up, down.
///
/// Computed rather than written down, because the resume path divides a
/// shard's record count by it to recover the block that shard stops at. A
/// hard-coded 7 that drifted from [`Act::consumers`] would make a resume start
/// at the wrong block, which is the one failure this whole path has to make
/// impossible.
pub fn matrices_per_block() -> usize {
    block_matrix_plan().len()
}

/// Whether the dense Hessian has to outlive its own factorization.
///
/// The GPTQ loop reads `H` itself — as opposed to [`GptqFactor`], which is
/// what the block loop actually consumes — in exactly **one** place: the
/// end-of-layer closed-form scale solve of Algorithm 3, which runs under
/// [`GptqConfig::group_scales`] or [`GptqConfig::design_c`]. Everywhere else
/// the factor is sufficient, so keeping `H` alive buys nothing.
///
/// It is not free. One copy is `8·n²` bytes, and the four activations of a
/// Qwen3-32B block are `n = 5120, 8192, 5120, 25600`, i.e. **6.2 GB** — as
/// much again as the four factors that were computed from them. Both flags
/// are off on the published path.
///
/// **One source of truth on purpose**, the same discipline as
/// [`effective_rotation_seed`]: the site that decides to *keep* `H` and the
/// site that decides to *read* it must not be able to disagree. A `None`
/// handed to a solve that needs it would not corrupt anything — the assertion
/// in `quantize_layer` refuses it — but a `Some` the solve never reads is
/// silent, which is the direction that had gone unnoticed. Consulting the
/// flags once, here, and letting the read site pass the `Option` straight
/// through makes both directions unwriteable.
///
/// ⚠️ This trims the *plateau*, not the *peak*. The factorization's own
/// transients (`faer` holds `H`, its `L`, the inverse, and a second `L` at
/// once) are what set the high-water mark of a block, and they need `H` by
/// construction.
pub fn needs_dense_hessian(cfg: &GptqConfig) -> bool {
    cfg.group_scales || cfg.design_c
}

/// What one activation contributes to the quantization of a block.
///
/// The matrices sharing that activation — q/k/v, and gate/up — all read this
/// one entry, which is the whole point of factoring per activation rather
/// than per matrix.
struct ActFactor {
    /// What the block loop consumes.
    factor: GptqFactor,
    /// The dense `n × n` Hessian, `Some` **exactly** when
    /// [`needs_dense_hessian`] is true.
    hessian: Option<Vec<f64>>,
    /// The basis `factor` and `hessian` were built in, `None` for the natural
    /// one. The weights are rotated into it and back out again per matrix.
    rotation: Option<Rotation>,
}

/// Receives each matrix's codes as it is quantized, so a whole model never has
/// to be held in memory at once — 151 M blocks of Qwen3-4B would be 14 GB of
/// lattice points.
pub trait MatrixSink {
    fn push(&mut self, m: crate::artifact2::QuantizedMatrix) -> anyhow::Result<()>;
}

pub fn quantize_model(
    model: &mut Qwen3,
    hidden: &mut [Tensor],
    run: &RunConfig,
    progress: impl FnMut(usize, usize, &str),
) -> anyhow::Result<Report> {
    quantize_model_capturing(model, hidden, run, progress, None)
}

/// `H ← ρ·H + (1 − ρ)·diag(H)` on a dense row-major `n × n` matrix: every
/// off-diagonal entry scaled by `ρ`, the diagonal untouched. See
/// [`RunConfig::h_shrink`] for why, and for why the natural basis.
///
/// `ρ = 1` is a no-op and callers skip it; `ρ = 0` keeps the diagonal only.
/// Symmetry is preserved because each entry is scaled by the same constant.
pub fn shrink_off_diagonal(h: &mut [f64], n: usize, rho: f64) {
    assert!(
        (0.0..=1.0).contains(&rho),
        "shrink_off_diagonal: ρ = {rho} outside [0, 1]"
    );
    assert_eq!(
        h.len(),
        n * n,
        "shrink_off_diagonal: {} is not {n}²",
        h.len()
    );
    for i in 0..n {
        let row = &mut h[i * n..(i + 1) * n];
        for (j, v) in row.iter_mut().enumerate() {
            if j != i {
                *v *= rho;
            }
        }
    }
}

/// [`quantize_model`], streaming every matrix's codes to `sink`.
///
/// Capture is refused for codebooks whose reconstruction the codes cannot
/// describe — the free-magnitude variant, and anything that is not shape–gain.
/// Writing a file that decodes to different weights than the ones measured is
/// the failure this whole module exists to prevent.
pub fn quantize_model_capturing(
    model: &mut Qwen3,
    hidden: &mut [Tensor],
    run: &RunConfig,
    mut progress: impl FnMut(usize, usize, &str),
    mut sink: Option<&mut dyn MatrixSink>,
) -> anyhow::Result<Report> {
    let RunConfig {
        gptq: cfg,
        damping,
        h_shrink,
        codebook,
        threads,
        start,
        limit,
        rotation_seed,
    } = run;
    let codebook = *codebook;
    let (damping, threads, start, limit) = (*damping, *threads, *start, *limit);
    let rotation_seed = *rotation_seed;
    let h_shrink = *h_shrink;
    anyhow::ensure!(
        (0.0..=1.0).contains(&h_shrink),
        "h_shrink = {h_shrink}: ρ must be in [0, 1] (1 = H as is)"
    );
    anyhow::ensure!(
        start < limit,
        "start = {start} and limit = {limit}: this run would quantize no block"
    );
    // Only shape–gain with a load-bearing gain code is describable by codes.
    let capturing = sink.is_some();
    if capturing {
        anyhow::ensure!(
            matches!(
                codebook,
                Codebook::ShapeGain {
                    free_magnitude: false,
                    ..
                }
            ),
            "this codebook's reconstruction cannot be described by block codes; \
             writing an artifact for it would produce a file that decodes to \
             different weights than the ones evaluated"
        );
        anyhow::ensure!(
            !cfg.group_scales,
            "group_scales rescales blocks after they are chosen, so no block \
             code describes the result"
        );
    }
    let t0 = std::time::Instant::now();
    let mut report = Report::default();
    let device = model.device().clone();
    let nblocks = model.blocks.len();
    let total_rows: usize = hidden.iter().map(|h| h.dim(1).unwrap_or(0)).sum();
    anyhow::ensure!(total_rows > 0, "no calibration data");

    for t in 0..nblocks {
        // Blocks outside `[start, limit)` are pushed through untouched.
        //
        // `t >= limit` is the diagnostic truncation — the four-block de-risking
        // pass. `t < start` is the **resume** case, and it is the whole reason
        // a killed run can be restarted: those blocks were quantized by an
        // earlier segment, their weights have already been loaded back from
        // that segment's shard, and all this pass does is reproduce the hidden
        // states a single run would have carried into block `start`. Nothing
        // else of the sequential state exists — see [`crate::artifact2`].
        if t < start || t >= limit {
            let mut none = crate::model::NoCapture;
            let mask = model.causal_mask_for(&hidden[0])?;
            let tp = std::time::Instant::now();
            for h in hidden.iter_mut() {
                let next = model.blocks[t].forward(&*h, model.rotary(), &mask, t, &mut none)?;
                *h = next;
            }
            report.phases.advance += tp.elapsed().as_secs_f64();
            continue;
        }
        // ---- pass 1: collect H with the original weights ----
        let tp = std::time::Instant::now();
        let mut cap = BlockCapture {
            target: t,
            acc: HashMap::new(),
        };
        for act in Act::ALL {
            let w = act.width(model.config());
            cap.acc.insert(act, Hessian::new(w, &device, total_rows)?);
        }
        let mask = model.causal_mask_for(&hidden[0])?;
        for h in hidden.iter() {
            let _ = model.blocks[t].forward(h, model.rotary(), &mask, t, &mut cap)?;
        }
        report.phases.capture += tp.elapsed().as_secs_f64();

        // ---- factor once per activation, not once per matrix ----
        let tp = std::time::Instant::now();
        let keep_hessian = needs_dense_hessian(cfg);
        let mut kept_bytes = 0u64;
        let mut factors: HashMap<Act, ActFactor> = HashMap::new();
        for act in Act::ALL {
            // Taken out of the capture map, not borrowed from it: the device
            // accumulator is `n × n` f32 (2.6 GB for `down_proj` at 32B) and
            // is dead the instant it has been read out. Holding all four of
            // them until the end of the block, as indexing did, keeps that
            // memory alive across the factorizations — the moment the block
            // needs it most.
            let mut h = cap
                .acc
                .remove(&act)
                .expect("every activation is inserted once, above")
                .to_f64()?;
            let n = act.width(model.config());
            // M1 — see `RunConfig::h_shrink`. On the estimate, before the
            // rotation; skipped entirely on the published path.
            if h_shrink < 1.0 {
                shrink_off_diagonal(&mut h, n, h_shrink);
            }
            // Quantizing in a rotated basis means the Hessian has to move
            // with it: H' = Q H Qᵀ, since x' = Q x.
            let rot = rotation_seed.map(|s| {
                Rotation::new(n, effective_rotation_seed(s, t, act))
            });
            if let Some(q) = &rot {
                q.rotate_hessian(&mut h);
            }
            let factor = GptqFactor::new(&h, n, damping)
                .map_err(|e| anyhow::anyhow!("block {t}, {act:?}: {e}"))?;
            // `h` is dropped here unless a downstream reader exists — the
            // decision is taken from the flags, so re-enabling `group_scales`
            // or `design_c` restores it with no other change.
            let hessian = if keep_hessian {
                kept_bytes += (n as u64) * (n as u64) * std::mem::size_of::<f64>() as u64;
                Some(h)
            } else {
                None
            };
            factors.insert(
                act,
                ActFactor {
                    factor,
                    hessian,
                    rotation: rot,
                },
            );
        }
        report.dense_hessian_bytes = report.dense_hessian_bytes.max(kept_bytes);

        report.phases.factor += tp.elapsed().as_secs_f64();

        // ---- quantize the seven matrices ----
        for act in Act::ALL {
            let ActFactor {
                factor,
                hessian,
                rotation: rot,
            } = &factors[&act];
            for name in act.consumers() {
                let tp = std::time::Instant::now();
                let lin = model.blocks[t].linear_mut(name);
                let w = lin.weight();
                let (d_out, d_in) = w.dims2()?;
                let flat: Vec<f64> = w
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?
                    .into_iter()
                    .map(|v| v as f64)
                    .collect();
                report.phases.transfer += tp.elapsed().as_secs_f64();
                let tp = std::time::Instant::now();
                let mut weights = Weights::new(d_out, d_in, flat);
                // W' = W Qᵀ, quantize there, then Ŵ = Ŵ' Q — a drop-in
                // replacement that needs no runtime transform.
                if let Some(q) = rot {
                    q.rotate_weight_rows(&mut weights.w, d_out);
                }
                // The gain levels are fitted to *this* matrix, in the basis it
                // will be quantized in.
                let gain = match codebook {
                    Codebook::ShapeGain { gain_bits, .. } => Some(fit_gain_centroids(
                        &weights.w,
                        d_out,
                        d_in,
                        cfg.block,
                        gain_bits,
                        40,
                    )),
                    _ => None,
                };
                // The closure takes `gain` by move; the sink needs the same
                // levels to store them.
                let gain_for_sink = gain.clone();
                let make = move || -> Box<dyn BlockQuantizer> {
                    match codebook {
                        Codebook::Identity => Box::new(Identity { block: cfg.block }),
                        Codebook::Grid { step } => Box::new(ScalarGrid {
                            block: cfg.block,
                            step,
                        }),
                        // `cfg.block` is the group: the caller sets it from
                        // `Codebook::block_len`, and `quantize_layer` asserts
                        // the two agree.
                        Codebook::ScalarGroup { bits, .. } => Box::new(ScalarGroupwise {
                            block: cfg.block,
                            bits,
                        }),
                        Codebook::Direction => Box::new(LeechDirection::new()),
                        Codebook::ShapeGain {
                            max_shell,
                            free_magnitude,
                            level_cap,
                            ..
                        } => {
                            let q = LeechShapeGain::with_caps(
                                gain.clone().expect("fitted above"),
                                max_shell,
                                level_cap,
                            );
                            Box::new(if free_magnitude {
                                q.with_free_magnitude()
                            } else {
                                q
                            })
                        }
                    }
                };
                // The row scales the loop will use, computed on the rotated
                // weights *before* quantization — exactly as `quantize_layer`
                // fixes them internally. Recomputing them afterwards would
                // read the quantized row and give different values.
                let row_scales: Vec<f64> = if capturing {
                    (0..d_out)
                        .map(|i| {
                            llvq_quant::quantizer::row_scale(
                                &weights.w[i * d_in..(i + 1) * d_in],
                            )
                        })
                        .collect()
                } else {
                    Vec::new()
                };
                // Named for what it counts — 24-column blocks in a row —
                // because `nblocks` in this function already means the
                // model's transformer-block count, and shadowing it fed
                // the wrong number to `progress`.
                let row_blocks = d_in / cfg.block;
                let mut codes = capturing.then(|| vec![None; d_out * row_blocks]);
                llvq_quant::gptq::quantize_layer_parallel_capturing(
                    &mut weights,
                    factor,
                    // The Hessian feeds the end-of-layer closed-form scale
                    // solve, which both flags run (design C then re-projects
                    // its result back onto the gain grid). It is `Some`
                    // exactly when [`needs_dense_hessian`] said to keep it —
                    // the flags are consulted there and nowhere else, so the
                    // two sites cannot drift apart.
                    hessian.as_deref(),
                    &make,
                    cfg,
                    threads,
                    codes.as_deref_mut(),
                );
                // Narrow the tail to the precision it is *stored* at, before
                // anything else reads it.
                //
                // The tail is kept "exact", but exact in f64 is not something
                // an artifact can carry — and the un-rotation mixes every
                // column into every other, so one rounded tail column shifts
                // the whole row by an ulp. Rounding here, in the rotated basis
                // and before the un-rotation, is what makes the file and the
                // evaluated model the same object rather than nearly the same.
                if capturing {
                    let tail_w = d_in % cfg.block;
                    for i in 0..d_out {
                        let at = i * d_in + row_blocks * cfg.block;
                        for v in weights.w[at..at + tail_w].iter_mut() {
                            *v = *v as f32 as f64;
                        }
                    }
                }
                report.phases.quantize += tp.elapsed().as_secs_f64();
                // The tail is read here, in the rotated basis, because that is
                // what the decoder rebuilds before un-rotating.
                let tp = std::time::Instant::now();
                if let Some(s) = sink.as_deref_mut() {
                    let tail_w = d_in % cfg.block;
                    let mut tail = Vec::with_capacity(d_out * tail_w);
                    for i in 0..d_out {
                        let at = i * d_in + row_blocks * cfg.block;
                        tail.extend_from_slice(&weights.w[at..at + tail_w]);
                    }
                    let codes = codes
                        .take()
                        .expect("allocated when capturing")
                        .into_iter()
                        .collect::<Option<Vec<_>>>()
                        .ok_or_else(|| {
                            anyhow::anyhow!("a quantized block emitted no code")
                        })?;
                    let max_shell = match codebook {
                        Codebook::ShapeGain { max_shell, .. } => max_shell,
                        _ => unreachable!("checked when capturing was enabled"),
                    };
                    // The *effective* seed, not the run's base seed: the
                    // rotation is per (block, activation). Storing the base
                    // one would un-rotate every matrix with block 0's
                    // transform and silently scramble the model.
                    let eff_seed = rotation_seed
                        .map(|s| effective_rotation_seed(s, t, act));
                    s.push(crate::artifact2::QuantizedMatrix {
                        name: crate::artifact::key(t, name),
                        d_out,
                        d_in,
                        codes,
                        row_scales,
                        centroids: gain_for_sink.expect("shape-gain fits centroids"),
                        rotation_seed: eff_seed,
                        shell_cap: max_shell,
                        tail,
                    })?;
                }
                report.phases.write += tp.elapsed().as_secs_f64();
                let tp = std::time::Instant::now();
                if let Some(q) = rot {
                    q.unrotate_weight_rows(&mut weights.w, d_out);
                }
                let recon: Vec<f32> = weights.w.iter().map(|v| *v as f32).collect();
                let t2 = Tensor::from_vec(recon, (d_out, d_in), &device)?
                    .to_dtype(w.dtype())?;
                *lin = candle_nn::Linear::new(t2, None);
                report.phases.transfer += tp.elapsed().as_secs_f64();

                report.matrices += 1;
                report.weights += (d_out * d_in) as u64;
                report.tail_weights += (d_out * (d_in % cfg.block)) as u64;
                report.rows += d_out as u64;
                report.block_bits = codebook.block_bits();
                report.block_len = codebook.block_len();
                progress(t, nblocks, name);
            }
        }

        // ---- pass 2: advance the activations through the quantized block ----
        let tp = std::time::Instant::now();
        let mut none = crate::model::NoCapture;
        for h in hidden.iter_mut() {
            let next = model.blocks[t].forward(&*h, model.rotary(), &mask, t, &mut none)?;
            *h = next;
        }
        report.phases.advance += tp.elapsed().as_secs_f64();
    }

    report.seconds = t0.elapsed().as_secs_f64();
    Ok(report)
}

#[cfg(test)]
mod shrink_tests {
    use super::shrink_off_diagonal;

    fn sample() -> Vec<f64> {
        // Symmetric, with a diagonal that is not constant and off-diagonals
        // of both signs — the shape of a real activation Hessian, in small.
        vec![
            4.0, 1.0, -0.5, //
            1.0, 2.0, 0.25, //
            -0.5, 0.25, 9.0,
        ]
    }

    #[test]
    fn rho_zero_keeps_the_diagonal_and_nothing_else() {
        let mut h = sample();
        shrink_off_diagonal(&mut h, 3, 0.0);
        assert_eq!(h, vec![4.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 9.0]);
    }

    #[test]
    fn rho_one_is_the_identity_bit_for_bit() {
        let mut h = sample();
        shrink_off_diagonal(&mut h, 3, 1.0);
        assert_eq!(h, sample());
    }

    /// Kills the two mutants that matter: one that also scales the diagonal,
    /// and one that scales by ρ² (a "symmetric" implementation applying the
    /// factor from both sides).
    #[test]
    fn off_diagonals_are_scaled_by_rho_exactly_and_the_diagonal_is_not() {
        let mut h = sample();
        shrink_off_diagonal(&mut h, 3, 0.5);
        assert_eq!(h, vec![4.0, 0.5, -0.25, 0.5, 2.0, 0.125, -0.25, 0.125, 9.0]);
        let mut h = sample();
        shrink_off_diagonal(&mut h, 3, 0.3);
        for i in 0..3 {
            assert_eq!(
                h[i * 3 + i],
                sample()[i * 3 + i],
                "diagonal moved at ρ = 0.3"
            );
        }
    }

    #[test]
    fn symmetry_survives_every_rho() {
        for rho in [0.0, 0.1, 0.5, 0.9, 1.0] {
            let mut h = sample();
            shrink_off_diagonal(&mut h, 3, rho);
            for i in 0..3 {
                for j in 0..3 {
                    assert_eq!(h[i * 3 + j], h[j * 3 + i], "ρ = {rho}, ({i}, {j})");
                }
            }
        }
    }

    #[test]
    #[should_panic(expected = "outside [0, 1]")]
    fn a_rho_outside_the_unit_interval_is_a_programming_error() {
        let mut h = sample();
        shrink_off_diagonal(&mut h, 3, 1.5);
    }
}
