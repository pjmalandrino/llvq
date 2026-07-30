//! Algorithm 1 (Appendix B) and Algorithm 3 (Appendix I.1) of the paper.
//!
//! A layer is quantized left-to-right over blocks of `b = 24` input
//! channels. After each block, the quantization error is projected onto the
//! not-yet-quantized columns through the inverse-Hessian Cholesky factor, so
//! every later block is chosen knowing what earlier ones got wrong. The rows
//! (`d_out` of them) are independent throughout.
//!
//! ## Two conventions the paper leaves ambiguous, and what we do
//!
//! Algorithm 3 is written compactly and disagrees with Algorithms 1–2 on two
//! points. Both are resolved here in favour of the standard GPTQ reading,
//! which Appendix I's prose also supports:
//!
//! 1. **The retraction is per row.** Algorithm 3 line 5 writes it with a
//!    Frobenius norm over the whole column block, but Appendix I is explicit:
//!    *"quantization is performed row-wise on each row-block […] and we apply
//!    the same retraction per row"*, with `Ŵ_{i,B} = (‖W_{i,B}‖₂/‖W̃_{i,B}‖₂)
//!    W̃_{i,B}` (Eq. 17). A block-level norm would let one row's magnitude
//!    error be paid for by another's.
//! 2. **The residual is formed against the compensated weights**, not the
//!    original ones. Algorithm 3 line 6 writes `E ← W − W̃`; Algorithms 1
//!    (line 14) and 2 (line 6) both use the current, already-compensated
//!    `W̃`. Using the original would double-count every correction already
//!    applied to this block, so error feedback would fight itself.
//!
//! ## Cost note
//!
//! The optional group-scale refinement ([`GptqConfig::group_scales`]) is the
//! expensive part: it builds and solves an `m × m` system per output row
//! (`m = d_in/24`), which is roughly `d_out · m · d_in · b` multiply-adds per
//! layer. On a 2560×2560 layer that is ~18 GFLOP, against ~0.4 for the block
//! loop itself. It is off by default.

use crate::linalg::GptqFactor;
use crate::quantizer::{BlockCode, BlockQuantizer};

/// What to do with input channels that do not fill a whole block.
///
/// A Leech block is exactly 24 weights, and real layers are not multiples of
/// 24 — Qwen3-0.6B has widths 1024 = 24·42+16, 2048 = 24·85+8 and 3072 =
/// 24·128. The paper says only "the last may be smaller" and never says what
/// quantizes it, so the choice is ours and it must be explicit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TailPolicy {
    /// Refuse to run. The default, so a mis-tiled layer is a loud failure
    /// rather than a silently degraded one.
    Reject,
    /// Leave the trailing columns at full precision.
    ///
    /// Cheap and honest: the tail is at most 23 of `d_in` columns, so on a
    /// 1024-wide layer it costs `16·14/1024 = 0.22` extra bits per weight and
    /// on a 3072-wide one, nothing at all.
    ///
    /// Note what this does *not* mean: the tail still **receives** the error
    /// feedback of every earlier block, and should — being unquantized, it can
    /// absorb that compensation exactly, which is strictly better than
    /// discarding it. It simply never contributes error of its own, so the
    /// block loop stops rather than quantizing a short block.
    KeepExact,
}

/// How to run the layer loop.
#[derive(Clone, Copy, Debug)]
pub struct GptqConfig {
    /// Input-channel block size. Must equal the quantizer's block length,
    /// except for a possibly shorter final block.
    pub block: usize,
    /// Apply the spherical retraction `Π_{‖w‖}` after quantizing each row
    /// block — the *Spherical GPTQ* of §3.3. Keeps the block norm exact and
    /// leaves only angular error for the Hessian feedback to propagate.
    pub retract: bool,
    /// Run the closed-form per-row block-scale refinement of Algorithm 3
    /// after all blocks are quantized. Expensive; see the module note.
    pub group_scales: bool,
    /// Ridge term for that refinement's normal equations, **relative** to
    /// the mean diagonal of the system (like the Hessian damping).
    pub lambda: f64,
    /// What to do with a trailing partial block.
    pub tail: TailPolicy,
}

impl Default for GptqConfig {
    fn default() -> Self {
        Self {
            block: llvq_core::DIM,
            retract: true,
            group_scales: false,
            lambda: 1e-2,
            tail: TailPolicy::Reject,
        }
    }
}

/// Number of weights per row this configuration actually quantizes, and the
/// resulting bits per weight given a `bits`-wide index per block and
/// `full_bits` for an untouched tail.
pub fn bits_per_weight(d_in: usize, block: usize, bits: usize, full_bits: usize) -> f64 {
    let whole = d_in / block;
    let tail = d_in - whole * block;
    (whole * bits + tail * full_bits) as f64 / d_in as f64
}

/// A layer's weights, `d_out × d_in`, row-major.
pub struct Weights {
    pub d_out: usize,
    pub d_in: usize,
    pub w: Vec<f64>,
}

impl Weights {
    pub fn new(d_out: usize, d_in: usize, w: Vec<f64>) -> Self {
        assert_eq!(w.len(), d_out * d_in, "weights must be d_out × d_in");
        Self { d_out, d_in, w }
    }

    #[inline]
    fn row(&self, i: usize) -> &[f64] {
        &self.w[i * self.d_in..(i + 1) * self.d_in]
    }

    #[inline]
    fn row_mut(&mut self, i: usize) -> &mut [f64] {
        let d = self.d_in;
        &mut self.w[i * d..(i + 1) * d]
    }
}

/// Quantize one layer in place.
///
/// `h` is the layer Hessian `AᵀA/N` (`d_in × d_in`, row-major) and `factor`
/// its GPTQ factor; passing both avoids recomputing the factor per layer when
/// several weight matrices share an input (q/k/v of one attention block do).
/// `h` is only read when [`GptqConfig::group_scales`] is set.
pub fn quantize_layer(
    weights: &mut Weights,
    factor: &GptqFactor,
    h: Option<&[f64]>,
    quant: &mut dyn BlockQuantizer,
    cfg: &GptqConfig,
) {
    quantize_layer_capturing(weights, factor, h, quant, cfg, None)
}

/// [`quantize_layer`], additionally recording the code of every block.
///
/// `codes`, when given, is `d_out × (d_in / cfg.block)` in **row-major** order
/// — `codes[i * nblocks + p]` is row `i`, block `p`. Entries stay `None` for
/// quantizers that carry no discrete code.
///
/// Tail columns under [`TailPolicy::KeepExact`] produce no code: they are not
/// quantized, and an artifact stores them verbatim.
pub fn quantize_layer_capturing(
    weights: &mut Weights,
    factor: &GptqFactor,
    h: Option<&[f64]>,
    quant: &mut dyn BlockQuantizer,
    cfg: &GptqConfig,
    mut codes: Option<&mut [Option<BlockCode>]>,
) {
    let (d_out, d_in) = (weights.d_out, weights.d_in);
    let nblocks = d_in / cfg.block;
    if let Some(c) = codes.as_deref() {
        assert_eq!(
            c.len(),
            d_out * nblocks,
            "code buffer must be d_out × (d_in / block)"
        );
        // The group-scale refinement rewrites the weights after every block is
        // chosen, so the codes would no longer describe what is stored.
        assert!(
            !cfg.group_scales,
            "capturing codes is incompatible with group_scales: the closed-form \
             refinement rescales blocks after the fact, and no block code can \
             express the result"
        );
    }
    assert_eq!(factor.dim(), d_in, "factor must match the input dimension");
    assert!(cfg.block > 0);
    if cfg.group_scales {
        assert!(
            h.is_some_and(|m| m.len() == d_in * d_in),
            "group-scale refinement needs the d_in × d_in Hessian"
        );
    }

    let original = cfg.group_scales.then(|| weights.w.clone());

    let mut qbuf = vec![0.0f64; cfg.block];
    // Error of the current block, d_out × b, row-major.
    let mut e = vec![0.0f64; d_out * cfg.block];

    // One reference scale per output row, fixed **before** the block loop.
    //
    // It is stored as a single f16 per row, so it must not drift as error
    // feedback rewrites the row: recomputing it per block would make the
    // reconstruction depend on d_in/24 distinct scales that no artifact can
    // carry at the rate the accounting charges.
    let row_scales: Vec<f64> = (0..d_out)
        .map(|i| crate::quantizer::row_scale(weights.row(i)))
        .collect();

    let mut s = 0usize;
    while s < d_in {
        let b = cfg.block.min(d_in - s);
        if b != quant.block_len() {
            match cfg.tail {
                TailPolicy::KeepExact => break, // exact columns, nothing to propagate
                TailPolicy::Reject => panic!(
                    "input dimension {d_in} leaves a {b}-wide tail that the \
                     quantizer (block {}) cannot take; choose a TailPolicy",
                    quant.block_len()
                ),
            }
        }

        for i in 0..d_out {
            let row = weights.row_mut(i);
            // The gain code needs the row's scale; quantizers without one
            // ignore this.
            quant.set_row_scale(row_scales[i]);
            let v = &row[s..s + b];
            let norm_before = if cfg.retract {
                v.iter().map(|a| a * a).sum::<f64>().sqrt()
            } else {
                0.0
            };
            quant.quantize(v, &mut qbuf[..b]);
            if cfg.retract {
                // Π_{‖w‖}: put the candidate back on a sphere, per row
                // (Eq. 17). The quantizer names which one — for a gain code it
                // must be a sphere the code can express, or the retraction
                // would hand the magnitude back as a free float and cancel the
                // gain entirely. `None` means it is already there.
                if let Some(target) = quant.retraction_target(norm_before) {
                    let n = qbuf[..b].iter().map(|a| a * a).sum::<f64>().sqrt();
                    if n > 0.0 {
                        let k = target / n;
                        for q in qbuf[..b].iter_mut() {
                            *q *= k;
                        }
                    }
                }
            }
            for k in 0..b {
                // Residual against the *compensated* weights (see module doc).
                e[i * b + k] = row[s + k] - qbuf[k];
                row[s + k] = qbuf[k];
            }
            if let Some(c) = codes.as_deref_mut() {
                c[i * nblocks + s / cfg.block] = quant.last_code();
            }
        }

        // Propagate onto the untouched columns:
        //   W̃_{:,R} ← W̃_{:,R} − (E_{:,Q} U_QQ⁻¹) U_QR
        let r0 = s + b;
        if r0 < d_in {
            let ecur = &mut e[..d_out * b];
            factor.solve_block(ecur, d_out, s, b);
            for i in 0..d_out {
                let row = &mut weights.w[i * d_in..(i + 1) * d_in];
                for k in 0..b {
                    let x = ecur[i * b + k];
                    if x == 0.0 {
                        continue;
                    }
                    for (j, cell) in row[r0..].iter_mut().enumerate() {
                        *cell -= x * factor.u(s + k, r0 + j);
                    }
                }
            }
        }

        s = r0;
    }

    if cfg.group_scales {
        let orig = original.expect("cloned when group_scales is set");
        refine_group_scales(weights, h.expect("checked above"), &orig, cfg);
    }
}

/// Algorithm 3, lines 11–18: with the directions frozen, pick the per-row
/// block scales that minimize the local proxy exactly.
///
/// For row `i`, writing `g_p = W̃_{i,Q_p}` (the quantized block, as a vector
/// supported on `Q_p`), the proxy is quadratic in the scale vector `s`:
///
/// ```text
/// M[p,q] = g_p H_{Q_p Q_q} g_qᵀ,   r[p] = g_p H_{Q_p,:} w_{i,:}ᵀ
/// s      = (M + λI)⁻¹ r
/// ```
///
/// so one small solve per row replaces any search over magnitudes. This is
/// what lets the direction code spend all of its bits on direction.
fn refine_group_scales(weights: &mut Weights, h: &[f64], original: &[f64], cfg: &GptqConfig) {
    let (d_out, d_in) = (weights.d_out, weights.d_in);
    let m = d_in.div_ceil(cfg.block);
    let bounds: Vec<(usize, usize)> = (0..m)
        .map(|p| {
            let s = p * cfg.block;
            (s, (s + cfg.block).min(d_in))
        })
        .collect();

    // g H, one row per block: (m × d_in).
    let mut gh = vec![0.0f64; m * d_in];
    let mut mat = vec![0.0f64; m * m];
    let mut rhs = vec![0.0f64; m];

    for i in 0..d_out {
        let wq = weights.row(i).to_vec();
        let worig = &original[i * d_in..(i + 1) * d_in];

        gh.fill(0.0);
        for (p, &(ps, pe)) in bounds.iter().enumerate() {
            let out = &mut gh[p * d_in..(p + 1) * d_in];
            for c in ps..pe {
                let g = wq[c];
                if g == 0.0 {
                    continue;
                }
                let hrow = &h[c * d_in..(c + 1) * d_in];
                for (o, &hv) in out.iter_mut().zip(hrow.iter()) {
                    *o += g * hv;
                }
            }
        }

        for p in 0..m {
            let ghp = &gh[p * d_in..(p + 1) * d_in];
            rhs[p] = ghp.iter().zip(worig.iter()).map(|(a, b)| a * b).sum();
            for (q, &(qs, qe)) in bounds.iter().enumerate() {
                let v: f64 = (qs..qe).map(|c| ghp[c] * wq[c]).sum();
                mat[p * m + q] = v;
            }
        }
        // The ridge must be **relative to the system it damps**, exactly as
        // the Hessian damping is relative to mean(diag H). `M` scales with
        // the square of the weight magnitude, so an absolute λ is meaningless:
        // on real LLM weights (~1e-2) it dwarfs `M`, the solve degenerates to
        // `s ≈ r/λ`, and every block gets scaled toward zero. That failure is
        // silent in the algebra and catastrophic in the model.
        let mean_diag: f64 = (0..m).map(|p| mat[p * m + p]).sum::<f64>() / m as f64;
        let ridge = cfg.lambda * mean_diag;
        for p in 0..m {
            mat[p * m + p] += ridge;
        }

        // `mat` and `rhs` are rebuilt from scratch each row, so the solve is
        // free to consume them.
        if solve_spd(&mut mat, &mut rhs, m) {
            let row = weights.row_mut(i);
            for (p, &(ps, pe)) in bounds.iter().enumerate() {
                let k = rhs[p];
                for cell in row[ps..pe].iter_mut() {
                    *cell *= k;
                }
            }
        }
        // A non-SPD system means this row's quantized blocks are linearly
        // dependent under H; leaving the scales at 1 is the safe answer.
    }
}

/// Solve `M s = r` in place on `r`, for a symmetric positive definite `M`
/// (Cholesky). Returns `false`, leaving `r` untouched, if `M` is not
/// positive definite. Destroys `m`.
fn solve_spd(m: &mut [f64], r: &mut [f64], n: usize) -> bool {
    if crate::linalg::cholesky_lower(m, n).is_err() {
        return false;
    }
    // Forward solve L y = r.
    for i in 0..n {
        let mut acc = r[i];
        for k in 0..i {
            acc -= m[i * n + k] * r[k];
        }
        r[i] = acc / m[i * n + i];
    }
    // Back solve Lᵀ s = y.
    for i in (0..n).rev() {
        let mut acc = r[i];
        for k in i + 1..n {
            acc -= m[k * n + i] * r[k];
        }
        r[i] = acc / m[i * n + i];
    }
    true
}

/// The local proxy objective of Appendix F.2: `Tr(ΔW H ΔWᵀ)`, the expected
/// output MSE of the layer under the calibration distribution.
///
/// This is the number every design choice in this module is trying to lower,
/// so it is the natural thing for tests to assert on.
pub fn proxy_loss(delta: &[f64], d_out: usize, d_in: usize, h: &[f64]) -> f64 {
    assert_eq!(delta.len(), d_out * d_in);
    assert_eq!(h.len(), d_in * d_in);
    let mut total = 0.0;
    for i in 0..d_out {
        let d = &delta[i * d_in..(i + 1) * d_in];
        for a in 0..d_in {
            if d[a] == 0.0 {
                continue;
            }
            let hrow = &h[a * d_in..(a + 1) * d_in];
            let mut s = 0.0;
            for b in 0..d_in {
                s += hrow[b] * d[b];
            }
            total += d[a] * s;
        }
    }
    total
}

/// [`quantize_layer`] across threads, splitting by output row.
///
/// Every step of the algorithm is per row: the block quantization, the
/// residual, the triangular solve and the propagation onto later columns all
/// act on one row at a time, and the group-scale refinement solves one system
/// per row. Rows therefore never interact, and splitting the matrix by rows
/// is exact rather than approximate — `parallel_matches_serial_exactly` in
/// the test suite holds it to bit equality.
///
/// `make_quantizer` is called once per thread: a `BlockQuantizer` carries
/// search scratch and cannot be shared.
pub fn quantize_layer_parallel(
    weights: &mut Weights,
    factor: &GptqFactor,
    h: Option<&[f64]>,
    make_quantizer: &(dyn Fn() -> Box<dyn BlockQuantizer> + Sync),
    cfg: &GptqConfig,
    threads: usize,
) {
    let (d_out, d_in) = (weights.d_out, weights.d_in);
    let threads = threads.max(1).min(d_out);
    if threads == 1 {
        let mut q = make_quantizer();
        quantize_layer(weights, factor, h, q.as_mut(), cfg);
        return;
    }
    let rows_per = d_out.div_ceil(threads);
    std::thread::scope(|sc| {
        for (chunk, slice) in weights.w.chunks_mut(rows_per * d_in).enumerate() {
            let rows = slice.len() / d_in;
            let _ = chunk;
            sc.spawn(move || {
                let mut sub = Weights::new(rows, d_in, slice.to_vec());
                let mut q = make_quantizer();
                quantize_layer(&mut sub, factor, h, q.as_mut(), cfg);
                slice.copy_from_slice(&sub.w);
            });
        }
    });
}
