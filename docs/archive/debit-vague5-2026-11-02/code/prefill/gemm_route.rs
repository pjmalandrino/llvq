// DRAFT, never compiled or run. A sketch of where the GEMM route of the prefill plugs into
// `llvq-llm/src/model.rs` (HEAD 06e1602), for the branch building step 5 of the 2026-10-29 audit,
// and of the harness pieces prereg-prefill-2 needs. Arithmetic C-dense (prefill.md §3.2): the
// projection output equals the dense arm's, because the matrix IS the dense arm's f16 tensor and
// the call IS the dense arm's call.
//
// Revision 2 (2026-11-02): `window_nll_split` so that gate 6 compares the two paths on the SAME
// positions (the review found the revision-1 gate biased: whole-window prefill NLL against a
// decode path scoring positions 2,048-4,095 only); `prompt_then_decode_nll`; step 3b's digest
// read from the bucket copy, since `hfdense` is in neither list of ops/Dockerfile.cuda.

use candle_core::{DType, IndexOp, Result, Tensor, D};

/// Rows below which the prefill keeps the rows path. 16 or more keeps gate 4 (a 5-token prompt)
/// and every k <= 16 verify on the bit-exact path; printed on the provenance line; a served value
/// comes from the config file (`"prefill": "gemm"`, absent = "rows"), never from a variable.
pub const GEMM_ROWS_MIN: usize = 64;

/// New capability on the device ports (crate::device): materialize this projection as the dense
/// arm holds it, f16 [d_out, d_in], natural basis. Lattice: `deq_tetra48_f64_h`; int4: `deq_q4_h`.
/// The returned tensor is freshly allocated from the stream-ordered pool and dropped by the caller
/// right after its matmul is enqueued, so at most one or two scratch matrices are live.
pub trait DenseScratch {
    fn dense_f16(&self) -> Result<Tensor>;
    /// Check-only twin: the f32 values before the f16 cast, for the digest comparison of step 3b.
    /// Never on a served path.
    fn dense_f32_check(&self) -> Result<Tensor>;
}

/// The branch, in `group_forward` (model.rs:1132), after the dense branch and before `SegPlan`:
///
/// ```ignore
/// let rows: usize = x.dims()[..x.dims().len() - 1].iter().product();
/// if gemm_route_on && rows >= GEMM_ROWS_MIN && projs.iter().all(|p| p.has_dense_scratch()) {
///     return projs
///         .iter()
///         .map(|p| {
///             let w = p.dense_f16()?;                         // one matrix, transient
///             // The dense arm's own call: for a contiguous [b, l, d_in] input candle reshapes
///             // to [b*l, d_in] and runs ONE 2-D matmul (candle-nn 0.9.2 linear.rs). Same
///             // operands, same shapes, same cuBLAS call, so the same bits as the dense arm.
///             candle_nn::Linear::new(w, None).forward(x)
///         })
///         .collect();
/// }
/// ```
///
/// No `prepare`, no rotation: the dense weights are in the natural basis, as `Proj::Dense` is.
/// `x` must be the tensor the dense arm would see (contiguous); a non-contiguous input makes
/// candle switch to a broadcast matmul, another cuBLAS call, other bits.
pub fn _doc_only() {}

/// The card check of prereg-prefill-2 step 3a: every record's route buffer against the dense
/// arm's resident f16 tensor, on the device. Returns the number of differing elements. `ne`
/// treats -0 and +0 as equal, so a zero count here is followed by the host bit check on the five
/// digest records of step 3b.
pub fn differing_elements(route: &Tensor, dense: &Tensor) -> Result<u64> {
    if route.dims() != dense.dims() {
        candle_core::bail!("shapes {:?} against {:?}", route.dims(), dense.dims());
    }
    let ne = route.ne(dense)?.to_dtype(DType::U32)?;
    let n: u32 = ne.flatten_all()?.sum(0)?.to_scalar()?;
    Ok(n as u64)
}

// Step 3b's reference. `bin/hfdense` writes one sha256 per record over the dequantized f32
// values, little-endian, row-major (hfpack.rs `dense_digest`, convention string in the JSON), and
// records its source file's name and sha256. It is not in the image. Rule 9 first: the bucket
// already holds one, `hf-cuda-4b-2026-09-30/model/llvq-dense-digest.json`, 28,987 B
// (wave5-deep/bucket-listing.txt:473), visible in a job as
// `/out/hf-cuda-4b-2026-09-30/model/llvq-dense-digest.json`. The job script:
//   1. checks the JSON's own sha256 against the value pinned in the stamped prereg (read from the
//      Mac with `hf buckets cp` before stamping, a $0 step that needs the operator's go);
//   2. checks that its `source.sha256` is the sealed file's, 886391a8...; a digest of another
//      file is refused by name;
//   3. compares sha256(f32 LE bytes of dense_f32_check()) with the record's entry, five records.
// The alternative, adding hfdense to both lists of ops/Dockerfile.cuda and to UPLOAD_ALLOW, is a
// recipe change and needs a go.

/// `window_nll` (model.rs:1954-1966) that also returns the NLL over the tail of the window, the
/// predictions of tokens `split..l`. Same forward, same logits, same log-softmax: only a second
/// sum over a narrowed slice of the already gathered log-probs. Row `i` predicts token `i + 1`,
/// so the tail is rows `split - 1 .. l - 1`.
/// Returns (total, count, tail_total, tail_count).
pub fn window_nll_split(model: &crate::model::Qwen3, tokens: &[u32], split: usize) -> Result<(f64, usize, f64, usize)> {
    let l = tokens.len();
    if l < 2 || split < 1 || split >= l {
        candle_core::bail!("window_nll_split: {l} tokens, split {split}");
    }
    let dev = model.device();
    let input = Tensor::from_slice(tokens, (1, l), dev)?;
    let logits = model.logits(&input, &mut crate::model::NoCapture)?.to_dtype(DType::F32)?;
    let logits = logits.i(0)?.narrow(0, 0, l - 1)?;
    let targets = Tensor::from_slice(&tokens[1..], l - 1, dev)?;
    let logp = candle_nn::ops::log_softmax(&logits, D::Minus1)?;
    let picked = logp.gather(&targets.reshape((l - 1, 1))?, 1)?; // [l - 1, 1]
    let total: f32 = picked.sum_all()?.to_scalar()?;
    let tail: f32 = picked.narrow(0, split - 1, l - split)?.sum_all()?.to_scalar()?;
    Ok((-total as f64, l - 1, -tail as f64, l - split))
}

/// The prompt-then-decode path of step 5: tokens `0..split` through the cached prefill under the
/// arm (`prefill_chunked`, model.rs:2190-2203, on the served grouped config: grouped heads on a
/// Grow(256) store), then tokens `split..l-1` one at a time through the fused decode, which is
/// the same code for every arm. Scores the predictions of tokens `split..l`: the prompt's last
/// logits, then one per decode step. Returns (tail_total, tail_count), comparable with the
/// `tail_*` of `window_nll_split` on the same window and split.
///
/// ```ignore
/// let mut caches = model.fresh_caches();
/// let first = model.prefill_chunked(&tokens[..split], MAX_PREFILL_ROWS, &mut caches)?; // [1, 1, vocab] f32
/// let mut nll = -log_softmax(first)[tokens[split]];
/// for (pos, &t) in tokens.iter().enumerate().take(l - 1).skip(split) {
///     let logits = model.step_logits(t, pos, &mut caches)?;          // model.rs:2207, the served decode step
///     nll += -log_softmax(logits)[tokens[pos + 1]];
/// }
/// Ok((nll, l - split))
/// ```
pub fn _prompt_then_decode_doc() {}

// The hidden-state identity of step 3c: run `hidden_cached` over the same 2,048-token prompt on
// the fused model with the route on and on the dense model, both under the expanded form on
// `Cat` (the dense arm's form), and compare the last position's final hidden state with
// `bitwise_diff` (attn_rows.rs). The head differs between the two models (fused emb_q4 against
// a dense matmul of the dequantized table), so logits are NOT expected to match; hidden states are.
