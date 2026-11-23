//! The pieces of an Inkling 2 Mini decoder layer, f32, for one sequence.
//!
//! Every function cites the reference line it reproduces: transformers
//! 046bdb8cc2, `src/transformers/models/inkling2_loop/modeling_inkling2_loop.py`
//! ("line N" below). Activations are `[T, width]` for one sequence; heads are
//! split as `[T, heads, dim]` and attention runs per head as `[heads, T, dim]`.
//!
//! One code path serves the prompt and the decode step. Each layer visit owns
//! a [`SlotCache`]: its keys and values after their norms, and the last `K - 1`
//! inputs of each of its four convolutions. A prompt runs on an empty slot
//! (the convolution history is then zeros, which is the reference's left
//! padding, lines 316-322 and 444), a step on the slot the prompt filled
//! (`causal_conv1d_update`, lines 286-302, and `cache.update`, line 612).

use candle_core::{DType, Device, IndexOp, Result, Tensor, D};

use super::config::InklingConfig;
use super::loader::{NativeCheckpoint, NativeExperts};
use phases::{mark, Phase};

/// Wall time of a forward split by part, for the measurement flag of
/// `inkrun`. Off by default: then [`mark`] is one atomic load and changes
/// nothing. On, each mark drains the device and charges the time since the
/// previous mark to the part just done, so the parts sum to the wall time of
/// the marked span, slowed by the drains.
pub mod phases {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Mutex;
    use std::time::Instant;

    use candle_core::{Device, Result};

    /// The parts of a forward.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum Phase {
        /// The embedding rows and their norm.
        Embed,
        /// The output head and its softcap.
        Head,
        /// The logits read back to the host.
        Readback,
        /// Router product, sigmoid, read back, top-k on the host.
        Routing,
        /// The attention projections (query, key, value, relative, sink,
        /// gate, output) with their clips and scales.
        AttnProducts,
        /// Relative bias and mask, scores, sinks, softmax, value product.
        AttnScores,
        /// The cache append and the copies of what it holds.
        CacheCopies,
        /// The four short convolutions of a visit.
        Convolutions,
        /// Block residuals, message gates, norms, residual adds.
        NormsGates,
        /// The MoE call.
        Moe,
        /// Whatever no other mark charged.
        Other,
    }

    pub const ALL: [Phase; 11] = [
        Phase::Embed,
        Phase::Head,
        Phase::Readback,
        Phase::Routing,
        Phase::AttnProducts,
        Phase::AttnScores,
        Phase::CacheCopies,
        Phase::Convolutions,
        Phase::NormsGates,
        Phase::Moe,
        Phase::Other,
    ];

    impl Phase {
        pub fn name(self) -> &'static str {
            match self {
                Phase::Embed => "embed",
                Phase::Head => "head",
                Phase::Readback => "readback",
                Phase::Routing => "routing",
                Phase::AttnProducts => "attention_products",
                Phase::AttnScores => "attention_scores_softmax",
                Phase::CacheCopies => "cache_append_copies",
                Phase::Convolutions => "convolutions",
                Phase::NormsGates => "norms_gates_residual",
                Phase::Moe => "moe",
                Phase::Other => "other",
            }
        }
    }

    static ON: AtomicBool = AtomicBool::new(false);
    static STATE: Mutex<Option<(Instant, [f64; 11])>> = Mutex::new(None);

    /// Drain `dev` and start charging from now.
    pub fn start(dev: &Device) -> Result<()> {
        dev.synchronize()?;
        *STATE.lock().expect("no panic holds this lock") = Some((Instant::now(), [0.0; 11]));
        ON.store(true, Ordering::SeqCst);
        Ok(())
    }

    /// Stop and return the seconds charged to each part, in the order of [`ALL`].
    pub fn stop() -> [f64; 11] {
        ON.store(false, Ordering::SeqCst);
        STATE.lock().expect("no panic holds this lock").take().map(|s| s.1).unwrap_or([0.0; 11])
    }

    /// Drain `dev` and charge the time since the last mark to `p`.
    #[inline]
    pub fn mark(dev: &Device, p: Phase) -> Result<()> {
        if !ON.load(Ordering::Relaxed) {
            return Ok(());
        }
        dev.synchronize()?;
        let now = Instant::now();
        if let Some((last, acc)) = STATE.lock().expect("no panic holds this lock").as_mut() {
            acc[p as usize] += now.duration_since(*last).as_secs_f64();
            *last = now;
        }
        Ok(())
    }
}

/// `x @ w.T` for `x` `[T, in]` and `w` `[out, in]` (`nn.Linear` without bias).
pub fn linear(x: &Tensor, w: &Tensor) -> Result<Tensor> {
    x.matmul(&w.t()?)
}

/// `Inkling2LoopUnweightedRMSNorm`, lines 67-76: `x * rsqrt(mean(x^2) + eps)` on the last dim.
pub fn rms_unweighted(x: &Tensor, eps: f64) -> Result<Tensor> {
    let ms = (x.sqr()?.mean_keepdim(D::Minus1)? + eps)?;
    x.broadcast_mul(&ms.sqrt()?.recip()?)
}

/// `Inkling2LoopRMSNorm`, lines 730-745: `weight * (x * rsqrt(mean(x^2) + eps))`.
pub fn rms_norm(x: &Tensor, w: &Tensor, eps: f64) -> Result<Tensor> {
    rms_unweighted(x, eps)?.broadcast_mul(w)
}

/// `Inkling2LoopL2Clip`, lines 176-187: each vector on the last dim scaled by
/// `min(threshold / (||x|| + 1e-6), 1)`.
pub fn l2_clip(x: &Tensor, threshold: f64) -> Result<Tensor> {
    let norm = x.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
    let f = ((norm + 1e-6)?.recip()? * threshold)?.minimum(1.0)?;
    x.broadcast_mul(&f)
}

/// `Inkling2LoopLogScale`, lines 205-220: `1 + alpha * ln(max((pos + 1) / n_floor, 1))`, as `[T, 1, 1]`.
pub fn log_scale(cfg: &InklingConfig, positions: &[usize], dev: &Device) -> Result<Tensor> {
    let v: Vec<f32> = positions
        .iter()
        .map(|&p| {
            let r = ((p + 1) as f32 / cfg.log_scaling_n_floor as f32).max(1.0);
            1.0 + cfg.log_scaling_alpha as f32 * r.ln()
        })
        .collect();
    Tensor::from_vec(v, (positions.len(), 1, 1), dev)
}

/// The last `K - 1` inputs of one short convolution, oldest first
/// (`conv_state`, lines 297, 379, 438). A prompt leaves a block `[K - 1, C]`;
/// a one-row step keeps a fixed ring of `K - 1` row handles `[1, C]` and
/// replaces the oldest, with no concatenation and no copy. Nothing is written
/// in place: a clone, or a mark, keeps what it saw.
#[derive(Debug, Clone, Default)]
pub enum ConvState {
    /// No input yet: the history is zeros (the reference's left padding).
    #[default]
    Empty,
    /// `[K - 1, C]`, oldest row first.
    Block(Tensor),
    /// `K - 1` rows `[1, C]`; `rows[head]` is the oldest.
    Ring { rows: Vec<Tensor>, head: usize },
}

impl ConvState {
    pub fn is_empty(&self) -> bool {
        matches!(self, ConvState::Empty)
    }

    /// The state as one block `[k - 1, C]`, oldest row first; `None` when
    /// empty (zeros). A ring is concatenated in time order: the same values.
    pub fn block(&self, k: usize) -> Result<Option<Tensor>> {
        match self {
            ConvState::Empty => Ok(None),
            ConvState::Block(s) => Ok(Some(s.clone())),
            ConvState::Ring { rows, head } => {
                let ordered: Vec<Tensor> = (0..k - 1).map(|j| rows[(head + j) % (k - 1)].clone()).collect();
                Ok(Some(Tensor::cat(&ordered, 0)?))
            }
        }
    }

    /// The `k - 1` history rows `[1, C]`, oldest first; zeros rows like `x` when empty.
    fn rows(&self, x: &Tensor, k: usize) -> Result<Vec<Tensor>> {
        match self {
            ConvState::Empty => {
                let z = Tensor::zeros((1, x.dim(1)?), x.dtype(), x.device())?;
                Ok(vec![z; k - 1])
            }
            ConvState::Block(s) => (0..k - 1).map(|i| s.narrow(0, i, 1)).collect(),
            ConvState::Ring { rows, head } => Ok((0..k - 1).map(|j| rows[(head + j) % (k - 1)].clone()).collect()),
        }
    }

    /// After a one-row step on `x` `[1, C]`: `x` replaces the oldest row.
    fn push_row(&mut self, x: &Tensor, k: usize) -> Result<()> {
        let (mut rows, head) = match std::mem::take(self) {
            ConvState::Ring { rows, head } => (rows, head),
            other => (other.rows(x, k)?, 0),
        };
        rows[head] = x.clone();
        *self = ConvState::Ring { rows, head: (head + 1) % (k - 1) };
        Ok(())
    }
}

/// The convolution context: the `k - 1` inputs before `x` (zeros when
/// `state` is empty) then `x`, `[k - 1 + T, C]`. `state` becomes the last
/// `k - 1` rows (`update_conv_state` / `conv_state.copy_`, lines 297, 379, 438).
fn conv_context(x: &Tensor, k: usize, state: &mut ConvState) -> Result<Tensor> {
    let t = x.dim(0)?;
    let ctx = match &*state {
        ConvState::Block(s) => Tensor::cat(&[s, x], 0)?,
        ConvState::Empty => x.pad_with_zeros(0, k - 1, 0)?,
        ring @ ConvState::Ring { .. } => {
            let mut parts = ring.rows(x, k)?;
            parts.push(x.clone());
            Tensor::cat(&parts, 0)?
        }
    };
    *state = ConvState::Block(ctx.narrow(0, t, k - 1)?.contiguous()?);
    Ok(ctx)
}

/// `Inkling2LoopShortConvolution.forward`, lines 362-393: a depthwise causal
/// convolution, `w` `[C, K]` from the file's `[C, 1, K]`, tap `K - 1` on the
/// current token (`F.conv1d` with left padding `K - 1`, lines 316-322, or the
/// cached history in decode, lines 373-376), then the residual and the output
/// scale `1 / sqrt(K + 1)` (line 392).
pub fn short_conv(x: &Tensor, w: &Tensor, state: &mut ConvState) -> Result<Tensor> {
    let k = w.dim(1)?;
    let t = x.dim(0)?;
    let mut acc = x.zeros_like()?;
    if t == 1 {
        // One row: the history rows where they are, the same products and sums as the context below.
        let rows = state.rows(x, k)?;
        for (j, row) in rows.iter().chain(std::iter::once(x)).enumerate() {
            let tap = w.i((.., j))?; // [C]
            acc = (acc + row.broadcast_mul(&tap)?)?;
        }
        state.push_row(x, k)?;
    } else {
        let ctx = conv_context(x, k, state)?;
        for j in 0..k {
            let tap = w.i((.., j))?; // [C]
            acc = (acc + ctx.narrow(0, j, t)?.broadcast_mul(&tap)?)?;
        }
    }
    let out = (acc + x)?;
    // line 392: the output scale (absent from Inkling 1, per the reference comment)
    out / ((k + 1) as f64).sqrt()
}

/// `Inkling2LoopDynamicShortConvolution.forward`, lines 428-455, prompt and
/// decode alike (the reference has one path for both, lines 437-446).
/// Tap `lag` (backward distance, `static_weight[0]` on the current token) is
/// `static_weight[lag, c] + (dynamic_input @ weight_proj.T)[t, head(c) * K + lag]`,
/// shared by the channels of a head. Residual, then scale `1 / sqrt(2K + 1)` (line 454).
pub fn dynamic_conv(
    x: &Tensor,
    dynamic_input: &Tensor,
    static_w: &Tensor,
    weight_proj: &Tensor,
    heads: usize,
    state: &mut ConvState,
) -> Result<Tensor> {
    let (t, c) = x.dims2()?;
    let k = static_w.dim(0)?;
    let dw = linear(dynamic_input, weight_proj)?.reshape((t, heads, k))?; // line 430-431
    // One row reads the history rows where they are; a prompt reads the context.
    let (rows, ctx) = if t == 1 { (Some(state.rows(x, k)?), None) } else { (None, Some(conv_context(x, k, state)?)) };
    let mut acc = x.zeros_like()?;
    for lag in 0..k {
        let d = dw.i((.., .., lag))?.unsqueeze(2)?.broadcast_as((t, heads, c / heads))?.reshape((t, c))?;
        let w = d.broadcast_add(&static_w.i(lag)?)?; // line 432
        let src = match (&rows, &ctx) {
            (Some(_), _) if lag == 0 => x.clone(),
            (Some(rows), _) => rows[k - 1 - lag].clone(),
            (None, Some(ctx)) => ctx.narrow(0, k - 1 - lag, t)?,
            (None, None) => unreachable!("one of the two is set"),
        };
        acc = (acc + src.mul(&w)?)?; // lines 444-453
    }
    if t == 1 {
        state.push_row(x, k)?;
    }
    let out = (acc + x)?;
    out / ((2 * k + 1) as f64).sqrt()
}

/// `Inkling2LoopBlockAttentionResidual.forward`, lines 838-853 (and the
/// message residual of lines 158-173 has the same body): the block states and
/// the partial state stacked, unweighted-normed (keys AND values), one
/// softmax over the stack per query, scaled by `hidden^-0.5`. `queries` is
/// `[nq, H]`; returns `nq` tensors `[T, H]`.
pub fn block_attn_res(queries: &Tensor, states: &[Tensor], partial: &Tensor) -> Result<Vec<Tensor>> {
    let mut all: Vec<Tensor> = states.to_vec();
    all.push(partial.clone());
    let h = partial.dim(D::Minus1)?;
    let s = rms_unweighted(&Tensor::stack(&all, 1)?, 1e-6)?; // [T, n, H], line 839-840
    let t = s.dim(0)?;
    let nq = queries.dim(0)?;
    let q = queries.unsqueeze(0)?.broadcast_as((t, nq, h))?.contiguous()?;
    let logits = (q.matmul(&s.transpose(1, 2)?)? * (h as f64).powf(-0.5))?; // [T, nq, n]
    let w = candle_nn::ops::softmax_last_dim(&logits)?;
    let out = w.matmul(&s)?; // [T, nq, H]
    (0..nq).map(|i| out.i((.., i, ..))?.contiguous()).collect()
}

/// `Inkling2LoopSoftcap`, lines 751-757.
fn softcap(x: &Tensor, cap: f64) -> Result<Tensor> {
    (x / cap)?.tanh()? * cap
}

/// The convolution of a key or value projection (lines 495-508).
pub enum KvConv {
    /// `sconv.conv1d.weight`, `[C, K]`.
    Static(Tensor),
    /// `static_weight` `[K, C]` and `weight_proj` `[heads * K, H]`.
    Dynamic { static_w: Tensor, weight_proj: Tensor },
}

/// The weights of one attention block (sliding, lines 546-572, or full, 639-665).
pub struct AttnWeights {
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub sliding: bool,
    pub rel_extent: usize,
    pub wq: Tensor,
    pub q_scale: Tensor,
    pub wk: Tensor,
    pub k_conv: KvConv,
    pub wv: Tensor,
    pub v_conv: KvConv,
    pub gate: Tensor,
    pub wo: Tensor,
    pub sink: Tensor,
    pub wr: Tensor,
    pub rel_scale: Tensor,
    pub rel_proj: Tensor,
    /// The attention through the kernels of `super::attn_dense`, when the
    /// loader chose them ([`LayerWeights::load_with`]): a one-row forward runs
    /// them, and the seven matrices above are then empty, held once in BF16
    /// in these buffers and widened for a several-row call.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub dense: Option<super::attn_dense::FusedAttn>,
}

/// The seven f32 matrices of one attention block, owned: what a several-row
/// call reads when the loader kept them in BF16 only.
pub struct AttnMats {
    pub wq: Tensor,
    pub wr: Tensor,
    pub sink: Tensor,
    pub gate: Tensor,
    pub wk: Tensor,
    pub wv: Tensor,
    pub wo: Tensor,
}

/// Every expert of a layer as f32 matrices: `(gate_up [2I, H], down [H, I])`.
pub trait ExpertSource {
    fn count(&self) -> usize;
    fn expert(&self, e: usize, dev: &Device) -> Result<(Tensor, Tensor)>;
}

impl ExpertSource for NativeExperts<'_> {
    fn count(&self) -> usize {
        NativeExperts::count(self)
    }
    fn expert(&self, e: usize, dev: &Device) -> Result<(Tensor, Tensor)> {
        let (gu, down) = self.dequant(e);
        let (r13, h) = (self.w13.rows, self.w13.cols);
        Ok((Tensor::from_vec(gu, (r13, h), dev)?, Tensor::from_vec(down, (self.w2.rows, self.w2.cols), dev)?))
    }
}

/// The one seam of the MoE call: what computes `sum_k w_k expert_k(x)` for a
/// layer. Every dense [`ExpertSource`] is one, through [`experts_forward`];
/// `resident::ResidentExperts` (Metal) is the other, through
/// `resident::experts_forward`, from lattice codes kept on the device.
pub trait MoeSource {
    /// `x` `[T, H]` and its routing to the MoE output `[T, H]` on the device of `x`.
    fn moe(&self, x: &Tensor, routing: &Routing, cfg: &InklingConfig) -> Result<Tensor>;

    /// GPU kernel launches of the lattice path since the last call (zero
    /// for a dense source), and reset.
    fn take_launches(&self) -> usize {
        0
    }

    /// The MoE of one row whose routing is still on the device: `sel` is
    /// `[2k]` u32, the ids of the `k` routed experts, ascending and distinct,
    /// chosen among `experts`, then the bits of their weights. The same
    /// result as [`Self::moe`] on that routing, bit for bit. `None` when this
    /// source has no such path: the caller then reads the routing back and
    /// calls [`Self::moe`].
    fn moe_routed_on_device(
        &self,
        _x: &Tensor,
        _sel: &Tensor,
        _experts: usize,
        _cfg: &InklingConfig,
    ) -> Result<Option<Tensor>> {
        Ok(None)
    }

    /// For a source with a fused one-row path: run its activation in one
    /// launch (`true`) or through candle's operations, the reference of that
    /// launch's gate. The same bits either way; nothing to do elsewhere.
    fn set_fused_act(&mut self, _on: bool) {}
}

impl<S: ExpertSource> MoeSource for S {
    fn moe(&self, x: &Tensor, routing: &Routing, cfg: &InklingConfig) -> Result<Tensor> {
        experts_forward(x, routing, self, cfg)
    }
}

/// The routing of one MoE call: per token, the experts sorted ascending and
/// their weights aligned with them.
#[derive(Debug, Clone)]
pub struct Routing {
    pub indices: Vec<Vec<u32>>,
    pub weights: Vec<Vec<f32>>,
    /// Per token, the 8th minus the 9th largest `score + bias`.
    pub margin: Vec<f32>,
    /// Per token, `score + bias` of every expert: what the top-k selected on.
    pub scores: Vec<Vec<f32>>,
}

/// `Inkling2LoopTopKRouter.forward`, lines 1163-1172, in f32: sigmoid scores,
/// top-k on `score + e_score_correction_bias`, weights the scores of the
/// chosen experts normalised to sum 1 (plus 1e-20) times `route_scale`.
pub fn route(x: &Tensor, w_gate: &Tensor, bias: &Tensor, top_k: usize, scale: f64) -> Result<Routing> {
    route_host_bias(x, w_gate, &bias.to_vec1::<f32>()?, top_k, scale)
}

/// [`route`] with the bias already on the host: the layer reads it once at
/// load ([`LayerWeights::router_bias_host`]), not at every visit. The same
/// f32 values in the same additions, so the same routing bit for bit.
pub fn route_host_bias(x: &Tensor, w_gate: &Tensor, bias: &[f32], top_k: usize, scale: f64) -> Result<Routing> {
    let scores = candle_nn::ops::sigmoid(&linear(x, w_gate)?)?.to_vec2::<f32>()?;
    if scores.first().is_some_and(|s| s.len() != bias.len()) {
        candle_core::bail!("route: {} scores and {} bias values", scores[0].len(), bias.len());
    }
    let mut r = Routing { indices: vec![], weights: vec![], margin: vec![], scores: vec![] };
    for s in &scores {
        let sel: Vec<f32> = s.iter().zip(bias).map(|(a, b)| a + b).collect();
        let mut order: Vec<usize> = (0..sel.len()).collect();
        order.sort_by(|&a, &b| sel[b].total_cmp(&sel[a]).then(a.cmp(&b)));
        r.margin.push(sel[order[top_k - 1]] - sel[order[top_k]]);
        let mut idx: Vec<usize> = order[..top_k].to_vec();
        idx.sort_unstable();
        let picked: Vec<f32> = idx.iter().map(|&i| s[i]).collect();
        let sum: f32 = picked.iter().sum::<f32>() + 1e-20;
        r.weights.push(picked.iter().map(|w| w / sum * scale as f32).collect());
        r.indices.push(idx.iter().map(|&i| i as u32).collect());
        r.scores.push(sel);
    }
    Ok(r)
}

/// `Inkling2LoopExperts.forward`, lines 787-807, with the softcapped SwiGLU of
/// lines 760-771: experts in ascending order, each over its tokens, the
/// weighted outputs accumulated in f32 by `index_add`.
pub fn experts_forward(
    x: &Tensor,
    routing: &Routing,
    src: &dyn ExpertSource,
    cfg: &InklingConfig,
) -> Result<Tensor> {
    let dev = x.device();
    let inter = cfg.intermediate_size;
    let mut out = x.zeros_like()?;
    if x.dim(0)? == 1 && routing.indices.len() == 1 {
        // One row: the experts in the ascending order of the general path
        // below, each on `x` itself (no `index_select`), its weighted output
        // added to the sum (`index_add` onto one row is that same addition),
        // the 8 weights uploaded once.
        let (idx, w) = (&routing.indices[0], &routing.weights[0]);
        let wts = Tensor::from_vec(w.clone(), (w.len(), 1), dev)?;
        let mut order: Vec<usize> = (0..idx.len()).collect();
        order.sort_by_key(|&j| idx[j]);
        for j in order {
            let (gu_w, down_w) = src.expert(idx[j] as usize, dev)?;
            let gu = linear(x, &gu_w)?;
            let g = gu.narrow(1, 0, inter)?.contiguous()?;
            let u = gu.narrow(1, inter, inter)?.contiguous()?;
            let act = softcap(&g, cfg.mlp_gate_softcap)?
                .mul(&candle_nn::ops::sigmoid(&g)?)?
                .mul(&softcap(&u, cfg.mlp_up_softcap)?)?;
            let y = linear(&act, &down_w)?.broadcast_mul(&wts.narrow(0, j, 1)?)?;
            out = (out + y)?;
        }
        return Ok(out);
    }
    let mut per_expert: Vec<Vec<(u32, f32)>> = vec![vec![]; src.count()];
    for (t, (idx, w)) in routing.indices.iter().zip(&routing.weights).enumerate() {
        for (&e, &wt) in idx.iter().zip(w) {
            per_expert[e as usize].push((t as u32, wt));
        }
    }
    for (e, toks) in per_expert.iter().enumerate() {
        if toks.is_empty() {
            continue;
        }
        let (gu_w, down_w) = src.expert(e, dev)?;
        let ids = Tensor::from_vec(toks.iter().map(|p| p.0).collect::<Vec<_>>(), toks.len(), dev)?;
        let wts = Tensor::from_vec(toks.iter().map(|p| p.1).collect::<Vec<_>>(), (toks.len(), 1), dev)?;
        let xs = x.index_select(&ids, 0)?;
        let gu = linear(&xs, &gu_w)?;
        let g = gu.narrow(1, 0, inter)?.contiguous()?;
        let u = gu.narrow(1, inter, inter)?.contiguous()?;
        // line 771: softcap(gate) * sigmoid(gate) * softcap(up)
        let act = softcap(&g, cfg.mlp_gate_softcap)?
            .mul(&candle_nn::ops::sigmoid(&g)?)?
            .mul(&softcap(&u, cfg.mlp_up_softcap)?)?;
        let y = linear(&act, &down_w)?.broadcast_mul(&wts)?;
        out = out.index_add(&ids, &y, 0)?;
    }
    Ok(out)
}

/// The additive bias of the attention scores, `[heads, T, S]`, for `T`
/// queries at positions `offset..offset + T` against `S` keys at positions
/// `0..S`: the relative position bias (lines 260-270, distances per lines
/// 1249-1268) plus the causal mask, and for a sliding layer the window (keys
/// at distance `>= window` masked).
fn score_bias(rel_logits: &Tensor, offset: usize, s: usize, extent: usize, window: Option<usize>) -> Result<Tensor> {
    let (heads, t, _) = rel_logits.dims3()?;
    let dev = rel_logits.device();
    let mut idx = Vec::with_capacity(t * s);
    let mut mask = Vec::with_capacity(t * s);
    let mut keep = Vec::with_capacity(t * s);
    for i in 0..t {
        for j in 0..s {
            let d = (offset + i) as i64 - j as i64; // q_position - kv_position, lines 1266-1268
            idx.push(d.clamp(0, extent as i64 - 1) as u32);
            keep.push(if d >= 0 && (d as usize) < extent { 1f32 } else { 0f32 });
            let visible = d >= 0 && window.is_none_or(|w| (d as usize) < w);
            mask.push(if visible { 0f32 } else { f32::NEG_INFINITY });
        }
    }
    let idx = Tensor::from_vec(idx, (1, t, s), dev)?.broadcast_as((heads, t, s))?.contiguous()?;
    let keep = Tensor::from_vec(keep, (1, t, s), dev)?;
    let mask = Tensor::from_vec(mask, (1, t, s), dev)?;
    let bias = rel_logits.contiguous()?.gather(&idx, 2)?.broadcast_mul(&keep)?;
    bias.broadcast_add(&mask)
}

/// Query rows of one attention score tensor, by default: a prompt longer
/// than this goes through scores, bias, softmax and the value product in
/// chunks of this many rows, after every key and value of the prompt is in
/// the cache. At 24 heads and 9,000 cached positions one chunk of scores is
/// 0.44 GB in f32.
pub const DEFAULT_ATTN_CHUNK: usize = 512;

/// `[kv_heads, T, d]` repeated to `[kv_heads * rep, T, d]`, head `h` reading kv head `h / rep` (lines 99-108).
fn repeat_kv(x: &Tensor, rep: usize) -> Result<Tensor> {
    if rep == 1 {
        return Ok(x.clone());
    }
    let (kv, t, d) = x.dims3()?;
    x.unsqueeze(1)?.broadcast_as((kv, rep, t, d))?.reshape((kv * rep, t, d))
}

/// Positions a slot's key and value buffers are allocated in.
const KV_GRAIN: usize = 256;

/// The capacity after growing from `cap` to hold `need` positions: at least
/// half again, in whole grains.
fn grown_capacity(cap: usize, need: usize) -> usize {
    need.max(cap + cap / 2).div_ceil(KV_GRAIN) * KV_GRAIN
}

/// What one layer visit keeps between calls: keys and values after their
/// norms (what `cache.update` stores, line 612), and the last `K - 1` inputs
/// of the key, value, post-attention and post-MLP convolutions (`conv_idx` 0
/// to 3, lines 563-566 and 868-873).
///
/// Keys and values live in buffers allocated for more positions than they
/// hold, grown by half when full: the new rows are written in place and the
/// attention reads a view of the positions held. Keys are held transposed,
/// `[kv_heads, d, cap]`, so the score product reads the head repeat of that
/// view with no transpose copy; values are `[kv_heads, cap, d]`. A clone
/// copies the buffers, since the next append writes into them.
#[derive(Debug, Default)]
pub struct SlotCache {
    k_buf: Option<Tensor>,
    v_buf: Option<Tensor>,
    len: usize,
    /// Positions the buffers hold at least once they exist ([`Self::reserve`]).
    min_capacity: usize,
    pub conv_k: ConvState,
    pub conv_v: ConvState,
    pub conv_attn: ConvState,
    pub conv_mlp: ConvState,
    /// While the fused glue steps this slot ([`LayerWeights::forward_fused`]),
    /// the post-attention and post-MLP states live in these device rings and
    /// `conv_attn` / `conv_mlp` are stale. Every other reader settles them
    /// first ([`Self::settle_rings`]); a clone and a mark read them back.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    fused: Option<super::glue::FusedRings>,
    /// While the attention kernels step this slot, the key and value states
    /// live in these device rings and `conv_k` / `conv_v` are stale. A
    /// several-row attention settles them first ([`Self::settle_attn`]); a
    /// clone and a mark read them back; a rollback drops them.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    attn: Option<super::attn_dense::AttnSlot>,
}

impl Clone for SlotCache {
    fn clone(&self) -> Self {
        let deep = |b: &Option<Tensor>| b.as_ref().map(|t| t.copy().expect("copy of a cache buffer"));
        let [conv_k, conv_v, conv_attn, conv_mlp] = self.conv_states().expect("read back of the convolution rings");
        Self {
            k_buf: deep(&self.k_buf),
            v_buf: deep(&self.v_buf),
            len: self.len,
            min_capacity: self.min_capacity,
            conv_k,
            conv_v,
            conv_attn,
            conv_mlp,
            #[cfg(all(target_os = "macos", feature = "metal"))]
            fused: None,
            #[cfg(all(target_os = "macos", feature = "metal"))]
            attn: None,
        }
    }
}

/// What [`SlotCache::mark`] keeps: the positions held and the four
/// convolution states (key, value, post-attention, post-MLP). The states are
/// never written in place, so a mark holds handles, not copies.
#[derive(Debug, Clone)]
pub struct SlotMark {
    len: usize,
    conv: [ConvState; 4],
}

impl SlotMark {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl SlotCache {
    /// Tokens held.
    pub fn len(&self) -> usize {
        self.len
    }

    /// The state needed to come back here. Rings the fused glue writes in
    /// place are read back into new tensors, so the mark keeps copies.
    pub fn mark(&self) -> SlotMark {
        SlotMark { len: self.len, conv: self.conv_states().expect("read back of the convolution rings") }
    }

    /// The four convolution states (key, value, post-attention, post-MLP),
    /// the last two read back from the rings when the fused glue holds them.
    fn conv_states(&self) -> Result<[ConvState; 4]> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        let (conv_k, conv_v) = match &self.attn {
            Some(a) => {
                let (k, v) = a.states()?;
                (ConvState::Block(k), ConvState::Block(v))
            }
            None => (self.conv_k.clone(), self.conv_v.clone()),
        };
        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        let (conv_k, conv_v) = (self.conv_k.clone(), self.conv_v.clone());
        #[cfg(all(target_os = "macos", feature = "metal"))]
        if let Some(r) = &self.fused {
            let (a, m) = r.states()?;
            return Ok([conv_k, conv_v, ConvState::Block(a), ConvState::Block(m)]);
        }
        Ok([conv_k, conv_v, self.conv_attn.clone(), self.conv_mlp.clone()])
    }

    /// Bring the key and value states back from the attention kernels'
    /// rings, if they hold them; the rings are dropped. Nothing changes otherwise.
    pub fn settle_attn(&mut self) -> Result<()> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        if let Some(a) = self.attn.take() {
            let (k, v) = a.states()?;
            self.conv_k = ConvState::Block(k);
            self.conv_v = ConvState::Block(v);
        }
        Ok(())
    }

    /// Whether the attention kernels' rings hold the key and value states.
    pub fn has_attn_rings(&self) -> bool {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        return self.attn.is_some();
        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        false
    }

    /// Bring the post-attention and post-MLP states back from the fused
    /// glue's rings, if it holds them; the rings are dropped. Nothing changes
    /// otherwise.
    pub fn settle_rings(&mut self) -> Result<()> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        if let Some(r) = self.fused.take() {
            let (a, m) = r.states()?;
            self.conv_attn = ConvState::Block(a);
            self.conv_mlp = ConvState::Block(m);
        }
        Ok(())
    }

    /// Whether the fused glue's rings hold the post-attention and post-MLP states.
    pub fn has_rings(&self) -> bool {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        return self.fused.is_some();
        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        false
    }

    /// Back to `m`: the positions past `m.len()` are forgotten (the next
    /// append writes over them) and the four convolution states are those of
    /// `m`. Refused when `m` holds more positions than the slot. The caller
    /// checks that the positions up to `m.len()` are those the mark saw.
    pub fn rollback(&mut self, m: &SlotMark) -> Result<()> {
        if m.len > self.len {
            candle_core::bail!("slot rollback: the mark holds {} positions, the slot {}", m.len, self.len);
        }
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            // the next fused step refills its rings from the states restored below
            self.fused = None;
            self.attn = None;
        }
        self.len = m.len;
        let [k, v, attn, mlp] = m.conv.clone();
        self.conv_k = k;
        self.conv_v = v;
        self.conv_attn = attn;
        self.conv_mlp = mlp;
        Ok(())
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Positions the buffers can hold before they grow.
    pub fn capacity(&self) -> usize {
        self.v_buf.as_ref().map_or(0, |v| v.dim(1).expect("rank-3 values"))
    }

    /// Views of what the slot holds: keys `[kv_heads, S, d]` and values
    /// `[kv_heads, S, d]`. They share the buffers: read them before the next append.
    pub fn keys_values(&self) -> Result<Option<(Tensor, Tensor)>> {
        match (&self.k_buf, &self.v_buf) {
            (Some(k), Some(v)) => Ok(Some((k.narrow(2, 0, self.len)?.transpose(1, 2)?, v.narrow(1, 0, self.len)?))),
            _ => Ok(None),
        }
    }

    /// Write `k` and `v` `[kv_heads, T, d]` at positions `len..len + T`, in
    /// place, growing the buffers first if they are full. Returns views of
    /// every position held: keys `[kv_heads, d, S]`, values `[kv_heads, S, d]`.
    /// Make room for `positions`: the buffers, once they exist, hold at
    /// least that many before they grow again. The values held do not change.
    pub fn reserve(&mut self, positions: usize) -> Result<()> {
        self.min_capacity = self.min_capacity.max(positions);
        if positions > self.capacity() {
            if let Some(k) = self.k_buf.clone() {
                let (nkv, d, _) = k.dims3()?;
                self.grow(positions.div_ceil(KV_GRAIN) * KV_GRAIN, nkv, d, &k)?;
            }
        }
        Ok(())
    }

    /// New buffers of `cap` positions, like `like` in dtype and device, the
    /// positions held copied in once.
    fn grow(&mut self, cap: usize, nkv: usize, d: usize, like: &Tensor) -> Result<()> {
        let kb = Tensor::zeros((nkv, d, cap), like.dtype(), like.device())?;
        let vb = Tensor::zeros((nkv, cap, d), like.dtype(), like.device())?;
        if let (Some(ok), Some(ov)) = (&self.k_buf, &self.v_buf) {
            if self.len > 0 {
                kb.slice_set(&ok.narrow(2, 0, self.len)?.contiguous()?, 2, 0)?;
                vb.slice_set(&ov.narrow(1, 0, self.len)?.contiguous()?, 1, 0)?;
            }
        }
        self.k_buf = Some(kb);
        self.v_buf = Some(vb);
        Ok(())
    }

    fn append_kv(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        let (nkv, t, d) = k.dims3()?;
        let need = self.len + t;
        if need > self.capacity() {
            let cap = grown_capacity(self.capacity(), need.max(self.min_capacity));
            self.grow(cap, nkv, d, k)?;
        }
        let (kb, vb) = (self.k_buf.as_ref().expect("allocated above"), self.v_buf.as_ref().expect("allocated above"));
        kb.slice_set(&k.transpose(1, 2)?.contiguous()?, 2, self.len)?;
        vb.slice_set(&v.contiguous()?, 1, self.len)?;
        self.len = need;
        Ok((kb.narrow(2, 0, need)?, vb.narrow(1, 0, need)?))
    }
}

impl AttnWeights {
    fn kv(&self, x: &Tensor, w: &Tensor, conv: &KvConv, state: &mut ConvState) -> Result<Tensor> {
        let s = linear(x, w)?;
        mark(x.device(), Phase::AttnProducts)?;
        let y = match conv {
            KvConv::Static(cw) => short_conv(&s, cw, state),
            KvConv::Dynamic { static_w, weight_proj } => {
                dynamic_conv(&s, x, static_w, weight_proj, self.kv_heads, state)
            }
        }?;
        mark(x.device(), Phase::Convolutions)?;
        Ok(y)
    }

    /// The attention forward: sliding (lines 598-636) or full (lines
    /// 691-727). `xq` feeds the query, the relative bias, the sinks and the
    /// output gate; `xk` and `xv` the key and value projections and their
    /// convolutions. The `T` rows sit at positions `cache.len()..`; their keys
    /// and values are appended to `cache` and attend over all it holds.
    pub fn forward(
        &self,
        cfg: &InklingConfig,
        xq: &Tensor,
        xk: &Tensor,
        xv: &Tensor,
        cache: &mut SlotCache,
    ) -> Result<Tensor> {
        self.forward_chunked(cfg, xq, xk, xv, cache, DEFAULT_ATTN_CHUNK)
    }

    /// [`Self::forward`] with the query rows of the scores taken `chunk`
    /// at a time (`0`: all at once). Every key and value of the `T` rows is
    /// appended first; each chunk then builds its own bias and mask and
    /// attends over every position the cache holds, its causal mask hiding
    /// the later rows of the same call. With `T <= chunk` the operations are
    /// those of one call.
    pub fn forward_chunked(
        &self,
        cfg: &InklingConfig,
        xq: &Tensor,
        xk: &Tensor,
        xv: &Tensor,
        cache: &mut SlotCache,
        chunk: usize,
    ) -> Result<Tensor> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        if let Some(d) = &self.dense {
            if xq.dim(0)? == 1 {
                return self.forward_dense(cfg, d, xq, xk, xv, cache, chunk);
            }
            // Several rows: the key and value states come back from the
            // rings, and the matrices are widened for this call only.
            cache.settle_attn()?;
            let m = d.widen()?;
            return self.forward_mats(cfg, &m.refs(), xq, xk, xv, cache, chunk);
        }
        self.forward_mats(cfg, &self.refs(), xq, xk, xv, cache, chunk)
    }

    /// The seven matrices this block holds in f32.
    fn refs(&self) -> MatRefs<'_> {
        MatRefs {
            wq: &self.wq,
            wr: &self.wr,
            sink: &self.sink,
            gate: &self.gate,
            wk: &self.wk,
            wv: &self.wv,
            wo: &self.wo,
        }
    }

    /// [`Self::forward_chunked`] through candle on the f32 matrices `m`.
    #[allow(clippy::too_many_arguments)]
    fn forward_mats(
        &self,
        cfg: &InklingConfig,
        m: &MatRefs,
        xq: &Tensor,
        xk: &Tensor,
        xv: &Tensor,
        cache: &mut SlotCache,
        chunk: usize,
    ) -> Result<Tensor> {
        let t = xq.dim(0)?;
        let dev = xq.device();
        let offset = cache.len();
        let (nh, nkv, hd) = (self.heads, self.kv_heads, self.head_dim);
        let positions: Vec<usize> = (offset..offset + t).collect();
        let tau = if self.sliding { None } else { Some(log_scale(cfg, &positions, dev)?) };

        // Query: linear, learned scale, L2 clip, and the log scale in full layers (lines 458-475, 190-236).
        let q = linear(xq, m.wq)?.reshape((t, nh, hd))?.broadcast_mul(&self.q_scale)?;
        let mut q = l2_clip(&q, cfg.qk_l2_clip_threshold)?;
        if let Some(tau) = &tau {
            q = q.broadcast_mul(tau)?;
        }
        let q = q.transpose(0, 1)?.contiguous()?; // [nh, T, hd]
        mark(dev, Phase::AttnProducts)?;

        // Key: linear, convolution, L2 clip without scale. Value: linear, convolution, unweighted RMS norm (lines 515-528).
        let k = self.kv(xk, m.wk, &self.k_conv, &mut cache.conv_k)?.reshape((t, nkv, hd))?;
        let k = l2_clip(&k, cfg.qk_l2_clip_threshold)?.transpose(0, 1)?.contiguous()?;
        let v = self.kv(xv, m.wv, &self.v_conv, &mut cache.conv_v)?.reshape((t, nkv, hd))?;
        let v = rms_unweighted(&v, cfg.rms_norm_eps)?.transpose(0, 1)?.contiguous()?;
        mark(dev, Phase::AttnProducts)?;
        // line 612: the cache returns every key and value it holds (keys as `[nkv, hd, S]`)
        let (kt, v) = cache.append_kv(&k, &v)?;
        let s = v.dim(1)?;
        mark(dev, Phase::CacheCopies)?;

        // Relative bias (lines 260-270): per-head states, scale, L2 clip at the rel threshold, log scale in full layers.
        let r = linear(xq, m.wr)?.reshape((t, nh, cfg.d_rel))?.broadcast_mul(&self.rel_scale)?;
        let mut r = l2_clip(&r, cfg.rel_l2_clip_threshold)?;
        if let Some(tau) = &tau {
            r = r.broadcast_mul(tau)?;
        }
        let rel_logits = r.transpose(0, 1)?.contiguous()?.broadcast_matmul(&self.rel_proj)?; // [nh, T, extent]
        mark(dev, Phase::AttnProducts)?;

        let sinks = || linear(xq, m.sink)?.t()?.unsqueeze(2); // [nh, T, 1], line 619
        let a = self.scores(cfg, &q, &kt, &v, &rel_logits, sinks, offset, s, t, chunk)?;

        // Gated output (lines 540-543).
        let gate = candle_nn::ops::sigmoid(&linear(xq, m.gate)?)?;
        let out = linear(&a.mul(&gate)?, m.wo)?;
        mark(dev, Phase::AttnProducts)?;
        Ok(out)
    }

    /// Scores over every position the slot holds, the per-token sink logit
    /// with zero value (lines 126-139), softmax in f32, and the value
    /// product, `chunk` query rows at a time: `[T, heads * head_dim]`.
    #[allow(clippy::too_many_arguments)]
    fn scores(
        &self,
        cfg: &InklingConfig,
        q: &Tensor,
        kt: &Tensor,
        v: &Tensor,
        rel_logits: &Tensor,
        sinks: impl FnOnce() -> Result<Tensor>,
        offset: usize,
        s: usize,
        t: usize,
        chunk: usize,
    ) -> Result<Tensor> {
        let dev = q.device();
        let (nh, nkv, hd) = (self.heads, self.kv_heads, self.head_dim);
        let window = if self.sliding { Some(cfg.sliding_window_size) } else { None };
        let rep = nh / nkv;
        // The repeat copies the views into contiguous `[nh, hd, S]` and `[nh, S, hd]`: the
        // matrices the score and value products read before, with no transpose copy.
        let kt = repeat_kv(kt, rep)?.contiguous()?;
        let v = repeat_kv(v, rep)?.contiguous()?;
        mark(dev, Phase::CacheCopies)?;
        let sinks = sinks()?;
        let rows = if chunk == 0 { t } else { chunk.min(t) };
        let mut parts = Vec::with_capacity(t.div_ceil(rows));
        let mut r0 = 0;
        while r0 < t {
            let n = rows.min(t - r0);
            let (qc, rc, sc) = if n == t {
                (q.clone(), rel_logits.clone(), sinks.clone())
            } else {
                (q.narrow(1, r0, n)?.contiguous()?, rel_logits.narrow(1, r0, n)?, sinks.narrow(1, r0, n)?)
            };
            let bias = score_bias(&rc, offset + r0, s, self.rel_extent, window)?;
            let scores = (qc.matmul(&kt)? * (hd as f64).powf(-0.5))?.add(&bias)?;
            let probs =
                candle_nn::ops::softmax_last_dim(&Tensor::cat(&[&scores, &sc], 2)?.contiguous()?)?.narrow(2, 0, s)?;
            parts.push(probs.contiguous()?.matmul(&v)?.transpose(0, 1)?.reshape((n, nh * hd))?);
            r0 += n;
        }
        let a = if parts.len() == 1 { parts.pop().expect("one part") } else { Tensor::cat(&parts, 0)? };
        mark(dev, Phase::AttnScores)?;
        Ok(a)
    }

    /// One row through the attention kernels (`super::attn_dense`): `stage_in`
    /// in place of the products and their small steps, the cache append and
    /// the score statements as [`Self::forward_mats`] runs them, `stage_out`
    /// for the gated output. The key and value states move into the slot's
    /// rings, filled from what it held (a prompt's block, a candle step's
    /// ring, or zeros). Class B against the candle attention.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[allow(clippy::too_many_arguments)]
    fn forward_dense(
        &self,
        cfg: &InklingConfig,
        d: &super::attn_dense::FusedAttn,
        xq: &Tensor,
        xk: &Tensor,
        xv: &Tensor,
        cache: &mut SlotCache,
        chunk: usize,
    ) -> Result<Tensor> {
        let dev = xq.device();
        let offset = cache.len();
        if cache.attn.is_none() {
            let k = cache.conv_k.block(d.w.taps)?;
            let v = cache.conv_v.block(d.w.taps)?;
            cache.attn = Some(super::attn_dense::AttnSlot::new(d, k.as_ref(), v.as_ref())?);
        }
        let (xq, xk, xv) = (xq.contiguous()?, xk.contiguous()?, xv.contiguous()?);
        let rings = &mut cache.attn.as_mut().expect("filled above").rings;
        let st = d.kernels.stage_in(&d.w, &xq, &xk, &xv, offset, rings)?;
        mark(dev, Phase::AttnProducts)?;
        let (kt, v) = cache.append_kv(&st.k, &st.v)?;
        let s = v.dim(1)?;
        mark(dev, Phase::CacheCopies)?;
        let a = match &d.row {
            // One launch over the cache in place (`super::attn_scores`): no
            // repeat of the key-value heads, no mask built on the host.
            Some(row) => {
                let window = if self.sliding { Some(cfg.sliding_window_size) } else { None };
                let keys = kt.transpose(1, 2)?;
                let a = row.forward(&super::attn_scores::RowInputs {
                    q: &st.q,
                    rel: &st.rel_logits,
                    sinks: &st.sinks,
                    keys: &keys,
                    values: &v,
                    pos: offset,
                    window,
                })?;
                mark(dev, Phase::AttnScores)?;
                a
            }
            None => {
                let sinks = || Ok(st.sinks.clone());
                self.scores(cfg, &st.q, &kt, &v, &st.rel_logits, sinks, offset, s, 1, chunk)?
            }
        };
        let out = d.kernels.stage_out(&d.w, &a.contiguous()?, &st.gate)?;
        mark(dev, Phase::AttnProducts)?;
        Ok(out)
    }
}

/// The seven f32 matrices a candle attention call reads, borrowed.
struct MatRefs<'a> {
    wq: &'a Tensor,
    wr: &'a Tensor,
    sink: &'a Tensor,
    gate: &'a Tensor,
    wk: &'a Tensor,
    wv: &'a Tensor,
    wo: &'a Tensor,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
impl AttnMats {
    fn refs(&self) -> MatRefs<'_> {
        MatRefs { wq: &self.wq, wr: &self.wr, sink: &self.sink, gate: &self.gate, wk: &self.wk, wv: &self.wv, wo: &self.wo }
    }
}

/// `Inkling2LoopMessageGate`, lines 79-96: `(1 - forget) * h + gain * message`,
/// both gates `tanh` of a rank-`gate_rank` projection of the two unweighted-
/// normed inputs side by side. Names per `conversion_mapping.py` 2685-2697.
pub struct MessageGate {
    /// `down_weight` `[rank, 2H]`.
    pub down: Tensor,
    /// `forget_weight` `[H, rank]`.
    pub forget: Tensor,
    /// `gain_weight` `[H, rank]` (`message_gate_proj`).
    pub gain: Tensor,
}

impl MessageGate {
    pub fn forward(&self, h: &Tensor, message: &Tensor, eps: f64) -> Result<Tensor> {
        let g = Tensor::cat(&[rms_unweighted(h, eps)?, rms_unweighted(message, eps)?], 1)?;
        let g = linear(&g, &self.down)?;
        let forget = linear(&g, &self.forget)?.tanh()?;
        let gain = linear(&g, &self.gain)?.tanh()?;
        (forget.neg()? + 1.0)?.mul(h)? + gain.mul(message)?
    }
}

/// The four message gates of a loop layer: query, key, value, MLP (lines 931-934).
pub struct MessageGates {
    pub q: MessageGate,
    pub k: MessageGate,
    pub v: MessageGate,
    pub mlp: MessageGate,
}

/// The weights of a decoder layer: sliding (lines 856-873, loop variant
/// 913-934) or full (lines 1000-1019, loop variant 1065-1088).
pub struct LayerWeights {
    pub index: usize,
    pub sliding: bool,
    pub boundary: bool,
    /// `[nq, H]`: 1 query in sliding layers, 3 in full ones; absent in layer 0 (lines 862, 1006).
    pub attention_input: Option<Tensor>,
    pub mlp_input: Tensor,
    /// `input_layernorm` in sliding layers; `q/k/v_input_layernorm` in full ones.
    pub norms_in: Vec<Tensor>,
    pub post_attention_layernorm: Tensor,
    pub attn: AttnWeights,
    pub post_attention_convolution: Tensor,
    pub post_mlp_convolution: Tensor,
    pub router_w: Tensor,
    pub router_bias: Tensor,
    /// `router_bias` read to the host once, at load.
    pub router_bias_host: Vec<f32>,
    /// Loop layers only (`controllers.*` in the file).
    pub message_gates: Option<MessageGates>,
}

fn conv_weight(ck: &NativeCheckpoint, name: &str, dev: &Device) -> Result<Tensor> {
    let w = ck.f32(name, dev)?; // [C, 1, K]
    let (c, one, k) = w.dims3()?;
    if one != 1 {
        candle_core::bail!("{name}: [{c}, {one}, {k}], expected a depthwise kernel");
    }
    w.reshape((c, k))
}

impl LayerWeights {
    /// Load layer `l` from the native file, names per `conversion_mapping.py`
    /// lines 353-416 (`inkling2_text`) and 2679-2706 (`inkling2_loop_text`).
    pub fn load(ck: &NativeCheckpoint, cfg: &InklingConfig, l: usize, dev: &Device) -> Result<Self> {
        Self::load_impl(ck, cfg, l, dev, false)
    }

    /// [`Self::load`] with the attention through the kernels of
    /// `super::attn_dense` (`kernels`, compiled once for the model): the seven
    /// attention matrices are read from the file in BF16 straight into the
    /// kernels' buffers, and no f32 copy of them is put on `dev`. With `row`,
    /// the scores over the cache run through `super::attn_scores` too.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn load_dense(
        ck: &NativeCheckpoint,
        cfg: &InklingConfig,
        l: usize,
        dev: &Device,
        kernels: &std::sync::Arc<super::attn_dense::AttnDense>,
        row: Option<&std::sync::Arc<super::attn_scores::AttnRow>>,
    ) -> Result<Self> {
        let mut lw = Self::load_impl(ck, cfg, l, dev, true)?;
        let a = &mut lw.attn;
        let w = super::attn_dense::AttnDenseWeights::new(a, cfg, dev)?;
        let empty = Tensor::zeros(0, DType::F32, &Device::Cpu)?;
        for m in [&mut a.wq, &mut a.wr, &mut a.sink, &mut a.gate, &mut a.wk, &mut a.wv, &mut a.wo] {
            *m = empty.clone();
        }
        let mut fused = super::attn_dense::FusedAttn::new(kernels.clone(), w, dev);
        fused.row = row.cloned();
        a.dense = Some(fused);
        Ok(lw)
    }

    /// The body of [`Self::load`]; `host_bf16` reads the seven attention
    /// matrices as the file stores them, on the host, for [`Self::load_dense`].
    fn load_impl(ck: &NativeCheckpoint, cfg: &InklingConfig, l: usize, dev: &Device, host_bf16: bool) -> Result<Self> {
        if l >= cfg.num_hidden_layers {
            candle_core::bail!("layer {l}: the model has {} layers", cfg.num_hidden_layers);
        }
        let p = |s: &str| format!("layers.{l}.{s}");
        let f = |s: &str| ck.f32(&p(s), dev);
        let big = |s: &str| if host_bf16 { ck.raw(&p(s), &Device::Cpu) } else { f(s) };
        let sliding = cfg.is_sliding(l);
        let (heads, kv_heads, head_dim) = cfg.attn_shape(l);
        let kv_conv = |which: &str| -> Result<KvConv> {
            if sliding {
                Ok(KvConv::Static(conv_weight(ck, &p(&format!("attn.{which}_sconv.weight")), dev)?))
            } else {
                Ok(KvConv::Dynamic {
                    static_w: f(&format!("attn.{which}_sconv.static_w"))?,
                    weight_proj: f(&format!("attn.{which}_sconv.weight_proj"))?,
                })
            }
        };
        let attn = AttnWeights {
            heads,
            kv_heads,
            head_dim,
            sliding,
            rel_extent: cfg.layer_rel_extent(l),
            wq: big("attn.wq_du.weight")?,
            q_scale: f("attn.q_l2_clip.scale")?,
            wk: big("attn.wk_dv.weight")?,
            k_conv: kv_conv("k")?,
            wv: big("attn.wv_dv.weight")?,
            v_conv: kv_conv("v")?,
            gate: big("attn.gate_attn.weight")?,
            wo: big("attn.wo_ud.weight")?,
            sink: big("attn.attn_sink_proj.weight")?,
            wr: big("attn.wr_du.weight")?,
            rel_scale: f("attn.rel_l2_clip.scale")?,
            rel_proj: f("attn.rel_logits_proj.proj")?,
            #[cfg(all(target_os = "macos", feature = "metal"))]
            dense: None,
        };
        // `attn_res.attn_queries.{i}` is layer i+1's (Inkling2IndexOffsetWeightRename, offset 1).
        let attention_input = if l > 0 {
            let q = ck.f32(&format!("attn_res.attn_queries.{}", l - 1), dev)?;
            let q = if q.rank() == 1 { q.unsqueeze(0)? } else { q };
            let want = if sliding { 1 } else { 3 };
            if q.dim(0)? != want {
                candle_core::bail!("layer {l}: {} attention-input queries, expected {want}", q.dim(0)?);
            }
            Some(q)
        } else {
            None
        };
        let norms_in = if sliding {
            vec![f("attn_norm.weight")?]
        } else {
            vec![f("q_input_norm.weight")?, f("k_input_norm.weight")?, f("v_input_norm.weight")?]
        };
        let message_gates = if cfg.is_loop_layer(l) {
            let gate = |which: &str| -> Result<MessageGate> {
                Ok(MessageGate {
                    down: f(&format!("controllers.{which}.down_weight"))?,
                    forget: f(&format!("controllers.{which}.forget_weight"))?,
                    gain: f(&format!("controllers.{which}.gain_weight"))?,
                })
            };
            // `controllers.attn` is the query gate (conversion_mapping.py line 2685)
            Some(MessageGates { q: gate("attn")?, k: gate("k")?, v: gate("v")?, mlp: gate("mlp")? })
        } else {
            None
        };
        Ok(Self {
            index: l,
            sliding,
            boundary: cfg.is_block_boundary(l),
            attention_input,
            mlp_input: ck.f32(&format!("attn_res.mlp_queries.{l}"), dev)?.unsqueeze(0)?,
            norms_in,
            post_attention_layernorm: f("mlp_norm.weight")?,
            attn,
            post_attention_convolution: conv_weight(ck, &p("attn_sconv.weight"), dev)?,
            post_mlp_convolution: conv_weight(ck, &p("mlp_sconv.weight"), dev)?,
            router_w: f("mlp.gate.weight")?,
            router_bias_host: f("mlp.gate.bias")?.to_vec1::<f32>()?,
            router_bias: f("mlp.gate.bias")?,
            message_gates,
        })
    }

    /// One visit of the decoder layer: sliding (lines 886-910, loop variant
    /// 957-997) or full (lines 1032-1062, loop variant 1111-1149).
    /// `block_states` are those the model holds before this layer; `messages`
    /// (query, key, value, MLP) are given in the loop passes after the first.
    /// Returns the layer output and the routing of its MoE.
    pub fn forward(
        &self,
        cfg: &InklingConfig,
        h: &Tensor,
        block_states: &[Tensor],
        messages: Option<&[Tensor]>,
        experts: &dyn MoeSource,
        cache: &mut SlotCache,
    ) -> Result<(Tensor, Routing)> {
        self.forward_chunked(cfg, h, block_states, messages, experts, cache, DEFAULT_ATTN_CHUNK)
    }

    /// [`Self::forward`] with the attention scores taken `attn_chunk` query
    /// rows at a time ([`AttnWeights::forward_chunked`]).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_chunked(
        &self,
        cfg: &InklingConfig,
        h: &Tensor,
        block_states: &[Tensor],
        messages: Option<&[Tensor]>,
        experts: &dyn MoeSource,
        cache: &mut SlotCache,
        attn_chunk: usize,
    ) -> Result<(Tensor, Routing)> {
        let eps = cfg.rms_norm_eps;
        let dev = h.device();
        let gates = match (messages, &self.message_gates) {
            (None, _) => None,
            (Some(m), Some(g)) if m.len() == 4 => Some((m, g)),
            _ => candle_core::bail!("layer {}: messages given to a layer without message gates", self.index),
        };
        // a slot the fused glue stepped hands its two rings back first
        cache.settle_rings()?;
        let block_residual = h;
        let attn_out = if self.sliding {
            let x = match &self.attention_input {
                Some(q) => block_attn_res(q, block_states, h)?.remove(0),
                None => h.clone(),
            };
            let w = &self.norms_in[0];
            match gates {
                // lines 961-962
                None => {
                    let xn = rms_norm(&x, w, eps)?;
                    mark(dev, Phase::NormsGates)?;
                    self.attn.forward_chunked(cfg, &xn, &xn, &xn, cache, attn_chunk)?
                }
                // lines 964-967: one input, three gated copies, one norm
                Some((m, g)) => {
                    let xq = rms_norm(&g.q.forward(&x, &m[0], eps)?, w, eps)?;
                    let xk = rms_norm(&g.k.forward(&x, &m[1], eps)?, w, eps)?;
                    let xv = rms_norm(&g.v.forward(&x, &m[2], eps)?, w, eps)?;
                    mark(dev, Phase::NormsGates)?;
                    self.attn.forward_chunked(cfg, &xq, &xk, &xv, cache, attn_chunk)?
                }
            }
        } else {
            let (mut xq, mut xk, mut xv) = match &self.attention_input {
                Some(q) => {
                    let mut v = block_attn_res(q, block_states, h)?;
                    let xv = v.pop().expect("3 queries");
                    let xk = v.pop().expect("3 queries");
                    (v.pop().expect("3 queries"), xk, xv)
                }
                None => (h.clone(), h.clone(), h.clone()),
            };
            // lines 1115-1119
            if let Some((m, g)) = gates {
                xq = g.q.forward(&xq, &m[0], eps)?;
                xk = g.k.forward(&xk, &m[1], eps)?;
                xv = g.v.forward(&xv, &m[2], eps)?;
            }
            let (xq, xk, xv) = (
                rms_norm(&xq, &self.norms_in[0], eps)?,
                rms_norm(&xk, &self.norms_in[1], eps)?,
                rms_norm(&xv, &self.norms_in[2], eps)?,
            );
            mark(dev, Phase::NormsGates)?;
            self.attn.forward_chunked(cfg, &xq, &xk, &xv, cache, attn_chunk)?
        };
        let a = short_conv(&attn_out, &self.post_attention_convolution, &mut cache.conv_attn)?;
        mark(dev, Phase::Convolutions)?;

        // lines 901-905: at a block boundary the layer input becomes a block state and the residual restarts
        let mut states: Vec<Tensor> = block_states.to_vec();
        let residual = if self.boundary {
            states.push(block_residual.clone());
            a
        } else {
            (block_residual + a)?
        };

        let mut m_in = block_attn_res(&self.mlp_input, &states, &residual)?.remove(0);
        if let Some((m, g)) = gates {
            m_in = g.mlp.forward(&m_in, &m[3], eps)?; // lines 991-992
        }
        let xn = rms_norm(&m_in, &self.post_attention_layernorm, eps)?;
        mark(dev, Phase::NormsGates)?;
        let routing =
            route_host_bias(&xn, &self.router_w, &self.router_bias_host, cfg.num_experts_per_tok, cfg.route_scale)?;
        mark(dev, Phase::Routing)?;
        let m = experts.moe(&xn, &routing, cfg)?;
        mark(dev, Phase::Moe)?;
        let m = short_conv(&m, &self.post_mlp_convolution, &mut cache.conv_mlp)?;
        mark(dev, Phase::Convolutions)?;
        let out = (residual + m)?;
        mark(dev, Phase::NormsGates)?;
        Ok((out, routing))
    }

    /// [`Self::forward_chunked`] for one row on Metal, its small steps
    /// through the fused glue kernels (`super::glue`, plan item D7): `pre`
    /// for the block residual, message gates and input norms; the attention
    /// as candle runs it; `post_attn` for the post-attention convolution, the
    /// residual, the MLP input and the router scores; `top8` for the routing,
    /// read back once when `want_routing` asks for it and handed to the MoE
    /// on the device otherwise (the returned routing is then empty); the MoE
    /// as the source runs it; `post_mlp` for the post-MLP convolution and the
    /// residual.
    ///
    /// The post-attention and post-MLP states move into two device rings,
    /// filled from what the slot held (a prompt's block, a candle step's
    /// ring, or zeros). Class B against the candle glue: `pre`, the normed
    /// MLP input and the scores sum in another order.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[allow(clippy::too_many_arguments)]
    pub fn forward_fused(
        &self,
        cfg: &InklingConfig,
        h: &Tensor,
        block_states: &[Tensor],
        messages: Option<&[Tensor]>,
        experts: &dyn MoeSource,
        cache: &mut SlotCache,
        glue: &std::sync::Arc<super::glue::Glue>,
        want_routing: bool,
    ) -> Result<(Tensor, Routing)> {
        use super::glue::{FusedRings, GateW, PostAttnInputs, PreInputs};
        fn gw(g: &MessageGate) -> GateW<'_> {
            GateW { down: &g.down, forget: &g.forget, gain: &g.gain }
        }
        let eps = cfg.rms_norm_eps;
        let dev = h.device();
        if h.dim(0)? != 1 {
            candle_core::bail!("layer {}: the fused glue takes one row, not {}", self.index, h.dim(0)?);
        }
        let gates = match (messages, &self.message_gates) {
            (None, _) => None,
            (Some(m), Some(g)) if m.len() == 4 => Some((m, g)),
            _ => candle_core::bail!("layer {}: messages given to a layer without message gates", self.index),
        };
        let c = self.post_attention_convolution.dim(0)?;
        let (k_attn, k_mlp) = (self.post_attention_convolution.dim(1)?, self.post_mlp_convolution.dim(1)?);
        if cache.fused.is_none() {
            let a = cache.conv_attn.block(k_attn)?;
            let m = cache.conv_mlp.block(k_mlp)?;
            cache.fused = Some(FusedRings::new(glue.clone(), dev, c, k_attn, a.as_ref(), k_mlp, m.as_ref())?);
        }
        let h = h.contiguous()?;
        let states = block_states.iter().map(|s| s.contiguous()).collect::<Result<Vec<_>>>()?;
        let msgs = match gates {
            Some((m, _)) => Some(m.iter().map(|t| t.contiguous()).collect::<Result<Vec<_>>>()?),
            None => None,
        };

        let pre = PreInputs {
            h: &h,
            states: &states,
            queries: self.attention_input.as_ref(),
            norms: &self.norms_in,
            gates: match (&msgs, gates) {
                (Some(m), Some((_, g))) => Some(([&m[0], &m[1], &m[2]], [gw(&g.q), gw(&g.k), gw(&g.v)])),
                _ => None,
            },
            sliding: self.sliding,
            eps,
        };
        let x = glue.pre(&pre)?;
        mark(dev, Phase::NormsGates)?;
        let attn_out = if x.dim(0)? == 1 {
            self.attn.forward_chunked(cfg, &x, &x, &x, cache, DEFAULT_ATTN_CHUNK)?
        } else {
            let (xq, xk, xv) = (x.narrow(0, 0, 1)?, x.narrow(0, 1, 1)?, x.narrow(0, 2, 1)?);
            self.attn.forward_chunked(cfg, &xq, &xk, &xv, cache, DEFAULT_ATTN_CHUNK)?
        };
        let attn_out = attn_out.contiguous()?;

        let rings = cache.fused.as_mut().expect("filled above");
        let post = PostAttnInputs {
            attn_out: &attn_out,
            h: &h,
            states: &states,
            boundary: self.boundary,
            conv_w: &self.post_attention_convolution,
            mlp_query: &self.mlp_input,
            gate: match (&msgs, gates) {
                (Some(m), Some((_, g))) => Some((&m[3], gw(&g.mlp))),
                _ => None,
            },
            norm: &self.post_attention_layernorm,
            router_w: &self.router_w,
            eps,
        };
        let pa = glue.post_attn(&post, &mut rings.attn)?;
        mark(dev, Phase::NormsGates)?;
        let k = cfg.num_experts_per_tok;
        let top = glue.top8(&pa.scores, &self.router_bias, k, cfg.route_scale)?;
        // Without a caller for the routing, hand it to the MoE on the device:
        // the host waits on nothing in this visit. Same ids, same weights.
        let on_device = if want_routing {
            None
        } else {
            let sel = glue.sel(&top, k)?;
            mark(dev, Phase::Routing)?;
            experts.moe_routed_on_device(&pa.xn, &sel, pa.scores.elem_count(), cfg)?
        };
        let (m, routing) = match on_device {
            Some(m) => (m.contiguous()?, Routing { indices: vec![], weights: vec![], margin: vec![], scores: vec![] }),
            None => {
                let r = top.read(k)?;
                let routing =
                    Routing { indices: vec![r.ids], weights: vec![r.weights], margin: vec![r.margin], scores: vec![r.sel] };
                mark(dev, Phase::Routing)?;
                (experts.moe(&pa.xn, &routing, cfg)?.contiguous()?, routing)
            }
        };
        mark(dev, Phase::Moe)?;
        let rings = cache.fused.as_mut().expect("filled above");
        let out = glue.post_mlp(&m, &pa.residual, &self.post_mlp_convolution, &mut rings.mlp)?;
        mark(dev, Phase::Convolutions)?;
        Ok((out, routing))
    }
}

/// `Inkling2LoopNormedEmbedding`, lines 1240-1246: the embedding row, then an RMS norm with weight.
pub fn embed(ck: &NativeCheckpoint, cfg: &InklingConfig, ids: &[u32], dev: &Device) -> Result<Tensor> {
    let rows = ck.embed_rows(ids, dev)?;
    let w = ck.f32("embed_norm.weight", dev)?;
    rms_norm(&rows, &w, cfg.rms_norm_eps)
}

/// Max absolute difference and the max absolute value of `reference`, both f32 tensors of one shape,
/// compared on the CPU whatever device each lives on.
pub fn max_abs_err(ours: &Tensor, reference: &Tensor) -> Result<(f32, f32)> {
    let ours = ours.to_device(&Device::Cpu)?;
    let reference = reference.to_device(&Device::Cpu)?;
    let d = (ours.to_dtype(DType::F32)? - reference.to_dtype(DType::F32)?)?.abs()?.flatten_all()?.max(0)?;
    let r = reference.to_dtype(DType::F32)?.abs()?.flatten_all()?.max(0)?;
    Ok((d.to_scalar()?, r.to_scalar()?))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_cfg() -> InklingConfig {
        let v = serde_json::json!({
            "hidden_size": 48, "num_hidden_layers": 12, "vocab_size": 64, "unpadded_vocab_size": 60,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
            "swa_head_dim": 8, "swa_num_attention_heads": 4, "swa_num_key_value_heads": 2,
            "sliding_window_size": 4, "d_rel": 4, "rel_extent": 6,
            "log_scaling_n_floor": 8192, "log_scaling_alpha": 0.1, "rms_norm_eps": 1e-6,
            "qk_l2_clip_threshold": 22.6, "rel_l2_clip_threshold": 8.0, "sconv_kernel_size": 4,
            "final_logit_softcapping": 30.0, "n_layers_per_attn_res_block": 6,
            "n_routed_experts": 8, "num_experts_per_tok": 2, "intermediate_size": 24,
            "mlp_gate_softcap": 4.0, "mlp_up_softcap": 25.0, "route_scale": 8.0,
            "loop_config": {"core_start": 6, "core_end": 12, "num_passes": 3, "gate_rank": 4}
        });
        let path = std::env::temp_dir().join(format!("inkling_tiny_{}.json", std::process::id()));
        std::fs::write(&path, v.to_string()).unwrap();
        let c = InklingConfig::from_file(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        c
    }

    struct RandExperts(Vec<(Tensor, Tensor)>);
    impl ExpertSource for RandExperts {
        fn count(&self) -> usize {
            self.0.len()
        }
        fn expert(&self, e: usize, _dev: &Device) -> Result<(Tensor, Tensor)> {
            Ok(self.0[e].clone())
        }
    }

    fn r(shape: &[usize]) -> Tensor {
        Tensor::randn(0f32, 0.2, shape, &Device::Cpu).unwrap()
    }

    fn layer(cfg: &InklingConfig, l: usize) -> LayerWeights {
        let h = cfg.hidden_size;
        let sliding = cfg.is_sliding(l);
        let (nh, nkv, hd) = cfg.attn_shape(l);
        let conv = |c: usize| {
            if sliding {
                KvConv::Static(r(&[c, 4]))
            } else {
                KvConv::Dynamic { static_w: r(&[4, c]), weight_proj: r(&[nkv * 4, h]) }
            }
        };
        let rank = cfg.loop_config.gate_rank;
        let mg = || MessageGate { down: r(&[rank, 2 * h]), forget: r(&[h, rank]), gain: r(&[h, rank]) };
        let mut lw = LayerWeights {
            index: l,
            sliding,
            boundary: cfg.is_block_boundary(l),
            attention_input: if l == 0 { None } else { Some(r(&[if sliding { 1 } else { 3 }, h])) },
            mlp_input: r(&[1, h]),
            norms_in: (0..if sliding { 1 } else { 3 }).map(|_| r(&[h])).collect(),
            post_attention_layernorm: r(&[h]),
            attn: AttnWeights {
                heads: nh,
                kv_heads: nkv,
                head_dim: hd,
                sliding,
                rel_extent: cfg.layer_rel_extent(l),
                wq: r(&[nh * hd, h]),
                q_scale: r(&[nh, hd]),
                wk: r(&[nkv * hd, h]),
                k_conv: conv(nkv * hd),
                wv: r(&[nkv * hd, h]),
                v_conv: conv(nkv * hd),
                gate: r(&[nh * hd, h]),
                wo: r(&[h, nh * hd]),
                sink: r(&[nh, h]),
                wr: r(&[nh * cfg.d_rel, h]),
                rel_scale: r(&[nh, cfg.d_rel]),
                rel_proj: r(&[cfg.d_rel, cfg.layer_rel_extent(l)]),
                #[cfg(all(target_os = "macos", feature = "metal"))]
                dense: None,
            },
            post_attention_convolution: r(&[h, 4]),
            post_mlp_convolution: r(&[h, 4]),
            router_w: r(&[cfg.n_routed_experts, h]),
            router_bias_host: vec![],
            router_bias: r(&[cfg.n_routed_experts]),
            message_gates: if cfg.is_loop_layer(l) {
                Some(MessageGates { q: mg(), k: mg(), v: mg(), mlp: mg() })
            } else {
                None
            },
        };
        lw.router_bias_host = lw.router_bias.to_vec1().unwrap();
        lw
    }

    /// Layers 0 to 6 of the tiny config on `x`, layers 6 (sliding loop) and
    /// 11 (full loop) with messages; one slot cache per layer.
    fn run_tiny(
        cfg: &InklingConfig,
        layers: &[LayerWeights],
        ex: &RandExperts,
        x: &Tensor,
        msgs: &[Tensor],
        caches: &mut [SlotCache],
    ) -> Result<Tensor> {
        let mut h = x.clone();
        let mut states = vec![];
        let t = x.dim(0)?;
        let msgs: Vec<Tensor> = msgs.iter().map(|m| m.narrow(0, m.dim(0).unwrap() - t, t).unwrap()).collect();
        for (lw, c) in layers.iter().zip(caches.iter_mut()) {
            let m = if lw.message_gates.is_some() { Some(&msgs[..]) } else { None };
            let (next, routing) = lw.forward(cfg, &h, &states, m, ex, c)?;
            assert_eq!(routing.indices.len(), t);
            if lw.boundary {
                states.push(h.clone());
            }
            h = next;
        }
        Ok(h)
    }

    fn tiny_stack() -> (InklingConfig, Vec<LayerWeights>, RandExperts) {
        let cfg = tiny_cfg();
        let (h, i) = (cfg.hidden_size, cfg.intermediate_size);
        let ex = RandExperts((0..cfg.n_routed_experts).map(|_| (r(&[2 * i, h]), r(&[h, i]))).collect());
        let layers: Vec<LayerWeights> = [0, 1, 2, 3, 4, 5, 6, 11].iter().map(|&l| layer(&cfg, l)).collect();
        (cfg, layers, ex)
    }

    /// Pre-loop layers 0 (sliding, boundary) to 5 (full) and two loop layers
    /// with messages run end to end on random weights, finite, and causal:
    /// changing the last token moves no earlier row beyond rounding.
    #[test]
    fn tiny_stack_runs_and_is_causal() -> Result<()> {
        let (cfg, layers, ex) = tiny_stack();
        let h = cfg.hidden_size;
        let t = 7;
        let x = r(&[t, h]);
        let msgs: Vec<Tensor> = (0..4).map(|_| r(&[t, h])).collect();
        let fresh = || vec![SlotCache::default(); layers.len()];
        let y = run_tiny(&cfg, &layers, &ex, &x, &msgs, &mut fresh())?;
        assert!(y.flatten_all()?.to_vec1::<f32>()?.iter().all(|v| v.is_finite()));
        let x2 = Tensor::cat(&[x.narrow(0, 0, t - 1)?, r(&[1, h])], 0)?;
        let y2 = run_tiny(&cfg, &layers, &ex, &x2, &msgs, &mut fresh())?;
        // Not bit for bit: when the last token routes elsewhere, an expert's
        // token batch changes size and the matmul may round its rows
        // differently. A causal leak moves rows by O(1), not by an ulp.
        let (d, scale) = max_abs_err(&y.narrow(0, 0, t - 1)?, &y2.narrow(0, 0, t - 1)?)?;
        assert!(d <= 1e-6 * scale, "an earlier row moved by {d:e} when only the last token changed");
        Ok(())
    }

    /// The decode path: a prompt of `t - 2` rows then two single-row steps on
    /// the same slot caches give the rows a single call over `t` gives, past
    /// the sliding window (4) and the full layers' relative extent (6).
    #[test]
    fn steps_on_the_cache_match_one_call() -> Result<()> {
        let (cfg, layers, ex) = tiny_stack();
        let h = cfg.hidden_size;
        let t = 9;
        let x = r(&[t, h]);
        let msgs: Vec<Tensor> = (0..4).map(|_| r(&[t, h])).collect();
        let whole = run_tiny(&cfg, &layers, &ex, &x, &msgs, &mut vec![SlotCache::default(); layers.len()])?;
        let mut caches = vec![SlotCache::default(); layers.len()];
        let msg_upto = |n: usize| -> Vec<Tensor> { msgs.iter().map(|m| m.narrow(0, 0, n).unwrap()).collect() };
        let mut rows = vec![run_tiny(&cfg, &layers, &ex, &x.narrow(0, 0, t - 2)?, &msg_upto(t - 2), &mut caches)?];
        for n in [t - 1, t] {
            rows.push(run_tiny(&cfg, &layers, &ex, &x.narrow(0, n - 1, 1)?, &msg_upto(n), &mut caches)?);
        }
        assert_eq!(caches[0].len(), t);
        let stepped = Tensor::cat(&rows, 0)?;
        let (d, scale) = max_abs_err(&stepped, &whole)?;
        assert!(d <= 1e-5 * scale, "decode drifts from the single call: {d:e} of {scale:e}");
        Ok(())
    }

    /// The ops the forward needs exist on Metal (`gather`, `index_add`,
    /// `index_select`, the BF16 widening of the head) and the tiny stack, a
    /// prompt then two steps, agrees with the CPU. Uses the GPU: run it alone,
    /// under the guard's lock.
    #[cfg(feature = "metal")]
    #[test]
    #[ignore = "uses the GPU; run under the guard's lock"]
    fn tiny_stack_on_metal_matches_cpu() -> Result<()> {
        let (cfg, layers, ex) = tiny_stack();
        let mtl = Device::new_metal(0)?;
        let to = |t: &Tensor| t.to_device(&mtl).unwrap();
        let mv = |w: &Tensor| to(w);
        let move_layer = |lw: &LayerWeights| LayerWeights {
            index: lw.index,
            sliding: lw.sliding,
            boundary: lw.boundary,
            attention_input: lw.attention_input.as_ref().map(mv),
            mlp_input: mv(&lw.mlp_input),
            norms_in: lw.norms_in.iter().map(mv).collect(),
            post_attention_layernorm: mv(&lw.post_attention_layernorm),
            attn: AttnWeights {
                heads: lw.attn.heads,
                kv_heads: lw.attn.kv_heads,
                head_dim: lw.attn.head_dim,
                sliding: lw.attn.sliding,
                rel_extent: lw.attn.rel_extent,
                wq: mv(&lw.attn.wq),
                q_scale: mv(&lw.attn.q_scale),
                wk: mv(&lw.attn.wk),
                k_conv: match &lw.attn.k_conv {
                    KvConv::Static(w) => KvConv::Static(mv(w)),
                    KvConv::Dynamic { static_w, weight_proj } => {
                        KvConv::Dynamic { static_w: mv(static_w), weight_proj: mv(weight_proj) }
                    }
                },
                wv: mv(&lw.attn.wv),
                v_conv: match &lw.attn.v_conv {
                    KvConv::Static(w) => KvConv::Static(mv(w)),
                    KvConv::Dynamic { static_w, weight_proj } => {
                        KvConv::Dynamic { static_w: mv(static_w), weight_proj: mv(weight_proj) }
                    }
                },
                gate: mv(&lw.attn.gate),
                wo: mv(&lw.attn.wo),
                sink: mv(&lw.attn.sink),
                wr: mv(&lw.attn.wr),
                rel_scale: mv(&lw.attn.rel_scale),
                rel_proj: mv(&lw.attn.rel_proj),
                #[cfg(all(target_os = "macos", feature = "metal"))]
                dense: None,
            },
            post_attention_convolution: mv(&lw.post_attention_convolution),
            post_mlp_convolution: mv(&lw.post_mlp_convolution),
            router_w: mv(&lw.router_w),
            router_bias: mv(&lw.router_bias),
            router_bias_host: lw.router_bias_host.clone(),
            message_gates: lw.message_gates.as_ref().map(|g| {
                let m = |x: &MessageGate| MessageGate { down: mv(&x.down), forget: mv(&x.forget), gain: mv(&x.gain) };
                MessageGates { q: m(&g.q), k: m(&g.k), v: m(&g.v), mlp: m(&g.mlp) }
            }),
        };
        let mlayers: Vec<LayerWeights> = layers.iter().map(move_layer).collect();
        let mex = RandExperts(ex.0.iter().map(|(a, b)| (to(a), to(b))).collect());
        let h = cfg.hidden_size;
        let t = 9;
        let x = r(&[t, h]);
        let msgs: Vec<Tensor> = (0..4).map(|_| r(&[t, h])).collect();
        let cpu = run_tiny(&cfg, &layers, &ex, &x, &msgs, &mut vec![SlotCache::default(); layers.len()])?;
        let mmsgs: Vec<Tensor> = msgs.iter().map(to).collect();
        let mut caches = vec![SlotCache::default(); layers.len()];
        let upto = |n: usize| -> Vec<Tensor> { mmsgs.iter().map(|m| m.narrow(0, 0, n).unwrap()).collect() };
        let mut rows = vec![run_tiny(&cfg, &mlayers, &mex, &to(&x.narrow(0, 0, t - 2)?), &upto(t - 2), &mut caches)?];
        for n in [t - 1, t] {
            rows.push(run_tiny(&cfg, &mlayers, &mex, &to(&x.narrow(0, n - 1, 1)?), &upto(n), &mut caches)?);
        }
        let metal = Tensor::cat(&rows, 0)?.to_device(&Device::Cpu)?;
        let (d, scale) = max_abs_err(&metal, &cpu)?;
        assert!(d <= 1e-4 * scale, "Metal drifts from the CPU: {d:e} of {scale:e}");
        // the head's BF16 widening
        let w = Tensor::randn(0f32, 1.0, (8, h), &Device::Cpu)?.to_dtype(DType::BF16)?;
        let a = w.to_dtype(DType::F32)?;
        let b = w.to_device(&mtl)?.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
        assert_eq!(max_abs_err(&a, &b)?.0, 0.0, "BF16 to f32 on Metal");
        Ok(())
    }

    /// The static convolution against a direct transcription of `F.conv1d`
    /// with left padding K - 1 (lines 316-322), plus residual and scale.
    #[test]
    fn short_conv_matches_direct_sum() -> Result<()> {
        let (t, c) = (6, 3);
        let x = r(&[t, c]);
        let w = r(&[c, 4]);
        let ours = short_conv(&x, &w, &mut ConvState::Empty)?.to_vec2::<f32>()?;
        let xv = x.to_vec2::<f32>()?;
        let wv = w.to_vec2::<f32>()?;
        for ti in 0..t {
            for ci in 0..c {
                let mut s = 0f32;
                for (k, wk) in wv[ci].iter().enumerate() {
                    let src = ti as i64 + k as i64 - 3;
                    if src >= 0 {
                        s += wk * xv[src as usize][ci];
                    }
                }
                let want = (s + xv[ti][ci]) / 5f32.sqrt();
                assert!((ours[ti][ci] - want).abs() <= 1e-5, "t {ti} c {ci}: {} vs {want}", ours[ti][ci]);
            }
        }
        Ok(())
    }
}
