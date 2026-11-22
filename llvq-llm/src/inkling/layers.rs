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

/// The convolution context: the `k - 1` inputs before `x` (zeros when
/// `state` is empty) then `x`, `[k - 1 + T, C]`. `state` becomes the last
/// `k - 1` rows (`update_conv_state` / `conv_state.copy_`, lines 297, 379, 438).
fn conv_context(x: &Tensor, k: usize, state: &mut Option<Tensor>) -> Result<Tensor> {
    let t = x.dim(0)?;
    let ctx = match state.as_ref() {
        Some(s) => Tensor::cat(&[s, x], 0)?,
        None => x.pad_with_zeros(0, k - 1, 0)?,
    };
    *state = Some(ctx.narrow(0, t, k - 1)?.contiguous()?);
    Ok(ctx)
}

/// `Inkling2LoopShortConvolution.forward`, lines 362-393: a depthwise causal
/// convolution, `w` `[C, K]` from the file's `[C, 1, K]`, tap `K - 1` on the
/// current token (`F.conv1d` with left padding `K - 1`, lines 316-322, or the
/// cached history in decode, lines 373-376), then the residual and the output
/// scale `1 / sqrt(K + 1)` (line 392).
pub fn short_conv(x: &Tensor, w: &Tensor, state: &mut Option<Tensor>) -> Result<Tensor> {
    let k = w.dim(1)?;
    let t = x.dim(0)?;
    let ctx = conv_context(x, k, state)?;
    let mut acc = x.zeros_like()?;
    for j in 0..k {
        let tap = w.i((.., j))?; // [C]
        acc = (acc + ctx.narrow(0, j, t)?.broadcast_mul(&tap)?)?;
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
    state: &mut Option<Tensor>,
) -> Result<Tensor> {
    let (t, c) = x.dims2()?;
    let k = static_w.dim(0)?;
    let dw = linear(dynamic_input, weight_proj)?.reshape((t, heads, k))?; // line 430-431
    let ctx = conv_context(x, k, state)?;
    let mut acc = x.zeros_like()?;
    for lag in 0..k {
        let d = dw.i((.., .., lag))?.unsqueeze(2)?.broadcast_as((t, heads, c / heads))?.reshape((t, c))?;
        let w = d.broadcast_add(&static_w.i(lag)?)?; // line 432
        acc = (acc + ctx.narrow(0, k - 1 - lag, t)?.mul(&w)?)?; // lines 444-453
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
    let scores = candle_nn::ops::sigmoid(&linear(x, w_gate)?)?.to_vec2::<f32>()?;
    let bias = bias.to_vec1::<f32>()?;
    let mut r = Routing { indices: vec![], weights: vec![], margin: vec![], scores: vec![] };
    for s in &scores {
        let sel: Vec<f32> = s.iter().zip(&bias).map(|(a, b)| a + b).collect();
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

/// `[kv_heads, T, d]` repeated to `[kv_heads * rep, T, d]`, head `h` reading kv head `h / rep` (lines 99-108).
fn repeat_kv(x: &Tensor, rep: usize) -> Result<Tensor> {
    if rep == 1 {
        return Ok(x.clone());
    }
    let (kv, t, d) = x.dims3()?;
    x.unsqueeze(1)?.broadcast_as((kv, rep, t, d))?.reshape((kv * rep, t, d))
}

/// What one layer visit keeps between calls: keys `[kv_heads, S, d]` and
/// values after their norms (what `cache.update` stores, line 612), and the
/// last `K - 1` inputs of the key, value, post-attention and post-MLP
/// convolutions (`conv_idx` 0 to 3, lines 563-566 and 868-873).
#[derive(Debug, Clone, Default)]
pub struct SlotCache {
    pub k: Option<Tensor>,
    pub v: Option<Tensor>,
    pub conv_k: Option<Tensor>,
    pub conv_v: Option<Tensor>,
    pub conv_attn: Option<Tensor>,
    pub conv_mlp: Option<Tensor>,
}

impl SlotCache {
    /// Tokens held.
    pub fn len(&self) -> usize {
        self.k.as_ref().map_or(0, |k| k.dim(1).expect("rank-3 keys"))
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

fn append(cache: &mut Option<Tensor>, x: Tensor) -> Result<Tensor> {
    let all = match cache.take() {
        Some(past) => Tensor::cat(&[&past, &x], 1)?,
        None => x,
    };
    *cache = Some(all.clone());
    Ok(all)
}

impl AttnWeights {
    fn kv(&self, x: &Tensor, w: &Tensor, conv: &KvConv, state: &mut Option<Tensor>) -> Result<Tensor> {
        let s = linear(x, w)?;
        match conv {
            KvConv::Static(cw) => short_conv(&s, cw, state),
            KvConv::Dynamic { static_w, weight_proj } => {
                dynamic_conv(&s, x, static_w, weight_proj, self.kv_heads, state)
            }
        }
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
        let t = xq.dim(0)?;
        let dev = xq.device();
        let offset = cache.len();
        let (nh, nkv, hd) = (self.heads, self.kv_heads, self.head_dim);
        let positions: Vec<usize> = (offset..offset + t).collect();
        let tau = if self.sliding { None } else { Some(log_scale(cfg, &positions, dev)?) };

        // Query: linear, learned scale, L2 clip, and the log scale in full layers (lines 458-475, 190-236).
        let q = linear(xq, &self.wq)?.reshape((t, nh, hd))?.broadcast_mul(&self.q_scale)?;
        let mut q = l2_clip(&q, cfg.qk_l2_clip_threshold)?;
        if let Some(tau) = &tau {
            q = q.broadcast_mul(tau)?;
        }
        let q = q.transpose(0, 1)?.contiguous()?; // [nh, T, hd]

        // Key: linear, convolution, L2 clip without scale. Value: linear, convolution, unweighted RMS norm (lines 515-528).
        let k = self.kv(xk, &self.wk, &self.k_conv, &mut cache.conv_k)?.reshape((t, nkv, hd))?;
        let k = l2_clip(&k, cfg.qk_l2_clip_threshold)?.transpose(0, 1)?.contiguous()?;
        let v = self.kv(xv, &self.wv, &self.v_conv, &mut cache.conv_v)?.reshape((t, nkv, hd))?;
        let v = rms_unweighted(&v, cfg.rms_norm_eps)?.transpose(0, 1)?.contiguous()?;
        // line 612: the cache returns every key and value it holds
        let k = append(&mut cache.k, k)?;
        let v = append(&mut cache.v, v)?;
        let s = k.dim(1)?;

        // Relative bias (lines 260-270): per-head states, scale, L2 clip at the rel threshold, log scale in full layers.
        let r = linear(xq, &self.wr)?.reshape((t, nh, cfg.d_rel))?.broadcast_mul(&self.rel_scale)?;
        let mut r = l2_clip(&r, cfg.rel_l2_clip_threshold)?;
        if let Some(tau) = &tau {
            r = r.broadcast_mul(tau)?;
        }
        let rel_logits = r.transpose(0, 1)?.contiguous()?.broadcast_matmul(&self.rel_proj)?; // [nh, T, extent]
        let window = if self.sliding { Some(cfg.sliding_window_size) } else { None };
        let bias = score_bias(&rel_logits, offset, s, self.rel_extent, window)?;

        // Scores, the per-token sink logit with zero value (lines 126-139), softmax in f32.
        let rep = nh / nkv;
        let k = repeat_kv(&k, rep)?;
        let v = repeat_kv(&v, rep)?;
        let scores = (q.matmul(&k.transpose(1, 2)?.contiguous()?)? * (hd as f64).powf(-0.5))?.add(&bias)?;
        let sinks = linear(xq, &self.sink)?.t()?.unsqueeze(2)?; // [nh, T, 1], line 619
        let probs = candle_nn::ops::softmax_last_dim(&Tensor::cat(&[&scores, &sinks], 2)?.contiguous()?)?.narrow(2, 0, s)?;
        let a = probs.contiguous()?.matmul(&v)?.transpose(0, 1)?.reshape((t, nh * hd))?;

        // Gated output (lines 540-543).
        let gate = candle_nn::ops::sigmoid(&linear(xq, &self.gate)?)?;
        linear(&a.mul(&gate)?, &self.wo)
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
        if l >= cfg.num_hidden_layers {
            candle_core::bail!("layer {l}: the model has {} layers", cfg.num_hidden_layers);
        }
        let p = |s: &str| format!("layers.{l}.{s}");
        let f = |s: &str| ck.f32(&p(s), dev);
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
            wq: f("attn.wq_du.weight")?,
            q_scale: f("attn.q_l2_clip.scale")?,
            wk: f("attn.wk_dv.weight")?,
            k_conv: kv_conv("k")?,
            wv: f("attn.wv_dv.weight")?,
            v_conv: kv_conv("v")?,
            gate: f("attn.gate_attn.weight")?,
            wo: f("attn.wo_ud.weight")?,
            sink: f("attn.attn_sink_proj.weight")?,
            wr: f("attn.wr_du.weight")?,
            rel_scale: f("attn.rel_l2_clip.scale")?,
            rel_proj: f("attn.rel_logits_proj.proj")?,
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
        let eps = cfg.rms_norm_eps;
        let gates = match (messages, &self.message_gates) {
            (None, _) => None,
            (Some(m), Some(g)) if m.len() == 4 => Some((m, g)),
            _ => candle_core::bail!("layer {}: messages given to a layer without message gates", self.index),
        };
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
                    self.attn.forward(cfg, &xn, &xn, &xn, cache)?
                }
                // lines 964-967: one input, three gated copies, one norm
                Some((m, g)) => {
                    let xq = rms_norm(&g.q.forward(&x, &m[0], eps)?, w, eps)?;
                    let xk = rms_norm(&g.k.forward(&x, &m[1], eps)?, w, eps)?;
                    let xv = rms_norm(&g.v.forward(&x, &m[2], eps)?, w, eps)?;
                    self.attn.forward(cfg, &xq, &xk, &xv, cache)?
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
            self.attn.forward(
                cfg,
                &rms_norm(&xq, &self.norms_in[0], eps)?,
                &rms_norm(&xk, &self.norms_in[1], eps)?,
                &rms_norm(&xv, &self.norms_in[2], eps)?,
                cache,
            )?
        };
        let a = short_conv(&attn_out, &self.post_attention_convolution, &mut cache.conv_attn)?;

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
        let routing = route(&xn, &self.router_w, &self.router_bias, cfg.num_experts_per_tok, cfg.route_scale)?;
        let m = experts.moe(&xn, &routing, cfg)?;
        let m = short_conv(&m, &self.post_mlp_convolution, &mut cache.conv_mlp)?;
        Ok(((residual + m)?, routing))
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
        LayerWeights {
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
            },
            post_attention_convolution: r(&[h, 4]),
            post_mlp_convolution: r(&[h, 4]),
            router_w: r(&[cfg.n_routed_experts, h]),
            router_bias: r(&[cfg.n_routed_experts]),
            message_gates: if cfg.is_loop_layer(l) {
                Some(MessageGates { q: mg(), k: mg(), v: mg(), mlp: mg() })
            } else {
                None
            },
        }
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
            },
            post_attention_convolution: mv(&lw.post_attention_convolution),
            post_mlp_convolution: mv(&lw.post_mlp_convolution),
            router_w: mv(&lw.router_w),
            router_bias: mv(&lw.router_bias),
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
        let ours = short_conv(&x, &w, &mut None)?.to_vec2::<f32>()?;
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
