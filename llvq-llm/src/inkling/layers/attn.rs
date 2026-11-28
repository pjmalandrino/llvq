//! The attention block: its weights in their three holdings (f32, the kernels' BF16, the CPU's
//! BF16), the score bias and the window switch, and the forward with its chunked and windowed
//! score paths. One file because every piece is read by [`AttnWeights`]'s forward and nothing else.

use candle_core::{Result, Tensor};

use crate::fused_cpu::phases::{timed, ATTN_CACHE, ATTN_PROJ, ATTN_SCORES, CONVS};
use crate::inkling::config::InklingConfig;

use super::phases::{mark, Phase};
use super::{dynamic_conv, l2_clip, linear, log_scale, rms_unweighted, short_conv, ConvState, SlotCache};

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
    #[cfg(ink_glue)]
    pub dense: Option<super::attn_dense::FusedAttn>,
    /// The seven matrices in BF16 on the host for the CPU kernels, when the
    /// loader chose them ([`LayerWeights::load_cpu`](super::LayerWeights::load_cpu)): every forward then
    /// reads them, and the seven f32 tensors above are empty.
    pub cpu: Option<CpuAttnProj>,
}

/// The seven attention matrices of one block in BF16 on the host, as the
/// file stores them, applied by `tetra_cpu::bf16_matvec` (one row, or a few)
/// or widened for candle's matmul (a longer prompt): [`crate::fused_cpu::CpuBf16`].
pub struct CpuAttnProj {
    pub wq: crate::fused_cpu::CpuBf16,
    pub wr: crate::fused_cpu::CpuBf16,
    pub sink: crate::fused_cpu::CpuBf16,
    pub gate: crate::fused_cpu::CpuBf16,
    pub wk: crate::fused_cpu::CpuBf16,
    pub wv: crate::fused_cpu::CpuBf16,
    pub wo: crate::fused_cpu::CpuBf16,
}

impl CpuAttnProj {
    fn refs(&self) -> MatRefs<'_> {
        MatRefs {
            wq: Proj::Cpu(&self.wq),
            wr: Proj::Cpu(&self.wr),
            sink: Proj::Cpu(&self.sink),
            gate: Proj::Cpu(&self.gate),
            wk: Proj::Cpu(&self.wk),
            wv: Proj::Cpu(&self.wv),
            wo: Proj::Cpu(&self.wo),
        }
    }

    /// Host bytes of the seven matrices.
    pub fn bytes(&self) -> u64 {
        [&self.wq, &self.wr, &self.sink, &self.gate, &self.wk, &self.wv, &self.wo].iter().map(|m| m.bytes()).sum()
    }
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

/// The additive bias of the attention scores, `[heads, T, S]`, for `T`
/// queries at positions `offset..offset + T` against `S` keys at positions
/// `0..S`, both counted from the first key the slot holds
/// ([`SlotCache::base`]): only the distance enters. The relative position bias (lines 260-270, distances per lines
/// 1249-1268) plus the causal mask, and for a sliding layer the window (keys
/// at distance `>= window` masked).
fn score_bias(rel_logits: &Tensor, offset: usize, s: usize, extent: usize, window: Option<usize>) -> Result<Tensor> {
    #[cfg(ink_glue)]
    if let Some(b) = super::attn_bias::bias(rel_logits, offset, s, extent, window)? {
        return Ok(b);
    }
    score_bias_host(rel_logits, offset, s, extent, window)
}

/// [`score_bias`] through the host tables, whatever the device kernel's
/// switch: the reference `super::attn_bias` is gated against.
pub fn score_bias_host(rel_logits: &Tensor, offset: usize, s: usize, extent: usize, window: Option<usize>) -> Result<Tensor> {
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

/// Whether the scores of a sliding layer read, chunk by chunk, only the keys
/// the chunk's rows can see ([`set_window_scores`]).
static WINDOW_SCORES: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Chunks scored over their visible keys only, since the process started.
static WINDOW_SCORE_CHUNKS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Score each chunk of a sliding layer's query rows over the keys its rows
/// can see (`true`), or over every key the slot holds, the keys outside the
/// window masked to minus infinity (`false`, the default). For a process:
/// `inkrun --attn-window narrow` turns it on, and `llvq serve --attn-window
/// narrow`, hidden and off by default, on Metal under the fused stack. The
/// dropped keys have probability exactly zero, so each row's value is the
/// same in exact arithmetic; the device's sums run over shorter rows, so the
/// bits are not meant to be the same (prereg `inkling-fenetre-2026-11-27`).
/// The full layers and the one-row score kernel are not touched.
pub fn set_window_scores(on: bool) {
    WINDOW_SCORES.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// Whether [`set_window_scores`] is on.
fn window_scores() -> bool {
    WINDOW_SCORES.load(std::sync::atomic::Ordering::Relaxed)
}

/// Chunks the narrowed path has scored since the process started: what a
/// test reads to see that the switch reached the scores.
pub fn window_score_chunks() -> usize {
    WINDOW_SCORE_CHUNKS.load(std::sync::atomic::Ordering::Relaxed)
}

/// The first key a query row at `pos` sees in a sliding layer of `window`
/// keys: the mask of [`score_bias`] keeps distances `0..window`.
fn first_visible(pos: usize, window: usize) -> usize {
    (pos + 1).saturating_sub(window)
}

/// `[kv_heads, T, d]` repeated to `[kv_heads * rep, T, d]`, head `h` reading kv head `h / rep` (lines 99-108).
fn repeat_kv(x: &Tensor, rep: usize) -> Result<Tensor> {
    if rep == 1 {
        return Ok(x.clone());
    }
    let (kv, t, d) = x.dims3()?;
    x.unsqueeze(1)?.broadcast_as((kv, rep, t, d))?.reshape((kv * rep, t, d))
}

impl AttnWeights {
    fn kv(&self, x: &Tensor, w: Proj, conv: &KvConv, state: &mut ConvState) -> Result<Tensor> {
        let s = timed(ATTN_PROJ, || w.apply(x))?;
        mark(x.device(), Phase::AttnProducts)?;
        let y = timed(CONVS, || match conv {
            KvConv::Static(cw) => short_conv(&s, cw, state),
            KvConv::Dynamic { static_w, weight_proj } => {
                dynamic_conv(&s, x, static_w, weight_proj, self.kv_heads, state)
            }
        })?;
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
        #[cfg(ink_glue)]
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
        if let Some(c) = &self.cpu {
            return self.forward_mats(cfg, &c.refs(), xq, xk, xv, cache, chunk);
        }
        self.forward_mats(cfg, &self.refs(), xq, xk, xv, cache, chunk)
    }

    /// The seven matrices this block holds in f32.
    fn refs(&self) -> MatRefs<'_> {
        MatRefs {
            wq: Proj::F32(&self.wq),
            wr: Proj::F32(&self.wr),
            sink: Proj::F32(&self.sink),
            gate: Proj::F32(&self.gate),
            wk: Proj::F32(&self.wk),
            wv: Proj::F32(&self.wv),
            wo: Proj::F32(&self.wo),
        }
    }

    /// [`Self::forward_chunked`] through candle, the seven products by `m`.
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
        let q = timed(ATTN_PROJ, || m.wq.apply(xq))?.reshape((t, nh, hd))?.broadcast_mul(&self.q_scale)?;
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
        // line 612: the cache returns every key and value it holds (keys as `[nkv, hd, S]`), from position `base`
        let (kt, v) = timed(ATTN_CACHE, || cache.append_kv(&k, &v))?;
        let s = v.dim(1)?;
        let base = cache.base();
        mark(dev, Phase::CacheCopies)?;

        // Relative bias (lines 260-270): per-head states, scale, L2 clip at the rel threshold, log scale in full layers.
        let r = timed(ATTN_PROJ, || m.wr.apply(xq))?.reshape((t, nh, cfg.d_rel))?.broadcast_mul(&self.rel_scale)?;
        let mut r = l2_clip(&r, cfg.rel_l2_clip_threshold)?;
        if let Some(tau) = &tau {
            r = r.broadcast_mul(tau)?;
        }
        let rel_logits = timed(ATTN_PROJ, || r.transpose(0, 1)?.contiguous()?.broadcast_matmul(&self.rel_proj))?; // [nh, T, extent]
        mark(dev, Phase::AttnProducts)?;

        let sinks = || timed(ATTN_PROJ, || m.sink.apply(xq))?.t()?.unsqueeze(2); // [nh, T, 1], line 619
        // The bias reads distances only: the query positions counted from the first key held.
        let a = self.scores(cfg, &q, &kt, &v, &rel_logits, sinks, offset - base, s, t, chunk)?;
        drop((kt, v));

        // Gated output (lines 540-543).
        let gate = candle_nn::ops::sigmoid(&timed(ATTN_PROJ, || m.gate.apply(xq))?)?;
        let gated = a.mul(&gate)?;
        let out = timed(ATTN_PROJ, || m.wo.apply(&gated))?;
        mark(dev, Phase::AttnProducts)?;
        cache.settle_window()?;
        Ok(out)
    }

    /// Scores over every position the slot holds, the per-token sink logit
    /// with zero value (lines 126-139), softmax in f32, and the value
    /// product, `chunk` query rows at a time: `[T, heads * head_dim]`.
    /// `offset` is the first query's position counted from the first key held.
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
        if self.sliding && window_scores() {
            return self.scores_window(cfg, q, kt, v, rel_logits, sinks, offset, s, t, chunk);
        }
        let dev = q.device();
        let (nh, nkv, hd) = (self.heads, self.kv_heads, self.head_dim);
        let window = if self.sliding { Some(cfg.sliding_window_size) } else { None };
        let rep = nh / nkv;
        // The repeat copies the views into contiguous `[nh, hd, S]` and `[nh, S, hd]`: the
        // matrices the score and value products read before, with no transpose copy.
        let (kt, v) = timed(ATTN_CACHE, || -> Result<(Tensor, Tensor)> {
            Ok((repeat_kv(kt, rep)?.contiguous()?, repeat_kv(v, rep)?.contiguous()?))
        })?;
        mark(dev, Phase::CacheCopies)?;
        let sinks = sinks()?;
        let a = timed(ATTN_SCORES, || -> Result<Tensor> {
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
            Ok(if parts.len() == 1 { parts.pop().expect("one part") } else { Tensor::cat(&parts, 0)? })
        })?;
        mark(dev, Phase::AttnScores)?;
        Ok(a)
    }

    /// [`Self::scores`] of a sliding layer under [`set_window_scores`]: each
    /// chunk of query rows reads the keys and values from the first one its
    /// first row sees to its last row's own, and its bias is built for that
    /// range. One copy a call, as the full path makes, of the keys any row of
    /// the call sees (from the first row's first visible key to the last
    /// position held, fewer than the slot holds), keys row-major `[nh, S, hd]`
    /// like the values: each chunk's range is then a view of both copies that
    /// the products read in place, the keys as a transposed operand, with no
    /// copy a chunk.
    #[allow(clippy::too_many_arguments)]
    fn scores_window(
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
        let w = cfg.sliding_window_size;
        let rep = nh / nkv;
        // The rows sit at `offset..offset + t` and the last of them is the last key held.
        if offset + t != s {
            candle_core::bail!("window scores: {t} rows from position {offset} in a slot of {s} positions");
        }
        let from = first_visible(offset, w);
        let span = s - from;
        let (keys, vals) = timed(ATTN_CACHE, || -> Result<(Tensor, Tensor)> {
            let k = kt.narrow(2, from, span)?.transpose(1, 2)?;
            Ok((repeat_kv(&k, rep)?.contiguous()?, repeat_kv(&v.narrow(1, from, span)?, rep)?.contiguous()?))
        })?;
        mark(dev, Phase::CacheCopies)?;
        let sinks = sinks()?;
        let a = timed(ATTN_SCORES, || -> Result<Tensor> {
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
                // Keys `lo..hi` of the copies: from the first key row `p0` sees to row `p0 + n - 1`'s own.
                let p0 = offset + r0;
                let lo = first_visible(p0, w) - from;
                let len = p0 + n - from - lo;
                let kc = keys.narrow(1, lo, len)?.transpose(1, 2)?; // [nh, hd, len]
                let vc = vals.narrow(1, lo, len)?; // [nh, len, hd]
                // The bias reads distances only: the first row counted from the first key of the range.
                let bias = score_bias(&rc, p0 - (from + lo), len, self.rel_extent, Some(w))?;
                let scores = (qc.matmul(&kc)? * (hd as f64).powf(-0.5))?.add(&bias)?;
                let probs =
                    candle_nn::ops::softmax_last_dim(&Tensor::cat(&[&scores, &sc], 2)?.contiguous()?)?.narrow(2, 0, len)?;
                parts.push(probs.contiguous()?.matmul(&vc)?.transpose(0, 1)?.reshape((n, nh * hd))?);
                WINDOW_SCORE_CHUNKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                r0 += n;
            }
            Ok(if parts.len() == 1 { parts.pop().expect("one part") } else { Tensor::cat(&parts, 0)? })
        })?;
        mark(dev, Phase::AttnScores)?;
        Ok(a)
    }

    /// One row through the attention kernels (`super::attn_dense`): `stage_in`
    /// in place of the products and their small steps, the cache append and
    /// the score statements as [`Self::forward_mats`] runs them, `stage_out`
    /// for the gated output. The key and value states move into the slot's
    /// rings, filled from what it held (a prompt's block, a candle step's
    /// ring, or zeros). Class B against the candle attention.
    #[cfg(ink_glue)]
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
        // The row's position counted from the first key held: the kernel reads keys `0..s` of the view.
        let pos = offset - cache.base();
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
                    pos,
                    window,
                })?;
                mark(dev, Phase::AttnScores)?;
                a
            }
            None => {
                let sinks = || Ok(st.sinks.clone());
                self.scores(cfg, &st.q, &kt, &v, &st.rel_logits, sinks, pos, s, 1, chunk)?
            }
        };
        drop((kt, v));
        let out = d.kernels.stage_out(&d.w, &a.contiguous()?, &st.gate)?;
        mark(dev, Phase::AttnProducts)?;
        cache.settle_window()?;
        Ok(out)
    }
}

/// One attention matrix as a candle attention call reads it: an f32
/// tensor (candle's matmul) or a BF16 matrix of the CPU path.
#[derive(Clone, Copy)]
enum Proj<'a> {
    F32(&'a Tensor),
    Cpu(&'a crate::fused_cpu::CpuBf16),
}

impl Proj<'_> {
    /// `x @ w.T` for `x` `[T, in]`.
    fn apply(self, x: &Tensor) -> Result<Tensor> {
        match self {
            Proj::F32(w) => linear(x, w),
            Proj::Cpu(w) => w.forward(x),
        }
    }
}

/// The seven matrices a candle attention call reads, borrowed.
struct MatRefs<'a> {
    wq: Proj<'a>,
    wr: Proj<'a>,
    sink: Proj<'a>,
    gate: Proj<'a>,
    wk: Proj<'a>,
    wv: Proj<'a>,
    wo: Proj<'a>,
}

#[cfg(ink_glue)]
impl AttnMats {
    fn refs(&self) -> MatRefs<'_> {
        MatRefs {
            wq: Proj::F32(&self.wq),
            wr: Proj::F32(&self.wr),
            sink: Proj::F32(&self.sink),
            gate: Proj::F32(&self.gate),
            wk: Proj::F32(&self.wk),
            wv: Proj::F32(&self.wv),
            wo: Proj::F32(&self.wo),
        }
    }
}
