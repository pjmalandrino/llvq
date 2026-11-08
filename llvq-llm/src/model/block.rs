//! One transformer block: RoPE, the attention forms over the cached heads,
//! and the batched projection step.

use super::*;

pub struct Rotary {
    pub(super) sin: Tensor,
    pub(super) cos: Tensor,
}

impl Rotary {
    pub fn new(dtype: DType, cfg: &Config, dev: &Device) -> Result<Self> {
        let dim = cfg.head_dim;
        let inv: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let n = inv.len();
        let inv = Tensor::from_vec(inv, (1, n), dev)?;
        let t = Tensor::arange(0u32, cfg.max_position_embeddings as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((cfg.max_position_embeddings, 1))?;
        let freqs = t.matmul(&inv)?;
        Ok(Self {
            sin: freqs.sin()?.to_dtype(dtype)?,
            cos: freqs.cos()?.to_dtype(dtype)?,
        })
    }

    fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let (_, _, l, _) = q.dims4()?;
        let cos = self.cos.narrow(0, offset, l)?;
        let sin = self.sin.narrow(0, offset, l)?;
        Ok((
            candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?,
            candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?,
        ))
    }
}

/// An attention form that reads the cached heads unexpanded: `q`, `k`, `v`,
/// the mask (or `None` where it would be all zeros), the score scale.
type AttnFn = fn(&Tensor, &Tensor, &Tensor, Option<&Tensor>, f64) -> Result<Tensor>;

/// `q` as `[b, n_kv, groups, l, hd]`, with the shapes it is read against.
///
/// Query head `h` reads key head `h / groups`, the convention of `repeat_kv`,
/// so the `groups` query heads of one key head are contiguous in `q`.
fn group_view(q: &Tensor, k: &Tensor) -> Result<(Tensor, usize, usize)> {
    let (b, n_heads, l, hd) = q.dims4()?;
    let n_kv = k.dim(1)?;
    if n_kv == 0 || n_heads % n_kv != 0 {
        candle_core::bail!("attention: {n_heads} query heads over {n_kv} key heads");
    }
    let groups = n_heads / n_kv;
    Ok((q.contiguous()?.reshape((b, n_kv, groups, l, hd))?, n_kv, groups))
}

/// Attention of `q`, `[b, n_heads, l, hd]`, over `k` and `v` holding the
/// `n_kv` heads the cache stores, `[b, n_kv, L, hd]`, one matmul per position
/// in the group, without expanding them.
///
/// Position `r` of every group is the `[b, n_kv, l, hd]` view `q5[.., .., r]`,
/// whose batch stride skips the other positions: the matmul multiplies it by
/// the stored key head with the shape the expanded form uses for one query
/// head, and the CPU gemm and cuBLAS read the stride rather than copy. The
/// scores are stacked back in query-head order before the scale, the mask and
/// the softmax, each of which works row by row. On the CPU this is bit for
/// bit the expanded form (`attention_forms_agree`); on another backend it is
/// as long as its matmul rounds a batch element the same way whatever the
/// batch count.
fn shared_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    scale: f64,
) -> Result<Tensor> {
    let (b, n_heads, l, hd) = q.dims4()?;
    let (q5, _, groups) = group_view(q, k)?;
    let kt = k.transpose(2, 3)?;
    let scores = (0..groups)
        .map(|r| q5.i((.., .., r))?.matmul(&kt))
        .collect::<Result<Vec<_>>>()?;
    let scores = (Tensor::stack(&scores, 2)? * scale)?;
    let scores = match mask {
        Some(m) => scores.broadcast_add(&m.unsqueeze(2)?)?,
        None => scores,
    };
    let p = candle_nn::ops::softmax_last_dim(&scores)?;
    let ctx = (0..groups)
        .map(|r| p.i((.., .., r))?.matmul(v))
        .collect::<Result<Vec<_>>>()?;
    Tensor::stack(&ctx, 2)?.reshape((b, n_heads, l, hd))
}

/// Attention of `q`, `[b, n_heads, l, hd]`, over `k` and `v` holding the
/// `n_kv` heads the cache stores, `[b, n_kv, L, hd]`, one matmul per key head,
/// without expanding them.
///
/// Query head `h` reads key head `h / groups`, the convention of `repeat_kv`,
/// so the `groups` query heads of one key head are contiguous in `q`. Viewed
/// as `[b, n_kv, groups·l, hd]`, they become the rows of one matrix that
/// multiplies its key head once: no copy of the history, and on Qwen3-4B a
/// quarter of the bytes `repeat_kv` makes the expanded matmuls read. The rows
/// are ordered (query head in the group, position), so the `[l, L]` causal
/// mask applies to each of the `groups` blocks of rows, through a
/// 5-dimensional view. Each score is the same dot product as in the expanded
/// form; whether a backend rounds it the same way at the new matmul shape is
/// a measurement, not a property of this function.
pub(super) fn grouped_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    scale: f64,
) -> Result<Tensor> {
    let (b, n_heads, l, hd) = q.dims4()?;
    let kv_len = k.dim(2)?;
    let (q5, n_kv, groups) = group_view(q, k)?;
    let q = q5.reshape((b, n_kv, groups * l, hd))?;
    let scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
    let scores = match mask {
        Some(m) => scores
            .reshape((b, n_kv, groups, l, kv_len))?
            .broadcast_add(&m.unsqueeze(2)?)?
            .reshape((b, n_kv, groups * l, kv_len))?,
        None => scores,
    };
    candle_nn::ops::softmax_last_dim(&scores)?
        .matmul(v)?
        .reshape((b, n_heads, l, hd))
}

/// The `Kernel` form's dispatch of one call of `l` rows, batch 1, over a
/// cache of `n` positions that already holds them (rows at `n - l .. n`).
/// Rows at position `from` or later go to `kern`; the rows below take
/// [`grouped_attention`] over the history they see, with their rows of
/// `mask`. A call on one side of `from` is one launch of one path; a call
/// that straddles it is split, so a row's path depends on its position and
/// on nothing else.
pub(super) fn kernel_rows(
    kern: &dyn crate::device::AttnKernel,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    scale: f64,
    from: usize,
) -> Result<Tensor> {
    let (l, n) = (q.dim(2)?, k.dim(2)?);
    // Rows at `from` or later: positions max(from, n - l) .. n.
    let tail = n.saturating_sub(from.max(n - l));
    let head = l - tail;
    if tail == 0 {
        return grouped_attention(q, &rows_contiguous(k)?, &rows_contiguous(v)?, mask, scale);
    }
    if head == 0 {
        return kern.attend(&q.contiguous()?, k, v);
    }
    let seen = n - tail;
    let mask = mask.map(|m| m.narrow(2, 0, head)?.narrow(3, 0, seen)).transpose()?;
    let low = grouped_attention(
        &q.narrow(2, 0, head)?,
        &rows_contiguous(&k.narrow(2, 0, seen)?)?,
        &rows_contiguous(&v.narrow(2, 0, seen)?)?,
        mask.as_ref(),
        scale,
    )?;
    let high = kern.attend(&q.narrow(2, head, tail)?.contiguous()?, k, v)?;
    Tensor::cat(&[&low, &high], 2)
}

pub struct Block {
    /// Cache mode this block's `forward` builds its scratch cache in.
    kv: KvMode,
    pub q_proj: Proj,
    pub k_proj: Proj,
    pub v_proj: Proj,
    pub o_proj: Proj,
    pub gate_proj: Proj,
    pub up_proj: Proj,
    pub down_proj: Proj,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    ln1: RmsNorm,
    ln2: RmsNorm,
    act: Activation,
    n_heads: usize,
    n_kv: usize,
    head_dim: usize,
    /// Whether a shared activation is rotated once for its group.
    ///
    /// **Not read from the environment here.** Every block built by
    /// `Qwen3::new` — the quantizer, `bin/oracle`, `bin/ppl`, `bin/mmlu` —
    /// gets [`RotShare::Off`], so those paths cannot change behaviour because
    /// a variable was exported in a shell. `fused_cuda::load` resolves
    /// `LLVQ_ROT_SHARE` and calls [`Qwen3::set_rot_share`] explicitly, on the
    /// one path where it means something.
    rot_share: RotShare,
    /// How attention reads the cached heads; [`AttnHeads::Expanded`] unless
    /// [`Qwen3::set_attn_heads`] says otherwise, for the same reason as
    /// `rot_share`.
    ///
    /// [`AttnHeads::Expanded`]: crate::kvq::AttnHeads::Expanded
    attn_heads: crate::kvq::AttnHeads,
    /// The decode-attention kernel the `Kernel` form calls, installed by
    /// [`Qwen3::set_attn_kernel`]; `None` everywhere else, and a `Kernel`
    /// block without one refuses to run rather than fall back.
    attn_kernel: Option<std::sync::Arc<dyn crate::device::AttnKernel>>,
    /// The norms' weights, the same tensors the norms hold.
    ln1_w: Tensor,
    q_norm_w: Tensor,
    k_norm_w: Tensor,
    /// ln2's weight and epsilon, the same tensor the norm holds.
    ln2_w: Tensor,
    ln2_eps: f64,
    /// The first position the `Kernel` form sends to its kernel:
    /// [`crate::kvq::KERNEL_FROM`]. A field only so the host tests can reach
    /// the kernel in a window of a few dozen positions; nothing outside
    /// `cfg(test)` writes it.
    pub(super) kernel_from: usize,
}

/// Take the supplied projection, or build the dense one.
///
/// The closure is only *called* when nothing was supplied — that laziness is
/// the mechanism, not a style choice: `candle_nn::linear_no_bias` reads a
/// tensor out of the `VarBuilder`, and in fused mode that tensor is not in the
/// file at all. Building it to overwrite it a line later would allocate the
/// eight gigabytes the encoded format exists to avoid.
fn pick(
    take: &mut ProjSource,
    layer: usize,
    name: &str,
    dense: impl FnOnce() -> Result<Linear>,
) -> Result<Proj> {
    match take(layer, name) {
        Some(p) => Ok(p),
        None => Ok(Proj::Dense(dense()?)),
    }
}

impl Block {
    /// Most rows one call of this block's prefill takes: the tightest of its
    /// four [`group_forward`] groups, as [`Self::forward`] forms them.
    pub(super) fn row_cap(&self) -> usize {
        [
            group_row_cap(&[&self.q_proj, &self.k_proj, &self.v_proj]),
            group_row_cap(&[&self.o_proj]),
            group_row_cap(&[&self.gate_proj, &self.up_proj]),
            group_row_cap(&[&self.down_proj]),
        ]
        .into_iter()
        .min()
        .expect("four groups")
    }

    /// `idx` names the layer, so a caller can hand over projections it has
    /// already built rather than have them read out of the `VarBuilder`.
    pub(super) fn new_with(
        cfg: &Config,
        vb: VarBuilder,
        idx: usize,
        take: &mut ProjSource,
        kv: KvMode,
    ) -> Result<Self> {
        let (nh, nkv, hd) = (
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim,
        );
        let a = vb.pp("self_attn");
        let m = vb.pp("mlp");
        // The same tensor candle_nn::rms_norm fetches (`weight`, hidden, the same
        // hint, so a VarMap-built test model gets ones as before),
        // kept beside the norm: route P's segment reads it in place.
        // The norms keep their weights beside them, fetched with candle_nn::rms_norm's
        // own hint: 4E (persist4e_cuda.rs) reads them in place.
        let q_norm_w = a.pp("q_norm").get_with_hints(hd, "weight", candle_nn::Init::Const(1.0))?;
        let k_norm_w = a.pp("k_norm").get_with_hints(hd, "weight", candle_nn::Init::Const(1.0))?;
        let ln1_w = vb.pp("input_layernorm").get_with_hints(cfg.hidden_size, "weight", candle_nn::Init::Const(1.0))?;
        let ln2_w = vb.pp("post_attention_layernorm").get_with_hints(
            cfg.hidden_size,
            "weight",
            candle_nn::Init::Const(1.0),
        )?;
        Ok(Self {
            kv,
            q_proj: pick(take, idx, "self_attn.q_proj", || {
                candle_nn::linear_no_bias(cfg.hidden_size, nh * hd, a.pp("q_proj"))
            })?,
            k_proj: pick(take, idx, "self_attn.k_proj", || {
                candle_nn::linear_no_bias(cfg.hidden_size, nkv * hd, a.pp("k_proj"))
            })?,
            v_proj: pick(take, idx, "self_attn.v_proj", || {
                candle_nn::linear_no_bias(cfg.hidden_size, nkv * hd, a.pp("v_proj"))
            })?,
            o_proj: pick(take, idx, "self_attn.o_proj", || {
                candle_nn::linear_no_bias(nh * hd, cfg.hidden_size, a.pp("o_proj"))
            })?,
            gate_proj: pick(take, idx, "mlp.gate_proj", || {
                candle_nn::linear_no_bias(cfg.hidden_size, cfg.intermediate_size, m.pp("gate_proj"))
            })?,
            up_proj: pick(take, idx, "mlp.up_proj", || {
                candle_nn::linear_no_bias(cfg.hidden_size, cfg.intermediate_size, m.pp("up_proj"))
            })?,
            down_proj: pick(take, idx, "mlp.down_proj", || {
                candle_nn::linear_no_bias(cfg.intermediate_size, cfg.hidden_size, m.pp("down_proj"))
            })?,
            q_norm: candle_nn::RmsNorm::new(q_norm_w.clone(), cfg.rms_norm_eps),
            k_norm: candle_nn::RmsNorm::new(k_norm_w.clone(), cfg.rms_norm_eps),
            ln1: candle_nn::RmsNorm::new(ln1_w.clone(), cfg.rms_norm_eps),
            q_norm_w,
            k_norm_w,
            ln1_w,
            ln2: candle_nn::RmsNorm::new(ln2_w.clone(), cfg.rms_norm_eps),
            ln2_w,
            ln2_eps: cfg.rms_norm_eps,
            act: cfg.hidden_act,
            n_heads: nh,
            n_kv: nkv,
            head_dim: hd,
            rot_share: RotShare::Off,
            attn_heads: crate::kvq::AttnHeads::Expanded,
            attn_kernel: None,
            kernel_from: crate::kvq::KERNEL_FROM,
        })
    }

    /// Whether this block hoists the rotation of a shared activation.
    pub fn set_rot_share(&mut self, share: RotShare) {
        self.rot_share = share;
    }

    pub fn rot_share(&self) -> RotShare {
        self.rot_share
    }

    /// How this block's attention reads the cached heads.
    pub fn set_attn_heads(&mut self, heads: crate::kvq::AttnHeads) {
        self.attn_heads = heads;
    }

    /// The kernel the `Kernel` form calls.
    pub fn set_attn_kernel(&mut self, k: Option<std::sync::Arc<dyn crate::device::AttnKernel>>) {
        self.attn_kernel = k;
    }

    /// Read-only view of a projection, by checkpoint name.
    pub fn linear(&self, name: &str) -> &Linear {
        self.proj(name).dense()
    }

    /// The projection itself, dense or fused.
    pub fn proj(&self, name: &str) -> &Proj {
        match name {
            "self_attn.q_proj" => &self.q_proj,
            "self_attn.k_proj" => &self.k_proj,
            "self_attn.v_proj" => &self.v_proj,
            "self_attn.o_proj" => &self.o_proj,
            "mlp.gate_proj" => &self.gate_proj,
            "mlp.up_proj" => &self.up_proj,
            "mlp.down_proj" => &self.down_proj,
            other => panic!("unknown projection {other}"),
        }
    }

    /// Replace one projection — how a fused runtime is installed.
    pub fn set_proj(&mut self, name: &str, p: Proj) {
        *self.proj_mut(name) = p;
    }

    /// The projection a safetensors name suffix refers to.
    ///
    /// The quantizer walks activations, not fields, so it needs to reach
    /// matrices by the name the checkpoint uses.
    pub fn linear_mut(&mut self, name: &str) -> &mut Linear {
        self.proj_mut(name).dense_mut()
    }

    pub fn proj_mut(&mut self, name: &str) -> &mut Proj {
        match name {
            "self_attn.q_proj" => &mut self.q_proj,
            "self_attn.k_proj" => &mut self.k_proj,
            "self_attn.v_proj" => &mut self.v_proj,
            "self_attn.o_proj" => &mut self.o_proj,
            "mlp.gate_proj" => &mut self.gate_proj,
            "mlp.up_proj" => &mut self.up_proj,
            "mlp.down_proj" => &mut self.down_proj,
            other => panic!("unknown projection {other}"),
        }
    }

    /// Full-sequence forward, scoring. A cached forward whose cache is empty
    /// and whose offset is zero — the *same code*, so the two paths cannot
    /// drift. `bin/oracle` pins this against `candle_transformers::qwen3` at
    /// `max |Δhidden| = 0`, and that gate covers both.
    pub fn forward(
        &self,
        x: &Tensor,
        rotary: &Rotary,
        mask: &Tensor,
        idx: usize,
        cap: &mut dyn Capture,
    ) -> Result<Tensor> {
        let mut fresh = KvCache::new(self.kv);
        self.forward_cached(x, rotary, Some(mask), 0, &mut fresh, idx, cap)
    }

    /// Forward over `l` new positions starting at absolute position `offset`,
    /// attending over everything in `cache` plus what this call adds.
    ///
    /// `mask` is `None` only where every entry of it would be zero: one new
    /// position on the `Cat` store, which sees the whole history and nothing
    /// in its future. Adding a zero leaves every score bit for bit what it
    /// was (a `-0.0` becomes `+0.0`, which softmax cannot tell apart), so
    /// skipping the add changes no output and saves 36 launches a token.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_cached(
        &self,
        x: &Tensor,
        rotary: &Rotary,
        mask: Option<&Tensor>,
        offset: usize,
        cache: &mut KvCache,
        idx: usize,
        cap: &mut dyn Capture,
    ) -> Result<Tensor> {
        let (b, l, _) = x.dims3()?;

        let h = self.ln1.forward(x)?;
        cap.on_activation(idx, Act::Attn, &h)?;

        // q, k and v consume `h` — one activation, one rotation, three uses.
        // The group is named here, at the site, rather than recovered from a
        // key at launch time: see `crate::rotplan` for why every key that
        // could have been used is a liar.
        let qkv = group_forward(
            &[&self.q_proj, &self.k_proj, &self.v_proj],
            &h,
            self.rot_share,
        )?;
        let q = qkv[0]
            .reshape((b, l, self.n_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = qkv[1]
            .reshape((b, l, self.n_kv, self.head_dim))?
            .transpose(1, 2)?;
        let v = qkv[2]
            .reshape((b, l, self.n_kv, self.head_dim))?
            .transpose(1, 2)?;

        // Per-head RMSNorm on q and k — specific to Qwen3, and the detail a
        // hand-written forward is most likely to drop.
        let q = self
            .q_norm
            .forward(&q.flatten(0, 2)?)?
            .reshape((b, self.n_heads, l, self.head_dim))?;
        let k = self
            .k_norm
            .forward(&k.flatten(0, 2)?)?
            .reshape((b, self.n_kv, l, self.head_dim))?;

        // RoPE at the *absolute* position. `Rotary::apply` has taken an
        // offset since it was written; nothing had ever passed one but zero.
        let (q, k) = rotary.apply(&q, &k, offset)?;
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        if self.attn_heads == crate::kvq::AttnHeads::Kernel {
            let kern = self.attn_kernel.as_ref().ok_or_else(|| {
                candle_core::Error::msg(
                    "attention \"kernel\" with no kernel installed: Served::apply installs it, \
                     and a block never falls back on its own",
                )
            })?;
            let grow = matches!(cache.store(), crate::kvq::KvStore::Grow(_));
            let (k, v) = cache.append(&k, &v)?;
            // Decode and verify rows at `kernel_from` or later go to the
            // kernel. A row below it, a prefill chunk, a batch, or the scratch
            // `Cat` cache of a full-window forward take the grouped path; the
            // cache bytes are the same either way, since `append` wrote them
            // before the branch.
            let ctx = if b == 1 && l <= kern.max_rows() && grow {
                if q.dtype() != DType::F16 {
                    candle_core::bail!(
                        "attention \"kernel\" decodes in f16 only, and this model runs {:?}",
                        q.dtype()
                    );
                }
                kernel_rows(kern.as_ref(), &q, &k, &v, mask, scale, self.kernel_from)?
            } else {
                grouped_attention(&q, &rows_contiguous(&k)?, &rows_contiguous(&v)?, mask, scale)?
            };
            return self.finish(x, ctx, idx, cap);
        }
        let unexpanded = match self.attn_heads {
            crate::kvq::AttnHeads::Expanded => None,
            crate::kvq::AttnHeads::Shared => Some(shared_attention as AttnFn),
            crate::kvq::AttnHeads::Grouped | crate::kvq::AttnHeads::Kernel => {
                Some(grouped_attention as AttnFn)
            }
        };
        if let Some(attend) = unexpanded {
            let (k, v) = cache.append(&k, &v)?;
            let (k, v) = (rows_contiguous(&k)?, rows_contiguous(&v)?);
            let ctx = attend(&q, &k, &v, mask, scale)?;
            return self.finish(x, ctx, idx, cap);
        }
        let groups = self.n_heads / self.n_kv;
        let (k, v) = cache.append(&k, &v)?;
        let (k, v) = (
            candle_transformers::utils::repeat_kv(k, groups)?.contiguous()?,
            candle_transformers::utils::repeat_kv(v, groups)?.contiguous()?,
        );

        let scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        let scores = match mask {
            Some(m) => scores.broadcast_add(m)?,
            None => scores,
        };
        let ctx = candle_nn::ops::softmax_last_dim(&scores)?.matmul(&v)?;
        self.finish(x, ctx, idx, cap)
    }

    /// Everything after the attention weights: `o_proj`, the residual, the
    /// MLP. `ctx` is `[b, n_heads, l, hd]`, whichever form computed it.
    fn finish(&self, x: &Tensor, ctx: Tensor, idx: usize, cap: &mut dyn Capture) -> Result<Tensor> {
        let (b, l, _) = x.dims3()?;
        let ctx = ctx
            .transpose(1, 2)?
            .reshape((b, l, self.n_heads * self.head_dim))?;
        cap.on_activation(idx, Act::AttnOut, &ctx)?;
        let x = (x + self.o_proj.forward(&ctx)?)?;
        self.mlp_residual(&x, idx, cap)
    }

    /// `x + down(act(gate(ln2 x)) * up(ln2 x))`, the MLP half of `finish`.
    fn mlp_residual(&self, x: &Tensor, idx: usize, cap: &mut dyn Capture) -> Result<Tensor> {
        let h = self.ln2.forward(x)?;
        cap.on_activation(idx, Act::Mlp, &h)?;
        // The second shared activation: gate and up both read `h`.
        let gu = group_forward(&[&self.gate_proj, &self.up_proj], &h, self.rot_share)?;
        let gated = gu[0].apply(&self.act)?.mul(&gu[1])?;
        cap.on_activation(idx, Act::MlpOut, &gated)?;
        x + self.down_proj.forward(&gated)?
    }

    /// What 4E's MLP phase reads from this block: ln2's weight and
    /// epsilon, the activation, and the three MLP projections.
    pub fn mlp_parts(&self) -> (&Tensor, f64, Activation, &Proj, &Proj, &Proj) {
        (&self.ln2_w, self.ln2_eps, self.act, &self.gate_proj, &self.up_proj, &self.down_proj)
    }

    /// What 4E reads from this block's attention half: ln1, q-norm and
    /// k-norm weights, and the q, k, v, o projections.
    pub fn attn_parts(&self) -> (&Tensor, &Tensor, &Tensor, &Proj, &Proj, &Proj, &Proj) {
        (&self.ln1_w, &self.q_norm_w, &self.k_norm_w, &self.q_proj, &self.k_proj, &self.v_proj, &self.o_proj)
    }

    /// One decode row for each of B sequences, `x` `[B, 1, d]`.
    ///
    /// The operations of [`Self::forward_cached`] on its decode path, in the
    /// same order, with two differences that move no bit of a row: RoPE reads
    /// a table row per sequence (`cos`/`sin` `[B, 1, hd/2]`, gathered at each
    /// position), and the attention is [`crate::batch::BatchDevice::attend`],
    /// which writes the fresh key and value into the sequence's slot itself.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_batch(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        kc: &Tensor,
        vc: &Tensor,
        pos: &Tensor,
        max_len: usize,
        dev: &dyn crate::batch::BatchDevice,
        projs: crate::batch::BatchProjs,
    ) -> Result<Tensor> {
        let (b, l, _) = x.dims3()?;
        if l != 1 {
            candle_core::bail!("a batched step carries one row per sequence, got {l}");
        }
        let h = self.ln1.forward(x)?;
        let qkv = batch_group(&[&self.q_proj, &self.k_proj, &self.v_proj], &h, self.rot_share, dev, projs)?;
        let q = qkv[0].reshape((b, 1, self.n_heads, self.head_dim))?.transpose(1, 2)?;
        let k = qkv[1].reshape((b, 1, self.n_kv, self.head_dim))?.transpose(1, 2)?;
        let v = qkv[2].reshape((b, 1, self.n_kv, self.head_dim))?.transpose(1, 2)?;
        let q = self.q_norm.forward(&q.flatten(0, 2)?)?.reshape((b, self.n_heads, 1, self.head_dim))?;
        let k = self.k_norm.forward(&k.flatten(0, 2)?)?.reshape((b, self.n_kv, 1, self.head_dim))?;
        let q = candle_nn::rotary_emb::rope(&q.contiguous()?, cos, sin)?;
        let k = candle_nn::rotary_emb::rope(&k.contiguous()?, cos, sin)?;
        let ctx = dev.attend(&q, &k, &v.contiguous()?, kc, vc, pos, max_len)?;
        let ctx = ctx.transpose(1, 2)?.reshape((b, 1, self.n_heads * self.head_dim))?;
        let x = (x + &batch_group(&[&self.o_proj], &ctx, self.rot_share, dev, projs)?[0])?;
        let h = self.ln2.forward(&x)?;
        let gu = batch_group(&[&self.gate_proj, &self.up_proj], &h, self.rot_share, dev, projs)?;
        let gated = gu[0].apply(&self.act)?.mul(&gu[1])?;
        x + &batch_group(&[&self.down_proj], &gated, self.rot_share, dev, projs)?[0]
    }
}

/// The projections of one shared activation for a batched step.
///
/// [`crate::batch::BatchProjs::Rows`] is [`group_forward`] itself, so every
/// row is the one-row kernel's. [`crate::batch::BatchProjs::Mma`] rotates the
/// rows once (once for the group under [`RotShare::On`], as `group_forward`
/// does) and hands each projection to the device's tensor-core kernel.
pub fn batch_group(
    projs: &[&Proj],
    x: &Tensor,
    share: RotShare,
    dev: &dyn crate::batch::BatchDevice,
    mode: crate::batch::BatchProjs,
) -> Result<Vec<Tensor>> {
    if mode == crate::batch::BatchProjs::Rows {
        return group_forward(projs, x, share);
    }
    let dims = x.dims().to_vec();
    let flat = row_block(x)?;
    let n = flat.dim(0)?;
    let mut shared: Option<Rotated> = None;
    let mut out = Vec::with_capacity(projs.len());
    for p in projs {
        let y = match p {
            Proj::Lattice(d) => {
                let own;
                let r = match share {
                    RotShare::On => {
                        if shared.is_none() {
                            shared = Some(p.prepare_rows(&flat)?);
                        }
                        shared.as_ref().expect("set above")
                    }
                    RotShare::Off => {
                        own = p.prepare_rows(&flat)?;
                        &own
                    }
                };
                check_key(p.site_name(), p.rot_key(), r.key).map_err(candle_core::Error::msg)?;
                dev.lattice_mma(d.as_ref(), &r.t, n)?
            }
            Proj::Int4(d) => dev.int4_mma(d.as_ref(), &flat, n)?,
            Proj::Dense(l) => l.forward(&flat)?,
            Proj::GroupPart { .. } => candle_core::bail!(
                "{}: a fused group in a batched step (LLVQ_FUSE=1 has no batched kernel)",
                p.site_name()
            ),
        };
        let mut od = dims.clone();
        *od.last_mut().expect("rank >= 1") = p.d_out()?;
        out.push(y.reshape(od)?);
    }
    Ok(out)
}
