//! The layers `Qwen3` stacks: the embedding, the head and one transformer block, with the batch
//! entry points that run a group of projections over several sequences. The batch entry points
//! sit here because the block's batched step is their caller in the model.

use super::*;
use candle_core::{DType, Module, Result, Tensor};
use candle_nn::{Activation, Embedding, Linear, RmsNorm, VarBuilder};
use candle_transformers::models::qwen3::Config;

use crate::kvq::KvMode;
use crate::rotplan::{check_key, RotShare};

/// The token embedding: a dense f16 table, or the int8 g64 payload the fused
/// runtime keeps on the device (`LLVQ_EMBED=q8`).
///
/// Same design as [`Proj`]: an enum whose device arm is a trait object, so it
/// compiles and runs on every target and a CPU fake drives it under
/// `cargo test`. The dense arm is byte-for-byte the code that produced every
/// published number.
pub enum Embed {
    Dense(Embedding),
    Q8(std::sync::Arc<dyn crate::device::QuantEmbedTable>),
}

impl Embed {
    /// Token ids `(.., l)` → hidden states `(.., l, d)`.
    pub fn forward(&self, ids: &Tensor) -> Result<Tensor> {
        match self {
            Embed::Dense(e) => e.forward(ids),
            Embed::Q8(t) => t.gather(ids),
        }
    }
}

/// The `lm_head`: a dense tensor multiplied through candle, or an int8 g64
/// buffer read by a dedicated matvec. Qwen3-4B ties the two ends, so that
/// buffer *is* [`Embed::Q8`]'s and one payload serves both; Qwen3-8B unties
/// them, and this is then its own table with its own values.
pub enum Head {
    Dense(Tensor),
    Q8(std::sync::Arc<dyn crate::device::QuantEmbedTable>),
}

impl Head {
    /// `h · Wᵀ` — logits from hidden states, in the model dtype.
    pub fn project(&self, h: &Tensor) -> Result<Tensor> {
        match self {
            Head::Dense(t) => h.broadcast_matmul(&t.t()?),
            Head::Q8(t) => t.project(h),
        }
    }
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
    /// How the `Kernel` form attends a call of more rows than the kernel
    /// carries ([`crate::kvq::PrefillAttn`]): set by
    /// [`Qwen3::set_prefill_attention`], `Grouped` everywhere else.
    prefill_attn: crate::kvq::PrefillAttn,
    /// What a `LLVQ_KV_LADDER` arm needs to code this layer's cache (its
    /// static key steps, the Hadamard matrix), installed by
    /// [`Qwen3::set_kv_ladder_aux`]; `None` everywhere else, and the `F16`
    /// and `Q8` caches never read it.
    pub(super) kv_aux: Option<crate::kvladder::LadderAux>,
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
        // own hint: 4E (persist4e_cuda/) reads them in place.
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
            prefill_attn: crate::kvq::PrefillAttn::Grouped,
            kv_aux: None,
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

    /// How the `Kernel` form attends a prefill chunk.
    pub fn set_prefill_attention(&mut self, p: crate::kvq::PrefillAttn) {
        self.prefill_attn = p;
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
        // The attention's locals (ln1, q, k, v before and after their norms
        // and RoPE) live in `attention_ctx` and are freed when it returns, so
        // none of them sits under the MLP: 0.22 GB at 4B for a 4,096-row
        // chunk, the peak's moment once no score tensor exists.
        let ctx = self.attention_ctx(x, rotary, mask, offset, cache, idx, cap)?;
        self.finish(x, ctx, idx, cap)
    }

    /// The attention half of [`Self::forward_cached`]: its context
    /// `[b, n_heads, l, hd]`, the cache appended.
    #[allow(clippy::too_many_arguments)]
    fn attention_ctx(
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
        cap.on_attention_query(idx, &q)?;
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        if self.attn_heads == crate::kvq::AttnHeads::Kernel {
            let kern = self.attn_kernel.as_ref().ok_or_else(|| {
                candle_core::Error::msg(
                    "attention \"kernel\" with no kernel installed: Served::apply installs it, \
                     and a block never falls back on its own",
                )
            })?;
            let grow = matches!(cache.store(), crate::kvq::KvStore::Grow(_));
            let (k, v) = cache.append_with(&k, &v, self.kv_aux.as_ref())?;
            // A chunk of at most `kern.max_rows()` rows on a growing cache,
            // decode or prefill alike, sends its rows at `kernel_from` or later
            // to the kernel (`kernel_rows`). A longer chunk goes to the kernel
            // under the kernel prefill form (`kernel_prefill`), and to the
            // grouped path otherwise, as do rows below `kernel_from`, a batch
            // and the scratch `Cat` cache of a full-window forward. The two
            // paths are not bit for bit, so where a prompt is cut into chunks
            // decides the bits of its attention. The cache bytes are the same
            // either way, since `append` wrote them before the branch.
            let ctx = if b == 1 && l <= kern.max_rows() && grow {
                if q.dtype() != DType::F16 {
                    candle_core::bail!(
                        "attention \"kernel\" decodes in f16 only, and this model runs {:?}",
                        q.dtype()
                    );
                }
                kernel_rows(kern.as_ref(), &q, &k, &v, mask, scale, self.kernel_from)?
            } else if b == 1 && grow && self.prefill_attn == crate::kvq::PrefillAttn::Kernel {
                if q.dtype() != DType::F16 {
                    candle_core::bail!(
                        "prefill attention \"kernel\" runs in f16 only, and this model runs {:?}",
                        q.dtype()
                    );
                }
                kernel_prefill(kern.as_ref(), &q, &k, &v, mask, scale, self.kernel_from)?
            } else {
                grouped_attention(&q, &rows_contiguous(&k)?, &rows_contiguous(&v)?, mask, scale)?
            };
            return Ok(ctx);
        }
        let unexpanded = match self.attn_heads {
            crate::kvq::AttnHeads::Expanded => None,
            crate::kvq::AttnHeads::Shared => Some(shared_attention as AttnFn),
            crate::kvq::AttnHeads::Grouped | crate::kvq::AttnHeads::Kernel => {
                Some(grouped_attention as AttnFn)
            }
        };
        if let Some(attend) = unexpanded {
            let (k, v) = cache.append_with(&k, &v, self.kv_aux.as_ref())?;
            let (k, v) = (rows_contiguous(&k)?, rows_contiguous(&v)?);
            let ctx = attend(&q, &k, &v, mask, scale)?;
            return Ok(ctx);
        }
        let groups = self.n_heads / self.n_kv;
        let (k, v) = cache.append_with(&k, &v, self.kv_aux.as_ref())?;
        let (k, v) = (
            candle_transformers::utils::repeat_kv(k, groups)?.contiguous()?,
            candle_transformers::utils::repeat_kv(v, groups)?.contiguous()?,
        );

        let raw = q.matmul(&k.transpose(2, 3)?)?;
        let p = match mask {
            Some(m) if in_place(&raw) => {
                softmax_in_place(&raw, m, scale, l)?;
                raw
            }
            Some(m) => {
                let scores = {
                    let scaled = (raw * scale)?;
                    scaled.broadcast_add(m)?
                };
                candle_nn::ops::softmax_last_dim(&scores)?
            }
            None => candle_nn::ops::softmax_last_dim(&(raw * scale)?)?,
        };
        let ctx = p.matmul(&v)?;
        Ok(ctx)
    }

    /// Everything after the attention weights: `o_proj`, the residual, the
    /// MLP. `ctx` is `[b, n_heads, l, hd]`, whichever form computed it.
    fn finish(&self, x: &Tensor, ctx: Tensor, idx: usize, cap: &mut dyn Capture) -> Result<Tensor> {
        let (b, l, _) = x.dims3()?;
        // `ctx` is consumed here and its flat copy dropped after `o_proj`: a
        // shadowed binding would keep both alive through the MLP.
        let x = {
            let flat = ctx.transpose(1, 2)?.reshape((b, l, self.n_heads * self.head_dim))?;
            drop(ctx);
            cap.on_activation(idx, Act::AttnOut, &flat)?;
            (x + self.o_proj.forward(&flat)?)?
        };
        self.mlp_residual(&x, idx, cap)
    }

    /// `x + down(act(gate(ln2 x)) * up(ln2 x))`, the MLP half of `finish`.
    fn mlp_residual(&self, x: &Tensor, idx: usize, cap: &mut dyn Capture) -> Result<Tensor> {
        // ln2, gate and up are dropped before `down_proj` runs.
        let gated = {
            let h = self.ln2.forward(x)?;
            cap.on_activation(idx, Act::Mlp, &h)?;
            // The second shared activation: gate and up both read `h`.
            let gu = group_forward(&[&self.gate_proj, &self.up_proj], &h, self.rot_share)?;
            gu[0].apply(&self.act)?.mul(&gu[1])?
        };
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
        slots: &Tensor,
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
        let ctx = dev.attend(&q, &k, &v.contiguous()?, kc, vc, pos, slots, max_len)?;
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
    if mode.is_gemm() {
        return batch_gemm_group(projs, x, share, dev, mode);
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

/// [`batch_group`] under [`crate::batch::BatchProjs::Gemm`]: the rows rotated
/// once (once for the group under [`RotShare::On`]), then every lattice and
/// int4 projection of the group in one call of the device's GEMM. A dense
/// projection runs on its own.
fn batch_gemm_group(
    projs: &[&Proj],
    x: &Tensor,
    share: RotShare,
    dev: &dyn crate::batch::BatchDevice,
    mode: crate::batch::BatchProjs,
) -> Result<Vec<Tensor>> {
    use crate::batch::GemmIn;
    let dims = x.dims().to_vec();
    let flat = row_block(x)?;
    let n = flat.dim(0)?;
    let mut shared: Option<Rotated> = None;
    let mut shared_h: Option<Tensor> = None;
    // The rows each projection reads, held here so the GEMM can borrow them.
    let mut inputs: Vec<Option<Tensor>> = Vec::with_capacity(projs.len());
    let mut out: Vec<Option<Tensor>> = vec![None; projs.len()];
    for (i, p) in projs.iter().enumerate() {
        match p {
            Proj::Lattice(_) => {
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
                // The GEMM reads the rotated rows in f16, two bytes a value for
                // every CTA that re-reads them: converted once a group under
                // the shared rotation.
                let h = match (share, &shared_h) {
                    (RotShare::On, Some(h)) => h.clone(),
                    _ => r.t.to_dtype(DType::F16)?.contiguous()?,
                };
                if share == RotShare::On {
                    shared_h = Some(h.clone());
                }
                inputs.push(Some(h));
            }
            Proj::Int4(_) => inputs.push(Some(flat.to_dtype(DType::F16)?.contiguous()?)),
            Proj::Dense(l) => {
                inputs.push(None);
                out[i] = Some(l.forward(&flat)?);
            }
            Proj::GroupPart { .. } => candle_core::bail!(
                "{}: a fused group in a batched step (LLVQ_FUSE=1 has no batched kernel)",
                p.site_name()
            ),
        }
    }
    let mut items = Vec::new();
    let mut at = Vec::new();
    for (i, p) in projs.iter().enumerate() {
        match (p, &inputs[i]) {
            (Proj::Lattice(d), Some(t)) => items.push(GemmIn::Lattice(d.as_ref(), t)),
            (Proj::Int4(d), Some(t)) => items.push(GemmIn::Int4(d.as_ref(), t)),
            _ => continue,
        }
        at.push(i);
    }
    if !items.is_empty() {
        for (i, y) in at.into_iter().zip(dev.gemm(&items, n, mode)?) {
            out[i] = Some(y);
        }
    }
    let mut res = Vec::with_capacity(projs.len());
    for (p, y) in projs.iter().zip(out) {
        let mut od = dims.clone();
        *od.last_mut().expect("rank >= 1") = p.d_out()?;
        res.push(y.expect("every projection ran").reshape(od)?);
    }
    Ok(res)
}
