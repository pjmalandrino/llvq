//! A Qwen3 forward pass we control, so the input of every linear layer is
//! observable.
//!
//! GPTQ needs `H = AᵀA/N` where `A` is the *input* to a linear layer. No
//! inference library exposes that: `candle`'s Qwen3 applies its `Linear`s
//! inline with no hook, and wrapping them from outside is not possible
//! because the intermediate tensors never leave the block. So the block
//! forward is written out here, with capture points on the four distinct
//! activations a Qwen3 block produces.
//!
//! **Four, not seven.** `q_proj`, `k_proj` and `v_proj` all consume the same
//! tensor (`input_layernorm(x)`), and `gate_proj`/`up_proj` both consume
//! `post_attention_layernorm(x)`. Only `o_proj` and `down_proj` have inputs
//! of their own. Accumulating one Hessian per *activation* rather than per
//! *matrix* is exact and saves 3/7 of the work.
//!
//! The implementation is validated against `candle_transformers`' own Qwen3
//! (see `bin/oracle.rs`): same weights, same window, logits must agree. That
//! reference is the only thing standing between a subtly wrong RoPE and
//! three weeks of chasing a perplexity that is off by 15 %.
//!
//! Derived from the architecture as implemented in `candle-transformers`
//! (MIT OR Apache-2.0), restructured for observability.

use candle_core::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::{Activation, Embedding, Linear, RmsNorm, VarBuilder};
use candle_transformers::models::qwen3::Config;

/// Which activation a capture callback is being handed.
///
/// Named after the tensor, not the matrix: `Attn` feeds `q_proj`, `k_proj`
/// and `v_proj` alike.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Act {
    /// `input_layernorm(x)` — input of q/k/v_proj. Width: `hidden_size`.
    Attn,
    /// The concatenated attention context — input of o_proj.
    /// Width: `head_dim · num_attention_heads`.
    AttnOut,
    /// `post_attention_layernorm(x)` — input of gate/up_proj.
    /// Width: `hidden_size`.
    Mlp,
    /// `act(gate(h)) · up(h)` — input of down_proj. Width: `intermediate_size`.
    MlpOut,
}

impl Act {
    pub const ALL: [Act; 4] = [Act::Attn, Act::AttnOut, Act::Mlp, Act::MlpOut];

    /// Stable index, so a rotation seed derived from it is reproducible
    /// across runs and across reorderings of `ALL`.
    pub fn index(&self) -> u64 {
        match self {
            Act::Attn => 0,
            Act::AttnOut => 1,
            Act::Mlp => 2,
            Act::MlpOut => 3,
        }
    }

    /// The matrices this activation feeds, as safetensors name suffixes.
    pub fn consumers(&self) -> &'static [&'static str] {
        match self {
            Act::Attn => &["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
            Act::AttnOut => &["self_attn.o_proj"],
            Act::Mlp => &["mlp.gate_proj", "mlp.up_proj"],
            Act::MlpOut => &["mlp.down_proj"],
        }
    }

    pub fn width(&self, cfg: &Config) -> usize {
        match self {
            Act::Attn | Act::Mlp => cfg.hidden_size,
            Act::AttnOut => cfg.head_dim * cfg.num_attention_heads,
            Act::MlpOut => cfg.intermediate_size,
        }
    }
}

/// Receives every linear-layer input as the forward pass produces it.
pub trait Capture {
    /// `x` is `(batch, seq, width)`; rows are samples for `AᵀA`.
    fn on_activation(&mut self, layer: usize, act: Act, x: &Tensor) -> Result<()>;
}

/// The do-nothing capture, for plain inference.
pub struct NoCapture;

impl Capture for NoCapture {
    fn on_activation(&mut self, _: usize, _: Act, _: &Tensor) -> Result<()> {
        Ok(())
    }
}

/// Per-phase wall time of a fenced greedy decode, milliseconds.
///
/// Produced by [`Qwen3::generate_phased`]; see its contract. A decode of `n`
/// tokens yields `n` samples for the head-side phases and `n − 1` for the
/// forward-side ones — the last emitted token needs no further forward.
#[derive(Default)]
pub struct PhaseReport {
    /// Token id → hidden state (`Embed::forward` plus the host→device id copy).
    pub embed_ms: Vec<f64>,
    /// The transformer blocks — attention, MLP, rotation if fused — plus the
    /// causal mask and the final norm.
    pub blocks_ms: Vec<f64>,
    /// `Head::project` plus the f32 upcast of the logits.
    pub head_ms: Vec<f64>,
    /// Argmax, device→host readback of the winning id, bookkeeping.
    pub rest_ms: Vec<f64>,
    /// Device fences issued. Pinned by a test: a fence dropped between two
    /// phases would silently charge one phase's work to the next, and no
    /// timing assertion could catch that on a fast host.
    pub fences: usize,
}

impl PhaseReport {
    /// `(median, min, max)` of one phase's samples. Even counts take the mean
    /// of the two central samples.
    pub fn stats(samples: &[f64]) -> (f64, f64, f64) {
        assert!(!samples.is_empty(), "no samples to summarise");
        let mut s = samples.to_vec();
        s.sort_by(|a, b| a.partial_cmp(b).expect("phase times are finite"));
        let n = s.len();
        let median = if n % 2 == 1 {
            s[n / 2]
        } else {
            (s[n / 2 - 1] + s[n / 2]) / 2.0
        };
        (median, s[0], s[n - 1])
    }
}

/// The `LLVQ_TIME_PHASES` gate: exactly `"1"` opts in. Anything else — unset
/// included — leaves the caller byte-identical to the published protocol.
pub fn time_phases_enabled(value: Option<&str>) -> bool {
    value == Some("1")
}

/// One phase boundary: a device fence, counted.
///
/// The synchronize and the count share these two lines on purpose — the test
/// that pins `fences` can then vouch that every boundary reached the device.
/// (A mutant keeping the count but dropping the sync is invisible on a
/// synchronous backend, where the sync is a no-op anyway; on CUDA it is the
/// pairing below that the count certifies.)
fn fence(dev: &Device, n: &mut usize) -> Result<()> {
    dev.synchronize()?;
    *n += 1;
    Ok(())
}

pub struct Rotary {
    sin: Tensor,
    cos: Tensor,
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

/// One block's keys and values, for every position already seen.
///
/// Stored **before** `repeat_kv`: the grouped-query expansion is a view over
/// `n_kv` heads, so caching the expanded form would hold `n_heads / n_kv`
/// times the bytes for nothing. On Qwen3-4B that is a factor 4.
#[derive(Default)]
pub struct KvCache {
    k: Option<Tensor>,
    v: Option<Tensor>,
}

impl KvCache {
    /// Append this step's keys and values, and return the whole history.
    ///
    /// Concatenation, not a preallocated ring: a ring needs a maximum length
    /// decided in advance, and every length in this repository is a property
    /// of the corpus rather than of the model. The copy is `O(context)` per
    /// step against the `O(context²)` full re-run it replaces.
    fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        let k = match &self.k {
            None => k.clone(),
            Some(p) => Tensor::cat(&[p, k], 2)?.contiguous()?,
        };
        let v = match &self.v {
            None => v.clone(),
            Some(p) => Tensor::cat(&[p, v], 2)?.contiguous()?,
        };
        self.k = Some(k.clone());
        self.v = Some(v.clone());
        Ok((k, v))
    }
}

/// One transformer block, with its seven projections reachable by name.
/// A projection: dense weights, or the fused kernel reading encoded ones.
///
/// The dense arm is what every published perplexity refers to and what the
/// quantizer operates on. The fused arm exists only after a model is loaded
/// from a sealed artifact by `fused::load`, and only on a CUDA build — the
/// kernels live in `llvq-cuda`, which does not compile anywhere else.
///
/// The two are *not* interchangeable in both directions: `Proj::dense` panics
/// on a fused projection rather than returning something plausible, because
/// every caller of it is a quantization path and quantizing an already
/// quantized model is a bug, not a use case.
pub enum Proj {
    Dense(Linear),
    #[cfg(all(target_os = "linux", feature = "cuda"))]
    Fused {
        rt: std::sync::Arc<crate::fused_cuda::FusedRuntime>,
        proj: std::sync::Arc<crate::fused_cuda::FusedProj>,
    },
}

impl Proj {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Proj::Dense(l) => l.forward(x),
            #[cfg(all(target_os = "linux", feature = "cuda"))]
            Proj::Fused { rt, proj } => rt.forward(proj, x),
        }
    }

    /// The dense weights, for the quantizer.
    ///
    /// Panics on a fused projection. That is the intended behaviour: the
    /// alternative is an `Option` every caller would `unwrap`, at five sites
    /// that can none of them proceed without it.
    pub fn dense(&self) -> &Linear {
        match self {
            Proj::Dense(l) => l,
            #[cfg(all(target_os = "linux", feature = "cuda"))]
            Proj::Fused { proj, .. } => panic!(
                "{} est une projection fusée — le quantifieur n'opère que sur un modèle dense",
                proj.name
            ),
        }
    }

    pub fn dense_mut(&mut self) -> &mut Linear {
        match self {
            Proj::Dense(l) => l,
            #[cfg(all(target_os = "linux", feature = "cuda"))]
            Proj::Fused { proj, .. } => panic!(
                "{} est une projection fusée — le quantifieur n'opère que sur un modèle dense",
                proj.name
            ),
        }
    }
}

/// The token embedding: a dense f16 table, or the int8 g64 payload the fused
/// runtime keeps on the device (`LLVQ_EMBED=q8`).
///
/// Same design as [`Proj`]: an enum, not a trait object, and the quantized
/// arm exists only on a CUDA build — the gather kernel lives beside the fused
/// matvec and compiles nowhere else. The dense arm is byte-for-byte the code
/// that produced every published number.
pub enum Embed {
    Dense(Embedding),
    #[cfg(all(target_os = "linux", feature = "cuda"))]
    Q8 {
        rt: std::sync::Arc<crate::fused_cuda::FusedRuntime>,
        q: std::sync::Arc<crate::fused_cuda::QuantEmbed>,
    },
}

impl Embed {
    /// Token ids `(.., l)` → hidden states `(.., l, d)`.
    pub fn forward(&self, ids: &Tensor) -> Result<Tensor> {
        match self {
            Embed::Dense(e) => e.forward(ids),
            #[cfg(all(target_os = "linux", feature = "cuda"))]
            Embed::Q8 { rt, q } => rt.embed(q, ids),
        }
    }
}

/// The `lm_head`: a dense tensor multiplied through candle, or an int8 g64
/// buffer read by a dedicated matvec. Qwen3-4B ties the two ends, so that
/// buffer *is* [`Embed::Q8`]'s and one payload serves both; Qwen3-8B unties
/// them, and this is then its own table with its own values.
pub enum Head {
    Dense(Tensor),
    #[cfg(all(target_os = "linux", feature = "cuda"))]
    Q8 {
        rt: std::sync::Arc<crate::fused_cuda::FusedRuntime>,
        q: std::sync::Arc<crate::fused_cuda::QuantEmbed>,
    },
}

impl Head {
    /// `h · Wᵀ` — logits from hidden states, in the model dtype.
    pub fn project(&self, h: &Tensor) -> Result<Tensor> {
        match self {
            Head::Dense(t) => h.broadcast_matmul(&t.t()?),
            #[cfg(all(target_os = "linux", feature = "cuda"))]
            Head::Q8 { rt, q } => rt.lm_head(q, h),
        }
    }
}

pub struct Block {
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
    /// `idx` names the layer, so a caller can hand over projections it has
    /// already built rather than have them read out of the `VarBuilder`.
    fn new_with(
        cfg: &Config,
        vb: VarBuilder,
        idx: usize,
        take: &mut ProjSource,
    ) -> Result<Self> {
        let (nh, nkv, hd) = (
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim,
        );
        let a = vb.pp("self_attn");
        let m = vb.pp("mlp");
        Ok(Self {
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
            q_norm: candle_nn::rms_norm(hd, cfg.rms_norm_eps, a.pp("q_norm"))?,
            k_norm: candle_nn::rms_norm(hd, cfg.rms_norm_eps, a.pp("k_norm"))?,
            ln1: candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?,
            ln2: candle_nn::rms_norm(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
            act: cfg.hidden_act,
            n_heads: nh,
            n_kv: nkv,
            head_dim: hd,
        })
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
        let mut fresh = KvCache::default();
        self.forward_cached(x, rotary, mask, 0, &mut fresh, idx, cap)
    }

    /// Forward over `l` new positions starting at absolute position `offset`,
    /// attending over everything in `cache` plus what this call adds.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_cached(
        &self,
        x: &Tensor,
        rotary: &Rotary,
        mask: &Tensor,
        offset: usize,
        cache: &mut KvCache,
        idx: usize,
        cap: &mut dyn Capture,
    ) -> Result<Tensor> {
        let (b, l, _) = x.dims3()?;

        let h = self.ln1.forward(x)?;
        cap.on_activation(idx, Act::Attn, &h)?;

        let q = self
            .q_proj
            .forward(&h)?
            .reshape((b, l, self.n_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = self
            .k_proj
            .forward(&h)?
            .reshape((b, l, self.n_kv, self.head_dim))?
            .transpose(1, 2)?;
        let v = self
            .v_proj
            .forward(&h)?
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
        let (k, v) = cache.append(&k, &v)?;
        let groups = self.n_heads / self.n_kv;
        let k = candle_transformers::utils::repeat_kv(k, groups)?.contiguous()?;
        let v = candle_transformers::utils::repeat_kv(v, groups)?.contiguous()?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?.broadcast_add(mask)?;
        let ctx = candle_nn::ops::softmax_last_dim(&scores)?.matmul(&v)?;
        let ctx = ctx
            .transpose(1, 2)?
            .reshape((b, l, self.n_heads * self.head_dim))?;
        cap.on_activation(idx, Act::AttnOut, &ctx)?;
        let x = (x + self.o_proj.forward(&ctx)?)?;

        let h = self.ln2.forward(&x)?;
        cap.on_activation(idx, Act::Mlp, &h)?;
        let gated = (self.gate_proj.forward(&h)?.apply(&self.act)? * self.up_proj.forward(&h)?)?;
        cap.on_activation(idx, Act::MlpOut, &gated)?;
        x + self.down_proj.forward(&gated)?
    }
}

/// Qwen3, loaded for scoring.
pub struct Qwen3 {
    cfg: Config,
    embed: Embed,
    pub blocks: Vec<Block>,
    norm: RmsNorm,
    /// `lm_head`; Qwen3-0.6B and -4B tie it to the embedding matrix.
    head: Head,
    rotary: Rotary,
    device: Device,
    dtype: DType,
}

/// Where a caller may hand over an already-built projection.
///
/// Called once per `(layer, "self_attn.q_proj")` pair. Returning `Some` means
/// the `VarBuilder` is **not** consulted for that weight — which is the whole
/// point: in fused mode the dense weights do not exist anywhere, and building
/// them to overwrite them a line later would allocate the eight gigabytes the
/// format exists to avoid.
pub type ProjSource<'a> = dyn FnMut(usize, &str) -> Option<Proj> + 'a;

impl Qwen3 {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        Self::new_with(cfg, vb, &mut |_, _| None)
    }

    /// [`Self::new`], with projections optionally supplied rather than loaded.
    pub fn new_with(cfg: &Config, vb: VarBuilder, take: &mut ProjSource) -> Result<Self> {
        let embed = candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("model.embed_tokens"))?;
        let head = if cfg.tie_word_embeddings {
            Head::Dense(embed.embeddings().clone())
        } else {
            Head::Dense(vb.get((cfg.vocab_size, cfg.hidden_size), "lm_head.weight")?)
        };
        Self::assemble(cfg, vb, take, Embed::Dense(embed), head)
    }

    /// [`Self::new_with`], with the embedding and head supplied rather than
    /// read out of the `VarBuilder` — the q8 path. Same laziness argument as
    /// [`pick`]: in that mode the f16 embedding tensor exists nowhere, and
    /// `candle_nn::embedding` would fail looking it up (or worse, force the
    /// 778 MB the mode exists to avoid).
    #[cfg(all(target_os = "linux", feature = "cuda"))]
    pub fn new_with_embed(
        cfg: &Config,
        vb: VarBuilder,
        take: &mut ProjSource,
        embed: Embed,
        head: Head,
    ) -> Result<Self> {
        Self::assemble(cfg, vb, take, embed, head)
    }

    fn assemble(
        cfg: &Config,
        vb: VarBuilder,
        take: &mut ProjSource,
        embed: Embed,
        head: Head,
    ) -> Result<Self> {
        let vb_l = vb.pp("model.layers");
        let blocks = (0..cfg.num_hidden_layers)
            .map(|i| Block::new_with(cfg, vb_l.pp(i), i, take))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            cfg: cfg.clone(),
            embed,
            blocks,
            norm: candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("model.norm"))?,
            head,
            rotary: Rotary::new(vb.dtype(), cfg, vb.device())?,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    fn causal_mask(&self, b: usize, l: usize) -> Result<Tensor> {
        self.causal_mask_offset(b, l, 0)
    }

    /// Mask for `l` new positions starting at absolute position `offset`,
    /// against the `offset + l` positions now visible.
    ///
    /// At `offset = 0` this is the plain lower triangle. At `l = 1` — one new
    /// token, everything before it already in the cache — every entry is zero:
    /// the token sees the whole history and itself, and nothing is in its
    /// future. That degenerate row is the entire attention arithmetic of a
    /// decode step.
    fn causal_mask_offset(&self, b: usize, l: usize, offset: usize) -> Result<Tensor> {
        let total = offset + l;
        let m: Vec<f32> = (0..l)
            .flat_map(|i| {
                (0..total).map(move |j| {
                    if j <= offset + i { 0.0 } else { f32::NEG_INFINITY }
                })
            })
            .collect();
        Tensor::from_slice(&m, (b, 1, l, total), &self.device)?.to_dtype(self.dtype)
    }

    /// One `KvCache` per block, empty.
    pub fn fresh_caches(&self) -> Vec<KvCache> {
        (0..self.blocks.len()).map(|_| KvCache::default()).collect()
    }

    /// Hidden states for `l` new positions, attending over `caches`.
    pub fn hidden_cached(
        &self,
        input: &Tensor,
        offset: usize,
        caches: &mut [KvCache],
        cap: &mut dyn Capture,
    ) -> Result<Tensor> {
        let (b, l) = input.dims2()?;
        let mask = self.causal_mask_offset(b, l, offset)?;
        let mut h = self.embed.forward(input)?;
        for (i, blk) in self.blocks.iter().enumerate() {
            h = blk.forward_cached(&h, &self.rotary, &mask, offset, &mut caches[i], i, cap)?;
        }
        self.norm.forward(&h)
    }

    /// Hidden states after the final norm, for every position.
    pub fn hidden(&self, input: &Tensor, cap: &mut dyn Capture) -> Result<Tensor> {
        let (b, l) = input.dims2()?;
        let mask = self.causal_mask(b, l)?;
        let mut h = self.embed.forward(input)?;
        for (i, blk) in self.blocks.iter().enumerate() {
            h = blk.forward(&h, &self.rotary, &mask, i, cap)?;
        }
        self.norm.forward(&h)
    }

    /// Logits for every position: `(batch, seq, vocab)`.
    pub fn logits(&self, input: &Tensor, cap: &mut dyn Capture) -> Result<Tensor> {
        let h = self.hidden(input, cap)?;
        self.head.project(&h)
    }

    /// Mean next-token negative log-likelihood over one window, in nats.
    ///
    /// Position `i` predicts token `i+1`, so a window of `l` tokens scores
    /// `l − 1` predictions.
    pub fn window_nll(&self, tokens: &[u32], cap: &mut dyn Capture) -> Result<(f64, usize)> {
        let l = tokens.len();
        assert!(l >= 2, "a window must hold at least two tokens");
        let input = Tensor::from_slice(tokens, (1, l), &self.device)?;
        let logits = self.logits(&input, cap)?.to_dtype(DType::F32)?;
        let logits = logits.i(0)?.narrow(0, 0, l - 1)?;
        let targets = Tensor::from_slice(&tokens[1..], l - 1, &self.device)?;
        let logp = candle_nn::ops::log_softmax(&logits, D::Minus1)?;
        let picked = logp.gather(&targets.reshape((l - 1, 1))?, 1)?;
        let total: f32 = picked.sum_all()?.to_scalar()?;
        Ok((-total as f64, l - 1))
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Greedy continuation of `prompt`, `max_new` tokens, with a KV cache.
    ///
    /// One prefill over the prompt, then one token at a time. What that buys
    /// is not only speed: a decode step becomes a **matvec** — a single
    /// activation vector against each projection — which is the shape the
    /// fused kernel implements. Without the cache each step is a GEMM over
    /// the whole prefix, and the kernel has no call site at all.
    ///
    /// [`Self::generate_uncached`] keeps the old quadratic path as a witness.
    /// The two must return the same tokens; `bin/run` checks it under
    /// `LLVQ_VERIFY_CACHE=1`.
    pub fn generate(
        &self,
        tokens: &[u32],
        max_new: usize,
        cap: &mut dyn Capture,
    ) -> Result<Vec<u32>> {
        // The stop test is after the push, so `out.len() == 0` is never
        // true: at `max_new = 0` the loop would run until RoPE walks off
        // `max_position_embeddings` — 40 960 decode steps and an opaque
        // candle error. The witness returns `[]` immediately, so the two
        // paths diverged at the very first edge case.
        if max_new == 0 {
            return Ok(Vec::new());
        }
        let mut caches = self.fresh_caches();
        let mut out = Vec::with_capacity(max_new);
        let input = Tensor::from_slice(tokens, (1, tokens.len()), &self.device)?;
        let mut h = self.hidden_cached(&input, 0, &mut caches, cap)?;
        let mut offset = tokens.len();
        loop {
            let l = h.dim(1)?;
            let last = h.narrow(1, l - 1, 1)?;
            let logits = self.head.project(&last)?.to_dtype(DType::F32)?;
            let next = logits.i((0, 0))?.argmax(D::Minus1)?.to_scalar::<u32>()?;
            out.push(next);
            if out.len() == max_new {
                return Ok(out);
            }
            let input = Tensor::from_slice(&[next], (1, 1), &self.device)?;
            h = self.hidden_cached(&input, offset, &mut caches, cap)?;
            offset += 1;
        }
    }

    /// The pre-cache path: re-run the whole prefix at every step.
    ///
    /// Quadratic, and kept on purpose. It is the only independent answer to
    /// "did the cache change what the model says" — a cache bug shifts RoPE
    /// positions or drops a mask entry, and both produce fluent, plausible,
    /// different text that no threshold would catch.
    pub fn generate_uncached(
        &self,
        tokens: &[u32],
        max_new: usize,
        cap: &mut dyn Capture,
    ) -> Result<Vec<u32>> {
        let mut ids = tokens.to_vec();
        let mut out = Vec::with_capacity(max_new);
        for _ in 0..max_new {
            let input = Tensor::from_slice(&ids, (1, ids.len()), &self.device)?;
            let logits = self.logits(&input, cap)?.to_dtype(DType::F32)?;
            let last = logits.i((0, ids.len() - 1))?;
            let next = last.argmax(D::Minus1)?.to_scalar::<u32>()?;
            ids.push(next);
            out.push(next);
        }
        Ok(out)
    }

    /// Greedy decode with every phase bracketed by device fences — diagnostics
    /// only, never on the published path.
    ///
    /// `bin/fusedrun` calls this behind `LLVQ_TIME_PHASES=1`, *after* the
    /// published measurement. Each phase boundary is a [`Device::synchronize`],
    /// so the wall time between two fences belongs to the phase between them
    /// and to nothing else. The price of that attribution is the attribution
    /// itself: the fences serialise work the normal path overlaps, so the
    /// fenced per-token total is **not** the published rate and must never be
    /// compared to it. Phases attribute; the unfenced run is the number.
    ///
    /// The prefill is not phased — the question this answers is where a
    /// *decode* token's time goes.
    ///
    /// Returns the tokens (they must match [`Self::generate`]; a test pins it)
    /// and one sample per phase per decode step.
    pub fn generate_phased(&self, tokens: &[u32], max_new: usize) -> Result<(Vec<u32>, PhaseReport)> {
        let mut report = PhaseReport::default();
        if max_new == 0 {
            return Ok((Vec::new(), report));
        }
        let mut caches = self.fresh_caches();
        let mut out = Vec::with_capacity(max_new);
        let input = Tensor::from_slice(tokens, (1, tokens.len()), &self.device)?;
        let mut h = self.hidden_cached(&input, 0, &mut caches, &mut NoCapture)?;
        let mut offset = tokens.len();
        // Close the prefill before the first timed phase, or its tail would be
        // charged to the first lm_head sample.
        fence(&self.device, &mut report.fences)?;
        loop {
            let t = std::time::Instant::now();
            let l = h.dim(1)?;
            let last = h.narrow(1, l - 1, 1)?;
            let logits = self.head.project(&last)?.to_dtype(DType::F32)?;
            fence(&self.device, &mut report.fences)?;
            report.head_ms.push(t.elapsed().as_secs_f64() * 1e3);

            let t = std::time::Instant::now();
            let next = logits.i((0, 0))?.argmax(D::Minus1)?.to_scalar::<u32>()?;
            fence(&self.device, &mut report.fences)?;
            report.rest_ms.push(t.elapsed().as_secs_f64() * 1e3);
            out.push(next);
            if out.len() == max_new {
                return Ok((out, report));
            }

            let t = std::time::Instant::now();
            let input = Tensor::from_slice(&[next], (1, 1), &self.device)?;
            let e = self.embed.forward(&input)?;
            fence(&self.device, &mut report.fences)?;
            report.embed_ms.push(t.elapsed().as_secs_f64() * 1e3);

            let t = std::time::Instant::now();
            let mask = self.causal_mask_offset(1, 1, offset)?;
            let mut hh = e;
            for (i, blk) in self.blocks.iter().enumerate() {
                hh = blk.forward_cached(&hh, &self.rotary, &mask, offset, &mut caches[i], i, &mut NoCapture)?;
            }
            h = self.norm.forward(&hh)?;
            fence(&self.device, &mut report.fences)?;
            report.blocks_ms.push(t.elapsed().as_secs_f64() * 1e3);
            offset += 1;
        }
    }

    pub fn rotary(&self) -> &Rotary {
        &self.rotary
    }

    /// Embed a token window — the input of block 0.
    pub fn embed_tokens(&self, input: &Tensor) -> Result<Tensor> {
        self.embed.forward(input)
    }

    /// Hidden states through the final norm, starting from block 0's input.
    /// Used to score a model whose blocks were replaced in place.
    pub fn head_from_hidden(&self, h: &Tensor) -> Result<Tensor> {
        let h = self.norm.forward(h)?;
        self.head.project(&h)
    }

    /// The causal mask matching a `(batch, seq, _)` hidden-state tensor.
    pub fn causal_mask_for(&self, h: &Tensor) -> Result<Tensor> {
        let (b, l, _) = h.dims3()?;
        self.causal_mask(b, l)
    }
}
