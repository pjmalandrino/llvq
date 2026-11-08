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

use crate::fused::RotKey;
use crate::kvq::KvMode;
use crate::rotplan::{check_key, drive_rows_batched, RotShare};

mod block;
mod capture;
mod generate;
mod kv;
mod proj;
#[cfg(test)]
mod tests;

pub use self::block::*;
pub use self::capture::*;
pub use self::generate::*;
pub use self::kv::*;
pub use self::proj::*;

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

/// Qwen3, loaded for scoring.
pub struct Qwen3 {
    cfg: Config,
    /// Cache mode, named by the caller at construction. `model.rs` reads no
    /// environment variable; the binaries resolve `LLVQ_KV` and pass it here.
    kv: KvMode,
    /// Cache growth store — [`crate::kvq::KvStore::Cat`] unless a served
    /// config or a measurement mode says otherwise via [`Self::set_kv_store`].
    /// The A/B guard is the printed arm line, which must name the store: an
    /// unwired arm prints `cat` and is visible.
    kv_store: crate::kvq::KvStore,
    /// How every block's attention reads the cached heads, kept here as well
    /// as in every block. Set through [`Self::set_attn_heads`] only.
    attn_heads: crate::kvq::AttnHeads,
    embed: Embed,
    pub blocks: Vec<Block>,
    norm: RmsNorm,
    /// The final norm's weight, the tensor the norm holds (4E reads it).
    norm_w: Tensor,
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
    pub fn new(cfg: &Config, vb: VarBuilder, kv: KvMode) -> Result<Self> {
        Self::new_with(cfg, vb, &mut |_, _| None, kv)
    }

    /// [`Self::new`], with projections optionally supplied rather than loaded.
    pub fn new_with(
        cfg: &Config,
        vb: VarBuilder,
        take: &mut ProjSource,
        kv: KvMode,
    ) -> Result<Self> {
        let embed = candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("model.embed_tokens"))?;
        let head = if cfg.tie_word_embeddings {
            Head::Dense(embed.embeddings().clone())
        } else {
            Head::Dense(vb.get((cfg.vocab_size, cfg.hidden_size), "lm_head.weight")?)
        };
        Self::assemble(cfg, vb, take, Embed::Dense(embed), head, kv)
    }

    /// [`Self::new_with`], with the embedding and head supplied rather than
    /// read out of the `VarBuilder` — the q8 path. Same laziness argument as
    /// [`pick`]: in that mode the f16 embedding tensor exists nowhere, and
    /// `candle_nn::embedding` would fail looking it up (or worse, force the
    /// 778 MB the mode exists to avoid).
    pub fn new_with_embed(
        cfg: &Config,
        vb: VarBuilder,
        take: &mut ProjSource,
        embed: Embed,
        head: Head,
        kv: KvMode,
    ) -> Result<Self> {
        Self::assemble(cfg, vb, take, embed, head, kv)
    }

    fn assemble(
        cfg: &Config,
        vb: VarBuilder,
        take: &mut ProjSource,
        embed: Embed,
        head: Head,
        kv: KvMode,
    ) -> Result<Self> {
        let vb_l = vb.pp("model.layers");
        let blocks = (0..cfg.num_hidden_layers)
            .map(|i| Block::new_with(cfg, vb_l.pp(i), i, take, kv))
            .collect::<Result<Vec<_>>>()?;
        let norm_w = vb.pp("model.norm").get_with_hints(cfg.hidden_size, "weight", candle_nn::Init::Const(1.0))?;
        Ok(Self {
            cfg: cfg.clone(),
            kv,
            kv_store: crate::kvq::KvStore::Cat,
            attn_heads: crate::kvq::AttnHeads::Expanded,
            embed,
            blocks,
            norm: candle_nn::RmsNorm::new(norm_w.clone(), cfg.rms_norm_eps),
            norm_w,
            head,
            rotary: Rotary::new(vb.dtype(), cfg, vb.device())?,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    /// Turn rotation hoisting on or off for every block.
    ///
    /// Called by `fused_cuda::load` after resolving `LLVQ_ROT_SHARE`, and by
    /// nothing else: a model built any other way keeps [`RotShare::Off`], so
    /// the paths every published number comes from cannot change because a
    /// variable was exported.
    pub fn set_rot_share(&mut self, share: RotShare) {
        for b in &mut self.blocks {
            b.set_rot_share(share);
        }
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
        build_causal_mask(b, l, offset, offset + l, self.dtype, &self.device)
    }

    /// The mask a cached forward of `l` new positions needs, or `None` when it
    /// would be all zeros: one position on the `Cat` store. That is every
    /// decode step of the served path, and building it cost a host vector, an
    /// upload and a cast a token, then one add a layer, for no effect.
    ///
    /// `Grow` hands attention exactly the history, as `Cat` does, so it takes
    /// the same masks.
    fn step_mask(&self, b: usize, l: usize, offset: usize) -> Result<Option<Tensor>> {
        use crate::kvq::KvStore;
        match self.kv_store {
            KvStore::Cat | KvStore::Grow(_) if l == 1 => Ok(None),
            KvStore::Cat | KvStore::Grow(_) => self.causal_mask_offset(b, l, offset).map(Some),
        }
    }

    /// Most positions one [`Self::hidden_cached`] call takes on this model,
    /// at most [`MAX_PREFILL_ROWS`]: 256 where a projection loops one launch
    /// a row, which is the served Metal path. A caller that feeds a prompt in
    /// chunks of this size is never refused.
    pub fn prefill_rows(&self) -> usize {
        self.blocks.iter().map(Block::row_cap).fold(MAX_PREFILL_ROWS, usize::min)
    }

    /// One `KvCache` per block, empty.
    pub fn fresh_caches(&self) -> Vec<KvCache> {
        (0..self.blocks.len())
            .map(|_| KvCache::with_store(self.kv, self.kv_store))
            .collect()
    }

    /// Name the cache growth store, after construction (the loaders' signatures predate the
    /// store, and threading it through them would touch every call site for
    /// a value that only the generation binaries exercise). Every arm line
    /// that a binary prints must include [`crate::kvq::KvStore::label`], so
    /// a wiring miss shows as `cat` instead of silently measuring it.
    pub fn set_kv_store(&mut self, store: crate::kvq::KvStore) {
        self.kv_store = store;
    }

    pub fn kv_store(&self) -> crate::kvq::KvStore {
        self.kv_store
    }

    /// Name how attention reads the cached heads, in every block. Same rule
    /// as [`Self::set_kv_store`]: a binary that sets it prints
    /// [`crate::kvq::AttnHeads::label`] on its arm line. Unset, every block
    /// keeps the expanded form every published number ran.
    pub fn set_attn_heads(&mut self, heads: crate::kvq::AttnHeads) {
        self.attn_heads = heads;
        for b in &mut self.blocks {
            b.set_attn_heads(heads);
        }
    }

    pub fn attn_heads(&self) -> crate::kvq::AttnHeads {
        self.attn_heads
    }

    /// Install the decode-attention kernel in every block (`None` removes
    /// it). The `Kernel` form refuses to run without one; the other forms
    /// never call it.
    pub fn set_attn_kernel(&mut self, k: Option<std::sync::Arc<dyn crate::device::AttnKernel>>) {
        for b in &mut self.blocks {
            b.set_attn_kernel(k.clone());
        }
    }

    /// The first position the `Kernel` form sends to its kernel:
    /// [`crate::kvq::KERNEL_FROM`], outside the host tests.
    pub fn kernel_from(&self) -> usize {
        self.blocks.first().map_or(crate::kvq::KERNEL_FROM, |b| b.kernel_from)
    }

    /// MEASUREMENT ONLY (`bin/fusedrun`'s `LLVQ_PERS_4E`): send every decode
    /// row of the `Kernel` form to its kernel, so the eager path's attention
    /// is attn_ref.rs's at every position, the arithmetic 4E owes its bits
    /// to. No served path calls it; the served threshold stays the constant.
    pub fn set_kernel_from_for_measurement(&mut self, from: usize) {
        for b in &mut self.blocks {
            b.kernel_from = from;
        }
    }

    /// The served config's threshold ([`crate::served::Served::apply`] only):
    /// the constant, or 0 under a config that names the persistent decode,
    /// the kernel form the operator's T8 extension of 2026-11-05 licenses on
    /// the sealed 4B at every position.
    pub(crate) fn set_served_kernel_from(&mut self, from: usize) {
        for b in &mut self.blocks {
            b.kernel_from = from;
        }
    }

    /// What 4E reads beside the blocks: the final norm's weight, the
    /// embedding and the head tables, and the rotary tables (cos, sin),
    /// `(max_positions, head_dim / 2)` in the model dtype.
    pub fn pers_4e_parts(&self) -> (&Tensor, &Embed, &Head, &Tensor, &Tensor) {
        (&self.norm_w, &self.embed, &self.head, &self.rotary.cos, &self.rotary.sin)
    }

    /// `generate`'s first step: the prompt through the eager path into
    /// `caches`, then the argmax of the last position's logits. The prompt
    /// goes in chunks of at most [`MAX_PREFILL_ROWS`] positions, as
    /// [`Self::generate_timed`] sends it: at or under that length this is
    /// one call, and an 8,192-token context is two (the fused prefill refuses
    /// more rows a call; the first 4E card reading stopped on that refusal).
    pub fn prefill_next(&self, tokens: &[u32], caches: &mut [KvCache]) -> Result<u32> {
        self.prefill_next_in(tokens, MAX_PREFILL_ROWS, caches)
    }

    /// [`Self::prefill_next`] in chunks of at most `chunk` positions.
    pub fn prefill_next_in(&self, tokens: &[u32], chunk: usize, caches: &mut [KvCache]) -> Result<u32> {
        argmax_last(&self.prefill_chunked(tokens, chunk, caches)?)
    }

    /// Lower the threshold so a host test reaches the kernel in a short
    /// window. Test builds only: the served threshold is a constant.
    #[cfg(test)]
    fn set_kernel_from(&mut self, from: usize) {
        for b in &mut self.blocks {
            b.kernel_from = from;
        }
    }

    /// `(n_kv, query heads per KV head, head_dim)`: what a kernel is sized by.
    pub fn attn_geometry(&self) -> (usize, usize, usize) {
        let n_kv = self.cfg.num_key_value_heads;
        (n_kv, self.cfg.num_attention_heads / n_kv, self.cfg.head_dim)
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
        // Under a preallocated window the keys are always `w` wide, so the
        // mask is too: the causal condition `j <= offset + i` already −infs
        // every position past the valid prefix, which is what makes the
        // zero-filled tail of the buffers exactly inert.
        let mask = self.step_mask(b, l, offset)?;
        let mut h = self.embed.forward(input)?;
        for (i, blk) in self.blocks.iter().enumerate() {
            h = blk.forward_cached(&h, &self.rotary, mask.as_ref(), offset, &mut caches[i], i, cap)?;
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

    /// `lm_head` from hidden states, the public form a driver of its own
    /// decode loop needs (the field itself stays private).
    pub fn project_head(&self, h: &Tensor) -> Result<Tensor> {
        self.head.project(h)
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

/// The greedy token of `[1, 1, vocab]` logits, read back to the host.
pub fn argmax_last(logits: &Tensor) -> Result<u32> {
    logits.i((0, 0))?.argmax(D::Minus1)?.to_scalar::<u32>()
}

/// The log-probability of `target` under `[1, 1, vocab]` f32 logits.
pub fn logp_of(logits: &Tensor, target: u32) -> Result<f64> {
    let logp = candle_nn::ops::log_softmax(&logits.i((0, 0))?, D::Minus1)?;
    Ok(logp.i(target as usize)?.to_scalar::<f32>()? as f64)
}

/// The causal mask at an imposed key width `w >= offset + l` — free-standing
/// so the unit tests can exercise the exact object the model uses, not a
/// re-derivation of it. Columns past `offset + i` are −inf: the causal rule
/// is also the validity rule of a preallocated window's unwritten tail.
pub fn build_causal_mask(
    b: usize,
    l: usize,
    offset: usize,
    w: usize,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    debug_assert!(w >= offset + l);
    let m: Vec<f32> = (0..l)
        .flat_map(|i| (0..w).map(move |j| if j <= offset + i { 0.0 } else { f32::NEG_INFINITY }))
        .collect();
    Tensor::from_slice(&m, (b, 1, l, w), device)?.to_dtype(dtype)
}
