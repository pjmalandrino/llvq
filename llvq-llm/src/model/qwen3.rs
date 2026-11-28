//! [`Qwen3`], the whole model: construction, prefill, decode, generation and scoring. The type
//! and its one `impl` stay whole because together they fit under the size target.

use super::*;
use candle_core::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::{RmsNorm, VarBuilder};
use candle_transformers::models::qwen3::Config;

use crate::kvq::KvMode;
use crate::rotplan::RotShare;

/// Qwen3, loaded for scoring.
pub struct Qwen3 {
    cfg: Config,
    /// Cache mode, named by the caller at construction. `model/` reads no
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
    /// How the `Kernel` form attends a prefill chunk, as every block holds
    /// it. Set through [`Self::set_prefill_attention`] only.
    prefill_attn: crate::kvq::PrefillAttn,
    pub(super) embed: Embed,
    pub blocks: Vec<Block>,
    pub(super) norm: RmsNorm,
    /// The final norm's weight, the tensor the norm holds (4E reads it).
    norm_w: Tensor,
    /// `lm_head`; Qwen3-0.6B and -4B tie it to the embedding matrix.
    head: Head,
    pub(super) rotary: Rotary,
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
            prefill_attn: crate::kvq::PrefillAttn::Grouped,
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
    pub(super) fn causal_mask_offset(&self, b: usize, l: usize, offset: usize) -> Result<Tensor> {
        build_causal_mask(b, l, offset, offset + l, self.dtype, &self.device)
    }

    /// The mask a cached forward of `l` new positions needs, or `None` when it
    /// would be all zeros: one position on the `Cat` store. That is every
    /// decode step of the served path, and building it cost a host vector, an
    /// upload and a cast a token, then one add a layer, for no effect.
    ///
    /// `Grow` hands attention exactly the history, as `Cat` does, so it takes
    /// the same masks.
    pub(super) fn step_mask(&self, b: usize, l: usize, offset: usize) -> Result<Option<Tensor>> {
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

    /// The cache mode this model was built with.
    pub fn kv_mode(&self) -> KvMode {
        self.kv
    }

    /// MEASUREMENT ONLY: install the code of each layer's cache under a
    /// measurement arm of [`KvMode`] ([`KvRoundTrip`]), one entry a block, or
    /// clear it with `None`. A cache in `F16` or `Q8` never calls it, so the
    /// served path is the same with or without it.
    pub fn set_kv_round_trip(&mut self, codes: Option<Vec<Box<dyn KvRoundTrip>>>) -> Result<()> {
        match codes {
            None => self.blocks.iter_mut().for_each(|b| b.kv_round_trip = None),
            Some(a) => {
                if a.len() != self.blocks.len() {
                    candle_core::bail!("{} round-trip entries for {} blocks", a.len(), self.blocks.len());
                }
                for (b, x) in self.blocks.iter_mut().zip(a) {
                    b.kv_round_trip = Some(x);
                }
            }
        }
        Ok(())
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

    /// How the `Kernel` form attends a prefill chunk, in every block
    /// ([`crate::kvq::PrefillAttn`]). Unset, `Grouped`: what every config
    /// served before the field existed.
    pub fn set_prefill_attention(&mut self, p: crate::kvq::PrefillAttn) {
        self.prefill_attn = p;
        for b in &mut self.blocks {
            b.set_prefill_attention(p);
        }
    }

    pub fn prefill_attention(&self) -> crate::kvq::PrefillAttn {
        self.prefill_attn
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

    /// [`Self::prefill_next_in`], keeping the residual entering each of
    /// `layers` at every prompt position: the next token and the taps,
    /// `[tokens][layers][hidden]` binary16 (eager arithmetic: the tree pass's
    /// taps of the same positions agree to rounding, not to the bit). The
    /// first layer must be tapped first: the layers in increasing order.
    pub fn prefill_taps(&self, tokens: &[u32], chunk: usize, caches: &mut [KvCache], layers: &[usize]) -> Result<(u32, Vec<u16>)> {
        if tokens.is_empty() || chunk == 0 || layers.is_empty() || layers.windows(2).any(|w| w[0] >= w[1]) {
            candle_core::bail!("prefill_taps: {} tokens, chunks of {chunk}, layers {layers:?}", tokens.len());
        }
        let hdim = self.cfg.hidden_size;
        let mut cap = TapCapture { layers: layers.to_vec(), chunks: Vec::new() };
        let mut offset = 0;
        let mut h = None;
        for part in tokens.chunks(chunk) {
            let input = Tensor::from_slice(part, (1, part.len()), &self.device)?;
            drop(h.take());
            h = Some(self.hidden_cached(&input, offset, caches, &mut cap)?);
            offset += part.len();
        }
        let next = argmax_last(&self.logits_last(&h.expect("the prompt is not empty"))?)?;
        let nl = layers.len();
        let mut taps = vec![0u16; tokens.len() * nl * hdim];
        let mut p0 = 0;
        for (l, per) in &cap.chunks {
            for (j, v) in per.iter().enumerate() {
                if v.len() != l * hdim {
                    candle_core::bail!("prefill_taps: layer {} tapped {} values for {l} positions", layers[j], v.len());
                }
                for p in 0..*l {
                    taps[((p0 + p) * nl + j) * hdim..((p0 + p) * nl + j + 1) * hdim].copy_from_slice(&v[p * hdim..(p + 1) * hdim]);
                }
            }
            p0 += l;
        }
        Ok((next, taps))
    }

    /// Lower the threshold so a host test reaches the kernel in a short
    /// window. Test builds only: the served threshold is a constant.
    #[cfg(test)]
    pub(super) fn set_kernel_from(&mut self, from: usize) {
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
        // A prefill chunk starts on a drained stream. On CUDA candle allocates
        // from the stream-ordered pool, which hands freed memory back to the
        // driver only at a synchronisation: without this, the last chunk's
        // score tensors stay reserved under this one's (1.07 GB on the 4B at
        // 8,192 tokens, docs/mesures/pic-vram-8k-2026-11-08.txt). It orders
        // nothing that is not already ordered, so no output moves; a decode
        // step or a verify call (at most KMAX rows) does not pay it.
        if l > crate::attn::canon::KMAX {
            self.device.synchronize()?;
        }
        let mask = self.step_mask(b, l, offset)?;
        let mut h = self.embed.forward(input)?;
        for (i, blk) in self.blocks.iter().enumerate() {
            cap.on_residual(i, &h)?;
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
    /// The logits of the LAST position of `h`, in f32.
    ///
    /// `generate` does this inline, in three lines it owns. A caller driving
    /// its own decode loop — `bin/chat`, which needs to sample, stop on a
    /// token and stream — needs the same three and has no access to `head`.
    ///
    /// Exposing the OPERATION rather than the field: `Head` stays private, and
    /// the day it changes shape this signature does not.
    pub fn logits_last(&self, h: &Tensor) -> Result<Tensor> {
        let l = h.dim(1)?;
        let last = h.narrow(1, l - 1, 1)?;
        self.head.project(&last)?.to_dtype(DType::F32)
    }

    pub fn generate(
        &self,
        tokens: &[u32],
        max_new: usize,
        cap: &mut dyn Capture,
    ) -> Result<Vec<u32>> {
        // The stop test is after the push, so `out.len() == 0` is never
        // true: at `max_new = 0` the loop would run until RoPE walks off
        // `max_position_embeddings` — 40,960 decode steps and an opaque
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

    /// Logits `[B, 1, vocab]` of `h` `[B, 1, d]`: the head's own path under
    /// [`crate::batch::BatchProjs::Rows`] (row by row for a quantized table),
    /// the tensor-core kernel under `Mma` when the head is a quantized table.
    pub fn batch_head(
        &self,
        h: &Tensor,
        dev: &dyn crate::batch::BatchDevice,
        projs: crate::batch::BatchProjs,
    ) -> Result<Tensor> {
        let b = h.dim(0)?;
        match (projs, &self.head) {
            (crate::batch::BatchProjs::Mma, Head::Q8(t)) => {
                dev.head_mma(t.as_ref(), &row_block(h)?, b)?.reshape((b, 1, ()))
            }
            (p, Head::Q8(t)) if p.is_gemm() => {
                let x = row_block(h)?.to_dtype(DType::F16)?.contiguous()?;
                let mut y = dev.gemm(&[crate::batch::GemmIn::Head(t.as_ref(), &x)], b, p)?;
                y.pop().expect("one output").reshape((b, 1, ()))
            }
            _ => self.head.project(h),
        }
    }

    /// A fresh [`crate::batch::BatchKv`] for `b` sequences of up to `cap`
    /// positions each, in the model's dtype and on its device.
    /// The bytes [`Self::batch_kv`] allocates for `b` sequences of `cap`
    /// positions: a K and a V buffer a layer, the cap rounded up to 256.
    pub fn batch_kv_bytes(&self, b: usize, cap: usize) -> u64 {
        let cap = cap.div_ceil(256).max(1) * 256;
        let per = 2 * self.cfg.num_key_value_heads * self.cfg.head_dim * self.dtype.size_in_bytes();
        (self.blocks.len() * b * cap * per) as u64
    }

    pub fn batch_kv(&self, b: usize, cap: usize) -> Result<crate::batch::BatchKv> {
        crate::batch::BatchKv::new(
            self.blocks.len(),
            b,
            self.cfg.num_key_value_heads,
            self.cfg.head_dim,
            cap,
            self.dtype,
            &self.device,
        )
    }

    /// One decode step for every sequence of `kv`: `tokens[b]` is fed to
    /// sequence `b` at position `kv.len[b]`, which then grows by one. Returns
    /// the greedy next token of every sequence, as [`Self::generate`] picks
    /// it (f32 logits, candle's argmax).
    pub fn batch_step(
        &self,
        tokens: &[u32],
        kv: &mut crate::batch::BatchKv,
        dev: &dyn crate::batch::BatchDevice,
        projs: crate::batch::BatchProjs,
    ) -> Result<Vec<u32>> {
        let logits = self.batch_logits(tokens, kv, dev, projs)?;
        logits.to_dtype(DType::F32)?.argmax(D::Minus1)?.flatten_all()?.to_vec1::<u32>()
    }

    /// [`Self::batch_logits_slots`]'s greedy picks, one per slot, as
    /// [`Self::batch_step`] takes them.
    pub fn batch_step_slots(
        &self,
        tokens: &[u32],
        slots: &[usize],
        kv: &mut crate::batch::BatchKv,
        dev: &dyn crate::batch::BatchDevice,
        projs: crate::batch::BatchProjs,
    ) -> Result<Vec<u32>> {
        let logits = self.batch_logits_slots(tokens, slots, kv, dev, projs)?;
        logits.to_dtype(DType::F32)?.argmax(D::Minus1)?.flatten_all()?.to_vec1::<u32>()
    }

    /// [`Self::batch_step`]'s logits, `[B, 1, vocab]` in the model dtype.
    pub fn batch_logits(
        &self,
        tokens: &[u32],
        kv: &mut crate::batch::BatchKv,
        dev: &dyn crate::batch::BatchDevice,
        projs: crate::batch::BatchProjs,
    ) -> Result<Tensor> {
        let b = kv.batch();
        if tokens.len() != b {
            candle_core::bail!("{} tokens for a batch of {b}", tokens.len());
        }
        let slots: Vec<usize> = (0..b).collect();
        self.batch_logits_slots(tokens, &slots, kv, dev, projs)
    }

    /// One decode step for the sequences in `slots` of the pool `kv`, the
    /// others untouched: `tokens[i]` is fed to slot `slots[i]` at position
    /// `kv.len[slots[i]]`, which then grows by one. Returns the logits
    /// `[n, 1, vocab]` in the model dtype, row i for slot `slots[i]`.
    pub fn batch_logits_slots(
        &self,
        tokens: &[u32],
        slots: &[usize],
        kv: &mut crate::batch::BatchKv,
        dev: &dyn crate::batch::BatchDevice,
        projs: crate::batch::BatchProjs,
    ) -> Result<Tensor> {
        let b = slots.len();
        if tokens.len() != b || b == 0 {
            candle_core::bail!("{} tokens for {b} slots", tokens.len());
        }
        if b > dev.max_rows(projs) {
            candle_core::bail!("{b} rows in one step, the {} kernels take {}", projs.name(), dev.max_rows(projs));
        }
        let mut seen = vec![false; kv.batch()];
        for &s in slots {
            if s >= kv.batch() || std::mem::replace(&mut seen[s], true) {
                candle_core::bail!("slot {s}: out of a pool of {}, or named twice", kv.batch());
            }
            if kv.len[s] >= kv.cap {
                candle_core::bail!("slot {s} holds {} positions, its capacity {}", kv.len[s], kv.cap);
            }
        }
        let pos_host: Vec<u32> = slots.iter().map(|&s| kv.len[s] as u32).collect();
        let slot_host: Vec<u32> = slots.iter().map(|&s| s as u32).collect();
        let max_len = pos_host.iter().max().copied().unwrap_or(0) as usize + 1;
        let pos = Tensor::from_slice(&pos_host, b, &self.device)?;
        let slot_t = Tensor::from_slice(&slot_host, b, &self.device)?;
        let half = self.rotary.cos.dim(1)?;
        let cos = self.rotary.cos.index_select(&pos, 0)?.reshape((b, 1, half))?;
        let sin = self.rotary.sin.index_select(&pos, 0)?.reshape((b, 1, half))?;
        let input = Tensor::from_slice(tokens, (b, 1), &self.device)?;
        let mut h = self.embed.forward(&input)?;
        for (i, blk) in self.blocks.iter().enumerate() {
            h = blk.forward_batch(&h, &cos, &sin, &kv.k[i], &kv.v[i], &pos, &slot_t, max_len, dev, projs)?;
        }
        let h = self.norm.forward(&h)?;
        let logits = self.batch_head(&h, dev, projs)?;
        for &s in slots {
            kv.len[s] += 1;
        }
        Ok(logits)
    }

    /// [`Self::generate`] with its clock split in two, for a prompt of any
    /// length: the time to the first token, and the decode after it.
    ///
    /// The prompt goes through in chunks of at most `chunk` positions, each
    /// attending over the cache the earlier ones filled. The fused prefill
    /// takes at most [`MAX_PREFILL_ROWS`] rows a call, and the product's
    /// context is 8,192. With `tokens.len() <= chunk` this is `generate`, call
    /// for call. Every step ends on the host read of its token, so both clocks
    /// stop on finished device work.
    pub fn generate_timed(
        &self,
        tokens: &[u32],
        max_new: usize,
        chunk: usize,
    ) -> Result<TimedGeneration> {
        if tokens.is_empty() || max_new == 0 || chunk == 0 {
            candle_core::bail!(
                "generate_timed: {} prompt tokens, {max_new} new, chunks of {chunk}: \
                 there is nothing to time",
                tokens.len()
            );
        }
        let mut caches = self.fresh_caches();
        let t0 = std::time::Instant::now();
        let first = argmax_last(&self.prefill_chunked(tokens, chunk, &mut caches)?)?;
        let prompt_s = t0.elapsed().as_secs_f64();
        let t1 = std::time::Instant::now();
        let out = self.decode_greedy(first, tokens.len(), max_new, &mut caches)?;
        Ok(TimedGeneration {
            tokens: out,
            prompt_s,
            decode_s: t1.elapsed().as_secs_f64(),
        })
    }

    /// The prompt through `caches` in chunks of at most `chunk` positions,
    /// each attending over what the earlier ones filled, and the f32 logits
    /// of its last position, `[1, 1, vocab]`.
    pub fn prefill_chunked(&self, tokens: &[u32], chunk: usize, caches: &mut [KvCache]) -> Result<Tensor> {
        if tokens.is_empty() || chunk == 0 {
            candle_core::bail!("prefill_chunked: {} tokens in chunks of {chunk}", tokens.len());
        }
        let mut offset = 0;
        let mut h = None;
        for part in tokens.chunks(chunk) {
            let input = Tensor::from_slice(part, (1, part.len()), &self.device)?;
            // The last chunk's hidden states are not read again: freed before
            // this chunk allocates, not after it returns.
            drop(h.take());
            h = Some(self.hidden_cached(&input, offset, caches, &mut NoCapture)?);
            offset += part.len();
        }
        self.logits_last(&h.expect("the prompt is not empty"))
    }

    /// One decode step: `token` at absolute position `offset`, appended to
    /// `caches`, and the f32 logits it yields, `[1, 1, vocab]`.
    pub fn step_logits(&self, token: u32, offset: usize, caches: &mut [KvCache]) -> Result<Tensor> {
        let input = Tensor::from_slice(&[token], (1, 1), &self.device)?;
        let h = self.hidden_cached(&input, offset, caches, &mut NoCapture)?;
        self.logits_last(&h)
    }

    /// Greedy decode from `first`, already chosen and fed at position
    /// `offset`, until `max_new` tokens counting `first`. Each step ends on
    /// the host read of its token.
    pub fn decode_greedy(
        &self,
        first: u32,
        offset: usize,
        max_new: usize,
        caches: &mut [KvCache],
    ) -> Result<Vec<u32>> {
        let mut out = Vec::with_capacity(max_new.max(1));
        let (mut next, mut offset) = (first, offset);
        out.push(next);
        while out.len() < max_new {
            next = argmax_last(&self.step_logits(next, offset, caches)?)?;
            out.push(next);
            offset += 1;
        }
        Ok(out)
    }

    /// One window scored token by token through the cache, under two
    /// attention arms at once (a store and a form each), in lockstep.
    ///
    /// Measurement only. Each step feeds the same true token to both arms,
    /// one cache set per arm, and scores the next true token: teacher forcing
    /// on the decode path, where [`Self::window_nll`] scores the prefill path.
    /// Running the arms side by side lets every step compare their logits,
    /// which two separate passes could not do without storing them. The model
    /// is left in the second arm's state; the caller sets what it needs next.
    pub fn decode_nll_paired(
        &mut self,
        tokens: &[u32],
        arms: [(crate::kvq::KvStore, crate::kvq::AttnHeads); 2],
    ) -> Result<PairedDecode> {
        self.decode_nll_split(tokens, arms, 0)
    }

    /// [`Self::decode_nll_paired`], with the steps split at position `split`.
    /// A step at position `p` (the input token's; the cache held `p`
    /// positions before it) below `split` has its two arms' logits compared
    /// bit for bit and counted in [`PairedDecode::below_differ`] when they
    /// differ; a step at `split` or later is also scored in the `tail_*`
    /// fields. `split` 0 makes the tail the whole window.
    pub fn decode_nll_split(
        &mut self,
        tokens: &[u32],
        arms: [(crate::kvq::KvStore, crate::kvq::AttnHeads); 2],
        split: usize,
    ) -> Result<PairedDecode> {
        let kv = self.kv;
        self.decode_nll_modes(tokens, arms.map(|(store, heads)| (store, heads, kv)), split)
    }

    /// [`Self::decode_nll_split`], with each arm's cache mode named: the
    /// `ladder` form of `LLVQ_GQA_PPL` pairs an f16 cache against a
    /// `LLVQ_KV_LADDER` arm under one attention form. Measurement only.
    pub fn decode_nll_modes(
        &mut self,
        tokens: &[u32],
        arms: [(crate::kvq::KvStore, crate::kvq::AttnHeads, KvMode); 2],
        split: usize,
    ) -> Result<PairedDecode> {
        if tokens.len() < 2 {
            candle_core::bail!("decode_nll_paired: a window must hold at least two tokens");
        }
        let n = self.blocks.len();
        let mut caches: [Vec<KvCache>; 2] =
            arms.map(|(store, _, kv)| (0..n).map(|_| KvCache::with_store(kv, store)).collect());
        let mut out = PairedDecode {
            nll: [0.0; 2],
            count: 0,
            top1_agree: 0,
            max_dlogit: 0.0,
            below_differ: 0,
            tail_nll: [0.0; 2],
            tail_count: 0,
            tail_top1_agree: 0,
        };
        for (offset, pair) in tokens.windows(2).enumerate() {
            let (input, target) = (pair[0], pair[1]);
            let mut logits = Vec::with_capacity(2);
            let mut logp = [0.0; 2];
            for (k, &(store, heads, _)) in arms.iter().enumerate() {
                self.set_kv_store(store);
                self.set_attn_heads(heads);
                let l = self.step_logits(input, offset, &mut caches[k])?;
                logp[k] = logp_of(&l, target)?;
                out.nll[k] -= logp[k];
                logits.push(l);
            }
            let agree = argmax_last(&logits[0])? == argmax_last(&logits[1])?;
            out.top1_agree += usize::from(agree);
            let d = (&logits[0] - &logits[1])?.abs()?.max_all()?.to_scalar::<f32>()?;
            out.max_dlogit = out.max_dlogit.max(d);
            out.count += 1;
            if offset < split {
                let bits = |l: &Tensor| -> Result<Vec<u32>> {
                    Ok(l.flatten_all()?.to_vec1::<f32>()?.into_iter().map(f32::to_bits).collect())
                };
                if bits(&logits[0])? != bits(&logits[1])? {
                    out.below_differ += 1;
                }
            } else {
                out.tail_nll[0] -= logp[0];
                out.tail_nll[1] -= logp[1];
                out.tail_count += 1;
                out.tail_top1_agree += usize::from(agree);
            }
        }
        Ok(out)
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
    /// `bin/metalrun` calls it beside a plain `generate` and checks its
    /// tokens. Each phase boundary is a [`Device::synchronize`], so the wall time between two fences belongs to the phase between them
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
            let mask = self.step_mask(1, 1, offset)?;
            let mut hh = e;
            for (i, blk) in self.blocks.iter().enumerate() {
                hh = blk.forward_cached(&hh, &self.rotary, mask.as_ref(), offset, &mut caches[i], i, &mut NoCapture)?;
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
