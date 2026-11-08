//! The generation loops of [`Qwen3`]: greedy, timed, batched, paired and
//! phased decodes over the KV cache.

use super::*;

impl Qwen3 {
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
            _ => self.head.project(h),
        }
    }

    /// A fresh [`crate::batch::BatchKv`] for `b` sequences of up to `cap`
    /// positions each, in the model's dtype and on its device.
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
        if let Some(full) = kv.len.iter().position(|&l| l >= kv.cap) {
            candle_core::bail!("sequence {full} holds {} positions, its slot {}", kv.len[full], kv.cap);
        }
        let pos_host: Vec<u32> = kv.len.iter().map(|&l| l as u32).collect();
        let max_len = kv.len.iter().max().copied().unwrap_or(0) + 1;
        let pos = Tensor::from_slice(&pos_host, b, &self.device)?;
        let half = self.rotary.cos.dim(1)?;
        let cos = self.rotary.cos.index_select(&pos, 0)?.reshape((b, 1, half))?;
        let sin = self.rotary.sin.index_select(&pos, 0)?.reshape((b, 1, half))?;
        let input = Tensor::from_slice(tokens, (b, 1), &self.device)?;
        let mut h = self.embed.forward(&input)?;
        for (i, blk) in self.blocks.iter().enumerate() {
            h = blk.forward_batch(&h, &cos, &sin, &kv.k[i], &kv.v[i], &pos, max_len, dev, projs)?;
        }
        let h = self.norm.forward(&h)?;
        let logits = self.batch_head(&h, dev, projs)?;
        for l in kv.len.iter_mut() {
            *l += 1;
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
        if tokens.len() < 2 {
            candle_core::bail!("decode_nll_paired: a window must hold at least two tokens");
        }
        let (n, kv) = (self.blocks.len(), self.kv);
        let mut caches: [Vec<KvCache>; 2] =
            arms.map(|(store, _)| (0..n).map(|_| KvCache::with_store(kv, store)).collect());
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
            for (k, &(store, heads)) in arms.iter().enumerate() {
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
}

/// What [`Qwen3::decode_nll_paired`] returns for one window.
#[derive(Debug, Clone, PartialEq)]
pub struct PairedDecode {
    /// Summed negative log-likelihood of each arm, in nats.
    pub nll: [f64; 2],
    /// Predictions scored, the window length minus one.
    pub count: usize,
    /// Steps where both arms put the same token on top.
    pub top1_agree: usize,
    /// The largest absolute difference between the two arms' logits, over
    /// every vocabulary entry of every step.
    pub max_dlogit: f32,
    /// Steps below the split whose two logit vectors differ in any bit.
    pub below_differ: usize,
    /// [`Self::nll`] over the steps at the split or later.
    pub tail_nll: [f64; 2],
    /// Steps at the split or later.
    pub tail_count: usize,
    /// [`Self::top1_agree`] over the steps at the split or later.
    pub tail_top1_agree: usize,
}

/// One greedy generation from [`Qwen3::generate_timed`], with its two clocks.
pub struct TimedGeneration {
    pub tokens: Vec<u32>,
    /// From the first prompt token sent to the first new token read back.
    pub prompt_s: f64,
    /// The `tokens.len() - 1` decode steps after the first new token.
    pub decode_s: f64,
}
