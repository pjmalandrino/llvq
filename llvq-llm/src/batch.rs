//! Batched decode: B sequences at once, one token each a step, a KV cache per
//! sequence.
//!
//! [`crate::model::Qwen3::batch_step`] runs one step for every sequence of a
//! [`BatchKv`]. A sequence consumes one token a step: the next token of its
//! prompt while it has one (teacher forcing), its last greedy token after.
//! The prompt is therefore read at decode speed, row by row, and every row of
//! a step is a decode row of its own sequence at its own position.
//!
//! ## What is shared and what is not
//!
//! Every projection of a step reads its weights once for the B rows. The
//! attention is per sequence by nature; [`BatchDevice::attend`] does all B in
//! one launch, each against its own cache slot and length. Norms, RoPE and the
//! elementwise operations are row by row, as at batch 1.
//!
//! ## Two kinds of projection
//!
//! [`BatchProjs::Rows`] is the path that exists: `model::group_forward`, which
//! takes the Tetra rows kernel (4 rows a launch, the prefill's) and fans the
//! int4 records and the head out row by row. Every row is then bit for bit the
//! one-row kernel's, so with an attention that returns the decode kernel's
//! bits, a sequence of the batch returns the batch-1 path's tokens exactly.
//! [`BatchProjs::Mma`] sends every projection and the head to the tensor-core
//! kernels of `kernels/tv_batch_mma.cu`, which decode a block once for all
//! rows. Their sums run in another order, so the licence of that arm is a
//! tolerance (CLAUDE.md's T8 band), not identity. [`BatchProjs::Gemm`] is the
//! second tensor-core kernel, `kernels/tv_batch_gemm.cu`: the weight words
//! staged in shared memory ahead of use, the scales folded into f16 weights,
//! one MMA pass, up to 128 rows, and the projections that read one activation
//! (q, k, v; gate, up) in one launch. Same licence as `Mma`.
//!
//! ## Slots
//!
//! A [`BatchKv`] is a pool of slots. [`crate::model::Qwen3::batch_logits_slots`]
//! steps any subset of them, row i in slot `slots[i]`, which is what
//! continuous batching needs: a sequence joins a free slot and leaves it
//! without the others moving.

use candle_core::{DType, Device, Result, Tensor};

/// Which kernels a batched step's projections take.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BatchProjs {
    /// The existing multi-row path, bit for bit with batch 1.
    Rows,
    /// The tensor-core kernels, decode once for B rows.
    Mma,
    /// The staged, folded tensor-core GEMM, several projections a launch.
    Gemm,
}

impl BatchProjs {
    pub fn parse(s: &str) -> std::result::Result<Self, String> {
        match s {
            "rows" => Ok(Self::Rows),
            "mma" => Ok(Self::Mma),
            "gemm" => Ok(Self::Gemm),
            other => Err(format!("batch projections {other:?}: expected \"rows\", \"mma\" or \"gemm\"")),
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::Rows => "rows",
            Self::Mma => "mma",
            Self::Gemm => "gemm",
        }
    }
}

/// One projection of a [`BatchDevice::gemm`] launch, with the rows it reads.
pub enum GemmIn<'a> {
    /// A lattice projection and its rotated rows, `[n, d_in]` f32.
    Lattice(&'a dyn crate::device::LatticeProj, &'a Tensor),
    /// An int4 record and its rows, `[n, d_in]` f16, natural basis.
    Int4(&'a dyn crate::device::Int4Proj, &'a Tensor),
    /// The quantized embedding table as the head, `[n, d]` f16.
    Head(&'a dyn crate::device::QuantEmbedTable, &'a Tensor),
}

/// What a backend owes a batched step: the attention of B sequences in one
/// call, and, for [`BatchProjs::Mma`] and [`BatchProjs::Gemm`], the
/// tensor-core products.
pub trait BatchDevice: Send + Sync {
    fn name(&self) -> &str;

    /// One decode row per sequence. `q` is `[n, n_heads, 1, hd]` after norm
    /// and RoPE, `k_new`/`v_new` `[n, n_kv, 1, hd]`; `kc`/`vc` are the
    /// layer's pool, `[P, n_kv, cap, hd]`. Row `i` is written at position
    /// `pos[i]` of slot `slots[i]` and attends over positions `0..=pos[i]` of
    /// it. `pos` and `slots` are `[n]` u32 on the device, `max_len` is
    /// `max(pos) + 1`. Returns `[n, n_heads, 1, hd]`.
    #[allow(clippy::too_many_arguments)]
    fn attend(
        &self,
        q: &Tensor,
        k_new: &Tensor,
        v_new: &Tensor,
        kc: &Tensor,
        vc: &Tensor,
        pos: &Tensor,
        slots: &Tensor,
        max_len: usize,
    ) -> Result<Tensor>;

    /// `Y = X W^T` for a lattice projection, `xr` `[n, d_in]` f32 rotated.
    fn lattice_mma(&self, p: &dyn crate::device::LatticeProj, xr: &Tensor, n: usize) -> Result<Tensor> {
        let _ = (xr, n);
        candle_core::bail!("{}: no tensor-core kernel for {}", self.name(), p.name())
    }

    /// `Y = X W^T` for an int4 record, `x` `[n, d_in]` f16 natural basis.
    fn int4_mma(&self, p: &dyn crate::device::Int4Proj, x: &Tensor, n: usize) -> Result<Tensor> {
        let _ = (x, n);
        candle_core::bail!("{}: no tensor-core kernel for {}", self.name(), p.name())
    }

    /// Logits `[n, vocab]` from `h` `[n, d]` f16 through a quantized table.
    fn head_mma(&self, t: &dyn crate::device::QuantEmbedTable, h: &Tensor, n: usize) -> Result<Tensor> {
        let _ = (t, h, n);
        candle_core::bail!("{}: no tensor-core head kernel", self.name())
    }

    /// `Y_i = X_i W_i^T` for every projection of `items`, `n` rows each, in
    /// as few launches as the kernel allows. Returns `[n, d_out_i]` f16 each.
    fn gemm(&self, items: &[GemmIn<'_>], n: usize) -> Result<Vec<Tensor>> {
        let _ = (items, n);
        candle_core::bail!("{}: no batched GEMM", self.name())
    }

    /// The most rows one step may carry.
    fn max_rows(&self, projs: BatchProjs) -> usize {
        let _ = projs;
        usize::MAX
    }
}

/// The caches of B sequences: one `[B, n_kv, cap, hd]` pair a layer, and how
/// many positions each sequence holds.
pub struct BatchKv {
    pub k: Vec<Tensor>,
    pub v: Vec<Tensor>,
    pub cap: usize,
    pub len: Vec<usize>,
}

impl BatchKv {
    /// `cap` is rounded up to a multiple of 256, the split the attention
    /// kernel stages.
    pub fn new(
        layers: usize,
        b: usize,
        n_kv: usize,
        hd: usize,
        cap: usize,
        dtype: DType,
        dev: &Device,
    ) -> Result<Self> {
        if b == 0 {
            candle_core::bail!("a batch of zero sequences");
        }
        let cap = cap.div_ceil(256).max(1) * 256;
        let mut k = Vec::with_capacity(layers);
        let mut v = Vec::with_capacity(layers);
        for _ in 0..layers {
            k.push(Tensor::zeros((b, n_kv, cap, hd), dtype, dev)?);
            v.push(Tensor::zeros((b, n_kv, cap, hd), dtype, dev)?);
        }
        Ok(Self { k, v, cap, len: vec![0; b] })
    }

    pub fn batch(&self) -> usize {
        self.len.len()
    }
}

/// A pool of slots for continuous batching: a sequence is admitted into a
/// free slot (its prompt through the batch-1 path, [`crate::model::Qwen3::prefill_slot`]),
/// stepped with whatever other sequences are live, and released, without the
/// others moving. The scheduler that decides who is live is the caller's.
pub struct Pool {
    pub kv: BatchKv,
    free: Vec<usize>,
}

impl Pool {
    /// `slots` sequences of up to `cap` positions each.
    pub fn new(model: &crate::model::Qwen3, slots: usize, cap: usize) -> Result<Self> {
        let kv = model.batch_kv(slots, cap)?;
        Ok(Self { kv, free: (0..slots).rev().collect() })
    }

    pub fn slots(&self) -> usize {
        self.kv.batch()
    }

    pub fn free(&self) -> usize {
        self.free.len()
    }

    /// Positions a slot holds at most.
    pub fn cap(&self) -> usize {
        self.kv.cap
    }

    /// Prefill `prompt` into a free slot. Returns the slot and the first
    /// greedy token, or `None` when every slot is taken.
    pub fn admit(&mut self, model: &crate::model::Qwen3, prompt: &[u32]) -> Result<Option<(usize, u32)>> {
        let Some(slot) = self.free.pop() else { return Ok(None) };
        match model.prefill_slot(prompt, &mut self.kv, slot) {
            Ok(first) => Ok(Some((slot, first))),
            Err(e) => {
                self.release(slot);
                Err(e)
            }
        }
    }

    /// Give a slot back. Its cache is not cleared: the next prefill
    /// overwrites the positions it reads.
    pub fn release(&mut self, slot: usize) {
        if slot < self.kv.batch() && !self.free.contains(&slot) {
            self.kv.len[slot] = 0;
            self.free.push(slot);
        }
    }

    /// One step of the live sequences, `(slot, token to feed)` each; their
    /// greedy picks in the same order.
    pub fn step(
        &mut self,
        model: &crate::model::Qwen3,
        live: &[(usize, u32)],
        dev: &dyn BatchDevice,
        projs: BatchProjs,
    ) -> Result<Vec<u32>> {
        let slots: Vec<usize> = live.iter().map(|l| l.0).collect();
        let tokens: Vec<u32> = live.iter().map(|l| l.1).collect();
        model.batch_step_slots(&tokens, &slots, &mut self.kv, dev, projs)
    }
}

/// The batched attention on any device, from candle operations: the reference
/// the CPU tests drive [`crate::model::Qwen3::batch_step`] with. Each row is
/// written into its slot, then attends over its own prefix, one sequence at a
/// time. Grouped heads, f32 softmax.
pub struct CandleBatch;

impl BatchDevice for CandleBatch {
    fn name(&self) -> &str {
        "candle"
    }

    fn attend(
        &self,
        q: &Tensor,
        k_new: &Tensor,
        v_new: &Tensor,
        kc: &Tensor,
        vc: &Tensor,
        pos: &Tensor,
        slots: &Tensor,
        _max_len: usize,
    ) -> Result<Tensor> {
        let (b, nh, _, hd) = q.dims4()?;
        let n_kv = k_new.dim(1)?;
        let g = nh / n_kv;
        let pos: Vec<u32> = pos.to_vec1()?;
        let slots: Vec<u32> = slots.to_vec1()?;
        let scale = 1.0 / (hd as f64).sqrt();
        let mut outs = Vec::with_capacity(b);
        for (i, &p) in pos.iter().enumerate() {
            let p = p as usize;
            // `slice_set` writes a contiguous block along one dim only, so the
            // row is placed through a host copy of the slot: a reference, not
            // a fast path.
            let si = slots[i] as usize;
            let write = |c: &Tensor, new: &Tensor| -> Result<()> {
                let slot = c.narrow(0, si, 1)?;
                let mut host: Vec<f32> = slot.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
                let row: Vec<f32> = new.narrow(0, i, 1)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
                let cap = c.dim(2)?;
                for h in 0..n_kv {
                    for d in 0..hd {
                        host[(h * cap + p) * hd + d] = row[h * hd + d];
                    }
                }
                let back = Tensor::from_vec(host, (1, n_kv, cap, hd), c.device())?.to_dtype(c.dtype())?;
                c.slice_set(&back, 0, si)
            };
            write(kc, k_new)?;
            write(vc, v_new)?;
            let k = kc.narrow(0, si, 1)?.narrow(2, 0, p + 1)?.to_dtype(DType::F32)?;
            let v = vc.narrow(0, si, 1)?.narrow(2, 0, p + 1)?.to_dtype(DType::F32)?;
            let qi = q.narrow(0, i, 1)?.to_dtype(DType::F32)?.reshape((n_kv, g, hd))?;
            let k = k.reshape((n_kv, p + 1, hd))?;
            let v = v.reshape((n_kv, p + 1, hd))?;
            let s = (qi.matmul(&k.t()?)? * scale)?;
            let a = candle_nn::ops::softmax_last_dim(&s)?;
            let o = a.matmul(&v)?.reshape((1, nh, 1, hd))?;
            outs.push(o.to_dtype(q.dtype())?);
        }
        Tensor::cat(&outs, 0)
    }
}

impl crate::model::Qwen3 {
    /// Run `prompt` through the batch-1 path (the served prefill: the rows
    /// kernel, and the attention form the model's config names), copy its
    /// cache into slot `slot` of `kv`, and return its first greedy token.
    pub fn prefill_slot(&self, prompt: &[u32], kv: &mut BatchKv, slot: usize) -> Result<u32> {
        use candle_core::{IndexOp, D};
        if prompt.is_empty() || slot >= kv.batch() || prompt.len() > kv.cap {
            candle_core::bail!("prefill_slot: {} tokens into slot {slot} of {} (cap {})", prompt.len(), kv.batch(), kv.cap);
        }
        let mut caches = self.fresh_caches();
        let mut offset = 0usize;
        let mut last = None;
        for chunk in prompt.chunks(crate::model::MAX_ROWS) {
            let input = Tensor::from_slice(chunk, (1, chunk.len()), self.device())?;
            last = Some(self.hidden_cached(&input, offset, &mut caches, &mut crate::model::NoCapture)?);
            offset += chunk.len();
        }
        let h = last.expect("a prompt of at least one token");
        for (l, c) in caches.iter().enumerate() {
            let (k, v) = c.held()?.ok_or_else(|| candle_core::Error::msg("prefill_slot: an empty cache"))?;
            let pad = kv.cap - k.dim(2)?;
            kv.k[l].slice_set(&k.pad_with_zeros(2, 0, pad)?.contiguous()?, 0, slot)?;
            kv.v[l].slice_set(&v.pad_with_zeros(2, 0, pad)?.contiguous()?, 0, slot)?;
        }
        kv.len[slot] = prompt.len();
        self.logits_last(&h)?.i((0, 0))?.argmax(D::Minus1)?.to_scalar::<u32>()
    }

    /// Negative log-likelihood of each window, every position scored by a
    /// decode row of the batched step (teacher forcing): the decode-path
    /// perplexity of B windows at once. Returns `(sum of -log p, positions)`
    /// per window. Windows share one length.
    pub fn batch_window_nll(
        &self,
        windows: &[&[u32]],
        dev: &dyn BatchDevice,
        projs: BatchProjs,
    ) -> Result<Vec<(f64, usize)>> {
        let b = windows.len();
        let l = windows.first().map(|w| w.len()).unwrap_or(0);
        if b == 0 || l < 2 || windows.iter().any(|w| w.len() != l) {
            candle_core::bail!("batch_window_nll: {b} windows of unequal or short length");
        }
        let mut kv = self.batch_kv(b, l)?;
        let mut nll = vec![0f64; b];
        for t in 0..l - 1 {
            let tokens: Vec<u32> = windows.iter().map(|w| w[t]).collect();
            let targets: Vec<u32> = windows.iter().map(|w| w[t + 1]).collect();
            let logits = self.batch_logits(&tokens, &mut kv, dev, projs)?.to_dtype(DType::F32)?.reshape((b, ()))?;
            let logp = candle_nn::ops::log_softmax(&logits, candle_core::D::Minus1)?;
            let tg = Tensor::from_slice(&targets, (b, 1), self.device())?;
            let picked: Vec<f32> = logp.gather(&tg, 1)?.flatten_all()?.to_vec1()?;
            for (n, p) in nll.iter_mut().zip(picked) {
                *n -= p as f64;
            }
        }
        Ok(nll.into_iter().map(|n| (n, l - 1)).collect())
    }

    /// Greedy generation for several prompts at once through
    /// [`Self::batch_step`]: up to `max_new` tokens each, a sequence stopping
    /// early when `stop(index, generated so far)` says so.
    ///
    /// `prefill` true runs each prompt through the batch-1 path first
    /// ([`Self::prefill_slot`]) and batches the decode only. `false` feeds the
    /// prompts one token a step (teacher forcing), so every row of every step
    /// is batched from the first. A sequence that has stopped keeps its row in
    /// the batch until the last one stops, its picks discarded: the minimal
    /// driver, not continuous batching.
    pub fn generate_batch(
        &self,
        prompts: &[Vec<u32>],
        max_new: usize,
        dev: &dyn BatchDevice,
        projs: BatchProjs,
        prefill: bool,
        stop: &dyn Fn(usize, &[u32]) -> bool,
    ) -> Result<Vec<Vec<u32>>> {
        let b = prompts.len();
        if b == 0 || max_new == 0 {
            return Ok(vec![Vec::new(); b]);
        }
        if let Some(i) = prompts.iter().position(|p| p.is_empty()) {
            candle_core::bail!("prompt {i} is empty");
        }
        let longest = prompts.iter().map(Vec::len).max().unwrap_or(0);
        let mut kv = self.batch_kv(b, longest + max_new)?;
        let mut fed: Vec<usize> = vec![0; b];
        let mut out: Vec<Vec<u32>> = vec![Vec::with_capacity(max_new); b];
        let mut done = vec![false; b];
        let mut next: Vec<u32> = prompts.iter().map(|p| p[0]).collect();
        if prefill {
            for (i, p) in prompts.iter().enumerate() {
                let first = self.prefill_slot(p, &mut kv, i)?;
                fed[i] = p.len();
                out[i].push(first);
                next[i] = first;
                done[i] = max_new == 1 || stop(i, &out[i]);
            }
        }
        while done.iter().any(|d| !d) {
            let picks = self.batch_step(&next, &mut kv, dev, projs)?;
            for i in 0..b {
                if done[i] {
                    continue;
                }
                if fed[i] < prompts[i].len() {
                    fed[i] += 1;
                }
                if fed[i] < prompts[i].len() {
                    next[i] = prompts[i][fed[i]];
                    continue;
                }
                out[i].push(picks[i]);
                next[i] = picks[i];
                if out[i].len() == max_new || stop(i, &out[i]) {
                    done[i] = true;
                }
            }
        }
        Ok(out)
    }
}
