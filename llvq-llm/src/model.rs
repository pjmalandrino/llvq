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
    /// Whether this capture reads activations at all. A block replaces its
    /// eager MLP ops by route P's segment only when it does not, since the
    /// segment exposes no intermediate. True unless an implementation says
    /// otherwise.
    fn wants_activations(&self) -> bool {
        true
    }
}

/// The do-nothing capture, for plain inference.
pub struct NoCapture;

impl Capture for NoCapture {
    fn on_activation(&mut self, _: usize, _: Act, _: &Tensor) -> Result<()> {
        Ok(())
    }
    fn wants_activations(&self) -> bool {
        false
    }
}

/// The key argmax as the decode loop sees it: write the greedy token of
/// `logits` (`[1, 1, vocab]`) into `ring[pos]`, on the device, with candle's
/// tie order, and never read anything back to the host.
///
/// CUDA: `fused_cuda::KeyArgmax`, the kernel `tok_argmax_h` as an in-place op
/// on the ring (no allocation, no upload), checked at load against candle's
/// own `fast_argmax_f16`. CPU tests: a stand-in that pins the plumbing of the
/// loops; the key itself is certified by `crate::argmax_key`.
pub trait KeyKernel: Send + Sync {
    fn write(&self, logits: &Tensor, ring: &Tensor, pos: usize) -> Result<()>;
    /// Zero the sticky NaN flag before a generation.
    fn arm(&self) -> Result<()>;
    /// Whether a NaN logit was met since [`Self::arm`]. A 4-byte read, once a
    /// generation.
    fn nan_seen(&self) -> Result<bool>;
}

/// How [`Qwen3::generate_loop`] picks and feeds each token. Each kind moves one
/// mechanism against its neighbour, so each A/B ratio isolates one gesture:
///
/// | kind | per token | ratio | gesture |
/// |---|---|---|---|
/// | `generate` | head, f32 cast, candle argmax, readback, upload | | today |
/// | `F16` | without the cast | `F16 / generate` | a1 |
/// | `Key` | F16 with the key kernel into a one-slot ring, readback, upload | `Key / F16` | a2 |
/// | `OnDevKey` | Key without the readback and upload: the next step embeds `ring[pos]` | `OnDevKey / Key` | a3 |
/// | `OnDevCandle` | F16 without the readback and upload: candle's argmax output is the next input | `OnDevCandle / F16` | a3, candle's kernel |
pub enum TokenLoop<'a> {
    F16,
    Key(&'a dyn KeyKernel),
    OnDevKey(&'a dyn KeyKernel),
    OnDevCandle,
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
        Self::apply_tables(q, k, &cos, &sin)
    }

    /// The same rotation from CALLER-OWNED tables — the stepped/capturable
    /// path. The tables are stable buffers whose content a driver refreshes
    /// outside the capture (via [`Self::refresh_step`]); inside the closure
    /// nothing varies but bytes behind stable pointers.
    fn apply_tables(q: &Tensor, k: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<(Tensor, Tensor)> {
        Ok((
            candle_nn::rotary_emb::rope(&q.contiguous()?, cos, sin)?,
            candle_nn::rotary_emb::rope(&k.contiguous()?, cos, sin)?,
        ))
    }

    /// Stable one-position tables for the stepped path, born zeroed.
    pub fn step_tables(&self) -> Result<(Tensor, Tensor)> {
        let half = self.cos.dim(1)?;
        Ok((
            Tensor::zeros((1, half), self.cos.dtype(), self.cos.device())?,
            Tensor::zeros((1, half), self.sin.dtype(), self.sin.device())?,
        ))
    }

    /// Refresh the stable tables to `offset` — an in-place `slice_set` at a
    /// CONSTANT destination offset, done outside any capture.
    pub fn refresh_step(&self, cos_step: &Tensor, sin_step: &Tensor, offset: usize) -> Result<()> {
        cos_step.slice_set(&self.cos.narrow(0, offset, 1)?, 0, 0)?;
        sin_step.slice_set(&self.sin.narrow(0, offset, 1)?, 0, 0)?;
        Ok(())
    }
}

/// Where a cached forward finds its rotation: a host `offset` (the shipped
/// path — the pointer into the tables is recomputed each call), or stable
/// `Tables` owned by a stepped driver (the capturable path — nothing about
/// the call varies with the token). One enum instead of a second forward:
/// the two paths CANNOT drift apart because they are the same code.
pub enum RopeAt<'a> {
    Offset(usize),
    Tables { cos: &'a Tensor, sin: &'a Tensor },
}

/// One block's keys and values, for every position already seen.
///
/// Stored **before** `repeat_kv`: the grouped-query expansion is a view over
/// `n_kv` heads, so caching the expanded form would hold `n_heads / n_kv`
/// times the bytes for nothing. On Qwen3-4B that is a factor 4.
///
/// 🕳️ **That argument holds for `Cat`, and was measured NOT to hold for a
/// bounded window (2026-09-01, step 1b).** Under
/// [`crate::kvq::KvStore::Prealloc`] this cache stores the EXPANDED form in
/// a fixed `[b, n_heads, W, hd]` buffer: ×4 of a bounded window is ~151 MB
/// at W = 256 on the 4B — trivial — and it removes the real per-step cost,
/// the ×4 HISTORY copy `repeat_kv` performs on every decode step of the
/// `Cat` path.
/// ⚠️ **No `Default`, deliberately.** Every construction site must name the
/// mode it caches in. With a `Default`, an incomplete wiring compiles and an
/// unwired arm reports Δppl = 0, ΔMMLU = 0 and 1.00× throughput — three greens
/// that mean nothing. The compiler is the cheapest place to catch that.
pub struct KvCache {
    k: Option<Tensor>,
    v: Option<Tensor>,
    mode: KvMode,
    /// The capturable write path: when set, the `Prealloc` arm writes through
    /// `scatter_set` with THIS index tensor — a stable device buffer whose
    /// CONTENT is the position, refreshed outside any capture. All caches of
    /// a model share one underlying storage (`Tensor::clone` shares), so one
    /// 4-byte refresh moves every layer's write. When unset (the shipped
    /// path, and the prefill), `slice_set` at the host-side `len` is used —
    /// its baked offset is exactly what a CUDA graph cannot replay, which is
    /// why this field exists.
    write_idx: Option<Tensor>,
    /// How the history grows — [`KvStore::Cat`] (shipped), a preallocated
    /// window, or chunks grown on demand. Under `Prealloc` and `Grow`, `k`/`v`
    /// hold the FULL buffers and `len` counts the valid positions; under `Cat`
    /// they hold exactly the history and `len` is unused.
    store: crate::kvq::KvStore,
    len: usize,
}

impl KvCache {
    /// An empty cache, in the named mode, growing by concatenation.
    pub fn new(mode: KvMode) -> Self {
        Self::with_store(mode, crate::kvq::KvStore::Cat)
    }

    /// An empty cache, naming both the payload mode and the growth store.
    pub fn with_store(mode: KvMode, store: crate::kvq::KvStore) -> Self {
        Self {
            k: None,
            v: None,
            mode,
            store,
            len: 0,
            write_idx: None,
        }
    }

    /// Append this step's keys and values, and return the whole history.
    ///
    /// Concatenation, not a preallocated ring: a ring needs a maximum length
    /// decided in advance, and every length in this repository is a property
    /// of the corpus rather than of the model. The copy is `O(context)` per
    /// step against the `O(context²)` full re-run it replaces.
    ///
    /// Under [`KvMode::Q8`] the **new** keys and values are quantized on the
    /// way in, once, and the history is kept in its reconstructed form —
    /// requantizing the whole history at every step would be `O(context)` of
    /// pointless work and, worse, would make a token's bytes depend on when it
    /// was read. What this function returns is therefore exactly what it
    /// stores, which is what `repeat_kv` and the attention matmul consume:
    /// there is no separate read path that could disagree with the write path.
    fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        let (k, v) = match self.mode {
            KvMode::F16 => (k.clone(), v.clone()),
            KvMode::Q8 => (
                crate::kvq::quantize_dequantize(k)?,
                crate::kvq::quantize_dequantize(v)?,
            ),
        };
        match self.store {
            crate::kvq::KvStore::Prealloc(w) => return self.append_prealloc(w, &k, &v),
            crate::kvq::KvStore::Grow(chunk) => return self.append_grow(chunk, &k, &v),
            crate::kvq::KvStore::Cat => {}
        }
        let k = match &self.k {
            None => k,
            Some(p) => Tensor::cat(&[p, &k], 2)?.contiguous()?,
        };
        let v = match &self.v {
            None => v,
            Some(p) => Tensor::cat(&[p, &v], 2)?.contiguous()?,
        };
        self.k = Some(k.clone());
        self.v = Some(v.clone());
        Ok((k, v))
    }

    /// The `Prealloc` arm of [`Self::append`]: write in place, return the
    /// FULL fixed buffers.
    ///
    /// The buffers are allocated on first use, from the shape/dtype/device of
    /// what arrives — the cache needs no model handle, and the CALLER decides
    /// what head count arrives: the block hands the EXPANDED step in, so the
    /// buffer is `[b, n_heads, W, hd]` and `repeat_kv` has no call site on
    /// this path. `slice_set` demands contiguity on both sides; the buffers
    /// are born contiguous and the incoming step is made so (a copy of `l`
    /// positions, not of the history).
    ///
    /// What goes OUT are the full `W`-wide buffers — contiguous, at a stable
    /// address: the shape the wide causal mask expects and the shape a CUDA
    /// graph can capture. 🕳️ The first form of this arm (step 1, 2026-09-01)
    /// returned `narrow` views instead: non-contiguous at every `len < W`,
    /// they pushed `repeat_kv`'s per-step history copy into its strided
    /// paths — r = 0.8919 [0.8884–0.8953] on CUDA, ~−29% on Metal. Measured,
    /// then replaced by this form (step 1b). `generate` and
    /// `generate_uncached` staying token-identical under `LLVQ_VERIFY_CACHE`
    /// remains the end-to-end witness.
    ///
    /// Overflow is a NAMED error: a silent ring would change what attention
    /// sees. The window is the operator's promise, and this arm holds it.
    fn append_prealloc(&mut self, w: usize, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        let l = k.dim(2)?;
        if self.k.is_none() {
            let (b, n_heads, _, hd) = k.dims4()?;
            self.k = Some(Tensor::zeros((b, n_heads, w, hd), k.dtype(), k.device())?);
            self.v = Some(Tensor::zeros((b, n_heads, w, hd), v.dtype(), v.device())?);
        }
        if self.len + l > w {
            candle_core::bail!(
                "KvCache prealloc: {} + {l} positions > window {w} (LLVQ_KV_PREALLOC). \
                 The window is a hard bound, not a ring.",
                self.len
            );
        }
        let (kb, vb) = (self.k.as_ref().unwrap(), self.v.as_ref().unwrap());
        // The scatter path serves the DECODE step (l = 1, the captured
        // shape). A multi-position prefill on an armed cache falls back to
        // `slice_set` — outside any capture by construction, since arming
        // happens after the prefill and a round reset keeps the arm.
        match self.write_idx.as_ref().filter(|_| l == 1) {
            None => {
                kb.slice_set(&k.contiguous()?, 2, self.len)?;
                vb.slice_set(&v.contiguous()?, 2, self.len)?;
            }
            // Indirection by CONTENT: the kernel reads the position from the
            // index tensor, so the captured node replays correctly at every
            // token. `scatter_set` wants indexes shaped like the source; the
            // driver refreshes the full (tiny) buffer outside the capture.
            Some(idx) => {
                kb.scatter_set(idx, &k.contiguous()?, 2)?;
                vb.scatter_set(idx, &v.contiguous()?, 2)?;
            }
        }
        self.len += l;
        Ok((kb.clone(), vb.clone()))
    }

    /// The `Grow` arm of [`Self::append`]: write in place, return views of
    /// the valid prefix.
    ///
    /// The buffers hold a whole number of `chunk` positions. When the new
    /// positions do not fit, both are reallocated to the next multiple and
    /// the valid prefix is copied once; at every other step the only write is
    /// the `l` new positions. What goes out are `narrow` views, whose batch
    /// stride is the capacity: the matmuls of grouped attention read them
    /// where they are (candle passes the stride to cuBLAS, Metal and the CPU
    /// gemm). Expanded attention accepts them too, and pays the strided copy
    /// of `repeat_kv`; that pairing is correct and never served.
    fn append_grow(&mut self, chunk: usize, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        if chunk == 0 {
            candle_core::bail!("KvCache grow: a chunk of zero positions never fits anything");
        }
        let l = k.dim(2)?;
        let need = self.len + l;
        let cap = match &self.k {
            Some(b) => b.dim(2)?,
            None => 0,
        };
        if need > cap {
            let cap = need.div_ceil(chunk) * chunk;
            self.k = Some(grown(self.k.as_ref(), k, self.len, cap)?);
            self.v = Some(grown(self.v.as_ref(), v, self.len, cap)?);
            settle_growth(k)?;
        }
        let (kb, vb) = (self.k.as_ref().unwrap(), self.v.as_ref().unwrap());
        kb.slice_set(&k.contiguous()?, 2, self.len)?;
        vb.slice_set(&v.contiguous()?, 2, self.len)?;
        self.len = need;
        Ok((kb.narrow(2, 0, need)?, vb.narrow(2, 0, need)?))
    }

    /// Keep the first `len` positions and forget the rest: the next append
    /// writes at `len`. A measurement uses it to decode several times after
    /// one prefill. Under `Cat` the kept history is copied out; the other
    /// stores only move their count, since the positions past it are either
    /// masked (`Prealloc`) or outside the view (`Grow`).
    pub fn truncate(&mut self, len: usize) -> Result<()> {
        let have = match self.store {
            crate::kvq::KvStore::Cat => match &self.k {
                Some(k) => k.dim(2)?,
                None => 0,
            },
            crate::kvq::KvStore::Prealloc(_) | crate::kvq::KvStore::Grow(_) => self.len,
        };
        if len > have {
            candle_core::bail!("KvCache truncate: {len} positions asked, {have} held");
        }
        match self.store {
            crate::kvq::KvStore::Cat if len == 0 => {
                self.k = None;
                self.v = None;
            }
            crate::kvq::KvStore::Cat => {
                if let (Some(k), Some(v)) = (&self.k, &self.v) {
                    let (k, v) = (k.narrow(2, 0, len)?.contiguous()?, v.narrow(2, 0, len)?.contiguous()?);
                    self.k = Some(k);
                    self.v = Some(v);
                }
            }
            crate::kvq::KvStore::Prealloc(_) | crate::kvq::KvStore::Grow(_) => self.len = len,
        }
        Ok(())
    }

    /// Arm the capturable write path — see the `write_idx` field. The stepped
    /// driver calls this AFTER the prefill (whose multi-position write stays
    /// on `slice_set`), with clones of one shared index tensor.
    /// Positions held.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The `Grow` store's two buffers, `(1, n_kv, cap, head_dim)`, once a
    /// first append made them. 4E reads and writes them in place.
    pub fn grow_buffers(&self) -> Option<(&Tensor, &Tensor)> {
        match (self.store, &self.k, &self.v) {
            (crate::kvq::KvStore::Grow(_), Some(k), Some(v)) => Some((k, v)),
            _ => None,
        }
    }

    /// Grow the `Grow` buffers to hold `need` positions, by the store's own
    /// chunk rule, keeping the positions held. 4E calls it before a launch
    /// writes positions past the capacity.
    pub fn reserve(&mut self, need: usize) -> Result<()> {
        let crate::kvq::KvStore::Grow(chunk) = self.store else {
            candle_core::bail!("KvCache reserve: only the Grow store is written in place");
        };
        let (Some(k), Some(v)) = (self.k.as_ref(), self.v.as_ref()) else {
            candle_core::bail!("KvCache reserve: no buffer before a first append");
        };
        if need > k.dim(2)? {
            let cap = need.div_ceil(chunk) * chunk;
            let (nk, nv) = (grown(Some(k), k, self.len, cap)?, grown(Some(v), v, self.len, cap)?);
            self.k = Some(nk);
            self.v = Some(nv);
            settle_growth(self.k.as_ref().expect("just set"))?;
        }
        Ok(())
    }

    /// After 4E wrote positions `len..n` into the buffers in place.
    pub fn set_len(&mut self, n: usize) -> Result<()> {
        let cap = self.grow_buffers().map(|(k, _)| k.dim(2)).transpose()?.unwrap_or(0);
        if n > cap {
            candle_core::bail!("KvCache set_len: {n} positions, the buffers hold {cap}");
        }
        self.len = n;
        Ok(())
    }

    pub fn set_write_index(&mut self, idx: Tensor) {
        self.write_idx = Some(idx);
    }

    /// Start a new generation WITHOUT reallocating: the window buffers, and
    /// therefore every pointer a captured graph baked, stay where they are.
    /// Stale positions past the new `len` are unreachable behind the wide
    /// causal mask — the same validity rule that makes the zero tail inert.
    pub fn reset(&mut self) {
        self.len = 0;
    }

    /// Which store this cache grows by — the block branches on it.
    pub fn store(&self) -> crate::kvq::KvStore {
        self.store
    }

    /// The history held, `(1, n_kv, len, head_dim)` views of keys and values,
    /// for a store that keeps the grouped heads (`Cat`, `Grow`). `None` before
    /// a first append. The batched driver copies a prefilled sequence into its
    /// slot through this.
    pub fn held(&self) -> Result<Option<(Tensor, Tensor)>> {
        let (Some(k), Some(v)) = (&self.k, &self.v) else {
            return Ok(None);
        };
        match self.store {
            crate::kvq::KvStore::Cat => Ok(Some((k.clone(), v.clone()))),
            crate::kvq::KvStore::Grow(_) => Ok(Some((k.narrow(2, 0, self.len)?, v.narrow(2, 0, self.len)?))),
            crate::kvq::KvStore::Prealloc(_) => {
                candle_core::bail!("KvCache held: a preallocated store holds the expanded heads")
            }
        }
    }
}

/// After a `Grow` cache moved to bigger buffers: drain the stream, so the
/// stream-ordered pool on CUDA hands the old pair back to the driver now.
///
/// Without it every layer's old pair stays reserved until the next
/// synchronisation, and a pool block of the old size cannot hold the new one:
/// at the growth past an 8,192-token prompt, 4E's `reserve` would hold the
/// whole cache twice (2.42 GB on the 4B) for the length of the loop. One
/// drain per layer and per growth, every 256 positions; no output moves.
fn settle_growth(like: &Tensor) -> Result<()> {
    like.device().synchronize()
}

/// A `Grow` buffer of `cap` positions, shaped like `like` on every other
/// axis, holding the first `len` positions of `old` and zeros after them.
fn grown(old: Option<&Tensor>, like: &Tensor, len: usize, cap: usize) -> Result<Tensor> {
    let (b, heads, _, hd) = like.dims4()?;
    let fresh = Tensor::zeros((b, heads, cap, hd), like.dtype(), like.device())?;
    if let Some(old) = old.filter(|_| len > 0) {
        fresh.slice_set(&old.narrow(2, 0, len)?.contiguous()?, 2, 0)?;
    }
    Ok(fresh)
}

/// `t` with its last two axes row-major, as the CUDA and Metal matmuls require
/// of each batch element, whatever its batch stride.
///
/// A view of a `Grow` buffer is already so and passes as it is: copying it
/// would be the history copy the store exists to avoid. The first append of a
/// `Cat` cache hands back the projection output as transposed, rows of one head
/// interleaved with the others; that one is copied. The CPU gemm accepts both,
/// which is why the CPU tests did not see it and the card did (deviation É1 of
/// the GQA perplexity prereg).
fn rows_contiguous(t: &Tensor) -> Result<Tensor> {
    let (dims, stride) = (t.dims(), t.stride());
    let n = dims.len();
    let row_major = n >= 2 && stride[n - 1] == 1 && (stride[n - 2] == dims[n - 1] || dims[n - 2] == 1);
    if row_major {
        Ok(t.clone())
    } else {
        t.contiguous()
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
fn grouped_attention(
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
    let raw = q.matmul(&k.transpose(2, 3)?)?;
    let p = match mask {
        Some(m) if in_place(&raw) => {
            softmax_in_place(&raw, m, scale, l)?;
            raw
        }
        // Each step drops its input before the next allocates: two score
        // tensors alive at most, where three shadowed bindings kept three.
        Some(m) => {
            let scores = {
                let scaled = (raw * scale)?;
                scaled
                    .reshape((b, n_kv, groups, l, kv_len))?
                    .broadcast_add(&m.unsqueeze(2)?)?
                    .reshape((b, n_kv, groups * l, kv_len))?
            };
            candle_nn::ops::softmax_last_dim(&scores)?
        }
        None => candle_nn::ops::softmax_last_dim(&(raw * scale)?)?,
    };
    p.matmul(v)?.reshape((b, n_heads, l, hd))
}

/// Score tensors of at least this many bytes take [`softmax_in_place`]. Below
/// it the three element-wise kernels run once on the whole tensor: fewer
/// launches, for a transient that does not matter. Both paths return the same
/// bits, so the threshold moves memory and launches, never an output.
const IN_PLACE_BYTES: usize = 64 << 20;

/// Rows one block of [`softmax_in_place`] carries: its two transients are
/// `SOFTMAX_ROWS × L` scores each, 16.8 MB in f16 at L = 8,192.
const SOFTMAX_ROWS: usize = 1024;

/// MEASUREMENT ONLY, `false` everywhere else: every masked softmax runs on
/// the whole score tensor, the memory form of every run before 2026-11-08.
/// `bin/fusedrun` sets it from `LLVQ_SOFTMAX_WHOLE=1` so a second process can
/// check on the card that the in-place form returns the same logits; it moves
/// memory and nothing else, which is what that check confirms. Process-wide
/// because the attention bodies are free functions; nothing served sets it.
static SOFTMAX_WHOLE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_softmax_whole_for_measurement(on: bool) {
    SOFTMAX_WHOLE.store(on, std::sync::atomic::Ordering::Relaxed);
}

#[cfg(not(test))]
fn in_place_from() -> usize {
    match SOFTMAX_WHOLE.load(std::sync::atomic::Ordering::Relaxed) {
        true => usize::MAX,
        false => IN_PLACE_BYTES,
    }
}

/// What a run line prints for the softmax's memory form.
pub fn softmax_form() -> &'static str {
    match in_place_from() {
        usize::MAX => "whole tensor (LLVQ_SOFTMAX_WHOLE=1)",
        _ => "in place",
    }
}

#[cfg(not(test))]
fn softmax_rows() -> usize {
    SOFTMAX_ROWS
}

// The host tests move both knobs per thread (each test runs on its own), so
// a tiny model reaches the in-place path and its remainder blocks.
#[cfg(test)]
thread_local! {
    static IN_PLACE_FROM: std::cell::Cell<usize> = const { std::cell::Cell::new(IN_PLACE_BYTES) };
    static SOFTMAX_ROWS_T: std::cell::Cell<usize> = const { std::cell::Cell::new(SOFTMAX_ROWS) };
}

#[cfg(test)]
fn in_place_from() -> usize {
    IN_PLACE_FROM.with(|c| c.get())
}

#[cfg(test)]
fn softmax_rows() -> usize {
    SOFTMAX_ROWS_T.with(|c| c.get())
}

fn in_place(raw: &Tensor) -> bool {
    raw.elem_count() * raw.dtype().size_in_bytes() >= in_place_from()
}

/// `softmax(raw · scale + mask)` over the last axis, written back into `raw`.
///
/// `raw` is the contiguous output of the score matmul, `[b, slabs, rows, L]`
/// with `rows` a whole number of blocks of `l` query rows: the grouped form's
/// `[b, n_kv, groups·l, L]` and the expanded form's `[b, n_heads, l, L]` are
/// both `b·slabs·rows / l` slabs of `l` rows, row `i` of a slab under row `i`
/// of `mask`, `[mb, 1, l, L]` with `mb` 1 or `b`.
///
/// Why: the whole-tensor form keeps the scaled, the masked and the softmaxed
/// scores alive, three tensors of `heads × l × L` each, 6.44 GB on the 4B at
/// the second chunk of an 8,192-token prompt. Here `raw` is the only one, plus
/// two blocks of [`SOFTMAX_ROWS`] rows.
///
/// Why it keeps every bit: the matmuls on both sides of it see the operands
/// they saw before, shape and strides included, so cuBLAS (and the CPU and
/// Metal gemms) take the same algorithm. Between them each element meets the
/// same three operations in the same order: the affine `x · scale + 0` (what
/// `Tensor * f64` is), then `x + m` with `m` either +0 or -inf, then a softmax
/// whose kernel reduces one row at a time and reads nothing outside it (one
/// warp a row on CUDA, `candle-kernels/src/reduce.cu`). A block is a
/// contiguous run of whole rows, so it takes each kernel's contiguous branch,
/// the branch the whole tensor took; the mask add, which ran broadcast,
/// becomes an add of equal shapes, the same `x + y` per element. The copy back
/// is a copy.
fn softmax_in_place(raw: &Tensor, mask: &Tensor, scale: f64, l: usize) -> Result<()> {
    let dims = raw.dims();
    let (b, kv_len) = (dims[0], dims[dims.len() - 1]);
    let rows_total = raw.elem_count() / kv_len;
    if !raw.is_contiguous() || dims.len() != 4 || l == 0 || !dims[2].is_multiple_of(l) {
        candle_core::bail!("softmax_in_place: scores {dims:?}, not contiguous slabs of {l} rows");
    }
    let (mb, m_rows, m_cols) = (mask.dim(0)?, mask.dim(2)?, mask.dim(3)?);
    if mask.dims4()?.1 != 1 || (mb != 1 && mb != b) || m_rows != l || m_cols != kv_len {
        candle_core::bail!("softmax_in_place: mask {:?} against scores {dims:?} in slabs of {l}", mask.dims());
    }
    // A view of the same storage: `slice_set` on it writes into `raw`.
    let flat = raw.reshape((rows_total, kv_len))?;
    let mask = mask.reshape((mb, l, kv_len))?;
    let per_batch = rows_total / (b * l);
    let step = softmax_rows();
    for s in 0..rows_total / l {
        let m = mask.get(if mb == 1 { 0 } else { s / per_batch })?;
        for r0 in (0..l).step_by(step) {
            let n = step.min(l - r0);
            let at = s * l + r0;
            let p = {
                let scaled = (flat.narrow(0, at, n)? * scale)?;
                let masked = scaled.add(&m.narrow(0, r0, n)?)?;
                drop(scaled);
                candle_nn::ops::softmax_last_dim(&masked)?
            };
            flat.slice_set(&p, 0, at)?;
        }
    }
    Ok(())
}

/// The `Kernel` form's dispatch of one call of `l` rows, batch 1, over a
/// cache of `n` positions that already holds them (rows at `n - l .. n`).
/// Rows at position `from` or later go to `kern`; the rows below take
/// [`grouped_attention`] over the history they see, with their rows of
/// `mask`. A call on one side of `from` is one launch of one path; a call
/// that straddles it is split, so a row's path depends on its position and
/// on nothing else.
fn kernel_rows(
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

/// [`crate::kvq::PrefillAttn::Kernel`]: a call of `l` rows, more than the
/// kernel carries, as consecutive calls of at most `kern.max_rows()` rows,
/// each through [`kernel_rows`] over the cache prefix its last row sees.
///
/// Every row at `from` or later gets the kernel's bits for its position,
/// which are a function of that row's query and the keys and values up to it
/// and of nothing else (`llvq-cuda/src/attn_ref.rs`): the result does not
/// depend on how the prompt was chunked, and it is what a decode step at the
/// same position returns. The memory is `l` output rows; no `l × L` score
/// tensor is ever made. Rows below `from` (none under a persistent config,
/// whose threshold is 0) take the grouped path block by block, which is the
/// grouped form at other matmul shapes.
fn kernel_prefill(
    kern: &dyn crate::device::AttnKernel,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    scale: f64,
    from: usize,
) -> Result<Tensor> {
    let (l, n) = (q.dim(2)?, k.dim(2)?);
    let step = kern.max_rows().max(1);
    let pos0 = n - l;
    let mut out = Vec::with_capacity(l.div_ceil(step));
    for r0 in (0..l).step_by(step) {
        let rows = step.min(l - r0);
        let seen = pos0 + r0 + rows;
        let m = mask.map(|m| m.narrow(2, r0, rows)?.narrow(3, 0, seen)).transpose()?;
        out.push(kernel_rows(
            kern,
            &q.narrow(2, r0, rows)?,
            &k.narrow(2, 0, seen)?,
            &v.narrow(2, 0, seen)?,
            m.as_ref(),
            scale,
            from,
        )?);
    }
    Tensor::cat(&out, 2)
}

/// One transformer block, with its seven projections reachable by name.
/// A projection: dense weights, or a device kernel reading encoded ones.
///
/// The dense arm is what every published perplexity refers to and what the
/// quantizer operates on. The device arms exist only after a model is loaded
/// from a sealed artifact, by a backend's own loader. They name no backend:
/// the trait objects of [`crate::device`] own their runtime and their
/// buffers, so this enum compiles and runs on every target, and a CPU fake
/// drives the whole dispatch under `cargo test`.
///
/// The two are *not* interchangeable in both directions: `Proj::dense` panics
/// on a device projection rather than returning something plausible, because
/// every caller of it is a quantization path and quantizing an already
/// quantized model is a bug, not a use case.
/// Route P, step A (`docs/archive/debit-vague5-2026-11-02/persistent.md`
/// §3.1): the MLP half of one decode row in one cooperative launch, from
/// ln2's input to the residual after down_proj. An implementation owes the
/// eager ops' bits and is checked against [`Block::mlp_eager`] at install.
pub trait MlpSegment: Send + Sync {
    /// `x`: the residual entering ln2, one row. Returns `x + mlp(ln2(x))`.
    fn forward(&self, x: &Tensor) -> Result<Tensor>;
}

pub enum Proj {
    Dense(Linear),
    /// One projection whose weights stay encoded, read by a device kernel.
    Lattice(std::sync::Arc<dyn crate::device::LatticeProj>),
    /// One projection served as affine int4 g128 — `v_proj` in the mixed file
    /// the operator chose on 2026-09-08.
    ///
    /// It behaves like [`Proj::Dense`] in every respect that matters to the
    /// caller and like [`Proj::Lattice`] in none: the weights are STORED, in
    /// the **natural basis**, so there is no rotation to carry, no key to
    /// check against, and `prepare` hands back the activation untouched. What
    /// it shares with the other device arms is only that a kernel reads it.
    Int4(std::sync::Arc<dyn crate::device::Int4Proj>),
    /// One part of a row-concatenated group — q, k or v of one layer, or gate
    /// or up. The whole group is launched **once**, by [`group_forward`], and
    /// the result is narrowed back into the parts; there is no per-projection
    /// launch to reach from here (see [`Proj::forward_with`]'s arm).
    GroupPart {
        /// Shared by every projection of the group — `Arc::ptr_eq` on it is how
        /// [`group_forward`] recognises a group, structurally rather than by a
        /// flag that could be set on the wrong three projections.
        group: std::sync::Arc<dyn crate::device::SegGroup>,
        /// First row of this projection inside the group.
        row0: usize,
        /// This projection's own width. `group.d_out()` is the group's.
        d_out: usize,
        /// Rank inside the group, 0-based — its index in `Act::consumers()`.
        rank: usize,
    },
}

/// An activation, in the basis the projections that consume it were quantized
/// in.
///
/// **A value, not a cache entry** — that distinction is the whole correctness
/// argument of lot A4 and [`crate::rotplan`] spells it out. It is produced by
/// [`Proj::prepare`] and consumed by [`Proj::forward_with`] in the same
/// lexical scope, so the property that has to hold is *positive and local* —
/// "computed from this activation, two lines up" — rather than *negative and
/// global*, which is what every keying scheme would have needed.
///
/// ⚠️ **What that argument does and does not buy, stated exactly**, because an
/// earlier draft of this comment claimed more than the type delivers. What is
/// enforced: [`Proj::forward_with`] refuses a `Rotated` whose key is not its
/// own, so a projection can never consume an activation rotated in another
/// basis. What is *not* enforced: this value carries no trace of **which**
/// vector was rotated. Two projections of the same `(layer, act)` share a key
/// by construction, so handing one of them a `Rotated` built from a different
/// activation of that same site would be accepted, and the fused arm — which
/// reads only the rotated buffer — would silently compute on the wrong input.
///
/// Nothing reaches that today: [`Proj::group_forward`] takes a single `x` for
/// the whole group, and the two call sites in [`Block::forward_cached`] cannot
/// diverge. But `prepare` and `forward_with` are `pub`, so the guarantee is
/// held by the caller's shape, not by the type. A lifetime would not fix it —
/// it would tie the value to *a* borrow, not to *the* activation. Closing it
/// properly means carrying the source identity, and that is a design change,
/// not a comment.
///
/// The fields are private on purpose: the only way to obtain one is to rotate
/// an activation, and the only thing that can be done with it is to hand it to
/// a projection that then checks it belongs to it.
pub struct Rotated {
    key: Option<RotKey>,
    /// Read by the device arms of [`Proj::forward_with`]. The dense arm
    /// deliberately multiplies the caller's own `x` instead, so that it stays
    /// the call `bin/oracle` certifies.
    ///
    /// It used to carry an `allow(dead_code)` off CUDA, because the only
    /// readers were behind a `cfg` no development machine here compiled. The
    /// device port removed that `cfg`, so the field is live on every target.
    t: Tensor,
}

impl Rotated {
    /// Which rotation produced it — `None` in the natural basis, which is
    /// every dense projection.
    pub fn key(&self) -> Option<RotKey> {
        self.key
    }
}

/// Most rows a single fused call may loop over.
///
/// Not an optimization gap that can be papered over: the fused kernel is a
/// matrix–*vector* product, so `l` tokens cost `l` launches. Generation always
/// begins by running the whole prompt through, so `rows > 1` is a live path
/// and refusing it outright was wrong (the very first run found that). The cap
/// survives because a 2048-token scoring window would issue half a million
/// launches and look like a hang; `bin/ppl` keeps the dense path.
pub const MAX_ROWS: usize = 256;
// ⚠️ Read it as a LAUNCH budget, not a row budget, and read it as the budget
// of the paths that spend ONE launch a row: the segmented kernel, and every
// layout that carries no rows kernel. A decode step never approaches it — a
// decode step is one row.
//
// The batched prefill path has its own budget, [`MAX_PREFILL_ROWS`], because
// it bounds a different quantity. Until 2026-09-11 it did not: the cap read
// `MAX_ROWS · batch`, so raising one raised the other, and a served layout's
// segmented path would have moved for a reason that has nothing to do with it.

/// Most rows the batched prefill path may take in one call.
///
/// ## Why it is not [`MAX_ROWS`] times the batch
///
/// It was, and the arithmetic was right: a chunk of `batch` rows spends one
/// launch, so the same launch budget admits `batch` times the rows. What was
/// wrong is that it tied two paths together. `MAX_ROWS` also bounds the
/// segmented kernel, which is served under `Planes14` and takes one launch a
/// row; moving the prefill bound would have moved that one too, four-fold,
/// with no measurement behind it.
///
/// ## Where 4,096 comes from
///
/// From the census, not from a round number. The longest 5-shot MMLU prompt
/// under the Qwen3 tokenizer is **3,096 tokens** — `high_school_european_history`,
/// whose subject mean is 2,796 (*measured* 2026-09-11 from the cached
/// `cais/mmlu` parquet, `docs/mesures/f1e0-2026-09-10.txt`). The previous
/// bound of 1,024 refused **239 of the 2,280** questions every published bar
/// is measured on — five subjects entirely, `professional_law` and four
/// histories. 4,096 admits the whole split and leaves a third of itself spare.
///
/// ## What it costs to be wrong high
///
/// The original cap existed because "a 2048-token scoring window would issue
/// half a million launches and look like a hang". That reason survives, and
/// now it has a number instead of a guess: at the *measured* 5.507 ms a
/// prompt token, 4,096 rows is 23 s of prefill. Slow, bounded, and not a
/// hang. `bin/ppl` still keeps the dense path for its 4,096-token window.
pub const MAX_PREFILL_ROWS: usize = 4096;

/// Which of the two budgets a chunk of `batch` rows answers to.
///
/// Two budgets because two shapes of work, and a function rather than an
/// expression inline because this is the line that decides whether a census
/// question is scored or refused — the kind of line that has to be reachable
/// from a test on a machine with no card.
pub fn row_cap(batch: usize) -> usize {
    match batch > 1 {
        // One launch a chunk: bounded by the longest prompt the census holds.
        true => MAX_PREFILL_ROWS,
        // One launch a row: the budget that has always been a launch budget.
        false => MAX_ROWS,
    }
}

/// Most rows one [`group_forward`] call on `projs` takes, by the branch that
/// call would take: no bound where no projection is rotated, [`MAX_ROWS`] for
/// a segmented group, and [`row_cap`] of the group's batch otherwise.
///
/// A caller that cuts a prompt into chunks reads it so that no chunk is
/// refused. The served Metal path batches nothing, so it is 256 there.
pub fn group_row_cap(projs: &[&Proj]) -> usize {
    if projs.iter().all(|p| p.rot_key().is_none()) {
        return usize::MAX;
    }
    if projs.iter().any(|p| matches!(p, Proj::GroupPart { .. })) {
        return MAX_ROWS;
    }
    row_cap(prefill_rows_of(projs))
}

impl Proj {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // A group of one: `Off` and `On` coincide, so the mode is not a
        // parameter here and no caller has to learn about it.
        Ok(group_forward(&[self], x, RotShare::Off)?
            .pop()
            .expect("a group of one yields one result"))
    }

    /// The rotation this projection's weights were quantized under, or `None`
    /// in the natural basis — which is every dense projection.
    pub fn rot_key(&self) -> Option<RotKey> {
        match self {
            Proj::Dense(_) => None,
            Proj::Lattice(d) => d.rotation(),
            // Stored in the natural basis, so there is nothing to carry and
            // nothing for `check_key` to compare — exactly the dense answer,
            // and for exactly the dense reason.
            Proj::Int4(_) => None,
            Proj::GroupPart { group, .. } => group.rotation(),
        }
    }

    /// The name the artifact stores, for error messages. Dense projections
    /// come out of a `VarBuilder` and carry none.
    pub fn site_name(&self) -> &str {
        match self {
            Proj::Dense(_) => "(dense projection)",
            Proj::Lattice(d) => d.name(),
            Proj::Int4(d) => d.name(),
            Proj::GroupPart { group, rank, .. } => group.part_name(*rank),
        }
    }

    /// Output width.
    pub fn d_out(&self) -> Result<usize> {
        match self {
            Proj::Dense(l) => l.weight().dim(0),
            Proj::Lattice(d) => Ok(d.d_out()),
            Proj::Int4(d) => Ok(d.d_out()),
            // The **part**, not the group: this is the width of the tensor the
            // caller gets back, which is what every consumer reshapes on.
            Proj::GroupPart { d_out, .. } => Ok(*d_out),
        }
    }

    /// Rotate `x` into the basis this projection was quantized in.
    ///
    /// Dense: a strict no-op — the tensor itself, no launch, no allocation
    /// beyond candle's refcount. That arm is what `bin/oracle` certifies at
    /// `max |Δhidden| = 0` and it must stay untouched.
    ///
    /// Fused: one `rot_apply`, one f32 `[1, d_in]` result. The output is a
    /// tensor of full standing rather than a scratch buffer, which is what
    /// lets the caller keep it alive across the group's uses without any
    /// aliasing argument.
    pub fn prepare(&self, x: &Tensor) -> Result<Rotated> {
        match self {
            Proj::Dense(_) => Ok(Rotated { key: None, t: x.clone() }),
            Proj::Lattice(d) => Ok(Rotated {
                key: d.rotation(),
                t: d.prepare(x)?,
            }),
            // No `rot_apply`: the weights are quantized in the basis the
            // activation already arrives in. Byte for byte the dense arm's
            // answer, which is what makes `check_key` accept both in one group
            // and refuse a rotated one beside them.
            Proj::Int4(_) => Ok(Rotated { key: None, t: x.clone() }),
            // One `rot_apply` for the whole group: the parts share one `d_in`
            // and one rotation key by construction, checked at load time.
            Proj::GroupPart { group, .. } => Ok(Rotated {
                key: group.rotation(),
                t: group.prepare(x)?,
            }),
        }
    }

    /// [`Self::prepare`] for a whole chunk of rows, one launch.
    ///
    /// `xs` is `[len, d_in]` and **contiguous** — `group_forward` narrows it out
    /// of the one allocation `row_block` reshapes, so the rows are already
    /// `d_in` apart and nothing is copied to make them so.
    ///
    /// ## A chunk of one is [`Self::prepare`], verbatim
    ///
    /// Same discipline as [`Self::forward_rows`], and for a sharper reason
    /// here: `rotate` refuses `rows != 1` and `rotate_rows` would launch a
    /// grid of one block, which is the same arithmetic through a second entry
    /// point. Taking the one-row path is what keeps every decode step on the
    /// code `bin/oracle` certifies.
    ///
    /// ## What batches and what does not
    ///
    /// `Dense` and `FusedInt4` consume the activation in the basis it arrives
    /// in, so their "rotation" is the chunk itself and costs nothing at any
    /// length. `Fused` takes the rows kernel. `FusedSeg` does not batch — its
    /// group is one launch already and `batches_rows` is false for it, so
    /// `group_forward` never asks it for a chunk of more than one; asking
    /// anyway is refused by `rotate_group`, by name, rather than silently
    /// rotating the first row and using it for four.
    pub fn prepare_rows(&self, xs: &Tensor) -> Result<Rotated> {
        let rows = xs.dim(0)?;
        if rows == 1 {
            return self.prepare(xs);
        }
        match self {
            Proj::Dense(_) => Ok(Rotated { key: None, t: xs.clone() }),
            Proj::Lattice(d) => Ok(Rotated {
                key: d.rotation(),
                t: d.prepare_rows(xs, rows)?,
            }),
            Proj::Int4(_) => Ok(Rotated { key: None, t: xs.clone() }),
            Proj::GroupPart { group, .. } => Ok(Rotated {
                key: group.rotation(),
                t: group.prepare(xs)?,
            }),
        }
    }

    /// Whether a chunk of this projection is one launch rather than one a row.
    ///
    /// Read only to choose the chunk size — the answer changes the SHAPE of
    /// the work and never its result, because a chunk of a projection that
    /// says `false` fans out through `forward_with` row by row.
    pub fn batches_rows(&self) -> bool {
        match self {
            Proj::Lattice(d) => d.rows_per_launch() > 1,
            _ => false,
        }
    }

    /// `y = W X` for a whole chunk of rows, one launch where the layout has a
    /// kernel that takes several.
    ///
    /// ## What it is, and what it is not
    ///
    /// It is a fallback that sometimes goes fast, never a second arithmetic.
    /// A chunk of ONE row is [`Self::forward_with`] verbatim — not a batch of
    /// one — and an arm whose layout carries no rows kernel fans out through
    /// the same function, row by row. So every result this returns is a result
    /// `forward_with` would have returned, and the fast path is proved
    /// bit-identical to it by `tetra48_matches_rust.rs`.
    ///
    /// ## Why a chunk of one is not a batch of one
    ///
    /// `Tensor::cat` of a one-element slice returns that element **verbatim**,
    /// layout included. The rows here are `row_views`' narrows — views at
    /// offset `r · d_in` into one allocation — so a "batch" of one would hand
    /// the kernel a view at a nonzero offset. `FusedRowsOp` refuses that rather
    /// than reading the wrong slice, which is the right failure and still a
    /// failure. Taking the one-row path is what makes it not arise.
    ///
    /// ## What it buys, stated honestly
    ///
    /// BYTES, not launches. `Tensor::cat` copies each row into a fresh
    /// allocation, so a chunk of four costs four copies plus one matvec where
    /// four rows cost four matvecs — the launch count barely moves, and on a
    /// q+k+v chunk it goes UP. What falls is the weight stream: `q_proj`'s
    /// words are read once instead of four times, and the stream is what
    /// costs. On the served 4B that is 776 GB a question against 194.
    pub fn forward_rows(&self, r: &Rotated, xs: &Tensor, len: usize) -> Result<Vec<Tensor>> {
        if len == 0 {
            return Ok(Vec::new());
        }
        if xs.dim(0)? != len {
            candle_core::bail!("{} rows of activation for a chunk of {len}", xs.dim(0)?);
        }
        // One row is the one-row path, byte for byte.
        if len == 1 {
            return Ok(vec![self.forward_with(r, xs)?]);
        }
        match self {
            Proj::Lattice(d) if d.rows_per_launch() > 1 => {
                check_key(self.site_name(), self.rot_key(), r.key)
                    .map_err(candle_core::Error::msg)?;
                // No `Tensor::cat` here any more, and that is the point of the
                // whole change: `prepare_rows` rotated the chunk in one launch
                // and its output is already `[len, d_in]` contiguous, which is
                // the shape this kernel wants. Stacking it cost one device copy
                // a row — 864 a chunk on the served 4B, the largest single term
                // in the prefill's operation count.
                //
                // And one `check_key` covers the chunk because there is one
                // rotation for the chunk. The loop that checked every row's key
                // against the first went with the vector of rotations it was
                // checking: a chunk cannot mix two sites' rotations when it
                // holds one.
                let y = d.matvec_rows(&r.t, len)?;
                (0..len).map(|i| y.narrow(0, i, 1)).collect()
            }
            // An int4 rows kernel: `xs`, the caller's activation, as
            // `forward_with` reads it for int4.
            Proj::Int4(d) if d.rows_per_launch() > 1 => {
                let y = int4_chunked(d.as_ref(), xs)?;
                (0..len).map(|i| y.narrow(0, i, 1)).collect()
            }
            // Dense, int4, a segmented part, or a layout with no rows kernel:
            // the same function, one row at a time. `tv_q4_h` takes no row
            // count and no offset, so int4 is here by construction and not by
            // omission.
            //
            // `r.t` is narrowed along with `xs`. In practice only `Dense` and
            // `FusedInt4` reach here with `len > 1` — a `Fused` whose layout
            // has no rows kernel is never given a chunk, because `batch` is 1
            // unless some projection of the group batches — and for both of
            // those `forward_with` reads `x` and not `r.t`. The narrow is
            // still done, because "in practice" is not a contract and a view
            // costs nothing.
            _ => (0..len)
                .map(|i| {
                    let one = Rotated { key: r.key, t: r.t.narrow(0, i, 1)? };
                    self.forward_with(&one, &xs.narrow(0, i, 1)?)
                })
                .collect(),
        }
    }

    /// `y = W x`, given `x` already rotated by [`Self::prepare`].
    ///
    /// `x` is passed beside the rotated form and is **not** redundant: it
    /// carries the caller's shape, so the result comes out with the rank the
    /// caller handed in, and on the dense arm it is the tensor the `Linear`
    /// multiplies — literally today's call.
    ///
    /// Refuses a rotation that is not this projection's, naming it. See
    /// [`check_key`].
    pub fn forward_with(&self, r: &Rotated, x: &Tensor) -> Result<Tensor> {
        check_key(self.site_name(), self.rot_key(), r.key).map_err(candle_core::Error::msg)?;
        match self {
            Proj::Dense(l) => l.forward(x),
            Proj::Lattice(d) => d.matvec(&r.t, x.dims()),
            // `x`, not `r.t` — the dense arm's choice, and the same one: `r.t`
            // IS `x` here, and reading it from the argument the caller handed
            // in is what keeps that true if `prepare` ever stops being the
            // identity for this arm.
            Proj::Int4(d) => d.matvec(x, x.dims()),
            // Deliberately unreachable, and it must stay that way: a launch per
            // projection here would compute the group's whole width and then
            // throw most of it away — silently, and at three times the cost.
            Proj::GroupPart { group, .. } => candle_core::bail!(
                "{} belongs to a fused group of {} rows: it only launches through \
                 model::group_forward, which emits ONE matvec for the whole group.",
                self.site_name(),
                group.d_out()
            ),
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
            Proj::Lattice(d) => panic!(
                "{} is a device projection: the quantizer only works on a dense model",
                d.name()
            ),
            Proj::GroupPart { group, rank, .. } => panic!(
                "{} is a device projection: the quantizer only works on a dense model",
                group.part_name(*rank)
            ),
            Proj::Int4(d) => panic!(
                "{} is served as int4: the quantizer only works on a dense model",
                d.name()
            ),
        }
    }

    pub fn dense_mut(&mut self) -> &mut Linear {
        match self {
            Proj::Dense(l) => l,
            Proj::Lattice(d) => panic!(
                "{} is a device projection: the quantizer only works on a dense model",
                d.name()
            ),
            Proj::GroupPart { group, rank, .. } => panic!(
                "{} is a device projection: the quantizer only works on a dense model",
                group.part_name(*rank)
            ),
            Proj::Int4(d) => panic!(
                "{} is served as int4: the quantizer only works on a dense model",
                d.name()
            ),
        }
    }
}

/// The single-row views of an activation, in row order.
///
/// One contiguity pass for the whole group, then `narrow` along dim 0 — which
/// yields a view at an *offset* into one allocation, and that is exactly what
/// `rot_apply`'s `x_off` argument exists for. Re-materializing each row would
/// be a copy per row per projection.
///
/// Extracted from [`group_forward`] with [`regroup`] because that function's
/// rotated branch is reachable only with a fused projection, i.e. on no
/// machine this test suite runs on. The shape bookkeeping is the part of it
/// that can be silently wrong — a transposed row order returns finite,
/// plausible, wrong tokens — so it lives in two functions that a CPU tensor
/// can exercise.
/// Rows one prefill launch carries, 1 where no projection batches.
///
/// From the PROJECTION that will take the chunk, not from a constant. Under
/// `LLVQ_PREFILL` a unit is compiled at eight or sixteen rows and
/// `tile::PREFILL_ROWS` still reads four; handing a chunk of eight to a kernel
/// compiled at four is the silent corruption this reader exists to prevent.
/// `matvec_rows` would refuse it, but only after the chunking had decided the
/// shape of the work.
///
/// One number, asked once. This used to read a `bool` and a `usize` off a
/// runtime, and those two could disagree. A single `rows_per_launch` cannot
/// disagree with itself.
///
/// Every projection of a group shares one runtime by construction, so the
/// first that batches answers for the group.
fn prefill_rows_of(projs: &[&Proj]) -> usize {
    for p in projs {
        if let Proj::Lattice(d) = p {
            let rows = d.rows_per_launch();
            if rows > 1 {
                return rows;
            }
        }
    }
    1
}

pub fn row_views(x: &Tensor) -> Result<Vec<Tensor>> {
    let flat = row_block(x)?;
    (0..flat.dim(0)?).map(|r| flat.narrow(0, r, 1)).collect()
}

/// The same rows as ONE contiguous `[rows, d_in]` tensor.
///
/// What [`row_views`] narrows out of, handed over whole. A caller that wants a
/// chunk narrows it once and gets a view; a caller that wanted a vector of
/// single rows and then stacked four of them back together was paying a device
/// copy a row to rebuild a shape it already had.
pub fn row_block(x: &Tensor) -> Result<Tensor> {
    let dims = x.dims();
    let d_in = *dims.last().expect("rank >= 1");
    let rows: usize = dims[..dims.len() - 1].iter().product();
    x.contiguous()?.reshape((rows, d_in))
}

/// `y = W x` for the rows of `flat`, `[rows, d_in]`, through an int4 rows
/// kernel in launches of [`crate::device::Int4Proj::rows_per_launch`] rows,
/// as `[rows, d_out]`. Each row is the matvec's by the port's contract, so
/// only the launch count moves. The one place the model reaches that kernel.
fn int4_chunked(d: &dyn crate::device::Int4Proj, flat: &Tensor) -> Result<Tensor> {
    let rows = flat.dim(0)?;
    let k = d.rows_per_launch();
    let parts = (0..rows)
        .step_by(k)
        .map(|row0| {
            let len = k.min(rows - row0);
            d.matvec_rows(&flat.narrow(0, row0, len)?, len)
        })
        .collect::<Result<Vec<_>>>()?;
    Tensor::cat(&parts, 0)
}

/// Reassemble one projection's per-row results into the caller's shape.
///
/// `dims` is the shape the caller handed to [`group_forward`]; the result is
/// that shape with `d_out` as its last axis. Inverse of [`row_views`] up to
/// the change of width.
pub fn regroup(rows: &[Tensor], dims: &[usize], d_out: usize) -> Result<Tensor> {
    let mut d = dims.to_vec();
    *d.last_mut().expect("rank >= 1") = d_out;
    Tensor::cat(rows, 0)?.reshape(d)
}

/// Run projections that share one activation.
///
/// This is the call site lot A4 turns on. `Block::forward_cached` hands q/k/v
/// one group and gate/up another; o_proj and down_proj are groups of one. Under
/// [`RotShare::On`] the group's activation is rotated **once**, and every
/// projection is handed that value; under [`RotShare::Off`] each rotates its
/// own, which is launch for launch what shipped before this lot.
///
/// ## Two branches, and the first one is the one that must not move
///
/// When no projection of the group carries a rotation — every dense model, so
/// every path `bin/oracle`, `bin/ppl`, `bin/mmlu` and the quantizer take —
/// there is nothing to hoist, and the group degenerates to `prepare` (a clone)
/// then `forward_with` (the `Linear`'s own `forward`) on the **whole** tensor,
/// in declaration order. No reshape, no row loop, no mode dependence. The
/// forward pass certified at `max |Δhidden| = 0` is unchanged, and
/// `the_restructured_block_is_the_same_calls_in_the_same_order` requires token
/// equality rather than a tolerance to say so.
///
/// The rotated branch does split rows, because the fused kernel is a matvec:
/// the row loop that used to live inside `FusedRuntime::forward` moves up one
/// level, so that a prefill row is rotated once for its whole group instead of
/// once per projection. Under `Off` the resulting order is still
/// rot/matvec/rot/matvec projection by projection — the old order exactly.
pub fn group_forward(projs: &[&Proj], x: &Tensor, share: RotShare) -> Result<Vec<Tensor>> {
    if projs.is_empty() {
        return Ok(Vec::new());
    }
    if projs.iter().all(|p| p.rot_key().is_none()) {
        return projs
            .iter()
            .map(|p| match p {
                // `tv_q4_h` is a matvec: one vector, no row count. A group with
                // no rotation that holds one — a lone `o_proj` or `down_proj`
                // served int4 — fans out row by row, as the rotated branch
                // does for an int4 `v_proj`. `Proj::Dense` still takes the
                // whole tensor, so the dense path does not move.
                Proj::Int4(_) => int4_rows(p, x),
                _ => {
                    let r = p.prepare(x)?;
                    p.forward_with(&r, x)
                }
            })
            .collect();
    }

    // After the dense branch, which must not move, and before the row loop:
    // a fused group replaces the loop's per-projection launches with one.
    if let Some(plan) = SegPlan::of(projs)? {
        return plan.run(x);
    }

    let dims = x.dims();
    let rows: usize = dims[..dims.len() - 1].iter().product();
    // The chunk: `PREFILL_ROWS` where a projection of this group can spend it,
    // and ONE everywhere else — which is the loop that shipped, unchanged, for
    // every layout that carries no rows kernel and for every group of one row.
    //
    // It changes the shape of the work and never its result: a projection that
    // cannot batch fans out through `forward_with`, row by row, inside the same
    // chunk. So a mixed q+k+v group of the served file issues one launch for q,
    // one for k, and four for the `v_proj` that is int4.
    let batch = match rows > 1 {
        true => prefill_rows_of(projs),
        false => 1,
    };
    let cap = row_cap(batch);
    if rows > cap {
        candle_core::bail!(
            "{}: {rows} vectors at once, more than {cap} (at {batch} rows a launch, \
             {} launches a projection). Past that the loop issues more launches than \
             it does arithmetic; `bin/ppl`'s 4096-token window keeps the dense path.",
            projs[0].site_name(),
            rows.div_ceil(batch)
        );
    }
    // The rows as ONE contiguous `[rows, d_in]`, not as a vector of views: a
    // chunk is then `narrow(0, row0, len)`, which is a view into it and costs
    // nothing, and the rotation kernel can read `len` rows from one pointer.
    let flat = row_block(x)?;
    let per_site = drive_rows_batched(
        share,
        projs,
        rows,
        batch,
        // A projection with no rotation key takes no share of the group's:
        // `Proj::Dense` and `Proj::Int4` both consume the activation in
        // the basis it arrives in.
        |p: &&Proj| p.rot_key().is_some(),
        |p: &&Proj, row0: usize, len: usize| p.prepare_rows(&flat.narrow(0, row0, len)?),
        |p: &&Proj, r: &Rotated, row0: usize, len: usize| {
            p.forward_rows(r, &flat.narrow(0, row0, len)?, len)
        },
    )?;

    per_site
        .into_iter()
        .zip(projs)
        .map(|(rows_of_site, p)| regroup(&rows_of_site, dims, p.d_out()?))
        .collect()
}

/// `y = W x` for every row of `x` through a single-vector int4 kernel,
/// reassembled into the caller's shape. One row is `forward_with` verbatim,
/// so a decode step takes the call it always took.
fn int4_rows(p: &Proj, x: &Tensor) -> Result<Tensor> {
    let dims = x.dims();
    let rows: usize = dims[..dims.len() - 1].iter().product();
    if rows == 1 {
        let r = p.prepare(x)?;
        return p.forward_with(&r, x);
    }
    if let Proj::Int4(d) = p {
        if d.rows_per_launch() > 1 {
            let mut out = dims.to_vec();
            *out.last_mut().expect("rank >= 1") = d.d_out();
            return int4_chunked(d.as_ref(), &row_block(x)?)?.reshape(out);
        }
    }
    let ys = row_views(x)?
        .iter()
        .map(|xi| {
            let r = p.prepare(xi)?;
            p.forward_with(&r, xi)
        })
        .collect::<Result<Vec<_>>>()?;
    regroup(&ys, dims, p.d_out()?)
}

/// One segmented launch, and the map back to the projections it stands for.
///
/// Built by [`Self::of`] from the projections `group_forward` was handed, so
/// the model states the row order **independently** of the loader that assigned
/// it. Two places have to agree and neither may assume: a group read in the
/// order k,q,v would otherwise return finite, plausible, wrong numbers with no
/// assertion anywhere.
struct SegPlan<'a> {
    /// Borrowed as the `Arc` rather than as its target: the group's identity
    /// test below is then `Arc::ptr_eq`, the only comparison that cannot be
    /// fooled, and the field is initialised straight from the enum's binding
    /// with no coercion in between.
    ///
    /// ⚠️ This is a FAT pointer, `Arc<dyn SegGroup>`. `Arc::ptr_eq` compares
    /// the data address alone and ignores the vtable, which is what makes the
    /// test correct here; that has held since Rust 1.76 and
    /// `rust-toolchain.toml` pins 1.95.0. That file's own header warns it pins
    /// FORWARD, not backward. Never `Arc::ptr_eq` a device trait object to
    /// "strengthen" this check: two coercions of one allocation are equal, but
    /// a vtable address is not guaranteed unique.
    ///
    /// The runtime is not a field any more. The group owns it.
    group: &'a std::sync::Arc<dyn crate::device::SegGroup>,
    /// `(rank, row0, d_out)` per projection, in the order the caller handed
    /// them — the shape [`crate::fused::check_seg_spans`] validates.
    spans: Vec<(usize, usize, usize)>,
}

impl<'a> SegPlan<'a> {
    /// `Some` when every projection handed in is a part of **one** group and,
    /// together and in this order, they are the whole group.
    ///
    /// `None` when none of them is segmented — every dense and every unfused
    /// build. `Err` when they are mixed, partial, or out of order: a wiring
    /// bug, never a fallback.
    fn of(projs: &[&'a Proj]) -> Result<Option<Self>> {
        if !projs.iter().any(|p| matches!(p, Proj::GroupPart { .. })) {
            return Ok(None);
        }
        let mut plan: Option<SegPlan<'a>> = None;
        // `&p` rather than `p`: the item is a `&&Proj`, and destructuring the
        // outer reference is what ties the bindings to the projections'
        // lifetime rather than to this loop's.
        for &p in projs {
            let Proj::GroupPart { group, row0, d_out, rank } = p else {
                candle_core::bail!(
                    "{} belongs to no fused group while its group neighbours do. A \
                     segmented launch cannot cover half a site",
                    p.site_name()
                );
            };
            match &mut plan {
                None => {
                    plan = Some(SegPlan {
                        group,
                        spans: vec![(*rank, *row0, *d_out)],
                    })
                }
                Some(acc) => {
                    // Pointer identity, not a key: two groups of two layers
                    // share a `d_in` and a shape, and comparing anything else
                    // would accept a q/k/v triple assembled across layers.
                    if !std::sync::Arc::ptr_eq(acc.group, group) {
                        candle_core::bail!(
                            "{} belongs to a different fused group than {}, two groups in \
                             a single call",
                            p.site_name(),
                            acc.group.name()
                        );
                    }
                    acc.spans.push((*rank, *row0, *d_out));
                }
            }
        }
        let plan = plan.expect("the `any` above found at least one segmented projection");
        crate::fused::check_seg_spans(plan.group.name(), &plan.spans, plan.group.d_out())
            .map_err(candle_core::Error::msg)?;
        Ok(Some(plan))
    }

    /// One rotation and one matvec per row, then one view per projection.
    fn run(&self, x: &Tensor) -> Result<Vec<Tensor>> {
        let dims = x.dims();
        let rows: usize = dims[..dims.len() - 1].iter().product();
        if rows > MAX_ROWS {
            candle_core::bail!(
                "{}: {rows} vectors at once, more than {MAX_ROWS}. The segmented kernel is \
                 still a matvec, it loops, so the cost is linear.",
                self.group.name()
            );
        }
        let slices = row_views(x)?;

        // One rotation and one matvec per row, whatever `RotShare` says: a
        // fused group is ONE site, so there is no second arm left to keep.
        // That is why `fused::check_fuse` refuses `LLVQ_FUSE=1` beside
        // `LLVQ_ROT_SHARE=0` — accepting it would let an `LLVQ_FUSE` A/B move
        // the hoist as well. `drive_rows` is deliberately not called here for
        // the same reason: its two arms would coincide, and calling it would
        // suggest a control that is gone.
        let mut per_row = Vec::with_capacity(rows);
        for slice in slices.iter().take(rows) {
            let r = self.group.prepare(slice)?;
            per_row.push(self.group.matvec(&r, slice.dims())?);
        }
        // `[rows, group.d_out]`, then one view per projection on the last axis.
        let stacked = Tensor::cat(&per_row, 0)?;
        self.spans
            .iter()
            .map(|&(_, row0, d)| {
                let mut shape = dims.to_vec();
                *shape.last_mut().expect("rank >= 1") = d;
                // `.contiguous()` unconditionally, and it is not defensive:
                //  * at `rows == 1` — the decode path this lot measures — the
                //    narrow is already contiguous (`Shape::is_contiguous` skips
                //    axes of extent ≤ 1), so this is a clone and costs nothing;
                //  * at `rows > 1` the narrow is NOT contiguous (axis 0 has
                //    stride `group.d_out` against an accumulator of `d`), so
                //    the caller's `reshape` would materialise anyway. Doing it
                //    here makes *where* the copy happens explicit, and it is
                //    the same volume `regroup`'s `Tensor::cat` already copies.
                // It also closes an aliasing retention: `KvCache::append`
                // stores `v.clone()`, and a view would keep the whole
                // 6144-wide buffer alive behind a 1024-wide tensor.
                stacked.narrow(1, row0, d)?.contiguous()?.reshape(shape)
            })
            .collect()
    }
}

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
    /// Route P, step A: the MLP half of a one-row decode step in one launch,
    /// installed by `persist_cuda::install` behind a measurement mode; `None`
    /// everywhere else, and then `finish` runs the eager ops as always.
    pers_mlp: Option<std::sync::Arc<dyn MlpSegment>>,
    /// The first position the `Kernel` form sends to its kernel:
    /// [`crate::kvq::KERNEL_FROM`]. A field only so the host tests can reach
    /// the kernel in a window of a few dozen positions; nothing outside
    /// `cfg(test)` writes it.
    kernel_from: usize,
    /// How the `Kernel` form attends a call of more rows than the kernel
    /// carries ([`crate::kvq::PrefillAttn`]): set by
    /// [`Qwen3::set_prefill_attention`], `Grouped` everywhere else.
    prefill_attn: crate::kvq::PrefillAttn,
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
    fn row_cap(&self) -> usize {
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
    fn new_with(
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
            pers_mlp: None,
            kernel_from: crate::kvq::KERNEL_FROM,
            prefill_attn: crate::kvq::PrefillAttn::Grouped,
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
        self.forward_cached(x, rotary, Some(mask), RopeAt::Offset(0), &mut fresh, idx, cap)
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
        rope: RopeAt,
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
        let (q, k) = match rope {
            RopeAt::Offset(o) => rotary.apply(&q, &k, o)?,
            RopeAt::Tables { cos, sin } => Rotary::apply_tables(&q, &k, cos, sin)?,
        };
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
        let (k, v) = match cache.store() {
            crate::kvq::KvStore::Cat | crate::kvq::KvStore::Grow(_) => {
                let (k, v) = cache.append(&k, &v)?;
                (
                    candle_transformers::utils::repeat_kv(k, groups)?.contiguous()?,
                    candle_transformers::utils::repeat_kv(v, groups)?.contiguous()?,
                )
            }
            // Prealloc stores the EXPANDED step: the ×`groups` copy happens
            // once, on the `l` new positions, instead of on the whole history
            // at every step — which is what `repeat_kv` costs on the `Cat`
            // path. What comes back is the full contiguous
            // `[b, n_heads, W, hd]` pair; the wide causal mask −infs
            // everything past the valid prefix, so softmax weights there are
            // exactly zero and the zero-filled tail contributes exactly
            // nothing.
            crate::kvq::KvStore::Prealloc(_) => {
                let ke = candle_transformers::utils::repeat_kv(k, groups)?.contiguous()?;
                let ve = candle_transformers::utils::repeat_kv(v, groups)?.contiguous()?;
                cache.append(&ke, &ve)?
            }
        };

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
        // Route P, step A: one decode row, nothing to capture.
        if let (Some(seg), 1, 1, false) = (&self.pers_mlp, b, l, cap.wants_activations()) {
            return seg.forward(&x);
        }
        self.mlp_residual(&x, idx, cap)
    }

    /// `x + down(act(gate(ln2 x)) * up(ln2 x))`, the eager ops: the half of
    /// `finish` route P's segment replaces, and its load-time reference.
    fn mlp_residual(&self, x: &Tensor, idx: usize, cap: &mut dyn Capture) -> Result<Tensor> {
        let h = self.ln2.forward(x)?;
        cap.on_activation(idx, Act::Mlp, &h)?;
        // The second shared activation: gate and up both read `h`.
        let gu = group_forward(&[&self.gate_proj, &self.up_proj], &h, self.rot_share)?;
        let gated = gu[0].apply(&self.act)?.mul(&gu[1])?;
        cap.on_activation(idx, Act::MlpOut, &gated)?;
        x + self.down_proj.forward(&gated)?
    }

    /// The eager MLP half on one row, no capture: route P's load check.
    pub fn mlp_eager(&self, x: &Tensor) -> Result<Tensor> {
        self.mlp_residual(x, 0, &mut NoCapture)
    }

    /// What route P's segment reads from this block: ln2's weight and
    /// epsilon, the activation, and the three MLP projections.
    pub fn mlp_parts(&self) -> (&Tensor, f64, Activation, &Proj, &Proj, &Proj) {
        (&self.ln2_w, self.ln2_eps, self.act, &self.gate_proj, &self.up_proj, &self.down_proj)
    }

    /// What 4E reads from this block's attention half: ln1, q-norm and
    /// k-norm weights, and the q, k, v, o projections.
    pub fn attn_parts(&self) -> (&Tensor, &Tensor, &Tensor, &Proj, &Proj, &Proj, &Proj) {
        (&self.ln1_w, &self.q_norm_w, &self.k_norm_w, &self.q_proj, &self.k_proj, &self.v_proj, &self.o_proj)
    }

    /// Install or remove route P's MLP segment on this block.
    pub fn set_pers_mlp(&mut self, seg: Option<std::sync::Arc<dyn MlpSegment>>) {
        self.pers_mlp = seg;
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

/// Qwen3, loaded for scoring.
pub struct Qwen3 {
    cfg: Config,
    /// Cache mode, named by the caller at construction. `model.rs` reads no
    /// environment variable; the binaries resolve `LLVQ_KV` and pass it here.
    kv: KvMode,
    /// Cache growth store — [`crate::kvq::KvStore::Cat`] unless a binary
    /// resolves `LLVQ_KV_PREALLOC` and says otherwise via
    /// [`Self::set_kv_store`]. The A/B guard is the printed arm line, which
    /// must name the store: an unwired arm prints `cat` and is visible.
    kv_store: crate::kvq::KvStore,
    /// How every block's attention reads the cached heads, kept here as well
    /// so the stepped path can size its write index. Set through
    /// [`Self::set_attn_heads`] only.
    attn_heads: crate::kvq::AttnHeads,
    /// How the `Kernel` form attends a prefill chunk, as every block holds
    /// it. Set through [`Self::set_prefill_attention`] only.
    prefill_attn: crate::kvq::PrefillAttn,
    /// Measurement only, `false` everywhere else: build and add the all-zero
    /// mask of a `Cat` decode step, as every decode step did before
    /// [`Self::step_mask`] learned to skip it. `bin/fusedrun`'s
    /// `LLVQ_TRIM_AB` sets it on one arm to time what the skip saves.
    decode_mask: bool,
    /// Measurement only, `None` everywhere else: a 1-element f16 tensor
    /// that [`Self::hidden_cached`] runs one candle affine on after each
    /// layer, the `candle-tiny` arm of `bin/fusedrun`'s `LLVQ_BOUNDARY_AB`.
    /// The result is dropped, so no output moves.
    candle_probe: Option<Tensor>,
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
            prefill_attn: crate::kvq::PrefillAttn::Grouped,
            decode_mask: false,
            candle_probe: None,
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
        self.causal_mask_width(b, l, offset, offset + l)
    }

    /// The same mask at an imposed key width `w >= offset + l` — the fixed
    /// window of a preallocated cache. Columns past `offset + i` are −inf,
    /// which covers BOTH the future and the not-yet-written tail: no second
    /// mechanism, the causal rule is the validity rule.
    fn causal_mask_width(&self, b: usize, l: usize, offset: usize, w: usize) -> Result<Tensor> {
        build_causal_mask(b, l, offset, w, self.dtype, &self.device)
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
            KvStore::Cat | KvStore::Grow(_) if l == 1 && !self.decode_mask => Ok(None),
            KvStore::Cat | KvStore::Grow(_) => self.causal_mask_offset(b, l, offset).map(Some),
            KvStore::Prealloc(w) => self.causal_mask_width(b, l, offset, w).map(Some),
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

    /// Name the cache growth store — resolved from `LLVQ_KV_PREALLOC` by the
    /// binaries, after construction (the loaders' signatures predate the
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

    /// Measurement only: make every `Cat` decode step build and add its
    /// all-zero mask again, the behaviour before 2026-10-30. The output does
    /// not move (adding zero changes no score), only the time does, which is
    /// what `LLVQ_TRIM_AB` measures. No served path calls it.
    pub fn set_decode_mask(&mut self, on: bool) {
        self.decode_mask = on;
    }

    /// Measurement only: one candle affine on a 1-element f16 tensor after
    /// each layer of [`Self::hidden_cached`], counted in `crate::boundary`.
    /// `LLVQ_BOUNDARY_AB` sets it on its `candle-tiny` arm. No served path
    /// calls it.
    pub fn set_candle_probe(&mut self, on: bool) -> Result<()> {
        self.candle_probe = match on {
            true => Some(Tensor::zeros(1, DType::F16, &self.device)?),
            false => None,
        };
        Ok(())
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
        // Under a preallocated window the keys are always `w` wide, so the
        // mask is too: the causal condition `j <= offset + i` already −infs
        // every position past the valid prefix, which is what makes the
        // zero-filled tail of the buffers exactly inert.
        let mask = self.step_mask(b, l, offset)?;
        let mut h = self.embed.forward(input)?;
        for (i, blk) in self.blocks.iter().enumerate() {
            h = blk.forward_cached(&h, &self.rotary, mask.as_ref(), RopeAt::Offset(offset), &mut caches[i], i, cap)?;
            if let Some(p) = &self.candle_probe {
                p.affine(2.0, 1.0)?;
                crate::boundary::note_added(1);
            }
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

    /// Build the stable buffers and ARM the caches' capturable write path.
    ///
    /// Requires a preallocated store (the fixed shapes are what make the
    /// stepped path meaningful), and must be called AFTER the prefill: the
    /// prefill's multi-position write stays on `slice_set`, and arming
    /// before it would scatter every prompt position to one slot.
    pub fn step_state(&self, caches: &mut [KvCache]) -> Result<StepState> {
        let w = match self.kv_store {
            crate::kvq::KvStore::Prealloc(w) => w,
            crate::kvq::KvStore::Cat | crate::kvq::KvStore::Grow(_) => candle_core::bail!(
                "the stepped path requires a preallocated store (LLVQ_KV_PREALLOC): \
                 its fixed shapes are what make it capturable"
            ),
        };
        // `scatter_set` wants its index shaped like the step it writes: the
        // expanded step carries every query head, the grouped one the key
        // heads only.
        let heads = match self.attn_heads {
            crate::kvq::AttnHeads::Expanded => {
                self.cfg.num_key_value_heads * (self.cfg.num_attention_heads / self.cfg.num_key_value_heads)
            }
            crate::kvq::AttnHeads::Shared | crate::kvq::AttnHeads::Grouped | crate::kvq::AttnHeads::Kernel => {
                self.cfg.num_key_value_heads
            }
        };
        let (cos, sin) = self.rotary.step_tables()?;
        let st = StepState {
            input: Tensor::zeros((1, 1), DType::U32, &self.device)?,
            mask: Tensor::zeros((1, 1, 1, w), self.dtype, &self.device)?,
            cos,
            sin,
            pos_idx: Tensor::zeros((1, heads, 1, self.cfg.head_dim), DType::U32, &self.device)?,
            w,
        };
        for c in caches.iter_mut() {
            c.set_write_index(st.pos_idx.clone());
        }
        Ok(st)
    }

    /// Refresh every stable buffer for the next step — OUTSIDE any capture.
    /// `next` is the token to feed, `offset` its absolute position. Each
    /// write is an in-place `slice_set` at a CONSTANT destination offset;
    /// the temporaries live and die host-side, before any capture begins.
    pub fn refresh_step(&self, st: &StepState, next: u32, offset: usize) -> Result<()> {
        st.input
            .slice_set(&Tensor::from_slice(&[next], (1, 1), &self.device)?, 0, 0)?;
        st.mask
            .slice_set(&self.causal_mask_width(1, 1, offset, st.w)?, 0, 0)?;
        self.rotary.refresh_step(&st.cos, &st.sin, offset)?;
        let pos = Tensor::full(offset as u32, st.pos_idx.dims4()?, &self.device)?;
        st.pos_idx.slice_set(&pos, 0, 0)?;
        Ok(())
    }

    /// `lm_head` from hidden states — the public form the stepped drivers
    /// need (the field itself stays private).
    pub fn project_head(&self, h: &Tensor) -> Result<Tensor> {
        self.head.project(h)
    }

    /// One decode step reading ONLY stable buffers — the capturable closure.
    /// Embed → blocks → norm → lm_head, logits out; the argmax and the D2H
    /// read stay with the caller, outside any capture.
    pub fn token_step(
        &self,
        st: &StepState,
        caches: &mut [KvCache],
        cap: &mut dyn Capture,
    ) -> Result<Tensor> {
        let mut h = self.embed.forward(&st.input)?;
        for (i, blk) in self.blocks.iter().enumerate() {
            h = blk.forward_cached(
                &h,
                &self.rotary,
                Some(&st.mask),
                RopeAt::Tables { cos: &st.cos, sin: &st.sin },
                &mut caches[i],
                i,
                cap,
            )?;
        }
        let h = self.norm.forward(&h)?;
        self.head.project(&h)
    }

    /// Greedy decode through the stepped path — same contract as
    /// [`Self::generate`], and the test that they return the SAME tokens is
    /// what certifies the buffer inventory of [`StepState`]: a missed
    /// per-token variation shows up as a divergence, never as a hunch.
    pub fn generate_stepped(&self, tokens: &[u32], max_new: usize) -> Result<Vec<u32>> {
        if max_new == 0 {
            return Ok(Vec::new());
        }
        let mut caches = self.fresh_caches();
        let mut out = Vec::with_capacity(max_new);
        let input = Tensor::from_slice(tokens, (1, tokens.len()), &self.device)?;
        let h = self.hidden_cached(&input, 0, &mut caches, &mut NoCapture)?;
        let st = self.step_state(&mut caches)?;
        let l = h.dim(1)?;
        let last = h.narrow(1, l - 1, 1)?;
        let logits = self.head.project(&last)?.to_dtype(DType::F32)?;
        let mut next = logits.i((0, 0))?.argmax(D::Minus1)?.to_scalar::<u32>()?;
        let mut offset = tokens.len();
        loop {
            out.push(next);
            if out.len() == max_new {
                return Ok(out);
            }
            self.refresh_step(&st, next, offset)?;
            let logits = self.token_step(&st, &mut caches, &mut NoCapture)?;
            let logits = logits.to_dtype(DType::F32)?;
            next = logits.i((0, 0))?.argmax(D::Minus1)?.to_scalar::<u32>()?;
            offset += 1;
        }
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

    /// [`Self::logits_last`] without the f32 cast: the logits of the last
    /// position of `h` in the model dtype, `[1, 1, vocab]`.
    ///
    /// For the greedy loops only. candle's `fast_argmax_f16` and
    /// `fast_argmax_f32` are one template (`reduce.cu:673`, `:688`), and
    /// widening f16 to f32 is exact and monotone, so the argmax of these
    /// logits is the argmax of their cast, ties and signed zeros included
    /// (`crate::argmax_key`). Callers that sample or score keep `logits_last`.
    pub fn logits_last_native(&self, h: &Tensor) -> Result<Tensor> {
        let l = h.dim(1)?;
        self.head.project(&h.narrow(1, l - 1, 1)?)
    }

    /// Greedy decode in one of the [`TokenLoop`] kinds. [`Self::generate`] is
    /// the base kind and stays untouched, so the published loop is the
    /// reference every kind is held to: the same tokens, which
    /// `token_loops_match_generate` pins on every store, form and dtype here
    /// and `LLVQ_GESTURE_AB` pins on the card.
    ///
    /// Each kind is one mechanism away from its neighbour, so an A/B ratio
    /// between neighbours measures one gesture (METHODE §1).
    pub fn generate_loop(&self, tokens: &[u32], max_new: usize, kind: &TokenLoop) -> Result<Vec<u32>> {
        if max_new == 0 {
            return Ok(Vec::new());
        }
        let mut caches = self.fresh_caches();
        // Key: one slot, read back every token. OnDevKey: one slot a token,
        // read back once at the end.
        let ring = match kind {
            TokenLoop::Key(k) => {
                k.arm()?;
                Some(Tensor::zeros(1, DType::U32, &self.device)?)
            }
            TokenLoop::OnDevKey(k) => {
                k.arm()?;
                Some(Tensor::zeros(max_new, DType::U32, &self.device)?)
            }
            TokenLoop::F16 | TokenLoop::OnDevCandle => None,
        };
        let input = Tensor::from_slice(tokens, (1, tokens.len()), &self.device)?;
        let mut h = self.hidden_cached(&input, 0, &mut caches, &mut NoCapture)?;
        let mut out = Vec::with_capacity(max_new); // host ids: F16, Key
        let mut kept = Vec::new(); // device ids: OnDevCandle
        // `offset` is the absolute position the token picked at `pos` is fed at.
        for (pos, offset) in (0..max_new).zip(tokens.len()..) {
            let logits = self.logits_last_native(&h)?;
            // Each kind picks its token, then stops BEFORE building the next
            // input on the last token, as `generate` does.
            let next = match kind {
                TokenLoop::F16 => {
                    let id = logits.i((0, 0))?.argmax(D::Minus1)?.to_scalar::<u32>()?;
                    out.push(id);
                    if pos + 1 == max_new {
                        break;
                    }
                    Tensor::from_slice(&[id], (1, 1), &self.device)?
                }
                TokenLoop::Key(k) => {
                    let slot = ring.as_ref().expect("built above for this kind");
                    k.write(&logits, slot, 0)?;
                    // A rank-0 view: `to_scalar` refuses any other rank. It
                    // copies the slot's 4 bytes, the wait every token pays.
                    let id = slot.i(0)?.to_scalar::<u32>()?;
                    out.push(id);
                    if pos + 1 == max_new {
                        break;
                    }
                    Tensor::from_slice(&[id], (1, 1), &self.device)?
                }
                TokenLoop::OnDevKey(k) => {
                    let r = ring.as_ref().expect("built above for this kind");
                    k.write(&logits, r, pos)?;
                    if pos + 1 == max_new {
                        break;
                    }
                    // A view of ring[pos]: the next step's gather reads the id
                    // where the kernel wrote it, with no copy and no wait.
                    r.narrow(0, pos, 1)?.reshape((1, 1))?
                }
                TokenLoop::OnDevCandle => {
                    let t = logits.i((0, 0))?.argmax_keepdim(D::Minus1)?;
                    kept.push(t.clone());
                    if pos + 1 == max_new {
                        break;
                    }
                    t.reshape((1, 1))?
                }
            };
            h = self.hidden_cached(&next, offset, &mut caches, &mut NoCapture)?;
        }
        match kind {
            TokenLoop::F16 => Ok(out),
            TokenLoop::Key(k) => {
                if k.nan_seen()? {
                    candle_core::bail!("generate_loop: a NaN logit; the key argmax refuses to pick a token");
                }
                Ok(out)
            }
            // The one wait of the decode.
            TokenLoop::OnDevKey(k) => {
                let v = ring.expect("built above for this kind").to_vec1::<u32>()?;
                if k.nan_seen()? {
                    candle_core::bail!("generate_loop: a NaN logit; the key argmax refuses to pick a token");
                }
                Ok(v)
            }
            TokenLoop::OnDevCandle => Tensor::cat(&kept, 0)?.to_vec1::<u32>(),
        }
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
            let mask = self.step_mask(1, 1, offset)?;
            let mut hh = e;
            for (i, blk) in self.blocks.iter().enumerate() {
                hh = blk.forward_cached(&hh, &self.rotary, mask.as_ref(), RopeAt::Offset(offset), &mut caches[i], i, &mut NoCapture)?;
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


/// The greedy token of `[1, 1, vocab]` logits, read back to the host.
pub fn argmax_last(logits: &Tensor) -> Result<u32> {
    logits.i((0, 0))?.argmax(D::Minus1)?.to_scalar::<u32>()
}

/// The log-probability of `target` under `[1, 1, vocab]` f32 logits.
pub fn logp_of(logits: &Tensor, target: u32) -> Result<f64> {
    let logp = candle_nn::ops::log_softmax(&logits.i((0, 0))?, D::Minus1)?;
    Ok(logp.i(target as usize)?.to_scalar::<f32>()? as f64)
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

/// The stable buffers of the stepped (capturable) decode path — every
/// per-token variation of a decode step, converted into the CONTENT of a
/// buffer at a stable address. What a CUDA graph replays is device work
/// against stable pointers; what changes between tokens must therefore
/// travel as bytes, never as a host value baked into a launch. This struct
/// is the exhaustive list of those bytes: the input id, the wide causal
/// mask, the one-position RoPE tables, and the KV write position.
pub struct StepState {
    input: Tensor,
    mask: Tensor,
    cos: Tensor,
    sin: Tensor,
    pos_idx: Tensor,
    w: usize,
}

impl StepState {
    /// Device read-backs for the graph diagnostics: what the buffers ACTUALLY
    /// contain after a refresh — not what the host believes it wrote.
    pub fn debug_input(&self) -> Result<u32> {
        self.input.flatten_all()?.to_vec1::<u32>().map(|v| v[0])
    }
    pub fn debug_pos(&self) -> Result<u32> {
        self.pos_idx.flatten_all()?.to_vec1::<u32>().map(|v| v[0])
    }
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
    // Built on the host in the dtype it is used in. +0 and -inf are exact in
    // every float dtype, so this is bit for bit the f32 mask cast on the
    // device, without its f32 copy: 134 MB at the second chunk of an
    // 8,192-token prompt, freed into the pool right before the attention.
    fn rows<T: Copy>(l: usize, w: usize, offset: usize, zero: T, ninf: T) -> Vec<T> {
        (0..l).flat_map(|i| (0..w).map(move |j| if j <= offset + i { zero } else { ninf })).collect()
    }
    let shape = (b, 1, l, w);
    match dtype {
        DType::F16 => Tensor::from_slice(&rows(l, w, offset, half::f16::ZERO, half::f16::NEG_INFINITY), shape, device),
        DType::BF16 => {
            Tensor::from_slice(&rows(l, w, offset, half::bf16::ZERO, half::bf16::NEG_INFINITY), shape, device)
        }
        _ => Tensor::from_slice(&rows(l, w, offset, 0f32, f32::NEG_INFINITY), shape, device)?.to_dtype(dtype),
    }
}

#[cfg(test)]
mod tests {

    /// A CPU int4 projection that declares `k` rows a launch, for the port's
    /// plumbing: `matvec_rows` is its own matvec row by row, as the port's
    /// contract demands of a real kernel, and it counts its launches.
    struct RowsInt4 {
        w: Tensor,
        k: usize,
        launches: std::sync::atomic::AtomicUsize,
    }

    impl crate::device::Int4Proj for RowsInt4 {
        fn name(&self) -> &str {
            "fake.int4"
        }
        fn d_out(&self) -> usize {
            self.w.dim(0).unwrap()
        }
        fn d_in(&self) -> usize {
            self.w.dim(1).unwrap()
        }
        fn matvec(&self, x: &Tensor, out_dims: &[usize]) -> Result<Tensor> {
            let y = x.reshape((1, self.d_in()))?.matmul(&self.w.t()?)?;
            let mut d = out_dims.to_vec();
            *d.last_mut().expect("rank >= 1") = self.d_out();
            y.reshape(d)
        }
        fn rows_per_launch(&self) -> usize {
            self.k
        }
        fn matvec_rows(&self, x: &Tensor, n: usize) -> Result<Tensor> {
            assert!((1..=self.k).contains(&n), "{n} rows for a launch of {}", self.k);
            assert_eq!(x.dims(), [n, self.d_in()], "the chunk as [rows, d_in]");
            self.launches.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let ys = (0..n)
                .map(|i| self.matvec(&x.narrow(0, i, 1)?, &[1, self.d_in()]))
                .collect::<Result<Vec<_>>>()?;
            Tensor::cat(&ys, 0)
        }
    }

    /// An int4 projection that declares rows a launch gets ceil(rows / k)
    /// launches, every row in its place, and the caller's shape back; the
    /// same numbers as the row-by-row loop it replaces.
    #[test]
    fn the_int4_rows_port_chunks_in_order_and_keeps_the_shape() {
        let dev = candle_core::Device::Cpu;
        let w = Tensor::randn(0f32, 1.0, (5, 8), &dev).unwrap();
        let x = Tensor::randn(0f32, 1.0, (1, 7, 8), &dev).unwrap();
        let fake = |k| std::sync::Arc::new(RowsInt4 { w: w.clone(), k, launches: 0.into() });
        let (one, three) = (fake(1), fake(3));
        let want = Proj::Int4(one.clone()).forward(&x).unwrap();
        let got = Proj::Int4(three.clone()).forward(&x).unwrap();
        assert_eq!(got.dims(), [1, 7, 5]);
        let flat = |t: &Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(flat(&got), flat(&want), "rows out of place or recomputed");
        assert_eq!(three.launches.load(std::sync::atomic::Ordering::SeqCst), 3, "7 rows at 3 a launch");
        assert_eq!(one.launches.load(std::sync::atomic::Ordering::SeqCst), 0, "k = 1 never reaches matvec_rows");
    }

    /// 4E's cache API on the Grow store: reserve keeps every held position
    /// and rounds the capacity up to the chunk; set_len refuses past it.
    #[test]
    fn grow_reserve_keeps_the_positions() {
        let dev = Device::Cpu;
        let mut c = KvCache::with_store(KvMode::F16, crate::kvq::KvStore::Grow(4));
        let k = Tensor::arange(0f32, 2.0 * 3.0 * 8.0, &dev).unwrap().reshape((1, 2, 3, 8)).unwrap();
        let v = (k.clone() * 2.0).unwrap();
        c.append(&k, &v).unwrap();
        assert_eq!(c.len(), 3);
        assert_eq!(c.grow_buffers().unwrap().0.dim(2).unwrap(), 4);
        c.reserve(9).unwrap();
        let (kb, vb) = c.grow_buffers().unwrap();
        assert_eq!(kb.dim(2).unwrap(), 12, "rounded up to the chunk");
        let held: Vec<f32> = kb.narrow(2, 0, 3).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(held, k.flatten_all().unwrap().to_vec1::<f32>().unwrap());
        let vh: Vec<f32> = vb.narrow(2, 0, 3).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(vh, v.flatten_all().unwrap().to_vec1::<f32>().unwrap());
        assert_eq!(c.len(), 3, "reserve moves no length");
        c.set_len(10).unwrap();
        assert_eq!(c.len(), 10);
        assert!(c.set_len(13).is_err());
        let mut cat = KvCache::new(KvMode::F16);
        assert!(cat.reserve(4).is_err(), "only the Grow store is written in place");
    }

    use super::*;

    fn dense(d_out: usize, d_in: usize) -> Proj {
        let w = Tensor::zeros((d_out, d_in), DType::F32, &Device::Cpu).expect("weights");
        Proj::Dense(Linear::new(w, None))
    }

    /// **T6.** A projection handed a rotation that is not its own refuses,
    /// naming itself.
    ///
    /// This is the one wiring mistake the value-based design leaves open, and
    /// it is three lines wide in `forward_cached`: the q/k/v group and the
    /// gate/up group both take a `hidden_size`-wide activation, so passing one
    /// group's [`Rotated`] to the other type-checks, runs, and returns finite
    /// plausible wrong numbers. Lives in the crate rather than in
    /// `tests/rotplan.rs` because [`Rotated`]'s fields are private — which is
    /// the point: outside this module the only way to get one is to rotate an
    /// activation.
    #[test]
    fn a_rotated_activation_of_the_wrong_key_is_refused() {
        let p = dense(4, 3);
        let x = Tensor::zeros((1usize, 3usize), DType::F32, &Device::Cpu).expect("x");
        let foreign = Rotated {
            key: Some((2560, 0xdead_beef)),
            t: x.clone(),
        };
        let e = p
            .forward_with(&foreign, &x)
            .expect_err("a foreign rotation must be refused");
        let msg = format!("{e}");
        assert!(msg.contains("2560"), "the message must cite the key: {msg}");
        assert!(
            msg.contains("dense projection"),
            "the message must name the site: {msg}"
        );
    }

    /// `prepare` on a dense projection is a strict no-op: the same tensor, no
    /// key, no allocation. Every path that is not the fused one goes through
    /// it, including the forward pass `bin/oracle` pins at `|Δhidden| = 0`.
    #[test]
    fn preparing_a_dense_projection_moves_nothing() {
        let p = dense(4, 3);
        let x = Tensor::arange(0f32, 3f32, &Device::Cpu)
            .expect("x")
            .reshape((1usize, 3usize))
            .expect("shape");
        let r = p.prepare(&x).expect("dense prepare");
        assert_eq!(r.key(), None);
        assert_eq!(
            x.id(),
            r.t.id(),
            "the dense prepare must return the tensor itself, not a copy"
        );
        assert!(p.forward_with(&r, &x).is_ok());
    }

    /// The row split and its inverse, on shapes a CPU tensor can carry.
    ///
    /// `group_forward`'s rotated branch is reachable only with a fused
    /// projection, so no test on this machine runs it end to end. Its shape
    /// bookkeeping — which row goes where, and what rank comes back — is the
    /// part that can be silently wrong, and it is pinned here: a transposed
    /// row order or a `reshape` on the wrong dims would return finite,
    /// plausible, wrong tokens on a card, with nothing red anywhere.
    #[test]
    fn the_row_split_is_undone_by_the_regroup() {
        for dims in [vec![1usize, 4], vec![1, 3, 4], vec![2, 3, 4]] {
            let n: usize = dims.iter().product();
            let x = Tensor::arange(0f32, n as f32, &Device::Cpu)
                .expect("x")
                .reshape(dims.clone())
                .expect("shape");
            let rows = row_views(&x).expect("split");
            let expected_rows: usize = dims[..dims.len() - 1].iter().product();
            assert_eq!(rows.len(), expected_rows);
            for r in &rows {
                assert_eq!(r.dims(), &[1, *dims.last().expect("rank >= 1")]);
            }
            let back = regroup(&rows, &dims, *dims.last().expect("rank >= 1")).expect("regroup");
            assert_eq!(back.dims(), dims.as_slice());
            assert_eq!(
                back.flatten_all().expect("flat").to_vec1::<f32>().expect("vals"),
                x.flatten_all().expect("flat").to_vec1::<f32>().expect("vals"),
                "dims {dims:?}: the row split is not returned identically"
            );

            // The order is load-bearing, and this says so: reversing the rows
            // must change the answer whenever there is more than one.
            if expected_rows > 1 {
                let mut rev = rows.clone();
                rev.reverse();
                let other =
                    regroup(&rev, &dims, *dims.last().expect("rank >= 1")).expect("regroup");
                assert_ne!(
                    other.flatten_all().expect("f").to_vec1::<f32>().expect("v"),
                    back.flatten_all().expect("f").to_vec1::<f32>().expect("v"),
                    "dims {dims:?}: the row order changes nothing, the test proves nothing"
                );
            }
        }
    }

    /// The width changes between the two halves: `d_in` in, `d_out` out.
    #[test]
    fn the_regroup_carries_the_output_width() {
        let rows: Vec<Tensor> = (0..3)
            .map(|r| {
                Tensor::arange(0f32, 5f32, &Device::Cpu)
                    .expect("row")
                    .affine(1.0, r as f64)
                    .expect("offset")
                    .reshape((1usize, 5usize))
                    .expect("shape")
            })
            .collect();
        let out = regroup(&rows, &[1, 3, 7], 5).expect("regroup");
        assert_eq!(out.dims(), &[1, 3, 5]);
    }

    /// A group of projections in the natural basis is driven identically in
    /// both modes — the branch that keeps every dense path byte-identical.
    #[test]
    fn a_group_without_rotation_ignores_the_mode() {
        let a = dense(2, 3);
        let b = dense(5, 3);
        let x = Tensor::arange(0f32, 6f32, &Device::Cpu)
            .expect("x")
            .reshape((1usize, 2usize, 3usize))
            .expect("shape");
        for share in [RotShare::Off, RotShare::On] {
            let out = group_forward(&[&a, &b], &x, share).expect("group");
            assert_eq!(out.len(), 2);
            assert_eq!(out[0].dims(), &[1, 2, 2]);
            assert_eq!(out[1].dims(), &[1, 2, 5]);
        }

        // And it does not go through the row loop at all — observable, because
        // the row loop carries `MAX_ROWS`, which a dense `Linear` has no reason
        // to obey. A refactor that routed dense projections through the rotated
        // branch "for uniformity" would break scoring on any window wider than
        // 256, and would otherwise be invisible: the numbers would be right.
        let wide = Tensor::zeros((1usize, MAX_ROWS + 44, 3usize), DType::F32, &Device::Cpu)
            .expect("wide");
        let out = group_forward(&[&a], &wide, RotShare::On).expect("dense on a wide window");
        assert_eq!(out[0].dims(), &[1, MAX_ROWS + 44, 2]);
    }

    /// Step 1b invariant: the expanded fixed-window store must hold, in its
    /// valid prefix, EXACTLY the bytes the `Cat` path hands to attention
    /// (`repeat_kv` of the history), and exactly zeros past it. A prefill
    /// then single-token steps, checked at every step — this is what catches
    /// a stale `len`, a wrong dim, a slice landing one position off, or an
    /// expansion done in the wrong order relative to Q8 quantization.
    #[test]
    fn prealloc_expanded_matches_repeat_kv_of_cat_at_every_step() {
        use crate::kvq::{KvMode, KvStore};
        use candle_transformers::utils::repeat_kv;
        let dev = Device::Cpu;
        let (b, n_kv, hd, groups, w) = (1, 2, 64, 4, 16); // hd a multiple of the q8 GROUP
        for mode in [KvMode::F16, KvMode::Q8] {
            let mut cat = KvCache::new(mode);
            let mut pre = KvCache::with_store(mode, KvStore::Prealloc(w));
            let mut pos = 0u32;
            let mut len = 0usize;
            for (step, l) in [5usize, 1, 1, 1, 1, 1].into_iter().enumerate() {
                let n = b * n_kv * l * hd;
                let kd: Vec<f32> = (0..n).map(|i| (pos as f32) + i as f32 * 0.25).collect();
                let vd: Vec<f32> = (0..n).map(|i| -(pos as f32) - i as f32 * 0.5).collect();
                pos += l as u32;
                len += l;
                let k = Tensor::from_vec(kd, (b, n_kv, l, hd), &dev).expect("k");
                let v = Tensor::from_vec(vd, (b, n_kv, l, hd), &dev).expect("v");
                // Cat path: append then repeat_kv, what forward_cached does
                let (ck, cv) = cat.append(&k, &v).expect("cat append");
                let ck = repeat_kv(ck, groups).unwrap().contiguous().unwrap();
                let cv = repeat_kv(cv, groups).unwrap().contiguous().unwrap();
                // Prealloc path: expand THEN append, what forward_cached does
                let ke = repeat_kv(k.clone(), groups).unwrap().contiguous().unwrap();
                let ve = repeat_kv(v.clone(), groups).unwrap().contiguous().unwrap();
                let (pk, pv) = pre.append(&ke, &ve).expect("prealloc append");
                assert_eq!(pk.dims(), &[b, n_kv * groups, w, hd], "step {step}: full buffer");
                for (full, reference, name) in [(&pk, &ck, "k"), (&pv, &cv, "v")] {
                    let prefix = full.narrow(2, 0, len).unwrap().contiguous().unwrap();
                    let a: Vec<f32> = reference.flatten_all().unwrap().to_vec1().unwrap();
                    let bb: Vec<f32> = prefix.flatten_all().unwrap().to_vec1().unwrap();
                    assert_eq!(a, bb, "step {step} ({mode:?}): prefix {name}");
                    let tail = full.narrow(2, len, w - len).unwrap();
                    let t: Vec<f32> = tail.flatten_all().unwrap().to_vec1().unwrap();
                    assert!(t.iter().all(|x| *x == 0.0), "step {step} ({mode:?}): tail {name} not zero");
                }
            }
        }
    }

    /// The padding of the fixed window must be EXACTLY inert through the
    /// whole attention arithmetic: same scores→softmax→ctx bytes over `len`
    /// keys as over `w` keys whose tail is zeros behind −inf. `exp(−inf) = 0`
    /// and `x + 0.0 = x` are what make this bitwise, not approximate — if
    /// either stops holding (a mask built at a finite floor, say), this test
    /// is the tripwire. It exercises `build_causal_mask` itself, not a
    /// re-derivation.
    #[test]
    fn wide_mask_padding_is_exactly_inert() {
        let dev = Device::Cpu;
        let (b, h, l, hd, offset, len, w) = (1usize, 2, 1, 8, 3, 4, 16);
        let q = Tensor::from_vec(
            (0..b * h * l * hd).map(|i| (i as f32).sin()).collect::<Vec<f32>>(),
            (b, h, l, hd),
            &dev,
        )
        .unwrap();
        let mk = |n: usize, sgn: f32| {
            Tensor::from_vec(
                (0..b * h * n * hd).map(|i| sgn * (i as f32).cos()).collect::<Vec<f32>>(),
                (b, h, n, hd),
                &dev,
            )
            .unwrap()
        };
        let (k, v) = (mk(len, 1.0), mk(len, -1.0));
        let zpad = Tensor::zeros((b, h, w - len, hd), DType::F32, &dev).unwrap();
        let kw = Tensor::cat(&[&k, &zpad], 2).unwrap().contiguous().unwrap();
        let vw = Tensor::cat(&[&v, &zpad], 2).unwrap().contiguous().unwrap();
        let attn = |k: &Tensor, v: &Tensor, width: usize| -> Vec<f32> {
            let mask = build_causal_mask(b, l, offset, width, DType::F32, &dev).unwrap();
            let scores = q.matmul(&k.transpose(2, 3).unwrap()).unwrap();
            let scores = scores.broadcast_add(&mask).unwrap();
            let p = candle_nn::ops::softmax_last_dim(&scores).unwrap();
            p.matmul(v).unwrap().flatten_all().unwrap().to_vec1().unwrap()
        };
        assert_eq!(attn(&k, &v, len), attn(&kw, &vw, w), "the padding is not inert");
    }

    #[test]
    fn prealloc_overflow_is_a_named_error() {
        use crate::kvq::{KvMode, KvStore};
        let dev = Device::Cpu;
        let mut c = KvCache::with_store(KvMode::F16, KvStore::Prealloc(4));
        let k = Tensor::zeros((1, 2, 3, 4), DType::F32, &dev).unwrap();
        c.append(&k, &k).expect("3 out of 4 passes");
        let e = c.append(&k, &k).expect_err("3+3 out of 4 must refuse");
        let msg = format!("{e}");
        assert!(msg.contains("window"), "message without a mechanism: {msg}");
    }

    /// THE certifier of the stepped path: on a tiny random model, the three
    /// decodes — `Cat`, `Prealloc` (eager), and `Prealloc` stepped (stable
    /// buffers + scatter write, the capturable closure) — must return the
    /// SAME tokens. A per-token variation missing from [`StepState`]'s
    /// inventory (mask, RoPE, input, write position) diverges here, on CPU,
    /// for free — not on a billed card.
    #[test]
    fn stepped_decode_matches_generate_on_all_stores() {
        use crate::kvq::KvStore;
        let mut model = tiny_model(DType::F32);
        let prompt = [3u32, 17, 42, 7];
        let n_new = 8;
        let cat = {
            model.set_kv_store(KvStore::Cat);
            model.generate(&prompt, n_new, &mut NoCapture).expect("cat")
        };
        let pre = {
            model.set_kv_store(KvStore::Prealloc(32));
            model.generate(&prompt, n_new, &mut NoCapture).expect("prealloc")
        };
        let stepped = {
            model.set_kv_store(KvStore::Prealloc(32));
            model.generate_stepped(&prompt, n_new).expect("stepped")
        };
        assert_eq!(cat, pre, "cat against eager prealloc");
        assert_eq!(pre, stepped, "eager prealloc against stepped (incomplete StepState?)");
        // The unexpanded forms write `n_kv` heads, so the write index must
        // be sized for them: a wrong shape refuses inside `scatter_set`.
        model.set_kv_store(KvStore::Prealloc(32));
        for heads in [crate::kvq::AttnHeads::Shared, crate::kvq::AttnHeads::Grouped] {
            model.set_attn_heads(heads);
            let eager = model.generate(&prompt, n_new, &mut NoCapture).expect("eager, unexpanded");
            let s = model.generate_stepped(&prompt, n_new).expect("stepped, unexpanded");
            assert_eq!(eager, s, "stepped under {}", heads.label());
        }
    }

    /// The f32 logits of a chunked prefill, then of one decode step per
    /// token of `feed`. Feeding fixed tokens rather than each arm's own
    /// greedy choice compares every arm on the same inputs: a near tie that
    /// one rounding flips would otherwise send two arms down different texts.
    fn logit_trace(m: &Qwen3, prompt: &[u32], chunk: usize, feed: &[u32]) -> Vec<Vec<f32>> {
        let mut c = m.fresh_caches();
        let mut out = vec![m.prefill_chunked(prompt, chunk, &mut c).unwrap()];
        for (offset, &t) in (prompt.len()..).zip(feed) {
            out.push(m.step_logits(t, offset, &mut c).unwrap());
        }
        out.iter().map(|l| l.flatten_all().unwrap().to_vec1::<f32>().unwrap()).collect()
    }

    fn max_abs_diff(a: &[Vec<f32>], b: &[Vec<f32>]) -> f32 {
        a.iter().flatten().zip(b.iter().flatten()).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
    }

    fn same_bits(a: &[Vec<f32>], b: &[Vec<f32>]) -> bool {
        a.len() == b.len()
            && a.iter().zip(b).all(|(x, y)| {
                x.len() == y.len() && x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits())
            })
    }

    /// Pseudo-random values in about [-amp, amp], as a tensor of `dtype`.
    fn noise(shape: &[usize], seed: u64, amp: f32, dtype: DType, dev: &Device) -> Tensor {
        let n: usize = shape.iter().product();
        let mut s = seed.wrapping_mul(0x9e3779b97f4a7c15) | 1;
        let v: Vec<f32> = (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * amp
            })
            .collect();
        Tensor::from_vec(v, shape, dev).unwrap().to_dtype(dtype).unwrap()
    }

    fn bits_of(t: &Tensor) -> Vec<u32> {
        t.flatten_all().unwrap().to_dtype(DType::F32).unwrap().to_vec1::<f32>().unwrap().iter().map(|x| x.to_bits()).collect()
    }

    /// Run `f` with the in-place softmax forced from `from` bytes, in blocks
    /// of `rows` rows, on this thread only; the defaults come back after.
    fn with_in_place<R>(from: usize, rows: usize, f: impl FnOnce() -> R) -> R {
        let old = (IN_PLACE_FROM.with(|c| c.replace(from)), SOFTMAX_ROWS_T.with(|c| c.replace(rows)));
        let r = f();
        IN_PLACE_FROM.with(|c| c.set(old.0));
        SOFTMAX_ROWS_T.with(|c| c.set(old.1));
        r
    }

    /// The in-place softmax against the whole-tensor one, bit for bit, on
    /// both attention bodies that call it: the grouped form and the expanded
    /// one (its scores are `[b, n_heads, l, L]`, the same slabs). Chunks at an
    /// offset (masked columns past the diagonal), blocks that do not divide
    /// `l` (a remainder block), a block larger than `l`, a batch of two under
    /// a one-row mask, f16 and f32, groups of 4 (the 4B and 8B) and 5 (the
    /// 14B). The scores span about +-24 before the scale, so the softmax
    /// meets near-zero and saturated probabilities both.
    fn in_place_matches_whole_on(dev: &Device) {
        for dtype in [DType::F16, DType::F32] {
            for (b, n_kv, groups, l, offset, hd) in
                [(1, 2, 4, 7, 5, 16), (1, 2, 5, 9, 0, 16), (2, 2, 4, 6, 3, 8), (1, 1, 4, 16, 40, 32)]
            {
                let n = offset + l;
                let q = noise(&[b, n_kv * groups, l, hd], 1, 1.5, dtype, dev);
                let k = noise(&[b, n_kv, n, hd], 2, 1.5, dtype, dev);
                let v = noise(&[b, n_kv, n, hd], 3, 1.0, dtype, dev);
                let mask = build_causal_mask(1, l, offset, n, dtype, dev).unwrap();
                let scale = 1.0 / (hd as f64).sqrt();
                let whole = with_in_place(usize::MAX, SOFTMAX_ROWS, || {
                    grouped_attention(&q, &k, &v, Some(&mask), scale).unwrap()
                });
                let expanded = |q: &Tensor| {
                    let ke = candle_transformers::utils::repeat_kv(k.clone(), groups).unwrap().contiguous().unwrap();
                    let ve = candle_transformers::utils::repeat_kv(v.clone(), groups).unwrap().contiguous().unwrap();
                    let raw = q.matmul(&ke.transpose(2, 3).unwrap()).unwrap();
                    let p = match in_place(&raw) {
                        true => {
                            softmax_in_place(&raw, &mask, scale, l).unwrap();
                            raw
                        }
                        false => {
                            candle_nn::ops::softmax_last_dim(&(raw * scale).unwrap().broadcast_add(&mask).unwrap())
                                .unwrap()
                        }
                    };
                    p.matmul(&ve).unwrap()
                };
                let whole_x = with_in_place(usize::MAX, SOFTMAX_ROWS, || expanded(&q));
                for rows in [1, 3, l, 4 * l] {
                    let arm = format!("{dtype:?} b{b} {n_kv}x{groups} l{l} off{offset} rows{rows}");
                    let got = with_in_place(0, rows, || grouped_attention(&q, &k, &v, Some(&mask), scale).unwrap());
                    assert_eq!(bits_of(&got), bits_of(&whole), "grouped, {arm}");
                    let got = with_in_place(0, rows, || expanded(&q));
                    assert_eq!(bits_of(&got), bits_of(&whole_x), "expanded, {arm}");
                }
            }
        }
    }

    #[test]
    fn in_place_softmax_matches_the_whole_tensor() {
        in_place_matches_whole_on(&Device::Cpu);
        // Refused, not mis-indexed: a mask of the wrong height, a non-slab.
        let dev = Device::Cpu;
        let raw = noise(&[1, 2, 6, 9], 4, 1.0, DType::F32, &dev);
        let mask = build_causal_mask(1, 4, 5, 9, DType::F32, &dev).unwrap();
        assert!(softmax_in_place(&raw, &mask, 1.0, 4).is_err(), "6 rows are not slabs of 4");
        let mask = build_causal_mask(1, 3, 6, 9, DType::F32, &dev).unwrap();
        assert!(softmax_in_place(&raw, &mask, 1.0, 6).is_err(), "a mask of 3 rows for slabs of 6");
    }

    /// The same identity on Metal, where candle's kernels are not the CPU's.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn in_place_softmax_matches_the_whole_tensor_on_metal() {
        let dev = Device::new_metal(0).expect("a Metal device");
        in_place_matches_whole_on(&dev);
    }

    /// The mask built in its own dtype is the f32 mask cast, bit for bit.
    #[test]
    fn the_causal_mask_is_built_in_its_dtype() {
        for dtype in [DType::F16, DType::BF16, DType::F32] {
            for (l, offset, w) in [(1, 0, 1), (4, 0, 4), (5, 3, 8), (3, 2, 9)] {
                let got = build_causal_mask(1, l, offset, w, dtype, &Device::Cpu).unwrap();
                let m: Vec<f32> = (0..l)
                    .flat_map(|i| (0..w).map(move |j| if j <= offset + i { 0.0 } else { f32::NEG_INFINITY }))
                    .collect();
                let want = Tensor::from_slice(&m, (1, 1, l, w), &Device::Cpu).unwrap().to_dtype(dtype).unwrap();
                assert_eq!(got.dtype(), dtype);
                assert_eq!(bits_of(&got), bits_of(&want), "{dtype:?} l{l} offset{offset} w{w}");
            }
        }
    }

    /// The whole model with the in-place softmax forced on every masked call
    /// (blocks of 3 rows, so most chunks end on a remainder) against the
    /// default, where none of these tiny calls reaches it: the same logits,
    /// bit for bit, through a chunked prefill and 12 decode steps, on every
    /// form and store a config can name, and in the uncached full window.
    #[test]
    fn the_in_place_softmax_moves_no_logit() {
        use crate::kvq::{AttnHeads, KvStore, GROW_CHUNK};
        let prompt = [3u32, 17, 42, 7, 9, 11, 5, 60, 11, 90, 2];
        let feed = [5u32, 60, 11, 90, 2, 33, 71, 8, 13, 44, 1, 96];
        let window = Tensor::from_slice(&[5u32, 60, 11, 90, 2, 33, 71, 8, 13], (1, 9), &Device::Cpu).unwrap();
        for dtype in [DType::F16, DType::F32] {
            let mut m = tiny_model_hd(dtype, 8, 2, 128);
            for (store, heads) in [
                (KvStore::Cat, AttnHeads::Expanded),
                (KvStore::Prealloc(32), AttnHeads::Expanded),
                (KvStore::Grow(GROW_CHUNK), AttnHeads::Grouped),
                (KvStore::Grow(4), AttnHeads::Grouped),
                (KvStore::Grow(GROW_CHUNK), AttnHeads::Kernel),
            ] {
                m.set_kv_store(store);
                m.set_attn_heads(heads);
                if heads == AttnHeads::Kernel {
                    if dtype != DType::F16 {
                        continue;
                    }
                    m.set_attn_kernel(Some(crate::attn::kernel_for(&Device::Cpu, 2, 4, 128).unwrap()));
                }
                let run = |m: &Qwen3| {
                    let full = m.logits(&window, &mut NoCapture).unwrap();
                    (logit_trace(m, &prompt, 4, &feed), logit_trace(m, &prompt, 16, &feed), bits_of(&full))
                };
                let base = with_in_place(usize::MAX, SOFTMAX_ROWS, || run(&m));
                let forced = with_in_place(0, 3, || run(&m));
                let arm = format!("{dtype:?} {} {}", store.label(), heads.label());
                assert!(same_bits(&base.0, &forced.0), "{arm}: chunks of 4");
                assert!(same_bits(&base.1, &forced.1), "{arm}: one chunk of 11");
                assert_eq!(base.2, forced.2, "{arm}: full window");
            }
        }
    }

    /// The kernel prefill on the CPU replay. Its point is one identity: the
    /// prompt's logits do not depend on how it is chunked, because each
    /// prompt row goes to the kernel and a kernel row depends on its own
    /// position only. The grouped prefill moves with every chunking (its
    /// matmul shapes do). One caveat belongs to the CPU, not to the kernel:
    /// candle's CPU gemm rounds a one-row product differently from a
    /// many-row one, so a chunking that leaves a piece of one row (19 as two
    /// 9s and a 1) moves the dense projections of this tiny model, under
    /// every form.
    /// On the card the projections' rows kernels own that identity, and the
    /// card gate reads it.
    #[test]
    fn the_kernel_prefill_does_not_depend_on_the_chunking() {
        use crate::kvq::{AttnHeads, KvStore, PrefillAttn, GROW_CHUNK};
        let mut m = tiny_model_hd(DType::F16, 8, 2, 128);
        let prompt = [3u32, 17, 42, 7, 9, 11, 5, 60, 11, 90, 2, 33, 71, 8, 13, 44, 1, 96, 27];
        let feed = [5u32, 60, 11, 90, 2, 33];
        m.set_kv_store(KvStore::Grow(GROW_CHUNK));
        m.set_attn_heads(AttnHeads::Kernel);
        m.set_attn_kernel(Some(crate::attn::kernel_for(&Device::Cpu, 2, 4, 128).expect("CPU kernel")));
        m.set_kernel_from(0);
        assert_eq!(m.prefill_attention(), PrefillAttn::Grouped, "the default is today's prefill");
        // Chunkings with no piece of one row: 19, 16 + 3, 11 + 8, 8 + 8 + 3,
        // 4 x 4 + 3. Every piece above KMAX goes through the kernel prefill;
        // 8 and 4 are plain kernel calls.
        let chunkings = [19, 16, 11, 8, 4];
        let grouped: Vec<_> = chunkings.iter().map(|&c| logit_trace(&m, &prompt, c, &feed)).collect();
        assert!(
            grouped[1..].iter().any(|g| !same_bits(g, &grouped[0])),
            "the grouped prefill did not move with the chunking, so this test cannot see the kernel's"
        );
        m.set_prefill_attention(PrefillAttn::Kernel);
        let kernel: Vec<_> = chunkings.iter().map(|&c| logit_trace(&m, &prompt, c, &feed)).collect();
        for (c, k) in chunkings.iter().zip(&kernel) {
            assert!(same_bits(k, &kernel[0]), "chunks of {c}: the kernel prefill moved with the chunking");
        }
        // Not the grouped prefill, and close to it.
        let d = max_abs_diff(&grouped[0][..1], &kernel[0][..1]);
        println!("kernel prefill against grouped: max |dlogit| {d:e}");
        assert!(d > 0.0 && d <= 5e-3, "kernel prefill against grouped: max |dlogit| {d:e}");
        let stepped = |m: &Qwen3| logit_trace(m, &prompt, 19, &feed);

        // Below a nonzero threshold the early rows are the grouped form's,
        // block by block: a call that straddles the threshold still runs.
        m.set_kernel_from(10);
        let mixed = logit_trace(&m, &prompt, 19, &feed);
        assert!(mixed.iter().flatten().all(|x| x.is_finite()));
        m.set_kernel_from(0);

        // The other forms never read the field; f32 is refused by name.
        m.set_attn_heads(AttnHeads::Grouped);
        let with_field = stepped(&m);
        m.set_prefill_attention(PrefillAttn::Grouped);
        assert!(same_bits(&stepped(&m), &with_field), "the grouped form ignores the field");
        m.set_prefill_attention(PrefillAttn::Kernel);
        let mut wide = tiny_model_hd(DType::F32, 8, 2, 128);
        wide.set_kv_store(KvStore::Grow(GROW_CHUNK));
        wide.set_attn_heads(AttnHeads::Kernel);
        wide.set_attn_kernel(Some(crate::attn::kernel_for(&Device::Cpu, 2, 4, 128).expect("CPU kernel")));
        wide.set_prefill_attention(PrefillAttn::Kernel);
        let mut c = wide.fresh_caches();
        let e = wide.prefill_chunked(&prompt, 19, &mut c).expect_err("f32 prefill through the kernel");
        assert!(e.to_string().contains("f16 only"), "{e}");
    }

    /// The kernel prefill's dispatch against the kernel called row by row,
    /// at the attention: the same bits for every row, whatever the call.
    #[test]
    fn kernel_prefill_rows_are_the_kernel_rows() {
        let dev = Device::Cpu;
        let kern = crate::attn::kernel_for(&dev, 2, 4, 128).expect("CPU kernel");
        let (n_kv, groups, l, offset) = (2, 4, 21, 300);
        let n = offset + l;
        let q = noise(&[1, n_kv * groups, l, 128], 5, 1.0, DType::F16, &dev);
        let k = noise(&[1, n_kv, n, 128], 6, 1.0, DType::F16, &dev);
        let v = noise(&[1, n_kv, n, 128], 7, 1.0, DType::F16, &dev);
        let mask = build_causal_mask(1, l, offset, n, DType::F16, &dev).unwrap();
        let got = kernel_prefill(kern.as_ref(), &q, &k, &v, Some(&mask), 0.0884, 0).unwrap();
        for r in 0..l {
            let seen = offset + r + 1;
            let one = kern
                .attend(&q.narrow(2, r, 1).unwrap().contiguous().unwrap(), &k.narrow(2, 0, seen).unwrap(), &v.narrow(2, 0, seen).unwrap())
                .unwrap();
            assert_eq!(bits_of(&got.narrow(2, r, 1).unwrap()), bits_of(&one), "row {r}");
        }
    }

    /// The three attention forms over the stores that hand attention exactly
    /// the history. Against `Cat` with expanded heads, every published path:
    /// the `Grow` store alone and the `Shared` form are bit for bit, in the
    /// chunked prefill (a masked chunk at an offset), in 20 decode steps and
    /// in the uncached full-window forward; `Grouped` changes the matmul
    /// shape, and is held to a bound on the logits instead, about ten times
    /// the largest difference measured on 2026-11-01. It is not held to the
    /// same greedy text: on this random model, whose logits tie often, one
    /// rounding difference flipped a token at the ninth step.
    /// Three head layouts: 4 over 2; 8 over 2, groups of 4 as in Qwen3-4B,
    /// where a group and a key head no longer have the same count; and 10
    /// over 2, groups of 5 as in Qwen3-14B (40 over 8), an odd group size.
    #[test]
    fn attention_forms_agree() {
        use crate::kvq::{AttnHeads, KvStore};
        for (dtype, bound, n_heads) in [
            (DType::F32, 6e-7f32, 4),
            (DType::F16, 4e-3, 4),
            (DType::F32, 6e-7, 8),
            (DType::F16, 4e-3, 8),
            (DType::F32, 6e-7, 10),
            (DType::F16, 4e-3, 10),
        ] {
            let mut m = tiny_model_heads(dtype, n_heads, 2);
            let prompt = [3u32, 17, 42, 7, 9, 11];
            let window = Tensor::from_slice(&[5u32, 60, 11, 90, 2, 33, 71, 8, 13], (1, 9), &Device::Cpu).unwrap();
            let full = |m: &Qwen3| {
                vec![m.logits(&window, &mut NoCapture).unwrap().flatten_all().unwrap().to_dtype(DType::F32).unwrap().to_vec1::<f32>().unwrap()]
            };
            let feed = [5u32, 60, 11, 90, 2, 33, 71, 8, 13, 44, 1, 96, 27, 64, 38, 19, 80, 6, 52, 23];
            let reference = (logit_trace(&m, &prompt, 4, &feed), full(&m));
            for (store, heads, exact) in [
                (KvStore::Grow(4), AttnHeads::Expanded, true),
                (KvStore::Cat, AttnHeads::Shared, true),
                (KvStore::Grow(4), AttnHeads::Shared, true),
                (KvStore::Grow(1), AttnHeads::Shared, true),
                (KvStore::Cat, AttnHeads::Grouped, false),
                (KvStore::Grow(4), AttnHeads::Grouped, false),
            ] {
                m.set_kv_store(store);
                m.set_attn_heads(heads);
                let arm = format!("{dtype:?} {n_heads}/2 {} {}", store.label(), heads.label());
                let got = (logit_trace(&m, &prompt, 4, &feed), full(&m));
                if exact {
                    assert!(same_bits(&reference.0, &got.0), "{arm}: cached logits not bit for bit");
                    assert!(same_bits(&reference.1, &got.1), "{arm}: full-window logits not bit for bit");
                } else {
                    let d = max_abs_diff(&reference.0, &got.0).max(max_abs_diff(&reference.1, &got.1));
                    println!("{arm}: max |dlogit| {d:e}");
                    assert!(d <= bound, "{arm}: max |dlogit| {d:e} above {bound:e}");
                    // Liveness, on macOS only: there the grouped matmul rounds
                    // differently from the expanded one, so equal bits would
                    // mean the arm never took it. On x86-64 Linux candle's CPU
                    // gemm returns the same bits for both layouts (CI,
                    // 2026-11-07, F32 4/2), so the check would fail on a
                    // correct grouped path; the bound above still holds there.
                    #[cfg(target_os = "macos")]
                    assert!(d > 0.0, "{arm}: bit for bit, so this arm is not testing the grouped matmul");
                }
            }
        }
    }

    /// What the unexpanded forms hand their matmuls: the transposed output of
    /// a projection is laid out again, rows row-major; a strided view of a
    /// `Grow` buffer keeps its storage and strides, so no history is copied.
    #[test]
    fn rows_contiguous_copies_only_what_a_matmul_refuses() {
        let dev = Device::Cpu;
        let x = Tensor::arange(0f32, 24.0, &dev).unwrap().reshape((1, 4, 2, 3)).unwrap();
        let t = x.transpose(1, 2).unwrap();
        assert_eq!(t.stride(), &[24, 3, 6, 1]);
        let r = rows_contiguous(&t).unwrap();
        assert_eq!(r.stride(), &[24, 12, 3, 1]);
        let flat = |t: &Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(flat(&r), flat(&t));
        let buf = Tensor::arange(0f32, 48.0, &dev).unwrap().reshape((1, 2, 8, 3)).unwrap();
        let view = buf.narrow(2, 0, 5).unwrap();
        let r = rows_contiguous(&view).unwrap();
        assert_eq!(r.stride(), &[48, 24, 3, 1], "a Grow view must not be copied");
        assert_eq!(flat(&r), flat(&view));
    }

    /// The teacher-forced decode scores the next true token, as the prefill
    /// does: on one arm twice it agrees with itself bit for bit and with
    /// [`Qwen3::window_nll`] to rounding. Against the grouped form it moves by
    /// rounding only, and against the shared form not at all on the CPU.
    #[test]
    fn paired_decode_scores_what_the_window_scores() {
        use crate::kvq::{AttnHeads, KvStore};
        let mut m = tiny_model_heads(DType::F32, 8, 2);
        let tokens = [5u32, 60, 11, 90, 2, 33, 71, 8, 13, 44, 1, 96, 27, 64, 38, 19];
        let published = (KvStore::Cat, AttnHeads::Expanded);
        let same = m.decode_nll_paired(&tokens, [published, published]).unwrap();
        assert_eq!(same.count, tokens.len() - 1);
        assert_eq!(same.nll[0].to_bits(), same.nll[1].to_bits());
        assert_eq!((same.top1_agree, same.max_dlogit), (same.count, 0.0));
        m.set_kv_store(KvStore::Cat);
        m.set_attn_heads(AttnHeads::Expanded);
        let (window, n) = m.window_nll(&tokens, &mut NoCapture).unwrap();
        assert_eq!(n, same.count);
        let rel = (same.nll[0] / window - 1.0).abs();
        assert!(rel < 1e-5, "decode against prefill: relative {rel:e}");
        let shared = m.decode_nll_paired(&tokens, [published, (KvStore::Grow(4), AttnHeads::Shared)]).unwrap();
        assert_eq!(shared.nll[0].to_bits(), shared.nll[1].to_bits());
        assert_eq!(shared.max_dlogit, 0.0);
        let grouped = m.decode_nll_paired(&tokens, [published, (KvStore::Grow(4), AttnHeads::Grouped)]).unwrap();
        assert!(grouped.max_dlogit < 6e-7, "{grouped:?}");
        // Above zero on macOS, where the grouped matmul rounds differently;
        // x86-64 Linux returns the same bits (see `attention_forms_agree`).
        #[cfg(target_os = "macos")]
        assert!(grouped.max_dlogit > 0.0, "{grouped:?}");
        assert!((grouped.nll[1] / grouped.nll[0] - 1.0).abs() < 1e-6, "{grouped:?}");
        assert!(m.decode_nll_paired(&tokens[..1], [published, published]).is_err());
    }

    /// At every append, across chunk boundaries and appends wider than a
    /// chunk, the `Grow` store hands attention the very history `Cat` does,
    /// in buffers sized to a whole number of chunks.
    #[test]
    fn grow_store_holds_what_cat_holds() {
        use crate::kvq::{KvMode, KvStore};
        let dev = Device::Cpu;
        let mut cat = KvCache::with_store(KvMode::F16, KvStore::Cat);
        let mut grow = KvCache::with_store(KvMode::F16, KvStore::Grow(4));
        let mut seed = 1u32;
        for l in [5usize, 1, 1, 3, 1, 1, 1, 1, 9, 1, 2] {
            let mut step = || {
                let n = 2 * l * 3;
                let v: Vec<f32> = (0..n)
                    .map(|_| {
                        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                        (seed >> 8) as f32 / (1u32 << 24) as f32
                    })
                    .collect();
                Tensor::from_vec(v, (1, 2, l, 3), &dev).unwrap()
            };
            let (k, v) = (step(), step());
            let (ck, cv) = cat.append(&k, &v).unwrap();
            let (gk, gv) = grow.append(&k, &v).unwrap();
            for (c, g) in [(ck, gk), (cv, gv)] {
                assert_eq!(c.dims(), g.dims());
                let flat = |t: &Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                assert_eq!(flat(&c), flat(&g), "after an append of {l}");
            }
            let cap = grow.k.as_ref().unwrap().dim(2).unwrap();
            assert!(cap.is_multiple_of(4) && cap >= grow.len && cap < grow.len + 4, "capacity {cap} for {}", grow.len);
        }
        let mut zero = KvCache::with_store(KvMode::F16, KvStore::Grow(0));
        let k = Tensor::zeros((1, 2, 1, 3), DType::F32, &dev).unwrap();
        assert!(format!("{}", zero.append(&k, &k).unwrap_err()).contains("chunk"));
    }

    /// Decoding again after a truncation to the prompt gives the logits of
    /// the first decode, bit for bit, on every store: `LLVQ_GQA_AB` times
    /// several decodes after one prefill on the strength of this.
    #[test]
    fn a_truncated_cache_decodes_like_a_fresh_one() {
        use crate::kvq::{AttnHeads, KvStore};
        let mut m = tiny_model(DType::F32);
        let prompt = [3u32, 17, 42, 7, 9, 11];
        for (store, heads) in [
            (KvStore::Cat, AttnHeads::Expanded),
            (KvStore::Grow(4), AttnHeads::Grouped),
            (KvStore::Prealloc(32), AttnHeads::Expanded),
        ] {
            m.set_kv_store(store);
            m.set_attn_heads(heads);
            let mut c = m.fresh_caches();
            let first = argmax_last(&m.prefill_chunked(&prompt, 4, &mut c).unwrap()).unwrap();
            let decode = |c: &mut [KvCache]| -> Vec<Vec<f32>> {
                let mut next = first;
                (prompt.len()..prompt.len() + 7)
                    .map(|offset| {
                        let l = m.step_logits(next, offset, c).unwrap();
                        next = argmax_last(&l).unwrap();
                        l.flatten_all().unwrap().to_vec1::<f32>().unwrap()
                    })
                    .collect()
            };
            let a = decode(&mut c);
            for cache in c.iter_mut() {
                cache.truncate(prompt.len()).unwrap();
            }
            let b = decode(&mut c);
            assert!(same_bits(&a, &b), "{}: a truncated cache decodes differently", store.label());
            assert!(c[0].truncate(prompt.len() + 100).is_err(), "{}: truncating up must refuse", store.label());
        }
    }

    /// A decode step on the `Cat` store skips its mask, which is all zeros.
    /// This replays each step twice from the same cache state, once through
    /// `hidden_cached` (no mask) and once with the zero mask added explicitly
    /// as before, and requires the hidden states to agree bit for bit.
    #[test]
    fn a_decode_step_without_its_zero_mask_is_bit_identical() {
        for dtype in [DType::F32, DType::F16] {
            let model = tiny_model(dtype);
            let dev = Device::Cpu;
            let prompt = [3u32, 17, 42, 7];
            let input = Tensor::from_slice(&prompt, (1, prompt.len()), &dev).unwrap();
            let mut a = model.fresh_caches();
            let mut b = model.fresh_caches();
            model.hidden_cached(&input, 0, &mut a, &mut NoCapture).unwrap();
            model.hidden_cached(&input, 0, &mut b, &mut NoCapture).unwrap();
            for (offset, tok) in (prompt.len()..).zip([5u32, 60, 11, 90, 2, 33, 71, 8]) {
                assert!(
                    model.step_mask(1, 1, offset).unwrap().is_none(),
                    "the decode step must take the unmasked path, or this test certifies nothing"
                );
                let x = Tensor::from_slice(&[tok], (1, 1), &dev).unwrap();
                let ha = model.hidden_cached(&x, offset, &mut a, &mut NoCapture).unwrap();
                let mask = model.causal_mask_offset(1, 1, offset).unwrap();
                let mut hb = model.embed.forward(&x).unwrap();
                for (i, blk) in model.blocks.iter().enumerate() {
                    hb = blk
                        .forward_cached(&hb, &model.rotary, Some(&mask), RopeAt::Offset(offset), &mut b[i], i, &mut NoCapture)
                        .unwrap();
                }
                let hb = model.norm.forward(&hb).unwrap();
                let bits = |t: &Tensor| -> Vec<u32> {
                    t.flatten_all()
                        .unwrap()
                        .to_dtype(DType::F32)
                        .unwrap()
                        .to_vec1::<f32>()
                        .unwrap()
                        .iter()
                        .map(|v| v.to_bits())
                        .collect()
                };
                assert_eq!(bits(&ha), bits(&hb), "{dtype:?}, step at offset {offset}");
            }
        }
    }

    /// `generate_timed` decodes what `generate` decodes, whole or in chunks,
    /// and the measurement switch of the decode mask moves no token either.
    #[test]
    fn a_timed_generation_returns_the_tokens_of_generate() {
        let mut model = tiny_model(DType::F32);
        let prompt: Vec<u32> = (0..13u32).map(|i| (i * 7 + 3) % 97).collect();
        let want = model.generate(&prompt, 9, &mut NoCapture).unwrap();
        let whole = model.generate_timed(&prompt, 9, MAX_PREFILL_ROWS).unwrap();
        assert_eq!(whole.tokens, want, "one chunk is generate, call for call");
        let chunked = model.generate_timed(&prompt, 9, 4).unwrap();
        assert_eq!(chunked.tokens, want, "a prompt in chunks of 4 decodes the same tokens");
        assert!(whole.prompt_s > 0.0 && whole.decode_s > 0.0);
        model.set_decode_mask(true);
        let masked = model.generate_timed(&prompt, 9, MAX_PREFILL_ROWS).unwrap();
        assert_eq!(masked.tokens, want, "the zero mask, added again, moves no token");
        assert!(model.step_mask(1, 1, 20).unwrap().is_some(), "the switch must reach step_mask");
    }

    /// `prefill_next` in chunks leaves the caches and the first token of an
    /// unchunked prompt, and the decode after it is `generate`'s: what the 4E
    /// driver needs at an 8,192-token context, at chunk sizes a host reaches.
    #[test]
    fn a_chunked_prefill_next_is_the_prefill_of_generate() {
        // 4E's store: Grow(256), whose count `len` tracks.
        let mut model = tiny_model(DType::F32);
        model.set_kv_store(crate::kvq::KvStore::Grow(crate::kvq::GROW_CHUNK));
        let prompt: Vec<u32> = (0..13u32).map(|i| (i * 11 + 5) % 97).collect();
        let want = model.generate(&prompt, 6, &mut NoCapture).unwrap();
        for chunk in [1usize, 4, 12, 13, MAX_PREFILL_ROWS] {
            let mut caches = model.fresh_caches();
            let first = model.prefill_next_in(&prompt, chunk, &mut caches).unwrap();
            assert!(caches.iter().all(|c| c.len() == prompt.len()), "chunk {chunk}: the caches hold the prompt");
            let got = model.decode_greedy(first, prompt.len(), 6, &mut caches).unwrap();
            assert_eq!(got, want, "chunk {chunk}: the tokens of generate");
        }
        let mut caches = model.fresh_caches();
        assert_eq!(model.prefill_next(&prompt, &mut caches).unwrap(), want[0], "the served chunk is one call here");
    }

    /// A CPU lattice projection over a dense weight: rotation key `(0, 0)`,
    /// the identity as its rotation, `k` rows a launch. The dense arithmetic,
    /// through the rotated branch of `group_forward` and its row budget.
    struct HostLattice {
        w: Tensor,
        k: usize,
    }

    impl crate::device::LatticeProj for HostLattice {
        fn name(&self) -> &str {
            "fake.lattice"
        }
        fn d_out(&self) -> usize {
            self.w.dim(0).unwrap()
        }
        fn d_in(&self) -> usize {
            self.w.dim(1).unwrap()
        }
        fn rotation(&self) -> Option<RotKey> {
            Some((0, 0))
        }
        fn prepare(&self, x: &Tensor) -> Result<Tensor> {
            Ok(x.clone())
        }
        fn prepare_rows(&self, xs: &Tensor, _rows: usize) -> Result<Tensor> {
            Ok(xs.clone())
        }
        fn matvec(&self, xr: &Tensor, out_dims: &[usize]) -> Result<Tensor> {
            let y = xr.reshape((1, self.d_in()))?.matmul(&self.w.t()?)?;
            let mut d = out_dims.to_vec();
            *d.last_mut().expect("rank >= 1") = self.d_out();
            y.reshape(d)
        }
        fn rows_per_launch(&self) -> usize {
            self.k
        }
        fn matvec_rows(&self, xr: &Tensor, n: usize) -> Result<Tensor> {
            assert!((1..=self.k).contains(&n), "{n} rows for a launch of {}", self.k);
            xr.matmul(&self.w.t()?)
        }
    }

    /// The tiny model with every `q_proj` served by [`HostLattice`] at `k`
    /// rows a launch, and RoPE tables for 512 positions.
    fn tiny_model_lattice_q(k: usize) -> Qwen3 {
        let mut m = tiny_model_long(DType::F32, 4, 2, 16, 512);
        for b in &mut m.blocks {
            let Proj::Dense(l) = &b.q_proj else { panic!("the tiny model's q_proj is dense") };
            b.q_proj = Proj::Lattice(std::sync::Arc::new(HostLattice { w: l.weight().clone(), k }));
        }
        m
    }

    /// A turn longer than the model takes in one call goes in chunks the
    /// model takes. On the served Metal path a projection loops one launch a
    /// row and refuses more than 256 rows a call; until 2026-11-07
    /// `Session::feed` cut at 4,096, so a 257-token turn failed there.
    #[test]
    fn a_session_feeds_a_long_turn_in_chunks_the_model_takes() {
        let dense = tiny_model_long(DType::F32, 4, 2, 16, 512);
        assert_eq!(dense.prefill_rows(), MAX_PREFILL_ROWS, "nothing rotated: the batched budget");
        assert_eq!(tiny_model_lattice_q(4).prefill_rows(), MAX_PREFILL_ROWS, "four rows a launch");
        let looped = tiny_model_lattice_q(1);
        assert_eq!(looped.prefill_rows(), MAX_ROWS, "one launch a row: the served Metal path");

        let ids: Vec<u32> = (0..300u32).map(|i| (i * 13 + 7) % 97).collect();
        let x = Tensor::from_slice(&ids, (1, ids.len()), &Device::Cpu).unwrap();
        let e = looped
            .hidden_cached(&x, 0, &mut looped.fresh_caches(), &mut NoCapture)
            .expect_err("300 rows in one call of the looped path");
        assert!(e.to_string().contains("more than 256"), "{e}");

        let mut s = crate::session::Session::new(&looped);
        let h = s.feed(&looped, &ids).expect("a 300-token turn on the looped path");
        assert_eq!(s.len(), ids.len(), "every position held");
        let mut d = crate::session::Session::new(&dense);
        let hd = d.feed(&dense, &ids).expect("the dense model takes it in one call");
        let next = |m: &Qwen3, h: &Tensor| argmax_last(&m.logits_last(h).unwrap()).unwrap();
        assert_eq!(next(&looped, &h), next(&dense, &hd), "the dense model's next token");
    }

    /// 🚨 The served config's pair reaches the model, BOTH halves of it.
    ///
    /// `served::Served::apply` is the one place `bin/fusedrun`, `bin/mmlu` and
    /// `bin/gsm8k` set the store and the form, and a half-wired apply is the
    /// defect that would not show: a `Grouped` form on a `Cat` store decodes
    /// the same tokens, slowly, and an `Expanded` form on a `Grow` store
    /// decodes them too. Only the getters tell them apart, so they are
    /// asserted, and then the pair is made to decode so the config cannot name
    /// a combination that falls over.
    #[test]
    fn a_served_config_sets_the_store_and_the_attention_form() {
        use crate::kvq::{AttnHeads, KvStore, GROW_CHUNK};
        let file = |attention: Option<&str>| crate::served::ServedFile {
            layout: "tetra48".into(),
            embed: "q4".into(),
            rot_share: "1".into(),
            fuse: "0".into(),
            kv: "f16".into(),
            attention: attention.map(str::to_string),
            decode: None,
            rows: None,
            prefill_attention: None,
            note: None,
        };
        let resolve = |attention: Option<&str>| {
            crate::served::Served::of(file(attention), std::path::PathBuf::from("t.json"))
                .expect("resolves")
        };

        let mut m = tiny_model(DType::F32);
        assert_eq!(m.attn_heads(), AttnHeads::Expanded, "the model's default");
        assert_eq!(m.kv_store(), KvStore::Cat);
        let prompt: Vec<u32> = (0..9u32).map(|i| (i * 11 + 5) % 97).collect();
        let want = m.generate(&prompt, 7, &mut NoCapture).unwrap();

        // A config that says nothing leaves the published path in place.
        resolve(None).apply(&mut m, &Device::Cpu).expect("expanded applies");
        assert_eq!((m.attn_heads(), m.kv_store()), (AttnHeads::Expanded, KvStore::Cat));

        // A config that says `grouped` moves both halves, and the pair decodes.
        resolve(Some("grouped")).apply(&mut m, &Device::Cpu).expect("grouped applies");
        assert_eq!(m.attn_heads(), AttnHeads::Grouped, "the form");
        assert_eq!(m.kv_store(), KvStore::Grow(GROW_CHUNK), "and the store with it");
        let got = m.generate(&prompt, 7, &mut NoCapture).unwrap();
        assert_eq!(got, want, "on the CPU in f32 the two forms agree token for token");

        // `Served::serve`, the door of `bin/gsm8k` and `bin/chat`: the same
        // pair, then the engine. A grouped config on a fresh model leaves it
        // grouped on `Grow` with the eager engine; an expanded one puts it
        // back. Before 2026-11-06 `bin/chat` printed the provenance and
        // skipped this step, so a grouped config ran the expanded form.
        let mut m = tiny_model(DType::F32);
        let dec = resolve(Some("grouped")).serve(&mut m, &Device::Cpu).expect("grouped serves");
        assert_eq!((m.attn_heads(), m.kv_store()), (AttnHeads::Grouped, KvStore::Grow(GROW_CHUNK)));
        assert!(!dec.is_persistent() && dec.max_positions().is_none(), "the eager engine");
        resolve(None).serve(&mut m, &Device::Cpu).expect("expanded serves");
        assert_eq!((m.attn_heads(), m.kv_store()), (AttnHeads::Expanded, KvStore::Cat));
    }

    /// The `kernel` form on the CPU, where its kernel is the canonical replay
    /// the card is checked against. Against the grouped form on the same
    /// `Grow` store: a prefill chunk of more than 8 rows takes the grouped
    /// path and is bit for bit; chunks and decode steps of 8 rows or fewer go
    /// through the kernel, whose f32 softmax differs from the grouped f16
    /// one, and are held to a bound, about ten times the largest difference
    /// measured on 2026-11-04, and to differing at all.
    #[test]
    fn the_kernel_form_attends_through_the_replay() {
        use crate::kvq::{AttnHeads, KvStore, GROW_CHUNK};
        let mut m = tiny_model_hd(DType::F16, 8, 2, 128);
        let prompt = [3u32, 17, 42, 7, 9, 11, 5, 60, 11, 90, 2];
        let feed = [5u32, 60, 11, 90, 2, 33, 71, 8, 13, 44, 1, 96];
        m.set_kv_store(KvStore::Grow(GROW_CHUNK));
        m.set_attn_heads(AttnHeads::Grouped);
        let grouped_long = logit_trace(&m, &prompt, 16, &feed);
        let grouped_short = logit_trace(&m, &prompt, 4, &feed);

        m.set_attn_heads(AttnHeads::Kernel);
        let mut c = m.fresh_caches();
        let e = m.prefill_chunked(&prompt, 4, &mut c).expect_err("no kernel installed");
        assert!(e.to_string().contains("no kernel installed"), "{e}");

        m.set_attn_kernel(Some(crate::attn::kernel_for(&Device::Cpu, 2, 4, 128).expect("CPU kernel")));
        // At the served threshold no position of this window reaches the
        // kernel: every step is the grouped form's, bit for bit.
        assert_eq!(m.kernel_from(), crate::kvq::KERNEL_FROM);
        assert!(same_bits(&logit_trace(&m, &prompt, 16, &feed), &grouped_long), "below the threshold");
        assert!(same_bits(&logit_trace(&m, &prompt, 4, &feed), &grouped_short), "below the threshold");
        m.set_kernel_from(0);
        let long = logit_trace(&m, &prompt, 16, &feed);
        assert!(
            same_bits(&grouped_long[..1], &long[..1]),
            "a chunk of 11 rows takes the grouped path: its logits are the grouped form's"
        );
        let d_decode = max_abs_diff(&grouped_long[1..], &long[1..]);
        let short = logit_trace(&m, &prompt, 4, &feed);
        let d_short = max_abs_diff(&grouped_short, &short);
        println!("kernel against grouped: decode max |dlogit| {d_decode:e}, chunks of 4 {d_short:e}");
        for (what, d) in [("decode", d_decode), ("chunks of 4", d_short)] {
            assert!(d > 0.0, "{what}: bit for bit with grouped, so the kernel did not run");
            assert!(d <= 5e-3, "{what}: max |dlogit| {d:e}");
        }

        // The served door installs it: the form, the store and the kernel.
        let mut m = tiny_model_hd(DType::F16, 8, 2, 128);
        let served_file_kernel = || crate::served::ServedFile {
            layout: "tetra48".into(),
            embed: "q4".into(),
            rot_share: "1".into(),
            fuse: "0".into(),
            kv: "f16".into(),
            attention: Some("kernel".into()),
            decode: None,
            rows: None,
            prefill_attention: None,
            note: None,
        };
        let f = served_file_kernel();
        let served = crate::served::Served::of(f, "kernel.json".into()).expect("kernel parses");
        served.apply(&mut m, &Device::Cpu).expect("kernel applies on the CPU");
        assert_eq!(m.attn_heads(), AttnHeads::Kernel);
        assert_eq!(m.kv_store(), KvStore::Grow(GROW_CHUNK));
        assert_eq!(m.kernel_from(), crate::kvq::KERNEL_FROM, "the door keeps the form's threshold");
        assert!(served.provenance().contains("attention kernel from position 1280 on grow(256)"));
        m.set_kernel_from(0);
        assert_eq!(logit_trace(&m, &prompt, 4, &feed), short, "the door's kernel is the same arithmetic");
        let e = served.check_device(true).expect_err("the kernel form stays refused on Metal");
        assert!(e.contains("no T8 of the form on Metal"), "{e}");

        // The persistent decode: refused on a model that is not the 4B, before
        // anything moves; past that check, every decode row goes to the
        // kernel (threshold 0), the arithmetic 4E owes its tokens to.
        let p = crate::served::Served::of(
            crate::served::ServedFile { decode: Some("persistent".into()), ..served_file_kernel() },
            "4e.json".into(),
        )
        .expect("4E parses");
        let mut toy = tiny_model_hd(DType::F16, 8, 2, 128);
        let e = p.apply(&mut toy, &Device::Cpu).expect_err("a toy is not the 4B");
        assert!(e.contains("4B"), "{e}");
        assert_eq!(toy.kernel_from(), crate::kvq::KERNEL_FROM, "a refused apply moves nothing");
        p.apply_form(&mut toy, &Device::Cpu).expect("the form applies");
        assert_eq!((toy.attn_heads(), toy.kernel_from()), (AttnHeads::Kernel, 0));
        assert_eq!(logit_trace(&toy, &prompt, 4, &feed), short, "threshold 0: the kernel on every decode row");
        served.apply(&mut toy, &Device::Cpu).expect("back to the eager kernel config");
        assert_eq!(toy.kernel_from(), crate::kvq::KERNEL_FROM, "an eager config restores the constant");
        // f32 is refused at the first decode step, far below the threshold;
        // a prefill chunk takes the grouped path in any dtype.
        let mut wide = tiny_model_hd(DType::F32, 8, 2, 128);
        served.apply(&mut wide, &Device::Cpu).expect("kernel applies on the CPU");
        let mut c = wide.fresh_caches();
        wide.prefill_chunked(&prompt, 16, &mut c).expect("an 11-row chunk is grouped");
        let e = wide.step_logits(feed[0], prompt.len(), &mut c).expect_err("f32 decode");
        assert!(e.to_string().contains("f16 only"), "{e}");
        // A head dimension the kernel does not compile is refused at the door.
        let mut small = tiny_model(DType::F16);
        let e = served.apply(&mut small, &Device::Cpu).expect_err("head dim 16");
        assert!(e.contains("head dim 16"), "{e}");
    }

    /// The T8 harness of the kernel form, on the CPU replay: the decode path
    /// pairs the grouped and the kernel arms in lockstep and scores both, and
    /// the prefill path of the two arms is the same forward (no call of more
    /// than 8 rows reaches the kernel), bit for bit.
    #[test]
    fn the_kernel_arm_runs_in_the_paired_decode() {
        use crate::kvq::{AttnHeads, KvStore, GROW_CHUNK};
        let mut m = tiny_model_hd(DType::F16, 8, 2, 128);
        m.set_attn_kernel(Some(crate::attn::kernel_for(&Device::Cpu, 2, 4, 128).expect("CPU kernel")));
        m.set_kernel_from(8);
        let win = [3u32, 17, 42, 7, 9, 11, 5, 60, 11, 90, 2, 33, 71, 8, 13, 44, 1, 96, 27, 64];
        let g = (KvStore::Grow(GROW_CHUNK), AttnHeads::Grouped);
        let k = (KvStore::Grow(GROW_CHUNK), AttnHeads::Kernel);
        let d = m.decode_nll_split(&win, [g, k], m.kernel_from()).expect("paired decode");
        assert_eq!(d.count, win.len() - 1);
        assert_eq!(d.below_differ, 0, "positions 0..8 are the grouped form's, bit for bit");
        assert_eq!(d.tail_count, win.len() - 1 - 8, "positions 8..19 scored past the split");
        assert!(d.tail_top1_agree <= d.tail_count && d.top1_agree <= d.count);
        assert!(d.max_dlogit > 0.0, "bit for bit: the kernel did not run");
        assert!(d.max_dlogit <= 5e-3, "max |dlogit| {:e}", d.max_dlogit);
        let rel = (d.nll[1] - d.nll[0]).abs() / d.nll[0];
        assert!(rel < 1e-3, "nll {} against {}", d.nll[1], d.nll[0]);
        let tail = (d.tail_nll[1] - d.tail_nll[0]).abs() / d.tail_nll[0];
        assert!(tail > 0.0 && tail < 1e-3, "tail nll {} against {}", d.tail_nll[1], d.tail_nll[0]);
        // The whole difference lies past the split: below it the arms agree.
        assert_eq!((d.nll[1] - d.tail_nll[1]).to_bits(), (d.nll[0] - d.tail_nll[0]).to_bits());
        // A split past the threshold sees the kernel's steps, each one.
        let late = m.decode_nll_split(&win, [g, k], 12).expect("paired decode");
        assert_eq!(late.below_differ, 4, "positions 8, 9, 10 and 11 are the kernel's");
        assert_eq!(late.tail_count, win.len() - 1 - 12);
        // Split 0 is the unsplit call.
        let whole = m.decode_nll_split(&win, [g, k], 0).expect("paired decode");
        assert_eq!(whole, m.decode_nll_paired(&win, [g, k]).expect("paired decode"));
        assert_eq!((whole.tail_nll, whole.tail_count, whole.below_differ), (whole.nll, whole.count, 0));
        let mut pre = |heads| {
            m.set_kv_store(KvStore::Grow(GROW_CHUNK));
            m.set_attn_heads(heads);
            m.window_nll(&win, &mut NoCapture).unwrap().0
        };
        let (a, b) = (pre(AttnHeads::Grouped), pre(AttnHeads::Kernel));
        assert_eq!(a.to_bits(), b.to_bits(), "the prefill path differs between the two arms");
    }

    /// The `Kernel` form's dispatch of one call, on the CPU replay. A row at
    /// the threshold or past it returns the kernel's bits for that row alone,
    /// whatever call carried it (the kernel's rows are invariant); a row
    /// below it is the grouped path's over the history it sees, and not the
    /// kernel's. Every way a call can sit against the threshold: below, at
    /// its first row, straddling it at each row, past it.
    #[test]
    fn a_call_is_split_at_the_threshold_by_position() {
        use crate::attn::canon;
        let kern = crate::attn::kernel_for(&Device::Cpu, 2, 4, 128).expect("CPU kernel");
        let t = |shape: (usize, usize, usize, usize), seed: u64| {
            let n = shape.0 * shape.1 * shape.2 * shape.3;
            let v: Vec<half::f16> =
                canon::fill(n, seed, 2.0).into_iter().map(half::f16::from_bits).collect();
            Tensor::from_vec(v, shape, &Device::Cpu).expect("tensor")
        };
        let (l, n) = (8usize, 20usize);
        let q = t((1, 8, l, 128), 11);
        let (k, v) = (t((1, 2, n, 128), 12), t((1, 2, n, 128), 13));
        let mask = build_causal_mask(1, l, n - l, n, DType::F16, &Device::Cpu).expect("mask");
        let scale = 1.0 / 128f64.sqrt();
        let rows = |x: &Tensor| -> Vec<Vec<u16>> {
            (0..l)
                .map(|r| {
                    let v = x.narrow(2, r, 1).unwrap().flatten_all().unwrap().to_vec1::<half::f16>().unwrap();
                    v.into_iter().map(half::f16::to_bits).collect()
                })
                .collect()
        };
        // Each row through the kernel alone, over the history it sees.
        let alone: Vec<Vec<u16>> = (0..l)
            .map(|r| {
                let seen = n - l + r + 1;
                let o = kern
                    .attend(
                        &q.narrow(2, r, 1).unwrap().contiguous().unwrap(),
                        &k.narrow(2, 0, seen).unwrap(),
                        &v.narrow(2, 0, seen).unwrap(),
                    )
                    .unwrap();
                rows(&o.broadcast_as((1, 8, l, 128)).unwrap().contiguous().unwrap())[0].clone()
            })
            .collect();
        let grouped = grouped_attention(&q, &rows_contiguous(&k).unwrap(), &rows_contiguous(&v).unwrap(), Some(&mask), scale)
            .expect("grouped");
        let grouped_rows = rows(&grouped);
        let h2f = |b: u16| half::f16::from_bits(b).to_f32();
        // Thresholds from all-grouped (20) to all-kernel (12 and below).
        for from in [0usize, 12, 13, 15, 19, 20, 64] {
            let got = rows(&kernel_rows(kern.as_ref(), &q, &k, &v, Some(&mask), scale, from).expect("dispatch"));
            for r in 0..l {
                let pos = n - l + r;
                if pos >= from {
                    assert_eq!(got[r], alone[r], "from {from}: position {pos} is the kernel's, as alone");
                } else {
                    assert_ne!(got[r], alone[r], "from {from}: position {pos} reached the kernel");
                    let d = got[r].iter().zip(&grouped_rows[r]).map(|(a, b)| (h2f(*a) - h2f(*b)).abs()).fold(0.0, f32::max);
                    assert!(d <= 2e-3, "from {from}: position {pos} is not the grouped path's ({d:e})");
                }
            }
            if from >= n {
                assert_eq!(got, grouped_rows, "a call wholly below the threshold is one grouped call");
            }
        }
        // A decode step, no mask: one row on each side.
        let one = q.narrow(2, l - 1, 1).unwrap();
        let at = rows(&kernel_rows(kern.as_ref(), &one, &k, &v, None, scale, n - 1).unwrap().broadcast_as((1, 8, l, 128)).unwrap().contiguous().unwrap());
        assert_eq!(at[0], alone[l - 1], "position n - 1 at threshold n - 1 is the kernel's");
        let below = kernel_rows(kern.as_ref(), &one, &k, &v, None, scale, n).unwrap();
        let g1 = grouped_attention(&one, &rows_contiguous(&k).unwrap(), &rows_contiguous(&v).unwrap(), None, scale).unwrap();
        let bits = |x: &Tensor| x.flatten_all().unwrap().to_vec1::<half::f16>().unwrap();
        assert_eq!(bits(&below), bits(&g1), "position n - 1 at threshold n is the grouped form's decode step");
    }

    /// The served threshold itself, in a model: a prefill of 1,270 positions
    /// (grouped, a chunk of more than 8 rows), then decode steps from 1,270
    /// to 1,289 under the grouped and the kernel forms. Below 1,280 every
    /// step is bit for bit; from 1,280 on, every step differs, within the
    /// bound of the replay test. A threshold one position off, or none, or
    /// read from anything but the position, fails here.
    #[test]
    fn the_served_threshold_is_position_1280() {
        use crate::kvq::{AttnHeads, KvStore, GROW_CHUNK, KERNEL_FROM};
        assert_eq!(KERNEL_FROM, 1280);
        let mut m = tiny_model_long(DType::F16, 8, 2, 128, 1300);
        m.set_attn_kernel(Some(crate::attn::kernel_for(&Device::Cpu, 2, 4, 128).expect("CPU kernel")));
        let mut s = 0x2545f4914f6cdd1du64;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s % 97) as u32
        };
        let prompt: Vec<u32> = (0..1270).map(|_| next()).collect();
        let feed: Vec<u32> = (0..20).map(|_| next()).collect();
        m.set_kv_store(KvStore::Grow(GROW_CHUNK));
        let mut trace = |heads| {
            m.set_attn_heads(heads);
            logit_trace(&m, &prompt, 512, &feed)
        };
        let (g, k) = (trace(AttnHeads::Grouped), trace(AttnHeads::Kernel));
        assert!(same_bits(&g[..1], &k[..1]), "the prefill takes the grouped path");
        for (i, (a, b)) in g[1..].iter().zip(&k[1..]).enumerate() {
            let pos = prompt.len() + i;
            let same = same_bits(std::slice::from_ref(a), std::slice::from_ref(b));
            if pos < KERNEL_FROM {
                assert!(same, "position {pos}: below the threshold, the grouped form's bits");
            } else {
                assert!(!same, "position {pos}: past the threshold, and the kernel did not run");
                let d = max_abs_diff(std::slice::from_ref(a), std::slice::from_ref(b));
                assert!(d <= 5e-3, "position {pos}: max |dlogit| {d:e}");
            }
        }
    }

    /// A two-layer Qwen3 with deterministic pseudo-random weights, on the CPU.
    /// The CPU stand-in of the key kernel: candle's own argmax on the host
    /// device, written into the ring. It pins the plumbing of the `Key` and
    /// `OnDevKey` kinds (the slot, the views, the offsets, the stop before the
    /// last upload, the final readback); the key itself is certified by
    /// `crate::argmax_key` and, on the card, against candle's kernel.
    struct HostKey;

    impl KeyKernel for HostKey {
        fn write(&self, logits: &Tensor, ring: &Tensor, pos: usize) -> Result<()> {
            let id = logits.i((0, 0))?.argmax_keepdim(D::Minus1)?;
            ring.slice_set(&id, 0, pos)
        }
        fn arm(&self) -> Result<()> {
            Ok(())
        }
        fn nan_seen(&self) -> Result<bool> {
            Ok(false)
        }
    }

    /// Every [`TokenLoop`] kind returns `generate`'s tokens, on `Cat`, `Grow`
    /// and `Prealloc`, under the three attention forms, in f32 and f16, with
    /// groups of 2, 4 and 5 (10 heads over 2: the 14B's shape, which no other
    /// test runs). Nine tokens, so `Grow(4)` regrows twice mid-decode.
    #[test]
    fn token_loops_match_generate() {
        use crate::kvq::{AttnHeads, KvStore};
        let key = HostKey;
        let prompt = [3u32, 17, 42, 7];
        for dtype in [DType::F32, DType::F16] {
            for (n_heads, n_kv) in [(4, 2), (8, 2), (10, 2)] {
                let mut m = tiny_model_heads(dtype, n_heads, n_kv);
                for store in [KvStore::Cat, KvStore::Grow(4), KvStore::Prealloc(32)] {
                    for heads in [AttnHeads::Expanded, AttnHeads::Shared, AttnHeads::Grouped] {
                        m.set_kv_store(store);
                        m.set_attn_heads(heads);
                        let want = m.generate(&prompt, 9, &mut NoCapture).unwrap();
                        assert_eq!(want.len(), 9);
                        for (name, kind) in [
                            ("f16", TokenLoop::F16),
                            ("key", TokenLoop::Key(&key)),
                            ("ondev-key", TokenLoop::OnDevKey(&key)),
                            ("ondev-candle", TokenLoop::OnDevCandle),
                        ] {
                            let got = m.generate_loop(&prompt, 9, &kind).unwrap();
                            assert_eq!(got, want, "{name} {dtype:?} {n_heads}/{n_kv} {store:?} {heads:?}");
                            assert!(m.generate_loop(&prompt, 0, &kind).unwrap().is_empty(), "{name}: zero tokens");
                            let one = m.generate_loop(&prompt, 1, &kind).unwrap();
                            assert_eq!(one, want[..1], "{name}: one token");
                        }
                    }
                }
            }
        }
    }

    /// [`HostKey`] that also keeps, widened to f32, every logits row it is
    /// handed.
    struct RecordingKey(std::sync::Mutex<Vec<Vec<f32>>>);

    impl KeyKernel for RecordingKey {
        fn write(&self, logits: &Tensor, ring: &Tensor, pos: usize) -> Result<()> {
            let row = logits.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
            self.0.lock().unwrap().push(row);
            HostKey.write(logits, ring, pos)
        }
        fn arm(&self) -> Result<()> {
            self.0.lock().unwrap().clear();
            Ok(())
        }
        fn nan_seen(&self) -> Result<bool> {
            Ok(false)
        }
    }

    /// The key kinds hand the key kernel, at every step, bit for bit the
    /// logits the step API gives for the same tokens at the same positions.
    /// The tiny model's greedy text barely depends on position (it settles
    /// into a two-token cycle), so `token_loops_match_generate` alone let a
    /// frozen offset and a prefill read at position 0 survive (host mutants
    /// of 2026-11-04); this test kills both.
    #[test]
    fn key_loops_feed_the_reference_logits() {
        use crate::kvq::{AttnHeads, KvStore};
        let prompt = [3u32, 17, 42, 7];
        for dtype in [DType::F32, DType::F16] {
            for (n_heads, n_kv) in [(4, 2), (10, 2)] {
                let mut m = tiny_model_heads(dtype, n_heads, n_kv);
                for (store, heads) in [
                    (KvStore::Cat, AttnHeads::Expanded),
                    (KvStore::Grow(4), AttnHeads::Grouped),
                    (KvStore::Prealloc(32), AttnHeads::Shared),
                ] {
                    m.set_kv_store(store);
                    m.set_attn_heads(heads);
                    for ondev in [false, true] {
                        let rec = RecordingKey(std::sync::Mutex::new(Vec::new()));
                        let kind = if ondev { TokenLoop::OnDevKey(&rec) } else { TokenLoop::Key(&rec) };
                        let tokens = m.generate_loop(&prompt, 9, &kind).unwrap();
                        let seen = rec.0.into_inner().unwrap();
                        let mut c = m.fresh_caches();
                        let mut want = vec![m.prefill_chunked(&prompt, prompt.len(), &mut c).unwrap()];
                        for (offset, &t) in (prompt.len()..).zip(&tokens[..8]) {
                            want.push(m.step_logits(t, offset, &mut c).unwrap());
                        }
                        assert_eq!(seen.len(), 9);
                        for (k, (s, w)) in seen.iter().zip(&want).enumerate() {
                            let w = w.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                            let same = s.len() == w.len() && s.iter().zip(&w).all(|(a, b)| a.to_bits() == b.to_bits());
                            assert!(same, "step {k}, ondev {ondev}, {dtype:?} {n_heads}/{n_kv} {store:?} {heads:?}");
                        }
                    }
                }
            }
        }
    }

    /// The token picked on the f16 logits is the token picked on their f32
    /// cast, on a model whose logits are f16, at every decode step (gesture
    /// a1, on candle's CPU argmax; the card's is one template for both).
    #[test]
    fn argmax_of_f16_logits_is_argmax_of_their_cast() {
        let m = tiny_model(DType::F16);
        let mut caches = m.fresh_caches();
        let input = Tensor::from_slice(&[3u32, 17, 42, 7], (1, 4), &Device::Cpu).unwrap();
        let mut h = m.hidden_cached(&input, 0, &mut caches, &mut NoCapture).unwrap();
        for offset in 4..20 {
            let native = m.logits_last_native(&h).unwrap();
            assert_eq!(native.dtype(), DType::F16);
            let a = argmax_last(&native).unwrap();
            let b = argmax_last(&m.logits_last(&h).unwrap()).unwrap();
            assert_eq!(a, b, "offset {offset}");
            let next = Tensor::from_slice(&[a], (1, 1), &Device::Cpu).unwrap();
            h = m.hidden_cached(&next, offset, &mut caches, &mut NoCapture).unwrap();
        }
    }

    fn tiny_model(dtype: DType) -> Qwen3 {
        tiny_model_heads(dtype, 4, 2)
    }

    /// The tiny model with `n_heads` query heads over `n_kv` key heads.
    fn tiny_model_heads(dtype: DType, n_heads: usize, n_kv: usize) -> Qwen3 {
        tiny_model_hd(dtype, n_heads, n_kv, 16)
    }

    /// The tiny model at a head dimension of its own: the attention kernel
    /// compiles 128, Qwen3's, and refuses anything else.
    fn tiny_model_hd(dtype: DType, n_heads: usize, n_kv: usize, head_dim: usize) -> Qwen3 {
        tiny_model_long(dtype, n_heads, n_kv, head_dim, 64)
    }

    /// The tiny model with RoPE tables for `max_pos` positions.
    fn tiny_model_long(dtype: DType, n_heads: usize, n_kv: usize, head_dim: usize, max_pos: usize) -> Qwen3 {
        use crate::kvq::KvMode;
        use candle_nn::VarBuilder;
        let dev = Device::Cpu;
        let cfg = Config {
            vocab_size: 97,
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 2,
            num_attention_heads: n_heads,
            head_dim,
            attention_bias: false,
            num_key_value_heads: n_kv,
            max_position_embeddings: max_pos,
            sliding_window: None,
            max_window_layers: 0,
            tie_word_embeddings: true,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-6,
            use_sliding_window: false,
            hidden_act: candle_nn::Activation::Silu,
        };
        let varmap = candle_nn::VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, &dev);
        let model = Qwen3::new(&cfg, vb, KvMode::F16).expect("model");
        // Deterministic pseudo-random weights — zeros would make every path
        // agree trivially and certify nothing. The variables are walked in
        // name order: the map's own order changes from one process to the
        // next, and with it every weight.
        let mut state = 0x9e3779b97f4a7c15u64;
        let data = varmap.data().lock().unwrap();
        let mut vars: Vec<_> = data.iter().collect();
        vars.sort_by(|a, b| a.0.cmp(b.0));
        for (_, var) in vars {
            let n = var.elem_count();
            let vals: Vec<f32> = (0..n)
                .map(|_| {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    ((state >> 33) as f32 / (1u64 << 31) as f32 - 1.0) * 0.1
                })
                .collect();
            let t = Tensor::from_vec(vals, var.shape(), &dev)
                .and_then(|t| t.to_dtype(dtype))
                .expect("rand");
            var.set(&t).expect("set");
        }
        model
    }
}

