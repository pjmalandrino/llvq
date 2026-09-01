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
use crate::rotplan::{check_key, drive_rows, RotShare};

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
/// bounded window (2026-09-01, étape 1b).** Under
/// [`crate::kvq::KvStore::Prealloc`] this cache stores the EXPANDED form in
/// a fixed `[b, n_heads, W, hd]` buffer: ×4 of a bounded window is ~151 Mo
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
    /// How the history grows — [`KvStore::Cat`] (shipped) or a preallocated
    /// window. Under `Prealloc`, `k`/`v` hold the FULL fixed buffers and
    /// `len` counts the valid positions; under `Cat` they hold exactly the
    /// history and `len` is unused.
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
        if let crate::kvq::KvStore::Prealloc(w) = self.store {
            return self.append_prealloc(w, &k, &v);
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
    /// graph can capture. 🕳️ The first form of this arm (étape 1, 2026-09-01)
    /// returned `narrow` views instead: non-contiguous at every `len < W`,
    /// they pushed `repeat_kv`'s per-step history copy into its strided
    /// paths — r = 0,8919 [0,8884–0,8953] on CUDA, ~−29 % on Metal. Measured,
    /// then replaced by this form (étape 1b). `generate` and
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
                "KvCache prealloc: {} + {l} positions > fenêtre {w} (LLVQ_KV_PREALLOC) —                  la fenêtre est une borne dure, pas un anneau",
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

    /// Arm the capturable write path — see the `write_idx` field. The stepped
    /// driver calls this AFTER the prefill (whose multi-position write stays
    /// on `slice_set`), with clones of one shared index tensor.
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
    /// One part of a row-concatenated group — q, k or v of one layer, or gate
    /// or up. The whole group is launched **once**, by [`group_forward`], and
    /// the result is narrowed back into the parts; there is no per-projection
    /// launch to reach from here (see [`Proj::forward_with`]'s arm).
    #[cfg(all(target_os = "linux", feature = "cuda"))]
    FusedSeg {
        rt: std::sync::Arc<crate::fused_cuda::FusedRuntime>,
        /// Shared by every projection of the group — `Arc::ptr_eq` on it is how
        /// [`group_forward`] recognises a group, structurally rather than by a
        /// flag that could be set on the wrong three projections.
        group: std::sync::Arc<crate::fused_cuda::FusedSegProj>,
        /// First row of this projection inside the group.
        row0: usize,
        /// This projection's own width. `group.d_out` is the group's.
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
    /// Read only by the fused arm of [`Proj::forward_with`], which exists on
    /// no target this workspace's development machine can build — hence the
    /// `allow` off CUDA. The dense arm deliberately multiplies the caller's
    /// own `x` instead, so that it stays the call `bin/oracle` certifies.
    #[cfg_attr(
        not(all(target_os = "linux", feature = "cuda")),
        allow(dead_code)
    )]
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
            #[cfg(all(target_os = "linux", feature = "cuda"))]
            Proj::Fused { proj, .. } => proj.rotation(),
            #[cfg(all(target_os = "linux", feature = "cuda"))]
            Proj::FusedSeg { group, .. } => group.rotation(),
        }
    }

    /// The name the artifact stores, for error messages. Dense projections
    /// come out of a `VarBuilder` and carry none.
    pub fn site_name(&self) -> &str {
        match self {
            Proj::Dense(_) => "(projection dense)",
            #[cfg(all(target_os = "linux", feature = "cuda"))]
            Proj::Fused { proj, .. } => &proj.name,
            #[cfg(all(target_os = "linux", feature = "cuda"))]
            Proj::FusedSeg { group, rank, .. } => group.part_name(*rank),
        }
    }

    /// Output width.
    pub fn d_out(&self) -> Result<usize> {
        match self {
            Proj::Dense(l) => l.weight().dim(0),
            #[cfg(all(target_os = "linux", feature = "cuda"))]
            Proj::Fused { proj, .. } => Ok(proj.d_out),
            // The **part**, not the group: this is the width of the tensor the
            // caller gets back, which is what every consumer reshapes on.
            #[cfg(all(target_os = "linux", feature = "cuda"))]
            Proj::FusedSeg { d_out, .. } => Ok(*d_out),
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
            #[cfg(all(target_os = "linux", feature = "cuda"))]
            Proj::Fused { rt, proj } => Ok(Rotated {
                key: proj.rotation(),
                t: rt.rotate(proj, x)?,
            }),
            // One `rot_apply` for the whole group: the parts share one `d_in`
            // and one rotation key by construction, checked at load time.
            #[cfg(all(target_os = "linux", feature = "cuda"))]
            Proj::FusedSeg { rt, group, .. } => Ok(Rotated {
                key: group.rotation(),
                t: rt.rotate_group(group, x)?,
            }),
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
            #[cfg(all(target_os = "linux", feature = "cuda"))]
            Proj::Fused { rt, proj } => rt.forward_rotated(proj, &r.t, x.dims()),
            // Deliberately unreachable, and it must stay that way: a launch per
            // projection here would compute the group's whole width and then
            // throw most of it away — silently, and at three times the cost.
            #[cfg(all(target_os = "linux", feature = "cuda"))]
            Proj::FusedSeg { group, .. } => candle_core::bail!(
                "{} appartient à un groupe fusé de {} lignes : il ne se lance qu'à travers \
                 model::group_forward, qui émet UN matvec pour tout le groupe.",
                self.site_name(),
                group.d_out
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
            #[cfg(all(target_os = "linux", feature = "cuda"))]
            Proj::Fused { proj, .. } => panic!(
                "{} est une projection fusée — le quantifieur n'opère que sur un modèle dense",
                proj.name
            ),
            #[cfg(all(target_os = "linux", feature = "cuda"))]
            Proj::FusedSeg { group, rank, .. } => panic!(
                "{} est une projection fusée — le quantifieur n'opère que sur un modèle dense",
                group.part_name(*rank)
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
            #[cfg(all(target_os = "linux", feature = "cuda"))]
            Proj::FusedSeg { group, rank, .. } => panic!(
                "{} est une projection fusée — le quantifieur n'opère que sur un modèle dense",
                group.part_name(*rank)
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
pub fn row_views(x: &Tensor) -> Result<Vec<Tensor>> {
    let dims = x.dims();
    let d_in = *dims.last().expect("rank >= 1");
    let rows: usize = dims[..dims.len() - 1].iter().product();
    let flat = x.contiguous()?.reshape((rows, d_in))?;
    (0..rows).map(|r| flat.narrow(0, r, 1)).collect()
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
            .map(|p| {
                let r = p.prepare(x)?;
                p.forward_with(&r, x)
            })
            .collect();
    }

    // After the dense branch, which must not move, and before the row loop:
    // a fused group replaces the loop's per-projection launches with one.
    #[cfg(all(target_os = "linux", feature = "cuda"))]
    if let Some(plan) = SegPlan::of(projs)? {
        return plan.run(x);
    }

    let dims = x.dims();
    let rows: usize = dims[..dims.len() - 1].iter().product();
    if rows > MAX_ROWS {
        candle_core::bail!(
            "{} : {rows} vecteurs d'un coup, au-delà de {MAX_ROWS}. Le noyau fusé est un \
             matvec — il boucle, donc le coût est linéaire. La passe de score garde le \
             chemin dense.",
            projs[0].site_name()
        );
    }
    let slices = row_views(x)?;

    let per_site = drive_rows(
        share,
        projs,
        rows,
        |p: &&Proj, row: usize| p.prepare(&slices[row]),
        |p: &&Proj, r: &Rotated, row: usize| p.forward_with(r, &slices[row]),
    )?;

    per_site
        .into_iter()
        .zip(projs)
        .map(|(rows_of_site, p)| regroup(&rows_of_site, dims, p.d_out()?))
        .collect()
}

/// One segmented launch, and the map back to the projections it stands for.
///
/// Built by [`Self::of`] from the projections `group_forward` was handed, so
/// the model states the row order **independently** of the loader that assigned
/// it. Two places have to agree and neither may assume: a group read in the
/// order k,q,v would otherwise return finite, plausible, wrong numbers with no
/// assertion anywhere.
#[cfg(all(target_os = "linux", feature = "cuda"))]
struct SegPlan<'a> {
    /// Both handles are borrowed as the `Arc` rather than as its target: the
    /// group's identity test below is then `Arc::ptr_eq`, the only comparison
    /// that cannot be fooled, and the two fields are initialised straight from
    /// the enum's bindings with no coercion in between — which matters in a
    /// file no machine of this workspace compiles.
    rt: &'a std::sync::Arc<crate::fused_cuda::FusedRuntime>,
    group: &'a std::sync::Arc<crate::fused_cuda::FusedSegProj>,
    /// `(rank, row0, d_out)` per projection, in the order the caller handed
    /// them — the shape [`crate::fused::check_seg_spans`] validates.
    spans: Vec<(usize, usize, usize)>,
}

#[cfg(all(target_os = "linux", feature = "cuda"))]
impl<'a> SegPlan<'a> {
    /// `Some` when every projection handed in is a part of **one** group and,
    /// together and in this order, they are the whole group.
    ///
    /// `None` when none of them is segmented — every dense and every unfused
    /// build. `Err` when they are mixed, partial, or out of order: a wiring
    /// bug, never a fallback.
    fn of(projs: &[&'a Proj]) -> Result<Option<Self>> {
        if !projs.iter().any(|p| matches!(p, Proj::FusedSeg { .. })) {
            return Ok(None);
        }
        let mut plan: Option<SegPlan<'a>> = None;
        // `&p` rather than `p`: the item is a `&&Proj`, and destructuring the
        // outer reference is what ties the bindings to the projections'
        // lifetime rather than to this loop's.
        for &p in projs {
            let Proj::FusedSeg { rt, group, row0, d_out, rank } = p else {
                candle_core::bail!(
                    "{} n'appartient à aucun groupe fusé alors que ses voisins de groupe \
                     oui — un lancement segmenté ne peut pas couvrir la moitié d'un site",
                    p.site_name()
                );
            };
            match &mut plan {
                None => {
                    plan = Some(SegPlan {
                        rt,
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
                            "{} appartient à un autre groupe fusé que {} — deux groupes dans \
                             un seul appel",
                            p.site_name(),
                            acc.group.name
                        );
                    }
                    acc.spans.push((*rank, *row0, *d_out));
                }
            }
        }
        let plan = plan.expect("the `any` above found at least one segmented projection");
        crate::fused::check_seg_spans(&plan.group.name, &plan.spans, plan.group.d_out)
            .map_err(candle_core::Error::msg)?;
        Ok(Some(plan))
    }

    /// One rotation and one matvec per row, then one view per projection.
    fn run(&self, x: &Tensor) -> Result<Vec<Tensor>> {
        let dims = x.dims();
        let rows: usize = dims[..dims.len() - 1].iter().product();
        if rows > MAX_ROWS {
            candle_core::bail!(
                "{} : {rows} vecteurs d'un coup, au-delà de {MAX_ROWS}. Le noyau segmenté \
                 reste un matvec — il boucle, donc le coût est linéaire.",
                self.group.name
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
        for row in 0..rows {
            let r = self.rt.rotate_group(self.group, &slices[row])?;
            per_row.push(self.rt.forward_rotated_seg(self.group, &r, slices[row].dims())?);
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
        kv: KvMode,
    ) -> Result<Self> {
        let (nh, nkv, hd) = (
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim,
        );
        let a = vb.pp("self_attn");
        let m = vb.pp("mlp");
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
            rot_share: RotShare::Off,
        })
    }

    /// Whether this block hoists the rotation of a shared activation.
    pub fn set_rot_share(&mut self, share: RotShare) {
        self.rot_share = share;
    }

    pub fn rot_share(&self) -> RotShare {
        self.rot_share
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
        self.forward_cached(x, rotary, mask, RopeAt::Offset(0), &mut fresh, idx, cap)
    }

    /// Forward over `l` new positions starting at absolute position `offset`,
    /// attending over everything in `cache` plus what this call adds.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_cached(
        &self,
        x: &Tensor,
        rotary: &Rotary,
        mask: &Tensor,
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
        let groups = self.n_heads / self.n_kv;
        let (k, v) = match cache.store() {
            crate::kvq::KvStore::Cat => {
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
        // The second shared activation: gate and up both read `h`.
        let gu = group_forward(&[&self.gate_proj, &self.up_proj], &h, self.rot_share)?;
        let gated = gu[0].apply(&self.act)?.mul(&gu[1])?;
        cap.on_activation(idx, Act::MlpOut, &gated)?;
        x + self.down_proj.forward(&gated)?
    }
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
    #[cfg(all(target_os = "linux", feature = "cuda"))]
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
        Ok(Self {
            cfg: cfg.clone(),
            kv,
            kv_store: crate::kvq::KvStore::Cat,
            embed,
            blocks,
            norm: candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("model.norm"))?,
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
        let mask = match self.kv_store {
            crate::kvq::KvStore::Cat => self.causal_mask_offset(b, l, offset)?,
            crate::kvq::KvStore::Prealloc(w) => self.causal_mask_width(b, l, offset, w)?,
        };
        let mut h = self.embed.forward(input)?;
        for (i, blk) in self.blocks.iter().enumerate() {
            h = blk.forward_cached(&h, &self.rotary, &mask, RopeAt::Offset(offset), &mut caches[i], i, cap)?;
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
            crate::kvq::KvStore::Cat => candle_core::bail!(
                "le chemin pas-à-pas exige un store préalloué (LLVQ_KV_PREALLOC) : \
                 ses formes fixes sont ce qui le rend capturable"
            ),
        };
        let (cos, sin) = self.rotary.step_tables()?;
        let st = StepState {
            input: Tensor::zeros((1, 1), DType::U32, &self.device)?,
            mask: Tensor::zeros((1, 1, 1, w), self.dtype, &self.device)?,
            cos,
            sin,
            pos_idx: Tensor::zeros(
                (1, self.cfg.num_key_value_heads * (self.cfg.num_attention_heads / self.cfg.num_key_value_heads), 1, self.cfg.head_dim),
                DType::U32,
                &self.device,
            )?,
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
                &st.mask,
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
                hh = blk.forward_cached(&hh, &self.rotary, &mask, RopeAt::Offset(offset), &mut caches[i], i, &mut NoCapture)?;
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

#[cfg(test)]
mod tests {
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
            .expect_err("une rotation étrangère doit être refusée");
        let msg = format!("{e}");
        assert!(msg.contains("2560"), "le message doit citer la clé : {msg}");
        assert!(
            msg.contains("projection dense"),
            "le message doit nommer le site : {msg}"
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
            "la préparation dense doit rendre le tenseur lui-même, pas une copie"
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
                "dims {dims:?} : le découpage en lignes n'est pas rendu à l'identique"
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
                    "dims {dims:?} : l'ordre des lignes ne change rien, le test ne prouve rien"
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
        let out = group_forward(&[&a], &wide, RotShare::On).expect("dense sur une fenêtre large");
        assert_eq!(out[0].dims(), &[1, MAX_ROWS + 44, 2]);
    }

    /// Étape 1b invariant: the expanded fixed-window store must hold, in its
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
        let (b, n_kv, hd, groups, w) = (1, 2, 64, 4, 16); // hd multiple du GROUP q8
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
                // chemin Cat : append puis repeat_kv — ce que forward_cached fait
                let (ck, cv) = cat.append(&k, &v).expect("cat append");
                let ck = repeat_kv(ck, groups).unwrap().contiguous().unwrap();
                let cv = repeat_kv(cv, groups).unwrap().contiguous().unwrap();
                // chemin Prealloc : étendre PUIS append — ce que forward_cached fait
                let ke = repeat_kv(k.clone(), groups).unwrap().contiguous().unwrap();
                let ve = repeat_kv(v.clone(), groups).unwrap().contiguous().unwrap();
                let (pk, pv) = pre.append(&ke, &ve).expect("prealloc append");
                assert_eq!(pk.dims(), &[b, n_kv * groups, w, hd], "step {step} : buffer plein");
                for (full, reference, name) in [(&pk, &ck, "k"), (&pv, &cv, "v")] {
                    let prefix = full.narrow(2, 0, len).unwrap().contiguous().unwrap();
                    let a: Vec<f32> = reference.flatten_all().unwrap().to_vec1().unwrap();
                    let bb: Vec<f32> = prefix.flatten_all().unwrap().to_vec1().unwrap();
                    assert_eq!(a, bb, "step {step} ({mode:?}) : préfixe {name}");
                    let tail = full.narrow(2, len, w - len).unwrap();
                    let t: Vec<f32> = tail.flatten_all().unwrap().to_vec1().unwrap();
                    assert!(t.iter().all(|x| *x == 0.0), "step {step} ({mode:?}) : queue {name} non nulle");
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
        assert_eq!(attn(&k, &v, len), attn(&kw, &vw, w), "le rembourrage n'est pas inerte");
    }

    #[test]
    fn prealloc_overflow_is_a_named_error() {
        use crate::kvq::{KvMode, KvStore};
        let dev = Device::Cpu;
        let mut c = KvCache::with_store(KvMode::F16, KvStore::Prealloc(4));
        let k = Tensor::zeros((1, 2, 3, 4), DType::F32, &dev).unwrap();
        c.append(&k, &k).expect("3 sur 4 passe");
        let e = c.append(&k, &k).expect_err("3+3 sur 4 doit refuser");
        let msg = format!("{e}");
        assert!(msg.contains("fenêtre"), "message sans mécanisme : {msg}");
    }

    /// THE certifier of the stepped path: on a tiny random model, the three
    /// decodes — `Cat`, `Prealloc` (eager), and `Prealloc` stepped (stable
    /// buffers + scatter write, the capturable closure) — must return the
    /// SAME tokens. A per-token variation missing from [`StepState`]'s
    /// inventory (mask, RoPE, input, write position) diverges here, on CPU,
    /// for free — not on a billed card.
    #[test]
    fn stepped_decode_matches_generate_on_all_stores() {
        use crate::kvq::{KvMode, KvStore};
        use candle_nn::VarBuilder;
        let dev = Device::Cpu;
        let cfg = Config {
            vocab_size: 97,
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            head_dim: 16,
            attention_bias: false,
            num_key_value_heads: 2,
            max_position_embeddings: 64,
            sliding_window: None,
            max_window_layers: 0,
            tie_word_embeddings: true,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-6,
            use_sliding_window: false,
            hidden_act: candle_nn::Activation::Silu,
        };
        let varmap = candle_nn::VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
        let mut model = Qwen3::new(&cfg, vb, KvMode::F16).expect("model");
        // Deterministic pseudo-random weights — zeros would make every path
        // agree trivially and certify nothing.
        let mut state = 0x9e3779b97f4a7c15u64;
        for (_, var) in varmap.data().lock().unwrap().iter() {
            let n = var.elem_count();
            let vals: Vec<f32> = (0..n)
                .map(|_| {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    ((state >> 33) as f32 / (1u64 << 31) as f32 - 1.0) * 0.1
                })
                .collect();
            let t = Tensor::from_vec(vals, var.shape(), &dev).expect("rand");
            var.set(&t).expect("set");
        }
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
        assert_eq!(cat, pre, "cat contre prealloc éager");
        assert_eq!(pre, stepped, "prealloc éager contre pas-à-pas (StepState incomplet ?)");
    }
}
