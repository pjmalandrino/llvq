//! Inkling 2 Mini (`thinkingmachines/Inkling-2-Mini` @ ed40e7da), the text
//! model in Rust, read from the native checkpoint (NVFP4 experts in layers 0
//! to 19, MXFP8 in 20 to 23), in f32.
//!
//! The reference is transformers 046bdb8cc2,
//! `models/inkling2_loop/modeling_inkling2_loop.py`; every piece cites its
//! line. [`Inkling`] is the whole text model: the embedding, the pre-loop
//! layers 0 to 5, the loop layers 6 to 17 run three times with messages
//! between passes, the post-loop layers 18 to 23, `final_input`, the norm and
//! the output head with its softcap. It implements [`crate::lm::CausalLm`]
//! on [`InklingCache`], one [`SlotCache`] per layer visit (48).
//!
//! The MoE of each layer goes through one seam, [`layers::MoeSource`], chosen
//! per layer by [`LayerExperts`]: the native experts decoded one at a time
//! from the mapping (the default), the dense reconstructions of the lattice
//! codes ([`resident::HatExperts`], a reference arm), or the lattice codes
//! resident on the Metal device ([`resident::ResidentExperts`]).
//!
//! `tests/inkling_oracle.rs` compares the hidden states against f32 dumps of
//! the reference (`ops/inkling/dump_ref.py`).

pub mod attn_dense;
pub mod attn_scores;
pub mod config;
pub mod expert_codes;
pub mod experts_cpu;
pub mod glue;
pub mod head;
pub mod layers;
pub mod loader;
pub mod moe_act;
pub mod tables;
#[cfg(all(target_os = "macos", feature = "metal"))]
pub mod resident;
pub mod snapshot;
#[cfg(ink_glue)]
pub mod attn_bias;
#[cfg(all(target_os = "linux", feature = "cuda"))]
pub mod resident_cuda;

use std::path::Path;

use candle_core::{DType, Device, Result, Tensor};

use config::InklingConfig;
use layers::{block_attn_res, LayerWeights, MoeSource, Routing, SlotCache, SlotMark};
use loader::NativeCheckpoint;

/// The embedding output and each pre-loop layer's output and routing.
pub struct PrefixOut {
    pub embed: Tensor,
    pub layers: Vec<Tensor>,
    pub routing: Vec<Routing>,
}

/// The pre-loop part of the model: embedding and layers `0..core_start`
/// (the first slice, kept for its oracle on 64 tokens).
pub struct InklingPrefix {
    pub cfg: InklingConfig,
    ck: NativeCheckpoint,
    layers: Vec<LayerWeights>,
    dev: Device,
}

impl InklingPrefix {
    /// Map `checkpoint` and load the dense weights of layers `0..n_layers`
    /// (`n_layers <= core_start`). Experts stay in the mapping and are
    /// decoded one at a time during the forward.
    pub fn load(config: &Path, checkpoint: &Path, n_layers: usize, dev: &Device) -> anyhow::Result<Self> {
        let cfg = InklingConfig::from_file(config)?;
        anyhow::ensure!(
            n_layers <= cfg.loop_config.core_start,
            "{n_layers} layers asked, the pre-loop part has {}",
            cfg.loop_config.core_start
        );
        let ck = NativeCheckpoint::open(checkpoint)?;
        let layers = (0..n_layers)
            .map(|l| LayerWeights::load(&ck, &cfg, l, dev))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { cfg, ck, layers, dev: dev.clone() })
    }

    /// `Inkling2LoopTextModel.forward`, lines 1359-1376, on fresh caches: the
    /// embedding, then each loaded layer, the layer input appended to the
    /// block states after a boundary layer.
    pub fn forward(&self, ids: &[u32]) -> Result<PrefixOut> {
        let embed = layers::embed(&self.ck, &self.cfg, ids, &self.dev)?;
        let mut h = embed.clone();
        let mut block_states: Vec<Tensor> = vec![];
        let mut out = PrefixOut { embed, layers: vec![], routing: vec![] };
        for lw in &self.layers {
            let experts = self.ck.experts(lw.index)?;
            let (next, routing) =
                lw.forward(&self.cfg, &h, &block_states, None, &experts, &mut SlotCache::default())?;
            if lw.boundary {
                block_states.push(h.clone());
            }
            h = next;
            out.layers.push(h.clone());
            out.routing.push(routing);
        }
        Ok(out)
    }

    pub fn checkpoint(&self) -> &NativeCheckpoint {
        &self.ck
    }
}

/// One layer visit of a forward: the layer, its pass (0 outside the loop, 1
/// to 3 inside), the hidden state after it and the routing of its MoE.
pub struct Visit {
    pub layer: usize,
    pub pass: usize,
    pub hidden: Tensor,
    pub routing: Routing,
}

/// What [`Inkling::forward`] returns.
pub struct ForwardOut {
    /// The normed embedding, `[T, H]`.
    pub embed: Tensor,
    /// The 48 visits in execution order; empty unless asked for.
    pub visits: Vec<Visit>,
    /// Softcapped logits, `[T, vocab]` (or `[1, vocab]` for the last
    /// position only), over the unpadded vocabulary.
    pub logits: Tensor,
}

/// Per-visit caches of one sequence (48 slots), and the tokens they hold.
#[derive(Debug, Clone)]
pub struct InklingCache {
    slots: Vec<SlotCache>,
    len: usize,
    tokens: Vec<u32>,
}

/// What [`crate::lm::CausalLm::mark`] keeps for [`Inkling`]: the tokens held
/// and, per slot, the positions and the four convolution states.
#[derive(Debug, Clone)]
pub struct InklingMark {
    tokens: Vec<u32>,
    slots: Vec<SlotMark>,
}

impl InklingMark {
    /// Tokens held at the mark.
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }
}

impl InklingCache {
    /// Tokens held.
    pub fn len(&self) -> usize {
        self.len
    }

    /// The token ids held, in order.
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    /// Make room for `positions` tokens in every slot before the next growth
    /// of its key and value buffers. Nothing held changes. A bounded sliding
    /// slot ([`Inkling::set_kv_window`]) ignores it.
    pub fn reserve(&mut self, positions: usize) -> Result<()> {
        self.slots.iter_mut().try_for_each(|s| s.reserve(positions))
    }

    /// Positions the key and value buffers of every slot are allocated for, summed.
    pub fn allocated_positions(&self) -> usize {
        self.slots.iter().map(SlotCache::capacity).sum()
    }

    /// The state needed to come back here.
    pub fn mark(&self) -> InklingMark {
        InklingMark { tokens: self.tokens.clone(), slots: self.slots.iter().map(SlotCache::mark).collect() }
    }

    /// Cut the cache back to `m`. Refused, with the cache unchanged, when the
    /// tokens of `m` are not a prefix of the tokens the cache holds.
    pub fn rollback(&mut self, m: &InklingMark) -> Result<()> {
        if m.slots.len() != self.slots.len() {
            candle_core::bail!("rollback: a mark of {} slots on a cache of {}", m.slots.len(), self.slots.len());
        }
        let n = m.tokens.len();
        if n > self.len || self.tokens[..n] != m.tokens[..] {
            candle_core::bail!("rollback: the {n} tokens of the mark are not a prefix of the {} the cache holds", self.len);
        }
        if let Some(s) = self.slots.iter().find(|s| s.len() < n) {
            candle_core::bail!("rollback: a slot holds {} positions, the mark {n}", s.len());
        }
        // Every slot accepts before any moves: a bounded sliding slot refuses a rollback deeper than its slack.
        for (i, (s, sm)) in self.slots.iter().zip(&m.slots).enumerate() {
            s.check_rollback(sm).map_err(|e| candle_core::Error::msg(format!("rollback: slot {i}: {e}")))?;
        }
        for (s, sm) in self.slots.iter_mut().zip(&m.slots) {
            s.rollback(sm)?;
        }
        self.len = n;
        self.tokens.truncate(n);
        Ok(())
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Write the whole state in the format of [`snapshot`]: the tokens held,
    /// then every slot. [`Inkling::read_cache`] makes a cache that holds the
    /// same. Nothing changes.
    pub fn write_state(&self, w: &mut dyn std::io::Write) -> Result<()> {
        use snapshot::put_u64;
        w.write_all(snapshot::MAGIC).map_err(|e| candle_core::Error::msg(format!("cache state: {e}")))?;
        put_u64(w, snapshot::VERSION)?;
        put_u64(w, self.slots.len() as u64)?;
        put_u64(w, self.len as u64)?;
        put_u64(w, self.tokens.len() as u64)?;
        let toks: Vec<u8> = self.tokens.iter().flat_map(|t| t.to_le_bytes()).collect();
        w.write_all(&toks).map_err(|e| candle_core::Error::msg(format!("cache state: {e}")))?;
        self.slots.iter().try_for_each(|s| s.write_state(w))
    }

    /// The slot of every visit, for inspection.
    pub fn slots(&self) -> &[SlotCache] {
        &self.slots
    }
}

/// Where the experts of one layer come from.
pub enum LayerExperts {
    /// The native checkpoint, one expert decoded at a time (NVFP4 or MXFP8).
    Native,
    /// Any other source, owned: the dense reconstructions
    /// (`resident::HatExperts`) or the resident lattice codes
    /// (`resident::ResidentExperts`).
    Source(Box<dyn MoeSource>),
}

/// The whole Inkling 2 Mini text model. Dense weights resident on the
/// device (about 3 GB in f32, the output head in BF16), experts in the
/// mapping.
pub struct Inkling {
    pub cfg: InklingConfig,
    ck: NativeCheckpoint,
    layers: Vec<LayerWeights>,
    /// `message_input.query`, `[4, H]` per loop layer: query, key, value, MLP.
    message_queries: Vec<Tensor>,
    /// `final_input.query` (`attn_res.output_query`), `[1, H]`.
    final_query: Tensor,
    norm: Tensor,
    /// `embed_norm.weight` in f32 on the device, read once at load: the RMS
    /// norm after the embedding row used to read, widen and upload it from
    /// the mapping on every forward call.
    embed_norm: Tensor,
    /// `unembed.weight` rows `0..unpadded_vocab_size`, BF16, `[V, H]`;
    /// `None` when [`LoadOptions::int4_head`] chose the int4 head at load.
    unembed: Option<Tensor>,
    dev: Device,
    /// One entry per layer; [`LayerExperts::Native`] unless set.
    experts: Vec<LayerExperts>,
    /// When set, the wall time of every MoE call, `(layer, seconds)`, the
    /// device synchronized on both sides. Off by default.
    moe_timing: std::sync::Mutex<Option<Vec<(usize, f64)>>>,
    /// The output projection, when set: `head` takes its logits instead of
    /// the BF16 `unembed` rows (plan item D4). `None` unless set.
    pub out_proj: Option<Box<dyn head::OutputProjection>>,
    /// The embedding rows of the sealed file, when set: `forward` takes its rows instead of the
    /// checkpoint's BF16 ones ([`tables`]). `None` unless set.
    pub embed_table: Option<tables::EmbedTable>,
    /// Query rows of one attention score tensor ([`layers::DEFAULT_ATTN_CHUNK`]; `0`: one chunk).
    attn_chunk: usize,
    /// Tokens of a prompt one call of [`Self::forward`] reads ([`Self::set_prefill_piece`]; `0`: the whole prompt).
    prefill_piece: usize,
    /// The slack of the sliding slots of a new cache, when they are bounded ([`Self::set_kv_window`]).
    kv_slack: Option<usize>,
    /// Called after every layer visit when set: a measurement reads the device memory there.
    visit_probe: std::sync::Mutex<Option<VisitProbe>>,
    /// The fused glue kernels, when [`Self::set_glue`] chose them: a one-row
    /// forward then runs [`layers::LayerWeights::forward_fused`].
    #[cfg(ink_glue)]
    glue: Option<std::sync::Arc<glue::Glue>>,
    /// The attention chosen at load ([`Self::load_with_attn`]).
    attn: AttnKind,
    /// Whether a fused one-row visit hands its routing to the MoE on the
    /// device ([`Self::set_routing_on_device`]); on by default.
    routing_on_device: bool,
}

/// The small steps of a one-row visit: candle's operations (the default) or
/// the fused glue kernels of [`glue`] (Metal only, class B).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GlueKind {
    Candle,
    Fused,
}

impl GlueKind {
    pub fn name(self) -> &'static str {
        match self {
            GlueKind::Candle => "candle",
            GlueKind::Fused => "fused",
        }
    }
}

impl std::str::FromStr for GlueKind {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> anyhow::Result<Self> {
        match s {
            "candle" => Ok(GlueKind::Candle),
            "fused" => Ok(GlueKind::Fused),
            o => anyhow::bail!("glue {o}: candle or fused"),
        }
    }
}

/// The attention products of a one-row visit: candle's operations on f32
/// matrices (the default), the kernels of [`attn_dense`] on the BF16
/// matrices as the file stores them (Metal and CUDA, class B), or the CPU
/// kernels of `tetra_cpu` on those BF16 matrices held on the host
/// ([`AttnKind::Cpu`]). Chosen at load ([`Inkling::load_with_attn`]):
/// it decides which copy of the matrices the device holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AttnKind {
    #[default]
    Candle,
    Fused,
    /// [`AttnKind::Fused`], and the scores of the row over the cache through
    /// the one launch of [`attn_scores`] (class B too).
    FusedScores,
    /// The seven attention matrices in BF16 on the host, their products
    /// through `tetra_cpu::bf16_matvec` for one row and a few, widened for
    /// candle's matmul on a longer prompt ([`layers::CpuAttnProj`]). The CPU
    /// only; the same products as [`AttnKind::Candle`] to rounding.
    Cpu,
}

impl AttnKind {
    pub fn name(self) -> &'static str {
        match self {
            AttnKind::Candle => "candle",
            AttnKind::Fused => "fused",
            AttnKind::FusedScores => "fused+scores",
            AttnKind::Cpu => "cpu",
        }
    }
}

impl std::str::FromStr for AttnKind {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> anyhow::Result<Self> {
        match s {
            "candle" => Ok(AttnKind::Candle),
            "fused" => Ok(AttnKind::Fused),
            "fused+scores" => Ok(AttnKind::FusedScores),
            "cpu" => Ok(AttnKind::Cpu),
            o => anyhow::bail!("attn {o}: candle, fused, fused+scores or cpu"),
        }
    }
}

/// What [`Inkling::load_with`] does beyond the dense weights.
#[derive(Debug, Clone, Default)]
pub struct LoadOptions {
    /// The int4 g64 output table (`bin/inktable`'s `unembed.q4g64.llvqraw`):
    /// the head runs through [`head::Q4MetalHead`] and the BF16 `unembed`
    /// rows are never put on the device (plan item D4). Metal, CUDA, or the
    /// CPU ([`head::Q4CpuHead`]). `None`:
    /// the BF16 head, as [`Inkling::load`].
    pub int4_head: Option<std::path::PathBuf>,
}

/// Prompt rows one launch of the resident lattice kernel takes in the
/// Inkling runtime ([`resident_runtime`]): the f32 rows kernel of plan item
/// P1, each row bit for bit the one-row matvec's.
pub const DEFAULT_PREFILL_ROWS: usize = 8;

static KERNEL_LAUNCHES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// One more launch of a kernel of this module's CUDA path (glue, attention,
/// experts, head): what a decode step is bound by on a fast card.
pub fn count_launch() {
    KERNEL_LAUNCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// The launches counted since the last call.
pub fn take_kernel_launches() -> u64 {
    KERNEL_LAUNCHES.swap(0, std::sync::atomic::Ordering::Relaxed)
}

static KERNEL_PROFILE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static KERNEL_TIMES: std::sync::Mutex<std::collections::BTreeMap<&'static str, (u64, f64)>> =
    std::sync::Mutex::new(std::collections::BTreeMap::new());

/// Time every kernel of the CUDA path on its own: the stream is drained before
/// and after each launch, so a step under this switch is not a step's time.
pub fn set_kernel_profile(on: bool) {
    KERNEL_PROFILE.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// Drain the stream (`sync`) and start the clock, under [`set_kernel_profile`].
pub fn probe_begin(sync: impl FnOnce() -> bool) -> Option<std::time::Instant> {
    if !KERNEL_PROFILE.load(std::sync::atomic::Ordering::Relaxed) {
        return None;
    }
    sync();
    Some(std::time::Instant::now())
}

/// Drain the stream and charge the time since [`probe_begin`] to `name`.
pub fn probe_end(probe: Option<std::time::Instant>, name: &'static str, sync: impl FnOnce() -> bool) {
    let Some(t) = probe else { return };
    sync();
    let s = t.elapsed().as_secs_f64();
    let mut m = KERNEL_TIMES.lock().expect("kernel times");
    let e = m.entry(name).or_insert((0, 0.0));
    e.0 += 1;
    e.1 += s;
}

/// The launches and seconds charged to each kernel since the last call.
pub fn take_kernel_times() -> Vec<(&'static str, u64, f64)> {
    let mut m = KERNEL_TIMES.lock().expect("kernel times");
    let v = m.iter().map(|(k, (n, s))| (*k, *n, *s)).collect();
    m.clear();
    v
}

/// Turn cudarc's event tracking off on the device's context: two events an
/// allocation and one a use, which order nothing while the context holds one
/// stream, candle's (the reasons are at the served door of
/// `fused_cuda::load_resolved`). Call it before the first upload: a slice
/// created while tracking was on keeps recording. Refused on several streams
/// and on another device.
#[cfg(all(target_os = "linux", feature = "cuda"))]
pub fn cuda_events_off(dev: &Device) -> Result<()> {
    let ctx = dev.as_cuda_device()?.cuda_stream().context().clone();
    if ctx.is_in_multi_stream_mode() {
        candle_core::bail!("events off: the context holds several streams, and that is only safe on one");
    }
    // Safe: one stream, checked above, orders every access by issue.
    unsafe { ctx.disable_event_tracking() };
    Ok(())
}

#[cfg(not(all(target_os = "linux", feature = "cuda")))]
pub fn cuda_events_off(_: &Device) -> Result<()> {
    candle_core::bail!("events off: a CUDA switch, on a CUDA build")
}

/// Plan item K2: compute encoders candle puts in one Metal command buffer on
/// the device Inkling is served on (candle's default is 50). 30 shortens a
/// decode step by 1.88 %, median of 5 alternated rounds [1.31, 3.52], logits
/// byte for byte; every value from 5 to 1,000 reads within the spread of one
/// run (`docs/mesures/inkling-mac-decode-2026-11-26.txt`).
pub const METAL_COMPUTE_PER_BUFFER: &str = "30";

/// The Metal device Inkling is served on, with [`METAL_COMPUTE_PER_BUFFER`].
/// candle reads `CANDLE_METAL_COMPUTE_PER_BUFFER` once, when the device is
/// made; a value already in the environment wins, and an unset variable is
/// unset again afterwards, so no other device of the process sees it. Call it
/// before the process starts reading its environment from other threads.
pub fn new_metal_device() -> Result<Device> {
    const VAR: &str = "CANDLE_METAL_COMPUTE_PER_BUFFER";
    let unset = std::env::var_os(VAR).is_none();
    if unset {
        std::env::set_var(VAR, METAL_COMPUTE_PER_BUFFER);
    }
    let dev = Device::new_metal(0);
    if unset {
        std::env::remove_var(VAR);
    }
    dev
}

/// The runtime the resident experts launch through: the served tables and
/// tile, with `prefill_rows` rows a prompt launch (1: one matvec a row, the
/// path before P1; 2, 4 or 8: the f32 rows kernel).
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn resident_runtime(dev: &Device, prefill_rows: usize) -> Result<std::sync::Arc<crate::fused_metal::MetalRuntime>> {
    let rt = resident::served_runtime(dev)?;
    if prefill_rows == 1 {
        return Ok(rt);
    }
    let mut rt = std::sync::Arc::try_unwrap(rt)
        .map_err(|_| candle_core::Error::msg("served_runtime handed a shared runtime"))?;
    let t = rt.tuning();
    rt.set_tuning(crate::fused_metal::MetalTuning { prefill_rows, ..t })?;
    Ok(std::sync::Arc::new(rt))
}

/// What [`Inkling::set_visit_probe`] takes.
pub type VisitProbe = Box<dyn FnMut() + Send>;

/// A source whose calls are timed, the device drained before and after.
struct TimedMoe<'a> {
    inner: &'a dyn MoeSource,
    dev: &'a Device,
    layer: usize,
    out: &'a std::sync::Mutex<Option<Vec<(usize, f64)>>>,
}

impl MoeSource for TimedMoe<'_> {
    fn moe(&self, x: &Tensor, routing: &Routing, cfg: &InklingConfig) -> Result<Tensor> {
        self.dev.synchronize()?;
        let t = std::time::Instant::now();
        let y = self.inner.moe(x, routing, cfg)?;
        self.dev.synchronize()?;
        if let Some(v) = self.out.lock().expect("no panic holds this lock").as_mut() {
            v.push((self.layer, t.elapsed().as_secs_f64()));
        }
        Ok(y)
    }

    fn moe_routed_on_device(
        &self,
        x: &Tensor,
        sel: &Tensor,
        experts: usize,
        cfg: &InklingConfig,
    ) -> Result<Option<Tensor>> {
        self.dev.synchronize()?;
        let t = std::time::Instant::now();
        let y = self.inner.moe_routed_on_device(x, sel, experts, cfg)?;
        self.dev.synchronize()?;
        if y.is_some() {
            if let Some(v) = self.out.lock().expect("no panic holds this lock").as_mut() {
                v.push((self.layer, t.elapsed().as_secs_f64()));
            }
        }
        Ok(y)
    }

    fn take_launches(&self) -> usize {
        self.inner.take_launches()
    }
}

/// `(start, length)` of the calls a prompt of `t` tokens is read in, at most
/// `piece` tokens a call (`0`: one call). No piece holds a single token
/// unless the prompt does: a one-row call takes the decode path (the fused
/// glue and attention kernels where they are on), so a lone last token joins
/// the piece before it, which then holds `piece + 1`.
pub fn prefill_pieces(t: usize, piece: usize) -> Vec<(usize, usize)> {
    if piece == 0 || t <= piece {
        return vec![(0, t)];
    }
    let mut out = vec![];
    let mut start = 0;
    while start < t {
        let mut n = piece.min(t - start);
        if t - start - n == 1 {
            n += 1;
        }
        out.push((start, n));
        start += n;
    }
    out
}

/// Output-head rows widened to f32 per matmul: bounds the f32 copy to
/// `HEAD_CHUNK * H * 4` bytes (151 MB) whatever the vocabulary.
const HEAD_CHUNK: usize = 16384;

impl Inkling {
    /// Map `checkpoint` and load every dense weight onto `dev`.
    pub fn load(config: &Path, checkpoint: &Path, dev: &Device) -> anyhow::Result<Self> {
        Self::load_with(config, checkpoint, dev, &LoadOptions::default())
    }

    /// [`Self::load`] with `opts`: the int4 head in place of the BF16 one,
    /// whose rows are then never put on the device.
    pub fn load_with(config: &Path, checkpoint: &Path, dev: &Device, opts: &LoadOptions) -> anyhow::Result<Self> {
        Self::load_with_attn(config, checkpoint, dev, opts, AttnKind::Candle)
    }

    /// [`Self::load_with`] with the attention `attn`. [`AttnKind::Fused`]:
    /// every layer's attention matrices are read in BF16 into the kernels'
    /// buffers and held there only; a one-row forward runs the kernels, a
    /// several-row one widens a layer's matrices for its call. Metal only.
    pub fn load_with_attn(
        config: &Path,
        checkpoint: &Path,
        dev: &Device,
        opts: &LoadOptions,
        attn: AttnKind,
    ) -> anyhow::Result<Self> {
        let cfg = InklingConfig::from_file(config)?;
        let ck = NativeCheckpoint::open(checkpoint)?;
        let layers = Self::load_layers(&ck, &cfg, dev, attn)?;
        let n_loop = cfg.loop_config.core_end - cfg.loop_config.core_start;
        // Inkling2LoopMergeMessageQueries (core_model_loading.py 823-853): the file holds
        // [query, mlp] and [key, value] per loop layer; merged as [query, key, value, mlp].
        let qm = ck.f32("middle_cycle_cross_pass_queries", dev)?;
        let kv = ck.f32("middle_cycle_cross_pass_kv_queries", dev)?;
        anyhow::ensure!(
            qm.dims2()? == (2 * n_loop, cfg.hidden_size) && kv.dims2()? == (2 * n_loop, cfg.hidden_size),
            "message queries: {:?} and {:?}, expected [{}, {}]",
            qm.dims(),
            kv.dims(),
            2 * n_loop,
            cfg.hidden_size
        );
        let message_queries = (0..n_loop)
            .map(|i| {
                Tensor::cat(
                    &[qm.narrow(0, 2 * i, 1)?, kv.narrow(0, 2 * i, 2)?, qm.narrow(0, 2 * i + 1, 1)?],
                    0,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let final_query = ck.f32("attn_res.output_query", dev)?.unsqueeze(0)?;
        let norm = ck.f32("norm.weight", dev)?;
        let embed_norm = ck.f32("embed_norm.weight", dev)?;
        // Inkling2UnpadVocab: the first `unpadded_vocab_size` rows (lm_head, line 1477)
        let (unembed, out_proj) = match &opts.int4_head {
            None => {
                let unembed = ck.raw("unembed.weight", dev)?;
                anyhow::ensure!(unembed.dtype() == DType::BF16, "unembed.weight: {:?}, expected BF16", unembed.dtype());
                (Some(unembed.narrow(0, 0, cfg.unpadded_vocab_size)?), None)
            }
            Some(path) => (None, Some(Self::int4_head(path, &cfg, dev)?)),
        };
        let experts = (0..cfg.num_hidden_layers).map(|_| LayerExperts::Native).collect();
        Ok(Self {
            cfg,
            ck,
            layers,
            message_queries,
            final_query,
            norm,
            embed_norm,
            unembed,
            dev: dev.clone(),
            experts,
            moe_timing: std::sync::Mutex::new(None),
            out_proj,
            embed_table: None,
            attn_chunk: layers::DEFAULT_ATTN_CHUNK,
            prefill_piece: 0,
            kv_slack: None,
            visit_probe: std::sync::Mutex::new(None),
            #[cfg(ink_glue)]
            glue: None,
            attn,
            routing_on_device: true,
        })
    }

    /// Every layer's dense weights, the attention as `attn` asks.
    fn load_layers(ck: &NativeCheckpoint, cfg: &InklingConfig, dev: &Device, attn: AttnKind) -> anyhow::Result<Vec<LayerWeights>> {
        let n = cfg.num_hidden_layers;
        match attn {
            AttnKind::Candle => Ok((0..n).map(|l| LayerWeights::load(ck, cfg, l, dev)).collect::<Result<Vec<_>>>()?),
            AttnKind::Cpu => {
                anyhow::ensure!(dev.is_cpu(), "the CPU attention runs on the CPU, the model is on {dev:?}");
                Ok((0..n).map(|l| LayerWeights::load_cpu(ck, cfg, l)).collect::<Result<Vec<_>>>()?)
            }
            #[cfg(ink_glue)]
            AttnKind::Fused | AttnKind::FusedScores => {
                anyhow::ensure!(
                    dev.is_metal() || dev.is_cuda(),
                    "the fused attention runs on Metal and on CUDA, the model is on {dev:?}"
                );
                let k = std::sync::Arc::new(attn_dense::AttnDense::new(dev)?);
                let row = match attn {
                    AttnKind::FusedScores => Some(std::sync::Arc::new(attn_scores::AttnRow::new(dev)?)),
                    _ => None,
                };
                Ok((0..n).map(|l| LayerWeights::load_dense(ck, cfg, l, dev, &k, row.as_ref())).collect::<Result<Vec<_>>>()?)
            }
            #[cfg(not(ink_glue))]
            AttnKind::Fused | AttnKind::FusedScores => {
                anyhow::bail!("the fused attention needs a macOS build with metal or a Linux build with cuda")
            }
        }
    }

    /// The attention the model was loaded with.
    pub fn attn(&self) -> AttnKind {
        self.attn
    }

    /// Bytes of the seven attention matrices of every layer on the device:
    /// (as f32 tensors, as BF16: the GPU kernels' buffers or the CPU path's
    /// host matrices).
    pub fn attention_matrix_bytes(&self) -> (usize, usize) {
        let mut f32_bytes = 0;
        #[allow(unused_mut)]
        let mut bf16_bytes = 0;
        for lw in &self.layers {
            let a = &lw.attn;
            for t in [&a.wq, &a.wr, &a.sink, &a.gate, &a.wk, &a.wv, &a.wo] {
                if t.device().same_device(&self.dev) && t.dtype() == DType::F32 {
                    f32_bytes += 4 * t.elem_count();
                }
            }
            #[cfg(ink_glue)]
            if let Some(d) = &a.dense {
                bf16_bytes += d.w.matrix_bytes();
            }
            if let Some(c) = &a.cpu {
                bf16_bytes += c.bytes() as usize;
            }
        }
        (f32_bytes, bf16_bytes)
    }

    /// The int4 head of `path` on `dev`: Metal, CUDA or the CPU.
    #[allow(unused_variables)]
    fn int4_head(path: &Path, cfg: &InklingConfig, dev: &Device) -> anyhow::Result<Box<dyn head::OutputProjection>> {
        if dev.is_cpu() {
            return Ok(Box::new(head::Q4CpuHead::load(path, cfg.unpadded_vocab_size, cfg.hidden_size)?));
        }
        #[cfg(all(target_os = "macos", feature = "metal"))]
        if dev.is_metal() {
            let rt = resident::served_runtime(dev)?;
            let h = head::Q4MetalHead::load(rt, path, cfg.unpadded_vocab_size, cfg.hidden_size)?;
            return Ok(Box::new(h));
        }
        #[cfg(all(target_os = "linux", feature = "cuda"))]
        if dev.is_cuda() {
            let rt = resident_cuda::served_runtime(dev, cfg)?;
            let h = head::Q4CudaHead::load(rt, path, cfg.unpadded_vocab_size, cfg.hidden_size)?;
            return Ok(Box::new(h));
        }
        anyhow::bail!("the int4 head runs on Metal, CUDA and the CPU; {} asked on {dev:?}", path.display())
    }

    /// Whether the BF16 output rows are on the device.
    pub fn has_bf16_head(&self) -> bool {
        self.unembed.is_some()
    }

    /// Run the small steps of every one-row forward from now on through
    /// `kind`. [`GlueKind::Fused`] compiles the kernels of [`glue`] and is
    /// refused off Metal. A prompt (several rows) always runs candle's.
    pub fn set_glue(&mut self, kind: GlueKind) -> Result<()> {
        match kind {
            GlueKind::Candle => {
                #[cfg(ink_glue)]
                {
                    self.glue = None;
                }
                Ok(())
            }
            #[cfg(ink_glue)]
            GlueKind::Fused => {
                if !(self.dev.is_metal() || self.dev.is_cuda()) {
                    candle_core::bail!("the fused glue runs on Metal and on CUDA, the model is on {:?}", self.dev);
                }
                self.glue = Some(std::sync::Arc::new(glue::Glue::new(&self.dev)?));
                Ok(())
            }
            #[cfg(not(ink_glue))]
            GlueKind::Fused => candle_core::bail!("the fused glue needs a macOS build with metal or a Linux build with cuda"),
        }
    }

    /// With the fused glue, hand the routing of a one-row visit to the MoE
    /// on the device instead of reading it back to the host (plan item D9,
    /// its first half): the same ids and weights reach the same kernels, so
    /// the logits are the same bits, and the host waits on nothing inside a
    /// visit. On by default; a forward that records its visits reads the
    /// routing back whatever this says. Off is the path before it, kept as
    /// the reference of its gate.
    pub fn set_routing_on_device(&mut self, on: bool) {
        self.routing_on_device = on;
    }

    /// Tell every layer's expert source to run the activation of its fused
    /// one-row path in one launch (`true`, the default of a resident layer)
    /// or through candle's operations, the reference of that launch's gate.
    pub fn set_moe_act(&mut self, on: bool) {
        for e in self.experts.iter_mut() {
            if let LayerExperts::Source(s) = e {
                s.set_fused_act(on);
            }
        }
    }

    /// Tell every layer's expert source to take the routed experts of one
    /// row in four launches (`true`, the default where the source has them)
    /// or one expert at a time, the reference of their gate.
    pub fn set_moe_four_launches(&mut self, on: bool) {
        for e in self.experts.iter_mut() {
            if let LayerExperts::Source(s) = e {
                s.set_four_launches(on);
            }
        }
    }

    /// Tell every layer's expert source to take a prompt's rows in one launch
    /// a step for the whole layer (`true`) or one expert and one chunk at a
    /// time (`false`, the default and the reference of the gate).
    pub fn set_moe_batch_rows(&mut self, on: bool) {
        for e in self.experts.iter_mut() {
            if let LayerExperts::Source(s) = e {
                s.set_batch_rows(on);
            }
        }
    }

    /// Whether [`Self::set_routing_on_device`] is on.
    pub fn routing_on_device(&self) -> bool {
        self.routing_on_device
    }

    /// The glue a one-row forward runs.
    pub fn glue(&self) -> GlueKind {
        #[cfg(ink_glue)]
        if self.glue.is_some() {
            return GlueKind::Fused;
        }
        GlueKind::Candle
    }

    pub fn device(&self) -> &Device {
        &self.dev
    }

    pub fn checkpoint(&self) -> &NativeCheckpoint {
        &self.ck
    }

    /// `(layer, pass)` of every visit in execution order (lines 1362-1440).
    pub fn schedule(&self) -> Vec<(usize, usize)> {
        let lc = &self.cfg.loop_config;
        let mut v: Vec<(usize, usize)> = (0..lc.core_start).map(|l| (l, 0)).collect();
        for p in 1..=lc.num_passes {
            v.extend((lc.core_start..lc.core_end).map(|l| (l, p)));
        }
        v.extend((lc.core_end..self.cfg.num_hidden_layers).map(|l| (l, 0)));
        v
    }

    /// The cache slot of visit `(layer, pass)`: each pass of a loop layer has
    /// its own (lines 1354-1357 and 1404-1405).
    pub fn slot(&self, layer: usize, pass: usize) -> usize {
        let lc = &self.cfg.loop_config;
        let n_loop = lc.core_end - lc.core_start;
        if layer < lc.core_start {
            layer
        } else if layer < lc.core_end {
            lc.core_start + (pass - 1) * n_loop + (layer - lc.core_start)
        } else {
            lc.core_start + lc.num_passes * n_loop + (layer - lc.core_end)
        }
    }

    pub fn n_slots(&self) -> usize {
        let lc = &self.cfg.loop_config;
        self.cfg.num_hidden_layers + (lc.num_passes - 1) * (lc.core_end - lc.core_start)
    }

    /// An empty cache, its sliding slots bounded when [`Self::set_kv_window`] asked.
    pub fn new_cache(&self) -> InklingCache {
        let mut slots = vec![SlotCache::default(); self.n_slots()];
        if let Some(slack) = self.kv_slack {
            for (layer, pass) in self.schedule() {
                if self.cfg.is_sliding(layer) {
                    slots[self.slot(layer, pass)]
                        .bound(self.cfg.sliding_window_size, slack)
                        .expect("a new slot is empty and the window is not 0");
                }
            }
        }
        InklingCache { slots, len: 0, tokens: vec![] }
    }

    /// A cache that holds what [`InklingCache::write_state`] wrote, its slots
    /// on this model's device. Refused by name, with nothing returned, when
    /// the state is not this model's: another number of slots, a bound that
    /// is not the one [`Self::set_kv_window`] gives a new cache, a damaged
    /// record. The reads after it give the logits the reads after the saved
    /// cache gave, bit for bit.
    pub fn read_cache(&self, r: &mut dyn std::io::Read) -> Result<InklingCache> {
        use snapshot::{get_u64, get_usize};
        let mut magic = [0u8; 8];
        r.read_exact(&mut magic).map_err(|e| candle_core::Error::msg(format!("cache state: {e}")))?;
        if &magic != snapshot::MAGIC {
            candle_core::bail!("cache state: not a cache state (magic {magic:?})");
        }
        let version = get_u64(r)?;
        if version != snapshot::VERSION {
            candle_core::bail!("cache state: version {version}, this build reads {}", snapshot::VERSION);
        }
        let mut c = self.new_cache();
        let n_slots = get_usize(r, "slots")?;
        if n_slots != c.slots.len() {
            candle_core::bail!("cache state: {n_slots} slots, the model has {}", c.slots.len());
        }
        let len = get_usize(r, "len")?;
        let n = get_usize(r, "tokens")?;
        if n != len {
            candle_core::bail!("cache state: {n} tokens for {len} positions");
        }
        let mut toks = vec![0u8; n.checked_mul(4).ok_or_else(|| candle_core::Error::msg("cache state: tokens"))?];
        r.read_exact(&mut toks).map_err(|e| candle_core::Error::msg(format!("cache state: {e}")))?;
        c.tokens = toks.chunks_exact(4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
        for (i, s) in c.slots.iter_mut().enumerate() {
            s.read_state(r, &self.dev).map_err(|e| candle_core::Error::msg(format!("slot {i}: {e}")))?;
            if s.len() != len {
                candle_core::bail!("cache state: slot {i} holds {} positions, the cache {len}", s.len());
            }
        }
        c.len = len;
        Ok(c)
    }

    /// Bound the sliding slots of every cache made from now on
    /// (`Some(slack)`; [`layers::DEFAULT_KV_SLACK`] is the default slack), or
    /// keep every position in every slot (`None`, the default). A bounded
    /// slot keeps the last `window + slack` positions it has seen, in buffers
    /// of `window + 2 * slack`, and drops its oldest when they are full; the
    /// full slots keep every position. Its scores read the same visible keys
    /// at the same distances. A rollback deeper than `slack` positions is
    /// refused by name. Values: the same products over the visible keys; the
    /// candle path's softmax and value product then run over a shorter row,
    /// so they are not meant to be bit for bit those of a full slot; the
    /// one-row score kernel reads the same keys and is.
    pub fn set_kv_window(&mut self, slack: Option<usize>) {
        self.kv_slack = slack;
    }

    pub fn kv_window(&self) -> Option<usize> {
        self.kv_slack
    }

    /// The native experts of `layer`, decoded one expert at a time.
    pub fn experts(&self, layer: usize) -> Result<loader::NativeExperts<'_>> {
        self.ck.experts(layer)
    }

    /// Take the experts of `layer` from `src` from now on.
    pub fn set_experts(&mut self, layer: usize, src: LayerExperts) {
        self.experts[layer] = src;
    }

    /// The layers whose experts are not the native ones.
    pub fn substituted_layers(&self) -> Vec<usize> {
        (0..self.experts.len()).filter(|&l| !matches!(self.experts[l], LayerExperts::Native)).collect()
    }

    /// Start (`true`) or stop timing every MoE call. Timing drains the
    /// device around each call, so it slows the forward; the results do not
    /// change.
    pub fn set_moe_timing(&self, on: bool) {
        *self.moe_timing.lock().expect("no panic holds this lock") = if on { Some(vec![]) } else { None };
    }

    /// The `(layer, seconds)` of every MoE call since the last take.
    pub fn take_moe_timing(&self) -> Vec<(usize, f64)> {
        self.moe_timing.lock().expect("no panic holds this lock").as_mut().map(std::mem::take).unwrap_or_default()
    }

    /// Take the attention scores of a prompt `rows` query rows at a time
    /// (`0`: the whole prompt in one score tensor). Default
    /// [`layers::DEFAULT_ATTN_CHUNK`]. A prompt no longer than `rows` runs the
    /// operations of one call.
    pub fn set_attn_chunk(&mut self, rows: usize) {
        self.attn_chunk = rows;
    }

    pub fn attn_chunk(&self) -> usize {
        self.attn_chunk
    }

    /// Read a prompt `tokens` at a time ([`Self::forward_pieces`], and so
    /// [`crate::lm::CausalLm::prefill`]); `0`, the default, reads it in one
    /// call. Every transient of a prompt call is then bounded by the piece:
    /// the hidden-wide copies, the relative logits `[heads, T, extent]`, the
    /// rows the experts take. The cache holds the same positions either way.
    /// The logits are not meant to be bit for bit those of one call: the
    /// score products run over other lengths and the experts batch other rows.
    pub fn set_prefill_piece(&mut self, tokens: usize) {
        self.prefill_piece = tokens;
    }

    pub fn prefill_piece(&self) -> usize {
        self.prefill_piece
    }

    /// `ids` through [`Self::forward`] in the pieces of [`prefill_pieces`],
    /// each one appended to `cache` before the next is read. The logits of
    /// the last position, `[1, vocab]`, or of every position, `[T, vocab]`.
    pub fn forward_pieces(&self, ids: &[u32], cache: &mut InklingCache, all_logits: bool) -> Result<Tensor> {
        let pieces = prefill_pieces(ids.len(), self.prefill_piece);
        let mut logits = Vec::with_capacity(if all_logits { pieces.len() } else { 1 });
        for (start, n) in pieces {
            let out = self.forward(&ids[start..start + n], cache, false, all_logits)?;
            if !all_logits {
                logits.clear();
            }
            logits.push(out.logits);
        }
        if logits.len() == 1 {
            return Ok(logits.pop().expect("one piece"));
        }
        Tensor::cat(&logits, 0)
    }

    /// Call `probe` after every layer visit from now on (`None`: stop). It
    /// sees nothing of the forward; a measurement drains the device and reads
    /// its memory there.
    pub fn set_visit_probe(&self, probe: Option<VisitProbe>) {
        *self.visit_probe.lock().expect("no panic holds this lock") = probe;
    }

    /// The source set for `layer`, if any.
    pub fn layer_source(&self, layer: usize) -> Option<&dyn MoeSource> {
        match &self.experts[layer] {
            LayerExperts::Native => None,
            LayerExperts::Source(s) => Some(s.as_ref()),
        }
    }

    /// `Inkling2LoopTextModel.forward` (lines 1350-1443) then the head of
    /// `Inkling2LoopForCausalLM` (lines 1477-1478): `ids` at positions
    /// `cache.len()..`, appended to `cache`. `record` keeps every visit;
    /// `all_logits` computes the logits of every position, else the last.
    /// The experts of each layer are those [`Self::set_experts`] chose.
    pub fn forward(&self, ids: &[u32], cache: &mut InklingCache, record: bool, all_logits: bool) -> Result<ForwardOut> {
        if cache.slots.len() != self.n_slots() {
            candle_core::bail!("cache has {} slots, the model {}", cache.slots.len(), self.n_slots());
        }
        let lc = &self.cfg.loop_config;
        layers::phases::mark(&self.dev, layers::phases::Phase::Other)?;
        // `layers::embed` on either rows, with the weight kept from the load.
        let rows = match &self.embed_table {
            Some(t) => t.rows(ids, &self.dev)?,
            None => self.ck.embed_rows(ids, &self.dev)?,
        };
        let embed = layers::rms_norm(&rows, &self.embed_norm, self.cfg.rms_norm_eps)?;
        layers::phases::mark(&self.dev, layers::phases::Phase::Embed)?;
        let mut visits = vec![];
        let mut h = embed.clone();
        let mut bs: Vec<Tensor> = vec![]; // block states, line 1360
        let mut pre: Option<(Tensor, Vec<Tensor>)> = None;
        let mut prev: Option<(Vec<Tensor>, Tensor)> = None;
        for (layer, pass) in self.schedule() {
            let lw = &self.layers[layer];
            if pass >= 2 && layer == lc.core_start {
                // lines 1398-1402: the previous pass's states past the first loop block feed the messages,
                // and this pass restarts from the pre-loop states
                let (pre_h, pre_bs) = pre.clone().expect("pre-loop states");
                prev = Some((bs[pre_bs.len() + 1..].to_vec(), h.clone()));
                h = pre_h;
                bs = pre_bs;
            }
            let messages = if pass >= 2 {
                let (pbs, ph) = prev.as_ref().expect("previous pass");
                // line 1407: `message_input`, the four queries of this loop layer
                let queries = &self.message_queries[layer - lc.core_start];
                #[cfg(ink_glue)]
                let m = match &self.glue {
                    // One launch for the four queries, with the fused glue's sums (class B, as `pre`).
                    Some(g) if ids.len() == 1 && pbs.len() <= glue::MAX_STATES => {
                        let states = pbs.iter().map(|s| s.contiguous()).collect::<Result<Vec<_>>>()?;
                        let out = g.messages(&queries.contiguous()?, &states, &ph.contiguous()?)?;
                        (0..queries.dim(0)?).map(|i| out.narrow(0, i, 1)).collect::<Result<Vec<_>>>()?
                    }
                    _ => block_attn_res(queries, pbs, ph)?,
                };
                #[cfg(not(ink_glue))]
                let m = block_attn_res(queries, pbs, ph)?;
                layers::phases::mark(&self.dev, layers::phases::Phase::NormsGates)?;
                Some(m)
            } else {
                None
            };
            let native;
            let src: &dyn MoeSource = match &self.experts[layer] {
                LayerExperts::Native => {
                    native = self.ck.experts(layer)?;
                    &native
                }
                LayerExperts::Source(s) => s.as_ref(),
            };
            let timing_on = self.moe_timing.lock().expect("no panic holds this lock").is_some();
            let timed = TimedMoe { inner: src, dev: &self.dev, layer, out: &self.moe_timing };
            let src: &dyn MoeSource = if timing_on { &timed } else { src };
            let slot = &mut cache.slots[self.slot(layer, pass)];
            #[cfg(ink_glue)]
            let (next, routing) = match &self.glue {
                Some(g) if ids.len() == 1 => {
                    let want_routing = record || !self.routing_on_device;
                    lw.forward_fused(&self.cfg, &h, &bs, messages.as_deref(), src, slot, g, want_routing)?
                }
                _ => lw.forward_chunked(&self.cfg, &h, &bs, messages.as_deref(), src, slot, self.attn_chunk)?,
            };
            #[cfg(not(ink_glue))]
            let (next, routing) =
                lw.forward_chunked(&self.cfg, &h, &bs, messages.as_deref(), src, slot, self.attn_chunk)?;
            if let Some(p) = self.visit_probe.lock().expect("no panic holds this lock").as_mut() {
                p();
            }
            if lw.boundary {
                bs.push(h.clone());
            }
            h = next;
            if pass == 0 && layer + 1 == lc.core_start {
                pre = Some((h.clone(), bs.clone())); // line 1378
            }
            if record {
                visits.push(Visit { layer, pass, hidden: h.clone(), routing });
            }
        }
        cache.len += ids.len();
        cache.tokens.extend_from_slice(ids);
        // line 1442-1443: `final_input` over the block states, then the norm
        let h = block_attn_res(&self.final_query, &bs, &h)?.remove(0);
        let h = layers::rms_norm(&h, &self.norm, self.cfg.rms_norm_eps)?;
        let h = if all_logits { h } else { h.narrow(0, h.dim(0)? - 1, 1)? };
        layers::phases::mark(&self.dev, layers::phases::Phase::NormsGates)?;
        let logits = self.head(&h)?;
        layers::phases::mark(&self.dev, layers::phases::Phase::Head)?;
        Ok(ForwardOut { embed, visits, logits })
    }

    /// `lm_head` then the softcap (lines 1477-1478 and 751-757), the BF16
    /// rows widened to f32 a chunk at a time, or the logits of `out_proj`
    /// when it is set.
    fn head(&self, h: &Tensor) -> Result<Tensor> {
        if let Some(p) = &self.out_proj {
            let logits = p.project(h)?;
            let cap = self.cfg.final_logit_softcapping;
            return (logits / cap)?.tanh()? * cap;
        }
        let Some(unembed) = &self.unembed else {
            candle_core::bail!("no output head: the BF16 rows were not loaded and no projection is set")
        };
        let v = unembed.dim(0)?;
        let mut parts = vec![];
        let mut start = 0;
        while start < v {
            let n = HEAD_CHUNK.min(v - start);
            let w = unembed.narrow(0, start, n)?.to_dtype(DType::F32)?;
            parts.push(layers::linear(h, &w)?);
            start += n;
        }
        let logits = Tensor::cat(&parts, 1)?;
        let cap = self.cfg.final_logit_softcapping;
        (logits / cap)?.tanh()? * cap
    }
}

#[cfg(all(target_os = "macos", feature = "metal"))]
impl Inkling {
    /// Make resident on the device every layer of `layers_dir` (`D/layers`)
    /// whose `VERIFIED` file exists; the others keep their native experts.
    /// Returns the resident layers and what each upload cost. The model must
    /// live on the Metal device of `rt`.
    pub fn make_resident(
        &mut self,
        layers_dir: &Path,
        rt: std::sync::Arc<crate::fused_metal::MetalRuntime>,
    ) -> Result<Vec<(usize, resident::UploadStats)>> {
        let mut out = vec![];
        for l in resident::verified_layers(layers_dir, self.cfg.num_hidden_layers) {
            let dir = layers_dir.join(format!("L{l:02}"));
            let r = resident::ResidentExperts::load(rt.clone(), &self.dev, &dir, l, &self.cfg)?;
            out.push((l, r.stats.clone()));
            self.experts[l] = LayerExperts::Source(Box::new(r));
        }
        Ok(out)
    }

    /// Take the experts of each layer of `layers` from the dense
    /// reconstructions beside its codes (`Lxx/w13_hat`, `Lxx/w2_hat`).
    pub fn use_reconstructions(&mut self, layers_dir: &Path, layers: &[usize]) -> Result<()> {
        for &l in layers {
            let h = resident::HatExperts::open(&layers_dir.join(format!("L{l:02}")), &self.cfg)?;
            self.experts[l] = LayerExperts::Source(Box::new(h));
        }
        Ok(())
    }
}

impl Inkling {
    /// Lattice-kernel launches of every substituted layer since the last
    /// call, and reset.
    pub fn take_launches(&self) -> usize {
        self.experts
            .iter()
            .map(|e| match e {
                LayerExperts::Native => 0,
                LayerExperts::Source(s) => s.take_launches(),
            })
            .sum()
    }
}

impl crate::lm::CausalLm for Inkling {
    type Cache = InklingCache;
    type Mark = InklingMark;

    // The logits have the unpadded rows only (`Inkling2UnpadVocab`); the
    // config's `vocab_size` is the padded embedding, 201,024.
    #[allow(clippy::misnamed_getters)]
    fn vocab_size(&self) -> usize {
        self.cfg.unpadded_vocab_size
    }

    fn new_cache(&self) -> Self::Cache {
        Inkling::new_cache(self)
    }

    fn prefill(&self, tokens: &[u32], cache: &mut Self::Cache) -> Result<Tensor> {
        let logits = self.forward_pieces(tokens, cache, false)?;
        logits.reshape((1, 1, self.cfg.unpadded_vocab_size))
    }

    fn step(&self, token: u32, offset: usize, cache: &mut Self::Cache) -> Result<Tensor> {
        if offset != cache.len() {
            candle_core::bail!("step at offset {offset}, the cache holds {} tokens", cache.len());
        }
        let out = self.forward(&[token], cache, false, false)?;
        out.logits.reshape((1, 1, self.cfg.unpadded_vocab_size))
    }

    fn mark(&self, cache: &Self::Cache) -> Self::Mark {
        cache.mark()
    }

    fn rollback(&self, cache: &mut Self::Cache, mark: &Self::Mark) -> Result<()> {
        cache.rollback(mark)
    }
}

#[cfg(test)]
mod tests {
    //! A tiny whole model, the one of `tests/inkling_wave1.rs`: 12 layers, the
    //! loop of layers 6 to 11 run three times (24 visits), a sliding window of
    //! 4 and a relative extent of 6, every layer on dense experts.

    use super::*;
    use crate::lm::CausalLm;
    use layers::ExpertSource;

    /// splitmix64, mapped to `[-a, a)`.
    struct Gen(u64);

    impl Gen {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }

        fn uniform(&mut self, a: f32) -> f32 {
            let u = (self.next() >> 40) as f32 / (1u64 << 24) as f32;
            (2.0 * u - 1.0) * a
        }

        fn tensor(&mut self, shape: &[usize], a: f32) -> Tensor {
            let n: usize = shape.iter().product();
            let v: Vec<f32> = (0..n).map(|_| self.uniform(a)).collect();
            Tensor::from_vec(v, shape, &Device::Cpu).expect("host tensor")
        }

        fn norm(&mut self, n: usize) -> Tensor {
            let v: Vec<f32> = (0..n).map(|_| 1.0 + self.uniform(0.1)).collect();
            Tensor::from_vec(v, n, &Device::Cpu).expect("host tensor")
        }
    }

    struct DenseExperts(Vec<(Tensor, Tensor)>);

    impl ExpertSource for DenseExperts {
        fn count(&self) -> usize {
            self.0.len()
        }
        fn expert(&self, e: usize, _dev: &Device) -> Result<(Tensor, Tensor)> {
            Ok(self.0[e].clone())
        }
    }

    /// The tiny model on the CPU, written to a directory of its own and loaded by [`Inkling::load`].
    fn tiny_model() -> Inkling {
        static CALL: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = CALL.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("inkling_mod_tiny_{}_{n}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let cfg_path = dir.join("config.json");
        let json = serde_json::json!({
            "hidden_size": 48, "num_hidden_layers": 12, "vocab_size": 64, "unpadded_vocab_size": 60,
            "num_attention_heads": 6, "num_key_value_heads": 2, "head_dim": 8,
            "swa_head_dim": 8, "swa_num_attention_heads": 6, "swa_num_key_value_heads": 2,
            "sliding_window_size": 4, "d_rel": 4, "rel_extent": 6,
            "log_scaling_n_floor": 16, "log_scaling_alpha": 0.1, "rms_norm_eps": 1e-6,
            "qk_l2_clip_threshold": 22.6, "rel_l2_clip_threshold": 8.0, "sconv_kernel_size": 4,
            "final_logit_softcapping": 30.0, "n_layers_per_attn_res_block": 6,
            "n_routed_experts": 8, "num_experts_per_tok": 2, "intermediate_size": 32,
            "mlp_gate_softcap": 4.0, "mlp_up_softcap": 25.0, "route_scale": 2.5,
            "loop_config": {"core_start": 6, "core_end": 12, "num_passes": 3, "gate_rank": 4}
        });
        std::fs::write(&cfg_path, json.to_string()).unwrap();
        let cfg = InklingConfig::from_file(&cfg_path).unwrap();
        let mut g = Gen(0x5eed_1a71);
        let (h, k, rank) = (cfg.hidden_size, cfg.sconv_kernel_size, cfg.loop_config.gate_rank);
        let mut t: Vec<(String, Tensor)> = vec![];
        let mut put = |name: String, x: Tensor| t.push((format!("model.llm.{name}"), x));
        for l in 0..cfg.num_hidden_layers {
            let p = |s: &str| format!("layers.{l}.{s}");
            let sliding = cfg.is_sliding(l);
            let (nh, nkv, hd) = cfg.attn_shape(l);
            let ext = cfg.layer_rel_extent(l);
            let wa = 1.0 / (h as f32).sqrt();
            put(p("attn.wq_du.weight"), g.tensor(&[nh * hd, h], wa));
            put(p("attn.q_l2_clip.scale"), g.tensor(&[nh, hd], 1.0));
            put(p("attn.wk_dv.weight"), g.tensor(&[nkv * hd, h], wa));
            put(p("attn.wv_dv.weight"), g.tensor(&[nkv * hd, h], wa));
            for which in ["k", "v"] {
                if sliding {
                    put(p(&format!("attn.{which}_sconv.weight")), g.tensor(&[nkv * hd, 1, k], 0.5));
                } else {
                    put(p(&format!("attn.{which}_sconv.static_w")), g.tensor(&[k, nkv * hd], 0.5));
                    put(p(&format!("attn.{which}_sconv.weight_proj")), g.tensor(&[nkv * k, h], wa));
                }
            }
            put(p("attn.gate_attn.weight"), g.tensor(&[nh * hd, h], wa));
            put(p("attn.wo_ud.weight"), g.tensor(&[h, nh * hd], 1.0 / ((nh * hd) as f32).sqrt()));
            put(p("attn.attn_sink_proj.weight"), g.tensor(&[nh, h], wa));
            put(p("attn.wr_du.weight"), g.tensor(&[nh * cfg.d_rel, h], wa));
            put(p("attn.rel_l2_clip.scale"), g.tensor(&[nh, cfg.d_rel], 1.0));
            put(p("attn.rel_logits_proj.proj"), g.tensor(&[cfg.d_rel, ext], 1.0));
            if l > 0 {
                put(format!("attn_res.attn_queries.{}", l - 1), g.tensor(&[if sliding { 1 } else { 3 }, h], 1.0));
            }
            if sliding {
                put(p("attn_norm.weight"), g.norm(h));
            } else {
                for n in ["q_input_norm", "k_input_norm", "v_input_norm"] {
                    put(p(&format!("{n}.weight")), g.norm(h));
                }
            }
            if cfg.is_loop_layer(l) {
                for which in ["attn", "k", "v", "mlp"] {
                    put(p(&format!("controllers.{which}.down_weight")), g.tensor(&[rank, 2 * h], 0.2));
                    put(p(&format!("controllers.{which}.forget_weight")), g.tensor(&[h, rank], 0.3));
                    put(p(&format!("controllers.{which}.gain_weight")), g.tensor(&[h, rank], 0.3));
                }
            }
            put(format!("attn_res.mlp_queries.{l}"), g.tensor(&[h], 1.0));
            put(p("mlp_norm.weight"), g.norm(h));
            put(p("attn_sconv.weight"), g.tensor(&[h, 1, k], 0.5));
            put(p("mlp_sconv.weight"), g.tensor(&[h, 1, k], 0.5));
            put(p("mlp.gate.weight"), g.tensor(&[cfg.n_routed_experts, h], wa));
            put(p("mlp.gate.bias"), g.tensor(&[cfg.n_routed_experts], 0.05));
        }
        let n_loop = cfg.loop_config.core_end - cfg.loop_config.core_start;
        put("middle_cycle_cross_pass_queries".into(), g.tensor(&[2 * n_loop, h], 1.0));
        put("middle_cycle_cross_pass_kv_queries".into(), g.tensor(&[2 * n_loop, h], 1.0));
        put("attn_res.output_query".into(), g.tensor(&[h], 1.0));
        put("norm.weight".into(), g.norm(h));
        put("embed_norm.weight".into(), g.norm(h));
        put("embed.weight".into(), g.tensor(&[cfg.vocab_size, h], 1.0).to_dtype(DType::BF16).unwrap());
        put("unembed.weight".into(), g.tensor(&[cfg.vocab_size, h], 1.0).to_dtype(DType::BF16).unwrap());
        let ck_path = dir.join("model.safetensors");
        let map: std::collections::HashMap<String, Tensor> = t.into_iter().collect();
        candle_core::safetensors::save(&map, &ck_path).unwrap();
        let mut m = Inkling::load(&cfg_path, &ck_path, &Device::Cpu).unwrap();
        let i = cfg.intermediate_size;
        let mut g = Gen(0xe4be_7500);
        for l in 0..cfg.num_hidden_layers {
            let ex = (0..cfg.n_routed_experts)
                .map(|_| (g.tensor(&[2 * i, h], 1.0 / (h as f32).sqrt()), g.tensor(&[h, i], 1.0 / (i as f32).sqrt())))
                .collect();
            m.set_experts(l, LayerExperts::Source(Box::new(DenseExperts(ex))));
        }
        let _ = std::fs::remove_dir_all(&dir);
        m
    }

    fn prompt(n: usize) -> Vec<u32> {
        let mut g = Gen(0x7070);
        (0..n).map(|_| (g.next() % 60) as u32).collect()
    }

    fn row(logits: &Tensor) -> Vec<f32> {
        logits.flatten_all().unwrap().to_dtype(DType::F32).unwrap().to_vec1::<f32>().unwrap()
    }

    fn argmax(v: &[f32]) -> u32 {
        (0..v.len()).fold(0, |b, i| if v[i] > v[b] { i } else { b }) as u32
    }

    /// Every position's logits of a prompt read in pieces of `piece` tokens
    /// on `cache`, then `steps` greedy steps: one row a position.
    fn prompt_then_steps(m: &mut Inkling, cache: &mut InklingCache, ids: &[u32], piece: usize, steps: usize) -> Vec<Vec<f32>> {
        m.set_prefill_piece(piece);
        let logits = m.forward_pieces(ids, cache, true).unwrap();
        assert_eq!(cache.len(), ids.len());
        let mut rows: Vec<Vec<f32>> = (0..logits.dim(0).unwrap()).map(|r| row(&logits.narrow(0, r, 1).unwrap())).collect();
        for _ in 0..steps {
            let pos = cache.len();
            let l = CausalLm::step(m, argmax(rows.last().unwrap()), pos, cache).unwrap();
            rows.push(row(&l));
        }
        rows
    }

    /// The largest gap between two sets of rows, and the largest magnitude.
    fn gap(a: &[Vec<f32>], b: &[Vec<f32>]) -> (f32, f32) {
        assert_eq!(a.len(), b.len());
        let (mut d, mut s) = (0f32, 0f32);
        for (x, y) in a.iter().zip(b) {
            for (u, v) in x.iter().zip(y) {
                d = d.max((u - v).abs());
                s = s.max(v.abs());
            }
        }
        (d, s)
    }

    /// The normed embedding `forward` hands out, on the checkpoint's rows and
    /// on a sealed table's, is bit for bit `layers::embed`, which reads
    /// `embed_norm.weight` from the checkpoint at the call: the weight the
    /// model keeps from its load is the file's.
    #[test]
    fn the_embedding_norm_is_the_checkpoints_on_both_arms() {
        let mut m = tiny_model();
        let ids = prompt(9);
        let bits = |t: &Tensor| row(t).iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
        let reference = bits(&layers::embed(&m.ck, &m.cfg, &ids, &m.dev).unwrap());
        let checkpoint_arm = m.forward(&ids, &mut m.new_cache(), false, false).unwrap();
        assert_eq!(bits(&checkpoint_arm.embed), reference, "checkpoint rows");

        let vocab = m.cfg.vocab_size;
        let all: Vec<u32> = (0..vocab as u32).collect();
        let rows = row(&m.ck.embed_rows(&all, &Device::Cpu).unwrap());
        m.embed_table = Some(tables::EmbedTable::from_rows(rows, vocab, m.cfg.hidden_size).unwrap());
        let table_arm = m.forward(&ids, &mut m.new_cache(), false, false).unwrap();
        assert_eq!(bits(&table_arm.embed), reference, "table rows");
    }

    #[test]
    fn prefill_pieces_cover_the_prompt_without_a_lone_token() {
        assert_eq!(prefill_pieces(10, 0), vec![(0, 10)]);
        assert_eq!(prefill_pieces(10, 16), vec![(0, 10)]);
        assert_eq!(prefill_pieces(10, 4), vec![(0, 4), (4, 4), (8, 2)]);
        assert_eq!(prefill_pieces(9, 4), vec![(0, 4), (4, 5)]);
        assert_eq!(prefill_pieces(1, 4), vec![(0, 1)]);
        for (t, p) in [(70, 16), (70, 23), (5, 2), (1000, 7)] {
            let v = prefill_pieces(t, p);
            assert_eq!(v.iter().map(|x| x.1).sum::<usize>(), t);
            assert!(v.windows(2).all(|w| w[0].0 + w[0].1 == w[1].0));
            assert!(v.iter().all(|x| x.1 >= 2 && x.1 <= p + 1), "{t} by {p}: {v:?}");
        }
    }

    /// A 70-token prompt (far past the window of 4 and the extent of 6) read
    /// in pieces of 16 and of 23 (its lone last token joins the piece before)
    /// gives every position's logits, and the 6 steps after, within 1e-5 of
    /// one call: the convolution states, the cache positions and the block
    /// states carry across the pieces.
    #[test]
    fn prompt_in_pieces_matches_one_call() {
        let mut m = tiny_model();
        let ids = prompt(70);
        let mut cache = m.new_cache();
        let one = prompt_then_steps(&mut m, &mut cache, &ids, 0, 6);
        for piece in [16, 23] {
            let mut cache = m.new_cache();
            let pieces = prompt_then_steps(&mut m, &mut cache, &ids, piece, 6);
            let (d, s) = gap(&pieces, &one);
            println!("pieces of {piece}: largest gap {d:e} over {s:e}");
            assert!(d <= 1e-5 * s, "pieces of {piece} drift from one call: {d:e} of {s:e}");
        }
    }

    /// Bounded sliding slots (window 4, slack 2: 6 positions kept in buffers
    /// of 8) against full slots: a 70-token prompt, in one call and in pieces
    /// of 16, then 12 steps, give every row within 1e-5; the sliding slots
    /// have dropped their oldest positions and the full ones kept them all.
    #[test]
    fn bounded_sliding_slots_match_full_slots() {
        let mut m = tiny_model();
        let ids = prompt(70);
        let steps = 12;
        let w = m.cfg.sliding_window_size;
        for piece in [0, 16] {
            m.set_kv_window(None);
            let mut full = m.new_cache();
            let want = prompt_then_steps(&mut m, &mut full, &ids, piece, steps);
            m.set_kv_window(Some(2));
            let mut bounded = m.new_cache();
            let got = prompt_then_steps(&mut m, &mut bounded, &ids, piece, steps);
            let (d, s) = gap(&got, &want);
            let same = got.iter().flatten().zip(want.iter().flatten()).all(|(a, b)| a.to_bits() == b.to_bits());
            println!("pieces of {piece}: bounded against full, largest gap {d:e} over {s:e}, bit for bit {same}");
            assert!(d <= 1e-5 * s, "pieces of {piece}: bounded slots drift from full ones: {d:e} of {s:e}");
            let mut n_bounded = 0;
            for (layer, pass) in m.schedule() {
                let slot = &bounded.slots()[m.slot(layer, pass)];
                assert_eq!(slot.len(), ids.len() + steps);
                if m.cfg.is_sliding(layer) {
                    n_bounded += 1;
                    assert!(slot.is_bounded(), "visit ({layer}, {pass}) is not bounded");
                    assert!(slot.capacity() <= w + 4, "visit ({layer}, {pass}): {} positions", slot.capacity());
                    assert!(slot.base() > 0, "visit ({layer}, {pass}) dropped nothing");
                } else {
                    assert!(!slot.is_bounded() && slot.base() == 0 && slot.capacity() >= slot.len());
                }
            }
            assert!(n_bounded > 0 && n_bounded < m.n_slots());

            // Right after the prompt, before any step: a sliding slot has already given back what the prompt took.
            let mut c = m.new_cache();
            m.forward_pieces(&ids, &mut c, false).unwrap();
            for (layer, pass) in m.schedule() {
                let slot = &c.slots()[m.slot(layer, pass)];
                if m.cfg.is_sliding(layer) {
                    assert!(slot.capacity() <= w + 4, "after the prompt, visit ({layer}, {pass}): {}", slot.capacity());
                }
            }
        }
    }

    /// A bounded cache accepts a rollback within its slack, and the steps
    /// after it give the rows of a full cache that ran the same; it refuses
    /// a deeper one by name, unchanged; a rollback to the empty cache is
    /// always accepted.
    #[test]
    fn bounded_slot_refuses_a_rollback_deeper_than_its_slack() {
        let mut m = tiny_model();
        let ids = prompt(40);
        let within = |m: &Inkling| -> Vec<Vec<f32>> {
            let mut c = m.new_cache();
            m.forward_pieces(&ids, &mut c, false).unwrap();
            CausalLm::step(m, 9, 40, &mut c).unwrap();
            let mk = c.mark();
            CausalLm::step(m, 7, 41, &mut c).unwrap();
            CausalLm::step(m, 8, 42, &mut c).unwrap();
            c.rollback(&mk).unwrap();
            assert_eq!(c.len(), 41);
            let mut rows = vec![row(&CausalLm::step(m, 3, 41, &mut c).unwrap())];
            for p in 42..46 {
                rows.push(row(&CausalLm::step(m, argmax(rows.last().unwrap()), p, &mut c).unwrap()));
            }
            rows
        };
        m.set_kv_window(None);
        let want = within(&m);
        m.set_kv_window(Some(2));
        let got = within(&m);
        let (d, s) = gap(&got, &want);
        assert!(d <= 1e-5 * s, "the steps after a rollback within the slack drift: {d:e} of {s:e}");

        // A mark at 40, then 10 steps: the sliding slots have dropped keys a row at 40 sees.
        let mut c = m.new_cache();
        let empty = c.mark();
        m.forward_pieces(&ids, &mut c, false).unwrap();
        let mk = c.mark();
        for p in 40..50 {
            CausalLm::step(&m, (p % 60) as u32, p, &mut c).unwrap();
        }
        let before = c.clone();
        let err = c.rollback(&mk).expect_err("a rollback deeper than the slack was accepted").to_string();
        assert!(err.contains("too deep for a bounded sliding slot"), "{err}");
        assert_eq!(c.len(), 50);
        assert_eq!(c.tokens(), before.tokens());
        let mut b = before;
        let a = row(&CausalLm::step(&m, 5, 50, &mut c).unwrap());
        let r = row(&CausalLm::step(&m, 5, 50, &mut b).unwrap());
        assert!(a.iter().zip(&r).all(|(x, y)| x.to_bits() == y.to_bits()), "a refused rollback moved the cache");
        c.rollback(&empty).unwrap();
        assert_eq!(c.len(), 0);
        assert!(c.slots().iter().all(|s| s.base() == 0));
    }
}
