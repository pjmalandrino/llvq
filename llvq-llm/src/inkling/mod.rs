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

pub mod config;
pub mod layers;
pub mod loader;
#[cfg(all(target_os = "macos", feature = "metal"))]
pub mod resident;

use std::path::Path;

use candle_core::{DType, Device, Result, Tensor};

use config::InklingConfig;
use layers::{block_attn_res, LayerWeights, MoeSource, Routing, SlotCache};
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
}

impl InklingCache {
    /// Tokens held.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
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
    /// `unembed.weight` rows `0..unpadded_vocab_size`, BF16, `[V, H]`.
    unembed: Tensor,
    dev: Device,
    /// One entry per layer; [`LayerExperts::Native`] unless set.
    experts: Vec<LayerExperts>,
    /// When set, the wall time of every MoE call, `(layer, seconds)`, the
    /// device synchronized on both sides. Off by default.
    moe_timing: std::sync::Mutex<Option<Vec<(usize, f64)>>>,
}

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

    fn take_launches(&self) -> usize {
        self.inner.take_launches()
    }
}

/// Output-head rows widened to f32 per matmul: bounds the f32 copy to
/// `HEAD_CHUNK * H * 4` bytes (151 MB) whatever the vocabulary.
const HEAD_CHUNK: usize = 16384;

impl Inkling {
    /// Map `checkpoint` and load every dense weight onto `dev`.
    pub fn load(config: &Path, checkpoint: &Path, dev: &Device) -> anyhow::Result<Self> {
        let cfg = InklingConfig::from_file(config)?;
        let ck = NativeCheckpoint::open(checkpoint)?;
        let layers = (0..cfg.num_hidden_layers)
            .map(|l| LayerWeights::load(&ck, &cfg, l, dev))
            .collect::<Result<Vec<_>>>()?;
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
        // Inkling2UnpadVocab: the first `unpadded_vocab_size` rows (lm_head, line 1477)
        let unembed = ck.raw("unembed.weight", dev)?;
        anyhow::ensure!(unembed.dtype() == DType::BF16, "unembed.weight: {:?}, expected BF16", unembed.dtype());
        let unembed = unembed.narrow(0, 0, cfg.unpadded_vocab_size)?;
        let experts = (0..cfg.num_hidden_layers).map(|_| LayerExperts::Native).collect();
        Ok(Self {
            cfg,
            ck,
            layers,
            message_queries,
            final_query,
            norm,
            unembed,
            dev: dev.clone(),
            experts,
            moe_timing: std::sync::Mutex::new(None),
        })
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

    pub fn new_cache(&self) -> InklingCache {
        InklingCache { slots: vec![SlotCache::default(); self.n_slots()], len: 0 }
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
        let embed = layers::embed(&self.ck, &self.cfg, ids, &self.dev)?;
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
                Some(block_attn_res(&self.message_queries[layer - lc.core_start], pbs, ph)?)
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
            let (next, routing) = lw.forward(&self.cfg, &h, &bs, messages.as_deref(), src, slot)?;
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
        // line 1442-1443: `final_input` over the block states, then the norm
        let h = block_attn_res(&self.final_query, &bs, &h)?.remove(0);
        let h = layers::rms_norm(&h, &self.norm, self.cfg.rms_norm_eps)?;
        let h = if all_logits { h } else { h.narrow(0, h.dim(0)? - 1, 1)? };
        let logits = self.head(&h)?;
        Ok(ForwardOut { embed, visits, logits })
    }

    /// `lm_head` then the softcap (lines 1477-1478 and 751-757), the BF16
    /// rows widened to f32 a chunk at a time.
    fn head(&self, h: &Tensor) -> Result<Tensor> {
        let v = self.unembed.dim(0)?;
        let mut parts = vec![];
        let mut start = 0;
        while start < v {
            let n = HEAD_CHUNK.min(v - start);
            let w = self.unembed.narrow(0, start, n)?.to_dtype(DType::F32)?;
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
        let out = self.forward(tokens, cache, false, false)?;
        out.logits.reshape((1, 1, self.cfg.unpadded_vocab_size))
    }

    fn step(&self, token: u32, offset: usize, cache: &mut Self::Cache) -> Result<Tensor> {
        if offset != cache.len() {
            candle_core::bail!("step at offset {offset}, the cache holds {} tokens", cache.len());
        }
        let out = self.forward(&[token], cache, false, false)?;
        out.logits.reshape((1, 1, self.cfg.unpadded_vocab_size))
    }
}
