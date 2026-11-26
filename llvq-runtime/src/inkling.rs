//! Inkling 2 Mini in the server: the local file that names its parts, and
//! its engine.
//!
//! The file is not published, so it is not a library entry. A small JSON file
//! ([`LocalModel`]) names the parts `inkchat` and `inkgsm` load: the config,
//! the checkpoint, the layers' codes, the tokenizer and the two tables.
//! `llvq serve --local <file>` registers it in the store under its name, with
//! the family `inkling2`, and every route then finds it as it finds a pulled
//! model.
//!
//! The engine owns the model, the template and the cache of the last
//! conversation, on the engine's thread, as the Qwen3 engine does.
//!
//! - The prompt is the checkpoint's template through `llvq_llm::inkchat`, with
//!   ids. The reasoning effort is OpenAI's `reasoning_effort` when the client
//!   sends one; otherwise `none`, the setting of our GSM8K runs, or the
//!   template's default (0.9) when an Ollama client asks to think.
//! - The model's turn is parsed by its ids (`inkchat::TurnParser`): thinking
//!   goes to the reasoning, text to the content, and each call to the content
//!   in the form [`crate::tools::inkling2::call_text`] writes, which the
//!   routes' parser reads back as a call.
//! - Prefix reuse. The cache is held as segments, each with a mark of the
//!   model's cache after it ([`TurnLm::mark`]). A segment is a stretch of the
//!   rendered prompt between two assistant turns, or one answer of the model.
//!   An answer is held as the ids the model wrote, and keyed by the ids the
//!   template renders for the message the client sends back. A request reads
//!   only what follows the last segment it starts with; the cache is rolled
//!   back to that segment's mark first.
//! - Greedy by default (temperature 0), as our measured runs. A temperature
//!   above zero samples (`repl::Sampler`).
//! - An answer cut by its cap, by a stop string, by a client that left, or by
//!   a turn the template's parser refuses, is dropped from the cache: the
//!   next request reads it again as the client renders it.
//! - The prefix cache on disk ([`crate::prefix`], `llvq serve --prefix-cache`):
//!   the first read of a prompt whose static block (system prompt and tools)
//!   is long saves the cache at the last piece boundary of that block; a later
//!   prompt that starts with the same tokens, in a new conversation or after a
//!   restart, loads it instead of reading them. The read after a load cuts the
//!   prompt where a read from 0 does, so its logits are those of a read from 0.

use crate::chat::{self, Msg, Role, Scan};
use crate::engine::{nanos, total, EngineInfo, Event, Job, Prompt, State, Stats, DEFAULT_TOP_P};
use crate::library::{Backend, Configs, ModelName};
use crate::store::{Manifest, Store};
use crate::tools::inkling2;
use candle_core::{DType, Device, Tensor};
use llvq_llm::inkchat::{Delta, InkChat, Json, Part, TurnParser};
use llvq_llm::repl::Sampler;
use serde::{Deserialize, Serialize};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant, SystemTime};

/// The family a local file serves.
pub const FAMILY: &str = "inkling2";

/// Positions served when the local file names none: the target of
/// `docs/plan-inkling2-mini.md` J7. The KV cache costs 1.24 GB there in f16
/// (*computed*, same plan).
pub const DEFAULT_CONTEXT: usize = 32_768;

// ---------------------------------------------------------------------------
// The local file.
// ---------------------------------------------------------------------------

/// A model served from files on this machine. Relative paths are read from
/// the file's own directory.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LocalModel {
    /// The name it is served under, `name:tag`.
    pub name: String,
    /// `inkling2`.
    pub family: String,
    /// The text model's `config.json`.
    pub config: PathBuf,
    /// The text model's `model.safetensors`.
    pub checkpoint: PathBuf,
    /// The directory of the experts' lattice codes.
    pub layers: PathBuf,
    /// `tokenizer.json` of the checkpoint.
    pub tokenizer: PathBuf,
    /// The directory of the sealed file's two int4 tables. Required by the
    /// fused stack and on a CPU build.
    #[serde(default)]
    pub tables: Option<PathBuf>,
    /// `fused` (default): the stack `inkrun` and `inkgsm --stack fused`
    /// measure. `candle`: the reference path, slower. A CPU build serves
    /// its own stack whatever this says (`experts_cpu::load_served`).
    #[serde(default)]
    pub stack: Option<String>,
    /// Positions served; default [`DEFAULT_CONTEXT`].
    #[serde(default)]
    pub context: Option<usize>,
    /// Layers whose experts must be resident; default every layer.
    #[serde(default)]
    pub require_layers: Option<usize>,
    /// Shown by `/api/tags` and `/api/show`.
    #[serde(default)]
    pub parameter_size: Option<String>,
    #[serde(default)]
    pub bits_per_param: Option<String>,
}

impl LocalModel {
    /// Read and check a local file, its paths made absolute.
    pub fn read(path: &Path) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path).map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
        let mut m: LocalModel = serde_json::from_str(&text).map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
        anyhow::ensure!(
            m.family == FAMILY,
            "{}: family {:?}; a local file serves the family {FAMILY:?} only",
            path.display(),
            m.family
        );
        ModelName::parse(&m.name).map_err(|e| anyhow::anyhow!("{}: name: {e}", path.display()))?;
        m.fused().map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
        anyhow::ensure!(m.context != Some(0), "{}: context 0 holds no position", path.display());
        let dir = path.parent().unwrap_or(Path::new("."));
        let abs = |p: &mut PathBuf| {
            if p.is_relative() {
                *p = dir.join(&*p);
            }
        };
        abs(&mut m.config);
        abs(&mut m.checkpoint);
        abs(&mut m.layers);
        abs(&mut m.tokenizer);
        if let Some(t) = m.tables.as_mut() {
            abs(t);
        }
        Ok(m)
    }

    /// Whether the stack is the fused one, or why the file is refused.
    pub fn fused(&self) -> Result<bool, String> {
        match self.stack.as_deref() {
            None | Some("fused") => match self.tables {
                Some(_) => Ok(true),
                None => Err("the fused stack reads the int4 tables: name their directory in \"tables\"".into()),
            },
            Some("candle") => Ok(false),
            Some(o) => Err(format!("stack {o:?}: fused or candle")),
        }
    }

    /// Every path the load reads, named, so a missing one is refused before
    /// anything loads.
    fn paths(&self) -> Vec<(&'static str, &Path)> {
        let mut v = vec![
            ("config", self.config.as_path()),
            ("checkpoint", self.checkpoint.as_path()),
            ("layers", self.layers.as_path()),
            ("tokenizer", self.tokenizer.as_path()),
        ];
        if let Some(t) = &self.tables {
            v.push(("tables", t.as_path()));
        }
        v
    }
}

/// Register the local file `path` in `store` under its name: a manifest of
/// the family `inkling2` whose `file` is the local file and whose digest is
/// the local file's. Nothing is copied. The model loads on first use, as a
/// pulled one does.
pub fn register(store: &Store, path: &Path) -> anyhow::Result<(ModelName, Manifest)> {
    let path = std::fs::canonicalize(path).map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
    let m = LocalModel::read(&path)?;
    for (what, p) in m.paths() {
        anyhow::ensure!(p.exists(), "{}: {what} {} does not exist", path.display(), p.display());
    }
    let name = ModelName::parse(&m.name).map_err(anyhow::Error::msg)?;
    let file = path.to_str().ok_or_else(|| anyhow::anyhow!("{}: not UTF-8", path.display()))?.to_string();
    let manifest = Manifest {
        schema: 1,
        sha256: crate::store::sha256_file(&path)?,
        size: std::fs::metadata(&m.checkpoint)?.len(),
        repo: "local".into(),
        revision: "local".into(),
        file,
        family: FAMILY.into(),
        parameter_size: m.parameter_size.clone().unwrap_or_default(),
        bits_per_param: m.bits_per_param.clone().unwrap_or_default(),
        configs: Configs { cuda: None, metal: None, cpu: None, cuda_batch: None, cuda_eager: None },
        modified_at: crate::store::now_secs(),
    };
    store.write_manifest(&name, &manifest)?;
    Ok((name, manifest))
}

// ---------------------------------------------------------------------------
// The model behind the engine.
// ---------------------------------------------------------------------------

/// One sequence on a model: what the engine drives. Implemented for Inkling 2
/// here, and for scripted models in the tests.
pub trait TurnLm {
    /// What [`Self::mark`] keeps.
    type Mark: Clone;
    /// Feed `ids` after what the cache holds. The logits of the last one.
    fn feed(&mut self, ids: &[u32]) -> anyhow::Result<Tensor>;
    /// Positions the cache holds.
    fn held(&self) -> usize;
    /// The state needed to come back here.
    fn mark(&self) -> Self::Mark;
    /// Back to `m`, a mark of what the cache holds now.
    fn rollback(&mut self, m: &Self::Mark) -> anyhow::Result<()>;
    /// An empty cache.
    fn reset(&mut self);
    /// Write the cache's whole state, for the prefix cache. Not every model can.
    fn save_state(&self, _w: &mut dyn std::io::Write) -> anyhow::Result<()> {
        anyhow::bail!("this model does not write its cache")
    }
    /// Replace the cache by a state [`Self::save_state`] wrote, reading all of
    /// it. On an error the cache may hold anything: the caller resets it.
    fn load_state(&mut self, _r: &mut dyn std::io::Read) -> anyhow::Result<()> {
        anyhow::bail!("this model does not read a cache")
    }
}

struct InklingLm {
    model: llvq_llm::inkling::Inkling,
    cache: llvq_llm::inkling::InklingCache,
}

impl TurnLm for InklingLm {
    type Mark = llvq_llm::inkling::InklingMark;

    fn feed(&mut self, ids: &[u32]) -> anyhow::Result<Tensor> {
        // In pieces when the model is set to read a prompt so (`set_prefill_piece`), in one call otherwise.
        Ok(self.model.forward_pieces(ids, &mut self.cache, false)?)
    }

    fn held(&self) -> usize {
        self.cache.len()
    }

    fn mark(&self) -> Self::Mark {
        self.cache.mark()
    }

    fn rollback(&mut self, m: &Self::Mark) -> anyhow::Result<()> {
        Ok(self.cache.rollback(m)?)
    }

    fn reset(&mut self) {
        self.cache = self.model.new_cache();
    }

    fn save_state(&self, w: &mut dyn std::io::Write) -> anyhow::Result<()> {
        Ok(self.cache.write_state(w)?)
    }

    fn load_state(&mut self, r: &mut dyn std::io::Read) -> anyhow::Result<()> {
        // The old cache goes first, so the device never holds both.
        self.cache = self.model.new_cache();
        self.cache = self.model.read_cache(r)?;
        Ok(())
    }
}

#[allow(unused_variables, clippy::match_single_binding)]
fn make_resident(model: &mut llvq_llm::inkling::Inkling, layers: &Path, dev: &Device) -> anyhow::Result<()> {
    match dev {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        Device::Metal(_) => {
            let rt = llvq_llm::inkling::resident_runtime(dev, llvq_llm::inkling::DEFAULT_PREFILL_ROWS)?;
            model.make_resident(layers, rt)?;
        }
        #[cfg(all(target_os = "linux", feature = "cuda"))]
        Device::Cuda(_) => {
            let rt = llvq_llm::inkling::resident_cuda::served_runtime(dev, &model.cfg)?;
            llvq_llm::inkling::resident_cuda::make_resident(model, layers, rt)?;
        }
        _ => eprintln!("llvq: no resident path on this device; every layer keeps its native experts"),
    }
    Ok(())
}

/// The model as `inkgsm --stack fused` loads it (or `--stack candle`), and on
/// CUDA a prompt's rows through the experts in one launch a step
/// (`Inkling::set_moe_batch_rows`, byte-identical values). Returns the lines
/// the load prints.
/// Tokens of a prompt one forward call reads, on CUDA and on Metal.
const PROMPT_PIECE: usize = 512;
/// Query rows of one attention score tensor, on CUDA and on Metal.
const SCORE_ROWS: usize = 128;
/// Positions past its window a bounded sliding slot keeps, and so the deepest
/// rollback it accepts.
const KV_SLACK: usize = 1024;

/// Whether a model loaded on `backend` reads a long prompt with the settings
/// above, and where its score bias is built.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LongPrompt {
    /// One call a prompt, every position in every slot (the candle stack, the CPU).
    Off,
    /// The pieces, the chunked scores and the bounded slots, the bias built on
    /// the host (Metal: `attn_bias` is a CUDA kernel).
    BiasOnHost,
    /// The same, the bias built on the card (`attn_bias::set_on_device`).
    BiasOnCard,
}

/// The three settings depend on no device: they cut a prompt's calls, its
/// score tensors and its sliding slots. Only the bias kernel is CUDA's.
fn long_prompt(backend: Backend, fused: bool) -> LongPrompt {
    match (backend, fused) {
        (Backend::Cuda, true) => LongPrompt::BiasOnCard,
        (Backend::Metal, true) => LongPrompt::BiasOnHost,
        _ => LongPrompt::Off,
    }
}

/// The model on the CPU: `experts_cpu::load_served`, the load `inkcpu`
/// measures. One stack whatever the local file's `stack` says: the attention
/// matrices in BF16 through the CPU kernels, the glue on candle in f32, the
/// int4 head and embedding of `tables`, the experts from their codes.
fn load_model_cpu(l: &LocalModel) -> anyhow::Result<(llvq_llm::inkling::Inkling, Vec<String>)> {
    let Some(tables) = &l.tables else {
        anyhow::bail!("the CPU stack reads the int4 tables: name their directory in \"tables\"");
    };
    let (model, mut lines) = llvq_llm::inkling::experts_cpu::load_served(&l.config, &l.checkpoint, &l.layers, tables)?;
    let resident = model.substituted_layers();
    let need = l.require_layers.unwrap_or(model.cfg.num_hidden_layers);
    anyhow::ensure!(resident.len() >= need, "{} layers on the CPU experts, {need} required", resident.len());
    lines.push(format!(
        "stack cpu, {} layers on the CPU experts, glue {}, attention {}",
        resident.len(),
        model.glue().name(),
        model.attn().name()
    ));
    model.take_launches();
    Ok((model, lines))
}

fn load_model(l: &LocalModel, dev: &Device) -> anyhow::Result<(llvq_llm::inkling::Inkling, Vec<String>)> {
    use llvq_llm::inkling::{tables, Inkling, LoadOptions};
    if dev.is_cpu() {
        return load_model_cpu(l);
    }
    let fused = l.fused().map_err(anyhow::Error::msg)?;
    let mut lines = Vec::new();
    let mut model = match (&l.tables, fused) {
        (Some(d), true) => {
            let opts = LoadOptions { int4_head: Some(d.join(tables::UNEMBED_FILE)) };
            Inkling::load_with_attn(&l.config, &l.checkpoint, dev, &opts, "fused+scores".parse()?)?
        }
        _ => Inkling::load(&l.config, &l.checkpoint, dev)?,
    };
    make_resident(&mut model, &l.layers, dev)?;
    if fused {
        model.set_glue("fused".parse()?)?;
        model.set_routing_on_device(true);
        model.set_moe_act(true);
        model.set_moe_four_launches(true);
    }
    if let Some(d) = &l.tables {
        if fused {
            // The int4 head reads its own table: no dense head is built to be dropped.
            let t = tables::load_embed(d, &model.cfg, false)?;
            lines.push(format!("tables: embed {}, unembed {}", &t.embed_sha256[..16], &t.unembed_sha256[..16]));
            model.embed_table = Some(t.embed);
        } else {
            let t = tables::load(d, &model.cfg, dev)?;
            lines.push(format!("tables: embed {}, unembed {}", &t.embed_sha256[..16], &t.unembed_sha256[..16]));
            model.embed_table = Some(t.embed);
            model.out_proj = Some(Box::new(t.head));
        }
    }
    if matches!(dev, Device::Cuda(_)) {
        model.set_moe_batch_rows(true);
        lines.push("experts: a prompt's rows in one launch a step".into());
    }
    let backend = if dev.is_cuda() { Backend::Cuda } else { Backend::Metal };
    let long = long_prompt(backend, fused);
    if long != LongPrompt::Off {
        // The long-prompt settings measured on CUDA on 2026-11-24 (9,000 tokens: 12.0 GB at the peak against
        // 19.5, read in 23.85 s against 55.89 on an L40S), and on Metal on 2026-11-26
        // (`docs/mesures/inkling-mac-serve-2026-11-26.txt`). The pieces change values inside the perplexity band
        // (-0.042 %); the bounded slots and the chunked scores add no difference. The slack is what a rollback
        // may undo: an answer longer than it is read again from the start (`Worker::back_to`).
        model.set_prefill_piece(PROMPT_PIECE);
        model.set_attn_chunk(SCORE_ROWS);
        model.set_kv_window(Some(KV_SLACK));
        #[cfg(all(target_os = "linux", feature = "cuda"))]
        if long == LongPrompt::BiasOnCard {
            llvq_llm::inkling::attn_bias::set_on_device(Some(dev))?;
        }
        let bias = if long == LongPrompt::BiasOnCard { "bias on the card" } else { "bias on the host" };
        lines.push(format!(
            "prompt: pieces of {PROMPT_PIECE} tokens, scores in chunks of {SCORE_ROWS} rows, sliding slots bounded with a slack of {KV_SLACK}, {bias}"
        ));
    }
    let resident = model.substituted_layers();
    let need = l.require_layers.unwrap_or(model.cfg.num_hidden_layers);
    anyhow::ensure!(resident.len() >= need, "{} layers resident, {need} required", resident.len());
    lines.push(format!(
        "stack {}, {} layers resident, glue {}, attention {}",
        if fused { "fused" } else { "candle" },
        resident.len(),
        model.glue().name(),
        model.attn().name()
    ));
    model.take_launches();
    dev.synchronize()?;
    Ok((model, lines))
}

/// The prefix cache of a model loaded on `backend` from the local file
/// `path`, in `<store>/prefix`, as the options of `llvq serve` ask: on by
/// default on Metal only, refused by name when asked for on a load without the
/// long-prompt settings (its save point is a boundary of their pieces).
fn prefix_cache(store: &Store, path: &Path, backend: Backend, fused: bool, long: LongPrompt) -> anyhow::Result<Option<crate::prefix::PrefixCache>> {
    let o = crate::prefix::options();
    let on = o.enabled.unwrap_or(backend == Backend::Metal);
    if !on {
        return Ok(None);
    }
    if long == LongPrompt::Off {
        anyhow::ensure!(
            o.enabled != Some(true),
            "--prefix-cache on: the prefix cache saves at a boundary of the prompt's pieces, and this load ({}, stack {}) \
             reads a prompt in one call; it serves the fused stack on Metal or CUDA",
            backend.label(),
            if fused { "fused" } else { "candle" }
        );
        return Ok(None);
    }
    let exe = std::env::current_exe()?;
    let key = crate::prefix::Key {
        format: crate::prefix::FORMAT,
        family: FAMILY.into(),
        model_sha256: crate::store::sha256_file(path)?,
        build_sha256: crate::store::sha256_file(&exe)?,
        backend: backend.label().into(),
        stack: if fused { "fused" } else { "candle" }.into(),
        prompt_piece: PROMPT_PIECE,
        score_rows: SCORE_ROWS,
        kv_slack: KV_SLACK,
        bias: if long == LongPrompt::BiasOnCard { "card" } else { "host" }.into(),
    };
    Ok(Some(crate::prefix::PrefixCache::new(store.root().join("prefix"), o.max_bytes, key)))
}

/// Load the model a local file names, on this build's device.
pub fn load_local(name: &ModelName, path: &Path, store: &Store) -> anyhow::Result<InkEngine> {
    let local = LocalModel::read(path)?;
    let backend = Backend::of_build();
    eprintln!("llvq: loading {name} (inkling2, {}) on {}", path.display(), backend.label());
    let dev = match backend {
        Backend::Cuda => Device::new_cuda(0)?,
        Backend::Metal => llvq_llm::inkling::new_metal_device()?,
        Backend::Cpu => Device::Cpu,
    };
    let tok = tokenizers::Tokenizer::from_file(&local.tokenizer)
        .map_err(|e| anyhow::anyhow!("{}: {e}", local.tokenizer.display()))?;
    let chat = InkChat::new(tok)?;
    // `inkchat` was written against one template: a checkpoint that carries
    // another is refused rather than prompted in a format it does not read.
    let template = local.tokenizer.with_file_name("chat_template.jinja");
    let mut first = vec![match template.exists() {
        true => {
            let sha = crate::store::sha256_file(&template)?;
            anyhow::ensure!(
                sha == llvq_llm::inkchat::TEMPLATE_SHA256,
                "{}: sha256 {sha}, the template llvq_llm::inkchat renders is {}",
                template.display(),
                llvq_llm::inkchat::TEMPLATE_SHA256
            );
            "template: chat_template.jinja beside the tokenizer is the one inkchat renders".to_string()
        }
        false => format!(
            "template: no chat_template.jinja beside the tokenizer; inkchat renders sha256 {}",
            &llvq_llm::inkchat::TEMPLATE_SHA256[..16]
        ),
    }];
    let fused = !dev.is_cpu() && local.fused().map_err(anyhow::Error::msg)?;
    let prefix = prefix_cache(store, path, backend, fused, long_prompt(backend, fused))?;
    let (model, mut lines) = load_model(&local, &dev)?;
    first.append(&mut lines);
    first.push(match &prefix {
        Some(p) => format!(
            "prefix cache: on, in {}, cap {:.1} GiB, key {}",
            p.dir().display(),
            p.max_bytes() as f64 / (1u64 << 30) as f64,
            p.key().id()
        ),
        None => "prefix cache: off".into(),
    });
    let lines = first;
    for l in &lines {
        eprintln!("llvq: {name}: {l}");
    }
    let lm = InklingLm { cache: model.new_cache(), model };
    let config = path.file_name().map(|f| f.to_string_lossy().into_owned());
    Ok(InkEngine::with_prefix(lm, chat, local.context.unwrap_or(DEFAULT_CONTEXT), backend.label(), config, prefix))
}

// ---------------------------------------------------------------------------
// The engine.
// ---------------------------------------------------------------------------

/// An Inkling engine, built and ready: what [`crate::engine::Built`] carries.
pub struct InkEngine(Box<dyn Turns>);

trait Turns {
    fn ready(&self) -> State;
    fn run(&mut self, job: &Job, load: Duration);
}

impl InkEngine {
    /// The engine over `lm`, with the template of `chat`, serving `context`
    /// positions. `backend` and `config` are what `/api/ps` shows.
    pub fn new<M: TurnLm + 'static>(lm: M, chat: InkChat, context: usize, backend: &str, config: Option<String>) -> Self {
        Self::with_prefix(lm, chat, context, backend, config, None)
    }

    /// [`Self::new`] with the prefix cache `prefix` ([`crate::prefix`]).
    pub fn with_prefix<M: TurnLm + 'static>(
        lm: M,
        chat: InkChat,
        context: usize,
        backend: &str,
        config: Option<String>,
        prefix: Option<crate::prefix::PrefixCache>,
    ) -> Self {
        InkEngine(Box::new(Worker { lm, chat, context, backend: backend.into(), config, held: Vec::new(), prefix }))
    }
}

/// Serve the jobs of `rx` until the channel closes: the batch-1 loop of the
/// Qwen3 engine, with its panic handling.
pub(crate) fn serve(
    mut e: InkEngine,
    rx: &std::sync::mpsc::Receiver<Job>,
    inf: &EngineInfo,
    name: &ModelName,
    load: Duration,
) {
    *inf.state.lock().expect("state lock") = e.0.ready();
    let mut load = Some(load);
    while let Ok(job) = rx.recv() {
        inf.take_job();
        let l = load.take().unwrap_or_default();
        let ran = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| e.0.run(&job, l)));
        if let Err(p) = ran {
            let msg = format!("the engine of {name} panicked: {}", crate::engine::panic_message(&*p));
            // The weights go now, as in the Qwen3 engine.
            drop(e);
            *inf.state.lock().expect("state lock") = State::Failed(msg.clone());
            let _ = job.events.blocking_send(Event::Error { status: 500, message: msg.clone() });
            inf.busy.store(false, Ordering::SeqCst);
            while let Ok(job) = rx.recv() {
                inf.queue.fetch_sub(1, Ordering::SeqCst);
                let _ = job.events.blocking_send(Event::Error { status: 500, message: msg.clone() });
            }
            return;
        }
        *inf.expires.lock().expect("expires lock") = job.keep_alive.until(SystemTime::now());
        inf.busy.store(false, Ordering::SeqCst);
    }
}

/// A stretch of the cache: the rendered ids a request must start with to
/// reuse it (an answer has one key, the message the client sends back), and
/// the cache's length and mark after it.
struct Held<K> {
    keys: Vec<Vec<u32>>,
    len: usize,
    mark: K,
}

struct Worker<M: TurnLm> {
    lm: M,
    chat: InkChat,
    context: usize,
    backend: String,
    config: Option<String>,
    held: Vec<Held<M::Mark>>,
    prefix: Option<crate::prefix::PrefixCache>,
}

fn refusal(status: u16, message: impl Into<String>) -> Event {
    Event::Error { status, message: message.into() }
}

/// The length of the key each held segment matched, in order, while `ids`
/// starts with one of its keys.
pub fn reuse(keys: &[&[Vec<u32>]], ids: &[u32]) -> Vec<usize> {
    let mut at = 0;
    let mut out = Vec::new();
    for ks in keys {
        match ks.iter().find(|k| !k.is_empty() && ids[at..].starts_with(k)) {
            Some(k) => {
                at += k.len();
                out.push(k.len());
            }
            None => break,
        }
    }
    out
}

/// The pieces `from..end` of a rendered prompt is fed in: cut at the start
/// and the end of every assistant turn, so each answer is a segment of its
/// own.
pub fn pieces(spans: &[Range<usize>], from: usize, end: usize) -> Vec<Range<usize>> {
    let mut cuts: Vec<usize> =
        spans.iter().flat_map(|s| [s.start, s.end]).filter(|&p| p > from && p < end).collect();
    cuts.sort_unstable();
    cuts.dedup();
    let mut out = Vec::with_capacity(cuts.len() + 1);
    let mut a = from;
    for c in cuts.into_iter().chain([end]) {
        if c > a {
            out.push(a..c);
            a = c;
        }
    }
    out
}

/// Where the static block of a prompt ends: the length of the common prefix
/// of `ids` (the rendering of `msgs`) and of the rendering of the leading
/// system messages of `msgs` alone, with the same tools and effort. `None`
/// when every message is a system one, or when that rendering is refused.
pub fn static_end(chat: &InkChat, msgs: &[Json], tools: &[Json], effort: &Json, ids: &[u32]) -> Option<usize> {
    let k = msgs.iter().take_while(|m| m.get("role").and_then(Json::as_str) == Some("system")).count();
    if k == msgs.len() {
        return None;
    }
    let r = chat.render(&msgs[..k], tools, Some(effort)).ok()?;
    Some(crate::prefix::common_prefix(ids, &r.ids))
}

/// Whether a prompt of `len` ids whose first piece ends at `first_cut` (its
/// first assistant turn, or its end) can be cut at `b` with the pieces of a
/// read from 0 unchanged: `b` is a boundary of the forward pieces of `piece`
/// tokens, and the rest of that first piece is not a lone token (a lone last
/// token joins the piece before it, `llvq_llm::inkling::prefill_pieces`).
pub fn cut_keeps_pieces(b: usize, piece: usize, first_cut: usize, len: usize) -> bool {
    piece > 0 && b > 0 && b.is_multiple_of(piece) && b < len && (first_cut == b || first_cut >= b + 2)
}

/// The ids the template renders for the assistant message a client sends
/// back after streaming `parts`: the text as the content, with the trailing
/// whitespace the routes drop before a call dropped, and the calls with
/// their arguments. The reasoning is not sent back: the routes do not read it.
pub fn answer_key(chat: &InkChat, parts: &[Part]) -> Option<Vec<u32>> {
    let mut text = String::new();
    let mut calls = Vec::new();
    for p in parts {
        match p {
            Part::Thinking { .. } => {}
            Part::Text { text: t, .. } => text.push_str(t),
            Part::ToolCall(c) => calls.push(c),
        }
    }
    let mut kv = vec![("role".to_string(), Json::str("assistant"))];
    match calls.is_empty() {
        true => kv.push(("content".into(), Json::Str(text))),
        false => {
            kv.push(("content".into(), Json::str(text.trim_end())));
            let calls = calls
                .iter()
                .map(|c| {
                    Json::Object(vec![
                        ("type".into(), Json::str("function")),
                        (
                            "function".into(),
                            Json::Object(vec![("name".into(), Json::str(&c.name)), ("arguments".into(), c.arguments.clone())]),
                        ),
                    ])
                })
                .collect();
            kv.push(("tool_calls".into(), Json::Array(calls)));
        }
    }
    let r = chat.render(&[Json::Object(kv)], &[], None).ok()?;
    let span = r.assistant_spans.first()?;
    Some(r.ids[span.clone()].to_vec())
}

/// The effort the template writes for a request.
fn effort(e: Option<&str>, think: bool) -> Json {
    match e {
        Some(s) if s.trim().parse::<f64>().is_ok() => Json::parse(s.trim()).unwrap_or_else(|_| Json::str(s)),
        Some(s) => Json::str(s),
        // the template's default, 0.9
        None if think => Json::Null,
        None => Json::str("none"),
    }
}

/// The first maximum at temperature zero, the sampler's draw above it.
fn pick(logits: &Tensor, sampler: &mut Sampler) -> anyhow::Result<u32> {
    let row = logits.flatten_all()?.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
    if sampler.temp > 0.0 {
        return sampler.pick(&row);
    }
    let v = row.to_vec1::<f32>()?;
    let (mut best, mut at) = (f32::NEG_INFINITY, 0usize);
    for (i, &x) in v.iter().enumerate() {
        if x > best {
            best = x;
            at = i;
        }
    }
    anyhow::ensure!(best.is_finite(), "logits without a finite maximum");
    Ok(at as u32)
}

/// How a turn ended.
#[derive(PartialEq)]
enum End {
    /// `<|content_model_end_sampling|>`.
    Turn,
    /// A stop string.
    Stop,
    Length,
    /// The template's parser refused the turn.
    Refused,
    /// The client left.
    Gone,
}

impl<M: TurnLm> Turns for Worker<M> {
    fn ready(&self) -> State {
        State::Ready {
            backend: self.backend.clone(),
            config: self.config.clone(),
            context: self.context,
            greedy_only: false,
            pool_bytes: 0,
        }
    }

    fn run(&mut self, job: &Job, load: Duration) {
        let started = Instant::now();
        let tx = &job.events;
        if tx.is_closed() {
            return;
        }
        let send = |e: Event| tx.blocking_send(e).is_ok();
        if let Err(e) = self.run_inner(job, started, load, &send) {
            // The cache may hold half a feed: the next request starts empty.
            self.lm.reset();
            self.held.clear();
            send(refusal(500, format!("{e:#}")));
        }
    }
}

impl<M: TurnLm> Worker<M> {
    fn run_inner(&mut self, job: &Job, started: Instant, load: Duration, send: &dyn Fn(Event) -> bool) -> anyhow::Result<()> {
        let req = &job.request;
        let o = &req.options;
        let (msgs, tools) = match &req.prompt {
            Prompt::Load => {
                send(Event::Accepted { prompt_tokens: 0, cached: 0 });
                send(Event::Done(Stats {
                    done_reason: "load".into(),
                    total_duration: nanos(total(job, started, load)),
                    load_duration: nanos(load),
                    ..Stats::default()
                }));
                return Ok(());
            }
            Prompt::Chat(m) => (inkling2::chat_messages(m), Vec::new()),
            Prompt::ToolChat(p) => match inkling2::messages(&p.tools, &p.msgs) {
                Ok((t, m)) => (m, t),
                Err(e) => {
                    send(refusal(400, e));
                    return Ok(());
                }
            },
            Prompt::Templated { system, prompt } => {
                let mut m = Vec::new();
                if let Some(s) = system {
                    m.push(Msg { role: Role::System, content: s.clone() });
                }
                m.push(Msg { role: Role::User, content: prompt.clone() });
                (inkling2::chat_messages(&m), Vec::new())
            }
            Prompt::Raw(_) => {
                send(refusal(400, "raw prompts are not served for the inkling2 family: send messages"));
                return Ok(());
            }
        };
        let temp = o.temperature.unwrap_or(0.0);
        if !(0.0..=100.0).contains(&temp) {
            send(refusal(400, format!("temperature {temp}: between 0 and 100")));
            return Ok(());
        }
        let top_p = o.top_p.unwrap_or(DEFAULT_TOP_P);
        if !(0.0..=1.0).contains(&top_p) {
            send(refusal(400, format!("top_p {top_p}: between 0 and 1")));
            return Ok(());
        }
        let limit = match o.num_ctx {
            Some(n) if n > self.context => {
                send(refusal(400, format!("num_ctx {n}: this model serves at most {} positions here", self.context)));
                return Ok(());
            }
            Some(0) => {
                send(refusal(400, "num_ctx 0: a context of no position"));
                return Ok(());
            }
            Some(n) => n,
            None => self.context,
        };
        let eff = effort(o.effort.as_deref(), req.think);
        let r = match self.chat.render(&msgs, &tools, Some(&eff)) {
            Ok(r) => r,
            Err(e) => {
                send(refusal(400, format!("{e:#}")));
                return Ok(());
            }
        };

        // What the cache already holds of this prompt.
        let keys: Vec<&[Vec<u32>]> = self.held.iter().map(|h| h.keys.as_slice()).collect();
        let mut matched = reuse(&keys, &r.ids);
        let mut from: usize = matched.iter().sum();
        if from == r.ids.len() {
            // The whole prompt is held: its last segment is read again for
            // the logits of its last position.
            from -= matched.pop().unwrap_or(0);
        }
        let kept = matched.len();
        let base = match kept {
            0 => 0,
            n => self.held[n - 1].len,
        };
        // Back to what is kept, before anything is counted: a bounded sliding slot refuses a rollback deeper than
        // its slack, and the whole prompt is then read again.
        let (kept, from, base) = if kept < self.held.len() || self.lm.held() != base {
            let back = match kept {
                0 => {
                    self.lm.reset();
                    Ok(())
                }
                n => self.lm.rollback(&self.held[n - 1].mark),
            };
            match back {
                Ok(()) => {
                    self.held.truncate(kept);
                    (kept, from, base)
                }
                Err(e) => {
                    eprintln!("llvq: the cache could not go back to its kept prefix, the prompt is read again: {e:#}");
                    self.lm.reset();
                    self.held.clear();
                    (0, 0, 0)
                }
            }
        } else {
            (kept, from, base)
        };
        let _ = kept;
        // The prefix cache: a saved state longer than what memory keeps is loaded in its place.
        let first_cut = pieces(&r.assistant_spans, 0, r.ids.len()).first().map_or(r.ids.len(), |p| p.end);
        let n_ids = r.ids.len();
        let fits = |b: usize, piece: usize| cut_keeps_pieces(b, piece, first_cut, n_ids);
        // Whether the read from `from` cuts the prompt where a read from 0 does: what a save needs.
        let mut aligned = from == 0;
        let (from, base) = match self.prefix.clone() {
            Some(pc) => match self.load_prefix(pc.clone(), &r.ids, from, &|b| fits(b, pc.piece())) {
                Some(b) => {
                    aligned = true;
                    (b, b)
                }
                None if self.held.is_empty() && self.lm.held() == 0 => {
                    aligned = true;
                    (0, 0)
                }
                None => (from, base),
            },
            None => (from, base),
        };
        let save_at = match (&self.prefix, aligned) {
            (Some(pc), true) => static_end(&self.chat, &msgs, &tools, &eff, &r.ids)
                .and_then(|l| pc.save_point(l))
                .filter(|&b| b > from && fits(b, pc.piece())),
            _ => None,
        };
        let room = limit.saturating_sub(base + (r.ids.len() - from) + 1);
        if room == 0 {
            send(refusal(
                400,
                format!(
                    "the prompt is {} tokens and leaves no room for an answer in a context of {limit} positions",
                    r.ids.len()
                ),
            ));
            return Ok(());
        }
        let max_new = match o.num_predict {
            Some(n) if n > 0 => (n as usize).min(room),
            _ => room,
        };
        if !send(Event::Accepted { prompt_tokens: r.ids.len(), cached: from }) {
            return Ok(());
        }

        let t_prompt = Instant::now();
        let mut logits = None;
        for p in pieces(&r.assistant_spans, from, r.ids.len()) {
            // The save point cuts the piece it falls in; the forward pieces stay those of the read without it.
            let parts = match save_at {
                Some(s) if p.start < s && s < p.end => vec![p.start..s, s..p.end],
                _ => vec![p.clone()],
            };
            for q in parts {
                logits = Some(self.lm.feed(&r.ids[q])?);
                if let (Some(s), Some(pc)) = (save_at, &self.prefix) {
                    if self.lm.held() == s {
                        self.save_prefix(pc, &r.ids, s);
                    }
                }
            }
            let ids = &r.ids[p];
            self.held.push(Held { keys: vec![ids.to_vec()], len: self.lm.held(), mark: self.lm.mark() });
        }
        let mut logits = logits.ok_or_else(|| anyhow::anyhow!("nothing of the prompt was fed"))?;
        let prompt_eval = t_prompt.elapsed();

        let seed = o.seed.unwrap_or_else(|| {
            SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(0)
        });
        let mut sampler = Sampler { temp, top_p, state: seed ^ 0x5EED_5EED_5EED_5EED };
        let end_sampling = self.chat.end_sampling;
        let t_eval = Instant::now();
        let mut parser = TurnParser::new(&self.chat);
        let mut out = Out { stops: &o.stop, text: String::new(), sent: 0 };
        let mut said: Vec<u32> = Vec::new();
        let end = loop {
            let next = pick(&logits, &mut sampler)?;
            let deltas = match parser.push_ids(&[next]) {
                Ok(d) => d,
                Err(e) => {
                    eprintln!("llvq: the model's turn broke the template's format, the answer ends there: {e:#}");
                    break End::Refused;
                }
            };
            if next == end_sampling {
                break End::Turn;
            }
            said.push(next);
            match out.deltas(deltas, send) {
                None => break End::Gone,
                Some(true) => break End::Stop,
                Some(false) => {}
            }
            if said.len() >= max_new {
                break End::Length;
            }
            logits = self.lm.feed(&[next])?;
        };
        if end == End::Gone {
            self.drop_answer()?;
            return Ok(());
        }
        let eval = t_eval.elapsed();
        if end != End::Stop && !out.flush(send) {
            self.drop_answer()?;
            return Ok(());
        }
        let done_reason = match end {
            End::Length => "length",
            _ => "stop",
        };
        send(Event::Done(Stats {
            done_reason: done_reason.into(),
            total_duration: nanos(total(job, started, load)),
            load_duration: nanos(load),
            prompt_eval_count: r.ids.len(),
            prompt_eval_cached_count: from,
            prompt_eval_duration: nanos(prompt_eval),
            eval_count: said.len(),
            eval_duration: nanos(eval),
        }));
        // A whole turn stays in the cache, closed as the template closes it,
        // keyed by what the client will send back. Anything else goes.
        let key = match end {
            End::Turn => answer_key(&self.chat, parser.parts()),
            _ => None,
        };
        match key {
            Some(key) => {
                self.lm.feed(&[end_sampling])?;
                self.held.push(Held { keys: vec![key], len: self.lm.held(), mark: self.lm.mark() });
            }
            None => self.drop_answer()?,
        }
        Ok(())
    }

    /// Load the longest saved prefix of `ids` longer than `keep`, the tokens
    /// the cache in memory keeps of them, and that `fits`. Its length, the
    /// cache then holding it as one held segment; `None` with nothing changed
    /// when there is none or every one is refused, or with the cache emptied
    /// when a load failed half way. Every refusal goes to the log by name.
    fn load_prefix(&mut self, pc: crate::prefix::PrefixCache, ids: &[u32], keep: usize, fits: &dyn Fn(usize) -> bool) -> Option<usize> {
        for f in pc.find(ids, keep, fits) {
            let t = Instant::now();
            let v = match pc.verify(&f, ids) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("llvq: prefix cache: {} refused, the prompt is read without it: {e:#}", f.name());
                    continue;
                }
            };
            let t_verify = t.elapsed();
            self.lm.reset();
            self.held.clear();
            match pc.load(&v, &mut |r| self.lm.load_state(r)) {
                Ok(()) if self.lm.held() == f.b => {
                    self.held.push(Held { keys: vec![ids[..f.b].to_vec()], len: f.b, mark: self.lm.mark() });
                    eprintln!(
                        "llvq: prefix cache: loaded {} tokens from {} in {:.2} s (checked in {:.2} s)",
                        f.b,
                        f.name(),
                        t.elapsed().as_secs_f64(),
                        t_verify.as_secs_f64()
                    );
                    return Some(f.b);
                }
                Ok(()) => {
                    eprintln!("llvq: prefix cache: {} loaded {} positions, not {}; the prompt is read from its start", f.name(), self.lm.held(), f.b);
                }
                Err(e) => {
                    eprintln!("llvq: prefix cache: {} failed to load, the prompt is read from its start: {e:#}", f.name());
                }
            }
            self.lm.reset();
            self.held.clear();
            return None;
        }
        None
    }

    /// Save the cache, which holds `ids[..b]`, as a prefix. A failure is
    /// logged and costs nothing else.
    fn save_prefix(&self, pc: &crate::prefix::PrefixCache, ids: &[u32], b: usize) {
        let t = Instant::now();
        match pc.save(ids, b, &mut |w| self.lm.save_state(w)) {
            Ok((path, bytes)) => eprintln!(
                "llvq: prefix cache: saved {b} tokens, {bytes} bytes, in {:.2} s: {}",
                t.elapsed().as_secs_f64(),
                path.file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default()
            ),
            Err(e) => eprintln!("llvq: prefix cache: the save of {b} tokens failed, nothing kept: {e:#}"),
        }
    }

    /// Back to the end of the prompt: the answer is not kept.
    fn drop_answer(&mut self) -> anyhow::Result<()> {
        let back = match self.held.last() {
            Some(h) => self.lm.rollback(&h.mark),
            None => {
                self.lm.reset();
                Ok(())
            }
        };
        if let Err(e) = back {
            // An answer longer than a bounded slot's slack: nothing is kept, the next prompt is read from its start.
            eprintln!("llvq: the cache could not drop the answer, it starts empty again: {e:#}");
            self.lm.reset();
            self.held.clear();
        }
        Ok(())
    }
}

/// The answer as it streams: reasoning at once, text up to a stop string,
/// each call whole.
struct Out<'a> {
    stops: &'a [String],
    /// The text since the last call.
    text: String,
    /// Bytes of `text` sent.
    sent: usize,
}

impl Out<'_> {
    /// Send what `deltas` made visible. `None` when the client left,
    /// `Some(true)` when a stop string ended the answer.
    fn deltas(&mut self, deltas: Vec<Delta>, send: &dyn Fn(Event) -> bool) -> Option<bool> {
        for d in deltas {
            let (content, thinking) = match d {
                Delta::Thinking(t) => (String::new(), t),
                Delta::Text(t) => {
                    self.text.push_str(&t);
                    let (upto, stop) = match chat::scan(&self.text, self.stops) {
                        Scan::Safe(n) => (n, false),
                        Scan::Stop(n) => (n, true),
                    };
                    let c = self.text.get(self.sent..upto.max(self.sent)).unwrap_or("").to_string();
                    self.sent = upto.max(self.sent);
                    if !c.is_empty() && !send(Event::Delta { content: c, thinking: String::new() }) {
                        return None;
                    }
                    if stop {
                        return Some(true);
                    }
                    continue;
                }
                Delta::ToolCall(c) => {
                    if !self.flush(send) {
                        return None;
                    }
                    self.text.clear();
                    self.sent = 0;
                    (inkling2::call_text(&c.name, &c.arguments), String::new())
                }
                Delta::End => continue,
            };
            if !send(Event::Delta { content, thinking }) {
                return None;
            }
        }
        Some(false)
    }

    /// Send the text held back for a stop string that never came. False
    /// when the client left.
    fn flush(&mut self, send: &dyn Fn(Event) -> bool) -> bool {
        let c = self.text[self.sent..].to_string();
        self.sent = self.text.len();
        c.is_empty() || send(Event::Delta { content: c, thinking: String::new() })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reuse_follows_the_held_keys_until_one_differs() {
        let a = vec![vec![1, 2, 3]];
        let b = vec![vec![9, 9], vec![4, 5]];
        let c = vec![vec![6]];
        let keys: Vec<&[Vec<u32>]> = vec![&a, &b, &c];
        assert_eq!(reuse(&keys, &[1, 2, 3, 4, 5, 6, 7]), vec![3, 2, 1]);
        assert_eq!(reuse(&keys, &[1, 2, 3, 4, 5, 8]), vec![3, 2]);
        assert_eq!(reuse(&keys, &[1, 2, 4]), Vec::<usize>::new());
        assert_eq!(reuse(&keys, &[1, 2, 3]), vec![3]);
        let empty = vec![Vec::new()];
        assert_eq!(reuse(&[&empty], &[1]), Vec::<usize>::new(), "an empty key never matches");
    }

    #[test]
    fn long_prompt_settings_follow_the_stack_not_only_cuda() {
        assert_eq!(long_prompt(Backend::Cuda, true), LongPrompt::BiasOnCard);
        assert_eq!(long_prompt(Backend::Metal, true), LongPrompt::BiasOnHost);
        assert_eq!(long_prompt(Backend::Cuda, false), LongPrompt::Off);
        assert_eq!(long_prompt(Backend::Metal, false), LongPrompt::Off);
        assert_eq!(long_prompt(Backend::Cpu, true), LongPrompt::Off);
    }

    #[test]
    fn pieces_cut_at_every_assistant_turn() {
        let spans = [4..7, 9..12];
        assert_eq!(pieces(&spans, 0, 15), vec![0..4, 4..7, 7..9, 9..12, 12..15]);
        assert_eq!(pieces(&spans, 7, 15), vec![7..9, 9..12, 12..15]);
        assert_eq!(pieces(&spans, 5, 12), vec![5..7, 7..9, 9..12]);
        assert_eq!(pieces(&[], 3, 8), vec![3..8]);
    }

    #[test]
    fn the_effort_is_the_clients_or_none() {
        assert_eq!(effort(None, false), Json::str("none"));
        assert_eq!(effort(None, true), Json::Null);
        assert_eq!(effort(Some("high"), false), Json::str("high"));
        assert_eq!(effort(Some("0.5"), false), Json::Float(0.5));
        assert_eq!(effort(Some("0"), false), Json::Int("0".into()));
    }

    #[test]
    fn a_local_file_reads_relative_paths_from_its_directory() {
        let dir = std::env::temp_dir().join(format!("llvq-local-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("inkling.json");
        std::fs::write(
            &p,
            r#"{"name": "inkling2-mini:local", "family": "inkling2", "config": "textonly/config.json",
                "checkpoint": "textonly/model.safetensors", "layers": "layers", "tokenizer": "native/tokenizer.json",
                "tables": "tables"}"#,
        )
        .unwrap();
        let m = LocalModel::read(&p).unwrap();
        assert_eq!(m.config, dir.join("textonly/config.json"));
        assert_eq!(m.tables, Some(dir.join("tables")));
        assert_eq!(m.fused(), Ok(true));
        std::fs::write(&p, r#"{"name": "x:y", "family": "qwen3", "config": "c", "checkpoint": "k", "layers": "l", "tokenizer": "t"}"#)
            .unwrap();
        assert!(format!("{:#}", LocalModel::read(&p).unwrap_err()).contains("inkling2"));
        std::fs::write(&p, r#"{"name": "x:y", "family": "inkling2", "config": "c", "checkpoint": "k", "layers": "l", "tokenizer": "t"}"#)
            .unwrap();
        assert!(format!("{:#}", LocalModel::read(&p).unwrap_err()).contains("tables"));
        // nothing on disk: registering refuses by name
        std::fs::write(&p, r#"{"name": "x:y", "family": "inkling2", "stack": "candle", "config": "c", "checkpoint": "k", "layers": "l", "tokenizer": "t"}"#)
            .unwrap();
        let store = Store::at(dir.join("home"));
        assert!(format!("{:#}", register(&store, &p).unwrap_err()).contains("config"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
