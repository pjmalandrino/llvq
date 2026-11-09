//! The engine: one OS thread per loaded model, jobs in, events out.
//!
//! The thread owns the `Qwen3`, its `Decoder`, the tokenizer and the cache of
//! the last conversation, and nothing else touches them. So the candle and
//! CUDA objects never cross into the async runtime, and their `Send` bounds
//! never come up: the model is built on the thread that uses it.
//!
//! By default the engine is batch 1. Requests to one model run one after
//! another, in arrival order, and `/api/ps` shows the queue. A client that
//! disconnects drops its event receiver; the next send fails and the answer
//! stops within one token on the eager path, one launch under 4E.
//!
//! Under `LLVQ_BATCH` on CUDA the engine batches continuously
//! ([`Worker::run_batch`]): a request takes a free slot of a pool as soon as
//! one exists, its prompt is prefilled alone, and every step decodes one token
//! for every live request at once. A request that ends gives its slot back
//! without stopping the others. A request alone decodes on the batch-1 path
//! (4E under a persistent config, the faster one for one sequence) and joins
//! the pool when a second one arrives. The batched engine is greedy, and it
//! keeps no conversation cache between requests.

use crate::chat::{self, Msg, Role, Scan, Segment, TextStream};
use crate::library::{Backend, ModelName};
use crate::store::{Manifest, Store};
use llvq_llm::chatfmt::{open_assistant, Marks};
use llvq_llm::model::Qwen3;
use llvq_llm::repl::{Sampler, CLOSE_LEN};
use llvq_llm::served_decode::Decoder;
use llvq_llm::session::{Finish, Flow, Generate, Session};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};

/// A model built and ready, on the thread that will run it.
pub struct Loaded {
    pub qwen: Qwen3,
    pub tokenizer: tokenizers::Tokenizer,
    pub decoder: Decoder,
    /// `cuda`, `metal` or `cpu` (`dense-cpu` for a test model).
    pub backend: String,
    /// The served config's file name, `None` for a test model.
    pub config: Option<String>,
    /// Lines the load printed (4E's install), kept for `/api/show`.
    pub lines: Vec<String>,
    /// The batched decode, when the server runs under `LLVQ_BATCH`.
    pub batch: Option<BatchEngine>,
}

/// What the batched engine steps with: the device's kernels, the kind of
/// projection, and the pool's shape.
pub struct BatchEngine {
    pub dev: Box<dyn llvq_llm::batch::BatchDevice>,
    pub projs: llvq_llm::batch::BatchProjs,
    pub serve: llvq_llm::served::BatchServe,
}

/// `LLVQ_BATCH`: `max` or `max:context`, the pool `llvq serve` batches into.
/// `None` when unset or `0`, the batch-1 engine.
pub fn batch_from_env(v: Option<String>) -> anyhow::Result<Option<(usize, Option<usize>)>> {
    let Some(v) = v else { return Ok(None) };
    let v = v.trim();
    let bad = || anyhow::anyhow!("LLVQ_BATCH={v:?}: a number of sequences, or sequences:positions, such as 64 or 64:4096");
    let (m, c) = match v.split_once(':') {
        Some((m, c)) => (m, Some(c.trim().parse::<usize>().map_err(|_| bad())?)),
        None => (v, None),
    };
    match m.trim().parse::<usize>().map_err(|_| bad())? {
        0 if c.is_none() => Ok(None),
        0 => Err(bad()),
        n => Ok(Some((n, c))),
    }
}

/// Builds a model from a manifest. Called on the engine's own thread.
pub trait Loader: Send + Sync + 'static {
    fn load(&self, name: &ModelName, m: &Manifest, store: &Store) -> anyhow::Result<Loaded>;
}

/// The loader of this build: the fused backend under the library's served
/// config, on CUDA, Metal, or the CPU anywhere else. On the CPU the codes stay
/// encoded too (`llvq_llm::fused_cpu`); nothing reconstructs dense weights.
pub struct DeviceLoader;

impl Loader for DeviceLoader {
    fn load(&self, name: &ModelName, m: &Manifest, store: &Store) -> anyhow::Result<Loaded> {
        use candle_core::DType;
        let path = store.blob_path(&m.sha256);
        let path_s = path.to_str().ok_or_else(|| anyhow::anyhow!("{}: not UTF-8", path.display()))?;
        let backend = Backend::of_build();
        let configs = crate::library::with_cpu_config(&m.configs, &m.sha256);
        let batch = batch_from_env(std::env::var("LLVQ_BATCH").ok())?;
        let config = match (batch, backend) {
            (None, _) => backend.config(&configs).map_err(|e| anyhow::anyhow!("{name}: {e}"))?,
            (Some(_), Backend::Cuda) => Some(
                configs
                    .cuda_batch
                    .as_deref()
                    .ok_or_else(|| anyhow::anyhow!("{name}: LLVQ_BATCH is set and this model has no batch config"))?,
            ),
            (Some(_), b) => anyhow::bail!(
                "LLVQ_BATCH is set on {}: the batched decode's kernels are CUDA. Unset it to serve one request at a time.",
                b.label()
            ),
        };
        let config = config.ok_or_else(|| anyhow::anyhow!("{name}: no served config on {}", backend.label()))?;
        let text = crate::library::config_text(config)
            .ok_or_else(|| anyhow::anyhow!("{name}: config {config} is not compiled into this binary"))?;
        let file: llvq_llm::served::ServedFile =
            serde_json::from_str(text).map_err(|e| anyhow::anyhow!("{config}: {e}"))?;
        // The served door refuses a measurement-mode variable beside a
        // config; the runtime is a served door.
        llvq_llm::served::check_env(&file, |v| std::env::var(v).ok())
            .map_err(|e| anyhow::anyhow!("{config}: {e}"))?;
        let cfg = llvq_llm::served::Served::of(file, config.into()).map_err(anyhow::Error::msg)?;
        let device = match backend {
            Backend::Cuda => candle_core::Device::new_cuda(0)?,
            Backend::Metal => candle_core::Device::new_metal(0)?,
            Backend::Cpu => candle_core::Device::Cpu,
        };
        // f32 on the CPU: candle's f16 ops there moved a 4,096-token window
        // by -0.47 % against the dense reference, f32 by -0.002 %, and f32 is
        // the faster of the two (`docs/mesures/cpu-tetra-2026-11-07.txt`).
        let dtype = match backend {
            Backend::Cpu => DType::F32,
            Backend::Cuda | Backend::Metal => DType::F16,
        };
        let mut f = llvq_llm::served::load_resolved(
            path_s,
            &device,
            dtype,
            cfg.layout,
            cfg.embed,
            cfg.rot_share,
            cfg.fuse,
            cfg.kv,
            cfg.int4_layout(),
            Some("the llvq library"),
        )?;
        let decoder = cfg.serve(&mut f.model, &device)?;
        let mut lines = decoder.lines().to_vec();
        let batch = match batch {
            None => None,
            Some((max, context)) => {
                let serve = cfg.batch.ok_or_else(|| anyhow::anyhow!("{config}: names no batch"))?;
                let serve = match context {
                    Some(c) => llvq_llm::served::BatchServe::of(llvq_llm::served::BatchFile { max, context: c }),
                    None => serve.with_max(max),
                }
                .map_err(|e| anyhow::anyhow!("LLVQ_BATCH: {e}"))?;
                Some(batch_engine(&f.model, serve, &mut lines)?)
            }
        };
        Ok(Loaded {
            qwen: f.model,
            tokenizer: f.tokenizer,
            decoder,
            backend: backend.label().into(),
            config: Some(config.to_string()),
            lines,
            batch,
        })
    }
}

/// The CUDA kernels of the batched decode, on the runtime the model lives on.
#[cfg(all(target_os = "linux", feature = "cuda"))]
fn batch_engine(
    model: &Qwen3,
    serve: llvq_llm::served::BatchServe,
    lines: &mut Vec<String>,
) -> anyhow::Result<BatchEngine> {
    let (_, groups, _) = model.attn_geometry();
    let rt = llvq_llm::batch_cuda::runtime_of(model)?;
    let dev = llvq_llm::batch_cuda::CudaBatch::new(rt, groups)?;
    lines.extend(dev.provenance.iter().cloned());
    lines.push(format!("batch: up to {} sequences of {} positions a step", serve.max, serve.context));
    Ok(BatchEngine { dev: Box::new(dev), projs: llvq_llm::batch::BatchProjs::Gemm, serve })
}

#[cfg(not(all(target_os = "linux", feature = "cuda")))]
fn batch_engine(_: &Qwen3, _: llvq_llm::served::BatchServe, _: &mut Vec<String>) -> anyhow::Result<BatchEngine> {
    anyhow::bail!("LLVQ_BATCH: this build has no CUDA kernels for the batched decode")
}

/// What a request asks the model to read.
#[derive(Debug, Clone)]
pub enum Prompt {
    /// `/api/chat`: messages through Qwen3's template.
    Chat(Vec<Msg>),
    /// `/api/generate` with the template: an optional system turn and a user
    /// turn.
    Templated { system: Option<String>, prompt: String },
    /// `/api/generate` with `raw`: the text as it is.
    Raw(String),
    /// An empty `/api/generate`: load the model, answer nothing.
    Load,
}

/// The options the engine reads. Anything else a client sends is listed in
/// the response's `warnings` (decision D3).
#[derive(Debug, Clone, Default)]
pub struct Options {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub seed: Option<u64>,
    /// Tokens at most; zero or negative is "until the context ends".
    pub num_predict: Option<i64>,
    pub stop: Vec<String>,
    pub num_ctx: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct Request {
    pub prompt: Prompt,
    pub think: bool,
    pub options: Options,
}

/// The timings and counts of a finished answer, Ollama's last chunk.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Stats {
    /// `stop`, `length`, `load` or `unload`.
    pub done_reason: String,
    pub total_duration: u64,
    pub load_duration: u64,
    pub prompt_eval_count: usize,
    pub prompt_eval_cached_count: usize,
    pub prompt_eval_duration: u64,
    pub eval_count: usize,
    pub eval_duration: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Event {
    /// The request passed its checks; the prefill starts.
    Accepted { prompt_tokens: usize, cached: usize },
    /// New text of the answer and of its reasoning.
    Delta { content: String, thinking: String },
    Done(Stats),
    /// Refused or failed. `status` is the HTTP status the server answers
    /// with when nothing was streamed yet.
    Error { status: u16, message: String },
}

pub type Events = tokio::sync::mpsc::Receiver<Event>;

struct Job {
    request: Request,
    events: tokio::sync::mpsc::Sender<Event>,
    keep_alive: KeepAlive,
    queued_at: Instant,
}

/// What `/api/ps` reads about an engine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum State {
    Loading,
    Ready {
        backend: String,
        config: Option<String>,
        context: usize,
        greedy_only: bool,
        /// The batch pool's KV caches, allocated beside the weights; 0
        /// without `LLVQ_BATCH`.
        pool_bytes: u64,
    },
    Failed(String),
}

pub struct EngineInfo {
    pub name: ModelName,
    pub manifest: Manifest,
    pub state: Mutex<State>,
    /// Jobs waiting, not counting the one running.
    pub queue: AtomicUsize,
    pub busy: std::sync::atomic::AtomicBool,
    /// When the engine may be unloaded; `None` is never.
    pub expires: Mutex<Option<SystemTime>>,
    /// Asked to unload while busy: it goes as soon as it is idle, and a new
    /// request for the same model takes it back.
    pub retiring: std::sync::atomic::AtomicBool,
}

impl EngineInfo {
    /// The worker takes a job off the queue: busy first, then one less
    /// queued, so that no reader ever sees the engine idle in between.
    fn take_job(&self) {
        self.busy.store(true, Ordering::SeqCst);
        self.queue.fetch_sub(1, Ordering::SeqCst);
    }

    /// Nothing queued and nothing running.
    fn idle(&self) -> bool {
        self.queue.load(Ordering::SeqCst) == 0 && !self.busy.load(Ordering::SeqCst)
    }
}

pub struct Engine {
    tx: std::sync::mpsc::Sender<Job>,
    pub info: Arc<EngineInfo>,
}

/// How long a model stays loaded after its last request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeepAlive {
    For(Duration),
    Forever,
}

impl KeepAlive {
    /// Ollama's forms: a number of seconds, or a duration string such as
    /// `"5m"`, `"1h30m"`, `"10s"`, `"0"`; a negative value keeps the model
    /// loaded forever.
    pub fn parse(v: &serde_json::Value) -> Result<Self, String> {
        match v {
            serde_json::Value::Number(n) => {
                let s = n.as_f64().ok_or("keep_alive: not a number")?;
                match s < 0.0 {
                    true => Ok(KeepAlive::Forever),
                    false => secs(s),
                }
            }
            serde_json::Value::String(s) => Self::parse_str(s),
            _ => Err("keep_alive: a number of seconds or a duration such as \"5m\"".into()),
        }
    }

    pub fn parse_str(s: &str) -> Result<Self, String> {
        let s = s.trim();
        if s.starts_with('-') {
            return Ok(KeepAlive::Forever);
        }
        if let Ok(n) = s.parse::<f64>() {
            return secs(n);
        }
        let mut total = 0f64;
        let mut num = String::new();
        let mut chars = s.chars().peekable();
        while let Some(c) = chars.next() {
            if c.is_ascii_digit() || c == '.' {
                num.push(c);
                continue;
            }
            let unit = match (c, chars.peek()) {
                ('m', Some('s')) => {
                    chars.next();
                    0.001
                }
                ('h', _) => 3600.0,
                ('m', _) => 60.0,
                ('s', _) => 1.0,
                _ => return Err(format!("keep_alive {s:?}: units are h, m, s and ms")),
            };
            let n: f64 = num.parse().map_err(|_| format!("keep_alive {s:?}: a number before each unit"))?;
            total += n * unit;
            num.clear();
        }
        if !num.is_empty() {
            return Err(format!("keep_alive {s:?}: a unit after the last number"));
        }
        secs(total)
    }

    /// When the model may go, `None` for never. A duration past what the
    /// clock holds is never.
    pub fn until(self, now: SystemTime) -> Option<SystemTime> {
        match self {
            KeepAlive::For(d) => now.checked_add(d),
            KeepAlive::Forever => None,
        }
    }
}

fn secs(s: f64) -> Result<KeepAlive, String> {
    Duration::try_from_secs_f64(s).map(KeepAlive::For).map_err(|_| format!("keep_alive: {s} seconds is not a duration"))
}

/// The sampler's defaults when a request names none. Greedy under 4E, which
/// only decodes greedily (decision D2); `bin/chat`'s defaults otherwise.
pub fn default_temperature(greedy_only: bool) -> f32 {
    match greedy_only {
        true => 0.0,
        false => 0.7,
    }
}
pub const DEFAULT_TOP_P: f32 = 0.9;

/// From the request to now. A job queued while the model loaded has waited
/// the load already, so the load is added only to what the job did not wait.
fn total(job: &Job, started: Instant, load: Duration) -> Duration {
    job.queued_at.elapsed().max(started.elapsed() + load)
}

fn nanos(d: Duration) -> u64 {
    d.as_nanos().min(u64::MAX as u128) as u64
}

/// The conversation state the thread carries between jobs.
struct Worker {
    loaded: Loaded,
    marks: Marks,
    session: Session,
    memo: Vec<Segment>,
    context: usize,
    load_duration: Option<Duration>,
}

/// A request's prompt as ids, checked against the context, and what the
/// answer may take.
struct Prepared {
    prompt: Vec<u32>,
    segments: Vec<Segment>,
    /// Whether the prompt ends on an open assistant turn (chat, templated).
    close: bool,
    /// Where the messages end in `prompt`, before the open assistant turn.
    messages_end: usize,
    max_new: usize,
    temp: f32,
    top_p: f32,
}

/// The answer's tokens, routed to the content or the reasoning by the think
/// tokens, and streamed as text up to a stop string.
#[derive(Default)]
struct Router {
    content: TextStream,
    thinking: TextStream,
    routed: usize,
    in_think: bool,
    /// Where a stop string started in the content, once one did.
    stopped_at: Option<usize>,
}

impl Router {
    /// Route the tokens of `said` not yet routed, and return the text that is
    /// safe to send now, and whether a stop string ended the content.
    fn step(
        &mut self,
        said: &[u32],
        tok: &tokenizers::Tokenizer,
        think: Option<(u32, u32)>,
        stop: &[String],
    ) -> anyhow::Result<(String, String, bool)> {
        for &t in &said[self.routed..] {
            match think {
                Some((open, _)) if t == open => self.in_think = true,
                Some((_, close)) if t == close => self.in_think = false,
                _ if self.in_think => self.thinking.ids.push(t),
                _ => self.content.ids.push(t),
            }
        }
        self.routed = said.len();
        let ftext = self.content.full(tok)?;
        let (upto, stop) = match chat::scan(&ftext, stop) {
            Scan::Safe(n) => (n, false),
            Scan::Stop(n) => (n, true),
        };
        if stop {
            self.stopped_at = Some(upto);
        }
        let c = self.content.take(&ftext, upto);
        let ttext = self.thinking.full(tok)?;
        let tsafe = llvq_llm::repl::printable(&ttext).len();
        let r = self.thinking.take(&ttext, tsafe);
        Ok((c, r, stop))
    }

    /// The held tail, when no stop string came: whatever is left of the
    /// decoded text is the answer's.
    fn tail(&mut self, tok: &tokenizers::Tokenizer) -> anyhow::Result<(String, String)> {
        let ftext = self.content.full(tok)?;
        let c = self.content.take(&ftext, ftext.len());
        let ttext = self.thinking.full(tok)?;
        let r = self.thinking.take(&ttext, ttext.len());
        Ok((c, r))
    }
}

/// One request of the batched engine, from its admission to its last token.
struct Live {
    job: Job,
    slot: usize,
    router: Router,
    said: Vec<u32>,
    /// The token the next step feeds.
    next: u32,
    max_new: usize,
    started: Instant,
    load: Duration,
    /// The prompt's ids: the positions before the answer, for a request
    /// that goes back from the pool to the batch-1 path.
    prompt: Vec<u32>,
    prompt_eval: Duration,
    t_eval: Instant,
}

/// Tokens the batch-1 path may still answer from `held` positions under a
/// cap of `max` (`None`: no cap): a launch from position p of n tokens needs
/// p + n positions, and the answer's last token is never fed.
fn solo_room(max: Option<usize>, held: usize) -> usize {
    max.map_or(usize::MAX, |m| m.saturating_sub(held))
}

/// How a token left a live request.
enum After {
    Continue,
    /// Done, with Ollama's reason.
    Done(&'static str),
    /// The client left.
    Gone,
}

impl Worker {
    fn greedy_only(&self) -> bool {
        self.loaded.decoder.is_persistent() || self.loaded.batch.is_some()
    }

    fn error(status: u16, message: impl Into<String>) -> Event {
        Event::Error { status, message: message.into() }
    }

    /// One request, start to end. Returns nothing: the outcome goes out as
    /// events, and a send that fails means the client left.
    fn run(&mut self, job: &Job) {
        let started = Instant::now();
        let tx = &job.events;
        // A client that left while its job was queued: running it would
        // still cut the cache down to its prompt, for an answer nobody reads.
        if tx.is_closed() {
            return;
        }
        let send = |e: Event| tx.blocking_send(e).is_ok();
        let load_duration = self.load_duration.take().unwrap_or_default();
        if let Err(e) = self.run_inner(job, started, load_duration, &send) {
            send(Self::error(500, format!("{e:#}")));
        }
    }

    /// The checks and the prompt of a request, or the refusal to send.
    /// `memo` is the conversation the cache holds (empty for the batched
    /// engine, which keeps none).
    fn prepare(&self, req: &Request, memo: &[Segment]) -> anyhow::Result<Result<Prepared, Event>> {
        let o = &req.options;
        let greedy_only = self.greedy_only();
        let temp = o.temperature.unwrap_or(default_temperature(greedy_only));
        if !(0.0..=100.0).contains(&temp) {
            return Ok(Err(Self::error(400, format!("temperature {temp}: between 0 and 100"))));
        }
        if greedy_only && temp > 0.0 {
            let why = match self.loaded.batch.is_some() {
                true => "this model is served with the batched decode (LLVQ_BATCH), which takes the argmax on the \
                         card and cannot sample",
                false => "this model is served with the persistent decode (4E), which takes the argmax on the \
                          card and cannot sample",
            };
            return Ok(Err(Self::error(
                400,
                format!("temperature {temp}: {why}. Send temperature 0, or leave it out."),
            )));
        }
        let top_p = o.top_p.unwrap_or(DEFAULT_TOP_P);
        if !(0.0..=1.0).contains(&top_p) {
            return Ok(Err(Self::error(400, format!("top_p {top_p}: between 0 and 1"))));
        }
        let limit = match o.num_ctx {
            Some(n) if n > self.context => {
                return Ok(Err(Self::error(
                    400,
                    format!("num_ctx {n}: this model serves at most {} positions here", self.context),
                )));
            }
            Some(0) => return Ok(Err(Self::error(400, "num_ctx 0: a context of no position"))),
            Some(n) => n,
            None => self.context,
        };

        // The prompt, as ids. For a chat, segment by segment, so the next
        // request finds the ids this one leaves in the cache.
        let tok = &self.loaded.tokenizer;
        let marks = &self.marks;
        let (segments, close) = match &req.prompt {
            Prompt::Chat(msgs) => (chat::render(tok, marks, memo, msgs)?, true),
            Prompt::Templated { system, prompt } => {
                let mut msgs = Vec::new();
                if let Some(s) = system {
                    msgs.push(Msg { role: Role::System, content: s.clone() });
                }
                msgs.push(Msg { role: Role::User, content: prompt.clone() });
                (chat::render(tok, marks, memo, &msgs)?, true)
            }
            Prompt::Raw(text) => {
                let ids = tok.encode(text.as_str(), false).map_err(|e| anyhow::anyhow!("{e}"))?.get_ids().to_vec();
                (vec![Segment { msg: None, ids }], false)
            }
            Prompt::Load => unreachable!("answered before"),
        };
        let mut prompt: Vec<u32> = segments.iter().flat_map(|s| s.ids.iter().copied()).collect();
        let messages_end = prompt.len();
        if close {
            prompt.extend(open_assistant(tok, marks, req.think)?);
        }
        if prompt.is_empty() {
            return Ok(Err(Self::error(400, "the prompt is empty")));
        }
        let room = limit.saturating_sub(prompt.len() + CLOSE_LEN);
        if room == 0 {
            return Ok(Err(Self::error(
                400,
                format!(
                    "the prompt is {} tokens and leaves no room for an answer in a context of {limit} positions",
                    prompt.len()
                ),
            )));
        }
        let max_new = match req.options.num_predict {
            Some(n) if n > 0 => (n as usize).min(room),
            _ => room,
        };
        Ok(Ok(Prepared { prompt, segments, close, messages_end, max_new, temp, top_p }))
    }

    fn run_inner(
        &mut self,
        job: &Job,
        started: Instant,
        load_duration: Duration,
        send: &dyn Fn(Event) -> bool,
    ) -> anyhow::Result<()> {
        let req = &job.request;
        let o = &req.options;
        if let Prompt::Load = req.prompt {
            send(Event::Accepted { prompt_tokens: 0, cached: 0 });
            send(Event::Done(Stats {
                done_reason: "load".into(),
                total_duration: nanos(total(job, started, load_duration)),
                load_duration: nanos(load_duration),
                ..Stats::default()
            }));
            return Ok(());
        }
        let Prepared { prompt, segments, close, messages_end, max_new, temp, top_p } =
            match self.prepare(req, &self.memo)? {
                Ok(p) => p,
                Err(refusal) => {
                    send(refusal);
                    return Ok(());
                }
            };
        let tok = &self.loaded.tokenizer;
        let marks = &self.marks;

        let cached = self.session.reuse_prefix(&prompt)?;
        if !send(Event::Accepted { prompt_tokens: prompt.len(), cached }) {
            return Ok(());
        }
        // The memo now describes the cache up to the messages; the answer's
        // segment is added only if the answer ends cleanly.
        self.memo = segments;

        let t_prompt = Instant::now();
        let h = self.session.feed(&self.loaded.qwen, &prompt[cached..])?;
        let prompt_eval = t_prompt.elapsed();

        let seed = o.seed.unwrap_or_else(|| {
            SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(0)
        });
        let mut sampler = Sampler { temp, top_p, state: seed ^ 0x5EED_5EED_5EED_5EED };
        let stops = marks.stops();
        let think = marks.think;

        // Route each token to the answer or to its reasoning, by the think
        // tokens, and stream each as text.
        let mut router = Router::default();
        let mut gone = false;
        let mut failure: Option<anyhow::Error> = None;
        let t_eval = Instant::now();
        let answer = self.session.answer(
            &self.loaded.qwen,
            &mut self.loaded.decoder,
            h,
            Generate { sampler: &mut sampler, max_new, stops: &stops },
            &mut |said| match router.step(said, tok, think, &o.stop) {
                Err(e) => {
                    failure = Some(e);
                    Flow::Cancel
                }
                Ok((c, r, stop)) => {
                    if (!c.is_empty() || !r.is_empty()) && !send(Event::Delta { content: c, thinking: r }) {
                        gone = true;
                        return Flow::Cancel;
                    }
                    match stop {
                        true => Flow::Cancel,
                        false => Flow::Continue,
                    }
                }
            },
        )?;
        let eval = t_eval.elapsed();
        if let Some(e) = failure {
            return Err(e);
        }
        if gone {
            return Ok(());
        }
        // The held tail: a stop string never came, so whatever is left of the
        // decoded text is the answer's.
        let stopped_at = router.stopped_at;
        if stopped_at.is_none() {
            let (c, r) = router.tail(tok)?;
            if (!c.is_empty() || !r.is_empty()) && !send(Event::Delta { content: c, thinking: r }) {
                return Ok(());
            }
        }
        let done_reason = match (answer.finish, stopped_at) {
            (_, Some(_)) | (Finish::Stop, None) => "stop",
            (Finish::Length, None) => "length",
            (Finish::Cancelled, None) => unreachable!("only a stop string or a gone client cancels"),
        };
        // The client has the whole answer: it is told so before the cache is
        // closed below, which costs a step of the model it would otherwise
        // wait for. A request it sends now queues behind that step.
        send(Event::Done(Stats {
            done_reason: done_reason.into(),
            total_duration: nanos(total(job, started, load_duration)),
            load_duration: nanos(load_duration),
            prompt_eval_count: prompt.len(),
            prompt_eval_cached_count: cached,
            prompt_eval_duration: nanos(prompt_eval),
            eval_count: answer.said.len(),
            eval_duration: nanos(eval),
        }));
        // A clean end closes the assistant's message in the cache, and the
        // memo learns its ids. A stop string leaves the cache mid-message:
        // the next request re-reads from the last message boundary.
        if close && stopped_at.is_none() {
            // On a length cut `Session::answer` stops before feeding its last
            // token; the client shows it, so the cache must hold it too, or the
            // memo would replay the answer one token short.
            let mut tail = Vec::with_capacity(3);
            if answer.finish == Finish::Length {
                tail.extend(answer.said.last().copied());
            }
            tail.extend([marks.im_end, marks.nl]);
            if let Err(e) = self.session.feed(&self.loaded.qwen, &tail) {
                // Too late for the client, which has its answer: the next
                // request starts from an empty cache rather than a doubtful one.
                eprintln!("llvq: closing the answer in the cache failed, the cache is reset: {e:#}");
                self.session.reset(&self.loaded.qwen);
                self.memo.clear();
                return Ok(());
            }
            self.memo.push(Segment {
                msg: Some(Msg { role: Role::Assistant, content: router.content.sent.clone() }),
                ids: self.session.ids()[messages_end..].to_vec(),
            });
        }
        Ok(())
    }

    /// The batched engine: jobs from `rx` into the slots of a pool, one token
    /// a step for every live request, until the channel closes.
    ///
    /// A job waits in the queue (`/api/ps` counts it) until a slot is free.
    /// Its prompt is then prefilled alone, on the eager path, while the live
    /// requests wait one prefill; its first token comes from that prefill,
    /// and from the next step on it decodes with the others. Each pick is
    /// routed and streamed as in [`Self::run_inner`]; a request that ends
    /// (stop token, stop string, its cap, or a client that left) gives its
    /// slot back at once.
    fn run_batch(&mut self, rx: &std::sync::mpsc::Receiver<Job>, inf: &EngineInfo) {
        let serve = self.loaded.batch.as_ref().expect("the batched engine").serve;
        let mut pool = match llvq_llm::batch::Pool::new(&self.loaded.qwen, serve.max, serve.context) {
            Ok(p) => p,
            Err(e) => {
                let msg = format!("the batch pool of {} slots of {} positions: {e}", serve.max, serve.context);
                *inf.state.lock().expect("state lock") = State::Failed(msg.clone());
                while let Ok(job) = rx.recv() {
                    inf.queue.fetch_sub(1, Ordering::SeqCst);
                    let _ = job.events.blocking_send(Event::Error { status: 500, message: msg.clone() });
                }
                return;
            }
        };
        let mut pending: VecDeque<Job> = VecDeque::new();
        let mut live: Vec<Live> = Vec::new();
        let mut open = true;
        loop {
            if live.is_empty() && pending.is_empty() {
                inf.busy.store(false, Ordering::SeqCst);
                if !open {
                    return;
                }
                match rx.recv() {
                    Ok(j) => pending.push_back(j),
                    Err(_) => return,
                }
            }
            loop {
                match rx.try_recv() {
                    Ok(j) => pending.push_back(j),
                    Err(std::sync::mpsc::TryRecvError::Empty) => break,
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                        open = false;
                        break;
                    }
                }
            }
            while pool.free() > 0 {
                if live.is_empty() && pending.len() == 1 {
                    // Alone: the batch-1 path, until another request arrives.
                    let job = pending.pop_front().expect("one pending");
                    inf.take_job();
                    let keep = job.keep_alive;
                    match self.solo(job, rx, &mut pending, &mut open, &mut pool, inf) {
                        Some(l) => live.push(l),
                        None => *inf.expires.lock().expect("expires lock") = keep.until(SystemTime::now()),
                    }
                    continue;
                }
                let Some(job) = pending.pop_front() else { break };
                inf.take_job();
                let keep = job.keep_alive;
                if let Some(l) = self.admit(job, &mut pool) {
                    live.push(l);
                } else {
                    *inf.expires.lock().expect("expires lock") = keep.until(SystemTime::now());
                }
            }
            if live.is_empty() {
                continue;
            }
            let step: Vec<(usize, u32)> = live.iter().map(|l| (l.slot, l.next)).collect();
            let be = self.loaded.batch.as_ref().expect("the batched engine");
            let picks = match pool.step(&self.loaded.qwen, &step, be.dev.as_ref(), be.projs) {
                Ok(p) => p,
                Err(e) => {
                    // A failed step leaves every live cache in doubt: each
                    // request is answered with the reason and its slot freed.
                    for l in live.drain(..) {
                        let _ = l.job.events.blocking_send(Self::error(500, format!("the batched step failed: {e:#}")));
                        pool.release(l.slot);
                    }
                    continue;
                }
            };
            let mut ended: Vec<(usize, After)> = Vec::new();
            for (i, (l, pick)) in live.iter_mut().zip(picks).enumerate() {
                match self.advance(l, pick) {
                    After::Continue => {}
                    a => ended.push((i, a)),
                }
            }
            for (i, a) in ended.into_iter().rev() {
                let mut l = live.swap_remove(i);
                pool.release(l.slot);
                if let After::Done(reason) = a {
                    self.finish(&mut l, reason);
                }
                *inf.expires.lock().expect("expires lock") = l.job.keep_alive.until(SystemTime::now());
            }
            if live.len() == 1 && pending.is_empty() && self.back_to_solo(&live[0]) {
                let l = live.pop().expect("one live");
                let keep = l.job.keep_alive;
                match self.resume(l, rx, &mut pending, &mut open, &mut pool, inf) {
                    Ok(Some(l)) => live.push(l),
                    Ok(None) => *inf.expires.lock().expect("expires lock") = keep.until(SystemTime::now()),
                    Err(l) => live.push(*l),
                }
            }
        }
    }

    /// A request with no other live: prefilled and decoded on the batch-1
    /// path ([`Session::answer`], 4E under a persistent config). `None` when
    /// it ended there; otherwise it is returned live in a slot of the pool,
    /// its next token not yet fed (see [`Self::solo_decode`]).
    #[allow(clippy::too_many_arguments)]
    fn solo(
        &mut self,
        job: Job,
        rx: &std::sync::mpsc::Receiver<Job>,
        pending: &mut VecDeque<Job>,
        open: &mut bool,
        pool: &mut llvq_llm::batch::Pool,
        inf: &EngineInfo,
    ) -> Option<Live> {
        let started = Instant::now();
        let load = self.load_duration.take().unwrap_or_default();
        let tx = job.events.clone();
        if tx.is_closed() {
            return None;
        }
        let send = |e: Event| tx.blocking_send(e).is_ok();
        if let Prompt::Load = job.request.prompt {
            Self::answer_load(&job, started, load);
            return None;
        }
        let p = match self.prepare(&job.request, &[]) {
            Ok(Ok(p)) => p,
            Ok(Err(refusal)) => {
                send(refusal);
                return None;
            }
            Err(e) => {
                send(Self::error(500, format!("{e:#}")));
                return None;
            }
        };
        let mut session = Session::new(&self.loaded.qwen);
        let t = Instant::now();
        let h = match session.feed(&self.loaded.qwen, &p.prompt) {
            Ok(h) => h,
            Err(e) => {
                send(Self::error(500, format!("{e:#}")));
                return None;
            }
        };
        let prompt_eval = t.elapsed();
        if !send(Event::Accepted { prompt_tokens: p.prompt.len(), cached: 0 }) {
            return None;
        }
        let l = Live {
            job,
            slot: usize::MAX,
            router: Router::default(),
            said: Vec::new(),
            next: 0,
            max_new: p.max_new,
            started,
            load,
            prompt: p.prompt,
            prompt_eval,
            t_eval: Instant::now(),
        };
        self.solo_decode(l, session, h, rx, pending, open, pool, inf)
    }

    /// Whether the lone live request of the pool goes back to the batch-1
    /// path: it decodes there at 225.4 tok/s on 4E against 106.2 on the
    /// batched step at B = 1 (4B, batch-l40s-v4), so a request that outlived
    /// its neighbours ran its tail at 0.47x (audit 2026-11-08 §1.2). Not when
    /// the batch-1 path has fewer positions left than one launch.
    fn back_to_solo(&self, l: &Live) -> bool {
        let held = l.prompt.len() + l.said.len();
        solo_room(self.loaded.decoder.max_positions(), held) > self.loaded.decoder.launch_tokens() as usize
    }

    /// [`Self::solo`] for a request taken out of the pool: its slot's caches
    /// copied back into a `Session`, its next token fed. `Err` gives the
    /// request back untouched in its slot, which keeps decoding in the pool.
    #[allow(clippy::too_many_arguments)]
    fn resume(
        &mut self,
        mut l: Live,
        rx: &std::sync::mpsc::Receiver<Job>,
        pending: &mut VecDeque<Job>,
        open: &mut bool,
        pool: &mut llvq_llm::batch::Pool,
        inf: &EngineInfo,
    ) -> Result<Option<Live>, Box<Live>> {
        let fed = l.said.len().saturating_sub(1);
        let ids: Vec<u32> = l.prompt.iter().chain(&l.said[..fed]).copied().collect();
        let back = pool
            .caches_of(&self.loaded.qwen, l.slot)
            .map_err(anyhow::Error::from)
            .and_then(|c| Session::from_caches(&self.loaded.qwen, c, ids))
            .and_then(|mut s| s.feed(&self.loaded.qwen, &[l.next]).map(|h| (s, h)));
        match back {
            Ok((session, h)) => {
                pool.release(l.slot);
                l.slot = usize::MAX;
                Ok(self.solo_decode(l, session, h, rx, pending, open, pool, inf))
            }
            Err(e) => {
                eprintln!("llvq: a lone request stays in the pool, its caches did not come back: {e:#}");
                Err(Box::new(l))
            }
        }
    }

    /// The batch-1 decode of `l` from `h`, the hidden state after the last
    /// position `session` holds. It stops between two launches as soon as
    /// another request arrives (it goes to `pending`), or when the batch-1
    /// path's positions run out (4E holds [`llvq_llm::served_decode::P4E_MAX_CAP`],
    /// the pool may hold more, audit 2026-11-08 §1.7). The conversation then
    /// takes a slot of the pool with its caches and is returned live, its
    /// next token not yet fed. `None` when it ended here.
    ///
    /// A job that only loads the model is answered at once and does not stop
    /// the decode: the model is loaded.
    #[allow(clippy::too_many_arguments)]
    fn solo_decode(
        &mut self,
        mut l: Live,
        mut session: Session,
        h: candle_core::Tensor,
        rx: &std::sync::mpsc::Receiver<Job>,
        pending: &mut VecDeque<Job>,
        open: &mut bool,
        pool: &mut llvq_llm::batch::Pool,
        inf: &EngineInfo,
    ) -> Option<Live> {
        let tx = l.job.events.clone();
        let send = |e: Event| tx.blocking_send(e).is_ok();
        let tok = &self.loaded.tokenizer;
        let (think, stops) = (self.marks.think, self.marks.stops());
        let stop_strings = l.job.request.options.stop.clone();
        let mut router = std::mem::take(&mut l.router);
        let before = l.said.clone();
        let budget = l.max_new.saturating_sub(before.len());
        // The first token comes from `h` on the eager path, never a launch.
        let max_new = budget.min(solo_room(self.loaded.decoder.max_positions(), session.len())).max(1);
        let (mut gone, mut arrived) = (false, false);
        let mut failure: Option<anyhow::Error> = None;
        let mut sampler = Sampler { temp: 0.0, top_p: 1.0, state: 0 };
        let mut whole = before.clone();
        let answer = session.answer(
            &self.loaded.qwen,
            &mut self.loaded.decoder,
            h,
            Generate { sampler: &mut sampler, max_new, stops: &stops },
            &mut |said| {
                whole.truncate(before.len());
                whole.extend_from_slice(said);
                match router.step(&whole, tok, think, &stop_strings) {
                    Err(e) => {
                        failure = Some(e);
                        return Flow::Cancel;
                    }
                    Ok((c, r, stop)) => {
                        if (!c.is_empty() || !r.is_empty()) && !send(Event::Delta { content: c, thinking: r }) {
                            gone = true;
                            return Flow::Cancel;
                        }
                        if stop {
                            return Flow::Cancel;
                        }
                    }
                }
                loop {
                    match rx.try_recv() {
                        Ok(j) if matches!(j.request.prompt, Prompt::Load) => {
                            inf.take_job();
                            Self::answer_load(&j, Instant::now(), Duration::ZERO);
                            *inf.expires.lock().expect("expires lock") = j.keep_alive.until(SystemTime::now());
                        }
                        Ok(j) => {
                            pending.push_back(j);
                            arrived = true;
                        }
                        Err(std::sync::mpsc::TryRecvError::Empty) => break,
                        Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                            *open = false;
                            break;
                        }
                    }
                }
                match arrived {
                    true => Flow::Cancel,
                    false => Flow::Continue,
                }
            },
        );
        let answer = match (answer, failure) {
            (_, Some(e)) | (Err(e), None) => {
                send(Self::error(500, format!("{e:#}")));
                return None;
            }
            (Ok(a), None) => a,
        };
        if gone {
            return None;
        }
        l.router = router;
        l.said.extend_from_slice(&answer.said);
        if let Some(&t) = answer.said.last() {
            l.next = t;
        }
        let reason = match (answer.finish, l.router.stopped_at) {
            (_, Some(_)) | (Finish::Stop, None) => Some("stop"),
            _ if l.said.len() >= l.max_new => Some("length"),
            // Cancelled by an arrival, or cut at the batch-1 path's cap.
            _ => None,
        };
        if let Some(reason) = reason {
            self.finish(&mut l, reason);
            return None;
        }
        match pool.adopt(&self.loaded.qwen, session.caches()) {
            Ok(Some(slot)) => {
                l.slot = slot;
                Some(l)
            }
            Ok(None) => unreachable!("a lone request leaves every slot free"),
            Err(e) => {
                let _ = l.job.events.blocking_send(Self::error(500, format!("joining the batch: {e:#}")));
                None
            }
        }
    }

    /// The answer to a job that only loads the model.
    fn answer_load(job: &Job, started: Instant, load: Duration) {
        let _ = job.events.blocking_send(Event::Accepted { prompt_tokens: 0, cached: 0 });
        let _ = job.events.blocking_send(Event::Done(Stats {
            done_reason: "load".into(),
            total_duration: nanos(total(job, started, load)),
            load_duration: nanos(load),
            ..Stats::default()
        }));
    }

    /// Prefill a job into a slot and take its first token. `None` when the
    /// job ended there (refused, answered, failed, or its client gone).
    fn admit(&mut self, job: Job, pool: &mut llvq_llm::batch::Pool) -> Option<Live> {
        let started = Instant::now();
        let load = self.load_duration.take().unwrap_or_default();
        let tx = job.events.clone();
        if tx.is_closed() {
            return None;
        }
        let send = |e: Event| tx.blocking_send(e).is_ok();
        if let Prompt::Load = job.request.prompt {
            Self::answer_load(&job, started, load);
            return None;
        }
        let p = match self.prepare(&job.request, &[]) {
            Ok(Ok(p)) => p,
            Ok(Err(refusal)) => {
                send(refusal);
                return None;
            }
            Err(e) => {
                send(Self::error(500, format!("{e:#}")));
                return None;
            }
        };
        let t = Instant::now();
        let (slot, first) = match pool.admit(&self.loaded.qwen, &p.prompt) {
            Ok(Some(s)) => s,
            Ok(None) => unreachable!("admitted only while a slot is free"),
            Err(e) => {
                send(Self::error(500, format!("{e:#}")));
                return None;
            }
        };
        let prompt_eval = t.elapsed();
        if !send(Event::Accepted { prompt_tokens: p.prompt.len(), cached: 0 }) {
            pool.release(slot);
            return None;
        }
        let mut l = Live {
            job,
            slot,
            router: Router::default(),
            said: Vec::new(),
            next: first,
            max_new: p.max_new,
            started,
            load,
            prompt: p.prompt,
            prompt_eval,
            t_eval: Instant::now(),
        };
        match self.advance(&mut l, first) {
            After::Continue => Some(l),
            a => {
                pool.release(slot);
                if let After::Done(reason) = a {
                    self.finish(&mut l, reason);
                }
                None
            }
        }
    }

    /// One picked token of a live request: Session::answer's rules, the
    /// stop token not said, the cap counted on said tokens, the text routed
    /// and streamed.
    fn advance(&self, l: &mut Live, pick: u32) -> After {
        if self.marks.stops().contains(&pick) {
            return After::Done("stop");
        }
        l.said.push(pick);
        l.next = pick;
        let tok = &self.loaded.tokenizer;
        match l.router.step(&l.said, tok, self.marks.think, &l.job.request.options.stop) {
            Err(e) => {
                let _ = l.job.events.blocking_send(Self::error(500, format!("{e:#}")));
                return After::Gone;
            }
            Ok((c, r, stop)) => {
                if (!c.is_empty() || !r.is_empty())
                    && l.job.events.blocking_send(Event::Delta { content: c, thinking: r }).is_err()
                {
                    return After::Gone;
                }
                if stop {
                    return After::Done("stop");
                }
            }
        }
        if l.said.len() >= l.max_new {
            return After::Done("length");
        }
        After::Continue
    }

    /// The tail of an answer that ended, and its last chunk.
    fn finish(&self, l: &mut Live, reason: &'static str) {
        let events = l.job.events.clone();
        let send = |e: Event| events.blocking_send(e).is_ok();
        if l.router.stopped_at.is_none() {
            match l.router.tail(&self.loaded.tokenizer) {
                Ok((c, r)) => {
                    if (!c.is_empty() || !r.is_empty()) && !send(Event::Delta { content: c, thinking: r }) {
                        return;
                    }
                }
                Err(e) => {
                    send(Self::error(500, format!("{e:#}")));
                    return;
                }
            }
        }
        send(Event::Done(Stats {
            done_reason: reason.into(),
            total_duration: nanos(total(&l.job, l.started, l.load)),
            load_duration: nanos(l.load),
            prompt_eval_count: l.prompt.len(),
            prompt_eval_cached_count: 0,
            prompt_eval_duration: nanos(l.prompt_eval),
            eval_count: l.said.len(),
            eval_duration: nanos(l.t_eval.elapsed()),
        }));
    }
}

impl Engine {
    /// Start the thread and load `name` on it. Jobs sent before the load ends
    /// wait in the channel.
    pub fn spawn(
        name: ModelName,
        manifest: Manifest,
        store: Arc<Store>,
        loader: Arc<dyn Loader>,
    ) -> anyhow::Result<Engine> {
        let info = Arc::new(EngineInfo {
            name: name.clone(),
            manifest: manifest.clone(),
            state: Mutex::new(State::Loading),
            queue: AtomicUsize::new(0),
            busy: std::sync::atomic::AtomicBool::new(false),
            expires: Mutex::new(None),
            retiring: std::sync::atomic::AtomicBool::new(false),
        });
        let (tx, rx) = std::sync::mpsc::channel::<Job>();
        let inf = info.clone();
        std::thread::Builder::new().name(format!("llvq-engine-{name}")).spawn(move || {
            let t0 = Instant::now();
            let built = loader.load(&name, &manifest, &store).and_then(|loaded| {
                let marks = Marks::resolve(&loaded.tokenizer)?;
                let rope = loaded.qwen.config().max_position_embeddings;
                let context = loaded.decoder.max_positions().map_or(rope, |e| e.min(rope));
                // Under the batch the pool's positions are the context: past
                // 4E's cap a lone request hands over to the pool
                // (`Worker::solo_decode`) instead of failing there.
                let context = loaded.batch.as_ref().map_or(context, |b| b.serve.context.min(rope));
                let session = Session::new(&loaded.qwen);
                Ok(Worker { loaded, marks, session, memo: Vec::new(), context, load_duration: Some(t0.elapsed()) })
            });
            let mut w = match built {
                Ok(w) => w,
                Err(e) => {
                    let msg = format!("loading {name}: {e:#}");
                    *inf.state.lock().expect("state lock") = State::Failed(msg.clone());
                    while let Ok(job) = rx.recv() {
                        inf.queue.fetch_sub(1, Ordering::SeqCst);
                        let _ = job.events.blocking_send(Event::Error { status: 500, message: msg.clone() });
                    }
                    return;
                }
            };
            *inf.state.lock().expect("state lock") = State::Ready {
                backend: w.loaded.backend.clone(),
                config: w.loaded.config.clone(),
                context: w.context,
                greedy_only: w.greedy_only(),
                pool_bytes: w.loaded.batch.as_ref().map_or(0, |b| w.loaded.qwen.batch_kv_bytes(b.serve.max, b.serve.context)),
            };
            if w.loaded.batch.is_some() {
                let ran = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| w.run_batch(&rx, &inf)));
                if let Err(p) = ran {
                    let why = p
                        .downcast_ref::<&str>()
                        .map(|s| s.to_string())
                        .or_else(|| p.downcast_ref::<String>().cloned())
                        .unwrap_or_else(|| "no message".into());
                    let msg = format!("the batched engine of {name} panicked: {why}");
                    *inf.state.lock().expect("state lock") = State::Failed(msg.clone());
                    inf.busy.store(false, Ordering::SeqCst);
                    while let Ok(job) = rx.recv() {
                        inf.queue.fetch_sub(1, Ordering::SeqCst);
                        let _ = job.events.blocking_send(Event::Error { status: 500, message: msg.clone() });
                    }
                }
                return;
            }
            while let Ok(job) = rx.recv() {
                inf.take_job();
                // A panic leaves the session in an unknown state: the engine
                // fails, answers its queue with the reason, and the next
                // request loads the model again (`Models::submit` forgets a
                // failed engine) instead of hitting a wedged one forever.
                let ran = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| w.run(&job)));
                if let Err(p) = ran {
                    let why = p
                        .downcast_ref::<&str>()
                        .map(|s| s.to_string())
                        .or_else(|| p.downcast_ref::<String>().cloned())
                        .unwrap_or_else(|| "no message".into());
                    let msg = format!("the engine of {name} panicked: {why}");
                    *inf.state.lock().expect("state lock") = State::Failed(msg.clone());
                    let _ = job.events.blocking_send(Event::Error { status: 500, message: msg.clone() });
                    inf.busy.store(false, Ordering::SeqCst);
                    while let Ok(job) = rx.recv() {
                        inf.queue.fetch_sub(1, Ordering::SeqCst);
                        let _ = job.events.blocking_send(Event::Error { status: 500, message: msg.clone() });
                    }
                    return;
                }
                // A retiring engine is left to the reaper, which reads the
                // flag itself: it may be set after this line.
                *inf.expires.lock().expect("expires lock") = job.keep_alive.until(SystemTime::now());
                inf.busy.store(false, Ordering::SeqCst);
            }
        })?;
        Ok(Engine { tx, info })
    }

    fn submit(&self, job: Job) -> Result<(), String> {
        self.info.queue.fetch_add(1, Ordering::SeqCst);
        self.tx.send(job).map_err(|_| {
            self.info.queue.fetch_sub(1, Ordering::SeqCst);
            format!("the engine of {} has stopped", self.info.name)
        })
    }

    fn idle(&self) -> bool {
        self.info.idle()
    }

    pub fn state(&self) -> State {
        self.info.state.lock().expect("state lock").clone()
    }
}

/// Unload the engines `pick` names: an idle one now, a busy one when its jobs
/// end (the reaper drops it then). Dropping a busy engine would not stop its
/// thread: the model would stay in memory, out of the list and of the cap,
/// while a new request loaded a second copy. True when `pick` named any.
fn retire(engines: &mut Vec<Engine>, pick: impl Fn(&Engine) -> bool) -> bool {
    let mut any = false;
    engines.retain(|e| {
        if !pick(e) {
            return true;
        }
        any = true;
        if e.idle() {
            return false;
        }
        e.info.retiring.store(true, Ordering::SeqCst);
        true
    });
    any
}

/// An error before any engine ran: the HTTP status and the message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Refusal {
    pub status: u16,
    pub message: String,
}

/// The loaded models, and the store they come from.
pub struct Models {
    pub store: Arc<Store>,
    loader: Arc<dyn Loader>,
    engines: Mutex<Vec<Engine>>,
    max_loaded: usize,
    default_keep_alive: KeepAlive,
}

/// A loaded model as `/api/ps` reports it.
#[derive(Debug, Clone)]
pub struct Running {
    pub name: ModelName,
    pub manifest: Manifest,
    pub state: State,
    pub queue: usize,
    pub busy: bool,
    pub expires: Option<SystemTime>,
}

impl Models {
    pub fn new(store: Arc<Store>, loader: Arc<dyn Loader>, max_loaded: usize, default_keep_alive: KeepAlive) -> Self {
        Models { store, loader, engines: Mutex::new(Vec::new()), max_loaded: max_loaded.max(1), default_keep_alive }
    }

    /// `LLVQ_MAX_LOADED` (default 1) and `LLVQ_KEEP_ALIVE` (default 5m).
    pub fn from_env(store: Arc<Store>, loader: Arc<dyn Loader>) -> anyhow::Result<Self> {
        let max_loaded = match std::env::var("LLVQ_MAX_LOADED") {
            Ok(v) => v.trim().parse::<usize>().ok().filter(|n| *n > 0).ok_or_else(|| {
                anyhow::anyhow!("LLVQ_MAX_LOADED={v:?}: a number of models, at least 1")
            })?,
            Err(_) => 1,
        };
        let keep = match std::env::var("LLVQ_KEEP_ALIVE") {
            Ok(v) => KeepAlive::parse_str(&v).map_err(|e| anyhow::anyhow!("LLVQ_KEEP_ALIVE: {e}"))?,
            Err(_) => KeepAlive::For(Duration::from_secs(300)),
        };
        Ok(Self::new(store, loader, max_loaded, keep))
    }

    fn not_found(name: &ModelName) -> Refusal {
        Refusal { status: 404, message: format!("model '{name}' not found, try pulling it first") }
    }

    /// Queue `request` on `name`'s engine, loading it if needed, and return
    /// its events.
    pub fn submit(&self, name: &ModelName, request: Request, keep_alive: Option<KeepAlive>) -> Result<Events, Refusal> {
        let manifest = self
            .store
            .get(name)
            .map_err(|e| Refusal { status: 500, message: format!("{e:#}") })?
            .ok_or_else(|| Self::not_found(name))?;
        let keep = keep_alive.unwrap_or(self.default_keep_alive);
        let mut engines = self.engines.lock().expect("engines lock");
        // An unload request does not load a model that is not loaded.
        if matches!(request.prompt, Prompt::Load) && keep == KeepAlive::For(Duration::ZERO) {
            retire(&mut engines, |e| e.info.name == *name);
            let (tx, rx) = tokio::sync::mpsc::channel(4);
            let _ = tx.try_send(Event::Accepted { prompt_tokens: 0, cached: 0 });
            let _ = tx.try_send(Event::Done(Stats { done_reason: "unload".into(), ..Stats::default() }));
            return Ok(rx);
        }
        // A failed load is forgotten, so the next request tries again.
        engines.retain(|e| !matches!(e.state(), State::Failed(_)));
        let pos = match engines.iter().position(|e| e.info.name == *name && e.info.manifest == manifest) {
            Some(p) => {
                // Asked back before it went.
                engines[p].info.retiring.store(false, Ordering::SeqCst);
                p
            }
            None => {
                // Another manifest under this name: the old engine goes when
                // it is idle, and counts against the cap until then.
                retire(&mut engines, |e| e.info.name == *name);
                while engines.len() >= self.max_loaded {
                    // Only an idle engine is unloaded: one retiring first,
                    // then the one expiring first. Dropping a busy one would
                    // leave its thread running beside the new load, two
                    // models in memory under a cap of one.
                    let victim = engines
                        .iter()
                        .enumerate()
                        .filter(|(_, e)| e.idle())
                        .min_by_key(|(_, e)| {
                            let t = *e.info.expires.lock().expect("expires lock");
                            (!e.info.retiring.load(Ordering::SeqCst), t.is_none(), t)
                        })
                        .map(|(i, _)| i);
                    let Some(victim) = victim else {
                        return Err(Refusal {
                            status: 503,
                            message: format!(
                                "{} model(s) loaded and busy (LLVQ_MAX_LOADED={}); try again when one is idle",
                                engines.len(),
                                self.max_loaded
                            ),
                        });
                    };
                    engines.remove(victim);
                }
                let e = Engine::spawn(name.clone(), manifest, self.store.clone(), self.loader.clone())
                    .map_err(|e| Refusal { status: 500, message: format!("{e:#}") })?;
                engines.push(e);
                engines.len() - 1
            }
        };
        let (tx, rx) = tokio::sync::mpsc::channel(64);
        let e = &engines[pos];
        *e.info.expires.lock().expect("expires lock") = None;
        e.submit(Job { request, events: tx, keep_alive: keep, queued_at: Instant::now() })
            .map_err(|m| Refusal { status: 500, message: m })?;
        Ok(rx)
    }

    /// Unload `name`: now when it is idle, after its jobs when it is busy.
    /// False when it was not loaded.
    pub fn unload(&self, name: &ModelName) -> bool {
        retire(&mut self.engines.lock().expect("engines lock"), |e| e.info.name == *name)
    }

    /// Drop the idle engines whose keep-alive has passed or that retire.
    pub fn reap(&self, now: SystemTime) {
        let mut engines = self.engines.lock().expect("engines lock");
        engines.retain(|e| {
            let expired = matches!(*e.info.expires.lock().expect("expires lock"), Some(t) if t <= now);
            !((expired || e.info.retiring.load(Ordering::SeqCst)) && e.idle())
        });
    }

    pub fn running(&self) -> Vec<Running> {
        let engines = self.engines.lock().expect("engines lock");
        engines
            .iter()
            .map(|e| Running {
                name: e.info.name.clone(),
                manifest: e.info.manifest.clone(),
                state: e.state(),
                queue: e.info.queue.load(Ordering::SeqCst),
                busy: e.info.busy.load(Ordering::SeqCst),
                expires: *e.info.expires.lock().expect("expires lock"),
            })
            .collect()
    }

    pub fn default_keep_alive(&self) -> KeepAlive {
        self.default_keep_alive
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A job taken off the queue is never seen idle: the eviction of
    /// `Models::submit` would drop the engine as it starts the job, and its
    /// thread would run beside the next model loaded under the cap.
    #[test]
    fn a_job_being_taken_is_never_seen_idle() {
        let info = Arc::new(EngineInfo {
            name: ModelName::parse("x:y").unwrap(),
            manifest: Manifest {
                schema: 1,
                sha256: String::new(),
                size: 0,
                repo: String::new(),
                revision: String::new(),
                file: String::new(),
                family: String::new(),
                parameter_size: String::new(),
                bits_per_param: String::new(),
                configs: crate::library::Configs { cuda: None, metal: None, cpu: None, cuda_batch: None },
                modified_at: 0,
            },
            state: Mutex::new(State::Loading),
            queue: AtomicUsize::new(0),
            busy: std::sync::atomic::AtomicBool::new(false),
            expires: Mutex::new(None),
            retiring: std::sync::atomic::AtomicBool::new(false),
        });
        // `round` is odd from a submit until the job is under way.
        let round = Arc::new(AtomicUsize::new(0));
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let seen = Arc::new(AtomicUsize::new(0));
        let watcher = {
            let (info, round, stop, seen) = (info.clone(), round.clone(), stop.clone(), seen.clone());
            std::thread::spawn(move || {
                while !stop.load(Ordering::SeqCst) {
                    let r = round.load(Ordering::SeqCst);
                    let idle = info.idle();
                    if r % 2 == 1 && idle && round.load(Ordering::SeqCst) == r {
                        seen.fetch_add(1, Ordering::SeqCst);
                    }
                }
            })
        };
        for _ in 0..200_000 {
            info.queue.fetch_add(1, Ordering::SeqCst);
            round.fetch_add(1, Ordering::SeqCst);
            info.take_job();
            round.fetch_add(1, Ordering::SeqCst);
            info.busy.store(false, Ordering::SeqCst);
        }
        stop.store(true, Ordering::SeqCst);
        watcher.join().unwrap();
        assert_eq!(seen.load(Ordering::SeqCst), 0, "an engine starting a job was seen idle");
    }

    #[test]
    fn keep_alive_reads_ollamas_forms() {
        let s = |v: &str| KeepAlive::parse_str(v).unwrap();
        assert_eq!(s("5m"), KeepAlive::For(Duration::from_secs(300)));
        assert_eq!(s("1h30m"), KeepAlive::For(Duration::from_secs(5400)));
        assert_eq!(s("10s"), KeepAlive::For(Duration::from_secs(10)));
        assert_eq!(s("250ms"), KeepAlive::For(Duration::from_millis(250)));
        assert_eq!(s("0"), KeepAlive::For(Duration::ZERO));
        assert_eq!(s("-1"), KeepAlive::Forever);
        assert_eq!(s("-1m"), KeepAlive::Forever);
        assert_eq!(KeepAlive::parse(&serde_json::json!(30)).unwrap(), KeepAlive::For(Duration::from_secs(30)));
        assert_eq!(KeepAlive::parse(&serde_json::json!(-1)).unwrap(), KeepAlive::Forever);
        for bad in ["5", "5x", "m", "1h5"] {
            let r = KeepAlive::parse_str(bad);
            assert!(bad == "5" || r.is_err(), "{bad:?} must be refused");
        }
    }
}
