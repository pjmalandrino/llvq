//! The HTTP routes: Ollama's native API in NDJSON, OpenAI's in server-sent
//! events ([`crate::openai`]), and the OpenAPI 3.1 document of both, generated
//! from the handlers' annotations so the document and the code cannot drift.
//!
//! Errors are Ollama's shape, `{"error": "..."}`, with the HTTP status. A
//! route that is not served answers `501` and names why, rather than half an
//! implementation.

// A refusal here is the HTTP response itself, built once a request; boxing
// it would buy nothing.
#![allow(clippy::result_large_err)]

use crate::chat::{Msg, Role};
use crate::engine::{Event, KeepAlive, Models, Options, Prompt, Request, State as EngineState, Stats};
use crate::library::{self, ModelName};
use crate::store::{Manifest, PullEvent};
use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use std::convert::Infallible;
use std::sync::Arc;
use std::time::SystemTime;
use utoipa::{OpenApi, ToSchema};
use utoipa_axum::router::OpenApiRouter;
use utoipa_axum::routes;

/// What every handler shares.
#[derive(Clone)]
pub struct AppState {
    pub models: Arc<Models>,
    /// Where `/api/pull` downloads from: `HF_ENDPOINT` or the Hub. Two
    /// pulls of one file wait for each other on the store's file lock, in
    /// this server or across processes.
    pub hub: String,
}

impl AppState {
    pub fn new(models: Arc<Models>, hub: String) -> Self {
        AppState { models, hub }
    }
}

// ---------------------------------------------------------------------------
// Shared helpers.
// ---------------------------------------------------------------------------

/// RFC 3339 in UTC with nanoseconds, the form Ollama writes `created_at` in.
pub fn rfc3339(t: SystemTime) -> String {
    let d = t.duration_since(SystemTime::UNIX_EPOCH).unwrap_or_default();
    let secs = d.as_secs() as i64;
    let (days, rem) = (secs.div_euclid(86_400), secs.rem_euclid(86_400));
    // Howard Hinnant's civil_from_days.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}.{:09}Z",
        rem / 3600,
        rem % 3600 / 60,
        rem % 60,
        d.subsec_nanos()
    )
}

pub fn unix(t: SystemTime) -> u64 {
    t.duration_since(SystemTime::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

pub fn error(status: u16, message: impl Into<String>) -> Response {
    let s = StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    (s, Json(ErrorResponse { error: message.into() })).into_response()
}

/// A JSON body, with serde's message as a `400` in Ollama's shape. Axum's own
/// extractor answers in plain text, which no Ollama client reads.
fn parse<T: for<'de> Deserialize<'de>>(body: &Bytes) -> Result<T, Response> {
    serde_json::from_slice(body).map_err(|e| error(400, format!("the request body: {e}")))
}

fn model_name(s: &str) -> Result<ModelName, Response> {
    ModelName::parse(s).map_err(|e| error(400, e))
}

fn keep_alive(v: &Option<Value>) -> Result<Option<KeepAlive>, Response> {
    match v {
        None | Some(Value::Null) => Ok(None),
        Some(v) => KeepAlive::parse(v).map(Some).map_err(|e| error(400, e)),
    }
}

/// Ollama's `think`: a boolean, a level name, or null for the model's
/// default. Qwen3 has one level; any name turns reasoning on. The default is
/// off, as in `bin/chat`.
fn think(v: &Option<Value>) -> Result<bool, Response> {
    match v {
        None | Some(Value::Null) => Ok(false),
        Some(Value::Bool(b)) => Ok(*b),
        Some(Value::String(s)) => Ok(!matches!(s.as_str(), "" | "none" | "false")),
        Some(o) => Err(error(400, format!("think {o}: true, false or a level name"))),
    }
}

/// The options the engine implements, and a warning for every other one
/// (decision D3: ignored and listed, because clients send them by default).
fn options(o: &Option<Map<String, Value>>, warnings: &mut Vec<String>) -> Result<Options, Response> {
    let mut out = Options::default();
    let Some(o) = o else { return Ok(out) };
    let f = |k: &str, v: &Value| v.as_f64().ok_or_else(|| error(400, format!("option {k}: a number")));
    for (k, v) in o {
        if v.is_null() {
            continue;
        }
        match k.as_str() {
            "temperature" => out.temperature = Some(f(k, v)? as f32),
            "top_p" => out.top_p = Some(f(k, v)? as f32),
            "seed" => out.seed = Some(v.as_i64().ok_or_else(|| error(400, "option seed: an integer"))? as u64),
            "num_predict" => {
                out.num_predict = Some(v.as_i64().ok_or_else(|| error(400, "option num_predict: an integer"))?)
            }
            "num_ctx" => {
                out.num_ctx =
                    Some(v.as_u64().ok_or_else(|| error(400, "option num_ctx: a positive integer"))? as usize)
            }
            "stop" => out.stop = stop_list(v).map_err(|e| error(400, e))?,
            other => warnings.push(format!("option {other} is not implemented by llvq and was ignored")),
        }
    }
    Ok(out)
}

/// A stop list, as a string or an array of strings.
pub fn stop_list(v: &Value) -> Result<Vec<String>, String> {
    match v {
        Value::Null => Ok(Vec::new()),
        Value::String(s) => Ok(vec![s.clone()]),
        Value::Array(a) => a
            .iter()
            .map(|x| x.as_str().map(str::to_string).ok_or_else(|| "stop: strings only".to_string()))
            .collect(),
        _ => Err("stop: a string or an array of strings".into()),
    }
}

/// Refuse what the engine does not do, by name.
fn refuse_unsupported(images: bool, format: &Option<Value>, tools: bool) -> Result<(), Response> {
    if images {
        return Err(error(400, "images: the served models read text only"));
    }
    if format.as_ref().is_some_and(|f| !f.is_null() && f != "") {
        return Err(error(400, "format: structured output is not implemented by llvq"));
    }
    if tools {
        return Err(error(400, "tools: tool calls are not served by llvq"));
    }
    Ok(())
}

/// Wait for the engine's first event: accepted, or a refusal to answer with.
pub async fn accepted(rx: &mut crate::engine::Events) -> Result<(), (u16, String)> {
    match rx.recv().await {
        Some(Event::Accepted { .. }) => Ok(()),
        Some(Event::Error { status, message }) => Err((status, message)),
        Some(other) => Err((500, format!("the engine answered {other:?} before accepting"))),
        None => Err((500, "the engine stopped before answering".into())),
    }
}

/// The engine's events as a stream, ending after `Done` or `Error`.
pub fn events(rx: crate::engine::Events) -> impl futures_util::Stream<Item = Event> + Send {
    futures_util::stream::unfold((rx, false), |(mut rx, over)| async move {
        if over {
            return None;
        }
        let e = rx.recv().await?;
        let end = matches!(e, Event::Done(_) | Event::Error { .. });
        Some((e, (rx, end)))
    })
}

fn ndjson(s: impl futures_util::Stream<Item = Value> + Send + 'static) -> Response {
    let body = Body::from_stream(s.map(|v| {
        let mut b = serde_json::to_vec(&v).expect("a JSON value serializes");
        b.push(b'\n');
        Ok::<_, Infallible>(Bytes::from(b))
    }));
    Response::builder().header(header::CONTENT_TYPE, "application/x-ndjson").body(body).expect("a response builds")
}

fn details(m: &Manifest) -> Details {
    Details {
        parent_model: String::new(),
        format: "llvq".into(),
        family: m.family.clone(),
        families: vec![m.family.clone()],
        parameter_size: m.parameter_size.clone(),
        quantization_level: format!("Tetra {} b/param", m.bits_per_param),
    }
}

// ---------------------------------------------------------------------------
// Schemas.
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, ToSchema)]
pub struct ErrorResponse {
    pub error: String,
}

#[derive(Serialize, Deserialize, ToSchema, Clone)]
pub struct Details {
    pub parent_model: String,
    pub format: String,
    pub family: String,
    pub families: Vec<String>,
    pub parameter_size: String,
    pub quantization_level: String,
}

#[derive(Serialize, Deserialize, ToSchema)]
pub struct VersionResponse {
    pub version: String,
    /// The commit the binary was built from.
    pub commit: String,
}

#[derive(Serialize, Deserialize, ToSchema)]
pub struct ModelSummary {
    pub name: String,
    pub model: String,
    pub modified_at: String,
    pub size: u64,
    pub digest: String,
    pub details: Details,
}

#[derive(Serialize, Deserialize, ToSchema)]
pub struct ListResponse {
    pub models: Vec<ModelSummary>,
}

/// What llvq adds to a running model.
#[derive(Serialize, Deserialize, ToSchema)]
pub struct RunningLlvq {
    /// `loading`, `ready` or `failed`.
    pub state: String,
    /// `cuda`, `metal` or `cpu`.
    pub backend: Option<String>,
    pub config: Option<String>,
    /// Requests waiting behind the one running.
    pub queue: usize,
    pub busy: bool,
    /// Why the load failed.
    pub error: Option<String>,
    /// Bytes of the batch pool's KV caches (`LLVQ_BATCH`), counted in
    /// `size_vram` on a device; 0 without the batch.
    pub batch_pool_bytes: u64,
}

#[derive(Serialize, Deserialize, ToSchema)]
pub struct RunningModel {
    pub name: String,
    pub model: String,
    pub size: u64,
    pub digest: String,
    pub details: Details,
    pub expires_at: String,
    pub size_vram: u64,
    pub context_length: Option<usize>,
    pub llvq: RunningLlvq,
}

#[derive(Serialize, Deserialize, ToSchema)]
pub struct PsResponse {
    pub models: Vec<RunningModel>,
}

#[derive(Serialize, Deserialize, ToSchema)]
pub struct ShowRequest {
    pub model: String,
    pub verbose: Option<bool>,
}

#[derive(Serialize, Deserialize, ToSchema)]
pub struct ShowLlvq {
    pub repo: String,
    pub revision: String,
    pub file: String,
    pub sha256: String,
    pub bits_per_param: String,
    pub config_cuda: Option<String>,
    pub config_metal: Option<String>,
    pub config_cpu: Option<String>,
    /// The backend this build serves with.
    pub backend: String,
}

#[derive(Serialize, Deserialize, ToSchema)]
pub struct ShowResponse {
    pub modelfile: String,
    pub parameters: String,
    pub template: String,
    pub details: Details,
    #[schema(value_type = Object)]
    pub model_info: Map<String, Value>,
    pub capabilities: Vec<String>,
    pub modified_at: String,
    pub llvq: ShowLlvq,
}

#[derive(Serialize, Deserialize, ToSchema)]
pub struct PullRequest {
    pub model: String,
    pub insecure: Option<bool>,
    pub stream: Option<bool>,
}

#[derive(Serialize, Deserialize, ToSchema, Debug, PartialEq)]
pub struct StatusEvent {
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub digest: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completed: Option<u64>,
}

#[derive(Serialize, Deserialize, ToSchema)]
pub struct DeleteRequest {
    pub model: String,
}

#[derive(Serialize, Deserialize, ToSchema)]
pub struct CopyRequest {
    pub source: String,
    pub destination: String,
}

#[derive(Serialize, Deserialize, ToSchema, Clone)]
pub struct ChatMessage {
    /// `system`, `user` or `assistant`.
    pub role: String,
    pub content: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub images: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(value_type = Option<Vec<Object>>)]
    pub tool_calls: Option<Vec<Value>>,
}

#[derive(Serialize, Deserialize, ToSchema)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[schema(value_type = Option<Vec<Object>>)]
    pub tools: Option<Vec<Value>>,
    #[schema(value_type = Option<Object>)]
    pub format: Option<Value>,
    /// `temperature`, `top_p`, `seed`, `num_predict`, `stop` and `num_ctx`
    /// are read; any other option is listed in `warnings` and ignored.
    #[schema(value_type = Option<Object>)]
    pub options: Option<Map<String, Value>>,
    /// Default true.
    pub stream: Option<bool>,
    /// true, false or a level name; default off.
    #[schema(value_type = Option<Object>)]
    pub think: Option<Value>,
    /// Seconds, or a duration such as `"5m"`; negative keeps the model loaded.
    #[schema(value_type = Option<Object>)]
    pub keep_alive: Option<Value>,
    pub logprobs: Option<bool>,
    pub top_logprobs: Option<u32>,
}

#[derive(Serialize, Deserialize, ToSchema)]
pub struct GenerateRequest {
    pub model: String,
    pub prompt: Option<String>,
    pub suffix: Option<String>,
    pub images: Option<Vec<String>>,
    #[schema(value_type = Option<Object>)]
    pub format: Option<Value>,
    pub system: Option<String>,
    pub stream: Option<bool>,
    #[schema(value_type = Option<Object>)]
    pub think: Option<Value>,
    pub raw: Option<bool>,
    #[schema(value_type = Option<Object>)]
    pub keep_alive: Option<Value>,
    #[schema(value_type = Option<Object>)]
    pub options: Option<Map<String, Value>>,
    pub logprobs: Option<bool>,
    pub top_logprobs: Option<u32>,
    /// A custom template is not applied: Qwen3's is built from ids.
    pub template: Option<String>,
    /// Deprecated in Ollama; ignored.
    pub context: Option<Vec<u32>>,
}

/// The timings and counts of the last chunk, in nanoseconds.
#[derive(Serialize, Deserialize, ToSchema, Default, Clone)]
pub struct DoneFields {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub done_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_duration: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub load_duration: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_eval_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_eval_cached_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_eval_duration: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval_duration: Option<u64>,
    /// Options and fields llvq ignored.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub warnings: Option<Vec<String>>,
}

impl DoneFields {
    fn of(s: &Stats, warnings: &[String]) -> Self {
        DoneFields {
            done_reason: Some(s.done_reason.clone()),
            total_duration: Some(s.total_duration),
            load_duration: Some(s.load_duration),
            prompt_eval_count: Some(s.prompt_eval_count),
            prompt_eval_cached_count: Some(s.prompt_eval_cached_count),
            prompt_eval_duration: Some(s.prompt_eval_duration),
            eval_count: Some(s.eval_count),
            eval_duration: Some(s.eval_duration),
            warnings: (!warnings.is_empty()).then(|| warnings.to_vec()),
        }
    }
}

#[derive(Serialize, Deserialize, ToSchema)]
pub struct ChatResponse {
    pub model: String,
    pub created_at: String,
    pub message: ChatMessage,
    pub done: bool,
    #[serde(flatten)]
    pub stats: DoneFields,
}

#[derive(Serialize, Deserialize, ToSchema)]
pub struct GenerateResponse {
    pub model: String,
    pub created_at: String,
    pub response: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<String>,
    pub done: bool,
    #[serde(flatten)]
    pub stats: DoneFields,
}

// ---------------------------------------------------------------------------
// Ollama routes.
// ---------------------------------------------------------------------------

/// The runtime's version and the commit it was built from.
#[utoipa::path(get, path = "/api/version", tag = "ollama", responses((status = 200, body = VersionResponse)))]
pub async fn version() -> Json<VersionResponse> {
    Json(VersionResponse { version: env!("CARGO_PKG_VERSION").into(), commit: env!("LLVQ_BUILD_COMMIT").into() })
}

/// The models in the store.
#[utoipa::path(get, path = "/api/tags", tag = "ollama", responses((status = 200, body = ListResponse)))]
pub async fn tags(State(st): State<AppState>) -> Response {
    match st.models.store.list() {
        Err(e) => error(500, format!("{e:#}")),
        Ok(list) => Json(ListResponse {
            models: list
                .into_iter()
                .map(|(n, m)| ModelSummary {
                    name: n.to_string(),
                    model: n.to_string(),
                    modified_at: rfc3339(SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(m.modified_at)),
                    size: m.size,
                    digest: m.sha256.clone(),
                    details: details(&m),
                })
                .collect(),
        })
        .into_response(),
    }
}

/// The loaded models, their backend, config and queue.
#[utoipa::path(get, path = "/api/ps", tag = "ollama", responses((status = 200, body = PsResponse)))]
pub async fn ps(State(st): State<AppState>) -> Json<PsResponse> {
    let models = st
        .models
        .running()
        .into_iter()
        .map(|r| {
            let (state, backend, config, context, err, pool) = match r.state {
                EngineState::Loading => ("loading", None, None, None, None, 0),
                EngineState::Ready { backend, config, context, pool_bytes, .. } => {
                    ("ready", Some(backend), config, Some(context), None, pool_bytes)
                }
                EngineState::Failed(e) => ("failed", None, None, None, Some(e), 0),
            };
            let on_device = backend.as_deref().is_some_and(|b| b != "cpu" && b != "dense-cpu");
            RunningModel {
                name: r.name.to_string(),
                model: r.name.to_string(),
                size: r.manifest.size,
                digest: r.manifest.sha256.clone(),
                details: details(&r.manifest),
                // Ollama writes a far date for a model kept forever.
                expires_at: rfc3339(r.expires.unwrap_or(
                    SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(253_402_300_799),
                )),
                // The weights and the batch pool: the pool weighs 19.3 GB at
                // 64 x 2,048 beside a 1.38 GB 4B (audit 2026-11-08 §1.8).
                size_vram: if on_device { r.manifest.size + pool } else { 0 },
                context_length: context,
                llvq: RunningLlvq {
                    state: state.into(),
                    backend,
                    config,
                    queue: r.queue,
                    busy: r.busy,
                    error: err,
                    batch_pool_bytes: pool,
                },
            }
        })
        .collect();
    Json(PsResponse { models })
}

/// Qwen3's chat format, as `chatfmt` builds it from ids.
const TEMPLATE: &str = "<|im_start|>{{role}}\n{{content}}<|im_end|>\n ... <|im_start|>assistant\n\
(<think>\n\n</think>\n\n when think is off). Built from token ids by llvq_llm::chatfmt, not from this text.";

/// A model's manifest, details and served config.
#[utoipa::path(post, path = "/api/show", tag = "ollama", request_body = ShowRequest,
    responses((status = 200, body = ShowResponse), (status = 404, body = ErrorResponse)))]
pub async fn show(State(st): State<AppState>, body: Bytes) -> Response {
    let req: ShowRequest = match parse(&body) {
        Ok(r) => r,
        Err(r) => return r,
    };
    let name = match model_name(&req.model) {
        Ok(n) => n,
        Err(r) => return r,
    };
    let m = match st.models.store.get(&name) {
        Err(e) => return error(500, format!("{e:#}")),
        Ok(None) => return error(404, format!("model '{name}' not found")),
        Ok(Some(m)) => m,
    };
    let served = library::served_configs(&m.configs, &m.sha256);
    let mut info = Map::new();
    info.insert("general.architecture".into(), json!("qwen3"));
    info.insert("general.parameter_count_label".into(), json!(m.parameter_size));
    info.insert("llvq.bits_per_param".into(), json!(m.bits_per_param));
    let mut capabilities = vec!["completion".to_string()];
    capabilities.push("thinking".into());
    Json(ShowResponse {
        modelfile: String::new(),
        parameters: String::new(),
        template: TEMPLATE.into(),
        details: details(&m),
        model_info: info,
        capabilities,
        modified_at: rfc3339(SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(m.modified_at)),
        llvq: ShowLlvq {
            repo: m.repo.clone(),
            revision: m.revision.clone(),
            file: m.file.clone(),
            sha256: m.sha256.clone(),
            bits_per_param: m.bits_per_param.clone(),
            config_cuda: served.cuda,
            config_metal: served.metal,
            config_cpu: served.cpu,
            backend: library::Backend::of_build().label().into(),
        },
    })
    .into_response()
}

fn status_event(e: PullEvent) -> StatusEvent {
    match e {
        PullEvent::Status(s) => StatusEvent { status: s, digest: None, total: None, completed: None },
        PullEvent::Progress { digest, total, completed } => StatusEvent {
            status: format!("pulling {}", &digest[..digest.len().min(19)]),
            digest: Some(digest),
            total: Some(total),
            completed: Some(completed),
        },
    }
}

/// Download a model of the library at its pinned revision, check its size and
/// sha256, and register it.
#[utoipa::path(post, path = "/api/pull", tag = "ollama", request_body = PullRequest,
    responses((status = 200, description = "NDJSON of StatusEvent, or one when stream is false", body = StatusEvent),
              (status = 404, body = ErrorResponse)))]
pub async fn pull(State(st): State<AppState>, body: Bytes) -> Response {
    let req: PullRequest = match parse(&body) {
        Ok(r) => r,
        Err(r) => return r,
    };
    let name = match model_name(&req.model) {
        Ok(n) => n,
        Err(r) => return r,
    };
    let Some(entry) = library::find(&name) else {
        let known: Vec<String> = library::entries().iter().map(|e| e.model_name().to_string()).collect();
        return error(404, format!("model '{name}' is not in the llvq library, which holds {}", known.join(", ")));
    };
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<StatusEvent, String>>(64);
    let store = st.models.store.clone();
    let hub = st.hub.clone();
    tokio::task::spawn_blocking(move || {
        let mut report = |e: PullEvent| {
            let _ = tx.blocking_send(Ok(status_event(e)));
        };
        if let Err(e) = store.pull(&name, entry, &hub, &mut report) {
            let _ = tx.blocking_send(Err(format!("{e:#}")));
        }
    });
    let stream = futures_util::stream::unfold(rx, |mut rx| async move { rx.recv().await.map(|e| (e, rx)) });
    if req.stream == Some(false) {
        let all: Vec<_> = stream.collect().await;
        return match all.into_iter().last() {
            Some(Ok(e)) => Json(e).into_response(),
            Some(Err(m)) => error(500, m),
            None => error(500, "the pull ended without a word"),
        };
    }
    ndjson(stream.map(|r| match r {
        Ok(e) => serde_json::to_value(e).expect("a status serializes"),
        Err(m) => json!({ "error": m }),
    }))
}

/// Remove a model, and its blob when no other name holds it.
#[utoipa::path(delete, path = "/api/delete", tag = "ollama", request_body = DeleteRequest,
    responses((status = 200), (status = 404, body = ErrorResponse)))]
pub async fn delete(State(st): State<AppState>, body: Bytes) -> Response {
    let req: DeleteRequest = match parse(&body) {
        Ok(r) => r,
        Err(r) => return r,
    };
    let name = match model_name(&req.model) {
        Ok(n) => n,
        Err(r) => return r,
    };
    st.models.unload(&name);
    match st.models.store.remove(&name) {
        Ok(true) => StatusCode::OK.into_response(),
        Ok(false) => error(404, format!("model '{name}' not found")),
        Err(e) => error(500, format!("{e:#}")),
    }
}

/// A second name on the same blob.
#[utoipa::path(post, path = "/api/copy", tag = "ollama", request_body = CopyRequest,
    responses((status = 200), (status = 404, body = ErrorResponse)))]
pub async fn copy(State(st): State<AppState>, body: Bytes) -> Response {
    let req: CopyRequest = match parse(&body) {
        Ok(r) => r,
        Err(r) => return r,
    };
    let (src, dst) = match (model_name(&req.source), model_name(&req.destination)) {
        (Ok(a), Ok(b)) => (a, b),
        (Err(r), _) | (_, Err(r)) => return r,
    };
    match st.models.store.copy(&src, &dst) {
        Ok(true) => StatusCode::OK.into_response(),
        Ok(false) => error(404, format!("model '{src}' not found")),
        Err(e) => error(500, format!("{e:#}")),
    }
}

fn chunk_chat(model: &str, content: String, thinking: String, done: Option<DoneFields>) -> Value {
    let message = ChatMessage {
        role: "assistant".into(),
        content,
        thinking: (!thinking.is_empty()).then_some(thinking),
        images: None,
        tool_calls: None,
    };
    serde_json::to_value(ChatResponse {
        model: model.into(),
        created_at: rfc3339(SystemTime::now()),
        message,
        done: done.is_some(),
        stats: done.unwrap_or_default(),
    })
    .expect("a chunk serializes")
}

fn chunk_generate(model: &str, content: String, thinking: String, done: Option<DoneFields>) -> Value {
    serde_json::to_value(GenerateResponse {
        model: model.into(),
        created_at: rfc3339(SystemTime::now()),
        response: content,
        thinking: (!thinking.is_empty()).then_some(thinking),
        done: done.is_some(),
        stats: done.unwrap_or_default(),
    })
    .expect("a chunk serializes")
}

type Chunk = fn(&str, String, String, Option<DoneFields>) -> Value;

/// Submit, wait for acceptance, then stream or aggregate.
async fn answer(
    st: &AppState,
    name: ModelName,
    request: Request,
    keep: Option<KeepAlive>,
    stream: bool,
    warnings: Vec<String>,
    chunk: Chunk,
) -> Response {
    let mut rx = match st.models.submit(&name, request, keep) {
        Ok(rx) => rx,
        Err(r) => return error(r.status, r.message),
    };
    if let Err((status, message)) = accepted(&mut rx).await {
        return error(status, message);
    }
    let model = name.to_string();
    if stream {
        return ndjson(events(rx).map(move |e| match e {
            Event::Delta { content, thinking } => chunk(&model, content, thinking, None),
            Event::Done(s) => chunk(&model, String::new(), String::new(), Some(DoneFields::of(&s, &warnings))),
            Event::Error { message, .. } => json!({ "error": message }),
            Event::Accepted { .. } => json!({ "error": "accepted twice" }),
        }));
    }
    let (mut content, mut thinking) = (String::new(), String::new());
    let mut ev = Box::pin(events(rx));
    while let Some(e) = ev.next().await {
        match e {
            Event::Delta { content: c, thinking: t } => {
                content.push_str(&c);
                thinking.push_str(&t);
            }
            Event::Done(s) => return Json(chunk(&model, content, thinking, Some(DoneFields::of(&s, &warnings)))).into_response(),
            Event::Error { status, message } => return error(status, message),
            Event::Accepted { .. } => {}
        }
    }
    error(500, "the engine stopped before the end of the answer")
}

/// A chat turn: the whole history in, the answer streamed out.
#[utoipa::path(post, path = "/api/chat", tag = "ollama", request_body = ChatRequest,
    responses((status = 200, description = "NDJSON of ChatResponse, or one when stream is false", body = ChatResponse),
              (status = 400, body = ErrorResponse), (status = 404, body = ErrorResponse)))]
pub async fn chat(State(st): State<AppState>, body: Bytes) -> Response {
    let req: ChatRequest = match parse(&body) {
        Ok(r) => r,
        Err(r) => return r,
    };
    let name = match model_name(&req.model) {
        Ok(n) => n,
        Err(r) => return r,
    };
    let images = req.messages.iter().any(|m| m.images.as_ref().is_some_and(|i| !i.is_empty()));
    if let Err(r) = refuse_unsupported(images, &req.format, req.tools.as_ref().is_some_and(|t| !t.is_empty())) {
        return r;
    }
    let mut warnings = Vec::new();
    if req.logprobs == Some(true) {
        warnings.push("logprobs are not implemented by llvq and were ignored".into());
    }
    let options = match options(&req.options, &mut warnings) {
        Ok(o) => o,
        Err(r) => return r,
    };
    let (think, keep) = match (think(&req.think), keep_alive(&req.keep_alive)) {
        (Ok(t), Ok(k)) => (t, k),
        (Err(r), _) | (_, Err(r)) => return r,
    };
    let mut msgs = Vec::with_capacity(req.messages.len());
    for m in &req.messages {
        match Role::parse(&m.role) {
            Ok(role) => msgs.push(Msg { role, content: m.content.clone() }),
            Err(e) => return error(400, e),
        }
    }
    let prompt = match msgs.is_empty() {
        true => Prompt::Load,
        false => Prompt::Chat(msgs),
    };
    let request = Request { prompt, think, options };
    answer(&st, name, request, keep, req.stream.unwrap_or(true), warnings, chunk_chat).await
}

/// A completion of one prompt, through the template or raw.
#[utoipa::path(post, path = "/api/generate", tag = "ollama", request_body = GenerateRequest,
    responses((status = 200, description = "NDJSON of GenerateResponse, or one when stream is false", body = GenerateResponse),
              (status = 400, body = ErrorResponse), (status = 404, body = ErrorResponse)))]
pub async fn generate(State(st): State<AppState>, body: Bytes) -> Response {
    let req: GenerateRequest = match parse(&body) {
        Ok(r) => r,
        Err(r) => return r,
    };
    let name = match model_name(&req.model) {
        Ok(n) => n,
        Err(r) => return r,
    };
    if let Err(r) = refuse_unsupported(req.images.as_ref().is_some_and(|i| !i.is_empty()), &req.format, false) {
        return r;
    }
    if req.suffix.as_ref().is_some_and(|s| !s.is_empty()) {
        return error(400, "suffix: fill-in-the-middle is not served by llvq");
    }
    let mut warnings = Vec::new();
    if req.template.as_ref().is_some_and(|t| !t.is_empty()) {
        warnings.push("template is not applied: Qwen3's format is built from ids".into());
    }
    if req.context.as_ref().is_some_and(|c| !c.is_empty()) {
        warnings.push("context is deprecated and was ignored".into());
    }
    if req.logprobs == Some(true) {
        warnings.push("logprobs are not implemented by llvq and were ignored".into());
    }
    let options = match options(&req.options, &mut warnings) {
        Ok(o) => o,
        Err(r) => return r,
    };
    let (think, keep) = match (think(&req.think), keep_alive(&req.keep_alive)) {
        (Ok(t), Ok(k)) => (t, k),
        (Err(r), _) | (_, Err(r)) => return r,
    };
    let text = req.prompt.unwrap_or_default();
    let prompt = match (text.is_empty(), req.raw == Some(true)) {
        (true, _) => Prompt::Load,
        (false, true) => Prompt::Raw(text),
        (false, false) => Prompt::Templated { system: req.system.filter(|s| !s.is_empty()), prompt: text },
    };
    let request = Request { prompt, think, options };
    answer(&st, name, request, keep, req.stream.unwrap_or(true), warnings, chunk_generate).await
}

// ---------------------------------------------------------------------------
// Routes not served: 501, with the reason. The POST handlers read the body
// before answering: a reply sent over an unread body makes hyper close the
// connection, and a client that reuses it gets a reset instead of the 501.
// ---------------------------------------------------------------------------

fn not_served(what: &str, why: &str) -> Response {
    error(501, format!("{what} is not implemented by llvq: {why}"))
}

const NO_EMBED: &str = "the served models are generative and no embedding head is sealed";

#[utoipa::path(post, path = "/api/embed", tag = "not served", responses((status = 501, body = ErrorResponse)))]
pub async fn embed(_body: Bytes) -> Response {
    not_served("/api/embed", NO_EMBED)
}

#[utoipa::path(post, path = "/api/embeddings", tag = "not served", responses((status = 501, body = ErrorResponse)))]
pub async fn embeddings(_body: Bytes) -> Response {
    not_served("/api/embeddings", NO_EMBED)
}

#[utoipa::path(post, path = "/api/create", tag = "not served", responses((status = 501, body = ErrorResponse)))]
pub async fn create(_body: Bytes) -> Response {
    not_served("/api/create", "a model is a sealed file of the library; there is no Modelfile")
}

#[utoipa::path(post, path = "/api/push", tag = "not served", responses((status = 501, body = ErrorResponse)))]
pub async fn push(_body: Bytes) -> Response {
    not_served("/api/push", "the library is read from the Hub; there is no registry to push to")
}

#[utoipa::path(head, path = "/api/blobs/{digest}", tag = "not served",
    params(("digest" = String, Path, description = "sha256:<hex>")), responses((status = 501, body = ErrorResponse)))]
pub async fn blob_head() -> Response {
    not_served("/api/blobs", "blobs come from llvq pull only, checked against the library")
}

#[utoipa::path(post, path = "/api/blobs/{digest}", tag = "not served",
    params(("digest" = String, Path, description = "sha256:<hex>")), responses((status = 501, body = ErrorResponse)))]
pub async fn blob_post(_body: Bytes) -> Response {
    not_served("/api/blobs", "blobs come from llvq pull only, checked against the library")
}

/// The chat page, one file compiled into the binary, with no script or font
/// fetched from elsewhere. The library's models are written into it, so the
/// page offers to pull what `llvq pull` would.
const CHAT_HTML: &str = include_str!("chat.html");

pub fn chat_page() -> String {
    let lib: Vec<Value> = library::entries()
        .iter()
        .map(|e| json!({ "name": e.model_name().to_string(), "size": e.size, "bits_per_param": e.bits_per_param }))
        .collect();
    // `</` would end the script element early.
    let lib = serde_json::to_string(&lib).expect("the library serializes").replace("</", "<\\/");
    CHAT_HTML.replace("__LLVQ_LIBRARY__", &lib)
}

/// A chat page in the browser, on `/api/chat`.
#[utoipa::path(get, path = "/", tag = "llvq",
    responses((status = 200, description = "The chat page", content_type = "text/html")))]
pub async fn index() -> Response {
    ([(header::CONTENT_TYPE, "text/html; charset=utf-8")], chat_page()).into_response()
}

/// The OpenAPI 3.1 document of this server.
#[utoipa::path(get, path = "/openapi.json", tag = "llvq", responses((status = 200, description = "OpenAPI 3.1")))]
pub async fn openapi_json() -> Response {
    Json(openapi()).into_response()
}

#[derive(OpenApi)]
#[openapi(
    info(
        title = "llvq",
        description = "Ollama's native API and OpenAI's chat format over the sealed LLVQ files. \
                       Generated from the handlers; llvq-runtime/openapi.json is its committed copy.",
        license(name = "MIT OR Apache-2.0")
    ),
    servers((url = "http://127.0.0.1:11435", description = "llvq serve"))
)]
struct ApiDoc;

fn api_router() -> OpenApiRouter<AppState> {
    OpenApiRouter::with_openapi(ApiDoc::openapi())
        .routes(routes!(version))
        .routes(routes!(tags))
        .routes(routes!(ps))
        .routes(routes!(show))
        .routes(routes!(pull))
        .routes(routes!(delete))
        .routes(routes!(copy))
        .routes(routes!(chat))
        .routes(routes!(generate))
        .routes(routes!(embed))
        .routes(routes!(embeddings))
        .routes(routes!(create))
        .routes(routes!(push))
        .routes(routes!(blob_head, blob_post))
        .routes(routes!(openapi_json))
        .routes(routes!(index))
        .merge(crate::openai::router())
}

/// The OpenAPI document, generated from the annotations.
pub fn openapi() -> utoipa::openapi::OpenApi {
    let mut doc = api_router().split_for_parts().1;
    doc.info.version = env!("CARGO_PKG_VERSION").into();
    doc
}

/// The whole server. Every route goes through `api_router`, so a route
/// without its annotation does not exist.
pub fn app(state: AppState) -> axum::Router {
    api_router().split_for_parts().0.with_state(state).layer(axum::middleware::from_fn(same_origin_only))
}

/// Refuse what a web page the user happens to visit could send.
///
/// The server binds to loopback, but a browser on the same machine reaches
/// loopback too: any site can POST a `text/plain` body to `/api/pull` (the
/// handlers parse JSON whatever the content type), and a DNS-rebound name can
/// read the answers. Two checks close both, the ones Ollama makes:
/// - `Host` must be a loopback name or an IP literal. A rebound attacker name
///   is neither; an IP literal cannot be rebound.
/// - `Origin`, when a browser sends one, must be a loopback origin or the
///   origin of the `Host` itself (the chat page served from this address).
///
/// Clients that send no `Origin` (curl, the CLI, SDKs) are not browsers and
/// pass. There is no allow-list for other origins yet.
async fn same_origin_only(req: axum::extract::Request, next: axum::middleware::Next) -> axum::response::Response {
    // Read both headers into owned strings first: a borrow of the request
    // held across the await below would make the future not `Send`.
    let verdict = {
        let get = |n: header::HeaderName| req.headers().get(n).map(|v| v.to_str().unwrap_or("\u{fffd}").to_string());
        check_origin(get(header::HOST).as_deref(), get(header::ORIGIN).as_deref())
    };
    match verdict {
        Ok(()) => next.run(req).await,
        Err(why) => (StatusCode::FORBIDDEN, Json(json!({ "error": why }))).into_response(),
    }
}

/// The host part of `authority` (`name`, `name:port`, `[v6]` or `[v6]:port`),
/// lower case.
fn authority_host(authority: &str) -> String {
    let a = authority.trim();
    let h = if let Some(rest) = a.strip_prefix('[') {
        rest.split(']').next().unwrap_or("")
    } else {
        a.rsplit_once(':').filter(|(_, p)| p.chars().all(|c| c.is_ascii_digit())).map_or(a, |(h, _)| h)
    };
    h.to_ascii_lowercase()
}

fn is_loopback_name(host: &str) -> bool {
    host == "localhost"
        || host.ends_with(".localhost")
        || host.parse::<std::net::IpAddr>().is_ok_and(|ip| ip.is_loopback())
}

/// The rule of [`same_origin_only`], on the two header values.
pub fn check_origin(host: Option<&str>, origin: Option<&str>) -> Result<(), String> {
    if let Some(h) = host {
        let name = authority_host(h);
        if !(is_loopback_name(&name) || name.parse::<std::net::IpAddr>().is_ok()) {
            return Err(format!("host '{h}' refused: this server answers loopback names and IP addresses only"));
        }
    }
    if let Some(o) = origin {
        let authority = o.split_once("://").map(|(scheme, rest)| (scheme, rest.split('/').next().unwrap_or("")));
        let ok = match authority {
            Some(("http" | "https", a)) => {
                let name = authority_host(a);
                is_loopback_name(&name) || host.is_some_and(|h| authority_host(h) == name && a.eq_ignore_ascii_case(h.trim()))
            }
            _ => false,
        };
        if !ok {
            return Err(format!("origin '{o}' refused: a web page from another site cannot drive this server"));
        }
    }
    Ok(())
}

/// Serve until Ctrl-C, unloading expired models every second.
pub async fn serve(listener: tokio::net::TcpListener, state: AppState) -> anyhow::Result<()> {
    let models = state.models.clone();
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(std::time::Duration::from_secs(1));
        loop {
            tick.tick().await;
            models.reap(SystemTime::now());
        }
    });
    axum::serve(listener, app(state))
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_loopback_or_same_origin_pages_reach_the_server() {
        // Not browsers: no Origin.
        assert!(check_origin(Some("127.0.0.1:11435"), None).is_ok());
        assert!(check_origin(Some("localhost:11435"), None).is_ok());
        assert!(check_origin(Some("[::1]:11435"), None).is_ok());
        assert!(check_origin(None, None).is_ok());
        // A LAN address the operator bound to is an IP literal.
        assert!(check_origin(Some("192.168.1.20:11435"), Some("http://192.168.1.20:11435")).is_ok());
        // The chat page, and other local pages.
        assert!(check_origin(Some("127.0.0.1:11435"), Some("http://127.0.0.1:11435")).is_ok());
        assert!(check_origin(Some("127.0.0.1:11435"), Some("http://localhost:3000")).is_ok());
        // DNS rebinding: a name that resolves to loopback but is not one.
        assert!(check_origin(Some("evil.example:11435"), None).is_err());
        assert!(check_origin(Some("evil.example:11435"), Some("http://evil.example:11435")).is_err());
        // Cross-site pages.
        assert!(check_origin(Some("127.0.0.1:11435"), Some("https://evil.example")).is_err());
        assert!(check_origin(Some("127.0.0.1:11435"), Some("null")).is_err());
        assert!(check_origin(Some("127.0.0.1:11435"), Some("http://127.0.0.1.evil.example")).is_err());
        assert!(check_origin(Some("192.168.1.20:11435"), Some("http://192.168.1.21:11435")).is_err());
        assert!(check_origin(Some("127.0.0.1:11435"), Some("file:///tmp/x.html")).is_err());
    }

    #[test]
    fn rfc3339_matches_known_instants() {
        let at = |s: u64, n: u32| rfc3339(SystemTime::UNIX_EPOCH + std::time::Duration::new(s, n));
        assert_eq!(at(0, 0), "1970-01-01T00:00:00.000000000Z");
        assert_eq!(at(951_782_400, 5), "2000-02-29T00:00:00.000000005Z");
        assert_eq!(at(1_762_387_200, 0), "2025-11-06T00:00:00.000000000Z");
        assert_eq!(at(253_402_300_799, 0), "9999-12-31T23:59:59.000000000Z");
    }

    #[test]
    fn unknown_options_are_listed_and_known_ones_read() {
        let o: Map<String, Value> =
            serde_json::from_value(json!({"temperature": 0.2, "top_k": 40, "stop": ["x"], "num_ctx": 512, "repeat_penalty": 1.1}))
                .unwrap();
        let mut w = Vec::new();
        let r = options(&Some(o), &mut w).unwrap_or_else(|_| panic!("options parse"));
        assert_eq!(r.temperature, Some(0.2));
        assert_eq!(r.stop, vec!["x".to_string()]);
        assert_eq!(r.num_ctx, Some(512));
        assert_eq!(w.len(), 2);
        assert!(w.iter().any(|m| m.contains("top_k")) && w.iter().any(|m| m.contains("repeat_penalty")));
    }
}
