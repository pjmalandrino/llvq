//! OpenAI's chat format, as Ollama's compatibility layer serves it: the
//! routes under `/v1`, streamed as server-sent events that end with
//! `data: [DONE]`, which is what OpenAI's clients read.
//!
//! The engine is the one `/api/chat` drives; only the shapes differ. Errors
//! are OpenAI's shape, `{"error": {"message", "type", "param", "code"}}`.

// A refusal here is the HTTP response itself, built once a request; boxing
// it would buy nothing.
#![allow(clippy::result_large_err)]

use crate::api::{accepted, events, stop_list, unix, AppState};
use crate::chat::{Msg, Role};
use crate::engine::{Event, Options, Prompt, Request};
use crate::library::ModelName;
use axum::body::{Body, Bytes};
use axum::extract::{Path, State};
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use std::convert::Infallible;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::SystemTime;
use utoipa::ToSchema;
use utoipa_axum::router::OpenApiRouter;
use utoipa_axum::routes;

#[derive(Serialize, Deserialize, ToSchema)]
pub struct OaiErrorBody {
    pub message: String,
    #[serde(rename = "type")]
    pub kind: String,
    #[schema(value_type = Option<String>)]
    pub param: Option<Value>,
    #[schema(value_type = Option<String>)]
    pub code: Option<Value>,
}

#[derive(Serialize, Deserialize, ToSchema)]
pub struct OaiError {
    pub error: OaiErrorBody,
}

pub fn oai_error(status: u16, message: impl Into<String>) -> Response {
    let kind = match status {
        400..=499 => "invalid_request_error",
        501 => "not_implemented",
        _ => "api_error",
    };
    let s = StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    let code = (status == 404).then(|| json!("model_not_found"));
    (s, Json(OaiError { error: OaiErrorBody { message: message.into(), kind: kind.into(), param: None, code } }))
        .into_response()
}

fn parse<T: for<'de> Deserialize<'de>>(body: &Bytes) -> Result<T, Response> {
    serde_json::from_slice(body).map_err(|e| oai_error(400, format!("the request body: {e}")))
}

fn new_id(prefix: &str) -> String {
    static N: AtomicU64 = AtomicU64::new(0);
    let t = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
    format!("{prefix}-{:x}{:04x}", t & 0xffff_ffff_ffff, N.fetch_add(1, Ordering::Relaxed) & 0xffff)
}

#[derive(Serialize, Deserialize, ToSchema)]
pub struct ModelObject {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
}

#[derive(Serialize, Deserialize, ToSchema)]
pub struct ModelList {
    pub object: String,
    pub data: Vec<ModelObject>,
}

#[derive(Serialize, Deserialize, ToSchema)]
pub struct OaiMessage {
    pub role: String,
    /// A string, an array of `{"type": "text", "text": ...}` parts, or null.
    #[schema(value_type = Object)]
    #[serde(default)]
    pub content: Value,
}

#[derive(Serialize, Deserialize, ToSchema)]
pub struct StreamOptions {
    pub include_usage: Option<bool>,
}

#[derive(Serialize, Deserialize, ToSchema)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<OaiMessage>,
    pub stream: Option<bool>,
    pub stream_options: Option<StreamOptions>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub max_tokens: Option<i64>,
    pub max_completion_tokens: Option<i64>,
    pub seed: Option<i64>,
    #[schema(value_type = Option<Object>)]
    pub stop: Option<Value>,
    pub n: Option<u32>,
    #[schema(value_type = Option<Vec<Object>>)]
    pub tools: Option<Vec<Value>>,
    #[schema(value_type = Option<Object>)]
    pub response_format: Option<Value>,
    /// Absent or `"none"`: no reasoning; any other level turns it on.
    pub reasoning_effort: Option<String>,
    /// Every other field: ignored, and listed in `warnings`.
    #[serde(flatten)]
    #[schema(value_type = Object)]
    pub extra: Map<String, Value>,
}

#[derive(Serialize, Deserialize, ToSchema)]
pub struct CompletionRequest {
    pub model: String,
    /// A string, or an array of one string.
    #[schema(value_type = Object)]
    pub prompt: Value,
    pub stream: Option<bool>,
    pub stream_options: Option<StreamOptions>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub max_tokens: Option<i64>,
    pub seed: Option<i64>,
    #[schema(value_type = Option<Object>)]
    pub stop: Option<Value>,
    pub n: Option<u32>,
    pub suffix: Option<String>,
    #[serde(flatten)]
    #[schema(value_type = Object)]
    pub extra: Map<String, Value>,
}

#[derive(Serialize, Deserialize, ToSchema, Clone, Copy)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Serialize, Deserialize, ToSchema)]
pub struct AssistantMessage {
    pub role: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
}

#[derive(Serialize, Deserialize, ToSchema)]
pub struct ChatChoice {
    pub index: u32,
    pub message: AssistantMessage,
    pub finish_reason: String,
}

#[derive(Serialize, Deserialize, ToSchema)]
pub struct ChatCompletion {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub system_fingerprint: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub warnings: Option<Vec<String>>,
}

#[derive(Serialize, Deserialize, ToSchema)]
pub struct CompletionChoice {
    pub index: u32,
    pub text: String,
    #[schema(value_type = Option<Object>)]
    pub logprobs: Option<Value>,
    pub finish_reason: String,
}

#[derive(Serialize, Deserialize, ToSchema)]
pub struct Completion {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub system_fingerprint: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: Usage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub warnings: Option<Vec<String>>,
}

/// One message's text: a string, text parts, or nothing.
fn content_text(v: &Value) -> Result<String, String> {
    match v {
        Value::Null => Ok(String::new()),
        Value::String(s) => Ok(s.clone()),
        Value::Array(parts) => {
            let mut out = String::new();
            for p in parts {
                match (p.get("type").and_then(Value::as_str), p.get("text").and_then(Value::as_str)) {
                    (Some("text"), Some(t)) => out.push_str(t),
                    (kind, _) => {
                        return Err(format!(
                            "content part {:?}: the served models read text parts only",
                            kind.unwrap_or("?")
                        ))
                    }
                }
            }
            Ok(out)
        }
        _ => Err("content: a string or an array of text parts".into()),
    }
}

/// Fields OpenAI clients send that llvq reads as no-ops, and so does not warn
/// about.
const HARMLESS: &[&str] = &["user", "metadata", "store", "parallel_tool_calls", "tool_choice", "logprobs"];

fn warn_extra(extra: &Map<String, Value>) -> Vec<String> {
    extra
        .iter()
        .filter(|(k, v)| !v.is_null() && !HARMLESS.contains(&k.as_str()))
        .map(|(k, _)| format!("{k} is not implemented by llvq and was ignored"))
        .collect()
}

fn options(
    temperature: Option<f32>,
    top_p: Option<f32>,
    max_tokens: Option<i64>,
    seed: Option<i64>,
    stop: &Option<Value>,
    n: Option<u32>,
) -> Result<Options, Response> {
    if n.is_some_and(|n| n != 1) {
        return Err(oai_error(400, "n: the engine decodes one sequence at a time; send n = 1"));
    }
    Ok(Options {
        temperature,
        top_p,
        seed: seed.map(|s| s as u64),
        num_predict: max_tokens,
        stop: stop.as_ref().map(stop_list).transpose().map_err(|e| oai_error(400, e))?.unwrap_or_default(),
        num_ctx: None,
    })
}

fn sse(s: impl futures_util::Stream<Item = String> + Send + 'static) -> Response {
    let body = Body::from_stream(s.map(|line| Ok::<_, Infallible>(Bytes::from(line))));
    Response::builder()
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .body(body)
        .expect("a response builds")
}

fn data(v: &Value) -> String {
    format!("data: {}\n\n", serde_json::to_string(v).expect("a JSON value serializes"))
}

/// Which of the two OpenAI shapes a request answers in.
#[derive(Clone, Copy)]
enum Shape {
    Chat,
    Text,
}

struct Ctx {
    id: String,
    created: u64,
    model: String,
    shape: Shape,
    include_usage: bool,
    warnings: Vec<String>,
}

impl Ctx {
    fn chunk(&self, delta: Value, finish: Option<&str>) -> Value {
        match self.shape {
            Shape::Chat => json!({
                "id": self.id, "object": "chat.completion.chunk", "created": self.created, "model": self.model,
                "system_fingerprint": "llvq",
                "choices": [{ "index": 0, "delta": delta, "finish_reason": finish }],
            }),
            Shape::Text => json!({
                "id": self.id, "object": "text_completion", "created": self.created, "model": self.model,
                "system_fingerprint": "llvq",
                "choices": [{ "index": 0, "text": delta.get("content").cloned().unwrap_or(json!("")),
                               "logprobs": null, "finish_reason": finish }],
            }),
        }
    }
}

fn finish_reason(done_reason: &str) -> &'static str {
    match done_reason {
        "length" => "length",
        _ => "stop",
    }
}

async fn complete(st: &AppState, name: ModelName, request: Request, stream: bool, ctx: Ctx) -> Response {
    let mut rx = match st.models.submit(&name, request, None) {
        Ok(rx) => rx,
        Err(r) => return oai_error(r.status, r.message),
    };
    if let Err((status, message)) = accepted(&mut rx).await {
        return oai_error(status, message);
    }
    if stream {
        let first = match ctx.shape {
            Shape::Chat => Some(data(&ctx.chunk(json!({"role": "assistant", "content": ""}), None))),
            Shape::Text => None,
        };
        let body = events(rx).flat_map(move |e| {
            let lines: Vec<String> = match e {
                Event::Delta { content, thinking } => {
                    let mut d = Map::new();
                    if !content.is_empty() {
                        d.insert("content".into(), json!(content));
                    }
                    if !thinking.is_empty() {
                        d.insert("reasoning".into(), json!(thinking));
                    }
                    vec![data(&ctx.chunk(Value::Object(d), None))]
                }
                Event::Done(s) => {
                    let mut out = vec![data(&ctx.chunk(json!({}), Some(finish_reason(&s.done_reason))))];
                    if ctx.include_usage {
                        let u = Usage {
                            prompt_tokens: s.prompt_eval_count,
                            completion_tokens: s.eval_count,
                            total_tokens: s.prompt_eval_count + s.eval_count,
                        };
                        let mut c = ctx.chunk(json!({}), None);
                        c["choices"] = json!([]);
                        c["usage"] = serde_json::to_value(u).expect("usage serializes");
                        out.push(data(&c));
                    }
                    out.push("data: [DONE]\n\n".into());
                    out
                }
                Event::Error { status, message } => vec![
                    data(&json!({"error": {"message": message, "type": "api_error", "code": status}})),
                    "data: [DONE]\n\n".into(),
                ],
                Event::Accepted { .. } => vec![],
            };
            futures_util::stream::iter(lines)
        });
        return sse(futures_util::stream::iter(first).chain(body));
    }
    let (mut content, mut thinking) = (String::new(), String::new());
    let mut ev = Box::pin(events(rx));
    while let Some(e) = ev.next().await {
        match e {
            Event::Delta { content: c, thinking: t } => {
                content.push_str(&c);
                thinking.push_str(&t);
            }
            Event::Error { status, message } => return oai_error(status, message),
            Event::Accepted { .. } => {}
            Event::Done(s) => {
                let usage = Usage {
                    prompt_tokens: s.prompt_eval_count,
                    completion_tokens: s.eval_count,
                    total_tokens: s.prompt_eval_count + s.eval_count,
                };
                let warnings = (!ctx.warnings.is_empty()).then(|| ctx.warnings.clone());
                let finish = finish_reason(&s.done_reason).to_string();
                return match ctx.shape {
                    Shape::Chat => Json(ChatCompletion {
                        id: ctx.id,
                        object: "chat.completion".into(),
                        created: ctx.created,
                        model: ctx.model,
                        system_fingerprint: "llvq".into(),
                        choices: vec![ChatChoice {
                            index: 0,
                            message: AssistantMessage {
                                role: "assistant".into(),
                                content,
                                reasoning: (!thinking.is_empty()).then_some(thinking),
                            },
                            finish_reason: finish,
                        }],
                        usage,
                        warnings,
                    })
                    .into_response(),
                    Shape::Text => Json(Completion {
                        id: ctx.id,
                        object: "text_completion".into(),
                        created: ctx.created,
                        model: ctx.model,
                        system_fingerprint: "llvq".into(),
                        choices: vec![CompletionChoice { index: 0, text: content, logprobs: None, finish_reason: finish }],
                        usage,
                        warnings,
                    })
                    .into_response(),
                };
            }
        }
    }
    oai_error(500, "the engine stopped before the end of the answer")
}

fn model_name(s: &str) -> Result<ModelName, Response> {
    ModelName::parse(s).map_err(|e| oai_error(400, e))
}

/// A chat completion, OpenAI's shape.
#[utoipa::path(post, path = "/v1/chat/completions", tag = "openai", request_body = ChatCompletionRequest,
    responses((status = 200, description = "ChatCompletion, or server-sent chunks ending in data: [DONE] when stream is true",
               body = ChatCompletion), (status = 400, body = OaiError), (status = 404, body = OaiError)))]
pub async fn chat_completions(State(st): State<AppState>, body: Bytes) -> Response {
    let req: ChatCompletionRequest = match parse(&body) {
        Ok(r) => r,
        Err(r) => return r,
    };
    let name = match model_name(&req.model) {
        Ok(n) => n,
        Err(r) => return r,
    };
    if req.tools.as_ref().is_some_and(|t| !t.is_empty()) {
        return oai_error(400, "tools: tool calls are not served by llvq");
    }
    if let Some(f) = &req.response_format {
        if f.get("type").and_then(Value::as_str).is_some_and(|t| t != "text") {
            return oai_error(400, "response_format: structured output is not implemented by llvq");
        }
    }
    let mut msgs = Vec::with_capacity(req.messages.len());
    for m in &req.messages {
        let role = match Role::parse(&m.role) {
            Ok(r) => r,
            Err(e) => return oai_error(400, e),
        };
        match content_text(&m.content) {
            Ok(content) => msgs.push(Msg { role, content }),
            Err(e) => return oai_error(400, e),
        }
    }
    if msgs.is_empty() {
        return oai_error(400, "messages: at least one");
    }
    let o = match options(
        req.temperature,
        req.top_p,
        req.max_completion_tokens.or(req.max_tokens),
        req.seed,
        &req.stop,
        req.n,
    ) {
        Ok(o) => o,
        Err(r) => return r,
    };
    let think = req.reasoning_effort.as_deref().is_some_and(|e| e != "none");
    let ctx = Ctx {
        id: new_id("chatcmpl"),
        created: unix(SystemTime::now()),
        model: name.to_string(),
        shape: Shape::Chat,
        include_usage: req.stream_options.as_ref().and_then(|s| s.include_usage).unwrap_or(false),
        warnings: warn_extra(&req.extra),
    };
    let request = Request { prompt: Prompt::Chat(msgs), think, options: o };
    complete(&st, name, request, req.stream.unwrap_or(false), ctx).await
}

/// A text completion, OpenAI's shape. The prompt goes through Qwen3's
/// template as a user turn, as Ollama's compatibility layer sends it.
#[utoipa::path(post, path = "/v1/completions", tag = "openai", request_body = CompletionRequest,
    responses((status = 200, description = "Completion, or server-sent chunks ending in data: [DONE] when stream is true",
               body = Completion), (status = 400, body = OaiError), (status = 404, body = OaiError)))]
pub async fn completions(State(st): State<AppState>, body: Bytes) -> Response {
    let req: CompletionRequest = match parse(&body) {
        Ok(r) => r,
        Err(r) => return r,
    };
    let name = match model_name(&req.model) {
        Ok(n) => n,
        Err(r) => return r,
    };
    let prompt = match &req.prompt {
        Value::String(s) => s.clone(),
        Value::Array(a) if a.len() == 1 && a[0].is_string() => a[0].as_str().unwrap_or_default().to_string(),
        _ => return oai_error(400, "prompt: one string"),
    };
    if prompt.is_empty() {
        return oai_error(400, "prompt: empty");
    }
    if req.suffix.as_ref().is_some_and(|s| !s.is_empty()) {
        return oai_error(400, "suffix: fill-in-the-middle is not served by llvq");
    }
    let o = match options(req.temperature, req.top_p, req.max_tokens, req.seed, &req.stop, req.n) {
        Ok(o) => o,
        Err(r) => return r,
    };
    let ctx = Ctx {
        id: new_id("cmpl"),
        created: unix(SystemTime::now()),
        model: name.to_string(),
        shape: Shape::Text,
        include_usage: req.stream_options.as_ref().and_then(|s| s.include_usage).unwrap_or(false),
        warnings: warn_extra(&req.extra),
    };
    let request = Request { prompt: Prompt::Templated { system: None, prompt }, think: false, options: o };
    complete(&st, name, request, req.stream.unwrap_or(false), ctx).await
}

fn model_object(n: &ModelName, modified: u64) -> ModelObject {
    ModelObject { id: n.to_string(), object: "model".into(), created: modified, owned_by: "library".into() }
}

/// The models in the store, OpenAI's shape.
#[utoipa::path(get, path = "/v1/models", tag = "openai", responses((status = 200, body = ModelList)))]
pub async fn models(State(st): State<AppState>) -> Response {
    match st.models.store.list() {
        Err(e) => oai_error(500, format!("{e:#}")),
        Ok(list) => Json(ModelList {
            object: "list".into(),
            data: list.iter().map(|(n, m)| model_object(n, m.modified_at)).collect(),
        })
        .into_response(),
    }
}

/// One model, OpenAI's shape.
#[utoipa::path(get, path = "/v1/models/{model}", tag = "openai",
    params(("model" = String, Path, description = "name:tag")),
    responses((status = 200, body = ModelObject), (status = 404, body = OaiError)))]
pub async fn model(State(st): State<AppState>, Path(model): Path<String>) -> Response {
    let name = match model_name(&model) {
        Ok(n) => n,
        Err(r) => return r,
    };
    match st.models.store.get(&name) {
        Err(e) => oai_error(500, format!("{e:#}")),
        Ok(None) => oai_error(404, format!("model '{name}' not found")),
        Ok(Some(m)) => Json(model_object(&name, m.modified_at)).into_response(),
    }
}

#[utoipa::path(post, path = "/v1/embeddings", tag = "not served", responses((status = 501, body = OaiError)))]
pub async fn embeddings() -> Response {
    oai_error(501, "/v1/embeddings is not implemented by llvq: no embedding head is sealed")
}

#[utoipa::path(post, path = "/v1/responses", tag = "not served", responses((status = 501, body = OaiError)))]
pub async fn responses() -> Response {
    oai_error(501, "/v1/responses is not implemented by llvq: use /v1/chat/completions")
}

pub fn router() -> OpenApiRouter<AppState> {
    OpenApiRouter::new()
        .routes(routes!(chat_completions))
        .routes(routes!(completions))
        .routes(routes!(models))
        .routes(routes!(model))
        .routes(routes!(embeddings))
        .routes(routes!(responses))
}
