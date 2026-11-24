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
use crate::engine::{Event, Options, Prompt, Request, ToolPrompt};
use crate::library::ModelName;
use crate::tools::{self, Args, Calls, HistCall, OrderedJson, Piece, ToolDef, ToolFormat, ToolMsg, ToolRole};
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
    /// `system` (or `developer`), `user`, `assistant` or `tool`.
    pub role: String,
    /// A string, an array of `{"type": "text", "text": ...}` parts, or null.
    #[schema(value_type = Object)]
    #[serde(default)]
    pub content: Value,
    /// An assistant message's calls, `[{"id", "type": "function", "function":
    /// {"name", "arguments"}}]`, `arguments` a JSON string.
    #[schema(value_type = Option<Vec<Object>>)]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OrderedJson>>,
    /// The call a `tool` message answers: required on a `tool` message.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
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
    /// Functions the model may call, `{"type": "function", "function":
    /// {"name", "description", "parameters"}}`. Served for the qwen3 and
    /// inkling2 families.
    #[schema(value_type = Option<Vec<Object>>)]
    pub tools: Option<Vec<OrderedJson>>,
    /// `"auto"` (the default) or `"none"`, which offers no tool. `"required"`
    /// and a named function are refused: the decode is not constrained.
    #[schema(value_type = Option<Object>)]
    pub tool_choice: Option<Value>,
    /// `false` is refused: the decode cannot be held to one call.
    pub parallel_tool_calls: Option<bool>,
    #[schema(value_type = Option<Object>)]
    pub response_format: Option<Value>,
    /// Absent or `"none"`: no reasoning; any other level turns it on. The
    /// inkling2 family writes the level into its prompt, as its template does.
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
pub struct FunctionCall {
    pub name: String,
    /// The arguments object as a JSON string.
    pub arguments: String,
}

#[derive(Serialize, Deserialize, ToSchema)]
pub struct OaiToolCall {
    pub id: String,
    /// `function`.
    #[serde(rename = "type")]
    pub kind: String,
    pub function: FunctionCall,
}

#[derive(Serialize, Deserialize, ToSchema)]
pub struct AssistantMessage {
    pub role: String,
    /// Null when the answer is tool calls only.
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OaiToolCall>>,
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
const HARMLESS: &[&str] = &["user", "metadata", "store", "logprobs"];

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
        effort: None,
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
    /// The format whose calls the answer is read for, when tools are offered.
    tools: Option<ToolFormat>,
}

fn tool_call(c: &tools::Call) -> OaiToolCall {
    OaiToolCall {
        id: new_id("call"),
        kind: "function".into(),
        function: FunctionCall { name: c.name.clone(), arguments: c.arguments.dumps() },
    }
}

/// One piece of an answer with tools as a chunk's delta. A call goes out
/// whole, in one chunk, with its `index`: the Vercel AI SDK finishes a call
/// whose first chunk carries arguments that parse, and OpenAI's SDKs append
/// the fragments of one `index`, here only one.
fn piece_delta(p: Piece, index: &mut usize) -> Value {
    match p {
        Piece::Content(t) => json!({ "content": t }),
        Piece::Reasoning(t) => json!({ "reasoning": t }),
        Piece::Call(c) => {
            let k = *index;
            *index += 1;
            let mut v = serde_json::to_value(tool_call(&c)).expect("a call serializes");
            v["index"] = json!(k);
            json!({ "tool_calls": [v] })
        }
        Piece::Malformed { text, .. } => json!({ "content": text }),
    }
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

fn finish_reason(done_reason: &str, calls: usize) -> &'static str {
    match done_reason {
        "length" => "length",
        _ if calls > 0 => "tool_calls",
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
        let mut calls = ctx.tools.map(Calls::new);
        let mut index = 0usize;
        let body = events(rx).flat_map(move |e| {
            let lines: Vec<String> = match e {
                Event::Delta { content, thinking } if calls.is_some() => {
                    let c = calls.as_mut().expect("tools");
                    c.delta(&content, &thinking).into_iter().map(|p| data(&ctx.chunk(piece_delta(p, &mut index), None))).collect()
                }
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
                    let mut out: Vec<String> = match calls.as_mut() {
                        Some(c) => c.finish().into_iter().map(|p| data(&ctx.chunk(piece_delta(p, &mut index), None))).collect(),
                        None => Vec::new(),
                    };
                    let n = calls.as_ref().map_or(0, |c| c.count);
                    out.push(data(&ctx.chunk(json!({}), Some(finish_reason(&s.done_reason, n)))));
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
    let mut calls = ctx.tools.map(Calls::new);
    let mut made: Vec<OaiToolCall> = Vec::new();
    let mut take = |ps: Vec<Piece>, content: &mut String, thinking: &mut String| {
        for p in ps {
            match p {
                Piece::Content(t) | Piece::Malformed { text: t, .. } => content.push_str(&t),
                Piece::Reasoning(t) => thinking.push_str(&t),
                Piece::Call(c) => made.push(tool_call(&c)),
            }
        }
    };
    let mut ev = Box::pin(events(rx));
    while let Some(e) = ev.next().await {
        match e {
            Event::Delta { content: c, thinking: t } => match calls.as_mut() {
                Some(k) => take(k.delta(&c, &t), &mut content, &mut thinking),
                None => {
                    content.push_str(&c);
                    thinking.push_str(&t);
                }
            },
            Event::Error { status, message } => return oai_error(status, message),
            Event::Accepted { .. } => {}
            Event::Done(s) => {
                let mut warnings = ctx.warnings.clone();
                if let Some(k) = calls.as_mut() {
                    take(k.finish(), &mut content, &mut thinking);
                    warnings.extend(k.warnings.iter().cloned());
                }
                let usage = Usage {
                    prompt_tokens: s.prompt_eval_count,
                    completion_tokens: s.eval_count,
                    total_tokens: s.prompt_eval_count + s.eval_count,
                };
                let warnings = (!warnings.is_empty()).then_some(warnings);
                let finish = finish_reason(&s.done_reason, made.len()).to_string();
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
                                content: (!content.is_empty() || made.is_empty()).then_some(content),
                                reasoning: (!thinking.is_empty()).then_some(thinking),
                                tool_calls: (!made.is_empty()).then_some(made),
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

/// The conversation with tools a request carries, or `None` when it offers
/// none and its history holds no tool call: that request takes the plain chat
/// path, unchanged.
fn tool_prompt(st: &AppState, name: &ModelName, req: &ChatCompletionRequest) -> Result<Option<ToolPrompt>, Response> {
    let offered = req.tools.as_deref().unwrap_or_default();
    let in_history =
        req.messages.iter().any(|m| m.role == "tool" || m.tool_calls.as_ref().is_some_and(|c| !c.is_empty()));
    if offered.is_empty() && !in_history {
        return Ok(None);
    }
    let format = family_format(st, name).map_err(|(s, m)| oai_error(s, m))?;
    let mut tools = Vec::with_capacity(offered.len());
    for (i, t) in offered.iter().enumerate() {
        tools.push(ToolDef::parse(i, t).map_err(|e| oai_error(400, e))?);
    }
    match &req.tool_choice {
        None | Some(Value::Null) => {}
        Some(Value::String(c)) if c == "auto" => {}
        // Not offered, so not called: the model is not told of the tools.
        Some(Value::String(c)) if c == "none" => tools.clear(),
        Some(Value::String(c)) if c == "required" => {
            return Err(oai_error(400, "tool_choice \"required\" is not served by llvq: the decode is not constrained, so \
                 a call cannot be forced; send \"auto\""))
        }
        Some(Value::Object(_)) => {
            return Err(oai_error(400, "tool_choice naming a function is not served by llvq: the decode is not \
                 constrained, so a call cannot be forced; send \"auto\""))
        }
        Some(o) => return Err(oai_error(400, format!("tool_choice {o}: \"auto\" or \"none\""))),
    }
    if req.parallel_tool_calls == Some(false) && !tools.is_empty() {
        return Err(oai_error(
            400,
            "parallel_tool_calls false is not served by llvq: the decode cannot be held to one call per answer",
        ));
    }
    let mut msgs = Vec::with_capacity(req.messages.len());
    let mut ids = Vec::with_capacity(req.messages.len());
    for (i, m) in req.messages.iter().enumerate() {
        let role = match m.role.as_str() {
            "system" | "developer" => ToolRole::System,
            "user" => ToolRole::User,
            "assistant" => ToolRole::Assistant,
            "tool" => ToolRole::Tool,
            o => return Err(oai_error(400, format!("messages[{i}]: role {o:?}: the roles are system, user, assistant and tool"))),
        };
        let content = content_text(&m.content).map_err(|e| oai_error(400, format!("messages[{i}]: {e}")))?;
        let mut calls = Vec::new();
        for (k, c) in m.tool_calls.iter().flatten().enumerate() {
            if role != ToolRole::Assistant {
                return Err(oai_error(400, format!("messages[{i}]: only an assistant message carries tool_calls")));
            }
            calls.push(hist_call(c).map_err(|e| oai_error(400, format!("messages[{i}].tool_calls[{k}]: {e}")))?);
        }
        let mut name = None;
        if role == ToolRole::Tool {
            let Some(id) = &m.tool_call_id else {
                return Err(oai_error(400, format!("messages[{i}]: a tool message needs the tool_call_id it answers")));
            };
            name = msgs[..i].iter().rev().flat_map(|p: &ToolMsg| p.calls.iter()).find(|c| c.id.as_ref() == Some(id)).map(|c| c.name.clone());
        }
        ids.push(m.tool_call_id.clone().filter(|_| role == ToolRole::Tool));
        msgs.push(ToolMsg { role, content, calls, name });
    }
    tools::check_results(&msgs, &ids).map_err(|e| oai_error(400, e))?;
    Ok(Some(ToolPrompt { format, tools, msgs }))
}

/// A call of an assistant message, OpenAI's shape. `arguments` is a JSON
/// string, written into the prompt as it is; an object is read too.
fn hist_call(c: &OrderedJson) -> Result<HistCall, String> {
    match c.get("type").and_then(OrderedJson::as_str) {
        None | Some("function") => {}
        Some(t) => return Err(format!("type {t:?}: the type is \"function\"")),
    }
    let f = c.get("function").ok_or("a function object")?;
    let name = f.get("name").and_then(OrderedJson::as_str).filter(|n| !n.is_empty()).ok_or("function.name, a string")?;
    let arguments = match f.get("arguments") {
        Some(OrderedJson::Str(s)) => Args::Text(s.clone()),
        Some(o @ OrderedJson::Obj(_)) => Args::Object(o.clone()),
        _ => return Err("function.arguments, a JSON string".into()),
    };
    Ok(HistCall { id: c.get("id").and_then(OrderedJson::as_str).map(str::to_string), name: name.to_string(), arguments })
}

/// The tool format of a model's family, from its manifest.
pub fn family_format(st: &AppState, name: &ModelName) -> Result<ToolFormat, (u16, String)> {
    match st.models.store.get(name) {
        Err(e) => Err((500, format!("{e:#}"))),
        Ok(None) => Err((404, format!("model '{name}' not found, try pulling it first"))),
        Ok(Some(m)) => ToolFormat::of_family(&m.family).map_err(|e| (400, e)),
    }
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
    if let Some(f) = &req.response_format {
        if f.get("type").and_then(Value::as_str).is_some_and(|t| t != "text") {
            return oai_error(400, "response_format: structured output is not implemented by llvq");
        }
    }
    if req.messages.is_empty() {
        return oai_error(400, "messages: at least one");
    }
    let tool_prompt = match tool_prompt(&st, &name, &req) {
        Ok(p) => p,
        Err(r) => return r,
    };
    let mut msgs = Vec::with_capacity(req.messages.len());
    if tool_prompt.is_none() {
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
    }
    let mut o = match options(
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
    o.effort = req.reasoning_effort.clone();
    let think = req.reasoning_effort.as_deref().is_some_and(|e| e != "none");
    let ctx = Ctx {
        id: new_id("chatcmpl"),
        created: unix(SystemTime::now()),
        model: name.to_string(),
        shape: Shape::Chat,
        include_usage: req.stream_options.as_ref().and_then(|s| s.include_usage).unwrap_or(false),
        warnings: warn_extra(&req.extra),
        tools: tool_prompt.as_ref().filter(|p| !p.tools.is_empty()).map(|p| p.format),
    };
    let prompt = match tool_prompt {
        Some(p) => Prompt::ToolChat(p),
        None => Prompt::Chat(msgs),
    };
    let request = Request { prompt, think, options: o };
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
        tools: None,
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
pub async fn embeddings(_body: Bytes) -> Response {
    oai_error(501, "/v1/embeddings is not implemented by llvq: no embedding head is sealed")
}

#[utoipa::path(post, path = "/v1/responses", tag = "not served", responses((status = 501, body = OaiError)))]
pub async fn responses(_body: Bytes) -> Response {
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
