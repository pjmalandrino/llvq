//! Tool calls: the seam between a model family's tool format and the routes.
//!
//! A family's format does two things ([`ToolFormat`]). It renders the tool
//! definitions and the tool-call and tool-result history into the prompt, as
//! text pieces the engine encodes one by one. And it gives a [`CallParser`],
//! which reads the answer as it streams and separates content, reasoning and
//! tool calls. The routes see only [`Piece`]s; nothing outside this module
//! knows a family's markers.
//!
//! Qwen3 is the one family served. Its format is the official chat template of
//! the Qwen3 checkpoints (`tokenizer_config.json` of `Qwen/Qwen3-4B`, snapshot
//! `1cfa9a7208912126459214e8b04321603b3df60c`, sha256 `d5d09f07…8101`): the
//! tools are listed in the system turn inside `<tools></tools>`, each call is
//! `<tool_call>\n{"name": …, "arguments": …}\n</tool_call>`, and the results
//! come back in a user turn, each inside `<tool_response>`. The renderer is
//! checked byte for byte against that template as `transformers` renders it
//! (`tests/fixtures/qwen3-tools-reference.json`, made by
//! `ops/qwen3_tools_reference.py`).
//!
//! What Inkling 2's format will need, read in its `chat_template.jinja`: the
//! tools declared in a system message as one compact JSON array with sorted
//! keys (`<|message_system|>tool_declare<|content_xml|>…<|end_message|>`); a
//! tool result headed by the function's name, which the template finds from
//! `tool_call_id` (`ToolMsg::name` carries it); each call as its own model
//! message, `<|message_model|>NAME<|content_invoke_tool_json|>{"name":…,"args":{…}}<|end_message|>`,
//! with the arguments an object (a JSON string is refused by the template),
//! keys sorted, compact; every assistant turn closed by
//! `<|content_model_end_sampling|>`; and its reasoning delimited in the text by
//! `<|content_thinking|>`, not routed by token ids as Qwen3's is, which is why
//! a parser reads the reasoning stream as well as the content.

use serde::de::{Deserialize, Deserializer, MapAccess, SeqAccess, Visitor};
use serde::ser::{Serialize, SerializeMap, SerializeSeq, Serializer};
use std::fmt;

// ---------------------------------------------------------------------------
// JSON that keeps the client's key order.
// ---------------------------------------------------------------------------

/// A JSON value that keeps object keys in the order they were written.
///
/// `serde_json::Value` sorts keys (the workspace does not enable
/// `preserve_order`, and turning it on would change every crate's output).
/// Qwen3's template writes each tool definition through Python's `json.dumps`,
/// in the client's order, so the order is part of the prompt.
#[derive(Debug, Clone, PartialEq)]
pub enum OrderedJson {
    Null,
    Bool(bool),
    /// The number as Python's `json.dumps` writes it.
    Num(String),
    Str(String),
    Arr(Vec<OrderedJson>),
    Obj(Vec<(String, OrderedJson)>),
}

impl OrderedJson {
    pub fn get(&self, key: &str) -> Option<&OrderedJson> {
        match self {
            OrderedJson::Obj(kv) => kv.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            OrderedJson::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn is_null(&self) -> bool {
        matches!(self, OrderedJson::Null)
    }

    /// Python's `json.dumps(x, ensure_ascii=False)`: `", "` and `": "` as
    /// separators, keys in their order, non-ASCII written as it is. This is
    /// the `tojson` filter of `transformers`' chat templates.
    pub fn dumps(&self) -> String {
        let mut s = String::new();
        self.dump_into(&mut s);
        s
    }

    fn dump_into(&self, s: &mut String) {
        match self {
            OrderedJson::Null => s.push_str("null"),
            OrderedJson::Bool(b) => s.push_str(if *b { "true" } else { "false" }),
            OrderedJson::Num(n) => s.push_str(n),
            OrderedJson::Str(t) => py_string(t, s),
            OrderedJson::Arr(a) => {
                s.push('[');
                for (i, v) in a.iter().enumerate() {
                    if i > 0 {
                        s.push_str(", ");
                    }
                    v.dump_into(s);
                }
                s.push(']');
            }
            OrderedJson::Obj(kv) => {
                s.push('{');
                for (i, (k, v)) in kv.iter().enumerate() {
                    if i > 0 {
                        s.push_str(", ");
                    }
                    py_string(k, s);
                    s.push_str(": ");
                    v.dump_into(s);
                }
                s.push('}');
            }
        }
    }

    /// The value as `serde_json` holds it (keys sorted), for a response.
    pub fn to_value(&self) -> serde_json::Value {
        serde_json::from_str(&self.dumps()).expect("a dumped value parses")
    }
}

/// A string as Python's `json.dumps(ensure_ascii=False)` writes it.
fn py_string(t: &str, s: &mut String) {
    s.push('"');
    for c in t.chars() {
        match c {
            '"' => s.push_str("\\\""),
            '\\' => s.push_str("\\\\"),
            '\n' => s.push_str("\\n"),
            '\r' => s.push_str("\\r"),
            '\t' => s.push_str("\\t"),
            '\u{08}' => s.push_str("\\b"),
            '\u{0c}' => s.push_str("\\f"),
            c if (c as u32) < 0x20 => s.push_str(&format!("\\u{:04x}", c as u32)),
            c => s.push(c),
        }
    }
    s.push('"');
}

/// A float as Python's `repr` writes it: `1.0`, `0.5`, `1e+16`, `1e-05`.
fn py_float(f: f64) -> String {
    if f.is_nan() {
        return "NaN".into();
    }
    if f.is_infinite() {
        return if f > 0.0 { "Infinity" } else { "-Infinity" }.into();
    }
    let a = f.abs();
    if a == 0.0 || (1e-4..1e16).contains(&a) {
        let s = format!("{f:?}");
        return if s.contains('.') || s.contains('e') { s } else { s + ".0" };
    }
    let s = format!("{f:e}");
    let (m, e) = s.split_once('e').expect("an exponent form");
    let e: i32 = e.parse().expect("an integer exponent");
    format!("{m}e{}{:02}", if e < 0 { '-' } else { '+' }, e.abs())
}

impl<'de> Deserialize<'de> for OrderedJson {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = OrderedJson;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a JSON value")
            }
            fn visit_unit<E>(self) -> Result<OrderedJson, E> {
                Ok(OrderedJson::Null)
            }
            fn visit_none<E>(self) -> Result<OrderedJson, E> {
                Ok(OrderedJson::Null)
            }
            fn visit_some<D: Deserializer<'de>>(self, d: D) -> Result<OrderedJson, D::Error> {
                OrderedJson::deserialize(d)
            }
            fn visit_bool<E>(self, b: bool) -> Result<OrderedJson, E> {
                Ok(OrderedJson::Bool(b))
            }
            fn visit_i64<E>(self, n: i64) -> Result<OrderedJson, E> {
                Ok(OrderedJson::Num(n.to_string()))
            }
            fn visit_u64<E>(self, n: u64) -> Result<OrderedJson, E> {
                Ok(OrderedJson::Num(n.to_string()))
            }
            fn visit_f64<E>(self, f: f64) -> Result<OrderedJson, E> {
                Ok(OrderedJson::Num(py_float(f)))
            }
            fn visit_str<E>(self, s: &str) -> Result<OrderedJson, E> {
                Ok(OrderedJson::Str(s.to_string()))
            }
            fn visit_string<E>(self, s: String) -> Result<OrderedJson, E> {
                Ok(OrderedJson::Str(s))
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut a: A) -> Result<OrderedJson, A::Error> {
                let mut v = Vec::new();
                while let Some(x) = a.next_element()? {
                    v.push(x);
                }
                Ok(OrderedJson::Arr(v))
            }
            fn visit_map<A: MapAccess<'de>>(self, mut a: A) -> Result<OrderedJson, A::Error> {
                let mut v: Vec<(String, OrderedJson)> = Vec::new();
                while let Some((k, x)) = a.next_entry::<String, OrderedJson>()? {
                    // Python keeps the last of two equal keys, at the place of
                    // the first.
                    match v.iter_mut().find(|(key, _)| *key == k) {
                        Some(slot) => slot.1 = x,
                        None => v.push((k, x)),
                    }
                }
                Ok(OrderedJson::Obj(v))
            }
        }
        d.deserialize_any(V)
    }
}

impl Serialize for OrderedJson {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            OrderedJson::Null => s.serialize_unit(),
            OrderedJson::Bool(b) => s.serialize_bool(*b),
            OrderedJson::Num(n) => match (n.parse::<i64>(), n.parse::<u64>(), n.parse::<f64>()) {
                (Ok(i), _, _) => s.serialize_i64(i),
                (_, Ok(u), _) => s.serialize_u64(u),
                (_, _, Ok(f)) => s.serialize_f64(f),
                _ => Err(serde::ser::Error::custom(format!("{n}: not a number"))),
            },
            OrderedJson::Str(t) => s.serialize_str(t),
            OrderedJson::Arr(a) => {
                let mut q = s.serialize_seq(Some(a.len()))?;
                for v in a {
                    q.serialize_element(v)?;
                }
                q.end()
            }
            OrderedJson::Obj(kv) => {
                let mut m = s.serialize_map(Some(kv.len()))?;
                for (k, v) in kv {
                    m.serialize_entry(k, v)?;
                }
                m.end()
            }
        }
    }
}

// ---------------------------------------------------------------------------
// What a request carries.
// ---------------------------------------------------------------------------

/// One function the model may call, as the client wrote it.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolDef {
    pub name: String,
    /// The whole definition, `{"type": "function", "function": {...}}`.
    pub json: OrderedJson,
}

impl ToolDef {
    /// A definition of OpenAI's and Ollama's shape. Any other type of tool is
    /// refused by name: the served models are prompted with functions only.
    pub fn parse(i: usize, v: &OrderedJson) -> Result<Self, String> {
        let at = format!("tools[{i}]");
        match v.get("type").and_then(OrderedJson::as_str) {
            Some("function") => {}
            Some(t) => return Err(format!("{at}: type {t:?} is not served by llvq; the type is \"function\"")),
            None => return Err(format!("{at}: a type, \"function\"")),
        }
        let f = v.get("function").ok_or_else(|| format!("{at}: a function object"))?;
        let name = f.get("name").and_then(OrderedJson::as_str).filter(|n| !n.is_empty());
        let name = name.ok_or_else(|| format!("{at}: function.name, a non-empty string"))?;
        if let Some(p) = f.get("parameters").filter(|p| !p.is_null()) {
            if !matches!(p, OrderedJson::Obj(_)) {
                return Err(format!("{at}: function.parameters, a JSON schema object"));
            }
        }
        Ok(ToolDef { name: name.to_string(), json: v.clone() })
    }
}

/// The arguments of a call in the history, in the form the client sent.
#[derive(Debug, Clone, PartialEq)]
pub enum Args {
    /// OpenAI's form, a JSON string: Qwen3's template writes it verbatim.
    Text(String),
    /// Ollama's form, an object: Qwen3's template writes it through `tojson`.
    Object(OrderedJson),
}

/// A call an assistant message of the history made.
#[derive(Debug, Clone, PartialEq)]
pub struct HistCall {
    pub id: Option<String>,
    pub name: String,
    pub arguments: Args,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolRole {
    System,
    User,
    Assistant,
    Tool,
}

/// A message of a conversation that uses tools.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolMsg {
    pub role: ToolRole,
    pub content: String,
    /// The calls of an assistant message.
    pub calls: Vec<HistCall>,
    /// The function a tool result answers, when the client named it or its
    /// call: Qwen3's template does not write it, Inkling's heads the result
    /// with it.
    pub name: Option<String>,
}

/// Check that every tool result follows an assistant message that made calls,
/// through other tool results, and, when `ids` are given, that it names one of
/// those calls. Qwen3 pairs a result with its call by position only.
pub fn check_results(msgs: &[ToolMsg], ids: &[Option<String>]) -> Result<(), String> {
    let mut open: Option<&[HistCall]> = None;
    for (i, m) in msgs.iter().enumerate() {
        match m.role {
            ToolRole::Assistant => open = (!m.calls.is_empty()).then_some(m.calls.as_slice()),
            ToolRole::Tool => {
                let calls = open.ok_or_else(|| {
                    format!("messages[{i}]: a tool result must follow an assistant message that made tool calls")
                })?;
                if let Some(Some(id)) = ids.get(i) {
                    if !calls.iter().any(|c| c.id.as_deref() == Some(id.as_str())) {
                        return Err(format!(
                            "messages[{i}]: tool_call_id {id:?} names no call of the assistant message before it"
                        ));
                    }
                }
            }
            _ => open = None,
        }
    }
    Ok(())
}

/// A call the model made: a function name and an arguments object.
#[derive(Debug, Clone, PartialEq)]
pub struct Call {
    pub name: String,
    pub arguments: OrderedJson,
}

/// What a parser hands to the routes.
#[derive(Debug, Clone, PartialEq)]
pub enum Piece {
    Content(String),
    Reasoning(String),
    Call(Call),
    /// A call block that does not read as a call: its text, markers included,
    /// and why. The routes send it as content with a warning, so the client
    /// sees what the model wrote.
    Malformed { text: String, why: String },
}

/// Reads an answer as it streams. `push` takes the new text of the content
/// and of the reasoning (one may be empty) and returns what can be sent now;
/// `finish` returns what was held at the end.
pub trait CallParser: Send {
    fn push(&mut self, content: &str, reasoning: &str) -> Vec<Piece>;
    fn finish(&mut self) -> Vec<Piece>;
}

// ---------------------------------------------------------------------------
// The families.
// ---------------------------------------------------------------------------

/// A model family's tool format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolFormat {
    Qwen3,
}

impl ToolFormat {
    /// The format of a manifest's family, or why there is none.
    pub fn of_family(family: &str) -> Result<Self, String> {
        match family {
            "qwen3" => Ok(ToolFormat::Qwen3),
            f => Err(format!("tools: llvq has no tool format for the model family {f:?}; it has one for qwen3")),
        }
    }

    /// The prompt of `msgs` with `tools` offered, as text pieces: about one a
    /// message (Qwen3 adds the tools to the system turn, or opens a system
    /// turn for them). The engine encodes each piece alone; every cut between
    /// two pieces falls next to an added token, so the ids are those of the
    /// whole text.
    pub fn render(self, tools: &[ToolDef], msgs: &[ToolMsg]) -> Vec<String> {
        match self {
            ToolFormat::Qwen3 => qwen3::render(tools, msgs),
        }
    }

    /// The piece an answer the model wrote as `raw` stands for in the next
    /// request's history: the key under which the engine keeps its ids.
    pub fn answer_piece(self, raw: &str) -> String {
        match self {
            ToolFormat::Qwen3 => format!("<|im_start|>assistant\n{raw}<|im_end|>\n"),
        }
    }

    pub fn parser(self) -> Box<dyn CallParser> {
        match self {
            ToolFormat::Qwen3 => Box::new(MarkerParser::new("<tool_call>", "</tool_call>", qwen3::parse_call)),
        }
    }
}

mod qwen3 {
    //! Qwen3's official template, line for line. Each comment quotes the
    //! template branch it ports.

    use super::{Args, Call, OrderedJson, ToolDef, ToolMsg, ToolRole};

    const TOOLS_HEAD: &str = "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n\
        You are provided with function signatures within <tools></tools> XML tags:\n<tools>";
    const TOOLS_TAIL: &str = "\n</tools>\n\nFor each function call, return a json object with function name and \
        arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \
        \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n";

    fn is_tool_response(s: &str) -> bool {
        s.starts_with("<tool_response>") && s.ends_with("</tool_response>")
    }

    pub fn render(tools: &[ToolDef], msgs: &[ToolMsg]) -> Vec<String> {
        let mut out = Vec::with_capacity(msgs.len() + 1);
        let first_system = msgs.first().is_some_and(|m| m.role == ToolRole::System);
        // `{%- if tools %}`: the system turn carries the tools, after the
        // system message when there is one.
        if !tools.is_empty() {
            let mut s = String::from("<|im_start|>system\n");
            if first_system {
                s.push_str(&msgs[0].content);
                s.push_str("\n\n");
            }
            s.push_str(TOOLS_HEAD);
            for t in tools {
                s.push('\n');
                s.push_str(&t.json.dumps());
            }
            s.push_str(TOOLS_TAIL);
            out.push(s);
        } else if first_system {
            out.push(format!("<|im_start|>system\n{}<|im_end|>\n", msgs[0].content));
        }
        // `ns.last_query_index`: the last user message that is not a tool
        // response written as a user message.
        let last_query = msgs
            .iter()
            .rposition(|m| m.role == ToolRole::User && !is_tool_response(&m.content))
            .unwrap_or(msgs.len().saturating_sub(1));
        let last = msgs.len().saturating_sub(1);
        for (i, m) in msgs.iter().enumerate() {
            let piece = match m.role {
                ToolRole::System if i == 0 => continue,
                // `(message.role == "user") or (message.role == "system" and not loop.first)`
                ToolRole::User | ToolRole::System => {
                    let role = if m.role == ToolRole::User { "user" } else { "system" };
                    format!("<|im_start|>{role}\n{}<|im_end|>\n", m.content)
                }
                ToolRole::Assistant => assistant(m, i > last_query, i == last),
                // `message.role == "tool"`: consecutive results share one user turn.
                ToolRole::Tool => {
                    let mut s = String::new();
                    if i == 0 || msgs[i - 1].role != ToolRole::Tool {
                        s.push_str("<|im_start|>user");
                    }
                    s.push_str("\n<tool_response>\n");
                    s.push_str(&m.content);
                    s.push_str("\n</tool_response>");
                    if i == last || msgs[i + 1].role != ToolRole::Tool {
                        s.push_str("<|im_end|>\n");
                    }
                    s
                }
            };
            out.push(piece);
        }
        out
    }

    /// `message.role == "assistant"`.
    fn assistant(m: &ToolMsg, after_query: bool, last: bool) -> String {
        let mut content = m.content.as_str();
        let mut reasoning = "";
        // `{%- if '</think>' in content %}`
        if content.contains("</think>") {
            let head = content.split("</think>").next().unwrap_or("");
            reasoning = head.trim_end_matches('\n').rsplit("<think>").next().unwrap_or("").trim_start_matches('\n');
            content = content.rsplit("</think>").next().unwrap_or("").trim_start_matches('\n');
        }
        let mut s = String::from("<|im_start|>assistant\n");
        if after_query && (last || !reasoning.is_empty()) {
            s.push_str("<think>\n");
            s.push_str(reasoning.trim_matches('\n'));
            s.push_str("\n</think>\n\n");
            s.push_str(content.trim_start_matches('\n'));
        } else {
            s.push_str(content);
        }
        for (k, c) in m.calls.iter().enumerate() {
            // `(loop.first and content) or (not loop.first)`
            if k > 0 || !content.is_empty() {
                s.push('\n');
            }
            s.push_str("<tool_call>\n{\"name\": \"");
            s.push_str(&c.name);
            s.push_str("\", \"arguments\": ");
            match &c.arguments {
                Args::Text(t) => s.push_str(t),
                Args::Object(j) => s.push_str(&j.dumps()),
            }
            s.push_str("}\n</tool_call>");
        }
        s.push_str("<|im_end|>\n");
        s
    }

    /// The body of a `<tool_call>` block: `{"name": ..., "arguments": {...}}`.
    /// Arguments written as a string that holds a JSON object are read as
    /// that object; absent arguments are a call with none.
    pub fn parse_call(body: &str) -> Result<Call, String> {
        let v: OrderedJson = serde_json::from_str(body.trim()).map_err(|e| format!("not JSON: {e}"))?;
        let name = v.get("name").and_then(OrderedJson::as_str).filter(|n| !n.is_empty());
        let name = name.ok_or("no \"name\" string")?.to_string();
        let arguments = match v.get("arguments") {
            None | Some(OrderedJson::Null) => OrderedJson::Obj(Vec::new()),
            Some(o @ OrderedJson::Obj(_)) => o.clone(),
            Some(OrderedJson::Str(s)) => match serde_json::from_str::<OrderedJson>(s) {
                Ok(o @ OrderedJson::Obj(_)) => o,
                _ => return Err("\"arguments\" is a string that does not hold a JSON object".into()),
            },
            Some(_) => return Err("\"arguments\" is not an object".into()),
        };
        Ok(Call { name, arguments })
    }
}

/// A call format of an opening and a closing marker around a body: Qwen3's
/// `<tool_call>…</tool_call>`, and any family that writes calls so.
///
/// The text outside the blocks is content. A tail that could still grow into
/// the opening marker is held, so a marker split across chunks is never sent
/// as content; a block is held whole until its closing marker, so a closing
/// marker split across chunks is found once its last byte arrives. Whitespace
/// next to a call is dropped: the template writes the newline between two
/// calls, and between a text and its first call, itself. Whitespace at the end
/// of an answer that made no call is content. Reasoning is passed through: a
/// marker inside it is reasoning, never a call.
pub struct MarkerParser {
    open: &'static str,
    close: &'static str,
    parse: fn(&str) -> Result<Call, String>,
    held: String,
    inside: bool,
    /// Trailing whitespace of the content, held until it is known whether a
    /// call follows.
    ws: String,
    calls: usize,
}

impl MarkerParser {
    pub fn new(open: &'static str, close: &'static str, parse: fn(&str) -> Result<Call, String>) -> Self {
        MarkerParser { open, close, parse, held: String::new(), inside: false, ws: String::new(), calls: 0 }
    }

    fn text(&mut self, t: &str, out: &mut Vec<Piece>) {
        if t.is_empty() {
            return;
        }
        let body = t.trim_end().len();
        if body == 0 {
            self.ws.push_str(t);
            return;
        }
        let mut s = std::mem::take(&mut self.ws);
        s.push_str(&t[..body]);
        out.push(Piece::Content(s));
        self.ws = t[body..].to_string();
    }
}

impl CallParser for MarkerParser {
    fn push(&mut self, content: &str, reasoning: &str) -> Vec<Piece> {
        let mut out = Vec::new();
        if !reasoning.is_empty() {
            out.push(Piece::Reasoning(reasoning.to_string()));
        }
        self.held.push_str(content);
        loop {
            if !self.inside {
                if let Some(i) = self.held.find(self.open) {
                    let before = self.held[..i].to_string();
                    self.text(&before, &mut out);
                    self.ws.clear();
                    self.held.drain(..i + self.open.len());
                    self.inside = true;
                    continue;
                }
                // The longest tail that is a start of the opening marker.
                let keep = (1..self.open.len()).rev().find(|&k| self.held.ends_with(&self.open[..k])).unwrap_or(0);
                let cut = self.held.len() - keep;
                let free: String = self.held.drain(..cut).collect();
                self.text(&free, &mut out);
                break;
            }
            let Some(j) = self.held.find(self.close) else { break };
            let body: String = self.held.drain(..j).collect();
            self.held.drain(..self.close.len());
            self.inside = false;
            match (self.parse)(&body) {
                Ok(c) => {
                    self.calls += 1;
                    out.push(Piece::Call(c));
                }
                Err(why) => out.push(Piece::Malformed { text: format!("{}{body}{}", self.open, self.close), why }),
            }
        }
        out
    }

    fn finish(&mut self) -> Vec<Piece> {
        let mut out = Vec::new();
        let held = std::mem::take(&mut self.held);
        if self.inside {
            self.inside = false;
            out.push(Piece::Malformed {
                text: format!("{}{held}", self.open),
                why: format!("the answer ended before {}", self.close),
            });
        } else {
            self.text(&held, &mut out);
        }
        let ws = std::mem::take(&mut self.ws);
        if self.calls == 0 && !ws.is_empty() {
            out.push(Piece::Content(ws));
        }
        out
    }
}

/// A parser and what the routes need around it: the number of calls, for
/// `finish_reason`, and a warning for each block that was not a call.
pub struct Calls {
    parser: Box<dyn CallParser>,
    pub count: usize,
    pub warnings: Vec<String>,
}

impl Calls {
    pub fn new(format: ToolFormat) -> Self {
        Calls { parser: format.parser(), count: 0, warnings: Vec::new() }
    }

    fn settle(&mut self, pieces: Vec<Piece>) -> Vec<Piece> {
        pieces
            .into_iter()
            .map(|p| match p {
                Piece::Malformed { text, why } => {
                    self.warnings.push(format!("a tool call the model wrote was returned as content: {why}"));
                    Piece::Content(text)
                }
                Piece::Call(c) => {
                    self.count += 1;
                    Piece::Call(c)
                }
                p => p,
            })
            .collect()
    }

    pub fn delta(&mut self, content: &str, reasoning: &str) -> Vec<Piece> {
        let p = self.parser.push(content, reasoning);
        self.settle(p)
    }

    pub fn finish(&mut self) -> Vec<Piece> {
        let p = self.parser.finish();
        self.settle(p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(chunks: &[&str]) -> Vec<Piece> {
        let mut p = ToolFormat::Qwen3.parser();
        let mut out = Vec::new();
        for c in chunks {
            out.extend(p.push(c, ""));
        }
        out.extend(p.finish());
        merge(out)
    }

    /// Adjacent content pieces as one, so a test reads what a client shows.
    fn merge(ps: Vec<Piece>) -> Vec<Piece> {
        let mut out: Vec<Piece> = Vec::new();
        for p in ps {
            match (out.last_mut(), p) {
                (Some(Piece::Content(a)), Piece::Content(b)) => a.push_str(&b),
                (_, p) => out.push(p),
            }
        }
        out
    }

    fn call(name: &str, args: &str) -> Piece {
        Piece::Call(Call { name: name.into(), arguments: serde_json::from_str(args).unwrap() })
    }

    const ONE: &str = "<tool_call>\n{\"name\": \"read_file\", \"arguments\": {\"path\": \"a.txt\"}}\n</tool_call>";

    #[test]
    fn a_call_split_at_every_byte_reads_as_the_whole() {
        let whole = run(&[ONE]);
        assert_eq!(whole, vec![call("read_file", r#"{"path": "a.txt"}"#)]);
        // Every cut in two, and one byte at a time: both markers split.
        for k in 1..ONE.len() {
            assert_eq!(run(&[&ONE[..k], &ONE[k..]]), whole, "cut at byte {k}");
        }
        let bytes: Vec<String> = ONE.chars().map(String::from).collect();
        let refs: Vec<&str> = bytes.iter().map(String::as_str).collect();
        assert_eq!(run(&refs), whole);
    }

    #[test]
    fn a_split_opening_marker_is_never_sent_as_content() {
        let mut p = ToolFormat::Qwen3.parser();
        assert_eq!(p.push("Let me look.<tool_", ""), vec![Piece::Content("Let me look.".into())]);
        assert_eq!(p.push("ca", ""), vec![]);
        let rest = &ONE["<tool_ca".len()..];
        assert_eq!(p.push(rest, ""), vec![call("read_file", r#"{"path": "a.txt"}"#)]);
        assert_eq!(p.finish(), vec![]);
        // A tail that turns out not to be a marker is content after all.
        assert_eq!(run(&["a <tool", "box> b"]), vec![Piece::Content("a <toolbox> b".into())]);
        assert_eq!(run(&["ends with <tool_c"]), vec![Piece::Content("ends with <tool_c".into())]);
    }

    #[test]
    fn two_calls_in_one_answer() {
        let two = format!("{ONE}\n<tool_call>\n{{\"name\": \"list_dir\", \"arguments\": {{\"path\": \"src\"}}}}\n</tool_call>");
        let want = vec![call("read_file", r#"{"path": "a.txt"}"#), call("list_dir", r#"{"path": "src"}"#)];
        assert_eq!(run(&[&two]), want);
        for k in 1..two.len() {
            if two.is_char_boundary(k) {
                assert_eq!(run(&[&two[..k], &two[k..]]), want, "cut at byte {k}");
            }
        }
    }

    #[test]
    fn a_call_after_text_keeps_the_text_and_drops_the_newline_the_template_writes() {
        let a = format!("I will read it.\n\n{ONE}");
        assert_eq!(
            run(&[&a]),
            vec![Piece::Content("I will read it.".into()), call("read_file", r#"{"path": "a.txt"}"#)]
        );
        // Text after a call is content too.
        let b = format!("{ONE}\nDone.");
        assert_eq!(run(&[&b]), vec![call("read_file", r#"{"path": "a.txt"}"#), Piece::Content("\nDone.".into())]);
        // An answer with no call keeps its whitespace.
        assert_eq!(run(&["Hello.\n", "\n"]), vec![Piece::Content("Hello.\n\n".into())]);
    }

    #[test]
    fn malformed_blocks_come_back_as_content_with_the_reason() {
        let bad = "<tool_call>\n{\"name\": \"read_file\", \"arguments\": {\"path\": }\n</tool_call>";
        match &run(&[bad])[..] {
            [Piece::Malformed { text, why }] => {
                assert_eq!(text, bad);
                assert!(why.contains("not JSON"), "{why}");
            }
            o => panic!("{o:?}"),
        }
        let noname = "<tool_call>{\"arguments\": {}}</tool_call>";
        assert!(matches!(&run(&[noname])[..], [Piece::Malformed { why, .. }] if why.contains("name")));
        let unclosed = "<tool_call>\n{\"name\": \"read_file\"";
        assert!(matches!(&run(&[unclosed])[..], [Piece::Malformed { text, why }]
            if text == unclosed && why.contains("</tool_call>")));
        // The routes turn it into content and a warning.
        let mut c = Calls::new(ToolFormat::Qwen3);
        let mut ps = c.delta(bad, "");
        ps.extend(c.finish());
        assert_eq!(ps, vec![Piece::Content(bad.into())]);
        assert_eq!((c.count, c.warnings.len()), (0, 1));
        // Arguments as a string holding an object are that object; absent
        // arguments are a call with none.
        let s = "<tool_call>{\"name\": \"f\", \"arguments\": \"{\\\"x\\\": 1}\"}</tool_call>";
        assert_eq!(run(&[s]), vec![call("f", r#"{"x": 1}"#)]);
        assert_eq!(run(&["<tool_call>{\"name\": \"f\"}</tool_call>"]), vec![call("f", "{}")]);
    }

    #[test]
    fn a_tool_marker_inside_the_reasoning_is_reasoning() {
        let mut p = ToolFormat::Qwen3.parser();
        let think = format!("I could write {ONE} here.");
        let mut out = p.push("", &think);
        out.extend(p.push("Plain answer.", ""));
        out.extend(p.finish());
        assert_eq!(out, vec![Piece::Reasoning(think.clone()), Piece::Content("Plain answer.".into())]);
    }

    #[test]
    fn python_dumps_keeps_order_unicode_and_number_forms() {
        let v: OrderedJson = serde_json::from_str(r#"{"z": 1, "a": [true, null, 0.5, 1.0, 1e16, 1e-5, -3], "é": "a\"b\\c\nd<&>"}"#)
            .unwrap();
        assert_eq!(v.dumps(), r#"{"z": 1, "a": [true, null, 0.5, 1.0, 1e+16, 1e-05, -3], "é": "a\"b\\c\nd<&>"}"#);
        assert_eq!(py_float(123.25), "123.25");
        assert_eq!(py_float(0.0001), "0.0001");
        assert_eq!(py_float(2.5e-7), "2.5e-07");
    }

    // -----------------------------------------------------------------------
    // The rendered prompt against Qwen3's own template.
    // -----------------------------------------------------------------------

    const REFERENCE: &str = include_str!("../tests/fixtures/qwen3-tools-reference.json");

    /// A case's messages, in OpenAI's form as the fixture holds them, as the
    /// OpenAI route reads them (string arguments) and as Ollama's would
    /// (object arguments).
    fn messages(case: &OrderedJson, objects: bool) -> Vec<ToolMsg> {
        let Some(OrderedJson::Arr(ms)) = case.get("messages") else { panic!("messages") };
        ms.iter()
            .map(|m| {
                let role = match m.get("role").and_then(OrderedJson::as_str).unwrap() {
                    "system" => ToolRole::System,
                    "user" => ToolRole::User,
                    "assistant" => ToolRole::Assistant,
                    "tool" => ToolRole::Tool,
                    r => panic!("{r}"),
                };
                let calls = match m.get("tool_calls") {
                    Some(OrderedJson::Arr(cs)) => cs
                        .iter()
                        .map(|c| {
                            let f = c.get("function").unwrap();
                            let text = f.get("arguments").and_then(OrderedJson::as_str).unwrap().to_string();
                            HistCall {
                                id: c.get("id").and_then(OrderedJson::as_str).map(str::to_string),
                                name: f.get("name").and_then(OrderedJson::as_str).unwrap().into(),
                                arguments: match objects {
                                    true => Args::Object(serde_json::from_str(&text).unwrap()),
                                    false => Args::Text(text),
                                },
                            }
                        })
                        .collect(),
                    _ => Vec::new(),
                };
                ToolMsg {
                    role,
                    content: m.get("content").and_then(OrderedJson::as_str).unwrap_or("").into(),
                    calls,
                    name: None,
                }
            })
            .collect()
    }

    fn cases() -> Vec<OrderedJson> {
        let doc: OrderedJson = serde_json::from_str(REFERENCE).unwrap();
        match doc.get("cases") {
            Some(OrderedJson::Arr(c)) => c.clone(),
            _ => panic!("cases"),
        }
    }

    fn tools(case: &OrderedJson) -> Vec<ToolDef> {
        match case.get("tools") {
            Some(OrderedJson::Arr(t)) => t.iter().enumerate().map(|(i, t)| ToolDef::parse(i, t).unwrap()).collect(),
            _ => Vec::new(),
        }
    }

    /// The prompt the engine builds: the pieces, then the opening of an
    /// assistant turn with the empty reasoning block (think off).
    fn prompt(case: &OrderedJson, objects: bool) -> String {
        let mut s = ToolFormat::Qwen3.render(&tools(case), &messages(case, objects)).concat();
        s.push_str("<|im_start|>assistant\n<think>\n\n</think>\n\n");
        s
    }

    #[test]
    fn the_rendered_prompt_is_the_official_templates_byte_for_byte() {
        let cs = cases();
        assert_eq!(cs.len(), 5);
        for c in &cs {
            let name = c.get("name").and_then(OrderedJson::as_str).unwrap();
            let want = c.get("text").and_then(OrderedJson::as_str).unwrap();
            for objects in [false, true] {
                let got = prompt(c, objects);
                if got != want {
                    let at = got.bytes().zip(want.bytes()).position(|(a, b)| a != b).unwrap_or(got.len().min(want.len()));
                    panic!(
                        "{name} (object arguments {objects}): differs at byte {at}:\n ours {:?}\n want {:?}",
                        &got[at.saturating_sub(40)..(at + 40).min(got.len())],
                        &want[at.saturating_sub(40)..(at + 40).min(want.len())]
                    );
                }
            }
        }
    }

    /// The ids too, with the real tokenizer: each piece encoded alone, as the
    /// engine does, gives the ids `transformers` gives for the whole text.
    /// Needs the tokenizer of `Qwen/Qwen3-4B`; fails naming it when absent.
    #[test]
    #[ignore = "needs the Qwen3-4B tokenizer.json of the Hub cache (LLVQ_QWEN3_TOKENIZER)"]
    fn pieces_encoded_alone_give_the_ids_of_the_whole_prompt() {
        let path = std::env::var("LLVQ_QWEN3_TOKENIZER").unwrap_or_else(|_| {
            format!(
                "{}/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c/tokenizer.json",
                std::env::var("HOME").unwrap_or_default()
            )
        });
        let tok = tokenizers::Tokenizer::from_file(&path)
            .unwrap_or_else(|e| panic!("{path}: {e}; set LLVQ_QWEN3_TOKENIZER to Qwen3's tokenizer.json"));
        for c in &cases() {
            let name = c.get("name").and_then(OrderedJson::as_str).unwrap();
            let Some(OrderedJson::Arr(want)) = c.get("ids") else { panic!("ids") };
            let want: Vec<u32> = want.iter().map(|v| match v {
                OrderedJson::Num(n) => n.parse().unwrap(),
                _ => panic!("an id"),
            }).collect();
            let mut pieces = ToolFormat::Qwen3.render(&tools(c), &messages(c, false));
            pieces.push("<|im_start|>assistant\n<think>\n\n</think>\n\n".into());
            let got: Vec<u32> =
                pieces.iter().flat_map(|p| tok.encode(p.as_str(), false).unwrap().get_ids().to_vec()).collect();
            assert_eq!(got, want, "{name}");
        }
    }

    #[test]
    fn a_result_must_answer_a_call_before_it() {
        let m = |role, calls: Vec<HistCall>| ToolMsg { role, content: String::new(), calls, name: None };
        let c = HistCall { id: Some("call_1".into()), name: "f".into(), arguments: Args::Text("{}".into()) };
        let ok = [m(ToolRole::User, vec![]), m(ToolRole::Assistant, vec![c.clone()]), m(ToolRole::Tool, vec![])];
        assert!(check_results(&ok, &[None, None, Some("call_1".into())]).is_ok());
        assert!(check_results(&ok, &[None, None, Some("call_9".into())]).unwrap_err().contains("call_9"));
        let orphan = [m(ToolRole::User, vec![]), m(ToolRole::Tool, vec![])];
        assert!(check_results(&orphan, &[]).unwrap_err().contains("must follow"));
    }
}
