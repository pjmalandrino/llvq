//! Inkling 2 Mini's chat format: the prompt as text and ids, and the model's turn parsed back.
//!
//! The authority is the checkpoint's own `chat_template.jinja` (6,868 bytes, sha256 `53a9035f…`), rendered by
//! `transformers` at 046bdb8cc2. `ops/inkling/chat_ref.py` writes that rendering for a fixed set of conversations and
//! `tests/inkling_chat.rs` holds this module to it, byte for byte in text and id for id in tokens. Line numbers below
//! (`tpl:N`) are lines of that template.
//!
//! What the template does, and so what this module does:
//! - Tools, when given, come first: `<|message_system|>tool_declare<|content_xml|>` then the declarations as one
//!   JSON list, every level sorted, compact (tpl:23-38).
//! - The reasoning effort is a system message, `Thinking effort level: <n>`, written before the first message that is
//!   not a system message, or at the end when there is none (tpl:4-21, 45-48, 135-137). The default is 0.9 (tpl:5).
//!   Named levels map to numbers (tpl:1); a number must lie in [0.0, 0.99] (tpl:15-17).
//! - Every content part is its own message: `<role><content type>body<|end_message|>` (tpl:69-95).
//! - An assistant turn is thinking, then text, then one `<|message_model|>NAME<|content_invoke_tool_json|>` part a
//!   tool call, and it closes with `<|content_model_end_sampling|>` (tpl:71-126), the eos of `generation_config.json`.
//! - There is NO generation prompt: the template never reads `add_generation_prompt`, so the prompt ends on the last
//!   message's `<|end_message|>` and the model writes its own `<|message_model|>`. This module takes no such flag.
//!
//! Inputs are OpenAI-shaped JSON values, read with the template's semantics (Jinja truthiness, `is defined`, keys
//! whose value is null dropped from a message as `apply_chat_template` does, `reasoning` and `thinking` read as
//! `reasoning_content`). They are held in [`Json`], an ordered value that keeps integers and floats apart, because the
//! template prints tool arguments with Python's `json.dumps` and nested objects in their given order.
//!
//! What is refused, by name:
//! - everything the template refuses, with the template's own message (unknown effort or role, arguments given as
//!   a JSON string or a non-object, a non-string tool name, an assistant name that is not its tool's, images outside
//!   user and tool messages, audio, unknown content parts);
//! - images anywhere: the template renders them for user and tool messages (tpl:84-88), but serving one needs the
//!   vision tower, which this engine does not have;
//! - inputs the template reads in ways nobody means (an effort that is a list, content that is an object, a tool
//!   without a name, a NaN effort);
//! - a control token inside a dictionary key, and the sanitizer's private-use markers U+F0000 and U+F0001 anywhere
//!   in the input (see [`InkChat::render`] on control tokens).

use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::Path;

use anyhow::{anyhow, bail, Context};

/// The template's markers.
pub mod mark {
    pub const USER: &str = "<|message_user|>";
    pub const MODEL: &str = "<|message_model|>";
    pub const SYSTEM: &str = "<|message_system|>";
    pub const TOOL: &str = "<|message_tool|>";
    pub const TEXT: &str = "<|content_text|>";
    pub const THINKING: &str = "<|content_thinking|>";
    pub const INVOKE_JSON: &str = "<|content_invoke_tool_json|>";
    pub const INVOKE_TEXT: &str = "<|content_invoke_tool_text|>";
    pub const XML: &str = "<|content_xml|>";
    pub const IMAGE: &str = "<|content_image|>";
    pub const TOOL_ERROR: &str = "<|content_tool_error|>";
    pub const END_MESSAGE: &str = "<|end_message|>";
    pub const END_SAMPLING: &str = "<|content_model_end_sampling|>";
    pub const END_OF_TEXT: &str = "<|endoftext|>";
}

/// The private-use characters `transformers` wraps around a control token found in user input
/// (`chat_template_utils.py` 55-57).
const SANITIZE_START: char = '\u{F0000}';
const SANITIZE_END: char = '\u{F0001}';

/// The named efforts, tpl:1.
const EFFORT_MAP: [(&str, f64); 7] = [
    ("none", 0.0),
    ("minimal", 0.1),
    ("low", 0.2),
    ("medium", 0.7),
    ("high", 0.9),
    ("xhigh", 0.99),
    ("max", 0.99),
];

/// The effort when none is given, tpl:5.
pub const DEFAULT_EFFORT: f64 = 0.9;

// ---------------------------------------------------------------------------------------------------------------
// JSON with Python's semantics
// ---------------------------------------------------------------------------------------------------------------

/// A JSON value as Python's `json` module holds it: objects keep their order, integers stay apart from floats and
/// keep every digit.
#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    /// The decimal digits of an integer, `-` included, never `-0`.
    Int(String),
    Float(f64),
    Str(String),
    Array(Vec<Json>),
    Object(Vec<(String, Json)>),
}

impl Json {
    pub fn str(s: &str) -> Json {
        Json::Str(s.to_string())
    }

    /// Parse JSON text. Duplicate keys, lone surrogates, NaN and infinities are refused: Python would accept the
    /// first by keeping the last value at the first place, and Rust strings cannot hold the second.
    pub fn parse(text: &str) -> anyhow::Result<Json> {
        let mut p = JsonParser { s: text.as_bytes(), i: 0 };
        p.ws();
        let v = p.value(0)?;
        p.ws();
        if p.i != p.s.len() {
            bail!("json: trailing characters at byte {}", p.i);
        }
        Ok(v)
    }

    /// Python's truthiness.
    pub fn truthy(&self) -> bool {
        match self {
            Json::Null => false,
            Json::Bool(b) => *b,
            Json::Int(s) => s != "0",
            Json::Float(f) => *f != 0.0,
            Json::Str(s) => !s.is_empty(),
            Json::Array(a) => !a.is_empty(),
            Json::Object(o) => !o.is_empty(),
        }
    }

    /// The value of `key` in an object; `None` for a missing key or a value that is not an object.
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Object(o) => o.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }

    /// `json.dumps(self, ensure_ascii=False, separators=…, sort_keys=…)`, the filter `tojson` of
    /// `transformers` (`chat_template_utils.py` 623-626). `compact` is `separators=(",", ":")`; otherwise Python's
    /// default `", "` and `": "`.
    pub fn dumps(&self, sort_keys: bool, compact: bool) -> String {
        let mut out = String::new();
        self.dump_into(&mut out, sort_keys, compact);
        out
    }

    fn dump_into(&self, out: &mut String, sort_keys: bool, compact: bool) {
        let (item, key) = if compact { (",", ":") } else { (", ", ": ") };
        match self {
            Json::Null => out.push_str("null"),
            Json::Bool(true) => out.push_str("true"),
            Json::Bool(false) => out.push_str("false"),
            Json::Int(s) => out.push_str(s),
            Json::Float(f) => out.push_str(&py_float_repr(*f)),
            Json::Str(s) => py_json_string(s, out),
            Json::Array(a) => {
                out.push('[');
                for (n, v) in a.iter().enumerate() {
                    if n > 0 {
                        out.push_str(item);
                    }
                    v.dump_into(out, sort_keys, compact);
                }
                out.push(']');
            }
            Json::Object(o) => {
                let mut items: Vec<&(String, Json)> = o.iter().collect();
                if sort_keys {
                    items.sort_by(|a, b| a.0.cmp(&b.0));
                }
                out.push('{');
                for (n, (k, v)) in items.into_iter().enumerate() {
                    if n > 0 {
                        out.push_str(item);
                    }
                    py_json_string(k, out);
                    out.push_str(key);
                    v.dump_into(out, sort_keys, compact);
                }
                out.push('}');
            }
        }
    }

    /// Equality as Python's `==` on dicts: object keys in any order.
    pub fn same(&self, other: &Json) -> bool {
        match (self, other) {
            (Json::Object(a), Json::Object(b)) => {
                a.len() == b.len() && a.iter().all(|(k, v)| other.get(k).is_some_and(|w| v.same(w)))
                    && b.iter().all(|(k, _)| self.get(k).is_some())
            }
            (Json::Array(a), Json::Array(b)) => a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.same(y)),
            (a, b) => a == b,
        }
    }
}

/// Python's `json` string escape with `ensure_ascii=False`: `"`, `\`, and the C0 controls, the five usual ones by
/// letter and the rest as `\u00xx` in lowercase hex. Nothing else, not `/`, not DEL, not non-ASCII.
fn py_json_string(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Python's `repr(float)`: the shortest digits that round-trip, in fixed notation when the decimal exponent lies in
/// [-4, 16), otherwise as `d.ddde±XX` (`PyOS_double_to_string`, mode `'r'`). Rust's `{:e}` gives the same shortest
/// digits. Finite values only.
pub fn py_float_repr(x: f64) -> String {
    debug_assert!(x.is_finite());
    if x == 0.0 {
        return if x.is_sign_negative() { "-0.0" } else { "0.0" }.to_string();
    }
    let sci = format!("{:e}", x.abs());
    let (mant, exp) = sci.split_once('e').expect("LowerExp writes an exponent");
    let exp: i32 = exp.parse().expect("LowerExp writes an integer exponent");
    let digits: String = mant.chars().filter(|c| *c != '.').collect();
    let sign = if x < 0.0 { "-" } else { "" };
    let decpt = exp + 1; // x = 0.digits * 10^decpt
    let n = digits.len() as i32;
    if decpt <= -4 || decpt > 16 {
        let mut m = digits[..1].to_string();
        if n > 1 {
            m.push('.');
            m.push_str(&digits[1..]);
        }
        format!("{sign}{m}e{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs())
    } else if decpt <= 0 {
        format!("{sign}0.{}{digits}", "0".repeat((-decpt) as usize))
    } else if decpt >= n {
        format!("{sign}{digits}{}.0", "0".repeat((decpt - n) as usize))
    } else {
        format!("{sign}{}.{}", &digits[..decpt as usize], &digits[decpt as usize..])
    }
}

/// Python's `str.strip()` set, which Jinja's `trim` uses: Rust's whitespace plus U+001C to U+001F.
fn py_strip(s: &str) -> &str {
    s.trim_matches(|c: char| c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c))
}

struct JsonParser<'a> {
    s: &'a [u8],
    i: usize,
}

impl JsonParser<'_> {
    fn ws(&mut self) {
        while self.i < self.s.len() && matches!(self.s[self.i], b' ' | b'\t' | b'\n' | b'\r') {
            self.i += 1;
        }
    }

    fn err<T>(&self, what: &str) -> anyhow::Result<T> {
        Err(anyhow!("json: {what} at byte {}", self.i))
    }

    fn value(&mut self, depth: usize) -> anyhow::Result<Json> {
        if depth > 256 {
            return self.err("nesting deeper than 256");
        }
        match self.s.get(self.i) {
            None => self.err("unexpected end"),
            Some(b'{') => {
                self.i += 1;
                let mut o: Vec<(String, Json)> = Vec::new();
                self.ws();
                if self.s.get(self.i) == Some(&b'}') {
                    self.i += 1;
                    return Ok(Json::Object(o));
                }
                loop {
                    self.ws();
                    if self.s.get(self.i) != Some(&b'"') {
                        return self.err("expected a key");
                    }
                    let k = self.string()?;
                    if o.iter().any(|(x, _)| *x == k) {
                        return self.err(&format!("duplicate key {k:?}"));
                    }
                    self.ws();
                    if self.s.get(self.i) != Some(&b':') {
                        return self.err("expected ':'");
                    }
                    self.i += 1;
                    self.ws();
                    let v = self.value(depth + 1)?;
                    o.push((k, v));
                    self.ws();
                    match self.s.get(self.i) {
                        Some(b',') => self.i += 1,
                        Some(b'}') => {
                            self.i += 1;
                            return Ok(Json::Object(o));
                        }
                        _ => return self.err("expected ',' or '}'"),
                    }
                }
            }
            Some(b'[') => {
                self.i += 1;
                let mut a = Vec::new();
                self.ws();
                if self.s.get(self.i) == Some(&b']') {
                    self.i += 1;
                    return Ok(Json::Array(a));
                }
                loop {
                    self.ws();
                    a.push(self.value(depth + 1)?);
                    self.ws();
                    match self.s.get(self.i) {
                        Some(b',') => self.i += 1,
                        Some(b']') => {
                            self.i += 1;
                            return Ok(Json::Array(a));
                        }
                        _ => return self.err("expected ',' or ']'"),
                    }
                }
            }
            Some(b'"') => Ok(Json::Str(self.string()?)),
            Some(b't') => self.word("true", Json::Bool(true)),
            Some(b'f') => self.word("false", Json::Bool(false)),
            Some(b'n') => self.word("null", Json::Null),
            Some(b'-' | b'0'..=b'9') => self.number(),
            Some(_) => self.err("unexpected character (NaN and Infinity are refused)"),
        }
    }

    fn word(&mut self, w: &str, v: Json) -> anyhow::Result<Json> {
        if self.s[self.i..].starts_with(w.as_bytes()) {
            self.i += w.len();
            Ok(v)
        } else {
            self.err("unknown literal")
        }
    }

    fn number(&mut self) -> anyhow::Result<Json> {
        let start = self.i;
        let digits = |p: &mut Self| {
            let s = p.i;
            while p.i < p.s.len() && p.s[p.i].is_ascii_digit() {
                p.i += 1;
            }
            p.i - s
        };
        if self.s[self.i] == b'-' {
            self.i += 1;
        }
        let int_start = self.i;
        let n = digits(self);
        if n == 0 || (n > 1 && self.s[int_start] == b'0') {
            return self.err("malformed number");
        }
        let mut float = false;
        if self.s.get(self.i) == Some(&b'.') {
            self.i += 1;
            if digits(self) == 0 {
                return self.err("malformed fraction");
            }
            float = true;
        }
        if matches!(self.s.get(self.i), Some(b'e' | b'E')) {
            self.i += 1;
            if matches!(self.s.get(self.i), Some(b'+' | b'-')) {
                self.i += 1;
            }
            if digits(self) == 0 {
                return self.err("malformed exponent");
            }
            float = true;
        }
        let text = std::str::from_utf8(&self.s[start..self.i]).expect("ASCII");
        if float {
            let f: f64 = text.parse().map_err(|e| anyhow!("json: {e} in {text}"))?;
            if !f.is_finite() {
                return self.err("a number beyond the range of a float");
            }
            Ok(Json::Float(f))
        } else if text.trim_start_matches('-').bytes().all(|b| b == b'0') {
            Ok(Json::Int("0".to_string()))
        } else {
            Ok(Json::Int(text.to_string()))
        }
    }

    fn hex4(&mut self) -> anyhow::Result<u32> {
        let h = self.s.get(self.i..self.i + 4).ok_or_else(|| anyhow!("json: short \\u escape"))?;
        let h = std::str::from_utf8(h).map_err(|_| anyhow!("json: bad \\u escape"))?;
        let v = u32::from_str_radix(h, 16).map_err(|_| anyhow!("json: bad \\u escape {h:?}"))?;
        self.i += 4;
        Ok(v)
    }

    fn string(&mut self) -> anyhow::Result<String> {
        self.i += 1; // the opening quote
        let mut out = String::new();
        loop {
            let start = self.i;
            while self.i < self.s.len() && !matches!(self.s[self.i], b'"' | b'\\') && self.s[self.i] >= 0x20 {
                self.i += 1;
            }
            out.push_str(std::str::from_utf8(&self.s[start..self.i]).map_err(|e| anyhow!("json: {e}"))?);
            match self.s.get(self.i) {
                None => return self.err("unterminated string"),
                Some(b'"') => {
                    self.i += 1;
                    return Ok(out);
                }
                Some(b'\\') => {
                    self.i += 1;
                    let c = *self.s.get(self.i).ok_or_else(|| anyhow!("json: unterminated escape"))?;
                    self.i += 1;
                    match c {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let hi = self.hex4()?;
                            let cp = if (0xD800..0xDC00).contains(&hi) {
                                if !self.s[self.i..].starts_with(b"\\u") {
                                    return self.err("lone surrogate");
                                }
                                self.i += 2;
                                let lo = self.hex4()?;
                                if !(0xDC00..0xE000).contains(&lo) {
                                    return self.err("lone surrogate");
                                }
                                0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00)
                            } else {
                                hi
                            };
                            out.push(char::from_u32(cp).ok_or_else(|| anyhow!("json: lone surrogate"))?);
                        }
                        _ => return self.err("unknown escape"),
                    }
                }
                Some(_) => return self.err("control character in a string"),
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------------------------------------------

/// The tokenizer, the markers' ids and the control tokens of the checkpoint.
pub struct InkChat {
    tok: tokenizers::Tokenizer,
    /// The same tokenizer with added-token matching off: `split_special_tokens=True` in `transformers`.
    plain: tokenizers::Tokenizer,
    /// Every special added token, longest first: the control tokens `transformers` sanitizes
    /// (`tokenization_utils_base.py` 3258-3273; every added token of this checkpoint is special).
    controls: Vec<(String, u32)>,
    specials: HashMap<u32, String>,
    /// `tokenizer.encode("\n")`.
    newline: Vec<u32>,
    /// The id of `<|content_model_end_sampling|>`.
    pub end_sampling: u32,
}

/// A rendered conversation.
#[derive(Debug, Clone, PartialEq)]
pub struct Rendered {
    pub text: String,
    pub ids: Vec<u32>,
    /// Each assistant message's span in `ids`, the template's `{% generation %}` block (tpl:128-129): what the model
    /// wrote, from its first marker to its `<|content_model_end_sampling|>`.
    pub assistant_spans: Vec<std::ops::Range<usize>>,
}

/// One piece of the rendered text and where it came from.
enum Seg {
    /// A marker the template writes.
    Ctl(&'static str),
    /// Text the template writes.
    Lit(String),
    /// Text from the input: content, names, arguments, declarations.
    User(String),
}

impl InkChat {
    /// Load `tokenizer.json` from a checkpoint directory.
    pub fn load(dir: &Path) -> anyhow::Result<Self> {
        let path = dir.join("tokenizer.json");
        let tok = tokenizers::Tokenizer::from_file(&path).map_err(|e| anyhow!("{}: {e}", path.display()))?;
        Self::new(tok)
    }

    pub fn new(tok: tokenizers::Tokenizer) -> anyhow::Result<Self> {
        let mut controls: Vec<(String, u32)> = tok
            .get_added_tokens_decoder()
            .iter()
            .filter(|(_, t)| t.special)
            .map(|(id, t)| (t.content.clone(), *id))
            .collect();
        // Longest first, then by id: `sorted(control_tokens, key=len, reverse=True)` over tokens in id order.
        controls.sort_by(|a, b| b.0.chars().count().cmp(&a.0.chars().count()).then(a.1.cmp(&b.1)));
        let specials: HashMap<u32, String> = controls.iter().map(|(s, id)| (*id, s.clone())).collect();
        for m in [
            mark::USER,
            mark::MODEL,
            mark::SYSTEM,
            mark::TOOL,
            mark::TEXT,
            mark::THINKING,
            mark::INVOKE_JSON,
            mark::XML,
            mark::END_MESSAGE,
            mark::END_SAMPLING,
        ] {
            if !controls.iter().any(|(s, _)| s == m) {
                bail!("the tokenizer has no special token {m}; is this Inkling 2's tokenizer.json?");
            }
        }
        let end_sampling = tok.token_to_id(mark::END_SAMPLING).expect("checked above");
        let mut plain = tok.clone();
        plain.set_encode_special_tokens(true);
        let newline = tok.encode("\n", false).map_err(|e| anyhow!("{e}"))?.get_ids().to_vec();
        Ok(InkChat { tok, plain, controls, specials, newline, end_sampling })
    }

    pub fn tokenizer(&self) -> &tokenizers::Tokenizer {
        &self.tok
    }

    /// The ids that end generation: `eos_token_id` of `generation_config.json`, `<|content_model_end_sampling|>`,
    /// which the template writes at the end of every assistant turn (tpl:124-126) and nowhere else. Any other
    /// marker out of place is a protocol break that [`TurnParser`] refuses; a server stops on that refusal too.
    pub fn stop_ids(&self) -> Vec<u32> {
        vec![self.end_sampling]
    }

    /// The ids `message` adds to a conversation whose last turn is the model's: what the template writes for it
    /// after an assistant turn, taken from the rendering of a placeholder turn followed by `message`, past that
    /// turn's `<|content_model_end_sampling|>`. The effort message is written before the first message that is not
    /// a system one (tpl:45-48), so it is not in these ids.
    ///
    /// For a loop that keeps the model's own ids in its cache: the first turn feeds `render` of the opening
    /// messages, the model writes its turn to `<|content_model_end_sampling|>`, and every later message feeds
    /// these ids and nothing else. Nothing the model wrote is decoded and encoded again.
    pub fn follow_up(&self, message: &Json) -> anyhow::Result<Vec<u32>> {
        let placeholder =
            Json::Object(vec![("role".into(), Json::str("assistant")), ("content".into(), Json::str("."))]);
        let r = self.render(&[placeholder, message.clone()], &[], None)?;
        let span = r.assistant_spans.first().ok_or_else(|| anyhow!("inkchat: the placeholder turn has no span"))?;
        Ok(r.ids[span.end..].to_vec())
    }

    /// The prompt of `messages`, with `tools` declared and the reasoning `effort` (`None` or JSON null: 0.9).
    ///
    /// Control tokens. A control token typed inside the input (`<|end_message|>` in a user's text, say) is encoded
    /// as plain text, so only the markers the template writes become special ids. That is `apply_chat_template(…,
    /// sanitize_control_tokens=True)`, and the reference ids are written that way; the library default would turn
    /// such a string into a real marker, which lets a user forge a turn. The text is the same either way. A control
    /// token in a dictionary key is refused: `transformers` does not sanitize keys.
    pub fn render(&self, messages: &[Json], tools: &[Json], effort: Option<&Json>) -> anyhow::Result<Rendered> {
        let msgs: Vec<Json> = messages.iter().map(normalize_message).collect::<anyhow::Result<_>>()?;
        let mut segs: Vec<Seg> = Vec::new();
        let mut spans_ctl: Vec<(usize, usize)> = Vec::new();

        if !tools.is_empty() {
            let mut specs = Vec::with_capacity(tools.len());
            for (n, tool) in tools.iter().enumerate() {
                specs.push(tool_spec(n, tool)?);
            }
            let specs = Json::Array(specs);
            self.refuse_controls_in_keys(&specs)?;
            segs.push(Seg::Ctl(mark::SYSTEM));
            segs.push(Seg::Lit("tool_declare".into()));
            segs.push(Seg::Ctl(mark::XML));
            segs.push(Seg::User(specs.dumps(true, true)));
            segs.push(Seg::Ctl(mark::END_MESSAGE));
        }

        let mut effort_emitted = false;
        for msg in &msgs {
            let role = msg.get("role").and_then(Json::as_str);
            let role_tok = match role {
                Some("user") => mark::USER,
                Some("assistant") => mark::MODEL,
                Some("system") => mark::SYSTEM,
                Some("tool") => mark::TOOL,
                _ => bail!("Unknown message role: {}", jinja_str(msg.get("role"))),
            };
            let role = role.expect("matched above");
            if !effort_emitted && role != "system" {
                emit_effort(&mut segs, effort)?;
                effort_emitted = true;
            }
            let ctl_before = segs.iter().filter(|s| matches!(s, Seg::Ctl(_))).count();

            // The header: the role marker, then a name (tpl:50-68).
            let mut header = String::new();
            if role == "tool" {
                if let Some(name) = msg.get("name").filter(|v| v.truthy()) {
                    header.push_str(name_str(name, "tool message name")?);
                } else if let Some(id) = msg.get("tool_call_id").filter(|v| v.truthy()) {
                    for prev in &msgs {
                        if prev.get("role").and_then(Json::as_str) != Some("assistant") {
                            continue;
                        }
                        let Some(calls) = prev.get("tool_calls").filter(|v| v.truthy()) else { continue };
                        for tc in list(calls, "tool_calls")? {
                            let matches = tc.get("id").is_some_and(|x| x == id);
                            if let (true, Some(n)) = (matches, tc.get("function").and_then(|f| f.get("name"))) {
                                header.push_str(name_str(n, "tool call function name")?);
                            }
                        }
                    }
                }
            } else if role == "assistant" {
                if let Some(name) = msg.get("name").filter(|v| v.truthy()) {
                    header.push_str(name_str(name, "assistant name")?);
                }
            }
            let open = |segs: &mut Vec<Seg>, content: &'static str| {
                segs.push(Seg::Ctl(role_tok));
                if !header.is_empty() {
                    segs.push(Seg::User(header.clone()));
                }
                segs.push(Seg::Ctl(content));
            };

            let calls = msg.get("tool_calls").filter(|v| v.truthy());
            let reasoning = msg.get("reasoning_content").filter(|v| v.truthy());
            if role == "assistant" {
                if let Some(r) = reasoning {
                    open(&mut segs, mark::THINKING);
                    segs.push(Seg::User(jinja_concat(r, "reasoning_content")?));
                    segs.push(Seg::Ctl(mark::END_MESSAGE));
                }
            }

            match msg.get("content") {
                Some(Json::Str(c)) if !c.is_empty() || !(role == "assistant" && (calls.is_some() || reasoning.is_some())) => {
                    open(&mut segs, mark::TEXT);
                    segs.push(Seg::User(c.clone()));
                    segs.push(Seg::Ctl(mark::END_MESSAGE));
                }
                Some(Json::Str(_)) | None => {}
                Some(v) if !v.truthy() => {}
                Some(Json::Array(parts)) => {
                    for part in parts {
                        let ty = part.get("type");
                        match ty.and_then(Json::as_str) {
                            _ if matches!(part, Json::Str(_)) => {
                                open(&mut segs, mark::TEXT);
                                segs.push(Seg::User(part.as_str().expect("matched").to_string()));
                                segs.push(Seg::Ctl(mark::END_MESSAGE));
                            }
                            None if ty.is_none() => {
                                self.text_part(&mut segs, part, &open);
                            }
                            Some("text" | "input_text") => {
                                self.text_part(&mut segs, part, &open);
                            }
                            Some("image" | "input_image" | "image_url") => {
                                if role != "user" && role != "tool" {
                                    bail!("Image content is only supported for user and tool messages");
                                }
                                bail!(
                                    "inkchat: images are refused: the template would write {}<|unused_200054|> \
                                     (tpl:88), and this engine has no vision tower to read it",
                                    mark::IMAGE
                                );
                            }
                            Some("audio" | "input_audio" | "audio_url") => {
                                bail!("Inkling 2 does not accept audio input")
                            }
                            _ => bail!("Unsupported content part type: {}", jinja_str(ty)),
                        }
                    }
                }
                Some(_) => bail!("inkchat: message content must be a string, a list of parts or null"),
            }

            if role == "assistant" {
                if let Some(calls) = calls {
                    for tc in list(calls, "tool_calls")? {
                        let f = tc.get("function");
                        let Some(name) = f.and_then(|f| f.get("name")).and_then(Json::as_str) else {
                            bail!("tool call function name must be a string");
                        };
                        if let Some(own) = msg.get("name").filter(|v| v.truthy()) {
                            if own.as_str() != Some(name) {
                                bail!("assistant name {} does not match its tool call {name}", jinja_str(Some(own)));
                            }
                        }
                        let args = match f.and_then(|f| f.get("arguments")).filter(|v| v.truthy()) {
                            None => Json::Object(Vec::new()),
                            Some(Json::Str(_)) => bail!(
                                "tool call arguments must be a parsed object, not a JSON string; canonicalize upstream"
                            ),
                            Some(Json::Object(o)) => Json::Object(o.clone()),
                            Some(_) => bail!("tool call arguments must be an object"),
                        };
                        self.refuse_controls_in_keys(&args)?;
                        segs.push(Seg::Ctl(mark::MODEL));
                        segs.push(Seg::User(name.to_string()));
                        segs.push(Seg::Ctl(mark::INVOKE_JSON));
                        segs.push(Seg::Lit("{\"name\":".into()));
                        segs.push(Seg::User(Json::str(name).dumps(true, true)));
                        segs.push(Seg::Lit(",\"args\":{".into()));
                        let Json::Object(mut items) = args else { unreachable!() };
                        // `dictsort(case_sensitive=true)` (tpl:116): by key, code point order.
                        items.sort_by(|a, b| a.0.cmp(&b.0));
                        for (n, (k, v)) in items.iter().enumerate() {
                            if n > 0 {
                                segs.push(Seg::Lit(",".into()));
                            }
                            segs.push(Seg::User(Json::str(k).dumps(false, false)));
                            segs.push(Seg::Lit(":".into()));
                            segs.push(Seg::User(v.dumps(false, true)));
                        }
                        segs.push(Seg::Lit("}}".into()));
                        segs.push(Seg::Ctl(mark::END_MESSAGE));
                    }
                }
                segs.push(Seg::Ctl(mark::END_SAMPLING));
                let ctl_after = segs.iter().filter(|s| matches!(s, Seg::Ctl(_))).count();
                spans_ctl.push((ctl_before, ctl_after));
            }
        }
        if !effort_emitted {
            emit_effort(&mut segs, effort)?;
        }
        self.encode_segs(&segs, &spans_ctl)
    }

    fn text_part(&self, segs: &mut Vec<Seg>, part: &Json, open: &dyn Fn(&mut Vec<Seg>, &'static str)) {
        let text = part.get("text").and_then(Json::as_str).unwrap_or("");
        open(segs, mark::TEXT);
        segs.push(Seg::User(text.to_string()));
        segs.push(Seg::Ctl(mark::END_MESSAGE));
    }

    fn refuse_controls_in_keys(&self, v: &Json) -> anyhow::Result<()> {
        match v {
            Json::Object(o) => {
                for (k, x) in o {
                    if let Some((c, _)) = self.controls.iter().find(|(c, _)| k.contains(c.as_str())) {
                        bail!("inkchat: the key {k:?} holds the control token {c}, which transformers does not sanitize");
                    }
                    self.refuse_controls_in_keys(x)?;
                }
                Ok(())
            }
            Json::Array(a) => a.iter().try_for_each(|x| self.refuse_controls_in_keys(x)),
            _ => Ok(()),
        }
    }

    /// The control token that starts at byte `at` of `s`, longest first.
    fn control_at(&self, s: &str, at: usize) -> Option<&str> {
        self.controls.iter().map(|(c, _)| c.as_str()).find(|c| s[at..].starts_with(c))
    }

    /// Text and ids of the segments, ids as `encode_marked_chat` builds them (`chat_template_utils.py` 499-541): the
    /// text between control tokens found in user strings is encoded normally, each such token as plain text.
    fn encode_segs(&self, segs: &[Seg], spans_ctl: &[(usize, usize)]) -> anyhow::Result<Rendered> {
        let mut text = String::new();
        // Pieces: (text, as_text). The first is always a normal piece.
        let mut pieces: Vec<(String, bool)> = vec![(String::new(), false)];
        for seg in segs {
            match seg {
                Seg::Ctl(s) => {
                    text.push_str(s);
                    pieces.last_mut().expect("never empty").0.push_str(s);
                }
                Seg::Lit(s) => {
                    text.push_str(s);
                    pieces.last_mut().expect("never empty").0.push_str(s);
                }
                Seg::User(s) => {
                    if s.contains([SANITIZE_START, SANITIZE_END]) {
                        bail!("inkchat: the input holds U+F0000 or U+F0001, the markers transformers' sanitizer reserves");
                    }
                    text.push_str(s);
                    let mut from = 0;
                    let mut at = 0;
                    while at < s.len() {
                        if let Some(c) = self.control_at(s, at) {
                            pieces.last_mut().expect("never empty").0.push_str(&s[from..at]);
                            pieces.push((c.to_string(), true));
                            pieces.push((String::new(), false));
                            at += c.len();
                            from = at;
                        } else {
                            at += s[at..].chars().next().expect("not at the end").len_utf8();
                        }
                    }
                    pieces.last_mut().expect("never empty").0.push_str(&s[from..]);
                }
            }
        }
        let enc = |t: &tokenizers::Tokenizer, s: &str| -> anyhow::Result<Vec<u32>> {
            Ok(t.encode(s, false).map_err(|e| anyhow!("{e}"))?.get_ids().to_vec())
        };
        let mut ids = enc(&self.tok, &pieces[0].0)?;
        for (piece, as_text) in &pieces[1..] {
            if piece.is_empty() {
                continue;
            }
            let t = if *as_text { &self.plain } else { &self.tok };
            // Encoded after a newline that is then removed (`_encode_chat_piece`, 528-541).
            let with_nl = enc(t, &format!("\n{piece}"))?;
            let piece_ids = match with_nl.strip_prefix(self.newline.as_slice()) {
                Some(rest) => rest.to_vec(),
                None => enc(t, piece)?,
            };
            if *as_text {
                let own = self.tok.token_to_id(piece).expect("a control token is in the vocabulary");
                if piece_ids.contains(&own) {
                    bail!("inkchat: can't sanitize {piece:?}: the tokenizer produces it from plain text");
                }
            }
            ids.extend(piece_ids);
        }

        // Every special id is one of the markers the template wrote, in order.
        let ctl: Vec<&str> = segs.iter().filter_map(|s| if let Seg::Ctl(c) = s { Some(*c) } else { None }).collect();
        let at: Vec<usize> = ids.iter().enumerate().filter(|(_, id)| self.specials.contains_key(id)).map(|(i, _)| i).collect();
        let got: Vec<&str> = at.iter().map(|i| self.specials[&ids[*i]].as_str()).collect();
        if got != ctl {
            bail!("inkchat: the ids hold {} special tokens where the template wrote {} markers", got.len(), ctl.len());
        }
        let assistant_spans = spans_ctl.iter().map(|&(a, b)| at[a]..at[b - 1] + 1).collect();
        Ok(Rendered { text, ids, assistant_spans })
    }
}

/// What `apply_chat_template` does to a message before the template sees it: keys whose value is null dropped
/// (`tokenization_utils_base.py` 3160-3161), and `reasoning_content` filled from the first of `reasoning_content`,
/// `thinking`, `reasoning` that is present (`chat_template_utils.py` 673-686).
fn normalize_message(m: &Json) -> anyhow::Result<Json> {
    let Json::Object(o) = m else { bail!("inkchat: a chat message must be a JSON object") };
    let mut o: Vec<(String, Json)> = o.iter().filter(|(_, v)| *v != Json::Null).cloned().collect();
    let reasoning = ["reasoning_content", "thinking", "reasoning"]
        .iter()
        .find_map(|k| o.iter().find(|(x, _)| x == k).map(|(_, v)| v.clone()));
    if let Some(r) = reasoning {
        o.retain(|(k, _)| k != "reasoning_content");
        o.push(("reasoning_content".into(), r));
    }
    Ok(Json::Object(o))
}

/// One declaration, tpl:26-32.
fn tool_spec(n: usize, tool: &Json) -> anyhow::Result<Json> {
    if !matches!(tool, Json::Object(_)) {
        bail!("inkchat: tool {n} is not a JSON object");
    }
    let f = tool.get("function").unwrap_or(tool);
    if !matches!(f, Json::Object(_)) {
        bail!("inkchat: tool {n} has a `function` that is not an object");
    }
    let Some(name) = f.get("name") else { bail!("inkchat: tool {n} has no function name") };
    let pick = |v: Option<&Json>, d: Json| v.filter(|x| x.truthy()).cloned().unwrap_or(d);
    Ok(Json::Object(vec![
        ("description".into(), pick(f.get("description"), Json::str(""))),
        ("name".into(), name.clone()),
        ("parameters".into(), pick(f.get("parameters"), Json::Object(Vec::new()))),
        ("type".into(), pick(tool.get("type"), Json::str("function"))),
    ]))
}

/// `Thinking effort level: <n>` as a system message, tpl:4-21.
fn emit_effort(segs: &mut Vec<Seg>, effort: Option<&Json>) -> anyhow::Result<()> {
    let num = match effort {
        None | Some(Json::Null) => DEFAULT_EFFORT,
        Some(Json::Str(s)) => match EFFORT_MAP.iter().find(|(k, _)| *k == py_strip(s)) {
            Some((_, v)) => *v,
            None => bail!("Unknown reasoning_effort: {s}"),
        },
        Some(Json::Float(f)) => *f,
        Some(Json::Int(s)) => s.parse::<f64>().map_err(|e| anyhow!("inkchat: reasoning_effort {s}: {e}"))?,
        // Jinja's `float` filter: float(True) = 1.0.
        Some(Json::Bool(b)) => f64::from(u8::from(*b)),
        Some(_) => bail!("inkchat: reasoning_effort must be a name or a number; Jinja's float filter would read 0"),
    };
    if num.is_nan() {
        bail!("inkchat: reasoning_effort is NaN, which the template would print as nan");
    }
    if !(0.0..=0.99).contains(&num) {
        bail!("reasoning_effort must be in [0.0, 0.99]");
    }
    segs.push(Seg::Ctl(mark::SYSTEM));
    segs.push(Seg::Ctl(mark::TEXT));
    segs.push(Seg::Lit("Thinking effort level: ".into()));
    segs.push(Seg::Lit(if num == 0.0 { "0".to_string() } else { py_float_repr(num) }));
    segs.push(Seg::Ctl(mark::END_MESSAGE));
    Ok(())
}

/// The effort as the template prints it, for a caller that wants to show it.
pub fn effort_text(effort: Option<&Json>) -> anyhow::Result<String> {
    let mut segs = Vec::new();
    emit_effort(&mut segs, effort)?;
    match &segs[3] {
        Seg::Lit(s) => Ok(s.clone()),
        _ => unreachable!(),
    }
}

fn list<'a>(v: &'a Json, what: &str) -> anyhow::Result<&'a [Json]> {
    match v {
        Json::Array(a) => Ok(a),
        _ => bail!("inkchat: {what} must be a list"),
    }
}

fn name_str<'a>(v: &'a Json, what: &str) -> anyhow::Result<&'a str> {
    v.as_str().ok_or_else(|| anyhow!("inkchat: {what} must be a string"))
}

/// Jinja's `~` on a value the template concatenates: strings only here.
fn jinja_concat(v: &Json, what: &str) -> anyhow::Result<String> {
    Ok(name_str(v, what)?.to_string())
}

/// A value as Jinja prints it in an error message, for the values those messages meet.
fn jinja_str(v: Option<&Json>) -> String {
    match v {
        None => String::new(),
        Some(Json::Str(s)) => s.clone(),
        Some(Json::Null) => "None".into(),
        Some(Json::Bool(true)) => "True".into(),
        Some(Json::Bool(false)) => "False".into(),
        Some(other) => other.dumps(false, false),
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Parsing the model's turn
// ---------------------------------------------------------------------------------------------------------------

/// A tool call as the model writes it: `<|message_model|>NAME<|content_invoke_tool_json|>{"name":NAME,"args":{…}}
/// <|end_message|>`, the form of tpl:113-120.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    pub name: String,
    /// Always an object.
    pub arguments: Json,
}

/// One part of a turn, in the order the model wrote it.
#[derive(Debug, Clone, PartialEq)]
pub enum Part {
    /// `header` is the name after `<|message_model|>`, empty unless the turn was written under an assistant name.
    Thinking { header: String, text: String },
    Text { header: String, text: String },
    ToolCall(ToolCall),
}

/// What a push made visible.
#[derive(Debug, Clone, PartialEq)]
pub enum Delta {
    Thinking(String),
    Text(String),
    /// A whole call, when its `<|end_message|>` arrives.
    ToolCall(ToolCall),
    /// `<|content_model_end_sampling|>`: the turn is over.
    End,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Body {
    Thinking,
    Text,
    ToolJson,
}

#[derive(Debug, Clone, PartialEq)]
enum State {
    /// Between parts: `<|message_model|>` or `<|content_model_end_sampling|>` next.
    Between,
    /// After `<|message_model|>`: a name, then a content type.
    Header(String),
    Body { kind: Body, header: String, text: String },
    Ended,
    Failed(String),
}

/// A token of the model's output, after decoding.
enum Tok {
    Special(String),
    Text(String),
}

/// The model's turn, parsed as it streams: from ids ([`TurnParser::push_ids`]) or from text
/// ([`TurnParser::push_text`]), never both on one parser.
///
/// Ids are authoritative: a special token is one id, and text is decoded only between them, holding back an
/// incomplete UTF-8 character. Text is for when only text is at hand: markers are matched as strings, and a tail
/// that could still grow into one is held until the next chunk. In text, a marker the model spelled out of ordinary
/// tokens cannot be told from a real one.
///
/// Anything the template would not have written in an assistant turn is refused by name, and the parser stays
/// refused: text before `<|message_model|>`, another role's marker, a content type the template never gives the
/// model (`<|content_invoke_tool_text|>`, `<|content_xml|>`, `<|content_tool_error|>`, `<|content_image|>`), a part
/// not closed by `<|end_message|>`, a tool call whose JSON is not `{"name": NAME, "args": {…}}` with NAME the header's
/// name, anything after the end of the turn. [`TurnParser::finish`] refuses a turn that did not end, and names the
/// truncated call when there is one.
pub struct TurnParser<'a> {
    chat: &'a InkChat,
    state: State,
    parts: Vec<Part>,
    /// Ids of the current text run not yet decoded to whole characters.
    pending: Vec<u32>,
    /// Text not yet lexed (text mode).
    hold: String,
}

impl<'a> TurnParser<'a> {
    pub fn new(chat: &'a InkChat) -> Self {
        TurnParser { chat, state: State::Between, parts: Vec::new(), pending: Vec::new(), hold: String::new() }
    }

    /// The parts so far, whole parts only.
    pub fn parts(&self) -> &[Part] {
        &self.parts
    }

    pub fn ended(&self) -> bool {
        self.state == State::Ended
    }

    /// Feed generated ids.
    pub fn push_ids(&mut self, ids: &[u32]) -> anyhow::Result<Vec<Delta>> {
        let mut out = Vec::new();
        for &id in ids {
            if let Some(s) = self.chat.specials.get(&id) {
                self.flush_pending(&mut out, true)?;
                self.step(Tok::Special(s.clone()), &mut out)?;
            } else {
                self.pending.push(id);
            }
        }
        self.flush_pending(&mut out, false)?;
        Ok(out)
    }

    fn flush_pending(&mut self, out: &mut Vec<Delta>, all: bool) -> anyhow::Result<()> {
        if self.pending.is_empty() {
            return Ok(());
        }
        let text = self.chat.tok.decode(&self.pending, false).map_err(|e| anyhow!("{e}"))?;
        // A character split across ids decodes to U+FFFD until its last byte arrives.
        if !all && text.ends_with('\u{FFFD}') {
            return Ok(());
        }
        self.pending.clear();
        if !text.is_empty() {
            self.step(Tok::Text(text), out)?;
        }
        Ok(())
    }

    /// Feed generated text.
    pub fn push_text(&mut self, chunk: &str) -> anyhow::Result<Vec<Delta>> {
        let mut out = Vec::new();
        self.hold.push_str(chunk);
        loop {
            // The first control token in the held text, longest first at its place.
            let found = (0..self.hold.len())
                .filter(|&i| self.hold.is_char_boundary(i))
                .find_map(|i| self.chat.control_at(&self.hold, i).map(|c| (i, c.to_string())));
            let Some((at, c)) = found else { break };
            let before: String = self.hold[..at].to_string();
            self.hold.drain(..at + c.len());
            if !before.is_empty() {
                self.step(Tok::Text(before), &mut out)?;
            }
            self.step(Tok::Special(c), &mut out)?;
        }
        // Hold back the longest tail that is the start of a control token.
        let keep = self
            .chat
            .controls
            .iter()
            .flat_map(|(c, _)| (1..c.len()).rev().filter(move |&k| c.is_char_boundary(k)).map(move |k| &c[..k]))
            .filter(|p| self.hold.ends_with(p))
            .map(str::len)
            .max()
            .unwrap_or(0);
        let ready = self.hold.len() - keep;
        if ready > 0 {
            let text: String = self.hold.drain(..ready).collect();
            self.step(Tok::Text(text), &mut out)?;
        }
        Ok(out)
    }

    fn fail<T>(&mut self, why: String) -> anyhow::Result<T> {
        self.state = State::Failed(why.clone());
        Err(anyhow!("inkchat: {why}"))
    }

    fn step(&mut self, tok: Tok, out: &mut Vec<Delta>) -> anyhow::Result<()> {
        let state = std::mem::replace(&mut self.state, State::Between);
        match (state, tok) {
            (State::Failed(why), _) => {
                self.state = State::Failed(why.clone());
                bail!("inkchat: the parser already refused this turn: {why}")
            }
            (State::Ended, Tok::Special(s)) => self.fail(format!("{s} after the end of the turn")),
            (State::Ended, Tok::Text(t)) => self.fail(format!("text after the end of the turn: {t:?}")),
            (State::Between, Tok::Special(s)) if s == mark::MODEL => {
                self.state = State::Header(String::new());
                Ok(())
            }
            (State::Between, Tok::Special(s)) if s == mark::END_SAMPLING => {
                self.state = State::Ended;
                out.push(Delta::End);
                Ok(())
            }
            (State::Between, Tok::Special(s)) if [mark::USER, mark::SYSTEM, mark::TOOL].contains(&s.as_str()) => {
                self.fail(format!("the model wrote another role's marker {s}"))
            }
            (State::Between, Tok::Special(s)) => self.fail(format!("{s} where a part or the end of the turn starts")),
            (State::Between, Tok::Text(t)) => self.fail(format!("text outside a message: {t:?}")),
            (State::Header(mut h), Tok::Text(t)) => {
                h.push_str(&t);
                self.state = State::Header(h);
                Ok(())
            }
            (State::Header(header), Tok::Special(s)) => {
                let kind = match s.as_str() {
                    mark::THINKING => Body::Thinking,
                    mark::TEXT => Body::Text,
                    mark::INVOKE_JSON => Body::ToolJson,
                    mark::INVOKE_TEXT | mark::XML | mark::TOOL_ERROR | mark::IMAGE => {
                        return self.fail(format!("{s}: the template never gives the model this content type"));
                    }
                    _ => return self.fail(format!("{s} where a content type was expected")),
                };
                self.state = State::Body { kind, header, text: String::new() };
                Ok(())
            }
            (State::Body { kind, header, mut text }, Tok::Text(t)) => {
                match kind {
                    Body::Thinking => out.push(Delta::Thinking(t.clone())),
                    Body::Text => out.push(Delta::Text(t.clone())),
                    Body::ToolJson => {}
                }
                text.push_str(&t);
                self.state = State::Body { kind, header, text };
                Ok(())
            }
            (State::Body { kind, header, text }, Tok::Special(s)) if s == mark::END_MESSAGE => {
                let part = match kind {
                    Body::Thinking => Part::Thinking { header, text },
                    Body::Text => Part::Text { header, text },
                    Body::ToolJson => match tool_call(&header, &text) {
                        Ok(call) => {
                            out.push(Delta::ToolCall(call.clone()));
                            Part::ToolCall(call)
                        }
                        Err(why) => return self.fail(why),
                    },
                };
                self.parts.push(part);
                Ok(())
            }
            (State::Body { kind, .. }, Tok::Special(s)) => {
                self.fail(format!("{s} inside a {kind:?} part, which <|end_message|> did not close"))
            }
        }
    }

    /// The whole turn, or the reason it is not whole.
    pub fn finish(mut self) -> anyhow::Result<Vec<Part>> {
        let mut out = Vec::new();
        self.flush_pending(&mut out, true)?;
        if !self.hold.is_empty() {
            let text = std::mem::take(&mut self.hold);
            self.step(Tok::Text(text), &mut out)?;
        }
        match self.state {
            State::Ended => Ok(self.parts),
            State::Failed(why) => bail!("inkchat: {why}"),
            State::Body { kind: Body::ToolJson, header, text } => {
                bail!("inkchat: truncated tool call {header:?}: the output stops inside its JSON: {text:?}")
            }
            State::Body { kind, .. } => bail!("inkchat: truncated turn: the output stops inside a {kind:?} part"),
            State::Header(h) => bail!("inkchat: truncated turn: the output stops in a part header {h:?}"),
            State::Between => bail!("inkchat: truncated turn: no {} ends it", mark::END_SAMPLING),
        }
    }
}

/// The body of `<|content_invoke_tool_json|>`, checked against the form the template writes.
fn tool_call(header: &str, body: &str) -> Result<ToolCall, String> {
    let v = Json::parse(body).map_err(|e| format!("tool call {header:?}: its JSON does not parse ({e}): {body:?}"))?;
    let Json::Object(o) = &v else { return Err(format!("tool call {header:?}: its JSON is not an object")) };
    let keys: Vec<&str> = o.iter().map(|(k, _)| k.as_str()).collect();
    if keys.len() != 2 || v.get("name").is_none() || v.get("args").is_none() {
        return Err(format!("tool call {header:?}: keys {keys:?}, where the template writes exactly name and args"));
    }
    let Some(name) = v.get("name").and_then(Json::as_str) else {
        return Err(format!("tool call {header:?}: its name is not a string"));
    };
    if name != header {
        return Err(format!("tool call {header:?}: its JSON names {name:?}"));
    }
    let args = v.get("args").expect("checked");
    if !matches!(args, Json::Object(_)) {
        return Err(format!("tool call {header:?}: its args are not an object"));
    }
    Ok(ToolCall { name: name.to_string(), arguments: args.clone() })
}

/// Parts back to an OpenAI-shaped assistant message, the inverse of the template when every part type appears at
/// most once and in the template's order (thinking, text, calls). Refuses a turn the template cannot write back.
pub fn parts_to_message(parts: &[Part], call_ids: &[&str]) -> anyhow::Result<Json> {
    let mut reasoning: Option<String> = None;
    let mut content: Option<String> = None;
    let mut calls = Vec::new();
    for p in parts {
        match p {
            Part::Thinking { text, .. } if reasoning.is_none() && content.is_none() && calls.is_empty() => {
                reasoning = Some(text.clone())
            }
            Part::Text { text, .. } if content.is_none() && calls.is_empty() => content = Some(text.clone()),
            Part::ToolCall(c) => {
                let id = call_ids.get(calls.len()).context("inkchat: fewer call ids than calls")?;
                calls.push(Json::Object(vec![
                    ("id".into(), Json::str(id)),
                    ("type".into(), Json::str("function")),
                    (
                        "function".into(),
                        Json::Object(vec![("name".into(), Json::str(&c.name)), ("arguments".into(), c.arguments.clone())]),
                    ),
                ]))
            }
            _ => bail!("inkchat: this turn's parts are not in the template's order (thinking, text, calls)"),
        }
    }
    let mut o = vec![("role".to_string(), Json::str("assistant"))];
    if let Some(r) = reasoning {
        o.push(("reasoning_content".into(), Json::Str(r)));
    }
    o.push(("content".into(), content.map(Json::Str).unwrap_or(Json::Null)));
    if !calls.is_empty() {
        o.push(("tool_calls".into(), Json::Array(calls)));
    }
    Ok(Json::Object(o))
}

/// `chat_template.jinja` and `tokenizer.json` of a checkpoint directory, for a caller that wants to check it holds
/// the template this module was written against.
pub const TEMPLATE_SHA256: &str = "53a9035f3255945704b5857657cfb00c1165090c0641c4f782a0cbf268cc16dc";
