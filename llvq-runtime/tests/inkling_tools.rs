//! Inkling 2's tool format in the runtime, with no model loaded: the routes'
//! messages rendered by the checkpoint's template (`llvq_llm::inkchat`), the
//! calls the engine streams read back by the routes' parser, and the key an
//! answer is held under against the message a client sends back.
//!
//! Two halves, as `llvq-llm/tests/inkling_chat.rs`. On a byte-level
//! tokenizer that carries Inkling's markers, everywhere. On the checkpoint's
//! tokenizer and the template's own rendering (`ops/inkling/chat_ref.py`,
//! `$INKLING_DATA/lot2-k/chat_ref.json`), behind an unconditional `#[ignore]`
//! that fails naming the file it lacks.

#[path = "common/ink.rs"]
mod ink;

use llvq_llm::inkchat::{mark, InkChat, Json, TurnParser};
use llvq_runtime::inkling::answer_key;
use llvq_runtime::tools::{inkling2, Args, Call, Calls, HistCall, OrderedJson, Piece, ToolDef, ToolFormat, ToolMsg, ToolRole};
use std::path::PathBuf;

fn chat() -> InkChat {
    InkChat::new(ink::byte_tokenizer()).unwrap()
}

fn oj(s: &str) -> OrderedJson {
    serde_json::from_str(s).unwrap()
}

fn tool_defs() -> Vec<ToolDef> {
    let t = oj(r#"{"type": "function", "function": {"name": "list_dir", "description": "List a directory.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}"#);
    vec![ToolDef::parse(0, &t).unwrap()]
}

fn msg(role: ToolRole, content: &str) -> ToolMsg {
    ToolMsg { role, content: content.into(), calls: Vec::new(), name: None }
}

/// A coding agent's round trip, as the OpenAI route reads it.
fn conversation() -> Vec<ToolMsg> {
    let call = HistCall {
        id: Some("call-1".into()),
        name: "list_dir".into(),
        arguments: Args::Text(r#"{"path": ".", "depth": {"max": 2, "follow": false}, "all": true}"#.into()),
    };
    vec![
        msg(ToolRole::System, "You are a coding agent."),
        msg(ToolRole::User, "list the files"),
        ToolMsg { role: ToolRole::Assistant, content: String::new(), calls: vec![call], name: None },
        ToolMsg { name: Some("list_dir".into()), ..msg(ToolRole::Tool, "a.txt b.txt") },
    ]
}

#[test]
fn a_tool_conversation_renders_as_the_template_writes_it() {
    let (tools, msgs) = inkling2::messages(&tool_defs(), &conversation()).unwrap();
    let r = chat().render(&msgs, &tools, Some(&Json::str("none"))).unwrap();
    // Tools first, sorted and compact; the system message; the effort before
    // the first message that is not a system one; the call as its own model
    // message, arguments sorted at the top level only; the result headed by
    // the function's name.
    let want = concat!(
        "<|message_system|>tool_declare<|content_xml|>",
        r#"[{"description":"List a directory.","name":"list_dir","parameters":{"properties":{"path":{"type":"string"}},"required":["path"],"type":"object"},"type":"function"}]"#,
        "<|end_message|>",
        "<|message_system|><|content_text|>You are a coding agent.<|end_message|>",
        "<|message_system|><|content_text|>Thinking effort level: 0<|end_message|>",
        "<|message_user|><|content_text|>list the files<|end_message|>",
        "<|message_model|>list_dir<|content_invoke_tool_json|>",
        r#"{"name":"list_dir","args":{"all":true,"depth":{"max":2,"follow":false},"path":"."}}"#,
        "<|end_message|><|content_model_end_sampling|>",
        "<|message_tool|>list_dir<|content_text|>a.txt b.txt<|end_message|>",
    );
    assert_eq!(r.text, want);
    assert_eq!(r.assistant_spans.len(), 1);
}

#[test]
fn arguments_must_be_an_object() {
    let mut c = conversation();
    c[2].calls[0].arguments = Args::Text("[1, 2]".into());
    let e = inkling2::messages(&tool_defs(), &c).unwrap_err();
    assert!(e.contains("messages[2].tool_calls[0]") && e.contains("object"), "{e}");
    c[2].calls[0].arguments = Args::Text("{\"path\": ".into());
    assert!(inkling2::messages(&tool_defs(), &c).unwrap_err().contains("messages[2]"));
    // Ollama's form, an object, and an empty string, a call with none.
    c[2].calls[0].arguments = Args::Object(oj(r#"{"path": "."}"#));
    assert!(inkling2::messages(&tool_defs(), &c).is_ok());
    c[2].calls[0].arguments = Args::Text(String::new());
    let (_, m) = inkling2::messages(&tool_defs(), &c).unwrap();
    let args = m[2].get("tool_calls").and_then(|t| match t {
        Json::Array(a) => a[0].get("function")?.get("arguments").cloned(),
        _ => None,
    });
    assert_eq!(args, Some(Json::Object(Vec::new())));
}

#[test]
fn numbers_keep_pythons_forms() {
    assert_eq!(inkling2::to_json(&oj("[1, -3, 0.5, 1e16, 2.5e-7, 12345678901234567890]")).dumps(false, true), "[1,-3,0.5,1e+16,2.5e-07,12345678901234567890]");
}

/// What the routes do with the engine's stream: the calls whole, the text
/// around them as content.
fn read(chunks: &[&str]) -> Vec<Piece> {
    let mut c = Calls::new(ToolFormat::Inkling2);
    let mut out = Vec::new();
    for k in chunks {
        out.extend(c.delta(k, ""));
    }
    out.extend(c.finish());
    let mut merged: Vec<Piece> = Vec::new();
    for p in out {
        match (merged.last_mut(), p) {
            (Some(Piece::Content(a)), Piece::Content(b)) => a.push_str(&b),
            (_, p) => merged.push(p),
        }
    }
    merged
}

fn call(name: &str, args: &str) -> Piece {
    Piece::Call(Call { name: name.into(), arguments: oj(args) })
}

#[test]
fn a_streamed_call_reads_back_whole_at_every_cut() {
    let args = Json::parse(r#"{"path": ".", "n": 2}"#).unwrap();
    let one = inkling2::call_text("list_dir", &args);
    assert_eq!(one, r#"<|content_invoke_tool_json|>{"name":"list_dir","args":{"path":".","n":2}}<|end_message|>"#);
    let stream = format!("Let me look.\n{one}{}", inkling2::call_text("read_file", &Json::parse(r#"{"path": "a.txt"}"#).unwrap()));
    let want = vec![
        Piece::Content("Let me look.".into()),
        call("list_dir", r#"{"path": ".", "n": 2}"#),
        call("read_file", r#"{"path": "a.txt"}"#),
    ];
    assert_eq!(read(&[&stream]), want);
    for k in 1..stream.len() {
        if stream.is_char_boundary(k) {
            assert_eq!(read(&[&stream[..k], &stream[k..]]), want, "cut at byte {k}");
        }
    }
    // A body that does not read as a call is content, with a warning.
    let bad = format!("{}{{\"args\": {{}}}}{}", mark::INVOKE_JSON, mark::END_MESSAGE);
    let mut c = Calls::new(ToolFormat::Inkling2);
    let mut ps = c.delta(&bad, "");
    ps.extend(c.finish());
    assert_eq!(ps, vec![Piece::Content(bad.clone())]);
    assert_eq!((c.count, c.warnings.len()), (0, 1));
}

/// The ids of a turn the model writes, as the byte tokenizer gives them.
fn ids(chat: &InkChat, text: &str) -> Vec<u32> {
    chat.tokenizer().encode(text, false).unwrap().get_ids().to_vec()
}

/// What a client sends back after the routes streamed `parts`: the content
/// it saw and each call with the arguments string it got.
fn echo(content: &str, calls: &[Piece]) -> ToolMsg {
    let calls = calls
        .iter()
        .enumerate()
        .map(|(k, p)| match p {
            Piece::Call(c) => HistCall { id: Some(format!("call-{k}")), name: c.name.clone(), arguments: Args::Text(c.arguments.dumps()) },
            _ => panic!("a call"),
        })
        .collect();
    ToolMsg { role: ToolRole::Assistant, content: content.into(), calls, name: None }
}

/// The key an answer is held under is the ids the template renders for the
/// message the client sends back: the prefix reuse depends on it. The model
/// wrote its JSON with spaces and keys out of order, and a thinking part the
/// routes do not send back.
#[test]
fn an_answer_is_keyed_by_the_message_the_client_sends_back() {
    let chat = chat();
    let turn = concat!(
        "<|message_model|><|content_thinking|>Look first.<|end_message|>",
        "<|message_model|><|content_text|>I will list it.\n<|end_message|>",
        "<|message_model|>list_dir<|content_invoke_tool_json|>",
        r#"{"name": "list_dir", "args": {"path": ".", "all": true, "opts": {"z": 1.50, "a": [1]}}}"#,
        "<|end_message|><|content_model_end_sampling|>",
    );
    let mut p = TurnParser::new(&chat);
    let mut streamed = String::new();
    for d in p.push_ids(&ids(&chat, turn)).unwrap() {
        match d {
            llvq_llm::inkchat::Delta::Text(t) => streamed.push_str(&t),
            llvq_llm::inkchat::Delta::ToolCall(c) => streamed.push_str(&inkling2::call_text(&c.name, &c.arguments)),
            _ => {}
        }
    }
    let key = answer_key(&chat, p.parts()).unwrap();
    let pieces = read(&[&streamed]);
    let (content, calls) = match &pieces[..] {
        [Piece::Content(c), rest @ ..] => (c.clone(), rest.to_vec()),
        o => panic!("{o:?}"),
    };
    assert_eq!(content, "I will list it.");
    assert_eq!(calls, vec![call("list_dir", r#"{"path": ".", "all": true, "opts": {"z": 1.5, "a": [1]}}"#)]);
    let back = vec![msg(ToolRole::User, "list the files"), echo(&content, &calls)];
    let (tools, msgs) = inkling2::messages(&tool_defs(), &back).unwrap();
    let r = chat.render(&msgs, &tools, Some(&Json::str("none"))).unwrap();
    let span = r.assistant_spans[0].clone();
    assert_eq!(r.ids[span].to_vec(), key, "the key is not what the client's message renders to");

    // A plain answer: the text as it is, trailing whitespace included.
    let mut p = TurnParser::new(&chat);
    p.push_ids(&ids(&chat, "<|message_model|><|content_text|>Done.\n<|end_message|><|content_model_end_sampling|>"))
        .unwrap();
    let key = answer_key(&chat, p.parts()).unwrap();
    let r = chat.render(&[Json::parse(r#"{"role": "assistant", "content": "Done.\n"}"#).unwrap()], &[], None).unwrap();
    assert_eq!(r.ids[r.assistant_spans[0].clone()].to_vec(), key);
}

// ---------------------------------------------------------------------------
// On the checkpoint.
// ---------------------------------------------------------------------------

fn data_dir() -> PathBuf {
    std::env::var("INKLING_DATA")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(std::env::var("HOME").unwrap_or_default()).join("llvq-inkling-mini"))
}

/// The reference cases the routes can carry (string content, calls, tool
/// results; no reasoning in the history, no assistant name, no parts), read
/// into `ToolMsg` as the OpenAI route reads them: arguments as the JSON
/// string a client sends, a tool result named by the call its id answers.
#[test]
#[ignore = "needs $INKLING_DATA/native/tokenizer.json and $INKLING_DATA/lot2-k/chat_ref.json"]
fn the_routes_messages_render_as_the_reference_byte_for_byte_and_id_for_id() {
    let dir = data_dir();
    let tok_path = dir.join("native/tokenizer.json");
    let tok = tokenizers::Tokenizer::from_file(&tok_path).unwrap_or_else(|e| panic!("{}: {e}", tok_path.display()));
    let chat = InkChat::new(tok).unwrap();
    let ref_path = dir.join("lot2-k/chat_ref.json");
    let text = std::fs::read_to_string(&ref_path).unwrap_or_else(|e| panic!("{}: {e}", ref_path.display()));
    let doc = Json::parse(&text).unwrap();
    let ordered: OrderedJson = serde_json::from_str(&text).unwrap();
    let Some(Json::Array(cases)) = doc.get("cases") else { panic!("cases") };
    let Some(OrderedJson::Arr(ocases)) = ordered.get("cases") else { panic!("cases") };
    let names = [
        "plain_user",
        "system_user",
        "multi_turn",
        "effort_none",
        "tools_declared_unused",
        "one_tool_call_and_result",
        "tool_result_json",
        "tool_result_plain_unmatched_id",
        "nested_arguments_numbers",
    ];
    let mut ran = 0;
    for (c, oc) in cases.iter().zip(ocases) {
        let name = c.get("name").and_then(Json::as_str).unwrap();
        if !names.contains(&name) {
            continue;
        }
        let Some(Json::Array(ms)) = c.get("messages") else { panic!("{name}: messages") };
        let mut msgs: Vec<ToolMsg> = Vec::new();
        for m in ms {
            let role = match m.get("role").and_then(Json::as_str).unwrap() {
                "system" => ToolRole::System,
                "user" => ToolRole::User,
                "assistant" => ToolRole::Assistant,
                "tool" => ToolRole::Tool,
                r => panic!("{name}: {r}"),
            };
            let calls: Vec<HistCall> = match m.get("tool_calls") {
                Some(Json::Array(cs)) => cs
                    .iter()
                    .map(|k| {
                        let f = k.get("function").unwrap();
                        HistCall {
                            id: k.get("id").and_then(Json::as_str).map(str::to_string),
                            name: f.get("name").and_then(Json::as_str).unwrap().into(),
                            arguments: Args::Text(f.get("arguments").unwrap().dumps(false, false)),
                        }
                    })
                    .collect(),
                _ => Vec::new(),
            };
            let name_of = m.get("name").and_then(Json::as_str).map(str::to_string).or_else(|| {
                let id = m.get("tool_call_id").and_then(Json::as_str)?;
                msgs.iter().rev().flat_map(|p| p.calls.iter()).find(|k| k.id.as_deref() == Some(id)).map(|k| k.name.clone())
            });
            let content = m.get("content").and_then(Json::as_str).unwrap_or("").to_string();
            msgs.push(ToolMsg { role, content, calls, name: if role == ToolRole::Tool { name_of } else { None } });
        }
        let tools: Vec<ToolDef> = match oc.get("tools") {
            // The routes require `type` and a `function` object; the template
            // reads a flat tool as its function and writes "function" for an
            // absent type, so the text is the same.
            Some(OrderedJson::Arr(t)) => t
                .iter()
                .enumerate()
                .map(|(i, t)| match (t.get("type"), t) {
                    (None, OrderedJson::Obj(_)) if t.get("function").is_none() => {
                        let w = vec![("type".into(), OrderedJson::Str("function".into())), ("function".into(), t.clone())];
                        ToolDef::parse(i, &OrderedJson::Obj(w)).unwrap()
                    }
                    (None, OrderedJson::Obj(kv)) => {
                        let mut kv = kv.clone();
                        kv.insert(0, ("type".into(), OrderedJson::Str("function".into())));
                        ToolDef::parse(i, &OrderedJson::Obj(kv)).unwrap()
                    }
                    _ => ToolDef::parse(i, t).unwrap(),
                })
                .collect(),
            _ => Vec::new(),
        };
        let (tools, msgs) = inkling2::messages(&tools, &msgs).unwrap();
        let effort = c.get("reasoning_effort");
        let r = chat.render(&msgs, &tools, effort).unwrap_or_else(|e| panic!("{name}: {e:#}"));
        assert_eq!(r.text, c.get("text").and_then(Json::as_str).unwrap(), "{name}: text");
        let want: Vec<u32> = match c.get("ids") {
            Some(Json::Array(a)) => a.iter().map(|v| match v {
                Json::Int(s) => s.parse().unwrap(),
                _ => panic!("an id"),
            }).collect(),
            _ => panic!("{name}: ids"),
        };
        assert_eq!(r.ids, want, "{name}: ids");
        ran += 1;
    }
    assert_eq!(ran, names.len(), "every named case is in the reference");
}
