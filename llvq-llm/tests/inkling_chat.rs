//! Inkling 2 Mini's chat format (`llvq_llm::inkchat`) against the checkpoint's own template.
//!
//! Two halves.
//! - On a synthetic byte-level tokenizer that carries Inkling's markers: the parser (chunks, split markers, split
//!   characters, refusals), the Python number and string forms, the refusals of the renderer. These run everywhere.
//! - On the checkpoint (`$INKLING_DATA/native/tokenizer.json`, default `~/llvq-inkling-mini`) and the reference
//!   that `ops/inkling/chat_ref.py` writes (`$INKLING_DATA/lot2-k/chat_ref.json`): every case byte for byte and id
//!   for id, every refusal with the template's message, and the round trip through the parser. These carry an
//!   unconditional `#[ignore]`, like the sealed-archive tests, and fail naming the file they lack.

use std::path::PathBuf;

use llvq_llm::inkchat::{mark, parts_to_message, py_float_repr, Delta, InkChat, Json, Part, ToolCall, TurnParser};

// ---------------------------------------------------------------------------------------------------------------
// Synthetic tokenizer
// ---------------------------------------------------------------------------------------------------------------

const MARKS: [&str; 15] = [
    mark::END_OF_TEXT,
    mark::USER,
    mark::MODEL,
    mark::SYSTEM,
    mark::TOOL,
    mark::TEXT,
    mark::IMAGE,
    mark::END_SAMPLING,
    mark::THINKING,
    mark::END_MESSAGE,
    mark::TOOL_ERROR,
    mark::XML,
    mark::INVOKE_JSON,
    mark::INVOKE_TEXT,
    "<|unused_200054|>",
];

/// GPT-2's byte alphabet, one token a byte, and Inkling's markers as special added tokens from id 300.
fn byte_tokenizer() -> tokenizers::Tokenizer {
    let chars: Vec<u32> = (33..=126).chain(161..=172).chain(174..=255).collect();
    let mut vocab = serde_json::Map::new();
    let mut extra = 0;
    for b in 0..=255u32 {
        let c = if chars.contains(&b) {
            b
        } else {
            extra += 1;
            255 + extra
        };
        vocab.insert(char::from_u32(c).unwrap().to_string(), b.into());
    }
    let added: Vec<serde_json::Value> = MARKS
        .iter()
        .enumerate()
        .map(|(i, m)| {
            serde_json::json!({"id": 300 + i, "content": m, "single_word": false, "lstrip": false,
                               "rstrip": false, "normalized": false, "special": true})
        })
        .collect();
    let bl = serde_json::json!({"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false});
    let j = serde_json::json!({
        "version": "1.0", "truncation": null, "padding": null, "added_tokens": added,
        "normalizer": null, "pre_tokenizer": bl, "post_processor": null, "decoder": bl,
        "model": {"type": "BPE", "vocab": vocab, "merges": [], "dropout": null, "unk_token": null,
                  "continuing_subword_prefix": null, "end_of_word_suffix": null, "fuse_unk": false,
                  "byte_fallback": false, "ignore_merges": false}
    });
    tokenizers::Tokenizer::from_bytes(serde_json::to_vec(&j).unwrap()).unwrap()
}

fn j(text: &str) -> Json {
    Json::parse(text).unwrap()
}

/// The assistant turn of the round trip: thinking, text, two calls, unicode, a marker look-alike, nested arguments.
fn round_trip_message() -> Json {
    j(r#"{"role": "assistant",
          "reasoning_content": "Two calls. caf\u00e9 \ud83d\ude00 and <|end_message| looks like a marker.",
          "content": "Checking both: \u4e2d\u6587.",
          "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "get_weather",
             "arguments": {"city": "Z\u00fcrich", "unit": "celsius"}}},
            {"id": "c2", "type": "function", "function": {"name": "search_files",
             "arguments": {"max_results": 5, "pattern": "TODO", "opts": {"z": 1.5, "a": [true, null, "x"]}}}}
          ]}"#)
}

fn expected_parts(msg: &Json) -> Vec<Part> {
    let mut parts = vec![
        Part::Thinking { header: String::new(), text: msg.get("reasoning_content").unwrap().as_str().unwrap().into() },
        Part::Text { header: String::new(), text: msg.get("content").unwrap().as_str().unwrap().into() },
    ];
    let Json::Array(calls) = msg.get("tool_calls").unwrap() else { panic!() };
    for c in calls {
        let f = c.get("function").unwrap();
        parts.push(Part::ToolCall(ToolCall {
            name: f.get("name").unwrap().as_str().unwrap().into(),
            arguments: f.get("arguments").unwrap().clone(),
        }));
    }
    parts
}

/// Parts equal, tool arguments compared as Python dicts (the template sorts the top level).
fn same_parts(a: &[Part], b: &[Part]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(x, y)| match (x, y) {
            (Part::ToolCall(p), Part::ToolCall(q)) => p.name == q.name && p.arguments.same(&q.arguments),
            _ => x == y,
        })
}

/// Feed `ids` in chunks of `step`, checking the deltas add up to the parts.
fn parse_ids(chat: &InkChat, ids: &[u32], step: usize) -> anyhow::Result<Vec<Part>> {
    let mut p = TurnParser::new(chat);
    let mut thinking = String::new();
    let mut text = String::new();
    let mut ends = 0;
    for chunk in ids.chunks(step) {
        for d in p.push_ids(chunk)? {
            match d {
                Delta::Thinking(t) => thinking.push_str(&t),
                Delta::Text(t) => text.push_str(&t),
                Delta::ToolCall(_) => {}
                Delta::End => ends += 1,
            }
        }
    }
    let parts = p.finish()?;
    check_deltas(&parts, &thinking, &text, ends);
    Ok(parts)
}

fn parse_text(chat: &InkChat, s: &str, step: usize) -> anyhow::Result<Vec<Part>> {
    let mut p = TurnParser::new(chat);
    let chars: Vec<char> = s.chars().collect();
    let mut thinking = String::new();
    let mut text = String::new();
    let mut ends = 0;
    for chunk in chars.chunks(step) {
        for d in p.push_text(&chunk.iter().collect::<String>())? {
            match d {
                Delta::Thinking(t) => thinking.push_str(&t),
                Delta::Text(t) => text.push_str(&t),
                Delta::ToolCall(_) => {}
                Delta::End => ends += 1,
            }
        }
    }
    let parts = p.finish()?;
    check_deltas(&parts, &thinking, &text, ends);
    Ok(parts)
}

fn check_deltas(parts: &[Part], thinking: &str, text: &str, ends: usize) {
    let t: String = parts.iter().filter_map(|p| if let Part::Thinking { text, .. } = p { Some(text.as_str()) } else { None }).collect();
    let x: String = parts.iter().filter_map(|p| if let Part::Text { text, .. } = p { Some(text.as_str()) } else { None }).collect();
    assert_eq!(thinking, t, "the thinking deltas add up to the thinking parts");
    assert_eq!(text, x, "the text deltas add up to the text parts");
    assert_eq!(ends, 1, "one end of turn");
}

/// The assistant span of `[user, msg]`.
fn assistant_ids(chat: &InkChat, msg: &Json) -> Vec<u32> {
    let msgs = [j(r#"{"role": "user", "content": "go"}"#), msg.clone()];
    let r = chat.render(&msgs, &[], None).unwrap();
    assert_eq!(r.assistant_spans.len(), 1);
    let span = r.assistant_spans[0].clone();
    assert_eq!(span.end, r.ids.len(), "the assistant turn ends the prompt");
    r.ids[span].to_vec()
}

fn round_trip(chat: &InkChat) {
    let msg = round_trip_message();
    let want = expected_parts(&msg);
    let ids = assistant_ids(chat, &msg);
    let text = chat.tokenizer().decode(&ids, false).unwrap();
    for step in [1, 2, 3, 7] {
        let got = parse_ids(chat, &ids, step).unwrap();
        assert!(same_parts(&got, &want), "ids in chunks of {step}: {got:?}");
        let got = parse_text(chat, &text, step).unwrap();
        assert!(same_parts(&got, &want), "text in chunks of {step}: {got:?}");
    }
    // And back to the message: rendering it again gives the same ids.
    let parts = parse_ids(chat, &ids, 1).unwrap();
    let back = parts_to_message(&parts, &["c1", "c2"]).unwrap();
    assert_eq!(assistant_ids(chat, &back), ids, "parse then render is the identity on ids");
}

#[test]
fn round_trip_on_a_byte_tokenizer() {
    let chat = InkChat::new(byte_tokenizer()).unwrap();
    round_trip(&chat);
}

#[test]
fn a_marker_split_across_text_chunks_is_still_a_marker() {
    let chat = InkChat::new(byte_tokenizer()).unwrap();
    let mut p = TurnParser::new(&chat);
    let mut deltas = Vec::new();
    for chunk in ["<|mess", "age_model|><|content_te", "xt|>Hi <|end", "_mess", "age|><|content_model_end", "_sampling|>"] {
        deltas.extend(p.push_text(chunk).unwrap());
    }
    assert_eq!(deltas, vec![Delta::Text("Hi ".into()), Delta::End]);
    assert_eq!(p.finish().unwrap(), vec![Part::Text { header: String::new(), text: "Hi ".into() }]);
}

#[test]
fn a_character_split_across_ids_is_not_streamed_broken() {
    let chat = InkChat::new(byte_tokenizer()).unwrap();
    let msg = j(r#"{"role": "assistant", "content": "\u00e9\ud83d\ude00"}"#);
    let ids = assistant_ids(&chat, &msg);
    let mut p = TurnParser::new(&chat);
    for id in &ids {
        for d in p.push_ids(&[*id]).unwrap() {
            if let Delta::Text(t) = d {
                assert!(!t.contains('\u{FFFD}'), "a delta holds a broken character: {t:?}");
            }
        }
    }
    assert_eq!(p.finish().unwrap(), vec![Part::Text { header: String::new(), text: "\u{e9}\u{1F600}".into() }]);
}

#[test]
fn the_parser_refuses_what_the_template_never_writes() {
    let chat = InkChat::new(byte_tokenizer()).unwrap();
    let refuse = |text: &str, why: &str| {
        let mut p = TurnParser::new(&chat);
        let e = match p.push_text(text) {
            Err(e) => e,
            Ok(_) => p.finish().unwrap_err(),
        };
        assert!(e.to_string().contains(why), "{text:?}: {e}");
    };
    let m = mark::MODEL;
    refuse("hello", "text outside a message");
    refuse(&format!("{m}{}x{}", mark::INVOKE_TEXT, mark::END_MESSAGE), "never gives the model");
    refuse(&format!("{m}{}x{}", mark::XML, mark::END_MESSAGE), "never gives the model");
    refuse(&format!("{m}{}hi{}{}", mark::TEXT, mark::END_MESSAGE, mark::USER), "another role's marker");
    refuse(&format!("{m}{}hi{}", mark::TEXT, mark::END_SAMPLING), "did not close");
    refuse(&format!("{m}{}hi{}{}x", mark::TEXT, mark::END_MESSAGE, mark::END_SAMPLING), "after the end of the turn");
    // A truncated call names itself.
    refuse(&format!("{m}get_weather{}{{\"name\":\"get_weather\",\"args\":{{\"ci", mark::INVOKE_JSON), "truncated tool call \"get_weather\"");
    refuse(&format!("{m}{}half", mark::THINKING), "truncated turn");
    refuse(&format!("{m}{}hi{}", mark::TEXT, mark::END_MESSAGE), "truncated turn: no <|content_model_end_sampling|>");
    // A call whose JSON is not the template's form.
    let call = |body: &str| format!("{m}f{}{body}{}{}", mark::INVOKE_JSON, mark::END_MESSAGE, mark::END_SAMPLING);
    refuse(&call("{\"name\":\"f\",\"args\":{"), "does not parse");
    refuse(&call("{\"name\":\"g\",\"args\":{}}"), "its JSON names \"g\"");
    refuse(&call("{\"name\":\"f\",\"args\":[]}"), "args are not an object");
    refuse(&call("{\"name\":\"f\"}"), "exactly name and args");
    // A refused parser stays refused.
    let mut p = TurnParser::new(&chat);
    assert!(p.push_text("oops").is_err());
    assert!(p.push_text(m).unwrap_err().to_string().contains("already refused"));
}

#[test]
fn two_calls_stream_as_two_deltas() {
    let chat = InkChat::new(byte_tokenizer()).unwrap();
    let ids = assistant_ids(&chat, &round_trip_message());
    let mut p = TurnParser::new(&chat);
    let calls: Vec<ToolCall> = p
        .push_ids(&ids)
        .unwrap()
        .into_iter()
        .filter_map(|d| if let Delta::ToolCall(c) = d { Some(c) } else { None })
        .collect();
    assert_eq!(calls.iter().map(|c| c.name.as_str()).collect::<Vec<_>>(), ["get_weather", "search_files"]);
}

#[test]
fn python_number_and_string_forms() {
    for (x, s) in [
        (0.9, "0.9"),
        (0.1, "0.1"),
        (1.0, "1.0"),
        (-0.0, "-0.0"),
        (1e-5, "1e-05"),
        (1e-4, "0.0001"),
        (2.5e-7, "2.5e-07"),
        (1e16, "1e+16"),
        (1234567890123456.0, "1234567890123456.0"),
        (123.456, "123.456"),
        (1.5e300, "1.5e+300"),
        (0.30000000000000004, "0.30000000000000004"),
    ] {
        assert_eq!(py_float_repr(x), s, "{x:e}");
    }
    let v = j(r#"{"b": [1, 1.0, -0, "\u00e9\n\t\"\\\u0001/"], "a": {}}"#);
    assert_eq!(v.dumps(false, true), "{\"b\":[1,1.0,0,\"\u{e9}\\n\\t\\\"\\\\\\u0001/\"],\"a\":{}}");
    assert_eq!(v.dumps(true, false), "{\"a\": {}, \"b\": [1, 1.0, 0, \"\u{e9}\\n\\t\\\"\\\\\\u0001/\"]}");
    for bad in ["{\"a\":1,\"a\":2}", "NaN", "[1e400]", "\"\\ud800\"", "01", "[1,]"] {
        assert!(Json::parse(bad).is_err(), "{bad} parsed");
    }
}

#[test]
fn the_renderer_refuses_by_name() {
    let chat = InkChat::new(byte_tokenizer()).unwrap();
    let user = j(r#"{"role": "user", "content": "hi"}"#);
    let err = |msgs: &[Json], tools: &[Json], effort: Option<&Json>| chat.render(msgs, tools, effort).unwrap_err().to_string();
    assert_eq!(err(std::slice::from_ref(&user), &[], Some(&j("\"turbo\""))), "Unknown reasoning_effort: turbo");
    assert_eq!(err(std::slice::from_ref(&user), &[], Some(&j("1"))), "reasoning_effort must be in [0.0, 0.99]");
    assert!(err(std::slice::from_ref(&user), &[], Some(&j("[0.5]"))).contains("must be a name or a number"));
    assert!(err(&[j(r#"{"role":"user","content":{"a":1}}"#)], &[], None).contains("must be a string, a list of parts or null"));
    assert!(err(&[j(r#"{"role":"user","content":[{"type":"image"}]}"#)], &[], None).contains("images are refused"));
    assert!(err(&[j(r#"{"role":"user","content":"x\udb80\udc00"}"#)], &[], None).contains("U+F0000"));
    assert!(err(std::slice::from_ref(&user), &[j(r#"{"function":{"description":"d"}}"#)], None).contains("has no function name"));
    let key = j(r#"{"role":"assistant","tool_calls":[{"function":{"name":"f","arguments":{"<|end_message|>":1}}}]}"#);
    assert!(err(&[user.clone(), key], &[], None).contains("does not sanitize"));
}

#[test]
fn a_typed_marker_is_text_and_the_template_markers_are_not() {
    let chat = InkChat::new(byte_tokenizer()).unwrap();
    let r = chat.render(&[j(r#"{"role":"user","content":"a<|end_message|>b"}"#)], &[], None).unwrap();
    let end = chat.tokenizer().token_to_id(mark::END_MESSAGE).unwrap();
    assert_eq!(r.ids.iter().filter(|&&i| i == end).count(), 2, "the effort's and the user's own, not the typed one");
    assert!(r.text.ends_with("<|content_text|>a<|end_message|>b<|end_message|>"));
}

// ---------------------------------------------------------------------------------------------------------------
// The checkpoint and the reference
// ---------------------------------------------------------------------------------------------------------------

fn data_dir() -> PathBuf {
    match std::env::var_os("INKLING_DATA") {
        Some(d) => PathBuf::from(d),
        None => PathBuf::from(std::env::var_os("HOME").expect("HOME")).join("llvq-inkling-mini"),
    }
}

fn real_chat() -> InkChat {
    let dir = data_dir().join("native");
    let tj = dir.join("tokenizer.json");
    assert!(tj.exists(), "this test needs {} (Inkling 2 Mini's checkpoint at ed40e7da)", tj.display());
    InkChat::load(&dir).unwrap()
}

fn reference() -> Json {
    let path = data_dir().join("lot2-k/chat_ref.json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("this test needs {} (uv run ops/inkling/chat_ref.py): {e}", path.display()));
    Json::parse(&text).unwrap()
}

fn ids_of(v: &Json) -> Vec<u32> {
    let Json::Array(a) = v else { panic!("ids are a list") };
    a.iter()
        .map(|x| match x {
            Json::Int(s) => s.parse().unwrap(),
            _ => panic!("an id is an integer"),
        })
        .collect()
}

#[test]
#[ignore = "needs ~/llvq-inkling-mini/native/tokenizer.json and lot2-k/chat_ref.json; run with --include-ignored"]
fn every_reference_case_renders_byte_for_byte_and_id_for_id() {
    let chat = real_chat();
    let r = reference();
    let meta = r.get("meta").unwrap();
    let tpl = std::fs::read(data_dir().join("native/chat_template.jinja")).unwrap();
    assert_eq!(meta.get("chat_template_sha256").unwrap().as_str(), Some(llvq_llm::inkchat::TEMPLATE_SHA256));
    assert_eq!(tpl.len(), 6868, "the template this module was written against");
    assert_eq!(meta.get("stop").unwrap().get("eos_token_id"), Some(&Json::Int("200006".into())));
    assert_eq!(chat.stop_ids(), vec![200006]);

    let Json::Array(cases) = r.get("cases").unwrap() else { panic!() };
    let (mut rendered, mut refused, mut ids_total) = (0, 0, 0);
    for case in cases {
        let name = case.get("name").unwrap().as_str().unwrap();
        let Json::Array(msgs) = case.get("messages").unwrap() else { panic!() };
        let tools: Vec<Json> = match case.get("tools") {
            Some(Json::Array(t)) => t.clone(),
            _ => Vec::new(),
        };
        let got = chat.render(msgs, &tools, case.get("reasoning_effort"));
        match (case.get("error"), case.get("text")) {
            (Some(want), _) => {
                let e = got.expect_err(name).to_string();
                assert_eq!(e, want.as_str().unwrap(), "{name}: the refusal message");
                refused += 1;
            }
            (None, Some(text)) if name == "image_in_user" => {
                // The template renders it; this module refuses images by name.
                assert!(text.as_str().unwrap().contains(mark::IMAGE));
                assert!(got.unwrap_err().to_string().contains("images are refused"), "{name}");
                refused += 1;
            }
            (None, Some(text)) => {
                let got = got.unwrap_or_else(|e| panic!("{name}: {e}"));
                assert_eq!(got.text, text.as_str().unwrap(), "{name}: text");
                let want = ids_of(case.get("ids").unwrap());
                assert_eq!(got.ids, want, "{name}: ids");
                // Each assistant span starts at a marker and ends on the end of sampling.
                for s in &got.assistant_spans {
                    assert_eq!(got.ids[s.end - 1], chat.end_sampling, "{name}: span end");
                }
                ids_total += want.len();
                rendered += 1;
            }
            _ => panic!("{name}: neither text nor error"),
        }
    }
    eprintln!("{rendered} cases equal in text and ids ({ids_total} ids), {refused} refusals equal");
    assert!(rendered >= 12, "the gate asks for at least 12 rendered cases, got {rendered}");

    // No generation prompt: on and off render the same.
    let by = |n: &str| cases.iter().find(|c| c.get("name").unwrap().as_str() == Some(n)).unwrap();
    for (on, off) in [("plain_user", "plain_user_no_generation_prompt"), ("multi_turn", "multi_turn_no_generation_prompt")] {
        assert_eq!(by(on).get("text"), by(off).get("text"));
        assert_eq!(by(on).get("ids"), by(off).get("ids"));
    }
    // Sanitizing changes the ids of the marker case only.
    for c in cases.iter().filter(|c| c.get("ids").is_some()) {
        let n = c.get("name").unwrap().as_str().unwrap();
        assert_eq!(c.get("ids") != c.get("ids_plain"), n == "unicode_and_marker_lookalikes", "{n}");
    }
}

#[test]
#[ignore = "needs ~/llvq-inkling-mini/native/tokenizer.json; run with --include-ignored"]
fn round_trip_on_the_checkpoint_tokenizer() {
    let chat = real_chat();
    round_trip(&chat);
    // The thinking cost of the default effort, in prompt ids: the system message it adds.
    let user = j(r#"{"role": "user", "content": "hi"}"#);
    let r = chat.render(&[user], &[], None).unwrap();
    assert!(r.text.starts_with("<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>"));
    let user_at = r.ids.iter().position(|&i| i == chat.tokenizer().token_to_id(mark::USER).unwrap()).unwrap();
    eprintln!("the default effort message costs {user_at} prompt ids; the whole prompt of \"hi\" is {}", r.ids.len());
}
