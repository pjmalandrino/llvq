//! Tool calls through the routes, end to end: the real server, engine and
//! `Session`, and a tiny Qwen3 on the CPU built to answer like a model that
//! uses tools.
//!
//! The model is a lookup table written into the weights. Its embedding is one
//! hot, its head reads the current token, so it says the token after it in a
//! fixed chain. One attention head adds a flag to the residual when a
//! `<tool_response>` token is anywhere in the context (the queries and keys are
//! zero, so the head averages every position; only that token has a value).
//! Without the flag the chain after the opening of an answer is a Qwen3 tool
//! call of `list_dir`; with it, the chain is the words `the cat sat`, then a
//! stop token. So the first request answers with a call, and the follow-up
//! that returns the result answers in words, as the round trip needs.

mod common;

use llvq_llm::candle::{DType, Device, Tensor};
use candle_nn::{Activation, VarBuilder, VarMap};
use candle_transformers::models::qwen3::Config;
use llvq_llm::kvq::KvMode;
use llvq_llm::model::Qwen3;
use llvq_llm::served_decode::Decoder;
use llvq_runtime::client::{Client, ClientError};
use llvq_runtime::engine::{Loaded, Loader};
use llvq_runtime::library::ModelName;
use llvq_runtime::store::{Manifest, Store};
use serde_json::{json, Value};
use std::io::{BufRead, BufReader};
use std::sync::Arc;

const SPECIAL: [&str; 11] = [
    "<|im_start|>",
    "<|im_end|>",
    "<|endoftext|>",
    "<think>",
    "</think>",
    "\n",
    "\n\n",
    "<tool_call>",
    "</tool_call>",
    "<tool_response>",
    "</tool_response>",
];
const WORDS: [&str; 11] = [
    "system", "user", "assistant", r#"{"name":"#, r#""list_dir","#, r#""arguments":"#, r#"{"path":"#, r#""."}}"#,
    "the", "cat", "sat",
];
const NL2: usize = 6;
const IM_END: usize = 1;
const EOT: usize = 2;
const TOOL_CALL: usize = 7;
const END_TOOL_CALL: usize = 8;
const TOOL_RESPONSE: usize = 9;
const UNK: usize = SPECIAL.len();

fn word(w: &str) -> usize {
    UNK + 1 + WORDS.iter().position(|x| *x == w).unwrap()
}

/// Qwen3's markers, the tool markers as added tokens that are not special (as
/// in Qwen3's tokenizer), and a few words.
fn tokenizer() -> llvq_llm::tokenizers::Tokenizer {
    let mut vocab = serde_json::Map::new();
    let mut added = Vec::new();
    for (i, s) in SPECIAL.iter().enumerate() {
        vocab.insert((*s).into(), i.into());
        added.push(json!({"id": i, "content": s, "single_word": false, "lstrip": false, "rstrip": false,
            "normalized": false, "special": i < TOOL_CALL}));
    }
    vocab.insert("<unk>".into(), UNK.into());
    for w in WORDS {
        vocab.insert(w.into(), word(w).into());
    }
    let j = json!({
        "version": "1.0", "truncation": null, "padding": null, "added_tokens": added,
        "normalizer": null, "pre_tokenizer": {"type": "WhitespaceSplit"}, "post_processor": null,
        "decoder": null, "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "<unk>"}
    });
    llvq_llm::tokenizers::Tokenizer::from_bytes(serde_json::to_vec(&j).unwrap()).unwrap()
}

const VOCAB: usize = 64;
const HIDDEN: usize = 96;
const FLAG: usize = 64;

fn config() -> Config {
    Config {
        vocab_size: VOCAB,
        hidden_size: HIDDEN,
        intermediate_size: 32,
        num_hidden_layers: 1,
        num_attention_heads: 4,
        head_dim: 16,
        attention_bias: false,
        num_key_value_heads: 2,
        max_position_embeddings: 4096,
        sliding_window: None,
        max_window_layers: 0,
        tie_word_embeddings: false,
        rope_theta: 10_000.0,
        rms_norm_eps: 1e-6,
        use_sliding_window: false,
        hidden_act: Activation::Silu,
    }
}

/// The chains: `(token, the token before it, weight)`. Chain A (the call)
/// weighs 1.0 and is pushed down by the flag; chain B (the words) weighs 0.9
/// and is pushed up by it. Without the flag A wins after `\n\n`; with any
/// flag above 0.05 of the one-hot, B does.
fn chains() -> Vec<(usize, usize, f32, f32)> {
    let a = [
        TOOL_CALL,
        word(r#"{"name":"#),
        word(r#""list_dir","#),
        word(r#""arguments":"#),
        word(r#"{"path":"#),
        word(r#""."}}"#),
        END_TOOL_CALL,
    ];
    let b = [word("the"), word("cat"), word("sat"), EOT];
    let mut out = Vec::new();
    let mut prev = NL2;
    for t in a {
        out.push((t, prev, 1.0, -1.0));
        prev = t;
    }
    out.push((IM_END, prev, 1.0, 0.0));
    let mut prev = NL2;
    for t in b {
        out.push((t, prev, 0.9, 1.0));
        prev = t;
    }
    out
}

fn model() -> Qwen3 {
    let map = VarMap::new();
    let vb = VarBuilder::from_varmap(&map, DType::F32, &Device::Cpu);
    let m = Qwen3::new(&config(), vb, KvMode::F16).unwrap();
    let data = map.data().lock().unwrap();
    let mut set = 0;
    for (n, var) in data.iter() {
        let shape = var.shape().clone();
        let mut v = vec![0f32; shape.elem_count()];
        let cols = *shape.dims().last().unwrap();
        if n.contains("norm") {
            v.fill(1.0);
        } else if n.contains("embed_tokens") {
            for t in 0..VOCAB {
                v[t * cols + t] = 1.0;
            }
            set += 1;
        } else if n.contains("lm_head") {
            for (t, prev, w, flag) in chains() {
                v[t * cols + prev] = w;
                v[t * cols + FLAG] = flag;
            }
            set += 1;
        } else if n.contains("v_proj") {
            // kv head 0, dim 0, reads `<tool_response>`.
            v[TOOL_RESPONSE] = 1000.0;
            set += 1;
        } else if n.contains("o_proj") {
            // head 0, dim 0, into the flag dimension.
            v[FLAG * cols] = 1.0;
            set += 1;
        }
        var.set(&Tensor::from_vec(v, shape, &Device::Cpu).unwrap()).unwrap();
    }
    assert_eq!(set, 4, "the embedding, the head, v_proj and o_proj are named as expected");
    drop(data);
    m
}

struct ToolLoader;

impl Loader for ToolLoader {
    fn load(&self, _: &ModelName, _: &Manifest, _: &Store) -> anyhow::Result<Loaded> {
        Ok(Loaded {
            qwen: model(),
            tokenizer: tokenizer(),
            decoder: Decoder::eager(),
            backend: "dense-cpu".into(),
            config: None,
            lines: Vec::new(),
            batch: None,
        })
    }
}

fn serve(tag: &str) -> Client {
    Client::new(&common::serve_with(common::store(tag), "http://127.0.0.1:9", Arc::new(ToolLoader)))
}

fn tools() -> Value {
    json!([
        {"type": "function", "function": {"name": "list_dir", "description": "List a directory.",
         "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
        {"type": "function", "function": {"name": "read_file", "description": "Read a file.",
         "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}
    ])
}

/// The server-sent events of a streamed OpenAI request, as JSON values, and
/// whether the stream ended with `[DONE]`.
fn sse(c: &Client, body: &Value) -> (Vec<Value>, bool) {
    let r = ureq::post(&format!("{}/v1/chat/completions", c.base())).send_string(&body.to_string()).unwrap();
    let (mut out, mut done) = (Vec::new(), false);
    for line in BufReader::new(r.into_reader()).lines() {
        let line = line.unwrap();
        let Some(d) = line.strip_prefix("data: ") else { continue };
        if d == "[DONE]" {
            done = true;
            continue;
        }
        out.push(serde_json::from_str(d).unwrap());
    }
    (out, done)
}

const ARGS: &str = r#"{"path": "."}"#;

#[test]
fn openai_round_trip_unstreamed() {
    let c = serve("tools-oai");
    let user = json!({"role": "user", "content": "list the files"});
    let r = c
        .send("POST", "/v1/chat/completions", &json!({"model": "tiny:test", "messages": [user], "tools": tools(),
            "temperature": 0, "max_tokens": 20}))
        .unwrap();
    let choice = &r["choices"][0];
    assert_eq!(choice["finish_reason"], "tool_calls", "{r}");
    assert_eq!(choice["message"]["content"], Value::Null, "{r}");
    let call = &choice["message"]["tool_calls"][0];
    assert_eq!(call["type"], "function");
    assert_eq!(call["function"]["name"], "list_dir");
    assert_eq!(call["function"]["arguments"], ARGS);
    let id = call["id"].as_str().unwrap();
    assert!(id.starts_with("call-"), "{id}");
    assert_eq!(choice["message"]["tool_calls"].as_array().unwrap().len(), 1);

    let follow = json!({"model": "tiny:test", "tools": tools(), "temperature": 0, "max_tokens": 20, "messages": [
        user,
        choice["message"].clone(),
        {"role": "tool", "tool_call_id": id, "content": "a.txt b.txt"},
    ]});
    let r = c.send("POST", "/v1/chat/completions", &follow).unwrap();
    let choice = &r["choices"][0];
    assert_eq!(choice["finish_reason"], "stop", "{r}");
    assert_eq!(choice["message"]["content"], "the cat sat");
    assert!(choice["message"].get("tool_calls").is_none(), "{r}");
}

#[test]
fn openai_round_trip_streamed() {
    let c = serve("tools-oai-sse");
    let user = json!({"role": "user", "content": "list the files"});
    let body = json!({"model": "tiny:test", "messages": [user], "tools": tools(), "temperature": 0, "max_tokens": 20,
        "stream": true, "stream_options": {"include_usage": true}});
    let (chunks, done) = sse(&c, &body);
    assert!(done, "the stream ends with [DONE]");
    let calls: Vec<&Value> = chunks.iter().filter_map(|v| v["choices"][0]["delta"].get("tool_calls")).collect();
    assert_eq!(calls.len(), 1, "{chunks:?}");
    let call = &calls[0][0];
    assert_eq!(call["index"], 0, "the stream names the call's index");
    assert_eq!(call["type"], "function");
    assert_eq!(call["function"]["name"], "list_dir");
    assert_eq!(call["function"]["arguments"], ARGS);
    let id = call["id"].as_str().unwrap().to_string();
    let content: String =
        chunks.iter().filter_map(|v| v["choices"][0]["delta"]["content"].as_str()).collect::<Vec<_>>().concat();
    assert_eq!(content, "", "no marker leaks into the content");
    let finish: Vec<&str> = chunks.iter().filter_map(|v| v["choices"][0]["finish_reason"].as_str()).collect();
    assert_eq!(finish, ["tool_calls"]);
    assert!(chunks.last().unwrap()["usage"]["completion_tokens"].as_u64().unwrap() > 0, "usage comes last");

    let follow = json!({"model": "tiny:test", "tools": tools(), "temperature": 0, "max_tokens": 20, "stream": true,
        "messages": [
            user,
            {"role": "assistant", "content": null, "tool_calls": [
                {"id": id, "type": "function", "function": {"name": "list_dir", "arguments": ARGS}}]},
            {"role": "tool", "tool_call_id": id, "content": "a.txt b.txt"},
        ]});
    let (chunks, done) = sse(&c, &follow);
    assert!(done);
    let content: String =
        chunks.iter().filter_map(|v| v["choices"][0]["delta"]["content"].as_str()).collect::<Vec<_>>().concat();
    assert_eq!(content, "the cat sat");
    assert!(chunks.iter().all(|v| v["choices"][0]["delta"].get("tool_calls").is_none()));
    let finish: Vec<&str> = chunks.iter().filter_map(|v| v["choices"][0]["finish_reason"].as_str()).collect();
    assert_eq!(finish, ["stop"]);
}

#[test]
fn ollama_round_trip_streamed_and_not() {
    let c = serve("tools-ollama");
    let user = json!({"role": "user", "content": "list the files"});
    for stream in [false, true] {
        let body = json!({"model": "tiny:test", "messages": [user], "tools": tools(), "stream": stream,
            "options": {"temperature": 0, "num_predict": 20}});
        let mut chunks = Vec::new();
        match stream {
            false => chunks.push(c.send("POST", "/api/chat", &body).unwrap()),
            true => c
                .stream("/api/chat", &body, &mut |v| {
                    chunks.push(v.clone());
                    Ok(())
                })
                .unwrap(),
        }
        let calls: Vec<&Value> = chunks.iter().filter_map(|v| v["message"].get("tool_calls")).collect();
        assert_eq!(calls.len(), 1, "stream {stream}: {chunks:?}");
        assert_eq!(calls[0][0]["function"]["name"], "list_dir");
        assert_eq!(calls[0][0]["function"]["arguments"], json!({"path": "."}));
        let content: String = chunks.iter().filter_map(|v| v["message"]["content"].as_str()).collect::<Vec<_>>().concat();
        assert_eq!(content, "", "stream {stream}");
        assert_eq!(chunks.last().unwrap()["done_reason"], "stop");

        let follow = json!({"model": "tiny:test", "tools": tools(), "stream": stream,
            "options": {"temperature": 0, "num_predict": 20}, "messages": [
                user,
                {"role": "assistant", "content": "", "tool_calls": [calls[0][0].clone()]},
                {"role": "tool", "tool_name": "list_dir", "content": "a.txt b.txt"},
            ]});
        let mut chunks = Vec::new();
        match stream {
            false => chunks.push(c.send("POST", "/api/chat", &follow).unwrap()),
            true => c
                .stream("/api/chat", &follow, &mut |v| {
                    chunks.push(v.clone());
                    Ok(())
                })
                .unwrap(),
        }
        let content: String = chunks.iter().filter_map(|v| v["message"]["content"].as_str()).collect::<Vec<_>>().concat();
        assert_eq!(content, "the cat sat", "stream {stream}");
        let last = chunks.last().unwrap();
        assert!(last["prompt_eval_cached_count"].as_u64().unwrap() > 0, "the tools and the question are reused: {last}");
    }
}

fn status(r: Result<Value, ClientError>) -> (u16, String) {
    match r {
        Err(ClientError::Status(s, m)) => (s, m),
        other => panic!("expected an HTTP error, got {other:?}"),
    }
}

#[test]
fn what_is_not_served_is_refused_by_name() {
    let store = common::store("tools-refuse");
    let mut other = common::manifest();
    other.family = "llama".into();
    store.write_manifest(&ModelName::parse("other:test").unwrap(), &other).unwrap();
    let c = Client::new(&common::serve_with(store, "http://127.0.0.1:9", Arc::new(ToolLoader)));
    let user = json!([{"role": "user", "content": "hi"}]);
    let call = json!({"role": "assistant", "content": null, "tool_calls": [
        {"id": "call_1", "type": "function", "function": {"name": "list_dir", "arguments": "{}"}}]});
    let cases = [
        (json!({"model": "tiny:test", "messages": user, "tools": tools(), "tool_choice": "required"}), "required"),
        (json!({"model": "tiny:test", "messages": user, "tools": tools(),
                "tool_choice": {"type": "function", "function": {"name": "list_dir"}}}), "naming a function"),
        (json!({"model": "tiny:test", "messages": user, "tools": tools(), "parallel_tool_calls": false}), "parallel_tool_calls"),
        (json!({"model": "tiny:test", "messages": user, "tools": [{"type": "web_search"}]}), "web_search"),
        (json!({"model": "tiny:test", "messages": user, "tools": [{"type": "function", "function": {}}]}), "function.name"),
        (json!({"model": "other:test", "messages": user, "tools": tools()}), "no tool format"),
        (json!({"model": "tiny:test", "messages": [user[0], call, {"role": "tool", "content": "x"}]}), "tool_call_id"),
        (json!({"model": "tiny:test", "messages": [user[0], call, {"role": "tool", "tool_call_id": "call_9", "content": "x"}]}), "call_9"),
        (json!({"model": "tiny:test", "messages": [user[0], {"role": "tool", "tool_call_id": "call_1", "content": "x"}]}), "must follow"),
    ];
    for (body, word) in cases {
        let (s, m) = status(c.send("POST", "/v1/chat/completions", &body));
        assert_eq!(s, 400, "{body}: {m}");
        assert!(m.contains(word), "{body}: {m:?} does not say {word:?}");
    }
    // `none` offers no tool: the model is not told of them, and nothing is
    // read for calls.
    let r = c
        .send("POST", "/v1/chat/completions", &json!({"model": "tiny:test", "messages": user, "tools": tools(),
            "tool_choice": "none", "temperature": 0, "max_tokens": 12}))
        .unwrap();
    assert!(r["choices"][0]["message"].get("tool_calls").is_none(), "{r}");
    assert_ne!(r["choices"][0]["finish_reason"], "tool_calls");
}
