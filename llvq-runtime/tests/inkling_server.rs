//! The Inkling engine behind the real server, end to end, with a scripted
//! model on the byte-level tokenizer: the template, the turn parsed by its
//! ids, the calls streamed back as OpenAI's `tool_calls`, and the prefix
//! reuse between requests.
//!
//! The model writes a fixed turn: a thinking part and a call of `list_dir`,
//! its JSON spaced and its keys out of the template's order, until a tool
//! result is in its context; then the words `the cat sat`. Every feed is
//! recorded, so a test reads what the engine read again.

mod common;
#[path = "common/ink.rs"]
mod ink;

use candle_core::{Device, Tensor};
use llvq_llm::inkchat::InkChat;
use llvq_runtime::client::Client;
use llvq_runtime::engine::{Built, Loaded, Loader};
use llvq_runtime::inkling::{InkEngine, TurnLm};
use llvq_runtime::library::{Configs, ModelName};
use llvq_runtime::store::{Manifest, Store};
use serde_json::{json, Value};
use std::io::{BufRead, BufReader};
use std::sync::{Arc, Mutex};

const CALL: &str = concat!(
    "<|message_model|><|content_thinking|>Look first.<|end_message|>",
    "<|message_model|>list_dir<|content_invoke_tool_json|>",
    r#"{"name": "list_dir", "args": {"path": ".", "all": true}}"#,
    "<|end_message|><|content_model_end_sampling|>",
);
const WORDS: &str = "<|message_model|><|content_text|>the cat sat<|end_message|><|content_model_end_sampling|>";

struct Script {
    ctx: Vec<u32>,
    feeds: Arc<Mutex<Vec<Vec<u32>>>>,
    tool: u32,
    call: Vec<u32>,
    words: Vec<u32>,
}

impl Script {
    /// The next id of the turn the context calls for: the longest start of
    /// it the context ends with, one more id.
    fn next(&self) -> u32 {
        let ans = match self.ctx.contains(&self.tool) {
            true => &self.words,
            false => &self.call,
        };
        let i = (1..ans.len()).rev().find(|&i| self.ctx.ends_with(&ans[..i])).unwrap_or(0);
        ans[i]
    }
}

impl TurnLm for Script {
    type Mark = usize;

    fn feed(&mut self, ids: &[u32]) -> anyhow::Result<Tensor> {
        self.feeds.lock().unwrap().push(ids.to_vec());
        self.ctx.extend_from_slice(ids);
        let mut v = vec![0f32; ink::VOCAB];
        v[self.next() as usize] = 1.0;
        Ok(Tensor::from_vec(v, (1, ink::VOCAB), &Device::Cpu)?)
    }

    fn held(&self) -> usize {
        self.ctx.len()
    }

    fn mark(&self) -> usize {
        self.ctx.len()
    }

    fn rollback(&mut self, m: &usize) -> anyhow::Result<()> {
        anyhow::ensure!(*m <= self.ctx.len(), "a mark past the cache");
        self.ctx.truncate(*m);
        Ok(())
    }

    fn reset(&mut self) {
        self.ctx.clear();
    }
}

struct ScriptLoader {
    feeds: Arc<Mutex<Vec<Vec<u32>>>>,
}

impl Loader for ScriptLoader {
    fn load(&self, _: &ModelName, _: &Manifest, _: &Store) -> anyhow::Result<Loaded> {
        anyhow::bail!("not a qwen3")
    }

    fn build(&self, _: &ModelName, _: &Manifest, _: &Store) -> anyhow::Result<Built> {
        let chat = InkChat::new(ink::byte_tokenizer())?;
        let enc = |s: &str| chat.tokenizer().encode(s, false).unwrap().get_ids().to_vec();
        let tool = chat.tokenizer().token_to_id(llvq_llm::inkchat::mark::TOOL).unwrap();
        let lm = Script { ctx: Vec::new(), feeds: self.feeds.clone(), tool, call: enc(CALL), words: enc(WORDS) };
        Ok(Built::Inkling(InkEngine::new(lm, chat, 4096, "script", None)))
    }
}

fn serve(tag: &str) -> (Client, Arc<Mutex<Vec<Vec<u32>>>>) {
    let dir = std::env::temp_dir().join(format!("llvq-runtime-test-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let store = Store::at(dir);
    let m = Manifest {
        schema: 1,
        sha256: "1".repeat(64),
        size: 1,
        repo: "local".into(),
        revision: "local".into(),
        file: "inkling.json".into(),
        family: "inkling2".into(),
        parameter_size: String::new(),
        bits_per_param: String::new(),
        configs: Configs { cuda: None, metal: None, cpu: None, cuda_batch: None, cuda_eager: None },
        modified_at: 1_762_387_200,
    };
    store.write_manifest(&ModelName::parse("ink:test").unwrap(), &m).unwrap();
    let feeds = Arc::new(Mutex::new(Vec::new()));
    let host = common::serve_with(Arc::new(store), "http://127.0.0.1:9", Arc::new(ScriptLoader { feeds: feeds.clone() }));
    (Client::new(&host), feeds)
}

fn tools() -> Value {
    json!([{"type": "function", "function": {"name": "list_dir", "description": "List a directory.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}])
}

fn text(ids: &[u32]) -> String {
    ink::byte_tokenizer().decode(ids, false).unwrap()
}

/// The feeds since `from`, as text. The engine closes a whole answer in the
/// cache after the client has it, so the end of sampling of the answer
/// before may come first, or may have been fed before `from`: it is dropped.
fn fed(feeds: &Arc<Mutex<Vec<Vec<u32>>>>, from: usize) -> Vec<String> {
    let mut v: Vec<String> = feeds.lock().unwrap()[from..].iter().map(|f| text(f)).collect();
    if v.first().is_some_and(|f| f == "<|content_model_end_sampling|>") {
        v.remove(0);
    }
    v
}

const ARGS: &str = r#"{"path": ".", "all": true}"#;

#[test]
fn a_tool_round_trip_reads_only_what_is_new() {
    let (c, feeds) = serve("ink-oai");
    let system = json!({"role": "system", "content": "You are a coding agent."});
    let user = json!({"role": "user", "content": "list the files"});
    let r = c
        .send("POST", "/v1/chat/completions", &json!({"model": "ink:test", "messages": [system, user], "tools": tools()}))
        .unwrap();
    let choice = &r["choices"][0];
    assert_eq!(choice["finish_reason"], "tool_calls", "{r}");
    assert_eq!(choice["message"]["content"], Value::Null, "{r}");
    assert_eq!(choice["message"]["reasoning"], "Look first.", "{r}");
    let call = &choice["message"]["tool_calls"][0];
    assert_eq!(call["function"]["name"], "list_dir");
    assert_eq!(call["function"]["arguments"], ARGS);
    // The prompt in one piece, with the effort `none` by default; then the
    // answer one id at a time.
    let first = fed(&feeds, 0);
    assert!(first[0].starts_with("<|message_system|>tool_declare<|content_xml|>"), "{}", first[0]);
    assert!(first[0].contains("Thinking effort level: 0<|end_message|>"), "{}", first[0]);
    assert!(first[0].ends_with("<|message_user|><|content_text|>list the files<|end_message|>"), "{}", first[0]);
    let n = feeds.lock().unwrap().len();

    let id = call["id"].as_str().unwrap();
    let echo = json!({"role": "assistant", "content": null, "tool_calls": [call.clone()]});
    let result = json!({"role": "tool", "tool_call_id": id, "content": "a.txt b.txt"});
    let body = json!({"model": "ink:test", "tools": tools(), "messages": [system, user, echo, result]});
    let r = c.send("POST", "/v1/chat/completions", &body).unwrap();
    let choice = &r["choices"][0];
    assert_eq!(choice["finish_reason"], "stop", "{r}");
    assert_eq!(choice["message"]["content"], "the cat sat");
    assert!(choice["message"].get("tool_calls").is_none(), "{r}");
    // Only the tool result was read: the system turn, the tools, the user
    // message and the model's own answer stayed in the cache.
    let second = fed(&feeds, n);
    assert_eq!(second[0], "<|message_tool|>list_dir<|content_text|>a.txt b.txt<|end_message|>");
    let n = feeds.lock().unwrap().len();

    // An earlier message changed: everything is read again.
    let user2 = json!({"role": "user", "content": "list the files, please"});
    let body = json!({"model": "ink:test", "tools": tools(), "messages": [system, user2, echo, result]});
    c.send("POST", "/v1/chat/completions", &body).unwrap();
    let third = fed(&feeds, n);
    assert!(third[0].starts_with("<|message_system|>tool_declare"), "{}", third[0]);
    // cut at the start and at the end of the assistant turn
    assert!(third[1].starts_with("<|message_model|>list_dir<|content_invoke_tool_json|>"), "{}", third[1]);
    assert!(third[2].starts_with("<|message_tool|>"), "{}", third[2]);
}

#[test]
fn a_streamed_call_comes_whole_in_one_chunk() {
    let (c, _) = serve("ink-sse");
    let body = json!({"model": "ink:test", "messages": [{"role": "user", "content": "list"}], "tools": tools(),
        "stream": true, "stream_options": {"include_usage": true}, "reasoning_effort": "low"});
    let r = ureq::post(&format!("{}/v1/chat/completions", c.base())).send_string(&body.to_string()).unwrap();
    let (mut chunks, mut done) = (Vec::<Value>::new(), false);
    for line in BufReader::new(r.into_reader()).lines() {
        let line = line.unwrap();
        let Some(d) = line.strip_prefix("data: ") else { continue };
        if d == "[DONE]" {
            done = true;
            continue;
        }
        chunks.push(serde_json::from_str(d).unwrap());
    }
    assert!(done);
    let calls: Vec<&Value> = chunks.iter().filter_map(|c| c["choices"][0]["delta"].get("tool_calls")).collect();
    assert_eq!(calls.len(), 1, "{chunks:?}");
    assert_eq!(calls[0][0]["function"]["arguments"], ARGS);
    assert_eq!(calls[0][0]["index"], 0);
    let reasoning: String =
        chunks.iter().filter_map(|c| c["choices"][0]["delta"]["reasoning"].as_str()).collect();
    assert_eq!(reasoning, "Look first.");
    let finish: Vec<&Value> =
        chunks.iter().map(|c| &c["choices"][0]["finish_reason"]).filter(|f| !f.is_null()).collect();
    assert_eq!(finish, vec![&json!("tool_calls")]);
    let usage = chunks.last().unwrap();
    assert!(usage["usage"]["prompt_tokens"].as_u64().unwrap() > 0, "{usage}");
}

#[test]
fn refusals_come_back_by_name() {
    let (c, _) = serve("ink-refuse");
    let body = json!({"model": "ink:test", "messages": [{"role": "user", "content": "hi"}], "reasoning_effort": "turbo"});
    let e = c.send("POST", "/v1/chat/completions", &body).unwrap_err();
    assert!(format!("{e:?}").contains("Unknown reasoning_effort: turbo"), "{e:?}");
    let echo = json!({"role": "assistant", "content": null, "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "list_dir", "arguments": "[1]"}}]});
    let body = json!({"model": "ink:test", "tools": tools(), "messages": [{"role": "user", "content": "hi"}, echo,
        {"role": "tool", "tool_call_id": "c1", "content": "x"}]});
    let e = c.send("POST", "/v1/chat/completions", &body).unwrap_err();
    assert!(format!("{e:?}").contains("object"), "{e:?}");
    let e = c.send("POST", "/api/generate", &json!({"model": "ink:test", "prompt": "x", "raw": true, "stream": false})).unwrap_err();
    assert!(format!("{e:?}").contains("raw"), "{e:?}");
}
