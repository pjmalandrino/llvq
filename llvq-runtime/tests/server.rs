//! The routes, end to end: the real server, the real engine and `Session`, a
//! tiny deterministic Qwen3 on the CPU (`common`). The gates of stages R2 and
//! R3 of `docs/plan-runtime.md` that need no card: a greedy chat streams the
//! ids a direct `Session` run gives, a stateless second turn reads only its new
//! tokens and answers as the stateful run, a client that leaves stops the
//! engine, and every refusal answers with its status.

mod common;

use llvq_llm::chatfmt::{open_assistant, turn, Marks};
use llvq_llm::repl::Sampler;
use llvq_llm::served_decode::Decoder;
use llvq_llm::session::{Flow, Generate, Session};
use llvq_runtime::client::{Client, ClientError};
use serde_json::{json, Value};
use std::io::{BufRead, BufReader};
use std::time::{Duration, Instant};

/// What a direct run gives: the decoded answer and the reasoning, routed by
/// the think tokens as the engine routes them.
fn routed(said: &[u32]) -> (String, String) {
    let tok = common::tokenizer();
    let (mut content, mut thinking, mut inside) = (Vec::new(), Vec::new(), false);
    for &t in said {
        match t {
            3 => inside = true,
            4 => inside = false,
            _ if inside => thinking.push(t),
            _ => content.push(t),
        }
    }
    (tok.decode(&content, false).unwrap(), tok.decode(&thinking, false).unwrap())
}

/// Greedy turns on a `Session`, stateful, as `bin/chat` runs them: the ids of
/// each answer.
fn reference(turns: &[&str], max_new: usize) -> Vec<Vec<u32>> {
    let qwen = common::model();
    let tok = common::tokenizer();
    let marks = Marks::resolve(&tok).unwrap();
    let mut s = Session::new(&qwen);
    let mut dec = Decoder::eager();
    let mut out = Vec::new();
    for t in turns {
        let mut ids = turn(&tok, &marks, "user", t).unwrap();
        ids.extend(open_assistant(&tok, &marks, false).unwrap());
        let h = s.feed(&qwen, &ids).unwrap();
        let mut sampler = Sampler { temp: 0.0, top_p: 0.9, state: 0 };
        let a = s
            .answer(&qwen, &mut dec, h, Generate { sampler: &mut sampler, max_new, stops: &marks.stops() }, &mut |_| {
                Flow::Continue
            })
            .unwrap();
        // A length cut leaves the last token unfed; the conversation holds it.
        let mut tail: Vec<u32> = Vec::new();
        if a.finish == llvq_llm::session::Finish::Length {
            tail.extend(a.said.last().copied());
        }
        tail.extend([marks.im_end, marks.nl]);
        s.feed(&qwen, &tail).unwrap();
        out.push(a.said);
    }
    out
}

/// A streamed `/api/chat`: the concatenated content, reasoning, and the last
/// chunk.
fn chat(c: &Client, body: Value) -> (String, String, Value) {
    let (mut content, mut thinking, mut last) = (String::new(), String::new(), Value::Null);
    c.stream("/api/chat", &body, &mut |v| {
        content.push_str(v["message"]["content"].as_str().unwrap_or(""));
        thinking.push_str(v["message"]["thinking"].as_str().unwrap_or(""));
        last = v.clone();
        Ok(())
    })
    .unwrap();
    (content, thinking, last)
}

fn status(r: Result<Value, ClientError>) -> (u16, String) {
    match r {
        Err(ClientError::Status(s, m)) => (s, m),
        other => panic!("expected an HTTP error, got {other:?}"),
    }
}

#[test]
fn a_greedy_chat_streams_the_ids_a_direct_session_gives() {
    let host = common::serve(common::store("greedy"), "http://127.0.0.1:9");
    let c = Client::new(&host);
    let want = reference(&["hello how are you"], 40);
    let (content, thinking, last) = chat(
        &c,
        json!({"model": "tiny:test", "messages": [{"role": "user", "content": "hello how are you"}],
               "options": {"temperature": 0, "num_predict": 40}}),
    );
    assert_eq!((content, thinking), routed(&want[0]));
    assert_eq!(last["done"], true);
    assert_eq!(last["done_reason"], "length", "the tiny model never stops on its own");
    assert_eq!(last["eval_count"], 40);
    assert_eq!(last["prompt_eval_cached_count"], 0);
    assert!(last["created_at"].as_str().unwrap().ends_with('Z'));
}

#[test]
fn a_stateless_second_turn_reads_only_its_new_tokens_and_answers_as_the_stateful_run() {
    let host = common::serve(common::store("stateless"), "http://127.0.0.1:9");
    let c = Client::new(&host);
    let want = reference(&["hello world", "why the cat"], 12);
    let opts = json!({"temperature": 0, "num_predict": 12});
    let (a1, _, _) =
        chat(&c, json!({"model": "tiny:test", "messages": [{"role": "user", "content": "hello world"}], "options": opts}));
    assert_eq!(a1, routed(&want[0]).0);
    let history = json!([
        {"role": "user", "content": "hello world"},
        {"role": "assistant", "content": a1},
        {"role": "user", "content": "why the cat"},
    ]);
    let (a2, _, last) = chat(&c, json!({"model": "tiny:test", "messages": history, "options": opts}));
    assert_eq!(a2, routed(&want[1]).0, "the replayed history must answer as the stateful conversation");
    // New: the user turn `<|im_start|>user\nwhy the cat<|im_end|>\n` (8 ids)
    // and the opening `<|im_start|>assistant\n<think>\n\n</think>\n\n` (7).
    let (n, cached) = (last["prompt_eval_count"].as_u64().unwrap(), last["prompt_eval_cached_count"].as_u64().unwrap());
    assert_eq!(n - cached, 15, "turn two re-read {} tokens of history", (n - cached).saturating_sub(15));
    // A history that changes an earlier message is re-read from there.
    let edited = json!([
        {"role": "user", "content": "hello world"},
        {"role": "assistant", "content": "yes"},
        {"role": "user", "content": "why the cat"},
    ]);
    let (_, _, last) = chat(&c, json!({"model": "tiny:test", "messages": edited, "options": opts}));
    let cached = last["prompt_eval_cached_count"].as_u64().unwrap();
    // Kept: the first user turn (7 ids) and the assistant's opening
    // `<|im_start|>assistant\n` (3), where the edited text starts to differ.
    assert_eq!(cached, 10, "an edit must cut the reuse back to the edited message");
}

/// A client that leaves a streamed answer after its first line: the engine
/// stops the answer, so the conversation cache never holds it. The engine
/// would end the answer, and the cache hold it, if it ignored the closed
/// channel; the test only waits for the engine to be idle.
#[test]
fn a_client_that_leaves_stops_the_engine() {
    let host = common::serve(common::store("cancel"), "http://127.0.0.1:9");
    let c = Client::new(&host);
    let long = 1000;
    let full = whole_answer(&c, long);
    let resp = ureq::AgentBuilder::new().build().post(&format!("http://{host}/api/chat")).send_json(ask(long, true)).unwrap();
    let mut lines = BufReader::new(resp.into_reader()).lines();
    lines.next().unwrap().unwrap();
    drop(lines);
    wait_idle(&c, "the engine after its client left");
    assert!(!holds_the_answer(&c, &full), "the engine answered to the end for a client that left");
}

#[test]
fn one_object_when_stream_is_false_and_generate_takes_raw_and_templated_prompts() {
    let host = common::serve(common::store("nostream"), "http://127.0.0.1:9");
    let c = Client::new(&host);
    let want = reference(&["the dog sat"], 9);
    let v = c
        .send(
            "POST",
            "/api/chat",
            &json!({"model": "tiny:test", "messages": [{"role": "user", "content": "the dog sat"}],
                    "stream": false, "options": {"temperature": 0, "num_predict": 9}}),
        )
        .unwrap();
    assert_eq!(v["message"]["content"].as_str().unwrap(), routed(&want[0]).0);
    assert_eq!(v["done"], true);

    // The template path of /api/generate is the chat's user turn.
    let g = c
        .send(
            "POST",
            "/api/generate",
            &json!({"model": "tiny:test", "prompt": "the dog sat", "stream": false,
                    "options": {"temperature": 0, "num_predict": 9}}),
        )
        .unwrap();
    assert_eq!(g["response"].as_str().unwrap(), routed(&want[0]).0);

    // Raw: no template, so a different prompt and a different count.
    let r = c
        .send(
            "POST",
            "/api/generate",
            &json!({"model": "tiny:test", "prompt": "the dog sat", "raw": true, "stream": false,
                    "options": {"temperature": 0, "num_predict": 5, "top_k": 3}}),
        )
        .unwrap();
    assert_eq!(r["prompt_eval_count"], 3);
    assert_eq!(r["eval_count"], 5);
    assert_eq!(r["warnings"][0].as_str().unwrap(), "option top_k is not implemented by llvq and was ignored");
    // A system turn has no place in a raw prompt: it is not applied, and the
    // answer says so.
    let r = c
        .send(
            "POST",
            "/api/generate",
            &json!({"model": "tiny:test", "prompt": "the dog sat", "raw": true, "system": "be brief", "stream": false,
                    "options": {"temperature": 0, "num_predict": 1}}),
        )
        .unwrap();
    assert!(r["warnings"][0].as_str().unwrap().contains("system"), "{r}");

    // An empty prompt loads the model and answers nothing.
    let l = c.send("POST", "/api/generate", &json!({"model": "tiny:test", "stream": false})).unwrap();
    assert_eq!(l["done_reason"], "load");
}

#[test]
fn a_stop_string_cuts_the_answer_where_it_starts() {
    let host = common::serve(common::store("stop"), "http://127.0.0.1:9");
    let c = Client::new(&host);
    let full = routed(&reference(&["are you fine"], 30)[0]).0;
    let words: Vec<&str> = full.split(' ').filter(|w| !w.is_empty()).collect();
    assert!(words.len() >= 4, "the reference answer is too short to cut: {full:?}");
    let stop = words[3];
    let at = full.find(stop).unwrap();
    let (content, _, last) = chat(
        &c,
        json!({"model": "tiny:test", "messages": [{"role": "user", "content": "are you fine"}],
               "options": {"temperature": 0, "num_predict": 30, "stop": [stop]}}),
    );
    assert_eq!(content, full[..at]);
    assert_eq!(last["done_reason"], "stop");
}

#[test]
fn refusals_answer_with_their_status_and_a_reason() {
    let host = common::serve(common::store("refuse"), "http://127.0.0.1:9");
    let c = Client::new(&host);
    let user = json!([{"role": "user", "content": "hello"}]);
    let cases = [
        (json!({"model": "nope:1", "messages": user}), 404, "not found"),
        (json!({"model": "tiny:test", "messages": user, "options": {"num_ctx": 5000}}), 400, "num_ctx"),
        (json!({"model": "tiny:test", "messages": user, "options": {"num_ctx": 10}}), 400, "no room"),
        (json!({"model": "tiny:test", "messages": user, "options": {"temperature": -1}}), 400, "temperature"),
        (json!({"model": "tiny:test", "messages": user, "tools": [{"type": "function"}]}), 400, "tools"),
        (json!({"model": "tiny:test", "messages": [{"role": "tool", "content": "x"}]}), 400, "tool"),
        (json!({"model": "tiny:test", "messages": [{"role": "user", "content": "x", "images": ["AA=="]}]}), 400, "images"),
        (json!({"model": "tiny:test", "messages": user, "format": "json"}), 400, "format"),
        (json!({"model": "../x", "messages": user}), 400, "not a model name"),
        (json!({"messages": user}), 400, "model"),
    ];
    for (body, code, word) in cases {
        let (s, m) = status(c.send("POST", "/api/chat", &body));
        assert_eq!(s, code, "{body}: {m}");
        assert!(m.contains(word), "{body}: {m:?} does not say {word:?}");
    }
    for (method, path) in [
        ("POST", "/api/embed"),
        ("POST", "/api/embeddings"),
        ("POST", "/api/create"),
        ("POST", "/api/push"),
        ("POST", "/api/blobs/sha256:00"),
        ("POST", "/v1/embeddings"),
        ("POST", "/v1/responses"),
    ] {
        let (s, m) = status(c.send(method, path, &json!({})));
        assert_eq!(s, 501, "{path}");
        assert!(m.contains("not implemented by llvq"), "{path}: {m}");
    }
}

#[test]
fn a_page_from_another_site_cannot_drive_the_server() {
    let host = common::serve(common::store("origin"), "http://127.0.0.1:9");
    // One token: the tiny model never stops on its own, and the test reads statuses only.
    let body = json!({"model": "tiny:test", "messages": [{"role": "user", "content": "hello"}], "stream": false,
                      "options": {"num_predict": 1}});
    let post = |origin: Option<&str>, host_header: Option<&str>| {
        let mut r = ureq::post(&format!("http://{host}/api/chat"));
        if let Some(o) = origin {
            r = r.set("Origin", o);
        }
        if let Some(h) = host_header {
            r = r.set("Host", h);
        }
        match r.send_string(&body.to_string()) {
            Ok(resp) => resp.status(),
            Err(ureq::Error::Status(code, _)) => code,
            Err(e) => panic!("{e}"),
        }
    };
    assert_eq!(post(None, None), 200, "curl and the CLI send no Origin");
    assert_eq!(post(Some(&format!("http://{host}")), None), 200, "the chat page is same-origin");
    assert_eq!(post(Some("https://evil.example"), None), 403, "a cross-site text/plain POST");
    assert_eq!(post(None, Some("evil.example:11435")), 403, "a DNS-rebound name");
    let pull = ureq::post(&format!("http://{host}/api/pull"))
        .set("Origin", "https://evil.example")
        .set("Content-Type", "text/plain")
        .send_string(r#"{"model":"qwen3:4b"}"#);
    assert!(matches!(pull, Err(ureq::Error::Status(403, _))), "a page cannot start a download");
}

#[test]
fn the_store_routes_list_show_copy_delete_and_ps_follows_loads() {
    let host = common::serve(common::store("routes"), "http://127.0.0.1:9");
    let c = Client::new(&host);
    let v = c.get("/api/version").unwrap();
    assert_eq!(v["version"], env!("CARGO_PKG_VERSION"));
    let names = |c: &Client| -> Vec<String> {
        c.get("/api/tags").unwrap()["models"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m["name"].as_str().unwrap().to_string())
            .collect()
    };
    assert_eq!(names(&c), ["tiny:test"]);
    let s = c.send("POST", "/api/show", &json!({"model": "tiny:test"})).unwrap();
    assert_eq!(s["details"]["format"], "llvq");
    assert_eq!(s["llvq"]["repo"], "test/tiny");
    assert_eq!(s["model_info"]["general.architecture"], "qwen3");
    assert!(s["template"].as_str().unwrap().contains("<|im_start|>"), "{s}");
    assert_eq!(s["capabilities"], json!(["completion", "thinking", "tools"]));
    c.send("POST", "/api/copy", &json!({"source": "tiny:test", "destination": "tiny:copy"})).unwrap();
    assert_eq!(names(&c), ["tiny:copy", "tiny:test"]);
    c.send("DELETE", "/api/delete", &json!({"model": "tiny:copy"})).unwrap();
    assert_eq!(status(c.send("DELETE", "/api/delete", &json!({"model": "tiny:copy"}))).0, 404);
    assert_eq!(status(c.send("POST", "/api/copy", &json!({"source": "x:y", "destination": "z:w"}))).0, 404);

    assert!(c.get("/api/ps").unwrap()["models"].as_array().unwrap().is_empty());
    c.send("POST", "/api/generate", &json!({"model": "tiny:test", "stream": false})).unwrap();
    let ps = c.get("/api/ps").unwrap();
    let m = &ps["models"][0];
    assert_eq!(m["name"], "tiny:test");
    assert_eq!(m["llvq"]["backend"], "dense-cpu");
    assert_eq!(m["llvq"]["state"], "ready");
    assert_eq!(m["context_length"], 4096);
    assert_eq!(m["size_vram"], 0);
    // `llvq stop`: keep_alive 0 unloads.
    let u = c.send("POST", "/api/generate", &json!({"model": "tiny:test", "keep_alive": 0, "stream": false})).unwrap();
    assert_eq!(u["done_reason"], "unload");
    // The engine leaves at once when idle. One still marked busy by the load that just answered is retired
    // instead, and the reaper drops it within its second: the list empties, not always on the first read.
    let t = Instant::now();
    while !c.get("/api/ps").unwrap()["models"].as_array().unwrap().is_empty() {
        assert!(t.elapsed() < Duration::from_secs(5), "the model outlived its unload");
        std::thread::sleep(Duration::from_millis(50));
    }
    // An expired keep-alive unloads within the reaper's second.
    c.send("POST", "/api/generate", &json!({"model": "tiny:test", "keep_alive": "100ms", "stream": false})).unwrap();
    let t = Instant::now();
    while !c.get("/api/ps").unwrap()["models"].as_array().unwrap().is_empty() {
        assert!(t.elapsed() < Duration::from_secs(5), "the model outlived its keep-alive");
        std::thread::sleep(Duration::from_millis(50));
    }
}

/// The SSE lines of a streamed OpenAI route, without the `data: ` prefix.
fn sse(host: &str, path: &str, body: Value) -> Vec<String> {
    let resp = ureq::post(&format!("http://{host}{path}")).send_json(body).unwrap();
    assert_eq!(resp.header("content-type"), Some("text/event-stream"));
    BufReader::new(resp.into_reader())
        .lines()
        .map(|l| l.unwrap())
        .filter(|l| !l.is_empty())
        .map(|l| l.strip_prefix("data: ").unwrap_or_else(|| panic!("not an SSE data line: {l:?}")).to_string())
        .collect()
}

#[test]
fn the_openai_routes_answer_as_the_ollama_ones_in_openais_shapes() {
    let host = common::serve(common::store("openai"), "http://127.0.0.1:9");
    let c = Client::new(&host);
    let want = routed(&reference(&["hello how are you"], 20)[0]).0;
    let body = json!({"model": "tiny:test", "messages": [{"role": "user", "content": [{"type": "text", "text": "hello how are you"}]}],
                      "temperature": 0, "max_tokens": 20, "frequency_penalty": 0.5});
    let v = c.send("POST", "/v1/chat/completions", &body).unwrap();
    assert_eq!(v["object"], "chat.completion");
    assert_eq!(v["choices"][0]["message"]["content"].as_str().unwrap(), want);
    assert_eq!(v["choices"][0]["finish_reason"], "length");
    assert_eq!(v["usage"]["completion_tokens"], 20);
    assert!(v["warnings"][0].as_str().unwrap().contains("frequency_penalty"));

    let mut sbody = body.clone();
    sbody["stream"] = json!(true);
    sbody["stream_options"] = json!({"include_usage": true});
    let lines = sse(&host, "/v1/chat/completions", sbody);
    assert_eq!(lines.last().unwrap(), "[DONE]");
    let chunks: Vec<Value> = lines[..lines.len() - 1].iter().map(|l| serde_json::from_str(l).unwrap()).collect();
    assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
    let text: String =
        chunks.iter().filter_map(|c| c["choices"].get(0)?["delta"]["content"].as_str().map(str::to_string)).collect();
    assert_eq!(text, want);
    let fin = chunks.iter().find(|c| c["choices"].get(0).is_some_and(|x| !x["finish_reason"].is_null())).unwrap();    assert_eq!(fin["choices"][0]["finish_reason"], "length");
    // The ignored field is said in the stream too, on the chunk that ends it.
    assert!(fin["warnings"][0].as_str().unwrap().contains("frequency_penalty"), "{fin}");
    assert_eq!(chunks.last().unwrap()["usage"]["completion_tokens"], 20);
    assert!(chunks.iter().all(|c| c["object"] == "chat.completion.chunk"));

    let t = c
        .send("POST", "/v1/completions", &json!({"model": "tiny:test", "prompt": "hello how are you", "temperature": 0, "max_tokens": 20}))
        .unwrap();
    assert_eq!(t["object"], "text_completion");
    assert_eq!(t["choices"][0]["text"].as_str().unwrap(), want);

    let m = c.get("/v1/models").unwrap();
    assert_eq!(m["data"][0]["id"], "tiny:test");
    assert_eq!(c.get("/v1/models/tiny:test").unwrap()["object"], "model");
    let (s, _) = status(c.get("/v1/models/nope:1"));
    assert_eq!(s, 404);
    let (s, m) = status(c.send("POST", "/v1/chat/completions", &json!({"model": "tiny:test", "messages": [{"role": "user", "content": "x"}], "n": 2})));
    assert_eq!(s, 400, "{m}");

    // `logprobs` asked for is not served, and the answer says so; false is
    // what llvq does anyway, and needs no warning.
    let ask = |logprobs: bool| {
        c.send(
            "POST",
            "/v1/chat/completions",
            &json!({"model": "tiny:test", "messages": [{"role": "user", "content": "hello"}],
                    "temperature": 0, "max_tokens": 2, "logprobs": logprobs}),
        )
        .unwrap()
    };
    let lp = ask(true);
    assert!(lp["warnings"][0].as_str().unwrap().contains("logprobs"), "{lp}");
    let lp = ask(false);
    assert!(lp.get("warnings").is_none(), "{lp}");
}

#[test]
fn the_chat_page_is_served_whole_with_the_library_written_in() {
    let host = common::serve(common::store("page"), "http://127.0.0.1:9");
    let resp = ureq::get(&format!("http://{host}/")).call().unwrap();
    assert_eq!(resp.header("content-type"), Some("text/html; charset=utf-8"));
    let html = resp.into_string().unwrap();
    assert!(html.starts_with("<!doctype html>"));
    assert!(!html.contains("__LLVQ_LIBRARY__"), "the library placeholder was left in the page");
    assert!(html.contains(r#""name":"qwen3:4b""#) && html.contains(r#""size":1418224685"#));
    // Self-contained: nothing fetched from another host. The SVG namespace of
    // the inline icon is a name, not a fetch.
    let fetched = html.replace("http://www.w3.org/2000/svg", "");
    for outside in ["http://", "https://", "//cdn", "<link rel=\"stylesheet", "@import", "url(http"] {
        assert!(!fetched.contains(outside), "the page reaches outside the server: {outside}");
    }
    // The routes it calls are served.
    for route in ["/api/tags", "/api/chat", "/api/pull"] {
        assert!(html.contains(&format!("\"{route}\"")), "the page does not call {route}");
    }
}

// ---------------------------------------------------------------------------
// The audit of 2026-11 (`docs/runtime-audit-2026-11.md`): each test below
// failed on the code it was written against.
// ---------------------------------------------------------------------------

const HUB: &str = "http://127.0.0.1:9";

/// A greedy answer of `n` tokens to "one two": the tiny model never stops on
/// its own, so `n` is the answer's length.
fn ask(n: u64, stream: bool) -> Value {
    json!({"model": "tiny:test", "messages": [{"role": "user", "content": "one two"}],
           "options": {"temperature": 0, "num_predict": n}, "stream": stream})
}

/// What bounds a wait in the tests below. A passing test never reaches it:
/// each test waits on the condition it needs, and the bound only ends a
/// failure.
const BOUND: Duration = Duration::from_secs(60);

/// Wait until `cond` holds; panic naming `what` past [`BOUND`].
fn wait_for(what: &str, mut cond: impl FnMut() -> bool) {
    let t = Instant::now();
    while !cond() {
        assert!(t.elapsed() < BOUND, "{what}: still not so after {BOUND:?}");
        std::thread::sleep(Duration::from_millis(2));
    }
}

/// The loaded model's `busy` and `queue`, as `/api/ps` shows them.
fn ps_state(c: &Client) -> (bool, u64) {
    let ps = c.get("/api/ps").unwrap();
    let m = &ps["models"][0]["llvq"];
    (m["busy"].as_bool().unwrap_or(false), m["queue"].as_u64().unwrap_or(0))
}

/// Wait until the engine runs nothing and holds nothing queued.
fn wait_idle(c: &Client, what: &str) {
    wait_for(what, || ps_state(c) == (false, 0));
}

/// A request whose client the test controls: written whole on a connection
/// of its own, which closes when the stream returned is dropped. Nothing is
/// read from it.
fn post_raw(host: &str, path: &str, body: &Value) -> std::net::TcpStream {
    use std::io::Write;
    let mut s = std::net::TcpStream::connect(host).unwrap();
    let b = body.to_string();
    write!(s, "POST {path} HTTP/1.1\r\nHost: {host}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{b}", b.len())
        .unwrap();
    s
}

/// Whether the conversation cache holds `content` as the whole answer to
/// "one two": the next turn, sent with that answer in its history, then reads
/// only its own 13 tokens. An answer the engine cut short leaves the cache at
/// the user's message, and the turn reads the answer again.
fn holds_the_answer(c: &Client, content: &str) -> bool {
    let history = json!([
        {"role": "user", "content": "one two"},
        {"role": "assistant", "content": content},
        {"role": "user", "content": "yes"},
    ]);
    let (_, _, last) =
        chat(c, json!({"model": "tiny:test", "messages": history, "options": {"temperature": 0, "num_predict": 1}}));
    let (n, cached) = (last["prompt_eval_count"].as_u64().unwrap(), last["prompt_eval_cached_count"].as_u64().unwrap());
    wait_idle(c, "the engine after the probe of its cache");
    // New: `<|im_start|>user\nyes<|im_end|>\n` (6) and the opening (7).
    n - cached == 13
}

/// The whole greedy answer of `n` tokens to "one two", from the server,
/// which also loads the model. The control: the cache then holds it.
fn whole_answer(c: &Client, n: u64) -> String {
    let v = c.send("POST", "/api/chat", &ask(n, false)).unwrap();
    let full = v["message"]["content"].as_str().unwrap().to_string();
    wait_idle(c, "the engine after the whole answer");
    assert!(holds_the_answer(c, &full), "the control: the cache does not hold a whole answer");
    full
}

fn counting(delay: Duration) -> (std::sync::Arc<std::sync::atomic::AtomicUsize>, std::sync::Arc<common::CountingLoader>) {
    let loads = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    (loads.clone(), std::sync::Arc::new(common::CountingLoader { loads, delay }))
}

#[test]
fn stopping_a_busy_model_keeps_one_copy_until_its_answer_ends() {
    let (loads, loader) = counting(Duration::ZERO);
    let (host, models) = common::serve_models(common::store("stopbusy"), HUB, loader);
    let c = Client::new(&host);
    let resp = ureq::post(&format!("http://{host}/api/chat")).send_json(ask(3000, true)).unwrap();
    let mut lines = BufReader::new(resp.into_reader()).lines();
    lines.next().unwrap().unwrap();
    // `llvq stop` while the model answers.
    let u = c.send("POST", "/api/generate", &json!({"model": "tiny:test", "keep_alive": 0, "stream": false})).unwrap();
    assert_eq!(u["done_reason"], "unload");
    let ps = c.get("/api/ps").unwrap();
    assert_eq!(ps["models"].as_array().unwrap().len(), 1, "the busy model is still in memory and /api/ps hides it: {ps}");
    // A request now waits for that copy instead of loading a second one.
    let h = host.clone();
    let short = std::thread::spawn(move || Client::new(&h).send("POST", "/api/chat", &ask(1, false)));
    wait_for("the request queued behind the busy model", || ps_state(&c).1 == 1);
    drop(lines);
    short.join().unwrap().unwrap();
    assert_eq!(loads.load(std::sync::atomic::Ordering::SeqCst), 1, "a second copy was loaded beside the busy one");
    // That request took the model back: idle, the reaper keeps it. The test
    // runs the reaper itself rather than sleeping past its second.
    wait_idle(&c, "the engine after the request that took it back");
    models.reap(std::time::SystemTime::now());
    assert_eq!(c.get("/api/ps").unwrap()["models"].as_array().unwrap().len(), 1, "the model taken back was unloaded");
}

#[test]
fn a_model_stopped_while_busy_unloads_when_its_answer_ends() {
    let host = common::serve(common::store("stoplater"), HUB);
    let c = Client::new(&host);
    let resp = ureq::post(&format!("http://{host}/api/chat")).send_json(ask(3000, true)).unwrap();
    let mut lines = BufReader::new(resp.into_reader()).lines();
    lines.next().unwrap().unwrap();
    c.send("POST", "/api/generate", &json!({"model": "tiny:test", "keep_alive": 0, "stream": false})).unwrap();
    drop(lines);
    let t = Instant::now();
    while !c.get("/api/ps").unwrap()["models"].as_array().unwrap().is_empty() {
        assert!(t.elapsed() < Duration::from_secs(5), "the stopped model stayed loaded after its answer ended");
        std::thread::sleep(Duration::from_millis(50));
    }
}

#[test]
fn a_request_abandoned_in_the_queue_leaves_the_cache_alone() {
    let host = common::serve(common::store("abandon"), HUB);
    let c = Client::new(&host);
    let n = 1500;
    let opts = |k: u64| json!({"temperature": 0, "num_predict": k});
    let (a1, _, _) =
        chat(&c, json!({"model": "tiny:test", "messages": [{"role": "user", "content": "hello world"}], "options": opts(4)}));
    let h1 = json!([
        {"role": "user", "content": "hello world"},
        {"role": "assistant", "content": a1},
        {"role": "user", "content": "why the cat"},
    ]);
    let (started_tx, started) = std::sync::mpsc::channel();
    let (h, body) = (host.clone(), json!({"model": "tiny:test", "messages": h1.clone(), "options": opts(n)}));
    let long = std::thread::spawn(move || {
        let (mut content, mut last, mut started_tx) = (String::new(), Value::Null, Some(started_tx));
        Client::new(&h)
            .stream("/api/chat", &body, &mut |v| {
                if let Some(tx) = started_tx.take() {
                    let _ = tx.send(());
                }
                content.push_str(v["message"]["content"].as_str().unwrap_or(""));
                last = v.clone();
                Ok(())
            })
            .unwrap();
        (content, last)
    });
    started.recv_timeout(BOUND).expect("the long answer never streamed");
    // Another conversation, queued behind the long answer, then abandoned.
    let gone = post_raw(
        &host,
        "/api/chat",
        &json!({"model": "tiny:test", "messages": [{"role": "user", "content": "the dog sat"}], "stream": false}),
    );
    wait_for("the second request queued behind the long answer", || ps_state(&c).1 == 1);
    drop(gone);
    let (x, last) = long.join().unwrap();
    assert_eq!(last["done_reason"], "length");
    let mut h2 = h1.as_array().unwrap().clone();
    h2.push(json!({"role": "assistant", "content": x}));
    h2.push(json!({"role": "user", "content": "yes"}));
    let (_, _, last) = chat(&c, json!({"model": "tiny:test", "messages": h2, "options": opts(2)}));
    let (n, cached) = (last["prompt_eval_count"].as_u64().unwrap(), last["prompt_eval_cached_count"].as_u64().unwrap());
    // New: `<|im_start|>user\nyes<|im_end|>\n` (6) and the opening (7).
    assert_eq!(n - cached, 13, "the abandoned request cut the cache: {cached} of {n} positions kept");
}

/// A client that leaves a non-streamed answer while the engine runs it: hyper
/// drops the handler, the engine stops, and the cache never holds the answer.
/// The test closes its connection once `/api/ps` shows the request running,
/// and then only waits for the engine to be idle.
#[test]
fn a_client_that_leaves_a_non_streamed_answer_stops_the_engine() {
    let host = common::serve(common::store("cancel-nostream"), HUB);
    let c = Client::new(&host);
    let long = 1000;
    let full = whole_answer(&c, long);
    let conn = post_raw(&host, "/api/chat", &ask(long, false));
    wait_for("the request running", || ps_state(&c) == (true, 0));
    drop(conn);
    wait_idle(&c, "the engine after its client left");
    assert!(!holds_the_answer(&c, &full), "the engine answered to the end for a client that left");
}

/// Which steps of the greedy answer `ids` stream nothing, as the engine's
/// router sees them: the content and the reasoning (between `<think>` (3) and
/// `</think>` (4)), each decoded whole and sent up to its last whole
/// character. With them, the content sent before each step, and whether
/// anything is held back after it.
fn silences(ids: &[u32]) -> (Vec<bool>, Vec<String>, Vec<bool>) {
    use llvq_runtime::chat::{scan, Scan, TextStream};
    let tok = common::tokenizer();
    let (mut content, mut thinking, mut inside) = (TextStream::default(), TextStream::default(), false);
    let (mut silent, mut sent, mut held) = (Vec::new(), Vec::new(), Vec::new());
    for &t in ids {
        sent.push(content.sent.clone());
        match t {
            3 => inside = true,
            4 => inside = false,
            _ if inside => thinking.ids.push(t),
            _ => content.ids.push(t),
        }
        let f = content.full(&tok).unwrap();
        let upto = match scan(&f, &[]) {
            Scan::Safe(n) | Scan::Stop(n) => n,
        };
        let c = content.take(&f, upto);
        let tf = thinking.full(&tok).unwrap();
        let r = thinking.take(&tf, llvq_llm::repl::printable(&tf).len());
        silent.push(c.is_empty() && r.is_empty());
        held.push(content.sent != f || thinking.sent != tf);
    }
    (silent, sent, held)
}

/// A client that leaves while the router has nothing to send: the tiny model
/// answers long stretches of ids the tokenizer has no word for, and during
/// one no `Delta` goes out, so no send can fail. The test finds the longest
/// such stretch in the greedy answer, asks for an answer that ends on its
/// last id, reads every line before it and leaves. An engine that only
/// noticed the client at a send would decode the silence to the answer's
/// end, with nothing left to send, and keep the whole answer in the cache;
/// this one asks the channel during the silence and stops there.
#[test]
fn a_client_that_leaves_during_a_silence_stops_the_engine() {
    let host = common::serve(common::store("cancel-silence"), HUB);
    let c = Client::new(&host);
    let ids = &reference(&["one two"], 3000)[0];
    let (silent, sent, held) = silences(ids);
    let first = silent.iter().position(|s| !s).expect("the answer sends something");
    let (mut start, mut end, mut i) = (0, 0, first);
    while i < ids.len() {
        let j = (i..ids.len()).find(|&j| !silent[j]).unwrap_or(ids.len());
        if j - i > end - start {
            (start, end) = (i, j);
        }
        i = j + 1;
    }
    assert!(end - start >= 50, "the longest silence is {} ids: too short to leave in", end - start);
    assert!(!held[end - 1], "the router holds text at the silence's end: the answer's end would send it");
    let lines_before = silent[..start].iter().filter(|s| !**s).count();
    let resp = ureq::post(&format!("http://{host}/api/chat")).send_json(ask(end as u64, true)).unwrap();
    let mut lines = BufReader::new(resp.into_reader()).lines();
    let mut got = String::new();
    for _ in 0..lines_before {
        let v: Value = serde_json::from_str(&lines.next().unwrap().unwrap()).unwrap();
        assert_eq!(v["done"], false, "the answer ended before its silence");
        got.push_str(v["message"]["content"].as_str().unwrap_or(""));
    }
    assert_eq!(got, sent[start], "the test's router and the engine's disagree before the silence");
    drop(lines);
    eprintln!("left at id {start} of a silence of {} ids, the answer ending on its last", end - start);
    wait_idle(&c, "the engine after its client left");
    assert!(!holds_the_answer(&c, &sent[start]), "the engine decoded a silence to the answer's end for a client that left");
}

#[test]
fn the_first_answer_counts_its_load_once() {
    let (_, loader) = counting(Duration::from_millis(500));
    let host = common::serve_with(common::store("loadtime"), HUB, loader);
    let c = Client::new(&host);
    let v = c.send("POST", "/api/chat", &ask(1, false)).unwrap();
    let (load, total) = (v["load_duration"].as_u64().unwrap(), v["total_duration"].as_u64().unwrap());
    assert!(load >= 500_000_000, "load_duration {load} ns, the load slept 500 ms");
    assert!(total >= load && total < load + 400_000_000, "total_duration {total} ns counts the {load} ns load twice");
}

/// The id of `far` in [`far_tokenizer`]: past the tiny model's 64 rows, so
/// a prompt that holds it fails the prefill at the embedding.
const FAR: u32 = 100;

/// The tiny tokenizer, and one more word whose id the model has no row for.
fn far_tokenizer() -> llvq_llm::tokenizers::Tokenizer {
    let mut vocab = serde_json::Map::new();
    let mut added = Vec::new();
    for (i, s) in common::SPECIAL.iter().enumerate() {
        vocab.insert((*s).into(), i.into());
        added.push(json!({"id": i, "content": s, "single_word": false, "lstrip": false,
            "rstrip": false, "normalized": false, "special": true}));
    }
    vocab.insert("<unk>".into(), common::SPECIAL.len().into());
    for (i, w) in common::WORDS.iter().enumerate() {
        vocab.insert((*w).into(), (common::SPECIAL.len() + 1 + i).into());
    }
    vocab.insert("far".into(), FAR.into());
    let j = json!({
        "version": "1.0", "truncation": null, "padding": null, "added_tokens": added,
        "normalizer": null, "pre_tokenizer": {"type": "WhitespaceSplit"}, "post_processor": null,
        "decoder": null, "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "<unk>"}
    });
    llvq_llm::tokenizers::Tokenizer::from_bytes(serde_json::to_vec(&j).unwrap()).expect("the tokenizer builds")
}

/// The tiny model behind [`far_tokenizer`].
struct FarLoader;

impl llvq_runtime::engine::Loader for FarLoader {
    fn load(
        &self,
        _: &llvq_runtime::library::ModelName,
        _: &llvq_runtime::store::Manifest,
        _: &llvq_runtime::store::Store,
    ) -> anyhow::Result<llvq_runtime::engine::Loaded> {
        Ok(llvq_runtime::engine::Loaded {
            qwen: common::model(),
            tokenizer: far_tokenizer(),
            decoder: Decoder::eager(),
            backend: "dense-cpu".into(),
            config: None,
            lines: Vec::new(),
            batch: None,
        })
    }
}

/// A prefill that fails may leave the caches longer than the ids the
/// session holds (a chain launch after the first, on the card), and every
/// request that kept the whole prefix then failed too. After a failed
/// prefill the next request starts from an empty cache.
#[test]
fn a_failed_prefill_leaves_an_empty_cache_to_the_next_request() {
    let host = common::serve_with(common::store("failed-prefill"), HUB, std::sync::Arc::new(FarLoader));
    let c = Client::new(&host);
    let opts = json!({"temperature": 0, "num_predict": 12});
    let (a1, _, _) =
        chat(&c, json!({"model": "tiny:test", "messages": [{"role": "user", "content": "hello world"}], "options": opts}));
    let history = |text: &str| {
        json!([
            {"role": "user", "content": "hello world"},
            {"role": "assistant", "content": a1},
            {"role": "user", "content": text},
        ])
    };
    let (s, m) = status(c.send(
        "POST",
        "/api/chat",
        &json!({"model": "tiny:test", "messages": history("why far"), "options": opts, "stream": false}),
    ));
    assert_eq!(s, 500, "a prompt past the model's rows was answered: {m}");
    let (_, _, last) = chat(&c, json!({"model": "tiny:test", "messages": history("why the cat"), "options": opts}));
    assert_eq!(last["done"], true);
    assert_eq!(last["prompt_eval_cached_count"], 0, "the request after a failed prefill reused its cache");
}

/// A stop list is bounded: each step scans the answer for every string, on
/// the engine's one thread, so one request with 200,000 of them held every
/// other request for hours.
#[test]
fn a_stop_list_past_its_bounds_is_refused() {
    use llvq_runtime::api::{stop_list, MAX_STOPS, MAX_STOP_BYTES};
    let list = |n: usize, len: usize| Value::from(vec!["a".repeat(len); n]);
    assert_eq!(stop_list(&list(MAX_STOPS, MAX_STOP_BYTES)).unwrap().len(), MAX_STOPS);
    assert!(stop_list(&list(MAX_STOPS + 1, 1)).is_err(), "{} strings", MAX_STOPS + 1);
    assert!(stop_list(&list(1, MAX_STOP_BYTES + 1)).is_err(), "a string of {} bytes", MAX_STOP_BYTES + 1);
    assert!(stop_list(&Value::from("a".repeat(MAX_STOP_BYTES + 1))).is_err(), "one string of {} bytes", MAX_STOP_BYTES + 1);
    let host = common::serve(common::store("stop-bound"), HUB);
    let c = Client::new(&host);
    let user = json!([{"role": "user", "content": "hello"}]);
    let many = list(MAX_STOPS + 1, 2);
    for (path, body) in [
        ("/api/chat", json!({"model": "tiny:test", "messages": user, "options": {"stop": many}})),
        ("/v1/chat/completions", json!({"model": "tiny:test", "messages": user, "stop": many})),
    ] {
        let (s, m) = status(c.send("POST", path, &body));
        assert_eq!(s, 400, "{path}: {m}");
        assert!(m.contains("stop"), "{path}: {m:?} does not say stop");
    }
    // Within the bounds, a request with stop strings is answered.
    let ok = json!({"model": "tiny:test", "messages": user, "stream": false,
                    "options": {"temperature": 0, "num_predict": 2, "stop": list(MAX_STOPS, MAX_STOP_BYTES)}});
    c.send("POST", "/api/chat", &ok).unwrap();
}
