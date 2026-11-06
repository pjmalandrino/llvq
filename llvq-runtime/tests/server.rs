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
        s.feed(&qwen, &[marks.im_end, marks.nl]).unwrap();
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

#[test]
fn a_client_that_leaves_stops_the_engine() {
    let host = common::serve(common::store("cancel"), "http://127.0.0.1:9");
    let c = Client::new(&host);
    let ask = |n: u64| {
        json!({"model": "tiny:test", "messages": [{"role": "user", "content": "one two"}],
               "options": {"temperature": 0, "num_predict": n}, "stream": false})
    };
    // Warm the model, then time 200 tokens to price the long answer.
    c.send("POST", "/api/chat", &ask(1)).unwrap();
    let t = Instant::now();
    c.send("POST", "/api/chat", &ask(200)).unwrap();
    let per_token = t.elapsed() / 200;
    let long = 3000u32;

    let agent = ureq::AgentBuilder::new().build();
    let mut body = ask(long as u64);
    body["stream"] = json!(true);
    let resp = agent.post(&format!("http://{host}/api/chat")).send_json(body).unwrap();
    let mut lines = BufReader::new(resp.into_reader()).lines();
    lines.next().unwrap().unwrap();
    let left = Instant::now();
    drop(lines);

    // The engine is free well before the whole answer would have ended.
    let t = Instant::now();
    c.send("POST", "/api/chat", &ask(1)).unwrap();
    let waited = t.elapsed();
    let whole = per_token * long;
    assert!(
        waited < whole / 4,
        "the next request waited {waited:?}; the abandoned answer would have run {whole:?} \
         ({per_token:?} a token), measured from {:?} after leaving",
        left.elapsed()
    );
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
    assert!(c.get("/api/ps").unwrap()["models"].as_array().unwrap().is_empty());
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
    let fin = chunks.iter().find(|c| c["choices"].get(0).is_some_and(|x| !x["finish_reason"].is_null())).unwrap();
    assert_eq!(fin["choices"][0]["finish_reason"], "length");
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
