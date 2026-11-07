//! What the runtime adds to `Session`, on the CPU, with no sealed file and no
//! card: the numbers of `docs/runtime-audit-2026-11.md`.
//!
//! ```text
//! cargo run --release -p llvq-runtime --example overhead -- <Qwen3 tokenizer.json>
//! ```
//!
//! Three parts, each a piece of the path from the HTTP request to the token
//! on the client, timed alone:
//!
//! * `text`: the engine's work on each token between two steps of the model
//!   (decode, scan the stop strings, take the new text), the whole-answer
//!   decode the runtime shipped with against [`TextStream`], at several
//!   answer lengths, and a check that both give the same text at every
//!   token, on the repository's docs and on random ids.
//! * `render`: a request's messages to ids (`chat::render`), with and without
//!   the memo, and the prefix kept against the cache.
//! * `transport`: one chunk from the engine's thread to the client's socket,
//!   the server's own types and channel: the worker's cost a send, the
//!   pipeline's cost a chunk, and the latency of one chunk.
//! * `accept`: the real server and the tiny model of the tests, from the
//!   request to the response's headers. The headers leave when the engine
//!   accepts the request, after the template and the prefix and before the
//!   prefill: that time is the runtime's, not the model's.

#[path = "../tests/common/mod.rs"]
mod common;

use llvq_llm::chatfmt::{open_assistant, Marks};
use llvq_llm::session::common_prefix;
use llvq_runtime::api::{rfc3339, ChatMessage, ChatResponse, DoneFields};
use llvq_runtime::chat::{self, Msg, Role, Scan, Segment, TextStream};
use llvq_runtime::engine::Event;
use std::io::{BufRead, BufReader};
use std::time::{Duration, Instant, SystemTime};

/// The engine's text work for one step as it shipped: decode the whole
/// answer, scan it, take the new part.
#[derive(Default)]
struct Whole {
    ids: Vec<u32>,
    sent: TextStream,
}

impl Whole {
    fn step(&mut self, tok: &tokenizers::Tokenizer, new: &[u32], stops: &[String]) -> String {
        self.ids.extend_from_slice(new);
        let full = tok.decode(&self.ids, false).unwrap();
        let upto = match chat::scan(&full, stops) {
            Scan::Safe(n) | Scan::Stop(n) => n,
        };
        self.sent.take(&full, upto)
    }
}

/// The same step through [`TextStream::full`], as the engine runs it now.
fn step(s: &mut TextStream, tok: &tokenizers::Tokenizer, new: &[u32], stops: &[String]) -> String {
    s.ids.extend_from_slice(new);
    let full = s.full(tok).unwrap();
    let upto = match chat::scan(&full, stops) {
        Scan::Safe(n) | Scan::Stop(n) => n,
    };
    s.take(&full, upto)
}

fn median_us(mut v: Vec<Duration>) -> f64 {
    v.sort();
    v[v.len() / 2].as_secs_f64() * 1e6
}

fn docs_ids(tok: &tokenizers::Tokenizer) -> Vec<u32> {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/..");
    let mut corpus = String::new();
    for f in ["docs/HISTORIQUE.md", "docs/ETAT.md", "docs/ROADMAP.md"] {
        corpus.push_str(&std::fs::read_to_string(format!("{root}/{f}")).unwrap_or_default());
    }
    let ids = tok.encode(corpus.as_str(), false).unwrap().get_ids().to_vec();
    ids[..ids.len().min(8192)].to_vec()
}

fn text_part(tok: &tokenizers::Tokenizer) {
    let ids = docs_ids(tok);
    let n = ids.len();
    println!("text: {n} tokens of the repository's docs");
    let stops = vec!["\n\n\n".to_string(), "</answer>".to_string(), "User:".to_string()];
    for (label, stops) in [("no stop string", vec![]), ("3 stop strings", stops)] {
        let mut whole = Whole::default();
        let mut stream = TextStream::default();
        let (mut tw, mut ts) = (Vec::with_capacity(n), Vec::with_capacity(n));
        let (mut ow, mut os) = (String::new(), String::new());
        for &id in &ids {
            let t = Instant::now();
            ow.push_str(&whole.step(tok, &[id], &stops));
            tw.push(t.elapsed());
            let t = Instant::now();
            os.push_str(&step(&mut stream, tok, &[id], &stops));
            ts.push(t.elapsed());
            assert_eq!(ow, os, "the two text paths differ at token {}", whole.ids.len());
        }
        let row = |v: &[Duration], from: usize| median_us(v[from.min(n - 1)..(from + 256).min(n)].to_vec());
        for (name, v) in [("whole decode", &tw), ("TextStream  ", &ts)] {
            println!(
                "  {label:<15} {name}: µs a token at 0 {:6.1}, 1,024 {:6.1}, 4,096 {:6.1}, 7,936 {:6.1}; \
                 whole answer {:7.1} ms",
                row(v, 0),
                row(v, 1024),
                row(v, 4096),
                row(v, 7936),
                v.iter().sum::<Duration>().as_secs_f64() * 1e3
            );
        }
    }
    // Random ids: byte-level pieces that split characters anywhere.
    let vocab = tok.get_vocab_size(false) as u64;
    let mut state = 0x1234_5678u64;
    let mut stream = TextStream::default();
    let mut ids = Vec::new();
    for _ in 0..4096 {
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        let id = ((state >> 33) % vocab) as u32;
        ids.push(id);
        stream.ids.push(id);
        assert_eq!(stream.full(tok).unwrap(), tok.decode(&ids, false).unwrap(), "random ids differ at {}", ids.len());
    }
    println!("  same text at every token: the docs with and without stop strings, and 4,096 random ids");
}

fn render_part(tok: &tokenizers::Tokenizer) {
    let marks = Marks::resolve(tok).unwrap();
    // Twenty turns of about 400 tokens each, from the docs.
    let text = tok.decode(&docs_ids(tok), false).unwrap();
    let pieces: Vec<&str> = text.as_bytes().chunks(1600).map(|c| std::str::from_utf8(c).unwrap_or("x")).collect();
    let msgs: Vec<Msg> = (0..20)
        .map(|i| Msg {
            role: if i % 2 == 0 { Role::User } else { Role::Assistant },
            content: pieces.get(i).copied().unwrap_or("x").to_string(),
        })
        .collect();
    let rounds = 21;
    let mut cold = Vec::new();
    let mut warm = Vec::new();
    let mut prefix = Vec::new();
    let mut memo: Vec<Segment> = Vec::new();
    let mut total = 0;
    for _ in 0..rounds {
        let t = Instant::now();
        let segs = chat::render(tok, &marks, &[], &msgs).unwrap();
        cold.push(t.elapsed());
        memo = segs;
        let t = Instant::now();
        let segs = chat::render(tok, &marks, &memo[..19], &msgs).unwrap();
        let mut prompt: Vec<u32> = segs.iter().flat_map(|s| s.ids.iter().copied()).collect();
        prompt.extend(open_assistant(tok, &marks, false).unwrap());
        warm.push(t.elapsed());
        total = prompt.len();
        let held: Vec<u32> = prompt[..prompt.len() - 300].to_vec();
        let t = Instant::now();
        std::hint::black_box(common_prefix(&held, &prompt));
        prefix.push(t.elapsed());
    }
    std::hint::black_box(&memo);
    println!("render: 20 messages, {total} tokens, medians of {rounds}");
    println!(
        "  all messages tokenized {:.0} µs; 19 from the memo and the last tokenized {:.0} µs; prefix against the cache {:.1} µs",
        median_us(cold),
        median_us(warm),
        median_us(prefix)
    );
}

/// The server's NDJSON chunk of a delta, as `api::chunk_chat` builds it.
fn chunk(content: String) -> Vec<u8> {
    let v = serde_json::to_value(ChatResponse {
        model: "qwen3:4b".into(),
        created_at: rfc3339(SystemTime::now()),
        message: ChatMessage { role: "assistant".into(), content, thinking: None, images: None, tool_calls: None },
        done: false,
        stats: DoneFields::default(),
    })
    .unwrap();
    let mut b = serde_json::to_vec(&v).unwrap();
    b.push(b'\n');
    b
}

type Slot = std::sync::mpsc::Sender<tokio::sync::mpsc::Sender<Event>>;

/// A server whose one route streams what the caller sends, through the
/// engine's channel, `api::events` and an NDJSON body, as `/api/chat` does.
/// Each request hands its event sender to the caller on the returned channel.
fn transport_server() -> (String, std::sync::mpsc::Receiver<tokio::sync::mpsc::Sender<Event>>) {
    use futures_util::StreamExt;
    let (slot, senders): (Slot, _) = std::sync::mpsc::channel();
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build().unwrap();
        rt.block_on(async move {
            let route = move || {
                let slot = slot.clone();
                async move {
                    let (etx, erx) = tokio::sync::mpsc::channel::<Event>(64);
                    slot.send(etx).unwrap();
                    let body = axum::body::Body::from_stream(llvq_runtime::api::events(erx).map(|e| {
                        let b = match e {
                            Event::Delta { content, .. } => chunk(content),
                            _ => b"{\"done\":true}\n".to_vec(),
                        };
                        Ok::<_, std::convert::Infallible>(axum::body::Bytes::from(b))
                    }));
                    axum::response::Response::builder().body(body).unwrap()
                }
            };
            let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            tx.send(l.local_addr().unwrap().to_string()).unwrap();
            axum::serve(l, axum::Router::new().route("/s", axum::routing::post(route))).await.unwrap();
        });
    });
    (rx.recv().unwrap(), senders)
}

/// One streamed request: a client thread reads the lines and reports each
/// on `lines`. Returns the engine-side sender.
fn open(
    url: &str,
    senders: &std::sync::mpsc::Receiver<tokio::sync::mpsc::Sender<Event>>,
) -> (tokio::sync::mpsc::Sender<Event>, std::sync::mpsc::Receiver<Instant>, std::thread::JoinHandle<()>) {
    let (line_tx, lines) = std::sync::mpsc::channel();
    let url = url.to_string();
    let client = std::thread::spawn(move || {
        let resp = ureq::post(&url).send_string("{}").unwrap();
        for l in BufReader::new(resp.into_reader()).lines() {
            l.unwrap();
            let _ = line_tx.send(Instant::now());
        }
    });
    (senders.recv().unwrap(), lines, client)
}

fn transport_part() {
    let (host, senders) = transport_server();
    let url = format!("http://{host}/s");
    let delta = || Event::Delta { content: " word".into(), thinking: String::new() };
    let rounds = 7;
    let n = 20_000u32;
    let (mut per_send, mut per_chunk, mut latency) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..rounds {
        // Throughput: the engine sends as fast as the pipeline takes.
        let (etx, lines, client) = open(&url, &senders);
        let t0 = Instant::now();
        let mut sending = Duration::ZERO;
        for _ in 0..n {
            let t = Instant::now();
            etx.blocking_send(delta()).unwrap();
            sending += t.elapsed();
        }
        etx.blocking_send(Event::Done(Default::default())).unwrap();
        drop(etx);
        let last = lines.iter().last().unwrap();
        client.join().unwrap();
        per_send.push(sending / n);
        per_chunk.push((last - t0) / n);
        // Latency: one chunk at a time, the next sent once the client has it.
        let (etx, lines, client) = open(&url, &senders);
        let mut rtt = Vec::new();
        for _ in 0..2000 {
            let t = Instant::now();
            etx.blocking_send(delta()).unwrap();
            let got = lines.recv().unwrap();
            rtt.push(got - t);
        }
        etx.blocking_send(Event::Done(Default::default())).unwrap();
        drop(etx);
        for _ in lines.iter() {}
        client.join().unwrap();
        latency.push(Duration::from_secs_f64(median_us(rtt) / 1e6));
    }
    println!("transport: the server's channel, events and NDJSON body to a client on loopback, medians of {rounds}");
    println!(
        "  engine thread a send {:.2} µs; pipeline a chunk {:.2} µs ({n} chunks a round); one chunk, send to client {:.0} µs",
        median_us(per_send),
        median_us(per_chunk),
        median_us(latency)
    );
}

fn accept_part() {
    let host = common::serve(common::store("overhead"), "http://127.0.0.1:9");
    let agent = ureq::AgentBuilder::new().build();
    let url = format!("http://{host}/api/chat");
    let ask = |content: &str| {
        serde_json::json!({"model": "tiny:test", "messages": [{"role": "user", "content": content}],
            "options": {"temperature": 0, "num_predict": 1}})
    };
    agent.post(&url).send_json(ask("one two")).unwrap().into_string().unwrap();
    let mut t = Vec::new();
    for i in 0..201 {
        let t0 = Instant::now();
        let resp = agent.post(&url).send_json(ask(if i % 2 == 0 { "one two" } else { "the cat" })).unwrap();
        t.push(t0.elapsed());
        resp.into_string().unwrap();
        // The engine closes the last answer in the cache after its last
        // chunk: let that step end, or the next request waits for it.
        std::thread::sleep(Duration::from_millis(20));
    }
    println!("accept: request to response headers on loopback, a loaded idle model, median of {}: {:.0} µs", t.len(), median_us(t));
    let mut floor = Vec::new();
    for _ in 0..201 {
        let t0 = Instant::now();
        agent.get(&format!("http://{host}/api/version")).call().unwrap().into_string().unwrap();
        floor.push(t0.elapsed());
    }
    let mut show = Vec::new();
    for _ in 0..201 {
        let t0 = Instant::now();
        agent.post(&format!("http://{host}/api/show")).send_json(serde_json::json!({"model": "tiny:test"})).unwrap().into_string().unwrap();
        show.push(t0.elapsed());
    }
    println!("  the same client on /api/version {:.0} µs, on /api/show (reads the manifest) {:.0} µs", median_us(floor), median_us(show));
    // Its pieces, timed alone.
    let store = common::store("overhead-pieces");
    let name = llvq_runtime::library::ModelName::parse("tiny:test").unwrap();
    let qwen = common::model();
    let tok = common::tokenizer();
    let marks = Marks::resolve(&tok).unwrap();
    let msgs = vec![Msg { role: Role::User, content: "one two".into() }];
    let (mut get, mut render, mut trunc) = (Vec::new(), Vec::new(), Vec::new());
    let mut session = llvq_llm::session::Session::new(&qwen);
    for _ in 0..201 {
        let t0 = Instant::now();
        std::hint::black_box(store.get(&name).unwrap());
        get.push(t0.elapsed());
        let t0 = Instant::now();
        let segs = chat::render(&tok, &marks, &[], &msgs).unwrap();
        let mut prompt: Vec<u32> = segs.iter().flat_map(|s| s.ids.iter().copied()).collect();
        prompt.extend(open_assistant(&tok, &marks, false).unwrap());
        render.push(t0.elapsed());
        session.reset(&qwen);
        session.feed(&qwen, &prompt).unwrap();
        session.feed(&qwen, &[10, 11, 12]).unwrap();
        let t0 = Instant::now();
        session.reuse_prefix(&prompt).unwrap();
        trunc.push(t0.elapsed());
    }
    println!(
        "  manifest read {:.0} µs, template {:.0} µs, cache cut to the prefix {:.0} µs",
        median_us(get),
        median_us(render),
        median_us(trunc)
    );
}

fn main() {
    let path = std::env::args().nth(1).or_else(|| std::env::var("LLVQ_TOKENIZER").ok());
    match path {
        Some(p) => {
            let tok = tokenizers::Tokenizer::from_file(&p).expect("the tokenizer loads");
            println!("tokenizer: {p}");
            text_part(&tok);
            render_part(&tok);
        }
        None => println!("text, render: skipped, no tokenizer.json given"),
    }
    transport_part();
    accept_part();
}
