//! The batched engine end to end (`LLVQ_BATCH`), on the CPU: the real server,
//! the engine's continuous batching, a tiny Qwen3 stepped through
//! `batch::Pool` with the candle reference attention. More requests than
//! slots arrive at once; each must stream the answer a direct `Session` gives
//! it alone, and a sampling request is refused by name.

mod common;

use candle_core::Tensor;
use llvq_llm::batch::{BatchDevice, BatchProjs, CandleBatch};
use std::sync::atomic::{AtomicUsize, Ordering};
use llvq_llm::chatfmt::{open_assistant, turn, Marks};
use llvq_llm::repl::Sampler;
use llvq_llm::served::{BatchFile, BatchServe};
use llvq_llm::served_decode::Decoder;
use llvq_llm::session::{Flow, Generate, Session};
use llvq_runtime::client::{Client, ClientError};
use llvq_runtime::engine::{BatchEngine, Loaded, Loader};
use llvq_runtime::library::ModelName;
use llvq_runtime::store::{Manifest, Store};
use serde_json::{json, Value};
use std::sync::Arc;

/// The candle reference, recording the widest step it ran.
struct Widest(Arc<AtomicUsize>);

impl BatchDevice for Widest {
    fn name(&self) -> &str {
        "candle, counting"
    }
    #[allow(clippy::too_many_arguments)]
    fn attend(
        &self,
        q: &Tensor,
        k_new: &Tensor,
        v_new: &Tensor,
        kc: &Tensor,
        vc: &Tensor,
        pos: &Tensor,
        slots: &Tensor,
        max_len: usize,
    ) -> candle_core::Result<Tensor> {
        self.0.fetch_max(q.dim(0)?, Ordering::SeqCst);
        CandleBatch.attend(q, k_new, v_new, kc, vc, pos, slots, max_len)
    }
}

struct BatchLoader(usize, Arc<AtomicUsize>);

impl Loader for BatchLoader {
    fn load(&self, _: &ModelName, _: &Manifest, _: &Store) -> anyhow::Result<Loaded> {
        let serve = BatchServe::of(BatchFile { max: self.0, context: 1024 }).map_err(anyhow::Error::msg)?;
        Ok(Loaded {
            qwen: common::model(),
            tokenizer: common::tokenizer(),
            decoder: Decoder::eager(),
            backend: "dense-cpu".into(),
            config: None,
            lines: Vec::new(),
            batch: Some(BatchEngine { dev: Box::new(Widest(self.1.clone())), projs: BatchProjs::Rows, serve }),
        })
    }
}

/// The ids a fresh `Session` answers one user turn with, greedily.
fn alone(text: &str, max_new: usize) -> Vec<u32> {
    let qwen = common::model();
    let tok = common::tokenizer();
    let marks = Marks::resolve(&tok).unwrap();
    let mut s = Session::new(&qwen);
    let mut ids = turn(&tok, &marks, "user", text).unwrap();
    ids.extend(open_assistant(&tok, &marks, false).unwrap());
    let h = s.feed(&qwen, &ids).unwrap();
    let mut sampler = Sampler { temp: 0.0, top_p: 0.9, state: 0 };
    s.answer(&qwen, &mut Decoder::eager(), h, Generate { sampler: &mut sampler, max_new, stops: &marks.stops() }, &mut |_| {
        Flow::Continue
    })
    .unwrap()
    .said
}

fn chat(c: &Client, text: &str, max_new: usize) -> (String, Value) {
    let (mut content, mut last) = (String::new(), Value::Null);
    let body = json!({"model": "tiny:test", "messages": [{"role": "user", "content": text}],
                      "options": {"temperature": 0, "num_predict": max_new}});
    c.stream("/api/chat", &body, &mut |v| {
        content.push_str(v["message"]["content"].as_str().unwrap_or(""));
        last = v.clone();
        Ok(())
    })
    .unwrap();
    (content, last)
}

#[test]
fn requests_beyond_the_slots_each_get_their_alone_answer() {
    let widest = Arc::new(AtomicUsize::new(0));
    let host = common::serve_with(common::store("batch"), "http://127.0.0.1:9", Arc::new(BatchLoader(3, widest.clone())));
    let asks: Vec<(&str, usize)> =
        vec![("hello how are you", 18), ("the cat sat on the mat", 7), ("why", 25), ("one two", 11), ("yes no yes", 14)];
    let handles: Vec<_> = asks
        .iter()
        .map(|&(t, n)| {
            let h = host.clone();
            std::thread::spawn(move || chat(&Client::new(&h), t, n))
        })
        .collect();
    let tok = common::tokenizer();
    for ((t, n), h) in asks.iter().zip(handles) {
        let (content, last) = h.join().unwrap();
        let want = alone(t, *n);
        assert_eq!(content, tok.decode(&want, false).unwrap(), "{t:?}: the batched answer differs from the one alone");
        assert_eq!(last["done"], true);
        assert_eq!(last["eval_count"].as_u64().unwrap() as usize, want.len());
        assert_eq!(last["done_reason"], "length", "the tiny model never stops on its own");
    }
    assert_eq!(widest.load(Ordering::SeqCst), 3, "the five requests never filled the three slots of one step");
}

#[test]
fn the_batched_engine_refuses_to_sample() {
    let host = common::serve_with(
        common::store("batch-sample"),
        "http://127.0.0.1:9",
        Arc::new(BatchLoader(2, Arc::new(AtomicUsize::new(0)))),
    );
    let c = Client::new(&host);
    let body = json!({"model": "tiny:test", "messages": [{"role": "user", "content": "hello"}],
                      "options": {"temperature": 0.7}, "stream": false});
    match c.send("POST", "/api/chat", &body) {
        Err(ClientError::Status(400, m)) => assert!(m.contains("batched decode"), "{m}"),
        other => panic!("expected a 400, got {other:?}"),
    }
}

#[test]
fn llvq_batch_reads_a_count_or_a_count_and_a_context() {
    use llvq_runtime::engine::batch_from_env;
    assert_eq!(batch_from_env(None).unwrap(), None);
    assert_eq!(batch_from_env(Some("0".into())).unwrap(), None);
    assert_eq!(batch_from_env(Some("64".into())).unwrap(), Some((64, None)));
    assert_eq!(batch_from_env(Some(" 128:4096 ".into())).unwrap(), Some((128, Some(4096))));
    for bad in ["x", "64:", ":4096", "0:4096", "-1"] {
        assert!(batch_from_env(Some(bad.into())).is_err(), "{bad:?} must be refused");
    }
}

#[test]
fn a_lone_request_runs_on_the_batch_1_path_and_joins_the_pool_when_another_arrives() {
    let widest = Arc::new(AtomicUsize::new(0));
    let host = common::serve_with(common::store("batch-solo"), "http://127.0.0.1:9", Arc::new(BatchLoader(4, widest.clone())));
    let tok = common::tokenizer();
    // Alone: no batched step at all.
    let (content, last) = chat(&Client::new(&host), "hello how are you", 30);
    assert_eq!(content, tok.decode(&alone("hello how are you", 30), false).unwrap());
    assert_eq!(last["done_reason"], "length");
    assert_eq!(widest.load(Ordering::SeqCst), 0, "a lone request took the batched step");
    // A long one, then a second while it decodes: the first leaves the batch-1
    // path for a slot where it stands, and both answer as they would alone.
    let h1 = host.clone();
    let first = std::thread::spawn(move || chat(&Client::new(&h1), "the cat sat on the mat", 400));
    std::thread::sleep(std::time::Duration::from_millis(150));
    let (c2, _) = chat(&Client::new(&host), "why", 20);
    let (c1, last1) = first.join().unwrap();
    assert_eq!(c2, tok.decode(&alone("why", 20), false).unwrap(), "the second request");
    assert_eq!(c1, tok.decode(&alone("the cat sat on the mat", 400), false).unwrap(), "the request that joined the pool");
    assert_eq!(last1["eval_count"], 400);
    assert_eq!(widest.load(Ordering::SeqCst), 2, "the two requests never stepped together");
}
