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

/// The candle reference, recording the widest step it ran and how many
/// layer calls ran a single row.
struct Widest(Arc<AtomicUsize>, Arc<AtomicUsize>);

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
        if q.dim(0)? == 1 {
            self.1.fetch_add(1, Ordering::SeqCst);
        }
        CandleBatch.attend(q, k_new, v_new, kc, vc, pos, slots, max_len)
    }
}

/// The batched engine over `slots` slots; `ones` counts the layer calls of
/// single-row steps; `cap` gives the batch-1 path a position cap, as 4E has.
struct BatchLoader {
    slots: usize,
    widest: Arc<AtomicUsize>,
    ones: Arc<AtomicUsize>,
    cap: Option<usize>,
}

impl BatchLoader {
    fn new(slots: usize, widest: Arc<AtomicUsize>) -> Self {
        BatchLoader { slots, widest, ones: Arc::new(AtomicUsize::new(0)), cap: None }
    }
}

impl Loader for BatchLoader {
    fn load(&self, _: &ModelName, _: &Manifest, _: &Store) -> anyhow::Result<Loaded> {
        let serve = BatchServe::of(BatchFile { max: self.slots, context: 1024 }).map_err(anyhow::Error::msg)?;
        Ok(Loaded {
            qwen: common::model(),
            tokenizer: common::tokenizer(),
            decoder: self.cap.map_or_else(Decoder::eager, Decoder::eager_capped),
            backend: "dense-cpu".into(),
            config: None,
            lines: Vec::new(),
            batch: Some(BatchEngine {
                dev: Box::new(Widest(self.widest.clone(), self.ones.clone())),
                projs: BatchProjs::Rows,
                serve,
            }),
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
    let host = common::serve_with(common::store("batch"), "http://127.0.0.1:9", Arc::new(BatchLoader::new(3, widest.clone())));
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
        Arc::new(BatchLoader::new(2, Arc::new(AtomicUsize::new(0)))),
    );
    let c = Client::new(&host);
    let body = json!({"model": "tiny:test", "messages": [{"role": "user", "content": "hello"}],
                      "options": {"temperature": 0.7}, "stream": false});
    match c.send("POST", "/api/chat", &body) {
        Err(ClientError::Status(400, m)) => assert_eq!(
            m,
            "temperature 0.7: this model is served with the batched decode (LLVQ_BATCH), which takes the argmax on \
             the card and cannot sample. Send temperature 0, or leave it out."
        ),
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
    let host = common::serve_with(common::store("batch-solo"), "http://127.0.0.1:9", Arc::new(BatchLoader::new(4, widest.clone())));
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

/// The prompt ids of one user turn, as the engine builds them.
fn prompt_len(text: &str) -> usize {
    let tok = common::tokenizer();
    let marks = Marks::resolve(&tok).unwrap();
    let mut ids = turn(&tok, &marks, "user", text).unwrap();
    ids.extend(open_assistant(&tok, &marks, false).unwrap());
    ids.len()
}

/// A request that outlives its neighbour goes back to the batch-1 path (4E
/// on the card) instead of stepping alone in the pool: the batched step at
/// B = 1 runs at 0.47x of 4E at the 4B (audit 2026-11-08 §1.2). Two rounds
/// on two slots: the slot the survivor leaves is free for the next round.
#[test]
fn a_request_left_alone_in_the_pool_goes_back_to_the_batch_1_path() {
    let loader = BatchLoader::new(2, Arc::new(AtomicUsize::new(0)));
    let (widest, ones) = (loader.widest.clone(), loader.ones.clone());
    let host = common::serve_with(common::store("batch-back"), "http://127.0.0.1:9", Arc::new(loader));
    let tok = common::tokenizer();
    let (want1, want2) = (alone("the cat sat on the mat", 400), alone("why", 20));
    for round in 0..2 {
        let h1 = host.clone();
        let first = std::thread::spawn(move || chat(&Client::new(&h1), "the cat sat on the mat", 400));
        std::thread::sleep(std::time::Duration::from_millis(150));
        let (c2, _) = chat(&Client::new(&host), "why", 20);
        let (c1, last1) = first.join().unwrap();
        assert_eq!(c2, tok.decode(&want2, false).unwrap(), "round {round}: the short request");
        assert_eq!(c1, tok.decode(&want1, false).unwrap(), "round {round}: the request that outlived it");
        assert_eq!(last1["eval_count"], 400, "round {round}");
        assert_eq!(widest.load(Ordering::SeqCst), 2, "round {round}: the two requests never stepped together");
        assert_eq!(ones.load(Ordering::SeqCst), 0, "round {round}: a request stepped alone in the pool");
    }
}

/// A lone request whose answer runs past the batch-1 path's positions (4E
/// holds 8,704, `LLVQ_BATCH` may give the pool more) hands over to the pool
/// at the cap and finishes there, as it would alone (audit 2026-11-08 §1.7:
/// 4E refused in mid-answer and the client got a 500).
#[test]
fn a_lone_request_past_the_batch_1_cap_finishes_in_the_pool() {
    let text = "hello how are you";
    let mut loader = BatchLoader::new(2, Arc::new(AtomicUsize::new(0)));
    loader.cap = Some(prompt_len(text) + 10);
    let ones = loader.ones.clone();
    let host = common::serve_with(common::store("batch-cap"), "http://127.0.0.1:9", Arc::new(loader));
    let (content, last) = chat(&Client::new(&host), text, 40);
    assert_eq!(content, common::tokenizer().decode(&alone(text, 40), false).unwrap());
    assert_eq!(last["eval_count"], 40);
    assert_eq!(last["done_reason"], "length");
    assert!(ones.load(Ordering::SeqCst) > 0, "the tokens past the cap were not decoded in the pool");
}

/// `/api/ps` counts the batch pool's caches: allocated once at load, beside
/// the weights (audit 2026-11-08 §1.8).
#[test]
fn ps_counts_the_batch_pool() {
    let host = common::serve_with(
        common::store("batch-ps"),
        "http://127.0.0.1:9",
        Arc::new(BatchLoader::new(4, Arc::new(AtomicUsize::new(0)))),
    );
    let c = Client::new(&host);
    chat(&c, "why", 2);
    let ps = c.get("/api/ps").unwrap();
    let want = common::model().batch_kv_bytes(4, 1024);
    assert!(want > 0);
    assert_eq!(ps["models"][0]["llvq"]["batch_pool_bytes"].as_u64(), Some(want), "{ps}");
}
