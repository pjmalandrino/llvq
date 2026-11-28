//! The batched engine end to end (`LLVQ_BATCH`), on the CPU: the real server,
//! the engine's continuous batching, a tiny Qwen3 stepped through
//! `batch::Pool` with the candle reference attention. More requests than
//! slots arrive at once; each must stream the answer a direct `Session` gives
//! it alone, and a sampling request is refused by name.

mod common;

use llvq_llm::candle::Tensor;
use llvq_llm::batch::{BatchDevice, BatchProjs, CandleBatch};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
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
    ) -> llvq_llm::candle::Result<Tensor> {
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
    chat_telling(c, text, max_new, None)
}

/// [`chat`], telling `started` when its first chunk arrives: from then on the
/// request is decoding, and a second one can join it. The tests that need two
/// requests to overlap wait for that instead of sleeping: a fixed 150 ms raced
/// the decode, and under the CI's release profile the tiny model's 400 tokens
/// can end inside it, so the two requests never stepped together.
fn chat_telling(c: &Client, text: &str, max_new: usize, started: Option<std::sync::mpsc::Sender<()>>) -> (String, Value) {
    let (mut content, mut last) = (String::new(), Value::Null);
    let mut started = started;
    let body = json!({"model": "tiny:test", "messages": [{"role": "user", "content": text}],
                      "options": {"temperature": 0, "num_predict": max_new}});
    c.stream("/api/chat", &body, &mut |v| {
        if let Some(tx) = started.take() {
            let _ = tx.send(());
        }
        content.push_str(v["message"]["content"].as_str().unwrap_or(""));
        last = v.clone();
        Ok(())
    })
    .unwrap();
    (content, last)
}

/// Start a long request on its own thread and return once it is decoding.
fn start_long(host: &str, text: &'static str, max_new: usize) -> std::thread::JoinHandle<(String, Value)> {
    let (tx, rx) = std::sync::mpsc::channel();
    let h = host.to_string();
    let first = std::thread::spawn(move || chat_telling(&Client::new(&h), text, max_new, Some(tx)));
    rx.recv_timeout(std::time::Duration::from_secs(60)).expect("the long request never sent a chunk");
    first
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
    let first = start_long(&host, "the cat sat on the mat", 400);
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
        let first = start_long(&host, "the cat sat on the mat", 400);
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

/// The candle reference, failing its `at`-th call: a panic, as an `expect`
/// reached in the step would, or an error, as a failed launch.
struct Faulty {
    calls: AtomicUsize,
    at: usize,
    panics: bool,
    /// Set when the device is dropped, with the weights beside it.
    dropped: Arc<AtomicBool>,
}

impl Drop for Faulty {
    fn drop(&mut self) {
        self.dropped.store(true, Ordering::SeqCst);
    }
}

impl BatchDevice for Faulty {
    fn name(&self) -> &str {
        "candle, failing once"
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
    ) -> llvq_llm::candle::Result<Tensor> {
        if self.calls.fetch_add(1, Ordering::SeqCst) + 1 == self.at {
            if self.panics {
                panic!("the batched step panics");
            }
            return Err(llvq_llm::candle::Error::Msg("the batched step fails".into()));
        }
        CandleBatch.attend(q, k_new, v_new, kc, vc, pos, slots, max_len)
    }
}

/// The batched engine over two slots, its device [`Faulty`].
struct FaultyLoader {
    at: usize,
    panics: bool,
    dropped: Arc<AtomicBool>,
}

impl Loader for FaultyLoader {
    fn load(&self, _: &ModelName, _: &Manifest, _: &Store) -> anyhow::Result<Loaded> {
        let serve = BatchServe::of(BatchFile { max: 2, context: 1024 }).map_err(anyhow::Error::msg)?;
        Ok(Loaded {
            qwen: common::model(),
            tokenizer: common::tokenizer(),
            decoder: Decoder::eager(),
            backend: "dense-cpu".into(),
            config: None,
            lines: Vec::new(),
            batch: Some(BatchEngine {
                dev: Box::new(Faulty {
                    calls: AtomicUsize::new(0),
                    at: self.at,
                    panics: self.panics,
                    dropped: self.dropped.clone(),
                }),
                projs: BatchProjs::Rows,
                serve,
            }),
        })
    }
}

/// A streamed chat with a keep-alive of one second, telling `started` when
/// its first chunk arrives. `Err` when a line carries an error.
fn stream_ended(host: &str, text: &str, max_new: usize, started: Option<std::sync::mpsc::Sender<()>>) -> anyhow::Result<()> {
    let mut started = started;
    let body = json!({"model": "tiny:test", "messages": [{"role": "user", "content": text}],
                      "options": {"temperature": 0, "num_predict": max_new}, "keep_alive": "1s"});
    Client::new(host).stream("/api/chat", &body, &mut |_| {
        if let Some(tx) = started.take() {
            let _ = tx.send(());
        }
        Ok(())
    })
}

/// Two requests that step together in the pool, the first decoding when the
/// second arrives. With two layers the 5th call of `attend` is in the third
/// batched step, while both are live.
///
/// Each stream runs on its own thread and must end, with `done` or with an
/// error, within 120 s: a stream the engine never ends would otherwise hold
/// the test, and the CI, until its global timeout.
fn two_in_the_pool(host: &str) -> [(&'static str, anyhow::Result<()>); 2] {
    let (started_tx, started) = std::sync::mpsc::channel();
    let (ended_tx, ended) = std::sync::mpsc::channel::<(usize, anyhow::Result<()>)>();
    let (h, tx) = (host.to_string(), ended_tx.clone());
    std::thread::spawn(move || {
        let _ = tx.send((0, stream_ended(&h, "the cat sat on the mat", 400, Some(started_tx))));
    });
    started.recv_timeout(std::time::Duration::from_secs(60)).expect("the long request never sent a chunk");
    let h = host.to_string();
    std::thread::spawn(move || {
        let _ = ended_tx.send((1, stream_ended(&h, "why", 20, None)));
    });
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(120);
    let mut results: [Option<anyhow::Result<()>>; 2] = [None, None];
    for _ in 0..2 {
        match ended.recv_timeout(deadline.saturating_duration_since(std::time::Instant::now())) {
            Ok((i, r)) => results[i] = Some(r),
            Err(e) => panic!("the streams neither ended nor failed within 120 s ({e})"),
        }
    }
    let [first, second] = results;
    [
        ("the request that joined the pool", first.expect("the first stream ended once")),
        ("the second request", second.expect("the second stream ended once")),
    ]
}

/// A panic in the batched step ends every live stream with an error line.
/// The streams ended without `done` and without `error`, and `llvq run`
/// showed the partial answer as a whole one. The weights go with the panic:
/// they stayed in memory until the next request dropped the failed engine.
#[test]
fn a_panic_in_the_batched_step_ends_every_live_stream_with_an_error() {
    let dropped = Arc::new(AtomicBool::new(false));
    let loader = FaultyLoader { at: 5, panics: true, dropped: dropped.clone() };
    let host = common::serve_with(common::store("batch-panic"), "http://127.0.0.1:9", Arc::new(loader));
    for (which, r) in two_in_the_pool(&host) {
        match r {
            Ok(()) => panic!("{which}: the stream ended without an error line"),
            Err(e) => assert!(format!("{e:#}").contains("panicked"), "{which}: {e:#}"),
        }
    }
    let t = std::time::Instant::now();
    while !dropped.load(Ordering::SeqCst) {
        assert!(t.elapsed() < std::time::Duration::from_secs(5), "the failed engine still holds its weights");
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
}

/// A failed batched step frees its requests, and their keep-alive then
/// unloads the model, as after a clean end. It stayed loaded until the next
/// request.
#[test]
fn a_failed_batched_step_leaves_the_model_to_its_keep_alive() {
    let host = common::serve_with(common::store("batch-step-err"), "http://127.0.0.1:9", Arc::new(FaultyLoader { at: 5, panics: false, dropped: Arc::default() }));
    for (which, r) in two_in_the_pool(&host) {
        match r {
            Ok(()) => panic!("{which}: the failed step was not reported"),
            Err(e) => assert!(format!("{e:#}").contains("the batched step failed"), "{which}: {e:#}"),
        }
    }
    let c = Client::new(&host);
    let t = std::time::Instant::now();
    while !c.get("/api/ps").unwrap()["models"].as_array().unwrap().is_empty() {
        assert!(t.elapsed() < std::time::Duration::from_secs(5), "the model outlived its keep-alive of one second");
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
}

/// The answer and the reasoning of a request's `Delta`s, and its last event,
/// read from the channel `Models::submit` returns; panics past 60 s without
/// an end.
fn read_all(mut rx: llvq_runtime::engine::Events, what: &str) -> ((String, String), Option<llvq_runtime::engine::Event>) {
    use llvq_runtime::engine::Event;
    let (mut text, mut last) = ((String::new(), String::new()), None);
    let t = std::time::Instant::now();
    loop {
        match rx.try_recv() {
            Ok(e) => {
                if let Event::Delta { content, thinking } = &e {
                    text.0.push_str(content);
                    text.1.push_str(thinking);
                }
                last = Some(e);
            }
            Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => return (text, last),
            Err(tokio::sync::mpsc::error::TryRecvError::Empty) => {
                assert!(t.elapsed() < std::time::Duration::from_secs(60), "{what}: no end after 60 s");
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }
    }
}

fn user(text: &str, max_new: i64) -> llvq_runtime::engine::Request {
    use llvq_runtime::engine::{Options, Prompt, Request};
    Request {
        prompt: Prompt::Chat(vec![llvq_runtime::chat::Msg { role: llvq_runtime::chat::Role::User, content: text.into() }]),
        think: false,
        options: Options { temperature: Some(0.0), num_predict: Some(max_new), ..Options::default() },
    }
}

/// Wait until `rx` holds at least `n` events unread.
fn holds(rx: &llvq_runtime::engine::Events, n: usize, what: &str) {
    let t = std::time::Instant::now();
    while rx.len() < n {
        assert!(t.elapsed() < std::time::Duration::from_secs(60), "{what}: fewer than {n} events after 60 s");
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
}

/// Under the batch a client that stops reading does not stop the others: a
/// request whose channel is full sits the steps out, the others decode on,
/// and once its client reads again it gets the whole answer, every event in
/// order. Its client never gives up here (no send timeout), so only an
/// engine that does not wait on a full channel lets the others run. Two
/// cases: the held request is alone on the batch-1 path when the reading one
/// arrives, and the held request fills its channel in the pool, beside a
/// reading one.
#[test]
fn under_the_batch_a_client_that_stops_reading_does_not_stop_the_others() {
    use llvq_runtime::engine::{Event, KeepAlive, Models};
    let loader = BatchLoader::new(2, Arc::new(AtomicUsize::new(0)));
    let m = Models::new(common::store("batch-slow"), Arc::new(loader), 1, KeepAlive::For(std::time::Duration::from_secs(300)))
        .with_send_timeout(None);
    let name = ModelName::parse("tiny:test").unwrap();
    let tok = common::tokenizer();
    // The ids alone, routed as the engine routes them: the answer, and the
    // reasoning between `<think>` (3) and `</think>` (4).
    let want = |text: &str, n: usize| {
        let (mut content, mut thinking, mut inside) = (Vec::new(), Vec::new(), false);
        for t in alone(text, n) {
            match t {
                3 => inside = true,
                4 => inside = false,
                _ if inside => thinking.push(t),
                _ => content.push(t),
            }
        }
        (tok.decode(&content, false).unwrap(), tok.decode(&thinking, false).unwrap())
    };
    let done = |last: &Option<Event>, n: usize| matches!(last, Some(Event::Done(s)) if s.eval_count == n);

    // Alone on the batch-1 path, then a reading request arrives.
    let held = m.submit(&name, user("the cat sat on the mat", 400), None).expect("queued");
    holds(&held, 64, "the held channel");
    let (text, last) = read_all(m.submit(&name, user("why", 20), None).expect("queued"), "the reading request");
    assert_eq!(text, want("why", 20), "the reading request");
    assert!(done(&last, 20), "{last:?}");
    let (text, last) = read_all(held, "the held request, read at last");
    assert_eq!(text, want("the cat sat on the mat", 400), "the held request lost or reordered text");
    assert!(done(&last, 400), "{last:?}");

    // In the pool: a reading request decodes, a held one joins it and fills
    // its channel at a step they share.
    let reading = m.submit(&name, user("hello how are you", 400), None).expect("queued");
    holds(&reading, 1, "the reading request");
    let reader = std::thread::spawn(move || read_all(reading, "the reading request beside the held one"));
    let held = m.submit(&name, user("one two", 300), None).expect("queued");
    let (text, last) = reader.join().unwrap();
    assert_eq!(text, want("hello how are you", 400), "the reading request beside the held one");
    assert!(done(&last, 400), "{last:?}");
    let (text, last) = read_all(held, "the request held in the pool, read at last");
    assert_eq!(text, want("one two", 300), "the request held in the pool lost or reordered text");
    assert!(done(&last, 300), "{last:?}");
}
