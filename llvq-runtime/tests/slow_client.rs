//! A client that stays connected and stops reading. The engine sends its
//! events on a channel of 64; once the channel is full, a send waits for the
//! client. The engine gives such a client `LLVQ_SEND_TIMEOUT` and then treats
//! it as gone, as it treats a client that closed the connection.
//!
//! These tests hold the channel `Models::submit` returns and read it, or not,
//! as they please: no socket buffer stands between the engine and the test,
//! so what the engine waits on is exactly what the test holds.

mod common;
#[path = "common/ink.rs"]
mod ink;

use llvq_llm::candle::{Device, Tensor};
use llvq_llm::inkchat::InkChat;
use llvq_runtime::chat::{Msg, Role};
use llvq_runtime::engine::{Built, Event, Events, KeepAlive, Loaded, Loader, Models, Options, Prompt, Request};
use llvq_runtime::inkling::{InkEngine, TurnLm};
use llvq_runtime::library::ModelName;
use llvq_runtime::store::{Manifest, Store};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// The capacity of a request's channel (`Models::submit`).
const CHANNEL: usize = 64;

/// What bounds a failure. A passing test never waits for it.
const BOUND: Duration = Duration::from_secs(60);

fn models(tag: &str, loader: Arc<dyn Loader>, send_timeout: Option<Duration>) -> Models {
    Models::new(common::store(tag), loader, 1, KeepAlive::For(Duration::from_secs(300))).with_send_timeout(send_timeout)
}

/// A greedy answer of `n` tokens: the tiny model never stops on its own.
fn raw(n: i64) -> Request {
    let options = Options { temperature: Some(0.0), num_predict: Some(n), ..Options::default() };
    Request { prompt: Prompt::Raw("one two".into()), think: false, options }
}

/// Wait until `cond` holds; panic naming `what` past [`BOUND`].
fn wait_for(what: &str, mut cond: impl FnMut() -> bool) {
    let t = Instant::now();
    while !cond() {
        assert!(t.elapsed() < BOUND, "{what}: still not so after {BOUND:?}");
        std::thread::sleep(Duration::from_millis(1));
    }
}

/// The next event, or `None` once the engine dropped the channel. Panics
/// past [`BOUND`]: an engine that waits on another client never answers.
fn next(rx: &mut Events, what: &str) -> Option<Event> {
    let t = Instant::now();
    loop {
        match rx.try_recv() {
            Ok(e) => return Some(e),
            Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => return None,
            Err(tokio::sync::mpsc::error::TryRecvError::Empty) => {
                assert!(t.elapsed() < BOUND, "{what}: no event and no end after {BOUND:?}");
                std::thread::sleep(Duration::from_millis(1));
            }
        }
    }
}

/// Every event up to the channel's end.
fn all(mut rx: Events, what: &str) -> Vec<Event> {
    let mut v = Vec::new();
    while let Some(e) = next(&mut rx, what) {
        v.push(e);
    }
    v
}

/// The events with the timings and the cached counts zeroed: two runs of one
/// greedy request give the same list, the second reading its prompt from the
/// cache the first left.
fn untimed(v: Vec<Event>) -> Vec<Event> {
    v.into_iter()
        .map(|e| match e {
            Event::Accepted { prompt_tokens, .. } => Event::Accepted { prompt_tokens, cached: 0 },
            Event::Done(s) => Event::Done(llvq_runtime::engine::Stats {
                total_duration: 0,
                load_duration: 0,
                prompt_eval_duration: 0,
                prompt_eval_cached_count: 0,
                eval_duration: 0,
                ..s
            }),
            e => e,
        })
        .collect()
}

/// A client that holds its channel and never reads is given up after the
/// send timeout: its answer stops where the channel filled, the channel is
/// dropped, and the request queued behind it runs. Before the timeout the
/// engine waited on that client for ever, and every later request with it.
#[test]
fn a_client_that_stops_reading_is_given_up_and_the_next_request_runs() {
    let m = models("slow-gone", Arc::new(common::TinyLoader), Some(Duration::from_millis(200)));
    let a = ModelName::parse("tiny:test").unwrap();
    let held = m.submit(&a, raw(2000), None).expect("queued");
    wait_for("the held channel is full", || held.len() == CHANNEL);
    let events = all(m.submit(&a, raw(4), None).expect("queued"), "the request behind the held one");
    assert!(matches!(events.last(), Some(Event::Done(s)) if s.eval_count == 4), "{events:?}");
    // The held answer was cut where its channel filled: no `Done`, and the
    // engine let go of the channel, which the routes turn into an error line.
    let held = all(held, "the held channel");
    assert_eq!(held.len(), CHANNEL, "events past a full channel were sent or lost");
    assert!(matches!(held[0], Event::Accepted { .. }));
    assert!(held.iter().all(|e| !matches!(e, Event::Done(_) | Event::Error { .. })), "{:?}", held.last());
}

/// A client slower than the engine, but never slower than the timeout, gets
/// every event of its answer, in order: the same list a client that reads at
/// once gets. The test reads one event each time the channel is full, so the
/// engine finds it full at every send it makes from the 65th on.
#[test]
fn a_slow_client_within_the_timeout_gets_every_event_in_order() {
    let m = models("slow-order", Arc::new(common::TinyLoader), Some(Duration::from_secs(30)));
    let a = ModelName::parse("tiny:test").unwrap();
    let fast = untimed(all(m.submit(&a, raw(300), None).expect("queued"), "the fast client"));
    assert!(fast.len() > 2 * CHANNEL, "the answer of {} events does not fill the channel twice", fast.len());
    let mut rx = m.submit(&a, raw(300), None).expect("queued");
    let mut slow = Vec::new();
    loop {
        wait_for("the channel is full or closed", || rx.len() == CHANNEL || rx.is_closed());
        match next(&mut rx, "the slow client") {
            Some(e) => slow.push(e),
            None => break,
        }
    }
    assert_eq!(untimed(slow), fast, "the slow client lost or reordered an event");
}

/// A scripted Inkling model that writes `a` for ever, and panics on the
/// first decode step it takes once the held request's channel is full.
/// `held` carries that channel; the prompt waits until it is there.
struct Stuck {
    script: Vec<u32>,
    filler: u32,
    ctx: usize,
    steps: usize,
    held: Arc<Mutex<Option<Events>>>,
    panicked: Arc<AtomicBool>,
}

impl TurnLm for Stuck {
    type Mark = usize;

    fn feed(&mut self, ids: &[u32]) -> anyhow::Result<Tensor> {
        self.ctx += ids.len();
        if ids.len() > 1 {
            self.steps = 0;
            wait_for("the test hands over the held channel", || self.held.lock().unwrap().is_some());
        } else {
            self.steps += 1;
            let full = self.held.lock().unwrap().as_ref().is_some_and(|r| r.len() == CHANNEL);
            if full && !self.panicked.swap(true, Ordering::SeqCst) {
                panic!("the scripted model panics");
            }
        }
        let next = self.script.get(self.steps).copied().unwrap_or(self.filler);
        let mut v = vec![0f32; ink::VOCAB];
        v[next as usize] = 10.0;
        Ok(Tensor::from_vec(v, (1, ink::VOCAB), &Device::Cpu)?)
    }

    fn held(&self) -> usize {
        self.ctx
    }

    fn mark(&self) -> usize {
        self.ctx
    }

    fn rollback(&mut self, m: &usize) -> anyhow::Result<()> {
        self.ctx = *m;
        Ok(())
    }

    fn reset(&mut self) {
        self.ctx = 0;
    }
}

struct StuckLoader {
    held: Arc<Mutex<Option<Events>>>,
    panicked: Arc<AtomicBool>,
}

impl Loader for StuckLoader {
    fn load(&self, _: &ModelName, _: &Manifest, _: &Store) -> anyhow::Result<Loaded> {
        anyhow::bail!("an Inkling engine: built by `build`")
    }

    fn build(&self, _: &ModelName, _: &Manifest, _: &Store) -> anyhow::Result<Built> {
        let chat = InkChat::new(ink::byte_tokenizer())?;
        let enc = |s: &str| chat.tokenizer().encode(s, false).unwrap().get_ids().to_vec();
        let (script, filler) = (enc("<|message_model|><|content_text|>"), enc("a")[0]);
        let lm = Stuck { script, filler, ctx: 0, steps: 0, held: self.held.clone(), panicked: self.panicked.clone() };
        Ok(Built::Inkling(InkEngine::new(lm, chat, 4096, "script", None)))
    }
}

fn hello() -> Request {
    Request {
        prompt: Prompt::Chat(vec![Msg { role: Role::User, content: "hello".into() }]),
        think: false,
        options: Options { num_predict: Some(1000), ..Options::default() },
    }
}

/// A panic while the running request's client has stopped reading fails the
/// engine without waiting on that client: the request queued behind it is
/// answered with the panic's reason, and the held channel is let go. The
/// handler sent the reason with a send that waited for room, and the thread
/// stayed parked there, with every queued request.
#[test]
fn a_panic_does_not_wait_on_a_client_that_stopped_reading() {
    let held = Arc::new(Mutex::new(None));
    let loader = StuckLoader { held: held.clone(), panicked: Arc::new(AtomicBool::new(false)) };
    let m = models("slow-panic", Arc::new(loader), None);
    let a = ModelName::parse("tiny:test").unwrap();
    let first = m.submit(&a, hello(), None).expect("queued");
    let second = m.submit(&a, hello(), None).expect("queued behind the first");
    *held.lock().unwrap() = Some(first);
    let events = all(second, "the request queued behind the panic");
    assert!(
        matches!(events.as_slice(), [Event::Error { status: 500, message }] if message.contains("panicked")),
        "{events:?}"
    );
    let first = held.lock().unwrap().take().expect("the held channel");
    let first = all(first, "the held channel");
    assert_eq!(first.len(), CHANNEL, "{:?}", first.last());
    assert!(first.iter().all(|e| !matches!(e, Event::Done(_))));
}
