//! The engine cap and the reaper never drop a busy engine. Dropping one would
//! leave its thread running beside the next load: two models in memory under
//! `LLVQ_MAX_LOADED=1`. Removing either `idle()` filter of `Models` survived
//! the 42 tests before this file (audit 2026-11-08, section 4).

mod common;

use llvq_runtime::engine::{KeepAlive, Models, Options, Prompt, Request};
use llvq_runtime::library::ModelName;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;
use std::time::{Duration, SystemTime};

/// Two names on the tiny model, under a cap of one, each load taking `delay`.
fn models(tag: &str, delay: Duration) -> (Models, ModelName, ModelName) {
    let store = common::store(tag);
    let (a, b) = (ModelName::parse("tiny:test").unwrap(), ModelName::parse("tiny:other").unwrap());
    store.write_manifest(&b, &common::manifest()).unwrap();
    let loader = common::CountingLoader { loads: Arc::new(AtomicUsize::new(0)), delay };
    (Models::new(store, Arc::new(loader), 1, KeepAlive::For(Duration::from_secs(300))), a, b)
}

fn ask() -> Request {
    Request { prompt: Prompt::Raw("hello".into()), think: false, options: Options { num_predict: Some(4), ..Options::default() } }
}

#[test]
fn a_busy_engine_is_not_evicted_for_another_model() {
    let (m, a, b) = models("busy-evict", Duration::from_millis(1500));
    let _ra = m.submit(&a, ask(), None).expect("the first model loads");
    // `a` is loading with a job queued: busy. A second model under a cap of
    // one must wait for it, not drop it.
    match m.submit(&b, ask(), None) {
        Err(r) => assert_eq!(r.status, 503, "{}", r.message),
        Ok(_) => panic!("the busy engine was evicted for another model"),
    }
    let names: Vec<ModelName> = m.running().into_iter().map(|r| r.name).collect();
    assert_eq!(names, vec![a]);
}

#[test]
fn the_reaper_leaves_a_busy_engine_that_retires() {
    let (m, a, _) = models("busy-reap", Duration::from_millis(1500));
    let _ra = m.submit(&a, ask(), None).expect("the model loads");
    // Asked to unload while busy: it is marked, not dropped, and the reaper
    // keeps it until it is idle.
    assert!(m.unload(&a));
    m.reap(SystemTime::now() + Duration::from_secs(3600));
    assert_eq!(m.running().len(), 1, "the reaper dropped a busy engine");
}
