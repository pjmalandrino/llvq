//! The engine cap and the reaper never drop a busy engine. Dropping one would
//! leave its thread running beside the next load: two models in memory under
//! `LLVQ_MAX_LOADED=1`. Removing either `idle()` filter of `Models` survived
//! the 42 tests before this file (audit 2026-11-08, section 4).

mod common;

use llvq_runtime::engine::{Event, Events, KeepAlive, Loaded, Loader, Models, Options, Prompt, Request, State};
use llvq_runtime::library::ModelName;
use llvq_runtime::store::{Manifest, Store};
use std::sync::atomic::{AtomicUsize, Ordering};
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

/// The last event of a request: its `Done` or its `Error`, or the last one
/// before the engine dropped the channel.
fn last_event(mut rx: Events) -> Option<Event> {
    let mut last = None;
    while let Some(e) = rx.blocking_recv() {
        let end = matches!(e, Event::Done(_) | Event::Error { .. });
        last = Some(e);
        if end {
            break;
        }
    }
    last
}

/// A loader that panics, as an assert in a device library does on a
/// malformed `LLVQ_NVRTC_ARCH`.
struct PanickingLoader {
    loads: Arc<AtomicUsize>,
}

impl Loader for PanickingLoader {
    fn load(&self, _: &ModelName, _: &Manifest, _: &Store) -> anyhow::Result<Loaded> {
        self.loads.fetch_add(1, Ordering::SeqCst);
        panic!("the loader panics");
    }
}

/// A load that panics fails the engine as a load that errs does: the
/// request is answered with the reason, the queue empties, and the next
/// request loads again instead of finding an engine loading forever, which
/// under a cap of one refused every other model with a 503.
#[test]
fn a_load_that_panics_fails_the_engine_and_the_next_request_loads_again() {
    let store = common::store("load-panic");
    let (a, b) = (ModelName::parse("tiny:test").unwrap(), ModelName::parse("tiny:other").unwrap());
    store.write_manifest(&b, &common::manifest()).unwrap();
    let loads = Arc::new(AtomicUsize::new(0));
    let m = Models::new(store, Arc::new(PanickingLoader { loads: loads.clone() }), 1, KeepAlive::For(Duration::from_secs(300)));
    match last_event(m.submit(&a, ask(), None).expect("the request is queued")) {
        Some(Event::Error { status: 500, message }) => assert!(message.contains("panicked"), "{message}"),
        other => panic!("the request ended on {other:?}, not on the load's panic"),
    }
    let ps = m.running();
    assert_eq!(ps.len(), 1);
    assert!(matches!(ps[0].state, State::Failed(_)), "the engine stays {:?}", ps[0].state);
    assert_eq!(ps[0].queue, 0, "the failed engine still counts a job");
    // The same model again: a second load, not "the engine has stopped".
    match last_event(m.submit(&a, ask(), None).expect("the failed engine is forgotten")) {
        Some(Event::Error { message, .. }) => assert!(message.contains("panicked"), "{message}"),
        other => panic!("the second request ended on {other:?}"),
    }
    assert_eq!(loads.load(Ordering::SeqCst), 2, "the second request did not load again");
    // Another model under the cap of one.
    if let Err(r) = m.submit(&b, ask(), None) {
        panic!("another model was refused beside a failed load: {} {}", r.status, r.message);
    }
}

/// A pull or an import of a model already present writes its manifest again
/// with a new date and nothing else. The loaded engine serves it as it is:
/// it is neither unloaded nor loaded a second time.
#[test]
fn a_manifest_written_again_with_a_new_date_keeps_the_loaded_engine() {
    let store = common::store("same-manifest");
    let a = ModelName::parse("tiny:test").unwrap();
    let loads = Arc::new(AtomicUsize::new(0));
    let loader = common::CountingLoader { loads: loads.clone(), delay: Duration::ZERO };
    let m = Models::new(store.clone(), Arc::new(loader), 1, KeepAlive::For(Duration::from_secs(300)));
    assert!(matches!(last_event(m.submit(&a, ask(), None).expect("the model loads")), Some(Event::Done(_))));
    let mut again = common::manifest();
    again.modified_at += 3600;
    store.write_manifest(&a, &again).unwrap();
    let rx = m.submit(&a, ask(), None).unwrap_or_else(|r| panic!("refused: {} {}", r.status, r.message));
    assert!(matches!(last_event(rx), Some(Event::Done(_))));
    assert_eq!(loads.load(Ordering::SeqCst), 1, "a manifest that changed only its date loaded the model again");
}

/// A pull that found the blob damaged removes it, and the next pull writes
/// the file again and its manifest with a new date and nothing else. The
/// engine loaded from the damaged bytes must not serve on: the next request
/// loads the new file. A pull that found the blob sound, the same manifest
/// rewritten over the same file, keeps the engine.
#[test]
fn a_blob_written_again_under_the_same_manifest_loads_the_model_again() {
    let store = common::store("blob-again");
    let a = ModelName::parse("tiny:test").unwrap();
    let blob = store.blob_path(&common::manifest().sha256);
    std::fs::create_dir_all(blob.parent().unwrap()).unwrap();
    std::fs::write(&blob, b"the sealed file").unwrap();
    let loads = Arc::new(AtomicUsize::new(0));
    let loader = common::CountingLoader { loads: loads.clone(), delay: Duration::ZERO };
    let m = Models::new(store.clone(), Arc::new(loader), 1, KeepAlive::For(Duration::from_secs(300)));
    assert!(matches!(last_event(m.submit(&a, ask(), None).expect("the model loads")), Some(Event::Done(_))));
    // The same file, the manifest written again: the engine stays.
    let mut again = common::manifest();
    again.modified_at += 3600;
    store.write_manifest(&a, &again).unwrap();
    assert!(matches!(last_event(m.submit(&a, ask(), None).expect("served")), Some(Event::Done(_))));
    assert_eq!(loads.load(Ordering::SeqCst), 1, "a pull that left the file as it was loaded the model again");
    // The engine idle, as after any answer: a busy one would refuse a second
    // load beside it under the cap of one.
    let t = std::time::Instant::now();
    while m.running().iter().any(|r| r.busy || r.queue > 0) {
        assert!(t.elapsed() < Duration::from_secs(10), "the engine never went idle");
        std::thread::sleep(Duration::from_millis(10));
    }
    // The file removed and written again, past the file system's clock step.
    std::fs::remove_file(&blob).unwrap();
    std::thread::sleep(Duration::from_millis(50));
    std::fs::write(&blob, b"the sealed file").unwrap();
    again.modified_at += 3600;
    store.write_manifest(&a, &again).unwrap();
    let rx = m.submit(&a, ask(), None).unwrap_or_else(|r| panic!("refused: {} {}", r.status, r.message));
    assert!(matches!(last_event(rx), Some(Event::Done(_))));
    assert_eq!(loads.load(Ordering::SeqCst), 2, "the engine read from the removed file still serves");
    let _ = std::fs::remove_dir_all(store.root());
}
