//! The store's pull, against a fake Hub on 127.0.0.1 that serves one file
//! with range requests. The gate of stage R0 that needs no network: a file
//! whose digest matches is registered; a library entry whose digest is
//! mutated is refused and leaves no manifest, no blob and no partial file;
//! an interrupted pull resumes where it stopped.

use axum::extract::Path;
use axum::http::{HeaderMap, StatusCode};
use axum::response::IntoResponse;
use llvq_runtime::library::{Configs, Entry, ModelName};
use llvq_runtime::store::{sha256_file, PullEvent, Store};
use sha2::Digest;

const REVISION: &str = "0123456789abcdef0123456789abcdef01234567";

fn content() -> Vec<u8> {
    (0..3_000_000u32).map(|i| (i.wrapping_mul(2_654_435_761) >> 13) as u8).collect()
}

fn digest(b: &[u8]) -> String {
    sha2::Sha256::digest(b).iter().map(|x| format!("{x:02x}")).collect()
}

/// Serves `content()` at `/<org>/<repo>/resolve/<rev>/<file>`, honouring
/// `Range: bytes=N-`. Returns the endpoint.
fn hub() -> String {
    async fn file(Path((_, _, rev, _)): Path<(String, String, String, String)>, h: HeaderMap) -> axum::response::Response {
        if rev != REVISION {
            return StatusCode::NOT_FOUND.into_response();
        }
        let body = content();
        match h.get("range").and_then(|v| v.to_str().ok()).and_then(|r| r.strip_prefix("bytes=")) {
            Some(r) => {
                let from: usize = r.trim_end_matches('-').parse().unwrap();
                (StatusCode::PARTIAL_CONTENT, body[from..].to_vec()).into_response()
            }
            None => (StatusCode::OK, body).into_response(),
        }
    }
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        rt.block_on(async move {
            let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            tx.send(l.local_addr().unwrap()).unwrap();
            let app = axum::Router::new().route("/{org}/{repo}/resolve/{rev}/{file}", axum::routing::get(file));
            axum::serve(l, app).await.unwrap();
        });
    });
    format!("http://{}", rx.recv().unwrap())
}

fn entry(sha256: String) -> Entry {
    Entry {
        name: "fake".into(),
        tag: "1b".into(),
        aliases: vec![],
        repo: "org/fake".into(),
        revision: REVISION.into(),
        file: "fake.bin".into(),
        size: content().len() as u64,
        sha256,
        parameter_size: "1B".into(),
        bits_per_param: "2.7".into(),
        configs: Configs { cuda: None, metal: Some("qwen3-4b-tetra-e4.json".into()), cpu: None },
    }
}

fn store(tag: &str) -> Store {
    let dir = std::env::temp_dir().join(format!("llvq-pull-test-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    Store::at(dir)
}

fn files(s: &Store) -> Vec<String> {
    let mut out = Vec::new();
    for d in ["blobs", "manifests/fake"] {
        if let Ok(r) = std::fs::read_dir(s.root().join(d)) {
            out.extend(r.map(|e| e.unwrap().file_name().to_string_lossy().into_owned()));
        }
    }
    out.sort();
    out
}

#[test]
fn a_matching_file_is_registered_and_a_mutated_digest_leaves_nothing() {
    let endpoint = hub();
    let name = ModelName::parse("fake:1b").unwrap();
    let good = digest(&content());

    let s = store("good");
    let mut seen = Vec::new();
    let m = s.pull(&name, &entry(good.clone()), &endpoint, &mut |e| seen.push(e)).unwrap();
    assert_eq!(m.sha256, good);
    assert_eq!(sha256_file(&s.blob_path(&good), |_| {}).unwrap(), good);
    assert_eq!(s.get(&name).unwrap(), Some(m));
    assert!(seen.contains(&PullEvent::Status("success".into())));
    assert!(seen.iter().any(|e| matches!(e, PullEvent::Progress { completed, total, .. } if completed == total)));

    // One hex digit changed in the library: refused, nothing left behind.
    let mut bad = good.clone();
    bad.replace_range(0..1, if bad.starts_with('0') { "1" } else { "0" });
    let s = store("bad");
    let err = s.pull(&name, &entry(bad), &endpoint, &mut |_| {}).unwrap_err();
    assert!(format!("{err:#}").contains("nothing was registered"), "{err:#}");
    assert!(files(&s).is_empty(), "left behind: {:?}", files(&s));
    assert_eq!(s.get(&name).unwrap(), None);
    for d in ["good", "bad"] {
        let _ = std::fs::remove_dir_all(store(d).root());
    }
}

#[test]
fn an_interrupted_pull_resumes_and_a_present_blob_is_checked_again() {
    let endpoint = hub();
    let name = ModelName::parse("fake:1b").unwrap();
    let good = digest(&content());
    let s = store("resume");
    std::fs::create_dir_all(s.root().join("blobs")).unwrap();
    let partial = s.root().join("blobs").join(format!("sha256-{good}-partial"));
    std::fs::write(&partial, &content()[..1_234_567]).unwrap();
    let mut first = None;
    s.pull(&name, &entry(good.clone()), &endpoint, &mut |e| {
        if let (None, PullEvent::Progress { completed, .. }) = (&first, &e) {
            first = Some(*completed);
        }
    })
    .unwrap();
    assert_eq!(first, Some(1_234_567), "the pull must start where the partial file ends");
    assert!(!partial.exists());
    assert_eq!(sha256_file(&s.blob_path(&good), |_| {}).unwrap(), good);

    // A blob already there is hashed again, not trusted: a corrupted one is
    // removed and the pull refused.
    s.remove(&name).unwrap();
    let mut b = content();
    b[17] ^= 1;
    std::fs::write(s.blob_path(&good), &b).unwrap();
    assert!(s.pull(&name, &entry(good.clone()), &endpoint, &mut |_| {}).is_err());
    assert!(!s.blob_path(&good).exists());
    assert_eq!(s.get(&name).unwrap(), None);
    let _ = std::fs::remove_dir_all(s.root());
}

#[test]
fn import_checks_a_local_copy_and_rm_keeps_a_blob_another_name_holds() {
    let name = ModelName::parse("fake:1b").unwrap();
    let good = digest(&content());
    let s = store("import");
    let src = s.root().with_extension("src.bin");
    std::fs::write(&src, content()).unwrap();
    assert!(s.import(&name, &entry(digest(b"other")), &src, &mut |_| {}).is_err());
    assert_eq!(s.get(&name).unwrap(), None);
    s.import(&name, &entry(good.clone()), &src, &mut |_| {}).unwrap();
    let copy = ModelName::parse("fake:copy").unwrap();
    assert!(s.copy(&name, &copy).unwrap());
    assert!(s.remove(&name).unwrap());
    assert!(s.blob_path(&good).exists(), "fake:copy still names the blob");
    assert!(s.remove(&copy).unwrap());
    assert!(!s.blob_path(&good).exists(), "no name holds the blob any more");
    assert!(!s.remove(&copy).unwrap());
    assert!(src.exists(), "the source of an import is never touched");
    std::fs::remove_file(src).unwrap();
    let _ = std::fs::remove_dir_all(s.root());
}

/// Serves `content()` slowly, in 256 KiB chunks 20 ms apart, so that two
/// pulls overlap. Returns the endpoint.
fn slow_hub() -> String {
    async fn file() -> axum::response::Response {
        use futures_util::StreamExt;
        let chunks: Vec<Vec<u8>> = content().chunks(256 << 10).map(<[u8]>::to_vec).collect();
        let s = futures_util::stream::iter(chunks).then(|c| async move {
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            Ok::<_, std::convert::Infallible>(c)
        });
        let mut r = axum::body::Body::from_stream(s).into_response();
        r.headers_mut().insert("content-length", content().len().into());
        r
    }
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build().unwrap();
        rt.block_on(async move {
            let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            tx.send(l.local_addr().unwrap()).unwrap();
            let app = axum::Router::new().route("/{org}/{repo}/resolve/{rev}/{file}", axum::routing::get(file));
            axum::serve(l, app).await.unwrap();
        });
    });
    format!("http://{}", rx.recv().unwrap())
}

#[test]
fn two_pulls_of_one_file_at_once_register_the_file_whole() {
    let endpoint = slow_hub();
    let good = digest(&content());
    let s = store("concurrent");
    let root = s.root().to_path_buf();
    // Two processes, as two `llvq pull` without a server would be: two stores
    // on one directory, no lock in common but the file system's.
    let pulls: Vec<_> = ["fake:1b", "fake:two"]
        .into_iter()
        .map(|n| {
            let (root, endpoint, good) = (root.clone(), endpoint.clone(), good.clone());
            std::thread::spawn(move || {
                Store::at(root).pull(&ModelName::parse(n).unwrap(), &entry(good), &endpoint, &mut |_| {})
            })
        })
        .collect();
    let ok = pulls.into_iter().map(|p| p.join().unwrap()).filter(Result::is_ok).count();
    assert_eq!(ok, 2, "both pulls must succeed, one after the other");
    let blob = s.blob_path(&good);
    assert_eq!(std::fs::metadata(&blob).unwrap().len(), content().len() as u64, "the registered blob is not the file");
    assert_eq!(sha256_file(&blob, |_| {}).unwrap(), good);
    let _ = std::fs::remove_dir_all(s.root());
}

#[test]
fn an_imported_blob_does_not_change_with_its_source() {
    use std::io::{Seek, Write};
    let name = ModelName::parse("fake:1b").unwrap();
    let good = digest(&content());
    let s = store("import-copy");
    std::fs::create_dir_all(s.root()).unwrap();
    let src = s.root().join("src.bin");
    std::fs::write(&src, content()).unwrap();
    s.import(&name, &entry(good.clone()), &src, &mut |_| {}).unwrap();
    // The source rewritten in place, as a re-seal to the same path would.
    let mut f = std::fs::OpenOptions::new().write(true).open(&src).unwrap();
    f.seek(std::io::SeekFrom::Start(17)).unwrap();
    f.write_all(&[!content()[17]]).unwrap();
    drop(f);
    assert_eq!(sha256_file(&s.blob_path(&good), |_| {}).unwrap(), good, "the registered blob changed with its source");
    let _ = std::fs::remove_dir_all(s.root());
}

#[test]
fn a_hub_that_stops_sending_fails_the_pull_instead_of_hanging_it() {
    use std::io::{Read, Write};
    let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let endpoint = format!("http://{}", l.local_addr().unwrap());
    std::thread::spawn(move || {
        let mut held = Vec::new();
        for c in l.incoming() {
            let mut c = c.unwrap();
            let mut buf = [0u8; 4096];
            let _ = c.read(&mut buf);
            let _ = c.write_all(b"HTTP/1.1 200 OK\r\ncontent-length: 3000000\r\n\r\nabc");
            held.push(c);
        }
    });
    let s = store("stall").read_timeout(std::time::Duration::from_millis(300));
    let good = digest(&content());
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let r = s.pull(&ModelName::parse("fake:1b").unwrap(), &entry(good), &endpoint, &mut |_| {});
        let _ = tx.send(r.map(|_| ()).map_err(|e| format!("{e:#}")));
    });
    match rx.recv_timeout(std::time::Duration::from_secs(10)) {
        Ok(r) => assert!(r.is_err(), "a pull of 3 bytes out of 3 MB succeeded"),
        Err(_) => panic!("the pull still hangs 10 s after the hub stopped sending"),
    }
}
