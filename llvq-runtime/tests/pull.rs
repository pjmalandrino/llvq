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
