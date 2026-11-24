//! The pull of a model made of many files (Inkling 2 Mini), against a fake
//! Hub on 127.0.0.1. The library pins an index; the index pins every file.
//! A whole set is registered and its local file reads; one wrong digest in
//! the index, a path that leaves the model's directory, and an entry whose
//! pins are not filled each leave no manifest; a second pull fetches nothing
//! again; `rm` removes the files.

use axum::extract::{Path, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::IntoResponse;
use llvq_runtime::library::{Configs, Entry, Kind, ModelName};
use llvq_runtime::store::{untar, FileIndex, IndexFile, Store};
use sha2::Digest;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

const REVISION: &str = "89abcdef0123456789abcdef0123456789abcdef";

fn digest(b: &[u8]) -> String {
    sha2::Sha256::digest(b).iter().map(|x| format!("{x:02x}")).collect()
}

fn bytes(seed: u32, n: usize) -> Vec<u8> {
    (0..n as u32).map(|i| (i.wrapping_add(seed).wrapping_mul(2_654_435_761) >> 13) as u8).collect()
}

/// One tar entry header, ustar, written by hand: the reader is checked
/// against the format, not against itself.
fn tar_header(name: &str, size: usize, kind: u8) -> [u8; 512] {
    let mut h = [0u8; 512];
    h[..name.len()].copy_from_slice(name.as_bytes());
    h[100..107].copy_from_slice(b"0000644");
    h[108..115].copy_from_slice(b"0000000");
    h[116..123].copy_from_slice(b"0000000");
    h[124..135].copy_from_slice(format!("{size:011o}").as_bytes());
    h[136..147].copy_from_slice(b"00000000000");
    h[156] = kind;
    h[257..263].copy_from_slice(b"ustar\0");
    h[263..265].copy_from_slice(b"00");
    h[148..156].copy_from_slice(b"        ");
    let sum: u32 = h.iter().map(|&b| b as u32).sum();
    h[148..155].copy_from_slice(format!("{sum:06o}\0").as_bytes());
    h
}

fn tar(entries: &[(&str, Option<&[u8]>)]) -> Vec<u8> {
    let mut out = Vec::new();
    for (name, body) in entries {
        match body {
            None => out.extend_from_slice(&tar_header(name, 0, b'5')),
            Some(b) => {
                out.extend_from_slice(&tar_header(name, b.len(), b'0'));
                out.extend_from_slice(b);
                out.resize(out.len().div_ceil(512) * 512, 0);
            }
        }
    }
    out.extend_from_slice(&[0u8; 1024]);
    out
}

/// The repository: two layers as archives, the rest as plain files.
fn repo_files() -> Vec<(String, Vec<u8>, Option<String>)> {
    let mut v = vec![
        ("config.json".to_string(), br#"{"fake": true}"#.to_vec(), None),
        ("model.safetensors".to_string(), bytes(1, 2_500_000), None),
        ("tokenizer.json".to_string(), b"{}".to_vec(), None),
        ("tables/embed.q4g64.llvqraw".to_string(), bytes(2, 70_000), None),
        ("tables/unembed.q4g64.llvqraw".to_string(), bytes(3, 70_001), None),
    ];
    for l in 0..2 {
        let (e0, e1) = (bytes(10 + l, 3_000), bytes(20 + l, 513));
        let t = tar(&[
            (&format!("L{l:02}/"), None),
            (&format!("L{l:02}/VERIFIED"), Some(b"2026-11-21".as_slice())),
            (&format!("L{l:02}/codes/e000.llvqrec"), Some(&e0)),
            (&format!("L{l:02}/codes/e001.llvqrec"), Some(&e1)),
        ]);
        v.push((format!("layers/L{l:02}.tar"), t, Some("layers".to_string())));
    }
    v
}

fn index_of(files: &[(String, Vec<u8>, Option<String>)]) -> FileIndex {
    let local = serde_json::json!({
        "config": "config.json", "checkpoint": "model.safetensors", "layers": "layers",
        "tokenizer": "tokenizer.json", "tables": "tables", "context": 4096
    });
    FileIndex {
        schema: 1,
        family: "inkling2".into(),
        files: files
            .iter()
            .map(|(p, b, u)| IndexFile { path: p.clone(), size: b.len() as u64, sha256: digest(b), unpack: u.clone() })
            .collect(),
        local: local.as_object().unwrap().clone(),
    }
}

struct Hub {
    files: HashMap<String, Vec<u8>>,
    gets: AtomicUsize,
}

/// Serves `files` at `/<org>/<repo>/resolve/<rev>/<path>`, honouring
/// `Range: bytes=N-`, and counts the requests.
fn hub(files: HashMap<String, Vec<u8>>) -> (String, Arc<Hub>) {
    async fn file(
        State(h): State<Arc<Hub>>,
        Path((_, _, rev, path)): Path<(String, String, String, String)>,
        hd: HeaderMap,
    ) -> axum::response::Response {
        h.gets.fetch_add(1, Ordering::SeqCst);
        let Some(body) = h.files.get(&path).filter(|_| rev == REVISION) else {
            return StatusCode::NOT_FOUND.into_response();
        };
        match hd.get("range").and_then(|v| v.to_str().ok()).and_then(|r| r.strip_prefix("bytes=")) {
            Some(r) => {
                let from: usize = r.trim_end_matches('-').parse().unwrap();
                (StatusCode::PARTIAL_CONTENT, body[from..].to_vec()).into_response()
            }
            None => (StatusCode::OK, body.clone()).into_response(),
        }
    }
    let state = Arc::new(Hub { files, gets: AtomicUsize::new(0) });
    let s2 = state.clone();
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        rt.block_on(async move {
            let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            tx.send(l.local_addr().unwrap()).unwrap();
            let app = axum::Router::new().route("/{org}/{repo}/resolve/{rev}/{*path}", axum::routing::get(file)).with_state(s2);
            axum::serve(l, app).await.unwrap();
        });
    });
    (format!("http://{}", rx.recv().unwrap()), state)
}

/// The hub for `index` over the repository's files, and the entry pinning it.
fn setup(index: &FileIndex) -> (String, Arc<Hub>, Entry) {
    let text = serde_json::to_vec_pretty(index).unwrap();
    let mut files: HashMap<String, Vec<u8>> = repo_files().into_iter().map(|(p, b, _)| (p, b)).collect();
    files.insert("llvq-files.json".into(), text.clone());
    let (endpoint, h) = hub(files);
    let e = Entry {
        name: "inkling2".into(),
        tag: "mini".into(),
        aliases: vec![],
        names: vec![],
        kind: Kind::Files,
        repo: "org/inkling".into(),
        revision: REVISION.into(),
        file: "llvq-files.json".into(),
        size: text.len() as u64,
        sha256: digest(&text),
        download_size: None,
        parameter_size: "26.1B".into(),
        bits_per_param: "2.473".into(),
        configs: Configs { cuda: None, metal: None, cpu: None, cuda_batch: None, cuda_eager: None },
    };
    (endpoint, h, e)
}

fn store(tag: &str) -> Store {
    let dir = std::env::temp_dir().join(format!("llvq-pullfiles-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    Store::at(dir)
}

#[test]
fn a_file_set_is_pulled_unpacked_and_registered_and_rm_removes_it() {
    let (endpoint, h, e) = setup(&index_of(&repo_files()));
    let s = store("whole");
    let name = ModelName::parse("inkling2:mini").unwrap();
    let m = s.pull(&name, &e, &endpoint, &mut |_| {}).unwrap();
    assert_eq!(m.family, "inkling2");
    assert_eq!(m.sha256, e.sha256);
    let total: u64 = repo_files().iter().map(|(_, b, _)| b.len() as u64).sum();
    assert_eq!(m.size, total);
    let dir = s.model_dir(&e.sha256);
    assert_eq!(std::path::Path::new(&m.file), dir.join("inkling.json"));
    // Every plain file at its digest; every archive unpacked, then gone.
    for (p, b, u) in repo_files() {
        match u {
            None => assert_eq!(std::fs::read(dir.join(&p)).unwrap(), b, "{p}"),
            Some(_) => assert!(!dir.join(&p).exists(), "{p} is kept after unpacking"),
        }
    }
    for l in 0..2u32 {
        let codes = dir.join(format!("layers/L{l:02}/codes"));
        assert_eq!(std::fs::read(codes.join("e000.llvqrec")).unwrap(), bytes(10 + l, 3_000));
        assert_eq!(std::fs::read(codes.join("e001.llvqrec")).unwrap(), bytes(20 + l, 513));
        assert!(dir.join(format!("layers/L{l:02}/VERIFIED")).exists());
    }
    // The engine's local file reads, under the registered name.
    let local = llvq_runtime::inkling::LocalModel::read(std::path::Path::new(&m.file)).unwrap();
    assert_eq!(local.name, "inkling2:mini");
    assert_eq!(local.checkpoint, dir.join("model.safetensors"));
    assert_eq!(local.context, Some(4096));
    assert_eq!(s.get(&name).unwrap(), Some(m));

    // A second pull checks what is there and fetches nothing but the index's
    // check, which reads the blob on disk.
    let before = h.gets.load(Ordering::SeqCst);
    s.pull(&name, &e, &endpoint, &mut |_| {}).unwrap();
    assert_eq!(h.gets.load(Ordering::SeqCst), before, "a whole set is fetched again");

    assert!(s.remove(&name).unwrap());
    assert!(!dir.exists(), "rm leaves the model's files");
    assert!(!s.blob_path(&e.sha256).exists());
    let _ = std::fs::remove_dir_all(s.root());
}

#[test]
fn one_wrong_digest_in_the_index_registers_nothing() {
    let mut idx = index_of(&repo_files());
    let f = idx.files.iter_mut().find(|f| f.path == "model.safetensors").unwrap();
    f.sha256.replace_range(0..1, if f.sha256.starts_with('0') { "1" } else { "0" });
    let (endpoint, _, e) = setup(&idx);
    let s = store("baddigest");
    let name = ModelName::parse("inkling2:mini").unwrap();
    let err = s.pull(&name, &e, &endpoint, &mut |_| {}).unwrap_err();
    let msg = format!("{err:#}");
    assert!(msg.contains("model.safetensors") && msg.contains("nothing was registered"), "{msg}");
    assert_eq!(s.get(&name).unwrap(), None);
    let dir = s.model_dir(&e.sha256);
    assert!(!dir.join("model.safetensors").exists());
    assert!(!dir.join("model.safetensors-partial").exists());
    assert!(!dir.join("inkling.json").exists());
    let _ = std::fs::remove_dir_all(s.root());
}

#[test]
fn an_index_or_an_archive_that_leaves_the_model_directory_is_refused() {
    for bad in ["../escape", "/abs", "a/../../b", "a//b", "./x", "inkling.json"] {
        let mut idx = index_of(&repo_files());
        idx.files[0].path = bad.into();
        let (endpoint, _, e) = setup(&idx);
        let s = store("badpath");
        let n = ModelName::parse("inkling2:mini").unwrap();
        let err = format!("{:#}", s.pull(&n, &e, &endpoint, &mut |_| {}).unwrap_err());
        assert!(err.contains("not a plain relative path"), "{bad}: {err}");
        assert_eq!(s.get(&n).unwrap(), None);
        let _ = std::fs::remove_dir_all(s.root());
    }
    let mut idx = index_of(&repo_files());
    idx.local.insert("checkpoint".into(), serde_json::json!("../../etc/passwd"));
    assert!(idx.check(&setup(&index_of(&repo_files())).2).unwrap_err().contains("checkpoint"));

    // The archive reader: a name that climbs out, a link, a cut archive.
    let to = std::env::temp_dir().join(format!("llvq-untar-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&to);
    let climb = tar(&[("../x", Some(b"no".as_slice()))]);
    assert!(untar(&mut climb.as_slice(), &to).unwrap_err().to_string().contains("not a plain relative path"));
    let mut link = tar(&[("a", Some(b"x".as_slice()))]);
    link[..512].copy_from_slice(&tar_header("a", 0, b'2'));
    assert!(untar(&mut link.as_slice(), &to).unwrap_err().to_string().contains("entry type"));
    let whole = tar(&[("d/f", Some(&bytes(5, 4000)))]);
    assert!(untar(&mut &whole[..1500], &to).is_err(), "a cut archive unpacks");
    assert_eq!(untar(&mut whole.as_slice(), &to).unwrap(), 1);
    assert_eq!(std::fs::read(to.join("d/f")).unwrap(), bytes(5, 4000));
    let _ = std::fs::remove_dir_all(&to);
}

/// An archive made by the system's `tar`, as the layer archives of the card
/// jobs are, unpacks like the hand-made one. Skipped where no `tar` runs.
#[test]
fn an_archive_of_the_system_tar_unpacks() {
    let base = std::env::temp_dir().join(format!("llvq-systar-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    let src = base.join("src");
    std::fs::create_dir_all(src.join("L03/codes")).unwrap();
    std::fs::write(src.join("L03/VERIFIED"), "ok").unwrap();
    std::fs::write(src.join("L03/codes/e127.llvqrec"), bytes(7, 9_999)).unwrap();
    let out = base.join("L03.tar");
    let st = std::process::Command::new("tar")
        .env("COPYFILE_DISABLE", "1")
        .args(["--format", "ustar", "-cf"])
        .arg(&out)
        .arg("-C")
        .arg(&src)
        .arg("L03")
        .status();
    if !st.is_ok_and(|s| s.success()) {
        eprintln!("no system tar: skipped");
        return;
    }
    let to = base.join("to");
    let n = untar(&mut std::io::BufReader::new(std::fs::File::open(&out).unwrap()), &to).unwrap();
    assert_eq!(n, 2);
    assert_eq!(std::fs::read(to.join("L03/codes/e127.llvqrec")).unwrap(), bytes(7, 9_999));
    let _ = std::fs::remove_dir_all(&base);
}

#[test]
fn an_unpublished_entry_is_refused_by_name_and_never_fetched() {
    let (endpoint, h, mut e) = setup(&index_of(&repo_files()));
    e.repo = String::new();
    e.revision = String::new();
    e.sha256 = String::new();
    e.size = 0;
    let s = store("unpublished");
    let n = ModelName::parse("inkling2:mini").unwrap();
    let err = format!("{:#}", s.pull(&n, &e, &endpoint, &mut |_| {}).unwrap_err());
    assert!(err.contains("inkling2:mini") && err.contains("not published"), "{err}");
    assert_eq!(h.gets.load(Ordering::SeqCst), 0);
    assert!(!s.root().exists() || s.get(&n).unwrap().is_none());
    // The library's own entry, as compiled in, is refused the same way while
    // its pins are empty.
    let lib = llvq_runtime::library::find(&n).unwrap();
    if lib.repo.is_empty() {
        assert!(s.pull(&n, lib, &endpoint, &mut |_| {}).is_err());
        assert_eq!(h.gets.load(Ordering::SeqCst), 0);
    }
}
