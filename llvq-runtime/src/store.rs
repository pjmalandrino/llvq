//! The model store under `LLVQ_HOME`, laid out like Ollama's:
//!
//! ```text
//! <home>/blobs/sha256-<hex>             the sealed file, named by its digest
//! <home>/manifests/<name>/<tag>.json    name:tag -> blob, Hub origin, configs
//! ```
//!
//! A manifest is written only after the blob's size and sha256 match the
//! library. A download that does not match leaves nothing registered.

use crate::library::{Configs, Entry, ModelName};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

/// What a manifest records. Self-contained: `show`, `ls` and the loader read
/// it without the library, so a copy (`cp`) serves like its source.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Manifest {
    pub schema: u32,
    pub sha256: String,
    pub size: u64,
    pub repo: String,
    pub revision: String,
    pub file: String,
    pub family: String,
    pub parameter_size: String,
    pub bits_per_param: String,
    pub configs: Configs,
    /// Seconds since the epoch.
    pub modified_at: u64,
}

impl Manifest {
    pub fn of(e: &Entry) -> Self {
        Manifest {
            schema: 1,
            sha256: e.sha256.clone(),
            size: e.size,
            repo: e.repo.clone(),
            revision: e.revision.clone(),
            file: e.file.clone(),
            family: e.name.clone(),
            parameter_size: e.parameter_size.clone(),
            bits_per_param: e.bits_per_param.clone(),
            configs: e.configs.clone(),
            modified_at: now_secs(),
        }
    }

    /// `sha256:<hex>`, Ollama's form of a digest.
    pub fn digest(&self) -> String {
        format!("sha256:{}", self.sha256)
    }
}

pub fn now_secs() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

/// What a pull reports as it goes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PullEvent {
    Status(String),
    Progress { digest: String, total: u64, completed: u64 },
}

pub struct Store {
    root: PathBuf,
}

/// `<user home>/.llvq`, or `LLVQ_HOME`.
pub fn default_home() -> anyhow::Result<PathBuf> {
    if let Ok(h) = std::env::var("LLVQ_HOME") {
        anyhow::ensure!(!h.trim().is_empty(), "LLVQ_HOME is set to the empty string");
        return Ok(PathBuf::from(h));
    }
    let home = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .ok_or_else(|| anyhow::anyhow!("no HOME: set LLVQ_HOME"))?;
    Ok(PathBuf::from(home).join(".llvq"))
}

/// Hash a whole file.
pub fn sha256_file(path: &Path, mut on_bytes: impl FnMut(u64)) -> anyhow::Result<String> {
    let mut f = std::fs::File::open(path)?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    let mut done = 0u64;
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
        done += n as u64;
        on_bytes(done);
    }
    Ok(hex(&h.finalize()))
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// Calls `f` at most every 1/200 of `total` bytes, and at the end.
struct Throttle {
    total: u64,
    last: u64,
}

impl Throttle {
    fn due(&mut self, done: u64) -> bool {
        let step = (self.total / 200).max(1 << 20);
        if done >= self.total || done >= self.last + step {
            self.last = done;
            return true;
        }
        false
    }
}

impl Store {
    pub fn at(root: impl Into<PathBuf>) -> Self {
        Store { root: root.into() }
    }

    pub fn from_env() -> anyhow::Result<Self> {
        Ok(Self::at(default_home()?))
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn blob_path(&self, sha256: &str) -> PathBuf {
        self.root.join("blobs").join(format!("sha256-{sha256}"))
    }

    fn manifest_path(&self, n: &ModelName) -> PathBuf {
        self.root.join("manifests").join(&n.name).join(format!("{}.json", n.tag))
    }

    pub fn get(&self, n: &ModelName) -> anyhow::Result<Option<Manifest>> {
        let p = self.manifest_path(n);
        match std::fs::read_to_string(&p) {
            Ok(t) => Ok(Some(serde_json::from_str(&t).map_err(|e| anyhow::anyhow!("{}: {e}", p.display()))?)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(anyhow::anyhow!("{}: {e}", p.display())),
        }
    }

    /// Every registered model, sorted by name.
    pub fn list(&self) -> anyhow::Result<Vec<(ModelName, Manifest)>> {
        let dir = self.root.join("manifests");
        let mut out = Vec::new();
        let names = match std::fs::read_dir(&dir) {
            Ok(r) => r,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(out),
            Err(e) => return Err(e.into()),
        };
        for n in names {
            let n = n?;
            if !n.file_type()?.is_dir() {
                continue;
            }
            let name = n.file_name().to_string_lossy().into_owned();
            for t in std::fs::read_dir(n.path())? {
                let t = t?;
                let file = t.file_name().to_string_lossy().into_owned();
                let Some(tag) = file.strip_suffix(".json") else { continue };
                let Ok(mn) = ModelName::parse(&format!("{name}:{tag}")) else { continue };
                if let Some(m) = self.get(&mn)? {
                    out.push((mn, m));
                }
            }
        }
        out.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(out)
    }

    /// Write a manifest, atomically: a reader sees the old one or the new.
    pub fn write_manifest(&self, n: &ModelName, m: &Manifest) -> anyhow::Result<()> {
        let p = self.manifest_path(n);
        std::fs::create_dir_all(p.parent().expect("a manifest has a directory"))?;
        let tmp = p.with_extension("json.tmp");
        std::fs::write(&tmp, serde_json::to_vec_pretty(m)?)?;
        std::fs::rename(&tmp, &p)?;
        Ok(())
    }

    /// Remove `n`, then its blob if no other manifest names it. False when
    /// `n` is not registered.
    pub fn remove(&self, n: &ModelName) -> anyhow::Result<bool> {
        let Some(m) = self.get(n)? else { return Ok(false) };
        let p = self.manifest_path(n);
        std::fs::remove_file(&p)?;
        if let Some(dir) = p.parent() {
            let _ = std::fs::remove_dir(dir);
        }
        if !self.list()?.iter().any(|(_, o)| o.sha256 == m.sha256) {
            match std::fs::remove_file(self.blob_path(&m.sha256)) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => return Err(e.into()),
            }
        }
        Ok(true)
    }

    /// A second manifest on the same blob. False when `src` is not registered.
    pub fn copy(&self, src: &ModelName, dst: &ModelName) -> anyhow::Result<bool> {
        let Some(mut m) = self.get(src)? else { return Ok(false) };
        m.modified_at = now_secs();
        self.write_manifest(dst, &m)?;
        Ok(true)
    }

    /// Check a blob on disk against `e`, size first, then sha256.
    fn verify(&self, path: &Path, e: &Entry, report: &mut dyn FnMut(PullEvent)) -> anyhow::Result<()> {
        let size = std::fs::metadata(path)?.len();
        anyhow::ensure!(
            size == e.size,
            "{}: {size} bytes, the library pins {} for {}",
            path.display(),
            e.size,
            e.model_name()
        );
        report(PullEvent::Status("verifying sha256 digest".into()));
        let got = sha256_file(path, |_| {})?;
        anyhow::ensure!(
            got == e.sha256,
            "{}: sha256 {got}, the library pins {} for {}. Nothing was registered.",
            path.display(),
            e.sha256,
            e.model_name()
        );
        Ok(())
    }

    fn register(&self, n: &ModelName, e: &Entry, report: &mut dyn FnMut(PullEvent)) -> anyhow::Result<Manifest> {
        report(PullEvent::Status("writing manifest".into()));
        let m = Manifest::of(e);
        self.write_manifest(n, &m)?;
        report(PullEvent::Status("success".into()));
        Ok(m)
    }

    /// Pull `e` from the Hub at its pinned revision and register it as `n`.
    ///
    /// The file streams into `blobs/sha256-<hex>-partial` and is hashed on
    /// the way; a partial file left by an interrupted pull is hashed first,
    /// then resumed with a range request. Size and digest are checked before
    /// the rename, and the manifest after it. A blob already present is
    /// checked again rather than trusted.
    pub fn pull(
        &self,
        n: &ModelName,
        e: &Entry,
        endpoint: &str,
        report: &mut dyn FnMut(PullEvent),
    ) -> anyhow::Result<Manifest> {
        report(PullEvent::Status("pulling manifest".into()));
        let blob = self.blob_path(&e.sha256);
        if blob.exists() {
            if let Err(err) = self.verify(&blob, e, report) {
                std::fs::remove_file(&blob)?;
                return Err(err.context("the blob on disk was removed; pull again"));
            }
            return self.register(n, e, report);
        }
        std::fs::create_dir_all(blob.parent().expect("blobs has a parent"))?;
        let partial = blob.with_file_name(format!("sha256-{}-partial", e.sha256));
        let digest = format!("sha256:{}", e.sha256);
        let mut h = Sha256::new();
        let mut done = 0u64;
        if partial.exists() {
            let mut f = std::fs::File::open(&partial)?;
            let mut buf = vec![0u8; 1 << 20];
            loop {
                let k = f.read(&mut buf)?;
                if k == 0 {
                    break;
                }
                h.update(&buf[..k]);
                done += k as u64;
            }
            if done > e.size {
                std::fs::remove_file(&partial)?;
                h = Sha256::new();
                done = 0;
            }
        }
        let mut out = std::fs::OpenOptions::new().create(true).append(true).open(&partial)?;
        let url = format!("{}/{}/resolve/{}/{}", endpoint.trim_end_matches('/'), e.repo, e.revision, e.file);
        report(PullEvent::Progress { digest: digest.clone(), total: e.size, completed: done });
        if done < e.size {
            let tls = native_tls::TlsConnector::new()?;
            let agent = ureq::AgentBuilder::new().tls_connector(std::sync::Arc::new(tls)).build();
            let mut req = agent.get(&url).set("User-Agent", concat!("llvq/", env!("CARGO_PKG_VERSION")));
            if let Ok(t) = std::env::var("HF_TOKEN") {
                req = req.set("Authorization", &format!("Bearer {t}"));
            }
            if done > 0 {
                req = req.set("Range", &format!("bytes={done}-"));
            }
            let resp = req.call().map_err(|err| anyhow::anyhow!("{err}"))?;
            if done > 0 && resp.status() != 206 {
                // The server ignored the range: start over.
                drop(out);
                std::fs::remove_file(&partial)?;
                out = std::fs::OpenOptions::new().create(true).append(true).open(&partial)?;
                h = Sha256::new();
                done = 0;
            }
            let mut body = resp.into_reader();
            let mut buf = vec![0u8; 1 << 20];
            let mut th = Throttle { total: e.size, last: done };
            loop {
                let k = body.read(&mut buf)?;
                if k == 0 {
                    break;
                }
                done += k as u64;
                anyhow::ensure!(
                    done <= e.size,
                    "GET {url}: more than the {} bytes the library pins; nothing was registered",
                    e.size
                );
                h.update(&buf[..k]);
                out.write_all(&buf[..k])?;
                if th.due(done) {
                    report(PullEvent::Progress { digest: digest.clone(), total: e.size, completed: done });
                }
            }
            out.flush()?;
        }
        drop(out);
        report(PullEvent::Status("verifying sha256 digest".into()));
        let got = hex(&h.finalize());
        if done != e.size || got != e.sha256 {
            std::fs::remove_file(&partial)?;
            anyhow::bail!(
                "{url}: {done} bytes with sha256 {got}, the library pins {} bytes with sha256 {}. \
                 The download was removed and nothing was registered.",
                e.size,
                e.sha256
            );
        }
        std::fs::rename(&partial, &blob)?;
        self.register(n, e, report)
    }

    /// Register a local copy of `e`'s file as `n`, with no network: the size
    /// and the sha256 are checked as a download's are. The file is
    /// hard-linked into the store when it can be, copied otherwise.
    pub fn import(
        &self,
        n: &ModelName,
        e: &Entry,
        file: &Path,
        report: &mut dyn FnMut(PullEvent),
    ) -> anyhow::Result<Manifest> {
        self.verify(file, e, report)?;
        let blob = self.blob_path(&e.sha256);
        if !blob.exists() {
            std::fs::create_dir_all(blob.parent().expect("blobs has a parent"))?;
            if std::fs::hard_link(file, &blob).is_err() {
                report(PullEvent::Status(format!("copying {}", file.display())));
                let tmp = blob.with_file_name(format!("sha256-{}-partial", e.sha256));
                std::fs::copy(file, &tmp)?;
                std::fs::rename(&tmp, &blob)?;
            }
        }
        self.register(n, e, report)
    }
}

/// The Hub endpoint: `HF_ENDPOINT`, or huggingface.co.
pub fn hub_endpoint() -> String {
    std::env::var("HF_ENDPOINT").unwrap_or_else(|_| "https://huggingface.co".to_string())
}
