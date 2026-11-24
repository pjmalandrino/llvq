//! The model store under `LLVQ_HOME`, laid out like Ollama's:
//!
//! ```text
//! <home>/blobs/sha256-<hex>             the sealed file, named by its digest
//! <home>/manifests/<name>/<tag>.json    name:tag -> blob, Hub origin, configs
//! <home>/models/sha256-<hex>/           the files of a file set, under its index's digest
//! ```
//!
//! A manifest is written only after the blob's size and sha256 match the
//! library. A download that does not match leaves nothing registered.

use crate::library::{Configs, Entry, Kind, ModelName};
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
    /// How long a pull waits for the next bytes before it fails.
    read_timeout: std::time::Duration,
}

/// The default of [`Store::read_timeout`]: a Hub that sends nothing for a
/// minute is gone.
pub const READ_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(60);

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
        Store { root: root.into(), read_timeout: READ_TIMEOUT }
    }

    /// The same store, with another wait for a stalled download.
    pub fn read_timeout(mut self, d: std::time::Duration) -> Self {
        self.read_timeout = d;
        self
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

    /// Hold the lock of one blob: a pull or an import of it, in this
    /// process or another, waits until the lock is released. Two writers
    /// appending to one partial file would register a file that is neither,
    /// and a pull still appending after another renamed the partial would
    /// write into the registered blob. The lock file stays: removing it while
    /// another process waits on it would let a third take a second lock.
    fn lock_blob(&self, sha256: &str, report: &mut dyn FnMut(PullEvent)) -> anyhow::Result<std::fs::File> {
        let dir = self.root.join("locks");
        std::fs::create_dir_all(&dir)?;
        let f = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(dir.join(format!("sha256-{sha256}.lock")))?;
        if let Err(std::fs::TryLockError::WouldBlock) = f.try_lock() {
            report(PullEvent::Status("waiting for another pull of this file".into()));
            f.lock()?;
        }
        Ok(f)
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
            // The files a file set was pulled into, when it was one.
            match std::fs::remove_dir_all(self.model_dir(&m.sha256)) {
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
    /// checked again rather than trusted. An entry of [`Kind::Files`] goes on
    /// to fetch every file its index lists ([`Self::pull_files`]).
    pub fn pull(
        &self,
        n: &ModelName,
        e: &Entry,
        endpoint: &str,
        report: &mut dyn FnMut(PullEvent),
    ) -> anyhow::Result<Manifest> {
        e.published().map_err(anyhow::Error::msg)?;
        report(PullEvent::Status("pulling manifest".into()));
        let _lock = self.lock_blob(&e.sha256, report)?;
        let blob = self.blob_path(&e.sha256);
        if blob.exists() {
            if let Err(err) = self.verify(&blob, e, report) {
                std::fs::remove_file(&blob)?;
                return Err(err.context("the blob on disk was removed; pull again"));
            }
        } else {
            let digest = format!("sha256:{}", e.sha256);
            let total = e.size;
            let pin = Pin { path: &e.file, size: e.size, sha256: &e.sha256 };
            self.fetch(endpoint, e, &pin, &blob, &mut |done| {
                report(PullEvent::Progress { digest: digest.clone(), total, completed: done })
            })?;
        }
        match e.kind {
            Kind::Sealed => self.register(n, e, report),
            Kind::Files => self.pull_files(n, e, endpoint, report),
        }
    }

    /// Fetch one file of `e`'s repository at its revision into `dest`,
    /// through `<dest>-partial`, resumed when that file exists. Size and
    /// sha256 are checked before the rename; a mismatch removes the partial
    /// file. `on_bytes` gets the bytes of this file held so far, first where
    /// the partial file ends.
    fn fetch(&self, endpoint: &str, e: &Entry, pin: &Pin, dest: &Path, on_bytes: &mut dyn FnMut(u64)) -> anyhow::Result<()> {
        std::fs::create_dir_all(dest.parent().expect("a destination has a directory"))?;
        let name = dest.file_name().expect("a destination has a name").to_string_lossy();
        let partial = dest.with_file_name(format!("{name}-partial"));
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
            if done > pin.size {
                std::fs::remove_file(&partial)?;
                h = Sha256::new();
                done = 0;
            }
        }
        let mut out = std::fs::OpenOptions::new().create(true).append(true).open(&partial)?;
        let url = format!("{}/{}/resolve/{}/{}", endpoint.trim_end_matches('/'), e.repo, e.revision, pin.path);
        on_bytes(done);
        if done < pin.size {
            let tls = native_tls::TlsConnector::new()?;
            // Without a read timeout, a connection that stops sending holds
            // the pull, and the blob's lock, forever.
            let agent = ureq::AgentBuilder::new()
                .tls_connector(std::sync::Arc::new(tls))
                .timeout_connect(std::time::Duration::from_secs(30))
                .timeout_read(self.read_timeout)
                .build();
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
            let mut th = Throttle { total: pin.size, last: done };
            loop {
                let k = body.read(&mut buf)?;
                if k == 0 {
                    break;
                }
                done += k as u64;
                anyhow::ensure!(done <= pin.size, "GET {url}: more than the {} bytes pinned; nothing was registered", pin.size);
                h.update(&buf[..k]);
                out.write_all(&buf[..k])?;
                if th.due(done) {
                    on_bytes(done);
                }
            }
            out.flush()?;
        }
        drop(out);
        let got = hex(&h.finalize());
        if done != pin.size || got != pin.sha256 {
            std::fs::remove_file(&partial)?;
            anyhow::bail!(
                "{url}: {done} bytes with sha256 {got}, the pin is {} bytes with sha256 {}. \
                 The download was removed and nothing was registered.",
                pin.size,
                pin.sha256
            );
        }
        std::fs::rename(&partial, dest)?;
        Ok(())
    }

    /// The directory a file set is pulled into, named by its index's digest.
    pub fn model_dir(&self, sha256: &str) -> PathBuf {
        self.root.join("models").join(format!("sha256-{sha256}"))
    }

    /// The second half of a [`Kind::Files`] pull, the index already a
    /// checked blob: every file it lists into [`Self::model_dir`], each
    /// checked by size and sha256, archives unpacked; then the local file
    /// `inkling.json` the engine loads, and the manifest last.
    ///
    /// A plain file already there is hashed again. An archive is unpacked
    /// once, removed, and remembered by `<path>.unpacked` holding its digest.
    fn pull_files(&self, n: &ModelName, e: &Entry, endpoint: &str, report: &mut dyn FnMut(PullEvent)) -> anyhow::Result<Manifest> {
        let text = std::fs::read_to_string(self.blob_path(&e.sha256))?;
        let index: FileIndex =
            serde_json::from_str(&text).map_err(|err| anyhow::anyhow!("{} index: {err}", e.model_name()))?;
        index.check(e).map_err(|err| anyhow::anyhow!("{} index: {err}", e.model_name()))?;
        let dir = self.model_dir(&e.sha256);
        let total: u64 = index.files.iter().map(|f| f.size).sum();
        let digest = format!("sha256:{}", e.sha256);
        let mut before = 0u64;
        let mut th = Throttle { total, last: 0 };
        for f in &index.files {
            let dest = dir.join(&f.path);
            let pin = Pin { path: &f.path, size: f.size, sha256: &f.sha256 };
            let marker = dest.with_file_name(format!("{}.unpacked", dest.file_name().expect("a path has a name").to_string_lossy()));
            let held = match &f.unpack {
                Some(_) => std::fs::read_to_string(&marker).is_ok_and(|s| s.trim() == f.sha256),
                None => dest.exists() && std::fs::metadata(&dest)?.len() == f.size && sha256_file(&dest, |_| {})? == f.sha256,
            };
            if !held {
                if f.unpack.is_none() && dest.exists() {
                    std::fs::remove_file(&dest)?;
                }
                self.fetch(endpoint, e, &pin, &dest, &mut |done| {
                    if th.due(before + done) {
                        report(PullEvent::Progress { digest: digest.clone(), total, completed: before + done });
                    }
                })?;
                if let Some(into) = &f.unpack {
                    report(PullEvent::Status(format!("unpacking {}", f.path)));
                    let to = dir.join(into);
                    std::fs::create_dir_all(&to)?;
                    untar(&mut std::io::BufReader::new(std::fs::File::open(&dest)?), &to)
                        .map_err(|err| anyhow::anyhow!("{}: {err}", f.path))?;
                    std::fs::write(&marker, &f.sha256)?;
                    std::fs::remove_file(&dest)?;
                }
            }
            before += f.size;
        }
        report(PullEvent::Progress { digest, total, completed: total });
        // The local file the engine reads, its paths relative to `dir`.
        let mut local = index.local.clone();
        local.insert("name".into(), serde_json::Value::String(n.to_string()));
        local.insert("family".into(), serde_json::Value::String(e.name.clone()));
        let file = dir.join("inkling.json");
        std::fs::write(&file, serde_json::to_vec_pretty(&local)?)?;
        crate::inkling::LocalModel::read(&file)?;
        report(PullEvent::Status("writing manifest".into()));
        let m = Manifest {
            size: total,
            file: file.to_str().ok_or_else(|| anyhow::anyhow!("{}: not UTF-8", file.display()))?.to_string(),
            ..Manifest::of(e)
        };
        self.write_manifest(n, &m)?;
        report(PullEvent::Status("success".into()));
        Ok(m)
    }

    /// Register a local copy of `e`'s file as `n`, with no network: the size
    /// and the sha256 are checked as a download's are.
    ///
    /// The file is copied, and the copy is what is checked. A hard link
    /// would share the source's bytes: a re-seal written in place to the
    /// same path would change the registered blob after its check. On APFS
    /// and on btrfs or XFS `std::fs::copy` clones, at no cost in space.
    pub fn import(
        &self,
        n: &ModelName,
        e: &Entry,
        file: &Path,
        report: &mut dyn FnMut(PullEvent),
    ) -> anyhow::Result<Manifest> {
        e.published().map_err(anyhow::Error::msg)?;
        anyhow::ensure!(
            e.kind == Kind::Sealed,
            "{} is a set of files: `pull --from` takes one sealed file. Pull it from the Hub, or serve local files \
             with `llvq serve --local <file.json>`.",
            e.model_name()
        );
        let _lock = self.lock_blob(&e.sha256, report)?;
        let blob = self.blob_path(&e.sha256);
        if blob.exists() {
            if let Err(err) = self.verify(&blob, e, report) {
                std::fs::remove_file(&blob)?;
                return Err(err.context("the blob on disk was removed; import again"));
            }
            return self.register(n, e, report);
        }
        std::fs::create_dir_all(blob.parent().expect("blobs has a parent"))?;
        report(PullEvent::Status(format!("copying {}", file.display())));
        let tmp = blob.with_file_name(format!("sha256-{}-partial", e.sha256));
        std::fs::copy(file, &tmp)?;
        if let Err(err) = self.verify(&tmp, e, report) {
            std::fs::remove_file(&tmp)?;
            return Err(err);
        }
        std::fs::rename(&tmp, &blob)?;
        self.register(n, e, report)
    }
}

/// The Hub endpoint: `HF_ENDPOINT`, or huggingface.co.
pub fn hub_endpoint() -> String {
    std::env::var("HF_ENDPOINT").unwrap_or_else(|_| "https://huggingface.co".to_string())
}

/// One file to fetch: its path in the repository, its size and sha256.
struct Pin<'a> {
    path: &'a str,
    size: u64,
    sha256: &'a str,
}

/// The index of a [`Kind::Files`] entry, `llvq-files.json` in its Hub
/// repository. The library pins the index; the index pins every file.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileIndex {
    pub schema: u32,
    /// The family the files serve; must be the entry's name.
    pub family: String,
    pub files: Vec<IndexFile>,
    /// The local file the engine reads ([`crate::inkling::LocalModel`])
    /// without `name` and `family`, its paths relative to the model's
    /// directory.
    pub local: serde_json::Map<String, serde_json::Value>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IndexFile {
    /// Relative, `/`-separated, no `..`.
    pub path: String,
    pub size: u64,
    pub sha256: String,
    /// A plain tar archive: unpacked into this directory, then removed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unpack: Option<String>,
}

/// Whether `p` is a relative path of plain parts: no root, no `..`, no `.`,
/// no empty part. Every path an index or an archive names goes through it.
pub fn plain_relative(p: &str) -> bool {
    !p.is_empty() && !p.contains('\\') && p.split('/').all(|c| !c.is_empty() && c != "." && c != ".." && !c.contains('\0'))
}

impl FileIndex {
    /// Refuse an index that does not belong to `e` or names a path outside
    /// the model's directory.
    pub fn check(&self, e: &Entry) -> Result<(), String> {
        if self.schema != 1 {
            return Err(format!("schema {}; this llvq reads schema 1", self.schema));
        }
        if self.family != e.name {
            return Err(format!("family {:?}, the library entry is {:?}", self.family, e.name));
        }
        if self.files.is_empty() {
            return Err("no file".into());
        }
        let mut seen = std::collections::HashSet::new();
        for f in &self.files {
            if !plain_relative(&f.path) || f.path == "inkling.json" {
                return Err(format!("{:?} is not a plain relative path the pull may write", f.path));
            }
            if !seen.insert(f.path.as_str()) {
                return Err(format!("{:?} is listed twice", f.path));
            }
            if f.sha256.len() != 64 || !f.sha256.chars().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()) {
                return Err(format!("{}: not a full sha256", f.path));
            }
            if f.unpack.as_deref().is_some_and(|u| !plain_relative(u)) {
                return Err(format!("{}: unpack into {:?}, not a plain relative path", f.path, f.unpack));
            }
        }
        for k in ["name", "family"] {
            if self.local.contains_key(k) {
                return Err(format!("local carries {k:?}, which the pull writes"));
            }
        }
        for k in ["config", "checkpoint", "layers", "tokenizer", "tables"] {
            match self.local.get(k) {
                None => {}
                Some(serde_json::Value::String(s)) if plain_relative(s) => {}
                Some(v) => return Err(format!("local {k} {v} is not a plain relative path")),
            }
        }
        Ok(())
    }
}

/// Unpack a plain tar stream (ustar or GNU, regular files and directories
/// only) into `to`. Any other entry (a link, a long-name or pax record) is
/// refused by its type, and every name must be a plain relative path.
/// Returns the files written.
pub fn untar(r: &mut dyn Read, to: &Path) -> anyhow::Result<usize> {
    let mut files = 0;
    let mut h = [0u8; 512];
    loop {
        r.read_exact(&mut h)?;
        if h.iter().all(|&b| b == 0) {
            return Ok(files);
        }
        let field = |a: usize, b: usize| {
            let f = &h[a..b];
            String::from_utf8_lossy(&f[..f.iter().position(|&c| c == 0).unwrap_or(f.len())]).into_owned()
        };
        let mut name = field(0, 100);
        // POSIX ustar carries a name prefix at 345; GNU's "ustar  " header
        // keeps other fields there.
        if &h[257..263] == b"ustar\0" {
            let prefix = field(345, 500);
            if !prefix.is_empty() {
                name = format!("{prefix}/{name}");
            }
        }
        let size_text = field(124, 136);
        let size = u64::from_str_radix(size_text.trim_matches(|c: char| c == ' ' || c == '\0'), 8)
            .map_err(|_| anyhow::anyhow!("{name}: size {size_text:?} is not octal"))?;
        let rel = name.trim_end_matches('/');
        let rel = rel.strip_prefix("./").unwrap_or(rel);
        let pad = (512 - size % 512) % 512;
        match h[156] {
            b'5' => {
                if !rel.is_empty() && rel != "." {
                    anyhow::ensure!(plain_relative(rel), "{name:?} is not a plain relative path");
                    std::fs::create_dir_all(to.join(rel))?;
                }
                std::io::copy(&mut r.take(size + pad), &mut std::io::sink())?;
            }
            b'0' | 0 => {
                anyhow::ensure!(plain_relative(rel), "{name:?} is not a plain relative path");
                let dest = to.join(rel);
                std::fs::create_dir_all(dest.parent().expect("a file has a directory"))?;
                let mut out = std::fs::File::create(&dest)?;
                let n = std::io::copy(&mut r.take(size), &mut out)?;
                anyhow::ensure!(n == size, "{name}: {n} bytes of {size}, the archive is cut");
                std::io::copy(&mut r.take(pad), &mut std::io::sink())?;
                files += 1;
            }
            t => anyhow::bail!("{name:?}: entry type {:?}; only plain files and directories are unpacked", t as char),
        }
    }
}
