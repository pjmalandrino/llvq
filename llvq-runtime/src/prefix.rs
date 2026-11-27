//! The prefix cache on disk (plan item P7, `docs/plan-debit-inkling2-mini.md`):
//! the model's cache after the static block of a prompt (the system prompt and
//! the tools), saved once, and loaded when a later prompt starts with the same
//! tokens, in a new conversation or after a restart.
//!
//! The cache is saved at `b`, the largest multiple of the prompt piece at most
//! the end of the static block, so the read after a load cuts the prompt where
//! a read from position 0 cuts it, and its logits are those of the read from 0,
//! bit for bit. Nothing is saved under [`MIN_TOKENS`].
//!
//! One file a prefix, in `<store>/prefix/`, mode 0600, written to a temporary
//! file and renamed. Its name is `<key id>-<b>-<tokens sha256, 32 hex>.llvqprefix`.
//! Its layout, every integer little-endian:
//!
//! ```text
//! magic "LLVQPFX\0" | u32 header length | header, JSON ([`Header`])
//! | u64 n | n x u32 token (the tokens 0..b)
//! | the model's cache state (`llvq_llm::inkling::snapshot` for Inkling 2)
//! | digest of every byte before it (32 bytes)
//! ```
//!
//! The digest is the sha256 of the sha256 of each 64 MiB chunk of those bytes,
//! in order ([`file_digest`]): the chunks are hashed on several threads, so a
//! 1.4 GB file is checked in a fraction of the time one sha256 stream takes.
//!
//! The header carries the [`Key`]: the format version, the model's digest, the
//! running executable's sha256, and every setting that changes the bits of a
//! read. A file is loaded only when its key is this server's, field for field,
//! its tokens are the prompt's first `b`, and its sha256 holds. Anything else
//! is refused by name in the server's log and the prompt is read as without
//! the cache. Files past the byte cap go least recently used first.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// The version of the file format above.
pub const FORMAT: u32 = 1;
/// The shortest prefix saved, in tokens.
pub const MIN_TOKENS: usize = 1024;
/// The default byte cap of the directory.
pub const DEFAULT_MAX_BYTES: u64 = 4 << 30;
/// The extension of a prefix file.
pub const SUFFIX: &str = "llvqprefix";
const MAGIC: &[u8; 8] = b"LLVQPFX\0";
/// The largest header read.
const MAX_HEADER: u32 = 1 << 20;
/// The extension of a file being written.
const PART: &str = "part";
/// How old a file being written is before [`PrefixCache::evict`] takes it for
/// the leftover of a write that never ended.
const STALE_PART: std::time::Duration = std::time::Duration::from_secs(3600);

/// What `llvq serve --prefix-cache` and `--prefix-cache-max` set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Options {
    /// `Some(true)` or `Some(false)` when the switch was given; `None`: on
    /// for Metal, off elsewhere.
    pub enabled: Option<bool>,
    pub max_bytes: u64,
}

impl Default for Options {
    fn default() -> Self {
        Options { enabled: None, max_bytes: DEFAULT_MAX_BYTES }
    }
}

static OPTIONS: OnceLock<Options> = OnceLock::new();

/// Set the options of this process, once, before any model loads.
pub fn set_options(o: Options) {
    let _ = OPTIONS.set(o);
}

/// The options of this process: those [`set_options`] set, or the defaults.
pub fn options() -> Options {
    OPTIONS.get().copied().unwrap_or_default()
}

/// `on` or `off`.
pub fn parse_switch(v: &str) -> Result<bool, String> {
    match v {
        "on" => Ok(true),
        "off" => Ok(false),
        o => Err(format!("--prefix-cache {o:?}: on or off")),
    }
}

/// A size in bytes: a whole number, with an optional binary suffix K, M, G or T.
pub fn parse_size(v: &str) -> Result<u64, String> {
    let bad = || format!("--prefix-cache-max {v:?}: a whole number of bytes, with K, M, G or T (binary) if wanted, such as 4G");
    let t = v.trim();
    let (num, shift) = match t.char_indices().last() {
        Some((i, c)) if c.is_ascii_alphabetic() => {
            let shift = match c.to_ascii_uppercase() {
                'K' => 10,
                'M' => 20,
                'G' => 30,
                'T' => 40,
                _ => return Err(bad()),
            };
            (&t[..i], shift)
        }
        _ => (t, 0),
    };
    let n: u64 = num.parse().map_err(|_| bad())?;
    let bytes = n.checked_shl(shift).filter(|b| b >> shift == n).ok_or_else(bad)?;
    if bytes == 0 {
        return Err(format!("--prefix-cache-max {v:?}: a cap of 0 bytes; turn the cache off with --prefix-cache off"));
    }
    Ok(bytes)
}

/// Everything a saved state depends on besides its tokens. Two servers load
/// each other's files only when their keys are equal, field for field.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Key {
    /// [`FORMAT`].
    pub format: u32,
    pub family: String,
    /// The sha256 of the model's local file, as its manifest holds it.
    pub model_sha256: String,
    /// What identifies the files that local file names: the local file alone
    /// does not, since files sealed again keep their paths and a pulled set
    /// writes the same local file for two releases. For Inkling 2,
    /// `inkling::files_id`.
    pub files_sha256: String,
    /// The sha256 of the running executable: another build may read a prompt
    /// with other kernels.
    pub build_sha256: String,
    /// `metal`, `cuda` or `cpu`.
    pub backend: String,
    /// `fused` or `candle` on a GPU, `cpu` on the CPU.
    pub stack: String,
    /// Tokens of a prompt one forward call reads.
    pub prompt_piece: usize,
    /// Query rows of one attention score tensor.
    pub score_rows: usize,
    /// Positions past its window a bounded sliding slot keeps.
    pub kv_slack: usize,
    /// Where the attention's score bias is built: `host` or `card`. On the CPU
    /// the host is the device, and `host` is what it builds.
    pub bias: String,
    /// The arm of `llvq serve --attn-window`: `full` or `narrow`. A file
    /// written before the field reads as `full`, and a key of `full` writes
    /// no field, so its bytes and its [`Key::id`] are the ones it had before
    /// the switch.
    #[serde(default = "full_window", skip_serializing_if = "is_full_window")]
    pub attn_window: String,
}

/// The arm a key without `attn_window` was written under.
pub const FULL_WINDOW: &str = "full";

fn full_window() -> String {
    FULL_WINDOW.into()
}

fn is_full_window(w: &str) -> bool {
    w == FULL_WINDOW
}

impl Key {
    /// 16 hex digits of the sha256 of the key: the first part of a file name.
    pub fn id(&self) -> String {
        let j = serde_json::to_vec(self).expect("a key serializes");
        hex(&Sha256::digest(&j))[..16].to_string()
    }

    /// The fields whose values differ between two keys, by name.
    pub fn differing(&self, other: &Key) -> Vec<String> {
        let (a, b) = (serde_json::to_value(self).expect("key"), serde_json::to_value(other).expect("key"));
        let (Some(a), Some(b)) = (a.as_object(), b.as_object()) else { return vec!["key".into()] };
        let mut names: Vec<&String> = a.keys().chain(b.keys()).collect();
        names.sort();
        names.dedup();
        names
            .into_iter()
            .filter(|k| a.get(*k) != b.get(*k))
            .map(|k| format!("{k} ({} here, {} in the file)", a.get(k).map_or("absent".into(), |v| v.to_string()), b.get(k).map_or("absent".into(), |v| v.to_string())))
            .collect()
    }
}

/// The header of a prefix file.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Header {
    pub key: Key,
    /// `b`, the tokens the state holds.
    pub tokens: usize,
    pub tokens_sha256: String,
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// The sha256 of tokens, as their little-endian bytes.
pub fn tokens_sha256(ids: &[u32]) -> String {
    let mut h = Sha256::new();
    for t in ids {
        h.update(t.to_le_bytes());
    }
    hex(&h.finalize())
}

/// The length of the longest common prefix of two id lists.
pub fn common_prefix(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

/// A file the lookup found: its path and its `b`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Found {
    pub path: PathBuf,
    pub b: usize,
}

impl Found {
    pub fn name(&self) -> String {
        self.path.file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default()
    }
}

/// A file whose key, tokens and sha256 were checked: where its state starts
/// and how many bytes it holds.
#[derive(Debug)]
pub struct Verified {
    path: PathBuf,
    state_at: u64,
    state_len: u64,
}

/// Bytes of one chunk of [`file_digest`].
const CHUNK: u64 = 64 << 20;
/// Threads [`file_digest`] hashes on at most.
const DIGEST_THREADS: usize = 6;

/// The digests one thread of [`file_digest`] made, by chunk index.
type ChunkDigests = std::io::Result<Vec<(usize, [u8; 32])>>;

/// The sha256 of the list of the sha256 of each [`CHUNK`] of the first `len`
/// bytes of `path`, the chunks hashed on up to [`DIGEST_THREADS`] threads.
pub fn file_digest(path: &Path, len: u64) -> std::io::Result<[u8; 32]> {
    let n = len.div_ceil(CHUNK).max(1) as usize;
    let threads = std::thread::available_parallelism().map_or(1, |t| t.get()).clamp(1, DIGEST_THREADS).min(n);
    let hash_chunk = |file: &mut std::fs::File, i: usize, buf: &mut [u8]| -> std::io::Result<[u8; 32]> {
        let start = i as u64 * CHUNK;
        let mut left = CHUNK.min(len - start.min(len));
        file.seek(SeekFrom::Start(start))?;
        let mut h = Sha256::new();
        while left > 0 {
            let k = (buf.len() as u64).min(left) as usize;
            file.read_exact(&mut buf[..k])?;
            h.update(&buf[..k]);
            left -= k as u64;
        }
        Ok(h.finalize().into())
    };
    let parts: Vec<ChunkDigests> = std::thread::scope(|sc| {
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                sc.spawn(move || -> ChunkDigests {
                    let mut file = std::fs::File::open(path)?;
                    let mut buf = vec![0u8; 1 << 20];
                    (t..n).step_by(threads).map(|i| Ok((i, hash_chunk(&mut file, i, &mut buf)?))).collect()
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().expect("a digest thread panicked")).collect()
    });
    let mut digests = vec![[0u8; 32]; n];
    for p in parts {
        for (i, d) in p? {
            digests[i] = d;
        }
    }
    let mut h = Sha256::new();
    for d in &digests {
        h.update(d);
    }
    Ok(h.finalize().into())
}

/// The prefix cache of one loaded model.
#[derive(Debug, Clone)]
pub struct PrefixCache {
    dir: PathBuf,
    max_bytes: u64,
    key: Key,
}

impl PrefixCache {
    /// The cache in `dir` (`<store>/prefix`), its files capped at `max_bytes`.
    pub fn new(dir: PathBuf, max_bytes: u64, key: Key) -> Self {
        PrefixCache { dir, max_bytes, key }
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn key(&self) -> &Key {
        &self.key
    }

    pub fn max_bytes(&self) -> u64 {
        self.max_bytes
    }

    /// Tokens of a prompt one forward call reads: the grain of a save point.
    pub fn piece(&self) -> usize {
        self.key.prompt_piece
    }

    /// Where a static block that ends at `l` is saved: the largest multiple
    /// of the piece at most `l`; `None` under [`MIN_TOKENS`].
    pub fn save_point(&self, l: usize) -> Option<usize> {
        let p = self.piece();
        if p == 0 {
            return None;
        }
        Some(l / p * p).filter(|&b| b >= MIN_TOKENS)
    }

    fn file_name(&self, b: usize, sha: &str) -> String {
        format!("{}-{b:08}-{}.{SUFFIX}", self.key.id(), &sha[..32])
    }

    /// The files of this key whose tokens are the first `b` of `ids`, by name
    /// (the tokens' sha256), with `b > keep` and `fits(b)`, longest first.
    pub fn find(&self, ids: &[u32], keep: usize, fits: &dyn Fn(usize) -> bool) -> Vec<Found> {
        let Ok(rd) = std::fs::read_dir(&self.dir) else { return Vec::new() };
        let id = self.key.id();
        let mut out: Vec<Found> = rd
            .filter_map(|e| e.ok())
            .filter_map(|e| {
                let name = e.file_name().to_string_lossy().into_owned();
                let stem = name.strip_suffix(&format!(".{SUFFIX}"))?;
                let mut parts = stem.splitn(3, '-');
                let (k, b, sha) = (parts.next()?, parts.next()?, parts.next()?);
                let b: usize = b.parse().ok()?;
                (k == id && b > keep && b <= ids.len() && fits(b) && tokens_sha256(&ids[..b])[..32] == *sha)
                    .then(|| Found { path: e.path(), b })
            })
            .collect();
        out.sort_by_key(|f| std::cmp::Reverse(f.b));
        out
    }

    /// Check `f` for the prompt `ids` without touching any state: the file's
    /// key is this cache's, field for field; its tokens are the first `b` of
    /// `ids`; the sha256 of its bytes holds. The error names what failed.
    pub fn verify(&self, f: &Found, ids: &[u32]) -> anyhow::Result<Verified> {
        let file = std::fs::File::open(&f.path)?;
        let size = file.metadata()?.len();
        let mut r = BufReader::with_capacity(1 << 22, file);
        let mut magic = [0u8; 8];
        r.read_exact(&mut magic).map_err(|e| anyhow::anyhow!("magic: {e}"))?;
        anyhow::ensure!(&magic == MAGIC, "not a prefix file: magic {magic:?}");
        let mut b4 = [0u8; 4];
        r.read_exact(&mut b4)?;
        let hlen = u32::from_le_bytes(b4);
        anyhow::ensure!(hlen <= MAX_HEADER, "a header of {hlen} bytes");
        let mut hb = vec![0u8; hlen as usize];
        r.read_exact(&mut hb).map_err(|e| anyhow::anyhow!("header: {e}"))?;
        let h: Header = serde_json::from_slice(&hb).map_err(|e| anyhow::anyhow!("header: {e}"))?;
        let diff = self.key.differing(&h.key);
        anyhow::ensure!(diff.is_empty(), "the key differs: {}", diff.join(", "));
        anyhow::ensure!(h.tokens == f.b, "the header holds {} tokens, the name says {}", h.tokens, f.b);
        anyhow::ensure!(f.b <= ids.len(), "{} tokens saved, the prompt has {}", f.b, ids.len());
        anyhow::ensure!(h.tokens_sha256 == tokens_sha256(&ids[..f.b]), "the header's tokens are not the prompt's");
        let mut b8 = [0u8; 8];
        r.read_exact(&mut b8)?;
        let n = u64::from_le_bytes(b8);
        anyhow::ensure!(n == f.b as u64, "{n} tokens stored, the header says {}", f.b);
        let mut tb = vec![0u8; f.b * 4];
        r.read_exact(&mut tb).map_err(|e| anyhow::anyhow!("tokens: {e}"))?;
        let stored = tb.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]));
        if let Some(at) = stored.zip(&ids[..f.b]).position(|(s, t)| s != *t) {
            anyhow::bail!("stored token {at} is not the prompt's");
        }
        let state_at = 8 + 4 + hlen as u64 + 8 + 4 * f.b as u64;
        anyhow::ensure!(size >= state_at + 32, "the file ends at {size} bytes, before its state");
        let state_len = size - state_at - 32;
        // The digest of every byte before the last 32, against those 32.
        let mut file = r.into_inner();
        file.seek(SeekFrom::Start(size - 32))?;
        let mut trailer = [0u8; 32];
        file.read_exact(&mut trailer).map_err(|e| anyhow::anyhow!("digest: {e}"))?;
        let digest = file_digest(&f.path, size - 32)?;
        anyhow::ensure!(digest == trailer, "its sha256 does not hold: the file is damaged");
        Ok(Verified { path: f.path.clone(), state_at, state_len })
    }

    /// Hand the state of a verified file to `load`, which must read all of
    /// it; then mark the file used (its modification time), for the eviction.
    pub fn load(&self, v: &Verified, load: &mut dyn FnMut(&mut dyn Read) -> anyhow::Result<()>) -> anyhow::Result<()> {
        let mut file = std::fs::File::open(&v.path)?;
        file.seek(SeekFrom::Start(v.state_at))?;
        let mut r = BufReader::with_capacity(1 << 22, file).take(v.state_len);
        load(&mut r)?;
        anyhow::ensure!(r.limit() == 0, "the model read {} of the state's {} bytes", v.state_len - r.limit(), v.state_len);
        if let Ok(f) = std::fs::File::options().write(true).open(&v.path) {
            let _ = f.set_modified(std::time::SystemTime::now());
        }
        Ok(())
    }

    /// Write the state `save` writes as the prefix `ids[..b]`: to a temporary
    /// file, then renamed; then the oldest files go until the directory is
    /// under the cap. The path and the file's bytes.
    pub fn save(&self, ids: &[u32], b: usize, save: &mut dyn FnMut(&mut dyn Write) -> anyhow::Result<()>) -> anyhow::Result<(PathBuf, u64)> {
        anyhow::ensure!(b <= ids.len(), "save at {b} of {} tokens", ids.len());
        make_dir(&self.dir)?;
        let sha = tokens_sha256(&ids[..b]);
        let path = self.dir.join(self.file_name(b, &sha));
        let tmp = self.dir.join(format!(
            ".{}.{}-{}.{PART}",
            self.file_name(b, &sha),
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0)
        ));
        let written = (|| -> anyhow::Result<u64> {
            let f = create_private(&tmp)?;
            let mut w = BufWriter::with_capacity(1 << 22, f);
            let header = serde_json::to_vec(&Header { key: self.key.clone(), tokens: b, tokens_sha256: sha.clone() })?;
            w.write_all(MAGIC)?;
            w.write_all(&(header.len() as u32).to_le_bytes())?;
            w.write_all(&header)?;
            w.write_all(&(b as u64).to_le_bytes())?;
            let toks: Vec<u8> = ids[..b].iter().flat_map(|t| t.to_le_bytes()).collect();
            w.write_all(&toks)?;
            save(&mut w)?;
            let mut f = w.into_inner().map_err(|e| anyhow::anyhow!("{}: {}", tmp.display(), e.error()))?;
            let n = f.stream_position()?;
            // The digest of what was written, read back from the page cache on several threads.
            let digest = file_digest(&tmp, n)?;
            f.write_all(&digest)?;
            Ok(n + 32)
        })();
        let n = match written {
            Ok(n) => n,
            Err(e) => {
                let _ = std::fs::remove_file(&tmp);
                return Err(e);
            }
        };
        if let Err(e) = std::fs::rename(&tmp, &path) {
            let _ = std::fs::remove_file(&tmp);
            anyhow::bail!("rename {}: {e}", tmp.display());
        }
        self.evict()?;
        Ok((path, n))
    }

    /// Remove the least recently used prefix files until the directory holds
    /// at most the cap, and first the temporary files a write that never ended
    /// left more than [`STALE_PART`] ago. The names of the prefix files removed.
    pub fn evict(&self) -> anyhow::Result<Vec<String>> {
        let now = std::time::SystemTime::now();
        for e in std::fs::read_dir(&self.dir)?.filter_map(|e| e.ok()) {
            let stale = e.file_name().to_string_lossy().ends_with(&format!(".{PART}"))
                && e.metadata().ok().and_then(|m| m.modified().ok()).and_then(|t| now.duration_since(t).ok()).is_some_and(|d| d > STALE_PART);
            if stale {
                let _ = std::fs::remove_file(e.path());
            }
        }
        let mut files: Vec<(std::time::SystemTime, u64, PathBuf)> = std::fs::read_dir(&self.dir)?
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().ends_with(&format!(".{SUFFIX}")))
            .filter_map(|e| {
                let m = e.metadata().ok()?;
                Some((m.modified().ok()?, m.len(), e.path()))
            })
            .collect();
        files.sort();
        let mut total: u64 = files.iter().map(|f| f.1).sum();
        let mut gone = Vec::new();
        for (_, len, p) in files {
            if total <= self.max_bytes {
                break;
            }
            std::fs::remove_file(&p)?;
            total -= len;
            gone.push(p.file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default());
        }
        Ok(gone)
    }
}

#[cfg(unix)]
fn create_private(p: &Path) -> std::io::Result<std::fs::File> {
    use std::os::unix::fs::OpenOptionsExt;
    std::fs::OpenOptions::new().write(true).create_new(true).mode(0o600).open(p)
}

#[cfg(not(unix))]
fn create_private(p: &Path) -> std::io::Result<std::fs::File> {
    std::fs::OpenOptions::new().write(true).create_new(true).open(p)
}

#[cfg(unix)]
fn make_dir(p: &Path) -> std::io::Result<()> {
    use std::os::unix::fs::DirBuilderExt;
    std::fs::DirBuilder::new().recursive(true).mode(0o700).create(p)
}

#[cfg(not(unix))]
fn make_dir(p: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(p)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key() -> Key {
        Key {
            format: FORMAT,
            family: "inkling2".into(),
            model_sha256: "a".repeat(64),
            files_sha256: "f".repeat(64),
            build_sha256: "b".repeat(64),
            backend: "metal".into(),
            stack: "fused".into(),
            prompt_piece: 512,
            score_rows: 128,
            kv_slack: 1024,
            bias: "host".into(),
            attn_window: FULL_WINDOW.into(),
        }
    }

    fn dir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("llvq-prefix-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    fn ids(n: usize, seed: u32) -> Vec<u32> {
        (0..n as u32).map(|i| i.wrapping_mul(2_654_435_761).wrapping_add(seed) % 50_000).collect()
    }

    /// A state of `n` bytes, and a reader that takes exactly that many back.
    fn state(n: usize) -> Vec<u8> {
        (0..n).map(|i| (i * 7 % 251) as u8).collect()
    }

    fn save(c: &PrefixCache, ids: &[u32], b: usize, s: &[u8]) -> (PathBuf, u64) {
        c.save(ids, b, &mut |w| Ok(w.write_all(s)?)).unwrap()
    }

    fn read_back(c: &PrefixCache, f: &Found, ids: &[u32], n: usize) -> anyhow::Result<Vec<u8>> {
        let v = c.verify(f, ids)?;
        let mut got = Vec::new();
        c.load(&v, &mut |r| {
            got = vec![0u8; n];
            r.read_exact(&mut got)?;
            Ok(())
        })?;
        Ok(got)
    }

    #[test]
    fn the_save_point_is_the_last_piece_boundary_of_the_static_block() {
        let c = PrefixCache::new(dir("point"), DEFAULT_MAX_BYTES, key());
        assert_eq!(c.save_point(8791), Some(8704));
        assert_eq!(c.save_point(8704), Some(8704));
        assert_eq!(c.save_point(8703), Some(8192));
        assert_eq!(c.save_point(1536), Some(1536));
        assert_eq!(c.save_point(1535), Some(1024));
        assert_eq!(c.save_point(1023), None, "under 1,024 tokens nothing is saved");
        assert_eq!(c.save_point(0), None);
    }

    #[test]
    fn a_saved_prefix_is_found_verified_and_loaded_back() {
        let d = dir("round");
        let c = PrefixCache::new(d.clone(), DEFAULT_MAX_BYTES, key());
        let p = ids(3000, 1);
        let s = state(10_000);
        let (path, n) = save(&c, &p, 2048, &s);
        assert_eq!(std::fs::metadata(&path).unwrap().len(), n);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(std::fs::metadata(&path).unwrap().permissions().mode() & 0o777, 0o600);
        }
        let found = c.find(&p, 0, &|_| true);
        assert_eq!(found, vec![Found { path: path.clone(), b: 2048 }]);
        assert_eq!(read_back(&c, &found[0], &p, s.len()).unwrap(), s);
        // what the memory keeps already covers it, or the caller's rule refuses it: not found
        assert!(c.find(&p, 2048, &|_| true).is_empty());
        assert!(c.find(&p, 0, &|b| b != 2048).is_empty());
        // another prompt with the same first 2,048 tokens finds it; one that differs inside does not
        let mut q = p[..2048].to_vec();
        q.extend(ids(500, 9));
        assert_eq!(c.find(&q, 0, &|_| true).len(), 1);
        let mut r = p.clone();
        r[100] ^= 1;
        assert!(c.find(&r, 0, &|_| true).is_empty());
        // the longest first
        save(&c, &p, 1024, &s);
        let b: Vec<usize> = c.find(&p, 0, &|_| true).iter().map(|f| f.b).collect();
        assert_eq!(b, vec![2048, 1024]);
        // the model must read the whole state
        let v = c.verify(&found[0], &p).unwrap();
        let e = c.load(&v, &mut |r| {
            let mut x = [0u8; 10];
            r.read_exact(&mut x)?;
            Ok(())
        });
        assert!(format!("{:#}", e.unwrap_err()).contains("read 10 of"));
        let _ = std::fs::remove_dir_all(&d);
    }

    /// The sha256 of the bytes before the trailer, written over a file whose
    /// bytes were changed on purpose: a damage the integrity check cannot see.
    fn rehash(path: &Path) {
        let mut b = std::fs::read(path).unwrap();
        let n = b.len() - 32;
        let d = file_digest(path, n as u64).unwrap();
        b[n..].copy_from_slice(&d);
        std::fs::write(path, b).unwrap();
    }

    /// The digest of several chunks is the chunks' own, in order: every chunk
    /// counts, a byte moved between two chunks counts.
    #[test]
    fn the_digest_covers_every_chunk_in_order() {
        let d = dir("digest");
        std::fs::create_dir_all(&d).unwrap();
        let p = d.join("f");
        let len = 2 * CHUNK as usize + 12_345;
        let mut b: Vec<u8> = (0..len).map(|i| (i * 31 % 253) as u8).collect();
        std::fs::write(&p, &b).unwrap();
        let mut want = Sha256::new();
        for c in b.chunks(CHUNK as usize) {
            want.update(Sha256::digest(c));
        }
        let want: [u8; 32] = want.finalize().into();
        assert_eq!(file_digest(&p, len as u64).unwrap(), want);
        for at in [0, CHUNK as usize - 1, CHUNK as usize, len - 1] {
            b[at] ^= 1;
            std::fs::write(&p, &b).unwrap();
            assert_ne!(file_digest(&p, len as u64).unwrap(), want, "a byte at {at} not covered");
            b[at] ^= 1;
        }
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn a_damaged_file_is_refused_by_name() {
        let d = dir("damage");
        let c = PrefixCache::new(d.clone(), DEFAULT_MAX_BYTES, key());
        let p = ids(2000, 2);
        let s = state(5000);
        let (path, _) = save(&c, &p, 1024, &s);
        let f = Found { path: path.clone(), b: 1024 };
        let clean = std::fs::read(&path).unwrap();
        // one byte of the state
        let mut b = clean.clone();
        let at = b.len() - 32 - 100;
        b[at] ^= 0x10;
        std::fs::write(&path, &b).unwrap();
        let e = format!("{:#}", c.verify(&f, &p).unwrap_err());
        assert!(e.contains("sha256 does not hold"), "{e}");
        // cut short
        std::fs::write(&path, &clean[..clean.len() - 7]).unwrap();
        assert!(c.verify(&f, &p).is_err());
        // a stored token that is not the prompt's, the file's sha256 made to hold again
        let mut b = clean.clone();
        let hlen = u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize;
        let last = 12 + hlen + 8 + 4 * 1023;
        b[last] ^= 1;
        std::fs::write(&path, &b).unwrap();
        rehash(&path);
        let e = format!("{:#}", c.verify(&f, &p).unwrap_err());
        assert!(e.contains("stored token 1023"), "{e}");
        std::fs::write(&path, &clean).unwrap();
        assert!(c.verify(&f, &p).is_ok());
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn a_file_of_another_key_is_refused_naming_the_field() {
        let d = dir("key");
        let a = PrefixCache::new(d.clone(), DEFAULT_MAX_BYTES, key());
        let p = ids(2000, 3);
        let (path, _) = save(&a, &p, 1024, &state(100));
        let f = Found { path, b: 1024 };
        let k = key();
        let variants: Vec<(&str, Key)> = vec![
            ("format", Key { format: FORMAT + 1, ..k.clone() }),
            ("family", Key { family: "qwen3".into(), ..k.clone() }),
            ("model_sha256", Key { model_sha256: "c".repeat(64), ..k.clone() }),
            ("files_sha256", Key { files_sha256: "e".repeat(64), ..k.clone() }),
            ("build_sha256", Key { build_sha256: "d".repeat(64), ..k.clone() }),
            ("backend", Key { backend: "cuda".into(), ..k.clone() }),
            ("stack", Key { stack: "candle".into(), ..k.clone() }),
            ("prompt_piece", Key { prompt_piece: 256, ..k.clone() }),
            ("score_rows", Key { score_rows: 64, ..k.clone() }),
            ("kv_slack", Key { kv_slack: 256, ..k.clone() }),
            ("bias", Key { bias: "card".into(), ..k.clone() }),
            ("attn_window", Key { attn_window: "narrow".into(), ..k.clone() }),
        ];
        // The key of `full` writes no `attn_window`: count the fields on a key that writes them all.
        let every = Key { attn_window: "narrow".into(), ..k.clone() };
        assert_eq!(variants.len(), serde_json::to_value(&every).unwrap().as_object().unwrap().len(), "a field of the key is not tested");
        for (field, other) in variants {
            let b = PrefixCache::new(d.clone(), DEFAULT_MAX_BYTES, other);
            assert!(b.find(&p, 0, &|_| true).is_empty(), "{field}: another key's file found by name");
            let e = format!("{:#}", b.verify(&f, &p).unwrap_err());
            assert!(e.contains("the key differs") && e.contains(field), "{field}: {e}");
        }
        assert!(a.verify(&f, &p).is_ok());
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn the_least_recently_used_files_go_past_the_cap() {
        let d = dir("evict");
        let big = PrefixCache::new(d.clone(), DEFAULT_MAX_BYTES, key());
        let p = ids(3000, 4);
        let s = state(9000);
        let (p1, n1) = save(&big, &p, 1024, &s);
        std::thread::sleep(std::time::Duration::from_millis(20));
        let (p2, n2) = save(&big, &p, 1536, &s);
        std::thread::sleep(std::time::Duration::from_millis(20));
        // room for p1 and p3 (p2 and 512 more tokens), not for all three
        let c = PrefixCache::new(d.clone(), n1 + n2 + 2048, key());
        // p1 used: p2 is now the oldest
        read_back(&c, &Found { path: p1.clone(), b: 1024 }, &p, s.len()).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(20));
        let (p3, _) = save(&c, &p, 2048, &s);
        assert!(p1.exists() && !p2.exists() && p3.exists(), "{:?}", std::fs::read_dir(&d).unwrap().collect::<Vec<_>>());
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn a_write_that_never_ended_is_removed_once_stale() {
        let d = dir("part");
        let c = PrefixCache::new(d.clone(), DEFAULT_MAX_BYTES, key());
        let p = ids(2000, 5);
        let (kept, _) = save(&c, &p, 1024, &state(100));
        let (old, new) = (d.join(format!(".x.1-1.{PART}")), d.join(format!(".y.1-2.{PART}")));
        std::fs::write(&old, b"half").unwrap();
        std::fs::write(&new, b"half").unwrap();
        let long_ago = std::time::SystemTime::now() - STALE_PART - std::time::Duration::from_secs(60);
        std::fs::File::options().write(true).open(&old).unwrap().set_modified(long_ago).unwrap();
        assert!(c.evict().unwrap().is_empty(), "no prefix file is past the cap");
        assert!(!old.exists(), "the stale temporary file stayed");
        assert!(new.exists(), "a write that may still run was removed");
        assert!(kept.exists());
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn the_key_of_the_full_window_has_the_bytes_it_had_before_the_switch() {
        // The JSON the key of `key()` serialized to before `attn_window` existed, field for field.
        let before = format!(
            r#"{{"format":{FORMAT},"family":"inkling2","model_sha256":"{}","files_sha256":"{}","build_sha256":"{}","backend":"metal","stack":"fused","prompt_piece":512,"score_rows":128,"kv_slack":1024,"bias":"host"}}"#,
            "a".repeat(64),
            "f".repeat(64),
            "b".repeat(64)
        );
        let k = key();
        assert_eq!(serde_json::to_string(&k).unwrap(), before);
        assert_eq!(k.id(), hex(&Sha256::digest(before.as_bytes()))[..16].to_string());
        // A header written before the field reads as `full`, and is the same key.
        let old: Key = serde_json::from_str(&before).unwrap();
        assert_eq!(old.attn_window, FULL_WINDOW);
        assert_eq!(old, k);
        // `narrow` writes the field, and its key is another.
        let narrow = Key { attn_window: "narrow".into(), ..k.clone() };
        let j = serde_json::to_string(&narrow).unwrap();
        assert!(j.ends_with(r#","attn_window":"narrow"}"#), "{j}");
        assert_ne!(narrow.id(), k.id());
        assert_eq!(serde_json::from_str::<Key>(&j).unwrap(), narrow);
        let d = k.differing(&narrow);
        assert_eq!(d.len(), 1, "{d:?}");
        assert!(d[0].starts_with("attn_window "), "{d:?}");
    }

    #[test]
    fn options_parse_or_refuse_by_name() {
        assert_eq!(parse_size("4G"), Ok(4 << 30));
        assert_eq!(parse_size("512m"), Ok(512 << 20));
        assert_eq!(parse_size("1000"), Ok(1000));
        assert!(parse_size("0").unwrap_err().contains("off"));
        assert!(parse_size("4GB").is_err());
        assert!(parse_size("-1").is_err());
        assert!(parse_size("99999999999T").is_err());
        assert_eq!(parse_switch("on"), Ok(true));
        assert_eq!(parse_switch("off"), Ok(false));
        assert!(parse_switch("yes").is_err());
    }
}
