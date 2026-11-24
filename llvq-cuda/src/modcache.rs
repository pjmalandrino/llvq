//! A disk cache of compiled kernel modules, keyed so a stale one never loads.
//!
//! NVRTC and the driver's JIT compile a module at every process start. For
//! the tree verify pass that is the larger part of a load: 1,127.1 s of the
//! 1,637 running seconds of the large-prefill-ft job went to compiling four
//! tree modules (*measured*, `docs/mesures/large-prefill-ft-2026-11-11.txt`).
//! A served form cannot pay that at every start.
//!
//! What is cached is the driver's cubin of a module (NVRTC's PTX, then the
//! driver's own ptxas through `cuLinkComplete`), the code the card runs. The
//! key names everything that decides those bytes:
//!
//! * the SHA-256 of the kernel text handed to NVRTC, so an edited kernel,
//!   tile define or arm misses;
//! * the NVRTC target (`compute_NN`), so a cubin built for another arch
//!   misses;
//! * NVRTC's version and the driver's version, so a new toolkit or a new
//!   driver, whose ptxas may emit other code, misses;
//! * the card's compute capability, so a cubin for an sm_89 card is never
//!   loaded on an sm_120 one.
//!
//! The file name is a digest of the key, and the file repeats the key in a
//! header line beside the cubin's length and SHA-256. A read checks the three
//! before it returns a byte: a hash collision of names, a truncated write or a
//! corrupted body reads as a miss, and the caller compiles again. The SHA-256
//! is the file's own, so it guards against corruption, not against an edit:
//! anyone who can write the directory can put another cubin under the
//! expected name with a matching header, and it is loaded. The cache is only
//! as trusted as the directory that holds it. A cubin
//! is written to a temporary name and renamed, so a reader never sees half a
//! file.
//!
//! Portable on purpose: the key, the file format, the reads and the refusals
//! are tested on the Mac. The card side (`gpu::Cuda::on_stream_cached`) reads
//! the versions from the driver and NVRTC and calls this module.

use std::path::{Path, PathBuf};

/// The format of a cache file. Bumped when the file layout or the meaning of
/// a key field changes, so every older file misses.
pub const FORMAT: &str = "llvq-modcache v1";

/// The variable that moves the cache: a directory, or `off`.
pub const ENV: &str = "LLVQ_MODULE_CACHE";

/// Everything that decides the bytes of a compiled module.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Key {
    /// SHA-256 of the kernel text handed to NVRTC, lowercase hex.
    pub text_sha256: String,
    /// NVRTC's target, `compute_NN`.
    pub arch: String,
    /// NVRTC's version, (major, minor).
    pub nvrtc: (i32, i32),
    /// The driver's version, as `cuDriverGetVersion` returns it (12080 is
    /// CUDA 12.8).
    pub driver: i32,
    /// The card's compute capability, (major, minor).
    pub sm: (i32, i32),
}

impl Key {
    /// The key as one line, the form the file header carries.
    pub fn line(&self) -> String {
        format!(
            "{FORMAT} text={} arch={} nvrtc={}.{} driver={} sm={}.{}",
            self.text_sha256, self.arch, self.nvrtc.0, self.nvrtc.1, self.driver, self.sm.0, self.sm.1
        )
    }

    /// The file this key is cached in: 32 hex digits of the line's SHA-256.
    pub fn file_name(&self) -> String {
        format!("{}.cubin", &crate::sha256::sha256_hex(self.line().as_bytes())[..32])
    }

    /// The key's fields, refused when one could not name a module.
    pub fn check(&self) -> Result<(), String> {
        let hex = |s: &str| s.len() == 64 && s.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b));
        if !hex(&self.text_sha256) {
            return Err(format!("module cache: the text digest {:?} is not 64 lowercase hex digits", self.text_sha256));
        }
        if self.arch.strip_prefix("compute_").and_then(|n| n.parse::<u32>().ok()).is_none() {
            return Err(format!("module cache: the arch {:?} is not compute_NN", self.arch));
        }
        if self.nvrtc.0 <= 0 || self.driver <= 0 || self.sm.0 <= 0 || self.nvrtc.1 < 0 || self.sm.1 < 0 {
            return Err(format!(
                "module cache: NVRTC {}.{}, driver {}, sm {}.{}: a version the driver did not report",
                self.nvrtc.0, self.nvrtc.1, self.driver, self.sm.0, self.sm.1
            ));
        }
        Ok(())
    }
}

/// A cache file: the key's line, the cubin's length and SHA-256, the cubin.
pub fn encode(key: &Key, cubin: &[u8]) -> Vec<u8> {
    let mut out = format!("{}\ncubin {} {}\n", key.line(), cubin.len(), crate::sha256::sha256_hex(cubin)).into_bytes();
    out.extend_from_slice(cubin);
    out
}

/// The cubin of a cache file, if the file is this key's and whole. `Err`
/// names why it is not, and the caller treats every `Err` as a miss.
pub fn decode(key: &Key, bytes: &[u8]) -> Result<Vec<u8>, String> {
    let mut parts = bytes.splitn(3, |&b| b == b'\n');
    let head = parts.next().ok_or("an empty file")?;
    let sizes = parts.next().ok_or("no cubin line")?;
    let body = parts.next().ok_or("no cubin")?;
    let head = std::str::from_utf8(head).map_err(|_| "a header that is not text")?;
    if head != key.line() {
        return Err(format!("the header names another module: {head:?}"));
    }
    let sizes = std::str::from_utf8(sizes).map_err(|_| "a cubin line that is not text")?;
    let f: Vec<&str> = sizes.split(' ').collect();
    let (len, sha) = match f.as_slice() {
        ["cubin", n, s] => (n.parse::<usize>().map_err(|_| format!("a cubin length {n:?}"))?, *s),
        _ => return Err(format!("a cubin line {sizes:?}")),
    };
    if body.len() != len {
        return Err(format!("{} bytes of cubin, the header says {len}", body.len()));
    }
    if crate::sha256::sha256_hex(body) != sha {
        return Err("the cubin's SHA-256 is not the header's".into());
    }
    Ok(body.to_vec())
}

/// What a lookup found.
#[derive(Debug, PartialEq, Eq)]
pub enum Lookup {
    /// The key's cubin, checked.
    Hit(Vec<u8>),
    /// No usable file, and why: absent, unreadable, or not this key's.
    Miss(String),
}

/// The cache directory.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Dir(pub PathBuf);

impl Dir {
    /// The file `key` lives in under this directory.
    pub fn path_of(&self, key: &Key) -> PathBuf {
        self.0.join(key.file_name())
    }

    /// The key's cubin, or why there is none.
    pub fn lookup(&self, key: &Key) -> Lookup {
        let p = self.path_of(key);
        match std::fs::read(&p) {
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Lookup::Miss(format!("{} absent", p.display())),
            Err(e) => Lookup::Miss(format!("{}: {e}", p.display())),
            Ok(bytes) => match decode(key, &bytes) {
                Ok(c) => Lookup::Hit(c),
                Err(why) => Lookup::Miss(format!("{}: {why}", p.display())),
            },
        }
    }

    /// Write the key's cubin: to a temporary name in the same directory, then
    /// renamed over the final one, so a concurrent reader sees the old file or
    /// the whole new one.
    pub fn store(&self, key: &Key, cubin: &[u8]) -> Result<PathBuf, String> {
        key.check()?;
        std::fs::create_dir_all(&self.0).map_err(|e| format!("{}: {e}", self.0.display()))?;
        let p = self.path_of(key);
        let tmp = self.0.join(format!("{}.{}.tmp", key.file_name(), std::process::id()));
        std::fs::write(&tmp, encode(key, cubin)).map_err(|e| format!("{}: {e}", tmp.display()))?;
        std::fs::rename(&tmp, &p).map_err(|e| {
            let _ = std::fs::remove_file(&tmp);
            format!("{} -> {}: {e}", tmp.display(), p.display())
        })?;
        Ok(p)
    }
}

/// The cache directory from the environment, the lookup handed in so a test
/// sets no variable: `LLVQ_MODULE_CACHE` names one, `off` turns the cache
/// off (`None`), and unset takes `$XDG_CACHE_HOME/llvq/modules`, else
/// `$HOME/.cache/llvq/modules`. An empty value, a relative path, and no home
/// to default to are refused by name.
pub fn dir_from(lookup: impl Fn(&str) -> Option<String>) -> Result<Option<Dir>, String> {
    match lookup(ENV) {
        Some(v) if v == "off" => Ok(None),
        Some(v) if v.trim().is_empty() => {
            Err(format!("{ENV} is empty: name a directory, or write off, or unset it for the default"))
        }
        Some(v) => {
            let p = PathBuf::from(&v);
            if !p.is_absolute() {
                return Err(format!("{ENV}={v:?}: an absolute directory, so two working directories share one cache"));
            }
            Ok(Some(Dir(p)))
        }
        None => {
            let base = match (lookup("XDG_CACHE_HOME"), lookup("HOME")) {
                (Some(x), _) if Path::new(&x).is_absolute() => PathBuf::from(x),
                (_, Some(h)) if Path::new(&h).is_absolute() => PathBuf::from(h).join(".cache"),
                _ => return Err(format!("{ENV} is unset and there is no absolute XDG_CACHE_HOME or HOME to default to")),
            };
            Ok(Some(Dir(base.join("llvq").join("modules"))))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key() -> Key {
        Key {
            text_sha256: crate::sha256::sha256_hex(b"extern \"C\" __global__ void k() {}"),
            arch: "compute_89".into(),
            nvrtc: (12, 4),
            driver: 12080,
            sm: (8, 9),
        }
    }

    fn scratch(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("llvq-modcache-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    #[test]
    fn a_stored_cubin_reads_back_whole() {
        let d = Dir(scratch("roundtrip"));
        let cubin: Vec<u8> = (0..4096u32).map(|i| (i * 7 % 251) as u8).collect();
        assert!(matches!(d.lookup(&key()), Lookup::Miss(w) if w.contains("absent")));
        let p = d.store(&key(), &cubin).unwrap();
        assert_eq!(p, d.path_of(&key()));
        assert_eq!(d.lookup(&key()), Lookup::Hit(cubin));
        let _ = std::fs::remove_dir_all(&d.0);
    }

    /// Every field of the key moves the file and the header: a cubin cached
    /// under one key is a miss under any other, even if a file were planted
    /// under the other key's name.
    #[test]
    fn every_key_field_makes_another_module() {
        let d = Dir(scratch("fields"));
        let k = key();
        d.store(&k, b"cubin bytes").unwrap();
        let others = [
            Key { text_sha256: crate::sha256::sha256_hex(b"another text"), ..k.clone() },
            Key { arch: "compute_80".into(), ..k.clone() },
            Key { nvrtc: (12, 8), ..k.clone() },
            Key { nvrtc: (13, 4), ..k.clone() },
            Key { driver: 12090, ..k.clone() },
            Key { sm: (12, 0), ..k.clone() },
            Key { sm: (8, 6), ..k.clone() },
        ];
        for o in &others {
            assert_ne!(o.line(), k.line(), "{o:?}");
            assert_ne!(o.file_name(), k.file_name(), "{o:?}");
            assert!(matches!(d.lookup(o), Lookup::Miss(_)), "{o:?} must miss");
            // a file of k's planted under o's name is refused by its header
            std::fs::copy(d.path_of(&k), d.path_of(o)).unwrap();
            match d.lookup(o) {
                Lookup::Miss(w) => assert!(w.contains("another module"), "{w}"),
                Lookup::Hit(_) => panic!("{o:?}: a planted file of another key loaded"),
            }
        }
        assert_eq!(d.lookup(&k), Lookup::Hit(b"cubin bytes".to_vec()));
        let _ = std::fs::remove_dir_all(&d.0);
    }

    #[test]
    fn a_damaged_file_is_a_miss() {
        let k = key();
        let cubin = vec![9u8; 300];
        let good = encode(&k, &cubin);
        assert_eq!(decode(&k, &good), Ok(cubin.clone()));
        // truncated
        let e = decode(&k, &good[..good.len() - 1]).unwrap_err();
        assert!(e.contains("299 bytes of cubin"), "{e}");
        // one byte of the cubin flipped
        let mut flip = good.clone();
        let last = flip.len() - 1;
        flip[last] ^= 1;
        assert!(decode(&k, &flip).unwrap_err().contains("SHA-256"));
        // a longer file
        let mut long = good.clone();
        long.push(0);
        assert!(decode(&k, &long).unwrap_err().contains("301 bytes"));
        // no header at all
        assert!(decode(&k, &cubin).is_err());
        assert!(decode(&k, b"").is_err());
    }

    #[test]
    fn a_key_the_driver_could_not_have_reported_is_refused_at_store() {
        let d = Dir(scratch("check"));
        for bad in [
            Key { text_sha256: "abc".into(), ..key() },
            Key { text_sha256: key().text_sha256.to_uppercase(), ..key() },
            Key { arch: "sm_89".into(), ..key() },
            Key { nvrtc: (0, 0), ..key() },
            Key { driver: 0, ..key() },
            Key { sm: (0, 0), ..key() },
        ] {
            assert!(d.store(&bad, b"x").is_err(), "{bad:?}");
        }
        assert!(key().check().is_ok());
        let _ = std::fs::remove_dir_all(&d.0);
    }

    #[test]
    fn the_directory_comes_from_the_variable_or_the_home() {
        let env = |pairs: &'static [(&'static str, &'static str)]| {
            move |k: &str| pairs.iter().find(|(n, _)| *n == k).map(|(_, v)| v.to_string())
        };
        assert_eq!(dir_from(env(&[(ENV, "off"), ("HOME", "/h")])), Ok(None));
        assert_eq!(dir_from(env(&[(ENV, "/c/m")])), Ok(Some(Dir(PathBuf::from("/c/m")))));
        assert_eq!(dir_from(env(&[("HOME", "/h")])), Ok(Some(Dir(PathBuf::from("/h/.cache/llvq/modules")))));
        assert_eq!(
            dir_from(env(&[("XDG_CACHE_HOME", "/x"), ("HOME", "/h")])),
            Ok(Some(Dir(PathBuf::from("/x/llvq/modules"))))
        );
        assert!(dir_from(env(&[(ENV, ""), ("HOME", "/h")])).unwrap_err().contains("empty"));
        assert!(dir_from(env(&[(ENV, "rel/dir")])).unwrap_err().contains("absolute"));
        assert!(dir_from(env(&[])).unwrap_err().contains("HOME"));
        assert!(dir_from(env(&[("HOME", "rel")])).is_err());
    }
}
