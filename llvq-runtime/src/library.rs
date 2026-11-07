//! Model names, the compiled-in library, and the backend this build serves.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::sync::OnceLock;

/// `name:tag`, with `latest` when the tag is left out, as Ollama names them.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ModelName {
    pub name: String,
    pub tag: String,
}

/// One name or tag: letters, digits, `.`, `_`, `-`, and not starting with a
/// dot. Names end up as directory names under `LLVQ_HOME`, so a `/` or a
/// `..` is a path, not a name.
fn valid_part(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= 128
        && !s.starts_with('.')
        && s.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
}

impl ModelName {
    pub fn parse(s: &str) -> Result<Self, String> {
        let s = s.trim();
        let (name, tag) = match s.split_once(':') {
            Some((n, t)) => (n, t),
            None => (s, "latest"),
        };
        let (name, tag) = (name.to_ascii_lowercase(), tag.to_ascii_lowercase());
        if !valid_part(&name) || !valid_part(&tag) {
            return Err(format!(
                "{s:?} is not a model name: write name:tag, each part letters, digits, '.', '_' or '-'"
            ));
        }
        Ok(ModelName { name, tag })
    }
}

impl fmt::Display for ModelName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.name, self.tag)
    }
}

/// The served config of an entry, per backend. `None` where the object is
/// not served on that backend.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Configs {
    pub cuda: Option<String>,
    pub metal: Option<String>,
}

/// One model of `library.json`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Entry {
    pub name: String,
    pub tag: String,
    #[serde(default)]
    pub aliases: Vec<String>,
    pub repo: String,
    pub revision: String,
    pub file: String,
    pub size: u64,
    pub sha256: String,
    pub parameter_size: String,
    pub bits_per_param: String,
    pub configs: Configs,
}

impl Entry {
    pub fn model_name(&self) -> ModelName {
        ModelName { name: self.name.clone(), tag: self.tag.clone() }
    }

    /// Whether `n` names this entry, by its tag or one of its aliases.
    pub fn answers(&self, n: &ModelName) -> bool {
        n.name == self.name && (n.tag == self.tag || self.aliases.contains(&n.tag))
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LibraryFile {
    #[allow(dead_code)]
    note: String,
    models: Vec<Entry>,
}

/// The library as compiled into the binary.
pub const LIBRARY_JSON: &str = include_str!("../library.json");

/// Every entry of the library. Panics at first use if `library.json` does not
/// parse, which `the_library_parses_and_every_entry_is_whole` catches first.
pub fn entries() -> &'static [Entry] {
    static LIB: OnceLock<Vec<Entry>> = OnceLock::new();
    LIB.get_or_init(|| {
        let f: LibraryFile = serde_json::from_str(LIBRARY_JSON).expect("library.json parses");
        f.models
    })
}

/// The entry `n` names, if any.
pub fn find(n: &ModelName) -> Option<&'static Entry> {
    entries().iter().find(|e| e.answers(n))
}

/// The served configs, compiled in from `configs/` so the runtime serves the
/// bytes the journals pinned. One row per file `library.json` names.
const CONFIGS: &[(&str, &str)] = &[
    (
        "qwen3-4b-tetra-e4-4e-v9.json",
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../configs/qwen3-4b-tetra-e4-4e-v9.json")),
    ),
    (
        "qwen3-4b-tetra-e4.json",
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../configs/qwen3-4b-tetra-e4.json")),
    ),
    (
        "qwen3-8b-tetra-e4-4e-v9.json",
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../configs/qwen3-8b-tetra-e4-4e-v9.json")),
    ),
    (
        "qwen3-8b-tetra-e4.json",
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../configs/qwen3-8b-tetra-e4.json")),
    ),
    (
        "qwen3-14b-tetra-e4-4e-v9.json",
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../configs/qwen3-14b-tetra-e4-4e-v9.json")),
    ),
    (
        "qwen3-14b-tetra-e4.json",
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../configs/qwen3-14b-tetra-e4.json")),
    ),
];

/// The text of a compiled-in config.
pub fn config_text(file: &str) -> Option<&'static str> {
    CONFIGS.iter().find(|(n, _)| *n == file).map(|(_, t)| *t)
}

/// The compiled-in config files, by name.
pub fn config_files() -> impl Iterator<Item = &'static str> {
    CONFIGS.iter().map(|(n, _)| *n)
}

/// The engine this build serves with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// The fused CUDA kernel, Linux with `--features cuda`.
    Cuda,
    /// The fused Metal kernel, macOS with `--features metal`.
    Metal,
    /// The dense reconstruction on the CPU (`sealed::load`): any other build.
    /// Its speed and memory are not measured.
    Dense,
}

impl Backend {
    pub fn of_build() -> Self {
        #[cfg(all(target_os = "linux", feature = "cuda"))]
        {
            Backend::Cuda
        }
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            Backend::Metal
        }
        #[cfg(not(any(all(target_os = "linux", feature = "cuda"), all(target_os = "macos", feature = "metal"))))]
        {
            Backend::Dense
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Backend::Cuda => "cuda",
            Backend::Metal => "metal",
            Backend::Dense => "dense-cpu",
        }
    }

    /// The config this backend serves `c` with: `None` on the dense path,
    /// which takes no config.
    pub fn config(self, c: &Configs) -> Result<Option<&str>, String> {
        let pick = match self {
            Backend::Dense => return Ok(None),
            Backend::Cuda => c.cuda.as_deref(),
            Backend::Metal => c.metal.as_deref(),
        };
        pick.map(Some).ok_or_else(|| format!("this model has no served config on {}", self.label()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llvq_llm::served::{Served, ServedFile};

    #[test]
    fn names_default_to_latest_and_refuse_paths() {
        assert_eq!(ModelName::parse("qwen3").unwrap().to_string(), "qwen3:latest");
        assert_eq!(ModelName::parse(" Qwen3:4B ").unwrap().to_string(), "qwen3:4b");
        for bad in ["", ":4b", "qwen3:", "../x", "a/b:c", "q:..", "q:.x", "q w", "a:b:c"] {
            assert!(ModelName::parse(bad).is_err(), "{bad:?} must be refused");
        }
    }

    #[test]
    fn the_library_parses_and_every_entry_is_whole() {
        let e = entries();
        assert_eq!(e.len(), 3);
        for m in e {
            assert_eq!(m.sha256.len(), 64, "{}: a full sha256", m.model_name());
            assert!(m.sha256.chars().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
            assert_eq!(m.revision.len(), 40, "{}: a pinned commit, not a branch", m.model_name());
            assert!(ModelName::parse(&m.model_name().to_string()).is_ok());
            for c in [&m.configs.cuda, &m.configs.metal].into_iter().flatten() {
                let text = config_text(c).unwrap_or_else(|| panic!("{c} is named and not compiled in"));
                let f: ServedFile = serde_json::from_str(text).unwrap();
                Served::of(f, c.into()).unwrap_or_else(|e| panic!("{c}: {e}"));
            }
        }
        // No two entries answer the same name.
        for a in e {
            for t in std::iter::once(&a.tag).chain(&a.aliases) {
                let n = ModelName { name: a.name.clone(), tag: t.clone() };
                assert_eq!(e.iter().filter(|b| b.answers(&n)).count(), 1, "{n} is ambiguous");
            }
        }
    }

    #[test]
    fn latest_is_the_4b_and_the_digests_are_the_readme_ones() {
        let m = find(&ModelName::parse("qwen3").unwrap()).unwrap();
        assert_eq!(m.tag, "4b");
        // configs/README.md, the authority on these files.
        let readme = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../configs/README.md"));
        for m in entries() {
            assert!(readme.contains(&m.sha256[..8]), "{}: digest not in configs/README.md", m.model_name());
            assert!(readme.contains(&m.file), "{}: file not in configs/README.md", m.model_name());
        }
        assert!(readme.contains(&m.sha256), "the 4B's full digest is in the README");
    }

    #[test]
    fn metal_serves_only_the_expanded_form() {
        // served.rs refuses every other form on Metal; a library that named
        // one would fail at the first load instead of here.
        for m in entries() {
            let c = m.configs.metal.as_deref().unwrap();
            let f: ServedFile = serde_json::from_str(config_text(c).unwrap()).unwrap();
            let s = Served::of(f, c.into()).unwrap();
            s.check_device(true).unwrap_or_else(|e| panic!("{c}: {e}"));
        }
    }
}
