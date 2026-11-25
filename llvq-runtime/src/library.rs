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
    /// The config the CPU backend serves. Absent in manifests written before
    /// it existed, which then have no CPU config.
    #[serde(default)]
    pub cpu: Option<String>,
    /// The config `llvq serve` takes on CUDA under `LLVQ_BATCH`: the batched
    /// decode, several requests a step. Absent in manifests written before it
    /// existed; [`served_configs`] fills it from the library.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cuda_batch: Option<String>,
    /// The config `llvq serve` and `llvq run` take on CUDA under
    /// `LLVQ_PROFILE=eager`, where the default `cuda` config prefills by the
    /// chain (the 4B since 2026-11-15): the same object with the eager
    /// prefill. Absent where `cuda` already prefills eagerly (the 8B, the
    /// 14B), and in manifests written before it existed; [`served_configs`]
    /// fills it from the library.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cuda_eager: Option<String>,
}

/// `LLVQ_PROFILE`: the way back from a default. Unset is the default config
/// of the backend, which for the 4B on CUDA prefills by the chain since
/// 2026-11-15; `eager` is the eager prefill.
pub const PROFILE_ENV: &str = "LLVQ_PROFILE";

/// The values `LLVQ_PROFILE` takes.
pub const PROFILES: [&str; 1] = ["eager"];

/// The line `llvq run` prints before a model that prefills by the chain on
/// CUDA loads: a first start compiles its two tree modules (509.2 s cold
/// against 55.2 s cached for the whole load, *measured* on an L40S,
/// `docs/mesures/chain-prefill-served-2026-11-15.txt`). `None` for any other
/// backend or config.
pub fn first_start_hint(backend: &str, config: &str) -> Option<String> {
    let chain = config_text(config)
        .and_then(|t| serde_json::from_str::<llvq_llm::served::ServedFile>(t).ok())
        .is_some_and(|f| f.prefill.as_deref() == Some("chain"));
    (backend == "cuda" && chain).then(|| {
        format!(
            "{config} prefills by the chain: a first start on this card compiles its two tree modules, about 8.5 min; \
             later starts load in about a minute (set {PROFILE_ENV}=eager on the server for the eager prefill)"
        )
    })
}

/// The config a load serves: the backend's own, the batch config under
/// `LLVQ_BATCH`, and under `LLVQ_PROFILE=eager` the eager prefill's (the
/// entry's `cuda_eager` on CUDA where it has one, else the default, which
/// already prefills eagerly). Pure, the two variables handed in. Refused by
/// name: any other profile, `chain` among them (the 4B's default), the batch
/// off CUDA, and a model with no config on the backend.
pub fn config_for<'a>(c: &'a Configs, backend: Backend, batch: bool, profile: Option<&str>) -> Result<&'a str, String> {
    match profile {
        None | Some("eager") => {}
        Some("chain") => {
            return Err(format!(
                "{PROFILE_ENV}=chain: the chain prefill is the default where a model carries it (the 4B on CUDA, since \
                 2026-11-15). Unset {PROFILE_ENV}, or write eager for the eager prefill."
            ))
        }
        Some(p) => return Err(format!("{PROFILE_ENV}={p:?}: unset for the default config, or one of {PROFILES:?}")),
    }
    if batch {
        // the batch pool prefills through the eager path, whatever the profile
        return match backend {
            Backend::Cuda => c.cuda_batch.as_deref().ok_or_else(|| "LLVQ_BATCH is set and this model has no batch config".into()),
            b => Err(format!(
                "LLVQ_BATCH is set on {}: the batched decode's kernels are CUDA. Unset it to serve one request at a time.",
                b.label()
            )),
        };
    }
    let default = backend.config(c)?.ok_or_else(|| format!("this model has no served config on {}", backend.label()))?;
    match (profile, backend) {
        (Some(_), Backend::Cuda) => Ok(c.cuda_eager.as_deref().unwrap_or(default)),
        _ => Ok(default),
    }
}

/// What the pinned file of an entry is.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    /// One sealed file, served from the store's blob (the Qwen3 files).
    #[default]
    Sealed,
    /// An index of files (`crate::store::FileIndex`): the pull fetches every
    /// file it lists, each checked by size and sha256, into one directory,
    /// and serves them as a local file of [`crate::inkling`] would.
    Files,
}

/// One model of `library.json`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Entry {
    /// Also the family the manifest records: `qwen3`, `inkling2`.
    pub name: String,
    pub tag: String,
    /// Other tags of the same name.
    #[serde(default)]
    pub aliases: Vec<String>,
    /// Other whole names, `name` or `name:tag`; a bare name means its
    /// `latest` tag.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub names: Vec<String>,
    #[serde(default)]
    pub kind: Kind,
    pub repo: String,
    pub revision: String,
    /// The sealed file, or for [`Kind::Files`] the index.
    pub file: String,
    pub size: u64,
    pub sha256: String,
    /// Bytes on disk once pulled, where `size` is not that (an index).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub download_size: Option<u64>,
    pub parameter_size: String,
    pub bits_per_param: String,
    pub configs: Configs,
}

impl Entry {
    pub fn model_name(&self) -> ModelName {
        ModelName { name: self.name.clone(), tag: self.tag.clone() }
    }

    /// Whether `n` names this entry, by its tag, one of its aliases or one of
    /// its other names.
    pub fn answers(&self, n: &ModelName) -> bool {
        (n.name == self.name && (n.tag == self.tag || self.aliases.contains(&n.tag)))
            || self.names.iter().any(|o| ModelName::parse(o).is_ok_and(|o| &o == n))
    }

    /// Whether the entry can be pulled: a Hub repository, a pinned commit,
    /// a size and a full sha256. An entry whose pins are not filled yet is
    /// refused by name, never guessed.
    pub fn published(&self) -> Result<(), String> {
        let hex = |s: &str, n: usize| s.len() == n && s.chars().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase());
        if self.repo.is_empty() || !hex(&self.revision, 40) || !hex(&self.sha256, 64) || self.size == 0 {
            return Err(format!(
                "{} is in the llvq library but not published yet: its Hub repository, revision, size and sha256 \
                 are not filled. Serve it from local files with `llvq serve --local <file.json>`.",
                self.model_name()
            ));
        }
        Ok(())
    }

    /// The bytes a pull writes, as the chat page and `llvq show` give them.
    pub fn bytes(&self) -> u64 {
        self.download_size.unwrap_or(self.size)
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

/// The configs a file is served with: the library entry of the same digest,
/// whatever the manifest wrote at pull time. A manifest copies the configs of
/// its day, and a config renamed since (v4 and grouped to v9, 7be533e) is no
/// longer compiled in, or is compiled in and stale (audit 2026-11-08 §1.5).
/// A file the library does not know keeps its manifest's own.
pub fn served_configs(c: &Configs, sha256: &str) -> Configs {
    entries().iter().find(|e| e.sha256 == sha256).map_or_else(|| c.clone(), |e| e.configs.clone())
}

/// One `(name, text)` row per config file, read at build time.
macro_rules! configs {
    ($($f:literal),* $(,)?) => {
        &[$(($f, include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../configs/", $f)))),*]
    };
}

/// The served configs, compiled in from `configs/` so the runtime serves the
/// bytes the journals pinned. One row per file `library.json` names.
const CONFIGS: &[(&str, &str)] = configs![
    "qwen3-4b-tetra-e4-4e-v9.json",
    "qwen3-4b-tetra-e4.json",
    "qwen3-8b-tetra-e4-4e-v9.json",
    "qwen3-8b-tetra-e4.json",
    "qwen3-14b-tetra-e4-4e-v9.json",
    "qwen3-14b-tetra-e4.json",
    "qwen3-4b-tetra-e4-4e-v9-batch.json",
    "qwen3-8b-tetra-e4-4e-v9-batch.json",
    "qwen3-14b-tetra-e4-4e-v9-batch.json",
    "qwen3-4b-tetra-e4-4e-v9-chain.json",
];

/// The text of a compiled-in config.
pub fn config_text(file: &str) -> Option<&'static str> {
    CONFIGS.iter().find(|(n, _)| *n == file).map(|(_, t)| *t)
}

/// The engine this build serves with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// The fused CUDA kernel, Linux with `--features cuda`.
    Cuda,
    /// The fused Metal kernel, macOS with `--features metal`.
    Metal,
    /// The Tetra codes read on the CPU (`llvq_llm::fused_cpu`): any other
    /// build, at f32. 8.6 tok/s in 1.86 GB on the 4B on an M3 Max at 12
    /// threads (`docs/mesures/cpu-tetra-2026-11-07.txt`).
    Cpu,
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
            Backend::Cpu
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Backend::Cuda => "cuda",
            Backend::Metal => "metal",
            Backend::Cpu => "cpu",
        }
    }

    /// The config this backend serves `c` with. `Option` for the callers'
    /// sake; every backend now takes one.
    pub fn config(self, c: &Configs) -> Result<Option<&str>, String> {
        let pick = match self {
            Backend::Cuda => c.cuda.as_deref(),
            Backend::Metal => c.metal.as_deref(),
            Backend::Cpu => c.cpu.as_deref(),
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
        assert_eq!(sealed().count(), 3);
        for m in sealed() {
            assert_eq!(m.sha256.len(), 64, "{}: a full sha256", m.model_name());
            assert!(m.sha256.chars().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
            assert_eq!(m.revision.len(), 40, "{}: a pinned commit, not a branch", m.model_name());
            assert!(ModelName::parse(&m.model_name().to_string()).is_ok());
            for c in [&m.configs.cuda, &m.configs.metal, &m.configs.cpu, &m.configs.cuda_batch, &m.configs.cuda_eager]
                .into_iter()
                .flatten()
            {
                let text = config_text(c).unwrap_or_else(|| panic!("{c} is named and not compiled in"));
                let f: ServedFile = serde_json::from_str(text).unwrap();
                Served::of(f, c.into()).unwrap_or_else(|e| panic!("{c}: {e}"));
            }
        }
        // No two entries answer the same name.
        for a in e {
            let tags = std::iter::once(&a.tag).chain(&a.aliases).map(|t| ModelName { name: a.name.clone(), tag: t.clone() });
            let others = a.names.iter().map(|o| ModelName::parse(o).unwrap_or_else(|err| panic!("{o}: {err}")));
            for n in tags.chain(others) {
                assert_eq!(e.iter().filter(|b| b.answers(&n)).count(), 1, "{n} is ambiguous");
            }
        }
    }

    fn sealed() -> impl Iterator<Item = &'static Entry> {
        entries().iter().filter(|e| e.kind == Kind::Sealed)
    }

    /// Inkling 2 Mini answers to its name, its tags and the operator's
    /// wording, and is refused by name until its pins are filled: a pull
    /// never guesses a repository or a digest.
    #[test]
    fn inkling_answers_its_names_and_is_refused_until_published() {
        for s in ["inkling2:mini", "inkling2", "inkling2:mini-tetra", "inklingmini-2-tetra", "InklingMini-2-Tetra:latest"] {
            let e = find(&ModelName::parse(s).unwrap()).unwrap_or_else(|| panic!("{s} names no entry"));
            assert_eq!(e.model_name().to_string(), "inkling2:mini", "{s}");
            assert_eq!(e.kind, Kind::Files);
            assert_eq!(e.name, crate::inkling::FAMILY, "the manifest's family is the entry's name");
        }
        assert!(find(&ModelName::parse("inklingmini-2-tetra:4b").unwrap()).is_none());
        let e = find(&ModelName::parse("inkling2:mini").unwrap()).unwrap();
        if e.repo.is_empty() {
            let err = e.published().unwrap_err();
            assert!(err.contains("inkling2:mini") && err.contains("not published"), "{err}");
        }
        for m in sealed() {
            assert_eq!(m.published(), Ok(()), "{}", m.model_name());
        }
        // A chat page never offers what cannot be pulled.
        let page = crate::api::chat_page();
        assert_eq!(page.contains("inkling2:mini"), e.published().is_ok());
    }

    #[test]
    fn latest_is_the_4b_and_the_digests_are_the_readme_ones() {
        let m = find(&ModelName::parse("qwen3").unwrap()).unwrap();
        assert_eq!(m.tag, "4b");
        // configs/README.md, the authority on these files.
        let readme = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../configs/README.md"));
        for m in sealed() {
            assert!(readme.contains(&m.sha256[..8]), "{}: digest not in configs/README.md", m.model_name());
            assert!(readme.contains(&m.file), "{}: file not in configs/README.md", m.model_name());
        }
        assert!(readme.contains(&m.sha256), "the 4B's full digest is in the README");
    }

    #[test]
    fn metal_serves_only_the_expanded_form() {
        // served.rs refuses every other form on Metal; a library that named
        // one would fail at the first load instead of here.
        for m in sealed() {
            let c = m.configs.metal.as_deref().unwrap();
            let f: ServedFile = serde_json::from_str(config_text(c).unwrap()).unwrap();
            let s = Served::of(f, c.into()).unwrap();
            s.check_device(true).unwrap_or_else(|e| panic!("{c}: {e}"));
        }
    }

    #[test]
    fn the_cpu_serves_what_its_loader_reads() {
        // fused_cpu refuses anything but tetra48, no fused group, the file's
        // int4 order, an int4 or f16 embedding, and the eager decode.
        for m in sealed() {
            let c = m.configs.cpu.as_deref().unwrap_or_else(|| panic!("{}: no CPU config", m.model_name()));
            let f: ServedFile = serde_json::from_str(config_text(c).unwrap()).unwrap();
            let s = Served::of(f, c.into()).unwrap();
            assert_eq!(s.layout, llvq_llm::fused::FusedLayout::Tetra48, "{c}");
            assert_eq!(s.fuse, llvq_llm::fused::FuseMode::Off, "{c}");
            assert_eq!(s.int4_layout(), llvq_llm::fused::Int4Layout::Served, "{c}");
            assert_ne!(s.embed, llvq_llm::fused::EmbedMode::Q8, "{c}");
            assert_eq!(s.decode, llvq_llm::served::Decode::Eager, "{c}");
        }
    }

    /// With no variable the 4B on CUDA loads the chain config; `eager` is the
    /// way back to the v9 config; the 8B and the 14B keep v9 either way; the
    /// batch, Metal and the CPU read as before; any other profile is refused.
    #[test]
    fn the_chain_is_the_4b_default_on_cuda_and_eager_is_the_way_back() {
        let four = find(&ModelName::parse("qwen3:4b").unwrap()).unwrap();
        let c = &four.configs;
        assert_eq!(config_for(c, Backend::Cuda, false, None), Ok("qwen3-4b-tetra-e4-4e-v9-chain.json"));
        assert_eq!(config_for(c, Backend::Cuda, false, Some("eager")), Ok("qwen3-4b-tetra-e4-4e-v9.json"));
        assert_eq!(config_for(c, Backend::Cuda, true, None), Ok("qwen3-4b-tetra-e4-4e-v9-batch.json"));
        assert_eq!(config_for(c, Backend::Cuda, true, Some("eager")), Ok("qwen3-4b-tetra-e4-4e-v9-batch.json"));
        assert_eq!(config_for(c, Backend::Metal, false, None), Ok("qwen3-4b-tetra-e4.json"));
        assert_eq!(config_for(c, Backend::Metal, false, Some("eager")), Ok("qwen3-4b-tetra-e4.json"));
        assert_eq!(config_for(c, Backend::Cpu, false, None), Ok("qwen3-4b-tetra-e4.json"));
        let refused = [
            (config_for(c, Backend::Cuda, false, Some("chain")), "is the default where a model carries it"),
            (config_for(c, Backend::Cuda, false, Some("fast")), "\"fast\""),
            (config_for(c, Backend::Cuda, false, Some("")), "\"\""),
            (config_for(c, Backend::Cuda, false, Some("Eager")), "\"Eager\""),
            (config_for(c, Backend::Cuda, true, Some("chain")), "is the default where a model carries it"),
            (config_for(c, Backend::Metal, true, None), "CUDA"),
        ];
        for (r, word) in refused {
            let e = r.unwrap_err();
            assert!(e.contains(word), "{e}");
        }
        let form = |name: &str| {
            let f: ServedFile = serde_json::from_str(config_text(name).unwrap()).unwrap();
            Served::of(f, name.into()).unwrap().prefill_form
        };
        use llvq_llm::served::Prefill;
        assert_eq!(form(c.cuda.as_deref().unwrap()), Prefill::Chain);
        assert_eq!(form(c.cuda_eager.as_deref().unwrap()), Prefill::Eager);
        // the 8B and the 14B keep the v9 config, which prefills eagerly, with or without the profile
        for tag in ["8b", "14b"] {
            let m = find(&ModelName::parse(&format!("qwen3:{tag}")).unwrap()).unwrap();
            assert_eq!(m.configs.cuda_eager, None, "{tag}");
            let v9 = format!("qwen3-{tag}-tetra-e4-4e-v9.json");
            assert_eq!(config_for(&m.configs, Backend::Cuda, false, None), Ok(v9.as_str()), "{tag}");
            assert_eq!(config_for(&m.configs, Backend::Cuda, false, Some("eager")), Ok(v9.as_str()), "{tag}");
            assert_eq!(form(&v9), Prefill::Eager, "{tag}");
        }
        // `llvq run` warns of the first-start compile for the chain on CUDA only
        assert!(first_start_hint("cuda", "qwen3-4b-tetra-e4-4e-v9-chain.json").is_some_and(|h| h.contains("8.5 min")));
        for (b, cfg) in [
            ("cuda", "qwen3-4b-tetra-e4-4e-v9.json"),
            ("cuda", "qwen3-8b-tetra-e4-4e-v9.json"),
            ("metal", "qwen3-4b-tetra-e4-4e-v9-chain.json"),
            ("cuda", "unknown.json"),
        ] {
            assert_eq!(first_start_hint(b, cfg), None, "{b} {cfg}");
        }
        // the chain config is the v9 config with the chain prefill
        let chain: ServedFile = serde_json::from_str(config_text("qwen3-4b-tetra-e4-4e-v9-chain.json").unwrap()).unwrap();
        let v9: ServedFile = serde_json::from_str(config_text("qwen3-4b-tetra-e4-4e-v9.json").unwrap()).unwrap();
        assert_eq!(ServedFile { prefill: None, note: None, ..chain }, ServedFile { note: None, ..v9 });
    }

    #[test]
    fn a_manifest_is_served_by_the_library_configs_of_its_file() {
        for e in sealed() {
            // Pulled before the CPU backend existed: no CPU config.
            let old = Configs { cuda: e.configs.cuda.clone(), metal: e.configs.metal.clone(), cpu: None, cuda_batch: None, cuda_eager: None };
            assert_eq!(served_configs(&old, &e.sha256), e.configs, "{}: the CPU and batch configs filled", e.model_name());
            // Pulled on 2026-11-06: the CUDA config of that day, no longer
            // compiled in, and a stale one that still is.
            for cuda in ["qwen3-4b-tetra-e4-4e-v4.json", "qwen3-4b-tetra-e4.json"] {
                let stale = Configs { cuda: Some(cuda.into()), ..e.configs.clone() };
                let got = served_configs(&stale, &e.sha256);
                assert_eq!(got, e.configs, "{}: {cuda} replaced by the library's", e.model_name());
                let c = Backend::Cuda.config(&got).unwrap().unwrap();
                assert!(config_text(c).is_some(), "{}: {c} is compiled in", e.model_name());
            }
        }
        // A file the library does not know keeps its own.
        let own = Configs { cuda: Some("x.json".into()), metal: None, cpu: None, cuda_batch: None, cuda_eager: None };
        assert_eq!(served_configs(&own, &"0".repeat(64)), own);
        // A manifest written before the CPU field parses.
        let parsed: Configs = serde_json::from_str(r#"{"cuda": null, "metal": "m.json"}"#).unwrap();
        assert_eq!(parsed.cpu, None);
    }
}
