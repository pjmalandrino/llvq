//! Fetching and mapping a checkpoint.
//!
//! ## The two things `LLVQ_MODEL` may name
//!
//! A Hugging Face repo id (`Qwen/Qwen3-4B`), optionally pinned to a revision
//! (`Qwen/Qwen3-4B@<sha>`), or a **directory on this machine** (`/model`,
//! `./ckpt`, `~/qwen3-4b`).
//!
//! The local form is what makes `Volume(type="model")` usable. `ops/run.py`
//! has carried a `--mount-model` flag since the start and it was inert: the
//! value it hands the binary is the mount path, `/model`, and this loader used
//! to resolve every value as a repo id. So the flag would have sent the job
//! looking for a Hub repo literally called `/model`, and the documented
//! workaround was not to pass it — every job re-downloading the checkpoint.
//!
//! ⚠️ **The cost of that is not the "26 min" the docs repeat.** 65.5 GB ÷
//! 42 MB/s = 26.0 min: the two figures are each other, and neither end was
//! measured — `docs/archive/plan-de-test-v2-cuda.md` presents the rate as measured
//! on the 32B run, while `CLAUDE.md` and `ops/run.py` present the duration as the
//! measurement. What the 32B de-risking run actually bounds: it billed 59 min
//! (5.43 $ at 5.50 $/h on `rtx-pro-6000x2`), of which 2694 s are the
//! quantization loop (the core-second table in `ops/run.py`), leaving **≤ 846 s
//! for the entire out-of-loop** — image pull, checkpoint download, load,
//! factorization, artifact write and verify, together. A 26-minute download
//! does not fit in 14 minutes. The item is worth doing because a mounted
//! volume and a resumable run both *require* a local path, not because of a
//! number nobody took.
//!
//! ## Telling the two apart
//!
//! By the **shape of the string, never by what happens to exist on disk**:
//! a value starting with `/`, `./`, `../` or `~/` — or equal to `.` or `..` —
//! is a directory; everything else is a repo id. A repo id cannot begin with
//! any of those, so no legal id is captured.
//!
//! Testing for existence instead would read more naturally and has two silent
//! failure modes, each of which spends a whole run on the wrong bytes:
//!
//! * a directory named `Qwen` in the working directory captures
//!   `Qwen/Qwen3-4B`, and the run quantizes whatever is in it;
//! * a mistyped `/modle` is not a directory, so it falls through to being a
//!   repo id — and the loader goes to the network instead of saying the path
//!   is wrong.
//!
//! The syntactic rule has neither. The same string means the same thing on
//! every machine, and a local path that is missing fails loudly **naming the
//! path**. `a_directory_named_qwen_does_not_capture_the_repo_id` pins it with
//! a complete, loadable checkpoint planted at `./Qwen/Qwen3-0.6B`.
//!
//! This is the doctrine [`crate::sealed::is_sealed_path`] already follows for
//! the neighbouring decision — artifact or repo id, decided on the suffix, and
//! pinned by `repo_ids_are_never_sealed_paths`. Two syntactic rules over one
//! string, neither of which can be changed by what a machine happens to hold.
//!
//! ## Pinning a revision
//!
//! `repo@revision`, rather than a second environment variable. The revision
//! then travels with the repo through `--model`, `LLVQ_MODEL`, the log line
//! and any provenance prologue, so it cannot be set on one and forgotten on
//! another — and a log that prints the model prints the pin with it. A repo id
//! cannot contain `@` (namespaces and names are `[A-Za-z0-9._-]`), so the split
//! is unambiguous; a local path may contain one, which is why `@` is only read
//! on the Hub branch, after the local/Hub decision is already made.
//!
//! No `@` means `main`, which is what the code did before: `hf_hub::Repo::new`
//! *is* `Repo::with_revision(.., "main")`, same cache folder, same URL. An
//! unpinned command keeps byte-identical behaviour.

use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen3::Config;
use std::path::{Path, PathBuf};

/// The files a checkpoint is made of, named once so the Hub branch and the
/// local branch cannot drift into looking for different things.
const CONFIG: &str = "config.json";
const TOKENIZER: &str = "tokenizer.json";
const SINGLE: &str = "model.safetensors";
const INDEX: &str = "model.safetensors.index.json";

/// What a `LLVQ_MODEL` value resolves to. See the module docs for the rule
/// that separates the two and why it is syntactic.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Source {
    /// A directory on this machine — a mounted volume, an unpacked snapshot,
    /// or a Hub cache snapshot directory.
    Local(PathBuf),
    /// A Hugging Face repo at a revision. `revision` is `main` unless the
    /// value carried an `@`.
    Hub { repo: String, revision: String },
}

impl Source {
    /// Classify a `LLVQ_MODEL` value.
    ///
    /// Pure: it reads the string and nothing else — no filesystem, no network.
    /// That is the property the whole item rests on, so it is stated here
    /// rather than left to be noticed.
    pub fn parse(spec: &str) -> Result<Self, String> {
        let s = spec.trim();
        if s.is_empty() {
            return Err(
                "LLVQ_MODEL is empty: expected a repo (\"Qwen/Qwen3-4B\", \
                 \"Qwen/Qwen3-4B@<sha>\") or a directory (\"/model\", \"./ckpt\", \
                 \"~/ckpt\")"
                    .to_string(),
            );
        }
        if is_path(s) {
            return Ok(Self::Local(expand_home(s)?));
        }
        match s.split_once('@') {
            None => Ok(Self::Hub {
                repo: s.to_string(),
                revision: "main".to_string(),
            }),
            Some((repo, revision)) => {
                // Refused rather than repaired. A value with two `@`, or an
                // empty half, is a typo in a string that decides which weights
                // a paid run reads; guessing which half was meant is how one
                // ends up quantizing `main` while the log says otherwise.
                if repo.is_empty() || revision.is_empty() || revision.contains('@') {
                    return Err(format!(
                        "LLVQ_MODEL={spec}: the pinning syntax is \"repo@revision\", \
                         one single \"@\" and no empty half (e.g. \"Qwen/Qwen3-4B@a1b2c3d\"). \
                         Without \"@\", the revision is \"main\", as before."
                    ));
                }
                Ok(Self::Hub {
                    repo: repo.to_string(),
                    revision: revision.to_string(),
                })
            }
        }
    }

    /// One line naming exactly what was read — for provenance prologues, where
    /// "Qwen3-4B" alone does not say whether the bytes came off a volume or
    /// off the Hub, nor at which revision.
    pub fn describe(&self) -> String {
        match self {
            Self::Local(p) => format!("local directory {}", p.display()),
            Self::Hub { repo, revision } => format!("repo {repo}@{revision}"),
        }
    }
}

/// The path shapes. Unix-shaped on purpose: every machine this runs on — the
/// dev Mac, the HF Jobs containers, the CUDA image — is one, and accepting
/// `C:\…` here would mean accepting `Qwen\Qwen3-4B` too.
fn is_path(s: &str) -> bool {
    s == "."
        || s == ".."
        || s.starts_with('/')
        || s.starts_with("./")
        || s.starts_with("../")
        || s.starts_with("~/")
}

/// Expand a leading `~/`. Worth the four lines: `LLVQ_MODEL` is set
/// programmatically by `ops/run.py` and inside `bash -c` payloads, where no
/// shell expands it, and a literal `~` directory is not what anyone meant.
fn expand_home(s: &str) -> Result<PathBuf, String> {
    let Some(rest) = s.strip_prefix("~/") else {
        return Ok(PathBuf::from(s));
    };
    let home = std::env::var("HOME").map_err(|_| {
        format!("LLVQ_MODEL={s}: HOME is not set, so \"~/\" cannot be expanded. \
                 Give an absolute path")
    })?;
    Ok(Path::new(&home).join(rest))
}

/// The distinct shard files an index lists, sorted and deduplicated.
///
/// Shared by both branches deliberately: "what the Hub downloads" and "what a
/// mounted directory reads" have to be the same set in the same order, and the
/// only way to guarantee that is to have one function decide it.
fn shard_names(index: &[u8]) -> anyhow::Result<Vec<String>> {
    let json: serde_json::Value = serde_json::from_slice(index)?;
    let map = json["weight_map"]
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("{INDEX}: no \"weight_map\" object"))?;
    let mut names: Vec<String> = map
        .values()
        .filter_map(|v| v.as_str().map(str::to_owned))
        .collect();
    names.sort();
    names.dedup();
    anyhow::ensure!(
        !names.is_empty(),
        "{INDEX}: \"weight_map\" names no file"
    );
    Ok(names)
}

/// Everything needed to run a model, resolved from a local directory or from
/// the Hugging Face cache (downloading on first use).
#[derive(Debug)]
pub struct Checkpoint {
    pub config: Config,
    pub weights: Vec<PathBuf>,
    pub tokenizer: PathBuf,
    /// The raw `config.json`. Sealing copies the file rather than
    /// re-serializing the parsed struct, so a field candle does not model is
    /// carried across instead of silently dropped.
    pub config_path: PathBuf,
    /// Where the files above came from. Kept so a log can name the source
    /// instead of the argument: `/model` says nothing about which checkpoint
    /// a volume happened to hold.
    pub source: Source,
}

impl Checkpoint {
    /// Resolve a `LLVQ_MODEL` value — repo id, `repo@revision`, or a local
    /// directory. See [`Source`] for the rule.
    pub fn fetch(spec: &str) -> anyhow::Result<Self> {
        match Source::parse(spec).map_err(anyhow::Error::msg)? {
            Source::Local(dir) => Self::from_dir(&dir),
            Source::Hub { repo, revision } => Self::from_hub(&repo, &revision),
        }
    }

    /// Read a checkpoint laid out on disk: a mounted `Volume(type="model")`,
    /// an unpacked snapshot, or a Hub cache snapshot directory (whose entries
    /// are symlinks into `blobs/` — followed, like any other file).
    ///
    /// Never falls back to the Hub. A path that does not resolve is an
    /// operator error worth a message, not a reason to spend a download.
    pub fn from_dir(dir: &Path) -> anyhow::Result<Self> {
        let d = dir.display();
        anyhow::ensure!(
            dir.is_dir(),
            "LLVQ_MODEL={d}: this directory does not exist (or is not a directory). \
             No repo of the same name was looked up, a path stays a path."
        );
        let need = |name: &str| -> anyhow::Result<PathBuf> {
            let p = dir.join(name);
            anyhow::ensure!(p.is_file(), "{d} carries no {name}");
            Ok(p)
        };
        let config_path = need(CONFIG)?;
        let tokenizer = need(TOKENIZER)?;
        let config: Config = serde_json::from_slice(&std::fs::read(&config_path)?)
            .map_err(|e| anyhow::anyhow!("{}: {e}", config_path.display()))?;

        // Small models ship one shard; larger ones an index listing many —
        // the case of every model that matters here.
        let single = dir.join(SINGLE);
        let weights = if single.is_file() {
            vec![single]
        } else {
            let idx = dir.join(INDEX);
            anyhow::ensure!(idx.is_file(), "{d} carries neither {SINGLE} nor {INDEX}");
            shard_names(&std::fs::read(&idx)?)?
                .iter()
                .map(|n| {
                    let p = dir.join(n);
                    anyhow::ensure!(p.is_file(), "{d} carries no {n}, which {INDEX} lists");
                    Ok(p)
                })
                .collect::<anyhow::Result<Vec<_>>>()?
        };
        Ok(Self {
            config,
            weights,
            tokenizer,
            config_path,
            source: Source::Local(dir.to_path_buf()),
        })
    }

    /// Read a checkpoint from the Hub, at `revision`.
    ///
    /// `revision = "main"` reproduces the previous `api.model(repo)` exactly:
    /// `Repo::new` is defined as `Repo::with_revision(.., "main")`.
    pub fn from_hub(repo: &str, revision: &str) -> anyhow::Result<Self> {
        let api = hf_hub::api::sync::Api::new()?;
        let r = api.repo(hf_hub::Repo::with_revision(
            repo.to_string(),
            hf_hub::RepoType::Model,
            revision.to_string(),
        ));
        let config_path = r.get(CONFIG)?;
        let tokenizer = r.get(TOKENIZER)?;
        let config: Config = serde_json::from_slice(&std::fs::read(&config_path)?)?;

        // Small models ship one shard; larger ones an index listing many.
        let weights = match r.get(SINGLE) {
            Ok(p) => vec![p],
            Err(_) => {
                let idx = r.get(INDEX)?;
                shard_names(&std::fs::read(idx)?)?
                    .iter()
                    .map(|n| r.get(n).map_err(anyhow::Error::from))
                    .collect::<anyhow::Result<Vec<_>>>()?
            }
        };
        Ok(Self {
            config,
            weights,
            tokenizer,
            config_path,
            source: Source::Hub {
                repo: repo.to_string(),
                revision: revision.to_string(),
            },
        })
    }

    /// Map the weights.
    ///
    /// # Safety
    /// The checkpoint files must not be modified while the mapping is alive.
    pub fn var_builder(&self, dtype: DType, device: &Device) -> anyhow::Result<VarBuilder<'_>> {
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&self.weights, dtype, device)? };
        Ok(vb)
    }

    pub fn tokenizer(&self) -> anyhow::Result<tokenizers::Tokenizer> {
        tokenizers::Tokenizer::from_file(&self.tokenizer)
            .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A self-deleting directory under the system temp dir. `tempfile` would
    /// do this, and `llvq-llm` is the crate allowed dependencies — but ten
    /// lines are cheaper than a new one for a checkpoint of four small files.
    struct Scratch(PathBuf);

    impl Scratch {
        fn new(tag: &str) -> Self {
            let n = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let p =
                std::env::temp_dir().join(format!("llvq_loader_{tag}_{}_{n}", std::process::id()));
            std::fs::create_dir_all(&p).unwrap();
            Self(p)
        }
        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// A Qwen3 config small enough to be a literal, complete enough that
    /// `candle_transformers::models::qwen3::Config` deserializes it. If candle
    /// gains a required field this test fails at the parse, which is the point:
    /// the local branch must be held to the same config contract as the Hub one.
    const TINY_CONFIG: &str = r#"{
        "vocab_size": 32, "hidden_size": 8, "intermediate_size": 16,
        "num_hidden_layers": 1, "num_attention_heads": 2, "head_dim": 4,
        "attention_bias": false, "num_key_value_heads": 1,
        "max_position_embeddings": 64, "sliding_window": null,
        "max_window_layers": 1, "tie_word_embeddings": true,
        "rope_theta": 10000.0, "rms_norm_eps": 1e-6,
        "use_sliding_window": false, "hidden_act": "silu"
    }"#;

    /// Lay out a checkpoint. `shards` empty means the single-file form;
    /// otherwise an index plus those shard files. Only the *shape* is real —
    /// `from_dir` resolves paths and never opens a weight file.
    fn plant(dir: &Path, shards: &[&str]) {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join(CONFIG), TINY_CONFIG).unwrap();
        std::fs::write(dir.join(TOKENIZER), "{}").unwrap();
        if shards.is_empty() {
            std::fs::write(dir.join(SINGLE), b"").unwrap();
            return;
        }
        // Several tensors per shard, listed out of order, so a `shard_names`
        // that stopped sorting or deduplicating would show up.
        let mut map = serde_json::Map::new();
        for (i, s) in shards.iter().enumerate().rev() {
            map.insert(format!("t{i}.a"), serde_json::Value::from(*s));
            map.insert(format!("t{i}.b"), serde_json::Value::from(*s));
            std::fs::write(dir.join(s), b"").unwrap();
        }
        let idx = serde_json::json!({ "weight_map": serde_json::Value::Object(map) });
        std::fs::write(dir.join(INDEX), serde_json::to_vec(&idx).unwrap()).unwrap();
    }

    fn hub(repo: &str, revision: &str) -> Source {
        Source::Hub {
            repo: repo.to_string(),
            revision: revision.to_string(),
        }
    }

    #[test]
    fn a_bare_repo_id_is_a_hub_repo_at_main() {
        assert_eq!(
            Source::parse("Qwen/Qwen3-4B").unwrap(),
            hub("Qwen/Qwen3-4B", "main")
        );
        assert_eq!(Source::parse("gpt2").unwrap(), hub("gpt2", "main"));
    }

    /// Non-regression of the Hub path, stated against hf-hub rather than
    /// against our own intent: the pinned call with `main` must address the
    /// same cache folder and the same ref as the `api.model(repo)` this
    /// replaced, or every cached checkpoint on every machine is re-downloaded
    /// once, silently.
    #[test]
    fn pinning_main_is_what_the_hub_path_already_did() {
        let a = hf_hub::Repo::model("Qwen/Qwen3-4B".to_string());
        let b = hf_hub::Repo::with_revision(
            "Qwen/Qwen3-4B".to_string(),
            hf_hub::RepoType::Model,
            "main".to_string(),
        );
        assert_eq!(a.folder_name(), b.folder_name());
        assert_eq!(a.revision(), b.revision());
        assert_eq!(a.revision(), "main");
    }

    #[test]
    fn an_at_sign_pins_the_revision() {
        assert_eq!(
            Source::parse("Qwen/Qwen3-4B@a1b2c3d").unwrap(),
            hub("Qwen/Qwen3-4B", "a1b2c3d")
        );
        assert_eq!(
            Source::parse("Qwen/Qwen3-4B@refs/pr/2").unwrap(),
            hub("Qwen/Qwen3-4B", "refs/pr/2")
        );
    }

    /// A malformed pin must be refused, never silently read as `main`: the
    /// whole point of the syntax is that a run cannot claim a revision it did
    /// not use.
    #[test]
    fn a_malformed_pin_is_refused() {
        for bad in ["", "   ", "Qwen/Qwen3-4B@", "@a1b2c3d", "Qwen/Qwen3-4B@a@b"] {
            assert!(Source::parse(bad).is_err(), "{bad:?} should have been refused");
        }
    }

    #[test]
    fn path_shapes_resolve_local() {
        for p in ["/model", "./ckpt", "../ckpt", ".", "..", "/model/sub@dir"] {
            assert_eq!(
                Source::parse(p).unwrap(),
                Source::Local(PathBuf::from(p)),
                "{p} should have been a path"
            );
        }
        // `@` is only read on the Hub branch, so a path keeps it verbatim.
        assert_eq!(
            Source::parse("/model/sub@dir").unwrap(),
            Source::Local(PathBuf::from("/model/sub@dir"))
        );
    }

    #[test]
    fn a_tilde_is_expanded() {
        let home = std::env::var("HOME").expect("HOME on this platform");
        assert_eq!(
            Source::parse("~/qwen3-4b").unwrap(),
            Source::Local(Path::new(&home).join("qwen3-4b"))
        );
    }

    /// The failure mode an existence-based rule would create: a mistyped path
    /// becoming a repo id, and the loader going to the network for it. It must
    /// stay a path, and the message must carry the path so the typo is visible.
    ///
    /// Driven through `fetch`, not `from_dir`: the mutant worth killing is a
    /// *fallback* — "the directory is not there, try the Hub" — and only the
    /// dispatch can commit it. Hence the assertion on the word `directory`,
    /// which no Hub error carries.
    #[test]
    fn a_missing_local_path_never_becomes_a_repo_id() {
        let p = "/llvq-does-not-exist/checkpoint";
        assert_eq!(Source::parse(p).unwrap(), Source::Local(PathBuf::from(p)));
        let e = Checkpoint::fetch(p).unwrap_err().to_string();
        assert!(
            e.contains(p) && e.contains("directory does not exist"),
            "{e}"
        );
    }

    #[test]
    fn a_local_directory_serves_a_single_shard() {
        let s = Scratch::new("single");
        plant(s.path(), &[]);
        let ck = Checkpoint::fetch(s.path().to_str().unwrap()).unwrap();
        assert_eq!(ck.weights, vec![s.path().join(SINGLE)]);
        assert_eq!(ck.config_path, s.path().join(CONFIG));
        assert_eq!(ck.tokenizer, s.path().join(TOKENIZER));
        assert_eq!(ck.config.hidden_size, 8);
        assert_eq!(ck.source, Source::Local(s.path().to_path_buf()));
    }

    /// The case of every model that matters here. The index lists each shard
    /// twice and in reverse order; the resolved list must be each shard once,
    /// sorted, as full paths inside the directory.
    #[test]
    fn a_local_directory_serves_a_sharded_checkpoint() {
        let s = Scratch::new("sharded");
        let shards = [
            "model-00001-of-00003.safetensors",
            "model-00002-of-00003.safetensors",
            "model-00003-of-00003.safetensors",
        ];
        plant(s.path(), &shards);
        let ck = Checkpoint::fetch(s.path().to_str().unwrap()).unwrap();
        assert_eq!(
            ck.weights,
            shards.iter().map(|n| s.path().join(n)).collect::<Vec<_>>()
        );
    }

    /// An incomplete directory must name the file it lacks. A run that dies
    /// saying only "checkpoint invalide" costs whoever mounted the volume an
    /// hour of guessing which of four files the mount dropped.
    #[test]
    fn a_missing_file_is_named() {
        for missing in [CONFIG, TOKENIZER, SINGLE] {
            let s = Scratch::new("missing");
            plant(s.path(), &[]);
            std::fs::remove_file(s.path().join(missing)).unwrap();
            let e = Checkpoint::from_dir(s.path()).unwrap_err().to_string();
            assert!(e.contains(missing), "{missing} absent, message : {e}");
        }
        // Neither weight form present: both names, so the reader knows a
        // sharded layout would also have been accepted.
        let s = Scratch::new("noweights");
        plant(s.path(), &[]);
        std::fs::remove_file(s.path().join(SINGLE)).unwrap();
        let e = Checkpoint::from_dir(s.path()).unwrap_err().to_string();
        assert!(e.contains(SINGLE) && e.contains(INDEX), "{e}");
    }

    #[test]
    fn a_shard_the_index_lists_but_the_directory_lacks_is_named() {
        let s = Scratch::new("gap");
        let shards = ["a.safetensors", "b.safetensors"];
        plant(s.path(), &shards);
        std::fs::remove_file(s.path().join("b.safetensors")).unwrap();
        let e = Checkpoint::from_dir(s.path()).unwrap_err().to_string();
        assert!(e.contains("b.safetensors") && e.contains(INDEX), "{e}");
    }

    /// **The awkward case.** Someone with a `Qwen/` directory in their working
    /// directory must still get the Hub repo from `Qwen/Qwen3-0.6B`, and a
    /// resolution error here means a whole run on the wrong weights.
    ///
    /// The planted directory is a *complete, loadable* checkpoint, so the test
    /// fails against any rule that consults the filesystem — including "local
    /// wins if it is a usable checkpoint", the tempting refinement. One `./`
    /// apart, the same bytes load: the two verdicts come from the string, and
    /// only from the string.
    ///
    /// ⚠️ The only test in this crate that moves the working directory, which
    /// is process-global and shared with every other test thread. The window
    /// holds no assertion, so nothing can panic while it is open, and the
    /// directory is restored before anything is checked.
    #[test]
    fn a_directory_named_qwen_does_not_capture_the_repo_id() {
        let s = Scratch::new("cwd");
        let planted = s.path().join("Qwen").join("Qwen3-0.6B");
        plant(&planted, &[]);

        let saved = std::env::current_dir().unwrap();
        std::env::set_current_dir(s.path()).unwrap();
        let bare = Source::parse("Qwen/Qwen3-0.6B");
        let dotted = Source::parse("./Qwen/Qwen3-0.6B");
        let dotted_loads = Checkpoint::from_dir(Path::new("./Qwen/Qwen3-0.6B")).is_ok();
        std::env::set_current_dir(&saved).unwrap();

        assert_eq!(bare.unwrap(), hub("Qwen/Qwen3-0.6B", "main"));
        assert_eq!(
            dotted.unwrap(),
            Source::Local(PathBuf::from("./Qwen/Qwen3-0.6B"))
        );
        assert!(dotted_loads, "the planted checkpoint should have loaded");
    }

    /// End to end on real bytes when the machine has them: the Hub cache lays
    /// snapshots out as a directory of symlinks into `blobs/`, which is
    /// exactly the shape a mounted volume has. Skipped, not failed, where the
    /// cache is absent — this is a proof when it can be had, never a gate.
    #[test]
    fn a_real_cached_snapshot_loads_and_maps() {
        let Some(dir) = cached_snapshot("Qwen--Qwen3-0.6B") else {
            eprintln!("HF cache absent: real load not exercised");
            return;
        };
        let ck = Checkpoint::fetch(dir.to_str().unwrap()).unwrap();
        assert_eq!(ck.weights, vec![dir.join(SINGLE)]);
        assert!(ck.config.num_hidden_layers > 0);
        ck.tokenizer().unwrap();
        let vb = ck.var_builder(DType::F32, &Device::Cpu).unwrap();
        assert!(vb.contains_tensor("model.embed_tokens.weight"));
    }

    /// The multi-shard half of the same proof, on a real index. Paths only —
    /// mapping 7.5 GB would say nothing the single-shard test has not.
    #[test]
    fn a_real_cached_sharded_snapshot_resolves_every_shard() {
        let Some(dir) = cached_snapshot("Qwen--Qwen3-4B") else {
            eprintln!("HF cache absent: multi-shard path not exercised");
            return;
        };
        let ck = Checkpoint::fetch(dir.to_str().unwrap()).unwrap();
        assert!(ck.weights.len() > 1, "{:?}", ck.weights);
        assert!(ck.weights.iter().all(|p| p.is_file()));
    }

    /// The newest snapshot directory of a cached model, if any.
    fn cached_snapshot(folder: &str) -> Option<PathBuf> {
        let home = std::env::var("HOME").ok()?;
        let snaps = Path::new(&home)
            .join(".cache/huggingface/hub")
            .join(format!("models--{folder}"))
            .join("snapshots");
        let mut best: Option<PathBuf> = None;
        for e in std::fs::read_dir(snaps).ok()? {
            let p = e.ok()?.path();
            if p.join(CONFIG).is_file() {
                best = Some(p);
            }
        }
        best
    }
}
