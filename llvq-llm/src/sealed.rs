//! Loading the sealed artifact — the deliverable — as a live model.
//!
//! ## Why this is one function and not three
//!
//! `bin/run` and `bin/mmlu` each carried their own copy of this loader, and
//! `bin/ppl` carried none: its quantized arm overlaid a *safetensors dump of
//! reconstructions* onto a freshly fetched checkpoint instead. So the two
//! harnesses that produce this project's headline numbers were reading two
//! different objects, and nothing in either output said so.
//!
//! It has already bitten. Two Qwen3-4B runs on 2026-07-31 print the identical
//! configuration line — `[leech1c12, 36 blocks, rot on, calib c4]` — and the
//! identical rate, and report `ppl = 15.3272` and `ppl = 16.9617`. The overlay
//! left on disk belongs to the first; the sealed file belongs to the second.
//! A perplexity taken from one and an MMLU score taken from the other are not
//! two measurements of one model, and no line of either output would tell you.
//!
//! One loader, three callers. `bin/ppl` can now score the sealed file itself,
//! which is the only way its number and `bin/mmlu`'s describe the same bytes.
//!
//! ## What "sealed" buys, precisely
//!
//! A format-v1 file carries the quantized projections and nothing else — no
//! embedding, no norms, no config, no tokenizer — so it cannot be run without
//! reaching back to a checkpoint. That is refused here rather than papered
//! over: an artifact that needs the internet to answer a question is not the
//! deliverable this project claims to build.

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen3::Config;
use std::collections::HashMap;
use std::io::Read;

use crate::loader::{Checkpoint, Source};
use crate::model::Qwen3;

/// The seven projection types of a Qwen3 block. The names are the tensor-name
/// segments — `k_proj`, not `k` — so a value copied from a checkpoint listing
/// is a valid spec.
pub const PROJ_TYPES: [&str; 7] = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
];

/// Projection types to take **from the reference checkpoint, at its own
/// precision**, instead of from the artifact's lattice codes.
///
/// ## What it is for
///
/// Attribution. The shipped 4B loses −14.73 pp of MMLU and nothing in the
/// dossier says which of the seven projections loses them: every matrix is
/// quantized under the same codebook, the same cap, the same bit budget. The
/// cheapest experiment that answers it restores **one type** to f16 — all
/// thirty-six `k_proj`, say — and scores the rest as shipped. Seven such arms
/// plus the shipped file, paired question by question, is the error budget by
/// function (`docs/ROADMAP-RECHERCHE.md`, M2).
///
/// ## Why it is exact
///
/// [`llvq_artifact::decode_matrix`] un-rotates on the way out, so the tensors
/// a sealed file yields are in the **natural basis** — the basis the
/// checkpoint's own weights are in — and a `Proj::Dense` carries no rotation
/// key. A restored matrix therefore needs no transform: the checkpoint's
/// `(d_out, d_in)` tensor, narrowed to the run dtype exactly as [`VarBuilder`]
/// would narrow it, drops into the same slot. With all seven restored the model
/// *is* the checkpoint — which is the control an attribution job runs first.
///
/// ## What it refuses
///
/// A name outside [`PROJ_TYPES`], a duplicate, and — at load time — a type the
/// artifact does not carry, a shape the checkpoint disagrees on, or a
/// restoration asked for without a checkpoint to take it from. An A/B that
/// silently restored nothing would be an A/B that lies.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RestoreF16 {
    types: Vec<&'static str>,
}

impl RestoreF16 {
    /// Parse a `LLVQ_RESTORE_F16` value: empty means nothing, `all` means the
    /// seven, otherwise a comma-separated subset of [`PROJ_TYPES`].
    pub fn parse(spec: &str) -> Result<Self, String> {
        let spec = spec.trim();
        if spec.is_empty() {
            return Ok(Self::default());
        }
        if spec == "all" {
            return Ok(Self {
                types: PROJ_TYPES.to_vec(),
            });
        }
        let mut types: Vec<&'static str> = Vec::new();
        for raw in spec.split(',') {
            let name = raw.trim();
            let Some(known) = PROJ_TYPES.iter().find(|t| **t == name) else {
                return Err(format!(
                    "LLVQ_RESTORE_F16={spec:?} : {name:?} n'est pas un type de projection. \
                     Admis : {}, ou `all`.",
                    PROJ_TYPES.join(", ")
                ));
            };
            if types.contains(known) {
                return Err(format!("LLVQ_RESTORE_F16={spec:?} : {name:?} est répété"));
            }
            types.push(known);
        }
        Ok(Self { types })
    }

    pub fn is_empty(&self) -> bool {
        self.types.is_empty()
    }

    pub fn types(&self) -> &[&'static str] {
        &self.types
    }

    /// Whether `name` — `model.layers.{b}.{site}.{proj}.weight` — belongs to
    /// a restored type.
    pub fn covers(&self, name: &str) -> bool {
        self.types
            .iter()
            .any(|t| name.ends_with(&format!(".{t}.weight")))
    }

    /// Comma-joined, for labels and error messages.
    pub fn describe(&self) -> String {
        self.types.join(",")
    }
}

/// What [`restore_projections`] did, for the label and the result line.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Restored {
    pub matrices: u32,
    pub weights: usize,
}

/// Replace, in `tensors`, every matrix of a restored type by what `source`
/// yields for the same name.
///
/// `source` is the checkpoint, abstracted as a closure so the logic is
/// testable without a safetensors file. Every restored type must match at
/// least one matrix and every replacement must keep its shape; either failure
/// aborts the load — a job that scored a half-restored model would publish
/// the wrong arm.
pub fn restore_projections(
    tensors: &mut HashMap<String, Tensor>,
    restore: &RestoreF16,
    mut source: impl FnMut(&str) -> anyhow::Result<Tensor>,
) -> anyhow::Result<Restored> {
    let mut done = Restored::default();
    if restore.is_empty() {
        return Ok(done);
    }
    let mut names: Vec<String> = tensors
        .keys()
        .filter(|n| restore.covers(n))
        .cloned()
        .collect();
    names.sort();
    for t in restore.types() {
        anyhow::ensure!(
            names.iter().any(|n| n.ends_with(&format!(".{t}.weight"))),
            "LLVQ_RESTORE_F16 demande {t}, et l'artefact ne porte aucune matrice de ce type"
        );
    }
    for name in names {
        let have = tensors[&name].dims().to_vec();
        let fresh = source(&name)?;
        anyhow::ensure!(
            fresh.dims() == have.as_slice(),
            "{name} : l'artefact porte {have:?}, le checkpoint {:?} — ce n'est pas le même modèle",
            fresh.dims()
        );
        done.weights += fresh.elem_count();
        done.matrices += 1;
        tensors.insert(name, fresh);
    }
    Ok(done)
}

fn describe_source(s: &Source) -> String {
    match s {
        Source::Local(p) => p.display().to_string(),
        Source::Hub { repo, revision } => format!("{repo}@{revision}"),
    }
}

/// A model rebuilt entirely from one file, plus what it took to do so.
pub struct SealedModel {
    pub model: Qwen3,
    pub tokenizer: tokenizers::Tokenizer,
    pub config: Config,
    /// Weights held by the quantized matrices.
    pub quantized_weights: usize,
    /// Weights carried verbatim at f16 — embeddings and norms.
    pub carried_weights: usize,
    pub matrices: u32,
    /// How many tensors were carried whole rather than quantized.
    pub raw_tensors: u32,
    /// Size on disk, so a compression ratio is quoted from the file rather
    /// than from a projection of what it ought to weigh.
    pub bytes: u64,
    /// Projection types taken from the checkpoint instead of the codes —
    /// empty for the deliverable as shipped. See [`RestoreF16`].
    pub restored_types: RestoreF16,
    /// How many matrices and weights that replaced.
    pub restored: Restored,
    /// Where they came from, for the label. `None` when nothing was restored.
    pub restored_from: Option<String>,
}

impl SealedModel {
    /// One phrase naming what was restored, or `None` for the shipped file —
    /// meant for the label every harness prints and writes into its dump, so
    /// that an arm's identity travels with its number.
    pub fn restore_note(&self) -> Option<String> {
        if self.restored_types.is_empty() {
            return None;
        }
        Some(format!(
            "f16 restored: {} ({} matrices, {} weights) from {}",
            self.restored_types.describe(),
            self.restored.matrices,
            self.restored.weights,
            self.restored_from.as_deref().unwrap_or("?")
        ))
    }
}

/// Whether a path names an artifact rather than a Hugging Face repo id.
///
/// `.bin` is included because that is what `bin/seal` has been writing; a
/// safetensors overlay (`.safetensors`) deliberately does not match, since it
/// is a dump of reconstructions and not a self-contained model.
pub fn is_sealed_path(s: &str) -> bool {
    s.ends_with(".llvq") || s.ends_with(".bin")
}

fn read_u32(r: &mut impl Read) -> anyhow::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

/// Rebuild a model from a sealed artifact — no checkpoint, no cache, no
/// network.
pub fn load(
    path: &str,
    dtype: DType,
    device: &Device,
    kv: crate::kvq::KvMode,
) -> anyhow::Result<SealedModel> {
    load_with_restored(path, dtype, device, kv, &RestoreF16::default(), None)
}

/// [`load`], with the projection types in `restore` taken from `checkpoint`
/// instead of from the codes — see [`RestoreF16`] for what that measures.
///
/// `checkpoint` is needed exactly when `restore` is non-empty. Asking for a
/// restoration without a source is refused rather than quietly scoring the
/// shipped file under a label that says otherwise.
pub fn load_with_restored(
    path: &str,
    dtype: DType,
    device: &Device,
    kv: crate::kvq::KvMode,
    restore: &RestoreF16,
    checkpoint: Option<&Checkpoint>,
) -> anyhow::Result<SealedModel> {
    let bytes = std::fs::metadata(path)?.len();
    let f = std::fs::File::open(path)?;
    let mut r = std::io::BufReader::with_capacity(1 << 20, f);
    let head = llvq_artifact::read_header(&mut r)?;
    anyhow::ensure!(
        head.is_self_contained(),
        "{path} is a projections-only artifact (format v{}). It cannot run on \
         its own — seal it first:\n  LLVQ_MODEL=<repo> cargo run --release -p \
         llvq-llm --bin seal -- {path} sealed.llvq",
        head.version
    );

    let ix = llvq_search::index::Indexer::new();
    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    let mut quantized_weights = 0usize;
    // One matrix at a time: a 4B model's lattice codes are 14 GB if held
    // together, and the decoded weights are handed to candle as they come.
    for _ in 0..head.matrices {
        let m = llvq_artifact::read_matrix(&mut r, &ix)?;
        quantized_weights += m.d_out * m.d_in;
        let w = llvq_artifact::decode_matrix(&m);
        let t = Tensor::from_vec(w, (m.d_out, m.d_in), device)?.to_dtype(dtype)?;
        tensors.insert(m.name, t);
    }

    let n_raw = read_u32(&mut r)?;
    let mut carried_weights = 0usize;
    for _ in 0..n_raw {
        let t = llvq_artifact::read_raw(&mut r, head.version)?;
        carried_weights += t.len();
        // f16 stays on the narrow path (no f32 blow-up of a 778 MB embedding);
        // a quantized tensor goes through the format's own decoder, so what is
        // evaluated is what the file stores.
        let tensor = match &t.data {
            llvq_artifact::RawData::F16(d) => {
                let vals: Vec<half::f16> = d.iter().map(|b| half::f16::from_bits(*b)).collect();
                Tensor::from_vec(vals, t.dims.clone(), device)?.to_dtype(dtype)?
            }
            llvq_artifact::RawData::Quant(_) => {
                Tensor::from_vec(t.to_f32(), t.dims.clone(), device)?.to_dtype(dtype)?
            }
        };
        tensors.insert(t.name, tensor);
    }

    let n_blob = read_u32(&mut r)?;
    let mut blobs: HashMap<String, Vec<u8>> = HashMap::new();
    for _ in 0..n_blob {
        let b = llvq_artifact::read_blob(&mut r)?;
        blobs.insert(b.name, b.bytes);
    }

    let config: Config = serde_json::from_slice(
        blobs
            .get("config.json")
            .ok_or_else(|| anyhow::anyhow!("{path} carries no config.json"))?,
    )?;
    let tokenizer = tokenizers::Tokenizer::from_bytes(
        blobs
            .get("tokenizer.json")
            .ok_or_else(|| anyhow::anyhow!("{path} carries no tokenizer.json"))?,
    )
    .map_err(|e| anyhow::anyhow!("{e}"))?;

    // ---- the restoration, after every sealed tensor is in and before the
    // model is built: it only ever touches projection names, and it must see
    // the complete map to refuse a type the file does not carry.
    let (restored, restored_from) = match (restore.is_empty(), checkpoint) {
        (true, _) => (Restored::default(), None),
        (false, None) => anyhow::bail!(
            "LLVQ_RESTORE_F16={} demande un checkpoint à lire (LLVQ_MODEL) : \
             les matrices restaurées viennent de là, pas du fichier scellé",
            restore.describe()
        ),
        (false, Some(ck)) => {
            // Safety: the checkpoint files are not modified while the mapping
            // is alive — the same contract as `Checkpoint::var_builder`, which
            // is the other reader of these very files.
            let st = unsafe { candle_core::safetensors::MmapedSafetensors::multi(&ck.weights)? };
            // Loaded then narrowed, in that order, which is what `VarBuilder`
            // does for the dense arm: a restored matrix and the checkpoint
            // arm's own copy of it are the same f16 bytes.
            let r = restore_projections(&mut tensors, restore, |name| {
                Ok(st.load(name, device)?.to_dtype(dtype)?)
            })?;
            (r, Some(describe_source(&ck.source)))
        }
    };

    let vb = VarBuilder::from_tensors(tensors, dtype, device);
    let model = Qwen3::new(&config, vb, kv)?;
    Ok(SealedModel {
        model,
        tokenizer,
        config,
        quantized_weights,
        carried_weights,
        matrices: head.matrices,
        raw_tensors: n_raw,
        bytes,
        restored_types: restore.clone(),
        restored,
        restored_from,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A safetensors overlay is a dump of reconstructions on top of a
    /// checkpoint, not a model. Treating one as sealed would load a file that
    /// carries no config and fail deep inside the reader, so the distinction
    /// is made on the way in.
    #[test]
    fn an_overlay_is_not_a_sealed_artifact() {
        assert!(is_sealed_path("qwen3-4b-llvq.bin"));
        assert!(is_sealed_path("/home/x/llvq-q4b.llvq"));
        assert!(!is_sealed_path("llvq-q4b-c12.safetensors"));
        assert!(!is_sealed_path("Qwen/Qwen3-4B"));
    }

    /// Repo ids are the other half of the same decision, and one of them ends
    /// in a digit-and-letter suffix that must not be mistaken for a suffix.
    #[test]
    fn repo_ids_are_never_sealed_paths() {
        for id in ["Qwen/Qwen3-0.6B", "Qwen/Qwen3-4B", "meta-llama/Llama-3-8B"] {
            assert!(!is_sealed_path(id), "{id}");
        }
    }
}

#[cfg(test)]
mod restore_tests {
    use super::*;

    fn t(v: f32, dims: (usize, usize)) -> Tensor {
        Tensor::full(v, dims, &Device::Cpu).unwrap()
    }

    /// Two layers, the seven projections each, plus the tensors a sealed file
    /// carries whole — filled with a marker value so "untouched" is checkable.
    fn artifact(marker: f32) -> HashMap<String, Tensor> {
        let mut m = HashMap::new();
        for b in 0..2 {
            for p in ["q_proj", "k_proj", "v_proj", "o_proj"] {
                m.insert(
                    format!("model.layers.{b}.self_attn.{p}.weight"),
                    t(marker, (4, 8)),
                );
            }
            for p in ["gate_proj", "up_proj", "down_proj"] {
                m.insert(
                    format!("model.layers.{b}.mlp.{p}.weight"),
                    t(marker, (6, 8)),
                );
            }
            m.insert(
                format!("model.layers.{b}.input_layernorm.weight"),
                t(marker, (1, 8)),
            );
        }
        m.insert("model.embed_tokens.weight".into(), t(marker, (16, 8)));
        m
    }

    fn value(m: &HashMap<String, Tensor>, name: &str) -> f32 {
        m[name].flatten_all().unwrap().to_vec1::<f32>().unwrap()[0]
    }

    #[test]
    fn restore_spec_parses_the_seven_names_and_refuses_the_rest() {
        assert!(RestoreF16::parse("").unwrap().is_empty());
        assert!(RestoreF16::parse("   ").unwrap().is_empty());
        assert_eq!(RestoreF16::parse("k_proj").unwrap().types(), &["k_proj"]);
        assert_eq!(
            RestoreF16::parse(" q_proj, k_proj ").unwrap().types(),
            &["q_proj", "k_proj"]
        );
        assert_eq!(RestoreF16::parse("all").unwrap().types(), &PROJ_TYPES);
        for bad in ["k", "K_PROJ", "attn", "k_proj,k_proj", "q_proj,,k_proj"] {
            let e = RestoreF16::parse(bad).expect_err(bad);
            assert!(e.contains("LLVQ_RESTORE_F16"), "{e}");
        }
    }

    #[test]
    fn covers_matches_the_tensor_name_segment_and_nothing_looser() {
        let r = RestoreF16::parse("k_proj").unwrap();
        assert!(r.covers("model.layers.7.self_attn.k_proj.weight"));
        assert!(!r.covers("model.layers.7.self_attn.q_proj.weight"));
        // A prefix is not a match: `k_proj` must not cover a hypothetical
        // `xk_proj`, nor a bias.
        assert!(!r.covers("model.layers.7.self_attn.xk_proj.weight"));
        assert!(!r.covers("model.layers.7.self_attn.k_proj.bias"));
    }

    #[test]
    fn restoring_nothing_touches_nothing_and_never_calls_the_source() {
        let mut m = artifact(1.0);
        let done = restore_projections(&mut m, &RestoreF16::default(), |n| {
            panic!("source consulted for {n}")
        })
        .unwrap();
        assert_eq!(done, Restored::default());
        assert!(m
            .values()
            .all(|v| v.flatten_all().unwrap().to_vec1::<f32>().unwrap()[0] == 1.0));
    }

    #[test]
    fn restoring_one_type_replaces_exactly_that_type() {
        let mut m = artifact(1.0);
        let r = RestoreF16::parse("k_proj").unwrap();
        let done = restore_projections(&mut m, &r, |n| Ok(t(9.0, m_dims(n)))).unwrap();
        assert_eq!(done.matrices, 2);
        assert_eq!(done.weights, 2 * 4 * 8);
        for b in 0..2 {
            assert_eq!(
                value(&m, &format!("model.layers.{b}.self_attn.k_proj.weight")),
                9.0
            );
            assert_eq!(
                value(&m, &format!("model.layers.{b}.self_attn.q_proj.weight")),
                1.0
            );
            assert_eq!(
                value(&m, &format!("model.layers.{b}.mlp.down_proj.weight")),
                1.0
            );
        }
        assert_eq!(value(&m, "model.embed_tokens.weight"), 1.0);
    }

    fn m_dims(name: &str) -> (usize, usize) {
        if name.contains("mlp") {
            (6, 8)
        } else {
            (4, 8)
        }
    }

    #[test]
    fn restoring_all_seven_leaves_no_quantized_projection() {
        let mut m = artifact(1.0);
        let done = restore_projections(&mut m, &RestoreF16::parse("all").unwrap(), |n| {
            Ok(t(9.0, m_dims(n)))
        })
        .unwrap();
        assert_eq!(done.matrices, 14);
        assert_eq!(done.weights, 2 * (4 * 4 * 8 + 3 * 6 * 8));
        for (name, v) in &m {
            let x = v.flatten_all().unwrap().to_vec1::<f32>().unwrap()[0];
            if name.ends_with("_proj.weight") {
                assert_eq!(x, 9.0, "{name}");
            } else {
                assert_eq!(x, 1.0, "{name}");
            }
        }
    }

    #[test]
    fn a_type_the_artifact_lacks_is_refused_not_skipped() {
        let mut m = artifact(1.0);
        m.retain(|n, _| !n.contains("down_proj"));
        let e = restore_projections(&mut m, &RestoreF16::parse("down_proj").unwrap(), |n| {
            Ok(t(9.0, m_dims(n)))
        })
        .unwrap_err();
        assert!(e.to_string().contains("down_proj"), "{e}");
    }

    #[test]
    fn a_shape_the_checkpoint_disagrees_on_is_refused() {
        let mut m = artifact(1.0);
        let e = restore_projections(&mut m, &RestoreF16::parse("k_proj").unwrap(), |_| {
            Ok(t(9.0, (4, 9)))
        })
        .unwrap_err();
        assert!(e.to_string().contains("le même modèle"), "{e}");
    }

    #[test]
    fn a_name_the_source_cannot_serve_aborts_the_load() {
        let mut m = artifact(1.0);
        let e = restore_projections(&mut m, &RestoreF16::parse("v_proj").unwrap(), |n| {
            anyhow::bail!("{n} absent du checkpoint")
        })
        .unwrap_err();
        assert!(e.to_string().contains("absent du checkpoint"), "{e}");
    }

    /// The load path used for a real checkpoint — `MmapedSafetensors::multi`
    /// then `load` then `to_dtype` — must yield the same bytes as the
    /// `VarBuilder` the dense arm reads the same file through. That equality
    /// is what makes "all seven restored" the checkpoint arm and not an
    /// approximation of it.
    #[test]
    fn the_mmap_source_narrows_like_the_var_builder_does() {
        let dir = std::env::temp_dir().join(format!(
            "llvq_restore_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("model.safetensors");
        // bf16 on disk, as Qwen3 checkpoints are; values that do not round
        // trip through f16 unchanged, so a narrowing done differently shows.
        let w = Tensor::arange(0f32, 24., &Device::Cpu)
            .unwrap()
            .reshape((4, 6))
            .unwrap()
            .affine(1.0 / 3.0, 0.1234567)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let mut m = HashMap::new();
        m.insert("model.layers.0.self_attn.k_proj.weight".to_string(), w);
        candle_core::safetensors::save(&m, &file).unwrap();

        let st = unsafe { candle_core::safetensors::MmapedSafetensors::multi(&[&file]).unwrap() };
        let ours = st
            .load("model.layers.0.self_attn.k_proj.weight", &Device::Cpu)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[&file], DType::F16, &Device::Cpu).unwrap()
        };
        let theirs = vb
            .get((4, 6), "model.layers.0.self_attn.k_proj.weight")
            .unwrap();
        let a = ours.flatten_all().unwrap().to_vec1::<half::f16>().unwrap();
        let b = theirs
            .flatten_all()
            .unwrap()
            .to_vec1::<half::f16>()
            .unwrap();
        assert_eq!(a, b);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
