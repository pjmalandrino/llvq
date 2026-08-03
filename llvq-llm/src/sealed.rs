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

use crate::model::Qwen3;

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
pub fn load(path: &str, dtype: DType, device: &Device) -> anyhow::Result<SealedModel> {
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
        let t = llvq_artifact::read_raw(&mut r)?;
        carried_weights += t.data.len();
        let vals: Vec<half::f16> = t.data.iter().map(|b| half::f16::from_bits(*b)).collect();
        let tensor = Tensor::from_vec(vals, t.dims.clone(), device)?.to_dtype(dtype)?;
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

    let vb = VarBuilder::from_tensors(tensors, dtype, device);
    let model = Qwen3::new(&config, vb)?;
    Ok(SealedModel {
        model,
        tokenizer,
        config,
        quantized_weights,
        carried_weights,
        matrices: head.matrices,
        raw_tensors: n_raw,
        bytes,
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
