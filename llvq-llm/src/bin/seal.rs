//! Turn a projections-only artifact into a self-contained model.
//!
//! Usage:
//!   `LLVQ_MODEL=Qwen/Qwen3-4B cargo run --release -p llvq-llm --bin seal -- in.llvq out.llvq`
//!
//! A file holding only the quantized linear layers still needs the original
//! checkpoint beside it to run — 981 MB that requires 8 GB of company. This
//! copies in everything else: every tensor the quantizer did not touch, plus
//! the config and the tokenizer.
//!
//! **The tensor set is a complement, not a list.** Everything in the
//! checkpoint that is not a quantized projection gets carried over. Writing
//! out "embedding, norms, q_norm, k_norm" by hand works until an architecture
//! has one more weight and the model silently loads a zero.
//!
//! Requires the checkpoint. That is a *build* dependency: the file it produces
//! has none.

use candle_core::{DType, Device};
use llvq_artifact::{ArtifactWriter, Blob, RawTensor};
use llvq_llm::loader::Checkpoint;
use std::collections::HashSet;

/// One record on its way into the sealed file: a lattice matrix decoded (so a
/// bad index fails while the checkpoint is still being read) and its kind, or
/// an int4 matrix carried as it is.
enum Sealing {
    Lattice(llvq_artifact::CodeKind, llvq_artifact::QuantizedMatrix),
    Int4(llvq_artifact::Int4Matrix),
}

impl Sealing {
    fn name(&self) -> &str {
        match self {
            Self::Lattice(_, m) => &m.name,
            Self::Int4(m) => &m.name,
        }
    }
    fn weights(&self) -> usize {
        match self {
            Self::Lattice(_, m) => m.d_out * m.d_in,
            Self::Int4(m) => m.d_out * m.d_in,
        }
    }
}

/// The checkpoint to seal from: `LLVQ_MODEL`, required, and equal to the
/// `model` the encoding run wrote in `<in>.state` when that sidecar exists.
///
/// The sealed file takes its norms, embedding, config and tokenizer from this
/// checkpoint and its projections from the artifact. A checkpoint of the same
/// shapes but other weights (a Base or 2507 variant) would load and run with
/// no message, so the name is compared as written on both sides, and there is
/// no default to fall back on.
fn checkpoint_name(
    model: Option<String>,
    state: Option<&llvq_llm::artifact2::RunState>,
    state_path: &str,
) -> anyhow::Result<String> {
    let repo = model.filter(|m| !m.is_empty()).ok_or_else(|| {
        anyhow::anyhow!(
            "LLVQ_MODEL=<checkpoint> is required: the sealed file takes every tensor the \
             artifact does not hold from it, so it must be the checkpoint the artifact was \
             encoded from"
        )
    })?;
    if let Some(encoded) = state.and_then(|s| s.get("model")) {
        anyhow::ensure!(
            encoded == repo,
            "LLVQ_MODEL={repo}, but {state_path} says the artifact was encoded from \
             model = {encoded}: the sealed file would pair one model's projections with \
             another's norms, embedding and tokenizer"
        );
    }
    Ok(repo)
}

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let (src, dst) = match (a.first(), a.get(1)) {
        (Some(s), Some(d)) => (s.clone(), d.clone()),
        _ => anyhow::bail!("usage: seal <in.llvq> <out.llvq>"),
    };
    // The sidecar `bin/smoke` writes beside the artifact names the checkpoint
    // it encoded. A file without one (copied alone, or older) is sealed on
    // `LLVQ_MODEL` alone, and the line below says so.
    let state_path = llvq_llm::artifact2::RunState::path_for(&src);
    let state = if std::path::Path::new(&state_path).exists() {
        Some(llvq_llm::artifact2::RunState::read(&src)?)
    } else {
        None
    };
    let repo = checkpoint_name(std::env::var("LLVQ_MODEL").ok(), state.as_ref(), &state_path)?;
    match state.as_ref().and_then(|s| s.get("model")) {
        Some(_) => eprintln!("checkpoint {repo}: the one {state_path} names"),
        None => eprintln!("checkpoint {repo}: no model recorded beside {src}, not cross-checked"),
    }

    // ---- read the quantized side ----
    let f = std::fs::File::open(&src)?;
    let mut r = std::io::BufReader::with_capacity(1 << 20, f);
    let head = llvq_artifact::read_header(&mut r)?;
    eprintln!(
        "{src}: format v{}, {} quantized matrices, kinds {}",
        head.version,
        head.matrices,
        head.kinds()
    );
    // Each record is decoded through the map it names, and its kind is kept
    // beside it: the sealed file has to carry the same kinds as the source,
    // record by record. Decoding here rather than copying raw is deliberate —
    // an index outside its codebook must fail while the checkpoint is still
    // being read, not the first time the sealed file is loaded to be scored.
    let cbs = llvq_artifact::Codebooks::new();
    let mut matrices: Vec<Sealing> =
        Vec::with_capacity((head.matrices as usize).min(llvq_llm::fused::PREALLOC_CAP));
    for _ in 0..head.matrices {
        // An int4 record is carried across untouched: it holds its weights,
        // so there is no map to check it against and nothing to re-encode.
        matrices.push(match llvq_artifact::read_record(&mut r, head.version)? {
            llvq_artifact::Record::Lattice(raw) => {
                let kind = raw.kind;
                Sealing::Lattice(kind, llvq_llm::artifact2::to_quantized(raw, &cbs)?)
            }
            llvq_artifact::Record::Int4(m) => Sealing::Int4(m),
        });
    }
    let quantized: HashSet<&str> = matrices.iter().map(Sealing::name).collect();
    let quantized_weights: usize = matrices.iter().map(Sealing::weights).sum();

    // ---- everything the quantizer did not touch ----
    eprintln!("reading {repo} for the tensors the artifact is missing…");
    let ck = Checkpoint::fetch(&repo)?;
    let device = Device::Cpu;
    let mut raws: Vec<RawTensor> = Vec::new();
    let mut carried = 0usize;
    for path in &ck.weights {
        let shard = candle_core::safetensors::load(path, &device)?;
        for (name, t) in shard {
            if quantized.contains(name.as_str()) {
                continue;
            }
            // A checkpoint that ties its two ends may still ship the head as
            // its own tensor (Qwen3-0.6B does). The loader reads the embedding
            // for both under `tie_word_embeddings` and never this copy, which
            // a sealed file would carry for nothing: 311 MB at f16 for the 0.6B.
            if ck.config.tie_word_embeddings && name == llvq_llm::fused::HEAD_NAME {
                eprintln!("  not carrying {name}: tie_word_embeddings is true, the embedding is the head");
                continue;
            }
            let dims = t.dims().to_vec();
            let data: Vec<u16> = t
                .to_dtype(DType::F16)?
                .flatten_all()?
                .to_vec1::<half::f16>()?
                .into_iter()
                .map(|v| v.to_bits())
                .collect();
            carried += data.len();
            raws.push(RawTensor {
                name,
                dims,
                data: llvq_artifact::RawData::F16(data),
            });
        }
    }
    raws.sort_by(|a, b| a.name.cmp(&b.name));
    eprintln!(
        "  carrying {} tensors, {carried} weights ({:.3} GB at f16)",
        raws.len(),
        carried as f64 * 2.0 / 1e9
    );

    let blobs = vec![
        Blob {
            name: "config.json".into(),
            bytes: std::fs::read(&ck.config_path)?,
        },
        Blob {
            name: "tokenizer.json".into(),
            bytes: std::fs::read(&ck.tokenizer)?,
        },
    ];

    // ---- write the sealed file ----
    let out = std::io::BufWriter::with_capacity(
        1 << 20,
        std::fs::File::create(&dst)?,
    );
    // The sealed file inherits the source's version and kinds. `max` with
    // [`llvq_artifact::DEFAULT_VERSION`] is what upgrades a projections-only
    // v1 file to a self-contained one — sealing at v1 would produce a file
    // `Header::is_self_contained` refuses — and it leaves a v4 Ball source
    // sealed at v4, byte for byte what it always was.
    let version = head.version.max(llvq_artifact::DEFAULT_VERSION);
    let mut w = ArtifactWriter::with_kinds(
        out,
        version,
        matrices.len() as u32,
        head.default_kind(),
        head.kinds(),
    )?;
    for m in &matrices {
        match m {
            Sealing::Lattice(kind, m) => w.push_kind(m, *kind)?,
            Sealing::Int4(m) => w.push_int4(m)?,
        }
    }
    // `code_bits` comes back from the writer, which adds each record's own
    // payload — `Int4Matrix::bits()` included. The rate line below therefore
    // counts the int4 half at its true 4.250 b/weight rather than at nothing.
    let (code_bits, extra_bits) = w.seal(&raws, &blobs)?;

    let bytes = std::fs::metadata(&dst)?.len();
    let fp16 = (quantized_weights + carried) * 2;
    println!("\n── {dst}");
    println!(
        "   quantized  {quantized_weights:>12} weights → {:.3} GB ({:.4} bits/weight)",
        code_bits as f64 / 8.0 / 1e9,
        code_bits as f64 / quantized_weights as f64
    );
    println!(
        "   carried    {carried:>12} weights → {:.3} GB at f16",
        extra_bits as f64 / 8.0 / 1e9
    );
    println!("   total          {:.3} GB on disk", bytes as f64 / 1e9);
    println!(
        "   against        {:.3} GB in FP16  →  ×{:.2}",
        fp16 as f64 / 1e9,
        fp16 as f64 / bytes as f64
    );
    println!("\n   this file needs no checkpoint, no cache and no network.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use llvq_llm::artifact2::RunState;

    #[test]
    fn the_checkpoint_is_required_and_must_be_the_encoded_one() {
        let mut st = RunState::new();
        st.set("model", "Qwen/Qwen3-4B");
        let p = "a.llvq.state";
        // Absent or empty: refused, with or without a sidecar, never the 0.6B.
        for m in [None, Some(String::new())] {
            for state in [Some(&st), None] {
                let e = checkpoint_name(m.clone(), state, p).unwrap_err();
                assert!(format!("{e}").contains("LLVQ_MODEL=<checkpoint> is required"), "{e}");
            }
        }
        // The encoded one: accepted, as written.
        assert_eq!(
            checkpoint_name(Some("Qwen/Qwen3-4B".into()), Some(&st), p).unwrap(),
            "Qwen/Qwen3-4B"
        );
        // Another model of the same shapes: refused, naming both.
        let e = checkpoint_name(Some("Qwen/Qwen3-4B-Base".into()), Some(&st), p).unwrap_err();
        let e = format!("{e}");
        assert!(e.contains("Qwen/Qwen3-4B-Base") && e.contains("model = Qwen/Qwen3-4B"), "{e}");
        // No sidecar, or one that records no model: nothing to compare against.
        assert!(checkpoint_name(Some("x".into()), None, p).is_ok());
        assert!(checkpoint_name(Some("x".into()), Some(&RunState::new()), p).is_ok());
    }
}
