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

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let (src, dst) = match (a.first(), a.get(1)) {
        (Some(s), Some(d)) => (s.clone(), d.clone()),
        _ => anyhow::bail!("usage: seal <in.llvq> <out.llvq>"),
    };
    let repo = std::env::var("LLVQ_MODEL").unwrap_or_else(|_| "Qwen/Qwen3-0.6B".into());

    // ---- read the quantized side ----
    let f = std::fs::File::open(&src)?;
    let mut r = std::io::BufReader::with_capacity(1 << 20, f);
    let head = llvq_artifact::read_header(&mut r)?;
    eprintln!(
        "{src}: format v{}, {} quantized matrices",
        head.version, head.matrices
    );
    let ix = llvq_search::index::Indexer::new();
    let mut matrices = Vec::with_capacity(head.matrices as usize);
    for _ in 0..head.matrices {
        matrices.push(llvq_artifact::read_matrix(&mut r, &ix)?);
    }
    let quantized: HashSet<&str> = matrices.iter().map(|m| m.name.as_str()).collect();
    let quantized_weights: usize = matrices.iter().map(|m| m.d_out * m.d_in).sum();

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
    let mut w = ArtifactWriter::new(out, matrices.len() as u32)?;
    for m in &matrices {
        w.push(m)?;
    }
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
