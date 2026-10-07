//! One SHA-256 per record of a sealed file's dequantized f32 weights.
//!
//! Usage: `cargo run --release -p llvq-llm --bin hfdense -- model.llvq out.json`
//!
//! The reference of gate A of stage 1
//! (`proofs/preregistration-hf-quantizer-2026-09-30.md`). It writes digests and
//! never the weights: the dense f32 of a 4B is 14 GB and `bin/export` is the
//! tool that wants it on disk.

use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let src: PathBuf = a
        .first()
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("give the path to a sealed .llvq model"))?
        .into();
    let out: PathBuf = a
        .get(1)
        .cloned()
        .unwrap_or_else(|| llvq_llm::hfpack::DENSE_DIGEST_FILE.into())
        .into();
    eprintln!("decoding {} …", src.display());
    let s = llvq_llm::hfpack::dense_digest(&src, &out)?;
    println!(
        "\n{} → {}\n  {} records, {} quantized raw tensors, {:.3} B weights decoded\n",
        src.display(),
        out.display(),
        s.records,
        s.quantized_tensors,
        s.weights as f64 / 1e9
    );
    Ok(())
}
