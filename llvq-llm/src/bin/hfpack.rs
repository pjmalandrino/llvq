//! Write a sealed `.llvq` as a Hugging Face directory, still compressed.
//!
//! Usage: `cargo run --release -p llvq-llm --bin hfpack -- model.llvq out_dir/`
//!
//! Stage 0 of `docs/plan-transformers.md`. The work is in
//! [`llvq_llm::hfpack`], which is where the tests reach it: a binary's `main`
//! is not callable from a test, and that is how an int4 export arm went
//! unexercised for a day on 2026-09-19.
//!
//! This writes a **distribution** artifact, unlike `bin/export`, which
//! dequantizes to f16 and is as large as the model. The gate on what this one
//! writes is `ops/llvq_hf_check.py`, an independent reader, and not this
//! binary's own round trip.

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
        .unwrap_or_else(|| "llvq-hf".into())
        .into();

    eprintln!("reading {} …", src.display());
    let s = llvq_llm::hfpack::pack(&src, &out)?;

    println!(
        "\n{} → {}\n  {} records: {} Tetra, {} Int4G128\n  \
         {} rotations, {} raw tensors, {} blobs, {} tensors written\n  \
         {:.2} B weights quantized, {:.0} M carried\n  \
         {}: {:.3} GB, {}: {:.1} KB, directory {:.3} GB\n",
        src.display(),
        out.display(),
        s.records,
        s.lattice,
        s.int4,
        s.rotations,
        s.raw_tensors,
        s.blobs,
        s.tensors,
        s.quantized_weights as f64 / 1e9,
        s.carried_weights as f64 / 1e6,
        llvq_llm::hfpack::MODEL_FILE,
        s.model_bytes as f64 / 1e9,
        llvq_llm::hfpack::CONFIG_FILE,
        s.config_bytes as f64 / 1e3,
        s.dir_bytes as f64 / 1e9,
    );
    println!(
        "Next, the gate of stage 0:\n  uv run ops/llvq_hf_check.py {}\n",
        out.display()
    );
    Ok(())
}
