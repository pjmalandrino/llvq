//! Dump the Tetra decode tables for a reader that does not run Rust.
//!
//! Usage: `cargo run --release -p llvq-llm --bin tetratables -- out.safetensors`
//!
//! Writes the arrays beside a `.json` of the constants and of the Tetra
//! fingerprint. The tables are universal, one per codebook and not one per
//! model, which is why they ship with the Python package of
//! `docs/plan-transformers.md` stage 1 rather than inside every `.llvq`.

use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let out: PathBuf = std::env::args()
        .nth(1)
        .ok_or_else(|| anyhow::anyhow!("give the output path, a .safetensors"))?
        .into();
    if let Some(dir) = out.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let (fingerprint, n) = llvq_llm::hfpack::tetra_tables(&out)?;
    println!(
        "{} tables → {}\n  tetra fingerprint {fingerprint}\n  constants → {}\n",
        n,
        out.display(),
        out.with_extension("json").display()
    );
    Ok(())
}
