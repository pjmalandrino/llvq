//! `inkwh`: the generic `(W, H)` driver of lot 1, item 1b.
//!
//! ```text
//! inkwh <manifest.json> <out_dir>
//! ```
//!
//! Encodes every matrix of the manifest (see `llvq_llm::inkseal::run_wh` for
//! its fields) in the sealed Tetra recipe of `llvq_llm::extq` against the
//! group's Hessian, or as an int4 g128 record, and writes per matrix one v5
//! record (`<name>.llvqrec`, readable by `inkseal seal --records`) and the
//! dense reconstruction (`<name>.f32`), plus `inkwh.json`.
//!
//! Widths: any `d_in`. A width that is not a multiple of 24 keeps its last
//! `d_in % 24` columns as the format's f32 tail, as the block loop does
//! (`TailPolicy::KeepExact`); an int4 record refuses a `d_in` that is not a
//! multiple of 128, by name. Inkling's attention inputs (2,304 and 3,072) are
//! multiples of both.
//!
//! Build with `--features fast-linalg` (the published factorization; refused
//! without it). Threads from `LLVQ_THREADS` (default 2).

use anyhow::{bail, Result};
use std::path::PathBuf;

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let [manifest, out] = a.as_slice() else { bail!("usage: inkwh <manifest.json> <out_dir>") };
    if !cfg!(feature = "fast-linalg") {
        bail!("inkwh must be built with --features fast-linalg (the published factorization path)");
    }
    let t0 = std::time::Instant::now();
    let s = llvq_llm::inkseal::run_wh(&PathBuf::from(manifest), &PathBuf::from(out), llvq_llm::inkseal::threads()?)?;
    println!("{}", serde_json::to_string_pretty(&s)?);
    eprintln!("inkwh: {:.1} s", t0.elapsed().as_secs_f64());
    Ok(())
}
