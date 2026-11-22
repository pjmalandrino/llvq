//! `inktable`: int4 g64 tables (lot 1, item 1c).
//!
//! ```text
//! inktable <model.safetensors> <tensor name> <out.llvqraw> [--name sealed_name]
//! ```
//!
//! Reads one table from a safetensors file, narrows it to f16 exactly as
//! `bin/seal` does (candle's `to_dtype(F16)`), and applies
//! `embedquant::quantize_affine` at 4 bits, groups of 64: the bytes
//! `embedq q4` writes for the same table. The output holds the one raw tensor
//! in the sealed stream's tagged framing, which `inkseal seal --raw` reads.
//! Its name is `--name`, else the native name through `inkseal::rename`
//! (`model.llm.embed.weight` becomes `model.embed.weight`).
//!
//! For Inkling 2 Mini: `model.llm.embed.weight` and `model.llm.unembed.weight`
//! (201,024 × 2,304 each, untied, BF16).

use anyhow::{bail, Context, Result};
use llvq_llm::inkseal;
use std::path::PathBuf;

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let (mut pos, mut name) = (Vec::new(), None);
    let mut it = a.iter();
    while let Some(x) = it.next() {
        match x.as_str() {
            "--name" => name = Some(it.next().context("--name needs a value")?.clone()),
            _ if x.starts_with("--") => bail!("unknown flag {x}"),
            _ => pos.push(x.clone()),
        }
    }
    let [st, tensor, out] = pos.as_slice() else {
        bail!("usage: inktable <model.safetensors> <tensor name> <out.llvqraw> [--name sealed_name]")
    };
    let t0 = std::time::Instant::now();
    let st = inkseal::read_safetensors(&PathBuf::from(st))?;
    let out_name = name.unwrap_or_else(|| inkseal::rename(tensor));
    let q = inkseal::table_q4(&st, tensor, &out_name)?;
    inkseal::write_raw_file(&PathBuf::from(out), &q)?;
    println!(
        "{tensor} -> {out_name} {:?}: int4 g{}, {} bytes payload ({:.4} b/weight), written to {out} in {:.1} s",
        q.dims,
        inkseal::TABLE_GROUP,
        q.bytes(),
        q.bytes() as f64 * 8.0 / q.len() as f64,
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}
