//! Decode `n` random Tetra labels in Rust, for a reader to compare against.
//!
//! Usage: `cargo run --release -p llvq-llm --bin tetravectors -- <n> <seed> out.safetensors`
//!
//! Control 2 of `proofs/preregistration-hf-quantizer-2026-09-30.md`: gate A
//! exercises the labels the 4B happens to hold, and this one samples the 47-bit
//! space. Every 48-bit value is a label (`llvq_search::tetra`), so a draw needs
//! no rejection.

use candle_core::{Device, Tensor};
use llvq_core::{SplitMix64, DIM};
use llvq_search::tetra::{Tetra, LABEL_MASK};
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let n: usize = a.first().map(|s| s.parse()).transpose()?.unwrap_or(100_000);
    let seed: u64 = a.get(1).map(|s| s.parse()).transpose()?.unwrap_or(0xF1_0001);
    let out: PathBuf = a
        .get(2)
        .cloned()
        .unwrap_or_else(|| "tetra-vectors.safetensors".into())
        .into();

    let tetra = Tetra::new();
    let mut rng = SplitMix64::new(seed);
    let mut labels = Vec::with_capacity(n);
    let mut points = Vec::with_capacity(n * DIM);
    for _ in 0..n {
        let label = rng.next() & LABEL_MASK;
        labels.push(label as i64);
        points.extend(tetra.decode(label).iter().map(|&v| v as i64));
    }
    let device = Device::Cpu;
    let tensors = std::collections::HashMap::from([
        ("labels".to_string(), Tensor::from_vec(labels, n, &device)?),
        ("points".to_string(), Tensor::from_vec(points, (n, DIM), &device)?),
    ]);
    candle_core::safetensors::save(&tensors, &out)?;
    println!("{n} labels at seed {seed:#x} → {}", out.display());
    Ok(())
}
