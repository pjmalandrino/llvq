//! Load a sealed `.llvq` model and generate from it — no checkpoint, no cache,
//! no network.
//!
//! Usage:
//!   `cargo run --release -p llvq-llm --features metal --bin run -- model.llvq [device] [n_new]`
//!
//! This is the deliverable's own test. Every other number in this project is a
//! claim about what a file *would* weigh; this one opens the file, rebuilds a
//! model out of it, and makes it answer. If it needs anything else on disk, it
//! is not a deployable model and the number was decoration.
//!
//! A projections-only file (format v1) is refused with the command that fixes
//! it, rather than silently reaching for the checkpoint. That refusal lives in
//! [`llvq_llm::sealed`], shared with `bin/mmlu` and `bin/ppl`.

use candle_core::{DType, Device};
use llvq_llm::model::NoCapture;

const PROMPTS: &[&str] = &[
    "The capital of France is",
    "In 1969, the first humans landed on",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return",
    "Water boils at a temperature of",
];

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let path = a
        .first()
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("give the path to a .llvq model"))?;
    let device = if a.get(1).map(|s| s == "metal").unwrap_or(false) {
        Device::new_metal(0).unwrap_or(Device::Cpu)
    } else {
        Device::Cpu
    };
    let n_new: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(24);
    // f16 rather than f32: the model ran at half precision anyway, and the
    // difference is 7 GB of RAM against 15 on a 4B. `LLVQ_DTYPE=f32` puts the
    // generation back on the exact weights `verify_artifact` round-trips —
    // the narrowing here is outside that proof.
    let dtype = llvq_llm::eval::dtype(DType::F16)?;

    let s = llvq_llm::sealed::load(&path, dtype, &device)?;
    let fp16 = (s.quantized_weights + s.carried_weights) * 2;
    eprintln!(
        "loaded {path} — {} quantized matrices + {} carried tensors\n",
        s.matrices, s.raw_tensors
    );
    println!("── model");
    println!("   running at dtype {}", llvq_llm::eval::dtype_name(dtype));
    println!(
        "   {:.3} GB on disk against {:.3} GB in FP16  →  ×{:.2}",
        s.bytes as f64 / 1e9,
        fp16 as f64 / 1e9,
        fp16 as f64 / s.bytes as f64
    );
    println!(
        "   {} weights quantized, {} carried at f16\n",
        s.quantized_weights, s.carried_weights
    );

    for p in PROMPTS {
        let ids = s
            .tokenizer
            .encode(*p, false)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .get_ids()
            .to_vec();
        let out = s.model.generate(&ids, n_new, &mut NoCapture)?;
        let text = s
            .tokenizer
            .decode(&out, false)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        println!("── {p:?}\n   →{text}");
    }
    Ok(())
}
