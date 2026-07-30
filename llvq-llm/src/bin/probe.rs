//! Side-by-side generation: the intact model against the quantized one.
//!
//! Usage: `cargo run --release -p llvq-llm --features metal --bin probe -- <quantized.safetensors> [device] [n_new]`
//!
//! Perplexity is the metric every quantization paper reports, and it is the
//! only one that makes our numbers comparable — but it is a statistic over a
//! fixed corpus, not evidence that the model still answers. A drop of a few
//! points of perplexity can coexist with answers that quietly go wrong, which
//! is the whole point of arXiv:2607.08734. So: read the output.
//!
//! Greedy decoding on both sides, so any difference is the codebook and not
//! sampling noise.

use candle_core::{DType, Device};
use llvq_llm::loader::Checkpoint;
use llvq_llm::model::{NoCapture, Qwen3};

const PROMPTS: &[&str] = &[
    "The capital of France is",
    "In 1969, the first humans landed on",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return",
    "The three primary colours are red, blue and",
    "Water boils at a temperature of",
];

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let path = a
        .first()
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("give the path to a quantized safetensors"))?;
    let device = if a.get(1).map(|s| s == "metal").unwrap_or(false) {
        Device::new_metal(0).unwrap_or(Device::Cpu)
    } else {
        Device::Cpu
    };
    let n_new: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(24);

    let repo = std::env::var("LLVQ_MODEL").unwrap_or_else(|_| "Qwen/Qwen3-0.6B".into());
    let ck = Checkpoint::fetch(&repo)?;
    let tok = ck.tokenizer()?;

    let vb = ck.var_builder(DType::F32, &device)?;
    let base = Qwen3::new(&ck.config, vb.clone())?;
    let mut quant = Qwen3::new(&ck.config, vb)?;
    let n = llvq_llm::artifact::load(&mut quant, &path, &device)?;
    eprintln!("loaded {n} quantized projections from {path}\n");

    for p in PROMPTS {
        let ids = tok
            .encode(*p, false)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .get_ids()
            .to_vec();
        let say = |m: &Qwen3| -> anyhow::Result<String> {
            let out = m.generate(&ids, n_new, &mut NoCapture)?;
            tok.decode(&out, false).map_err(|e| anyhow::anyhow!("{e}"))
        };
        println!("── {p:?}");
        println!("   FP32  →{}", say(&base)?);
        println!("   2-bit →{}\n", say(&quant)?);
    }
    Ok(())
}
