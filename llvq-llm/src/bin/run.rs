//! Load a compressed `.llvq` artifact and generate from it.
//!
//! Usage:
//!   `cargo run --release -p llvq-llm --features metal --bin run -- <model.llvq> [device] [n_new]`
//!
//! This is the deliverable's own test: not a perplexity, not a computed rate —
//! a file on disk that turns back into a model and answers. Every other number
//! in this project is a claim about what a file *would* weigh; this one reads
//! the file.
//!
//! It prints the compression bilan honestly, which means counting the
//! embedding the artifact does **not** hold. On a 4B model that embedding is
//! 9.7 % of the weights and drags the end-to-end ratio from ~7.6× down to
//! ~4.6×; quoting the linear layers' rate as the model's compression is the
//! single easiest way to be wrong by a factor of two.

use candle_core::{DType, Device};
use llvq_llm::loader::Checkpoint;
use llvq_llm::model::{NoCapture, Qwen3};

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
        .ok_or_else(|| anyhow::anyhow!("give the path to a .llvq artifact"))?;
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
    let mut model = Qwen3::new(&ck.config, vb)?;

    let bytes = std::fs::metadata(&path)?.len();
    let (n, packed_weights) = llvq_llm::artifact2::load(&mut model, &path, &device)?;
    eprintln!("loaded {n} projections from {path} ({:.3} GB)\n", bytes as f64 / 1e9);

    // The bilan, counting what the artifact does not carry.
    let cfg = &ck.config;
    let emb = cfg.vocab_size * cfg.hidden_size;
    let linear: usize = (0..model.blocks.len())
        .flat_map(|b| {
            llvq_llm::model::Act::ALL
                .iter()
                .flat_map(move |act| act.consumers().iter().map(move |name| (b, *name)))
        })
        .map(|(b, name)| {
            let d = model.blocks[b].linear(name).weight().dims().to_vec();
            d[0] * d[1]
        })
        .sum();

    println!("── compression");
    println!(
        "   packed           {:>12} weights in {} matrices → {:.3} GB ({:.4} bits/weight)",
        packed_weights,
        n,
        bytes as f64 / 1e9,
        bytes as f64 * 8.0 / packed_weights as f64
    );
    if packed_weights < linear {
        // A partial artifact would otherwise quote a ratio for a model it does
        // not contain — the single most misleading number this project can
        // print, and one it has already got wrong twice.
        println!(
            "   ⚠️  PARTIAL ARTIFACT: {packed_weights} of {linear} linear weights \
             ({:.1} %). The remaining projections are still FP32 in memory and \
             are NOT in the file, so no end-to-end ratio is quoted.\n",
            100.0 * packed_weights as f64 / linear as f64
        );
    } else {
        let fp16 = (linear + emb) * 2;
        let ours = bytes as usize + emb * 2;
        println!(
            "   embedding        {:>12} weights at f16 → {:.3} GB ({:.1} % of the model)",
            emb,
            emb as f64 * 2.0 / 1e9,
            100.0 * emb as f64 / (linear + emb) as f64
        );
        println!(
            "   artifact + embedding {:.3} GB against {:.3} GB in FP16 → ×{:.2}\n",
            ours as f64 / 1e9,
            fp16 as f64 / 1e9,
            fp16 as f64 / ours as f64
        );
    }

    for p in PROMPTS {
        let ids = tok
            .encode(*p, false)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .get_ids()
            .to_vec();
        let out = model.generate(&ids, n_new, &mut NoCapture)?;
        let text = tok.decode(&out, false).map_err(|e| anyhow::anyhow!("{e}"))?;
        println!("── {p:?}\n   →{text}");
    }
    Ok(())
}
