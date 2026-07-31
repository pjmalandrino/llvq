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
//! it, rather than silently reaching for the checkpoint.

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen3::Config;
use llvq_llm::model::{NoCapture, Qwen3};
use std::collections::HashMap;

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
    // difference is 7 GB of RAM against 15 on a 4B.
    let dtype = DType::F16;

    let bytes = std::fs::metadata(&path)?.len();
    let f = std::fs::File::open(&path)?;
    let mut r = std::io::BufReader::with_capacity(1 << 20, f);
    let head = llvq_artifact::read_header(&mut r)?;
    anyhow::ensure!(
        head.is_self_contained(),
        "{path} is a projections-only artifact (format v{}). It cannot run on \
         its own — seal it first:\n  LLVQ_MODEL=<repo> cargo run --release -p \
         llvq-llm --bin seal -- {path} sealed.llvq",
        head.version
    );

    // ---- rebuild every tensor, one at a time ----
    let ix = llvq_search::index::Indexer::new();
    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    let mut quantized_weights = 0usize;
    for _ in 0..head.matrices {
        let m = llvq_artifact::read_matrix(&mut r, &ix)?;
        quantized_weights += m.d_out * m.d_in;
        let w = llvq_artifact::decode_matrix(&m);
        let t = Tensor::from_vec(w, (m.d_out, m.d_in), &device)?.to_dtype(dtype)?;
        tensors.insert(m.name, t);
    }

    let n_raw = read_u32(&mut r)?;
    let mut carried = 0usize;
    for _ in 0..n_raw {
        let t = llvq_artifact::read_raw(&mut r)?;
        carried += t.data.len();
        let vals: Vec<half::f16> = t.data.iter().map(|b| half::f16::from_bits(*b)).collect();
        let tensor = Tensor::from_vec(vals, t.dims.clone(), &device)?.to_dtype(dtype)?;
        tensors.insert(t.name, tensor);
    }

    let n_blob = read_u32(&mut r)?;
    let mut blobs: HashMap<String, Vec<u8>> = HashMap::new();
    for _ in 0..n_blob {
        let b = llvq_artifact::read_blob(&mut r)?;
        blobs.insert(b.name, b.bytes);
    }

    let cfg_bytes = blobs
        .get("config.json")
        .ok_or_else(|| anyhow::anyhow!("sealed file carries no config.json"))?;
    let config: Config = serde_json::from_slice(cfg_bytes)?;
    let tok_bytes = blobs
        .get("tokenizer.json")
        .ok_or_else(|| anyhow::anyhow!("sealed file carries no tokenizer.json"))?;
    let tok = tokenizers::Tokenizer::from_bytes(tok_bytes).map_err(|e| anyhow::anyhow!("{e}"))?;

    let vb = VarBuilder::from_tensors(tensors, dtype, &device);
    let model = Qwen3::new(&config, vb)?;

    let fp16 = (quantized_weights + carried) * 2;
    eprintln!(
        "loaded {path} — {} quantized matrices + {n_raw} carried tensors\n",
        head.matrices
    );
    println!("── model");
    println!(
        "   {:.3} GB on disk against {:.3} GB in FP16  →  ×{:.2}",
        bytes as f64 / 1e9,
        fp16 as f64 / 1e9,
        fp16 as f64 / bytes as f64
    );
    println!(
        "   {quantized_weights} weights quantized, {carried} carried at f16\n"
    );

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

fn read_u32(r: &mut impl std::io::Read) -> anyhow::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}
