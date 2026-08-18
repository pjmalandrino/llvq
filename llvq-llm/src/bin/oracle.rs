//! Validate our observable Qwen3 forward against `candle-transformers`'.
//!
//! Usage: `cargo run --release -p llvq-llm --bin oracle [-- repo tokens device]`
//!
//! Same checkpoint, same window, same dtype — the hidden states must match.
//! Our version exists only to expose linear-layer inputs; if it diverges,
//! every Hessian built from it is wrong, and the error would surface much
//! later as an unexplained perplexity gap. A hand-written RoPE or a dropped
//! per-head q/k norm shows up here immediately.
//!
//! ## Run it on the backend you are about to quantize on
//!
//! This used to hard-code `Device::Cpu`, so the proof only ever covered CPU.
//! It does not transfer for free: the first remote run reproduced the 0.6B
//! baseline perplexity as 19.4990 against 19.5038 on Metal — 0.025 % apart,
//! benign for a perplexity, and a reminder that backends do not agree bit for
//! bit. Whether the *hidden states* still agree to 1e-4 on CUDA is a question
//! only CUDA can answer, and it costs a few cents to ask.

use candle_core::{DType, Tensor};
use llvq_llm::loader::Checkpoint;
use llvq_llm::model::{NoCapture, Qwen3};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let repo = args.first().cloned().unwrap_or_else(|| "Qwen/Qwen3-0.6B".into());
    let n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(64);

    let device = llvq_llm::eval::device(args.get(2).map(String::as_str).unwrap_or("cpu"))?;
    let dtype = llvq_llm::eval::dtype(DType::F32)?;
    eprintln!(
        "device {device:?}, dtype {}",
        llvq_llm::eval::dtype_name(dtype)
    );

    eprintln!("resolving {repo} (downloads on first use)…");
    let ck = Checkpoint::fetch(&repo)?;
    eprintln!(
        "config: hidden {} inter {} layers {} heads {}/{} head_dim {} tied {}",
        ck.config.hidden_size,
        ck.config.intermediate_size,
        ck.config.num_hidden_layers,
        ck.config.num_attention_heads,
        ck.config.num_key_value_heads,
        ck.config.head_dim,
        ck.config.tie_word_embeddings
    );

    let tok = ck.tokenizer()?;
    let text = "The Leech lattice is the unique even unimodular lattice in 24 \
                dimensions with no vectors of norm two. Its automorphism group \
                is the Conway group, and its kissing number is 196560.";
    let mut ids = tok
        .encode(text, false)
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .get_ids()
        .to_vec();
    ids.truncate(n);
    anyhow::ensure!(ids.len() >= 8, "need a few tokens to compare");
    eprintln!("window: {} tokens", ids.len());

    let input = Tensor::from_slice(&ids, (1, ids.len()), &device)?;

    let vb = ck.var_builder(dtype, &device)?;
    let ours = Qwen3::new(&ck.config, vb.clone(), llvq_llm::kvq::KvMode::F16)?;
    let mine = ours.hidden(&input, &mut NoCapture)?;

    let mut theirs_model =
        candle_transformers::models::qwen3::Model::new(&ck.config, vb.clone())?;
    let theirs = theirs_model.forward(&input, 0)?;

    anyhow::ensure!(mine.dims() == theirs.dims(), "shape mismatch: {:?} vs {:?}", mine.dims(), theirs.dims());
    let diff = (&mine - &theirs)?.abs()?;
    let max: f32 = diff.max_all()?.to_scalar()?;
    let scale: f32 = theirs.abs()?.max_all()?.to_scalar()?;
    let rel = max / scale.max(1e-6);

    println!(
        "{repo} on {device:?} — max |Δhidden| = {max:.3e}   \
         (relative to max|h| = {scale:.3e}) → {rel:.3e}"
    );
    if rel < 1e-4 {
        println!("MATCH — the observable forward reproduces candle's Qwen3");
        Ok(())
    } else {
        anyhow::bail!(
            "DIVERGENCE on {device:?} — do not trust any Hessian built from \
             this forward, and do not pay for a quantization run on it"
        )
    }
}
