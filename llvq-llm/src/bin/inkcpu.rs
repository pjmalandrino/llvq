//! `inkcpu`: Inkling 2 Mini on the CPU, timed, loaded as the runtime serves it.
//!
//! ```text
//! inkcpu <file-set directory> [n_new]
//! ```
//!
//! The directory is the file set `llvq pull inkling2:mini` unpacks:
//! `config.json`, `model.safetensors`, `tokenizer.json`, `tables/` and
//! `layers/`. The load is `inkling::experts_cpu::load_served`, the function
//! the runtime calls on a CPU build: candle attention and glue in f32, the
//! int4 head and embedding, the experts from their codes, the prompt bounds.
//! Every layer must load its experts, or the run is refused before any token.
//!
//! One fixed user message, rendered by the checkpoint's template at reasoning
//! effort `none`, is read in one prefill, then `n_new` tokens (default 32)
//! are decoded greedily whatever they are. The report: the load time, the
//! prompt time, the decode rate, the generated text, and the decode's wall
//! time split between the parts of the CPU backend (`fused_cpu::phases`) and
//! the model around them, in ms a token, in the format of `cpurun phases`.
//! `moe` is the whole expert call; its matvecs count in `tetra kernel` and
//! its rotations in `rotate kernel`. `rest` is attention, glue, routing,
//! norms and the embedding.
//!
//! The CPU pool takes `LLVQ_THREADS` threads, candle's own `RAYON_NUM_THREADS`;
//! `LLVQ_CPU_WEIGHTS=nibbles` decodes the codes once at load. Peak memory is
//! the caller's to read (`/usr/bin/time -v` on Linux, `-l` on macOS).

use std::path::Path;

use anyhow::{ensure, Context, Result};
use candle_core::{DType, Device, Tensor};
use llvq_llm::inkchat::{InkChat, Json};
use llvq_llm::lm::CausalLm;

const PROMPT: &str = "Explain in a few sentences why the sky is blue during the day and red at sunset.";

fn argmax(logits: &Tensor) -> Result<u32> {
    let v = logits.flatten_all()?.to_dtype(DType::F32)?.to_device(&Device::Cpu)?.to_vec1::<f32>()?;
    let (mut best, mut at) = (f32::NEG_INFINITY, 0usize);
    for (i, &x) in v.iter().enumerate() {
        if x > best {
            best = x;
            at = i;
        }
    }
    ensure!(best.is_finite(), "logits without a finite maximum");
    Ok(at as u32)
}

fn main() -> Result<()> {
    use llvq_llm::fused_cpu::phases::{snapshot, NAMES};
    let a: Vec<String> = std::env::args().skip(1).collect();
    let usage = "usage: inkcpu <file-set directory> [n_new]";
    let dir = Path::new(a.first().ok_or_else(|| anyhow::anyhow!(usage))?);
    let n_new: usize = match a.get(1) {
        Some(s) => s.parse().with_context(|| format!("n_new {s:?}: {usage}"))?,
        None => 32,
    };
    ensure!(n_new >= 2, "n_new {n_new}: two tokens at least, one decode step to split");

    let t = std::time::Instant::now();
    let (model, lines) = llvq_llm::inkling::experts_cpu::load_served(
        &dir.join("config.json"),
        &dir.join("model.safetensors"),
        &dir.join("layers"),
        &dir.join("tables"),
    )?;
    let load_s = t.elapsed().as_secs_f64();
    for l in &lines {
        println!("{l}");
    }
    let layers = model.substituted_layers().len();
    ensure!(
        layers == model.cfg.num_hidden_layers,
        "{layers} of {} layers on the CPU experts: the run would read native experts the file set does not hold",
        model.cfg.num_hidden_layers
    );
    println!("load: {load_s:.1} s");

    let tok_path = dir.join("tokenizer.json");
    let tok = tokenizers::Tokenizer::from_file(&tok_path).map_err(|e| anyhow::anyhow!("{}: {e}", tok_path.display()))?;
    let chat = InkChat::new(tok.clone())?;
    let msg = Json::parse(&serde_json::json!({"role": "user", "content": PROMPT}).to_string())?;
    let ids = chat.render(&[msg], &[], Some(&Json::str("none")))?.ids;

    let mut cache = CausalLm::new_cache(&model);
    let t = std::time::Instant::now();
    let mut next = argmax(&model.prefill(&ids, &mut cache)?)?;
    let prompt_s = t.elapsed().as_secs_f64();
    println!("prompt: {} tokens in {prompt_s:.2} s, {:.1} tok/s", ids.len(), ids.len() as f64 / prompt_s);

    let mut out = vec![next];
    let before = snapshot();
    let t = std::time::Instant::now();
    for _ in 1..n_new {
        let pos = cache.len();
        next = argmax(&model.step(next, pos, &mut cache)?)?;
        out.push(next);
    }
    let wall = t.elapsed().as_secs_f64();
    let after = snapshot();
    let steps = n_new - 1;
    let text = tok.decode(&out, false).map_err(|e| anyhow::anyhow!("{e}"))?;
    println!("generated: {out:?}");
    println!("text: {text:?}");

    let per = |ns: u64| ns as f64 / 1e6 / steps as f64;
    let token_ms = wall * 1e3 / steps as f64;
    println!("decode: {steps} steps in {wall:.2} s, {:.2} tok/s", steps as f64 / wall);
    println!("phases: {steps} decode steps, {token_ms:.2} ms a token, {:.2} tok/s", 1e3 / token_ms);
    let spent: Vec<(u64, u64)> = after.iter().zip(before).map(|(a, b)| (a.0 - b.0, a.1 - b.1)).collect();
    for (name, &(ns, calls)) in NAMES.iter().zip(&spent) {
        println!("phases: {name:<13} {:>8.2} ms a token  {:>7.1} calls a token", per(ns), calls as f64 / steps as f64);
    }
    // A name ending in `kernel` is inside another entry: not counted twice.
    let backend: u64 = NAMES.iter().zip(&spent).filter(|(n, _)| !n.ends_with("kernel")).map(|(_, (ns, _))| ns).sum();
    println!("phases: {:<13} {:>8.2} ms a token  (attention, glue, routing, norms, embedding)", "rest", token_ms - per(backend));
    Ok(())
}
