//! Generation throughput, for any arm of the campaign.
//!
//! The campaign promised a speed line on three arms and nothing could produce
//! it: `bin/run` is the only binary in this repository that times a
//! generation, it is not in the CUDA image, and it refuses a checkpoint
//! outright. So the f16 and AWQ arms had no throughput producer at all. This
//! is that producer, and it takes the same model argument as `bin/ppl` so the
//! four arms are loaded by the same code.
//!
//! ```text
//! gbench [device] [n_new] [model.llvq | overlay.safetensors | none]
//! ```
//!
//! With no model argument it scores `$LLVQ_MODEL` (default `Qwen/Qwen3-0.6B`)
//! — the f16 arm, and the AWQ arm once its reconstruction is overlaid.
//!
//! ## Phase markers, and why they are epoch milliseconds
//!
//! Each phase boundary prints `[phase] <epoch_ms> <name>` on stderr. The job
//! monitor samples VRAM and utilisation on its own clock; epoch milliseconds
//! let the two series be aligned afterwards without either side knowing about
//! the other. No timezone, no format to agree on, no dependency — and it
//! answers the question that matters when reading a metric spike: *which
//! phase was running*. Download, load and decode have completely different
//! memory profiles, and averaging across them is a mistake this repository
//! has already published once.

use candle_core::DType;
use llvq_llm::loader::Checkpoint;
use llvq_llm::model::{NoCapture, Qwen3};

/// Prompts of deliberately different lengths.
///
/// With a KV cache the prefill is `O(prompt)` and each decode step is `O(1)`
/// in work but `O(context)` in cache traffic, so one prompt length would
/// measure one point of a curve and be quoted as the curve.
const PROMPTS: &[&str] = &[
    "The capital of France is",
    "In 1969, the first humans landed on the moon, and the mission that carried \
     them there was the culmination of a decade of work by hundreds of thousands \
     of engineers. The programme began when",
];

fn epoch_ms() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

fn phase(name: &str) {
    eprintln!("[phase] {} {name}", epoch_ms());
}

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let device_arg = a.first().map(String::as_str).unwrap_or("cpu");
    let n_new: usize = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(64);
    let quantized = a.get(2).filter(|s| !s.is_empty() && *s != "none").cloned();

    // `generate` needs at least one decode step past the prefill, or the KV
    // cache is never exercised and the number is a prefill in disguise.
    anyhow::ensure!(n_new >= 2, "n_new must be at least 2, got {n_new}");

    let device = llvq_llm::eval::device(device_arg)?;
    let dtype = llvq_llm::eval::dtype(DType::F16)?;
    let repo = std::env::var("LLVQ_MODEL").unwrap_or_else(|_| "Qwen/Qwen3-0.6B".into());
    eprintln!(
        "device {device:?}, dtype {}, {n_new} nouveaux tokens",
        llvq_llm::eval::dtype_name(dtype)
    );

    phase("load-start");
    let t0 = std::time::Instant::now();
    // The same match as `bin/ppl`, on purpose: four arms loaded by one code
    // path is the only way their numbers describe the same objects.
    let (model, tok, label) = match quantized.as_deref() {
        Some(p) if llvq_llm::sealed::is_sealed_path(p) => {
            let s = llvq_llm::sealed::load(p, dtype, &device)?;
            eprintln!(
                "sealed {p}: {} matrices quantifiées, {:.3} Go sur disque",
                s.matrices,
                s.bytes as f64 / 1e9
            );
            (s.model, s.tokenizer, format!("{p} [LLVQ 2 bits, scellé]"))
        }
        other => {
            let ck = Checkpoint::fetch(&repo)?;
            let tok = ck.tokenizer()?;
            let vb = ck.var_builder(dtype, &device)?;
            let mut model = Qwen3::new(&ck.config, vb)?;
            let label = match other {
                Some(p) => {
                    let n = llvq_llm::artifact::load(&mut model, p, &device)?;
                    eprintln!("{n} projections recouvertes depuis {p}");
                    format!("{repo} [recouvert {p}]")
                }
                None => format!("{repo} [f16]"),
            };
            (model, tok, label)
        }
    };
    let load_s = t0.elapsed().as_secs_f64();
    phase("load-end");
    println!("modèle : {label}\n  chargé en {load_s:.1} s");

    println!("\n  {:<10}{:>8}{:>12}{:>12}", "prompt", "tokens", "s", "tok/s");
    println!("  {}", "-".repeat(44));
    let mut best = 0.0f64;
    for p in PROMPTS {
        let ids = tok
            .encode(*p, false)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .get_ids()
            .to_vec();
        phase(&format!("gen-start-{}", ids.len()));
        let t = std::time::Instant::now();
        let out = model.generate(&ids, n_new, &mut NoCapture)?;
        let s = t.elapsed().as_secs_f64();
        phase(&format!("gen-end-{}", ids.len()));
        anyhow::ensure!(out.len() == n_new, "generate rendit {} tokens", out.len());
        let rate = n_new as f64 / s;
        best = best.max(rate);
        println!("  {:<10}{:>8}{:>12.2}{:>12.1}", ids.len(), n_new, s, rate);
    }
    println!("  {}", "-".repeat(44));
    println!("  meilleur : {best:.1} tok/s");
    println!(
        "\n  ⚠️ génération gloutonne, batch 1, avec cache KV. Le prompt est court et le\n  \
         contexte le reste : le débit d'un contexte long est plus bas, parce que le coût\n  \
         du cache croît avec lui. Ce chiffre n'est comparable qu'à un autre pris ici.\n  \
         Les marqueurs `[phase] <epoch_ms>` sur stderr servent à recaler la série de\n  \
         métriques du job : un pic de VRAM pendant `load` ne dit pas la même chose\n  \
         qu'un pic pendant `gen`."
    );
    Ok(())
}
