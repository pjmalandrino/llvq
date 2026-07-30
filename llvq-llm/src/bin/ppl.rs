//! WikiText-2 perplexity — the number G5 is judged on.
//!
//! Usage: `cargo run --release -p llvq-llm --bin ppl [-- ctx max_windows device]`
//!
//! Non-overlapping windows over the concatenated raw test split, mean
//! next-token NLL, `ppl = exp(mean)`. The paper reports at 4096 context.

use candle_core::{DType, Device};
use llvq_llm::corpus::{c4_validation, wikitext2_test};
use llvq_llm::loader::Checkpoint;
use llvq_llm::model::{NoCapture, Qwen3};

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let ctx: usize = a.first().and_then(|s| s.parse().ok()).unwrap_or(2048);
    let max_windows: usize = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(usize::MAX);
    let want_metal = a.get(2).map(|s| s == "metal").unwrap_or(false);
    // Optional overlay of quantized projections, so a model can be scored
    // properly without paying for the quantization again.
    let overlay = a.get(3).filter(|s| !s.is_empty() && *s != "none").cloned();
    // "c4" scores the same model out of domain; anything else keeps
    // wikitext-2, the corpus our calibration set is drawn from.
    let corpus = a.get(4).cloned().unwrap_or_else(|| "wikitext2".into());

    let device = if want_metal {
        Device::new_metal(0).unwrap_or(Device::Cpu)
    } else {
        Device::Cpu
    };
    eprintln!("device: {device:?}, context {ctx}");

    let repo = std::env::var("LLVQ_MODEL").unwrap_or_else(|_| "Qwen/Qwen3-0.6B".into());
    let ck = Checkpoint::fetch(&repo)?;
    let tok = ck.tokenizer()?;
    let vb = ck.var_builder(DType::F32, &device)?;
    let mut model = Qwen3::new(&ck.config, vb)?;
    if let Some(path) = &overlay {
        let n = llvq_llm::artifact::load(&mut model, path, &device)?;
        eprintln!("overlaid {n} quantized projections from {path}");
    }

    eprintln!("loading corpus {corpus}…");
    let text = if corpus == "c4" {
        c4_validation(8_000_000)?
    } else {
        wikitext2_test()?
    };
    let ids = tok
        .encode(text.as_str(), false)
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .get_ids()
        .to_vec();
    let nwin = (ids.len() / ctx).min(max_windows);
    anyhow::ensure!(nwin > 0, "corpus shorter than one window");
    eprintln!("{} tokens → {nwin} windows of {ctx}", ids.len());

    let t0 = std::time::Instant::now();
    let (mut nll, mut count) = (0.0f64, 0usize);
    for w in 0..nwin {
        let (s, e) = (w * ctx, (w + 1) * ctx);
        let (n, c) = model.window_nll(&ids[s..e], &mut NoCapture)?;
        nll += n;
        count += c;
        eprintln!(
            "  window {:>3}/{nwin}  running ppl {:>8.3}  ({:.1}s)",
            w + 1,
            (nll / count as f64).exp(),
            t0.elapsed().as_secs_f64()
        );
    }
    println!(
        "\n{repo} — {corpus}, ctx {ctx}, {nwin} windows{}\nppl = {:.4}",
        if overlay.is_some() { " [LLVQ 2-bit]" } else { " [FP32]" },
        (nll / count as f64).exp()
    );
    Ok(())
}
