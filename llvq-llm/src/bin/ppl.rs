//! WikiText-2 perplexity — the number G5 is judged on.
//!
//! Usage: `cargo run --release -p llvq-llm --bin ppl [-- ctx max_windows device model corpus]`
//!
//! Non-overlapping windows over the concatenated raw test split, mean
//! next-token NLL, `ppl = exp(mean)`. The paper reports at 4096 context.
//!
//! ## What the fourth argument accepts, and why it matters
//!
//! * nothing / `none` — the bare checkpoint named by `LLVQ_MODEL`: the
//!   baseline;
//! * a `.safetensors` — a dump of **reconstructions** overlaid on that
//!   checkpoint. Cheap, and it is what every published quantized perplexity
//!   used;
//! * a `.llvq` / `.bin` — the **sealed artifact itself**, decoded exactly as
//!   `bin/run` and `bin/mmlu` decode it.
//!
//! The third form is new and it is the one to prefer, because the second has
//! already produced a trap. Two Qwen3-4B runs on 2026-07-31 print the same
//! configuration line and the same rate, and report `ppl = 15.3272` and
//! `ppl = 16.9617`; the overlay left on disk belongs to the first run and the
//! sealed file to the second. Quoting a perplexity from one next to an MMLU
//! score from the other compares two different models, and nothing in either
//! output says so.
//!
//! The model dtype defaults to F32 here and to F16 in `bin/mmlu` — fine within
//! each metric, and not fine across them (see [`llvq_llm::eval`]).
//! `LLVQ_DTYPE=f16` scores perplexity on the object MMLU scores. The resolved
//! dtype and a fingerprint of the scored token stream are printed with the
//! result, so "these two numbers describe the same thing" is checkable rather
//! than assumed.

use candle_core::DType;
use llvq_llm::corpus::{c4_validation, wikitext2_test};
use llvq_llm::loader::Checkpoint;
use llvq_llm::model::{NoCapture, Qwen3};

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let ctx: usize = a.first().and_then(|s| s.parse().ok()).unwrap_or(2048);
    let max_windows: usize = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(usize::MAX);
    let device_arg = a.get(2).map(String::as_str).unwrap_or("cpu");
    // A quantized model to score: a safetensors overlay of reconstructions, or
    // the sealed artifact itself. See the module note.
    let quantized = a.get(3).filter(|s| !s.is_empty() && *s != "none").cloned();
    // "c4" scores the same model out of domain; anything else keeps
    // wikitext-2, the corpus our calibration set is drawn from.
    let corpus = a.get(4).cloned().unwrap_or_else(|| "wikitext2".into());

    let device = llvq_llm::eval::device(device_arg)?;
    let dtype = llvq_llm::eval::dtype(DType::F32)?;
    eprintln!(
        "device: {device:?}, context {ctx}, dtype {}",
        llvq_llm::eval::dtype_name(dtype)
    );

    let repo = std::env::var("LLVQ_MODEL").unwrap_or_else(|_| "Qwen/Qwen3-0.6B".into());
    let (model, tok, label) = match quantized.as_deref() {
        // The deliverable, rebuilt from one file — no checkpoint, no network.
        // Its tokenizer comes out of the file too, which is why the token
        // fingerprint below is worth printing.
        Some(p) if llvq_llm::sealed::is_sealed_path(p) => {
            let s = llvq_llm::sealed::load(p, dtype, &device)?;
            eprintln!(
                "sealed {p}: {} quantized matrices, {:.3} GB on disk",
                s.matrices,
                s.bytes as f64 / 1e9
            );
            (s.model, s.tokenizer, format!("{p} [LLVQ 2-bit, sealed]"))
        }
        other => {
            let ck = Checkpoint::fetch(&repo)?;
            let tok = ck.tokenizer()?;
            let vb = ck.var_builder(dtype, &device)?;
            let mut model = Qwen3::new(&ck.config, vb)?;
            let label = match other {
                Some(p) => {
                    let n = llvq_llm::artifact::load(&mut model, p, &device)?;
                    eprintln!("overlaid {n} quantized projections from {p}");
                    format!("{repo} [LLVQ 2-bit, overlay {p}]")
                }
                None => format!("{repo} [baseline]"),
            };
            (model, tok, label)
        }
    };

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
    // Over the tokens actually scored, not the whole corpus: two runs with
    // different window counts are not comparable either.
    let fingerprint = llvq_llm::eval::token_fingerprint(&ids[..nwin * ctx]);
    eprintln!(
        "{} tokens → {nwin} windows of {ctx} (fingerprint {fingerprint:016x})",
        ids.len()
    );

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
    // Everything needed to know whether this number may be compared to another
    // one goes on the result line: what was scored, at what precision, over
    // which tokens.
    println!(
        "\n{label} — {corpus}, ctx {ctx}, {nwin} windows, dtype {}, tokens {fingerprint:016x}\nppl = {:.4}",
        llvq_llm::eval::dtype_name(dtype),
        (nll / count as f64).exp()
    );
    Ok(())
}
