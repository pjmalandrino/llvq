//! The sealed file on the CPU, timed: the Tetra backend against the dense
//! reconstruction, on the same prompts and the same tokens.
//!
//! ```text
//! LLVQ_CONFIG=configs/qwen3-4b-tetra-e4.json cpurun <sealed> tetra [n_new]
//! cpurun <sealed> dense [n_new]
//! LLVQ_CONFIG=configs/qwen3-4b-tetra-e4.json cpurun <sealed> gate [windows] [len]
//! LLVQ_CONFIG=configs/qwen3-4b-tetra-e4.json cpurun <sealed> ppl [windows] [len] [first]
//! ```
//!
//! `tetra` loads through `served::load_resolved` on `Device::Cpu`, which is
//! `fused_cpu`: the codes stay encoded. `dense` is `sealed::load` on the CPU,
//! what the runtime served before. `tetra`, `ppl` and `gate` run at f32, the
//! runtime's CPU dtype; `dense` at f16, as the runtime ran it; `LLVQ_DTYPE`
//! overrides both. Each prints the prompt time, the decode rate, and
//! the generated ids; `LLVQ_RUN_DUMP=<path>` writes them as JSON so two runs
//! compare token for token.
//!
//! `gate` loads both and scores WikiText-2 windows through each, paired
//! window by window: the mean NLL difference and its spread, the measure the
//! T8 band is stated in.
//!
//! `ppl` scores the Tetra backend alone on WikiText-2 windows and prints each
//! window's mean NLL in `bin/ppl`'s format, so it pairs with a journal of the
//! dense reconstruction scored on the same windows elsewhere.
//!
//! The CPU pool takes `LLVQ_THREADS` threads; candle's own takes
//! `RAYON_NUM_THREADS`. Peak memory is the caller's to read
//! (`/usr/bin/time -l` on macOS, `-v` on Linux).

use candle_core::{DType, Device};
use llvq_llm::model::{NoCapture, Qwen3};

const PROMPTS: &[&str] = &[
    "The capital of France is",
    "In 1969, the first humans landed on",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return",
    "Water boils at a temperature of",
];

fn load_tetra(path: &str, dtype: DType) -> anyhow::Result<(Qwen3, tokenizers::Tokenizer)> {
    let cfg = llvq_llm::served::Served::from_env()
        .map_err(anyhow::Error::msg)?
        .ok_or_else(|| anyhow::anyhow!("tetra needs LLVQ_CONFIG, e.g. configs/qwen3-4b-tetra-e4.json"))?;
    let device = Device::Cpu;
    let mut f = llvq_llm::served::load_resolved(
        path,
        &device,
        dtype,
        cfg.layout,
        cfg.embed,
        cfg.rot_share,
        cfg.fuse,
        cfg.kv,
        cfg.int4_layout(),
        Some("LLVQ_CONFIG"),
    )?;
    cfg.apply(&mut f.model, &device).map_err(anyhow::Error::msg)?;
    println!("tetra on the CPU: {} ({})", cfg.path.display(), llvq_llm::tetra_cpu::Isa::detect().name());
    Ok((f.model, f.tokenizer))
}

fn load_dense(path: &str, dtype: DType) -> anyhow::Result<(Qwen3, tokenizers::Tokenizer)> {
    let s = llvq_llm::sealed::load(path, dtype, &Device::Cpu, llvq_llm::kvq::KvMode::F16)?;
    println!("dense reconstruction on the CPU (sealed::load)");
    Ok((s.model, s.tokenizer))
}

fn generate(model: &Qwen3, tok: &tokenizers::Tokenizer, n_new: usize) -> anyhow::Result<Vec<serde_json::Value>> {
    let mut out = Vec::new();
    let (mut decoded, mut decode_s) = (0usize, 0f64);
    for p in PROMPTS {
        let ids = tok.encode(*p, false).map_err(|e| anyhow::anyhow!("{e}"))?.get_ids().to_vec();
        let g = model.generate_timed(&ids, n_new, ids.len())?;
        let text = tok.decode(&g.tokens, false).map_err(|e| anyhow::anyhow!("{e}"))?;
        let steps = g.tokens.len().saturating_sub(1);
        println!(
            "── {p:?}\n   →{text}\n   prompt {} tokens in {:.2} s | {steps} decode steps in {:.2} s, {:.2} tok/s",
            ids.len(),
            g.prompt_s,
            g.decode_s,
            steps as f64 / g.decode_s
        );
        decoded += steps;
        decode_s += g.decode_s;
        out.push(serde_json::json!({ "prompt": p, "input_ids": ids, "output_ids": g.tokens, "text": text }));
    }
    println!("decode over the {} prompts: {decoded} steps in {decode_s:.2} s, {:.2} tok/s", PROMPTS.len(), decoded as f64 / decode_s);
    Ok(out)
}

fn gate(path: &str, dtype: DType, windows: usize, len: usize) -> anyhow::Result<()> {
    let (tetra, tok) = load_tetra(path, dtype)?;
    let (dense, _) = load_dense(path, dtype)?;
    let text = llvq_llm::corpus::wikitext2_test()?;
    let ids = tok.encode(text.as_str(), false).map_err(|e| anyhow::anyhow!("{e}"))?.get_ids().to_vec();
    let n = (ids.len() / len).min(windows);
    anyhow::ensure!(n > 0, "corpus shorter than one window");
    println!(
        "{n} windows of {len} tokens from WikiText-2 test (fingerprint {:016x})",
        llvq_llm::eval::token_fingerprint(&ids[..n * len])
    );
    let (mut deltas, mut st, mut sd, mut count) = (Vec::new(), 0f64, 0f64, 0usize);
    for w in 0..n {
        let win = &ids[w * len..(w + 1) * len];
        let t0 = std::time::Instant::now();
        let (a, c) = tetra.window_nll(win, &mut NoCapture)?;
        let ta = t0.elapsed().as_secs_f64();
        let t1 = std::time::Instant::now();
        let (b, _) = dense.window_nll(win, &mut NoCapture)?;
        let tb = t1.elapsed().as_secs_f64();
        let (ma, mb) = (a / c as f64, b / c as f64);
        // Relative change of the window's perplexity, tetra against dense.
        let d = ((ma - mb).exp() - 1.0) * 100.0;
        println!("window {w:>2}: ppl tetra {:.4} dense {:.4} ({d:+.4} %) | {ta:.1} s against {tb:.1} s", ma.exp(), mb.exp());
        deltas.push(d);
        st += a;
        sd += b;
        count += c;
    }
    let mean = deltas.iter().sum::<f64>() / n as f64;
    let sdev = (deltas.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / (n.max(2) - 1) as f64).sqrt();
    // Student's t at n - 1 degrees of freedom: 1.96 made the interval 38 %
    // too narrow at 4 windows (t(3) = 3.182; audit 2026-11-08, section 4).
    let t = llvq_llm::eval::t95(n.saturating_sub(1));
    let half = t * sdev / (n as f64).sqrt();
    println!(
        "pooled ppl tetra {:.4} dense {:.4} ({:+.4} %); per-window mean {mean:+.4} % [{:+.4} ; {:+.4}] (±t({}) s/√n, t = {t:.3})",
        (st / count as f64).exp(),
        (sd / count as f64).exp(),
        (((st - sd) / count as f64).exp() - 1.0) * 100.0,
        mean - half,
        mean + half,
        n.saturating_sub(1)
    );
    Ok(())
}

fn ppl(path: &str, dtype: DType, windows: usize, len: usize, first: usize) -> anyhow::Result<()> {
    let (tetra, tok) = load_tetra(path, dtype)?;
    let text = llvq_llm::corpus::wikitext2_test()?;
    let ids = tok.encode(text.as_str(), false).map_err(|e| anyhow::anyhow!("{e}"))?.get_ids().to_vec();
    let n = (ids.len() / len).min(windows);
    anyhow::ensure!(n > 0, "corpus shorter than one window");
    println!(
        "{} tokens → {n} windows of {len} (fingerprint {:016x})",
        ids.len(),
        llvq_llm::eval::token_fingerprint(&ids[..n * len])
    );
    let t0 = std::time::Instant::now();
    let (mut nll, mut count) = (0f64, 0usize);
    for w in first.min(n)..n {
        let (a, c) = tetra.window_nll(&ids[w * len..(w + 1) * len], &mut NoCapture)?;
        nll += a;
        count += c;
        println!(
            "  window {:>3}/{n}  nll {:.9}  n {c}  running ppl {:>9.3}  ({:.1}s)",
            w + 1,
            a / c as f64,
            (nll / count as f64).exp(),
            t0.elapsed().as_secs_f64()
        );
    }
    println!("ppl = {:.4}", (nll / count as f64).exp());
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let usage = "usage: cpurun <sealed> <tetra|dense|gate|ppl> [n_new | windows len]";
    let path = a.first().ok_or_else(|| anyhow::anyhow!(usage))?;
    let arm = a.get(1).map(String::as_str).ok_or_else(|| anyhow::anyhow!(usage))?;
    let dtype = llvq_llm::eval::dtype(match arm {
        "dense" => DType::F16,
        _ => DType::F32,
    })?;
    if arm == "ppl" {
        let windows = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(12);
        let len = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(4096);
        // Windows before this 0-based index are skipped, not scored.
        let first = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(0);
        return ppl(path, dtype, windows, len, first);
    }
    if arm == "gate" {
        let windows = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(4);
        let len = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(512);
        return gate(path, dtype, windows, len);
    }
    let n_new: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(32);
    let t = std::time::Instant::now();
    let (model, tok) = match arm {
        "tetra" => load_tetra(path, dtype)?,
        "dense" => load_dense(path, dtype)?,
        other => anyhow::bail!("{other:?}: {usage}"),
    };
    println!("loaded in {:.1} s at {}", t.elapsed().as_secs_f64(), llvq_llm::eval::dtype_name(dtype));
    let dumped = generate(&model, &tok, n_new)?;
    if let Ok(p) = std::env::var("LLVQ_RUN_DUMP") {
        let out = serde_json::json!({ "model": path, "arm": arm, "dtype": llvq_llm::eval::dtype_name(dtype), "n_new": n_new, "prompts": dumped });
        std::fs::write(&p, serde_json::to_vec_pretty(&out)?)?;
        println!("dumped {} prompts to {p}", dumped.len());
    }
    Ok(())
}
