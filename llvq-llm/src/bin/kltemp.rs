//! How much of a quantized model's gap to its dense teacher is one number.
//!
//! ```text
//! LLVQ_MODEL=Qwen/Qwen3-4B LLVQ_KL_ARTIFACT=~/llvq-4b-tetra.llvq \
//!   cargo run --release -p llvq-llm --features metal --bin kltemp -- 4 4 512 metal
//! ```
//!
//! ## Why it runs before anything is fitted
//!
//! A distillation pilot would fit a few hundred gain parameters against
//! `KL(dense ‖ quantized)`. Before spending that, one number decides how to
//! read whatever it produces: a positive rescaling of logits cannot move an
//! argmax, so every nat of divergence a lone temperature removes is a nat that
//! was never going to change an answer. If one scalar closes most of the gap,
//! a fit against the same objective spends most of its freedom where accuracy
//! cannot follow.
//!
//! So this binary measures the fraction, and nothing else. It does not decide
//! whether the remainder is worth fitting, and it does not explain the
//! accuracy loss of `docs/mesures/errmap-mmlu-4b-2026-09-16.txt` — a rescaling
//! leaving accuracy unchanged is true by construction, not evidence.
//!
//! ## Cost
//!
//! Four forward passes, and the temperature itself costs none of them:
//! `Σ p log p` and `⟨p, z⟩` do not depend on the temperature, so the logits
//! are computed once and the search reads them ([`llvq_llm::tempfit`]). The
//! teacher and the student are the same model object — the checkpoint runs
//! first, then the artifact is loaded over its projections — so one model is
//! resident at a time.
//!
//! ## Splits
//!
//! The temperature is fitted on one split and scored on another it never saw.
//! Both start past `LLVQ_KL_SKIP` tokens of the training corpus, so neither
//! overlaps the windows GPTQ calibrated on. `LLVQ_KL_VAL=c4` moves the second
//! split to another domain, which is the harder question and the one worth
//! asking if the first passes.
//!
//! Environment: `LLVQ_MODEL` (checkpoint), `LLVQ_KL_ARTIFACT` (the quantized
//! arm, required), `LLVQ_KL_VAL` (`wikitext2` default, or `c4`),
//! `LLVQ_KL_SKIP` (tokens skipped at the head of the training corpus, default
//! 131072 = the 64 × 2048 windows `smoke` calibrates on).

use anyhow::Context;
use candle_core::{DType, IndexOp, Tensor, D};
use llvq_llm::corpus::hf_parquet_text;
use llvq_llm::loader::Checkpoint;
use llvq_llm::model::{NoCapture, Qwen3};
use llvq_llm::tempfit::{parabola_vertex, Fit, Totals};

/// The `β` interval the search brackets, and the width it stops at.
///
/// Wide enough that a hit on either end means something other than a
/// temperature is wrong; the result carries `at_bound` so that case is read as
/// a failure rather than as a fit.
const BETA_LO: f64 = 0.25;
const BETA_HI: f64 = 4.0;
const GRID: usize = 41;
const TOL: f64 = 1e-5;

fn env_usize(key: &str, default: usize) -> anyhow::Result<usize> {
    match std::env::var(key) {
        Ok(v) => v.parse().with_context(|| format!("{key}={v:?} is not an integer")),
        Err(_) => Ok(default),
    }
}

/// One window's logits, `(positions, vocabulary)`, in f32.
///
/// Position `i` predicts token `i+1`, so a window of `l` tokens contributes
/// `l − 1` rows — the same accounting as [`Qwen3::window_nll`], so the numbers
/// here sit beside the perplexity journals rather than beside a different
/// convention.
fn window_logits(model: &Qwen3, ids: &[u32], device: &candle_core::Device) -> anyhow::Result<Tensor> {
    let l = ids.len();
    anyhow::ensure!(l >= 2, "a window must hold at least two tokens");
    let input = Tensor::from_slice(ids, (1, l), device)?;
    let logits = model.logits(&input, &mut NoCapture)?.to_dtype(DType::F32)?;
    Ok(logits.i(0)?.narrow(0, 0, l - 1)?.contiguous()?)
}

/// Sum a per-position reduction on the host, in f64.
///
/// Reducing to one scalar on the device would accumulate 78 million f32 terms
/// per window in whatever order the backend picks. Reducing over the vocabulary
/// on the device and summing the few hundred survivors here keeps the wide
/// reduction on the GPU and the long one in f64.
fn sum_rows(t: &Tensor) -> anyhow::Result<f64> {
    Ok(t.to_vec1::<f32>()?.into_iter().map(f64::from).sum())
}

/// `Σᵢ logsumexp(β zᵢ)` over every stored window.
fn log_partition(windows: &[Tensor], beta: f64) -> anyhow::Result<f64> {
    let mut acc = 0.0;
    for z in windows {
        acc += sum_rows(&(z * beta)?.log_sum_exp((D::Minus1,))?)?;
    }
    Ok(acc)
}

/// The inverse temperature that minimizes the divergence on one split.
///
/// A grid first, because it shows the shape and catches a minimum pinned to an
/// end; then golden section inside the bracketing triple, which the convexity
/// of the divergence in `β` licenses. The parabola through the final triple is
/// reported beside the search's own answer: the two disagreeing by more than
/// the tolerance would mean the surface is not what the module claims.
fn fit(totals: &Totals, windows: &[Tensor]) -> anyhow::Result<(Fit, Option<f64>)> {
    let at = |beta: f64| -> anyhow::Result<f64> {
        Ok(totals.divergence(beta, log_partition(windows, beta)?))
    };
    let step = (BETA_HI / BETA_LO).powf(1.0 / (GRID - 1) as f64);
    let mut grid = Vec::with_capacity(GRID);
    for i in 0..GRID {
        let b = BETA_LO * step.powi(i as i32);
        grid.push((b, at(b)?));
    }
    let best = (0..GRID).min_by(|&i, &j| grid[i].1.total_cmp(&grid[j].1)).unwrap();
    let at_bound = best == 0 || best == GRID - 1;
    let baseline = at(1.0)?;
    if at_bound {
        return Ok((
            Fit { beta: grid[best].0, divergence: grid[best].1, baseline, at_bound },
            None,
        ));
    }

    let (mut lo, mut hi) = (grid[best - 1].0, grid[best + 1].0);
    let phi = 0.5 * (5.0_f64.sqrt() - 1.0);
    let (mut c, mut d) = (hi - phi * (hi - lo), lo + phi * (hi - lo));
    let (mut fc, mut fd) = (at(c)?, at(d)?);
    while hi - lo > TOL {
        if fc < fd {
            hi = d;
            d = c;
            fd = fc;
            c = hi - phi * (hi - lo);
            fc = at(c)?;
        } else {
            lo = c;
            c = d;
            fc = fd;
            d = lo + phi * (hi - lo);
            fd = at(d)?;
        }
    }
    let beta = 0.5 * (lo + hi);
    let divergence = at(beta)?;
    let vertex = parabola_vertex(grid[best - 1], grid[best], grid[best + 1]);
    Ok((Fit { beta, divergence, baseline, at_bound: false }, vertex))
}

/// How often the quantized model's top token is the dense model's top token.
///
/// Reported once and checked at both temperatures. It cannot change — a
/// positive scaling preserves an argmax — so a difference here is the harness
/// being wrong, not a result.
fn agreement(teacher: &[Tensor], student: &[Tensor], beta: f64) -> anyhow::Result<(usize, usize)> {
    let (mut same, mut total) = (0usize, 0usize);
    for (p, z) in teacher.iter().zip(student) {
        let a = p.argmax(D::Minus1)?.to_vec1::<u32>()?;
        let b = (z * beta)?.argmax(D::Minus1)?.to_vec1::<u32>()?;
        same += a.iter().zip(&b).filter(|(x, y)| x == y).count();
        total += a.len();
    }
    Ok((same, total))
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    anyhow::ensure!(args.len() == 4, "usage: kltemp <n_fit> <n_val> <ctx> <device>");
    let n_fit: usize = args[0].parse().context("n_fit")?;
    let n_val: usize = args[1].parse().context("n_val")?;
    let ctx: usize = args[2].parse().context("ctx")?;
    anyhow::ensure!(n_fit > 0 && n_val > 0 && ctx >= 2, "n_fit, n_val ≥ 1 and ctx ≥ 2");
    let device = llvq_llm::eval::device(&args[3])?;
    let dtype = llvq_llm::eval::dtype(DType::F32)?;

    let artifact = std::env::var("LLVQ_KL_ARTIFACT")
        .ok()
        .filter(|p| !p.is_empty())
        .context("LLVQ_KL_ARTIFACT must name the quantized artifact to measure")?;
    let repo = std::env::var("LLVQ_MODEL").context("LLVQ_MODEL must name the checkpoint")?;
    let skip = env_usize("LLVQ_KL_SKIP", 64 * 2048)?;
    let val_split = std::env::var("LLVQ_KL_VAL").unwrap_or_else(|_| "wikitext2".into());

    println!("kltemp: one temperature between a quantized model and its teacher");
    println!("  checkpoint  {repo}");
    println!(
        "  artifact    {artifact} ({:.3} GB)",
        std::fs::metadata(&artifact)?.len() as f64 / 1e9
    );
    println!("  splits      fit {n_fit} × {ctx}, validation {n_val} × {ctx} on {val_split}");
    println!("  skipped     {skip} tokens of the training corpus");
    println!("  dtype       {dtype:?} on {}", args[3]);

    // ---- corpora ----
    let ck = Checkpoint::fetch(&repo)?;
    let tok = ck.tokenizer()?;
    let train_text = hf_parquet_text(
        "Salesforce/wikitext",
        "wikitext-2-raw-v1/train-00000-of-00001.parquet",
    )?
    .join("\n\n");
    let train_ids: Vec<u32> = tok
        .encode(train_text, false)
        .map_err(anyhow::Error::msg)?
        .get_ids()
        .to_vec();
    anyhow::ensure!(
        train_ids.len() >= skip + (n_fit + n_val) * ctx,
        "training corpus holds {} tokens, short of {skip} skipped plus {} windows",
        train_ids.len(),
        n_fit + n_val
    );
    let fit_ids: Vec<Vec<u32>> = (0..n_fit)
        .map(|w| train_ids[skip + w * ctx..skip + (w + 1) * ctx].to_vec())
        .collect();
    let val_ids: Vec<Vec<u32>> = match val_split.as_str() {
        "wikitext2" => (n_fit..n_fit + n_val)
            .map(|w| train_ids[skip + w * ctx..skip + (w + 1) * ctx].to_vec())
            .collect(),
        "c4" => {
            let c4 = tok
                .encode(llvq_llm::corpus::c4_validation(4_000_000)?, false)
                .map_err(anyhow::Error::msg)?
                .get_ids()
                .to_vec();
            anyhow::ensure!(c4.len() >= n_val * ctx, "c4 validation shorter than {n_val} windows");
            (0..n_val).map(|w| c4[w * ctx..(w + 1) * ctx].to_vec()).collect()
        }
        other => anyhow::bail!("LLVQ_KL_VAL={other:?}: expected wikitext2 or c4"),
    };

    // ---- the teacher, before the artifact touches the projections ----
    let vb = ck.var_builder(dtype, &device)?;
    let mut model = Qwen3::new(&ck.config, vb, llvq_llm::kvq::KvMode::F16)?;
    let t0 = std::time::Instant::now();
    let mut teacher: Vec<Tensor> = Vec::with_capacity(n_fit + n_val);
    let mut entropy: Vec<f64> = Vec::with_capacity(n_fit + n_val);
    for ids in fit_ids.iter().chain(&val_ids) {
        let z = window_logits(&model, ids, &device)?;
        let logp = candle_nn::ops::log_softmax(&z, D::Minus1)?;
        let p = logp.exp()?;
        entropy.push(-sum_rows(&(&p * &logp)?.sum(D::Minus1)?)?);
        teacher.push(p);
    }
    let vocab = teacher[0].dim(1)?;
    println!(
        "\ndense pass: {} windows, {vocab} tokens of vocabulary, {:.0}s",
        teacher.len(),
        t0.elapsed().as_secs_f64()
    );

    // ---- the student: the same object, its projections replaced ----
    let t1 = std::time::Instant::now();
    let (matrices, weights) = llvq_llm::artifact2::load(&mut model, &artifact, &device)?;
    println!(
        "loaded {matrices} matrices, {weights} weights in {:.1}s",
        t1.elapsed().as_secs_f64()
    );
    let t2 = std::time::Instant::now();
    let mut student: Vec<Tensor> = Vec::with_capacity(teacher.len());
    for ids in fit_ids.iter().chain(&val_ids) {
        student.push(window_logits(&model, ids, &device)?);
    }
    println!("quantized pass: {:.0}s", t2.elapsed().as_secs_f64());

    // ---- the statistics that do not depend on the temperature ----
    let mut totals = [Totals::default(), Totals::default()];
    for (w, (p, z)) in teacher.iter().zip(&student).enumerate() {
        let dot = sum_rows(&(p * z)?.sum(D::Minus1)?)?;
        let which = usize::from(w >= n_fit);
        totals[which].add_window(entropy[w], dot, p.dim(0)?);
    }
    let (fit_t, val_t) = (totals[0], totals[1]);

    // ---- the fit, on the first split only ----
    let (found, vertex) = fit(&fit_t, &student[..n_fit])?;
    let val_windows = &student[n_fit..];
    let val_base = val_t.divergence(1.0, log_partition(val_windows, 1.0)?);
    let val_fitted = val_t.divergence(found.beta, log_partition(val_windows, found.beta)?);
    let val = Fit {
        beta: found.beta,
        divergence: val_fitted,
        baseline: val_base,
        at_bound: found.at_bound,
    };

    println!("\nKL(dense ‖ quantized), nats per position");
    println!("  split        at T = 1     at T*        removed");
    println!(
        "  fit         {:>9.6}    {:>9.6}    {:>6.2} %",
        found.baseline,
        found.divergence,
        100.0 * found.recovered()
    );
    println!(
        "  validation  {:>9.6}    {:>9.6}    {:>6.2} %",
        val.baseline,
        val.divergence,
        100.0 * val.recovered()
    );
    println!(
        "\n  T* = {:.5} (beta {:.5}), fitted on {} positions, scored on {}",
        found.temperature(),
        found.beta,
        fit_t.positions,
        val_t.positions
    );
    if let Some(v) = vertex {
        println!("  the parabola through the bracketing triple puts it at beta {v:.5}");
    }
    if found.at_bound {
        println!(
            "\n  REFUSED: the minimum sits on the end of [{BETA_LO}, {BETA_HI}]. That is not a \
             temperature, it is something else being wrong."
        );
    }

    // ---- the control: a temperature cannot move an answer ----
    let (same_1, total) = agreement(&teacher, &student, 1.0)?;
    let (same_t, _) = agreement(&teacher, &student, found.beta)?;
    println!(
        "\ntop-1 agreement with the dense model: {}/{} = {:.4} % at T = 1, {:.4} % at T*",
        same_1,
        total,
        100.0 * same_1 as f64 / total as f64,
        100.0 * same_t as f64 / total as f64
    );
    anyhow::ensure!(
        same_1 == same_t,
        "a positive temperature moved {} argmaxes, which is impossible: the harness is wrong",
        same_1.abs_diff(same_t)
    );
    println!("  identical, as a positive scaling must be — the harness agrees with its own claim");

    println!(
        "\nWhat this does and does not say: {:.1} % of the divergence on held-out text is one \
         number, and that share cannot move an answer. The rest is not thereby shown to matter.",
        100.0 * val.recovered()
    );
    Ok(())
}
