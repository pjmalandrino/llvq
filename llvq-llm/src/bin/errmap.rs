//! Measures the model's own sensitivity to a scale error on each matrix, and
//! turns it into a calibration map.
//!
//! ```text
//! LLVQ_MODEL=Qwen/Qwen3-0.6B cargo run --release -p llvq-llm \
//!   --features metal,fast-linalg --bin errmap -- 64 2048 12 2048 metal
//! ```
//!
//! ## What it measures, and why not the other thing
//!
//! GPTQ calibrates against `tr(E H Eᵀ)`, the error one matrix puts on its own
//! output. On 2026-09-15 that objective and the model disagreed on the same
//! knob: the layer objective's closed-form optimum is 0.999, the perplexity
//! minimum is 1.02 (*measured*, `docs/mesures/gain-scale-0.6b-2026-09-15.txt`).
//! So this binary probes the endpoint — the model's NLL — and never the layer.
//!
//! For each matrix it scales the **quantized** weights by `1 ± ε` and evaluates.
//! Two evaluations give the gradient and the curvature of the NLL in that
//! direction ([`llvq_llm::errmodel`]), and from there any vector of per-matrix
//! scales is predicted without evaluating again. That is the point: candidates
//! are ranked for the cost of the probes, not the cost of the candidates.
//!
//! ## What a probe here is not
//!
//! Scaling the quantized weights **after** the loop is not the same as scaling
//! the fitted centroids **before** it (`LLVQ_GAIN_SCALE`). The second changes
//! which level each block picks and changes the error every later column is
//! compensated against; the first changes neither. The two coincide only to
//! the extent that the sequential compensation does not react, which is an
//! assumption this binary is built to *measure* rather than to make: the
//! pooled probe prediction is printed against the sweep's own measured
//! minimum, and a disagreement there is the size of the compensation's
//! reaction.
//!
//! Environment: `LLVQ_ERRMAP_EPS` (default 0.01), `LLVQ_ERRMAP_OUT` (CSV path),
//! `LLVQ_ERRMAP_TYPES` (comma-separated projection names, default all seven),
//! `LLVQ_ERRMAP_TOP` (how many matrices the validation combination moves,
//! default 8), `LLVQ_ERRMAP_BLOCKS` (bound on the transformer blocks quantized
//! and probed, default all). `LLVQ_MODEL` picks the checkpoint.

use anyhow::Context;
use candle_core::{DType, Tensor};
use llvq_llm::calib::{Codebook, RunConfig};
use llvq_llm::corpus::{hf_parquet_text, wikitext2_test};
use llvq_llm::errmodel::{differentiate, Probe, Residual, Sensitivity, Surrogate};
use llvq_llm::loader::Checkpoint;
use llvq_llm::model::{NoCapture, Qwen3};
use llvq_quant::gptq::{GptqConfig, TailPolicy};

/// Every projection of a Qwen3 block, in the order the loop walks them.
const ALL_TYPES: [&str; 7] = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
];

fn env_f64(name: &str, default: f64) -> anyhow::Result<f64> {
    match std::env::var(name) {
        Ok(s) => s
            .parse()
            .with_context(|| format!("{name}={s:?} is not a number")),
        Err(_) => Ok(default),
    }
}

fn env_usize(name: &str, default: usize) -> anyhow::Result<usize> {
    match std::env::var(name) {
        Ok(s) => s
            .parse()
            .with_context(|| format!("{name}={s:?} is not an integer")),
        Err(_) => Ok(default),
    }
}

/// One probed matrix: which block and which projection.
struct Target {
    layer: usize,
    proj: &'static str,
}

impl Target {
    fn name(&self) -> String {
        format!("blocks.{}.{}", self.layer, self.proj)
    }
}

/// Replaces one projection's weights by `scale ×` the tensor given, and returns
/// nothing: the caller owns the original and is responsible for putting it back.
fn set_scaled(model: &mut Qwen3, t: &Target, original: &Tensor, scale: f64) -> anyhow::Result<()> {
    let w = if scale == 1.0 {
        original.clone()
    } else {
        (original * scale)?
    };
    *model.blocks[t.layer].linear_mut(t.proj) = candle_nn::Linear::new(w, None);
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    anyhow::ensure!(
        args.len() >= 5,
        "usage: errmap <n_calib> <calib_len> <n_eval> <eval_ctx> <device>"
    );
    let n_calib: usize = args[0].parse()?;
    let calib_len: usize = args[1].parse()?;
    let n_eval: usize = args[2].parse()?;
    let eval_ctx: usize = args[3].parse()?;
    let device = llvq_llm::eval::device(&args[4])?;
    let dtype = llvq_llm::eval::dtype(DType::F32)?;

    let eps = env_f64("LLVQ_ERRMAP_EPS", 0.01)?;
    anyhow::ensure!(
        eps.is_finite() && eps > 0.0 && eps < 0.5,
        "LLVQ_ERRMAP_EPS must be in (0, 0.5), got {eps}"
    );
    let top = env_usize("LLVQ_ERRMAP_TOP", 8)?;
    // Blocks to quantize AND to probe. The two must be the same set: probing a
    // block the loop never touched would measure the sensitivity of an f32
    // matrix and report it as a calibration figure.
    let limit = env_usize("LLVQ_ERRMAP_BLOCKS", usize::MAX)?;
    // How far the quadratic is believed past the probes. Measured curvatures
    // are not reliably positive — a run does not sit at a minimum of its own
    // loss — so the map minimizes over this interval instead of solving -g/h
    // and extrapolating wherever that lands.
    let trust = env_f64("LLVQ_ERRMAP_TRUST", 3.0 * eps)?;
    anyhow::ensure!(
        trust.is_finite() && trust > 0.0 && trust <= 0.5,
        "LLVQ_ERRMAP_TRUST must be in (0, 0.5], got {trust}"
    );
    let types: Vec<&'static str> = match std::env::var("LLVQ_ERRMAP_TYPES") {
        Ok(s) if !s.is_empty() => s
            .split(',')
            .map(|want| {
                ALL_TYPES
                    .iter()
                    .copied()
                    .find(|t| *t == want.trim())
                    .with_context(|| format!("unknown projection {want:?}"))
            })
            .collect::<anyhow::Result<_>>()?,
        _ => ALL_TYPES.to_vec(),
    };
    let repo = std::env::var("LLVQ_MODEL").unwrap_or_else(|_| "Qwen/Qwen3-0.6B".into());

    println!("=== errmap ===");
    println!("model           {repo}");
    println!("probe step      ±{eps}");
    println!("trust region    ±{trust}");
    println!("projections     {}", types.join(", "));
    println!("dtype / device  {} / {device:?}", llvq_llm::eval::dtype_name(dtype));

    // ---- model and corpora ----
    let ck = Checkpoint::fetch(&repo)?;
    let tok = ck.tokenizer()?;
    let vb = ck.var_builder(dtype, &device)?;
    let mut model = Qwen3::new(&ck.config, vb, llvq_llm::kvq::KvMode::F16)?;

    let test_ids = tok
        .encode(wikitext2_test()?, false)
        .map_err(anyhow::Error::msg)?
        .get_ids()
        .to_vec();
    let n_eval = n_eval.min(test_ids.len() / eval_ctx);
    anyhow::ensure!(n_eval > 0, "corpus shorter than one eval window");

    // The NLL rather than its exponential: the surrogate is a second-order
    // expansion, and expanding a perplexity would model `exp` of the thing
    // that is actually additive over windows.
    let nll = |m: &Qwen3| -> anyhow::Result<f64> {
        let (mut total, mut count) = (0.0, 0usize);
        for w in 0..n_eval {
            let (n, c) = m.window_nll(&test_ids[w * eval_ctx..(w + 1) * eval_ctx], &mut NoCapture)?;
            total += n;
            count += c;
        }
        Ok(total / count as f64)
    };

    let train_text = hf_parquet_text(
        "Salesforce/wikitext",
        "wikitext-2-raw-v1/train-00000-of-00001.parquet",
    )?
    .join("\n\n");
    let train_ids = tok
        .encode(train_text, false)
        .map_err(anyhow::Error::msg)?
        .get_ids()
        .to_vec();
    anyhow::ensure!(
        train_ids.len() >= n_calib * calib_len,
        "calibration corpus is shorter than {n_calib} × {calib_len} tokens"
    );
    let mut hidden: Vec<Tensor> = Vec::with_capacity(n_calib);
    for w in 0..n_calib {
        let ids = &train_ids[w * calib_len..(w + 1) * calib_len];
        hidden.push(model.embed_tokens(&Tensor::from_slice(ids, (1, calib_len), &device)?)?);
    }

    // ---- quantize once, the served way ----
    let codebook = Codebook::Tetra {
        gain_bits: 1,
        post_shape_gain: false,
    };
    let run = RunConfig {
        gptq: GptqConfig {
            block: codebook.block_len(),
            retract: true,
            group_scales: false,
            design_c: false,
            lambda: 1e-2,
            tail: TailPolicy::KeepExact,
        },
        int4_types: Vec::new(),
        damping: 1e-2,
        h_shrink: 1.0,
        gain_scale: 1.0,
        codebook,
        threads: env_usize("LLVQ_THREADS", 0)?,
        start: 0,
        limit,
        rotation_seed: Some(0x110feed),
    };
    println!("\nquantizing {} blocks…", limit.min(model.blocks.len()));
    let t0 = std::time::Instant::now();
    let report = llvq_llm::calib::quantize_model_capturing(
        &mut model,
        &mut hidden,
        &run,
        |t, n, _| {
            if t % 8 == 0 || t + 1 == n {
                eprintln!("  block {}/{n}", t + 1);
            }
        },
        None,
    )?;
    println!(
        "  {} matrices, {:.4} bits/weight, {:.0}s",
        report.matrices,
        report.bits_per_weight(),
        t0.elapsed().as_secs_f64()
    );

    // ---- the baseline the whole map is an expansion around ----
    let base = nll(&model)?;
    println!("\nbaseline NLL    {base:.8}   (perplexity {:.4})", base.exp());

    // ---- probes ----
    let probed_blocks = limit.min(model.blocks.len());
    let targets: Vec<Target> = (0..probed_blocks)
        .flat_map(|layer| {
            types.iter().map(move |&proj| Target { layer, proj })
        })
        .collect();
    println!(
        "probing {} matrices, {} evaluations…",
        targets.len(),
        2 * targets.len()
    );

    let mut terms: Vec<Sensitivity> = Vec::with_capacity(targets.len());
    let mut originals: Vec<Tensor> = Vec::with_capacity(targets.len());
    let probe_start = std::time::Instant::now();
    for (index, t) in targets.iter().enumerate() {
        let original = model.blocks[t.layer].linear(t.proj).weight().clone();
        let mut at = |scale: f64| -> anyhow::Result<f64> {
            set_scaled(&mut model, t, &original, scale)?;
            nll(&model)
        };
        let minus = Probe {
            matrix: index,
            delta: -eps,
            loss: at(1.0 - eps)?,
        };
        let plus = Probe {
            matrix: index,
            delta: eps,
            loss: at(1.0 + eps)?,
        };
        set_scaled(&mut model, t, &original, 1.0)?;
        originals.push(original);
        terms.push(differentiate(base, minus, plus).map_err(anyhow::Error::msg)?);
        if index % 16 == 0 || index + 1 == targets.len() {
            eprintln!(
                "  {}/{}  {:.0}s elapsed",
                index + 1,
                targets.len(),
                probe_start.elapsed().as_secs_f64()
            );
        }
    }
    let surrogate = Surrogate::new(base, terms).map_err(anyhow::Error::msg)?;

    // ---- the map ----
    let invariant = surrogate.scale_invariant();
    println!(
        "\nshape of the surface: {} of {} directions concave or flat, {} invisible to the loss",
        surrogate.non_convex(),
        surrogate.terms.len(),
        invariant.len()
    );
    if !invariant.is_empty() {
        // Expected rather than surprising: Qwen3 puts an RMS norm on each
        // head of q and k, and an RMS norm is scale invariant. A scale error
        // on those matrices is free, which is a calibration fact.
        let mut kinds: Vec<&str> = invariant.iter().map(|&i| targets[i].proj).collect();
        kinds.sort_unstable();
        kinds.dedup();
        println!("  invisible directions are all of: {}", kinds.join(", "));
    }

    println!("\n--- the ten matrices where a scale error costs most ---");
    println!(
        "  {:<28} {:>12} {:>12} {:>9} {:>12}",
        "matrix", "gradient", "curvature", "scale", "gain(NLL)"
    );
    for s in surrogate.ranked_within(trust).iter().take(10) {
        let t = &targets[s.matrix];
        println!(
            "  {:<28} {:>12.4e} {:>12.4e} {:>9.5} {:>12.4e}",
            t.name(),
            s.gradient,
            s.curvature,
            1.0 + s.optimum_within(trust),
            s.best_gain_within(trust)
        );
    }
    let claimed = surrogate.claimed_gain_within(trust);
    println!(
        "\nclaimed by the map, all matrices at their own optimum: {claimed:+.6e} of NLL"
    );
    println!(
        "  which is {:.4} of perplexity, from {:.4}",
        (base + claimed).exp(),
        base.exp()
    );

    // The pooled scale the map would pick if every matrix had to share one,
    // weighted by curvature — the quantity the 62-minute sweep measured
    // independently, and therefore the map's first confrontation with reality.
    let (num, den): (f64, f64) = surrogate
        .terms
        .iter()
        .fold((0.0, 0.0), |(n, d), t| (n - t.gradient, d + t.curvature));
    if den > 0.0 {
        println!(
            "\npooled single scale from the map: {:.5}",
            1.0 + num / den
        );
        println!("  the sweep of 2026-09-15 measured its minimum at 1.02 by re-encoding");
        println!("  a gap here is the size of the compensation's reaction, not an error");
    }

    // ---- the held-out test: a combination the probes never saw ----
    let moves: Vec<(usize, f64)> = surrogate
        .ranked_within(trust)
        .iter()
        .take(top)
        .map(|s| (s.matrix, s.optimum_within(trust)))
        .filter(|&(_, d)| d != 0.0)
        .collect();
    if !moves.is_empty() {
        println!(
            "\n--- validation: the top {} matrices moved together ---",
            moves.len()
        );
        let predicted = surrogate.predict(&moves);
        for &(index, delta) in &moves {
            set_scaled(&mut model, &targets[index], &originals[index], 1.0 + delta)?;
        }
        let measured = nll(&model)?;
        for &(index, _) in &moves {
            set_scaled(&mut model, &targets[index], &originals[index], 1.0)?;
        }
        let r = Residual {
            predicted,
            measured,
            base,
        };
        println!("  predicted NLL {predicted:.8}   measured {measured:.8}");
        println!("  absolute error {:+.4e}", r.absolute());
        match r.relative_to_move() {
            Some(rel) => println!("  error as a fraction of the predicted move: {rel:+.4}"),
            None => println!("  the surrogate predicted no move; no ratio is meaningful"),
        }
        println!(
            "  sign of the move: {}",
            if r.agrees_in_sign() {
                "prediction and measurement agree"
            } else {
                "DISAGREE — the map cannot rank what it cannot sign"
            }
        );
        // Restoring is not optional: a later evaluation on a model left
        // perturbed would silently measure the combination instead of the run.
        let restored = nll(&model)?;
        anyhow::ensure!(
            (restored - base).abs() < 1e-12,
            "the model did not come back to its baseline: {restored} against {base}"
        );
        println!("  model restored to baseline, NLL {restored:.8}");
    }

    if let Ok(path) = std::env::var("LLVQ_ERRMAP_OUT") {
        use std::io::Write;
        let mut f = std::fs::File::create(&path)?;
        writeln!(f, "layer,projection,gradient,curvature,optimal_scale,gain_nll,step")?;
        for s in &surrogate.terms {
            let t = &targets[s.matrix];
            writeln!(
                f,
                "{},{},{:.10e},{:.10e},{:.8},{:.10e},{}",
                t.layer,
                t.proj,
                s.gradient,
                s.curvature,
                1.0 + s.optimum_within(trust),
                s.best_gain_within(trust),
                s.step
            )?;
        }
        println!("\ncsv             {path}");
    }
    Ok(())
}
