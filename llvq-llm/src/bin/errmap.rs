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
//! `LLVQ_ERRMAP_BLOCKS` (bound on the transformer blocks quantized
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

/// Reads a map written by an earlier run, keyed by (layer, projection).
///
/// Sensitivities cost two model evaluations each and do not change when the
/// question does. Re-measuring them to ask a second question of the same run is
/// two hours spent reproducing numbers already on disk, so a map is loaded and
/// spot-checked rather than recomputed. The spot check is not optional: a map
/// from a different quantization describes a different model, and nothing in
/// the file itself would say so.
fn load_map(path: &std::path::Path, targets: &[Target]) -> anyhow::Result<Vec<Sensitivity>> {
    let text = std::fs::read_to_string(path)?;
    let mut lines = text.lines();
    let header = lines.next().context("empty map")?;
    anyhow::ensure!(
        header.starts_with("layer,projection,gradient,curvature"),
        "{}: not an errmap CSV (header reads {header:?})",
        path.display()
    );
    let mut found: std::collections::HashMap<(usize, String), (f64, f64, f64)> =
        std::collections::HashMap::new();
    for (n, line) in lines.enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        anyhow::ensure!(f.len() >= 7, "{}: line {} has {} fields", path.display(), n + 2, f.len());
        found.insert(
            (f[0].parse()?, f[1].to_string()),
            (f[2].parse()?, f[3].parse()?, f[6].parse()?),
        );
    }
    targets
        .iter()
        .enumerate()
        .map(|(index, t)| {
            let (gradient, curvature, step) = *found
                .get(&(t.layer, t.proj.to_string()))
                .with_context(|| format!("{} is absent from {}", t.name(), path.display()))?;
            Ok(Sensitivity { matrix: index, gradient, curvature, step })
        })
        .collect()
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
    // Two disjoint window ranges of the same corpus. Split A is what the map
    // was fitted on; split B is never used to fit anything, so it is the only
    // number that says whether the map generalizes rather than memorizes.
    // Split B is the real generalization test, and which corpus it reads is the
    // whole question. Another slice of wikitext-2 shows the map survives a
    // different sample of the SAME text; C4 shows it survives a different
    // domain. The probes read wikitext-2 train, which is also GPTQ's own
    // calibration corpus, so a gain that exists only on wikitext would be the
    // map re-tuning the model to one style rather than repairing quantization.
    let split_b = std::env::var("LLVQ_ERRMAP_SPLIT_B").unwrap_or_else(|_| "wikitext2".into());
    let b_ids: Vec<u32> = match split_b.as_str() {
        "wikitext2" => test_ids
            .get(n_eval * eval_ctx..2 * n_eval * eval_ctx)
            .unwrap_or_default()
            .to_vec(),
        "c4" => tok
            .encode(llvq_llm::corpus::c4_validation(4_000_000)?, false)
            .map_err(anyhow::Error::msg)?
            .get_ids()
            .to_vec(),
        other => anyhow::bail!("LLVQ_ERRMAP_SPLIT_B={other:?}: expected wikitext2 or c4"),
    };
    let have_b = b_ids.len() / eval_ctx >= n_eval;
    let nll_over = |m: &Qwen3, ids: &[u32], from: usize, to: usize| -> anyhow::Result<f64> {
        let (mut total, mut count) = (0.0, 0usize);
        for w in from..to {
            let (n, c) = m.window_nll(&ids[w * eval_ctx..(w + 1) * eval_ctx], &mut NoCapture)?;
            total += n;
            count += c;
        }
        Ok(total / count as f64)
    };
    let nll_range = |m: &Qwen3, _from: usize, _to: usize| nll_over(m, &b_ids, 0, n_eval);
    let nll = |m: &Qwen3| nll_over(m, &test_ids, 0, n_eval);

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
    // The probes read CALIBRATION text, never the evaluation corpus. Reading
    // the same windows for both is what a first version did, and its held-out
    // transfer measured -0.008: the map reproduced its own evaluation set and
    // predicted nothing beyond it. A surrogate fitted on the set it is scored
    // on is not a predictive model, it is a lookup of that set.
    let probe_windows = env_usize("LLVQ_ERRMAP_PROBE_WINDOWS", 8)?;
    anyhow::ensure!(
        train_ids.len() >= (n_calib + probe_windows) * calib_len,
        "corpus too short for {n_calib} calibration windows plus {probe_windows} probe windows"
    );
    let probe_ids: Vec<u32> =
        train_ids[n_calib * calib_len..(n_calib + probe_windows) * calib_len].to_vec();
    let probe_nll = |m: &Qwen3| -> anyhow::Result<f64> {
        let (mut total, mut count) = (0.0, 0usize);
        for w in 0..probe_windows {
            let (n, c) =
                m.window_nll(&probe_ids[w * calib_len..(w + 1) * calib_len], &mut NoCapture)?;
            total += n;
            count += c;
        }
        Ok(total / count as f64)
    };

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
    // An artifact already on disk is a quantized model that took two hours to
    // make. Loading it costs seconds, and — this is the part that matters — it
    // is the SAME object the file holds, so a map probed here describes what
    // `bin/mmlu` and `bin/ppl` will read. Two quantizations with identical
    // settings are not the same model: the loop is sequential, so a rounding
    // difference at block 2 is amplified through the remaining blocks, and one
    // measured 6.6 % of perplexity by block 36
    // (docs/mesures/errmap-4b-2026-09-15.txt).
    let from_artifact = std::env::var("LLVQ_ERRMAP_ARTIFACT").ok().filter(|p| !p.is_empty());
    if let Some(path) = &from_artifact {
        let t0 = std::time::Instant::now();
        let (matrices, weights) = llvq_llm::artifact2::load(&mut model, path, &device)?;
        println!(
            "\nloaded {matrices} matrices, {weights} weights from {path} in {:.1}s",
            t0.elapsed().as_secs_f64()
        );
    } else {
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
    }

    // ---- the baselines: one per corpus the map is scored against ----
    let base = probe_nll(&model)?;
    let base_eval = nll(&model)?;
    println!("\nprobe baseline NLL  {base:.8}   (held-in calibration text)");
    println!("eval  baseline NLL  {base_eval:.8}   (perplexity {:.4}, held out)", base_eval.exp());

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

    let one_sided = std::env::var("LLVQ_ERRMAP_ONESIDED").map(|v| v == "1").unwrap_or(false);
    let loaded: Option<Vec<Sensitivity>> = match std::env::var("LLVQ_ERRMAP_LOAD") {
        Ok(path) if !path.is_empty() => {
            let m = load_map(std::path::Path::new(&path), &targets)?;
            println!("loaded {} sensitivities from {path}", m.len());
            Some(m)
        }
        _ => None,
    };
    let mut terms: Vec<Sensitivity> = Vec::with_capacity(targets.len());
    let mut originals: Vec<Tensor> = Vec::with_capacity(targets.len());
    let probe_start = std::time::Instant::now();
    // A loaded map still needs every original tensor kept, and it needs three
    // matrices re-probed: a map describes one quantization, and applying it to
    // another would be silently wrong rather than loudly wrong.
    let spot: Vec<usize> = match &loaded {
        Some(_) => {
            let mut r = llvq_core::SplitMix64::new(0x0059_0715);
            (0..3).map(|_| (r.next() % targets.len() as u64) as usize).collect()
        }
        None => Vec::new(),
    };
    for (index, t) in targets.iter().enumerate() {
        if let Some(map) = &loaded {
            let original = model.blocks[t.layer].linear(t.proj).weight().clone();
            if spot.contains(&index) {
                let mut at = |scale: f64| -> anyhow::Result<f64> {
                    set_scaled(&mut model, t, &original, scale)?;
                    probe_nll(&model)
                };
                let minus = Probe { matrix: index, delta: -eps, loss: at(1.0 - eps)? };
                let plus = Probe { matrix: index, delta: eps, loss: at(1.0 + eps)? };
                set_scaled(&mut model, t, &original, 1.0)?;
                let fresh = differentiate(base, minus, plus).map_err(anyhow::Error::msg)?;
                let want = map[index];
                let scale = want.gradient.abs().max(1e-9);
                let drift = (fresh.gradient - want.gradient).abs() / scale;
                println!(
                    "  spot check {:<28} gradient {:+.6e} against {:+.6e}, drift {:.3}",
                    t.name(),
                    fresh.gradient,
                    want.gradient,
                    drift
                );
                anyhow::ensure!(
                    drift < 0.05,
                    "{} drifted {drift:.3} from the loaded map: it describes another run",
                    t.name()
                );
            }
            originals.push(original);
            terms.push(map[index]);
            continue;
        }
        let original = model.blocks[t.layer].linear(t.proj).weight().clone();
        let mut at = |scale: f64| -> anyhow::Result<f64> {
            set_scaled(&mut model, t, &original, scale)?;
            probe_nll(&model)
        };
        let sensitivity = if one_sided {
            // Half the evaluations, and no curvature. Justified by measurement,
            // not by thrift: 96.2 % of the reachable gain comes from the SIGN
            // of the gradient (docs/mesures/c4-baselines-0.6b-2026-09-15.txt),
            // and a sign survives a one-sided difference. Curvature is reported
            // as zero, which makes `optimum_within` return ±trust by sign —
            // exactly the map of directions this mode is for, and never a
            // fabricated interior optimum.
            let plus = at(1.0 + eps)?;
            Sensitivity {
                matrix: index,
                gradient: (plus - base) / eps,
                curvature: 0.0,
                step: eps,
            }
        } else {
            let minus = Probe { matrix: index, delta: -eps, loss: at(1.0 - eps)? };
            let plus = Probe { matrix: index, delta: eps, loss: at(1.0 + eps)? };
            differentiate(base, minus, plus).map_err(anyhow::Error::msg)?
        };
        set_scaled(&mut model, t, &original, 1.0)?;
        originals.push(original);
        terms.push(sensitivity);
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

    // A cheap map is only worth its saving if it agrees with the expensive one
    // where it matters. What matters is the sign, so that is what is scored,
    // on the matrices the reference itself can see.
    if let Ok(path) = std::env::var("LLVQ_ERRMAP_COMPARE") {
        let reference = load_map(std::path::Path::new(&path), &targets)?;
        let gmax = reference.iter().fold(0.0f64, |m, s| m.max(s.gradient.abs()));
        let visible: Vec<usize> = reference
            .iter()
            .filter(|s| s.gradient.abs() > 1e-3 * gmax)
            .map(|s| s.matrix)
            .collect();
        let agree = visible
            .iter()
            .filter(|&&i| surrogate.terms[i].gradient.signum() == reference[i].gradient.signum())
            .count();
        let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
        for &i in &visible {
            let (x, y) = (reference[i].gradient, surrogate.terms[i].gradient);
            sxy += x * y;
            sxx += x * x;
            syy += y * y;
        }
        println!("\n--- agreement with {path} ---");
        println!(
            "  sign agreement on the reference's {} visible matrices: {}/{} ({:.2} %)",
            visible.len(),
            agree,
            visible.len(),
            100.0 * agree as f64 / visible.len() as f64
        );
        println!("  cosine between gradient vectors: {:.6}", sxy / (sxx * syy).sqrt());
        println!("  regression slope, cheap on reference: {:.4}", sxy / sxx);
    }

    // ---- the map ----
    // Four orders of magnitude below the rest is the measured gap for q/k.
    let invariant = surrogate.scale_invariant(1e-3);
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

    // ---- the validation plan ----
    //
    // One combination tells almost nothing: it says the surrogate worked once,
    // at one size, in the one regime its own ranking favours. What a predictive
    // model owes is a domain — how large a move, over how many matrices, before
    // its error stops being small. So the plan sweeps both, picks matrices two
    // ways, and scores every point on a split the map never saw.
    let sizes: Vec<usize> = [1usize, 2, 4, 8, 16, 32, 64, 140]
        .into_iter()
        .filter(|&k| k <= surrogate.terms.len())
        .collect();
    let amplitudes = [0.5f64, 1.0];
    let ranked: Vec<Sensitivity> = surrogate.ranked_within(trust);
    // Matrices the loss can actually see. Including the scale-invariant ones
    // would pad every combination with terms that move nothing, and flatter
    // the error by diluting it.
    let visible: Vec<Sensitivity> = ranked
        .iter()
        .copied()
        .filter(|s| s.gradient != 0.0 || s.curvature != 0.0)
        .collect();
    let mut rng = llvq_core::SplitMix64::new(0xe22_0915);

    let base_a = base_eval;
    let base_b = if have_b { nll_range(&model, n_eval, 2 * n_eval)? } else { f64::NAN };
    println!(
        "\nsplit A  {base_a:.8}  (wikitext-2 test)\nsplit B  {base_b:.8}  ({split_b})"
    );
    let run_plan = std::env::var("LLVQ_ERRMAP_PLAN").map(|v| v != "0").unwrap_or(true);
    if run_plan {
    println!("\n--- validation plan: {} points ---", sizes.len() * amplitudes.len() * 2);
    // Moves, not absolute losses: the surrogate predicts a *change*, the two
    // splits have different baselines, and a table of absolute NLLs hides
    // whether a gain transferred at all.
    println!(
        "  {:<10} {:>4} {:>6} {:>11} {:>11} {:>11} {:>9} {:>9}",
        "selection", "k", "alpha", "move pred", "move A", "move B", "err A", "err B"
    );
    let mut worst_a: f64 = 0.0;
    let mut worst_b: f64 = 0.0;
    let mut sign_failures = 0usize;
    // How much of the predicted improvement actually appears on the split the
    // map was never fitted on. One is full transfer, zero is none, and
    // negative means the move hurt where it was supposed to help.
    let mut transfer: Vec<f64> = Vec::new();
    for &k in &sizes {
        for &alpha in &amplitudes {
            for pick in ["top", "random"] {
                let chosen: Vec<Sensitivity> = if pick == "top" {
                    visible.iter().copied().take(k).collect()
                } else {
                    // Sampling without replacement, so a "random" combination
                    // is a genuine subset and not a multiset of one matrix.
                    let mut pool: Vec<Sensitivity> = visible.clone();
                    let mut out = Vec::with_capacity(k);
                    for _ in 0..k.min(pool.len()) {
                        let i = (rng.next() % pool.len() as u64) as usize;
                        out.push(pool.swap_remove(i));
                    }
                    out
                };
                let moves: Vec<(usize, f64)> = chosen
                    .iter()
                    .map(|s| (s.matrix, alpha * s.optimum_within(trust)))
                    .filter(|&(_, d)| d != 0.0)
                    .collect();
                if moves.is_empty() {
                    continue;
                }
                let predicted = surrogate.predict(&moves);
                for &(index, delta) in &moves {
                    set_scaled(&mut model, &targets[index], &originals[index], 1.0 + delta)?;
                }
                let measured_a = nll(&model)?;
                let measured_b = if have_b {
                    nll_range(&model, n_eval, 2 * n_eval)?
                } else {
                    f64::NAN
                };
                for &(index, _) in &moves {
                    set_scaled(&mut model, &targets[index], &originals[index], 1.0)?;
                }
                // The surrogate predicts a CHANGE, measured on the probe
                // corpus; each validation split carries it from its own base.
                let move_predicted = predicted - base;
                let ra = Residual {
                    predicted: base_a + move_predicted,
                    measured: measured_a,
                    base: base_a,
                };
                let rel_a = ra.relative_to_move().unwrap_or(f64::NAN);
                // Split B has its own baseline, and the surrogate's predicted
                // *change* is what transfers — not its predicted absolute loss.
                let rel_b = if have_b {
                    let moved = move_predicted;
                    (measured_b - (base_b + moved)) / moved.abs()
                } else {
                    f64::NAN
                };
                let move_pred = move_predicted;
                let move_a = measured_a - base_a;
                let move_b = measured_b - base_b;
                // A predicted move at the level of the evaluation's own
                // precision makes every ratio meaningless, so it is shown and
                // excluded from the worst case rather than allowed to dominate.
                let scorable = move_pred.abs() > 1e-3;
                if scorable {
                    if !ra.agrees_in_sign() {
                        sign_failures += 1;
                    }
                    worst_a = worst_a.max(rel_a.abs());
                    if rel_b.is_finite() {
                        worst_b = worst_b.max(rel_b.abs());
                    }
                    transfer.push(move_b / move_pred);
                }
                println!(
                    "  {:<10} {:>4} {:>6.2} {:>+11.6} {:>+11.6} {:>+11.6} {:>+9.4} {:>+9.4}{}",
                    pick,
                    moves.len(),
                    alpha,
                    move_pred,
                    move_a,
                    move_b,
                    rel_a,
                    rel_b,
                    if scorable { "" } else { "  (below precision, not scored)" }
                );
            }
        }
    }
    println!(
        "\nworst |error| as a fraction of the move: split A {worst_a:.4}, split B {worst_b:.4}"
    );
    println!("sign disagreements: {sign_failures}");
    if !transfer.is_empty() {
        let mut v = transfer.clone();
        v.sort_by(f64::total_cmp);
        let mean = v.iter().sum::<f64>() / v.len() as f64;
        println!(
            "transfer to the held-out split: mean {mean:+.4}, median {:+.4}, min {:+.4}, max {:+.4}",
            v[v.len() / 2],
            v[0],
            v[v.len() - 1]
        );
        println!(
            "  1.0 = the predicted gain appears in full on data the map never saw; \n               0.0 = none of it does, and the map fitted the evaluation set"
        );
    }
    }
    // ---- how far to believe the parabola ----
    //
    // 147 of 196 optima sit at the edge of the trust region, so the region and
    // not the curvature is what bounds the map. Widening it claims more and
    // predicts worse, and where that trade sits is a measurement. The claims
    // alone already bound the sweep: past T = 0.08 the model asserts a
    // quantized perplexity below the f32 model's own, which is impossible, so
    // the grid stops before the parabola starts inventing.
    if let Ok(spec) = std::env::var("LLVQ_ERRMAP_TRUST_SWEEP") {
        let widths: Vec<f64> = spec
            .split(',')
            .map(|s| s.trim().parse::<f64>().context("LLVQ_ERRMAP_TRUST_SWEEP is a comma list"))
            .collect::<anyhow::Result<_>>()?;
        println!("\n--- trust region sweep, all matrices at their own optimum ---");
        println!(
            "  {:>6} {:>7} {:>12} {:>12} {:>12} {:>9} {:>9}",
            "T", "at edge", "claimed", "move A", "move B", "realized", "err A"
        );
        for &w in &widths {
            anyhow::ensure!(w > 0.0 && w <= 0.5, "trust width {w} is outside (0, 0.5]");
            let moves: Vec<(usize, f64)> = surrogate
                .terms
                .iter()
                .map(|s| (s.matrix, s.optimum_within(w)))
                .filter(|&(_, d)| d != 0.0)
                .collect();
            let at_edge = moves.iter().filter(|(_, d)| (d.abs() - w).abs() < 1e-12).count();
            let claimed = surrogate.predict(&moves) - base;
            for &(index, delta) in &moves {
                set_scaled(&mut model, &targets[index], &originals[index], 1.0 + delta)?;
            }
            let ma = nll(&model)? - base_a;
            let mb = if have_b { nll_range(&model, n_eval, 2 * n_eval)? - base_b } else { f64::NAN };
            for &(index, _) in &moves {
                set_scaled(&mut model, &targets[index], &originals[index], 1.0)?;
            }
            println!(
                "  {:>6.3} {:>7} {:>+12.6} {:>+12.6} {:>+12.6} {:>9.3} {:>+9.4}",
                w,
                at_edge,
                claimed,
                ma,
                mb,
                ma / claimed,
                (ma - claimed) / claimed.abs()
            );
        }
        println!("  realized = measured move / claimed move; 1.0 would be a model that delivers");
    }

    // ---- baselines, because "better than nothing" is not a claim ----
    //
    // The map moves 196 matrices and gets a large number. Three things could
    // produce a large number: the map being right, ANY set of moves of that
    // size being right, or one global direction being right. These separate
    // them, and the review that asked for them was right that `random k` above
    // does not: that one draws a random SUBSET of the map's own corrections,
    // which still carries the map's signs and sizes.
    if std::env::var("LLVQ_ERRMAP_BASELINES").map(|v| v != "0").unwrap_or(true) {
        let width = env_f64("LLVQ_ERRMAP_BASELINE_TRUST", 0.04)?;
        let visible_idx: Vec<usize> = surrogate
            .terms
            .iter()
            .filter(|s| s.optimum_within(width) != 0.0)
            .map(|s| s.matrix)
            .collect();
        println!("\n--- baselines at T = {width}, {} matrices moved ---", visible_idx.len());
        println!("  {:<34} {:>11} {:>11}", "arm", "move A", "move B");
        let mut measure = |name: &str, moves: &[(usize, f64)]| -> anyhow::Result<()> {
            for &(index, delta) in moves {
                set_scaled(&mut model, &targets[index], &originals[index], 1.0 + delta)?;
            }
            let a = nll(&model)? - base_a;
            let b = if have_b { nll_range(&model, n_eval, 2 * n_eval)? - base_b } else { f64::NAN };
            for &(index, _) in moves {
                set_scaled(&mut model, &targets[index], &originals[index], 1.0)?;
            }
            println!("  {name:<34} {a:>+11.6} {b:>+11.6}");
            Ok(())
        };

        let map_moves: Vec<(usize, f64)> = surrogate
            .terms
            .iter()
            .map(|s| (s.matrix, s.optimum_within(width)))
            .filter(|&(_, d)| d != 0.0)
            .collect();
        measure("the map", &map_moves)?;

        // Random sign and size over the same matrices: the null hypothesis
        // that any perturbation of this magnitude would do.
        let mut r = llvq_core::SplitMix64::new(0x0ba5_0915);
        for trial in 1..=3 {
            let moves: Vec<(usize, f64)> = visible_idx
                .iter()
                .map(|&i| {
                    let u = (r.next() >> 11) as f64 / (1u64 << 53) as f64;
                    (i, width * (2.0 * u - 1.0))
                })
                .collect();
            measure(&format!("random sign and size, draw {trial}"), &moves)?;
        }
        // The map's signs with a constant size, which asks how much of the
        // gain is in knowing WHICH WAY each matrix should go rather than by
        // how much.
        let signs: Vec<(usize, f64)> = map_moves
            .iter()
            .map(|&(i, d)| (i, width * d.signum()))
            .collect();
        measure("the map's signs, constant size", &signs)?;
        // One global scale for every matrix, the arm the four re-encoded
        // sweeps measured at 1.02.
        for s in [0.99, 1.01, 1.02] {
            let moves: Vec<(usize, f64)> = visible_idx.iter().map(|&i| (i, s - 1.0)).collect();
            measure(&format!("one global scale {s}"), &moves)?;
        }
    }

    // ---- write the corrected model out ----
    //
    // Every arm above restores the weights, which is right for a measurement
    // and useless for a deliverable: the best model this binary ever holds is
    // gone the moment it is scored. This applies the map once more and saves,
    // so the corrected object exists as a file that `bin/ppl` and `bin/mmlu`
    // can read.
    if let Ok(path) = std::env::var("LLVQ_ERRMAP_SAVE") {
        let width = env_f64("LLVQ_ERRMAP_SAVE_TRUST", 0.04)?;
        let moves: Vec<(usize, f64)> = surrogate
            .terms
            .iter()
            .map(|s| (s.matrix, s.optimum_within(width)))
            .filter(|&(_, d)| d != 0.0)
            .collect();
        for &(index, delta) in &moves {
            set_scaled(&mut model, &targets[index], &originals[index], 1.0 + delta)?;
        }
        let corrected = nll(&model)?;
        llvq_llm::artifact::save(&model, &path)?;
        println!(
            "\nsaved the corrected model to {path} at T = {width}: {} matrices moved, NLL {corrected:.8}, perplexity {:.4}",
            moves.len(),
            corrected.exp()
        );
        // Left applied on purpose: the process ends here and the file is what
        // matters. The restore check below is skipped for the same reason.
        return Ok(());
    }

    let restored = nll(&model)?;
    anyhow::ensure!(
        (restored - base_a).abs() < 1e-12,
        "the model did not come back to its baseline: {restored} against {base_a}"
    );
    println!("model restored, NLL {restored:.8}");

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
