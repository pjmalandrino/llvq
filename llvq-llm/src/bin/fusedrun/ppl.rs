//! The perplexity modes, `LLVQ_GQA_PPL` in its forms and the `LLVQ_KV_LADDER`
//! arm it scores. One file because every one of them pairs two arms window by
//! window and reads the result through [`paired_ppl_pct`].

// Unconditional since the mode functions the Mac type-checks use them too.
use std::time::Instant;
use llvq_llm::model::NoCapture;

use super::common::{GqaArm, GqaPplSpec};

/// Per-window differences of mean NLL, `b − a`, as `ppl_b / ppl_a − 1` in %:
/// the median-free point `exp(mean) − 1` and its paired 95 % t interval.
/// Every window scores the same number of tokens, so `exp(mean d)` is the
/// ratio of the two perplexities exactly.
#[cfg_attr(not(all(target_os = "linux", feature = "cuda")), allow(dead_code))]
pub(super) fn paired_ppl_pct(a: &[f64], b: &[f64]) -> (f64, f64, f64) {
    let d: Vec<f64> = a.iter().zip(b).map(|(x, y)| y - x).collect();
    let n = d.len() as f64;
    let m = d.iter().sum::<f64>() / n;
    let sd = (d.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
    let h = llvq_llm::eval::t95(d.len().saturating_sub(1)) * sd / n.sqrt();
    let pct = |x: f64| x.exp_m1() * 100.0;
    (pct(m), pct(m - h), pct(m + h))
}

/// What `LLVQ_KV_LADDER`'s arm reads besides the tensors, built and installed
/// on every block before the first scored window: the Hadamard matrix of
/// `k4v4h`, and the static key steps of `k4v4c`, calibrated on the first
/// [`llvq_llm::kvladder::CALIB_TOKENS`] tokens of `dclm-edu` (the corpus the
/// sealed files were calibrated on, never the scored WikiText-2), one prefill
/// on the f16 cache under the arm's attention form. The steps are printed as a
/// fingerprint and a range, so two runs can be told to be the same arm.
#[cfg_attr(not(all(target_os = "linux", feature = "cuda")), allow(dead_code))]
pub(super) fn install_ladder(
    model: &mut llvq_llm::model::Qwen3,
    tokenizer: &tokenizers::Tokenizer,
    ladder: &llvq_llm::kvladder::Ladder,
    device: &candle_core::Device,
) -> anyhow::Result<()> {
    use llvq_llm::kvladder::{calibrate, hadamard_matrix, LadderAux, CALIB_TOKENS};
    use llvq_llm::kvq::KvMode;
    let n = model.config().num_hidden_layers;
    let hd = model.config().head_dim;
    let hadamard = match ladder.mode {
        KvMode::K4V4H => Some(hadamard_matrix(hd, device)?),
        _ => None,
    };
    let steps = match ladder.mode {
        KvMode::K4V4C => {
            let clip = ladder.clip.expect("k4v4c carries its clip");
            GqaArm::Kernel.apply(model);
            let text = llvq_llm::kvcapture::CaptureCorpus::DclmEdu.text(CALIB_TOKENS * 8)?;
            let all = tokenizer.encode(text.as_str(), false).map_err(|e| anyhow::anyhow!("{e}"))?.get_ids().to_vec();
            if all.len() < CALIB_TOKENS {
                anyhow::bail!("k4v4c calibration: {} dclm-edu tokens, {CALIB_TOKENS} needed", all.len());
            }
            let calib = &all[..CALIB_TOKENS];
            let t = Instant::now();
            let steps = calibrate(model, calib, llvq_llm::model::MAX_PREFILL_ROWS, clip)?;
            let mut flat = Vec::new();
            for s in &steps {
                flat.extend(s.flatten_all()?.to_vec1::<f32>()?);
            }
            let bits: Vec<u32> = flat.iter().map(|x| x.to_bits()).collect();
            let mut sorted = flat.clone();
            sorted.sort_unstable_by(f32::total_cmp);
            println!(
                "k4v4c: static key steps from {CALIB_TOKENS} dclm-edu tokens (fingerprint {:016x}), clip quantile \
                 {clip} of each RoPE pair's norm over 7 levels, {n} layers x {} values; step min {:.4e} median \
                 {:.4e} max {:.4e}; steps fingerprint {:016x} ({:.1} s)",
                llvq_llm::eval::token_fingerprint(calib),
                flat.len() / n,
                sorted[0],
                sorted[sorted.len() / 2],
                sorted[sorted.len() - 1],
                llvq_llm::eval::token_fingerprint(&bits),
                t.elapsed().as_secs_f64()
            );
            Some(steps)
        }
        _ => None,
    };
    let aux = (0..n)
        .map(|i| LadderAux { key_step: steps.as_ref().map(|s| s[i].clone()), hadamard: hadamard.clone() })
        .collect();
    model.set_kv_round_trip(Some(LadderAux::boxed(aux)))?;
    Ok(())
}

/// `LLVQ_GQA_PPL`'s `ladder` form: WikiText-2 perplexity of the f16 cache and
/// of `LLVQ_KV_LADDER`'s arm, on the same windows, by the decode path only
/// ([`llvq_llm::model::Qwen3::decode_nll_modes`]): each window fed one true
/// token at a time through each arm's cache, the two in lockstep, both under
/// the kernel form on every decode row. The arm codes each position once, on
/// its way into the cache, from that position alone, so the decode path reads
/// the bytes a prefill would write. The windows are the sampling unit.
#[cfg_attr(not(all(target_os = "linux", feature = "cuda")), allow(dead_code))]
pub(super) fn run_ladder_ppl(
    model: &mut llvq_llm::model::Qwen3,
    ids: &[u32],
    spec: &GqaPplSpec,
    ladder: &llvq_llm::kvladder::Ladder,
) -> anyhow::Result<()> {
    use llvq_llm::kvq::KvMode;
    let (ctx, windows, first) = (spec.ctx, spec.windows, spec.first);
    let nwin = (ids.len() / ctx).saturating_sub(first).min(windows);
    if nwin < 2 || ctx < 2 {
        anyhow::bail!(
            "LLVQ_GQA_PPL ladder needs two windows of two tokens at least: {} tokens, ctx {ctx}, {windows} asked",
            ids.len()
        );
    }
    let span = &ids[first * ctx..(first + nwin) * ctx];
    let fingerprint = llvq_llm::eval::token_fingerprint(span);
    let arm = GqaArm::Kernel;
    let arms = [(arm.store(), arm.heads(), KvMode::F16), (arm.store(), arm.heads(), ladder.mode)];
    let cfg = model.config();
    let bytes = |m| {
        llvq_llm::kvq::bytes_per_token(cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim, m, model.dtype())
    };
    let (b0, b1) = (bytes(KvMode::F16), bytes(ladder.mode));
    println!(
        "KV ladder perplexity: wikitext2, {nwin} windows of {ctx} from window {first} (tokens {fingerprint:016x}), \
         cache {} against f16, both {} on every decode row from position {}, decode path in lockstep",
        ladder.label(),
        arm.name(),
        model.kernel_from()
    );
    println!(
        "  bytes a position (computed, this model's geometry): f16 {b0:.0}, {} {b1:.0}, {:.4} of f16",
        ladder.mode.name(),
        b1 / b0
    );
    let mut dec: [Vec<f64>; 2] = [Vec::new(), Vec::new()];
    let (mut agree, mut steps, mut max_d) = (0usize, 0usize, 0f32);
    let t0 = Instant::now();
    for w in 0..nwin {
        let win = &span[w * ctx..(w + 1) * ctx];
        let d = model.decode_nll_modes(win, arms, 0)?;
        for (v, nll) in dec.iter_mut().zip(d.nll) {
            v.push(nll / d.count as f64);
        }
        agree += d.top1_agree;
        steps += d.count;
        max_d = max_d.max(d.max_dlogit);
        println!(
            "  window {:>3}/{nwin}  decode nll {:.9} {:.9}  top-1 {}/{}  max |dlogit| {:.3e}  ({:.0} s)",
            w + 1,
            dec[0][w],
            dec[1][w],
            d.top1_agree,
            d.count,
            d.max_dlogit,
            t0.elapsed().as_secs_f64()
        );
    }
    arm.apply(model);
    let ppl = |v: &[f64]| (v.iter().sum::<f64>() / v.len() as f64).exp();
    let (m, lo, hi) = paired_ppl_pct(&dec[0], &dec[1]);
    println!(
        "\n  path      ppl f16               ppl {:<18}  {} / f16 − 1, paired 95 % t interval over {nwin} windows",
        ladder.mode.name(),
        ladder.mode.name()
    );
    println!("  decode    {:<20.4}  {:<22.4}  {m:+.4} % [{lo:+.4} ; {hi:+.4}]", ppl(&dec[0]), ppl(&dec[1]));
    println!(
        "  decode top-1 agreement {agree}/{steps} steps ({:.3} %), max |dlogit| {max_d:.3e}",
        100.0 * agree as f64 / steps as f64
    );
    Ok(())
}

/// `LLVQ_GQA_PPL`: WikiText-2 perplexity of the published attention and of
/// the grouped form, on the same windows, by two paths. The prefill path is
/// [`llvq_llm::model::Qwen3::window_nll`], the whole window in one forward.
/// The decode path is [`llvq_llm::model::Qwen3::decode_nll_paired`], the
/// window fed one true token at a time through each arm's cache, the two arms
/// in lockstep. The windows are the sampling unit: each difference is paired
/// window by window. The published arm's two paths must agree to rounding,
/// which checks the harness, not the forms.
#[cfg_attr(not(all(target_os = "linux", feature = "cuda")), allow(dead_code))]
pub(super) fn run_gqa_ppl(
    model: &mut llvq_llm::model::Qwen3,
    ids: &[u32],
    spec: &GqaPplSpec,
) -> anyhow::Result<()> {
    let (ctx, windows, first) = (spec.ctx, spec.windows, spec.first);
    let nwin = (ids.len() / ctx).saturating_sub(first).min(windows);
    if nwin < 2 || ctx < 2 {
        anyhow::bail!(
            "LLVQ_GQA_PPL needs two windows of two tokens at least: {} tokens, ctx {ctx}, {windows} asked",
            ids.len()
        );
    }
    let span = &ids[first * ctx..(first + nwin) * ctx];
    let fingerprint = llvq_llm::eval::token_fingerprint(span);
    let arms = match spec.kernel {
        false => [GqaArm::Published, GqaArm::Grouped],
        true => [GqaArm::Grouped, GqaArm::Kernel],
    };
    // The kernel takes no call of more than 8 rows, so its prefill path IS
    // the grouped arm's: checked bit for bit on the first window when that
    // window fits one prefill call, never scored for the interval.
    let prefill_windows = match (spec.kernel, ctx <= llvq_llm::model::MAX_PREFILL_ROWS) {
        (false, _) => nwin,
        (true, true) => 1,
        (true, false) => 0,
    };
    if spec.kernel {
        println!(
            "GQA perplexity: wikitext2, {nwin} windows of {ctx} from window {first} (tokens \
             {fingerprint:016x}), {} against {}, decode path in lockstep; prefill path on {} window(s), \
             checked bit for bit",
            arms[1].name(),
            arms[0].name(),
            prefill_windows
        );
        println!(
            "  the kernel form takes positions {} and later: positions 0..{} checked bit for bit \
             against the grouped arm, positions {}..{} scored apart ({} a window)",
            model.kernel_from(),
            model.kernel_from(),
            model.kernel_from(),
            ctx - 1,
            ctx - 1 - model.kernel_from()
        );
    } else {
        println!(
            "GQA perplexity: wikitext2, {nwin} windows of {ctx} (tokens {fingerprint:016x}), {} against {}, \
             prefill path then decode path in lockstep",
            arms[1].name(),
            arms[0].name()
        );
    }
    let mut pre: [Vec<f64>; 2] = [Vec::new(), Vec::new()];
    let mut dec: [Vec<f64>; 2] = [Vec::new(), Vec::new()];
    let mut tail: [Vec<f64>; 2] = [Vec::new(), Vec::new()];
    let (mut agree, mut steps, mut max_d) = (0usize, 0usize, 0f32);
    let (mut tail_agree, mut tail_steps) = (0usize, 0usize);
    // The kernel arm is the grouped form's below its threshold: those steps
    // are checked bit for bit and scored apart from the ones past it.
    let split = if spec.kernel { model.kernel_from() } else { 0 };
    let t0 = Instant::now();
    for w in 0..nwin {
        let win = &span[w * ctx..(w + 1) * ctx];
        if w < prefill_windows {
            for (k, arm) in arms.into_iter().enumerate() {
                arm.apply(model);
                let (n, c) = model.window_nll(win, &mut NoCapture)?;
                pre[k].push(n / c as f64);
            }
            if spec.kernel {
                if pre[0][w].to_bits() != pre[1][w].to_bits() {
                    anyhow::bail!(
                        "window {}: the kernel arm's prefill path is not the grouped arm's ({:.12} against \
                         {:.12}); it never reaches the kernel, so this is a wiring defect",
                        w + 1,
                        pre[1][w],
                        pre[0][w]
                    );
                }
                println!("  window {:>3}: prefill path bit for bit between the two arms", w + 1);
            }
        }
        let d = model.decode_nll_split(
            win,
            [(arms[0].store(), arms[0].heads()), (arms[1].store(), arms[1].heads())],
            split,
        )?;
        for (v, nll) in dec.iter_mut().zip(d.nll) {
            v.push(nll / d.count as f64);
        }
        for (v, nll) in tail.iter_mut().zip(d.tail_nll) {
            v.push(nll / d.tail_count as f64);
        }
        agree += d.top1_agree;
        steps += d.count;
        tail_agree += d.tail_top1_agree;
        tail_steps += d.tail_count;
        max_d = max_d.max(d.max_dlogit);
        if spec.kernel {
            if d.below_differ > 0 {
                anyhow::bail!(
                    "window {}: {} of the {split} steps below position {split} differ between the \
                     two arms; the kernel form is the grouped form's there, so this is a wiring defect",
                    w + 1,
                    d.below_differ
                );
            }
            println!(
                "  window {:>3}/{nwin}  {}  decode nll {:.9} {:.9}  \
                 from {split}: nll {:.9} {:.9}  top-1 {}/{}  max |dlogit| {:.3e}  ({:.0} s)",
                w + 1,
                match split {
                    0 => "every position on the kernel".to_string(),
                    _ => format!("positions < {split}: bit for bit"),
                },
                dec[0][w],
                dec[1][w],
                tail[0][w],
                tail[1][w],
                d.tail_top1_agree,
                d.tail_count,
                d.max_dlogit,
                t0.elapsed().as_secs_f64()
            );
        } else {
            println!(
                "  window {:>3}/{nwin}  prefill nll {:.9} {:.9}  decode nll {:.9} {:.9}  top-1 {}/{}  \
                 max |dlogit| {:.3e}  ({:.0} s)",
                w + 1,
                pre[0][w],
                pre[1][w],
                dec[0][w],
                dec[1][w],
                d.top1_agree,
                d.count,
                d.max_dlogit,
                t0.elapsed().as_secs_f64()
            );
        }
    }
    arms[0].apply(model);
    if spec.kernel {
        let ppl = |v: &[f64]| (v.iter().sum::<f64>() / v.len() as f64).exp();
        let (m, lo, hi) = paired_ppl_pct(&dec[0], &dec[1]);
        println!(
            "\n  path      ppl {:<16}  ppl {:<18}  kernel / grouped − 1, paired 95 % t interval over {nwin} windows",
            arms[0].name(),
            arms[1].name()
        );
        println!("  decode    {:<20.4}  {:<22.4}  {m:+.4} % [{lo:+.4} ; {hi:+.4}]", ppl(&dec[0]), ppl(&dec[1]));
        let (m, lo, hi) = paired_ppl_pct(&tail[0], &tail[1]);
        println!(
            "  from {split:<4} {:<20.4}  {:<22.4}  {m:+.4} % [{lo:+.4} ; {hi:+.4}]  (positions the kernel takes)",
            ppl(&tail[0]),
            ppl(&tail[1])
        );
        println!(
            "  decode top-1 agreement {agree}/{steps} steps ({:.3} %), max |dlogit| {max_d:.3e}",
            100.0 * agree as f64 / steps as f64
        );
        println!(
            "  from {split}: top-1 agreement {tail_agree}/{tail_steps} steps ({:.3} %); below {split}: \
             bit for bit in all {nwin} windows",
            100.0 * tail_agree as f64 / tail_steps as f64
        );
        return Ok(());
    }
    let ppl = |v: &[f64]| (v.iter().sum::<f64>() / v.len() as f64).exp();
    println!(
        "\n  path      ppl {:<16}  ppl {:<18}  grouped / published − 1, paired 95 % t interval over {nwin} windows",
        arms[0].name(),
        arms[1].name()
    );
    for (name, v) in [("prefill", &pre), ("decode", &dec)] {
        let (m, lo, hi) = paired_ppl_pct(&v[0], &v[1]);
        println!(
            "  {name:<8}  {:<20.4}  {:<22.4}  {m:+.4} % [{lo:+.4} ; {hi:+.4}]",
            ppl(&v[0]),
            ppl(&v[1])
        );
    }
    println!(
        "  harness: decode over prefill on the published arm, ppl ratio {:.5}",
        ppl(&dec[0]) / ppl(&pre[0])
    );
    println!(
        "  decode top-1 agreement {agree}/{steps} steps ({:.3} %), max |dlogit| {max_d:.3e}",
        100.0 * agree as f64 / steps as f64
    );
    Ok(())
}
