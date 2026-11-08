//! `LLVQ_GQA_PPL` and `LLVQ_GQA_AB`: the grouped attention forms against the published one.

use super::*;

/// The value of `LLVQ_GQA_PPL`: `<ctx>,<windows>` scores the grouped form
/// against the published one from window 0, as every control so far did;
/// `<ctx>,<windows>,<first>,kernel` scores the kernel form against the grouped
/// one (the served form) from window index `<first>`, so that a T8 control
/// takes windows no earlier control scored.
#[cfg_attr(not(all(target_os = "linux", feature = "cuda")), allow(dead_code))]
#[derive(Debug, PartialEq, Eq)]
struct GqaPplSpec {
    ctx: usize,
    windows: usize,
    first: usize,
    kernel: bool,
    /// `kernel0`: MEASUREMENT FORM, every decode row on the kernel from
    /// position 0, the T8 extension below `KERNEL_FROM` that serving 4E at a
    /// short prompt would need. Implies `kernel`.
    every_row: bool,
}

#[cfg_attr(not(all(target_os = "linux", feature = "cuda")), allow(dead_code))]
fn parse_gqa_ppl(v: &str) -> Result<GqaPplSpec, String> {
    let parts: Vec<&str> = v.split(',').map(str::trim).collect();
    let (nums, kernel, every_row) = match parts.as_slice() {
        [c, w] => (vec![*c, *w], false, false),
        [c, w, f, "kernel"] => (vec![*c, *w, *f], true, false),
        [c, w, f, "kernel0"] => (vec![*c, *w, *f], true, true),
        _ => {
            return Err(format!(
                "LLVQ_GQA_PPL={v:?}: expected <ctx>,<windows> (grouped against published, from \
                 window 0), <ctx>,<windows>,<first>,kernel (kernel against grouped) or \
                 <ctx>,<windows>,<first>,kernel0 (kernel on every row against grouped)"
            ))
        }
    };
    let n = nums
        .iter()
        .map(|s| s.parse::<usize>().map_err(|e| format!("LLVQ_GQA_PPL={v:?}: {s:?}: {e}")))
        .collect::<Result<Vec<_>, _>>()?;
    let spec = GqaPplSpec { ctx: n[0], windows: n[1], first: n.get(2).copied().unwrap_or(0), kernel, every_row };
    if spec.ctx < 2 || spec.windows < 2 {
        return Err(format!("LLVQ_GQA_PPL={v:?}: two windows of two tokens at least"));
    }
    // The published-against-grouped mode scores the prefill path of every
    // window in one forward, bounded by the fused prefill's rows. The kernel
    // mode scores that path on its first window only, and only when it fits.
    if !spec.kernel && spec.ctx > llvq_llm::model::MAX_PREFILL_ROWS {
        return Err(format!(
            "LLVQ_GQA_PPL: a window of {} exceeds the fused prefill's {} rows",
            spec.ctx,
            llvq_llm::model::MAX_PREFILL_ROWS
        ));
    }
    // The kernel form sends a position to its kernel from KERNEL_FROM on: a
    // window must reach past it, or the mode scores the grouped form twice.
    let from = llvq_llm::kvq::KERNEL_FROM;
    if spec.kernel && !spec.every_row && spec.ctx <= from + 1 {
        return Err(format!(
            "LLVQ_GQA_PPL: a window of {} never reaches the kernel, which takes positions {from} \
             and later; ask {} tokens at least",
            spec.ctx,
            from + 2
        ));
    }
    Ok(spec)
}

/// "bit for bit" when two logit vectors agree in every bit, else their
/// largest absolute difference.
#[cfg_attr(not(all(target_os = "linux", feature = "cuda")), allow(dead_code))]
fn logit_agreement(a: &[f32], b: &[f32]) -> String {
    if a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()) {
        return "bit for bit".to_string();
    }
    let d = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
    format!("max |dlogit| {d:.3e}")
}

/// What one arm's prefill leaves for its decode rounds.
#[cfg_attr(not(all(target_os = "linux", feature = "cuda")), allow(dead_code))]
struct Prefilled {
    caches: Vec<llvq_llm::model::KvCache>,
    first: u32,
    prompt_logits: Vec<f32>,
    step_logits: Vec<f32>,
}

/// `LLVQ_GQA_AB`: for each prompt length, one prefill per arm, outside every
/// clock, then one discarded and [`ROUNDS_TIMED`] greedy decodes of `n_new`
/// tokens from it, every arm in every round, each cache truncated back to the
/// prompt before each decode (`a_truncated_cache_decodes_like_a_fresh_one`).
/// The decode speed is `(n_new − 1) / t`, the reading of `LLVQ_CONTEXTS`.
///
/// Two outcomes are printed, never gated: whether each unexpanded arm's logits
/// are the published arm's bit for bit (the prompt's last position, then the
/// first decode step), and whether its text is. Within one arm, a text that
/// changes between rounds stops the run: that is a speed of two things.
#[cfg_attr(not(all(target_os = "linux", feature = "cuda")), allow(dead_code))]
fn run_gqa_ab(
    model: &mut llvq_llm::model::Qwen3,
    seed: &[u32],
    lengths: &[usize],
    n_new: usize,
) -> anyhow::Result<()> {
    if n_new < 2 || seed.is_empty() {
        anyhow::bail!(
            "LLVQ_GQA_AB needs a seed prompt and at least 2 new tokens: the decode is timed \
             after the first one ({n_new} asked)"
        );
    }
    let chunk = llvq_llm::model::MAX_PREFILL_ROWS;
    let reference = GqaArm::Published.name();
    println!(
        "GQA A/B: {} arms, {n_new} new tokens after each prompt, the prompt in chunks of {chunk}; \
         one prefill an arm, then 1 discarded + {ROUNDS_TIMED} decodes from it, every arm in \
         every round",
        GqaArm::ALL.len()
    );
    for &n in lengths {
        let ids: Vec<u32> = (0..n).map(|i| seed[i % seed.len()]).collect();
        let mut arms = Vec::with_capacity(GqaArm::ALL.len());
        for arm in GqaArm::ALL {
            arm.apply(model);
            let mut caches = model.fresh_caches();
            let lp = model.prefill_chunked(&ids, chunk, &mut caches)?;
            let first = llvq_llm::model::argmax_last(&lp)?;
            let ls = model.step_logits(first, n, &mut caches)?;
            arms.push(Prefilled {
                caches,
                first,
                prompt_logits: lp.flatten_all()?.to_vec1::<f32>()?,
                step_logits: ls.flatten_all()?.to_vec1::<f32>()?,
            });
        }
        println!("\nprompt {n}");
        for (k, arm) in GqaArm::ALL.into_iter().enumerate().skip(1) {
            println!(
                "  {:<18}  logits against {reference}: prompt {}, first decode step {}",
                arm.name(),
                logit_agreement(&arms[0].prompt_logits, &arms[k].prompt_logits),
                logit_agreement(&arms[0].step_logits, &arms[k].step_logits)
            );
        }
        let mut rates: Vec<Vec<f64>> = vec![Vec::new(); GqaArm::ALL.len()];
        let mut texts: Vec<Option<Vec<u32>>> = vec![None; GqaArm::ALL.len()];
        for round in 0..=ROUNDS_TIMED {
            for (k, arm) in GqaArm::ALL.into_iter().enumerate() {
                arm.apply(model);
                let p = &mut arms[k];
                for c in p.caches.iter_mut() {
                    c.truncate(n)?;
                }
                let t = Instant::now();
                let out = model.decode_greedy(p.first, n, n_new, &mut p.caches)?;
                let rate = (n_new - 1) as f64 / t.elapsed().as_secs_f64();
                match &texts[k] {
                    None => texts[k] = Some(out),
                    Some(r) => {
                        if let Some(i) = first_divergence(r, &out).map_err(|e| anyhow::anyhow!("{e}"))? {
                            anyhow::bail!(
                                "prompt {n}, arm {}, round {round}: divergence at token {i}. \
                                 An arm must repeat its own text, or its speed is not the speed \
                                 of one thing",
                                arm.name()
                            );
                        }
                    }
                }
                if round > 0 {
                    rates[k].push(rate);
                }
            }
        }
        println!("  arm                 decode tok/s [range]     arm / {reference}, round by round    text");
        let base = texts[0].clone().unwrap_or_default();
        for (k, arm) in GqaArm::ALL.into_iter().enumerate() {
            let (m, lo, hi) = rate_stats(&rates[k]);
            if k == 0 {
                println!("  {:<18}  {m:7.1} [{lo:.1}–{hi:.1}]", arm.name());
                continue;
            }
            let ratios: Vec<f64> = rates[k].iter().zip(&rates[0]).map(|(a, b)| a / b).collect();
            let (mr, lr, hr) = rate_stats(&ratios);
            let text = match first_divergence(&base, texts[k].as_deref().unwrap_or_default())
                .map_err(|e| anyhow::anyhow!("{e}"))?
            {
                None => format!("{} tokens identical", base.len()),
                Some(i) => format!("differs from token {i}"),
            };
            println!(
                "  {:<18}  {m:7.1} [{lo:.1}–{hi:.1}]    {mr:.4} [{lr:.4}–{hr:.4}]          {text}",
                arm.name()
            );
        }
    }
    GqaArm::Published.apply(model);
    Ok(())
}

/// Two-sided 95 % Student quantile for `df` degrees of freedom, from the
/// table for 1 to 30 and the normal value beyond.
#[cfg_attr(not(all(target_os = "linux", feature = "cuda")), allow(dead_code))]
fn t95(df: usize) -> f64 {
    const T: [f64; 30] = [
        12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160,
        2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056,
        2.052, 2.048, 2.045, 2.042,
    ];
    match df {
        0 => f64::NAN,
        1..=30 => T[df - 1],
        _ => 1.96,
    }
}

/// Per-window differences of mean NLL, `b − a`, as `ppl_b / ppl_a − 1` in %:
/// the median-free point `exp(mean) − 1` and its paired 95 % t interval.
/// Every window scores the same number of tokens, so `exp(mean d)` is the
/// ratio of the two perplexities exactly.
#[cfg_attr(not(all(target_os = "linux", feature = "cuda")), allow(dead_code))]
fn paired_ppl_pct(a: &[f64], b: &[f64]) -> (f64, f64, f64) {
    let d: Vec<f64> = a.iter().zip(b).map(|(x, y)| y - x).collect();
    let n = d.len() as f64;
    let m = d.iter().sum::<f64>() / n;
    let sd = (d.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
    let h = t95(d.len().saturating_sub(1)) * sd / n.sqrt();
    let pct = |x: f64| x.exp_m1() * 100.0;
    (pct(m), pct(m - h), pct(m + h))
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
fn run_gqa_ppl(
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

#[cfg(all(target_os = "linux", feature = "cuda"))]
/// The `LLVQ_GQA_PPL` mode: one load, then [`run_gqa_ppl`].
pub(super) fn gqa_ppl_mode(v: String, path: String, device: candle_core::Device, dtype: candle_core::DType) -> anyhow::Result<()> {
    let spec = parse_gqa_ppl(&v).map_err(|e| anyhow::anyhow!("{e}"))?;
    runs_alone("LLVQ_GQA_PPL", &[])?;
    let fuse = llvq_llm::fused::FuseMode::from_env().map_err(|e| anyhow::anyhow!("{e}"))?;
    let t = Instant::now();
    let mut f = llvq_llm::fused_cuda::load_with(&path, &device, dtype, fuse)?;
    println!(
        "loaded in {:.1} s. GQA perplexity: 2 arms, one model",
        t.elapsed().as_secs_f64()
    );
    println!(
        "LLVQ_ROT_SHARE={}, {} rot_launches/token · LLVQ_FUSE={}, {} matvec_launches/token",
        f.rot_share.name(), f.rot_launches, f.fuse.name(), f.matvec_launches
    );
    let ctx = device.as_cuda_device()?.cuda_stream().context().clone();
    if ctx.is_in_multi_stream_mode() {
        anyhow::bail!("LLVQ_GQA_PPL: the context holds several streams, and turning \
                       cudarc's events off is only safe on one");
    }
    // Safe: one stream, checked above, orders every access by issue.
    unsafe {
        ctx.disable_event_tracking();
    }
    println!("cudarc event tracking: off (one stream, ordered by issue)");
    let text = llvq_llm::corpus::wikitext2_test()?;
    let ids = f
        .tokenizer
        .encode(text.as_str(), false)
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .get_ids()
        .to_vec();
    if spec.kernel {
        // Built, reported and checked against the CPU replay on the
        // card before the first window, as the served door does.
        let (n_kv, groups, hd) = f.model.attn_geometry();
        let k = llvq_llm::attn::kernel_for(&device, n_kv, groups, hd).map_err(|e| anyhow::anyhow!("{e}"))?;
        f.model.set_attn_kernel(Some(k));
    }
    if spec.every_row {
        // The T8 extension's measurement form: the kernel takes every
        // decode row, so no position is checked bit for bit and every
        // one is scored. The served threshold stays the constant.
        f.model.set_kernel_from_for_measurement(0);
        println!(
            "measurement form: every decode row on the kernel, from position 0 (the served \
             config runs the grouped path below {})",
            llvq_llm::kvq::KERNEL_FROM
        );
    }
    run_gqa_ppl(&mut f.model, &ids, &spec)?;
    Ok(())
}

#[cfg(all(target_os = "linux", feature = "cuda"))]
/// The `LLVQ_GQA_AB` mode: one load, then [`run_gqa_ab`].
pub(super) fn gqa_ab_mode(v: String, path: String, device: candle_core::Device, dtype: candle_core::DType, n_new: usize) -> anyhow::Result<()> {
    let lengths = parse_lengths("LLVQ_GQA_AB", &v).map_err(|e| anyhow::anyhow!("{e}"))?;
    runs_alone("LLVQ_GQA_AB", &[])?;
    let fuse = llvq_llm::fused::FuseMode::from_env().map_err(|e| anyhow::anyhow!("{e}"))?;
    let t = Instant::now();
    let mut f = llvq_llm::fused_cuda::load_with(&path, &device, dtype, fuse)?;
    println!(
        "loaded in {:.1} s. GQA A/B: {} arms, one model, interleaved rounds",
        t.elapsed().as_secs_f64(),
        GqaArm::ALL.len()
    );
    println!(
        "LLVQ_ROT_SHARE={}, {} rot_launches/token · LLVQ_FUSE={}, {} matvec_launches/token",
        f.rot_share.name(), f.rot_launches, f.fuse.name(), f.matvec_launches
    );
    let ctx = device.as_cuda_device()?.cuda_stream().context().clone();
    if ctx.is_in_multi_stream_mode() {
        anyhow::bail!("LLVQ_GQA_AB: the context holds several streams, and turning \
                       cudarc's events off is only safe on one");
    }
    // Safe: one stream, checked above, orders every access by issue.
    unsafe {
        ctx.disable_event_tracking();
    }
    println!("cudarc event tracking: off (one stream, ordered by issue)");
    let seed = f
        .tokenizer
        .encode(PROMPT, false)
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .get_ids()
        .to_vec();
    run_gqa_ab(&mut f.model, &seed, &lengths, n_new)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{paired_ppl_pct, parse_gqa_ppl, t95, GqaArm, GqaPplSpec};

    /// The two forms of `LLVQ_GQA_PPL`, and what each refuses.
    #[test]
    fn gqa_ppl_spec() {
        assert_eq!(
            parse_gqa_ppl("4096,12").unwrap(),
            GqaPplSpec { ctx: 4096, windows: 12, first: 0, kernel: false, every_row: false }
        );
        assert_eq!(
            parse_gqa_ppl("8192,6,6,kernel").unwrap(),
            GqaPplSpec { ctx: 8192, windows: 6, first: 6, kernel: true, every_row: false }
        );
        assert_eq!(
            parse_gqa_ppl(" 4096 , 12 , 60 , kernel ").unwrap(),
            GqaPplSpec { ctx: 4096, windows: 12, first: 60, kernel: true, every_row: false }
        );
        // Above one prefill call: refused for the first form only.
        assert!(parse_gqa_ppl("8192,6").unwrap_err().contains("4096 rows"));
        // The kernel mode needs a window that reaches past the threshold:
        // position KERNEL_FROM must be an input with a target after it.
        let from = llvq_llm::kvq::KERNEL_FROM;
        let at = |ctx: usize| parse_gqa_ppl(&format!("{ctx},12,0,kernel"));
        assert!(at(from + 1).unwrap_err().contains("never reaches the kernel"));
        assert_eq!(at(from + 2).unwrap().ctx, from + 2);
        assert!(parse_gqa_ppl("1024,12").is_ok(), "the first form has no threshold");
        // kernel0: every row on the kernel, so a short window is the point.
        assert_eq!(
            parse_gqa_ppl("1280,24,40,kernel0").unwrap(),
            GqaPplSpec { ctx: 1280, windows: 24, first: 40, kernel: true, every_row: true }
        );
        assert!(parse_gqa_ppl("1280,24,40,kernel").unwrap_err().contains("never reaches the kernel"));
        assert!(parse_gqa_ppl("1280,24,kernel0").is_err(), "kernel0 needs its first window");
        for bad in ["4096", "4096,12,60", "4096,12,60,grouped", "4096,1", "1,12", "4096,x", "4096,12,60,kernel,1"] {
            assert!(parse_gqa_ppl(bad).is_err(), "{bad} must be refused");
        }
        // The kernel arm reads the Grow store, and is not one of LLVQ_GQA_AB's arms.
        assert_eq!(GqaArm::Kernel.store(), llvq_llm::kvq::KvStore::Grow(llvq_llm::kvq::GROW_CHUNK));
        assert_eq!(GqaArm::Kernel.name(), "grow(256)+kernel");
        assert!(!GqaArm::ALL.contains(&GqaArm::Kernel));
    }

    /// The paired perplexity interval: zero for identical arms, the exact
    /// ratio for a constant shift, and the t(1) half-width worked by hand for
    /// two windows that disagree.
    #[test]
    fn paired_ppl_interval() {
        let a = [2.0, 2.1, 2.2, 2.3];
        assert_eq!(paired_ppl_pct(&a, &a), (0.0, 0.0, 0.0));
        let b: Vec<f64> = a.iter().map(|x| x + 0.01).collect();
        let (m, lo, hi) = paired_ppl_pct(&a, &b);
        let want = 0.01f64.exp_m1() * 100.0;
        assert!((m - want).abs() < 1e-9 && (lo - want).abs() < 1e-9 && (hi - want).abs() < 1e-9);
        // d = [0, 0.02]: mean 0.01, sd 0.02 / sqrt(2), half-width 12.706 * 0.01 = 0.12706.
        let (m, lo, hi) = paired_ppl_pct(&[1.0, 1.0], &[1.0, 1.02]);
        assert!((m - want).abs() < 1e-9, "{m}");
        assert!((lo - (0.01f64 - 0.12706).exp_m1() * 100.0).abs() < 1e-6, "{lo}");
        assert!((hi - (0.01f64 + 0.12706).exp_m1() * 100.0).abs() < 1e-6, "{hi}");
        assert_eq!((t95(11), t95(30), t95(31)), (2.201, 2.042, 1.96));
        assert!(t95(0).is_nan());
    }
}
