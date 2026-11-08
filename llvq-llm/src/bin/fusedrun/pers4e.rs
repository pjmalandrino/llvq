//! `LLVQ_PERS_4E`: the whole token in one launch, against the eager path.

use super::*;

#[cfg(all(target_os = "linux", feature = "cuda"))]
/// `LLVQ_PERS_4E` at one prompt length: the reference tokens of the eager
/// path, each 4E geometry's after the same prefill (identical or void), the
/// served door's beside, then 2 + 5 interleaved rounds of the decode, eager
/// and every geometry in each round, then one stamped launch a geometry.
fn run_pers_4e(
    model: &mut llvq_llm::model::Qwen3,
    rts: &mut Vec<llvq_llm::persist4e_cuda::Runtime4e>,
    prompt: &[u32],
    n_new: usize,
) -> anyhow::Result<()> {
    use llvq_llm::stats::median_range;
    const WARMUP: usize = 2;
    // The form 4E and the eager side run. From KERNEL_FROM on, the served
    // config itself (the kernel from position 1,280, operator T8): every
    // decode row of a prompt that long is past the threshold. Below it, the
    // served config runs candle's grouped arithmetic, which 4E does not
    // compute: both sides then run the kernel on every row, a measurement
    // form with no T8, as the first three readings did.
    let from = llvq_llm::kvq::KERNEL_FROM;
    let served_form = prompt.len() >= from;
    let eager_from = if served_form { from } else { 0 };
    for rt in rts.iter_mut() {
        rt.set_kernel_from_for_measurement(eager_from);
    }
    let kernel_form = |m: &mut llvq_llm::model::Qwen3| {
        GqaArm::Kernel.apply(m);
        m.set_kernel_from_for_measurement(eager_from);
    };
    // The served door: the served 4B config (kernel from 1,280).
    GqaArm::Kernel.apply(model);
    model.set_kernel_from_for_measurement(from);
    // In chunks of MAX_PREFILL_ROWS, the context sweep's way: an 8,192-token
    // prompt in one call is refused by the fused prefill (run 2 of the 4E
    // reading stopped there). At or under one chunk this is `generate`.
    let served = model.generate_timed(prompt, n_new, llvq_llm::model::MAX_PREFILL_ROWS)?.tokens;
    kernel_form(model);
    println!(
        "\n  context {}: attention {} on {}, {}",
        prompt.len(),
        model.attn_heads().label(),
        model.kv_store().label(),
        if served_form {
            format!("the served config: every decode row at or past position {from}, inside the T8")
        } else {
            format!("measurement form: every decode row on the kernel (no T8; the served config runs the grouped path below {from})")
        }
    );
    // One prefill for the context (chunked, `prefill_next`); every run truncates the caches back to
    // the prompt (a Grow store's truncate moves only its length) and writes
    // the positions after it again, eager or 4E.
    let mut caches = model.fresh_caches();
    let first = model.prefill_next(prompt, &mut caches)?;
    let eager = |m: &mut llvq_llm::model::Qwen3, caches: &mut Vec<llvq_llm::model::KvCache>| -> anyhow::Result<(Vec<u32>, f64)> {
        for c in caches.iter_mut() {
            c.truncate(prompt.len())?;
        }
        let t = Instant::now();
        let mut out = vec![first];
        let mut offset = prompt.len();
        while out.len() < n_new {
            let logits = m.step_logits(*out.last().expect("one token"), offset, caches)?;
            out.push(logits.to_dtype(candle_core::DType::F32)?.flatten_all()?.argmax(0)?.to_scalar::<u32>()?);
            offset += 1;
        }
        Ok((out, t.elapsed().as_secs_f64()))
    };
    let pers = |rt: &mut llvq_llm::persist4e_cuda::Runtime4e, caches: &mut Vec<llvq_llm::model::KvCache>| -> anyhow::Result<(Vec<u32>, f64)> {
        for c in caches.iter_mut() {
            c.truncate(prompt.len())?;
        }
        rt.sync()?;
        let t = Instant::now();
        let mut out = vec![first];
        out.extend(rt.decode(caches, prompt.len(), first, (n_new - 1) as u32, u32::MAX)?);
        Ok((out, t.elapsed().as_secs_f64()))
    };
    let (want, _) = eager(model, &mut caches)?;
    // v4 (served) and v6 are the references: a divergence of either voids the
    // run. A newer form that diverges leaves the run alone, its numbers void,
    // and the forms that kept the bits are still timed.
    let mut k = 0;
    while k < rts.len() {
        let (got, _) = pers(&mut rts[k], &mut caches)?;
        let label = rts[k].label();
        match first_divergence(&want, &got).map_err(|e| anyhow::anyhow!("{e}"))? {
            None => {
                println!("  4E {label}: tokens equal the eager path's, {} of {}", got.len(), want.len());
                k += 1;
            }
            Some(i) if rts[k].rows_form <= 4 => {
                anyhow::bail!("context {}: 4E {label} diverges from the eager path at token {i}; 4E owes its bits, the run is void", prompt.len());
            }
            Some(i) => {
                println!("  4E {label}: DIVERGES from the eager path at token {i}; this form's numbers are void and it leaves the run");
                rts.remove(k);
            }
        }
    }
    if rts.is_empty() {
        anyhow::bail!("context {}: no 4E form kept the eager path's tokens", prompt.len());
    }
    match first_divergence(&served, &want).map_err(|e| anyhow::anyhow!("{e}"))? {
        None => println!("  served door (kernel from {from}): the same {} tokens", served.len()),
        Some(i) if served_form => anyhow::bail!("context {}: the served door diverges from the eager served config at token {i}; the run is void", prompt.len()),
        Some(i) => println!("  served door (kernel from {from}): diverges at token {i} (the measurement form's T8 reading, not a gate)"),
    }
    let mut rate: Vec<Vec<f64>> = vec![Vec::new(); 1 + rts.len()];
    for round in 0..WARMUP + ROUNDS_TIMED {
        let (a, ta) = eager(model, &mut caches)?;
        if a != want {
            anyhow::bail!("round {round}: the eager decode moved its tokens; the run is void");
        }
        if round >= WARMUP {
            rate[0].push((n_new - 1) as f64 / ta);
        }
        for (k, rt) in rts.iter_mut().enumerate() {
            let (b, tb) = pers(rt, &mut caches)?;
            if b != want {
                anyhow::bail!("round {round}: 4E {} moved its tokens; the run is void", rt.label());
            }
            if round >= WARMUP {
                rate[1 + k].push((n_new - 1) as f64 / tb);
            }
        }
    }
    let eager_name = if served_form { "eager (served config)" } else { "eager (kernel form)" };
    let (m, lo, hi) = median_range(&rate[0]);
    println!("  {eager_name:<22} {m:7.1} tok/s [{lo:.1}-{hi:.1}]");
    for (k, rt) in rts.iter().enumerate() {
        let (m, lo, hi) = median_range(&rate[1 + k]);
        let name = format!("4E {}", rt.label());
        println!("  {name:<22} {m:7.1} tok/s [{lo:.1}-{hi:.1}]");
        let ratio: Vec<f64> = rate[1 + k].iter().zip(&rate[0]).map(|(a, b)| a / b).collect();
        let (m, lo, hi) = median_range(&ratio);
        println!("  {:<22} 4E / eager, round by round: x{m:.3} [{lo:.3}-{hi:.3}]", "");
        // against every earlier form, the same rounds
        for (j, rj) in rts.iter().enumerate().take(k) {
            let ratio: Vec<f64> = rate[1 + k].iter().zip(&rate[1 + j]).map(|(a, b)| a / b).collect();
            let (m, lo, hi) = median_range(&ratio);
            println!("  {:<22} 4E {} / 4E {}, round by round: x{m:.3} [{lo:.3}-{hi:.3}]", "", rt.label(), rj.label());
        }
    }
    // One more launch a geometry, on the phase-stamp symbol, after the timed
    // rounds and outside them: where the token's time goes, phase by phase
    // and CTA by CTA (pers.rs::p4e_phase_report). The same tokens, or its
    // stamps are void.
    for (k, rt) in rts.iter_mut().enumerate() {
        if !rt.can_stamp() {
            println!("  phase stamps: no stamp symbol for 4E {}", rt.label());
            continue;
        }
        for c in caches.iter_mut() {
            c.truncate(prompt.len())?;
        }
        let (made, stamps) = rt.decode_stamped(&mut caches, prompt.len(), first, (n_new - 1) as u32, u32::MAX)?;
        let mut got = vec![first];
        got.extend(&made);
        if got != want {
            anyhow::bail!("context {}: the stamped launch of 4E {} moved its tokens; its stamps are void", prompt.len(), rt.label());
        }
        let (layers, grid) = rt.stamp_shape();
        let r = llvq_cuda::pers::p4e_phase_report(&stamps, made.len() as u32, layers, grid).map_err(anyhow::Error::msg)?;
        println!("  phase stamps, 4E {}:", rt.label());
        for line in llvq_cuda::pers::p4e_report_lines(&r) {
            println!("{line}");
        }
        let (m4e, _, _) = median_range(&rate[1 + k]);
        println!("  stamped token {:.1} us against the timed 4E's {:.1} us (the stamps' own cost)", r.token.0 / 1e3, 1e6 / m4e);
    }
    for rt in rts.iter_mut() {
        rt.set_kernel_from_for_measurement(from);
    }
    Ok(())
}

/// The `LLVQ_PERS_4E` mode: one load, every bench geometry installed, then [`run_pers_4e`] per context.
pub(super) fn pers_4e_mode(v: String, path: String, device: candle_core::Device, dtype: candle_core::DType) -> anyhow::Result<()> {
    let n_new: usize = v.parse().map_err(|_| anyhow::anyhow!("LLVQ_PERS_4E={v:?}: a token count, 2 to 256"))?;
    if !(2..=256).contains(&n_new) {
        anyhow::bail!("LLVQ_PERS_4E={n_new}: 2 to 256 tokens a run");
    }
    runs_alone("LLVQ_PERS_4E", &["LLVQ_KERNEL_DIR"])?;
    let layout = llvq_llm::fused::FusedLayout::from_env().map_err(|e| anyhow::anyhow!("{e}"))?;
    if layout != llvq_llm::fused::FusedLayout::Tetra48 {
        anyhow::bail!("LLVQ_PERS_4E: layout {}, and 4E reads Tetra48 streams", layout.name());
    }
    let fuse = llvq_llm::fused::FuseMode::from_env().map_err(|e| anyhow::anyhow!("{e}"))?;
    if fuse != llvq_llm::fused::FuseMode::Off {
        anyhow::bail!("LLVQ_PERS_4E: LLVQ_FUSE={}, and the served 4B runs 0", fuse.name());
    }
    let ctx = device.as_cuda_device()?.cuda_stream().context().clone();
    if ctx.is_in_multi_stream_mode() {
        anyhow::bail!("LLVQ_PERS_4E: the context holds several streams, and turning \
                       cudarc's events off is only safe on one");
    }
    // Safe: one stream, checked above, orders every access by issue.
    unsafe {
        ctx.disable_event_tracking();
    }
    println!("cudarc event tracking: off (one stream, ordered by issue)");
    let t = Instant::now();
    let mut f = llvq_llm::fused_cuda::load_with(&path, &device, dtype, fuse)?;
    let (n_kv, groups, hd) = f.model.attn_geometry();
    let k = llvq_llm::attn::kernel_for(&device, n_kv, groups, hd).map_err(|e| anyhow::anyhow!("{e}"))?;
    f.model.set_attn_kernel(Some(k));
    // The bench forms at 512 threads, each its own module (the
    // 768-thread symbol spills under the card's driver, v2 and v3):
    // pers::P4E_BENCH_FORMS, v4, v6 and v9 (the served forms), then
    // v6s. Each owes the eager path's tokens.
    let mut rts = Vec::new();
    for (form, label) in llvq_cuda::pers::P4E_BENCH_FORMS {
        match llvq_llm::persist4e_cuda::install_rows(&f.model, &device, n_new as u32, Some(512), form) {
            Ok(rt) => {
                for line in &rt.lines {
                    println!("{line}");
                }
                rts.push(rt);
            }
            Err(e) => println!("4E {label} at 512 threads: not run: {e}"),
        }
    }
    // v6s: v6's module asking for v8's shared bytes, the same code and
    // bits under v8's L1/shared carve-out (plan-300 step 3's
    // diagnostic of v8's slower rows).
    match llvq_llm::persist4e_cuda::install_rows_as(&f.model, &device, n_new as u32, Some(512), 4, Some(6)) {
        Ok(rt) => {
            for line in &rt.lines {
                println!("{line}");
            }
            rts.push(rt);
        }
        Err(e) => println!("4E v6s at 512 threads: not run: {e}"),
    }
    if rts.is_empty() {
        anyhow::bail!("LLVQ_PERS_4E: no 4E geometry loads on this card");
    }
    if ctx.is_in_multi_stream_mode() {
        anyhow::bail!("LLVQ_PERS_4E: a second stream appeared at the load, with events off");
    }
    println!("loaded and checked in {:.1} s", t.elapsed().as_secs_f64());
    let base = f.tokenizer.encode(PROMPT, false).map_err(|e| anyhow::anyhow!("{e}"))?.get_ids().to_vec();
    for len in [base.len(), 8192] {
        let prompt: Vec<u32> = base.iter().copied().cycle().take(len).collect();
        run_pers_4e(&mut f.model, &mut rts, &prompt, n_new)?;
    }
    if ctx.is_in_multi_stream_mode() {
        anyhow::bail!("LLVQ_PERS_4E: a second stream appeared during the run: the numbers above are void");
    }
    Ok(())
}
