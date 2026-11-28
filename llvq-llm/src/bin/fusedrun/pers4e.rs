//! The persistent decode against the eager path: the served door under a 4E
//! config, and `LLVQ_PERS_4E`. One file because both exist only on Linux with
//! `cuda`, the only build that has a 4E runtime to time.

use std::time::Instant;

use super::common::{first_divergence, GqaArm, ROUNDS_TIMED};

#[cfg(all(target_os = "linux", feature = "cuda"))]
/// The served door under a config that names the persistent decode. The eager
/// path under the same config (the kernel form from position 0) makes the
/// reference tokens; 4E must make the same ones in every round, or the run is
/// void. Then 1 discarded + [`ROUNDS_TIMED`] rounds, eager and 4E in each, both
/// timed as the door times a generate (prompt and decode), and the decode
/// alone beside it. The token ids are printed so a second config's door can
/// be compared token for token after the job.
pub(super) fn served_door_4e(
    f: &llvq_llm::fused::FusedSealed,
    dec: &mut llvq_llm::served_decode::Decoder,
    ids: &[u32],
    n_new: usize,
    load_s: f64,
) -> anyhow::Result<()> {
    use llvq_llm::stats::median_range;
    anyhow::ensure!(n_new >= 2, "the served 4E door times the decode after the first token: {n_new} asked");
    let chunk = llvq_llm::model::MAX_PREFILL_ROWS;
    let eager_want = f.model.generate_timed(ids, n_new, chunk)?.tokens;
    // The reference 4E owes bit for bit: the eager decode after the same
    // prefill. Under the eager prefill that is the eager path itself; under a
    // chain config it is the chain prefill, then the eager decode on the
    // caches it left (`Decoder::generate_eager_decode`).
    let want = match dec.has_chain() {
        true => dec.generate_eager_decode(&f.model, ids, n_new, chunk)?,
        false => eager_want.clone(),
    };
    let (mut door, mut dec_rate, mut door_eager, mut dec_eager) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for round in 0..=ROUNDS_TIMED {
        let t = Instant::now();
        let e = f.model.generate_timed(ids, n_new, chunk)?;
        let te = t.elapsed().as_secs_f64();
        if e.tokens != eager_want {
            anyhow::bail!("round {round}: the eager path moved its tokens; the run is void");
        }
        let t = Instant::now();
        let g = dec.generate_timed(&f.model, ids, n_new, chunk)?;
        let tg = t.elapsed().as_secs_f64();
        if let Some(i) = first_divergence(&want, &g.tokens).map_err(|e| anyhow::anyhow!("{e}"))? {
            match dec.has_chain() {
                true => anyhow::bail!(
                    "round {round}: 4E diverges from the eager decode after the chain prefill at token {i}; the run is void"
                ),
                false => anyhow::bail!("round {round}: 4E diverges from the eager path under this config at token {i}; the run is void"),
            }
        }
        if round > 0 {
            door_eager.push(n_new as f64 / te);
            dec_eager.push((n_new - 1) as f64 / e.decode_s);
            door.push(n_new as f64 / tg);
            dec_rate.push((n_new - 1) as f64 / g.decode_s);
        }
    }
    if dec.has_chain() {
        // Not a gate: the chain prefill is not bit for bit with the eager one.
        let same = match first_divergence(&eager_want, &want).map_err(|e| anyhow::anyhow!("{e}"))? {
            None => format!("all {} identical", want.len()),
            Some(i) => format!("first divergence at token {i}"),
        };
        println!(
            "served door (prefill chain, decode persistent): 4E tokens equal the eager decode's after the chain prefill, \
             {} of {}; against the eager door's tokens: {same}",
            want.len(),
            want.len()
        );
    } else {
        println!("served door (decode persistent): 4E tokens equal the eager path's under this config, {} of {}", want.len(), want.len());
    }
    let bytes = f.runtime_bytes + f.carried_bytes;
    let (m, lo, hi) = median_range(&door);
    println!(
        "served : loaded in {load_s:6.1} s, {m:6.1} tok/s [{lo:.1}–{hi:.1}, {ROUNDS_TIMED} rounds], {:.2} GB on the card          (4E, prompt and decode, as the door times a generate)",
        bytes as f64 / 1e9
    );
    let (m, lo, hi) = median_range(&dec_rate);
    println!("  4E decode alone          {m:7.1} tok/s [{lo:.1}-{hi:.1}]");
    let (m, lo, hi) = median_range(&door_eager);
    println!("  eager, same config, door {m:7.1} tok/s [{lo:.1}-{hi:.1}]");
    let (m, lo, hi) = median_range(&dec_eager);
    println!("  eager decode alone       {m:7.1} tok/s [{lo:.1}-{hi:.1}]");
    let ratio: Vec<f64> = door.iter().zip(&door_eager).map(|(a, b)| a / b).collect();
    let (m, lo, hi) = median_range(&ratio);
    println!("  4E / eager, door, round by round: x{m:.3} [{lo:.3}-{hi:.3}]");
    let s: Vec<String> = want.iter().map(u32::to_string).collect();
    println!("  token ids: {:016x} {}", llvq_llm::eval::token_fingerprint(&want), s.join(","));
    println!("  {}", f.tokenizer.decode(&want, true).unwrap_or_default());
    Ok(())
}

#[cfg(all(target_os = "linux", feature = "cuda"))]
/// `LLVQ_PERS_4E` at one prompt length: the reference tokens of the eager
/// path, each 4E geometry's after the same prefill (identical or void), the
/// served door's beside, then 2 + 5 interleaved rounds of the decode, eager
/// and every geometry in each round, then one stamped launch a geometry.
pub(super) fn run_pers_4e(
    model: &mut llvq_llm::model::Qwen3,
    rts: &mut Vec<llvq_llm::persist4e_cuda::Runtime4e>,
    prompt: &[u32],
    n_new: usize,
    again: bool,
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
    // Whether a runtime owes the eager path's bits: every form and E1 arm but
    // those that owe a bound on rows (pers::p4e_e1_owes_bits, the research
    // phase's flat body), whose tokens are compared and reported, never
    // required. `own[k]`: the tokens each runtime is held to in the rounds.
    let owes_bits = |rt: &llvq_llm::persist4e_cuda::Runtime4e| rt.e1_arm.is_none_or(llvq_cuda::pers::p4e_e1_owes_bits);
    // v4 (served) and v6 are the references: a divergence of either voids the
    // run. A newer form that diverges leaves the run alone, its numbers void,
    // and the forms that kept the bits are still timed. Under the F1 control
    // the first runtime is the reference of every ratio: its divergence voids
    // the run too.
    let mut own: Vec<Vec<u32>> = Vec::new();
    let mut k = 0;
    while k < rts.len() {
        let (got, _) = pers(&mut rts[k], &mut caches)?;
        let label = rts[k].label();
        match first_divergence(&want, &got).map_err(|e| anyhow::anyhow!("{e}"))? {
            None => {
                println!("  4E {label}: tokens equal the eager path's, {} of {}", got.len(), want.len());
                own.push(got);
                k += 1;
            }
            Some(i) if !owes_bits(&rts[k]) => {
                let eq = got.iter().zip(&want).filter(|(a, b)| a == b).count();
                println!(
                    "  4E {label}: owes {}: {eq} of {} tokens equal the eager path's, the first divergence at token {i}",
                    rts[k].e1_arm.map_or("the bits", llvq_cuda::pers::p4e_e1_debt),
                    want.len()
                );
                own.push(got);
                k += 1;
            }
            Some(i) if rts[k].rows_form <= 4 || (again && k == 0) => {
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
    // The bound arms against the reference (the first runtime), token by token.
    for (k, rt) in rts.iter().enumerate().skip(1).filter(|(_, r)| !owes_bits(r)) {
        let eq = own[k].iter().zip(&own[0]).filter(|(a, b)| a == b).count();
        let first = own[k].iter().zip(&own[0]).position(|(a, b)| a != b);
        println!(
            "  4E {} against 4E {}: {eq} of {} tokens equal{}",
            rt.label(),
            rts[0].label(),
            own[0].len(),
            first.map_or(String::new(), |i| format!(", the first divergence at token {i}"))
        );
    }
    // v9k8: its int8 caches after its first launch of this context
    for rt in rts.iter() {
        if let Some(l) = rt.kv8_report() {
            println!("  {l}");
        }
    }
    match first_divergence(&served, &want).map_err(|e| anyhow::anyhow!("{e}"))? {
        None => println!("  served door (kernel from {from}): the same {} tokens", served.len()),
        Some(i) if served_form => anyhow::bail!("context {}: the served door diverges from the eager served config at token {i}; the run is void", prompt.len()),
        Some(i) => println!("  served door (kernel from {from}): diverges at token {i} (the measurement form's T8 reading, not a gate)"),
    }
    // rate[0] eager, rate[1 + k] runtime k, and under the F1 control rate[1 +
    // rts.len()] the first runtime timed again. The order of a round: eager
    // then every runtime; under the control, every runtime, the first one
    // again, then eager, so the reference is the first and the last 4E timed.
    let n_rt = rts.len();
    let mut rate: Vec<Vec<f64>> = vec![Vec::new(); 1 + n_rt + usize::from(again)];
    for round in 0..WARMUP + ROUNDS_TIMED {
        let time_eager = |rate: &mut [Vec<f64>], model: &mut llvq_llm::model::Qwen3, caches: &mut Vec<llvq_llm::model::KvCache>| -> anyhow::Result<()> {
            let (a, ta) = eager(model, caches)?;
            if a != want {
                anyhow::bail!("round {round}: the eager decode moved its tokens; the run is void");
            }
            if round >= WARMUP {
                rate[0].push((n_new - 1) as f64 / ta);
            }
            Ok(())
        };
        if !again {
            time_eager(&mut rate, model, &mut caches)?;
        }
        let order: Vec<(usize, usize)> = (0..n_rt).map(|k| (k, 1 + k)).chain(again.then_some((0, 1 + n_rt))).collect();
        for (k, slot) in order {
            let (b, tb) = pers(&mut rts[k], &mut caches)?;
            if b != own[k] {
                anyhow::bail!("round {round}: 4E {} moved its tokens; the run is void", rts[k].label());
            }
            if round >= WARMUP {
                rate[slot].push((n_new - 1) as f64 / tb);
            }
        }
        if again {
            time_eager(&mut rate, model, &mut caches)?;
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
    // Rule F1 to F2 (docs/mesures/recherche-debit-r3-espace-2026-11-12.txt), on
    // tok/s: F1, the reference timed last over timed first, round by round, its
    // median inside [0.98, 1.02], its range the floor; F2, a runtime is FASTER
    // only if its whole range over the reference lies above the floor's upper
    // end, SLOWER only if under its lower end.
    if again {
        let (m0, lo0, hi0) = median_range(&rate[1 + n_rt]);
        println!("  {:<22} {m0:7.1} tok/s [{lo0:.1}-{hi0:.1}] (the F1 control, timed last)", format!("4E {} again", rts[0].label()));
        let f1: Vec<f64> = rate[1 + n_rt].iter().zip(&rate[1]).map(|(a, b)| a / b).collect();
        let (fm, flo, fhi) = median_range(&f1);
        let pass = (0.98..=1.02).contains(&fm);
        println!(
            "  F1 context {}: 4E {} last / first, round by round: x{fm:.4} [{flo:.4}-{fhi:.4}]: {}; the floor is [{flo:.4}, {fhi:.4}]",
            prompt.len(),
            rts[0].label(),
            if pass { "PASS" } else { "FAIL, every SCORE below describes" }
        );
        for (k, rt) in rts.iter().enumerate().skip(1) {
            let r: Vec<f64> = rate[1 + k].iter().zip(&rate[1]).map(|(a, b)| a / b).collect();
            let (m, lo, hi) = median_range(&r);
            let verdict = if lo > fhi {
                "FASTER"
            } else if hi < flo {
                "SLOWER"
            } else {
                "NOT SEPARATED"
            };
            println!(
                "  SCORE context {} {}: x{m:.4} [{lo:.4}-{hi:.4}] over 4E {}, round by round; owes {}; {verdict}{}",
                prompt.len(),
                rt.e1_arm.unwrap_or("?"),
                rts[0].label(),
                rt.e1_arm.map_or("the bits", llvq_cuda::pers::p4e_e1_debt),
                if pass { "" } else { " (F1 failed: a description)" }
            );
        }
    }
    // One more launch a geometry, on the phase-stamp symbol, after the timed
    // rounds and outside them: where the token's time goes, phase by phase
    // and CTA by CTA (pers/report.rs::p4e_phase_report). The same tokens, or its
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
        if got != own[k] {
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
    // E1: one launch a runtime on its logits symbol, outside the timed
    // rounds, n_new tokens from the first: the runtime's own tokens (the
    // first n_new - 1), and the head's binary16 logits of every position,
    // compared with the first runtime's: bit for bit; for an arm that owes a
    // bound, on the positions whose inputs are the reference's (up to its
    // first token that differs), the values that differ and the largest
    // |difference|, read as binary16.
    let mut reference: Option<(String, Vec<u16>, Vec<u32>)> = None;
    for (k, rt) in rts.iter_mut().enumerate().filter(|(_, r)| r.can_logits()) {
        for c in caches.iter_mut() {
            c.truncate(prompt.len())?;
        }
        let (made, logits) = rt.decode_logits(&mut caches, prompt.len(), first, n_new as u32, u32::MAX)?;
        let label = rt.label();
        if made.len() != n_new || made[..n_new - 1] != own[k][1..] {
            anyhow::bail!("context {}: the logits launch of 4E {label} moved its tokens; its logits are void", prompt.len());
        }
        let bits: Vec<u32> = logits.iter().map(|&x| u32::from(x)).collect();
        println!(
            "  logits, 4E {label}: {} positions x {} rows, fingerprint {:016x}, last token {}",
            made.len(),
            logits.len() / made.len(),
            llvq_llm::eval::token_fingerprint(&bits),
            made[n_new - 1]
        );
        let Some((rl, rv, rmade)) = reference.as_ref() else {
            reference = Some((label, logits, made));
            continue;
        };
        let vocab = logits.len() / made.len();
        // position t's logits read the tokens first, made[..t]: equal inputs up to the first made token that differs
        let same_in = made.iter().zip(rmade).position(|(a, b)| a != b).map_or(made.len(), |p| p + 1);
        let span = same_in.min(made.len()) * vocab;
        let differ = rv[..span].iter().zip(&logits[..span]).filter(|(a, b)| a != b).count();
        let first_at = rv[..span].iter().zip(&logits[..span]).position(|(a, b)| a != b);
        if differ == 0 && rv.len() == logits.len() && same_in == made.len() {
            println!("  logits, 4E {label} against 4E {rl}: bit for bit, {} values", logits.len());
        } else {
            let (mut dmax, mut at) = (0f32, 0usize);
            for (i, (a, b)) in rv[..span].iter().zip(&logits[..span]).enumerate() {
                let d = (half::f16::from_bits(*a).to_f32() - half::f16::from_bits(*b).to_f32()).abs();
                if d > dmax || d.is_nan() {
                    (dmax, at) = (d, i);
                }
            }
            println!(
                "  logits, 4E {label} against 4E {rl}: DIFFER, {differ} of {span} values on the {} positions whose \
                 inputs are equal (of {}), the first at position {} row {}; max |dlogit| {dmax:.4e} at position {} row {}",
                same_in.min(made.len()),
                made.len(),
                first_at.map_or(0, |i| i / vocab),
                first_at.map_or(0, |i| i % vocab),
                at / vocab,
                at % vocab
            );
        }
    }
    for rt in rts.iter_mut() {
        rt.set_kernel_from_for_measurement(from);
    }
    Ok(())
}
