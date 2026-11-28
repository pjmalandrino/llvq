//! The two decode-speed sweeps over prompt length, `LLVQ_GQA_AB` and
//! `LLVQ_CONTEXTS`. One file because both prefill outside the clock, then time
//! the decode after it, round by round.

use std::time::Instant;

use super::common::{first_divergence, rate_stats, GqaArm, ROUNDS_TIMED};

/// The second chunking `LLVQ_CONTEXTS` reads a kernel-prefill prompt in, to
/// check its logits do not depend on the chunking.
#[cfg_attr(not(all(target_os = "linux", feature = "cuda")), allow(dead_code))]
const PREFILL_PROBE_ROWS: usize = 512;

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
pub(super) fn run_gqa_ab(
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

/// `LLVQ_CONTEXTS`: for each prompt length, the time to the first token and
/// the decode speed after it, one discarded generation then
/// [`ROUNDS_TIMED`], tokens identical across rounds. The prompt repeats
/// `seed` to length, as `LLVQ_PREFILL_TOKENS` does: no kernel branches on
/// the ids. The published tok/s divides the whole generation, prompt
/// included, by the new tokens; at a 5-token prompt the two readings agree,
/// at 8,192 they do not, which is why this one splits the clock.
#[cfg_attr(not(all(target_os = "linux", feature = "cuda")), allow(dead_code))]
pub(super) fn run_context_sweep(
    model: &llvq_llm::model::Qwen3,
    dec: &mut llvq_llm::served_decode::Decoder,
    seed: &[u32],
    lengths: &[usize],
    n_new: usize,
) -> anyhow::Result<()> {
    if n_new < 2 || seed.is_empty() {
        anyhow::bail!(
            "LLVQ_CONTEXTS needs a seed prompt and at least 2 new tokens: the decode is \
             timed after the first one ({n_new} asked)"
        );
    }
    let chunk = llvq_llm::model::MAX_PREFILL_ROWS;
    println!(
        "context sweep: {n_new} new tokens after each prompt, the prompt in chunks of \
         {chunk}, 1 discarded + {ROUNDS_TIMED} rounds"
    );
    println!("  prompt    to first token, s       decode tok/s [range]      ms a decode token");
    let mut generated = Vec::with_capacity(lengths.len());
    for &n in lengths {
        let ids: Vec<u32> = (0..n).map(|i| seed[i % seed.len()]).collect();
        let (mut first, mut rate) = (Vec::new(), Vec::new());
        let mut reference: Option<Vec<u32>> = None;
        // The rounds of this length between two marks, for a job's
        // epoch-stamped card-memory sampler: the identity runs below are
        // outside them, so a phase peak is the timed form's alone.
        let mark = |what: &str| {
            let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0);
            println!("mark ctx-{n} {what} {t:.3}");
        };
        mark("begin");
        for round in 0..=ROUNDS_TIMED {
            let g = dec.generate_timed(model, &ids, n_new, chunk)?;
            match &reference {
                None => reference = Some(g.tokens),
                Some(r) => {
                    if let Some(i) =
                        first_divergence(r, &g.tokens).map_err(|e| anyhow::anyhow!("{e}"))?
                    {
                        anyhow::bail!("prompt {n}, round {round}: divergence at token {i}");
                    }
                }
            }
            if round > 0 {
                first.push(g.prompt_s);
                rate.push((n_new - 1) as f64 / g.decode_s);
            }
        }
        mark("end");
        // Under the persistent decode, 4E owes the eager path's tokens under
        // the same config (the kernel form from position 0): one eager run a
        // length, compared token for token, or the sweep is void.
        if dec.is_persistent() && dec.has_chain() {
            // Under the chain prefill, 4E owes the eager DECODE's tokens on the
            // caches the chain left; the eager prefill's are reported only.
            let after_chain = dec.generate_eager_decode(model, &ids, n_new, chunk)?;
            let r = reference.as_ref().expect("set by round 0");
            if let Some(i) = first_divergence(&after_chain, r).map_err(|e| anyhow::anyhow!("{e}"))? {
                anyhow::bail!("prompt {n}: 4E diverges from the eager decode after the chain prefill at token {i}; the run is void");
            }
            let eager = model.generate_timed(&ids, n_new, chunk)?;
            let same = first_divergence(&eager.tokens, r).map_err(|e| anyhow::anyhow!("{e}"))?;
            println!(
                "  prompt {n}: 4E tokens equal the eager decode's after the chain prefill, {} of {}; against the eager prefill's: {}",
                r.len(),
                after_chain.len(),
                same.map_or_else(|| "identical".to_string(), |i| format!("first divergence at token {i}"))
            );
        } else if dec.is_persistent() {
            let eager = model.generate_timed(&ids, n_new, chunk)?;
            let r = reference.as_ref().expect("set by round 0");
            if let Some(i) = first_divergence(&eager.tokens, r).map_err(|e| anyhow::anyhow!("{e}"))? {
                anyhow::bail!("prompt {n}: 4E diverges from the eager path under this config at token {i}; the run is void");
            }
            println!("  prompt {n}: 4E tokens equal the eager path's under this config, {} of {}", r.len(), eager.tokens.len());
        }
        // The prompt's last-position logits, bit for bit, as a fingerprint:
        // what two processes compare when one runs the whole-tensor softmax
        // (`LLVQ_SOFTMAX_WHOLE=1`) and the other the in-place one, or two
        // configs that should prefill alike. One more prefill a length.
        let logit_bits = |chunk: usize| -> anyhow::Result<Vec<u32>> {
            let mut c = model.fresh_caches();
            let l = model.prefill_chunked(&ids, chunk, &mut c)?;
            Ok(l.flatten_all()?.to_vec1::<f32>()?.iter().map(|x| x.to_bits()).collect())
        };
        let a = logit_bits(chunk)?;
        println!(
            "  prompt {n}: last-position logits fingerprint {:016x} (prefill in chunks of {chunk}, softmax {})",
            llvq_llm::eval::token_fingerprint(&a),
            llvq_llm::model::softmax_form()
        );
        // Under the kernel prefill those logits must not depend on the
        // chunking: every attention row is the kernel's for its position, and
        // what is left to move them is the projections' rows kernels. Read,
        // not enforced: the card gate of the field decides on it.
        if n > PREFILL_PROBE_ROWS && model.prefill_attention() == llvq_llm::kvq::PrefillAttn::Kernel {
            let b = logit_bits(PREFILL_PROBE_ROWS)?;
            let differ = a.iter().zip(&b).filter(|(x, y)| x != y).count();
            println!(
                "  prompt {n}: prefill attention kernel, last-position logits in chunks of {chunk} against \
                 chunks of {PREFILL_PROBE_ROWS}: {} ({differ} of {} differ)",
                if differ == 0 { "bit for bit" } else { "DIFFER" },
                a.len()
            );
        }
        let (fm, fl, fh) = rate_stats(&first);
        let (rm, rl, rh) = rate_stats(&rate);
        println!(
            "  {n:>6}    {fm:7.2} [{fl:.2}–{fh:.2}]    {rm:7.1} [{rl:.1}–{rh:.1}]    {:7.2}",
            1000.0 / rm
        );
        generated.push((n, reference.unwrap_or_default()));
    }
    // The tokens each prompt produced, below the table, so two configs'
    // sweeps can be compared token for token after the job: the attention
    // kernel's form must equal the grouped form's wherever every decoded
    // position lies below its threshold.
    for (n, t) in &generated {
        let ids: Vec<String> = t.iter().map(u32::to_string).collect();
        println!(
            "  tokens after {n}: {:016x} {}",
            llvq_llm::eval::token_fingerprint(t),
            ids.join(",")
        );
    }
    Ok(())
}
