//! `LLVQ_CONTEXTS`: the time to the first token and the decode speed after it, per prompt length.

use super::*;

/// The prompt lengths of `LLVQ_CONTEXTS`, comma-separated, each above zero.
#[cfg_attr(not(all(target_os = "linux", feature = "cuda")), allow(dead_code))]
fn parse_contexts(v: &str) -> Result<Vec<usize>, String> {
    parse_lengths("LLVQ_CONTEXTS", v)
}

/// `LLVQ_CONTEXTS`: for each prompt length, the time to the first token and
/// the decode speed after it, one discarded generation then
/// [`ROUNDS_TIMED`], tokens identical across rounds. The prompt repeats
/// `seed` to length, as `LLVQ_PREFILL_TOKENS` does: no kernel branches on
/// the ids. The published tok/s divides the whole generation, prompt
/// included, by the new tokens; at a 5-token prompt the two readings agree,
/// at 8,192 they do not, which is why this one splits the clock.
#[cfg_attr(not(all(target_os = "linux", feature = "cuda")), allow(dead_code))]
fn run_context_sweep(
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
        // Under the persistent decode, 4E owes the eager path's tokens under
        // the same config (the kernel form from position 0): one eager run a
        // length, compared token for token, or the sweep is void.
        if dec.is_persistent() {
            let eager = model.generate_timed(&ids, n_new, chunk)?;
            let r = reference.as_ref().expect("set by round 0");
            if let Some(i) = first_divergence(&eager.tokens, r).map_err(|e| anyhow::anyhow!("{e}"))? {
                anyhow::bail!("prompt {n}: 4E diverges from the eager path under this config at token {i}; the run is void");
            }
            println!("  prompt {n}: 4E tokens equal the eager path's under this config, {} of {}", r.len(), eager.tokens.len());
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

#[cfg(all(target_os = "linux", feature = "cuda"))]
/// The `LLVQ_CONTEXTS` mode: one load, served or bench, then [`run_context_sweep`].
pub(super) fn contexts_mode(
    v: String,
    served: Option<llvq_llm::served::Served>,
    path: String,
    device: candle_core::Device,
    dtype: candle_core::DType,
    n_new: usize,
) -> anyhow::Result<()> {
    let lengths = parse_contexts(&v).map_err(|e| anyhow::anyhow!("{e}"))?;
    runs_alone("LLVQ_CONTEXTS", &[])?;
    let t = Instant::now();
    let mut f = match &served {
        Some(cfg) => {
            println!("{}", cfg.provenance());
            llvq_llm::fused_cuda::load_resolved(
                &path, &device, dtype, cfg.layout, cfg.embed, cfg.rot_share, cfg.fuse,
                cfg.kv, cfg.int4_layout(), Some("LLVQ_CONFIG"),
            )?
        }
        None => {
            let fuse =
                llvq_llm::fused::FuseMode::from_env().map_err(|e| anyhow::anyhow!("{e}"))?;
            llvq_llm::fused_cuda::load_with(&path, &device, dtype, fuse)?
        }
    };
    // The served attention form rides this mode too, for the reason
    // the loader choice above rides it: the object timed at 8,192 must
    // be the object served, not the form the model defaults to while a
    // provenance line names another.
    if let Some(cfg) = &served {
        cfg.apply(&mut f.model, &device).map_err(|e| anyhow::anyhow!("{e}"))?;
    }
    println!(
        "layout {}, attention {} on {}, loaded in {:.1} s",
        f.layout.name(),
        f.model.attn_heads().label(),
        f.model.kv_store().label(),
        t.elapsed().as_secs_f64()
    );
    let seed = f
        .tokenizer
        .encode(PROMPT, false)
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .get_ids()
        .to_vec();
    // The engine the config names (eager without one, or without the
    // field): 4E installs here, after the form and its threshold.
    let mut dec = match &served {
        Some(cfg) => cfg.decoder(&f.model, &device)?,
        None => llvq_llm::served_decode::Decoder::eager(),
    };
    for line in dec.lines() {
        println!("{line}");
    }
    println!("decode engine: {}", if dec.is_persistent() { "persistent (4E)" } else { "eager" });
    run_context_sweep(&f.model, &mut dec, &seed, &lengths, n_new)?;
    Ok(())
}
