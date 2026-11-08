//! The served door: one arm under `LLVQ_CONFIG`, no dense reference.

use super::*;

#[cfg(all(target_os = "linux", feature = "cuda"))]
/// The served door under a config that names the persistent decode. The eager
/// path under the same config (the kernel form from position 0) makes the
/// reference tokens; 4E must make the same ones in every round, or the run is
/// void. Then 1 discarded + [`ROUNDS_TIMED`] rounds, eager and 4E in each, both
/// timed as the door times a generate (prompt and decode), and the decode
/// alone beside it. The token ids are printed so a second config's door can
/// be compared token for token after the job.
fn served_door_4e(
    f: &llvq_llm::fused::FusedSealed,
    dec: &mut llvq_llm::served_decode::Decoder,
    ids: &[u32],
    n_new: usize,
    load_s: f64,
) -> anyhow::Result<()> {
    use llvq_llm::stats::median_range;
    anyhow::ensure!(n_new >= 2, "the served 4E door times the decode after the first token: {n_new} asked");
    let chunk = llvq_llm::model::MAX_PREFILL_ROWS;
    let want = f.model.generate_timed(ids, n_new, chunk)?.tokens;
    let (mut door, mut dec_rate, mut door_eager, mut dec_eager) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for round in 0..=ROUNDS_TIMED {
        let t = Instant::now();
        let e = f.model.generate_timed(ids, n_new, chunk)?;
        let te = t.elapsed().as_secs_f64();
        if e.tokens != want {
            anyhow::bail!("round {round}: the eager path moved its tokens; the run is void");
        }
        let t = Instant::now();
        let g = dec.generate_timed(&f.model, ids, n_new, chunk)?;
        let tg = t.elapsed().as_secs_f64();
        if let Some(i) = first_divergence(&want, &g.tokens).map_err(|e| anyhow::anyhow!("{e}"))? {
            anyhow::bail!("round {round}: 4E diverges from the eager path under this config at token {i}; the run is void");
        }
        if round > 0 {
            door_eager.push(n_new as f64 / te);
            dec_eager.push((n_new - 1) as f64 / e.decode_s);
            door.push(n_new as f64 / tg);
            dec_rate.push((n_new - 1) as f64 / g.decode_s);
        }
    }
    println!("served door (decode persistent): 4E tokens equal the eager path's under this config, {} of {}", want.len(), want.len());
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

/// The served path under `LLVQ_CONFIG`: one arm, timed, and its tokens printed.
pub(super) fn served_door(
    cfg: &llvq_llm::served::Served,
    path: String,
    device: candle_core::Device,
    dtype: candle_core::DType,
    n_new: usize,
) -> anyhow::Result<()> {
    println!("{}", cfg.provenance());
    let t = Instant::now();
    let mut f = llvq_llm::fused_cuda::load_resolved(
        &path,
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
    // The attention form and its store, from the file and from
    // nowhere else. The provenance line above already named them.
    cfg.apply(&mut f.model, &device).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut dec = cfg.decoder(&f.model, &device)?;
    for line in dec.lines() {
        println!("{line}");
    }
    // The card's own count, the whole context (weights, tables, 4E's
    // module and scratch, candle's pool), before the first token:
    // what serve-v9 reads to show the int4 bytes are held once.
    let (free, total) = candle_core::cuda_backend::cudarc::driver::result::mem_get_info()
        .map_err(|e| anyhow::anyhow!("cuMemGetInfo: {e}"))?;
    println!(
        "card memory: {:.3} GB used of {:.3} GB after the load and the decoder install (cuMemGetInfo)",
        (total - free) as f64 / 1e9,
        total as f64 / 1e9
    );
    let load_s = t.elapsed().as_secs_f64();
    let ids = f
        .tokenizer
        .encode(PROMPT, false)
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .get_ids()
        .to_vec();
    if dec.is_persistent() {
        return served_door_4e(&f, &mut dec, &ids, n_new, load_s);
    }
    f.model.generate(&ids, n_new, &mut NoCapture)?;
    let mut rounds = Vec::with_capacity(ROUNDS_TIMED);
    let mut tokens: Vec<u32> = Vec::new();
    for round in 0..ROUNDS_TIMED {
        let t = Instant::now();
        let out = f.model.generate(&ids, n_new, &mut NoCapture)?;
        rounds.push(n_new as f64 / t.elapsed().as_secs_f64());
        match round {
            0 => tokens = out,
            _ if out != tokens => println!(
                "  WARNING: the round {round} tokens differ from round 0. \
                 Nondeterministic decode, to investigate before publishing"
            ),
            _ => {}
        }
    }
    let (rate, lo, hi) = rate_stats(&rounds);
    let bytes = f.runtime_bytes + f.carried_bytes;
    println!(
        "served : loaded in {load_s:6.1} s, {rate:6.1} tok/s [{lo:.1}–{hi:.1}, \
         {ROUNDS_TIMED} rounds], {:.2} GB on the card",
        bytes as f64 / 1e9
    );
    println!("  {}", f.tokenizer.decode(&tokens, true).unwrap_or_default());
    let s: Vec<String> = tokens.iter().map(u32::to_string).collect();
    println!("  token ids: {:016x} {}", llvq_llm::eval::token_fingerprint(&tokens), s.join(","));
    println!(
        "\n  No dense arm on this path, so no ratio and no token comparison. \
         Both are the bench's\n  business: unset LLVQ_CONFIG and it runs, on \
         this same file."
    );
    Ok(())
}
