//! The bench door: the fused arm against the dense one, same file, same prompt.

use super::*;

/// The bench: the fused arm, then the dense arm, then their tokens and costs side by side.
pub(super) fn bench(path: String, device: candle_core::Device, dtype: candle_core::DType, n_new: usize) -> anyhow::Result<()> {
    // The fused arm, named by `LLVQ_FUSE`. A list of one since
    // `LLVQ_FUSE_AB`, which ran both, was removed on 2026-11-08.
    let modes: Vec<llvq_llm::fused::FuseMode> =
        vec![llvq_llm::fused::FuseMode::from_env().map_err(|e| anyhow::anyhow!("{e}"))?];

    // ---- fused arms, first ----
    //
    // Deliberately before the dense arm: this is the path that can fail,
    // and it loads in ~145 s against the dense arm's ~209. Two runs died
    // here on 2026-08-05 — a refused prefill, then a length read off a
    // range — and each one had already paid for a dense load it never
    // used. Order is not neutral when a job is billed by the minute.
    let mut arms: Vec<Arm> = Vec::with_capacity(modes.len());
    let mut tok: Option<tokenizers::Tokenizer> = None;
    let mut fused_rt_bits = 0.0f64;
    let mut fused_file = 0.0f64;
    for fuse in modes {
        let t = Instant::now();
        let f = llvq_llm::fused_cuda::load_with(&path, &device, dtype, fuse)?;
        let load_s = t.elapsed().as_secs_f64();
        let ids = f
            .tokenizer
            .encode(PROMPT, false)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .get_ids()
            .to_vec();
        // The guard clauses of the card gate. Both thresholds live in
        // `rotplan`, where a test can reach them — this whole block
        // compiles on no developer machine, and a mutant that weakened the
        // rotation bound to zero survived the entire suite while the
        // comparison was written out here.
        if !llvq_llm::rotplan::arms_are_discriminating(ids.len(), n_new) {
            anyhow::bail!(
                "prompt of {} token(s), {n_new} new: the two LLVQ_ROT_SHARE arms \
                 then walk the same path, and the token comparison would print a \
                 green that says nothing.",
                ids.len()
            );
        }
        if !llvq_llm::rotplan::fuse_arms_are_discriminating(ids.len(), n_new) {
            anyhow::bail!(
                "prompt of {} token(s), {n_new} new: too short to separate the two \
                 LLVQ_FUSE arms from load and clock-ramp noise.",
                ids.len()
            );
        }
        f.model.generate(&ids, n_new, &mut NoCapture)?;
        let mut rounds = Vec::with_capacity(ROUNDS_TIMED);
        let mut tokens: Vec<u32> = Vec::new();
        for round in 0..ROUNDS_TIMED {
            let t = Instant::now();
            let out = f.model.generate(&ids, n_new, &mut NoCapture)?;
            rounds.push(n_new as f64 / t.elapsed().as_secs_f64());
            if round == 0 {
                tokens = out;
            } else if out != tokens {
                // Greedy decode on fixed weights should be deterministic;
                // a flip across rounds is a finding, not a nuisance.
                println!(
                    "  WARNING: fused arm LLVQ_FUSE={}: the round {round} tokens differ \
                     from round 0. Nondeterministic decode, to investigate before \
                     publishing",
                    f.fuse.name()
                );
            }
        }
        let (rate, lo, hi) = rate_stats(&rounds);
        // `carried_bytes`, not `carried_weights * 2`: under LLVQ_EMBED=q8
        // the embedding sits on the card as int8 + f16 scales, and the old
        // identity would over-report by ~365 MB.
        let bytes = f.runtime_bytes + f.carried_bytes;
        // The trailing "projections … b/weight" line reports the first arm.
        if arms.is_empty() {
            fused_rt_bits = f.runtime_bytes as f64 * 8.0 / f.quantized_weights as f64;
            fused_file = f.file_bytes as f64 / 1e9;
        }
        println!(
            "fused  : loaded in {load_s:6.1} s, {rate:6.1} tok/s \
             [{lo:.1}–{hi:.1}, {ROUNDS_TIMED} rounds], {:.2} GB on the card",
            bytes as f64 / 1e9
        );
        // On the arm line, not only in the loader's log: an A/B whose two
        // runs are told apart by scrolling back to a load-time message is
        // an A/B waiting to be misread. The matvec count is here for the
        // same reason the rotation count is — a gate reading "128 tokens
        // identical" while both arms issued 252 matvecs proves the tokens
        // and nothing about this lot.
        println!(
            "         LLVQ_ROT_SHARE={}, {} rot_lancements/token · LLVQ_FUSE={}, {} \
             matvec_lancements/token · kv_store={}",
            f.rot_share.name(),
            f.rot_launches,
            f.fuse.name(),
            f.matvec_launches,
            f.model.kv_store().label()
        );
        if tok.is_none() {
            tok = Some(f.tokenizer.clone());
        }
        arms.push(Arm {
            fuse: f.fuse,
            tokens,
            load_s,
            rate,
            lo,
            hi,
            bytes,
            rot_launches: f.rot_launches,
            matvec_launches: f.matvec_launches,
        });
        // Released before the next arm — and before the dense one — loads:
        // the point is to report what each one costs, not to prove a card
        // can hold two.
        drop(f);
    }
    let tok = tok.expect("at least one fused arm ran");

    // ---- dense arm ----
    let (dense_tokens, dense_load, dense_rate, dense_lo, dense_hi, dense_bytes) = {
        let t = Instant::now();
        let m = llvq_llm::sealed::load(&path, dtype, &device, llvq_llm::kvq::KvMode::F16)?;
        let load = t.elapsed().as_secs_f64();
        let ids = m
            .tokenizer
            .encode(PROMPT, false)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .get_ids()
            .to_vec();
        // One discarded generation: the first on CUDA pays kernel
        // selection, allocator growth and clock ramp.
        m.model.generate(&ids, n_new, &mut NoCapture)?;
        let mut rounds = Vec::with_capacity(ROUNDS_TIMED);
        let mut out: Vec<u32> = Vec::new();
        for round in 0..ROUNDS_TIMED {
            let t = Instant::now();
            let o = m.model.generate(&ids, n_new, &mut NoCapture)?;
            rounds.push(n_new as f64 / t.elapsed().as_secs_f64());
            if round == 0 {
                out = o;
            } else if o != out {
                println!(
                    "  WARNING: dense arm: the round {round} tokens differ from round 0. \
                     Nondeterministic decode, to investigate before publishing"
                );
            }
        }
        let (rate, lo, hi) = rate_stats(&rounds);
        let bytes = (m.quantized_weights + m.carried_weights) as u64 * 2;
        println!(
            "dense  : loaded in {load:6.1} s, {rate:6.1} tok/s \
             [{lo:.1}–{hi:.1}, {ROUNDS_TIMED} rounds], {:.2} GB on the card",
            bytes as f64 / 1e9
        );
        println!("         kv_store={}", m.model.kv_store().label());
        (out, load, rate, lo, hi, bytes)
    };

    // ---- the comparison ----
    println!("\n--- the two arms ---");
    for a in &arms {
        let first = first_divergence(&dense_tokens, &a.tokens)
            .map_err(|e| anyhow::anyhow!("dense against fused F{}: {e}", a.fuse.name()))?;
        match first {
            None => println!(
                "  fused F{}: {} tokens identical to the dense arm",
                a.fuse.name(),
                dense_tokens.len()
            ),
            Some(i) => {
                println!(
                    "  WARNING: fused F{}: divergence at token {i} out of {}",
                    a.fuse.name(),
                    dense_tokens.len()
                );
                println!("     dense {:?}", &dense_tokens[i.saturating_sub(2)..]);
                println!("     fused {:?}", &a.tokens[i.saturating_sub(2)..]);
                println!(
                    "     a greedy decode is a chain of argmaxes: one flipped \
                     token\n     changes every token after it. Only the POSITION of \
                     the first divergence means anything."
                );
            }
        }
    }
    // 🚨 The criterion of this lot, and it is the POSITION of the first
    // divergence rather than a count: a greedy decode is a chain of
    // argmaxes, so one flipped token changes every token after it. What the
    // 2026-08-06 journal establishes is that `Planes14` and `slot32`
    // diverge from the dense arm AT THE SAME TOKEN (89 of 128) — that
    // control is what licenses reading 89 as a tie-break rather than a bug.
    // Transposed here: the two `LLVQ_FUSE` arms must diverge from the dense
    // arm at the same position, and — strictly stronger — agree with each
    // other everywhere.
    if let [a, b] = &arms[..] {
        match first_divergence(&a.tokens, &b.tokens)
            .map_err(|e| anyhow::anyhow!("fused F{} against F{}: {e}", a.fuse.name(), b.fuse.name()))?
        {
            None => println!(
                "  OK: fused F{} and F{}: {} tokens identical to each other",
                a.fuse.name(),
                b.fuse.name(),
                a.tokens.len()
            ),
            Some(i) => println!(
                "  ALERT: fused F{} and F{} diverge at token {i}. The fusion itself \
                 changed the arithmetic, not a tie-break",
                a.fuse.name(),
                b.fuse.name()
            ),
        }
    }
    println!("  dense: {}", tok.decode(&dense_tokens, true).unwrap_or_default());
    for a in &arms {
        println!(
            "  fused F{}: {}",
            a.fuse.name(),
            tok.decode(&a.tokens, true).unwrap_or_default()
        );
    }

    println!("\n--- what it costs ---");
    println!(
        "  {:<10}{:>12}{:>16}{:>18}{:>12}{:>10}",
        "arm", "load", "tok/s (median)", "range", "GB card", "matvec/t"
    );
    println!("  {}", "-".repeat(78));
    println!(
        "  {:<10}{dense_load:>11.1} s{dense_rate:>16.1}{:>18}{:>12.2}{:>10}",
        "dense",
        format!("[{dense_lo:.1}–{dense_hi:.1}]"),
        dense_bytes as f64 / 1e9,
        "—"
    );
    for a in &arms {
        println!(
            "  {:<10}{:>11.1} s{:>16.1}{:>18}{:>12.2}{:>10}",
            format!("fused F{}", a.fuse.name()),
            a.load_s,
            a.rate,
            format!("[{:.1}–{:.1}]", a.lo, a.hi),
            a.bytes as f64 / 1e9,
            a.matvec_launches
        );
    }
    println!("  {}", "-".repeat(78));
    // A quotient of two medians, NOT a median of per-round ratios: the
    // two arms' rounds never coexist (each load is exclusive), so no
    // per-round pairing exists. The envelope divides the extremes the
    // conservative way round; read the second decimal as dispersion.
    for a in &arms {
        println!(
            "  fused F{}: speed ×{:.2} [×{:.2}–×{:.2}] against the dense arm (quotient \
             of medians, {ROUNDS_TIMED} rounds per arm, never interleaved), memory ÷{:.2}",
            a.fuse.name(),
            a.rate / dense_rate,
            a.lo / dense_hi,
            a.hi / dense_lo,
            dense_bytes as f64 / a.bytes as f64
        );
    }
    // 🚨 The number this lot is about, and the only comparison in this
    // output whose two sides share a translation unit and a card: fused
    // against unfused, both of them ours. The 11.7% the bench measured on
    // `tv_planes_seg` (5.096 → 4.504 ms) does **not** transport here — it is
    // f32, out of model, on matvecs alone, where this is f16 end to end and
    // the matvecs are one share of a token's time.
    if let [a, b] = &arms[..] {
        println!(
            "  fusion: ×{:.3} [×{:.3}–×{:.3}] of F{} over F{} ({} against {} \
             matvec_lancements/token), memory {:+} bytes",
            a.rate / b.rate,
            a.lo / b.hi,
            a.hi / b.lo,
            a.fuse.name(),
            b.fuse.name(),
            a.matvec_launches,
            b.matvec_launches,
            a.bytes as i64 - b.bytes as i64
        );
        if a.rot_launches != b.rot_launches {
            println!(
                "  WARNING: {} rot_lancements/token against {}: the two arms do NOT \
                 differ by the fusion alone, the ratio above mixes two mechanisms",
                a.rot_launches, b.rot_launches
            );
        }
    }
    println!(
        "\n  projections: {:.3} b/weight on the card, file {:.2} GB on disk.\n  \
         WARNING: these two accountings do not compare. The file carries compact\n  \
         indexes, the card reads the unfolded layout the kernel can read fast.",
        fused_rt_bits,
        fused_file
    );
    Ok(())
}
