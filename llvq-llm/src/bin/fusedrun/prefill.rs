//! `LLVQ_PREFILL_TOKENS`: what a prompt costs through the kernel, and its batched-against-per-row gate.

use super::*;

/// The `LLVQ_PREFILL_TOKENS` mode.
pub(super) fn prefill_mode(
    v: String,
    served: Option<llvq_llm::served::Served>,
    path: String,
    device: candle_core::Device,
    dtype: candle_core::DType,
) -> anyhow::Result<()> {
    use candle_core::IndexOp;
    let n: usize = v
        .parse()
        .map_err(|e| anyhow::anyhow!("LLVQ_PREFILL_TOKENS={v:?}: {e}"))?;
    if n == 0 {
        anyhow::bail!("LLVQ_PREFILL_TOKENS=0: there is nothing to time");
    }
    let t = Instant::now();
    // Through the served door when there is a config, so the object
    // timed is the object served. It was `load_with` — the bench door
    // — for every config, and `LLVQ_CONFIG=… LLVQ_PREFILL_TOKENS=800`
    // would have timed planes14 at f16 while printing a provenance
    // naming variables nobody set. Found by the audit of 2026-09-11.
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
    // Served: the config decides the store and the form. Bench: the
    // shipped store.
    if let Some(cfg) = &served {
        cfg.apply(&mut f.model, &device).map_err(|e| anyhow::anyhow!("{e}"))?;
    }
    let load = t.elapsed().as_secs_f64();
    // The ids do not matter for a timing — the kernel has no
    // data-dependent branch — so the prompt is repeated to length
    // rather than a corpus being fetched for it.
    let seed = f
        .tokenizer
        .encode(PROMPT, false)
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .get_ids()
        .to_vec();
    let ids: Vec<u32> = (0..n).map(|i| seed[i % seed.len()]).collect();
    println!(
        "prefill: {n} tokens, layout {}, attention {} on {}, loaded in {load:.1} s",
        f.layout.name(),
        f.model.attn_heads().label(),
        f.model.kv_store().label(),
    );
    let mut ms: Vec<f64> = Vec::new();
    // Six passes, the first discarded: the first on CUDA pays kernel
    // selection, allocator growth and the clock ramp.
    for round in 0..6usize {
        let mut caches = f.model.fresh_caches();
        let input = candle_core::Tensor::from_slice(&ids, (1, n), &device)?;
        let t = Instant::now();
        let h = f.model.hidden_cached(&input, 0, &mut caches, &mut NoCapture)?;
        let l = h.dim(1)?;
        let last = h.narrow(1, l - 1, 1)?;
        // The logits too: that is what a scoring pass reads, and
        // leaving them out would time three quarters of the question.
        let _ = f.model.project_head(&last)?.i((0, 0))?.to_dtype(candle_core::DType::F32)?;
        device.synchronize()?;
        if round > 0 {
            ms.push(t.elapsed().as_secs_f64() * 1e3);
        }
    }
    ms.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    let (lo, med, hi) = (ms[0], ms[ms.len() / 2], ms[ms.len() - 1]);
    println!(
        "  {med:8.1} ms a prefill of {n} tokens  [{lo:.1}–{hi:.1}], 5 rounds",
    );
    println!(
        "  {:8.3} ms a token of prompt — multiply by the prompt and by the questions",
        med / n as f64
    );
    // The census of this repository is 2,280 questions — 40 a subject
    // over 57 — not the 14,042 of the full split: every published bar
    // is measured on that sample and `mmlupair` refuses two dumps of
    // different plans. Both are printed, because the second is what a
    // reader who knows MMLU and not this repository will expect.
    for (label, q) in [("the census of this repository", 2_280.0f64),
                       ("the whole MMLU test split   ", 14_042.0f64)] {
        println!(
            "  {label}: {:>6} questions -> {:5.2} h at {n} tokens a prompt",
            q as u64,
            med * q / 1000.0 / 3600.0
        );
    }
    println!(
        "  (the kernel issues ceil({n}/{}) launches a projection a question; the dense\n  \
         path issues ONE and does the 2,280 in 4 min 22 s — *measured*, job 6aa414dd)",
        // What this unit COMPILED at, not the served constant: under
        // `LLVQ_PREFILL` they differ, and the whole point of the line
        // is to say how many launches THIS run issues.
        f.prefill.rows
    );

    // ---- THE GATE, and this mode had none --------------------------
    //
    // A timing is not a result. What is timed here is the BATCHED
    // path — `group_forward` takes `PREFILL_ROWS` rows a launch as
    // soon as `rows > 1` — and an addressing bug in it would make
    // this loop faster and wrong at once.
    //
    // The second arm needs no flag and no second build: fed ONE token
    // at a time against the same growing cache, every call has
    // `rows == 1`, so `batch` is 1 and the per-row path runs. Same
    // weights, same kernels, same order of layers; the only thing
    // that differs is the chunking this lot changed.
    //
    // Not bit-exact, and saying so is the point: attention over `n`
    // rows at once accumulates in a different order from `n` steps of
    // one row, so the two agree to f16 and not beyond. What a wrong
    // row offset produces is not an ulp.
    let mut caches = f.model.fresh_caches();
    let input = candle_core::Tensor::from_slice(&ids, (1, n), &device)?;
    let h = f.model.hidden_cached(&input, 0, &mut caches, &mut NoCapture)?;
    let l = h.dim(1)?;
    let batched: Vec<f32> = f
        .model
        .project_head(&h.narrow(1, l - 1, 1)?)?
        .i((0, 0))?
        .to_dtype(candle_core::DType::F32)?
        .to_vec1()?;

    let mut one_at_a_time = f.model.fresh_caches();
    let mut last_row: Option<candle_core::Tensor> = None;
    for (pos, id) in ids.iter().enumerate() {
        let step = candle_core::Tensor::from_slice(&[*id], (1, 1), &device)?;
        last_row =
            Some(f.model.hidden_cached(&step, pos, &mut one_at_a_time, &mut NoCapture)?);
    }
    let stepwise: Vec<f32> = f
        .model
        .project_head(&last_row.expect("n >= 1 rows"))?
        .i((0, 0))?
        .to_dtype(candle_core::DType::F32)?
        .to_vec1()?;

    let worst = batched
        .iter()
        .zip(&stepwise)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let scale = batched.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let argmax = |v: &[f32]| {
        v.iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &x)| {
                if x > bv { (i, x) } else { (bi, bv) }
            })
            .0
    };
    println!(
        "\n  gate: {n} tokens in one call against {n} calls of one token\n  \
         max |delta logit| = {worst:.3e} on a scale of {scale:.3e}, argmax {} vs {}",
        argmax(&batched),
        argmax(&stepwise)
    );
    if argmax(&batched) != argmax(&stepwise) {
        anyhow::bail!(
            "the batched prefill and the per-row one pick different tokens: \
             {} against {}",
            argmax(&batched),
            argmax(&stepwise)
        );
    }
    Ok(())
}
