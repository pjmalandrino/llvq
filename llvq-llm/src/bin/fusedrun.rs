//! The fused path against the dense one: same file, same prompt, same tokens?
//!
//! Every check so far has been local. `fused_path_matches_dense.rs` pins one
//! matrix on the CPU; `rotation_matches_rust.rs` pins the rotation kernel;
//! `bin/matvec` pins `slot_dot` against an f64 reference. None of them can say
//! what happens when 252 of those compose across 36 layers, twice per token,
//! with a KV cache in between — which is the only question that decides
//! whether this path is usable.
//!
//! So this binary loads **the same artifact twice**:
//!
//!   * dense — `sealed::load`, the object every published perplexity refers to;
//!   * fused — `fused_cuda::load`, weights left encoded, two kernels per
//!     projection.
//!
//! and requires the same tokens out. Not the same logits: the two accumulate
//! in different orders, so bit-equality is not on offer and demanding it would
//! be theatre. Identical *tokens* is the property that matters and the one a
//! user would notice breaking.
//!
//! ## Reading the divergence, if there is one
//!
//! A greedy decode is a chain of argmaxes, so one flipped token changes every
//! token after it. The position of the **first** divergence is therefore the
//! only informative number: token 1 means the arithmetic is wrong, token 30
//! means two logits were within a rounding error of each other and the tie
//! broke the other way. The second is expected on a long enough run and is not
//! a defect; the first is.
//!
//! ## What it also reports, and why it is the real headline
//!
//! Device bytes. The speed difference is bounded by things this path does not
//! touch — roughly half a token's time is attention, norms and launch
//! overhead — but the memory difference is the whole thesis: 8.04 GB of f16
//! against the encoded projections plus a full-precision embedding.

#[cfg(all(target_os = "linux", feature = "cuda"))]
use std::time::Instant;
#[cfg(all(target_os = "linux", feature = "cuda"))]
use llvq_llm::model::NoCapture;

#[cfg(all(target_os = "linux", feature = "cuda"))]
const PROMPT: &str = "The capital of France is";

/// Decode steps of the extra, fenced pass under `LLVQ_TIME_PHASES=1`.
///
/// Not cfg-gated like the rest of this binary, on purpose: the call sites
/// only exist on linux+cuda, but a Mac cannot type-check gated code, and an
/// error here would otherwise surface in a paid image build. Compiling it
/// everywhere keeps `cargo check` on the dev machine load-bearing.
#[cfg_attr(not(all(target_os = "linux", feature = "cuda")), allow(dead_code))]
const PHASE_TOKENS: usize = 32;

/// The measurement protocol, aligned on the kernel benches (2026-08-18):
/// one discarded generation per arm (kernel selection, allocator growth,
/// clock ramp), then `ROUNDS_TIMED` timed ones, reported as median + min–max
/// range. Until then every published tok/s was a single point — in
/// contradiction with rule 2 of CLAUDE.md §7, which the benches obeyed and
/// this binary did not.
///
/// Unlike `planesbench`, the two arms can NOT interleave round by round:
/// each arm loads its model exclusively (the card does not have to hold
/// both), so their rounds never coexist and the speed ratio is a **quotient
/// of two medians**, not a median of per-round ratios. The summary labels it
/// as such and prints the conservative envelope
/// `[fused_lo/dense_hi ; fused_hi/dense_lo]` beside it.
#[cfg_attr(not(all(target_os = "linux", feature = "cuda")), allow(dead_code))]
const ROUNDS_TIMED: usize = 5;

/// Median and min–max of per-round rates. Not cfg-gated, same reason as
/// [`PHASE_TOKENS`]: the Mac must type-check what a paid image build would
/// otherwise be the first to compile.
#[cfg_attr(not(all(target_os = "linux", feature = "cuda")), allow(dead_code))]
fn rate_stats(rounds: &[f64]) -> (f64, f64, f64) {
    assert!(!rounds.is_empty(), "rate_stats over zero rounds");
    let mut sorted = rounds.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let med = if sorted.len() % 2 == 1 {
        sorted[sorted.len() / 2]
    } else {
        (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
    };
    (med, sorted[0], sorted[sorted.len() - 1])
}

/// One fused arm's published numbers, so two of them can be run in one process
/// and compared side by side.
///
/// Not cfg-gated, same reason as [`PHASE_TOKENS`]: a Mac cannot type-check
/// gated code, and this struct's shape is exactly what an image build would
/// otherwise be the first to compile.
#[cfg_attr(not(all(target_os = "linux", feature = "cuda")), allow(dead_code))]
struct Arm {
    fuse: llvq_llm::fused::FuseMode,
    tokens: Vec<u32>,
    load_s: f64,
    rate: f64,
    lo: f64,
    hi: f64,
    bytes: u64,
    rot_launches: usize,
    matvec_launches: usize,
}

/// Position of the first token the two arms disagree on, or `None` when they
/// agree everywhere.
///
/// Refuses arms of different lengths instead of comparing their common prefix.
/// `zip` truncates, so an arm that returned fewer tokens would have printed
/// "N tokens identical" over the prefix and said nothing about the rest — a
/// green that means nothing, which is the failure mode this binary's own gate
/// clause exists to prevent. Out here rather than inline for the reason
/// `rotplan::arms_are_discriminating` is out of this file: a comparison written
/// inside the cfg-gated body is checked by no machine in this workspace.
#[cfg_attr(not(all(target_os = "linux", feature = "cuda")), allow(dead_code))]
fn first_divergence(a: &[u32], b: &[u32]) -> Result<Option<usize>, String> {
    if a.len() != b.len() {
        return Err(format!(
            "{} tokens against {}: the two arms did not decode the same length, and \
             comparing their common prefix would print a green that says nothing",
            a.len(),
            b.len()
        ));
    }
    Ok(a.iter().zip(b).position(|(x, y)| x != y))
}

/// The per-phase table of one arm's fenced pass. Additive diagnostics: this
/// runs *after* the arm's published measurement and prints below it, so the
/// published lines stay byte-identical.
#[cfg_attr(not(all(target_os = "linux", feature = "cuda")), allow(dead_code))]
fn print_phases(arm: &str, r: &llvq_llm::model::PhaseReport) {
    use llvq_llm::model::PhaseReport;
    println!("\n--- phases, arm {arm} (LLVQ_TIME_PHASES, {PHASE_TOKENS} tokens, off protocol) ---");
    println!("  WARNING: each phase is bounded by a device synchronization. The");
    println!("     attribution holds, but the fences serialize what the normal path");
    println!("     overlaps. This total does NOT replace the tok/s published above.");
    println!("  {:<15}{:>10}  range (ms/token)", "phase", "median");
    let mut total = 0.0;
    for (name, samples) in [
        ("embed", &r.embed_ms),
        ("blocks+norm", &r.blocks_ms),
        ("lm_head", &r.head_ms),
        ("argmax+misc", &r.rest_ms),
    ] {
        let (med, lo, hi) = PhaseReport::stats(samples);
        total += med;
        println!("  {name:<15}{med:>10.3}  [{lo:.3}–{hi:.3}]  ({} samples)", samples.len());
    }
    println!("  {:<15}{total:>10.3}  (sum of medians, fences included)", "total fenced");
}

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .ok_or_else(|| anyhow::anyhow!("usage: fusedrun <model.llvq> [n_tokens]"))?;
    let n_new: usize = args.next().map_or(Ok(32), |s| s.parse())?;

    #[cfg(not(all(target_os = "linux", feature = "cuda")))]
    {
        let _ = (path, n_new);
        anyhow::bail!("fusedrun requires an NVIDIA card and the `cuda` feature");
    }

    #[cfg(all(target_os = "linux", feature = "cuda"))]
    {
        use candle_core::{DType, Device};
        let device = Device::new_cuda(0)?;
        let dtype = DType::F16;
        // Resolved once, here — `model.rs` reads no environment variable. The
        // store rides BOTH arms identically: it is never the variable of this
        // A/B (the `check_fuse` rule), and each arm line prints it below so a
        // wiring miss shows as `cat` instead of silently measuring it.
        let kv_store = llvq_llm::kvq::KvStore::from_env().map_err(|e| anyhow::anyhow!("{e}"))?;
        println!("{device:?}, dtype {dtype:?}, {n_new} tokens\n");

        // Whether to run the extra fenced pass after each arm's published
        // measurement. With the variable unset (or anything but "1") this
        // binary is byte-identical to the published protocol: the gate is the
        // only reference to it, and no fence exists outside the gated calls.
        let time_phases = llvq_llm::model::time_phases_enabled(
            std::env::var("LLVQ_TIME_PHASES").ok().as_deref(),
        );

        // ---- A2 steps 2+3: the capture, and the graph-against-eager A/B ----
        //
        // `LLVQ_GRAPH_AB=1`: ONE model on a FRESH stream (the legacy NULL one
        // does not capture), event tracking off BEFORE any allocation (the
        // graphbench lesson of 08-06: a foreign event invalidates the
        // capture). The two arms share everything, weights, caches, StepState,
        // logits buffer, and differ by ONE thing only: the decode step is run
        // eagerly, or replayed from the captured graph. Correctness gate
        // first (identical tokens everywhere), numbers second, round by round.
        // Phase prereg 802006c5 (frozen thresholds).
        if std::env::var("LLVQ_GRAPH_AB").ok().as_deref() == Some("1") {
            use candle_core::IndexOp;
            use llvq_llm::kvq::KvStore;
            let w = match kv_store {
                KvStore::Prealloc(w) => w,
                KvStore::Cat => anyhow::bail!(
                    "LLVQ_GRAPH_AB=1 requires LLVQ_KV_PREALLOC=<window>: fixed shapes \
                     are what make the step capturable"
                ),
            };
            // This mode's device: fresh stream plus tracking off, BEFORE any
            // allocation. The published protocol's `device` (created above) is
            // not touched; this one shadows it for this mode only.
            let device = Device::new_cuda_with_stream(0)?;
            let stream = device.as_cuda_device()?.cuda_stream();
            unsafe { stream.context().disable_event_tracking() };
            let fuse = llvq_llm::fused::FuseMode::from_env().map_err(|e| anyhow::anyhow!("{e}"))?;
            let t = Instant::now();
            let mut f = llvq_llm::fused_cuda::load_with(&path, &device, dtype, fuse)?;
            f.model.set_kv_store(kv_store);
            println!(
                "loaded in {:.1} s. graph A/B: eager against replay, prealloc({w}), fresh stream, events off",
                t.elapsed().as_secs_f64()
            );
            let ids = f
                .tokenizer
                .encode(PROMPT, false)
                .map_err(|e| anyhow::anyhow!("{e}"))?
                .get_ids()
                .to_vec();
            if ids.len() + n_new > w {
                anyhow::bail!("window {w} < {} + {n_new} tokens", ids.len());
            }
            let vocab = f.model.config().vocab_size;
            let mut caches = f.model.fresh_caches();
            // The prefill sizes the buffers; step_state then ARMS the
            // capturable write path (never before: the prefill's
            // multi-position write stays on slice_set).
            let prefill = |model: &llvq_llm::model::Qwen3,
                           caches: &mut [llvq_llm::model::KvCache]|
             -> anyhow::Result<u32> {
                for c in caches.iter_mut() {
                    c.reset();
                }
                let input = candle_core::Tensor::from_slice(&ids, (1, ids.len()), &device)?;
                let h = model.hidden_cached(&input, 0, caches, &mut NoCapture)?;
                let l = h.dim(1)?;
                let last = h.narrow(1, l - 1, 1)?;
                let logits = model.project_head(&last)?.to_dtype(candle_core::DType::F32)?;
                Ok(logits.i((0, 0))?.argmax(candle_core::D::Minus1)?.to_scalar::<u32>()?)
            };
            let first = prefill(&f.model, &mut caches)?;
            let st = f.model.step_state(&mut caches)?;
            let logits_out =
                candle_core::Tensor::zeros((1usize, 1, vocab), candle_core::DType::F32, &device)?;
            // THE step, bit-identical between the two arms: what the eager arm
            // runs is what the capture recorded.
            let step = |model: &llvq_llm::model::Qwen3,
                        caches: &mut [llvq_llm::model::KvCache]|
             -> anyhow::Result<()> {
                let lg = model.token_step(&st, caches, &mut NoCapture)?;
                logits_out.slice_set(&lg.to_dtype(candle_core::DType::F32)?, 0, 0)?;
                Ok(())
            };
            let read_next = || -> anyhow::Result<u32> {
                Ok(logits_out
                    .i((0, 0))?
                    .argmax(candle_core::D::Minus1)?
                    .to_scalar::<u32>()?)
            };
            // Warmup: one complete eager generation (allocator, cuBLAS,
            // clocks), discarded.
            let gen_eager = |f: &llvq_llm::fused_cuda::FusedSealed,
                             caches: &mut [llvq_llm::model::KvCache]|
             -> anyhow::Result<Vec<u32>> {
                let mut next = prefill(&f.model, caches)?;
                let mut offset = ids.len();
                let mut out = Vec::with_capacity(n_new);
                loop {
                    out.push(next);
                    if out.len() == n_new {
                        return Ok(out);
                    }
                    f.model.refresh_step(&st, next, offset)?;
                    step(&f.model, caches)?;
                    next = read_next()?;
                    offset += 1;
                }
            };
            let warm = gen_eager(&f, &mut caches)?;
            // THE CAPTURE: prepare the state of the first real step, record
            // the step (the body runs host-side, the device RECORDS without
            // executing), then the graph is replayable at every token.
            let next0 = prefill(&f.model, &mut caches)?;
            f.model.refresh_step(&st, next0, ids.len())?;
            let graph = {
                let fm = &f.model;
                let cs = &mut caches;
                let mut err: Option<anyhow::Error> = None;
                let g = llvq_cuda::gpu::capture_on(&stream, || {
                    // THE SAME `step` as the eager arm: what the capture
                    // records is, byte for byte, what the eager arm runs.
                    if let Err(e) = step(fm, cs) {
                        err = Some(e);
                        return Err("the capture body failed".to_string());
                    }
                    Ok(())
                });
                if let Some(e) = err {
                    return Err(e.context("during the capture"));
                }
                g.map_err(|e| anyhow::anyhow!("{e}"))?
            };
            println!("capture: OK, the decode step is a replayable graph");
            // The FIRST launch of an AUTO_FREE graph is the one that
            // materializes its allocation nodes, and the 2026-09-01 diagnostic
            // measured it SLIGHTLY WRONG (max|Δlogits| = 11.2 on the first
            // launch, then 0.000e0 EXACTLY on every later one, twelve tokens
            // long). So it is spent here as a dry run, on the capture's state:
            // its writes land on a state that every generation re-prefills
            // anyway.
            graph.launch().map_err(|e| anyhow::anyhow!("graph launch (throwaway): {e}"))?;
            device.synchronize()?;
            println!("first launch spent as a dry run (the diagnostic measured it inexact)");
            let gen_graph = |f: &llvq_llm::fused_cuda::FusedSealed,
                             caches: &mut [llvq_llm::model::KvCache]|
             -> anyhow::Result<Vec<u32>> {
                let mut next = prefill(&f.model, caches)?;
                let mut offset = ids.len();
                let mut out = Vec::with_capacity(n_new);
                loop {
                    out.push(next);
                    if out.len() == n_new {
                        return Ok(out);
                    }
                    f.model.refresh_step(&st, next, offset)?;
                    graph.launch().map_err(|e| anyhow::anyhow!("graph launch: {e}"))?;
                    next = read_next()?;
                    offset += 1;
                }
            };
            // Diagnostic mode: every token does replay THEN eager on the SAME
            // state (the scatter rewrites the same position with the same
            // values, idempotent), the two logit vectors are compared in
            // full, and the StepState channels are read back FROM THE DEVICE
            // after the refresh. What this discriminates: a channel frozen by
            // value (the read-backs are right, the replay is wrong, the eager
            // arm is right), an allocator corruption (massively different
            // logits), or a numerical drift (small and growing gap).
            if std::env::var("LLVQ_GRAPH_DIAG").ok().as_deref() == Some("1") {
                let mut next = prefill(&f.model, &mut caches)?;
                let mut offset = ids.len();
                for tok in 0..12usize {
                    f.model.refresh_step(&st, next, offset)?;
                    let inp: u32 = st.debug_input()?;
                    let pos: u32 = st.debug_pos()?;
                    graph.launch().map_err(|e| anyhow::anyhow!("graph launch: {e}"))?;
                    let lg_graph: Vec<f32> = logits_out.flatten_all()?.to_vec1()?;
                    let ng = read_next()?;
                    step(&f.model, &mut caches)?;
                    let lg_eager: Vec<f32> = logits_out.flatten_all()?.to_vec1()?;
                    let ne = read_next()?;
                    let maxd = lg_graph
                        .iter()
                        .zip(&lg_eager)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0f32, f32::max);
                    println!(
                        "  diag t{tok:02}: input={inp} pos={pos} · graph→{ng} eager→{ne} \
                         {} · max|Δlogits|={maxd:.3e}",
                        if ng == ne { "==" } else { "≠≠ DIVERGENT" },
                    );
                    next = ne;
                    offset += 1;
                }
                anyhow::bail!("diagnostic done, no timing in DIAG mode");
            }
            // The fallback hybrid: the first decode token eager (it "lands"
            // the state), replay for all the following ones.
            let gen_hybrid = |f: &llvq_llm::fused_cuda::FusedSealed,
                              caches: &mut [llvq_llm::model::KvCache]|
             -> anyhow::Result<Vec<u32>> {
                let mut next = prefill(&f.model, caches)?;
                let mut offset = ids.len();
                let mut out = Vec::with_capacity(n_new);
                loop {
                    out.push(next);
                    if out.len() == n_new {
                        return Ok(out);
                    }
                    f.model.refresh_step(&st, next, offset)?;
                    if out.len() == 1 {
                        step(&f.model, caches)?;
                    } else {
                        graph.launch().map_err(|e| anyhow::anyhow!("graph launch: {e}"))?;
                    }
                    next = read_next()?;
                    offset += 1;
                }
            };
            // Correctness gate BEFORE any timing: pure replay first, hybrid
            // as the fallback, and BOTH verdicts print.
            let g0 = gen_graph(&f, &mut caches)?;
            let pure_ok = first_divergence(&warm, &g0).map_err(|e| anyhow::anyhow!("{e}"))?;
            let use_hybrid = match pure_ok {
                None => {
                    println!("PURE replay gate: {} identical tokens, timing the pure arm", warm.len());
                    false
                }
                Some(i) => {
                    println!("pure replay gate: RED (divergence at token {i}), trying the hybrid");
                    let h0 = gen_hybrid(&f, &mut caches)?;
                    match first_divergence(&warm, &h0).map_err(|e| anyhow::anyhow!("{e}"))? {
                        None => {
                            println!("HYBRID gate: {} identical tokens, timing the hybrid \
                                      (1st token eager, replay after)", warm.len());
                            true
                        }
                        Some(j) => anyhow::bail!(
                            "RED GATE on both arms: pure at token {i}, hybrid at token {j}, \
                             no timing"
                        ),
                    }
                }
            };
            let _ = first;
            let (mut r_e, mut r_g, mut ratios) = (Vec::new(), Vec::new(), Vec::new());
            for round in 0..ROUNDS_TIMED {
                let t = Instant::now();
                let oe = gen_eager(&f, &mut caches)?;
                let te = n_new as f64 / t.elapsed().as_secs_f64();
                let t = Instant::now();
                let og = if use_hybrid {
                    gen_hybrid(&f, &mut caches)?
                } else {
                    gen_graph(&f, &mut caches)?
                };
                let tg = n_new as f64 / t.elapsed().as_secs_f64();
                for (o, name) in [(&oe, "eager"), (&og, "graph")] {
                    if let Some(i) = first_divergence(&warm, o).map_err(|e| anyhow::anyhow!("{e}"))? {
                        anyhow::bail!("round {round}, {name}: divergence at token {i}");
                    }
                }
                r_e.push(te);
                r_g.push(tg);
                ratios.push(tg / te);
            }
            let (me, le, he) = rate_stats(&r_e);
            let (mg, lg2, hg) = rate_stats(&r_g);
            let (mr, lr, hr) = rate_stats(&ratios);
            println!("\n{ROUNDS_TIMED} interleaved round pairs, {} identical tokens everywhere", warm.len());
            println!("  eager (prealloc)  {me:6.1} tok/s [{le:.1}–{he:.1}]");
            println!("  graph (replay)    {mg:6.1} tok/s [{lg2:.1}–{hg:.1}]");
            println!("  r = graph/eager = {mr:.4} [{lr:.4}–{hr:.4}]  (formed round by round)");
            println!("\n  frozen reading (phase prereg 802006c5): end-to-end gain");
            println!("  ≥ 8% → adopted · < 3% → closed · in between: a curve point.");
            println!("  WARNING: net against the v1 config = this gain − 0.83% (fixed-base cost, e1b).");
            return Ok(());
        }

        // ---- A2 step 1: the prealloc-against-cat A/B, in one process ----
        //
        // `LLVQ_KV_AB=1` short-circuits the published protocol: ONE fused
        // model loaded once (config taken from the environment, the v1 frozen
        // for the preregistered job), then INTERLEAVED round pairs where the
        // store switches through `set_kv_store` between two `generate` calls,
        // same weights, same NVRTC translation unit, same process, same
        // prompt. The ratio forms round by round, never as a quotient of
        // medians of separate arms. The only mechanism that moves is the cache
        // storage: the `check_fuse` rule, applied to the axis it protects.
        // Prereg: proofs/preregistration-a2-etape1-prealloc-2026-09-01.md.
        if std::env::var("LLVQ_KV_AB").ok().as_deref() == Some("1") {
            use llvq_llm::kvq::KvStore;
            let w = match kv_store {
                KvStore::Prealloc(w) => w,
                KvStore::Cat => anyhow::bail!(
                    "LLVQ_KV_AB=1 requires LLVQ_KV_PREALLOC=<window>: the A/B compares \
                     cat to the prealloc store, it needs the window"
                ),
            };
            let fuse = llvq_llm::fused::FuseMode::from_env().map_err(|e| anyhow::anyhow!("{e}"))?;
            let t = Instant::now();
            let mut f = llvq_llm::fused_cuda::load_with(&path, &device, dtype, fuse)?;
            println!("loaded in {:.1} s. kv_store A/B: cat against prealloc({w})", t.elapsed().as_secs_f64());
            println!(
                "LLVQ_ROT_SHARE={}, {} rot_lancements/token · LLVQ_FUSE={}, {} matvec_lancements/token",
                f.rot_share.name(), f.rot_launches, f.fuse.name(), f.matvec_launches
            );
            let ids = f
                .tokenizer
                .encode(PROMPT, false)
                .map_err(|e| anyhow::anyhow!("{e}"))?
                .get_ids()
                .to_vec();
            if ids.len() + n_new > w {
                anyhow::bail!(
                    "window {w} < {} + {n_new} tokens: the prealloc arm would die \
                     mid-round, grow LLVQ_KV_PREALLOC",
                    ids.len()
                );
            }
            // One discarded generation PER STORE: the first on CUDA pays
            // kernel selection and clock ramp, and the first prealloc round
            // pays its window allocations.
            let mut reference: Option<Vec<u32>> = None;
            for store in [KvStore::Cat, KvStore::Prealloc(w)] {
                f.model.set_kv_store(store);
                let out = f.model.generate(&ids, n_new, &mut NoCapture)?;
                match &reference {
                    None => reference = Some(out),
                    Some(r) => {
                        if let Some(i) = first_divergence(r, &out).map_err(|e| anyhow::anyhow!("{e}"))? {
                            anyhow::bail!(
                                "warmup: cat/prealloc divergence at token {i}. The A/B is not \
                                 measured on arms that do not return the same tokens"
                            );
                        }
                    }
                }
            }
            let reference = reference.expect("warmup done");
            let (mut r_cat, mut r_pre, mut ratios) = (Vec::new(), Vec::new(), Vec::new());
            for round in 0..ROUNDS_TIMED {
                let mut pair = [0f64; 2];
                for (slot, store) in [KvStore::Cat, KvStore::Prealloc(w)].into_iter().enumerate() {
                    f.model.set_kv_store(store);
                    let t = Instant::now();
                    let out = f.model.generate(&ids, n_new, &mut NoCapture)?;
                    pair[slot] = n_new as f64 / t.elapsed().as_secs_f64();
                    if let Some(i) = first_divergence(&reference, &out).map_err(|e| anyhow::anyhow!("{e}"))? {
                        anyhow::bail!(
                            "round {round}, {}: divergence at token {i} against the reference",
                            store.label()
                        );
                    }
                }
                r_cat.push(pair[0]);
                r_pre.push(pair[1]);
                ratios.push(pair[1] / pair[0]);
            }
            let (mc, lc, hc) = rate_stats(&r_cat);
            let (mp, lp, hp) = rate_stats(&r_pre);
            let (mr, lr, hr) = rate_stats(&ratios);
            println!("\n{ROUNDS_TIMED} interleaved round pairs, {} identical tokens everywhere", reference.len());
            println!("  cat            {mc:6.1} tok/s [{lc:.1}–{hc:.1}]");
            println!("  prealloc({w})  {mp:6.1} tok/s [{lp:.1}–{hp:.1}]");
            println!("  r = prealloc/cat = {mr:.4} [{lr:.4}–{hr:.4}]  (formed round by round)");
            println!("\n  preregistered reading: r ≥ 0.97 → prealloc carries step 2;");
            println!("  r < 0.97 → regression, stop and back to the operator.");
            return Ok(());
        }

        // Which fused arms to run. Normally one, named by `LLVQ_FUSE`; under
        // `LLVQ_FUSE_AB=1` both, `On` then `Off`, each loaded and **dropped**
        // before the next — the card never holds two arms, and the two share a
        // process, an NVRTC translation unit, a card and a prompt. The only
        // thing that moves between them is the number of launches, which is
        // what makes the delta attributable. It costs one extra transcode
        // (~145 s at the 4B).
        let fuse_ab = std::env::var("LLVQ_FUSE_AB").ok().as_deref() == Some("1");
        let modes: Vec<llvq_llm::fused::FuseMode> = if fuse_ab {
            vec![llvq_llm::fused::FuseMode::On, llvq_llm::fused::FuseMode::Off]
        } else {
            vec![llvq_llm::fused::FuseMode::from_env().map_err(|e| anyhow::anyhow!("{e}"))?]
        };

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
            let mut f = llvq_llm::fused_cuda::load_with(&path, &device, dtype, fuse)?;
            f.model.set_kv_store(kv_store);
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
            // The trailing "projections … b/weight" line reports the FIRST arm,
            // which is the fused one under `LLVQ_FUSE_AB=1`. Letting the loop
            // overwrite it would print the control arm's accounting under the
            // fused arm's heading — the two differ by `gs_off`, 3.69 MB on the
            // 4B, and that is exactly the term this lot adds.
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
                kv_store.label()
            );
            if tok.is_none() {
                tok = Some(f.tokenizer.clone());
            }
            if time_phases {
                let (_, report) = f.model.generate_phased(&ids, PHASE_TOKENS)?;
                print_phases(&format!("fused F{}", f.fuse.name()), &report);
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
            let mut m = llvq_llm::sealed::load(&path, dtype, &device, llvq_llm::kvq::KvMode::F16)?;
            m.model.set_kv_store(kv_store);
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
            println!("         kv_store={}", kv_store.label());
            if time_phases {
                let (_, report) = m.model.generate_phased(&ids, PHASE_TOKENS)?;
                print_phases("dense", &report);
            }
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
}

#[cfg(test)]
mod tests {
    use super::{first_divergence, rate_stats};

    #[test]
    fn median_is_the_middle_rate_odd_and_even() {
        let (m, lo, hi) = rate_stats(&[3.0, 1.0, 2.0]);
        assert_eq!((m, lo, hi), (2.0, 1.0, 3.0));
        let (m, lo, hi) = rate_stats(&[4.0, 1.0, 2.0, 3.0]);
        assert_eq!((m, lo, hi), (2.5, 1.0, 4.0));
    }

    /// The comparison refuses arms of different lengths instead of walking
    /// their common prefix and printing a tick.
    ///
    /// The hole this closes is real and predates the lot: `zip` truncates, so
    /// an arm that returned fewer tokens compared green over the prefix. The
    /// assertion is on the *refusal*, because the truncating version passes
    /// every "identical tokens" test there is.
    #[test]
    fn a_short_arm_is_refused_rather_than_compared_on_its_prefix() {
        assert_eq!(first_divergence(&[1, 2, 3], &[1, 2, 3]), Ok(None));
        assert_eq!(first_divergence(&[1, 2, 3], &[1, 9, 3]), Ok(Some(1)));
        assert_eq!(first_divergence(&[9, 2], &[1, 2]), Ok(Some(0)));
        let e = first_divergence(&[1, 2, 3], &[1, 2]).expect_err("unequal lengths");
        assert!(e.contains('3') && e.contains('2'), "{e}");
        assert_eq!(first_divergence(&[], &[]), Ok(None));
    }
}
