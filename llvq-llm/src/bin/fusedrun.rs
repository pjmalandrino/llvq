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

// Unconditional since the mode functions the Mac type-checks use them too.
use std::time::Instant;
use llvq_llm::model::NoCapture;

#[cfg(all(target_os = "linux", feature = "cuda"))]
const PROMPT: &str = "The capital of France is";

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

/// Median and min–max of per-round rates. Not cfg-gated: the Mac must
/// type-check what a paid image build would otherwise be the first to compile.
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
/// Not cfg-gated: a Mac cannot type-check gated code, and this struct's shape
/// is exactly what an image build would otherwise be the first to compile.
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

/// The prompt lengths of `LLVQ_CONTEXTS`, comma-separated, each above zero.
#[cfg_attr(not(all(target_os = "linux", feature = "cuda")), allow(dead_code))]
fn parse_contexts(v: &str) -> Result<Vec<usize>, String> {
    parse_lengths("LLVQ_CONTEXTS", v)
}

/// Prompt lengths, comma-separated, each above zero, read from `var`.
#[cfg_attr(not(all(target_os = "linux", feature = "cuda")), allow(dead_code))]
fn parse_lengths(var: &str, v: &str) -> Result<Vec<usize>, String> {
    let lengths = v
        .split(',')
        .map(|s| s.trim().parse::<usize>())
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| format!("{var}={v:?}: {e}"))?;
    if lengths.is_empty() || lengths.contains(&0) {
        return Err(format!("{var}={v:?}: every prompt length must be above zero"));
    }
    Ok(lengths)
}

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

/// The arms of `LLVQ_GQA_AB`: the published attention (`Cat` store, expanded
/// heads), and the two forms that read the cached heads unexpanded from a
/// store grown in place by [`llvq_llm::kvq::GROW_CHUNK`] positions.
#[cfg_attr(not(all(target_os = "linux", feature = "cuda")), allow(dead_code))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GqaArm {
    Published,
    Shared,
    Grouped,
    /// Decode and verify rows through the split-KV attention kernel
    /// (`llvq_llm::attn`), longer calls through the grouped path. Not in
    /// [`GqaArm::ALL`]: only `LLVQ_GQA_PPL`'s kernel mode runs it, which
    /// installs the kernel first.
    Kernel,
}

#[cfg_attr(not(all(target_os = "linux", feature = "cuda")), allow(dead_code))]
impl GqaArm {
    const ALL: [GqaArm; 3] = [GqaArm::Published, GqaArm::Shared, GqaArm::Grouped];

    fn store(self) -> llvq_llm::kvq::KvStore {
        match self {
            GqaArm::Published => llvq_llm::kvq::KvStore::Cat,
            GqaArm::Shared | GqaArm::Grouped | GqaArm::Kernel => {
                llvq_llm::kvq::KvStore::Grow(llvq_llm::kvq::GROW_CHUNK)
            }
        }
    }

    fn heads(self) -> llvq_llm::kvq::AttnHeads {
        match self {
            GqaArm::Published => llvq_llm::kvq::AttnHeads::Expanded,
            GqaArm::Shared => llvq_llm::kvq::AttnHeads::Shared,
            GqaArm::Grouped => llvq_llm::kvq::AttnHeads::Grouped,
            GqaArm::Kernel => llvq_llm::kvq::AttnHeads::Kernel,
        }
    }

    /// What every line of the table prints: the store and the form, as set.
    fn name(self) -> String {
        format!("{}+{}", self.store().label(), self.heads().label())
    }

    fn apply(self, m: &mut llvq_llm::model::Qwen3) {
        m.set_kv_store(self.store());
        m.set_attn_heads(self.heads());
    }
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

/// The measurement modes and prompt variables of this binary that run alone.
///
/// One table replaces the eight lists each mode used to spell out, which had
/// drifted apart: one mode listed itself, so `=1` always failed, and the
/// modes dispatched last returned before reading the ones dispatched after
/// them, so a mode beside `LLVQ_GQA_PPL` was silently ignored. A mode calls
/// [`runs_alone`]: every OTHER name here, set beside it, is refused.
#[cfg_attr(not(all(target_os = "linux", feature = "cuda")), allow(dead_code))]
const ALONE: [&str; 7] = [
    "LLVQ_INT4_WIDE_CHECK", "LLVQ_PERS_4E", "LLVQ_GQA_PPL", "LLVQ_GQA_AB", "LLVQ_CONTEXTS", "LLVQ_PREFILL_TOKENS",
    "LLVQ_SEG_ARMS",
];

/// `mode` against every other name of [`ALONE`] and the mode's own `extra`
/// names, with the environment handed in so a test can drive it.
#[cfg_attr(not(all(target_os = "linux", feature = "cuda")), allow(dead_code))]
fn alone_in(mode: &str, extra: &[&str], lookup: impl Fn(&str) -> Option<String>) -> Result<(), String> {
    debug_assert!(ALONE.contains(&mode), "{mode} is not in ALONE");
    for other in ALONE.iter().chain(extra).filter(|o| **o != mode) {
        if let Some(o) = lookup(other) {
            return Err(format!("{mode} runs alone, and {other}={o:?} is set beside it"));
        }
    }
    Ok(())
}

/// [`alone_in`] on the real environment.
#[cfg_attr(not(all(target_os = "linux", feature = "cuda")), allow(dead_code))]
fn runs_alone(mode: &str, extra: &[&str]) -> anyhow::Result<()> {
    alone_in(mode, extra, |v| std::env::var(v).ok()).map_err(anyhow::Error::msg)
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
        // A variable whose code is gone fails here, by name, instead of
        // leaving an old command to measure the plain bench.
        llvq_llm::served::check_removed(|v| std::env::var(v).ok()).map_err(anyhow::Error::msg)?;
        // Resolved once, here: one read of the outside world, at the top, so
        // no path below can pick up a different answer. `None` is the bench;
        // `Some` is the runner.
        let served = llvq_llm::served::Served::from_env().map_err(|e| anyhow::anyhow!("{e}"))?;
        // The served path takes no measurement mode. Each of these turns this
        // binary into a bench of one thing, and a bench needs its arms named
        // by variables — that is what `LLVQ_CONFIG` exists to replace. Refused
        // by name rather than silently dropped, which is what happened before
        // the audit of 2026-09-11 found it. `LLVQ_PREFILL_TOKENS` and
        // `LLVQ_CONTEXTS` are the two exceptions, handled below: they time the
        // served object as served, and change only the prompt it is given.
        // `LLVQ_TIME_EVENTS` and `LLVQ_BENCH_ARMS` are read by `planesbench`
        // only, so this binary lets them pass.
        if served.is_some() {
            for var in llvq_llm::served::MEASUREMENT_MODES
                .into_iter()
                .filter(|v| !["LLVQ_TIME_EVENTS", "LLVQ_BENCH_ARMS"].contains(v))
            {
                if let Ok(v) = std::env::var(var) {
                    anyhow::bail!(
                        "{var}={v:?} beside LLVQ_CONFIG: that is a measurement mode, and the \
                         served path runs one arm with no comparison to name. Unset it, or \
                         unset LLVQ_CONFIG and run the bench."
                    );
                }
            }
        }
        println!("{device:?}, dtype {dtype:?}, {n_new} tokens\n");

        // ---- LLVQ_INT4_WIDE_CHECK: the f16-staged int4 kernel, bytes ----
        //
        // A MEASUREMENT MODE on the bench door, refused beside LLVQ_CONFIG
        // (above) and beside every other mode. One load, then
        // `FusedRuntime::int4_wide_check`: `tv_q4_hs` against `tv_q4_hh` on the
        // model's first int4 record, and against the host's sha256 on the
        // synthetic records of `fused::INT4_WIDE_CARD_CASES`
        // (`prereg-bigmodel-2`, step 4). No token is decoded or timed. The
        // check runs inside the load and fails it on any difference.
        if std::env::var("LLVQ_INT4_WIDE_CHECK").ok().as_deref() == Some("1") {
            runs_alone("LLVQ_INT4_WIDE_CHECK", &[])?;
            let fuse = llvq_llm::fused::FuseMode::from_env().map_err(|e| anyhow::anyhow!("{e}"))?;
            llvq_llm::fused_cuda::set_int4_wide_check(true);
            let t = Instant::now();
            let loaded = llvq_llm::fused_cuda::load_with(&path, &device, dtype, fuse);
            llvq_llm::fused_cuda::set_int4_wide_check(false);
            let _f = loaded?;
            println!(
                "LLVQ_INT4_WIDE_CHECK: load and checks in {:.1} s; no token decoded",
                t.elapsed().as_secs_f64()
            );
            return Ok(());
        }
        if let Ok(v) = std::env::var("LLVQ_INT4_WIDE_CHECK") {
            anyhow::bail!("LLVQ_INT4_WIDE_CHECK={v:?}: the only value is 1");
        }
        // ---- LLVQ_PERS_4E: the whole token in one launch, against the eager path ----
        //
        // A MEASUREMENT MODE on the bench door, refused beside LLVQ_CONFIG and
        // every other mode, the served flags spelled out on the command line.
        // ONE model on Grow(256) with the "kernel" attention form. At 8,192
        // prompt tokens the served config itself (the kernel from position
        // 1,280, every decode row past it); at the short prompt, every decode
        // row on the kernel (threshold 0, measurement only), since 4E does
        // not compute the grouped arithmetic the served config runs below
        // 1,280. Per context: the eager path makes the reference tokens; each
        // 4E geometry (512 and 768 threads, llvq_llm::persist4e_cuda), after
        // the same eager prefill, must make the same ones, or the run is void.
        // The served door's tokens are printed beside. Then 2 + 5 interleaved
        // rounds of the decode, eager and every geometry, tok/s and the
        // ratios round by round, then one stamped launch a geometry. Value:
        // the tokens a run, 2 to 256.
        if let Ok(v) = std::env::var("LLVQ_PERS_4E") {
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
            // pers::P4E_BENCH_FORMS, v4 (the served form), v6, v7, v8 and
            // v9, then v6s. Each owes the eager path's tokens.
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
            return Ok(());
        }

        // ---- LLVQ_GQA_PPL: the grouped form's perplexity, prefill and decode ----
        //
        // A MEASUREMENT MODE on the bench door, refused beside LLVQ_CONFIG, the
        // served flags spelled out on the command line. The value is
        // `<ctx>,<windows>` (grouped against published) or
        // `<ctx>,<windows>,<first>,kernel` (kernel against grouped, from window
        // `<first>`, the kernel installed and checked on the card first): see
        // `parse_gqa_ppl`. ONE model, two arms (`run_gqa_ppl`), the three trims
        // in force as in LLVQ_GQA_AB. The first form puts every window through
        // the prefill in one call, so it is bounded by the fused prefill's rows.
        if let Ok(v) = std::env::var("LLVQ_GQA_PPL") {
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
            return Ok(());
        }

        // ---- LLVQ_GQA_AB: attention over the cached heads, unexpanded ----
        //
        // A MEASUREMENT MODE on the bench door, refused beside LLVQ_CONFIG: the
        // served flags are spelled out on the command line. The value is the
        // list of prompt lengths. ONE model, three arms (`run_gqa_ab`), with the
        // three trims of 2026-10-30 in force: cudarc's events go off here as on
        // the served door, which is safe on one stream only, so that is checked.
        if let Ok(v) = std::env::var("LLVQ_GQA_AB") {
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
            return Ok(());
        }

        // ---- LLVQ_CONTEXTS: the served speed at the product's context ----
        //
        // Every published tok/s was measured after a 5-token prompt, so at
        // most 261 positions of cache; the product promises 8,192. This times,
        // for each prompt length named, the first token and the decode after
        // it (`run_context_sweep`). Through the served door when there is a
        // config, like `LLVQ_PREFILL_TOKENS`, so the object timed is the
        // object served, down to its attention form. A PREALLOCATED window is
        // refused, because a fixed window would bound the prompt this mode
        // exists to lengthen; the store a served config asks for is not
        // refused, and `Grow` promises no window.
        if let Ok(v) = std::env::var("LLVQ_CONTEXTS") {
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
            return Ok(());
        }

        // ---- LLVQ_PREFILL_TOKENS: what a prompt costs through the kernel ----
        //
        // A MEASUREMENT MODE, never a served config. Unset, this binary is
        // byte-identical to the published protocol.
        //
        // It answers one question and no other: how long does the fused path
        // take to swallow N tokens? That is what decides whether a real MMLU
        // census can run through the kernel at all. The census scores 14,042
        // 5-shot questions of several hundred tokens each, and the dense path
        // does the whole thing in 26 minutes because it issues ONE matmul a
        // projection a question. The kernel issues `ceil(N / PREFILL_ROWS)`,
        // and this prints the constant that multiplies.
        //
        // No dense arm, no generation, no comparison: a number and its spread.
        if let Ok(v) = std::env::var("LLVQ_PREFILL_TOKENS") {
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
            return Ok(());
        }

        // ---- LLVQ_CONFIG: the served path, and nothing beside it ----------
        //
        // The runner, as opposed to the bench. Everything below this block
        // exists to compare arms: two `LLVQ_FUSE` modes, and a DENSE arm that
        // loads 8.04 GB of f16 weights so the fused tokens have something to
        // be identical to. That is the right shape for a measurement and the
        // wrong shape for serving — an object that fits in 1.39 GB should not
        // need six times itself on the card to start.
        //
        // So: with a served config, one arm, no dense reference, no ratio.
        // Without one, this binary is byte for byte the bench it has always
        // been, and every number in `docs/mesures/` keeps the protocol that
        // produced it.
        //
        // The correctness of this path is not asserted here and it would be
        // dishonest to imply it: it is asserted by the bench, on the same
        // file, where a dense arm exists to disagree.
        if let Some(cfg) = &served {
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
            return Ok(());
        }

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
}

#[cfg(test)]
mod tests {
    use super::{alone_in, first_divergence, paired_ppl_pct, parse_gqa_ppl, rate_stats, t95, GqaArm, GqaPplSpec, ALONE};

    fn env_of<'a>(set: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
        move |v| set.iter().find(|(k, _)| *k == v).map(|(_, x)| x.to_string())
    }

    /// A mode set alone passes its own check. One mode failed this from
    /// 2026-11-04 to 2026-11-07, when its list named itself.
    #[test]
    fn no_mode_refuses_itself() {
        for mode in ALONE {
            alone_in(mode, &["LLVQ_KERNEL_DIR"], env_of(&[(mode, "1")]))
                .unwrap_or_else(|e| panic!("{mode} alone: {e}"));
        }
    }

    /// Every pair is refused both ways, naming both variables, whatever the
    /// order the modes are dispatched in.
    #[test]
    fn every_pair_is_refused_both_ways() {
        for mode in ALONE {
            for other in ALONE.iter().filter(|o| **o != mode) {
                let e = alone_in(mode, &[], env_of(&[(mode, "1"), (other, "1")]))
                    .expect_err(&format!("{mode} beside {other}"));
                assert!(e.contains(mode) && e.contains(other), "{e}");
            }
        }
        // The hole the eight lists left: GQA_PPL returned before the modes
        // dispatched after it were read.
        assert!(alone_in("LLVQ_GQA_PPL", &[], env_of(&[("LLVQ_PREFILL_TOKENS", "1")])).is_err());
    }

    /// The extra names are refused beside their mode only, and an unrelated
    /// variable is not.
    #[test]
    fn extras_belong_to_their_mode() {
        let dir = [("LLVQ_KERNEL_DIR", "/tmp/k")];
        assert!(alone_in("LLVQ_PERS_4E", &["LLVQ_KERNEL_DIR"], env_of(&dir)).is_err());
        alone_in("LLVQ_GQA_AB", &[], env_of(&dir)).expect("GQA_AB takes a kernel directory");
        alone_in("LLVQ_GQA_AB", &[], env_of(&[("LLVQ_CONFIG", "x.json")])).expect("not a mode");
    }

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
