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

// Unconditional since `run_trim_ab` and `run_context_sweep`, which the Mac
// type-checks, use them too.
use std::time::Instant;
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

/// The second chunking `LLVQ_CONTEXTS` reads a kernel-prefill prompt in, to
/// check its logits do not depend on the chunking.
#[cfg_attr(not(all(target_os = "linux", feature = "cuda")), allow(dead_code))]
const PREFILL_PROBE_ROWS: usize = 512;

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

/// The arms of `LLVQ_TRIM_AB`: the trimmed decode step, each of its three
/// trims undone alone, and all three undone at once.
///
/// The three trims of 2026-10-30 are the all-zero decode mask skipped, the
/// int4 activation read in f16 by `tv_q4_hh` instead of cast for `tv_q4_h`,
/// and cudarc's per-allocation events turned off. Each is bit-identical by
/// design, so every arm must return the same tokens. Each single-trim arm
/// moves one mechanism against `trimmed` (the `check_fuse` rule); `untrimmed`
/// is the path the published numbers were measured on.
#[cfg_attr(not(all(target_os = "linux", feature = "cuda")), allow(dead_code))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TrimArm {
    Trimmed,
    Mask,
    Cast,
    Events,
    Untrimmed,
}

#[cfg_attr(not(all(target_os = "linux", feature = "cuda")), allow(dead_code))]
impl TrimArm {
    const ALL: [TrimArm; 5] =
        [TrimArm::Trimmed, TrimArm::Mask, TrimArm::Cast, TrimArm::Events, TrimArm::Untrimmed];

    fn name(self) -> &'static str {
        match self {
            TrimArm::Trimmed => "trimmed",
            TrimArm::Mask => "+mask",
            TrimArm::Cast => "+cast",
            TrimArm::Events => "+events",
            TrimArm::Untrimmed => "untrimmed",
        }
    }

    /// Builds and adds the all-zero decode mask again.
    fn mask(self) -> bool {
        matches!(self, TrimArm::Mask | TrimArm::Untrimmed)
    }

    /// Casts the int4 activation to f32 for `tv_q4_h` again.
    fn cast(self) -> bool {
        matches!(self, TrimArm::Cast | TrimArm::Untrimmed)
    }

    /// Turns cudarc's per-allocation events back on.
    fn events(self) -> bool {
        matches!(self, TrimArm::Events | TrimArm::Untrimmed)
    }
}

/// `LLVQ_TRIM_AB`: one model, every arm in every round, same order, two
/// rounds discarded then [`ROUNDS_TIMED`]; each ratio formed round by round
/// against `trimmed`. `set` puts the model and the runtime in an arm's state.
/// Any token that differs between two arms or two rounds stops the run before
/// a number is printed: the trims claim bit identity, and a speed measured on
/// arms that disagree is not a speed of the same thing.
///
/// Out of the cfg-gated body so the Mac type-checks it.
#[cfg_attr(not(all(target_os = "linux", feature = "cuda")), allow(dead_code))]
fn run_trim_ab(
    model: &mut llvq_llm::model::Qwen3,
    ids: &[u32],
    n_new: usize,
    mut set: impl FnMut(&mut llvq_llm::model::Qwen3, TrimArm) -> anyhow::Result<()>,
) -> anyhow::Result<()> {
    const WARMUP: usize = 2;
    let mut reference: Option<Vec<u32>> = None;
    let mut rates: Vec<Vec<f64>> = vec![Vec::new(); TrimArm::ALL.len()];
    for round in 0..WARMUP + ROUNDS_TIMED {
        for (k, arm) in TrimArm::ALL.into_iter().enumerate() {
            set(model, arm)?;
            let t = Instant::now();
            let out = model.generate(ids, n_new, &mut NoCapture)?;
            let rate = n_new as f64 / t.elapsed().as_secs_f64();
            match &reference {
                None => reference = Some(out),
                Some(r) => {
                    if let Some(i) = first_divergence(r, &out).map_err(|e| anyhow::anyhow!("{e}"))? {
                        anyhow::bail!(
                            "round {round}, arm {}: divergence at token {i}. The trims are \
                             bit-identical by design, and arms that disagree void the run",
                            arm.name()
                        );
                    }
                }
            }
            if round >= WARMUP {
                rates[k].push(rate);
            }
        }
    }
    set(model, TrimArm::Trimmed)?;
    let n_tokens = reference.map_or(0, |r| r.len());
    println!(
        "\n{} arms × {ROUNDS_TIMED} interleaved rounds after {WARMUP} discarded, \
         {n_tokens} identical tokens in every arm and round",
        TrimArm::ALL.len()
    );
    println!("  arm         tok/s [range]           trimmed / arm, round by round");
    for (k, arm) in TrimArm::ALL.into_iter().enumerate() {
        let (m, lo, hi) = rate_stats(&rates[k]);
        if k == 0 {
            println!("  {:<10}  {m:6.1} [{lo:.1}–{hi:.1}]", arm.name());
            continue;
        }
        let ratios: Vec<f64> = rates[0].iter().zip(&rates[k]).map(|(a, b)| a / b).collect();
        let (mr, lr, hr) = rate_stats(&ratios);
        println!(
            "  {:<10}  {m:6.1} [{lo:.1}–{hi:.1}]    {mr:.4} [{lr:.4}–{hr:.4}]",
            arm.name()
        );
    }
    Ok(())
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
    let h = llvq_llm::eval::t95(d.len().saturating_sub(1)) * sd / n.sqrt();
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

/// The arms of `LLVQ_GESTURE_AB`, Part A of the gestures prereg: the token
/// boundary of the decode loop, one mechanism an arm against its neighbour
/// (`llvq_llm::model::TokenLoop`). `Base` is `generate` itself, untouched.
#[cfg_attr(not(all(target_os = "linux", feature = "cuda")), allow(dead_code))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GestureArm {
    Base,
    F16,
    Key,
    OnDevKey,
    OnDevCandle,
}

#[cfg_attr(not(all(target_os = "linux", feature = "cuda")), allow(dead_code))]
impl GestureArm {
    const ALL: [GestureArm; 5] =
        [GestureArm::Base, GestureArm::F16, GestureArm::Key, GestureArm::OnDevKey, GestureArm::OnDevCandle];

    fn name(self) -> &'static str {
        match self {
            GestureArm::Base => "base",
            GestureArm::F16 => "f16",
            GestureArm::Key => "key",
            GestureArm::OnDevKey => "ondev-key",
            GestureArm::OnDevCandle => "ondev-candle",
        }
    }

    fn run(
        self,
        m: &llvq_llm::model::Qwen3,
        ids: &[u32],
        n: usize,
        key: &dyn llvq_llm::model::KeyKernel,
    ) -> candle_core::Result<Vec<u32>> {
        use llvq_llm::model::TokenLoop;
        match self {
            GestureArm::Base => m.generate(ids, n, &mut NoCapture),
            GestureArm::F16 => m.generate_loop(ids, n, &TokenLoop::F16),
            GestureArm::Key => m.generate_loop(ids, n, &TokenLoop::Key(key)),
            GestureArm::OnDevKey => m.generate_loop(ids, n, &TokenLoop::OnDevKey(key)),
            GestureArm::OnDevCandle => m.generate_loop(ids, n, &TokenLoop::OnDevCandle),
        }
    }
}

/// The ratios of `LLVQ_GESTURE_AB`, `numerator / denominator` in tok/s,
/// formed round by round. The first four move one mechanism each and are the
/// ones the decision rules read; the last moves three and decides nothing.
#[cfg_attr(not(all(target_os = "linux", feature = "cuda")), allow(dead_code))]
const GESTURE_RATIOS: [(GestureArm, GestureArm, &str); 5] = [
    (GestureArm::F16, GestureArm::Base, "a1, the cast"),
    (GestureArm::Key, GestureArm::F16, "a2, the key kernel"),
    (GestureArm::OnDevKey, GestureArm::Key, "a3, readback and upload, key kernel"),
    (GestureArm::OnDevCandle, GestureArm::F16, "a3, readback and upload, candle's argmax"),
    (GestureArm::OnDevKey, GestureArm::Base, "three mechanisms, reported only"),
];

/// `LLVQ_GESTURE_AB`, the arms: one model, every arm in every round, same
/// order, two rounds discarded then [`ROUNDS_TIMED`], the `run_trim_ab`
/// protocol. Any token that differs from `Base` in any arm or round stops the
/// run before a number is printed: every gesture claims bit identity.
/// Returns `Base`'s tokens, the reference of the clocks.
///
/// Out of the cfg-gated body so the Mac type-checks it.
#[cfg_attr(not(all(target_os = "linux", feature = "cuda")), allow(dead_code))]
fn run_gesture_arms(
    m: &llvq_llm::model::Qwen3,
    ids: &[u32],
    n_new: usize,
    key: &dyn llvq_llm::model::KeyKernel,
) -> anyhow::Result<Vec<u32>> {
    const WARMUP: usize = 2;
    let mut reference: Option<Vec<u32>> = None;
    let mut rates: Vec<Vec<f64>> = vec![Vec::new(); GestureArm::ALL.len()];
    for round in 0..WARMUP + ROUNDS_TIMED {
        for (k, arm) in GestureArm::ALL.into_iter().enumerate() {
            let t = Instant::now();
            let out = arm.run(m, ids, n_new, key)?;
            let rate = n_new as f64 / t.elapsed().as_secs_f64();
            match &reference {
                None => reference = Some(out),
                Some(r) => {
                    if let Some(i) = first_divergence(r, &out).map_err(|e| anyhow::anyhow!("{e}"))? {
                        anyhow::bail!(
                            "round {round}, arm {}: divergence at token {i}. Every gesture claims bit \
                             identity, and arms that disagree void the run",
                            arm.name()
                        );
                    }
                }
            }
            if round >= WARMUP {
                rates[k].push(rate);
            }
        }
    }
    let reference = reference.expect("at least one round ran");
    println!(
        "\n{} arms × {ROUNDS_TIMED} interleaved rounds after {WARMUP} discarded, {} identical tokens \
         in every arm and round",
        GestureArm::ALL.len(),
        reference.len()
    );
    println!("  arm            tok/s [range]");
    for (k, arm) in GestureArm::ALL.into_iter().enumerate() {
        let (m, lo, hi) = rate_stats(&rates[k]);
        println!("  {:<13}  {m:6.1} [{lo:.1}–{hi:.1}]", arm.name());
    }
    println!("  ratio, round by round                    median [range]");
    let at = |a: GestureArm| GestureArm::ALL.iter().position(|&x| x == a).expect("every arm is in ALL");
    for (num, den, what) in GESTURE_RATIOS {
        let r: Vec<f64> = rates[at(num)].iter().zip(&rates[at(den)]).map(|(a, b)| a / b).collect();
        let (mr, lr, hr) = rate_stats(&r);
        println!(
            "  {:>12} / {:<5}  {mr:.4} [{lr:.4}–{hr:.4}]   {what}",
            num.name(),
            den.name()
        );
    }
    Ok(reference)
}

/// Decode steps of each pass of the `LLVQ_GESTURE_AB` clocks.
#[cfg_attr(not(all(target_os = "linux", feature = "cuda")), allow(dead_code))]
const CLOCK_TOKENS: usize = 64;

/// `E_base` and `E_ondev`, ms a decode step: `(T(64) − T(1)) / 63` for the
/// `Base` and `OnDevKey` arms, each pass ending on its readback, five passes
/// interleaved, median and range. The difference removes the prefill and the
/// first head, so what is left is 63 decode steps of each arm as it runs.
#[cfg_attr(not(all(target_os = "linux", feature = "cuda")), allow(dead_code))]
fn decode_step_clock(
    m: &llvq_llm::model::Qwen3,
    ids: &[u32],
    key: &dyn llvq_llm::model::KeyKernel,
    reference: &[u32],
) -> anyhow::Result<[(f64, f64, f64); 2]> {
    let arms = [GestureArm::Base, GestureArm::OnDevKey];
    let mut e: [Vec<f64>; 2] = [Vec::new(), Vec::new()];
    for _ in 0..ROUNDS_TIMED {
        for (k, arm) in arms.into_iter().enumerate() {
            let t = Instant::now();
            let long = arm.run(m, ids, CLOCK_TOKENS, key)?;
            let t_long = t.elapsed().as_secs_f64();
            let t = Instant::now();
            let short = arm.run(m, ids, 1, key)?;
            let t_short = t.elapsed().as_secs_f64();
            for (got, n) in [(&long, CLOCK_TOKENS), (&short, 1)] {
                if let Some(i) = first_divergence(&reference[..n], got).map_err(|e| anyhow::anyhow!("{e}"))? {
                    anyhow::bail!("clock, arm {}, {n} tokens: divergence at token {i}", arm.name());
                }
            }
            e[k].push((t_long - t_short) * 1e3 / (CLOCK_TOKENS - 1) as f64);
        }
    }
    Ok([rate_stats(&e[0]), rate_stats(&e[1])])
}

/// `LLVQ_GESTURE_AB` on the card, in order, each step gating the next: P1 the
/// key's load check, P2 the card mutants, P3 the kernel times, P4 the
/// pageable-copy probe, the arms, then the clocks.
#[cfg(all(target_os = "linux", feature = "cuda"))]
fn run_gesture_mode(
    m: &llvq_llm::model::Qwen3,
    device: &candle_core::Device,
    ids: &[u32],
    n_new: usize,
) -> anyhow::Result<()> {
    use llvq_llm::gesture_cuda::{kill_card_mutants, GestureProbe, KeyArgmax, P4_SPIN_NS};

    let key = KeyArgmax::new(device)?;
    println!("tok_argmax_h: own NVRTC module, source sha256 {}", key.sha256);
    match key.check(device)? {
        Ok(line) => println!("P1 {line}"),
        Err(why) => anyhow::bail!("P1, the key's load check: {why}. The run is void"),
    }
    let killed = kill_card_mutants(device)?;
    println!(
        "P2 {} of {} card mutants killed on the card, each edit found exactly once",
        killed.len(),
        llvq_llm::argmax_key::KERNEL_MUTANTS.len()
    );
    for line in &killed {
        println!("{line}");
    }

    let probe = GestureProbe::new(device)?;
    let logits = {
        let mut caches = m.fresh_caches();
        let input = candle_core::Tensor::from_slice(ids, (1, ids.len()), device)?;
        let h = m.hidden_cached(&input, 0, &mut caches, &mut NoCapture)?;
        m.logits_last_native(&h)?
    };
    let kt = probe.kernel_times(device, &key, &logits)?;
    println!(
        "P3 kernel time, us a launch ({} x {} launches, median): argmax_f16 {:.2}, argmax_f32 {:.2}, \
         cast {:.2}, key {:.2}, candle argmax call {:.2}",
        llvq_llm::gesture_cuda::PROBE_REPS,
        llvq_llm::gesture_cuda::PROBE_BATCH,
        kt.argmax_f16_direct,
        kt.argmax_f32_direct,
        kt.cast,
        kt.key,
        kt.candle_argmax_call
    );
    let ct = probe.copy_times(device, &logits)?;
    let power = if ct.to_scalar > 4.0 {
        "the control waited"
    } else {
        "the control did NOT wait: P4 has no power, its readings are not used"
    };
    println!(
        "P4 host time behind a {} ms spin, ms (median of {}): from_slice {:.3}, candle argmax {:.3}, \
         to_scalar {:.3} ({power})",
        P4_SPIN_NS / 1_000_000,
        llvq_llm::gesture_cuda::PROBE_REPS,
        ct.from_slice,
        ct.candle_argmax,
        ct.to_scalar
    );

    let reference = run_gesture_arms(m, ids, n_new, &key)?;

    let [eb, eo] = decode_step_clock(m, ids, &key, &reference)?;
    println!(
        "\nclock E, ms a decode step ((T({CLOCK_TOKENS}) - T(1)) / {}, {ROUNDS_TIMED} passes): \
         E_base {:.3} [{:.3}–{:.3}], E_ondev {:.3} [{:.3}–{:.3}], E_base - E_ondev {:.3}",
        CLOCK_TOKENS - 1,
        eb.0, eb.1, eb.2, eo.0, eo.1, eo.2,
        eb.0 - eo.0
    );
    spun_clock(m, device, ids, &key, &probe, &reference, eo.0)
}

/// The spun clock: the `OnDevKey` step cut in segments, each enqueued behind
/// a 5 ms spin, so the host's enqueue span H and the device's work D are
/// read apart. A step counts only if every segment was enqueued at least
/// 1 ms before its spin could end. No pageable copy happens inside a step.
#[cfg(all(target_os = "linux", feature = "cuda"))]
fn spun_clock(
    m: &llvq_llm::model::Qwen3,
    device: &candle_core::Device,
    ids: &[u32],
    key: &llvq_llm::gesture_cuda::KeyArgmax,
    probe: &llvq_llm::gesture_cuda::GestureProbe,
    reference: &[u32],
    e_ondev: f64,
) -> anyhow::Result<()> {
    use candle_core::{DType, Tensor};
    use llvq_llm::model::KeyKernel;
    const SPIN_NS: u64 = 5_000_000;
    let layers = m.config().num_hidden_layers;
    let at: Vec<usize> = (1..6).map(|k| k * layers / 6).collect();
    let (mut hs, mut ds, mut slack) = (Vec::new(), Vec::new(), Vec::new());
    let (mut valid, mut total) = (0usize, 0usize);
    for _ in 0..ROUNDS_TIMED {
        let mut caches = m.fresh_caches();
        key.arm()?;
        let ring = Tensor::zeros(CLOCK_TOKENS, DType::U32, device)?;
        let input = Tensor::from_slice(ids, (1, ids.len()), device)?;
        let h = m.hidden_cached(&input, 0, &mut caches, &mut NoCapture)?;
        key.write(&m.logits_last_native(&h)?, &ring, 0)?;
        probe.sync()?;
        let mut cap = llvq_llm::gesture_cuda::SpinCapture::new(probe, SPIN_NS, at.clone())?;
        let last = at.len();
        for (pos, offset) in (1..CLOCK_TOKENS).zip(ids.len()..) {
            cap.open(0)?;
            let next = ring.narrow(0, pos - 1, 1)?.reshape((1, 1))?;
            let h = m.hidden_cached(&next, offset, &mut caches, &mut cap)?;
            key.write(&m.logits_last_native(&h)?, &ring, pos)?;
            cap.close(last)?;
            probe.sync()?;
            let (h_ms, d_ms, ok) = cap.read()?;
            total += 1;
            if ok {
                valid += 1;
                hs.push(h_ms);
                ds.push(d_ms);
                slack.push(d_ms - h_ms);
            }
        }
        let tokens = ring.to_vec1::<u32>()?;
        if key.nan_seen()? {
            anyhow::bail!("spun clock: a NaN logit");
        }
        if let Some(i) = first_divergence(&reference[..CLOCK_TOKENS], &tokens).map_err(|e| anyhow::anyhow!("{e}"))? {
            anyhow::bail!("spun clock: divergence at token {i} from the base arm");
        }
    }
    println!(
        "clock H/D: {valid} of {total} spun steps valid ({:.1} %), {} segments a step after ln1 of \
         layers {at:?}, {} ms spins",
        100.0 * valid as f64 / total as f64,
        at.len() + 1,
        SPIN_NS / 1_000_000
    );
    if valid == 0 {
        println!("  no valid step: H and D are not measured");
        return Ok(());
    }
    let (h, hl, hh) = rate_stats(&hs);
    let (d, dl, dh) = rate_stats(&ds);
    let (s, sl, sh) = rate_stats(&slack);
    println!(
        "  H {h:.3} [{hl:.3}–{hh:.3}] ms, D {d:.3} [{dl:.3}–{dh:.3}] ms, slack D - H {s:.3} \
         [{sl:.3}–{sh:.3}] ms, X_ondev = E_ondev - D {:.3} ms",
        e_ondev - d
    );
    Ok(())
}

/// `LLVQ_BOUNDARY_AB`: one model, every arm in every round, same order, two
/// rounds discarded then [`ROUNDS_TIMED`], the `run_trim_ab` protocol. `set`
/// puts the model and the runtime in an arm's state. Any token that differs
/// between two arms or two rounds stops the run before a number is printed:
/// every arm adds boundaries and no arithmetic. So does an arm whose counted
/// boundaries fall outside what its table row allows, or move between
/// rounds: an arm that added nothing would otherwise time as a free boundary.
///
/// `t = elapsed / n_new`, and the cost of one added boundary is
/// `(t_arm - t_served) / boundaries a token`, formed round by round
/// (`llvq_llm::boundary::per_boundary_us`).
///
/// Out of the cfg-gated body so the Mac type-checks it.
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
    use llvq_llm::boundary::median_range;
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

/// The prompt lengths `LLVQ_PERS_4E` runs after the seed prompt when its
/// value names none: 8,192, the long context of every reading before
/// 2026-11-11.
#[cfg_attr(not(all(target_os = "linux", feature = "cuda")), allow(dead_code))]
#[cfg(any(target_os = "macos", feature = "cuda"))] // names llvq_cuda, a dependency on macOS and under `cuda` only
const PERS_4E_LONG: [usize; 1] = [8192];

/// `LLVQ_PERS_4E`'s value: a token count, 2 to 256, alone (the bench forms,
/// `pers::P4E_BENCH_FORMS`) or followed by `:` and E1 arms
/// (`pers::P4E_E1_ARMS`), comma-separated, each once, which replace the
/// bench forms; then optionally `:` and the prompt lengths run after the seed
/// prompt, comma-separated, strictly ascending, each 64 to 8,192 (default
/// [`PERS_4E_LONG`]). One repetition is taken: the first arm named again as
/// the LAST of three or more (`v9,v9ep,v9fl,v9`), the fidelity control of
/// rule F1 (docs/mesures/recherche-debit-r3-espace-2026-11-12.txt): the first
/// arm's runtime timed again at the end of every round, no second module.
/// The fourth value is that control.
/// [`parse_pers_4e`]'s value: the token count, the E1 arms, the prompt
/// lengths, and whether the first arm is timed again last (rule F1).
#[cfg(any(target_os = "macos", feature = "cuda"))]
type Pers4eSpec = (usize, Vec<&'static str>, Vec<usize>, bool);

#[cfg(any(target_os = "macos", feature = "cuda"))] // names llvq_cuda, a dependency on macOS and under `cuda` only
#[cfg_attr(not(all(target_os = "linux", feature = "cuda")), allow(dead_code))]
fn parse_pers_4e(v: &str) -> Result<Pers4eSpec, String> {
    let mut fields = v.splitn(3, ':');
    let count = fields.next().unwrap_or_default();
    let arms = fields.next();
    let lengths = match fields.next() {
        None => PERS_4E_LONG.to_vec(),
        Some(l) => {
            let lengths = parse_lengths("LLVQ_PERS_4E", l)?;
            if lengths.iter().any(|n| !(64..=8192).contains(n)) || lengths.windows(2).any(|w| w[0] >= w[1]) {
                return Err(format!("LLVQ_PERS_4E={v:?}: lengths strictly ascending, each 64 to 8192"));
            }
            lengths
        }
    };
    let n: usize = count.parse().map_err(|_| format!("LLVQ_PERS_4E={v:?}: a token count, 2 to 256, then optionally :<E1 arms>"))?;
    if !(2..=256).contains(&n) {
        return Err(format!("LLVQ_PERS_4E={v:?}: 2 to 256 tokens a run"));
    }
    let mut out: Vec<&'static str> = Vec::new();
    let mut again = false;
    if let Some(a) = arms {
        let names: Vec<&str> = a.split(',').collect();
        let names = match names.as_slice() {
            [first, .., last] if names.len() >= 3 && first == last => {
                again = true;
                &names[..names.len() - 1]
            }
            _ => &names[..],
        };
        for &name in names {
            let arm = llvq_cuda::pers::P4E_E1_ARMS
                .iter()
                .find(|x| **x == name)
                .ok_or_else(|| format!("LLVQ_PERS_4E={v:?}: no E1 arm {name:?}; the arms are {:?}", llvq_cuda::pers::P4E_E1_ARMS))?;
            if out.contains(arm) {
                return Err(format!("LLVQ_PERS_4E={v:?}: arm {name} named twice"));
            }
            out.push(arm);
        }
    }
    Ok((n, out, lengths, again))
}

/// `LLVQ_PERS_4E=check:<arms>`'s arms: E1 arms (`pers::P4E_E1_ARMS`),
/// comma-separated, each once, at least one.
#[cfg(any(target_os = "macos", feature = "cuda"))] // names llvq_cuda, a dependency on macOS and under `cuda` only
#[cfg_attr(not(all(target_os = "linux", feature = "cuda")), allow(dead_code))]
fn parse_pers_4e_check(v: &str) -> Result<Vec<&'static str>, String> {
    let mut out: Vec<&'static str> = Vec::new();
    for name in v.split(',') {
        // the load probes (pers::P4E_RC_PROBES) are named here and nowhere else: never timed
        let arm = llvq_cuda::pers::P4E_E1_ARMS
            .iter()
            .chain(llvq_cuda::pers::P4E_RC_PROBES.iter().map(|p| &p.0))
            .find(|x| **x == name)
            .ok_or_else(|| {
                format!(
                    "LLVQ_PERS_4E=check:{v}: no E1 arm or load probe {name:?}; the arms are {:?}, the probes {:?}",
                    llvq_cuda::pers::P4E_E1_ARMS,
                    llvq_cuda::pers::P4E_RC_PROBES.map(|p| p.0)
                )
            })?;
        if out.contains(arm) {
            return Err(format!("LLVQ_PERS_4E=check:{v}: arm {name} named twice"));
        }
        out.push(arm);
    }
    Ok(out)
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
    again: bool,
) -> anyhow::Result<()> {
    use llvq_llm::boundary::median_range;
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
                    "  4E {label}: owes a bound on rows, not the bits: {eq} of {} tokens equal the eager path's, \
                     the first divergence at token {i}",
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
                if owes_bits(rt) { "the bits" } else { "a bound on rows" },
                if pass { "" } else { " (F1 failed: a description)" }
            );
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

#[cfg(all(target_os = "linux", feature = "cuda"))]
/// `LLVQ_PERS_SEGMENT`: one model, the eager MLP and route P's segment in
/// every round, same order, two rounds discarded, then [`ROUNDS_TIMED`].
/// Identical tokens or the run is void; tok/s per arm, and the ratio formed
/// round by round.
fn run_pers_ab(
    model: &mut llvq_llm::model::Qwen3,
    ids: &[u32],
    n_new: usize,
    inst: &llvq_llm::persist_cuda::Installed,
) -> anyhow::Result<()> {
    use llvq_llm::boundary::median_range;
    const WARMUP: usize = 2;
    let names = ["eager", "segment"];
    let mut reference: Option<Vec<u32>> = None;
    let mut rate: [Vec<f64>; 2] = [Vec::new(), Vec::new()];
    for round in 0..WARMUP + ROUNDS_TIMED {
        for (k, name) in names.iter().enumerate() {
            if k == 0 {
                llvq_llm::persist_cuda::clear(model);
            } else {
                inst.set(model);
            }
            let t = Instant::now();
            let out = model.generate(ids, n_new, &mut NoCapture)?;
            let s = t.elapsed().as_secs_f64();
            match &reference {
                None => reference = Some(out),
                Some(r) => {
                    if let Some(i) = first_divergence(r, &out).map_err(|e| anyhow::anyhow!("{e}"))? {
                        anyhow::bail!("round {round}, arm {name}: divergence at token {i}; the segment owes the \
                                       eager bits, and the run is void");
                    }
                }
            }
            if round >= WARMUP {
                rate[k].push(n_new as f64 / s);
            }
        }
    }
    llvq_llm::persist_cuda::clear(model);
    println!("  {n_new} tokens identical across 2 arms and {} rounds", WARMUP + ROUNDS_TIMED);
    for (k, name) in names.iter().enumerate() {
        let (m, lo, hi) = median_range(&rate[k]);
        println!("  {name:<8} {m:7.1} tok/s [{lo:.1}-{hi:.1}]");
    }
    let ratio: Vec<f64> = rate[1].iter().zip(&rate[0]).map(|(a, b)| a / b).collect();
    let (m, lo, hi) = median_range(&ratio);
    println!("  segment / eager, round by round: x{m:.3} [{lo:.3}-{hi:.3}]");
    Ok(())
}

#[cfg_attr(not(all(target_os = "linux", feature = "cuda")), allow(dead_code))]
fn run_boundary_ab(
    model: &mut llvq_llm::model::Qwen3,
    ids: &[u32],
    n_new: usize,
    mut set: impl FnMut(&mut llvq_llm::model::Qwen3, llvq_llm::boundary::BoundaryArm) -> anyhow::Result<()>,
) -> anyhow::Result<()> {
    use llvq_llm::boundary::{
        counted_bounds, derived_series, median_range, per_boundary_us, take_added, BoundaryArm,
        DERIVED,
    };
    const WARMUP: usize = 2;
    let arms = BoundaryArm::ALL;
    let mut reference: Option<Vec<u32>> = None;
    let mut secs: Vec<Vec<f64>> = vec![Vec::new(); arms.len()];
    let mut counted: Vec<Option<u64>> = vec![None; arms.len()];
    for round in 0..WARMUP + ROUNDS_TIMED {
        for (k, arm) in arms.into_iter().enumerate() {
            set(model, arm)?;
            take_added();
            let t = Instant::now();
            let out = model.generate(ids, n_new, &mut NoCapture)?;
            let s = t.elapsed().as_secs_f64();
            let added = take_added();
            match &reference {
                None => reference = Some(out),
                Some(r) => {
                    if let Some(i) = first_divergence(r, &out).map_err(|e| anyhow::anyhow!("{e}"))? {
                        anyhow::bail!(
                            "round {round}, arm {}: divergence at token {i}. Every arm adds \
                             boundaries and no arithmetic, and arms that disagree void the run",
                            arm.name()
                        );
                    }
                }
            }
            let (lo, hi) = counted_bounds(arm, n_new, ids.len());
            if added < lo || added > hi {
                anyhow::bail!(
                    "round {round}, arm {}: {added} boundaries added in a generate, outside \
                     [{lo}, {hi}] for {} a decode step. The arm is not wired as its row says, \
                     and the run is void",
                    arm.name(),
                    arm.boundaries_per_token()
                );
            }
            match counted[k] {
                None => counted[k] = Some(added),
                Some(c) if c != added => anyhow::bail!(
                    "round {round}, arm {}: {added} boundaries added, {c} in round 0. The run is void",
                    arm.name()
                ),
                Some(_) => {}
            }
            if round >= WARMUP {
                secs[k].push(s / n_new as f64);
            }
        }
    }
    set(model, BoundaryArm::Served)?;
    let n_tokens = reference.map_or(0, |r| r.len());
    println!(
        "\n{} arms × {ROUNDS_TIMED} interleaved rounds after {WARMUP} discarded, {n_tokens} \
         identical tokens in every arm and round",
        arms.len()
    );
    println!(
        "  t = elapsed / {n_new}; us per added boundary = (t_arm - t_served) / boundaries a \
         token, round by round"
    );
    println!("  arm             a token  a generate   tok/s [range]          us per added boundary [range]");
    let mut per: Vec<Vec<f64>> = vec![Vec::new(); arms.len()];
    for (k, arm) in arms.into_iter().enumerate() {
        let rates: Vec<f64> = secs[k].iter().map(|t| 1.0 / t).collect();
        let (m, lo, hi) = median_range(&rates);
        let nb = arm.boundaries_per_token();
        let c = counted[k].unwrap_or(0);
        if nb == 0 {
            println!("  {:<14}  {nb:>7}  {c:>10}   {m:6.1} [{lo:.1}–{hi:.1}]", arm.name());
            continue;
        }
        per[k] = per_boundary_us(&secs[0], &secs[k], nb).map_err(|e| anyhow::anyhow!("{e}"))?;
        let (pm, pl, ph) = median_range(&per[k]);
        println!(
            "  {:<14}  {nb:>7}  {c:>10}   {m:6.1} [{lo:.1}–{hi:.1}]   {pm:7.3} [{pl:.3}–{ph:.3}]",
            arm.name()
        );
    }
    println!("\nderived classes, us per boundary, round by round (printed, never gated)");
    for (name, a, b) in DERIVED {
        let d = derived_series(&per, a, b).map_err(|e| anyhow::anyhow!("{e}"))?;
        let (m, lo, hi) = median_range(&d);
        let note = if name == "linearity" { "   (0 when linear)" } else { "" };
        println!(
            "  {name:<10} = {:<14} - {:<10}  {m:7.3} [{lo:.3}–{hi:.3}]{note}",
            a.name(),
            b.name()
        );
    }
    Ok(())
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

/// The measurement modes and prompt variables of this binary that run alone.
///
/// One table replaces the eight lists each mode used to spell out, which had
/// drifted apart: `LLVQ_PERS_SEGMENT` listed itself, so `=1` always failed,
/// and the modes dispatched last returned before reading the ones dispatched
/// after them, so `LLVQ_GRAPH_AB` beside `LLVQ_GQA_PPL` was silently ignored.
/// A mode calls [`runs_alone`]: every OTHER name here, set beside it, is
/// refused. `LLVQ_GRAPH_DIAG` rides inside `LLVQ_GRAPH_AB`, which does not
/// call it.
#[cfg_attr(not(all(target_os = "linux", feature = "cuda")), allow(dead_code))]
const ALONE: [&str; 16] = [
    "LLVQ_INT4_WIDE_CHECK", "LLVQ_GESTURE_AB", "LLVQ_BOUNDARY_AB", "LLVQ_PERS_4E", "LLVQ_PERS_SEGMENT",
    "LLVQ_GQA_PPL", "LLVQ_GQA_AB", "LLVQ_CONTEXTS", "LLVQ_TRIM_AB", "LLVQ_PREFILL_TOKENS",
    "LLVQ_GRAPH_AB", "LLVQ_GRAPH_DIAG", "LLVQ_KV_AB", "LLVQ_FUSE_AB", "LLVQ_TIME_PHASES", "LLVQ_SEG_ARMS",
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
        // Resolved once, here — `model.rs` reads no environment variable. The
        // store rides BOTH arms identically: it is never the variable of this
        // A/B (the `check_fuse` rule), and each arm line prints it below so a
        // wiring miss shows as `cat` instead of silently measuring it.
        let kv_store = llvq_llm::kvq::KvStore::from_env().map_err(|e| anyhow::anyhow!("{e}"))?;
        // Resolved here, beside `kv_store`, and for the same reason: one read
        // of the outside world, at the top, so no path below can pick up a
        // different answer. `None` is the bench; `Some` is the runner.
        let served = llvq_llm::served::Served::from_env().map_err(|e| anyhow::anyhow!("{e}"))?;
        // The served path takes no measurement mode. Each of these turns this
        // binary into a bench of one thing, and a bench needs its arms named
        // by variables — that is what `LLVQ_CONFIG` exists to replace. Refused
        // by name rather than silently dropped, which is what happened before
        // the audit of 2026-09-11 found it. `LLVQ_PREFILL_TOKENS` and
        // `LLVQ_CONTEXTS` are the two exceptions, handled below: they time the
        // served object as served, and change only the prompt it is given.
        if served.is_some() {
            for var in ["LLVQ_GRAPH_AB", "LLVQ_GRAPH_DIAG", "LLVQ_KV_AB", "LLVQ_FUSE_AB",
                        "LLVQ_TIME_PHASES", "LLVQ_KV_PREALLOC", "LLVQ_SEG_ARMS",
                        "LLVQ_TRIM_AB", "LLVQ_GQA_AB", "LLVQ_GQA_PPL",
                        "LLVQ_INT4_WIDE_CHECK", "LLVQ_GESTURE_AB", "LLVQ_BOUNDARY_AB", "LLVQ_PERS_SEGMENT", "LLVQ_PERS_4E"] {
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
        // ---- LLVQ_GESTURE_AB: the token boundary of the decode loop ----
        //
        // A MEASUREMENT MODE on the bench door, refused beside LLVQ_CONFIG, the
        // served flags spelled out on the command line: Part A of the gestures
        // prereg. ONE model in the served 4B state (Grow(256) + grouped, the
        // three trims in force, events off before the load), the key kernel's load check
        // and card mutants, two probes, five arms, two clocks
        // (`run_gesture_mode`). Every arm must return the base arm's tokens.
        // `LLVQ_KERNEL_DIR` is refused: the mutants are applied in memory to
        // the embedded kernel text, and a directory would swap the text the
        // load check certifies.
        if let Ok(v) = std::env::var("LLVQ_GESTURE_AB") {
            if v != "1" {
                anyhow::bail!("LLVQ_GESTURE_AB={v:?}: the only value is 1");
            }
            runs_alone("LLVQ_GESTURE_AB", &["LLVQ_KERNEL_DIR"])?;
            if !matches!(kv_store, llvq_llm::kvq::KvStore::Cat) {
                anyhow::bail!("LLVQ_GESTURE_AB sets the served store itself: unset LLVQ_KV_PREALLOC");
            }
            if n_new < CLOCK_TOKENS {
                anyhow::bail!("LLVQ_GESTURE_AB: {n_new} tokens, and its clocks decode {CLOCK_TOKENS}");
            }
            // Events off BEFORE the load, as the served door does in
            // `load_resolved`: cudarc records a read event at every use of a
            // slice created while tracking was on, whatever the flag says at the
            // use, so turning it off after the load would leave every weight
            // recording an event at every launch, a state the served door never
            // runs in.
            let ctx = device.as_cuda_device()?.cuda_stream().context().clone();
            if ctx.is_in_multi_stream_mode() {
                anyhow::bail!("LLVQ_GESTURE_AB: the context holds several streams, and turning \
                               cudarc's events off is only safe on one");
            }
            // Safe: one stream, checked above, orders every access by issue.
            unsafe {
                ctx.disable_event_tracking();
            }
            println!("cudarc event tracking: off (one stream, ordered by issue), before the load");
            let fuse = llvq_llm::fused::FuseMode::from_env().map_err(|e| anyhow::anyhow!("{e}"))?;
            let t = Instant::now();
            let mut f = llvq_llm::fused_cuda::load_with(&path, &device, dtype, fuse)?;
            println!(
                "loaded in {:.1} s. gesture A/B: {} arms, one model, interleaved rounds",
                t.elapsed().as_secs_f64(),
                GestureArm::ALL.len()
            );
            println!(
                "LLVQ_ROT_SHARE={}, {} rot_launches/token · LLVQ_FUSE={}, {} matvec_launches/token",
                f.rot_share.name(), f.rot_launches, f.fuse.name(), f.matvec_launches
            );
            GqaArm::Grouped.apply(&mut f.model);
            println!(
                "attention {} on {}, layout {}",
                f.model.attn_heads().label(),
                f.model.kv_store().label(),
                f.layout.name()
            );
            let ids = f
                .tokenizer
                .encode(PROMPT, false)
                .map_err(|e| anyhow::anyhow!("{e}"))?
                .get_ids()
                .to_vec();
            run_gesture_mode(&f.model, &device, &ids, n_new)?;
            if ctx.is_in_multi_stream_mode() {
                anyhow::bail!("LLVQ_GESTURE_AB: a second stream appeared during the run, with \
                               cudarc's events off: the numbers above are void");
            }
            return Ok(());
        }

        // ---- LLVQ_BOUNDARY_AB: what one added kernel boundary costs ----
        //
        // A MEASUREMENT MODE on the bench door, refused beside LLVQ_CONFIG, the
        // served flags spelled out on the command line: process 2b of
        // prereg-persistent-0. ONE model in the served 4B state (Grow(256) +
        // grouped, events off before the load), eight arms that each add kernel
        // boundaries and no arithmetic (`llvq_llm::boundary`), every arm in
        // every round (`run_boundary_ab`). The empty kernels come from their own
        // module on candle's stream, so the served unit and its `NVRTC source:`
        // line do not move. `LLVQ_KERNEL_DIR` is refused for that line's sake.
        if let Ok(v) = std::env::var("LLVQ_BOUNDARY_AB") {
            use llvq_llm::boundary::BoundaryArm;
            if v != "1" {
                anyhow::bail!("LLVQ_BOUNDARY_AB={v:?}: the only value is 1");
            }
            runs_alone("LLVQ_BOUNDARY_AB", &["LLVQ_KERNEL_DIR"])?;
            if !matches!(kv_store, llvq_llm::kvq::KvStore::Cat) {
                anyhow::bail!("LLVQ_BOUNDARY_AB sets the served store itself: unset LLVQ_KV_PREALLOC");
            }
            if n_new < 2 {
                anyhow::bail!("LLVQ_BOUNDARY_AB: {n_new} tokens, and its cost is read on decode steps");
            }
            // Both refused before the transcode. The splits are Tetra launches,
            // and a fused gate+up would never reach them.
            let layout = llvq_llm::fused::FusedLayout::from_env().map_err(|e| anyhow::anyhow!("{e}"))?;
            if layout != llvq_llm::fused::FusedLayout::Tetra48 {
                anyhow::bail!(
                    "LLVQ_BOUNDARY_AB: layout {}, and its split arms are tv_tetra48_h launches. \
                     Set LLVQ_FUSED_LAYOUT=tetra48",
                    layout.name()
                );
            }
            let fuse = llvq_llm::fused::FuseMode::from_env().map_err(|e| anyhow::anyhow!("{e}"))?;
            if fuse != llvq_llm::fused::FuseMode::Off {
                anyhow::bail!("LLVQ_BOUNDARY_AB: LLVQ_FUSE={}, and the served 4B runs 0", fuse.name());
            }
            // Events off BEFORE the load, as `LLVQ_GESTURE_AB` and the served
            // door do: cudarc records a read event at every use of a slice
            // created while tracking was on, whatever the flag says at the use.
            let ctx = device.as_cuda_device()?.cuda_stream().context().clone();
            if ctx.is_in_multi_stream_mode() {
                anyhow::bail!("LLVQ_BOUNDARY_AB: the context holds several streams, and turning \
                               cudarc's events off is only safe on one");
            }
            // Safe: one stream, checked above, orders every access by issue.
            unsafe {
                ctx.disable_event_tracking();
            }
            println!("cudarc event tracking: off (one stream, ordered by issue)");
            let t = Instant::now();
            let mut f = llvq_llm::fused_cuda::load_with(&path, &device, dtype, fuse)?;
            let sha = llvq_llm::fused_cuda::install_boundary_probe(&device)?;
            if ctx.is_in_multi_stream_mode() {
                anyhow::bail!("LLVQ_BOUNDARY_AB: a second stream appeared at the load, with \
                               cudarc's events off");
            }
            println!("llvq_empty: own NVRTC module on candle's stream, source sha256 {sha}");
            println!(
                "loaded in {:.1} s. boundary A/B: {} arms, one model, interleaved rounds",
                t.elapsed().as_secs_f64(),
                BoundaryArm::ALL.len()
            );
            println!(
                "LLVQ_ROT_SHARE={}, {} rot_launches/token · LLVQ_FUSE={}, {} matvec_launches/token",
                f.rot_share.name(), f.rot_launches, f.fuse.name(), f.matvec_launches
            );
            let c = f.model.config();
            llvq_llm::boundary::check_4b_shape(c.num_hidden_layers, c.hidden_size, c.intermediate_size)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            GqaArm::Grouped.apply(&mut f.model);
            println!("attention {} on {}", f.model.attn_heads().label(), f.model.kv_store().label());
            println!(
                "layout {}, {} layers, gate_proj {} rows, down_proj {} rows",
                f.layout.name(),
                llvq_llm::boundary::LAYERS_4B,
                llvq_llm::boundary::GATE_D_OUT_4B,
                llvq_llm::boundary::DOWN_D_OUT_4B
            );
            let ids = f
                .tokenizer
                .encode(PROMPT, false)
                .map_err(|e| anyhow::anyhow!("{e}"))?
                .get_ids()
                .to_vec();
            let ran = run_boundary_ab(&mut f.model, &ids, n_new, |m, arm| {
                llvq_llm::fused_cuda::set_boundary_arm(arm);
                m.set_candle_probe(arm.candle_probe())?;
                Ok(())
            });
            // Back to the served state whatever the run returned.
            llvq_llm::fused_cuda::set_boundary_arm(BoundaryArm::Served);
            f.model.set_candle_probe(false)?;
            ran?;
            if ctx.is_in_multi_stream_mode() {
                anyhow::bail!("LLVQ_BOUNDARY_AB: a second stream appeared during the run, with \
                               cudarc's events off: the numbers above are void");
            }
            return Ok(());
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
        // `<count>:<arms>` (experiment E1 of
        // docs/mesures/large-4e-budget-2026-11-11.txt): the E1 arms named,
        // comma-separated from pers::P4E_E1_ARMS, instead of the bench forms,
        // each with its logits symbol; the logits of every arm are compared
        // with the first's, bit for bit, at every context. A third field,
        // `<count>:<arms>:<lengths>`, names the prompt lengths after the seed
        // prompt (8,192 when absent). The first arm named again as the last
        // (`v9,v9ep,v9fl,v9`) is rule F1's control: the reference timed first
        // and last in every round, eager after it, F1 and a SCORE line an arm
        // (F2). An arm that owes a bound on rows (v9fl) is timed on its own
        // tokens, which are compared with the reference's and reported.
        if let Ok(v) = std::env::var("LLVQ_PERS_4E") {
            // `check:<arms>`: the load check alone (persist4e_cuda::check_e1_arms), before
            // any model is loaded: each arm's module compiled and its symbols read, nothing
            // launched, then the process ends. The R6 card run printed its refusal 339 s in,
            // after the model and three modules (docs/mesures/recherche-debit-jeton-2026-11-13.txt).
            if let Some(list) = v.strip_prefix("check:") {
                let arms = parse_pers_4e_check(list).map_err(|e| anyhow::anyhow!("{e}"))?;
                runs_alone("LLVQ_PERS_4E", &["LLVQ_KERNEL_DIR"])?;
                let t = Instant::now();
                // the check prints its own lines as it reads them
                llvq_llm::persist4e_cuda::check_e1_arms(&device, &arms)?;
                println!("4E check done in {:.1} s", t.elapsed().as_secs_f64());
                return Ok(());
            }
            let (count, e1_arms, long, again) = parse_pers_4e(&v).map_err(|e| anyhow::anyhow!("{e}"))?;
            let n_new: usize = count;
            if !(2..=256).contains(&n_new) {
                anyhow::bail!("LLVQ_PERS_4E={n_new}: 2 to 256 tokens a run");
            }
            runs_alone("LLVQ_PERS_4E", &["LLVQ_KERNEL_DIR"])?;
            if !matches!(kv_store, llvq_llm::kvq::KvStore::Cat) {
                anyhow::bail!("LLVQ_PERS_4E sets the store itself: unset LLVQ_KV_PREALLOC");
            }
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
            for arm in &e1_arms {
                match llvq_llm::persist4e_cuda::install_e1(&f.model, &device, n_new as u32, arm) {
                    Ok(rt) => {
                        for line in &rt.lines {
                            println!("{line}");
                        }
                        rts.push(rt);
                    }
                    Err(e) => println!("4E e1:{arm} at 512 threads: not run: {e}"),
                }
            }
            for (form, label) in llvq_cuda::pers::P4E_BENCH_FORMS.into_iter().filter(|_| e1_arms.is_empty()) {
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
            if e1_arms.is_empty() {
                match llvq_llm::persist4e_cuda::install_rows_as(&f.model, &device, n_new as u32, Some(512), 4, Some(6)) {
                    Ok(rt) => {
                        for line in &rt.lines {
                            println!("{line}");
                        }
                        rts.push(rt);
                    }
                    Err(e) => println!("4E v6s at 512 threads: not run: {e}"),
                }
            }
            if rts.is_empty() {
                anyhow::bail!("LLVQ_PERS_4E: no 4E geometry loads on this card");
            }
            if again && rts[0].e1_arm != e1_arms.first().copied() {
                anyhow::bail!("LLVQ_PERS_4E: the F1 control's arm {:?} did not load; every ratio would read another reference", e1_arms[0]);
            }
            if ctx.is_in_multi_stream_mode() {
                anyhow::bail!("LLVQ_PERS_4E: a second stream appeared at the load, with events off");
            }
            println!("loaded and checked in {:.1} s", t.elapsed().as_secs_f64());
            let base = f.tokenizer.encode(PROMPT, false).map_err(|e| anyhow::anyhow!("{e}"))?.get_ids().to_vec();
            for len in std::iter::once(base.len()).chain(long) {
                let prompt: Vec<u32> = base.iter().copied().cycle().take(len).collect();
                run_pers_4e(&mut f.model, &mut rts, &prompt, n_new, again)?;
            }
            if ctx.is_in_multi_stream_mode() {
                anyhow::bail!("LLVQ_PERS_4E: a second stream appeared during the run: the numbers above are void");
            }
            return Ok(());
        }

        // ---- LLVQ_PERS_SEGMENT: route P, step A, against the eager MLP ----
        //
        // A MEASUREMENT MODE on the bench door, refused beside LLVQ_CONFIG, the
        // served flags spelled out on the command line (gate L3 of
        // persistent.md §5; its prereg is written when L2 has spoken). ONE model
        // in the served 4B state (Grow(256) + grouped, events off before the
        // load), two arms in every round: the eager MLP half, and the
        // persistent segment (`llvq_llm::persist_cuda`) on every layer whose
        // install check passed. Identical tokens are required: the segment owes
        // the eager ops' bits, and any divergence voids the run.
        if let Ok(v) = std::env::var("LLVQ_PERS_SEGMENT") {
            if v != "1" {
                anyhow::bail!("LLVQ_PERS_SEGMENT={v:?}: the only value is 1");
            }
            runs_alone("LLVQ_PERS_SEGMENT", &["LLVQ_KERNEL_DIR"])?;
            if !matches!(kv_store, llvq_llm::kvq::KvStore::Cat) {
                anyhow::bail!("LLVQ_PERS_SEGMENT sets the served store itself: unset LLVQ_KV_PREALLOC");
            }
            if n_new < 2 {
                anyhow::bail!("LLVQ_PERS_SEGMENT: {n_new} tokens, and the segment runs on decode steps");
            }
            let layout = llvq_llm::fused::FusedLayout::from_env().map_err(|e| anyhow::anyhow!("{e}"))?;
            if layout != llvq_llm::fused::FusedLayout::Tetra48 {
                anyhow::bail!("LLVQ_PERS_SEGMENT: layout {}, and the segment reads Tetra48 streams", layout.name());
            }
            let fuse = llvq_llm::fused::FuseMode::from_env().map_err(|e| anyhow::anyhow!("{e}"))?;
            if fuse != llvq_llm::fused::FuseMode::Off {
                anyhow::bail!("LLVQ_PERS_SEGMENT: LLVQ_FUSE={}, and the served 4B runs 0", fuse.name());
            }
            let ctx = device.as_cuda_device()?.cuda_stream().context().clone();
            if ctx.is_in_multi_stream_mode() {
                anyhow::bail!("LLVQ_PERS_SEGMENT: the context holds several streams, and turning \
                               cudarc's events off is only safe on one");
            }
            // Safe: one stream, checked above, orders every access by issue.
            unsafe {
                ctx.disable_event_tracking();
            }
            println!("cudarc event tracking: off (one stream, ordered by issue)");
            let t = Instant::now();
            let mut f = llvq_llm::fused_cuda::load_with(&path, &device, dtype, fuse)?;
            GqaArm::Grouped.apply(&mut f.model);
            println!("attention {} on {}", f.model.attn_heads().label(), f.model.kv_store().label());
            let inst = llvq_llm::persist_cuda::install(&f.model, &device)?;
            for line in &inst.lines {
                println!("{line}");
            }
            if ctx.is_in_multi_stream_mode() {
                anyhow::bail!("LLVQ_PERS_SEGMENT: a second stream appeared at the load, with events off");
            }
            println!("loaded and checked in {:.1} s. route P A/B: eager against segment, interleaved rounds",
                     t.elapsed().as_secs_f64());
            let ids = f.tokenizer.encode(PROMPT, false).map_err(|e| anyhow::anyhow!("{e}"))?.get_ids().to_vec();
            let ran = run_pers_ab(&mut f.model, &ids, n_new, &inst);
            llvq_llm::persist_cuda::clear(&mut f.model);
            ran?;
            if inst.aborted()? {
                anyhow::bail!("LLVQ_PERS_SEGMENT: a segment launch aborted on its watchdog: the numbers above are void");
            }
            if ctx.is_in_multi_stream_mode() {
                anyhow::bail!("LLVQ_PERS_SEGMENT: a second stream appeared during the run: the numbers above are void");
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
            if !matches!(kv_store, llvq_llm::kvq::KvStore::Cat) {
                anyhow::bail!("LLVQ_GQA_PPL sets the store of each arm: unset LLVQ_KV_PREALLOC");
            }
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
            if !matches!(kv_store, llvq_llm::kvq::KvStore::Cat) {
                anyhow::bail!("LLVQ_GQA_AB sets the store of each arm: unset LLVQ_KV_PREALLOC");
            }
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
            // The softmax's memory form, the one variable this mode reads
            // beside a config: it moves no output bit (the logits fingerprint
            // each length prints is how a second process checks that).
            match std::env::var("LLVQ_SOFTMAX_WHOLE").as_deref() {
                Err(_) | Ok("0") => {}
                Ok("1") => llvq_llm::model::set_softmax_whole_for_measurement(true),
                Ok(o) => anyhow::bail!("LLVQ_SOFTMAX_WHOLE={o:?}: 1 (the whole-tensor softmax) or 0 (in place)"),
            }
            if !matches!(kv_store, llvq_llm::kvq::KvStore::Cat) {
                anyhow::bail!(
                    "LLVQ_CONTEXTS with a preallocated window: the window would bound the \
                     prompt this mode exists to lengthen. Unset LLVQ_KV_PREALLOC. A served \
                     config's own store is applied below and is not this check's business."
                );
            }
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

        // ---- LLVQ_TRIM_AB: what the three trims of 2026-10-30 save ----
        //
        // A MEASUREMENT MODE on the bench door, refused beside LLVQ_CONFIG: the
        // served flags are spelled out on the command line. ONE model, every arm
        // in every round (`run_trim_ab`). The events arm flips cudarc's tracking
        // between rounds, which is safe only on one stream, so that is checked.
        if std::env::var("LLVQ_TRIM_AB").ok().as_deref() == Some("1") {
            if !matches!(kv_store, llvq_llm::kvq::KvStore::Cat) {
                anyhow::bail!("LLVQ_TRIM_AB times the served decode step, on the Cat store: \
                               unset LLVQ_KV_PREALLOC");
            }
            let fuse = llvq_llm::fused::FuseMode::from_env().map_err(|e| anyhow::anyhow!("{e}"))?;
            let t = Instant::now();
            let mut f = llvq_llm::fused_cuda::load_with(&path, &device, dtype, fuse)?;
            println!(
                "loaded in {:.1} s. trim A/B: {} arms, one model, interleaved rounds",
                t.elapsed().as_secs_f64(),
                TrimArm::ALL.len()
            );
            println!(
                "LLVQ_ROT_SHARE={}, {} rot_launches/token · LLVQ_FUSE={}, {} matvec_launches/token",
                f.rot_share.name(), f.rot_launches, f.fuse.name(), f.matvec_launches
            );
            let ctx = device.as_cuda_device()?.cuda_stream().context().clone();
            if ctx.is_in_multi_stream_mode() {
                anyhow::bail!("LLVQ_TRIM_AB: the context holds several streams, and the \
                               events arm is only safe on one");
            }
            let ids = f
                .tokenizer
                .encode(PROMPT, false)
                .map_err(|e| anyhow::anyhow!("{e}"))?
                .get_ids()
                .to_vec();
            run_trim_ab(&mut f.model, &ids, n_new, |m, arm| {
                m.set_decode_mask(arm.mask());
                llvq_llm::fused_cuda::set_int4_via_f32(arm.cast());
                // Safe: one stream, checked above, orders every access by issue,
                // so slices created under either setting are freed in order.
                unsafe {
                    if arm.events() {
                        ctx.enable_event_tracking();
                    } else {
                        ctx.disable_event_tracking();
                    }
                }
                Ok(())
            })?;
            return Ok(());
        }

        // ---- LLVQ_PREFILL_TOKENS: what a prompt costs through the kernel ----
        //
        // A MEASUREMENT MODE, never a served config — the `LLVQ_TIME_PHASES`
        // and `LLVQ_GRAPH_AB` rule. Unset, this binary is byte-identical to
        // the published protocol.
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
            // store comes from the variable, as it always did.
            match &served {
                Some(cfg) => cfg.apply(&mut f.model, &device).map_err(|e| anyhow::anyhow!("{e}"))?,
                None => f.model.set_kv_store(kv_store),
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

        if std::env::var("LLVQ_GRAPH_AB").ok().as_deref() == Some("1") {
            use candle_core::IndexOp;
            use llvq_llm::kvq::KvStore;
            let w = match kv_store {
                KvStore::Prealloc(w) => w,
                KvStore::Cat | KvStore::Grow(_) => anyhow::bail!(
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
                #[allow(clippy::explicit_counter_loop)]
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
                KvStore::Cat | KvStore::Grow(_) => anyhow::bail!(
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
    use super::{alone_in, first_divergence, paired_ppl_pct, parse_gqa_ppl, rate_stats, GqaArm, GqaPplSpec, ALONE};
    use llvq_llm::eval::t95;

    /// `LLVQ_PERS_4E`: a count alone keeps the bench forms; a count and E1
    /// arms names those arms in order, then optionally the long prompt
    /// lengths (8,192 alone when absent); anything else is refused.
    #[test]
    #[cfg(any(target_os = "macos", feature = "cuda"))]
    fn pers_4e_takes_a_count_and_optional_e1_arms() {
        use super::parse_pers_4e;
        assert_eq!(parse_pers_4e("64"), Ok((64, vec![], vec![8192], false)));
        assert_eq!(parse_pers_4e("64:v9,v9np,v9mv"), Ok((64, vec!["v9", "v9np", "v9mv"], vec![8192], false)));
        assert_eq!(parse_pers_4e("64:v9mv"), Ok((64, vec!["v9mv"], vec![8192], false)));
        assert_eq!(parse_pers_4e("64:v9,v9ft"), Ok((64, vec!["v9", "v9ft"], vec![8192], false)));
        assert_eq!(parse_pers_4e("64:v9,v9ft:1024,4096,8192"), Ok((64, vec!["v9", "v9ft"], vec![1024, 4096, 8192], false)));
        // the F1 control: the first arm again as the last of three or more
        assert_eq!(parse_pers_4e("256:v9,v9ep,v9fl,v9"), Ok((256, vec!["v9", "v9ep", "v9fl"], vec![8192], true)));
        assert_eq!(parse_pers_4e("256:v9,v9fl,v9:8192"), Ok((256, vec!["v9", "v9fl"], vec![8192], true)));
        // the load check's arms
        use super::parse_pers_4e_check;
        assert_eq!(parse_pers_4e_check("v9,v9ep2,v9fl3"), Ok(vec!["v9", "v9ep2", "v9fl3"]));
        for bad in ["", "v9,v9", "v9,v8", "v9 ", "v9ep,"] {
            assert!(parse_pers_4e_check(bad).is_err(), "check:{bad}");
        }
        assert!(parse_pers_4e("check:v9").is_err(), "the check is dispatched before parse_pers_4e");
        // the load probes: the check names them, a timed run never does
        assert_eq!(parse_pers_4e_check("v9pe0,v9peq,v9peg,v9ped,v9peo"), Ok(vec!["v9pe0", "v9peq", "v9peg", "v9ped", "v9peo"]));
        assert!(parse_pers_4e_check("v9peq,v9peq").is_err());
        // R6d's arms: E1 arms, so the check and a timed run both name them
        assert_eq!(parse_pers_4e_check("v9,v9fg,v9fn,v9eg"), Ok(vec!["v9", "v9fg", "v9fn", "v9eg"]));
        assert_eq!(parse_pers_4e("256:v9,v9fg,v9eg,v9"), Ok((256, vec!["v9", "v9fg", "v9eg"], vec![8192], true)));
        assert_eq!(parse_pers_4e("256:v9,v9fn,v9"), Ok((256, vec!["v9", "v9fn"], vec![8192], true)));
        assert_eq!(parse_pers_4e("256:v9fg"), Ok((256, vec!["v9fg"], vec![8192], false)));
        for probe in llvq_cuda::pers::P4E_RC_PROBES {
            assert!(parse_pers_4e(&format!("64:v9,{}", probe.0)).is_err(), "{} is never timed", probe.0);
        }
        for bad in [
            "1", "257", "x", "64:", "64:v9,v9", "64:v8", "64:v9 ,v9np", ":v9", "64:v9:", "64::8192", "64:v9:8192,4096",
            "64:v9:4096,4096", "64:v9:8193", "64:v9:5", "64:v9:0", "64:v9:1024:4096", "64:v9,v9ep,v9ep", "64:v9,v9,v9ep",
            "64:v9,v9ep,v9,v9", "64:v9ep,v9,v9",
        ] {
            assert!(parse_pers_4e(bad).is_err(), "{bad}");
        }
    }

    fn env_of<'a>(set: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
        move |v| set.iter().find(|(k, _)| *k == v).map(|(_, x)| x.to_string())
    }

    /// A mode set alone passes its own check. `LLVQ_PERS_SEGMENT` failed this
    /// from 2026-11-04 to 2026-11-07, when its list named itself.
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
        // The hole the eight lists left: GQA_PPL returned before GRAPH_AB was read.
        assert!(alone_in("LLVQ_GQA_PPL", &[], env_of(&[("LLVQ_GRAPH_AB", "1")])).is_err());
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
