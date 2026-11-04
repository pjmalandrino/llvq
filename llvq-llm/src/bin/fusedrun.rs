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
}

#[cfg_attr(not(all(target_os = "linux", feature = "cuda")), allow(dead_code))]
fn parse_gqa_ppl(v: &str) -> Result<GqaPplSpec, String> {
    let parts: Vec<&str> = v.split(',').map(str::trim).collect();
    let (nums, kernel) = match parts.as_slice() {
        [c, w] => (vec![*c, *w], false),
        [c, w, f, "kernel"] => (vec![*c, *w, *f], true),
        _ => {
            return Err(format!(
                "LLVQ_GQA_PPL={v:?}: expected <ctx>,<windows> (grouped against published, from \
                 window 0) or <ctx>,<windows>,<first>,kernel (kernel against grouped)"
            ))
        }
    };
    let n = nums
        .iter()
        .map(|s| s.parse::<usize>().map_err(|e| format!("LLVQ_GQA_PPL={v:?}: {s:?}: {e}")))
        .collect::<Result<Vec<_>, _>>()?;
    let spec = GqaPplSpec { ctx: n[0], windows: n[1], first: n.get(2).copied().unwrap_or(0), kernel };
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
    if spec.kernel && spec.ctx <= from + 1 {
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
            let g = model.generate_timed(&ids, n_new, chunk)?;
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
                "  window {:>3}/{nwin}  positions < {split}: bit for bit  decode nll {:.9} {:.9}  \
                 from {split}: nll {:.9} {:.9}  top-1 {}/{}  max |dlogit| {:.3e}  ({:.0} s)",
                w + 1,
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
                        "LLVQ_TRIM_AB", "LLVQ_GQA_AB", "LLVQ_GQA_PPL"] {
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
            for other in ["LLVQ_GQA_AB", "LLVQ_CONTEXTS", "LLVQ_TRIM_AB", "LLVQ_PREFILL_TOKENS"] {
                if std::env::var(other).is_ok() {
                    anyhow::bail!("LLVQ_GQA_PPL runs alone, and {other} is set beside it");
                }
            }
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
            for other in ["LLVQ_CONTEXTS", "LLVQ_TRIM_AB", "LLVQ_PREFILL_TOKENS"] {
                if std::env::var(other).is_ok() {
                    anyhow::bail!("LLVQ_GQA_AB runs alone, and {other} is set beside it");
                }
            }
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
            for other in ["LLVQ_TRIM_AB", "LLVQ_PREFILL_TOKENS"] {
                if std::env::var(other).is_ok() {
                    anyhow::bail!("LLVQ_CONTEXTS runs alone, and {other} is set beside it");
                }
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
                        cfg.kv, Some("LLVQ_CONFIG"),
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
            run_context_sweep(&f.model, &seed, &lengths, n_new)?;
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
                        cfg.kv, Some("LLVQ_CONFIG"),
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
                Some("LLVQ_CONFIG"),
            )?;
            // The attention form and its store, from the file and from
            // nowhere else. The provenance line above already named them.
            cfg.apply(&mut f.model, &device).map_err(|e| anyhow::anyhow!("{e}"))?;
            let load_s = t.elapsed().as_secs_f64();
            let ids = f
                .tokenizer
                .encode(PROMPT, false)
                .map_err(|e| anyhow::anyhow!("{e}"))?
                .get_ids()
                .to_vec();
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
    use super::{first_divergence, paired_ppl_pct, parse_gqa_ppl, rate_stats, t95, GqaArm, GqaPplSpec};

    /// The two forms of `LLVQ_GQA_PPL`, and what each refuses.
    #[test]
    fn gqa_ppl_spec() {
        assert_eq!(
            parse_gqa_ppl("4096,12").unwrap(),
            GqaPplSpec { ctx: 4096, windows: 12, first: 0, kernel: false }
        );
        assert_eq!(
            parse_gqa_ppl("8192,6,6,kernel").unwrap(),
            GqaPplSpec { ctx: 8192, windows: 6, first: 6, kernel: true }
        );
        assert_eq!(
            parse_gqa_ppl(" 4096 , 12 , 60 , kernel ").unwrap(),
            GqaPplSpec { ctx: 4096, windows: 12, first: 60, kernel: true }
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
