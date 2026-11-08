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

#[path = "fusedrun/context.rs"]
mod context;
#[path = "fusedrun/gqa.rs"]
mod gqa;
#[cfg(all(target_os = "linux", feature = "cuda"))]
#[path = "fusedrun/bench.rs"]
mod bench;
#[cfg(all(target_os = "linux", feature = "cuda"))]
#[path = "fusedrun/door.rs"]
mod door;
#[cfg(all(target_os = "linux", feature = "cuda"))]
#[path = "fusedrun/pers4e.rs"]
mod pers4e;
#[cfg(all(target_os = "linux", feature = "cuda"))]
#[path = "fusedrun/prefill.rs"]
mod prefill;

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

/// The measurement modes and prompt variables of this binary that run alone.
///
/// One table replaces the eight lists each mode used to spell out, which had
/// drifted apart: one mode listed itself, so `=1` always failed, and the
/// modes dispatched last returned before reading the ones dispatched after
/// them, so a mode beside `LLVQ_GQA_PPL` was silently ignored. A mode calls
/// [`runs_alone`]: every OTHER name here, set beside it, is refused.
#[cfg_attr(not(all(target_os = "linux", feature = "cuda")), allow(dead_code))]
const ALONE: [&str; 6] = [
    "LLVQ_INT4_WIDE_CHECK", "LLVQ_PERS_4E", "LLVQ_GQA_PPL", "LLVQ_GQA_AB", "LLVQ_CONTEXTS", "LLVQ_PREFILL_TOKENS",
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

#[cfg(all(target_os = "linux", feature = "cuda"))]
/// `LLVQ_INT4_WIDE_CHECK=1`: one load with the int4 checks on, no token decoded.
fn int4_wide_check(path: String, device: candle_core::Device, dtype: candle_core::DType) -> anyhow::Result<()> {
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
    Ok(())
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
        // `LLVQ_BENCH_ARMS` is read by `planesbench` only, so this binary
        // lets it pass.
        if served.is_some() {
            for var in llvq_llm::served::MEASUREMENT_MODES.into_iter().filter(|v| *v != "LLVQ_BENCH_ARMS") {
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
            return int4_wide_check(path, device, dtype);
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
            return pers4e::pers_4e_mode(v, path, device, dtype);
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
            return gqa::gqa_ppl_mode(v, path, device, dtype);
        }

        // ---- LLVQ_GQA_AB: attention over the cached heads, unexpanded ----
        //
        // A MEASUREMENT MODE on the bench door, refused beside LLVQ_CONFIG: the
        // served flags are spelled out on the command line. The value is the
        // list of prompt lengths. ONE model, three arms (`run_gqa_ab`), with the
        // three trims of 2026-10-30 in force: cudarc's events go off here as on
        // the served door, which is safe on one stream only, so that is checked.
        if let Ok(v) = std::env::var("LLVQ_GQA_AB") {
            return gqa::gqa_ab_mode(v, path, device, dtype, n_new);
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
            return context::contexts_mode(v, served, path, device, dtype, n_new);
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
            return prefill::prefill_mode(v, served, path, device, dtype);
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
            return door::served_door(cfg, path, device, dtype, n_new);
        }

        bench::bench(path, device, dtype, n_new)
    }
}

#[cfg(test)]
mod tests {
    use super::{alone_in, first_divergence, rate_stats, ALONE};

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
