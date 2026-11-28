//! Gate G5 smoke test: quantize Qwen3-0.6B to ~2 bits and measure the cost.
//!
//! Usage:
//!   cargo run --release -p llvq-llm --features metal --bin smoke -- \
//!       [calib_windows] [calib_len] [eval_windows] [eval_ctx] [device]
//!
//! Reports baseline and quantized perplexity on the *same* windows, so the
//! only difference between the two numbers is the codebook.
//!
//! Deviations from the paper, stated up front:
//!   * calibration on WikiText-2 **train**, not 6,100 DCLM-edu sequences —
//!     same-domain calibration flatters the result, so this is a pipeline
//!     check, not a comparable figure;
//!   * far fewer calibration tokens than the paper uses;
//!   * shape–gain, whose gain-bit count is an *argument* and is printed with
//!     the result — the shipped `leech1c12` spends one. This paragraph used to
//!     say "0 gain bits and the spherical retraction", which was wrong twice:
//!     the published runs spend a gain bit, and with a finite gain codebook
//!     `quantize` has already placed the block on the nearest level's sphere,
//!     so Eq. 17's retraction is a no-op (`docs/fiche-4b.md` §2.3).
//!
//! Environment knobs: `LLVQ_MODEL`, `LLVQ_CALIB` (`dclm-edu` for the paper's
//! own calibration set, `c4` for the out-of-domain control), `LLVQ_ARTIFACT`,
//! `LLVQ_RESUME`, `LLVQ_THREADS`,
//! `LLVQ_DAMPING`, `LLVQ_DTYPE`, and `LLVQ_CALIB_SEED` — the last one draws
//! the calibration windows at random offsets instead of taking a prefix, which
//! is how a **run-to-run error bar** gets measured. Three seeds on 3 blocks is
//! the cheapest thing in this repo that turns a 3 % difference from an
//! anecdote into a result.
//!
//! `LLVQ_H_SHRINK=<ρ>` (default 1, the published path) shrinks the
//! off-diagonal of every layer Hessian towards its diagonal before the
//! rotation — `H ← ρ·H + (1 − ρ)·diag(H)` — the M1 knob of
//! `docs/ROADMAP.md`. It is recorded in the `.state` sidecar only
//! when it is not 1, so a shard produced before the knob existed still resumes
//! under the published configuration, and a shard produced under ρ < 1 refuses
//! a resume that does not say so. See `calib::RunConfig::h_shrink`.
//!
//! ## Running in segments (`LLVQ_RESUME`)
//!
//! A 32B run is ~11.4 h of rented card, and an incident in the tenth hour used
//! to lose the ten hours and the bill both — which is why that run kept not
//! being launched. A run can now be cut in two:
//!
//! ```text
//! LLVQ_ARTIFACT=/out/seg1.llvq  smoke … <codebook> 32 rot
//! LLVQ_RESUME=/out/seg1.llvq LLVQ_ARTIFACT=/out/seg2.llvq  smoke … <codebook> 64 rot
//! ```
//!
//! and `seg2.llvq` is the **whole** artifact — a resumed segment copies its
//! shard forward, so there is no merge step to get wrong. Measured on
//! Qwen3-0.6B, 28 blocks, CPU: the two-segment file is byte-identical to the
//! single run's (`tests/resume.rs`).
//!
//! Three things to know before using it:
//!
//! * **`blocks` is an absolute bound, not a count.** It always was — the loop
//!   tests `t >= limit` — and with a resume the difference becomes visible: a
//!   second segment of a 64-layer model is `64`, never `32`. A bound at or
//!   below the shard's last block is refused rather than silently doing
//!   nothing.
//! * **The restart block is read off the shard**, never passed. One source of
//!   truth: a run that could be told two different things would produce an
//!   artifact with a hole or an overlap, and nothing downstream detects either.
//! * **The hidden states are recomputed, not restored.** On CPU and Metal that
//!   is exact — the test above demands it. On a backend that is not
//!   deterministic between processes (cuBLAS picks algorithms per launch), two
//!   segments are not *guaranteed* to reproduce a single run to the last bit.
//!   The residual is far below the ±0.7 % run-to-run spread this project has
//!   measured across calibration seeds
//!   (`docs/archive/verdicts-lot-b-2026-08-06.md`), but it is a real caveat and the
//!   result line prints it rather than leaving it to a shell history.
//!
//! ## Why every knob here refuses instead of falling back
//!
//! This binary is the one that runs for hours on a rented card and produces
//! the artifact. Two properties follow from that, and neither is optional:
//!
//! 1. **A malformed argument stops the run, before anything is downloaded.**
//!    Every positional and every environment variable is parsed and
//!    range-checked at the top of `main`, ahead of `Checkpoint::fetch`. The
//!    version this replaced wrote `parse().ok().unwrap_or(default)`: `blocks`
//!    mistyped meant `usize::MAX`, i.e. the *full* run — the $5.43 de-risking
//!    pass silently becoming the $61 one. And `leach1c12` (one letter off)
//!    resolved to `leech1`, a **different codebook at a different rate**, with
//!    no warning; that class of confounder already cost this project $12.61.
//!    The doctrine is the one [`llvq_llm::eval::parse_dtype`] and
//!    [`llvq_llm::fused::FusedLayout::parse`] already apply: name the
//!    admissible values, refuse everything else.
//! 2. **The log reconstructs the configuration.** The banner printed before
//!    the download, and the result block printed at the end, carry the
//!    *resolved* codebook — gain bits, shell cap, level cap, class count,
//!    bits per block — not the string the caller typed. The line this
//!    replaced was a hard-coded literal claiming "0 gain bits" on every run
//!    including the shipped 1-gain-bit one (README.md, `docs/fiche-4b.md`
//!    §2.3), so the logs of all three published runs describe a configuration
//!    that never ran.

use candle_core::{DType, Tensor};

use llvq_llm::calib::Codebook;
use llvq_llm::corpus::wikitext2_test;
use llvq_llm::loader::Checkpoint;
use llvq_llm::model::{NoCapture, Qwen3};
use llvq_quant::gptq::{GptqConfig, TailPolicy};
use llvq_llm::calib::ROTATION_SEED;
use llvq_llm::corpus::CalibCorpus;
#[cfg(test)]
use llvq_llm::corpus::{calib_chars, window_starts};

mod codebook;
mod sink;

use codebook::{codebook_line, parse_codebook};
use sink::FileSink;

/// The positional layout, quoted in every refusal so the reader can count.
const USAGE: &str = "n_calib calib_len n_eval eval_ctx device mode codebook \
                     blocks rotation [dump_path]";


/// A positional argument, parsed **strictly**.
///
/// Absent means the default. Present-but-unparsable is an error: the previous
/// `parse().ok().unwrap_or(d)` swallowed `blocks=4x` into `usize::MAX` and ran
/// the full model instead of the four-block de-risking pass.
fn arg<T>(a: &[String], i: usize, name: &str, default: T) -> anyhow::Result<T>
where
    T: std::str::FromStr,
    <T as std::str::FromStr>::Err: std::fmt::Display,
{
    match a.get(i) {
        None => Ok(default),
        Some(s) => s.parse().map_err(|e| {
            anyhow::anyhow!(
                "argument {i} ({name}) = {s:?} is not a number: {e}\n\
                 Expected order: {USAGE}"
            )
        }),
    }
}

/// The gain-refinement stage, selected by the `mode` positional.
///
/// A typo used to land on `NoGs`, because the code tested `mode == "gs"` and
/// `mode == "dc"` and let everything else be the default. That one is *less*
/// dangerous than the codebook — the resolved state is printed — but "less
/// dangerous" is not a reason to accept `gsp`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Mode {
    /// No refinement. The published configuration.
    NoGs,
    /// Algorithm 3's raw closed-form refinement — not sealable at the
    /// advertised rate.
    Gs,
    /// Design C: free magnitude during the loop, the same closed-form solve at
    /// the end of the layer, then a re-projection onto the gain grid
    /// (`docs/archive/retraction-et-gain.md`).
    DesignC,
    /// Spherical GPTQ under the coded gain: the error feedback is formed
    /// against the block retracted to its exact norm, the stored block stays
    /// on the gain grid (`GptqConfig::spherical_feedback`). Same file, same
    /// rate, same decoder as `nogs`.
    Spherical,
}

impl Mode {
    fn parse(v: Option<&str>) -> Result<Self, String> {
        match v {
            None | Some("nogs") => Ok(Self::NoGs),
            Some("gs") => Ok(Self::Gs),
            Some("dc") => Ok(Self::DesignC),
            Some("sph") => Ok(Self::Spherical),
            Some(other) => Err(format!(
                "mode {other:?}: accepted values `nogs` (default), `gs`, `dc` and `sph`"
            )),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::NoGs => "nogs",
            Self::Gs => "gs",
            Self::DesignC => "dc",
            Self::Spherical => "sph",
        }
    }
}

/// The `rotation` positional. `ops/run.py` passes `norot` when it is off, and
/// the docs pass `rot`; anything else is a typo, and a typo used to mean off.
fn parse_rotation(v: Option<&str>) -> Result<Option<u64>, String> {
    match v {
        None | Some("norot") => Ok(None),
        Some("rot") => Ok(Some(ROTATION_SEED)),
        Some(other) => Err(format!(
            "rotation {other:?}: accepted values `rot` and `norot` (default)"
        )),
    }
}

fn main() -> anyhow::Result<()> {
    // ---- everything is resolved and range-checked before any I/O ----
    //
    // Nothing below this block may fall back on a default it was not asked
    // for, and nothing above `Checkpoint::fetch` may cost money. A run that is
    // going to be refused has to be refused now, not after the checkpoint is
    // pulled (26 min on a 65.5 GB model) and the baseline perplexity measured.
    let a: Vec<String> = std::env::args().skip(1).collect();
    let n_calib: usize = arg(&a, 0, "n_calib", 16)?;
    let calib_len: usize = arg(&a, 1, "calib_len", 2048)?;
    let n_eval: usize = arg(&a, 2, "n_eval", 12)?;
    let eval_ctx: usize = arg(&a, 3, "eval_ctx", 2048)?;
    // Zero would be a division by zero when the window count is clamped to the
    // corpus, three seconds after the model finishes loading.
    anyhow::ensure!(calib_len > 0, "calib_len must be > 0");
    anyhow::ensure!(eval_ctx > 0, "eval_ctx must be > 0");
    let device = llvq_llm::eval::device(a.get(4).map(String::as_str).unwrap_or("cpu"))?;
    // Algorithm 3's closed-form gain refinement. Appendix I treats it as part
    // of the 0-gain-bit configuration, not as an extra. `gs` is the raw
    // (unsealable) refinement; `dc` is design C — free magnitude during the
    // loop, the same closed-form solve at the end of the layer, then a
    // re-projection onto the gain grid so the result stays sealable at the
    // advertised rate (docs/archive/retraction-et-gain.md, design C).
    let mode = Mode::parse(a.get(5).map(String::as_str)).map_err(|e| anyhow::anyhow!(e))?;
    let group_scales = mode == Mode::Gs;
    let design_c = mode == Mode::DesignC;
    // `sph` touches the feedback only: nothing about the written file changes,
    // so the codebook's own checks are the same as under `nogs`.
    let spherical_feedback = mode == Mode::Spherical;
    // Which codebook. `kind` is kept as typed so the result line still shows
    // what was asked for; `codebook` is what will actually run, and the two
    // are printed together.
    let kind = a.get(6).cloned().unwrap_or_else(|| "leech".into());
    let codebook = parse_codebook(&kind).map_err(|e| anyhow::anyhow!(e))?;
    // How many transformer blocks to touch. The default is "all of them", so
    // this is the single most expensive argument in the file: the de-risking
    // pass and the full run differ by this number alone, and by ~$56.
    let limit: usize = arg(&a, 7, "blocks", usize::MAX)?;
    anyhow::ensure!(limit > 0, "blocks must be > 0 — 0 quantizes nothing");
    // Incoherence rotation on each linear's input basis (paper Table 9's
    // "Input" column). `rot` to enable, `norot` to state it is off.
    let rotation_seed =
        parse_rotation(a.get(8).map(String::as_str)).map_err(|e| anyhow::anyhow!(e))?;
    // Where to persist the quantized projections, so the model can be probed
    // later instead of being re-quantized.
    let save_to = a.get(9).cloned();
    // `LLVQ_THREADS` caps the encoder pool. A full run wants every core, but
    // an A/B launched on a machine someone is working on should not take the
    // whole machine — the Leech encoder is the part that saturates it.
    let threads = llvq_llm::eval::env_threads(
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4),
    )?;
    // F32 by default — every published run used it, and the identity control
    // depends on it. `LLVQ_DTYPE=bf16` halves the resident model, which is
    // what makes a 32B fit: 131 GB in f32 exceeds every single-card flavor,
    // 65.5 GB in bf16 sits comfortably on a 96 GB one. That is a $130
    // difference on one run.
    //
    // The round-trip proof follows the dtype rather than being weakened by it:
    // `verify_artifact` narrows the decode to the model's precision, so the
    // claim stays "the file decodes to the weights that were evaluated".
    // The Hessian accumulator is F32 regardless (see `Hessian::new`), so
    // calibration precision does not ride on this.
    let dtype = llvq_llm::eval::dtype(DType::F32)?;
    // `LLVQ_CALIB=c4` reproduces the paper's setup: calibrate out of domain,
    // evaluate on wikitext. (The "~12 %" this comment used to claim was a
    // misreading — it measured how much harder C4 is as an *evaluation*
    // corpus, not a calibration advantage. See CLAUDE.md §Qwen3-4B.)
    let calib_env = std::env::var("LLVQ_CALIB").ok();
    let calib = CalibCorpus::parse(calib_env.as_deref()).map_err(|e| anyhow::anyhow!(e))?;
    // Unset = the contiguous prefix every published run used. Set = random
    // offsets, which is how an error bar gets measured. See `window_starts`.
    // A misspelled seed used to mean "prefix", i.e. a different Hessian from
    // the one asked for, on a run whose whole purpose is the spread.
    let calib_seed = match std::env::var("LLVQ_CALIB_SEED") {
        Ok(s) if !s.is_empty() => Some(
            s.parse::<u64>()
                .map_err(|e| anyhow::anyhow!("LLVQ_CALIB_SEED={s:?} is not an integer: {e}"))?,
        ),
        _ => None,
    };
    // Projection types written as int4 g128 instead of lattice codes — the
    // mixed-precision `v_proj` of `docs/ROADMAP.md` §2.3. Empty by default, so
    // an unset variable is the run that was always run, byte for byte.
    //
    // Refused rather than ignored on a name outside `PROJ_TYPES`: a
    // misspelling here would silently produce a wholly lattice file under a
    // label that says mixed.
    let int4_types: Vec<String> = match std::env::var("LLVQ_INT4_TYPES") {
        Ok(v) if !v.trim().is_empty() => {
            let mut out: Vec<String> = Vec::new();
            for t in v.split(',').map(str::trim).filter(|t| !t.is_empty()) {
                anyhow::ensure!(
                    llvq_llm::sealed::PROJ_TYPES.contains(&t),
                    "LLVQ_INT4_TYPES={v:?}: {t:?} is not one of {:?}",
                    llvq_llm::sealed::PROJ_TYPES
                );
                anyhow::ensure!(!out.contains(&t.to_string()), "LLVQ_INT4_TYPES: {t} twice");
                out.push(t.to_string());
            }
            out
        }
        _ => Vec::new(),
    };
    // Hessian damping, relative to `mean(diag H)` — `H + λ·mean(diag H)·I`.
    //
    // 1e-2 was the only value ever passed, on every layer width (2560, 4096,
    // 9728), and it was never swept. That is not a defensible state for the
    // parameter that conditions the whole error compensation, in a repo that
    // sweeps β to the thousandth. `LLVQ_DAMPING` makes {3e-3, 1e-2, 3e-2} on
    // 3 blocks an eight-minute A/B.
    //
    // The prediction is that it changes nothing: because the damping is
    // relative to `mean(diag H)` and an LLM activation Hessian is heavy-tailed
    // — a few massive-activation directions carry most of the trace — that
    // floor already dominates the noisy directions. It is why GPTQ, QuIP# and
    // QTIP all use a single relative `percdamp` whatever the width. Publish
    // the null result; "never measured" is the thing to fix.
    let damping = match std::env::var("LLVQ_DAMPING") {
        Ok(s) => s
            .parse::<f64>()
            .map_err(|_| anyhow::anyhow!("LLVQ_DAMPING={s:?} is not a number"))?,
        Err(_) => 1e-2,
    };
    anyhow::ensure!(
        damping >= 0.0 && damping.is_finite(),
        "LLVQ_DAMPING must be finite and non-negative, got {damping}"
    );
    // Off-diagonal shrinkage of the Hessian estimate (M1). Unset is 1, the
    // published path, and 1 is not a multiply by one: `calib` skips the call.
    let h_shrink = match std::env::var("LLVQ_H_SHRINK") {
        Ok(s) => s
            .parse::<f64>()
            .map_err(|_| anyhow::anyhow!("LLVQ_H_SHRINK={s:?} is not a number"))?,
        Err(_) => 1.0,
    };
    anyhow::ensure!(
        (0.0..=1.0).contains(&h_shrink),
        "LLVQ_H_SHRINK must be in [0, 1] (1 = H as estimated), got {h_shrink}"
    );
    // Multiplies the fitted gain centroids of every matrix. Unset is 1, the
    // published path, and 1 is not a multiply by one: `calib` skips it.
    // Measurement knob of the reconstruction bias, `calib::RunConfig`.
    let gain_scale = match std::env::var("LLVQ_GAIN_SCALE") {
        Ok(s) => s
            .parse::<f64>()
            .map_err(|_| anyhow::anyhow!("LLVQ_GAIN_SCALE={s:?} is not a number"))?,
        Err(_) => 1.0,
    };
    anyhow::ensure!(
        gain_scale.is_finite() && (0.5..=2.0).contains(&gain_scale),
        "LLVQ_GAIN_SCALE must be in [0.5, 2] (1 = the fit as it stands), got {gain_scale}"
    );

    // `LLVQ_GUIDED_WEIGHTS=<dir>` encodes gate_proj and up_proj against
    // gradient-weighted Hessians, one per group of `LLVQ_GUIDED_GROUPS` row
    // groups (`llvq_llm::guided`), each group's weights clipped at
    // `LLVQ_GUIDED_CLIP` times their mean when it is set. Unset: the published
    // path, byte for byte. Read here, before any download, so a malformed
    // value costs nothing.
    let guided_env =
        llvq_llm::guided::from_env(|k| std::env::var(k).ok()).map_err(anyhow::Error::msg)?;
    let guided_manifest = match &guided_env {
        Some(e) => Some(llvq_llm::guided::manifest_sha256(&e.dir)?),
        None => None,
    };

    // `LLVQ_ARTIFACT=<path>` writes the real compressed artifact: packed
    // lattice indices, not reconstructions. The file's size is the bit rate.
    let artifact_path = std::env::var("LLVQ_ARTIFACT")
        .ok()
        .filter(|p| !p.is_empty());
    let repo = std::env::var("LLVQ_MODEL").unwrap_or_else(|_| "Qwen/Qwen3-0.6B".into());
    // `LLVQ_RESUME=<shard>` continues an earlier segment. Unset — the default
    // — is the behaviour every published run had, byte for byte.
    //
    // The block to restart at is **not** a second knob: it is read off the
    // shard (`artifact2::shard_extent`). A run told both where to resume from
    // *and* where to restart would be a run that can be told two different
    // things, and the artifact it produced would have a hole or an overlap
    // that nothing downstream detects.
    let resume_path = std::env::var("LLVQ_RESUME").ok().filter(|p| !p.is_empty());
    codebook.validate_encoding_mode(group_scales, design_c, spherical_feedback, resume_path.is_some())
        .map_err(anyhow::Error::msg)?;
    if let Some(r) = &resume_path {
        // A resume that writes nothing throws away the copy of the shard it
        // just made *and* the blocks it just paid for.
        let Some(a) = &artifact_path else {
            anyhow::bail!(
                "LLVQ_RESUME={r} without LLVQ_ARTIFACT: a resume copies its shard \
                 into its own output, so with no output file it throws away both \
                 the shard and the blocks it just computed."
            );
        };
        anyhow::ensure!(
            a != r,
            "LLVQ_RESUME and LLVQ_ARTIFACT name the same file ({r}): the output \
             is opened for creation, so the shard would be truncated before it \
             is read."
        );
    }
    // What a shard cannot record about the run that wrote it, and what a
    // resume therefore has to be handed separately. See `RunState`.
    //
    // Deliberately **not** in here: the eval windows (they do not touch the
    // artifact), the thread count (`parallel_matches_serial_exactly` pins the
    // loop bit-exact whatever it is, and `ops/run.py` sets it from the
    // flavor's vCPU count — refusing a resume on a wider machine would defeat
    // the point), and the block bound (that is the segment, not the run).
    let mut state = llvq_llm::artifact2::RunState::new();
    // Recorded only when it changes the published path — see the module doc
    // for the resume contract this preserves in both directions.
    if h_shrink < 1.0 {
        state.set("h_shrink", format!("{h_shrink:e}"));
    }
    // A guided file must never pass for a default one: the state beside it
    // names the weights by their manifest's sha256, and a resume refuses a
    // shard encoded under other weights or none.
    if let (Some(e), Some(sha)) = (&guided_env, &guided_manifest) {
        state.set("guided", llvq_llm::guided::state_line(e.groups, sha, e.clip));
    }
    state
        .set("model", &repo)
        .set("codebook", codebook_line(&kind, &codebook))
        .set("calibration", calib.name())
        .set("calib_windows", n_calib)
        .set("calib_len", calib_len)
        .set(
            "calib_seed",
            match calib_seed {
                Some(s) => s.to_string(),
                None => "prefix".into(),
            },
        )
        .set(
            "rotation",
            match rotation_seed {
                Some(s) => format!("{s:#x}"),
                None => "off".into(),
            },
        )
        .set("mode", mode.name())
        .set("damping", format!("{damping:e}"))
        .set("dtype", llvq_llm::eval::dtype_name(dtype))
        // The device *as asked for*, not as candle prints it: `Metal(…
        // DeviceId(1))` is not stable across processes. Two backends do not
        // produce the same hidden states, so this one is load-bearing.
        .set("device", a.get(4).map(String::as_str).unwrap_or("cpu"));

    // ---- the journal header ----
    //
    // Everything an archived log needs to name the object it produced, printed
    // once, before the first byte is downloaded. The rule this enforces: a
    // number in this repo is worthless without the configuration that made it,
    // and the configuration has to be the **resolved** one — `docs/fiche-4b.md`
    // §2.3 spent a day reconstructing a published run because the old header
    // printed a hard-coded literal instead.
    eprintln!("=== configuration ===");
    eprintln!("  model        {repo}");
    eprintln!("  codebook     {}", codebook_line(&kind, &codebook));
    eprintln!(
        "  calibration  {}, {n_calib} × {calib_len} = {} tokens requested, {}",
        calib.name(),
        n_calib.saturating_mul(calib_len),
        match calib_seed {
            Some(s) => format!("offsets drawn (seed {s})"),
            None => "contiguous prefix from token 0".into(),
        }
    );
    eprintln!(
        "  rotation     {}",
        match rotation_seed {
            Some(s) => format!("on (seed {s:#x})"),
            None => "off".into(),
        }
    );
    eprintln!(
        "  mode         {} (group scales {}, design C {}, spherical feedback {})",
        mode.name(),
        if group_scales { "on" } else { "off" },
        if design_c { "on" } else { "off" },
        if spherical_feedback { "on" } else { "off" }
    );
    eprintln!(
        "  blocks       {}",
        if limit == usize::MAX {
            "all".into()
        } else {
            format!("{limit} at most")
        }
    );
    eprintln!("  eval         wikitext-2 test, {n_eval} windows × {eval_ctx}");
    eprintln!("  dtype        {}", llvq_llm::eval::dtype_name(dtype));
    eprintln!("  device       {device:?}, {threads} encoder threads");
    eprintln!("  damping      {damping:e} (relative to mean(diag H))");
    eprintln!(
        "  h_shrink     {h_shrink} ({})",
        if h_shrink < 1.0 {
            "ρ·H + (1−ρ)·diag H, natural basis — M1"
        } else {
            "H as is, published path"
        }
    );
    eprintln!(
        "  gain_scale   {gain_scale} ({})",
        if gain_scale == 1.0 {
            "centroids as fitted, published path"
        } else {
            "fitted centroids multiplied — reconstruction-bias knob"
        }
    );
    eprintln!(
        "  guided       {}",
        match (&guided_env, &guided_manifest) {
            (Some(e), Some(sha)) => format!(
                "gate_proj, up_proj against gradient-weighted Hessians, G = {}, weights {}, \
                 manifest sha256 {sha}{}",
                e.groups,
                e.dir.display(),
                match e.clip {
                    Some(c) => format!(", weights clipped at {c} x the group mean"),
                    None => String::new(),
                }
            ),
            _ => "off (one input Hessian per activation, published path)".into(),
        }
    );
    eprintln!(
        "  artifact     {}",
        artifact_path.as_deref().unwrap_or("(none)")
    );
    eprintln!(
        "  resume       {}",
        resume_path.as_deref().unwrap_or("(no — start at block 0)")
    );
    eprintln!("=====================");

    // ---- the resume is refused here, before anything is downloaded ----
    //
    // Everything a resume can be wrong about that does not need the model:
    // the shard's extent, and the half of the configuration the shard cannot
    // record. Both cost milliseconds. The per-matrix checks — dimensions,
    // shell cap, centroid count, rotation seed, record order — need the model
    // and happen in `resume_from_shard`.
    let start = match &resume_path {
        None => 0usize,
        Some(p) => {
            let (matrices, blocks) = llvq_llm::artifact2::shard_extent(p)?;
            let prev = llvq_llm::artifact2::RunState::read(p)?;
            let diffs = prev.diff(&state);
            anyhow::ensure!(
                diffs.is_empty(),
                "resume refused: {p} was produced under a different configuration.\n  \
                 {}\n\nAn artifact whose halves do not share the same calibration \
                 looks valid and is wrong: nothing downstream detects it, and \
                 nobody can explain the perplexity that comes out.",
                diffs.join("\n  ")
            );
            // The sidecar's own claim about the shard, against the file's. Two
            // independent witnesses, which is what catches a `.state` left
            // over from a *different* shard at the same path.
            if let Some(claim) = prev
                .get(llvq_llm::artifact2::RunState::BLOCKS)
                .and_then(|v| v.parse::<usize>().ok())
            {
                anyhow::ensure!(
                    blocks <= claim,
                    "resume refused: {p} carries {blocks} whole blocks but its \
                     state claims {claim}. The file and the state do not \
                     describe the same object."
                );
                if blocks < claim {
                    eprintln!(
                        "  WARNING: the shard targeted {claim} blocks and holds \
                         {blocks}: run interrupted, the resume restarts at block \
                         {blocks}."
                    );
                }
            }
            eprintln!(
                "resume from {p}: {matrices} matrices, blocks 0..{} already \
                 quantized — quantization restarts at block {blocks}.",
                blocks.saturating_sub(1)
            );
            blocks
        }
    };
    anyhow::ensure!(
        limit > start,
        "blocks = {limit} and the resume starts at block {start}: this run would \
         quantize no block. `blocks` is an absolute bound, not a count of blocks \
         to do."
    );

    let ck = Checkpoint::fetch(&repo)?;
    let tok = ck.tokenizer()?;
    // Guided weights are checked against this run's own calibration tokens
    // before the model is mapped: a wrong fingerprint or a corrupted file is
    // refused in seconds, not after the baseline perplexity. Only under the
    // variable: the published path tokenizes where it always did, below.
    let mut early_cal = None;
    let guided = match &guided_env {
        Some(e) => {
            eprintln!(
                "guided weights: cutting the calibration windows ({}) to check them before the model loads…",
                calib.name()
            );
            let cal =
                llvq_llm::guided::calibration_ids(&tok, calib, n_calib, calib_len, calib_seed)?;
            let g = llvq_llm::guided::GuidedWeights::open(
                &e.dir,
                e.groups,
                e.clip,
                cal.fingerprint(),
                cal.ids.len(),
            )?;
            eprintln!("  guided weights: {}", g.describe());
            early_cal = Some(cal);
            Some(g)
        }
        None => None,
    };
    let vb = ck.var_builder(dtype, &device)?;
    let mut model = Qwen3::new(&ck.config, vb, llvq_llm::kvq::KvMode::F16)?;

    // ---- evaluation windows (fixed, used before and after) ----
    let test_ids = tok
        .encode(wikitext2_test()?.as_str(), false)
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .get_ids()
        .to_vec();
    let n_eval = n_eval.min(test_ids.len() / eval_ctx);
    anyhow::ensure!(n_eval > 0, "corpus shorter than one eval window");

    let ppl = |m: &Qwen3| -> anyhow::Result<f64> {
        let (mut nll, mut cnt) = (0.0, 0usize);
        for w in 0..n_eval {
            let (n, c) =
                m.window_nll(&test_ids[w * eval_ctx..(w + 1) * eval_ctx], &mut NoCapture)?;
            nll += n;
            cnt += c;
        }
        Ok((nll / cnt as f64).exp())
    };

    eprintln!("baseline perplexity over {n_eval} windows of {eval_ctx}…");
    let base = ppl(&model)?;
    eprintln!("  baseline ppl = {base:.4}");

    // ---- calibration windows ----
    eprintln!("tokenizing calibration set ({})…", calib.name());
    // The cut lives in `llvq_llm::guided::calibration_ids`, the one source of
    // truth `bin/calibtokens` also reads, so that guided weights computed on
    // its output weight the very rows this run accumulates. Its refusal is the
    // one that stood here: a run asking for more windows than the corpus holds
    // fails rather than serving fewer in silence.
    let cal = match early_cal.take() {
        Some(c) => c,
        None => llvq_llm::guided::calibration_ids(&tok, calib, n_calib, calib_len, calib_seed)?,
    };
    let available = cal.available;
    eprintln!(
        // `available` is printed alongside because this line is what a
        // volume ladder's control reads back: "we asked for N and got N" is
        // only reassuring next to how many the corpus actually held. The
        // refusal above already refuses N > available, so the two can no
        // longer disagree — this is the line that lets a reader see it.
        "  {n_calib} windows of {calib_len} = {} tokens ({available} available), {}",
        n_calib * calib_len,
        match calib_seed {
            Some(s) => format!("seeded offsets (seed {s})"),
            None => "contiguous prefix from token 0".into(),
        }
    );
    // Printed on every run: two arms that claim the same calibration show the
    // same line, and guided weights are refused unless they carry it.
    eprintln!(
        "  calibration token fingerprint {:016x} ({} ids, FNV-1a 64 over the windows in order)",
        cal.fingerprint(),
        cal.ids.len()
    );

    let mut hidden: Vec<Tensor> = Vec::with_capacity(n_calib);
    for ids in cal.ids.chunks_exact(calib_len) {
        let t = Tensor::from_slice(ids, (1, calib_len), &device)?;
        hidden.push(model.embed_tokens(&t)?);
    }

    // ---- quantize ----
    let cfg = GptqConfig {
        // The codebook decides it: a Leech block is 24 weights, an affine
        // scalar arm's block is its group. `quantize_layer` asserts the two
        // agree, so a mismatch is a panic rather than a silent regrouping.
        block: codebook.block_len(),
        retract: true, // Spherical GPTQ
        group_scales,
        design_c,
        spherical_feedback,
        lambda: 1e-2,
        tail: TailPolicy::KeepExact,
    };
    // The block count is the target, not the model's: `limit` is what the run
    // will actually touch, and printing the model's total instead is how a log
    // ends up describing a 36-block run as a 36-block run *whatever* was asked.
    let n_target = limit.min(model.blocks.len());
    anyhow::ensure!(
        n_target > start,
        "the resume starts at block {start} and the model has only {} blocks under \
         the requested bound: nothing to quantize",
        n_target
    );
    let per_block = llvq_llm::calib::matrices_per_block();
    eprintln!(
        "quantizing blocks {start}..{} of {} of {repo} — {}, retract {}, \
         group scales {}, design C {}, spherical feedback {}, input rotation {}…",
        n_target - 1,
        model.blocks.len(),
        codebook_line(&kind, &codebook),
        if cfg.retract { "on" } else { "off" },
        if group_scales { "on" } else { "off" },
        if design_c { "on" } else { "off" },
        if spherical_feedback { "on" } else { "off" },
        if rotation_seed.is_some() { "on" } else { "off" }
    );

    // The output carries the shard's records *and* this segment's, because a
    // resumed run copies its input forward — see `artifact2`'s module header.
    // The count goes in the header before a byte of payload, so it has to be
    // right here and not be discovered later.
    let n_total = per_block * n_target;
    let t0 = std::time::Instant::now();
    // Row C of `ROADMAP-QUALITY`, off by default: with it the block captures
    // each activation AFTER the matrices upstream of it in the same block
    // have been quantized, four passes instead of one. Refused by name rather
    // than coerced, like every other switch here.
    let sequential_block = llvq_llm::eval::env_flag("LLVQ_SEQ_BLOCK")?;

    let run = llvq_llm::calib::RunConfig {
        sequential_block,
        gptq: cfg,
        int4_types: int4_types.clone(),
        damping,
        h_shrink,
        gain_scale,
        codebook,
        threads,
        start,
        limit,
        rotation_seed,
    };
    // One line per transformer block, with an ETA. On a multi-hour rented job
    // the question is never "did it start" but "is it on track", and a bare
    // elapsed counter cannot answer that. `t` is the **absolute** block index,
    // so the rate is measured over the blocks this segment actually did —
    // dividing the elapsed time by `t + 1` on a segment starting at block 18
    // would report roughly half the true cost per block and an ETA to match.
    let progress = move |t: usize, _n: usize, name: &str| {
        if name == "mlp.down_proj" {
            let done = t + 1 - start;
            let el = t0.elapsed().as_secs_f64();
            let per = el / done as f64;
            let left = n_target.saturating_sub(t + 1);
            eprintln!(
                "  block {:>3}/{n_target}  {:.0} min elapsed  {per:.0} s/block  \
                 ~{:.0} min left  (estimated end at +{:.1} h)",
                t + 1,
                el / 60.0,
                per * left as f64 / 60.0,
                (el + per * left as f64) / 3600.0
            );
        }
    };
    let mut sink = match &artifact_path {
        Some(p) => {
            eprintln!("writing the compressed artifact to {p}");
            let s = FileSink::create(
                p,
                n_total as u32,
                codebook.code_kind(),
                !int4_types.is_empty(),
            )?;
            // Written **now**, not at the end: it describes the file being
            // produced, and a run killed at hour ten must leave a state that
            // names its own configuration rather than the previous run's at
            // the same path. `blocks_done` is the target here, and a resume
            // treats it as an upper bound (see the refusal above).
            let mut st = state.clone();
            st.set(llvq_llm::artifact2::RunState::BLOCKS, n_target);
            let sp = st.write(p)?;
            eprintln!("  run state written to {sp}");
            Some(s)
        }
        None => None,
    };

    // ---- the resume: copy the shard forward, and load it back ----
    //
    // One pass over the shard's bytes does all three jobs (validate, copy,
    // load). It has to happen after the baseline perplexity — the baseline is
    // the *unquantized* model — and before the loop, which starts at `start`
    // assuming blocks below it already hold their quantized weights.
    let resumed = match (&resume_path, sink.as_mut()) {
        (Some(p), Some(s)) => {
            // Tetra stores `TETRA_SHELL_CAP` as a sentinel, not a cap — the
            // same value the sink writes — so the shard check compares what
            // the file actually carries on either arm.
            let (max_shell, gain_bits) = match codebook {
                Codebook::ShapeGain {
                    max_shell,
                    gain_bits,
                    ..
                } => (max_shell, gain_bits),
                Codebook::Tetra { gain_bits, .. } => (llvq_artifact::TETRA_SHELL_CAP, gain_bits),
                _ => anyhow::bail!(
                    "resume refused: only a shape-gain codebook writes an \
                     artifact, so only it can resume one"
                ),
            };
            let expect = llvq_llm::artifact2::ShardExpect {
                kind: codebook.code_kind(),
                int4_types: int4_types.clone(),
                shell_cap: max_shell,
                centroids: 1usize << gain_bits,
                rotation_seed,
            };
            let scan =
                llvq_llm::artifact2::resume_from_shard(&mut model, p, s.writer(), &expect, &device)?;
            // Both halves, or a mixed shard reports fewer weights than the
            // matrices it just copied: the scan keeps the two apart because
            // they are billed at two rates, and this line is a count, not a
            // rate.
            eprintln!(
                "  ✓ {} matrices copied and reloaded ({} weights, {} blocks) in {:.1} s",
                scan.matrices,
                scan.weights + scan.int4_weights,
                scan.blocks,
                scan.seconds
            );
            scan
        }
        _ => llvq_llm::artifact2::ShardScan::default(),
    };
    let mut report = llvq_llm::calib::quantize_model_guided(
        &mut model,
        &mut hidden,
        &run,
        guided.as_ref(),
        progress,
        sink.as_mut()
            .map(|s| s as &mut dyn llvq_llm::calib::MatrixSink),
    )?;
    // The loop reports the **segment**; the file holds the model up to
    // `n_target`. Fold the copied section in, or the bits/weight line divides
    // the whole file's payload by the tail of the model's weights and reports
    // a rate no configuration has — which is the shape of the accounting error
    // of 2026-07-31, in the other direction.
    //
    // The phases and the wall clock are deliberately **not** folded: they
    // measure what this segment did, and the resume's own cost is printed on
    // its own line above.
    report.matrices += resumed.matrices;
    report.weights += resumed.weights;
    report.tail_weights += resumed.tail_weights;
    report.rows += resumed.rows;
    // The two halves are folded into the two totals the scan kept them in.
    // Folding a resumed int4 matrix into `weights` would bill it at the
    // lattice rate and make the same run report a different total for every
    // block it happened to be interrupted at.
    report.int4_matrices += resumed.int4_matrices;
    report.int4_weights += resumed.int4_weights;
    if let Some(s) = sink {
        let (bits, path) = s.finish()?;
        let bytes = std::fs::metadata(&path)?.len();
        // `bits` is the writer's whole payload, int4 records included, so the
        // divisor has to be the whole file's coded weights — see
        // [`Report::quantized_weights`]. The int4 half's own flat rate goes
        // on its own line rather than into this average, because the average
        // of two accountings is the number nobody can reproduce.
        eprintln!(
            "artifact: {:.3} GB on disk, payload {:.4} bits/weight over {} \
             quantized weights",
            bytes as f64 / 1e9,
            bits as f64 / report.quantized_weights() as f64,
            report.quantized_weights(),
        );
        if report.int4_matrices > 0 {
            eprintln!(
                "  a blend: {} of those weights are int4 g128 at a flat {:.4} b/weight",
                report.int4_weights,
                llvq_llm::calib::Report::int4_bits_per_weight(),
            );
        }
        llvq_llm::artifact2::verify_artifact(
            &path,
            &model,
            dtype,
            codebook.code_kind(),
            &int4_types,
        )?;
    }

    // A run with no int4 prints the line it has always printed. As soon as
    // one projection is int4, `report.matrices` and `report.weights` are two
    // different populations — the first counts it, the second does not — and
    // one sentence carrying both, with the lattice rate between them, reads as
    // the rate of the whole file. So the mixed case names its two halves
    // instead. A run whose every projection is int4 has no lattice half at
    // all, and `Report::bits_per_weight` has no block length to divide by.
    let resumed_clause = if resumed.matrices > 0 {
        format!(
            ", of which {} matrices resumed from the shard",
            resumed.matrices
        )
    } else {
        String::new()
    };
    if report.int4_matrices == 0 {
        eprintln!(
            "quantized {} matrices, {} weights ({:.4} bits/weight){resumed_clause}; \
             this segment ran {:.0}s",
            report.matrices,
            report.weights,
            report.bits_per_weight(),
            report.seconds
        );
    } else {
        let lattice_matrices = report.matrices - report.int4_matrices;
        if lattice_matrices > 0 {
            eprintln!(
                "quantized {lattice_matrices} matrices, {} weights on the lattice \
                 ({:.4} bits/weight)",
                report.weights,
                report.bits_per_weight(),
            );
        }
        eprintln!(
            "and {} matrices, {} weights in int4 g128 ({:.4} bits/weight)",
            report.int4_matrices,
            report.int4_weights,
            llvq_llm::calib::Report::int4_bits_per_weight(),
        );
        eprintln!(
            "{} matrices in the file{resumed_clause}; this segment ran {:.0}s",
            report.matrices, report.seconds
        );
    }
    // Where the time went, largest first. Which phase dominates flips with the
    // backend — Leech encoding on Metal, forward passes on a CPU-only job — so
    // the only way to know what to optimize (and which flavor to rent) is to
    // read it off the run itself.
    eprintln!("phases:");
    for (name, secs, pct) in report.phases.ranked() {
        eprintln!("  {name:<22}{secs:>9.1}s{pct:>7.1} %");
    }
    // A 40× difference that is otherwise invisible. Measured on one block of
    // Qwen3-0.6B: factorization 28.4 s without `fast-linalg`, 0.7 s with it,
    // for a bit-identical perplexity. The feature has to stay opt-in — it is
    // what keeps `llvq-quant` free of external dependencies, which the project
    // claims — so the omission is made loud instead.
    if !cfg!(feature = "fast-linalg") {
        eprintln!(
            "\n  WARNING: compiled WITHOUT `fast-linalg`. The factorization runs on \
             the reference\n      implementation, ~40× slower than `faer` \
             for an identical result.\n      Add `--features fast-linalg` \
             before paying for hardware."
        );
    }

    if let Some(path) = &save_to {
        llvq_llm::artifact::save(&model, path)?;
        eprintln!("quantized projections written to {path}");
    }

    let quant = ppl(&model)?;
    // The result block is what gets pasted into `docs/mesures/` and read back
    // months later, so it carries the whole configuration and not a subset:
    // codebook **as resolved**, calibration, rotation, mode, blocks, dtype,
    // device, damping. Everything a reader needs to say which object produced
    // these two numbers, without the stderr stream and without the shell
    // history that launched it.
    println!(
        "\n=== {repo} [{kind}, {n_target} blocks{}, rot {}, mode {}, calib {}/{}], \
         wikitext-2, ctx {eval_ctx}, {n_eval} windows ===",
        if start > 0 {
            format!(" ({start} resumed)")
        } else {
            String::new()
        },
        if rotation_seed.is_some() { "on" } else { "off" },
        mode.name(),
        calib.name(),
        match calib_seed {
            Some(s) => format!("seed {s}"),
            None => "prefix".into(),
        }
    );
    println!(
        "codebook                 = {}",
        codebook_line(&kind, &codebook)
    );
    println!(
        "calibration              = {} × {calib_len} = {} tokens",
        n_calib,
        n_calib * calib_len
    );
    // A segmented run is a fact about the object, not about the invocation:
    // its hidden states were recomputed rather than carried, so on a backend
    // that is not deterministic between processes it is not *guaranteed* to be
    // the same object as a single run. Small next to the ±0.7 % inter-seed
    // spread (docs/archive/verdicts-lot-b-2026-08-06.md), and never to be discovered
    // from a shell history six weeks later.
    if resumed.blocks > 0 {
        println!(
            "segments                 = resumed at block {start} from {} \
             (hidden states recomputed, not restored)",
            resume_path.as_deref().unwrap_or("?")
        );
    }
    let dt = llvq_llm::eval::dtype_name(dtype);
    // Not "FP32": that literal was printed on every run whatever `LLVQ_DTYPE`
    // said, including the bf16 32B pass.
    println!("{:<21}ppl = {base:.4}", format!("baseline ({dt})"));
    println!("LLVQ 2-bit           ppl = {quant:.4}");
    println!("degradation          ×{:.3}", quant / base);
    // Full precision, on its own line, for the one comparison the display
    // lines above cannot serve: **two runs at the same seed must agree**.
    //
    // At ppl ≈ 20, `{:.4}` resolves 5e-6 relative — three orders too coarse to
    // tell a faithful re-run apart from a backend that reordered a reduction.
    // A variance study cannot start without that check, because a σ measured
    // on a non-deterministic pipeline is the sum of two things and separates
    // neither.
    //
    // A *new* line rather than widening the two above: `ops/manifest.py`
    // matches declared values against `ppl = <number>` in a log, and the
    // published numbers are quoted at four decimals. The label deliberately
    // avoids that spelling so the manifest matcher cannot pick it up as a
    // second candidate for the same claim.
    println!("exact-ppl  baseline {base:.12e}  quantized {quant:.12e}");
    println!(
        "effective rate           = {:.4} bits/weight",
        report.bits_per_weight()
    );
    // On the result line, not only in stderr: an A/B whose swept parameter is
    // not printed with its number is an A/B nobody can re-read six weeks later.
    println!("hessian damping          = {damping:e}");
    println!("hessian shrink ρ         = {h_shrink}");
    println!("gain centroid scale      = {gain_scale}");
    println!("dtype / device           = {dt} / {device:?}");
    Ok(())
}

#[cfg(test)]
mod tests;
