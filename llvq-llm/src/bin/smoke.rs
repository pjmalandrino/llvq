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
//!   * calibration on WikiText-2 **train**, not 6 100 DCLM-edu sequences —
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
//! Environment knobs: `LLVQ_MODEL`, `LLVQ_CALIB` (`c4` for the paper's
//! out-of-domain setup), `LLVQ_ARTIFACT`, `LLVQ_RESUME`, `LLVQ_THREADS`,
//! `LLVQ_DAMPING`, `LLVQ_DTYPE`, and `LLVQ_CALIB_SEED` — the last one draws
//! the calibration windows at random offsets instead of taking a prefix, which
//! is how a **run-to-run error bar** gets measured. Three seeds on 3 blocks is
//! the cheapest thing in this repo that turns a 3 % difference from an
//! anecdote into a result.
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
use llvq_llm::corpus::{hf_parquet_text, wikitext2_test};
use llvq_llm::loader::Checkpoint;
use llvq_llm::model::{NoCapture, Qwen3};
use llvq_quant::gptq::{GptqConfig, TailPolicy};

/// The positional layout, quoted in every refusal so the reader can count.
const USAGE: &str = "n_calib calib_len n_eval eval_ctx device mode codebook \
                     blocks rotation [dump_path]";

/// Base seed of the incoherence rotation. Historical value — changing it
/// re-quantizes every model differently, so it is a constant, not a knob.
const ROTATION_SEED: u64 = 0x11_0FEED;

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
                "argument {i} ({name}) = {s:?} n'est pas un nombre : {e}\n\
                 Ordre attendu : {USAGE}"
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
}

impl Mode {
    fn parse(v: Option<&str>) -> Result<Self, String> {
        match v {
            None | Some("nogs") => Ok(Self::NoGs),
            Some("gs") => Ok(Self::Gs),
            Some("dc") => Ok(Self::DesignC),
            Some(other) => Err(format!(
                "mode {other:?} : valeurs admises « nogs » (défaut), « gs » et « dc »"
            )),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::NoGs => "nogs",
            Self::Gs => "gs",
            Self::DesignC => "dc",
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
            "rotation {other:?} : valeurs admises « rot » et « norot » (défaut)"
        )),
    }
}

/// Which corpus the Hessians are built on, from `LLVQ_CALIB`.
///
/// Unknown values used to fall through to wikitext-2 **train** while the log
/// line printed the string that was asked for — so an archived log could read
/// `calib c44` on a run calibrated on wikitext. That is the same shape of
/// defect as the codebook one, on the other input of the same run.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum CalibCorpus {
    Wikitext2Train,
    C4,
    /// Deliberate contamination — calibrate on the very text the eval windows
    /// score. Bounds the ceiling of the calibration family; nobody ships it.
    Wikitext2Test,
}

impl CalibCorpus {
    /// Empty means unset here, unlike the positionals: that is the contract
    /// [`llvq_llm::fused::FusedLayout::parse`] already fixed for environment
    /// variables, and `FOO=${UNSET}` is a normal way to write "leave it alone".
    fn parse(v: Option<&str>) -> Result<Self, String> {
        match v {
            None | Some("") | Some("wikitext2") => Ok(Self::Wikitext2Train),
            Some("c4") => Ok(Self::C4),
            Some("wikitext2-test") => Ok(Self::Wikitext2Test),
            Some(other) => Err(format!(
                "LLVQ_CALIB={other} : valeurs admises « wikitext2 » (défaut), \
                 « c4 » et « wikitext2-test »"
            )),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Wikitext2Train => "wikitext2",
            Self::C4 => "c4",
            Self::Wikitext2Test => "wikitext2-test",
        }
    }
}

/// Gain bits this binary will accept. The published configurations use 0, 1
/// and 2; `Planes14`, the reference runtime layout, freezes the field at one
/// bit. 8 is already 256 levels re-fitted by 40 k-means passes over every
/// block of every matrix — past it, `fit_gain_centroids`'s `1 << k_bits` is
/// the next thing to break, and it breaks three hours in.
const MAX_GAIN_BITS: u32 = 8;

/// Parse the `codebook` positional. **Nothing here falls back.**
///
/// Grammar:
///
/// ```text
/// identity | grid | direction
/// leech[<gain>][c<shell>][L<levels>][f]
/// ```
///
/// in that order — `leech1c12L3` is the 8B configuration, `leech1c12f` charges
/// the pre-2026-07-31 free f16 magnitude. An absent `<gain>` means 1, which is
/// what the bare default `leech` has always meant.
///
/// What the previous parser did instead, and why each one mattered:
///
/// * `strip_prefix("leech").unwrap_or("1")` — `leach1c12` became `leech1`:
///   the **full ball**, 48 index bits, a different rate and a different file;
/// * `c.parse().unwrap_or(13)` — `leech1cx` became the full ball too;
/// * `split_once('c')` is case-sensitive, so `leech1C12` also became the full
///   ball;
/// * `l.parse().unwrap_or(5)` — `leech1L3c12` (fields out of order) became
///   `leech1` with no level cap, silently dropping *both* modifiers.
fn parse_codebook(spec: &str) -> Result<Codebook, String> {
    match spec {
        "identity" => return Ok(Codebook::Identity),
        "grid" => return Ok(Codebook::Grid { step: 0.01 }),
        "direction" => return Ok(Codebook::Direction),
        _ => {}
    }
    if spec.starts_with("int") {
        return parse_int(spec);
    }
    let grammar = "valeurs admises « identity », « grid », « direction », \
                   « int<bits>g<groupe> » et \
                   « leech[<gain>][c<coquille>][L<niveaux>][f] » — \
                   ex. « int3g24 », « leech1 », « leech1c12 », « leech1c12L3 », « leech1c12f »";
    let rest = spec
        .strip_prefix("leech")
        .ok_or_else(|| format!("codebook {spec:?} : {grammar}"))?;

    let (gain, rest) = split_digits(rest);
    let gain_bits: u32 = if gain.is_empty() {
        1
    } else {
        gain.parse()
            .map_err(|e| format!("codebook {spec:?} : gain {gain:?} illisible ({e})"))?
    };
    // The `f` suffix sits at the very end, so it comes off before the fields
    // are read left to right. Order is then enforced by requiring an empty
    // remainder — which is what rejects `leech1L3c12`.
    let (rest, free_magnitude) = match rest.strip_suffix('f') {
        Some(r) => (r, true),
        None => (rest, false),
    };
    let (rest, max_shell) = take_field(spec, rest, 'c', 13)?;
    let (rest, level_cap) =
        take_field(spec, rest, 'L', llvq_search::generic::MAX_LEVELS_ANY as u32)?;
    if !rest.is_empty() {
        return Err(format!("codebook {spec:?} : {rest:?} en trop — {grammar}"));
    }

    if gain_bits > MAX_GAIN_BITS {
        return Err(format!(
            "codebook {spec:?} : {gain_bits} bits de gain, maximum {MAX_GAIN_BITS}"
        ));
    }
    // `BallSearcher::set_shell_cap` asserts this range three hours into a run;
    // `enumerate_classes` asserts the upper half of it one line below.
    if !(2..=llvq_search::classes::MAX_SHELL).contains(&max_shell) {
        return Err(format!(
            "codebook {spec:?} : coquille {max_shell}, admis 2..={}",
            llvq_search::classes::MAX_SHELL
        ));
    }
    let cap = llvq_search::generic::MAX_LEVELS_ANY as u32;
    if level_cap == 0 || level_cap > cap {
        return Err(format!(
            "codebook {spec:?} : {level_cap} niveaux, admis 1..={cap} ({cap} n'exclut rien)"
        ));
    }
    let level_cap = level_cap as usize;
    // The cap combination has to leave something to search. `L1` passes every
    // range check above and empties the codebook; the failure would otherwise
    // surface as a searcher with no classes, well after the model is loaded.
    let classes = class_count(max_shell, level_cap);
    if classes == 0 {
        return Err(format!(
            "codebook {spec:?} : aucune classe ne survit à (coquille ≤ {max_shell}, \
             niveaux ≤ {level_cap}) — le codebook serait vide"
        ));
    }
    Ok(Codebook::ShapeGain {
        gain_bits,
        max_shell,
        free_magnitude,
        level_cap,
    })
}

/// Parse `int<bits>g<groupe>` — the affine scalar arm.
///
/// Both fields are mandatory, and neither falls back. `int3` alone would have
/// to invent a group size, and the group is not a detail: it *is* the GPTQ
/// block (see [`Codebook::block_len`]), so a defaulted one would silently
/// change the error-feedback granularity of the run rather than just its rate.
///
/// `g24` pairs with a Leech arm — same block, same feedback, one variable.
/// `g128` is the field default of AutoGPTQ, and costs `bits + 0.25` instead of
/// `bits + 1.33`.
fn parse_int(spec: &str) -> Result<Codebook, String> {
    let grammar = "grammaire « int<bits>g<groupe> » — bits 1..=8, groupe 8..=1024, \
                   ex. « int3g24 » (apparié au bloc de Leech) ou « int4g128 » \
                   (le défaut d'AutoGPTQ)";
    let rest = spec
        .strip_prefix("int")
        .ok_or_else(|| format!("codebook {spec:?} : {grammar}"))?;
    let (digits, rest) = split_digits(rest);
    if digits.is_empty() {
        return Err(format!("codebook {spec:?} : bits manquants — {grammar}"));
    }
    let bits: u32 = digits
        .parse()
        .map_err(|e| format!("codebook {spec:?} : bits {digits:?} illisibles ({e})"))?;
    let (rest, group) = take_field_required(spec, rest, 'g', grammar)?;
    if !rest.is_empty() {
        return Err(format!("codebook {spec:?} : {rest:?} en trop — {grammar}"));
    }
    // `1u32 << bits` overflows at 32, and a 9-bit "INT" is not a thing anyone
    // ships; the bound is the useful one, not the representable one.
    if !(1..=8).contains(&bits) {
        return Err(format!(
            "codebook {spec:?} : {bits} bits, admis 1..=8 — {grammar}"
        ));
    }
    // The lower bound keeps the f16 scale and zero from dominating the rate
    // (at 8 they already cost 4 bits/weight); the upper one is the narrowest
    // `d_in` of the models this harness loads (1024 on Qwen3-0.6B), past
    // which `TailPolicy::KeepExact` would quantize nothing at all and the run
    // would report a baseline as if it were a result.
    if !(8..=1024).contains(&group) {
        return Err(format!(
            "codebook {spec:?} : groupe {group}, admis 8..=1024 — {grammar}"
        ));
    }
    Ok(Codebook::ScalarGroup {
        bits,
        group: group as usize,
    })
}

/// Read a mandatory `<tag><digits>` field off the front of `rest`.
///
/// The optional twin is [`take_field`]; this one exists because
/// [`parse_int`]'s group has no defensible default.
fn take_field_required<'a>(
    spec: &str,
    rest: &'a str,
    tag: char,
    grammar: &str,
) -> Result<(&'a str, u32), String> {
    let after = rest
        .strip_prefix(tag)
        .ok_or_else(|| format!("codebook {spec:?} : « {tag} » attendu — {grammar}"))?;
    let (digits, rest) = split_digits(after);
    if digits.is_empty() {
        return Err(format!(
            "codebook {spec:?} : « {tag} » doit être suivi d'un nombre — {grammar}"
        ));
    }
    let v = digits
        .parse()
        .map_err(|e| format!("codebook {spec:?} : « {tag}{digits} » illisible ({e})"))?;
    Ok((rest, v))
}

/// Split off the leading run of ASCII digits.
fn split_digits(s: &str) -> (&str, &str) {
    let n = s.len() - s.trim_start_matches(|c: char| c.is_ascii_digit()).len();
    s.split_at(n)
}

/// Read an optional `<tag><digits>` field off the front of `rest`.
///
/// A tag with no digits behind it is an error rather than the default: `c` and
/// `cx` both used to mean "shell 13".
fn take_field<'a>(
    spec: &str,
    rest: &'a str,
    tag: char,
    default: u32,
) -> Result<(&'a str, u32), String> {
    let Some(after) = rest.strip_prefix(tag) else {
        return Ok((rest, default));
    };
    let (digits, tail) = split_digits(after);
    if digits.is_empty() {
        return Err(format!(
            "codebook {spec:?} : « {tag} » doit être suivi d'un nombre"
        ));
    }
    let v = digits
        .parse()
        .map_err(|e| format!("codebook {spec:?} : « {tag}{digits} » illisible ({e})"))?;
    Ok((tail, v))
}

/// Lattice classes the codebook actually keeps: shells `2..=max_shell`, then
/// the level filter `BallSearcher::with_level_cap` applies.
///
/// Printed in the journal because it is the quantity this project keeps
/// arguing about — 301 classes for Λ₂₄(12), 383 for Λ₂₄(13) (CLAUDE.md, §une
/// seule coquille) — and because it is the one number that separates two
/// codebooks at a glance in an archived log.
fn class_count(max_shell: u32, level_cap: usize) -> usize {
    let cs = llvq_search::classes::enumerate_classes(max_shell);
    cs.even.iter().filter(|c| c.n_levels() <= level_cap).count()
        + cs.odd.iter().filter(|c| c.n_levels() <= level_cap).count()
}

/// One line describing the codebook that will actually be used.
///
/// This is the line the run logs were missing. `docs/fiche-4b.md` §2.3 had to
/// reconstruct the shipped configuration from the *result* line, because the
/// configuration line was a literal that said "0 gain bits" while the run used
/// one — on all three published runs.
fn codebook_line(spec: &str, c: &Codebook) -> String {
    let what = match c {
        Codebook::Identity => "lossless control, no code".to_string(),
        Codebook::Grid { step } => format!("scalar grid, step {step}"),
        // The rate is spelled out because it is the one thing a reader will
        // want to compare against a lattice arm, and `bits + 32/group` is not
        // something to do in one's head at the top of a log.
        Codebook::ScalarGroup { bits, group } => format!(
            "affine INT{bits} (AutoGPTQ sym=false), f16 scale + {bits}-bit zero \
             per group of {group}, {:.4} b/weight before the per-row f16 \
             (which this arm does not use), no artifact",
            *bits as f64 + (16.0 + *bits as f64) / *group as f64
        ),
        Codebook::Direction => {
            "direction only, magnitude left a free float — NOT a 2-bit code".to_string()
        }
        Codebook::ShapeGain {
            gain_bits,
            max_shell,
            free_magnitude,
            level_cap,
        } => format!(
            "shape–gain, {gain_bits} gain bit{}, shell ≤ {max_shell}, levels ≤ {level_cap}, \
             {} classes, magnitude {}",
            if *gain_bits == 1 { "" } else { "s" },
            class_count(*max_shell, *level_cap),
            if *free_magnitude {
                "a free f16 (charged as 16 bits)"
            } else {
                "on the gain grid"
            }
        ),
    };
    format!("{spec} → {what}, {:.0} bits/block", c.block_bits())
}

/// Where each calibration window starts in the tokenized corpus.
///
/// `seed = None` reproduces the historical behaviour — a contiguous prefix
/// from token 0 — and stays the default, because that is what every published
/// run used and silently moving it would orphan those numbers.
///
/// But a fixed prefix makes run-to-run variance **unmeasurable**, and that is
/// the gap worth closing: several conclusions in this project rest on 3–7 %
/// differences whose noise floor nobody knows. `LLVQ_CALIB_SEED=<n>` draws the
/// windows at random offsets over the whole corpus, which is what GPTQ, QuIP#
/// and QTIP all do, and which makes "run it under three seeds and look at the
/// spread" possible.
///
/// ⚠️ Do **not** expect a perplexity gain from it. Under `LLVQ_CALIB=c4` the
/// corpus is already hundreds of unrelated web documents concatenated, so "the
/// first 500 documents of a crawl" and "500 documents drawn at random" are
/// nearly the same sample. The deliverable here is the error bar, not the
/// mean — and if the three seeds land far apart, that is itself the finding.
fn window_starts(n: usize, ntokens: usize, len: usize, seed: Option<u64>) -> Vec<usize> {
    let Some(seed) = seed else {
        return (0..n).map(|w| w * len).collect();
    };
    assert!(ntokens >= len, "corpus shorter than one window");
    // Offsets are unaligned on purpose: aligning them to multiples of `len`
    // would sample the same grid the prefix already walks, only in a different
    // order, and would not probe the corpus any more widely.
    let span = (ntokens - len + 1) as u64;
    let mut rng = llvq_core::SplitMix64::new(seed);
    let mut seen = std::collections::HashSet::with_capacity(n);
    let mut out = Vec::with_capacity(n);
    // Distinct windows: a repeated one would weight its tokens twice in the
    // Hessian for nothing. `span` dwarfs `n` on any corpus large enough to
    // calibrate on, so the retry budget is a formality — but an unbounded
    // loop on a short corpus would hang instead of reporting.
    for _ in 0..(64 * n).max(1024) {
        if out.len() == n {
            break;
        }
        let s = (rng.next() % span) as usize;
        if seen.insert(s) {
            out.push(s);
        }
    }
    assert_eq!(
        out.len(),
        n,
        "corpus too short to draw {n} distinct windows of {len}"
    );
    out
}

/// Streams matrices to disk as they are quantized. Buffered, because the
/// index stream is written in 6-byte units and 151 M of them through an
/// unbuffered `File` would be 151 M syscalls.
struct FileSink {
    w: llvq_llm::artifact2::ArtifactWriter<std::io::BufWriter<std::fs::File>>,
    path: String,
}

impl FileSink {
    fn create(path: &str, n: u32) -> anyhow::Result<Self> {
        let f = std::fs::File::create(path)?;
        Ok(Self {
            w: llvq_llm::artifact2::ArtifactWriter::new(
                std::io::BufWriter::with_capacity(1 << 20, f),
                n,
            )?,
            path: path.to_string(),
        })
    }
    /// The raw writer, for the resume path — which copies its input shard's
    /// records straight through rather than pushing decoded matrices.
    fn writer(
        &mut self,
    ) -> &mut llvq_llm::artifact2::ArtifactWriter<std::io::BufWriter<std::fs::File>> {
        &mut self.w
    }
    fn finish(self) -> anyhow::Result<(u64, String)> {
        let bits = self.w.finish()?;
        Ok((bits, self.path))
    }
}

impl llvq_llm::calib::MatrixSink for FileSink {
    fn push(&mut self, m: llvq_llm::artifact2::QuantizedMatrix) -> anyhow::Result<()> {
        // `llvq-artifact` has its own error type on purpose — it carries no
        // dependencies, `anyhow` included. The bridge converts.
        Ok(self.w.push(&m)?)
    }
}

/// Decode the artifact and demand the evaluated weights back, bit for bit.
///
/// This is the whole point of writing a file rather than reporting a number:
/// a rate you cannot decode is a claim, not a measurement. Done one matrix at
/// a time — decoding Qwen3-4B in one go would be 14 GB.
fn verify_artifact(path: &str, model: &Qwen3, dtype: DType) -> anyhow::Result<()> {
    let f = std::fs::File::open(path)?;
    let mut r = std::io::BufReader::with_capacity(1 << 20, f);
    let head = llvq_llm::artifact2::read_header(&mut r)?;
    let n = head.matrices;
    eprintln!("verifying {n} matrices against the evaluated model…");
    // One matrix at a time: a 4B model's codes are 14 GB of lattice points.
    let ix = llvq_search::index::Indexer::new();

    let mut checked = 0usize;
    for _ in 0..n {
        let m = &llvq_llm::artifact2::read_matrix(&mut r, &ix)?;
        let decoded = llvq_llm::artifact2::decode_matrix(m);
        // `name` is `model.layers.{b}.{proj}.weight`.
        let parts: Vec<&str> = m.name.split('.').collect();
        let b: usize = parts[2].parse()?;
        let proj = parts[3..parts.len() - 1].join(".");
        let want = model.blocks[b]
            .linear(&proj)
            .weight()
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        anyhow::ensure!(
            decoded.len() == want.len(),
            "{}: decoded {} weights, model holds {}",
            m.name,
            decoded.len(),
            want.len()
        );
        for (k, (g, e)) in decoded.iter().zip(want.iter()).enumerate() {
            // The decoder works in f32; the model holds its weights at the
            // run's dtype. At F32 this narrowing is the identity and the
            // comparison is the one this proof has always made. At a half
            // precision it becomes "the file decodes to the evaluated weights
            // at the precision the model stores them" — see `eval::narrow`.
            let g = llvq_llm::eval::narrow(*g, dtype);
            anyhow::ensure!(
                g.to_bits() == e.to_bits(),
                "{name} weight {k}: artifact decodes {g:e}, model holds {e:e} \
                 (Δ = {delta:e}). The file is a different model from the one \
                 measured.",
                name = m.name,
                delta = g - e
            );
        }
        checked += decoded.len();
    }
    eprintln!(
        "  ✓ {checked} weights identical, bit for bit (à {})",
        llvq_llm::eval::dtype_name(dtype)
    );
    Ok(())
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
    anyhow::ensure!(calib_len > 0, "calib_len doit être > 0");
    anyhow::ensure!(eval_ctx > 0, "eval_ctx doit être > 0");
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
    // Which codebook. `kind` is kept as typed so the result line still shows
    // what was asked for; `codebook` is what will actually run, and the two
    // are printed together.
    let kind = a.get(6).cloned().unwrap_or_else(|| "leech".into());
    let codebook = parse_codebook(&kind).map_err(|e| anyhow::anyhow!(e))?;
    // How many transformer blocks to touch. The default is "all of them", so
    // this is the single most expensive argument in the file: the de-risking
    // pass and the full run differ by this number alone, and by ~$56.
    let limit: usize = arg(&a, 7, "blocks", usize::MAX)?;
    anyhow::ensure!(limit > 0, "blocks doit être > 0 — 0 ne quantifie rien");
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
    let threads = match std::env::var("LLVQ_THREADS") {
        Ok(s) if !s.is_empty() => {
            let n: usize = s
                .parse()
                .map_err(|e| anyhow::anyhow!("LLVQ_THREADS={s:?} n'est pas un nombre : {e}"))?;
            anyhow::ensure!(n > 0, "LLVQ_THREADS={s:?} : il en faut au moins un");
            n
        }
        _ => std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4),
    };
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
                .map_err(|e| anyhow::anyhow!("LLVQ_CALIB_SEED={s:?} n'est pas un entier : {e}"))?,
        ),
        _ => None,
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
    if let Some(r) = &resume_path {
        // A resume that writes nothing throws away the copy of the shard it
        // just made *and* the blocks it just paid for.
        let Some(a) = &artifact_path else {
            anyhow::bail!(
                "LLVQ_RESUME={r} sans LLVQ_ARTIFACT : une reprise recopie son shard \
                 dans sa propre sortie, donc sans fichier de sortie elle jette à la \
                 fois le shard et les blocs qu'elle vient de calculer."
            );
        };
        anyhow::ensure!(
            a != r,
            "LLVQ_RESUME et LLVQ_ARTIFACT désignent le même fichier ({r}) : la \
             sortie est ouverte en création, donc le shard serait tronqué avant \
             d'être lu."
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
        "  calibration  {}, {n_calib} × {calib_len} = {} tokens demandés, {}",
        calib.name(),
        n_calib.saturating_mul(calib_len),
        match calib_seed {
            Some(s) => format!("offsets tirés (graine {s})"),
            None => "préfixe contigu depuis le token 0".into(),
        }
    );
    eprintln!(
        "  rotation     {}",
        match rotation_seed {
            Some(s) => format!("on (graine {s:#x})"),
            None => "off".into(),
        }
    );
    eprintln!(
        "  mode         {} (group scales {}, design C {})",
        mode.name(),
        if group_scales { "on" } else { "off" },
        if design_c { "on" } else { "off" }
    );
    eprintln!(
        "  blocks       {}",
        if limit == usize::MAX {
            "tous".into()
        } else {
            format!("{limit} au plus")
        }
    );
    eprintln!("  eval         wikitext-2 test, {n_eval} fenêtres × {eval_ctx}");
    eprintln!("  dtype        {}", llvq_llm::eval::dtype_name(dtype));
    eprintln!("  device       {device:?}, {threads} encoder threads");
    eprintln!("  damping      {damping:e} (relatif à mean(diag H))");
    eprintln!(
        "  artifact     {}",
        artifact_path.as_deref().unwrap_or("(aucun)")
    );
    eprintln!(
        "  reprise      {}",
        resume_path.as_deref().unwrap_or("(non — départ au bloc 0)")
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
                "reprise refusée : {p} a été produit sous une autre configuration.\n  \
                 {}\n\nUn artefact dont les moitiés ne partagent pas la même \
                 calibration est valide en apparence et faux : rien en aval ne le \
                 détecte, et la perplexité qui en sort n'est explicable par personne.",
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
                    "reprise refusée : {p} porte {blocks} blocs entiers mais son \
                     état en annonce {claim}. Le fichier et l'état ne décrivent pas \
                     le même objet."
                );
                if blocks < claim {
                    eprintln!(
                        "  ⚠️  le shard visait {claim} blocs et en a {blocks} : run \
                         interrompu, la reprise repart du bloc {blocks}."
                    );
                }
            }
            eprintln!(
                "reprise depuis {p} : {matrices} matrices, blocs 0..{} déjà \
                 quantifiés — la quantification repart au bloc {blocks}.",
                blocks.saturating_sub(1)
            );
            blocks
        }
    };
    anyhow::ensure!(
        limit > start,
        "blocks = {limit} et la reprise démarre au bloc {start} : ce run ne \
         quantifierait aucun bloc. « blocks » est une borne absolue, pas un \
         nombre de blocs à faire."
    );

    let ck = Checkpoint::fetch(&repo)?;
    let tok = ck.tokenizer()?;
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
    let train = match calib {
        CalibCorpus::C4 => {
            // A different shard from the one `bin/ppl` evaluates on —
            // otherwise calibrating on C4 and scoring on C4 is the same text
            // twice.
            // Sized from what this run asked for, not from a literal.
            //
            // 🕳️ The literal was `8_000_000`, and lot B measured what it
            // actually yields: **847 windows of 2048**
            // (`docs/archive/verdicts-lot-b-2026-08-06.md:33`), i.e. 4.61
            // characters per token. Any run asking for more than ~13× the
            // published volume was served ~13× — the `min` below did it
            // without a word in the log. A calibration-volume ladder built on
            // that would have published two rungs measuring the same point.
            //
            // Six characters per token is the measured 4.61 plus margin. It
            // only avoids a needless second read: what actually guarantees
            // the volume is the `ensure!` below.
            let want = n_calib.saturating_mul(calib_len).saturating_mul(6);
            llvq_llm::corpus::c4_calibration(want)?
        }
        CalibCorpus::Wikitext2Test => {
            // The calibration *oracle* (pistes-battre-q4.md P3): deliberate
            // contamination — calibrate on the very text the eval windows
            // score. Not a config anyone ships; it bounds the ceiling of the
            // whole calibration family (volume, corpus, length) in one
            // 3-block run.
            wikitext2_test()?
        }
        CalibCorpus::Wikitext2Train => hf_parquet_text(
            "Salesforce/wikitext",
            "wikitext-2-raw-v1/train-00000-of-00001.parquet",
        )?
        .join("\n\n"),
    };
    let train_ids = tok
        .encode(train.as_str(), false)
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .get_ids()
        .to_vec();
    anyhow::ensure!(n_calib > 0, "n_calib doit être > 0");
    // 🚨 This used to be `n_calib.min(train_ids.len() / calib_len)` — a
    // **silent clamp**. A run that asked for 2048 windows and got 847 said so
    // nowhere except in a line nobody diffs, and reported its result under the
    // volume it *asked* for. That is the §7 rule of `CLAUDE.md` — a mechanism
    // that skips when its resource is missing must fail, not pass — applied to
    // a run rather than to a test.
    let available = train_ids.len() / calib_len;
    anyhow::ensure!(
        available >= n_calib,
        "le corpus de calibration ne porte que {available} fenêtres de {calib_len} \
         ({} tokens lus) — {n_calib} demandées. Ce run échoue ici plutôt que d'en \
         servir moins en silence : un volume publié doit être un volume mesuré.",
        train_ids.len()
    );
    eprintln!(
        // `available` is printed alongside because this line is what a
        // volume ladder's control reads back: "we asked for N and got N" is
        // only reassuring next to how many the corpus actually held. The
        // `ensure!` above already refuses N > available, so the two can no
        // longer disagree — this is the line that lets a reader see it.
        "  {n_calib} windows of {calib_len} = {} tokens ({available} available), {}",
        n_calib * calib_len,
        match calib_seed {
            Some(s) => format!("seeded offsets (seed {s})"),
            None => "contiguous prefix from token 0".into(),
        }
    );

    let starts = window_starts(n_calib, train_ids.len(), calib_len, calib_seed);
    let mut hidden: Vec<Tensor> = Vec::with_capacity(n_calib);
    for &s in &starts {
        let ids = &train_ids[s..s + calib_len];
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
        lambda: 1e-2,
        tail: TailPolicy::KeepExact,
    };
    // The block count is the target, not the model's: `limit` is what the run
    // will actually touch, and printing the model's total instead is how a log
    // ends up describing a 36-block run as a 36-block run *whatever* was asked.
    let n_target = limit.min(model.blocks.len());
    anyhow::ensure!(
        n_target > start,
        "la reprise démarre au bloc {start} et le modèle n'a que {} blocs sous \
         la borne demandée : rien à quantifier",
        n_target
    );
    let per_block = llvq_llm::calib::matrices_per_block();
    eprintln!(
        "quantizing blocks {start}..{} of {} of {repo} — {}, retract {}, \
         group scales {}, design C {}, input rotation {}…",
        n_target - 1,
        model.blocks.len(),
        codebook_line(&kind, &codebook),
        if cfg.retract { "on" } else { "off" },
        if group_scales { "on" } else { "off" },
        if design_c { "on" } else { "off" },
        if rotation_seed.is_some() { "on" } else { "off" }
    );

    // The output carries the shard's records *and* this segment's, because a
    // resumed run copies its input forward — see `artifact2`'s module header.
    // The count goes in the header before a byte of payload, so it has to be
    // right here and not be discovered later.
    let n_total = per_block * n_target;
    let t0 = std::time::Instant::now();
    let run = llvq_llm::calib::RunConfig {
        gptq: cfg,
        damping,
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
                "  bloc {:>3}/{n_target}  {:.0} min écoulées  {per:.0} s/bloc  \
                 reste ~{:.0} min  (fin estimée à +{:.1} h)",
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
            let s = FileSink::create(p, n_total as u32)?;
            // Written **now**, not at the end: it describes the file being
            // produced, and a run killed at hour ten must leave a state that
            // names its own configuration rather than the previous run's at
            // the same path. `blocks_done` is the target here, and a resume
            // treats it as an upper bound (see the refusal above).
            let mut st = state.clone();
            st.set(llvq_llm::artifact2::RunState::BLOCKS, n_target);
            let sp = st.write(p)?;
            eprintln!("  état du run écrit dans {sp}");
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
            let (max_shell, gain_bits) = match codebook {
                Codebook::ShapeGain {
                    max_shell,
                    gain_bits,
                    ..
                } => (max_shell, gain_bits),
                _ => anyhow::bail!(
                    "reprise refusée : seul un codebook shape–gain écrit un \
                     artefact, donc seul lui peut en reprendre un"
                ),
            };
            let expect = llvq_llm::artifact2::ShardExpect {
                shell_cap: max_shell,
                centroids: 1usize << gain_bits,
                rotation_seed,
            };
            let scan =
                llvq_llm::artifact2::resume_from_shard(&mut model, p, s.writer(), &expect, &device)?;
            eprintln!(
                "  ✓ {} matrices recopiées et rechargées ({} poids, {} blocs) en {:.1} s",
                scan.matrices, scan.weights, scan.blocks, scan.seconds
            );
            scan
        }
        _ => llvq_llm::artifact2::ShardScan::default(),
    };
    let mut report = llvq_llm::calib::quantize_model_capturing(
        &mut model,
        &mut hidden,
        &run,
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
    if let Some(s) = sink {
        let (bits, path) = s.finish()?;
        let bytes = std::fs::metadata(&path)?.len();
        eprintln!(
            "artifact: {:.3} GB on disk, payload {:.4} bits/weight over {} \
             quantized weights",
            bytes as f64 / 1e9,
            bits as f64 / (report.weights - report.tail_weights) as f64,
            report.weights - report.tail_weights,
        );
        verify_artifact(&path, &model, dtype)?;
    }

    eprintln!(
        "quantized {} matrices, {} weights ({:.4} bits/weight){}; \
         ce segment a tourné {:.0}s",
        report.matrices,
        report.weights,
        report.bits_per_weight(),
        if resumed.matrices > 0 {
            format!(
                ", dont {} matrices reprises du shard",
                resumed.matrices
            )
        } else {
            String::new()
        },
        report.seconds
    );
    // Where the time went, largest first. Which phase dominates flips with the
    // backend — Leech encoding on Metal, forward passes on a CPU-only job — so
    // the only way to know what to optimize (and which flavor to rent) is to
    // read it off the run itself.
    eprintln!("phases :");
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
            "\n  ⚠️  compilé SANS `fast-linalg` : la factorisation tourne sur \
             l'implémentation\n      de référence, ~40× plus lente que `faer` \
             pour un résultat identique.\n      Ajouter `--features fast-linalg` \
             avant de payer du matériel."
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
            format!(" ({start} repris)")
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
            "segments                 = repris au bloc {start} depuis {} \
             (états cachés recalculés, pas restaurés)",
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
    println!("dtype / device           = {dt} / {device:?}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const N: usize = 8;
    const LEN: usize = 2048;
    const TOKENS: usize = 4_000_000;

    fn shape_gain(spec: &str) -> (u32, u32, bool, usize) {
        match parse_codebook(spec).unwrap_or_else(|e| panic!("{spec}: {e}")) {
            Codebook::ShapeGain {
                gain_bits,
                max_shell,
                free_magnitude,
                level_cap,
            } => (gain_bits, max_shell, free_magnitude, level_cap),
            other => panic!("{spec} resolved to {other:?}, not a shape–gain"),
        }
    }

    /// Every spelling this repo has ever launched has to keep meaning exactly
    /// what it meant, or the strict parser silently reinterprets the published
    /// runs instead of protecting them.
    #[test]
    fn the_documented_spellings_still_resolve() {
        // `leech` is the default when the positional is absent, and has always
        // meant one gain bit over the full ball.
        assert_eq!(shape_gain("leech"), (1, 13, false, 5));
        assert_eq!(shape_gain("leech1"), (1, 13, false, 5));
        // The sealed 4B artifact.
        assert_eq!(shape_gain("leech1c12"), (1, 12, false, 5));
        // The 8B run.
        assert_eq!(shape_gain("leech1c12L3"), (1, 12, false, 3));
        // The free-magnitude arm of docs/archive/retraction-et-gain.md.
        assert_eq!(shape_gain("leech1c12f"), (1, 12, true, 5));
        assert_eq!(shape_gain("leech0"), (0, 13, false, 5));
        assert_eq!(shape_gain("leech2c12"), (2, 12, false, 5));
        assert!(matches!(
            parse_codebook("identity").unwrap(),
            Codebook::Identity
        ));
        assert!(matches!(
            parse_codebook("grid").unwrap(),
            Codebook::Grid { .. }
        ));
        assert!(matches!(
            parse_codebook("direction").unwrap(),
            Codebook::Direction
        ));
    }

    /// **The $12.61 test.** One letter off used to resolve to a *different*
    /// codebook at a different rate, and the log printed the string that was
    /// asked for, so nothing looked wrong.
    ///
    /// Each case below is a way the old parser fell back, not a made-up typo:
    /// a bad prefix (`unwrap_or("1")`), an unreadable shell (`unwrap_or(13)`),
    /// the wrong case (`split_once('c')` is case-sensitive), an unreadable
    /// level cap (`unwrap_or(5)`), and fields out of order (`L` was split
    /// before `c`, so the tail was swallowed whole).
    #[test]
    fn a_misspelled_codebook_is_refused() {
        for spec in [
            "leach1c12",  // the audit's typo → used to be leech1, full ball
            "Leech1c12",  // capitalized → used to be leech1
            "leech1C12",  // capital C → used to be the full ball
            "leech1cx",   // unreadable shell → used to be the full ball
            "leech1c",    // bare tag → used to be the full ball
            "leech1c12L", // bare tag → used to be 5 levels
            "leech1c12Lx",
            "leech1L3c12", // fields swapped → used to drop *both*
            "leech1c12x",  // trailing junk → used to be ignored
            "leech1c12 ",  // trailing space, the classic shell accident
            "leechx",
            "",
            "leech-1",
            "identity2",
            "gridd",
        ] {
            assert!(
                parse_codebook(spec).is_err(),
                "{spec:?} was accepted instead of refused"
            );
        }
    }

    /// Ranges the encoder asserts on hours into a run, asserted here in
    /// microseconds instead. `set_shell_cap` panics outside `2..=13`, and a
    /// level cap of 0 leaves the searcher with no classes at all.
    #[test]
    fn out_of_range_caps_are_refused() {
        for spec in [
            "leech1c14",
            "leech1c1",
            "leech1c0",
            "leech1c99",
            "leech9",
            "leech64",
            "leech1c12L0",
            "leech1c12L6",
            "leech1c12L99",
        ] {
            assert!(
                parse_codebook(spec).is_err(),
                "{spec:?} was accepted instead of refused"
            );
        }
        // …and the edges of those ranges stay legal, or the guard is a wall
        // rather than a range.
        assert_eq!(shape_gain("leech1c2").1, 2);
        assert_eq!(shape_gain("leech1c13").1, 13);
        assert_eq!(shape_gain("leech1c12L5").3, 5);
        assert_eq!(shape_gain("leech8").0, 8);
    }

    /// A codebook can be **empty** while every individual range holds, and
    /// that combination has to be refused on its own.
    ///
    /// This test exists because the guard survived mutation without it: both
    /// caps were in range on every case the other tests tried, so deleting the
    /// emptiness check changed nothing they could see. The measured grid says
    /// otherwise — `class_count(m, 1)` is **0** for every shell up to 5, and
    /// only reaches 1 at shell 6 — so `leech1c5L1` passes `2..=13` and
    /// `1..=5` and leaves the searcher with nothing to search. Same lesson as
    /// CLAUDE.md §5: an assertion that never exercises the parameter it covers
    /// tests nothing.
    #[test]
    fn an_empty_codebook_is_refused() {
        for spec in ["leech1c2L1", "leech1c3L1", "leech1c4L1", "leech1c5L1"] {
            assert_eq!(class_count_of(spec), 0, "{spec} is not empty after all");
            assert!(
                parse_codebook(spec).is_err(),
                "{spec:?} is an empty codebook and was accepted"
            );
        }
        // Shell 6 is where a single class appears — degenerate but searchable,
        // so it stays legal. The guard is an emptiness check, not a taste one.
        assert_eq!(class_count(6, 1), 1);
        assert_eq!(shape_gain("leech1c6L1"), (1, 6, false, 1));
    }

    /// The class count the parser would compute for `spec`, ignoring whether
    /// it accepts it — so the test above can assert emptiness *and* refusal
    /// rather than assuming they coincide.
    fn class_count_of(spec: &str) -> usize {
        let (shell, levels) = spec
            .trim_start_matches("leech1c")
            .split_once('L')
            .expect("the probes above are all `leech1c<m>L<l>`");
        class_count(shell.parse().unwrap(), levels.parse().unwrap())
    }

    /// The class count is the one number that separates two codebooks at a
    /// glance in an archived log, so it has to be the real one: 301 classes
    /// for Λ₂₄(12) — 180 even and 121 odd — against 383 for Λ₂₄(13)
    /// (CLAUDE.md, §une seule coquille). A count computed on the wrong shell
    /// would make the journal line agree with itself and with nothing else.
    #[test]
    fn the_class_count_is_the_documented_one() {
        assert_eq!(class_count(12, llvq_search::generic::MAX_LEVELS_ANY), 301);
        assert_eq!(class_count(13, llvq_search::generic::MAX_LEVELS_ANY), 383);
        // The level cap has to actually cut, or `L3` would be decoration.
        assert!(class_count(12, 3) < 301);
        assert!(class_count(12, 3) > 0);
    }

    /// The defect this file existed with on all three published runs: a
    /// hard-coded "0 gain bits" on a configuration that spends one. The line
    /// has to report the resolved value, and it has to carry enough to name
    /// the object — shell cap, level cap, class count, bits per block.
    #[test]
    fn the_journal_line_reports_the_real_gain_bits() {
        let line = codebook_line("leech1c12", &parse_codebook("leech1c12").unwrap());
        // The trailing comma is load-bearing: without it `"1 gain bit"` is a
        // substring of `"1 gain bits"`, and a frozen plural survives the test.
        assert!(line.contains("1 gain bit,"), "{line}");
        assert!(!line.contains("0 gain"), "{line}");
        assert!(line.contains("shell ≤ 12"), "{line}");
        assert!(line.contains("301 classes"), "{line}");
        // 47 index bits over Λ₂₄(12) plus one gain bit — the sealed artifact's
        // rate, and the number the whole G5 section is written around.
        assert!(line.contains("48 bits/block"), "{line}");

        let zero = codebook_line("leech0c12", &parse_codebook("leech0c12").unwrap());
        assert!(zero.contains("0 gain bits"), "{zero}");
        assert!(zero.contains("47 bits/block"), "{zero}");

        // And the two must not print the same thing, which is what a literal
        // did.
        assert_ne!(line, zero);
    }

    /// A free magnitude is not a 2-bit code, and the line has to say so:
    /// mistaking one for the other is the accounting error of 2026-07-31,
    /// 2.1117 announced for 2.7338 real.
    #[test]
    fn the_journal_line_separates_a_free_magnitude() {
        let f = codebook_line("leech1c12f", &parse_codebook("leech1c12f").unwrap());
        assert!(f.contains("free f16"), "{f}");
        assert!(f.contains("63 bits/block"), "{f}");
        let g = codebook_line("leech1c12", &parse_codebook("leech1c12").unwrap());
        assert_ne!(f, g);
    }

    /// **The $56 test.** `blocks` mistyped used to become `usize::MAX`, i.e.
    /// the full run — the de-risking pass turning into the thing it exists to
    /// de-risk.
    #[test]
    fn a_malformed_positional_is_refused_not_defaulted() {
        let a: Vec<String> = [
            "64",
            "2048",
            "12",
            "4096",
            "metal",
            "nogs",
            "leech1c12",
            "4x",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let e = arg::<usize>(&a, 7, "blocks", usize::MAX).unwrap_err();
        assert!(format!("{e}").contains("blocks"), "{e}");
        // An empty positional is a shell accident (`$UNSET` expanded), not a
        // request for the default.
        let empty = vec![String::new()];
        assert!(arg::<usize>(&empty, 0, "n_calib", 16).is_err());
        // Absent stays the default, or every short invocation in the docs
        // stops working.
        assert_eq!(arg::<usize>(&a, 9, "blocks", 7).unwrap(), 7);
        assert_eq!(arg::<usize>(&a, 7, "blocks", usize::MAX).ok(), None);
        assert_eq!(arg::<usize>(&a, 0, "n_calib", 16).unwrap(), 64);
    }

    /// `gs` / `nogs` / `dc`, and nothing that merely looks like them.
    #[test]
    fn a_misspelled_mode_is_refused() {
        assert_eq!(Mode::parse(None).unwrap(), Mode::NoGs);
        assert_eq!(Mode::parse(Some("nogs")).unwrap(), Mode::NoGs);
        assert_eq!(Mode::parse(Some("gs")).unwrap(), Mode::Gs);
        assert_eq!(Mode::parse(Some("dc")).unwrap(), Mode::DesignC);
        for bad in ["", "GS", "gsp", "no-gs", "designc", "off"] {
            assert!(Mode::parse(Some(bad)).is_err(), "{bad:?} was accepted");
        }
    }

    /// `ops/run.py` passes `norot` when the rotation is off and the docs pass
    /// `rot`; anything else used to silently mean off — and the rotation is
    /// the single largest quality lever measured in this project (×2.290 →
    /// ×1.811).
    #[test]
    fn a_misspelled_rotation_is_refused() {
        assert_eq!(parse_rotation(None).unwrap(), None);
        assert_eq!(parse_rotation(Some("norot")).unwrap(), None);
        assert_eq!(parse_rotation(Some("rot")).unwrap(), Some(ROTATION_SEED));
        for bad in ["", "ROT", "rott", "on", "true", "1"] {
            assert!(parse_rotation(Some(bad)).is_err(), "{bad:?} was accepted");
        }
    }

    /// The affine scalar arm parses, and **nothing about it falls back**.
    ///
    /// The group is not a cosmetic field: it *is* `GptqConfig::block` (see
    /// `Codebook::block_len`), so a defaulted one would change the run's
    /// error-feedback granularity and not merely its rate. That is why
    /// `int3` alone is an error rather than "group 128".
    #[test]
    fn the_scalar_arm_parses_and_never_falls_back() {
        assert!(matches!(
            parse_codebook("int3g24").unwrap(),
            Codebook::ScalarGroup { bits: 3, group: 24 }
        ));
        assert!(matches!(
            parse_codebook("int4g128").unwrap(),
            Codebook::ScalarGroup {
                bits: 4,
                group: 128
            }
        ));
        for spec in [
            "int3",      // no group — would have to be invented
            "intg24",    // no bits
            "int3g",     // tag with no digits behind it
            "int0g24",   // zero bits
            "int9g24",   // past the 8-bit bound
            "int3g4",    // group below the f16 overhead bound
            "int3g2048", // group past the narrowest d_in this harness loads
            "int3g24x",  // trailing junk
            "int3G24",   // the tag is case-sensitive, as `c` is for leech
        ] {
            assert!(
                parse_codebook(spec).is_err(),
                "{spec:?} should be refused, not resolved"
            );
        }
    }

    /// 🚨 The block length follows the codebook, and the rate follows the
    /// block length.
    ///
    /// While every codebook was 24 wide, `Report::bits_per_weight` could
    /// divide by `llvq_core::DIM` and be right by accident. The first one
    /// that is not would have made it report a rate no file has, and nothing
    /// downstream would have noticed — the failure `Codebook::block_len`
    /// exists to make unwriteable.
    #[test]
    fn the_block_length_follows_the_codebook_and_the_rate_follows_it() {
        assert_eq!(parse_codebook("leech1c12").unwrap().block_len(), 24);
        assert_eq!(parse_codebook("identity").unwrap().block_len(), 24);
        assert_eq!(parse_codebook("int3g24").unwrap().block_len(), 24);
        assert_eq!(parse_codebook("int3g128").unwrap().block_len(), 128);

        // `bits + (16 + bits)/group`: an f16 scale and a zero point packed
        // at `bits` width, which is what a deployed INT-k file stores.
        //
        // 🕳️ This read `bits + 32/group` until the upstream source was
        // fetched — charging the zero point a second f16 it does not cost.
        // That overstated `int3g24` by 0.54 b/weight, on the arm whose whole
        // job is to be the field's own quantizer.
        for (spec, bits, group) in [
            ("int2g24", 2.0, 24.0),
            ("int3g24", 3.0, 24.0),
            ("int3g128", 3.0, 128.0),
            ("int4g128", 4.0, 128.0),
        ] {
            let c = parse_codebook(spec).unwrap();
            let want = group * bits + 16.0 + bits;
            assert!(
                (c.block_bits() - want).abs() < 1e-9,
                "{spec}: block_bits {} != {want}",
                c.block_bits()
            );
            let per_weight = c.block_bits() / c.block_len() as f64;
            assert!(
                (per_weight - (bits + (16.0 + bits) / group)).abs() < 1e-9,
                "{spec}: {per_weight} b/weight"
            );
        }
    }

    /// An unknown `LLVQ_CALIB` used to fall through to wikitext-2 train while
    /// the log printed the string that was asked for — the codebook defect,
    /// on the other input of the same run.
    #[test]
    fn an_unknown_calibration_corpus_is_refused() {
        assert_eq!(
            CalibCorpus::parse(None).unwrap(),
            CalibCorpus::Wikitext2Train
        );
        assert_eq!(
            CalibCorpus::parse(Some("")).unwrap(),
            CalibCorpus::Wikitext2Train
        );
        assert_eq!(CalibCorpus::parse(Some("c4")).unwrap(), CalibCorpus::C4);
        assert_eq!(
            CalibCorpus::parse(Some("wikitext2-test")).unwrap(),
            CalibCorpus::Wikitext2Test
        );
        for bad in ["c44", "C4", "wikitext", "wikitext2-train", "dclm-edu"] {
            assert!(
                CalibCorpus::parse(Some(bad)).is_err(),
                "{bad:?} was accepted"
            );
        }
    }

    /// Names printed in the journal have to parse back, or a log cannot be
    /// replayed from what it says — the property `eval::printed_names_parse_back`
    /// already fixes for the dtype.
    #[test]
    fn printed_names_parse_back() {
        for m in [Mode::NoGs, Mode::Gs, Mode::DesignC] {
            assert_eq!(Mode::parse(Some(m.name())).unwrap(), m);
        }
        for c in [
            CalibCorpus::Wikitext2Train,
            CalibCorpus::C4,
            CalibCorpus::Wikitext2Test,
        ] {
            assert_eq!(CalibCorpus::parse(Some(c.name())).unwrap(), c);
        }
    }

    /// No seed must reproduce the historical prefix exactly. Every published
    /// number was measured on it, so a change here reinterprets them.
    #[test]
    fn no_seed_is_the_contiguous_prefix() {
        let s = window_starts(N, TOKENS, LEN, None);
        assert_eq!(s, (0..N).map(|w| w * LEN).collect::<Vec<_>>());
    }

    /// Every window must fit. An off-by-one in the span would slice past the
    /// end of the token vector and panic three hours into a run.
    #[test]
    fn every_window_fits_in_the_corpus() {
        // Deliberately tight: the last legal start is exactly `ntokens - len`.
        for tokens in [TOKENS, N * LEN, N * LEN + 1] {
            for &s in &window_starts(N, tokens, LEN, Some(7)) {
                assert!(s + LEN <= tokens, "window at {s} overruns {tokens}");
            }
        }
    }

    /// A seed has to be reproducible, and two seeds have to disagree —
    /// otherwise the three runs that are supposed to produce an error bar
    /// would produce the same number three times and report a spread of zero.
    #[test]
    fn seeds_are_reproducible_and_distinct() {
        assert_eq!(
            window_starts(N, TOKENS, LEN, Some(1)),
            window_starts(N, TOKENS, LEN, Some(1))
        );
        assert_ne!(
            window_starts(N, TOKENS, LEN, Some(1)),
            window_starts(N, TOKENS, LEN, Some(2))
        );
    }

    /// The point of seeding is to leave the head of the corpus, not to shuffle
    /// within it. Drawing from `0..n·len` would satisfy every test above and
    /// sample exactly the same tokens as the prefix.
    #[test]
    fn seeded_windows_leave_the_prefix() {
        let s = window_starts(N, TOKENS, LEN, Some(3));
        assert!(
            s.iter().any(|&x| x > N * LEN),
            "every window landed inside the prefix: {s:?}"
        );
    }

    /// Repeated windows would weight their tokens twice in the Hessian.
    #[test]
    fn windows_are_distinct() {
        let s = window_starts(64, TOKENS, LEN, Some(5));
        let uniq: std::collections::HashSet<_> = s.iter().copied().collect();
        assert_eq!(uniq.len(), s.len());
    }
}
