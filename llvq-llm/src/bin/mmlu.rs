//! MMLU, through **our** pipeline, on the file we actually ship.
//!
//! Usage:
//!   `cargo run --release -p llvq-llm --features metal --bin mmlu -- <model> [device] [limit]`
//!
//! `<model>` is either a sealed `.llvq` — the deliverable, loaded exactly as
//! `bin/run` loads it — or a Hugging Face repo id for the unquantized
//! reference.
//!
//! ## Why not a standard harness on a dequantized checkpoint
//!
//! Because that measures *our weights inside someone else's engine*. The
//! weights are bit-for-bit identical either way — that part is verified — but
//! the inference path is not: different framework, different accumulation
//! order, different attention implementation. We have the experimental proof:
//! MLX and this pipeline, fed the same exported checkpoint, diverge on the
//! fifth token of a greedy continuation. For a number that claims to say what
//! the shipped package is worth, that gap is not acceptable.
//!
//! So the harness runs here. The price is that a home-made harness could be
//! subtly non-standard, and a score nobody can compare is worthless — which is
//! why the protocol has a built-in test.
//!
//! ## The protocol, and the test of the protocol
//!
//! Hendrycks 5-shot, the configuration every 2-bit paper reports:
//!
//! * the five worked examples come from the `dev` split **of the same
//!   subject**, in order;
//! * the header names the subject, underscores turned back into spaces;
//! * each block is `question / A. … / B. … / C. … / D. … / Answer: X`;
//! * the scored question ends at `Answer:` and the four options are compared
//!   by the logit of the single tokens ` A`, ` B`, ` C`, ` D` at the final
//!   position — one forward pass per question, not four;
//! * the score is the **micro** average — one weight per question over the
//!   whole test split — which is what `lm-eval-harness` reports and therefore
//!   what the paper's 70.2 / 60.7 are. See [`micro`]: this is the axis that
//!   moves the number by several points and it must never be left implicit
//!   again;
//! * the model dtype, printed with the score. It defaults to F16 here and to
//!   F32 in `bin/ppl`, so an MMLU score and a perplexity are not by default
//!   two measurements of one object — `LLVQ_DTYPE` is what makes them one.
//!
//! **Run the FP16 baseline first.** The paper reports 70.2 on Qwen3-4B; if
//! this harness does not land there, the protocol is wrong and no other number
//! it produces means anything. That is the same discipline as the identity
//! control of Phase 5: the test you re-run first when a result looks odd.
//!
//! ## The per-question dump, and why it is not optional
//!
//! Set `LLVQ_MMLU_DUMP=<path>` and every scored question lands in a CSV. Two
//! arms are always scored on the *same* questions — the sample depends only on
//! the subject name's length — so their results are **paired data**, and every
//! error bar this project has published so far is the unpaired one, which is
//! the wrong test and the conservative one. The paired statistics live in
//! `bin/mmlupair`; they need this file and nothing else.
//!
//! The asymmetry is what makes it mandatory rather than nice: writing the file
//! costs one `writeln!` per forward pass, and *not* writing it costs the whole
//! run again — 0.8 h per arm at `limit=40`, 16.5 h at census on the Mac, or
//! 0.75–1.35 $ per arm on a rented L40S. The three-arm campaigns of 2026-08-06
//! and 2026-08-08 were run without it: their per-question answers are gone, and
//! the −0.28 pp that carries "4-bit is indistinguishable from f16 at 4B" cannot
//! be tested without paying for those runs a second time.
//!
//! ## The sampling plan (`LLVQ_MMLU_ALLOC`)
//!
//! The reported figure is the stratified micro, which weights each subject by
//! its population, and MMLU's populations span a factor of 15. Giving all 57
//! subjects the same 40 questions therefore spends the budget where it buys the
//! least. Spending the same 2,280 questions in proportion to the populations
//! takes the **accuracy** bar from 1.355 pp to 0.925 pp, a factor of 1.464
//! (*computed*, `docs/data/mmlu-dumps/mmlu-4b-llvq.csv`), and the same 1.355 pp
//! bar can be had for 1,191 questions instead of 2,280.
//!
//! The intervals this campaign publishes are paired, and their factor is a
//! different number. Recomputed on the eight pairs of dumps on disk, holding
//! each subject's variance of per-question differences fixed and moving only
//! the counts, it runs from 1.32 (8B llvq/f16) to 1.65 (14B llvq/awq)
//! (*computed*, `docs/data/mmlu-dumps/`). Quote the range, and use 1.32 when a
//! conclusion has to survive the worst case.
//!
//! None of that reaches a number already paid for. The proportional plan is not
//! a re-reading of an existing dump: at this budget it asks 630 questions the
//! flat plan never asked and drops 630 it did ask, so the two samples nest
//! subject by subject and neither contains the other (see
//! `the_plan_on_the_real_populations`). `bin/mmlupair` refuses two dumps whose
//! question sets differ, and `bin/mmlu` has no resume, so re-barring a
//! published result costs one full MMLU run per arm. The plan is for runs not
//! yet paid for.
//!
//! `flat` is the default, so a run that sets nothing draws exactly the sample
//! every dump on disk was drawn with. See [`Alloc`].
//!
//! ## Attribution arms (`LLVQ_RESTORE_F16`)
//!
//! `LLVQ_RESTORE_F16=k_proj` (a comma list of the seven projection types, or
//! `all`) scores the sealed file with that type taken **from the checkpoint at
//! f16**, all layers at once, everything else as shipped. Seven such arms plus
//! the shipped file, paired on the dumps, are the error budget by function
//! that `docs/ROADMAP.md` (M2) asks for. The checkpoint is the one
//! `LLVQ_MODEL` names — required, never defaulted, because the default would
//! be the 0.6B — and the restoration is written into the label, the dump
//! header and the result line, so an arm cannot be mistaken for the deliverable.
//! On a checkpoint argument the variable is refused: a knob that is silently
//! ignored is the A/B that lies. See `llvq_llm::sealed::RestoreF16`.

use candle_core::{DType, IndexOp, Tensor};
use llvq_llm::corpus::{mmlu_split, MmluItem};
use llvq_llm::model::NoCapture;
use std::collections::BTreeMap;
use std::io::Write;

/// `underscored_subject` → `underscored subject`, as the standard prompt wants.
fn pretty(subject: &str) -> String {
    subject.replace('_', " ")
}

/// Which questions of a subject get scored, and **where they came from**.
///
/// The parquet index is zipped in *before* the shuffle because [`MmluItem`]
/// carries no identifier: once Fisher–Yates has run, the only stable key to a
/// question is its position in the parquet order, and that is precisely what
/// the shuffle destroys. Without this, a per-question dump cannot be joined
/// across two runs — and joining across arms is the entire point of a paired
/// test, which is the statistic this campaign publishes.
///
/// The seed depends only on the *length* of the subject name, so the sample is
/// identical across models by construction rather than by convention.
///
/// The shuffle runs over the whole subject and *then* truncates, so two depths
/// nest: the shallower sample is a prefix of the deeper one. One branch escapes
/// that, and [`Alloc`] can reach it: when `limit >= picked.len()` nothing is
/// shuffled and the stratum comes back in parquet order. The sample is then the
/// whole stratum, so it still contains every shallower sample **as a set**,
/// which is all a join on `(subject, index)` needs; only the row order differs.
fn select<'a>(items: &[&'a MmluItem], subject: &str, limit: usize) -> Vec<(usize, &'a MmluItem)> {
    let mut picked: Vec<(usize, &'a MmluItem)> = items.iter().copied().enumerate().collect();
    if limit < picked.len() {
        let mut rng = llvq_core::SplitMix64::new(0x6_11B0 ^ subject.len() as u64);
        for i in (1..picked.len()).rev() {
            picked.swap(i, (rng.next() % (i as u64 + 1)) as usize);
        }
        picked.truncate(limit);
    }
    picked
}

/// Floor on the questions asked of any one subject.
///
/// The stratified variance divides by `n−1` (see [`micro_stderr`]), so a
/// stratum of one contributes no bar at all and a stratum of zero contributes
/// no estimate. Two is the smallest count that leaves both defined.
const MIN_PER_SUBJECT: usize = 2;

/// How the question budget is spread over the subjects, from `LLVQ_MMLU_ALLOC`.
///
/// The reported figure is the stratified micro (see [`micro`]), which weights
/// each subject by its population, and MMLU's populations span a factor of 15.
/// A flat allocation therefore spends most of the budget where it buys the
/// least: `abstract_algebra` carries weight 100/14042 and gets the same 40
/// questions as `professional_law`, which carries 1534/14042. Spending the
/// same 2,280 questions in proportion to the weights divides the accuracy bar
/// by 1.464, and the paired bar by 1.32 to 1.65 depending on the arm. The
/// module note gives both, with what they cost.
///
/// `Flat` is the default, and a run that sets nothing draws exactly the
/// questions it drew before — same sample, same token fingerprint. That
/// control is what keeps every dump already on disk joinable.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Alloc {
    /// `limit` questions per subject, whatever the subject holds.
    Flat,
    /// A total budget spread in proportion to the populations. `None` takes
    /// the budget from the positional limit, as `limit × subjects`, which is
    /// the constant-budget comparison against `Flat`.
    Proportional(Option<usize>),
}

impl Alloc {
    fn from_env() -> anyhow::Result<Self> {
        Self::parse(&std::env::var("LLVQ_MMLU_ALLOC").unwrap_or_default())
    }

    /// An unknown value is refused rather than defaulted: a sampling plan that
    /// silently falls back to another one is an A/B that lies about its own
    /// error bar.
    fn parse(raw: &str) -> anyhow::Result<Self> {
        let raw = raw.trim();
        if raw.is_empty() || raw == "flat" {
            return Ok(Alloc::Flat);
        }
        if raw == "proportional" {
            return Ok(Alloc::Proportional(None));
        }
        if let Some(total) = raw.strip_prefix("proportional=") {
            let total: usize = total.parse().map_err(|_| {
                anyhow::anyhow!("LLVQ_MMLU_ALLOC=proportional=<total>: {total:?} is not a count")
            })?;
            anyhow::ensure!(total > 0, "LLVQ_MMLU_ALLOC=proportional=0 asks for no question");
            return Ok(Alloc::Proportional(Some(total)));
        }
        anyhow::bail!(
            "LLVQ_MMLU_ALLOC={raw:?} is not a sampling plan. Accepted: \
             `flat` (default, `limit` per subject), `proportional` (budget = \
             limit × subjects), `proportional=<total questions>`"
        )
    }

    /// Questions to ask of each subject, in the order of `populations`.
    ///
    /// The proportional plan is the largest-remainder method under two
    /// constraints: never below [`MIN_PER_SUBJECT`], never above the stratum's
    /// own population. Remainders are compared as integers — `n·ΣN − B·N` — so
    /// the plan is exactly reproducible and does not depend on a rounding mode.
    fn plan(&self, populations: &[usize], limit: usize) -> anyhow::Result<Vec<usize>> {
        match self {
            Alloc::Flat => Ok(populations.iter().map(|&n| n.min(limit)).collect()),
            Alloc::Proportional(explicit) => {
                let subjects = populations.len();
                anyhow::ensure!(subjects > 0, "no subject to spread a budget over");
                let total_pop: usize = populations.iter().sum();
                anyhow::ensure!(total_pop > 0, "the subjects hold no question");
                let budget = match explicit {
                    Some(b) => *b,
                    None => {
                        anyhow::ensure!(
                            limit != usize::MAX,
                            "LLVQ_MMLU_ALLOC=proportional with no limit has no budget to \
                             spread: pass a limit, whose budget is limit × subjects, or \
                             write LLVQ_MMLU_ALLOC=proportional=<total questions>. A \
                             census already scores every stratum whole."
                        );
                        limit
                            .checked_mul(subjects)
                            .ok_or_else(|| anyhow::anyhow!("limit × subjects overflows"))?
                    }
                };
                let floors: Vec<usize> =
                    populations.iter().map(|&n| MIN_PER_SUBJECT.min(n)).collect();
                let floor_total: usize = floors.iter().sum();
                anyhow::ensure!(
                    budget >= floor_total,
                    "a budget of {budget} questions cannot give {MIN_PER_SUBJECT} to each \
                     of {subjects} subjects: the stratified variance divides by n−1, so a \
                     stratum below that contributes no bar"
                );
                // A budget that covers the whole split is a census, and every
                // stratum is scored whole. Returning it here keeps the branch
                // below on the strict inequality it needs to terminate.
                if budget >= total_pop {
                    return Ok(populations.to_vec());
                }
                // Distance from the exact share, scaled by ΣN to stay integral.
                let off = |n: usize, pop: usize| -> i128 {
                    n as i128 * total_pop as i128 - budget as i128 * pop as i128
                };
                let mut n: Vec<usize> = populations
                    .iter()
                    .zip(&floors)
                    .map(|(&pop, &floor)| {
                        let quota =
                            (budget as u128 * pop as u128 / total_pop as u128) as usize;
                        quota.clamp(floor, pop)
                    })
                    .collect();
                let mut placed: usize = n.iter().sum();
                // Lifting small strata to the floor can overshoot; take the
                // excess back from whoever sits furthest above its exact share.
                while placed > budget {
                    let take = (0..subjects)
                        .filter(|&i| n[i] > floors[i])
                        .max_by_key(|&i| (off(n[i], populations[i]), std::cmp::Reverse(i)))
                        .expect("the floors fit inside the budget");
                    n[take] -= 1;
                    placed -= 1;
                }
                while placed < budget {
                    let give = (0..subjects)
                        .filter(|&i| n[i] < populations[i])
                        .min_by_key(|&i| (off(n[i], populations[i]), i))
                        .expect("the budget fits inside the population");
                    n[give] += 1;
                    placed += 1;
                }
                Ok(n)
            }
        }
    }

    /// One line for the dump header and the result line. An arm whose sampling
    /// plan is not printed with its bar is an arm nobody can re-read.
    fn describe(&self, plan: &[usize]) -> String {
        let name = match self {
            Alloc::Flat => "flat",
            Alloc::Proportional(_) => "proportional",
        };
        let total: usize = plan.iter().sum();
        let (lo, hi) = (
            plan.iter().copied().min().unwrap_or(0),
            plan.iter().copied().max().unwrap_or(0),
        );
        let spread = if lo == hi {
            format!("{lo} per subject")
        } else {
            format!("{lo}..{hi} per subject")
        };
        format!("{name}, {spread}, {total} questions")
    }
}

/// One worked example, or the scored question when `answer` is `None`.
fn block(it: &MmluItem, answer: Option<usize>) -> String {
    let mut s = format!("{}\n", it.question.trim());
    for (i, c) in it.choices.iter().enumerate() {
        s.push_str(&format!("{}. {}\n", ["A", "B", "C", "D"][i], c.trim()));
    }
    s.push_str("Answer:");
    if let Some(a) = answer {
        s.push_str(&format!(" {}\n\n", ["A", "B", "C", "D"][a]));
    }
    s
}

/// One subject's result: what it scored, out of how many we asked, out of how
/// many the test split holds.
#[derive(Clone, Debug)]
struct SubjectScore {
    subject: String,
    right: usize,
    /// Questions actually put to the model.
    scored: usize,
    /// Questions the subject holds in the `test` split, sampled or not.
    population: usize,
}

impl SubjectScore {
    fn rate(&self) -> f64 {
        if self.scored == 0 {
            0.0
        } else {
            self.right as f64 / self.scored as f64
        }
    }
}

/// The **micro** average: one weight per question of the test split.
///
/// This is the axis that decides the number, and it is not a detail of
/// presentation. MMLU's test split is violently unbalanced —
/// `professional_law` holds 1,534 questions, `abstract_algebra` 100, a ratio
/// of 15 — so the two averages are different statistics, not two roundings of
/// one.
///
/// Pooling `Σright / Σscored` computes the micro average **only when every
/// subject is scored whole**. Under a `limit`, every subject contributes the
/// same count, and that pooled ratio is algebraically the *unweighted mean of
/// the 57 subject rates* — the macro average. That silently over-weights the
/// small STEM subjects by up to 2.5× and under-weights law by 6×, which is
/// precisely where 2-bit quantization does its damage: the profile of our own
/// run has abstract algebra at chance and law above 80 %. A macro/micro swap
/// therefore moves the quantized arm much more than the baseline, and produces
/// two errors pointing in opposite directions — which is exactly the signature
/// one would otherwise read as "not a protocol shift".
///
/// So the subject rates are re-weighted by their true population. With no
/// limit this reduces to `Σright / Σscored` bit for bit; with a limit it is
/// the stratified estimator of that same quantity.
fn micro(scores: &[SubjectScore]) -> f64 {
    let pop: f64 = scores.iter().map(|s| s.population as f64).sum();
    if pop == 0.0 {
        return 0.0;
    }
    scores
        .iter()
        .map(|s| s.population as f64 * s.rate())
        .sum::<f64>()
        / pop
}

/// The **macro** average: one weight per subject. Reported alongside so the
/// two can never again be confused for one another.
fn macro_avg(scores: &[SubjectScore]) -> f64 {
    if scores.is_empty() {
        return 0.0;
    }
    scores.iter().map(SubjectScore::rate).sum::<f64>() / scores.len() as f64
}

/// Standard error of [`micro`] under stratified sampling without replacement.
///
/// `Var = Σ wₛ²·(pₛ(1−pₛ)/(nₛ−1))·(1 − nₛ/Nₛ)` with `wₛ = Nₛ/ΣN`. The finite
/// population correction is what makes this honest at both ends: a subject
/// scored whole contributes exactly zero, so a full run reports ±0.00 — the
/// remaining uncertainty is then no longer *sampling* uncertainty and claiming
/// a bar would be a lie.
fn micro_stderr(scores: &[SubjectScore]) -> f64 {
    let pop: f64 = scores.iter().map(|s| s.population as f64).sum();
    if pop == 0.0 {
        return 0.0;
    }
    scores
        .iter()
        .map(|s| {
            if s.scored <= 1 || s.population == 0 {
                return 0.0;
            }
            let (n, big_n) = (s.scored as f64, s.population as f64);
            let p = s.rate();
            let w = big_n / pop;
            w * w * (p * (1.0 - p) / (n - 1.0)) * (1.0 - n / big_n)
        })
        .sum::<f64>()
        .sqrt()
}

/// First line of a dump. `bin/mmlupair` refuses a file that does not open with
/// it — the version is what lets the format change later without silently
/// feeding an old file to a reader that expects a new column.
const DUMP_VERSION: &str = "# llvq-mmlu-dump v1";

/// The column line. **The reader resolves columns by name, never by
/// position**, because the writer lives here and the parser lives in another
/// binary: two files that cannot share code cannot share a struct either, and
/// a positional contract between them would break silently the first time
/// someone inserts a column. Names make that a loud error instead.
const DUMP_COLUMNS: &str =
    "subject,index,population,qhash,answer,pick,correct,logit_a,logit_b,logit_c,logit_d";

/// One dump line.
///
/// Three fields are here for reasons that are not obvious from the name:
///
/// * `population` — the subject's size in the *test* split, not the number of
///   questions asked. The published figure is the **stratified** micro (see
///   [`micro`]), so the stratum weight is part of the datum. A dump without it
///   can only reproduce the pooled rate, which is a different statistic — the
///   exact confusion §3ter of `CLAUDE.md` cost this project a session to
///   untangle, and re-introducing it in the dump would let it back in through
///   the analysis tool.
/// * `qhash` — [`llvq_llm::eval::token_fingerprint`] over the tokens of *this*
///   prompt. The run-level fingerprint on the result line proves two arms saw
///   the same stream; a per-question hash proves it question by question,
///   which is what a paired join actually needs. It also survives the one case
///   the run-level fingerprint cannot express: a census and a `limit=40` run
///   share 2,280 questions but necessarily print different run fingerprints,
///   so only the per-question hash can certify the overlap.
/// * the four logits, verbatim. `pick` is an argmax and throws away the
///   margin: a question missed by 1e-4 and one missed by 8 are the same row
///   otherwise. `{}` on an `f32` is the shortest round-tripping form, so the
///   file re-reads bit for bit and a later analysis can rank confidence,
///   compute a margin, or re-derive the pick without a second forward pass.
#[allow(clippy::too_many_arguments)]
fn dump_row(
    subject: &str,
    index: usize,
    population: usize,
    qhash: u64,
    answer: usize,
    pick: usize,
    logits: [f32; 4],
) -> String {
    format!(
        "{subject},{index},{population},{qhash:016x},{answer},{pick},{},{},{},{},{}",
        u8::from(pick == answer),
        logits[0],
        logits[1],
        logits[2],
        logits[3]
    )
}

/// The trailer, written once the loop is over.
///
/// It carries the run fingerprint — which is only known at the end, so it
/// cannot be a header — and it doubles as a **completion marker**. A job killed
/// at a platform timeout (the HF Jobs default is 30 min, and a census needs
/// hours) leaves a dump that parses, scores, and is short by however many
/// subjects never ran. Requiring this line turns that silent truncation into a
/// refusal.
fn dump_trailer(fingerprint: u64, questions: usize) -> String {
    format!("# end fingerprint={fingerprint:016x} questions={questions}")
}

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let model_arg = a
        .first()
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("give a sealed .llvq path or a HF repo id"))?;
    let device = llvq_llm::eval::device(a.get(1).map(String::as_str).unwrap_or("cpu"))?;
    // Questions per subject, for a cheap protocol check before the full run.
    // Sampled at random from a fixed seed, never the first N: MMLU test sets
    // are not shuffled, and the head of a subject is not a fair sample of it.
    let limit: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(usize::MAX);
    // F16 here, F32 in `bin/ppl` — internally consistent per metric, and a
    // confound the moment the two are compared. `LLVQ_DTYPE` moves either one
    // onto the other, and the resolved value is printed with the score.
    let dtype = llvq_llm::eval::dtype(DType::F16)?;
    // Resolved once, here: `model.rs` reads no environment variable, so the
    // mode travels by value from this line down to every `KvCache`. An unknown
    // name is an error, never a silent fallback — a typo would make an A/B lie.
    let kv_mode = llvq_llm::kvq::KvMode::from_env().map_err(anyhow::Error::msg)?;
    // Resolved here too, before the model is even fetched: a typo in a sampling
    // plan must cost a second, not an hour of forward passes.
    let alloc = Alloc::from_env()?;

    // ---- the model: the shipped artifact, or the reference checkpoint ----
    //
    // The sealed path is shared with `bin/run` and `bin/ppl` — see
    // `llvq_llm::sealed` for why having three copies of it was a problem and
    // not just duplication.
    // Resolved once, like `kv_mode`: the restoration travels by value into the
    // loader, and an unknown name is an error rather than a fallback.
    let restore = llvq_llm::sealed::RestoreF16::from_env().map_err(anyhow::Error::msg)?;
    // ---- the served arm: the kernel, not a reconstruction of it ----
    //
    // 🕳️ Until 2026-09-11 this binary had TWO arms and neither was the served
    // object. `sealed::load_with_restored` rebuilds every projection into a
    // dense f16 tensor and multiplies with candle; the fused kernel it exists
    // to score never ran. That is not a detail here: MMLU picks its answer
    // from four logits that can sit within an f16 ulp of each other, so
    // "the tokens match" — which is what `bin/fusedrun` proves — does not
    // carry to "the score is the same".
    //
    // The arm is entered only by `LLVQ_CONFIG` naming a served config, and
    // never by inference from the file: every published bar in this
    // repository was measured on the dense arm, and a binary that silently
    // switched would make the next one incomparable to all of them while
    // looking like a bug fix.
    let served = llvq_llm::served::Served::from_env().map_err(anyhow::Error::msg)?;
    let (model, tok, label, restore_note) = if let Some(cfg) = &served {
        anyhow::ensure!(
            llvq_llm::sealed::is_sealed_path(&model_arg),
            "LLVQ_CONFIG names a served config, but {model_arg} is not a sealed file. \
             The served path reads a .llvq; a checkpoint has nothing to transcode."
        );
        anyhow::ensure!(
            restore.is_empty(),
            "{}={} beside LLVQ_CONFIG: a restoration takes matrices out \
             of the served object, so the two together would score neither.",
            match restore.prec() {
                llvq_llm::sealed::RestorePrec::F16 => "LLVQ_RESTORE_F16",
                llvq_llm::sealed::RestorePrec::Q4 { .. } => "LLVQ_RESTORE_Q4",
            },
            restore.describe()
        );
        println!("{}", cfg.provenance());
        #[cfg(all(target_os = "linux", feature = "cuda"))]
        {
            let f = llvq_llm::fused_cuda::load_resolved(
                &model_arg,
                &device,
                dtype,
                cfg.layout,
                cfg.embed,
                cfg.rot_share,
                cfg.fuse,
                // The KV mode rides the config too, so one file decides every
                // choice and `LLVQ_KV` cannot move half of them.
                cfg.kv,
                Some("LLVQ_CONFIG"),
            )?;
            (
                f.model,
                f.tokenizer,
                format!("{model_arg} [LLVQ 2-bit, SERVED KERNEL, {}]", cfg.layout.name()),
                None,
            )
        }
        #[cfg(not(all(target_os = "linux", feature = "cuda")))]
        {
            anyhow::bail!(
                "LLVQ_CONFIG={} asks for the served kernel, which needs Linux, an \
                 NVIDIA card and --features cuda. Unset it to score the dense \
                 reconstruction instead — and say which one produced the number.",
                cfg.path.display()
            )
        }
    } else if llvq_llm::sealed::is_sealed_path(&model_arg) {
        // A restoration reads the checkpoint the file was sealed from, named by
        // `LLVQ_MODEL` as for `bin/seal` — required here, because its default
        // elsewhere is Qwen3-0.6B and a 4B file would only find out at the
        // first shape mismatch.
        let ck = if restore.is_empty() {
            None
        } else {
            let repo = std::env::var("LLVQ_MODEL").map_err(|_| {
                anyhow::anyhow!(
                    "LLVQ_RESTORE_F16={} requires LLVQ_MODEL=<checkpoint>: the restored \
                     matrices come from there",
                    restore.describe()
                )
            })?;
            Some(llvq_llm::loader::Checkpoint::fetch(&repo)?)
        };
        let s = llvq_llm::sealed::load_with_restored(
            &model_arg,
            dtype,
            &device,
            kv_mode,
            &restore,
            ck.as_ref(),
        )?;
        let note = s.restore_note();
        let label = match &note {
            Some(n) => format!("{model_arg} [LLVQ 2-bit, sealed; {n}]"),
            None => format!("{model_arg} [LLVQ 2-bit, sealed]"),
        };
        (s.model, s.tokenizer, label, note)
    } else {
        anyhow::ensure!(
            restore.is_empty(),
            "LLVQ_RESTORE_F16={} only applies to a sealed file; {model_arg} is a \
             checkpoint, which is already all f16",
            restore.describe()
        );
        let ck = llvq_llm::loader::Checkpoint::fetch(&model_arg)?;
        let tok = ck.tokenizer()?;
        let vb = ck.var_builder(dtype, &device)?;
        (
            llvq_llm::model::Qwen3::new(&ck.config, vb, kv_mode)?,
            tok,
            format!("{model_arg} [reference checkpoint]"),
            None,
        )
    };
    eprintln!(
        "model: {label}\ndevice: {device:?}, dtype {}",
        llvq_llm::eval::dtype_name(dtype)
    );

    // ---- the four answer tokens, resolved once ----
    //
    // The scored continuation is a single token — " A" and not "A" — because
    // the prompt ends at "Answer:" with no trailing space. If the tokenizer
    // ever split one of them, comparing single logits would silently compare
    // the wrong things, so it is checked rather than assumed.
    let mut answer_ids = [0u32; 4];
    for (i, letter) in [" A", " B", " C", " D"].iter().enumerate() {
        let ids = tok
            .encode(*letter, false)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .get_ids()
            .to_vec();
        anyhow::ensure!(
            ids.len() == 1,
            "{letter:?} tokenizes to {ids:?}, not a single token — the \
             single-logit comparison would be meaningless"
        );
        answer_ids[i] = ids[0];
    }

    // ---- data ----
    eprintln!("loading MMLU…");
    let test = mmlu_split("test")?;
    let dev = mmlu_split("dev")?;
    let mut shots: BTreeMap<String, Vec<&MmluItem>> = BTreeMap::new();
    for it in &dev {
        shots.entry(it.subject.clone()).or_default().push(it);
    }
    let mut by_subject: BTreeMap<String, Vec<&MmluItem>> = BTreeMap::new();
    for it in &test {
        by_subject.entry(it.subject.clone()).or_default().push(it);
    }
    eprintln!(
        "{} questions, {} subjects, {} dev examples\n",
        test.len(),
        by_subject.len(),
        dev.len()
    );

    // ---- the sampling plan ----
    //
    // How many questions each subject gets, decided before the first forward
    // pass and printed with the score. `Flat` is the default and reproduces
    // every dump on disk question for question.
    let populations: Vec<usize> = by_subject.values().map(Vec::len).collect();
    let take_per_subject = alloc.plan(&populations, limit)?;
    let alloc_note = alloc.describe(&take_per_subject);
    eprintln!("allocation: {alloc_note}");
    let whole: usize = take_per_subject
        .iter()
        .zip(&populations)
        .filter(|(&n, &pop)| n == pop && pop > 0)
        .count();
    if whole > 0 && alloc != Alloc::Flat {
        eprintln!(
            "  {whole} subject(s) taken whole: those strata are scored in parquet order \
             and carry no sampling error"
        );
    }

    // ---- score ----
    //
    // `LLVQ_MMLU_DUMP` writes one line per question — see [`dump_row`] for what
    // is on it and why. The header block carries what is known before the first
    // forward pass; the fingerprint is only known after the last one, so it
    // goes in the trailer.
    let mut dump = match std::env::var("LLVQ_MMLU_DUMP") {
        Ok(p) if !p.is_empty() => {
            let mut w = std::io::BufWriter::new(std::fs::File::create(&p)?);
            writeln!(w, "{DUMP_VERSION}")?;
            writeln!(w, "# model={label}")?;
            writeln!(w, "# dtype={}", llvq_llm::eval::dtype_name(dtype))?;
            writeln!(
                w,
                "# limit={}",
                if limit == usize::MAX {
                    "census".to_string()
                } else {
                    limit.to_string()
                }
            )?;
            writeln!(w, "# alloc={alloc_note}")?;
            // Which arithmetic scored the file, and every served choice — as
            // separate keys, never the free-text note (a newline in it would
            // split the header). `mmlupair` prints them under A = / B =; a
            // reader without this line cannot tell a kernel dump from a dense
            // one of the same file, and that pair is exactly the census.
            match &served {
                Some(cfg) => {
                    writeln!(w, "# config={}", cfg.path.display())?;
                    writeln!(w, "# arithmetic=served kernel")?;
                    writeln!(w, "# layout={}", cfg.layout.name())?;
                    writeln!(w, "# embed={}", cfg.embed.name())?;
                    writeln!(w, "# rot_share={}", cfg.rot_share.name())?;
                    writeln!(w, "# fuse={}", cfg.fuse.name())?;
                    writeln!(w, "# kv={}", cfg.kv.name())?;
                }
                None => {
                    writeln!(w, "# config=none")?;
                    writeln!(w, "# arithmetic=dense reconstruction")?;
                    writeln!(w, "# kv={}", kv_mode.name())?;
                }
            }
            writeln!(w, "{DUMP_COLUMNS}")?;
            eprintln!("dumping per-question results to {p}");
            Some(w)
        }
        _ => None,
    };
    // Every token actually put to the model, in order. Two arms that print the
    // same fingerprint were asked the same questions in the same words — the
    // one thing that made `bin/ppl` comparable and that this harness has so far
    // established by reading the code rather than by reading a result line.
    let mut scored_ids: Vec<u32> = Vec::new();
    let t0 = std::time::Instant::now();
    let mut total = 0usize;
    let mut per_subject: Vec<SubjectScore> = Vec::new();
    for ((subject, items), &take) in by_subject.iter().zip(&take_per_subject) {
        let prefix = {
            let mut s = format!(
                "The following are multiple choice questions (with answers) about {}.\n\n",
                pretty(subject)
            );
            for ex in shots.get(subject).map(|v| v.as_slice()).unwrap_or(&[]) {
                s.push_str(&block(ex, Some(ex.answer)));
            }
            s
        };
        // Seeded shuffle, then take: reproducible, and unbiased in a way
        // that `take(limit)` on an ordered corpus is not.
        let picked = select(items, subject, take);
        let (mut sr, mut st) = (0usize, 0usize);
        for (index, it) in picked.iter() {
            let prompt = format!("{prefix}{}", block(it, None));
            let ids = tok
                .encode(prompt.as_str(), false)
                .map_err(|e| anyhow::anyhow!("{e}"))?
                .get_ids()
                .to_vec();
            scored_ids.extend_from_slice(&ids);
            let input = Tensor::new(ids.as_slice(), &device)?.unsqueeze(0)?;
            // Last position, as f32 — the comparison is between four values
            // that can sit within an f16 ulp of each other.
            //
            // On the served arm the hidden states are narrowed to that
            // position BEFORE the head: `tv_q8_h` is one launch a row, each
            // streaming the 413 MB int8 table, and projecting 600 rows to read
            // one was ~15 % of the prompt's cost (*estimated* on the measured
            // 0.598 ms a row, phases-2026-08-07). The scored row is
            // bit-identical either way — the launches are independent. The
            // dense arm keeps `logits` whole: it is the call every published
            // bar was measured through, and it does not move.
            let last_row = match &served {
                Some(_) => {
                    let h = model.hidden(&input, &mut NoCapture)?;
                    let l = h.dim(1)?;
                    model.project_head(&h.narrow(1, l - 1, 1)?)?.i((0, 0))?
                }
                None => {
                    let logits = model.logits(&input, &mut NoCapture)?;
                    let last = logits.dim(1)? - 1;
                    logits.i((0, last))?
                }
            };
            let row: Vec<f32> = last_row.to_dtype(DType::F32)?.to_vec1()?;
            let options = [
                row[answer_ids[0] as usize],
                row[answer_ids[1] as usize],
                row[answer_ids[2] as usize],
                row[answer_ids[3] as usize],
            ];
            let pick = (0..4)
                .max_by(|&x, &y| options[x].total_cmp(&options[y]))
                .expect("four options");
            sr += usize::from(pick == it.answer);
            st += 1;
            if let Some(w) = dump.as_mut() {
                writeln!(
                    w,
                    "{}",
                    dump_row(
                        subject,
                        *index,
                        items.len(),
                        llvq_llm::eval::token_fingerprint(&ids),
                        it.answer,
                        pick,
                        options,
                    )
                )?;
            }
        }
        total += st;
        per_subject.push(SubjectScore {
            subject: subject.clone(),
            right: sr,
            scored: st,
            population: items.len(),
        });
        eprintln!(
            "  {:<40}{sr:>4}/{st:<4} {:>6.1} %   (micro {:>5.2} %, {:.0}s)",
            pretty(subject),
            100.0 * sr as f64 / st as f64,
            100.0 * micro(&per_subject),
            t0.elapsed().as_secs_f64()
        );
    }

    let fingerprint = llvq_llm::eval::token_fingerprint(&scored_ids);
    if let Some(w) = dump.as_mut() {
        writeln!(w, "{}", dump_trailer(fingerprint, total))?;
        w.flush()?;
    }

    let population: usize = per_subject.iter().map(|s| s.population).sum();
    let (mic, mac, se) = (
        micro(&per_subject),
        macro_avg(&per_subject),
        micro_stderr(&per_subject),
    );

    per_subject.sort_by(|a, b| b.rate().total_cmp(&a.rate()));
    println!("\n{label}");
    println!(
        "MMLU 5-shot — {total} questions scored out of {population}, {} subjects, dtype {}, tokens {fingerprint:016x}",
        per_subject.len(),
        llvq_llm::eval::dtype_name(dtype)
    );
    println!("  {}", "-".repeat(56));
    println!("  best:");
    for s in per_subject.iter().take(3) {
        println!("    {:<40}{:>6.1} %", pretty(&s.subject), 100.0 * s.rate());
    }
    println!("  worst:");
    for s in per_subject.iter().rev().take(3) {
        println!("    {:<40}{:>6.1} %", pretty(&s.subject), 100.0 * s.rate());
    }
    println!("  {}", "-".repeat(56));
    // The micro average is the reported figure — the one the paper's 70.2 and
    // 60.7 are. The macro is printed next to it because the gap between them
    // is a property of MMLU, not noise, and a reader who sees only one number
    // cannot tell which they are holding.
    println!(
        "  MMLU (micro, = paper) = {:.2} % ± {:.2}  [kv {}]",
        100.0 * mic,
        100.0 * se,
        kv_mode.name()
    );
    println!("  MMLU (macro, per subject) = {:.2} %", 100.0 * mac);
    // On the result line, not only in the label: an arm whose defining
    // parameter is not printed with its number is an arm nobody can re-read.
    if let Some(n) = &restore_note {
        println!("  restored (M2/M2b)        = {n}");
    }
    if total < population {
        println!(
            "  sample: {total}/{population} questions, {:.1} %, allocation {alloc_note}\n  \
             ± is the sampling error alone",
            100.0 * total as f64 / population as f64
        );
    }
    if dump.is_some() {
        println!(
            "\n  Dump written. Two arms are compared on the dumps, not on these two\n  \
             lines: `cargo run --release -p llvq-llm --bin mmlupair -- <a> <b>`."
        );
    }
    println!(
        "\n  Paper reference points (Qwen3-4B, Table 6): FP16 70.2 · LLVQ 60.7 · QTIP 57.4.\n  \
         If FP16 does not land near 70, the protocol is what needs fixing,\n  \
         not the model."
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two subjects, sizes 100 and 1,500, rates 25 % and 80 % — the real shape
    /// of MMLU's tail. Micro and macro must land 12 points apart, and the
    /// sampled estimate must recover the census one.
    fn unbalanced(scored: usize) -> Vec<SubjectScore> {
        vec![
            SubjectScore {
                subject: "abstract_algebra".into(),
                right: scored / 4,
                scored,
                population: 100,
            },
            SubjectScore {
                subject: "professional_law".into(),
                right: scored * 4 / 5,
                scored,
                population: 1_500,
            },
        ]
    }

    #[test]
    fn micro_and_macro_are_different_statistics() {
        let s = unbalanced(20);
        // macro = (0.25 + 0.80)/2 = 0.525
        assert!((macro_avg(&s) - 0.525).abs() < 1e-12);
        // micro = (100·0.25 + 1500·0.80)/1600 = 0.765625
        assert!((micro(&s) - 0.765_625).abs() < 1e-12);
        // 24 points apart. If this ever collapses, the harness has stopped
        // weighting and the reported score has silently changed meaning.
        assert!(micro(&s) - macro_avg(&s) > 0.2);
    }

    /// The property that matters: on a census, the weighted estimator *is*
    /// `Σright / Σscored`. A weighting bug that only shows up when sampling
    /// would otherwise hide behind full runs.
    #[test]
    fn micro_reduces_to_pooled_ratio_on_a_census() {
        let scores: Vec<SubjectScore> = [(100usize, 25usize), (1_500, 1_200), (783, 500)]
            .iter()
            .enumerate()
            .map(|(i, &(population, right))| SubjectScore {
                subject: format!("s{i}"),
                right,
                scored: population,
                population,
            })
            .collect();
        let pooled: f64 = scores.iter().map(|s| s.right).sum::<usize>() as f64
            / scores.iter().map(|s| s.scored).sum::<usize>() as f64;
        assert!((micro(&scores) - pooled).abs() < 1e-12);
        // And a census has no sampling error, by the finite population
        // correction — not by a special case.
        assert_eq!(micro_stderr(&scores), 0.0);
    }

    /// Sampling more of the same populations must shrink the bar.
    #[test]
    fn stderr_shrinks_with_the_sample() {
        let (few, many) = (micro_stderr(&unbalanced(20)), micro_stderr(&unbalanced(80)));
        assert!(few > many, "{few} should exceed {many}");
        assert!(many > 0.0, "80 scored out of 1,500 is still a sample");
    }

    fn corpus(n: usize) -> Vec<MmluItem> {
        (0..n)
            .map(|i| MmluItem {
                subject: "professional_law".into(),
                question: format!("q{i}"),
                choices: ["a".into(), "b".into(), "c".into(), "d".into()],
                answer: i % 4,
            })
            .collect()
    }

    /// Two arms must be asked the *same* questions, or the paired test they
    /// feed is meaningless. The sample depends only on the subject name's
    /// length and the limit — never on the model, the device or the dtype — so
    /// this holds by construction. This test is what keeps it that way.
    #[test]
    fn the_sample_is_identical_across_arms_and_moves_with_the_limit() {
        let items = corpus(500);
        let refs: Vec<&MmluItem> = items.iter().collect();

        let a = select(&refs, "professional_law", 40);
        let b = select(&refs, "professional_law", 40);
        let ka: Vec<usize> = a.iter().map(|(i, _)| *i).collect();
        let kb: Vec<usize> = b.iter().map(|(i, _)| *i).collect();
        assert_eq!(ka, kb, "two runs of one subject must draw the same questions");
        assert_eq!(ka.len(), 40);

        // The shuffle runs over the whole subject and *then* truncates, so the
        // samples are **nested**: a deeper run contains a shallower one as a
        // prefix. That is worth pinning — it means limit=40 and limit=100 can
        // be compared question by question, and that re-running deeper never
        // invalidates what was already scored.
        assert_eq!(
            ka,
            select(&refs, "professional_law", 100)
                .iter()
                .take(40)
                .map(|(i, _)| *i)
                .collect::<Vec<_>>(),
            "samples must nest — limit is a depth, not a different draw"
        );

        // And the sample must never be the head of the corpus: MMLU's test
        // split is not shuffled, so `take(limit)` would be a biased sample of
        // whatever the subject happens to open with.
        let census: Vec<usize> = select(&refs, "professional_law", usize::MAX)
            .iter()
            .map(|(i, _)| *i)
            .collect();
        assert_eq!(census, (0..500).collect::<Vec<_>>(), "a census keeps parquet order");
        assert_ne!(ka, census[..40].to_vec(), "the sample must not be the head");
    }

    /// The index must survive the shuffle. It is the only stable key to a
    /// question — `MmluItem` has no identifier — and the per-question dump is
    /// useless without it.
    #[test]
    fn the_parquet_index_survives_the_shuffle() {
        let items = corpus(200);
        let refs: Vec<&MmluItem> = items.iter().collect();
        for (index, it) in select(&refs, "abstract_algebra", 25) {
            assert_eq!(
                it.question,
                format!("q{index}"),
                "index {index} no longer points at its question"
            );
        }
    }

    /// The dump has to carry everything the paired analysis needs, and the
    /// analysis lives in another binary that cannot import this one. What
    /// stands between the two is this column line, so it is pinned here: drop
    /// `population` and the stratified micro is no longer reconstructible;
    /// drop `qhash` and two dumps can no longer be certified as the same
    /// questions. Either loss is silent at the CSV level and fatal at the
    /// statistics level.
    #[test]
    fn the_dump_carries_what_the_paired_analysis_needs() {
        for column in [
            "subject",
            "index",
            "population",
            "qhash",
            "answer",
            "pick",
            "correct",
        ] {
            assert!(
                DUMP_COLUMNS.split(',').any(|c| c == column),
                "the dump lost its {column} column"
            );
        }
        let row = dump_row("abstract_algebra", 17, 100, 0xdead_beef, 2, 2, [1.0, 2.0, 9.5, -0.5]);
        let fields: Vec<&str> = row.split(',').collect();
        assert_eq!(
            fields.len(),
            DUMP_COLUMNS.split(',').count(),
            "row and header disagree on arity: {row}"
        );
        // Position of each field, read through the header exactly as the
        // reader does it.
        let at = |name: &str| fields[DUMP_COLUMNS.split(',').position(|c| c == name).unwrap()];
        assert_eq!(at("subject"), "abstract_algebra");
        assert_eq!(at("index"), "17");
        assert_eq!(at("population"), "100");
        assert_eq!(at("qhash"), "00000000deadbeef");
        assert_eq!(at("correct"), "1", "pick 2 == answer 2");
        // The logits must round-trip: they exist to let a later analysis rank
        // confidence without a second forward pass, and a lossy print would
        // make that analysis quietly wrong rather than impossible.
        assert_eq!(at("logit_c").parse::<f32>().unwrap(), 9.5_f32);
        assert_eq!(at("logit_d").parse::<f32>().unwrap(), -0.5_f32);
    }

    /// A miss must be recorded as a miss. The `correct` column is derived, and
    /// a derived column that never disagrees with its inputs is a column that
    /// was never computed.
    #[test]
    fn a_wrong_pick_is_written_as_wrong() {
        let miss = dump_row("us_history", 3, 204, 1, 0, 3, [0.0; 4]);
        assert!(miss.ends_with("0,0,0,0,0"), "{miss}");
        assert!(miss.contains(",0,3,0,"), "answer 0, pick 3, correct 0: {miss}");
    }

    /// The trailer is what tells a reader the run finished. A census on a
    /// rented card runs for hours against a 30-minute platform default, so
    /// "the file exists and parses" is not evidence that all 57 subjects ran.
    #[test]
    fn the_trailer_carries_the_fingerprint_and_the_count() {
        let t = dump_trailer(0x65dc_d536_55e8_bfa5, 2_280);
        assert_eq!(t, "# end fingerprint=65dcd53655e8bfa5 questions=2280");
        assert!(t.starts_with('#'), "the trailer must not parse as a data row");
    }

    /// One stratum, so the weight is 1 whatever the population and the *only*
    /// thing that can move is the finite population correction. Drop the
    /// correction and the two bars become equal, which the first assertion
    /// rejects.
    #[test]
    fn the_finite_population_correction_is_load_bearing() {
        let one = |scored: usize, population: usize| {
            vec![SubjectScore {
                subject: "s".into(),
                right: scored / 2,
                scored,
                population,
            }]
        };
        assert_eq!(micro_stderr(&one(100, 100)), 0.0, "a census cannot have sampling error");
        assert!(micro_stderr(&one(100, 1_000)) > 0.0, "100 of 1,000 is a sample");
    }

    /// The control that is not negotiable: a run that asks for nothing draws
    /// exactly the sample every dump on disk was drawn with. `Flat` only ever
    /// hands `select` `min(limit, N)`, and `select` truncates on the same
    /// condition, so the two arguments are interchangeable at every depth.
    #[test]
    fn the_default_allocation_is_todays_sample() {
        assert_eq!(Alloc::parse("").unwrap(), Alloc::Flat);
        assert_eq!(Alloc::parse("flat").unwrap(), Alloc::Flat);
        assert_eq!(Alloc::Flat.plan(&[100, 1_534, 545], 40).unwrap(), vec![40, 40, 40]);
        assert_eq!(Alloc::Flat.plan(&[100, 1_534, 545], 200).unwrap(), vec![100, 200, 200]);
        assert_eq!(
            Alloc::Flat.plan(&[100, 1_534, 545], usize::MAX).unwrap(),
            vec![100, 1_534, 545]
        );

        let items = corpus(500);
        let refs: Vec<&MmluItem> = items.iter().collect();
        for limit in [1usize, 40, 499, 500, 900, usize::MAX] {
            let today: Vec<usize> = select(&refs, "professional_law", limit)
                .iter()
                .map(|(i, _)| *i)
                .collect();
            let take = Alloc::Flat.plan(&[refs.len()], limit).unwrap()[0];
            let now: Vec<usize> = select(&refs, "professional_law", take)
                .iter()
                .map(|(i, _)| *i)
                .collect();
            assert_eq!(today, now, "the flat plan moved the sample at limit={limit}");
        }
    }

    /// Inside one subject, the two plans nest: both draws come from the same
    /// shuffle of the stratum, so the shallower is a prefix of the deeper.
    ///
    /// That is a per-stratum property and nothing more. It does **not** make
    /// either sample a subset of the other over the 57 subjects, because the
    /// proportional plan goes deeper on some and shallower on others.
    /// `the_plan_on_the_real_populations` counts what that costs on the real
    /// populations.
    #[test]
    fn the_proportional_sample_is_a_per_stratum_prefix_of_the_flat_one() {
        let items = corpus(500);
        let refs: Vec<&MmluItem> = items.iter().collect();
        let pops = vec![100usize, 500, 1_534];
        let flat = Alloc::Flat.plan(&pops, 40).unwrap();
        let prop = Alloc::Proportional(None).plan(&pops, 40).unwrap();
        assert_eq!(flat.iter().sum::<usize>(), prop.iter().sum::<usize>(), "same budget");
        // The middle subject is the one this corpus can actually draw from.
        let (a, b) = (flat[1].min(prop[1]), flat[1].max(prop[1]));
        assert!(a < b, "the two plans must differ, or the test proves nothing");
        let deep: Vec<usize> = select(&refs, "professional_law", b)
            .iter()
            .map(|(i, _)| *i)
            .collect();
        let shallow: Vec<usize> = select(&refs, "professional_law", a)
            .iter()
            .map(|(i, _)| *i)
            .collect();
        assert_eq!(shallow, deep[..a].to_vec(), "the shallower draw must be a prefix");
    }

    /// The branch where `select` does not shuffle, pinned instead of avoided.
    /// A stratum allocated its whole population comes back in parquet order,
    /// which is a different *order* from the flat sample but a superset of its
    /// *content*, and the dump joins on `(subject, index)`.
    #[test]
    fn a_stratum_taken_whole_still_contains_the_flat_sample() {
        let items = corpus(60);
        let refs: Vec<&MmluItem> = items.iter().collect();
        let whole: Vec<usize> = select(&refs, "professional_law", 60)
            .iter()
            .map(|(i, _)| *i)
            .collect();
        assert_eq!(whole, (0..60).collect::<Vec<_>>(), "a whole stratum keeps parquet order");
        let flat: Vec<usize> = select(&refs, "professional_law", 40)
            .iter()
            .map(|(i, _)| *i)
            .collect();
        assert_ne!(flat, whole[..40].to_vec(), "the order differs, and that is the trap");
        assert!(
            flat.iter().all(|i| whole.contains(i)),
            "a whole stratum must contain every shallower sample"
        );
        // The plan is what keeps that the only case: it never asks a subject
        // for more than it holds.
        assert_eq!(
            Alloc::Proportional(Some(300)).plan(&[10, 1_000], usize::MAX).unwrap(),
            vec![3, 297],
            "the small stratum gets its share, not a quarter of the budget"
        );
        assert_eq!(
            Alloc::Proportional(Some(990)).plan(&[10, 1_000], usize::MAX).unwrap(),
            vec![10, 980],
            "a stratum is never asked for more than it holds"
        );
        assert_eq!(
            Alloc::Proportional(Some(5_000)).plan(&[60, 40], usize::MAX).unwrap(),
            vec![60, 40],
            "a budget above the population is a census"
        );
    }

    /// No subject falls to 0 or 1, at any budget the plan accepts, and a budget
    /// that cannot pay the floor is refused rather than quietly rounded away.
    #[test]
    fn the_floor_keeps_every_stratum_estimable() {
        let pops = vec![100usize, 1_534, 545, 100];
        for budget in [8usize, 9, 60, 137, 2_279] {
            let n = Alloc::Proportional(Some(budget)).plan(&pops, usize::MAX).unwrap();
            assert_eq!(n.iter().sum::<usize>(), budget, "budget {budget}: {n:?}");
            assert!(n.iter().all(|&x| x >= MIN_PER_SUBJECT), "budget {budget}: {n:?}");
            assert!(
                n.iter().zip(&pops).all(|(&x, &pop)| x <= pop),
                "budget {budget}: {n:?}"
            );
        }
        assert!(
            Alloc::Proportional(Some(7)).plan(&pops, usize::MAX).is_err(),
            "7 questions cannot give 2 to each of 4 subjects"
        );
    }

    /// The plan is proportional where it is free to be: the big stratum gets
    /// its share of the budget, not its share of the subjects.
    #[test]
    fn the_budget_follows_the_population() {
        let pops = vec![100usize, 1_534, 545];
        let n = Alloc::Proportional(None).plan(&pops, 40).unwrap();
        assert_eq!(n.iter().sum::<usize>(), 120);
        // 120 · 1534/2179 = 84.5, 120 · 100/2179 = 5.5, 120 · 545/2179 = 30.0
        assert_eq!(n, vec![6, 84, 30]);
        assert!(n[1] > n[0] * 10, "the weighted stratum must get the budget");
    }

    /// `micro_stderr` must stay correct once the allocation stops being equal.
    /// The value is hand-computed from the formula, term by term, so a change
    /// that quietly assumes a common `n` fails here.
    #[test]
    fn the_stderr_formula_holds_under_an_unequal_allocation() {
        let scores = vec![
            SubjectScore { subject: "small".into(), right: 5, scored: 10, population: 100 },
            SubjectScore { subject: "big".into(), right: 40, scored: 50, population: 1_500 },
        ];
        let (w1, w2) = (100.0 / 1600.0_f64, 1_500.0 / 1600.0_f64);
        let expect = (w1 * w1 * (0.25 / 9.0) * (1.0 - 10.0 / 100.0)
            + w2 * w2 * (0.16 / 49.0) * (1.0 - 50.0 / 1_500.0))
            .sqrt();
        assert!(
            (micro_stderr(&scores) - expect).abs() < 1e-15,
            "{} against {expect}",
            micro_stderr(&scores)
        );
    }

    /// The point of the whole thing: at one budget, the proportional plan
    /// carries a smaller bar than the flat one.
    #[test]
    fn proportional_shrinks_the_bar_at_constant_budget() {
        let pops = vec![100usize, 1_500];
        let rates = [0.25_f64, 0.80];
        let bar = |n: &[usize]| {
            let s: Vec<SubjectScore> = n
                .iter()
                .zip(&pops)
                .zip(&rates)
                .enumerate()
                .map(|(i, ((&scored, &population), &rate))| SubjectScore {
                    subject: format!("s{i}"),
                    right: (scored as f64 * rate).round() as usize,
                    scored,
                    population,
                })
                .collect();
            micro_stderr(&s)
        };
        let flat = Alloc::Flat.plan(&pops, 40).unwrap();
        let prop = Alloc::Proportional(None).plan(&pops, 40).unwrap();
        assert_eq!(flat.iter().sum::<usize>(), prop.iter().sum::<usize>());
        assert_eq!(prop, vec![5, 75]);
        assert!(bar(&prop) < bar(&flat), "{} against {}", bar(&prop), bar(&flat));
    }

    /// An unknown plan is refused, and a proportional census with no budget is
    /// refused too: there is nothing to spread.
    #[test]
    fn an_unknown_allocation_is_refused() {
        assert_eq!(Alloc::parse("proportional").unwrap(), Alloc::Proportional(None));
        assert_eq!(
            Alloc::parse("proportional=1200").unwrap(),
            Alloc::Proportional(Some(1_200))
        );
        for bad in ["neyman", "prop", "proportional=0", "proportional=x", "PROPORTIONAL", "1"] {
            assert!(Alloc::parse(bad).is_err(), "{bad:?} must be refused");
        }
        assert!(Alloc::Proportional(None).plan(&[100, 200], usize::MAX).is_err());
    }

    /// The plan is printed, so it is part of the record and not of the folklore.
    #[test]
    fn the_plan_is_printed_with_its_numbers() {
        assert_eq!(
            Alloc::Flat.describe(&[40, 40, 40]),
            "flat, 40 per subject, 120 questions"
        );
        assert_eq!(
            Alloc::Proportional(None).describe(&[6, 84, 30]),
            "proportional, 6..84 per subject, 120 questions"
        );
    }

    /// The 57 populations of MMLU's `test` split, in the subject order the
    /// harness walks (alphabetical, from a `BTreeMap`). Read off
    /// `docs/data/mmlu-dumps/mmlu-4b-llvq.csv`, which carries each subject's
    /// population on every row.
    const MMLU_TEST: [usize; 57] = [
        100, 135, 152, 100, 265, 144, 100, 100, 100, 173, 102, 100, 235, 114, 145, 378, 126,
        100, 310, 203, 100, 165, 198, 193, 390, 270, 238, 151, 545, 216, 204, 237, 223, 131,
        121, 108, 163, 112, 103, 234, 100, 783, 346, 895, 306, 311, 324, 282, 1534, 272, 612,
        110, 245, 201, 100, 166, 171,
    ];

    /// The plan behind the headline number, pinned on the real populations.
    ///
    /// At the budget of the published run (2,280 questions, 40 per subject),
    /// the proportional plan runs from 16 to 249 questions and no stratum is
    /// taken whole, so `select` shuffles everywhere and each subject's draw
    /// nests with the dumps on disk. Subject by subject, and no further: the
    /// test counts the 630 questions the plan drops and the 630 it adds. The
    /// accuracy bar it carries is 0.925 pp against 1.355 pp flat, the factor
    /// the module note gives.
    #[test]
    fn the_plan_on_the_real_populations() {
        assert_eq!(MMLU_TEST.iter().sum::<usize>(), 14_042);
        let flat = Alloc::Flat.plan(&MMLU_TEST, 40).unwrap();
        let prop = Alloc::Proportional(None).plan(&MMLU_TEST, 40).unwrap();
        assert_eq!(flat.iter().sum::<usize>(), 2_280);
        assert_eq!(prop.iter().sum::<usize>(), 2_280, "the budget is held constant");
        assert_eq!(prop.iter().copied().min(), Some(16), "abstract_algebra, N = 100");
        assert_eq!(prop.iter().copied().max(), Some(249), "professional_law, N = 1534");
        assert_eq!(prop[41], 127, "N = 783");
        assert!(
            prop.iter().zip(&MMLU_TEST).all(|(&n, &pop)| n < pop),
            "no stratum is taken whole at this budget, so every subject nests"
        );
        // Nesting is per stratum, and the two samples are not nested as sets:
        // 40 subjects of 57 go below 40 questions, which drops 630 of the 2,280
        // questions the dumps on disk carry and puts 630 new ones in their
        // place. A dump drawn under this plan is a new exam, not a re-reading.
        let shallower = flat.iter().zip(&prop).filter(|(f, p)| p < f).count();
        let dropped: usize =
            flat.iter().zip(&prop).map(|(&f, &p)| f.saturating_sub(p)).sum();
        let added: usize = flat.iter().zip(&prop).map(|(&f, &p)| p.saturating_sub(f)).sum();
        assert_eq!((shallower, dropped, added), (40, 630, 630));
        // Halving the budget is still legal, still floored, still capped.
        let half = Alloc::Proportional(Some(1_191)).plan(&MMLU_TEST, usize::MAX).unwrap();
        assert_eq!(half.iter().sum::<usize>(), 1_191);
        assert!(half.iter().all(|&n| n >= MIN_PER_SUBJECT));
    }

    /// Ties are broken toward the lowest subject index, in both directions.
    /// MMLU has fourteen subjects of exactly 100 questions, so ties are the
    /// normal case, not the corner one: a plan that resolved them by hash order
    /// would draw a different sample on every run and no dump would join.
    #[test]
    fn ties_are_broken_deterministically() {
        // The floors overshoot by one, and the two equal strata are equally far
        // from their exact share, so the trim loop must choose between them.
        assert_eq!(
            Alloc::Proportional(Some(7)).plan(&[100, 100, 2], usize::MAX).unwrap(),
            vec![2, 3, 2]
        );
        // Same on the way up: one question left to place, two equal claims.
        assert_eq!(
            Alloc::Proportional(Some(606)).plan(&[100, 100, 1_000], usize::MAX).unwrap(),
            vec![51, 50, 505]
        );
    }
}
