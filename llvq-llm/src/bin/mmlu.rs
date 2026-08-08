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
/// `professional_law` holds 1 534 questions, `abstract_algebra` 100, a ratio
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
///   share 2 280 questions but necessarily print different run fingerprints,
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

    // ---- the model: the shipped artifact, or the reference checkpoint ----
    //
    // The sealed path is shared with `bin/run` and `bin/ppl` — see
    // `llvq_llm::sealed` for why having three copies of it was a problem and
    // not just duplication.
    let (model, tok, label) = if llvq_llm::sealed::is_sealed_path(&model_arg) {
        let s = llvq_llm::sealed::load(&model_arg, dtype, &device)?;
        (
            s.model,
            s.tokenizer,
            format!("{model_arg} [LLVQ 2-bit, sealed]"),
        )
    } else {
        let ck = llvq_llm::loader::Checkpoint::fetch(&model_arg)?;
        let tok = ck.tokenizer()?;
        let vb = ck.var_builder(dtype, &device)?;
        (
            llvq_llm::model::Qwen3::new(&ck.config, vb)?,
            tok,
            format!("{model_arg} [reference checkpoint]"),
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
    for (subject, items) in &by_subject {
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
        let picked = select(items, subject, limit);
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
            let logits = model.logits(&input, &mut NoCapture)?;
            // Last position, as f32 — the comparison is between four values
            // that can sit within an f16 ulp of each other.
            let last = logits.dim(1)? - 1;
            let row: Vec<f32> = logits
                .i((0, last))?
                .to_dtype(DType::F32)?
                .to_vec1()?;
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
        "MMLU 5-shot — {total} questions scorées sur {population}, {} matières, dtype {}, tokens {fingerprint:016x}",
        per_subject.len(),
        llvq_llm::eval::dtype_name(dtype)
    );
    println!("  {}", "-".repeat(56));
    println!("  meilleures :");
    for s in per_subject.iter().take(3) {
        println!("    {:<40}{:>6.1} %", pretty(&s.subject), 100.0 * s.rate());
    }
    println!("  pires :");
    for s in per_subject.iter().rev().take(3) {
        println!("    {:<40}{:>6.1} %", pretty(&s.subject), 100.0 * s.rate());
    }
    println!("  {}", "-".repeat(56));
    // The micro average is the reported figure — the one the paper's 70.2 and
    // 60.7 are. The macro is printed next to it because the gap between them
    // is a property of MMLU, not noise, and a reader who sees only one number
    // cannot tell which they are holding.
    println!("  MMLU (micro, = papier) = {:.2} % ± {:.2}", 100.0 * mic, 100.0 * se);
    println!("  MMLU (macro, par matière) = {:.2} %", 100.0 * mac);
    if total < population {
        println!(
            "  échantillon : {total}/{population} questions, {:.1} % — \
             ± est l'erreur d'échantillonnage seule",
            100.0 * total as f64 / population as f64
        );
    }
    if dump.is_some() {
        println!(
            "\n  Dump écrit. La comparaison de deux bras se fait sur les dumps, pas sur\n  \
             ces deux lignes : `cargo run --release -p llvq-llm --bin mmlupair -- <a> <b>`."
        );
    }
    println!(
        "\n  Repères papier (Qwen3-4B, Table 6) : FP16 70,2 · LLVQ 60,7 · QTIP 57,4.\n  \
         Si le FP16 ne tombe pas vers 70, c'est le protocole qu'il faut corriger,\n  \
         pas le modèle."
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two subjects, sizes 100 and 1 500, rates 25 % and 80 % — the real shape
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
        assert!(many > 0.0, "80 scored out of 1 500 is still a sample");
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
        assert!(micro_stderr(&one(100, 1_000)) > 0.0, "100 of 1 000 is a sample");
    }
}
