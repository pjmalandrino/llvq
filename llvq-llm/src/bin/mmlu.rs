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
//!   again.
//!
//! **Run the FP16 baseline first.** The paper reports 70.2 on Qwen3-4B; if
//! this harness does not land there, the protocol is wrong and no other number
//! it produces means anything. That is the same discipline as the identity
//! control of Phase 5: the test you re-run first when a result looks odd.

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen3::Config;
use llvq_llm::corpus::{mmlu_split, MmluItem};
use llvq_llm::model::{NoCapture, Qwen3};
use std::collections::{BTreeMap, HashMap};
use std::io::Read;

fn read_u32(r: &mut impl Read) -> anyhow::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

/// `underscored_subject` → `underscored subject`, as the standard prompt wants.
fn pretty(subject: &str) -> String {
    subject.replace('_', " ")
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

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let model_arg = a
        .first()
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("give a sealed .llvq path or a HF repo id"))?;
    let device = if a.get(1).map(|s| s == "metal").unwrap_or(false) {
        Device::new_metal(0).unwrap_or(Device::Cpu)
    } else {
        Device::Cpu
    };
    // Questions per subject, for a cheap protocol check before the full run.
    // Sampled at random from a fixed seed, never the first N: MMLU test sets
    // are not shuffled, and the head of a subject is not a fair sample of it.
    let limit: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(usize::MAX);
    let dtype = DType::F16;

    // ---- the model: the shipped artifact, or the reference checkpoint ----
    let sealed = model_arg.ends_with(".llvq") || model_arg.ends_with(".bin");
    let (model, tok, label) = if sealed {
        let f = std::fs::File::open(&model_arg)?;
        let mut r = std::io::BufReader::with_capacity(1 << 20, f);
        let head = llvq_artifact::read_header(&mut r)?;
        anyhow::ensure!(
            head.is_self_contained(),
            "{model_arg} is projections-only (format v{}); seal it first",
            head.version
        );
        let ix = llvq_search::index::Indexer::new();
        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        for _ in 0..head.matrices {
            let m = llvq_artifact::read_matrix(&mut r, &ix)?;
            let w = llvq_artifact::decode_matrix(&m);
            let t = Tensor::from_vec(w, (m.d_out, m.d_in), &device)?.to_dtype(dtype)?;
            tensors.insert(m.name, t);
        }
        let n_raw = read_u32(&mut r)?;
        for _ in 0..n_raw {
            let t = llvq_artifact::read_raw(&mut r)?;
            let vals: Vec<half::f16> =
                t.data.iter().map(|b| half::f16::from_bits(*b)).collect();
            let tensor = Tensor::from_vec(vals, t.dims.clone(), &device)?.to_dtype(dtype)?;
            tensors.insert(t.name, tensor);
        }
        let n_blob = read_u32(&mut r)?;
        let mut blobs: HashMap<String, Vec<u8>> = HashMap::new();
        for _ in 0..n_blob {
            let b = llvq_artifact::read_blob(&mut r)?;
            blobs.insert(b.name, b.bytes);
        }
        let config: Config = serde_json::from_slice(
            blobs
                .get("config.json")
                .ok_or_else(|| anyhow::anyhow!("no config.json in {model_arg}"))?,
        )?;
        let tok = tokenizers::Tokenizer::from_bytes(
            blobs
                .get("tokenizer.json")
                .ok_or_else(|| anyhow::anyhow!("no tokenizer.json in {model_arg}"))?,
        )
        .map_err(|e| anyhow::anyhow!("{e}"))?;
        let vb = VarBuilder::from_tensors(tensors, dtype, &device);
        (
            Qwen3::new(&config, vb)?,
            tok,
            format!("{model_arg} [LLVQ 2-bit, sealed]"),
        )
    } else {
        let ck = llvq_llm::loader::Checkpoint::fetch(&model_arg)?;
        let tok = ck.tokenizer()?;
        let vb = ck.var_builder(dtype, &device)?;
        (
            Qwen3::new(&ck.config, vb)?,
            tok,
            format!("{model_arg} [FP16 reference]"),
        )
    };
    eprintln!("model: {label}\ndevice: {device:?}");

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
        let mut picked: Vec<&MmluItem> = items.clone();
        if limit < picked.len() {
            let mut rng = llvq_core::SplitMix64::new(0x6_11B0 ^ subject.len() as u64);
            for i in (1..picked.len()).rev() {
                picked.swap(i, (rng.next() % (i as u64 + 1)) as usize);
            }
            picked.truncate(limit);
        }
        let (mut sr, mut st) = (0usize, 0usize);
        for it in picked.iter() {
            let prompt = format!("{prefix}{}", block(it, None));
            let ids = tok
                .encode(prompt.as_str(), false)
                .map_err(|e| anyhow::anyhow!("{e}"))?
                .get_ids()
                .to_vec();
            let input = Tensor::new(ids.as_slice(), &device)?.unsqueeze(0)?;
            let logits = model.logits(&input, &mut NoCapture)?;
            // Last position, as f32 — the comparison is between four values
            // that can sit within an f16 ulp of each other.
            let last = logits.dim(1)? - 1;
            let row: Vec<f32> = logits
                .i((0, last))?
                .to_dtype(DType::F32)?
                .to_vec1()?;
            let pick = (0..4)
                .max_by(|&x, &y| {
                    row[answer_ids[x] as usize]
                        .total_cmp(&row[answer_ids[y] as usize])
                })
                .expect("four options");
            sr += usize::from(pick == it.answer);
            st += 1;
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

    let population: usize = per_subject.iter().map(|s| s.population).sum();
    let (mic, mac, se) = (
        micro(&per_subject),
        macro_avg(&per_subject),
        micro_stderr(&per_subject),
    );

    per_subject.sort_by(|a, b| b.rate().total_cmp(&a.rate()));
    println!("\n{label}");
    println!(
        "MMLU 5-shot — {total} questions scorées sur {population}, {} matières",
        per_subject.len()
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
