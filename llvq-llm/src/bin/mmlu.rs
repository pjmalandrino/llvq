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
//!   position — one forward pass per question, not four.
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
    let (mut right, mut total) = (0usize, 0usize);
    let mut per_subject: Vec<(String, usize, usize)> = Vec::new();
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
        right += sr;
        total += st;
        per_subject.push((subject.clone(), sr, st));
        eprintln!(
            "  {:<40}{sr:>4}/{st:<4} {:>6.1} %   (global {:>5.2} %, {:.0}s)",
            pretty(subject),
            100.0 * sr as f64 / st as f64,
            100.0 * right as f64 / total as f64,
            t0.elapsed().as_secs_f64()
        );
    }

    per_subject.sort_by(|a, b| {
        (b.1 as f64 / b.2 as f64).total_cmp(&(a.1 as f64 / a.2 as f64))
    });
    println!("\n{label}");
    println!("MMLU 5-shot — {total} questions, {} subjects", per_subject.len());
    println!("  {}", "-".repeat(56));
    println!("  meilleures :");
    for (s, r, t) in per_subject.iter().take(3) {
        println!("    {:<40}{:>6.1} %", pretty(s), 100.0 * *r as f64 / *t as f64);
    }
    println!("  pires :");
    for (s, r, t) in per_subject.iter().rev().take(3) {
        println!("    {:<40}{:>6.1} %", pretty(s), 100.0 * *r as f64 / *t as f64);
    }
    println!("  {}", "-".repeat(56));
    println!(
        "  MMLU = {:.2} %   ({right}/{total})",
        100.0 * right as f64 / total as f64
    );
    println!(
        "\n  Repères papier (Qwen3-4B, Table 6) : FP16 70,2 · LLVQ 60,7 · QTIP 57,4.\n  \
         Si le FP16 ne tombe pas vers 70, c'est le protocole qu'il faut corriger,\n  \
         pas le modèle."
    );
    Ok(())
}
