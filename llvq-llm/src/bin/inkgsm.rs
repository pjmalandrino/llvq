//! `inkgsm`: GSM8K for Inkling 2 Mini through this engine, with its experts resident in lattice codes on Metal or
//! CUDA, loaded as `inkrun` loads them. One JSON line a problem, in the dump format `gsm8kpair` reads.
//!
//! ```text
//! inkgsm run --config D/textonly/config.json --checkpoint D/textonly/model.safetensors --layers D/layers \
//!            --tokenizer D/native/tokenizer.json --device metal|cuda --dump out.jsonl \
//!            [--problems ops/data/spec-gsm8k-200.jsonl] [--limit 200] [--max-new 1536] [--effort none] \
//!            [--resident verified|none] [--require-layers 24] [--deadline-seconds S] [--parquet <test.parquet>]
//!            [--tables <dir of embed.q4g64.llvqraw and unembed.q4g64.llvqraw>]
//! inkgsm prompts --tokenizer <tokenizer.json> --out ids.json [--problems ..] [--limit 3] [--effort none] \
//!                [--parquet <test.parquet>]
//! inkgsm score <a.jsonl> [<b.jsonl> ...]
//! ```
//!
//! ## The protocol, and what each default is
//!
//! - The problems are the first `--limit` (200) rows of `--problems`, which must be GSM8K test rows `0..limit` in
//!   parquet order. `ops/data/spec-gsm8k-200.jsonl` holds rows 0 to 199 of `openai/gsm8k`, `main/test` at revision
//!   `740312ad`, each `gsm8k::user_body(question)` with its gold (`gsm8k::gold`). `--parquet` checks that claim row
//!   by row against the parquet file before any token. This is not the set `bin/gsm8k <model> <dev> 200` scores:
//!   that one is a seeded sample of the 1,319 (`gsm8k::select`), so a Qwen3 dump does not pair with these.
//! - The prompt is one user message, the row's text, rendered by `llvq_llm::inkchat` (the checkpoint's template)
//!   with reasoning effort `--effort` (`none`). The template has no generation prompt: the model writes its own
//!   `<|message_model|>`.
//! - Greedy, the first maximum of the f32 logits. Generation stops on `InkChat::stop_ids`
//!   (`<|content_model_end_sampling|>`, the eos of `generation_config.json`), which is not counted, or after
//!   `--max-new` (1,536) generated tokens.
//! - The completion is the generated ids decoded with their markers. It is graded by `llvq_llm::gsm8k` unchanged
//!   (`extract`, `is_correct`), and a completion cut by the cap counts wrong: the row's `correct` is false when
//!   `stop` is `cap`. `gsm8kpair` re-grades every row by the module's rules and does not know the cap rule, so it
//!   can count a capped row right; `inkgsm score` applies the same re-grade, then the cap rule.
//!
//! ## The dump, and resuming
//!
//! A header line (`gsm8k::DUMP_TAG`, the protocol, the sha256 of the problems, tokenizer and config files, the
//! resident layers), one line a problem (the `gsm8k::Row` fields plus `turn`, whether the template's turn parser
//! accepts the generated ids, and `step_s_median`), and a trailer once every problem is in. Each line is flushed as
//! it is written. A rerun with the same `--dump` reads it back, refuses a header that differs from this run's in any
//! field, checks every row's prompt hash, drops a last line cut by a kill, and generates only the problems that are
//! not in it. A finished dump is summarized without loading the model.
//!
//! `--deadline-seconds` (from the process start): the problem under way when it passes is dropped, not written, and
//! the run exits with code 3 and no trailer, so a job stops before its timeout with every written line complete.

use std::collections::BTreeSet;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{anyhow, bail, ensure, Context, Result};
use candle_core::Device;
use serde_json::{json, Value};

use llvq_llm::gsm8k::{self, Row};
use llvq_llm::inkchat::{InkChat, Json, TurnParser};
use llvq_llm::inkling::Inkling;
use llvq_llm::lm::CausalLm;

const DEFAULT_PROBLEMS: &str = "ops/data/spec-gsm8k-200.jsonl";
const DEFAULT_LIMIT: usize = 200;
const DEFAULT_MAX_NEW: usize = 1536;
const DEFAULT_EFFORT: &str = "none";
/// The commit of `openai/gsm8k` the problems file was cut from (its own `_comment` line says so).
const GSM8K_REVISION: &str = "740312add88f781978c0658806c59bc2815b9866";
/// The exit code of a run stopped by `--deadline-seconds`.
const EXIT_DEADLINE: i32 = 3;

// ---------------------------------------------------------------------------------------------------------------
// Arguments
// ---------------------------------------------------------------------------------------------------------------

fn usage() -> ! {
    eprintln!(
        "usage:\n  inkgsm run --config <config.json> --checkpoint <model.safetensors> --layers <D/layers> \
         --tokenizer <tokenizer.json> --device metal|cuda|cpu --dump <out.jsonl> [--problems {DEFAULT_PROBLEMS}] \
         [--limit {DEFAULT_LIMIT}] [--max-new {DEFAULT_MAX_NEW}] [--effort {DEFAULT_EFFORT}] \
         [--resident verified|none] [--require-layers N] [--deadline-seconds S] [--parquet <test.parquet>]\n  \
         inkgsm prompts --tokenizer <tokenizer.json> --out <ids.json> [--problems ..] [--limit 3] [--effort none]\n  \
         inkgsm score <a.jsonl> [<b.jsonl> ...]"
    );
    std::process::exit(2)
}

/// `--key value` pairs, each key at most once.
fn flags(args: &[String]) -> Result<std::collections::HashMap<String, String>> {
    let mut a = std::collections::HashMap::new();
    let mut it = args.iter();
    while let Some(k) = it.next() {
        let Some(name) = k.strip_prefix("--") else { bail!("unexpected argument {k:?}") };
        let v = it.next().ok_or_else(|| anyhow!("--{name} needs a value"))?;
        ensure!(a.insert(name.to_string(), v.clone()).is_none(), "--{name} given twice");
    }
    Ok(a)
}

fn count(a: &mut std::collections::HashMap<String, String>, key: &str, default: usize) -> Result<usize> {
    match a.remove(key) {
        None => Ok(default),
        Some(v) => v.trim().parse().map_err(|_| anyhow!("--{key} {v:?} is not a count")),
    }
}

fn no_more(a: &std::collections::HashMap<String, String>) -> Result<()> {
    if let Some(k) = a.keys().next() {
        bail!("unknown option --{k}");
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------------------------
// Problems
// ---------------------------------------------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
struct Problem {
    /// Row of the test split, parquet order.
    index: usize,
    /// The user turn: `gsm8k::user_body(question)`.
    prompt: String,
    /// Canonical, as `gsm8k::gold` writes it.
    gold: String,
}

/// The first `limit` problems of a problems file (`ops/data/spec-gsm8k-200.jsonl` format), refused unless they are
/// test rows `0..limit` in order, each a question followed by the house instruction, with a canonical gold.
fn read_problems(text: &str, limit: usize) -> Result<Vec<Problem>> {
    ensure!(limit > 0, "--limit 0 scores nothing");
    let suffix = format!("\n{}", gsm8k::INSTRUCTION);
    let mut out = Vec::with_capacity(limit);
    for (n, line) in text.lines().enumerate() {
        if out.len() == limit {
            break;
        }
        if line.trim().is_empty() {
            continue;
        }
        let v: Value = serde_json::from_str(line).with_context(|| format!("problems line {}", n + 1))?;
        if v.get("_comment").is_some() {
            continue;
        }
        let field = |k: &str| v.get(k).and_then(Value::as_str).ok_or_else(|| anyhow!("problems line {}: no {k}", n + 1));
        let id = field("id")?;
        let index: usize = id
            .strip_prefix("gsm8k-test-")
            .and_then(|s| s.parse().ok())
            .ok_or_else(|| anyhow!("problems line {}: id {id:?} is not gsm8k-test-<row>", n + 1))?;
        ensure!(field("kind")? == "math", "problems line {}: kind is not math", n + 1);
        let prompt = field("prompt")?.to_string();
        let question = prompt
            .strip_suffix(&suffix)
            .ok_or_else(|| anyhow!("{id}: the prompt does not end with a newline and gsm8k::INSTRUCTION"))?;
        ensure!(gsm8k::user_body(question) == prompt, "{id}: the prompt is not gsm8k::user_body of its question");
        let gold = field("gold")?.to_string();
        ensure!(gsm8k::canonical(&gold).as_deref() == Some(gold.as_str()), "{id}: gold {gold:?} is not canonical");
        out.push(Problem { index, prompt, gold });
    }
    ensure!(out.len() == limit, "the problems file holds {} problems, {limit} asked", out.len());
    for (k, p) in out.iter().enumerate() {
        ensure!(
            p.index == k,
            "row {k} of the problems file is test row {}: the protocol is the first {limit} rows in parquet order",
            p.index
        );
    }
    Ok(out)
}

/// `(question, answer)` of every row of a GSM8K parquet split, in file order, columns read by name as
/// `llvq_llm::corpus::gsm8k_split` reads them.
fn read_parquet(path: &Path) -> Result<Vec<(String, String)>> {
    use parquet::file::reader::{FileReader, SerializedFileReader};
    use parquet::record::Field;
    let reader = SerializedFileReader::new(std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?)?;
    let mut out = Vec::new();
    for (i, row) in reader.get_row_iter(None)?.enumerate() {
        let row = row?;
        let (mut q, mut a) = (None, None);
        for (name, f) in row.get_column_iter() {
            match (name.as_str(), f) {
                ("question", Field::Str(s)) => q = Some(s.clone()),
                ("answer", Field::Str(s)) => a = Some(s.clone()),
                _ => {}
            }
        }
        out.push((q.ok_or_else(|| anyhow!("row {i}: no question"))?, a.ok_or_else(|| anyhow!("row {i}: no answer"))?));
    }
    Ok(out)
}

/// The problems against the parquet rows they claim to be, as `bin/gsm8k` would build them.
fn check_against_parquet(problems: &[Problem], rows: &[(String, String)]) -> Result<()> {
    for p in problems {
        let (q, a) = rows.get(p.index).ok_or_else(|| anyhow!("the parquet has no row {}", p.index))?;
        ensure!(gsm8k::user_body(q) == p.prompt, "row {}: the problems file's prompt is not the parquet's", p.index);
        let g = gsm8k::gold(a).ok_or_else(|| anyhow!("row {}: no gold in the parquet answer", p.index))?;
        ensure!(g == p.gold, "row {}: gold {} in the problems file, {g} in the parquet", p.index, p.gold);
    }
    Ok(())
}

/// The prompt ids of one problem: one user message, the reasoning effort, the checkpoint's template.
fn render(chat: &InkChat, prompt: &str, effort: &Json) -> Result<Vec<u32>> {
    let msg = Json::parse(&json!({"role": "user", "content": prompt}).to_string())?;
    Ok(chat.render(&[msg], &[], Some(effort))?.ids)
}

fn effort_json(effort: &str) -> Result<Json> {
    let j = match effort.parse::<f64>() {
        Ok(_) => Json::parse(effort)?,
        Err(_) => Json::str(effort),
    };
    llvq_llm::inkchat::effort_text(Some(&j)).with_context(|| format!("--effort {effort}"))?;
    Ok(j)
}

// ---------------------------------------------------------------------------------------------------------------
// Greedy decode
// ---------------------------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stop {
    Eos,
    Cap,
}

impl Stop {
    fn name(self) -> &'static str {
        match self {
            Stop::Eos => "eos",
            Stop::Cap => "cap",
        }
    }
}

/// The first maximum, as `inkrun` takes it. A vector without a finite maximum is refused.
fn argmax(v: &[f32]) -> Result<u32> {
    let (mut i1, mut b1) = (0usize, f32::NEG_INFINITY);
    for (i, &x) in v.iter().enumerate() {
        if x > b1 {
            b1 = x;
            i1 = i;
        }
    }
    ensure!(b1.is_finite(), "logits without a finite maximum");
    Ok(i1 as u32)
}

/// Greedy decode from the prompt's logits. `step(token, k)` feeds the `k`-th generated token (from 0) and returns
/// the logits that follow it. Stops before a stop id, which is not kept, or once `max_new` tokens are kept.
fn greedy<F>(first: &[f32], stops: &[u32], max_new: usize, mut step: F) -> Result<(Vec<u32>, Stop)>
where
    F: FnMut(u32, usize) -> Result<Vec<f32>>,
{
    let mut said = Vec::new();
    let mut next = argmax(first)?;
    loop {
        if stops.contains(&next) {
            return Ok((said, Stop::Eos));
        }
        said.push(next);
        if said.len() >= max_new {
            return Ok((said, Stop::Cap));
        }
        let logits = step(next, said.len() - 1)?;
        next = argmax(&logits)?;
    }
}

/// Whether a row counts: graded by `llvq_llm::gsm8k`, and wrong when the cap cut it.
fn strict_correct(completion: &str, gold: &str, stop: &str) -> (gsm8k::Source, Option<String>, bool) {
    let (source, extracted) = gsm8k::extract(completion);
    let correct = stop == "eos" && gsm8k::is_correct(extracted.as_deref(), gold);
    (source, extracted, correct)
}

#[derive(Debug)]
struct DeadlineHit;

impl std::fmt::Display for DeadlineHit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "--deadline-seconds passed")
    }
}

impl std::error::Error for DeadlineHit {}

// ---------------------------------------------------------------------------------------------------------------
// The dump
// ---------------------------------------------------------------------------------------------------------------

/// The text up to and including its last newline: a line cut by a kill is dropped.
fn complete_lines(text: &str) -> &str {
    match text.rfind('\n') {
        Some(i) => &text[..=i],
        None => "",
    }
}

/// An existing dump read back against this run: its rows, and whether it carries its trailer. `expected` is the
/// run's problems in order, `(index, qhash)`. Refuses another header, a row out of order or with another prompt
/// hash, and anything after the trailer.
fn read_existing(text: &str, header: &Value, expected: &[(usize, String)]) -> Result<(Vec<Row>, bool)> {
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let Some(first) = lines.next() else { return Ok((vec![], false)) };
    let old: Value = serde_json::from_str(first).context("dump header")?;
    if &old != header {
        let (o, n) = (old.as_object(), header.as_object());
        let mut keys: BTreeSet<&String> = BTreeSet::new();
        if let (Some(o), Some(n)) = (o, n) {
            keys.extend(o.keys().chain(n.keys()).filter(|k| o.get(*k) != n.get(*k)));
        }
        bail!("the dump's header differs from this run's in {keys:?}: another protocol, refused; use a new --dump");
    }
    let mut rows: Vec<Row> = Vec::new();
    let mut finished = false;
    for line in lines {
        ensure!(!finished, "the dump holds a line after its trailer");
        let v: Value = serde_json::from_str(line).context("dump line")?;
        if v.get("end").is_some() {
            ensure!(
                v.get("questions").and_then(Value::as_u64) == Some(rows.len() as u64),
                "the dump's trailer does not count its rows"
            );
            finished = true;
            continue;
        }
        let row: Row = serde_json::from_value(v).context("dump row")?;
        let k = rows.len();
        let (index, qhash) = expected
            .get(k)
            .ok_or_else(|| anyhow!("the dump holds more rows than this run's {} problems", expected.len()))?;
        ensure!(row.index == *index, "dump row {k} is problem {}, this run's problem {k} is {index}", row.index);
        ensure!(&row.qhash == qhash, "problem {index}: prompt hash {} in the dump, {qhash} now", row.qhash);
        rows.push(row);
    }
    ensure!(!finished || rows.len() == expected.len(), "the dump is finished with {} of {} problems", rows.len(), expected.len());
    Ok((rows, finished))
}

fn summary(rows: &[Row], max_new: usize) -> String {
    let n = rows.len();
    let right = rows.iter().filter(|r| r.correct).count();
    let capped = rows.iter().filter(|r| r.stop == "cap").count();
    let gen: usize = rows.iter().map(|r| r.n_gen).sum();
    let decode: f64 = rows.iter().map(|r| r.decode_s).sum();
    let prefill: f64 = rows.iter().map(|r| r.prefill_s).sum();
    let acc = right as f64 / n.max(1) as f64;
    let se = (acc * (1.0 - acc) / n.max(1) as f64).sqrt();
    format!(
        "{right}/{n} right = {:.2} % ± {:.2} (binomial SE), cut by the cap ({max_new}) {capped} (counted wrong); \
         generated mean {:.1} tokens; decode {:.3} s a token, prefill {:.2} s a problem, {:.1} s a problem in all",
        100.0 * acc,
        100.0 * se,
        gen as f64 / n.max(1) as f64,
        decode / gen.max(1) as f64,
        prefill / n.max(1) as f64,
        (decode + prefill) / n.max(1) as f64
    )
}

// ---------------------------------------------------------------------------------------------------------------
// The model
// ---------------------------------------------------------------------------------------------------------------

fn device(name: &str) -> Result<Device> {
    Ok(match name {
        "cpu" => Device::Cpu,
        "metal" => Device::new_metal(0).context("Metal device (built with --features metal?)")?,
        "cuda" => Device::new_cuda(0).context("CUDA device (built with --features cuda?)")?,
        o => bail!("--device {o}: cpu, metal or cuda"),
    })
}

/// The experts of every verified layer resident on the device, as `inkrun` makes them.
#[allow(unused_variables, unused_mut, clippy::match_single_binding)]
fn make_resident(model: &mut Inkling, layers: &Path, dev: &Device) -> Result<()> {
    match dev {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        Device::Metal(_) => {
            let rt = llvq_llm::inkling::resident::served_runtime(dev)?;
            model.make_resident(layers, rt)?;
        }
        #[cfg(all(target_os = "linux", feature = "cuda"))]
        Device::Cuda(_) => {
            let rt = llvq_llm::inkling::resident_cuda::served_runtime(dev, &model.cfg)?;
            llvq_llm::inkling::resident_cuda::make_resident(model, layers, rt)?;
        }
        _ => eprintln!("inkgsm: no resident path on this device; every layer keeps its native experts"),
    }
    Ok(())
}

fn logits_vec(t: &candle_core::Tensor) -> Result<Vec<f32>> {
    Ok(t.flatten_all()?.to_dtype(candle_core::DType::F32)?.to_vec1::<f32>()?)
}

fn sha(path: &Path) -> Result<String> {
    llvq_llm::digest::sha256_file(path).with_context(|| format!("sha256 of {}", path.display()))
}

// ---------------------------------------------------------------------------------------------------------------
// Modes
// ---------------------------------------------------------------------------------------------------------------

fn cmd_prompts(args: &[String]) -> Result<()> {
    let mut a = flags(args)?;
    let tokenizer = PathBuf::from(a.remove("tokenizer").ok_or_else(|| anyhow!("--tokenizer is required"))?);
    let out = PathBuf::from(a.remove("out").ok_or_else(|| anyhow!("--out is required"))?);
    let problems = PathBuf::from(a.remove("problems").unwrap_or_else(|| DEFAULT_PROBLEMS.into()));
    let limit = count(&mut a, "limit", 3)?;
    let effort = a.remove("effort").unwrap_or_else(|| DEFAULT_EFFORT.into());
    let parquet = a.remove("parquet").map(PathBuf::from);
    no_more(&a)?;
    let chat = InkChat::new(
        tokenizers::Tokenizer::from_file(&tokenizer).map_err(|e| anyhow!("{}: {e}", tokenizer.display()))?,
    )?;
    let ps = read_problems(&std::fs::read_to_string(&problems)?, limit)?;
    if let Some(pq) = &parquet {
        check_against_parquet(&ps, &read_parquet(pq)?)?;
        eprintln!("inkgsm: the {limit} problems equal parquet rows 0..{limit} of {}", pq.display());
    }
    let e = effort_json(&effort)?;
    let mut cases = vec![];
    for p in &ps {
        let ids = render(&chat, &p.prompt, &e)?;
        cases.push(json!({"index": p.index, "content": p.prompt, "ids": ids,
                          "qhash": format!("{:016x}", llvq_llm::eval::token_fingerprint(&ids))}));
    }
    let doc = json!({"effort": effort, "tokenizer_sha256": sha(&tokenizer)?, "cases": cases});
    std::fs::write(&out, serde_json::to_string_pretty(&doc)?)?;
    eprintln!("inkgsm: {} prompts -> {}", ps.len(), out.display());
    Ok(())
}

fn cmd_score(paths: &[String]) -> Result<()> {
    ensure!(!paths.is_empty(), "inkgsm score <a.jsonl> [<b.jsonl> ...]");
    let mut dumps = vec![];
    for p in paths {
        let text = std::fs::read_to_string(p).with_context(|| format!("read {p}"))?;
        let mut d = gsm8k::parse_dump(&text, p)?;
        let (mut moved, mut capped) = (0usize, 0usize);
        for r in &mut d.rows {
            moved += usize::from(gsm8k::regrade(r)?);
            ensure!(r.stop == "eos" || r.stop == "cap", "{p}: problem {} has stop {:?}, not eos or cap", r.index, r.stop);
            if r.stop == "cap" {
                capped += 1;
                r.correct = false;
            }
        }
        let max_new = d.field("max_new");
        println!("{p}: {moved} rows re-graded by llvq_llm::gsm8k, {capped} cut by the cap counted wrong");
        println!(
            "    {} | {} | effort {} | max_new {max_new}\n    {}",
            d.field("model"),
            d.field("arithmetic"),
            d.field("effort"),
            summary(&d.rows, max_new.parse().unwrap_or(0))
        );
        dumps.push((p, d));
    }
    let (pa, a) = &dumps[0];
    for (pb, b) in &dumps[1..] {
        ensure!(a.field("max_new") == b.field("max_new"), "{pa} and {pb} ran different caps");
        ensure!(a.fingerprint == b.fingerprint, "{pa} and {pb}: run fingerprints differ, not the same prompt stream");
        let p = gsm8k::pair(a, b)?;
        println!(
            "\n{pa} − {pb}: {} problems, right {} and {}, only A {}, only B {}, A − B = {:+.2} pp, \
             95 % [{:+.2} ; {:+.2}], McNemar exact p = {:.3e}",
            p.n,
            p.right_a,
            p.right_b,
            p.only_a,
            p.only_b,
            100.0 * p.delta.0,
            100.0 * p.delta.1,
            100.0 * p.delta.2,
            p.p_mcnemar
        );
    }
    Ok(())
}

fn cmd_run(args: &[String], t_start: Instant) -> Result<i32> {
    let mut a = flags(args)?;
    let mut need = |k: &str| a.remove(k).ok_or_else(|| anyhow!("--{k} is required"));
    let config = PathBuf::from(need("config")?);
    let checkpoint = PathBuf::from(need("checkpoint")?);
    let layers = PathBuf::from(need("layers")?);
    let tokenizer = PathBuf::from(need("tokenizer")?);
    let dev_name = need("device")?;
    let dump_path = PathBuf::from(need("dump")?);
    let problems_path = PathBuf::from(a.remove("problems").unwrap_or_else(|| DEFAULT_PROBLEMS.into()));
    let limit = count(&mut a, "limit", DEFAULT_LIMIT)?;
    let max_new = count(&mut a, "max-new", DEFAULT_MAX_NEW)?;
    ensure!(max_new > 0, "--max-new 0 generates nothing to grade");
    let effort = a.remove("effort").unwrap_or_else(|| DEFAULT_EFFORT.into());
    let resident = match a.remove("resident").as_deref() {
        None | Some("verified") => true,
        Some("none") => false,
        Some(o) => bail!("--resident {o}: verified or none"),
    };
    let require_layers = count(&mut a, "require-layers", 0)?;
    let deadline = a.remove("deadline-seconds").map(|v| v.parse::<f64>()).transpose().context("--deadline-seconds")?;
    let parquet = a.remove("parquet").map(PathBuf::from);
    // The directory of the sealed file's two int4 tables (`bin/inktable`); absent, the tables are the checkpoint's.
    let tables_dir = a.remove("tables").map(PathBuf::from);
    no_more(&a)?;

    // Everything that can be refused without the model, first.
    let problems = read_problems(&std::fs::read_to_string(&problems_path).with_context(|| format!("read {}", problems_path.display()))?, limit)?;
    if let Some(pq) = &parquet {
        check_against_parquet(&problems, &read_parquet(pq)?)?;
        eprintln!("inkgsm: the {limit} problems equal parquet rows 0..{limit} of {}", pq.display());
    }
    let chat = InkChat::new(
        tokenizers::Tokenizer::from_file(&tokenizer).map_err(|e| anyhow!("{}: {e}", tokenizer.display()))?,
    )?;
    let stops = chat.stop_ids();
    let e = effort_json(&effort)?;
    let prompts: Vec<Vec<u32>> = problems.iter().map(|p| render(&chat, &p.prompt, &e)).collect::<Result<_>>()?;
    let expected: Vec<(usize, String)> = problems
        .iter()
        .zip(&prompts)
        .map(|(p, ids)| (p.index, format!("{:016x}", llvq_llm::eval::token_fingerprint(ids))))
        .collect();
    let stream: Vec<u32> = prompts.concat();
    let fingerprint = format!("{:016x}", llvq_llm::eval::token_fingerprint(&stream));

    // Which vocabulary tables the model reads: the sealed file's int4 g64 ones, by their sha256, or the checkpoint's.
    let tables_header: Value = match &tables_dir {
        Some(d) => json!({
            "embed_sha256": sha(&d.join(llvq_llm::inkling::tables::EMBED_FILE))?,
            "unembed_sha256": sha(&d.join(llvq_llm::inkling::tables::UNEMBED_FILE))?,
            "encoding": "int4 g64, decoded to f32",
        }),
        None => json!("checkpoint, BF16"),
    };
    let base_header = |resident_layers: &Value| -> Result<Value> {
        Ok(json!({
            gsm8k::DUMP_TAG: gsm8k::DUMP_VERSION,
            "model": "thinkingmachines/Inkling-2-Mini",
            "arithmetic": (if resident { "llvq engine, routed experts resident in lattice codes" }
                           else { "llvq engine, native experts" }),
            "engine": "inkgsm",
            "device": dev_name,
            "dtype": "f32",
            "kv": "engine",
            "max_new": max_new,
            "think": effort != "none",
            "effort": effort,
            "limit": limit.to_string(),
            "questions": limit,
            "selection": format!("the first {limit} rows of the test split in parquet order"),
            "dataset": llvq_llm::corpus::GSM8K_REPO,
            "split": "test",
            "revision": GSM8K_REVISION,
            "instruction": gsm8k::INSTRUCTION,
            "stop_ids": stops,
            "cap_rule": "a completion cut by the cap counts wrong",
            "problems_sha256": sha(&problems_path)?,
            "tokenizer_sha256": sha(&tokenizer)?,
            "config_sha256": sha(&config)?,
            "template_sha256": llvq_llm::inkchat::TEMPLATE_SHA256,
            "tables": tables_header,
            "resident_layers": resident_layers,
            "prompt_stream": fingerprint,
        }))
    };

    // A finished dump needs no model: read its header for the protocol fields only.
    let old_text = match std::fs::read_to_string(&dump_path) {
        Ok(t) => Some(t),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
        Err(e) => return Err(e).with_context(|| format!("read {}", dump_path.display())),
    };
    if let Some(t) = &old_text {
        if let Ok(d) = gsm8k::parse_dump(t, &dump_path.display().to_string()) {
            let h = d.header.clone();
            let want = base_header(h.get("resident_layers").unwrap_or(&Value::Null))?;
            read_existing(complete_lines(t), &want, &expected)?;
            println!("inkgsm: {} is finished: {}", dump_path.display(), summary(&d.rows, max_new));
            return Ok(0);
        }
    }

    let dev = device(&dev_name)?;
    let t = Instant::now();
    let mut model = Inkling::load(&config, &checkpoint, &dev)?;
    if resident {
        make_resident(&mut model, &layers, &dev)?;
    }
    if let Some(d) = &tables_dir {
        let t = llvq_llm::inkling::tables::load(d, &model.cfg, &dev)?;
        eprintln!(
            "inkgsm: tables of the sealed file from {}: embed {}, unembed {}",
            d.display(),
            &t.embed_sha256[..16],
            &t.unembed_sha256[..16]
        );
        model.embed_table = Some(t.embed);
        model.out_proj = Some(Box::new(t.head));
    }
    dev.synchronize()?;
    let resident_layers = model.substituted_layers();
    eprintln!("inkgsm: loaded in {:.1} s, {} layers resident: {resident_layers:?}", t.elapsed().as_secs_f64(), resident_layers.len());
    ensure!(
        resident_layers.len() >= require_layers,
        "{} layers resident, --require-layers {require_layers}",
        resident_layers.len()
    );
    let header = base_header(&json!(resident_layers))?;

    let (mut rows, mut file) = match &old_text {
        Some(t) if !t.trim().is_empty() => {
            let keep = complete_lines(t);
            if keep.len() < t.len() {
                eprintln!("inkgsm: dropping a last line cut at {} bytes", t.len() - keep.len());
            }
            let (rows, finished) = read_existing(keep, &header, &expected)?;
            ensure!(!finished, "unreachable: a finished dump returned above");
            let f = std::fs::OpenOptions::new().write(true).open(&dump_path)?;
            f.set_len(keep.len() as u64)?;
            let mut f = std::io::BufWriter::new(f);
            std::io::Seek::seek(f.get_mut(), std::io::SeekFrom::End(0))?;
            eprintln!("inkgsm: resuming {} after {} problems", dump_path.display(), rows.len());
            f.flush()?;
            (rows, f)
        }
        _ => {
            let mut f = std::io::BufWriter::new(std::fs::File::create(&dump_path)?);
            writeln!(f, "{header}")?;
            f.flush()?;
            (vec![], f)
        }
    };

    let tok = chat.tokenizer();
    let done = rows.len();
    for (k, (p, ids)) in problems.iter().zip(&prompts).enumerate().skip(done) {
        if deadline.is_some_and(|d| t_start.elapsed().as_secs_f64() > d) {
            eprintln!("inkgsm: deadline passed before problem {}; {} of {limit} in the dump", p.index, rows.len());
            return Ok(EXIT_DEADLINE);
        }
        let mut cache = model.new_cache();
        let t0 = Instant::now();
        let first = logits_vec(&CausalLm::prefill(&model, ids, &mut cache)?)?;
        let prefill_s = t0.elapsed().as_secs_f64();
        let t1 = Instant::now();
        let mut steps: Vec<f64> = Vec::new();
        let r = greedy(&first, &stops, max_new, |token, j| {
            if deadline.is_some_and(|d| t_start.elapsed().as_secs_f64() > d) {
                return Err(anyhow::Error::new(DeadlineHit));
            }
            let ts = Instant::now();
            let l = logits_vec(&CausalLm::step(&model, token, ids.len() + j, &mut cache)?)?;
            steps.push(ts.elapsed().as_secs_f64());
            Ok(l)
        });
        let (said, stop) = match r {
            Ok(x) => x,
            Err(e) if e.downcast_ref::<DeadlineHit>().is_some() => {
                eprintln!("inkgsm: deadline passed during problem {}, dropped; {} of {limit} in the dump", p.index, rows.len());
                return Ok(EXIT_DEADLINE);
            }
            Err(e) => return Err(e),
        };
        let decode_s = t1.elapsed().as_secs_f64();
        let completion = tok.decode(&said, false).map_err(|e| anyhow!("{e}"))?;
        let (source, extracted, correct) = strict_correct(&completion, &p.gold, stop.name());
        let turn = {
            let mut tp = TurnParser::new(&chat);
            let mut all = said.clone();
            if stop == Stop::Eos {
                all.extend(stops.first());
            }
            match tp.push_ids(&all).and_then(|_| tp.finish()) {
                Ok(_) => "ok".to_string(),
                Err(e) => format!("refused: {e}"),
            }
        };
        let row = Row {
            index: p.index,
            qhash: expected[k].1.clone(),
            n_prompt: ids.len(),
            n_gen: said.len(),
            stop: stop.name().to_string(),
            gold: p.gold.clone(),
            extracted,
            source: source.name().to_string(),
            correct,
            prefill_s,
            decode_s,
            completion,
        };
        let mut v = serde_json::to_value(&row)?;
        let mut st = steps.clone();
        st.sort_by(f64::total_cmp);
        let obj = v.as_object_mut().expect("a row is an object");
        obj.insert("turn".into(), json!(turn));
        obj.insert("step_s_median".into(), json!(st.get(st.len() / 2)));
        writeln!(file, "{v}")?;
        file.flush()?;
        rows.push(row);
        let right = rows.iter().filter(|r| r.correct).count();
        eprintln!(
            "  [{:>3}/{limit}] #{:<4} {} gen {:>4} {:<3} {:<5} acc {:>5.1} %  ({:.1} s prefill, {:.1} s decode)",
            k + 1,
            p.index,
            if correct { "ok" } else { "--" },
            said.len(),
            stop.name(),
            source.name(),
            100.0 * right as f64 / rows.len() as f64,
            prefill_s,
            decode_s
        );
    }
    writeln!(file, "{}", json!({"end": true, "fingerprint": fingerprint, "questions": rows.len()}))?;
    file.flush()?;
    println!("inkgsm: {}", summary(&rows, max_new));
    Ok(0)
}

fn main() -> Result<()> {
    let t_start = Instant::now();
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("run") => {
            let code = cmd_run(&args[1..], t_start)?;
            if code != 0 {
                std::process::exit(code);
            }
            Ok(())
        }
        Some("prompts") => cmd_prompts(&args[1..]),
        Some("score") => cmd_score(&args[1..]),
        _ => usage(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fake model: logits that put the maximum on `seq[k]` after the `k`-th fed token.
    fn one_hot(i: u32) -> Vec<f32> {
        let mut v = vec![0.0; 16];
        v[i as usize] = 1.0;
        v
    }

    fn run(seq: &[u32], stops: &[u32], max_new: usize) -> (Vec<u32>, Stop, usize) {
        let mut fed = 0;
        let (said, stop) = greedy(&one_hot(seq[0]), stops, max_new, |_, k| {
            assert_eq!(k, fed, "tokens are fed in order");
            fed += 1;
            Ok(one_hot(seq[k + 1]))
        })
        .unwrap();
        (said, stop, fed)
    }

    #[test]
    fn the_stop_id_ends_the_completion_and_is_not_kept() {
        let (said, stop, fed) = run(&[3, 4, 9, 5, 6], &[9], 100);
        assert_eq!((said, stop, fed), (vec![3, 4], Stop::Eos, 2));
        // A stop id as the very first token: an empty completion.
        let (said, stop, fed) = run(&[9, 4], &[9], 100);
        assert_eq!((said, stop, fed), (vec![], Stop::Eos, 0));
    }

    #[test]
    fn the_cap_keeps_exactly_max_new_tokens_and_feeds_one_fewer() {
        let (said, stop, fed) = run(&[1, 2, 3, 4, 5, 9], &[9], 3);
        assert_eq!((said, stop, fed), (vec![1, 2, 3], Stop::Cap, 2));
        // The stop id right after the cap is never read: the cap ends first, as in bin/gsm8k.
        let (said, stop, _) = run(&[1, 2, 9], &[9], 2);
        assert_eq!((said, stop), (vec![1, 2], Stop::Cap));
        let (said, stop, _) = run(&[1, 2, 9], &[9], 3);
        assert_eq!((said, stop), (vec![1, 2], Stop::Eos));
    }

    #[test]
    fn argmax_takes_the_first_maximum_and_refuses_no_maximum() {
        assert_eq!(argmax(&[0.0, 2.0, 2.0, 1.0]).unwrap(), 1);
        assert!(argmax(&[f32::NAN, f32::NAN]).is_err());
        assert!(argmax(&[]).is_err());
    }

    #[test]
    fn a_capped_completion_counts_wrong_even_with_the_right_box() {
        let (s, x, ok) = strict_correct("so \\boxed{18}", "18", "eos");
        assert_eq!((s, x.as_deref(), ok), (gsm8k::Source::Boxed, Some("18"), true));
        let (_, x, ok) = strict_correct("so \\boxed{18}", "18", "cap");
        assert_eq!((x.as_deref(), ok), (Some("18"), false));
        assert!(!strict_correct("so \\boxed{17}", "18", "eos").2);
    }

    fn line(id: usize, prompt: &str, gold: &str) -> String {
        json!({"id": format!("gsm8k-test-{id:04}"), "kind": "math", "prompt": prompt, "gold": gold}).to_string()
    }

    fn body(q: &str) -> String {
        gsm8k::user_body(q)
    }

    #[test]
    fn the_problems_are_the_head_of_the_split_in_order() {
        let text = [
            r#"{"_comment": "x"}"#.to_string(),
            line(0, &body("A?"), "1"),
            line(1, &body("B?"), "2"),
            line(2, &body("C?"), "3"),
        ]
        .join("\n");
        let p = read_problems(&text, 2).unwrap();
        assert_eq!(p.iter().map(|p| p.index).collect::<Vec<_>>(), vec![0, 1]);
        assert!(read_problems(&text, 4).is_err(), "more asked than held");
        let skipped = [line(0, &body("A?"), "1"), line(2, &body("C?"), "3")].join("\n");
        assert!(read_problems(&skipped, 2).is_err(), "row 1 missing: not the head of the split");
        let no_instr = line(0, "A?", "1");
        assert!(read_problems(&no_instr, 1).is_err(), "no instruction");
        let bad_gold = line(0, &body("A?"), "1,000");
        assert!(read_problems(&bad_gold, 1).is_err(), "gold not canonical");
    }

    #[test]
    fn the_parquet_check_catches_a_changed_question_or_gold() {
        let p = vec![Problem { index: 0, prompt: body("A?"), gold: "18".into() }];
        let rows = vec![("A?".to_string(), "x\n#### 18".to_string())];
        check_against_parquet(&p, &rows).unwrap();
        assert!(check_against_parquet(&p, &[("B?".into(), "#### 18".into())]).is_err());
        assert!(check_against_parquet(&p, &[("A?".into(), "#### 17".into())]).is_err());
        assert!(check_against_parquet(&p, &[]).is_err());
    }

    fn row(index: usize, qhash: &str) -> Row {
        Row {
            index,
            qhash: qhash.into(),
            n_prompt: 3,
            n_gen: 2,
            stop: "eos".into(),
            gold: "1".into(),
            extracted: Some("1".into()),
            source: "boxed".into(),
            correct: true,
            prefill_s: 0.1,
            decode_s: 0.2,
            completion: "\\boxed{1}".into(),
        }
    }

    #[test]
    fn a_dump_resumes_after_its_rows_and_refuses_another_protocol() {
        let header = json!({gsm8k::DUMP_TAG: 1, "max_new": 1536});
        let expected = vec![(0, "a".to_string()), (1, "b".to_string()), (2, "c".to_string())];
        let mut text = format!("{header}\n");
        for r in [row(0, "a"), row(1, "b")] {
            let mut v = serde_json::to_value(&r).unwrap();
            v.as_object_mut().unwrap().insert("turn".into(), json!("ok"));
            text.push_str(&format!("{v}\n"));
        }
        let (rows, done) = read_existing(&text, &header, &expected).unwrap();
        assert_eq!((rows.len(), done), (2, false));

        // A line cut by a kill is dropped before reading.
        let cut = format!("{text}{{\"index\": 2, \"qha");
        assert_eq!(complete_lines(&cut), text);
        assert_eq!(read_existing(complete_lines(&cut), &header, &expected).unwrap().0.len(), 2);

        // Another cap, another prompt, another order: refused.
        assert!(read_existing(&text, &json!({gsm8k::DUMP_TAG: 1, "max_new": 1024}), &expected).is_err());
        let other = vec![(0, "a".to_string()), (1, "z".to_string())];
        assert!(read_existing(&text, &header, &other).is_err());
        let order = vec![(1, "b".to_string()), (0, "a".to_string())];
        assert!(read_existing(&text, &header, &order).is_err());

        // Finished: the trailer counts the rows, and nothing follows it.
        let mut fin = text.clone();
        fin.push_str(&format!("{}\n", serde_json::to_value(row(2, "c")).unwrap()));
        fin.push_str("{\"end\": true, \"fingerprint\": \"f\", \"questions\": 3}\n");
        assert!(read_existing(&fin, &header, &expected).unwrap().1);
        assert!(read_existing(&format!("{fin}{}\n", serde_json::to_value(row(2, "c")).unwrap()), &header, &expected).is_err());
        // An empty file is a fresh start.
        assert_eq!(read_existing("", &header, &expected).unwrap().0.len(), 0);
    }

    #[test]
    fn the_effort_is_a_named_level_or_a_number_in_range() {
        assert!(effort_json("none").is_ok());
        assert!(effort_json("0.2").is_ok());
        assert!(effort_json("turbo").is_err());
        assert!(effort_json("1.5").is_err());
    }
}
