//! GSM8K through **our** pipeline: the served kernel, the dense reconstruction
//! of a sealed file, or a reference checkpoint.
//!
//! ```text
//! cargo run --release -p llvq-llm --features metal --bin gsm8k -- <model> [device] [limit]
//! LLVQ_CONFIG=configs/qwen3-4b-tetra-e4.json \
//!   cargo run --release -p llvq-llm --features cuda --bin gsm8k -- qwen3-4b-sealed.bin cuda
//! ```
//!
//! `<model>` is a sealed `.llvq` / `.bin`, or a checkpoint: a repo id,
//! `repo@revision`, or a local directory. `LLVQ_CONFIG` puts a sealed file on
//! the served kernel instead of its dense reconstruction, the same door as
//! `bin/mmlu` and `bin/chat`. `limit` is a sample size; absent, the whole test
//! split (1,319 problems).
//!
//! ## Why a generation benchmark
//!
//! MMLU reads one logit per question. What the served object does is write
//! text, one token after another, and a reasoning error compounds along a
//! chain that a single logit never sees. GSM8K scores the chain: grade-school
//! problems whose answer is one integer, reached in several steps.
//!
//! It is also the benchmark where the served kernel is cheap. A prompt token
//! costs 3.94 ms through the kernel at 4B, which is why MMLU through it is
//! dear; a generated token costs 8.8 ms against 23 ms on the dense path
//! (*measured*, `f1e-census-2026-09-11`, `paper-table-2026-09-25`). A
//! short-prompt, long-answer benchmark is the one the kernel should run.
//!
//! The protocol and the grading are in [`llvq_llm::gsm8k`], pure and tested.
//! This binary only generates around them.
//!
//! ## Environment
//!
//! * `LLVQ_GSM8K_DUMP=<path>`: one JSON line per problem, raw completion
//!   included, after a header line and before a trailer line. Flushed per
//!   problem, so a killed run keeps what it paid for; the missing trailer
//!   then marks it unfinished and `bin/gsm8kpair` refuses it.
//! * `LLVQ_GSM8K_MAX_NEW` (default 1024): the cap on generated tokens.
//! * `LLVQ_GSM8K_THINK=1`: let Qwen3 reason in its `<think>` block. Off by
//!   default: the block is pre-filled empty, the non-thinking mode.
//! * `LLVQ_DTYPE`, `LLVQ_KV`, `LLVQ_CONFIG`: as for `bin/mmlu`.
//!
//! ## What the timings are
//!
//! Per problem, the prefill and the decode are timed apart, each closed by a
//! device synchronisation. They price a campaign. They are not throughput
//! figures: those come from `fusedrun`, under its protocol of discarded and
//! timed rounds.

use candle_core::{DType, IndexOp, Tensor, D};
use llvq_llm::chatfmt::{open_assistant, turn, Marks};
use llvq_llm::gsm8k::{self, Row};
use llvq_llm::model::NoCapture;
use std::io::Write;
use std::time::Instant;

/// A count from the environment, `default` when unset or blank.
fn parse_count(key: &str, v: Option<&str>, default: usize) -> anyhow::Result<usize> {
    match v.map(str::trim) {
        None | Some("") => Ok(default),
        Some(t) => t
            .parse()
            .map_err(|_| anyhow::anyhow!("{key}={t:?} is not a count")),
    }
}

/// `0`, `1`, or unset. Anything else is refused: a flag that reads `yes` as
/// off is an arm that lies about its own protocol.
fn parse_flag(key: &str, v: Option<&str>) -> anyhow::Result<bool> {
    match v.map(str::trim) {
        None | Some("") | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(other) => anyhow::bail!("{key}={other:?}: accepted values are 0 and 1"),
    }
}

fn env_count(key: &str, default: usize) -> anyhow::Result<usize> {
    parse_count(key, std::env::var(key).ok().as_deref(), default)
}

fn env_flag(key: &str) -> anyhow::Result<bool> {
    parse_flag(key, std::env::var(key).ok().as_deref())
}

/// Median and 90th percentile of a count, nearest rank.
fn quantiles(v: &[usize]) -> (usize, usize) {
    if v.is_empty() {
        return (0, 0);
    }
    let mut s = v.to_vec();
    s.sort_unstable();
    let at = |q: f64| s[((q * s.len() as f64).ceil() as usize).clamp(1, s.len()) - 1];
    (at(0.5), at(0.9))
}

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let model_arg = a.first().cloned().ok_or_else(|| {
        anyhow::anyhow!("usage: gsm8k <sealed file | checkpoint> [device] [limit]")
    })?;
    let dev_name = a.get(1).map(String::as_str).unwrap_or("cpu");
    let device = llvq_llm::eval::device(dev_name)?;
    let limit: usize = match a.get(2) {
        None => usize::MAX,
        Some(s) => s
            .parse()
            .map_err(|_| anyhow::anyhow!("limit {s:?} is not a count"))?,
    };
    // Every knob resolved before the model is fetched: a typo must cost a
    // second, not a load.
    let max_new = env_count("LLVQ_GSM8K_MAX_NEW", 1024)?;
    anyhow::ensure!(max_new > 0, "LLVQ_GSM8K_MAX_NEW=0 generates nothing to score");
    let think = env_flag("LLVQ_GSM8K_THINK")?;
    let dtype = llvq_llm::eval::dtype(DType::F16)?;
    let kv_mode = llvq_llm::kvq::KvMode::from_env().map_err(anyhow::Error::msg)?;
    let kv_store = llvq_llm::kvq::KvStore::from_env().map_err(anyhow::Error::msg)?;
    let served = llvq_llm::served::Served::from_env().map_err(anyhow::Error::msg)?;

    // ---- the model ----
    let t_load = Instant::now();
    let (mut model, tok, label, arithmetic) = if let Some(cfg) = &served {
        anyhow::ensure!(
            llvq_llm::sealed::is_sealed_path(&model_arg),
            "LLVQ_CONFIG names a served config, but {model_arg} is not a sealed file. \
             The served path reads a .llvq; a checkpoint has nothing to transcode."
        );
        println!("{}", cfg.provenance());
        let f = llvq_llm::served::load_resolved(
            &model_arg,
            &device,
            dtype,
            cfg.layout,
            cfg.embed,
            cfg.rot_share,
            cfg.fuse,
            cfg.kv,
            Some("LLVQ_CONFIG"),
        )
        .map_err(|e| {
            anyhow::anyhow!(
                "LLVQ_CONFIG={} asks for the served kernel, and loading it failed: {e}",
                cfg.path.display()
            )
        })?;
        (
            f.model,
            f.tokenizer,
            format!("{model_arg} [LLVQ 2-bit, SERVED KERNEL, {}]", cfg.layout.name()),
            "served kernel",
        )
    } else if llvq_llm::sealed::is_sealed_path(&model_arg) {
        let s = llvq_llm::sealed::load(&model_arg, dtype, &device, kv_mode)?;
        (
            s.model,
            s.tokenizer,
            format!("{model_arg} [LLVQ 2-bit, sealed]"),
            "dense reconstruction",
        )
    } else {
        let ck = llvq_llm::loader::Checkpoint::fetch(&model_arg)?;
        let tok = ck.tokenizer()?;
        let vb = ck.var_builder(dtype, &device)?;
        (
            llvq_llm::model::Qwen3::new(&ck.config, vb, kv_mode)?,
            tok,
            format!("{} [reference checkpoint]", ck.source.describe()),
            "reference checkpoint",
        )
    };
    model.set_kv_store(kv_store);
    let load_s = t_load.elapsed().as_secs_f64();
    let kv_name = match &served {
        Some(cfg) => cfg.kv.name(),
        None => kv_mode.name(),
    };
    eprintln!(
        "model: {label}\ndevice: {device:?}, dtype {}, kv {kv_name}, loaded in {load_s:.1} s",
        llvq_llm::eval::dtype_name(dtype)
    );

    let marks = Marks::resolve(&tok)?;
    let stops = marks.stops();
    anyhow::ensure!(
        !think || marks.think.is_some(),
        "LLVQ_GSM8K_THINK=1, but this tokenizer has no <think> block"
    );
    let positions = model.config().max_position_embeddings;

    // ---- the problems ----
    let (items, revision) = llvq_llm::corpus::gsm8k_split("test")?;
    let picked = gsm8k::select(items.len(), limit);
    let limit_name = if picked.len() == items.len() {
        "census".to_string()
    } else {
        picked.len().to_string()
    };
    eprintln!(
        "GSM8K test: {} problems, scoring {} ({limit_name}), revision {}, max_new {max_new}, \
         reasoning {}",
        items.len(),
        picked.len(),
        revision.as_deref().unwrap_or("not from the Hub cache"),
        if think { "on" } else { "off" }
    );

    let mut dump = match std::env::var("LLVQ_GSM8K_DUMP") {
        Ok(p) if !p.is_empty() => {
            let mut w = std::io::BufWriter::new(std::fs::File::create(&p)?);
            let mut header = serde_json::json!({
                gsm8k::DUMP_TAG: gsm8k::DUMP_VERSION,
                "model": label,
                "arithmetic": arithmetic,
                "device": dev_name,
                "dtype": llvq_llm::eval::dtype_name(dtype),
                "kv": kv_name,
                "max_new": max_new,
                "think": think,
                "limit": limit_name,
                "questions": picked.len(),
                "dataset": llvq_llm::corpus::GSM8K_REPO,
                "split": "test",
                "revision": revision,
                "instruction": gsm8k::INSTRUCTION,
                "sample_seed": format!("{:#x}", gsm8k::SAMPLE_SEED),
            });
            // Every served choice as its own key, so a reader tells a kernel
            // dump from a dense one of the same file without parsing a note.
            if let Some(cfg) = &served {
                let h = header.as_object_mut().expect("an object");
                h.insert("config".into(), cfg.path.display().to_string().into());
                h.insert("layout".into(), cfg.layout.name().into());
                h.insert("embed".into(), cfg.embed.name().into());
                h.insert("rot_share".into(), cfg.rot_share.name().into());
                h.insert("fuse".into(), cfg.fuse.name().into());
            }
            writeln!(w, "{header}")?;
            w.flush()?;
            eprintln!("dumping per-problem results to {p}");
            Some(w)
        }
        _ => None,
    };

    // ---- generate and grade ----
    let dev = model.device().clone();
    let mut prompt_stream: Vec<u32> = Vec::new();
    let mut rows: Vec<Row> = Vec::with_capacity(picked.len());
    let t_run = Instant::now();
    for (k, &i) in picked.iter().enumerate() {
        let it = &items[i];
        let gold = gsm8k::gold(&it.answer)
            .ok_or_else(|| anyhow::anyhow!("gsm8k/test row {i}: no `#### <number>` to grade against"))?;
        let mut ids = turn(&tok, &marks, "user", &gsm8k::user_body(&it.question))?;
        ids.extend(open_assistant(&tok, &marks, think)?);
        anyhow::ensure!(
            ids.len() + max_new <= positions,
            "problem {i}: {} prompt tokens plus {max_new} new ones exceed the {positions} positions",
            ids.len()
        );
        prompt_stream.extend_from_slice(&ids);

        // Prefill in chunks of `MAX_ROWS`: a served group without a rows
        // kernel spends one launch a row and refuses past that budget. A
        // GSM8K prompt is far under it; the chunking only keeps a long one
        // from failing late, and a chunk boundary changes nothing the cache
        // does not already carry.
        let mut caches = model.fresh_caches();
        let t0 = Instant::now();
        let mut offset = 0usize;
        let mut last = None;
        for chunk in ids.chunks(llvq_llm::model::MAX_ROWS) {
            let input = Tensor::from_slice(chunk, (1, chunk.len()), &dev)?;
            last = Some(model.hidden_cached(&input, offset, &mut caches, &mut NoCapture)?);
            offset += chunk.len();
        }
        let mut h = last.expect("a prompt holds at least its chat markers");
        dev.synchronize()?;
        let prefill_s = t0.elapsed().as_secs_f64();

        let t1 = Instant::now();
        let mut said: Vec<u32> = Vec::new();
        let stop = loop {
            let logits = model.logits_last(&h)?.i((0, 0))?;
            let next = logits.argmax(D::Minus1)?.to_scalar::<u32>()?;
            if stops.contains(&next) {
                break "eos";
            }
            said.push(next);
            if said.len() >= max_new {
                break "cap";
            }
            let input = Tensor::from_slice(&[next], (1, 1), &dev)?;
            h = model.hidden_cached(&input, offset, &mut caches, &mut NoCapture)?;
            offset += 1;
        };
        let decode_s = t1.elapsed().as_secs_f64();

        let completion = tok
            .decode(&said, false)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let (source, extracted) = gsm8k::extract(&completion);
        let correct = gsm8k::is_correct(extracted.as_deref(), &gold);
        let row = Row {
            index: i,
            qhash: format!("{:016x}", llvq_llm::eval::token_fingerprint(&ids)),
            n_prompt: ids.len(),
            n_gen: said.len(),
            stop: stop.to_string(),
            gold,
            extracted,
            source: source.name().to_string(),
            correct,
            prefill_s,
            decode_s,
            completion,
        };
        if let Some(w) = dump.as_mut() {
            writeln!(w, "{}", serde_json::to_string(&row)?)?;
            w.flush()?;
        }
        rows.push(row);
        let right = rows.iter().filter(|r| r.correct).count();
        let r = rows.last().expect("just pushed");
        eprintln!(
            "  [{:>4}/{}] #{:<4} {} gen {:>4} {:<3} {:<5} acc {:>5.1} %  ({:.0} s)",
            k + 1,
            picked.len(),
            i,
            if r.correct { "ok" } else { "--" },
            r.n_gen,
            r.stop,
            r.source,
            100.0 * right as f64 / rows.len() as f64,
            t_run.elapsed().as_secs_f64()
        );
    }
    let wall_s = t_run.elapsed().as_secs_f64();

    let fingerprint = llvq_llm::eval::token_fingerprint(&prompt_stream);
    if let Some(w) = dump.as_mut() {
        writeln!(
            w,
            "{}",
            serde_json::json!({
                "end": true,
                "fingerprint": format!("{fingerprint:016x}"),
                "questions": rows.len(),
            })
        )?;
        w.flush()?;
    }

    // ---- the result ----
    let n = rows.len();
    let right = rows.iter().filter(|r| r.correct).count();
    let acc = right as f64 / n.max(1) as f64;
    let se = (acc * (1.0 - acc) / n.max(1) as f64).sqrt();
    let count = |f: &dyn Fn(&Row) -> bool| rows.iter().filter(|r| f(r)).count();
    let boxed = count(&|r| r.source == "boxed");
    let last_number = count(&|r| r.source == "last");
    let none = count(&|r| r.source == "none");
    let capped = count(&|r| r.stop == "cap");
    let boxed_right = count(&|r| r.source == "boxed" && r.correct);
    let gens: Vec<usize> = rows.iter().map(|r| r.n_gen).collect();
    let prompts: usize = rows.iter().map(|r| r.n_prompt).sum();
    let generated: usize = gens.iter().sum();
    let (median, p90) = quantiles(&gens);
    let prefill: f64 = rows.iter().map(|r| r.prefill_s).sum();
    let decode: f64 = rows.iter().map(|r| r.decode_s).sum();

    println!("\n{label}");
    println!(
        "GSM8K zero-shot, reasoning {}, {n} problems of {} ({limit_name}), dtype {}, kv {kv_name}, \
         max_new {max_new}, tokens {fingerprint:016x}",
        if think { "on" } else { "off" },
        items.len(),
        llvq_llm::eval::dtype_name(dtype)
    );
    println!(
        "  accuracy          = {:.2} % ({right}/{n}) ± {:.2} (binomial SE)",
        100.0 * acc,
        100.0 * se
    );
    println!(
        "  read from a box   = {boxed_right}/{boxed} right; last number {last_number}; nothing {none}"
    );
    println!(
        "  generated tokens  = mean {:.1}, median {median}, p90 {p90}, max {}; stopped at the cap {capped}",
        generated as f64 / n.max(1) as f64,
        gens.iter().copied().max().unwrap_or(0)
    );
    println!(
        "  prompt tokens     = mean {:.1}",
        prompts as f64 / n.max(1) as f64
    );
    println!(
        "  timing ({dev_name}) = prefill {:.3} ms a prompt token, decode {:.3} ms a token ({:.1} tok/s), \
         {:.2} s a problem, {:.0} s in all, load {load_s:.0} s",
        1e3 * prefill / prompts.max(1) as f64,
        1e3 * decode / generated.max(1) as f64,
        generated as f64 / decode.max(1e-9),
        wall_s / n.max(1) as f64,
        wall_s
    );
    println!("  timings price a campaign; they are not a throughput measurement");
    if dump.is_some() {
        println!(
            "\n  Dump written. Two arms are compared on the dumps:\n  \
             `cargo run --release -p llvq-llm --bin gsm8kpair -- <a> <b>`."
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantiles_are_nearest_rank() {
        assert_eq!(quantiles(&[]), (0, 0));
        assert_eq!(quantiles(&[5]), (5, 5));
        assert_eq!(quantiles(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), (5, 9));
        assert_eq!(quantiles(&[10, 1, 9, 2, 8, 3, 7, 4, 6, 5]), (5, 9));
    }

    #[test]
    fn a_flag_is_zero_one_or_refused() {
        assert!(!parse_flag("F", None).unwrap());
        assert!(!parse_flag("F", Some(" ")).unwrap());
        assert!(!parse_flag("F", Some("0")).unwrap());
        assert!(parse_flag("F", Some("1")).unwrap());
        assert!(parse_flag("F", Some("yes")).is_err(), "`yes` must not read as off");
        assert!(parse_flag("F", Some("true")).is_err());
    }

    #[test]
    fn a_count_defaults_only_when_unset() {
        assert_eq!(parse_count("C", None, 1024).unwrap(), 1024);
        assert_eq!(parse_count("C", Some(""), 1024).unwrap(), 1024);
        assert_eq!(parse_count("C", Some(" 512 "), 1024).unwrap(), 512);
        assert!(parse_count("C", Some("1e3"), 1024).is_err());
        assert!(parse_count("C", Some("-1"), 1024).is_err());
    }
}
