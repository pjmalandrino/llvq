//! Gate B-ii of `proofs/preregistration-servi-au-bit-2026-11-26.md`: the real
//! Inkling 2 Mini on the CPU, loaded as a CPU build of `llvq serve` loads it
//! (`experts_cpu::load_served`, the long-prompt settings it sets), a prompt
//! read straight and read cut at `b` = 2,048 with the cache's state written
//! to a file, loaded into a new cache, and the rest read. The logits of the
//! prompt's last position and of 8 greedy steps after it are written as raw
//! f32 for each arm, with their sha256 and the times of the write and the
//! load, in a JSON report.
//!
//! A heavy job: about 12 GB of memory and minutes of every core it is given.
//! Ignored by default; it needs, and fails naming, three variables:
//! `INKLING_DIR` (the file set: config.json, model.safetensors, layers/,
//! tables/), `INKLING_IDS` (a JSON object whose `ids` array is the prompt) and
//! `INKLING_OUT` (an existing directory for the report and the logits). Run
//! it in release, without the `metal` feature:
//!
//! ```text
//! INKLING_DIR=... INKLING_IDS=... INKLING_OUT=... INKLING_LEN=2600 \
//!   cargo test --release -p llvq-runtime --test inkling_cpu_prefix_real -- --ignored --nocapture
//! ```

use llvq_llm::candle::{DType, Device, Tensor};
use llvq_llm::inkling::InklingCache;
use llvq_llm::lm::CausalLm;
use sha2::{Digest, Sha256};
use std::path::PathBuf;
use std::time::Instant;

const B: usize = 2048;
const STEPS: usize = 8;

fn var(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("{name} is not set: see the head of tests/inkling_cpu_prefix_real.rs"))
}

fn row(t: &Tensor) -> Vec<f32> {
    t.flatten_all().unwrap().to_dtype(DType::F32).unwrap().to_device(&Device::Cpu).unwrap().to_vec1::<f32>().unwrap()
}

fn argmax(v: &[f32]) -> u32 {
    v.iter().enumerate().fold((0, f32::NEG_INFINITY), |(bi, bv), (i, &x)| if x > bv { (i, x) } else { (bi, bv) }).0 as u32
}

/// The last prompt row then `STEPS` greedy rows on `cache`, and the tokens.
fn steps(m: &llvq_llm::inkling::Inkling, cache: &mut InklingCache, last: Vec<f32>) -> (Vec<Vec<f32>>, Vec<u32>) {
    let mut rows = vec![last];
    let mut toks = vec![];
    for _ in 0..STEPS {
        let t = argmax(rows.last().unwrap());
        toks.push(t);
        let pos = cache.len();
        rows.push(row(&m.step(t, pos, cache).unwrap()));
    }
    (rows, toks)
}

fn write(path: &PathBuf, rows: &[Vec<f32>]) -> String {
    let bytes: Vec<u8> = rows.iter().flatten().flat_map(|x| x.to_le_bytes()).collect();
    std::fs::write(path, &bytes).unwrap();
    Sha256::digest(&bytes).iter().map(|b| format!("{b:02x}")).collect()
}

#[test]
#[ignore = "the real model on the CPU: INKLING_DIR, INKLING_IDS, INKLING_OUT"]
fn a_state_saved_and_loaded_at_2048_gives_the_straight_reads_logits_on_the_cpu() {
    let dir = PathBuf::from(var("INKLING_DIR"));
    let ids_file = PathBuf::from(var("INKLING_IDS"));
    let out = PathBuf::from(var("INKLING_OUT"));
    assert!(out.is_dir(), "INKLING_OUT {}: not a directory", out.display());
    let len: usize = std::env::var("INKLING_LEN").map_or(2600, |v| v.parse().expect("INKLING_LEN: a number"));
    let j: serde_json::Value = serde_json::from_slice(&std::fs::read(&ids_file).unwrap()).unwrap();
    let all: Vec<u32> = j["ids"].as_array().expect("an ids array").iter().map(|v| v.as_u64().unwrap() as u32).collect();
    assert!(all.len() >= len && len > B + 2, "{} ids, {len} wanted, more than {}", all.len(), B + 2);
    let ids = &all[..len];

    let t = Instant::now();
    let (m, lines) = llvq_llm::inkling::experts_cpu::load_served(
        &dir.join("config.json"),
        &dir.join("model.safetensors"),
        &dir.join("layers"),
        &dir.join("tables"),
    )
    .unwrap();
    let load_s = t.elapsed().as_secs_f64();
    for l in &lines {
        println!("{l}");
    }
    assert_eq!(m.substituted_layers().len(), m.cfg.num_hidden_layers, "every layer on the CPU experts");

    // The straight read.
    let t = Instant::now();
    let mut c = m.new_cache();
    let last = row(&m.forward_pieces(ids, &mut c, false).unwrap());
    let straight_prompt_s = t.elapsed().as_secs_f64();
    let (want, want_toks) = steps(&m, &mut c, last);
    drop(c);
    let want_sha = write(&out.join("straight.logits.f32"), &want);
    println!("straight: prompt {len} read in {straight_prompt_s:.1} s, tokens {want_toks:?}, sha256 {want_sha}");

    // The cut read: 0..B, the state to a file, a new cache from the file, the rest.
    let t = Instant::now();
    let mut first = m.new_cache();
    m.forward_pieces(&ids[..B], &mut first, false).unwrap();
    let to_b_s = t.elapsed().as_secs_f64();
    let state = out.join("cut-at-2048.state");
    let t = Instant::now();
    {
        let mut w = std::io::BufWriter::new(std::fs::File::create(&state).unwrap());
        first.write_state(&mut w).unwrap();
        std::io::Write::flush(&mut w).unwrap();
    }
    let save_s = t.elapsed().as_secs_f64();
    let state_bytes = std::fs::metadata(&state).unwrap().len();
    drop(first);
    let t = Instant::now();
    let mut loaded = m.read_cache(&mut std::io::BufReader::new(std::fs::File::open(&state).unwrap())).unwrap();
    let load_state_s = t.elapsed().as_secs_f64();
    assert_eq!((loaded.len(), loaded.tokens()), (B, &ids[..B]));
    let t = Instant::now();
    let last = row(&m.forward_pieces(&ids[B..], &mut loaded, false).unwrap());
    let rest_s = t.elapsed().as_secs_f64();
    let (got, got_toks) = steps(&m, &mut loaded, last);
    let got_sha = write(&out.join("cut-at-2048.logits.f32"), &got);
    std::fs::remove_file(&state).unwrap();
    println!(
        "cut at {B}: 0..{B} in {to_b_s:.1} s, state {state_bytes} bytes written in {save_s:.2} s, loaded in {load_state_s:.2} s, \
         the rest in {rest_s:.1} s, tokens {got_toks:?}, sha256 {got_sha}"
    );

    let report = serde_json::json!({
        "prompt_tokens": len, "b": B, "steps": STEPS, "vocab": want[0].len(),
        "ids_file": ids_file.display().to_string(), "model_dir": dir.display().to_string(),
        "load_s": load_s, "load_lines": lines,
        "straight": {"prompt_s": straight_prompt_s, "tokens": want_toks, "logits_sha256": want_sha},
        "cut": {"to_b_s": to_b_s, "state_bytes": state_bytes, "save_s": save_s, "load_s": load_state_s,
                "rest_s": rest_s, "tokens": got_toks, "logits_sha256": got_sha},
        "equal": want_sha == got_sha,
        "threads": std::env::var("LLVQ_THREADS").ok(),
    });
    std::fs::write(out.join("b-ii.json"), serde_json::to_string_pretty(&report).unwrap()).unwrap();
    assert_eq!(got_sha, want_sha, "the logits after a loaded state differ from the straight read's");
}
