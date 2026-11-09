//! One long prefill on a dense checkpoint, and the bits it ends on: the
//! Mac-side check of the prefill's memory (docs/mesures/pic-vram-8k-2026-11-08.txt).
//!
//! ```text
//! /usr/bin/time -l cargo run --release -p llvq-llm --features metal --example prefill_peak -- \
//!     Qwen/Qwen3-4B 8192 metal grouped
//! ```
//!
//! The prompt repeats a fixed sentence to the length asked and goes through
//! `Qwen3::prefill_chunked` in chunks of `MAX_PREFILL_ROWS`, the served
//! prefill's driver, under the attention form named (`expanded` or
//! `grouped`). It prints the time, the first token and a fingerprint of the
//! last position's f32 logits, bit for bit: two builds of the code that print
//! the same fingerprint returned the same logits. The memory is read from
//! outside, by `/usr/bin/time -l` ("peak memory footprint"), which on Apple
//! silicon counts the Metal buffers the process holds.

use candle_core::DType;
use llvq_llm::kvq::{AttnHeads, KvMode};
use llvq_llm::loader::Checkpoint;
use llvq_llm::model::{Qwen3, MAX_PREFILL_ROWS};

const SEED: &str = "The Leech lattice is the unique even unimodular lattice in 24 dimensions with no vectors of \
                    norm two. Its automorphism group is the Conway group, and its kissing number is 196560. ";

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let repo = args.first().map(String::as_str).unwrap_or("Qwen/Qwen3-4B");
    let n: usize = args.get(1).map(|s| s.parse()).transpose()?.unwrap_or(8192);
    let device = llvq_llm::eval::device(args.get(2).map(String::as_str).unwrap_or("metal"))?;
    let form = match args.get(3).map(String::as_str).unwrap_or("grouped") {
        "expanded" => AttnHeads::Expanded,
        "grouped" => AttnHeads::Grouped,
        o => anyhow::bail!("form {o:?}: expanded or grouped"),
    };
    let ck = Checkpoint::fetch(repo)?;
    let vb = ck.var_builder(DType::F16, &device)?;
    let mut m = Qwen3::new(&ck.config, vb, KvMode::F16)?;
    m.set_attn_heads(form);
    m.set_kv_store(form.store());
    let seed = ck.tokenizer()?.encode(SEED, false).map_err(|e| anyhow::anyhow!("{e}"))?.get_ids().to_vec();
    let ids: Vec<u32> = (0..n).map(|i| seed[i % seed.len()]).collect();
    let mut caches = m.fresh_caches();
    let t = std::time::Instant::now();
    let logits = m.prefill_chunked(&ids, MAX_PREFILL_ROWS, &mut caches)?;
    let v = logits.flatten_all()?.to_vec1::<f32>()?;
    let secs = t.elapsed().as_secs_f64();
    let first = v.iter().enumerate().fold((0, f32::NEG_INFINITY), |a, (i, &x)| if x > a.1 { (i, x) } else { a }).0;
    let bits: Vec<u32> = v.iter().map(|x| x.to_bits()).collect();
    println!(
        "{repo} {n} tokens, {} on {:?}, chunks of {MAX_PREFILL_ROWS}: {secs:.1} s, first token {first}, \
         last-position logits fingerprint {:016x}",
        form.label(),
        device,
        llvq_llm::eval::token_fingerprint(&bits)
    );
    Ok(())
}
