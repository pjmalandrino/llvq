//! Load a sealed `.llvq` model and generate from it — no checkpoint, no
//! network.
//!
//! Set `LLVQ_VERIFY_CACHE=1` to run each prompt a second time through the
//! pre-cache quadratic path and require the **same tokens**. That is the KV
//! cache's only real gate: a cache bug produces fluent, plausible, different
//! text, which no threshold can catch.
//!
//! Usage:
//!   `cargo run --release -p llvq-llm --features metal --bin run -- model.llvq [device] [n_new]`
//!
//! This is the deliverable's own test. Every other number in this project is a
//! claim about what a file *would* weigh; this one opens the file, rebuilds a
//! model out of it, and makes it answer. If it needs anything else on disk, it
//! is not a deployable model and the number was decoration.
//!
//! A projections-only file (format v1) is refused with the command that fixes
//! it, rather than silently reaching for the checkpoint. That refusal lives in
//! [`llvq_llm::sealed`], shared with `bin/mmlu` and `bin/ppl`.

use candle_core::DType;
use llvq_llm::model::NoCapture;

const PROMPTS: &[&str] = &[
    "The capital of France is",
    "In 1969, the first humans landed on",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return",
    "Water boils at a temperature of",
];

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let path = a
        .first()
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("give the path to a .llvq model"))?;
    let device = llvq_llm::eval::device(a.get(1).map(String::as_str).unwrap_or("cpu"))?;
    let n_new: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(24);
    // f16 rather than f32: the model ran at half precision anyway, and the
    // difference is 7 GB of RAM against 15 on a 4B. `LLVQ_DTYPE=f32` puts the
    // generation back on the exact weights `verify_artifact` round-trips —
    // the narrowing here is outside that proof.
    let dtype = llvq_llm::eval::dtype(DType::F16)?;
    // Resolved once, here: `model.rs` reads no environment variable, so the
    // mode travels by value from this line down to every `KvCache`. An unknown
    // name is an error, never a silent fallback — a typo would make an A/B lie.
    let kv_mode = llvq_llm::kvq::KvMode::from_env().map_err(anyhow::Error::msg)?;

    let s = llvq_llm::sealed::load(&path, dtype, &device, kv_mode)?;
    let fp16 = (s.quantized_weights + s.carried_weights) * 2;
    eprintln!(
        "loaded {path} — {} quantized matrices + {} carried tensors\n",
        s.matrices, s.raw_tensors
    );
    println!("── model");
    println!(
        "   running at dtype {}, kv {}",
        llvq_llm::eval::dtype_name(dtype),
        kv_mode.name()
    );
    println!(
        "   {:.3} GB on disk against {:.3} GB in FP16  →  ×{:.2}",
        s.bytes as f64 / 1e9,
        fp16 as f64 / 1e9,
        fp16 as f64 / s.bytes as f64
    );
    println!(
        "   {} weights quantized, {} carried at f16\n",
        s.quantized_weights, s.carried_weights
    );

    for p in PROMPTS {
        let ids = s
            .tokenizer
            .encode(*p, false)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .get_ids()
            .to_vec();
        let t = std::time::Instant::now();
        let out = s.model.generate(&ids, n_new, &mut NoCapture)?;
        let secs = t.elapsed().as_secs_f64();
        let text = s
            .tokenizer
            .decode(&out, false)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        println!(
            "── {p:?}\n   →{text}\n   {n_new} tokens en {secs:.2} s — {:.1} tok/s",
            n_new as f64 / secs
        );

        // The cache's only honest gate.
        //
        // A KV-cache bug does not crash and does not overflow a threshold: it
        // shifts a RoPE position or drops a mask entry, and the model goes on
        // producing fluent, plausible, *different* text. The pre-cache path is
        // kept for exactly this — an independent answer to "did the cache
        // change what the model says". Greedy decoding is deterministic, so
        // the two token sequences must be equal, not close.
        if std::env::var("LLVQ_VERIFY_CACHE").is_ok() {
            // Below three tokens the cache is never read: `generate` returns
            // right after the prefill, so `append` never sees a non-empty
            // cache and the mask never sees a nonzero offset. The check would
            // compare a code path with itself and print a green tick — the
            // exact motif CLAUDE.md §5 documents three times.
            anyhow::ensure!(
                n_new >= 3,
                "LLVQ_VERIFY_CACHE n'exerce rien sous 3 tokens (n_new = {n_new})"
            );
            let t = std::time::Instant::now();
            let want = s.model.generate_uncached(&ids, n_new, &mut NoCapture)?;
            let slow = t.elapsed().as_secs_f64();
            anyhow::ensure!(
                out == want,
                "le cache KV change ce que le modèle dit :\n  avec  {out:?}\n  sans  {want:?}"
            );
            println!(
                "   ✓ identique au chemin sans cache ({slow:.2} s, ×{:.1} plus lent)",
                slow / secs
            );
        }
    }
    Ok(())
}
