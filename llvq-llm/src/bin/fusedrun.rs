//! The fused path against the dense one: same file, same prompt, same tokens?
//!
//! Every check so far has been local. `fused_path_matches_dense.rs` pins one
//! matrix on the CPU; `rotation_matches_rust.rs` pins the rotation kernel;
//! `bin/matvec` pins `slot_dot` against an f64 reference. None of them can say
//! what happens when 252 of those compose across 36 layers, twice per token,
//! with a KV cache in between — which is the only question that decides
//! whether this path is usable.
//!
//! So this binary loads **the same artifact twice**:
//!
//!   * dense — `sealed::load`, the object every published perplexity refers to;
//!   * fused — `fused_cuda::load`, weights left encoded, two kernels per
//!     projection.
//!
//! and requires the same tokens out. Not the same logits: the two accumulate
//! in different orders, so bit-equality is not on offer and demanding it would
//! be theatre. Identical *tokens* is the property that matters and the one a
//! user would notice breaking.
//!
//! ## Reading the divergence, if there is one
//!
//! A greedy decode is a chain of argmaxes, so one flipped token changes every
//! token after it. The position of the **first** divergence is therefore the
//! only informative number: token 1 means the arithmetic is wrong, token 30
//! means two logits were within a rounding error of each other and the tie
//! broke the other way. The second is expected on a long enough run and is not
//! a defect; the first is.
//!
//! ## What it also reports, and why it is the real headline
//!
//! Device bytes. The speed difference is bounded by things this path does not
//! touch — roughly half a token's time is attention, norms and launch
//! overhead — but the memory difference is the whole thesis: 8.04 GB of f16
//! against the encoded projections plus a full-precision embedding.

#[cfg(all(target_os = "linux", feature = "cuda"))]
use std::time::Instant;
#[cfg(all(target_os = "linux", feature = "cuda"))]
use llvq_llm::model::NoCapture;

#[cfg(all(target_os = "linux", feature = "cuda"))]
const PROMPT: &str = "The capital of France is";

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .ok_or_else(|| anyhow::anyhow!("usage: fusedrun <model.llvq> [n_tokens]"))?;
    let n_new: usize = args.next().map_or(Ok(32), |s| s.parse())?;

    #[cfg(not(all(target_os = "linux", feature = "cuda")))]
    {
        let _ = (path, n_new);
        anyhow::bail!("fusedrun exige une carte NVIDIA et la feature `cuda`");
    }

    #[cfg(all(target_os = "linux", feature = "cuda"))]
    {
        use candle_core::{DType, Device};
        let device = Device::new_cuda(0)?;
        let dtype = DType::F16;
        println!("{device:?}, dtype {dtype:?}, {n_new} tokens\n");

        // ---- dense arm, then dropped ----
        //
        // Dropped before the fused arm loads on purpose: holding both would
        // need 8.04 GB plus the encoded copy, and the point of the exercise is
        // to report what each one costs, not to prove a card can hold both.
        let (dense_tokens, dense_load, dense_rate, dense_bytes) = {
            let t = Instant::now();
            let m = llvq_llm::sealed::load(&path, dtype, &device)?;
            let load = t.elapsed().as_secs_f64();
            let ids = m
                .tokenizer
                .encode(PROMPT, false)
                .map_err(|e| anyhow::anyhow!("{e}"))?
                .get_ids()
                .to_vec();
            // One discarded generation: the first on CUDA pays kernel
            // selection, allocator growth and clock ramp. Without it the
            // measurement is of the warm-up.
            m.model.generate(&ids, n_new, &mut NoCapture)?;
            let t = Instant::now();
            let out = m.model.generate(&ids, n_new, &mut NoCapture)?;
            let rate = n_new as f64 / t.elapsed().as_secs_f64();
            let bytes = (m.quantized_weights + m.carried_weights) as u64 * 2;
            println!("dense  : chargé en {load:6.1} s, {rate:6.1} tok/s, {:.2} Go sur la carte",
                     bytes as f64 / 1e9);
            (out, load, rate, bytes)
        };

        // ---- fused arm ----
        let t = Instant::now();
        let f = llvq_llm::fused_cuda::load(&path, &device, dtype)?;
        let fused_load = t.elapsed().as_secs_f64();
        let ids = f
            .tokenizer
            .encode(PROMPT, false)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .get_ids()
            .to_vec();
        f.model.generate(&ids, n_new, &mut NoCapture)?;
        let t = Instant::now();
        let fused_tokens = f.model.generate(&ids, n_new, &mut NoCapture)?;
        let fused_rate = n_new as f64 / t.elapsed().as_secs_f64();
        let fused_bytes = f.runtime_bytes + f.carried_weights as u64 * 2;
        println!(
            "fusé   : chargé en {fused_load:6.1} s, {fused_rate:6.1} tok/s, {:.2} Go sur la carte",
            fused_bytes as f64 / 1e9
        );

        // ---- the comparison ----
        println!("\n--- les deux bras ---");
        let first = dense_tokens
            .iter()
            .zip(&fused_tokens)
            .position(|(a, b)| a != b);
        match first {
            None => println!("  {} tokens identiques", dense_tokens.len()),
            Some(i) => {
                println!("  ⚠️ divergence au token {i} sur {}", dense_tokens.len());
                println!("     dense {:?}", &dense_tokens[i.saturating_sub(2)..]);
                println!("     fusé  {:?}", &fused_tokens[i.saturating_sub(2)..]);
                println!(
                    "     un décodage glouton est une chaîne d'argmax : un token retourné\n     \
                     change tous les suivants. Seule la POSITION du premier écart informe."
                );
            }
        }
        println!("  dense : {}", f.tokenizer.decode(&dense_tokens, true).unwrap_or_default());
        println!("  fusé  : {}", f.tokenizer.decode(&fused_tokens, true).unwrap_or_default());

        println!("\n--- ce que ça coûte ---");
        println!(
            "  {:<10}{:>12}{:>12}{:>12}",
            "bras", "chargement", "tok/s", "Go carte"
        );
        println!("  {}", "-".repeat(46));
        println!("  {:<10}{dense_load:>11.1} s{dense_rate:>12.1}{:>12.2}",
                 "dense", dense_bytes as f64 / 1e9);
        println!("  {:<10}{fused_load:>11.1} s{fused_rate:>12.1}{:>12.2}",
                 "fusé", fused_bytes as f64 / 1e9);
        println!("  {}", "-".repeat(46));
        println!(
            "  vitesse ×{:.2}, mémoire ÷{:.2}",
            fused_rate / dense_rate,
            dense_bytes as f64 / fused_bytes as f64
        );
        println!(
            "\n  projections : {:.3} b/poids sur la carte, fichier {:.2} Go sur disque.\n  \
             ⚠️ ces deux comptabilités ne se comparent pas : le fichier porte des index\n  \
             compacts, la carte lit la disposition Slot32 que le noyau peut lire vite.",
            f.runtime_bytes as f64 * 8.0 / f.quantized_weights as f64,
            f.file_bytes as f64 / 1e9
        );
        Ok(())
    }
}
