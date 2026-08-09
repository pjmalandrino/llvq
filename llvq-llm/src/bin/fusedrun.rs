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

/// Decode steps of the extra, fenced pass under `LLVQ_TIME_PHASES=1`.
///
/// Not cfg-gated like the rest of this binary, on purpose: the call sites
/// only exist on linux+cuda, but a Mac cannot type-check gated code, and an
/// error here would otherwise surface in a paid image build. Compiling it
/// everywhere keeps `cargo check` on the dev machine load-bearing.
#[cfg_attr(not(all(target_os = "linux", feature = "cuda")), allow(dead_code))]
const PHASE_TOKENS: usize = 32;

/// The per-phase table of one arm's fenced pass. Additive diagnostics: this
/// runs *after* the arm's published measurement and prints below it, so the
/// published lines stay byte-identical.
#[cfg_attr(not(all(target_os = "linux", feature = "cuda")), allow(dead_code))]
fn print_phases(arm: &str, r: &llvq_llm::model::PhaseReport) {
    use llvq_llm::model::PhaseReport;
    println!("\n--- phases, bras {arm} (LLVQ_TIME_PHASES, {PHASE_TOKENS} tokens, hors protocole) ---");
    println!("  ⚠️ chaque phase est bornée par une synchronisation device : les phases");
    println!("     s'attribuent, mais les fences sérialisent ce que le chemin normal");
    println!("     recouvre — ce total ne remplace PAS le tok/s publié ci-dessus.");
    println!("  {:<15}{:>10}  plage (ms/token)", "phase", "médiane");
    let mut total = 0.0;
    for (name, samples) in [
        ("embed", &r.embed_ms),
        ("blocs+norme", &r.blocks_ms),
        ("lm_head", &r.head_ms),
        ("argmax+divers", &r.rest_ms),
    ] {
        let (med, lo, hi) = PhaseReport::stats(samples);
        total += med;
        println!("  {name:<15}{med:>10.3}  [{lo:.3}–{hi:.3}]  ({} éch.)", samples.len());
    }
    println!("  {:<15}{total:>10.3}  (somme des médianes, fences comprises)", "total fencé");
}

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

        // Whether to run the extra fenced pass after each arm's published
        // measurement. With the variable unset (or anything but "1") this
        // binary is byte-identical to the published protocol: the gate is the
        // only reference to it, and no fence exists outside the gated calls.
        let time_phases = llvq_llm::model::time_phases_enabled(
            std::env::var("LLVQ_TIME_PHASES").ok().as_deref(),
        );

        // ---- fused arm, first ----
        //
        // Deliberately before the dense arm: this is the path that can fail,
        // and it loads in ~145 s against the dense arm's ~209. Two runs died
        // here on 2026-08-05 — a refused prefill, then a length read off a
        // range — and each one had already paid for a dense load it never
        // used. Order is not neutral when a job is billed by the minute.
        let t = Instant::now();
        let f = llvq_llm::fused_cuda::load(&path, &device, dtype)?;
        let fused_load = t.elapsed().as_secs_f64();
        let ids = f
            .tokenizer
            .encode(PROMPT, false)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .get_ids()
            .to_vec();
        // The guard clause of the card gate. The threshold lives in
        // `rotplan::arms_are_discriminating`, where a test can reach it — this
        // whole block compiles on no developer machine, and a mutant that
        // weakened the bound to zero survived the entire suite while the
        // comparison was written out here.
        if !llvq_llm::rotplan::arms_are_discriminating(ids.len(), n_new) {
            anyhow::bail!(
                "prompt de {} token(s), {n_new} nouveau(x) : les deux bras LLVQ_ROT_SHARE \
                 parcourent alors le même chemin, et la comparaison de tokens imprimerait \
                 un vert qui ne dit rien.",
                ids.len()
            );
        }
        let rot_share = f.rot_share;
        let rot_launches = f.rot_launches;
        f.model.generate(&ids, n_new, &mut NoCapture)?;
        let t = Instant::now();
        let fused_tokens = f.model.generate(&ids, n_new, &mut NoCapture)?;
        let fused_rate = n_new as f64 / t.elapsed().as_secs_f64();
        // `carried_bytes`, not `carried_weights * 2`: under LLVQ_EMBED=q8 the
        // embedding sits on the card as int8 + f16 scales, and the old
        // identity would over-report by ~365 MB.
        let fused_bytes = f.runtime_bytes + f.carried_bytes;
        let (fused_rt_bits, fused_file) = (
            f.runtime_bytes as f64 * 8.0 / f.quantized_weights as f64,
            f.file_bytes as f64 / 1e9,
        );
        println!(
            "fusé   : chargé en {fused_load:6.1} s, {fused_rate:6.1} tok/s, {:.2} Go sur la carte",
            fused_bytes as f64 / 1e9
        );
        // On the arm line, not only in the loader's log: an A/B whose two runs
        // are told apart by scrolling back to a load-time message is an A/B
        // waiting to be misread.
        println!(
            "         LLVQ_ROT_SHARE={}, {rot_launches} rot_lancements/token",
            rot_share.name()
        );
        let tok = f.tokenizer.clone();
        if time_phases {
            let (_, report) = f.model.generate_phased(&ids, PHASE_TOKENS)?;
            print_phases("fusé", &report);
        }
        // Released before the dense arm loads: the point is to report what each
        // one costs, not to prove a card can hold both.
        drop(f);

        // ---- dense arm ----
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
            // selection, allocator growth and clock ramp.
            m.model.generate(&ids, n_new, &mut NoCapture)?;
            let t = Instant::now();
            let out = m.model.generate(&ids, n_new, &mut NoCapture)?;
            let rate = n_new as f64 / t.elapsed().as_secs_f64();
            let bytes = (m.quantized_weights + m.carried_weights) as u64 * 2;
            println!("dense  : chargé en {load:6.1} s, {rate:6.1} tok/s, {:.2} Go sur la carte",
                     bytes as f64 / 1e9);
            if time_phases {
                let (_, report) = m.model.generate_phased(&ids, PHASE_TOKENS)?;
                print_phases("dense", &report);
            }
            (out, load, rate, bytes)
        };

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
        println!("  dense : {}", tok.decode(&dense_tokens, true).unwrap_or_default());
        println!("  fusé  : {}", tok.decode(&fused_tokens, true).unwrap_or_default());

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
            fused_rt_bits,
            fused_file
        );
        Ok(())
    }
}
