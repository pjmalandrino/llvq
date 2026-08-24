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

/// The measurement protocol, aligned on the kernel benches (2026-08-18):
/// one discarded generation per arm (kernel selection, allocator growth,
/// clock ramp), then `ROUNDS_TIMED` timed ones, reported as median + min–max
/// range. Until then every published tok/s was a single point — in
/// contradiction with rule 2 of CLAUDE.md §7, which the benches obeyed and
/// this binary did not.
///
/// Unlike `planesbench`, the two arms can NOT interleave round by round:
/// each arm loads its model exclusively (the card does not have to hold
/// both), so their rounds never coexist and the speed ratio is a **quotient
/// of two medians**, not a median of per-round ratios. The summary labels it
/// as such and prints the conservative envelope
/// `[fused_lo/dense_hi ; fused_hi/dense_lo]` beside it.
#[cfg_attr(not(all(target_os = "linux", feature = "cuda")), allow(dead_code))]
const ROUNDS_TIMED: usize = 5;

/// Median and min–max of per-round rates. Not cfg-gated, same reason as
/// [`PHASE_TOKENS`]: the Mac must type-check what a paid image build would
/// otherwise be the first to compile.
#[cfg_attr(not(all(target_os = "linux", feature = "cuda")), allow(dead_code))]
fn rate_stats(rounds: &[f64]) -> (f64, f64, f64) {
    assert!(!rounds.is_empty(), "rate_stats over zero rounds");
    let mut sorted = rounds.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let med = if sorted.len() % 2 == 1 {
        sorted[sorted.len() / 2]
    } else {
        (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
    };
    (med, sorted[0], sorted[sorted.len() - 1])
}

/// One fused arm's published numbers, so two of them can be run in one process
/// and compared side by side.
///
/// Not cfg-gated, same reason as [`PHASE_TOKENS`]: a Mac cannot type-check
/// gated code, and this struct's shape is exactly what an image build would
/// otherwise be the first to compile.
#[cfg_attr(not(all(target_os = "linux", feature = "cuda")), allow(dead_code))]
struct Arm {
    fuse: llvq_llm::fused::FuseMode,
    tokens: Vec<u32>,
    load_s: f64,
    rate: f64,
    lo: f64,
    hi: f64,
    bytes: u64,
    rot_launches: usize,
    matvec_launches: usize,
}

/// Position of the first token the two arms disagree on, or `None` when they
/// agree everywhere.
///
/// Refuses arms of different lengths instead of comparing their common prefix.
/// `zip` truncates, so an arm that returned fewer tokens would have printed
/// "N tokens identiques" over the prefix and said nothing about the rest — a
/// green that means nothing, which is the failure mode this binary's own gate
/// clause exists to prevent. Out here rather than inline for the reason
/// `rotplan::arms_are_discriminating` is out of this file: a comparison written
/// inside the cfg-gated body is checked by no machine in this workspace.
#[cfg_attr(not(all(target_os = "linux", feature = "cuda")), allow(dead_code))]
fn first_divergence(a: &[u32], b: &[u32]) -> Result<Option<usize>, String> {
    if a.len() != b.len() {
        return Err(format!(
            "{} tokens contre {} : les deux bras n'ont pas décodé la même longueur, et \
             comparer leur préfixe commun imprimerait un vert qui ne dit rien",
            a.len(),
            b.len()
        ));
    }
    Ok(a.iter().zip(b).position(|(x, y)| x != y))
}

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

        // Which fused arms to run. Normally one, named by `LLVQ_FUSE`; under
        // `LLVQ_FUSE_AB=1` both, `On` then `Off`, each loaded and **dropped**
        // before the next — the card never holds two arms, and the two share a
        // process, an NVRTC translation unit, a card and a prompt. The only
        // thing that moves between them is the number of launches, which is
        // what makes the delta attributable. It costs one extra transcode
        // (~145 s at the 4B).
        let fuse_ab = std::env::var("LLVQ_FUSE_AB").ok().as_deref() == Some("1");
        let modes: Vec<llvq_llm::fused::FuseMode> = if fuse_ab {
            vec![llvq_llm::fused::FuseMode::On, llvq_llm::fused::FuseMode::Off]
        } else {
            vec![llvq_llm::fused::FuseMode::from_env().map_err(|e| anyhow::anyhow!("{e}"))?]
        };

        // ---- fused arms, first ----
        //
        // Deliberately before the dense arm: this is the path that can fail,
        // and it loads in ~145 s against the dense arm's ~209. Two runs died
        // here on 2026-08-05 — a refused prefill, then a length read off a
        // range — and each one had already paid for a dense load it never
        // used. Order is not neutral when a job is billed by the minute.
        let mut arms: Vec<Arm> = Vec::with_capacity(modes.len());
        let mut tok: Option<tokenizers::Tokenizer> = None;
        let mut fused_rt_bits = 0.0f64;
        let mut fused_file = 0.0f64;
        for fuse in modes {
            let t = Instant::now();
            let f = llvq_llm::fused_cuda::load_with(&path, &device, dtype, fuse)?;
            let load_s = t.elapsed().as_secs_f64();
            let ids = f
                .tokenizer
                .encode(PROMPT, false)
                .map_err(|e| anyhow::anyhow!("{e}"))?
                .get_ids()
                .to_vec();
            // The guard clauses of the card gate. Both thresholds live in
            // `rotplan`, where a test can reach them — this whole block
            // compiles on no developer machine, and a mutant that weakened the
            // rotation bound to zero survived the entire suite while the
            // comparison was written out here.
            if !llvq_llm::rotplan::arms_are_discriminating(ids.len(), n_new) {
                anyhow::bail!(
                    "prompt de {} token(s), {n_new} nouveau(x) : les deux bras \
                     LLVQ_ROT_SHARE parcourent alors le même chemin, et la comparaison de \
                     tokens imprimerait un vert qui ne dit rien.",
                    ids.len()
                );
            }
            if !llvq_llm::rotplan::fuse_arms_are_discriminating(ids.len(), n_new) {
                anyhow::bail!(
                    "prompt de {} token(s), {n_new} nouveau(x) : trop court pour séparer les \
                     deux bras LLVQ_FUSE du bruit de chargement et de montée en fréquence.",
                    ids.len()
                );
            }
            f.model.generate(&ids, n_new, &mut NoCapture)?;
            let mut rounds = Vec::with_capacity(ROUNDS_TIMED);
            let mut tokens: Vec<u32> = Vec::new();
            for round in 0..ROUNDS_TIMED {
                let t = Instant::now();
                let out = f.model.generate(&ids, n_new, &mut NoCapture)?;
                rounds.push(n_new as f64 / t.elapsed().as_secs_f64());
                if round == 0 {
                    tokens = out;
                } else if out != tokens {
                    // Greedy decode on fixed weights should be deterministic;
                    // a flip across rounds is a finding, not a nuisance.
                    println!(
                        "  ⚠️ bras fusé LLVQ_FUSE={} : les tokens du round {round} diffèrent \
                         du round 0 — décodage non déterministe, à investiguer avant de \
                         publier",
                        f.fuse.name()
                    );
                }
            }
            let (rate, lo, hi) = rate_stats(&rounds);
            // `carried_bytes`, not `carried_weights * 2`: under LLVQ_EMBED=q8
            // the embedding sits on the card as int8 + f16 scales, and the old
            // identity would over-report by ~365 MB.
            let bytes = f.runtime_bytes + f.carried_bytes;
            // The trailing "projections … b/poids" line reports the FIRST arm,
            // which is the fused one under `LLVQ_FUSE_AB=1`. Letting the loop
            // overwrite it would print the control arm's accounting under the
            // fused arm's heading — the two differ by `gs_off`, 3.69 MB on the
            // 4B, and that is exactly the term this lot adds.
            if arms.is_empty() {
                fused_rt_bits = f.runtime_bytes as f64 * 8.0 / f.quantized_weights as f64;
                fused_file = f.file_bytes as f64 / 1e9;
            }
            println!(
                "fusé   : chargé en {load_s:6.1} s, {rate:6.1} tok/s \
                 [{lo:.1}–{hi:.1}, {ROUNDS_TIMED} rounds], {:.2} Go sur la carte",
                bytes as f64 / 1e9
            );
            // On the arm line, not only in the loader's log: an A/B whose two
            // runs are told apart by scrolling back to a load-time message is
            // an A/B waiting to be misread. The matvec count is here for the
            // same reason the rotation count is — a gate reading "128 tokens
            // identiques" while both arms issued 252 matvecs proves the tokens
            // and nothing about this lot.
            println!(
                "         LLVQ_ROT_SHARE={}, {} rot_lancements/token · LLVQ_FUSE={}, {} \
                 matvec_lancements/token",
                f.rot_share.name(),
                f.rot_launches,
                f.fuse.name(),
                f.matvec_launches
            );
            if tok.is_none() {
                tok = Some(f.tokenizer.clone());
            }
            if time_phases {
                let (_, report) = f.model.generate_phased(&ids, PHASE_TOKENS)?;
                print_phases(&format!("fusé F{}", f.fuse.name()), &report);
            }
            arms.push(Arm {
                fuse: f.fuse,
                tokens,
                load_s,
                rate,
                lo,
                hi,
                bytes,
                rot_launches: f.rot_launches,
                matvec_launches: f.matvec_launches,
            });
            // Released before the next arm — and before the dense one — loads:
            // the point is to report what each one costs, not to prove a card
            // can hold two.
            drop(f);
        }
        let tok = tok.expect("at least one fused arm ran");

        // ---- dense arm ----
        let (dense_tokens, dense_load, dense_rate, dense_lo, dense_hi, dense_bytes) = {
            let t = Instant::now();
            let m = llvq_llm::sealed::load(&path, dtype, &device, llvq_llm::kvq::KvMode::F16)?;
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
            let mut rounds = Vec::with_capacity(ROUNDS_TIMED);
            let mut out: Vec<u32> = Vec::new();
            for round in 0..ROUNDS_TIMED {
                let t = Instant::now();
                let o = m.model.generate(&ids, n_new, &mut NoCapture)?;
                rounds.push(n_new as f64 / t.elapsed().as_secs_f64());
                if round == 0 {
                    out = o;
                } else if o != out {
                    println!(
                        "  ⚠️ bras dense : les tokens du round {round} diffèrent du round 0 — \
                         décodage non déterministe, à investiguer avant de publier"
                    );
                }
            }
            let (rate, lo, hi) = rate_stats(&rounds);
            let bytes = (m.quantized_weights + m.carried_weights) as u64 * 2;
            println!(
                "dense  : chargé en {load:6.1} s, {rate:6.1} tok/s \
                 [{lo:.1}–{hi:.1}, {ROUNDS_TIMED} rounds], {:.2} Go sur la carte",
                bytes as f64 / 1e9
            );
            if time_phases {
                let (_, report) = m.model.generate_phased(&ids, PHASE_TOKENS)?;
                print_phases("dense", &report);
            }
            (out, load, rate, lo, hi, bytes)
        };

        // ---- the comparison ----
        println!("\n--- les deux bras ---");
        for a in &arms {
            let first = first_divergence(&dense_tokens, &a.tokens)
                .map_err(|e| anyhow::anyhow!("dense contre fusé F{} : {e}", a.fuse.name()))?;
            match first {
                None => println!(
                    "  fusé F{} : {} tokens identiques au dense",
                    a.fuse.name(),
                    dense_tokens.len()
                ),
                Some(i) => {
                    println!(
                        "  ⚠️ fusé F{} : divergence au token {i} sur {}",
                        a.fuse.name(),
                        dense_tokens.len()
                    );
                    println!("     dense {:?}", &dense_tokens[i.saturating_sub(2)..]);
                    println!("     fusé  {:?}", &a.tokens[i.saturating_sub(2)..]);
                    println!(
                        "     un décodage glouton est une chaîne d'argmax : un token \
                         retourné\n     change tous les suivants. Seule la POSITION du \
                         premier écart informe."
                    );
                }
            }
        }
        // 🚨 The criterion of this lot, and it is the POSITION of the first
        // divergence rather than a count: a greedy decode is a chain of
        // argmaxes, so one flipped token changes every token after it. What the
        // 2026-08-06 journal establishes is that `Planes14` and `slot32`
        // diverge from the dense arm AT THE SAME TOKEN (89 of 128) — that
        // control is what licenses reading 89 as a tie-break rather than a bug.
        // Transposed here: the two `LLVQ_FUSE` arms must diverge from the dense
        // arm at the same position, and — strictly stronger — agree with each
        // other everywhere.
        if let [a, b] = &arms[..] {
            match first_divergence(&a.tokens, &b.tokens)
                .map_err(|e| anyhow::anyhow!("fusé F{} contre F{} : {e}", a.fuse.name(), b.fuse.name()))?
            {
                None => println!(
                    "  ✅ fusé F{} et F{} : {} tokens identiques entre eux",
                    a.fuse.name(),
                    b.fuse.name(),
                    a.tokens.len()
                ),
                Some(i) => println!(
                    "  🚨 fusé F{} et F{} divergent au token {i} — c'est la fusion elle-même \
                     qui a changé l'arithmétique, pas un tie-break",
                    a.fuse.name(),
                    b.fuse.name()
                ),
            }
        }
        println!("  dense : {}", tok.decode(&dense_tokens, true).unwrap_or_default());
        for a in &arms {
            println!(
                "  fusé F{} : {}",
                a.fuse.name(),
                tok.decode(&a.tokens, true).unwrap_or_default()
            );
        }

        println!("\n--- ce que ça coûte ---");
        println!(
            "  {:<10}{:>12}{:>16}{:>18}{:>12}{:>10}",
            "bras", "chargement", "tok/s (médiane)", "plage", "Go carte", "matvec/t"
        );
        println!("  {}", "-".repeat(78));
        println!(
            "  {:<10}{dense_load:>11.1} s{dense_rate:>16.1}{:>18}{:>12.2}{:>10}",
            "dense",
            format!("[{dense_lo:.1}–{dense_hi:.1}]"),
            dense_bytes as f64 / 1e9,
            "—"
        );
        for a in &arms {
            println!(
                "  {:<10}{:>11.1} s{:>16.1}{:>18}{:>12.2}{:>10}",
                format!("fusé F{}", a.fuse.name()),
                a.load_s,
                a.rate,
                format!("[{:.1}–{:.1}]", a.lo, a.hi),
                a.bytes as f64 / 1e9,
                a.matvec_launches
            );
        }
        println!("  {}", "-".repeat(78));
        // A quotient of two medians, NOT a median of per-round ratios: the
        // two arms' rounds never coexist (each load is exclusive), so no
        // per-round pairing exists. The envelope divides the extremes the
        // conservative way round; read the second decimal as dispersion.
        for a in &arms {
            println!(
                "  fusé F{} : vitesse ×{:.2} [×{:.2}–×{:.2}] contre le dense (quotient des \
                 médianes, {ROUNDS_TIMED} rounds par bras jamais entrelacés), mémoire ÷{:.2}",
                a.fuse.name(),
                a.rate / dense_rate,
                a.lo / dense_hi,
                a.hi / dense_lo,
                dense_bytes as f64 / a.bytes as f64
            );
        }
        // 🚨 The number this lot is about, and the only comparison in this
        // output whose two sides share a translation unit and a card: fused
        // against unfused, both of them ours. The 11.7 % the bench measured on
        // `tv_planes_seg` (5.096 → 4.504 ms) does **not** transport here — it is
        // f32, out of model, on matvecs alone, where this is f16 end to end and
        // the matvecs are one share of a token's time.
        if let [a, b] = &arms[..] {
            println!(
                "  fusion : ×{:.3} [×{:.3}–×{:.3}] de F{} sur F{} ({} contre {} \
                 matvec_lancements/token), mémoire {:+} octets",
                a.rate / b.rate,
                a.lo / b.hi,
                a.hi / b.lo,
                a.fuse.name(),
                b.fuse.name(),
                a.matvec_launches,
                b.matvec_launches,
                a.bytes as i64 - b.bytes as i64
            );
            if a.rot_launches != b.rot_launches {
                println!(
                    "  ⚠️ {} rot_lancements/token contre {} : les deux bras ne diffèrent PAS \
                     que par la fusion, le rapport ci-dessus mêle deux mécanismes",
                    a.rot_launches, b.rot_launches
                );
            }
        }
        println!(
            "\n  projections : {:.3} b/poids sur la carte, fichier {:.2} Go sur disque.\n  \
             ⚠️ ces deux comptabilités ne se comparent pas : le fichier porte des index\n  \
             compacts, la carte lit la disposition dépliée que le noyau peut lire vite.",
            fused_rt_bits,
            fused_file
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{first_divergence, rate_stats};

    #[test]
    fn median_is_the_middle_rate_odd_and_even() {
        let (m, lo, hi) = rate_stats(&[3.0, 1.0, 2.0]);
        assert_eq!((m, lo, hi), (2.0, 1.0, 3.0));
        let (m, lo, hi) = rate_stats(&[4.0, 1.0, 2.0, 3.0]);
        assert_eq!((m, lo, hi), (2.5, 1.0, 4.0));
    }

    /// The comparison refuses arms of different lengths instead of walking
    /// their common prefix and printing a tick.
    ///
    /// The hole this closes is real and predates the lot: `zip` truncates, so
    /// an arm that returned fewer tokens compared green over the prefix. The
    /// assertion is on the *refusal*, because the truncating version passes
    /// every "identical tokens" test there is.
    #[test]
    fn a_short_arm_is_refused_rather_than_compared_on_its_prefix() {
        assert_eq!(first_divergence(&[1, 2, 3], &[1, 2, 3]), Ok(None));
        assert_eq!(first_divergence(&[1, 2, 3], &[1, 9, 3]), Ok(Some(1)));
        assert_eq!(first_divergence(&[9, 2], &[1, 2]), Ok(Some(0)));
        let e = first_divergence(&[1, 2, 3], &[1, 2]).expect_err("longueurs inégales");
        assert!(e.contains('3') && e.contains('2'), "{e}");
        assert_eq!(first_divergence(&[], &[]), Ok(None));
    }
}
