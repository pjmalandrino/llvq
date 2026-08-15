//! **V0 of P1 — exactness, and not one nanosecond.**
//!
//! `proofs/preregistration-p1-2026-08-13.md` §3: *aucune milliseconde n'est
//! chronométrée avant que le décodeur soit prouvé*. This binary is that proof
//! for the two new arms, on real blocks of the sealed 4B. It prints no timing,
//! no throughput and no derived tok/s — deliberately. A number of that kind
//! here would be read as a P1 result, and P1 has not run.
//!
//! ## The two arms do not have the same standard, and confusing them is the
//! ## trap this file exists to avoid
//!
//! | arm | etalon |
//! |---|---|
//! | `cascade_uniform` | the dot product of `FastDecoder::decode`'s point, in f64 — it decodes the **archive's** order, so equality is the requirement |
//! | `decode_walk` | the dot product `binomial_walk` (the CPU reference) gives **on the same ranks** — it decodes **its own** combinatorial order |
//!
//! Amendment É2 of the pre-registration settles the second: relating a
//! binomial walk's order to the archive's multiset-permutation order *is* the
//! CNS re-bijection, which is P5, which this bench gates. Checking the walk
//! against `fd.decode` would demand the transcoder its own verdict authorises
//! — a circularity, and a V0 it could never pass. So the walk arm is fed
//! ranks drawn uniformly inside the **real class of each real block**, and the
//! GPU is required to agree with the Rust on those ranks. It is a round trip
//! across the CPU/GPU boundary on one bijection, not a comparison to the
//! archive.
//!
//! ## What this run does NOT cover, and must not be read as covering
//!
//! A real draw is bounded by the file, not by the codebook: the whole
//! published 4B touches 286 of the table's 384 entries — **no origin block**
//! and no shell-13 class — and a prefix draw of a few million blocks touches
//! fewer still. The count of untouched entries is printed for the draw that
//! actually ran, and this binary says nothing about them. The synthetic
//! fixture that would (pre-registration §1.6) is a separate obligation and it
//! is **ABSENT** here.
//!
//! Run: `cargo run --release -p llvq-metal --bin p1v0 [N] [model.llvq]`

use llvq_core::{Golay, Leech, SplitMix64, DIM};
use llvq_metal::p1host::{
    binom_table, cascade_ends, cascade_records, div_table, pack_walk_block, walk_radix_table,
    walk_records, GpuWalkRec,
};
use llvq_search::fastdec::{FastDecoder, MAX_KINDS};
use llvq_search::rankdec::binomial_walk;
use std::collections::BTreeSet;
use std::fs::File;
use std::io::BufReader;

/// Blocks drawn by default. V0 is an exactness gate, not a measurement: the
/// pre-registration's `2^24` is a floor on a *timed* arm (§1.2), and it buys
/// nothing here. Overridable in argv.
const N_DEFAULT: usize = 1 << 20;

/// Threads per threadgroup. Both shaders stage the 24 activations with
/// `if (tid < DIM)` and a barrier, so a final threadgroup narrower than 24
/// would read an unwritten `xs`. The draw is truncated to a multiple of this.
const GROUP: usize = 256;

/// Deterministic feed for the walk arm — printed, so the run is replayable.
const SEED: u64 = 0x0000_B1A0_5EED;

/// Gain centroids. **Not the artifact's**: the published matrices each carry
/// their own pair, and this bench mixes 252 of them. The gain scale is a
/// scalar applied *after* the decode, identical on both sides of every
/// comparison, so a single pair checks exactly what V0 is about — the decode —
/// and claims nothing about the centroids on disk. Same constant as
/// `decreal.rs:102`, for comparability.
const GSCALE: [f32; 2] = [0.9, 1.1];

/// Tolerance shape of `decreal.rs:129` — a floor OR a relative term, whichever
/// is larger — with the floor made **block-relative** rather than the absolute
/// `2e-3`. `cascade_uniform.metal` §4 asks for exactly that, and gives the
/// reason: an absolute floor that coarse cannot see a flipped sign bit on a
/// small dot, and "a fire alarm is not a guard".
const REL: f64 = 2e-3;

fn xvec() -> Vec<f32> {
    // Distinct magnitudes per slot, so a *permuted* arrangement moves the dot.
    // Same activation as `decreal`.
    (0..DIM).map(|i| 1.0 + i as f32 * 0.125).collect()
}

/// Worst deviation seen over an arm, and where.
struct Worst {
    abs: f64,
    rel: f64,
    at: usize,
    got: f64,
    want: f64,
    bad: usize,
}

/// Compare one arm's output to its own etalon. `want[b] = (expected, scale)`,
/// where `scale` is `Σ|wᵢxᵢ|` — the magnitude the accumulation actually ran at,
/// which is what a tolerance on a dot product has to be relative to.
fn verify(name: &str, got: &[f32], want: &[(f64, f64)]) -> Worst {
    assert_eq!(got.len(), want.len());
    // 🚨 A NaN is not a discrepancy to measure, it is a failed dispatch, and it
    // must kill the run rather than flatter it. Every comparison below is
    // written so NaN counts against the arm — but a kernel that returned NaN
    // *everywhere* would still slip through a purely relative reading, because
    // `f64::max` ignores NaN and the worst error would stay at zero: the arm
    // would print a BETTER line than a correct one. This assertion is what
    // makes that impossible.
    let n_bad = got.iter().filter(|v| !v.is_finite()).count();
    assert_eq!(
        n_bad, 0,
        "{name}: {n_bad} valeur(s) non finies rendues par le noyau — \
         ce n'est pas un écart, c'est un dispatch en échec"
    );
    let mut w = Worst {
        abs: 0.0,
        rel: 0.0,
        at: 0,
        got: 0.0,
        want: 0.0,
        bad: 0,
    };
    for (b, (&g, &(exp, mag))) in got.iter().zip(want).enumerate() {
        let d = (g as f64 - exp).abs();
        let tol = (REL * mag).max(REL * exp.abs());
        let rel = if mag > 0.0 { d / mag } else { d };
        if rel > w.rel {
            w.rel = rel;
            w.at = b;
            w.got = g as f64;
            w.want = exp;
        }
        w.abs = w.abs.max(d);
        // NaN is named, not implied. `d > tol` alone is FALSE when `d` is
        // NaN, so the block would escape the count entirely; writing the NaN
        // case out says so, and reads better than the `!(d <= tol)` that
        // clippy rightly calls hard to refactor.
        if d.is_nan() || d > tol {
            w.bad += 1;
        }
    }
    println!(
        "  {name:<18} {} blocs vérifiés — pire écart {:.3e} absolu, {:.3e} relatif à Σ|w·x| \
         (bloc {}: GPU {:.9} / CPU {:.9})",
        got.len(),
        w.abs,
        w.rel,
        w.at,
        w.got,
        w.want
    );
    if w.bad > 0 {
        println!(
            "  {:<18} ROUGE — {} blocs hors tolérance sur {}",
            "",
            w.bad,
            got.len()
        );
    }
    w
}

fn main() -> Result<(), String> {
    let mut n_req = N_DEFAULT;
    let mut path = format!(
        "{}/llvq-q4b.llvq",
        std::env::var("HOME").map_err(|_| "$HOME is not set, so there is no default path")?
    );
    for a in std::env::args().skip(1) {
        match a.parse::<usize>() {
            Ok(v) => n_req = v,
            Err(_) => path = a,
        }
    }

    // ---- the archive, or a loud failure -----------------------------------
    //
    // House rule: a test that skips when its file is missing must FAIL. P1's
    // whole reason to exist is the real class mix (pre-registration §1.5);
    // there is no substitute draw and no degraded mode.
    let f = File::open(&path).map_err(|e| {
        format!(
            "l'archive scellée du 4B n'est pas sur cette machine : {path} ({e})\n\n\
             P1 se mesure sur le mélange de classes RÉEL du modèle publié — il n'y a ni \
             substitut ni saut (pré-enregistrement §1.5).\n\
             La récupérer :\n\n    \
             hf download Pier-Jean/Qwen3-4B-LLVQ-2bit qwen3-4b-llvq.bin --local-dir .\n\n\
             puis la passer en argument :\n\n    \
             cargo run --release -p llvq-metal --bin p1v0 -- {n_req} ./qwen3-4b-llvq.bin\n\n\
             (cf. LAUNCH_ME.md pour le téléchargement, README.md pour l'invocation de \
             `bin/smoke` qui en produit une.)"
        )
    })?;

    let fd = FastDecoder::new();
    let golay = Golay::new();
    let mut r = BufReader::new(f);
    let h = llvq_artifact::read_header(&mut r).map_err(|e| e.to_string())?;
    let per_matrix = n_req.div_ceil(h.matrices as usize).max(1);

    let mut indices: Vec<u64> = Vec::with_capacity(n_req);
    let mut gains: Vec<u8> = Vec::with_capacity(n_req);
    for _ in 0..h.matrices {
        let m = llvq_artifact::read_matrix_raw(&mut r).map_err(|e| e.to_string())?;
        // Both shaders hardcode ONE gain bit, like every other MSL of this
        // repo. Asserted BEFORE any work: with two, every field after the gain
        // is shifted by one and the walk arm would decode the wrong classes
        // while looking perfectly healthy.
        assert_eq!(
            m.centroids.len().next_power_of_two().trailing_zeros(),
            1,
            "{}: les deux shaders de P1 codent en dur 1 bit de gain",
            m.name
        );
        let take = per_matrix.min(m.indices.len()).min(n_req - indices.len());
        indices.extend_from_slice(&m.indices[..take]);
        gains.extend(m.gains[..take].iter().map(|&g| {
            assert!(g < 2, "{}: rang de gain {g} avec 1 bit de gain", m.name);
            g as u8
        }));
    }

    // Truncate to a whole number of threadgroups: see GROUP.
    let n = indices.len() / GROUP * GROUP;
    assert!(
        n > 0,
        "{} blocs lus, moins d'un threadgroup de {GROUP}",
        indices.len()
    );
    indices.truncate(n);
    gains.truncate(n);

    println!("P1 — V0 : exactitude seule. Aucun chronométrage dans ce binaire.\n");
    println!(
        "{n} blocs réels ({} matrices, préfixes contigus) — {path}",
        h.matrices
    );

    // ---- what the draw covers, and what it cannot -------------------------
    let mut classes = Vec::with_capacity(n);
    for &idx in &indices {
        classes.push(
            fd.class_of(idx)
                .ok_or_else(|| format!("index {idx} hors de la boule"))?,
        );
    }
    let seen: BTreeSet<usize> = classes.iter().copied().collect();
    let origin = indices.iter().filter(|&&i| i == 0).count();
    println!(
        "  {} classes distinctes sur {} de la table, {origin} bloc(s) origine — \
         les entrées non touchées ne sont PAS couvertes par ce run",
        seen.len(),
        fd.n_classes()
    );
    println!("  graine du tirage de rangs (bras marche) : {SEED:#x}\n");

    // ---- host tables ------------------------------------------------------
    let ends = cascade_ends(&fd);
    let recs = cascade_records(&fd, &golay);
    let dv = div_table();
    let walk = walk_records(&fd);
    let radices = walk_radix_table(&walk);
    let binom = binom_table();
    println!(
        "tables hôtes : {} classes cascade (88 o), {} entrées marche (52 o), \
         DivTab 600 o, binomiaux {}×{}, Golay {} mots",
        recs.len(),
        walk.len(),
        DIM + 1,
        DIM + 1,
        golay.codewords().len()
    );

    let x = xvec();

    // =======================================================================
    // ARM 1 — cascade uniformisée, against FastDecoder::decode
    // =======================================================================
    let src_cascade = include_str!("../../shaders/cascade_uniform.metal");
    let kc = llvq_metal::Kernel::new(src_cascade, "cascade_uniform")?;
    println!("\nGPU : {}\n", kc.device_name());

    let b_idx = kc.buffer(&indices);
    let b_gain = kc.buffer(&gains);
    let b_ends = kc.buffer(&ends);
    let b_recs = kc.buffer(&recs);
    let b_golay = kc.buffer(golay.codewords());
    let b_dv = kc.buffer(std::slice::from_ref(&dv));
    let b_gs = kc.buffer(&GSCALE);
    let b_x = kc.buffer(&x);
    let b_out = kc.empty::<f32>(n);

    kc.dispatch(n as u64, GROUP as u64, |enc| {
        enc.set_buffer(0, Some(&b_idx), 0);
        enc.set_buffer(1, Some(&b_gain), 0);
        enc.set_buffer(2, Some(&b_ends), 0);
        enc.set_buffer(3, Some(&b_recs), 0);
        enc.set_buffer(4, Some(&b_golay), 0);
        enc.set_buffer(5, Some(&b_dv), 0);
        enc.set_buffer(6, Some(&b_gs), 0);
        enc.set_buffer(7, Some(&b_x), 0);
        enc.set_buffer(8, Some(&b_out), 0);
    });
    // SAFETY: the dispatch above completed and wrote n floats into b_out.
    let got_cascade: Vec<f32> = unsafe { kc.read(&b_out, n) };

    // The etalon: the archive decoder's own point, scaled in f64. `Leech::
    // shell_index` recomputes ‖p‖²/16 from the decoded point, so a wrong
    // `inv_norm` in the host table cannot cancel against itself here.
    let mut want_cascade = Vec::with_capacity(n);
    for b in 0..n {
        let p = fd
            .decode(indices[b])
            .ok_or_else(|| format!("index {} hors de la boule", indices[b]))?;
        let (mut want, mut mag) = (0.0f64, 0.0f64);
        if let Some(m) = Leech::shell_index(&p).filter(|&m| m > 0) {
            let s = GSCALE[gains[b] as usize] as f64 / ((16 * m) as f64).sqrt();
            for (&v, &xi) in p.iter().zip(&x) {
                let t = v as f64 * s * xi as f64;
                want += t;
                mag += t.abs();
            }
        }
        want_cascade.push((want, mag));
    }
    println!("bras CASCADE — étalon : produit scalaire f64 de FastDecoder::decode");
    let wc = verify("cascade_uniform", &got_cascade, &want_cascade);

    // =======================================================================
    // ARM 2 — marche binomiale, against binomial_walk on the SAME ranks
    // =======================================================================
    //
    // No transcoder exists (amendment É2), so the feed is: for the REAL class
    // of each real block, ranks drawn uniformly in `walk_radices`. The etalon
    // is the CPU walk on those very ranks — NOT `fd.decode`, which decodes
    // another order entirely.
    let mut rng = SplitMix64::new(SEED);
    let mut words: Vec<u32> = Vec::with_capacity(3 * n);
    let mut want_walk = Vec::with_capacity(n);
    let mut kinds = [0u8; DIM];
    for b in 0..n {
        let id = 1 + classes[b];
        let rec: &GpuWalkRec = &walk[id];
        let k = rec.k as usize;
        let mut ranks = [0u64; MAX_KINDS];
        for j in 0..k {
            let radix = radices[id][j];
            ranks[j] = if radix > 1 { rng.next() % radix } else { 0 };
        }
        let smask = (rng.next() as u32) & ((1 << DIM) - 1);
        words.extend_from_slice(&pack_walk_block(
            rec,
            id as u32,
            gains[b] as u32,
            smask,
            &ranks,
        ));

        binomial_walk(&ranks, &rec.counts, k, &mut kinds);
        let g = GSCALE[gains[b] as usize] as f64;
        let (mut want, mut mag) = (0.0f64, 0.0f64);
        for (i, &kd) in kinds.iter().enumerate() {
            let v = rec.vals[kd as usize] as f64;
            let v = if (smask >> i) & 1 == 1 { -v } else { v };
            let t = v * g * x[i] as f64;
            want += t;
            mag += t.abs();
        }
        want_walk.push((want, mag));
    }

    let src_walk = include_str!("../../shaders/binomial_walk.metal");
    let kw = llvq_metal::Kernel::new(src_walk, "decode_walk")?;
    let b_words = kw.buffer(&words);
    let b_tab = kw.buffer(&walk);
    let b_binom = kw.buffer(&binom);
    let b_gs2 = kw.buffer(&GSCALE);
    let b_x2 = kw.buffer(&x);
    let b_out2 = kw.empty::<f32>(n);

    kw.dispatch(n as u64, GROUP as u64, |enc| {
        enc.set_buffer(0, Some(&b_words), 0);
        enc.set_buffer(1, Some(&b_tab), 0);
        enc.set_buffer(2, Some(&b_binom), 0);
        enc.set_buffer(3, Some(&b_gs2), 0);
        enc.set_buffer(4, Some(&b_x2), 0);
        enc.set_buffer(5, Some(&b_out2), 0);
    });
    // SAFETY: the dispatch above completed and wrote n floats into b_out2.
    let got_walk: Vec<f32> = unsafe { kw.read(&b_out2, n) };

    println!(
        "\nbras MARCHE — étalon : binomial_walk (CPU) sur LES MÊMES rangs (É2), pas fd.decode"
    );
    let ww = verify("decode_walk", &got_walk, &want_walk);

    // ---- verdict ----------------------------------------------------------
    println!("\n{}", "-".repeat(78));
    let ok = wc.bad == 0 && ww.bad == 0;
    if ok {
        println!(
            "V0 VERT sur les deux bras : {n} blocs réels chacun, tolérance {REL:.0e}·Σ|w·x|.\n\
             Ce que ça autorise : rien d'autre que d'écrire le banc. Restent dus avant \
             tout chronométrage —\n  · la fixture synthétique (origine, coquille 13, \
             {} entrées de table que CE tirage n'atteint pas) ;\n  · le sweep intégral \
             des 150 681 600 blocs ;\n  · pour la marche, l'aller-retour rang → \
             arrangement → rang sur toutes les petites classes.",
            fd.n_classes() + 1 - seen.len()
        );
    } else {
        println!(
            "V0 ROUGE — cascade {} bloc(s) hors tolérance, marche {} bloc(s). \
             Aucun chronométrage n'est autorisé.",
            wc.bad, ww.bad
        );
    }
    if !ok {
        std::process::exit(1);
    }
    Ok(())
}
