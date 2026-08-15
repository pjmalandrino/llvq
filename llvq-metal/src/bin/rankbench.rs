//! **P1 — the five-arm bench.** What a rank decode costs on real blocks.
//!
//! Pre-registration: `proofs/preregistration-p1-2026-08-13.md`, whose §1 fixes
//! the accounting, §2 the five arms, §3 the exactness gate, §4 the thresholds
//! and §7 what would invalidate the whole thing. Nothing here is decided at
//! run time; this file is the pre-registration executed.
//!
//! ## The five arms, in the frozen order of §2
//!
//! | # | arm | reads | what it costs |
//! |---|---|---|---|
//! | 0 | `sol` | 12 o | nothing is decoded — the machine's floor |
//! | 1 | `masques` | 12 o | nested masks, `Fixed96` — the fastest decoder this machine has ever run |
//! | 2 | `cascade-archive` | 8 o | the incumbent's unranking, as it is |
//! | 3 | `cascade-uniformisée` | 8 o | 24 identical steps, branchless, magic reciprocals |
//! | 4 | `marche-binomiale` | 12 o | the E1v decoder: table lookups, no division |
//!
//! 🚨 **The bytes per block are NOT the same across arms, and no ns/bloc here is
//! corrected for traffic.** `sol` reads the 12 bytes of `Fixed96`; the rank arms
//! read 8 (arms 2-3) or the walk's 12-byte record (arm 4). `sol` is therefore
//! the floor of `masques` and **not** the floor of the rank arms. A rank decoder
//! that beats the floor on time does not beat the floor on work.
//!
//! ⚠️ **`masques` is not "the served path" and the journal must not say so.**
//! The served layout is `Planes14`, on CUDA (`llvq-llm/src/fused.rs:68`); on
//! Metal it does not exist — `grep -rn "Planes" llvq-metal/src/` returns
//! nothing. `masques` is here as the practical floor a rank decoder is judged
//! against, nothing more.
//!
//! ## What this bench can and cannot conclude (§0 of the pre-registration)
//!
//! It measures a **decode alone**, on **Metal**, on an **M3 Max**, **one block
//! per lane**, with no matvec, no cross-lane reduction and no tiling. It is not
//! the served kernel and it never will be. There is also no GPU timestamp:
//! metal-rs 0.29 does not expose `GPUStartTime`, so every number here is
//! wall-clock around `commit()`/`wait_until_completed()` minus a measured
//! submission overhead. **There is no way, in the current tooling, to separate
//! GPU time from submission time other than that subtraction**, and the
//! dispersion of the thing subtracted is printed for exactly that reason.
//!
//! Run: `cargo run --release -p llvq-metal --bin rankbench [model.llvq]`

use llvq_artifact::runtime::{transcode, ClassTable, Layout};
use llvq_core::{Golay, SplitMix64, DIM};
use llvq_metal::p1host::{
    binom_table, block_records, cascade_ends, cascade_records, div_table, etalon_cascade,
    pack_block, walk_feed, walk_levels, walk_radix_table, walk_records, WalkFeed,
};
use llvq_search::cns::cns_encode;
use llvq_search::fastdec::{FastDecoder, MAX_KINDS};
use std::fs::File;
use std::io::BufReader;

// ---------------------------------------------------------------------------
// The constants of §1, and where each comes from
// ---------------------------------------------------------------------------

/// §1.1 / §1.2. A floor, not a choice: at 2 M blocks the submission overhead
/// was worth the work being measured, which is one of the three defects that
/// made "25 tok/s, c'est mort" before correction. **Any arm measured on fewer
/// than 2^24 blocks is null and void.**
const N: usize = 1 << 24;

/// §1.1: 3 warm-ups plus the best of 15 repetitions.
const ROUNDS: usize = 18;
const WARMUP_ROUNDS: usize = 3;

/// Submissions averaged into one round's overhead measurement. The per-round
/// overhead is itself a **minimum**, consistent with §1.1's "the minimum, not
/// the mean" — a GPU shared with a compositor has a noise floor above the
/// machine's capability, never below.
const OH_REPS: usize = 3;

/// Threads per threadgroup. Every kernel stages the 24 activations behind
/// `if (tid < DIM)` and a barrier, so a final threadgroup narrower than 24
/// would read an unwritten `xs`. Clipped by `max_threads_per_group()` and
/// printed.
const GROUP: usize = 256;

/// Reservoir seed — printed, so the draw is replayable to the block.
const SEED: u64 = 0x0000_BA1C_5EED;

/// Tolerance: `thesis.rs:84`'s form — relative to `Σ|wᵢxᵢ|`, the magnitude the
/// accumulation ran at — and **not** `decreal.rs:129`'s, which carries an
/// absolute floor of 2e-3 that a small dot product passes almost whatever it
/// decodes. "A fire alarm is not a guard."
const TOL: f64 = 1e-5;

/// §5 of the pre-registration: a number better than this asks for a search for
/// the error *before* it becomes a headline — degraded arm, unrepresentative
/// draw, a loop the compiler removed for want of being observable, anchors that
/// do not reproduce.
const SUSPICIOUS_NS: f64 = 0.20;

/// The two arms' kill thresholds (§4.1) and the CUDA gate (§4.2), verbatim.
const KILL_WALK_NS: f64 = 1.5;
const KILL_CASCADE_NS: f64 = 2.0;
const CUDA_GATE_NS: f64 = 0.45;

/// The pre-registrations this run is bound by. **The bench refuses to time
/// anything until each carries an OpenTimestamps proof.**
///
/// This is not ceremony. P1's own §3 asks for the stamp *before the first line
/// of the bench is written*, "contrairement au pré-enregistrement du lot du 13,
/// dont l'antériorité ne repose que sur un mtime". A rule that lives only in
/// prose gets skipped on the evening someone wants a number; a rule the binary
/// enforces does not.
const STAMPED: &str = "proofs/preregistration-p1-2026-08-13.md";

fn xvec() -> Vec<f32> {
    // Distinct magnitudes per slot, so a *permuted* arrangement moves the dot.
    (0..DIM).map(|i| 1.0 + i as f32 * 0.125).collect()
}

/// Min, median and max — the spread a point value hides (`thesis.rs:535`).
fn spread(mut v: Vec<f64>) -> (f64, f64, f64) {
    v.sort_by(f64::total_cmp);
    (v[0], v[v.len() / 2], v[v.len() - 1])
}

// ---------------------------------------------------------------------------
// The draw — §1.5, §1.6
// ---------------------------------------------------------------------------

/// Reservoir sampling, algorithm R, on a fixed seed.
///
/// `decreal` takes **contiguous prefixes** of every matrix (`decreal.rs:149`)
/// and prints neither seed nor histogram. That is not a draw, it is the
/// beginning of a file: the first blocks of a matrix are the first rows, and
/// nothing says a row's class mix is the matrix's.
///
/// Algorithm R is exactly uniform over the `C(total, N)` subsets, returns
/// exactly `N` elements, runs in one pass, and — unlike "every ninth block" —
/// cannot alias with the matrix/row/block order of the file.
struct Reservoir {
    idx: Vec<u64>,
    gain: Vec<u8>,
    seen: u64,
    rng: SplitMix64,
    cap: usize,
}

impl Reservoir {
    fn new(cap: usize, seed: u64) -> Self {
        Self {
            idx: Vec::with_capacity(cap),
            gain: Vec::with_capacity(cap),
            seen: 0,
            rng: SplitMix64::new(seed),
            cap,
        }
    }

    fn offer(&mut self, idx: u64, gain: u8) {
        self.seen += 1;
        if self.idx.len() < self.cap {
            self.idx.push(idx);
            self.gain.push(gain);
            return;
        }
        // Element `seen` (1-based) is kept with probability cap/seen.
        let j = self.rng.next() % self.seen;
        if (j as usize) < self.cap {
            self.idx[j as usize] = idx;
            self.gain[j as usize] = gain;
        }
    }
}

/// How far the draw's class histogram sits from the file's — **described, not
/// judged**.
///
/// §7 says a draw whose histogram departs from the file's does not answer the
/// question the run poses; it puts no figure on "departs", and the bench plan
/// refused to invent one. A chiffré criterion was proposed as É3(c) — `|z| ≤ 4`
/// over the classes of expected ≥ 25, refusal on an empty class of expected ≥ 5
/// — and **the operator declined it on 2026-08-15**. So this function reports
/// and returns; nothing here refuses a run.
///
/// `z = (observed − expected)/√expected`, `expected = f·N/total`. The counts of
/// classes too small for that `z` to mean anything, and of classes that came out
/// empty, are returned alongside — a number that cannot be judged should at
/// least not be invisible.
///
/// ⚠️ The draw is **without replacement**, so the exact law is hypergeometric,
/// whose σ is `√(expected·(1 − N/total))`. Dividing by `√expected` therefore
/// **understates** the true deviation by a factor ≈ 0,94. Stated rather than
/// corrected, so the printed `z` stays the simple quantity a reader can
/// recompute from the two histograms.
struct DrawCheck {
    max_z: f64,
    at: usize,
    judged: usize,
    too_small: usize,
    empty: Vec<usize>,
}

/// Expected count below which a class is reported but not judged.
const Z_MIN_EXPECTED: f64 = 25.0;
/// Expected count below which an empty class is not evidence of anything.
const EMPTY_MIN_EXPECTED: f64 = 5.0;

fn check_draw(draw: &[u64], file: &[u64], n_draw: u64, n_file: u64) -> DrawCheck {
    let p = n_draw as f64 / n_file as f64;
    let mut c = DrawCheck {
        max_z: 0.0,
        at: usize::MAX,
        judged: 0,
        too_small: 0,
        empty: Vec::new(),
    };
    for (ci, (&d, &f)) in draw.iter().zip(file).enumerate() {
        if f == 0 {
            continue;
        }
        let exp = f as f64 * p;
        if exp >= EMPTY_MIN_EXPECTED && d == 0 {
            c.empty.push(ci);
        }
        if exp < Z_MIN_EXPECTED {
            c.too_small += 1;
        } else {
            c.judged += 1;
        }
        let z = ((d as f64 - exp) / exp.sqrt()).abs();
        if z > c.max_z {
            c.max_z = z;
            c.at = ci;
        }
    }
    c
}

// ---------------------------------------------------------------------------
// The arms
// ---------------------------------------------------------------------------

/// Appended to `llvq_metal::PAYLOAD_MSL`, which already defines `DIM`, the
/// cursor, `ClassRec` and `decode_payload` — the same source `matvec` and
/// `decreal` use. Copied from `decreal.rs:41-78` unchanged: arms 0 and 1 are
/// anchors, and an anchor that drifted would make the run unreadable.
const ANCHOR_MSL: &str = include_str!("../../shaders/anchors.metal");
#[derive(Clone, Copy, PartialEq)]
enum Arm {
    Sol,
    Masques,
    CascadeArchive,
    CascadeUniform,
    Marche,
    SolRang,
    MarcheBloc,
    MarcheBlocPlat,
}

impl Arm {
    /// The frozen order of §2. **An arm added never reorders the existing
    /// ones** (§1.3), so anything new goes at the end of this list.
    const ALL: [Arm; 8] = [
        Arm::Sol,
        Arm::Masques,
        Arm::CascadeArchive,
        Arm::CascadeUniform,
        Arm::Marche,
        // É3(a): added LAST so no existing arm is reordered (§1.3). It reads
        // the 8 bytes of the rank stream and decodes nothing — the floor the
        // rank arms actually have, `sol` being the floor of `masques` alone.
        Arm::SolRang,
        // P1b (proofs/preregistration-p1b-2026-08-15.md), added LAST again for
        // the same reason. `marche-binomiale` decodes ONE 24-slot walk; a real
        // even block needs TWO, plus the codeword and the parity repair. This
        // is the cost of a BLOCK, which is the quantity P1's thresholds name.
        Arm::MarcheBloc,
        // É1 of P1b, added LAST again. Same output, same record, same stride;
        // 48 slot-indexed bytes of per-slot state become 8 counter-indexed
        // words. The gap between it and `marche-bloc` is what the spill cost.
        Arm::MarcheBlocPlat,
    ];

    fn name(self) -> &'static str {
        match self {
            Arm::Sol => "sol",
            Arm::Masques => "masques",
            Arm::CascadeArchive => "cascade-archive",
            Arm::CascadeUniform => "cascade-uniformisée",
            Arm::Marche => "marche-binomiale",
            Arm::SolRang => "sol-rang",
            Arm::MarcheBloc => "marche-bloc",
            Arm::MarcheBlocPlat => "marche-bloc-plat",
        }
    }

    /// Bytes of payload the arm reads per block. Padding added so a windowed
    /// read cannot run off the end of the buffer is **not** counted — the
    /// repo's convention (`thesis.rs:731`).
    fn bytes_per_block(self) -> usize {
        match self {
            Arm::Sol | Arm::Masques => 12,
            Arm::CascadeArchive | Arm::CascadeUniform => 8,
            Arm::Marche => 12,
            Arm::SolRang => 8,
            Arm::MarcheBloc | Arm::MarcheBlocPlat => 12,
        }
    }

    /// Whether §4 puts a threshold on this arm, and which.
    fn threshold(self) -> Option<f64> {
        match self {
            Arm::Marche => Some(KILL_WALK_NS),
            Arm::CascadeUniform => Some(KILL_CASCADE_NS),
            // §4.3: the incumbent is judged against the *uniformised cascade's*
            // threshold, and passing it kills E1v rather than the arm.
            Arm::CascadeArchive => Some(KILL_CASCADE_NS),
            // P1b keeps P1's thresholds without amending them: the kill at 1,5
            // and the CUDA gate at 0,45, read on the block rather than on a
            // walk.
            // P1b keeps P1's thresholds without amending them: the kill at 1,5
            // and the CUDA gate at 0,45, read on the BLOCK rather than on a
            // walk.
            Arm::MarcheBloc | Arm::MarcheBlocPlat => Some(KILL_WALK_NS),
            Arm::Sol | Arm::Masques | Arm::SolRang => None,
        }
    }
}

/// `main` returns `()` and prints with `Display`, not `Debug`.
///
/// A `Result<(), String>` from `main` is reported through `Debug`, which turns
/// every newline of a multi-line message into a literal `\n`. The two messages
/// that matter most here — the stamp gate and the missing archive — are the
/// ones a reader most needs to be able to read.
fn main() {
    if let Err(e) = run() {
        eprintln!("\n{e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    // =======================================================================
    // The stamp gate — before anything, and deliberately not overridable
    // =======================================================================
    let ots = format!("{STAMPED}.ots");
    if !std::path::Path::new(&ots).exists() {
        return Err(format!(
            "le pré-enregistrement de P1 n'est pas horodaté : {ots} est absent.\n\n\
             Ce banc produit la PREMIÈRE MILLISECONDE de P1. Son §3 demande le tampon \
             AVANT elle,\nprécisément pour ne pas hériter de la dette de provenance qu'il \
             reproche au lot du 13,\ndont l'antériorité ne repose que sur un mtime.\n\n\
             Le poser (l'opérateur, pas ce binaire) :\n\n    \
             ots stamp {STAMPED}\n\n\
             Les quatre autres pré-enregistrements (p2..p5) le méritent aussi ; celui-ci \
             est le seul\nque ce banc peut vérifier, parce que c'est le seul qui le lie.\n\n\
             Ce garde n'a pas de dérogation. Une règle qui ne vit que dans la prose se \
             saute le soir\noù quelqu'un veut un chiffre ; une règle que le binaire tient \
             ne se saute pas."
        ));
    }

    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| format!("{}/llvq-q4b.llvq", std::env::var("HOME").unwrap_or_default()));

    // =======================================================================
    // 1. PROVENANCE
    // =======================================================================
    println!("P1 — banc à 5 bras. Pré-enregistrement : {STAMPED} (horodaté).\n");
    let meta = std::fs::metadata(&path).map_err(|e| {
        format!(
            "l'archive scellée du 4B n'est pas sur cette machine : {path} ({e})\n\n\
             P1 se mesure sur le mélange de classes RÉEL du modèle publié — il n'y a ni \
             substitut ni repli\nsynthétique : « un run sur codes synthétiques ne peut pas \
             porter ces seuils, c'est exactement\nla réserve qu'il existe pour lever » (§7)."
        )
    })?;
    println!("FICHIER   {path}");
    println!("          {} octets", meta.len());

    let fd = FastDecoder::new();
    let golay = Golay::new();
    let table = ClassTable::new(&fd, 1);

    // =======================================================================
    // 2. FILE HISTOGRAM + 3. THE DRAW — one pass, §5.2 and §5.3
    // =======================================================================
    let f = File::open(&path).map_err(|e| format!("open {path}: {e}"))?;
    let mut r = BufReader::new(f);
    let h = llvq_artifact::read_header(&mut r).map_err(|e| e.to_string())?;
    let mut file_hist = vec![0u64; fd.n_classes()];
    let mut origin_blocks = 0u64;
    let mut total = 0u64;
    let mut res = Reservoir::new(N, SEED);
    let mut centroids: Option<[f32; 2]> = None;

    let t0 = std::time::Instant::now();
    for _ in 0..h.matrices {
        let m = llvq_artifact::read_matrix_raw(&mut r).map_err(|e| e.to_string())?;
        // Without this, an artifact with 2 gain bits would decode wrong classes
        // in silence: every MSL of this repo reads the gain with `take(c, 1)`.
        assert_eq!(
            m.centroids.len().next_power_of_two().trailing_zeros(),
            1,
            "{}: les shaders de ce banc codent en dur 1 bit de gain",
            m.name
        );
        if centroids.is_none() {
            centroids = Some([m.centroids[0] as f32, m.centroids[1] as f32]);
        }
        for (&idx, &g) in m.indices.iter().zip(&m.gains) {
            total += 1;
            match fd.class_of(idx) {
                Some(ci) => file_hist[ci] += 1,
                None => {
                    assert_eq!(idx, 0, "{}: index {idx} hors de la boule", m.name);
                    origin_blocks += 1;
                }
            }
            res.offer(idx, g as u8);
        }
    }
    let observed = file_hist.iter().filter(|&&c| c > 0).count();
    println!("          {} matrices, {total} blocs, {origin_blocks} origine", h.matrices);
    println!(
        "          {observed} classes observées sur {} — lu en {:.1} s",
        fd.n_classes(),
        t0.elapsed().as_secs_f64()
    );

    // The three numbers `docs/mesures/shell-distribution-4b-2026-08-10.txt:47`
    // already carries. Checked, not assumed: a divergence here is an alert
    // before any timing, not a footnote after it.
    assert_eq!(total, 150_681_600, "le 4B publié fait 150 681 600 blocs");
    assert_eq!(origin_blocks, 0, "le 4B publié ne porte aucun bloc origine");
    assert_eq!(observed, 286, "le 4B publié touche 286 classes");

    let (indices, gains) = (res.idx, res.gain);
    // `transcode` wants u32 gains, the kernels and the etalons want u8. One
    // draw, two views — never two draws.
    let gains32: Vec<u32> = gains.iter().map(|&g| g as u32).collect();
    assert_eq!(indices.len(), N, "le réservoir doit rendre exactement N");
    let mut draw_hist = vec![0u64; fd.n_classes()];
    for &idx in &indices {
        if let Some(ci) = fd.class_of(idx) {
            draw_hist[ci] += 1;
        }
    }
    let drawn = draw_hist.iter().filter(|&&c| c > 0).count();
    let dc = check_draw(&draw_hist, &file_hist, N as u64, total);
    println!("\nTIRAGE    réservoir (algorithme R), graine {SEED:#018x}, N = {N}");
    println!(
        "          {drawn} classes tirées sur {observed} observées au fichier, \
         max |z| = {:.2} (classe {})",
        dc.max_z, dc.at
    );
    println!(
        "          {} classes d'attendu ≥ {Z_MIN_EXPECTED:.0}, {} en dessous (le z n'y a pas de \
         sens), {} vide(s) d'attendu ≥ {EMPTY_MIN_EXPECTED:.0}.\n             Le z divise par \
         √attendu, pas par le σ hypergéométrique : sans remise, il SOUS-ESTIME l'écart\n     \
         vrai d'un facteur ≈ 0,94.",
        dc.judged,
        dc.too_small,
        dc.empty.len()
    );
    println!(
        "          ⚠️ AUCUN SEUIL n'est appliqué à ce |z|. Le §7 exige que le tirage suive le \
         fichier, il ne\n             chiffre pas « suit » ; un critère chiffré a été PROPOSÉ \
         (É3c) et ÉCARTÉ. Le tirage est\n             donc décrit, pas jugé, et l'accepter \
         revient à l'opérateur."
    );
    // §1.6's coverage declaration, and it has to separate three things a single
    // subtraction would merge. `384 − drawn` is NOT the count of entries a file
    // cannot reach: it also swallows the classes this draw simply missed
    // because they are rare, which is a fact about the draw and not about the
    // codebook.
    let shell13 = (0..fd.n_classes())
        .filter(|&ci| fd.levels(ci).shell == 13)
        .count();
    let entries = fd.n_classes() + 1;
    println!(
        "          couverture, en trois termes qui ne se confondent pas :\n            \
         · {entries} entrées de table (383 classes cap 13 + l'origine) ;\n            \
         · {observed} touchées par LE FICHIER — les {} autres se décomposent en 1 origine, \
         {shell13} classes\n              de coquille 13 (hors d'atteinte de tout fichier \
         cap 12) et {} classes cap 12 que ce\n              modèle n'utilise pas ;\n            \
         · {drawn} touchées par CE TIRAGE — les {} restantes sont des classes trop rares \
         pour\n              survivre à un tirage de 1 bloc sur 9, ce qui est un fait sur le \
         tirage, pas sur le codebook.\n          Les entrées hors fichier sont couvertes par \
         la fixture de `bin/p1v0`, jamais ici.",
        entries - observed,
        entries - observed - 1 - shell13,
        observed - drawn
    );

    // =======================================================================
    // 4. THE THREE FEEDS
    // =======================================================================
    let gscale = centroids.expect("au moins une matrice");
    println!(
        "\nFLUX      centroïdes de gain RÉELS de la 1re matrice : [{:.6}, {:.6}]",
        gscale[0], gscale[1]
    );

    let t = std::time::Instant::now();
    let f96 = transcode(&fd, &table, &indices, &gains32, Layout::Fixed96).map_err(|e| e.to_string())?;
    println!(
        "          Fixed96 (bras 0-1) : 12 o/bloc, {:.4} b/poids, transcodé en {:.1} s",
        f96.bits_per_weight(),
        t.elapsed().as_secs_f64()
    );

    // The rank stream: the file's own bits, untouched. Both cascade arms read
    // THE SAME buffer — that is what makes the gap between them a measure of
    // the loop's shape and of nothing else.
    let rank_words: Vec<u64> = indices
        .iter()
        .zip(&gains)
        .map(|(&i, &g)| (i << 1) | u64::from(g))
        .collect();
    println!("          rang d'archive (bras 2-3) : 8 o/bloc, le contenu du fichier tel quel");

    let walk = walk_records(&fd);
    let radices = walk_radix_table(&walk);
    let lev = walk_levels(&fd);
    let binom = binom_table();
    let mut rng = SplitMix64::new(SEED ^ 0x5741_4c4b);
    let mut wfeed = WalkFeed::default();
    for (b, &idx) in indices.iter().enumerate() {
        // É2: no transcoder exists, so the walk is fed ranks drawn uniformly
        // inside the REAL class of each real block. The cost of the decode
        // depends on the data only through the class and the magnitude of the
        // rank, and both are exercised at their real distribution. What is not
        // exercised is the correlation between a block and its particular CNS
        // rank — and nothing suggests one exists, the walk being fixed-count.
        let id = 1 + fd.class_of(idx).expect("index dans la boule");
        let k = walk[id].k as usize;
        let mut ranks = [0u64; MAX_KINDS];
        for (j, rj) in ranks.iter_mut().enumerate().take(k) {
            let radix = radices[id][j];
            *rj = if radix > 1 { rng.next() % radix } else { 0 };
        }
        wfeed.push(id as u32, gains[b], ranks, (rng.next() as u32) & ((1 << DIM) - 1));
    }
    let x = xvec();
    let (walk_words, walk_want) = walk_feed(&walk, &lev, &wfeed, &gscale, &x);

    // P1b's feed: the CNS record of each REAL block, packed at the same 12-byte
    // stride as the walk arm so the two differ only in what they decode. Its
    // etalon is `want_cascade` — the block arm rebuilds the archive's point, so
    // it owes the archive exact equality, like the two cascades.
    let brecs = block_records(&fd, &golay);
    let mut block_words: Vec<u32> = Vec::with_capacity(3 * N);
    for (b, &idx) in indices.iter().enumerate() {
        let ci = fd.class_of(idx).expect("index dans la boule");
        let rec = cns_encode(&fd, idx, gains[b]).expect("index dans la boule");
        block_words.extend_from_slice(&pack_block(
            &brecs[1 + ci],
            (1 + ci) as u32,
            u32::from(gains[b]),
            rec.golay,
            &rec,
        ));
    }
    println!(
        "          marche (bras 4) : {} bits d'enregistrement, stride 12 o — \
         l'adressage des ancres,\n            pour que le bras ne soit pas prix contre elles \
         en payant une autre facture d'adresses",
        llvq_metal::p1host::WALK_RECORD_BITS
    );

    // =======================================================================
    // 5. GPU SETUP
    // =======================================================================
    let anchor_src = format!("{}{}", llvq_metal::PAYLOAD_MSL, ANCHOR_MSL);
    let kernels: Vec<llvq_metal::Kernel> = vec![
        llvq_metal::Kernel::new(&anchor_src, "floor96")?,
        llvq_metal::Kernel::new(&anchor_src, "decode_f96")?,
        llvq_metal::Kernel::new(
            include_str!("../../shaders/cascade_archive.metal"),
            "cascade_archive",
        )?,
        llvq_metal::Kernel::new(
            include_str!("../../shaders/cascade_uniform.metal"),
            "cascade_uniform",
        )?,
        llvq_metal::Kernel::new(
            include_str!("../../shaders/binomial_walk.metal"),
            "decode_walk",
        )?,
        llvq_metal::Kernel::new(&anchor_src, "floor_rank")?,
        llvq_metal::Kernel::new(
            include_str!("../../shaders/binomial_block.metal"),
            "decode_block",
        )?,
        llvq_metal::Kernel::new(
            include_str!("../../shaders/binomial_block_flat.metal"),
            "decode_block_flat",
        )?,
    ];
    let group = GROUP.min(kernels[0].max_threads_per_group() as usize);
    println!(
        "\nGPU       {} — simd {} (lu), max group {}, GROUP effectif {group}",
        kernels[0].device_name(),
        kernels[0].simd_width(),
        kernels[0].max_threads_per_group()
    );
    println!(
        "          {ROUNDS} rounds dont {WARMUP_ROUNDS} jetés, {OH_REPS} soumissions par \
         mesure de surcoût\n          ⚠️ aucun horodatage GPU : metal-rs 0.29 n'expose pas \
         GPUStartTime. Tout est du wall-clock\n             autour de commit()/\
         wait_until_completed(), moins un surcoût MESURÉ — il n'existe aucun\n             \
         autre moyen de séparer temps GPU et temps de soumission dans cet outillage."
    );

    // Per-arm buffers. `Kernel::new` creates a Device AND a command queue per
    // arm (`lib.rs:51`, `:61`), so a buffer belongs to the kernel that made it
    // and the five arms submit on five distinct queues. `thesis` lives with
    // this and its protocol holds; it is a fact to know, not an obstacle.
    let ends = cascade_ends(&fd);
    let recs = cascade_records(&fd, &golay);
    let dv = div_table();
    let class_recs = llvq_metal::gpu_class_table(&fd);

    let b_f96: Vec<_> = kernels[..2].iter().map(|k| k.buffer(&f96.data)).collect();
    let b_x: Vec<_> = kernels.iter().map(|k| k.buffer(&x)).collect();
    let b_gs: Vec<_> = kernels.iter().map(|k| k.buffer(&gscale)).collect();
    let b_out: Vec<_> = kernels.iter().map(|k| k.empty::<f32>(N)).collect();
    let b_tab = kernels[1].buffer(&class_recs);
    // Arms 2, 3 and 5 read THE SAME bits; each holds its own copy because a
    // buffer belongs to the queue that made it.
    let b_rank: Vec<_> = kernels[2..4].iter().map(|k| k.buffer(&rank_words)).collect();
    let b_rank5 = kernels[5].buffer(&rank_words);
    let b_ends: Vec<_> = kernels[2..4].iter().map(|k| k.buffer(&ends)).collect();
    let b_recs: Vec<_> = kernels[2..4].iter().map(|k| k.buffer(&recs)).collect();
    let b_golay: Vec<_> = kernels[2..4]
        .iter()
        .map(|k| k.buffer(golay.codewords()))
        .collect();
    let b_dv = kernels[3].buffer(std::slice::from_ref(&dv));
    // `cascade_uniform` takes the gain from its own byte buffer where
    // `cascade_archive` rides it in bit 0 of the index word. Same bits, two
    // bindings — the feed differs, the arm does not.
    let b_gain = kernels[3].buffer(&gains);
    let b_idx_u = kernels[3].buffer(&indices);
    let b_words = kernels[4].buffer(&walk_words);
    let b_bwords = kernels[6].buffer(&block_words);
    let b_btab = kernels[6].buffer(&brecs);
    let b_bbinom = kernels[6].buffer(&binom);
    let b_bgolay = kernels[6].buffer(golay.codewords());
    let b_fwords = kernels[7].buffer(&block_words);
    let b_ftab = kernels[7].buffer(&brecs);
    let b_fbinom = kernels[7].buffer(&binom);
    let b_fgolay = kernels[7].buffer(golay.codewords());
    let b_walktab = kernels[4].buffer(&walk);
    let b_binom = kernels[4].buffer(&binom);

    let bind = |enc: &metal::ComputeCommandEncoderRef, ai: usize| match Arm::ALL[ai] {
        Arm::Sol => {
            enc.set_buffer(0, Some(&b_f96[0]), 0);
            enc.set_buffer(1, Some(&b_x[0]), 0);
            enc.set_buffer(2, Some(&b_out[0]), 0);
        }
        Arm::Masques => {
            enc.set_buffer(0, Some(&b_f96[1]), 0);
            enc.set_buffer(1, Some(&b_tab), 0);
            enc.set_buffer(2, Some(&b_gs[1]), 0);
            enc.set_buffer(3, Some(&b_x[1]), 0);
            enc.set_buffer(4, Some(&b_out[1]), 0);
        }
        Arm::CascadeArchive => {
            enc.set_buffer(0, Some(&b_rank[0]), 0);
            enc.set_buffer(1, Some(&b_ends[0]), 0);
            enc.set_buffer(2, Some(&b_recs[0]), 0);
            enc.set_buffer(3, Some(&b_golay[0]), 0);
            enc.set_buffer(4, Some(&b_gs[2]), 0);
            enc.set_buffer(5, Some(&b_x[2]), 0);
            enc.set_buffer(6, Some(&b_out[2]), 0);
        }
        Arm::CascadeUniform => {
            enc.set_buffer(0, Some(&b_idx_u), 0);
            enc.set_buffer(1, Some(&b_gain), 0);
            enc.set_buffer(2, Some(&b_ends[1]), 0);
            enc.set_buffer(3, Some(&b_recs[1]), 0);
            enc.set_buffer(4, Some(&b_golay[1]), 0);
            enc.set_buffer(5, Some(&b_dv), 0);
            enc.set_buffer(6, Some(&b_gs[3]), 0);
            enc.set_buffer(7, Some(&b_x[3]), 0);
            enc.set_buffer(8, Some(&b_out[3]), 0);
        }
        Arm::Marche => {
            enc.set_buffer(0, Some(&b_words), 0);
            enc.set_buffer(1, Some(&b_walktab), 0);
            enc.set_buffer(2, Some(&b_binom), 0);
            enc.set_buffer(3, Some(&b_gs[4]), 0);
            enc.set_buffer(4, Some(&b_x[4]), 0);
            enc.set_buffer(5, Some(&b_out[4]), 0);
        }
        Arm::SolRang => {
            enc.set_buffer(0, Some(&b_rank5), 0);
            enc.set_buffer(1, Some(&b_x[5]), 0);
            enc.set_buffer(2, Some(&b_out[5]), 0);
        }
        Arm::MarcheBlocPlat => {
            enc.set_buffer(0, Some(&b_fwords), 0);
            enc.set_buffer(1, Some(&b_ftab), 0);
            enc.set_buffer(2, Some(&b_fbinom), 0);
            enc.set_buffer(3, Some(&b_fgolay), 0);
            enc.set_buffer(4, Some(&b_gs[7]), 0);
            enc.set_buffer(5, Some(&b_x[7]), 0);
            enc.set_buffer(6, Some(&b_out[7]), 0);
        }
        Arm::MarcheBloc => {
            enc.set_buffer(0, Some(&b_bwords), 0);
            enc.set_buffer(1, Some(&b_btab), 0);
            enc.set_buffer(2, Some(&b_bbinom), 0);
            enc.set_buffer(3, Some(&b_bgolay), 0);
            enc.set_buffer(4, Some(&b_gs[6]), 0);
            enc.set_buffer(5, Some(&b_x[6]), 0);
            enc.set_buffer(6, Some(&b_out[6]), 0);
        }
    };

    // =======================================================================
    // 6. V0 — every arm verified before one millisecond is believed (§3, §7)
    // =======================================================================
    println!("\nV0        aucune milliseconde n'est crue avant ce bloc");
    let want_cascade = etalon_cascade(&fd, &indices, &gains, &gscale, &x)?;
    let mut worst = vec![0.0f64; Arm::ALL.len()];
    for (ai, k) in kernels.iter().enumerate() {
        k.dispatch(N as u64, group as u64, |enc| bind(enc, ai));
        // SAFETY: the dispatch above completed and wrote N floats into b_out.
        let got: Vec<f32> = unsafe { k.read(&b_out[ai], N) };
        // Arms 0 and 1 are anchors: `sol` decodes nothing, so it has no etalon
        // beyond being observable, and `masques` is pinned by `decreal`'s own
        // check against the runtime decoder. What must be true of both here is
        // that the output was WRITTEN — a kernel the compiler emptied for want
        // of an observable result would time beautifully and mean nothing (§7).
        let etalon: Option<&[(f64, f64)]> = match Arm::ALL[ai] {
            Arm::CascadeArchive | Arm::CascadeUniform | Arm::MarcheBloc
            | Arm::MarcheBlocPlat => Some(&want_cascade),
            Arm::Marche => Some(&walk_want),
            Arm::Sol | Arm::Masques | Arm::SolRang => None,
        };
        match etalon {
            Some(want) => {
                let mut e = 0.0f64;
                let mut bad = 0usize;
                for (b, (&g, &(exp, mag))) in got.iter().zip(want).enumerate() {
                    let d = (g as f64 - exp).abs();
                    let rel = if mag > 0.0 { d / mag } else { d };
                    if rel > e {
                        e = rel;
                    }
                    if d.is_nan() || rel > TOL {
                        bad += 1;
                        if bad == 1 {
                            println!(
                                "          {} : bloc {b} GPU {g:.9} / CPU {exp:.9}",
                                Arm::ALL[ai].name()
                            );
                        }
                    }
                }
                worst[ai] = e;
                println!(
                    "          {:<22} {N} blocs, pire erreur {e:.1e}·Σ|w·x|{}",
                    Arm::ALL[ai].name(),
                    if bad > 0 {
                        format!("  ROUGE, {bad} blocs > {TOL:.0e}")
                    } else {
                        String::new()
                    }
                );
                if bad > 0 {
                    return Err(format!(
                        "{} échoue V0 : le bras n'existe pas, il n'est pas chronométré \
                         (§3, §7). Correction d'abord.",
                        Arm::ALL[ai].name()
                    ));
                }
            }
            None => {
                let nz = got.iter().filter(|v| **v != 0.0 && v.is_finite()).count();
                println!(
                    "          {:<22} {N} blocs, {nz} sorties non nulles et finies \
                     (ancre : observable, pas d'étalon)",
                    Arm::ALL[ai].name()
                );
                assert!(
                    nz > N / 2,
                    "{}: sortie majoritairement nulle — boucle éliminée ? (§7)",
                    Arm::ALL[ai].name()
                );
            }
        }
    }

    // =======================================================================
    // 7. V1 — the rounds. `Kernel::time` is forbidden here (§2, §1.3)
    // =======================================================================
    let mut times: Vec<Vec<f64>> = vec![Vec::new(); Arm::ALL.len()];
    let mut ohs: Vec<Vec<f64>> = vec![Vec::new(); Arm::ALL.len()];
    for rep in 0..ROUNDS {
        for (ai, k) in kernels.iter().enumerate() {
            // §1.2 as amended: the overhead is measured EVERY round, on the
            // arm's own queue, and its dispersion is printed. At 2^24 blocks
            // the repo chiffres it at 12 % of a floor-speed arm — a number
            // that large is not noise, it is a term of the result.
            let mut oh = f64::INFINITY;
            for _ in 0..OH_REPS {
                oh = oh.min(k.dispatch(1, 1, |_| {}).seconds);
            }
            let t = k.dispatch(N as u64, group as u64, |enc| bind(enc, ai));
            if rep >= WARMUP_ROUNDS {
                times[ai].push(t.seconds);
                ohs[ai].push(oh);
            }
        }
    }

    // =======================================================================
    // 8. THE JOURNAL
    // =======================================================================
    println!("\nSURCOÛT   par bras, min / méd / max sur les {} rounds gardés", ROUNDS - WARMUP_ROUNDS);
    let mut oh_span_ns = vec![0.0f64; Arm::ALL.len()];
    for (ai, arm) in Arm::ALL.iter().enumerate() {
        let (lo, med, hi) = spread(ohs[ai].clone());
        oh_span_ns[ai] = (hi - lo) / N as f64 * 1e9;
        println!(
            "          {:<22} {:.4} / {:.4} / {:.4} ms   =  {:.4} / {:.4} / {:.4} ns/bloc",
            arm.name(),
            lo * 1e3,
            med * 1e3,
            hi * 1e3,
            lo / N as f64 * 1e9,
            med / N as f64 * 1e9,
            hi / N as f64 * 1e9
        );
    }

    // Net per round, and the anomaly that must not be rounded away.
    let mut nets: Vec<Vec<f64>> = Vec::with_capacity(Arm::ALL.len());
    for (ai, arm) in Arm::ALL.iter().enumerate() {
        let v: Vec<f64> = times[ai]
            .iter()
            .zip(&ohs[ai])
            .map(|(t, o)| t - o)
            .collect();
        if v.iter().any(|&n| n <= 0.0) {
            return Err(format!(
                "{}: un round rend un net ≤ 0 après soustraction du surcoût. \
                 Ce n'est pas ramené à zéro et le bras est suspendu (§6.2).",
                arm.name()
            ));
        }
        nets.push(v);
    }

    let ns: Vec<f64> = nets
        .iter()
        .map(|v| v.iter().cloned().fold(f64::INFINITY, f64::min) / N as f64 * 1e9)
        .collect();
    // Ratios formed ROUND BY ROUND, never as a quotient of two minima issued
    // from rounds that never coexisted (règle de maison n°2). Arm 0 is the
    // fast one here — unlike `thesis`, whose arm 0 is FP16 — so the column is
    // "× le sol" = net[ai] / net[0], and `sol` comes out at 1,00× by
    // construction.
    let ratios: Vec<(f64, f64, f64)> = (0..Arm::ALL.len())
        .map(|ai| {
            spread(
                nets[ai]
                    .iter()
                    .zip(&nets[0])
                    .map(|(a, b)| a / b)
                    .collect(),
            )
        })
        .collect();

    println!("\nMESURE    N = {N} blocs, minimum sur {} rounds gardés", ROUNDS - WARMUP_ROUNDS);
    println!(
        "  {:<22}{:>7}{:>10}{:>10}{:>10}{:>11}   × le sol méd [min–max]",
        "bras", "o/bloc", "min ms", "méd ms", "max ms", "ns/bloc"
    );
    for (ai, arm) in Arm::ALL.iter().enumerate() {
        let (lo, med, hi) = spread(nets[ai].clone());
        let (rlo, rmed, rhi) = ratios[ai];
        println!(
            "  {:<22}{:>7}{:>10.3}{:>10.3}{:>10.3}{:>11.4}   {:.2}× [{:.2}–{:.2}]",
            arm.name(),
            arm.bytes_per_block(),
            lo * 1e3,
            med * 1e3,
            hi * 1e3,
            ns[ai],
            rmed,
            rlo,
            rhi
        );
    }
    println!(
        "  ⚠️ les bras 2-4 lisent moins d'octets que le sol ; leur ns/bloc n'est PAS corrigé \
         du trafic,\n     et un décodeur de rang qui bat le sol sur le temps ne bat pas le \
         sol sur le travail."
    );

    // =======================================================================
    // 9. VERDICTS — §4, and the suspension rule of §1.2
    // =======================================================================
    println!("\nVERDICTS  seuils du §4, posés avant la première mesure");
    for (ai, arm) in Arm::ALL.iter().enumerate() {
        let Some(seuil) = arm.threshold() else {
            continue;
        };
        let dist = (ns[ai] - seuil).abs();
        let verdict = if ns[ai] <= seuil { "VERT" } else { "ROUGE" };
        println!(
            "          {:<22} {:.4} ns/bloc contre {seuil:.2} — {verdict} (distance {dist:.4}, \
             étendue du surcoût {:.4} ns)",
            arm.name(),
            ns[ai],
            oh_span_ns[ai]
        );
    }
    // É3(b) proposed transposing §1.2's suspension rule to these absolute
    // thresholds; the operator declined it on 2026-08-15. The rule therefore
    // covers arm-vs-arm gaps only, the two quantities are printed side by side,
    // and the binary does not conclude in the operator's place.
    println!(
        "          ⚠️ la règle de suspension du §1.2 porte sur un écart ENTRE DEUX BRAS ; \
         sa transposition\n             à un seuil ABSOLU a été PROPOSÉE (É3b) et ÉCARTÉE. \
         Les deux quantités sont imprimées,\n             la conclusion revient à \
         l'opérateur."
    );

    // The arm-vs-arm comparison the rule does cover.
    let (a, b) = (3usize, 4usize); // cascade-uniformisée contre marche
    let ecart = (ns[a] - ns[b]).abs();
    let span = oh_span_ns[a].max(oh_span_ns[b]);
    println!(
        "          cascade-uniformisée contre marche : écart {ecart:.4} ns, étendue du \
         surcoût {span:.4} ns — {}",
        if span > ecart / 2.0 {
            "VERDICT SUSPENDU (§1.2)"
        } else {
            "VERDICT RENDU"
        }
    );

    let best = ns[3].min(ns[4]);
    // P1b: the gate read on a BLOCK rather than on a walk. Written before the
    // measurement, in its own document, and it can only withdraw — never grant.
    println!(
        "\n          gate CUDA de P4 (§4.2) : meilleur des deux décodeurs = {best:.4} ns \
         contre {CUDA_GATE_NS:.2}"
    );
    println!(
        "          {}",
        if best <= CUDA_GATE_NS {
            "le bras CUDA de P4 est AUTORISÉ (il reste soumis au go de dépense)"
        } else if ns[4] <= KILL_WALK_NS || ns[3] <= KILL_CASCADE_NS {
            "régime intermédiaire : le bras SURVIT comme point de courbe et n'achète AUCUN \
             bras CUDA — il faut une idée neuve, pas un job"
        } else {
            "les deux bras sont rouges : le package C meurt, le package B se réduit au \
             prefill pur (§4.3)"
        }
    );
    println!(
        "          E1v (§4.3) : cascade-archive rend {:.4} ns contre {KILL_CASCADE_NS:.2} — {}",
        ns[2],
        if ns[2] <= KILL_CASCADE_NS {
            "E1v est MORT-NÉ : l'archive existe, elle est prouvée, elle est plus petite, \
             et elle franchit la barre imposée aux décodeurs neufs"
        } else {
            "l'archive ne franchit pas la barre — elle ne ferme pas la ligne"
        }
    );

    println!(
        "\n          P1b — le même gate lu sur un BLOC : le meilleur décodeur de bloc rend \
         {:.4} ns contre \
         {CUDA_GATE_NS:.2}\n          {}",
        ns[6].min(ns[7]),
        if ns[6].min(ns[7]) <= CUDA_GATE_NS {
            "l'autorisation du bras CUDA de P4 est RÉTABLIE (É1) : un décodeur de bloc franchit le seuil"
        } else {
            "🚨 l'autorisation du bras CUDA de P4 est RETIRÉE : le gate avait été franchi \
             par un nombre\n          qui décrivait une marche, pas un bloc"
        }
    );
    if ns.iter().any(|&v| v < SUSPICIOUS_NS) {
        println!(
            "\n🚨 un bras rend mieux que {SUSPICIOUS_NS} ns/bloc. Le §5 demande de CHERCHER \
             L'ERREUR avant\n   d'en faire un titre : bras dégradé, tirage non représentatif, \
             boucle éliminée faute d'être\n   observable, ancres non reproduites. Les sorties \
             ont été relues (bloc V0) ; le reste est\n   à vérifier à la main avant publication."
        );
    }
    println!(
        "\n⚠️ Aucun repère d'un autre run n'apparaît ci-dessus, et aucun ne doit y être \
         ajouté : ni les\n   0,084 / 0,152 / 0,158 de `decreal` (08-01), ni les 0,08 / 0,11 / \
         8,27 de `decode` (07-31).\n   Les cinq bras se lisent les uns contre les autres DANS \
         CE RUN (§1.4). Si le sol s'écarte\n   notablement du 0,084 du 08-01, ce n'est pas un \
         résultat de P1 : c'est un signal à expliquer."
    );
    println!("⚠️ Aucun tok/s n'est dérivé ici : ce banc ne mesure aucune bande passante.");
    let _ = worst;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The only **mechanical** guard on the draw's uniformity — the histogram
    /// printed at run time is a check on the implementation, not a proof.
    ///
    /// Over many seeds, every element of a small stream must be kept at a
    /// frequency compatible with `cap/total`. A reservoir that favoured the
    /// head (the bug `decreal`'s contiguous prefixes *are*) shows up here as a
    /// first element kept far too often.
    #[test]
    fn the_reservoir_is_uniform() {
        const TOTAL: usize = 500;
        const CAP: usize = 50;
        const TRIALS: usize = 4_000;
        let mut kept = vec![0u32; TOTAL];
        for s in 0..TRIALS as u64 {
            let mut r = Reservoir::new(CAP, 0x9E37_79B9 ^ s);
            for i in 0..TOTAL {
                r.offer(i as u64, 0);
            }
            assert_eq!(r.idx.len(), CAP, "le réservoir doit rendre exactement CAP");
            for &v in &r.idx {
                kept[v as usize] += 1;
            }
        }
        let expect = TRIALS as f64 * CAP as f64 / TOTAL as f64;
        // Binomial: σ = √(T·p·(1−p)); 5σ over 500 bins is a ~1e-4 false-alarm
        // budget for the whole test, which is the right side to err on for a
        // guard that must not go red on a good implementation.
        let sigma = (TRIALS as f64 * (CAP as f64 / TOTAL as f64) * (1.0 - CAP as f64 / TOTAL as f64))
            .sqrt();
        for (i, &c) in kept.iter().enumerate() {
            let z = (c as f64 - expect) / sigma;
            assert!(
                z.abs() < 5.0,
                "élément {i} gardé {c} fois, attendu {expect:.1} (z = {z:.2}) — \
                 le tirage n'est pas uniforme"
            );
        }
    }
}
