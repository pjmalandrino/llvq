//! **P1 — the rank-decode bench.** What a rank decode costs on real blocks.
//!
//! Pre-registration: `proofs/preregistration-p1-2026-08-13.md`, whose §1 fixes
//! the accounting, §2 the five arms, §3 the exactness gate, §4 the thresholds
//! and §7 what would invalidate the whole thing. Nothing here is decided at
//! run time; this file is the pre-registration executed. Two later documents
//! add arms to it without amending a threshold: `…-p1b-2026-08-15.md` (the
//! block arms) and `…-p1c-2026-08-15.md` (the E1v stream's addressing).
//!
//! ## The arms, in the frozen order of §2 — an arm added never reorders one
//!
//! | # | arm | reads | what it costs |
//! |---|---|---|---|
//! | 0 | `sol` | 12 o | nothing is decoded — the machine's floor |
//! | 1 | `masques` | 12 o | nested masks, `Fixed96` — the fastest decoder this machine has ever run |
//! | 2 | `cascade-archive` | 8 o | the incumbent's unranking, as it is |
//! | 3 | `cascade-uniformisée` | 8 o | 24 identical steps, branchless, magic reciprocals |
//! | 4 | `marche-binomiale` | 12 o | one 24-slot walk: table lookups, no division |
//! | 5 | `sol-rang` | 8 o | the rank stream's own floor (É3a) |
//! | 6 | `marche-bloc` | 12 o | a whole BLOCK, at a fixed stride (P1b) |
//! | 7 | `marche-bloc-plat` | 12 o | the same, without the register spill (É1) |
//! | 8 | `e1v-flux` | variable | the same decode, on the REAL E1v stream (P1c) |
//!
//! 🚨 **Arms 6 and 8 decode the same thing and differ only in how they find
//! it.** That is the point of running them together: their gap is the price of
//! E1v's addressing — base word, fixed-stride header, warp-scan over 32 widths,
//! a field read at an arbitrary bit offset — and of nothing else.
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
    binom_table, block_records, cascade_ends, cascade_records, div_table, e1v_fixture,
    e1v_payload_bits, etalon_cascade, etalon_cns, pack_block, walk_feed, walk_levels,
    walk_radix_table, walk_records, WalkFeed,
};
use llvq_search::cns::cns_encode;
use llvq_search::fastdec::{FastDecoder, MAX_KINDS};
use std::collections::BTreeSet;
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
///
/// P1c is on the list for the same reason and by its own §0: it asks for the
/// stamp *before the first millisecond*, "comme P1", precisely because P5 and
/// P1b were stamped after theirs and their journals carry that debt. Adding an
/// arm to this bench therefore adds a document to this gate — which is what
/// stops the arm from being timed on the strength of a file's mtime.
///
/// Each document is pinned to the sha256 its `.ots` attests (`ots info` on
/// the proof, cross-checked with `shasum -a 256` on 2026-08-18). Until then
/// the gate checked only that the proof *exists* — which does not protect
/// against the defect that actually happened in this same directory: the
/// 2026-08-10 and 2026-08-11 preregistrations were edited after anchoring,
/// detaching their anchors while passing every existence check. Editing a
/// stamped document now means re-anchoring it and updating its pin here —
/// two operator steps, both deliberate.
const STAMPED: [(&str, &str); 3] = [
    (
        "proofs/preregistration-p1-2026-08-13.md",
        "5109b35f85618e9a3fef32f1fc325f25068b875416cc2693940e1a83fad6a5c6",
    ),
    (
        "proofs/preregistration-p1b-2026-08-15.md",
        "d027c9d2144720f6d59e76b15f8cde7b5801fa5916ff56fbb47c8e65589ea7d0",
    ),
    (
        "proofs/preregistration-p1c-2026-08-15.md",
        "5b2ccc3bed009e4ac2bf0e748960bb3904e54bef77919fa06132643ce6f7336d",
    ),
];

/// SHA-256, written out rather than pulled in — a twin of the one in
/// `llvq-cuda/src/gpu.rs`, copied and not shared: that crate is
/// target-gated to Linux and this one to macOS, and forty lines of hashing
/// do not justify a crate the core's zero-dependency policy would then have
/// to carry.
fn sha256_hex(data: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let mut msg = data.to_vec();
    let bitlen = (data.len() as u64) * 8;
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bitlen.to_be_bytes());

    for chunk in msg.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (i, word) in w.iter_mut().enumerate().take(16) {
            let b = &chunk[i * 4..i * 4 + 4];
            *word = u32::from_be_bytes([b[0], b[1], b[2], b[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let mut v = h;
        for i in 0..64 {
            let s1 = v[4].rotate_right(6) ^ v[4].rotate_right(11) ^ v[4].rotate_right(25);
            let ch = (v[4] & v[5]) ^ ((!v[4]) & v[6]);
            let t1 = v[7]
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = v[0].rotate_right(2) ^ v[0].rotate_right(13) ^ v[0].rotate_right(22);
            let maj = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
            let t2 = s0.wrapping_add(maj);
            v = [
                t1.wrapping_add(t2),
                v[0],
                v[1],
                v[2],
                v[3].wrapping_add(t1),
                v[4],
                v[5],
                v[6],
            ];
        }
        for (hi, vi) in h.iter_mut().zip(v) {
            *hi = hi.wrapping_add(vi);
        }
    }
    h.iter().map(|x| format!("{x:08x}")).collect()
}

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

/// Worst deviation relative to `Σ|wᵢxᵢ|`, and how many blocks are out of
/// [`TOL`] — the same reading the V0 loop makes of a timed arm, factored out
/// because the fixture makes it too.
fn worst_rel(got: &[f32], want: &[(f64, f64)]) -> (f64, usize) {
    assert_eq!(got.len(), want.len(), "un flottant par bloc");
    let (mut worst, mut bad) = (0.0f64, 0usize);
    for (&g, &(exp, mag)) in got.iter().zip(want) {
        let d = (g as f64 - exp).abs();
        let rel = if mag > 0.0 { d / mag } else { d };
        if rel > worst {
            worst = rel;
        }
        if d.is_nan() || rel > TOL {
            bad += 1;
        }
    }
    (worst, bad)
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
    E1vFlux,
}

impl Arm {
    /// The frozen order of §2. **An arm added never reorders the existing
    /// ones** (§1.3), so anything new goes at the end of this list.
    const ALL: [Arm; 9] = [
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
        // P1c (proofs/preregistration-p1c-2026-08-15.md), added LAST again.
        // `marche-bloc` decodes a block out of a FIXED 12-byte stride; the real
        // E1v stream has variable widths and a warp-scan. Same decode body, to
        // the byte — only the way the record is found changes, so the gap is
        // the price of the addressing.
        Arm::E1vFlux,
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
            Arm::E1vFlux => "e1v-flux",
        }
    }

    /// Bytes of payload the arm reads per block, when that is a *number*.
    ///
    /// Padding added so a windowed read cannot run off the end of the buffer is
    /// **not** counted — the repo's convention (`thesis.rs:731`).
    ///
    /// `None` for `e1v-flux`, and the `None` is the result rather than a gap in
    /// the table: E1v's whole shape is that a record's width depends on its
    /// class. Printing a rounded average in this column would put a fixed
    /// stride next to a variable one under the same heading, which is exactly
    /// the kind of comparison the dossier's errata are made of. The average is
    /// printed once, in the feed section, next to the b/poids it comes from.
    fn bytes_per_block(self) -> Option<usize> {
        match self {
            Arm::Sol | Arm::Masques => Some(12),
            Arm::CascadeArchive | Arm::CascadeUniform => Some(8),
            Arm::Marche => Some(12),
            Arm::SolRang => Some(8),
            Arm::MarcheBloc | Arm::MarcheBlocPlat => Some(12),
            Arm::E1vFlux => None,
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
            // P1c §3 keeps P1's thresholds without amending them either: above
            // 1,5 ns/bloc the decoder of E1v is dead. The CUDA gate at 0,45 is
            // NOT read on this arm but on the best block decoder of the run —
            // see the verdict block, where that is spelled out rather than
            // assumed.
            Arm::E1vFlux => Some(KILL_WALK_NS),
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
    for (doc, pinned) in STAMPED {
        let ots = format!("{doc}.ots");
        if !std::path::Path::new(&ots).exists() {
            return Err(format!(
                "un pré-enregistrement qui lie ce run n'est pas horodaté : {ots} est \
                 absent.\n\n\
                 Ce banc produit la PREMIÈRE MILLISECONDE de chacun des trois documents \
                 ci-dessous.\nLe §3 de P1 demande le tampon AVANT elle, précisément pour ne \
                 pas hériter de la dette de\nprovenance qu'il reproche au lot du 13, dont \
                 l'antériorité ne repose que sur un mtime ;\nle §0 de P1c le redemande pour \
                 son propre bras, P5 et P1b ayant été tamponnés APRÈS.\n\n\
                 Le poser (l'opérateur, pas ce binaire) :\n\n    \
                 ots stamp {doc}\n\n\
                 Les documents que ce banc exige :\n{}\n\n\
                 Ce garde n'a pas de dérogation. Une règle qui ne vit que dans la prose se \
                 saute le soir\noù quelqu'un veut un chiffre ; une règle que le binaire tient \
                 ne se saute pas.",
                STAMPED
                    .iter()
                    .map(|(d, _)| format!(
                        "    {} {d}",
                        if std::path::Path::new(&format!("{d}.ots")).exists() {
                            "✅"
                        } else {
                            "❌"
                        }
                    ))
                    .collect::<Vec<_>>()
                    .join("\n")
            ));
        }
        // Existence is not integrity: the .ots binds a *hash*, and the defect
        // this half of the gate closes has already happened twice in proofs/
        // (documents edited after anchoring, anchors silently detached).
        let bytes = std::fs::read(doc)
            .map_err(|e| format!("impossible de lire {doc} pour vérifier son empreinte : {e}"))?;
        let got = sha256_hex(&bytes);
        if got != pinned {
            return Err(format!(
                "l'empreinte de {doc} n'est plus celle que son tampon atteste.\n\n\
                 attendue (ots info {doc}.ots) : {pinned}\n\
                 calculée sur le fichier actuel : {got}\n\n\
                 Le document a été modifié après son ancrage — exactement le défaut des \
                 préregs du 08-10\net du 08-11, que le garde à existence-seule laissait \
                 passer. Si l'édition est légitime :\nré-ancrer (ots stamp) puis mettre à \
                 jour l'empreinte épinglée dans STAMPED — deux gestes\nd'opérateur, aucun \
                 des deux à ce binaire."
            ));
        }
    }

    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| format!("{}/llvq-q4b.llvq", std::env::var("HOME").unwrap_or_default()));

    // =======================================================================
    // 1. PROVENANCE
    // =======================================================================
    println!(
        "P1 — banc à {} bras. Pré-enregistrements, tous horodatés :",
        Arm::ALL.len()
    );
    for (doc, _) in STAMPED {
        println!("          {doc}");
    }

    // =======================================================================
    // LA RÉSERVE — en tête, et pas en note de bas de page (P1c §4)
    // =======================================================================
    //
    // Elle est imprimée AVANT le premier chiffre, et par le binaire plutôt que
    // par la prose du journal, pour une raison mécanique : un journal se
    // fabrique en redirigeant cette sortie, donc une réserve imprimée ici est
    // dans son en-tête quoi qu'il arrive, tandis qu'une réserve qu'il faut
    // penser à recopier finit sous les tableaux — ou nulle part.
    println!(
        "\n🚨 RÉSERVE, EN TÊTE — le bras `e1v-flux` mesure le MEILLEUR CAS de l'adressage E1v.\n   \
         Ici `gid` EST l'indice de bloc : un SIMD group de 32 lanes consécutives EST exactement\n   \
         un groupe E1v, et l'alignement est donc vrai PAR CONSTRUCTION. Le matvec servi met un\n   \
         warp par LIGNE, et `nblocks mod 32` vaut 10 ou 21 sur les cinq formes du 4B — aucun warp\n   \
         n'y lit un seul groupe : il lirait deux mots de base et scannerait deux régions d'en-têtes.\n   \
         Ce banc ne mesure pas non plus le coût EN BITS de cet alignement, acquis ailleurs à +0,48 %.\n   \
         (`docs/mesures/x3-alignement-warp-2026-08-15.txt`, et §4 du pré-enregistrement de P1c.)"
    );
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
         la fixture de `bin/p1v0` pour les bras 3 et 4, et\n          par celle du bras 8, \
         qui tourne ici même — parce que l'origine y est un CAS D'ADRESSAGE\n          \
         (charge utile vide, entrée de table 0) et pas seulement une classe de plus.",
        entries - observed,
        entries - observed - 1 - shell13,
        observed - drawn
    );

    // =======================================================================
    // 4. THE FOUR FEEDS
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

    // P1c's feed: the REAL E1v stream over the SAME draw. The writer is
    // `llvq_artifact::e1v::transcode_e1v` — the one whose round trip is proved
    // over the 150 681 600 blocks of the 4B (P5) — not a re-implementation, so
    // what is new in this arm is the reading and nothing else.
    //
    // 🕳️ Two assertions stood here, pinning `p1host`'s restatement of E1V_GROUP
    // and E1V_ORIGIN_ID against the writer's. They are gone because the
    // restatement is gone: the table moved to `llvq_artifact::blockrec`, which
    // reads the format's own constants. A pin between a value and itself is not
    // a weaker guard, it is a guard about nothing.
    let pay = e1v_payload_bits(&fd, &golay, &brecs);
    let t = std::time::Instant::now();
    let e1v = llvq_artifact::e1v::transcode_e1v(&fd, &golay, &indices, &gains32)
        .map_err(|e| e.to_string())?;
    let e1v_secs = t.elapsed().as_secs_f64();
    let e1v_bpw = e1v.bits_per_weight();
    let e1v_bytes = (e1v.data.len() + 4 * e1v.bases.len()) as f64 / N as f64;
    let pay_max = pay.iter().copied().max().expect("la table n'est pas vide");
    println!(
        "          E1v (bras 8) : le VRAI flux, {} o + {} o de mots de base, transcodé en \
         {e1v_secs:.1} s\n            largeur variable — {e1v_bytes:.3} o/bloc en moyenne, \
         {e1v_bpw:.4} b/poids adressage compris ; en-têtes 10 bits\n            à stride fixe, \
         somme préfixe SIMD sur les 32 largeurs du groupe, charge utile au pire\n            \
         {pay_max} bits — le §1 en divulgue 56, mais 56 est un RECORD, en-tête compris, et \
         c'est la\n            charge utile que la somme préfixe additionne",
        e1v.data.len(),
        4 * e1v.bases.len()
    );
    println!(
        "          ⚠️ ce {e1v_bpw:.4} est celui du TIRAGE, pas du fichier : les 32 blocs d'un \
         groupe sont ici\n            des blocs sans voisinage, et le terme d'arrondi au mot ne \
         tombe donc pas comme en\n            ordre de fichier, où la mesure publiée est 2,3877 \
         b/poids noyau."
    );
    // Two words of padding so the three-word window of the last lane cannot run
    // off the end. Not counted in the arm's bytes — `thesis.rs:731`.
    let mut e1v_data = e1v.data;
    e1v_data.extend_from_slice(&[0u8; 8]);
    let e1v_bases = e1v.bases;

    // The fixture, and it is not decoration: the published 4B carries ZERO
    // origin block (asserted above), so the draw never exercises the one header
    // value whose payload is empty — the id that maps to table entry 0 and
    // contributes nothing to the warp-scan. It also reaches the 97 classes the
    // file never uses, and one group made entirely of the widest class, which is
    // the largest prefix sum the addressing can ever be asked for.
    let (fix_idx, fix_gain) = e1v_fixture(&fd, &pay, GROUP);
    let fix = llvq_artifact::e1v::transcode_e1v(
        &fd,
        &golay,
        &fix_idx,
        &fix_gain.iter().map(|&g| u32::from(g)).collect::<Vec<_>>(),
    )
    .map_err(|e| e.to_string())?;
    let mut fix_data = fix.data;
    fix_data.extend_from_slice(&[0u8; 8]);

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
        llvq_metal::Kernel::new(include_str!("../../shaders/e1v_flux.metal"), "decode_e1v")?,
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
    // Arm 8. Its `tab`, `binom` and `golay` are the SAME tables arms 6 and 7
    // read, copied onto its own queue — so a divergence between it and
    // `marche-bloc` cannot come from a table.
    let b_edata = kernels[8].buffer(&e1v_data);
    let b_ebases = kernels[8].buffer(&e1v_bases);
    let b_etab = kernels[8].buffer(&brecs);
    let b_epay = kernels[8].buffer(&pay);
    let b_ebinom = kernels[8].buffer(&binom);
    let b_egolay = kernels[8].buffer(golay.codewords());

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
        Arm::E1vFlux => {
            enc.set_buffer(0, Some(&b_edata), 0);
            enc.set_buffer(1, Some(&b_ebases), 0);
            enc.set_buffer(2, Some(&b_etab), 0);
            enc.set_buffer(3, Some(&b_epay), 0);
            enc.set_buffer(4, Some(&b_ebinom), 0);
            enc.set_buffer(5, Some(&b_egolay), 0);
            enc.set_buffer(6, Some(&b_gs[8]), 0);
            enc.set_buffer(7, Some(&b_x[8]), 0);
            enc.set_buffer(8, Some(&b_out[8]), 0);
        }
    };

    // =======================================================================
    // 6. V0 — every arm verified before one millisecond is believed (§3, §7)
    // =======================================================================
    println!("\nV0        aucune milliseconde n'est crue avant ce bloc");

    // ---- V0(a) : la fixture E1v, ce que le tirage ne peut pas atteindre ----
    //
    // Elle passe AVANT le tirage : si le décodeur est faux sur l'origine ou sur
    // une classe que le fichier n'habite pas, le tirage ne le dira jamais, et
    // un bras vert sur 2^24 blocs serait vert pour la mauvaise raison.
    let nf = fix_idx.len();
    assert!(
        nf.is_multiple_of(group),
        "{nf} blocs de fixture ne font pas un nombre entier de threadgroups de {group}"
    );
    // La couverture est ASSERTÉE, pas imprimée : une fixture qui raterait
    // l'origine imprimerait exactement la même ligne verte qu'une fixture qui
    // l'atteint (§5 du dossier).
    let fseen: BTreeSet<Option<usize>> = fix_idx.iter().map(|&i| fd.class_of(i)).collect();
    assert!(
        fseen.contains(&None),
        "la fixture ne contient pas l'origine, qui est la moitié de sa raison d'être"
    );
    for ci in 0..fd.n_classes() {
        assert!(fseen.contains(&Some(ci)), "classe {ci} absente de la fixture E1v");
    }
    let fb_data = kernels[8].buffer(&fix_data);
    let fb_bases = kernels[8].buffer(&fix.bases);
    let fb_out = kernels[8].empty::<f32>(nf);
    kernels[8].dispatch(nf as u64, group as u64, |enc| {
        enc.set_buffer(0, Some(&fb_data), 0);
        enc.set_buffer(1, Some(&fb_bases), 0);
        enc.set_buffer(2, Some(&b_etab), 0);
        enc.set_buffer(3, Some(&b_epay), 0);
        enc.set_buffer(4, Some(&b_ebinom), 0);
        enc.set_buffer(5, Some(&b_egolay), 0);
        enc.set_buffer(6, Some(&b_gs[8]), 0);
        enc.set_buffer(7, Some(&b_x[8]), 0);
        enc.set_buffer(8, Some(&fb_out), 0);
    });
    // SAFETY: the dispatch above completed and wrote nf floats into fb_out.
    let fix_got: Vec<f32> = unsafe { kernels[8].read(&fb_out, nf) };
    let fix_want = etalon_cns(&fd, &golay, &fix_idx, &fix_gain, &gscale, &x)?;
    let (fix_worst, fix_bad) = worst_rel(&fix_got, &fix_want);
    println!(
        "          {:<22} {nf} blocs de FIXTURE — les {} classes (toutes), l'origine que le \
         4B ne\n          {:<22} porte pas, et un groupe entier de la classe la plus large. \
         Pire erreur {fix_worst:.1e}·Σ|w·x|{}",
        Arm::E1vFlux.name(),
        fd.n_classes(),
        "",
        if fix_bad > 0 {
            format!("\n          ROUGE, {fix_bad} blocs > {TOL:.0e}")
        } else {
            String::new()
        }
    );
    if fix_bad > 0 {
        return Err(format!(
            "{} échoue V0 sur la fixture : le bras n'existe pas, il n'est pas chronométré \
             (§5 du pré-enregistrement de P1c). Correction d'abord.",
            Arm::E1vFlux.name()
        ));
    }

    // ---- V0(b) : le tirage, et ses deux étalons ----------------------------
    let want_cascade = etalon_cascade(&fd, &indices, &gains, &gscale, &x)?;
    // L'étalon que P1c nomme (§2) : `cns_decode`, sur les mêmes blocs. Il n'est
    // pas neuf — P5 C2 l'a balayé contre `FastDecoder::decode` sur les
    // 150 681 600 blocs du 4B — et il n'est pas non plus le même chemin de code
    // que `etalon_cascade`. Les deux sont donc calculés et leur égalité EXIGÉE :
    // pour le prix d'une passe CPU, C2 est rétabli sur les blocs mêmes qu'on
    // s'apprête à chronométrer, au lieu d'être cité.
    let want_cns = etalon_cns(&fd, &golay, &indices, &gains, &gscale, &x)?;
    let disagree = want_cns
        .iter()
        .zip(&want_cascade)
        .filter(|(a, b)| a.0 != b.0 || a.1 != b.1)
        .count();
    assert_eq!(
        disagree, 0,
        "les deux étalons divergent sur {disagree} blocs : la re-bijection de P5 (C2) ne \
         tient pas sur ce tirage, et aucun bras n'est chronométrable avant d'en connaître \
         la cause"
    );
    println!(
        "          {:<22} les deux étalons — `cns_decode` (nommé par P1c) et \
         `FastDecoder::decode` — coïncident\n          {:<22} au bit près sur les {N} blocs : \
         C2 de P5 refait sur le tirage, pas cité.",
        "étalons", ""
    );

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
            // §2 of P1c names `cns_decode`, and the block above has just shown
            // it to be the same numbers as the archive's on this draw.
            Arm::E1vFlux => Some(&want_cns),
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
            arm.bytes_per_block()
                .map_or_else(|| "var".to_string(), |b| b.to_string()),
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
    // =======================================================================
    // 9bis. P1c — le prix de l'adressage, et la règle de restitution
    // =======================================================================
    let (a8, a6) = (ns[8], ns[6]);
    let addr = a8 - a6;
    let span68 = oh_span_ns[6].max(oh_span_ns[8]);
    println!(
        "\n          P1c — le FLUX E1v, adressage compris : {a8:.4} ns/bloc contre \
         {KILL_WALK_NS:.2} — {}",
        if a8 <= KILL_WALK_NS { "VERT" } else { "ROUGE" }
    );
    println!(
        "          prix de l'adressage = e1v-flux − marche-bloc = {addr:+.4} ns/bloc \
         ({:+.1} %), étendue du surcoût {span68:.4} ns\n          {} — mot de base, en-tête à \
         stride fixe, somme préfixe SIMD sur 32 largeurs, fenêtre à 3 mots",
        100.0 * addr / a6,
        if span68 > addr.abs() / 2.0 {
            "VERDICT SUSPENDU (§1.2) : l'écart n'est pas grand devant le bruit du surcoût"
        } else {
            "VERDICT RENDU (§1.2)"
        }
    );
    // §6 of P1c: this arm pays everything `marche-bloc` pays PLUS the
    // addressing, so a faster figure is a reason to look for the error, not a
    // headline. Named here rather than left to the generic suspicion below,
    // because the generic one only fires under 0,20 ns.
    if a8 < a6 {
        println!(
            "          🚨 e1v-flux est PLUS RAPIDE que marche-bloc, ce que le §6 du \
             pré-enregistrement\n             désigne d'avance comme un signal d'erreur : ce \
             bras paie tout ce que l'autre paie,\n             plus l'adressage. Chercher la \
             cause avant d'en faire quoi que ce soit."
        );
    }

    let (best_i, best_block) = [6usize, 7, 8]
        .iter()
        .map(|&i| (i, ns[i]))
        .min_by(|x, y| x.1.total_cmp(&y.1))
        .expect("trois bras décodent un bloc");
    println!(
        "\n          P1c §3 — RESTITUTION, écrite avant la mesure et symétrique de celle qui a \
         retiré.\n          Elle porte sur le meilleur décodeur de BLOC du run, pas sur le bras \
         neuf : c'est `{}`\n          à {best_block:.4} ns contre {CUDA_GATE_NS:.2} — {}",
        Arm::ALL[best_i].name(),
        if best_block <= CUDA_GATE_NS {
            "l'autorisation du bras CUDA de P4 est RÉTABLIE"
        } else {
            "l'autorisation reste RETIRÉE (une règle qui ne saurait que retirer ne vaudrait \
             pas mieux qu'une\n          règle qui ne saurait que donner ; celle-ci pouvait \
             tirer, et elle ne tire pas)"
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
