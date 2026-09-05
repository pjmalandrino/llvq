//! F1 — the REAL access distribution of the decoder table, on the Gaussian
//! bench, with a held-out (cross-validated) hot set.
//!
//! `nice -n 10 cargo run --release -p llvq-bench --example f1accesscv -- [n_blocks] [minutes_cap] [threads]`
//!
//! The F1 table floor (`proofs/preregistration-f1-plancher-table-2026-09-04.md`)
//! was measured with UNIFORM random lookups over a table of footprint S, and
//! its §5 says the true cost lies in the bracket `D(16 KiB) ≤ D(real) ≤ D(4 MiB)`
//! because nobody had computed how skewed the real lookups are. This computes
//! it, at $0: encode F1b's own evaluation blocks with the very codebook F1b
//! measured (`Codebook::new([12, 15, 12])`, seed `0x0f1b_2026_0904` after the
//! 4,000 training blocks `bin/f1bench` draws first, the same 18-point scale
//! grid), recover for every block its three `(section set, region)`, map each
//! region to its sign-flip orbit and to the sign vector ε that carries it onto
//! the orbit's representative, and rank the ε-image of the chosen 8-coordinate
//! point inside the representative's region — that rank is the table entry the
//! kernel would read. Then count.
//!
//! Bench-only: touches nothing outside `llvq-bench`, and nothing here is a gate.
//!
//! ## What is reported
//!
//! * the CDF of lookups against cumulative table bytes, hottest entries first,
//!   for all lookups and for the end / middle families separately — both
//!   IN-SAMPLE (the hot set chosen and scored on the same lookups, an
//!   optimistic upper bound: with n lookups over a 570k-entry table, a
//!   uniform draw would still show n/... "coverage" from its singletons) and
//!   HELD-OUT (chosen on the even blocks, scored on the odd ones and vice
//!   versa, the honest number);
//! * the plug-in entropy of the entry index per section, and its finite-sample
//!   cap `log2(n_lookups)`;
//! * which orbits could pack an entry in 4 B (max |coord| ≤ 7) and the share
//!   of lookups they carry;
//! * the sector-granular CDF (32 B sectors of a rank-ordered table, what an L1
//!   actually fills);
//! * the estimate `D_real = c8·D(8 KiB) + (c16 − c8)·D(16 KiB) + (1 − c16)·D(4 MiB)`
//!   against the budget `H = 1.538 ms`, and the `c16` that would be needed.

use llvq_bench::f1::{Codebook, SectionSet, Trellis, GOLAY_STATES, SECTION};
use llvq_bench::gauss_block;
use llvq_core::SplitMix64;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

/// The split F1b measured.
const W: [u32; 3] = [12, 15, 12];
/// The prereg's fixed seed (`bin/f1bench.rs`).
const SEED: u64 = 0x0f1b_2026_0904;
/// `f1bench` draws these training blocks before the evaluation blocks; skipped
/// here so the blocks encoded are exactly F1b's evaluation blocks, in order.
const N_TRAIN_SKIP: usize = 4_000;
/// Points of norm² ≤ `T_SIG`, sorted — enough to identify a coset up to sign
/// (`examples/f1table.rs`).
const T_SIG: usize = 64;
/// Bytes per packed entry (8 coordinates, max |coord| = 9 → 5 bits signed).
const ENTRY_BYTES: usize = 6;
/// The measured points of the floor (job 6a9b4b98e686246ca69a2b20, L40S), ms.
const D_8K: f64 = 0.344;
const D_16K: f64 = 0.663;
const D_4M: f64 = 4.517;
/// The budget F1 must hold for table AND decoding, ms (prereg §3).
const H_BUDGET: f64 = 1.538;

// ---------------------------------------------------------------------------
// Ranking a member of a region: by norm, then lexicographic — the rank the
// label field carries. Reimplemented here (the library keeps its own private)
// and checked against `region_points` before any lookup is counted.
// ---------------------------------------------------------------------------

struct Ranker {
    set: SectionSet,
    rho2: usize,
    n_tie: u64,
    size: u64,
    /// `cum_below[t]` = members with norm² < t, for t in 0..=rho2.
    cum_below: Vec<u64>,
    /// `suffix[pattern][j][n][par]`: fills of positions j..8 with exact norm n
    /// and k-parity par, for that pattern alone.
    suffix: Vec<Vec<Vec<[u64; 2]>>>,
    /// Members of norm² ≤ rho2, including the whole boundary shell: every rank
    /// this returns is below it.
    n_total: u64,
    /// Largest |coordinate| over the region's members.
    max_abs: i32,
    /// `hist[t]` = members of norm² exactly t; on the boundary shell only
    /// `n_tie` of them are entries.
    hist: Vec<u64>,
}

impl Ranker {
    fn new(set: SectionSet, w: u32) -> Self {
        let region = set.region(w);
        let rho2 = region.rho2;
        let hist = set.norm_histogram(rho2);
        let mut cum_below = vec![0u64; rho2 + 1];
        for t in 1..=rho2 {
            cum_below[t] = cum_below[t - 1] + hist[t - 1];
        }
        let n_total = cum_below[rho2] + hist[rho2];
        let suffix = set
            .patterns
            .iter()
            .map(|&pattern| {
                let mut s = vec![vec![[0u64; 2]; rho2 + 1]; SECTION + 1];
                s[SECTION][0][0] = 1;
                for j in (0..SECTION).rev() {
                    let base = set.p as i32 + 2 * ((pattern >> j & 1) as i32);
                    let reach = (rho2 as f64).sqrt().floor() as i32;
                    let lo = (-reach - base).div_euclid(4);
                    let hi = (reach - base).div_euclid(4) + 1;
                    for k in lo..=hi {
                        let v = base + 4 * k;
                        let sq = (v * v) as usize;
                        if sq > rho2 {
                            continue;
                        }
                        let kp = k.rem_euclid(2) as usize;
                        for n in 0..=(rho2 - sq) {
                            for par in 0..2 {
                                let c = s[j + 1][n][par];
                                if c != 0 {
                                    s[j][n + sq][par ^ kp] += c;
                                }
                            }
                        }
                    }
                }
                s
            })
            .collect();
        let mut me = Self {
            set,
            rho2,
            n_tie: region.n_tie,
            size: region.size,
            cum_below,
            suffix,
            n_total,
            max_abs: 0,
            hist,
        };
        let mut max_abs = 0i32;
        for y in me.set.enumerate_below(rho2) {
            if me.rank(&y).1 {
                max_abs = max_abs.max(y.iter().map(|v| v.abs()).max().unwrap_or(0));
            }
        }
        me.max_abs = max_abs;
        me
    }

    /// Entries on the shell of norm² `t`: all of them below the boundary, the
    /// tie-break's share on it.
    fn shell_entries(&self, t: usize) -> u64 {
        if t < self.rho2 {
            self.hist[t]
        } else {
            self.n_tie
        }
    }

    /// `(rank, inside)`: the (norm, lex) rank among members of norm² ≤ rho2,
    /// and whether that rank is one of the region's 2^w. Panics on a
    /// non-member or on a norm beyond the shell: neither can come out of the
    /// encoder, and silently ranking one would be a wrong count.
    fn rank(&self, y: &[i32; SECTION]) -> (u64, bool) {
        assert!(self.set.contains(y), "not a member of the section set: {y:?}");
        let n: usize = y.iter().map(|&v| (v * v) as usize).sum();
        assert!(n <= self.rho2, "norm² {n} beyond the boundary shell {}: {y:?}", self.rho2);
        let on_shell = self.lex_rank_on_shell(y, n);
        let r = self.cum_below[n] + on_shell;
        let inside = n < self.rho2 || on_shell < self.n_tie;
        debug_assert!(inside == (r < self.size));
        (r, inside)
    }

    /// Members of exact norm² `n` lexicographically before `y`, summed per
    /// pattern — never pooled (see `SectionSet::in_region`).
    fn lex_rank_on_shell(&self, y: &[i32; SECTION], n: usize) -> u64 {
        let mut rank = 0u64;
        for (pi, &pattern) in self.set.patterns.iter().enumerate() {
            let mut norm = 0usize;
            let mut kpar = 0u32;
            for (j, &yj) in y.iter().enumerate() {
                let base = self.set.p as i32 + 2 * ((pattern >> j & 1) as i32);
                let reach = ((n - norm) as f64).sqrt().floor() as i32;
                let lo = (-reach - base).div_euclid(4);
                let hi = (reach - base).div_euclid(4) + 1;
                for k in lo..=hi {
                    let v = base + 4 * k;
                    if v >= yj {
                        continue;
                    }
                    let sq = (v * v) as usize;
                    if norm + sq > n {
                        continue;
                    }
                    let row = &self.suffix[pi][j + 1][n - norm - sq];
                    rank += match self.set.k_parity {
                        Some(r) => row[(r ^ kpar ^ (k.rem_euclid(2) as u32)) as usize],
                        None => row[0] + row[1],
                    };
                }
                let d = yj - base;
                if d.rem_euclid(4) != 0 {
                    break;
                }
                norm += (yj * yj) as usize;
                kpar ^= d.div_euclid(4).rem_euclid(2) as u32;
                if norm > n {
                    break;
                }
            }
        }
        rank
    }
}

/// The ranker must agree with the enumerated, sorted region on every member,
/// and reject every point of the shell the tie-break leaves out.
fn check_ranker(label: &str, set: &SectionSet, w: u32) {
    let rk = Ranker::new(set.clone(), w);
    let pts = set.region_points(w);
    assert_eq!(pts.len() as u64, rk.size);
    for (i, y) in pts.iter().enumerate() {
        let (r, inside) = rk.rank(y);
        assert!(inside && r == i as u64, "{label} w={w}: point {i} ranked {r} inside={inside}");
    }
    let inside: HashSet<[i32; SECTION]> = pts.into_iter().collect();
    let mut outside = 0usize;
    for y in set.enumerate_below(rk.rho2) {
        if !inside.contains(&y) {
            let (r, ins) = rk.rank(&y);
            assert!(!ins && r >= rk.size && r < rk.n_total, "{label} w={w}: shell point {y:?} ranked {r}");
            outside += 1;
        }
    }
    assert!(outside > 0, "{label} w={w}: the tie-break is untested");
    println!("  contrôle rang ↔ region_points : {label} w={w} — {} points, {outside} exclus de coquille, OK", rk.size);
}

// ---------------------------------------------------------------------------
// Orbits under the 256 coordinate sign flips (same canonical form as f1table)
// ---------------------------------------------------------------------------

fn flip(y: &[i32; SECTION], eps: u32) -> [i32; SECTION] {
    let mut z = *y;
    for (j, v) in z.iter_mut().enumerate() {
        if eps >> j & 1 == 1 {
            *v = -*v;
        }
    }
    z
}

fn signature(set: &SectionSet) -> Vec<[i32; SECTION]> {
    let mut v = set.enumerate_below(T_SIG);
    v.sort_unstable();
    v
}

fn flip_sig(sig: &[[i32; SECTION]], eps: u32) -> Vec<[i32; SECTION]> {
    let mut img: Vec<[i32; SECTION]> = sig.iter().map(|y| flip(y, eps)).collect();
    img.sort_unstable();
    img
}

fn canonical(sig: &[[i32; SECTION]]) -> Vec<[i32; SECTION]> {
    (0u32..256).map(|eps| flip_sig(sig, eps)).min().expect("256 flips")
}

/// A family of regions (the 512 ends or the 16 middles) reduced to orbits.
struct Family {
    name: &'static str,
    w: u32,
    /// Per region: its orbit and the ε carrying it onto the representative.
    orbit_of: Vec<usize>,
    eps_of: Vec<u32>,
    /// Per orbit: the representative's ranker, and how many regions it holds.
    rankers: Vec<Ranker>,
    orbit_size: Vec<usize>,
    /// Per region: its own ranker, to check that the encoder's point is inside.
    own: Vec<Ranker>,
}

impl Family {
    fn new(name: &'static str, sets: Vec<SectionSet>, w: u32) -> Self {
        let sigs: Vec<Vec<[i32; SECTION]>> = sets.iter().map(signature).collect();
        let mut key_to_orbit: HashMap<Vec<[i32; SECTION]>, usize> = HashMap::new();
        let mut rep_sig: Vec<Vec<[i32; SECTION]>> = Vec::new();
        let mut rankers = Vec::new();
        let mut orbit_size = Vec::new();
        let mut orbit_of = Vec::with_capacity(sets.len());
        let mut eps_of = Vec::with_capacity(sets.len());
        for (i, set) in sets.iter().enumerate() {
            assert!(!sigs[i].is_empty(), "{name}: region {i} has an empty signature");
            let key = canonical(&sigs[i]);
            let o = match key_to_orbit.get(&key) {
                Some(&o) => o,
                None => {
                    let o = rep_sig.len();
                    key_to_orbit.insert(key, o);
                    rep_sig.push(sigs[i].clone());
                    rankers.push(Ranker::new(set.clone(), w));
                    orbit_size.push(0);
                    o
                }
            };
            orbit_size[o] += 1;
            // The smallest ε whose image of this region is the representative.
            // Several may exist (the stabilizer); the kernel would pin one per
            // region, and the smallest is a deterministic pin.
            let eps = (0u32..256)
                .find(|&e| flip_sig(&sigs[i], e) == rep_sig[o])
                .expect("same canonical form, so some flip maps one onto the other");
            // An even flip: an odd one would carry a 4·E₈ coset onto a coset of
            // the other E₈, which is no region of this family.
            assert_eq!(eps.count_ones() % 2, 0, "{name}: region {i} needs an odd flip");
            orbit_of.push(o);
            eps_of.push(eps);
        }
        let own = sets.iter().map(|s| Ranker::new(s.clone(), w)).collect();
        Self { name, w, orbit_of, eps_of, rankers, orbit_size, own }
    }

    /// Table entry of the lookup `(region, y)`: rank of ε·y in the
    /// representative's region. Also returns whether that rank is one of the
    /// 2^w (a boundary-shell tie can fall outside under ε), and the own-region
    /// rank for the entropy comparison.
    fn entry(&self, region: usize, y: &[i32; SECTION]) -> (usize, u64, bool, u64) {
        let (own_rank, own_inside) = self.own[region].rank(y);
        assert!(own_inside, "{}: the encoder's point is outside its own region {region}: {y:?}", self.name);
        let o = self.orbit_of[region];
        let z = flip(y, self.eps_of[region]);
        let (r, inside) = self.rankers[o].rank(&z);
        (o, r, inside, own_rank)
    }
}

// ---------------------------------------------------------------------------
// From a 24-point in trio order back to its path: (p, r, s8, mid, δ, s16, r_out)
// ---------------------------------------------------------------------------

struct Path {
    p: u32,
    r: u32,
    s8: usize,
    s16: usize,
    r_out: u32,
}

struct Decomposer<'a> {
    t: &'a Trellis,
    words: HashSet<u32>,
    state_of_prefix: HashMap<u8, usize>,
    state_of_suffix: HashMap<u8, usize>,
    mset_of: Vec<usize>,
}

impl<'a> Decomposer<'a> {
    fn new(t: &'a Trellis, mset_of: Vec<usize>) -> Self {
        let words: HashSet<u32> = t.code.words.iter().copied().collect();
        let mut state_of_prefix = HashMap::new();
        let mut state_of_suffix = HashMap::new();
        for s in 0..GOLAY_STATES {
            for &b in &t.prefixes[s] {
                assert!(state_of_prefix.insert(b, s).is_none(), "prefix byte in two states");
            }
            for &b in &t.suffixes[s] {
                assert!(state_of_suffix.insert(b, s).is_none(), "suffix byte in two states");
            }
        }
        assert_eq!(state_of_prefix.len(), 128);
        assert_eq!(state_of_suffix.len(), 128);
        Self { t, words, state_of_prefix, state_of_suffix, mset_of }
    }

    fn decompose(&self, y: &[i32; 24]) -> Path {
        let p = y[0].rem_euclid(2) as u32;
        let mut c = 0u32;
        let mut k = [0i32; 24];
        for (j, &v) in y.iter().enumerate() {
            let d = v - p as i32;
            assert_eq!(d.rem_euclid(2), 0, "mixed parities in {y:?}");
            let half = d.div_euclid(2);
            let cj = half.rem_euclid(2);
            c |= (cj as u32) << j;
            k[j] = (d - 2 * cj).div_euclid(4);
            assert_eq!(p as i32 + 2 * cj + 4 * k[j], v);
        }
        assert!(self.words.contains(&c), "{c:#08x} is not a trio-ordered Golay word");
        let s8 = self.state_of_prefix[&((c & 0xff) as u8)];
        let s16 = self.state_of_suffix[&((c >> 16 & 0xff) as u8)];
        let mid = (c >> 8 & 0xff) as u8;
        assert!(
            self.t.branches[s8].contains(&(mid, s16 as u8)),
            "middle byte {mid:#04x} does not join states {s8} → {s16}"
        );
        let par = |lo: usize, hi: usize| -> u32 {
            k[lo..hi].iter().map(|&kk| kk.rem_euclid(2) as u32).fold(0, |a, b| a ^ b)
        };
        let (r, delta, r3) = (par(0, 8), par(8, 16), par(16, 24));
        let r_out = (p ^ r ^ delta) & 1;
        assert_eq!(r3, r_out, "section 3 does not close the block on Σk ≡ p");
        Path { p, r, s8, s16, r_out }
    }

    fn end_region_1(&self, path: &Path) -> usize {
        ((path.p * 2 + path.r) as usize) * GOLAY_STATES + path.s8
    }
    fn end_region_3(&self, path: &Path) -> usize {
        4 * GOLAY_STATES + ((path.p * 2 + path.r_out) as usize) * GOLAY_STATES + path.s16
    }
    fn mid_region(&self, path: &Path) -> usize {
        path.p as usize * 8 + self.mset_of[path.s8]
    }
}

// ---------------------------------------------------------------------------
// Counting
// ---------------------------------------------------------------------------

/// One lookup: which family (0 = ends, 1 = middle), which section (1, 2, 3),
/// orbit, entry, and the half of the sample the block belongs to.
#[derive(Clone, Copy)]
struct Lookup {
    family: u8,
    section: u8,
    orbit: u16,
    entry: u32,
    half: u8,
    inside: bool,
    own_entry: u32,
    norm2: u32,
}

type Key = (u8, u16, u32);

fn counts(lookups: &[Lookup], keep: impl Fn(&Lookup) -> bool) -> HashMap<Key, u64> {
    let mut m: HashMap<Key, u64> = HashMap::new();
    for l in lookups.iter().filter(|l| keep(l)) {
        *m.entry((l.family, l.orbit, l.entry)).or_insert(0) += 1;
    }
    m
}

fn entropy_bits(m: &HashMap<Key, u64>) -> (f64, u64, usize) {
    let n: u64 = m.values().sum();
    let h = m
        .values()
        .map(|&c| {
            let p = c as f64 / n as f64;
            -p * p.log2()
        })
        .sum::<f64>();
    (h, n, m.len())
}

const KIB_POINTS: [usize; 14] = [2, 4, 8, 12, 16, 24, 28, 32, 48, 64, 96, 128, 256, 512];

/// Coverage at each KiB threshold when entries are taken in the order `order`
/// (hottest first by the SELECTION counts), scored with the EVALUATION counts.
/// `cost` gives the bytes of one entry.
fn coverage_curve(
    order: &[Key],
    eval: &HashMap<Key, u64>,
    cost: impl Fn(&Key) -> usize,
) -> Vec<(usize, f64)> {
    let total: u64 = eval.values().sum();
    let mut out = Vec::new();
    let mut bytes = 0usize;
    let mut mass = 0u64;
    let mut i = 0usize;
    for &kib in &KIB_POINTS {
        let limit = kib * 1024;
        while i < order.len() && bytes + cost(&order[i]) <= limit {
            bytes += cost(&order[i]);
            mass += eval.get(&order[i]).copied().unwrap_or(0);
            i += 1;
        }
        out.push((kib, mass as f64 / total as f64));
    }
    out
}

fn hottest_first(sel: &HashMap<Key, u64>) -> Vec<Key> {
    let mut v: Vec<(Key, u64)> = sel.iter().map(|(&k, &c)| (k, c)).collect();
    // Ties broken by key so the order is deterministic.
    v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    v.into_iter().map(|(k, _)| k).collect()
}

fn print_curve(label: &str, rows: &[(usize, f64)], n_eval: u64) {
    print!("  {label:<34}");
    for (_, c) in rows {
        print!(" {:>5.1}", 100.0 * c);
    }
    println!("   (n = {n_eval})");
}

fn binom_pm(c: f64, n: u64) -> f64 {
    (c * (1.0 - c) / n as f64).sqrt()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n_blocks: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(60_000);
    let minutes_cap: f64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(45.0);
    let nt: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or_else(|| {
        std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).saturating_sub(4).max(1)
    });
    let t_start = Instant::now();

    println!("F1 — distribution d'accès réelle de la table du décodeur, découpe {}/{}/{}", W[0], W[1], W[2]);
    println!("graine {SEED:#x}, {N_TRAIN_SKIP} blocs d'entraînement sautés, {n_blocks} blocs d'évaluation demandés, plafond {minutes_cap:.0} min\n");

    // ---- the codebook F1b measured, and the trellis behind it ----
    let cb = Codebook::new(W);
    let t = &cb.trellis;
    let scales: Vec<f64> = (0..18).map(|i| 0.10 * 1.14f64.powi(i)).collect();

    // The eight middle-byte sets, in the codebook's own order of first appearance.
    let mut msets: Vec<Vec<u8>> = Vec::new();
    let mut mset_of = vec![0usize; GOLAY_STATES];
    for (s8, slot) in mset_of.iter_mut().enumerate() {
        let mut m: Vec<u8> = t.branches[s8].iter().map(|&(b, _)| b).collect();
        m.sort_unstable();
        *slot = match msets.iter().position(|s| *s == m) {
            Some(i) => i,
            None => {
                msets.push(m);
                msets.len() - 1
            }
        };
    }
    assert_eq!(msets.len(), 8);

    // ---- the regions: 256 of section 1, 256 of section 3, 16 of section 2 ----
    let mut ends: Vec<SectionSet> = Vec::with_capacity(512);
    for p in 0..2u32 {
        for r in 0..2u32 {
            for s in 0..GOLAY_STATES {
                ends.push(t.section1(s, p, r));
            }
        }
    }
    for p in 0..2u32 {
        for r in 0..2u32 {
            for s in 0..GOLAY_STATES {
                ends.push(SectionSet { patterns: t.suffixes[s].to_vec(), p, k_parity: Some(r) });
            }
        }
    }
    let mut mids: Vec<SectionSet> = Vec::with_capacity(16);
    for p in 0..2u32 {
        for m in &msets {
            mids.push(SectionSet { patterns: m.clone(), p, k_parity: None });
        }
    }

    println!("contrôles du rang (le rang recalculé contre la région énumérée et triée) :");
    check_ranker("section 1 (0,0,0)", &ends[0], 8);
    check_ranker("section 1 (0,0,0)", &ends[0], W[0]);
    check_ranker("section 1 (1,1,7)", &ends[3 * GOLAY_STATES + 7], W[0]);
    check_ranker("section 3 (1,0,40)", &ends[4 * GOLAY_STATES + 2 * GOLAY_STATES + 40], W[2]);
    check_ranker("section 2 (p=0, m=0)", &mids[0], 10);
    check_ranker("section 2 (p=0, m=3)", &mids[3], W[1]);
    check_ranker("section 2 (p=1, m=5)", &mids[8 + 5], W[1]);

    let t_fam = Instant::now();
    let fam_end = Family::new("extrémités", ends, W[0]);
    let fam_mid = Family::new("milieu", mids, W[1]);
    assert_eq!(fam_end.rankers.len(), 67, "the end family does not give 67 orbits");
    assert_eq!(fam_mid.rankers.len(), 9, "the middle family does not give 9 orbits");
    let mut sz_end = fam_end.orbit_size.clone();
    sz_end.sort_unstable_by(|a, b| b.cmp(a));
    let mut sz_mid = fam_mid.orbit_size.clone();
    sz_mid.sort_unstable_by(|a, b| b.cmp(a));
    println!(
        "\norbites : extrémités {} (tailles {:?}…), milieu {} (tailles {:?}) — {:.1} s",
        fam_end.rankers.len(),
        &sz_end[..8],
        fam_mid.rankers.len(),
        sz_mid,
        t_fam.elapsed().as_secs_f64()
    );
    let rho_end: Vec<usize> = fam_end.rankers.iter().map(|r| r.rho2).collect();
    let rho_mid: Vec<usize> = fam_mid.rankers.iter().map(|r| r.rho2).collect();
    println!(
        "coquille frontière ρ² : extrémités {}..{}, milieu {}..{} ; max |coord| : extrémités {}, milieu {}",
        rho_end.iter().min().unwrap(),
        rho_end.iter().max().unwrap(),
        rho_mid.iter().min().unwrap(),
        rho_mid.iter().max().unwrap(),
        fam_end.rankers.iter().map(|r| r.max_abs).max().unwrap(),
        fam_mid.rankers.iter().map(|r| r.max_abs).max().unwrap()
    );
    let table_end = fam_end.rankers.len() * (1usize << W[0]) * ENTRY_BYTES;
    let table_mid = fam_mid.rankers.len() * (1usize << W[1]) * ENTRY_BYTES;
    println!(
        "table à {ENTRY_BYTES} o/entrée : extrémités {:.0} Kio + milieu {:.0} Kio = {:.0} Kio",
        table_end as f64 / 1024.0,
        table_mid as f64 / 1024.0,
        (table_end + table_mid) as f64 / 1024.0
    );

    let dec = Decomposer::new(t, mset_of.clone());

    // ---- the blocks: exactly F1b's evaluation blocks ----
    let mut rng = SplitMix64::new(SEED);
    for _ in 0..N_TRAIN_SKIP {
        let _ = gauss_block(&mut rng);
    }
    let blocks: Vec<[f64; 24]> = (0..n_blocks).map(|_| gauss_block(&mut rng)).collect();

    // ---- encode, in parallel, under a wall-clock cap ----
    println!("\nencodage sur {nt} threads…");
    let next = AtomicUsize::new(0);
    let stop = AtomicBool::new(false);
    let deadline_s = minutes_cap * 60.0;
    let results: Mutex<Vec<(usize, [i32; 24])>> = Mutex::new(Vec::with_capacity(n_blocks));
    let t_enc = Instant::now();
    std::thread::scope(|sc| {
        for _ in 0..nt {
            sc.spawn(|| {
                let mut local: Vec<(usize, [i32; 24])> = Vec::new();
                loop {
                    if stop.load(Ordering::Relaxed) {
                        break;
                    }
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= n_blocks {
                        break;
                    }
                    let (_, y) = cb.best_t(&blocks[i], &scales);
                    local.push((i, y));
                    if t_enc.elapsed().as_secs_f64() > deadline_s {
                        stop.store(true, Ordering::Relaxed);
                    }
                    if i.is_multiple_of(5_000) && i > 0 {
                        eprintln!("  {i} blocs, {:.1} min", t_enc.elapsed().as_secs_f64() / 60.0);
                    }
                }
                results.lock().unwrap().extend(local);
            });
        }
    });
    let mut encoded = results.into_inner().unwrap();
    encoded.sort_by_key(|&(i, _)| i);
    let n = encoded.len();
    let enc_min = t_enc.elapsed().as_secs_f64() / 60.0;
    println!(
        "{n} blocs encodés en {enc_min:.1} min ({:.3} s/bloc·thread){}",
        t_enc.elapsed().as_secs_f64() * nt as f64 / n as f64,
        if n < n_blocks { " — ARRÊTÉ AU PLAFOND" } else { "" }
    );

    // ---- decompose and count ----
    let mut lookups: Vec<Lookup> = Vec::with_capacity(3 * n);
    let mut p1 = 0usize;
    let mut outside = [0u64; 2];
    let mut norm2_sum = [0f64; 3];
    for &(i, y) in &encoded {
        let path = dec.decompose(&y);
        p1 += path.p as usize;
        let half = (i % 2) as u8;
        let y1: [i32; SECTION] = y[..8].try_into().unwrap();
        let y2: [i32; SECTION] = y[8..16].try_into().unwrap();
        let y3: [i32; SECTION] = y[16..].try_into().unwrap();
        for (sec, fam, region, ys) in [
            (1u8, &fam_end, dec.end_region_1(&path), &y1),
            (2u8, &fam_mid, dec.mid_region(&path), &y2),
            (3u8, &fam_end, dec.end_region_3(&path), &y3),
        ] {
            let (o, e, inside, own) = fam.entry(region, ys);
            let family = if sec == 2 { 1u8 } else { 0u8 };
            if !inside {
                outside[family as usize] += 1;
            }
            norm2_sum[sec as usize - 1] += ys.iter().map(|&v| (v * v) as f64).sum::<f64>();
            lookups.push(Lookup {
                family,
                section: sec,
                orbit: o as u16,
                entry: e as u32,
                half,
                inside,
                own_entry: own as u32,
                norm2: ys.iter().map(|&v| (v * v) as u32).sum(),
            });
        }
    }
    let n_look = lookups.len() as u64;
    println!(
        "\n{n_look} consultations ({n} blocs × 3) ; p = 1 sur {:.1} % des blocs ; ‖y‖² moyen par section 1/2/3 : {:.1} / {:.1} / {:.1}",
        100.0 * p1 as f64 / n as f64,
        norm2_sum[0] / n as f64,
        norm2_sum[1] / n as f64,
        norm2_sum[2] / n as f64
    );
    println!(
        "images ε·y tombées HORS de la région représentante (égalité de coquille non équivariante) : extrémités {} / {}, milieu {} / {}",
        outside[0],
        2 * n,
        outside[1],
        n
    );

    // ---- orbit shares: measured against the region-count shares f1table assumed ----
    println!("\nparts de consultations par orbite (mesurées) contre parts de régions (f1table) :");
    for (fam, family, nl) in [(&fam_end, 0u8, 2 * n as u64), (&fam_mid, 1u8, n as u64)] {
        let mut per: Vec<(usize, u64)> = vec![(0, 0); fam.rankers.len()];
        for (o, slot) in per.iter_mut().enumerate() {
            slot.0 = o;
        }
        for l in lookups.iter().filter(|l| l.family == family) {
            per[l.orbit as usize].1 += 1;
        }
        per.sort_by_key(|a| std::cmp::Reverse(a.1));
        let total_regions: usize = fam.orbit_size.iter().sum();
        print!("  {:<12}", fam.name);
        for &(o, c) in per.iter().take(6) {
            print!(
                " [orb {o}: {:.1} % mes. / {:.1} % rég., {} rég., max|c| {}]",
                100.0 * c as f64 / nl as f64,
                100.0 * fam.orbit_size[o] as f64 / total_regions as f64,
                fam.orbit_size[o],
                fam.rankers[o].max_abs
            );
        }
        let top2: u64 = per.iter().take(2).map(|x| x.1).sum();
        let top1: u64 = per[0].1;
        println!("\n               2 orbites les plus chaudes : {:.1} % ; 1 : {:.1} %", 100.0 * top2 as f64 / nl as f64, 100.0 * top1 as f64 / nl as f64);
    }

    // ---- entropies ----
    println!("\nentropie empirique de l'index d'entrée (orbite, entrée), bits — plug-in, biaisée vers le bas, plafond log2(n) :");
    for (label, keep) in [
        ("section 1", Box::new(|l: &Lookup| l.section == 1) as Box<dyn Fn(&Lookup) -> bool>),
        ("section 2 (milieu)", Box::new(|l: &Lookup| l.section == 2)),
        ("section 3", Box::new(|l: &Lookup| l.section == 3)),
        ("extrémités (1+3)", Box::new(|l: &Lookup| l.family == 0)),
    ] {
        let m = counts(&lookups, &keep);
        let (h, nn, distinct) = entropy_bits(&m);
        let mm = (distinct as f64 - 1.0) / (2.0 * nn as f64 * std::f64::consts::LN_2);
        // Own-region rank, for comparison: same bits if the orbit pooling is a symmetry.
        let mut own: HashMap<Key, u64> = HashMap::new();
        for l in lookups.iter().filter(|l| keep(l)) {
            *own.entry((l.section, l.orbit, l.own_entry)).or_insert(0) += 1;
        }
        let (h_own, _, d_own) = entropy_bits(&own);
        println!(
            "  {label:<20} H = {h:>6.3} (+Miller-Madow {:.3}) sur n = {nn}, plafond {:.2} ; entrées distinctes {distinct} ; rang propre : H {h_own:.3}, {d_own} distinctes",
            h + mm,
            (nn as f64).log2()
        );
    }
    let uniform_bits = |fam: &Family| (fam.rankers.len() as f64).log2() + fam.w as f64;
    println!(
        "  (uniforme sur la table : extrémités {:.2} bits, milieu {:.2} bits)",
        uniform_bits(&fam_end),
        uniform_bits(&fam_mid)
    );

    // ---- CDF vs table bytes ----
    let cost6 = |_: &Key| ENTRY_BYTES;
    let header = || {
        print!("  {:<34}", "Kio →");
        for k in KIB_POINTS {
            print!(" {k:>5}");
        }
        println!();
    };
    println!("\nCDF des consultations (%) contre les octets de table cumulés, entrées les plus chaudes d'abord, {ENTRY_BYTES} o/entrée");
    println!("IN-SAMPLE (jeu chaud choisi et noté sur les mêmes consultations : borne HAUTE) :");
    header();
    let all_full = counts(&lookups, |_| true);
    let end_full = counts(&lookups, |l| l.family == 0);
    let mid_full = counts(&lookups, |l| l.family == 1);
    let in_all = coverage_curve(&hottest_first(&all_full), &all_full, cost6);
    let in_end = coverage_curve(&hottest_first(&end_full), &end_full, cost6);
    let in_mid = coverage_curve(&hottest_first(&mid_full), &mid_full, cost6);
    print_curve("toutes", &in_all, n_look);
    print_curve("extrémités", &in_end, 2 * n as u64);
    print_curve("milieu", &in_mid, n as u64);

    println!("HELD-OUT (jeu chaud choisi sur les blocs pairs, noté sur les impairs, et réciproquement ; moyenne) :");
    header();
    let mut held: Vec<Vec<(usize, f64)>> = Vec::new();
    let mut held_n: Vec<u64> = Vec::new();
    for (label, keep) in [
        ("toutes", Box::new(|_: &Lookup| true) as Box<dyn Fn(&Lookup) -> bool>),
        ("extrémités", Box::new(|l: &Lookup| l.family == 0)),
        ("milieu", Box::new(|l: &Lookup| l.family == 1)),
    ] {
        let a = counts(&lookups, |l| keep(l) && l.half == 0);
        let b = counts(&lookups, |l| keep(l) && l.half == 1);
        let ab = coverage_curve(&hottest_first(&a), &b, cost6);
        let ba = coverage_curve(&hottest_first(&b), &a, cost6);
        let avg: Vec<(usize, f64)> = ab.iter().zip(&ba).map(|(x, y)| (x.0, 0.5 * (x.1 + y.1))).collect();
        let n_eval: u64 = a.values().sum::<u64>() + b.values().sum::<u64>();
        print_curve(label, &avg, n_eval);
        held.push(avg);
        held_n.push(n_eval);
    }
    let c_at = |rows: &[(usize, f64)], kib: usize| rows.iter().find(|r| r.0 == kib).map(|r| r.1).unwrap();
    println!(
        "  ± binomial 1σ sur « toutes » à 16 Kio : held-out ±{:.2} pt, in-sample ±{:.2} pt",
        100.0 * binom_pm(c_at(&held[0], 16), held_n[0]),
        100.0 * binom_pm(c_at(&in_all, 16), n_look)
    );

    // ---- 4-byte entries ----
    println!("\nentrées à 4 o (max |coord| ≤ 7, 4 bits signés par coordonnée) :");
    for (fam, family, nl) in [(&fam_end, 0u8, 2 * n as u64), (&fam_mid, 1u8, n as u64)] {
        let qual: Vec<usize> = (0..fam.rankers.len()).filter(|&o| fam.rankers[o].max_abs <= 7).collect();
        let share: u64 = lookups.iter().filter(|l| l.family == family && qual.contains(&(l.orbit as usize))).count() as u64;
        println!(
            "  {:<12} {} orbites sur {} qualifient, portant {:.1} % des consultations de la famille",
            fam.name,
            qual.len(),
            fam.rankers.len(),
            100.0 * share as f64 / nl as f64
        );
    }
    let qualifies = |k: &Key| -> bool {
        let fam = if k.0 == 0 { &fam_end } else { &fam_mid };
        fam.rankers[k.1 as usize].max_abs <= 7
    };
    let cost_mixed = |k: &Key| if qualifies(k) { 4 } else { ENTRY_BYTES };
    let cost4 = |_: &Key| 4usize;
    let order_all = hottest_first(&all_full);
    let mixed = coverage_curve(&order_all, &all_full, cost_mixed);
    let four = coverage_curve(&order_all, &all_full, cost4);
    let a = counts(&lookups, |l| l.half == 0);
    let b = counts(&lookups, |l| l.half == 1);
    let ho4: Vec<(usize, f64)> = coverage_curve(&hottest_first(&a), &b, cost4)
        .iter()
        .zip(&coverage_curve(&hottest_first(&b), &a, cost4))
        .map(|(x, y)| (x.0, 0.5 * (x.1 + y.1)))
        .collect();
    println!(
        "  couverture « toutes » à 8 / 16 Kio : 6 o {:.1} / {:.1} % ; 4 o où l'orbite qualifie, 6 sinon {:.1} / {:.1} % ; 4 o partout (hypothèse) in-sample {:.1} / {:.1} %, held-out {:.1} / {:.1} %",
        100.0 * c_at(&in_all, 8),
        100.0 * c_at(&in_all, 16),
        100.0 * c_at(&mixed, 8),
        100.0 * c_at(&mixed, 16),
        100.0 * c_at(&four, 8),
        100.0 * c_at(&four, 16),
        100.0 * c_at(&ho4, 8),
        100.0 * c_at(&ho4, 16)
    );
    // The hot set's own coordinate range: could a separate hot table be 4 B?
    let mut bytes = 0usize;
    let mut hot_max = 0i32;
    let mut member_cache: HashMap<(u8, u16), Vec<[i32; SECTION]>> = HashMap::new();
    for k in &order_all {
        if bytes + ENTRY_BYTES > 16 * 1024 {
            break;
        }
        bytes += ENTRY_BYTES;
        let fam = if k.0 == 0 { &fam_end } else { &fam_mid };
        let rk = &fam.rankers[k.1 as usize];
        let members = member_cache.entry((k.0, k.1)).or_insert_with(|| {
            let mut pts = rk.set.enumerate_below(rk.rho2);
            pts.sort_by_key(|y| (y.iter().map(|&v| (v as i64) * (v as i64)).sum::<i64>(), *y));
            pts
        });
        let y = members[k.2 as usize];
        debug_assert_eq!(rk.rank(&y).0, k.2 as u64);
        hot_max = hot_max.max(y.iter().map(|v| v.abs()).max().unwrap());
    }
    println!("  max |coord| sur les entrées du jeu chaud de 16 Kio (in-sample) : {hot_max}");

    // ---- sector granularity: 32 B sectors of a rank-ordered table ----
    println!("\nCDF « toutes » à la granularité du secteur de 32 o (table rangée par rang, {ENTRY_BYTES} o/entrée ; ce que la L1 remplit vraiment) :");
    header();
    let sector_key = |k: &Key| -> Key { (k.0, k.1, (k.2 as usize * ENTRY_BYTES / 32) as u32) };
    let regroup = |m: &HashMap<Key, u64>| -> HashMap<Key, u64> {
        let mut s: HashMap<Key, u64> = HashMap::new();
        for (k, &c) in m {
            *s.entry(sector_key(k)).or_insert(0) += c;
        }
        s
    };
    let cost32 = |_: &Key| 32usize;
    let sa = regroup(&all_full);
    let sec_in = coverage_curve(&hottest_first(&sa), &sa, cost32);
    let (sa_a, sa_b) = (regroup(&a), regroup(&b));
    let sec_ho: Vec<(usize, f64)> = coverage_curve(&hottest_first(&sa_a), &sa_b, cost32)
        .iter()
        .zip(&coverage_curve(&hottest_first(&sa_b), &sa_a, cost32))
        .map(|(x, y)| (x.0, 0.5 * (x.1 + y.1)))
        .collect();
    print_curve("secteurs, in-sample", &sec_in, n_look);
    print_curve("secteurs, held-out", &sec_ho, n_look);

    // ---- shell-smoothed hot set: shells of one norm² inside one orbit table ----
    // A shell is a contiguous run of ranks, its entry count is exact (the
    // histogram), and the selection statistic (lookups per entry of the shell)
    // averages over hundreds of entries — far less noisy than a per-entry
    // count. Under the ASSUMPTION that lookups are uniform within a shell the
    // partial fill is proportional; the held-out score of that selection is an
    // unbiased coverage of THAT hot set whatever the assumption, and a lower
    // bound on the best hot set's.
    println!("\nCDF « toutes » avec un jeu chaud choisi PAR COQUILLE (orbite, ‖y‖²), densité décroissante, remplissage partiel proportionnel :");
    header();
    let shell_counts = |keep: &dyn Fn(&Lookup) -> bool| -> HashMap<Key, u64> {
        let mut m: HashMap<Key, u64> = HashMap::new();
        for l in lookups.iter().filter(|l| keep(l)) {
            *m.entry((l.family, l.orbit, l.norm2)).or_insert(0) += 1;
        }
        m
    };
    let entries_of = |k: &Key| -> u64 {
        let fam = if k.0 == 0 { &fam_end } else { &fam_mid };
        fam.rankers[k.1 as usize].shell_entries(k.2 as usize)
    };
    let shell_curve = |sel: &HashMap<Key, u64>, eval: &HashMap<Key, u64>, per_entry: usize| -> Vec<(usize, f64)> {
        let total: u64 = eval.values().sum();
        let mut order: Vec<(Key, f64)> =
            sel.iter().map(|(&k, &c)| (k, c as f64 / entries_of(&k) as f64)).collect();
        order.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        let mut out = Vec::new();
        for &kib in &KIB_POINTS {
            let limit = (kib * 1024) as f64;
            let mut bytes = 0.0;
            let mut mass = 0.0;
            for (k, _) in &order {
                let cost = (entries_of(k) as usize * per_entry) as f64;
                let c = eval.get(k).copied().unwrap_or(0) as f64;
                if bytes + cost <= limit {
                    bytes += cost;
                    mass += c;
                } else {
                    mass += c * (limit - bytes) / cost;
                    break;
                }
            }
            out.push((kib, mass / total as f64));
        }
        out
    };
    let sh_all = shell_counts(&|_| true);
    let sh_a = shell_counts(&|l| l.half == 0);
    let sh_b = shell_counts(&|l| l.half == 1);
    let sh_in = shell_curve(&sh_all, &sh_all, ENTRY_BYTES);
    let sh_ho: Vec<(usize, f64)> = shell_curve(&sh_a, &sh_b, ENTRY_BYTES)
        .iter()
        .zip(&shell_curve(&sh_b, &sh_a, ENTRY_BYTES))
        .map(|(x, y)| (x.0, 0.5 * (x.1 + y.1)))
        .collect();
    print_curve("coquilles, in-sample", &sh_in, n_look);
    print_curve("coquilles, held-out", &sh_ho, n_look);
    let sh_ho4: Vec<(usize, f64)> = shell_curve(&sh_a, &sh_b, 4)
        .iter()
        .zip(&shell_curve(&sh_b, &sh_a, 4))
        .map(|(x, y)| (x.0, 0.5 * (x.1 + y.1)))
        .collect();
    print_curve("coquilles, held-out, 4 o/entrée", &sh_ho4, n_look);
    println!(
        "  coquilles distinctes touchées : {} ; les 10 plus denses :",
        sh_all.len()
    );
    {
        let mut order: Vec<(Key, u64)> = sh_all.iter().map(|(&k, &c)| (k, c)).collect();
        order.sort_by(|a, b| {
            let da = a.1 as f64 / entries_of(&a.0) as f64;
            let db = b.1 as f64 / entries_of(&b.0) as f64;
            db.total_cmp(&da)
        });
        for (k, c) in order.iter().take(10) {
            println!(
                "    {} orbite {:>2} ‖y‖² {:>3} : {:>6} consultations sur {:>5} entrées ({:.1} Kio) → {:.4} % par entrée",
                if k.0 == 0 { "ext." } else { "mil." },
                k.1,
                k.2,
                c,
                entries_of(k),
                entries_of(k) as f64 * ENTRY_BYTES as f64 / 1024.0,
                100.0 * *c as f64 / entries_of(k) as f64 / n_look as f64
            );
        }
    }

    // ---- is the access concentrated at low rank (low norm)? ----
    println!("\npart des consultations d'entrée < f·2^w (préfixe contigu de chaque table, coût f × table entière) :");
    for &(num, den) in &[(1usize, 64usize), (1, 32), (1, 16), (1, 8), (1, 4), (1, 2)] {
        let m = lookups
            .iter()
            .filter(|l| {
                let w = if l.family == 0 { W[0] } else { W[1] };
                (l.entry as u64) * den as u64 <= (num as u64) << w
            })
            .count();
        println!(
            "  f = {num}/{den:<3} → {:.1} % des consultations pour {:.0} Kio",
            100.0 * m as f64 / n_look as f64,
            (num * (table_end + table_mid)) as f64 / den as f64 / 1024.0
        );
    }

    // ---- the number that decides ----
    println!("\nD_real = c8·{D_8K} + (c16 − c8)·{D_16K} + (1 − c16)·{D_4M} ms, contre H = {H_BUDGET} ms (pessimiste au-delà de 16 Kio) :");
    for (label, rows) in [
        ("in-sample (borne haute de c)", &in_all),
        ("held-out (honnête)", &held[0]),
        ("secteurs 32 o, held-out", &sec_ho),
        ("coquilles, held-out", &sh_ho),
        ("coquilles, held-out, 4 o", &sh_ho4),
    ] {
        let (c8, c16) = (c_at(rows, 8), c_at(rows, 16));
        let d = c8 * D_8K + (c16 - c8) * D_16K + (1.0 - c16) * D_4M;
        println!(
            "  {label:<30} c8 = {:.1} %, c16 = {:.1} % → D_real ≈ {d:.3} ms = {:.2} × H",
            100.0 * c8,
            100.0 * c16,
            d / H_BUDGET
        );
    }
    let need_c16_pess = (D_4M - H_BUDGET) / (D_4M - D_16K);
    let need_c16_opt = (D_4M - H_BUDGET) / (D_4M - D_8K);
    println!(
        "  c16 nécessaire pour D_real ≤ H : {:.1} % si c8 = 0, {:.1} % si c8 = c16",
        100.0 * need_c16_pess,
        100.0 * need_c16_opt
    );

    // ---- raw counts, for anyone who wants to recheck ----
    if let Ok(dir) = std::env::var("F1ACCESS_DUMP") {
        let path = format!("{dir}/f1accesscv-counts.csv");
        let mut s = String::from("family,orbit,entry,count_even,count_odd,inside\n");
        let mut ins: HashMap<Key, bool> = HashMap::new();
        for l in &lookups {
            ins.insert((l.family, l.orbit, l.entry), l.inside);
        }
        let mut keys: Vec<&Key> = all_full.keys().collect();
        keys.sort();
        for k in keys {
            s.push_str(&format!(
                "{},{},{},{},{},{}\n",
                k.0,
                k.1,
                k.2,
                a.get(k).copied().unwrap_or(0),
                b.get(k).copied().unwrap_or(0),
                ins[k] as u8
            ));
        }
        std::fs::write(&path, s).expect("write dump");
        println!("\ncomptes bruts écrits dans {path}");
    }

    println!(
        "\ntemps total {:.1} min ({n} blocs, {nt} threads, encodage {enc_min:.1} min)",
        t_start.elapsed().as_secs_f64() / 60.0
    );
}
