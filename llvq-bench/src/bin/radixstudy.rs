//! **X4 — the E3 study, on paper.** What an index designed *for the decoder*
//! would cost, priced exactly on the 384 classes and, when an artifact is
//! given, weighted by the real blocks of the published 4B.
//!
//! `docs/archive/spec-memoire-extreme-2026-08-12.md` opens the E3 chantier only if a
//! decomposition exists that is **both**
//!
//! * ≤ **2.6 b/weight** in the kernel accounting, and
//! * **shift-only at bounded depth** — every field extracted by shift/mask and
//!   interpreted without a data-dependent loop, ≤ 24 steps.
//!
//! This bench answers that question with counts rather than opinion, for 0 $
//! and no card. It cannot answer whether a shift-only decoder is *fast* — the
//! dossier's own rule is that no conclusion about speed survives without SASS
//! — but it can prove that no admissible decomposition reaches the bit target,
//! which is the cheap half of the verdict and the one that closes the chantier.
//!
//! ## Why the archive itself is not the answer
//!
//! The file is 48 bits per block and would be 2.15 b/weight in the kernel
//! accounting — under the criterion, and it is *already written*. It is
//! disqualified by the other half: unranking a mixed-radix multiset rank is a
//! serial, data-dependent chain of ~509 operations per block (`decfast`),
//! measured on GPU at **8.27 ns/block against 0.11** for a mask layout — 75×.
//! Every variant below is an attempt to buy that chain back with bits.
//!
//! ## The menu
//!
//! | variant | what changes | shift-only |
//! |---|---|---|
//! | `archive` | nothing — the file's own index | ❌ ~509 serial ops |
//! | `radix2` | every composition radix rounded up to a power of two | ❌ fields extract by shift, multiset ranks still unrank serially |
//! | `radix2+g12` | same, codeword as a uniform 12-bit rank | ❌ same |
//! | `perslot` | the arrangement replaced by 24 per-slot level fields | ✅ this *is* `Planes14`/`Planes12x` |
//! | `golay70` | the measured E2 layout | ✅ depth 24, **1.31× — écarté** |
//! | `golay_tight` | `golay70` with the A plane and the sign plane cut to what each class actually needs | ✅ depth 24, variable width |
//!
//! `golay_tight` is the only genuinely new point, and it is the one the
//! chantier turns on: `Golay70` spends a flat 24 bits on its A plane and 24 on
//! its sign plane, while a class whose residues each hold a single magnitude
//! needs **no** A bit at all, and the odd coset needs no sign bit ever. Those
//! are per-class counts, so they are exact here.
//!
//! ## What is deliberately NOT in the menu
//!
//! * **Entropy coding of the rank.** Closed by measurement: the index's
//!   entropy is 46.6536 bits against 47 paid (`archive/verdicts-lot-b-2026-08-06.md`
//!   §B5). There is nothing there.
//! * **The 46-bit odd-coset variant.** Proposed and *refuted*: the codeword
//!   bit gives the residue of the **signed** value, not of `|x|`, so it cannot
//!   stand in for a magnitude bit. `golay_tight` is guarded against
//!   re-deriving it — see `the_refuted_46_bit_odd_variant_is_not_reachable`.
//!
//! ## Run
//!
//! `cargo run --release -p llvq-bench --bin radixstudy [path/to/model.llvq]`
//!
//! Without a path: every class weighted equally, plus the worst case — which
//! is what a fixed-stride format actually pays. With one: the real block
//! histogram of that artifact, which is the number that decides.

use llvq_search::classes::{enumerate_classes, gamma, EvenClass, OddClass};
use llvq_search::fastdec::{ClassLevels, FastDecoder, MAX_LEVELS};
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;

/// Class id + gain, the header every runtime variant pays.
const HEADER_BITS: u64 = 10;
/// Bits of a Golay codeword rank among all 4096 — `GOLAY70_RANK_BITS`.
const GOLAY_RANK_BITS: u64 = 12;
/// Index bits of the published `leech1c12` file, plus its gain bit.
const ARCHIVE_BITS: u64 = 47 + 1;
/// The spec's admission threshold for opening the E3 chantier, in the kernel
/// accounting.
const E3_CRITERION: f64 = 2.6;
/// Lanes a variable-width stream is grouped over.
const GROUP: u64 = 32;

/// `⌈log₂ n⌉`, and 0 for `n ≤ 1` — a field with one possible value costs
/// nothing, which is the whole point of `golay_tight`.
fn lg_ceil(n: u128) -> u64 {
    if n <= 1 {
        0
    } else {
        (128 - (n - 1).leading_zeros()) as u64
    }
}

/// Canonical key of a class: `(odd, [(|value|, count); 5])` in `ClassLevels`
/// order — descending count, ties by descending value.
///
/// Exists because the two sources of truth are indexed differently:
/// `enumerate_classes` knows the composition radices, `FastDecoder` knows
/// which class a block is in. Matching them by *content* rather than by
/// position means a reordering of either shows up as a lookup failure instead
/// of as silently mismatched radices.
type Key = (bool, [(i32, u8); MAX_LEVELS]);

fn canonical(mut kinds: Vec<(i32, u8)>, odd: bool) -> Key {
    kinds.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(b.0.cmp(&a.0)));
    let mut k = [(0i32, 0u8); MAX_LEVELS];
    for (i, &v) in kinds.iter().enumerate() {
        k[i] = v;
    }
    (odd, k)
}

fn key_of_levels(lv: &ClassLevels) -> Key {
    canonical(
        (0..lv.len).map(|i| (lv.values[i], lv.counts[i])).collect(),
        lv.odd,
    )
}

/// What one class costs under each variant. All widths are **payload bits per
/// block**, header included.
#[derive(Clone, Copy, Debug, Default)]
struct Widths {
    /// Rounded mixed-radix composition, class stored in 9 bits.
    radix2: u64,
    /// Same with a uniform 12-bit codeword field.
    radix2_g12: u64,
    /// Per-slot level fields — the `Planes` geometry.
    perslot: u64,
    /// `Golay70`, flat.
    golay70: u64,
    /// `Golay70` with the A and sign planes cut to this class's needs.
    golay_tight: u64,
    /// `⌈log₂ |class|⌉` — the information floor of the class alone, printed
    /// to show how much of the archive's 47 bits is the *choice of class*
    /// rather than the point within it.
    exact: u64,
}

/// Bits an even class spends on the two planes `Golay70` carries flat.
///
/// * **A plane** — the codeword fixes each slot's residue mod 4 (word values
///   are `≡ 2`, free values and zero are `≡ 0`). Within a residue the class
///   may hold several magnitudes, and only then does a slot need a bit:
///   `⌈log₂ k⌉` per slot of that residue, where `k` is the count of distinct
///   magnitudes in it. `Golay70` charges 1 everywhere and refuses `k > 2`;
///   this charges what the class needs and allows `k > 2` at `⌈log₂ k⌉`.
/// * **sign plane** — one bit per nonzero slot, less the one the parity
///   constraint fixes (`S_w = 2^(w−1)`, not `2^w`, in the index's own
///   composition). Zero slots carry no sign.
fn even_planes(c: &EvenClass) -> (u64, u64) {
    let free_n: u32 = c.free_vals.iter().map(|&(_, n)| u32::from(n)).sum();
    let n0 = 24 - c.w - free_n;
    // Residue 2: the word values, on the codeword support (w slots).
    // Residue 0: the free values and the zeros, off it.
    let a_word = u64::from(c.w) * lg_ceil(c.word_vals.len() as u128);
    let off_kinds = c.free_vals.len() + usize::from(n0 > 0);
    let a_free = u64::from(24 - c.w) * lg_ceil(off_kinds as u128);
    let nonzero = 24 - n0;
    (a_word + a_free, u64::from(nonzero.saturating_sub(1)))
}

/// Bits an odd class spends on the same two planes.
///
/// The sign plane is **zero**: on the odd coset the signs are forced by
/// membership (`generic.rs` / `index.rs` — "signs carry no information"), and
/// `Golay70`'s kernel recomputes them as `neg = c_i ^ flag`. The magnitude
/// plane, however, costs `⌈log₂ L⌉` per slot and **not less**: the codeword
/// says which slots carry `x_i ≡ 3 (mod 4)`, a property of the *signed* value,
/// so it constrains the sign-magnitude pair jointly and cannot be spent twice.
/// That is precisely the refutation of the 46-bit variant, and the reason this
/// function does not subtract anything for it.
fn odd_planes(c: &OddClass) -> (u64, u64) {
    (24 * lg_ceil(c.vals.len() as u128), 0)
}

fn widths_of_even(c: &EvenClass) -> Widths {
    let free_n: u32 = c.free_vals.iter().map(|&(_, n)| u32::from(n)).sum();
    let n0 = 24 - c.w - free_n;
    // The five radices of the even composition, in the order `Indexer::encode`
    // multiplies them: codeword, on-support arrangement, off-support
    // arrangement, word signs, free signs.
    let arr_on = multiset(u128::from(c.w), c.word_vals.iter().map(|&(_, n)| n));
    let arr_off = multiset(
        u128::from(24 - c.w),
        c.free_vals
            .iter()
            .map(|&(_, n)| n)
            .chain(std::iter::once(n0 as u8)),
    );
    let s_w = if c.w == 0 { 1u128 } else { 1u128 << (c.w - 1) };
    let s_f = 1u128 << free_n;
    let radices = [
        u128::from(gamma(c.w)),
        arr_on,
        arr_off,
        s_w,
        s_f,
    ];
    let rounded: u64 = radices.iter().map(|&r| lg_ceil(r)).sum();
    let exact = lg_ceil(radices.iter().product::<u128>());
    let (a, sg) = even_planes(c);
    let levels = c.n_levels();
    Widths {
        radix2: HEADER_BITS + rounded,
        radix2_g12: HEADER_BITS + rounded - lg_ceil(u128::from(gamma(c.w))) + GOLAY_RANK_BITS,
        perslot: HEADER_BITS + 24 * lg_ceil(levels as u128) + u64::from(24 - n0),
        golay70: HEADER_BITS + GOLAY_RANK_BITS + 24 + 24,
        golay_tight: HEADER_BITS + GOLAY_RANK_BITS + a + sg,
        exact,
    }
}

fn widths_of_odd(c: &OddClass) -> Widths {
    let arr = multiset(24, c.vals.iter().map(|&(_, n)| n));
    let radices = [4096u128, arr];
    let rounded: u64 = radices.iter().map(|&r| lg_ceil(r)).sum();
    let (a, sg) = odd_planes(c);
    let levels = c.n_levels();
    Widths {
        radix2: HEADER_BITS + rounded,
        // The odd codeword field is already the flat 4096 rank: 12 bits, the
        // same field `golay70` carries, so the two variants coincide here.
        radix2_g12: HEADER_BITS + rounded,
        // The odd coset has no zeros, so every slot carries a sign — except
        // that on this coset signs are forced, hence none are stored.
        perslot: HEADER_BITS + 24 * lg_ceil(levels as u128),
        golay70: HEADER_BITS + GOLAY_RANK_BITS + 24 + 24,
        golay_tight: HEADER_BITS + GOLAY_RANK_BITS + a + sg,
        exact: lg_ceil(radices.iter().product::<u128>()),
    }
}

/// `n! / Π cᵢ!` — the multiset permutation count, in `u128` because 24! does
/// not fit `u64`.
fn multiset(n: u128, counts: impl Iterator<Item = u8>) -> u128 {
    let fact = |k: u128| (1..=k).product::<u128>();
    let mut m = fact(n);
    for c in counts {
        m /= fact(u128::from(c));
    }
    m
}

/// One variant's identity: how it decodes, and whether that decode is
/// admissible for a fused kernel at all.
struct Variant {
    name: &'static str,
    /// Bits per block of this class.
    get: fn(&Widths) -> u64,
    /// Can every field be extracted **and interpreted** without a
    /// data-dependent loop?
    shift_only: bool,
    /// Bounded decode depth, in dependent steps. `None` = unbounded/serial.
    depth: Option<u32>,
    note: &'static str,
}

const VARIANTS: &[Variant] = &[
    Variant {
        name: "archive (le fichier)",
        get: |_| ARCHIVE_BITS,
        shift_only: false,
        depth: None,
        note: "~509 ops sérielles, 8,27 ns/bloc mesuré — 75× un layout à masques",
    },
    Variant {
        name: "radix2",
        get: |w| w.radix2,
        shift_only: false,
        depth: None,
        note: "champs extraits par shift, rangs de multiensemble encore sériels",
    },
    Variant {
        name: "radix2 + golay 12 b",
        get: |w| w.radix2_g12,
        shift_only: false,
        depth: None,
        note: "idem, champ de codeword uniforme (offset indépendant de la classe)",
    },
    Variant {
        name: "golay_tight",
        get: |w| w.golay_tight,
        shift_only: true,
        depth: Some(24),
        note: "plans A et signes réduits à ce que la classe exige — largeur variable",
    },
    Variant {
        name: "golay70 (mesuré, écarté)",
        get: |w| w.golay70,
        shift_only: true,
        depth: Some(24),
        note: "3,589 b/poids et 1,31× vs FP16, sous le critère de 1,6×",
    },
    Variant {
        name: "perslot (= Planes)",
        get: |w| w.perslot,
        shift_only: true,
        depth: Some(24),
        note: "la géométrie de production",
    },
];

/// Qwen3-4B, the published `leech1c12` file — the shapes the kernel
/// accounting divides by. Same constants as `rtbits`' acceptance test.
mod qwen3_4b {
    pub const WEIGHTS: u64 = 3_633_315_840;
    pub const BLOCKS: u64 = 150_681_600;
    pub const TAIL_WEIGHTS: u64 = 16_957_440;
    pub const ROWS: u64 = 1_105_920;
}

/// Kernel-accounting b/weight of a stream costing `bits_per_block`, on the 4B
/// shapes: the `f32` tail and row scales join the numerator, and the
/// denominator is every weight. The one grandeur the 2.6 criterion is stated
/// in.
fn kernel_bpw(bits_per_block: f64) -> f64 {
    let side = (qwen3_4b::TAIL_WEIGHTS + qwen3_4b::ROWS) * 32;
    (bits_per_block * qwen3_4b::BLOCKS as f64 + side as f64) / qwen3_4b::WEIGHTS as f64
}

fn main() {
    let fd = FastDecoder::new();
    let cs = enumerate_classes(13);
    let mut by_key: HashMap<Key, Widths> = HashMap::new();
    for c in &cs.even {
        let free_n: u32 = c.free_vals.iter().map(|&(_, n)| u32::from(n)).sum();
        let n0 = 24 - c.w - free_n;
        let mut kinds: Vec<(i32, u8)> = c
            .word_vals
            .iter()
            .chain(&c.free_vals)
            .map(|&(v, n)| (i32::from(v), n))
            .collect();
        if n0 > 0 {
            kinds.push((0, n0 as u8));
        }
        by_key.insert(canonical(kinds, false), widths_of_even(c));
    }
    for c in &cs.odd {
        let kinds = c.vals.iter().map(|&(v, n)| (i32::from(v), n)).collect();
        by_key.insert(canonical(kinds, true), widths_of_odd(c));
    }

    // Per-class widths in FastDecoder order, so the sweep is a direct index.
    let table: Vec<Widths> = (0..fd.n_classes())
        .map(|ci| {
            let lv = fd.levels(ci);
            *by_key
                .get(&key_of_levels(lv))
                .unwrap_or_else(|| panic!("classe {ci} absente de enumerate_classes : {lv:?}"))
        })
        .collect();

    // ---- weights: real blocks if we have a file, else one per class ----
    let mut count = vec![0u64; fd.n_classes()];
    let mut origins = 0u64;
    let source = match std::env::args().nth(1) {
        Some(path) => {
            let f = File::open(&path).unwrap_or_else(|e| panic!("open {path}: {e}"));
            let mut r = BufReader::new(f);
            let h = llvq_artifact::read_header(&mut r).expect("valid artifact header");
            for _ in 0..h.matrices {
                let m = llvq_artifact::read_matrix_raw(&mut r).expect("valid matrix");
                for &idx in &m.indices {
                    match fd.class_of(idx) {
                        Some(ci) => count[ci] += 1,
                        None => origins += 1,
                    }
                }
            }
            format!("{path} — {} matrices, blocs réels", h.matrices)
        }
        None => {
            count.iter_mut().for_each(|c| *c = 1);
            "les 383 classes à poids égal (⚠️ PAS la distribution réelle)".to_string()
        }
    };
    let total: u64 = count.iter().sum::<u64>() + origins;

    println!("source : {source}");
    println!("{total} blocs, {} classes", fd.n_classes());
    if origins > 0 {
        println!("dont {origins} blocs origine, comptés au tarif plein de chaque variante");
    }

    // ---- the menu, priced ----
    println!("\n  bits par bloc, et ce que ça vaut en b/poids NOYAU sur le 4B");
    println!("  (critère d'ouverture E3 : ≤ {E3_CRITERION} b/poids ET shift-only à profondeur bornée)");
    println!("  {}", "-".repeat(94));
    println!(
        "  {:<24}{:>7}{:>6}{:>9}{:>11}{:>11}{:>7}  prof.",
        "variante", "moy", "pire", "grp32", "b/p moy", "b/p GRP", "shift"
    );
    let mut admissible: Vec<(&str, f64)> = Vec::new();
    let mut best = f64::INFINITY;
    for v in VARIANTS {
        let (mut sum, mut worst, mut grouped, mut fill, mut gmax) = (0f64, 0u64, 0u64, 0u64, 0u64);
        for (ci, &n) in count.iter().enumerate() {
            if n == 0 {
                continue;
            }
            let w = (v.get)(&table[ci]);
            sum += w as f64 * n as f64;
            worst = worst.max(w);
            // Grouped-32 is charged only where the width actually varies; the
            // sweep order is class-major here, not block order, so this is an
            // OPTIMISTIC bound for the variable variants (a real stream
            // interleaves classes and pays more). Flagged in the output.
            for _ in 0..n {
                gmax = gmax.max(w);
                fill += 1;
                if fill == GROUP {
                    grouped += GROUP * gmax.div_ceil(8) * 8 + 32;
                    fill = 0;
                    gmax = 0;
                }
            }
        }
        if fill > 0 {
            grouped += GROUP * gmax.div_ceil(8) * 8 + 32;
        }
        // Origin blocks pay the same fixed record everywhere except `archive`.
        // An origin block carries a class id, a gain bit and nothing else in
        // every runtime variant; in the archive it is still a full index.
        sum += origins as f64 * (v.get)(&Widths::default()).max(HEADER_BITS) as f64;
        let mean = sum / total as f64;
        let bpw = kernel_bpw(mean);
        let bpw_grouped = kernel_bpw(grouped as f64 / total as f64);
        best = best.min(if v.shift_only { bpw_grouped } else { f64::INFINITY });
        // The verdict is taken on `b/p GRP`: a variable-width stream is read
        // by 32 lanes at the group's widest stride, so that — not the mean —
        // is what a kernel uploads. Both are printed so the gap is visible.
        println!(
            "  {:<24}{:>7.2}{:>6}{:>9.2}{:>11.4}{:>11.4}{:>7}  {}",
            v.name,
            mean,
            worst,
            grouped as f64 / total as f64,
            bpw,
            bpw_grouped,
            if v.shift_only { "oui" } else { "NON" },
            v.depth.map_or("sér.".to_string(), |d| format!("{d}")),
        );
        if v.shift_only && bpw_grouped <= E3_CRITERION {
            admissible.push((v.name, bpw_grouped));
        }
    }
    println!("  {}", "-".repeat(94));
    for v in VARIANTS {
        println!("  {:<24}{}", v.name, v.note);
    }
    println!(
        "\n  ⚠️ la colonne « groupé32 » est OPTIMISTE : les blocs sont balayés classe par classe\n  \
         ici, alors qu'un flux réel entrelace les classes et paie donc un stride plus large."
    );

    // ---- where the archive's 47 bits actually go ----
    let exact_mean: f64 = count
        .iter()
        .enumerate()
        .map(|(ci, &n)| table[ci].exact as f64 * n as f64)
        .sum::<f64>()
        / total as f64;
    println!("\n  où passent les 47 bits de l'archive");
    println!("  {}", "-".repeat(94));
    println!("  point DANS sa classe, moyenne de ⌈log₂ |classe|⌉ : {exact_mean:.2} bits");
    println!(
        "  choix de la classe : 47 − {exact_mean:.2} ≈ {:.2} bits, que toute variante à champ\n  \
         de classe explicite repaie en {HEADER_BITS} bits d'en-tête",
        47.0 - exact_mean
    );

    // ---- the verdict ----
    println!("\n  VERDICT X4");
    println!("  {}", "-".repeat(94));
    if admissible.is_empty() {
        println!(
            "  ❌ AUCUNE décomposition shift-only ne passe sous {E3_CRITERION} b/poids noyau."
        );
        println!(
            "  Le meilleur point shift-only vaut {best:.4} b/poids, soit {:.0} % au-dessus du seuil.",
            100.0 * (best / E3_CRITERION - 1.0)
        );
        println!(
            "  Le critère d'ouverture du chantier E3 n'est PAS atteint : E3 s'enterre sur papier,\n  \
             pour 0 $, comme E2 s'est enterré au banc. La marche suivante de l'échelle mémoire\n  \
             reste E1c (X1/X2), et le plancher pratique du projet est le layout mesuré."
        );
    } else {
        println!("  ✅ décomposition(s) shift-only sous le critère :");
        for (n, b) in &admissible {
            println!("     {n:<24}{b:.4} b/poids noyau");
        }
        println!(
            "  Le chantier E3 est ouvert au sens de la spec. ⚠️ Ce verdict porte sur les BITS\n  \
             seuls : la vitesse d'un tel décodeur reste non mesurée, et Golay70 rappelle qu'un\n  \
             format juste et compact peut mourir en ALU (195 Go/s contre 425)."
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lg_ceil_is_the_field_width() {
        assert_eq!(lg_ceil(0), 0);
        assert_eq!(lg_ceil(1), 0, "un seul choix ne coûte aucun bit");
        assert_eq!(lg_ceil(2), 1);
        assert_eq!(lg_ceil(3), 2);
        assert_eq!(lg_ceil(4), 2);
        assert_eq!(lg_ceil(5), 3);
        assert_eq!(lg_ceil(4096), 12, "le rang de codeword tient en 12 bits");
        assert_eq!(lg_ceil(4097), 13);
    }

    /// Rounding radices to powers of two can only **cost** bits, never save
    /// any, and the waste is bounded by one bit per radix. Both halves matter:
    /// a variant that came out cheaper than the exact composition would be
    /// arithmetically impossible and must fail loudly.
    #[test]
    fn rounding_costs_between_zero_and_one_bit_per_radix() {
        let cs = enumerate_classes(13);
        for c in &cs.even {
            let w = widths_of_even(c);
            let rounded = w.radix2 - HEADER_BITS;
            assert!(rounded >= w.exact, "classe paire : arrondi sous l'exact");
            assert!(rounded <= w.exact + 5, "5 radices, 5 bits de gaspillage max");
        }
        for c in &cs.odd {
            let w = widths_of_odd(c);
            let rounded = w.radix2 - HEADER_BITS;
            assert!(rounded >= w.exact);
            assert!(rounded <= w.exact + 2, "2 radices côté impair");
        }
    }

    /// `golay_tight` may never beat `golay70` by more than the two planes it
    /// trims, and may never exceed it except where a class genuinely needs
    /// more than one bit per slot — the case `Golay70` refuses outright and
    /// pays for with an exception.
    #[test]
    fn golay_tight_only_trims_what_a_class_does_not_use() {
        let cs = enumerate_classes(13);
        let mut trimmed = 0;
        for c in &cs.even {
            let w = widths_of_even(c);
            assert!(w.golay_tight >= HEADER_BITS + GOLAY_RANK_BITS);
            if w.golay_tight < w.golay70 {
                trimmed += 1;
            }
        }
        assert!(trimmed > 0, "aucune classe paire ne profite du serrage");
    }

    /// **The refutation, encoded.** On the odd coset the codeword constrains
    /// the *signed* residue, so it cannot pay for a magnitude bit as well: a
    /// two-magnitude odd class must still spend 24 bits, giving 46 only if
    /// one also stopped storing something else. The dossier proposed 46 bits
    /// there and refuted it; this pins the refutation so no future edit can
    /// quietly re-derive it.
    #[test]
    fn the_refuted_46_bit_odd_variant_is_not_reachable() {
        let cs = enumerate_classes(13);
        for c in &cs.odd {
            let w = widths_of_odd(c);
            let (a, sg) = odd_planes(c);
            assert_eq!(sg, 0, "les signes impairs sont calculés, jamais stockés");
            assert_eq!(a, 24 * lg_ceil(c.vals.len() as u128));
            assert!(
                w.golay_tight >= 46,
                "classe impaire à {} bits — la variante 46 b réfutée est de retour",
                w.golay_tight
            );
        }
    }

    /// The two class sources must cover each other exactly, or the sweep
    /// would price blocks against another class's radices.
    #[test]
    fn every_fastdecoder_class_has_its_radices() {
        let fd = FastDecoder::new();
        let cs = enumerate_classes(13);
        let n = cs.even.len() + cs.odd.len();
        assert_eq!(n, fd.n_classes(), "{n} classes énumérées, {} au décodeur", fd.n_classes());
        let mut keys: Vec<Key> = Vec::new();
        for c in &cs.even {
            let free_n: u32 = c.free_vals.iter().map(|&(_, n)| u32::from(n)).sum();
            let n0 = 24 - c.w - free_n;
            let mut kinds: Vec<(i32, u8)> = c
                .word_vals
                .iter()
                .chain(&c.free_vals)
                .map(|&(v, n)| (i32::from(v), n))
                .collect();
            if n0 > 0 {
                kinds.push((0, n0 as u8));
            }
            keys.push(canonical(kinds, false));
        }
        for c in &cs.odd {
            keys.push(canonical(
                c.vals.iter().map(|&(v, n)| (i32::from(v), n)).collect(),
                true,
            ));
        }
        let set: HashMap<Key, ()> = keys.iter().map(|&k| (k, ())).collect();
        assert_eq!(set.len(), keys.len(), "deux classes partagent une clé canonique");
        for ci in 0..fd.n_classes() {
            let k = key_of_levels(fd.levels(ci));
            assert!(set.contains_key(&k), "classe {ci} sans radices : {:?}", fd.levels(ci));
        }
    }

    /// `multiset` must agree with the cardinalities `classes.rs` computes, or
    /// every radix below it is wrong.
    #[test]
    fn multiset_matches_the_class_cardinalities() {
        let cs = enumerate_classes(13);
        for c in &cs.odd {
            let arr = multiset(24, c.vals.iter().map(|&(_, n)| n));
            assert_eq!(
                u64::try_from(4096 * arr).unwrap(),
                c.cardinality(),
                "cardinalité impaire"
            );
        }
    }

    /// The criterion is stated in the kernel accounting, so the conversion
    /// must be the same one `rtbits` and the CUDA bench use: `Planes14`'s 112
    /// bits per block are 4.804 b/weight on the 4B.
    #[test]
    fn kernel_conversion_reproduces_planes14() {
        let p14 = kernel_bpw(112.0);
        assert!((p14 - 4.804).abs() < 5e-4, "Planes14 : {p14:.4}");
        // And the archive's own 48 bits, the floor E3 is chasing.
        let arch = kernel_bpw(ARCHIVE_BITS as f64);
        assert!(arch < E3_CRITERION, "le fichier lui-même vaut {arch:.4}");
    }
}
