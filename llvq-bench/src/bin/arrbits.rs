//! What does it cost, in bits, to say *which value goes where* — without a
//! permutation rank?
//!
//! The rank is optimal: `log2(24! / Π nᵢ!)` bits, and nothing beats it. It is
//! also 71 % of the decode cost, because reaching a rank means dividing
//! 128-bit integers sixty times per block. Every alternative here trades bits
//! for divisions; this measures the exchange rate on real quantized blocks
//! rather than on a worked example.
//!
//! Schemes, cheapest to decode first:
//!
//! * **positional** — `ceil(log2(types))` bits per coordinate. One shift and
//!   mask each, no divisions, no tables. The obvious thing, and the most
//!   wasteful.
//! * **nested masks** — a 24-bit mask for the first magnitude, then a mask
//!   over what remains, and so on. Branch-free, and adapts to skewed counts.
//! * **grouped ranks** — split the 24 slots into G groups and rank each from a
//!   lookup table. Divisions become table reads; the loss is that the per-group
//!   counts must be transmitted. Computed on the actual positions, not assumed
//!   proportional.
//! * **optimal rank** — what the archive format does, for reference.
//!
//! Run: `cargo run --release -p llvq-bench --bin arrbits`

use llvq_core::{SplitMix64, DIM};
use llvq_search::generic::BallSearcher;
use llvq_search::Searcher;

const N: usize = 20_000;

fn lg_fact(k: usize) -> f64 {
    (1..=k).map(|i| (i as f64).log2()).sum()
}

/// `log2(n! / Π counts!)` — the information content of an arrangement.
fn rank_bits(counts: &[usize]) -> f64 {
    let n: usize = counts.iter().sum();
    lg_fact(n) - counts.iter().map(|&c| lg_fact(c)).sum::<f64>()
}

fn lg_binom(n: usize, k: usize) -> f64 {
    if k > n {
        return 0.0;
    }
    lg_fact(n) - lg_fact(k) - lg_fact(n - k)
}

/// Fixed-width type tag per coordinate.
fn positional_bits(n_types: usize) -> f64 {
    DIM as f64 * (n_types as f64).log2().ceil().max(1.0)
}

/// One mask over the still-unassigned slots per type, largest count first. The
/// last type needs no mask — it is whatever is left.
fn nested_mask_bits(counts: &[usize]) -> f64 {
    let mut left = DIM;
    let mut bits = 0.0;
    for &c in counts.iter().take(counts.len().saturating_sub(1)) {
        bits += left as f64;
        left -= c;
    }
    bits
}

/// Rank each group independently, from the **actual** type layout.
///
/// Two terms: the ranks themselves, and the cost of saying how each type
/// splits across groups — without which a decoder cannot size its tables.
fn grouped_rank_bits(kinds: &[usize], n_types: usize, g: usize) -> f64 {
    let per = DIM / g;
    let mut bits = 0.0;
    let mut split = vec![vec![0usize; n_types]; g];
    for (slot, &k) in kinds.iter().enumerate() {
        split[slot / per][k] += 1;
    }
    for group in &split {
        bits += rank_bits(group);
    }
    // Transmitting the split: for each type, how its `c` copies land in `g`
    // groups — a composition, `C(c + g - 1, g - 1)`.
    for t in 0..n_types {
        let c: usize = split.iter().map(|s| s[t]).sum();
        bits += lg_binom(c + g - 1, g - 1);
    }
    bits
}

fn main() {
    let mut rng = SplitMix64::new(0x6_A88B);
    let s = Searcher::new();
    let mut ball = BallSearcher::new();

    let (mut opt, mut pos, mut mask, mut g4, mut g6, mut g8) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    let mut worst_types = 0usize;

    for _ in 0..N {
        let x: [f64; DIM] = core::array::from_fn(|_| rng.next_gaussian());
        let f = ball.nearest_angular(&s, &x);

        // Distinct magnitudes, descending, and the type index of each slot.
        let mut mags: Vec<i32> = f.point.iter().map(|v| v.abs()).collect();
        mags.sort_unstable_by(|a, b| b.cmp(a));
        mags.dedup();
        let kinds: Vec<usize> = f
            .point
            .iter()
            .map(|v| mags.iter().position(|m| *m == v.abs()).expect("present"))
            .collect();
        let mut counts = vec![0usize; mags.len()];
        for &k in &kinds {
            counts[k] += 1;
        }
        counts.sort_unstable_by(|a, b| b.cmp(a));
        worst_types = worst_types.max(mags.len());

        opt += rank_bits(&counts);
        pos += positional_bits(mags.len());
        mask += nested_mask_bits(&counts);
        g4 += grouped_rank_bits(&kinds, mags.len(), 4);
        g6 += grouped_rank_bits(&kinds, mags.len(), 6);
        g8 += grouped_rank_bits(&kinds, mags.len(), 8);
    }

    let n = N as f64;
    let base = opt / n;
    let row = |name: &str, total: f64, decode: &str| {
        let per_block = total / n;
        println!(
            "  {name:<22}{per_block:>9.1}{:>9.1}{:>12.3}   {decode}",
            per_block - base,
            (per_block - base) / DIM as f64
        );
    };

    println!("{N} blocs gaussiens quantifiés — coût de l'arrangement seul\n");
    println!(
        "  {:<22}{:>9}{:>9}{:>12}   décodage",
        "schéma", "bits", "Δ", "Δ b/poids"
    );
    println!("  {}", "-".repeat(86));
    row("rang optimal", opt, "~575 ns — 60 divisions u128");
    row("rangs groupés (4)", g4, "4 lectures de table");
    row("rangs groupés (6)", g6, "6 lectures de table");
    row("rangs groupés (8)", g8, "8 lectures de table");
    row("masques imbriqués", mask, "1 masque par type, branch-free");
    row("positionnel", pos, "1 décalage par coordonnée");
    println!("  {}", "-".repeat(86));
    println!("\n  jusqu'à {worst_types} magnitudes distinctes dans un bloc");
    println!(
        "\n  Δ b/poids s'ajoute aux 2,1595 mesurés sur le fichier. Repère : au-delà\n  \
         de +0,35 on dépasse 2,5 bits/poids, et l'écart avec QTIP (2,000)\n  \
         devient difficile à défendre."
    );
}
