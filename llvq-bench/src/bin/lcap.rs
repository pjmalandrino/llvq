//! What does capping the magnitudes per block cost in quality, and buy in RAM?
//!
//! The fused kernel's runtime layout spends `34 + 24(L−1)` bits per block,
//! where L is the number of distinct magnitudes it holds — **zero included**.
//! So L *is* the memory cost, and capping it is the only knob that trades
//! distortion for RAM directly. That matters because the measured `Slot32`
//! rate, 5.51 bits/weight, is *worse* than 4-bit quantization's 4.50: the
//! speed was bought with the very bits that justify going to 2 bits at all.
//!
//! This bench answers whether the trade is worth taking. For each cap it
//! restricts the **codebook** — the excluded points do not exist for the
//! searcher — then measures shape–gain distortion on a Gaussian source with
//! the same protocol as `llvq-bench`: gain centroids fitted by Lloyd–Max on a
//! separate train split, evaluated on a held-out one.
//!
//! The rate moves too, and in our favour: a smaller codebook needs fewer index
//! bits. Retention is reported against each cap's **own** rate, which is the
//! only comparison that means anything.
//!
//! ## What to expect, and why it is not obvious
//!
//! Capping at L ≤ 3 throws away 78 % of the codebook — but only 2 bits of
//! index out of 48. In 24 dimensions, dividing the point count by 4.5 grows
//! the covering radius by just `4.5^(1/24) ≈ 6.6 %`. High dimension is why
//! codebook size buys so little resolution, and why this may be nearly free.
//!
//! ⚠️ Gaussian source, one seed, no GPTQ. Real weights are not Gaussian and
//! the GPTQ loop reshapes their distribution — this indicates, it does not
//! decide. A 3-block A/B on the real model is what settles it.
//!
//! Run: `cargo run --release -p llvq-bench --bin lcap`

use llvq_bench::{gauss_block, lloyd_max, nearest_centroid, retention_pct};
use llvq_core::{SplitMix64, DIM};
use llvq_search::classes::{enumerate_classes, ClassSet, MAX_SHELL};
use llvq_search::generic::{BallSearcher, MAX_LEVELS_ANY};
use llvq_search::Searcher;

const TRAIN: usize = 20_000;
const EVAL: usize = 20_000;
/// Gain bits per block, the published configuration.
const GAIN_BITS: u32 = 1;

/// Codebook size under a level cap — the same predicate the searcher filters
/// on, so the rate and the search can never describe different codebooks.
fn codebook_size(cap: usize) -> u128 {
    let ClassSet { even, odd } = enumerate_classes(MAX_SHELL);
    even.iter()
        .filter(|c| c.n_levels() <= cap)
        .map(|c| c.cardinality() as u128)
        .chain(
            odd.iter()
                .filter(|c| c.n_levels() <= cap)
                .map(|c| c.cardinality() as u128),
        )
        .sum()
}

/// `⟨x, v⟩ / ‖v‖` for the block's chosen direction, plus `‖x‖²`.
fn project(s: &Searcher, ball: &mut BallSearcher, x: &[f64; DIM]) -> (f64, f64) {
    let f = ball.nearest_angular(s, x);
    let (mut dot, mut nn) = (0.0f64, 0.0f64);
    for (&xi, &vi) in x.iter().zip(f.point.iter()) {
        dot += xi * vi as f64;
        nn += (vi as f64) * (vi as f64);
    }
    let xx: f64 = x.iter().map(|v| v * v).sum();
    // A zero direction (the origin block) projects to nothing.
    (if nn > 0.0 { dot / nn.sqrt() } else { 0.0 }, xx)
}

fn main() {
    // The same blocks for every cap: the comparison is between codebooks, so
    // nothing else may move.
    let mut rng = SplitMix64::new(0x6_1CAB);
    let train: Vec<[f64; DIM]> = (0..TRAIN).map(|_| gauss_block(&mut rng)).collect();
    let eval: Vec<[f64; DIM]> = (0..EVAL).map(|_| gauss_block(&mut rng)).collect();
    let s = Searcher::new();

    println!(
        "source gaussienne, {TRAIN} blocs d'entraînement + {EVAL} d'évaluation, \
         shape–gain {GAIN_BITS} bit\n"
    );
    println!(
        "  {:<7}{:>9}{:>18}{:>9}{:>9}{:>9}{:>13}{:>11}",
        "L max", "classes", "codebook", "index", "b/dim", "MSE", "rétention", "RAM b/p"
    );
    println!("  {}", "-".repeat(88));

    let mut baseline = None;
    for cap in 2..=MAX_LEVELS_ANY {
        let n = codebook_size(cap);
        if n == 0 {
            continue;
        }
        let index_bits = 128 - n.leading_zeros();
        let rate = (index_bits + GAIN_BITS) as f64 / DIM as f64;

        let ClassSet { even, odd } = enumerate_classes(MAX_SHELL);
        let n_classes = even.iter().filter(|c| c.n_levels() <= cap).count()
            + odd.iter().filter(|c| c.n_levels() <= cap).count();
        let mut ball = BallSearcher::with_level_cap(cap);
        let t_train: Vec<f64> = train
            .iter()
            .map(|x| project(&s, &mut ball, x).0)
            .collect();
        let centroids = lloyd_max(&t_train, GAIN_BITS, 60);

        let mse: f64 = eval
            .iter()
            .map(|x| {
                let (t, xx) = project(&s, &mut ball, x);
                let g = centroids[nearest_centroid(&centroids, t)];
                xx - 2.0 * g * t + g * g
            })
            .sum::<f64>()
            / (DIM * eval.len()) as f64;

        // Runtime cost of the layout this cap permits: class id + gain +
        // 24-bit sign mask + one 24-bit slot mask per level above the first,
        // rounded to a byte because a block has to start somewhere.
        let payload = 9 + GAIN_BITS + 24 + 24 * (cap as u32 - 1);
        let ram = payload.div_ceil(8) * 8;

        let ret = retention_pct(mse, rate);
        if cap == MAX_LEVELS_ANY {
            baseline = Some((ret, ram as f64 / 24.0));
        }
        println!(
            "  {cap:<7}{n_classes:>9}{n:>18}{index_bits:>9}{rate:>9.4}{mse:>9.4}{:>12.2} %{:>11.3}",
            ret,
            ram as f64 / 24.0
        );
    }
    println!("  {}", "-".repeat(88));
    println!(
        "  {:<7}{:>9}{:>18}{:>9}{:>9}{:>9.4}{:>12.2} %{:>11}",
        "—", "—", "Shannon", "—", 2.0, 0.0625, 100.0, "—"
    );

    if let Some((ret5, ram5)) = baseline {
        println!(
            "\n  Repères : 4 bits (MLX, g64) = 4,500 b/poids en RAM · \
             Slot32 non contraint mesuré = 5,510\n  \
             Sans contrainte ici : {ret5:.2} % de rétention à {ram5:.3} b/poids de RAM.\n"
        );
    }
    println!(
        "  ⚠️  Source gaussienne, une seule graine, pas de GPTQ. Ceci indique,\n  \
         ça ne tranche pas — l'A/B sur 3 blocs du vrai modèle tranche."
    );
}
