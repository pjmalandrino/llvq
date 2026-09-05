//! Single-core throughput of the Trio encoder beside `nearest_angular`, in
//! one process — `encbench`'s model (F1c's encoder line, roadmap §2.2).
//!
//! Usage: `nice -n 10 cargo run --release -p llvq-bench --bin trioencbench [-- n seed]`
//!
//! The gate is encoder against encoder: F1c's 656 µs/block/core is the
//! served ball encoder's own figure (`encbench`, `docs/HISTORIQUE.md`), so
//! `nearest_angular` runs here on the same blocks and the two numbers stand
//! side by side. Kill: the Trio median over 656. Blocks at mixed scales as
//! in `encbench`, 2,000 by default; five runs of the whole set, the median
//! reported with the range; a checksum per line so that a speedup that moves
//! a single point is visible.

use llvq_bench::gauss_block;
use llvq_core::leech::DIM;
use llvq_core::SplitMix64;
use llvq_search::generic::BallSearcher;
use llvq_search::trio::{Encoder, Scratch, Trio};
use llvq_search::Searcher;
use std::time::Instant;

/// F1c's encoder gate, µs/block/core.
const GATE_US: f64 = 656.0;

const RUNS: usize = 5;

fn main() {
    let args: Vec<u64> = std::env::args().skip(1).map(|a| a.parse().expect("numeric args: n seed")).collect();
    let n = *args.first().unwrap_or(&2_000) as usize;
    let seed = *args.get(1).unwrap_or(&0x64A4_2026);

    let mut rng = SplitMix64::new(seed);
    // Mixed scales, as `encbench`: the adaptive scale removes the dependence
    // for Trio, the angular search never had one — the same set serves both.
    let mixed: Vec<[f64; DIM]> = (0..n)
        .map(|i| {
            let scale = 0.5 + 0.25 * (i % 8) as f64;
            let mut b = gauss_block(&mut rng);
            for v in b.iter_mut() {
                *v *= scale;
            }
            b
        })
        .collect();

    let t0 = Instant::now();
    let trio = Trio::new();
    let enc = Encoder::new(&trio);
    let enc3 = Encoder::with_shrinks(&trio, 3);
    let mut scratch = Scratch::new();
    let s = Searcher::new();
    let mut ball = BallSearcher::new();
    println!("trioencbench — {n} blocs N(0,1) à échelles mixtes, graine {seed:#x}, un cœur, médiane de {RUNS} passes   (tables prêtes en {:.2} s)\n", t0.elapsed().as_secs_f64());

    // Warm-up: page in the tables, settle the branch predictor.
    for x in mixed.iter().take(n.min(64)) {
        std::hint::black_box(enc.encode(x, &mut scratch));
        std::hint::black_box(enc3.encode(x, &mut scratch));
        std::hint::black_box(ball.nearest_angular(&s, x));
    }

    // `RUNS` passes over the set; the checksum must not move between passes.
    let time = |label: &str, f: &mut dyn FnMut(&[f64; DIM]) -> f64| -> f64 {
        let mut per_block = [0.0f64; RUNS];
        let mut sums = [0.0f64; RUNS];
        for (run, slot) in per_block.iter_mut().enumerate() {
            let t = Instant::now();
            let mut sum = 0.0f64;
            for x in &mixed {
                sum += f(x);
            }
            *slot = t.elapsed().as_secs_f64() * 1e6 / n as f64;
            sums[run] = sum;
        }
        assert!(sums.iter().all(|&v| v.to_bits() == sums[0].to_bits()), "{label}: the checksum moved between passes: {sums:?}");
        let mut sorted = per_block;
        sorted.sort_unstable_by(f64::total_cmp);
        let median = sorted[RUNS / 2];
        println!(
            "{label:<44} {median:>8.1} µs/block/core   [{:.1} .. {:.1}]   {:>7.0} blocks/s/core   checksum {:.9}",
            sorted[0],
            sorted[RUNS - 1],
            1e6 / median,
            sums[0]
        );
        median
    };

    let trio_us = time("Trio encode, 2 scales, bench rule", &mut |x| enc.encode(x, &mut scratch).t);
    let trio3_us = time("Trio encode, 2 scales, rule cut at 3 shrinks", &mut |x| enc3.encode(x, &mut scratch).t);
    time("Trio encode_at_scale, 1 scale (s₀)", &mut |x| enc.encode_at_scale(x, Encoder::lower_scale(x).1, &mut scratch).t);
    let angular_us = time("nearest_angular (Q_dir), the served encoder", &mut |x| ball.nearest_angular(&s, x).dot);

    println!(
        "\nporte F1c (encodeur seul, ≤ {GATE_US:.0} µs/block/core) : Trio {trio_us:.1} → {}   ({:.2} de la porte ; règle tronquée {trio3_us:.1} ; nearest_angular {angular_us:.1} dans le même processus)",
        if trio_us <= GATE_US { "PASSE" } else { "TUÉ" },
        trio_us / GATE_US
    );
}
