//! Single-core encoder throughput for the generic ball-13 engine (Phase 2c).
//!
//! Usage: `cargo run --release -p llvq-bench --bin encbench [-- n seed]`
//!
//! Two paths, because they answer different questions:
//!
//! * `shell_bests` returns the twelve per-shell maxima — what the G4
//!   rate-distortion sweep needs, since it re-ranks the same search at many
//!   scales β;
//! * `nearest_scaled` returns the single nearest codeword at a fixed β
//!   (spherical shaping) and `nearest_angular` the nearest *direction*
//!   (`Q_dir` of shape–gain) — what a quantizer actually calls, and the ones
//!   whose cost multiplies by the weight count of a model.
//!
//! Each reports µs/block on one core plus a checksum, so a speedup that
//! changes a single winner is visible immediately.

use llvq_bench::gauss_block;
use llvq_core::SplitMix64;
use llvq_search::generic::BallSearcher;
use llvq_search::Searcher;

/// The G4 optimum for an N(0,1) source at 2 bits/dim.
const BETA: f64 = 0.350;

fn main() {
    let args: Vec<u64> = std::env::args()
        .skip(1)
        .map(|a| a.parse().expect("numeric args: n seed"))
        .collect();
    let n = *args.first().unwrap_or(&2_000) as usize;
    let seed = *args.get(1).unwrap_or(&0x64A4_2026);

    let s = Searcher::new();
    let mut ball = BallSearcher::new();
    let mut rng = SplitMix64::new(seed);

    // Mixed scales: high shells only win on large ‖x‖, and the pruning
    // behaves differently there. A pure N(0,1) benchmark would over-report.
    let mixed: Vec<_> = (0..n)
        .map(|i| {
            let scale = 0.5 + 0.25 * (i % 8) as f64;
            let mut b = gauss_block(&mut rng);
            for v in b.iter_mut() {
                *v *= scale;
            }
            b
        })
        .collect();
    let unit: Vec<_> = (0..n).map(|_| gauss_block(&mut rng)).collect();

    let time = |label: &str, sum: f64, el: f64| {
        println!(
            "{label:<34} {:>8.1} µs/block  {:>8.0} blocks/s/core   checksum {sum:.9}",
            el * 1e6 / n as f64,
            n as f64 / el
        );
    };

    // Warm-up: page in the class tables, settle the branch predictor.
    for x in mixed.iter().take(n.min(64)) {
        std::hint::black_box(ball.shell_bests(&s, x));
        std::hint::black_box(ball.nearest_scaled(&s, x, BETA));
    }

    let t0 = std::time::Instant::now();
    let mut sum = 0.0f64;
    for x in &mixed {
        for f in ball.shell_bests(&s, x) {
            sum += f.dot;
        }
    }
    time("shell_bests, mixed scale", sum, t0.elapsed().as_secs_f64());

    for (label, xs) in [("mixed scale", &mixed), ("N(0,1)", &unit)] {
        let t = std::time::Instant::now();
        let mut sum = 0.0f64;
        for x in xs.iter() {
            sum += ball.nearest_scaled(&s, x, BETA).dot;
        }
        time(
            &format!("nearest_scaled β={BETA}, {label}"),
            sum,
            t.elapsed().as_secs_f64(),
        );
    }

    // The shape quantizer of shape–gain: scale-invariant, so one input set.
    let t = std::time::Instant::now();
    let mut sum = 0.0f64;
    for x in unit.iter() {
        sum += ball.nearest_angular(&s, x).dot;
    }
    time("nearest_angular (Q_dir)", sum, t.elapsed().as_secs_f64());
}
