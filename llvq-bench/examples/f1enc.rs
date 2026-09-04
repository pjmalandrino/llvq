//! How exact the F1 section encoder is, against exhaustive search on the region.
//!
//! `cargo run --release -p llvq-bench --example f1enc`
//!
//! The candidate list — the unconstrained D₈ winner, its single-coordinate
//! re-roundings, and radial shrinks — is a heuristic. Where it is not exact it
//! can only return a point farther from the target, so the measured MSE is an
//! upper bound and the retention a lower one: the bias is one-sided and against
//! F1. An adversarial review of the F1b spec named this the design's
//! uncontrolled term, so it is measured here rather than assumed small.

use llvq_bench::f1::{Prepared, Trellis, SECTION};
use llvq_core::SplitMix64;

fn main() {
    let t = Trellis::new();
    let mut rng = SplitMix64::new(0x5f1b_2026_0904);
    println!("{:<22} {:>7} {:>9} {:>10}", "section (w)", "exact", "excès moy", "excès max");
    for (label, set, w) in [
        ("section 1 (w=12)", t.section1(2, 0, 0), 12u32),
        ("section 2 (w=15)", t.section2(2, 0), 15),
        ("section 3 (w=12)", t.section3(31, 1, 1), 12),
    ] {
        let region = set.region_points(w);
        let prep = Prepared::new(set, w);
        let (mut exact, mut sum, mut worst, n) = (0usize, 0.0f64, 0.0f64, 400usize);
        for _ in 0..n {
            let mut x = [0.0f64; SECTION];
            for v in x.iter_mut() {
                let (u, z): (f64, f64) = (rng.next_f64(), rng.next_f64());
                // Scaled to the region's own radius: the scale sweep of the
                // full encoder puts the target here, so measuring far outside
                // it would measure the sweep and not the search.
                *v = 3.0 * (-2.0f64 * (u + 1e-12).ln()).sqrt() * (std::f64::consts::TAU * z).cos();
            }
            let truth = region
                .iter()
                .map(|y| y.iter().zip(&x).map(|(&v, &e)| (v as f64 - e).powi(2)).sum::<f64>())
                .fold(f64::INFINITY, f64::min);
            let got = prep.nearest(&x).1;
            if got <= truth + 1e-9 {
                exact += 1;
            } else {
                let e = got / truth - 1.0;
                sum += e;
                worst = worst.max(e);
            }
        }
        let miss = n - exact;
        println!(
            "{label:<22} {:>6.1}% {:>9} {:>10}",
            100.0 * exact as f64 / n as f64,
            if miss == 0 { "—".into() } else { format!("{:.4}", sum / miss as f64) },
            if miss == 0 { "—".into() } else { format!("{worst:.4}") },
        );
    }
}
