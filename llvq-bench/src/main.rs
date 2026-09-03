//! Gaussian-source rate-distortion report (paper §4 protocol).
//!
//! Usage: `cargo run --release -p llvq-bench [-- train_n eval_n seed]`

use llvq_bench::*;
use llvq_core::SplitMix64;
use llvq_search::Searcher;

fn main() {
    let args: Vec<u64> = std::env::args()
        .skip(1)
        .map(|a| a.parse().expect("numeric args: train_n eval_n seed"))
        .collect();
    let train_n = *args.first().unwrap_or(&4_000) as usize;
    let eval_n = *args.get(1).unwrap_or(&20_000) as usize;
    let seed = *args.get(2).unwrap_or(&0x64A4_2026);

    let s = Searcher::new();
    let mut rng = SplitMix64::new(seed);
    let train: Vec<_> = (0..train_n).map(|_| gauss_block(&mut rng)).collect();
    let eval: Vec<_> = (0..eval_n).map(|_| gauss_block(&mut rng)).collect();

    eprintln!("searching {train_n} train + {eval_n} eval blocks (union m ≤ 3)…");
    let t0 = std::time::Instant::now();
    let train_dots = precompute(&s, &train);
    let eval_dots = precompute(&s, &eval);
    eprintln!(
        "search pass: {:.1}s ({:.0} blocks/s total)",
        t0.elapsed().as_secs_f64(),
        (train_n + eval_n) as f64 / t0.elapsed().as_secs_f64()
    );

    // Spherical shaping: fit β on train, evaluate on eval.
    let beta = optimize_beta(&train_dots, 0.4, 1.1, 140);
    let mse_sph = spherical_mse(&eval_dots, beta);
    let r_sph = rate_spherical();

    // Shape–gain, two rows per rate. The first is the shipped rule — the gain
    // code rounds the block norm, exactly as `LeechShapeGain` does, centroids
    // fitted on norms. The second rounds the projection instead, which is the
    // best any gain code can do for a given direction and therefore a bound,
    // not a configuration. Reporting only the bound is what made every G4
    // retention figure describe a quantizer that never ran on a model.
    let train_norm: Vec<f64> = train_dots.iter().map(BlockDots::norm).collect();
    let train_t: Vec<f64> = train_dots.iter().map(BlockDots::t).collect();
    let rows_sg: Vec<(u32, f64, f64, f64)> = [0u32, 2]
        .into_iter()
        .map(|k| {
            (
                k,
                rate_shape_gain(k),
                shape_gain_mse_shipped(&eval_dots, &lloyd_max(&train_norm, k, 60)),
                shape_gain_mse_projected(&eval_dots, &lloyd_max(&train_t, k, 60)),
            )
        })
        .collect();

    println!("\nGaussian source N(0,1), {eval_n} eval blocks of 24 (seed {seed:#x})");
    println!("codebook: Shell(2) ∪ Shell(3), spherical β* = {beta:.3}\n");
    println!(
        "{:<38} {:>9} {:>9} {:>12} {:>9}",
        "method", "bits/dim", "MSE", "SQNR(bits)", "Ret(%)"
    );
    let row = |name: &str, r: f64, mse: f64| {
        println!(
            "{name:<38} {r:>9.4} {mse:>9.4} {:>12.4} {:>9.2}",
            sqnr_bits(mse),
            retention_pct(mse, r)
        );
    };
    row(
        "Lloyd–Max scalar 1-bit (analytic)",
        1.0,
        lloyd_max_1bit_scalar_mse(),
    );
    row("LLVQ spherical shaping (m ≤ 3)", r_sph, mse_sph);
    for (k, r, mse, bound) in &rows_sg {
        row(&format!("LLVQ shape–gain, {k}-bit gain (m ≤ 3)"), *r, *mse);
        row(&format!("  ↳ bound, optimal gain ({k}-bit)"), *r, *bound);
    }
    println!(
        "{:<38} {:>9.4} {:>9.4} {:>12.4} {:>9.2}",
        "Shannon limit @ r_sph",
        r_sph,
        2f64.powf(-2.0 * r_sph),
        r_sph,
        100.0
    );

    // ------------------------------------------------------------------
    // Full ball Λ₂₄(13): the 2 bit/dim regime of the paper's Table 3.
    // ------------------------------------------------------------------
    eprintln!("\nsearching ball-13 (12 shells, generic class engine)…");
    let t1 = std::time::Instant::now();
    let train13 = precompute13(&s, &train);
    let eval13 = precompute13(&s, &eval);
    eprintln!(
        "ball-13 pass: {:.1}s ({:.0} blocks/s total)",
        t1.elapsed().as_secs_f64(),
        (train_n + eval_n) as f64 / t1.elapsed().as_secs_f64()
    );

    let beta13 = optimize_beta13(&train13, 0.2, 0.9, 140);
    let mse13 = spherical_mse13(&eval13, beta13);
    let r13 = rate_spherical13();

    let train_norm13: Vec<f64> = train13.iter().map(BlockDots13::norm).collect();
    let train_t13: Vec<f64> = train13.iter().map(BlockDots13::t).collect();
    let rows_sg13: Vec<(u32, f64, f64, f64)> = [0u32, 2]
        .into_iter()
        .map(|k| {
            (
                k,
                rate_shape_gain13(k),
                shape_gain_mse13_shipped(&eval13, &lloyd_max(&train_norm13, k, 60)),
                shape_gain_mse13_projected(&eval13, &lloyd_max(&train_t13, k, 60)),
            )
        })
        .collect();

    println!("\n— ball Λ24(13), 2 bits/dim (paper Table 3), β* = {beta13:.3} —");
    println!(
        "{:<38} {:>9} {:>9} {:>12} {:>9}",
        "method", "bits/dim", "MSE", "SQNR(bits)", "Ret(%)"
    );
    // Paper Table 8 (App. H), read off the rendered PDF — text extraction of
    // this document doubles digits (0.084 came out as "0.1084").
    row("paper: spherical shaping Λ24(13)", 2.0, 0.084);
    row("paper: shape–gain, 0 gain bits", 2.0, 0.085);
    row("paper: shape–gain, 1 gain bit", 2.0, 0.078);
    row("LLVQ spherical shaping (m ≤ 13)", r13, mse13);
    for (k, r, mse, bound) in &rows_sg13 {
        row(&format!("LLVQ shape–gain, {k}-bit gain (m ≤ 13)"), *r, *mse);
        row(&format!("  ↳ bound, optimal gain ({k}-bit)"), *r, *bound);
    }
    // Single shell vs the union: same harness, one line of difference. Gain
    // centroids are fitted on `train`, like every other row — measuring them
    // on the set they were fitted to would flatter the single-shell codes.
    for m in [12u32, 13] {
        for k in [0u32, 1] {
            let centroids = lloyd_max(&train_norm13, k, 60);
            row(
                &format!("LLVQ shape–gain, {k}-bit gain (shell {m} only)"),
                rate_shape_gain13_single(m, k),
                shape_gain_mse13_single(&eval13, &centroids, m),
            );
        }
    }

    println!(
        "{:<38} {:>9.4} {:>9.4} {:>12.4} {:>9.2}",
        "Shannon limit @ 2 bits/dim",
        2.0,
        0.0625,
        2.0,
        100.0
    );
    println!(
        "\nThe \"shape–gain\" rows code the gain on the block NORM, the way\n\
         LeechShapeGain does. That is the shipped quantizer. The \"bound\" rows\n\
         code the projection ⟨x,v̂⟩, which minimizes the error at fixed\n\
         direction: a floor, not a configuration, and the gap is 2/(1+cos θ).\n\
         The docs/ tables written before 2026-08-01 reported the bound."
    );
}
