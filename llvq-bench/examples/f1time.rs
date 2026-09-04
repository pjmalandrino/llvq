//! Throughput of the F1 encoder, and whether every codeword it emits is in Λ₂₄.
//!
//! `cargo run --release -p llvq-bench --example f1time`
//!
//! F1b wants 20,000 blocks. This says what that costs and, more importantly,
//! checks the only thing that would invalidate the number whatever the timing:
//! that the assembled block is a Leech point in the repository's own coordinate
//! order, decided by `llvq_core` and not by this module.

use llvq_bench::f1::{point_to_natural, Codebook};
use llvq_core::SplitMix64;

fn main() {
    let t0 = std::time::Instant::now();
    let cb = Codebook::new([12, 15, 12]);
    println!("préparation du codebook : {:.2} s", t0.elapsed().as_secs_f64());

    let leech = llvq_core::Leech::new();
    let scales: Vec<f64> = (0..18).map(|i| 0.10 * 1.14f64.powi(i)).collect();
    let mut rng = SplitMix64::new(0xf1b_2026);
    let n = 20usize;
    let t1 = std::time::Instant::now();
    let mut bad = 0;
    let mut tsum = 0.0;
    for _ in 0..n {
        let mut x = [0.0f64; 24];
        for v in x.iter_mut() {
            let (u, z): (f64, f64) = (rng.next_f64(), rng.next_f64());
            *v = (-2.0f64 * (u + 1e-12).ln()).sqrt() * (std::f64::consts::TAU * z).cos();
        }
        let (t, y) = cb.best_t(&x, &scales);
        tsum += t;
        if !leech.contains(&point_to_natural(&y, &cb.trellis.code.order)) {
            bad += 1;
        }
    }
    let per = t1.elapsed().as_secs_f64() / n as f64;
    println!("{n} blocs, {} échelles : {:.3} s/bloc", scales.len(), per);
    println!("extrapolation à 20 000 blocs : {:.1} min", per * 20_000.0 / 60.0);
    println!("t moyen : {:.4}", tsum / n as f64);
    println!("blocs hors Λ₂₄ : {bad} / {n}");
    assert_eq!(bad, 0, "l'encodeur a produit un point qui n'est pas dans le réseau");
}
