//! How much retention does spherical shaping lose at a non-optimal scale?
//!
//! Our Λ24(13) spherical shaping reports 92.23 % retention where the paper's
//! Table 8 reports 89.37 % on the same codebook. The scale β is the only
//! free parameter; this sweep says how sharp the optimum is.

use llvq_bench::*;
use llvq_core::SplitMix64;
use llvq_search::Searcher;

fn main() {
    let s = Searcher::new();
    let mut rng = SplitMix64::new(0x64A4_2026);
    let eval: Vec<_> = (0..20_000).map(|_| gauss_block(&mut rng)).collect();
    let dots = precompute13(&s, &eval);
    let r = rate_spherical13();
    println!("  β      MSE     Ret(%)");
    for k in 0..=16 {
        let beta = 0.20 + 0.025 * k as f64;
        let mse = spherical_mse13(&dots, beta);
        println!("{beta:5.3} {mse:8.4} {:9.2}", retention_pct(mse, r));
    }
}
