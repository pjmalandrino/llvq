//! Where does the scale sweep actually need to look?
//!
//! `cargo run --release -p llvq-bench --example f1scale`
//!
//! Diagnostic. The first F1b pilot returned 64.65% retention against a ceiling
//! argument that says ~89%, so something in the encoder gives away a quarter of
//! the answer. `t = ⟨x,y⟩/‖y‖` as a function of the sweep scale says whether
//! the grid was looking in the right place.

use llvq_bench::f1::Codebook;
use llvq_bench::gauss_block;
use llvq_core::SplitMix64;

fn main() {
    let cb = Codebook::new([12, 15, 12]);
    let mut rng = SplitMix64::new(0x0f1b_2026_0904);
    let xs: Vec<_> = (0..8).map(|_| gauss_block(&mut rng)).collect();
    println!("{:>7}  {:>8}  {:>8}", "échelle", "t moyen", "‖y‖ moyen");
    let mut best_overall = (0.0f64, 0.0f64);
    for i in 0..24 {
        let s = 0.08 * 1.20f64.powi(i);
        let (mut tsum, mut nsum) = (0.0f64, 0.0f64);
        for x in &xs {
            let y = cb.encode_at_scale(x, s);
            let dot: f64 = x.iter().zip(&y).map(|(&a, &b)| a * b as f64).sum();
            let nn: f64 = y.iter().map(|&b| (b as f64) * (b as f64)).sum();
            tsum += dot / nn.sqrt();
            nsum += nn.sqrt();
        }
        let t = tsum / xs.len() as f64;
        if t > best_overall.1 {
            best_overall = (s, t);
        }
        println!("{s:7.3}  {t:8.4}  {:8.2}", nsum / xs.len() as f64);
    }
    println!("\nmeilleure échelle {:.3}, t = {:.4}", best_overall.0, best_overall.1);
    println!("la grille du banc allait de 0.600 à {:.3}", 0.6 * 1.18f64.powi(11));
}
