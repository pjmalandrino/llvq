//! Cost of the GPTQ factorization, at the widths the real models use.
//!
//! `GptqFactor::new` is four cubic passes: Cholesky of H, triangular inverse,
//! the L⁻ᵀL⁻¹ product, and a second Cholesky — about `2n³/3` multiply-adds in
//! total. That is the term that dominates a quantization run, and the one
//! that decides whether a model larger than 0.6B is tractable at all.

use llvq_core::SplitMix64;
use llvq_quant::linalg::GptqFactor;

/// A well-conditioned `AᵀA/N`, like a real calibration Hessian.
fn hessian(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = SplitMix64::new(seed);
    let rows = 2 * n;
    let a: Vec<f64> = (0..rows * n).map(|_| rng.next_gaussian()).collect();
    let mut h = vec![0.0f64; n * n];
    for r in 0..rows {
        let row = &a[r * n..(r + 1) * n];
        for i in 0..n {
            let ri = row[i];
            for j in 0..n {
                h[i * n + j] += ri * row[j];
            }
        }
    }
    let inv = 1.0 / rows as f64;
    for v in h.iter_mut() {
        *v *= inv;
    }
    h
}

fn main() {
    let widths: Vec<usize> = std::env::args()
        .skip(1)
        .filter_map(|s| s.parse().ok())
        .collect();
    let widths = if widths.is_empty() {
        vec![1024usize, 2048, 3072]
    } else {
        widths
    };

    println!("{:>7}  {:>12}  {:>10}  {:>14}", "n", "G mult-add", "s", "G mult-add/s");
    for n in widths {
        let h = hessian(n, 0xC0_1234 + n as u64);
        let t = std::time::Instant::now();
        let f = GptqFactor::new(&h, n, 1e-2).expect("SPD");
        let el = t.elapsed().as_secs_f64();
        std::hint::black_box(f.u(0, 0));
        let macs = 2.0 / 3.0 * (n as f64).powi(3) / 1e9;
        println!("{n:>7}  {macs:>12.2}  {el:>10.2}  {:>14.2}", macs / el);
    }
}
