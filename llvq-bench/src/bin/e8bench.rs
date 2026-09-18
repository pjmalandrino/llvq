//! Stage 0 of the E8 arbitration: the invariants, the rate ladder, and the
//! normalized second moment measured against the value the dossier cites.
//!
//! ```text
//! cargo run --release -p llvq-bench --bin e8bench
//! ```
//!
//! ## What it decides
//!
//! `docs/arbitrage-e8-2026-09-18.md` rests on two constants taken from the
//! literature and transcribed nowhere in this repository: G(E8) = 0.071682 and
//! G(Leech) = 0.065771. The whole 9.0 % bound comes from their ratio, so the
//! dossier's own section 2 calls verifying them stage 0. This verifies the E8
//! one, which is the one a decoder here can reach.
//!
//! ## Why the cubic control is not decoration
//!
//! A Monte-Carlo second moment can be wrong in the estimator as easily as in
//! the lattice: a bad sampling region, a normalization by the wrong covolume,
//! or a decoder that quietly returns the input. So the same harness measures
//! the integer lattice first, whose G is exactly 1/12, by a decoder that is
//! one `round`. If the control misses 1/12 nothing below is worth reading.
//!
//! Sampling is uniform over `[0, 2)^8`. That is a fundamental domain of
//! `(2Z)^8`, which sits inside E8, so the draw is uniform modulo E8 and no
//! Voronoi cell has to be constructed.

use llvq_bench::e8::{enumerate, in_e8, nearest, qnorm, E8DIM};

const SEED: u64 = 0x2026_0918_00e8;

struct Lcg(u64);

impl Lcg {
    fn f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn sigma3(n: i32) -> i32 {
    (1..=n).filter(|d| n % d == 0).map(|d| d * d * d).sum()
}

/// Mean and standard error of the squared error, over `n` uniform draws.
fn second_moment(n: usize, side: f64, q: impl Fn(&[f64; E8DIM]) -> [f64; E8DIM]) -> (f64, f64) {
    let mut rng = Lcg(SEED);
    let (mut s, mut s2) = (0.0f64, 0.0f64);
    for _ in 0..n {
        let x: [f64; E8DIM] = std::array::from_fn(|_| rng.f64() * side);
        let p = q(&x);
        let e: f64 = (0..E8DIM).map(|i| (x[i] - p[i]) * (x[i] - p[i])).sum();
        s += e;
        s2 += e * e;
    }
    let m = s / n as f64;
    let var = (s2 / n as f64 - m * m).max(0.0);
    (m, (var / n as f64).sqrt())
}

fn main() {
    // `e8bench dump <path>` writes the verified codebook for the stage 1
    // analysis to read. The analysis re-checks what it reads, because a
    // codebook that crossed a file boundary is a codebook that can have been
    // truncated.
    let a: Vec<String> = std::env::args().collect();
    if (a.len() == 3 || a.len() == 4) && a[1] == "dump" {
        let cap: i32 = a.get(3).map(|v| v.parse().expect("cap")).unwrap_or(10);
        let book = enumerate(cap);
        let mut out = String::with_capacity(book.len() * 24);
        for y in &book {
            for (i, c) in y.iter().enumerate() {
                if i > 0 {
                    out.push(' ');
                }
                out.push_str(&c.to_string());
            }
            out.push('\n');
        }
        std::fs::write(&a[2], out).expect("write the codebook");
        println!("wrote {} points of E8 (norm <= {cap}) in doubled coordinates to {}", book.len(), a[2]);
        return;
    }

    println!("E8 stage 0 — invariants, rate ladder, normalized second moment");
    println!("seed {SEED:#x}, label: measured for the moments, computed for the counts\n");

    println!("1. INVARIANTS");
    let all = enumerate(10);
    let mut cum = 0usize;
    println!("   norm   vectors    240*sigma3   cumulative   bits/8dim   index   b/weight of a 24-block");
    for n in 1..=5i32 {
        let got = all.iter().filter(|y| qnorm(y) == 8 * n).count();
        let want = 240 * sigma3(n);
        cum += got;
        let bits = (cum as f64).log2();
        let idx = bits.ceil() as usize;
        println!(
            "   {:4}   {:7}   {:10}   {:10}   {:9.3}   {:5}   {:.4}   (+1 gain bit {:.4})",
            2 * n,
            got,
            want,
            cum,
            bits,
            idx,
            3.0 * idx as f64 / 24.0,
            (3.0 * idx as f64 + 1.0) / 24.0
        );
        assert_eq!(got as i32, want, "theta coefficient of E8 at norm {}", 2 * n);
    }
    assert_eq!(all.len(), 56_880);
    assert!(all.iter().all(in_e8));
    println!("   kissing number {} , all {} points in E8, minimum |x|^2 = {}",
        all.iter().filter(|y| qnorm(y) == 8).count(),
        all.len(),
        all.iter().map(qnorm).min().unwrap() / 4);
    println!("   Tetra's measured stream, for comparison: 1.9907 b/weight\n");

    println!("2. THE CONTROL: the integer lattice, whose G is exactly 1/12");
    let n = 10_000_000usize;
    let (m, se) = second_moment(n, 1.0, |x| std::array::from_fn(|i| x[i].round()));
    let g = m / E8DIM as f64;
    println!("   {n} draws, uniform over [0,1)^8, decoder = round");
    println!("   E|e|^2 {m:.6} +- {se:.6}   G = {g:.6}   exact 1/12 = {:.6}   gap {:+.3} %",
        1.0 / 12.0, 100.0 * (g * 12.0 - 1.0));

    println!("\n3. E8");
    let (m, se) = second_moment(n, 2.0, |x| {
        let y = nearest(x);
        std::array::from_fn(|i| y[i] as f64 / 2.0)
    });
    let g = m / E8DIM as f64;
    let cited = 0.071682;
    println!("   {n} draws, uniform over [0,2)^8, decoder = Conway and Sloane");
    println!("   E|e|^2 {m:.6} +- {se:.6}   G = {g:.6}   cited {cited:.6}   gap {:+.3} %",
        100.0 * (g / cited - 1.0));
    let z = (g - cited) / (se / E8DIM as f64);
    println!("   distance to the cited value: {z:+.2} standard errors");

    println!("\n4. WHAT THE RATIO BOUNDS");
    let leech = 0.065771;
    println!("   G(Leech) {leech:.6} is CITED and unverified here: this repository has no");
    println!("   infinite-Lambda24 decoder, every search is shell-capped.");
    println!("   ratio G(E8)/G(Leech) = {:.4} on the measured E8, {:.4} on the cited one",
        g / leech, cited / leech);
    println!("   so Leech leads by {:.1} % of MSE, {:.3} dB",
        100.0 * (g / leech - 1.0), 10.0 * (g / leech).log10());
}
