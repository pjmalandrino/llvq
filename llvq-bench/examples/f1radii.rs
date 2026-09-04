//! Truncation radii of the F1 sections, over every state and both parities.
//!
//! `cargo run --release -p llvq-bench --example f1radii`
//!
//! Cross-check rather than output: an adversarial review of the F1b spec
//! derived these independently, in Python, from the repository's own Golay
//! construction, and reported 88..96 for the end sections and 72..80 for the
//! middle. This reproduces them from the Rust implementation, which shares no
//! code with that one. A disagreement would mean one of the two builds a
//! different lattice.

fn main() {
    let t = llvq_bench::f1::Trellis::new();
    for (label, w, mk) in [
        ("section 1 (w=12)", 12u32, 0usize),
        ("section 2 (w=15)", 15, 1),
        ("section 3 (w=12)", 12, 2),
    ] {
        let (mut lo, mut hi) = (usize::MAX, 0usize);
        for s in 0..llvq_bench::f1::GOLAY_STATES {
            for p in 0..2u32 {
                let set = match mk {
                    0 => t.section1(s, p, 0),
                    1 => t.section2(s, p),
                    _ => t.section3(s, p, 0),
                };
                let r = set.region(w);
                lo = lo.min(r.rho2);
                hi = hi.max(r.rho2);
            }
        }
        println!("{label:18} rho2 dans {lo}..{hi}   rayon {:.1}..{:.1}", (lo as f64).sqrt(), (hi as f64).sqrt());
    }
}
