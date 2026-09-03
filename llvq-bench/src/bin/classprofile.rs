//! What do quantized blocks actually look like?
//!
//! The permutation rank exists to say *which value goes to which position*.
//! Its cost grows with the number of distinct values in a block: a block whose
//! coordinates take only two magnitudes needs a 24-bit mask, no divisions at
//! all. A block with five needs the full machinery.
//!
//! So before designing anything, measure the distribution. If most blocks are
//! simple, a fast path is worth having. If they are all complex, it is not,
//! and the format has to change rather than the code.
//!
//! Gaussian source, which is what the encoder sees after incoherence rotation
//! — that is the whole point of rotating.
//!
//! Run: `cargo run --release -p llvq-bench --bin classprofile`

use llvq_core::{Leech, SplitMix64, DIM};
use llvq_search::generic::BallSearcher;
use llvq_search::Searcher;
use std::collections::BTreeMap;

const N: usize = 20_000;

fn main() {
    let mut rng = SplitMix64::new(0x6_C1A5);
    let s = Searcher::new();
    let mut ball = BallSearcher::new();

    // Distinct |value| count → how many blocks, and which shells they sit on.
    let mut by_types: BTreeMap<usize, usize> = BTreeMap::new();
    let mut by_shell: BTreeMap<u32, usize> = BTreeMap::new();
    // For two-magnitude blocks: how many coordinates carry the larger one.
    let mut two_type_popcount: BTreeMap<usize, usize> = BTreeMap::new();
    let mut odd_coset = 0usize;

    for _ in 0..N {
        let x: [f64; DIM] = core::array::from_fn(|_| rng.next_gaussian());
        let f = ball.nearest_angular(&s, &x);

        let mut mags: Vec<i32> = f.point.iter().map(|v| v.abs()).collect();
        mags.sort_unstable();
        mags.dedup();
        *by_types.entry(mags.len()).or_default() += 1;
        *by_shell.entry(f.shell).or_default() += 1;
        if f.point.iter().all(|v| v % 2 != 0) {
            odd_coset += 1;
        }
        if mags.len() == 2 {
            let hi = mags[1];
            let k = f.point.iter().filter(|v| v.abs() == hi).count();
            *two_type_popcount.entry(k).or_default() += 1;
        }
        debug_assert_eq!(Leech::shell_index(&f.point), Some(f.shell as u64));
    }

    let pct = |n: usize| 100.0 * n as f64 / N as f64;

    println!("{N} gaussian blocks, quantized by angular search\n");
    println!("  distinct |xᵢ| values per block");
    println!("  {}", "-".repeat(46));
    let mut cum = 0usize;
    for (types, n) in &by_types {
        cum += n;
        println!(
            "  {types} type(s){:>12} blocks{:>8.1} %   cum {:>5.1} %",
            n,
            pct(*n),
            pct(cum)
        );
    }

    println!("\n  shell reached");
    println!("  {}", "-".repeat(46));
    for (shell, n) in &by_shell {
        println!("  m = {shell:<3}{:>15} blocks{:>8.1} %", n, pct(*n));
    }

    println!("\n  odd coset (every coordinate odd): {:.1} %", pct(odd_coset));

    if !two_type_popcount.is_empty() {
        println!("\n  2-magnitude blocks: how many coordinates carry the larger one");
        println!("  {}", "-".repeat(46));
        for (k, n) in &two_type_popcount {
            println!("  {k:>2} of 24{:>15} blocks{:>8.1} %", n, pct(*n));
        }
    }

    let simple: usize = by_types.iter().filter(|(t, _)| **t <= 2).map(|(_, n)| n).sum();
    println!(
        "\n  ≤ 2 magnitudes: {:.1} % of blocks — decodable by mask,\n  \
         with no permutation rank and no division.",
        pct(simple)
    );
}
