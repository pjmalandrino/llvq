//! How expensive is decoding, really — the number G6 turns on.
//!
//! The fused kernel's whole premise is that decoding a block hides inside the
//! time spent waiting for memory. On an M3 Max that budget is roughly 65
//! operations per 24-weight block. The archive format's decoder walks a
//! multiset-permutation rank with 128-bit divisions, which is visibly not
//! that — but "visibly not" is an estimate, and this project has been wrong
//! about estimates three times in two days.
//!
//! So: measure. Two figures, because each answers a different question.
//!
//! * **Absolute throughput** — blocks per second on one core. Compare against
//!   the ~60 G blocks/s a 4B model at 400 tokens/s would need. A GPU is
//!   roughly one to two orders of magnitude above one CPU core on this kind of
//!   work, so a CPU figure ten thousand times short is a verdict, not a hint.
//!
//! * **Ratio against a trivial decoder** — a Golay codeword rebuilt by XOR of
//!   12 masks, which is what a kernel-friendly format would cost. This ratio
//!   is what a new format has to close, and unlike the absolute figure it is
//!   insensitive to what else the machine is doing.
//!
//! Run: `cargo run --release -p llvq-bench --bin decbench`

use llvq_core::{Golay, SplitMix64, DIM};
use llvq_search::index::{Indexer, N13};
use std::time::Instant;

/// A decoder a GPU would not mind: rebuild a Golay codeword from 12 information
/// bits by XOR-ing the generator's masks, then place ±1 by sign bits.
///
/// This is not a Leech decoder — it is a *floor*, the cost of the cheapest
/// structured thing one could put in a kernel. It exists to be divided by.
#[inline(never)]
fn trivial_decode(masks: &[u32; 12], info: u32, signs: u32, out: &mut [i32; DIM]) {
    let mut c = 0u32;
    for (b, m) in masks.iter().enumerate() {
        if info >> b & 1 == 1 {
            c ^= *m;
        }
    }
    for (i, o) in out.iter_mut().enumerate() {
        let bit = (c >> i & 1) as i32;
        let sgn = 1 - 2 * ((signs >> i & 1) as i32);
        *o = sgn * (1 + 2 * bit);
    }
}

fn main() {
    const N: usize = 200_000;
    let mut rng = SplitMix64::new(0x6_DEC0);

    // Indices drawn across the whole ball, so shell lookup and class search
    // are exercised the way a real model would exercise them.
    let idx: Vec<u64> = (0..N).map(|_| 1 + rng.next() % N13).collect();

    let ix = Indexer::new();
    // Warm the caches and prove the indices are all valid before timing.
    let mut checksum = 0i64;
    for &i in idx.iter().take(1000) {
        checksum += ix.decode(i).expect("valid index")[0] as i64;
    }

    let t0 = Instant::now();
    for &i in &idx {
        let p = ix.decode(i).expect("valid index");
        checksum += p[0] as i64;
    }
    let bijective = t0.elapsed().as_secs_f64();

    // The floor.
    let g = Golay::new();
    // Any twelve codewords will do: the point is the *cost* of twelve XORs,
    // not which basis they span.
    let cw = g.codewords();
    let mut masks = [0u32; 12];
    for (b, m) in masks.iter_mut().enumerate() {
        *m = cw[1usize << b];
    }
    let mut out = [0i32; DIM];
    let t1 = Instant::now();
    for &i in &idx {
        trivial_decode(&masks, (i & 0xFFF) as u32, (i >> 12) as u32, &mut out);
        checksum += out[0] as i64;
    }
    let trivial = t1.elapsed().as_secs_f64();

    let bij_per = bijective / N as f64;
    let triv_per = trivial / N as f64;

    println!("decode of {N} blocks, one core (checksum {checksum})\n");
    println!("  format v1 (bijective)   {:>10.1} ns/block   {:>12.0} blocks/s",
             bij_per * 1e9, 1.0 / bij_per);
    println!("  trivial (Golay XOR)     {:>10.1} ns/block   {:>12.0} blocks/s",
             triv_per * 1e9, 1.0 / triv_per);
    println!("\n  ratio                   {:>10.1}×   ← what a kernel format must close",
             bij_per / triv_per);

    // What the kernel actually needs, for scale.
    const NEED: f64 = 60e9;
    println!("\n  a 4B model at ~400 tok/s needs   {:>12.0} blocks/s", NEED);
    println!("  one CPU core, format v1, is short by a factor of {:>8.0}",
             NEED * bij_per);
    println!(
        "\n  ⚠️  A GPU is one to two orders of magnitude above one CPU core on\n  \
         this kind of work — integer division and data-dependent branching are\n  \
         its worst case, not its best. Treat the CPU figure as an optimistic\n  \
         bound on what a GPU port of the same algorithm would reach."
    );
}
