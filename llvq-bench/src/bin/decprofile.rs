//! Where the 827 ns of a bijective decode actually go.
//!
//! `decbench` says the archive format costs 183× a trivial decoder. That is
//! the verdict; this is the diagnosis. Designing a kernel format without it
//! would be guessing at which part to remove.
//!
//! Each stage is timed on its own, reproduced here rather than instrumented in
//! place — `Indexer::decode` stays untouched, and a stage that turns out to be
//! free is one we do not need to redesign.
//!
//! ⚠️ Audited 2026-07-31: stage 4 understates its own tries (~2.2×, tiny
//! ranks break early) while overstating the per-call shape (4 kinds, n = 24),
//! and the 806 ns also hide ~100-150 ns of Vec allocations no stage shows.
//! The 71 % headline lands near the truth (~75-82 %) by accidental
//! compensation. Kept for the stage breakdown; `decfull` is the end-to-end
//! measurement that supersedes it.
//!
//! Run: `cargo run --release -p llvq-bench --bin decprofile`

use llvq_core::{SplitMix64, DIM};
use llvq_search::classes::{enumerate_classes, MAX_SHELL};
use llvq_search::index::{Indexer, N13};
use std::time::Instant;

const N: usize = 100_000;

/// `n! / Π counts[i]!`, the quantity `perm_unrank` recomputes at every slot.
/// Copied rather than imported: it is `pub(crate)` in `llvq-search`, and a
/// benchmark has no business widening an API.
fn multinomial(fact: &[u128; 25], counts: &[u8]) -> u128 {
    let n: usize = counts.iter().map(|&c| c as usize).sum();
    let mut m = fact[n];
    for &c in counts {
        m /= fact[c as usize];
    }
    m
}

/// The inner loop of `perm_unrank`: walk 24 slots, and at each one try the
/// kinds in turn, computing a multinomial per attempt.
fn perm_unrank(fact: &[u128; 25], mut rank: u128, counts: &mut [u8], out: &mut [u8]) {
    for slot in out.iter_mut() {
        for j in 0..counts.len() {
            if counts[j] == 0 {
                continue;
            }
            counts[j] -= 1;
            let m = multinomial(fact, counts);
            if rank < m {
                *slot = j as u8;
                break;
            }
            rank -= m;
            counts[j] += 1;
        }
    }
}

fn main() {
    let mut fact = [1u128; 25];
    for i in 1..25 {
        fact[i] = fact[i - 1] * i as u128;
    }

    let mut rng = SplitMix64::new(0x6_9807);
    let idx: Vec<u64> = (0..N).map(|_| 1 + rng.next() % N13).collect();

    // ---- stage 1: locate the class ----
    // The real decoder binary-searches 383 (offset, card) pairs. Rebuilding
    // the same offsets here gives the same memory access pattern.
    let cs = enumerate_classes(MAX_SHELL);
    let mut offsets: Vec<u64> = Vec::new();
    let mut acc = 0u64;
    for c in &cs.even {
        offsets.push(acc);
        acc += c.cardinality();
    }
    for c in &cs.odd {
        offsets.push(acc);
        acc += c.cardinality();
    }
    offsets.sort_unstable();
    let mut sink = 0u64;
    let t = Instant::now();
    for &i in &idx {
        sink += offsets.partition_point(|&o| o <= i) as u64;
    }
    let locate = t.elapsed().as_secs_f64() / N as f64;

    // ---- stage 2: the mixed-radix divisions ----
    // Four successive `%` and `/` on u128 to split the local index into
    // codeword rank, on-support arrangement, off-support arrangement and sign
    // patterns.
    let radices = [4096u128, 735471u128, 2704156u128, 2048u128];
    let t = Instant::now();
    for &i in &idx {
        let mut local = i as u128;
        for r in radices {
            sink = sink.wrapping_add((local % r) as u64);
            local /= r;
        }
    }
    let divide = t.elapsed().as_secs_f64() / N as f64;

    // ---- stage 3: one multinomial ----
    let counts = [3u8, 5, 8, 8];
    let t = Instant::now();
    for _ in 0..N {
        sink = sink.wrapping_add(multinomial(&fact, &counts) as u64);
    }
    let one_multinomial = t.elapsed().as_secs_f64() / N as f64;

    // ---- stage 4: a full permutation unrank over 24 slots ----
    let mut out = [0u8; DIM];
    let t = Instant::now();
    for &i in &idx {
        let mut c = counts;
        perm_unrank(&fact, i as u128 % 1000, &mut c, &mut out);
        sink = sink.wrapping_add(out[0] as u64);
    }
    let unrank = t.elapsed().as_secs_f64() / N as f64;

    // ---- the real thing ----
    let ix = Indexer::new();
    let t = Instant::now();
    for &i in &idx {
        let p = ix.decode(i).expect("valid");
        sink = sink.wrapping_add(p[0] as u64);
    }
    let full = t.elapsed().as_secs_f64() / N as f64;

    let ns = |x: f64| x * 1e9;
    let pct = |x: f64| 100.0 * x / full;
    println!("decode profile, {N} blocks, one core (sink {sink})\n");
    println!("  {:<34}{:>9} ns  {:>6}", "étape", "coût", "part");
    println!("  {}", "-".repeat(52));
    println!("  {:<34}{:>9.1} ns  {:>5.1} %", "1. localiser la classe (383)", ns(locate), pct(locate));
    println!("  {:<34}{:>9.1} ns  {:>5.1} %", "2. divisions mixed-radix (u128)", ns(divide), pct(divide));
    println!("  {:<34}{:>9.1} ns  {:>5.1} %", "3. un seul multinomial", ns(one_multinomial), pct(one_multinomial));
    println!("  {:<34}{:>9.1} ns  {:>5.1} %", "4. perm_unrank, 24 créneaux", ns(unrank), pct(unrank));
    println!("  {}", "-".repeat(52));
    println!("  {:<34}{:>9.1} ns  {:>5.1} %", "decode complet", ns(full), 100.0);
    println!(
        "\n  Le décodeur en fait DEUX (support et hors-support), plus les\n  \
         allocations que ce banc ne reproduit pas."
    );
    println!(
        "\n  Cible noyau : ~16 ns/bloc (65 opérations). Le poste à supprimer\n  \
         est celui qui domine — pas celui qu'on croyait."
    );
}
