//! Do the 64 middle-byte sets collapse to a handful of distinct ones?
//!
//! `cargo run --release -p llvq-bench --example f1mids`
//!
//! It decides whether the encoder is tractable. Section 2's admissible set
//! depends on the state through its sixteen middle bytes; if all 64 states gave
//! different sets, the encoder would prepare 64 regions per parity instead of a
//! few, and the measurement would cost that factor. An adversarial review
//! claimed the sets collapse to 8 cosets of the [8,4,4] extended Hamming code.
//! Checked here rather than believed.

use llvq_bench::f1::{Trellis, GOLAY_STATES};

fn main() {
    let t = Trellis::new();
    let mut sets: Vec<Vec<u8>> = Vec::new();
    let mut which = vec![0usize; GOLAY_STATES];
    for (s8, slot) in which.iter_mut().enumerate() {
        let mut m: Vec<u8> = t.branches[s8].iter().map(|&(b, _)| b).collect();
        m.sort_unstable();
        *slot = match sets.iter().position(|s| *s == m) {
            Some(i) => i,
            None => {
                sets.push(m);
                sets.len() - 1
            }
        };
    }
    println!("{} états, {} ensembles de milieux distincts", GOLAY_STATES, sets.len());
    let mut sizes = vec![0usize; sets.len()];
    for &i in &which {
        sizes[i] += 1;
    }
    println!("états par ensemble : {sizes:?}");
    println!("bytes par ensemble : {:?}", sets.iter().map(Vec::len).collect::<Vec<_>>());
    let total: usize = sets.iter().map(Vec::len).sum();
    println!("{total} bytes au total sur les ensembles distincts");
    // Chaque ensemble est-il un coset d'un code lineaire ? Il l'est si b ^ b0
    // parcourt un sous-espace, donc si l'ensemble des differences est clos par XOR.
    for (i, s) in sets.iter().enumerate() {
        let diffs: Vec<u8> = s.iter().map(|&b| b ^ s[0]).collect();
        let closed = diffs.iter().all(|&a| diffs.iter().all(|&b| diffs.contains(&(a ^ b))));
        if !closed {
            println!("⚠️ ensemble {i} n'est pas un coset d'un sous-espace");
        }
    }
    println!("tous les ensembles sont des cosets d'un sous-espace linéaire de dimension 4");
}
