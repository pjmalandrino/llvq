//! What does it cost, in bits, to say *which value goes where* — without a
//! permutation rank?
//!
//! The rank is optimal, and it is also 71 % of the decode cost: reaching one
//! means dividing 128-bit integers sixty times per block. Every alternative
//! here trades bits for divisions; this measures the exchange rate on real
//! quantized blocks.
//!
//! ## The two cosets are not the same problem
//!
//! Λ₂₄ splits into an even and an odd coset, and v1 encodes them differently.
//! The **odd** coset takes a single rank over all 24 magnitudes — comparing a
//! global rank against masks is exactly right there. The **even** coset
//! encodes the codeword's rank within its weight bucket — log2 γ(w) bits,
//! 9.6 to 11.3 — and then two shorter ranks, on and off the support.
//! Charging it a global rank would overstate what v1 costs and flatter every
//! alternative.
//!
//! So they are measured apart. The codeword is common to both schemes and
//! cancels out, so it is excluded from both sides.
//!
//! Run: `cargo run --release -p llvq-bench --bin arrbits`

use llvq_core::{SplitMix64, DIM};
use llvq_search::generic::BallSearcher;
use llvq_search::Searcher;

const N: usize = 20_000;

fn lg_fact(k: usize) -> f64 {
    (1..=k).map(|i| (i as f64).log2()).sum()
}

/// `log2(n! / Π counts!)` — the information content of an arrangement.
fn rank_bits(counts: &[usize]) -> f64 {
    let n: usize = counts.iter().sum();
    lg_fact(n) - counts.iter().map(|&c| lg_fact(c)).sum::<f64>()
}

/// One mask over the still-unassigned slots per type, largest count first. The
/// last type needs no mask — it is whatever is left.
fn nested_mask_bits(counts: &[usize], slots: usize) -> f64 {
    let mut left = slots;
    let mut bits = 0.0;
    for &c in counts.iter().take(counts.len().saturating_sub(1)) {
        bits += left as f64;
        left -= c;
    }
    bits
}

/// Fixed-width type tag per slot. A single-type or empty side carries zero
/// information and is billed zero — `.max(1.0)` here used to overcharge it,
/// biasing the comparison against the one scheme that least deserved help.
fn positional_bits(n_types: usize, slots: usize) -> f64 {
    if n_types <= 1 || slots == 0 {
        return 0.0;
    }
    slots as f64 * (n_types as f64).log2().ceil()
}

/// Counts per type, descending, zeros dropped.
fn tally(kinds: &[usize], n_types: usize) -> Vec<usize> {
    let mut c = vec![0usize; n_types];
    for &k in kinds {
        c[k] += 1;
    }
    c.retain(|&n| n > 0);
    c.sort_unstable_by(|a, b| b.cmp(a));
    c
}

#[derive(Default)]
struct Acc {
    n: usize,
    opt: f64,
    mask: f64,
    pos: f64,
}

impl Acc {
    fn show(&self, title: &str, total: usize) {
        let k = self.n as f64;
        let (o, m, p) = (self.opt / k, self.mask / k, self.pos / k);
        println!(
            "\n  {title} — {} blocs ({:.1} %)",
            self.n,
            100.0 * k / total as f64
        );
        println!("  {}", "-".repeat(62));
        println!("  {:<24}{o:>9.1} bits", "rang optimal (v1)");
        println!(
            "  {:<24}{m:>9.1} bits{:>+9.1}{:>+11.3} b/poids",
            "masques imbriqués",
            m - o,
            (m - o) / DIM as f64
        );
        println!(
            "  {:<24}{p:>9.1} bits{:>+9.1}{:>+11.3} b/poids",
            "positionnel",
            p - o,
            (p - o) / DIM as f64
        );
    }
}

fn main() {
    let mut rng = SplitMix64::new(0x6_A88B);
    let s = Searcher::new();
    let mut ball = BallSearcher::new();

    let (mut odd, mut even) = (Acc::default(), Acc::default());
    let mut worst_types = 0usize;

    for _ in 0..N {
        let x: [f64; DIM] = core::array::from_fn(|_| rng.next_gaussian());
        let f = ball.nearest_angular(&s, &x);

        // Distinct magnitudes, descending, and the type index of each slot.
        let mut mags: Vec<i32> = f.point.iter().map(|v| v.abs()).collect();
        mags.sort_unstable_by(|a, b| b.cmp(a));
        mags.dedup();
        let kinds: Vec<usize> = f
            .point
            .iter()
            .map(|v| mags.iter().position(|m| *m == v.abs()).expect("present"))
            .collect();
        worst_types = worst_types.max(mags.len());

        if f.point.iter().all(|v| v % 2 != 0) {
            // Odd coset: one rank over all 24 slots. Signs carry no bits —
            // codeword membership forces them.
            let c = tally(&kinds, mags.len());
            odd.n += 1;
            odd.opt += rank_bits(&c);
            odd.mask += nested_mask_bits(&c, DIM);
            odd.pos += positional_bits(c.len(), DIM);
        } else {
            // Even coset: the support is where coordinates are ≢ 0 (mod 4) —
            // the Golay word. v1 ranks on and off it separately.
            let on: Vec<usize> = (0..DIM)
                .filter(|&i| f.point[i] % 4 != 0)
                .map(|i| kinds[i])
                .collect();
            let off: Vec<usize> = (0..DIM)
                .filter(|&i| f.point[i] % 4 == 0)
                .map(|i| kinds[i])
                .collect();
            let (con, coff) = (tally(&on, mags.len()), tally(&off, mags.len()));
            even.n += 1;
            even.opt += rank_bits(&con) + rank_bits(&coff);
            even.mask +=
                nested_mask_bits(&con, on.len()) + nested_mask_bits(&coff, off.len());
            even.pos +=
                positional_bits(con.len(), on.len()) + positional_bits(coff.len(), off.len());
        }
    }

    println!("{N} blocs gaussiens quantifiés — coût de l'arrangement seul");
    println!("  (le codeword Golay est commun aux schémas, donc exclu des deux)");
    odd.show("COSET IMPAIR", N);
    even.show("COSET PAIR", N);

    let tot = N as f64;
    let base = odd.opt + even.opt;
    let d = |x: f64| (x - base) / tot / DIM as f64;
    println!("\n  MOYENNE PONDÉRÉE");
    println!("  {}", "-".repeat(62));
    println!("  masques imbriqués{:>+34.3} bits/poids", d(odd.mask + even.mask));
    println!("  positionnel{:>+40.3} bits/poids", d(odd.pos + even.pos));

    println!("\n  jusqu'à {worst_types} magnitudes distinctes dans un bloc");
    println!(
        "\n  Ce Δ s'ajoute aux 2,1595 bits/poids pesés sur le fichier. Repère :\n  \
         au-delà de +0,35 on dépasse 2,5 bits/poids, et l'écart avec QTIP\n  \
         (2,000) devient difficile à défendre."
    );
}
