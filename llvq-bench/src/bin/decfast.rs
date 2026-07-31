//! Is the permutation rank slow — or just naively decoded?
//!
//! `decprofile` blamed the rank for ~71 % of the decode cost, and the next
//! step was about to be replacing it with a costlier-in-bits format. Before
//! paying +0.34 bits/weight, check the premise: v1's `perm_unrank` recomputes
//! each multinomial **from factorials, in u128** — five 128-bit divisions per
//! candidate kind, per slot. Neither the width nor the recomputation is
//! forced by the format:
//!
//! * every arrangement count is a factor of a class's local index, which is
//!   ≤ N13 < 2⁴⁸ — so u64 holds everything, and `m·c ≤ 2⁴⁸·24 < 2⁵³` holds
//!   the intermediate product too;
//! * removing one element of kind `j` from a multiset of `n` obeys
//!   `M' = M·c_j/n`, **exactly** (n divides M·c_j: M·c_j/n is the multinomial
//!   of the reduced multiset) — one multiply and one divide replace the whole
//!   factorial recomputation.
//!
//! Same format, same bits, bit-identical output — checked here against the
//! naive reference on every block before anything is timed.
//!
//! Run: `cargo run --release -p llvq-bench --bin decfast`

use llvq_core::{SplitMix64, DIM};
use llvq_search::generic::BallSearcher;
use llvq_search::Searcher;
use std::time::Instant;

const N: usize = 20_000;
/// Word kinds (≤3) + free kinds (≤4) + zero: 8 is comfortable.
const MAXK: usize = 8;

// ---------------------------------------------------------------------------
// Reference: the shape of index.rs's perm_unrank — factorials, u128, one
// multinomial recomputed per candidate.
// ---------------------------------------------------------------------------

fn multinomial_u128(fact: &[u128; 25], counts: &[u8]) -> u128 {
    let n: usize = counts.iter().map(|&c| c as usize).sum();
    let mut m = fact[n];
    for &c in counts {
        m /= fact[c as usize];
    }
    m
}

fn unrank_ref(fact: &[u128; 25], mut rank: u128, counts: &mut [u8], out: &mut [u8]) {
    for slot in out.iter_mut() {
        for j in 0..counts.len() {
            if counts[j] == 0 {
                continue;
            }
            counts[j] -= 1;
            let m = multinomial_u128(fact, counts);
            if rank < m {
                *slot = j as u8;
                break;
            }
            rank -= m;
            counts[j] += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// The recurrence: u64, one multiply + one divide per candidate.
// ---------------------------------------------------------------------------

/// `m0` is the multinomial of the initial counts — one u64 per class, read
/// from the class table in a real decoder. Returns the number of candidate
/// kinds tried, which is what a GPU port pays for.
fn unrank_fast(
    mut rank: u64,
    counts: &mut [u8; MAXK],
    k: usize,
    m0: u64,
    out: &mut [u8],
) -> u32 {
    let mut m = m0;
    let mut n = out.len() as u64;
    let mut tried = 0u32;
    for slot in out.iter_mut() {
        for j in 0..k {
            let c = counts[j] as u64;
            if c == 0 {
                continue;
            }
            tried += 1;
            // Exact: M·c_j/n is the multinomial of the multiset with one
            // element of kind j removed.
            let mj = m * c / n;
            if rank < mj {
                *slot = j as u8;
                counts[j] -= 1;
                m = mj;
                break;
            }
            rank -= mj;
        }
        n -= 1;
    }
    tried
}

// ---------------------------------------------------------------------------

/// One unrank problem, as the decoder meets it: a class's kind counts, the
/// initial multinomial, and a uniformly drawn rank.
struct Problem {
    counts: [u8; MAXK],
    k: usize,
    m0: u64,
    rank: u64,
    slots: usize,
}

fn make_problem(fact: &[u128; 25], kinds: &[usize], n_kinds: usize, rng: &mut SplitMix64) -> Problem {
    let mut counts = [0u8; MAXK];
    for &k in kinds {
        counts[k] += 1;
    }
    let m0_128 = multinomial_u128(fact, &counts[..n_kinds]);
    assert!(m0_128 < 1u128 << 53, "arrangement count exceeds u64-safe range");
    let m0 = m0_128 as u64;
    Problem {
        counts,
        k: n_kinds,
        m0,
        rank: if m0 > 0 { rng.next() % m0 } else { 0 },
        slots: kinds.len(),
    }
}

fn main() {
    let mut fact = [1u128; 25];
    for i in 1..25 {
        fact[i] = fact[i - 1] * i as u128;
    }

    let mut rng = SplitMix64::new(0x6_FA57);
    let s = Searcher::new();
    let mut ball = BallSearcher::new();

    // Real blocks → real unrank problems: one per odd block (24 slots), two
    // per even block (on-support, off-support), uniform random ranks.
    let mut problems: Vec<Problem> = Vec::new();
    let mut blocks = 0usize;
    for _ in 0..N {
        let x: [f64; DIM] = core::array::from_fn(|_| rng.next_gaussian());
        let f = ball.nearest_angular(&s, &x);
        blocks += 1;

        let mut mags: Vec<i32> = f.point.iter().map(|v| v.abs()).collect();
        mags.sort_unstable_by(|a, b| b.cmp(a));
        mags.dedup();
        let kind_of = |v: i32| mags.iter().position(|m| *m == v.abs()).expect("present");

        if f.point.iter().all(|v| v % 2 != 0) {
            let kinds: Vec<usize> = f.point.iter().map(|v| kind_of(*v)).collect();
            problems.push(make_problem(&fact, &kinds, mags.len(), &mut rng));
        } else {
            let on: Vec<usize> = f.point.iter().filter(|v| *v % 4 != 0).map(|v| kind_of(*v)).collect();
            let off: Vec<usize> = f.point.iter().filter(|v| *v % 4 == 0).map(|v| kind_of(*v)).collect();
            if !on.is_empty() {
                problems.push(make_problem(&fact, &on, mags.len(), &mut rng));
            }
            if !off.is_empty() {
                problems.push(make_problem(&fact, &off, mags.len(), &mut rng));
            }
        }
    }

    // ---- correctness first: bit-identical output on every problem ----
    let mut tries = 0u64;
    let mut out_a = [0u8; DIM];
    let mut out_b = [0u8; DIM];
    for p in &problems {
        let mut ca: Vec<u8> = p.counts[..p.k].to_vec();
        unrank_ref(&fact, p.rank as u128, &mut ca, &mut out_a[..p.slots]);
        let mut cb = p.counts;
        tries += unrank_fast(p.rank, &mut cb, p.k, p.m0, &mut out_b[..p.slots]) as u64;
        assert_eq!(
            out_a[..p.slots],
            out_b[..p.slots],
            "the recurrence disagrees with the factorial reference"
        );
    }

    // ---- then time, separately ----
    let mut sink = 0u64;
    let t = Instant::now();
    for p in &problems {
        let mut c: Vec<u8> = p.counts[..p.k].to_vec();
        unrank_ref(&fact, p.rank as u128, &mut c, &mut out_a[..p.slots]);
        sink = sink.wrapping_add(out_a[0] as u64);
    }
    let t_ref = t.elapsed().as_secs_f64();

    let t = Instant::now();
    for p in &problems {
        let mut c = p.counts;
        unrank_fast(p.rank, &mut c, p.k, p.m0, &mut out_b[..p.slots]);
        sink = sink.wrapping_add(out_b[0] as u64);
    }
    let t_fast = t.elapsed().as_secs_f64();

    let per_block_ref = t_ref / blocks as f64 * 1e9;
    let per_block_fast = t_fast / blocks as f64 * 1e9;
    let total_slots: usize = problems.iter().map(|p| p.slots).sum();
    let avg_tries = tries as f64 / total_slots as f64;
    let tries_per_block = tries as f64 / blocks as f64;

    println!(
        "{} problèmes d'unrank issus de {blocks} blocs réels (sink {sink})\n",
        problems.len()
    );
    println!("  {:<38}{:>10}  {:>12}", "variante", "ns/bloc", "vs réf");
    println!("  {}", "-".repeat(64));
    println!(
        "  {:<38}{per_block_ref:>10.1}  {:>12}",
        "référence (u128, factorielles)", "1,0×"
    );
    println!(
        "  {:<38}{per_block_fast:>10.1}  {:>11.1}×",
        "récurrence (u64, M·c/n)",
        per_block_ref / per_block_fast
    );
    println!("  {}", "-".repeat(64));
    println!(
        "\n  sortie identique bit à bit sur les {} problèmes — même format,\n  \
         mêmes bits, seule l'implémentation change",
        problems.len()
    );
    println!(
        "\n  candidats essayés : {avg_tries:.2} par créneau en moyenne,\n  \
         {tries_per_block:.1} par bloc — c'est ce qu'un portage GPU paie :\n  \
         à ~11 opérations entières par candidat (mul64, mulhi-magic pour la\n  \
         division par n ≤ 24, comparaison), compter ~{:.0} opérations par bloc.",
        tries_per_block * 11.0
    );
}
