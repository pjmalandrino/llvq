//! A complete fast decoder for format v1 — same bits, same points, no u128,
//! no factorials, no allocations.
//!
//! `decfast` showed the unrank recurrence alone is 4.7× faster than the
//! factorial reference. This is the end-to-end version: class lookup,
//! mixed-radix split, both unranks, signs and placement, in fixed buffers —
//! everything `Indexer::decode` does, decoded from the same index to the same
//! point.
//!
//! Correctness comes first and is exhaustive where it matters: every class's
//! first and last index (383 × 2 boundary cases), plus 200 000 uniform draws,
//! all compared bit-for-bit against `Indexer::decode` before a single
//! measurement is taken.
//!
//! Run: `cargo run --release -p llvq-bench --bin decfull`

use llvq_core::{Golay, SplitMix64, DIM};
use llvq_search::classes::{enumerate_classes, gamma, ClassSet, MAX_SHELL};
use llvq_search::index::{Indexer, N13};
use std::time::Instant;

const MAXK: usize = 8;

// ---------------------------------------------------------------------------
// Class metadata, flattened into fixed-size fields.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Default)]
struct FastClass {
    end: u64, // offset + card, for the binary search
    offset: u64,
    odd: bool,
    // Even side.
    w: u32,
    p_req: u32,
    word_vals: [i32; 4],
    word_counts: [u8; 4],
    n_word: usize,
    free_vals: [i32; 4],
    off_counts: [u8; MAXK],
    n_off: usize,
    a_on: u64,
    a_off: u64,
    s_w: u64,
    s_f: u64,
    // Odd side.
    vals: [i32; MAXK],
    counts: [u8; MAXK],
    n_kinds: usize,
    m_arr: u64,
}

fn multinomial_u64(counts: &[u8]) -> u64 {
    let mut fact = [1u128; 25];
    for i in 1..25 {
        fact[i] = fact[i - 1] * i as u128;
    }
    let n: usize = counts.iter().map(|&c| c as usize).sum();
    let mut m = fact[n];
    for &c in counts {
        m /= fact[c as usize];
    }
    u64::try_from(m).expect("arrangement count fits u64")
}

/// Mirror of `Indexer::new`'s layout: shells ascending, even classes then odd
/// classes, enumeration order within each. Any divergence here fails the
/// exhaustive boundary check below.
fn build_classes() -> Vec<FastClass> {
    let ClassSet { even, odd } = enumerate_classes(MAX_SHELL);
    let mut out: Vec<FastClass> = Vec::new();
    for m in 2..=MAX_SHELL {
        for c in even.iter().filter(|c| c.shell == m) {
            let mut fc = FastClass {
                w: c.w,
                p_req: c.p_req as u32,
                ..Default::default()
            };
            fc.n_word = c.word_vals.len();
            for (i, &(v, n)) in c.word_vals.iter().enumerate() {
                fc.word_vals[i] = v as i32;
                fc.word_counts[i] = n;
            }
            let free_n: u32 = c.free_vals.iter().map(|&(_, n)| n as u32).sum();
            for (i, &(v, n)) in c.free_vals.iter().enumerate() {
                fc.free_vals[i] = v as i32;
                fc.off_counts[i] = n;
            }
            fc.off_counts[c.free_vals.len()] = (24 - c.w - free_n) as u8;
            fc.n_off = c.free_vals.len() + 1;
            fc.a_on = multinomial_u64(&fc.word_counts[..fc.n_word]);
            fc.a_off = multinomial_u64(&fc.off_counts[..fc.n_off]);
            fc.s_w = if c.w == 0 { 1 } else { 1u64 << (c.w - 1) };
            fc.s_f = 1u64 << free_n;
            let card = gamma(c.w) * fc.a_on * fc.a_off * fc.s_w * fc.s_f;
            fc.end = card; // fixed up below
            out.push(fc);
        }
        for c in odd.iter().filter(|c| c.shell == m) {
            let mut fc = FastClass {
                odd: true,
                ..Default::default()
            };
            fc.n_kinds = c.vals.len();
            for (i, &(v, n)) in c.vals.iter().enumerate() {
                fc.vals[i] = v as i32;
                fc.counts[i] = n;
            }
            fc.m_arr = multinomial_u64(&fc.counts[..fc.n_kinds]);
            fc.end = 4096 * fc.m_arr;
            out.push(fc);
        }
    }
    let mut off = 0u64;
    for fc in out.iter_mut() {
        fc.offset = off;
        off += fc.end;
        fc.end = off;
    }
    assert_eq!(off, N13, "class layout must cover exactly N(13)");
    out
}

// ---------------------------------------------------------------------------
// The decoder.
// ---------------------------------------------------------------------------

/// One multiply and one divide per candidate — `M' = M·c_j/n`, exact.
#[inline]
fn unrank_fast(mut rank: u64, counts: &mut [u8; MAXK], k: usize, m0: u64, out: &mut [u8]) {
    let mut m = m0;
    let mut n = out.len() as u64;
    for slot in out.iter_mut() {
        #[allow(clippy::needless_range_loop)] // `j` is written into `*slot`
        for j in 0..k {
            let c = counts[j] as u64;
            if c == 0 {
                continue;
            }
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
}

fn decode_fast(
    classes: &[FastClass],
    ends: &[u64],
    g: &Golay,
    idx: u64,
) -> Option<[i32; DIM]> {
    if idx == 0 {
        return Some([0; DIM]);
    }
    let i = idx - 1;
    if i >= N13 {
        return None;
    }
    let ci = ends.partition_point(|&e| e <= i);
    let fc = &classes[ci];
    let mut local = i - fc.offset;
    let mut p = [0i32; DIM];

    if fc.odd {
        let r_arr = local % fc.m_arr;
        let gi = (local / fc.m_arr) as usize;
        let c = g.codewords()[gi];
        let mut counts = fc.counts;
        let mut kinds = [0u8; DIM];
        unrank_fast(r_arr, &mut counts, fc.n_kinds, fc.m_arr, &mut kinds);
        for (i, pi) in p.iter_mut().enumerate() {
            let v = fc.vals[kinds[i] as usize];
            // Forced signs: value ≡ 3 (mod 4) is positive on the support,
            // negative off it; ≡ 1 (mod 4) the opposite.
            let a = if v % 4 == 3 { v } else { -v };
            *pi = if c >> i & 1 == 1 { a } else { -a };
        }
    } else {
        let r_sf = local % fc.s_f;
        local /= fc.s_f;
        let r_sw = local % fc.s_w;
        local /= fc.s_w;
        let r_off = local % fc.a_off;
        local /= fc.a_off;
        let r_on = local % fc.a_on;
        let gi = (local / fc.a_on) as usize;
        let c = g.of_weight(fc.w as usize)[gi];
        let w = fc.w as usize;

        let mut wc = [0u8; MAXK];
        wc[..4].copy_from_slice(&fc.word_counts);
        let mut on_kinds = [0u8; DIM];
        unrank_fast(r_on, &mut wc, fc.n_word, fc.a_on, &mut on_kinds[..w]);

        let mut oc = fc.off_counts;
        let mut off_kinds = [0u8; DIM];
        unrank_fast(r_off, &mut oc, fc.n_off, fc.a_off, &mut off_kinds[..DIM - w]);

        let mut sign_bits = [0u32; DIM];
        let mut par = 0u32;
        for (j, sb) in sign_bits.iter_mut().enumerate().take(w.saturating_sub(1)) {
            let b = (r_sw >> j & 1) as u32;
            *sb = b;
            par ^= b;
        }
        if w > 0 {
            sign_bits[w - 1] = par ^ fc.p_req;
        }

        let zero_kind = (fc.n_off - 1) as u8;
        let (mut s_on, mut s_off, mut fbit) = (0usize, 0usize, 0u32);
        for (i, pi) in p.iter_mut().enumerate() {
            if c >> i & 1 == 1 {
                let v = fc.word_vals[on_kinds[s_on] as usize];
                *pi = if sign_bits[s_on] == 1 { -v } else { v };
                s_on += 1;
            } else {
                let k = off_kinds[s_off];
                if k != zero_kind {
                    let v = fc.free_vals[k as usize];
                    *pi = if r_sf >> fbit & 1 == 1 { -v } else { v };
                    fbit += 1;
                }
                s_off += 1;
            }
        }
    }
    Some(p)
}

// ---------------------------------------------------------------------------

fn main() {
    let classes = build_classes();
    let ends: Vec<u64> = classes.iter().map(|c| c.end).collect();
    let g = Golay::new();
    let ix = Indexer::new();

    // ---- correctness: every class boundary, then 200 000 uniform draws ----
    let mut checked = 0usize;
    for fc in &classes {
        for idx in [fc.offset + 1, fc.end] {
            let a = ix.decode(idx).expect("valid");
            let b = decode_fast(&classes, &ends, &g, idx).expect("valid");
            assert_eq!(a, b, "boundary index {idx} disagrees");
            checked += 1;
        }
    }
    let mut rng = SplitMix64::new(0x6_F0FF);
    const N: usize = 200_000;
    let idx: Vec<u64> = (0..N).map(|_| 1 + rng.next() % N13).collect();
    for &i in &idx {
        let a = ix.decode(i).expect("valid");
        let b = decode_fast(&classes, &ends, &g, i).expect("valid");
        assert_eq!(a, b, "index {i} disagrees");
        checked += 1;
    }
    assert_eq!(decode_fast(&classes, &ends, &g, 0), Some([0; DIM]));
    assert_eq!(decode_fast(&classes, &ends, &g, N13 + 1), None);

    // ---- then time, both on the same indices ----
    let mut sink = 0i64;
    let t = Instant::now();
    for &i in &idx {
        sink += ix.decode(i).expect("valid")[0] as i64;
    }
    let t_ref = t.elapsed().as_secs_f64() / N as f64;

    let t = Instant::now();
    for &i in &idx {
        sink += decode_fast(&classes, &ends, &g, i).expect("valid")[0] as i64;
    }
    let t_fast = t.elapsed().as_secs_f64() / N as f64;

    println!(
        "décodage complet, {N} indices uniformes + {} bornes de classes (sink {sink})\n",
        2 * classes.len()
    );
    println!("  {:<40}{:>10}  {:>8}", "décodeur", "ns/bloc", "blocs/s");
    println!("  {}", "-".repeat(64));
    println!(
        "  {:<40}{:>10.1}  {:>10.2e}",
        "Indexer::decode (u128, factorielles, Vec)",
        t_ref * 1e9,
        1.0 / t_ref
    );
    println!(
        "  {:<40}{:>10.1}  {:>10.2e}",
        "decode_fast (u64, récurrence, buffers fixes)",
        t_fast * 1e9,
        1.0 / t_fast
    );
    println!("  {}", "-".repeat(64));
    println!("  {:>50.1}× plus rapide", t_ref / t_fast);
    println!(
        "\n  {checked} points comparés bit à bit — même format v1, mêmes 2,1595\n  \
         bits/poids sur le fichier, aucun bit ne change."
    );

    // What it means for the load-time transcode of a 4B model.
    let blocks_4b = 3_633_315_840u64 / 24;
    let secs = blocks_4b as f64 * t_fast / 12.0;
    println!(
        "\n  transcodage d'un 4B au chargement ({} blocs, 12 cœurs) : ~{secs:.1} s",
        blocks_4b
    );
}
