//! The shell bound of the Tetra codebook: the largest `m = ‖y‖²/16` any word
//! decodes to, and the largest `|y_j|`. Both are what `llvq_tetra48.cuh` sizes
//! its inverse-norm table and its byte arithmetic against.
//!
//! Exhaustive over all 2^47 labels, by decomposition. Given `p` and the
//! codeword, the three sections are independent except through the classes:
//! section 1 reads class `r`, the middle section the mixed list, whose row
//! carries `δ`, and section 3 reads class `p ⊕ r ⊕ δ`. So the maximum is taken
//! per (parity, pattern byte, class) over the rows, then over the 4,096
//! codewords, both `p`, both `r` and both `δ`. The word that reaches it is
//! then decoded, and a sample of random words is checked under it.
//!
//! An earlier version swept each row with one residue for all eight
//! coordinates of a section. Real pattern bytes mix residues, so that sweep
//! was not a bound (it printed 144 per section, and a mixed byte reaches 152).

use llvq_search::tetra::{val, Tetra, CLASS_ROWS, N0_MIXED, SECTION};

/// Σy² of one section, and the row that reaches it, over `rows`.
fn best(rows: &[u32], p: u32, c: u8) -> (i64, usize) {
    let mut out = (-1i64, 0usize);
    for (i, &row) in rows.iter().enumerate() {
        let s: i64 = (0..SECTION)
            .map(|j| {
                let v = val(p + 2 * ((c >> j) & 1) as u32, (row >> (4 * j)) & 15) as i64;
                v * v
            })
            .sum();
        if s > out.0 {
            out = (s, i);
        }
    }
    out
}

fn n2(y: &[i32; 24]) -> i64 {
    y.iter().map(|&v| (v as i64) * (v as i64)).sum()
}

fn main() {
    let t = Tetra::new();
    let rows = t.rows();
    let class = |k: usize| &rows[CLASS_ROWS * k..CLASS_ROWS * (k + 1)];
    let mixed = |d: usize| if d == 0 { &rows[..N0_MIXED] } else { &rows[CLASS_ROWS..CLASS_ROWS + CLASS_ROWS - N0_MIXED] };

    // Per (p, byte, class): the best Σy² of an end section and of the middle one.
    let mut end = vec![[(0i64, 0usize); 2]; 2 * 256];
    let mut mid = vec![[(0i64, 0usize); 2]; 2 * 256];
    let mut max_abs = 0i32;
    for p in 0..2u32 {
        for c in 0..=255u8 {
            for k in 0..2 {
                end[256 * p as usize + c as usize][k] = best(class(k), p, c);
                mid[256 * p as usize + c as usize][k] = best(mixed(k), p, c);
            }
        }
        for &row in rows.iter() {
            for o in [p, p + 2] {
                for j in 0..SECTION {
                    max_abs = max_abs.max(val(o, (row >> (4 * j)) & 15).abs());
                }
            }
        }
    }

    // Every codeword, both p, both r, both δ.
    let mut top = (-1i64, 0u64);
    for p in 0..2u64 {
        for s8 in 0..64u64 {
            for b1 in 0..2u64 {
                for b2 in 0..16u64 {
                    for b3 in 0..2u64 {
                        let path = (s8 << 2) | (b1 << 8) | (b2 << 20) | (b3 << 35);
                        let (c1, c2, c3) = t.patterns(path);
                        let base = 256 * p as usize;
                        for r in 0..2u64 {
                            for d in 0..2u64 {
                                let r3 = (p ^ r ^ d) as usize;
                                let (e1, i1) = end[base + c1 as usize][r as usize];
                                let (e2, i2) = mid[base + c2 as usize][d as usize];
                                let (e3, i3) = end[base + c3 as usize][r3];
                                let s = e1 + e2 + e3;
                                if s > top.0 {
                                    let i2 = if d == 0 { i2 } else { N0_MIXED + i2 } as u64;
                                    let word = p | (r << 1) | path | ((i1 as u64) << 9) | (i2 << 24) | ((i3 as u64) << 36);
                                    top = (s, word);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    let (n2_max, word) = top;
    let y = t.decode(word);
    assert_eq!(n2(&y), n2_max, "the arg-max word does not decode to the maximum");
    assert_eq!(n2_max % 16, 0, "the maximum is not a multiple of 16");

    // A sample of random words must stay under it (splitmix64, fixed seed).
    let mut s = 0x7465_7472_6173_6865u64;
    let mut next = || {
        s = s.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    };
    let mut sample_max = 0i64;
    for _ in 0..1_000_000 {
        let y = t.decode(next() & ((1u64 << 47) - 1));
        sample_max = sample_max.max(n2(&y));
        assert!(y.iter().all(|v| v.abs() <= max_abs), "a coordinate exceeds max |y_j|");
    }
    assert!(sample_max <= n2_max, "a random word exceeds the exhaustive maximum");

    println!("rows              = {}", rows.len());
    println!("CLASS_ROWS        = {CLASS_ROWS}, N0_MIXED = {N0_MIXED}");
    println!("max |y_j|         = {max_abs}");
    println!("max n2, exhaustive = {n2_max}, reached by word {word:#014x}");
    println!("max m = n2/16     = {}", n2_max / 16);
    println!("1,000,000 random words: max n2 {sample_max}, none above");
    let o = t.order();
    println!("\ntrio -> natural, 24 entries:");
    print!("   ");
    for (j, v) in o.iter().enumerate() {
        print!("{v:>3},");
        if j % 12 == 11 {
            println!();
            print!("   ");
        }
    }
    println!();
}
