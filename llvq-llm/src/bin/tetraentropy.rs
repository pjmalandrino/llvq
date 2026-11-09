//! The empirical entropy of each field of a file's Tetra words, and its shell histogram.
//!
//! Usage: `cargo run --release -p llvq-llm --bin tetraentropy -- <file>`
//!
//! Reads every Tetra matrix of the file in the card's word order (`fused::load`, layout `Tetra48`) and counts,
//! over every block, the fields the decoder reads: p, r, the 12 Golay path bits, the three rank-row indices (with
//! their class bit), the gain bit, and the shell m = ‖y‖² / 16 of the decoded point. The sum of the marginal
//! entropies bounds what a field-by-field lossless code could store a block, against the 48 bits served
//! (docs/recherche-debit-maths.md, "Breaking the ceiling"). Nothing runs on a card; every number is *computed*.

use llvq_llm::fused::{self, FusedLayout, HostStream};

const N0_MIXED: u32 = 1240; // F1R_N0_MIXED, llvq_f1rank.cuh
                            // The Golay path to the three pattern bytes, linear (llvq_f1rank_v2.cuh, F1R_V2_COL_*).
const COLS: [u32; 12] = [
    0x2d002e, 0x3a005a, 0x740033, 0x03061e, 0x050963, 0x090578, 0x0000ff, 0x2d1d00, 0x3a2b00,
    0x744700, 0x638e00, 0xff0000,
];
// val(o, ρ), o = 0..3 (llvq_f1rank_v3.cuh, F1R_V3_T*).
const VAL: [[i32; 8]; 4] = [
    [0, 4, -4, 8, -8, 12, -12, 16],
    [1, -3, 5, -7, 9, -11, 13, -15],
    [2, -2, 6, -6, 10, -10, 14, -14],
    [-1, 3, -5, 7, -9, 11, -13, 15],
];

fn entropy(h: &[u64]) -> f64 {
    let n: u64 = h.iter().sum();
    h.iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / n as f64;
            -p * p.log2()
        })
        .sum()
}

fn main() -> Result<(), String> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: tetraentropy <file>")?;
    let tetra = llvq_search::tetra::Tetra::new();
    let rows = tetra.rows().to_vec();
    let m = fused::load(&path, FusedLayout::Tetra48)?;
    let (mut hp, mut hr, mut hg) = ([0u64; 2], [0u64; 2], [0u64; 2]);
    let mut hgol = vec![0u64; 4096];
    let (mut h1, mut h2, mut h3) = (vec![0u64; 4096], vec![0u64; 4096], vec![0u64; 4096]);
    let mut hm = vec![0u64; 64];
    let mut nblk = 0u64;
    for mx in &m.matrices {
        let HostStream::Tetra48 { words, stride_u32 } = &mx.stream else {
            continue;
        };
        let nb = mx.nblocks as u32;
        for row in 0..mx.d_out {
            let wrow = &words[row * *stride_u32 as usize..];
            for j in 0..nb {
                let w = (3 * j) >> 1;
                let (w0, w1) = (wrow[w as usize], wrow[w as usize + 1]);
                let (lo, hi) = if j & 1 == 1 {
                    ((w0 >> 16) | (w1 << 16), w1 >> 16)
                } else {
                    (w0, w1 & 0xffff)
                };
                let p = lo & 1;
                let r = (lo >> 1) & 1;
                let s8 = (lo >> 2) & 63;
                let b1 = (lo >> 8) & 1;
                let i1 = (lo >> 9) & 0x7ff;
                let b2 = (lo >> 20) & 15;
                let i2 = ((lo >> 24) | ((hi & 7) << 8)) & 0x7ff;
                let b3 = (hi >> 3) & 1;
                let i3 = (hi >> 4) & 0x7ff;
                let g = (hi >> 15) & 1;
                let u = s8 | b1 << 6 | b2 << 7 | b3 << 11;
                let mut c = 0u32;
                for (k, col) in COLS.iter().enumerate() {
                    if u >> k & 1 == 1 {
                        c ^= col;
                    }
                }
                let mid = i2 < N0_MIXED;
                let idx2 = if mid { i2 } else { 2048 - N0_MIXED + i2 };
                let r3 = (p ^ r ^ u32::from(!mid)) & 1;
                let idx = [2048 * r + i1, idx2, 2048 * r3 + i3];
                let mut n2 = 0i32;
                for (s, &ix) in idx.iter().enumerate() {
                    let row_bits = rows[ix as usize];
                    for jj in 0..8 {
                        let rho = (row_bits >> (4 * jj)) & 7;
                        let cj = (c >> (8 * s + jj)) & 1;
                        let v = VAL[(p + 2 * cj) as usize][rho as usize];
                        n2 += v * v;
                    }
                }
                hp[p as usize] += 1;
                hr[r as usize] += 1;
                hg[g as usize] += 1;
                hgol[u as usize] += 1;
                h1[idx[0] as usize] += 1;
                h2[idx[1] as usize] += 1;
                h3[idx[2] as usize] += 1;
                hm[((n2 / 16) as usize).min(63)] += 1;
                nblk += 1;
            }
        }
    }
    let f = [
        ("p", entropy(&hp), 1.0),
        ("r", entropy(&hr), 1.0),
        ("golay path", entropy(&hgol), 12.0),
        ("row 1 (r, i1)", entropy(&h1), 12.0),
        ("row 2 (idx2)", entropy(&h2), 12.0),
        ("row 3 (r3, i3)", entropy(&h3), 12.0),
        ("g", entropy(&hg), 1.0),
    ];
    println!("file {path}: {nblk} Tetra blocks");
    for (n, h, b) in &f {
        println!("  {n:16} {h:7.3} bits of {b:4.0}");
    }
    // r is inside row 1, r3 inside row 3: the stored fields are p, golay(12), i1+r, i2, i3, g = 48 bits.
    let stored = f[0].1 + f[2].1 + f[3].1 + entropy(&h2) + f[5].1 + f[6].1;
    println!("  sum of marginals over the stored fields (p, golay, r+i1, i2 as idx2, r3+i3, g): {stored:.3} of 48");
    println!(
        "  (row 3's class is a function of p, r and i2, so this sum overcounts it by up to 1 bit)"
    );
    println!("shell m histogram (m: share):");
    for (k, &c) in hm.iter().enumerate().filter(|(_, &c)| c > 0) {
        println!("  m {k:2}: {:.4}", c as f64 / nblk as f64);
    }
    println!("shell entropy {:.3} bits", entropy(&hm));
    Ok(())
}
