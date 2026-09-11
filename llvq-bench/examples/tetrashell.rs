//! The shell bound of the Tetra codebook: the largest `m = ‖y‖²/16` any word
//! decodes to, and the largest `|y_j|`. Both are what `llvq_tetra48.cuh` sizes
//! its inverse-norm table and its byte arithmetic against.

use llvq_search::tetra::{Tetra, CLASS_ROWS, N0_MIXED, SECTION};

fn main() {
    let t = Tetra::new();
    let rows = t.rows();
    // The worst section is the one whose row costs most, and the sections are
    // independent given `p` and the codeword bits. So sweep every row against
    // both residues of every parity, and take the three worst.
    let mut best_sec = 0i64;
    let mut max_abs = 0i32;
    for &row in rows.iter() {
        for p in 0..2u32 {
            for c in 0..2u32 {
                let o = p + 2 * c;
                let mut s = 0i64;
                for j in 0..SECTION {
                    let v = llvq_search::tetra::val(o, (row >> (4 * j)) & 15);
                    s += (v as i64) * (v as i64);
                    max_abs = max_abs.max(v.abs());
                }
                best_sec = best_sec.max(s);
            }
        }
    }
    let n2_max = 3 * best_sec;
    println!("rows            = {}", rows.len());
    println!("CLASS_ROWS      = {CLASS_ROWS}, N0_MIXED = {N0_MIXED}");
    println!("max |y_j|       = {max_abs}");
    println!("max section Σy² = {best_sec}");
    println!("bound n2        = {n2_max}   (3 worst sections, an over-bound)");
    println!("bound m = n2/16 = {}", n2_max / 16);
    assert_eq!(n2_max % 16, 0, "the bound is not a multiple of 16");
    let o = t.order();
    println!("\ntrio -> natural, 24 entries:");
    print!("   ");
    for (j, v) in o.iter().enumerate() {
        print!("{v:>3},");
        if j % 12 == 11 { println!(); print!("   "); }
    }
    println!();
}
// printed by a second pass below
