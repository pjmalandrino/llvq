//! Are the three trellis maps affine over F₂ under the CURRENT numbering?
//!
//! `cargo run --release -p llvq-bench --example f1linear`
//!
//! It decides whether a decoder can replace the three chained small-table
//! loads of `llvq_f1rank.cuh` (`prefixes[2·s8+b1]` → `branches[16·s8+b2]` →
//! `suffixes[2·s16+b3]`) by F₂ algebra on the bits of the word. The maps are
//! linear in the cosets by construction; what is NOT given is that they are
//! linear in the NUMBERS the word stores: `Trellis::new` numbers the states
//! by sorting canonical coset representatives (`index_of`) and the branches by
//! sorting middle bytes. Checked here on the whole domain rather than argued:
//! every map is fitted to an affine model from its values at 0 and at the unit
//! vectors, then the model is compared to the table at EVERY input.
//!
//! The durable pin is `llvq_bench::f1::rank::TrellisLinear` and its tests;
//! this example prints the numbers a reader wants to see.

use llvq_bench::f1::{Trellis, BRANCHES, GOLAY_STATES};

/// Fit `f` on `n_in` input bits to `f(x) = ⊕ x_i·col_i ⊕ k`, then count the
/// inputs where the model and the table disagree. Returns (k, columns, bad).
fn fit(n_in: u32, f: &dyn Fn(u32) -> u32) -> (u32, Vec<u32>, usize) {
    let k = f(0);
    let cols: Vec<u32> = (0..n_in).map(|i| f(1 << i) ^ k).collect();
    let model = |x: u32| (0..n_in).fold(k, |a, i| if x >> i & 1 == 1 { a ^ cols[i as usize] } else { a });
    let bad = (0..1u32 << n_in).filter(|&x| f(x) != model(x)).count();
    (k, cols, bad)
}

fn main() {
    let t = Trellis::new();

    // (s8, b1) → c1 : x = s8 | b1 << 6, seven bits.
    let m1 = |x: u32| t.prefixes[(x & 63) as usize][(x >> 6 & 1) as usize] as u32;
    let (k1, c1, bad1) = fit(7, &m1);
    println!("(s8, b1) → c1        : {} inputs, {bad1} off the affine model, k = {k1:#04x}", 1 << 7);
    println!("    columns {:?}", c1.iter().map(|c| format!("{c:#04x}")).collect::<Vec<_>>());

    // (s8, b2) → (c2, s16) : x = s8 | b2 << 6, ten bits; output c2 | s16 << 8.
    let m2 = |x: u32| {
        let (c2, s16) = t.branches[(x & 63) as usize][(x >> 6 & 15) as usize];
        c2 as u32 | (s16 as u32) << 8
    };
    let (k2, c2, bad2) = fit(10, &m2);
    println!("(s8, b2) → (c2, s16) : {} inputs, {bad2} off the affine model, k = {k2:#06x}", 1 << 10);
    println!("    columns {:?}", c2.iter().map(|c| format!("{c:#06x}")).collect::<Vec<_>>());

    // (s16, b3) → c3 : x = s16 | b3 << 6, seven bits.
    let m3 = |x: u32| t.suffixes[(x & 63) as usize][(x >> 6 & 1) as usize] as u32;
    let (k3, c3, bad3) = fit(7, &m3);
    println!("(s16, b3) → c3       : {} inputs, {bad3} off the affine model, k = {k3:#04x}", 1 << 7);
    println!("    columns {:?}", c3.iter().map(|c| format!("{c:#04x}")).collect::<Vec<_>>());

    // The composition the kernel wants: (s8, b1, b2, b3) → c1 | c2 << 8 | c3 << 16,
    // x = s8 | b1 << 6 | b2 << 7 | b3 << 11, twelve bits, s16 never materialised.
    let m = |x: u32| {
        let (s8, b1, b2, b3) = ((x & 63) as usize, (x >> 6 & 1) as usize, (x >> 7 & 15) as usize, (x >> 11 & 1) as usize);
        let c1 = t.prefixes[s8][b1] as u32;
        let (c2, s16) = t.branches[s8][b2];
        let c3 = t.suffixes[s16 as usize][b3] as u32;
        c1 | (c2 as u32) << 8 | c3 << 16
    };
    let (k, cols, bad) = fit(12, &m);
    println!(
        "(s8, b1, b2, b3) → (c1, c2, c3) : {} inputs = {GOLAY_STATES} × 2 × {BRANCHES} × 2, {bad} off the affine model, k = {k:#08x}",
        1 << 12
    );
    println!("    columns {:?}", cols.iter().map(|c| format!("{c:#08x}")).collect::<Vec<_>>());

    // Why it holds, checked and not believed: the 64 sorted representatives at
    // each cut form a subspace listed in numeric order, so the index IS the
    // linear coordinate. Read back through the maps: s8 → (c1 at b1 = 0) and
    // s16 → (c3 at b3 = 0) must be linear with the zero state at zero.
    let sub8 = (0..GOLAY_STATES as u32).all(|a| (0..GOLAY_STATES as u32).all(|b| m1(a ^ b) == m1(a) ^ m1(b)));
    let sub16 = (0..GOLAY_STATES as u32).all(|a| (0..GOLAY_STATES as u32).all(|b| m3(a ^ b) == m3(a) ^ m3(b)));
    println!("s8 ↦ prefix is a group homomorphism over all 64 × 64 pairs: {sub8}; s16 ↦ suffix: {sub16}");

    let verdict = if bad1 + bad2 + bad3 + bad == 0 { "AFFINE (linear, k = 0) under the current numbering: no relabelling needed" } else { "NOT affine: a relabelling is needed" };
    println!("{verdict}");
}
