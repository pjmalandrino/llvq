//! F1a: how many trellis states a three-section reading of Λ₂₄ carries.
//!
//! F1 proposes to stop unfolding by coding a block as
//! `[state][s₁][s₂][s₃][gain]`, three sections of eight coordinates chained by
//! a state, each section decoded through its own small table
//! (`docs/ROADMAP.md` §2.2). Its adoption gate is a **size**: tables of at most
//! 16 KiB per section. The state count is the first half of that budget and it
//! is exact arithmetic on the code this repository already carries — no
//! measurement, no card, no estimate.
//!
//! ## What a state is, and why it is countable here
//!
//! For a linear code `C` of length n and a cut after `i` coordinates, the
//! minimal trellis carries
//!
//! ```text
//!   |S_i| = 2^(k − k_past(i) − k_future(i))
//! ```
//!
//! where `k_past(i)` is the dimension of the subcode living entirely on the
//! first `i` coordinates and `k_future(i)` the dimension of the one living
//! entirely on the last `n − i`. Both are counted here by enumeration over the
//! 4,096 Golay codewords `llvq-core` already builds and the G1 suite already
//! pins against the published weight distribution, so the numbers inherit that
//! validation rather than asserting their own.
//!
//! Λ₂₄ adds two bits to the code's state, and no more. A point of the integer
//! embedding is `xⱼ = p + 2cⱼ + 4kⱼ` with `p` the shared parity, `c` a Golay
//! codeword and `Σk ≡ p (mod 2)` — the third constraint of
//! `llvq_core::Leech::contains`, reduced in its own comment. A later section
//! therefore needs exactly `p` (to lay down its coordinates) and the running
//! parity of `Σk` (to know what the remainder owes). Two bits, four states,
//! multiplying the code's own count.
//!
//! ## The ordering is a design choice, not a given
//!
//! `k_past` and `k_future` depend on **which** coordinates the cut separates:
//! an octad sitting inside the first eight makes `k_past(8) = 1` where a
//! scattered one leaves it at 0. So the state count of a three-section reading
//! is a property of the coordinate order, and F1 gets to pick that order. This
//! binary prints the whole profile under the repository's natural order, which
//! is the one `Leech::contains` and every packed index already use.

use llvq_core::Golay;

/// `(k_past, k_future, state bits)` at a cut after `i` of the 24 coordinates.
fn profile(codewords: &[u32], i: usize) -> (u32, u32, u32) {
    let low = if i >= 32 { u32::MAX } else { (1u32 << i) - 1 };
    // Subcodes are linear, so counting members gives the dimension outright.
    let past = codewords.iter().filter(|&&c| c & !low == 0).count();
    let future = codewords.iter().filter(|&&c| c & low == 0).count();
    let (kp, kf) = (past.trailing_zeros(), future.trailing_zeros());
    debug_assert!(past.is_power_of_two() && future.is_power_of_two());
    (kp, kf, 12 - kp - kf)
}

fn main() {
    let g = Golay::new();
    let cw = g.codewords();
    assert_eq!(cw.len(), 4096, "the Golay code has 2^12 words");

    println!("F1a — state complexity of a sectioned reading of Λ₂₄");
    println!("Golay [24,12,8] from llvq-core, {} codewords, natural order\n", cw.len());

    println!("  cut   k_past  k_future   Golay states   Λ₂₄ states (×4)");
    println!("  ----  ------  --------   ------------   ---------------");
    let mut worst = 0u32;
    for i in 0..=24 {
        let (kp, kf, bits) = profile(cw, i);
        worst = worst.max(bits);
        let mark = if i == 8 || i == 16 { "  <- section boundary" } else { "" };
        println!(
            "  {i:>4}  {kp:>6}  {kf:>8}   2^{bits:<2} = {:<6}   2^{} = {:<6}{mark}",
            1u32 << bits,
            bits + 2,
            1u32 << (bits + 2),
            );
    }

    let (_, _, b8) = profile(cw, 8);
    let (_, _, b16) = profile(cw, 16);
    println!("\nPeak over every cut: 2^{worst} Golay states, 2^{} for Λ₂₄.", worst + 2);
    println!(
        "At the two section boundaries F1 actually cuts: 2^{} and 2^{} states.",
        b8 + 2,
        b16 + 2
    );

    // ---- the ordering that the roadmap's 8-bit state field presupposes ----
    //
    // Raising `k_past(8) + k_future(8)` is the only way to cut the state count,
    // and the lever is the coordinate order. The extreme case is a cut that
    // isolates an **octad**: the code shortened on an octad is the [16, 5, 8]
    // first-order Reed-Muller code, so `k_future(8)` climbs from 4 to 5 while
    // `k_past(8)` climbs from 0 to 1. Applying that at both cuts asks the 24
    // coordinates to split into three disjoint octads — a *trio*, a classical
    // object of the Golay code. This searches for one and re-measures rather
    // than assuming either that it exists or that it helps.
    let octads = g.of_weight(8);
    println!("\nOctads in the code: {}", octads.len());
    let trio = octads.iter().find_map(|&a| {
        octads.iter().find_map(|&b| {
            let c = !(a | b) & 0xff_ffff;
            (a & b == 0 && c.count_ones() == 8 && g.contains(c)).then_some((a, b, c))
        })
    });
    let Some((a, b, c)) = trio else {
        println!("No trio found: the 8-bit state field is unreachable by reordering.");
        return;
    };
    println!("A trio: {a:#08x} | {b:#08x} | {c:#08x}  (disjoint octads covering the 24)");

    // Relabel so that the trio's three octads become coordinates 0-7, 8-15,
    // 16-23, then recount. A permutation of coordinates is an isometry of the
    // ambient space, so the lattice it yields is Λ₂₄ again — but it is coded by
    // a *different* index map, which is why this is a format change and not a
    // reading convention (`codebook_fingerprint` pins the map).
    let order: Vec<u32> = (0..24)
        .filter(|i| a >> i & 1 == 1)
        .chain((0..24).filter(|i| b >> i & 1 == 1))
        .chain((0..24).filter(|i| c >> i & 1 == 1))
        .collect();
    let permuted: Vec<u32> = cw
        .iter()
        .map(|&w| order.iter().enumerate().fold(0u32, |acc, (j, &i)| acc | (w >> i & 1) << j))
        .collect();

    println!("\n  cut   k_past  k_future   Golay states   Λ₂₄ states (×4)");
    println!("  ----  ------  --------   ------------   ---------------");
    for i in [8usize, 12, 16] {
        let (kp, kf, bits) = profile(&permuted, i);
        println!(
            "  {i:>4}  {kp:>6}  {kf:>8}   2^{bits:<2} = {:<6}   2^{} = {:<6}",
            1u32 << bits,
            bits + 2,
            1u32 << (bits + 2)
        );
    }
    let (_, _, t8) = profile(&permuted, 8);
    let (_, _, t16) = profile(&permuted, 16);
    let natural = 1u64 << (b8 + 2);
    let trio_states = 1u64 << (t8 + 2);
    println!(
        "\nState field: {} bits under the natural order, {} bits under the trio.",
        b8 + 2,
        t8 + 2
    );
    assert_eq!(t8, t16, "a trio must cut both boundaries the same way");

    println!(
        "\nGate arithmetic (docs/ROADMAP.md §2.2, tables ≤ 16 KiB per section).\n\
         A section table is indexed by (state, label). With {trio_states} states a\n\
         16 KiB budget leaves {} bytes per state, and an E₈ point costs 8\n\
         coordinates — so at one byte per coordinate a state may carry {} labels,\n\
         at four bits per coordinate {}. The roadmap sizes the label field at\n\
         ~13 bits, i.e. 8,192 values.",
        16 * 1024 / trio_states,
        16 * 1024 / trio_states / 8,
        16 * 1024 / trio_states / 4
    );
    println!(
        "Under the natural order the same budget gives {} bytes per state.",
        16 * 1024 / natural
    );

    // ---- how many branches actually leave a state ----
    //
    // The budget above assumed the worst reading of the word: that the whole
    // ~13-bit label indexes a table of E₈ points. A coset code does not work
    // that way. Only the **coded** bits pick a branch, and the branch count is
    // what the table has to hold; the rest of the label addresses a point
    // inside the chosen coset, which is arithmetic and not a table.
    //
    // So count the edges. Two codewords sit in the same state at a cut when
    // their difference lies in `L_past ⊕ L_future`, so a canonical coset
    // representative is the smallest word of `c ⊕ V`. `V` has 64 elements
    // here, which makes the reduction exhaustive and exact rather than a
    // basis-reduction that could be subtly wrong.
    let subspace = |i: usize| -> Vec<u32> {
        let low = (1u32 << i) - 1;
        let gens: Vec<u32> =
            permuted.iter().copied().filter(|&c| c & !low == 0 || c & low == 0).collect();
        let mut v = vec![0u32];
        for g in gens {
            if !v.contains(&g) {
                let grown: Vec<u32> = v.iter().map(|&x| x ^ g).collect();
                v.extend(grown);
            }
        }
        v.sort_unstable();
        v.dedup();
        v
    };
    let (v8, v16) = (subspace(8), subspace(16));
    let coset = |c: u32, v: &[u32]| v.iter().map(|&x| c ^ x).min().expect("non-empty");

    let mut edges: Vec<(u32, u32, u32)> = permuted
        .iter()
        .map(|&c| (coset(c, &v8), c >> 8 & 0xff, coset(c, &v16)))
        .collect();
    edges.sort_unstable();
    edges.dedup();
    // ⚠️ The denominator is the GOLAY state count, not the Λ₂₄ one. `edges` is
    // built from Golay codewords and Golay cosets, so there are 2^6 = 64 source
    // states in it, not the 256 that Λ₂₄ carries once the two extra bits are
    // added. Dividing 1,024 edges by 256 was the arithmetic error of the first
    // run of this file (docs/mesures/f1a-comptes-2026-09-04.txt, corrected in
    // its own ÉCARTS): it under-reported the out-degree by a factor of four.
    let golay_states = 1u64 << t8;
    let branches = edges.len() as u64 / golay_states;
    println!("\n(sanity: {} Golay states at the cut, {trio_states} Λ₂₄ states)", golay_states);
    println!(
        "Middle section: {} distinct edges over {golay_states} Golay states = {branches} branches per state\n\
         (|V₈| = {}, |V₁₆| = {}, exhaustive coset reduction).",
        edges.len(),
        v8.len(),
        v16.len()
    );
    let bytes_1b = trio_states * branches * 8;
    println!(
        "A branch table of (state, branch) → 8 coordinates therefore weighs\n\
         {trio_states} × {branches} × 8 = {} bytes at one byte per coordinate ({:.1} KiB),\n\
         and {} bytes at four bits ({:.1} KiB). The gate is 16 KiB.",
        bytes_1b,
        bytes_1b as f64 / 1024.0,
        bytes_1b / 2,
        bytes_1b as f64 / 2048.0
    );
    println!(
        "\nThe label field therefore splits: {} coded bits pick the branch, the rest\n\
         of the ~13 addresses a point inside that coset arithmetically.",
        (branches as f64).log2() as u32
    );
}
