//! Audit of the F1 production-encoder design report (2026-09-05), $0.
//!
//! Checks, against `llvq_bench::f1::rank::RankTable::build()` — the table of
//! record — the closed-form membership the report's solver relies on:
//! member ⟺ cost < C, or cost = C and ρ ≤ cut (lexicographic, ρ₀ first),
//! with (C, cut) = (88, 1103_0011) / (96, 0022_2011) / (72, 3000_1010).
//! Also recounts the trellis facts the report quotes and the parity claim
//! "class is Σ[ρ_j ∈ {1,2}] mod 2 and every pattern byte has even weight".
//!
//! `cargo run --release -p llvq-bench --example f1encaudit`

use llvq_bench::f1::rank::{cost, rank_class, unpack, RankTable, CLASS_ROWS, N0_MIXED};
use llvq_bench::f1::{Trellis, BRANCHES, GOLAY_STATES, SECTION};
use std::collections::HashSet;

#[derive(Clone, Copy)]
struct Bound {
    c: i64,
    cut: u32,
}
const BOUND_CLASS: [Bound; 2] = [Bound { c: 88, cut: 0x1103_0011 }, Bound { c: 96, cut: 0x0022_2011 }];
const BOUND_MIXED: Bound = Bound { c: 72, cut: 0x3000_1010 };

/// Big-endian nibble key: ρ₀ in the top nibble, so u32 order = lexicographic.
fn key_be(rho: &[u32; SECTION]) -> u32 {
    rho.iter().enumerate().fold(0u32, |a, (j, &r)| a | (r << (4 * (7 - j))))
}

fn member(b: Bound, rho: &[u32; SECTION]) -> bool {
    let c = cost(rho);
    c < b.c || (c == b.c && key_be(rho) <= b.cut)
}

fn main() {
    let t = RankTable::build();
    let sets: [HashSet<[u32; SECTION]>; 3] = [
        t.class_rows(0).iter().map(|&w| unpack(w)).collect(),
        t.class_rows(1).iter().map(|&w| unpack(w)).collect(),
        t.mixed_rows().iter().map(|&w| unpack(w)).collect(),
    ];
    println!("table de référence : {} lignes par classe, N0 = {} (const {})", CLASS_ROWS, t.n0_mixed, N0_MIXED);

    // C and cut read off the module's own rows (last row of each ordered list).
    for (name, rows) in [("classe 0", t.class_rows(0).to_vec()), ("classe 1", t.class_rows(1).to_vec()), ("milieu", t.mixed_rows())] {
        let last = unpack(*rows.last().unwrap());
        // For the middle, the last row in i2 order is the last class-1 row, not the lexicographic cut:
        // take the max (cost, key) instead.
        let mx = rows.iter().map(|&w| { let r = unpack(w); (cost(&r), key_be(&r)) }).max().unwrap();
        println!("  {name}: dernière ligne ρ = {:?}, max (coût, clé) = ({}, {:#010x})", last, mx.0, mx.1);
    }

    // Closed form against the sets, over every ρ ∈ {0..7}^8 (16.8 M vectors).
    let mut mism = [0usize; 3];
    let mut n_member = [0usize; 3];
    let mut class_from_cost_violations = 0usize;
    let mut rho = [0u32; SECTION];
    let total = 8usize.pow(8);
    for code in 0..total {
        let mut c = code;
        for slot in rho.iter_mut() {
            *slot = (c % 8) as u32;
            c /= 8;
        }
        let cls = rank_class(&rho);
        // Class is a function of cost mod 16 only while every rank ≤ 4; test on that domain.
        if rho.iter().all(|&r| r <= 4) && ((cost(&rho) % 16 == 8) != (cls == 0)) {
            class_from_cost_violations += 1;
        }
        let want = [
            cls == 0 && sets[0].contains(&rho),
            cls == 1 && sets[1].contains(&rho),
            sets[2].contains(&rho),
        ];
        let got = [
            cls == 0 && member(BOUND_CLASS[0], &rho),
            cls == 1 && member(BOUND_CLASS[1], &rho),
            member(BOUND_MIXED, &rho),
        ];
        for i in 0..3 {
            if want[i] != got[i] {
                mism[i] += 1;
            }
            n_member[i] += got[i] as usize;
        }
    }
    println!("forme close sur {total} vecteurs de rangs : désaccords classe 0 / classe 1 / milieu = {:?} ; membres = {:?}", mism, n_member);
    println!("classe = [coût ≡ 8 mod 16] sur {{0..4}}^8 : {class_from_cost_violations} violations");
    // Does the closed form need the class test at all for the middle? (a class-1 vector at cost 72 would break it)
    let mid_classes: HashSet<u32> = sets[2].iter().filter(|r| cost(r) == 72).map(rank_class).collect();
    println!("milieu : classes présentes sur la coquille 72 = {:?}", mid_classes);

    // Trellis facts.
    let tr = Trellis::new();
    let pre: HashSet<u8> = tr.prefixes.iter().flatten().copied().collect();
    let suf: HashSet<u8> = tr.suffixes.iter().flatten().copied().collect();
    let mids = tr.middle_bytes();
    let mut msets: Vec<Vec<u8>> = Vec::new();
    for s in 0..GOLAY_STATES {
        let mut m: Vec<u8> = tr.branches[s].iter().map(|&(b, _)| b).collect();
        m.sort_unstable();
        if !msets.contains(&m) {
            msets.push(m);
        }
    }
    let odd = pre.iter().chain(&suf).chain(&mids).filter(|b| b.count_ones() % 2 == 1).count();
    println!(
        "treillis : {} états, {} préfixes distincts, {} suffixes distincts, {} octets de milieu, {} msets, {} arêtes, {} branches/état ; octets de poids impair = {odd}",
        GOLAY_STATES, pre.len(), suf.len(), mids.len(), msets.len(), tr.edge_count(), BRANCHES
    );
    let per_p = 2 * pre.len() + 2 * mids.len() + 2 * suf.len();
    println!("résolutions distinctes (motif, parité) par p = {per_p}, par échelle = {}", 2 * per_p);
    println!("chemins de jonction par p = 2 r × 64 × 16 × 2 δ = {}", 2 * GOLAY_STATES * BRANCHES * 2);
}
