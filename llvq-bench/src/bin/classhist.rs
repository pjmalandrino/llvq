//! Per-class histogram of a sealed artifact — experiment #0 of
//! `docs/archive/pistes-format-vram-2026-08-05.md`.
//!
//! `classprofile` profiles a gaussian source; `rtbits` prices layouts and
//! histograms blocks by **level count**. What neither produces is the
//! distribution of real blocks over the **classes themselves** — the one
//! number the E2 (Golay/GF(2) stage) bracket depends on: E2 requires at most
//! 2 distinct |values| per residue mod 4 (one bit picks within a residue once
//! codeword membership gives the residue), and some classes violate that.
//! Whether the violating classes are populated decides between the 2.92–3.05
//! b/w bracket (rare) and 3.2–3.4 (~10 %).
//!
//! Zero counts as a residue-0 value: the runtime layout spends a level on it,
//! and the within-residue discriminator must cover it like any other value.
//!
//! This is a separate bin rather than a section of `rtbits` because rtbits'
//! output is cited verbatim by the dossier and the full per-class table
//! (301 lines on the m ≤ 12 ball) would drown its decision table.
//!
//! Run: `cargo run --release -p llvq-bench --bin classhist ~/llvq-q4b.llvq`

use llvq_search::fastdec::{ClassLevels, FastDecoder, MAX_LEVELS};
use std::fs::File;
use std::io::BufReader;

/// Distinct |values| per residue mod 4, zero included. Class values are
/// distinct by construction, so counting slots is counting distinct values.
fn residue_profile(lv: &ClassLevels) -> [u32; 4] {
    let mut r = [0u32; 4];
    for i in 0..lv.len {
        r[(lv.values[i] % 4) as usize] += 1;
    }
    r
}

/// E2 violation: some residue would need more than one discriminator bit.
fn violates(r: &[u32; 4]) -> bool {
    r.iter().any(|&n| n > 2)
}

/// `v^n` pairs in the canonical (count-descending) level order.
fn fmt_values(lv: &ClassLevels) -> String {
    (0..lv.len)
        .map(|i| format!("{}^{}", lv.values[i], lv.counts[i]))
        .collect::<Vec<_>>()
        .join(" ")
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: classhist <path/to/model.llvq>");
    let fd = FastDecoder::new();

    let mut counts = vec![0u64; fd.n_classes()];
    let mut origin_blocks = 0u64;
    let mut shell_cap = 0u32;
    let mut matrices = 0u32;

    let f = File::open(&path).unwrap_or_else(|e| panic!("open {path}: {e}"));
    let mut r = BufReader::new(f);
    let h = llvq_artifact::read_header(&mut r).expect("valid artifact header");
    for _ in 0..h.matrices {
        let m = llvq_artifact::read_matrix_raw(&mut r).expect("valid matrix");
        assert!(
            m.shell_cap <= 13,
            "{}: shell cap {} exceeds the class table's 13",
            m.name,
            m.shell_cap
        );
        shell_cap = shell_cap.max(m.shell_cap);
        matrices += 1;
        for &idx in &m.indices {
            match fd.class_of(idx) {
                Some(ci) => counts[ci] += 1,
                None => {
                    assert_eq!(idx, 0, "{}: index {idx} outside the ball", m.name);
                    origin_blocks += 1;
                }
            }
        }
    }

    let total: u64 = counts.iter().sum::<u64>() + origin_blocks;
    let observed = counts.iter().filter(|&&n| n > 0).count();
    // Every populated class must sit inside the file's cap: the class layout
    // runs shells ascending, so a cap-c file can only reach shell ≤ c.
    for (ci, &n) in counts.iter().enumerate() {
        assert!(
            n == 0 || fd.levels(ci).shell <= shell_cap,
            "class {ci} beyond the file's shell cap {shell_cap}"
        );
    }

    let pct = |n: u64| 100.0 * n as f64 / total as f64;
    println!("source : {path} — {matrices} matrices, cap {shell_cap}");
    println!(
        "{total} blocs, {origin_blocks} origine, {observed} classes observées"
    );

    // ---- 1. histogram by class, descending population ----
    let mut order: Vec<usize> = (0..fd.n_classes()).filter(|&ci| counts[ci] > 0).collect();
    order.sort_by_key(|&ci| std::cmp::Reverse(counts[ci]));

    println!("\n  top 20 classes par population (valeurs v^count, ordre canonique)");
    println!("  {}", "-".repeat(76));
    let mut cum = 0u64;
    for (rank, &ci) in order.iter().take(20).enumerate() {
        let lv = fd.levels(ci);
        cum += counts[ci];
        println!(
            "  {:>2}. c{:<3} {} m={:<2} L={} {:<26}{:>12} blocs{:>8.3} %  cum {:>6.2} %",
            rank + 1,
            ci,
            if lv.odd { "imp." } else { "pair" },
            lv.shell,
            lv.len,
            fmt_values(lv),
            counts[ci],
            pct(counts[ci]),
            pct(cum)
        );
    }
    println!(
        "  … {} autres classes, {} blocs ({:.3} %)",
        order.len().saturating_sub(20),
        total - origin_blocks - cum,
        pct(total - origin_blocks - cum)
    );

    // ---- 2. entropy of the class distribution, and of the whole index ----
    //
    // H_class = −Σ p·log₂ p over the observed symbols (origin included when
    // present). Modelling the arrangement as uniform within its class — which
    // it is by construction of the rank — the index entropy per block is
    // H_index = H_class + Σ_c p_c·log₂|c|, with |c| the class cardinality
    // (`class_range` span; the origin contributes log₂ 1 = 0).
    let mut h_class = 0f64;
    let mut h_arr = 0f64;
    for &ci in &order {
        let p = counts[ci] as f64 / total as f64;
        h_class -= p * p.log2();
        let (lo, hi) = fd.class_range(ci);
        h_arr += p * ((hi - lo + 1) as f64).log2();
    }
    if origin_blocks > 0 {
        let p = origin_blocks as f64 / total as f64;
        h_class -= p * p.log2();
    }
    let paid = llvq_quant::quantizer::index_bits(shell_cap) as f64;
    println!("\n  entropie de l'index (arrangement uniforme dans la classe)");
    println!("  {}", "-".repeat(76));
    println!("  H(classe)                : {h_class:>8.4} bits/bloc");
    println!("  Σ p·log₂|classe|         : {h_arr:>8.4} bits/bloc");
    println!(
        "  H(index) = somme         : {:>8.4} bits/bloc — payés {paid:.0} ({:.4} de marge)",
        h_class + h_arr,
        paid - h_class - h_arr
    );

    // ---- 3. every class of the m ≤ cap ball: residues mod 4 and violations ----
    println!("\n  les classes de la boule m ≤ {shell_cap} — résidus mod 4 (zéro compté)");
    println!("  {}", "-".repeat(76));
    let ball: Vec<usize> = (0..fd.n_classes())
        .filter(|&ci| fd.levels(ci).shell <= shell_cap)
        .collect();
    let mut viol_classes = [0u64; 2]; // [even, odd]
    let mut viol_blocks = 0u64;
    let mut viol_blocks_le4 = 0u64;
    let mut blocks_le4 = origin_blocks; // the origin is a 1-level block
    let mut level_hist = [0u64; MAX_LEVELS + 1];
    level_hist[1] += origin_blocks;
    let mut max_in_residue = 0u32;
    let mut both_residues_ge3 = 0u64;
    for &ci in &ball {
        let lv = fd.levels(ci);
        let r = residue_profile(lv);
        let v = violates(&r);
        let res: Vec<String> = (0..4)
            .filter(|&i| r[i] > 0)
            .map(|i| format!("r{}={}", i, r[i]))
            .collect();
        println!(
            "  c{:<3} {} m={:<2} L={} {:<26} {:<12}{}{:>14} blocs",
            ci,
            if lv.odd { "imp." } else { "pair" },
            lv.shell,
            lv.len,
            fmt_values(lv),
            res.join(" "),
            if v { "VIOLATION" } else { "         " },
            counts[ci]
        );
        max_in_residue = max_in_residue.max(*r.iter().max().unwrap());
        if r.iter().filter(|&&n| n >= 3).count() >= 2 {
            both_residues_ge3 += 1;
        }
        level_hist[lv.len] += counts[ci];
        if lv.len <= 4 {
            blocks_le4 += counts[ci];
        }
        if v {
            viol_classes[usize::from(lv.odd)] += 1;
            viol_blocks += counts[ci];
            if lv.len <= 4 {
                viol_blocks_le4 += counts[ci];
            }
        }
    }

    // ---- 4. the totals the E2 bracket is decided from ----
    println!("\n  totaux");
    println!("  {}", "-".repeat(76));
    println!(
        "  boule m ≤ {shell_cap} : {} classes ({} paires + {} impaires)",
        ball.len(),
        ball.iter().filter(|&&ci| !fd.levels(ci).odd).count(),
        ball.iter().filter(|&&ci| fd.levels(ci).odd).count()
    );
    println!(
        "  classes violantes (>2 valeurs dans un résidu) : {} ({} paires + {} impaires)",
        viol_classes[0] + viol_classes[1],
        viol_classes[0],
        viol_classes[1]
    );
    println!(
        "  max de valeurs distinctes dans un résidu : {max_in_residue} ; \
         classes à ≥3 dans les deux résidus : {both_residues_ge3}"
    );
    println!("\n  niveaux par bloc (recoupement rtbits)");
    for (l, &n) in level_hist.iter().enumerate().skip(1) {
        if n > 0 {
            println!("    L = {l}{:>16} blocs{:>9.4} %", n, pct(n));
        }
    }
    println!(
        "\n  BLOCS dans des classes violantes : {viol_blocks} / {total} = {:.4} %",
        pct(viol_blocks)
    );
    println!(
        "  parmi les blocs L ≤ 4 seulement  : {viol_blocks_le4} / {blocks_le4} = {:.4} %",
        100.0 * viol_blocks_le4 as f64 / blocks_le4 as f64
    );

    // ---- 5. E2 exceptions — the rule the Golay/GF(2) stage actually needs ----
    //
    // Section 3's VIOLATION flag counts >2 values per residue for *every*
    // class. The E2 design only escapes when the 2-bit in-class level code
    // cannot work:
    //   * even class with >2 distinct |values| in some residue mod 4 (zero
    //     counted in r0): the residue is fixed by codeword membership, so one
    //     discriminator bit must pick among ≤2 values — 3 breaks it;
    //   * any 5-level class, either coset: the level itself no longer fits
    //     2 direct bits.
    // An odd class at L ≤ 4 is NOT an exception even with 3 values in a
    // residue: its level is coded in 2 direct bits, residues play no role.
    let mut exc_even_viol_le4 = 0u64; // even, L ≤ 4, residue violation
    let mut exc_l5_even = 0u64;
    let mut exc_l5_odd = 0u64;
    let mut exc_classes = 0u64;
    let mut l5_even_also_viol = 0u64; // informational overlap
    for &ci in &ball {
        let lv = fd.levels(ci);
        let v = violates(&residue_profile(lv));
        let exc = (lv.len == 5) || (!lv.odd && v);
        if !exc {
            continue;
        }
        if counts[ci] > 0 {
            exc_classes += 1;
        }
        if lv.len == 5 {
            if lv.odd {
                exc_l5_odd += counts[ci];
            } else {
                exc_l5_even += counts[ci];
                if v {
                    l5_even_also_viol += counts[ci];
                }
            }
        } else {
            exc_even_viol_le4 += counts[ci];
        }
    }
    let exc_blocks = exc_even_viol_le4 + exc_l5_even + exc_l5_odd;
    // Cross-checks against this run's own tallies: the three buckets are
    // disjoint by construction, L=5 buckets must tile the L=5 histogram bar,
    // and the even violating-L≤4 bucket is a subset of section 3's
    // both-coset violating-L≤4 count.
    assert_eq!(
        exc_l5_even + exc_l5_odd,
        level_hist[5],
        "L=5 buckets must tile the L=5 histogram bar"
    );
    assert!(
        exc_even_viol_le4 <= viol_blocks_le4,
        "even violating L≤4 blocks exceed the both-coset count"
    );

    let frac = exc_blocks as f64 / total as f64;
    println!("\n  exceptions E2 (pair violant, ou L = 5 quel que soit le coset)");
    println!("  {}", "-".repeat(76));
    println!(
        "  pairs violants (L ≤ 4)   : {exc_even_viol_le4:>12} blocs{:>9.4} %",
        pct(exc_even_viol_le4)
    );
    println!(
        "  L = 5 pairs              : {exc_l5_even:>12} blocs{:>9.4} %  (dont violants : {l5_even_also_viol})",
        pct(exc_l5_even)
    );
    println!(
        "  L = 5 impairs            : {exc_l5_odd:>12} blocs{:>9.4} %",
        pct(exc_l5_odd)
    );
    println!(
        "  TOTAL exceptions         : {exc_blocks:>12} blocs{:>9.4} %  ({exc_classes} classes peuplées)",
        pct(exc_blocks)
    );
    println!(
        "  payload E2 = 3,0 + 6·fraction = {:.4} b/poids  (144 bits/exception ; Planes12x : 4,342)",
        3.0 + 6.0 * frac
    );
}
