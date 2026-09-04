//! What the F1 decoder would actually have to hold in the kernel.
//!
//! `cargo run --release -p llvq-bench --example f1table`
//!
//! The number this repository published on the morning of 2026-09-04 — 8.0 KiB,
//! "green with a factor of two of margin" — was withdrawn the same evening: it
//! divided 1,024 Golay edges by the 256 Λ₂₄ states instead of the 64 Golay
//! ones, and its "8 coordinates at one byte each" entry was a guess rather than
//! a derivation. Nothing was put in its place, because a wrong figure is not
//! replaced by an unverified one. This computes the real thing, for the split
//! the measurement actually uses.
//!
//! ## The question, precisely
//!
//! Decoding is `label → point`. A direct table is one entry per label per
//! region, and there are 256 + 16 + 256 = 528 regions — 20 MiB, dead on
//! arrival. What shrinks it is symmetry: two regions that differ by an isometry
//! the kernel can apply for free share one table. The only such isometry here
//! is a coordinate sign flip — a permutation would need a dynamically indexed
//! shared read, which `llvq_planes.cuh` is written to avoid.
//!
//! So the table is `(orbits under sign flips) × (entries) × (bytes per entry)`,
//! and the orbit count is what this measures. An adversarial review reported 67
//! orbits on the end-section cosets and 9 on the middle; that claim is checked
//! here from the Rust side, which shares no code with it.
//!
//! The budget is the card's, measured at preflight and recorded in
//! `docs/format-noyau.md` §8: 49,152 B per block by default, 101,376 B opt-in,
//! 102,400 B per SM. The fused matvec already stages a 12 KiB activation tile.

use llvq_bench::f1::{SectionSet, Trellis, GOLAY_STATES, SECTION};
use std::collections::HashMap;

/// Points of norm² ≤ `T`, sorted — enough to identify a coset up to sign.
const T_SIG: usize = 64;

fn signature(set: &SectionSet) -> Vec<[i32; SECTION]> {
    let mut v = set.enumerate_below(T_SIG);
    v.sort_unstable();
    v
}

/// Smallest image of `sig` under the 256 coordinate sign flips.
fn canonical(sig: &[[i32; SECTION]]) -> Vec<[i32; SECTION]> {
    (0u32..256)
        .map(|eps| {
            let mut img: Vec<[i32; SECTION]> = sig
                .iter()
                .map(|y| {
                    let mut z = *y;
                    for (j, v) in z.iter_mut().enumerate() {
                        if eps >> j & 1 == 1 {
                            *v = -*v;
                        }
                    }
                    z
                })
                .collect();
            img.sort_unstable();
            img
        })
        .min()
        .expect("256 flips")
}

/// Orbit count without the printout, for the section-1 vs section-3 comparison.
fn orbits_quiet(sets: &[SectionSet]) -> usize {
    let mut seen = std::collections::HashSet::new();
    for s in sets {
        let sig = signature(s);
        if !sig.is_empty() {
            seen.insert(canonical(&sig));
        }
    }
    seen.len()
}

fn orbits(sets: Vec<SectionSet>, label: &str) -> usize {
    let mut seen: HashMap<Vec<[i32; SECTION]>, usize> = HashMap::new();
    let mut empty = 0;
    for s in &sets {
        let sig = signature(s);
        if sig.is_empty() {
            empty += 1;
            continue;
        }
        *seen.entry(canonical(&sig)).or_insert(0) += 1;
    }
    let mut sizes: Vec<usize> = seen.values().copied().collect();
    sizes.sort_unstable();
    println!(
        "{label:<28} {} régions → {} orbites sous les 256 changements de signe{}",
        sets.len(),
        seen.len(),
        if empty > 0 { format!(" ({empty} signatures vides)") } else { String::new() }
    );
    seen.len()
}

/// Orbit sizes, largest first, and how much of the table the hottest few hold.
fn concentration(sets: &[SectionSet], label: &str, entries: usize) {
    let mut seen: HashMap<Vec<[i32; SECTION]>, usize> = HashMap::new();
    for s in sets {
        let sig = signature(s);
        if !sig.is_empty() {
            *seen.entry(canonical(&sig)).or_insert(0) += 1;
        }
    }
    let mut sizes: Vec<usize> = seen.values().copied().collect();
    sizes.sort_unstable_by(|a, b| b.cmp(a));
    let total: usize = sizes.iter().sum();
    let per_orbit = entries * 6; // six bytes packed per entry
    println!("\n{label} — {} orbites, {total} régions", sizes.len());
    println!("  tailles, de la plus grosse : {:?}...", &sizes[..sizes.len().min(8)]);
    let mut cum = 0usize;
    for k in [1usize, 2, 4, 8, 16] {
        if k > sizes.len() {
            break;
        }
        cum = sizes[..k].iter().sum();
        println!(
            "  les {k:>2} plus consultées : {:>5.1} % des consultations, {:>7.1} Kio de table",
            100.0 * cum as f64 / total as f64,
            (k * per_orbit) as f64 / 1024.0
        );
    }
    let _ = cum;
}

fn main() {
    let t = Trellis::new();
    let w = [12u32, 15, 12];

    // ⚠️ Sections 1 and 3 are counted SEPARATELY and then together. Section 1
    // uses a state's prefix pair, section 3 its suffix pair, and those are not
    // the same bytes. If the two families share their orbits, one table serves
    // both ends; if they do not, the end table is twice what was published on
    // the evening of 2026-09-04. The question was raised by the operator and it
    // moves the central figure, so it is counted rather than assumed.
    let mut sec1 = Vec::new();
    let mut sec3 = Vec::new();
    for p in 0..2u32 {
        for r in 0..2u32 {
            for s in 0..GOLAY_STATES {
                sec1.push(t.section1(s, p, r));
                sec3.push(SectionSet {
                    patterns: t.suffixes[s].to_vec(),
                    p,
                    k_parity: Some(r),
                });
            }
        }
    }
    let o1 = orbits_quiet(&sec1);
    let o3 = orbits_quiet(&sec3);
    let mut both = sec1;
    both.extend(sec3);
    let o_both = orbits_quiet(&both);
    println!(
        "section 1 seule : {o1} orbites | section 3 seule : {o3} | les deux ensemble : {o_both}"
    );
    println!(
        "  {} — le nombre de tables aux extrémités est {o_both}, pas {o1}\n",
        if o_both == o1.max(o3) {
            "les deux familles PARTAGENT leurs orbites"
        } else {
            "les deux familles ont des orbites DISTINCTES"
        }
    );
    let ends = both;
    let mut mids = Vec::new();
    let mut seen_msets: Vec<Vec<u8>> = Vec::new();
    for s8 in 0..GOLAY_STATES {
        let mut m: Vec<u8> = t.branches[s8].iter().map(|&(b, _)| b).collect();
        m.sort_unstable();
        if !seen_msets.contains(&m) {
            seen_msets.push(m);
        }
    }
    for p in 0..2u32 {
        for m in &seen_msets {
            mids.push(SectionSet { patterns: m.clone(), p, k_parity: None });
        }
    }

    println!("découpe {}/{}/{}\n", w[0], w[1], w[2]);
    let o_end = orbits(ends.clone(), "sections extrêmes");
    let o_mid = orbits(mids.clone(), "section du milieu");

    // How concentrated is the access? An orbit that many regions map to is
    // consulted proportionally more often, so the orbit-size distribution says
    // whether a hot/cold split of the table would buy anything — the operator's
    // question, and the design that might make F1 fit at all.
    concentration(&ends, "sections extrêmes", 1 << w[0]);
    concentration(&mids, "section du milieu", 1 << w[1]);

    // An entry holds the eight coordinates. Their range decides the width.
    let mut max_abs = 0i32;
    for p in 0..2u32 {
        for r in 0..2u32 {
            let set = t.section1(0, p, r);
            for y in set.region_points(w[0]) {
                max_abs = max_abs.max(y.iter().map(|v| v.abs()).max().unwrap_or(0));
            }
        }
    }
    let bits = (2 * max_abs as u32 + 1).next_power_of_two().trailing_zeros().max(1) + 1;
    println!("\ncoordonnée maximale en valeur absolue : {max_abs} → {bits} bits signés par coordonnée");

    let tile = 12 * 1024usize;
    for (name, per_entry) in [("un octet la coordonnée", 8usize), ("empaqueté", (8 * bits as usize).div_ceil(8))] {
        let end_bytes = o_end * (1usize << w[0]) * per_entry;
        let mid_bytes = o_mid * (1usize << w[1]) * per_entry;
        let total = end_bytes + mid_bytes;
        println!(
            "\n{name} ({per_entry} o/entrée)\n  \
             extrêmes {:.0} Kio + milieu {:.0} Kio = {:.0} Kio\n  \
             + tuile d'activation 12 Kio = {:.0} Kio par bloc",
            end_bytes as f64 / 1024.0,
            mid_bytes as f64 / 1024.0,
            total as f64 / 1024.0,
            (total + tile) as f64 / 1024.0
        );
        for (lim_name, lim) in [("défaut 48 Kio", 49_152usize), ("opt-in 99 Kio", 101_376)] {
            println!(
                "    {lim_name} : {}{}",
                if total + tile <= lim { "tient" } else { "NE TIENT PAS" },
                if total + tile <= lim {
                    String::new()
                } else {
                    format!(" (×{:.0} de trop)", (total + tile) as f64 / lim as f64)
                }
            );
        }
    }

    // The table does not fit in shared memory, so it lives in L2 — 48 MB on
    // L40S, so capacity is not the question. Traffic is.
    const WEIGHTS: u64 = 3_633_315_840; // projection weights of the 4B, bin/seal
    let blocks = WEIGHTS / 24;
    let lookups = blocks * 3;
    let sector = 32u64; // one cache-line sector per lookup, the optimistic floor
    let f1_bytes = WEIGHTS as f64 * 2.159 / 8.0;
    println!(
        "\ntrafic par passe modèle (4B, {WEIGHTS} poids de projection)\n           {blocks} blocs → {lookups} consultations de table, 3 par bloc\n           table : {:.1} Go à un secteur de {sector} o par consultation\n           poids : {:.2} Go\n           rapport {:.0} pour 1",
        lookups as f64 * sector as f64 / 1e9,
        f1_bytes / 1e9,
        lookups as f64 * sector as f64 / f1_bytes
    );
    println!(
        "\nÀ comparer à `Planes14`, qui sert aujourd'hui : 2,18 Go de DRAM au banc\n         252 projections, et une table constante de 12 Kio — résidente en L1.\n         F1 lit 0,45× les octets de DRAM et demande une table 275× plus grosse."
    );
}
