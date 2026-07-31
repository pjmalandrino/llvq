//! What each candidate **runtime** layout costs, in bits, on real blocks —
//! the measurement the passation said to take before writing any kernel.
//!
//! The archive rank is optimal in bits and unusable in a kernel (8.27 ns/block
//! on GPU, 106× the floor); a mask layout decodes at 0.11 ns/block but pays
//! for it in RAM. This bench prices that exchange **completely**: not the
//! arrangement alone (`arrbits` did that, information-theoretically), but
//! everything a runtime decoder actually reads per block — level-to-slot
//! assignment, signs, class id, gain bit, and the addressing overhead a
//! variable-width stream cannot avoid.
//!
//! ## The accounting, per block
//!
//! Everything below is a function of the block's **class** (a class is a
//! fixed multiset of |values|), so the sweep over a real artifact is exact
//! and needs no lattice decoding — a binary search per index, nothing more.
//!
//! * class id: 9 bits (383 classes + origin), gain: as many bits as the file
//!   says (1 on the published 4B). Both variants pay them.
//! * signs: one bit per **nonzero** coordinate. The archive gets signs partly
//!   for free (forced by the codeword on the odd coset); a runtime layout
//!   that drops the codeword must spell them out. Zero slots carry no sign —
//!   the decoder walks slots in order and indexes signs by a running nonzero
//!   count it maintains anyway.
//! * assignment, three candidate encodings:
//!   - **positionnel** — ⌈log₂ L⌉ bits per slot, L the block's level count.
//!     Decode is one shift+mask+table lookup per slot, no popcount chain.
//!   - **masques imbriqués** — level 0's mask over 24 slots, level 1's over
//!     what level 0 left free, … Compact, but decode chains popcounts.
//!   - **champ fixe 128** — 24 sign+3-bit-level nibbles + class + gain in a
//!     uint4. One aligned load, zero divergence, 5.33 b/w flat.
//! * addressing: a variable-width stream needs to say where block `i` starts.
//!   Charged two ways: +16 bits/block (a u16 offset each), and **grouped-32**
//!   (all 32 lanes of a SIMD group read the group's max width, one u32 base
//!   per group — padding traded against offsets).
//!
//! ## Run
//!
//! `cargo run --release -p llvq-bench --bin rtbits [path/to/model.llvq]`
//!
//! Without a path: 20 000 gaussian blocks through `nearest_angular`, the
//! same source as `classprofile`/`arrbits`. With one: every block of the
//! artifact, exhaustively.

use llvq_core::{SplitMix64, DIM};
use llvq_search::fastdec::{ClassLevels, FastDecoder, MAX_LEVELS};
use llvq_search::generic::BallSearcher;
use llvq_search::Searcher;
use std::fs::File;
use std::io::BufReader;

/// Bits for the class id: 383 classes plus the origin fit in 9.
const CLASS_BITS: u64 = 9;
/// SIMD group width the grouped layout is sized for.
const GROUP: usize = 32;

/// ⌈log₂ L⌉ — bits per slot of the positional encoding.
fn lg_ceil(l: usize) -> u64 {
    match l {
        0 | 1 => 0,
        _ => (usize::BITS - (l - 1).leading_zeros()) as u64,
    }
}

/// Nested-mask bits: canonical order, one mask per level over the slots the
/// previous levels left free; the last level is whatever remains.
fn mask_bits(lv: &ClassLevels) -> u64 {
    let mut left = DIM as u64;
    let mut bits = 0u64;
    for i in 0..lv.len.saturating_sub(1) {
        bits += left;
        left -= lv.counts[i] as u64;
    }
    bits
}

/// Per-block widths of the two variable candidates, everything included
/// except addressing.
#[derive(Clone, Copy)]
struct Widths {
    mask: u64,
    pos: u64,
    levels: usize,
}

fn widths(lv: &ClassLevels, gain_bits: u64) -> Widths {
    let common = lv.nonzero as u64 + CLASS_BITS + gain_bits;
    Widths {
        mask: mask_bits(lv) + common,
        pos: lg_ceil(lv.len) * DIM as u64 + common,
        levels: lv.len,
    }
}

#[derive(Default)]
struct Acc {
    blocks: u64,
    mask: u64,
    pos: u64,
    levels: [u64; MAX_LEVELS + 1],
    worst_mask: u64,
    worst_pos: u64,
}

impl Acc {
    fn push(&mut self, w: &Widths) {
        self.blocks += 1;
        self.mask += w.mask;
        self.pos += w.pos;
        self.levels[w.levels] += 1;
        self.worst_mask = self.worst_mask.max(w.mask);
        self.worst_pos = self.worst_pos.max(w.pos);
    }
}

/// Tracks the grouped-32 layout: every lane of a group reads the group's max
/// width, one u32 base pointer per group.
#[derive(Default)]
struct Grouped {
    filled: usize,
    max_mask: u64,
    max_pos: u64,
    max_levels: usize,
    sum_levels: u64,
    /// Totals over closed groups.
    mask_bits: u64,
    pos_bits: u64,
    /// Same stride, rounded up to a whole byte — what lanes can actually
    /// address.
    mask_bits_byte: u64,
    max_levels_sum: u64,
    groups: u64,
}

impl Grouped {
    fn push(&mut self, w: &Widths) {
        self.filled += 1;
        self.max_mask = self.max_mask.max(w.mask);
        self.max_pos = self.max_pos.max(w.pos);
        self.max_levels = self.max_levels.max(w.levels);
        self.sum_levels += w.levels as u64;
        if self.filled == GROUP {
            self.close();
        }
    }

    fn close(&mut self) {
        if self.filled == 0 {
            return;
        }
        // A partial trailing group still pays full-width lanes.
        self.mask_bits += GROUP as u64 * self.max_mask + 32;
        self.pos_bits += GROUP as u64 * self.max_pos + 32;
        self.mask_bits_byte += GROUP as u64 * self.max_mask.div_ceil(8) * 8 + 32;
        self.max_levels_sum += self.max_levels as u64;
        self.groups += 1;
        self.filled = 0;
        self.max_mask = 0;
        self.max_pos = 0;
        self.max_levels = 0;
    }
}

/// Levels of a raw lattice point, canonical order — for the gaussian mode,
/// where blocks come from the searcher rather than from an artifact.
fn levels_of_point(p: &[i32; DIM]) -> ClassLevels {
    let mut kinds: Vec<(i32, u8)> = Vec::new();
    for &v in p {
        let a = v.abs();
        match kinds.iter_mut().find(|(u, _)| *u == a) {
            Some((_, n)) => *n += 1,
            None => kinds.push((a, 1)),
        }
    }
    kinds.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(b.0.cmp(&a.0)));
    let mut lv = ClassLevels {
        len: kinds.len(),
        odd: p.iter().all(|v| v % 2 != 0),
        shell: 0, // unused here
        ..Default::default()
    };
    for (i, &(v, n)) in kinds.iter().enumerate() {
        lv.values[i] = v;
        lv.counts[i] = n;
        if v != 0 {
            lv.nonzero += n;
        }
    }
    lv
}

fn main() {
    let path = std::env::args().nth(1);
    let fd = FastDecoder::new();

    let mut odd = Acc::default();
    let mut even = Acc::default();
    let mut grouped = Grouped::default();
    let mut origin_blocks = 0u64;
    let mut gain_bits = 1u64;
    // Archive rate anchor: index + gain bits per block, from the file when
    // there is one (the published 4B is cap 12: 47 + 1).
    let mut arch_bits = 49u64;
    let source: String;

    // Per-class cross-check bookkeeping: the first block seen of each class
    // is decoded and its point-derived levels compared to the class table.
    let mut verified = vec![false; fd.n_classes()];
    let mut verified_classes = 0usize;

    if let Some(path) = path {
        // ---- every block of a real artifact ----
        let f = File::open(&path).unwrap_or_else(|e| panic!("open {path}: {e}"));
        let mut r = BufReader::new(f);
        let h = llvq_artifact::read_header(&mut r).expect("valid artifact header");
        source = format!("{path} — {} matrices", h.matrices);
        for _ in 0..h.matrices {
            let m = llvq_artifact::read_matrix_raw(&mut r).expect("valid matrix");
            // The class layout runs shells ascending, so the cap-c ball is a
            // strict prefix of the cap-13 layout: a table built for 13
            // decodes any cap ≤ 13 file identically.
            assert!(
                m.shell_cap <= 13,
                "{}: shell cap {} exceeds the class table's 13",
                m.name, m.shell_cap
            );
            gain_bits =
                m.centroids.len().next_power_of_two().trailing_zeros() as u64;
            arch_bits =
                llvq_quant::quantizer::index_bits(m.shell_cap) as u64 + gain_bits;
            for &idx in &m.indices {
                let Some(ci) = fd.class_of(idx) else {
                    assert_eq!(idx, 0, "{}: index {idx} outside the ball", m.name);
                    origin_blocks += 1;
                    // The origin still occupies a lane: one level, no signs,
                    // no masks — only the class id and the gain bit.
                    let w = Widths {
                        mask: CLASS_BITS + gain_bits,
                        pos: CLASS_BITS + gain_bits,
                        levels: 1,
                    };
                    even.push(&w);
                    grouped.push(&w);
                    continue;
                };
                let lv = fd.levels(ci);
                if !verified[ci] {
                    let p = fd.decode(idx).expect("valid index");
                    assert_eq!(
                        levels_of_point(&p),
                        ClassLevels { shell: 0, ..*lv },
                        "class {ci}: table disagrees with a decoded point"
                    );
                    verified[ci] = true;
                    verified_classes += 1;
                }
                let w = widths(lv, gain_bits);
                if lv.odd { &mut odd } else { &mut even }.push(&w);
                grouped.push(&w);
            }
        }
    } else {
        // ---- gaussian source, same harness as classprofile/arrbits ----
        const N: usize = 20_000;
        source = format!("{N} blocs gaussiens, nearest_angular");
        let mut rng = SplitMix64::new(0x6_A88B);
        let s = Searcher::new();
        let mut ball = BallSearcher::new();
        for _ in 0..N {
            let x: [f64; DIM] = core::array::from_fn(|_| rng.next_gaussian());
            let f = ball.nearest_angular(&s, &x);
            let lv = levels_of_point(&f.point);
            let w = widths(&lv, gain_bits);
            if lv.odd { &mut odd } else { &mut even }.push(&w);
            grouped.push(&w);
        }
    }
    grouped.close();

    let total = odd.blocks + even.blocks;
    let pct = |n: u64| 100.0 * n as f64 / total as f64;
    let bpw = |bits: u64, blocks: u64| bits as f64 / (blocks * DIM as u64) as f64;

    println!("source : {source}");
    println!("{total} blocs — impair {:.1} %, pair {:.1} %", pct(odd.blocks), pct(even.blocks));
    if origin_blocks > 0 {
        println!("dont {origin_blocks} blocs origine (tout zéro)");
    }
    if verified_classes > 0 {
        println!("{verified_classes} classes recoupées point décodé ↔ table");
    }

    println!("\n  niveaux de magnitude par bloc");
    println!("  {}", "-".repeat(56));
    for l in 1..=MAX_LEVELS {
        let n = odd.levels[l] + even.levels[l];
        if n > 0 {
            println!("  {l} niveaux{:>16} blocs{:>8.2} %", n, pct(n));
        }
    }

    // ---- the table the decision is made from ----
    let all_mask = odd.mask + even.mask;
    let all_pos = odd.pos + even.pos;
    let worst_mask = odd.worst_mask.max(even.worst_mask);
    let worst_pos = odd.worst_pos.max(even.worst_pos);
    // Pad the worst case to the next u32 so every lane load is aligned.
    let fixed_mask = worst_mask.div_ceil(32) * 32;
    let fixed_pos = worst_pos.div_ceil(32) * 32;

    println!("\n  bits/poids en RAM, comptabilité complète");
    println!("  (assignation + signes + classe {CLASS_BITS} b + gain {gain_bits} b + adressage)");
    println!("  {}", "-".repeat(72));
    println!(
        "  {:<44}{:>10}{:>10}",
        "layout", "masques", "positionnel"
    );
    println!(
        "  {:<44}{:>10.4}{:>10.4}",
        "variable, adressage gratuit (borne inf.)",
        bpw(all_mask, total),
        bpw(all_pos, total)
    );
    println!(
        "  {:<44}{:>10.4}{:>10.4}",
        "variable + offset u16 par bloc",
        bpw(all_mask + 16 * total, total),
        bpw(all_pos + 16 * total, total)
    );
    println!(
        "  {:<44}{:>10.4}{:>10.4}",
        "groupé 32 (stride = max du groupe + base u32)",
        bpw(grouped.mask_bits, total),
        bpw(grouped.pos_bits, total)
    );
    println!(
        "  {:<44}{:>10.4}{:>10}",
        "groupé 32, stride arrondi à l'octet",
        bpw(grouped.mask_bits_byte, total),
        "—"
    );
    println!(
        "  {:<44}{:>10.4}{:>10.4}",
        format!("champ fixe au pire cas ({worst_mask}/{worst_pos} b → u32)"),
        bpw(fixed_mask * total, total),
        bpw(fixed_pos * total, total)
    );
    println!(
        "  {:<44}{:>10}{:>10.4}",
        "champ fixe 128 (uint4, nibbles)", "—",
        128.0 / DIM as f64
    );

    println!("\n  par coset (variable, adressage gratuit)");
    println!("  {}", "-".repeat(72));
    for (name, a) in [("impair", &odd), ("pair", &even)] {
        if a.blocks > 0 {
            println!(
                "  {:<10}{:>9.1} %   masques {:.4}   positionnel {:.4}   pire {}/{} b",
                name,
                pct(a.blocks),
                bpw(a.mask, a.blocks),
                bpw(a.pos, a.blocks),
                a.worst_mask,
                a.worst_pos
            );
        }
    }

    // ---- divergence proxy ----
    if grouped.groups > 0 {
        let mean_l = grouped.sum_levels as f64 / total as f64;
        let mean_max_l = grouped.max_levels_sum as f64 / grouped.groups as f64;
        println!("\n  divergence sur groupes de {GROUP} lanes consécutives");
        println!("  {}", "-".repeat(72));
        println!(
            "  niveaux : moyenne {mean_l:.3}, moyenne des max par groupe {mean_max_l:.3} \
             (×{:.3})",
            mean_max_l / mean_l
        );
        println!(
            "  padding groupé 32 : masques ×{:.3}, positionnel ×{:.3} vs adressage gratuit",
            grouped.mask_bits as f64 / all_mask as f64,
            grouped.pos_bits as f64 / all_pos as f64
        );
    }

    // ---- what each rate means for a 4B token ----
    let linear_w = 3_633_315_840f64;
    let lm_head_gb = 0.778; // tied embedding read once per token for logits
    println!("\n  trafic par token, Qwen3-4B (linéaires + lm_head f16 {lm_head_gb} Go)");
    println!("  {}", "-".repeat(72));
    let show = |name: &str, b: f64| {
        let gb = linear_w * b / 8.0 / 1e9 + lm_head_gb;
        println!(
            "  {name:<44}{b:>7.3} b/p{:>9.2} Go{:>9.0} tok/s",
            gb,
            400.0 / gb
        );
    };
    show(
        &format!("archive v1, {arch_bits} b/bloc (indécodable en fusé)"),
        arch_bits as f64 / 24.0,
    );
    show("variable masques + u16", bpw(all_mask + 16 * total, total));
    show("variable positionnel + u16", bpw(all_pos + 16 * total, total));
    show("groupé 32, masques", bpw(grouped.mask_bits, total));
    show("groupé 32, masques, stride octet", bpw(grouped.mask_bits_byte, total));
    show("groupé 32, positionnel", bpw(grouped.pos_bits, total));
    show("champ fixe au pire cas (masques, u32)", bpw(fixed_mask * total, total));
    show("champ fixe au pire cas (pos., u32)", bpw(fixed_pos * total, total));
    show("champ fixe 128", 128.0 / 24.0);
    println!("\n  plafond FP16 : ~50 tok/s ; plafond lm_head seul : ~514 tok/s");
}
