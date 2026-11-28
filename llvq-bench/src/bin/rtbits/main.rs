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
//!   - **positional** — ⌈log₂ L⌉ bits per slot, L the block's level count.
//!     Decode is one shift+mask+table lookup per slot, no popcount chain.
//!   - **nested masks** — level 0's mask over 24 slots, level 1's over
//!     what level 0 left free, … Compact, but decode chains popcounts.
//!   - **fixed field 128** — 24 sign+3-bit-level nibbles + class + gain in a
//!     uint4. One aligned load, zero divergence, 5.33 b/w flat.
//!   - **Slot32** — the layout the fused kernel actually reads: one 24-bit
//!     mask per level *in slot space*, level 0 implicit, signs indexed by
//!     slot rather than by nonzero rank. Costs `9 + gain + 24·L` bits, so it
//!     spends `24 − nonzero` bits more than the flat layout to buy a fixed
//!     field offset. Nothing priced it here before; the two figures the
//!     dossier publishes for it come from two other benches.
//!   - **Planes14** — today's production layout ([`PlanesBlocks`]): the same
//!     decoded content as `Slot32`, re-arranged as *bit-planes*. The level of
//!     a slot stops being a one-hot over `L` masks and becomes an explicit
//!     3-bit index spread over three 24-bit planes, so the record no longer
//!     grows with `L`: `9 + 1 + 24 + 3·24 = 106` bits of payload in a frozen
//!     [`PLANES14_BYTES`]-byte stride, padding included. Flat, class-blind,
//!     **and with no base table at all** — the addressing line below simply
//!     does not apply to it.
//!   - **Planes12x** — [`Planes12xBlocks`], `Planes14` minus its third plane:
//!     two planes address levels `0..4`, so the main record is
//!     `9 + 1 + 24 + 2·24 = 82` bits in a frozen [`PLANES12X_BYTES`]-byte
//!     stride. A block whose class carries five magnitudes cannot be said in
//!     two planes; the main stream takes its best `L ≤ 4` neighbour and the
//!     block is repeated **exactly** in an exception table, at
//!     [`PLANES12X_EXC_BYTES`] bytes each (a `u32` block index plus a full
//!     `Planes14` record). Cost is therefore `96·n + 144·n_exc` bits, and
//!     `n_exc` is a *count of five-level classes* — which this bench knows
//!     per block without decoding anything.
//! * addressing: a variable-width stream needs to say where block `i` starts.
//!   Charged two ways: +16 bits/block (a u16 offset each), and **grouped-32**
//!   (all 32 lanes of a SIMD group read the group's max width, one u32 base
//!   per group — padding traded against offsets). The two `Planes*` layouts
//!   have a constant stride and pay nothing here.
//!
//! ## ⚠️ Two accountings, and they are not interchangeable
//!
//! The dossier has been bitten three times by mixing byte accountings in one
//! list, so this bench states both and never blends them.
//!
//! * **payload** (`b/weight payload`, the historical column of this bench):
//!   numerator = the per-block stream and its base table, nothing else;
//!   denominator = **quantized weights only**, `24 × blocks`. Tails and row
//!   scales are excluded from both ends. This is the number that answers
//!   "what does one block cost", and it is what `Slot32 = 5.3756` means here.
//! * **kernel** (`b/weight kernel`), the accounting of the CUDA bench
//!   (`llvq-cuda/src/bin/planesbench.rs`, `docs/mesures/e2-golay70-*.txt`):
//!   numerator = the same stream **plus** the `f32` tail block and the `f32`
//!   row-scale vector every arm uploads; denominator = **every weight of the
//!   matrix**, `d_out · d_in`, tail columns included. This is what a matvec
//!   actually streams per weight, and it is what `Slot32 = 5.510`,
//!   `Planes14 = 4.804`, `Planes12x = 4.342` mean. Reproducing those three
//!   numbers from real blocks, on the CPU, is this bench's acceptance test —
//!   see `published_4b_kernel_rates_are_reproduced`.
//!
//! Neither is `b/param whole model`: that third quantity divides by the
//! whole model, embedding included, and is the only one comparable to an
//! AWQ figure. It is printed only when the file is self-contained, so the
//! carried parameters are **counted from the file** rather than assumed.
//!
//! ## Run
//!
//! `cargo run --release -p llvq-bench --bin rtbits [path/to/model.llvq]`
//!
//! Without a path: 20,000 gaussian blocks through `nearest_angular`, the
//! same source as `classprofile`/`arrbits`. With one: every block of the
//! artifact, exhaustively.
//!
//! [`PlanesBlocks`]: llvq_artifact::runtime::PlanesBlocks
//! [`Planes12xBlocks`]: llvq_artifact::runtime::Planes12xBlocks

use llvq_artifact::e1c::{group_words, E1C12_PLANES, E1C14_PLANES, E1C_GROUP};
use llvq_artifact::runtime::{
    PLANES12X_BYTES, PLANES12X_EXC_BYTES, PLANES12X_LEVEL_CAP, PLANES12X_PLANE_BIT, PLANES14_BYTES,
};
use llvq_core::{SplitMix64, DIM};
use llvq_search::fastdec::{ClassLevels, FastDecoder, MAX_LEVELS};
use llvq_search::generic::BallSearcher;
use llvq_search::Searcher;
use std::fs::File;
use std::io::{BufReader, Read};

mod blocks;
mod rates;
mod v5;
mod whole;

use blocks::*;
use rates::*;
use v5::*;
use whole::*;

/// A minimal JSON reader for the two documents this bench meets (a
/// safetensors header and `inkling-seal.json`): `llvq-bench` stays free of
/// external dependencies, so no `serde_json`.
mod json;

fn main() {
    // `rtbits [model.llvq] [--sidecar model.sidecar.safetensors]`: the sidecar
    // joins the whole-model accounting of a v5 file (the Inkling seal).
    let mut path = None;
    let mut sidecar: Option<String> = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        if a == "--sidecar" {
            sidecar = Some(args.next().expect("--sidecar needs a path"));
        } else {
            assert!(!a.starts_with("--"), "unknown flag {a}");
            path = Some(a);
        }
    }
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

    // ---- kernel-accounting bookkeeping (file mode only) ----
    // Shapes, so the tail and row-scale terms are the file's own and not a
    // remembered constant; and the exception count of `Planes12x`, which is
    // a plain count of five-level classes.
    let mut carried = Shapes::default();
    let mut exceptions = 0u64;
    // Parameters the file carries unquantized (embedding, lm_head). Counted
    // from the raw-tensor section when the file is self-contained, left at 0
    // otherwise — a projections-only file cannot know them, and guessing
    // them is exactly how a b/param figure goes wrong.
    let mut carried_params = 0u64;
    // Of those, the ones the `LLVQ_EMBED=q8` path actually replaces. The
    // names are `llvq_llm::fused::{EMBED_NAME, HEAD_NAME}`, repeated here
    // rather than depended on: `llvq-bench` stays dependency-free, and a
    // 690-crate tree to learn two strings is not a trade. Everything else
    // carried (the norms) stays f16 on both lines — charging norms at 8.5
    // would be inventing a quantizer that does not exist.
    let mut carried_embed = 0u64;
    // And what those tensors actually weigh in the file, encoding included.
    // The `f16 = 16` / `q8 = 8.5` rows below are a *model* of the carried
    // side; this is the file's own answer, and printing both is how the
    // model gets audited instead of trusted.
    let mut carried_bits = 0u64;

    // Per-class cross-check bookkeeping: the first block seen of each class
    // is decoded and its point-derived levels compared to the class table.
    let mut verified = vec![false; fd.n_classes()];
    let mut verified_classes = 0usize;

    if let Some(path) = path {
        // ---- every block of a real artifact ----
        let f = File::open(&path).unwrap_or_else(|e| panic!("open {path}: {e}"));
        let mut r = BufReader::new(f);
        let h = llvq_artifact::read_header(&mut r).expect("valid artifact header");
        // Every accumulator below keys on a v1 class; a Tetra word names none,
        // and the class table would file it under some class all the same. So
        // a file that is not ball-only takes a different report — the rate
        // arithmetic, with no class histogram and no claim to one — and this
        // function ends there.
        if llvq_artifact::runtime::require_ball_kinds(h.kinds(), "rtbits").is_err() {
            report_v5(&path, &mut r, &h, sidecar.as_deref());
            return;
        }
        assert!(sidecar.is_none(), "--sidecar is read for a v5 file only");
        source = format!("{path} — {} matrices", h.matrices);
        for _ in 0..h.matrices {
            let m = llvq_artifact::read_matrix_raw(&mut r, h.version).expect("valid matrix");
            // The record's own kind, not only the header's declared set: the
            // set is what the writer promised, this is what the record is.
            llvq_artifact::runtime::require_ball(m.kind, "rtbits").unwrap_or_else(|e| {
                panic!(
                    "{}: a {} record in a {} file — rtbits reads indices as v1 classes; {e}",
                    m.name, m.kind, h.kinds()
                )
            });
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
            // The two bit-plane layouts freeze the gain field at bit 9, one
            // bit wide (`transcode_planes14` refuses anything else). Pricing
            // them on a file they could not encode would be a number about
            // nothing, so refuse here rather than print it.
            assert_eq!(
                gain_bits, 1,
                "{}: {} gain bits — Planes14/Planes12x freeze the field at 1",
                m.name,
                gain_bits
            );
            carried.push(m.d_out, m.d_in);
            for &idx in &m.indices {
                let Some(ci) = fd.class_of(idx) else {
                    assert_eq!(idx, 0, "{}: index {idx} outside the ball", m.name);
                    origin_blocks += 1;
                    // The origin still occupies a lane: one level, no signs,
                    // no masks — only the class id and the gain bit.
                    let w = Widths {
                        mask: CLASS_BITS + gain_bits,
                        pos: CLASS_BITS + gain_bits,
                        // Mirrors the origin record of `ClassTable::new`:
                        // no sign field and no masks, so the general
                        // `9 + gain + 24·L` does not apply.
                        slot: CLASS_BITS + gain_bits,
                        levels: 1,
                        origin: true,
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
                exceptions += u64::from(is_planes12x_exception(lv.len));
                let w = widths(lv, gain_bits);
                if lv.odd { &mut odd } else { &mut even }.push(&w);
                grouped.push(&w);
            }
        }
        // ---- carried tensors, counted rather than assumed ----
        // A self-contained file lists its raw tensors right after the
        // matrices: a u32 count, then one framed tensor each. Reading them
        // costs one transient allocation per tensor and turns "b/param whole
        // model" from a figure with a remembered vocabulary size into one the
        // file itself answers for.
        if h.is_self_contained() {
            let mut b = [0u8; 4];
            r.read_exact(&mut b).expect("raw-tensor count");
            for _ in 0..u32::from_le_bytes(b) {
                let t = llvq_artifact::read_raw(&mut r, h.version).expect("valid raw tensor");
                carried_params += t.len() as u64;
                carried_bits += t.bytes() * 8;
                if matches!(
                    t.name.as_str(),
                    "model.embed_tokens.weight" | "lm_head.weight"
                ) {
                    carried_embed += t.len() as u64;
                }
            }
        }
    } else {
        // ---- gaussian source, same harness as classprofile/arrbits ----
        const N: usize = 20_000;
        source = format!("{N} gaussian blocks, nearest_angular");
        let mut rng = SplitMix64::new(0x6_A88B);
        let s = Searcher::new();
        let mut ball = BallSearcher::new();
        for _ in 0..N {
            let x: [f64; DIM] = core::array::from_fn(|_| rng.next_gaussian());
            let f = ball.nearest_angular(&s, &x);
            let lv = levels_of_point(&f.point);
            exceptions += u64::from(is_planes12x_exception(lv.len));
            let w = widths(&lv, gain_bits);
            if lv.odd { &mut odd } else { &mut even }.push(&w);
            grouped.push(&w);
        }
    }
    grouped.close();

    let total = odd.blocks + even.blocks;
    let pct = |n: u64| 100.0 * n as f64 / total as f64;
    let bpw = |bits: u64, blocks: u64| bits as f64 / (blocks * DIM as u64) as f64;

    println!("source: {source}");
    println!("{total} blocks — odd {:.1} %, even {:.1} %", pct(odd.blocks), pct(even.blocks));
    if origin_blocks > 0 {
        println!("of which {origin_blocks} origin blocks (all zero)");
    }
    if verified_classes > 0 {
        println!("{verified_classes} classes cross-checked, decoded point ↔ table");
    }

    println!("\n  magnitude levels per block");
    println!("  {}", "-".repeat(56));
    for l in 1..=MAX_LEVELS {
        let n = odd.levels[l] + even.levels[l];
        if n > 0 {
            println!("  {l} levels{:>17} blocks{:>8.2} %", n, pct(n));
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

    println!("\n  bits/weight in RAM, full accounting");
    println!("  (assignment + signs + class {CLASS_BITS} b + gain {gain_bits} b + addressing)");
    println!("  {}", "-".repeat(72));
    println!(
        "  {:<44}{:>10}{:>10}",
        "layout", "masks", "positional"
    );
    println!(
        "  {:<44}{:>10.4}{:>10.4}",
        "variable, free addressing (lower bound)",
        bpw(all_mask, total),
        bpw(all_pos, total)
    );
    println!(
        "  {:<44}{:>10.4}{:>10.4}",
        "variable + u16 offset per block",
        bpw(all_mask + 16 * total, total),
        bpw(all_pos + 16 * total, total)
    );
    println!(
        "  {:<44}{:>10.4}{:>10.4}",
        "grouped 32 (stride = group max + u32 base)",
        bpw(grouped.mask_bits, total),
        bpw(grouped.pos_bits, total)
    );
    println!(
        "  {:<44}{:>10.4}{:>10}",
        "grouped 32, stride rounded to the byte",
        bpw(grouped.mask_bits_byte, total),
        "—"
    );
    println!(
        "  {:<44}{:>10.4}{:>10.4}",
        format!("worst-case fixed field ({worst_mask}/{worst_pos} b → u32)"),
        bpw(fixed_mask * total, total),
        bpw(fixed_pos * total, total)
    );
    println!(
        "  {:<44}{:>10}{:>10.4}",
        "fixed field 128 (uint4, nibbles)", "—",
        128.0 / DIM as f64
    );

    println!("\n  by coset (variable, free addressing)");
    println!("  {}", "-".repeat(72));
    for (name, a) in [("odd", &odd), ("even", &even)] {
        if a.blocks > 0 {
            println!(
                "  {:<10}{:>9.1} %   masks {:.4}   positional {:.4}   worst {}/{} b",
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
        println!("\n  divergence over groups of {GROUP} consecutive lanes");
        println!("  {}", "-".repeat(72));
        println!(
            "  levels: mean {mean_l:.3}, mean of the per-group maxima {mean_max_l:.3} \
             (×{:.3})",
            mean_max_l / mean_l
        );
        println!(
            "  grouped-32 padding: masks ×{:.3}, positional ×{:.3} vs free addressing",
            grouped.mask_bits as f64 / all_mask as f64,
            grouped.pos_bits as f64 / all_pos as f64
        );
    }

    // ---- Slot32: the layout the fused kernel reads, and what a level cap
    // would do to it ----
    //
    // `Slot32` is a third assignment encoding, not a variant of the two
    // above: one 24-bit mask per level in **slot** space, level 0 implicit,
    // signs indexed by slot. It buys fixed field offsets — the whole reason
    // the fused kernel has no popcount chain and no serial state — and pays
    // `24 − nonzero` bits per block for them.
    if grouped.groups > 0 {
        // Same total by a second, independent route: from the histogram of
        // per-group maxima, through the closed-form stride. If the running
        // accumulator and this disagree, one of the two is wrong and no
        // figure below can be believed.
        let from_hist: u64 = (1..=MAX_LEVELS)
            .map(|l| grouped.maxl_hist[l] * GROUP as u64 * slot_stride_bits(l, gain_bits))
            .sum::<u64>()
            + grouped.origin_only_groups
                * GROUP as u64
                * (CLASS_BITS + gain_bits).div_ceil(8)
                * 8
            + 32 * grouped.groups;
        assert_eq!(
            from_hist, grouped.slot_bits_byte,
            "the histogram of maxima and the stride accumulator disagree"
        );

        println!("\n  Slot32 — the layout the fused kernel actually reads");
        println!("  {}", "-".repeat(72));
        println!(
            "  grouped 32, byte stride, u32 base: {:.4} b/weight",
            bpw(grouped.slot_bits_byte, total)
        );
        println!("  {} groups, level maxima:", grouped.groups);
        for l in 1..=MAX_LEVELS {
            if grouped.maxl_hist[l] > 0 {
                println!(
                    "    L = {l}{:>14} groups{:>9.4} %   stride {} B",
                    grouped.maxl_hist[l],
                    100.0 * grouped.maxl_hist[l] as f64 / grouped.groups as f64,
                    slot_stride_bits(l, gain_bits) / 8
                );
            }
        }
        if grouped.origin_only_groups > 0 {
            println!(
                "    origin only{:>15} groups",
                grouped.origin_only_groups
            );
        }

        // What an L ≤ 4 encoder cap would cost. This is a **bound**, not a
        // simulation: capping re-quantizes, so the classes of the L = 5
        // blocks would change and cannot be predicted here. What can be
        // stated exactly:
        //
        //  * `L ≤ 4 ⇒ width_slot ≤ 9 + gain + 96 = 106 b ⇒ stride ≤ 14 B`,
        //    so 14 bytes is an unconditional majorant of every group;
        //  * a block already at `L ≤ 4` keeps its class under the cap — its
        //    codeword is the argmin over the full ball and stays the argmin
        //    over a subset that still contains it — so a group holding an
        //    L = 4 block reaches exactly 14 B, majorant attained.
        let capped_bits = grouped.groups * (GROUP as u64 * slot_stride_bits(4, gain_bits) + 32);
        println!(
            "  under an L ≤ 4 cap: ≤ {:.4} b/weight ({} B of stride, unconditional majorant)",
            bpw(capped_bits, total),
            slot_stride_bits(4, gain_bits) / 8
        );
        println!(
            "  majorant ATTAINED on {} groups out of {} ({:.4} %): those that already carry\n  \
             an L = 4 block, whose class the cap leaves unchanged. Saves {:.3} b/weight ({:.1} %).",
            grouped.groups_with_four,
            grouped.groups,
            100.0 * grouped.groups_with_four as f64 / grouped.groups as f64,
            bpw(grouped.slot_bits_byte, total) - bpw(capped_bits, total),
            100.0 * (1.0 - bpw(capped_bits, total) / bpw(grouped.slot_bits_byte, total))
        );
    }

    // ---- Planes14 / Planes12x: the bit-plane layouts ----
    //
    // Neither is grouped and neither has a base table, so both are priced by
    // a stride times a block count — plus, for Planes12x, a table whose size
    // is a *count of five-level classes*. Payload accounting: the stream
    // alone, over `24 · blocks`, exactly like every column above.
    let p14_bits = planes14_bits(total);
    let p12_bits = planes12x_bits(total, exceptions);
    let exc_pct = 100.0 * exceptions as f64 / total as f64;
    println!("\n  Planes14 and Planes12x — the bit planes (b/weight payload)");
    println!("  {}", "-".repeat(72));
    println!(
        "  Planes14   fixed {PLANES14_BYTES} B stride, no base and no exception: {:.4} b/weight",
        bpw(p14_bits, total)
    );
    println!(
        "  Planes12x  {PLANES12X_BYTES} B stride + {} exceptions at {PLANES12X_EXC_BYTES} B \
         ({exc_pct:.4} % of blocks)",
        exceptions
    );
    println!(
        "             = {:.4} b/weight, of which {:.4} stream and {:.4} exceptions",
        bpw(p12_bits, total),
        bpw(total * PLANES12X_BYTES as u64 * 8, total),
        bpw(exceptions * PLANES12X_EXC_BYTES as u64 * 8, total)
    );
    // The exception count is a histogram identity, not a second measurement:
    // a block is an exception iff its class has more than PLANES12X_LEVEL_CAP
    // levels. Stating it as an assert makes a drift between the predicate and
    // the level histogram impossible to publish.
    let from_hist: u64 = (1..=MAX_LEVELS)
        .filter(|&l| exceeds_planes12x_geometry(l))
        .map(|l| odd.levels[l] + even.levels[l])
        .sum();
    assert_eq!(
        from_hist, exceptions,
        "the exception count and the level histogram disagree"
    );
    println!(
        "  threshold: L > {PLANES12X_LEVEL_CAP} — {} blocks at {} levels, cross-checked on the histogram",
        exceptions,
        MAX_LEVELS
    );

    // ---- E1c: the same two streams with the byte rounding removed ----
    let e14_bits = e1c14_bits(total);
    let e12_bits = e1c12_bits(total, exceptions);
    println!("\n  E1c — the same streams transposed over the warp (b/weight payload)");
    println!("  {}", "-".repeat(72));
    println!(
        "  E1c14      {} words per group of {E1C_GROUP}, no padding: {:.4} b/weight \
         ({:+.4} vs Planes14)",
        group_words(E1C14_PLANES),
        bpw(e14_bits, total),
        bpw(e14_bits, total) - bpw(p14_bits, total)
    );
    println!(
        "  E1c12      {} words per group + the SAME {} exceptions: {:.4} b/weight \
         ({:+.4} vs Planes12x)",
        group_words(E1C12_PLANES),
        exceptions,
        bpw(e12_bits, total),
        bpw(e12_bits, total) - bpw(p12_bits, total)
    );
    println!(
        "             of which {:.4} stream and {:.4} exceptions — the exception term\n  \
         is identical to Planes12x's, so the gap between the two layouts IS the padding",
        bpw(e12_bits - exceptions * PLANES12X_EXC_BYTES as u64 * 8, total),
        bpw(exceptions * PLANES12X_EXC_BYTES as u64 * 8, total)
    );

    // ---- the kernel accounting, the one the CUDA bench publishes ----
    //
    // Same streams, two changes and only two: the `f32` tail block and the
    // `f32` row-scale vector join the numerator, and the denominator becomes
    // every weight of every matrix instead of `24 · blocks`. Both changes are
    // *inflationary on the numerator* and *inflationary on the denominator*,
    // and they do not cancel — which is precisely why the two columns must
    // never be quoted in one list.
    if carried.weights > 0 {
        let tail_w = carried.weights - total * DIM as u64;
        println!("\n  b/weight KERNEL — the CUDA bench accounting (planesbench.rs)");
        println!("  {}", "-".repeat(72));
        println!(
            "  numerator += f32 tail ({tail_w} weights) + f32 row scales ({} rows)",
            carried.rows
        );
        println!(
            "  denominator = {} weights (Σ d_out·d_in, tail included), not {} quantized",
            carried.weights,
            total * DIM as u64
        );
        assert_eq!(
            tail_w, carried.tail_weights,
            "tail weights counted by shape and by difference disagree"
        );
        for (name, bits) in [
            ("Slot32", grouped.slot_bits_byte),
            ("Planes14", p14_bits),
            ("Planes12x", p12_bits),
            ("E1c14", e14_bits),
            ("E1c12", e12_bits),
        ] {
            println!(
                "  {name:<12}{:>10.4} b/weight kernel  ({:.4} payload, {:.2} GB of stream)",
                carried.kernel_bpw(bits),
                bpw(bits, total),
                (bits + carried.side_bits()) as f64 / 8.0 / 1e9
            );
        }

        // ---- E1c ALIGNED: the verdict e1c12 was missing ----
        //
        // X3 (2026-08-15) showed that no warp of the published model reads a
        // single E1c group — `nblocks mod 32` is 10 or 21 on the five shapes —
        // and that the only remedy is to pad every row. It returned its
        // verdict on `e1c14` (5.2354, bigger than the Planes14 it replaces)
        // and STOPPED at `e1c12`, for want of the exception term: its model
        // gave 4.1404 for Planes12x where the repository publishes 4.342, and
        // both sides of the ratio depend on it.
        //
        // Here both sides come from the same sweep, exceptions included. The
        // exception term does NOT move with the padding: a padding block is
        // synthetic, it has no five-level class.
        let pad = carried.padded_blocks;
        println!("
  E1c WARP-ALIGNED — every row padded to a multiple of 32");
        println!("  {}", "-".repeat(72));
        println!(
            "  {total} blocks → {pad} ({:+.2} %), and these blocks carry NO weight:
               the numerator swells, the denominator does not move",
            (pad as f64 / total as f64 - 1.0) * 100.0
        );
        for (name, aligned, served, served_name) in [
            ("E1c14", e1c14_bits(pad), p14_bits, "Planes14"),
            ("E1c12", e1c12_bits(pad, exceptions), p12_bits, "Planes12x"),
        ] {
            let a = carried.kernel_bpw(aligned);
            let s = carried.kernel_bpw(served);
            println!(
                "  {name:<12}{a:>10.4} b/weight kernel aligned  against {s:.4} for {served_name}                  — {:+.1} %  {}",
                (a / s - 1.0) * 100.0,
                if a < s { "OK" } else { "FAIL: BIGGER than the layout it replaces" }
            );
        }
        println!(
            "  WARNING: neither of these two figures is a speed. It is a counting
               identity on the real shapes, the kind that buried E3. It depends on no
               hardware, and a better kernel does not get around it."
        );
    }

    // ---- b/param whole model: the only figure comparable to an AWQ one ----
    //
    // Printed only when the file carries its embedding, so both counts of the
    // division come from the same bytes. `q8` is the `LLVQ_EMBED=q8` path:
    // int8 plus one f16 scale and one f16 bias per group of 64, i.e. exactly
    // 8 + 32/64 = 8.5 bits per parameter when the row length is a multiple of
    // 64 (2560 and 4096 both are).
    if carried_params > 0 && carried.weights > 0 {
        let embed_q8 = 8.0 + EMBED_GROUP_BITS / EMBED_GROUP;
        let other = carried_params - carried_embed;
        println!("\n  b/param WHOLE MODEL — the only quantity comparable to an AWQ figure");
        println!("  {}", "-".repeat(72));
        println!(
            "  {} quantized weights + {} of embedding + {other} carried in f16 = {} parameters",
            carried.weights,
            carried_embed,
            carried.weights + carried_params
        );
        println!(
            "  embedding f16 = 16.000 b/param; q8 g{:.0} = {embed_q8:.3} b/param \
             (int8 + f16 scale and bias per group)",
            EMBED_GROUP
        );
        // The third column is not a model: it is the carried section's own
        // byte count, so the row that matches this file's encoding must
        // reproduce one of the two modelled columns. If it does not, the
        // model is wrong and the modelled columns are not publishable.
        let carried_measured = carried_bits as f64 / carried_params as f64;
        println!(
            "  carried MEASURED in the file: {carried_measured:.3} b/param \
             ({:.2} GB over {carried_params} parameters)",
            carried_bits as f64 / 8.0 / 1e9
        );
        println!(
            "  {:<12}{:>16}{:>16}{:>16}",
            "layout", "embed f16", "embed q8", "carried measured"
        );
        for (name, bits) in [
            ("Slot32", grouped.slot_bits_byte),
            ("Planes14", p14_bits),
            ("Planes12x", p12_bits),
            ("E1c14", e14_bits),
            ("E1c12", e12_bits),
        ] {
            let lin = (carried.weights, carried.kernel_bpw(bits));
            println!(
                "  {name:<12}{:>16.3}{:>16.3}{:>16.3}",
                model_bpw(&[lin, (carried_embed, 16.0), (other, 16.0)]),
                model_bpw(&[lin, (carried_embed, embed_q8), (other, 16.0)]),
                model_bpw(&[lin, (carried_params, carried_measured)])
            );
        }
    }

    // ---- what each rate means for one token ----
    //
    // Both terms used to be Qwen3-4B constants. They are now taken from the
    // file when there is one — a 4B file reproduces the old numbers exactly
    // (3,633,315,840 weights, 0.778 GB of tied embedding), and an 8B file no
    // longer gets a 4B's head silently charged to it.
    let (linear_w, lm_head_gb, head_src) = match (carried.weights, carried_params) {
        (0, _) => (3_633_315_840f64, 0.778, "Qwen3-4B constants"),
        (w, 0) => (w as f64, 0.778, "WARNING: lm_head ASSUMED (projections-only file)"),
        (w, c) => (w as f64, c as f64 * 2.0 / 1e9, "measured on the file"),
    };
    println!(
        "\n  traffic per token — linears + f16 embedding {lm_head_gb:.3} GB ({head_src})"
    );
    println!("  {}", "-".repeat(72));
    let show = |name: &str, b: f64| {
        let gb = linear_w * b / 8.0 / 1e9 + lm_head_gb;
        println!(
            "  {name:<44}{b:>7.3} b/p{:>9.2} GB{:>9.0} tok/s",
            gb,
            400.0 / gb
        );
    };
    show(
        &format!("archive v1, {arch_bits} b/block (no fused decode)"),
        arch_bits as f64 / 24.0,
    );
    show("variable masks + u16", bpw(all_mask + 16 * total, total));
    show("variable positional + u16", bpw(all_pos + 16 * total, total));
    show("grouped 32, masks", bpw(grouped.mask_bits, total));
    show("grouped 32, masks, byte stride", bpw(grouped.mask_bits_byte, total));
    show("grouped 32, positional", bpw(grouped.pos_bits, total));
    show("worst-case fixed field (masks, u32)", bpw(fixed_mask * total, total));
    show("worst-case fixed field (pos., u32)", bpw(fixed_pos * total, total));
    show("fixed field 128", 128.0 / 24.0);
    if grouped.groups > 0 {
        show(
            "grouped 32, Slot32",
            bpw(grouped.slot_bits_byte, total),
        );
        show(
            "grouped 32, Slot32, L ≤ 4 cap (majorant)",
            bpw(
                grouped.groups * (GROUP as u64 * slot_stride_bits(4, gain_bits) + 32),
                total,
            ),
        );
    }
    show("Planes14 — WHAT RUNS TODAY", bpw(p14_bits, total));
    show(
        &format!("Planes12x ({exc_pct:.2} % exceptions)"),
        bpw(p12_bits, total),
    );
    show("E1c14 — transposed, no padding", bpw(e14_bits, total));
    show("E1c12 — transposed + overlay", bpw(e12_bits, total));
    let fp16_gb = linear_w * 2.0 / 1e9 + lm_head_gb;
    println!(
        "\n  FP16 ceiling: ~{:.0} tok/s; embedding-only ceiling: ~{:.0} tok/s",
        400.0 / fp16_gb,
        400.0 / lm_head_gb
    );
}

#[cfg(test)]
mod tests;
