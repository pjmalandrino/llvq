//! **P5 C1, second half — the ADDRESSED width of the CNS flow.**
//!
//! The sweep (`p5_cns_sweep.rs`) gives the **raw** width: the record's own bits,
//! header included. The number the dossier argues with is the **addressed** one
//! — `53.332 bits/block, FO/warp-scan` for the per-stage variant, and the
//! `2.3709 b/weight kernel` that derives from it. Amendment É0 moved C1's target
//! to the per-kind accounting, and **nobody has ever computed the per-kind
//! addressed figure**. This does.
//!
//! ## The addressing model, restated from the pre-registration's prose
//!
//! P5 §1.1: *"+ a 32-bit base word per group of 32, + rounding to the word"*. So
//! a group of 32 consecutive blocks costs
//!
//! ```text
//!   32  (the base word)  +  ceil( Σ widths / 32 ) × 32
//! ```
//!
//! and the figure is that, averaged over blocks. It is restated here rather than
//! imported: `radixstudy` is a **binary** in another crate, so importing it is
//! not merely forbidden by §4 — it is impossible. Which makes the model a second
//! derivation, and a second derivation has to be validated before it is used.
//!
//! ## Validated before it is believed
//!
//! The same model, fed the **per-stage** widths, must reproduce the two published
//! numbers: **52.869** in class-major order and **53.332** in file order, within
//! the tolerance §4 already uses for those columns (`5e-3` on a bits/block mean).
//! The per-stage column exists **only** as that control — it is computed here,
//! never in `llvq_search::cns`, whose own test forbids it (§4's independence
//! clause).
//!
//! 🚨 **If the control misses, nothing downstream is published.** A model that
//! cannot reproduce a known number has no business producing an unknown one.
//!
//! `#[ignore]`d unconditionally: it reads a 981 MB archive that is not in the
//! repo, and it fails loudly rather than skipping green when the file is absent.

use llvq_core::{Golay, DIM};
use llvq_search::cns::{cns_layout, lg_ceil, HEADER_BITS};
use llvq_search::fastdec::{FastDecoder, MAX_KINDS};
use llvq_search::rankdec::walk_radices;

mod common;

/// Blocks per addressing group — the warp.
const GROUP: usize = 32;

/// The two published per-stage figures this model must reproduce, and the
/// tolerance §4 already applies to a bits/block mean.
const PUBLISHED_CM: f64 = 52.869;
const PUBLISHED_FO: f64 = 53.332;
const TOL_BITS: f64 = 5e-3;

/// §1.2's kernel accounting, read from the pre-registration rather than
/// recomputed: `(bits/block × blocks + (tail + rows) × 32) / weights`. The four
/// counts are the sealed 4B's, pinned in `radixstudy` against the shapes read
/// from the file itself.
const WEIGHTS: f64 = 3_633_315_840.0;
const TAIL: f64 = 16_957_440.0;
const ROWS: f64 = 1_105_920.0;
const BLOCKS: f64 = 150_681_600.0;
const TOL_BPW: f64 = 5e-5;

fn kernel_bpw(bits_per_block: f64) -> f64 {
    (bits_per_block * BLOCKS + (TAIL + ROWS) * 32.0) / WEIGHTS
}

/// The **per-stage** width of a class — the control column, and the one the
/// published 51.87 / 53.332 belong to.
///
/// One `⌈log₂⌉` per composition radix: the codeword, the two arrangement
/// counts **taken whole**, and the two sign fields. Its arrangement field is a
/// packed multiset rank, which is exactly why peeling it needs a division and
/// why É0 moved C1 off it.
fn per_stage_bits(fd: &FastDecoder, golay: &Golay, ci: usize) -> u64 {
    let c = fd.cascade_class(ci);
    let lg = |x: u64| u64::from(lg_ceil(u128::from(x)));
    if c.odd {
        return HEADER_BITS + lg(golay.codewords().len() as u64) + lg(c.m_arr);
    }
    HEADER_BITS
        + lg(golay.of_weight(c.w as usize).len() as u64)
        + lg(c.a_on)
        + lg(c.a_off)
        + lg(c.s_w)
        + lg(c.s_f)
}

/// A group's addressed cost, in bits.
fn group_bits(widths: &[u64]) -> u64 {
    let raw: u64 = widths.iter().sum();
    32 + raw.div_ceil(32) * 32
}

#[test]
#[ignore = "reads the sealed 4B archive; a minute even in release"]
fn the_cns_addressed_width_is_what_it_is() {
    let path = common::sealed_artifact_path();
    let fd = FastDecoder::new();
    let golay = Golay::new();

    // Two width tables, same classes, two rounding granularities.
    let stage: Vec<u64> = (0..fd.n_classes())
        .map(|ci| per_stage_bits(&fd, &golay, ci))
        .collect();
    let kind: Vec<u64> = (0..fd.n_classes())
        .map(|ci| cns_layout(&fd, &golay, ci).bits())
        .collect();

    // The two arrangements are where the whole difference lives: the sign
    // fields are already powers of two in the archive's composition, so both
    // granularities agree on them. Asserted rather than assumed — if it ever
    // stopped being true, the delta below would be measuring something else.
    for ci in 0..fd.n_classes() {
        let c = fd.cascade_class(ci);
        let per_kind_on: u64 = if c.odd {
            let mut counts = [0u8; MAX_KINDS];
            counts[..c.n_kinds].copy_from_slice(&c.counts[..c.n_kinds]);
            walk_radices(&counts, c.n_kinds, DIM)
                .iter()
                .map(|&r| u64::from(lg_ceil(u128::from(r))))
                .sum()
        } else {
            0
        };
        if c.odd {
            assert_eq!(
                kind[ci] - stage[ci],
                per_kind_on - u64::from(lg_ceil(u128::from(c.m_arr))),
                "class {ci}: the gap does not come from the arrangement alone"
            );
        }
        assert!(kind[ci] >= stage[ci], "class {ci}: the per-kind width is narrower");
    }

    // ---- one pass, both orders -------------------------------------------
    let f = std::fs::File::open(&path).expect("open the sealed artifact");
    let mut r = std::io::BufReader::new(f);
    let h = llvq_artifact::read_header(&mut r).expect("valid artifact header");
    let mut hist = vec![0u64; fd.n_classes()];
    let (mut fo_stage, mut fo_kind, mut groups, mut total) = (0u64, 0u64, 0u64, 0u64);

    for _ in 0..h.matrices {
        let m = llvq_artifact::read_matrix_raw(&mut r).expect("valid matrix");
        assert_eq!(
            m.indices.len() % GROUP,
            0,
            "{}: {} blocks, partial group — the dossier counts none",
            m.name,
            m.indices.len()
        );
        // FILE ORDER: groups of 32 consecutive blocks, classes mixed as the
        // model mixes them. This is what makes the rounding cost anything.
        for g in m.indices.chunks(GROUP) {
            let (mut ws, mut wk) = ([0u64; GROUP], [0u64; GROUP]);
            for (i, &idx) in g.iter().enumerate() {
                let ci = fd.class_of(idx).expect("the 4B carries no origin block");
                hist[ci] += 1;
                total += 1;
                ws[i] = stage[ci];
                wk[i] = kind[ci];
            }
            fo_stage += group_bits(&ws);
            fo_kind += group_bits(&wk);
            groups += 1;
        }
    }
    assert_eq!(total, 150_681_600, "the published 4B has 150,681,600 blocks");
    assert_eq!(groups, 4_708_800, "4,708,800 groups, none partial");

    // CLASS-MAJOR: every block of a class together, so a group is 32 identical
    // widths — `32·w + 32` is a whole number of words and the rounding costs
    // exactly nothing. That is why the CM column is the raw mean plus one bit,
    // and why it is the cleaner of the two controls.
    let cm = |t: &[u64]| -> f64 {
        let mut bits = 0u64;
        for (ci, &c) in hist.iter().enumerate() {
            let g = c / GROUP as u64;
            bits += g * group_bits(&vec![t[ci]; GROUP]);
            let rest = c % GROUP as u64;
            if rest > 0 {
                bits += group_bits(&vec![t[ci]; rest as usize]);
            }
        }
        bits as f64 / total as f64
    };
    let (raw_stage, raw_kind) = (
        hist.iter().zip(&stage).map(|(&c, &w)| c * w).sum::<u64>() as f64 / total as f64,
        hist.iter().zip(&kind).map(|(&c, &w)| c * w).sum::<u64>() as f64 / total as f64,
    );
    let (cm_stage, cm_kind) = (cm(&stage), cm(&kind));
    let (fo_s, fo_k) = (fo_stage as f64 / total as f64, fo_kind as f64 / total as f64);

    eprintln!(
        "P5 C1 — addressed width, {total} blocks, {groups} groups of {GROUP}\n\n  \
         CONTROL, PER-STAGE accounting (the one the published figures use)\n    \
         raw {raw_stage:.4}  ·  class-major {cm_stage:.4} (published {PUBLISHED_CM})  ·  \
         file-order {fo_s:.4} (published {PUBLISHED_FO})\n\n  \
         RESULT, PER-KIND accounting (É0 — the one that decodes without division)\n    \
         raw {raw_kind:.4}  ·  class-major {cm_kind:.4}  ·  file-order {fo_k:.4}\n    \
         b/weight kernel: {:.4}  (against {:.4} per stage, and a C0 criterion of 2.60)\n\n  \
         gap between the two accountings: {:+.4} bit/block raw, {:+.4} addressed FO",
        kernel_bpw(fo_k),
        kernel_bpw(fo_s),
        raw_kind - raw_stage,
        fo_k - fo_s,
    );

    // 🚨 The control decides whether any of the above may be published.
    assert!(
        (cm_stage - PUBLISHED_CM).abs() < TOL_BITS,
        "the addressing model does not reproduce the published class-major: \
         {cm_stage:.4} against {PUBLISHED_CM} (tolerance {TOL_BITS})"
    );
    assert!(
        (fo_s - PUBLISHED_FO).abs() < TOL_BITS,
        "the addressing model does not reproduce the published file-order: \
         {fo_s:.4} against {PUBLISHED_FO} (tolerance {TOL_BITS})"
    );
    // And the b/weight it derives, against the published 2.3709.
    assert!(
        (kernel_bpw(fo_s) - 2.3709).abs() < TOL_BPW,
        "the per-stage b/weight kernel does not reproduce 2.3709: {:.6}",
        kernel_bpw(fo_s)
    );
}
