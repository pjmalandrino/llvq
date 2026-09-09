//! `FusedLayout::Tetra48`: the layout resolves end to end, on every machine.
//!
//! This file exists because of a defect class, not a feature. On 2026-09-05
//! `f1floor.cu` reached a **billed job** and died on `no embedded copy of
//! f1floor.cu`: the name was in a source list and had no arm in the table that
//! turns names into text. Nothing caught it — the table is a *runtime* match,
//! so the compiler is silent, and the failure surfaces after the card is
//! rented and the image pulled.
//!
//! `llvq-cuda` closed that hole for its own table (`bin/cuhcheck`'s
//! `check_embedded`). `llvq-llm` carries a **second** table, in
//! `fused::load_planes_sources`, and it had no such check. This is it, and it
//! sweeps every layout rather than the new one — a test that only covered
//! Tetra48 would be the same defect waiting for the next layout.

use llvq_llm::fused::{
    load_planes_sources, matvec_kernel_name, planes_source_names, seg_kernel_name, FusedLayout,
};

const LAYOUTS: [FusedLayout; 5] = [
    FusedLayout::Planes14,
    FusedLayout::Planes12x,
    FusedLayout::Slot32,
    FusedLayout::Golay70,
    FusedLayout::Tetra48,
];

/// Every name every layout asks for has an embedded copy. The check a rented
/// card should never be the first to make.
#[test]
fn every_source_name_resolves_to_embedded_text() {
    for layout in LAYOUTS {
        let names = planes_source_names(layout);
        let (parts, overridden) = load_planes_sources(layout)
            .unwrap_or_else(|e| panic!("{}: {e}", layout.name()));
        assert!(overridden.is_none(), "{}: LLVQ_KERNEL_DIR set during tests", layout.name());
        assert_eq!(parts.len(), names.len(), "{}", layout.name());
        for (n, text) in names.iter().zip(&parts) {
            assert!(!text.is_empty(), "{}: {n} embedded empty", layout.name());
        }
    }
}

/// The entry point each layout names is defined by the text that layout ships.
///
/// A list that resolves and a kernel name that nothing defines is the same
/// failure one stage later, and just as expensive.
#[test]
fn every_kernel_name_is_defined_by_its_own_sources() {
    for layout in LAYOUTS {
        let (parts, _) = load_planes_sources(layout).expect("embedded copies");
        let unit = parts.join("\n");
        // Slot32's entry point lives in matvec.cu, which comes from
        // `load_sources_many` ahead of this list — it ships no plane source
        // at all, and its empty list is the assertion.
        if layout == FusedLayout::Slot32 {
            assert!(parts.is_empty(), "slot32 ships no plane source");
            continue;
        }
        let name = matvec_kernel_name(layout);
        assert!(
            unit.contains(&format!("void {name}(")),
            "{}: no definition of {name} in its own sources",
            layout.name()
        );
        if let Some(seg) = seg_kernel_name(layout) {
            assert!(unit.contains(&format!("void {seg}(")), "{}: {seg}", layout.name());
        }
    }
}

/// `tetra48` parses, and a typo does not fall back to the default.
#[test]
fn the_tetra48_spelling_resolves_and_nothing_near_it_does() {
    assert_eq!(FusedLayout::parse(Some("tetra48")).expect("parses"), FusedLayout::Tetra48);
    assert_eq!(FusedLayout::parse(None).expect("default"), FusedLayout::Planes14);
    for bad in ["tetra", "tetra_48", "Tetra48", "tetra48 ", "48"] {
        assert!(FusedLayout::parse(Some(bad)).is_err(), "{bad:?} must not resolve");
    }
}

/// Tetra48 shares no source with the ball layouts, and that is structural.
///
/// A Tetra word names no class, so `llvq_planes.cuh` and its `ClassRec` table
/// have no meaning here. A list that quietly picked them up would compile and
/// then decode ball classes out of Tetra labels — finite, plausible, wrong.
#[test]
fn tetra48_shares_no_source_with_the_ball_layouts() {
    let tetra = planes_source_names(FusedLayout::Tetra48);
    for layout in [FusedLayout::Planes14, FusedLayout::Planes12x, FusedLayout::Golay70] {
        for n in planes_source_names(layout) {
            assert!(!tetra.contains(n), "tetra48 shares {n} with {}", layout.name());
        }
    }
    assert!(tetra.contains(&"llvq_tetra48.cuh"), "the served decode must be in the list");
    assert!(tetra.contains(&"tv_tetra48_h.cu"), "and its entry point");
}

/// `LLVQ_FUSE=1` is refused on Tetra48, for the reason Planes12x and Golay70
/// are refused: the segmented path is Planes14's alone.
#[test]
fn tetra48_does_not_segment() {
    assert!(seg_kernel_name(FusedLayout::Tetra48).is_none());
}

/// The int4 kernel ships, and its entry point is in the text that ships it.
///
/// `tv_q4_h.cu` has never run on a GPU. It is verified as host C++ by
/// `tests/proj_q4.rs` and by nothing else, so the least a Mac can do is
/// guarantee that the day it *is* launched, the source reaches NVRTC — the
/// failure `f1floor.cu` paid for on 2026-09-05.
#[test]
fn the_int4_kernel_ships_with_its_entry_point() {
    use llvq_llm::fused::{load_int4_sources, INT4_KERNEL_NAME};
    let (text, overridden) = load_int4_sources().expect("embedded copy");
    assert!(overridden.is_none(), "LLVQ_KERNEL_DIR set during tests");
    assert!(!text.is_empty(), "tv_q4_h.cu embedded empty");
    assert!(
        text.contains(&format!("void {INT4_KERNEL_NAME}(")),
        "no definition of {INT4_KERNEL_NAME} in the text that ships it"
    );
    // It is NOT a layout source: a mixed file carries int4 records beside
    // lattice ones, so it is selected by the file's kinds and never by
    // LLVQ_FUSED_LAYOUT. A list that picked it up would tie the two.
    for layout in LAYOUTS {
        assert!(
            !planes_source_names(layout).contains(&"tv_q4_h.cu"),
            "{}: the int4 kernel is not a layout source",
            layout.name()
        );
    }
}

// --------------------------------------------------------------------------
// The kind gate: which layout may serve which record
// --------------------------------------------------------------------------

use llvq_artifact::{CodeKind, KindSet};
use llvq_llm::fused::{check_kinds, lattice_kind, serves_kind};

/// Each layout reads exactly one lattice map, and reads no other.
///
/// This is the gate that keeps a Tetra word from being read as a ball index.
/// Neither direction fails loudly: a Tetra word names no class, a ball index
/// decodes to a different point, and both return finite, plausible, wrong
/// weights. That is the whole reason the gate exists.
#[test]
fn a_layout_reads_its_own_lattice_and_no_other() {
    for layout in LAYOUTS {
        let own = lattice_kind(layout);
        assert!(serves_kind(layout, own), "{}: its own map", layout.name());
        for kind in [CodeKind::Ball, CodeKind::Tetra] {
            if kind != own {
                assert!(
                    !serves_kind(layout, kind),
                    "{} must not read {kind:?}",
                    layout.name()
                );
            }
        }
    }
    assert_eq!(lattice_kind(FusedLayout::Tetra48), CodeKind::Tetra);
    for layout in [FusedLayout::Planes14, FusedLayout::Planes12x, FusedLayout::Golay70] {
        assert_eq!(lattice_kind(layout), CodeKind::Ball, "{}", layout.name());
    }
}

/// int4 is served by every layout, because no layout transcodes it.
///
/// `tv_q4_h` reads stored weights; it consults neither a class table nor a
/// word map. That orthogonality is exactly what makes a mixed file possible,
/// and pinning it here keeps a future layout from accidentally claiming int4
/// as its own.
#[test]
fn int4_is_orthogonal_to_the_layout() {
    for layout in LAYOUTS {
        assert!(serves_kind(layout, CodeKind::Int4G128), "{}", layout.name());
    }
}

/// A header set is accepted whole or refused by name, before any record.
#[test]
fn the_header_set_is_gated_before_the_first_record() {
    let mixed_tetra = {
        let mut k = KindSet::of(CodeKind::Tetra);
        k.insert(CodeKind::Int4G128);
        k
    };
    let mixed_ball = {
        let mut k = KindSet::of(CodeKind::Ball);
        k.insert(CodeKind::Int4G128);
        k
    };
    // The object step 6.0 produced.
    assert!(check_kinds(FusedLayout::Tetra48, mixed_tetra).is_ok(), "tetra + int4");
    // And the one Q5 was first measured on.
    assert!(check_kinds(FusedLayout::Planes14, mixed_ball).is_ok(), "ball + int4");
    // Crossed, both ways.
    let e = check_kinds(FusedLayout::Planes14, mixed_tetra).expect_err("ball layout, tetra file");
    assert!(e.contains("Tetra"), "{e}");
    assert!(e.contains("planes14"), "{e}");
    let e = check_kinds(FusedLayout::Tetra48, mixed_ball).expect_err("tetra layout, ball file");
    assert!(e.contains("Ball"), "{e}");
    // A pure file of the wrong map is refused too — the set has one member.
    assert!(check_kinds(FusedLayout::Tetra48, KindSet::of(CodeKind::Ball)).is_err());
    assert!(check_kinds(FusedLayout::Planes14, KindSet::of(CodeKind::Tetra)).is_err());
    // And an int4-only file is accepted by every layout: nothing to transcode.
    for layout in LAYOUTS {
        assert!(check_kinds(layout, KindSet::of(CodeKind::Int4G128)).is_ok(), "{}", layout.name());
    }
}

/// The four constant tables, in the shapes the kernel declares.
///
/// Sizes first, because a table of the right content at the wrong length is a
/// read past the end on a card and nothing at all here.
#[test]
fn the_constant_tables_have_the_shapes_the_kernel_declares() {
    use llvq_llm::fused::tetra48_tables;
    let t = llvq_search::tetra::Tetra::new();
    let tb = tetra48_tables(&t);
    assert_eq!(tb.rows.len(), 4096, "two classes of 2048");
    assert_eq!(tb.rows.len() * 4, 16_384, "16 KiB, the figure the floor swept");
    assert_eq!(tb.prefixes.len(), 32, "128 bytes packed four to a u32");
    assert_eq!(tb.suffixes.len(), 32);
    assert_eq!(tb.branches.len(), 1024, "64 states x 16 branches");
    assert_eq!(tb.invnorm.len(), llvq_artifact::tetra48::TETRA48_SHELLS);

    // The byte tables carry the map's own bytes, at byte i.
    for s in 0..64usize {
        for b in 0..2usize {
            let at = 2 * s + b;
            let got = (tb.prefixes[at / 4] >> (8 * (at % 4))) as u8;
            assert_eq!(got, t.prefixes()[s][b], "prefixes[{s}][{b}]");
            let got = (tb.suffixes[at / 4] >> (8 * (at % 4))) as u8;
            assert_eq!(got, t.suffixes()[s][b], "suffixes[{s}][{b}]");
        }
        for b in 0..16usize {
            let (byte, s16) = t.branches()[s][b];
            assert_eq!(tb.branches[16 * s + b], byte as u16 | (s16 as u16) << 8);
        }
    }
}

/// `invnorm[0]` is zero, and that entry is the whole handling of the origin.
///
/// Word 0 is a legal code and `1/||y||` is a division by zero there. The
/// kernel has no branch for it; it has this table entry.
#[test]
fn the_inverse_norm_table_carries_the_origin_as_zero() {
    use llvq_llm::fused::tetra48_tables;
    let tb = tetra48_tables(&llvq_search::tetra::Tetra::new());
    assert_eq!(tb.invnorm[0], 0.0, "the origin");
    for (m, &v) in tb.invnorm.iter().enumerate().skip(1) {
        let want = (1.0f64 / ((16 * m) as f64).sqrt()) as f32;
        assert_eq!(v, want, "m = {m}");
        assert!(v > 0.0 && v.is_finite());
    }
    // Strictly decreasing: a bigger shell is a smaller scale.
    for m in 2..tb.invnorm.len() {
        assert!(tb.invnorm[m] < tb.invnorm[m - 1], "m = {m}");
    }
}
