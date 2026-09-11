//! `embedq`, `export` and the fused loader refuse a mixed file by name.
//!
//! Twin of `tetra_refusal.rs`, one kind further. `embedq` copies every record
//! through `read_matrix_raw` / `push_raw` under a header rewritten by
//! `ArtifactWriter::new`; `export` decodes every record to a safetensors
//! directory; the fused loader transcodes every index into a runtime layout.
//! None has a path for a record that stores its weights, and each must stop at
//! the header rather than in the middle of the file — `export`'s middle would
//! already be half a directory of tensors that looks complete.
//!
//! The fused refusal is the one that has to land **off a card**. The defect
//! found on 2026-09-05 was a table gated on Linux, so nothing on a Mac could
//! see a hole in it; a refusal placed behind the device open would be the same
//! defect again. This test runs on any machine, with no model and no card.

use llvq_artifact::{
    ArtifactWriter, CodeKind, Int4Matrix, KindSet, QuantizedMatrix, INT4G128_BITS, INT4G128_GROUP,
};
use llvq_core::DIM;
use llvq_quant::quantizer::BlockCode;
use llvq_search::index::Indexer;
use std::path::PathBuf;
use std::process::Command;

fn mixed_file(name: &str) -> PathBuf {
    let ix = Indexer::new();
    let point = ix.decode(0).expect("index 0 decodes");
    let ball = QuantizedMatrix {
        name: "model.layers.0.self_attn.q_proj.weight".into(),
        d_out: 1,
        d_in: 2 * DIM,
        codes: vec![
            BlockCode { point, gain: 0 },
            BlockCode { point, gain: 1 },
        ],
        row_scales: vec![1.0],
        centroids: vec![0.7, 1.1],
        rotation_seed: None,
        shell_cap: 12,
        tail: vec![],
    };
    let int4 = Int4Matrix {
        name: "model.layers.0.self_attn.v_proj.weight".into(),
        d_out: 2,
        d_in: 256,
        bits: INT4G128_BITS,
        group: INT4G128_GROUP,
        packed: vec![0x5a; 2 * 256 / 2],
        scales: vec![0x3c00; 4],
        biases: vec![0; 4],
    };
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    std::fs::create_dir_all(&dir).expect("tmpdir");
    let path = dir.join(name);
    let f = std::fs::File::create(&path).expect("create");
    let mut w = ArtifactWriter::with_kinds(
        std::io::BufWriter::new(f),
        llvq_artifact::FIRST_KINDED_VERSION,
        2,
        CodeKind::Ball,
        KindSet::of(CodeKind::Ball).with(CodeKind::Int4G128),
    )
    .expect("header");
    w.push(&ball).expect("ball");
    w.push_int4(&int4).expect("int4");
    w.finish().expect("flush");
    path
}

#[test]
fn embedq_refuses_a_mixed_file_by_name() {
    let src = mixed_file("int4-embedq.llvq");
    let dst = src.with_extension("out.llvq");
    let _ = std::fs::remove_file(&dst);
    let out = Command::new(env!("CARGO_BIN_EXE_embedq"))
        .args([src.to_str().unwrap(), dst.to_str().unwrap()])
        .output()
        .expect("run embedq");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "embedq accepted a mixed file:\n{stderr}");
    assert!(stderr.contains(src.to_str().unwrap()), "must name the file:\n{stderr}");
    assert!(stderr.contains("Int4G128"), "must name the kind:\n{stderr}");
    assert!(!dst.exists(), "embedq must not leave an output behind a refusal");
}

#[test]
fn export_refuses_a_mixed_file_by_name() {
    let src = mixed_file("int4-export.llvq");
    let dir = src.with_extension("dir");
    let _ = std::fs::remove_dir_all(&dir);
    let out = Command::new(env!("CARGO_BIN_EXE_export"))
        .args([src.to_str().unwrap(), dir.to_str().unwrap()])
        .output()
        .expect("run export");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "export accepted a mixed file:\n{stderr}");
    assert!(stderr.contains(src.to_str().unwrap()), "must name the file:\n{stderr}");
    assert!(stderr.contains("Int4G128"), "must name the kind:\n{stderr}");
    assert!(
        !dir.exists(),
        "export must not create its output directory behind a refusal"
    );
}

#[test]
fn a_mixed_file_is_served_by_the_ball_layouts_and_a_tetra_record_is_not() {
    // 🕳️ This asserted the opposite until 2026-09-10, and it had been red
    // since step 6.6 without anyone reading past a truncated test log.
    //
    // Before 6.6 the fused path refused a mixed file outright, on every
    // layout, at the header's `KindSet`. 6.6 removed that DELIBERATELY:
    // `serves_kind` is `kind == Int4G128 || kind == lattice_kind(layout)`,
    // because an int4 record is read by `tv_q4_h`, which is orthogonal to the
    // lattice layout — that orthogonality is the whole reason a mixed file can
    // be served at all, and the served object of 2026-09-08 is one.
    //
    // So the kind gate is tested for what it now IS: it lets int4 through on
    // every layout, and it stops a lattice record whose kind the layout does
    // not read.
    use llvq_llm::fused::{check_kinds, FusedLayout};
    let ball = [
        FusedLayout::Planes14,
        FusedLayout::Planes12x,
        FusedLayout::Slot32,
        FusedLayout::Golay70,
    ];
    for layout in ball {
        check_kinds(layout, KindSet::of(CodeKind::Ball).with(CodeKind::Int4G128)).unwrap_or_else(
            |e| panic!("{}: a Ball+int4 file must be served since 6.6: {e}", layout.name()),
        );
        let e = check_kinds(layout, KindSet::of(CodeKind::Tetra).with(CodeKind::Int4G128))
            .expect_err("a Tetra record has no ball layout");
        assert!(
            e.contains("LLVQ_FUSED_LAYOUT=tetra48"),
            "{}: the refusal must send the operator to the layout that DOES read it: {e}",
            layout.name()
        );
    }
    // And the reverse: `tetra48` reads no ball record, and says where to go.
    let e = check_kinds(FusedLayout::Tetra48, KindSet::of(CodeKind::Ball))
        .expect_err("a Ball record has no tetra48 layout");
    assert!(e.contains("planes14"), "the refusal must name a ball layout: {e}");
    check_kinds(FusedLayout::Tetra48, KindSet::of(CodeKind::Tetra).with(CodeKind::Int4G128))
        .expect("the served object's own kind set");

    // The fixture itself is still refused by `load`, and NOT by kind: its
    // lattice record carries `rotation_seed: None`, and the fused path cannot
    // read a matrix quantized in the natural basis. Asserted so that nobody
    // reads this file's refusal as a kind refusal again.
    let src = mixed_file("int4-fused.llvq");
    let e = llvq_llm::fused::load(src.to_str().unwrap(), FusedLayout::Planes14)
        .err()
        .expect("the fixture is unrotated and cannot be served");
    assert!(
        e.contains("no rotation in the file"),
        "the fixture must fail on its rotation, not on its kinds: {e}"
    );
}
