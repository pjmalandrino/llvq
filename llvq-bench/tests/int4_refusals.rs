//! The tools that read an index as a v1 class refuse a mixed file by name.
//!
//! Twin of `tetra_refusals.rs`, one kind further. `rtbits`, `classhist`,
//! `lswap` and `radixstudy` walk a `.llvq` and file every index under one of
//! the 383 classes; `driftcheck` compares two records field by field. An
//! `Int4G128` record has no index at all — its bytes are group-affine
//! weights — so each of them would print a number about nothing, or compare
//! two things that are not indices. Every one must stop at the header, naming
//! the file and the kind, before a record is read.
//!
//! `driftcheck` had **no** kind check of any sort until this file was written:
//! it would have compared a Tetra file against a Ball one, field by field,
//! and reported a difference in the wrong units. It is an example rather than
//! a binary, so it is located beside the test executable rather than through
//! `CARGO_BIN_EXE_`.

use llvq_artifact::{
    ArtifactWriter, CodeKind, Int4Matrix, KindSet, QuantizedMatrix, INT4G128_BITS, INT4G128_GROUP,
};
use llvq_core::DIM;
use llvq_quant::quantizer::BlockCode;
use llvq_search::index::Indexer;
use std::path::PathBuf;
use std::process::Command;

const REFUSAL: &str = "no runtime layout for Int4G128 records";

/// A v5 file whose default kind is Ball and whose declared set also holds
/// `Int4G128`: one Ball matrix, one int4 matrix. The Ball record is there so
/// a tool that only looked at the *first* record would sail past.
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

fn refuses(bin: &std::path::Path, args: &[&str], path: &std::path::Path) {
    let out = Command::new(bin)
        .args(args)
        .output()
        .unwrap_or_else(|e| panic!("run {}: {e}", bin.display()));
    let stderr = String::from_utf8_lossy(&out.stderr);
    let what = bin.file_name().and_then(|s| s.to_str()).unwrap_or("the tool");
    assert!(!out.status.success(), "{what} accepted a mixed file:\n{stderr}");
    assert!(stderr.contains(REFUSAL), "{what}: the refusal must say why:\n{stderr}");
    assert!(
        stderr.contains(path.to_str().expect("utf-8 path")),
        "{what}: the refusal must name the file:\n{stderr}"
    );
    assert!(
        stderr.contains("Int4G128"),
        "{what}: the refusal must name the kind it read:\n{stderr}"
    );
}

#[test]
fn rtbits_refuses_a_mixed_file_by_name() {
    let p = mixed_file("int4-rtbits.llvq");
    refuses(
        std::path::Path::new(env!("CARGO_BIN_EXE_rtbits")),
        &[p.to_str().unwrap()],
        &p,
    );
}

#[test]
fn classhist_refuses_a_mixed_file_by_name() {
    let p = mixed_file("int4-classhist.llvq");
    refuses(
        std::path::Path::new(env!("CARGO_BIN_EXE_classhist")),
        &[p.to_str().unwrap()],
        &p,
    );
}

#[test]
fn lswap_refuses_a_mixed_file_by_name() {
    let p = mixed_file("int4-lswap.llvq");
    let out = p.with_extension("out.llvq");
    // Idempotent: the assertion on `out` is about this run, not a leftover.
    let _ = std::fs::remove_file(&out);
    refuses(
        std::path::Path::new(env!("CARGO_BIN_EXE_lswap")),
        &[p.to_str().unwrap(), out.to_str().unwrap()],
        &p,
    );
    assert!(!out.exists(), "lswap must not leave an output behind a refusal");
}

#[test]
fn radixstudy_refuses_a_mixed_file_by_name() {
    let p = mixed_file("int4-radixstudy.llvq");
    refuses(
        std::path::Path::new(env!("CARGO_BIN_EXE_radixstudy")),
        &[p.to_str().unwrap()],
        &p,
    );
}

#[test]
fn driftcheck_refuses_a_mixed_file_by_name() {
    // An example, not a binary: cargo builds it for `cargo test` and puts it
    // beside the test executable's own directory.
    let exe = std::env::current_exe().expect("test exe");
    let bin = exe
        .parent()
        .and_then(|d| d.parent())
        .expect("target/debug")
        .join("examples")
        .join("driftcheck");
    assert!(
        bin.exists(),
        "{}: the driftcheck example must be built for this test to mean anything",
        bin.display()
    );
    let p = mixed_file("int4-driftcheck.llvq");
    refuses(&bin, &[p.to_str().unwrap(), p.to_str().unwrap()], &p);
}
