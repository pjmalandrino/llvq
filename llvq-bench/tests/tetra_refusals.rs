//! The tools that read an index as a v1 class refuse a Tetra file by name.
//!
//! `rtbits`, `classhist`, `lswap` and `radixstudy` walk a `.llvq` through
//! `read_matrix_raw` and file every index under one of the 383 classes. A
//! Tetra word (format v5, kind `Tetra`) names no class, and each of them would
//! print a number about nothing. Every one must stop at the header, naming
//! the file and the reason, before a record is read. (`decbench`, `decfull`
//! and `decprofile` take no file: nothing to refuse.)
//!
//! Each tool is run as the binary it is, on a two-block Tetra file written
//! into cargo's test tmpdir — the refusal is an `assert!` in `main`, so the
//! evidence is the exit status and the message on stderr.

use llvq_artifact::{ArtifactWriter, CodeKind, QuantizedMatrix, TETRA_SHELL_CAP};
use llvq_core::DIM;
use llvq_quant::quantizer::BlockCode;
use llvq_search::tetra::Tetra;
use std::path::PathBuf;
use std::process::Command;

const REFUSAL: &str = "no runtime layout for Tetra before F1d";

/// A one-matrix v5 Tetra file, two blocks, on disk.
fn tetra_file(name: &str) -> PathBuf {
    let tetra = Tetra::new();
    let m = QuantizedMatrix {
        name: "model.layers.0.self_attn.q_proj.weight".into(),
        d_out: 1,
        d_in: 2 * DIM,
        codes: vec![
            BlockCode { point: tetra.decode(0x1234_5678_9abc), gain: 0 },
            BlockCode { point: tetra.decode(0x7210_0905_0003), gain: 1 },
        ],
        row_scales: vec![1.0],
        centroids: vec![0.7, 1.1],
        rotation_seed: None,
        shell_cap: TETRA_SHELL_CAP,
        tail: vec![],
    };
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    std::fs::create_dir_all(&dir).expect("tmpdir");
    let path = dir.join(name);
    let f = std::fs::File::create(&path).expect("create");
    let mut w = ArtifactWriter::with_kind(std::io::BufWriter::new(f), CodeKind::Tetra, 1).expect("header");
    w.push(&m).expect("write");
    w.finish().expect("flush");
    path
}

fn refuses(bin: &str, args: &[&str], path: &std::path::Path) {
    let out = Command::new(bin).args(args).output().unwrap_or_else(|e| panic!("run {bin}: {e}"));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "{bin} accepted a Tetra file:\n{stderr}");
    assert!(stderr.contains(REFUSAL), "{bin}: the refusal must say why:\n{stderr}");
    assert!(
        stderr.contains(path.to_str().expect("utf-8 path")),
        "{bin}: the refusal must name the file:\n{stderr}"
    );
    assert!(stderr.contains("Tetra"), "{bin}: the refusal must name the kind:\n{stderr}");
}

#[test]
fn rtbits_refuses_a_tetra_file_by_name() {
    let p = tetra_file("tetra-rtbits.llvq");
    refuses(env!("CARGO_BIN_EXE_rtbits"), &[p.to_str().unwrap()], &p);
}

#[test]
fn classhist_refuses_a_tetra_file_by_name() {
    let p = tetra_file("tetra-classhist.llvq");
    refuses(env!("CARGO_BIN_EXE_classhist"), &[p.to_str().unwrap()], &p);
}

#[test]
fn lswap_refuses_a_tetra_file_by_name() {
    let p = tetra_file("tetra-lswap.llvq");
    let out = p.with_extension("out.llvq");
    // Idempotent: a previous run (or a mutant of the tool) may have left the
    // output behind, and the assertions below are about THIS run.
    let _ = std::fs::remove_file(&out);
    let _ = std::fs::remove_file(format!("{}.tmp", out.display()));
    refuses(env!("CARGO_BIN_EXE_lswap"), &[p.to_str().unwrap(), out.to_str().unwrap()], &p);
    // The refusal comes before the temporary output is even created.
    assert!(!out.exists(), "lswap must not leave an output behind a refusal");
    assert!(!PathBuf::from(format!("{}.tmp", out.display())).exists(), "nor its temporary");
}

#[test]
fn radixstudy_refuses_a_tetra_file_by_name() {
    let p = tetra_file("tetra-radixstudy.llvq");
    refuses(env!("CARGO_BIN_EXE_radixstudy"), &[p.to_str().unwrap()], &p);
}
