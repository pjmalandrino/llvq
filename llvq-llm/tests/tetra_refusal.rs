//! `embedq` refuses a Tetra file by name.
//!
//! The tool copies every record through `read_matrix_raw` / `push_raw` under
//! a header rewritten by `ArtifactWriter::new` — a v4 Ball header. Over a v5
//! Tetra file that would yield a Ball file of Tetra words, opened by every
//! reader without complaint and wrong. The refusal is at the header, before
//! a record is read; the evidence is the binary's exit status and its stderr,
//! on a two-block Tetra file written into cargo's test tmpdir. Portable: no
//! model, no card.

use llvq_artifact::{ArtifactWriter, CodeKind, QuantizedMatrix, TETRA_SHELL_CAP};
use llvq_core::DIM;
use llvq_quant::quantizer::BlockCode;
use llvq_search::tetra::Tetra;
use std::path::PathBuf;
use std::process::Command;

#[test]
fn embedq_refuses_a_tetra_file_by_name() {
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
    let src = dir.join("tetra-embedq.llvq");
    let dst = dir.join("tetra-embedq.out.llvq");
    {
        let f = std::fs::File::create(&src).expect("create");
        let mut w = ArtifactWriter::with_kind(std::io::BufWriter::new(f), CodeKind::Tetra, 1).expect("header");
        w.push(&m).expect("write");
        w.finish().expect("flush");
    }

    // Idempotent: the assertion on `dst` is about this run, not a leftover.
    let _ = std::fs::remove_file(&dst);
    let out = Command::new(env!("CARGO_BIN_EXE_embedq"))
        .args([src.to_str().unwrap(), dst.to_str().unwrap()])
        .output()
        .expect("run embedq");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "embedq accepted a Tetra file:\n{stderr}");
    assert!(
        stderr.contains("no runtime layout for Tetra: this reads the v1 ball's classes, and a Tetra word names none"),
        "the refusal must say why:\n{stderr}"
    );
    assert!(stderr.contains(src.to_str().unwrap()), "the refusal must name the file:\n{stderr}");
    assert!(stderr.contains("Tetra"), "the refusal must name the kind:\n{stderr}");
    assert!(!dst.exists(), "embedq must not leave an output behind a refusal");

    // And the fused loader itself, which is what `fusedrun` and the CUDA
    // path go through: refused at the header's kind, before any record is
    // read as a Ball record, on every layout that reads the ball map.
    //
    // ⚠️ This assertion changed on 2026-09-09 and the change is the point.
    // It used to demand the sentence "no runtime layout for Tetra: this reads the v1 ball's classes, and a Tetra word names none".
    // That sentence was TRUE when it was written and is now FALSE: `tetra48`
    // is the runtime layout, `llvq_artifact::tetra48` is its transcoder and
    // `tv_tetra48_h` its kernel. The four below still refuse a Tetra file, and
    // the reason is sharper than it was — they read a different map, not an
    // absent one — so the refusal now names the layout that does read it.
    for layout in [
        llvq_llm::fused::FusedLayout::Planes14,
        llvq_llm::fused::FusedLayout::Planes12x,
        llvq_llm::fused::FusedLayout::Slot32,
        llvq_llm::fused::FusedLayout::Golay70,
    ] {
        let e = llvq_llm::fused::load(src.to_str().unwrap(), layout)
            .err()
            .unwrap_or_else(|| panic!("{}: fused::load accepted a Tetra file", layout.name()));
        assert!(e.contains("Tetra"), "{}: the refusal must name the kind: {e}", layout.name());
        assert!(
            e.contains("LLVQ_FUSED_LAYOUT=tetra48"),
            "{}: the refusal must name the layout that reads it: {e}",
            layout.name()
        );
    }
}
