//! Control 0 of the Tetra 4B prereg: has the quantization path moved since the
//! published file was written?
//!
//! `cargo run --release -p llvq-bench --example driftcheck -- <fresh.llvq> <published.bin>`
//!
//! The published 4B was encoded on 2026-08-03; twelve commits have touched
//! `llvq-quant`, `llvq-core`, `llvq-search`'s indexing modules and
//! `llvq-llm/src/calib.rs` since (*measured*, `git log`). None is supposed to
//! change what `leech1c12` writes — M1 and M2's knobs default to off, design C
//! is a flag, the English pass was proved identical on 127 files of 128 — but
//! "supposed to" is not a control, and the operator's decision of 2026-09-06
//! is to read Tetra against the published file rather than re-encode a witness
//! for four hours. That decision rests entirely on this comparison.
//!
//! So one transformer block is re-encoded today with the same command and the
//! same corpus, and its records are held to the published file's, field by
//! field. Bit equality or nothing: a run that agrees to a few ulps has moved.
//!
//! What this proves: the ENCODER has not moved. What it does not prove: that
//! the evaluation harness has not moved — that is the card's replay of the
//! published file, in the same job as Tetra.

use llvq_artifact::{read_header, read_matrix_raw, RawMatrix};

fn read_records(path: &str, want: usize) -> (u32, Vec<RawMatrix>) {
    let f = std::fs::File::open(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let mut r = std::io::BufReader::with_capacity(1 << 22, f);
    let head = read_header(&mut r).unwrap_or_else(|e| panic!("{path}: {e}"));
    // The comparison below reads `indices` and `gains` field by field, which
    // only means anything for a v1 ball index: a Tetra word names no class and
    // an int4 record has no index at all. There was no check here at all until
    // the third kind was added, and a Tetra file would have been compared
    // against a Ball one field by field, reporting a difference in the wrong
    // units.
    if let Err(e) = llvq_artifact::runtime::require_ball_kinds(head.kinds(), "driftcheck") {
        panic!(
            "{path}: a {} file (format v{}); driftcheck compares v1 ball records — {e}",
            head.kinds(),
            head.version
        );
    }
    let mut out = Vec::new();
    for _ in 0..head.matrices.min(want as u32) {
        out.push(read_matrix_raw(&mut r, head.version).unwrap_or_else(|e| panic!("{path}: {e}")));
    }
    (head.version, out)
}

/// Every field of a record, compared exactly. Returns the failures.
fn diff(a: &RawMatrix, b: &RawMatrix) -> Vec<String> {
    let mut bad = Vec::new();
    for (what, ok) in [
        ("d_out", a.d_out == b.d_out),
        ("d_in", a.d_in == b.d_in),
        ("kind", a.kind == b.kind),
        ("shell_cap", a.shell_cap == b.shell_cap),
        ("rotation_seed", a.rotation_seed == b.rotation_seed),
        ("centroids", a.centroids == b.centroids),
        ("row_scales", a.row_scales == b.row_scales),
    ] {
        if !ok {
            bad.push(what.to_string());
        }
    }
    if a.tail.len() != b.tail.len() {
        bad.push("tail length".into());
    } else if a.tail != b.tail {
        // The tail is raw weights of the ROTATED matrix, kept exactly. How far
        // apart they are separates two very different failures: a rotation
        // that changed (relative differences of order one) from arithmetic
        // that reassociated (relative differences of order 1e-15).
        // Three buckets, because they name three different failures: an exact
        // negation is a sign that flipped, an equal value is untouched, and
        // anything else is arithmetic that moved.
        let (mut neg, mut same, mut other) = (0usize, 0usize, 0usize);
        let mut worst_other = 0f64;
        for (x, y) in a.tail.iter().zip(&b.tail) {
            if x == y {
                same += 1;
            } else if *x == -*y {
                neg += 1;
            } else {
                other += 1;
                worst_other = worst_other.max((x - y).abs() / x.abs().max(y.abs()).max(1e-30));
            }
        }
        // The magnitude separates the two candidates: values of the same order
        // as the weights mean two unrelated matrices (different rotation, or
        // different Hessians feeding GPTQ's compensation); values orders of
        // magnitude smaller mean the same computation reassociated.
        let n = a.tail.len() as f64;
        let mean_abs = a.tail.iter().map(|v| v.abs()).sum::<f64>() / n;
        let mean_diff = a.tail.iter().zip(&b.tail).map(|(x, y)| (x - y).abs()).sum::<f64>() / n;
        bad.push(format!(
            "tail ({same} equal, {neg} negated, {other} other; mean |w| {mean_abs:.3e}, \
             mean |Δ| {mean_diff:.3e}, ratio {:.3})",
            mean_diff / mean_abs
        ));
    }
    if a.gains != b.gains {
        let n = a.gains.iter().zip(&b.gains).filter(|(x, y)| x != y).count();
        bad.push(format!("{n} of {} gains", a.gains.len()));
    }
    if a.indices.len() != b.indices.len() {
        bad.push("index count".into());
    } else {
        let n = a.indices.iter().zip(&b.indices).filter(|(x, y)| x != y).count();
        if n > 0 {
            bad.push(format!("{n} of {} indices", a.indices.len()));
        }
    }
    bad
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (fresh, published) = (&args[1], &args[2]);

    // The fresh run holds one transformer block: seven matrices.
    let (vf, f) = read_records(fresh, 7);
    let (vp, p) = read_records(published, 7);
    println!("fresh     {fresh}: v{vf}, {} records", f.len());
    println!("published {published}: v{vp}, {} records", p.len());

    let mut failed = 0usize;
    let mut weights = 0usize;
    for a in &f {
        let Some(b) = p.iter().find(|m| m.name == a.name) else {
            println!("  {}: absent from the published file", a.name);
            failed += 1;
            continue;
        };
        let bad = diff(a, b);
        weights += a.d_out * a.d_in;
        if bad.is_empty() {
            println!("  {:<44} identical  ({} × {})", a.name, a.d_out, a.d_in);
        } else {
            println!("  {:<44} DIFFERS on {}", a.name, bad.join(", "));
            failed += 1;
        }
    }

    println!();
    if failed == 0 {
        println!(
            "CONTROL 0 PASSES: {} matrices, {weights} weights, every field identical.\n\
             The quantization path has not moved since the published file was written, so its\n\
             perplexity and MMLU are a witness Tetra can be read against.",
            f.len()
        );
    } else {
        println!(
            "CONTROL 0 FAILS: {failed} of {} matrices differ. The encoder has moved since\n\
             2026-08-03. No Tetra number is published until the operator has decided between\n\
             re-running the witness and characterizing the drift.",
            f.len()
        );
        std::process::exit(1);
    }
}
