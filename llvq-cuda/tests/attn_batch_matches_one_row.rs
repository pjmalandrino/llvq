//! The batched decode attention (`kernels/attn_dec_batch.cu`) returns, for
//! every sequence of the batch, the bits of the served one-row entry called on
//! that sequence alone, and writes the same cache. Executed on the CPU by
//! `host_attn_batch.cpp` through `host_cta.h`. Per hard rule 10 a missing
//! `clang++` fails, naming the tool.

use std::io::Write;
use std::process::{Command, Stdio};

fn probe() -> std::path::PathBuf {
    let out = std::env::temp_dir().join(format!("llvq_host_attn_batch_{}", std::process::id()));
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests");
    let st = Command::new("clang++")
        .args(["-std=c++17", "-O2", "-ffp-contract=off", "-Wall", "-Wextra", "-Werror", "-pthread"])
        .arg(dir.join("host_attn_batch.cpp"))
        .arg("-o")
        .arg(&out)
        .status()
        .expect("clang++ must be on PATH: the batched attention is checked by it (rule 10)");
    assert!(st.success(), "attn_dec_batch.cu does not compile as host C++ through host_cta.h");
    out
}

fn run(exe: &std::path::Path, g: u32, n_kv: u32, cap: u32, pos: &[i32], seed: u64) {
    let mut f = Vec::new();
    for v in [g, n_kv, cap, pos.len() as u32] {
        f.extend(v.to_le_bytes());
    }
    for p in pos {
        f.extend(p.to_le_bytes());
    }
    f.extend(seed.to_le_bytes());
    let mut ch = Command::new(exe).stdin(Stdio::piped()).stdout(Stdio::piped()).spawn().expect("probe runs");
    ch.stdin.take().expect("stdin").write_all(&f).expect("fixture");
    let out = ch.wait_with_output().expect("probe exits");
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success() && text.starts_with("MATCH"), "G {g}, positions {pos:?}: {text}");
}

#[test]
#[cfg_attr(debug_assertions, ignore = "compiles C++ and runs whole CTAs, run in release")]
fn every_sequence_of_a_batch_gets_the_one_row_bits() {
    let exe = probe();
    run(&exe, 4, 2, 768, &[0, 300, 700], 7);
    run(&exe, 5, 2, 512, &[255, 256, 1, 511], 9);
    run(&exe, 4, 8, 256, &[17], 3);
    run(&exe, 4, 3, 1024, &[1000, 0, 513, 64, 255], 11);
}
