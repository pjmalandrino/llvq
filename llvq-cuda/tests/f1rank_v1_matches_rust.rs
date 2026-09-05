//! The V1 ("no I2F") F1 decoder of `llvq_f1rank_v1.cuh`, compiled by `clang++`
//! and diffed against `llvq_bench::f1::rank` — the 24 floats bit for bit, the
//! per-block product against an f64 sum, the chained product bit for bit
//! against the f32 FMA chain `tv_f1r` runs.
//!
//! The pattern of `f1rank_matches_rust.rs`: the **same text** NVRTC will
//! compile, through `host_shim.h`, on 10,000 random words. Nothing in the
//! V1 header is a warp primitive, shared memory or an atomic, so everything
//! that is not a hardware property is checked here for nothing. What is NOT
//! proved here: the tile loop of `f1rank_v1.cu`, its launch and its staging —
//! those are the card's, and the bench's equality control against `tv_f1r`'s
//! rows is what closes them.
//!
//! ## What each test would catch
//!
//! * the 24 floats: a wrong sign branch, a lane that crosses a byte, a
//!   spreading product that carries into a wanted bit, a bias constant off by
//!   one — every one of these gives a float that is not `(float)decode_word`;
//! * the section sweep: the same, over ALL 256 pattern bytes, both parities
//!   and every nibble value 0..15, which random words through the Golay
//!   tables (rank ≤ 4, pattern bytes of even weight) would never reach;
//! * the chain: a coordinate order that is not `tv_f1r`'s, or a float that is
//!   not exactly the integer, shows as a bit difference in the running sum.

use llvq_bench::f1::rank::{branch_words, decode_word, prefix_bytes, split, suffix_bytes, val, RankTable, WORD_BITS};
use llvq_bench::f1::Trellis;
use llvq_core::{SplitMix64, DIM};
use std::io::Write;
use std::process::{Command, Stdio};

const MASK48: u64 = (1u64 << WORD_BITS) - 1;
/// Relative to `Σ|y_s · x_s|`, the sibling tests' scale: 24 products chained
/// in f32 against an f64 sum, the gap is float order, not decode.
const TOL: f64 = 1e-5;

/// Compile the harness. `tag` is not decoration: cargo runs the tests of this
/// file as parallel threads of one process, and a shared output path would let
/// one test execute the binary the other is still writing.
fn build_harness(tag: &str) -> std::path::PathBuf {
    let out = std::env::temp_dir().join(format!("llvq_host_f1rank_v1_{tag}"));
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests");
    let res = Command::new("clang++")
        .args([
            "-std=c++17",
            "-O2",
            // The lane→float step is an FADD the device also issues as one
            // instruction; contraction off so the host cannot fold it into
            // the FMA that follows and hide a non-exact conversion.
            "-ffp-contract=off",
            "-Wall",
            "-Wextra",
            "-Werror",
        ])
        .arg(dir.join("host_f1rank_v1.cpp"))
        .arg("-o")
        .arg(&out)
        .output()
        .expect("clang++ is on PATH");
    if !res.status.success() {
        panic!(
            "tests/host_f1rank_v1.cpp does not compile as host C++ — `kernels/llvq_f1rank_v1.cuh` is absent, \
             or does not match the API this harness was written against. clang++ said:\n{}",
            String::from_utf8_lossy(&res.stderr)
        );
    }
    out
}

fn run_harness(exe: &std::path::Path, fixture: &[u8], want_floats: usize) -> Vec<f32> {
    let mut child = Command::new(exe)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("the harness runs");
    child.stdin.take().expect("stdin").write_all(fixture).expect("fixture written");
    let res = child.wait_with_output().expect("the harness exits");
    assert!(res.status.success(), "the harness failed with {}", res.status);
    assert_eq!(
        res.stdout.len(),
        4 * want_floats,
        "the harness wrote {} bytes, expected {}",
        res.stdout.len(),
        4 * want_floats
    );
    res.stdout.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

/// The tables as the device receives them, then the words, the activations
/// and the section triples.
fn fixture(table: &RankTable, tr: &Trellis, words: &[u64], x: &[f32], sections: &[(u32, u32, u32)]) -> Vec<u8> {
    assert_eq!(x.len() % DIM, 0);
    let mut f = Vec::new();
    for v in [words.len() as u32, (x.len() / DIM) as u32, sections.len() as u32] {
        f.extend_from_slice(&v.to_le_bytes());
    }
    for &w in &table.rows {
        f.extend_from_slice(&w.to_le_bytes());
    }
    f.extend_from_slice(&prefix_bytes(tr));
    for &w in &branch_words(tr) {
        f.extend_from_slice(&w.to_le_bytes());
    }
    f.extend_from_slice(&suffix_bytes(tr));
    for &w in words {
        f.extend_from_slice(&w.to_le_bytes());
    }
    for &v in x {
        f.extend_from_slice(&v.to_le_bytes());
    }
    for &(p, c, row) in sections {
        for v in [p, c, row] {
            f.extend_from_slice(&v.to_le_bytes());
        }
    }
    f
}

/// Output layout of the harness, in floats: y, dot, chain, sections.
fn out_len(n: usize, nx: usize, nsec: usize) -> usize {
    n * DIM + 2 * nx * n + nsec * 8
}

#[test]
fn the_v1_decoder_returns_the_reference_points_as_floats_bit_for_bit() {
    let exe = build_harness("decode");
    let (table, tr) = (RankTable::build(), Trellis::new());
    let mut rng = SplitMix64::new(0x00f1_2026_0905_0001);
    let n = 10_000usize;
    let labels: Vec<u64> = (0..n).map(|_| rng.next() & MASK48).collect();
    let want: Vec<[i32; DIM]> = labels.iter().map(|&w| decode_word(w, &table, &tr)).collect();

    // Twice: the contract (nothing above bit 47), then the same labels with
    // random bits 48..63 — the upper half of `hi16`, which the header must
    // ignore for the device path to be independent of what `f1r_load` masks.
    let garbage: Vec<u64> = labels.iter().map(|&w| w | (rng.next() << WORD_BITS)).collect();
    for (what, words) in [("bits 48..63 zero", &labels), ("random bits 48..63, which f1r_v1_lanes must ignore", &garbage)] {
        let out = run_harness(&exe, &fixture(&table, &tr, words, &[], &[]), out_len(n, 0, 0));
        let got: Vec<[u32; DIM]> = out.chunks_exact(DIM).map(|c| core::array::from_fn(|j| c[j].to_bits())).collect();
        let bad: Vec<usize> = (0..n).filter(|&b| got[b] != want[b].map(|v| (v as f32).to_bits())).collect();
        if let Some(&b) = bad.first() {
            let w = split(words[b]);
            panic!(
                "{what}: {} of {n} words decode differently; first is word {b} = {:#014x}\n  fields {w:?}\n  rust   {:?}\n  kernel {:?}",
                bad.len(),
                words[b],
                want[b],
                got[b].map(f32::from_bits)
            );
        }
        // Bit for bit includes the sign of zero: +0.0f, as `(float)0`.
        assert!(out.iter().all(|v| v.to_bits() != (-0.0f32).to_bits()), "{what}: a −0.0f reached the output");
    }
}

/// Every pattern byte, both parities, every nibble value — the section
/// arithmetic alone, against `val(p + 2·c_j, ρ_j)`. Rows: the sixteen
/// constant rows (all ranks equal), then 400 with independent nibbles in
/// 0..16, per (p, c): 2 × 256 × 416 sections. The table never exceeds rank
/// 4 and its pattern bytes have even weight, so only this sweep reaches the
/// carries of the spreading products and the lane bounds at ρ = 15.
#[test]
fn every_pattern_byte_parity_and_nibble_decodes_as_the_reference_val() {
    let exe = build_harness("sections");
    let (table, tr) = (RankTable::build(), Trellis::new());
    let mut rng = SplitMix64::new(0x00f1_2026_0905_0002);
    let mut sections = Vec::new();
    for p in 0..2u32 {
        for c in 0..256u32 {
            for rho in 0..16u32 {
                sections.push((p, c, rho * 0x1111_1111));
            }
            for _ in 0..400 {
                sections.push((p, c, rng.next() as u32));
            }
        }
    }
    let out = run_harness(&exe, &fixture(&table, &tr, &[], &[], &sections), out_len(0, 0, sections.len()));
    for (i, &(p, c, row)) in sections.iter().enumerate() {
        let got = &out[8 * i..8 * i + 8];
        let want: [f32; 8] = core::array::from_fn(|j| val(p + 2 * ((c >> j) & 1), (row >> (4 * j)) & 15) as f32);
        assert!(
            got.iter().zip(&want).all(|(g, w)| g.to_bits() == w.to_bits()),
            "p={p} c={c:#04x} row={row:#010x}: kernel {got:?}, rust {want:?}"
        );
    }
}

/// The per-block product against an f64 sum on 200 random activations, and
/// the chained product bit for bit against the f32 `mul_add` chain in
/// `tv_f1r`'s order — `acc = fma((float)y_s, x_s, acc)`, s = 0..23, word
/// after word — which is what makes the arm's rows equal to `tv_f1r`'s.
#[test]
fn the_v1_dot_matches_the_f64_sum_and_the_chain_matches_tv_f1r_bit_for_bit() {
    let exe = build_harness("dot");
    let (table, tr) = (RankTable::build(), Trellis::new());
    let mut rng = SplitMix64::new(0x00f1_2026_0905_0003);
    let (n, nx) = (10_000usize, 200usize);
    let words: Vec<u64> = (0..n).map(|_| rng.next() & MASK48).collect();
    let x: Vec<f32> = (0..nx * DIM).map(|_| rng.next_gaussian() as f32).collect();
    let y: Vec<[i32; DIM]> = words.iter().map(|&w| decode_word(w, &table, &tr)).collect();

    let out = run_harness(&exe, &fixture(&table, &tr, &words, &x, &[]), out_len(n, nx, 0));
    let dot = &out[n * DIM..n * DIM + nx * n];
    let chain = &out[n * DIM + nx * n..];

    let mut worst = 0.0f64;
    for m in 0..nx {
        let xb = &x[m * DIM..(m + 1) * DIM];
        let mut acc = 0.0f32;
        for i in 0..n {
            let (mut want, mut scale) = (0.0f64, 0.0f64);
            for s in 0..DIM {
                let t = f64::from(y[i][s]) * f64::from(xb[s]);
                want += t;
                scale += t.abs();
            }
            let rel = (f64::from(dot[m * n + i]) - want).abs() / scale.max(1.0);
            worst = worst.max(rel);
            assert!(rel <= TOL, "x {m}, word {i}: dot {} against f64 {want} ({rel:.3e} of Σ|terms|)", dot[m * n + i]);
            for s in 0..DIM {
                acc = (y[i][s] as f32).mul_add(xb[s], acc);
            }
            assert_eq!(
                chain[m * n + i].to_bits(),
                acc.to_bits(),
                "x {m}, after word {i}: chained {} is not tv_f1r's chain {acc}",
                chain[m * n + i]
            );
        }
    }
    eprintln!("worst per-block dot error: {worst:.3e} of Σ|terms| (tolerance {TOL:.0e})");
}
