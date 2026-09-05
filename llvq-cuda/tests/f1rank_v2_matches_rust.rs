//! The V2 F1 decoder of `llvq_f1rank_v2.cuh` — the trellis by F₂ algebra —
//! compiled by `clang++` and diffed against `llvq_bench::f1::rank::decode_word`
//! bit for bit, then its per-block dot against an f64 reference.
//!
//! The pattern of `f1rank_matches_rust.rs`: the **same text** NVRTC will
//! compile, through `host_shim.h`, on 10,000 random words. `f1r_decode_v2_f`
//! and `f1r_dot_v2` have no warp primitive, no shared memory and no atomic, so
//! everything that is not a hardware property is checked here for nothing.
//! What is NOT proved here: the tile loop of `f1rank_v2.cu`, its launch
//! geometry and its shared staging — those are the card's, and the bench's
//! control compares `tv_f1r_v2`'s y to `tv_f1r`'s on the card before printing
//! a number.
//!
//! ## What a disagreement would be
//!
//! The header reads no small table: its trellis is twelve 24-bit immediates.
//! So beyond the failures the sibling test names (a bit field cut short, the
//! `N0` split, `r3`, a sign in `val`), a mismatch here is a wrong column, a
//! column applied to the wrong bit of the word, or a wrong packing of
//! `c1 | c2 << 8 | c3 << 16`. The harness writes the twelve immediates it was
//! compiled with, and they are compared to the pinned `LINEAR_COLUMNS` and to
//! the columns derived from the trellis BEFORE any word is decoded, so a
//! mistyped immediate is named as such rather than as "word 17 differs".
//!
//! The harness passes NULL for `prefixes`, `branches` and `suffixes`: a V2
//! that read one of them would crash the test, not silently pass.

use llvq_bench::f1::rank::{decode_word, split, RankTable, TrellisLinear, LINEAR_COLUMNS, WORD_BITS};
use llvq_bench::f1::{point_to_natural, Trellis};
use llvq_core::{SplitMix64, DIM};
use std::collections::HashSet;
use std::io::Write;
use std::process::{Command, Stdio};

const MASK48: u64 = (1u64 << WORD_BITS) - 1;

/// Words decoded, and activation blocks the dot is formed against.
const N_WORDS: usize = 10_000;
const N_X: usize = 200;

/// Compile one harness. `tag` is not decoration: cargo runs the tests of this
/// file as parallel threads of one process, and a shared output path would let
/// one test execute the binary the other is still writing — an intermittent
/// failure that reads as an arithmetic fault (`llvq-llm/tests/proj_q4.rs`).
fn build_harness(source: &str, tag: &str) -> std::path::PathBuf {
    let out = std::env::temp_dir().join(format!("llvq_host_f1rank_v2_{tag}"));
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests");
    let res = Command::new("clang++")
        .args([
            "-std=c++17",
            "-O2",
            // The dot is an `__fmaf_rn` chain, which the shim maps to
            // `std::fma`; no other contraction may be introduced, or the host
            // would round differently from the device.
            "-ffp-contract=off",
            "-Wall",
            "-Wextra",
            "-Werror",
        ])
        .arg(dir.join(source))
        .arg("-o")
        .arg(&out)
        .output()
        .expect("clang++ is on PATH");
    if !res.status.success() {
        panic!(
            "tests/{source} does not compile as host C++ — `kernels/llvq_f1rank_v2.cuh` is absent, \
             or does not match the API this harness was written against. clang++ said:\n{}",
            String::from_utf8_lossy(&res.stderr)
        );
    }
    out
}

fn run_harness(exe: &std::path::Path, fixture: &[u8], want_bytes: usize) -> Vec<u8> {
    let mut child = Command::new(exe)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("the harness runs");
    // The fixture is written from another thread: the harness writes 8 MB
    // back, and a single thread pushing the input while the output pipe fills
    // would deadlock.
    let mut stdin = child.stdin.take().expect("stdin");
    let fixture = fixture.to_vec();
    let writer = std::thread::spawn(move || stdin.write_all(&fixture).expect("fixture written"));
    let res = child.wait_with_output().expect("the harness exits");
    writer.join().expect("the writer thread");
    assert!(res.status.success(), "the harness failed with {}", res.status);
    assert_eq!(res.stdout.len(), want_bytes, "the harness wrote {} bytes, expected {want_bytes}", res.stdout.len());
    res.stdout
}

/// The rank table as the device receives it, the words, then the activations.
fn fixture(table: &RankTable, words: &[u64], x: &[f32]) -> Vec<u8> {
    let mut f = Vec::new();
    f.extend_from_slice(&(words.len() as u32).to_le_bytes());
    for &w in &table.rows {
        f.extend_from_slice(&w.to_le_bytes());
    }
    for &w in words {
        f.extend_from_slice(&w.to_le_bytes());
    }
    f.extend_from_slice(&((x.len() / DIM) as u32).to_le_bytes());
    for &v in x {
        f.extend_from_slice(&v.to_le_bytes());
    }
    f
}

/// The harness's three outputs, cut from its stdout.
struct Out {
    columns: [u32; 12],
    y: Vec<f32>,
    dots: Vec<f32>,
}

fn run(exe: &std::path::Path, table: &RankTable, words: &[u64], x: &[f32]) -> Out {
    let (n, nx) = (words.len(), x.len() / DIM);
    let want = 4 * 12 + 4 * n * DIM + 4 * n * nx;
    let out = run_harness(exe, &fixture(table, words, x), want);
    let u32s = |b: &[u8]| -> Vec<u32> { b.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect() };
    let f32s = |b: &[u8]| -> Vec<f32> { b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect() };
    let (cols, rest) = out.split_at(4 * 12);
    let (y, dots) = rest.split_at(4 * n * DIM);
    Out { columns: u32s(cols).try_into().expect("12 columns"), y: f32s(y), dots: f32s(dots) }
}

/// Uniform in [−1, 1) as f32, with a full mantissa — the products are NOT
/// exact, so the dot test exercises the rounding of the FMA chain and not
/// only its integer inputs.
fn activations(rng: &mut SplitMix64, n: usize) -> Vec<f32> {
    (0..n * DIM).map(|_| (rng.next_f64() * 2.0 - 1.0) as f32).collect()
}

#[test]
fn the_cuda_f1rank_v2_decoder_decides_what_the_rust_decoder_decides() {
    let exe = build_harness("host_f1rank_v2.cpp", "decode");
    let (table, tr) = (RankTable::build(), Trellis::new());
    let lin = TrellisLinear::derive(&tr).expect("the trellis is linear under its numbering");
    let mut rng = SplitMix64::new(0x00f1_2026_0905_0002);
    let labels: Vec<u64> = (0..N_WORDS).map(|_| rng.next() & MASK48).collect();
    let want: Vec<[i32; DIM]> = labels.iter().map(|&w| decode_word(w, &table, &tr)).collect();
    let x = activations(&mut rng, 1);

    // Twice: the contract (nothing above bit 47), then the same labels with
    // random bits 48..63 — the upper half of `hi16`, which the header must
    // ignore for the device path to be independent of what `f1r_load` masks.
    let garbage: Vec<u64> = labels.iter().map(|&w| w | (rng.next() << WORD_BITS)).collect();
    for (what, words) in [("bits 48..63 zero", &labels), ("random bits 48..63, which f1r_decode_v2_f must ignore", &garbage)] {
        let out = run(&exe, &table, words, &x);

        // The immediates first: the header's twelve columns are the pinned
        // ones and the derived ones, so a wrong constant is named here.
        assert_eq!(out.columns, LINEAR_COLUMNS, "{what}: the header's columns are not LINEAR_COLUMNS");
        assert_eq!(out.columns, lin.columns, "{what}: the header's columns are not the ones derived from the trellis");

        // Bit for bit: each value is a small integer, exactly representable,
        // and `(float)` of it on the device must be the same f32 as `as f32`
        // here. `to_bits` so that a −0.0 would not pass as 0.
        let got: Vec<[f32; DIM]> = out.y.chunks_exact(DIM).map(|c| core::array::from_fn(|j| c[j])).collect();
        let bad: Vec<usize> = (0..N_WORDS)
            .filter(|&b| (0..DIM).any(|j| got[b][j].to_bits() != (want[b][j] as f32).to_bits()))
            .collect();
        if let Some(&b) = bad.first() {
            let w = split(words[b]);
            panic!(
                "{what}: {} of {N_WORDS} words decode differently; first is word {b} = {:#014x}\n  fields {w:?}\n  rust   {:?}\n  kernel {:?}",
                bad.len(),
                words[b],
                want[b],
                got[b]
            );
        }

        // The spirit of the module's own tests, on the kernel's points rather
        // than the reference's: in Λ₂₄ under the lattice's own test, and one
        // point per word.
        let leech = llvq_core::Leech::new();
        let points: Vec<[i32; DIM]> = got.iter().map(|y| y.map(|v| v as i32)).collect();
        for (b, y) in points.iter().enumerate() {
            assert!(
                leech.contains(&point_to_natural(y, &tr.code.order)),
                "{what}: the kernel's point for word {b} is outside Λ₂₄: {y:?}"
            );
        }
        let distinct: HashSet<[i32; DIM]> = points.iter().copied().collect();
        let distinct_labels: HashSet<u64> = words.iter().map(|w| w & MASK48 & !(1 << 47)).collect();
        assert_eq!(distinct.len(), distinct_labels.len(), "{what}: distinct labels did not give distinct points");
    }
}

/// `f1r_dot_v2` against an f64 dot of the reference decode, 10,000 words ×
/// 200 activation blocks. Two errors are formed: relative to `Σ|y·x|`, the
/// sibling tests' scale (`golay70_decoder_matches_rust.rs`), which is what a
/// 24-term f32 FMA chain is bounded against; and the bench's own control,
/// `|Δ| ≤ 1e-5 · max(1, |dot|)`. Both must hold on every one of the two
/// million dots.
#[test]
fn the_cuda_f1rank_v2_dot_matches_the_f64_reference() {
    let exe = build_harness("host_f1rank_v2.cpp", "dot");
    let (table, tr) = (RankTable::build(), Trellis::new());
    let mut rng = SplitMix64::new(0x00f1_d070_2026_0905);
    let words: Vec<u64> = (0..N_WORDS).map(|_| rng.next() & MASK48).collect();
    let x = activations(&mut rng, N_X);
    let out = run(&exe, &table, &words, &x);

    let (mut worst_scaled, mut worst_control) = (0f64, 0f64);
    for (i, &w) in words.iter().enumerate() {
        let y = decode_word(w, &table, &tr);
        for k in 0..N_X {
            let xb = &x[k * DIM..(k + 1) * DIM];
            let (mut want, mut scale) = (0f64, 0f64);
            for s in 0..DIM {
                let t = y[s] as f64 * xb[s] as f64;
                want += t;
                scale += t.abs();
            }
            let got = out.dots[i * N_X + k] as f64;
            let scaled = (got - want).abs() / scale.max(1e-12);
            let control = (got - want).abs() / want.abs().max(1.0);
            worst_scaled = worst_scaled.max(scaled);
            worst_control = worst_control.max(control);
            assert!(
                scaled < 1e-5 && control < 1e-5,
                "word {i} = {w:#014x}, activation block {k}: kernel {got} against f64 {want} ({scaled:.3e}·Σ|y·x|, {control:.3e}·max(1, |dot|))"
            );
        }
    }
    eprintln!(
        "f1r_dot_v2 on {N_WORDS} × {N_X} dots: worst {worst_scaled:.3e}·Σ|y·x|, worst {worst_control:.3e}·max(1, |dot|)"
    );
}
