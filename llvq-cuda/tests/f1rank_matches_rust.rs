//! The F1 universal-table decoder of `llvq_f1rank.cuh`, compiled by `clang++`
//! and diffed against `llvq_bench::f1::rank::decode_word` — bit for bit, not
//! within a tolerance, because both sides produce small integers.
//!
//! The pattern of `decoder_matches_rust.rs`: the **same text** NVRTC will
//! compile, through `host_shim.h`, on 10,000 random words. `f1r_decode` has no
//! warp primitive, no shared memory and no atomic, so everything that is not a
//! hardware property is checked here for nothing, on the machine that cannot
//! run the kernel. What is NOT proved here: the tile loop of `f1rank.cu`, its
//! launch geometry and its shared staging — those are the card's, and the
//! bench's dump control (`bin/f1rankfloor`) compares the same decode ON the
//! card against the same Rust reference before printing a number.
//!
//! ## What a disagreement would be
//!
//! Every field range is a power of two, so a random 48-bit word is a valid
//! label: a mismatch is never "an invalid input", it is one of the places the
//! two decoders can differ — a bit field cut one bit off, the `N0` split of the
//! middle index, the `r3` formula, a sign in one of the four progressions, or
//! the row order of the table. The failure message names the first word, its
//! fields, and both points, so the reviewer can tell which.
//!
//! ## Why the harness may not compile
//!
//! The header is written by another hand against the API this test assumes
//! (`F1rTables`, `f1r_decode(lo, hi16, tables, y[24])`, `f1r_load(row, j, lo,
//! hi16)`). Until it exists, or if it drifts, `clang++` fails — and that is
//! reported as the panic below with the compiler's own stderr, not as a
//! mysterious "no such file".

use llvq_bench::f1::rank::{branch_words, decode_word, prefix_bytes, suffix_bytes, split, RankTable, WORD_BITS};
use llvq_bench::f1::{point_to_natural, Trellis};
use llvq_core::{SplitMix64, DIM};
use std::collections::HashSet;
use std::io::Write;
use std::process::{Command, Stdio};

const MASK48: u64 = (1u64 << WORD_BITS) - 1;

/// Compile one harness. `tag` is not decoration: cargo runs the tests of this
/// file as parallel threads of one process, and a shared output path would let
/// one test execute the binary the other is still writing — an intermittent
/// failure that reads as an arithmetic fault (`llvq-llm/tests/proj_q4.rs`).
fn build_harness(source: &str, tag: &str) -> std::path::PathBuf {
    let out = std::env::temp_dir().join(format!("llvq_host_f1rank_{tag}"));
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests");
    let res = Command::new("clang++")
        .args([
            "-std=c++17",
            "-O2",
            // The decoder is integer arithmetic; the flag is kept for parity
            // with the sibling probes so a future float in the header is
            // compared under the same contraction rule as on the device.
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
            "tests/{source} does not compile as host C++ — `kernels/llvq_f1rank.cuh` is absent, \
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
    child.stdin.take().expect("stdin").write_all(fixture).expect("fixture written");
    let res = child.wait_with_output().expect("the harness exits");
    assert!(res.status.success(), "the harness failed with {}", res.status);
    assert_eq!(res.stdout.len(), want_bytes, "the harness wrote {} bytes, expected {want_bytes}", res.stdout.len());
    res.stdout
}

/// The tables as the device receives them, then the words.
fn decode_fixture(table: &RankTable, tr: &Trellis, words: &[u64]) -> Vec<u8> {
    let mut f = Vec::new();
    f.extend_from_slice(&(words.len() as u32).to_le_bytes());
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
    f
}

#[test]
fn the_cuda_f1rank_decoder_decides_what_the_rust_decoder_decides() {
    let exe = build_harness("host_f1rank.cpp", "decode");
    let (table, tr) = (RankTable::build(), Trellis::new());
    let mut rng = SplitMix64::new(0x00f1_2026_0905);
    let n = 10_000usize;
    let labels: Vec<u64> = (0..n).map(|_| rng.next() & MASK48).collect();
    let want: Vec<[i32; DIM]> = labels.iter().map(|&w| decode_word(w, &table, &tr)).collect();

    // Twice: the contract (nothing above bit 47), then the same labels with
    // random bits 48..63 — the upper half of `hi16`, which the header must
    // ignore for the device path to be independent of what `f1r_load` masks.
    let garbage: Vec<u64> = labels.iter().map(|&w| w | (rng.next() << WORD_BITS)).collect();
    for (what, words) in [("bits 48..63 zero", &labels), ("random bits 48..63, which f1r_decode must ignore", &garbage)] {
        let out = run_harness(&exe, &decode_fixture(&table, &tr, words), n * DIM);
        let got: Vec<[i32; DIM]> = out
            .chunks_exact(DIM)
            .map(|c| core::array::from_fn(|j| i32::from(c[j] as i8)))
            .collect();
        let bad: Vec<usize> = (0..n).filter(|&b| got[b] != want[b]).collect();
        if let Some(&b) = bad.first() {
            let w = split(words[b]);
            panic!(
                "{what}: {} of {n} words decode differently; first is word {b} = {:#014x}\n  fields {w:?}\n  rust   {:?}\n  kernel {:?}",
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
        for (b, y) in got.iter().enumerate() {
            assert!(
                leech.contains(&point_to_natural(y, &tr.code.order)),
                "{what}: the kernel's point for word {b} is outside Λ₂₄: {y:?}"
            );
        }
        // Two words that differ only by the gain bit decode alike by design;
        // every other pair must not.
        let distinct: HashSet<[i32; DIM]> = got.iter().copied().collect();
        let distinct_labels: HashSet<u64> = words.iter().map(|w| w & MASK48 & !(1 << 47)).collect();
        assert_eq!(distinct.len(), distinct_labels.len(), "{what}: distinct labels did not give distinct points");
    }
}

/// `f1r_load` assembles the 6-byte word of lane `j` from the two aligned u32
/// covering bytes `[6j, 6j+6)` of a row — checked at both lane parities, on
/// rows of every length the shapes of the bench produce (2560/24 = 106 and
/// 9728/24 = 405 blocks among them), against the words the row was built from.
/// The bytes after a block belong to the next one and are random, so a
/// `hi16` left unmasked fails here.
#[test]
fn f1r_load_reads_the_six_byte_word_of_every_lane() {
    let exe = build_harness("host_f1rank_load.cpp", "load");
    let mut rng = SplitMix64::new(0x00f1_10ad_2026_0905);
    for n in [1usize, 2, 3, 4, 5, 8, 106, 405] {
        let words: Vec<u64> = (0..n).map(|_| rng.next() & MASK48).collect();
        // Row-major stream: block j at byte 6j, row stride round_up(6n, 8).
        let stride = (6 * n).div_ceil(8) * 8;
        let mut bytes = vec![0u8; stride];
        for (j, &w) in words.iter().enumerate() {
            bytes[6 * j..6 * j + 6].copy_from_slice(&w.to_le_bytes()[..6]);
        }
        let row: Vec<u32> = bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();

        let mut fixture = Vec::new();
        fixture.extend_from_slice(&(n as u32).to_le_bytes());
        fixture.extend_from_slice(&(row.len() as u32).to_le_bytes());
        for &w in &row {
            fixture.extend_from_slice(&w.to_le_bytes());
        }
        let out = run_harness(&exe, &fixture, n * 8);
        let got: Vec<u32> = out.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        for (j, &w) in words.iter().enumerate() {
            let (lo, hi16) = ((w & 0xffff_ffff) as u32, (w >> 32) as u32);
            assert_eq!(
                (got[2 * j], got[2 * j + 1]),
                (lo, hi16),
                "row of {n} blocks, lane {j}: f1r_load gave (lo, hi16) = ({:#010x}, {:#06x}), the word is {w:#014x}",
                got[2 * j],
                got[2 * j + 1]
            );
        }
    }
}
