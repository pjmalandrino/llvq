//! The V3 F1 decoder of `llvq_f1rank_v3.cuh` — values by byte tables and
//! `prmt` — compiled by `clang++` and diffed against
//! `llvq_bench::f1::rank::decode_word`: the 24 floats bit for bit, the dot
//! against the f64 reference, and the shim's `__byte_perm` against the PTX
//! ISA before either.
//!
//! The pattern of `f1rank_matches_rust.rs`: the **same text** NVRTC will
//! compile, through `host_shim.h`, on 10,000 random words. Neither
//! `f1r_dot_v3` nor `f1r_decode_v3_f` has a warp primitive, shared memory or
//! an atomic, so everything that is not a hardware property is checked here
//! for nothing. What is NOT proved here: the tile loop of `f1rank_v3.cu`, its
//! launch geometry and its shared staging — those are the card's, and the
//! bench's control (`bin/f1rankfloor`) compares the arm's `y` to `tv_f1r`'s
//! on every row before printing a number.
//!
//! ## What is new against the base test, and why
//!
//! 1. **The shim is under test too.** `llvq_f1rank.cuh` uses nothing
//!    `host_shim.h` did not already define; V3 uses `__byte_perm`, which had
//!    to be added, and a shim that got the instruction wrong would let a
//!    wrong header pass. So the first test feeds the shim 16 hand-computed
//!    cases of `prmt.b32` — identity, reverse, broadcast, the kernel's own
//!    controls, the sign-replicating msb on a 0 and on a 1 msb, the ignored
//!    upper 16 bits of the selector — and the decode tests run only once it
//!    is right.
//! 2. **Floats, bit for bit.** The header builds each value as a float
//!    without an int→float conversion; the claim is that the bits equal
//!    `(f32)` of the reference integer — `+0.0`, not `−0.0`, for zero — and
//!    that is what `to_bits` compares.
//! 3. **Ranks 5..7.** The served table holds ranks ≤ 4, so a run on it never
//!    reads the twelve table bytes beyond; a synthetic table of random ranks
//!    0..7 does, against the same `decode_word`, whose `val` is defined there.
//! 4. **Both masks.** The byte mask has a fallback without the msb mode of
//!    `prmt` (`LLVQ_F1R_V3_NO_SIGN_PRMT`); the harness is built twice and both
//!    builds must decode alike, so the fallback is live if a card ever needs it.
//! 5. **The dot.** `f1r_dot_v3` is what the arm calls. Every word against 200
//!    random `x`: within 1e-5 of the f64 dot, relative to the block's
//!    `Σ|y_s·x_s|` — the convention of `e1v_decoder_matches_rust.rs`, since a
//!    24-term f32 chain cannot be held to `|y|` under cancellation — and, the
//!    stronger statement, equal bit for bit to the same `__fmaf_rn` chain
//!    computed in Rust with `f32::mul_add`, which is the order `tv_f1r` sums in.
//!
//! ## Why the harness may not compile
//!
//! `clang++` fails if the header drifts from the API this test assumes
//! (`f1r_decode_v3_f(lo, hi16, tables, y[24])`, `f1r_dot_v3(lo, hi16, tables,
//! xb)`), and that is reported as the panic below with the compiler's own
//! stderr, not as a mysterious "no such file".

use llvq_bench::f1::rank::{
    branch_words, decode_word, prefix_bytes, rank_of, suffix_bytes, split, RankTable, MAX_RANK, WORD_BITS,
};
use llvq_bench::f1::{point_to_natural, Trellis};
use llvq_core::{SplitMix64, DIM};
use std::collections::HashSet;
use std::io::Write;
use std::process::{Command, Stdio};

const MASK48: u64 = (1u64 << WORD_BITS) - 1;
const N_WORDS: usize = 10_000;
const N_X: usize = 200;
/// Relative to the block's `Σ|y_s·x_s|`, floor 1.
const DOT_TOL: f64 = 1e-5;

/// Compile the harness. `tag` is not decoration: cargo runs the tests of this
/// file as parallel threads of one process, and a shared output path would let
/// one test execute the binary the other is still writing — an intermittent
/// failure that reads as an arithmetic fault (`llvq-llm/tests/proj_q4.rs`).
/// `defines` are passed through as `-D`.
fn build_harness(tag: &str, defines: &[&str]) -> std::path::PathBuf {
    let out = std::env::temp_dir().join(format!("llvq_host_f1rank_v3_{tag}"));
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests");
    let mut cmd = Command::new("clang++");
    cmd.args([
        "-std=c++17",
        "-O2",
        // NVRTC compiles with `--fmad=true` but every multiply-add of the
        // header is an explicit `__fmaf_rn`; the FADD that strips the bias
        // must stay a separate, exact operation on both sides, which this
        // guarantees on the host.
        "-ffp-contract=off",
        "-Wall",
        "-Wextra",
        "-Werror",
    ]);
    for d in defines {
        cmd.arg(format!("-D{d}"));
    }
    let res = cmd.arg(dir.join("host_f1rank_v3.cpp")).arg("-o").arg(&out).output().expect("clang++ is on PATH");
    if !res.status.success() {
        panic!(
            "tests/host_f1rank_v3.cpp does not compile as host C++ — `kernels/llvq_f1rank_v3.cuh` is absent, \
             or does not match the API this harness was written against. clang++ said:\n{}",
            String::from_utf8_lossy(&res.stderr)
        );
    }
    out
}

fn run_harness(exe: &std::path::Path, args: &[&str], fixture: &[u8], want_bytes: usize) -> Vec<u8> {
    let mut child = Command::new(exe)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("the harness runs");
    child.stdin.take().expect("stdin").write_all(fixture).expect("fixture written");
    let res = child.wait_with_output().expect("the harness exits");
    assert!(
        res.status.success(),
        "the harness failed with {}: {}",
        res.status,
        String::from_utf8_lossy(&res.stderr)
    );
    assert_eq!(res.stdout.len(), want_bytes, "the harness wrote {} bytes, expected {want_bytes}", res.stdout.len());
    res.stdout
}

fn le_u32s(b: &[u8]) -> Vec<u32> {
    b.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

/// PTX `prmt.b32`, default mode, by hand. `{b, a}` is the 8-byte source, byte
/// 0 the low byte of `a`; control nibble k of `s` (bits 4k..4k+3) picks result
/// byte k: low 3 bits the source byte, msb set → eight copies of that byte's
/// msb. Bits 16..31 of `s` are not read.
///
/// With a = 0x44332211 (bytes 11 22 33 44) and b = 0x88776655 (55 66 77 88):
const PRMT_CASES: [(u32, u32, u32, u32); 16] = [
    (0x4433_2211, 0x8877_6655, 0x3210, 0x4433_2211),      // identity on a
    (0x4433_2211, 0x8877_6655, 0x7654, 0x8877_6655),      // identity on b
    (0x4433_2211, 0x8877_6655, 0x0123, 0x1122_3344),      // a reversed
    (0x4433_2211, 0x8877_6655, 0x4444, 0x5555_5555),      // broadcast byte 4
    (0x4433_2211, 0x8877_6655, 0x7440, 0x8855_5511),      // the float control: a.b0, b.b0, b.b0, b.b3
    (0x4433_2211, 0x8877_6655, 0x5410, 0x6655_2211),      // low half of a, low half of b
    (0x4433_2211, 0x8877_6655, 0x8888, 0x0000_0000),      // msb: replicate the 0 msb of 0x11
    (0x4433_2211, 0x8877_6655, 0xffff, 0xffff_ffff),      // msb: replicate the 1 msb of 0x88
    (0x4433_2211, 0x8877_6655, 0xb210, 0x0033_2211),      // one replicated byte, msb 0, at the top
    (0x4433_2211, 0x8877_6655, 0xf210, 0xff33_2211),      // one replicated byte, msb 1, at the top
    (0x4433_2211, 0x8877_6655, 0xdead_3210, 0x4433_2211), // bits 16..31 of s are not read
    (0x4433_2211, 0x8877_6655, 0x0000, 0x1111_1111),      // all nibbles zero: broadcast byte 0
    (0x8000_8000, 0x0000_0000, 0xba98, 0xff00_ff00),      // the mask control on msbs 0,1,0,1
    (0x7f7f_7f7f, 0x8080_8080, 0xcb98, 0xff00_0000),      // replicate across the a/b boundary
    (0x887c_8480, 0x9074_8c78, 0x4130, 0x7884_8880),      // a table lookup: ranks 0, 3, 1, 4 of o = 0
    (0x0000_0000, 0xffff_ffff, 0x0c48, 0x00ff_ff00),      // copy and replicate of the same byte
];

#[test]
fn the_shim_byte_perm_matches_the_ptx_isa_on_sixteen_hand_computed_cases() {
    let exe = build_harness("prmt", &[]);
    let mut fixture = Vec::new();
    fixture.extend_from_slice(&(PRMT_CASES.len() as u32).to_le_bytes());
    for &(a, b, s, _) in &PRMT_CASES {
        for v in [a, b, s] {
            fixture.extend_from_slice(&v.to_le_bytes());
        }
    }
    let got = le_u32s(&run_harness(&exe, &["prmt"], &fixture, 4 * PRMT_CASES.len()));
    for (i, (&(a, b, s, want), &g)) in PRMT_CASES.iter().zip(&got).enumerate() {
        assert_eq!(
            g, want,
            "case {i}: __byte_perm({a:#010x}, {b:#010x}, {s:#x}) gave {g:#010x}, prmt.b32 gives {want:#010x}"
        );
    }
}

/// The tables as the device receives them, then the words, then the `x`.
fn fixture(table: &RankTable, tr: &Trellis, words: &[u64], xs: &[[f32; DIM]]) -> Vec<u8> {
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
    f.extend_from_slice(&(xs.len() as u32).to_le_bytes());
    for x in xs {
        for v in x {
            f.extend_from_slice(&v.to_le_bytes());
        }
    }
    f
}

/// One run of the harness: the 24 floats of every word and the dots.
fn run_decode(exe: &std::path::Path, table: &RankTable, tr: &Trellis, words: &[u64], xs: &[[f32; DIM]]) -> (Vec<[f32; DIM]>, Vec<f32>) {
    let n = words.len();
    let out = run_harness(exe, &[], &fixture(table, tr, words, xs), 4 * (n * DIM + n * xs.len()));
    let floats: Vec<f32> = out.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let y: Vec<[f32; DIM]> = floats[..n * DIM].chunks_exact(DIM).map(|c| core::array::from_fn(|j| c[j])).collect();
    (y, floats[n * DIM..].to_vec())
}

/// The 24 values of `words` under the reference, as floats — what the header
/// must reproduce bit for bit. `what` names the run in the failure message.
fn check_values(what: &str, words: &[u64], got: &[[f32; DIM]], table: &RankTable, tr: &Trellis) {
    let n = words.len();
    let want: Vec<[f32; DIM]> = words.iter().map(|&w| decode_word(w, table, tr).map(|v| v as f32)).collect();
    let bad: Vec<usize> = (0..n).filter(|&b| got[b].map(f32::to_bits) != want[b].map(f32::to_bits)).collect();
    if let Some(&b) = bad.first() {
        let w = split(words[b]);
        panic!(
            "{what}: {} of {n} words decode differently; first is word {b} = {:#014x}\n  fields {w:?}\n  rust   {:?}\n  kernel {:?}\n  kernel bits {:?}",
            bad.len(),
            words[b],
            want[b],
            got[b],
            got[b].map(f32::to_bits).map(|u| format!("{u:#010x}"))
        );
    }
}

#[test]
fn the_v3_decoder_gives_the_rust_decoder_s_values_as_floats_bit_for_bit() {
    let exe = build_harness("decode", &[]);
    let (table, tr) = (RankTable::build(), Trellis::new());
    let mut rng = SplitMix64::new(0x00f1_2026_0905_0003);
    let labels: Vec<u64> = (0..N_WORDS).map(|_| rng.next() & MASK48).collect();
    // The dot output is not read here; one `x` keeps the fixture honest.
    let one_x = [[0.5f32; DIM]];

    // Twice: the contract (nothing above bit 47), then the same labels with
    // random bits 48..63 — the upper half of `hi16`, which the header must
    // ignore for the device path to be independent of what `f1r_load` masks.
    let garbage: Vec<u64> = labels.iter().map(|&w| w | (rng.next() << WORD_BITS)).collect();
    for (what, words) in [("bits 48..63 zero", &labels), ("random bits 48..63, which f1r_decode_v3_f must ignore", &garbage)] {
        let (got, _) = run_decode(&exe, &table, &tr, words, &one_x);
        check_values(what, words, &got, &table, &tr);

        // The kernel's points, under the lattice's own test, and one point per
        // word — as the base test states it.
        let leech = llvq_core::Leech::new();
        for (b, y) in got.iter().enumerate() {
            let yi: [i32; DIM] = y.map(|v| v as i32);
            assert!(
                leech.contains(&point_to_natural(&yi, &tr.code.order)),
                "{what}: the kernel's point for word {b} is outside Λ₂₄: {y:?}"
            );
        }
        let distinct: HashSet<[u32; DIM]> = got.iter().map(|y| y.map(f32::to_bits)).collect();
        let distinct_labels: HashSet<u64> = words.iter().map(|w| w & MASK48 & !(1 << 47)).collect();
        assert_eq!(distinct.len(), distinct_labels.len(), "{what}: distinct labels did not give distinct points");
    }

    // Every table entry the served table can reach was exercised: the 20
    // pairs (o, ρ ≤ MAX_RANK), recovered from the decoded values alone.
    let (got, _) = run_decode(&exe, &table, &tr, &labels, &one_x);
    let mut seen = HashSet::new();
    for y in &got {
        for &v in y {
            let v = v as i32;
            let o = v.rem_euclid(4) as u32;
            seen.insert((o, rank_of(o, v).expect("a decoded value has a rank")));
        }
    }
    for o in 0..4u32 {
        for rho in 0..=MAX_RANK {
            assert!(seen.contains(&(o, rho)), "o = {o}, ρ = {rho} never came out of the table on {N_WORDS} words");
        }
    }
}

/// The twelve table bytes ranks 5..7 address, which the served table never
/// reaches: a table of random ranks 0..7 in every row, decoded by the same
/// `decode_word` — `val` is defined there, and the header claims the whole
/// nibble. Not a lattice check: these rows are not the table's.
#[test]
fn the_v3_tables_hold_ranks_five_to_seven_as_val_defines_them() {
    let exe = build_harness("ranks", &[]);
    let (real, tr) = (RankTable::build(), Trellis::new());
    let mut rng = SplitMix64::new(0x00f1_2026_0905_0007);
    let rows: Vec<u32> = (0..real.rows.len()).map(|_| (rng.next() as u32) & 0x7777_7777).collect();
    let table = RankTable { rows, n0_mixed: real.n0_mixed };
    let words: Vec<u64> = (0..N_WORDS).map(|_| rng.next() & MASK48).collect();
    let (got, _) = run_decode(&exe, &table, &tr, &words, &[[0.5f32; DIM]]);
    check_values("random ranks 0..7", &words, &got, &table, &tr);
    let mut seen = HashSet::new();
    for y in &got {
        for &v in y {
            let v = v as i32;
            let o = v.rem_euclid(4) as u32;
            seen.insert((o, rank_of(o, v).expect("a decoded value has a rank")));
        }
    }
    assert_eq!(seen.len(), 32, "the 32 (o, ρ ≤ 7) pairs were not all reached");
}

/// The mask without the sign-replicating `prmt`: the same decode, so the
/// fallback is live if a card contradicts the ISA on the intrinsic.
#[test]
fn the_v3_decoder_without_the_sign_replicating_prmt_decodes_alike() {
    let exe = build_harness("nosign", &["LLVQ_F1R_V3_NO_SIGN_PRMT"]);
    let (table, tr) = (RankTable::build(), Trellis::new());
    let mut rng = SplitMix64::new(0x00f1_2026_0905_0011);
    let words: Vec<u64> = (0..N_WORDS).map(|_| rng.next() & MASK48).collect();
    let (got, _) = run_decode(&exe, &table, &tr, &words, &[[0.5f32; DIM]]);
    check_values("LLVQ_F1R_V3_NO_SIGN_PRMT", &words, &got, &table, &tr);
}

#[test]
fn the_v3_dot_matches_the_f64_reference_on_two_hundred_x_and_the_f32_chain_bit_for_bit() {
    let exe = build_harness("dot", &[]);
    let (table, tr) = (RankTable::build(), Trellis::new());
    let mut rng = SplitMix64::new(0x00f1_2026_0905_00d0);
    let words: Vec<u64> = (0..N_WORDS).map(|_| rng.next() & MASK48).collect();
    // Uniform in [−1, 1): the sign changes are what makes cancellation, and
    // the tolerance's floor of 1 matters only when the products are small.
    let xs: Vec<[f32; DIM]> = (0..N_X).map(|_| core::array::from_fn(|_| (rng.next_f64() * 2.0 - 1.0) as f32)).collect();
    let (got_y, got_dot) = run_decode(&exe, &table, &tr, &words, &xs);
    check_values("dot run", &words, &got_y, &table, &tr);

    let mut worst = 0.0f64;
    let mut chain_bad = 0usize;
    for (i, &w) in words.iter().enumerate() {
        let y = decode_word(w, &table, &tr);
        for (k, x) in xs.iter().enumerate() {
            let g = got_dot[i * N_X + k];
            // The f64 reference and the scale of its terms.
            let (mut r64, mut scale) = (0.0f64, 0.0f64);
            for s in 0..DIM {
                let t = y[s] as f64 * x[s] as f64;
                r64 += t;
                scale += t.abs();
            }
            let err = (g as f64 - r64).abs() / scale.max(1.0);
            worst = worst.max(err);
            assert!(
                err <= DOT_TOL,
                "word {i} = {w:#014x}, x {k}: kernel dot {g}, f64 {r64}, relative gap {err:e} > {DOT_TOL:e}"
            );
            // The same chain, `__fmaf_rn` from zero in coordinate order, as
            // `tv_f1r` sums a block: bit for bit.
            let r32 = (0..DIM).fold(0.0f32, |a, s| (y[s] as f32).mul_add(x[s], a));
            if r32.to_bits() != g.to_bits() {
                chain_bad += 1;
            }
        }
    }
    assert_eq!(chain_bad, 0, "{chain_bad} of {} dots differ from the f32 fma chain in coordinate order", N_WORDS * N_X);
    // The gap that was seen, for the journal of whoever loosens or tightens
    // DOT_TOL: a 24-term f32 chain against f64, on this scale, is ~1e-7.
    println!("worst relative gap to f64 over {} dots: {worst:e}", N_WORDS * N_X);
}
