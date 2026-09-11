//! The served Tetra decode of `llvq_tetra48.cuh` — the v3 word decode plus the
//! gain bit, the magnitude, the trio permutation and the origin — compiled by
//! `clang++` and diffed against the two functions the rest of the repository
//! uses: `llvq_search::tetra::Tetra::decode` and
//! `llvq_quant::reconstruct_shape_gain`.
//!
//! The pattern of `f1rank_v3_matches_rust.rs`: the **same text** NVRTC will
//! compile, through `host_shim.h`, on random words. Nothing in
//! `llvq_tetra48.cuh` is a warp primitive, shared memory or an atomic, so
//! everything that is not a hardware property is checked here for nothing.
//!
//! ## Why this file exists, and what it is guarding
//!
//! Three of the four things this header adds to the v3 floor fail **silently**:
//!
//! 1. **The permutation.** `f1r_dot_v3` dots in trio order; the activation is
//!    staged in natural order. Getting it wrong is a wrong model that runs at
//!    full speed, and the floor's own check (`bin/f1rankfloor`) compares trio
//!    against trio, so it structurally cannot see it. Test 1 pins the header's
//!    literal against `Tetra::order()` by parsing the header, and test 2 pins
//!    the decode against `Tetra::decode`, which speaks natural order.
//! 2. **The `__dp4a` sign.** The quads carry `val + 128`; the header reaches
//!    the signed reading by XOR with `0x80`. An unsigned dot would return
//!    `Σ (val+128)²` — always positive, wrong by a factor of order 170, and
//!    invisible to any sign or finiteness check. Test 3 compares `n2` against
//!    the reference's own `Σ y²` as **integers**, where there is no tolerance
//!    to hide in.
//! 3. **The origin.** Word 0 is a legal code (`quantizer.rs:770`) and
//!    `1/‖y‖` is a division by zero there. Test 4 puts word 0 in the fixture
//!    explicitly; `invnorm[0] = 0` is what must carry it.
//!
//! The fourth, the gain bit, is the one that fails loudly — but only if
//! something reads it, so test 5 sweeps both values of bit 47 on the same
//! label and requires the ratio to be exactly the ratio of the centroids.
//!
//! ## The three levels of the comparison
//!
//! Level 1 is **exact**: with both scales neutralised (`gscale = [1, 1]`,
//! `invnorm` all ones) the header must reproduce `Tetra::decode` cast to f32,
//! bit for bit, in natural order. That is the decode and the permutation, with
//! no float arithmetic left to excuse a difference.
//!
//! Level 2 is exact too, on integers: `tetra48_n2` against `Σ y²`.
//!
//! Level 3 carries the real scales and is held to a relative tolerance: the
//! reference multiplies in f64 and casts once, the header multiplies twice in
//! f32, so the two associations differ by construction and a bit-for-bit claim
//! would be false. What is pinned instead is the **dot**, which is what the
//! kernel actually computes, against the same `f32::mul_add` chain in Rust in
//! the header's own order — and that one is bit for bit.
//!
//! ## Why the harness may not compile
//!
//! `clang++` fails if the header drifts from the API this test assumes, and
//! that is reported as the panic below with the compiler's own stderr. Per
//! hard rule 10 a missing `clang++` is a **failure that names the tool**, never
//! a skip: a test that quietly does not run is worse than no test.

use llvq_core::{SplitMix64, DIM};
use llvq_quant::quantizer::{reconstruct_shape_gain, BlockCode};
use llvq_search::tetra::{Tetra, WORD_MASK};
use std::io::Write;
use std::process::{Command, Stdio};

const N_WORDS: usize = 4_000;
/// Activations in the fixture. **37, not 32, and not a multiple of any of
/// [`ROW_SWEEP`].**
///
/// 32 divides 4, 8 and 16, so every chunk was full and the fold at
/// `r < n_rows ? r : 0` — the line that keeps a short chunk's reads in bounds
/// — was never taken on any machine. 37 leaves a tail of 1 at R = 4 and of 5
/// at R = 8 and 16, which is the shape half the census prompts end on.
const N_X: usize = 37;
const SHELLS: usize = 32;
/// The bound `llvq-bench/examples/tetrashell.rs` derives over the whole table.
const MAX_SHELL: u32 = 27;
/// Relative to the block's `Σ|w_j·x_j|`, floor 1 — the convention of
/// `e1v_decoder_matches_rust.rs`: a 24-term f32 chain cannot be held to `|y|`
/// under cancellation.
const DOT_TOL: f64 = 1e-5;

/// Compile the harness. `tag` keeps two parallel tests from executing a binary
/// the other is still writing — an intermittent failure that reads as an
/// arithmetic fault (`llvq-llm/tests/proj_q4.rs`).
fn build_harness(tag: &str) -> std::path::PathBuf {
    let out = std::env::temp_dir().join(format!("llvq_host_tetra48_{tag}"));
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests");
    let res = Command::new("clang++")
        .args([
            "-std=c++17",
            "-O2",
            // NVRTC compiles with `--fmad=true`, but every multiply-add of the
            // header is an explicit `__fmaf_rn`; the other operations must stay
            // separate and exact on both sides, which this guarantees here.
            "-ffp-contract=off",
            "-Wall",
            "-Wextra",
            "-Werror",
        ])
        .arg(dir.join("host_tetra48.cpp"))
        .arg("-o")
        .arg(&out)
        .output()
        .expect("clang++ is on PATH — hard rule 10: this test fails, it does not skip");
    assert!(
        res.status.success(),
        "clang++ refused tests/host_tetra48.cpp:\n{}",
        String::from_utf8_lossy(&res.stderr)
    );
    out
}

/// The header's own tables, packed the way `f1rankfloor` uploads them: the two
/// byte tables as bytes, the branches as `byte | s16 << 8`.
struct Tables {
    rows: Vec<u32>,
    prefixes: Vec<u8>,
    branches: Vec<u16>,
    suffixes: Vec<u8>,
}

fn tables(t: &Tetra) -> Tables {
    let mut prefixes = Vec::with_capacity(128);
    let mut suffixes = Vec::with_capacity(128);
    for s in 0..64usize {
        prefixes.extend_from_slice(&t.prefixes()[s]);
        suffixes.extend_from_slice(&t.suffixes()[s]);
    }
    let mut branches = Vec::with_capacity(1024);
    for s in 0..64usize {
        for b in 0..16usize {
            let (byte, s16) = t.branches()[s][b];
            branches.push(byte as u16 | (s16 as u16) << 8);
        }
    }
    Tables { rows: t.rows().to_vec(), prefixes, branches, suffixes }
}

/// What the harness returns for one fixture.
struct Out {
    y: Vec<f32>,
    dot: Vec<f32>,
    n2: Vec<u32>,
    /// `tetra48_dot_rows<R>` accumulated over EVERY word, `R` activation rows
    /// a call, staged at a padded stride — one `nx`-long block per R in
    /// [`ROW_SWEEP`], in that order. The tail rows of the last chunk fold onto
    /// activation 0 and the harness drops them, so every block is `nx` long
    /// and all three answer to ONE reference.
    dot_rows: Vec<Vec<f32>>,
}

fn run(bin: &std::path::Path, tb: &Tables, gscale: [f32; 2], invnorm: &[f32; SHELLS], words: &[u64], x: &[f32]) -> Out {
    let nx = x.len() / DIM;
    let mut fx: Vec<u8> = Vec::new();
    fx.extend_from_slice(&(words.len() as u32).to_le_bytes());
    for v in &tb.rows {
        fx.extend_from_slice(&v.to_le_bytes());
    }
    fx.extend_from_slice(&tb.prefixes);
    for v in &tb.branches {
        fx.extend_from_slice(&v.to_le_bytes());
    }
    fx.extend_from_slice(&tb.suffixes);
    for v in gscale {
        fx.extend_from_slice(&v.to_le_bytes());
    }
    for v in invnorm {
        fx.extend_from_slice(&v.to_le_bytes());
    }
    for w in words {
        fx.extend_from_slice(&w.to_le_bytes());
    }
    fx.extend_from_slice(&(nx as u32).to_le_bytes());
    for v in x {
        fx.extend_from_slice(&v.to_le_bytes());
    }

    let mut child = Command::new(bin).stdin(Stdio::piped()).stdout(Stdio::piped()).spawn().expect("harness runs");
    child.stdin.take().expect("stdin").write_all(&fx).expect("fixture written");
    let res = child.wait_with_output().expect("harness exits");
    assert!(res.status.success(), "harness failed: {}", String::from_utf8_lossy(&res.stderr));

    let n = words.len();
    let (ny, nd) = (n * DIM, n * nx);
    let nr = nx * ROW_SWEEP.len();
    assert_eq!(
        res.stdout.len(),
        4 * (ny + nd + n + nr),
        "harness wrote the wrong number of bytes"
    );
    let f32s = |b: &[u8]| -> Vec<f32> { b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect() };
    let u32s = |b: &[u8]| -> Vec<u32> { b.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect() };
    Out {
        y: f32s(&res.stdout[..4 * ny]),
        dot: f32s(&res.stdout[4 * ny..4 * (ny + nd)]),
        n2: u32s(&res.stdout[4 * (ny + nd)..4 * (ny + nd + n)]),
        dot_rows: f32s(&res.stdout[4 * (ny + nd + n)..])
            .chunks_exact(nx)
            .map(<[f32]>::to_vec)
            .collect(),
    }
}

/// The row counts a batched call can carry, swept by the harness.
///
/// The bound is shared memory and it bounds the PRODUCT `rows × tile`, not
/// either factor: 4×128, 8×64 and 16×32 all stage 49,152 bytes, the per-block
/// allowance. Since 2026-09-11 the prefill kernel has its own `TETRA48_TILE`,
/// so quadrupling the rows costs no shared memory and no decode number — and
/// four times fewer passes over the weight stream is the whole cost of a
/// prefill.
///
/// `tetra48_dot_rows<R>` is a template, and R = 8 and 16 had been instantiated
/// NOWHERE — not on a machine, not on a card — until this sweep ran them.
const ROW_SWEEP: [usize; 3] = [4, 8, 16];

/// Random 48-bit words, with the origin and both gain bits over one label
/// forced in: every 48-bit value is a label, so a uniform draw is a fair sweep
/// of the map, but the two cases that break the arithmetic are not uniform.
fn words(seed: u64) -> Vec<u64> {
    let mut rng = SplitMix64::new(seed);
    let mut w: Vec<u64> = vec![
        0,                    // the origin: m = 0, the division by zero
        1u64 << 47,           // the origin carrying a gain bit
        WORD_MASK,            // every field at its maximum
        WORD_MASK & !(1 << 47),
    ];
    while w.len() < N_WORDS {
        // The upper half above bit 47 is deliberately dirty: the header reads
        // `hi16` and must mask, not trust.
        let r = rng.next();
        w.push((r & WORD_MASK) | (r & !WORD_MASK));
    }
    w
}

fn activations(seed: u64, n: usize) -> Vec<f32> {
    let mut rng = SplitMix64::new(seed);
    (0..n * DIM).map(|_| ((rng.next() >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) as f32).collect()
}

fn ones() -> [f32; SHELLS] {
    [1.0f32; SHELLS]
}

/// `1/√(16 m)`, entry 0 zero — the origin, reconstructed without a branch.
fn invnorm_table() -> [f32; SHELLS] {
    let mut t = [0.0f32; SHELLS];
    for (m, e) in t.iter_mut().enumerate().skip(1) {
        *e = (1.0f64 / ((16 * m) as f64).sqrt()) as f32;
    }
    t
}

/// The header's `TETRA48_ORDER`, read out of the header itself.
///
/// Hardcoding the permutation twice and comparing the two copies proves
/// nothing. This parses the literal the compiler will use, so a drift between
/// the kernel and `Tetra::order()` fails here rather than in a $8 job.
fn header_order() -> Vec<u32> {
    let src = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels/llvq_tetra48.cuh"),
    )
    .expect("llvq_tetra48.cuh is readable");
    let head = src.split("TETRA48_ORDER[24] = {").nth(1).expect("the literal is spelled as the test expects");
    let body = head.split('}').next().expect("the literal closes");
    body.split(',').filter_map(|s| s.trim().parse::<u32>().ok()).collect()
}

/// Level 1: the permutation and the decode, exactly.
///
/// Both scales neutralised, so no float arithmetic separates the two sides:
/// the header must produce `(f32)` of `Tetra::decode`, bit for bit, in natural
/// order. `+0.0` and not `−0.0` for a zero coordinate is part of the claim.
#[test]
fn the_decode_and_the_permutation_are_exact() {
    let t = Tetra::new();
    assert_eq!(header_order(), t.order().to_vec(), "TETRA48_ORDER has drifted from Tetra::order()");

    let bin = build_harness("decode");
    let tb = tables(&t);
    let w = words(0x7e_42a4_0001);
    let out = run(&bin, &tb, [1.0, 1.0], &ones(), &w, &activations(0x1, 1));

    for (i, &word) in w.iter().enumerate() {
        let want = t.decode(word);
        for (j, (&got, &w)) in out.y[i * DIM..(i + 1) * DIM].iter().zip(want.iter()).enumerate() {
            assert_eq!(
                got.to_bits(),
                (w as f32).to_bits(),
                "word {word:#014x} coordinate {j}: {got} against {w}"
            );
        }
    }
}

/// Level 2: `tetra48_n2` is `Σ y²`, on integers, where nothing hides.
///
/// This is the `__dp4a` sign. It also pins the two facts the header's
/// magnitude rests on: the sum is a multiple of 16 (it is `16 m` by the
/// definition of the shell), and `m` never leaves the 32-entry table.
#[test]
fn the_shell_sum_is_exact_and_within_the_table() {
    let t = Tetra::new();
    let bin = build_harness("n2");
    let tb = tables(&t);
    let w = words(0x7e_42a4_0002);
    let out = run(&bin, &tb, [1.0, 1.0], &ones(), &w, &activations(0x2, 1));

    let mut seen = 0u32;
    for (i, &word) in w.iter().enumerate() {
        let y = t.decode(word);
        let want: u32 = y.iter().map(|&v| (v * v) as u32).sum();
        assert_eq!(out.n2[i], want, "word {word:#014x}: Σy² by __dp4a");
        assert_eq!(want % 16, 0, "word {word:#014x}: ‖y‖² = {want} is not 16 m");
        assert!(want / 16 <= MAX_SHELL, "word {word:#014x}: shell {} past the derived bound", want / 16);
        seen = seen.max(want / 16);
    }
    // A sweep that only ever saw small shells would pass the bound vacuously.
    assert!(seen >= 8, "the draw never reached shell 8; the bound is untested");
}

/// Level 3: the served value against `reconstruct_shape_gain`, the function
/// `decode_matrix` uses — including the origin, which is the whole reason
/// `invnorm[0]` is zero rather than infinite.
#[test]
fn the_served_value_matches_the_reconstruction() {
    let t = Tetra::new();
    let bin = build_harness("value");
    let tb = tables(&t);
    let w = words(0x7e_42a4_0003);
    // Two centroids well apart, so a swapped gain bit cannot pass as noise.
    let centroids = [0.481_562_5f64, 1.372_25f64];
    let gscale = [centroids[0] as f32, centroids[1] as f32];
    let out = run(&bin, &tb, gscale, &invnorm_table(), &w, &activations(0x3, 1));

    let mut ref_block = [0.0f64; DIM];
    for (i, &word) in w.iter().enumerate() {
        let code = BlockCode { point: t.decode(word), gain: ((word >> 47) & 1) as u32 };
        // `row_scale = 1`: the kernel folds the row scale once at the end of a
        // row, so it is not this function's business.
        reconstruct_shape_gain(&code, &centroids, 1.0, &mut ref_block);
        for (j, (&g, &want)) in out.y[i * DIM..(i + 1) * DIM].iter().zip(ref_block.iter()).enumerate() {
            let got = g as f64;
            let tol = want.abs().max(1.0) * 1e-6;
            assert!(
                (got - want).abs() <= tol,
                "word {word:#014x} coordinate {j}: {got} against {want}"
            );
        }
        if word & !(1u64 << 47) == 0 {
            assert!(ref_block.iter().all(|&v| v == 0.0), "the origin is not a zero block in the reference");
            assert!(out.y[i * DIM..(i + 1) * DIM].iter().all(|&v| v == 0.0), "the origin decoded non-zero");
        }
    }
}

/// Level 5: four rows a call is the same answer as four calls, BIT FOR BIT.
///
/// The batched path exists to read the weight stream R times less on a prompt
/// — 776 GB a question becomes 194 at R = 4 — and it may not change one bit of
/// the answer while doing it. That is a requirement, not a consequence: the
/// obvious optimisation is to hoist `gscale[g] * invnorm[m]` out of the row
/// loop, and it is wrong. `tetra48_dot` returns `(acc · g) · inv`, and
/// `acc · (g · inv)` differs by one ULP on roughly one word in a thousand.
/// This repository has already paid for that association once.
///
/// Both routes run on the SAME fixture inside the same harness, so a
/// disagreement is a disagreement between two implementations and not a shared
/// error — the argument the file header makes for the whole test.
#[test]
fn a_batched_call_is_its_rows_one_at_a_time_bit_for_bit() {
    let t = Tetra::new();
    let bin = build_harness("rows");
    let tb = tables(&t);
    let w = words(0x7e_42a4_0004);
    let gscale = [0.481_562_5f32, 1.372_25f32];
    let inv = invnorm_table();
    let x = activations(0x4, N_X);
    let out = run(&bin, &tb, gscale, &inv, &w, &x);

    assert_eq!(out.dot_rows.len(), ROW_SWEEP.len(), "one block a row count");
    for (block, rows) in out.dot_rows.iter().zip(ROW_SWEEP) {
        assert_eq!(block.len(), N_X, "the batched block is the wrong size at R = {rows}");
        for (k, got) in block.iter().enumerate() {
            // What a row of the kernel accumulates: the per-block dots of that
            // activation, summed in word order, in f32. Formed here rather
            // than read from the harness, so the two routes are two
            // implementations — and ONE reference for all three row counts,
            // which is what makes R = 8 and 16 a test rather than a rerun.
            let want = w
                .iter()
                .enumerate()
                .fold(0.0f32, |a, (i, _)| a + out.dot[i * N_X + k]);
            assert_eq!(
                want.to_bits(),
                got.to_bits(),
                "activation {k} at R = {rows}: one row at a time accumulates to {want:e}, \
                 {rows} rows at once give {got:e}. The batched path changed the answer, \
                 which it may not — at any R."
            );
        }
    }
    // And it is not vacuous: a fixture of origins would pass the equality
    // above on a table of zeros.
    let nonzero = out.dot.iter().filter(|v| **v != 0.0).count();
    assert!(
        nonzero > out.dot.len() / 2,
        "only {nonzero} of {} dots are non-zero: the fixture is degenerate and the \
         equality above proves nothing",
        out.dot.len()
    );
}

/// Level 4: the dot, bit for bit, against the same `__fmaf_rn` chain in Rust.
///
/// This is what the kernel computes and what a row accumulates, so it is the
/// statement that matters. The association is the header's: coordinates in
/// trio order, `xb` indexed through the permutation, then the two scales.
#[test]
fn the_dot_is_the_same_chain_in_the_same_order() {
    let t = Tetra::new();
    let bin = build_harness("dot");
    let tb = tables(&t);
    let w = words(0x7e_42a4_0004);
    let centroids = [0.481_562_5f64, 1.372_25f64];
    let gscale = [centroids[0] as f32, centroids[1] as f32];
    let inv = invnorm_table();
    let x = activations(0x4, N_X);
    let out = run(&bin, &tb, gscale, &inv, &w, &x);

    let order = t.order();
    for (i, &word) in w.iter().enumerate() {
        let trio = t.decode_trio_order(word);
        let n2: u32 = trio.iter().map(|&v| (v * v) as u32).sum();
        let m = (n2 / 16) as usize;
        let g = ((word >> 47) & 1) as usize;
        for k in 0..N_X {
            let xb = &x[k * DIM..(k + 1) * DIM];
            let mut acc = 0.0f32;
            for (idx, &v) in trio.iter().enumerate() {
                acc = (v as f32).mul_add(xb[order[idx] as usize], acc);
            }
            // The header's association, and it is not the obvious one: it
            // writes `acc * gscale[g] * invnorm[m]`, which is
            // `(acc · g) · inv` and not `acc · (g · inv)`. The two differ by
            // an ULP on about one word in a thousand, and pinning the wrong
            // one here would have turned a correct kernel into a red test.
            let want = (acc * gscale[g]) * inv[m];
            let got = out.dot[i * N_X + k];
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "word {word:#014x}, x {k}: {got} against {want}"
            );
            // And the same number, loosely, from the f64 reference — so a
            // shared bug between the two f32 chains cannot pass both.
            let f64_dot: f64 = trio.iter().enumerate().map(|(idx, &v)| v as f64 * xb[order[idx] as usize] as f64).sum();
            let scale = if m == 0 { 0.0 } else { centroids[g] / ((16 * m) as f64).sqrt() };
            let want64 = f64_dot * scale;
            let mag: f64 =
                trio.iter().enumerate().map(|(idx, &v)| (v as f64 * xb[order[idx] as usize] as f64).abs()).sum();
            let tol = (mag * scale.abs()).max(1.0) * DOT_TOL;
            assert!((got as f64 - want64).abs() <= tol, "word {word:#014x}, x {k}: {got} against {want64}");
        }
    }
}

/// The gain bit is read, and it is read at bit 47.
///
/// Same label, both gain bits: the two dots must differ by exactly the ratio
/// of the centroids. A header that ignored the bit — which is what the v3
/// floor does, deliberately — gives a ratio of 1 and fails here.
#[test]
fn the_gain_bit_scales_the_block_and_sits_at_bit_forty_seven() {
    let t = Tetra::new();
    let bin = build_harness("gain");
    let tb = tables(&t);
    let mut rng = SplitMix64::new(0x7e_42a4_0005);
    let mut w = Vec::new();
    for _ in 0..256 {
        let label = rng.next() & (WORD_MASK >> 1);
        w.push(label);
        w.push(label | 1u64 << 47);
    }
    let centroids = [0.481_562_5f64, 1.372_25f64];
    let gscale = [centroids[0] as f32, centroids[1] as f32];
    let x = activations(0x5, 1);
    let out = run(&bin, &tb, gscale, &invnorm_table(), &w, &x);

    let ratio = (centroids[1] / centroids[0]) as f32;
    let mut moved = 0;
    for p in 0..w.len() / 2 {
        let (a, b) = (out.dot[2 * p], out.dot[2 * p + 1]);
        // The two words decode to the same point, so `n2` must agree: the gain
        // bit must not have leaked into the magnitude.
        assert_eq!(out.n2[2 * p], out.n2[2 * p + 1], "the gain bit changed the shell");
        if a.abs() > 1e-6 {
            let got = b / a;
            assert!((got - ratio).abs() <= 1e-5, "gain ratio {got} against {ratio}");
            moved += 1;
        }
    }
    assert!(moved > 200, "only {moved} of 256 pairs had a usable dot; the sweep is vacuous");
}
