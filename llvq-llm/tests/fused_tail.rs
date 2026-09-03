//! The `TailPolicy::KeepExact` tail, on the card — lot A7a.
//!
//! ## The question this lot had to answer before writing a line
//!
//! The tail is full precision *by design*. Narrowing it looks like buying
//! bytes with quality, which this dossier has refused for far better trades.
//! The answer is in the code, not in an argument: **the arm the fused path is
//! compared against already holds those columns at binary16**, because
//! `bin/fusedrun` runs both arms at `DType::F16` and its dense arm is
//! `sealed::load`, which calls `.to_dtype(dtype)` on the whole decoded matrix
//! — tail columns included, nothing exempts them. And no published perplexity
//! or MMLU reads this array at all: `bin/ppl` and `bin/mmlu` both score
//! through `sealed::load`, the fused path being a matvec. So the conversion
//! aligns the fused path with its own reference; it does not spend anything
//! the reference still has.
//!
//! ## What is proved here, and what is not
//!
//! Proved, on this machine, no card:
//!
//!  1. the conversion — reconstruction is exactly `f16(v)`, and the `f64 →
//!     f32 → f16` route rounds **once** on values that came out of the file;
//!  2. the accounting — device bytes fall by exactly `2 · tail_weights`, and
//!     the whole-model b/param recount reproduces the two published cells
//!     (4B 5.162, 8B 5.322) when the tail is billed at 32 bits before
//!     predicting them at 16;
//!  3. `tail_dot_h` itself — the device function all three f16-storing
//!     kernels now call is *executed* through the clang++ harness, against an
//!     f64 reference and an f32 replay.
//!
//! 🚨 **Not proved: the kernels.** `tv_slot_h`, `tv_planes_h` and
//! `tv_planes12x_h` are compile-only here — grid, barriers, `warp_sum`, the
//! f16 store — as everything on a machine with no NVIDIA card. That they call
//! `tail_dot_h` with the right pointers is a card's job, and the first run on
//! one must diff `fusedrun`'s two arms.

use llvq_llm::fused::{matrix_side_bytes, tail_f16_bits, ROW_SCALE_BYTES, TAIL_BYTES};
use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};

// ---------------------------------------------------------------------------
// 1. The conversion
// ---------------------------------------------------------------------------

/// What comes back off the card is the binary16 rounding of what went in.
///
/// The mutant this kills is the one that matters: a conversion that keeps the
/// f32 bits, or truncates instead of rounding to nearest even, produces a
/// stream that decodes, runs, and is quietly a different model.
#[test]
fn the_narrowed_tail_is_the_binary16_rounding() {
    // Values chosen to exercise the parts of binary16 a random gaussian never
    // reaches: exact halfway cases (round to even), subnormals, the overflow
    // edge, and zero of both signs.
    let probes: Vec<f64> = vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        1.0 + 2f64.powi(-11),  // halfway between 1 and 1+2^-10 → ties to even
        1.0 + 3.0 * 2f64.powi(-11), // halfway the other way → ties to even, up
        6.0e-8,                // below the smallest subnormal → flushes to 0
        6.104e-5,              // around the smallest normal
        5.96e-8,               // smallest subnormal, near enough
        65504.0,               // largest finite binary16
        65520.0,               // rounds to infinity
        -65520.0,
        1.0 / 3.0,
        std::f64::consts::PI,
    ];
    let bits = tail_f16_bits(&probes);
    assert_eq!(bits.len(), probes.len());
    for (&v, &b) in probes.iter().zip(&bits) {
        let want = half::f16::from_f32(v as f32);
        let got = half::f16::from_bits(b);
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "{v}: {got} instead of {want}"
        );
        // Sign of zero survives, which is not automatic through a naive
        // truncation and is free through a correct one.
        if v == 0.0 {
            assert_eq!(got.to_f32().is_sign_negative(), v.is_sign_negative());
        }
    }
}

/// `f64 → f32 → f16` rounds once, because the `f64` came out of a file that
/// stores 32 bits.
///
/// Double rounding is the classic way a "harmless" narrowing becomes wrong on
/// a few values in a million. It cannot happen here, and this test says why
/// rather than hoping: `calib.rs` narrows the tail to `f32` before writing it
/// and `llvq_artifact::format` reads it back from 32 bits, so every value
/// reaching [`tail_f16_bits`] is *exactly* representable in `f32` and the
/// first step is the identity. The test feeds only such values and demands
/// agreement with the direct `f64 → f16` rounding.
#[test]
fn narrowing_the_tail_rounds_once() {
    let mut rng = llvq_core::SplitMix64::new(0x00A7_A001);
    // Values as the file yields them: an f32 widened to f64.
    let from_file: Vec<f64> = (0..20_000)
        .map(|_| (rng.next_gaussian() as f32) as f64)
        .collect();
    for (&v, &b) in from_file.iter().zip(&tail_f16_bits(&from_file)) {
        assert_eq!(
            b,
            half::f16::from_f64(v).to_bits(),
            "{v:e}: the f32 route and the direct route diverge"
        );
    }
}

// ---------------------------------------------------------------------------
// 2. The accounting
// ---------------------------------------------------------------------------

/// One matrix's side bytes: the tail at two bytes, the row scales still at
/// four.
///
/// Two mutants die here — a tail left at four bytes, and a row-scale vector
/// dragged down to two along with it (the change is deliberately *not*
/// symmetric: `rscale[row]` multiplies the whole block sum, so its rounding is
/// relative on every quantized weight of the row at once).
#[test]
fn side_bytes_bill_the_tail_at_two_and_the_scales_at_four() {
    assert_eq!(TAIL_BYTES, 2);
    assert_eq!(ROW_SCALE_BYTES, 4);
    // Real 4B shapes: (d_out, d_in) → tail_w = d_in % 24.
    for &(d_out, d_in) in &[(2560usize, 2560usize), (2560, 9728), (9728, 2560)] {
        let tail_w = d_in % 24;
        assert_eq!(
            matrix_side_bytes(d_out, tail_w),
            (d_out * tail_w) as u64 * 2 + d_out as u64 * 4
        );
        // And the saving against the layout that shipped before A7a.
        let before = (d_out * tail_w) as u64 * 4 + d_out as u64 * 4;
        assert_eq!(
            before - matrix_side_bytes(d_out, tail_w),
            (d_out * tail_w) as u64 * 2
        );
    }
    // A matrix whose `d_in` is a multiple of 24 has no tail at all, and must
    // still be billed its row scales — the 8B's `down_proj` is exactly this
    // (12288 = 24 · 512) and it is 26 % of that model's weights.
    assert_eq!(matrix_side_bytes(4096, 12288 % 24), 4096 * 4);
}

/// The published shapes of a model, as `rtbits` counts them from the file.
///
/// Not invented: every field is read off
/// `docs/mesures/rtbits-planes-8b-2026-08-09.txt`, which billed both sealed
/// artifacts block by block on the CPU. They are constants here because the
/// artifacts are not in the repo; the test that uses them re-derives the
/// journal's own published cells before predicting anything, so a mistyped
/// constant fails rather than propagates.
struct Model {
    name: &'static str,
    /// `Σ d_out · d_in`, tail columns included.
    linear_weights: u64,
    /// `Σ d_out · (d_in mod 24)`.
    tail_weights: u64,
    /// `Σ d_out`, one scale each.
    rows: u64,
    /// Blocks of 24 quantized weights.
    blocks: u64,
    /// Embedding plus, when the ends are untied, `lm_head`.
    embed_params: u64,
    /// Norms, carried f16 whatever happens to the embedding.
    other_params: u64,
    /// `Planes14` + q8 embedding, whole model, as the journal publishes it
    /// with the tail at 32 bits.
    published_bpp_q8: f64,
}

const MODELS: [Model; 2] = [
    Model {
        name: "Qwen3-4B",
        linear_weights: 3_633_315_840,
        tail_weights: 16_957_440,
        rows: 1_105_920,
        blocks: 150_681_600,
        embed_params: 388_956_160,
        other_params: 196_096,
        published_bpp_q8: 5.1619,
    },
    Model {
        name: "Qwen3-8B",
        linear_weights: 6_945_767_424,
        tail_weights: 20_054_016,
        rows: 1_400_832,
        blocks: 288_571_392,
        embed_params: 1_244_659_712,
        other_params: 308_224,
        published_bpp_q8: 5.3220,
    },
];

/// `Planes14`'s payload: a frozen 14-byte record per block, no bases, no
/// exceptions.
const PLANES14_RECORD_BITS: u64 = 14 * 8;
/// int8 plus one f16 scale and one f16 bias per group of 64 — 8 + 32/64.
const EMBED_Q8_BPP: f64 = 8.5;

/// The width the *runtime* actually bills a tail weight at — read from the
/// shipped constant, never spelled `16` here.
///
/// 🕳️ The `rtbits` lot lost a mutant to exactly this shortcut: its tests coded
/// `8.5` on both sides of the q8 embedding, so deleting the group-scale term
/// broke nothing. Deriving the number from the code under test is what makes
/// `TAIL_BYTES = 4` fail this file rather than pass it.
const RUNTIME_TAIL_BITS: u64 = TAIL_BYTES * 8;
/// What the tail used to cost, before lot A7a — the column the published
/// cells were computed in.
const FILE_TAIL_BITS: u64 = 32;

/// Whole-model b/param, the only grandeur comparable to an AWQ figure, with
/// the tail billed at `tail_bits`.
fn whole_model_bpp(m: &Model, tail_bits: u64) -> f64 {
    let linear = m.blocks * PLANES14_RECORD_BITS
        + m.tail_weights * tail_bits
        + m.rows * ROW_SCALE_BYTES * 8;
    let total = linear as f64
        + m.embed_params as f64 * EMBED_Q8_BPP
        + m.other_params as f64 * 16.0;
    total / (m.linear_weights + m.embed_params + m.other_params) as f64
}

/// What the tail costs the whole model, recounted rather than quoted — and
/// the reason the answer does not transport from one model to the next.
///
/// The test is built so it cannot merely restate its own arithmetic: it first
/// reproduces the **published** cells (4B 5.162, 8B 5.322,
/// `docs/mesures/rtbits-planes-8b-2026-08-09.txt` §3) from the same shapes
/// with the tail at 32 bits. Only then does it read off the 16-bit column. A
/// wrong shape constant, a wrong record width or a wrong embedding rate fails
/// on the first half.
#[test]
fn narrowing_the_tail_is_worth_less_on_a_model_whose_down_proj_has_none() {
    let expect = [
        // (b/param at 16-bit tail, saving) — the numbers this lot publishes.
        (5.0945, 0.0674),
        (5.2828, 0.0392),
    ];
    for (m, &(want_bpp, want_gain)) in MODELS.iter().zip(&expect) {
        let at32 = whole_model_bpp(m, FILE_TAIL_BITS);
        assert!(
            (at32 - m.published_bpp_q8).abs() < 5e-4,
            "{}: the formula gives {at32:.4} against {:.4} published, one of the \
             shape constants is wrong, and nothing below it holds",
            m.name,
            m.published_bpp_q8
        );

        let at16 = whole_model_bpp(m, RUNTIME_TAIL_BITS);
        assert!(
            (at16 - want_bpp).abs() < 5e-4,
            "{}: tail at {RUNTIME_TAIL_BITS} b → {at16:.4}, expected {want_bpp:.4}",
            m.name
        );

        // The saving is the width the tail gave up, over *every* parameter,
        // embedding included. Stated twice, once as a difference and once in
        // closed form, so a sign or a denominator slip cannot pass.
        let gain = at32 - at16;
        let closed = (FILE_TAIL_BITS - RUNTIME_TAIL_BITS) as f64 * m.tail_weights as f64
            / (m.linear_weights + m.embed_params + m.other_params) as f64;
        assert!((gain - closed).abs() < 1e-9, "{}: closed form", m.name);
        assert!(
            (gain - want_gain).abs() < 5e-4,
            "{}: gain {gain:.4} b/param, expected {want_gain:.4}",
            m.name
        );
    }

    // ⚠️ The headline of this lot, and the reason it must never be quoted as
    // a property of the method: the 8B's `intermediate_size` is 12288 = 24·512
    // exactly, so its `down_proj` — 26 % of its weights — has no tail. The
    // same change is worth 42 % less there.
    let saving = |m: &Model| {
        whole_model_bpp(m, FILE_TAIL_BITS) - whole_model_bpp(m, RUNTIME_TAIL_BITS)
    };
    let (g4, g8) = (saving(&MODELS[0]), saving(&MODELS[1]));
    assert!(g8 < 0.6 * g4, "the 8B gain must sit clearly under the 4B one");
}

// ---------------------------------------------------------------------------
// 3. The device function, executed
// ---------------------------------------------------------------------------

/// `tail_dot_h`, run through the clang++ harness, decides what Rust decides.
///
/// This is the one part of the device change a machine with no card can hold
/// to account, and it earns the test three times over:
///
///  * it forces `h2f`'s host branch to be a real widening. It returned `0`
///    until this lot, licensed by "nothing executes it"; with a stub, every
///    row below comes out zero and the first assertion fires;
///  * it pins the indexing — `tail[row · tail_w + i]` against `xt[i]`. A
///    transposed tail or an absolutely-indexed activation returns finite,
///    plausible, wrong numbers on the card, and the f64 reference is what
///    separates them;
///  * it compiles all three f16-storing entry points with the new
///    `const unsigned short*` signature, which is what would otherwise cost a
///    billed job to discover.
#[test]
#[cfg_attr(debug_assertions, ignore = "compiles C++, run in release")]
fn the_tail_epilogue_decides_what_rust_decides() {
    const D_OUT: usize = 24;
    const TAIL_W: usize = 16; // the 4B's `d_in = 2560` tail

    let out = std::env::temp_dir().join("llvq_host_tail_h");
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests");
    let st = Command::new("clang++")
        .args([
            "-std=c++17",
            "-O2",
            // Contraction off: the epilogue is deliberately multiply-then-add
            // and not `__fmaf_rn`, so a float difference stays attributable.
            "-ffp-contract=off",
            "-Wall",
            "-Wextra",
            "-Werror",
        ])
        .arg(dir.join("host_tail_h.cpp"))
        .arg("-o")
        .arg(&out)
        .status()
        .expect("clang++ is on PATH");
    assert!(st.success(), "the f16 tail source does not compile as host C++");

    let mut rng = llvq_core::SplitMix64::new(0xA7AC_0DE0);
    // As they arrive: f32 values out of the file, narrowed on the host.
    let tail_f32: Vec<f64> = (0..D_OUT * TAIL_W)
        .map(|_| (rng.next_gaussian() as f32) as f64)
        .collect();
    let tail = tail_f16_bits(&tail_f32);
    let xt: Vec<f32> = (0..TAIL_W).map(|_| rng.next_gaussian() as f32).collect();

    let mut fx = Vec::new();
    fx.extend_from_slice(&(D_OUT as u32).to_le_bytes());
    fx.extend_from_slice(&(TAIL_W as u32).to_le_bytes());
    fx.extend(tail.iter().flat_map(|v| v.to_le_bytes()));
    fx.extend(xt.iter().flat_map(|v| v.to_le_bytes()));

    let mut child = Command::new(&out)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("host probe runs");
    child
        .stdin
        .take()
        .expect("stdin")
        .write_all(&fx)
        .expect("fixture written");
    let res = child.wait_with_output().expect("host probe finishes");
    assert!(res.status.success(), "host probe failed");
    assert_eq!(res.stdout.len(), D_OUT * 4, "the probe wrote something else");
    let got: Vec<f32> = res
        .stdout
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let mut worst = 0.0f64;
    for row in 0..D_OUT {
        // Reference in f64, over the *narrowed* weights: what is being checked
        // is the epilogue, not the narrowing (that is the first test above).
        let (mut want, mut scale) = (0.0f64, 0.0f64);
        // And a replay in f32, in the epilogue's own association, which must
        // agree bit for bit — that is what says the device function sums in
        // f32 and multiplies before it adds.
        let mut replay = 0.0f32;
        for i in 0..TAIL_W {
            let w = half::f16::from_bits(tail[row * TAIL_W + i]);
            want += w.to_f32() as f64 * xt[i] as f64;
            scale += (w.to_f32() as f64 * xt[i] as f64).abs();
            replay += w.to_f32() * xt[i];
        }
        assert_eq!(
            got[row].to_bits(),
            replay.to_bits(),
            "row {row}: {} against the f32 replay {replay}",
            got[row]
        );
        assert!(
            got[row] != 0.0,
            "row {row} is zero, a stubbed `h2f` would give exactly that"
        );
        let e = (got[row] as f64 - want).abs() / scale.max(1e-12);
        worst = worst.max(e);
        assert!(e < 1e-6, "row {row}: gap {e:.2e}·Σ|w·x|");
    }

    // An empty tail must be a zero, not a read: `d_in` a multiple of 24 is the
    // 8B's `down_proj`, 26 % of that model, and the loader hands the kernel a
    // one-element dummy array in that case.
    let mut fx = Vec::new();
    fx.extend_from_slice(&4u32.to_le_bytes());
    fx.extend_from_slice(&0u32.to_le_bytes());
    let mut child = Command::new(&out)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("host probe runs");
    child.stdin.take().expect("stdin").write_all(&fx).expect("written");
    let res = child.wait_with_output().expect("host probe finishes");
    assert!(res.status.success());
    assert_eq!(res.stdout, vec![0u8; 16], "empty tail: contribution not zero");

    println!("tail_dot_h host: {D_OUT} rows × {TAIL_W}, worst gap {worst:.1e}·Σ|w·x|");
}
