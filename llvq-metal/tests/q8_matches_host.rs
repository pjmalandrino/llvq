//! The q8 embedding kernels in MSL, against llvq-artifact's own dequantizer.
//!
//! ## Where the reference comes from, and why it is not this test's own idea
//!
//! Nothing here was written by reading `emb_q8.metal`. Two pre-existing pieces
//! of the repository carry the truth, and both predate the port:
//!
//!  1. `llvq_artifact::RawTensor::to_f32` says what a q8 payload MEANS:
//!     `f16_to_f32(scale) * q + f16_to_f32(bias)`, per group of 64 along the
//!     row. It is the reader every non-GPU path in the workspace uses. The
//!     expected value of every weight comes from calling it.
//!  2. `llvq-llm/src/embedquant.rs` says how a table is PACKED: MLX's scheme,
//!     f16 scale and bias a group, `q = round((w - bias) / scale)` clamped,
//!     with the constant-group branch that stores scale 1 and q = 0.
//!     `quantize_q8` below mirrors that function line for line, because
//!     `llvq-llm` is not a dev-dependency of this crate and adding one was out
//!     of scope. `llvq-llm/tests/embed_q8.rs` already pins the shipped
//!     function against an independent reimplementation, so the scheme itself
//!     is under test elsewhere.
//!
//! The lane striding and the butterfly are the one thing this file does take
//! from the kernel side, and from the CUDA original rather than from the MSL:
//! `tv_q8_h` gives lane `l` the words `l, l + 32, ...` and reduces with an
//! `__shfl_xor` butterfly. Floating-point addition is not associative, so a
//! reference that summed in a plain loop would need a tolerance, and a
//! tolerance is what lets a real defect through. A transposed lane partition
//! or a byte read from the wrong shift lands inside any epsilon a 128-wide dot
//! product would need. The assertion is equality.
//!
//! An audit of 2026-09-20 found both host references for the Tetra matvec had
//! been written from the shader, so both carried its mistake and the gate
//! passed on a wrong kernel. That is the failure this header exists to
//! prevent.
//!
//! ## The mutation run
//!
//! Nine mutants of `emb_q8.metal` on 2026-09-21, eight killed. Failing tests
//! in brackets, out of seven:
//!
//!   * the gather's group index `c >> 6` widened to `c >> 5` [3],
//!   * the gather's byte select `(c & 3) * 8` narrowed to `(c & 3) * 4` [3],
//!   * the gather dropping `ids_off` [2],
//!   * the head's lane stride 32 changed to 31, so lanes overlap [4],
//!   * the head's butterfly stopping at `k > 1` [5],
//!   * the head reading column 3 where it should read column 2 [5],
//!   * the head dropping `y_off` [1],
//!   * the `threadgroup_barrier` in `tv_q8_metal` removed [5].
//!
//! The barrier one is worth a sentence, because the same mutant SURVIVES in
//! `tetra48_matvec_matches_host.rs` and was written up there as a race a
//! functional test cannot catch. Here it dies on every run. A threadgroup of
//! 256 threads is eight SIMD-groups that Apple does not lock-step, so the
//! groups that reach the weight loop first read a staging area the others have
//! not filled. The Tetra kernel stages inside a tile loop that already carries
//! a second barrier, which is why the difference. A race that happens to fire
//! is still a race: this kills the mutant, it does not promise the mutant
//! would die on another driver.
//!
//! ONE DID NOT DIE, and it cannot: `q8_deq` contracted into a single
//! `fma(float(s), float(q), float(b))` gives the same bits as the written
//! `float(s) * float(q) + float(b)` on every input the format can hold. An f16
//! significand is 11 bits and `q` is 8, so the product needs at most 19 of the
//! 24 an f32 holds, and 65504 * 255 is about 1.7e7, nowhere near an exponent
//! limit. The multiply is exact, so there is no first rounding to differ
//! about. Checked by exhaustion over all 63,488 finite f16 scales times all
//! 256 byte values: zero products rounded. That makes this an equivalent
//! mutant, which is the one honest reason a survivor is not a hole, and it
//! means `#pragma clang fp contract(off)` protects nothing in THIS file. It
//! stays for the reason the shader gives.
//!
//! ## What it does not cover
//!
//! Nothing runs through candle. The shipped Metal adapter must reach these
//! kernels through `CustomOp1::metal_fwd` and objc2-metal; this file drives
//! them through `llvq-metal`'s metal-rs host layer, which proves the KERNELS
//! and says nothing about the binding. There is no f16 store here either: both
//! kernels write f32 on purpose, and narrowing is a later lot with its own
//! gate.

#![cfg(target_os = "macos")]

use llvq_artifact::{f16_to_f32, QuantData, RawData, RawTensor};
use llvq_core::SplitMix64;
use llvq_metal::{f16_bits, Kernel};

const SOURCE: &str = include_str!("../../llvq-llm/kernels/emb_q8.metal");

/// The format's constant, not a knob: the kernels spell it `c >> 6`.
const GROUP_WIDTH: usize = 64;
/// Lanes in an Apple SIMD-group. The head gives one group to a row.
const LANES: usize = 32;
/// Threads a threadgroup, so eight rows each, the number the CUDA host uses.
const THREADS: usize = 256;
/// Threadgroup memory on an M3 Max, measured. The head stages `d` floats and
/// the host owes the allocation; asking for more than this fails the dispatch.
const TG_LIMIT: u64 = 32_768;
/// IEEE binary16 one, the scale `embedquant` stores for a constant group.
const F16_ONE: u16 = 0x3c00;

/// A packed q8 table and what every one of its weights dequantizes to.
struct Table {
    rows: usize,
    d: usize,
    gpr: usize,
    /// The packed bytes as the kernel reads them, four to a word.
    words: Vec<u32>,
    scales: Vec<u16>,
    biases: Vec<u16>,
    /// `RawTensor::to_f32` over the whole table, row-major. The reference.
    deq: Vec<f32>,
}

/// `embedquant::quantize_affine(t, 8, 64)`, mirrored.
///
/// Same order of operations, and that matters in two places: `q` is chosen
/// against the ROUNDED scale and bias, so the error the encoder minimizes is
/// the one the reader sees, and a group whose range rounds to zero in f16 gets
/// scale 1 with every `q` at zero, the bias carrying the value.
fn quantize_q8(weights: &[u16], rows: usize, d: usize) -> QuantData {
    let gpr = d.div_ceil(GROUP_WIDTH);
    let mut packed = vec![0u8; rows * d];
    let mut scales = Vec::with_capacity(rows * gpr);
    let mut biases = Vec::with_capacity(rows * gpr);
    for row in 0..rows {
        for g in 0..gpr {
            let lo = row * d + g * GROUP_WIDTH;
            let hi = row * d + ((g + 1) * GROUP_WIDTH).min(d);
            let mut mn = f32::INFINITY;
            let mut mx = f32::NEG_INFINITY;
            for &b in &weights[lo..hi] {
                let v = f16_to_f32(b);
                mn = mn.min(v);
                mx = mx.max(v);
            }
            let s = f16_bits((mx - mn) / 255.0);
            let graded = f16_to_f32(s) > 0.0;
            let s = if graded { s } else { F16_ONE };
            let b16 = f16_bits(mn);
            scales.push(s);
            biases.push(b16);
            let (sf, bf) = (f16_to_f32(s), f16_to_f32(b16));
            for (k, &b) in weights[lo..hi].iter().enumerate() {
                packed[lo + k] = match graded {
                    true => ((f16_to_f32(b) - bf) / sf).round().clamp(0.0, 255.0) as u8,
                    false => 0,
                };
            }
        }
    }
    QuantData {
        bits: 8,
        group: GROUP_WIDTH,
        packed,
        scales,
        biases,
    }
}

/// A random table, with one constant group planted on purpose.
///
/// The constant group is the encoder's ungraded branch: scale 1, every `q` at
/// zero, the bias alone carrying the value. It costs nothing to include and it
/// is a real state of a shipped table, where a row of an untrained token is
/// flat.
fn table(seed: u64, rows: usize, d: usize) -> Table {
    assert!(d.is_multiple_of(4), "the kernels read rows as u32 words");
    let mut rng = SplitMix64::new(seed);
    let mut weights: Vec<u16> = (0..rows * d)
        .map(|_| f16_bits(rng.next_gaussian() as f32))
        .collect();
    for w in weights.iter_mut().take(GROUP_WIDTH) {
        *w = f16_bits(0.25);
    }

    let q = quantize_q8(&weights, rows, d);
    // The kernel reads packed bytes through a u32 pointer, so word `i` holds
    // columns 4i..4i+4 with column 4i in the low byte. Little-endian, spelled
    // out rather than reinterpreted, so the test says what it assumes.
    let words = q
        .packed
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let (scales, biases, gpr) = (q.scales.clone(), q.biases.clone(), d.div_ceil(GROUP_WIDTH));
    let tensor = RawTensor {
        name: "embed.weight".into(),
        dims: vec![rows, d],
        data: RawData::Quant(q),
    };
    Table {
        rows,
        d,
        gpr,
        words,
        scales,
        biases,
        deq: tensor.to_f32(),
    }
}

/// f16 activation bits, `off` values of decoy ahead of the real row.
///
/// The decoy is what makes `x_off` load-bearing: a kernel that ignored the
/// offset would read the decoy and the answer would move.
fn activation(seed: u64, d: usize, off: usize) -> Vec<u16> {
    let mut rng = SplitMix64::new(seed);
    (0..off + d)
        .map(|_| f16_bits(rng.next_gaussian() as f32))
        .collect()
}

fn compile(name: &str, exact: bool) -> Kernel {
    match exact {
        true => Kernel::new_exact(SOURCE, name),
        false => Kernel::new(SOURCE, name),
    }
    .expect("the q8 kernels compile")
}

fn run_gather(t: &Table, ids: &[u32], ids_off: u32, exact: bool) -> Vec<f32> {
    let k = compile("emb_q8_gather_metal", exact);
    let ntok = ids.len() - ids_off as usize;
    let b_w = k.buffer(&t.words);
    let b_s = k.buffer(&t.scales);
    let b_b = k.buffer(&t.biases);
    let b_i = k.buffer(ids);
    let b_y = k.empty::<f32>(ntok * t.d);
    let (d, gpr) = (t.d as u32, t.gpr as u32);

    k.dispatch((ntok * THREADS) as u64, THREADS as u64, |enc| {
        enc.set_buffer(0, Some(&b_w), 0);
        enc.set_buffer(1, Some(&b_s), 0);
        enc.set_buffer(2, Some(&b_b), 0);
        enc.set_buffer(3, Some(&b_i), 0);
        enc.set_buffer(4, Some(&b_y), 0);
        set_u32(enc, 5, &d);
        set_u32(enc, 6, &gpr);
        set_u32(enc, 7, &ids_off);
    });
    unsafe { k.read::<f32>(&b_y, ntok * t.d) }
}

/// The head, writing into a buffer pre-filled with `sentinel`.
///
/// The sentinel is how `y_off` is checked: every row outside
/// `y_off..y_off + rows` must still hold it when the dispatch returns.
fn run_head(
    t: &Table,
    x: &[u16],
    x_off: u32,
    y_off: u32,
    y_len: usize,
    sentinel: f32,
    exact: bool,
) -> Vec<f32> {
    let k = compile("tv_q8_metal", exact);
    assert!(
        (t.rows * LANES).is_multiple_of(THREADS),
        "a partial threadgroup would split a row's SIMD-group"
    );
    // Metal wants the threadgroup allocation in multiples of 16 bytes. Omitting
    // the call entirely is not an error on Apple: the kernel runs and writes
    // zeros, which reads as a plausible logit.
    let tg_bytes = ((t.d * 4) as u64).next_multiple_of(16);
    assert!(
        tg_bytes <= TG_LIMIT,
        "{tg_bytes} B of threadgroup memory, the device offers {TG_LIMIT}"
    );

    let b_w = k.buffer(&t.words);
    let b_s = k.buffer(&t.scales);
    let b_b = k.buffer(&t.biases);
    let b_x = k.buffer(x);
    let b_y = k.buffer(&vec![sentinel; y_len]);
    let (d, gpr) = (t.d as u32, t.gpr as u32);

    k.dispatch((t.rows * LANES) as u64, THREADS as u64, |enc| {
        enc.set_buffer(0, Some(&b_w), 0);
        enc.set_buffer(1, Some(&b_s), 0);
        enc.set_buffer(2, Some(&b_b), 0);
        enc.set_buffer(3, Some(&b_x), 0);
        enc.set_buffer(4, Some(&b_y), 0);
        set_u32(enc, 5, &d);
        set_u32(enc, 6, &gpr);
        set_u32(enc, 7, &x_off);
        set_u32(enc, 8, &y_off);
        enc.set_threadgroup_memory_length(0, tg_bytes);
    });
    unsafe { k.read::<f32>(&b_y, y_len) }
}

fn set_u32(enc: &metal::ComputeCommandEncoderRef, index: u64, v: &u32) {
    enc.set_bytes(index, 4, v as *const u32 as *const std::ffi::c_void);
}

/// What the gather owes: the artifact's dequantized row, verbatim.
fn gather_reference(t: &Table, ids: &[u32], ids_off: usize) -> Vec<f32> {
    let mut y = Vec::with_capacity((ids.len() - ids_off) * t.d);
    for &id in &ids[ids_off..] {
        let row = id as usize;
        y.extend_from_slice(&t.deq[row * t.d..(row + 1) * t.d]);
    }
    y
}

/// The head's arithmetic, on the host, in the kernel's order.
///
/// The weights are the artifact's, the order is the CUDA original's: lane `l`
/// takes words `l, l + 32, ...`, the four columns of a word accumulate in
/// order, and the lanes meet in an `__shfl_xor` butterfly.
fn head_reference(t: &Table, x: &[u16], x_off: usize) -> Vec<f32> {
    let xs: Vec<f32> = (0..t.d).map(|i| f16_to_f32(x[x_off + i])).collect();
    let mut y = vec![0f32; t.rows];
    for (row, out) in y.iter_mut().enumerate() {
        let mut lanes = [0f32; LANES];
        for wi in 0..t.d / 4 {
            let c = wi * 4;
            let acc = &mut lanes[wi % LANES];
            for k in 0..4 {
                *acc = t.deq[row * t.d + c + k].mul_add(xs[c + k], *acc);
            }
        }
        for k in [16usize, 8, 4, 2, 1] {
            let mut next = [0f32; LANES];
            for (l, n) in next.iter_mut().enumerate() {
                *n = lanes[l] + lanes[l ^ k];
            }
            lanes = next;
        }
        *out = lanes[0];
    }
    y
}

/// Full groups, and `ids_off` skipping two decoy ids that name other rows.
#[test]
fn the_gather_matches_the_artifact_dequantizer() {
    let t = table(0xE_08A0, 40, 128);
    let ids = [37u32, 5, 0, 12, 39, 1, 26];
    let got = run_gather(&t, &ids, 2, true);
    let want = gather_reference(&t, &ids, 2);
    assert_eq!(got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert_eq!(g, w, "value {i}: metal {g} against the artifact {w}");
    }
}

/// A row length that is not a multiple of 64, so the last group is short.
///
/// `gpr` is 2 here and the second group covers 36 columns. A kernel that
/// derived the group from `gpr` instead of from the constant 64 would read the
/// wrong pair on every column past 64.
#[test]
fn the_gather_matches_across_a_short_last_group() {
    let t = table(0xE_08A1, 24, 100);
    assert_eq!(t.gpr, 2);
    let ids = [23u32, 0, 17, 9];
    let got = run_gather(&t, &ids, 1, true);
    let want = gather_reference(&t, &ids, 1);
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert_eq!(g, w, "value {i}: metal {g} against the artifact {w}");
    }
}

/// Every lane busy: 128 columns is 32 words, one a lane.
#[test]
fn the_head_matches_the_host_on_a_full_simd_group() {
    let t = table(0xE_08A2, 16, 128);
    let x = activation(0xE_08B2, t.d, 7);
    let got = run_head(&t, &x, 7, 0, t.rows, f32::NAN, true);
    let want = head_reference(&t, &x, 7);
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert_eq!(g, w, "row {i}: metal {g} against host {w}");
    }
}

/// Twenty-five words, so lanes 25 to 31 take no word at all and enter the
/// butterfly at zero. An off-by-one in the lane loop shows here and nowhere
/// else.
#[test]
fn the_head_matches_the_host_with_idle_lanes() {
    let t = table(0xE_08A3, 24, 100);
    let x = activation(0xE_08B3, t.d, 3);
    let got = run_head(&t, &x, 3, 0, t.rows, f32::NAN, true);
    let want = head_reference(&t, &x, 3);
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert_eq!(g, w, "row {i}: metal {g} against host {w}");
    }
}

/// `y_off` writes the second token's block and leaves the first alone.
///
/// This is the multi-token shape: the CUDA host loops rows of one call into a
/// single output buffer, one launch a row.
#[test]
fn the_head_writes_at_its_offset_and_touches_nothing_else() {
    let t = table(0xE_08A4, 16, 128);
    let x = activation(0xE_08B4, t.d, 0);
    let sentinel = -12345.0f32;
    let got = run_head(&t, &x, 0, t.rows as u32, 3 * t.rows, sentinel, true);
    let want = head_reference(&t, &x, 0);
    for (i, (g, w)) in got[t.rows..2 * t.rows].iter().zip(&want).enumerate() {
        assert_eq!(g, w, "row {i}: metal {g} against host {w}");
    }
    for (i, g) in got.iter().enumerate() {
        if !(t.rows..2 * t.rows).contains(&i) {
            assert_eq!(*g, sentinel, "slot {i} was written and should not be");
        }
    }
}

/// The result depends on the activation: a kernel that ignored `x` would pass
/// every equality above if the reference ignored it too.
#[test]
fn a_different_activation_gives_a_different_answer() {
    let t = table(0xE_08A5, 16, 128);
    let a_bits = activation(0xE_08B5, t.d, 0);
    let b_bits = activation(0xE_08C5, t.d, 0);
    let a = run_head(&t, &a_bits, 0, 0, t.rows, f32::NAN, true);
    let b = run_head(&t, &b_bits, 0, 0, t.rows, f32::NAN, true);
    assert!(
        a.iter().zip(&b).any(|(p, q)| p != q),
        "the head must read the activation"
    );
    let want = head_reference(&t, &b_bits, 0);
    for (i, (g, w)) in b.iter().zip(&want).enumerate() {
        assert_eq!(g, w, "row {i} on the second activation");
    }
}

/// The pragma, not the compile option, is what carries the arithmetic.
///
/// The shipped path does not use `new_exact`: `llvq-llm/src/fused_metal.rs`
/// compiles MSL through candle with `None` options, which is Metal's default
/// and has fast math ON. If the answer depended on the option, the gate and
/// the shipped path would be two different kernels and this file would judge
/// the wrong one.
///
/// What it covers is NOT the contraction in `q8_deq`: the mutation run showed
/// that one is exact either way. It is everything fast math does besides
/// contracting, chiefly reassociating the `fma` chain and the butterfly, which
/// no proof above covers and which would move the answer if it happened.
#[test]
fn the_same_source_gives_the_same_numbers_with_fast_math_either_way() {
    let t = table(0xE_08A6, 16, 128);
    let x = activation(0xE_08B6, t.d, 2);
    let exact = run_head(&t, &x, 2, 0, t.rows, f32::NAN, true);
    let fast = run_head(&t, &x, 2, 0, t.rows, f32::NAN, false);
    let want = head_reference(&t, &x, 2);
    for (i, ((a, b), w)) in exact.iter().zip(&fast).zip(&want).enumerate() {
        assert_eq!(a, b, "row {i}: fast math moved the answer, {a} against {b}");
        assert_eq!(a, w, "row {i}: and neither matches the host reference");
    }

    let ids = [0u32, 9, 15];
    let g_exact = run_gather(&t, &ids, 0, true);
    let g_fast = run_gather(&t, &ids, 0, false);
    let g_want = gather_reference(&t, &ids, 0);
    for (i, ((a, b), w)) in g_exact.iter().zip(&g_fast).zip(&g_want).enumerate() {
        assert_eq!(a, b, "value {i}: fast math moved the gather");
        assert_eq!(a, w, "value {i}: and neither matches the artifact");
    }
}

// ---------------------------------------------------------------------------
// The striding loops, actually strided.
//
// A review of 2026-09-21 planted `wi += 32u -> wi += 4096u` in the head and
// `c += tgs -> c += 4096u` in the gather. Both SURVIVED all seven tests,
// because every fixture ran d = 128 or d = 100: that is 32 or 25 words against
// 32 lanes, and 128 or 100 columns against 256 threads, so each loop took at
// most ONE trip and the stride was never read.
//
// These two run d large enough that the loops must iterate. Nothing else
// about them is new.
// ---------------------------------------------------------------------------

/// 1,024 columns is 256 words, eight a lane.
#[test]
fn the_head_matches_the_host_when_every_lane_takes_eight_words() {
    let t = table(0xE_08C0, 16, 1024);
    assert_eq!(t.d / 4 / 32, 8, "eight words a lane, so the stride is read seven times");
    let x = activation(0xE_08D0, t.d, 0);
    let got = run_head(&t, &x, 0, 0, t.rows, f32::NAN, true);
    let want = head_reference(&t, &x, 0);
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert_eq!(g, w, "row {i}: metal {g} against host {w}");
    }
}

/// 1,600 columns is 400 words, so the last lanes take one fewer than the
/// first. A stride that is right for a round count and wrong for a ragged one
/// shows here.
#[test]
fn the_gather_matches_when_every_thread_takes_several_columns() {
    let t = table(0xE_08C1, 24, 1600);
    assert_eq!(t.d % 64, 0, "whole groups, so the group index is not the thing under test");
    let ids = [23u32, 0, 17, 9, 11];
    let got = run_gather(&t, &ids, 1, true);
    let want = gather_reference(&t, &ids, 1);
    assert_eq!(got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert_eq!(g, w, "value {i}: metal {g} against the artifact {w}");
    }
}
