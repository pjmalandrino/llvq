//! The affine int4 matvec in MSL, against the repository's own dequantization.
//!
//! ## Where the ground truth comes from
//!
//! `llvq_artifact::Int4Matrix::to_f32`. It is the reader's own decode of an
//! int4 g128 record, it predates this port by weeks, and it was written from
//! the FORMAT rather than from any kernel. `llvq-llm/src/bin/export.rs`
//! dequantizes records to safetensors through it, and
//! `the_two_int4_paths_agree_bit_for_bit` pins it against
//! `RawTensor::to_f32`.
//!
//! That provenance is the point of this file. On 2026-09-21 an audit found
//! both host references for the Tetra matvec had been written by reading the
//! shader, so both carried the shader's mistake and the gate passed while the
//! kernel was wrong. A reference derived from the implementation proves
//! nothing. Here the WEIGHTS come from the artifact crate and nothing else;
//! only the summation order is taken from `llvq-llm/kernels/tv_q4_h.cu`, the
//! CUDA original this file's kernel is a port of.
//!
//! ## Why the summation order has to be reproduced
//!
//! Floating-point addition is not associative, so "the same matvec" in a
//! different order is a different number. A reference that summed the columns
//! in a plain loop would need a tolerance, and a tolerance is what lets a real
//! defect through: a swapped nibble pair, a group index off by one and a lane
//! stride of 31 all land inside any epsilon a 2560-wide dot product needs.
//!
//! So the reference below takes 32 lanes striding the words, then the same
//! shuffle-xor butterfly. The assertion is equality.
//!
//! ## The mutation run, 2026-09-21
//!
//! Nine mutants of the shader, eight killed: the nibble index reversed
//! (`4 * (7 - k)`), the group term dropped from the scale index, the row term
//! dropped from the scale index, the `& 0xf` nibble mask removed, the lane
//! stride at 31, the butterfly stopped at two lanes, `threadgroup_barrier`
//! removed, and the row term dropped from the word index.
//!
//! ONE DID NOT DIE, and it cannot: contracting `q4_deq` into a single `fma`
//! returns the same f32. A binary16 scale carries 11 significant bits and `q`
//! carries 4, so the product needs 15 and f32 offers 24. Over all 63,488
//! finite f16 scales and all 16 levels, 1,015,808 products, not one rounds, so
//! the rounding `fma` skips is a no-op. That is an equivalent mutant with a
//! proof, not a hole in the gate. The kernel still writes the two roundings,
//! because the equality is a property of the operand ranges.
//!
//! The barrier mutant is a race, so its kill is a fact about this machine
//! rather than a guarantee. It failed five of the seven GPU tests and passed
//! two, which is what a race looks like from the outside. Treat it as killed
//! here and argued from the memory model everywhere else, the way `matvec.cu`
//! argues it.
//!
//! ## What it does not cover
//!
//! Nothing runs through candle. The shipped Metal adapter must reach this
//! kernel through `CustomOp1::metal_fwd` and objc2-metal; this file drives it
//! through `llvq-metal`'s own metal-rs host layer, which proves the KERNEL and
//! says nothing about the binding. Nothing here reads a `.llvq` file either:
//! the fixture is synthetic, so the packer is exercised by
//! `llvq-llm/tests/proj_q4.rs` and not by this file.
//!
//! Runs on any Mac, needs no model and no artifact.

#![cfg(target_os = "macos")]

use llvq_artifact::{Int4Matrix, INT4G128_BITS, INT4G128_GROUP};
use llvq_core::SplitMix64;
use llvq_metal::{f16_bits, Kernel};

const SOURCE: &str = include_str!("../../llvq-llm/kernels/tv_q4_h.metal");

/// Apple's SIMD group is 32 lanes wide, and the kernel gives one to a row.
const LANES: usize = 32;

/// Threads a threadgroup, so eight rows each, as the CUDA host uses.
const GROUP: usize = 256;

/// `d_in` of `v_proj` on the served 4B, the shape this kernel exists for.
const SERVED_D_IN: usize = 2560;

/// Threadgroup memory this machine offers, asked rather than assumed.
///
/// An M3 Max answers 32,768 B (*measured*, 2026-09-21). The kernel stages the
/// whole activation, so the limit is a refusal boundary and not a slow path,
/// and a hard-coded number would make the gate pass on a device where the
/// served shape does not fit.
fn threadgroup_limit() -> usize {
    metal::Device::system_default().expect("no Metal device").max_threadgroup_memory_length()
        as usize
}

/// An int4 g128 record with adversarial groups, and the activation to hit it
/// with.
///
/// The nibbles are random bytes. Every byte is two valid levels, so this draws
/// the whole range 0..16 many times over, which is what a reversed nibble
/// index needs in order to show.
///
/// Every group gets its own scale and bias, drawn far enough apart that a
/// group index off by one moves the answer. One group in seven is UNGRADED:
/// scale exactly 1.0 and every `q` zero, which is the branch
/// `embedquant::quantize_affine` takes when a group is constant or its range
/// rounds to zero in f16. That branch exists on disk, so it belongs in a
/// fixture rather than in an argument about reachability.
fn fixture(seed: u64, d_out: usize, d_in: usize) -> (Int4Matrix, Vec<f32>) {
    assert!(d_in.is_multiple_of(INT4G128_GROUP), "the reader refuses a short group");
    let mut rng = SplitMix64::new(seed);
    let gpr = d_in / INT4G128_GROUP;
    let mut packed: Vec<u8> = (0..d_out * d_in / 2).map(|_| rng.next() as u8).collect();
    let mut scales = Vec::with_capacity(d_out * gpr);
    let mut biases = Vec::with_capacity(d_out * gpr);
    for row in 0..d_out {
        for g in 0..gpr {
            let ungraded = (row * gpr + g).is_multiple_of(7);
            let s = match ungraded {
                true => 1.0,
                // Spread over two decades, so a neighbouring group's scale is
                // never a near miss for this one's.
                false => 0.002 + 0.08 * rng.next_f64() as f32,
            };
            scales.push(f16_bits(s));
            biases.push(f16_bits(rng.next_gaussian() as f32 * 0.05));
            if ungraded {
                // The bias carries the value, so every level is zero.
                let lo = (row * d_in + g * INT4G128_GROUP) / 2;
                packed[lo..lo + INT4G128_GROUP / 2].fill(0);
            }
        }
    }
    let m = Int4Matrix {
        name: "model.layers.0.self_attn.v_proj.weight".into(),
        d_out,
        d_in,
        bits: INT4G128_BITS,
        group: INT4G128_GROUP,
        packed,
        scales,
        biases,
    };
    let x = (0..d_in).map(|_| rng.next_gaussian() as f32).collect();
    (m, x)
}

/// The kernel's summation order over the artifact crate's weights.
///
/// The weights are `Int4Matrix::to_f32`, untouched. The ORDER is the CUDA
/// original's: lane `l` takes words `l, l + 32, ...`, eight columns a word in
/// ascending order with `fma`, then the shuffle-xor butterfly.
fn reference(m: &Int4Matrix, x: &[f32]) -> Vec<f32> {
    let w = m.to_f32();
    let nwords = m.d_in / 8;
    let mut y = vec![0f32; m.d_out];
    for (row, out) in y.iter_mut().enumerate() {
        let mut lanes = [0f32; LANES];
        for (lane, acc) in lanes.iter_mut().enumerate() {
            let mut wi = lane;
            while wi < nwords {
                let c = wi * 8;
                let wr = &w[row * m.d_in + c..row * m.d_in + c + 8];
                for (wv, xv) in wr.iter().zip(&x[c..c + 8]) {
                    *acc = wv.mul_add(*xv, *acc);
                }
                wi += LANES;
            }
        }
        // `__shfl_xor` lane for lane. Every lane holds the total after the
        // last round; the kernel stores lane 0.
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

fn run_on_metal(m: &Int4Matrix, x: &[f32]) -> Vec<f32> {
    run_with(m, x, true)
}

/// `exact` chooses whether Metal's fast math is off. The answer must not
/// depend on it; `the_same_source_gives_the_same_numbers_with_fast_math_either_way`
/// is what says so.
fn run_with(m: &Int4Matrix, x: &[f32], exact: bool) -> Vec<f32> {
    let k = match exact {
        true => Kernel::new_exact(SOURCE, "tv_q4_metal"),
        false => Kernel::new(SOURCE, "tv_q4_metal"),
    }
    .expect("the int4 matvec compiles");

    // Little-endian, so byte `b` of the stream is bits `8b` of the word and
    // nibble `i` stays at `4 * (i % 8)`. A big-endian view here would
    // transpose every pair of columns and produce plausible, wrong weights.
    let words: Vec<u32> = m
        .packed
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_eq!(words.len() * 4, m.packed.len(), "the stream is whole words");

    let b_wq = k.buffer(&words);
    let b_sc = k.buffer(&m.scales);
    let b_bi = k.buffer(&m.biases);
    let b_x = k.buffer(x);
    let b_y = k.empty::<f32>(m.d_out);

    let d_in = m.d_in as u32;
    let gpr = m.groups_per_row() as u32;
    // The whole activation, no tile. Omitting this call is not an error: the
    // kernel would run and write zeros.
    let tg_bytes = m.d_in * 4;
    let limit = threadgroup_limit();
    assert!(tg_bytes <= limit, "staging {} values needs {tg_bytes} B of {limit}", m.d_in);

    k.dispatch((m.d_out * LANES) as u64, GROUP as u64, |enc| {
        enc.set_buffer(0, Some(&b_wq), 0);
        enc.set_buffer(1, Some(&b_sc), 0);
        enc.set_buffer(2, Some(&b_bi), 0);
        enc.set_buffer(3, Some(&b_x), 0);
        enc.set_buffer(4, Some(&b_y), 0);
        enc.set_bytes(5, 4, &d_in as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(6, 4, &gpr as *const u32 as *const std::ffi::c_void);
        enc.set_threadgroup_memory_length(0, tg_bytes as u64);
    });

    unsafe { k.read::<f32>(&b_y, m.d_out) }
}

fn expect_equal(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(g, w, "{what}, row {i}: metal {g} against host {w}");
    }
}

/// Two groups a row, so the group index is under test, and 32 words a row, so
/// every lane takes exactly one.
#[test]
fn the_int4_matvec_matches_the_repository_dequantization() {
    let (m, x) = fixture(0x4A_0001, 64, 2 * INT4G128_GROUP);
    expect_equal(&run_on_metal(&m, &x), &reference(&m, &x), "two groups");
}

/// The served width: `v_proj` of the 4B is 2560 wide, so 20 groups a row and
/// 320 words, ten to a lane. This is the shape the shared staging was sized
/// for, 10,240 B.
#[test]
fn the_matvec_matches_at_the_served_width() {
    let (m, x) = fixture(0x4A_0002, 64, SERVED_D_IN);
    expect_equal(&run_on_metal(&m, &x), &reference(&m, &x), "served width");
}

/// One group a row is 16 words, so HALF the lanes take no word at all and
/// reach the butterfly with a zero accumulator. A reduction that assumed every
/// lane contributed would show here.
#[test]
fn the_matvec_matches_when_half_the_lanes_take_no_word() {
    let (m, x) = fixture(0x4A_0003, 32, INT4G128_GROUP);
    expect_equal(&run_on_metal(&m, &x), &reference(&m, &x), "idle lanes");
}

/// `d_out` not a multiple of eight, so the last threadgroup is PARTIAL.
///
/// The CUDA host refuses this shape, because a block that is not full would
/// need a bounds guard and a guard before `__syncthreads()` deadlocks. Metal
/// has no such problem: `dispatchThreads` launches exactly `d_out * 32`
/// threads and `tgs` reports the short group's real size. This test is what
/// turns that claim into a measurement.
#[test]
fn the_matvec_matches_on_a_partial_threadgroup() {
    let (m, x) = fixture(0x4A_0004, 12, 3 * INT4G128_GROUP);
    expect_equal(&run_on_metal(&m, &x), &reference(&m, &x), "partial group");
}

/// The result depends on the activation. A kernel that ignored `x` would pass
/// every equality above if the reference ignored it too.
#[test]
fn a_different_activation_gives_a_different_answer() {
    let (m, mut x) = fixture(0x4A_0005, 16, 2 * INT4G128_GROUP);
    let before = run_on_metal(&m, &x);
    for v in x.iter_mut() {
        *v += 1.0;
    }
    let after = run_on_metal(&m, &x);
    assert!(
        before.iter().zip(&after).any(|(p, q)| p != q),
        "the matvec must read the activation"
    );
    expect_equal(&after, &reference(&m, &x), "after the shift");
}

/// The result depends on the weights, and on WHICH row of them.
///
/// Swapping two rows of the record must swap the two outputs. A kernel that
/// dropped `row * nwords` would return the same number for every row and still
/// match a reference that made the same mistake; this test needs no reference
/// at all.
#[test]
fn swapping_two_rows_swaps_two_outputs() {
    let (m, x) = fixture(0x4A_0006, 16, INT4G128_GROUP);
    let before = run_on_metal(&m, &x);
    // `d_in` nibbles is `d_in / 2` bytes, so a row is that many bytes long.
    let (rb, gpr) = (m.d_in / 2, m.groups_per_row());
    let mut packed = m.packed.clone();
    for i in 0..rb {
        packed.swap(i, rb + i);
    }
    let mut scales = m.scales.clone();
    let mut biases = m.biases.clone();
    for g in 0..gpr {
        scales.swap(g, gpr + g);
        biases.swap(g, gpr + g);
    }
    let swapped = Int4Matrix {
        name: m.name.clone(),
        d_out: m.d_out,
        d_in: m.d_in,
        bits: m.bits,
        group: m.group,
        packed,
        scales,
        biases,
    };
    let after = run_on_metal(&swapped, &x);
    assert_eq!(after[0], before[1], "row 1 did not move to row 0");
    assert_eq!(after[1], before[0], "row 0 did not move to row 1");
    assert_ne!(before[0], before[1], "the fixture must give the rows different sums");
    expect_equal(&after, &reference(&swapped, &x), "after the swap");
}

/// The pragma, not the compile option, is what carries the arithmetic.
///
/// This matters because the shipped path will not use `new_exact`: a candle
/// adapter compiles the same source with `None` options, which is Metal's
/// default and has fast math ON. If the arithmetic depended on the option, the
/// gate and the shipped path would be two different kernels.
#[test]
fn the_same_source_gives_the_same_numbers_with_fast_math_either_way() {
    let (m, x) = fixture(0x4A_0007, 32, 2 * INT4G128_GROUP);
    let exact = run_with(&m, &x, true);
    let fast = run_with(&m, &x, false);
    let want = reference(&m, &x);
    for (i, ((a, b), w)) in exact.iter().zip(&fast).zip(&want).enumerate() {
        assert_eq!(a, b, "row {i}: fast math moved the answer, {a} against {b}");
        assert_eq!(a, w, "row {i}: and neither matches the host reference");
    }
}

/// The kernel's group is the format's group.
///
/// `llvq_llm::fused` makes the same assertion against the CUDA source. The two
/// kernels read the same records, so a drift on either side is a silent
/// misread, and the constant is the one place it can be caught.
#[test]
fn the_kernel_group_is_the_format_group() {
    assert_eq!(INT4G128_GROUP, 128, "the format moved under the kernel");
    assert_eq!(INT4G128_BITS, 4, "the width moved under the kernel");
    assert!(
        SOURCE.contains(&format!("#define LLVQ_Q4_GROUP {INT4G128_GROUP}u")),
        "the kernel's group and `INT4G128_GROUP` disagree"
    );
}

/// The whole activation fits this machine's threadgroup memory at the served
/// width: 10,240 B against 32,768 on an M3 Max.
///
/// The CUDA host checks the same thing against the card's limit at load time,
/// and states the reason: this kernel does not tile, so a wider `d_in` is a
/// refusal rather than a slower path.
#[test]
fn the_staged_activation_fits_apple_threadgroup_memory() {
    let limit = threadgroup_limit();
    assert!(
        SERVED_D_IN * 4 <= limit,
        "{} B of activation against {limit} B of threadgroup memory",
        SERVED_D_IN * 4
    );
}
