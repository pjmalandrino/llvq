//! `tv_q4_h`, the mixed-precision arm of Q5, checked on the development
//! machine.
//!
//! The kernel serves one projection as affine int4 g128 while the other 251
//! stay on the Leech path. Two things must hold before a card ever sees it,
//! and both are provable here:
//!
//!  1. **it decodes what the file says.** `q4_deq`, executed through the
//!     clang++ harness, must reproduce `llvq_artifact::RawTensor::to_f32`
//!     **bit for bit** on every weight — not within a tolerance. The
//!     addressing is where a mixed-precision path fails silently: the packer
//!     writes nibbles at the *global* flat index and the kernel reads them
//!     through a u32 pointer, so a wrong nibble order gives weights that load,
//!     run, and give numbers. `ops/awq_dequant.py` carries the same hazard
//!     under the name `AWQ_REVERSE_ORDER`;
//!  2. **it accumulates like the family.** The row dot, mirrored in the
//!     kernel's lane order with its warp butterfly, must match an f64
//!     reference over the reader's own weights.
//!
//! What is NOT proved here: the launch geometry, the shared staging and the
//! `__syncthreads()` the harness cannot reproduce. Those are an open claim
//! until a job compares this path's greedy tokens against the f16 arm — the
//! same standing caveat `tv_planes_seg_h` carries at the top of its source.

use std::io::Write;
use std::process::{Command, Stdio};

use half::f16;
use llvq_artifact::{RawData, RawTensor};
use llvq_llm::embedquant::quantize_affine;

/// The format's group width. Not a knob: the kernel hardcodes it and the host
/// refuses any other value, so a test that quantized at another width would be
/// measuring a scheme nothing serves.
const Q4_GROUP: usize = 128;

/// A projection-shaped f16 matrix, with the adversarial values that reach the
/// quantizer's two branches: zeros, and a constant run whose range rounds to
/// zero in f16 so the ungraded path (`s = 1`, every `q = 0`, the bias carrying
/// the value) is exercised rather than assumed unreachable.
fn test_matrix(d_out: usize, d_in: usize, seed: u64) -> RawTensor {
    let mut s = seed;
    let data: Vec<u16> = (0..d_out * d_in)
        .map(|i| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = ((s >> 33) as f64 / (1u64 << 31) as f64) - 0.5;
            if i % 97 == 0 {
                f16::ZERO.to_bits()
            } else if (Q4_GROUP..2 * Q4_GROUP).contains(&(i % d_in)) && i / d_in == 1 {
                // A whole group of one row held constant.
                f16::from_f64(0.125).to_bits()
            } else {
                f16::from_f64(u * 0.08).to_bits()
            }
        })
        .collect();
    RawTensor {
        name: "model.layers.0.self_attn.v_proj.weight".into(),
        dims: vec![d_out, d_in],
        data: RawData::F16(data),
    }
}

fn as_quant(t: &RawTensor) -> (&Vec<u8>, &Vec<u16>, &Vec<u16>) {
    let RawData::Quant(q) = &t.data else { panic!("not quantized") };
    assert_eq!((q.bits, q.group), (4, Q4_GROUP), "not the served scheme");
    (&q.packed, &q.scales, &q.biases)
}

/// Compile the CPU driver once per test that needs it.
///
/// `tag` is not decoration: cargo runs these tests as parallel threads of one
/// process, so a shared output path lets one test execute the binary another
/// is still writing. That failure is intermittent and reads as an arithmetic
/// fault, which is the worst way to spend an afternoon.
fn build_host_probe(tag: &str) -> std::path::PathBuf {
    let out = std::env::temp_dir().join(format!("llvq_host_tv_q4_h_{tag}"));
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests");
    let st = Command::new("clang++")
        .args(["-std=c++17", "-O2", "-ffp-contract=off", "-Wall", "-Wextra", "-Werror"])
        .arg(dir.join("host_tv_q4_h.cpp"))
        .arg("-o")
        .arg(&out)
        .status()
        .expect("clang++ is on PATH");
    assert!(st.success(), "tv_q4_h does not compile as host C++");
    out
}

/// `(dequant of every weight, per-row dot against `x`)` — what the device
/// would produce from *these* bytes and no others.
fn run_host_probe(
    exe: &std::path::Path,
    packed: &[u8],
    scales: &[u16],
    biases: &[u16],
    d_out: usize,
    d_in: usize,
    x: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let gpr = d_in.div_ceil(Q4_GROUP);
    let mut fixture = Vec::new();
    for v in [d_out as u32, d_in as u32, gpr as u32] {
        fixture.extend_from_slice(&v.to_le_bytes());
    }
    // The packer's bytes, read back as the u32 words the kernel loads.
    for c in packed.chunks_exact(4) {
        fixture.extend_from_slice(c);
    }
    for v in scales.iter().chain(biases.iter()) {
        fixture.extend_from_slice(&v.to_le_bytes());
    }
    for v in x {
        fixture.extend_from_slice(&v.to_le_bytes());
    }

    let mut ch = Command::new(exe)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("the harness runs");
    ch.stdin.take().expect("piped").write_all(&fixture).expect("fixture written");
    let out = ch.wait_with_output().expect("the harness exits");
    assert!(out.status.success(), "the harness failed");

    let want = (d_out * d_in + d_out) * 4;
    assert_eq!(out.stdout.len(), want, "the harness wrote {} bytes, expected {want}", out.stdout.len());
    let f32s: Vec<f32> = out
        .stdout
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let (deq, dot) = f32s.split_at(d_out * d_in);
    (deq.to_vec(), dot.to_vec())
}

fn fixture(d_out: usize, d_in: usize, seed: u64) -> (RawTensor, Vec<f32>) {
    let t = test_matrix(d_out, d_in, seed);
    let q = quantize_affine(&t, 4, Q4_GROUP).expect("the shipped quantizer");
    let mut s = seed ^ 0x5deece66d;
    let x: Vec<f32> = (0..d_in)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            (((s >> 33) as f64 / (1u64 << 31) as f64) - 0.5) as f32
        })
        .collect();
    (q, x)
}

#[test]
fn the_kernel_dequantizes_exactly_like_the_reader() {
    let (d_out, d_in) = (16, 256);
    let (q, x) = fixture(d_out, d_in, 0xc0ffee);
    let (packed, scales, biases) = as_quant(&q);
    let exe = build_host_probe("deq");
    let (deq, _) = run_host_probe(&exe, packed, scales, biases, d_out, d_in, &x);

    let want = q.to_f32();
    assert_eq!(deq.len(), want.len());
    let bad = deq
        .iter()
        .zip(&want)
        .enumerate()
        .filter(|(_, (a, b))| a.to_bits() != b.to_bits())
        .count();
    assert_eq!(bad, 0, "{bad} of {} weights differ from the reader, bit for bit", want.len());
}

#[test]
fn the_row_dot_matches_an_f64_reference() {
    let (d_out, d_in) = (16, 256);
    let (q, x) = fixture(d_out, d_in, 0xbadc0de);
    let (packed, scales, biases) = as_quant(&q);
    let exe = build_host_probe("dot");
    let (_, dot) = run_host_probe(&exe, packed, scales, biases, d_out, d_in, &x);

    let w = q.to_f32();
    for r in 0..d_out {
        let terms = (0..d_in).map(|c| w[r * d_in + c] as f64 * x[c] as f64);
        let want: f64 = terms.clone().sum();
        let abs_sum: f64 = terms.map(f64::abs).sum();
        let got = dot[r] as f64;
        // The bound is on the sum of the magnitudes, not on the sum. A row dot
        // cancels — here to a thousandth of what it accumulates — so scaling
        // the tolerance to `want` would demand of f32 a precision no
        // accumulation order can deliver, and would tighten as the row got
        // closer to zero. Depth is 8 sequential f32 fma per lane then a
        // 5-level butterfly, 13 roundings; 16 ULP of 1.0 covers it with margin
        // and still leaves any real addressing fault, which moves a dot by a
        // fraction of `abs_sum`, far outside.
        let tol = 16.0 * f64::from(f32::EPSILON) * abs_sum;
        assert!(
            (got - want).abs() <= tol,
            "row {r}: kernel {got:e} against f64 reference {want:e}, \
             error {:e} over a tolerance of {tol:e} on an accumulated {abs_sum:e}",
            (got - want).abs()
        );
    }
}

/// The bit-for-bit check above is only worth what a wrong addressing costs it.
/// Swapping the two nibbles of every packed byte is exactly the mistake the
/// AWQ port documents, and it must move the dequantized weights — otherwise
/// the first test would pass on a kernel that reads the stream backwards.
#[test]
fn swapping_the_nibbles_is_caught() {
    let (d_out, d_in) = (16, 256);
    let (q, x) = fixture(d_out, d_in, 0xfeedface);
    let (packed, scales, biases) = as_quant(&q);
    let exe = build_host_probe("nibbles");
    let (straight, _) = run_host_probe(&exe, packed, scales, biases, d_out, d_in, &x);

    let swapped: Vec<u8> = packed.iter().map(|b| b.rotate_left(4)).collect();
    let (mutated, _) = run_host_probe(&exe, &swapped, scales, biases, d_out, d_in, &x);

    let moved = straight
        .iter()
        .zip(&mutated)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    assert!(
        moved > straight.len() / 4,
        "swapping every nibble moved only {moved} of {} weights: the addressing check is not lethal",
        straight.len()
    );
}
