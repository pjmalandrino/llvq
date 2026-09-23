//! The q4 embedding path: the load-time rule and the CPU mirror of its two
//! device kernels.
//!
//! `tests/embed_q8.rs` at half the weight width. The chain it pins without a
//! card:
//!
//!  1. an f16 table under `EmbedMode::Q4` becomes exactly the bytes an
//!     independent, test-local implementation of the int4 g64 scheme produces
//!     (so a mutation inside `quantize_affine` or in the wiring fails), and a
//!     table already int4 g64, what `bin/embedq q4` wrote into the sealed file,
//!     passes through untouched;
//!  2. `e4_deq` and the gather, the exact text NVRTC compiles, executed on the
//!     CPU through `host_embq4.cpp`, reproduce `RawTensor::to_f32` bit for bit,
//!     through both the gather's and the matvec's nibble addressing;
//!  3. the matvec's lane-ordered accumulation stays within f32 rounding of an
//!     f64 recomputation.
//!
//! What only the card can validate: NVRTC compilation, spill, throughput.

use half::f16;
use llvq_artifact::{QuantData, RawData, RawTensor};
use llvq_llm::fused::{
    embed_q4, embed_q8, q4_device_bytes, take_embed_tables_at, EmbedMode, EmbedReport, EMBED_GROUP,
    EMBED_NAME, HEAD_NAME,
};
use std::io::Write;
use std::process::{Command, Stdio};

/// Deterministic f16 values with the spread of real embedding weights, and a
/// zero every 97 values so the ungraded branch is reached.
fn test_tensor(name: &str, rows: usize, row_len: usize, seed: u64) -> RawTensor {
    let mut s = seed;
    let data: Vec<u16> = (0..rows * row_len)
        .map(|i| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((s >> 33) as f64 / (1u64 << 31) as f64) - 0.5;
            if i % 97 == 0 {
                f16::ZERO.to_bits()
            } else {
                f16::from_f64(u * 0.08).to_bits()
            }
        })
        .collect();
    RawTensor {
        name: name.into(),
        dims: vec![rows, row_len],
        data: RawData::F16(data),
    }
}

/// The int4 g64 scheme written against its spec, not a call into
/// `embedquant`: min and max per group, scale and bias rounded to f16 first,
/// `q` graded against the rounded pair, nibbles low first at the global flat
/// index.
fn reference_quantize(t: &RawTensor) -> (Vec<u8>, Vec<u16>, Vec<u16>) {
    let RawData::F16(data) = &t.data else {
        panic!("not f16")
    };
    let row_len = *t.dims.last().unwrap();
    let rows = t.len() / row_len;
    let gpr = row_len.div_ceil(64);
    let (mut packed, mut scales, mut biases) =
        (vec![0u8; t.len().div_ceil(2)], Vec::new(), Vec::new());
    for r in 0..rows {
        for g in 0..gpr {
            let lo = r * row_len + g * 64;
            let hi = r * row_len + ((g + 1) * 64).min(row_len);
            let vals: Vec<f32> = data[lo..hi]
                .iter()
                .map(|&b| f16::from_bits(b).to_f32())
                .collect();
            let mn = vals.iter().cloned().fold(f32::INFINITY, f32::min);
            let mx = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let s = f16::from_f32((mx - mn) / 15.0);
            let graded = s.to_f32() > 0.0;
            let s = if graded { s } else { f16::ONE };
            let b = f16::from_f32(mn);
            scales.push(s.to_bits());
            biases.push(b.to_bits());
            for (k, &v) in vals.iter().enumerate() {
                let q = if graded {
                    ((v - b.to_f32()) / s.to_f32()).round().clamp(0.0, 15.0) as u8
                } else {
                    0
                };
                let i = lo + k;
                packed[i / 2] |= q << (4 * (i % 2));
            }
        }
    }
    (packed, scales, biases)
}

fn as_q4(t: &RawTensor) -> &QuantData {
    let RawData::Quant(q) = &t.data else {
        panic!("not quantized")
    };
    assert_eq!(
        (q.bits, q.group),
        (4, EMBED_GROUP),
        "not the int4 g64 scheme"
    );
    q
}

#[test]
fn the_load_path_writes_the_spec_bytes() {
    // A short last group (224 = 3·64 + 32), and d % 8 == 0 as the kernels need.
    let t = test_tensor(EMBED_NAME, 40, 224, 7);
    let (p, s, b) = reference_quantize(&t);
    let q = embed_q4(t).expect("quantize");
    let q = as_q4(&q);
    assert_eq!(q.packed, p, "packed nibbles");
    assert_eq!(q.scales, s, "scales");
    assert_eq!(q.biases, b, "biases");
}

#[test]
fn a_q4_table_passes_through_and_other_widths_are_refused() {
    let q = embed_q4(test_tensor(EMBED_NAME, 8, 128, 3)).expect("quantize");
    let bytes = as_q4(&q).packed.clone();
    let through = embed_q4(q).expect("passthrough");
    assert_eq!(
        as_q4(&through).packed,
        bytes,
        "a q4 table must come back byte-identical"
    );

    let q8 = embed_q8(test_tensor(EMBED_NAME, 8, 128, 3)).expect("q8");
    let e = match embed_q4(q8) {
        Ok(_) => panic!("an int8 table must be refused on the q4 path"),
        Err(e) => e,
    };
    assert!(e.contains("int8 g64") && e.contains("q4 path"), "{e}");
    let q4 = embed_q4(test_tensor(EMBED_NAME, 8, 128, 3)).expect("q4");
    assert!(
        embed_q8(q4).is_err(),
        "the q8 path must not requantize an int4 table"
    );
    let g32 =
        llvq_llm::embedquant::quantize_affine(&test_tensor(EMBED_NAME, 8, 128, 3), 4, 32).unwrap();
    assert!(embed_q4(g32).is_err(), "group 32 must be refused");
}

#[test]
fn the_mode_parses_and_counts_its_bytes() {
    assert_eq!(EmbedMode::parse(Some("q4")).unwrap(), EmbedMode::Q4);
    assert!(EmbedMode::parse(Some("Q4")).is_err(), "case must be exact");
    assert!(EmbedMode::parse(Some("int4")).is_err());
    assert_eq!(EmbedMode::Q4.name(), "q4");
    assert_eq!(
        [EmbedMode::F16, EmbedMode::Q8, EmbedMode::Q4].map(|m| m.bits()),
        [None, Some(8), Some(4)]
    );
    // The served 4B: 151,936 × 2,560 nibbles and one f16 pair per 64.
    assert_eq!(q4_device_bytes(&[151_936, 2560]), (194_478_080, 24_309_760));
}

#[test]
fn the_tables_come_off_at_four_bits() {
    let mut raw = vec![
        test_tensor("model.norm.weight", 1, 64, 1),
        test_tensor(EMBED_NAME, 16, 128, 21),
    ];
    let tied = take_embed_tables_at(&mut raw, true, 4).expect("tied");
    assert!(tied.head.is_none());
    as_q4(&tied.embed);
    assert_eq!(raw.len(), 1, "the embedding left the carried list");

    let mut raw = vec![
        test_tensor(EMBED_NAME, 16, 128, 21),
        test_tensor(HEAD_NAME, 16, 128, 77),
    ];
    let untied = take_embed_tables_at(&mut raw, false, 4).expect("untied");
    let head = untied.head.as_ref().expect("a separate head");
    assert_ne!(
        as_q4(&untied.embed).packed,
        as_q4(head).packed,
        "two tables, two payloads"
    );

    let r = EmbedReport::new(EmbedMode::Q4, &untied.buffers());
    let (p, sb) = q4_device_bytes(&[16, 128]);
    assert_eq!((r.packed(), r.meta()), (2 * p, 2 * sb));
    let line = r.line("LLVQ_EMBED");
    assert!(
        line.starts_with("embedding: q4 g64 (LLVQ_EMBED), 2 tables"),
        "{line}"
    );
    assert!(line.contains("int4"), "{line}");
}

fn build_host_probe() -> std::path::PathBuf {
    let out = std::env::temp_dir().join("llvq_host_embq4");
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests");
    let st = Command::new("clang++")
        .args([
            "-std=c++17",
            "-O2",
            "-ffp-contract=off",
            "-Wall",
            "-Wextra",
            "-Werror",
        ])
        .arg(dir.join("host_embq4.cpp"))
        .arg("-o")
        .arg(&out)
        .status()
        .expect("clang++ is on PATH");
    assert!(
        st.success(),
        "the q4 embedding kernels do not compile as host C++"
    );
    out
}

/// Run the probe: `(gather f16 bits, f32 dequant, per-row dot against x)`.
fn run_host_probe(
    exe: &std::path::Path,
    t: &RawTensor,
    ids: &[u32],
    x: &[u16],
) -> (Vec<u16>, Vec<f32>, Vec<f32>) {
    let d = *t.dims.last().expect("2-D");
    let rows = t.len() / d;
    let gpr = d.div_ceil(EMBED_GROUP);
    let q = as_q4(t);
    let mut fixture = Vec::new();
    for v in [rows as u32, d as u32, gpr as u32, ids.len() as u32] {
        fixture.extend_from_slice(&v.to_le_bytes());
    }
    fixture.extend_from_slice(&q.packed);
    for v in q.scales.iter().chain(&q.biases) {
        fixture.extend_from_slice(&v.to_le_bytes());
    }
    for v in ids {
        fixture.extend_from_slice(&v.to_le_bytes());
    }
    for v in x {
        fixture.extend_from_slice(&v.to_le_bytes());
    }
    let mut child = Command::new(exe)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("host probe runs");
    child
        .stdin
        .take()
        .expect("stdin")
        .write_all(&fixture)
        .expect("fixture written");
    let res = child.wait_with_output().expect("host probe finishes");
    assert!(res.status.success(), "host probe failed");

    let mut r = res.stdout.as_slice();
    let mut take = |k: usize, w: usize| -> Vec<u8> {
        let (a, b) = r.split_at(k * w);
        r = b;
        a.to_vec()
    };
    let gather = take(ids.len() * d, 2)
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    let deq = take(rows * d, 4)
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let dot = take(rows, 4)
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert!(
        r.is_empty(),
        "the host probe wrote more than it was asked to"
    );
    (gather, deq, dot)
}

#[test]
fn the_kernel_dequant_decides_what_the_reference_decides() {
    let exe = build_host_probe();
    let (rows, d) = (96usize, 224usize);
    let t = embed_q4(test_tensor(EMBED_NAME, rows, d, 11)).expect("quantize");
    let reference = t.to_f32();
    let ids: Vec<u32> = vec![0, 95, 3, 3, 17, 64, 90, 1];
    let mut s = 0xE1Bu64;
    let x: Vec<u16> = (0..d)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            f16::from_f64((((s >> 33) as f64 / (1u64 << 31) as f64) - 0.5) * 2.0).to_bits()
        })
        .collect();
    let (gather, deq, dot) = run_host_probe(&exe, &t, &ids, &x);

    for (i, (&got, &want)) in deq.iter().zip(&reference).enumerate() {
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "weight {i}: kernel {got} vs reference {want}"
        );
    }
    for (ti, &id) in ids.iter().enumerate() {
        for c in 0..d {
            let want = f16::from_f32(reference[id as usize * d + c]).to_bits();
            assert_eq!(gather[ti * d + c], want, "token {ti} (row {id}), col {c}");
        }
    }
    let xf: Vec<f64> = x.iter().map(|&b| f16::from_bits(b).to_f64()).collect();
    let mut worst = 0.0f64;
    for row in 0..rows {
        let (mut want, mut scale) = (0.0f64, 0.0f64);
        for c in 0..d {
            let term = reference[row * d + c] as f64 * xf[c];
            want += term;
            scale += term.abs();
        }
        worst = worst.max((dot[row] as f64 - want).abs() / scale.max(1e-12));
    }
    assert!(worst < 1e-6, "worst dot error {worst:.2e}·Σ|w·x|");
}
