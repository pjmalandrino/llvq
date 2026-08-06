//! The q8 embedding path — the load-time quantization and the CPU mirror of
//! the two device kernels.
//!
//! What transfers lot B's quality verdict to `LLVQ_EMBED=q8` is a chain of
//! identities, each pinned here without a card:
//!
//!  1. the load path produces **bit-identical** bytes to `bin/embedq` — by
//!     calling the same function, checked against an independent test-local
//!     reimplementation of the MLX scheme so a mutation in either the shared
//!     function or the wiring (group width, bit width) fails loudly;
//!  2. the kernels' dequant — `q8_deq`, the exact text NVRTC compiles,
//!     executed on the CPU through the clang++ harness — reproduces
//!     `RawTensor::to_f32` bit for bit, and the gather's f16 store reproduces
//!     `f16::from_f32` of it bit for bit;
//!  3. when the sealed artifacts are on disk, sampled real rows quantized by
//!     the load path match the bytes `bin/embedq` actually wrote into
//!     `~/q4b-e8.llvq` — the file lot B scored.
//!
//! What only the card can validate: NVRTC compilation of the two kernels,
//! spill (asserted at runtime startup), and throughput.

use half::f16;
use llvq_artifact::{RawData, RawTensor};
use llvq_llm::embedquant::quantize_affine;
use llvq_llm::fused::{embed_q8, q8_device_bytes, EmbedMode, EMBED_GROUP};
use std::io::Write;
use std::process::{Command, Stdio};

/// Deterministic f16 values with the spread of real embedding weights.
fn test_tensor(rows: usize, row_len: usize, seed: u64) -> RawTensor {
    let mut s = seed;
    let data: Vec<u16> = (0..rows * row_len)
        .map(|i| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = ((s >> 33) as f64 / (1u64 << 31) as f64) - 0.5;
            // A few adversarial values among the noise: zeros and a constant
            // group head, so the ungraded (s = 1, all-bias) branch is hit.
            if i % 97 == 0 {
                f16::ZERO.to_bits()
            } else {
                f16::from_f64(u * 0.08).to_bits()
            }
        })
        .collect();
    RawTensor {
        name: "model.embed_tokens.weight".into(),
        dims: vec![rows, row_len],
        data: RawData::F16(data),
    }
}

/// An independent implementation of the MLX q8 g64 scheme, written against
/// the *spec* (min/max per group, scale and bias rounded to f16 first, `q`
/// graded against the rounded values). Deliberately not a call into
/// `embedquant`: this is what makes the bit-comparison below able to kill a
/// mutant *inside* the shared function, not only in the wiring around it.
fn reference_quantize(t: &RawTensor) -> (Vec<u8>, Vec<u16>, Vec<u16>) {
    let RawData::F16(data) = &t.data else { panic!("not f16") };
    let row_len = *t.dims.last().unwrap();
    let rows = t.len() / row_len;
    let gpr = row_len.div_ceil(64);
    let (mut packed, mut scales, mut biases) = (vec![0u8; t.len()], Vec::new(), Vec::new());
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
            let s = f16::from_f32((mx - mn) / 255.0);
            let graded = s.to_f32() > 0.0;
            let s = if graded { s } else { f16::ONE };
            let b = f16::from_f32(mn);
            scales.push(s.to_bits());
            biases.push(b.to_bits());
            for (k, &v) in vals.iter().enumerate() {
                packed[lo + k] = if graded {
                    ((v - b.to_f32()) / s.to_f32()).round().clamp(0.0, 255.0) as u8
                } else {
                    0
                };
            }
        }
    }
    (packed, scales, biases)
}

fn as_quant(t: &RawTensor) -> (&Vec<u8>, &Vec<u16>, &Vec<u16>) {
    let RawData::Quant(q) = &t.data else { panic!("not quantized") };
    assert_eq!((q.bits, q.group), (8, EMBED_GROUP), "not the validated scheme");
    (&q.packed, &q.scales, &q.biases)
}

/// The load-time quantization is bit-identical to `bin/embedq`'s — same
/// packed bytes, same scales, same biases — and both match the independent
/// reference. Two widths: an exact multiple of 64 and a short last group.
#[test]
fn load_path_quantization_is_bit_identical_to_embedq() {
    for (rows, d, seed) in [(7usize, 192usize, 1u64), (5, 224, 2), (3, 130, 3)] {
        let via_load = embed_q8(test_tensor(rows, d, seed)).expect("load path");
        let via_embedq = quantize_affine(&test_tensor(rows, d, seed), 8, 64).expect("embedq");
        let (lp, ls, lb) = as_quant(&via_load);
        let (ep, es, eb) = as_quant(&via_embedq);
        assert_eq!((lp, ls, lb), (ep, es, eb), "{rows}×{d}: load path vs embedq");

        let (rp, rs, rb) = reference_quantize(&test_tensor(rows, d, seed));
        assert_eq!(lp, &rp, "{rows}×{d}: packed vs independent reference");
        assert_eq!(ls, &rs, "{rows}×{d}: scales vs independent reference");
        assert_eq!(lb, &rb, "{rows}×{d}: biases vs independent reference");
    }
}

/// A tensor already stored as int8 g64 — an `embedq` output — passes through
/// byte-identical; any other quantized form is refused, not requantized.
#[test]
fn stored_q8_passes_through_and_nothing_else_does() {
    let q = quantize_affine(&test_tensor(4, 128, 7), 8, 64).expect("quantize");
    let (p0, s0, b0) = {
        let (p, s, b) = as_quant(&q);
        (p.clone(), s.clone(), b.clone())
    };
    let through = embed_q8(q).expect("passthrough");
    let (p1, s1, b1) = as_quant(&through);
    assert_eq!((&p0, &s0, &b0), (p1, s1, b1), "passthrough changed bytes");

    let q4 = quantize_affine(&test_tensor(4, 128, 8), 4, 64).expect("quantize int4");
    assert!(embed_q8(q4).is_err(), "int4 must be refused, not requantized");
    let g32 = quantize_affine(&test_tensor(4, 128, 9), 8, 32).expect("quantize g32");
    assert!(embed_q8(g32).is_err(), "group 32 must be refused");
}

/// `LLVQ_EMBED` parses exactly: default f16, explicit both, typo refused.
#[test]
fn embed_mode_parses_exactly() {
    assert_eq!(EmbedMode::parse(None).unwrap(), EmbedMode::F16);
    assert_eq!(EmbedMode::parse(Some("")).unwrap(), EmbedMode::F16);
    assert_eq!(EmbedMode::parse(Some("f16")).unwrap(), EmbedMode::F16);
    assert_eq!(EmbedMode::parse(Some("q8")).unwrap(), EmbedMode::Q8);
    assert!(EmbedMode::parse(Some("q08")).is_err(), "a typo must not default");
    assert!(EmbedMode::parse(Some("Q8")).is_err(), "case must be exact");
}

/// The announced footprint on Qwen3-4B: 388.96 MB of int8 + 24.31 MB of
/// scales/biases = 413.3 MB, against 778.1 MB at f16.
#[test]
fn the_4b_footprint_is_the_announced_413_mb() {
    let dims = [151_936usize, 2560];
    let (packed, sb) = q8_device_bytes(&dims);
    assert_eq!(packed, 388_956_160);
    assert_eq!(sb, 24_309_760);
    assert_eq!(packed + sb, 413_265_920);
    let f16_bytes = (dims[0] * dims[1] * 2) as u64;
    assert_eq!(f16_bytes, 777_912_320);
    assert!(f16_bytes - (packed + sb) > 364_000_000, "the −365 MB claim");
}

/// The kernels' dequant, executed on the CPU through the same text NVRTC
/// compiles, against llvq-artifact's own dequantizer — bit for bit on the
/// f32 dequant, bit for bit on the gather's f16 store, and the matvec's
/// lane-ordered accumulation against an f64 recomputation.
#[test]
fn the_kernel_dequant_decides_what_the_reference_decides() {
    let out = std::env::temp_dir().join("llvq_host_embq8");
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests");
    let st = Command::new("clang++")
        .args(["-std=c++17", "-O2", "-ffp-contract=off", "-Wall", "-Wextra", "-Werror"])
        .arg(dir.join("host_embq8.cpp"))
        .arg("-o")
        .arg(&out)
        .status()
        .expect("clang++ is on PATH");
    assert!(st.success(), "the q8 embedding kernels do not compile as host C++");

    // ---- fixture: short last group (224 = 3·64 + 32), 96 rows ----
    let (rows, d) = (96usize, 224usize);
    let gpr = d.div_ceil(64);
    let t = embed_q8(test_tensor(rows, d, 11)).expect("quantize");
    let reference = t.to_f32();
    let (packed, scales, biases) = {
        let (p, s, b) = as_quant(&t);
        (p.clone(), s.clone(), b.clone())
    };
    let words: Vec<u32> = packed
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let ids: Vec<u32> = vec![0, 95, 3, 3, 17, 64, 90, 1];
    let mut s = 0xE1B_u64;
    let x: Vec<u16> = (0..d)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = ((s >> 33) as f64 / (1u64 << 31) as f64) - 0.5;
            f16::from_f64(u * 2.0).to_bits()
        })
        .collect();

    let mut fixture = Vec::new();
    for v in [rows as u32, d as u32, gpr as u32, ids.len() as u32] {
        fixture.extend_from_slice(&v.to_le_bytes());
    }
    for w in &words {
        fixture.extend_from_slice(&w.to_le_bytes());
    }
    for v in scales.iter().chain(&biases) {
        fixture.extend_from_slice(&v.to_le_bytes());
    }
    for v in &ids {
        fixture.extend_from_slice(&v.to_le_bytes());
    }
    for v in &x {
        fixture.extend_from_slice(&v.to_le_bytes());
    }

    let mut child = Command::new(&out)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("host probe runs");
    child.stdin.take().expect("stdin").write_all(&fixture).expect("fixture written");
    let res = child.wait_with_output().expect("host probe finishes");
    assert!(res.status.success(), "host probe failed");

    let mut r = res.stdout.as_slice();
    let take_u16 = |r: &mut &[u8], k: usize| -> Vec<u16> {
        let (a, b) = r.split_at(k * 2);
        *r = b;
        a.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
    };
    let take_f32 = |r: &mut &[u8], k: usize| -> Vec<f32> {
        let (a, b) = r.split_at(k * 4);
        *r = b;
        a.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };
    let gather = take_u16(&mut r, ids.len() * d);
    let deq = take_f32(&mut r, rows * d);
    let dot = take_f32(&mut r, rows);
    assert!(r.is_empty(), "the host probe wrote more than it was asked to");

    // (2a) f32 dequant, bit for bit against RawTensor::to_f32.
    for (i, (&got, &want)) in deq.iter().zip(&reference).enumerate() {
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "weight {i}: kernel dequant {got} vs reference {want}"
        );
    }
    // (2b) the gather's f16 store, bit for bit against f16::from_f32.
    for (t_i, &id) in ids.iter().enumerate() {
        for c in 0..d {
            let want = f16::from_f32(reference[id as usize * d + c]).to_bits();
            let got = gather[t_i * d + c];
            assert_eq!(got, want, "token {t_i} (row {id}), col {c}");
        }
    }
    // (2c) the matvec's lane-ordered sum against f64, relative to Σ|w·x|.
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

/// Read every carried tensor of a sealed artifact, skipping the matrices.
fn read_raws(path: &std::path::Path) -> Option<Vec<RawTensor>> {
    let f = std::fs::File::open(path).ok()?;
    let mut r = std::io::BufReader::with_capacity(1 << 20, f);
    let head = llvq_artifact::read_header(&mut r).ok()?;
    if !head.is_self_contained() {
        return None;
    }
    for _ in 0..head.matrices {
        llvq_artifact::read_matrix_raw(&mut r).ok()?;
    }
    let mut b = [0u8; 4];
    std::io::Read::read_exact(&mut r, &mut b).ok()?;
    let n = u32::from_le_bytes(b);
    let mut raws = Vec::with_capacity(n as usize);
    for _ in 0..n {
        raws.push(llvq_artifact::read_raw(&mut r, head.version).ok()?);
    }
    Some(raws)
}

/// Sampled real rows: the load path applied to the f16 embedding of the
/// published 4B must reproduce, byte for byte, what `bin/embedq` wrote into
/// the artifact lot B scored. Row-independent by construction (groups never
/// cross rows), so a sample proves what a full pass would.
///
/// Skipped silently when either artifact is not on this machine.
#[test]
#[cfg_attr(debug_assertions, ignore = "reads two ~1 GB artifacts, run in release")]
fn real_rows_match_the_sealed_q8_file() {
    let home = match std::env::var("HOME") {
        Ok(h) => std::path::PathBuf::from(h),
        Err(_) => return,
    };
    let (src, e8) = (home.join("qwen3-4b-llvq.bin"), home.join("q4b-e8.llvq"));
    if !src.exists() || !e8.exists() {
        eprintln!("artefacts absents ({src:?}, {e8:?}) — test sauté");
        return;
    }
    let find = |raws: Vec<RawTensor>| {
        raws.into_iter().find(|t| t.name == "model.embed_tokens.weight")
    };
    let src_emb = find(read_raws(&src).expect("readable source")).expect("embedding in source");
    let e8_emb = find(read_raws(&e8).expect("readable e8")).expect("embedding in e8");
    let RawData::F16(f16_data) = &src_emb.data else {
        panic!("source embedding is not f16")
    };
    let (vocab, d) = (src_emb.dims[0], src_emb.dims[1]);
    assert_eq!(e8_emb.dims, src_emb.dims, "the two artifacts disagree on dims");
    let (e8p, e8s, e8b) = as_quant(&e8_emb);
    let gpr = d.div_ceil(64);

    // 64 rows spread across the vocabulary, plus both ends.
    let mut rows: Vec<usize> = (0..64).map(|i| i * (vocab - 1) / 63).collect();
    rows.push(0);
    rows.push(vocab - 1);
    let sample: Vec<u16> = rows
        .iter()
        .flat_map(|&r| f16_data[r * d..(r + 1) * d].iter().copied())
        .collect();
    let sub = RawTensor {
        name: src_emb.name.clone(),
        dims: vec![rows.len(), d],
        data: RawData::F16(sample),
    };
    let q = embed_q8(sub).expect("load-path quantization");
    let (qp, qs, qb) = as_quant(&q);
    for (i, &r) in rows.iter().enumerate() {
        assert_eq!(
            &qp[i * d..(i + 1) * d],
            &e8p[r * d..(r + 1) * d],
            "row {r}: packed bytes differ from the sealed q8 file"
        );
        assert_eq!(
            &qs[i * gpr..(i + 1) * gpr],
            &e8s[r * gpr..(r + 1) * gpr],
            "row {r}: scales differ"
        );
        assert_eq!(
            &qb[i * gpr..(i + 1) * gpr],
            &e8b[r * gpr..(r + 1) * gpr],
            "row {r}: biases differ"
        );
    }
}
