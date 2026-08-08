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
use llvq_llm::fused::{
    carried_embed_tables, embed_q8, q8_device_bytes, take_embed_tables, EmbedMode, EmbedReport,
    EMBED_GROUP, EMBED_NAME, HEAD_NAME,
};
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

/// [`test_tensor`] under an arbitrary name — the untied case needs two tables
/// that differ in **both** name and content.
fn named_tensor(name: &str, rows: usize, row_len: usize, seed: u64) -> RawTensor {
    RawTensor { name: name.into(), ..test_tensor(rows, row_len, seed) }
}

/// Compile the CPU driver of the two q8 kernels once per test that needs it.
fn build_host_probe(tag: &str) -> std::path::PathBuf {
    let out = std::env::temp_dir().join(format!("llvq_host_embq8_{tag}"));
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests");
    let st = Command::new("clang++")
        .args(["-std=c++17", "-O2", "-ffp-contract=off", "-Wall", "-Wextra", "-Werror"])
        .arg(dir.join("host_embq8.cpp"))
        .arg("-o")
        .arg(&out)
        .status()
        .expect("clang++ is on PATH");
    assert!(st.success(), "the q8 embedding kernels do not compile as host C++");
    out
}

/// Run the probe on one int8 g64 table: `(gather f16 bits, f32 dequant, per-row
/// dot against `x`)` — the outputs `emb_q8_gather` and `tv_q8_h` would produce
/// on the device from *that* buffer, and from no other.
fn run_host_probe(
    exe: &std::path::Path,
    t: &RawTensor,
    ids: &[u32],
    x: &[u16],
) -> (Vec<u16>, Vec<f32>, Vec<f32>) {
    let d = *t.dims.last().expect("2-D");
    let rows = t.len() / d;
    let gpr = d.div_ceil(EMBED_GROUP);
    let (packed, scales, biases) = as_quant(t);
    assert_eq!(x.len(), d, "the staged activation must be one row wide");

    let mut fixture = Vec::new();
    for v in [rows as u32, d as u32, gpr as u32, ids.len() as u32] {
        fixture.extend_from_slice(&v.to_le_bytes());
    }
    for c in packed.chunks_exact(4) {
        fixture.extend_from_slice(c);
    }
    for v in scales.iter().chain(biases) {
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
        a.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
    };
    let gather = take_u16(&mut r, ids.len() * d);
    let deq = take_f32(&mut r, rows * d);
    let dot = take_f32(&mut r, rows);
    assert!(r.is_empty(), "the host probe wrote more than it was asked to");
    (gather, deq, dot)
}

/// Deterministic f16 activation, `d` wide.
fn probe_activation(d: usize, seed: u64) -> Vec<u16> {
    let mut s = seed;
    (0..d)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = ((s >> 33) as f64 / (1u64 << 31) as f64) - 0.5;
            f16::from_f64(u * 2.0).to_bits()
        })
        .collect()
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
    let out = build_host_probe("one");

    // ---- fixture: short last group (224 = 3·64 + 32), 96 rows ----
    let (rows, d) = (96usize, 224usize);
    let t = embed_q8(test_tensor(rows, d, 11)).expect("quantize");
    let reference = t.to_f32();
    let ids: Vec<u32> = vec![0, 95, 3, 3, 17, 64, 90, 1];
    let x = probe_activation(d, 0xE1B);
    let (gather, deq, dot) = run_host_probe(&out, &t, &ids, &x);

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

// ---------------------------------------------------------------------------
// Untied ends — Qwen3-8B carries two embedding tables, not one.
//
// `tie_word_embeddings = false` means `model.embed_tokens.weight` and
// `lm_head.weight` are different weights of the same shape. Everything below
// exists to make one failure impossible: a run where the output end silently
// reads the input end's table. That model loads, generates fluent text, and is
// wrong — nothing downstream would flag it.
// ---------------------------------------------------------------------------

/// A carried list in a deliberately awkward order — a norm first, then
/// `lm_head` **before** the embedding. File order must decide nothing.
fn carried_list(rows: usize, d: usize, with_head: bool) -> Vec<RawTensor> {
    carried_list_ordered(rows, d, with_head, false)
}

/// The same list, with the option of the **natural** checkpoint order —
/// embedding first, `lm_head` last.
///
/// This order is not decoration: extraction removes the embedding by
/// `swap_remove`, which moves the *last* element into the freed slot. An
/// implementation that memoized `lm_head`'s index before that removal is
/// correct on an embedding-last list and reads the wrong tensor — or panics —
/// on an embedding-first one. With only the awkward order in the fixtures,
/// the property "the second lookup re-searches by name" is asserted by a
/// comment and exercised by nothing (CLAUDE.md §5).
fn carried_list_ordered(
    rows: usize,
    d: usize,
    with_head: bool,
    embed_first: bool,
) -> Vec<RawTensor> {
    let norm = RawTensor {
        name: "model.norm.weight".into(),
        dims: vec![d],
        data: RawData::F16(vec![f16::ONE.to_bits(); d]),
    };
    let embed = named_tensor(EMBED_NAME, rows, d, 21);
    if !with_head {
        return if embed_first { vec![embed, norm] } else { vec![norm, embed] };
    }
    let head = named_tensor(HEAD_NAME, rows, d, 77);
    if embed_first {
        vec![embed, norm, head]
    } else {
        vec![norm, head, embed]
    }
}

/// Tied ends: one table off the carried list, one buffer, both ends reading
/// it — the shipped 4B behaviour, byte for byte.
#[test]
fn tie_true_takes_one_table_and_wires_both_ends_to_it() {
    let (rows, d) = (32usize, 128usize);
    let mut raw = carried_list(rows, d, false);
    let tables = take_embed_tables(&mut raw, true).expect("tied extraction");

    assert!(tables.head.is_none(), "a tied model must not build a second table");
    assert_eq!(tables.buffers().len(), 1, "one payload on the device");
    assert_eq!(tables.wiring(), (0, 0), "both ends read buffer 0");

    // Bit-identical to what the shipped single-tensor path produced.
    let shipped = embed_q8(named_tensor(EMBED_NAME, rows, d, 21)).expect("shipped path");
    let (sp, ss, sb) = as_quant(&shipped);
    let (tp, ts, tb) = as_quant(tables.buffers()[0]);
    assert_eq!((tp, ts, tb), (sp, ss, sb), "the tied path changed the embedding bytes");

    // The embedding left the carried list; nothing else did.
    assert_eq!(raw.len(), 1, "only the embedding is taken");
    assert_eq!(raw[0].name, "model.norm.weight");
}

/// Untied ends: two tables, two **distinct** buffers, each quantized from its
/// own tensor — and the two kernels, executed on the CPU through the exact
/// text NVRTC compiles, reading the buffer the wiring points them at.
///
/// The fixture is built so a shared buffer cannot pass. The two tables hold
/// different values (asserted, so the test cannot go vacuous), and the gather
/// is required to reproduce the *embedding* while the matvec reproduces the
/// *lm_head*. Point the head at buffer 0, or quantize both from one tensor,
/// and the second half fails by a margin `separation` measures explicitly.
#[test]
fn tie_false_gives_two_buffers_each_read_by_its_own_end() {
    let (rows, d) = (32usize, 128usize);
    // Both file orders, because `swap_remove` only reorders when the
    // embedding is not last: the natural checkpoint order is the one that
    // catches a memoized `lm_head` index.
    for embed_first in [false, true] {
        let mut raw = carried_list_ordered(rows, d, true, embed_first);
        let tables = take_embed_tables(&mut raw, false)
            .unwrap_or_else(|e| panic!("untied extraction (embed_first={embed_first}): {e}"));
        let (ie, ih) = tables.wiring();
        assert_eq!((ie, ih), (0, 1), "untied ends must read two buffers");
        let bufs = tables.buffers();
        assert_eq!(bufs[ie].name, EMBED_NAME, "embed_first={embed_first}");
        assert_eq!(bufs[ih].name, HEAD_NAME, "embed_first={embed_first}");
        // The payloads themselves, not just the names: a memoized index can
        // hand back a correctly-named entry holding the other table's bytes.
        assert_eq!(
            bufs[ie].dims, vec![rows, d],
            "embed_first={embed_first}: embedding payload"
        );
        let (pe, ph) = (as_quant(bufs[ie]).0, as_quant(bufs[ih]).0);
        assert_ne!(
            pe, ph,
            "embed_first={embed_first}: the two buffers carry the same bytes"
        );
        assert_eq!(raw.len(), 1, "embed_first={embed_first}: both tables leave");
        assert_eq!(raw[0].name, "model.norm.weight");
    }

    let mut raw = carried_list(rows, d, true);
    let tables = take_embed_tables(&mut raw, false).expect("untied extraction");

    let (ie, ih) = tables.wiring();
    assert_eq!((ie, ih), (0, 1), "untied ends must read two different buffers");
    let bufs = tables.buffers();
    assert_eq!(bufs.len(), 2, "two payloads on the device");
    assert_eq!(bufs[ie].name, EMBED_NAME, "buffer {ie} is not the embedding");
    assert_eq!(bufs[ih].name, HEAD_NAME, "buffer {ih} is not the lm_head");
    assert_eq!(raw.len(), 1, "both tables leave the carried list");
    assert_eq!(raw[0].name, "model.norm.weight");

    // Each buffer is its own tensor's quantization, not the other's.
    let emb_q = embed_q8(named_tensor(EMBED_NAME, rows, d, 21)).expect("q embed");
    let head_q = embed_q8(named_tensor(HEAD_NAME, rows, d, 77)).expect("q head");
    assert_eq!(as_quant(bufs[ie]).0, as_quant(&emb_q).0, "buffer 0 is not the embedding");
    assert_eq!(as_quant(bufs[ih]).0, as_quant(&head_q).0, "buffer 1 is not the lm_head");
    assert_ne!(
        as_quant(&emb_q).0,
        as_quant(&head_q).0,
        "the fixture must make the two tables distinguishable, or this test is dead"
    );

    // The two kernels, on the two buffers the wiring names.
    let probe = build_host_probe("untied");
    let ids: Vec<u32> = vec![0, 31, 7, 7, 19, 2];
    let x = probe_activation(d, 0x51D);
    let (gather, _, dot_from_embed) = run_host_probe(&probe, bufs[ie], &ids, &x);
    let (gather_from_head, _, dot) = run_host_probe(&probe, bufs[ih], &ids, &x);

    // The input end reproduces the embedding table, bit for bit.
    let emb_ref = emb_q.to_f32();
    for (t_i, &id) in ids.iter().enumerate() {
        for c in 0..d {
            assert_eq!(
                gather[t_i * d + c],
                f16::from_f32(emb_ref[id as usize * d + c]).to_bits(),
                "token {t_i} (row {id}), col {c}: the gather did not read the embedding"
            );
        }
    }
    assert_ne!(
        gather_from_head, gather,
        "the two buffers are indistinguishable through the gather — dead fixture"
    );

    // The output end reproduces the lm_head table, and the embedding buffer
    // would visibly have given something else.
    let head_ref = head_q.to_f32();
    let xf: Vec<f64> = x.iter().map(|&b| f16::from_bits(b).to_f64()).collect();
    let want = |w: &[f32], row: usize| -> (f64, f64) {
        let (mut s, mut a) = (0.0f64, 0.0f64);
        for c in 0..d {
            let term = w[row * d + c] as f64 * xf[c];
            s += term;
            a += term.abs();
        }
        (s, a)
    };
    let (mut worst, mut separation) = (0.0f64, f64::INFINITY);
    for row in 0..rows {
        let (h, hs) = want(&head_ref, row);
        let (e, es) = want(&emb_ref, row);
        worst = worst.max((dot[row] as f64 - h).abs() / hs.max(1e-12));
        worst = worst.max((dot_from_embed[row] as f64 - e).abs() / es.max(1e-12));
        separation = separation.min((h - e).abs() / hs.max(1e-12));
    }
    assert!(worst < 1e-6, "a matvec read the wrong table (worst {worst:.2e}·Σ|w·x|)");
    assert!(
        separation > 1e-2,
        "the two tables give the same logits on some row (min separation \
         {separation:.2e}) — a shared buffer could pass this test"
    );
}

/// `tie_word_embeddings = false` on a file carrying no `lm_head.weight` is an
/// error that names it. Falling back on the embedding would produce a model
/// that loads, runs, and is wrong.
#[test]
fn untied_without_an_lm_head_fails_and_says_so() {
    let mut raw = carried_list(32, 128, false);
    let err = match take_embed_tables(&mut raw, false) {
        Err(e) => e,
        Ok(_) => panic!("an untied model with no lm_head.weight must not load"),
    };
    assert!(err.contains(HEAD_NAME), "the message must name the missing tensor: {err}");
    assert!(err.contains("tie_word_embeddings"), "the message must say why: {err}");
}

/// Pre-cooked q8 tensors — a `bin/embedq` output — pass through for **both**
/// tables, and the int8-g64-only refusal applies to both as well.
#[test]
fn pre_quantized_tables_pass_through_on_both_ends() {
    let (rows, d) = (32usize, 128usize);
    let e = quantize_affine(&named_tensor(EMBED_NAME, rows, d, 21), 8, 64).expect("q embed");
    let h = quantize_affine(&named_tensor(HEAD_NAME, rows, d, 77), 8, 64).expect("q head");
    let (ep, hp) = (as_quant(&e).0.clone(), as_quant(&h).0.clone());
    let mut raw = vec![h, e];
    let tables = take_embed_tables(&mut raw, false).expect("untied passthrough");
    assert_eq!(as_quant(&tables.embed).0, &ep, "embedding bytes changed in passthrough");
    assert_eq!(
        as_quant(tables.head.as_ref().expect("head")).0,
        &hp,
        "lm_head bytes changed in passthrough"
    );

    let mut bad = vec![
        quantize_affine(&named_tensor(HEAD_NAME, rows, d, 77), 4, 64).expect("int4"),
        quantize_affine(&named_tensor(EMBED_NAME, rows, d, 21), 8, 64).expect("q embed"),
    ];
    assert!(
        take_embed_tables(&mut bad, false).is_err(),
        "an int4 lm_head must be refused, not requantized"
    );
}

/// The footprint line counts every table the device holds. This is the defect
/// it was written for: with the ends untied, the old line announced one
/// table's bytes — a whole table short — next to a total that was right.
#[test]
fn the_accounting_carries_both_tables() {
    let (rows, d) = (32usize, 128usize);

    // f16, untied: two carried tables, embedding first whatever the file order.
    let untied = carried_list(rows, d, true);
    let carried = carried_embed_tables(&untied);
    assert_eq!(
        carried.iter().map(|t| t.name.as_str()).collect::<Vec<_>>(),
        [EMBED_NAME, HEAD_NAME],
        "the report's order must not follow the file's"
    );
    let f16r = EmbedReport::new(EmbedMode::F16, &carried);
    assert_eq!(f16r.tables.len(), 2, "two tables carried, two counted");
    assert_eq!(f16r.total(), (rows * d * 2 * 2) as u64, "both tables, f16");
    assert_eq!(f16r.meta(), 0, "f16 carries no scales");
    assert!(f16r.line().contains("2 tables"), "{}", f16r.line());
    assert!(f16r.line().contains(HEAD_NAME), "{}", f16r.line());

    // f16, tied: one table, and the line says the head sits on it.
    let tied = carried_list(rows, d, false);
    let tied_r = EmbedReport::new(EmbedMode::F16, &carried_embed_tables(&tied));
    assert_eq!(tied_r.total(), (rows * d * 2) as u64, "one table, f16");
    assert!(tied_r.line().contains("1 table ("), "{}", tied_r.line());
    assert!(tied_r.line().contains("lm_head lié"), "{}", tied_r.line());

    // q8, untied: both tables counted, payload and metadata separately.
    let mut raw = carried_list(rows, d, true);
    let tables = take_embed_tables(&mut raw, false).expect("untied");
    let q8r = EmbedReport::new(EmbedMode::Q8, &tables.buffers());
    let (p1, s1) = q8_device_bytes(&[rows, d]);
    assert_eq!(q8r.tables.len(), 2, "two buffers uploaded, two counted");
    assert_eq!(q8r.packed(), p1 * 2, "int8 payload of both tables");
    assert_eq!(q8r.meta(), s1 * 2, "scales and biases of both tables");
    assert_eq!(q8r.total(), (p1 + s1) * 2);
    assert!(q8r.line().contains("2 tables"), "{}", q8r.line());
    assert!(q8r.total() < f16r.total(), "q8 must be the smaller of the two");

    // q8, tied: one buffer, and the report says so.
    let mut raw = carried_list(rows, d, false);
    let tied_t = take_embed_tables(&mut raw, true).expect("tied");
    let tied_q8 = EmbedReport::new(EmbedMode::Q8, &tied_t.buffers());
    assert_eq!(tied_q8.total(), p1 + s1, "one table, q8");
    assert!(tied_q8.line().contains("1 table ("), "{}", tied_q8.line());
}

/// A shape-only stand-in: [`EmbedReport`] reads a table's name and dims and
/// nothing else, which is precisely what makes an 8B footprint checkable here
/// without 2.5 GB of fixture.
fn shape_only(name: &str, dims: &[usize]) -> RawTensor {
    RawTensor { name: name.into(), dims: dims.to_vec(), data: RawData::F16(Vec::new()) }
}

/// The **printed** line carries the total of every table, not the first one's.
///
/// Asserted at 8B scale on purpose: at test-fixture sizes both readings round
/// to `0.0 Mo` and the assertion would be vacuous. This is the one that fails
/// on the defect as it actually shipped — a line announcing one table beside a
/// total, below, that counted two.
#[test]
fn the_printed_line_announces_every_table() {
    let dims = [151_936usize, 4096];
    let two = [shape_only(EMBED_NAME, &dims), shape_only(HEAD_NAME, &dims)];
    let refs: Vec<&RawTensor> = two.iter().collect();

    let f16_line = EmbedReport::new(EmbedMode::F16, &refs).line();
    assert!(f16_line.contains("2489.3 Mo"), "f16 line under-reports: {f16_line}");
    assert!(!f16_line.contains("1244.7 Mo"), "f16 line announces one table: {f16_line}");

    let q8_line = EmbedReport::new(EmbedMode::Q8, &refs).line();
    assert!(q8_line.contains("1322.5 Mo"), "q8 line under-reports: {q8_line}");
    assert!(!q8_line.contains("661.2 Mo"), "q8 line announces one table: {q8_line}");
    assert!(q8_line.contains("int8 1244.7"), "q8 payload split wrong: {q8_line}");
    assert!(q8_line.contains("échelles/biais 77.8"), "q8 metadata split wrong: {q8_line}");

    // Tied: one table, and the figure is that one table's.
    let one = [shape_only(EMBED_NAME, &dims)];
    let tied = EmbedReport::new(EmbedMode::Q8, &one.iter().collect::<Vec<_>>()).line();
    assert!(tied.contains("661.2 Mo"), "tied q8 line: {tied}");
    assert!(tied.contains("1 table ("), "tied q8 line: {tied}");
}

/// The 8B in numbers, before any card: two untied tables of 151936×4096.
#[test]
fn the_8b_untied_footprint_is_two_tables() {
    let dims = [151_936usize, 4096];
    assert_eq!(dims[0] * dims[1], 622_329_856, "one 8B table");
    let (packed, sb) = q8_device_bytes(&dims);
    assert_eq!(packed, 622_329_856);
    assert_eq!(sb, 38_895_616);
    assert_eq!((packed + sb) * 2, 1_322_450_944, "q8, both tables");
    assert_eq!((dims[0] * dims[1] * 2 * 2) as u64, 2_489_319_424, "f16, both tables");
    assert_eq!(
        2_489_319_424u64 - 1_322_450_944,
        1_166_868_480,
        "what q8 saves on the 8B — one f16 table's worth, near enough"
    );
}
