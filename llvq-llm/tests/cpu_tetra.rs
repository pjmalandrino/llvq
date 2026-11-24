//! The CPU backend against the reference decoder.
//!
//! The Tetra matvec of `fused_cpu` reads the served `tetra48` stream and a
//! rotated, trio-ordered activation. The reference is `decode_matrix`: the
//! dense weights every published number was scored on, multiplied by the
//! natural-basis activation in f64. The two agree to f32 rounding, on every
//! SIMD path this CPU has, one row at a time and in batches, through the
//! candle adapter. The int4 records and the int4 embedding are checked the
//! same way against `to_f32`.

use candle_core::{DType, Device, Tensor};
use llvq_artifact::{decode_matrix, QuantizedMatrix};
use llvq_core::{SplitMix64, DIM};
use llvq_llm::device::{Int4Proj, LatticeProj, QuantEmbedTable};
use llvq_llm::fused_cpu::{CpuEmbed, CpuInt4, CpuLattice};
use llvq_llm::tetra_cpu::{available_isas, Q4Matrix, Tables, TetraMatrix};
use llvq_quant::quantizer::BlockCode;
use llvq_quant::rotation::Rotation;
use llvq_search::tetra::Tetra;
use std::sync::Arc;

/// Worst `|y − y_ref| / Σ|w·x|` over the rows, the repository's scale.
fn worst(w: &[f32], d_in: usize, x: &[f64], got: &[f32]) -> f64 {
    let mut e = 0f64;
    for (r, &g) in got.iter().enumerate() {
        let row = &w[r * d_in..(r + 1) * d_in];
        let want: f64 = row.iter().zip(x).map(|(&a, &b)| a as f64 * b).sum();
        let scale: f64 = row.iter().zip(x).map(|(&a, &b)| (a as f64 * b).abs()).sum::<f64>().max(1e-12);
        e = e.max((want - g as f64).abs() / scale);
    }
    e
}

struct Case {
    q: QuantizedMatrix,
    proj: CpuLattice,
}

/// A Tetra matrix with a rotation, a tail and distinct row scales, as the
/// reference holds it and as the CPU backend holds it.
fn case(d_out: usize, d_in: usize, seed: u64, rotate: bool) -> Case {
    let tetra = Tetra::new();
    let mut rng = SplitMix64::new(seed);
    let (nblocks, tail_w) = (d_in / DIM, d_in % DIM);
    let n = d_out * nblocks;
    // One block in six is the origin: the zero entry of the norm table.
    let labels: Vec<u64> =
        (0..n).map(|_| if rng.next().is_multiple_of(6) { 0 } else { rng.next() & ((1u64 << 47) - 1) }).collect();
    let gains: Vec<u32> = (0..n).map(|_| (rng.next() & 1) as u32).collect();
    let codes = labels.iter().zip(&gains).map(|(&l, &g)| BlockCode { point: tetra.decode(l), gain: g }).collect();
    // The tail at the precision the loader holds it: binary16.
    let tail_bits: Vec<u16> =
        (0..d_out * tail_w).map(|_| half::f16::from_f64(rng.next_gaussian()).to_bits()).collect();
    let row_scales: Vec<f64> = (0..d_out).map(|_| f64::from((0.5 + rng.next_gaussian().abs()) as f32)).collect();
    let centroids = vec![0.625f64, 1.375];
    let seed_rot = rotate.then_some(seed ^ 0xbeef);
    let q = QuantizedMatrix {
        name: format!("model.layers.0.test.{d_out}x{d_in}.weight"),
        d_out,
        d_in,
        codes,
        row_scales: row_scales.clone(),
        centroids: centroids.clone(),
        rotation_seed: seed_rot,
        shell_cap: llvq_artifact::TETRA_SHELL_CAP,
        tail: tail_bits.iter().map(|&b| half::f16::from_bits(b).to_f64()).collect(),
    };
    let stream = llvq_artifact::tetra48::transcode_tetra48(&labels, &gains, d_out, nblocks).unwrap();
    let words = stream.data.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let m = TetraMatrix {
        d_out,
        d_in,
        nblocks,
        tail_w,
        stride: stream.stride_u32,
        words,
        gscale: [centroids[0] as f32, centroids[1] as f32],
        rscale: row_scales.iter().map(|&v| v as f32).collect(),
        tail: tail_bits,
    };
    let rot = seed_rot.map(|s| Arc::new(Rotation::new(d_in, s)));
    let key = seed_rot.map(|s| (d_in, s));
    let proj = CpuLattice::new(q.name.clone(), Arc::new(Tables::new(&tetra)), m, rot, key, DType::F32).unwrap();
    Case { q, proj }
}

fn activations(d_in: usize, rows: usize, seed: u64) -> Vec<f64> {
    let mut rng = SplitMix64::new(seed);
    (0..rows * d_in).map(|_| rng.next_gaussian()).collect()
}

const SHAPES: [(usize, usize, bool); 5] = [
    (8, 2560, true),  // Qwen3-4B hidden: tail 16, rotation k = 5
    (6, 9728, true),  // down_proj input: tail 8, k = 19
    (8, 4096, true),  // power of two, no tail
    (5, 240, true),   // small, no tail, odd block count
    (7, 72, false),   // natural basis
];

#[test]
fn the_tetra_matvec_computes_what_the_reference_decodes() {
    for (i, &(d_out, d_in, rotate)) in SHAPES.iter().enumerate() {
        let c = case(d_out, d_in, 0x5ee0 + i as u64, rotate);
        let w = decode_matrix(&c.q);
        let x = activations(d_in, 1, 0xa11ce + i as u64);
        let xf: Vec<f32> = x.iter().map(|&v| v as f32).collect();
        for isa in available_isas() {
            let proj = case(d_out, d_in, 0x5ee0 + i as u64, rotate).proj.with_isa(isa);
            let y = proj.matvec_vec(&proj.prepare_vec(&xf, 1), 1);
            let e = worst(&w, d_in, &x, &y);
            assert!(e < 2e-6, "{d_out}x{d_in} on {}: {e:.3e} Σ|w·x|", isa.name());
        }
        drop(c.proj);
    }
}

#[test]
fn a_batch_computes_what_one_row_at_a_time_computes() {
    for (i, &(d_out, d_in, rotate)) in SHAPES.iter().enumerate() {
        let c = case(d_out, d_in, 0xba7c + i as u64, rotate);
        let w = decode_matrix(&c.q);
        // 19 rows: one full batch of 16 and a remainder.
        let rows = 19;
        let x = activations(d_in, rows, 0xfeed + i as u64);
        let xf: Vec<f32> = x.iter().map(|&v| v as f32).collect();
        for isa in available_isas() {
            let proj = case(d_out, d_in, 0xba7c + i as u64, rotate).proj.with_isa(isa);
            let ys = proj.matvec_vec(&proj.prepare_vec(&xf, rows), rows);
            for r in 0..rows {
                let one = proj.matvec_vec(&proj.prepare_vec(&xf[r * d_in..(r + 1) * d_in], 1), 1);
                let got = &ys[r * d_out..(r + 1) * d_out];
                let e = worst(&w, d_in, &x[r * d_in..(r + 1) * d_in], got);
                assert!(e < 2e-6, "{d_out}x{d_in} row {r} on {}: {e:.3e}", isa.name());
                for (a, b) in got.iter().zip(&one) {
                    assert!((a - b).abs() <= 1e-5 * b.abs().max(1.0), "{d_out}x{d_in} row {r}: {a} vs {b}");
                }
            }
        }
    }
}

/// A rotation left out is a different answer, not a small error: the check
/// above cannot pass by accident on a path that skips it.
#[test]
fn skipping_the_rotation_is_detected() {
    let c = case(8, 2560, 0x77, true);
    let w = decode_matrix(&c.q);
    let x = activations(2560, 1, 0x5a17);
    let xf: Vec<f32> = x.iter().map(|&v| v as f32).collect();
    let tables = Tables::new(&Tetra::new());
    let mut permuted = vec![0f32; xf.len()];
    llvq_llm::tetra_cpu::permute_blocks(&tables, &xf, &mut permuted);
    let y = c.proj.matvec_vec(&permuted, 1);
    assert!(worst(&w, 2560, &x, &y) > 1e-2, "an unrotated activation must not pass");
}

#[test]
fn the_candle_adapter_binds_the_same_product() {
    let c = case(8, 2560, 0xcafe, true);
    let w = decode_matrix(&c.q);
    let x = activations(2560, 1, 0xd00d);
    let xt = Tensor::from_vec(x.iter().map(|&v| v as f32).collect::<Vec<_>>(), (1, 1, 2560), &Device::Cpu)
        .unwrap()
        .to_dtype(DType::F16)
        .unwrap();
    // The f16 activation the model hands over, widened exactly.
    let xh: Vec<f64> = xt.flatten_all().unwrap().to_dtype(DType::F32).unwrap().to_vec1::<f32>().unwrap()
        .into_iter().map(f64::from).collect();
    let xr = c.proj.prepare(&xt).unwrap();
    let y = c.proj.matvec(&xr, xt.dims()).unwrap();
    assert_eq!(y.dims(), &[1, 1, 8]);
    let y: Vec<f32> = y.flatten_all().unwrap().to_vec1().unwrap();
    assert!(worst(&w, 2560, &xh, &y) < 2e-6);

    let xs = Tensor::from_vec(activations(2560, 4, 1).iter().map(|&v| v as f32).collect::<Vec<_>>(), (4, 2560), &Device::Cpu).unwrap();
    let yr = c.proj.matvec_rows(&c.proj.prepare_rows(&xs, 4).unwrap(), 4).unwrap();
    assert_eq!(yr.dims(), &[4, 8]);
}

fn q4(d_out: usize, d_in: usize, group: usize, seed: u64) -> Q4Matrix {
    let mut rng = SplitMix64::new(seed);
    let groups = d_out * d_in / group;
    Q4Matrix {
        d_out,
        d_in,
        group,
        packed: (0..d_out * d_in / 2).map(|_| rng.next() as u8).collect(),
        scales: (0..groups).map(|_| half::f16::from_f64(0.01 + 0.02 * rng.next_gaussian().abs()).to_bits()).collect(),
        biases: (0..groups).map(|_| half::f16::from_f64(-0.1 * rng.next_gaussian().abs()).to_bits()).collect(),
    }
}

fn q4_dense(m: &Q4Matrix) -> Vec<f32> {
    let mut w = vec![0f32; m.d_out * m.d_in];
    for r in 0..m.d_out {
        m.decode_row(r, &mut w[r * m.d_in..(r + 1) * m.d_in]);
    }
    w
}

#[test]
fn the_int4_matvec_computes_what_the_record_decodes() {
    for (d_out, d_in, seed) in [(16usize, 2560usize, 1u64), (8, 9728, 2), (4, 128, 3)] {
        let m = q4(d_out, d_in, 128, seed);
        // The record's own decoder, independent of `decode_row`.
        let rec = llvq_artifact::Int4Matrix {
            name: "model.layers.0.self_attn.v_proj.weight".into(),
            d_out,
            d_in,
            bits: 4,
            group: 128,
            packed: m.packed.clone(),
            scales: m.scales.clone(),
            biases: m.biases.clone(),
        };
        let w = rec.to_f32();
        assert_eq!(w, q4_dense(&m), "decode_row is the record's decode");
        let x = activations(d_in, 3, seed);
        let xf: Vec<f32> = x.iter().map(|&v| v as f32).collect();
        for isa in available_isas() {
            let p = CpuInt4::new(rec.name.clone(), q4(d_out, d_in, 128, seed), DType::F32).unwrap().with_isa(isa);
            let y = p.matvec_vec(&xf[..d_in], 1);
            assert!(worst(&w, d_in, &x[..d_in], &y) < 2e-6, "{d_out}x{d_in} on {}", isa.name());
            let ys = p.matvec_vec(&xf, 3);
            for r in 0..3 {
                let e = worst(&w, d_in, &x[r * d_in..(r + 1) * d_in], &ys[r * d_out..(r + 1) * d_out]);
                assert!(e < 2e-6, "{d_out}x{d_in} batch row {r} on {}: {e:.3e}", isa.name());
            }
            let t = p.matvec(&Tensor::from_vec(xf[..d_in].to_vec(), (1, d_in), &Device::Cpu).unwrap(), &[1, d_in]).unwrap();
            assert_eq!(t.dims(), &[1, d_out]);
        }
    }
}

#[test]
fn the_int4_embedding_gathers_and_projects_its_rows() {
    let (vocab, d) = (300usize, 256usize);
    let m = Arc::new(q4(vocab, d, 64, 9));
    let w = q4_dense(&m);
    for isa in available_isas() {
        let e = CpuEmbed::new("model.embed_tokens.weight".into(), m.clone(), DType::F32).unwrap().with_isa(isa);
        let ids = Tensor::from_vec(vec![0u32, 7, 299], (1, 3), &Device::Cpu).unwrap();
        let h = e.gather(&ids).unwrap();
        assert_eq!(h.dims(), &[1, 3, d]);
        let h: Vec<f32> = h.flatten_all().unwrap().to_vec1().unwrap();
        for (k, &id) in [0usize, 7, 299].iter().enumerate() {
            assert_eq!(&h[k * d..(k + 1) * d], &w[id * d..(id + 1) * d], "row {id}: the table's own values");
        }
        // 70 rows: one pass of 64 and a remainder.
        let rows = 70;
        let x = activations(d, rows, 4);
        let xt = Tensor::from_vec(x.iter().map(|&v| v as f32).collect::<Vec<_>>(), (1, rows, d), &Device::Cpu).unwrap();
        let logits = e.project(&xt).unwrap();
        assert_eq!(logits.dims(), &[1, rows, vocab]);
        let l: Vec<f32> = logits.flatten_all().unwrap().to_vec1().unwrap();
        for r in 0..rows {
            let err = worst(&w, d, &x[r * d..(r + 1) * d], &l[r * vocab..(r + 1) * vocab]);
            assert!(err < 2e-6, "row {r} on {}: {err:.3e}", isa.name());
        }
        assert!(e.gather(&Tensor::from_vec(vec![300u32], (1, 1), &Device::Cpu).unwrap()).is_err(), "past the vocabulary");
    }
}

// ---------------------------------------------------------------------------
// The microbenchmark: random codes at the served 4B's shapes, no model.
//
//   LLVQ_THREADS=8 cargo test --profile ci -p llvq-llm --test cpu_tetra -- --ignored --nocapture bench_
//
// Prints the median time of one call over rounds of about 0.15 s, the range
// of the rounds, and the bytes the matrix holds over that time. The pool's
// size is fixed per process by LLVQ_THREADS.
// ---------------------------------------------------------------------------

/// A random Tetra matrix as the loader holds it: random 47-bit labels and
/// gain bits through the served transcode, random row scales, a binary16 tail.
fn random_tetra(d_out: usize, d_in: usize, seed: u64) -> TetraMatrix {
    let mut rng = SplitMix64::new(seed);
    let (nblocks, tail_w) = (d_in / DIM, d_in % DIM);
    let labels: Vec<u64> = (0..d_out * nblocks).map(|_| rng.next() & ((1u64 << 47) - 1)).collect();
    let gains: Vec<u32> = (0..d_out * nblocks).map(|_| (rng.next() & 1) as u32).collect();
    let s = llvq_artifact::tetra48::transcode_tetra48(&labels, &gains, d_out, nblocks).unwrap();
    TetraMatrix {
        d_out,
        d_in,
        nblocks,
        tail_w,
        stride: s.stride_u32,
        words: s.data.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        gscale: [0.625, 1.375],
        rscale: (0..d_out).map(|_| 0.5 + rng.next_gaussian().abs() as f32).collect(),
        tail: (0..d_out * tail_w).map(|_| half::f16::from_f64(rng.next_gaussian()).to_bits()).collect(),
    }
}

/// A random int4 matrix, bytes drawn eight at a time.
fn random_q4(d_out: usize, d_in: usize, group: usize, seed: u64) -> Q4Matrix {
    let mut rng = SplitMix64::new(seed);
    let groups = d_out * d_in / group;
    let packed: Vec<u8> = (0..d_out * d_in / 16).flat_map(|_| rng.next().to_le_bytes()).collect();
    Q4Matrix {
        d_out,
        d_in,
        group,
        packed,
        scales: (0..groups).map(|_| half::f16::from_f64(0.01 + 0.02 * rng.next_gaussian().abs()).to_bits()).collect(),
        biases: (0..groups).map(|_| half::f16::from_f64(-0.1 * rng.next_gaussian().abs()).to_bits()).collect(),
    }
}

/// Median and range, in ns, of one call of `f` over rounds of about 0.15 s.
fn time_ns(mut f: impl FnMut()) -> (f64, f64, f64) {
    use std::time::Instant;
    for _ in 0..3 {
        f();
    }
    let t0 = Instant::now();
    f();
    let one = t0.elapsed().as_secs_f64().max(1e-7);
    let per_round = ((0.15 / one) as usize).clamp(1, 100_000);
    let mut v: Vec<f64> = (0..7)
        .map(|_| {
            let t = Instant::now();
            for _ in 0..per_round {
                f();
            }
            t.elapsed().as_secs_f64() * 1e9 / per_round as f64
        })
        .collect();
    v.sort_by(f64::total_cmp);
    (v[v.len() / 2], v[0], v[v.len() - 1])
}

fn report(what: &str, bytes: u64, weights: usize, (med, lo, hi): (f64, f64, f64)) -> f64 {
    println!(
        "bench {what:<34} {:>11.0} ns  [{:.0} ; {:.0}]  {:>6.2} GB/s  {:>6.2} Gweight/s",
        med,
        lo,
        hi,
        bytes as f64 / med,
        weights as f64 / med
    );
    med
}

#[test]
#[ignore = "a microbenchmark: run by hand with --ignored --nocapture"]
fn bench_cpu_matvecs_at_the_4b_shapes() {
    use llvq_llm::tetra_cpu::{self, Isa};
    let isa = Isa::detect();
    let t = Tables::new(&Tetra::new());
    println!("bench isa {} threads {}", isa.name(), tetra_cpu::pool().current_num_threads());
    let mut ms_token = 0f64;
    // (name, d_out, d_in, calls per token): the served 4B, Tetra sites.
    for (name, d_out, d_in, per_token) in [
        ("tetra q_proj 4096x2560", 4096usize, 2560usize, 36usize),
        ("tetra k_proj 1024x2560", 1024, 2560, 36),
        ("tetra gate/up 9728x2560", 9728, 2560, 72),
        ("tetra down 2560x9728", 2560, 9728, 24),
    ] {
        let m = random_tetra(d_out, d_in, d_out as u64 ^ d_in as u64);
        let x: Vec<f32> = activations(d_in, 1, 3).iter().map(|&v| v as f32).collect();
        let mut xp = vec![0f32; d_in];
        tetra_cpu::permute_blocks(&t, &x, &mut xp);
        let mut y = vec![0f32; d_out];
        let r = time_ns(|| tetra_cpu::tetra_matvec(isa, &t, &m, &xp, &mut y));
        ms_token += report(name, m.bytes(), d_out * d_in, r) * per_token as f64 / 1e6;
    }
    // int4 sites: v_proj, o_proj, down_proj of layers 12 to 23, g128; the tied head, g64.
    for (name, d_out, d_in, group, per_token) in [
        ("int4 g128 v_proj 1024x2560", 1024usize, 2560usize, 128usize, 36usize),
        ("int4 g128 o_proj 2560x4096", 2560, 4096, 128, 36),
        ("int4 g128 down 2560x9728", 2560, 9728, 128, 12),
        ("int4 g64 head 151936x2560", 151_936, 2560, 64, 1),
    ] {
        let m = random_q4(d_out, d_in, group, 11 + d_out as u64);
        let x: Vec<f32> = activations(d_in, 1, 5).iter().map(|&v| v as f32).collect();
        let mut y = vec![0f32; d_out];
        let r = time_ns(|| tetra_cpu::q4_matvec(isa, &m, &x, &mut y));
        ms_token += report(name, m.bytes(), d_out * d_in, r) * per_token as f64 / 1e6;
    }
    println!("bench projections and head of one 4B token (computed from the rows above): {ms_token:.2} ms");

    // The candle adapter around the largest site: f32 tensor in, rotation,
    // trio order, matvec, f32 tensor out.
    let m = random_tetra(9728, 2560, 99);
    let bytes = m.bytes();
    let p = CpuLattice::new("bench.gate".into(), Arc::new(Tables::new(&Tetra::new())), m, Some(Arc::new(Rotation::new(2560, 7))), Some((2560, 7)), DType::F32)
        .unwrap();
    let x = Tensor::from_vec(activations(2560, 1, 9).iter().map(|&v| v as f32).collect::<Vec<_>>(), (1, 1, 2560), &Device::Cpu).unwrap();
    let xr = p.prepare(&x).unwrap();
    let xv: Vec<f32> = xr.flatten_all().unwrap().to_vec1().unwrap();
    report("adapter prepare 2560", 0, 0, time_ns(|| drop(p.prepare(&x).unwrap())));
    // The same matrix through the slices, then through the tensors, twice, so
    // that a change of machine state between the two shows as a disagreement.
    for _ in 0..2 {
        report("slices matvec gate 9728x2560", bytes, 9728 * 2560, time_ns(|| drop(p.matvec_vec(&xv, 1))));
        report("adapter matvec gate 9728x2560", bytes, 9728 * 2560, time_ns(|| drop(p.matvec(&xr, x.dims()).unwrap())));
    }
}
