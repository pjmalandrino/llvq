//! Inkling 2 Mini on the CPU, on synthetic data only.
//!
//! The experts of `inkling::experts_cpu` read Tetra codes and rotate their
//! input as the served kernels do. The reference is `layers::experts_forward`
//! over the dense matrices `decode_matrix` reconstructs from the same codes:
//! the two agree to f32 rounding, one row and a prompt of rows with mixed
//! routings, at a small shape and at the model's. One row equals the same row
//! inside a prompt to rounding, the nibble store equals the packed words bit
//! for bit, and a bad routing or a bad input is refused by name. The int4
//! head is held against the plain product of its decoded table. Last, a tiny
//! whole model is written as the served file set and loaded by
//! `experts_cpu::load_served`, the runtime's CPU load: its logits are those of
//! the same model on the dense reconstructions, and a prompt read in pieces
//! gives what one call gives, both to rounding.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use candle_core::{DType, Device, Result, Tensor};
use llvq_artifact::{decode_matrix, CodeKind, QuantData, QuantizedMatrix, RawData, RawMatrix, RawTensor, Record};
use llvq_core::{SplitMix64, DIM};
use llvq_llm::fused_cpu::CpuLattice;
use llvq_llm::inkling::config::InklingConfig;
use llvq_llm::inkling::expert_codes::{record_name, rotation_seed};
use llvq_llm::inkling::experts_cpu::{CpuExpert, CpuExperts};
use llvq_llm::inkling::head::{OutputProjection, Q4CpuHead};
use llvq_llm::inkling::layers::{experts_forward, ExpertSource, MoeSource, Routing};
use llvq_llm::inkling::{AttnKind, Inkling, LayerExperts};
use llvq_llm::lm::CausalLm;
use llvq_llm::tetra_cpu::{available_isas, Tables, TetraMatrix};
use llvq_quant::quantizer::BlockCode;
use llvq_quant::rotation::Rotation;
use llvq_search::tetra::Tetra;

// ---------------------------------------------------------------------------
// Synthetic Tetra records.
// ---------------------------------------------------------------------------

/// A Tetra record with distinct row scales, a binary16-exact tail when the
/// width has one, and the rotation `seed`.
fn record(name: String, d_out: usize, d_in: usize, gen: u64, seed: u64) -> RawMatrix {
    let mut rng = SplitMix64::new(gen);
    let n = d_out * (d_in / DIM);
    // One block in six is the origin: the zero entry of the norm table.
    let indices = (0..n).map(|_| if rng.next().is_multiple_of(6) { 0 } else { rng.next() & ((1u64 << 47) - 1) }).collect();
    let gains = (0..n).map(|_| (rng.next() & 1) as u32).collect();
    let tail = (0..d_out * (d_in % DIM)).map(|_| half::f16::from_f64(0.3 * rng.next_gaussian()).to_f64()).collect();
    let row_scales = (0..d_out).map(|_| f64::from((0.02 + 0.03 * rng.next_gaussian().abs()) as f32)).collect();
    RawMatrix {
        name,
        d_out,
        d_in,
        kind: CodeKind::Tetra,
        indices,
        gains,
        row_scales,
        centroids: vec![0.625, 1.375],
        rotation_seed: Some(seed),
        shell_cap: llvq_artifact::TETRA_SHELL_CAP,
        tail,
    }
}

/// `r` again: the record type is not `Clone`.
fn copy(r: &RawMatrix) -> RawMatrix {
    RawMatrix {
        name: r.name.clone(),
        d_out: r.d_out,
        d_in: r.d_in,
        kind: r.kind,
        indices: r.indices.clone(),
        gains: r.gains.clone(),
        row_scales: r.row_scales.clone(),
        centroids: r.centroids.clone(),
        rotation_seed: r.rotation_seed,
        shell_cap: r.shell_cap,
        tail: r.tail.clone(),
    }
}

/// The record as `decode_matrix` reads it: the reference weights, in the natural basis.
fn dense_of(r: &RawMatrix) -> Vec<f32> {
    let tetra = Tetra::new();
    let codes = r.indices.iter().zip(&r.gains).map(|(&l, &g)| BlockCode { point: tetra.decode(l), gain: g }).collect();
    decode_matrix(&QuantizedMatrix {
        name: r.name.clone(),
        d_out: r.d_out,
        d_in: r.d_in,
        codes,
        row_scales: r.row_scales.clone(),
        centroids: r.centroids.clone(),
        rotation_seed: r.rotation_seed,
        shell_cap: r.shell_cap,
        tail: r.tail.clone(),
    })
}

/// The record as the CPU backend holds it, turned by `rot`.
fn lattice(r: &RawMatrix, rot: &Arc<Rotation>, tables: &Arc<Tables>, nibbles: bool) -> CpuLattice {
    let (nblocks, tail_w) = (r.d_in / DIM, r.d_in % DIM);
    let stream = llvq_artifact::tetra48::transcode_tetra48(&r.indices, &r.gains, r.d_out, nblocks).unwrap();
    let m = TetraMatrix {
        d_out: r.d_out,
        d_in: r.d_in,
        nblocks,
        tail_w,
        stride: stream.stride_u32,
        words: stream.data.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        gscale: [r.centroids[0] as f32, r.centroids[1] as f32],
        rscale: r.row_scales.iter().map(|&v| v as f32).collect(),
        tail: r.tail.iter().map(|&v| half::f16::from_f64(v).to_bits()).collect(),
    };
    let key = r.rotation_seed.map(|s| (r.d_in, s));
    let p = CpuLattice::new(r.name.clone(), tables.clone(), m, Some(rot.clone()), key, DType::F32).unwrap();
    if nibbles {
        p.expanded()
    } else {
        p
    }
}

/// The three records of every expert of one layer: gate, up, down.
struct Set {
    layer: usize,
    h: usize,
    i: usize,
    recs: Vec<[RawMatrix; 3]>,
}

fn set(layer: usize, n: usize, h: usize, i: usize, gen: u64) -> Set {
    let recs = (0..n)
        .map(|e| {
            let g = gen ^ ((e as u64) << 8);
            let (s0, s1) = (rotation_seed(layer, e, 0), rotation_seed(layer, e, 1));
            [
                record(record_name(layer, e, "gate"), i, h, g ^ 1, s0),
                record(record_name(layer, e, "up"), i, h, g ^ 2, s0),
                record(record_name(layer, e, "down"), h, i, g ^ 3, s1),
            ]
        })
        .collect();
    Set { layer, h, i, recs }
}

fn cpu_experts(s: &Set, nibbles: bool) -> CpuExperts {
    let tables = Arc::new(Tables::default());
    let experts = s
        .recs
        .iter()
        .map(|[g, u, d]| {
            let rot_in = Arc::new(Rotation::new(s.h, g.rotation_seed.unwrap()));
            let rot_down = Arc::new(Rotation::new(s.i, d.rotation_seed.unwrap()));
            CpuExpert {
                gate: lattice(g, &rot_in, &tables, nibbles),
                up: lattice(u, &rot_in, &tables, nibbles),
                down: lattice(d, &rot_down, &tables, nibbles),
            }
        })
        .collect();
    CpuExperts::new(s.layer, experts, s.h, s.i).unwrap()
}

/// Dense experts of one layer: `(gate_up [2I, H], down [H, I])`.
struct DenseExperts(Vec<(Tensor, Tensor)>);

impl ExpertSource for DenseExperts {
    fn count(&self) -> usize {
        self.0.len()
    }
    fn expert(&self, e: usize, _dev: &Device) -> Result<(Tensor, Tensor)> {
        Ok(self.0[e].clone())
    }
}

fn dense_experts(s: &Set) -> DenseExperts {
    DenseExperts(
        s.recs
            .iter()
            .map(|[g, u, d]| {
                let mut gu = dense_of(g);
                gu.extend(dense_of(u));
                (
                    Tensor::from_vec(gu, (2 * s.i, s.h), &Device::Cpu).unwrap(),
                    Tensor::from_vec(dense_of(d), (s.h, s.i), &Device::Cpu).unwrap(),
                )
            })
            .collect(),
    )
}

// ---------------------------------------------------------------------------
// Configs, routings, comparisons.
// ---------------------------------------------------------------------------

/// The tiny config of `tests/inkling_wave1.rs` with its widths and experts replaced.
fn config_json(h: usize, i: usize, n: usize, k: usize) -> serde_json::Value {
    serde_json::json!({
        "hidden_size": h, "num_hidden_layers": 12, "vocab_size": 64, "unpadded_vocab_size": 60,
        "num_attention_heads": 6, "num_key_value_heads": 2, "head_dim": 8,
        "swa_head_dim": 8, "swa_num_attention_heads": 6, "swa_num_key_value_heads": 2,
        "sliding_window_size": 4, "d_rel": 4, "rel_extent": 6,
        "log_scaling_n_floor": 16, "log_scaling_alpha": 0.1, "rms_norm_eps": 1e-6,
        "qk_l2_clip_threshold": 22.6, "rel_l2_clip_threshold": 8.0, "sconv_kernel_size": 4,
        "final_logit_softcapping": 30.0, "n_layers_per_attn_res_block": 6,
        "n_routed_experts": n, "num_experts_per_tok": k, "intermediate_size": i,
        "mlp_gate_softcap": 4.0, "mlp_up_softcap": 25.0, "route_scale": 2.5,
        "loop_config": {"core_start": 6, "core_end": 12, "num_passes": 3, "gate_rank": 4}
    })
}

fn temp_dir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("inkling_cpu_{tag}_{}_{:?}", std::process::id(), std::thread::current().id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn config(h: usize, i: usize, n: usize, k: usize) -> InklingConfig {
    let d = temp_dir(&format!("cfg{h}x{i}"));
    let p = d.join("config.json");
    std::fs::write(&p, config_json(h, i, n, k).to_string()).unwrap();
    let cfg = InklingConfig::from_file(&p).unwrap();
    let _ = std::fs::remove_dir_all(&d);
    cfg
}

/// `t` tokens, each routed to `k` distinct experts of `n` in a random order,
/// with positive weights.
fn routing(seed: u64, t: usize, n: usize, k: usize) -> Routing {
    let mut rng = SplitMix64::new(seed);
    let mut r = Routing { indices: vec![], weights: vec![], margin: vec![], scores: vec![] };
    for _ in 0..t {
        let mut pool: Vec<u32> = (0..n as u32).collect();
        let mut idx = vec![];
        for _ in 0..k {
            idx.push(pool.remove((rng.next() % pool.len() as u64) as usize));
        }
        r.weights.push((0..k).map(|_| (0.1 + rng.next_gaussian().abs()) as f32).collect());
        r.indices.push(idx);
        r.margin.push(0.0);
        r.scores.push(vec![]);
    }
    r
}

fn input(seed: u64, t: usize, h: usize) -> Tensor {
    let mut rng = SplitMix64::new(seed);
    let v: Vec<f32> = (0..t * h).map(|_| rng.next_gaussian() as f32).collect();
    Tensor::from_vec(v, (t, h), &Device::Cpu).unwrap()
}

fn rows_of(t: &Tensor) -> Vec<Vec<f32>> {
    t.to_dtype(DType::F32).unwrap().to_vec2::<f32>().unwrap()
}

/// The worst `max |a - b| / max |b|` over the rows.
fn worst_gap(a: &[Vec<f32>], b: &[Vec<f32>]) -> f64 {
    assert_eq!(a.len(), b.len(), "rows");
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            assert_eq!(x.len(), y.len(), "row length");
            let scale = y.iter().fold(0f64, |m, v| m.max(v.abs() as f64)).max(1e-30);
            x.iter().zip(y).fold(0f64, |m, (p, q)| m.max((p - q).abs() as f64)) / scale
        })
        .fold(0f64, f64::max)
}

/// Tolerance of the CPU experts against the dense reference, relative to a
/// row's largest value: f32 rounding through two matvecs and the activation.
const EXPERT_TOL: f64 = 1e-5;

/// The CPU experts against the dense reference on one row and on `t` rows.
fn against_reference(s: &Set, n: usize, k: usize, t: usize, seed: u64) {
    let cfg = config(s.h, s.i, n, k);
    let cpu = cpu_experts(s, false);
    let dense = dense_experts(s);
    for rows in [1, t] {
        let x = input(seed ^ rows as u64, rows, s.h);
        let r = routing(seed ^ 0x5157 ^ rows as u64, rows, n, k);
        let got = cpu.moe(&x, &r, &cfg).unwrap();
        assert_eq!(got.dims(), [rows, s.h]);
        let want = experts_forward(&x, &r, &dense, &cfg).unwrap();
        let gap = worst_gap(&rows_of(&got), &rows_of(&want));
        println!("{}x{} experts, {rows} rows, top-{k} of {n}: largest relative gap {gap:.3e}", s.h, s.i);
        assert!(gap < EXPERT_TOL, "{}x{}, {rows} rows: gap {gap:.3e} to the dense reference", s.h, s.i);
    }
}

// ---------------------------------------------------------------------------
// The experts.
// ---------------------------------------------------------------------------

#[test]
fn cpu_experts_compute_what_the_dense_reference_computes_at_a_small_shape() {
    // Tail-free widths: two and three blocks.
    against_reference(&set(0, 16, 48, 72, 0xa1), 16, 4, 21, 0x1001);
}

#[test]
fn cpu_experts_compute_what_the_dense_reference_computes_with_a_tail() {
    // The tiny model's widths: 32 is a block and a tail of 8.
    against_reference(&set(3, 8, 48, 32, 0xa2), 8, 2, 9, 0x1002);
}

#[test]
#[cfg_attr(debug_assertions, ignore = "decodes three experts at the model's shape: run in release")]
fn cpu_experts_compute_what_the_dense_reference_computes_at_the_model_shape() {
    // H = 2,304 and I = 1,152 of the real config, rotations of odd factor 9.
    against_reference(&set(6, 3, 2304, 1152, 0xa3), 3, 2, 3, 0x1003);
}

#[test]
fn one_row_equals_the_same_row_inside_a_prompt() {
    let (n, k, t) = (16, 4, 19);
    let s = set(1, n, 48, 72, 0xb1);
    let cfg = config(s.h, s.i, n, k);
    let cpu = cpu_experts(&s, false);
    let x = input(0xb2, t, s.h);
    let r = routing(0xb3, t, n, k);
    let all = rows_of(&cpu.moe(&x, &r, &cfg).unwrap());
    for row in 0..t {
        let one = Routing {
            indices: vec![r.indices[row].clone()],
            weights: vec![r.weights[row].clone()],
            margin: vec![0.0],
            scores: vec![vec![]],
        };
        let y = rows_of(&cpu.moe(&x.narrow(0, row, 1).unwrap(), &one, &cfg).unwrap());
        let gap = worst_gap(&y, &all[row..row + 1]);
        assert!(gap < 1e-5, "row {row}: alone and inside the prompt differ by {gap:.3e}");
    }
}

#[test]
fn packed_and_nibbles_give_the_same_bits() {
    let (n, k, t) = (16, 4, 21);
    let s = set(2, n, 48, 72, 0xc1);
    let cfg = config(s.h, s.i, n, k);
    let (packed, nib) = (cpu_experts(&s, false), cpu_experts(&s, true));
    assert!(nib.stats.matrix_bytes > packed.stats.matrix_bytes, "nibbles hold more bytes than words");
    for rows in [1, t] {
        let x = input(0xc2 ^ rows as u64, rows, s.h);
        let r = routing(0xc3 ^ rows as u64, rows, n, k);
        let a = packed.moe(&x, &r, &cfg).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = nib.moe(&x, &r, &cfg).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let diff = a.iter().zip(&b).position(|(p, q)| p.to_bits() != q.to_bits());
        assert_eq!(diff, None, "{rows} rows: packed and nibbles differ");
    }
}

#[test]
fn a_bad_routing_or_input_is_refused_by_name() {
    let (n, k) = (8, 2);
    let s = set(4, n, 48, 72, 0xd1);
    let cfg = config(s.h, s.i, n, k);
    let cpu = cpu_experts(&s, false);
    let x = input(0xd2, 2, s.h);
    let mut r = routing(0xd3, 2, n, k);
    let err = |x: &Tensor, r: &Routing| cpu.moe(x, r, &cfg).expect_err("refused").to_string();

    r.indices[1][0] = n as u32;
    let e = err(&x, &r);
    assert!(e.contains("layer 4") && e.contains(&format!("expert {n} of {n}")), "{e}");

    let mut r = routing(0xd3, 2, n, k);
    r.indices[0][1] = r.indices[0][0];
    let e = err(&x, &r);
    assert!(e.contains("twice"), "{e}");

    let r = routing(0xd3, 2, n, k);
    let e = err(&input(0xd4, 2, s.h + 24), &r);
    assert!(e.contains("rows of 72"), "{e}");
    let e = err(&x.flatten_all().unwrap(), &r);
    assert!(e.contains("dims"), "{e}");
    let e = err(&x.to_dtype(DType::U32).unwrap(), &r);
    assert!(e.contains("expected a float"), "{e}");
    let e = err(&input(0xd5, 3, s.h), &r);
    assert!(e.contains("2 tokens") && e.contains("3 rows"), "{e}");

    // gate and up turned by two rotations, and a projection of the wrong shape
    let tables = Arc::new(Tables::default());
    let [g, u, d] = &s.recs[0];
    let other = Arc::new(Rotation::new(s.h, 7));
    let rot_in = Arc::new(Rotation::new(s.h, g.rotation_seed.unwrap()));
    let rot_down = Arc::new(Rotation::new(s.i, d.rotation_seed.unwrap()));
    let mut bad_u = copy(u);
    bad_u.rotation_seed = Some(7);
    let x = CpuExpert {
        gate: lattice(g, &rot_in, &tables, false),
        up: lattice(&bad_u, &other, &tables, false),
        down: lattice(d, &rot_down, &tables, false),
    };
    let e = CpuExperts::new(0, vec![x], s.h, s.i).err().expect("two rotations");
    assert!(e.contains("two rotations"), "{e}");
    let x = CpuExpert {
        gate: lattice(g, &rot_in, &tables, false),
        up: lattice(u, &rot_in, &tables, false),
        down: lattice(d, &rot_down, &tables, false),
    };
    let e = CpuExperts::new(0, vec![x], s.h, s.i + 24).err().expect("wrong width");
    assert!(e.contains("gate is 72x48, expected 96x48"), "{e}");
}

// ---------------------------------------------------------------------------
// The int4 head.
// ---------------------------------------------------------------------------

/// A synthetic int4 g64 table of `rows` rows of `d`.
fn q4_table(name: &str, seed: u64, rows: usize, d: usize) -> RawTensor {
    let mut rng = SplitMix64::new(seed);
    let gpr = d.div_ceil(64);
    let packed = (0..rows * d / 2).map(|_| (rng.next() & 0xff) as u8).collect();
    let scales =
        (0..rows * gpr).map(|_| half::f16::from_f32((0.01 + 0.04 * rng.next_gaussian().abs()) as f32).to_bits()).collect();
    let biases = (0..rows * gpr).map(|_| half::f16::from_f32((0.2 * rng.next_gaussian()) as f32).to_bits()).collect();
    RawTensor { name: name.into(), dims: vec![rows, d], data: RawData::Quant(QuantData { bits: 4, group: 64, packed, scales, biases }) }
}

fn argmax(v: &[f32]) -> usize {
    v.iter().enumerate().fold((0, f32::NEG_INFINITY), |(bi, bv), (i, &x)| if x > bv { (i, x) } else { (bi, bv) }).0
}

#[test]
fn the_cpu_int4_head_computes_the_plain_product_of_its_table() {
    let (rows, d, vocab, t_rows) = (1032usize, 2304usize, 1002usize, 3usize);
    let t = q4_table("model.unembed.weight", 0xF1, rows, d);
    let mut rng = SplitMix64::new(0xF2);
    let h: Vec<f32> = (0..t_rows * d).map(|_| rng.next_gaussian() as f32).collect();
    let w = t.to_f32();
    let wt = Tensor::from_slice(&w[..vocab * d], (vocab, d), &Device::Cpu).unwrap();
    for rows_in in [1, t_rows] {
        let x = Tensor::from_slice(&h[..rows_in * d], (rows_in, d), &Device::Cpu).unwrap();
        let plain = rows_of(&x.matmul(&wt.t().unwrap()).unwrap());
        for isa in available_isas() {
            let head = Q4CpuHead::from_raw(&t, vocab, d).unwrap().with_isa(isa);
            assert_eq!(head.vocab(), vocab);
            assert_eq!(head.device_bytes(), (vocab * d / 2 + 4 * vocab * d / 64) as u64);
            let y = head.project(&x).unwrap();
            assert_eq!(y.dims(), [rows_in, vocab]);
            assert_eq!(y.dtype(), DType::F32);
            let y = rows_of(&y);
            let gap = worst_gap(&y, &plain);
            println!("cpu head on {}, {rows_in} rows: largest relative gap {gap:.3e}", isa.name());
            assert!(gap < 1e-5, "{}: gap {gap:.3e} to the plain product", isa.name());
            for (a, b) in y.iter().zip(&plain) {
                assert_eq!(argmax(a), argmax(b), "{}: argmax", isa.name());
            }
        }
    }
    assert!(Q4CpuHead::from_raw(&t, rows, d).is_ok(), "every row of the table");
    assert!(Q4CpuHead::from_raw(&t, rows + 1, d).is_err(), "more rows than the table holds");
    assert!(Q4CpuHead::from_raw(&t, vocab, 2560).is_err(), "a row of another width");
    let mut g128 = q4_table("model.unembed.weight", 0xF3, rows, d);
    if let RawData::Quant(q) = &mut g128.data {
        q.group = 128;
    }
    assert!(Q4CpuHead::from_raw(&g128, vocab, d).is_err(), "groups of 128");
    let head = Q4CpuHead::from_raw(&t, vocab, d).unwrap();
    let e = head.project(&Tensor::zeros((1, d + 64), DType::F32, &Device::Cpu).unwrap()).unwrap_err().to_string();
    assert!(e.contains("hidden of 2368"), "{e}");
}

// ---------------------------------------------------------------------------
// A tiny whole model, as the served file set.
// ---------------------------------------------------------------------------

/// A fixed generator: splitmix64, mapped to `[-a, a)`.
struct Gen(u64);

impl Gen {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn uniform(&mut self, a: f32) -> f32 {
        let u = (self.next() >> 40) as f32 / (1u64 << 24) as f32;
        (2.0 * u - 1.0) * a
    }

    fn tensor(&mut self, shape: &[usize], a: f32) -> Tensor {
        let n: usize = shape.iter().product();
        let v: Vec<f32> = (0..n).map(|_| self.uniform(a)).collect();
        Tensor::from_vec(v, shape, &Device::Cpu).expect("host tensor")
    }

    fn norm(&mut self, n: usize) -> Tensor {
        let v: Vec<f32> = (0..n).map(|_| 1.0 + self.uniform(0.1)).collect();
        Tensor::from_vec(v, n, &Device::Cpu).expect("host tensor")
    }
}

/// The non-expert checkpoint of the tiny model of `tests/inkling_wave1.rs`,
/// without `embed.weight` and `unembed.weight`, as the served file holds it:
/// the seven attention matrices in BF16, the rest in f32.
fn write_checkpoint(path: &Path, cfg: &InklingConfig) {
    let mut g = Gen(0x5eed_1a71);
    let (h, k, rank) = (cfg.hidden_size, cfg.sconv_kernel_size, cfg.loop_config.gate_rank);
    let mut t: Vec<(String, Tensor)> = vec![];
    let mut put = |name: String, x: Tensor| {
        let bf16 = ["wq_du", "wk_dv", "wv_dv", "gate_attn", "wo_ud", "attn_sink_proj", "wr_du"]
            .iter()
            .any(|m| name.ends_with(&format!("attn.{m}.weight")));
        let x = if bf16 { x.to_dtype(DType::BF16).unwrap() } else { x };
        t.push((format!("model.llm.{name}"), x))
    };
    for l in 0..cfg.num_hidden_layers {
        let p = |s: &str| format!("layers.{l}.{s}");
        let sliding = cfg.is_sliding(l);
        let (nh, nkv, hd) = cfg.attn_shape(l);
        let ext = cfg.layer_rel_extent(l);
        let wa = 1.0 / (h as f32).sqrt();
        put(p("attn.wq_du.weight"), g.tensor(&[nh * hd, h], wa));
        put(p("attn.q_l2_clip.scale"), g.tensor(&[nh, hd], 1.0));
        put(p("attn.wk_dv.weight"), g.tensor(&[nkv * hd, h], wa));
        put(p("attn.wv_dv.weight"), g.tensor(&[nkv * hd, h], wa));
        for which in ["k", "v"] {
            if sliding {
                put(p(&format!("attn.{which}_sconv.weight")), g.tensor(&[nkv * hd, 1, k], 0.5));
            } else {
                put(p(&format!("attn.{which}_sconv.static_w")), g.tensor(&[k, nkv * hd], 0.5));
                put(p(&format!("attn.{which}_sconv.weight_proj")), g.tensor(&[nkv * k, h], wa));
            }
        }
        put(p("attn.gate_attn.weight"), g.tensor(&[nh * hd, h], wa));
        put(p("attn.wo_ud.weight"), g.tensor(&[h, nh * hd], 1.0 / ((nh * hd) as f32).sqrt()));
        put(p("attn.attn_sink_proj.weight"), g.tensor(&[nh, h], wa));
        put(p("attn.wr_du.weight"), g.tensor(&[nh * cfg.d_rel, h], wa));
        put(p("attn.rel_l2_clip.scale"), g.tensor(&[nh, cfg.d_rel], 1.0));
        put(p("attn.rel_logits_proj.proj"), g.tensor(&[cfg.d_rel, ext], 1.0));
        if l > 0 {
            put(format!("attn_res.attn_queries.{}", l - 1), g.tensor(&[if sliding { 1 } else { 3 }, h], 1.0));
        }
        if sliding {
            put(p("attn_norm.weight"), g.norm(h));
        } else {
            for n in ["q_input_norm", "k_input_norm", "v_input_norm"] {
                put(p(&format!("{n}.weight")), g.norm(h));
            }
        }
        if cfg.is_loop_layer(l) {
            for which in ["attn", "k", "v", "mlp"] {
                put(p(&format!("controllers.{which}.down_weight")), g.tensor(&[rank, 2 * h], 0.2));
                put(p(&format!("controllers.{which}.forget_weight")), g.tensor(&[h, rank], 0.3));
                put(p(&format!("controllers.{which}.gain_weight")), g.tensor(&[h, rank], 0.3));
            }
        }
        put(format!("attn_res.mlp_queries.{l}"), g.tensor(&[h], 1.0));
        put(p("mlp_norm.weight"), g.norm(h));
        put(p("attn_sconv.weight"), g.tensor(&[h, 1, k], 0.5));
        put(p("mlp_sconv.weight"), g.tensor(&[h, 1, k], 0.5));
        put(p("mlp.gate.weight"), g.tensor(&[cfg.n_routed_experts, h], wa));
        put(p("mlp.gate.bias"), g.tensor(&[cfg.n_routed_experts], 0.05));
    }
    let n_loop = cfg.loop_config.core_end - cfg.loop_config.core_start;
    put("middle_cycle_cross_pass_queries".into(), g.tensor(&[2 * n_loop, h], 1.0));
    put("middle_cycle_cross_pass_kv_queries".into(), g.tensor(&[2 * n_loop, h], 1.0));
    put("attn_res.output_query".into(), g.tensor(&[h], 1.0));
    put("norm.weight".into(), g.norm(h));
    put("embed_norm.weight".into(), g.norm(h));
    let map: std::collections::HashMap<String, Tensor> = t.into_iter().collect();
    candle_core::safetensors::save(&map, path).unwrap();
}

/// The served file set of a tiny model under `dir`: its config, its
/// checkpoint, its two int4 tables and every layer's codes. Returns the
/// config and each layer's records.
fn write_file_set(dir: &Path) -> (InklingConfig, Vec<Set>) {
    // H = 192 so the int4 tables' rows are whole groups of 64; I = 32 is a
    // block and a tail.
    std::fs::write(dir.join("config.json"), config_json(192, 32, 8, 2).to_string()).unwrap();
    let cfg = InklingConfig::from_file(&dir.join("config.json")).unwrap();
    write_checkpoint(&dir.join("model.safetensors"), &cfg);
    let tables = dir.join("tables");
    std::fs::create_dir_all(&tables).unwrap();
    let h = cfg.hidden_size;
    llvq_llm::inkseal::write_raw_file(&tables.join("embed.q4g64.llvqraw"), &q4_table("model.embed.weight", 0xE1, cfg.vocab_size, h))
        .unwrap();
    llvq_llm::inkseal::write_raw_file(
        &tables.join("unembed.q4g64.llvqraw"),
        &q4_table("model.unembed.weight", 0xE2, cfg.vocab_size, h),
    )
    .unwrap();
    let mut sets = vec![];
    for l in 0..cfg.num_hidden_layers {
        let s = set(l, cfg.n_routed_experts, h, cfg.intermediate_size, 0x7000 + l as u64);
        let codes = dir.join(format!("layers/L{l:02}/codes"));
        std::fs::create_dir_all(&codes).unwrap();
        for (e, recs) in s.recs.iter().enumerate() {
            let mut bytes = vec![];
            for r in recs {
                llvq_artifact::write_record(&mut bytes, llvq_artifact::VERSION, &Record::Lattice(copy(r))).unwrap();
            }
            std::fs::write(codes.join(format!("e{e:03}.llvqrec")), bytes).unwrap();
        }
        std::fs::write(dir.join(format!("layers/L{l:02}/VERIFIED")), b"").unwrap();
        sets.push(s);
    }
    (cfg, sets)
}

fn load(dir: &Path) -> Inkling {
    let (m, lines) = llvq_llm::inkling::experts_cpu::load_served(
        &dir.join("config.json"),
        &dir.join("model.safetensors"),
        &dir.join("layers"),
        &dir.join("tables"),
    )
    .unwrap();
    for l in lines {
        println!("{l}");
    }
    m
}

/// [`load`] with the attention `attn` in place of the served one.
fn load_attn(dir: &Path, attn: AttnKind) -> Inkling {
    let (m, lines) = llvq_llm::inkling::experts_cpu::load_served_with(
        &dir.join("config.json"),
        &dir.join("model.safetensors"),
        &dir.join("layers"),
        &dir.join("tables"),
        attn,
    )
    .unwrap();
    for l in lines {
        println!("{l}");
    }
    m
}

/// Tolerance of the logits of the BF16 attention kernels against the candle
/// f32 attention, relative to a row's largest logit: f32 rounding of sums
/// taken in another order, through the 24 visits of the tiny model.
const ATTN_TOL: f64 = 1e-5;

#[test]
fn the_cpu_attention_kernels_track_the_candle_attention() {
    let dir = temp_dir("attn");
    let (cfg, _) = write_file_set(&dir);
    let cpu = load(&dir);
    let cand = load_attn(&dir, AttnKind::Candle);
    assert_eq!(cpu.attn(), AttnKind::Cpu);
    // What each holds: BF16 only on the served path, f32 only on the reference, twice the bytes.
    let (cf, cb) = cpu.attention_matrix_bytes();
    let (rf, rb) = cand.attention_matrix_bytes();
    println!("attention bytes: served BF16 {cb} + f32 {cf}, reference BF16 {rb} + f32 {rf}");
    assert!(cb > 0 && cf == 0 && rb == 0, "served {cb} + {cf}, reference {rb} + {rf}");
    assert_eq!(rf, 2 * cb, "the f32 reference holds twice the BF16 bytes");
    let v = cfg.unpadded_vocab_size;

    // A prompt longer than PREFILL_ROWS widens the matrices for candle's
    // matmul: the same bits as the f32 reference.
    let long: Vec<u32> = (0..21u32).map(|i| (i * 5 + 2) % v as u32).collect();
    let a = rows_of(&cpu.forward(&long, &mut cpu.new_cache(), false, true).unwrap().logits);
    let b = rows_of(&cand.forward(&long, &mut cand.new_cache(), false, true).unwrap().logits);
    let diff = a.iter().flatten().zip(b.iter().flatten()).position(|(p, q)| p.to_bits() != q.to_bits());
    assert_eq!(diff, None, "a prompt of {} rows: the widened matrices differ from the f32 ones", long.len());

    // A short prompt and the greedy steps after it through the kernels.
    let ids: Vec<u32> = (0..13u32).map(|i| (i * 7 + 3) % v as u32).collect();
    let a = rows_of(&cpu.forward(&ids, &mut cpu.new_cache(), false, true).unwrap().logits);
    let b = rows_of(&cand.forward(&ids, &mut cand.new_cache(), false, true).unwrap().logits);
    let gap = worst_gap(&a, &b);
    println!("tiny model: a prompt of {} rows, BF16 kernels against candle f32, largest relative gap {gap:.3e}", ids.len());
    assert!(gap < ATTN_TOL, "prompt: gap {gap:.3e}");
    let (mut ca, mut cb) = (cpu.new_cache(), cand.new_cache());
    let mut ta = argmax(&rows_of(&cpu.prefill(&ids, &mut ca).unwrap().reshape((1, v)).unwrap())[0]) as u32;
    let mut tb = argmax(&rows_of(&cand.prefill(&ids, &mut cb).unwrap().reshape((1, v)).unwrap())[0]) as u32;
    let mut worst = 0f64;
    for step in 0..8 {
        assert_eq!(ta, tb, "step {step}");
        let pos = ca.len();
        let la = rows_of(&cpu.step(ta, pos, &mut ca).unwrap().reshape((1, v)).unwrap());
        let lb = rows_of(&cand.step(tb, pos, &mut cb).unwrap().reshape((1, v)).unwrap());
        worst = worst.max(worst_gap(&la, &lb));
        ta = argmax(&la[0]) as u32;
        tb = argmax(&lb[0]) as u32;
    }
    println!("tiny model: 8 greedy steps, BF16 kernels against candle f32, largest relative gap {worst:.3e}");
    assert!(worst < ATTN_TOL, "steps: gap {worst:.3e}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_tiny_served_file_set_loads_on_the_cpu_and_matches_its_dense_reconstruction() {
    let dir = temp_dir("fileset");
    let (cfg, sets) = write_file_set(&dir);
    let cpu = load(&dir);
    assert_eq!(cpu.substituted_layers().len(), cfg.num_hidden_layers, "every layer on the CPU experts");
    assert!(!cpu.has_bf16_head(), "the int4 head, no BF16 rows");

    // The same load, every layer's experts then replaced by the dense reconstructions.
    let mut dense = load(&dir);
    for s in &sets {
        dense.set_experts(s.layer, LayerExperts::Source(Box::new(dense_experts(s))));
    }

    let ids: Vec<u32> = (0..13u32).map(|i| (i * 7 + 3) % cfg.unpadded_vocab_size as u32).collect();
    let a = rows_of(&cpu.forward(&ids, &mut cpu.new_cache(), false, true).unwrap().logits);
    let b = rows_of(&dense.forward(&ids, &mut dense.new_cache(), false, true).unwrap().logits);
    assert_eq!(a.len(), ids.len());
    assert_eq!(a[0].len(), cfg.unpadded_vocab_size);
    let gap = worst_gap(&a, &b);
    println!("tiny model: logits of the CPU experts against the dense reconstruction, largest relative gap {gap:.3e}");
    assert!(gap < 1e-5, "gap {gap:.3e}");

    // Greedy steps on both: the same tokens.
    let (mut ca, mut cb) = (cpu.new_cache(), dense.new_cache());
    let mut ta = argmax(&rows_of(&cpu.prefill(&ids, &mut ca).unwrap().reshape((1, cfg.unpadded_vocab_size)).unwrap())[0]) as u32;
    let mut tb = argmax(&rows_of(&dense.prefill(&ids, &mut cb).unwrap().reshape((1, cfg.unpadded_vocab_size)).unwrap())[0]) as u32;
    for step in 0..6 {
        assert_eq!(ta, tb, "step {step}");
        let pos = ca.len();
        let la = cpu.step(ta, pos, &mut ca).unwrap().reshape((1, cfg.unpadded_vocab_size)).unwrap();
        let lb = dense.step(tb, pos, &mut cb).unwrap().reshape((1, cfg.unpadded_vocab_size)).unwrap();
        ta = argmax(&rows_of(&la)[0]) as u32;
        tb = argmax(&rows_of(&lb)[0]) as u32;
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// One decode step charges every part of the layer `inkcpu` splits: each
/// counter of `fused_cpu::phases` named for the layer moves.
#[test]
fn a_step_on_the_cpu_charges_every_part_of_the_layer() {
    use llvq_llm::fused_cpu::phases::{snapshot, NAMES};
    let dir = temp_dir("phases");
    let (cfg, _) = write_file_set(&dir);
    let m = load(&dir);
    let ids: Vec<u32> = (0..5u32).map(|i| (i * 3 + 1) % cfg.unpadded_vocab_size as u32).collect();
    let mut c = m.new_cache();
    let next = argmax(&rows_of(&m.prefill(&ids, &mut c).unwrap().reshape((1, cfg.unpadded_vocab_size)).unwrap())[0]) as u32;
    let before = snapshot();
    let pos = c.len();
    m.step(next, pos, &mut c).unwrap();
    let after = snapshot();
    // The counters are global and other tests run beside this one: a lower bound only.
    let visits = cfg.num_hidden_layers + (cfg.loop_config.num_passes - 1) * (cfg.loop_config.core_end - cfg.loop_config.core_start);
    for (name, at_least) in
        [("attn proj", 8 * visits), ("attn cache", 2 * visits), ("attn scores", visits), ("router", visits), ("convs", 4 * visits), ("glue", 3 * visits)]
    {
        let i = NAMES.iter().position(|n| *n == name).unwrap_or_else(|| panic!("no phase {name:?}"));
        let calls = (after[i].1 - before[i].1) as usize;
        println!("one step: {name} {calls} calls, {visits} visits");
        assert!(calls >= at_least, "{name}: {calls} calls in a step of {visits} visits, {at_least} at least");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_prompt_in_pieces_gives_what_one_call_gives_on_the_cpu() {
    let dir = temp_dir("pieces");
    let (cfg, _) = write_file_set(&dir);
    let mut m = load(&dir);
    let ids: Vec<u32> = (0..13u32).map(|i| (i * 11 + 5) % cfg.unpadded_vocab_size as u32).collect();
    m.set_prefill_piece(0);
    let one = rows_of(&m.forward_pieces(&ids, &mut m.new_cache(), true).unwrap());
    m.set_prefill_piece(4);
    let pieces = rows_of(&m.forward_pieces(&ids, &mut m.new_cache(), true).unwrap());
    assert_eq!(one.len(), pieces.len());
    let gap = worst_gap(&pieces, &one);
    println!("tiny model: a prompt in pieces of 4 against one call, largest relative gap {gap:.3e}");
    assert!(gap < 1e-4, "gap {gap:.3e}");
    let _ = std::fs::remove_dir_all(&dir);
}
