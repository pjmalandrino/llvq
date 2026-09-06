//! # The two int4 paths, and the walker that has to know about the second one
//!
//! `v_proj` in int4 g128 beside the lattice matrices (`docs/ROADMAP.md` §2.3,
//! operator decision of 2026-09-06: +3.47 pp of MMLU, IC95 [+1.42; +5.57],
//! for +0.0493 b/param). Two things about that can go wrong quietly, and both
//! are here.
//!
//! 1. **The measurement path and the file path can drift.** `LLVQ_RESTORE_Q4`
//!    quantizes and dequantizes in memory; the file path quantizes and writes.
//!    If the two stop agreeing the gap is of order one ULP per weight —
//!    invisible in perplexity, visible nowhere, and it would invalidate the
//!    measurement that decided the format.
//! 2. **The generic record walker can miss the branch.** `shard_extent` sizes
//!    a record from its own fields; an int4 record has three fewer arrays than
//!    a lattice one, and the lattice formula overshoots it by `d_out·8` bytes.
//!    The symptom is a shard declared shorter than it is, with a warning that
//!    reads exactly like a normal interruption, and a resume that re-quantizes
//!    blocks already done.
//!
//! No card, no checkpoint, no model download: a tiny Qwen3 on a `VarMap` and
//! bytes in cargo's test tmpdir.

use candle_core::{DType, Device, Tensor};
use candle_nn::{Activation, VarBuilder, VarMap};
use candle_transformers::models::qwen3::Config;
use llvq_llm::artifact2::{shard_extent, ArtifactWriter, CodeKind, KindSet, Record};
use llvq_llm::model::{Act, Qwen3};
use std::path::PathBuf;

fn tmp(name: &str) -> String {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    std::fs::create_dir_all(&dir).expect("tmpdir");
    dir.join(name).to_str().expect("utf-8 path").to_string()
}

// ---------------------------------------------------------------------------
// 1 — the two quantizers are one quantizer
// ---------------------------------------------------------------------------

#[test]
fn the_two_group_constants_agree() {
    // One constant derived from the other, not two literals in two crates: a
    // file written with one stride and read with the other decodes plausible,
    // wrong weights.
    assert_eq!(llvq_llm::sealed::Q4_GROUP, llvq_artifact::INT4G128_GROUP);
    assert_eq!(llvq_llm::sealed::Q4_GROUP, 128);
    assert_eq!(llvq_artifact::INT4G128_BITS, 4);
}

#[test]
fn the_two_int4_paths_agree_bit_for_bit() {
    let dev = Device::Cpu;
    let (d_out, d_in) = (5usize, 256usize);
    // Deliberately uneven: a row of large values, a row of tiny ones, a
    // constant row (whose f16 range rounds to zero and whose bias carries the
    // value), and rows around zero. Each exercises a different branch of the
    // affine fit.
    let data: Vec<f32> = (0..d_out * d_in)
        .map(|i| {
            let (row, col) = (i / d_in, i % d_in);
            match row {
                0 => ((col as f32) * 0.31).sin() * 40.0,
                1 => ((col as f32) * 0.17).cos() * 1e-4,
                2 => 0.125,
                3 => (col as f32) * 1e-3 - 0.128,
                _ => ((col as f32) * 0.07).sin() * 0.9 + 0.05,
            }
        })
        .collect();
    let t = Tensor::from_vec(data, (d_out, d_in), &dev).expect("tensor");

    // Arm A: the measurement instrument, untouched — what `LLVQ_RESTORE_Q4`
    // hands the model.
    let a = llvq_llm::sealed::quantize_dequantize_q4(&t, "v", 128, DType::F32)
        .expect("arm A")
        .flatten_all()
        .expect("flatten")
        .to_vec1::<f32>()
        .expect("readback");

    // Arm B: the file. Quantize, write, read the bytes back, decode.
    let rec = llvq_llm::sealed::int4_record("v", &t).expect("arm B");
    let mut bytes = Vec::new();
    llvq_artifact::write_matrix_int4(&mut bytes, llvq_artifact::FIRST_KINDED_VERSION, &rec)
        .expect("write");
    let Record::Int4(back) =
        llvq_artifact::read_record(&mut &bytes[..], llvq_artifact::FIRST_KINDED_VERSION)
            .expect("read")
    else {
        panic!("not an int4 record");
    };
    let b = back.to_f32();

    assert_eq!(a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(&b).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "weight {i} (row {}): the measured arm and the stored arm disagree",
            i / d_in
        );
    }
    // And the rate the two share is the one the cost table quotes.
    assert_eq!(rec.bits() as f64 / (d_out * d_in) as f64, 4.25);
}

// ---------------------------------------------------------------------------
// 2 — the generic walker
// ---------------------------------------------------------------------------

/// A shard of `blocks` blocks of `matrices_per_block()` records, in which the
/// record at `int4_at` of each block is an int4 g128 matrix and the rest are
/// Ball. `shard_extent` walks record heads only, so the names and the weights
/// are arbitrary; the shape of each record is not.
fn mixed_shard(path: &str, blocks: usize, int4_at: usize) {
    use llvq_quant::quantizer::BlockCode;
    let per = llvq_llm::calib::matrices_per_block();
    let ix = llvq_search::index::Indexer::new();
    let point = ix.decode(0).expect("index 0 decodes");
    let f = std::fs::File::create(path).expect("create");
    let mut w = ArtifactWriter::with_kinds(
        std::io::BufWriter::new(f),
        llvq_artifact::FIRST_KINDED_VERSION,
        (per * blocks) as u32,
        CodeKind::Ball,
        KindSet::of(CodeKind::Ball).with(CodeKind::Int4G128),
    )
    .expect("header");
    for t in 0..blocks {
        for j in 0..per {
            let name = format!("model.layers.{t}.m{j}.weight");
            if j == int4_at {
                w.push_int4(&llvq_artifact::Int4Matrix {
                    name,
                    d_out: 3,
                    d_in: 256,
                    bits: llvq_artifact::INT4G128_BITS,
                    group: llvq_artifact::INT4G128_GROUP,
                    packed: vec![0x5a; 3 * 256 / 2],
                    scales: vec![0x3c00; 3 * 2],
                    biases: vec![0; 3 * 2],
                })
                .expect("int4");
            } else {
                // d_out = 3 and a tail: the lattice formula the walker used to
                // apply unconditionally adds `d_out·8` for the row scales and
                // `d_out·(d_in % 24)·4` for the tail, and both are what an
                // int4 record does not have.
                w.push(&llvq_artifact::QuantizedMatrix {
                    name,
                    d_out: 3,
                    d_in: 2 * llvq_core::DIM + 5,
                    codes: (0..6).map(|_| BlockCode { point, gain: 0 }).collect(),
                    row_scales: vec![1.0, 2.0, 3.0],
                    centroids: vec![0.5, 1.5],
                    rotation_seed: None,
                    shell_cap: 12,
                    tail: vec![0.25; 3 * 5],
                })
                .expect("ball");
            }
        }
    }
    w.finish().expect("finish");
}

#[test]
fn shard_extent_walks_a_mixed_shard() {
    let per = llvq_llm::calib::matrices_per_block();
    // Every position in a block, so the branch is exercised wherever the int4
    // record falls — first, last and in between.
    for int4_at in [0usize, 2, per - 1] {
        let p = tmp(&format!("int4-shard-{int4_at}.llvq"));
        mixed_shard(&p, 2, int4_at);
        let (matrices, blocks) = shard_extent(&p).expect("walk");
        assert_eq!(
            (matrices, blocks),
            (per * 2, 2),
            "int4 at {int4_at}: the walker stopped short, which is how a resume \
             silently re-quantizes blocks already done"
        );
    }
}

// ---------------------------------------------------------------------------
// 3 — the resume checks the kind per projection
// ---------------------------------------------------------------------------

fn tiny() -> Config {
    Config {
        vocab_size: 128,
        hidden_size: 32,
        intermediate_size: 64,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        head_dim: 8,
        attention_bias: false,
        num_key_value_heads: 2,
        max_position_embeddings: 64,
        sliding_window: None,
        max_window_layers: 0,
        tie_word_embeddings: true,
        rope_theta: 10_000.0,
        rms_norm_eps: 1e-6,
        use_sliding_window: false,
        hidden_act: Activation::Silu,
    }
}

fn fresh(map: &VarMap, dev: &Device) -> Qwen3 {
    let vb = VarBuilder::from_varmap(map, DType::F32, dev);
    Qwen3::new(&tiny(), vb, llvq_llm::kvq::KvMode::F16).expect("tiny model builds")
}

/// A Ball shard of `blocks` blocks, in the order the loop writes — enough for
/// the resume to walk it and reach `v_proj`.
fn ball_shard(path: &str, model: &Qwen3, blocks: usize) {
    use llvq_quant::quantizer::BlockCode;
    let ix = llvq_search::index::Indexer::new();
    let point = ix.decode(0).expect("index 0 decodes");
    let per = llvq_llm::calib::matrices_per_block();
    let f = std::fs::File::create(path).expect("create");
    let mut w = ArtifactWriter::new(std::io::BufWriter::new(f), (per * blocks) as u32)
        .expect("header");
    for t in 0..blocks {
        for (_, proj) in llvq_llm::calib::block_matrix_plan() {
            let (d_out, d_in) = model.blocks[t].linear(proj).weight().dims2().expect("dims");
            let nb = d_in / llvq_core::DIM;
            w.push(&llvq_artifact::QuantizedMatrix {
                name: llvq_llm::artifact::key(t, proj),
                d_out,
                d_in,
                codes: (0..d_out * nb).map(|_| BlockCode { point, gain: 0 }).collect(),
                row_scales: vec![1.0; d_out],
                centroids: vec![0.5, 1.5],
                rotation_seed: None,
                shell_cap: 12,
                tail: vec![0.0; d_out * (d_in % llvq_core::DIM)],
            })
            .expect("push");
        }
    }
    w.finish().expect("finish");
}

#[test]
fn resume_refuses_a_shard_whose_v_proj_is_lattice() {
    let dev = Device::Cpu;
    let map = VarMap::new();
    let mut model = fresh(&map, &dev);
    let shard = tmp("int4-resume-shard.llvq");
    ball_shard(&shard, &model, 2);

    // The run writes `v_proj` in int4; the shard's is Ball. Every other check
    // passes — the names, the dimensions, the shell cap, the centroid count,
    // the rotation — and the file is well formed. Only a per-matrix kind
    // check sees it.
    let expect = llvq_llm::artifact2::ShardExpect {
        kind: CodeKind::Ball,
        int4_types: vec!["v_proj".into()],
        shell_cap: 12,
        centroids: 2,
        rotation_seed: None,
    };
    let out = tmp("int4-resume-out.llvq");
    let f = std::fs::File::create(&out).expect("create");
    let mut w = ArtifactWriter::with_kinds(
        std::io::BufWriter::new(f),
        llvq_artifact::FIRST_KINDED_VERSION,
        14,
        CodeKind::Ball,
        KindSet::of(CodeKind::Ball).with(CodeKind::Int4G128),
    )
    .expect("header");
    let e = llvq_llm::artifact2::resume_from_shard(&mut model, &shard, &mut w, &expect, &dev)
        .expect_err("a lattice v_proj under an int4 run must be refused")
        .to_string();
    assert!(e.contains("v_proj"), "the refusal must name the projection: {e}");
    assert!(e.contains("Ball"), "the refusal must name the shard's kind: {e}");
    assert!(e.contains("Int4G128"), "the refusal must name the run's kind: {e}");

    // The control: with an empty list the same shard resumes.
    let mut model = fresh(&map, &dev);
    let expect = llvq_llm::artifact2::ShardExpect {
        int4_types: Vec::new(),
        ..expect
    };
    let out = tmp("int4-resume-ok.llvq");
    let f = std::fs::File::create(&out).expect("create");
    let mut w = ArtifactWriter::new(std::io::BufWriter::new(f), 14).expect("header");
    llvq_llm::artifact2::resume_from_shard(&mut model, &shard, &mut w, &expect, &dev)
        .expect("the same shard, with no int4 type asked for");
}

// ---------------------------------------------------------------------------
// 4 — the proof that a run writes what it measured, on the int4 half
// ---------------------------------------------------------------------------

/// A model every one of whose projections has `d_in = 128`, so all seven can
/// be stored as int4 g128 and the verification needs no lattice arm.
fn wide() -> Config {
    Config {
        vocab_size: 64,
        hidden_size: 128,
        intermediate_size: 128,
        num_hidden_layers: 1,
        num_attention_heads: 8,
        head_dim: 16,
        attention_bias: false,
        num_key_value_heads: 2,
        max_position_embeddings: 32,
        sliding_window: None,
        max_window_layers: 0,
        tie_word_embeddings: true,
        rope_theta: 10_000.0,
        rms_norm_eps: 1e-6,
        use_sliding_window: false,
        hidden_act: Activation::Silu,
    }
}

/// Write every projection of `model` as int4, after replacing the model's own
/// weights with the dequantized values — so the file and the model hold the
/// same numbers and `verify_artifact` must pass.
fn all_int4(path: &str, model: &mut Qwen3) -> Vec<String> {
    let per = llvq_llm::calib::matrices_per_block();
    let f = std::fs::File::create(path).expect("create");
    let mut w = ArtifactWriter::with_kinds(
        std::io::BufWriter::new(f),
        llvq_artifact::FIRST_KINDED_VERSION,
        per as u32,
        CodeKind::Ball,
        KindSet::of(CodeKind::Ball).with(CodeKind::Int4G128),
    )
    .expect("header");
    let mut names = Vec::new();
    for (_, proj) in llvq_llm::calib::block_matrix_plan() {
        let key = llvq_llm::artifact::key(0, proj);
        let lin = model.blocks[0].linear_mut(proj);
        let rec = llvq_llm::sealed::int4_record(&key, lin.weight()).expect("quantize");
        let (d_out, d_in) = (rec.d_out, rec.d_in);
        let deq = Tensor::from_vec(rec.to_f32(), (d_out, d_in), &Device::Cpu).expect("tensor");
        *lin = candle_nn::Linear::new(deq, None);
        w.push_int4(&rec).expect("push");
        names.push(key);
    }
    w.finish().expect("finish");
    names
}

#[test]
fn verify_artifact_checks_the_int4_half() {
    let dev = Device::Cpu;
    let map = VarMap::new();
    let vb = VarBuilder::from_varmap(&map, DType::F32, &dev);
    let mut model = Qwen3::new(&wide(), vb, llvq_llm::kvq::KvMode::F16).expect("model");
    let path = tmp("int4-verify.llvq");
    let names = all_int4(&path, &mut model);
    let types: Vec<String> = llvq_llm::sealed::PROJ_TYPES.iter().map(|s| s.to_string()).collect();

    // The control: the file decodes to the evaluated weights.
    llvq_llm::artifact2::verify_artifact(&path, &model, DType::F32, CodeKind::Ball, &types)
        .expect("an int4 file that matches its model");

    // One bias moved by one ULP. Nothing about the file is malformed; every
    // weight of one group is wrong by one quantization step of the bias.
    let bytes = std::fs::read(&path).expect("read");
    let target = &names[2]; // v_proj, the matrix the decision is about
    let at = bytes
        .windows(target.len())
        .position(|w| w == target.as_bytes())
        .expect("the record is in the file");
    // Past the name: d_out, d_in, shell cap, kind, centroids, seed, flag,
    // payload length, bits, group, packed length, the nibbles, the group
    // count, the scales — then the biases.
    let head = at + target.len() + 4 * 4 + 4 + 8 + 4 + 8 + 4 + 4 + 8;
    let (d_out, d_in) = model.blocks[0].linear("self_attn.v_proj").weight().dims2().expect("dims");
    let groups = d_out * (d_in / llvq_artifact::INT4G128_GROUP);
    let biases_at = head + d_out * d_in / 2 + 8 + groups * 2;
    let mut broken = bytes.clone();
    broken[biases_at] ^= 1;
    let bad = tmp("int4-verify-broken.llvq");
    std::fs::write(&bad, &broken).expect("write");
    let e = llvq_llm::artifact2::verify_artifact(&bad, &model, DType::F32, CodeKind::Ball, &types)
        .expect_err("a bias moved by one ULP must be caught")
        .to_string();
    assert!(e.contains(target), "the refusal must name the matrix: {e}");

    // And a record wearing the wrong label is refused before any weight is
    // compared: the first record's kind tag flipped to Ball, under a run that
    // wrote int4 for that projection.
    let first = bytes
        .windows(names[0].len())
        .position(|w| w == names[0].as_bytes())
        .expect("the first record is in the file");
    let tag_at = first + names[0].len() + 12;
    let mut mislabelled = bytes.clone();
    assert_eq!(
        u32::from_le_bytes(mislabelled[tag_at..tag_at + 4].try_into().unwrap()),
        CodeKind::Int4G128.tag()
    );
    mislabelled[tag_at..tag_at + 4].copy_from_slice(&CodeKind::Ball.tag().to_le_bytes());
    let bad = tmp("int4-verify-mislabelled.llvq");
    std::fs::write(&bad, &mislabelled).expect("write");
    assert!(
        llvq_llm::artifact2::verify_artifact(&bad, &model, DType::F32, CodeKind::Ball, &types)
            .is_err(),
        "a mislabelled record passed verification"
    );

    // Not dead weight: the model has the seven projections the plan names.
    assert_eq!(names.len(), llvq_llm::calib::matrices_per_block());
    let _ = Act::ALL;
}

// ---------------------------------------------------------------------------
// 5 — the branch that produces a mixed file, in the loop that produces it
// ---------------------------------------------------------------------------
//
// `llvq_llm::calib`'s int4 branch is the only place an encoding run can write
// a mixed file, and until this section nothing exercised it: every `RunConfig`
// of the suite set `int4_types: Vec::new()`. Two mutations survived a green
// `cargo test` — dropping the `*lin = …` that puts the dequantized weights
// back in the model, and moving the branch's weights into `Report::weights` —
// and neither would fail a run, only its numbers and its Hessians.

/// A sink that keeps what it was handed, per kind.
#[derive(Default)]
struct Kinds {
    lattice: Vec<String>,
    int4: Vec<llvq_artifact::Int4Matrix>,
}

impl llvq_llm::calib::MatrixSink for Kinds {
    fn push(&mut self, m: llvq_llm::artifact2::QuantizedMatrix) -> anyhow::Result<()> {
        self.lattice.push(m.name);
        Ok(())
    }
    fn push_int4(&mut self, m: llvq_artifact::Int4Matrix) -> anyhow::Result<()> {
        self.int4.push(m);
        Ok(())
    }
}

/// The narrowest model whose `v_proj` can be int4 at all: `d_in = 128` is the
/// group, so `hidden_size` cannot go below it. Everything else is as small as
/// the shapes allow, because the lattice arm is a nearest-neighbour search per
/// 24 weights and this file runs in the fast loop.
fn narrow() -> Config {
    Config {
        vocab_size: 64,
        hidden_size: 128,
        intermediate_size: 32,
        num_hidden_layers: 1,
        num_attention_heads: 2,
        head_dim: 16,
        attention_bias: false,
        num_key_value_heads: 1,
        max_position_embeddings: 32,
        sliding_window: None,
        max_window_layers: 0,
        tie_word_embeddings: true,
        rope_theta: 10_000.0,
        rms_norm_eps: 1e-6,
        use_sliding_window: false,
        hidden_act: Activation::Silu,
    }
}

fn windows(cfg: &Config, seq: usize, n: usize, dev: &Device) -> Vec<Tensor> {
    (0..n)
        .map(|w| {
            let len = seq * cfg.hidden_size;
            let data: Vec<f32> = (0..len)
                .map(|i| (((i + 977 * w) as f32) * 0.37).sin() * 0.8 + 0.1)
                .collect();
            Tensor::from_vec(data, (1, seq, cfg.hidden_size), dev).expect("window")
        })
        .collect()
}

fn mixed_run(int4_types: Vec<String>, start: usize, limit: usize) -> llvq_llm::calib::RunConfig {
    llvq_llm::calib::RunConfig {
        h_shrink: 1.0,
        int4_types,
        gptq: llvq_quant::gptq::GptqConfig {
            block: llvq_core::DIM,
            retract: true,
            group_scales: false,
            design_c: false,
            lambda: 1e-2,
            tail: llvq_quant::gptq::TailPolicy::KeepExact,
        },
        damping: 1e-2,
        codebook: llvq_llm::calib::Codebook::ShapeGain {
            gain_bits: 1,
            max_shell: 12,
            free_magnitude: false,
            level_cap: 5,
        },
        threads: 1,
        start,
        limit,
        rotation_seed: Some(0x11_0FEED),
    }
}

#[test]
fn the_loop_writes_v_proj_as_int4_and_puts_it_back_in_the_model() {
    let dev = Device::Cpu;
    let map = VarMap::new();
    let vb = VarBuilder::from_varmap(&map, DType::F32, &dev);
    let mut model = Qwen3::new(&narrow(), vb, llvq_llm::kvq::KvMode::F16).expect("model");
    let mut hidden = windows(&narrow(), 8, 2, &dev);
    let mut sink = Kinds::default();
    let report = llvq_llm::calib::quantize_model_capturing(
        &mut model,
        &mut hidden,
        &mixed_run(vec!["v_proj".into()], 0, usize::MAX),
        |_, _, _| {},
        Some(&mut sink),
    )
    .expect("mixed run");

    // One matrix took the int4 branch, and it is the one named. A prefix match
    // on the short name would also catch `v_proj` inside another projection's
    // name, so the whole key is asserted.
    assert_eq!(sink.int4.len(), 1, "int4 records: {:?}", sink.int4.len());
    assert_eq!(sink.int4[0].name, llvq_llm::artifact::key(0, "self_attn.v_proj"));
    assert_eq!(sink.lattice.len(), llvq_llm::calib::matrices_per_block() - 1);
    assert!(
        !sink.lattice.iter().any(|n| n.contains("v_proj")),
        "v_proj reached the lattice sink as well: {:?}",
        sink.lattice
    );

    // The model must hold what the file holds. Drop the `*lin = …` and the
    // block's later activations are built on full-precision `v_proj` while
    // the file stores the quantized one: every Hessian after it in the block
    // is fitted to a model nobody will run, and no error surfaces.
    let held = model.blocks[0]
        .linear("self_attn.v_proj")
        .weight()
        .flatten_all()
        .expect("flatten")
        .to_vec1::<f32>()
        .expect("readback");
    let stored = sink.int4[0].to_f32();
    assert_eq!(held.len(), stored.len());
    for (k, (h, s)) in held.iter().zip(&stored).enumerate() {
        assert_eq!(h.to_bits(), s.to_bits(), "v_proj weight {k}: model {h:e}, record {s:e}");
    }

    // And the two accountings stay apart. `Report::weights` is the divisor of
    // the lattice rate; an int4 matrix in it would be billed at `block_bits`
    // per 24 weights, a rate it does not have.
    let (d_out, d_in) = (sink.int4[0].d_out, sink.int4[0].d_in);
    let int4_weights = (d_out * d_in) as u64;
    assert_eq!(report.int4_matrices, 1);
    assert_eq!(report.int4_weights, int4_weights);
    assert_eq!(report.matrices, llvq_llm::calib::matrices_per_block());
    let lattice: u64 = llvq_llm::calib::block_matrix_plan()
        .iter()
        .filter(|(_, proj)| !proj.ends_with("v_proj"))
        .map(|(_, proj)| {
            let (o, i) = model.blocks[0].linear(proj).weight().dims2().expect("dims");
            (o * i) as u64
        })
        .sum();
    assert_eq!(report.weights, lattice, "int4 weights leaked into the lattice divisor");
    assert_eq!(report.quantized_weights(), lattice + int4_weights - report.tail_weights);
}

/// One segment or two, the same file and the same numbers.
///
/// The resume counts a record's weights from its own dimensions; if it counts
/// an int4 record the way it counts a lattice one, the same configuration
/// reports a different `Report::weights` — and a different lattice rate — for
/// every block it happened to be interrupted at.
#[test]
fn a_mixed_run_reports_the_same_totals_in_one_segment_or_two() {
    let dev = Device::Cpu;
    let map = VarMap::new();
    let types = vec!["v_proj".to_string()];
    let per = llvq_llm::calib::matrices_per_block();

    let one_path = tmp("int4-seg-one.llvq");
    let mut model = Qwen3::new(
        &narrow(),
        VarBuilder::from_varmap(&map, DType::F32, &dev),
        llvq_llm::kvq::KvMode::F16,
    )
    .expect("model");
    let mut hidden = windows(&narrow(), 8, 2, &dev);
    let mut sink = MixedSink::create(&one_path, per as u32);
    let one = llvq_llm::calib::quantize_model_capturing(
        &mut model,
        &mut hidden,
        &mixed_run(types.clone(), 0, usize::MAX),
        |_, _, _| {},
        Some(&mut sink),
    )
    .expect("one segment");
    sink.finish();

    // Segment A: block 0 only, into a shard. `narrow()` has one block, so the
    // shard is the whole model and the resume walks every record of it.
    let shard = tmp("int4-seg-shard.llvq");
    let mut model = Qwen3::new(
        &narrow(),
        VarBuilder::from_varmap(&map, DType::F32, &dev),
        llvq_llm::kvq::KvMode::F16,
    )
    .expect("model");
    let mut hidden = windows(&narrow(), 8, 2, &dev);
    let mut sink = MixedSink::create(&shard, per as u32);
    llvq_llm::calib::quantize_model_capturing(
        &mut model,
        &mut hidden,
        &mixed_run(types.clone(), 0, 1),
        |_, _, _| {},
        Some(&mut sink),
    )
    .expect("segment A");
    sink.finish();

    // Segment B: resume the shard, quantize nothing more.
    let two_path = tmp("int4-seg-two.llvq");
    let mut model = Qwen3::new(
        &narrow(),
        VarBuilder::from_varmap(&map, DType::F32, &dev),
        llvq_llm::kvq::KvMode::F16,
    )
    .expect("model");
    let mut sink = MixedSink::create(&two_path, per as u32);
    let expect = llvq_llm::artifact2::ShardExpect {
        kind: CodeKind::Ball,
        int4_types: types.clone(),
        shell_cap: 12,
        centroids: 2,
        rotation_seed: Some(0x11_0FEED),
    };
    let resumed =
        llvq_llm::artifact2::resume_from_shard(&mut model, &shard, &mut sink.0, &expect, &dev)
            .expect("resume");
    sink.finish();

    // Fold exactly as `bin/smoke` folds, and demand the single segment's
    // numbers back.
    let mut two = llvq_llm::calib::Report {
        block_bits: one.block_bits,
        block_len: one.block_len,
        ..Default::default()
    };
    two.matrices += resumed.matrices;
    two.weights += resumed.weights;
    two.tail_weights += resumed.tail_weights;
    two.rows += resumed.rows;
    two.int4_matrices += resumed.int4_matrices;
    two.int4_weights += resumed.int4_weights;

    assert_eq!(two.matrices, one.matrices);
    assert_eq!(two.weights, one.weights, "the resume counts int4 weights as lattice ones");
    assert_eq!(two.int4_weights, one.int4_weights);
    assert_eq!(two.int4_matrices, one.int4_matrices);
    assert_eq!(two.tail_weights, one.tail_weights);
    assert_eq!(two.quantized_weights(), one.quantized_weights());
    assert_eq!(
        two.bits_per_weight().to_bits(),
        one.bits_per_weight().to_bits(),
        "the lattice rate moved with the interruption point"
    );
    assert_eq!(
        std::fs::read(&one_path).expect("one"),
        std::fs::read(&two_path).expect("two"),
        "two segments must produce the single run's bytes"
    );
}

/// A sink that writes a real mixed file: Ball by default, int4 declared.
struct MixedSink(ArtifactWriter<std::io::BufWriter<std::fs::File>>);

impl MixedSink {
    fn create(path: &str, n: u32) -> Self {
        let f = std::fs::File::create(path).expect("create");
        Self(
            ArtifactWriter::with_kinds(
                std::io::BufWriter::new(f),
                llvq_artifact::FIRST_KINDED_VERSION,
                n,
                CodeKind::Ball,
                KindSet::of(CodeKind::Ball).with(CodeKind::Int4G128),
            )
            .expect("header"),
        )
    }
    fn finish(self) {
        self.0.finish().expect("finish");
    }
}

impl llvq_llm::calib::MatrixSink for MixedSink {
    fn push(&mut self, m: llvq_llm::artifact2::QuantizedMatrix) -> anyhow::Result<()> {
        Ok(self.0.push(&m)?)
    }
    fn push_int4(&mut self, m: llvq_artifact::Int4Matrix) -> anyhow::Result<()> {
        Ok(self.0.push_int4(&m)?)
    }
}
