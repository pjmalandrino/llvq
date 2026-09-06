//! # Step 4 of the Trio plan: the wiring, end to end on a model
//!
//! `llvq-quant` proves the quantizer, `llvq-artifact` proves the format. What
//! neither can prove is the seam this file exists for: that a run configured
//! with `Codebook::Trio` writes a **v5 Trio file** whose records decode back
//! to the weights the model was left holding, bit for bit — the promise
//! `bin/smoke`'s `verify_artifact` makes on every run that writes a file, and
//! the only reason a rate is a measurement rather than a claim.
//!
//! Three things can go wrong at this seam and nowhere else, and each has a
//! test below:
//!
//! * the codebook token resolves to the **other quantizer** — the rate line
//!   is identical (both arms spend 48 bits per block, deliberately), so
//!   nothing in a log would say which map the file holds;
//! * the sink writes the **other kind** into the header, and 47-bit reads stay
//!   aligned with the stream while meaning something else at every block;
//! * a resume splices two halves quantized on **different maps**, which is the
//!   same failure with the file's own two halves.
//!
//! The tiny model is `resume.rs`'s: two blocks, every matrix carrying a tail,
//! four deterministic windows. It runs in the fast loop — the whole file is a
//! couple of seconds — because a seam nobody exercises before a commit is a
//! seam that breaks four hours into a 4B run.

use candle_core::{DType, Device, Tensor};
use candle_nn::{Activation, VarBuilder, VarMap};
use candle_transformers::models::qwen3::Config;
use llvq_artifact::{CodeKind, KindSet};
use llvq_llm::artifact2::{
    decode_matrix, read_header, read_matrix_with, resume_from_shard, ArtifactWriter, Codebooks,
    ShardExpect,
};
use llvq_llm::calib::{
    matrices_per_block, quantize_model_capturing, Codebook, MatrixSink, Report, RunConfig,
};
use llvq_llm::model::{Act, Qwen3};
use llvq_quant::gptq::{GptqConfig, TailPolicy};

/// The rotation is on, as it is in every artifact test here: it is the input
/// whose per-matrix value is derived rather than stored twice, so a decode
/// that rebuilt it wrong yields plausible garbage instead of an error.
const ROT: u64 = 0x11_0FEED;

/// The two arms this step compares, at the same 48 bits per block.
fn trio() -> Codebook {
    Codebook::Trio { gain_bits: 1 }
}

fn ball() -> Codebook {
    Codebook::ShapeGain {
        gain_bits: 1,
        max_shell: 12,
        free_magnitude: false,
        level_cap: 5,
    }
}

/// Two transformer blocks, every matrix one 24-wide block plus a tail.
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

/// A model on the *same* weights every time — the loop rebuilds each `Linear`
/// rather than writing through the `Var`, so the `VarMap` survives a run.
fn fresh(map: &VarMap, dev: &Device) -> Qwen3 {
    let vb = VarBuilder::from_varmap(map, DType::F32, dev);
    Qwen3::new(&tiny(), vb, llvq_llm::kvq::KvMode::F16).expect("tiny model builds")
}

fn windows(dev: &Device) -> Vec<Tensor> {
    let cfg = tiny();
    (0..4)
        .map(|w| {
            let n = 32 * cfg.hidden_size;
            let data: Vec<f32> = (0..n)
                .map(|i| (((i + 977 * w) as f32) * 0.37).sin() * 0.8 + 0.1)
                .collect();
            Tensor::from_vec(data, (1, 32, cfg.hidden_size), dev).expect("window")
        })
        .collect()
}

fn run_config(codebook: Codebook, start: usize, limit: usize) -> RunConfig {
    RunConfig {
        h_shrink: 1.0,
        gptq: GptqConfig {
            block: llvq_core::DIM,
            retract: true,
            group_scales: false,
            design_c: false,
            lambda: 1e-2,
            tail: TailPolicy::KeepExact,
        },
        damping: 1e-2,
        codebook,
        threads: 1,
        start,
        limit,
        rotation_seed: Some(ROT),
    }
}

/// A scratch directory, removed on drop. No `tempfile` dev-dependency, for the
/// reason `resume.rs` gives: this crate's dependency list is already the
/// workspace's exception.
struct Scratch(std::path::PathBuf);

impl Scratch {
    fn new(tag: &str) -> Self {
        let p = std::env::temp_dir().join(format!("llvq-trio-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).expect("scratch dir");
        Self(p)
    }
    fn at(&self, name: &str) -> String {
        self.0.join(name).to_string_lossy().into_owned()
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// A sink writing a real file at the kind the codebook names — exactly what
/// `bin/smoke`'s `FileSink` does, and the line under test in half of these.
struct FileSink(ArtifactWriter<std::io::BufWriter<std::fs::File>>);

impl FileSink {
    fn create(path: &str, n: u32, kind: CodeKind) -> Self {
        let f = std::fs::File::create(path).expect("create");
        Self(ArtifactWriter::with_kind(std::io::BufWriter::new(f), kind, n).expect("header"))
    }
    fn finish(self) {
        self.0.finish().expect("finish");
    }
}

impl MatrixSink for FileSink {
    fn push(&mut self, m: llvq_llm::artifact2::QuantizedMatrix) -> anyhow::Result<()> {
        Ok(self.0.push(&m)?)
    }
}

/// Quantize blocks `0..limit` of the tiny model into `path`, and hand back the
/// report and the model the run left behind.
fn quantize(
    map: &VarMap,
    dev: &Device,
    path: &str,
    codebook: Codebook,
    limit: usize,
) -> (Report, Qwen3) {
    let mut model = fresh(map, dev);
    let mut hidden = windows(dev);
    let blocks = limit.min(model.blocks.len());
    let mut sink = FileSink::create(path, (matrices_per_block() * blocks) as u32, codebook.code_kind());
    let report = quantize_model_capturing(
        &mut model,
        &mut hidden,
        &run_config(codebook, 0, limit),
        |_, _, _| {},
        Some(&mut sink),
    )
    .expect("quantize");
    sink.finish();
    (report, model)
}

/// `verify_artifact` in miniature: every record decoded through the map it
/// names, against the weights the model was left holding.
fn verify(path: &str, model: &Qwen3) -> usize {
    let f = std::fs::File::open(path).expect("open");
    let mut r = std::io::BufReader::new(f);
    let head = read_header(&mut r).expect("header");
    let cbs = Codebooks::new();
    let mut checked = 0usize;
    for _ in 0..head.matrices {
        let m = read_matrix_with(&mut r, head.version, &cbs).expect("record");
        let decoded = decode_matrix(&m);
        let parts: Vec<&str> = m.name.split('.').collect();
        let b: usize = parts[2].parse().expect("block index");
        let proj = parts[3..parts.len() - 1].join(".");
        let want = model.blocks[b]
            .linear(&proj)
            .weight()
            .flatten_all()
            .expect("flatten")
            .to_vec1::<f32>()
            .expect("readback");
        assert_eq!(decoded.len(), want.len(), "{}: length", m.name);
        for (k, (g, e)) in decoded.iter().zip(want.iter()).enumerate() {
            assert_eq!(
                g.to_bits(),
                e.to_bits(),
                "{} weight {k}: file {g:e}, model {e:e}",
                m.name
            );
        }
        checked += decoded.len();
    }
    checked
}

/// Every quantized weight of the model, in a fixed order.
fn projections(m: &Qwen3) -> Vec<f32> {
    let mut out = Vec::new();
    for b in &m.blocks {
        for act in Act::ALL {
            for name in act.consumers() {
                out.extend(
                    b.linear(name)
                        .weight()
                        .flatten_all()
                        .expect("flatten")
                        .to_vec1::<f32>()
                        .expect("readback"),
                );
            }
        }
    }
    out
}

// --------------------------------------------------------------------------
// The load-bearing test.
// --------------------------------------------------------------------------

/// **The whole of step 4, in one assertion.** A `Codebook::Trio` run writes a
/// v5 file declaring Trio and nothing else, and every record in it decodes to
/// the weights the run left in the model, bit for bit.
///
/// It is also the mutation net for most of the wiring. Mapping the `trio`
/// codebook to `LeechShapeGain` puts ball points under a Trio writer, which
/// `Trio::encode` refuses; making `Codebook::code_kind` answer `Ball` writes a
/// v4 Ball header over Trio words, which the ball indexer refuses; decoding
/// the word without its gain bit, or in trio order, moves these bits.
#[test]
fn a_trio_run_writes_a_v5_trio_file_that_decodes_bit_for_bit() {
    let dev = Device::Cpu;
    let s = Scratch::new("verify");
    let map = VarMap::new();
    let path = s.at("trio.llvq");
    let (report, model) = quantize(&map, &dev, &path, trio(), usize::MAX);

    // The header says what the file is, before a record is read. `Trio` alone:
    // a set that also carried Ball would let a Ball record through every
    // refusal written against `kinds()`.
    let f = std::fs::File::open(&path).expect("open");
    let head = read_header(&mut std::io::BufReader::new(f)).expect("header");
    assert_eq!(head.version, 5, "a Trio file needs the first kinded version");
    assert_eq!(head.default_kind(), CodeKind::Trio);
    assert_eq!(head.kinds(), KindSet::of(CodeKind::Trio));
    assert!(!head.is_ball_only(), "the runtime refusals read this");
    assert_eq!(head.matrices as usize, report.matrices);

    let checked = verify(&path, &model);
    assert_eq!(checked, report.weights as usize);
}

/// The two arms are compared **at a constant rate**, and that is a property of
/// the code and not of a coincidence: 47 bits of label plus one gain bit is
/// the 47 index bits of `Λ₂₄(12)` plus one gain bit.
///
/// This is the equality `bin/smoke`'s step (c) reads to the fourth decimal on
/// the 0.6B. Asserted here on the *reported* rate — the number a journal
/// carries — over the same model, so a `block_bits` or `block_len` that
/// drifted would be caught in the fast loop rather than by two runs of an
/// hour each.
#[test]
fn both_arms_report_the_same_rate_on_the_same_model() {
    assert_eq!(trio().block_bits(), 48.0);
    assert_eq!(trio().block_bits(), ball().block_bits());
    assert_eq!(trio().block_len(), ball().block_len());
    assert_eq!(trio().code_kind(), CodeKind::Trio);
    assert_eq!(ball().code_kind(), CodeKind::Ball);

    let dev = Device::Cpu;
    let s = Scratch::new("rate");
    let map = VarMap::new();
    let (rt, _) = quantize(&map, &dev, &s.at("t.llvq"), trio(), usize::MAX);
    let (rb, _) = quantize(&map, &dev, &s.at("b.llvq"), ball(), usize::MAX);
    assert_eq!(rt.weights, rb.weights);
    assert_eq!(rt.tail_weights, rb.tail_weights);
    assert_eq!(rt.rows, rb.rows);
    assert_eq!(
        rt.bits_per_weight(),
        rb.bits_per_weight(),
        "the two arms must spend the same bits, or the A/B moves two things"
    );
    // …and the files are *not* the same, or the equality above would be the
    // equality of one arm with itself.
    assert_ne!(
        std::fs::read(s.at("t.llvq")).expect("t"),
        std::fs::read(s.at("b.llvq")).expect("b")
    );
}

/// A Trio shard resumed on the Trio arm must produce the file a single run
/// produces, byte for byte.
///
/// This is the only place a v5 record is walked by `shard_extent` — which
/// parses record headers by hand and had to learn the kind field's four bytes
/// — and the only place `to_quantized` decodes a Trio record. A walk that
/// skipped those four bytes reads the kind tag as a centroid count and lands
/// mid-record; the byte comparison is what says so.
#[test]
fn two_trio_segments_produce_the_single_run_file() {
    let dev = Device::Cpu;
    let s = Scratch::new("resume");
    let map = VarMap::new();

    let one = s.at("one.llvq");
    let (_, whole) = quantize(&map, &dev, &one, trio(), usize::MAX);

    let a = s.at("a.llvq");
    quantize(&map, &dev, &a, trio(), 1);

    let b = s.at("b.llvq");
    let mut model = fresh(&map, &dev);
    let mut hidden = windows(&dev);
    let blocks = model.blocks.len();
    let mut sink = FileSink::create(&b, (matrices_per_block() * blocks) as u32, CodeKind::Trio);
    let expect = ShardExpect {
        kind: CodeKind::Trio,
        shell_cap: llvq_artifact::TRIO_SHELL_CAP,
        centroids: 2,
        rotation_seed: Some(ROT),
    };
    let scan = resume_from_shard(&mut model, &a, &mut sink.0, &expect, &dev).expect("resume");
    assert_eq!(scan.blocks, 1);
    quantize_model_capturing(
        &mut model,
        &mut hidden,
        &run_config(trio(), scan.blocks, usize::MAX),
        |_, _, _| {},
        Some(&mut sink),
    )
    .expect("segment B");
    sink.finish();

    assert_eq!(
        std::fs::read(&one).expect("one"),
        std::fs::read(&b).expect("b"),
        "two Trio segments must produce the single run's bytes"
    );
    assert_eq!(projections(&whole), projections(&model));
}

/// A shard of the **other map** is refused **by the resume**, in both
/// directions.
///
/// Both indices are 47 bits wide at `shell_cap = 12` — and `TRIO_SHELL_CAP`
/// *is* 12, so the shell check passes, the centroid count passes and the
/// rotation seed passes. Such a splice misreads not one bit: it decodes every
/// block of one half against the other half's codebook, and the file that
/// comes out opens, runs, and is a model nobody measured. Nothing but the
/// record's kind separates the two.
///
/// 🕳️ The refusal is asserted on the resume's **own** sentence, not merely on
/// "some error naming both kinds". Deleting the kind check leaves the shard
/// stopped one line later by `ArtifactWriter::push_raw` — whose
/// `KindNotDeclared` also names both maps — so an assertion on the names
/// alone passed the mutant. That is only luck: it holds because this sink
/// declares one kind, and a mixed-kind sink (Q5's, `v_proj` in int4 beside
/// Trio) declares both and would let the splice through.
#[test]
fn a_shard_of_the_other_map_is_refused_both_ways() {
    let dev = Device::Cpu;
    let s = Scratch::new("kinds");
    let map = VarMap::new();

    let ball_shard = s.at("ball.llvq");
    quantize(&map, &dev, &ball_shard, ball(), 1);
    let trio_shard = s.at("trio.llvq");
    quantize(&map, &dev, &trio_shard, trio(), 1);

    for (shard, want, other) in [
        (&ball_shard, CodeKind::Trio, CodeKind::Ball),
        (&trio_shard, CodeKind::Ball, CodeKind::Trio),
    ] {
        let mut model = fresh(&map, &dev);
        let out = s.at("out.llvq");
        let mut sink = FileSink::create(&out, matrices_per_block() as u32 * 2, want);
        let expect = ShardExpect {
            kind: want,
            shell_cap: if want == CodeKind::Trio {
                llvq_artifact::TRIO_SHELL_CAP
            } else {
                12
            },
            centroids: 2,
            rotation_seed: Some(ROT),
        };
        let e = resume_from_shard(&mut model, shard, &mut sink.0, &expect, &dev)
            .expect_err("a shard of the other map was accepted")
            .to_string();
        assert!(
            e.contains("Two maps in one file"),
            "the resume itself must refuse, before the writer does: {e}"
        );
        assert!(e.contains(want.name()), "the refusal must name this run: {e}");
        assert!(e.contains(other.name()), "…and the shard's map: {e}");
        assert!(
            e.contains("model.layers.0."),
            "…and the record it stopped at: {e}"
        );
    }
}
