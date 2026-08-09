//! # Resuming a killed quantization run
//!
//! A 32B run is ~11.4 h of rented card and, until now, nothing could restart
//! one: an incident in the tenth hour lost the ten hours and the bill both.
//! The claim this file has to establish is not "the resume runs" but the much
//! stronger one the feature is worth nothing without:
//!
//! > **quantizing a model in two segments produces the same artifact, byte for
//! > byte, as quantizing it in one.**
//!
//! Anything weaker is a trap. A resume that reconstitutes *almost* the right
//! sequential state produces a file that opens, decodes, runs, and is a
//! slightly different model from the one anybody measured — with no error
//! anywhere and a perplexity nobody can explain six weeks later. That is the
//! failure mode this repository has already paid for three times (CLAUDE.md
//! §5), and it is the reason the assertions below are on `to_bits()` and on
//! `Vec<u8>` equality rather than on a tolerance.
//!
//! ## What makes the property reachable at all
//!
//! Calibration is sequential — block `t` is quantized against activations that
//! have flowed through blocks `0..t` **in their quantized form** — so the
//! state to reconstitute looks like the whole loop. It is not, and the reason
//! is the property `verify_artifact` asserts on every run that writes a file:
//! `decode_matrix` returns the evaluated weights bit for bit. The shard *is*
//! the state; the hidden states are recomputed from it.
//!
//! `single_run_and_two_segments_agree` is the load-bearing test. The rest
//! exist because it is not enough: it says a *correct* resume reproduces the
//! run, and says nothing about what an *incorrect* one does. Each refusal test
//! below corresponds to one way two halves of a model can end up quantized
//! under different configurations, all of which produce a valid file.

use candle_core::{DType, Device, Tensor};
use candle_nn::{Activation, VarBuilder, VarMap};
use candle_transformers::models::qwen3::Config;
use llvq_llm::artifact2::{
    resume_from_shard, shard_extent, ArtifactWriter, RunState, ShardExpect,
};
use llvq_llm::calib::{
    block_matrix_plan, matrices_per_block, quantize_model_capturing, Codebook, MatrixSink,
    RunConfig,
};
use llvq_llm::model::{Act, Qwen3};
use llvq_quant::gptq::{GptqConfig, TailPolicy};

/// The rotation is **on** in every artifact test here. It is the one input
/// whose per-matrix value is *derived* rather than stored twice
/// (`effective_rotation_seed`), so a resume that rebuilt it from the run's
/// base seed instead of the matrix's would decode to plausible garbage. With
/// the rotation off, that whole class of defect is invisible.
const ROT: u64 = 0x11_0FEED;

/// Two transformer blocks, and every matrix carrying a tail: `hidden_size` 32
/// is one 24-wide block plus 8, `intermediate_size` 64 is two plus 16. The
/// real shape, in miniature — and the tail matters here, because it is the
/// part of a matrix that is *narrowed to f32 in the rotated basis* before the
/// un-rotation, which is what makes the file and the model the same object.
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

/// A model on the *same* weights every time — `quantize_model` rebuilds each
/// `Linear` rather than writing through the `Var`, so the `VarMap` survives a
/// run untouched and this hands back the pristine checkpoint.
fn fresh(map: &VarMap, dev: &Device) -> Qwen3 {
    let vb = VarBuilder::from_varmap(map, DType::F32, dev);
    Qwen3::new(&tiny(), vb).expect("tiny model builds")
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

/// Every quantized weight of the model, in a fixed order — so "the two runs
/// produced the same model" can be asserted on the *model*, not only on the
/// file. The two are different claims: the file could match while the loop
/// left the model somewhere else.
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

fn codebook() -> Codebook {
    Codebook::ShapeGain {
        gain_bits: 1,
        max_shell: 12,
        free_magnitude: false,
        level_cap: 5,
    }
}

fn run_config(start: usize, limit: usize, rotation_seed: Option<u64>) -> RunConfig {
    RunConfig {
        gptq: GptqConfig {
            block: llvq_core::DIM,
            retract: true,
            group_scales: false,
            design_c: false,
            lambda: 1e-2,
            tail: TailPolicy::KeepExact,
        },
        damping: 1e-2,
        codebook: codebook(),
        threads: 1,
        start,
        limit,
        rotation_seed,
    }
}

fn expect(shell_cap: u32, rotation_seed: Option<u64>) -> ShardExpect {
    ShardExpect {
        shell_cap,
        centroids: 2,
        rotation_seed,
    }
}

// --------------------------------------------------------------------------
// A scratch directory. No `tempfile` dev-dependency: this crate's dependency
// list is already the exception in a workspace that claims auditability, and
// two `create_dir_all` calls are not worth widening it.
// --------------------------------------------------------------------------

struct Scratch(std::path::PathBuf);

impl Scratch {
    fn new(tag: &str) -> Self {
        let p = std::env::temp_dir().join(format!("llvq-resume-{}-{tag}", std::process::id()));
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

/// A sink writing to a real file, exactly as `bin/smoke` does.
struct FileSink(ArtifactWriter<std::io::BufWriter<std::fs::File>>);

impl FileSink {
    fn create(path: &str, n: u32) -> Self {
        let f = std::fs::File::create(path).expect("create");
        Self(ArtifactWriter::new(std::io::BufWriter::new(f), n).expect("header"))
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

/// Quantize the whole model in one go, to `path`.
fn single(map: &VarMap, dev: &Device, path: &str) -> Vec<f32> {
    let mut model = fresh(map, dev);
    let mut hidden = windows(dev);
    let blocks = model.blocks.len();
    let mut sink = FileSink::create(path, (matrices_per_block() * blocks) as u32);
    quantize_model_capturing(
        &mut model,
        &mut hidden,
        &run_config(0, usize::MAX, Some(ROT)),
        |_, _, _| {},
        Some(&mut sink),
    )
    .expect("single run");
    sink.finish();
    projections(&model)
}

/// Quantize blocks `0..cut` to `path` — one segment of a split run.
fn segment_a(map: &VarMap, dev: &Device, path: &str, cut: usize, rot: Option<u64>, shell: u32) {
    segment_a_with(map, dev, path, cut, rot, shell, 1)
}

/// …at a chosen gain width too, which the shell cap alone cannot vary.
fn segment_a_with(
    map: &VarMap,
    dev: &Device,
    path: &str,
    cut: usize,
    rot: Option<u64>,
    shell: u32,
    gain: u32,
) {
    let mut model = fresh(map, dev);
    let mut hidden = windows(dev);
    let mut sink = FileSink::create(path, (matrices_per_block() * cut) as u32);
    let mut cfg = run_config(0, cut, rot);
    if let Codebook::ShapeGain {
        max_shell,
        gain_bits,
        ..
    } = &mut cfg.codebook
    {
        *max_shell = shell;
        *gain_bits = gain;
    }
    quantize_model_capturing(&mut model, &mut hidden, &cfg, |_, _, _| {}, Some(&mut sink))
        .expect("segment A");
    sink.finish();
}

/// Resume from `shard` and finish the model into `path`.
fn segment_b(
    map: &VarMap,
    dev: &Device,
    shard: &str,
    path: &str,
    exp: &ShardExpect,
) -> anyhow::Result<Vec<f32>> {
    let mut model = fresh(map, dev);
    let mut hidden = windows(dev);
    let blocks = model.blocks.len();
    let mut sink = FileSink::create(path, (matrices_per_block() * blocks) as u32);
    let scan = resume_from_shard(&mut model, shard, &mut sink.0, exp, dev)?;
    quantize_model_capturing(
        &mut model,
        &mut hidden,
        &run_config(scan.blocks, usize::MAX, Some(ROT)),
        |_, _, _| {},
        Some(&mut sink),
    )?;
    sink.finish();
    Ok(projections(&model))
}

// --------------------------------------------------------------------------
// The load-bearing test.
// --------------------------------------------------------------------------

/// **The whole item, in one assertion.** Two segments must produce the file a
/// single run produces, byte for byte — and leave the model in the same state.
///
/// Byte equality is the right bar and not an excessive one: every input is
/// deterministic (the checkpoint, the calibration windows, the rotation seeds,
/// the gain fit, the parallel loop — `parallel_matches_serial_exactly` pins
/// that one bit-exact), so the *only* thing that can move is the sequential
/// state the resume reconstitutes. A tolerance here would accept precisely the
/// defect the test exists to catch.
///
/// It is also the mutation net for most of this path. Making the loop ignore
/// `start`, making it skip the forward passes over the resumed blocks, making
/// `shard_extent` return a block count off by one, breaking the record order —
/// each of those changes these bytes.
#[test]
fn single_run_and_two_segments_agree() {
    let dev = Device::Cpu;
    let s = Scratch::new("agree");
    let map = VarMap::new();

    let one = s.at("one.llvq");
    let whole = single(&map, &dev, &one);

    let a = s.at("a.llvq");
    let b = s.at("b.llvq");
    segment_a(&map, &dev, &a, 1, Some(ROT), 12);
    let split = segment_b(&map, &dev, &a, &b, &expect(12, Some(ROT))).expect("resume");

    let want = std::fs::read(&one).expect("read one");
    let got = std::fs::read(&b).expect("read b");
    assert_eq!(
        want.len(),
        got.len(),
        "the segmented artifact is {} bytes against {} for the single run",
        got.len(),
        want.len()
    );
    if let Some(k) = want.iter().zip(&got).position(|(x, y)| x != y) {
        panic!(
            "the segmented artifact diverges from the single run at byte {k} \
             ({:#04x} vs {:#04x}) — the resume did not reconstitute the \
             sequential state",
            want[k], got[k]
        );
    }

    // …and the model the loop left behind, which is a separate claim: the file
    // could match while the second segment ended up holding something else.
    assert_eq!(whole.len(), split.len());
    for (k, (x, y)) in whole.iter().zip(&split).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "weight {k}: single run {x:e}, two segments {y:e}"
        );
    }
}

/// The shard has to be *read*, not merely counted. A resume that copied the
/// records but left the model's first blocks unquantized would still produce
/// the right byte count and the right names.
///
/// So: the model handed back by the resume must hold, for the blocks the shard
/// covers, exactly what the shard decodes to — and that must differ from the
/// pristine checkpoint, or the assertion is vacuous.
#[test]
fn the_resume_actually_loads_the_shard_into_the_model() {
    let dev = Device::Cpu;
    let s = Scratch::new("loads");
    let map = VarMap::new();
    let a = s.at("a.llvq");
    segment_a(&map, &dev, &a, 1, Some(ROT), 12);

    let pristine = projections(&fresh(&map, &dev));
    let mut model = fresh(&map, &dev);
    let out = s.at("out.llvq");
    let mut sink = FileSink::create(&out, (matrices_per_block() * 2) as u32);
    let scan = resume_from_shard(&mut model, &a, &mut sink.0, &expect(12, Some(ROT)), &dev)
        .expect("resume");
    assert_eq!(scan.blocks, 1);
    let loaded = projections(&model);

    // Block 0's weights moved (it was quantized), block 1's did not (it was
    // not in the shard). Both halves are load-bearing: the first says the
    // shard was read, the second says the resume did not touch what it must
    // not.
    let per_block = loaded.len() / 2;
    assert!(
        pristine[..per_block]
            .iter()
            .zip(&loaded[..per_block])
            .any(|(p, l)| p.to_bits() != l.to_bits()),
        "block 0 is bit-identical to the checkpoint — the shard was counted, \
         not loaded"
    );
    for (k, (p, l)) in pristine[per_block..]
        .iter()
        .zip(&loaded[per_block..])
        .enumerate()
    {
        assert_eq!(
            p.to_bits(),
            l.to_bits(),
            "weight {k} of block 1 moved, and nothing in the shard covers it"
        );
    }
}

// --------------------------------------------------------------------------
// Refusals. Each one is a way to build a valid file out of two different
// quantizations — which is the only failure mode this path can have.
// --------------------------------------------------------------------------

fn resume_error(shard: &str, out: &str, exp: &ShardExpect, dev: &Device, map: &VarMap) -> String {
    let mut model = fresh(map, dev);
    let mut sink = FileSink::create(out, (matrices_per_block() * 2) as u32);
    match resume_from_shard(&mut model, shard, &mut sink.0, exp, dev) {
        Ok(scan) => panic!(
            "the resume accepted an incoherent shard ({} blocks)",
            scan.blocks
        ),
        Err(e) => format!("{e}"),
    }
}

/// A shard quantized over Λ₂₄(13) resumed under Λ₂₄(12) — 48 index bits
/// behind 47, two rates in one file. The record carries its shell cap, so this
/// is checkable and must be checked.
#[test]
fn a_shard_from_another_codebook_is_refused() {
    let dev = Device::Cpu;
    let s = Scratch::new("shell");
    let map = VarMap::new();
    let a = s.at("a.llvq");
    segment_a(&map, &dev, &a, 1, Some(ROT), 13);
    let e = resume_error(&a, &s.at("o.llvq"), &expect(12, Some(ROT)), &dev, &map);
    assert!(e.contains("coquille"), "{e}");
}

/// A shard with **two** gain bits resumed under one — four magnitude levels
/// behind two, i.e. two rates in one file.
///
/// This test exists because the guard survived mutation without it: every
/// other case here varies the shell cap or the rotation, so deleting the
/// centroid check changed nothing any of them could see. It is not a
/// theoretical hole — `leech1c12` and `leech2c12` are both configurations this
/// project has run, they differ *only* in this field, and unlike the shell cap
/// the gain width leaves the index width untouched, so the resulting file is
/// perfectly well formed. Same lesson as CLAUDE.md §5: an assertion that never
/// exercises the parameter it covers tests nothing.
#[test]
fn a_shard_with_another_gain_width_is_refused() {
    let dev = Device::Cpu;
    let s = Scratch::new("gain");
    let map = VarMap::new();
    let a = s.at("a.llvq");
    segment_a_with(&map, &dev, &a, 1, Some(ROT), 12, 2);
    let e = resume_error(&a, &s.at("o.llvq"), &expect(12, Some(ROT)), &dev, &map);
    assert!(e.contains("niveaux de gain"), "{e}");
    // …and the same shard under its own gain width is accepted, or the guard
    // is a wall rather than a check.
    let mut model = fresh(&map, &dev);
    let o = s.at("ok.llvq");
    let mut sink = FileSink::create(&o, (matrices_per_block() * 2) as u32);
    let exp = ShardExpect {
        shell_cap: 12,
        centroids: 4,
        rotation_seed: Some(ROT),
    };
    resume_from_shard(&mut model, &a, &mut sink.0, &exp, &dev).expect("its own width");
}

/// A shard quantized in the natural basis resumed under the incoherence
/// rotation. Both halves decode; the model is half in one basis and half in
/// another, and nothing downstream notices.
#[test]
fn a_shard_from_another_rotation_is_refused() {
    let dev = Device::Cpu;
    let s = Scratch::new("rot");
    let map = VarMap::new();
    let a = s.at("a.llvq");
    segment_a(&map, &dev, &a, 1, None, 12);
    let e = resume_error(&a, &s.at("o.llvq"), &expect(12, Some(ROT)), &dev, &map);
    assert!(e.contains("rotation"), "{e}");
}

/// …and the symmetric case, which is the one a *derived* seed can get wrong
/// without a stored one ever disagreeing: same base seed, but the shard's
/// per-matrix values built for the wrong block or activation.
///
/// Written as "resume a rotated shard with the rotation off", because the
/// derivation is the thing under test and there is no other way to make the
/// expected value wrong without reimplementing it here.
#[test]
fn a_rotated_shard_is_refused_by_an_unrotated_run() {
    let dev = Device::Cpu;
    let s = Scratch::new("norot");
    let map = VarMap::new();
    let a = s.at("a.llvq");
    segment_a(&map, &dev, &a, 1, Some(ROT), 12);
    let e = resume_error(&a, &s.at("o.llvq"), &expect(12, None), &dev, &map);
    assert!(e.contains("rotation"), "{e}");
}

/// The records have to arrive in the order the loop writes them. A file whose
/// matrices are permuted is not the prefix of any run, and appending behind it
/// hands each decoded matrix to the wrong projection.
#[test]
fn a_shard_whose_records_are_out_of_order_is_refused() {
    let dev = Device::Cpu;
    let s = Scratch::new("order");
    let map = VarMap::new();
    let a = s.at("a.llvq");
    segment_a(&map, &dev, &a, 1, Some(ROT), 12);

    // Rewrite the file with two records swapped, through the raw passthrough
    // so that every *other* byte stays exactly what it was.
    let swapped = s.at("swapped.llvq");
    {
        let mut r = std::io::BufReader::new(std::fs::File::open(&a).expect("open"));
        let head = llvq_llm::artifact2::read_header(&mut r).expect("header");
        let mut ms: Vec<_> = (0..head.matrices)
            .map(|_| llvq_llm::artifact2::read_matrix_raw(&mut r).expect("record"))
            .collect();
        ms.swap(0, 1);
        let f = std::fs::File::create(&swapped).expect("create");
        let mut w = ArtifactWriter::new(std::io::BufWriter::new(f), head.matrices).expect("hdr");
        for m in &ms {
            w.push_raw(m).expect("push");
        }
        w.finish().expect("finish");
    }
    let e = resume_error(&swapped, &s.at("o.llvq"), &expect(12, Some(ROT)), &dev, &map);
    assert!(e.contains("attendue"), "{e}");
}

/// A shard is a *prefix*, and `shard_extent` has to say how long a prefix.
///
/// The header cannot: `ArtifactWriter::new` writes the matrix count before a
/// single record exists, so a run killed at hour ten leaves a header claiming
/// 252 matrices over a file holding 160 and a half. Trusting it would make the
/// resume read past the end of the data.
#[test]
fn a_torn_shard_is_truncated_to_whole_blocks() {
    let dev = Device::Cpu;
    let s = Scratch::new("torn");
    let map = VarMap::new();
    let full = s.at("full.llvq");
    segment_a(&map, &dev, &full, 2, Some(ROT), 12);
    let (m, b) = shard_extent(&full).expect("intact");
    assert_eq!((m, b), (matrices_per_block() * 2, 2));

    let bytes = std::fs::read(&full).expect("read");
    // Cut in the middle. The exact byte does not matter — what matters is that
    // whatever complete records survive, the answer is a whole number of
    // blocks and never more than the file holds.
    for cut in [bytes.len() * 3 / 4, bytes.len() / 2, bytes.len() / 3] {
        let torn = s.at("torn.llvq");
        std::fs::write(&torn, &bytes[..cut]).expect("write");
        match shard_extent(&torn) {
            Ok((m, b)) => {
                assert_eq!(m, b * matrices_per_block(), "{m} records is not {b} blocks");
                assert!(b <= 2, "a truncated file cannot cover more blocks than the whole one");
                // …and what it claims is readable: resuming from it must not
                // error out halfway through the copy.
                let mut model = fresh(&map, &dev);
                let o = s.at("o.llvq");
                let mut sink = FileSink::create(&o, (matrices_per_block() * 2) as u32);
                let scan =
                    resume_from_shard(&mut model, &torn, &mut sink.0, &expect(12, Some(ROT)), &dev)
                        .unwrap_or_else(|e| panic!("cut {cut}: {e}"));
                assert_eq!(scan.blocks, b);
            }
            // Fewer than one whole block survived. That is a refusal, not a
            // silent zero — resuming "from block 0" would quietly re-run the
            // whole model and overwrite a shard the operator wanted salvaged.
            Err(e) => assert!(format!("{e}").contains("rien à reprendre"), "{e}"),
        }
    }
}

/// A shard whose header is a byte off — a different build's codebook, a
/// truncated magic — must not be salvaged into something.
#[test]
fn a_file_that_is_not_an_artifact_is_refused() {
    let s = Scratch::new("notartifact");
    let p = s.at("junk.llvq");
    std::fs::write(&p, b"not an artifact at all, just some bytes").expect("write");
    assert!(shard_extent(&p).is_err());
}

// --------------------------------------------------------------------------
// The sidecar: everything a shard cannot record about the run that wrote it.
// --------------------------------------------------------------------------

fn state(calib: &str, damping: &str) -> RunState {
    let mut s = RunState::new();
    s.set("model", "Qwen/Qwen3-0.6B")
        .set("calibration", calib)
        .set("damping", damping)
        .set(RunState::BLOCKS, 14);
    s
}

/// The fields a shard cannot carry are exactly the ones that silently produce
/// a hybrid artifact, so a difference in any of them has to be a refusal.
#[test]
fn the_sidecar_catches_what_the_shard_cannot() {
    let a = state("c4", "1e-2");
    assert!(a.diff(&state("c4", "1e-2")).is_empty());

    let corpus = a.diff(&state("wikitext2", "1e-2"));
    assert_eq!(corpus.len(), 1, "{corpus:?}");
    assert!(corpus[0].contains("calibration"), "{corpus:?}");

    let damp = a.diff(&state("c4", "3e-2"));
    assert_eq!(damp.len(), 1, "{damp:?}");
    assert!(damp[0].contains("damping"), "{damp:?}");

    // A field one side stopped recording is a difference too: a resume waved
    // through on the strength of the fields that survived is the same defect
    // as one waved through on none.
    let mut fewer = RunState::new();
    fewer.set("model", "Qwen/Qwen3-0.6B").set(RunState::BLOCKS, 14);
    assert_eq!(a.diff(&fewer).len(), 2, "{:?}", a.diff(&fewer));
    assert_eq!(fewer.diff(&a).len(), 2, "{:?}", fewer.diff(&a));
}

/// `blocks_done` is the one key `diff` must ignore: it states how far the
/// *shard* goes, which is precisely what a resume expects to differ. Diffing
/// it would refuse every resume that has anything to do.
#[test]
fn the_block_count_is_not_a_configuration_difference() {
    let mut a = state("c4", "1e-2");
    a.set(RunState::BLOCKS, 14);
    let mut b = state("c4", "1e-2");
    b.set(RunState::BLOCKS, 36);
    assert!(
        a.diff(&b).is_empty(),
        "the shard's extent was diffed as a configuration field: {:?}",
        a.diff(&b)
    );
}

/// The sidecar has to survive a round trip through the file, or the refusal it
/// exists for fires on formatting rather than on configuration.
#[test]
fn the_sidecar_round_trips_through_a_file() {
    let s = Scratch::new("sidecar");
    let p = s.at("art.llvq");
    let a = state("c4", "1e-2");
    let path = a.write(&p).expect("write");
    assert_eq!(path, RunState::path_for(&p));
    let back = RunState::read(&p).expect("read");
    assert!(a.diff(&back).is_empty(), "{:?}", a.diff(&back));
    assert_eq!(back.get(RunState::BLOCKS), Some("14"));
    // Comments and blank lines are decoration, not data.
    assert!(a.render().starts_with('#'));
}

/// A resume with no sidecar at all is refused rather than assumed compatible.
#[test]
fn a_missing_sidecar_is_refused() {
    let s = Scratch::new("nosidecar");
    let e = format!("{}", RunState::read(&s.at("absent.llvq")).unwrap_err());
    assert!(e.contains("reprise"), "{e}");
}

// --------------------------------------------------------------------------
// The emission order, which both the loop and the resume read.
// --------------------------------------------------------------------------

/// The plan has to be the order the loop actually emits, and the count has to
/// be the divisor `shard_extent` uses. A drift between them puts a resume at
/// the wrong block — the one failure this path must make impossible.
#[test]
fn the_matrix_plan_is_the_loops_emission_order() {
    let plan = block_matrix_plan();
    let mut emitted = Vec::new();
    for act in Act::ALL {
        for name in act.consumers() {
            emitted.push((act, *name));
        }
    }
    assert_eq!(plan, emitted);
    assert_eq!(matrices_per_block(), plan.len());
    // Qwen3: q, k, v, o, gate, up, down. Written out because a plan that lost
    // a projection would still agree with itself.
    assert_eq!(matrices_per_block(), 7);
    assert!(plan.iter().any(|(_, n)| *n == "mlp.down_proj"));
    assert!(plan.iter().any(|(_, n)| *n == "self_attn.q_proj"));
}
