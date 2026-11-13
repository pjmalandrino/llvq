//! # Gradient-weighted input Hessians for gate/up (`LLVQ_GUIDED_WEIGHTS`)
//!
//! V1 of `docs/mesures/recherche-qualite-r2-chiffrage-encodeur-2026-11-13.txt`: for `mlp.gate_proj`
//! and `mlp.up_proj` only, the one input Hessian `Σ_t x_t x_tᵀ` is replaced by `G` Hessians, one per
//! group of consecutive output rows, `H_g = Σ_t w_tg x_t x_tᵀ`, the weights read from a directory.
//!
//! The tiny model is `tetra_wiring.rs`'s shape (two blocks, every matrix one 24-wide block and a
//! tail, intermediate 64 so that 16 row groups divide it), on weights set deterministically so a
//! digest of the file can be stored. Four windows of 32 rows enter block 0.

use candle_core::{DType, Device, Tensor};
use candle_nn::{Activation, VarBuilder, VarMap};
use candle_transformers::models::qwen3::Config;
use llvq_artifact::CodeKind;
use llvq_llm::artifact2::ArtifactWriter;
use llvq_llm::calib::{matrices_per_block, quantize_model_capturing, Codebook, MatrixSink, RunConfig};
use llvq_llm::model::Qwen3;
use llvq_quant::gptq::{GptqConfig, TailPolicy};

const ROT: u64 = 0x11_0FEED;
const WINDOWS: usize = 4;
const ROWS: usize = 32;

fn tetra() -> Codebook {
    Codebook::Tetra {
        gain_bits: 1,
        post_shape_gain: false,
    }
}

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

/// A `VarMap` whose every variable holds a deterministic pattern, so that two processes (and the
/// commit before this file's code) encode the same model. Norm weights sit near 1.
fn seeded_map(dev: &Device) -> VarMap {
    let map = VarMap::new();
    {
        let vb = VarBuilder::from_varmap(&map, DType::F32, dev);
        let _ = Qwen3::new(&tiny(), vb, llvq_llm::kvq::KvMode::F16).expect("tiny model builds");
    }
    let data = map.data().lock().expect("varmap lock");
    let mut names: Vec<&String> = data.keys().collect();
    names.sort();
    for (k, name) in names.into_iter().enumerate() {
        let var = &data[name];
        let shape = var.as_tensor().dims().to_vec();
        let n: usize = shape.iter().product();
        let norm = name.contains("norm");
        let v: Vec<f32> = (0..n)
            .map(|i| {
                let s = (((i * 7 + 131 * k + 3) as f32) * 0.613).sin();
                if norm {
                    1.0 + 0.05 * s
                } else {
                    0.2 * s
                }
            })
            .collect();
        var.set(&Tensor::from_vec(v, shape, dev).expect("tensor")).expect("set");
    }
    drop(data);
    map
}

fn fresh(map: &VarMap, dev: &Device) -> Qwen3 {
    let vb = VarBuilder::from_varmap(map, DType::F32, dev);
    Qwen3::new(&tiny(), vb, llvq_llm::kvq::KvMode::F16).expect("tiny model builds")
}

/// The four windows, or the subset named.
fn windows(dev: &Device, which: &[usize]) -> Vec<Tensor> {
    let cfg = tiny();
    which
        .iter()
        .map(|&w| {
            let n = ROWS * cfg.hidden_size;
            let data: Vec<f32> = (0..n)
                .map(|i| (((i + 977 * w) as f32) * 0.37).sin() * 0.8 + 0.1)
                .collect();
            Tensor::from_vec(data, (1, ROWS, cfg.hidden_size), dev).expect("window")
        })
        .collect()
}

fn run_config(limit: usize) -> RunConfig {
    RunConfig {
        sequential_block: false,
        h_shrink: 1.0,
        gain_scale: 1.0,
        int4_types: Vec::new(),
        gptq: GptqConfig {
            block: llvq_core::DIM,
            retract: true,
            group_scales: false,
            design_c: false,
            spherical_feedback: false,
            lambda: 1e-2,
            tail: TailPolicy::KeepExact,
        },
        damping: 1e-2,
        codebook: tetra(),
        threads: 2,
        start: 0,
        limit,
        rotation_seed: Some(ROT),
    }
}

struct Scratch(std::path::PathBuf);

impl Scratch {
    fn new(tag: &str) -> Self {
        let p = std::env::temp_dir().join(format!("llvq-guided-{}-{tag}", std::process::id()));
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
    fn push_int4(&mut self, m: llvq_llm::artifact2::Int4Matrix) -> anyhow::Result<()> {
        Ok(self.0.push_int4(&m)?)
    }
}

/// The unguided encode of the model on the windows named, into `path`. Returns the model it left.
fn encode_unset(map: &VarMap, dev: &Device, path: &str, which: &[usize], limit: usize) -> Qwen3 {
    let mut model = fresh(map, dev);
    let mut hidden = windows(dev, which);
    let blocks = limit.min(model.blocks.len());
    let mut sink = FileSink::create(path, (matrices_per_block() * blocks) as u32, CodeKind::Tetra);
    quantize_model_capturing(&mut model, &mut hidden, &run_config(limit), |_, _, _| {}, Some(&mut sink))
        .expect("quantize");
    sink.finish();
    model
}

/// sha256 of the unguided encode of the whole tiny model (two blocks, four windows, Tetra,
/// rotation on), taken BEFORE `llvq-llm/src/guided.rs` and the changes to `calib.rs`, `smoke.rs`
/// and `served.rs` existed. How, exactly: on 2026-11-13, in the worktree at HEAD 8d6c664 with every
/// tracked file unmodified, this file existed as an untracked file holding only what precedes the
/// line `// Guided weights on the tiny model.` below (the harness and this one test, the constant
/// set to "TO BE MEASURED"), beside the untracked fixture `tests/fixtures/guided/`. Command, on
/// aarch64-apple-darwin: `~/.llvq-guard/heavy.sh "…" env CARGO_PROFILE_DEV_DEBUG=0
/// CARGO_INCREMENTAL=0 cargo test -p llvq-llm --test guided -- --nocapture`. The test failed on
/// the placeholder and printed `unset encode sha256 8b107e9f…`, the value below; both encodes of
/// the run agreed. That part of the file compiles against 8d6c664 because it uses only
/// `quantize_model_capturing` and types that existed there.
const DIGEST_AT_8D6C664: &str = "8b107e9f0c210f2e1b3eaa5fef683b6d55089810d7ea75f510489a4dc9c1c69f";

/// With the variable unset the encode writes the bytes it wrote before the change. The stored
/// digest is asserted on the platform it was taken on: candle's CPU GEMM picks its micro-kernel by
/// target, so another target may round differently and was never recorded. Everywhere, two encodes
/// in one process must agree byte for byte.
#[test]
fn unset_encode_is_byte_identical_to_the_commit_before_the_change() {
    let dev = Device::Cpu;
    let map = seeded_map(&dev);
    let s = Scratch::new("digest");
    let (a, b) = (s.at("a.llvq"), s.at("b.llvq"));
    encode_unset(&map, &dev, &a, &[0, 1, 2, 3], usize::MAX);
    encode_unset(&map, &dev, &b, &[0, 1, 2, 3], usize::MAX);
    let da = llvq_llm::digest::sha256_file(std::path::Path::new(&a)).expect("sha a");
    let db = llvq_llm::digest::sha256_file(std::path::Path::new(&b)).expect("sha b");
    eprintln!("unset encode sha256 {da}");
    assert_eq!(da, db, "two unguided encodes in one process differ");
    if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        assert_eq!(da, DIGEST_AT_8D6C664, "the unguided encode moved against the commit before the change");
    }
}

// --------------------------------------------------------------------------
// Guided weights on the tiny model.
// --------------------------------------------------------------------------

use llvq_llm::calib::quantize_model_guided;
use llvq_llm::guided::{file_name, read_raw, GuidedWeights, FORMAT, G_MAX, TYPES};

/// The tiny model's calibration rows: four windows of 32.
const TOKENS: usize = WINDOWS * ROWS;
/// A fingerprint for the tiny runs, which feed hidden states and not tokens.
const FP: u64 = 0x0123_4567_89ab_cdef;

/// Write a weights directory for the tiny model: `value(layer, type index, token, column)`.
fn write_weights(dir: &std::path::Path, value: impl Fn(usize, usize, usize, usize) -> f32) {
    std::fs::create_dir_all(dir).expect("weights dir");
    let layers = tiny().num_hidden_layers;
    let mut files = serde_json::Map::new();
    for l in 0..layers {
        for (k, t) in TYPES.iter().enumerate() {
            let mut bytes = Vec::with_capacity(TOKENS * G_MAX * 4);
            for tok in 0..TOKENS {
                for j in 0..G_MAX {
                    bytes.extend_from_slice(&value(l, k, tok, j).to_le_bytes());
                }
            }
            let n = file_name(l, t);
            std::fs::write(dir.join(&n), &bytes).expect("weight file");
            files.insert(n, llvq_llm::digest::sha256_hex(&bytes).into());
        }
    }
    let m = serde_json::json!({
        "format": FORMAT,
        "g_max": G_MAX,
        "tokens": TOKENS,
        "layers": layers,
        "token_fingerprint": format!("{FP:016x}"),
        "files": files,
    });
    std::fs::write(dir.join("manifest.json"), serde_json::to_string_pretty(&m).expect("json")).expect("manifest");
}

fn open(dir: &std::path::Path, groups: usize) -> GuidedWeights {
    GuidedWeights::open(dir, groups, FP, TOKENS).expect("open guided weights")
}

/// The guided encode on all four windows, into `path`.
fn encode_guided(map: &VarMap, dev: &Device, path: &str, g: &GuidedWeights, limit: usize) -> Qwen3 {
    let mut model = fresh(map, dev);
    let mut hidden = windows(dev, &[0, 1, 2, 3]);
    let blocks = limit.min(model.blocks.len());
    let mut sink = FileSink::create(path, (matrices_per_block() * blocks) as u32, CodeKind::Tetra);
    quantize_model_guided(&mut model, &mut hidden, &run_config(limit), Some(g), |_, _, _| {}, Some(&mut sink))
        .expect("guided quantize");
    sink.finish();
    model
}

fn bytes(path: &str) -> Vec<u8> {
    std::fs::read(path).expect("read artifact")
}

/// The f32 bits of one projection of one block, as the run left it.
fn bits(m: &Qwen3, block: usize, name: &str) -> Vec<u32> {
    m.blocks[block]
        .linear(name)
        .weight()
        .flatten_all()
        .expect("flatten")
        .to_vec1::<f32>()
        .expect("readback")
        .into_iter()
        .map(f32::to_bits)
        .collect()
}

/// Weights of exactly 1 per token give the published Hessian bit for bit: the weighted side is
/// `A ∘ 1 = A`, the mean is 1, so `(A ∘ w)ᵀ A` is `AᵀA` with the same operands. With one group the
/// file equals the unguided one byte for byte; with four groups of the same weights too, since
/// every group then holds the published factor and rows never interact.
#[test]
fn unit_weights_encode_the_bytes_of_the_unguided_run() {
    let dev = Device::Cpu;
    let map = seeded_map(&dev);
    let s = Scratch::new("unit");
    let wdir = s.0.join("w");
    write_weights(&wdir, |_, _, _, _| 1.0 / 16.0);
    let unset = s.at("unset.llvq");
    encode_unset(&map, &dev, &unset, &[0, 1, 2, 3], usize::MAX);
    for groups in [1, 4, 16] {
        let p = s.at(&format!("g{groups}.llvq"));
        encode_guided(&map, &dev, &p, &open(&wdir, groups), usize::MAX);
        assert!(bytes(&p) == bytes(&unset), "G = {groups} with unit weights differs from the unguided file");
    }
}

/// Weights that zero windows 2 and 3 encode block 0's gate/up exactly as an unguided run on windows
/// 0 and 1 alone. Block 0's MLP input is captured before anything in the block is quantized, so it
/// depends on each window's own rows only; the kept half's weights are 2 after normalisation, and
/// `2·AᵀA·(1/128)` is `AᵀA·(1/64)` bit for bit.
#[test]
fn zeroing_half_the_tokens_equals_encoding_the_other_half_alone() {
    let dev = Device::Cpu;
    let map = seeded_map(&dev);
    let s = Scratch::new("half");
    let wdir = s.0.join("w");
    write_weights(&wdir, |_, _, t, _| if t < 2 * ROWS { 1.0 / 16.0 } else { 0.0 });
    let g = encode_guided(&map, &dev, &s.at("g.llvq"), &open(&wdir, 1), 1);
    let half = encode_unset(&map, &dev, &s.at("half.llvq"), &[0, 1], 1);
    let all = encode_unset(&map, &dev, &s.at("all.llvq"), &[0, 1, 2, 3], 1);
    for name in TYPES {
        assert_eq!(bits(&g, 0, name), bits(&half, 0, name), "{name}: guided half against the half alone");
        assert_ne!(bits(&g, 0, name), bits(&all, 0, name), "{name}: the weights changed nothing");
    }
}

/// Two groups whose weights are identical encode what one group does, byte for byte, and both
/// differ from the unguided file (the weights are not uniform).
#[test]
fn two_identical_groups_equal_one_group() {
    let dev = Device::Cpu;
    let map = seeded_map(&dev);
    let s = Scratch::new("g2");
    let wdir = s.0.join("w");
    write_weights(&wdir, |l, k, t, _| 0.125 + ((t * 13 + 7 * l + 3 * k) % 11) as f32 * 0.0625);
    let (p1, p2, pu) = (s.at("g1.llvq"), s.at("g2.llvq"), s.at("u.llvq"));
    encode_guided(&map, &dev, &p1, &open(&wdir, 1), usize::MAX);
    encode_guided(&map, &dev, &p2, &open(&wdir, 2), usize::MAX);
    encode_unset(&map, &dev, &pu, &[0, 1, 2, 3], usize::MAX);
    assert!(bytes(&p1) == bytes(&p2), "G = 2 with identical groups differs from G = 1");
    assert!(bytes(&p1) != bytes(&pu), "non-uniform weights changed nothing");
}

/// Each row group is encoded against its own weights: with G = 2, group 0 weighting windows 0-1
/// and group 1 windows 2-3, block 0's rows [0, 32) are those of an unguided run on windows 0-1 and
/// rows [32, 64) those of a run on windows 2-3.
#[test]
fn each_row_group_is_encoded_against_its_own_weights() {
    let dev = Device::Cpu;
    let map = seeded_map(&dev);
    let s = Scratch::new("route");
    let wdir = s.0.join("w");
    write_weights(&wdir, |_, _, t, j| {
        let first_half = t < 2 * ROWS;
        if (j < G_MAX / 2) == first_half {
            1.0
        } else {
            0.0
        }
    });
    let g = encode_guided(&map, &dev, &s.at("g.llvq"), &open(&wdir, 2), 1);
    let a = encode_unset(&map, &dev, &s.at("a.llvq"), &[0, 1], 1);
    let b = encode_unset(&map, &dev, &s.at("b.llvq"), &[2, 3], 1);
    let d_in = tiny().hidden_size;
    let half = tiny().intermediate_size / 2 * d_in;
    for name in TYPES {
        let (gb, ab, bb) = (bits(&g, 0, name), bits(&a, 0, name), bits(&b, 0, name));
        assert_eq!(gb[..half], ab[..half], "{name}: rows of group 0 against windows 0-1");
        assert_eq!(gb[half..], bb[half..], "{name}: rows of group 1 against windows 2-3");
        assert_ne!(ab[..half], bb[..half], "{name}: the two halves of the corpus encode alike");
    }
}

fn run_config_from(start: usize, limit: usize) -> RunConfig {
    RunConfig {
        start,
        ..run_config(limit)
    }
}

/// Routing by layer and by type, over both blocks. Layer 0 weights gate on windows 0-1 and up on
/// windows 2-3; layer 1 weights both with ones, so layer 1 differs from layer 0 and gate from up.
///
/// The references, each exact:
/// - Block 0, gate: an unguided run on windows 0-1 alone, and up: one on windows 2-3 alone. Block 0's
///   MLP input is captured before anything in the block is quantized, so it depends on each
///   window's own rows only, and kept weights of 2 after normalisation give `2·AᵀA·(1/128)`, which
///   is `AᵀA·(1/64)` bit for bit (the half-zero test).
/// - Block 1, gate and up: a two-step run on a fresh model. Step one is this very guided encode
///   stopped after block 0 (`limit = 1`), so the reference's block 0 is the test's, byte for byte;
///   step two encodes block 1 UNGUIDED (`start = 1`, the resume path, which pushes the four windows
///   through the quantized block 0 exactly as the test's own pass 2 does). Block 1 therefore sees the
///   same inputs in both, and unit weights give the unguided Hessian bit for bit (the unit-weight
///   test), so block 1's gate and up must be equal.
/// A load of layer 0 at block 1 puts the window weights on block 1 and breaks the second; a
/// gate/up swap breaks the first; `t - 1` underflows at block 0.
#[test]
fn layers_and_types_route_to_their_own_weights() {
    let dev = Device::Cpu;
    let map = seeded_map(&dev);
    let s = Scratch::new("layers");
    let wdir = s.0.join("w");
    write_weights(&wdir, |l, k, t, _| {
        let first_half = t < 2 * ROWS;
        match (l, k) {
            (0, 0) => if first_half { 1.0 / 16.0 } else { 0.0 }, // layer 0 gate: windows 0-1
            (0, _) => if first_half { 0.0 } else { 1.0 / 16.0 }, // layer 0 up: windows 2-3
            _ => 1.0 / 16.0,                                       // layer 1: ones
        }
    });
    let g = open(&wdir, 2);
    let test = encode_guided(&map, &dev, &s.at("t.llvq"), &g, usize::MAX);

    // Block 0 against the two half runs.
    let a = encode_unset(&map, &dev, &s.at("a.llvq"), &[0, 1], 1);
    let b = encode_unset(&map, &dev, &s.at("b.llvq"), &[2, 3], 1);
    assert_eq!(bits(&test, 0, "mlp.gate_proj"), bits(&a, 0, "mlp.gate_proj"), "block 0 gate: windows 0-1");
    assert_eq!(bits(&test, 0, "mlp.up_proj"), bits(&b, 0, "mlp.up_proj"), "block 0 up: windows 2-3");
    assert_ne!(bits(&a, 0, "mlp.gate_proj"), bits(&b, 0, "mlp.gate_proj"), "the halves encode alike");

    // Block 1 against block 0 guided, then block 1 unguided on the same model.
    let mut reference = fresh(&map, &dev);
    let mut hidden = windows(&dev, &[0, 1, 2, 3]);
    quantize_model_guided(&mut reference, &mut hidden, &run_config(1), Some(&g), |_, _, _| {}, None)
        .expect("reference, block 0 guided");
    let mut hidden = windows(&dev, &[0, 1, 2, 3]);
    quantize_model_guided(&mut reference, &mut hidden, &run_config_from(1, 2), None, |_, _, _| {}, None)
        .expect("reference, block 1 unguided");
    for name in llvq_llm::model::Act::ALL.iter().flat_map(|a| a.consumers().iter()) {
        assert_eq!(bits(&test, 0, name), bits(&reference, 0, name), "{name}: block 0, test against reference");
    }
    for name in TYPES {
        assert_eq!(bits(&test, 1, name), bits(&reference, 1, name), "{name}: block 1 against its unit-weight reference");
    }
    // Non-vacuous: block 1 encoded with layer 0's window weights would differ.
    let wrong = s.0.join("wrong");
    write_weights(&wrong, |_, k, t, _| {
        let first_half = t < 2 * ROWS;
        if (k == 0) == first_half {
            1.0 / 16.0
        } else {
            0.0
        }
    });
    let w = encode_guided(&map, &dev, &s.at("w.llvq"), &open(&wrong, 2), usize::MAX);
    for name in TYPES {
        assert_ne!(bits(&w, 1, name), bits(&reference, 1, name), "{name}: layer 0's weights at block 1 change nothing");
    }
}

/// The shared round-trip vector: `ops/guided_weights.py --selftest` writes the same 3 × 16 floats
/// and compares its bytes to this fixture.
#[test]
fn the_fixture_reads_back_as_written() {
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/guided/roundtrip-3x16.f32");
    let v = read_raw(&p, 3).expect("fixture");
    let want: Vec<f32> = (0..3 * G_MAX).map(|i| i as f32 * 0.25).collect();
    assert_eq!(v, want);
}

#[test]
fn malformed_inputs_are_refused_by_name() {
    let dev = Device::Cpu;
    let s = Scratch::new("refuse");
    let wdir = s.0.join("w");
    write_weights(&wdir, |_, _, _, _| 1.0);

    // Another fingerprint, another token count.
    let e = GuidedWeights::open(&wdir, 4, FP ^ 1, TOKENS).expect_err("fingerprint").to_string();
    assert!(e.contains("fingerprint") && e.contains("other tokens"), "{e}");
    let e = GuidedWeights::open(&wdir, 4, FP, TOKENS + 1).expect_err("tokens").to_string();
    assert!(e.contains("tokens") && e.contains(&(TOKENS + 1).to_string()), "{e}");
    let e = GuidedWeights::open(&wdir, 3, FP, TOKENS).expect_err("groups").to_string();
    assert!(e.contains("LLVQ_GUIDED_GROUPS"), "{e}");

    // A manifest without one of its files.
    let mpath = wdir.join("manifest.json");
    let good = std::fs::read_to_string(&mpath).expect("manifest");
    let mut m: serde_json::Value = serde_json::from_str(&good).expect("json");
    m["files"].as_object_mut().expect("files").remove(&file_name(1, "mlp.up_proj"));
    std::fs::write(&mpath, m.to_string()).expect("write");
    let e = GuidedWeights::open(&wdir, 4, FP, TOKENS).expect_err("entry").to_string();
    assert!(e.contains("w-001-up_proj.f32"), "{e}");
    let mut m: serde_json::Value = serde_json::from_str(&good).expect("json");
    m["format"] = "something-else".into();
    std::fs::write(&mpath, m.to_string()).expect("write");
    let e = GuidedWeights::open(&wdir, 4, FP, TOKENS).expect_err("format").to_string();
    assert!(e.contains("format"), "{e}");
    std::fs::write(&mpath, &good).expect("restore");

    // A file changed after the manifest, a truncated file, a negative weight.
    let g = open(&wdir, 4);
    let f = wdir.join(file_name(0, "mlp.gate_proj"));
    let orig = std::fs::read(&f).expect("file");
    let mut changed = orig.clone();
    changed[0] ^= 1;
    std::fs::write(&f, &changed).expect("write");
    let e = g.load(0, "mlp.gate_proj").expect_err("sha").to_string();
    assert!(e.contains("sha256") && e.contains("w-000-gate_proj.f32"), "{e}");
    let e = GuidedWeights::open(&wdir, 4, FP, TOKENS).expect_err("sha at open").to_string();
    assert!(e.contains("sha256") && e.contains("w-000-gate_proj.f32"), "{e}");
    let short = &orig[..orig.len() - 4];
    let e = read_raw(&f.with_extension("short"), TOKENS).map(|_| ()).expect_err("absent").to_string();
    assert!(e.contains("w-000-gate_proj.short"), "{e}");
    std::fs::write(&f, short).expect("write");
    let e = read_raw(&f, TOKENS).expect_err("size").to_string();
    assert!(e.contains("bytes") && e.contains("w-000-gate_proj.f32"), "{e}");
    let mut neg = orig.clone();
    neg[4..8].copy_from_slice(&(-1.0f32).to_le_bytes());
    std::fs::write(&f, &neg).expect("write");
    let e = read_raw(&f, TOKENS).expect_err("negative").to_string();
    assert!(e.contains("sum of squares"), "{e}");
    std::fs::write(&f, &orig).expect("restore");

    // Weights for 128 rows against a run on 64; gate_proj stored int4 beside guided weights.
    let map = seeded_map(&dev);
    let mut model = fresh(&map, &dev);
    let mut hidden = windows(&dev, &[0, 1]);
    let e = quantize_model_guided(&mut model, &mut hidden, &run_config(1), Some(&g), |_, _, _| {}, None)
        .expect_err("rows")
        .to_string();
    assert!(e.contains("calibration rows"), "{e}");
    let mut model = fresh(&map, &dev);
    let mut hidden = windows(&dev, &[0, 1, 2, 3]);
    let mut run = run_config(1);
    run.int4_types = vec!["gate_proj".into()];
    let e = quantize_model_guided(&mut model, &mut hidden, &run, Some(&g), |_, _, _| {}, None)
        .expect_err("int4")
        .to_string();
    assert!(e.contains("gate_proj") && e.contains("LLVQ_INT4_TYPES"), "{e}");
}
