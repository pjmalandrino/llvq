//! Stage 0 of `docs/plan-transformers.md`, on a file small enough to read.
//!
//! The real gate is `ops/llvq_hf_check.py` over the served 4B, which is 1.4 GB
//! and 119 M blocks. These tests hold the same claims on two records: that the
//! tensor names and shapes are what `quantization_config` says, that the code
//! bytes unpack MSB-first to the indices the artifact holds, and that a Ball
//! file is refused at the header rather than half written.
//!
//! They live here and not in the binary because a binary's `main` is not
//! callable from a test, which is how an int4 export arm went unexercised on
//! 2026-09-19.

use candle_core::Device;
use std::path::PathBuf;
use llvq_artifact::{
    ArtifactWriter, Blob, CodeKind, Int4Matrix, KindSet, QuantData, QuantizedMatrix, RawData,
    RawTensor, INT4G128_BITS, INT4G128_GROUP, TETRA_SHELL_CAP, VERSION,
};
use llvq_core::{SplitMix64, DIM};
use llvq_llm::hfpack;
use llvq_quant::quantizer::BlockCode;
use llvq_search::tetra::{Tetra, LABEL_MASK};

const ROT_SEED: u64 = 0x5EED;

/// A Tetra matrix whose points come from the map, the only source of valid ones.
fn tetra_matrix(name: &str, d_out: usize, d_in: usize, rng: &mut SplitMix64) -> QuantizedMatrix {
    let tetra = Tetra::new();
    QuantizedMatrix {
        name: name.to_string(),
        d_out,
        d_in,
        codes: (0..d_out * (d_in / DIM))
            .map(|_| BlockCode {
                point: tetra.decode(rng.next() & LABEL_MASK),
                gain: (rng.next() & 1) as u32,
            })
            .collect(),
        row_scales: (0..d_out).map(|_| 1e-3 + rng.next_f64()).collect(),
        centroids: vec![0.7, 1.1],
        rotation_seed: Some(ROT_SEED),
        shell_cap: TETRA_SHELL_CAP,
        // Exactly f32-representable: the stream stores the tail in f32, and a
        // value that is not would come back changed for a reason that has
        // nothing to do with this packer.
        tail: (0..d_out * (d_in % DIM))
            .map(|_| rng.next_gaussian() as f32 as f64)
            .collect(),
    }
}

fn int4_matrix(name: &str, d_out: usize, d_in: usize, rng: &mut SplitMix64) -> Int4Matrix {
    let gpr = d_in / INT4G128_GROUP;
    Int4Matrix {
        name: name.to_string(),
        d_out,
        d_in,
        bits: INT4G128_BITS,
        group: INT4G128_GROUP,
        packed: (0..d_out * d_in / 2).map(|_| rng.next() as u8).collect(),
        scales: (0..d_out * gpr)
            .map(|_| half::f16::from_f64(1e-3 + rng.next_f64()).to_bits())
            .collect(),
        biases: (0..d_out * gpr)
            .map(|_| half::f16::from_f64(rng.next_gaussian()).to_bits())
            .collect(),
    }
}

/// `config.json` and `tokenizer.json`, the two blobs a sealed file carries.
fn blobs() -> Vec<Blob> {
    vec![
        Blob {
            name: "config.json".into(),
            bytes: br#"{"hidden_size": 96, "num_hidden_layers": 1, "rms_norm_eps": 1e-06,
                        "tie_word_embeddings": true, "model_type": "qwen3"}"#
                .to_vec(),
        },
        Blob {
            name: "tokenizer.json".into(),
            bytes: br#"{"version": "1.0", "model": {"type": "BPE"}}"#.to_vec(),
        },
    ]
}

/// A sealed file with one record of each kind, one f16 raw tensor and one
/// group-affine quantized one. Returns its path inside `dir`.
fn sealed_file(dir: &std::path::Path) -> std::path::PathBuf {
    let mut rng = SplitMix64::new(0x4F_0D);
    let path = dir.join("tiny.llvq");
    let f = std::fs::File::create(&path).expect("create");
    let mut w = ArtifactWriter::with_kinds(
        std::io::BufWriter::new(f),
        VERSION,
        2,
        CodeKind::Tetra,
        KindSet::of(CodeKind::Tetra).with(CodeKind::Int4G128),
    )
    .expect("header");
    let lattice = tetra_matrix("model.layers.0.self_attn.q_proj.weight", 4, 3 * DIM + 16, &mut rng);
    w.push_kind(&lattice, CodeKind::Tetra).expect("push tetra");
    let int4 = int4_matrix("model.layers.0.self_attn.v_proj.weight", 3, 256, &mut rng);
    w.push_int4(&int4).expect("push int4");

    let embed = RawTensor {
        name: "model.embed_tokens.weight".into(),
        dims: vec![4, 128],
        data: RawData::Quant(QuantData {
            bits: 4,
            group: 64,
            packed: (0..4 * 128 / 2).map(|_| rng.next() as u8).collect(),
            scales: (0..4 * 2)
                .map(|_| half::f16::from_f64(1e-3 + rng.next_f64()).to_bits())
                .collect(),
            biases: (0..4 * 2)
                .map(|_| half::f16::from_f64(rng.next_gaussian()).to_bits())
                .collect(),
        }),
    };
    let norm = RawTensor {
        name: "model.layers.0.input_layernorm.weight".into(),
        dims: vec![8],
        data: RawData::F16((0..8).map(|i| half::f16::from_f64(1.0 + i as f64).to_bits()).collect()),
    };
    w.seal(&[embed, norm], &blobs()).expect("seal");
    path
}

#[test]
fn a_sealed_file_packs_into_a_described_directory() {
    let dir = tempdir("hfpack-ok");
    let src = sealed_file(&dir);
    let out = dir.join("hf");
    let s = hfpack::pack(&src, &out).expect("pack");

    assert_eq!((s.records, s.lattice, s.int4), (2, 1, 1));
    assert_eq!(s.rotations, 1, "one rotation, shared by nothing else here");
    assert_eq!(s.raw_tensors, 2);
    assert_eq!(s.blobs, 2);
    // 4 tetra fields with a tail, 3 int4, 2 rotation tables, 1 f16 raw, 3 quant raw.
    assert_eq!(s.tensors, 4 + 3 + 2 + 1 + 3);
    assert_eq!(s.quantized_weights, 4 * (3 * DIM + 16) + 3 * 256);
    assert_eq!(s.carried_weights, 4 * 128 + 8);

    let written: serde_json::Value =
        serde_json::from_slice(&std::fs::read(out.join("config.json")).expect("config")).unwrap();
    let qc = &written["quantization_config"];
    assert_eq!(qc["quant_method"], "llvq");
    assert_eq!(qc["code_order"], "msb_first_dense");
    assert_eq!(qc["block_dim"], DIM);
    // The base config survives, key for key.
    assert_eq!(written["hidden_size"], 96);
    assert_eq!(written["model_type"], "qwen3");
    assert_eq!(written["tie_word_embeddings"], true);

    let q = &qc["records"]["model.layers.0.self_attn.q_proj.weight"];
    assert_eq!(q["kind"], "tetra");
    assert_eq!(q["prefix"], "model.layers.0.self_attn.q_proj");
    assert_eq!((q["index_bits"].as_u64(), q["gain_bits"].as_u64()), (Some(47), Some(1)));
    assert_eq!(q["tail_cols"], 16);
    assert_eq!(q["nblocks"], 3);
    assert_eq!(q["code_bytes"], 4 * 3 * 6, "48 bits a block, byte aligned");
    assert_eq!(q["rotation"], format!("{}_{:016x}", 3 * DIM + 16, ROT_SEED));
    let v = &qc["records"]["model.layers.0.self_attn.v_proj.weight"];
    assert_eq!((v["kind"].as_str(), v["group"].as_u64()), (Some("int4g128"), Some(128)));

    // Tokenizer verbatim; the sealed config is not, and says so.
    let dig: serde_json::Value =
        serde_json::from_slice(&std::fs::read(out.join(hfpack::DIGEST_FILE)).unwrap()).unwrap();
    assert_eq!(dig["blobs"]["tokenizer.json"]["written_verbatim"], true);
    assert_eq!(dig["blobs"]["config.json"]["written_verbatim"], false);
    assert_eq!(
        std::fs::read(out.join("tokenizer.json")).unwrap(),
        blobs()[1].bytes,
        "the tokenizer must come out byte for byte"
    );
    assert_eq!(dig["config_base"]["hidden_size"], 96);
    for field in ["codes", "indices", "gains", "row_scales", "centroids", "tail"] {
        assert!(
            dig["records"]["model.layers.0.self_attn.q_proj.weight"][field].is_string(),
            "no {field} digest"
        );
    }

    // The Python package tests read this same object. `LLVQ_HF_FIXTURE=<dir>`
    // writes it there, so the fixture is by construction the directory this test
    // asserts on, and not a copy free to drift from it. A test binary runs with
    // the crate as its working directory, so the path is one level up:
    //
    //   LLVQ_HF_FIXTURE=../llvq-tetra/tests/fixtures/tiny \
    //       cargo test -p llvq-llm --test hfpack
    if let Ok(dest) = std::env::var("LLVQ_HF_FIXTURE") {
        let dest = std::path::PathBuf::from(dest);
        std::fs::create_dir_all(&dest).expect("fixture directory");
        for e in std::fs::read_dir(&out).expect("read out") {
            let e = e.expect("entry");
            std::fs::copy(e.path(), dest.join(e.file_name())).expect("copy");
        }
        let digest = dest.join(llvq_llm::hfpack::DENSE_DIGEST_FILE);
        llvq_llm::hfpack::dense_digest(&src, &digest).expect("dense digest");
        eprintln!("fixture written to {}", dest.display());
    }
}

/// The shader the Python package ships is the one this repository serves.
///
/// `llvq-tetra` carries a copy of `kernels/llvq_tetra48.metal` so it can still build
/// its Metal op after the extraction of stage 5, and `bin/tetratables` records the
/// copy's sha256 beside the tables. Two copies of a shader is one too many unless
/// something compares them, and a drifted copy would decode plausible wrong
/// points. The Python side refuses a copy whose digest moved; this refuses one
/// whose bytes moved, in the fast loop, where a `tetratables` nobody re-ran is
/// what would go unnoticed.
/// Read the table the package ships beside the tables. A test binary runs with
/// the crate as its working directory, so the package is one level up.
fn shipped_table() -> serde_json::Map<String, serde_json::Value> {
    let tables = std::path::Path::new("../llvq-tetra/llvq_tetra/data/tetra-tables.json");
    assert!(tables.exists(), "{} is missing; re-run bin/tetratables", tables.display());
    let meta: serde_json::Value =
        serde_json::from_slice(&std::fs::read(tables).expect("read")).expect("parse");
    meta["shaders"].as_object().expect("a shader table").clone()
}

/// Where the package keeps a shipped source, and where the repository serves it.
///
/// The Metal shaders go flat beside the tables. The CUDA sources keep the
/// repository's two-level shape, so the entry carries its own relative `path`.
fn shipped_and_served(file: &str, entry: &serde_json::Value) -> (PathBuf, PathBuf) {
    let data = std::path::Path::new("../llvq-tetra/llvq_tetra/data");
    match entry["path"].as_str() {
        None => (data.join(file), std::path::Path::new("kernels").join(file)),
        Some(rel) => {
            let under = rel.strip_prefix("kernels/").expect("a kernels/ prefix");
            (data.join(rel), std::path::Path::new("..").join(under))
        }
    }
}

#[test]
fn the_shipped_shader_is_the_repositorys() {
    let shaders = shipped_table();
    assert_eq!(shaders.len(), 8, "two Metal shaders and six CUDA sources travel");
    let backends = shaders
        .values()
        .filter(|e| e["backend"] == "cuda")
        .count();
    assert_eq!(backends, 6, "the CUDA include closure is six files");

    for (file, recorded) in &shaders {
        let (shipped, served) = shipped_and_served(file, recorded);
        for p in [&served, &shipped] {
            assert!(p.exists(), "{} is missing; re-run bin/tetratables", p.display());
        }
        let a = std::fs::read(&served).expect("read the served source");
        let b = std::fs::read(&shipped).expect("read the shipped copy");
        assert_eq!(
            a, b,
            "the shipped {file} differs from {}: re-run \
             `cargo run --release -p llvq-llm --bin tetratables -- \
             llvq-tetra/llvq_tetra/data/tetra-tables.safetensors`",
            served.display()
        );
        assert_eq!(
            recorded["sha256"].as_str(),
            Some(llvq_llm::hfpack::sha256_bytes(&a).as_str()),
            "the digest recorded beside the tables is not {file}'s"
        );
    }
    assert!(shaders["tv_q4_h.metal"]["entries"]
        .as_str()
        .expect("entries")
        .contains("tv_q4_metal_tiled"));
}

/// Every `#include "..."` of a shipped CUDA source resolves inside the package.
///
/// The list of six is written by hand in `hfpack::tetra_tables`, and a hand
/// written closure goes stale the first time someone adds an include upstream.
/// The wheel would then carry a CUDA arm that cannot compile, and nothing on a
/// Mac would notice, because nvcc never runs here. So recompute the closure from
/// the bytes rather than trust the list: follow every quoted include from
/// `tetra_cuda.cu` and require each target to be a file the package ships.
#[test]
fn the_shipped_cuda_closure_is_complete() {
    let shaders = shipped_table();
    let data = std::path::Path::new("../llvq-tetra/llvq_tetra/data");
    let glue = std::path::Path::new("../llvq-tetra/llvq_tetra/csrc/tetra_cuda.cu");
    assert!(glue.exists(), "{} is missing", glue.display());

    let mut seen: std::collections::BTreeSet<PathBuf> = std::collections::BTreeSet::new();
    let mut queue = vec![glue.to_path_buf()];
    while let Some(file) = queue.pop() {
        let text = std::fs::read_to_string(&file)
            .unwrap_or_else(|e| panic!("read {}: {e}", file.display()));
        let dir = file.parent().expect("a parent").to_path_buf();
        for line in text.lines() {
            let line = line.trim_start();
            // Only quoted includes: the angled ones are torch's and CUDA's.
            let Some(rest) = line.strip_prefix("#include \"") else { continue };
            let Some(name) = rest.split('"').next() else { continue };
            let target = normalize(&dir.join(name));
            assert!(
                target.exists(),
                "{} includes {name}, which the package does not ship. Add it to the \
                 list in `hfpack::tetra_tables` and re-run bin/tetratables",
                file.display()
            );
            if seen.insert(target.clone()) {
                queue.push(target);
            }
        }
    }

    // Every file reached is one the table attests, and every file attested for
    // CUDA is reached. Neither direction alone is enough: the first would let a
    // stale entry sit in the table forever, the second an unattested file ship.
    let reached: std::collections::BTreeSet<String> = seen
        .iter()
        .map(|p| p.file_name().expect("a name").to_string_lossy().into_owned())
        .collect();
    let attested: std::collections::BTreeSet<String> = shaders
        .iter()
        .filter(|(_, e)| e["backend"] == "cuda")
        .map(|(f, _)| f.clone())
        .collect();
    assert_eq!(reached, attested, "the reached closure and the attested list differ");
    for p in &seen {
        assert!(
            p.starts_with(data),
            "{} is included from outside the package's data directory",
            p.display()
        );
    }
}

/// Resolve `..` segments without touching the filesystem, which `canonicalize`
/// would, and which would turn a missing file into an error instead of a `false`.
fn normalize(p: &std::path::Path) -> PathBuf {
    let mut out: Vec<std::ffi::OsString> = Vec::new();
    for c in p.components() {
        match c {
            // A leading `..` has nothing to pop and must survive, or the path
            // silently reroots at the working directory. That cost one red test.
            std::path::Component::ParentDir => match out.last() {
                Some(last) if last != ".." => {
                    out.pop();
                }
                _ => out.push("..".into()),
            },
            std::path::Component::CurDir => {}
            other => out.push(other.as_os_str().to_owned()),
        }
    }
    out.iter().collect()
}

/// The codes tensor holds the bytes the record holds, and they unpack MSB-first
/// to the indices the artifact reads back.
///
/// This is the claim `ops/llvq_hf_check.py` makes at scale, held here in Rust so
/// a byte-order regression fails in the fast loop and not only under `uv`.
#[test]
fn the_codes_tensor_unpacks_msb_first_to_the_records_indices() {
    let dir = tempdir("hfpack-codes");
    let src = sealed_file(&dir);
    let out = dir.join("hf");
    hfpack::pack(&src, &out).expect("pack");

    // What the artifact says the indices are.
    let mut r = std::io::BufReader::new(std::fs::File::open(&src).expect("open"));
    let head = llvq_artifact::read_header(&mut r).expect("header");
    let raw = match llvq_artifact::read_record(&mut r, head.version).expect("record") {
        llvq_artifact::Record::Lattice(m) => m,
        llvq_artifact::Record::Int4(_) => panic!("the first record is the lattice one"),
    };

    let tensors =
        candle_core::safetensors::load(out.join(hfpack::MODEL_FILE), &Device::Cpu).expect("load");
    let codes: Vec<u8> = tensors["model.layers.0.self_attn.q_proj.codes"]
        .to_vec1()
        .expect("codes");
    assert_eq!(codes.len(), raw.indices.len() * 6);
    for (b, (&want_idx, &want_gain)) in raw.indices.iter().zip(&raw.gains).enumerate() {
        let word = codes[b * 6..b * 6 + 6]
            .iter()
            .fold(0u64, |acc, &byte| (acc << 8) | byte as u64);
        assert_eq!(word >> 1, want_idx, "index of block {b}");
        assert_eq!((word & 1) as u32, want_gain, "gain of block {b}");
    }
}

/// A Ball file is refused at the header, and nothing is written.
///
/// A Ball index is 47 or 48 bits wide depending on a shell cap, so its blocks
/// are not byte aligned and no reader of the written file unpacks them. The
/// refusal has to land before the first record, or half a directory exists and
/// looks complete.
#[test]
fn a_ball_file_is_refused_and_leaves_nothing_behind() {
    let dir = tempdir("hfpack-ball");
    let path = dir.join("ball.llvq");
    let mut rng = SplitMix64::new(0xBA11);
    {
        let ix = llvq_search::index::Indexer::new();
        let f = std::fs::File::create(&path).expect("create");
        let mut w = ArtifactWriter::new(std::io::BufWriter::new(f), 1).expect("header");
        let m = QuantizedMatrix {
            name: "model.layers.0.mlp.up_proj.weight".into(),
            d_out: 2,
            d_in: 2 * DIM,
            codes: (0..2 * 2)
                .map(|_| BlockCode {
                    point: ix.decode(rng.next() % 1000).expect("a low index decodes"),
                    gain: (rng.next() & 1) as u32,
                })
                .collect(),
            row_scales: vec![1.0, 2.0],
            centroids: vec![0.7, 1.1],
            rotation_seed: None,
            shell_cap: 12,
            tail: vec![],
        };
        w.push(&m).expect("push ball");
        w.seal(&[], &blobs()).expect("seal");
    }
    let out = dir.join("hf");
    let err = hfpack::pack(&path, &out).expect_err("a Ball file must be refused").to_string();
    assert!(err.contains("Ball"), "the refusal must name the kind: {err}");
    assert!(!out.exists(), "a refused file must leave no directory behind");
}

/// A scratch directory under the system temp, removed by the OS and not by us:
/// a test that deletes its own output on failure deletes the evidence.
fn tempdir(tag: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("llvq-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).expect("temp dir");
    d
}

// ---------------------------------------------------------------------------
// The mini fixture: a coherent Qwen3, small enough to commit, real enough to load.
//
// `sealed_file` above describes nothing: a 4 by 88 Tetra record beside a base
// config that says `hidden_size` 96. That is deliberate and it tests the packer
// field by field, so its shapes are not touched here. But it means
// `from_pretrained` refuses it, and until 2026-10-02 no test in either language
// loaded a model at all. The registration defect of `llvq_tetra/__init__.py` lived in
// exactly that hole.
//
// So this is a second object, beside the first. Every dimension below is forced
// by something:
//
//   * 136 = 17 x 8. The rotation is `Q = (Q_odd (x) H_m) D` with `m` the largest
//     power of two dividing `n`, so a power-of-two `n` gives a 1 by 1 `Q_odd` and
//     the trivial rotation. At 136 the odd part is 17 and `Q_odd` is a real
//     matrix. `o_proj` then takes `d_in` 128, where it is trivial. Both paths.
//   * 136 = 5*24 + 16 and 128 = 5*24 + 8, so two different non-empty tails. The
//     tail is the f32 part M1 found visible per row and invisible to the tokens.
//   * int4 sits on `down_proj` and nowhere else, because `int4_matrix` takes
//     `gpr = d_in / 128` by integer division and a `d_in` that is not a multiple
//     of 128 would silently drop its last partial group. 256 is, 136 is not. That
//     is also where the sealed 4B puts its int4.
//
// The weights are random, so the logits mean nothing. What the Python side can
// assert on it is what a fixture can say: it loads, every projection is replaced,
// and a forward pass gives finite numbers of the right shape.

const MINI_HIDDEN: usize = 17 * 8;
const MINI_INTER: usize = 256;
const MINI_HEAD_DIM: usize = 32;
const MINI_Q_HEADS: usize = 4;
const MINI_KV_HEADS: usize = 2;
const MINI_VOCAB: usize = 64;
/// The embedding is group 64, and 136 is 2 whole groups plus a partial third.
const MINI_EMBED_GROUPS: usize = MINI_HIDDEN.div_ceil(64);

/// `config.json` for the mini fixture: a Qwen3 that describes its own records.
fn mini_blobs() -> Vec<Blob> {
    let config = format!(
        r#"{{"model_type": "qwen3", "hidden_size": {h}, "num_hidden_layers": 1,
             "num_attention_heads": {q}, "num_key_value_heads": {kv}, "head_dim": {hd},
             "intermediate_size": {i}, "vocab_size": {v}, "rms_norm_eps": 1e-06,
             "tie_word_embeddings": true, "max_position_embeddings": 128,
             "rope_theta": 10000.0, "attention_bias": false, "hidden_act": "silu"}}"#,
        h = MINI_HIDDEN,
        q = MINI_Q_HEADS,
        kv = MINI_KV_HEADS,
        hd = MINI_HEAD_DIM,
        i = MINI_INTER,
        v = MINI_VOCAB,
    );
    vec![
        Blob { name: "config.json".into(), bytes: config.into_bytes() },
        Blob {
            name: "tokenizer.json".into(),
            bytes: br#"{"version": "1.0", "model": {"type": "BPE"}}"#.to_vec(),
        },
    ]
}

fn f16_tensor(name: &str, n: usize) -> RawTensor {
    RawTensor {
        name: name.into(),
        dims: vec![n],
        data: RawData::F16((0..n).map(|_| half::f16::from_f64(1.0).to_bits()).collect()),
    }
}

/// A sealed file that is a whole one-layer Qwen3. Returns its path inside `dir`.
fn sealed_mini(dir: &std::path::Path) -> std::path::PathBuf {
    let mut rng = SplitMix64::new(0x11_17);
    let path = dir.join("mini.llvq");
    let f = std::fs::File::create(&path).expect("create");
    let mut w = ArtifactWriter::with_kinds(
        std::io::BufWriter::new(f),
        VERSION,
        7,
        CodeKind::Tetra,
        KindSet::of(CodeKind::Tetra).with(CodeKind::Int4G128),
    )
    .expect("header");

    let q_out = MINI_Q_HEADS * MINI_HEAD_DIM;
    let kv_out = MINI_KV_HEADS * MINI_HEAD_DIM;
    // Six Tetra projections, in the order a layer reads them. `o_proj` is the one
    // whose `d_in` is a power of two, so it carries the trivial `Q_odd`.
    for (name, d_out, d_in) in [
        ("self_attn.q_proj", q_out, MINI_HIDDEN),
        ("self_attn.k_proj", kv_out, MINI_HIDDEN),
        ("self_attn.v_proj", kv_out, MINI_HIDDEN),
        ("self_attn.o_proj", MINI_HIDDEN, q_out),
        ("mlp.gate_proj", MINI_INTER, MINI_HIDDEN),
        ("mlp.up_proj", MINI_INTER, MINI_HIDDEN),
    ] {
        let full = format!("model.layers.0.{name}.weight");
        let m = tetra_matrix(&full, d_out, d_in, &mut rng);
        w.push_kind(&m, CodeKind::Tetra).expect("push tetra");
    }
    let down = int4_matrix("model.layers.0.mlp.down_proj.weight", MINI_HIDDEN, MINI_INTER, &mut rng);
    w.push_int4(&down).expect("push int4");

    let embed = RawTensor {
        name: "model.embed_tokens.weight".into(),
        dims: vec![MINI_VOCAB, MINI_HIDDEN],
        data: RawData::Quant(QuantData {
            bits: 4,
            group: 64,
            packed: (0..MINI_VOCAB * MINI_HIDDEN / 2).map(|_| rng.next() as u8).collect(),
            // Ceiling and not floor: 136 is two whole groups of 64 and a partial
            // third, and the format carries a scale for it. Only the
            // `int4_matrix` helper above floors, which is why int4 projections
            // here take a `d_in` that is a whole number of groups.
            scales: (0..MINI_VOCAB * MINI_EMBED_GROUPS)
                .map(|_| half::f16::from_f64(1e-3 + rng.next_f64()).to_bits())
                .collect(),
            biases: (0..MINI_VOCAB * MINI_EMBED_GROUPS)
                .map(|_| half::f16::from_f64(rng.next_gaussian()).to_bits())
                .collect(),
        }),
    };
    // Every norm the architecture asks for. A missing one is not a soft warning:
    // `from_pretrained` reports it MISSING and reinitializes it, which would make
    // the forward pass pass while measuring nothing.
    let norms = [
        f16_tensor("model.layers.0.input_layernorm.weight", MINI_HIDDEN),
        f16_tensor("model.layers.0.post_attention_layernorm.weight", MINI_HIDDEN),
        f16_tensor("model.layers.0.self_attn.q_norm.weight", MINI_HEAD_DIM),
        f16_tensor("model.layers.0.self_attn.k_norm.weight", MINI_HEAD_DIM),
        f16_tensor("model.norm.weight", MINI_HIDDEN),
    ];
    let mut raw = vec![embed];
    raw.extend(norms);
    w.seal(&raw, &mini_blobs()).expect("seal");
    path
}

/// The mini fixture is a model: the config describes the records, and nothing is
/// missing.
///
/// `LLVQ_HF_MINI_FIXTURE=<dir>` writes it where the Python suite reads it, so the
/// committed bytes are by construction the ones this test asserts on:
///
///   LLVQ_HF_MINI_FIXTURE=../llvq-tetra/tests/fixtures/mini \
///       cargo test -p llvq-llm --test hfpack
#[test]
fn the_mini_fixture_describes_a_whole_qwen3_layer() {
    let dir = tempdir("hfpack-mini");
    let src = sealed_mini(&dir);
    let out = dir.join("hf");
    let s = hfpack::pack(&src, &out).expect("pack");

    assert_eq!((s.records, s.lattice, s.int4), (7, 6, 1));
    assert_eq!(s.raw_tensors, 6, "the embedding and five norms");
    // Two rotations and not one: `d_in` 136 for five projections, 128 for o_proj.
    assert_eq!(s.rotations, 2, "the odd part differs between 136 and 128");

    let written: serde_json::Value =
        serde_json::from_slice(&std::fs::read(out.join("config.json")).expect("config")).unwrap();
    let qc = &written["quantization_config"];
    assert_eq!(written["hidden_size"], MINI_HIDDEN);
    assert_eq!(written["head_dim"], MINI_HEAD_DIM);

    // The config and the records agree, which is the whole difference with the
    // other fixture. Each projection's `d_in` is what the architecture implies.
    let q_out = MINI_Q_HEADS * MINI_HEAD_DIM;
    for (name, d_out, d_in) in [
        ("self_attn.q_proj", q_out, MINI_HIDDEN),
        ("self_attn.k_proj", MINI_KV_HEADS * MINI_HEAD_DIM, MINI_HIDDEN),
        ("self_attn.o_proj", MINI_HIDDEN, q_out),
        ("mlp.down_proj", MINI_HIDDEN, MINI_INTER),
    ] {
        let r = &qc["records"][format!("model.layers.0.{name}.weight")];
        assert_eq!(r["d_out"].as_u64(), Some(d_out as u64), "{name} d_out");
        assert_eq!(r["d_in"].as_u64(), Some(d_in as u64), "{name} d_in");
    }
    // Both tails are non-empty, and they differ.
    assert_eq!(qc["records"]["model.layers.0.self_attn.q_proj.weight"]["tail_cols"], 16);
    assert_eq!(qc["records"]["model.layers.0.self_attn.o_proj.weight"]["tail_cols"], 8);
    // int4 only where `d_in` is a whole number of groups.
    let down = &qc["records"]["model.layers.0.mlp.down_proj.weight"];
    assert_eq!(down["kind"], "int4g128");
    assert_eq!(down["groups_per_row"].as_u64(), Some((MINI_INTER / INT4G128_GROUP) as u64));

    if let Ok(dest) = std::env::var("LLVQ_HF_MINI_FIXTURE") {
        let dest = std::path::PathBuf::from(dest);
        std::fs::create_dir_all(&dest).expect("fixture directory");
        for e in std::fs::read_dir(&out).expect("read out") {
            let e = e.expect("entry");
            std::fs::copy(e.path(), dest.join(e.file_name())).expect("copy");
        }
        let digest = dest.join(llvq_llm::hfpack::DENSE_DIGEST_FILE);
        llvq_llm::hfpack::dense_digest(&src, &digest).expect("dense digest");
        eprintln!("mini fixture written to {}", dest.display());
    }
}
