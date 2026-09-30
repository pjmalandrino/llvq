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
    //   LLVQ_HF_FIXTURE=../llvq-hf/tests/fixtures/tiny \
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
