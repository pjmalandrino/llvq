//! # Gate G6 — the artifact format, tested without a model
//!
//! The end-to-end check in `llvq-llm`'s smoke run is the real guarantee, but
//! it costs a
//! quantization run. These tests exercise the same format on synthetic
//! matrices in milliseconds, so a serialization bug fails here rather than
//! three hours into Qwen3-4B.
//!
//! Every field in the format is one the decoder needs. The way to prove that
//! is to corrupt each one in turn and watch the round-trip break — which is
//! what `every_stored_field_is_load_bearing` does.

use llvq_core::{SplitMix64, DIM};
use llvq_artifact::{
    decode_matrix, read_all, read_raw, write_matrix, write_raw, ArtifactWriter, QuantizedMatrix,
    RawData, RawTensor,
};
use llvq_quant::quantizer::BlockCode;
use llvq_search::index::Indexer;

/// Build a matrix whose codes are real lattice points, drawn by decoding
/// random indices — the codebook is the only source of valid points.
#[allow(clippy::too_many_arguments)] // a format has many fields; that is the point
fn synthetic(
    ix: &Indexer,
    rng: &mut SplitMix64,
    name: &str,
    d_out: usize,
    d_in: usize,
    shell_cap: u32,
    n_gain: usize,
    rotation_seed: Option<u64>,
) -> QuantizedMatrix {
    let nblocks = d_in / DIM;
    let hi = llvq_quant::quantizer::index_bits(shell_cap);
    let codes: Vec<BlockCode> = (0..d_out * nblocks)
        .map(|_| {
            // Draw until the index lands inside the capped ball.
            loop {
                let i = rng.next() % (1u64 << hi);
                if let Some(point) = ix.decode(i) {
                    if let Some(m) = llvq_core::Leech::shell_index(&point) {
                        if m <= shell_cap as u64 {
                            return BlockCode {
                                point,
                                gain: (rng.next() % n_gain as u64) as u32,
                            };
                        }
                    } else if point == [0; DIM] {
                        return BlockCode { point, gain: 0 };
                    }
                }
            }
        })
        .collect();
    QuantizedMatrix {
        name: name.to_string(),
        d_out,
        d_in,
        codes,
        row_scales: (0..d_out).map(|_| 1e-3 + rng.next_f64()).collect(),
        centroids: (0..n_gain).map(|k| 0.5 + 0.4 * k as f64).collect(),
        rotation_seed,
        shell_cap,
        tail: (0..d_out * (d_in % DIM))
            .map(|_| rng.next_gaussian() as f32 as f64)
            .collect(),
    }
}

fn round_trip(m: &QuantizedMatrix) -> QuantizedMatrix {
    let mut buf: Vec<u8> = Vec::new();
    {
        let mut w = ArtifactWriter::new(&mut buf, 1).expect("header");
        w.push(m).expect("write");
        w.finish().expect("flush");
    }
    let mut r = std::io::Cursor::new(buf);
    let mut got = read_all(&mut r).expect("read");
    assert_eq!(got.len(), 1);
    got.pop().unwrap()
}

#[test]
fn a_matrix_survives_a_round_trip() {
    let ix = Indexer::new();
    let mut rng = SplitMix64::new(0x6_F001);
    for (d_out, d_in, cap, gains, seed) in [
        (4usize, 3 * DIM, 13u32, 2usize, None),
        (4, 3 * DIM, 12, 2, Some(0xABCD_1234)),
        (3, 100, 12, 1, Some(7)), // 100 = 24·4 + 4, so there is a tail
        (5, 5 * DIM, 12, 4, None),
    ] {
        let m = synthetic(&ix, &mut rng, "model.layers.0.self_attn.q_proj.weight", d_out, d_in, cap, gains, seed);
        let got = round_trip(&m);
        assert_eq!(got.name, m.name);
        assert_eq!((got.d_out, got.d_in), (m.d_out, m.d_in));
        assert_eq!(got.shell_cap, m.shell_cap);
        assert_eq!(got.rotation_seed, m.rotation_seed);
        assert_eq!(got.codes, m.codes, "codes differ for {d_out}×{d_in}");
        assert_eq!(got.centroids, m.centroids);
        assert_eq!(got.row_scales, m.row_scales, "scales must survive exactly");
        assert_eq!(got.tail, m.tail, "the tail must survive exactly");
        // And the decoded weights, which is what actually matters.
        assert_eq!(decode_matrix(&got), decode_matrix(&m));
    }
}

#[test]
fn several_matrices_stream_back_in_order() {
    let ix = Indexer::new();
    let mut rng = SplitMix64::new(0x6_F002);
    let mats: Vec<QuantizedMatrix> = (0..5)
        .map(|k| {
            synthetic(
                &ix,
                &mut rng,
                &format!("model.layers.{k}.mlp.down_proj.weight"),
                3,
                2 * DIM,
                12,
                2,
                Some(k as u64),
            )
        })
        .collect();

    let mut buf: Vec<u8> = Vec::new();
    {
        let mut w = ArtifactWriter::new(&mut buf, mats.len() as u32).expect("header");
        for m in &mats {
            w.push(m).expect("write");
        }
        w.finish().expect("flush");
    }
    let got = read_all(&mut std::io::Cursor::new(buf)).expect("read");
    assert_eq!(got.len(), mats.len());
    for (a, b) in got.iter().zip(mats.iter()) {
        assert_eq!(a.name, b.name);
        assert_eq!(a.codes, b.codes);
        assert_eq!(decode_matrix(a), decode_matrix(b));
    }
}

/// Every field the format stores must change the decoded weights.
///
/// A field that can be corrupted without the reconstruction moving is a field
/// the format did not need — or, far worse, one the decoder silently ignores
/// while the encoder writes it. Both are worth knowing.
#[test]
fn every_stored_field_is_load_bearing() {
    let ix = Indexer::new();
    let mut rng = SplitMix64::new(0x6_F003);
    let base = synthetic(
        &ix,
        &mut rng,
        "model.layers.0.mlp.up_proj.weight",
        4,
        3 * DIM,
        12,
        2,
        Some(0x1234),
    );
    let want = decode_matrix(&base);

    let mut m = QuantizedMatrix { ..clone_of(&base) };
    m.row_scales[0] *= 1.5;
    assert_ne!(decode_matrix(&m), want, "row scale is ignored by the decoder");

    let mut m = clone_of(&base);
    m.centroids[0] *= 1.5;
    assert_ne!(decode_matrix(&m), want, "centroids are ignored");

    let mut m = clone_of(&base);
    m.rotation_seed = Some(0x5678);
    assert_ne!(decode_matrix(&m), want, "rotation seed is ignored");

    let mut m = clone_of(&base);
    m.rotation_seed = None;
    assert_ne!(decode_matrix(&m), want, "un-rotation is skipped entirely");

    let mut m = clone_of(&base);
    m.codes[0].gain ^= 1;
    assert_ne!(decode_matrix(&m), want, "gain rank is ignored");

    let mut m = clone_of(&base);
    m.codes[0].point = ix.decode(12345).expect("a valid point");
    assert_ne!(decode_matrix(&m), want, "the lattice point is ignored");
}

fn clone_of(m: &QuantizedMatrix) -> QuantizedMatrix {
    QuantizedMatrix {
        name: m.name.clone(),
        d_out: m.d_out,
        d_in: m.d_in,
        codes: m.codes.clone(),
        row_scales: m.row_scales.clone(),
        centroids: m.centroids.clone(),
        rotation_seed: m.rotation_seed,
        shell_cap: m.shell_cap,
        tail: m.tail.clone(),
    }
}

/// A point outside the capped ball must be refused at write time, not written
/// as a truncated index that decodes to a different, perfectly valid point.
#[test]
fn a_point_above_the_shell_cap_is_refused() {
    let ix = Indexer::new();
    let mut rng = SplitMix64::new(0x6_F004);
    let mut m = synthetic(&ix, &mut rng, "m", 2, DIM, 12, 2, None);
    // A shell-13 point cannot be indexed in the 47 bits a cap of 12 allows.
    let big = (0..)
        .map(|k| ix.decode(llvq_search::index::N13 - k).expect("valid"))
        .find(|p| llvq_core::Leech::shell_index(p) == Some(13))
        .expect("shell 13 exists at the top of the ball");
    m.codes[0].point = big;
    let mut buf: Vec<u8> = Vec::new();
    let err = write_matrix(&mut buf, &ix, &m).unwrap_err().to_string();
    assert!(
        err.contains("does not fit") || err.contains("shell cap"),
        "expected a refusal about the index width, got: {err}"
    );
}

/// A matrix read undecoded and written back must produce the same bytes.
///
/// This is the contract that lets `embedq` rewrite a sealed file's raw-tensor
/// section while copying 150 M blocks through untouched: no decode, no
/// re-encode, no trust required. Any drift in field order or width between
/// `write_matrix` and `write_matrix_raw` fails here.
#[test]
fn raw_passthrough_is_byte_identical() {
    let ix = Indexer::new();
    let mut rng = SplitMix64::new(0x6_F005);
    let mats = [
        // A tail, a rotation, cap 13 — every optional part present.
        synthetic(&ix, &mut rng, "model.layers.0.self_attn.q_proj.weight", 4, 3 * DIM + 16, 13, 2, Some(7)),
        synthetic(&ix, &mut rng, "model.layers.1.mlp.gate_proj.weight", 3, 2 * DIM, 12, 2, None),
    ];
    let mut original: Vec<u8> = Vec::new();
    {
        let mut w = ArtifactWriter::new(&mut original, mats.len() as u32).expect("header");
        for m in &mats {
            w.push(m).expect("write");
        }
        w.finish().expect("flush");
    }

    let mut r = std::io::Cursor::new(&original);
    let head = llvq_artifact::read_header(&mut r).expect("header");
    let mut copied: Vec<u8> = Vec::new();
    {
        let mut w = ArtifactWriter::new(&mut copied, head.matrices).expect("header");
        for _ in 0..head.matrices {
            let raw = llvq_artifact::read_matrix_raw(&mut r).expect("read raw");
            w.push_raw(&raw).expect("write raw");
        }
        w.finish().expect("flush");
    }
    assert_eq!(original, copied, "passthrough must not change a single byte");
}

/// The file published on Hugging Face is `LVQ2`; the writer now emits `LVQ3`.
/// Backward compatibility is therefore **load-bearing**: 1.77 GB of published
/// artifact depend on `read_raw` still understanding an untagged record, and
/// nothing tested it.
///
/// The second assertion is the lethal one. `LVQ3` prefixes every raw tensor
/// with a 4-byte encoding tag and `LVQ2` does not, so if `read_raw` ignored
/// its `version` argument and always consumed a tag, a v2 record would be
/// parsed with its **name length** mistaken for the tag. Reading v2 bytes as
/// v3 must fail; a version argument that changes nothing is dead code, and a
/// dead version argument here silently breaks every published file.
#[test]
fn an_untagged_v2_raw_tensor_still_reads() {
    let t = RawTensor {
        name: "model.embed_tokens.weight".into(),
        dims: vec![3, 4],
        // 1.0, -1.0, the smallest subnormal, the largest finite, then filler.
        data: RawData::F16(vec![0x3C00, 0xBC00, 0x0001, 0x7BFF, 5, 6, 7, 8, 9, 10, 11, 12]),
    };

    let mut v3: Vec<u8> = Vec::new();
    write_raw(&mut v3, &t).expect("write");

    // `LVQ2` framing is exactly `LVQ3` minus the leading encoding tag.
    let v2 = &v3[4..];

    let mut r = std::io::Cursor::new(v2);
    let back = read_raw(&mut r, 2).expect("a v2 record must still read");
    assert_eq!(back.name, t.name);
    assert_eq!(back.dims, t.dims);
    match (&back.data, &t.data) {
        (RawData::F16(a), RawData::F16(b)) => assert_eq!(a, b, "v2 values must survive"),
        _ => panic!("an untagged record must decode as f16"),
    }

    let mut r = std::io::Cursor::new(v2);
    assert!(
        read_raw(&mut r, 3).is_err(),
        "reading a v2 record as v3 must fail — if it succeeds, the tag is not \
         being consumed and the version argument is dead code"
    );

    // And the converse: a v3 record read as v2 must not silently pass either.
    let mut r = std::io::Cursor::new(&v3);
    match read_raw(&mut r, 2) {
        Err(_) => {}
        Ok(bad) => assert_ne!(bad.name, t.name, "a tagged record must not read as untagged"),
    }
}
