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
    decode_matrix, read_all, write_matrix, ArtifactWriter, QuantizedMatrix,
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
