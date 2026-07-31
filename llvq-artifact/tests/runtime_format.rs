//! The transcoder's contract: transcode then decode gives **exactly** what
//! `Indexer::decode` gives — bit for bit, over the same points `decfull`
//! swept — and the gain rides along untouched.
//!
//! A round-trip that only compared the transcoder to itself would prove
//! nothing (encoder and decoder could share a bug); every comparison below is
//! against the archive decoder, which was itself pinned against the format's
//! reference implementation. The GPU shader of `llvq-metal` is a third,
//! independent reading of the same bytes.

use llvq_artifact::runtime::{transcode, ClassTable, Layout, FIXED96_BYTES, GROUP};
use llvq_artifact::QuantizedMatrix;
use llvq_core::SplitMix64;
use llvq_quant::quantizer::BlockCode;
use llvq_search::fastdec::FastDecoder;
use llvq_search::index::{Indexer, N13};

/// Class boundaries + uniform draws, the `decfull` sweep, through both
/// layouts; the grouped stream is decoded in reverse order so random access
/// through the base table is exercised, not just sequential walks.
#[test]
#[cfg_attr(debug_assertions, ignore = "full-space sweep, run in release")]
fn runtime_roundtrip_is_bit_exact_everywhere() {
    let fd = FastDecoder::new();
    let ix = Indexer::new();
    let table = ClassTable::new(&fd, 1);

    let mut indices: Vec<u64> = Vec::new();
    for ci in 0..fd.n_classes() {
        let (first, last) = fd.class_range(ci);
        indices.push(first);
        indices.push(last);
    }
    let mut rng = SplitMix64::new(0x6_F0FF);
    indices.extend((0..200_000).map(|_| 1 + rng.next() % N13));
    indices.push(0); // the origin block, mid-stream
    let gains: Vec<u32> = indices.iter().map(|_| (rng.next() & 1) as u32).collect();

    for layout in [Layout::Fixed96, Layout::Grouped32] {
        let rt = transcode(&fd, &table, &indices, &gains, layout).expect("transcodes");
        assert_eq!(rt.n_blocks, indices.len());
        for b in (0..indices.len()).rev() {
            let (p, g) = rt.decode_block(&table, b);
            let want = ix.decode(indices[b]).expect("valid");
            assert_eq!(p, want, "layout {layout:?}, block {b}: point differs");
            assert_eq!(g, gains[b], "layout {layout:?}, block {b}: gain differs");
        }
    }
}

/// The whole path a real model takes: lattice points through `write_matrix`,
/// the bytes back through `read_matrix_raw`, the raw codes through the
/// transcoder — and out the far end, the same points that went in.
#[test]
fn artifact_stream_to_runtime_is_bit_exact() {
    let fd = FastDecoder::new();
    let ix = Indexer::new();
    let table = ClassTable::new(&fd, 1);
    let mut rng = SplitMix64::new(0x6_ABCD);

    // 50 rows of 4 blocks, plus a 16-column tail — the shape that exercises
    // the raw reader's field order.
    let (d_out, d_in) = (50usize, 4 * 24 + 16);
    let n = d_out * 4;
    let indices: Vec<u64> = (0..n).map(|_| 1 + rng.next() % N13).collect();
    let gains: Vec<u32> = (0..n).map(|_| (rng.next() & 1) as u32).collect();
    let m = QuantizedMatrix {
        name: "model.layers.0.test_proj.weight".into(),
        d_out,
        d_in,
        codes: indices
            .iter()
            .zip(&gains)
            .map(|(&i, &g)| BlockCode {
                point: ix.decode(i).expect("valid"),
                gain: g,
            })
            .collect(),
        row_scales: (0..d_out).map(|i| 1.0 + i as f64).collect(),
        centroids: vec![0.9, 1.1],
        rotation_seed: Some(7),
        shell_cap: 13,
        tail: vec![0.25; d_out * 16],
    };

    let mut bytes = Vec::new();
    llvq_artifact::write_matrix(&mut bytes, &ix, &m).expect("writes");
    let raw = llvq_artifact::read_matrix_raw(&mut bytes.as_slice()).expect("reads");
    assert_eq!(raw.indices, indices, "raw indices survive the stream");
    assert_eq!(raw.gains, gains, "raw gains survive the stream");

    for layout in [Layout::Fixed96, Layout::Grouped32] {
        let rt =
            transcode(&fd, &table, &raw.indices, &raw.gains, layout).expect("transcodes");
        for (b, code) in m.codes.iter().enumerate() {
            let (p, g) = rt.decode_block(&table, b);
            assert_eq!(p, code.point, "layout {layout:?}, block {b}");
            assert_eq!(g, code.gain, "layout {layout:?}, block {b}");
        }
    }
}

/// The exact worst case over the whole cap-13 table is 74 bits with a 1-bit
/// gain — the fact that makes a 96-bit fixed field lossless forever. If a
/// change to the class layout ever pushed it past 96, this fails before any
/// file does.
#[test]
fn the_table_is_bounded_by_the_fixed_layout() {
    let fd = FastDecoder::new();
    let table = ClassTable::new(&fd, 1);
    assert_eq!(table.n_entries(), 384);
    assert_eq!(table.worst_width(), 74);
    assert!(table.worst_width() <= FIXED96_BYTES as u32 * 8);
    // Two gain bits still fit with room to spare.
    assert_eq!(ClassTable::new(&fd, 2).worst_width(), 75);
}

/// Grouped strides divide by 32 even when the trailing group is partial, and
/// the base table's last entry closes the stream.
#[test]
fn grouped_strides_are_exact() {
    let fd = FastDecoder::new();
    let table = ClassTable::new(&fd, 1);
    let mut rng = SplitMix64::new(0x6_5EED);
    // 45 blocks: one full group, one partial.
    let indices: Vec<u64> = (0..45).map(|_| 1 + rng.next() % N13).collect();
    let gains = vec![0u32; indices.len()];
    let rt = transcode(&fd, &table, &indices, &gains, Layout::Grouped32).expect("ok");
    assert_eq!(rt.bases.len(), 3);
    assert_eq!(*rt.bases.last().unwrap() as usize, rt.data.len());
    for g in 0..rt.bases.len() - 1 {
        assert_eq!(
            (rt.bases[g + 1] - rt.bases[g]) % GROUP as u32,
            0,
            "group {g} stride must divide by {GROUP}"
        );
    }
}

/// Gains wider than one bit ride through unchanged — the format is not
/// married to the published file's 1-bit configuration.
#[test]
fn two_bit_gains_roundtrip() {
    let fd = FastDecoder::new();
    let ix = Indexer::new();
    let table = ClassTable::new(&fd, 2);
    let mut rng = SplitMix64::new(0x6_9A1B);
    let indices: Vec<u64> = (0..500).map(|_| 1 + rng.next() % N13).collect();
    let gains: Vec<u32> = (0..500).map(|_| (rng.next() & 3) as u32).collect();
    for layout in [Layout::Fixed96, Layout::Grouped32] {
        let rt = transcode(&fd, &table, &indices, &gains, layout).expect("ok");
        for b in 0..indices.len() {
            let (p, g) = rt.decode_block(&table, b);
            assert_eq!(p, ix.decode(indices[b]).expect("valid"));
            assert_eq!(g, gains[b]);
        }
    }
}

/// The bit order is part of the format, not of this implementation: a shader
/// reads the bytes by the spec, so an encoder and decoder that agreed on the
/// *wrong* order would pass every CPU round-trip and break on the GPU. One
/// hand-computed block pins it: the origin (class field 0) with gain 1 puts
/// its gain bit at stream bit 9 — LSB-first, that is byte 1, bit 1.
#[test]
fn the_bit_order_is_lsb_first() {
    let fd = FastDecoder::new();
    let table = ClassTable::new(&fd, 1);
    let rt = transcode(&fd, &table, &[0], &[1], Layout::Fixed96).expect("ok");
    assert_eq!(rt.data.len(), FIXED96_BYTES);
    assert_eq!(rt.data[0], 0, "class field 0 leaves byte 0 empty");
    assert_eq!(rt.data[1], 0b0000_0010, "gain bit lands at byte 1, bit 1");
    assert!(rt.data[2..].iter().all(|&b| b == 0), "padding is zero");
}

/// An index outside the ball must refuse to transcode — a truncated or
/// corrupt archive becomes an error, never plausible weights.
#[test]
fn out_of_range_index_is_refused() {
    let fd = FastDecoder::new();
    let table = ClassTable::new(&fd, 1);
    let r = transcode(&fd, &table, &[N13 + 1], &[0], Layout::Fixed96);
    assert!(r.is_err(), "an index past N13 must not transcode");
}
