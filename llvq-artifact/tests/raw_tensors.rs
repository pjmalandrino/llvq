//! The raw-tensor section of the sealed format: encodings, versions, decode.
//!
//! Everything here runs without a model and without `half` — this crate has
//! no dependencies, and its f16 widening is checked against hand-computed
//! values here and exhaustively against the `half` crate in `llvq-llm`'s
//! test suite.

use llvq_artifact::{f16_to_f32, read_raw, write_raw, QuantData, RawData, RawTensor};

#[test]
fn f16_widening_matches_hand_computed_values() {
    // (bits, exact f32 value) — normals, subnormals, extremes.
    for (bits, want) in [
        (0x0000u16, 0.0f32),
        (0x3C00, 1.0),
        (0xC000, -2.0),
        (0x3555, 0.333_251_95), // 1365/4096
        (0x7BFF, 65504.0),      // largest normal
        (0x0400, 6.103_515_6e-5), // smallest normal, 2^-14
        (0x0001, 5.960_464_5e-8), // smallest subnormal, 2^-24
        (0x03FF, 6.097_555e-5),   // largest subnormal, 1023·2^-24
        (0x8001, -5.960_464_5e-8),
    ] {
        let got = f16_to_f32(bits);
        assert_eq!(got.to_bits(), want.to_bits(), "bits {bits:#06x}: {got} vs {want}");
    }
    assert!(f16_to_f32(0x8000).is_sign_negative() && f16_to_f32(0x8000) == 0.0);
    assert_eq!(f16_to_f32(0x7C00), f32::INFINITY);
    assert_eq!(f16_to_f32(0xFC00), f32::NEG_INFINITY);
    assert!(f16_to_f32(0x7E00).is_nan());
}

fn roundtrip(t: &RawTensor) -> RawTensor {
    let mut buf: Vec<u8> = Vec::new();
    write_raw(&mut buf, t).expect("write");
    read_raw(&mut std::io::Cursor::new(buf), 3).expect("read")
}

#[test]
fn an_f16_tensor_survives_a_v3_round_trip() {
    let t = RawTensor {
        name: "model.norm.weight".into(),
        dims: vec![2, 3],
        data: RawData::F16(vec![0x3C00, 0xC000, 0x0001, 0x7BFF, 0x0000, 0x8000]),
    };
    let got = roundtrip(&t);
    assert_eq!(got.name, t.name);
    assert_eq!(got.dims, t.dims);
    let RawData::F16(d) = &got.data else { panic!("wrong encoding came back") };
    let RawData::F16(orig) = &t.data else { unreachable!() };
    assert_eq!(d, orig);
}

/// Two rows of five values in groups of two: the last group of each row is
/// short, which is exactly the case a wrong group-indexing formula gets wrong.
fn small_quant() -> RawTensor {
    RawTensor {
        name: "model.embed_tokens.weight".into(),
        dims: vec![2, 5],
        data: RawData::Quant(QuantData {
            bits: 4,
            group: 2,
            // Values 0..10 as nibbles, low nibble first: (1,0), (3,2), … per byte.
            packed: vec![0x10, 0x32, 0x54, 0x76, 0x98],
            // 2 rows × ceil(5/2) = 6 groups.
            scales: vec![0x3C00; 6], // 1.0
            biases: vec![0x0000, 0x3C00, 0x4000, 0x4200, 0x4400, 0x4500], // 0,1,2,3,4,5
        }),
    }
}

#[test]
fn a_quant_tensor_survives_a_v3_round_trip() {
    let t = small_quant();
    let got = roundtrip(&t);
    let (RawData::Quant(a), RawData::Quant(b)) = (&got.data, &t.data) else {
        panic!("wrong encoding came back")
    };
    assert_eq!((a.bits, a.group), (b.bits, b.group));
    assert_eq!(a.packed, b.packed);
    assert_eq!(a.scales, b.scales);
    assert_eq!(a.biases, b.biases);
    assert_eq!(got.to_f32(), t.to_f32());
}

/// The decode, against an independent reference: walk groups outer-to-inner
/// rather than values in flat order, unpack nibbles by arithmetic rather than
/// shifts. A packing or group-indexing mutant passes its own round-trip and
/// fails here.
#[test]
fn quant_decode_matches_an_independent_reference() {
    let t = small_quant();
    let RawData::Quant(q) = &t.data else { unreachable!() };
    let (rows, row_len) = (t.dims[0], t.dims[1]);
    let gpr = row_len.div_ceil(q.group);
    let mut want = vec![0.0f32; rows * row_len];
    for row in 0..rows {
        for g in 0..gpr {
            let s = f16_to_f32(q.scales[row * gpr + g]);
            let b = f16_to_f32(q.biases[row * gpr + g]);
            for col in g * q.group..((g + 1) * q.group).min(row_len) {
                let i = row * row_len + col;
                let byte = q.packed[i / 2] as u32;
                let qv = (byte / 16u32.pow((i % 2) as u32)) % 16;
                want[i] = s * qv as f32 + b;
            }
        }
    }
    assert_eq!(t.to_f32(), want);
    // And the values are what the construction says: q=i, scale 1, bias per group.
    assert_eq!(want[0], 0.0); // q=0, row 0 group 0, bias 0
    assert_eq!(want[4], 6.0); // q=4, row 0's short last group (bias 2)
    assert_eq!(want[5], 8.0); // row 1 starts a fresh group row: q=5, bias 3
}

/// Pre-`LVQ3` files carry untagged f16 records; the version parameter is what
/// keeps them readable. The bytes here are hand-assembled to the `LVQ2` layout
/// so the test cannot inherit a framing bug from the writer.
#[test]
fn a_legacy_v2_record_still_reads() {
    let mut bytes: Vec<u8> = Vec::new();
    bytes.extend_from_slice(&1u32.to_le_bytes()); // name length
    bytes.extend_from_slice(b"w");
    bytes.extend_from_slice(&1u32.to_le_bytes()); // rank
    bytes.extend_from_slice(&3u64.to_le_bytes()); // dims
    bytes.extend_from_slice(&3u64.to_le_bytes()); // value count
    for v in [0x3C00u16, 0xC000, 0x7BFF] {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    let got = read_raw(&mut std::io::Cursor::new(bytes), 2).expect("legacy read");
    assert_eq!(got.name, "w");
    assert_eq!(got.dims, vec![3]);
    assert_eq!(got.to_f32(), vec![1.0, -2.0, 65504.0]);
}

#[test]
fn an_unknown_encoding_tag_is_refused() {
    let mut bytes: Vec<u8> = Vec::new();
    bytes.extend_from_slice(&7u32.to_le_bytes()); // no such tag
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(b"w");
    let err = match read_raw(&mut std::io::Cursor::new(bytes), 3) {
        Err(e) => e.to_string(),
        Ok(_) => panic!("an unknown tag must not parse"),
    };
    assert!(err.contains("encoding tag 7"), "got: {err}");
}

/// The writer must refuse a payload whose accounting disagrees with the dims —
/// a silent mismatch would be read back as plausible, wrong weights.
#[test]
fn inconsistent_quant_payloads_are_refused() {
    let mut t = small_quant();
    if let RawData::Quant(q) = &mut t.data {
        q.scales.pop();
    }
    let mut buf: Vec<u8> = Vec::new();
    assert!(write_raw(&mut buf, &t).is_err(), "one scale short must not write");
}
