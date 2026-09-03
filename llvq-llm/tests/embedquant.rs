//! The embedding quantizer, held to the same standard as the lattice one.
//!
//! The accounting lesson (`the_gain_is_actually_quantized`) applies here
//! verbatim: a quantizer whose output can take more distinct values than its
//! bit budget allows is spending bits the format does not charge for. So the
//! central test is not "the error is small" but "the reconstruction really is
//! `scale·q + bias` with q in `[0, 2^bits)`".

use half::f16;
use llvq_artifact::{RawData, RawTensor};
use llvq_llm::embedquant::quantize_affine;

/// The dependency-free f16 widening in `llvq-artifact`, against the `half`
/// crate, for every one of the 65,536 bit patterns. NaNs compare as NaNs;
/// everything else must match to the bit.
#[test]
fn f16_widening_matches_the_half_crate_exhaustively() {
    for bits in 0..=u16::MAX {
        let ours = llvq_artifact::f16_to_f32(bits);
        let theirs = f16::from_bits(bits).to_f32();
        if theirs.is_nan() {
            assert!(ours.is_nan(), "bits {bits:#06x}: {ours} should be NaN");
        } else {
            assert_eq!(
                ours.to_bits(),
                theirs.to_bits(),
                "bits {bits:#06x}: {ours} vs {theirs}"
            );
        }
    }
}

/// Deterministic values with realistic spread — a hand-rolled generator so
/// the test owns its randomness.
fn test_tensor(rows: usize, row_len: usize, seed: u64) -> RawTensor {
    let mut s = seed;
    let data: Vec<u16> = (0..rows * row_len)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            // Roughly N(0, 0.02) — the magnitude of real embedding weights.
            let u = ((s >> 33) as f64 / (1u64 << 31) as f64) - 0.5;
            f16::from_f64(u * 0.08).to_bits()
        })
        .collect();
    RawTensor {
        name: "model.embed_tokens.weight".into(),
        dims: vec![rows, row_len],
        data: RawData::F16(data),
    }
}

/// An independent dequantizer written against the *spec* (groups along the
/// innermost dim, low nibble first), not against the implementation.
fn reference_dequant(t: &RawTensor) -> Vec<f32> {
    let RawData::Quant(q) = &t.data else { panic!("not quantized") };
    let row_len = *t.dims.last().unwrap();
    let rows = t.len() / row_len;
    let gpr = row_len.div_ceil(q.group);
    let mut out = Vec::with_capacity(t.len());
    for i in 0..rows * row_len {
        let g = (i / row_len) * gpr + (i % row_len) / q.group;
        let qv = if q.bits == 8 {
            q.packed[i] as u32
        } else if i % 2 == 0 {
            (q.packed[i / 2] & 0xf) as u32
        } else {
            (q.packed[i / 2] >> 4) as u32
        };
        out.push(
            f16::from_bits(q.scales[g]).to_f32() * qv as f32 + f16::from_bits(q.biases[g]).to_f32(),
        );
    }
    out
}

/// 130 = 2·64 + 2: every row ends in a short group, the case a wrong
/// group-indexing formula gets wrong. Both widths, decode checked bit-for-bit
/// against the independent reference, error bounded by the stored scale.
#[test]
fn quantize_then_decode_is_within_one_scale_step() {
    for (bits, seed) in [(8u8, 1u64), (4, 2)] {
        let t = test_tensor(6, 130, seed);
        let RawData::F16(orig) = &t.data else { unreachable!() };
        let q = quantize_affine(&t, bits, 64).expect("quantize");
        let got = q.to_f32();
        assert_eq!(got, reference_dequant(&q), "int{bits}: decoder disagrees with the spec");

        let RawData::Quant(qd) = &q.data else { unreachable!() };
        let gpr = 130usize.div_ceil(64);
        for (i, (&ob, &gv)) in orig.iter().zip(&got).enumerate() {
            let g = (i / 130) * gpr + (i % 130) / 64;
            let step = f16::from_bits(qd.scales[g]).to_f32();
            let err = (f16::from_bits(ob).to_f32() - gv).abs();
            assert!(
                err <= step,
                "int{bits}, value {i}: error {err} exceeds the scale step {step}"
            );
        }
    }
}

/// The bit budget is real: within one group the reconstruction may take at
/// most 2^bits distinct values. A quantizer that smuggles extra precision
/// (the shape–gain accounting bug, in miniature) fails here.
#[test]
fn the_embedding_is_actually_quantized() {
    for bits in [8u8, 4] {
        let t = test_tensor(4, 128, 3);
        let q = quantize_affine(&t, bits, 64).expect("quantize");
        let vals = q.to_f32();
        for (g, chunk) in vals.chunks(64).enumerate() {
            let mut distinct: Vec<u32> = chunk.iter().map(|v| v.to_bits()).collect();
            distinct.sort_unstable();
            distinct.dedup();
            assert!(
                distinct.len() <= 1 << bits,
                "group {g}: {} distinct values from {bits} bits",
                distinct.len()
            );
        }
    }
}

/// A constant group carries its value in the bias, exactly — and must not
/// divide by a zero scale on the way.
#[test]
fn a_constant_group_reconstructs_exactly() {
    let v = f16::from_f32(0.03125); // exactly representable
    let t = RawTensor {
        name: "w".into(),
        dims: vec![2, 64],
        data: RawData::F16(vec![v.to_bits(); 128]),
    };
    for bits in [8u8, 4] {
        let q = quantize_affine(&t, bits, 64).expect("quantize");
        assert!(q.to_f32().iter().all(|&x| x == v.to_f32()), "int{bits}");
    }
}

/// What `embedq` writes, `read_raw` gives back — including through the
/// serialized form, since the file is the deliverable.
#[test]
fn a_quantized_embedding_survives_serialization() {
    let t = test_tensor(3, 96, 4);
    let q = quantize_affine(&t, 4, 64).expect("quantize");
    let mut buf: Vec<u8> = Vec::new();
    llvq_artifact::write_raw(&mut buf, &q).expect("write");
    let got = llvq_artifact::read_raw(&mut std::io::Cursor::new(buf), 3).expect("read");
    assert_eq!(got.name, q.name);
    assert_eq!(got.dims, q.dims);
    assert_eq!(got.to_f32(), q.to_f32());
}
