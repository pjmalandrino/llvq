//! Group-affine scalar quantization for the carried tensors — the embedding.
//!
//! The tied embedding is 389 M weights at f16 on Qwen3-4B: 44 % of the sealed
//! file and 24 % of the loaded model, all for one tensor the Leech pipeline
//! never touches. This module quantizes it with the exact scheme MLX ships as
//! `q4`/`q8`: `w ≈ scale·q + bias` over groups of 64 along the innermost
//! dimension, scale and bias one f16 each per group, `q ∈ [0, 2^bits)`. MLX's
//! own 2.263 GB Qwen3-4B file — embedding included — is the evidence this
//! holds an embedding at 4 bits.
//!
//! Encoding lives here rather than in `llvq-artifact` because it needs
//! f32→f16 **rounding** (the `half` crate); the decode direction is exact
//! widening and stays in the dependency-free crate ([`RawTensor::to_f32`]).
//! `q` is chosen against the *rounded* scale and bias — the values the file
//! stores — so the error the encoder minimizes is the error the reader sees.

use half::f16;
use llvq_artifact::{QuantData, RawData, RawTensor};

/// Quantize an f16 raw tensor to `bits` ∈ {4, 8}, groups of `group` along the
/// innermost dimension. The last group of a row may be short.
pub fn quantize_affine(t: &RawTensor, bits: u8, group: usize) -> anyhow::Result<RawTensor> {
    let RawData::F16(data) = &t.data else {
        anyhow::bail!("{}: not an f16 tensor", t.name);
    };
    anyhow::ensure!(bits == 4 || bits == 8, "bits must be 4 or 8, got {bits}");
    anyhow::ensure!(group > 0, "group must be positive");
    let n = t.len();
    anyhow::ensure!(
        data.len() == n,
        "{}: {} values for dims {:?}",
        t.name,
        data.len(),
        t.dims
    );
    let row_len = t.dims.last().copied().unwrap_or(1).max(1);
    let rows = n / row_len;
    let gpr = row_len.div_ceil(group);
    let qmax = (1u32 << bits) - 1;

    let mut packed = vec![0u8; if bits == 8 { n } else { n.div_ceil(2) }];
    let mut scales = Vec::with_capacity(rows * gpr);
    let mut biases = Vec::with_capacity(rows * gpr);
    for row in 0..rows {
        for g in 0..gpr {
            let lo = row * row_len + g * group;
            let hi = row * row_len + ((g + 1) * group).min(row_len);
            let mut mn = f32::INFINITY;
            let mut mx = f32::NEG_INFINITY;
            for &b in &data[lo..hi] {
                let v = f16::from_bits(b).to_f32();
                mn = mn.min(v);
                mx = mx.max(v);
            }
            let s = f16::from_f32((mx - mn) / qmax as f32);
            // A constant group (or one whose range rounds to zero in f16) has
            // nothing to grade: every q is 0 and the bias carries the value.
            let graded = s.to_f32() > 0.0;
            let s = if graded { s } else { f16::ONE };
            let b16 = f16::from_f32(mn);
            scales.push(s.to_bits());
            biases.push(b16.to_bits());
            let (sf, bf) = (s.to_f32(), b16.to_f32());
            for (k, &b) in data[lo..hi].iter().enumerate() {
                let q = if graded {
                    ((f16::from_bits(b).to_f32() - bf) / sf)
                        .round()
                        .clamp(0.0, qmax as f32) as u32
                } else {
                    0
                };
                let i = lo + k;
                if bits == 8 {
                    packed[i] = q as u8;
                } else {
                    packed[i / 2] |= (q as u8) << (4 * (i % 2));
                }
            }
        }
    }
    Ok(RawTensor {
        name: t.name.clone(),
        dims: t.dims.clone(),
        data: RawData::Quant(QuantData {
            bits,
            group,
            packed,
            scales,
            biases,
        }),
    })
}

/// Effective storage cost of a `bits`/`group` encoding, in bits per weight.
pub fn bits_per_weight(bits: u8, group: usize) -> f64 {
    bits as f64 + 32.0 / group as f64
}
