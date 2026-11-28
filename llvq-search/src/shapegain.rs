//! The block code a shape-gain quantizer emits, and the two functions an
//! artifact needs to read one back: the reconstruction and the index width.
//!
//! They live here rather than in `llvq-quant` so that `llvq-artifact`, the
//! decoder a reader audits alone, depends on no quantizer. `llvq-quant`
//! re-exports all three at `llvq_quant::quantizer::*`, their old paths.

use llvq_core::DIM;

/// What a quantizer emitted for the block it just reconstructed — everything
/// an artifact needs to rebuild that block, and nothing else.
///
/// The lattice point rather than its 48-bit index: bijective indexing costs a
/// multiset-permutation rank per block, which has no business running inside
/// the quantization loop. A layer's points are encoded once, when the layer is
/// written out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockCode {
    /// Direction, as a point of `√8·Λ₂₄ ⊂ Z²⁴`.
    pub point: llvq_core::Point,
    /// Rank of the gain level within the matrix's fitted centroids.
    pub gain: u32,
}

/// The shape-gain reconstruction, free of any searcher: `point · g / √(16 m)`
/// with `g = centroids[gain] · row_scale` and `m` the point's shell, the
/// origin reconstructed as zero.
///
/// One function for both sides of a file: `llvq_quant::quantizer::LeechShapeGain::reconstruct`
/// calls it, and so does `llvq_artifact::decode_matrix`, which used to build
/// a whole `LeechShapeGain` — a ball searcher included — per matrix to reach
/// these six lines. The direction code is not looked at: a Tetra point is a
/// Λ₂₄ point like any other, and reconstructs through the same formula.
pub fn reconstruct_shape_gain(code: &BlockCode, centroids: &[f64], row_scale: f64, out: &mut [f64]) {
    assert_eq!(out.len(), DIM);
    match llvq_core::Leech::shell_index(&code.point) {
        Some(m) if m > 0 => {
            let picked = centroids[code.gain as usize] * row_scale;
            let scale = picked / ((16 * m) as f64).sqrt();
            for (o, &p) in out.iter_mut().zip(code.point.iter()) {
                *o = p as f64 * scale;
            }
        }
        // The origin: a zero block, representable and reconstructed as is.
        _ => out.fill(0.0),
    }
}

/// Bits needed to index `Λ₂₄(cap) ∪ {0}` — `⌈log₂(N(cap) + 1)⌉`.
///
/// 48 for the full ball, 47 for `cap = 12`. That one bit is what makes a
/// gain bit free at 2.000 bits/dim.
pub fn index_bits(cap: u32) -> u32 {
    let cs = crate::classes::enumerate_classes(cap);
    let n: u64 = (2..=cap).map(|m| cs.shell_cardinality(m)).sum();
    (128 - (n as u128).leading_zeros()).max(1)
}
