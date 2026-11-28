//! The bit price of each candidate layout, per block and per matrix: the
//! constants, the widths of the variable encodings, the frozen strides of the
//! bit-plane and E1c streams, and the shape counts of the kernel accounting.

use super::*;

/// Bits for the class id: 383 classes plus the origin fit in 9.
pub(super) const CLASS_BITS: u64 = 9;
/// SIMD group width the grouped layout is sized for.
pub(super) const GROUP: usize = 32;
/// Bits an `f32` costs. The kernel accounting stores the tail block and the
/// row-scale vector in single precision — not `f16`, not the `f64` the
/// archive keeps them in — because that is what every arm of the CUDA bench
/// uploads. Naming it once keeps the three layouts on one convention.
pub(super) const F32_BITS: u64 = 32;
/// Values per scale/bias pair on the `q8` embedding path
/// (`llvq_llm::fused::EMBED_GROUP`), and the 32 bits that pair costs.
pub(super) const EMBED_GROUP: f64 = 64.0;
pub(super) const EMBED_GROUP_BITS: f64 = 32.0;

/// ⌈log₂ L⌉ — bits per slot of the positional encoding.
fn lg_ceil(l: usize) -> u64 {
    match l {
        0 | 1 => 0,
        _ => (usize::BITS - (l - 1).leading_zeros()) as u64,
    }
}

/// Nested-mask bits: canonical order, one mask per level over the slots the
/// previous levels left free; the last level is whatever remains.
fn mask_bits(lv: &ClassLevels) -> u64 {
    let mut left = DIM as u64;
    let mut bits = 0u64;
    for i in 0..lv.len.saturating_sub(1) {
        bits += left;
        left -= lv.counts[i] as u64;
    }
    bits
}

/// Per-block widths of the variable candidates, everything included
/// except addressing.
#[derive(Clone, Copy)]
pub(super) struct Widths {
    pub(super) mask: u64,
    pub(super) pos: u64,
    /// `Layout::Slot32`, the production layout — mirrors `width_slot` in
    /// `llvq_artifact::runtime::ClassTable::new`. The sign field there is
    /// one bit per **slot**, not per nonzero, so `nonzero` cancels out of
    /// `common` and must not appear.
    pub(super) slot: u64,
    pub(super) levels: usize,
    /// The all-zero block, whose record stops after the header and so does
    /// not follow the `9 + gain + 24·L` rule.
    pub(super) origin: bool,
}

pub(super) fn widths(lv: &ClassLevels, gain_bits: u64) -> Widths {
    let common = lv.nonzero as u64 + CLASS_BITS + gain_bits;
    Widths {
        mask: mask_bits(lv) + common,
        pos: lg_ceil(lv.len) * DIM as u64 + common,
        slot: CLASS_BITS + gain_bits + DIM as u64 * lv.len as u64,
        levels: lv.len,
        origin: false,
    }
}

/// `Slot32` stride, in bits, of a group whose widest block has `levels`
/// levels — the byte-rounded field every lane of the group then reads.
pub(super) fn slot_stride_bits(levels: usize, gain_bits: u64) -> u64 {
    (CLASS_BITS + gain_bits + DIM as u64 * levels as u64).div_ceil(8) * 8
}

// ---------------------------------------------------------------------------
// The two bit-plane layouts, priced post by post
// ---------------------------------------------------------------------------
//
// Both are **frozen strides**, not computed ones: `transcode_planes14` and
// `transcode_planes12x` write a constant number of bytes per block whatever
// the class, so the constants come from `llvq_artifact::runtime` rather than
// from a local literal. If the format moves, this bench moves with it or
// fails to compile — the drift that a hard-coded `14` would hide.

/// `Planes14` block stream, in bits. Post by post, per block:
///
/// * class 9 + gain 1 — the header, same two fields every layout pays;
/// * sign mask 24 — one bit per **slot**, not per nonzero: zero slots carry
///   a bit written 0, which is what buys the fixed field offsets;
/// * three 24-bit planes — the slot's level as an explicit 3-bit index
///   (`len <= 5`, so 3 planes always suffice), instead of `L−1` one-hot
///   masks. This is the whole trade: `Slot32` pays `24·L` and needs a
///   per-group stride, `Planes14` pays a flat `72` and needs none;
/// * 6 padding bits to close the [`PLANES14_BYTES`]-byte record.
///
/// Sums to `9 + 1 + 24 + 72 + 6 = 112` bits — and the origin block, whose
/// planes and sign mask are all zero, pays the same 112: the stride is the
/// format. **No base table**, so there is no addressing term to add.
pub(super) fn planes14_bits(blocks: u64) -> u64 {
    blocks * PLANES14_BYTES as u64 * 8
}

/// `Planes12x` main stream **plus** its exception table, in bits.
///
/// Main record: class 9 + gain 1 + sign mask 24 + **two** 24-bit planes + 14
/// padding bits = [`PLANES12X_BYTES`] bytes. Two planes address levels
/// `0..4`, so this record is exact for every block whose class holds at most
/// [`PLANES12X_LEVEL_CAP`] magnitudes.
///
/// A five-level block cannot be said in two planes. The transcoder puts its
/// best `L ≤ 4` neighbour in the main stream and repeats the block exactly in
/// the exception table, at [`PLANES12X_EXC_BYTES`] bytes: a `u32` block index
/// (4 B, so the correction pass can find it) plus a full [`PLANES14_BYTES`]
/// record of the exact block. The exception is an **addition**, never a
/// replacement — the main record is written either way, which is why the
/// two terms add rather than trade.
///
/// So the count of exceptions is a count of five-level classes, and nothing
/// else: no lattice search, no decoding, no probability. `n_exc` here is the
/// same number `transcode_planes12x` will push onto `exc_idx`.
pub(super) fn planes12x_bits(blocks: u64, exceptions: u64) -> u64 {
    blocks * PLANES12X_BYTES as u64 * 8 + exceptions * PLANES12X_EXC_BYTES as u64 * 8
}

// ---------------------------------------------------------------------------
// E1c — the same two streams, transposed over the warp (item X1/X2 of
// docs/archive/spec-memoire-extreme-2026-08-12.md)
// ---------------------------------------------------------------------------
//
// `Planes14` and `Planes12x` each close their record on a byte boundary and
// pay for it: 6 padding bits per block and 14 respectively. Transposing the
// per-slot bits over a group of [`E1C_GROUP`] blocks removes that rounding
// entirely — a group is a whole number of 32-bit words, because the group *is*
// the word width — so E1c costs exactly the source's **payload** bits and
// nothing more.
//
// This is a pricing arm, not a claim about speed. The layout trades 82 or 106
// broadcast loads per group against 3 or 4 per lane, and which side wins is
// the question `bin/planesbench` exists to answer (spec X3: ≥ 1.9× to replace
// `Planes12x`, ≥ 2.05× to replace `Planes14`). What this bench can settle, and
// what it settles exactly, is the byte count.

/// `E1c14` stream, in bits: whole groups of [`E1C_GROUP`] blocks, each
/// [`group_words`]`(3)` words. The trailing group is padded with origin
/// blocks and **is charged** — a matrix whose block count is not a multiple
/// of 32 really does upload those words.
pub(super) fn e1c14_bits(blocks: u64) -> u64 {
    blocks.div_ceil(E1C_GROUP as u64) * group_words(E1C14_PLANES) as u64 * 32
}

/// `E1c12` main stream plus the **same** exception table as `Planes12x` —
/// same predicate, same records, carried verbatim by
/// [`E1c12Blocks::from_planes12x`]. The two layouts therefore differ by
/// exactly the main stream's padding, which is what makes the comparison
/// below a single-variable one.
///
/// [`E1c12Blocks::from_planes12x`]: llvq_artifact::e1c::E1c12Blocks::from_planes12x
pub(super) fn e1c12_bits(blocks: u64, exceptions: u64) -> u64 {
    blocks.div_ceil(E1C_GROUP as u64) * group_words(E1C12_PLANES) as u64 * 32
        + exceptions * PLANES12X_EXC_BYTES as u64 * 8
}

/// A block is an exception of `Planes12x` exactly when its class carries more
/// than [`PLANES12X_LEVEL_CAP`] magnitude levels — the predicate
/// `planes12x_chunk` tests, written once so a test can mutate it.
///
/// The origin (one level, every coordinate zero) is never an exception.
pub(super) fn is_planes12x_exception(levels: usize) -> bool {
    levels > PLANES12X_LEVEL_CAP
}

/// The same predicate read off the **record geometry** instead of the named
/// cap: `k` bit-planes address `2^k` levels, and a `Planes12x` main record
/// holds [`PLANES12X_PLANE_BIT`]`.len()` of them.
///
/// Two routes to one number, on purpose. A cross-check whose two sides call
/// the same function is a tautology and proves nothing; this one would
/// survive only if the cap constant and the number of planes moved together,
/// which is the only way they *may* move.
pub(super) fn exceeds_planes12x_geometry(levels: usize) -> bool {
    levels > 1 << PLANES12X_PLANE_BIT.len()
}

/// The shape counts one matrix contributes to the **kernel** accounting.
///
/// Named for shapes, not for the carried tensors: `carried_params` below is
/// a different thing entirely (the embedding), and confusing the two is how
/// a b/param figure goes wrong.
///
/// Mirrors `Mat`'s three counts in `llvq-cuda/src/bin/planesbench.rs`:
/// `nblocks = d_out · (d_in / 24)`, `tail_w = d_in % 24` columns kept in
/// full precision on every row, and `d_out` row scales.
#[derive(Default, Clone, Copy)]
pub(super) struct Shapes {
    /// `Σ d_out · d_in` — every weight, tail columns included. **This** is
    /// the kernel accounting's denominator, and it is larger than
    /// `24 · blocks`; swapping the two is the mistake that turns 4.804 into
    /// 4.827.
    pub(super) weights: u64,
    /// `Σ d_out · (d_in mod 24)` — the `TailPolicy::KeepExact` columns, one
    /// `f32` each on the device.
    pub(super) tail_weights: u64,
    /// `Σ d_out` — one `f32` row scale each.
    pub(super) rows: u64,
    /// `Σ d_out · ceil((d_in/24)/32)·32` — the blocks an E1c stream holds once
    /// every ROW is padded to a whole number of warp groups.
    ///
    /// 🚨 **It adds bits and NOT ONE weight**, and that asymmetry is the whole
    /// verdict: the padded blocks are synthetic, so they inflate the numerator
    /// of a b/weight while [`Self::weights`] stays put. X3 measured the block
    /// inflation at +15.47 % on the shapes of the published 4B; what it did
    /// not do is carry it through the exception term, which is why `e1c12` was
    /// left without a verdict.
    pub(super) padded_blocks: u64,
}

impl Shapes {
    pub(super) fn push(&mut self, d_out: usize, d_in: usize) {
        self.weights += (d_out * d_in) as u64;
        self.tail_weights += (d_out * (d_in % DIM)) as u64;
        self.rows += d_out as u64;
        // The remedy X3 showed to be the only one compatible with the served
        // matvec: pad each ROW out to a multiple of 32 blocks. A partial E1c
        // group costs a full one whatever the occupancy, so there is nothing
        // cheaper to do.
        let row_blocks = d_in / DIM;
        self.padded_blocks += (d_out * row_blocks.div_ceil(32) * 32) as u64;
    }

    /// Bits the kernel accounting adds to a block stream: the `f32` tail and
    /// the `f32` row scales, uploaded identically by every LLVQ arm.
    pub(super) fn side_bits(&self) -> u64 {
        (self.tail_weights + self.rows) * F32_BITS
    }

    /// Bits per weight in the kernel accounting: stream + side data over
    /// **all** weights.
    pub(super) fn kernel_bpw(&self, stream_bits: u64) -> f64 {
        (stream_bits + self.side_bits()) as f64 / self.weights as f64
    }
}
