//! `tetra48`, the served runtime layout of the Tetra word — and the one
//! transcoding in this crate that adds nothing and removes nothing.
//!
//! Every other layout in [`crate::runtime`] takes a ball index, looks its class
//! up, and **re-encodes** the block into a kernel-shaped record: Planes14
//! spends 14 bytes where the disk spent 6, and buys a branchless decode with
//! them. Tetra does not unfold. The 48-bit word the encoder wrote is the word
//! the kernel reads, and this file only moves it between two conventions:
//!
//! | | disk | word |
//! |---|---|---|
//! | order | **MSB-first**, dense, 47 index bits then 1 gain bit, back to back with no padding ([`llvq_search::pack`]) | **little-endian**, 48 bits in 6 bytes |
//! | addressing | one continuous stream per matrix | **row-strided**, `stride_u32` u32 per row |
//!
//! Both halves are format decisions and both are load-bearing. A reader that
//! disagrees about either returns *different weights* rather than failing —
//! which is why `pack.rs` spells its byte order out, and why this module
//! spells out the other one.
//!
//! ## Why rows are strided and Planes14's are not
//!
//! `tv_planes` addresses blocks flat, `row · nblocks + j`, because a 14-byte
//! record starting at `14·b` is 2-aligned and its four-word read window is a
//! constant of the layout. A 6-byte record is not: `f1r_load`
//! (`llvq-cuda/kernels/llvq_f1rank.cuh`) reads `row[(3j) >> 1]` and the next
//! word, so a row must start on a u32 boundary or every row after the first
//! reads at a shifted phase. Hence [`stride_u32`], and hence the pad: the last
//! block of a row reads up to two words past its own six bytes.
//!
//! ## What this file does NOT do
//!
//! It does not validate. A Tetra word is 48 bits and **every** 48-bit value is
//! a label (`llvq_search::tetra`, "the decode is a bijection from the 2⁴⁷
//! labels onto 2⁴⁷ lattice points"), so there is no out-of-range index to
//! catch here — unlike a ball index, where `class_id` is a real check. What
//! this file must get right is the two orders, and that is what its tests pin.

use crate::{Error, Result};
use llvq_core::DIM;
use llvq_search::tetra::{Tetra, LABEL_BITS, WORD_MASK};

/// Bytes one Tetra word occupies in the served stream.
pub const TETRA48_BYTES: usize = 6;

/// Entries of the inverse-norm table the kernel reads.
///
/// `m = ‖y‖²/16` never exceeds 27 on this codebook — swept over all 4,096 rows
/// against both parities and both residues by `llvq-bench/examples/tetrashell.rs`,
/// which reports `max |y_j| = 10` and a worst section sum of 144, hence
/// `n2 ≤ 432`. 32 is that bound rounded up to a power of two, and
/// `llvq_tetra48.cuh` masks with it so a corrupt word reads a wrong scale
/// rather than off the end of the table.
pub const TETRA48_SHELLS: usize = 32;

/// u32 per row: `round_up(6 · nblocks, 8) / 4`.
///
/// Rounding to **8** bytes and not 4 is what covers the last block's two-word
/// read window. At `nblocks = 106` the row holds 636 bytes of words and the
/// stride is 640, so `f1r_load`'s window on block 105 stays inside the row.
/// The host asserts it below rather than trusting this comment.
pub fn stride_u32(nblocks: usize) -> usize {
    (6 * nblocks).div_ceil(8) * 2
}

/// One matrix, transcoded into the served stream.
pub struct Tetra48Blocks {
    pub d_out: usize,
    pub nblocks: usize,
    /// u32 per row, `>= 6·nblocks/4`, padded so the last block's window fits.
    pub stride_u32: usize,
    /// `d_out · stride_u32 · 4` bytes. Little-endian words, row-strided.
    pub data: Vec<u8>,
}

impl Tetra48Blocks {
    /// Bits the stream spends per weight, **addressing and padding included**.
    ///
    /// Not `48/24 = 2.0`: the pad is real memory the kernel reads over, and a
    /// rate that omits it is the accounting error `docs/METHODE.md` records
    /// for 2026-07-31. At the seven shapes of the 4B the pad costs between 0
    /// and 4 bytes a row.
    pub fn bits_per_weight(&self) -> f64 {
        (self.data.len() as u64 * 8) as f64 / (self.d_out * self.nblocks * DIM) as f64
    }

    /// The 48-bit word of block `j` of row `row`, read back out of the bytes.
    ///
    /// Deliberately **not** through `f1r_load`'s shift arithmetic: this reads
    /// byte `4·row·stride + 6·j` and the five after it, so a mistake in that
    /// arithmetic is a disagreement between two routes and not a shared error.
    /// The card-side dump check uses the same convention (`f1rankfloor.rs`).
    pub fn word(&self, row: usize, j: usize) -> u64 {
        assert!(row < self.d_out && j < self.nblocks, "block {j} of row {row}");
        let at = row * self.stride_u32 * 4 + TETRA48_BYTES * j;
        let mut b = [0u8; 8];
        b[..TETRA48_BYTES].copy_from_slice(&self.data[at..at + TETRA48_BYTES]);
        u64::from_le_bytes(b)
    }

    /// Decode block `j` of row `row` back to its lattice point and gain rank —
    /// the CPU reference every GPU reading of these bytes is checked against,
    /// the same role `RuntimeBlocks::decode_block` plays for the ball layouts.
    pub fn decode_block(&self, tetra: &Tetra, row: usize, j: usize) -> ([i32; DIM], u32) {
        let w = self.word(row, j);
        (tetra.decode(w), ((w >> LABEL_BITS) & 1) as u32)
    }
}

/// Disk `(index, gain)` pairs → the served `tetra48` stream.
///
/// `indices` and `gains` are row-major `d_out × nblocks`, exactly as
/// [`crate::RawMatrix`] carries them.
pub fn transcode_tetra48(
    indices: &[u64],
    gains: &[u32],
    d_out: usize,
    nblocks: usize,
) -> Result<Tetra48Blocks> {
    if indices.len() != gains.len() {
        return Err(Error::Inconsistent {
            name: "tetra48 transcode".to_string(),
            detail: format!("{} indices against {} gains", indices.len(), gains.len()),
        });
    }
    if indices.len() != d_out * nblocks {
        return Err(Error::Inconsistent {
            name: "tetra48 transcode".to_string(),
            detail: format!(
                "{} blocks against d_out {d_out} × nblocks {nblocks} = {}",
                indices.len(),
                d_out * nblocks
            ),
        });
    }

    let stride = stride_u32(nblocks);
    // The window `f1r_load` opens on the last block of a row, in bytes past the
    // row's start: it reads u32 index `(3(n−1)) >> 1` and the one after it.
    if nblocks > 0 {
        let window = 4 * (((3 * (nblocks - 1)) >> 1) + 2);
        assert!(
            window <= stride * 4,
            "nblocks {nblocks}: the last block's two-word window ends at byte {window}, \
             past a row stride of {} bytes",
            stride * 4
        );
    }

    let mut data = vec![0u8; d_out * stride * 4];
    for row in 0..d_out {
        let base = row * stride * 4;
        for j in 0..nblocks {
            let (idx, gain) = (indices[row * nblocks + j], gains[row * nblocks + j]);
            if idx >> LABEL_BITS != 0 {
                return Err(Error::Inconsistent {
                    name: "tetra48 transcode".to_string(),
                    detail: format!("row {row} block {j}: index {idx:#x} exceeds {LABEL_BITS} bits"),
                });
            }
            if gain > 1 {
                return Err(Error::Inconsistent {
                    name: "tetra48 transcode".to_string(),
                    detail: format!("row {row} block {j}: gain {gain} overflows the 1-bit field"),
                });
            }
            // The word, assembled exactly as `CodeMap::decode` assembles it for
            // the reader (`format.rs`, "the pair is put back together as the
            // 48-bit word the kernel reads"). One definition, two consumers.
            let word = idx | (u64::from(gain) << LABEL_BITS);
            debug_assert_eq!(word & !WORD_MASK, 0);
            let at = base + TETRA48_BYTES * j;
            data[at..at + TETRA48_BYTES].copy_from_slice(&word.to_le_bytes()[..TETRA48_BYTES]);
        }
    }
    Ok(Tetra48Blocks { d_out, nblocks, stride_u32: stride, data })
}

#[cfg(test)]
mod tests {
    use super::*;
    use llvq_core::SplitMix64;

    /// The two orders, pinned against the two things they must agree with.
    ///
    /// A transcoder that got either order wrong would produce a stream that
    /// decodes to *different weights* without failing anywhere, which is the
    /// failure mode `pack.rs` states its own byte order to prevent.
    #[test]
    fn the_word_survives_both_conventions() {
        let t = Tetra::new();
        let mut rng = SplitMix64::new(0x7e_47a4_0001);
        let (d_out, nblocks) = (7, 106);
        let n = d_out * nblocks;
        let idx: Vec<u64> = (0..n).map(|_| rng.next() & (WORD_MASK >> 1)).collect();
        let gains: Vec<u32> = (0..n).map(|_| (rng.next() & 1) as u32).collect();

        let out = transcode_tetra48(&idx, &gains, d_out, nblocks).expect("transcodes");
        assert_eq!(out.data.len(), d_out * stride_u32(nblocks) * 4);

        for row in 0..d_out {
            for j in 0..nblocks {
                let k = row * nblocks + j;
                // The word, byte for byte.
                assert_eq!(
                    out.word(row, j),
                    idx[k] | u64::from(gains[k]) << LABEL_BITS,
                    "row {row} block {j}"
                );
                // And what it decodes to, against the production map — the
                // point the served kernel must reproduce.
                let (p, g) = out.decode_block(&t, row, j);
                assert_eq!(p, t.decode(idx[k]), "row {row} block {j}: the point");
                assert_eq!(g, gains[k], "row {row} block {j}: the gain");
            }
        }
    }

    /// The row stride covers the last block's two-word read window, at every
    /// shape of the published 4B and at the awkward sizes around them.
    #[test]
    fn every_row_stride_covers_the_last_window() {
        // The seven shapes of Qwen3-4B carry d_in ∈ {2560, 4096, 9728}.
        for &d_in in &[2560usize, 4096, 9728] {
            let nblocks = d_in / DIM;
            let stride = stride_u32(nblocks);
            let window = 4 * (((3 * (nblocks - 1)) >> 1) + 2);
            assert!(window <= stride * 4, "d_in {d_in}: {window} past {}", stride * 4);
        }
        // And the parities around them: an odd block count shifts the phase.
        for nblocks in 1..200usize {
            let stride = stride_u32(nblocks);
            let window = 4 * (((3 * (nblocks - 1)) >> 1) + 2);
            assert!(window <= stride * 4, "nblocks {nblocks}: {window} past {}", stride * 4);
            assert!(stride * 4 >= 6 * nblocks, "nblocks {nblocks}: stride below the payload");
        }
    }

    /// The origin is a legal code and it must survive the round trip as one.
    #[test]
    fn word_zero_is_carried_and_not_treated_as_absent() {
        let t = Tetra::new();
        let out = transcode_tetra48(&[0, 0], &[0, 1], 1, 2).expect("transcodes");
        assert_eq!(out.word(0, 0), 0);
        assert_eq!(out.word(0, 1), 1 << LABEL_BITS, "the gain bit alone");
        assert_eq!(out.decode_block(&t, 0, 0), ([0; DIM], 0));
        assert_eq!(out.decode_block(&t, 0, 1).1, 1);
    }

    /// A gain that does not fit its one bit is refused, not truncated: a
    /// truncated gain is a valid gain for a different block.
    #[test]
    fn an_overflowing_field_is_refused() {
        assert!(transcode_tetra48(&[0], &[2], 1, 1).is_err(), "gain 2");
        assert!(transcode_tetra48(&[1 << LABEL_BITS], &[0], 1, 1).is_err(), "index past 47 bits");
        assert!(transcode_tetra48(&[0, 0], &[0], 2, 1).is_err(), "one gain for two indices");
        assert!(transcode_tetra48(&[0], &[0], 2, 3).is_err(), "count against the shape");
    }

    /// The rate includes the pad, because the kernel reads over it.
    #[test]
    fn the_rate_counts_the_padding() {
        // 106 blocks: 636 bytes of words, 640 of stride. 4 bytes a row.
        let out = transcode_tetra48(&vec![0; 106], &vec![0; 106], 1, 106).expect("transcodes");
        assert_eq!(out.stride_u32, 160);
        let exact = 48.0 / DIM as f64;
        assert!(out.bits_per_weight() > exact, "the pad has to show");
        assert!(out.bits_per_weight() < exact * 1.02, "and it is small: {}", out.bits_per_weight());
    }
}
