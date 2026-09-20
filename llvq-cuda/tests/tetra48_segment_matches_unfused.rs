//! The row-concatenation behind `tv_tetra48_seg`, proved on the development
//! machine — the only half of the Tetra fusion that can be.
//!
//! ## Why this file exists, and what it cost not to have it
//!
//! `tests/planes_segment_matches_unfused.rs` proves the host side of the ball
//! fusion. Nothing proved the Tetra one, and on 2026-09-20 the bench's own
//! bit-exact comparison caught the kernel at the first group it could check:
//!
//! ```text
//! 000.gateup / model.layers.0.mlp.gate_proj.weight / Tetra48:
//!   row 0 is -0.3080684 fused against -0.57255346 separate
//! ```
//!
//! Slot32 and Planes14 passed on that same group, in the same run, which puts
//! the row order, `gs_off`, `rscale` and `tail` above suspicion: those are
//! shared. What is left is the part that is Tetra's alone — the row-strided
//! word stream — and this file is where that can be tested without a card and
//! without $0.85 a try.
//!
//! ## What this proves, and what it deliberately does not
//!
//! `kernels/tetra48_seg.cu` cannot be validated here. A single-threaded driver
//! reproduces neither `__syncthreads()` nor a warp shuffle, so `host_tetra48.cpp`
//! compiles it and stops there.
//!
//! What is provable here is that the **stream a fused launch reads is the same
//! stream the unfused launches read, block for block**. The fusion changes no
//! arithmetic: the same blocks, in the same order, with the same centroids. So
//! every risk is bookkeeping, and a bookkeeping error produces plausible
//! numbers — which is why both this file and the card demand equality rather
//! than a tolerance.

use llvq_artifact::tetra48::{stride_u32, transcode_tetra48};
use llvq_core::{SplitMix64, DIM};
use llvq_search::index::N13;
use llvq_search::tetra::Tetra;

include!("../src/seg_host.rs");

/// A part's own arrays, owned — `SegPart` borrows, and the fixture has to own
/// them somewhere.
struct Part {
    d_out: usize,
    indices: Vec<u64>,
    gains: Vec<u32>,
    centroids: [f32; 2],
    rscale: Vec<f32>,
    tail: Vec<f32>,
}

/// Two segments of a plausible gate/up group, which is the group that failed
/// on the card: equal rows, one shared `d_in` with a tail (`128 = 24·5 + 8`),
/// and two **distinct** centroid pairs, without which every `gs_off` would be
/// as good as every other.
fn fixture(rng: &mut SplitMix64) -> (Vec<Part>, usize) {
    let d_in = 128;
    let nblocks = d_in / DIM;
    let tail_w = d_in % DIM;
    let centroids = [[0.625f32, 1.375], [0.5, 2.0]];
    let parts = [16usize, 16]
        .iter()
        .zip(centroids)
        .map(|(&d_out, centroids)| Part {
            d_out,
            // A sixth of the blocks are the origin. It is the one class whose
            // magnitude is zero, so a stride or phase error trips over it
            // first: a shifted read turns a zero block into a non-zero one.
            //
            // Capped at 47 bits, which the ball fixture does not need to be:
            // `N13` is 1.96e14 and a Tetra word holds 47 bits of label plus one
            // of gain, so the top of the ball index space has no Tetra word and
            // `transcode_tetra48` refuses it by name. The cap is a property of
            // the FIXTURE, not of the concatenation under test.
            indices: (0..d_out * nblocks)
                .map(|_| {
                    match rng.next().is_multiple_of(6) {
                        true => 0,
                        false => 1 + rng.next() % (N13.min(1u64 << 47) - 1),
                    }
                })
                .collect(),
            gains: (0..d_out * nblocks).map(|_| (rng.next() & 1) as u32).collect(),
            centroids,
            rscale: (0..d_out).map(|_| 0.5 + rng.next_gaussian().abs() as f32).collect(),
            tail: (0..d_out * tail_w).map(|_| rng.next_gaussian() as f32).collect(),
        })
        .collect();
    (parts, d_in)
}

fn borrow(parts: &[Part], d_in: usize) -> Vec<SegPart<'_>> {
    parts
        .iter()
        .map(|p| SegPart {
            d_out: p.d_out,
            d_in,
            indices: &p.indices,
            gains: &p.gains,
            centroids: p.centroids,
            rscale: &p.rscale,
            tail: &p.tail,
        })
        .collect()
}

/// The claim the kernel rests on: row `at + r` of the fused stream carries the
/// same 48-bit word as row `r` of the segment's own stream, for every block.
///
/// If this fails, the defect is host-side and the kernel is innocent. If it
/// passes, the defect is in the kernel and a card is needed to find it — but
/// the search is halved either way.
#[test]
fn a_fused_row_carries_the_same_words_as_its_unfused_row() {
    let mut rng = SplitMix64::new(0x7E_47A4);
    let (parts, d_in) = fixture(&mut rng);
    let sp = borrow(&parts, d_in);
    let seg = seg_concat(&sp);
    let nblocks = d_in / DIM;

    let fused = transcode_tetra48(&seg.indices, &seg.gains, seg.d_out, nblocks)
        .expect("the fused stream transcodes");

    let mut at = 0usize;
    for p in &parts {
        let own = transcode_tetra48(&p.indices, &p.gains, p.d_out, nblocks)
            .expect("the segment's own stream transcodes");
        // The strides must agree BEFORE any word is compared: they depend only
        // on `nblocks`, so a difference here would mean the two streams do not
        // even describe the same geometry.
        assert_eq!(
            own.stride_u32, fused.stride_u32,
            "the strides differ, {} against {}",
            own.stride_u32, fused.stride_u32
        );
        assert_eq!(own.stride_u32, stride_u32(nblocks));

        for r in 0..p.d_out {
            for j in 0..nblocks {
                assert_eq!(
                    fused.word(at + r, j),
                    own.word(r, j),
                    "row {r} of the segment at offset {at}, block {j}"
                );
            }
        }
        at += p.d_out;
    }
    assert_eq!(at, seg.d_out, "the segments do not cover the fused rows");
}

/// The same claim one level down: the decoded point and gain, not the packed
/// word. A word that matched while its decode did not would mean the stream is
/// right and the table read is not.
#[test]
fn a_fused_row_decodes_to_the_same_points() {
    let tetra = Tetra::new();
    let mut rng = SplitMix64::new(0x7E_47A5);
    let (parts, d_in) = fixture(&mut rng);
    let sp = borrow(&parts, d_in);
    let seg = seg_concat(&sp);
    let nblocks = d_in / DIM;
    let fused = transcode_tetra48(&seg.indices, &seg.gains, seg.d_out, nblocks).expect("fused");

    let mut at = 0usize;
    for p in &parts {
        let own = transcode_tetra48(&p.indices, &p.gains, p.d_out, nblocks).expect("own");
        for r in 0..p.d_out {
            for j in 0..nblocks {
                assert_eq!(
                    fused.decode_block(&tetra, at + r, j),
                    own.decode_block(&tetra, r, j),
                    "row {r} at offset {at}, block {j}"
                );
            }
        }
        at += p.d_out;
    }
}

/// `gs_off[row]` must name that row's OWN centroid pair in the concatenated
/// `centroids`. This is what a wrong answer would hide behind: the two pairs
/// differ by a factor of about two, so a swap moves some rows a lot and leaves
/// the rest untouched.
#[test]
fn every_row_gets_its_own_centroid_pair() {
    let mut rng = SplitMix64::new(0x7E_47A6);
    let (parts, d_in) = fixture(&mut rng);
    let sp = borrow(&parts, d_in);
    let seg = seg_concat(&sp);

    let mut at = 0usize;
    for (s, p) in parts.iter().enumerate() {
        for r in 0..p.d_out {
            let o = seg.gs_off[at + r] as usize;
            assert_eq!(o, 2 * s, "row {r} of segment {s} points at pair {}", o / 2);
            assert_eq!(seg.centroids[o], p.centroids[0]);
            assert_eq!(seg.centroids[o + 1], p.centroids[1]);
        }
        at += p.d_out;
    }
}

/// `rscale` and `tail` are indexed by row and must line up past the join. A
/// constant shift there produces plausible numbers on every row, which is the
/// failure this whole file is shaped against.
#[test]
fn rscale_and_tail_line_up_past_the_join() {
    let mut rng = SplitMix64::new(0x7E_47A7);
    let (parts, d_in) = fixture(&mut rng);
    let tail_w = d_in % DIM;
    let sp = borrow(&parts, d_in);
    let seg = seg_concat(&sp);
    assert_eq!(seg.tail_w, tail_w);

    let mut at = 0usize;
    for p in &parts {
        for r in 0..p.d_out {
            assert_eq!(seg.rscale[at + r], p.rscale[r], "rscale at offset {at}, row {r}");
            for i in 0..tail_w {
                assert_eq!(
                    seg.tail[(at + r) * tail_w + i],
                    p.tail[r * tail_w + i],
                    "tail at offset {at}, row {r}, column {i}"
                );
            }
        }
        at += p.d_out;
    }
}
