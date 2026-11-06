//! The row-concatenation behind `tv_tetra48_seg`, proved on the development
//! machine: the only half of the Tetra fusion that can be.
//!
//! ## The 2026-09-20 failure, and its cause
//!
//! On 2026-09-20 the bench's bit-exact comparison failed at the first Tetra
//! group it could check:
//!
//! ```text
//! 000.gateup / model.layers.0.mlp.gate_proj.weight / Tetra48:
//!   row 0 is -0.3080684 fused against -0.57255346 separate
//! ```
//!
//! The kernel was not the cause. `bin/planesbench.rs` reads two files. The
//! separate Tetra launches read the second file, the Tetra one. The fused
//! Tetra stream was transcoded from the first file, the ball one: its indices,
//! its centroids, its row scales, its tail. The check compared two different
//! matrices. Found by reading the code on 2026-11-04.
//!
//! The first version of this file blamed the kernel's row-strided word stream.
//! That was wrong: its tests concatenated one source and compared it with
//! itself, so they could not see a second file. The last two tests below build
//! the group from two different sources, and fail on the old construction.
//!
//! ## What this proves, and what it deliberately does not
//!
//! `kernels/tetra48_seg.cu` cannot be validated here. A single-threaded driver
//! reproduces neither `__syncthreads()` nor a warp shuffle, so `host_tetra48.cpp`
//! compiles it and stops there. A card run is still needed for the kernel.
//!
//! What is provable here is that the **stream a fused launch reads is the same
//! stream the unfused launches read, block for block**, with the same
//! centroids, row scales and tail. The fusion changes no arithmetic. So every
//! risk is bookkeeping, and a bookkeeping error produces plausible numbers.
//! That is why both this file and the card demand equality, not a tolerance.

use llvq_artifact::tetra48::{stride_u32, transcode_tetra48};
use llvq_core::{SplitMix64, DIM};
use llvq_search::index::N13;
use llvq_search::tetra::{Tetra, LABEL_BITS};

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
/// One source only: it proves the stacking, not which file the stack is read
/// from. That second claim is the last two tests' job.
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

/// A second source for the same group, standing for the Tetra file: same
/// shapes as `fixture`, but other indices, other centroids, other row scales
/// and other tails. Transcoded at once, as the bench transcodes the Tetra file
/// when it reads it.
struct TetraPart {
    d_out: usize,
    blocks: llvq_artifact::tetra48::Tetra48Blocks,
    centroids: [f32; 2],
    rscale: Vec<f32>,
    tail: Vec<f32>,
}

fn tetra_fixture(rng: &mut SplitMix64, ball: &[Part], d_in: usize) -> Vec<TetraPart> {
    let nblocks = d_in / DIM;
    let tail_w = d_in % DIM;
    // Distinct from both ball pairs, and from each other.
    let centroids = [[0.75f32, 1.5], [0.4375, 1.8125]];
    ball.iter()
        .zip(centroids)
        .map(|(b, centroids)| {
            let n = b.d_out * nblocks;
            let indices: Vec<u64> = (0..n)
                .map(|_| match rng.next().is_multiple_of(6) {
                    true => 0,
                    false => 1 + rng.next() % ((1u64 << LABEL_BITS) - 1),
                })
                .collect();
            let gains: Vec<u32> = (0..n).map(|_| (rng.next() & 1) as u32).collect();
            TetraPart {
                d_out: b.d_out,
                blocks: transcode_tetra48(&indices, &gains, b.d_out, nblocks)
                    .expect("the Tetra source transcodes"),
                centroids,
                rscale: (0..b.d_out).map(|_| 0.5 + rng.next_gaussian().abs() as f32).collect(),
                tail: (0..b.d_out * tail_w).map(|_| rng.next_gaussian() as f32).collect(),
            }
        })
        .collect()
}

fn borrow_tetra(parts: &[TetraPart], d_in: usize) -> Vec<SegTetraPart<'_>> {
    parts
        .iter()
        .map(|p| SegTetraPart {
            d_out: p.d_out,
            d_in,
            blocks: &p.blocks,
            centroids: p.centroids,
            rscale: &p.rscale,
            tail: &p.tail,
        })
        .collect()
}

/// Block `j` of row `row` in a u32 row-strided stream, read byte by byte like
/// `Tetra48Blocks::word`, so the fused words are read by the same route as the
/// separate ones.
fn word_u32(words: &[u32], stride_u32: usize, row: usize, j: usize) -> u64 {
    let at = row * stride_u32 * 4 + 6 * j;
    let mut b = [0u8; 8];
    for (k, out) in b.iter_mut().take(6).enumerate() {
        let byte = at + k;
        *out = words[byte / 4].to_le_bytes()[byte % 4];
    }
    u64::from_le_bytes(b)
}

/// One output row, as the kernel forms it: every block's point scaled by its
/// gain centroid over its norm, dotted with `x`, times the row scale, plus the
/// tail. In f64 and in a fixed order, so equal inputs give equal bits.
fn row_out(
    tetra: &Tetra,
    word: impl Fn(usize) -> u64,
    nblocks: usize,
    pair: [f32; 2],
    rscale: f32,
    tail: &[f32],
    x: &[f32],
) -> f64 {
    let mut acc = 0.0f64;
    for j in 0..nblocks {
        let w = word(j);
        let pt = tetra.decode(w);
        let gain = ((w >> LABEL_BITS) & 1) as usize;
        let n2: i64 = pt.iter().map(|&v| (v as i64) * (v as i64)).sum();
        if n2 == 0 {
            continue;
        }
        let k = pair[gain] as f64 / (n2 as f64).sqrt();
        for (i, &v) in pt.iter().enumerate() {
            acc += v as f64 * k * x[j * DIM + i] as f64;
        }
    }
    let tv: f64 =
        tail.iter().enumerate().map(|(i, &t)| t as f64 * x[nblocks * DIM + i] as f64).sum();
    acc * rscale as f64 + tv
}

/// The check the card runs, on the host: build the fused Tetra group the way
/// the bench does, from a ball file and a Tetra file that DIFFER, and demand
/// that every fused row equals the same row of the Tetra file's own matrix.
///
/// The old construction read the ball file and fails here at row 0 of the
/// first segment, which is exactly where the card failed.
#[test]
fn the_fused_tetra_group_reads_the_tetra_file_not_the_ball_file() {
    let tetra = Tetra::new();
    let mut rng = SplitMix64::new(0x7E_47A8);
    let (ball, d_in) = fixture(&mut rng);
    let tparts = tetra_fixture(&mut rng, &ball, d_in);
    let nblocks = d_in / DIM;
    let tail_w = d_in % DIM;
    let x: Vec<f32> = (0..d_in).map(|_| rng.next_gaussian() as f32).collect();

    let seg = seg_concat(&borrow(&ball, d_in));
    let ts = seg_tetra48_group("000.gateup", &seg, &borrow_tetra(&tparts, d_in));
    assert_eq!(ts.d_out, seg.d_out);
    assert_eq!(ts.words.len(), ts.d_out * ts.stride_u32);

    let mut at = 0usize;
    for (s, p) in tparts.iter().enumerate() {
        for r in 0..p.d_out {
            let row = at + r;
            let o = ts.gs_off[row] as usize;
            let got = row_out(
                &tetra,
                |j| word_u32(&ts.words, ts.stride_u32, row, j),
                nblocks,
                [ts.centroids[o], ts.centroids[o + 1]],
                ts.rscale[row],
                &ts.tail[row * tail_w..(row + 1) * tail_w],
                &x,
            );
            let want = row_out(
                &tetra,
                |j| p.blocks.word(r, j),
                nblocks,
                p.centroids,
                p.rscale[r],
                &p.tail[r * tail_w..(r + 1) * tail_w],
                &x,
            );
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "segment {s}: row {r} is {got} fused against {want} separate"
            );
        }
        at += p.d_out;
    }
}

/// The same claim on the arrays, so a failure names the field: every fused
/// row carries the Tetra file's words, centroid pair, row scale and tail.
#[test]
fn every_fused_tetra_array_comes_from_the_tetra_file() {
    let mut rng = SplitMix64::new(0x7E_47A9);
    let (ball, d_in) = fixture(&mut rng);
    let tparts = tetra_fixture(&mut rng, &ball, d_in);
    let nblocks = d_in / DIM;
    let tail_w = d_in % DIM;

    let seg = seg_concat(&borrow(&ball, d_in));
    let ts = seg_tetra48_group("000.gateup", &seg, &borrow_tetra(&tparts, d_in));
    assert_eq!(ts.stride_u32, stride_u32(nblocks));
    assert_eq!(ts.row_offsets, seg.row_offsets);

    let mut at = 0usize;
    for (s, p) in tparts.iter().enumerate() {
        for r in 0..p.d_out {
            let row = at + r;
            for j in 0..nblocks {
                assert_eq!(
                    word_u32(&ts.words, ts.stride_u32, row, j),
                    p.blocks.word(r, j),
                    "segment {s}: word of row {r}, block {j}"
                );
            }
            let o = ts.gs_off[row] as usize;
            assert_eq!(o, 2 * s, "segment {s}: row {r} points at pair {}", o / 2);
            assert_eq!([ts.centroids[o], ts.centroids[o + 1]], p.centroids, "segment {s}: centroids");
            assert_eq!(ts.rscale[row], p.rscale[r], "segment {s}: rscale of row {r}");
            assert_eq!(
                &ts.tail[row * tail_w..(row + 1) * tail_w],
                &p.tail[r * tail_w..(r + 1) * tail_w],
                "segment {s}: tail of row {r}"
            );
        }
        at += p.d_out;
    }
}
