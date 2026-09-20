//! The Tetra decoder in MSL, against the Rust one, exactly.
//!
//! ## The gate this file is
//!
//! `llvq-llm/kernels/llvq_tetra48.metal` is a port of three CUDA headers. A
//! port of a lattice decoder is the kind of code where a mistake returns
//! finite, plausible, wrong numbers: a shifted table read moves some
//! coordinates and leaves the rest alone, and the result still has the right
//! shape and roughly the right norm.
//!
//! So this file demands EQUALITY, coordinate by coordinate, against
//! `llvq_search::tetra::Tetra`, which is the definition. Not a tolerance. The
//! decoded point is a vector of small integers, and the MSL returns them as
//! f32, so every value is exact on both sides and there is nothing to round.
//!
//! ## What it does not cover
//!
//! The matvec, the tile, the reduction and the scaling. This is the decoder
//! alone, one thread a block, no threadgroup memory and no `simd_sum`. That is
//! deliberate: it is the half that can be wrong silently, so it is judged
//! first and on its own.
//!
//! Runs on any Mac, needs no model and no artifact.

#![cfg(target_os = "macos")]

use llvq_artifact::tetra48::{stride_u32, transcode_tetra48};
use llvq_bench::f1::rank::{branch_words, prefix_bytes, suffix_bytes, RankTable};
use llvq_bench::f1::Trellis;
use llvq_core::{SplitMix64, DIM};
use llvq_metal::Kernel;
use llvq_search::index::N13;
use llvq_search::tetra::Tetra;

const SOURCE: &str = include_str!("../../llvq-llm/kernels/llvq_tetra48.metal");

/// Rows and blocks of the fixture.
///
/// 64 rows of 12 blocks is 768 words, which draws every table path many times
/// over while staying instant. `d_in = 12 * 24 = 288`.
const ROWS: usize = 64;
const NBLOCKS: usize = 12;

/// Indices capped at 47 bits, which is a property of the FIXTURE.
///
/// `N13` is 1.96e14 and a Tetra word holds 47 bits of label plus one of gain,
/// so the top of the ball index space has no Tetra word and
/// `transcode_tetra48` refuses it by name.
fn draw(rng: &mut SplitMix64) -> (Vec<u64>, Vec<u32>) {
    let n = ROWS * NBLOCKS;
    let indices = (0..n)
        .map(|_| match rng.next().is_multiple_of(6) {
            // A sixth are the origin, the one class whose magnitude is zero.
            // A shifted read turns a zero block into a non-zero one, so this
            // is the cheapest tripwire the fixture can carry.
            true => 0,
            false => 1 + rng.next() % (N13.min(1u64 << 47) - 1),
        })
        .collect();
    let gains = (0..n).map(|_| (rng.next() & 1) as u32).collect();
    (indices, gains)
}

/// The shell index the kernel derives, recomputed from the point.
///
/// `(|x|^2 >> 4) & 31`, which is what `tetra48_dot` indexes `invnorm` with.
fn shell_of(point: &[i32; DIM]) -> u32 {
    let n2: i32 = point.iter().map(|&v| v * v).sum();
    ((n2 as u32) >> 4) & 31
}

struct Decoded {
    points: Vec<f32>,
    shells: Vec<u32>,
}

/// Compile the MSL, upload the tables and the words, decode every block.
fn decode_on_metal(data: &[u8], stride: usize, nblocks: usize, rows: usize) -> Decoded {
    let k = Kernel::new(SOURCE, "tetra48_probe").expect("the MSL compiles and carries the probe");
    let table = RankTable::build();
    let tr = Trellis::new();

    // `data` is the packed byte stream; the kernel reads it as u32, which is what
    // `stride_u32` counts. A Metal buffer is page aligned, so the view is legal.
    assert!(data.len().is_multiple_of(4), "the stream must be a whole number of words");
    let b_words = k.buffer(data);
    let b_rows = k.buffer(&table.rows);
    let b_pref = k.buffer(&prefix_bytes(&tr));
    let b_bran = k.buffer(&branch_words(&tr));
    let b_suff = k.buffer(&suffix_bytes(&tr));

    let n = rows * nblocks;
    let b_out = k.empty::<f32>(n * DIM);
    let b_shell = k.empty::<u32>(n * 2);
    let stride_u32 = stride as u32;
    let nb = nblocks as u32;

    k.dispatch(n as u64, 64, |enc| {
        enc.set_buffer(0, Some(&b_words), 0);
        enc.set_buffer(1, Some(&b_rows), 0);
        enc.set_buffer(2, Some(&b_pref), 0);
        enc.set_buffer(3, Some(&b_bran), 0);
        enc.set_buffer(4, Some(&b_suff), 0);
        enc.set_buffer(5, Some(&b_out), 0);
        enc.set_buffer(6, Some(&b_shell), 0);
        enc.set_bytes(7, 4, &stride_u32 as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(8, 4, &nb as *const u32 as *const std::ffi::c_void);
    });

    let points = unsafe { std::slice::from_raw_parts(b_out.contents() as *const f32, n * DIM) };
    let shells = unsafe { std::slice::from_raw_parts(b_shell.contents() as *const u32, n * 2) };
    Decoded {
        points: points.to_vec(),
        shells: shells.to_vec(),
    }
}

/// The gate. Every coordinate of every block, exactly.
#[test]
fn the_msl_decoder_returns_the_rust_lattice_point() {
    let mut rng = SplitMix64::new(0x7E_47B0);
    let (indices, gains) = draw(&mut rng);
    let stream =
        transcode_tetra48(&indices, &gains, ROWS, NBLOCKS).expect("the stream transcodes");
    assert_eq!(stream.stride_u32, stride_u32(NBLOCKS));

    let got = decode_on_metal(&stream.data, stream.stride_u32, NBLOCKS, ROWS);
    let tetra = Tetra::new();

    let mut nonzero = 0usize;
    for row in 0..ROWS {
        for j in 0..NBLOCKS {
            let (want, want_gain) = stream.decode_block(&tetra, row, j);
            let at = row * NBLOCKS + j;
            for (i, &w) in want.iter().enumerate() {
                let g = got.points[at * DIM + i];
                assert_eq!(
                    g,
                    w as f32,
                    "row {row}, block {j}, coordinate {i}: metal {g} against rust {w}"
                );
            }
            assert_eq!(
                got.shells[at * 2],
                shell_of(&want),
                "row {row}, block {j}: shell index"
            );
            assert_eq!(
                got.shells[at * 2 + 1],
                want_gain,
                "row {row}, block {j}: gain bit"
            );
            // Why a mutation of `tetra48_n2` survives, stated as an
            // assertion instead of as an excuse.
            //
            // Dropping the `^ 0x80808080` debias turns each squared term from
            // `v^2` into `(|v| - 128)^2`, so the sum moves by
            // `-256 * sum|v| + 24 * 16384`. After `>> 4` and `& 31` the second
            // term vanishes and the first leaves `16 * (sum|v| mod 2)`. On
            // this lattice every coordinate vector has an EVEN `sum|v|`, so
            // the shell index does not move and the mutant is equivalent for
            // the one thing `n2` is read for. That is an accident of Lambda_24,
            // not a property of the code, so it is pinned here: if it ever
            // stops holding, this fails before the kernel does.
            let l1: i32 = want.iter().map(|v| v.abs()).sum();
            assert_eq!(l1 & 1, 0, "row {row}, block {j}: sum|v| = {l1} is odd");
            if want.iter().any(|&v| v != 0) {
                nonzero += 1;
            }
        }
    }
    // A decoder that returned zeros everywhere would pass every equality above
    // if the fixture were degenerate. It is not, and this says so.
    assert!(
        nonzero > ROWS * NBLOCKS / 2,
        "the fixture must exercise non-zero points, got {nonzero}"
    );
}

/// A second seed, so the first is not the one that happens to work.
#[test]
fn the_msl_decoder_agrees_on_a_second_draw() {
    let mut rng = SplitMix64::new(0x7E_47B1);
    let (indices, gains) = draw(&mut rng);
    let stream = transcode_tetra48(&indices, &gains, ROWS, NBLOCKS).expect("transcodes");
    let got = decode_on_metal(&stream.data, stream.stride_u32, NBLOCKS, ROWS);
    let tetra = Tetra::new();

    for row in 0..ROWS {
        for j in 0..NBLOCKS {
            let (want, _) = stream.decode_block(&tetra, row, j);
            let at = row * NBLOCKS + j;
            for (i, &w) in want.iter().enumerate() {
                assert_eq!(got.points[at * DIM + i], w as f32, "row {row}, block {j}, coord {i}");
            }
        }
    }
}

/// The origin decodes to the origin, on both sides.
///
/// Its own test because it is the one class whose magnitude is zero, and
/// because a stride or phase error shows there first.
#[test]
fn the_origin_block_decodes_to_zero() {
    let n = ROWS * NBLOCKS;
    let indices = vec![0u64; n];
    let gains = vec![0u32; n];
    let stream = transcode_tetra48(&indices, &gains, ROWS, NBLOCKS).expect("transcodes");
    let got = decode_on_metal(&stream.data, stream.stride_u32, NBLOCKS, ROWS);
    for (at, v) in got.points.iter().enumerate() {
        assert_eq!(*v, 0.0, "coordinate {at} of an all-origin stream");
    }
    for at in 0..n {
        assert_eq!(got.shells[at * 2], 0, "the origin sits on shell 0");
    }
}
