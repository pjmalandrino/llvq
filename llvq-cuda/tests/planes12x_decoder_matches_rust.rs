//! The Planes12x CUDA decoder and its overlay correction, compiled by
//! `clang++` and diffed against Rust — the same two-lock pattern as
//! `planes_decoder_matches_rust.rs`:
//!
//! 1. **The overlay transcoding is exact, in pure Rust.** Every block of a
//!    synthetic matrix must decode through `Planes12xBlocks::decode_block`
//!    to exactly what the Slot32 stream carries; the main stream alone
//!    (`decode_approx_block`) must differ on precisely the exception blocks.
//!
//! 2. **The kernel text decides what Rust decides.** `planes12_fields`,
//!    `planes12_dot`, `planes12x_locate`, `planes12x_lane_terms` and
//!    `planes12x_combine` — the same text NVRTC will compile, through the
//!    same shim — are executed per block and per exception, and the full
//!    `y` they produce (row pass + corrections, the emulated launch) is
//!    checked row by row against an f64 reference built from the **exact**
//!    archive decode. That check is the host-side twin of the bench's
//!    line-by-line verification: it can only pass if the correction pass
//!    cancels the main stream's approximation exactly. Including planes12.cu
//!    also compiles `tv_planes12x`, so a syntax or type error costs two
//!    seconds here instead of a billed job.
//!
//! The fixture is a 32-row matrix whose blocks mix both ends of every class,
//! uniform draws over the cap-13 ball and the origin — so there are rows
//! with several exceptions, rows with none, and 5-level classes on shell 13
//! exercising the p2 plane of the exception records.
//!
//! What is left for the card: occupancy, the atomicity of atomicAdd under a
//! real schedule, the memset-before-launch protocol, and registers/spill
//! (`local_bytes == 0` is read at bench startup).

use llvq_artifact::runtime::{
    transcode, transcode_planes12x, ClassTable, Layout, PLANES14_BYTES,
};
use llvq_core::{SplitMix64, DIM};
use llvq_search::fastdec::FastDecoder;
use llvq_search::index::{Indexer, N13};
use llvq_search::Searcher;
use std::io::Write;
use std::process::{Command, Stdio};

const GSCALE: [f32; 2] = [0.625, 1.375];
const TABLE_ENTRIES: usize = 512;
const REC_WORDS: usize = 6;
const D_OUT: usize = 32;
const NBLOCKS: usize = 45;
const TOL: f64 = 1e-5;

/// The synthetic matrix: uniform draws over the cap-13 ball, with both ends
/// of every class and the origin scattered through the stream at a fixed
/// cadence, so every class id and every level count crosses every row shape.
fn fixture_indices(fd: &FastDecoder, rng: &mut SplitMix64) -> Vec<u64> {
    let mut pool: Vec<u64> = Vec::new();
    for ci in 0..fd.n_classes() {
        let (first, last) = fd.class_range(ci);
        pool.push(first);
        pool.push(last);
    }
    pool.push(0);
    let mut indices: Vec<u64> = (0..D_OUT * NBLOCKS).map(|_| 1 + rng.next() % N13).collect();
    let n = indices.len();
    for (k, &p) in pool.iter().enumerate() {
        // Stride 1 past a divisor of the row length, so the pool wraps
        // through different rows and columns rather than one column.
        indices[(k * (NBLOCKS + 1)) % n] = p;
    }
    indices
}

fn le_u32(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn le_f32(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn pad_words(mut bytes: Vec<u8>) -> Vec<u32> {
    // The +4 tail padding of the frozen spec, then word packing.
    bytes.extend_from_slice(&[0u8; 4]);
    while !bytes.len().is_multiple_of(4) {
        bytes.push(0);
    }
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[test]
#[cfg_attr(debug_assertions, ignore = "searches every L = 5 block, run in release")]
fn planes12x_overlay_is_exact_against_slot32() {
    let fd = FastDecoder::new();
    let s = Searcher::new();
    let table = ClassTable::new(&fd, 1);
    let mut rng = SplitMix64::new(0x12_C0DE);
    let indices = fixture_indices(&fd, &mut rng);
    let gains: Vec<u32> = indices.iter().map(|_| (rng.next() & 1) as u32).collect();

    let p12 = transcode_planes12x(&fd, &table, &s, &indices, &gains).expect("transcodes");
    let slot = transcode(&fd, &table, &indices, &gains, Layout::Slot32).expect("transcodes");
    assert!(
        p12.n_exceptions() > 0,
        "fixture must exercise the overlay (no 5-level block drawn?)"
    );
    for b in 0..indices.len() {
        let want = slot.decode_block(&table, b);
        let got = p12.decode_block(&table, b);
        assert_eq!(got, want, "block {b}: overlay disagrees with Slot32");
        let approx = p12.decode_approx_block(&table, b);
        let is_exc = p12.exc_idx.binary_search(&(b as u32)).is_ok();
        assert_eq!(
            approx != want,
            is_exc,
            "block {b}: main stream must differ exactly on exceptions"
        );
    }
}

#[test]
#[cfg_attr(debug_assertions, ignore = "compiles C++ and searches, run in release")]
fn the_planes12x_kernel_decides_what_the_rust_decoder_decides() {
    let out = std::env::temp_dir().join("llvq_host_planes12");
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests");
    let st = Command::new("clang++")
        .args([
            "-std=c++17",
            "-O2",
            // Contraction off, every load-bearing multiply-add an explicit
            // `__fmaf_rn` → `std::fma`: same reasoning as the other probes,
            // so a float difference is attributable.
            "-ffp-contract=off",
            "-Wall",
            "-Wextra",
            "-Werror",
        ])
        .arg(dir.join("host_planes12.cpp"))
        .arg("-o")
        .arg(&out)
        .status()
        .expect("clang++ is on PATH");
    assert!(st.success(), "the Planes12x CUDA sources do not compile as host C++");

    // ---- fixture ----
    let fd = FastDecoder::new();
    let ix = Indexer::new();
    let s = Searcher::new();
    let table = ClassTable::new(&fd, 1);
    let mut rng = SplitMix64::new(0x12_C0DE);
    let indices = fixture_indices(&fd, &mut rng);
    let gains: Vec<u32> = indices.iter().map(|_| (rng.next() & 1) as u32).collect();
    let n = indices.len();

    let p12 = transcode_planes12x(&fd, &table, &s, &indices, &gains).expect("transcodes");
    let n_exc = p12.n_exceptions();
    assert!(n_exc > 0, "fixture must exercise the correction pass");
    // At least one row must carry several exceptions: the accumulation
    // (vs a store) is only observable then and on rows with row-pass mass.
    let rows_of_exc: Vec<u32> = p12.exc_idx.iter().map(|&b| b / NBLOCKS as u32).collect();
    assert!(
        rows_of_exc.windows(2).any(|w| w[0] == w[1]),
        "fixture must put two exceptions on one row"
    );

    let words = pad_words(p12.data.clone());
    let exc_words = pad_words(p12.exc_data.clone());

    let mut tab = vec![0u32; TABLE_ENTRIES * REC_WORDS];
    for e in 0..TABLE_ENTRIES {
        tab[e * REC_WORDS + REC_WORDS - 1] = 1;
    }
    for ci in 0..fd.n_classes() {
        let lv = fd.levels(ci);
        let norm = ((16 * lv.shell) as f64).sqrt();
        let base = (1 + ci) * REC_WORDS;
        for k in 0..lv.len {
            tab[base + k] = ((lv.values[k] as f64 / norm) as f32).to_bits();
        }
        tab[base + REC_WORDS - 1] = lv.len as u32;
    }

    let rscale: Vec<f32> = (0..D_OUT)
        .map(|_| 0.5 + rng.next_gaussian().abs() as f32)
        .collect();
    let x: Vec<f32> = (0..NBLOCKS * DIM).map(|_| rng.next_gaussian() as f32).collect();

    let mut fixture = Vec::new();
    fixture.extend_from_slice(&le_u32(&[
        D_OUT as u32,
        NBLOCKS as u32,
        n_exc as u32,
        words.len() as u32,
        exc_words.len() as u32,
        tab.len() as u32,
    ]));
    fixture.extend_from_slice(&le_u32(&words));
    fixture.extend_from_slice(&le_u32(&p12.exc_idx));
    fixture.extend_from_slice(&le_u32(&exc_words));
    fixture.extend_from_slice(&le_u32(&tab));
    fixture.extend_from_slice(&le_f32(&GSCALE));
    fixture.extend_from_slice(&le_f32(&rscale));
    fixture.extend_from_slice(&le_f32(&x));

    let mut child = Command::new(&out)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("host probe runs");
    child
        .stdin
        .take()
        .expect("stdin")
        .write_all(&fixture)
        .expect("fixture written");
    let res = child.wait_with_output().expect("host probe finishes");
    assert!(res.status.success(), "host probe failed");

    let mut r = res.stdout.as_slice();
    let take_u32 = |r: &mut &[u8], k: usize| -> Vec<u32> {
        let (a, b) = r.split_at(k * 4);
        *r = b;
        a.chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };
    let take_f32 = |r: &mut &[u8], k: usize| -> Vec<f32> {
        let (a, b) = r.split_at(k * 4);
        *r = b;
        a.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };
    let slots = take_u32(&mut r, n * DIM);
    let cls = take_u32(&mut r, n * 2);
    let y = take_f32(&mut r, D_OUT);
    assert!(r.is_empty(), "the host probe wrote more than it was asked to");

    // ---- lock A: the 2-plane field extraction, block by block ----
    for b in 0..n {
        let (ap, ag) = p12.decode_approx_block(&table, b);
        // The class id the main stream must carry: re-encode the approx
        // point through the archive indexer — an independent path to the id.
        let id = if ap == [0; DIM] {
            0
        } else {
            let q = ix.encode(&ap).expect("approx point is a codebook member");
            1 + fd.class_of(q).expect("member of the ball")
        };
        assert_eq!(cls[b * 2] as usize, id, "block {b}: class id");
        assert_eq!(cls[b * 2 + 1], ag, "block {b}: gain bit");
        let lv = (id > 0).then(|| fd.levels(id - 1));
        for j in 0..DIM {
            let v = ap[j];
            let lev = lv.map_or(0, |lv| {
                lv.values[..lv.len]
                    .iter()
                    .position(|&u| u == v.abs())
                    .expect("decoded magnitude is one of the class's levels")
            });
            let g = slots[b * DIM + j];
            assert_eq!((g & 7) as usize, lev, "block {b} slot {j}: level");
            assert_eq!((g >> 3) & 1, u32::from(v < 0), "block {b} slot {j}: sign");
        }
    }

    // ---- lock B: the emulated launch equals the exact f64 reference ----
    // The reference decodes the EXACT blocks (overlay resolved); the harness
    // computed the approximate row pass plus the correction pass from the
    // kernel's own device functions. Equality within TOL·Σ|w·x| is the proof
    // that the correction cancels the swap exactly — the host twin of the
    // bench's pre-timing verification.
    let mut worst = 0.0f64;
    for row in 0..D_OUT {
        let (mut want, mut scale) = (0.0f64, 0.0f64);
        for j in 0..NBLOCKS {
            let (pt, gain) = p12.decode_block(&table, row * NBLOCKS + j);
            if let Some(shell) = llvq_core::Leech::shell_index(&pt).filter(|&s| s > 0) {
                let k = GSCALE[gain as usize] as f64 * rscale[row] as f64
                    / ((16 * shell) as f64).sqrt();
                for (i, &v) in pt.iter().enumerate() {
                    let w = v as f64 * k;
                    want += w * x[j * DIM + i] as f64;
                    scale += (w * x[j * DIM + i] as f64).abs();
                }
            }
        }
        let e = (y[row] as f64 - want).abs() / scale.max(1e-12);
        worst = worst.max(e);
        assert!(
            e < TOL,
            "row {row}: overlay y differs from the exact reference by {e:.2e}·Σ|w·x|"
        );
    }
    println!(
        "planes12x host launch: {D_OUT} rows, {n} blocks, {n_exc} exceptions, \
         worst {worst:.1e}·Σ|w·x| (records: {} exception bytes)",
        n_exc * PLANES14_BYTES
    );
}
