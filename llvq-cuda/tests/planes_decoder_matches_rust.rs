//! The Planes14 CUDA decoder, compiled by `clang++` and diffed against Rust.
//!
//! Two independent locks, per the pattern of `decoder_matches_rust.rs`:
//!
//! 1. **The transcoding is a bijection.** Every Planes14 block must decode —
//!    in pure Rust, no C++ involved — to exactly the point and gain the
//!    Slot32 stream carries. This pins `planes14_from_slot32` to the frozen
//!    spec (14-byte stride, fields at 0/9/10/34/58/82, six trailing zeros)
//!    with the Slot32 decoder as the reference, so a field offset error or a
//!    plane/mask confusion cannot survive.
//!
//! 2. **The kernel text decides what Rust decides.** `planes_fields` and
//!    `planes_dot` — the same text NVRTC will compile, through the same shim
//!    as the Slot32 probe — are executed per block and checked field by
//!    field: class, gain, level of every slot, sign of every slot, and the
//!    dot product against an f64 recomputation. Including planes.cu also
//!    compiles `tv_planes`, so a syntax or type error costs two seconds here
//!    instead of a billed job.
//!
//! The fixture is both ends of every one of the 384 classes plus uniform
//! draws over the whole cap-13 ball plus the origin — wider than the sealed
//! file (capped at Λ₂₄(12)), so the 5-level records exercise all three
//! bit-planes, including the p2 plane only level 4 sets.
//!
//! What is left for the card: occupancy, the four-word coalescing pattern,
//! and register allocation (`local_bytes == 0` is read at bench startup).

use llvq_artifact::runtime::{transcode, ClassTable, Layout};
use llvq_core::{SplitMix64, DIM};
use llvq_search::fastdec::FastDecoder;
use llvq_search::index::N13;
use std::io::Write;
use std::process::{Command, Stdio};

include!("../src/planes14_host.rs");

const GSCALE: [f32; 2] = [0.625, 1.375];
const TABLE_ENTRIES: usize = 512;
const REC_WORDS: usize = 6;

fn le_u32(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn le_f32(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// The adversarial block set: both ends of every class (all zero patterns),
/// 20,000 uniform draws over the cap-13 ball, and the origin in mid-stream.
fn fixture_indices(fd: &FastDecoder, rng: &mut SplitMix64) -> Vec<u64> {
    let mut indices: Vec<u64> = Vec::new();
    for ci in 0..fd.n_classes() {
        let (first, last) = fd.class_range(ci);
        indices.push(first);
        indices.push(last);
    }
    indices.extend((0..20_000).map(|_| 1 + rng.next() % N13));
    indices.push(0);
    indices
}

#[test]
#[cfg_attr(debug_assertions, ignore = "sweeps the ball, run in release")]
fn planes14_is_a_bijection_of_slot32() {
    let fd = FastDecoder::new();
    let table = ClassTable::new(&fd, 1);
    // Three bit-planes address 8 levels; the layout is only bijective while
    // every class has at most 5. Structural today (MAX_LEVELS = 5), asserted
    // so a future cap change fails here and not in a decoded weight.
    assert!(
        (0..table.n_entries()).all(|e| table.record(e).len <= 5),
        "a class exceeds 5 levels: three planes no longer suffice"
    );

    let mut rng = SplitMix64::new(0x6_C0DE);
    let indices = fixture_indices(&fd, &mut rng);
    let gains: Vec<u32> = indices.iter().map(|_| (rng.next() & 1) as u32).collect();
    let n = indices.len();

    let rt = transcode(&fd, &table, &indices, &gains, Layout::Slot32).expect("transcodes");
    let planes = planes14_from_slot32(&rt, &table);
    assert_eq!(planes.len(), n * PLANES14_STRIDE, "14 bytes per block, exactly");

    for b in 0..n {
        let want = rt.decode_block(&table, b);
        let got = planes14_decode_block(&planes, &table, b);
        assert_eq!(got, want, "block {b}: Planes14 disagrees with Slot32");
    }
}

#[test]
#[cfg_attr(debug_assertions, ignore = "compiles C++ and sweeps the ball, run in release")]
fn the_planes_kernel_decides_what_the_rust_decoder_decides() {
    let out = std::env::temp_dir().join("llvq_host_planes");
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests");
    let st = Command::new("clang++")
        .args([
            "-std=c++17",
            "-O2",
            // Contraction off, every load-bearing multiply-add written as an
            // explicit `__fmaf_rn` → `std::fma`: same reasoning as the Slot32
            // probe, so a float difference is attributable.
            "-ffp-contract=off",
            "-Wall",
            "-Wextra",
            "-Werror",
        ])
        .arg(dir.join("host_planes.cpp"))
        .arg("-o")
        .arg(&out)
        .status()
        .expect("clang++ is on PATH");
    assert!(st.success(), "the Planes14 CUDA sources do not compile as host C++");

    // ---- fixture ----
    let fd = FastDecoder::new();
    let table = ClassTable::new(&fd, 1);
    let mut rng = SplitMix64::new(0x6_C0DE);
    let indices = fixture_indices(&fd, &mut rng);
    let gains: Vec<u32> = indices.iter().map(|_| (rng.next() & 1) as u32).collect();
    let n = indices.len();

    let rt = transcode(&fd, &table, &indices, &gains, Layout::Slot32).expect("transcodes");
    let mut bytes = planes14_from_slot32(&rt, &table);
    // The last block's four-word window reaches at most 2 bytes past the
    // stream; pad 4 and align, exactly as the bench does before upload.
    bytes.extend_from_slice(&[0u8; 4]);
    while !bytes.len().is_multiple_of(4) {
        bytes.push(0);
    }
    let words: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

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

    let x: Vec<f32> = (0..n * DIM).map(|_| rng.next_gaussian() as f32).collect();

    let mut fixture = Vec::new();
    fixture.extend_from_slice(&le_u32(&[
        n as u32,
        words.len() as u32,
        tab.len() as u32,
        x.len() as u32,
    ]));
    fixture.extend_from_slice(&le_u32(&words));
    fixture.extend_from_slice(&le_u32(&tab));
    fixture.extend_from_slice(&le_f32(&GSCALE));
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
    let dot = take_f32(&mut r, n);
    assert!(r.is_empty(), "the host probe wrote more than it was asked to");

    // ---- the comparison ----
    let mut worst = 0.0f64;
    for b in 0..n {
        let (pt, gain) = rt.decode_block(&table, b);
        let id = fd.class_of(indices[b]).map_or(0, |ci| 1 + ci);
        assert_eq!(cls[b * 2] as usize, id, "block {b}: class id");
        assert_eq!(cls[b * 2 + 1], gain, "block {b}: gain bit");

        let lv = (id > 0).then(|| fd.levels(id - 1));
        let (mut want, mut scale) = (0.0f64, 0.0f64);
        for j in 0..DIM {
            let v = pt[j];
            let lev = lv.map_or(0, |lv| {
                lv.values[..lv.len]
                    .iter()
                    .position(|&u| u == v.abs())
                    .expect("decoded magnitude is one of the class's levels")
            });
            let g = slots[b * DIM + j];
            assert_eq!((g & 7) as usize, lev, "block {b} slot {j}: level");
            // Stricter than the Slot32 probe can be: Planes14 *always* has a
            // real 24-bit sign field (the origin included — its planes and
            // smask are materialized zeros, not the next block's bits), and
            // the canonical stream writes zero slots' signs as 0. So every
            // slot's sign bit is comparable: negative iff the value is
            // negative, 0 on zero slots.
            assert_eq!((g >> 3) & 1, u32::from(v < 0), "block {b} slot {j}: sign");
            let a = lv.map_or(0.0, |lv| {
                lv.values[lev].abs() as f64 / ((16 * lv.shell) as f64).sqrt()
            });
            let w = if v < 0 { -a } else { a } * GSCALE[gain as usize] as f64;
            want += w * x[b * DIM + j] as f64;
            scale += (w * x[b * DIM + j] as f64).abs();
        }
        worst = worst.max((dot[b] as f64 - want).abs() / scale.max(1e-12));
    }
    assert!(
        worst < 1e-6,
        "worst dot error {worst:.2e}·Σ|w·x| — the levels agree, so this is arithmetic"
    );
}
