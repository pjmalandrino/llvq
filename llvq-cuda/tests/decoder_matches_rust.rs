//! The CUDA decoder, compiled by `clang++` and diffed against the Rust one.
//!
//! `slot_dot` has no warp primitive, no shared memory and no atomic: it is
//! scalar bit arithmetic. So everything that is not a *hardware* property can
//! be checked here, on the development machine, in two seconds and for
//! nothing — instead of ten minutes and twenty cents on a rented card, which
//! is the difference between iterating and guessing.
//!
//! It compiles the **same text** NVRTC will compile, through a shim that
//! neutralises `__device__`/`__global__` and turns the block indices into
//! globals a driver loop sets. A separate host reimplementation would be
//! worthless: it could share a bug with the kernel, or drift from it.
//!
//! The fixture is both ends of every one of the 384 classes plus uniform
//! draws over the whole cap-13 ball — wider than the sealed file, which is
//! capped at Λ₂₄(12) and therefore never exercises the 5-level, 130-bit
//! records that are the worst case for the kernel's five-word read.

use llvq_artifact::runtime::{transcode, ClassTable, Layout};
use llvq_core::{SplitMix64, DIM};
use llvq_search::fastdec::FastDecoder;
use llvq_search::index::N13;
use std::io::Write;
use std::process::{Command, Stdio};

const GSCALE: [f32; 2] = [0.625, 1.375];
const TABLE_ENTRIES: usize = 512;
const REC_WORDS: usize = 6;

fn le_u32(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn le_f32(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

#[test]
#[cfg_attr(debug_assertions, ignore = "compiles C++ and sweeps the ball, run in release")]
fn the_cuda_decoder_decides_what_the_rust_decoder_decides() {
    let out = std::env::temp_dir().join("llvq_host_probe");
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests");
    let st = Command::new("clang++")
        .args([
            "-std=c++17",
            "-O2",
            // The one flag that is not a convenience. NVRTC compiles with
            // `--fmad=true`, so the device contracts `a*b+c` into an FMA; a
            // host compiler left free to do the same, or not, at its own
            // discretion would make a float difference impossible to
            // attribute. Contraction off here, and every multiply-add the
            // kernel actually depends on is written as an explicit
            // `__fmaf_rn`, which the shim maps to a correctly-rounded
            // `std::fma`. What is left is comparable.
            "-ffp-contract=off",
            "-Wall",
            "-Wextra",
            "-Werror",
        ])
        .arg(dir.join("host_probe.cpp"))
        .arg("-o")
        .arg(&out)
        .status()
        .expect("clang++ is on PATH");
    assert!(st.success(), "the CUDA sources do not compile as host C++");

    // The fused matvec, compile-only. It cannot run here — a single-threaded
    // driver reproduces neither `__syncthreads` nor a warp shuffle — but a
    // syntax or type error in it costs a fifty-minute image rebuild to find,
    // and this costs a second.
    let st = Command::new("clang++")
        .args(["-std=c++17", "-O1", "-Wall", "-Wextra", "-Werror", "-c"])
        .arg(dir.join("host_matvec.cpp"))
        .arg("-o")
        .arg(std::env::temp_dir().join("llvq_host_matvec.o"))
        .status()
        .expect("clang++ is on PATH");
    assert!(st.success(), "the fused matvec kernels do not compile");

    // ---- fixture ----
    let fd = FastDecoder::new();
    let table = ClassTable::new(&fd, 1);
    assert!(
        24 + table.worst_width_slot() <= 160,
        "the class table overflows the kernel's five-word window"
    );

    let mut indices: Vec<u64> = Vec::new();
    for ci in 0..fd.n_classes() {
        let (first, last) = fd.class_range(ci);
        indices.push(first);
        indices.push(last);
    }
    let mut rng = SplitMix64::new(0x6_C0DE);
    indices.extend((0..20_000).map(|_| 1 + rng.next() % N13));
    indices.push(0);
    let gains: Vec<u32> = indices.iter().map(|_| (rng.next() & 1) as u32).collect();
    let n = indices.len();

    let rt = transcode(&fd, &table, &indices, &gains, Layout::Slot32).expect("transcodes");

    let mut bytes = rt.data.clone();
    bytes.extend_from_slice(&[0u8; 20]);
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

    // The C++ harness recomputes this length rather than being told it, so a
    // disagreement surfaces here as a truncated fixture instead of on the
    // device as an illegal address. It already has: the first version used
    // `n / 32 + 1`, which is one entry short whenever the block count is not
    // a multiple of 32 — and `slot_dot` reads `bases[g + 1]`.
    assert_eq!(rt.bases.len(), n.div_ceil(32) + 1, "bases length contract");

    let mut fixture = Vec::new();
    fixture.extend_from_slice(&le_u32(&[
        n as u32,
        words.len() as u32,
        tab.len() as u32,
        x.len() as u32,
    ]));
    fixture.extend_from_slice(&le_u32(&words));
    fixture.extend_from_slice(&le_u32(&rt.bases));
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
    let _floor = take_f32(&mut r, n);
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
            // A zero slot's sign bit is meaningless, and for the origin the
            // sign field is not part of the record at all — the kernel reads
            // the 24 bits that follow the header, which belong to the next
            // blocks. Harmless (`len = 1` forces every value to 0.0f) but it
            // means only nonzero slots carry a comparable sign.
            if v != 0 {
                assert_eq!((g >> 3) & 1, u32::from(v < 0), "block {b} slot {j}: sign");
            }
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
