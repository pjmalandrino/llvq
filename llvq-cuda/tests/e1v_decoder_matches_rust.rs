//! The E1v CUDA decoder, compiled by `clang++` and run against Rust.
//!
//! The pattern of `decoder_matches_rust.rs`, applied to the one layout that had
//! only a *transcription* to vouch for it. `llvq-artifact/tests/e1v_cuda_mirror.rs`
//! re-derives the arithmetic in Rust and checks it slot by slot; this runs **the
//! text NVRTC will compile**, on the same bytes, and the difference between the
//! two matters — a mirror can drift from what it mirrors, and nothing notices.
//!
//! ## The split, and what each half is worth
//!
//! `e1v_dot` = a warp scan, then `e1v_decode_at`. The scan cannot run here:
//! `__shfl_up_sync` has no host semantics, and `host_shim.h` states that of
//! every shuffle in this crate. So the kernel was written with the decode
//! **split out**, and this test drives the split half — the codeword lookup,
//! the two binomial walks, the parity repair, the three sign rules and the
//! deposit, which is where every bug this repository has actually had lives.
//! The scan's *value* is checked by the mirror; the scan's *primitive* is the
//! card's business, and `preflight` reads the spill.
//!
//! Including `llvq_e1v.cuh` also compiles it, so a syntax or type error costs
//! two seconds here instead of a billed job.
//!
//! ## The fixture
//!
//! Five row shapes, including the three of the published 4B, a row narrower
//! than a group and one that divides exactly — the partial group is what
//! distinguishes the served cut from the file-order one, and a fixture of full
//! groups would prove the wrong thing. Every class of the codebook plus the
//! origin, the latter being the id whose payload is empty and whose table entry
//! is 0 rather than `1 + ci`.

use llvq_artifact::blockrec::{binom_table, block_records, e1v_payload_bits, GpuBlockRec};
use llvq_artifact::e1v::{transcode_e1v_rows, E1V_GROUP, E1V_ORIGIN_ID};
use llvq_core::{Golay, DIM};
use llvq_search::cns::{cns_layout, CnsLayout};
use llvq_search::fastdec::FastDecoder;
use std::io::Write;
use std::process::{Command, Stdio};

const GSCALE: [f32; 2] = [0.9, 1.1];
/// Relative to `Σ|wᵢxᵢ|`. The kernel accumulates four f32 chains, the reference
/// sums in f64: the gap is float order, not decode.
const TOL: f64 = 1e-6;

fn le_u32(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn le_f32(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// `GpuBlockRec` as the bytes the device receives.
fn tab_bytes(recs: &[GpuBlockRec]) -> Vec<u8> {
    // SAFETY: `GpuBlockRec` is `#[repr(C)]`, `Copy`, holds no padding the
    // struct does not declare (a `pad` byte is a field), and its size is
    // asserted at build time. This is the same reinterpretation the upload
    // performs; doing it by hand field by field would be a second layout to
    // get wrong.
    unsafe {
        std::slice::from_raw_parts(
            recs.as_ptr() as *const u8,
            std::mem::size_of_val(recs),
        )
        .to_vec()
    }
}

#[test]
#[cfg_attr(debug_assertions, ignore = "compiles C++ and sweeps every class, run in release")]
fn the_cuda_e1v_decoder_decides_what_the_rust_decoder_decides() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests");
    let bin = std::env::temp_dir().join("llvq_host_e1v");
    let st = Command::new("clang++")
        .args([
            "-std=c++17",
            "-O2",
            // NVRTC compiles with `--fmad=true`; a host compiler free to
            // contract at its own discretion would make a float difference
            // impossible to attribute. Off here, and every multiply-add the
            // kernel depends on is an explicit `fmaf`.
            "-ffp-contract=off",
            "-Wall",
            "-Wextra",
            "-Werror",
        ])
        .arg(dir.join("host_e1v.cpp"))
        .arg("-o")
        .arg(&bin)
        .status()
        .expect("clang++ is on PATH");
    assert!(st.success(), "llvq_e1v.cuh ne compile pas en C++ hôte");

    let fd = FastDecoder::new();
    let golay = Golay::new();
    let lays: Vec<CnsLayout> = (0..fd.n_classes()).map(|ci| cns_layout(&fd, &golay, ci)).collect();
    let recs = block_records(&fd, &golay);
    let pay = e1v_payload_bits(&fd, &golay, &recs);
    let binom = binom_table();
    let x: Vec<f32> = (0..DIM).map(|i| 1.0 + i as f32 * 0.125).collect();

    // Every class at both ends and in the middle, plus the origin.
    let mut poolv = vec![0u64];
    for ci in 0..fd.n_classes() {
        let (first, last) = fd.class_range(ci);
        poolv.push(first);
        poolv.push(last);
        poolv.push(first + (last - first) / 2);
    }

    let mut seen: std::collections::BTreeSet<Option<usize>> = std::collections::BTreeSet::new();
    let mut total = 0usize;
    for row_blocks in [106usize, 170, 405, 7, 64] {
        let rows = 4usize;
        let n = rows * row_blocks;
        let idxs: Vec<u64> = (0..n).map(|i| poolv[i % poolv.len()]).collect();
        let gains: Vec<u32> = (0..n).map(|i| (i % 2) as u32).collect();
        let s = transcode_e1v_rows(&fd, &golay, &idxs, &gains, row_blocks).expect("transcodes");

        let mut words: Vec<u32> = s
            .data
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        words.push(0);
        words.push(0); // the three-word window's landing pad

        // The addressing the scan would produce, computed serially. This is the
        // one part the probe is not asked to reproduce.
        let per_row = row_blocks.div_ceil(E1V_GROUP);
        let mut addr: Vec<u32> = Vec::with_capacity(n * 5);
        for row in 0..rows {
            for gl in 0..per_row {
                let g = row * per_row + gl;
                let len = E1V_GROUP.min(row_blocks - gl * E1V_GROUP);
                let gbit = u64::from(s.bases[g]) * 32;
                let mut acc = 0u32;
                for lane in 0..len {
                    let b = row * row_blocks + gl * E1V_GROUP + lane;
                    let id = fd.class_of(idxs[b]).map_or(E1V_ORIGIN_ID as u32, |ci| ci as u32);
                    addr.extend_from_slice(&[
                        gbit as u32,
                        (gbit >> 32) as u32,
                        len as u32,
                        acc,
                        lane as u32,
                    ]);
                    acc += pay[id as usize];
                    let _ = b;
                }
            }
        }
        assert_eq!(addr.len(), n * 5, "une adresse par bloc");

        let mut fixture = le_u32(&[n as u32, words.len() as u32, recs.len() as u32, pay.len() as u32]);
        fixture.extend(le_u32(&words));
        fixture.extend(tab_bytes(&recs));
        fixture.extend(le_u32(&pay));
        fixture.extend(le_u32(&binom));
        fixture.extend(le_u32(golay.codewords()));
        fixture.extend(le_f32(&GSCALE));
        fixture.extend(le_f32(&x));
        fixture.extend(le_u32(&addr));

        let mut child = Command::new(&bin)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .expect("host probe runs");
        child.stdin.as_mut().expect("stdin").write_all(&fixture).expect("fixture written");
        let res = child.wait_with_output().expect("host probe finishes");
        assert!(res.status.success(), "host probe failed");
        assert_eq!(res.stdout.len(), n * 12, "sortie de taille inattendue");

        let dot: Vec<f32> = res.stdout[..n * 4]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let ids: Vec<u32> = res.stdout[n * 4..n * 8]
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let gs: Vec<u32> = res.stdout[n * 8..]
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        for b in 0..n {
            seen.insert(fd.class_of(idxs[b]));
            // The header the kernel read out of the stream, against the format.
            assert_eq!(
                ids[b],
                fd.class_of(idxs[b]).map_or(E1V_ORIGIN_ID as u32, |ci| ci as u32),
                "forme {row_blocks}, bloc {b} : classe lue par le noyau"
            );
            assert_eq!(gs[b], gains[b], "forme {row_blocks}, bloc {b} : gain lu par le noyau");

            // The reference: the format's own reader, and the archive decoder
            // behind it (P5 C2).
            let (point, _) = s.decode_block(&fd, &golay, &lays, b);
            let (mut want, mut mag) = (0.0f64, 0.0f64);
            if let Some(ci) = fd.class_of(idxs[b]) {
                let norm = ((16 * fd.levels(ci).shell) as f64).sqrt();
                let sc = f64::from(GSCALE[gains[b] as usize]) / norm;
                for (&p, &xi) in point.iter().zip(&x) {
                    let t = f64::from(p) * sc * f64::from(xi);
                    want += t;
                    mag += t.abs();
                }
            }
            let d = (f64::from(dot[b]) - want).abs();
            let rel = if mag > 0.0 { d / mag } else { d };
            assert!(
                rel <= TOL,
                "forme {row_blocks}, bloc {b} : noyau {} contre Rust {want} ({rel:.3e} relatif)",
                dot[b]
            );
        }
        total += n;
    }

    // The §5 discipline: a fixture that quietly missed a class prints the same
    // green line as one that did not.
    assert!(seen.contains(&None), "aucun bloc origine dans la sonde");
    for ci in 0..fd.n_classes() {
        assert!(seen.contains(&Some(ci)), "classe {ci} absente de la sonde");
    }
    eprintln!(
        "sonde hôte E1v — {total} blocs sur 5 formes, les {} classes plus l'origine,\n  \
         le TEXTE de llvq_e1v.cuh exécuté par clang++ contre E1vBlocks::decode_block.",
        fd.n_classes()
    );
}
