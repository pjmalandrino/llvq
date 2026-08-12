//! The Golay70 inference path, checked as far as a machine without a CUDA
//! card can check it — the exact locks `fused_planes12x.rs` holds on the
//! Planes12x path, applied to the layout the v2 campaign wires (step 5 of
//! `docs/archive/passation-golay70-2026-08-11.md`):
//!
//! 1. the translation unit: source list, concatenation order, entry point —
//!    the one piece of `fused_cuda`'s plumbing checkable here, since that
//!    module compiles on **no** local target;
//! 2. the decode arithmetic: `golay70_dot` (the v2 prologue decoder) and
//!    `golay70_row_correction` executed by the clang++ harness
//!    (`host_golay70_h.cpp`) over streams and GPU tables built by the
//!    *shipping* host code, against an f64 reference formed straight from
//!    the source indices — Golay70 reconstructs exactly, so the reference
//!    owes the layout no tolerance beyond float summation.
//!
//! 🚨 **What is not established here.** `tv_golay70_h` is compile-only: no
//! barrier, no `warp_sum`, no f16 store runs on this machine. The first run
//! on a card must diff its greedy tokens against the dense arm — that is the
//! lot C / step 5 gate, and nothing local substitutes for it.

use llvq_core::{Leech, SplitMix64, DIM};
use llvq_llm::fused::{
    golay70_gpu_class_table, golay70_gpu_codewords, FusedLayout, HostStream, Transcoder,
};
use llvq_search::fastdec::FastDecoder;
use llvq_search::index::N13;
use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};

const GSCALE: [f32; 2] = [0.625, 1.375];
const TABLE_ENTRIES: usize = 512;
const REC_WORDS: usize = 6;
const TOL: f64 = 1e-5;

/// Both ends of every class — every zero pattern, every level count, so the
/// E2 exception classes are all present — scattered through a uniform draw
/// over the cap-13 ball, plus the origin. Same construction as the Planes12x
/// fixture; at Golay70's ~7.4 % exception rate the scatter leaves rows with
/// several exceptions and rows with none, which the correction assertions
/// below rely on.
fn fixture(fd: &FastDecoder, d_out: usize, nblocks: usize, seed: u64) -> (Vec<u64>, Vec<u32>) {
    let mut rng = SplitMix64::new(seed);
    let n = d_out * nblocks;
    let mut indices: Vec<u64> = (0..n).map(|_| 1 + rng.next() % N13).collect();
    let mut pool: Vec<u64> = Vec::new();
    for ci in 0..fd.n_classes() {
        let (first, last) = fd.class_range(ci);
        pool.push(first);
        pool.push(last);
    }
    pool.push(0);
    for (k, &p) in pool.iter().enumerate() {
        indices[(k * (nblocks + 1)) % n] = p;
    }
    let gains: Vec<u32> = (0..n).map(|_| (rng.next() & 1) as u32).collect();
    (indices, gains)
}

/// The 512-entry `ClassRec` table the exception path reads — verbatim the
/// runtime's construction (`FusedRuntime::new`), restated here so the test
/// does not depend on the module it cannot compile.
fn gpu_class_table(fd: &FastDecoder) -> Vec<u32> {
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
    tab
}

struct G70<'a> {
    words: &'a [u32],
    exc_idx: &'a [u32],
    exc_words: &'a [u32],
    row_exc: &'a [u32],
}

fn unwrap_g70(s: &HostStream) -> G70<'_> {
    match s {
        HostStream::Golay70 { words, exc_idx, exc_words, row_exc } => G70 {
            words,
            exc_idx,
            exc_words,
            row_exc,
        },
        _ => panic!("layout golay70 demandé, autre flux rendu"),
    }
}

// ---------------------------------------------------------------------------
// 1. The translation unit
// ---------------------------------------------------------------------------

/// Golay70's unit extends Planes12x's — the same deliberate sharing of
/// translation-unit shape that Planes12x has with Planes14, so the register
/// report of `tv_planes_h` stays a drift detector on this build too — and it
/// carries the v2 decoder, its own kernel, and NEITHER bench kernel.
#[test]
fn the_golay70_unit_extends_planes12x() {
    use llvq_llm::fused::{load_planes_sources, matvec_kernel_name, planes_source_names};

    assert_eq!(matvec_kernel_name(FusedLayout::Golay70), "tv_golay70_h");
    let p12 = planes_source_names(FusedLayout::Planes12x);
    let g70 = planes_source_names(FusedLayout::Golay70);
    assert_eq!(&g70[..p12.len()], p12, "golay70 doit prolonger planes12x");
    assert_eq!(&g70[p12.len()..], ["llvq_golay.cuh", "tv_golay70_h.cu"]);

    let (parts, overridden) = load_planes_sources(FusedLayout::Golay70).expect("embarquées");
    assert!(overridden.is_none(), "LLVQ_KERNEL_DIR ne doit pas être posé ici");
    assert_eq!(parts.len(), g70.len());
    let unit = parts.concat();
    // The v2 decoder and the pieces the kernel calls, by name.
    for symbol in [
        "void tv_golay70_h(",
        "golay70_prologue",
        "golay70_dot",
        "golay70_row_correction",
        "golay70_exc_lane_term",
        "golay70_exc_combine",
        "LLVQ_GOLAY_CUH",
    ] {
        assert!(unit.contains(symbol), "l'unité golay70 ne porte pas {symbol}");
    }
    // And neither bench kernel: `tv_golay70` memsets `y` and accumulates
    // with atomicAdd — the protocol this path replaces — and `tv_golay70_v1`
    // is the frozen witness arm, which has no business in inference.
    assert!(
        !unit.contains("void tv_golay70("),
        "l'unité d'inférence embarque le noyau de banc tv_golay70"
    );
    assert!(
        !unit.contains("void tv_golay70_v1("),
        "l'unité d'inférence embarque le témoin tv_golay70_v1"
    );
}

// ---------------------------------------------------------------------------
// 2. The decode arithmetic, executed by the clang++ harness
// ---------------------------------------------------------------------------

#[test]
#[cfg_attr(debug_assertions, ignore = "compiles C++, run in release")]
fn the_golay70_half_kernel_decides_what_rust_decides() {
    const D_OUT: usize = 32;
    const NBLOCKS: usize = 45;

    let out = std::env::temp_dir().join("llvq_host_golay70_h");
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests");
    let st = Command::new("clang++")
        .args([
            "-std=c++17",
            "-O2",
            // Contraction off, every load-bearing multiply-add an explicit
            // `__fmaf_rn` → `std::fma`: same reasoning as every other probe,
            // so a float difference is attributable.
            "-ffp-contract=off",
            "-Wall",
            "-Wextra",
            "-Werror",
        ])
        .arg(dir.join("host_golay70_h.cpp"))
        .arg("-o")
        .arg(&out)
        .status()
        .expect("clang++ is on PATH");
    assert!(
        st.success(),
        "la source Golay70 demi-stockante ne compile pas en C++ hôte"
    );

    let fd = FastDecoder::new();
    let (idx, gains) = fixture(&fd, D_OUT, NBLOCKS, 0x60_1A_47);
    let tr = Transcoder::new(FusedLayout::Golay70).unwrap();
    let (s, _) = tr.stream(&idx, &gains, D_OUT, NBLOCKS).unwrap();
    let g = unwrap_g70(&s);
    assert!(!g.exc_idx.is_empty(), "la fixture doit exercer la correction");

    let tab = gpu_class_table(&fd);
    let g70t = tr.golay70_table().expect("table golay70 du transcodeur");
    let gtab = golay70_gpu_class_table(&fd, g70t);
    let cw = golay70_gpu_codewords(g70t);
    let mut rng = SplitMix64::new(0x5EED);
    let rscale: Vec<f32> = (0..D_OUT)
        .map(|_| 0.5 + rng.next_gaussian().abs() as f32)
        .collect();
    let x: Vec<f32> = (0..NBLOCKS * DIM).map(|_| rng.next_gaussian() as f32).collect();

    let le_u32 = |v: &[u32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    let le_f32 = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    let mut fx = Vec::new();
    fx.extend_from_slice(&le_u32(&[
        D_OUT as u32,
        NBLOCKS as u32,
        g.exc_idx.len() as u32,
        g.words.len() as u32,
        g.exc_words.len() as u32,
        tab.len() as u32,
        gtab.len() as u32,
        cw.len() as u32,
    ]));
    fx.extend_from_slice(&le_u32(g.words));
    fx.extend_from_slice(&le_u32(g.exc_idx));
    fx.extend_from_slice(&le_u32(g.exc_words));
    fx.extend_from_slice(&le_u32(g.row_exc));
    fx.extend_from_slice(&le_u32(&tab));
    fx.extend_from_slice(&le_u32(&gtab));
    fx.extend_from_slice(&le_u32(&cw));
    fx.extend_from_slice(&le_f32(&GSCALE));
    fx.extend_from_slice(&le_f32(&rscale));
    fx.extend_from_slice(&le_f32(&x));

    let mut child = Command::new(&out)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("host probe runs");
    child
        .stdin
        .take()
        .expect("stdin")
        .write_all(&fx)
        .expect("fixture written");
    let res = child.wait_with_output().expect("host probe finishes");
    assert!(res.status.success(), "host probe failed");
    let mut r = res.stdout.as_slice();
    let mut take_f32 = |k: usize| -> Vec<f32> {
        let (a, b) = r.split_at(k * 4);
        r = b;
        a.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };
    let y = take_f32(D_OUT);
    let corr = take_f32(D_OUT);
    assert!(r.is_empty(), "la sonde hôte a écrit plus qu'on ne lui demandait");

    // The correction is zero exactly where a row has no exception, and
    // non-zero where it has one. The first half kills a correction pass that
    // fires on every block; the second kills one that never fires — which,
    // Golay70's main stream holing exceptions to the ORIGIN, would not leave
    // an approximation behind but a hole: y would be missing those blocks'
    // whole contribution, a mutation a loose row tolerance must not absorb.
    for (row, &c) in corr.iter().enumerate() {
        let n_exc = g.row_exc[row + 1] - g.row_exc[row];
        if n_exc == 0 {
            assert_eq!(c, 0.0, "ligne {row} sans exception, correction non nulle");
        } else {
            assert!(c != 0.0, "ligne {row} : {n_exc} exceptions et une correction nulle");
        }
    }

    // The f64 reference, straight from the source indices: Golay70
    // reconstructs exactly (main stream for regular blocks, exception record
    // for the rest), so the decoded lattice point IS the expectation — no
    // overlay algebra, no tolerance beyond float summation order.
    let mut worst = 0.0f64;
    for row in 0..D_OUT {
        let (mut want, mut scale) = (0.0f64, 0.0f64);
        for col in 0..NBLOCKS {
            let b = row * NBLOCKS + col;
            let Some(pt) = fd.decode(idx[b]) else {
                assert_eq!(idx[b], 0, "index hors boule");
                continue;
            };
            let Some(shell) = Leech::shell_index(&pt).filter(|&s| s > 0) else {
                continue;
            };
            let k = GSCALE[gains[b] as usize] as f64 * rscale[row] as f64
                / ((16 * shell) as f64).sqrt();
            for (i, &v) in pt.iter().enumerate() {
                let w = v as f64 * k;
                want += w * x[col * DIM + i] as f64;
                scale += (w * x[col * DIM + i] as f64).abs();
            }
        }
        let e = (y[row] as f64 - want).abs() / scale.max(1e-12);
        worst = worst.max(e);
        assert!(e < TOL, "ligne {row} : écart {e:.2e}·Σ|w·x|");
    }
    println!(
        "golay70_h hôte : {D_OUT} lignes, {} blocs, {} exceptions, pire écart \
         {worst:.1e}·Σ|w·x|",
        idx.len(),
        g.exc_idx.len()
    );
}
