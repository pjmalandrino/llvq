//! The `Planes12x` arm of the fused loader, pinned on a machine with no card.
//!
//! `llvq-artifact` already proves the layout itself: `Planes12xBlocks`'s
//! overlay reconstructs the archive bit for bit over the sealed 4B's
//! 150,681,600 blocks (`planes12x_format.rs`). `llvq-cuda` already proves the
//! bench decoder's field extraction against Rust. What neither can see is the
//! layer this crate adds, and every part of it is a way to ship a correct
//! stream and still be wrong on the card:
//!
//!  1. the packing of three byte arrays into the `u32` words the kernel
//!     indexes, and the read-window padding of the exception table;
//!  2. **`row_exc`** — the `d_out + 1` prefix offsets that let the inference
//!     kernel keep one warp per output row and drop the bench's
//!     memset-and-atomicAdd protocol. It is new here, it has no counterpart
//!     upstream, and a row filed wrong reads a valid-looking slice of the
//!     activation and returns finite, plausible, wrong numbers;
//!  3. the row-local correction itself — `planes12x_row_correction`, the
//!     device function this lot adds — executed lane by lane through the
//!     clang++ harness against an f64 reference.
//!
//! 🚨 **What is not established here.** `tv_planes12x_h` is compile-only:
//! the grid, the two barriers, `warp_sum` and the f16 store cannot be driven
//! by a single-threaded host. Everything below is about the arithmetic and
//! the host tables. The kernel's correctness is a card's job.

use llvq_artifact::runtime::{ClassTable, PLANES12X_BYTES, PLANES14_BYTES};
use llvq_core::{Leech, Point, SplitMix64, DIM};
use llvq_llm::fused::{FusedLayout, HostStream, Transcoder};
use llvq_search::fastdec::FastDecoder;
use llvq_search::index::N13;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

const GSCALE: [f32; 2] = [0.625, 1.375];
const TABLE_ENTRIES: usize = 512;
const REC_WORDS: usize = 6;
const TOL: f64 = 1e-5;

// ---------------------------------------------------------------------------
// The kernel's own read windows, restated in Rust
// ---------------------------------------------------------------------------
//
// Deliberately not a call into `Planes12xBlocks::decode_block`: these mirror
// `planes12_fields` and `planes_fields` word for word — same word indices,
// same shifts, same masks — so that indexing `words` directly is the test. A
// missing end pad is an out-of-bounds panic here, exactly where it is an
// illegal address there, and a packing bug that both the encoder and the
// artifact decoder agree on cannot hide behind a shared implementation.

/// `planes12_fields`: the three aligned words at `12·b`, zero shift.
/// Returns `(class, gain, smask, [p0, p1])`.
fn planes12_fields(words: &[u32], b: usize) -> (usize, u32, u32, [u32; 2]) {
    let w = 3 * b;
    let (w0, w1, w2) = (words[w], words[w + 1], words[w + 2]);
    let lo = (u64::from(w1) << 32) | u64::from(w0);
    let p1 = ((lo >> 58) | (u64::from(w2) << 6)) as u32 & 0xff_ffff;
    (
        (w0 & 0x1ff) as usize,
        (w0 >> 9) & 1,
        (lo >> 10) as u32 & 0xff_ffff,
        [(lo >> 34) as u32 & 0xff_ffff, p1],
    )
}

/// `planes_fields`: the four words at `14·r`, shift 0 or 16.
/// Returns `(class, gain, smask, [p0, p1, p2])`.
fn planes_fields(words: &[u32], r: usize) -> (usize, u32, u32, [u32; 3]) {
    let byte = PLANES14_BYTES * r;
    let w = byte >> 2;
    let (w0, w1, w2, w3) = (
        u64::from(words[w]),
        u64::from(words[w + 1]),
        u64::from(words[w + 2]),
        u64::from(words[w + 3]),
    );
    let sh = ((byte & 3) * 8) as u32;
    let lo = w1 << 32 | w0;
    let hi = w3 << 32 | w2;
    let hdr = ((lo >> sh) & 0x3ff) as u32;
    let fs = sh + 10;
    let pay_lo = (lo >> fs) | (hi << (64 - fs));
    let pay_hi = hi >> fs;
    let ext24 = |off: u32| -> u32 {
        if off < 64 {
            (((pay_lo >> off) | (pay_hi << (64 - off))) & 0xff_ffff) as u32
        } else {
            ((pay_hi >> (off - 64)) & 0xff_ffff) as u32
        }
    };
    (
        (hdr & 0x1ff) as usize,
        hdr >> 9,
        (pay_lo & 0xff_ffff) as u32,
        [ext24(24), ext24(48), ext24(72)],
    )
}

/// Rebuild a lattice point from a class, a sign mask and `n` bit-planes —
/// the level of slot `j` is the little-endian read of its plane bits, which
/// is the layout's definition and the tree the kernels evaluate.
fn point_from_planes(
    table: &ClassTable,
    id: usize,
    smask: u32,
    planes: &[u32],
    what: &str,
) -> Point {
    let mut p = [0i32; DIM];
    if id == 0 {
        for (k, &m) in planes.iter().enumerate() {
            assert_eq!(m, 0, "{what}: plane {k} not zero on an origin record");
        }
        assert_eq!(smask, 0, "{what}: sign mask not zero on the origin");
        return p;
    }
    let rec = table.record(id);
    for (j, pj) in p.iter_mut().enumerate() {
        let mut lvl = 0usize;
        for (k, &m) in planes.iter().enumerate() {
            lvl |= (((m >> j) & 1) as usize) << k;
        }
        assert!(
            lvl < rec.len as usize,
            "{what}, slot {j}: level {lvl} outside class {id} ({} levels)",
            rec.len
        );
        let v = rec.values[lvl];
        *pj = if (smask >> j) & 1 == 1 { -v } else { v };
    }
    p
}

/// The four arrays the `Planes12x` kernel is handed, as one borrow.
///
/// Grouped rather than passed loose because that is what they are on the
/// device — `launch_planes12x_h` takes them as one `[&CudaSlice<u32>; 4]` for
/// the same reason: four arguments of identical type, two of which
/// (`exc_idx`, `row_exc`) index into each other, are transposable in silence.
#[derive(Clone, Copy)]
struct P12<'a> {
    words: &'a [u32],
    exc_idx: &'a [u32],
    exc_words: &'a [u32],
    row_exc: &'a [u32],
}

/// What the shipped `Planes12x` stream says about block `col` of row `row`,
/// read exactly the way `tv_planes12x_h` reads it: the main stream's
/// approximation, then the exception record if **this row's slice** names the
/// block. Going through `row_exc` rather than a global search over `exc_idx`
/// is the point — that is the lookup the kernel performs.
fn overlay_block(
    table: &ClassTable,
    s: P12<'_>,
    row: usize,
    nblocks: usize,
    col: usize,
) -> (Point, u32) {
    let b = row * nblocks + col;
    let (lo, hi) = (s.row_exc[row] as usize, s.row_exc[row + 1] as usize);
    for (k, &eb) in s.exc_idx[lo..hi].iter().enumerate() {
        if eb as usize == b {
            let e = lo + k;
            let (id, gain, smask, p) = planes_fields(s.exc_words, e);
            return (
                point_from_planes(table, id, smask, &p, &format!("exception {e}")),
                gain,
            );
        }
    }
    let (id, gain, smask, p) = planes12_fields(s.words, b);
    (
        point_from_planes(table, id, smask, &p, &format!("block {b}")),
        gain,
    )
}

/// The main stream alone, ignoring the exception table — what the row pass
/// computes before the correction cancels it.
fn approx_block(table: &ClassTable, words: &[u32], b: usize) -> (Point, u32) {
    let (id, gain, smask, p) = planes12_fields(words, b);
    (
        point_from_planes(table, id, smask, &p, &format!("block {b}")),
        gain,
    )
}

fn planes14_block(table: &ClassTable, words: &[u32], b: usize) -> (Point, u32) {
    let (id, gain, smask, p) = planes_fields(words, b);
    (
        point_from_planes(table, id, smask, &p, &format!("block {b}")),
        gain,
    )
}

fn unwrap_p12(s: &HostStream) -> P12<'_> {
    match s {
        HostStream::Planes12x { words, exc_idx, exc_words, row_exc } => P12 {
            words,
            exc_idx,
            exc_words,
            row_exc,
        },
        _ => panic!("planes12x layout asked for, another stream returned"),
    }
}

fn unwrap_p14(s: &HostStream) -> &[u32] {
    match s {
        HostStream::Planes14 { words } => words,
        _ => panic!("planes14 layout asked for, another stream returned"),
    }
}

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

/// Both ends of every class — every zero pattern, every level count, so the
/// 5-level classes that become exceptions are all present — scattered through
/// a uniform draw over the cap-13 ball, plus the origin. The stride past a
/// divisor of the row length spreads the pool across rows and columns rather
/// than into one column, so there are rows with several exceptions and rows
/// with none.
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

// ---------------------------------------------------------------------------
// 1. The shipped words rebuild the Planes14 content
// ---------------------------------------------------------------------------

/// Every block of the packed `Planes12x` arrays, read through the kernel's own
/// windows and resolved through `row_exc`, equals the `Planes14` block — and
/// the main stream alone differs on **exactly** the exception blocks.
///
/// The second half is what makes the first non-trivial. Without it a
/// transcoder that emitted an exception for every block would pass: the
/// overlay would be exact and the layout would weigh 14 bytes a block instead
/// of 12, i.e. the format would have been silently thrown away.
#[test]
#[cfg_attr(debug_assertions, ignore = "one lattice search per 5-level block, run in release")]
fn planes12x_words_rebuild_the_planes14_content() {
    let fd = FastDecoder::new();
    let table = ClassTable::new(&fd, 1);
    // Two row shapes: `nblocks` odd and even, so `14·n mod 4` takes both
    // values on the exception table and its last window exercises the pad.
    for &(d_out, nblocks, seed) in &[(16usize, 45usize, 0x12_C0DEu64), (8, 32, 0xBEEF)] {
        let (idx, gains) = fixture(&fd, d_out, nblocks, seed);
        let t12 = Transcoder::new(FusedLayout::Planes12x).unwrap();
        let t14 = Transcoder::new(FusedLayout::Planes14).unwrap();
        let (s12, bytes12) = t12.stream(&idx, &gains, d_out, nblocks).unwrap();
        let (s14, _) = t14.stream(&idx, &gains, d_out, nblocks).unwrap();
        let p12 = unwrap_p12(&s12);
        let (words, exc_idx, exc_words) = (p12.words, p12.exc_idx, p12.exc_words);
        let p14 = unwrap_p14(&s14);
        assert!(
            !exc_idx.is_empty(),
            "d_out={d_out}: the fixture must produce exceptions"
        );
        assert!(
            exc_idx.len() < idx.len() / 4,
            "d_out={d_out}: {} exceptions over {} blocks, the main stream carries \
             nothing any more",
            exc_idx.len(),
            idx.len()
        );

        // The read windows must sit inside the buffers.
        assert!(3 * (idx.len() - 1) + 3 <= words.len(), "3-word window out of buffer");
        assert!(
            PLANES14_BYTES * (exc_idx.len() - 1) / 4 + 4 <= exc_words.len(),
            "4-word window of the exception table out of buffer"
        );

        let mut approx_differs = 0usize;
        for row in 0..d_out {
            for col in 0..nblocks {
                let b = row * nblocks + col;
                let want = planes14_block(&table, p14, b);
                let got = overlay_block(&table, p12, row, nblocks, col);
                assert_eq!(got, want, "block {b}: the overlay diverges from Planes14");
                let ap = approx_block(&table, words, b);
                let is_exc = exc_idx.binary_search(&(b as u32)).is_ok();
                assert_eq!(
                    ap != want,
                    is_exc,
                    "block {b}: the main stream must differ exactly on the \
                     exceptions"
                );
                if is_exc {
                    approx_differs += 1;
                    // The swap moves the direction only: the gain bit is copied.
                    assert_eq!(ap.1, want.1, "block {b}: the swap changed the gain");
                }
            }
        }
        assert_eq!(approx_differs, exc_idx.len(), "every exception seen");

        // The accounting: 12 bytes a block, 18 an exception, 4 a row + 4.
        let want_bytes = (idx.len() * PLANES12X_BYTES
            + exc_idx.len() * (4 + PLANES14_BYTES)
            + (d_out + 1) * 4) as u64;
        assert_eq!(bytes12, want_bytes, "planes12x payload accounting");
    }
}

/// `row_exc` partitions the exception table, in order, row by row.
///
/// The kernel derives an exception's column as `b − row·nblocks` from the row
/// it already owns, so this table *is* the row attribution. A slice that
/// straddled a row boundary would still index inside the activation and
/// produce a plausible number — nothing downstream would notice.
#[test]
#[cfg_attr(debug_assertions, ignore = "one lattice search per 5-level block, run in release")]
fn the_row_offsets_partition_the_exception_table() {
    let fd = FastDecoder::new();
    let (d_out, nblocks) = (16usize, 45usize);
    let (idx, gains) = fixture(&fd, d_out, nblocks, 0x12_C0DE);
    let tr = Transcoder::new(FusedLayout::Planes12x).unwrap();
    let (s, _) = tr.stream(&idx, &gains, d_out, nblocks).unwrap();
    let p12 = unwrap_p12(&s);
    let (exc_idx, row_exc) = (p12.exc_idx, p12.row_exc);

    assert_eq!(row_exc.len(), d_out + 1, "one bound per row, plus the sentinel");
    assert_eq!(row_exc[0], 0);
    assert_eq!(*row_exc.last().unwrap() as usize, exc_idx.len(), "coverage");
    assert!(row_exc.windows(2).all(|w| w[0] <= w[1]), "increasing bounds");
    let mut with_several = 0usize;
    for r in 0..d_out {
        let sl = &exc_idx[row_exc[r] as usize..row_exc[r + 1] as usize];
        if sl.len() > 1 {
            with_several += 1;
        }
        for &b in sl {
            assert_eq!(
                b as usize / nblocks,
                r,
                "block {b} is filed under row {r}, whose blocks are \
                 [{}, {})",
                r * nblocks,
                (r + 1) * nblocks
            );
        }
        assert!(sl.windows(2).all(|w| w[0] < w[1]), "row {r}: strict order");
    }
    assert!(
        with_several > 0,
        "the fixture must put two exceptions on one row, otherwise the kernel's \
         per-lane loop is never exercised past a single turn"
    );
}

// ---------------------------------------------------------------------------
// 2. The kernel's own arithmetic, executed
// ---------------------------------------------------------------------------

/// The overlay correction, run through the device function the kernel calls,
/// cancels the main stream's approximation exactly.
///
/// Compiles the shipped translation unit as host C++ (which is the only check
/// `tv_planes12x_h` gets on this machine — see the module note) and then
/// *executes* `planes12_dot` and `planes12x_row_correction` over every lane of
/// every row, comparing the assembled `y` against an f64 reference built from
/// the **exact** blocks. It can only pass if the correction is the difference
/// the row pass owes.
#[test]
#[cfg_attr(debug_assertions, ignore = "compiles C++ and searches, run in release")]
fn the_planes12x_half_kernel_decides_what_rust_decides() {
    const D_OUT: usize = 32;
    const NBLOCKS: usize = 45;

    let out = std::env::temp_dir().join("llvq_host_planes12x_h");
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
        .arg(dir.join("host_planes12x_h.cpp"))
        .arg("-o")
        .arg(&out)
        .status()
        .expect("clang++ is on PATH");
    assert!(
        st.success(),
        "the half-storing Planes12x source does not compile as host C++"
    );

    let fd = FastDecoder::new();
    let table = ClassTable::new(&fd, 1);
    let (idx, gains) = fixture(&fd, D_OUT, NBLOCKS, 0x12_C0DE);
    let tr = Transcoder::new(FusedLayout::Planes12x).unwrap();
    let (s, _) = tr.stream(&idx, &gains, D_OUT, NBLOCKS).unwrap();
    let p12 = unwrap_p12(&s);
    let (words, exc_idx, exc_words) = (p12.words, p12.exc_idx, p12.exc_words);
    assert!(!exc_idx.is_empty(), "the fixture must exercise the correction");

    let tab = gpu_class_table(&fd);
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
        exc_idx.len() as u32,
        words.len() as u32,
        exc_words.len() as u32,
        tab.len() as u32,
    ]));
    fx.extend_from_slice(&le_u32(words));
    fx.extend_from_slice(&le_u32(exc_idx));
    fx.extend_from_slice(&le_u32(exc_words));
    fx.extend_from_slice(&le_u32(p12.row_exc));
    fx.extend_from_slice(&le_u32(&tab));
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
    assert!(r.is_empty(), "the host probe wrote more than it was asked for");

    // The correction is zero exactly where a row has no exception, and
    // non-zero where it has one. The first half kills a correction pass that
    // fires on every block; the second kills one that fires on none — which
    // is the mutation that would otherwise leave `y` merely *approximate*
    // and still within a loose tolerance on most rows.
    for (row, &c) in corr.iter().enumerate() {
        let n_exc = p12.row_exc[row + 1] - p12.row_exc[row];
        if n_exc == 0 {
            assert_eq!(c, 0.0, "row {row} has no exception, correction not zero");
        } else {
            assert!(c != 0.0, "row {row}: {n_exc} exceptions and a zero correction");
        }
    }

    let mut worst = 0.0f64;
    for row in 0..D_OUT {
        let (mut want, mut scale) = (0.0f64, 0.0f64);
        for col in 0..NBLOCKS {
            let (pt, gain) = overlay_block(&table, p12, row, NBLOCKS, col);
            let Some(shell) = Leech::shell_index(&pt).filter(|&s| s > 0) else {
                continue;
            };
            let k = GSCALE[gain as usize] as f64 * rscale[row] as f64
                / ((16 * shell) as f64).sqrt();
            for (i, &v) in pt.iter().enumerate() {
                let w = v as f64 * k;
                want += w * x[col * DIM + i] as f64;
                scale += (w * x[col * DIM + i] as f64).abs();
            }
        }
        let e = (y[row] as f64 - want).abs() / scale.max(1e-12);
        worst = worst.max(e);
        assert!(e < TOL, "row {row}: gap {e:.2e}·Σ|w·x|");
    }
    println!(
        "planes12x_h host: {D_OUT} rows, {} blocks, {} exceptions, worst gap \
         {worst:.1e}·Σ|w·x|",
        idx.len(),
        exc_idx.len()
    );
}

// ---------------------------------------------------------------------------
// 3. The translation unit — the only part of the CUDA plumbing checkable here
// ---------------------------------------------------------------------------

/// The source list, the concatenation order, and the entry point each layout
/// names.
///
/// This is the one piece of the device wiring a machine without a card can
/// hold to account, and it earns the test: `fused_cuda` sits behind
/// `cfg(all(target_os = "linux", feature = "cuda"))` and compiles on **no**
/// target here — `--target x86_64-unknown-linux-gnu` does not reach it either,
/// because `candle-kernels` needs a real `nvcc`. So the list was moved into
/// `fused` precisely so this could exist. What it catches: a renamed entry
/// point, a part dropped from the list, and the concatenation order NVRTC
/// depends on (it receives one string, and each part's `#ifndef` guard keys on
/// a macro an earlier part defines).
#[test]
fn each_layout_names_its_own_translation_unit() {
    use llvq_llm::fused::{
        load_planes_sources, matvec_kernel_name, planes_source_names, seg_kernel_name,
    };

    assert_eq!(planes_source_names(FusedLayout::Slot32), &[] as &[&str]);
    assert_eq!(matvec_kernel_name(FusedLayout::Slot32), "tv_slot_h");
    assert_eq!(matvec_kernel_name(FusedLayout::Planes14), "tv_planes_h");
    assert_eq!(matvec_kernel_name(FusedLayout::Planes12x), "tv_planes12x_h");
    // A second entry point beside the first, on the one layout that can be
    // segmented. `None` elsewhere is not a scoping choice — see
    // `seg_kernel_name` for the three silent mechanisms.
    assert_eq!(seg_kernel_name(FusedLayout::Planes14), Some("tv_planes_seg_h"));
    for layout in [FusedLayout::Slot32, FusedLayout::Planes12x, FusedLayout::Golay70] {
        assert_eq!(seg_kernel_name(layout), None, "{layout:?} does not segment");
    }

    let p14 = planes_source_names(FusedLayout::Planes14);
    let p12 = planes_source_names(FusedLayout::Planes12x);
    // `tv_planes_seg_h.cu` at its exact position, right after the kernel it is
    // derived from and whose helpers it reuses. It is in **every** list even
    // on the layouts that never launch it, so all three builds keep one shape
    // of translation unit and `tv_planes_h`'s register report stays a drift
    // detector on all of them. See `planes_source_names` for what that costs.
    assert_eq!(
        p14,
        ["llvq_planes.cuh", "planes.cu", "tv_planes_h.cu", "tv_planes_seg_h.cu"]
    );
    // Planes12x extends Planes14's unit rather than replacing it — that is
    // what keeps `tv_planes_h`'s register report a drift detector on this
    // build too, and it means the shared prefix cannot diverge silently.
    assert_eq!(&p12[..p14.len()], p14, "planes12x must extend planes14");
    assert_eq!(&p12[p14.len()..], ["llvq_planes12.cuh", "tv_planes12x_h.cu"]);

    for layout in [FusedLayout::Slot32, FusedLayout::Planes14, FusedLayout::Planes12x] {
        let (parts, overridden) = load_planes_sources(layout).expect("embedded copies");
        assert!(overridden.is_none(), "LLVQ_KERNEL_DIR must not be set here");
        assert_eq!(parts.len(), planes_source_names(layout).len());
        // Each embedded part is the file it claims to be, and the whole unit
        // defines the entry point the launcher will look up by name.
        let unit = parts.concat();
        if !parts.is_empty() {
            let entry = format!("void {}(", matvec_kernel_name(layout));
            assert!(
                unit.contains(&entry),
                "{layout:?}: the translation unit does not define {entry}"
            );
            // …and the segmented entry point, on every bit-plane layout,
            // launched or not. ⚠️ This is a substring search, not a
            // compilation: `tests/host_planes_seg_h.cpp` is what puts the unit
            // through `clang++ -Werror` in this very order.
            assert!(
                unit.contains("void tv_planes_seg_h("),
                "{layout:?}: the translation unit does not define tv_planes_seg_h"
            );
        }
    }

    // The 12x unit must carry the decoder its kernel calls, and the guard the
    // concatenation relies on to skip the file's own `#include`.
    let unit = load_planes_sources(FusedLayout::Planes12x).unwrap().0.concat();
    for symbol in [
        "planes12_fields",
        "planes12_dot",
        "planes12x_lane_terms",
        "planes12x_combine",
        "planes12x_row_correction",
        "LLVQ_PLANES12_CUH",
    ] {
        assert!(unit.contains(symbol), "the planes12x unit carries no {symbol}");
    }
    // And it must NOT carry the bench arm's kernel: `tv_planes12x` memsets `y`
    // and accumulates with atomicAdd, which is exactly the protocol this path
    // replaces. Compiling it here would put a second, incompatible entry point
    // of nearly the same name in the module.
    assert!(
        !unit.contains("void tv_planes12x("),
        "the inference unit embeds the bench kernel tv_planes12x"
    );
}

// ---------------------------------------------------------------------------
// 4. The real artifact — the proof that counts, and the load-time chrono
// ---------------------------------------------------------------------------

/// Where to look when the sealed model lives somewhere other than `$HOME`.
///
/// A pointer, deliberately not an escape hatch — the same contract as
/// `llvq-artifact/tests/common`: it can move the search, never satisfy it.
const MODEL_ENV: &str = "LLVQ_SEALED_MODEL";
const MODEL_DEFAULT: &str = "qwen3-4b-llvq.bin";

fn sealed_model_path() -> PathBuf {
    let howto = format!(
        "This sweep checks the Planes12x stream this crate produces against the \
         Planes14 stream, block by block, on a real sealed model. It is #[ignore]d, so \
         being here means it was asked for: it fails rather than print SKIP \
         and return `ok`.\n\n\
         Point {MODEL_ENV} at a SELF-CONTAINED .llvq (not a projections-only \
         artifact):\n\n    \
         {MODEL_ENV}=/path/qwen3-8b-llvq.bin \\\\\n        \
         cargo test --release -p llvq-llm --test fused_planes12x -- --include-ignored"
    );
    let (path, origin) = match (std::env::var_os(MODEL_ENV), std::env::var_os("HOME")) {
        (Some(p), _) => (PathBuf::from(p), MODEL_ENV.to_string()),
        (None, Some(h)) => (
            Path::new(&h).join(MODEL_DEFAULT),
            format!("$HOME/{MODEL_DEFAULT}"),
        ),
        (None, None) => panic!("neither ${MODEL_ENV} nor $HOME is set.\n\n{howto}"),
    };
    assert!(
        path.exists(),
        "the sealed model is not on this machine: {} (from {origin})\n\n{howto}",
        path.display()
    );
    path
}

/// The integral proof: on a real sealed model, the `Planes12x` words this
/// crate ships — main stream, exception table and `row_exc` — rebuild the
/// `Planes14` words block for block, read through the kernel's own windows.
///
/// It also **times** both transcodes, which is the load-time cost the arm
/// carries in production: `Planes12x` runs one exact `nearest_angular` search
/// per 5-level block on top of the same work as `Planes14`, and that number
/// belongs next to the b/weight it buys.
///
/// ⚠️ The longest test in this crate by two orders of magnitude — tens of
/// minutes on the 4B, more on the 8B. Out of the default run for that reason;
/// asked for explicitly, a missing model is an error, never a green skip.
#[test]
#[ignore = "tens of minutes of searches on a real sealed model (--include-ignored)"]
fn transcode_of_the_sealed_model_matches_planes14() {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::Instant;

    let path = sealed_model_path();
    let fd = FastDecoder::new();
    let table = ClassTable::new(&fd, 1);
    let t12 = Transcoder::new(FusedLayout::Planes12x).expect("planes12x transcoder");
    let t14 = Transcoder::new(FusedLayout::Planes14).expect("planes14 transcoder");

    let f = std::fs::File::open(&path).expect("open the sealed model");
    let mut r = std::io::BufReader::with_capacity(1 << 20, f);
    let h = llvq_artifact::read_header(&mut r).expect("valid artifact header");
    assert!(
        h.is_self_contained(),
        "{} is only a projections artifact (v{}), fused::load would refuse it",
        path.display(),
        h.version
    );

    let nthreads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let (mut blocks, mut exceptions) = (0u64, 0u64);
    let (mut b12, mut b14) = (0u64, 0u64);
    let (mut secs12, mut secs14, mut secs_read) = (0.0f64, 0.0f64, 0.0f64);
    let wall = Instant::now();

    for _ in 0..h.matrices {
        let t = Instant::now();
        let m = llvq_artifact::read_matrix_raw(&mut r).expect("valid matrix");
        secs_read += t.elapsed().as_secs_f64();
        let nblocks = m.d_in / DIM;
        assert_eq!(
            m.indices.len(),
            m.d_out * nblocks,
            "{}: {} codes for {}×{nblocks}",
            m.name,
            m.indices.len(),
            m.d_out
        );

        let t = Instant::now();
        let (s12, by12) = t12
            .stream(&m.indices, &m.gains, m.d_out, nblocks)
            .unwrap_or_else(|e| panic!("{} : {e}", m.name));
        secs12 += t.elapsed().as_secs_f64();
        let t = Instant::now();
        let (s14, by14) = t14
            .stream(&m.indices, &m.gains, m.d_out, nblocks)
            .unwrap_or_else(|e| panic!("{} : {e}", m.name));
        secs14 += t.elapsed().as_secs_f64();
        b12 += by12;
        b14 += by14;

        let p12 = unwrap_p12(&s12);
        let p14 = unwrap_p14(&s14);
        exceptions += p12.exc_idx.len() as u64;
        blocks += m.indices.len() as u64;

        // Rows are independent, so the sweep splits on them; `row_exc` makes
        // each thread's slice of the exception table self-contained, which is
        // the same property the kernel's grid relies on.
        let seen = AtomicU64::new(0);
        let chunk = m.d_out.div_ceil(nthreads);
        std::thread::scope(|sc| {
            for c in 0..nthreads {
                let (table, name, seen) = (&table, &m.name, &seen);
                sc.spawn(move || {
                    let lo = c * chunk;
                    let hi = ((c + 1) * chunk).min(m.d_out);
                    let mut n = 0u64;
                    for row in lo..hi {
                        for col in 0..nblocks {
                            let b = row * nblocks + col;
                            let want = planes14_block(table, p14, b);
                            let got = overlay_block(table, p12, row, nblocks, col);
                            assert_eq!(got, want, "{name} block {b}: overlay ≠ Planes14");
                            n += 1;
                        }
                    }
                    seen.fetch_add(n, Ordering::Relaxed);
                });
            }
        });
        assert_eq!(
            seen.load(Ordering::Relaxed) as usize,
            m.indices.len(),
            "{}: the sweep skipped blocks",
            m.name
        );
    }

    let bpw = |bytes: u64| bytes as f64 * 8.0 / (blocks * DIM as u64) as f64;
    println!(
        "\n{} — {} matrices, {blocks} blocks checked block by block\n  \
         exceptions : {exceptions} ({:.4}% of blocks)\n  \
         payload    : Planes12x {:.4} b/weight against Planes14 {:.4}\n  \
         transcode  : Planes12x {:.1} s, Planes14 {:.1} s, read {:.1} s, \
         total {:.1} s ({nthreads} cores)",
        path.display(),
        h.matrices,
        exceptions as f64 * 100.0 / blocks as f64,
        bpw(b12),
        bpw(b14),
        secs12,
        secs14,
        secs_read,
        wall.elapsed().as_secs_f64(),
    );
    assert!(exceptions > 0, "no exception: the sweep exercised nothing");
}
