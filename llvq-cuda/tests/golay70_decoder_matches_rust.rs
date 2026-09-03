//! The Golay70 CUDA decoder and its exception correction, compiled by
//! `clang++` and diffed against Rust — the same locks as the Planes12x test,
//! plus the hand-packed record probe of the C1 review:
//!
//! 1. **The reconstruction is exact, in pure Rust.** Every block of a
//!    synthetic matrix must decode through `golay70_decode_block` to exactly
//!    what the Slot32 stream carries — exceptions included — and the
//!    exception set must be exactly the blocks whose class violates the E2
//!    rule (5 levels either coset, or an even class with > 2 |values| in a
//!    residue mod 4).
//!
//! 2. **The kernel text decides what Rust decides.** `golay70_fields`,
//!    `golay70_slot_value`, `golay70_dot`, `planes12x_locate`,
//!    `golay70_exc_lane_term` and `golay70_exc_combine` — the same text NVRTC
//!    will compile, through the same shim — are executed per block, per slot
//!    and per exception, and the full `y` they produce (row pass +
//!    corrections, the emulated launch) is checked row by row against an f64
//!    reference built from the **exact** Slot32 decode. It can only pass if
//!    the origin-carrying main stream plus the correction pass reproduce the
//!    exact weights. Including golay70.cu also compiles `tv_golay70`, so a
//!    syntax or type error costs two seconds here instead of a billed job.
//!
//! 3. **Hand-packed records decode per the frozen spec.** Records are packed
//!    bit by bit from the literal offsets [class:9][gain:1][golay:12][A:24]
//!    [B:24] at four consecutive blocks — one per window shift sh ∈ {0, 8,
//!    16, 24} — with an independent per-bit setter, and decoded both by the
//!    Rust reference and by the kernel text. This is the probe that catches
//!    a *coherent* writer+reader offset drift, which every round-trip test
//!    is structurally blind to (the C1 reviewer's M6 precedent).
//!
//! What is left for the card: occupancy, the atomicity of atomicAdd under a
//! real schedule, the memset-before-launch protocol, and registers/spill
//! (`local_bytes == 0` is read at bench startup).

use llvq_artifact::runtime::{transcode, ClassTable, Layout};
use llvq_core::{SplitMix64, DIM};
use llvq_search::fastdec::FastDecoder;
use llvq_search::index::N13;
use std::io::Write;
use std::process::{Command, Stdio};

include!("../src/golay70_host.rs");

const GSCALE: [f32; 2] = [0.625, 1.375];
const TABLE_ENTRIES: usize = 512;
const REC_WORDS: usize = 6;
const D_OUT: usize = 32;
const NBLOCKS: usize = 45;
const TOL: f64 = 1e-5;

/// The synthetic matrix: uniform draws over the cap-13 ball, with both ends
/// of every class and the origin scattered through the stream at a fixed
/// cadence — so every class id (E2 violators included) crosses several rows.
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

/// The +4-byte tail padding of the frozen spec, then word packing.
fn pad_words(mut bytes: Vec<u8>) -> Vec<u32> {
    bytes.extend_from_slice(&[0u8; 4]);
    while !bytes.len().is_multiple_of(4) {
        bytes.push(0);
    }
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// The 512-entry ClassRec table the exception path reads — verbatim the
/// bench's construction.
fn class_rec_table(fd: &FastDecoder) -> Vec<u32> {
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

fn compile_probe() -> std::path::PathBuf {
    let out = std::env::temp_dir().join("llvq_host_golay70");
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
        .arg(dir.join("host_golay70.cpp"))
        .arg("-o")
        .arg(&out)
        .status()
        .expect("clang++ is on PATH");
    assert!(st.success(), "the Golay70 CUDA sources do not compile as host C++");
    out
}

struct ProbeOut {
    slots: Vec<f32>,
    cls: Vec<u32>,
    y: Vec<f32>,
}

#[allow(clippy::too_many_arguments)]
fn run_probe(
    bin: &std::path::Path,
    d_out: usize,
    nblocks: usize,
    blocks: &Golay70Blocks,
    tab: &[u32],
    gtab: &[u32],
    cw: &[u32],
    rscale: &[f32],
    x: &[f32],
) -> ProbeOut {
    let words = pad_words(blocks.data.clone());
    let exc_words = pad_words(blocks.exc_data.clone());
    let n_exc = blocks.exc_idx.len();

    let mut fixture = Vec::new();
    fixture.extend_from_slice(&le_u32(&[
        d_out as u32,
        nblocks as u32,
        n_exc as u32,
        words.len() as u32,
        exc_words.len() as u32,
        tab.len() as u32,
        gtab.len() as u32,
        cw.len() as u32,
    ]));
    fixture.extend_from_slice(&le_u32(&words));
    fixture.extend_from_slice(&le_u32(&blocks.exc_idx));
    fixture.extend_from_slice(&le_u32(&exc_words));
    fixture.extend_from_slice(&le_u32(tab));
    fixture.extend_from_slice(&le_u32(gtab));
    fixture.extend_from_slice(&le_u32(cw));
    fixture.extend_from_slice(&le_f32(&GSCALE));
    fixture.extend_from_slice(&le_f32(rscale));
    fixture.extend_from_slice(&le_f32(x));

    let mut child = Command::new(bin)
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

    let n = d_out * nblocks;
    let mut r = res.stdout.as_slice();
    let take = |r: &mut &[u8], k: usize| -> Vec<u8> {
        let (a, b) = r.split_at(k * 4);
        *r = b;
        a.to_vec()
    };
    let slots = take(&mut r, n * DIM)
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let cls = take(&mut r, n * 3)
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let y = take(&mut r, d_out)
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert!(r.is_empty(), "the host probe wrote more than it was asked to");
    ProbeOut { slots, cls, y }
}

/// The codeword of a decoded point, per coset — the independent path the
/// grank field is checked against.
fn codeword_of(p: &llvq_core::Point, odd: bool) -> u32 {
    let mut cw = 0u32;
    for (i, &v) in p.iter().enumerate() {
        let hit = if odd { v.rem_euclid(4) == 3 } else { v.abs() % 4 == 2 };
        if hit {
            cw |= 1 << i;
        }
    }
    cw
}

#[test]
#[cfg_attr(debug_assertions, ignore = "sweeps the ball, run in release")]
fn golay70_reconstruction_is_exact_against_slot32() {
    let fd = FastDecoder::new();
    let golay = llvq_core::Golay::new();
    let table = ClassTable::new(&fd, 1);
    let cls = golay70_classes(&fd);
    let mut rng = SplitMix64::new(0x60_1AC0);

    // Both ends of every class + 20,000 uniform draws + the origin: every
    // class id — E2 violators included — crosses the stream.
    let mut indices: Vec<u64> = Vec::new();
    for ci in 0..fd.n_classes() {
        let (first, last) = fd.class_range(ci);
        indices.push(first);
        indices.push(last);
    }
    indices.extend((0..20_000).map(|_| 1 + rng.next() % N13));
    indices.push(0);
    let gains: Vec<u32> = indices.iter().map(|_| (rng.next() & 1) as u32).collect();
    let n = indices.len();

    let slot = transcode(&fd, &table, &indices, &gains, Layout::Slot32).expect("transcodes");
    let g70 = golay70_transcode(&fd, &golay, &cls, &table, &indices, &gains);
    assert_eq!(g70.data.len(), n * GOLAY70_STRIDE, "9 bytes per block, exactly");
    assert!(!g70.exc_idx.is_empty(), "fixture must exercise the exception path");

    for (b, &idx) in indices.iter().enumerate() {
        let want = slot.decode_block(&table, b);
        let got = golay70_decode_block(&g70, &cls, &golay, &table, b);
        assert_eq!(got, want, "block {b}: Golay70 disagrees with Slot32");
        // The exception set is exactly the E2 rule, block by block.
        let id = fd.class_of(idx).map_or(0, |ci| 1 + ci);
        let is_exc = g70.exc_idx.binary_search(&(b as u32)).is_ok();
        assert_eq!(
            is_exc,
            cls[id].is_exc,
            "block {b}: exception status disagrees with the class rule"
        );
    }
    let bpw = (g70.data.len() * 8 + g70.exc_idx.len() * GOLAY70_EXC_BYTES * 8) as f64
        / (n * DIM) as f64;
    println!(
        "golay70: {n} blocks, {} exceptions, {bpw:.4} b/w payload",
        g70.exc_idx.len()
    );
}

#[test]
#[cfg_attr(debug_assertions, ignore = "compiles C++ and sweeps, run in release")]
fn the_golay70_kernel_decides_what_the_rust_decoder_decides() {
    let bin = compile_probe();

    // ---- fixture ----
    let fd = FastDecoder::new();
    let golay = llvq_core::Golay::new();
    let table = ClassTable::new(&fd, 1);
    let cls = golay70_classes(&fd);
    let mut rng = SplitMix64::new(0x60_1AC0);
    let indices = fixture_indices(&fd, &mut rng);
    let gains: Vec<u32> = indices.iter().map(|_| (rng.next() & 1) as u32).collect();
    let n = indices.len();

    let slot = transcode(&fd, &table, &indices, &gains, Layout::Slot32).expect("transcodes");
    let g70 = golay70_transcode(&fd, &golay, &cls, &table, &indices, &gains);
    let n_exc = g70.exc_idx.len();
    assert!(n_exc > 0, "fixture must exercise the correction pass");
    // At least one row must carry several exceptions: the accumulation
    // (vs a store) is only observable then and on rows with row-pass mass.
    let rows_of_exc: Vec<u32> = g70.exc_idx.iter().map(|&b| b / NBLOCKS as u32).collect();
    assert!(
        rows_of_exc.windows(2).any(|w| w[0] == w[1]),
        "fixture must put two exceptions on one row"
    );

    let tab = class_rec_table(&fd);
    let gtab = golay70_gpu_table(&cls);
    let cw = golay70_gpu_codewords(&golay);
    let rscale: Vec<f32> = (0..D_OUT)
        .map(|_| 0.5 + rng.next_gaussian().abs() as f32)
        .collect();
    let x: Vec<f32> = (0..NBLOCKS * DIM).map(|_| rng.next_gaussian() as f32).collect();

    let out = run_probe(&bin, D_OUT, NBLOCKS, &g70, &tab, &gtab, &cw, &rscale, &x);

    // ---- lock A: fields and per-slot values, block by block ----
    for b in 0..n {
        let id = fd.class_of(indices[b]).map_or(0, |ci| 1 + ci);
        let is_exc = g70.exc_idx.binary_search(&(b as u32)).is_ok();
        let main_id = if is_exc { 0 } else { id };
        assert_eq!(out.cls[b * 3] as usize, main_id, "block {b}: class id");
        assert_eq!(out.cls[b * 3 + 1], gains[b], "block {b}: gain bit");
        if main_id == 0 {
            // Origin or exception: canonical zero rank, and every slot must
            // contribute exactly zero through the all-zero table entry.
            assert_eq!(out.cls[b * 3 + 2], 0, "block {b}: origin golay rank");
            for j in 0..DIM {
                assert_eq!(
                    out.slots[b * DIM + j], 0.0,
                    "block {b} slot {j}: origin/exception slot must be zero"
                );
            }
            continue;
        }
        let (pt, _) = slot.decode_block(&table, b);
        let want_cw = codeword_of(&pt, cls[id].is_odd);
        let want_rank = golay.rank(want_cw).expect("residue plane is a codeword");
        assert_eq!(
            out.cls[b * 3 + 2] as usize,
            want_rank,
            "block {b}: golay rank"
        );
        let norm = ((16 * cls[id].shell) as f64).sqrt();
        for (j, &pj) in pt.iter().enumerate() {
            // Same rounding as the uploaded table (|v|/norm as f32, then an
            // exact IEEE negation), so the comparison is bit-strict.
            let want = ((pj.abs() as f64 / norm) as f32) * if pj < 0 { -1.0 } else { 1.0 };
            assert_eq!(
                out.slots[b * DIM + j],
                want,
                "block {b} slot {j}: decoded value"
            );
        }
    }

    // ---- lock B: the emulated launch equals the exact f64 reference ----
    // The reference decodes the EXACT blocks from the Slot32 stream; the
    // harness computed the origin-holed row pass plus the correction pass
    // from the kernel's own device functions. Equality within TOL·Σ|w·x| is
    // the proof that main stream + corrections reconstruct exactly — the
    // host twin of the bench's pre-timing verification.
    let mut worst = 0.0f64;
    for (row, &rs) in rscale.iter().enumerate() {
        let (mut want, mut scale) = (0.0f64, 0.0f64);
        for j in 0..NBLOCKS {
            let (pt, gain) = slot.decode_block(&table, row * NBLOCKS + j);
            if let Some(shell) = llvq_core::Leech::shell_index(&pt).filter(|&s| s > 0) {
                let k = GSCALE[gain as usize] as f64 * rs as f64
                    / ((16 * shell) as f64).sqrt();
                for (i, &v) in pt.iter().enumerate() {
                    let w = v as f64 * k;
                    want += w * x[j * DIM + i] as f64;
                    scale += (w * x[j * DIM + i] as f64).abs();
                }
            }
        }
        let e = (out.y[row] as f64 - want).abs() / scale.max(1e-12);
        worst = worst.max(e);
        assert!(
            e < TOL,
            "row {row}: y differs from the exact reference by {e:.2e}·Σ|w·x|"
        );
    }
    println!(
        "golay70 host launch: {D_OUT} rows, {n} blocks, {n_exc} exceptions, \
         worst {worst:.1e}·Σ|w·x|"
    );
}

/// The integral proof on the sealed 4B artifact: every block of every matrix,
/// decoded through Golay70 (exceptions resolved) and through Slot32, must
/// equal the archive decoder bit for bit — the central claim of the layout on
/// the exact object the bench will time. Skipped (loudly) when the file is
/// not on this machine; ignored in debug like every heavy sweep in this repo.
#[test]
#[cfg_attr(debug_assertions, ignore = "full 150 M-block artifact, run in release")]
fn the_sealed_artifact_decodes_identically_through_golay70() {
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

    let path = match std::env::var_os("HOME") {
        Some(h) => std::path::Path::new(&h).join("llvq-q4b.llvq"),
        None => {
            eprintln!("SKIP: no $HOME to find the sealed artifact");
            return;
        }
    };
    if !path.exists() {
        eprintln!("SKIP: {} not present on this machine", path.display());
        return;
    }

    let fd = FastDecoder::new();
    let golay = llvq_core::Golay::new();
    let table = ClassTable::new(&fd, 1);
    let cls = golay70_classes(&fd);
    let f = std::fs::File::open(&path).expect("open the sealed artifact");
    let mut r = std::io::BufReader::new(f);
    let h = llvq_artifact::read_header(&mut r).expect("valid artifact header");

    let nthreads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    // Work units aligned to the Slot32 group so chunked transcodes see whole
    // groups; only decoded points are compared, never chunk bytes. The
    // Golay70 side transcodes through `golay70_chunk` directly (b0 = 0, so
    // exception indices are chunk-local) rather than the threaded wrapper —
    // the outer loop already saturates the cores.
    const CHUNK: usize = 32 * 4096;
    let verified = AtomicU64::new(0);
    let exceptions = AtomicU64::new(0);

    for _ in 0..h.matrices {
        let m = llvq_artifact::read_matrix_raw(&mut r).expect("valid matrix");
        assert!(m.shell_cap <= 13, "{}: cap {}", m.name, m.shell_cap);
        assert_eq!(m.centroids.len(), 2, "{}: not a 1-bit-gain matrix", m.name);
        let nchunks = m.indices.len().div_ceil(CHUNK);
        let next = AtomicUsize::new(0);
        std::thread::scope(|s| {
            for _ in 0..nthreads {
                s.spawn(|| loop {
                    let c = next.fetch_add(1, Ordering::Relaxed);
                    if c >= nchunks {
                        break;
                    }
                    let lo = c * CHUNK;
                    let hi = ((c + 1) * CHUNK).min(m.indices.len());
                    let idxs = &m.indices[lo..hi];
                    let gns = &m.gains[lo..hi];
                    let (data, exc_idx, exc_data) =
                        golay70_chunk(&fd, &golay, &cls, &table, idxs, gns, 0);
                    let n_exc = exc_idx.len();
                    let g70 = Golay70Blocks {
                        n_blocks: idxs.len(),
                        data,
                        exc_idx,
                        exc_data,
                    };
                    let slot = transcode(&fd, &table, idxs, gns, Layout::Slot32)
                        .expect("transcodes");
                    for (b, &idx) in idxs.iter().enumerate() {
                        let want = match fd.decode(idx) {
                            Some(p) => p,
                            None => {
                                assert_eq!(idx, 0, "{}: index outside the ball", m.name);
                                [0i32; DIM]
                            }
                        };
                        let (pg, gg) = golay70_decode_block(&g70, &cls, &golay, &table, b);
                        let (ps, gs) = slot.decode_block(&table, b);
                        assert_eq!(pg, want, "{} block {}: Golay70", m.name, lo + b);
                        assert_eq!(ps, want, "{} block {}: Slot32", m.name, lo + b);
                        assert_eq!(gg, gns[b], "{} block {}: gain", m.name, lo + b);
                        assert_eq!(gs, gns[b], "{} block {}: gain", m.name, lo + b);
                    }
                    verified.fetch_add((hi - lo) as u64, Ordering::Relaxed);
                    exceptions.fetch_add(n_exc as u64, Ordering::Relaxed);
                });
            }
        });
    }
    let total = verified.load(Ordering::Relaxed);
    let exc = exceptions.load(Ordering::Relaxed);
    println!(
        "sealed sweep: {total} blocks verified identical in Golay70 and Slot32, \
         {exc} exceptions ({:.4} %), payload {:.4} b/w",
        exc as f64 * 100.0 / total as f64,
        (total * 72 + exc * 144) as f64 / (total * DIM as u64) as f64
    );
}

/// Set bits `[at, at+width)` of an LSB-first byte stream — the independent
/// packer of the hand-packed probe: per-bit, from the spec's literal offsets,
/// sharing no code with the transcoder's u128 composition.
fn poke_bits(data: &mut [u8], at: usize, width: usize, value: u64) {
    assert!(width == 64 || value < 1u64 << width, "value overflows width");
    for i in 0..width {
        if value >> i & 1 == 1 {
            data[(at + i) / 8] |= 1 << ((at + i) % 8);
        }
    }
}

#[test]
#[cfg_attr(debug_assertions, ignore = "compiles C++, run in release")]
fn hand_packed_records_decode_per_spec() {
    let bin = compile_probe();
    let fd = FastDecoder::new();
    let golay = llvq_core::Golay::new();
    let table = ClassTable::new(&fd, 1);
    let cls = golay70_classes(&fd);

    // One odd and one even non-exception class, the richest available.
    let odd_id = (1..cls.len())
        .find(|&id| cls[id].is_odd && !cls[id].is_exc && cls[id].len == 4)
        .expect("an odd 4-level class exists in the ball");
    let even_id = (1..cls.len())
        .find(|&id| !cls[id].is_odd && !cls[id].is_exc && cls[id].res_len == [2, 2])
        .expect("an even class with two full residue pairs exists in the ball");

    // Four blocks — one per window shift sh ∈ {0, 8, 16, 24} — alternating
    // cosets, with asymmetric bit patterns so a swapped or shifted field
    // cannot cancel out.
    let granks: [usize; 4] = [1, 4095, 1234, 2048];
    let ids = [odd_id, even_id, odd_id, even_id];
    let gains_hp: [u32; 4] = [0, 1, 1, 0];
    let a_raw: [u32; 4] = [0xA5A5A5, 0x0F0F0F, 0x333333, 0xFFFFFF];
    let b_raw: [u32; 4] = [0x3C3C3C, 0x924924, 0x000001, 0x800000];

    let nblocks = 4usize;
    let mut data = vec![0u8; nblocks * GOLAY70_STRIDE];
    let mut expected: Vec<[i32; DIM]> = Vec::new();
    for b in 0..nblocks {
        let c = &cls[ids[b]];
        let cw = golay.codewords()[granks[b]];
        // Canonicalize the raw patterns against the class's structure, and
        // compute the expected point by applying the spec's rules directly.
        let (mut a_plane, mut b_plane) = (0u32, 0u32);
        let mut pt = [0i32; DIM];
        for (j, pj) in pt.iter_mut().enumerate() {
            let cbit = cw >> j & 1;
            if c.is_odd {
                let lev = ((a_raw[b] >> j & 1) | (b_raw[b] >> j & 1) << 1) as usize % c.len;
                a_plane |= ((lev as u32) & 1) << j;
                b_plane |= ((lev as u32) >> 1) << j;
                let v = c.vals[lev];
                let neg = cbit ^ (c.flags >> lev & 1);
                *pj = if neg == 1 { -v } else { v };
            } else {
                let ri = cbit as usize;
                let abit = if c.res_len[ri] == 2 { a_raw[b] >> j & 1 } else { 0 };
                let v = c.vals[ri * 2 + abit as usize];
                let bbit = if v == 0 { 0 } else { b_raw[b] >> j & 1 };
                a_plane |= abit << j;
                b_plane |= bbit << j;
                *pj = if bbit == 1 { -v } else { v };
            }
        }
        expected.push(pt);
        // The frozen offsets, literally: 0 / 9 / 10 / 22 / 46, stride 9·b.
        let bit0 = b * 72;
        poke_bits(&mut data, bit0, 9, ids[b] as u64);
        poke_bits(&mut data, bit0 + 9, 1, u64::from(gains_hp[b]));
        poke_bits(&mut data, bit0 + 10, 12, granks[b] as u64);
        poke_bits(&mut data, bit0 + 22, 24, u64::from(a_plane));
        poke_bits(&mut data, bit0 + 46, 24, u64::from(b_plane));
    }
    let g70 = Golay70Blocks {
        n_blocks: nblocks,
        data,
        exc_idx: Vec::new(),
        exc_data: Vec::new(),
    };

    // ---- the Rust reference decodes the hand-packed stream per spec ----
    for b in 0..nblocks {
        let (pt, gain) = golay70_decode_block(&g70, &cls, &golay, &table, b);
        assert_eq!(gain, gains_hp[b], "block {b}: gain");
        assert_eq!(pt, expected[b], "block {b}: hand-packed point");
    }

    // ---- the kernel text decodes the same bytes to the same values ----
    let tab = class_rec_table(&fd);
    let gtab = golay70_gpu_table(&cls);
    let cw = golay70_gpu_codewords(&golay);
    let rscale = vec![1.0f32];
    let x = vec![0.0f32; nblocks * DIM];
    let out = run_probe(&bin, 1, nblocks, &g70, &tab, &gtab, &cw, &rscale, &x);
    for b in 0..nblocks {
        assert_eq!(out.cls[b * 3] as usize, ids[b], "block {b}: class id");
        assert_eq!(out.cls[b * 3 + 1], gains_hp[b], "block {b}: gain");
        assert_eq!(out.cls[b * 3 + 2] as usize, granks[b], "block {b}: golay rank");
        let norm = ((16 * cls[ids[b]].shell) as f64).sqrt();
        for (j, &v) in expected[b].iter().enumerate() {
            let want = ((v.abs() as f64 / norm) as f32) * if v < 0 { -1.0 } else { 1.0 };
            assert_eq!(out.slots[b * DIM + j], want, "block {b} slot {j}");
        }
    }
}
