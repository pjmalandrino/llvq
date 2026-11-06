//! The row-loop twins of the served Tetra matvec on Metal
//! (`kernels/llvq_tetra48_rows.metal`), reached through candle, against the
//! served kernel and the CPU oracle arm, every f32 equal.
//!
//! `_lutp` keeps block j + 32's words in flight while block j decodes, the
//! port of 4E v4's row loop. `_lutd` reads the activation from device memory
//! with no staging, no tile and no barrier. `_lutdp` is both. None of them
//! changes a lane's blocks or their order, so none may change a bit; this
//! file is where that is demanded rather than argued.
//!
//! The shapes are the sealed 4B's Tetra shapes (d_in 2,560: q, k, gate, up;
//! 9,728: down@0-11) and shapes that stress the loop edges: a lane with no
//! block, a last tile of one block, tiles 32 and 256. The served kernel runs
//! at fast math on, as the shipped path compiles it, and so do the twins.
//!
//! The half-io twins (`+h`): the same matvecs reading and storing f16, against
//! the served kernels narrowed by candle's cast, every f16 equal.
//!
//! Mutation run (2026-11-06, journal `docs/mesures/metal-nuit-2026-11-06.txt`).

#![cfg(all(target_os = "macos", feature = "metal"))]

use candle_core::{Device, Tensor};
use llvq_artifact::tetra48::{transcode_tetra48, TETRA48_SHELLS};
use llvq_bench::f1::rank::{branch_words, prefix_bytes, suffix_bytes, RankTable};
use llvq_bench::f1::Trellis;
use llvq_core::{SplitMix64, DIM};
use llvq_llm::fused_metal::{MetalRuntime, MetalTetraProj, MetalTuning, SERVED_TETRA, TETRA_ROW_LOOPS};

fn invnorm_table() -> Vec<f32> {
    let mut t = vec![0.0f32; TETRA48_SHELLS];
    for (m, e) in t.iter_mut().enumerate().skip(1) {
        *e = (1.0f64 / ((16 * m) as f64).sqrt()) as f32;
    }
    t
}

/// Round to nearest even, f32 to binary16 bits, normal range only (the
/// fixture draws no subnormal tail value).
fn f16_bits(v: f32) -> u16 {
    half::f16::from_f32(v).to_bits()
}

struct Bench {
    rt: MetalRuntime,
    proj: MetalTetraProj,
    x: Vec<f32>,
}

fn build(seed: u64, d_out: usize, nblocks: usize, tail_w: usize, tile: usize) -> Bench {
    let dev = Device::new_metal(0).expect("a Metal device");
    let mut rng = SplitMix64::new(seed);
    let n = d_out * nblocks;
    let indices: Vec<u64> = (0..n)
        .map(|_| match rng.next().is_multiple_of(6) {
            true => 0,
            false => 1 + rng.next() % (llvq_search::index::N13.min(1u64 << 47) - 1),
        })
        .collect();
    let gains: Vec<u32> = (0..n).map(|_| (rng.next() & 1) as u32).collect();
    let stream = transcode_tetra48(&indices, &gains, d_out, nblocks).expect("transcodes");
    let table = RankTable::build();
    let tr = Trellis::new();
    let rt = MetalRuntime::new(
        &dev,
        &table.rows,
        &prefix_bytes(&tr),
        &branch_words(&tr),
        &suffix_bytes(&tr),
        &invnorm_table(),
        tile,
        true,
    )
    .expect("the runtime builds");
    let rscale: Vec<f32> = (0..d_out).map(|_| 0.5 + rng.next_gaussian().abs() as f32).collect();
    let tail: Vec<u16> = (0..d_out * tail_w).map(|_| f16_bits(rng.next_gaussian() as f32)).collect();
    let d_in = nblocks * DIM + tail_w;
    let proj = rt
        .upload("000.gate_proj", d_out, d_in, nblocks, stream.stride_u32, &stream.data, &[0.625, 1.375], &rscale, &tail)
        .expect("the projection uploads");
    // f16-representable, as the rotation hands the matvec its input.
    let x: Vec<f32> = (0..d_in).map(|_| half::f16::from_f32(rng.next_gaussian() as f32).to_f32()).collect();
    Bench { rt, proj, x }
}

fn on_metal(b: &Bench) -> Vec<f32> {
    let dev = Device::new_metal(0).expect("a Metal device");
    let xm = Tensor::from_slice(&b.x, (1, b.x.len()), &dev).expect("upload x");
    b.rt.matvec(&b.proj, &xm).expect("metal_fwd").flatten_all().unwrap().to_vec1::<f32>().unwrap()
}

fn on_cpu(b: &Bench) -> Vec<f32> {
    let xc = Tensor::from_slice(&b.x, (1, b.x.len()), &Device::Cpu).expect("host x");
    b.rt.matvec(&b.proj, &xc).expect("cpu_fwd").flatten_all().unwrap().to_vec1::<f32>().unwrap()
}

fn first_difference(a: &[f32], b: &[f32]) -> Option<(usize, f32, f32)> {
    assert_eq!(a.len(), b.len(), "lengths");
    a.iter().zip(b).enumerate().find(|(_, (p, q))| p.to_bits() != q.to_bits()).map(|(i, (p, q))| (i, *p, *q))
}

fn with_kernel(b: &mut Bench, name: &'static str) {
    b.rt.set_tuning(MetalTuning { tetra_kernel: name, ..Default::default() }).expect("a known kernel");
    assert_eq!(b.rt.tuning().tetra_kernel, name, "the tuning took");
}

/// Every twin, against the served kernel and the oracle, at one shape.
fn gate(seed: u64, d_out: usize, nblocks: usize, tail_w: usize, tile: usize, reps: usize) {
    let mut b = build(seed, d_out, nblocks, tail_w, tile);
    let oracle = on_cpu(&b);
    assert!(oracle.iter().any(|v| *v != 0.0), "zeros would pass every equality");
    with_kernel(&mut b, SERVED_TETRA);
    let served = on_metal(&b);
    assert_eq!(first_difference(&served, &oracle), None, "{d_out}x{nblocks} tile {tile}: served against the oracle");
    for name in TETRA_ROW_LOOPS {
        with_kernel(&mut b, name);
        for rep in 0..reps {
            let got = on_metal(&b);
            assert!(b.rt.compiled_pipelines().contains(&name), "{name} was launched");
            assert_eq!(
                first_difference(&got, &served),
                None,
                "{name}, {d_out} rows x {nblocks} blocks + {tail_w}, tile {tile}, rep {rep}"
            );
        }
    }
}

/// q, k, gate and up of the sealed 4B: d_in 2,560 = 106 blocks + 16. Two
/// tiles of 64, the second partial: lanes 10..31 take two blocks, the rest
/// three... and the prefetch crosses the tile boundary.
#[test]
fn the_twins_return_the_served_bits_at_d_in_2560() {
    gate(0x5EED_0001, 1024, 106, 16, 64, 4);
}

/// down_proj@0-11 of the sealed 4B: d_in 9,728 = 405 blocks + 8. Seven tiles,
/// thirteen blocks a lane for some lanes. Many threadgroups (320), repeated:
/// the staged twin's barriers race only above the machine's concurrency.
#[test]
fn the_twins_return_the_served_bits_at_d_in_9728() {
    gate(0x5EED_0002, 2560, 405, 8, 64, 8);
}

/// The loop's edges: fewer blocks than lanes (some lanes take none and the
/// prefetch guard is all that keeps them in bounds), exactly 32, one past a
/// tile, and no tail.
#[test]
fn the_twins_return_the_served_bits_at_the_loop_edges() {
    for (i, (nblocks, tail_w)) in [(7usize, 5usize), (32, 0), (65, 3), (33, 16)].into_iter().enumerate() {
        gate(0x5EED_0010 + i as u64, 64, nblocks, tail_w, 64, 2);
    }
}

/// The staged twin at the other tiles the runtime accepts. A tile of 32 is
/// the smallest at which a lane's blocks run on across tiles in order.
#[test]
fn the_staged_twin_holds_at_tiles_32_and_256() {
    gate(0x5EED_0020, 512, 405, 8, 32, 2);
    gate(0x5EED_0021, 512, 405, 8, 256, 2);
}

/// A multi-row twin refuses a d_out that eight simdgroups of its rows do not
/// tile, by name, at the launch: 48 rows are six threadgroups of the served
/// kernel and one and a half of `_lutd4`'s.
#[test]
fn a_multi_row_twin_refuses_a_partial_threadgroup() {
    let mut b = build(0x5EED_0040, 48, 7, 5, 64);
    with_kernel(&mut b, "tv_tetra48_metal_lutd4");
    let dev = Device::new_metal(0).expect("a Metal device");
    let xm = Tensor::from_slice(&b.x, (1, b.x.len()), &dev).expect("upload x");
    let e = b.rt.matvec(&b.proj, &xm).unwrap_err().to_string();
    assert!(e.contains("d_out 48") && e.contains("tv_tetra48_metal_lutd4"), "{e}");
    with_kernel(&mut b, "tv_tetra48_metal_lutd2");
    let oracle = on_cpu(&b);
    assert_eq!(first_difference(&on_metal(&b), &oracle), None, "48 rows are three groups of 16");
}

/// The tuning refuses what it cannot serve, by name, before anything is set.
#[test]
fn a_bad_tuning_is_refused_by_name() {
    let mut b = build(0x5EED_0030, 64, 7, 5, 64);
    for bad in ["tv_tetra48_metal_ilp", "tv_q4_metal", "tv_tetra48_nope"] {
        let e = b.rt.set_tuning(MetalTuning { tetra_kernel: bad, ..Default::default() }).unwrap_err().to_string();
        assert!(e.contains(bad), "{e}");
    }
    for tile in [0usize, 128, 300, 16384] {
        let e = b.rt.set_tuning(MetalTuning { q4_tile_cols: tile, ..Default::default() }).unwrap_err().to_string();
        assert!(e.contains(&tile.to_string()), "{e}");
    }
    assert_eq!(b.rt.tuning(), MetalTuning::default(), "a refused tuning sets nothing");
}

/// The f16 bits of a tensor, whatever its dtype: candle's own narrowing for
/// an f32 one, which is the served adapter's cast.
fn bits16_of(t: &Tensor) -> Vec<u16> {
    t.to_dtype(candle_core::DType::F16)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<half::f16>()
        .unwrap()
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

/// The half-store twins against the served kernel narrowed by candle's cast,
/// every f16 equal, at one shape; and that the twin is what ran.
fn gate_half(seed: u64, d_out: usize, nblocks: usize, tail_w: usize, tile: usize) {
    let mut b = build(seed, d_out, nblocks, tail_w, tile);
    let dev = Device::new_metal(0).expect("a Metal device");
    let xm = Tensor::from_slice(&b.x, (1, b.x.len()), &dev).expect("upload x");
    for (base, twin) in [(SERVED_TETRA, "tv_tetra48_metal_luth"), ("tv_tetra48_metal_lutd", "tv_tetra48_metal_lutdh")] {
        b.rt.set_tuning(MetalTuning { tetra_kernel: base, ..Default::default() }).unwrap();
        let served = b.rt.matvec(&b.proj, &xm).expect("served");
        assert_eq!(served.dtype(), candle_core::DType::F32, "the served kernel stores f32");
        let want = bits16_of(&served);
        assert!(want.iter().any(|v| *v != 0), "zeros would pass every equality");
        b.rt.set_tuning(MetalTuning { tetra_kernel: base, half_io: true, ..Default::default() }).unwrap();
        let got = b.rt.matvec(&b.proj, &xm).expect("half twin");
        assert_eq!(got.dtype(), candle_core::DType::F16, "{twin} stores f16");
        assert!(b.rt.compiled_pipelines().contains(&twin), "{twin} was launched");
        let got = bits16_of(&got);
        let diff = got.iter().zip(&want).position(|(p, q)| p != q);
        assert_eq!(diff, None, "{twin}, {d_out} rows x {nblocks} blocks + {tail_w}, tile {tile}");
    }
}

/// The sealed 4B's two Tetra widths, the loop edges and the other tiles.
#[test]
fn the_half_store_twins_return_the_served_bits_narrowed() {
    gate_half(0x5EED_0101, 1024, 106, 16, 64);
    gate_half(0x5EED_0102, 2560, 405, 8, 64);
    for (i, (nblocks, tail_w)) in [(7usize, 5usize), (32, 0), (65, 3)].into_iter().enumerate() {
        gate_half(0x5EED_0110 + i as u64, 64, nblocks, tail_w, 64);
    }
    gate_half(0x5EED_0120, 512, 405, 8, 32);
    gate_half(0x5EED_0121, 512, 405, 8, 256);
}

/// Half-io names its refusals: a kernel with no half twin, and the
/// prefetching int4 twins beside it.
#[test]
fn half_io_refuses_what_it_cannot_serve() {
    let mut b = build(0x5EED_0130, 64, 7, 5, 64);
    for bad in ["tv_tetra48_metal_lutp", "tv_tetra48_metal_lutd4"] {
        let e = b
            .rt
            .set_tuning(MetalTuning { tetra_kernel: bad, half_io: true, ..Default::default() })
            .unwrap_err()
            .to_string();
        assert!(e.contains(bad) && e.contains("half"), "{e}");
    }
    let e = b
        .rt
        .set_tuning(MetalTuning { half_io: true, q4_prefetch: true, ..Default::default() })
        .unwrap_err()
        .to_string();
    assert!(e.contains("half-io") && e.contains("prefetch"), "{e}");
    assert_eq!(b.rt.tuning(), MetalTuning::default(), "a refused tuning sets nothing");
}

/// The prefill kernel (`+p`, `tv_tetra48_metal_lutdr`) against the served
/// matvec run row by row and narrowed by candle's cast, every f16 equal, for
/// every chunk length 1..=8 and the row order; and that the kernel ran.
fn gate_rows(seed: u64, d_out: usize, nblocks: usize, tail_w: usize) {
    let mut b = build(seed, d_out, nblocks, tail_w, 64);
    let dev = Device::new_metal(0).expect("a Metal device");
    let d_in = b.x.len();
    let mut rng = SplitMix64::new(seed ^ 0xA5A5);
    // Full f32, as the rotation kernel writes it: an f16-representable
    // activation times a lattice value (an integer of at most 32) is exact in
    // f32, which would hide an unfused multiply-add (mutant P4, 2026-11-06).
    let xs: Vec<f32> = (0..8 * d_in).map(|_| rng.next_gaussian() as f32).collect();
    let want: Vec<Vec<u16>> = (0..8)
        .map(|r| {
            let xr = Tensor::from_slice(&xs[r * d_in..(r + 1) * d_in], (1, d_in), &dev).unwrap();
            bits16_of(&b.rt.matvec(&b.proj, &xr).expect("served"))
        })
        .collect();
    assert!(want[0].iter().any(|v| *v != 0), "zeros would pass every equality");
    assert_ne!(want[0], want[1], "two equal rows would hide a row mix-up");
    b.rt.set_tuning(MetalTuning { prefill_rows: 8, ..Default::default() }).unwrap();
    for n in 1..=8usize {
        let xm = Tensor::from_slice(&xs[..n * d_in], (n, d_in), &dev).unwrap();
        let got = b.rt.matvec_rows(&b.proj, &xm, n).expect("prefill rows");
        assert_eq!(got.dtype(), candle_core::DType::F16, "the prefill kernel stores f16");
        assert_eq!(got.dims(), [n, d_out], "one output row a prompt row");
        for (r, w) in want.iter().enumerate().take(n) {
            let g = bits16_of(&got.narrow(0, r, 1).unwrap());
            let diff = g.iter().zip(w).position(|(p, q)| p != q);
            assert_eq!(diff, None, "row {r} of {n}, {d_out} rows x {nblocks} blocks + {tail_w}");
        }
    }
    assert!(b.rt.compiled_pipelines().contains(&"tv_tetra48_metal_lutdr"), "the prefill kernel ran");
}

/// The sealed 4B's two Tetra widths and the loop edges.
#[test]
fn the_prefill_kernel_returns_each_rows_served_bits_narrowed() {
    gate_rows(0x5EED_0201, 1024, 106, 16);
    gate_rows(0x5EED_0202, 2560, 405, 8);
    for (i, (nblocks, tail_w)) in [(7usize, 5usize), (32, 0), (65, 3)].into_iter().enumerate() {
        gate_rows(0x5EED_0210 + i as u64, 64, nblocks, tail_w);
    }
}

/// The prefill tuning names its refusals: a row count the kernel does not
/// take, and a chunk longer than the tuning.
#[test]
fn the_prefill_kernel_refuses_what_it_cannot_take() {
    let mut b = build(0x5EED_0220, 64, 7, 5, 64);
    for bad in [0usize, 3, 16] {
        let e = b
            .rt
            .set_tuning(MetalTuning { prefill_rows: bad, ..Default::default() })
            .unwrap_err()
            .to_string();
        assert!(e.contains("prefill rows"), "{e}");
    }
    assert_eq!(b.rt.tuning(), MetalTuning::default(), "a refused tuning sets nothing");
    b.rt.set_tuning(MetalTuning { prefill_rows: 4, ..Default::default() }).unwrap();
    let dev = Device::new_metal(0).expect("a Metal device");
    let xm = Tensor::zeros((5, b.x.len()), candle_core::DType::F32, &dev).unwrap();
    let e = b.rt.matvec_rows(&b.proj, &xm, 5).unwrap_err().to_string();
    assert!(e.contains("5 rows") && e.contains("at most 4"), "{e}");
}
