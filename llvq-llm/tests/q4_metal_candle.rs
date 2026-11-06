//! The int4 arm of the Metal path reached through candle: `upload_int4_tiled`
//! and `matvec_int4`, against a host reference in the kernel's lane order,
//! every f32 equal.
//!
//! `llvq-metal/tests/q4_tiled_matches_served.rs` proves the kernels through
//! metal-rs. This file proves the BINDING the shipped path uses: the routing
//! of a record past the wall to `tv_q4_metal_tiled`, buffer 7, the
//! threadgroup length (the tile, never d_in), an activation bound at a
//! non-zero offset, and the refusals by name. No test covered the int4 arm
//! through candle before, tiled or not.
//!
//! The half-io twins (`+h`): the same matvecs reading and storing f16, against
//! the served kernels narrowed by candle's cast, every f16 equal.
//!
//! Mutation run (2026-11-06, journal `docs/mesures/metal-nuit-2026-11-06.txt`).

#![cfg(all(target_os = "macos", feature = "metal"))]

use candle_core::{DType, Device, Tensor};
use llvq_artifact::{Int4Matrix, INT4G128_BITS, INT4G128_GROUP};
use llvq_bench::f1::rank::{branch_words, prefix_bytes, suffix_bytes, RankTable};
use llvq_bench::f1::Trellis;
use llvq_core::SplitMix64;
use llvq_llm::fused_metal::{MetalRuntime, MetalTuning, Q4_TILE_COLS};

const LANES: usize = 32;

fn runtime(dev: &Device) -> MetalRuntime {
    let table = RankTable::build();
    let tr = Trellis::new();
    let invnorm = vec![0.0f32; 32];
    MetalRuntime::new(dev, &table.rows, &prefix_bytes(&tr), &branch_words(&tr), &suffix_bytes(&tr), &invnorm, 64, false)
        .expect("the runtime builds")
}

fn f16(v: f32) -> u16 {
    half::f16::from_f32(v).to_bits()
}

/// Random nibbles, one group in seven ungraded, scales over two decades: the
/// fixture of `llvq-metal/tests/q4_matches_host.rs`. The activation is f16, as
/// the model hands it to the int4 arm.
fn fixture(seed: u64, d_out: usize, d_in: usize) -> (Int4Matrix, Vec<half::f16>) {
    let mut rng = SplitMix64::new(seed);
    let gpr = d_in / INT4G128_GROUP;
    let mut packed: Vec<u8> = (0..d_out * d_in / 2).map(|_| rng.next() as u8).collect();
    let (mut scales, mut biases) = (Vec::new(), Vec::new());
    for row in 0..d_out {
        for g in 0..gpr {
            let ungraded = (row * gpr + g).is_multiple_of(7);
            scales.push(f16(if ungraded { 1.0 } else { 0.002 + 0.08 * rng.next_f64() as f32 }));
            biases.push(f16(rng.next_gaussian() as f32 * 0.05));
            if ungraded {
                let lo = (row * d_in + g * INT4G128_GROUP) / 2;
                packed[lo..lo + INT4G128_GROUP / 2].fill(0);
            }
        }
    }
    let m = Int4Matrix {
        name: "model.layers.12.mlp.down_proj.weight".into(),
        d_out,
        d_in,
        bits: INT4G128_BITS,
        group: INT4G128_GROUP,
        packed,
        scales,
        biases,
    };
    let x = (0..d_in).map(|_| half::f16::from_f32(rng.next_gaussian() as f32)).collect();
    (m, x)
}

/// The kernel's order: lane `l` takes words `l, l + 32, ...`, eight columns a
/// word with `mul_add`, then the shuffle-xor butterfly.
fn reference(m: &Int4Matrix, x: &[f32]) -> Vec<f32> {
    let w = m.to_f32();
    let nwords = m.d_in / 8;
    (0..m.d_out)
        .map(|row| {
            let mut lanes = [0f32; LANES];
            for (lane, acc) in lanes.iter_mut().enumerate() {
                let mut wi = lane;
                while wi < nwords {
                    let c = wi * 8;
                    for (wv, xv) in w[row * m.d_in + c..row * m.d_in + c + 8].iter().zip(&x[c..c + 8]) {
                        *acc = wv.mul_add(*xv, *acc);
                    }
                    wi += LANES;
                }
            }
            for k in [16usize, 8, 4, 2, 1] {
                let mut next = [0f32; LANES];
                for (l, n) in next.iter_mut().enumerate() {
                    *n = lanes[l] + lanes[l ^ k];
                }
                lanes = next;
            }
            lanes[0]
        })
        .collect()
}

fn words(m: &Int4Matrix) -> Vec<u32> {
    m.packed.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

/// `matvec_int4` on `x`, bound at a non-zero offset: the activation is the
/// second row of a two-row tensor, narrowed, as a decode step's view can be.
fn run(rt: &MetalRuntime, dev: &Device, m: &Int4Matrix, x: &[half::f16], tile: usize) -> (Option<usize>, Vec<f32>) {
    let p = rt
        .upload_int4_tiled(&m.name, m.d_out, m.d_in, m.d_in / 128, &words(m), &m.scales, &m.biases, tile)
        .expect("uploads");
    let junk: Vec<half::f16> = (0..m.d_in).map(|i| half::f16::from_f32(i as f32)).collect();
    let both: Vec<half::f16> = junk.iter().chain(x).copied().collect();
    let t = Tensor::from_vec(both, (2, m.d_in), dev).unwrap().narrow(0, 1, 1).unwrap();
    let y = rt.matvec_int4(&p, &t).expect("matvec").to_dtype(DType::F32).unwrap();
    (p.tile_cols(), y.flatten_all().unwrap().to_vec1::<f32>().unwrap())
}

fn first_difference(a: &[f32], b: &[f32]) -> Option<(usize, f32, f32)> {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).enumerate().find(|(_, (p, q))| p.to_bits() != q.to_bits()).map(|(i, (p, q))| (i, *p, *q))
}

/// The records that fit keep the served kernel; the down_proj widths of the
/// three sealed sizes route to the tiled one; both give the reference's bits,
/// served row loop and prefetching twin alike.
#[test]
fn the_binding_routes_and_matches_at_six_widths() {
    let dev = Device::new_metal(0).expect("a Metal device");
    let mut rt = runtime(&dev);
    for (i, (d_out, d_in)) in [(512usize, 2560usize), (256, 4096), (256, 5120), (2560, 9728), (256, 12288), (256, 17408)]
        .into_iter()
        .enumerate()
    {
        let (m, xh) = fixture(0x4A_C0 + i as u64, d_out, d_in);
        let x: Vec<f32> = xh.iter().map(|v| v.to_f32()).collect();
        let want = reference(&m, &x);
        assert!(want.iter().any(|v| *v != 0.0));
        for prefetch in [false, true] {
            rt.set_tuning(MetalTuning { q4_prefetch: prefetch, ..Default::default() }).unwrap();
            for tile in [Q4_TILE_COLS, 256, 8192] {
                let (routed, got) = run(&rt, &dev, &m, &xh, tile);
                let past = d_in * 4 > 32_768;
                assert_eq!(routed, past.then_some(tile), "{d_out}x{d_in}: routing at tile {tile}");
                assert_eq!(first_difference(&got, &want), None, "{d_out}x{d_in} tile {tile} prefetch {prefetch}");
            }
        }
    }
    // The twins return the served bits by design, so only the pipeline cache
    // can say they ran: all four int4 entry points were launched.
    assert_eq!(
        rt.compiled_pipelines(),
        vec!["tv_q4_metal", "tv_q4_metal_p", "tv_q4_metal_tiled", "tv_q4_metal_tiled_p"]
    );
}

/// A bad tile, a wrong group count and a short row are refused by name at
/// upload, before any launch. A zero tile would hang the GPU.
#[test]
fn the_binding_refuses_by_name() {
    let dev = Device::new_metal(0).expect("a Metal device");
    let rt = runtime(&dev);
    let (m, _) = fixture(9, 64, 9728);
    let up = |tile: usize, gpr: usize| {
        rt.upload_int4_tiled(&m.name, m.d_out, m.d_in, gpr, &words(&m), &m.scales, &m.biases, tile)
            .err()
            .map(|e| e.to_string())
    };
    for bad in [0usize, 128, 300, 16384] {
        let e = up(bad, 76).expect("refused");
        assert!(e.contains(&bad.to_string()), "{e}");
    }
    // A bad tile is refused on a record that fits, too: the first record of a
    // load names it, not the first record past the wall.
    let (small, _) = fixture(10, 64, 2560);
    let e = rt
        .upload_int4_tiled(&small.name, 64, 2560, 20, &words(&small), &small.scales, &small.biases, 0)
        .err()
        .expect("refused")
        .to_string();
    assert!(e.contains("tile_cols 0"), "{e}");
    let e = up(Q4_TILE_COLS, 75).expect("refused");
    assert!(e.contains("75 groups"), "{e}");
    assert!(up(Q4_TILE_COLS, 76).is_none(), "the honest record uploads");
}

/// `matvec_int4` as the model calls it, narrowed view and all, returning the
/// tensor as the arm hands it back (f32 served, f16 half-io).
fn run_raw(rt: &MetalRuntime, dev: &Device, m: &Int4Matrix, x: &[half::f16], tile: usize) -> Tensor {
    let p = rt
        .upload_int4_tiled(&m.name, m.d_out, m.d_in, m.d_in / 128, &words(m), &m.scales, &m.biases, tile)
        .expect("uploads");
    let junk: Vec<half::f16> = (0..m.d_in).map(|i| half::f16::from_f32(i as f32)).collect();
    let both: Vec<half::f16> = junk.iter().chain(x).copied().collect();
    let t = Tensor::from_vec(both, (2, m.d_in), dev).unwrap().narrow(0, 1, 1).unwrap();
    rt.matvec_int4(&p, &t).expect("matvec")
}

fn bits16_of(t: &Tensor) -> Vec<u16> {
    t.to_dtype(DType::F16).unwrap().flatten_all().unwrap().to_vec1::<half::f16>().unwrap().iter().map(|v| v.to_bits()).collect()
}

/// The half-io twins against the served kernels narrowed by candle's cast,
/// every f16 equal, at the six widths and three tiles; and that both twins
/// are what ran.
#[test]
fn the_half_io_twins_return_the_served_bits_narrowed() {
    let dev = Device::new_metal(0).expect("a Metal device");
    let mut served_rt = runtime(&dev);
    served_rt.set_tuning(MetalTuning::default()).unwrap();
    let mut half_rt = runtime(&dev);
    half_rt.set_tuning(MetalTuning { half_io: true, ..Default::default() }).unwrap();
    for (i, (d_out, d_in)) in [(512usize, 2560usize), (256, 4096), (256, 5120), (2560, 9728), (256, 12288), (256, 17408)]
        .into_iter()
        .enumerate()
    {
        let (m, xh) = fixture(0x4A_D0 + i as u64, d_out, d_in);
        for tile in [Q4_TILE_COLS, 256, 8192] {
            let served = run_raw(&served_rt, &dev, &m, &xh, tile);
            assert_eq!(served.dtype(), DType::F32);
            let want = bits16_of(&served);
            assert!(want.iter().any(|v| *v != 0));
            let got = run_raw(&half_rt, &dev, &m, &xh, tile);
            assert_eq!(got.dtype(), DType::F16, "{d_out}x{d_in}: the half-io arm returns f16");
            let got = bits16_of(&got);
            assert_eq!(got.iter().zip(&want).position(|(p, q)| p != q), None, "{d_out}x{d_in} tile {tile}");
        }
    }
    assert_eq!(half_rt.compiled_pipelines(), vec!["tv_q4_metal_h", "tv_q4_metal_tiled_h"]);
}

/// An f32 activation on a half-io runtime keeps the served f32 kernels: the
/// twins read f16 only, and a model in f32 must not be narrowed.
#[test]
fn an_f32_activation_keeps_the_served_kernel_under_half_io() {
    let dev = Device::new_metal(0).expect("a Metal device");
    let mut rt = runtime(&dev);
    rt.set_tuning(MetalTuning { half_io: true, ..Default::default() }).unwrap();
    let (m, xh) = fixture(0x4A_E0, 256, 2560);
    let x: Vec<f32> = xh.iter().map(|v| v.to_f32()).collect();
    let p = rt.upload_int4_tiled(&m.name, m.d_out, m.d_in, m.d_in / 128, &words(&m), &m.scales, &m.biases, Q4_TILE_COLS).unwrap();
    let t = Tensor::from_vec(x.clone(), (1, m.d_in), &dev).unwrap();
    let y = rt.matvec_int4(&p, &t).unwrap();
    assert_eq!(y.dtype(), DType::F32);
    assert_eq!(first_difference(&y.flatten_all().unwrap().to_vec1::<f32>().unwrap(), &reference(&m, &x)), None);
    assert_eq!(rt.compiled_pipelines(), vec!["tv_q4_metal"]);
}

/// The int4 prefill kernel (`+pq`, `tv_q4_metal_rows`) against the served
/// matvec run row by row and narrowed, every f16 equal, for chunks 1..=8 at
/// the six widths, untiled and past the wall alike; the chunk is bound at a
/// non-zero offset, as `forward_rows` hands a narrowed view.
#[test]
fn the_int4_prefill_kernel_returns_each_rows_served_bits_narrowed() {
    let dev = Device::new_metal(0).expect("a Metal device");
    let served_rt = runtime(&dev);
    let mut rows_rt = runtime(&dev);
    rows_rt.set_tuning(MetalTuning { prefill_rows: 8, q4_rows: true, ..Default::default() }).unwrap();
    for (i, (d_out, d_in)) in [(512usize, 2560usize), (256, 4096), (256, 5120), (2560, 9728), (256, 12288), (256, 17408)]
        .into_iter()
        .enumerate()
    {
        let (m, _) = fixture(0x4A_F0 + i as u64, d_out, d_in);
        let mut rng = SplitMix64::new(0x4A_F8 + i as u64);
        // Row 0 is junk the chunks start past.
        let xs: Vec<half::f16> =
            (0..9 * d_in).map(|_| half::f16::from_f32(rng.next_gaussian() as f32)).collect();
        let want: Vec<Vec<u16>> = (1..9)
            .map(|r| bits16_of(&run_raw(&served_rt, &dev, &m, &xs[r * d_in..(r + 1) * d_in], Q4_TILE_COLS)))
            .collect();
        assert!(want[0].iter().any(|v| *v != 0));
        assert_ne!(want[0], want[1], "two equal rows would hide a row mix-up");
        let p = rows_rt
            .upload_int4_tiled(&m.name, m.d_out, m.d_in, m.d_in / 128, &words(&m), &m.scales, &m.biases, Q4_TILE_COLS)
            .unwrap();
        let all = Tensor::from_vec(xs.clone(), (9, d_in), &dev).unwrap();
        for n in 1..=8usize {
            let got = rows_rt.matvec_int4_rows(&p, &all.narrow(0, 1, n).unwrap(), n).expect("int4 rows");
            assert_eq!(got.dtype(), DType::F16);
            assert_eq!(got.dims(), [n, d_out]);
            for (r, w) in want.iter().enumerate().take(n) {
                let g = bits16_of(&got.narrow(0, r, 1).unwrap());
                assert_eq!(g.iter().zip(w).position(|(a, b)| a != b), None, "{d_out}x{d_in}: row {r} of {n}");
            }
        }
    }
    assert_eq!(rows_rt.compiled_pipelines(), vec!["tv_q4_metal_rows"]);
}

/// `q4_rows` names its refusals: without prefill rows at the tuning, and on a
/// runtime that has not switched it on.
#[test]
fn the_int4_prefill_kernel_refuses_what_it_cannot_take() {
    let dev = Device::new_metal(0).expect("a Metal device");
    let mut rt = runtime(&dev);
    let e = rt.set_tuning(MetalTuning { q4_rows: true, ..Default::default() }).unwrap_err().to_string();
    assert!(e.contains("q4 rows"), "{e}");
    assert_eq!(rt.tuning(), MetalTuning::default(), "a refused tuning sets nothing");
    rt.set_tuning(MetalTuning { prefill_rows: 8, ..Default::default() }).unwrap();
    let (m, _) = fixture(0x4B_00, 256, 2560);
    let p = rt.upload_int4_tiled(&m.name, m.d_out, m.d_in, m.d_in / 128, &words(&m), &m.scales, &m.biases, Q4_TILE_COLS).unwrap();
    let x = Tensor::zeros((2, m.d_in), DType::F16, &dev).unwrap();
    let e = rt.matvec_int4_rows(&p, &x, 2).unwrap_err().to_string();
    assert!(e.contains("q4 rows false"), "{e}");
}
