//! What one launch of the f32 prefill rows kernel (`tv_tetra48_metal_lutdrf`)
//! costs at Inkling 2 Mini's expert shapes, on synthetic codes: gate and up
//! (1,152 rows of 2,304) and down (2,304 rows of 1,152), 8 rows a launch.
//! Read against the MoE share of a prompt's read: whether the experts are
//! bound by the kernel's own time or by what surrounds it.
//!
//! Ignored by default: it uses the GPU for a few seconds; run it by name under
//! the guard's lock.
#![cfg(all(target_os = "macos", feature = "metal"))]

use std::time::Instant;

use candle_core::{Device, Tensor};
use llvq_artifact::tetra48::{transcode_tetra48, TETRA48_SHELLS};
use llvq_bench::f1::rank::{branch_words, prefix_bytes, suffix_bytes, RankTable};
use llvq_bench::f1::Trellis;
use llvq_core::{SplitMix64, DIM};
use llvq_llm::fused_metal::{MetalRuntime, MetalTetraProj, MetalTuning};

fn invnorm_table() -> Vec<f32> {
    let mut t = vec![0.0f32; TETRA48_SHELLS];
    for (m, e) in t.iter_mut().enumerate().skip(1) {
        *e = (1.0f64 / ((16 * m) as f64).sqrt()) as f32;
    }
    t
}

fn build(dev: &Device, seed: u64, d_out: usize, nblocks: usize) -> (MetalRuntime, MetalTetraProj) {
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
    let mut rt = MetalRuntime::new(
        dev,
        &table.rows,
        &prefix_bytes(&tr),
        &branch_words(&tr),
        &suffix_bytes(&tr),
        &invnorm_table(),
        64,
        true,
    )
    .expect("the runtime builds");
    let rscale: Vec<f32> = (0..d_out).map(|_| 0.5 + rng.next_gaussian().abs() as f32).collect();
    let d_in = nblocks * DIM;
    let proj = rt
        .upload("000.gate_proj", d_out, d_in, nblocks, stream.stride_u32, &stream.data, &[0.625, 1.375], &rscale, &[])
        .expect("the projection uploads");
    rt.set_tuning(MetalTuning { prefill_rows: 8, ..Default::default() }).unwrap();
    (rt, proj)
}

fn time_shape(d_out: usize, nblocks: usize) {
    let dev = Device::new_metal(0).expect("a Metal device");
    let (rt, proj) = build(&dev, 0x5EED_7001 ^ d_out as u64, d_out, nblocks);
    let d_in = nblocks * DIM;
    let mut rng = SplitMix64::new(7);
    let xs: Vec<f32> = (0..8 * d_in).map(|_| rng.next_gaussian() as f32).collect();
    let x = Tensor::from_slice(&xs, (8, d_in), &dev).unwrap();
    for _ in 0..200 {
        rt.matvec_rows_f32(&proj, &x, 8).unwrap();
    }
    dev.synchronize().unwrap();
    for rows in [1usize, 2, 4, 8, 8] {
        let xr = x.narrow(0, 0, rows).unwrap().contiguous().unwrap();
        let n = 2000;
        let t = Instant::now();
        let mut keep = Vec::with_capacity(n);
        for _ in 0..n {
            keep.push(rt.matvec_rows_f32(&proj, &xr, rows).unwrap());
        }
        dev.synchronize().unwrap();
        let s = t.elapsed().as_secs_f64() / n as f64;
        let flops = 2.0 * rows as f64 * (d_out * d_in) as f64;
        println!(
            "lutdrf {d_out}x{d_in}, {rows} rows, {n} launches: {:.1} us a launch, {:.2} TFLOP/s, {:.1} GB/s of codes",
            s * 1e6,
            flops / s / 1e12,
            proj.device_bytes() as f64 / s / 1e9
        );
    }
}

#[test]
#[ignore = "uses the GPU; run under the guard's lock"]
fn prefill_rows_launch_cost_at_expert_shapes() {
    time_shape(1152, 96);
    time_shape(2304, 48);
}

/// What candle's f32 product costs at the experts' shapes, against the rows
/// kernel: the price of plan item P3 (decode an expert once into a dense
/// tile, then a plain product), before the decode. A visit of a 512-token
/// piece hands each of 128 experts about 32 rows.
#[test]
#[ignore = "uses the GPU; run under the guard's lock"]
fn dense_product_cost_at_expert_shapes() {
    let dev = Device::new_metal(0).expect("a Metal device");
    let mut rng = SplitMix64::new(11);
    for (rows, d_out, d_in) in [(32usize, 1152usize, 2304usize), (32, 2304, 1152), (64, 1152, 2304), (256, 1152, 2304)] {
        let w: Vec<f32> = (0..d_out * d_in).map(|_| rng.next_gaussian() as f32 * 0.02).collect();
        let w = Tensor::from_slice(&w, (d_out, d_in), &dev).unwrap();
        let x: Vec<f32> = (0..rows * d_in).map(|_| rng.next_gaussian() as f32).collect();
        let x = Tensor::from_slice(&x, (rows, d_in), &dev).unwrap();
        let wt = w.t().unwrap();
        for _ in 0..20 {
            drop(x.matmul(&wt).unwrap());
        }
        dev.synchronize().unwrap();
        let n = 384;
        let t = Instant::now();
        let mut keep = Vec::with_capacity(n);
        for _ in 0..n {
            keep.push(x.matmul(&wt).unwrap());
        }
        dev.synchronize().unwrap();
        let s = t.elapsed().as_secs_f64() / n as f64;
        println!(
            "candle f32 matmul [{rows}, {d_in}] x [{d_in}, {d_out}]: {:.1} us, {:.2} TFLOP/s",
            s * 1e6,
            2.0 * (rows * d_out * d_in) as f64 / s / 1e12
        );
    }
}
