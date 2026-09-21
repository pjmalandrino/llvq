//! The Tetra kernel reached through candle, against its own CPU arm.
//!
//! ## What this proves that the shader tests do not
//!
//! `llvq-metal/tests/tetra48_matches_rust.rs` proves the DECODER against
//! `llvq_search::tetra::Tetra`, and `tetra48_matvec_matches_host.rs` proves
//! the MATVEC against a host reference. Both drive the MSL through metal-rs,
//! which the shipped path cannot use: candle reaches Metal through
//! objc2-metal and the two sets of types do not meet.
//!
//! So what is left unproved is the BINDING: the buffer offsets, the
//! threadgroup length, the dispatch shape, the output allocation, and the
//! pipeline cache. This file is one `CustomOp1` run on two devices. Same op,
//! same buffers, same host copies; one goes through `metal_fwd` and the other
//! through `cpu_fwd`, and they must agree exactly.
//!
//! That shape is the point. `CustomOp1` demands a CPU arm, so making it the
//! reference rather than a `bail!` costs nothing and buys an oracle that
//! cannot drift from the op it is judging.
//!
//! ## Why equality and not a tolerance
//!
//! `cpu_fwd` reproduces the kernel's summation order: 32 lanes striding the
//! blocks, the shuffle-xor butterfly, the multiply-then-add tail. A tolerance
//! wide enough to cover a different order is wide enough to hide a wrong
//! buffer offset, and a wrong offset is the defect this file exists for.
//!
//! ## The mutation run
//!
//! Six mutants of `fused_metal.rs`, four killed: the activation offset left
//! in ELEMENTS instead of bytes, `set_threadgroup_memory_length` omitted,
//! `prefixes` and `branches` bound to each other's slots, and `rscale` bound
//! where `gscale` belongs.
//!
//! Two did not die, and neither is a hole in this file.
//!
//! Dropping the tile from the pipeline cache key survives because the cache
//! is per runtime and a runtime has ONE tile, so the key cannot collide
//! today. The tile stays in the key as a guard for the day the cache is
//! shared, and that is the only claim made for it.
//!
//! Halving the output allocation survives because candle's `new_buffer`
//! rounds up to a power of two and does not zero, so the overflow lands
//! inside the allocation and corrupts nothing the test can see. That is the
//! hazard the module header records as number four, and the defence is
//! structural rather than tested: the element count is derived from `d_out`
//! at one site.

#![cfg(all(target_os = "macos", feature = "metal"))]

use candle_core::{Device, Tensor};
use llvq_artifact::tetra48::{transcode_tetra48, TETRA48_SHELLS};
use llvq_bench::f1::rank::{branch_words, prefix_bytes, suffix_bytes, RankTable};
use llvq_bench::f1::Trellis;
use llvq_core::{SplitMix64, DIM};
use llvq_llm::fused_metal::{MetalRuntime, MetalTetraProj};

const TILE: usize = 64;

fn invnorm_table() -> Vec<f32> {
    let mut t = vec![0.0f32; TETRA48_SHELLS];
    for (m, e) in t.iter_mut().enumerate().skip(1) {
        *e = (1.0f64 / ((16 * m) as f64).sqrt()) as f32;
    }
    t
}

fn f32_to_f16_bits(v: f32) -> u16 {
    let b = v.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let mut exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let mant = b & 0x7f_ffff;
    if exp <= 0 {
        return sign;
    }
    if exp >= 31 {
        return sign | (30u16 << 10) | 0x3ff;
    }
    let mut m = (mant >> 13) as u16;
    let rest = mant & 0x1fff;
    if rest > 0x1000 || (rest == 0x1000 && (m & 1) == 1) {
        m += 1;
        if m == 0x400 {
            m = 0;
            exp += 1;
        }
    }
    sign | ((exp as u16) << 10) | m
}

fn plain_runtime(dev: &Device, tile: usize) -> MetalRuntime {
    let table = RankTable::build();
    let tr = Trellis::new();
    MetalRuntime::new(
        dev,
        &table.rows,
        &prefix_bytes(&tr),
        &branch_words(&tr),
        &suffix_bytes(&tr),
        &invnorm_table(),
        tile,
        false,
    )
    .expect("the runtime builds")
}

struct Bench {
    rt: MetalRuntime,
    proj: MetalTetraProj,
    x: Vec<f32>,
}

/// A runtime, a projection and an activation, all from one seed.
fn build(seed: u64, d_out: usize, nblocks: usize, tail_w: usize) -> Option<Bench> {
    let dev = Device::new_metal(0).ok()?;
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
        TILE,
        true,
    )
    .expect("the runtime builds");

    let rscale: Vec<f32> = (0..d_out).map(|_| 0.5 + rng.next_gaussian().abs() as f32).collect();
    let tail: Vec<u16> = (0..d_out * tail_w)
        .map(|_| f32_to_f16_bits(rng.next_gaussian() as f32))
        .collect();
    let d_in = nblocks * DIM + tail_w;
    let proj = rt
        .upload(
            "000.q_proj",
            d_out,
            d_in,
            nblocks,
            stream.stride_u32,
            &stream.data,
            &[0.625, 1.375],
            &rscale,
            &tail,
        )
        .expect("the projection uploads");

    let x: Vec<f32> = (0..d_in).map(|_| rng.next_gaussian() as f32).collect();
    Some(Bench { rt, proj, x })
}

/// Run the op on both devices and hand back the two answers.
fn both(b: &Bench) -> (Vec<f32>, Vec<f32>) {
    let dev = Device::new_metal(0).expect("a Metal device");
    let xm = Tensor::from_slice(&b.x, (1, b.x.len()), &dev).expect("upload x");
    let xc = Tensor::from_slice(&b.x, (1, b.x.len()), &Device::Cpu).expect("host x");
    let ym = b.rt.matvec(&b.proj, &xm).expect("metal_fwd");
    let yc = b.rt.matvec(&b.proj, &xc).expect("cpu_fwd");
    (
        ym.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        yc.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
    )
}

/// The gate. One tile, a tail, both devices, exactly equal.
#[test]
fn the_candle_binding_agrees_with_the_cpu_arm() {
    let Some(b) = build(0x7E_49A0, 64, TILE, 8) else {
        panic!("this test needs a Metal device, and macOS always has one");
    };
    let (m, c) = both(&b);
    assert_eq!(m.len(), 64, "one value a row");
    for (i, (a, e)) in m.iter().zip(&c).enumerate() {
        assert_eq!(a, e, "row {i}: metal {a} against the cpu arm {e}");
    }
    assert!(m.iter().any(|v| *v != 0.0), "zeros would pass every equality above");
}

/// Several tiles with a partial last one, which is where the staging length
/// and the tile clamp have to agree.
#[test]
fn the_binding_holds_across_a_partial_tile() {
    let Some(b) = build(0x7E_49A1, 32, 3 * TILE + 17, 5) else {
        panic!("needs a Metal device");
    };
    let (m, c) = both(&b);
    for (i, (a, e)) in m.iter().zip(&c).enumerate() {
        assert_eq!(a, e, "row {i}");
    }
}

/// No tail: the case that needs a one-element dummy buffer, because Metal
/// returns a null pointer for a zero-length allocation.
#[test]
fn the_binding_holds_with_no_tail() {
    let Some(b) = build(0x7E_49A2, 16, TILE + 1, 0) else {
        panic!("needs a Metal device");
    };
    let (m, c) = both(&b);
    for (i, (a, e)) in m.iter().zip(&c).enumerate() {
        assert_eq!(a, e, "row {i}");
    }
}

/// The kernel reads the activation it was handed, at the offset the layout
/// implies. A narrow moves `start_offset` off zero, which is the one thing
/// `metal_fwd` has to convert from elements into bytes.
#[test]
fn a_narrowed_activation_is_read_at_its_own_offset() {
    let Some(b) = build(0x7E_49A3, 16, TILE, 4) else {
        panic!("needs a Metal device");
    };
    let dev = Device::new_metal(0).expect("a Metal device");
    let d_in = b.x.len();

    // Two rows, the second of which is the activation under test. Narrowing
    // to it puts `start_offset` at `d_in` rather than 0.
    let mut wide = vec![0f32; d_in];
    wide.extend_from_slice(&b.x);
    let t = Tensor::from_slice(&wide, (2, d_in), &dev).expect("upload");
    let narrowed = t.narrow(0, 1, 1).expect("narrow");
    let got = b.rt.matvec(&b.proj, &narrowed).expect("metal_fwd on a narrow");

    let xc = Tensor::from_slice(&b.x, (1, d_in), &Device::Cpu).expect("host x");
    let want = b.rt.matvec(&b.proj, &xc).expect("cpu_fwd");
    let got = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let want = want.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    for (i, (a, e)) in got.iter().zip(&want).enumerate() {
        assert_eq!(a, e, "row {i}: the narrow must read row 1, not row 0");
    }
}

/// A served runtime keeps no host copy, so its CPU arm refuses by name rather
/// than standing in as a fallback.
#[test]
fn a_runtime_without_host_copies_refuses_its_cpu_arm() {
    let dev = Device::new_metal(0).expect("a Metal device");
    let table = RankTable::build();
    let tr = Trellis::new();
    let rt = MetalRuntime::new(
        &dev,
        &table.rows,
        &prefix_bytes(&tr),
        &branch_words(&tr),
        &suffix_bytes(&tr),
        &invnorm_table(),
        TILE,
        false,
    )
    .expect("builds");
    let nblocks = 4usize;
    let d_out = 8usize;
    let n = d_out * nblocks;
    let stream = transcode_tetra48(&vec![0u64; n], &vec![0u32; n], d_out, nblocks).expect("ok");
    let proj = rt
        .upload(
            "000.q_proj",
            d_out,
            nblocks * DIM,
            nblocks,
            stream.stride_u32,
            &stream.data,
            &[0.625, 1.375],
            &vec![1.0f32; d_out],
            &[],
        )
        .expect("uploads");
    let x = Tensor::zeros((1, nblocks * DIM), candle_core::DType::F32, &Device::Cpu).unwrap();
    let err = rt.matvec(&proj, &x).expect_err("no host copies, no CPU arm");
    assert!(
        err.to_string().contains("without host copies"),
        "the refusal must name the reason: {err}"
    );
}

/// `d_out` that is not a multiple of the rows a threadgroup covers is refused
/// at upload. The kernel carries no row guard, so a partial group would
/// compute and STORE past the output.
#[test]
fn a_ragged_d_out_is_refused_at_upload() {
    let dev = Device::new_metal(0).expect("a Metal device");
    let table = RankTable::build();
    let tr = Trellis::new();
    let rt = MetalRuntime::new(
        &dev,
        &table.rows,
        &prefix_bytes(&tr),
        &branch_words(&tr),
        &suffix_bytes(&tr),
        &invnorm_table(),
        TILE,
        false,
    )
    .expect("builds");
    let nblocks = 4usize;
    let d_out = 12usize; // 12 is not a multiple of 8
    let n = d_out * nblocks;
    let stream = transcode_tetra48(&vec![0u64; n], &vec![0u32; n], d_out, nblocks).expect("ok");
    let out = rt.upload(
        "000.q_proj",
        d_out,
        nblocks * DIM,
        nblocks,
        stream.stride_u32,
        &stream.data,
        &[0.625, 1.375],
        &vec![1.0f32; d_out],
        &[],
    );
    // `MetalTetraProj` holds device buffers and is deliberately not `Debug`,
    // so the refusal is matched rather than unwrapped.
    let Err(err) = out else {
        panic!("12 rows is not a whole number of threadgroups and must be refused");
    };
    assert!(err.to_string().contains("not a multiple"), "must say why: {err}");
}

// ---------------------------------------------------------------------------
// The refusals. Each was MISSING until an audit of 2026-09-21, and each has a
// CUDA twin that already refused it, in one case with a comment saying the
// check had already cost a run.
// ---------------------------------------------------------------------------

/// An activation shorter than `d_in` reads past the buffer and returns
/// finite, plausible, wrong numbers.
#[test]
fn a_short_activation_is_refused() {
    let Some(b) = build(0x7E_49B0, 16, TILE, 4) else { panic!("needs Metal") };
    let dev = Device::new_metal(0).expect("Metal");
    let short = &b.x[..b.x.len() - 1];
    for d in [dev, Device::Cpu] {
        let x = Tensor::from_slice(short, (1, short.len()), &d).expect("upload");
        let err = b.rt.matvec(&b.proj, &x).expect_err("one value short");
        assert!(err.to_string().contains("for d_in="), "must name d_in: {err}");
    }
}

/// Several vectors at once. The kernel is a matvec and the output is
/// allocated at `d_out`, so accepting them would hand back a tensor whose
/// shape claims more elements than its storage holds.
#[test]
fn a_multi_row_activation_is_refused() {
    let Some(b) = build(0x7E_49B1, 16, TILE, 4) else { panic!("needs Metal") };
    let dev = Device::new_metal(0).expect("Metal");
    let d_in = b.x.len();
    let mut two = b.x.clone();
    two.extend_from_slice(&b.x);
    for d in [dev, Device::Cpu] {
        let x = Tensor::from_slice(&two, (2, d_in), &d).expect("upload");
        let err = b.rt.matvec(&b.proj, &x).expect_err("two vectors");
        assert!(err.to_string().contains("vectors at once"), "must say so: {err}");
    }
}

/// A non-contiguous activation. `metal_fwd` refused it before the audit and
/// `cpu_fwd` did not, so the two arms did not implement the same function.
#[test]
fn a_non_contiguous_activation_is_refused_on_both_arms() {
    let Some(b) = build(0x7E_49B2, 16, TILE, 4) else { panic!("needs Metal") };
    let dev = Device::new_metal(0).expect("Metal");
    let d_in = b.x.len();
    let mut two = b.x.clone();
    two.extend_from_slice(&b.x);
    for d in [dev, Device::Cpu] {
        // A transpose makes the last axis strided without copying.
        let x = Tensor::from_slice(&two, (2, d_in), &d)
            .expect("upload")
            .t()
            .expect("transpose")
            .narrow(0, 0, 1)
            .expect("narrow");
        let err = b.rt.matvec(&b.proj, &x).expect_err("strided");
        let m = err.to_string();
        assert!(
            m.contains("contiguous") || m.contains("for d_in="),
            "must refuse by name: {m}"
        );
    }
}

/// A truncated weight stream. `rscale` and `tail` were checked at upload and
/// the stream was not.
#[test]
fn a_truncated_stream_is_refused_at_upload() {
    let dev = Device::new_metal(0).expect("Metal");
    let rt = plain_runtime(&dev, TILE);
    let (d_out, nblocks) = (8usize, 4usize);
    let n = d_out * nblocks;
    let stream = transcode_tetra48(&vec![0u64; n], &vec![0u32; n], d_out, nblocks).expect("ok");
    let short = &stream.data[..stream.data.len() - 4];
    let out = rt.upload(
        "000.q_proj", d_out, nblocks * DIM, nblocks, stream.stride_u32,
        short, &[0.625, 1.375], &vec![1.0f32; d_out], &[],
    );
    let Err(err) = out else { panic!("a stream one word short must be refused") };
    assert!(err.to_string().contains("stream bytes"), "must say so: {err}");
}

/// A stride that does not match `nblocks`. The kernel multiplies by it for
/// every row, so a wrong one walks off the end after a few rows.
#[test]
fn a_wrong_row_stride_is_refused_at_upload() {
    let dev = Device::new_metal(0).expect("Metal");
    let rt = plain_runtime(&dev, TILE);
    let (d_out, nblocks) = (8usize, 4usize);
    let n = d_out * nblocks;
    let stream = transcode_tetra48(&vec![0u64; n], &vec![0u32; n], d_out, nblocks).expect("ok");
    let out = rt.upload(
        "000.q_proj", d_out, nblocks * DIM, nblocks, stream.stride_u32 + 1,
        &stream.data, &[0.625, 1.375], &vec![1.0f32; d_out], &[],
    );
    let Err(err) = out else { panic!("a stride that does not match nblocks must be refused") };
    assert!(err.to_string().contains("row stride"), "must say so: {err}");
}

/// A tile of 512 passed a validator copied from the CUDA contract and then
/// asked for 49,152 B of threadgroup memory against Apple's 32,768.
#[test]
fn a_tile_beyond_apples_threadgroup_memory_is_refused() {
    let dev = Device::new_metal(0).expect("Metal");
    let table = RankTable::build();
    let tr = Trellis::new();
    let out = MetalRuntime::new(
        &dev, &table.rows, &prefix_bytes(&tr), &branch_words(&tr),
        &suffix_bytes(&tr), &invnorm_table(), 512, false,
    );
    let Err(err) = out else { panic!("512 blocks stage 49152 B and must be refused") };
    assert!(err.to_string().contains("32768"), "must name the limit: {err}");
    // 256 is the largest that fits, and it is accepted.
    assert!(
        MetalRuntime::new(
            &dev, &table.rows, &prefix_bytes(&tr), &branch_words(&tr),
            &suffix_bytes(&tr), &invnorm_table(), 256, false,
        )
        .is_ok(),
        "256 stages 24576 B and fits"
    );
}
