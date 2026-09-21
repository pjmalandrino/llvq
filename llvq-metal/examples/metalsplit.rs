//! Where the Metal decode step goes, at the served shapes.
//!
//! ## Why this exists
//!
//! The fused Metal path decodes at 33.6 tok/s, which is 46 GB/s effective on
//! a chip whose DENSE arm reaches 73. It moves less data and goes slower per
//! byte, so the cost is not bandwidth. Raising candle's
//! `CANDLE_METAL_COMPUTE_PER_BUFFER` from 50 to 10,000 made it SLOWER, 33.9 to
//! 27.7, so it is not the submission cadence either.
//!
//! Both of those were guesses that measurement refused. This times the two
//! kernels themselves, at the shapes the served 4B actually launches, and
//! multiplies by the launches a token costs. What it cannot explain is then
//! the host's, and that is a different search.
//!
//! Prints microseconds a launch and the projected milliseconds a token. Each
//! figure is `K` dispatches in ONE command buffer divided by `K`, so the
//! submission is paid once and what is reported is the marginal cost.
//!
//! ⚠️ The passes do NOT overlap, and that is deliberate: every dispatch binds
//! the same output buffer, so the write-write hazard serialises them. Giving
//! each its own output would measure something prettier and false.
//!
//! It reads no model and costs nothing.

#![cfg(target_os = "macos")]

use llvq_artifact::tetra48::{transcode_tetra48, TETRA48_SHELLS};
use llvq_core::{SplitMix64, DIM};
use llvq_metal::Kernel;
use std::ffi::c_void;

const TETRA_SRC: &str = include_str!("../../llvq-llm/kernels/llvq_tetra48.metal");
const ROT_SRC: &str = include_str!("../../llvq-llm/kernels/llvq_rot.metal");

/// Qwen3-4B, from `config.json`.
const HIDDEN: usize = 2560;
const INTERMEDIATE: usize = 9728;
/// The served config's launch counts, printed by the loader.
const MATVEC_LAUNCHES: usize = 252;
const ROT_LAUNCHES: usize = 144;

const TILE: usize = 64;
const LANES: usize = 32;
const GROUP: usize = 256;
const ROUNDS: usize = 12;
/// Dispatches inside ONE command buffer.
///
/// A synchronous Metal submit costs about 0.15 ms, which `Kernel::dispatch`'s
/// own doc records and which a first version of this file walked straight
/// into: it timed commit-and-wait per launch and projected 139 ms a token
/// against a step that measures 29.8. With `k` back to back the commit is
/// paid once and what is left is the marginal cost a real decode pays.
const K: usize = 64;

fn invnorm() -> Vec<f32> {
    let mut t = vec![0.0f32; TETRA48_SHELLS];
    for (m, e) in t.iter_mut().enumerate().skip(1) {
        *e = (1.0f64 / ((16 * m) as f64).sqrt()) as f32;
    }
    t
}

/// The median of `ROUNDS`, first discarded, the protocol `docs/METHODE.md`
/// asks for. A mean would be moved by one scheduling hiccup.
fn median(mut v: Vec<f64>) -> f64 {
    v.remove(0);
    v.sort_by(|a, b| a.partial_cmp(b).expect("no NaN"));
    v[v.len() / 2]
}

fn time_matvec(d_out: usize, d_in: usize) -> f64 {
    time_matvec_named(d_out, d_in, "tv_tetra48_metal")
}

fn time_matvec_named(d_out: usize, d_in: usize, name: &str) -> f64 {
    time_matvec_tile(d_out, d_in, name, TILE)
}

/// The same, at a chosen tile. The tile is a host-injected `#define`, the
/// shape the CUDA side uses through NVRTC.
fn time_matvec_tile(d_out: usize, d_in: usize, name: &str, tile: usize) -> f64 {
    let pinned = name.ends_with("_tg");
    let src = format!("#define LLVQ_TILE_BLOCKS {tile}u\n{TETRA_SRC}");
    let nblocks = d_in / DIM;
    let tail_w = d_in % DIM;
    let mut rng = SplitMix64::new(0x7E_4C01);
    let n = d_out * nblocks;
    let indices: Vec<u64> = (0..n)
        .map(|_| 1 + rng.next() % (llvq_search::index::N13.min(1u64 << 47) - 1))
        .collect();
    let gains: Vec<u32> = (0..n).map(|_| (rng.next() & 1) as u32).collect();
    let stream = transcode_tetra48(&indices, &gains, d_out, nblocks).expect("transcodes");

    let k = Kernel::new_exact(&src, name).expect("compiles");
    let table = llvq_bench::f1::rank::RankTable::build();
    let tr = llvq_bench::f1::Trellis::new();
    let b_words = k.buffer(&stream.data);
    let b_rows = k.buffer(&table.rows);
    let b_pref = k.buffer(&llvq_bench::f1::rank::prefix_bytes(&tr));
    let b_bran = k.buffer(&llvq_bench::f1::rank::branch_words(&tr));
    let b_suff = k.buffer(&llvq_bench::f1::rank::suffix_bytes(&tr));
    let b_gs = k.buffer(&[0.625f32, 1.375]);
    let b_iv = k.buffer(&invnorm());
    let b_rs = k.buffer(&vec![1.0f32; d_out]);
    let b_tail = k.buffer(&vec![0u16; (d_out * tail_w).max(1)]);
    let b_x = k.buffer(&vec![0.5f32; d_in]);
    let b_y = k.empty::<f32>(d_out);
    let (stride, nb, tw) = (stream.stride_u32 as u32, nblocks as u32, tail_w as u32);
    let tg = (tile * DIM * 4) as u64;

    let mut t = Vec::with_capacity(ROUNDS);
    for _ in 0..ROUNDS {
        let r = k.dispatch_many((d_out * LANES) as u64, GROUP as u64, K, |enc, _| {
            enc.set_buffer(0, Some(&b_words), 0);
            enc.set_bytes(1, 4, &stride as *const u32 as *const c_void);
            enc.set_buffer(2, Some(&b_rows), 0);
            enc.set_buffer(3, Some(&b_pref), 0);
            enc.set_buffer(4, Some(&b_bran), 0);
            enc.set_buffer(5, Some(&b_suff), 0);
            enc.set_buffer(6, Some(&b_gs), 0);
            enc.set_buffer(7, Some(&b_iv), 0);
            enc.set_buffer(8, Some(&b_rs), 0);
            enc.set_buffer(9, Some(&b_tail), 0);
            enc.set_buffer(10, Some(&b_x), 0);
            enc.set_buffer(11, Some(&b_y), 0);
            enc.set_bytes(12, 4, &nb as *const u32 as *const c_void);
            enc.set_bytes(13, 4, &tw as *const u32 as *const c_void);
            enc.set_threadgroup_memory_length(0, tg);
            if pinned {
                enc.set_threadgroup_memory_length(1, 4096 * 4);
                enc.set_threadgroup_memory_length(2, 1024 * 2);
                enc.set_threadgroup_memory_length(3, 128);
                enc.set_threadgroup_memory_length(4, 128);
            }
        });
        t.push(r.seconds * 1e6 / K as f64);
    }
    median(t)
}

fn time_rotation(n: usize) -> f64 {
    let k = Kernel::new_exact(ROT_SRC, "rot_apply_metal").expect("compiles");
    let b_xin = k.buffer(&vec![0x3c00u16; n]);
    let b_sign = k.buffer(&vec![0u32; n.div_ceil(32)]);
    let b_small = k.buffer(&[0.0f32]);
    let b_out = k.empty::<f32>(n);
    let b_scratch = k.empty::<f32>(n);
    let m = n.next_power_of_two().min(n);
    let (nn, mm, kk) = (n as u32, m as u32, 1u32);
    let inv = 1.0f32 / (m as f32).sqrt();
    let x_off = 0u32;
    let threads = (n as u32).next_power_of_two().clamp(32, 1024) as u64;

    let mut t = Vec::with_capacity(ROUNDS);
    for _ in 0..ROUNDS {
        let r = k.dispatch_many(threads, threads, K, |enc, _| {
            enc.set_buffer(0, Some(&b_xin), 0);
            enc.set_buffer(1, Some(&b_sign), 0);
            enc.set_buffer(2, Some(&b_small), 0);
            enc.set_buffer(3, Some(&b_out), 0);
            enc.set_buffer(4, Some(&b_scratch), 0);
            enc.set_bytes(5, 4, &nn as *const u32 as *const c_void);
            enc.set_bytes(6, 4, &mm as *const u32 as *const c_void);
            enc.set_bytes(7, 4, &kk as *const u32 as *const c_void);
            enc.set_bytes(8, 4, &inv as *const f32 as *const c_void);
            enc.set_bytes(9, 4, &x_off as *const u32 as *const c_void);
        });
        t.push(r.seconds * 1e6 / K as f64);
    }
    median(t)
}

fn main() {
    println!("Qwen3-4B served shapes, medians of {ROUNDS} rounds, first discarded\n");

    // The served 4B: 216 Tetra matrices. q/k/o are hidden-wide, gate/up read
    // hidden and write intermediate, down reads intermediate.
    let mv_hidden = time_matvec(HIDDEN, HIDDEN);
    let mv_up = time_matvec(INTERMEDIATE, HIDDEN);
    let mv_down = time_matvec(HIDDEN, INTERMEDIATE);
    println!("matvec  {HIDDEN}x{HIDDEN}      {mv_hidden:8.1} us");
    println!("matvec  {INTERMEDIATE}x{HIDDEN}      {mv_up:8.1} us");
    println!("matvec  {HIDDEN}x{INTERMEDIATE}      {mv_down:8.1} us");

    let rot_hidden = time_rotation(HIDDEN);
    let rot_inter = time_rotation(INTERMEDIATE);
    println!("rotation n={HIDDEN}       {rot_hidden:8.1} us");
    println!("rotation n={INTERMEDIATE}       {rot_inter:8.1} us\n");

    // The same three shapes with the decoder tables pinned in threadgroup
    // memory. Proven equal by `the_pinned_variant_matches_the_host_too`.
    let tg_hidden = time_matvec_named(HIDDEN, HIDDEN, "tv_tetra48_metal_tg");
    let tg_up = time_matvec_named(INTERMEDIATE, HIDDEN, "tv_tetra48_metal_tg");
    let tg_down = time_matvec_named(HIDDEN, INTERMEDIATE, "tv_tetra48_metal_tg");
    println!("tables pinned in threadgroup memory:");
    println!("  {HIDDEN}x{HIDDEN}   {tg_hidden:8.1} us  {:+6.1} %", 100.0 * (tg_hidden / mv_hidden - 1.0));
    println!("  {INTERMEDIATE}x{HIDDEN}   {tg_up:8.1} us  {:+6.1} %", 100.0 * (tg_up / mv_up - 1.0));
    println!("  {HIDDEN}x{INTERMEDIATE}   {tg_down:8.1} us  {:+6.1} %", 100.0 * (tg_down / mv_down - 1.0));
    let tg_total = 36.0 * (3.0 * tg_hidden + 2.0 * tg_up + tg_down) / 1000.0;
    println!("  252 launches          {tg_total:7.2} ms a token\n");

    // The value computed instead of gathered, which drops the PRMT emulation
    // the CUDA v3 representation forces. Proven equal by the same gate.
    let ar_hidden = time_matvec_named(HIDDEN, HIDDEN, "tv_tetra48_metal_ar");
    let ar_up = time_matvec_named(INTERMEDIATE, HIDDEN, "tv_tetra48_metal_ar");
    let ar_down = time_matvec_named(HIDDEN, INTERMEDIATE, "tv_tetra48_metal_ar");
    println!("value computed, no PRMT emulation:");
    println!("  {HIDDEN}x{HIDDEN}   {ar_hidden:8.1} us  {:+6.1} %", 100.0 * (ar_hidden / mv_hidden - 1.0));
    println!("  {INTERMEDIATE}x{HIDDEN}   {ar_up:8.1} us  {:+6.1} %", 100.0 * (ar_up / mv_up - 1.0));
    println!("  {HIDDEN}x{INTERMEDIATE}   {ar_down:8.1} us  {:+6.1} %", 100.0 * (ar_down / mv_down - 1.0));
    let ar_total = 36.0 * (3.0 * ar_hidden + 2.0 * ar_up + ar_down) / 1000.0;
    println!("  252 launches          {ar_total:7.2} ms a token\n");

    // 36 layers. Per layer: q, k, o at hidden, gate and up at intermediate
    // output, down at intermediate input. v_proj is int4 and is not timed here.
    let per_layer_mv = 3.0 * mv_hidden + 2.0 * mv_up + mv_down;
    let mv_total = 36.0 * per_layer_mv / 1000.0;
    // `rot_share=1`: four rotations a layer, one per activation site.
    let rot_total = 36.0 * (3.0 * rot_hidden + rot_inter) / 1000.0;
    println!("projected, 36 layers:");
    println!("  {MATVEC_LAUNCHES} matvec launches   {mv_total:7.2} ms a token");
    println!("  {ROT_LAUNCHES} rotation launches {rot_total:7.2} ms a token");
    println!("  sum                    {:7.2} ms", mv_total + rot_total);
    // The tile is a host-injected define and the served value, 64, was
    // measured on sm_89 and never here. Apple's threadgroup memory is not
    // shared with a cache, so the mechanism that made it matter on NVIDIA
    // does not obviously transfer.
    // The value looked up in 64 constant floats instead of computed.
    let lu_hidden = time_matvec_named(HIDDEN, HIDDEN, "tv_tetra48_metal_lut");
    let lu_up = time_matvec_named(INTERMEDIATE, HIDDEN, "tv_tetra48_metal_lut");
    let lu_down = time_matvec_named(HIDDEN, INTERMEDIATE, "tv_tetra48_metal_lut");
    println!("value looked up in 64 constant floats:");
    println!("  {HIDDEN}x{HIDDEN}   {lu_hidden:8.1} us  {:+6.1} % vs computed", 100.0 * (lu_hidden / ar_hidden - 1.0));
    println!("  {INTERMEDIATE}x{HIDDEN}   {lu_up:8.1} us  {:+6.1} %", 100.0 * (lu_up / ar_up - 1.0));
    println!("  {HIDDEN}x{INTERMEDIATE}   {lu_down:8.1} us  {:+6.1} %", 100.0 * (lu_down / ar_down - 1.0));
    let lu_total = 36.0 * (3.0 * lu_hidden + 2.0 * lu_up + lu_down) / 1000.0;
    println!("  252 launches          {lu_total:7.2} ms a token\n");

    println!("tile sweep on the arithmetic kernel, {HIDDEN}x{HIDDEN} and {HIDDEN}x{INTERMEDIATE}:");
    for tile in [32usize, 64, 128, 256] {
        let a = time_matvec_tile(HIDDEN, HIDDEN, "tv_tetra48_metal_ar", tile);
        let b = time_matvec_tile(HIDDEN, INTERMEDIATE, "tv_tetra48_metal_ar", tile);
        println!("  tile {tile:3}   {a:7.1} us   {b:7.1} us   staged {:6} B", tile * DIM * 4);
    }
    println!();

    println!("\nthe measured step is 29.8 ms at 33.6 tok/s.");
    println!("what the sum does not reach is the host's, and is a different search.");
}
