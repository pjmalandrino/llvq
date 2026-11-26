//! A prompt's rows through the resident experts in a few launches
//! (`ResidentExperts::forward_rows_batched`) against the path it replaces,
//! `forward_general` (one expert and one chunk of 8 rows at a time).
//!
//! Class A: every f32 of the output equal. Synthetic experts at the model's
//! shapes (128 experts, 2,304 by 1,152, 8 a token) on prompts of 2, 37, 512
//! and 1,100 tokens (the last one crosses a block of `BATCH_TOKENS`), and on
//! small layers with and without tail columns, 1 to 12 experts a token. A
//! token naming an expert twice, and a runtime without the rows kernel, send
//! the caller back to the general path.
//!
//! Synthetic, a few seconds of GPU: not a heavy job, but one at a time.
#![cfg(all(target_os = "macos", feature = "metal"))]

use std::sync::{Arc, Mutex};

use candle_core::{Device, Tensor};
use llvq_artifact::tetra48::transcode_tetra48;
use llvq_core::{SplitMix64, DIM};
use llvq_llm::fused::RotationTables;
use llvq_llm::fused_metal::{MetalRuntime, MetalTuning, MoeHostExpert, MoeHostProj};
use llvq_llm::inkling::config::InklingConfig;
use llvq_llm::inkling::layers::Routing;
use llvq_llm::inkling::resident::{self, ResidentExperts};

static GPU: Mutex<()> = Mutex::new(());

fn invnorm_table() -> Vec<f32> {
    let mut t = vec![0.0f32; llvq_artifact::tetra48::TETRA48_SHELLS];
    for (m, e) in t.iter_mut().enumerate().skip(1) {
        *e = (1.0f64 / ((16 * m) as f64).sqrt()) as f32;
    }
    t
}

/// A runtime on the bench's rank tables at the served tile, rows kernel on.
fn runtime(dev: &Device, prefill_rows: usize) -> MetalRuntime {
    use llvq_bench::f1::rank::{branch_words, prefix_bytes, suffix_bytes, RankTable};
    let table = RankTable::build();
    let tr = llvq_bench::f1::Trellis::new();
    let mut rt = MetalRuntime::new(
        dev,
        &table.rows,
        &prefix_bytes(&tr),
        &branch_words(&tr),
        &suffix_bytes(&tr),
        &invnorm_table(),
        resident::SERVED_TILE,
        false,
    )
    .expect("the runtime builds");
    rt.set_tuning(MetalTuning { prefill_rows, ..MetalTuning::default() }).unwrap();
    rt
}

/// The model's MoE numbers without reading its files.
fn config() -> InklingConfig {
    let p = std::path::PathBuf::from(std::env::var("HOME").unwrap()).join("llvq-inkling-mini/textonly/config.json");
    InklingConfig::from_file(&p).expect("the text config of Inkling 2 Mini")
}

fn rec(name: String, seed: u64, d_out: usize, nblocks: usize, tail_w: usize) -> MoeHostProj {
    let mut rng = SplitMix64::new(seed);
    let n = d_out * nblocks;
    let indices: Vec<u64> = (0..n)
        .map(|_| match rng.next().is_multiple_of(6) {
            true => 0,
            false => 1 + rng.next() % (llvq_search::index::N13.min(1u64 << 47) - 1),
        })
        .collect();
    let gains: Vec<u32> = (0..n).map(|_| (rng.next() & 1) as u32).collect();
    let s = transcode_tetra48(&indices, &gains, d_out, nblocks).expect("transcodes");
    let words = s.data.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let rscale = (0..d_out).map(|_| (0.002 + 0.004 * rng.next_gaussian().abs()) as f32).collect();
    let tail = (0..d_out * tail_w).map(|_| half::f16::from_f32(0.05 * rng.next_gaussian() as f32).to_bits()).collect();
    let g0 = 0.5 + 0.25 * (rng.next() % 4) as f32;
    MoeHostProj {
        name,
        d_out,
        d_in: nblocks * DIM + tail_w,
        nblocks,
        row_stride_u32: s.stride_u32,
        words,
        gscale: [g0, g0 + 0.75],
        rscale,
        tail,
    }
}

fn hosts(n: usize, hidden: usize, inter: usize, tail: usize, salt: u64) -> Vec<MoeHostExpert> {
    (0..n)
        .map(|e| {
            let s = salt + 8 * e as u64;
            let mk = |m: &str, seed: u64, d_out: usize, d_in: usize| {
                rec(format!("{e}.{m}"), seed, d_out, (d_in - tail) / DIM, tail)
            };
            MoeHostExpert {
                projs: [mk("gate", s, inter, hidden), mk("up", s + 1, inter, hidden), mk("down", s + 2, hidden, inter)],
                rot_in: RotationTables::build(hidden, s + 3).unwrap(),
                rot_down: RotationTables::build(inter, s + 4).unwrap(),
            }
        })
        .collect()
}

/// `t` tokens, each routed to `k` distinct experts of `n` in a random order.
/// `skew` sends half the tokens to the first experts, so some experts take
/// many chunks, some one row, some none.
fn routing(seed: u64, t: usize, n: usize, k: usize, skew: bool) -> Routing {
    let mut rng = SplitMix64::new(seed);
    let mut r = Routing { indices: vec![], weights: vec![], margin: vec![], scores: vec![] };
    for i in 0..t {
        let span = if skew && i % 2 == 0 { (k + 3).min(n) } else { n };
        let mut pool: Vec<u32> = (0..span as u32).collect();
        let mut idx = vec![];
        for _ in 0..k {
            idx.push(pool.remove((rng.next() % pool.len() as u64) as usize));
        }
        // Some weights negative, so a product can be -0 and the +0 start shows.
        r.weights.push((0..k).map(|_| (0.1 + rng.next_gaussian().abs()) as f32 * if rng.next().is_multiple_of(7) { -1.0 } else { 1.0 }).collect());
        r.indices.push(idx);
        r.margin.push(0.0);
        r.scores.push(vec![]);
    }
    r
}

fn values(seed: u64, n: usize) -> Vec<f32> {
    let mut rng = SplitMix64::new(seed);
    (0..n).map(|_| rng.next_gaussian() as f32).collect()
}

fn to_vec(t: &Tensor) -> Vec<f32> {
    t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
}

fn first_difference(a: &[f32], b: &[f32]) -> Option<(usize, f32, f32)> {
    assert_eq!(a.len(), b.len(), "lengths");
    a.iter().zip(b).enumerate().find(|(_, (p, q))| p.to_bits() != q.to_bits()).map(|(i, (p, q))| (i, *p, *q))
}

fn batched_against_general(layer: &ResidentExperts, cfg: &InklingConfig, dev: &Device, seed: u64, t: usize, skew: bool) {
    let r = routing(seed, t, layer.count(), cfg.num_experts_per_tok, skew);
    let x = Tensor::from_slice(&values(seed ^ 0x55, t * cfg.hidden_size), (t, cfg.hidden_size), dev).unwrap();
    let old = to_vec(&layer.forward_general(&x, &r, cfg).unwrap());
    let new = to_vec(&layer.forward_rows_batched(&x, &r, cfg).unwrap().expect("the batched path runs"));
    assert!(old.iter().filter(|v| **v != 0.0).count() > old.len() / 2, "zeros would pass every equality");
    assert_eq!(first_difference(&new, &old), None, "{t} tokens, k {}, skew {skew}", cfg.num_experts_per_tok);
}

#[test]
fn batched_rows_are_the_general_path_at_model_shapes() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let dev = Device::new_metal(0).expect("Metal");
    let cfg = config();
    let rt = Arc::new(runtime(&dev, 8));
    let hs = hosts(cfg.n_routed_experts, cfg.hidden_size, cfg.intermediate_size, 0, 0xB700);
    let mut layer = ResidentExperts::from_host(rt, &dev, 6, &hs).expect("slab layer");
    drop(hs);
    assert!(layer.batch_rows_available(), "a uniform slab with the rows kernel");
    for (i, (t, skew)) in [(2usize, false), (37, true), (512, false), (1100, true)].into_iter().enumerate() {
        batched_against_general(&layer, &cfg, &dev, 0xB710 + i as u64, t, skew);
    }
    // The switch sends `forward` to the batched path and back, the same bits.
    let r = routing(0xB7F0, 20, layer.count(), cfg.num_experts_per_tok, true);
    let x = Tensor::from_slice(&values(0xB7F1, 20 * cfg.hidden_size), (20, cfg.hidden_size), &dev).unwrap();
    layer.take_launches();
    let off = to_vec(&layer.forward(&x, &r, &cfg).unwrap());
    let l_off = layer.take_launches();
    layer.set_batch_rows(true);
    let on = to_vec(&layer.forward(&x, &r, &cfg).unwrap());
    let l_on = layer.take_launches();
    assert_eq!(first_difference(&on, &off), None, "forward, switch on against off");
    assert_eq!(l_on, 4, "four lattice launches a block");
    assert!(l_off > 40, "the general path launches per expert: {l_off}");
    // A token naming an expert twice goes back to the general path.
    let mut rep = r.clone();
    rep.indices[3][1] = rep.indices[3][0];
    assert!(layer.forward_rows_batched(&x, &rep, &cfg).unwrap().is_none(), "a repeated expert is not batched");
    let general = to_vec(&layer.forward_general(&x, &rep, &cfg).unwrap());
    assert_eq!(first_difference(&to_vec(&layer.forward(&x, &rep, &cfg).unwrap()), &general), None, "repeated expert");
}

#[test]
fn batched_rows_are_the_general_path_on_small_layers() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let dev = Device::new_metal(0).expect("Metal");
    let base = config();
    let rt = Arc::new(runtime(&dev, 8));
    for (hidden, inter, tail, salt) in [(768usize, 384usize, 0usize, 0xB800u64), (1024, 400, 16, 0xB900)] {
        let mut cfg = base.clone();
        cfg.hidden_size = hidden;
        cfg.intermediate_size = inter;
        let hs = hosts(12, hidden, inter, tail, salt);
        let layer = ResidentExperts::from_host(rt.clone(), &dev, 6, &hs).expect("slab layer");
        for k in [1usize, 2, 5, 8, 12] {
            cfg.num_experts_per_tok = k;
            batched_against_general(&layer, &cfg, &dev, salt + 16 * k as u64, 29, k < 8);
        }
    }
}

#[test]
fn batched_rows_need_the_rows_kernel() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let dev = Device::new_metal(0).expect("Metal");
    let mut cfg = config();
    cfg.hidden_size = 768;
    cfg.intermediate_size = 384;
    let hs = hosts(12, 768, 384, 0, 0xBA00);
    let layer = ResidentExperts::from_host(Arc::new(runtime(&dev, 1)), &dev, 6, &hs).expect("slab layer");
    assert!(!layer.batch_rows_available(), "one row a launch: no rows kernel to reproduce");
    let r = routing(0xBA10, 9, 12, cfg.num_experts_per_tok, false);
    let x = Tensor::from_slice(&values(0xBA11, 9 * 768), (9, 768), &dev).unwrap();
    assert!(layer.forward_rows_batched(&x, &r, &cfg).unwrap().is_none());
}

/// Where a layer visit of 512 tokens goes in the batched path, at the
/// model's shapes on synthetic codes: each stage timed alone over 20 runs,
/// the device drained around it. Ignored: run by name under the guard.
#[test]
#[ignore = "times the GPU; run under the guard's lock"]
fn batched_rows_stage_times() {
    use llvq_llm::fused_metal::MoeRowsMats;
    use std::time::Instant;
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let dev = Device::new_metal(0).expect("Metal");
    let cfg = config();
    let rt = Arc::new(runtime(&dev, 8));
    let hs = hosts(cfg.n_routed_experts, cfg.hidden_size, cfg.intermediate_size, 0, 0xBB00);
    let layer = ResidentExperts::from_host(rt.clone(), &dev, 6, &hs).expect("slab layer");
    drop(hs);
    let slab = layer.slab().expect("a slab");
    let (t, k, h, inter) = (512usize, cfg.num_experts_per_tok, cfg.hidden_size, cfg.intermediate_size);
    let r = routing(0xBB10, t, layer.count(), k, false);
    let x = Tensor::from_slice(&values(0xBB11, t * h), (t, h), &dev).unwrap();
    let mut pairs: Vec<(u32, u32)> = vec![];
    for (ti, idx) in r.indices.iter().enumerate() {
        for &e in idx {
            pairs.push((e, ti as u32));
        }
    }
    pairs.sort();
    let mut chunks = vec![];
    let mut s = 0;
    while s < pairs.len() {
        let e = pairs[s].0;
        let mut end = s;
        while end < pairs.len() && end - s < 8 && pairs[end].0 == e {
            end += 1;
        }
        chunks.extend([e, s as u32, (end - s) as u32]);
        s = end;
    }
    let n = pairs.len();
    let pe = Tensor::from_vec(pairs.iter().map(|p| p.0).collect::<Vec<_>>(), n, &dev).unwrap();
    let pt = Tensor::from_vec(pairs.iter().map(|p| p.1).collect::<Vec<_>>(), n, &dev).unwrap();
    let ch = Tensor::from_vec(chunks.clone(), chunks.len(), &dev).unwrap();
    let xs = x.index_select(&pt, 0).unwrap().to_dtype(candle_core::DType::F16).unwrap();
    let xr = rt.moe_rotate_pairs(slab, false, &xs, &pe).unwrap();
    let gu = rt.moe_rows(slab, MoeRowsMats::GateUp, &xr, &ch).unwrap();
    let a16 = gu.get(0).unwrap().to_dtype(candle_core::DType::F16).unwrap();
    let ar = rt.moe_rotate_pairs(slab, true, &a16, &pe).unwrap();
    dev.synchronize().unwrap();
    let time = |what: &str, f: &dyn Fn()| {
        f();
        dev.synchronize().unwrap();
        let t0 = Instant::now();
        for _ in 0..20 {
            f();
        }
        dev.synchronize().unwrap();
        println!("{what}: {:.2} ms", t0.elapsed().as_secs_f64() * 1e3 / 20.0);
    };
    println!("{n} pairs, {} chunks, hidden {h}, inter {inter}", chunks.len() / 3);
    time("gather + f16", &|| drop(x.index_select(&pt, 0).unwrap().to_dtype(candle_core::DType::F16).unwrap()));
    time("rotation in", &|| drop(rt.moe_rotate_pairs(slab, false, &xs, &pe).unwrap()));
    time("rotation down", &|| drop(rt.moe_rotate_pairs(slab, true, &a16, &pe).unwrap()));
    time("gate and up", &|| drop(rt.moe_rows(slab, MoeRowsMats::GateUp, &xr, &ch).unwrap()));
    time("down", &|| drop(rt.moe_rows(slab, MoeRowsMats::Down, &ar, &ch).unwrap()));
    time("whole batched call", &|| drop(layer.forward_rows_batched(&x, &r, &cfg).unwrap().unwrap()));
    time("general path", &|| drop(layer.forward_general(&x, &r, &cfg).unwrap()));
}
