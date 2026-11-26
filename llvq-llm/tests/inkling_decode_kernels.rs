//! A bit-for-bit variant of the fused one-row MoE call of Inkling 2 Mini on
//! Metal (prereg `proofs/preregistration-inkling-mac-decode-2026-11-26.md`):
//! `moe_tetra_down_sg_metal`, `down` with one simdgroup a selected slot and
//! the slots summed in threadgroup memory, instead of one simdgroup a row
//! looping over the slots ([`MetalRuntime::set_moe_down_sg`]).
//!
//! The gate: on synthetic experts at the model's shapes and on small layers
//! with tails, the fused call with the switch on and off returns every f32 of
//! the one-expert path. The bench (ignored) times the launch and the whole
//! call chained, the device drained only around the chain.
//!
//! The same lot staged the two rotations of the call in threadgroup memory
//! (`moe_rot_tg_metal`): bit for bit too, but 12.14 against 11.75 us on the
//! input rotation, 9.80 against 9.83 on the down one and nothing on the whole
//! call (`docs/mesures/inkling-mac-decode-2026-11-26-brut/kbench-1.log`), so it
//! was removed.

#![cfg(all(target_os = "macos", feature = "metal"))]

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use candle_core::{Device, Tensor};
use llvq_artifact::tetra48::transcode_tetra48;
use llvq_core::{SplitMix64, DIM};
use llvq_llm::fused::RotationTables;
use llvq_llm::fused_metal::{MetalRuntime, MoeHostExpert, MoeHostProj};
use llvq_llm::inkling::config::InklingConfig;
use llvq_llm::inkling::layers::Routing;
use llvq_llm::inkling::resident::{self, ResidentExperts};

static GPU: Mutex<()> = Mutex::new(());

fn gpu() -> std::sync::MutexGuard<'static, ()> {
    GPU.lock().unwrap_or_else(|e| e.into_inner())
}

fn config() -> InklingConfig {
    let p = PathBuf::from(std::env::var("HOME").expect("HOME")).join("llvq-inkling-mini/textonly/config.json");
    assert!(p.exists(), "missing {}: the model's config gives the shapes", p.display());
    InklingConfig::from_file(&p).expect("config")
}

fn invnorm_table() -> Vec<f32> {
    let mut t = vec![0.0f32; llvq_artifact::tetra48::TETRA48_SHELLS];
    for (m, e) in t.iter_mut().enumerate().skip(1) {
        *e = (1.0f64 / ((16 * m) as f64).sqrt()) as f32;
    }
    t
}

fn runtime(dev: &Device) -> MetalRuntime {
    use llvq_bench::f1::rank::{branch_words, prefix_bytes, suffix_bytes, RankTable};
    let table = RankTable::build();
    let tr = llvq_bench::f1::Trellis::new();
    MetalRuntime::new(
        dev,
        &table.rows,
        &prefix_bytes(&tr),
        &branch_words(&tr),
        &suffix_bytes(&tr),
        &invnorm_table(),
        resident::SERVED_TILE,
        false,
    )
    .expect("the runtime builds")
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

fn routing(seed: u64, n: usize, k: usize) -> Routing {
    let mut rng = SplitMix64::new(seed);
    let mut pool: Vec<u32> = (0..n as u32).collect();
    let mut idx = vec![];
    for _ in 0..k {
        idx.push(pool.remove((rng.next() % pool.len() as u64) as usize));
    }
    let w = (0..k).map(|_| (0.1 + rng.next_gaussian().abs()) as f32).collect();
    Routing { indices: vec![idx], weights: vec![w], margin: vec![0.0], scores: vec![vec![]] }
}

fn values(seed: u64, n: usize) -> Vec<f32> {
    let mut rng = SplitMix64::new(seed);
    (0..n).map(|_| rng.next_gaussian() as f32).collect()
}

fn to_vec(t: &Tensor) -> Vec<f32> {
    t.flatten_all().unwrap().to_dtype(candle_core::DType::F32).unwrap().to_vec1().unwrap()
}

fn first_difference(a: &[f32], b: &[f32]) -> Option<(usize, f32, f32)> {
    assert_eq!(a.len(), b.len(), "lengths");
    a.iter().zip(b).enumerate().find(|(_, (p, q))| p.to_bits() != q.to_bits()).map(|(i, (p, q))| (i, *p, *q))
}

const ARMS: [bool; 2] = [false, true];

/// Every arm of the two switches against the one-expert path, `count`
/// routings of `k` experts.
fn arms_against_one_row(rt: &MetalRuntime, layer: &ResidentExperts, cfg: &InklingConfig, dev: &Device, seed: u64, count: u64) {
    for i in 0..count {
        let r = routing(seed + i, layer.count(), cfg.num_experts_per_tok);
        let x = Tensor::from_slice(&values(seed ^ (i << 8), cfg.hidden_size), (1, cfg.hidden_size), dev).unwrap();
        let want = to_vec(&layer.forward_one_row(&x, &r, cfg).unwrap().expect("distinct experts"));
        assert!(want.iter().any(|v| *v != 0.0), "zeros would pass every equality");
        for sg in ARMS {
            rt.set_moe_down_sg(sg);
            let got = to_vec(&layer.forward_one_row_fused(&x, &r, cfg).unwrap().expect("the fused path runs"));
            assert_eq!(first_difference(&got, &want), None, "routing {i} {:?}, down_sg {sg}", r.indices[0]);
        }
    }
    rt.set_moe_down_sg(llvq_llm::fused_metal::MOE_DOWN_SG_DEFAULT);
}

/// 128 synthetic experts at the model's shapes, 24 routings of 8.
#[test]
fn decode_variants_are_one_row_at_model_shapes() {
    let _g = gpu();
    let dev = Device::new_metal(0).expect("Metal");
    let cfg = config();
    let rt = Arc::new(runtime(&dev));
    let hs = hosts(cfg.n_routed_experts, cfg.hidden_size, cfg.intermediate_size, 0, 0xDC00);
    let layer = ResidentExperts::from_host(rt.clone(), &dev, 6, &hs).expect("slab layer");
    drop(hs);
    assert!(layer.fused_available(), "a uniform slab on the served tuning has a fused path");
    arms_against_one_row(&rt, &layer, &cfg, &dev, 0xDC10, 24);
    let p = rt.compiled_pipelines();
    for k in ["moe_tetra_down_metal", "moe_tetra_down_sg_metal"] {
        assert!(p.contains(&k), "{k} did not run");
    }
}

/// Two small shapes, one with tails and another odd factor, 1 to 12 experts
/// a call (one simdgroup a slot up to 12 simdgroups a threadgroup).
#[test]
fn decode_variants_are_one_row_on_small_layers() {
    let _g = gpu();
    let dev = Device::new_metal(0).expect("Metal");
    let base = config();
    let rt = Arc::new(runtime(&dev));
    for (hidden, inter, tail, salt) in [(768usize, 384usize, 0usize, 0xDD00u64), (1024, 400, 16, 0xDE00)] {
        let mut cfg = base.clone();
        cfg.hidden_size = hidden;
        cfg.intermediate_size = inter;
        let hs = hosts(12, hidden, inter, tail, salt);
        let layer = ResidentExperts::from_host(rt.clone(), &dev, 6, &hs).expect("slab layer");
        assert!(layer.fused_available(), "{hidden}x{inter}");
        for k in [1usize, 2, 5, 8, 12] {
            cfg.num_experts_per_tok = k;
            arms_against_one_row(&rt, &layer, &cfg, &dev, salt + 16 * k as u64, 3);
        }
    }
}

/// Each launch of the fused call, chained `reps` times with the device
/// drained only around the chain, the served kernel and its variant
/// alternated over `rounds` rounds. Microseconds a launch.
#[test]
#[ignore = "times the GPU; run under the guard's lock"]
fn decode_variants_kernel_bench() {
    let _g = gpu();
    let dev = Device::new_metal(0).expect("Metal");
    let cfg = config();
    let rt = Arc::new(runtime(&dev));
    let hs = hosts(cfg.n_routed_experts, cfg.hidden_size, cfg.intermediate_size, 0, 0xDB00);
    let layer = ResidentExperts::from_host(rt.clone(), &dev, 6, &hs).expect("slab layer");
    drop(hs);
    let slab = layer.slab().expect("a slab");
    let (k, h, inter) = (cfg.num_experts_per_tok, cfg.hidden_size, cfg.intermediate_size);
    let r = routing(0xDB10, layer.count(), k);
    let mut order: Vec<(u32, f32)> = r.indices[0].iter().copied().zip(r.weights[0].iter().copied()).collect();
    order.sort_by_key(|p| p.0);
    let mut sel: Vec<u32> = order.iter().map(|p| p.0).collect();
    sel.extend(order.iter().map(|p| p.1.to_bits()));
    let sel = Tensor::from_vec(sel, 2 * k, &dev).unwrap();
    let x16 = Tensor::from_slice(&values(0xDB11, h), (1, h), &dev).unwrap().to_dtype(candle_core::DType::F16).unwrap();
    let ar = Tensor::from_slice(&values(0xDB13, k * inter), (k, inter), &dev).unwrap();
    let (reps, rounds) = (400usize, 7usize);
    let time = |f: &dyn Fn() -> Tensor| -> f64 {
        let _ = f();
        dev.synchronize().unwrap();
        let t = Instant::now();
        let mut last = None;
        for _ in 0..reps {
            last = Some(f());
        }
        dev.synchronize().unwrap();
        drop(last);
        1e6 * t.elapsed().as_secs_f64() / reps as f64
    };
    let median = |mut v: Vec<f64>| {
        v.sort_by(|a, b| a.total_cmp(b));
        v[v.len() / 2]
    };
    let mut rows = vec![];
    type Launch<'a> = (&'a str, Box<dyn Fn(bool) + 'a>, Box<dyn Fn() -> Tensor + 'a>);
    let launches: Vec<Launch> = vec![
        ("rotation in, n 2304 (no switch: the served kernel twice)", Box::new(|_| ()), Box::new(|| rt.moe_rotate(slab, false, &x16, &sel).unwrap())),
        ("down and sum, 2304 rows", Box::new(|on| rt.set_moe_down_sg(on)), Box::new(|| rt.moe_down_sum(slab, &ar, &sel).unwrap())),
    ];
    for (name, set, f) in &launches {
        let (mut a, mut b) = (vec![], vec![]);
        for _ in 0..rounds {
            set(false);
            a.push(time(f.as_ref()));
            set(true);
            b.push(time(f.as_ref()));
        }
        set(false);
        let (ma, mb) = (median(a.clone()), median(b.clone()));
        println!("{name}: served {ma:.2} us, variant {mb:.2} us, {:.3}x; rounds served {a:.2?}, variant {b:.2?}", ma / mb);
        rows.push(serde_json::json!({"launch": name, "served_us": a, "variant_us": b, "median_served": ma, "median_variant": mb}));
    }
    // The whole fused call, the activation included.
    let x = Tensor::from_slice(&values(0xDB14, h), (1, h), &dev).unwrap();
    let call = || layer.forward_one_row_fused(&x, &r, &cfg).unwrap().unwrap();
    let mut per_arm = vec![vec![]; ARMS.len()];
    for _ in 0..rounds {
        for (i, sg) in ARMS.iter().enumerate() {
            rt.set_moe_down_sg(*sg);
            per_arm[i].push(time(&call));
        }
    }
    rt.set_moe_down_sg(llvq_llm::fused_metal::MOE_DOWN_SG_DEFAULT);
    for (i, sg) in ARMS.iter().enumerate() {
        println!("whole call, down_sg {sg}: {:.2} us, rounds {:.2?}", median(per_arm[i].clone()), per_arm[i]);
    }
    rows.push(serde_json::json!({"launch": "whole call", "arms": ARMS, "us": per_arm}));
    println!("{}", serde_json::json!({"reps": reps, "rounds": rounds, "rows": rows}));
}
