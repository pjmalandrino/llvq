//! The routed experts of one Inkling 2 Mini layer computed from their lattice
//! codes resident on the Metal GPU (`inkling::resident`), against the dense
//! reference `inkling::layers::experts_forward` run on the reconstructions
//! `w13_hat` / `w2_hat` the same codes decode to.
//!
//! Inputs, under `INKLING_DATA` (default `~/llvq-inkling-mini`):
//! `textonly/config.json`, `native/model-00001-of-00001.safetensors` (mapped;
//! only the router and the norm of the layer are read), `lot1-c/ref_l0_5.safetensors`
//! (the reference's f32 dumps of a 64-token prompt; the first 32 rows are the
//! states of its 32-token prefix, the model being causal), and `layers/Lxx/`
//! with its `VERIFIED` file. A missing input fails the test and names it.
//! Measurements go to `lot2-g/resident_Lxx.json` under the same directory.
//! The tests are `#[ignore]`d: they use the GPU and read up to 4 GB of
//! reconstructions. Run them under the guard's lock:
//! `cargo test --release -p llvq-llm --features metal --test inkling_resident -- --ignored --nocapture --test-threads=1`.
//!
//! ## The inputs
//!
//! Layer 0: the real MoE input, the reference's `post_attention_layernorm`
//! output, with the reference's own routing (`router0`).
//! Layer 6: a PROXY. The forward of the loop layers is not written yet, so the
//! real MoE input of layer 6 does not exist in Rust; the dumps stop at layer 5.
//! The proxy is layer 5's output (the input of layer 6) through layer 6's own
//! MLP norm, routed by layer 6's own router. It has the scale and the routing
//! statistics of a real input, not its exact values.
//!
//! ## The tolerance, fixed before the first measurement
//!
//! Both arms read the same weights: the codes decode to the reconstructions bit
//! for bit (`extq verify`). What the kernel path changes is the arithmetic:
//! `MetalRuntime::rotate` reads its input in binary16, so the layer input and
//! the SwiGLU output are each narrowed once, a relative perturbation of at most
//! `u = 2^-11` an element; row scales and gain centroids are applied in f32
//! instead of f64 (`2^-24`, negligible); the sums run in another order (f32,
//! about `sqrt(K) 2^-24`, negligible). Through an orthogonal rotation and the
//! projections, each narrowing moves the output by about `u / sqrt(3)` in
//! relative Frobenius norm, so two of them by under `u`. The gate is
//! `||ours - dense||_F <= 4u ||dense||_F` (1.95e-3) and
//! `max |ours - dense| <= 8u max |dense|` (3.9e-3): four to eight times the
//! estimate, and more than two orders of magnitude under a structural error (a
//! wrong rotation or swapped gate and up decorrelate the output: relative error
//! of order 1).
//!
//! The Qwen3 fused tests do not set a tolerance against a dense arm:
//! `tests/tetra48_metal_candle.rs` demands equality with the kernel's own CPU
//! oracle (same summation order), and the served Qwen3 path is judged against
//! its dense arm by token identity in `bin/fusedrun`, not by a numeric bound.

#![cfg(all(target_os = "macos", feature = "metal"))]

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use candle_core::{DType, Device, Tensor};
use llvq_llm::inkling::config::InklingConfig;
use llvq_llm::inkling::layers::{experts_forward, rms_norm, route, Routing};
use llvq_llm::inkling::loader::NativeCheckpoint;
use llvq_llm::inkling::resident::{self, HatExperts, ResidentExperts};

const U: f64 = 1.0 / 2048.0;
const REL_FRO_TOL: f64 = 4.0 * U;
const MAX_REL_TOL: f64 = 8.0 * U;
const T_PROMPT: usize = 32;
const REPS: usize = 5;

fn data_dir() -> PathBuf {
    match std::env::var("INKLING_DATA") {
        Ok(d) => PathBuf::from(d),
        Err(_) => PathBuf::from(std::env::var("HOME").expect("HOME")).join("llvq-inkling-mini"),
    }
}

fn need(p: PathBuf) -> PathBuf {
    assert!(p.exists(), "missing input {}: the gate cannot run without it", p.display());
    p
}

fn allocated(dev: &Device) -> usize {
    match dev {
        Device::Metal(m) => m.device().current_allocated_size(),
        _ => 0,
    }
}

fn dumps() -> HashMap<String, Tensor> {
    let p = need(data_dir().join("lot1-c/ref_l0_5.safetensors"));
    candle_core::safetensors::load(&p, &Device::Cpu).expect("dumps")
}

fn get<'a>(d: &'a HashMap<String, Tensor>, k: &str) -> &'a Tensor {
    d.get(k).unwrap_or_else(|| panic!("the dumps lack {k}"))
}

/// The first `n` tokens of `r`.
fn head(r: &Routing, n: usize) -> Routing {
    Routing {
        indices: r.indices[..n].to_vec(),
        weights: r.weights[..n].to_vec(),
        margin: r.margin[..n].to_vec(),
        scores: r.scores[..n].to_vec(),
    }
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

/// Uploads layer `layer`, runs both arms on `x` / `routing`, measures, writes
/// the journal, and asserts the gate.
fn gate(layer: usize, label: &str, x: Tensor, routing: Routing) -> anyhow::Result<()> {
    let d = data_dir();
    let cfg = InklingConfig::from_file(&need(d.join("textonly/config.json")))?;
    let layer_dir = need(d.join(format!("layers/L{layer:02}")));
    need(layer_dir.join("VERIFIED"));
    let metal = Device::new_metal(0)?;

    // Upload: every expert of the layer, once.
    let alloc0 = allocated(&metal);
    let t0 = Instant::now();
    let rt = resident::served_runtime(&metal)?;
    let res = ResidentExperts::load(rt, &metal, &layer_dir, layer, &cfg)?;
    metal.synchronize()?;
    let t_load = t0.elapsed().as_secs_f64();
    let alloc1 = allocated(&metal);
    let s = res.stats.clone();
    let computed = 128.0 * 7_962_624.0 * 2.0185 / 8.0;
    println!(
        "L{layer:02} upload: {} experts, {} weights, {} B uploaded (words {} + side {} + rotations {}) against \
         {computed:.0} B computed at 2.0185 b/weight; codes files {} B; device allocated {alloc0} -> {alloc1} \
         (+{}); {t_load:.2} s (read {:.2}, transcode {:.2}, upload {:.2})",
        s.experts, s.weights, s.uploaded_bytes(), s.word_bytes, s.side_bytes, s.rotation_bytes, s.file_bytes,
        alloc1 as i64 - alloc0 as i64, s.seconds_read, s.seconds_transcode, s.seconds_upload
    );

    // The dense reference, on the CPU in f32.
    let hats = HatExperts::open(&layer_dir, &cfg)?;
    let t = Instant::now();
    let dense = experts_forward(&x, &routing, &hats, &cfg)?;
    let t_dense = t.elapsed().as_secs_f64();

    // The resident arm; the first call compiles the pipelines.
    let t = Instant::now();
    let ours = resident::experts_forward(&x, &routing, &res, &cfg)?;
    let t_first = t.elapsed().as_secs_f64();
    let launches_t = res.take_launches();

    let diff = (&ours - &dense)?;
    let fro = |t: &Tensor| -> anyhow::Result<f64> { Ok(t.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt() as f64) };
    let rel_fro = fro(&diff)? / fro(&dense)?;
    let max_abs = diff.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()? as f64;
    let ref_max = dense.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()? as f64;
    let finite = ours.flatten_all()?.to_vec1::<f32>()?.iter().all(|v| v.is_finite());
    let active = {
        let mut seen = vec![false; cfg.n_routed_experts];
        routing.indices.iter().flatten().for_each(|&e| seen[e as usize] = true);
        seen.iter().filter(|&&b| b).count()
    };
    println!(
        "L{layer:02} {label}: T={} active experts {active}; rel Frobenius {rel_fro:.3e} (tol {REL_FRO_TOL:.3e}); \
         max abs {max_abs:.3e} against max |dense| {ref_max:.3e} = {:.3e} relative (tol {MAX_REL_TOL:.3e}); \
         dense arm {t_dense:.2} s, first resident call {t_first:.3} s, {launches_t} LLVQ launches",
        x.dim(0)?,
        max_abs / ref_max
    );

    // Time: T tokens, then one token.
    let timing =|x: &Tensor, r: &Routing| -> anyhow::Result<(f64, f64, usize)> {
        let mut v = Vec::with_capacity(REPS);
        let mut launches = 0;
        for _ in 0..REPS {
            metal.synchronize()?;
            let t = Instant::now();
            let y = resident::experts_forward(x, r, &res, &cfg)?;
            let _ = y.sum_all()?.to_scalar::<f32>()?; // on the CPU: the GPU has finished
            v.push(t.elapsed().as_secs_f64());
            launches = res.take_launches();
        }
        let lo = v.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = v.iter().cloned().fold(0.0, f64::max);
        println!("    {} rounds: {:?}", REPS, v.iter().map(|s| format!("{s:.4}")).collect::<Vec<_>>());
        Ok((median(v), hi - lo, launches))
    };
    let (t_t, spread_t, l_t) = timing(&x, &routing)?;
    let x1 = x.narrow(0, 0, 1)?.contiguous()?;
    let r1 = head(&routing, 1);
    let (t_1, spread_1, l_1) = timing(&x1, &r1)?;
    println!(
        "L{layer:02} time: T={} median {t_t:.4} s (range {spread_t:.4}), {l_t} LLVQ launches; T=1 median {t_1:.4} s \
         (range {spread_1:.4}), {l_1} LLVQ launches",
        x.dim(0)?
    );

    let out_dir = d.join("lot2-g");
    std::fs::create_dir_all(&out_dir)?;
    let j = serde_json::json!({
        "layer": layer, "input": label, "tokens": x.dim(0)?, "active_experts": active,
        "tolerance": {"rel_fro": REL_FRO_TOL, "max_rel": MAX_REL_TOL},
        "measured": {
            "rel_fro": rel_fro, "max_abs": max_abs, "ref_max_abs": ref_max, "max_rel": max_abs / ref_max,
            "finite": finite,
            "upload": {
                "experts": s.experts, "weights": s.weights, "uploaded_bytes": s.uploaded_bytes(),
                "word_bytes": s.word_bytes, "side_bytes": s.side_bytes, "rotation_bytes": s.rotation_bytes,
                "codes_file_bytes": s.file_bytes, "device_allocated_before": alloc0, "device_allocated_after": alloc1,
                "seconds_total": t_load, "seconds_read": s.seconds_read, "seconds_transcode": s.seconds_transcode,
                "seconds_upload": s.seconds_upload,
            },
            "seconds_dense_arm_cpu": t_dense, "seconds_first_call": t_first,
            "time_T": {"median_s": t_t, "range_s": spread_t, "llvq_launches": l_t, "rounds": REPS},
            "time_1": {"median_s": t_1, "range_s": spread_1, "llvq_launches": l_1, "rounds": REPS},
        },
        "computed": {"uploaded_bytes_at_2_0185": computed},
    });
    let path = out_dir.join(format!("resident_L{layer:02}.json"));
    std::fs::write(&path, serde_json::to_string_pretty(&j)?)?;
    println!("journal: {}", path.display());

    assert!(finite, "L{layer:02}: a non-finite output");
    assert!(rel_fro <= REL_FRO_TOL, "L{layer:02}: relative Frobenius error {rel_fro:.3e} above {REL_FRO_TOL:.3e}");
    assert!(
        max_abs <= MAX_REL_TOL * ref_max,
        "L{layer:02}: max abs error {max_abs:.3e} above {MAX_REL_TOL:.3e} x {ref_max:.3e}"
    );
    Ok(())
}

/// Layer 6 (loop, f32 reconstructions), on the proxy input described above.
#[test]
#[ignore = "uses the GPU and reads 4 GB of reconstructions; run under the guard's lock"]
fn layer6_resident_matches_dense_reconstruction() -> anyhow::Result<()> {
    let d = data_dir();
    let cfg = InklingConfig::from_file(&need(d.join("textonly/config.json")))?;
    let ck = NativeCheckpoint::open(&need(d.join("native/model-00001-of-00001.safetensors")))?;
    let dm = dumps();
    let h5 = get(&dm, "layer5").narrow(0, 0, T_PROMPT)?.contiguous()?;
    let cpu = Device::Cpu;
    let xn = rms_norm(&h5, &ck.f32("layers.6.mlp_norm.weight", &cpu)?, cfg.rms_norm_eps)?;
    let routing = route(
        &xn,
        &ck.f32("layers.6.mlp.gate.weight", &cpu)?,
        &ck.f32("layers.6.mlp.gate.bias", &cpu)?,
        cfg.num_experts_per_tok,
        cfg.route_scale,
    )?;
    gate(6, "proxy: layer5 output, layer 6 mlp_norm and router", xn, routing)
}

/// Layer 0 (outside the loop, f16 reconstructions), on its real MoE input and
/// the reference's own routing.
#[test]
#[ignore = "uses the GPU and reads 2 GB of reconstructions; run under the guard's lock"]
fn layer0_resident_matches_dense_reconstruction() -> anyhow::Result<()> {
    let dm = dumps();
    let x = get(&dm, "L0.post_attention_layernorm").narrow(0, 0, T_PROMPT)?.contiguous()?;
    let idx = get(&dm, "router0.indices").to_dtype(DType::I64)?.to_vec2::<i64>()?;
    let w = get(&dm, "router0.weights").to_vec2::<f32>()?;
    let mut routing = Routing { indices: vec![], weights: vec![], margin: vec![], scores: vec![] };
    for t in 0..T_PROMPT {
        let mut pairs: Vec<(u32, f32)> = idx[t].iter().zip(&w[t]).map(|(&e, &wt)| (e as u32, wt)).collect();
        pairs.sort_by_key(|p| p.0);
        routing.indices.push(pairs.iter().map(|p| p.0).collect());
        routing.weights.push(pairs.iter().map(|p| p.1).collect());
        routing.margin.push(0.0);
    }
    gate(0, "real: reference post_attention_layernorm and router0", x, routing)
}

// ---------------------------------------------------------------- gate 3: the whole model with resident experts

use llvq_llm::inkling::layers::max_abs_err;
use llvq_llm::inkling::Inkling;
use llvq_llm::lm::CausalLm;

/// Gate 3 tolerance, on the hidden state after every visit and on the final
/// logits, as `max |resident - reconstruction| <= GATE3_TOL * max |reconstruction|`.
/// Fixed before the first measurement. G measured one resident MoE call
/// against its dense reconstruction at 2.6e-4 to 3.1e-4 relative Frobenius
/// (two f16 narrowings a call). The prompt crosses 48 visits, 39 of them on
/// resident layers with the set verified at 09:40 (0-2, 6-17 three times). If
/// each call's error added to the residual stream with no cancellation and no
/// amplification, the drift would reach 39 x 3.1e-4 = 1.2e-2; the bound is
/// that rounded up to 2e-2. A structural error (a layer on another layer's
/// codes) moves the state by O(1). The dense model under both arms is the
/// same f32 code, so everything else agrees to about 1e-6 (gate 1).
const GATE3_TOL: f32 = 2e-2;

fn layers_dir() -> PathBuf {
    need(data_dir().join("layers"))
}

fn prompt_ids() -> anyhow::Result<Vec<u32>> {
    let dump = need(data_dir().join("lot2-d/ref_full.safetensors"));
    let refs = candle_core::safetensors::load(&dump, &Device::Cpu)?;
    Ok(refs["ids"].to_dtype(DType::I64)?.to_vec1::<i64>()?.iter().map(|&i| i as u32).collect())
}

fn write_json(name: &str, j: &serde_json::Value) -> anyhow::Result<()> {
    let out = data_dir().join("lot2-d");
    std::fs::create_dir_all(&out)?;
    // `INKLING_GATE3_TAG` keeps a mutant's journal apart from the clean one.
    let name = match std::env::var("INKLING_GATE3_TAG") {
        Ok(tag) => name.replace(".json", &format!("_{tag}.json")),
        Err(_) => name.to_string(),
    };
    std::fs::write(out.join(&name), serde_json::to_string_pretty(j)?)?;
    println!("{j}");
    Ok(())
}

/// Top-1 minus top-2 of one logits row.
fn top2_margin(row: &[f32]) -> f32 {
    let (mut a, mut b) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
    for &v in row {
        if v > a {
            b = a;
            a = v;
        } else if v > b {
            b = v;
        }
    }
    a - b
}

/// The resident arm: the model on Metal, every verified layer resident.
fn resident_model(metal: &Device) -> anyhow::Result<(Inkling, Vec<usize>, serde_json::Value)> {
    let d = data_dir();
    let a0 = allocated(metal);
    let t0 = Instant::now();
    let mut m = Inkling::load(&need(d.join("textonly/config.json")), &need(d.join("native/model-00001-of-00001.safetensors")), metal)?;
    metal.synchronize()?;
    let t_dense = t0.elapsed().as_secs_f64();
    let a1 = allocated(metal);
    let rt = resident::served_runtime(metal)?;
    let t1 = Instant::now();
    let stats = m.make_resident(&layers_dir(), rt)?;
    metal.synchronize()?;
    let t_res = t1.elapsed().as_secs_f64();
    let a2 = allocated(metal);
    let layers: Vec<usize> = stats.iter().map(|s| s.0).collect();
    let uploaded: u64 = stats.iter().map(|s| s.1.uploaded_bytes()).sum();
    println!(
        "resident layers {layers:?}; Metal allocated {a0} before, {a1} after the dense weights (+{}), {a2} after the \
         experts (+{}, {uploaded} B uploaded as codes, rotations and side data); load {t_dense:.1} s dense, {t_res:.1} s experts",
        a1 - a0,
        a2 - a1
    );
    let j = serde_json::json!({
        "resident_layers": layers, "allocated_before": a0, "allocated_after_dense": a1, "allocated_after_experts": a2,
        "expert_bytes_uploaded": uploaded, "seconds_load_dense": t_dense, "seconds_load_experts": t_res,
        "per_layer": stats.iter().map(|(l, s)| serde_json::json!({"layer": l, "uploaded_bytes": s.uploaded_bytes(),
            "seconds_upload": s.seconds_upload, "seconds_transcode": s.seconds_transcode, "seconds_read": s.seconds_read})).collect::<Vec<_>>(),
    });
    Ok((m, layers, j))
}

/// The reference arm: the model on the CPU, the same layers on their dense reconstructions.
fn reconstruction_model(layers: &[usize]) -> anyhow::Result<Inkling> {
    let d = data_dir();
    let mut m = Inkling::load(&need(d.join("textonly/config.json")), &need(d.join("native/model-00001-of-00001.safetensors")), &Device::Cpu)?;
    m.use_reconstructions(&layers_dir(), layers)?;
    Ok(m)
}

/// Gate 3a: on the 32-token prompt of gate 2, the resident arm (Metal) against
/// the reconstruction arm (CPU) on the same layer set: hidden state after
/// every visit and final logits within [`GATE3_TOL`]; routing flips and the
/// argmax agreement reported.
#[test]
#[ignore = "uses the GPU and reads the reconstructions of every verified layer; run under the guard's lock"]
fn gate3_resident_model_matches_reconstruction() -> anyhow::Result<()> {
    let ids = prompt_ids()?;
    let metal = Device::new_metal(0)?;
    let (res, layers, load) = resident_model(&metal)?;
    let t = Instant::now();
    let ours = res.forward(&ids, &mut res.new_cache(), true, true)?;
    let _ = ours.logits.sum_all()?.to_scalar::<f32>()?;
    let t_res = t.elapsed().as_secs_f64();
    let launches = res.take_launches();
    let rf = reconstruction_model(&layers)?;
    let t = Instant::now();
    let refo = rf.forward(&ids, &mut rf.new_cache(), true, true)?;
    let t_ref = t.elapsed().as_secs_f64();

    let mut fail = vec![];
    let mut rows = vec![];
    let mut flips = vec![];
    // Tokens whose routing is identical in both arms at every visit so far.
    let mut same: Vec<bool> = vec![true; ids.len()];
    // The first token any flip has touched so far: tokens before it cannot read a flip through attention.
    let mut first_flip = ids.len();
    for (v, (a, b)) in ours.visits.iter().zip(&refo.visits).enumerate() {
        let (abs, scale) = max_abs_err(&a.hidden, &b.hidden)?;
        let rel = abs / scale;
        let mut nflip = 0;
        for (tk, (ia, ib)) in a.routing.indices.iter().zip(&b.routing.indices).enumerate() {
            if ia != ib {
                nflip += 1;
                same[tk] = false;
                first_flip = first_flip.min(tk);
                // the change of this token's `score + bias` between the arms, over the 128 experts
                let dmax = a.routing.scores[tk]
                    .iter()
                    .zip(&b.routing.scores[tk])
                    .map(|(x, y)| (x - y).abs())
                    .fold(0f32, f32::max);
                let m = b.routing.margin[tk];
                flips.push(serde_json::json!({"visit": v, "layer": a.layer, "pass": a.pass, "token": tk,
                    "margin_reconstruction": m, "margin_resident": a.routing.margin[tk],
                    "max_score_change": dmax, "margin_over_change": m / dmax,
                    "experts_reconstruction": ib, "experts_resident": ia}));
            }
        }
        // the error on the never-rerouted tokens only
        let keep: Vec<u32> = (0..ids.len() as u32).filter(|&t| same[t as usize]).collect();
        let (sabs, sscale) = if keep.is_empty() {
            (f32::NAN, f32::NAN)
        } else {
            let k = Tensor::from_vec(keep.clone(), keep.len(), &Device::Cpu)?;
            let ha = a.hidden.to_device(&Device::Cpu)?.index_select(&k, 0)?;
            let hb = b.hidden.index_select(&k, 0)?;
            max_abs_err(&ha, &hb)?
        };
        let srel = sabs / sscale;
        // per token: the row's max abs error over the visit's max |ref|
        let diff = (a.hidden.to_device(&Device::Cpu)? - &b.hidden)?.abs()?.max(1)?.to_vec1::<f32>()?;
        let per_token: Vec<f32> = diff.iter().map(|x| x / scale).collect();
        let upstream = per_token[..first_flip].iter().cloned().fold(0f32, f32::max);
        let resident = layers.contains(&a.layer);
        println!("visit {v:02} L{:02} p{} {}: rel {rel:.3e}; {nflip} tokens route differently; on the {} tokens \
                  never rerouted: rel {srel:.3e}; on the {first_flip} tokens before the first flip: rel {upstream:.3e}",
            a.layer, a.pass, if resident { "resident" } else { "native  " }, keep.len());
        rows.push(serde_json::json!({"visit": v, "layer": a.layer, "pass": a.pass, "resident": resident,
            "max_abs_err": abs, "max_abs_ref": scale, "rel": rel, "tokens_route_differently": nflip,
            "tokens_never_rerouted": keep.len(), "rel_never_rerouted": srel,
            "tokens_before_first_flip": first_flip, "rel_before_first_flip": upstream, "rel_per_token": per_token}));
        if rel.is_nan() || rel > GATE3_TOL {
            fail.push(format!("visit {v}: rel {rel:.3e} > {GATE3_TOL:e}"));
        }
    }
    let (labs, lscale) = max_abs_err(&ours.logits, &refo.logits)?;
    let lrel = labs / lscale;
    let la = ours.logits.to_device(&Device::Cpu)?.to_vec2::<f32>()?;
    let lb = refo.logits.to_vec2::<f32>()?;
    let mut agree = 0;
    let mut differ = vec![];
    for (p, (ra, rb)) in la.iter().zip(&lb).enumerate() {
        let am = |r: &Vec<f32>| r.iter().enumerate().fold((0, f32::NEG_INFINITY), |m, (i, &v)| if v > m.1 { (i, v) } else { m }).0;
        if am(ra) == am(rb) {
            agree += 1;
        } else {
            differ.push(serde_json::json!({"position": p, "resident": am(ra), "reconstruction": am(rb),
                "reconstruction_top2_margin": top2_margin(rb)}));
        }
    }
    println!("logits: max abs {labs:.3e}, max |ref| {lscale:.4e}, rel {lrel:.3e}; argmax equal at {agree} of {}; \
              {} routing flips; resident prefill {t_res:.2} s ({launches} lattice launches), reconstruction prefill {t_ref:.1} s",
             la.len(), flips.len());
    if lrel.is_nan() || lrel > GATE3_TOL {
        fail.push(format!("logits: rel {lrel:.3e} > {GATE3_TOL:e}"));
    }
    write_json("gate3a.json", &serde_json::json!({
        "test": "gate3_resident_model_matches_reconstruction", "tokens": ids.len(), "tolerance": GATE3_TOL,
        "load": load, "visits": rows, "routing_flips": flips,
        "logits": {"max_abs_err": labs, "max_abs_ref": lscale, "rel": lrel, "argmax_equal": agree, "argmax_differ": differ},
        "seconds_prefill_resident": t_res, "lattice_launches_prefill": launches, "seconds_prefill_reconstruction_cpu": t_ref,
        "allocated_end": allocated(&metal), "pass": fail.is_empty(),
    }))?;
    assert!(fail.is_empty(), "gate 3a failed: {fail:?}");
    Ok(())
}

/// Tokens, top-2 margin of each step, prefill seconds, seconds a token, lattice launches a step.
type Greedy = (Vec<u32>, Vec<f32>, f64, f64, Vec<usize>);

fn greedy(m: &Inkling, prompt: &[u32], n: usize) -> anyhow::Result<Greedy> {
    let mut cache = CausalLm::new_cache(m);
    let t = Instant::now();
    let mut logits = CausalLm::prefill(m, prompt, &mut cache)?;
    let t_prefill = t.elapsed().as_secs_f64();
    let _ = m.take_launches();
    let mut got = vec![];
    let mut margins = vec![];
    let mut launches = vec![];
    let t = Instant::now();
    for i in 0..n {
        let row = logits.flatten_all()?.to_device(&Device::Cpu)?.to_vec1::<f32>()?;
        let next = row.iter().enumerate().fold((0, f32::NEG_INFINITY), |m, (i, &v)| if v > m.1 { (i, v) } else { m }).0 as u32;
        margins.push(top2_margin(&row));
        got.push(next);
        if i + 1 < n {
            logits = CausalLm::step(m, next, prompt.len() + i, &mut cache)?;
            launches.push(m.take_launches());
        }
    }
    Ok((got, margins, t_prefill, t.elapsed().as_secs_f64() / (n - 1) as f64, launches))
}

/// Gate 3b: 32 greedy tokens of the resident arm through `CausalLm`, against
/// the reconstruction arm's own greedy tokens and the native model's
/// (`D/evalfull/generate_native.json`); memory at the end and time a token.
#[test]
#[ignore = "uses the GPU and reads the reconstructions of every verified layer; run under the guard's lock"]
fn gate3_resident_model_generates() -> anyhow::Result<()> {
    let d = data_dir();
    let prompt = prompt_ids()?;
    let gen: serde_json::Value = serde_json::from_slice(&std::fs::read(need(d.join("evalfull/generate_native.json")))?)?;
    let native: Vec<u32> = gen["generated"].as_array().expect("generated").iter().map(|v| v.as_u64().unwrap() as u32).collect();
    let n = native.len();
    let metal = Device::new_metal(0)?;
    let (res, layers, load) = resident_model(&metal)?;
    let (got, m_res, tp, spt, launches) = greedy(&res, &prompt, n)?;
    metal.synchronize()?;
    let a_end = allocated(&metal);
    drop(res);
    let rf = reconstruction_model(&layers)?;
    let (want, m_ref, tp_ref, spt_ref, _) = greedy(&rf, &prompt, n)?;
    let tok = tokenizers::Tokenizer::from_file(need(d.join("native/tokenizer.json"))).map_err(anyhow::Error::msg)?;
    let text = tok.decode(&got, false).map_err(anyhow::Error::msg)?;
    let text_ref = tok.decode(&want, false).map_err(anyhow::Error::msg)?;
    let eq_ref = got.iter().zip(&want).filter(|(a, b)| a == b).count();
    let eq_nat = got.iter().zip(&native).filter(|(a, b)| a == b).count();
    let first_ref = got.iter().zip(&want).position(|(a, b)| a != b);
    let first_nat = got.iter().zip(&native).position(|(a, b)| a != b);
    let lmean = launches.iter().sum::<usize>() as f64 / launches.len().max(1) as f64;
    println!("resident: {got:?}\n  text {text:?}\nreconstruction: {want:?}\n  text {text_ref:?}");
    println!("equal to the reconstruction arm at {eq_ref} of {n} (first difference {first_ref:?}); to native at {eq_nat} \
              of {n} (first difference {first_nat:?}); resident prefill {tp:.2} s, {spt:.3} s/token, {lmean:.0} lattice \
              launches a token; reconstruction arm (CPU) prefill {tp_ref:.1} s, {spt_ref:.2} s/token; Metal allocated at the end {a_end}");
    if let Some(p) = first_ref {
        println!("  at the first difference: top-2 margin {:.4} resident, {:.4} reconstruction", m_res[p], m_ref[p]);
    }
    write_json("gate3b.json", &serde_json::json!({
        "test": "gate3_resident_model_generates", "load": load, "resident_tokens": got, "resident_text": text,
        "reconstruction_tokens": want, "reconstruction_text": text_ref, "native_tokens": native,
        "equal_to_reconstruction": eq_ref, "first_difference_reconstruction": first_ref,
        "equal_to_native": eq_nat, "first_difference_native": first_nat,
        "top2_margin_resident": m_res, "top2_margin_reconstruction": m_ref,
        "seconds_prefill": tp, "seconds_per_token": spt, "lattice_launches_per_token": launches,
        "seconds_prefill_reconstruction_cpu": tp_ref, "seconds_per_token_reconstruction_cpu": spt_ref,
        "allocated_end": a_end,
    }))?;
    Ok(())
}

/// Item 4 of step 3, a measurement and not a gate: the same resident model
/// with the runtime on the rows kernel (`prefill_rows = 8`,
/// `tv_tetra48_metal_lutdr`, each block decoded once for up to 8 prompt rows,
/// every row narrowed to f16 on output) against the served tuning (one matvec
/// a row), on the 32-token prompt: prefill seconds (median of 3), lattice
/// launches, and the logits difference between the two.
#[test]
#[ignore = "uses the GPU; run under the guard's lock"]
fn gate3_rows_kernel_prefill() -> anyhow::Result<()> {
    use llvq_llm::fused_metal::{MetalRuntime, MetalTuning};
    let d = data_dir();
    let prompt = prompt_ids()?;
    let window = window_ids()?;
    let metal = Device::new_metal(0)?;
    let cfg_p = need(d.join("textonly/config.json"));
    let ck_p = need(d.join("native/model-00001-of-00001.safetensors"));
    let tables = llvq_llm::fused::tetra48_tables(&llvq_search::tetra::Tetra::new());
    let mut out = serde_json::Map::new();
    let mut logits = vec![];
    for rows in [1usize, 8] {
        let mut rt = MetalRuntime::from_tables(&metal, &tables, resident::SERVED_TILE)?;
        if rows > 1 {
            rt.set_tuning(MetalTuning { prefill_rows: rows, ..MetalTuning::default() })?;
        }
        let mut m = Inkling::load(&cfg_p, &ck_p, &metal)?;
        let layers = m.make_resident(&layers_dir(), std::sync::Arc::new(rt))?;
        let mut secs = vec![];
        let mut launches = 0;
        let mut last = None;
        for _ in 0..3 {
            metal.synchronize()?;
            let t = Instant::now();
            let o = m.forward(&prompt, &mut m.new_cache(), false, true)?;
            let _ = o.logits.sum_all()?.to_scalar::<f32>()?;
            secs.push(t.elapsed().as_secs_f64());
            launches = m.take_launches();
            last = Some(o.logits.to_device(&Device::Cpu)?);
        }
        let med = median(secs.clone());
        // the window of gate 3c, one round, last position's logits only
        metal.synchronize()?;
        let t = Instant::now();
        let ow = m.forward(&window, &mut m.new_cache(), false, false)?;
        let _ = ow.logits.sum_all()?.to_scalar::<f32>()?;
        let t_win = t.elapsed().as_secs_f64();
        let l_win = m.take_launches();
        println!("prefill_rows {rows}: {} resident layers; 32 tokens {secs:?} s (median {med:.3}), {launches} lattice \
                  launches; {} tokens {t_win:.2} s, {l_win} lattice launches", layers.len(), window.len());
        out.insert(format!("rows_{rows}"), serde_json::json!({"seconds_32": secs, "median_32": med,
            "lattice_launches_32": launches, "window_tokens": window.len(), "seconds_window": t_win,
            "lattice_launches_window": l_win}));
        logits.push(last.expect("three rounds"));
        drop(m);
    }
    let (abs, scale) = max_abs_err(&logits[1], &logits[0])?;
    let am = |t: &Tensor| t.argmax(1).unwrap().to_vec1::<u32>().unwrap();
    let agree = am(&logits[1]).iter().zip(&am(&logits[0])).filter(|(a, b)| a == b).count();
    println!("rows 8 against rows 1: logits max abs {abs:.3e} of {scale:.3e}, argmax equal at {agree} of {}", prompt.len());
    out.insert("logits_rows8_vs_rows1".into(), serde_json::json!({"max_abs": abs, "max_abs_ref": scale, "argmax_equal": agree}));
    write_json("gate3_rows.json", &serde_json::Value::Object(out))?;
    Ok(())
}

/// The ids of gate 3c: the first `n` tokens of WikiText-2 test window 0
/// (`dump_ref.py window`), `n` from `INKLING_GATE3C_TOKENS` (default 2,048).
fn window_ids() -> anyhow::Result<Vec<u32>> {
    let n: usize = std::env::var("INKLING_GATE3C_TOKENS").ok().and_then(|v| v.parse().ok()).unwrap_or(2048);
    let p = need(data_dir().join("lot2-d/wikitext_w0_2048.json"));
    let j: serde_json::Value = serde_json::from_slice(&std::fs::read(p)?)?;
    let ids: Vec<u32> = j["ids"].as_array().expect("ids").iter().map(|v| v.as_u64().unwrap() as u32).collect();
    anyhow::ensure!(ids.len() >= n, "{} ids for {n} asked", ids.len());
    Ok(ids[..n].to_vec())
}

/// Row-wise log-softmax of `[T, V]` logits, in f64 sums, as f32 rows.
fn log_softmax_rows(logits: &Tensor) -> anyhow::Result<Vec<Vec<f32>>> {
    let rows = logits.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.to_vec2::<f32>()?;
    Ok(rows
        .into_iter()
        .map(|r| {
            let m = r.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let lse = m as f64 + r.iter().map(|&v| ((v - m) as f64).exp()).sum::<f64>().ln();
            r.into_iter().map(|v| (v as f64 - lse) as f32).collect()
        })
        .collect())
}

/// Gate 3c. Stated before its first run (coordinator, 2026-11-22): the
/// tolerance the repository serves a not-bit-for-bit form under
/// (`docs/ETAT.md` §4, "Attention tolerance"): paired WikiText-2 perplexity
/// within +-0.12 %. Here: one window, the first 2,048 tokens of test window 0;
/// the resident arm (Metal) against the reconstruction arm (CPU) on the same
/// layer set; the next-token NLL of every position but the last, paired.
/// PASS when `ppl_resident / ppl_reconstruction - 1` is within +-0.12 %.
/// Reported beside it: the 95 % interval of the paired mean NLL difference
/// over positions (`1.96 sd / sqrt(n)`; one window, so positions are the only
/// replicates, and they are correlated: the interval is indicative), the mean
/// and max KL(reconstruction || resident) per position, top-1 agreement.
#[test]
#[ignore = "uses the GPU and runs 2,048 tokens through both arms; run under the guard's lock"]
fn gate3c_resident_perplexity_matches_reconstruction() -> anyhow::Result<()> {
    let ids = window_ids()?;
    let n = ids.len();
    let metal = Device::new_metal(0)?;
    let (res, layers, load) = resident_model(&metal)?;
    let t = Instant::now();
    let lr = res.forward(&ids, &mut res.new_cache(), false, true)?.logits;
    let lp_res = log_softmax_rows(&lr)?;
    let t_res = t.elapsed().as_secs_f64();
    drop(lr);
    drop(res);
    let rf = reconstruction_model(&layers)?;
    let t = Instant::now();
    let lf = rf.forward(&ids, &mut rf.new_cache(), false, true)?.logits;
    let lp_ref = log_softmax_rows(&lf)?;
    let t_ref = t.elapsed().as_secs_f64();
    drop(lf);

    let argmax = |r: &[f32]| r.iter().enumerate().fold((0, f32::NEG_INFINITY), |m, (i, &v)| if v > m.1 { (i, v) } else { m }).0;
    let (mut nll_res, mut nll_ref, mut diffs, mut kls) = (0f64, 0f64, vec![], vec![]);
    let mut agree = 0;
    for p in 0..n - 1 {
        let y = ids[p + 1] as usize;
        let (a, b) = (-(lp_res[p][y] as f64), -(lp_ref[p][y] as f64));
        nll_res += a;
        nll_ref += b;
        diffs.push(a - b);
        let kl: f64 = lp_ref[p].iter().zip(&lp_res[p]).map(|(&r, &s)| (r as f64).exp() * (r as f64 - s as f64)).sum();
        kls.push(kl);
        if argmax(&lp_res[p]) == argmax(&lp_ref[p]) {
            agree += 1;
        }
    }
    let m = (n - 1) as f64;
    let (ppl_res, ppl_ref) = ((nll_res / m).exp(), (nll_ref / m).exp());
    let ratio_pct = (ppl_res / ppl_ref - 1.0) * 100.0;
    let md = diffs.iter().sum::<f64>() / m;
    let sd = (diffs.iter().map(|d| (d - md).powi(2)).sum::<f64>() / (m - 1.0)).sqrt();
    let half = 1.96 * sd / m.sqrt();
    let (lo, hi) = (((md - half).exp() - 1.0) * 100.0, ((md + half).exp() - 1.0) * 100.0);
    let kl_mean = kls.iter().sum::<f64>() / m;
    let kl_max = kls.iter().cloned().fold(0f64, f64::max);
    let pass = ratio_pct.abs() <= 0.12;
    println!("gate 3c: {n} tokens; ppl resident {ppl_res:.4}, reconstruction {ppl_ref:.4}; ratio {ratio_pct:+.4} % \
              [{lo:+.4}, {hi:+.4}] (band +-0.12 %): {}; KL(rec || res) mean {kl_mean:.3e}, max {kl_max:.3e}; top-1 \
              agreement {agree} of {}; resident {t_res:.1} s, reconstruction {t_ref:.1} s",
             if pass { "PASS" } else { "FAIL" }, n - 1);
    write_json("gate3c.json", &serde_json::json!({
        "test": "gate3c_resident_perplexity_matches_reconstruction", "tokens": n, "positions": n - 1,
        "resident_layers": layers, "load": load,
        "ppl_resident": ppl_res, "ppl_reconstruction": ppl_ref, "ratio_pct": ratio_pct,
        "ratio_pct_interval95_positions": [lo, hi], "band_pct": 0.12, "pass": pass,
        "kl_rec_res_mean": kl_mean, "kl_rec_res_max": kl_max, "top1_agreement": agree,
        "seconds_resident_metal": t_res, "seconds_reconstruction_cpu": t_ref,
    }))?;
    assert!(pass, "gate 3c: perplexity ratio {ratio_pct:+.4} % outside +-0.12 %");
    Ok(())
}

/// Where the 0.5 s of a decoded token goes: the resident model, the 32-token
/// prompt, then 8 timed steps with every MoE call timed (device drained
/// around each call, so the total runs a little slower than untimed). Split:
/// resident MoE calls, native MoE calls (decoded lazily on the CPU and
/// uploaded), and the rest (attention, convolutions, norms, routing, head).
#[test]
#[ignore = "uses the GPU; run under the guard's lock"]
fn gate3_decode_breakdown() -> anyhow::Result<()> {
    let prompt = prompt_ids()?;
    let metal = Device::new_metal(0)?;
    let (m, layers, _) = resident_model(&metal)?;
    let mut cache = CausalLm::new_cache(&m);
    let mut logits = CausalLm::prefill(&m, &prompt, &mut cache)?;
    let mut rows = vec![];
    for i in 0..8 {
        let next = logits.flatten_all()?.argmax(0)?.to_scalar::<u32>()?;
        let _ = m.take_launches();
        m.set_moe_timing(true);
        metal.synchronize()?;
        let t = Instant::now();
        logits = CausalLm::step(&m, next, prompt.len() + i, &mut cache)?;
        let _ = logits.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        let total = t.elapsed().as_secs_f64();
        let calls = m.take_moe_timing();
        m.set_moe_timing(false);
        let res_s: f64 = calls.iter().filter(|c| layers.contains(&c.0)).map(|c| c.1).sum();
        let nat_s: f64 = calls.iter().filter(|c| !layers.contains(&c.0)).map(|c| c.1).sum();
        let n_res = calls.iter().filter(|c| layers.contains(&c.0)).count();
        let launches = m.take_launches();
        println!("step {i}: total {total:.3} s; resident MoE {res_s:.3} s over {n_res} calls; native MoE {nat_s:.3} s \
                  over {} calls; rest {:.3} s; {launches} lattice launches", calls.len() - n_res, total - res_s - nat_s);
        rows.push(serde_json::json!({"step": i, "total": total, "resident_moe": res_s, "resident_calls": n_res,
            "native_moe": nat_s, "native_calls": calls.len() - n_res, "rest": total - res_s - nat_s,
            "lattice_launches": launches}));
    }
    let med = |k: &str| median(rows.iter().map(|r| r[k].as_f64().unwrap()).collect());
    println!("median: total {:.3}, resident MoE {:.3}, native MoE {:.3}, rest {:.3}", med("total"), med("resident_moe"),
             med("native_moe"), med("rest"));
    write_json("gate3_decode_breakdown.json", &serde_json::json!({"resident_layers": layers, "steps": rows,
        "median": {"total": med("total"), "resident_moe": med("resident_moe"), "native_moe": med("native_moe"),
                   "rest": med("rest")}}))?;
    Ok(())
}
