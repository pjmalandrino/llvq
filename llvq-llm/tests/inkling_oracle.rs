//! Oracle of the Inkling 2 Mini prefix (embedding and layers 0 to 5) against
//! f32 dumps of the reference code, `ops/inkling/dump_ref.py`.
//!
//! Inputs, under `INKLING_DATA` (default `~/llvq-inkling-mini`):
//! `textonly/config.json`, `native/model-00001-of-00001.safetensors` (mapped,
//! read one tensor at a time) and `lot1-c/ref_l0_5.safetensors`. A missing
//! file fails the test and names it. The test is `#[ignore]`d because it maps
//! 3.4 GB of the native experts: run it under the guard's lock with
//! `cargo test --release -p llvq-llm --test inkling_oracle -- --ignored --nocapture`.
//!
//! Tolerance, fixed before the first measurement. Both sides run in f32 on
//! bit-identical weights: BF16 widens exactly, and the expert decode is the
//! reference's to the bit (checked here on two experts). What differs is only
//! the order of floating-point sums (Accelerate against candle's gemm, the
//! convolution taps, `rsqrt` against `1 / sqrt`, a multiply by `1/c` against
//! a divide by `c`). One f32 dot product of length K drifts by about
//! `sqrt(K) * 2^-24` relative, under 1e-5 for K <= 3072; a layer chains about
//! thirty such operations and six layers are stacked. The bound is therefore
//! `max |ours - ref| <= 1e-4 * max |ref|` at every layer: ten times that
//! estimate, and four orders of magnitude under a structural error (dropping
//! the convolution's output scale multiplies its output by sqrt(5)).
//! The routed experts of layer 0 must be equal as sets, token by token, and
//! their weights within 1e-5 relative.

use std::collections::HashMap;
use std::path::PathBuf;

use candle_core::{DType, Device, Tensor};
use llvq_llm::inkling::layers::max_abs_err;
use llvq_llm::inkling::InklingPrefix;

const REL_TOL: f32 = 1e-4;
const ROUTE_W_TOL: f32 = 1e-5;

fn data_dir() -> PathBuf {
    match std::env::var("INKLING_DATA") {
        Ok(d) => PathBuf::from(d),
        Err(_) => PathBuf::from(std::env::var("HOME").expect("HOME")).join("llvq-inkling-mini"),
    }
}

fn need(p: PathBuf) -> PathBuf {
    assert!(p.exists(), "missing input {}: the oracle cannot run without it", p.display());
    p
}

#[test]
#[ignore = "maps the 19.5 GB native checkpoint (3.4 GB touched); run under the guard's lock"]
fn prefix_matches_reference_dumps() -> anyhow::Result<()> {
    let d = data_dir();
    let config = need(d.join("textonly/config.json"));
    let native = need(d.join("native/model-00001-of-00001.safetensors"));
    let dump = need(d.join("lot1-c/ref_l0_5.safetensors"));
    let dev = Device::Cpu;

    let refs: HashMap<String, Tensor> = candle_core::safetensors::load(&dump, &dev)?;
    let get = |k: &str| refs.get(k).unwrap_or_else(|| panic!("{} lacks {k}", dump.display()));
    let ids: Vec<u32> = get("ids").to_dtype(DType::I64)?.to_vec1::<i64>()?.iter().map(|&i| i as u32).collect();

    let n_layers = 6;
    let t0 = std::time::Instant::now();
    let model = InklingPrefix::load(&config, &native, n_layers, &dev)?;
    let t_load = t0.elapsed().as_secs_f64();

    // The expert decode, bit for bit against the library's float32 dequantization.
    let ex = model.checkpoint().experts(0)?;
    for e in [0usize, 77] {
        let (gu, down) = ex.dequant(e);
        for (name, ours) in [("w13", gu), ("w2", down)] {
            let r = get(&format!("L0.e{e}.{name}")).flatten_all()?.to_vec1::<f32>()?;
            assert_eq!(r.len(), ours.len(), "L0.e{e}.{name}: length");
            let diff = r.iter().zip(&ours).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
            assert_eq!(diff, 0, "L0.e{e}.{name}: {diff} values differ from the library's dequantization");
        }
    }
    drop(ex);

    let t1 = std::time::Instant::now();
    let out = model.forward(&ids)?;
    let t_fwd = t1.elapsed().as_secs_f64();

    let mut rows = vec![];
    let mut fail = vec![];
    let names = std::iter::once("embed".to_string()).chain((0..n_layers).map(|l| format!("layer{l}")));
    let ours = std::iter::once(&out.embed).chain(out.layers.iter());
    for (name, t) in names.zip(ours) {
        let (abs, scale) = max_abs_err(t, get(&name))?;
        let rel = abs / scale;
        println!("{name:>7}: max abs err {abs:.3e}, max |ref| {scale:.4e}, rel {rel:.3e}");
        rows.push(serde_json::json!({"tensor": name, "max_abs_err": abs, "max_abs_ref": scale, "rel": rel}));
        if rel.is_nan() || rel > REL_TOL {
            fail.push(format!("{name}: rel {rel:.3e} > {REL_TOL:e}"));
        }
    }

    // Routing: layer 0 must agree exactly; the other layers are reported.
    let mut route_rows = vec![];
    for (l, r) in out.routing.iter().enumerate() {
        let ri = get(&format!("router{l}.indices")).to_dtype(DType::I64)?.to_vec2::<i64>()?;
        let rw = get(&format!("router{l}.weights")).to_vec2::<f32>()?;
        let mut differ = 0;
        let mut wmax = 0f32;
        for (t, (ours_i, ref_i)) in r.indices.iter().zip(&ri).enumerate() {
            let ref_i: Vec<u32> = ref_i.iter().map(|&i| i as u32).collect();
            if *ours_i != ref_i {
                differ += 1;
                continue;
            }
            for (a, b) in r.weights[t].iter().zip(&rw[t]) {
                wmax = wmax.max((a - b).abs() / b.abs().max(1e-30));
            }
        }
        let margin = r.margin.iter().cloned().fold(f32::INFINITY, f32::min);
        println!("router{l}: {differ} of {} tokens route differently, weight rel err {wmax:.3e}, min margin {margin:.3e}", ri.len());
        route_rows.push(serde_json::json!({"layer": l, "tokens_differ": differ, "weight_rel_err": wmax, "min_margin": margin}));
        if l == 0 {
            if differ != 0 {
                fail.push(format!("layer 0: {differ} tokens route to other experts"));
            }
            if wmax.is_nan() || wmax > ROUTE_W_TOL {
                fail.push(format!("layer 0: routing weight rel err {wmax:.3e} > {ROUTE_W_TOL:e}"));
            }
        }
    }

    let report = serde_json::json!({
        "test": "inkling_oracle::prefix_matches_reference_dumps", "device": "cpu", "tokens": ids.len(),
        "rel_tol": REL_TOL, "layers": rows, "routing": route_rows, "seconds_load": t_load,
        "seconds_forward": t_fwd, "pass": fail.is_empty(),
    });
    println!("{report}");
    if let Ok(p) = std::env::var("INKLING_ORACLE_REPORT") {
        std::fs::write(&p, serde_json::to_string_pretty(&report)?)?;
    }
    assert!(fail.is_empty(), "oracle failed: {fail:?}");
    Ok(())
}

// ---------------------------------------------------------------- lot 2: the whole model

use llvq_llm::inkling::Inkling;
use llvq_llm::lm::CausalLm;

/// A routing flip whose 8th-minus-9th margin (ours, in `score + bias`) is
/// under this is reported as a near-tie. Scores are sigmoids in (0, 1); the
/// hidden states agree to about 1e-6 relative on CPU, so the router logits,
/// and the margins, carry noise of that order. Ten times it is the line.
const NEAR_TIE: f32 = 1e-5;

/// `INKLING_DEVICE`: `cpu` (default) or `metal`.
fn device() -> anyhow::Result<Device> {
    match std::env::var("INKLING_DEVICE").as_deref() {
        Ok("cpu") | Err(_) => Ok(Device::Cpu),
        #[cfg(feature = "metal")]
        Ok("metal") => Ok(Device::new_metal(0)?),
        Ok(other) => anyhow::bail!("INKLING_DEVICE={other}: cpu, or metal with --features metal"),
    }
}

fn dev_name(dev: &Device) -> &'static str {
    if dev.is_metal() {
        "metal"
    } else {
        "cpu"
    }
}

fn write_report(report: &serde_json::Value) -> anyhow::Result<()> {
    println!("{report}");
    if let Ok(p) = std::env::var("INKLING_ORACLE_REPORT") {
        std::fs::write(&p, serde_json::to_string_pretty(report)?)?;
    }
    Ok(())
}

/// Gate 1 of lot 2. The whole text model on the 32-token prompt of
/// `D/evalfull/generate_native.json`, against `D/lot2-d/ref_full.safetensors`
/// (`dump_ref.py full`): the MXFP8 decode of layer 20 bit for bit on two
/// experts; at each of the 48 visits the routed experts equal token by token
/// (a flip under [`NEAR_TIE`] is reported as a near-tie, it does not fail the
/// gate by itself) and the hidden state within `REL_TOL` of its max; the
/// logits within `REL_TOL` of their max, and their argmax equal at every
/// position. `REL_TOL` keeps the first slice's justification: f32 on both
/// sides, bit-identical weights, only the order of sums differs; 48 visits
/// instead of 6 multiply the chain by 8, and the bound still sits two orders
/// of magnitude over the 5e-7 the first slice measured after 6 layers.
#[test]
#[ignore = "maps the 19.5 GB native checkpoint (all 24 layers); run under the guard's lock"]
fn full_model_matches_reference_dumps() -> anyhow::Result<()> {
    let d = data_dir();
    let config = need(d.join("textonly/config.json"));
    let native = need(d.join("native/model-00001-of-00001.safetensors"));
    let dump = need(d.join("lot2-d/ref_full.safetensors"));
    let dev = device()?;

    let refs: HashMap<String, Tensor> = candle_core::safetensors::load(&dump, &Device::Cpu)?;
    let get = |k: &str| refs.get(k).unwrap_or_else(|| panic!("{} lacks {k}", dump.display()));
    let ids: Vec<u32> = get("ids").to_dtype(DType::I64)?.to_vec1::<i64>()?.iter().map(|&i| i as u32).collect();

    let t0 = std::time::Instant::now();
    let model = Inkling::load(&config, &native, &dev)?;
    let t_load = t0.elapsed().as_secs_f64();

    // The MXFP8 decode, bit for bit against the library's float32 dequantization.
    let ex = model.experts(20)?;
    assert!(ex.w13.is_mxfp8() && ex.w2.is_mxfp8(), "layer 20 should be MXFP8");
    for e in [0usize, 77] {
        let (gu, down) = ex.dequant(e);
        for (name, ours) in [("w13", gu), ("w2", down)] {
            let r = get(&format!("L20.e{e}.{name}")).flatten_all()?.to_vec1::<f32>()?;
            assert_eq!(r.len(), ours.len(), "L20.e{e}.{name}: length");
            let diff = r.iter().zip(&ours).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
            assert_eq!(diff, 0, "L20.e{e}.{name}: {diff} values differ from the library's dequantization");
        }
    }
    drop(ex);

    let t1 = std::time::Instant::now();
    let mut cache = model.new_cache();
    let out = model.forward(&ids, &mut cache, true, true)?;
    let t_fwd = t1.elapsed().as_secs_f64();
    assert_eq!(out.visits.len(), 48, "visits");

    let mut fail = vec![];
    let mut rows = vec![];
    let mut near_ties = vec![];
    let (abs, scale) = max_abs_err(&out.embed, get("embed"))?;
    println!("  embed: rel {:.3e}", abs / scale);
    for (v, vis) in out.visits.iter().enumerate() {
        let (abs, scale) = max_abs_err(&vis.hidden, get(&format!("visit{v}")))?;
        let rel = abs / scale;
        let ri = get(&format!("visit{v}.indices")).to_dtype(DType::I64)?.to_vec2::<i64>()?;
        let rw = get(&format!("visit{v}.weights")).to_vec2::<f32>()?;
        let mut flips = vec![];
        let mut wmax = 0f32;
        for (t, (ours_i, ref_i)) in vis.routing.indices.iter().zip(&ri).enumerate() {
            let ref_i: Vec<u32> = ref_i.iter().map(|&i| i as u32).collect();
            if *ours_i != ref_i {
                flips.push((t, vis.routing.margin[t]));
                continue;
            }
            for (a, b) in vis.routing.weights[t].iter().zip(&rw[t]) {
                wmax = wmax.max((a - b).abs() / b.abs().max(1e-30));
            }
        }
        let margin = vis.routing.margin.iter().cloned().fold(f32::INFINITY, f32::min);
        println!(
            "visit {v:02} L{:02} p{}: max abs err {abs:.3e}, max |ref| {scale:.4e}, rel {rel:.3e}; \
             {} tokens route differently, weight rel err {wmax:.3e}, min margin {margin:.3e}",
            vis.layer,
            vis.pass,
            flips.len()
        );
        if rel.is_nan() || rel > REL_TOL {
            fail.push(format!("visit {v}: rel {rel:.3e} > {REL_TOL:e}"));
        }
        for &(t, m) in &flips {
            if m < NEAR_TIE {
                near_ties.push(serde_json::json!({"visit": v, "token": t, "margin": m}));
                println!("  near-tie: visit {v} token {t} margin {m:.3e}");
            } else {
                fail.push(format!("visit {v} token {t}: routes elsewhere at margin {m:.3e} >= {NEAR_TIE:e}"));
            }
        }
        rows.push(serde_json::json!({
            "visit": v, "layer": vis.layer, "pass": vis.pass, "max_abs_err": abs, "max_abs_ref": scale,
            "rel": rel, "tokens_route_differently": flips.len(), "weight_rel_err": wmax, "min_margin": margin,
        }));
    }

    let ref_logits = get("logits");
    let (labs, lscale) = max_abs_err(&out.logits, ref_logits)?;
    let lrel = labs / lscale;
    let ours_arg = out.logits.argmax(1)?.to_vec1::<u32>()?;
    let ref_arg = ref_logits.argmax(1)?.to_vec1::<u32>()?;
    let arg_same = ours_arg.iter().zip(&ref_arg).filter(|(a, b)| a == b).count();
    println!("logits: max abs err {labs:.3e}, max |ref| {lscale:.4e}, rel {lrel:.3e}; argmax equal at {arg_same} of {}", ref_arg.len());
    if lrel.is_nan() || lrel > REL_TOL {
        fail.push(format!("logits: rel {lrel:.3e} > {REL_TOL:e}"));
    }
    if arg_same != ref_arg.len() {
        fail.push(format!("logits: argmax differs at {} positions", ref_arg.len() - arg_same));
    }

    write_report(&serde_json::json!({
        "test": "inkling_oracle::full_model_matches_reference_dumps", "device": dev_name(&dev), "tokens": ids.len(),
        "rel_tol": REL_TOL, "near_tie": NEAR_TIE, "visits": rows, "near_ties": near_ties,
        "logits": {"max_abs_err": labs, "max_abs_ref": lscale, "rel": lrel, "argmax_equal": arg_same},
        "seconds_load": t_load, "seconds_forward": t_fwd, "pass": fail.is_empty(),
    }))?;
    assert!(fail.is_empty(), "oracle failed: {fail:?}");
    Ok(())
}

/// Gate 2 of lot 2. Greedy decode through [`CausalLm`]: a prefill of the
/// 32-token prompt, then 31 steps on the per-visit caches, give the 32 tokens
/// of `D/evalfull/generate_native.json` (the library's cached greedy decode);
/// the logits of every step stay within `REL_TOL` of their max against one
/// forward of this model without cache over the prompt and the first 31
/// generated tokens (by causality, its rows 31..62 are the steps' logits).
#[test]
#[ignore = "maps the 19.5 GB native checkpoint (all 24 layers); run under the guard's lock"]
fn greedy_decode_matches_reference_generation() -> anyhow::Result<()> {
    let d = data_dir();
    let config = need(d.join("textonly/config.json"));
    let native = need(d.join("native/model-00001-of-00001.safetensors"));
    let dump = need(d.join("lot2-d/ref_full.safetensors"));
    let gen_file = need(d.join("evalfull/generate_native.json"));
    let dev = device()?;

    let refs: HashMap<String, Tensor> = candle_core::safetensors::load(&dump, &Device::Cpu)?;
    let prompt: Vec<u32> = refs
        .get("ids")
        .expect("ids")
        .to_dtype(DType::I64)?
        .to_vec1::<i64>()?
        .iter()
        .map(|&i| i as u32)
        .collect();
    let gen_json: serde_json::Value = serde_json::from_slice(&std::fs::read(&gen_file)?)?;
    let want: Vec<u32> = gen_json["generated"]
        .as_array()
        .expect("generated")
        .iter()
        .map(|v| v.as_u64().expect("token id") as u32)
        .collect();
    assert_eq!(gen_json["prompt_tokens"].as_u64(), Some(prompt.len() as u64), "prompt length");

    let model = Inkling::load(&config, &native, &dev)?;
    let n = want.len();
    let t0 = std::time::Instant::now();
    let mut cache = CausalLm::new_cache(&model);
    let mut logits = CausalLm::prefill(&model, &prompt, &mut cache)?;
    let t_prefill = t0.elapsed().as_secs_f64();
    let mut got = vec![];
    let mut step_logits = vec![];
    let t1 = std::time::Instant::now();
    for (i, &reference) in want.iter().enumerate() {
        let row = logits.flatten_all()?;
        let next = row.argmax(0)?.to_scalar::<u32>()?;
        got.push(next);
        step_logits.push(row.unsqueeze(0)?);
        println!("step {i:02}: token {next} (reference {reference})");
        if i + 1 < n {
            logits = CausalLm::step(&model, next, prompt.len() + i, &mut cache)?;
        }
    }
    let t_steps = t1.elapsed().as_secs_f64();
    let sec_per_token = t_steps / (n - 1) as f64;

    let mut seq = prompt.clone();
    seq.extend_from_slice(&got[..n - 1]);
    let t2 = std::time::Instant::now();
    let full = model.forward(&seq, &mut model.new_cache(), false, true)?.logits;
    let t_full = t2.elapsed().as_secs_f64();
    let full = full.narrow(0, prompt.len() - 1, n)?;
    let stepped = Tensor::cat(&step_logits, 0)?;
    let mut per_step = vec![];
    let mut worst = 0f32;
    for i in 0..n {
        let (a, s) = max_abs_err(&stepped.narrow(0, i, 1)?, &full.narrow(0, i, 1)?)?;
        per_step.push(a);
        worst = worst.max(a / s);
    }
    let (labs, lscale) = max_abs_err(&stepped, &full)?;
    let first_mismatch = got.iter().zip(&want).position(|(a, b)| a != b);
    println!(
        "tokens equal: {} (first mismatch {first_mismatch:?}); step vs full logits: max abs {labs:.3e}, max |full| \
         {lscale:.4e}, worst step rel {worst:.3e}; prefill {t_prefill:.1}s, {sec_per_token:.2} s/token, full {t_full:.1}s",
        first_mismatch.is_none()
    );
    let mut fail = vec![];
    if first_mismatch.is_some() {
        fail.push(format!("tokens differ from position {first_mismatch:?}"));
    }
    if worst.is_nan() || worst > REL_TOL {
        fail.push(format!("step logits: worst rel {worst:.3e} > {REL_TOL:e}"));
    }
    write_report(&serde_json::json!({
        "test": "inkling_oracle::greedy_decode_matches_reference_generation", "device": dev_name(&dev),
        "prompt_tokens": prompt.len(), "generated": got, "reference": want, "first_mismatch": first_mismatch,
        "step_vs_full": {"max_abs_err": labs, "max_abs_full": lscale, "worst_step_rel": worst,
                         "per_step_max_abs": per_step},
        "rel_tol": REL_TOL, "seconds_prefill": t_prefill, "seconds_per_token": sec_per_token,
        "seconds_full_forward": t_full, "pass": fail.is_empty(),
    }))?;
    assert!(fail.is_empty(), "gate 2 failed: {fail:?}");
    Ok(())
}
