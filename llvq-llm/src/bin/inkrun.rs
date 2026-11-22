//! `inkrun`: Inkling 2 Mini generating in this engine, with its experts
//! resident in lattice codes, on CPU, Metal or CUDA. One JSON report.
//!
//! ```text
//! inkrun --config D/textonly/config.json --checkpoint D/textonly/model.safetensors \
//!        --layers D/layers --tokens prompt32.json --device cuda --out report.json \
//!        [--n 32] [--resident verified|none] [--require-layers 24]
//! ```
//!
//! The dense weights (attention, router, convolutions, norms, head) come from
//! `--checkpoint`, the native file or its text-only copy without experts
//! (`D/textonly/model.safetensors`, 646 tensors, the same names). The experts
//! of every layer of `--layers` whose `Lxx/VERIFIED` exists are made resident
//! on the device from their codes: through `inkling::resident` on Metal,
//! `inkling::resident_cuda` on CUDA. A layer without `VERIFIED` keeps its
//! native experts, decoded on the host at every call, which needs the native
//! checkpoint; `--require-layers` refuses the run before any token when fewer
//! layers would be resident. On the CPU nothing is resident.
//!
//! `--tokens` is a JSON object: `prompt` (token ids) and optionally
//! `reference` (token ids to compare the generation with). The generation is
//! greedy through `CausalLm`: one prefill, then one step a token.
//!
//! The report carries the tokens, the prefill and per-step seconds (each step
//! ends on reading its logits back to the host, which waits for the device),
//! the top-1 minus top-2 logit of every step (to read a divergence against a
//! near-tie), the lattice kernel launches a token (rotations and Tetra
//! matvecs only; candle's own launches are not counted), the device memory
//! (CUDA: the driver's total minus free, before the load, after the dense
//! load, after the experts, at the end; Metal: the device's allocated size),
//! the resident layers and their upload costs, and the comparison with the
//! reference. It exits non-zero only on an error; a token difference is a
//! reading, written in the report.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use candle_core::{Device, Tensor};
use serde_json::{json, Value};

use llvq_llm::inkling::Inkling;
use llvq_llm::lm::CausalLm;

struct Args {
    config: PathBuf,
    checkpoint: PathBuf,
    layers: PathBuf,
    tokens: PathBuf,
    device: String,
    out: PathBuf,
    n: usize,
    resident: bool,
    require_layers: usize,
}

fn usage() -> ! {
    eprintln!(
        "usage: inkrun --config <config.json> --checkpoint <model.safetensors> --layers <D/layers> \
         --tokens <tokens.json> --device cpu|metal|cuda --out <report.json> [--n 32] \
         [--resident verified|none] [--require-layers N]"
    );
    std::process::exit(2)
}

fn parse() -> Result<Args> {
    let mut a: std::collections::HashMap<String, String> = Default::default();
    let mut it = std::env::args().skip(1);
    while let Some(k) = it.next() {
        let Some(name) = k.strip_prefix("--") else { usage() };
        let v = it.next().unwrap_or_else(|| usage());
        if a.insert(name.to_string(), v).is_some() {
            bail!("--{name} given twice");
        }
    }
    let mut take = |k: &str| a.remove(k);
    let need = |v: Option<String>, k: &str| v.ok_or_else(|| anyhow::anyhow!("--{k} is required"));
    let args = Args {
        config: need(take("config"), "config")?.into(),
        checkpoint: need(take("checkpoint"), "checkpoint")?.into(),
        layers: need(take("layers"), "layers")?.into(),
        tokens: need(take("tokens"), "tokens")?.into(),
        device: need(take("device"), "device")?,
        out: need(take("out"), "out")?.into(),
        n: take("n").map(|v| v.parse()).transpose().context("--n")?.unwrap_or(32),
        resident: match take("resident").as_deref() {
            None | Some("verified") => true,
            Some("none") => false,
            Some(o) => bail!("--resident {o}: verified or none"),
        },
        require_layers: take("require-layers").map(|v| v.parse()).transpose().context("--require-layers")?.unwrap_or(0),
    };
    if let Some(k) = a.keys().next() {
        bail!("unknown option --{k}");
    }
    Ok(args)
}

fn device(name: &str) -> Result<Device> {
    Ok(match name {
        "cpu" => Device::Cpu,
        "metal" => Device::new_metal(0).context("Metal device (built with --features metal?)")?,
        "cuda" => Device::new_cuda(0).context("CUDA device (built with --features cuda?)")?,
        o => bail!("--device {o}: cpu, metal or cuda"),
    })
}

/// Device memory in use, in bytes, and what the number is.
// One arm when neither backend is compiled in (the Linux CI build without features).
#[allow(clippy::match_single_binding)]
fn memory(dev: &Device) -> Value {
    match dev {
        #[cfg(all(target_os = "linux", feature = "cuda"))]
        Device::Cuda(d) => {
            let r = d
                .cuda_stream()
                .context()
                .bind_to_thread()
                .map_err(|e| format!("bind: {e}"))
                .and_then(|_| {
                    candle_core::cuda_backend::cudarc::driver::result::mem_get_info()
                        .map_err(|e| format!("cuMemGetInfo: {e}"))
                });
            match r {
                Ok((free, total)) => json!({"kind": "cuda driver, total minus free", "used": total - free,
                                            "free": free, "total": total}),
                Err(e) => json!({"kind": "cuda driver", "error": e}),
            }
        }
        #[cfg(all(target_os = "macos", feature = "metal"))]
        Device::Metal(m) => json!({"kind": "metal currentAllocatedSize", "used": m.device().current_allocated_size()}),
        _ => json!({"kind": "none: the CPU shares host memory and reports no budget"}),
    }
}

/// Greedy argmax (first maximum) and the margin to the runner-up.
fn top2(logits: &Tensor) -> Result<(u32, f32)> {
    let v = logits.flatten_all()?.to_dtype(candle_core::DType::F32)?.to_vec1::<f32>()?;
    let (mut i1, mut b1, mut b2) = (0usize, f32::NEG_INFINITY, f32::NEG_INFINITY);
    for (i, &x) in v.iter().enumerate() {
        if x > b1 {
            b2 = b1;
            b1 = x;
            i1 = i;
        } else if x > b2 {
            b2 = x;
        }
    }
    if !b1.is_finite() {
        bail!("logits without a finite maximum");
    }
    Ok((i1 as u32, b1 - b2))
}

fn ids(v: &Value, key: &str) -> Result<Option<Vec<u32>>> {
    match v.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Array(a)) => a
            .iter()
            .map(|x| x.as_u64().and_then(|u| u32::try_from(u).ok()).ok_or_else(|| anyhow::anyhow!("{key}: not a token id: {x}")))
            .collect::<Result<Vec<_>>>()
            .map(Some),
        Some(o) => bail!("{key}: expected an array of token ids, got {o}"),
    }
}

#[allow(unused_variables, unused_mut, clippy::match_single_binding)]
fn make_resident(model: &mut Inkling, args: &Args, dev: &Device) -> Result<Vec<Value>> {
    if !args.resident {
        return Ok(vec![]);
    }
    let mut rows = vec![];
    match dev {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        Device::Metal(_) => {
            let rt = llvq_llm::inkling::resident::served_runtime(dev)?;
            for (l, s) in model.make_resident(&args.layers, rt)? {
                rows.push(json!({"layer": l, "experts": s.experts, "weights": s.weights, "file_bytes": s.file_bytes,
                    "uploaded_bytes": s.uploaded_bytes(), "word_bytes": s.word_bytes, "side_bytes": s.side_bytes,
                    "rotation_bytes": s.rotation_bytes, "seconds_read": s.seconds_read,
                    "seconds_transcode": s.seconds_transcode, "seconds_upload": s.seconds_upload}));
            }
        }
        #[cfg(all(target_os = "linux", feature = "cuda"))]
        Device::Cuda(_) => {
            let rt = llvq_llm::inkling::resident_cuda::served_runtime(dev, &model.cfg)?;
            for (l, s) in llvq_llm::inkling::resident_cuda::make_resident(model, &args.layers, rt)? {
                rows.push(json!({"layer": l, "experts": s.experts, "weights": s.weights, "file_bytes": s.file_bytes,
                    "uploaded_bytes": s.uploaded_bytes(), "matrix_bytes": s.matrix_bytes,
                    "rotation_bytes": s.rotation_bytes, "seconds_read": s.seconds_read,
                    "seconds_transcode": s.seconds_transcode, "seconds_upload": s.seconds_upload}));
            }
        }
        _ => eprintln!("inkrun: no resident path on this device; every layer keeps its native experts"),
    }
    Ok(rows)
}

fn main() -> Result<()> {
    let args = parse()?;
    let tok_json: Value = serde_json::from_slice(
        &std::fs::read(&args.tokens).with_context(|| format!("read {}", args.tokens.display()))?,
    )
    .with_context(|| format!("parse {}", args.tokens.display()))?;
    let prompt = ids(&tok_json, "prompt")?.ok_or_else(|| anyhow::anyhow!("{}: no prompt", args.tokens.display()))?;
    if prompt.is_empty() {
        bail!("{}: empty prompt", args.tokens.display());
    }
    let reference = ids(&tok_json, "reference")?;

    let dev = device(&args.device)?;
    let mem_before = memory(&dev);
    let t = Instant::now();
    let mut model = Inkling::load(&args.config, &args.checkpoint, &dev)?;
    dev.synchronize()?;
    let seconds_dense = t.elapsed().as_secs_f64();
    let mem_dense = memory(&dev);
    eprintln!("inkrun: dense weights on {} in {seconds_dense:.1} s", args.device);

    let t = Instant::now();
    let upload = make_resident(&mut model, &args, &dev)?;
    dev.synchronize()?;
    let seconds_resident = t.elapsed().as_secs_f64();
    let mem_resident = memory(&dev);
    let resident_layers = model.substituted_layers();
    let n_layers = model.cfg.num_hidden_layers;
    let native_layers: Vec<usize> = (0..n_layers).filter(|l| !resident_layers.contains(l)).collect();
    eprintln!("inkrun: {} layers resident in {seconds_resident:.1} s: {resident_layers:?}", resident_layers.len());
    if resident_layers.len() < args.require_layers {
        bail!(
            "{} layers resident, --require-layers {}: no VERIFIED file for layers {native_layers:?} under {}",
            resident_layers.len(),
            args.require_layers,
            args.layers.display()
        );
    }
    model.take_launches();

    let mut cache = model.new_cache();
    dev.synchronize()?;
    let t = Instant::now();
    let logits = CausalLm::prefill(&model, &prompt, &mut cache)?;
    let (mut next, mut margin) = top2(&logits)?;
    let seconds_prefill = t.elapsed().as_secs_f64();
    let launches_prefill = model.take_launches();
    eprintln!("inkrun: prefill of {} tokens in {seconds_prefill:.2} s", prompt.len());

    let mut tokens = vec![];
    let mut margins = vec![];
    let mut step_seconds = vec![];
    let mut step_launches = vec![];
    for i in 0..args.n {
        tokens.push(next);
        margins.push(margin);
        if i + 1 == args.n {
            break;
        }
        let t = Instant::now();
        let logits = CausalLm::step(&model, next, prompt.len() + i, &mut cache)?;
        (next, margin) = top2(&logits)?;
        step_seconds.push(t.elapsed().as_secs_f64());
        step_launches.push(model.take_launches());
    }
    let mem_end = memory(&dev);

    let (equal_prefix, first_mismatch, all_equal) = match &reference {
        None => (None, None, None),
        Some(r) => {
            let k = tokens.iter().zip(r).take_while(|(a, b)| a == b).count();
            let all = k == tokens.len() && r.len() >= tokens.len();
            (Some(k), if all { None } else { Some(k) }, Some(all))
        }
    };
    let mut sorted = step_seconds.clone();
    sorted.sort_by(f64::total_cmp);
    let median = sorted.get(sorted.len() / 2).copied();
    let report = json!({
        "bin": "inkrun",
        "device": args.device,
        "config": args.config, "checkpoint": args.checkpoint, "layers_dir": args.layers, "tokens_file": args.tokens,
        "prompt_tokens": prompt.len(),
        "new_tokens": tokens.len(),
        "resident_layers": resident_layers,
        "native_layers": native_layers,
        "upload": upload,
        "memory": {"before_load": mem_before, "after_dense": mem_dense, "after_resident": mem_resident, "end": mem_end},
        "seconds_dense_load": seconds_dense,
        "seconds_resident_load": seconds_resident,
        "seconds_prefill": seconds_prefill,
        "step_seconds": step_seconds,
        "step_seconds_median": median,
        "tokens_per_second_median_step": median.map(|m| 1.0 / m),
        "lattice_launches_prefill": launches_prefill,
        "lattice_launches_per_step": step_launches,
        "tokens": tokens,
        "margins": margins,
        "min_margin": margins.iter().copied().fold(f32::INFINITY, f32::min),
        "reference": reference,
        "equal_prefix": equal_prefix,
        "first_mismatch": first_mismatch,
        "all_equal": all_equal,
    });
    let text = serde_json::to_string_pretty(&report)?;
    std::fs::write(&args.out, &text).with_context(|| format!("write {}", args.out.display()))?;
    eprintln!(
        "inkrun: {} tokens, median step {:.3} s, equal to the reference: {:?} (first mismatch {:?}), report {}",
        tokens.len(),
        median.unwrap_or(f64::NAN),
        all_equal,
        first_mismatch,
        args.out.display()
    );
    Ok(())
}
