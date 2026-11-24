//! `inkrun`: Inkling 2 Mini generating in this engine, with its experts
//! resident in lattice codes, on CPU, Metal or CUDA. One JSON report.
//!
//! ```text
//! inkrun --config D/textonly/config.json --checkpoint D/textonly/model.safetensors \
//!        --layers D/layers --tokens prompt32.json --device cuda --out report.json \
//!        [--n 32] [--resident verified|none] [--require-layers 24] \
//!        [--logits-out logits.f32] [--prompt-key ids] [--prompt-len 600]
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
//! `--prompt-key` names another array of the same file as the prompt (the
//! `ids` of a WikiText window, for a longer one), and `--prompt-len` keeps its
//! first tokens.
//!
//! `--phases K` splits the wall time of the first `K` steps by part
//! (`inkling::layers::phases`): the device is drained at every boundary, so
//! those steps run slower; the steps after them run untimed and give the
//! total the split is read against.
//!
//! `--attn-chunk N` reads a prompt's attention scores `N` query rows at a
//! time (default 512, `0` for one score tensor whatever the prompt).
//! `--prefill-chunk N` reads every prompt of the run `N` tokens a call of
//! `Inkling::forward` (`Inkling::forward_pieces`), each piece appended to the
//! cache before the next; a lone last token joins the piece before. Default
//! `0`: one call, the behaviour before. It bounds every transient of a prompt
//! call by `N` rows. Not meant to be bit for bit one call: compare the
//! `--logits-out` bytes of the two arms. Refused with `--routing-out`.
//! `--peak-probe` drains the device after every layer visit and keeps the
//! largest allocated size it reads there (Metal only), the peak the report
//! gives beside the four fixed points.
//!
//! `--rollback-gate on` runs, before the generation, the gate of the cache
//! mark: the prompt cut in halves `a` and `b`, `x` the reversed `b`;
//! prefill(a), mark, prefill(x), rollback, prefill(b) and two greedy steps,
//! against prefill(a), prefill(b) and the same steps, bit for bit.
//!
//! `--routing-out <file>` runs the prefill through `Inkling::forward` with
//! the visits recorded (the same operations; the record only keeps handles)
//! and writes, per visit and prompt token, the experts chosen and the margin
//! of the 8th over the 9th, as JSON.
//!
//! `--prefill-rows N` sets the prompt rows one launch of the resident
//! lattice kernel takes (default 8, the f32 rows kernel; 1 for one matvec a
//! row, the path before plan item P1). Metal only; each row is bit for bit
//! the one-row matvec's either way.
//!
//! `--head int4 --head-table <unembed.q4g64.llvqraw>` runs the head through
//! the int4 g64 table on Metal and never puts the BF16 output rows on the
//! device (plan item D4). `--head bf16` is the default.
//!
//! `--glue candle|fused` chooses the small steps of a one-row visit: candle's
//! operations (default) or the fused glue kernels (Metal only, class B). A
//! prompt of several rows runs candle's either way.
//!
//! `--attn candle|fused` chooses, at load, the attention products of a
//! one-row visit: candle's operations on f32 matrices (default), or the
//! kernels of `inkling::attn_dense` on the matrices held once in BF16 (Metal
//! only, class B); a prompt of several rows then widens one layer's matrices
//! for its call. It composes with both values of `--glue`. `fused+scores`
//! also runs the scores of the row over the cache in one launch
//! (`inkling::attn_scores`), reading the keys and values in place.
//!
//! `--routing device|host`: with the fused glue, a one-row visit hands its
//! routing to the experts on the device (default), or reads it back to the
//! host first as before. Same logits, bit for bit; `host` is the reference.
//! A run that dumps its routing reads it back in either case.
//!
//! `--moe-rows chunk|batch`: a prompt's rows through the resident experts one
//! expert and one chunk at a time (`chunk`, the default and the reference), or
//! one launch a step for every (token, expert) pair of a layer (`batch`).
//! CUDA only; meant to be the same bits.
//!
//! `--moe-act fused|candle`: the activation between the launches of a
//! resident layer's one-row experts in one launch (default), or through
//! candle's operations as before. Same bits; `candle` is the reference.
//!
//! `--kernel-times on`: on CUDA, every kernel timed on its own between two
//! drains of the stream, summed by name over the decode steps
//! (`kernel_times` in the report). The step time of such a run is not a
//! step's time.
//! `--events on|off`: on CUDA, `off` turns cudarc's event tracking off before
//! anything is uploaded, as the served door of the Qwen3 path does: two events
//! an allocation and one a use order nothing on one stream. Default `on`, the
//! protocol of every run before 2026-11-23. Refused on another device.
//! `--moe four|one`: on CUDA, the routed experts of a one-row visit in four
//! launches (`kernels/moe_tetra.cu`, default) or one expert at a time, as
//! before. Meant to be the same bits; `one` is the reference.
//!
//! `--routing-steps-out <file>` records the routing of every decode step
//! (each step runs through `Inkling::forward` with its visits recorded):
//! per step and visit, the experts and the margin of the 8th over the 9th.
//!
//! `--score-from N` reads the `--prompt-key` array as a text to score and
//! generates nothing: prefill of its first `N` tokens, then one-row steps fed
//! the true tokens to its end; every position from `N - 1` on scores the next
//! true token. `--score-prompt on` scores every position of one prefill of
//! the whole array instead (all logits of the prompt path). Both write the
//! negative log-likelihood of each scored position (f64, from the f32
//! logits read back, log-sum-exp on the host) to `--nll-out` as JSON.
//!
//! `--logits-out` writes the f32 logits of the prefill's last position and
//! of every step, as read back by the greedy choice: raw little-endian f32,
//! one row of `vocab` values a position, prefill first, no header. The
//! report gives the row count and the vocabulary. The class A gates compare
//! two such files byte for byte.
//!
//! Every cache of a run reserves, before its prompt, the positions the run
//! appends to it: the prompt plus `--n`. The key and value buffers are then
//! allocated once and never grow by half. The values do not change.
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

use llvq_llm::inkling::layers::phases;
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
    logits_out: Option<PathBuf>,
    prompt_key: String,
    prompt_len: Option<usize>,
    phases: usize,
    attn_chunk: Option<usize>,
    prefill_chunk: usize,
    peak_probe: bool,
    rollback_gate: bool,
    routing_out: Option<PathBuf>,
    prefill_rows: usize,
    head_table: Option<PathBuf>,
    glue: llvq_llm::inkling::GlueKind,
    attn: llvq_llm::inkling::AttnKind,
    routing_on_device: bool,
    moe_act: bool,
    moe_rows_batch: bool,
    prefill_phases: bool,
    attn_bias_device: bool,
    moe_four: bool,
    routing_steps_out: Option<PathBuf>,
    events_off: bool,
    kernel_times: bool,
    score_from: Option<usize>,
    score_prompt: bool,
    nll_out: Option<PathBuf>,
}

fn usage() -> ! {
    eprintln!(
        "usage: inkrun --config <config.json> --checkpoint <model.safetensors> --layers <D/layers> \
         --tokens <tokens.json> --device cpu|metal|cuda --out <report.json> [--n 32] \
         [--resident verified|none] [--require-layers N] [--logits-out <file>] [--prompt-key <key>] \
         [--prompt-len N] [--phases K] [--attn-chunk N] [--prefill-chunk N] [--peak-probe on] \
         [--rollback-gate on] [--prefill-rows 1|2|4|8] [--head bf16|int4 --head-table <file>] \
         [--glue candle|fused] [--attn candle|fused|fused+scores] [--routing device|host] [--moe-act fused|candle] [--moe four|one] [--moe-rows chunk|batch] [--attn-bias host|device] [--prefill-phases on] [--events on|off] [--kernel-times on|off] [--routing-steps-out <file>] [--score-from N | --score-prompt on] \
         [--nll-out <file>]"
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
        logits_out: take("logits-out").map(PathBuf::from),
        prompt_key: take("prompt-key").unwrap_or_else(|| "prompt".into()),
        prompt_len: take("prompt-len").map(|v| v.parse()).transpose().context("--prompt-len")?,
        phases: take("phases").map(|v| v.parse()).transpose().context("--phases")?.unwrap_or(0),
        attn_chunk: take("attn-chunk").map(|v| v.parse()).transpose().context("--attn-chunk")?,
        prefill_chunk: take("prefill-chunk").map(|v| v.parse()).transpose().context("--prefill-chunk")?.unwrap_or(0),
        peak_probe: match take("peak-probe").as_deref() {
            None | Some("off") => false,
            Some("on") => true,
            Some(o) => bail!("--peak-probe {o}: on or off"),
        },
        routing_out: take("routing-out").map(PathBuf::from),
        rollback_gate: match take("rollback-gate").as_deref() {
            None | Some("off") => false,
            Some("on") => true,
            Some(o) => bail!("--rollback-gate {o}: on or off"),
        },
        prefill_rows: take("prefill-rows")
            .map(|v| v.parse())
            .transpose()
            .context("--prefill-rows")?
            .unwrap_or(llvq_llm::inkling::DEFAULT_PREFILL_ROWS),
        head_table: {
            let table = take("head-table").map(PathBuf::from);
            match (take("head").as_deref(), table) {
                (None | Some("bf16"), None) => None,
                (Some("int4"), Some(t)) => Some(t),
                (Some("int4"), None) => bail!("--head int4 needs --head-table <unembed.q4g64.llvqraw>"),
                (None | Some("bf16"), Some(_)) => bail!("--head-table without --head int4"),
                (Some(o), _) => bail!("--head {o}: bf16 or int4"),
            }
        },
        glue: take("glue").as_deref().unwrap_or("candle").parse()?,
        attn: take("attn").as_deref().unwrap_or("candle").parse()?,
        routing_on_device: match take("routing").as_deref() {
            None | Some("device") => true,
            Some("host") => false,
            Some(o) => bail!("--routing {o}: device or host"),
        },
        moe_act: match take("moe-act").as_deref() {
            None | Some("fused") => true,
            Some("candle") => false,
            Some(o) => bail!("--moe-act {o}: fused or candle"),
        },
        attn_bias_device: match take("attn-bias").as_deref() {
            None | Some("host") => false,
            Some("device") => true,
            Some(o) => bail!("--attn-bias {o}: host or device"),
        },
        prefill_phases: match take("prefill-phases").as_deref() {
            None | Some("off") => false,
            Some("on") => true,
            Some(o) => bail!("--prefill-phases {o}: on or off"),
        },
        moe_rows_batch: match take("moe-rows").as_deref() {
            None | Some("chunk") => false,
            Some("batch") => true,
            Some(o) => bail!("--moe-rows {o}: chunk or batch"),
        },
        moe_four: match take("moe").as_deref() {
            None | Some("four") => true,
            Some("one") => false,
            Some(o) => bail!("--moe {o}: four or one"),
        },
        routing_steps_out: take("routing-steps-out").map(PathBuf::from),
        kernel_times: match take("kernel-times").as_deref() {
            None | Some("off") => false,
            Some("on") => true,
            Some(o) => bail!("--kernel-times {o}: on or off"),
        },
        events_off: match take("events").as_deref() {
            None | Some("on") => false,
            Some("off") => true,
            Some(o) => bail!("--events {o}: on or off"),
        },
        score_from: take("score-from").map(|v| v.parse()).transpose().context("--score-from")?,
        score_prompt: match take("score-prompt").as_deref() {
            None | Some("off") => false,
            Some("on") => true,
            Some(o) => bail!("--score-prompt {o}: on or off"),
        },
        nll_out: take("nll-out").map(PathBuf::from),
    };
    if args.score_from.is_some() && args.score_prompt {
        bail!("--score-from and --score-prompt are two modes; name one");
    }
    if (args.score_from.is_some() || args.score_prompt) != args.nll_out.is_some() {
        bail!("--nll-out goes with --score-from or --score-prompt");
    }
    if args.prefill_chunk == 1 {
        bail!("--prefill-chunk 1: a one-row call is a decode step; give 0 (one call) or at least 2");
    }
    if args.prefill_chunk > 0 && args.routing_out.is_some() {
        bail!("--prefill-chunk with --routing-out: the routing dump reads the prompt in one recorded call");
    }
    if args.score_from == Some(0) {
        bail!("--score-from 0: the prefill needs at least one token");
    }
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

/// Bytes the device has allocated, where the backend says: Metal's own count, and on CUDA the driver's total
/// minus free, which counts what the stream's pool still holds as well as the live buffers.
#[allow(unused_variables)]
fn allocated(dev: &Device) -> Option<u64> {
    match dev {
        #[cfg(all(target_os = "linux", feature = "cuda"))]
        Device::Cuda(d) => {
            d.cuda_stream().context().bind_to_thread().ok()?;
            let (free, total) = candle_core::cuda_backend::cudarc::driver::result::mem_get_info().ok()?;
            Some((total - free) as u64)
        }
        #[cfg(all(target_os = "macos", feature = "metal"))]
        Device::Metal(m) => Some(m.device().current_allocated_size() as u64),
        _ => None,
    }
}

/// Greedy argmax (first maximum) and the margin to the runner-up. When
/// `sink` is given, the row read back is appended to it as little-endian f32.
fn top2(logits: &Tensor, sink: Option<&mut Vec<u8>>) -> Result<(u32, f32)> {
    let v = logits.flatten_all()?.to_dtype(candle_core::DType::F32)?.to_vec1::<f32>()?;
    if let Some(buf) = sink {
        buf.extend(v.iter().flat_map(|x| x.to_le_bytes()));
    }
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

/// The bits of a logits row.
fn bits(logits: &Tensor) -> Result<Vec<u32>> {
    Ok(logits.flatten_all()?.to_dtype(candle_core::DType::F32)?.to_vec1::<f32>()?.iter().map(|x| x.to_bits()).collect())
}

/// prefill(a), [mark, prefill(x), rollback,] prefill(b), then two greedy
/// steps: the bits of the three logits rows.
fn rollback_arm(model: &Inkling, a: &[u32], x: Option<&[u32]>, b: &[u32]) -> Result<Vec<Vec<u32>>> {
    let mut cache = model.new_cache();
    cache.reserve(a.len() + x.map_or(0, <[u32]>::len) + b.len() + 2)?;
    CausalLm::prefill(model, a, &mut cache)?;
    if let Some(x) = x {
        let m = CausalLm::mark(model, &cache);
        CausalLm::prefill(model, x, &mut cache)?;
        CausalLm::rollback(model, &mut cache, &m)?;
    }
    let mut logits = CausalLm::prefill(model, b, &mut cache)?;
    let mut rows = vec![bits(&logits)?];
    for i in 0..2 {
        let (next, _) = top2(&logits, None)?;
        logits = CausalLm::step(model, next, a.len() + b.len() + i, &mut cache)?;
        rows.push(bits(&logits)?);
    }
    Ok(rows)
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
            let rt = llvq_llm::inkling::resident_runtime(dev, args.prefill_rows)?;
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
    let mut prompt = ids(&tok_json, &args.prompt_key)?
        .ok_or_else(|| anyhow::anyhow!("{}: no array {}", args.tokens.display(), args.prompt_key))?;
    if let Some(n) = args.prompt_len {
        if n > prompt.len() {
            bail!("--prompt-len {n}: {} holds {} tokens under {}", args.tokens.display(), prompt.len(), args.prompt_key);
        }
        prompt.truncate(n);
    }
    if prompt.is_empty() {
        bail!("{}: empty prompt", args.tokens.display());
    }
    let reference = ids(&tok_json, "reference")?;

    let dev = device(&args.device)?;
    if args.events_off {
        llvq_llm::inkling::cuda_events_off(&dev)?;
        eprintln!("inkrun: cudarc event tracking off (one stream, ordered by issue), before any upload");
    }
    let mem_before = memory(&dev);
    let t = Instant::now();
    let opts = llvq_llm::inkling::LoadOptions { int4_head: args.head_table.clone() };
    let mut model = Inkling::load_with_attn(&args.config, &args.checkpoint, &dev, &opts, args.attn)?;
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
    if let Some(n) = args.attn_chunk {
        model.set_attn_chunk(n);
    }
    model.set_prefill_piece(args.prefill_chunk);
    model.set_glue(args.glue)?;
    model.set_routing_on_device(args.routing_on_device);
    model.set_moe_act(args.moe_act);
    model.set_moe_four_launches(args.moe_four);
    model.set_moe_batch_rows(args.moe_rows_batch);
    #[cfg(all(target_os = "linux", feature = "cuda"))]
    llvq_llm::inkling::attn_bias::set_on_device(args.attn_bias_device.then_some(&dev))?;
    #[cfg(not(all(target_os = "linux", feature = "cuda")))]
    if args.attn_bias_device {
        bail!("--attn-bias device: CUDA only");
    }
    let provenance = json!({"prefill_rows": args.prefill_rows, "moe_rows": if args.moe_rows_batch { "batch" } else { "chunk" }, "attn_bias": if args.attn_bias_device { "device" } else { "host" }, "glue": model.glue().name(),
        "head": if args.head_table.is_some() { "int4" } else { "bf16" }, "head_table": args.head_table,
        "bf16_head_on_device": model.has_bf16_head(), "attn": model.attn().name(),
        "routing_on_device": model.routing_on_device(), "moe": if args.moe_four { "four" } else { "one" }, "events": if args.events_off { "off" } else { "on" }, "moe_act": if args.moe_act { "fused" } else { "candle" },
        "attention_matrix_bytes": {"f32": model.attention_matrix_bytes().0, "bf16": model.attention_matrix_bytes().1}});
    if args.score_from.is_some() || args.score_prompt {
        return score(&model, &args, &dev, &prompt, provenance,
            json!({"before_load": mem_before, "after_dense": mem_dense, "after_resident": mem_resident}));
    }
    let peak = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
    if args.peak_probe {
        let (d, p) = (dev.clone(), peak.clone());
        model.set_visit_probe(Some(Box::new(move || {
            // Read before the drain too: on CUDA the pool may hand memory back at a drain.
            for drained in [false, true] {
                if drained {
                    let _ = d.synchronize();
                }
                if let Some(b) = allocated(&d) {
                    p.fetch_max(b, std::sync::atomic::Ordering::Relaxed);
                }
            }
        })));
    }

    let rollback_gate = if args.rollback_gate {
        if prompt.len() < 4 {
            bail!("--rollback-gate needs a prompt of at least 4 tokens");
        }
        let (a, b) = prompt.split_at(prompt.len() / 2);
        let x: Vec<u32> = b.iter().rev().copied().collect();
        let t = Instant::now();
        let with = rollback_arm(&model, a, Some(&x), b)?;
        let without = rollback_arm(&model, a, None, b)?;
        let differing: Vec<usize> =
            (0..with.len()).filter(|&r| with[r] != without[r]).collect();
        eprintln!("inkrun: rollback gate, rows differing {differing:?}");
        model.take_launches();
        json!({"a": a.len(), "x": x.len(), "b": b.len(), "rows": with.len(), "differing_rows": differing,
               "bit_for_bit": differing.is_empty(), "seconds": t.elapsed().as_secs_f64()})
    } else {
        Value::Null
    };

    let mut cache = model.new_cache();
    // Every position the run appends, before the prompt is read: the key and value buffers are allocated once at
    // this size and never grow by half. The values do not change.
    cache.reserve(prompt.len() + args.n)?;
    dev.synchronize()?;
    let t = Instant::now();
    let mut logit_rows: Option<Vec<u8>> = args.logits_out.as_ref().map(|_| Vec::new());
    if args.prefill_phases {
        phases::start(&dev)?;
    }
    let logits = match &args.routing_out {
        None => CausalLm::prefill(&model, &prompt, &mut cache)?,
        Some(p) => {
            let out = model.forward(&prompt, &mut cache, true, false)?;
            let visits: Vec<Value> = out
                .visits
                .iter()
                .map(|v| json!({"layer": v.layer, "pass": v.pass, "experts": v.routing.indices, "margin": v.routing.margin}))
                .collect();
            std::fs::write(p, serde_json::to_string(&visits)?).with_context(|| format!("write {}", p.display()))?;
            out.logits.reshape((1, 1, CausalLm::vocab_size(&model)))?
        }
    };
    let (mut next, mut margin) = top2(&logits, logit_rows.as_mut())?;
    // The prompt's read by part, the device drained at each mark (`--prefill-phases on`).
    let prefill_parts: Option<serde_json::Map<String, Value>> = if args.prefill_phases {
        phases::mark(&dev, phases::Phase::Readback)?;
        let acc = phases::stop();
        Some(phases::ALL.iter().zip(acc).map(|(p, s)| (p.name().to_string(), json!(s))).collect())
    } else {
        None
    };
    let seconds_prefill = t.elapsed().as_secs_f64();
    let launches_prefill = model.take_launches();
    let peak_prefill = peak.load(std::sync::atomic::Ordering::Relaxed);
    let mem_prefill = memory(&dev);
    eprintln!("inkrun: prefill of {} tokens in {seconds_prefill:.2} s", prompt.len());

    let mut tokens = vec![];
    let mut margins = vec![];
    let mut step_seconds = vec![];
    let mut step_launches = vec![];
    let mut step_kernel_launches: Vec<u64> = vec![];
    llvq_llm::inkling::take_kernel_launches();
    llvq_llm::inkling::set_kernel_profile(args.kernel_times);
    let mut phase_rows: Vec<Value> = vec![];
    let mut step_routing: Vec<Value> = vec![];
    for i in 0..args.n {
        tokens.push(next);
        margins.push(margin);
        if i + 1 == args.n {
            break;
        }
        let profiled = i < args.phases;
        if profiled {
            phases::start(&dev)?;
        }
        let t = Instant::now();
        let logits = match &args.routing_steps_out {
            None => CausalLm::step(&model, next, prompt.len() + i, &mut cache)?,
            Some(_) => {
                let out = model.forward(&[next], &mut cache, true, false)?;
                step_routing.push(Value::Array(
                    out.visits
                        .iter()
                        .map(|v| json!({"layer": v.layer, "pass": v.pass, "experts": v.routing.indices[0],
                                        "margin": v.routing.margin[0]}))
                        .collect(),
                ));
                out.logits.reshape((1, 1, CausalLm::vocab_size(&model)))?
            }
        };
        (next, margin) = top2(&logits, logit_rows.as_mut())?;
        if profiled {
            phases::mark(&dev, phases::Phase::Readback)?;
            let acc = phases::stop();
            let total = t.elapsed().as_secs_f64();
            let parts: serde_json::Map<String, Value> =
                phases::ALL.iter().zip(acc).map(|(p, s)| (p.name().to_string(), json!(s))).collect();
            phase_rows.push(json!({"step": i, "total": total, "sum_of_parts": acc.iter().sum::<f64>(), "parts": parts}));
        }
        step_seconds.push(t.elapsed().as_secs_f64());
        step_launches.push(model.take_launches());
        step_kernel_launches.push(llvq_llm::inkling::take_kernel_launches());
    }
    llvq_llm::inkling::set_kernel_profile(false);
    let kernel_times: Vec<Value> = llvq_llm::inkling::take_kernel_times()
        .into_iter()
        .map(|(k, n, s)| json!({"kernel": k, "launches": n, "seconds": s}))
        .collect();
    let mem_end = memory(&dev);
    if let Some(p) = &args.routing_steps_out {
        std::fs::write(p, serde_json::to_string(&step_routing)?).with_context(|| format!("write {}", p.display()))?;
    }
    let vocab = CausalLm::vocab_size(&model);
    let logits_file = match (&args.logits_out, &logit_rows) {
        (Some(p), Some(buf)) => {
            std::fs::write(p, buf).with_context(|| format!("write {}", p.display()))?;
            json!({"path": p, "rows": buf.len() / (4 * vocab), "vocab": vocab, "bytes": buf.len()})
        }
        _ => Value::Null,
    };

    let (equal_prefix, first_mismatch, all_equal) = match &reference {
        None => (None, None, None),
        Some(r) => {
            let k = tokens.iter().zip(r).take_while(|(a, b)| a == b).count();
            let all = k == tokens.len() && r.len() >= tokens.len();
            (Some(k), if all { None } else { Some(k) }, Some(all))
        }
    };
    let med = |mut v: Vec<f64>| -> Option<f64> {
        v.sort_by(f64::total_cmp);
        v.get(v.len() / 2).copied()
    };
    let phases_report = if phase_rows.is_empty() {
        Value::Null
    } else {
        let col = |k: &str| phase_rows.iter().map(|r| r[k].as_f64().unwrap_or(f64::NAN)).collect::<Vec<_>>();
        let parts: serde_json::Map<String, Value> = phases::ALL
            .iter()
            .map(|p| {
                let v = phase_rows.iter().map(|r| r["parts"][p.name()].as_f64().unwrap_or(f64::NAN)).collect();
                (p.name().to_string(), json!(med(v)))
            })
            .collect();
        let untimed: Vec<f64> = step_seconds.iter().skip(args.phases).copied().collect();
        json!({"profiled_steps": phase_rows.len(), "rows": phase_rows,
               "median_parts": parts, "median_total_profiled": med(col("total")),
               "median_sum_of_parts": med(col("sum_of_parts")),
               "untimed_steps": untimed.len(), "median_total_untimed": med(untimed)})
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
        "memory": {"before_load": mem_before, "after_dense": mem_dense, "after_resident": mem_resident, "after_prefill": mem_prefill, "end": mem_end},
        "seconds_dense_load": seconds_dense,
        "seconds_resident_load": seconds_resident,
        "seconds_prefill": seconds_prefill,
        "prefill_parts": prefill_parts,
        "step_seconds": step_seconds,
        "step_seconds_median": median,
        "tokens_per_second_median_step": median.map(|m| 1.0 / m),
        "lattice_launches_prefill": launches_prefill,
        "lattice_launches_per_step": step_launches,
        "kernel_launches_per_step": step_kernel_launches,
        "kernel_times": kernel_times,
        "tokens": tokens,
        "margins": margins,
        "min_margin": margins.iter().copied().fold(f32::INFINITY, f32::min),
        "reference": reference,
        "equal_prefix": equal_prefix,
        "first_mismatch": first_mismatch,
        "all_equal": all_equal,
        "prompt_key": args.prompt_key,
        "phases": phases_report,
        "attn_chunk": model.attn_chunk(),
        "prefill_chunk": model.prefill_piece(),
        "rollback_gate": rollback_gate,
        "peak_probe": if args.peak_probe {
            json!({"kind": "largest allocated size read after a layer visit, before and after a drain",
                   "prefill": peak_prefill, "whole_run": peak.load(std::sync::atomic::Ordering::Relaxed)})
        } else {
            Value::Null
        },
        "logits_file": logits_file,
        "provenance": provenance,
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

/// `-log softmax(row)[target]` in f64, from an f32 row.
fn nll(row: &[f32], target: u32) -> Result<f64> {
    let t = target as usize;
    if t >= row.len() {
        bail!("token {target} outside a vocabulary of {}", row.len());
    }
    let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let z: f64 = row.iter().map(|&x| (x as f64 - m).exp()).sum();
    Ok(m + z.ln() - row[t] as f64)
}

/// The scoring modes: per-position negative log-likelihood of the true next
/// token, by one-row steps (`--score-from`) or by one prefill (`--score-prompt`).
fn score(model: &Inkling, args: &Args, dev: &Device, text: &[u32], provenance: Value, memory_load: Value) -> Result<()> {
    if text.len() < 2 {
        bail!("scoring needs at least 2 tokens");
    }
    let vocab = CausalLm::vocab_size(model);
    let mut cache = model.new_cache();
    cache.reserve(text.len())?;
    let mut nlls: Vec<f64> = vec![];
    let mut positions: Vec<usize> = vec![];
    dev.synchronize()?;
    let t = Instant::now();
    let (mode, seconds_prefill, steps) = if args.score_prompt {
        let logits = model.forward_pieces(text, &mut cache, true)?;
        let rows = logits.to_dtype(candle_core::DType::F32)?.to_vec2::<f32>()?;
        if rows.len() != text.len() || rows[0].len() != vocab {
            bail!("{} logits rows of {} for {} tokens", rows.len(), rows[0].len(), text.len());
        }
        for (p, row) in rows.iter().enumerate().take(text.len() - 1) {
            nlls.push(nll(row, text[p + 1])?);
            positions.push(p);
        }
        ("prompt", t.elapsed().as_secs_f64(), 0)
    } else {
        let n0 = args.score_from.expect("checked in parse");
        if n0 >= text.len() {
            bail!("--score-from {n0}: the text holds {} tokens", text.len());
        }
        let logits = CausalLm::prefill(model, &text[..n0], &mut cache)?;
        let mut row = logits.flatten_all()?.to_vec1::<f32>()?;
        let seconds_prefill = t.elapsed().as_secs_f64();
        let mut p = n0 - 1;
        let mut steps = 0;
        loop {
            nlls.push(nll(&row, text[p + 1])?);
            positions.push(p);
            p += 1;
            if p + 1 >= text.len() {
                break;
            }
            let logits = CausalLm::step(model, text[p], p, &mut cache)?;
            row = logits.flatten_all()?.to_vec1::<f32>()?;
            steps += 1;
        }
        ("steps", seconds_prefill, steps)
    };
    let seconds = t.elapsed().as_secs_f64();
    let mean = nlls.iter().sum::<f64>() / nlls.len() as f64;
    if let Some(p) = &args.nll_out {
        std::fs::write(p, serde_json::to_string(&json!({"positions": positions, "nll": nlls}))?)
            .with_context(|| format!("write {}", p.display()))?;
    }
    let report = json!({
        "bin": "inkrun", "mode": format!("score-{mode}"), "device": args.device, "tokens_file": args.tokens,
        "prompt_key": args.prompt_key, "text_tokens": text.len(), "score_from": args.score_from,
        "scored": nlls.len(), "mean_nll": mean, "ppl": mean.exp(), "seconds": seconds,
        "seconds_prefill": seconds_prefill, "steps": steps, "attn_chunk": model.attn_chunk(),
        "prefill_chunk": model.prefill_piece(),
        "nll_file": args.nll_out, "provenance": provenance,
        "memory": {"load": memory_load, "end": memory(dev)},
    });
    std::fs::write(&args.out, serde_json::to_string_pretty(&report)?).with_context(|| format!("write {}", args.out.display()))?;
    eprintln!("inkrun: scored {} positions ({mode}), ppl {:.4}, {seconds:.1} s, report {}", nlls.len(), mean.exp(), args.out.display());
    Ok(())
}
