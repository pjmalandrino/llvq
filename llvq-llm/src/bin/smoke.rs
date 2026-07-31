//! Gate G5 smoke test: quantize Qwen3-0.6B to ~2 bits and measure the cost.
//!
//! Usage:
//!   cargo run --release -p llvq-llm --features metal --bin smoke -- \
//!       [calib_windows] [calib_len] [eval_windows] [eval_ctx] [device]
//!
//! Reports baseline and quantized perplexity on the *same* windows, so the
//! only difference between the two numbers is the codebook.
//!
//! Deviations from the paper, stated up front:
//!   * calibration on WikiText-2 **train**, not 6 100 DCLM-edu sequences —
//!     same-domain calibration flatters the result, so this is a pipeline
//!     check, not a comparable figure;
//!   * far fewer calibration tokens than the paper uses;
//!   * shape–gain with 0 gain bits and the spherical retraction, which is the
//!     configuration Appendix I recommends.

use candle_core::{DType, Device, Tensor};

use llvq_llm::corpus::{hf_parquet_text, wikitext2_test};
use llvq_llm::loader::Checkpoint;
use llvq_llm::model::{NoCapture, Qwen3};
use llvq_quant::gptq::{GptqConfig, TailPolicy};


fn arg<T: std::str::FromStr>(a: &[String], i: usize, d: T) -> T {
    a.get(i).and_then(|s| s.parse().ok()).unwrap_or(d)
}

/// Streams matrices to disk as they are quantized. Buffered, because the
/// index stream is written in 6-byte units and 151 M of them through an
/// unbuffered `File` would be 151 M syscalls.
struct FileSink {
    w: llvq_llm::artifact2::ArtifactWriter<std::io::BufWriter<std::fs::File>>,
    path: String,
}

impl FileSink {
    fn create(path: &str, n: u32) -> anyhow::Result<Self> {
        let f = std::fs::File::create(path)?;
        Ok(Self {
            w: llvq_llm::artifact2::ArtifactWriter::new(
                std::io::BufWriter::with_capacity(1 << 20, f),
                n,
            )?,
            path: path.to_string(),
        })
    }
    fn finish(self) -> anyhow::Result<(u64, String)> {
        let bits = self.w.finish()?;
        Ok((bits, self.path))
    }
}

impl llvq_llm::calib::MatrixSink for FileSink {
    fn push(&mut self, m: llvq_llm::artifact2::QuantizedMatrix) -> anyhow::Result<()> {
        // `llvq-artifact` has its own error type on purpose — it carries no
        // dependencies, `anyhow` included. The bridge converts.
        Ok(self.w.push(&m)?)
    }
}

/// Decode the artifact and demand the evaluated weights back, bit for bit.
///
/// This is the whole point of writing a file rather than reporting a number:
/// a rate you cannot decode is a claim, not a measurement. Done one matrix at
/// a time — decoding Qwen3-4B in one go would be 14 GB.
fn verify_artifact(path: &str, model: &Qwen3) -> anyhow::Result<()> {
    use candle_core::DType;
    let f = std::fs::File::open(path)?;
    let mut r = std::io::BufReader::with_capacity(1 << 20, f);
    let n = llvq_llm::artifact2::read_header(&mut r)?;
    eprintln!("verifying {n} matrices against the evaluated model…");
    // One matrix at a time: a 4B model's codes are 14 GB of lattice points.
    let ix = llvq_search::index::Indexer::new();

    let mut checked = 0usize;
    for _ in 0..n {
        let m = &llvq_llm::artifact2::read_matrix(&mut r, &ix)?;
        let decoded = llvq_llm::artifact2::decode_matrix(m);
        // `name` is `model.layers.{b}.{proj}.weight`.
        let parts: Vec<&str> = m.name.split('.').collect();
        let b: usize = parts[2].parse()?;
        let proj = parts[3..parts.len() - 1].join(".");
        let want = model.blocks[b]
            .linear(&proj)
            .weight()
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        anyhow::ensure!(
            decoded.len() == want.len(),
            "{}: decoded {} weights, model holds {}",
            m.name,
            decoded.len(),
            want.len()
        );
        for (k, (g, e)) in decoded.iter().zip(want.iter()).enumerate() {
            anyhow::ensure!(
                g.to_bits() == e.to_bits(),
                "{name} weight {k}: artifact decodes {g:e}, model holds {e:e} \
                 (Δ = {delta:e}). The file is a different model from the one \
                 measured.",
                name = m.name,
                delta = g - e
            );
        }
        checked += decoded.len();
    }
    eprintln!("  ✓ {checked} weights identical, bit for bit");
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let n_calib: usize = arg(&a, 0, 16);
    let calib_len: usize = arg(&a, 1, 2048);
    let n_eval: usize = arg(&a, 2, 12);
    let eval_ctx: usize = arg(&a, 3, 2048);
    let device = if a.get(4).map(|s| s == "metal").unwrap_or(false) {
        Device::new_metal(0).unwrap_or(Device::Cpu)
    } else {
        Device::Cpu
    };
    // Algorithm 3's closed-form gain refinement. Appendix I treats it as part
    // of the 0-gain-bit configuration, not as an extra.
    let group_scales = a.get(5).map(|s| s == "gs").unwrap_or(false);
    // Diagnostics: which codebook, and how many blocks to touch.
    let kind = a.get(6).cloned().unwrap_or_else(|| "leech".into());
    let limit: usize = arg(&a, 7, usize::MAX);
    // Incoherence rotation on each linear's input basis (paper Table 9's
    // "Input" column). `rot` to enable.
    let rotation_seed = a.get(8).filter(|s| *s == "rot").map(|_| 0x11_0FEEDu64);
    // Where to persist the quantized projections, so the model can be probed
    // later instead of being re-quantized.
    let save_to = a.get(9).cloned();
    // `LLVQ_THREADS` caps the encoder pool. A full run wants every core, but
    // an A/B launched on a machine someone is working on should not take the
    // whole machine — the Leech encoder is the part that saturates it.
    let threads = std::env::var("LLVQ_THREADS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|n| *n > 0)
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
        });
    eprintln!("device {device:?}, {threads} encoder threads");

    let repo = std::env::var("LLVQ_MODEL").unwrap_or_else(|_| "Qwen/Qwen3-0.6B".into());
    let ck = Checkpoint::fetch(&repo)?;
    let tok = ck.tokenizer()?;
    let vb = ck.var_builder(DType::F32, &device)?;
    let mut model = Qwen3::new(&ck.config, vb)?;

    // ---- evaluation windows (fixed, used before and after) ----
    let test_ids = tok
        .encode(wikitext2_test()?.as_str(), false)
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .get_ids()
        .to_vec();
    let n_eval = n_eval.min(test_ids.len() / eval_ctx);
    anyhow::ensure!(n_eval > 0, "corpus shorter than one eval window");

    let ppl = |m: &Qwen3| -> anyhow::Result<f64> {
        let (mut nll, mut cnt) = (0.0, 0usize);
        for w in 0..n_eval {
            let (n, c) = m.window_nll(&test_ids[w * eval_ctx..(w + 1) * eval_ctx], &mut NoCapture)?;
            nll += n;
            cnt += c;
        }
        Ok((nll / cnt as f64).exp())
    };

    eprintln!("baseline perplexity over {n_eval} windows of {eval_ctx}…");
    let base = ppl(&model)?;
    eprintln!("  baseline ppl = {base:.4}");

    // ---- calibration windows ----
    // Calibrating on the corpus we evaluate on flatters the result by ~12 %,
    // measured. `LLVQ_CALIB=c4` reproduces the paper's setup: calibrate out of
    // domain, evaluate on wikitext.
    let calib_kind = std::env::var("LLVQ_CALIB").unwrap_or_else(|_| "wikitext2".into());
    eprintln!("tokenizing calibration set ({calib_kind})…");
    let train = if calib_kind == "c4" {
        llvq_llm::corpus::c4_validation(8_000_000)?
    } else {
        hf_parquet_text(
            "Salesforce/wikitext",
            "wikitext-2-raw-v1/train-00000-of-00001.parquet",
        )?
        .join("\n\n")
    };
    let train_ids = tok
        .encode(train.as_str(), false)
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .get_ids()
        .to_vec();
    let n_calib = n_calib.min(train_ids.len() / calib_len);
    anyhow::ensure!(n_calib > 0, "calibration corpus too short");
    eprintln!("  {n_calib} windows of {calib_len} = {} tokens", n_calib * calib_len);

    let mut hidden: Vec<Tensor> = Vec::with_capacity(n_calib);
    for w in 0..n_calib {
        let ids = &train_ids[w * calib_len..(w + 1) * calib_len];
        let t = Tensor::from_slice(ids, (1, calib_len), &device)?;
        hidden.push(model.embed_tokens(&t)?);
    }

    // ---- quantize ----
    let cfg = GptqConfig {
        block: llvq_core::DIM,
        retract: true, // Spherical GPTQ
        group_scales,
        lambda: 1e-2,
        tail: TailPolicy::KeepExact,
    };
    eprintln!(
        "quantizing {} blocks of {repo} (shape–gain, 0 gain bits, spherical \
         retraction, group scales {}, input rotation {})…",
        model.blocks.len(),
        if group_scales { "on" } else { "off" },
        if rotation_seed.is_some() { "on" } else { "off" }
    );
    let codebook = match kind.as_str() {
        // The control that decides "bug or just hard": a lossless codebook
        // must bring the perplexity back exactly.
        "identity" => llvq_llm::calib::Codebook::Identity,
        "grid" => llvq_llm::calib::Codebook::Grid { step: 0.01 },
        // Direction only: the magnitude stays a free float, so this is NOT a
        // 2 bit/weight code. Kept as the quality ceiling of the direction.
        "direction" => llvq_llm::calib::Codebook::Direction,
        // `leech1` = one gain bit over the full ball; `leech1c12` caps the
        // direction code at shell 12, which frees the bit the gain costs.
        // A trailing `f` — `leech1c12f` — restores the pre-2026-07-31
        // behaviour where the retraction cancelled the gain code and the
        // magnitude was a free f16 per block. It is charged as 16 bits, so the
        // two are directly comparable on the rate line.
        other => {
            let rest = other.strip_prefix("leech").unwrap_or("1");
            let (rest, free_magnitude) = match rest.strip_suffix('f') {
                Some(r) => (r, true),
                None => (rest, false),
            };
            let (g, c) = match rest.split_once('c') {
                Some((g, c)) => (g, c.parse().unwrap_or(13)),
                None => (rest, 13),
            };
            llvq_llm::calib::Codebook::ShapeGain {
                gain_bits: g.parse().unwrap_or(1),
                max_shell: c,
                free_magnitude,
            }
        }
    };

    // `LLVQ_ARTIFACT=<path>` writes the real compressed artifact: packed
    // lattice indices, not reconstructions. The file's size is the bit rate.
    let artifact_path = std::env::var("LLVQ_ARTIFACT").ok();
    let n_matrices = 7 * limit.min(model.blocks.len());

    let t0 = std::time::Instant::now();
    let run = llvq_llm::calib::RunConfig {
        gptq: cfg,
        damping: 1e-2,
        codebook,
        threads,
        limit,
        rotation_seed,
    };
    let progress = |t: usize, n: usize, name: &str| {
        if name == "mlp.down_proj" {
            eprintln!("  block {:>2}/{n} done ({:.0}s)", t + 1, t0.elapsed().as_secs_f64());
        }
    };
    let mut sink = match &artifact_path {
        Some(p) => {
            eprintln!("writing the compressed artifact to {p}");
            Some(FileSink::create(p, n_matrices as u32)?)
        }
        None => None,
    };
    let report = llvq_llm::calib::quantize_model_capturing(
        &mut model,
        &mut hidden,
        &run,
        progress,
        sink.as_mut().map(|s| s as &mut dyn llvq_llm::calib::MatrixSink),
    )?;
    if let Some(s) = sink {
        let (bits, path) = s.finish()?;
        let bytes = std::fs::metadata(&path)?.len();
        eprintln!(
            "artifact: {:.3} GB on disk, payload {:.4} bits/weight over {} \
             quantized weights",
            bytes as f64 / 1e9,
            bits as f64 / (report.weights - report.tail_weights) as f64,
            report.weights - report.tail_weights,
        );
        verify_artifact(&path, &model)?;
    }

    eprintln!(
        "quantized {} matrices, {} weights in {:.0}s ({:.4} bits/weight)",
        report.matrices,
        report.weights,
        report.seconds,
        report.bits_per_weight()
    );

    if let Some(path) = &save_to {
        llvq_llm::artifact::save(&model, path)?;
        eprintln!("quantized projections written to {path}");
    }

    let quant = ppl(&model)?;
    println!("\n=== {repo} [{kind}, {} blocks, rot {}, calib {calib_kind}], wikitext-2, ctx {eval_ctx}, {n_eval} windows ===",
        if limit == usize::MAX { model.blocks.len() } else { limit.min(model.blocks.len()) },
        if rotation_seed.is_some() { "on" } else { "off" });
    println!("baseline (FP32)      ppl = {base:.4}");
    println!("LLVQ 2-bit           ppl = {quant:.4}");
    println!("degradation          ×{:.3}", quant / base);
    println!("effective rate           = {:.4} bits/weight", report.bits_per_weight());
    Ok(())
}
