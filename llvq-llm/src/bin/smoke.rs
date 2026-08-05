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
//!
//! Environment knobs: `LLVQ_MODEL`, `LLVQ_CALIB` (`c4` for the paper's
//! out-of-domain setup), `LLVQ_ARTIFACT`, `LLVQ_THREADS`, `LLVQ_DAMPING`, and
//! `LLVQ_CALIB_SEED` — the last one draws the calibration windows at random
//! offsets instead of taking a prefix, which is how a **run-to-run error bar**
//! gets measured. Three seeds on 3 blocks is the cheapest thing in this repo
//! that turns a 3 % difference from an anecdote into a result.

use candle_core::{DType, Tensor};

use llvq_llm::corpus::{hf_parquet_text, wikitext2_test};
use llvq_llm::loader::Checkpoint;
use llvq_llm::model::{NoCapture, Qwen3};
use llvq_quant::gptq::{GptqConfig, TailPolicy};


fn arg<T: std::str::FromStr>(a: &[String], i: usize, d: T) -> T {
    a.get(i).and_then(|s| s.parse().ok()).unwrap_or(d)
}

/// Where each calibration window starts in the tokenized corpus.
///
/// `seed = None` reproduces the historical behaviour — a contiguous prefix
/// from token 0 — and stays the default, because that is what every published
/// run used and silently moving it would orphan those numbers.
///
/// But a fixed prefix makes run-to-run variance **unmeasurable**, and that is
/// the gap worth closing: several conclusions in this project rest on 3–7 %
/// differences whose noise floor nobody knows. `LLVQ_CALIB_SEED=<n>` draws the
/// windows at random offsets over the whole corpus, which is what GPTQ, QuIP#
/// and QTIP all do, and which makes "run it under three seeds and look at the
/// spread" possible.
///
/// ⚠️ Do **not** expect a perplexity gain from it. Under `LLVQ_CALIB=c4` the
/// corpus is already hundreds of unrelated web documents concatenated, so "the
/// first 500 documents of a crawl" and "500 documents drawn at random" are
/// nearly the same sample. The deliverable here is the error bar, not the
/// mean — and if the three seeds land far apart, that is itself the finding.
fn window_starts(n: usize, ntokens: usize, len: usize, seed: Option<u64>) -> Vec<usize> {
    let Some(seed) = seed else {
        return (0..n).map(|w| w * len).collect();
    };
    assert!(ntokens >= len, "corpus shorter than one window");
    // Offsets are unaligned on purpose: aligning them to multiples of `len`
    // would sample the same grid the prefix already walks, only in a different
    // order, and would not probe the corpus any more widely.
    let span = (ntokens - len + 1) as u64;
    let mut rng = llvq_core::SplitMix64::new(seed);
    let mut seen = std::collections::HashSet::with_capacity(n);
    let mut out = Vec::with_capacity(n);
    // Distinct windows: a repeated one would weight its tokens twice in the
    // Hessian for nothing. `span` dwarfs `n` on any corpus large enough to
    // calibrate on, so the retry budget is a formality — but an unbounded
    // loop on a short corpus would hang instead of reporting.
    for _ in 0..(64 * n).max(1024) {
        if out.len() == n {
            break;
        }
        let s = (rng.next() % span) as usize;
        if seen.insert(s) {
            out.push(s);
        }
    }
    assert_eq!(
        out.len(),
        n,
        "corpus too short to draw {n} distinct windows of {len}"
    );
    out
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
fn verify_artifact(path: &str, model: &Qwen3, dtype: DType) -> anyhow::Result<()> {
    let f = std::fs::File::open(path)?;
    let mut r = std::io::BufReader::with_capacity(1 << 20, f);
    let head = llvq_llm::artifact2::read_header(&mut r)?;
    let n = head.matrices;
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
            // The decoder works in f32; the model holds its weights at the
            // run's dtype. At F32 this narrowing is the identity and the
            // comparison is the one this proof has always made. At a half
            // precision it becomes "the file decodes to the evaluated weights
            // at the precision the model stores them" — see `eval::narrow`.
            let g = llvq_llm::eval::narrow(*g, dtype);
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
    eprintln!(
        "  ✓ {checked} weights identical, bit for bit (à {})",
        llvq_llm::eval::dtype_name(dtype)
    );
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let n_calib: usize = arg(&a, 0, 16);
    let calib_len: usize = arg(&a, 1, 2048);
    let n_eval: usize = arg(&a, 2, 12);
    let eval_ctx: usize = arg(&a, 3, 2048);
    let device = llvq_llm::eval::device(a.get(4).map(String::as_str).unwrap_or("cpu"))?;
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
    // F32 by default — every published run used it, and the identity control
    // depends on it. `LLVQ_DTYPE=bf16` halves the resident model, which is
    // what makes a 32B fit: 131 GB in f32 exceeds every single-card flavor,
    // 65.5 GB in bf16 sits comfortably on a 96 GB one. That is a $130
    // difference on one run.
    //
    // The round-trip proof follows the dtype rather than being weakened by it:
    // `verify_artifact` narrows the decode to the model's precision, so the
    // claim stays "the file decodes to the weights that were evaluated".
    // The Hessian accumulator is F32 regardless (see `Hessian::new`), so
    // calibration precision does not ride on this.
    let dtype = llvq_llm::eval::dtype(DType::F32)?;
    eprintln!("model dtype {}", llvq_llm::eval::dtype_name(dtype));
    let vb = ck.var_builder(dtype, &device)?;
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
    // `LLVQ_CALIB=c4` reproduces the paper's setup: calibrate out of domain,
    // evaluate on wikitext. (The "~12 %" this comment used to claim was a
    // misreading — it measured how much harder C4 is as an *evaluation*
    // corpus, not a calibration advantage. See CLAUDE.md §Qwen3-4B.)
    let calib_kind = std::env::var("LLVQ_CALIB").unwrap_or_else(|_| "wikitext2".into());
    eprintln!("tokenizing calibration set ({calib_kind})…");
    let train = if calib_kind == "c4" {
        // A different shard from the one `bin/ppl` evaluates on — otherwise
        // calibrating on C4 and scoring on C4 is the same text twice.
        llvq_llm::corpus::c4_calibration(8_000_000)?
    } else if calib_kind == "wikitext2-test" {
        // The calibration *oracle* (pistes-battre-q4.md P3): deliberate
        // contamination — calibrate on the very text the eval windows score.
        // Not a config anyone ships; it bounds the ceiling of the whole
        // calibration family (volume, corpus, length) in one 3-block run.
        wikitext2_test()?
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
    // Unset = the contiguous prefix every published run used. Set = random
    // offsets, which is how an error bar gets measured. See `window_starts`.
    let calib_seed = std::env::var("LLVQ_CALIB_SEED")
        .ok()
        .and_then(|s| s.parse::<u64>().ok());
    eprintln!(
        "  {n_calib} windows of {calib_len} = {} tokens, {}",
        n_calib * calib_len,
        match calib_seed {
            Some(s) => format!("seeded offsets (seed {s})"),
            None => "contiguous prefix from token 0".into(),
        }
    );

    let starts = window_starts(n_calib, train_ids.len(), calib_len, calib_seed);
    let mut hidden: Vec<Tensor> = Vec::with_capacity(n_calib);
    for &s in &starts {
        let ids = &train_ids[s..s + calib_len];
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
            // A trailing `L<n>` caps the distinct magnitudes per block —
            // `leech1c12L3` is one gain bit, shell ≤ 12, at most three
            // levels. That cap is what sets the fused kernel's RAM width.
            let (rest, level_cap) = match rest.split_once('L') {
                Some((r, l)) => (r, l.parse().unwrap_or(5)),
                None => (rest, llvq_search::generic::MAX_LEVELS_ANY),
            };
            let (g, c) = match rest.split_once('c') {
                Some((g, c)) => (g, c.parse().unwrap_or(13)),
                None => (rest, 13),
            };
            llvq_llm::calib::Codebook::ShapeGain {
                gain_bits: g.parse().unwrap_or(1),
                max_shell: c,
                free_magnitude,
                level_cap,
            }
        }
    };

    // `LLVQ_ARTIFACT=<path>` writes the real compressed artifact: packed
    // lattice indices, not reconstructions. The file's size is the bit rate.
    let artifact_path = std::env::var("LLVQ_ARTIFACT").ok();
    let n_matrices = 7 * limit.min(model.blocks.len());

    // Hessian damping, relative to `mean(diag H)` — `H + λ·mean(diag H)·I`.
    //
    // 1e-2 was the only value ever passed, on every layer width (2560, 4096,
    // 9728), and it was never swept. That is not a defensible state for the
    // parameter that conditions the whole error compensation, in a repo that
    // sweeps β to the thousandth. `LLVQ_DAMPING` makes {3e-3, 1e-2, 3e-2} on
    // 3 blocks an eight-minute A/B.
    //
    // The prediction is that it changes nothing: because the damping is
    // relative to `mean(diag H)` and an LLM activation Hessian is heavy-tailed
    // — a few massive-activation directions carry most of the trace — that
    // floor already dominates the noisy directions. It is why GPTQ, QuIP# and
    // QTIP all use a single relative `percdamp` whatever the width. Publish
    // the null result; "never measured" is the thing to fix.
    let damping = match std::env::var("LLVQ_DAMPING") {
        Ok(s) => s
            .parse::<f64>()
            .map_err(|_| anyhow::anyhow!("LLVQ_DAMPING={s:?} is not a number"))?,
        Err(_) => 1e-2,
    };
    anyhow::ensure!(
        damping >= 0.0 && damping.is_finite(),
        "LLVQ_DAMPING must be finite and non-negative, got {damping}"
    );
    eprintln!("hessian damping {damping:e} (relative to mean(diag H))");

    let t0 = std::time::Instant::now();
    let run = llvq_llm::calib::RunConfig {
        gptq: cfg,
        damping,
        codebook,
        threads,
        limit,
        rotation_seed,
    };
    // One line per transformer block, with an ETA. On a multi-hour rented job
    // the question is never "did it start" but "is it on track", and a bare
    // elapsed counter cannot answer that. `n` is the model's block count; the
    // run may stop earlier under `limit`, so the projection uses the target.
    let n_target = limit.min(model.blocks.len());
    let progress = move |t: usize, _n: usize, name: &str| {
        if name == "mlp.down_proj" {
            let done = t + 1;
            let el = t0.elapsed().as_secs_f64();
            let per = el / done as f64;
            let left = n_target.saturating_sub(done);
            eprintln!(
                "  bloc {done:>3}/{n_target}  {:.0} min écoulées  {per:.0} s/bloc  \
                 reste ~{:.0} min  (fin estimée à +{:.1} h)",
                el / 60.0,
                per * left as f64 / 60.0,
                (el + per * left as f64) / 3600.0
            );
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
        verify_artifact(&path, &model, dtype)?;
    }

    eprintln!(
        "quantized {} matrices, {} weights in {:.0}s ({:.4} bits/weight)",
        report.matrices,
        report.weights,
        report.seconds,
        report.bits_per_weight()
    );
    // Where the time went, largest first. Which phase dominates flips with the
    // backend — Leech encoding on Metal, forward passes on a CPU-only job — so
    // the only way to know what to optimize (and which flavor to rent) is to
    // read it off the run itself.
    eprintln!("phases :");
    for (name, secs, pct) in report.phases.ranked() {
        eprintln!("  {name:<22}{secs:>9.1}s{pct:>7.1} %");
    }
    // A 40× difference that is otherwise invisible. Measured on one block of
    // Qwen3-0.6B: factorization 28.4 s without `fast-linalg`, 0.7 s with it,
    // for a bit-identical perplexity. The feature has to stay opt-in — it is
    // what keeps `llvq-quant` free of external dependencies, which the project
    // claims — so the omission is made loud instead.
    if !cfg!(feature = "fast-linalg") {
        eprintln!(
            "\n  ⚠️  compilé SANS `fast-linalg` : la factorisation tourne sur \
             l'implémentation\n      de référence, ~40× plus lente que `faer` \
             pour un résultat identique.\n      Ajouter `--features fast-linalg` \
             avant de payer du matériel."
        );
    }

    if let Some(path) = &save_to {
        llvq_llm::artifact::save(&model, path)?;
        eprintln!("quantized projections written to {path}");
    }

    let quant = ppl(&model)?;
    println!("\n=== {repo} [{kind}, {} blocks, rot {}, calib {calib_kind}/{}], wikitext-2, ctx {eval_ctx}, {n_eval} windows ===",
        if limit == usize::MAX { model.blocks.len() } else { limit.min(model.blocks.len()) },
        if rotation_seed.is_some() { "on" } else { "off" },
        match calib_seed {
            Some(s) => format!("seed {s}"),
            None => "prefix".into(),
        });
    println!("baseline (FP32)      ppl = {base:.4}");
    println!("LLVQ 2-bit           ppl = {quant:.4}");
    println!("degradation          ×{:.3}", quant / base);
    println!("effective rate           = {:.4} bits/weight", report.bits_per_weight());
    // On the result line, not only in stderr: an A/B whose swept parameter is
    // not printed with its number is an A/B nobody can re-read six weeks later.
    println!("hessian damping          = {damping:e}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::window_starts;

    const N: usize = 8;
    const LEN: usize = 2048;
    const TOKENS: usize = 4_000_000;

    /// No seed must reproduce the historical prefix exactly. Every published
    /// number was measured on it, so a change here reinterprets them.
    #[test]
    fn no_seed_is_the_contiguous_prefix() {
        let s = window_starts(N, TOKENS, LEN, None);
        assert_eq!(s, (0..N).map(|w| w * LEN).collect::<Vec<_>>());
    }

    /// Every window must fit. An off-by-one in the span would slice past the
    /// end of the token vector and panic three hours into a run.
    #[test]
    fn every_window_fits_in_the_corpus() {
        // Deliberately tight: the last legal start is exactly `ntokens - len`.
        for tokens in [TOKENS, N * LEN, N * LEN + 1] {
            for &s in &window_starts(N, tokens, LEN, Some(7)) {
                assert!(s + LEN <= tokens, "window at {s} overruns {tokens}");
            }
        }
    }

    /// A seed has to be reproducible, and two seeds have to disagree —
    /// otherwise the three runs that are supposed to produce an error bar
    /// would produce the same number three times and report a spread of zero.
    #[test]
    fn seeds_are_reproducible_and_distinct() {
        assert_eq!(
            window_starts(N, TOKENS, LEN, Some(1)),
            window_starts(N, TOKENS, LEN, Some(1))
        );
        assert_ne!(
            window_starts(N, TOKENS, LEN, Some(1)),
            window_starts(N, TOKENS, LEN, Some(2))
        );
    }

    /// The point of seeding is to leave the head of the corpus, not to shuffle
    /// within it. Drawing from `0..n·len` would satisfy every test above and
    /// sample exactly the same tokens as the prefix.
    #[test]
    fn seeded_windows_leave_the_prefix() {
        let s = window_starts(N, TOKENS, LEN, Some(3));
        assert!(
            s.iter().any(|&x| x > N * LEN),
            "every window landed inside the prefix: {s:?}"
        );
    }

    /// Repeated windows would weight their tokens twice in the Hessian.
    #[test]
    fn windows_are_distinct() {
        let s = window_starts(64, TOKENS, LEN, Some(5));
        let uniq: std::collections::HashSet<_> = s.iter().copied().collect();
        assert_eq!(uniq.len(), s.len());
    }
}
