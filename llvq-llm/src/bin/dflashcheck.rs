//! The DFlash drafter against its MLX fixture, then the time of one round.
//!
//! ```text
//! dflashcheck <fixture.safetensors> [cpu|metal|cuda] [packed_dir] [drafter_dir] [f32|f16|bf16] [dump=<file>|against=<file>|profile]
//! ```
//!
//! `dump=<file>` writes the timed dtype's logits of the six fixture rounds;
//! `against=<file>` requires them within 1e-3 relative (max |dz| / max |z| a
//! round) of such a dump: the gate of a change that must not move what the
//! drafter drafts beyond its dtype's noise.
//!
//! `profile` (measurement only, nothing else runs): every operation of a round
//! alone, a device synchronisation after each, median of 30 after 5, for the
//! extension of 4 positions at a context of 512, one block layer in detail
//! and the five layers' sums, the head, both readouts, and three candidates
//! (argmax only; top 4 on f16 logits with lse by six operations; the readout
//! before e3bbaee); each stage also whole, one synchronisation at its end.
//!
//! `fixture`: written by `ops/dflash_fixture.py`. `packed_dir`: the hfpack
//! directory whose int4 g64 table the fixture's reference used (default
//! `~/q4b-hf-2026-09-28`), or the sealed file itself, whose table is the same
//! (read through its records, `dflash::vocab_from_sealed`): on the card. `drafter_dir`: the snapshot of
//! `z-lab/Qwen3-4B-DFlash-b16` at the pinned revision (default: the local
//! Hugging Face cache; never downloads). The equivalence runs in f32 whatever
//! the last argument, which sets the dtype of the timed rounds only.
//!
//! A round, timed: extend the context by 4 positions (`fc` and five k/v
//! projections), then the block of 16 rows through the five layers, the head
//! over 15 rows and the readout (argmax, top 4, logsumexp). Median of 20 after
//! 3 warm-up rounds, at contexts of 128, 512 and 2,048 positions.
//!
//! Then the round as the loop pays it (`DFlashSpec::draft`, binary16 taps from
//! the host): one extension of n positions (n = 1, 4, 8) and one draft, chain
//! and tree, no synchronisation between stages, one at the end; median of 50
//! after 10 discarded, from a context of 512. And the same rounds with a
//! synchronisation between the stages, for the split.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::Context;
use candle_core::DType;
use llvq_llm::dflash::{self, DFlashDrafter, DenseVocab};

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let fixture = PathBuf::from(a.get(1).context("usage: dflashcheck <fixture> [device] [packed] [drafter] [dtype]")?);
    let device = llvq_llm::eval::device(a.get(2).map(String::as_str).unwrap_or("cpu"))?;
    let home = std::env::var("HOME").unwrap_or_default();
    let packed = a.get(3).map(PathBuf::from).unwrap_or_else(|| PathBuf::from(&home).join("q4b-hf-2026-09-28"));
    let ddir = match a.get(4) {
        Some(d) => PathBuf::from(d),
        None => dflash::cached_snapshot()
            .with_context(|| format!("{}@{} is not in the local cache", dflash::REPO, dflash::REVISION))?,
    };
    let dtype = match a.get(5).map(String::as_str).unwrap_or("f32") {
        "f32" => DType::F32,
        "f16" => DType::F16,
        "bf16" => DType::BF16,
        o => anyhow::bail!("dtype {o}: f32, f16 or bf16"),
    };

    if a.get(6).map(String::as_str) == Some("profile") {
        // Measurement only: every operation of a round alone (dflash::profile), in the timed dtype.
        let drafter = DFlashDrafter::load(&ddir, &device, dtype)?;
        let vocab = if packed.is_file() {
            dflash::vocab_from_sealed(&packed, &device, dtype)?
        } else {
            dflash::vocab_from_packed(&packed, &device, dtype)?
        };
        let fx = candle_core::safetensors::load(&fixture, &candle_core::Device::Cpu)?;
        let taps = fx.get("p0.taps").context("fixture: no p0.taps")?;
        let rows = taps.dim(0)?;
        // a context of 512 positions from the fixture's taps, repeated
        let reps: Vec<candle_core::Tensor> = (0..512usize.div_ceil(rows)).map(|_| taps.clone()).collect();
        let ctx0 = candle_core::Tensor::cat(&reps, 0)?.narrow(0, 0, 512)?.to_device(&device)?;
        let n = 4;
        let new: Vec<u16> = taps.narrow(0, 0, n)?.flatten_all()?.to_vec1::<f32>()?.iter().map(|&x| half::f16::from_f32(x).to_bits()).collect();
        let t = Instant::now();
        dflash::profile::run(&drafter, &vocab, &ctx0, &new, n)?;
        println!("profile done in {:.1} s", t.elapsed().as_secs_f64());
        return Ok(());
    }
    let t0 = Instant::now();
    let drafter = DFlashDrafter::load(&ddir, &device, DType::F32)?;
    let vocab = if packed.is_file() {
        dflash::vocab_from_sealed(&packed, &device, DType::F32)?
    } else {
        dflash::vocab_from_packed(&packed, &device, DType::F32)?
    };
    println!(
        "drafter {}@{} ({:.1} MB f32), table {:?} from {}, device {:?}, loaded in {:.1} s",
        dflash::REPO,
        &dflash::REVISION[..8],
        drafter.weight_bytes() as f64 / 1e6,
        vocab.table.dims(),
        packed.display(),
        device,
        t0.elapsed().as_secs_f64()
    );
    let rep = dflash::check_fixture(&fixture, &drafter, &vocab)?;
    println!("equivalence (f32): {rep}");
    let ok = rep.chain_equal == rep.rounds && rep.tree_equal == rep.rounds && rep.logit_rel < 1e-4;
    println!("equivalence {}", if ok { "PASS" } else { "FAIL" });

    let (drafter, vocab) = if dtype == DType::F32 {
        (drafter, vocab)
    } else {
        let v = DenseVocab { table: vocab.table.to_dtype(dtype)? };
        drop(drafter);
        (DFlashDrafter::load(&ddir, &device, dtype)?, v)
    };
    // The timed dtype must draft too: the f16 drafter overflowed in fc and drafted NaN (SP2,
    // 2026-11-10) while the f32 check above passed. Below f32 the tokens may part on near-ties,
    // so the gate is the logits' error (NaN fails it) and at least half the rounds' chains.
    let ok = if dtype == DType::F32 {
        ok
    } else {
        let low = dflash::check_fixture(&fixture, &drafter, &vocab)?;
        println!("equivalence ({dtype:?}): {low}");
        let pass = low.logit_rel < 5e-2 && 2 * low.chain_equal >= low.rounds;
        println!("equivalence ({dtype:?}) {}", if pass { "PASS" } else { "FAIL" });
        ok && pass
    };
    let fx = candle_core::safetensors::load(&fixture, &candle_core::Device::Cpu)?;
    let taps = fx.get("p0.taps").context("fixture: no p0.taps")?.to_device(&device)?;
    let n = taps.dim(0)?;
    for target in [128usize, 512, 2048] {
        let mut ctx = drafter.new_context(target + 256)?;
        while ctx.len() < target {
            let k = (target - ctx.len()).min(n);
            drafter.extend(&mut ctx, &taps.narrow(0, 0, k)?)?;
        }
        let four = taps.narrow(0, 0, 4)?;
        let (mut te, mut td) = (Vec::new(), Vec::new());
        for i in 0..23 {
            device.synchronize()?;
            let t = Instant::now();
            drafter.extend(&mut ctx, &four)?;
            device.synchronize()?;
            let e = t.elapsed().as_secs_f64();
            let t = Instant::now();
            let out = drafter.draft(&ctx, 151643, &vocab)?;
            let _tree = dflash::draft_tree(&out, 15);
            device.synchronize()?;
            let d = t.elapsed().as_secs_f64();
            if i >= 3 {
                te.push(e);
                td.push(d);
            }
        }
        let med = |v: &mut Vec<f64>| {
            v.sort_by(f64::total_cmp);
            (v[v.len() / 2] * 1e3, v[0] * 1e3, v[v.len() - 1] * 1e3)
        };
        let (e, e0, e1) = med(&mut te);
        let (d, d0, d1) = med(&mut td);
        println!(
            "round at context {target} ({dtype:?}): extend 4 {e:.2} ms [{e0:.2}-{e1:.2}], block + head + readout \
             {d:.2} ms [{d0:.2}-{d1:.2}], median of 20; context cache {:.1} MB",
            ctx.bytes() as f64 / 1e6
        );
    }
    let mut ok = ok;
    if let Some(arg) = a.get(6) {
        let logits = dflash::fixture_logits(&fixture, &drafter, &vocab)?;
        if let Some(file) = arg.strip_prefix("dump=") {
            let map: std::collections::HashMap<String, candle_core::Tensor> =
                logits.into_iter().enumerate().map(|(i, t)| (format!("r{i}"), t)).collect();
            candle_core::safetensors::save(&map, file)?;
            println!("logits ({dtype:?}) of {} rounds written to {file}", map.len());
        } else if let Some(file) = arg.strip_prefix("against=") {
            let base = candle_core::safetensors::load(file, &candle_core::Device::Cpu)?;
            let mut worst = 0f64;
            for (i, t) in logits.iter().enumerate() {
                let b = base.get(&format!("r{i}")).with_context(|| format!("{file}: no r{i}"))?;
                let d = (t - b)?.abs()?.max_all()?.to_scalar::<f32>()? as f64;
                let s = b.abs()?.max_all()?.to_scalar::<f32>()? as f64;
                worst = worst.max(d / s);
            }
            let pass = worst < 1e-3 && worst.is_finite() && logits.len() == base.len();
            println!("logits ({dtype:?}) against {file}: max rel {worst:.2e} over {} rounds: {}", logits.len(), if pass { "PASS" } else { "FAIL" });
            // The f16 noise floor, for reading the line above: both against the f32 logits of this build.
            let f32d = DFlashDrafter::load(&ddir, &device, DType::F32)?;
            let f32v = DenseVocab { table: vocab.table.to_dtype(DType::F32)? };
            let truth = dflash::fixture_logits(&fixture, &f32d, &f32v)?;
            drop((f32d, f32v));
            let (mut now, mut then) = (0f64, 0f64);
            for (i, t) in truth.iter().enumerate() {
                let b = base.get(&format!("r{i}")).with_context(|| format!("{file}: no r{i}"))?;
                let s = t.abs()?.max_all()?.to_scalar::<f32>()? as f64;
                now = now.max((&logits[i] - t)?.abs()?.max_all()?.to_scalar::<f32>()? as f64 / s);
                then = then.max((b - t)?.abs()?.max_all()?.to_scalar::<f32>()? as f64 / s);
            }
            println!("logits ({dtype:?}) against the f32 logits: this build {now:.2e}, the dump {then:.2e}");
            ok &= pass;
        } else {
            anyhow::bail!("sixth argument {arg}: dump=<file> or against=<file>");
        }
    }
    loop_rounds(&fixture, &device, drafter, vocab)?;
    if !ok {
        std::process::exit(1);
    }
    Ok(())
}

/// Without `llvq-cuda` (Linux without `cuda`, CI's CPU build) there is no
/// loop to time: refused by name.
#[cfg(not(any(feature = "cuda", target_os = "macos")))]
fn loop_rounds(_: &std::path::Path, _: &candle_core::Device, _: DFlashDrafter, _: DenseVocab) -> anyhow::Result<()> {
    anyhow::bail!("dflashcheck: the loop round needs llvq-cuda (macOS, or Linux with --features cuda)")
}

/// The round as the loop pays it, through `DFlashSpec`.
#[cfg(any(feature = "cuda", target_os = "macos"))]
fn loop_rounds(fixture: &std::path::Path, device: &candle_core::Device, drafter: DFlashDrafter, vocab: DenseVocab) -> anyhow::Result<()> {
    use llvq_cuda::spec_loop::{DraftInput, Drafter, TAPS};
    use llvq_llm::dflash_spec::{DFlashSpec, Shape};
    let fx = candle_core::safetensors::load(fixture, &candle_core::Device::Cpu)?;
    let taps = fx.get("p0.taps").context("fixture: no p0.taps")?;
    let hidden = drafter.cfg.hidden;
    let w = TAPS * hidden;
    let bits: Vec<u16> = taps.flatten_all()?.to_vec1::<f32>()?.iter().map(|&x| half::f16::from_f32(x).to_bits()).collect();
    let rows = bits.len() / w;
    let l = drafter.launches();
    println!(
        "device operations a round, counted from the code: extend {}, block {}, head {}, readout {} for a tree, {} for a chain \
         ({} with the softmax readout, {} with the chunked one); device-to-host transfers: {} ({} with the chunked one, plus its upload)",
        l.extend, l.block, l.head, l.readout, dflash::LseReadout::LAUNCHES_CHAIN, dflash::DeviceReadout::LAUNCHES, 5, l.syncs, 2
    );
    // the readout alone, on logits of the head's shape, each call ending in its transfer
    let z = candle_core::Tensor::randn(0f32, 4f32, (dflash::DRAFTS, vocab.table.dim(0)?), device)?;
    let dr = dflash::DeviceReadout::new(dflash::DRAFTS, vocab.table.dim(0)?, device)?;
    let lr = dflash::LseReadout::new(dflash::DRAFTS, device)?;
    for kind in 0..4 {
        let mut t = Vec::new();
        for i in 0..60 {
            device.synchronize()?;
            let t0 = Instant::now();
            let _ = match kind {
                0 => lr.tree(&z)?,
                1 => lr.chain(&z)?,
                2 => dr.run(&z)?,
                _ => dflash::readout_chunked(&z)?,
            };
            device.synchronize()?;
            if i >= 10 {
                t.push(t0.elapsed().as_secs_f64() * 1e3);
            }
        }
        t.sort_by(f64::total_cmp);
        let name = ["no softmax, tree", "no softmax, chain", "softmax, 077aa5b", "chunked, e3bbaee"][kind];
        println!("readout alone ({name}): {:.3} ms [{:.3}-{:.3}], median of 50", t[t.len() / 2], t[0], t[t.len() - 1]);
    }
    let mut sp = DFlashSpec::new(drafter, Box::new(vocab), Shape::Chain(15))?;
    let ctx0 = 512;
    for (kind, shape) in [(0, Shape::Chain(15)), (0, Shape::Tree(15)), (1, Shape::Tree(15)), (2, Shape::Tree(15))] {
        sp.softmax_readout = kind == 1;
        sp.chunked_readout = kind == 2;
        let tag = [" (no-softmax readout)", " (softmax readout)", " (chunked readout)"][kind];
        for n in [1usize, 4, 8] {
            for split in [false, true] {
                sp.shape = shape;
                sp.reset()?;
                sp.timing = split;
                let mut tokens: Vec<u32> = (0..=ctx0 as u32).map(|i| 1000 + i % 5000).collect();
                // the first call takes the prompt's 512 positions
                let mut feed: Vec<u16> = Vec::with_capacity(ctx0 * w);
                while feed.len() < ctx0 * w {
                    let k = ((ctx0 * w - feed.len()) / w).min(rows);
                    feed.extend_from_slice(&bits[..k * w]);
                }
                sp.draft(&DraftInput { ctx_len: ctx0, tokens: &tokens, new_taps: &feed, hidden });
                let mut t = Vec::new();
                for i in 0..60 {
                    for _ in 0..n {
                        tokens.push(1000 + tokens.len() as u32 % 5000);
                    }
                    let new = &bits[(i % (rows - 8)) * w..(i % (rows - 8) + n) * w];
                    device.synchronize()?;
                    let t0 = Instant::now();
                    let nodes = sp.draft(&DraftInput { ctx_len: tokens.len() - 1, tokens: &tokens, new_taps: new, hidden });
                    device.synchronize()?;
                    if i >= 10 {
                        t.push(t0.elapsed().as_secs_f64() * 1e3);
                    }
                    anyhow::ensure!(!nodes.is_empty() && sp.last_error.is_none(), "round {i}: {:?}", sp.last_error);
                }
                t.sort_by(f64::total_cmp);
                let med = |v: &[f64]| {
                    let mut v = v.to_vec();
                    v.sort_by(f64::total_cmp);
                    if v.is_empty() { f64::NAN } else { v[v.len() / 2] * 1e3 }
                };
                if split {
                    // the last 50 rounds of each stage vector
                    let tail = |v: &Vec<f64>| v[v.len().saturating_sub(50)..].to_vec();
                    println!(
                        "loop round{tag} {shape:?} n {n}, synchronised between stages: total {:.2} ms; extend {:.2} (of which host conversion and upload {:.2}), block {:.2}, head {:.2}, readout {:.2} (ms, medians of 50)",
                        t[t.len() / 2], med(&tail(&sp.t_extend)), med(&tail(&sp.t_upload)), med(&tail(&sp.t_block)), med(&tail(&sp.t_head)), med(&tail(&sp.t_readout))
                    );
                } else {
                    println!("loop round{tag} {shape:?} n {n}: {:.2} ms [{:.2}-{:.2}], median of 50 after 10, one synchronisation", t[t.len() / 2], t[0], t[t.len() - 1]);
                }
            }
        }
    }
    Ok(())
}
