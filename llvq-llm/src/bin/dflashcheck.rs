//! The DFlash drafter against its MLX fixture, then the time of one round.
//!
//! ```text
//! dflashcheck <fixture.safetensors> [cpu|metal|cuda] [packed_dir] [drafter_dir] [f32|f16|bf16]
//! ```
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
    if !ok {
        std::process::exit(1);
    }
    Ok(())
}
