//! What a launch costs, and what a CUDA Graph gives back.
//!
//! The 2026-08-05 performance audit makes this figure a **publication
//! condition**: any kernel gain measured across 252 launches is diluted by the
//! constant term `252·g`, and `g` has to be known before any sweep.
//! `bin/matvec` already gave a bound, `t_submit` = 1.85 µs of CPU per launch,
//! 8% of the wall. But submission is not execution: the GPU also pays a
//! per-launch cost that the CPU never sees.
//!
//! ## Why a tiny kernel
//!
//! What is measured is the **launch**, not the traffic. A 64-row matrix
//! launches 8 blocks and reads a few kilobytes: the per-iteration time is then
//! almost entirely fixed cost, which is exactly the quantity wanted. On the
//! real shapes the work would mask what has to be isolated.
//!
//! The buffers are zeros. That is not a shortcut: a null `bases` gives a null
//! stride, so every block reads entry 0 of the table, the origin class, which
//! is perfectly valid (`len = 1`, null values). The kernel runs its full path
//! on legal data.
//!
//! ## The three arms, and why three are needed
//!
//! `Cuda::new` takes `ctx.default_stream()`, the legacy NULL stream, which the
//! driver **refuses to capture**. A graph arm therefore forces a fresh stream,
//! and changing the stream changes the object measured: the NULL stream
//! carries an implicit synchronization against every other stream of the
//! context, a fresh stream does not. Comparing "graph on a fresh stream" to a
//! figure published on the legacy stream would credit the graph with what the
//! stream change is worth. Hence legacy, fresh, fresh + graph, in one job.

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("graphbench targets NVIDIA GPUs; there is nothing to run here.");
}

#[cfg(target_os = "linux")]
fn main() -> Result<(), String> {
    linux::run()
}

#[cfg(target_os = "linux")]
mod linux {
    use llvq_cuda::gpu::{Cuda, KernelSource};
    use std::time::Instant;

    const TILE_BLOCKS: usize = 128;
    const THREADS: u32 = 256;
    const TABLE_ENTRIES: usize = 512;
    const REC_WORDS: usize = 6;
    /// The 252 projections of one Qwen3-4B token, the count that gives `252·g`
    /// its meaning.
    const LAUNCHES: usize = 252;
    const ROUNDS: usize = 20;
    const WARMUP: usize = 5;
    /// 64 rows = 8 blocks: enough for the grid to be legal, few enough that
    /// the work does not mask the launch.
    const D_OUT: usize = 64;
    const D_IN: usize = 2560;

    fn spread(mut v: Vec<f64>) -> (f64, f64, f64) {
        v.sort_by(f64::total_cmp);
        (v[0], v[v.len() / 2], v[v.len() - 1])
    }

    pub fn run() -> Result<(), String> {
        let sources =
            llvq_cuda::load_sources_many(&["llvq_slot.cuh", "matvec.cu", "llvq_floor.cuh"])?;
        let defines = format!("#define TILE_BLOCKS {TILE_BLOCKS}u\n");
        let parts: Vec<&str> = std::iter::once(defines.as_str())
            .chain(sources.parts.iter().map(String::as_str))
            .collect();
        let src = KernelSource::new(&parts);
        println!("NVRTC source: {} bytes, sha256 {}", src.text.len(), src.sha256);

        let nblocks = D_IN / 24;
        let mut results: Vec<(&str, f64, f64, f64)> = Vec::new();

        for (label, fresh) in [("legacy stream (default)", false), ("fresh stream", true)] {
            let cuda = if fresh {
                Cuda::new_on_fresh_stream(&src)?
            } else {
                Cuda::new(&src)?
            };
            let dev = cuda.device()?;
            if !fresh {
                println!("\n{} · {} SM", dev.name, dev.sm_count);
            }
            let f = cuda.func("tv_floor_stream")?;

            let words = cuda.zeros_u32(1 << 16)?;
            let bases = cuda.zeros_u32(D_OUT * nblocks / 32 + 2)?;
            let tab = cuda.up_u32(&vec![1u32; TABLE_ENTRIES * REC_WORDS])?;
            let x = cuda.zeros_f32(D_IN)?;
            let mut y = cuda.zeros_f32(D_OUT)?;
            let shared = (TILE_BLOCKS * 24 * 4) as u32;

            let one = |y: &mut cudarc::driver::CudaSlice<f32>| -> Result<(), String> {
                cuda.launch_floor(
                    &f, &words, &bases, &tab, &x, y, nblocks as u32, D_OUT as u32, THREADS,
                    shared,
                )
            };

            // --- without graph ---
            let mut us = Vec::new();
            for r in 0..ROUNDS {
                let t = Instant::now();
                for _ in 0..LAUNCHES {
                    one(&mut y)?;
                }
                cuda.sync()?;
                if r >= WARMUP {
                    us.push(t.elapsed().as_secs_f64() * 1e6 / LAUNCHES as f64);
                }
            }
            let (lo, md, hi) = spread(us);
            println!("  {label:<26} {md:7.2} µs/launch  [{lo:.2}–{hi:.2}]");
            results.push((if fresh { "frais" } else { "legacy" }, lo, md, hi));

            // --- with graph, only where the driver accepts it ---
            if fresh {
                let graph = cuda.capture(|| {
                    for _ in 0..LAUNCHES {
                        one(&mut y)?;
                    }
                    Ok(())
                })?;
                let mut us = Vec::new();
                for r in 0..ROUNDS {
                    let t = Instant::now();
                    graph.launch().map_err(|e| format!("graph launch: {e}"))?;
                    cuda.sync()?;
                    if r >= WARMUP {
                        us.push(t.elapsed().as_secs_f64() * 1e6 / LAUNCHES as f64);
                    }
                }
                let (lo, md, hi) = spread(us);
                println!("  {:<26} {md:7.2} µs/launch  [{lo:.2}–{hi:.2}]", "fresh + graph");
                results.push(("graph", lo, md, hi));
            }
        }

        // --- what it is worth on one token ---
        let get = |k: &str| results.iter().find(|(n, ..)| *n == k).map(|(_, _, m, _)| *m);
        if let (Some(legacy), Some(fresh), Some(graph)) = (get("legacy"), get("frais"), get("graph"))
        {
            println!("\n  Over {LAUNCHES} launches, what one token of projections is worth:");
            println!("    legacy         {:6.3} ms", legacy * LAUNCHES as f64 / 1e3);
            println!("    fresh          {:6.3} ms", fresh * LAUNCHES as f64 / 1e3);
            println!("    fresh + graph  {:6.3} ms", graph * LAUNCHES as f64 / 1e3);
            println!(
                "\n  The graph gives back {:.3} ms against the fresh stream, {:.3} against legacy.",
                (fresh - graph) * LAUNCHES as f64 / 1e3,
                (legacy - graph) * LAUNCHES as f64 / 1e3
            );
            println!(
                "  WARNING: this is a CEILING, not a gain in hand. This kernel does almost\n  \
                 nothing, so its time is almost all launch. On `tv_slot`, whose work masks\n  \
                 part of the fixed cost, the real saving is smaller. Full decode also has a\n  \
                 KV cache of varying shapes that a static graph does not capture\n  \
                 (see A3(b) of the lot A spec)."
            );
        }
        Ok(())
    }
}
