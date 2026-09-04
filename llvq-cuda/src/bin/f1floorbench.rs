//! The F1 decoder-table floor: what the lookups cost, alone, before anyone
//! writes `tv_l3e8`.
//!
//! Lead F1 stops unfolding the served index, and its quality is measured and
//! fine: 89.55% Gaussian retention against a ball-12 control at 92.00% in one
//! process, −2.45 pp (*measured*, `docs/mesures/f1b-*`). The risk is entirely
//! in the decoder, because decoding becomes `label → point` through a table
//! that does not fit in shared memory — 3,336 KiB against the card's 101,376 B
//! opt-in, 34× over (*computed*, `llvq-bench/examples/f1table.rs`) — and a model
//! pass is **454 M lookups**, 14.5 GB of table reads against 0.98 GB of weight
//! reads.
//!
//! Three format leads died on decode cost and none on quality. E1v had better
//! bytes than F1, 2.3877 b/weight, and measured 0.25× FP16. So this bench asks
//! one question for ~$0.09 instead of a week: **does the L2 absorb them?**
//!
//! ## The ladder, and why no arm is read alone
//!
//! ```text
//!   Didx  = t(hash3) − t(nullk)     manufacturing the indices, which a real F1
//!                                   kernel gets FREE from the weight word this
//!                                   bench does not read
//!   D(S)  = t(tab3, S) − t(hash3)   THE TABLE ALONE, at footprint S
//!   Dsm   = t(smem) − t(smem_fill)  the lookups when the hot set is PLACED in
//!                                   shared memory, at that occupancy
//! ```
//!
//! ⚠️ `Dsm` may not be formed against `nullk`. A 60 KiB/block arm and a
//! 12 KiB/block one do not have the same occupancy, and differencing across
//! that is what `docs/format-noyau.md` §6 forbids — the same prohibition that
//! struck out F1d's first threshold on 2026-09-04. Hence the matched anchor.
//!
//! ## Why the footprint is swept and the access is not modelled
//!
//! 🚨 A uniform draw over the whole table is **pessimistic**, not optimistic.
//! Of the 67 end orbits, two hold 256 of the 512 regions — half the end
//! lookups, a third of all lookups, inside 48 KiB (*computed*, f1table.rs,
//! sizes `[128, 128, 4, 4, …]`). A uniform draw destroys a hot set the hardware
//! would cache for free, so a kill read off it would be a kill on the mixer.
//! Uniform access over `S` bytes is monotone in `S`, so the sweep brackets the
//! truth from both sides and needs no distribution anyone has to invent.
//!
//! ## What this bench cannot be
//!
//! Verified against an f64 reference: like `nullk` and like `sol` in
//! `bin/rankbench` it computes no product of the model. What is asked of it is
//! to be OBSERVABLE. The post-run check demands that every output row was
//! written, is finite, and — the part that catches a deleted load — that the
//! table arms disagree with `nullk`'s output.
//!
//! And it is a floor, not a cost: with no weight stream competing for L1 and
//! the LSU, and with the labels not arriving from that stream, the regime that
//! decides F1 is not reproduced here. That is F1d's job. This says whether F1d
//! is worth writing.

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("f1floorbench targets NVIDIA GPUs; there is nothing to run here.");
}

#[cfg(target_os = "linux")]
fn main() -> Result<(), String> {
    linux::run()
}

#[cfg(target_os = "linux")]
mod linux {
    use cudarc::driver::PushKernelArg;
    use llvq_core::{SplitMix64, DIM};
    use llvq_cuda::gpu::{Cuda, KernelSource};
    use llvq_cuda::TILE_BLOCKS;
    use std::time::Instant;

    const ROUNDS: usize = 9;
    const WARMUP: usize = 2;
    const THREADS: u32 = 256;
    const LAYERS: usize = 36;

    /// The seven projection shapes of Qwen3-4B — `planesbench`'s table and
    /// `nullkbench`'s, so 252 launches a round in the geometry every published
    /// floor uses.
    const SHAPES: [(&str, usize, usize); 7] = [
        ("q_proj", 4096, 2560),
        ("k_proj", 1024, 2560),
        ("v_proj", 1024, 2560),
        ("o_proj", 2560, 4096),
        ("gate_proj", 9728, 2560),
        ("up_proj", 9728, 2560),
        ("down_proj", 2560, 9728),
    ];

    /// Table footprints, in u32 entries. Powers of two so the address is one
    /// AND and the mixer is the only thing deciding where a lookup lands.
    /// The two figures F1 actually carries — 80 KiB for the odd-coset variant
    /// and 3,336 KiB for the full table — fall between measured points, which
    /// is what a bracket is for. The last point is the DRAM calibration: it
    /// must be at least 20× the L2 or a "miss" is a third hit.
    const FOOTPRINTS: [(&str, usize); 8] = [
        ("8 KiB", 2 * 1024),
        ("16 KiB", 4 * 1024),
        ("128 KiB", 32 * 1024),
        ("1 MiB", 256 * 1024),
        ("4 MiB", 1024 * 1024),
        ("16 MiB", 4 * 1024 * 1024),
        ("64 MiB", 16 * 1024 * 1024),
        ("1 GiB", 256 * 1024 * 1024),
    ];

    /// Hot sets placed in shared memory, in u32 entries, with the share of
    /// lookups routed there in 1/256ths. 24 KiB covers a quarter of end lookups
    /// (17% of all) at 2 blocks/SM; 48 KiB covers half (33%) at 1 block/SM,
    /// against 8 blocks/SM today (*computed*, f1table.rs).
    const HOT: [(&str, usize, u32); 2] = [("24 KiB", 6 * 1024, 43), ("48 KiB", 12 * 1024, 85)];

    struct Shape {
        name: &'static str,
        d_out: u32,
        nblocks: u32,
        tail_w: u32,
        rscale: cudarc::driver::CudaSlice<f32>,
        tail: cudarc::driver::CudaSlice<f32>,
        x: cudarc::driver::CudaSlice<f32>,
        y: cudarc::driver::CudaSlice<f32>,
    }

    fn build(cuda: &Cuda) -> Result<Vec<Shape>, String> {
        let mut rng = SplitMix64::new(0x00F1_2026_0905);
        let mut out = Vec::new();
        for &(name, d_out, d_in) in SHAPES.iter() {
            assert_eq!(d_out as u32 % (THREADS / 32), 0, "{name}: rows must fill whole blocks");
            let mut f = |n: usize| -> Vec<f32> {
                (0..n).map(|_| 0.5 + (rng.next() >> 40) as f32 / 16_777_216.0).collect()
            };
            let tail_w = (d_in % DIM) as u32;
            let x = f(d_in);
            let tail = f(d_out * tail_w as usize);
            let rscale = f(d_out);
            out.push(Shape {
                name,
                d_out: d_out as u32,
                nblocks: (d_in / DIM) as u32,
                tail_w,
                rscale: cuda.up_f32(&rscale)?,
                tail: cuda.up_f32(&tail)?,
                x: cuda.up_f32(&x)?,
                y: cuda.zeros_f32(d_out)?,
            });
        }
        Ok(out)
    }

    fn cfg(s: &Shape, shared: u32) -> cudarc::driver::LaunchConfig {
        cudarc::driver::LaunchConfig {
            grid_dim: (s.d_out * 32 / THREADS, 1, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: shared,
        }
    }

    /// One timed round over the 252 launches, for an arm with no table.
    fn round_plain(
        cuda: &Cuda,
        f: &cudarc::driver::CudaFunction,
        shapes: &mut [Shape],
        shared: u32,
    ) -> Result<f64, String> {
        let t = Instant::now();
        for _ in 0..LAYERS {
            for s in shapes.iter_mut() {
                let c = cfg(s, shared);
                let mut b = cuda.stream().launch_builder(f);
                b.arg(&s.rscale).arg(&s.tail).arg(&s.x).arg(&mut s.y).arg(&s.nblocks).arg(&s.tail_w);
                unsafe { b.launch(c) }.map_err(|e| format!("{}: {e}", s.name))?;
            }
        }
        cuda.sync()?;
        Ok(t.elapsed().as_secs_f64() * 1e3)
    }

    /// One timed round for `tv_f1_tab3`.
    fn round_tab(
        cuda: &Cuda,
        f: &cudarc::driver::CudaFunction,
        shapes: &mut [Shape],
        shared: u32,
        tab: &cudarc::driver::CudaSlice<u32>,
        mask: u32,
    ) -> Result<f64, String> {
        let t = Instant::now();
        for _ in 0..LAYERS {
            for s in shapes.iter_mut() {
                let c = cfg(s, shared);
                let mut b = cuda.stream().launch_builder(f);
                b.arg(&s.rscale)
                    .arg(&s.tail)
                    .arg(&s.x)
                    .arg(&mut s.y)
                    .arg(tab)
                    .arg(&mask)
                    .arg(&s.nblocks)
                    .arg(&s.tail_w);
                unsafe { b.launch(c) }.map_err(|e| format!("{}: {e}", s.name))?;
            }
        }
        cuda.sync()?;
        Ok(t.elapsed().as_secs_f64() * 1e3)
    }

    /// One timed round for the shared-memory pair.
    #[allow(clippy::too_many_arguments)]
    fn round_smem(
        cuda: &Cuda,
        f: &cudarc::driver::CudaFunction,
        shapes: &mut [Shape],
        shared: u32,
        tab: &cudarc::driver::CudaSlice<u32>,
        mask: u32,
        hot_words: u32,
        hot_frac: u32,
    ) -> Result<f64, String> {
        let t = Instant::now();
        for _ in 0..LAYERS {
            for s in shapes.iter_mut() {
                let c = cfg(s, shared);
                let mut b = cuda.stream().launch_builder(f);
                b.arg(&s.rscale)
                    .arg(&s.tail)
                    .arg(&s.x)
                    .arg(&mut s.y)
                    .arg(tab)
                    .arg(&mask)
                    .arg(&hot_words)
                    .arg(&hot_frac)
                    .arg(&s.nblocks)
                    .arg(&s.tail_w);
                unsafe { b.launch(c) }.map_err(|e| format!("{}: {e}", s.name))?;
            }
        }
        cuda.sync()?;
        Ok(t.elapsed().as_secs_f64() * 1e3)
    }

    fn median_range(v: &[f64]) -> (f64, f64, f64) {
        let mut s = v.to_vec();
        s.sort_by(f64::total_cmp);
        (s[s.len() / 2], s[0], s[s.len() - 1])
    }

    /// Every output row written, finite, and — for a table arm — different from
    /// the floor's. The last part is what catches an elided load: a compiler
    /// that deleted the three fetches would leave the fold constant and the
    /// output would match `nullk`'s to the bit.
    fn observable(cuda: &Cuda, shapes: &[Shape], floor: Option<&[Vec<f32>]>, who: &str) -> Result<Vec<Vec<f32>>, String> {
        let mut out = Vec::new();
        for (i, s) in shapes.iter().enumerate() {
            let y = cuda.down_f32(&s.y)?;
            if y.iter().any(|v| !v.is_finite()) || y.iter().all(|v| *v == 0.0) {
                return Err(format!("{who}/{}: output not observable", s.name));
            }
            if let Some(f) = floor {
                if y == f[i] {
                    return Err(format!(
                        "{who}/{}: output identical to the floor's — the lookups were elided",
                        s.name
                    ));
                }
            }
            out.push(y);
        }
        Ok(out)
    }

    pub fn run() -> Result<(), String> {
        let base = llvq_cuda::load_sources_many(&["llvq_slot.cuh", "matvec.cu", "f1floor.cu", "nullk.cu"])?;
        let defines = format!("#define TILE_BLOCKS {TILE_BLOCKS}u\n");
        let mut parts: Vec<&str> = vec![defines.as_str()];
        parts.extend(base.parts.iter().map(String::as_str));
        let src = KernelSource::new(&parts);
        println!("F1 table floor — the lookups alone, 252 launches, one process");
        println!("NVRTC source: {} bytes, sha256 {}", src.text.len(), src.sha256);
        let cuda = Cuda::new(&src)?;
        let dev = cuda.device()?;
        println!("card: {} · {} SM · shared/block {} default, {} opt-in · L2 {:.0} MiB",
            dev.name, dev.sm_count, dev.shared_per_block, dev.shared_per_block_optin,
            dev.l2_bytes as f64 / (1024.0 * 1024.0));

        let tile = (TILE_BLOCKS * DIM * 4) as u32;
        let mut shapes = build(&cuda)?;

        // The tables. Contents are never read for meaning, only folded.
        let mut rng = SplitMix64::new(0x00F1_2026_0905);
        let biggest = FOOTPRINTS.iter().map(|&(_, n)| n).max().expect("non-empty");
        let words: Vec<u32> = (0..biggest).map(|_| rng.next() as u32).collect();
        let mut tables = Vec::new();
        for &(name, n) in FOOTPRINTS.iter() {
            assert!(n.is_power_of_two(), "{name}: the mask needs a power of two");
            tables.push((name, n, cuda.up_u32(&words[..n])?));
        }
        let dram = FOOTPRINTS.last().expect("non-empty").1 * 4;
        if (dram as u64) < 20 * dev.l2_bytes as u64 {
            return Err(format!(
                "the DRAM point is {dram} B against an L2 of {} B; below 20× it is a hit rate, not a miss rate",
                dev.l2_bytes
            ));
        }

        let f_null = cuda.func("tv_nullk")?;
        let f_hash = cuda.func("tv_f1_hash3")?;
        let f_tab = cuda.func("tv_f1_tab3")?;

        // Interleaved rounds: every arm every round, fixed order, warmup
        // discarded, differences formed ROUND BY ROUND and never as a quotient
        // of minima.
        let n_arms = 2 + tables.len() + HOT.len() * 2;
        let mut times: Vec<Vec<f64>> = vec![Vec::new(); n_arms];
        let mut floor_y: Option<Vec<Vec<f32>>> = None;
        for rep in 0..ROUNDS {
            let mut k = 0usize;
            let t = round_plain(&cuda, &f_null, &mut shapes, tile)?;
            if rep == ROUNDS - 1 {
                floor_y = Some(observable(&cuda, &shapes, None, "nullk")?);
            }
            if rep >= WARMUP { times[k].push(t); }
            k += 1;

            let t = round_plain(&cuda, &f_hash, &mut shapes, tile)?;
            if rep >= WARMUP { times[k].push(t); }
            k += 1;

            for (_, n, tab) in tables.iter() {
                let t = round_tab(&cuda, &f_tab, &mut shapes, tile, tab, (*n - 1) as u32)?;
                if rep >= WARMUP { times[k].push(t); }
                k += 1;
            }

            for &(hname, hwords, hfrac) in HOT.iter() {
                let shared = tile + (hwords * 4) as u32;
                if shared > dev.shared_per_block_optin as u32 {
                    return Err(format!("{hname}: {shared} B of shared, the card allows {}", dev.shared_per_block_optin));
                }
                let f_sm = cuda.func_dynamic_shared("tv_f1_smem", shared)?;
                let f_fl = cuda.func_dynamic_shared("tv_f1_smem_fill", shared)?;
                let (_, n, tab) = &tables[4]; // the 4 MiB point, which brackets F1's 3,336 KiB
                let t = round_smem(&cuda, &f_sm, &mut shapes, shared, tab, (*n - 1) as u32, hwords as u32, hfrac)?;
                if rep >= WARMUP { times[k].push(t); }
                k += 1;
                let t = round_smem(&cuda, &f_fl, &mut shapes, shared, tab, (*n - 1) as u32, hwords as u32, hfrac)?;
                if rep >= WARMUP { times[k].push(t); }
                k += 1;
            }
        }

        // Elision check on the last round's outputs.
        let floor_y = floor_y.expect("the last round ran");
        round_tab(&cuda, &f_tab, &mut shapes, tile, &tables[4].2, (tables[4].1 - 1) as u32)?;
        observable(&cuda, &shapes, Some(floor_y.as_slice()), "tab3")?;

        let mut names: Vec<String> = vec!["nullk".into(), "hash3".into()];
        names.extend(tables.iter().map(|&(n, _, _)| format!("tab3 {n}")));
        for &(hname, _, hfrac) in HOT.iter() {
            names.push(format!("smem {hname} @{}%", hfrac * 100 / 256));
            names.push(format!("smem-fill {hname}"));
        }

        println!("\n  {ROUNDS} rounds, {WARMUP} discarded; every difference formed ROUND BY ROUND\n");
        for (i, name) in names.iter().enumerate() {
            let (m, lo, hi) = median_range(&times[i]);
            println!("  {name:<22} {m:8.3} ms  [{lo:.3}–{hi:.3}]");
        }

        let didx: Vec<f64> = times[1].iter().zip(&times[0]).map(|(a, b)| a - b).collect();
        let (m, lo, hi) = median_range(&didx);
        println!("\n  Didx = hash3 − nullk      {m:8.3} ms  [{lo:.3}–{hi:.3}]   the index source, which F1 gets free");
        println!("\n  D(S) = tab3(S) − hash3    THE TABLE ALONE");
        for (i, &(fname, n)) in FOOTPRINTS.iter().enumerate() {
            let d: Vec<f64> = times[2 + i].iter().zip(&times[1]).map(|(a, b)| a - b).collect();
            let (m, lo, hi) = median_range(&d);
            let hit = (dev.l2_bytes as f64 / (n as f64 * 4.0)).min(1.0) * 100.0;
            println!("    {fname:<9} {m:8.3} ms  [{lo:.3}–{hi:.3}]   L2 modélisée {hit:5.1} %");
        }
        println!("\n  Dsm = smem − smem-fill    THE LOOKUPS WITH THE HOT SET PLACED");
        println!("  ⚠️ never against nullk: another occupancy, §6 of format-noyau forbids it");
        for (i, &(hname, hwords, _)) in HOT.iter().enumerate() {
            let a = &times[2 + tables.len() + i * 2];
            let b = &times[2 + tables.len() + i * 2 + 1];
            let d: Vec<f64> = a.iter().zip(b).map(|(x, y)| x - y).collect();
            let (m, lo, hi) = median_range(&d);
            let per_block = tile as usize + hwords * 4;
            let blocks = dev.shared_per_sm as usize / per_block.max(1);
            println!("    {hname:<9} {m:8.3} ms  [{lo:.3}–{hi:.3}]   {per_block} B/bloc → {blocks} bloc(s)/SM");
        }
        println!("\n  WARNING: a FLOOR, not a cost. No weight stream competes for L1 or the LSU here,");
        println!("  and the labels do not arrive from it. That regime is F1d's; this says whether");
        println!("  F1d is worth writing. And no time here compares to a time from ANOTHER process.");
        Ok(())
    }
}
