//! A1 — the floor under two launch geometries: 252 launches against 144.
//!
//! Preregistered: `proofs/preregistration-vague2-gel-geometrie-2026-08-31.md`
//! §A1 (sha256 `e23e9895…`, stamped before this file was written). The
//! question: does `nullk`'s 2.306 ms floor follow the launch count when the
//! q+k+v and gate+up projections are row-concatenated (252 → 144), or does it
//! stay put? The reading, fixed in the prereg before any number exists:
//!
//!   r = t(nullk, fused shapes) ÷ t(nullk, unfused shapes)
//!   r ≤ 0.65 → the post is per-launch latency  → A2 (CUDA Graphs) first
//!   r ≥ 0.90 → the post is occupancy           → A3 first
//!   between  → mixed; publish the parts, eliminate neither
//!
//! A prior worth disclosing: `nullk.cu`'s own header notes "108 fewer
//! launches are worth 0.392 ms measured elsewhere" — that would put r near
//! 0.83, in the prereg's *mixed* band. The bench exists to measure, not to
//! confirm that comment.
//!
//! ## Why a dedicated bin instead of a new `planesbench` arm
//!
//! `nullk`'s own doc says it: this arm *measures*, it does not candidate.
//! Inserting an arm into `planesbench` would perturb the object its published
//! tables measure (its header forbids mid-list insertion for exactly that
//! reason), and would drag the byte-accounting report into shapes no layout
//! ever serves. The protocol here is `planesbench`'s, reproduced rather than
//! referenced: interleaved rounds, warmup discarded, ratios formed **round by
//! round** and reported as median with range — never a quotient of minima.
//!
//! ## What this bin cannot be
//!
//! Verified against an f64 reference. Like `nullk` itself it computes no
//! model product and has no oracle — it is an anchor, required to be
//! *observable*, not correct. The post-run check only demands that every row
//! of every output was actually written (finite, and the kernel's staged
//! accumulation is load-bearing by construction — see `nullk.cu`).

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("nullkbench targets NVIDIA GPUs; there is nothing to run here.");
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

    const ROUNDS: usize = 7;
    const WARMUP: usize = 2;
    const THREADS: u32 = 256;
    const LAYERS: usize = 36;

    /// The seven projection shapes of Qwen3-4B — `planesbench`'s table.
    const UNFUSED: [(&str, usize, usize); 7] = [
        ("q_proj", 4096, 2560),
        ("k_proj", 1024, 2560),
        ("v_proj", 1024, 2560),
        ("o_proj", 2560, 4096),
        ("gate_proj", 9728, 2560),
        ("up_proj", 9728, 2560),
        ("down_proj", 2560, 9728),
    ];

    /// D1's fusion, applied to the same table: q+k+v and gate+up concatenated
    /// **by rows** (they share their input), o and down untouched. 4 sites ×
    /// 36 layers = 144 launches per round, against 7 × 36 = 252.
    const FUSED: [(&str, usize, usize); 4] = [
        ("qkv", 4096 + 1024 + 1024, 2560),
        ("o_proj", 2560, 4096),
        ("gate_up", 9728 + 9728, 2560),
        ("down_proj", 2560, 9728),
    ];

    /// One shape's device buffers. `nullk` reads no weights, so a shape is
    /// fully described by its activation, its row scales and its tail — the
    /// three things the kernel actually touches.
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

    fn build(cuda: &Cuda, list: &[(&'static str, usize, usize)]) -> Result<Vec<Shape>, String> {
        let mut rng = SplitMix64::new(0xA1_2026_08_31);
        let mut out = Vec::new();
        for &(name, d_out, d_in) in list {
            assert_eq!(
                d_out as u32 % (THREADS / 32),
                0,
                "{name}: rows must fill whole blocks"
            );
            let nblocks = (d_in / DIM) as u32;
            let tail_w = (d_in % DIM) as u32;
            // Deterministic, denormal-free values in [0.5, 1.5): the numbers
            // are never checked, but a NaN or a denormal would change timing.
            let mut f = |n: usize| -> Vec<f32> {
                (0..n).map(|_| 0.5 + (rng.next() >> 40) as f32 / 16_777_216.0).collect()
            };
            let x = f(d_in);
            let tail = f(d_out * tail_w as usize);
            let rscale = f(d_out);
            out.push(Shape {
                name,
                d_out: d_out as u32,
                nblocks,
                tail_w,
                rscale: cuda.up_f32(&rscale)?,
                tail: cuda.up_f32(&tail)?,
                x: cuda.up_f32(&x)?,
                y: cuda.zeros_f32(d_out)?,
            });
        }
        Ok(out)
    }

    fn launch(
        cuda: &Cuda,
        f: &cudarc::driver::CudaFunction,
        s: &mut Shape,
        shared: u32,
    ) -> Result<(), String> {
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (s.d_out * 32 / THREADS, 1, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: shared,
        };
        let mut b = cuda.stream().launch_builder(f);
        b.arg(&s.rscale)
            .arg(&s.tail)
            .arg(&s.x)
            .arg(&mut s.y)
            .arg(&s.nblocks)
            .arg(&s.tail_w);
        unsafe { b.launch(cfg) }.map_err(|e| format!("tv_nullk({}): {e}", s.name))?;
        Ok(())
    }

    fn median_range(v: &[f64]) -> (f64, f64, f64) {
        let mut s = v.to_vec();
        s.sort_by(f64::total_cmp);
        (s[s.len() / 2], s[0], s[s.len() - 1])
    }

    pub fn run() -> Result<(), String> {
        let matvec = llvq_cuda::load_sources_many(&["matvec.cu"])?;
        let nullk = llvq_cuda::load_sources_many(&["nullk.cu"])?;
        let defines = format!("#define TILE_BLOCKS {TILE_BLOCKS}u\n");
        let src = KernelSource::new(&[
            defines.as_str(),
            matvec.parts[0].as_str(),
            nullk.parts[0].as_str(),
        ]);
        println!("A1 — nullk : 252 lancements contre 144 (préreg e23e9895…, §A1)");
        println!("source NVRTC : {} octets, sha256 {}", src.text.len(), src.sha256);
        let cuda = Cuda::new(&src)?;
        let dev = cuda.device()?;
        println!("carte : {} · {} SM", dev.name, dev.sm_count);

        let f = cuda.func("tv_nullk")?;
        let shared = (TILE_BLOCKS * DIM * 4) as u32;
        let mut unfused = build(&cuda, &UNFUSED)?;
        let mut fused = build(&cuda, &FUSED)?;
        println!(
            "géométries : {} formes × {LAYERS} couches = {} lancements/round, contre {} × {LAYERS} = {}",
            UNFUSED.len(),
            UNFUSED.len() * LAYERS,
            FUSED.len(),
            FUSED.len() * LAYERS,
        );

        // Interleaved rounds, both arms every round, fixed order; warmup
        // discarded; the ratio is formed round by round.
        let (mut t252, mut t144, mut ratios) = (Vec::new(), Vec::new(), Vec::new());
        for rep in 0..ROUNDS {
            let t = Instant::now();
            for _ in 0..LAYERS {
                for s in unfused.iter_mut() {
                    launch(&cuda, &f, s, shared)?;
                }
            }
            cuda.sync()?;
            let a = t.elapsed().as_secs_f64() * 1e3;

            let t = Instant::now();
            for _ in 0..LAYERS {
                for s in fused.iter_mut() {
                    launch(&cuda, &f, s, shared)?;
                }
            }
            cuda.sync()?;
            let b = t.elapsed().as_secs_f64() * 1e3;

            if rep >= WARMUP {
                t252.push(a);
                t144.push(b);
                ratios.push(b / a);
            }
        }

        // Observability: every output row of both lists was written and is
        // finite. Not an oracle — the anchor's only obligation.
        for s in unfused.iter().chain(fused.iter()) {
            let y = cuda.down_f32(&s.y)?;
            let bad = y.iter().filter(|v| !v.is_finite()).count();
            if bad > 0 || y.iter().all(|v| *v == 0.0) {
                return Err(format!(
                    "{} : sortie non observable ({bad} non-finis, tout-zéro {})",
                    s.name,
                    y.iter().all(|v| *v == 0.0)
                ));
            }
        }

        let (m252, lo252, hi252) = median_range(&t252);
        let (m144, lo144, hi144) = median_range(&t144);
        let (mr, lor, hir) = median_range(&ratios);
        println!("\n  {ROUNDS} rounds, {WARMUP} jetés ; le rapport est formé ROUND PAR ROUND");
        println!("  nullk 252   méd {m252:.3} ms  [{lo252:.3}–{hi252:.3}]");
        println!("  nullk 144   méd {m144:.3} ms  [{lo144:.3}–{hi144:.3}]");
        println!("  r = t(144)/t(252) = {mr:.4} [{lor:.4}–{hir:.4}]");
        println!("\n  lecture pré-enregistrée (préreg §A1, AVANT ce chiffre) :");
        println!("    r ≤ 0,65 → latence par lancement → A2 (Graphs) prioritaire");
        println!("    r ≥ 0,90 → occupation → A3 prioritaire");
        println!("    entre    → mixte : publier les parts, n'éliminer ni A2 ni A3");
        println!("\n  ⚠️ aucun de ces temps ne se compare à un temps d'un AUTRE processus.");
        Ok(())
    }
}
