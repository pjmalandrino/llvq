//! The rotation kernel on a real card: does it decide what Rust decides, and
//! what does it cost?
//!
//! `tests/rotation_matches_rust.rs` already runs this kernel's *text* on the
//! development machine and pins every arithmetic decision it makes. Three
//! things it structurally cannot check, and they are this job's entire
//! purpose:
//!
//!   * **register pressure** — `rot_mix` holds a `LLVQ_ROT_KMAX`-wide array in
//!     registers, and that only works if the unrolled loops let it. A spill
//!     would not change a single result; it would quietly cost occupancy. The
//!     driver reports it, so it is an error here rather than a mystery later.
//!   * **the barriers** — at one thread a barrier is vacuous, so the host
//!     harness cannot tell a correct `__syncthreads()` from a missing one.
//!     With 1024 threads on a card, a missing one is a wrong answer.
//!   * **the cost** — 144 of these run per token in the fused decode. If one
//!     is not a few microseconds, the whole plan changes.
//!
//! ## Why the fixture is built here and not downloaded
//!
//! Same reasoning as `preflight`: a job is billed by the minute and the
//! rotation depends on nothing in the artifact. `Rotation::new(n, seed)` is
//! the same object the quantizer built, so a locally-drawn activation
//! exercises exactly the transform the sealed weights were quantized under.
//!
//! ## What "agrees" means
//!
//! The reference is f64, the kernel is f32, so the comparison is a threshold,
//! not an equality. Two numbers are reported per width because they fail
//! differently: the relative error in the **norm** catches a systematic fault
//! (a dropped scale, a transposed `Q_odd`), and the worst **coordinate**,
//! measured against a typical one, catches a single mis-addressed butterfly
//! that leaves the norm almost intact.
//!
//! The norm is also checked against the input's, which is the one property no
//! diff against Rust can establish: `Q` is orthogonal, so `‖Qx‖ = ‖x‖`, and a
//! shared misconception between the two implementations would satisfy the diff
//! while failing this.

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("rotbench targets NVIDIA GPUs; there is nothing to run here.");
}

#[cfg(target_os = "linux")]
fn main() -> Result<(), String> {
    linux::run()
}

#[cfg(target_os = "linux")]
mod linux {
    use llvq_core::SplitMix64;
    use llvq_cuda::gpu::{Cuda, KernelSource};
    use llvq_cuda::{f16_bits, f16_to_f64, ROT_KMAX};
    use llvq_quant::rotation::Rotation;
    use std::time::Instant;

    /// Thresholds. Ten times looser than what the host harness measures
    /// (9.5e-8 and 6.2e-7 over the same widths), which leaves room for a
    /// different reduction order without leaving room for a bug.
    const TOL_REL: f64 = 1e-5;
    const TOL_COORD: f64 = 1e-3;

    /// `(n, seed, what it is)`. The three widths of the published 4B first,
    /// then the shapes that exercise the branches around them.
    const CASES: [(usize, u64, &str); 8] = [
        (2560, 0x5, "Qwen3-4B hidden — q/k/v/gate/up"),
        (4096, 0x6, "Qwen3-4B o_proj — k=1, no mix"),
        (9728, 0x7, "Qwen3-4B down_proj — k=19, widest"),
        (12288, 0x8, "Qwen3-8B intermediate"),
        (3072, 0x4, "Qwen3-0.6B intermediate"),
        (1024, 0x3, "narrow, k=1"),
        (96, 0x2, "small"),
        (24, 0x1, "one Leech block"),
    ];

    const ROUNDS: usize = 200;
    const WARMUP: usize = 20;

    pub fn run() -> Result<(), String> {
        let sources = llvq_cuda::load_sources_named(&["llvq_rot.cuh", "rotate.cu"])?;
        let src = KernelSource::new(&[&sources.slot, &sources.cu]);
        println!("source NVRTC : {} octets, sha256 {}", src.text.len(), src.sha256);
        if let Some(d) = &sources.overridden_from {
            println!("  ⚠️ SOURCES SURCHARGÉES depuis {d}");
        }

        let cuda = Cuda::new(&src)?;
        let dev = cuda.device()?;
        println!(
            "\n{} — {} SM, {} o de partagée par bloc",
            dev.name, dev.sm_count, dev.shared_per_block
        );

        let rep = cuda.report("rot_apply")?;
        println!(
            "  {:<10} {:>3} registres, {} o locaux, sm_{}, {} threads max",
            rep.name, rep.num_regs, rep.local_bytes, rep.binary_version, rep.max_threads
        );
        // The one hardware fact the host harness cannot produce. `rot_mix`
        // keeps a 32-float column in registers only if both inner loops
        // unrolled; if they did not, this is where it shows.
        if rep.local_bytes != 0 {
            return Err(format!(
                "rot_apply : {} octets de spill — `col` n'est pas resté en registres",
                rep.local_bytes
            ));
        }
        let f = cuda.func("rot_apply")?;

        println!("\nVérification contre la référence f64");
        println!("  {:<34}{:>7}{:>5}{:>4}{:>12}{:>12}", "forme", "n", "m", "k", "rel", "pire coord");
        println!("  {}", "-".repeat(74));

        let mut worst_rel = 0.0f64;
        let mut worst_coord = 0.0f64;
        let mut timings: Vec<(usize, f64)> = Vec::new();

        for (n, seed, what) in CASES {
            let rot = Rotation::new(n, seed);
            let (m, k) = (rot.pow2(), rot.odd());
            if k > ROT_KMAX {
                return Err(format!("{what} : k={k} dépasse KMAX={ROT_KMAX}"));
            }
            if n * 4 > dev.shared_per_block as usize {
                return Err(format!(
                    "{what} : {} o de partagée demandés, la carte en offre {}",
                    n * 4,
                    dev.shared_per_block
                ));
            }

            // Drawn in f32, narrowed to f16, widened for the reference — so
            // the kernel is compared against the vector it was handed, not
            // against one it never saw.
            let mut rng = SplitMix64::new(seed ^ 0xa5a5_a5a5);
            let bits: Vec<u16> = (0..n).map(|_| f16_bits(rng.next_gaussian() as f32)).collect();
            let mut want: Vec<f64> = bits.iter().map(|&h| f16_to_f64(h)).collect();
            let norm_in = want.iter().map(|v| v * v).sum::<f64>().sqrt();
            rot.apply(&mut want);

            let mut words = vec![0u32; n.div_ceil(32)];
            for (i, &s) in rot.signs().iter().enumerate() {
                if s < 0.0 {
                    words[i >> 5] |= 1 << (i & 31);
                }
            }
            // Zero-padded: the kernel's inner loops run to KMAX and the pad is
            // what makes the extra terms contribute nothing.
            let mut small = vec![0.0f32; ROT_KMAX * ROT_KMAX];
            for g in 0..k {
                for t in 0..k {
                    small[g * ROT_KMAX + t] = rot.small()[g * k + t] as f32;
                }
            }
            let inv = (1.0f64 / (m as f64).sqrt()) as f32;

            let d_x = cuda.up_u16(&bits)?;
            let d_sign = cuda.up_u32(&words)?;
            let d_small = cuda.up_f32(&small)?;
            let mut d_out = cuda.zeros_f32(n)?;

            // 1024 is the widest block the driver allows and the one the fused
            // decode would use; anything narrower would leave the barriers
            // less exercised, which is what this job is here for.
            let threads = (n as u32).next_power_of_two().clamp(32, 1024);
            cuda.launch_rot(&f, &d_x, &d_sign, &d_small, &mut d_out,
                            n as u32, m as u32, k as u32, inv, 0, threads)?;
            cuda.sync()?;
            let got = cuda.down_f32(&d_out)?;

            let nrm = want.iter().map(|w| w * w).sum::<f64>().sqrt();
            let dev_norm = got
                .iter()
                .zip(&want)
                .map(|(&g, &w)| (g as f64 - w) * (g as f64 - w))
                .sum::<f64>()
                .sqrt();
            let coord = got
                .iter()
                .zip(&want)
                .map(|(&g, &w)| (g as f64 - w).abs())
                .fold(0.0, f64::max);
            let typical = nrm / (n as f64).sqrt();
            let (rel, cmax) = (dev_norm / nrm, coord / typical);

            println!("  {what:<34}{n:>7}{m:>5}{k:>4}{rel:>12.3e}{cmax:>12.3e}");
            if !(rel < TOL_REL && cmax < TOL_COORD) {
                return Err(format!("{what} : rel {rel:.3e}, pire coord {cmax:.3e} — hors seuil"));
            }
            worst_rel = worst_rel.max(rel);
            worst_coord = worst_coord.max(cmax);

            // Orthogonality, checked on the device's own output.
            let norm_out = got.iter().map(|&g| g as f64 * g as f64).sum::<f64>().sqrt();
            let dn = (norm_out - norm_in).abs() / norm_in;
            if dn > TOL_REL {
                return Err(format!("{what} : norme {norm_in:.9} → {norm_out:.9} ({dn:.3e})"));
            }

            for _ in 0..WARMUP {
                cuda.launch_rot(&f, &d_x, &d_sign, &d_small, &mut d_out,
                                n as u32, m as u32, k as u32, inv, 0, threads)?;
            }
            cuda.sync()?;
            let t = Instant::now();
            for _ in 0..ROUNDS {
                cuda.launch_rot(&f, &d_x, &d_sign, &d_small, &mut d_out,
                                n as u32, m as u32, k as u32, inv, 0, threads)?;
            }
            cuda.sync()?;
            timings.push((n, t.elapsed().as_secs_f64() * 1e6 / ROUNDS as f64));
        }

        println!("  {}", "-".repeat(74));
        println!("  pire sur {} formes : rel {worst_rel:.3e}, coord {worst_coord:.3e}", CASES.len());

        println!("\nCoût — {ROUNDS} lancements enchaînés, {WARMUP} jetés");
        println!("  ⚠️ un lancement à la fois, sur un stream vide : c'est la latence isolée,");
        println!("     pas ce que coûtera la même rotation intercalée entre deux matvecs.");
        println!("  {:>7}{:>12}", "n", "µs");
        println!("  {}", "-".repeat(19));
        for (n, us) in &timings {
            println!("  {n:>7}{us:>12.2}");
        }

        // What the decode would actually pay: four rotations per transformer
        // block, one per distinct activation, over 36 blocks.
        let per = |n: usize| timings.iter().find(|(w, _)| *w == n).map(|(_, u)| *u).unwrap_or(0.0);
        let token = 36.0 * (2.0 * per(2560) + per(4096) + per(9728));
        println!("\n  Projection Qwen3-4B : 36 blocs × (2×2560 + 4096 + 9728) = {token:.0} µs/token");
        println!("  À rapporter aux ~23 400 µs d'un token f16 mesuré sur L40S le 2026-08-05.");
        Ok(())
    }
}
