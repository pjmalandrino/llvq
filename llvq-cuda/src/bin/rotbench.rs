//! The rotation kernels on a real card: do they return the host's bytes, and
//! what do they cost?
//!
//! `tests/rotation_matches_rust.rs` runs the kernel *text* on the development
//! machine at every width below and writes the sha256 of `rot_apply`'s output
//! to `tests/fixtures/rot-hashes.txt`. This binary compiles that file in and
//! compares the card's bytes against it. Three things the host cannot check,
//! and they are this job's purpose:
//!
//!   * **register pressure**: `rot_mix` holds a `LLVQ_ROT_KMAX`-wide array in
//!     registers. A spill changes no result and costs occupancy; the driver
//!     reports it, so it is an error here.
//!   * **the barriers**: at one thread a barrier is vacuous, so the host cannot
//!     tell a correct `__syncthreads()` from a missing one. With 1,024 threads
//!     on a card, a missing one is a wrong byte, and the hash sees it.
//!   * **the cost**, in isolation, on an empty stream.
//!
//! ## The kernels and their handles
//!
//! `rot_apply` (K0, `rotate.cu`) stages the vector in shared memory; it runs
//! where `llvq_cuda::wide::rot_kernel` says `Shared`. `rot_apply_dev` (R0,
//! `rotate_wide.cu`) stages it in device memory and runs at every width, so
//! the two are compared on the same input wherever both run, and R0 alone
//! covers the widths past the card's ceiling (25,600 and 31,744 on a 99 KiB
//! card).
//!
//! The opt-in belongs to the loaded function and may move its carveout. The
//! served 4B poses none (its widest rotation, 38,912 B, fits the default) and
//! the served 14B poses 69,632 B on every width. So K0 is loaded from two
//! modules of the same text: **handle A**, no attribute, for every width that
//! fits the default; **handle B**, opted in at 69,632 B, for 5,120 and 17,408.
//! A width past the default that no handle B covers gets **handle C**, its
//! own module opted in at `n * 4` (only on cards with a larger opt-in, such as
//! the A100 at 25,600). R0 asks for no shared memory and has one handle.
//!
//! ## The fixture is compiled in
//!
//! The runtime image holds binaries, `fetch-qtip.sh` and `configs/` only, so a
//! path read at run time would not exist on the card. `llvq_cuda::wide`
//! `include_str!`s it, and a missing file fails the free image build.
//!
//! ## What "agrees" means
//!
//! First the input: its sha256 must equal the fixture's, or the card drew
//! another vector than the host and no output comparison is read (the binary
//! stops). Then the output, byte for byte. Then, as a second net, today's f64
//! tolerances and the norm. The rows twins run at 1, 3 and 4 rows, rows
//! `n + 8` apart and distinct; each row must equal the single-row kernel read
//! at that row's offset, and row 0 the fixture.

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
    use cudarc::driver::{CudaFunction, CudaSlice};
    use llvq_core::SplitMix64;
    use llvq_cuda::gpu::{Cuda, KernelSource};
    use llvq_cuda::sha256::sha256_hex;
    use llvq_cuda::shared::{rot_bytes, Fit};
    use llvq_cuda::wide::{
        f32_le_bytes, parse_rot_fixture, rot_input_bytes, rot_kernel, rot_scratch_floats,
        RotKernel, CARD_WIDTHS, ROT_FIXTURE,
    };
    use llvq_cuda::{f16_bits, f16_to_f64, ROT_KMAX};
    use llvq_quant::rotation::Rotation;
    use std::time::Instant;

    /// Ten times looser than what the host harness measures; a second net
    /// behind the byte comparison, never the gate.
    const TOL_REL: f64 = 1e-5;
    const TOL_COORD: f64 = 1e-3;

    const ROUNDS: usize = 5;
    const LAUNCHES: usize = 200;
    const WARMUP: usize = 20;
    /// 17,408 * 4: the served 14B's object.
    const OPTIN_14B: u32 = 69_632;
    /// The rows twins: a grid of one, a short tail, a full `PREFILL_ROWS`.
    const ROWS: [u32; 3] = [1, 3, 4];
    const STRIDE_PAD: usize = 8;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Arm {
        /// `rot_apply`, no attribute.
        K0A,
        /// `rot_apply`, opted in at [`OPTIN_14B`].
        K0B,
        /// `rot_apply`, its own module opted in at `n * 4`.
        K0C,
        /// `rot_apply_dev`.
        R0,
    }

    impl Arm {
        fn label(self) -> &'static str {
            match self {
                Arm::K0A => "rot_apply/A",
                Arm::K0B => "rot_apply/B",
                Arm::K0C => "rot_apply/C",
                Arm::R0 => "rot_apply_dev",
            }
        }
    }

    /// The arms a width runs on a card with these bounds.
    fn arms_for(n: usize, def: usize, optin: usize) -> Vec<Arm> {
        let mut v = Vec::new();
        match rot_kernel(n, def, optin) {
            RotKernel::Shared(Fit::Default) => {
                v.push(Arm::K0A);
                if n == 5_120 {
                    v.push(Arm::K0B);
                }
            }
            RotKernel::Shared(Fit::OptIn) => {
                if rot_bytes(n) <= OPTIN_14B as usize {
                    v.push(Arm::K0B);
                } else {
                    v.push(Arm::K0C);
                }
            }
            RotKernel::Device => {}
        }
        v.push(Arm::R0);
        v
    }

    /// The two functions of an arm. Every launch goes through ONE `Cuda`,
    /// handle A's, on its stream and with its buffers: the other modules
    /// only provide functions, loaded in the same (primary) context.
    struct Handle {
        one: CudaFunction,
        rows: CudaFunction,
    }

    /// The tables and the shape of one width, on the device.
    struct Case {
        n: u32,
        m: u32,
        k: u32,
        inv: f32,
        threads: u32,
        sign: CudaSlice<u32>,
        small: CudaSlice<f32>,
    }

    /// One single-row launch of `arm`, reading `x` at `x_off`.
    #[allow(clippy::too_many_arguments)]
    fn launch(
        cuda: &Cuda,
        h: &Handle,
        arm: Arm,
        c: &Case,
        x: &CudaSlice<u16>,
        out: &mut CudaSlice<f32>,
        scratch: &mut CudaSlice<f32>,
        x_off: u32,
    ) -> Result<(), String> {
        match arm {
            Arm::R0 => cuda.launch_rot_dev(
                &h.one, x, &c.sign, &c.small, out, scratch, c.n, c.m, c.k, c.inv, x_off,
                c.threads,
            ),
            _ => cuda.launch_rot(
                &h.one, x, &c.sign, &c.small, out, c.n, c.m, c.k, c.inv, x_off, c.threads,
            ),
        }
    }

    pub fn run() -> Result<(), String> {
        // Everything checked before a single launch: the fixture parses and
        // has a line for every width this binary will read.
        let fixture = parse_rot_fixture(ROT_FIXTURE)?;
        for (n, seed, what) in CARD_WIDTHS {
            if !fixture.contains_key(&(n, seed)) {
                return Err(format!("{what}: no fixture line for ({n}, {seed:#x})"));
            }
        }

        // One bench unit: K0 and R0 side by side. Not the served unit, whose
        // text never sees rotate_wide.cu.
        let sources = llvq_cuda::load_sources_many(&["llvq_rot.cuh", "rotate.cu", "rotate_wide.cu"])?;
        let parts: Vec<&str> = sources.parts.iter().map(String::as_str).collect();
        let src = KernelSource::new(&parts);
        println!("NVRTC source: {} bytes, sha256 {}", src.text.len(), src.sha256);
        if let Some(d) = &sources.overridden_from {
            println!("  WARNING: SOURCES OVERRIDDEN from {d}");
        }
        println!("fixture: {} lines, sha256 {}", fixture.len(), sha256_hex(ROT_FIXTURE.as_bytes()));

        let cuda_a = Cuda::new(&src)?;
        let dev = cuda_a.device()?;
        let (def, optin) = (dev.shared_per_block as usize, dev.shared_per_block_optin as usize);
        println!(
            "\n{}: {} SM, {def} B of shared memory per block by default, {optin} after opt-in",
            dev.name, dev.sm_count
        );

        // The register report of every entry point, on handle A.
        for name in ["rot_apply", "rot_apply_rows", "rot_apply_dev", "rot_apply_rows_dev"] {
            let rep = cuda_a.report(name)?;
            println!(
                "  {:<20} {:>3} registers, {} local B, sm_{}, {} max threads",
                rep.name, rep.num_regs, rep.local_bytes, rep.binary_version, rep.max_threads
            );
            if rep.local_bytes != 0 {
                return Err(format!("{name}: {} bytes of spill", rep.local_bytes));
            }
        }

        // Handle A: no attribute, ever. Handle B: a second module of the same
        // text, opted in at the 14B's size. R0 lives on handle A's module and
        // asks for no shared memory.
        let a = Handle { one: cuda_a.func("rot_apply")?, rows: cuda_a.func("rot_apply_rows")? };
        let r0 = Handle {
            one: cuda_a.func("rot_apply_dev")?,
            rows: cuda_a.func("rot_apply_rows_dev")?,
        };
        // A second module load of the same text; kept alive for the run.
        let cuda_b = Cuda::new(&src)?;
        let b = Handle {
            one: cuda_b.func_dynamic_shared("rot_apply", OPTIN_14B)?,
            rows: cuda_b.func_dynamic_shared("rot_apply_rows", OPTIN_14B)?,
        };
        // Handle C modules, one per width that needs one, alive for the run.
        let mut modules_c: Vec<Cuda> = Vec::new();
        println!("  handle A: rot_apply, no attribute; handle B: rot_apply opted in at {OPTIN_14B} B;");
        println!("  handle C: rot_apply, its own module opted in at n * 4 B; rot_apply_dev: 0 B");

        println!("\nVerification: input sha256, output sha256 against the host, then f64 and norm");
        println!(
            "  {:<44}{:>7}{:>6}{:>4}  {:<15}{:>6}{:>6}{:>11}{:>11}",
            "shape", "n", "m", "k", "arm", "bytes", "rows", "rel", "coord"
        );
        println!("  {}", "-".repeat(112));

        let mut failures: Vec<String> = Vec::new();
        // (arm, n, per-round µs)
        let mut timings: Vec<(Arm, usize, Vec<f64>)> = Vec::new();

        for (n, seed, what) in CARD_WIDTHS {
            let rot = Rotation::new(n, seed);
            let (m, k) = (rot.pow2(), rot.odd());
            if k > ROT_KMAX {
                return Err(format!("{what}: k={k} exceeds KMAX={ROT_KMAX}"));
            }
            // The draw of `tests/rotation_matches_rust.rs`, unchanged.
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
            let mut small = vec![0.0f32; ROT_KMAX * ROT_KMAX];
            for g in 0..k {
                for t in 0..k {
                    small[g * ROT_KMAX + t] = rot.small()[g * k + t] as f32;
                }
            }
            let inv = (1.0f64 / (m as f64).sqrt()) as f32;

            // Gate: the card's input is the host's input.
            let (want_in, want_out) = &fixture[&(n, seed)];
            let got_in = sha256_hex(&rot_input_bytes(&bits, &words, &small, inv));
            if &got_in != want_in {
                return Err(format!(
                    "{what}: INPUT sha256 {got_in} is not the fixture's {want_in}: the card drew \
                     another vector than the host, so no output comparison is read"
                ));
            }

            // The rows input: row r is the activation shifted cyclically by r,
            // rows `n + 8` apart, and the padding is 1.0 so a misread shows.
            let stride = n + STRIDE_PAD;
            let rows_max = *ROWS.iter().max().expect("rows") as usize;
            let mut batch = vec![0x3c00u16; stride * rows_max];
            for r in 0..rows_max {
                for i in 0..n {
                    batch[r * stride + i] = bits[(i + r) % n];
                }
            }

            let threads = (n as u32).next_power_of_two().clamp(32, 1024);
            let cu = &cuda_a;
            let c = Case {
                n: n as u32,
                m: m as u32,
                k: k as u32,
                inv,
                threads,
                sign: cu.up_u32(&words)?,
                small: cu.up_f32(&small)?,
            };
            let d_x = cu.up_u16(&bits)?;
            let d_batch = cu.up_u16(&batch)?;
            let mut out = cu.zeros_f32(n)?;
            let mut scratch = cu.zeros_f32(rot_scratch_floats(rows_max, n, m, k))?;

            let arms = arms_for(n, def, optin);
            let mut handle_c: Option<Handle> = None;
            if arms.contains(&Arm::K0C) {
                let cc = Cuda::new(&src)?;
                handle_c = Some(Handle {
                    one: cc.func_dynamic_shared("rot_apply", rot_bytes(n) as u32)?,
                    rows: cc.func_dynamic_shared("rot_apply_rows", rot_bytes(n) as u32)?,
                });
                modules_c.push(cc);
            }
            let pick = |arm: Arm| -> &Handle {
                match arm {
                    Arm::K0A => &a,
                    Arm::K0B => &b,
                    Arm::K0C => handle_c.as_ref().expect("handle C loaded for this width"),
                    Arm::R0 => &r0,
                }
            };

            for &arm in &arms {
                let h = pick(arm);
                launch(cu, h, arm, &c, &d_x, &mut out, &mut scratch, 0)?;
                cu.sync()?;
                let got = cu.down_f32(&out)?;
                let bytes_ok = &sha256_hex(&f32_le_bytes(&got)) == want_out;

                // Rows: each row against the single-row kernel read at that
                // row's offset in the batch, and row 0 against the fixture.
                let mut rows_ok = true;
                for &nr in &ROWS {
                    let mut out_rows = cu.zeros_f32(n * nr as usize)?;
                    match arm {
                        Arm::R0 => cu.launch_rot_rows_dev(
                            &h.rows, &d_batch, &c.sign, &c.small, &mut out_rows, &mut scratch,
                            c.n, c.m, c.k, inv, 0, stride as u32, nr, threads,
                        )?,
                        _ => cu.launch_rot_rows(
                            &h.rows, &d_batch, &c.sign, &c.small, &mut out_rows, c.n, c.m, c.k,
                            inv, 0, stride as u32, nr, threads,
                        )?,
                    }
                    cu.sync()?;
                    let many = cu.down_f32(&out_rows)?;
                    if f32_le_bytes(&many[..n]) != f32_le_bytes(&got) {
                        rows_ok = false;
                    }
                    for r in 1..nr as usize {
                        let mut single_out = cu.zeros_f32(n)?;
                        launch(cu, h, arm, &c, &d_batch, &mut single_out, &mut scratch,
                               (r * stride) as u32)?;
                        cu.sync()?;
                        let single = cu.down_f32(&single_out)?;
                        if f32_le_bytes(&many[r * n..(r + 1) * n]) != f32_le_bytes(&single) {
                            rows_ok = false;
                        }
                    }
                }

                let (rel, cmax, dn) = errors(&got, &want, norm_in);
                println!(
                    "  {what:<44}{n:>7}{m:>6}{k:>4}  {:<15}{:>6}{:>6}{rel:>11.3e}{cmax:>11.3e}",
                    arm.label(),
                    if bytes_ok { "equal" } else { "DIFF" },
                    if rows_ok { "equal" } else { "DIFF" },
                );
                if !bytes_ok {
                    failures.push(format!("{what}: {} output sha256 differs from the host", arm.label()));
                }
                if !rows_ok {
                    failures.push(format!("{what}: {} rows twin differs", arm.label()));
                }
                if !(rel < TOL_REL && cmax < TOL_COORD && dn < TOL_REL) {
                    failures.push(format!(
                        "{what}: {} rel {rel:.3e}, coord {cmax:.3e}, norm {dn:.3e}",
                        arm.label()
                    ));
                }
            }

            // Timing: ROUNDS rounds; in each, every arm runs WARMUP discarded
            // and LAUNCHES timed chained launches, in an order rotated by one
            // each round.
            let mut per: Vec<Vec<f64>> = vec![Vec::new(); arms.len()];
            for round in 0..ROUNDS {
                for j in 0..arms.len() {
                    let i = (j + round) % arms.len();
                    let (arm, h) = (arms[i], pick(arms[i]));
                    for _ in 0..WARMUP {
                        launch(cu, h, arm, &c, &d_x, &mut out, &mut scratch, 0)?;
                    }
                    cu.sync()?;
                    let t = Instant::now();
                    for _ in 0..LAUNCHES {
                        launch(cu, h, arm, &c, &d_x, &mut out, &mut scratch, 0)?;
                    }
                    cu.sync()?;
                    per[i].push(t.elapsed().as_secs_f64() * 1e6 / LAUNCHES as f64);
                }
            }
            for (i, v) in per.into_iter().enumerate() {
                timings.push((arms[i], n, v));
            }
        }

        println!("\nCost: {ROUNDS} rounds of {LAUNCHES} chained launches, {WARMUP} discarded, order rotated");
        println!("  WARNING: one launch at a time, on an empty stream. Isolated latency, not in-situ time.");
        println!("  {:>7}  {:<15}{:>10}  {:<20}", "n", "arm", "median µs", "[min-max]");
        println!("  {}", "-".repeat(56));
        for (arm, n, v) in &timings {
            let (med, lo, hi) = median_range(v);
            println!("  {n:>7}  {:<15}{med:>10.2}  [{lo:.2}-{hi:.2}]", arm.label());
        }

        // R0/K0 round by round, on the same handle object the served path
        // would use at that width: A up to the default, B at 17,408.
        println!("\nR0 / K0, formed round by round, then median [min-max]");
        for &(n, _, _) in CARD_WIDTHS.iter() {
            let get = |arm: Arm| timings.iter().find(|(a, w, _)| *a == arm && *w == n).map(|t| &t.2);
            let base = if n == 17_408 { Arm::K0B } else { Arm::K0A };
            if let (Some(r), Some(k0)) = (get(Arm::R0), get(base)) {
                let ratios: Vec<f64> = r.iter().zip(k0).map(|(a, b)| a / b).collect();
                let (med, lo, hi) = median_range(&ratios);
                println!("  {n:>7}  R0/{:<12} {med:.3} [{lo:.3}-{hi:.3}]", base.label());
            }
        }

        if !failures.is_empty() {
            println!("\n{} FAILURES:", failures.len());
            for f in &failures {
                println!("  {f}");
            }
            return Err(format!("{} identity failures; see above", failures.len()));
        }
        println!("\nevery output and every rows twin equals the host's bytes");
        Ok(())
    }

    /// (relative error in the norm of the difference, worst coordinate over a
    /// typical one, relative change of the norm against the input's).
    fn errors(got: &[f32], want: &[f64], norm_in: f64) -> (f64, f64, f64) {
        let nrm = want.iter().map(|w| w * w).sum::<f64>().sqrt();
        let dev = got.iter().zip(want).map(|(&g, &w)| (g as f64 - w).powi(2)).sum::<f64>().sqrt();
        let coord = got.iter().zip(want).map(|(&g, &w)| (g as f64 - w).abs()).fold(0.0, f64::max);
        let typical = nrm / (want.len() as f64).sqrt();
        let norm_out = got.iter().map(|&g| g as f64 * g as f64).sum::<f64>().sqrt();
        (dev / nrm, coord / typical, (norm_out - norm_in).abs() / norm_in)
    }

    fn median_range(v: &[f64]) -> (f64, f64, f64) {
        let mut s = v.to_vec();
        s.sort_by(|a, b| a.total_cmp(b));
        let med = if s.len() % 2 == 1 {
            s[s.len() / 2]
        } else {
            (s[s.len() / 2 - 1] + s[s.len() / 2]) / 2.0
        };
        (med, s[0], s[s.len() - 1])
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn the_4b_widths_run_on_handle_a_and_the_14b_down_on_handle_b() {
            let (def, l40s) = (49_152, 101_376);
            assert_eq!(arms_for(2_560, def, l40s), vec![Arm::K0A, Arm::R0]);
            assert_eq!(arms_for(9_728, def, l40s), vec![Arm::K0A, Arm::R0]);
            assert_eq!(arms_for(12_288, def, l40s), vec![Arm::K0A, Arm::R0]);
            assert_eq!(arms_for(5_120, def, l40s), vec![Arm::K0A, Arm::K0B, Arm::R0]);
            assert_eq!(arms_for(17_408, def, l40s), vec![Arm::K0B, Arm::R0]);
            assert_eq!(arms_for(25_600, def, l40s), vec![Arm::R0]);
            assert_eq!(arms_for(31_744, def, l40s), vec![Arm::R0]);
        }

        #[test]
        fn on_the_a100_the_32b_width_runs_k0_on_its_own_opt_in() {
            let (def, a100) = (49_152, 166_912);
            assert_eq!(arms_for(25_600, def, a100), vec![Arm::K0C, Arm::R0]);
            assert_eq!(arms_for(31_744, def, a100), vec![Arm::K0C, Arm::R0]);
            assert_eq!(arms_for(17_408, def, a100), vec![Arm::K0B, Arm::R0]);
        }
    }
}
