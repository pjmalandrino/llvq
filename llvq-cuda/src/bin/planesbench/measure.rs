//! Everything after the build: the launch closures, the verification, the phases, the drift.

use super::*;
use super::fusion::{a4_section, build_fused};
use super::report::report;

/// The launch closures, then the phases, the drift between them, and section A4.
#[allow(clippy::too_many_arguments)]
pub(super) fn measure(
    cuda: Cuda,
    tile: llvq_cuda::tile::Tile,
    dev: llvq_cuda::gpu::DeviceReport,
    union: ArmSet,
    phases: Vec<ArmSet>,
    g70_needed: bool,
    seg_tetra: bool,
    pers_flags: llvq_cuda::persbench::Flags,
    src: KernelSource,
    inputs: Inputs,
    tetra: std::collections::HashMap<String, TetraSrc>,
    mats: Vec<Mat>,
    srcs: Vec<Src>,
    source: String,
    max_dout: usize,
    d_y: cudarc::driver::CudaSlice<f32>,
    d_yh: Option<cudarc::driver::CudaSlice<u16>>,
) -> Result<(), String> {
    let Inputs {
        fd, table, d_tab, tetra_tabs, d_gtab, d_cw, d_e1vtab, d_e1vpay, d_e1vbinom, d_e1vcw, x, d_x, d_xh, ..
    } = inputs;
    let f_slot = cuda.func("tv_slot")?;
    let f_slot_seg = cuda.func("tv_slot_seg")?;
    let f_planes = cuda.func("tv_planes")?;
    let f_planes_seg = cuda.func("tv_planes_seg")?;
    // A4 for Tetra. Loaded unconditionally like every other arm's
    // function: the unit always contains it, and a group without a
    // Tetra stream is skipped at launch rather than at load.
    let f_tetra48_seg = cuda.func("tv_tetra48_seg")?;
    let f_planes12x = cuda.func("tv_planes12x")?;
    let f_golay70 = cuda.func("tv_golay70")?;
    let f_golay70_v1 = cuda.func("tv_golay70_v1")?;
    let f_e1v = cuda.func("tv_e1v")?;
    let f_nullk = cuda.func("tv_nullk")?;
    // Plain `func`, not `func_dynamic_shared`: `tv_f1r_v3g` stages the
    // same tile every other arm passes through `shared`, well under the
    // default per-block allowance. Only QTIP needs the opt-in.
    let f_tetra48 = cuda.func("tv_f1r_v3g")?;
    let f_f16 = cuda.func("tv_f16")?;
    let f_awq = cuda.func("awq_gemv_g128")?;
    let shared = tile.shared_bytes();

    // Each closure is called only if its arm is in the current phase; the
    // named `expect` turns any sequencing error into a readable panic
    // rather than into the measurement of an empty arm.
    let run_slot = |m: &Mat, y: &mut cudarc::driver::CudaSlice<f32>| -> Result<(), String> {
        let a = m.slot.as_ref().expect("slot32 arm not built");
        cuda.launch_slot(
            &f_slot, a.words.dev(), a.bases.dev(), &d_tab, &m.gscale, &m.rscale,
            &m.tail, &d_x, y, m.nblocks as u32, m.tail_w as u32, m.d_out as u32,
            THREADS, shared,
        )
    };
    let run_planes = |m: &Mat, y: &mut cudarc::driver::CudaSlice<f32>| -> Result<(), String> {
        let a = m.planes.as_ref().expect("planes14 arm not built");
        launch_planes(
            &cuda, &f_planes, a.pwords.dev(), &d_tab, &m.gscale, &m.rscale, &m.tail,
            &d_x, y, m.nblocks as u32, m.tail_w as u32, m.d_out as u32, THREADS, shared,
        )
    };
    let run_nullk = |m: &Mat, y: &mut cudarc::driver::CudaSlice<f32>| -> Result<(), String> {
        launch_nullk(
            &cuda, &f_nullk, &m.rscale, &m.tail, &d_x, y,
            m.nblocks as u32, m.tail_w as u32, m.d_out as u32, THREADS, shared,
        )
    };
    let run_e1v = |m: &Mat, y: &mut cudarc::driver::CudaSlice<f32>| -> Result<(), String> {
        let a = m.e1v.as_ref().expect("e1v arm not built");
        // 🚨 `a.row_blocks`, not `m.nblocks`. They are the same number,
        // and that is exactly why we take the one the CUT used: the day
        // they diverge, the stream would be right and the kernel would
        // read elsewhere. The assertion says so before the launch.
        assert_eq!(
            a.row_blocks, m.nblocks,
            "{}: the stream is cut on {} blocks per row and the kernel would read {}",
            m.name, a.row_blocks, m.nblocks
        );
        launch_e1v(
            &cuda, &f_e1v, a.data.dev(), a.bases.dev(),
            d_e1vtab.as_ref().expect("E1v tables not uploaded"),
            d_e1vpay.as_ref().expect("E1v tables not uploaded"),
            d_e1vbinom.as_ref().expect("E1v tables not uploaded"),
            d_e1vcw.as_ref().expect("E1v tables not uploaded"),
            &m.gscale, &m.rscale, &m.tail, &d_x, y,
            a.row_blocks as u32, m.tail_w as u32, m.d_out as u32, THREADS, shared,
        )
    };
    // Returns Ok on a matrix with NO Tetra arm rather than panicking, and
    // that is the arm's defining asymmetry: 36 of the 252 projections are
    // `v_proj`, stored as int4 g128 in the served file, and read by a
    // second kernel that is not this one. So a timed pass of this arm
    // carries 216 launches where every ball arm carries 252. The count is
    // printed beside every ratio it enters — see the pair line.
    let run_tetra48 = |m: &Mat, y: &mut cudarc::driver::CudaSlice<f32>| -> Result<(), String> {
        let Some(a) = m.tetra.as_ref() else {
            return Ok(());
        };
        launch_tetra48(
            &cuda, &f_tetra48, a.words.dev(), a.stride_u32,
            tetra_tabs.as_ref().expect("tetra48 tables not uploaded"),
            &a.gscale, &a.rscale, &a.tail, &d_x, y,
            m.nblocks as u32, m.tail_w as u32, m.d_out as u32, THREADS, shared,
        )
    };
    let run_planes12x =
        |m: &Mat, y: &mut cudarc::driver::CudaSlice<f32>| -> Result<(), String> {
            let a = m.p12.as_ref().expect("planes12x arm not built");
            launch_planes12x(
                &cuda, &f_planes12x, a.words.dev(), a.exc_idx.dev(), a.exc_words.dev(),
                &d_tab, &m.gscale, &m.rscale, &m.tail, &d_x, y, m.nblocks as u32,
                m.tail_w as u32, a.n_exc as u32, m.d_out as u32, THREADS, shared,
            )
        };
    // v1 and v2: same buffers, same grid, same memset protocol, only the
    // KERNEL changes. That is the whole comparison.
    let run_g70 = |m: &Mat,
                   f: &cudarc::driver::CudaFunction,
                   y: &mut cudarc::driver::CudaSlice<f32>|
     -> Result<(), String> {
        let a = m.g70.as_ref().expect("golay70 arm not built");
        launch_golay70(
            &cuda, f, a.words.dev(), a.exc_idx.dev(), a.exc_words.dev(),
            d_cw.as_ref().expect("golay70 tables not uploaded"),
            d_gtab.as_ref().expect("golay70 tables not uploaded"),
            &d_tab, &m.gscale, &m.rscale, &m.tail, &d_x, y,
            m.nblocks as u32, m.tail_w as u32, a.n_exc as u32, m.d_out as u32,
            THREADS, shared,
        )
    };
    let run_awq =
        |m: &Mat, y: &mut cudarc::driver::CudaSlice<u16>| -> Result<(), String> {
            let a = m.awq.as_ref().expect("awq arm not built");
            launch_awq(
                &cuda, &f_awq,
                d_xh.as_ref().expect("binary16 activation not uploaded"),
                a.w.dev(), a.z.dev(), a.s.dev(), y,
                m.d_in as u32, m.d_out as u32,
            )
        };
    // QTIP resolves ONE function per shape — the kernel is templated on M
    // and K — and each lookup must ask for the 64 KiB of dynamic shared
    // memory the codebook needs: `func`, which every other arm uses, would
    // return a function that fails at launch.
    //
    // 🚨 Resolved ONCE, here, and not inside the closure. A first draft
    // called `func_dynamic_shared` per matrix, hence 252 times **inside
    // the timed loop**, where every other arm resolves its function once
    // outside it (`f_planes`, `f_golay70`, `f_f16`…). That is a module
    // lookup plus a device query per dispatch, charged to this arm and to
    // no other — a handicap on the competitor, in the direction that
    // flatters us. Found by an adversarial review of the wiring, before a
    // single job.
    let qtip_funcs: Vec<(String, cudarc::driver::CudaFunction)> = match union.has(arms::QTIP) {
        false => Vec::new(),
        true => llvq_cuda::qtip_host::QTIP_SHAPES
            .iter()
            .filter_map(|&(d_out, d_in)| llvq_cuda::qtip_host::kernel_name(d_out, d_in))
            .map(|n| {
                cuda.func_dynamic_shared(&n, llvq_cuda::qtip_host::QTIP_SHARED_BYTES)
                    .map(|f| (n, f))
            })
            .collect::<Result<_, _>>()?,
    };
    let run_qtip =
        |m: &Mat, y: &mut cudarc::driver::CudaSlice<f32>| -> Result<(), String> {
            let a = m.qtip.as_ref().expect("qtip arm not built");
            let f = &qtip_funcs
                .iter()
                .find(|(n, _)| *n == a.kernel)
                .expect("qtip shim not resolved")
                .1;
            launch_qtip(
                &cuda,
                f,
                y,
                a.compressed.dev(),
                d_xh.as_ref().expect("binary16 activation not uploaded"),
                a.codebook.dev(),
            )
        };
    let run_f16 = |m: &Mat, y: &mut cudarc::driver::CudaSlice<f32>| -> Result<(), String> {
        cuda.launch_f16(&f_f16, &m.w16, &d_x, y, m.d_in as u32, m.d_out as u32, THREADS, shared)
    };
    // The cuBLAS denominator (F1, 2026-08-18): a handle bound to the SAME
    // stream as every launch, so that `cuda.sync()` bounds its calls too.
    // A handle on another stream would time enqueues, not executions.
    let blas = match union.has(arms::CUBLASF16) {
        true => Some(
            cudarc::cublas::CudaBlas::new(cuda.stream().clone())
                .map_err(|e| format!("cublasCreate: {e:?}"))?,
        ),
        false => None,
    };
    let run_cublas = |m: &Mat, y: &mut cudarc::driver::CudaSlice<u16>| -> Result<(), String> {
        use cudarc::cublas::{result as cbr, sys as cbs};
        use cudarc::driver::{DevicePtr, DevicePtrMut};
        let blas = blas.as_ref().expect("cublasf16 arm not built");
        let xh = d_xh.as_ref().expect("binary16 activation not uploaded");
        // W is d_out×d_in row-major, hence (d_in, d_out) column-major
        // with lda = d_in; op(A) = T gives the (d_out, d_in) that the
        // matvec y = W·x wants, n = 1. All in R_16F, 32F accumulation,
        // the standard GemmEx combo, the one candle uses.
        let (wp, _wg) = m.w16.device_ptr(cuda.stream());
        let (xp, _xg) = xh.device_ptr(cuda.stream());
        let (yp, _yg) = y.device_ptr_mut(cuda.stream());
        let (alpha, beta) = (1.0f32, 0.0f32);
        unsafe {
            cbr::gemm_ex(
                *blas.handle(),
                cbs::cublasOperation_t::CUBLAS_OP_T,
                cbs::cublasOperation_t::CUBLAS_OP_N,
                m.d_out as i32,
                1,
                m.d_in as i32,
                &alpha as *const f32 as *const std::ffi::c_void,
                wp as *const std::ffi::c_void,
                cbs::cudaDataType_t::CUDA_R_16F,
                m.d_in as i32,
                xp as *const std::ffi::c_void,
                cbs::cudaDataType_t::CUDA_R_16F,
                m.d_in as i32,
                &beta as *const f32 as *const std::ffi::c_void,
                yp as *mut std::ffi::c_void,
                cbs::cudaDataType_t::CUDA_R_16F,
                m.d_out as i32,
                cbs::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                cbs::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            )
            .map_err(|e| format!("cublasGemmEx: {e:?}"))
        }
    };
    let run_slot_seg =
        |fm: &FusedMat, y: &mut cudarc::driver::CudaSlice<f32>| -> Result<(), String> {
            cuda.launch_slot_seg(
                &f_slot_seg, &fm.words, &fm.bases, &d_tab, &fm.gscale, &fm.gs_off,
                &fm.rscale, &fm.tail, &d_x, y, fm.nblocks as u32, fm.tail_w as u32,
                fm.d_out as u32, THREADS, shared,
            )
        };
    // A fused group whose members are not all Tetra has no stream, and
    // this returns `Ok(false)` so the caller skips it rather than launching
    // on an absent buffer. On the served object that is every q+k+v group,
    // because `v_proj` is int4.
    let run_tetra48_seg =
        |fm: &FusedMat, y: &mut cudarc::driver::CudaSlice<f32>| -> Result<bool, String> {
            let Some(t) = fm.tetra.as_ref() else {
                return Ok(false);
            };
            launch_tetra48_seg(
                &cuda, &f_tetra48_seg, &t.words, fm.tstride,
                tetra_tabs.as_ref().expect("tetra48 tables not uploaded"),
                &t.gscale, &t.gs_off, &t.rscale, &t.tail, &d_x, y,
                fm.nblocks as u32, fm.tail_w as u32, fm.d_out as u32, THREADS, shared,
            )?;
            Ok(true)
        };
    let run_planes_seg =
        |fm: &FusedMat, y: &mut cudarc::driver::CudaSlice<f32>| -> Result<(), String> {
            launch_planes_seg(
                &cuda, &f_planes_seg, &fm.pwords, &d_tab, &fm.gscale, &fm.gs_off,
                &fm.rscale, &fm.tail, &d_x, y, fm.nblocks as u32, fm.tail_w as u32,
                fm.d_out as u32, THREADS, shared,
            )
        };

    let rows: usize = mats.iter().map(|m| m.d_out).sum();
    let total_blocks: u64 = mats.iter().map(|m| (m.d_out * m.nblocks) as u64).sum();
    if union.has(arms::PLANES12X) {
        let total_exc: u64 = mats
            .iter()
            .map(|m| m.p12.as_ref().expect("planes12x arm not built").n_exc as u64)
            .sum();
        println!(
            "  L = 5 exceptions: {total_exc} out of {total_blocks} blocks \
             ({:.4}%), corrected in the same launch",
            total_exc as f64 * 100.0 / total_blocks as f64
        );
    }
    if g70_needed {
        let total_gexc: u64 = mats
            .iter()
            .map(|m| m.g70.as_ref().expect("golay70 arm not built").n_exc as u64)
            .sum();
        println!(
            "  E2 exceptions (parity-violating or L = 5): {total_gexc} out of \
             {total_blocks} blocks ({:.4}%), corrected in the same launch",
            total_gexc as f64 * 100.0 / total_blocks as f64
        );
    }

    // One pass = all matrices, one stream, in order — the layers' real
    // dependency. Wall clock around the pass plus a synchronize; the
    // phase's arms interleave inside each round, in REGISTRATION order
    // (a selection skips arms, it never moves one). The correction arms'
    // passes include their per-matrix memset — part of what those
    // layouts cost.
    let per_round = |num: &[f64], den: &[f64]| -> Vec<f64> {
        num.iter().zip(den).map(|(a, b)| a / b).collect()
    };
    // 🕳️ Both tables used to live HERE, sized on `arms::N_ARMS` with seven
    // literals — and no development machine type-checks this file. P4 took
    // N_ARMS from 7 to 15 on 2026-08-15 without touching them, and the CUDA
    // image stayed uncompilable for a day. They are in `arms.rs`, which
    // compiles and is tested on the Mac.
    //
    // 🚨 And NO local alias is left of them, because an alias must announce
    // a type, hence restate a length — which reproduces the defect three
    // lines from its own fix. That cost a build: the first version wrote
    // `const DISPLAY: [usize; 8]`, and the floor took `DISPLAY_ORDER` to
    // nine the next day.
    let n_phases = phases.len();
    let (mats, d_y, _d_yh, phase_times) = time_phases(
        mats, d_y, d_yh, f_golay70, f_golay70_v1, &cuda, &phases, n_phases, rows, &per_round, tile, &source,
        &dev, &run_slot, &run_planes, &run_planes12x, &run_g70, &run_f16, &run_awq, &run_cublas, &run_e1v,
        &run_nullk, &run_qtip, &run_tetra48,
    )?;
    delta_control(&phases, n_phases, &phase_times, &per_round);

    // ---- A4: the q+k+v and gate+up fusion, correctness first, cost after ----
    //
    // Built HERE, after the five-arm table: neither its ~2.9 GB nor its
    // doubled `d_out` may exist while the table is being measured (see the
    // long comment above `max_dout`).
    let fused = build_fused(&srcs, union, &x, fd, table, seg_tetra, &tetra, &cuda)?;

    a4_section(
        mats, fused, d_y, max_dout, &cuda, seg_tetra, &pers_flags, &src, tile, &tetra_tabs, &d_x, &tetra, &x,
        shared, &run_slot, &run_planes, &run_tetra48, &run_slot_seg, &run_planes_seg, &run_tetra48_seg,
    )
}

    // ---- f64 verification, PER ARM, at the latest before the first
    // phase that times the arm (the §7 guard of the preregistration:
    // never a timing before its proof).
    //
    // The substantive comments of the six original verifications hold
    // unchanged: the same exact reference for every LLVQ arm (the proof
    // of the Planes12x overlay and of the E2 correction is precisely that
    // they reach it), y16_ref for the FP16 witness, and for AWQ ITS
    // reference, ITS denominator and ITS tolerance, looser by a factor of
    // ~100 because its output is rounded to binary16 by the kernel
    // (`f2h`), not because anything is forgiven it: its reference
    // `scale·(q − zero)` stays exact bit for bit.
// The body was moved verbatim out of `run`, where these names were owned locals; they are
// reference parameters here, so `&name` borrows a reference. Kept as moved.
#[allow(clippy::too_many_arguments, clippy::needless_borrow)]
fn verify_arm(
    a: usize,
    mats: &[Mat],
    d_y: &mut cudarc::driver::CudaSlice<f32>,
    d_yh: &mut Option<cudarc::driver::CudaSlice<u16>>,
    cuda: &Cuda,
    f_golay70: &cudarc::driver::CudaFunction,
    f_golay70_v1: &cudarc::driver::CudaFunction,
    run_slot: &impl RunMat<f32>,
    run_planes: &impl RunMat<f32>,
    run_planes12x: &impl RunMat<f32>,
    run_g70: &impl RunG70,
    run_f16: &impl RunMat<f32>,
    run_awq: &impl RunMat<u16>,
    run_cublas: &impl RunMat<u16>,
    run_e1v: &impl RunMat<f32>,
    run_nullk: &impl RunMat<f32>,
    run_qtip: &impl RunMat<f32>,
    run_tetra48: &impl RunMat<f32>,
) -> Result<f64, String> {
    // 🕳️ Seeded at 0.0, so an arm whose every matrix returned
    // `NEG_INFINITY` folded to **0.0** and printed `0.0e0` — which
    // reads as perfect agreement with the reference. The floor arm's
    // own comment says it "prints as `-inf`, which nobody reads as a
    // measured error": that was the intent and the seed defeated it,
    // silently, since P4 added the arm. It also caught the Tetra arm,
    // whose unmatched matrices return the same value.
    let mut worst = f64::NEG_INFINITY;
    for m in mats {
        let e = match a {
            // QTIP writes f32, so it is held to OUR threshold. Its
            // reference is exact — we know the bits, hence the state,
            // hence the codebook entry and the sign — so the only
            // difference left is accumulation order.
            arms::QTIP => {
                run_qtip(m, d_y)?;
                cuda.sync()?;
                let got = cuda.down_f32(d_y)?;
                let q = m.qtip.as_ref().expect("qtip arm not built");
                let e = worst_error(&got[..m.d_out], &q.y_ref, &q.scale);
                assert!(e < TOL, "{} / QTIP: {e:.2e}·Σ|w·x|", m.name);
                e
            }
            arms::AWQ => {
                let yh = d_yh.as_mut().expect("d_yh of the awq arm");
                run_awq(m, yh)?;
                cuda.sync()?;
                let goth = cuda.down_u16(yh)?;
                let gotf: Vec<f32> =
                    goth[..m.d_out].iter().map(|&h| f16_to_f64(h) as f32).collect();
                let aw = m.awq.as_ref().expect("awq arm not built");
                let e = worst_error(&gotf, &aw.y_ref, &aw.scale);
                assert!(e < AWQ_TOL, "{} / AWQ: {e:.2e}·Σ|w·x|", m.name);
                e
            }
            // The served Tetra arm, held to `TOL` and not `AWQ_TOL`:
            // its output is f32, so nothing is forgiven it. Its
            // reference is its OWN — the second file's weights are not
            // the first's, so neither `m.y_ref` nor `m.y16_ref`
            // describes them, the same situation as AWQ and QTIP and
            // the same answer.
            //
            // A matrix with no Tetra arm is skipped, not verified
            // against a stale `d_y`: the launch would not have run and
            // the buffer would still hold the previous arm's output,
            // which is the one way this check could pass on nothing.
            arms::TETRA48 => match m.tetra.as_ref() {
                None => f64::NEG_INFINITY,
                Some(t) => {
                    run_tetra48(m, d_y)?;
                    cuda.sync()?;
                    let got = cuda.down_f32(d_y)?;
                    let e = worst_error(&got[..m.d_out], &t.y_ref, &t.scale);
                    assert!(e < TOL, "{} / tetra48: {e:.2e}·Σ|w·x|", m.name);
                    e
                }
            },
            // cuBLAS: same tolerance as AWQ and for the same reason,
            // the output is rounded to binary16 by GemmEx (C in
            // R_16F), not because anything is forgiven it. Its
            // reference is its own: f16 weights × binary16 input,
            // accumulated in f64.
            arms::CUBLASF16 => {
                let yh = d_yh.as_mut().expect("d_yh of the cublasf16 arm");
                run_cublas(m, yh)?;
                cuda.sync()?;
                let goth = cuda.down_u16(yh)?;
                let gotf: Vec<f32> =
                    goth[..m.d_out].iter().map(|&h| f16_to_f64(h) as f32).collect();
                let want = m.y16h_ref.as_ref().expect("cublasf16 reference not built");
                let e = worst_error(&gotf, want, &m.scale);
                assert!(e < AWQ_TOL, "{} / cublasf16: {e:.2e}·Σ|w·x|", m.name);
                e
            }
            // 🚨 The floor has NO yardstick, and cannot have one: it
            // does not compute the model's product. What is required
            // of it is what `bin/rankbench` requires of its `sol`
            // anchor, to be OBSERVABLE. A kernel the compiler had
            // emptied would time beautifully and measure nothing.
            arms::NULLK => {
                run_nullk(m, d_y)?;
                cuda.sync()?;
                let got = cuda.down_f32(d_y)?;
                let nz = got[..m.d_out].iter().filter(|v| **v != 0.0 && v.is_finite()).count();
                assert!(
                    nz > m.d_out / 2,
                    "{} / nullk: output mostly zero, loop eliminated?",
                    m.name
                );
                // 🕳️ Returning 0.0 here puts "worst errors nullk
                // 0.0e0" in the V0 line, which reads as PERFECT
                // agreement with the reference while this arm is not
                // compared at all. A NaN would be worse (it would
                // propagate). f64::NEG_INFINITY is neutral for the
                // aggregating `max` and prints as `-inf`, which nobody
                // reads as a measured error.
                f64::NEG_INFINITY
            }
            _ => {
                match a {
                    arms::SLOT32 => run_slot(m, d_y)?,
                    arms::PLANES14 => run_planes(m, d_y)?,
                    arms::PLANES12X => run_planes12x(m, d_y)?,
                    arms::GOLAY70V1 => run_g70(m, &f_golay70_v1, d_y)?,
                    arms::GOLAY70V2 => run_g70(m, &f_golay70, d_y)?,
                    arms::E1V => run_e1v(m, d_y)?,
                    arms::FP16 => run_f16(m, d_y)?,
                    _ => unreachable!("unknown arm"),
                }
                cuda.sync()?;
                let got = cuda.down_f32(d_y)?;
                let want = if a == arms::FP16 { &m.y16_ref } else { &m.y_ref };
                let e = worst_error(&got[..m.d_out], want, &m.scale);
                assert!(
                    e < TOL,
                    "{} / {}: {e:.2e}·Σ|w·x|",
                    m.name,
                    arms::ARM_NAMES[a]
                );
                e
            }
        };
        worst = worst.max(e);
    }
    Ok(worst)
}

    // A phase's upload wave: the buffers of the entering arms go up to
    // the card, those of the arms already staged do not move (stage_up_*
    // is idempotent, v1 and v2 share the golay70 group, so the second of
    // the two re-uploads nothing).
// The body was moved verbatim out of `run`, where these names were owned locals; they are
// reference parameters here, so `&name` borrows a reference. Kept as moved.
#[allow(clippy::needless_borrow)]
fn upload_added(
    mats: &mut [Mat],
    added: ArmSet,
    cuda: &Cuda,
) -> Result<(), String> {
    for m in mats.iter_mut() {
        if added.has(arms::SLOT32) {
            let a = m.slot.as_mut().expect("slot32 arm not built");
            stage_up_u32(&cuda, &mut a.words)?;
            stage_up_u32(&cuda, &mut a.bases)?;
        }
        if added.has(arms::PLANES14) {
            let a = m.planes.as_mut().expect("planes14 arm not built");
            stage_up_u32(&cuda, &mut a.pwords)?;
        }
        if added.has(arms::PLANES12X) {
            let a = m.p12.as_mut().expect("planes12x arm not built");
            stage_up_u32(&cuda, &mut a.words)?;
            stage_up_u32(&cuda, &mut a.exc_idx)?;
            stage_up_u32(&cuda, &mut a.exc_words)?;
        }
        if added.has(arms::GOLAY70V1) || added.has(arms::GOLAY70V2) {
            let a = m.g70.as_mut().expect("golay70 arm not built");
            stage_up_u32(&cuda, &mut a.words)?;
            stage_up_u32(&cuda, &mut a.exc_idx)?;
            stage_up_u32(&cuda, &mut a.exc_words)?;
        }
        if added.has(arms::TETRA48) {
            // `if let`, not `expect`: a matrix whose `v_proj` is int4
            // has no Tetra arm and nothing to stage. Every other arm
            // here is built for every matrix, which is why they can
            // afford to panic.
            if let Some(a) = m.tetra.as_mut() {
                stage_up_u32(&cuda, &mut a.words)?;
            }
        }
        if added.has(arms::E1V) {
            let a = m.e1v.as_mut().expect("e1v arm not built");
            stage_up_u32(&cuda, &mut a.data)?;
            stage_up_u32(&cuda, &mut a.bases)?;
        }
        if added.has(arms::AWQ) {
            let a = m.awq.as_mut().expect("awq arm not built");
            stage_up_u32(&cuda, &mut a.w)?;
            stage_up_u32(&cuda, &mut a.z)?;
            stage_up_u16(&cuda, &mut a.s)?;
        }
        // 🚨 Without this branch, an arm present in phase 2 but not
        // in phase 1 stays `Staged::Host` and panics on `dev()`, and
        // that is EXACTLY the shape of the preregistration's P3
        // command. The panic would have landed after the transcode of
        // the 252 matrices and after the whole of phase 1, about 90%
        // of a job lost; P2, which puts qtip in phase 0, escaped it by
        // accident, so the first job would have revealed nothing.
        // `stage_up_*` is idempotent, so the branch is harmless when
        // the arm is already staged.
        if added.has(arms::QTIP) {
            let a = m.qtip.as_mut().expect("qtip arm not built");
            stage_up_u32(&cuda, &mut a.compressed)?;
            stage_up_u16(&cuda, &mut a.codebook)?;
        }
    }
    Ok(())
}

// ---- the phases: upload, verification, rounds, report ----
/// The phases: each one's upload wave, its verification, its interleaved rounds and its table.
/// The matrices and the output buffers go in and come back out, for section A4.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn time_phases(
    mut mats: Vec<Mat>,
    mut d_y: cudarc::driver::CudaSlice<f32>,
    mut d_yh: Option<cudarc::driver::CudaSlice<u16>>,
    f_golay70: cudarc::driver::CudaFunction,
    f_golay70_v1: cudarc::driver::CudaFunction,
    cuda: &Cuda,
    phases: &[ArmSet],
    n_phases: usize,
    rows: usize,
    per_round: &impl Fn(&[f64], &[f64]) -> Vec<f64>,
    tile: llvq_cuda::tile::Tile,
    source: &str,
    dev: &llvq_cuda::gpu::DeviceReport,
    run_slot: &impl RunMat<f32>,
    run_planes: &impl RunMat<f32>,
    run_planes12x: &impl RunMat<f32>,
    run_g70: &impl RunG70,
    run_f16: &impl RunMat<f32>,
    run_awq: &impl RunMat<u16>,
    run_cublas: &impl RunMat<u16>,
    run_e1v: &impl RunMat<f32>,
    run_nullk: &impl RunMat<f32>,
    run_qtip: &impl RunMat<f32>,
    run_tetra48: &impl RunMat<f32>,
) -> Result<(Vec<Mat>, cudarc::driver::CudaSlice<f32>, Option<cudarc::driver::CudaSlice<u16>>, Vec<[Vec<f64>; arms::N_ARMS]>), String> {
    let mut verified = [false; arms::N_ARMS];
    let mut phase_times: Vec<[Vec<f64>; arms::N_ARMS]> = Vec::new();
    for pi in 0..n_phases {
        let phase = phases[pi];
        if pi > 0 {
            let added = phase.minus(phases[pi - 1]);
            if !added.is_empty() {
                println!(
                    "\n— phase {}/{n_phases}: upload of the arms added ({}) —",
                    pi + 1,
                    added.label()
                );
                upload_added(&mut mats, added, cuda)?;
            }
        }
        let mut worsts: Vec<String> = Vec::new();
        for a in phase.iter() {
            if !verified[a] {
                if worsts.is_empty() {
                    println!("\nVerifying every row against the f64 reference…");
                }
                let w = verify_arm(
                    a, &mats, &mut d_y, &mut d_yh, cuda, &f_golay70, &f_golay70_v1, run_slot, run_planes,
                    run_planes12x, run_g70, run_f16, run_awq, run_cublas, run_e1v, run_nullk, run_qtip,
                    run_tetra48,
                )?;
                worsts.push(format!("{} {w:.1e}", arms::ARM_NAMES[a]));
                verified[a] = true;
            }
        }
        if !worsts.is_empty() {
            println!(
                "  {rows} rows, threshold {TOL:.0e} (AWQ, cublasf16: {AWQ_TOL:.0e}, \
                 binary16 output) — worst errors {} ·Σ|w·x|",
                worsts.join(", ")
            );
        }
        let mut times: [Vec<f64>; arms::N_ARMS] = Default::default();
        for rep in 0..ROUNDS {
            for (arm, t_arm) in times.iter_mut().enumerate() {
                if !phase.has(arm) {
                    continue;
                }
                let t = Instant::now();
                for m in &mats {
                    match arm {
                        arms::SLOT32 => run_slot(m, &mut d_y)?,
                        arms::PLANES14 => run_planes(m, &mut d_y)?,
                        arms::PLANES12X => run_planes12x(m, &mut d_y)?,
                        arms::GOLAY70V1 => run_g70(m, &f_golay70_v1, &mut d_y)?,
                        arms::FP16 => run_f16(m, &mut d_y)?,
                        arms::AWQ => {
                            run_awq(m, d_yh.as_mut().expect("d_yh of the awq arm"))?
                        }
                        arms::CUBLASF16 => {
                            run_cublas(m, d_yh.as_mut().expect("d_yh of the cublasf16 arm"))?
                        }
                        arms::GOLAY70V2 => run_g70(m, &f_golay70, &mut d_y)?,
                        arms::E1V => run_e1v(m, &mut d_y)?,
                        arms::NULLK => run_nullk(m, &mut d_y)?,
                        arms::QTIP => run_qtip(m, &mut d_y)?,
                        // Last, because it registered last: registration
                        // order is dispatch order, and inserting it
                        // anywhere else would move the dispatch of an arm
                        // that carries a published number.
                        arms::TETRA48 => run_tetra48(m, &mut d_y)?,
                        _ => unreachable!("unknown arm"),
                    }
                }
                cuda.sync()?;
                let s = t.elapsed().as_secs_f64();
                if rep >= WARMUP {
                    t_arm.push(s);
                }
            }
        }
        report(&mats, pi, phase, &times, n_phases, per_round, tile, source, dev);
        phase_times.push(times);
    }
    Ok((mats, d_y, d_yh, phase_times))
}

// ---- Δ_contrôle: the drift of the common arms between phases ----
//
// The number the §4 rule of the 2026-08-10 preregistration asks the job
// to report, printed by the bench itself so that no ratio is ever taken
// between two processes: R = max(Δ_contrôle, intra-run half-range of
// the most dispersed arm); |Δ| > 2R separates, |Δ| < R is
// indistinguishable at this resolution, and between the two: not
// resolved, published as such.
fn delta_control(
    phases: &[ArmSet],
    n_phases: usize,
    phase_times: &[[Vec<f64>; arms::N_ARMS]],
    per_round: &impl Fn(&[f64], &[f64]) -> Vec<f64>,
) {
    if n_phases > 1 {
        println!("\nΔ_contrôle — drift of the common arms between consecutive phases");
        let med = |v: &[f64]| -> f64 {
            let mut s = v.to_vec();
            s.sort_by(f64::total_cmp);
            s[s.len() / 2]
        };
        for k in 1..n_phases {
            let (prev, cur) = (&phase_times[k - 1], &phase_times[k]);
            let common = phases[k - 1]; // ⊆ phases[k], guaranteed at parsing
            let mut delta_ctrl = 0.0f64;
            for a in common.iter() {
                let (m0, m1) = (med(&prev[a]), med(&cur[a]));
                let dms = (m1 - m0) / m0;
                if a == arms::FP16 {
                    println!(
                        "  phases {k}→{} {:<10} med {:>8.3} → {:>8.3} ms ({:+.2}%)",
                        k + 1,
                        arms::ARM_NAMES[a],
                        m0 * 1e3,
                        m1 * 1e3,
                        dms * 100.0
                    );
                } else {
                    let r0 = med(&per_round(&prev[arms::FP16], &prev[a]));
                    let r1 = med(&per_round(&cur[arms::FP16], &cur[a]));
                    let dr = (r1 - r0) / r0;
                    delta_ctrl = delta_ctrl.max(dr.abs());
                    println!(
                        "  phases {k}→{} {:<10} med {:>8.3} → {:>8.3} ms ({:+.2}%), \
                         vs FP16 {:.3} → {:.3} ({:+.2}%)",
                        k + 1,
                        arms::ARM_NAMES[a],
                        m0 * 1e3,
                        m1 * 1e3,
                        dms * 100.0,
                        r0,
                        r1,
                        dr * 100.0
                    );
                }
            }
            let mut half = 0.0f64;
            let mut half_name = "";
            for a in phases[k].iter() {
                if a == arms::FP16 {
                    continue;
                }
                let (lo, md, hi) = spread(per_round(&cur[arms::FP16], &cur[a]));
                let h = (hi - lo) / 2.0 / md;
                if h > half {
                    half = h;
                    half_name = arms::ARM_NAMES[a];
                }
            }
            let r = delta_ctrl.max(half);
            println!(
                "  Δ_contrôle (max |Δ| of the ratios vs FP16) = {:.2}%; max intra-run \
                 half-range = {:.2}% ({half_name}); R = {:.2}%",
                delta_ctrl * 100.0,
                half * 100.0,
                r * 100.0
            );
        }
    }
}
