//! Planes14 against Slot32 against the FP16 witness, on NVIDIA — plus the
//! M2 arm: `tv_planes12x`, the sparse overlay.
//!
//! `planesbench <model.llvq>` measures the published model: one token's worth
//! of linear algebra over its 252 projections, four arms interleaved in every
//! round — `tv_slot` (the published kernel, untouched), `tv_planes` (layout
//! E1a: uniform 14-byte records, no bases array, explicit bit-plane levels),
//! `tv_planes12x` (M2: a 12-byte main stream capped at 4 levels, the 5-level
//! blocks swapped for their best L ≤ 4 direction and corrected **exactly**
//! by an exception pass in the same launch — one memset + one atomicAdd per
//! row and per exception is the price, timed inside the arm) and `tv_f16`.
//! With no argument it falls back to the synthetic seven-shape path, sized
//! past the card's L2 — for iterating on the kernel, nothing else.
//!
//! The protocol is `bin/matvec`'s, reproduced rather than referenced: rounds
//! with warmup discarded, all arms dispatched every round in the same order,
//! ratios formed **round by round** and reported as median with range — never
//! a quotient of two minima. Every row of every matrix is verified against an
//! f64 CPU reference before any timing, and the Planes14 stream is proved a
//! bit-exact bijection of the Slot32 content block by block during the build.
//!
//! The Planes14 stream itself comes from `planes14_from_slot32` (see
//! `src/planes14_host.rs`): a pure bit-level transcoding of the Slot32
//! stream, and the **single point to re-plug** once `llvq-artifact` grows the
//! layout.

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("planesbench targets NVIDIA GPUs; there is nothing to run here.");
}

#[cfg(target_os = "linux")]
fn main() -> Result<(), String> {
    linux::run()
}

#[cfg(target_os = "linux")]
mod linux {
    use cudarc::driver::PushKernelArg;
    use llvq_artifact::runtime::{transcode, transcode_planes12x, ClassTable, Layout};
    use llvq_core::{SplitMix64, DIM};
    use llvq_cuda::gpu::{Cuda, KernelSource};
    use llvq_cuda::{f16_bits, f16_to_f64};
    use llvq_search::fastdec::FastDecoder;
    use llvq_search::index::N13;
    use llvq_search::Searcher;
    use std::time::Instant;

    include!("../planes14_host.rs");

    /// Blocks staged per tile: 3072 columns, 12 KB. Injected into the kernel
    /// source by the host so the staging size and the tiling are one constant.
    const TILE_BLOCKS: usize = 128;
    const THREADS: u32 = 256;
    const GSCALE: [f32; 2] = [0.625, 1.375];
    const TABLE_ENTRIES: usize = 512;
    const REC_WORDS: usize = 6;
    const TOL: f64 = 1e-5;
    const ROUNDS: usize = 7;
    const WARMUP: usize = 2;
    /// Arm order inside a round, fixed: Slot32, Planes14, Planes12x, FP16.
    const ARMS: usize = 4;

    /// The seven projection shapes of Qwen3-4B, `(name, d_out, d_in)`.
    const SHAPES: [(&str, usize, usize); 7] = [
        ("q_proj", 4096, 2560),
        ("k_proj", 1024, 2560),
        ("v_proj", 1024, 2560),
        ("o_proj", 2560, 4096),
        ("gate_proj", 9728, 2560),
        ("up_proj", 9728, 2560),
        ("down_proj", 2560, 9728),
    ];

    /// The two Planes14 sources are not embedded in `lib.rs` — new files
    /// only, that file belongs to another lot — so the bin carries its own
    /// copies and honours `LLVQ_KERNEL_DIR` with the same contract as
    /// `load_sources_many`: without the variable, the embedded text; with it,
    /// the files from the directory, disclosed loudly by the caller.
    const PLANES_CUH_EMBED: &str = include_str!("../../kernels/llvq_planes.cuh");
    const PLANES_CU_EMBED: &str = include_str!("../../kernels/planes.cu");
    const PLANES12_CUH_EMBED: &str = include_str!("../../kernels/llvq_planes12.cuh");
    const PLANES12_CU_EMBED: &str = include_str!("../../kernels/planes12.cu");

    fn load_planes_sources() -> Result<(String, String, Option<String>), String> {
        match std::env::var("LLVQ_KERNEL_DIR") {
            Err(_) => Ok((PLANES_CUH_EMBED.to_string(), PLANES_CU_EMBED.to_string(), None)),
            Ok(dir) => {
                let rd = |n: &str| {
                    std::fs::read_to_string(std::path::Path::new(&dir).join(n))
                        .map_err(|e| format!("LLVQ_KERNEL_DIR={dir} : {n} : {e}"))
                };
                let cuh = rd("llvq_planes.cuh")?;
                let cu = rd("planes.cu")?;
                Ok((cuh, cu, Some(dir)))
            }
        }
    }

    /// Same contract for the two Planes12x sources — the M2 arm's kernel.
    fn load_planes12_sources() -> Result<(String, String, Option<String>), String> {
        match std::env::var("LLVQ_KERNEL_DIR") {
            Err(_) => Ok((
                PLANES12_CUH_EMBED.to_string(),
                PLANES12_CU_EMBED.to_string(),
                None,
            )),
            Ok(dir) => {
                let rd = |n: &str| {
                    std::fs::read_to_string(std::path::Path::new(&dir).join(n))
                        .map_err(|e| format!("LLVQ_KERNEL_DIR={dir} : {n} : {e}"))
                };
                let cuh = rd("llvq_planes12.cuh")?;
                let cu = rd("planes12.cu")?;
                Ok((cuh, cu, Some(dir)))
            }
        }
    }

    /// One matrix's inputs, whatever they came from — same split as
    /// `bin/matvec`: everything downstream of `Src` is shared, so the
    /// synthetic and artifact runs differ only in their data.
    struct Src {
        name: String,
        d_out: usize,
        d_in: usize,
        indices: Vec<u64>,
        gains: Vec<u32>,
        centroids: [f32; 2],
        rscale: Vec<f32>,
        tail: Vec<f32>,
    }

    struct Mat {
        name: String,
        d_out: usize,
        d_in: usize,
        nblocks: usize,
        tail_w: usize,
        words: cudarc::driver::CudaSlice<u32>,
        bases: cudarc::driver::CudaSlice<u32>,
        pwords: cudarc::driver::CudaSlice<u32>,
        p12words: cudarc::driver::CudaSlice<u32>,
        exc_idx: cudarc::driver::CudaSlice<u32>,
        exc_words: cudarc::driver::CudaSlice<u32>,
        n_exc: usize,
        gscale: cudarc::driver::CudaSlice<f32>,
        rscale: cudarc::driver::CudaSlice<f32>,
        tail: cudarc::driver::CudaSlice<f32>,
        w16: cudarc::driver::CudaSlice<u16>,
        slot_bytes: u64,
        planes_bytes: u64,
        p12_bytes: u64,
        f16_bytes: u64,
        y_ref: Vec<f64>,
        y16_ref: Vec<f64>,
        scale: Vec<f64>,
    }

    fn worst_error(got: &[f32], want: &[f64], scale: &[f64]) -> f64 {
        got.iter()
            .zip(want)
            .zip(scale)
            .map(|((&g, &w), &s)| (g as f64 - w).abs() / s.max(1e-12))
            .fold(0.0, f64::max)
    }

    fn spread(mut v: Vec<f64>) -> (f64, f64, f64) {
        v.sort_by(f64::total_cmp);
        (v[0], v[v.len() / 2], v[v.len() - 1])
    }

    /// Same shape as `gpu.rs`'s private `row_grid`, replicated rather than
    /// exported: one warp per row, whole blocks only — the kernels have no
    /// bounds guard (a return before `__syncthreads()` deadlocks).
    fn row_grid(d_out: u32, threads: u32, shared: u32) -> cudarc::driver::LaunchConfig {
        assert_eq!(d_out % (threads / 32), 0, "rows must fill whole blocks");
        cudarc::driver::LaunchConfig {
            grid_dim: (d_out * 32 / threads, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: shared,
        }
    }

    /// `tv_planes(words, tab, gscale, rscale, tail, x, y, nblocks, tail_w)` —
    /// `tv_slot` minus the bases array, which Planes14 does not have.
    #[allow(clippy::too_many_arguments)]
    fn launch_planes(
        cuda: &Cuda,
        f: &cudarc::driver::CudaFunction,
        words: &cudarc::driver::CudaSlice<u32>,
        tab: &cudarc::driver::CudaSlice<u32>,
        gscale: &cudarc::driver::CudaSlice<f32>,
        rscale: &cudarc::driver::CudaSlice<f32>,
        tail: &cudarc::driver::CudaSlice<f32>,
        x: &cudarc::driver::CudaSlice<f32>,
        y: &mut cudarc::driver::CudaSlice<f32>,
        nblocks: u32,
        tail_w: u32,
        d_out: u32,
        threads: u32,
        shared: u32,
    ) -> Result<(), String> {
        let cfg = row_grid(d_out, threads, shared);
        let mut b = cuda.stream().launch_builder(f);
        b.arg(words).arg(tab).arg(gscale).arg(rscale).arg(tail).arg(x).arg(y)
            .arg(&nblocks).arg(&tail_w);
        unsafe { b.launch(cfg) }.map_err(|e| format!("tv_planes: {e}"))?;
        Ok(())
    }

    /// `tv_planes12x(words, exc_idx, exc_words, tab, gscale, rscale, tail, x,
    /// y, nblocks, tail_w, row_cta, n_exc)` — the overlay arm: `row_cta` CTAs
    /// of rows plus `ceil(n_exc/8)` CTAs of corrections, ONE launch. Both
    /// regions accumulate into `y` atomically, so `y` is zeroed first — an
    /// async memset on the same stream, deliberately inside the timed arm:
    /// it is part of what the layout costs.
    #[allow(clippy::too_many_arguments)]
    fn launch_planes12x(
        cuda: &Cuda,
        f: &cudarc::driver::CudaFunction,
        words: &cudarc::driver::CudaSlice<u32>,
        exc_idx: &cudarc::driver::CudaSlice<u32>,
        exc_words: &cudarc::driver::CudaSlice<u32>,
        tab: &cudarc::driver::CudaSlice<u32>,
        gscale: &cudarc::driver::CudaSlice<f32>,
        rscale: &cudarc::driver::CudaSlice<f32>,
        tail: &cudarc::driver::CudaSlice<f32>,
        x: &cudarc::driver::CudaSlice<f32>,
        y: &mut cudarc::driver::CudaSlice<f32>,
        nblocks: u32,
        tail_w: u32,
        n_exc: u32,
        d_out: u32,
        threads: u32,
        shared: u32,
    ) -> Result<(), String> {
        assert_eq!(d_out % (threads / 32), 0, "rows must fill whole blocks");
        let row_cta = d_out * 32 / threads;
        let exc_cta = n_exc.div_ceil(threads / 32);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (row_cta + exc_cta, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: shared,
        };
        cuda.stream()
            .memset_zeros(y)
            .map_err(|e| format!("memset y: {e}"))?;
        let mut b = cuda.stream().launch_builder(f);
        b.arg(words).arg(exc_idx).arg(exc_words).arg(tab).arg(gscale).arg(rscale)
            .arg(tail).arg(x).arg(y).arg(&nblocks).arg(&tail_w).arg(&row_cta).arg(&n_exc);
        unsafe { b.launch(cfg) }.map_err(|e| format!("tv_planes12x: {e}"))?;
        Ok(())
    }

    pub fn run() -> Result<(), String> {
        // Five parts, concatenated in dependency order: llvq_planes.cuh needs
        // llvq_slot.cuh (ClassRec, ext24), planes.cu needs matvec.cu
        // (warp_sum, TILE_COLS). NVRTC has no filesystem, the host includes.
        //
        // ⚠️ tv_slot and tv_f16 sit in a larger translation unit than in
        // `bin/matvec`: their register report below is the drift detector.
        let base = llvq_cuda::load_sources_many(&["llvq_slot.cuh", "matvec.cu"])?;
        let (pcuh, pcu, planes_overridden) = load_planes_sources()?;
        let (p12cuh, p12cu, planes12_overridden) = load_planes12_sources()?;
        let defines = format!("#define TILE_BLOCKS {TILE_BLOCKS}u\n");
        let parts = [
            defines.as_str(),
            base.parts[0].as_str(),
            base.parts[1].as_str(),
            pcuh.as_str(),
            pcu.as_str(),
            p12cuh.as_str(),
            p12cu.as_str(),
        ];
        let src = KernelSource::new(&parts);
        println!("source NVRTC : {} octets, sha256 {}", src.text.len(), src.sha256);
        if let Some(d) = &base.overridden_from {
            println!("  ⚠️ SOURCES Slot32 SURCHARGÉES depuis {d}");
        }
        if let Some(d) = &planes_overridden {
            println!("  ⚠️ SOURCES Planes14 SURCHARGÉES depuis {d}");
        }
        if let Some(d) = &planes12_overridden {
            println!("  ⚠️ SOURCES Planes12x SURCHARGÉES depuis {d}");
        }

        let cuda = Cuda::new(&src)?;
        let dev = cuda.device()?;
        println!(
            "\n{} — {} SM, L2 {:.1} Mo (lue), {} o de partagée par bloc",
            dev.name,
            dev.sm_count,
            dev.l2_bytes as f64 / 1e6,
            dev.shared_per_block
        );
        for name in ["tv_slot", "tv_f16", "tv_planes", "tv_planes12x"] {
            let r = cuda.report(name)?;
            println!(
                "  {:<10} {:>3} registres, {} o locaux, sm_{}",
                r.name, r.num_regs, r.local_bytes, r.binary_version
            );
            if r.local_bytes != 0 {
                return Err(format!("{name} : {} octets de spill", r.local_bytes));
            }
        }

        // Enough repetitions that the LIGHTEST arm streams past the L2 twice
        // over. Sizing on a heavier arm would let the light one replay from
        // cache. With the M2 arm the lower bound is Planes12x's 12-byte main
        // stream (its exceptions only add bytes), so size on 12, not 14.
        let one_pass: u64 = SHAPES
            .iter()
            .map(|&(_, o, i)| (o * (i / DIM)) as u64 * 12)
            .sum();
        let reps = (2 * dev.l2_bytes as u64 / one_pass + 1).max(2) as usize;

        let fd = FastDecoder::new();
        let table = ClassTable::new(&fd, 1);
        // The Slot32 arm's five-word window bound, re-asserted as every
        // consumer of slot_dot must.
        assert!(24 + table.worst_width_slot() <= 160, "fenêtre de 5 mots dépassée");
        // The Planes14 bound: three bit-planes address 8 levels; the layout
        // is only bijective while every class has at most 5.
        assert!(
            (0..table.n_entries()).all(|e| table.record(e).len <= 5),
            "une classe dépasse 5 niveaux : 3 plans ne suffisent plus"
        );

        let mut tab = vec![0u32; TABLE_ENTRIES * REC_WORDS];
        for e in 0..TABLE_ENTRIES {
            tab[e * REC_WORDS + REC_WORDS - 1] = 1;
        }
        for ci in 0..fd.n_classes() {
            let lv = fd.levels(ci);
            let norm = ((16 * lv.shell) as f64).sqrt();
            let base = (1 + ci) * REC_WORDS;
            for k in 0..lv.len {
                tab[base + k] = ((lv.values[k] as f64 / norm) as f32).to_bits();
            }
            tab[base + REC_WORDS - 1] = lv.len as u32;
        }
        let d_tab = cuda.up_u32(&tab)?;

        let mut rng = SplitMix64::new(0x6_D07);
        let x: Vec<f32> = (0..16384).map(|_| rng.next_gaussian() as f32).collect();
        let d_x = cuda.up_f32(&x)?;

        // The L = 5 swap of the Planes12x arm needs the exact searcher the
        // artifact was encoded with; built once, shared by every transcode.
        let searcher = Searcher::new();

        let build = |s: &Src| -> Result<Mat, String> {
            let (d_out, d_in) = (s.d_out, s.d_in);
            let nblocks = d_in / DIM;
            let tail_w = d_in % DIM;
            assert_eq!(d_out % 8, 0, "{}: CUDA lance des blocs entiers", s.name);
            assert!(d_in <= x.len(), "{}: d_in {d_in} dépasse l'activation", s.name);
            let rt = transcode(&fd, &table, &s.indices, &s.gains, Layout::Slot32)
                .map_err(|e| e.to_string())?;
            // The Planes14 stream: a bit-level bijection of the Slot32
            // content, proved block by block in the reference loop below.
            let planes_data = planes14_from_slot32(&rt, &table);
            // The M2 overlay: 12-byte main stream (L = 5 blocks swapped for
            // their best L ≤ 4 direction) + exact exception records. The
            // swap searches inside, threaded — the expensive build step.
            let p12 = transcode_planes12x(&fd, &table, &searcher, &s.indices, &s.gains)
                .map_err(|e| e.to_string())?;

            let mut w16 = vec![0u16; d_out * d_in];
            let mut y_ref = vec![0.0f64; d_out];
            let mut y16_ref = vec![0.0f64; d_out];
            let mut scale = vec![0.0f64; d_out];
            let nthreads = std::thread::available_parallelism().map_or(8, |n| n.get());
            let chunk = d_out.div_ceil(nthreads);
            std::thread::scope(|sc| {
                for (ci, (((w16c, yc), y16c), scc)) in w16
                    .chunks_mut(chunk * d_in)
                    .zip(y_ref.chunks_mut(chunk))
                    .zip(y16_ref.chunks_mut(chunk))
                    .zip(scale.chunks_mut(chunk))
                    .enumerate()
                {
                    let (rt, table, x, src, planes) = (&rt, &table, &x, &s, &planes_data);
                    let p12 = &p12;
                    sc.spawn(move || {
                        let mut wrow = vec![0.0f64; d_in];
                        for lr in 0..yc.len() {
                            let row = ci * chunk + lr;
                            wrow.fill(0.0);
                            for p in 0..nblocks {
                                let (pt, gain) = rt.decode_block(table, row * nblocks + p);
                                // The bijection proof: every block of the
                                // Planes14 stream decodes to exactly the
                                // point and gain the Slot32 stream carries.
                                let (ppt, pgain) =
                                    planes14_decode_block(planes, table, row * nblocks + p);
                                assert_eq!(
                                    (pt, gain),
                                    (ppt, pgain),
                                    "{}: bloc {} — Planes14 n'est pas une bijection de Slot32",
                                    src.name,
                                    row * nblocks + p
                                );
                                // The overlay proof: main stream + exception
                                // records reconstruct the exact block — the
                                // approximation must be invisible here.
                                let (xpt, xgain) = p12.decode_block(table, row * nblocks + p);
                                assert_eq!(
                                    (pt, gain),
                                    (xpt, xgain),
                                    "{}: bloc {} — l'overlay Planes12x ne reconstruit pas \
                                     l'exact",
                                    src.name,
                                    row * nblocks + p
                                );
                                if let Some(shell) =
                                    llvq_core::Leech::shell_index(&pt).filter(|&s| s > 0)
                                {
                                    let k = src.centroids[gain as usize] as f64
                                        * src.rscale[row] as f64
                                        / ((16 * shell) as f64).sqrt();
                                    for (i, &v) in pt.iter().enumerate() {
                                        wrow[p * DIM + i] = v as f64 * k;
                                    }
                                }
                            }
                            for t in 0..tail_w {
                                wrow[nblocks * DIM + t] = src.tail[row * tail_w + t] as f64;
                            }
                            let (mut a, mut b, mut ss) = (0.0, 0.0, 0.0);
                            for c in 0..d_in {
                                let xv = x[c] as f64;
                                let wv = wrow[c];
                                let hb = f16_bits(wv as f32);
                                w16c[lr * d_in + c] = hb;
                                a += wv * xv;
                                b += f16_to_f64(hb) * xv;
                                ss += (wv * xv).abs();
                            }
                            yc[lr] = a;
                            y16c[lr] = b;
                            scc[lr] = ss;
                        }
                    });
                }
            });

            let mut bytes = rt.data.clone();
            bytes.extend_from_slice(&[0u8; 20]); // the five-word read of the last block
            while !bytes.len().is_multiple_of(4) {
                bytes.push(0);
            }
            let words: Vec<u32> = bytes
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();

            let planes_bytes = planes_data.len() as u64
                + (d_out * tail_w) as u64 * 4
                + d_out as u64 * 4;
            let mut pbytes = planes_data;
            // The last block's four-word window reaches at most 2 bytes past
            // the stream; pad 4 and align, mirroring the slot padding.
            pbytes.extend_from_slice(&[0u8; 4]);
            while !pbytes.len().is_multiple_of(4) {
                pbytes.push(0);
            }
            let pwords: Vec<u32> = pbytes
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();

            // The M2 arm's three device arrays and its byte accounting:
            // main stream + exception indices + exception records + the two
            // Same tail/rscale terms as the other LLVQ arms. Upload paddings
            // are NOT billed: the slot arm ignores its 20-byte pad and the
            // planes arm its 4-byte pad, so billing ours here would mix two
            // byte accountings — the exact mistake the K-1 lot eliminated.
            let n_exc = p12.exc_idx.len();
            let p12_bytes = p12.data.len() as u64
                + n_exc as u64 * 4
                + p12.exc_data.len() as u64
                + (d_out * tail_w) as u64 * 4
                + d_out as u64 * 4;
            let pad12 = |mut b: Vec<u8>| -> Vec<u32> {
                b.extend_from_slice(&[0u8; 4]);
                while !b.len().is_multiple_of(4) {
                    b.push(0);
                }
                b.chunks_exact(4)
                    .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect()
            };
            let p12words: Vec<u32> = pad12(p12.data);
            let excwords: Vec<u32> = pad12(p12.exc_data);
            // cudarc refuses a zero-length upload; a matrix without any
            // L = 5 block gets a one-word dummy the kernel never reads
            // (n_exc == 0 spawns no correction CTA).
            let exc_idx_up: Vec<u32> = if p12.exc_idx.is_empty() {
                vec![0]
            } else {
                p12.exc_idx
            };

            Ok(Mat {
                name: s.name.clone(),
                d_out,
                d_in,
                nblocks,
                tail_w,
                words: cuda.up_u32(&words)?,
                bases: cuda.up_u32(&rt.bases)?,
                pwords: cuda.up_u32(&pwords)?,
                p12words: cuda.up_u32(&p12words)?,
                exc_idx: cuda.up_u32(&exc_idx_up)?,
                exc_words: cuda.up_u32(&excwords)?,
                n_exc,
                gscale: cuda.up_f32(&s.centroids)?,
                rscale: cuda.up_f32(&s.rscale)?,
                tail: cuda.up_f32(if s.tail.is_empty() { &[0.0f32] } else { &s.tail })?,
                w16: cuda.up_u16(&w16)?,
                slot_bytes: rt.data.len() as u64
                    + rt.bases.len() as u64 * 4
                    + (d_out * tail_w) as u64 * 4
                    + d_out as u64 * 4,
                planes_bytes,
                p12_bytes,
                f16_bytes: (d_out * d_in * 2) as u64,
                y_ref,
                y16_ref,
                scale,
            })
        };

        println!(
            "\nConstruction, transcodage Slot32, Planes14 et Planes12x (swap L = 5 → \
             L ≤ 4 inclus), preuves de bijection et d'overlay…"
        );
        let t0 = Instant::now();
        let mut mats = Vec::new();
        let mut n_weights = 0u64;
        let source;

        match std::env::args().nth(1) {
            // ---- the published model ----
            Some(path) => {
                let f = std::fs::File::open(&path).map_err(|e| format!("open {path}: {e}"))?;
                let mut r = std::io::BufReader::new(f);
                let h = llvq_artifact::read_header(&mut r).map_err(|e| e.to_string())?;
                println!("  {path} — {} matrices", h.matrices);
                source = format!("le modèle publié ({path})");
                for _ in 0..h.matrices {
                    let m = llvq_artifact::read_matrix_raw(&mut r).map_err(|e| e.to_string())?;
                    // Every decoder hard-codes one gain bit (`hdr >> 9`).
                    assert_eq!(
                        m.centroids.len(),
                        2,
                        "{}: les noyaux codent 1 bit de gain en dur",
                        m.name
                    );
                    n_weights += (m.d_out * m.d_in) as u64;
                    mats.push(build(&Src {
                        name: m.name.clone(),
                        d_out: m.d_out,
                        d_in: m.d_in,
                        indices: m.indices,
                        gains: m.gains,
                        centroids: [m.centroids[0] as f32, m.centroids[1] as f32],
                        rscale: m.row_scales.iter().map(|&v| v as f32).collect(),
                        tail: m.tail.iter().map(|&v| v as f32).collect(),
                    })?);
                }
            }
            // ---- synthetic, real shapes, repeated past the L2 ----
            None => {
                source = format!("{reps} répétitions synthétiques des 7 formes");
                for r in 0..reps {
                    for &(name, d_out, d_in) in &SHAPES {
                        let n = d_out * (d_in / DIM);
                        n_weights += (d_out * d_in) as u64;
                        mats.push(build(&Src {
                            name: format!("{name}#{r}"),
                            d_out,
                            d_in,
                            indices: (0..n).map(|_| 1 + rng.next() % N13).collect(),
                            gains: (0..n).map(|_| (rng.next() & 1) as u32).collect(),
                            centroids: GSCALE,
                            rscale: (0..d_out)
                                .map(|_| 0.5 + rng.next_gaussian().abs() as f32)
                                .collect(),
                            tail: (0..d_out * (d_in % DIM))
                                .map(|_| rng.next_gaussian() as f32)
                                .collect(),
                        })?);
                    }
                }
            }
        }
        let max_dout = mats.iter().map(|m| m.d_out).max().unwrap();
        let mut d_y = cuda.zeros_f32(max_dout)?;
        println!(
            "  {} matrices, {:.2} Md de poids, en {:.0} s — bijection Planes14 vérifiée \
             bloc par bloc",
            mats.len(),
            n_weights as f64 / 1e9,
            t0.elapsed().as_secs_f64()
        );

        let f_slot = cuda.func("tv_slot")?;
        let f_planes = cuda.func("tv_planes")?;
        let f_planes12x = cuda.func("tv_planes12x")?;
        let f_f16 = cuda.func("tv_f16")?;
        let shared = (TILE_BLOCKS * DIM * 4) as u32;

        let run_slot = |m: &Mat, y: &mut cudarc::driver::CudaSlice<f32>| -> Result<(), String> {
            cuda.launch_slot(
                &f_slot, &m.words, &m.bases, &d_tab, &m.gscale, &m.rscale, &m.tail, &d_x, y,
                m.nblocks as u32, m.tail_w as u32, m.d_out as u32, THREADS, shared,
            )
        };
        let run_planes = |m: &Mat, y: &mut cudarc::driver::CudaSlice<f32>| -> Result<(), String> {
            launch_planes(
                &cuda, &f_planes, &m.pwords, &d_tab, &m.gscale, &m.rscale, &m.tail, &d_x, y,
                m.nblocks as u32, m.tail_w as u32, m.d_out as u32, THREADS, shared,
            )
        };
        let run_planes12x =
            |m: &Mat, y: &mut cudarc::driver::CudaSlice<f32>| -> Result<(), String> {
                launch_planes12x(
                    &cuda, &f_planes12x, &m.p12words, &m.exc_idx, &m.exc_words, &d_tab,
                    &m.gscale, &m.rscale, &m.tail, &d_x, y, m.nblocks as u32,
                    m.tail_w as u32, m.n_exc as u32, m.d_out as u32, THREADS, shared,
                )
            };
        let run_f16 = |m: &Mat, y: &mut cudarc::driver::CudaSlice<f32>| -> Result<(), String> {
            cuda.launch_f16(&f_f16, &m.w16, &d_x, y, m.d_in as u32, m.d_out as u32, THREADS, shared)
        };

        println!("\nVérification de chaque ligne contre la référence f64…");
        let (mut ws, mut wp, mut wx, mut wf) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for m in &mats {
            run_slot(m, &mut d_y)?;
            cuda.sync()?;
            let got = cuda.down_f32(&d_y)?;
            let e = worst_error(&got[..m.d_out], &m.y_ref, &m.scale);
            assert!(e < TOL, "{} / Slot32 : {e:.2e}·Σ|w·x|", m.name);
            ws = ws.max(e);

            run_planes(m, &mut d_y)?;
            cuda.sync()?;
            let got = cuda.down_f32(&d_y)?;
            let e = worst_error(&got[..m.d_out], &m.y_ref, &m.scale);
            assert!(e < TOL, "{} / Planes14 : {e:.2e}·Σ|w·x|", m.name);
            wp = wp.max(e);

            // Against the SAME exact reference as the other LLVQ arms: this
            // passes only if the correction pass cancels the main stream's
            // L = 5 approximation on every one of the model's rows — the
            // proof of the overlay, before a single timing.
            run_planes12x(m, &mut d_y)?;
            cuda.sync()?;
            let got = cuda.down_f32(&d_y)?;
            let e = worst_error(&got[..m.d_out], &m.y_ref, &m.scale);
            assert!(e < TOL, "{} / Planes12x : {e:.2e}·Σ|w·x|", m.name);
            wx = wx.max(e);

            run_f16(m, &mut d_y)?;
            cuda.sync()?;
            let got = cuda.down_f32(&d_y)?;
            let e = worst_error(&got[..m.d_out], &m.y16_ref, &m.scale);
            assert!(e < TOL, "{} / FP16 : {e:.2e}·Σ|w·x|", m.name);
            wf = wf.max(e);
        }
        let rows: usize = mats.iter().map(|m| m.d_out).sum();
        let total_exc: u64 = mats.iter().map(|m| m.n_exc as u64).sum();
        let total_blocks: u64 = mats.iter().map(|m| (m.d_out * m.nblocks) as u64).sum();
        println!(
            "  {rows} lignes, seuil {TOL:.0e} — pires erreurs Slot32 {ws:.1e}, \
             Planes14 {wp:.1e}, Planes12x {wx:.1e} (overlay exact), FP16 {wf:.1e} ·Σ|w·x|"
        );
        println!(
            "  exceptions L = 5 : {total_exc} sur {total_blocks} blocs \
             ({:.4} %), corrigées dans le même lancement",
            total_exc as f64 * 100.0 / total_blocks as f64
        );

        // One pass = all matrices, one stream, in order — the layers' real
        // dependency. Wall clock around the pass plus a synchronize; the
        // four arms interleave inside each round. The Planes12x arm's pass
        // includes its per-matrix memset — part of what the layout costs.
        let mut times: [Vec<f64>; ARMS] =
            [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        for rep in 0..ROUNDS {
            for (arm, t_arm) in times.iter_mut().enumerate() {
                let t = Instant::now();
                for m in &mats {
                    match arm {
                        0 => run_slot(m, &mut d_y)?,
                        1 => run_planes(m, &mut d_y)?,
                        2 => run_planes12x(m, &mut d_y)?,
                        _ => run_f16(m, &mut d_y)?,
                    }
                }
                cuda.sync()?;
                let s = t.elapsed().as_secs_f64();
                if rep >= WARMUP {
                    t_arm.push(s);
                }
            }
        }
        let [t_slot, t_planes, t_p12, t_f16] = times;
        let per_round = |num: &[f64], den: &[f64]| -> Vec<f64> {
            num.iter().zip(den).map(|(a, b)| a / b).collect()
        };
        let (s_lo, s_md, s_hi) = spread(t_slot.clone());
        let (p_lo, p_md, p_hi) = spread(t_planes.clone());
        let (x_lo, x_md, x_hi) = spread(t_p12.clone());
        let (f_lo, f_md, f_hi) = spread(t_f16.clone());
        let (rs_lo, rs_md, rs_hi) = spread(per_round(&t_f16, &t_slot));
        let (rp_lo, rp_md, rp_hi) = spread(per_round(&t_f16, &t_planes));
        let (rx_lo, rx_md, rx_hi) = spread(per_round(&t_f16, &t_p12));
        let (sp_lo, sp_md, sp_hi) = spread(per_round(&t_slot, &t_planes));
        let (px_lo, px_md, px_hi) = spread(per_round(&t_planes, &t_p12));
        let sb: u64 = mats.iter().map(|m| m.slot_bytes).sum();
        let pb: u64 = mats.iter().map(|m| m.planes_bytes).sum();
        let xb: u64 = mats.iter().map(|m| m.p12_bytes).sum();
        let fb: u64 = mats.iter().map(|m| m.f16_bytes).sum();

        println!("\nUN TOKEN — {} matrices, un stream, quatre bras entrelacés", mats.len());
        println!("  {ROUNDS} rounds, {WARMUP} jetés ; les rapports sont formés ROUND PAR ROUND");
        println!("  {}", "-".repeat(80));
        println!(
            "  {:<22}{:>9}{:>9}{:>9}{:>9}{:>9}{:>9}",
            "format", "min ms", "méd ms", "max ms", "Go lus", "b/poids", "Go/s"
        );
        for (n, (lo, md, hi), b) in [
            ("FP16 (128 bits)", (f_lo, f_md, f_hi), fb),
            ("LLVQ Slot32", (s_lo, s_md, s_hi), sb),
            ("LLVQ Planes14", (p_lo, p_md, p_hi), pb),
            ("LLVQ Planes12x", (x_lo, x_md, x_hi), xb),
        ] {
            println!(
                "  {n:<22}{:>9.3}{:>9.3}{:>9.3}{:>9.2}{:>9.3}{:>9.0}",
                lo * 1e3,
                md * 1e3,
                hi * 1e3,
                b as f64 / 1e9,
                b as f64 * 8.0 / n_weights as f64,
                b as f64 / lo / 1e9
            );
        }
        println!("  {}", "-".repeat(80));
        println!("  Slot32    vs FP16     : {rs_md:.2}× [{rs_lo:.2}–{rs_hi:.2}]");
        println!("  Planes14  vs FP16     : {rp_md:.2}× [{rp_lo:.2}–{rp_hi:.2}]");
        println!("  Planes12x vs FP16     : {rx_md:.2}× [{rx_lo:.2}–{rx_hi:.2}]");
        println!(
            "  Planes14  vs Slot32   : {sp_md:.2}× [{sp_lo:.2}–{sp_hi:.2}]  (>1 = Planes14 \
             plus rapide, même contenu décodé)"
        );
        println!(
            "  Planes12x vs Planes14 : {px_md:.2}× [{px_lo:.2}–{px_hi:.2}]  (>1 = Planes12x \
             plus rapide ; memset + correction inclus, même y exact)"
        );
        println!("\n  source : {source}");
        let light = pb.min(xb);
        println!(
            "  {:.0} Mo distincts par passe sur le bras le plus léger ({}), soit \
             {:.1}× la L2 lue.\n  Sous 1× on mesurerait le cache et pas la DRAM — le piège \
             qui a rendu\n  optimiste toute mesure LLVQ antérieure au 2026-07-31.",
            light as f64 / 1e6,
            if xb <= pb { "Planes12x" } else { "Planes14" },
            light as f64 / dev.l2_bytes as f64
        );
        if std::env::args().nth(1).is_none() {
            println!(
                "  ⚠️ blocs SYNTHÉTIQUES, tirés uniformément sur la boule m ≤ 13 : le mélange\n  \
                 de classes d'un vrai artefact n'est pas exercé, donc les strides de groupe\n  \
                 du bras Slot32 et le trafic d'octets diffèrent du modèle publié. Ce rapport\n  \
                 mesure les NOYAUX — passer le chemin du .llvq en argument pour le modèle."
            );
        }
        // The tied lm_head, read once per token by every arm at the FP16
        // arm's measured rate — the constant that caps every ratio.
        let head_bytes = 389_070_848f64 * 2.0;
        let head_s = head_bytes / (fb as f64 / f_lo);
        println!(
            "\n  Avec le lm_head f16 non quantifié ({:.0} M poids, {:.2} ms au débit FP16\n  \
             mesuré, ajouté aux quatre bras) : Slot32 {:.2}×, Planes14 {:.2}×, \
             Planes12x {:.2}×\n  au lieu de {rs_md:.2}× / {rp_md:.2}× / {rx_md:.2}×. Normes, \
             activations, attention et\n  rotation ne sont mesurées ni ici ni là.",
            389_070_848f64 / 1e6,
            head_s * 1e3,
            (f_lo + head_s) / (s_lo + head_s),
            (f_lo + head_s) / (p_lo + head_s),
            (f_lo + head_s) / (x_lo + head_s)
        );
        println!(
            "\n  ⚠️ à ne JAMAIS comparer au chiffre Metal ligne à ligne, ni soustraire d'un\n  \
             run de bin/matvec : autres rounds, autre unité de traduction NVRTC. Chaque\n  \
             bras contre sa référence f64 ; les rapports se forment round par round,\n  \
             dans un même processus."
        );
        Ok(())
    }
}
