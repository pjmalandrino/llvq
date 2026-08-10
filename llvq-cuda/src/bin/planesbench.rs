//! Planes14 against Slot32 against the FP16 witness, on NVIDIA — plus the
//! M2 arm: `tv_planes12x`, the sparse overlay.
//!
//! `planesbench <model.llvq>` measures the published model: one token's worth
//! of linear algebra over its 252 projections, five arms interleaved in every
//! round — `tv_slot` (the published kernel, untouched), `tv_planes` (layout
//! E1a: uniform 14-byte records, no bases array, explicit bit-plane levels),
//! `tv_planes12x` (M2: a 12-byte main stream capped at 4 levels, the 5-level
//! blocks swapped for their best L ≤ 4 direction and corrected **exactly**
//! by an exception pass in the same launch — one memset + one atomicAdd per
//! row and per exception is the price, timed inside the arm), `tv_golay70`
//! (E2: a 9-byte main stream whose per-slot decode goes through the Golay
//! codeword rank — the residue-mod-4 plane of every block is a codeword —
//! with the E2 exception blocks holed to the origin and added back exactly
//! by the same-launch correction pass) and `tv_f16`.
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
//!
//! ## The A4 section, after the table: fusing what shares an input
//!
//! With a model path, a sixth and seventh arm run *after* the five-arm table,
//! in their own interleaved rounds: `tv_slot_seg` and `tv_planes_seg` over the
//! row-concatenation of q+k+v and of gate+up — 144 launches where there were
//! 252, and 768-block grids where k/v alone were 128. The published Slot32
//! delta is 0.803 ms (`docs/mesures/fusion-qkv-cuda-2026-08-05.txt`); the
//! Planes14 one is unknown, which is why **both** are re-timed here rather than
//! one being compared against a number from another job. Correctness first, and
//! bit-exact: a fused row runs the same blocks in the same order with the same
//! centroids, so nothing is reassociated and a tolerance would let a wrong
//! `gs_off` through. The concatenation itself (`src/seg_host.rs`) is proved on
//! the development machine by `tests/planes_segment_matches_unfused.rs`.
//!
//! Cost of the section: one extra transcode of the fusible matrices, and about
//! 4 GB of VRAM for the two fused streams on top of the ~15 GB the five arms
//! already hold. It is skipped entirely on the synthetic path.

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
    include!("../golay70_host.rs");
    include!("../seg_host.rs");

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
    /// Arm order inside a round, fixed: Slot32, Planes14, Planes12x,
    /// Golay70, FP16.
    const ARMS: usize = 5;

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
    const GOLAY_CUH_EMBED: &str = include_str!("../../kernels/llvq_golay.cuh");
    const GOLAY_CU_EMBED: &str = include_str!("../../kernels/golay70.cu");
    /// The segmented Planes14 kernel — item A4. One file, no header of its
    /// own: it reuses `llvq_planes.cuh`'s `planes_dot`. It lives apart from
    /// `planes.cu` on purpose — `llvq-llm/src/fused_cuda.rs` concatenates
    /// `llvq_planes.cuh` + `planes.cu` + `tv_planes_h.cu` as the *shipped*
    /// inference kernel, and appending to `planes.cu` would change that
    /// string's bytes and its sha256 for a bench's convenience.
    const PLANES_SEG_CU_EMBED: &str = include_str!("../../kernels/planes_seg.cu");

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

    /// Same contract for the two Golay70 sources — the E2 arm's kernel.
    fn load_golay_sources() -> Result<(String, String, Option<String>), String> {
        match std::env::var("LLVQ_KERNEL_DIR") {
            Err(_) => Ok((GOLAY_CUH_EMBED.to_string(), GOLAY_CU_EMBED.to_string(), None)),
            Ok(dir) => {
                let rd = |n: &str| {
                    std::fs::read_to_string(std::path::Path::new(&dir).join(n))
                        .map_err(|e| format!("LLVQ_KERNEL_DIR={dir} : {n} : {e}"))
                };
                let cuh = rd("llvq_golay.cuh")?;
                let cu = rd("golay70.cu")?;
                Ok((cuh, cu, Some(dir)))
            }
        }
    }

    /// Same contract for the one segmented source — the A4 arm's kernel.
    fn load_planes_seg_source() -> Result<(String, Option<String>), String> {
        match std::env::var("LLVQ_KERNEL_DIR") {
            Err(_) => Ok((PLANES_SEG_CU_EMBED.to_string(), None)),
            Ok(dir) => {
                let cu = std::fs::read_to_string(
                    std::path::Path::new(&dir).join("planes_seg.cu"),
                )
                .map_err(|e| format!("LLVQ_KERNEL_DIR={dir} : planes_seg.cu : {e}"))?;
                Ok((cu, Some(dir)))
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
        gwords: cudarc::driver::CudaSlice<u32>,
        gexc_idx: cudarc::driver::CudaSlice<u32>,
        gexc_words: cudarc::driver::CudaSlice<u32>,
        n_gexc: usize,
        gscale: cudarc::driver::CudaSlice<f32>,
        rscale: cudarc::driver::CudaSlice<f32>,
        tail: cudarc::driver::CudaSlice<f32>,
        w16: cudarc::driver::CudaSlice<u16>,
        slot_bytes: u64,
        planes_bytes: u64,
        p12_bytes: u64,
        g70_bytes: u64,
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

    /// `tv_planes_seg(words, tab, gscale, gs_off, rscale, tail, x, y, nblocks,
    /// tail_w)` — `tv_planes` over a row-concatenation of projections sharing
    /// an input, with one extra table naming each row's centroid pair.
    ///
    /// Same grid function as the unfused arm, on the *summed* `d_out`: a
    /// segmented matrix is one matrix, and the only thing the launch knows
    /// about the segments is `gs_off`. No memset, no atomic — rows partition
    /// the output, and `kernels/planes_seg.cu` argues that at length.
    #[allow(clippy::too_many_arguments)]
    fn launch_planes_seg(
        cuda: &Cuda,
        f: &cudarc::driver::CudaFunction,
        words: &cudarc::driver::CudaSlice<u32>,
        tab: &cudarc::driver::CudaSlice<u32>,
        gscale: &cudarc::driver::CudaSlice<f32>,
        gs_off: &cudarc::driver::CudaSlice<u32>,
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
        b.arg(words).arg(tab).arg(gscale).arg(gs_off).arg(rscale).arg(tail).arg(x).arg(y)
            .arg(&nblocks).arg(&tail_w);
        unsafe { b.launch(cfg) }.map_err(|e| format!("tv_planes_seg: {e}"))?;
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

    /// `tv_golay70(words, exc_idx, exc_words, cwtab, gtab, tab, gscale,
    /// rscale, tail, x, y, nblocks, tail_w, row_cta, n_exc)` — the E2 arm:
    /// same two-region grid and same zeroed-y protocol as `tv_planes12x`,
    /// with the codeword table and the dedicated class table added, and an
    /// exact-only correction (the main stream holds the origin at every
    /// exception). The memset is deliberately inside the timed arm.
    #[allow(clippy::too_many_arguments)]
    fn launch_golay70(
        cuda: &Cuda,
        f: &cudarc::driver::CudaFunction,
        words: &cudarc::driver::CudaSlice<u32>,
        exc_idx: &cudarc::driver::CudaSlice<u32>,
        exc_words: &cudarc::driver::CudaSlice<u32>,
        cwtab: &cudarc::driver::CudaSlice<u32>,
        gtab: &cudarc::driver::CudaSlice<u32>,
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
        b.arg(words).arg(exc_idx).arg(exc_words).arg(cwtab).arg(gtab).arg(tab)
            .arg(gscale).arg(rscale).arg(tail).arg(x).arg(y)
            .arg(&nblocks).arg(&tail_w).arg(&row_cta).arg(&n_exc);
        unsafe { b.launch(cfg) }.map_err(|e| format!("tv_golay70: {e}"))?;
        Ok(())
    }

    // ================= le bras concurrent : AWQ 4 bits, w4g128 =================
    //
    // Groupe de 128 canaux d'entrée, un facteur d'échelle binary16 et un zéro
    // de 4 bits par groupe, quatre bits par poids. C'est la configuration des
    // checkpoints AWQ officiels de Qwen3, et donc la seule que ce bras porte.
    const AWQ_GROUP: usize = 128;

    /// Les trois pas de ligne que `awq_gemv_g128` calcule lui-même. Recopiés
    /// ici parce que l'hôte doit allouer et remplir exactement ce que le noyau
    /// va lire — et deux d'entre eux sont **rembourrés**, ce qui n'est ni
    /// cosmétique ni devinable depuis les formes.
    fn awq_strides(d_in: usize) -> (usize, usize, usize) {
        let g = d_in / AWQ_GROUP; // groupes réels
        let npg = g.div_ceil(8); // `num_groups_packed`
        (d_in / 8, npg, npg * 8) // weight_w, zeros_w, sf_w
    }

    /// Octets qu'un bras AWQ lit pour une matrice, **dans la comptabilité du
    /// banc** : le flux et rien d'autre, rembourrage structurel compris parce
    /// que le noyau l'indexe vraiment.
    ///
    /// Pas de queue ni d'échelle de ligne ici, contrairement aux bras LLVQ :
    /// w4g128 quantifie *toutes* les colonnes, il n'a pas de politique de
    /// queue. C'est une différence réelle entre les formats, pas un oubli de
    /// facturation — et elle joue en faveur d'AWQ, donc elle se déclare.
    fn awq_bytes(d_out: usize, d_in: usize) -> u64 {
        let (ww, zw, sw) = awq_strides(d_in);
        (d_out * ww * 4 + d_out * zw * 4 + d_out * sw * 2) as u64
    }

    /// Quantifie une ligne de poids en w4g128, l'empaquette dans les trois
    /// tampons du noyau, et rend le produit scalaire EXACT en f64 de ce que le
    /// noyau va décoder contre l'activation **telle qu'il la voit**.
    ///
    /// ## Pourquoi la référence se calcule ici et pas ailleurs
    ///
    /// Le bras AWQ décode un contenu qui n'est celui d'aucun autre bras : ni
    /// `y_ref` (le contenu LLVQ) ni `y16_ref` (les mêmes poids en binary16) ne
    /// le décrivent. Il lui faut la sienne, sur le modèle du bras FP16 — et
    /// elle est **exacte par construction** : on connaît `q`, `scale` et
    /// `zero`, donc `scale·(q − zero)` est le poids que le noyau reconstruira,
    /// au bit près, sans approximation à borner.
    ///
    /// ⚠️ `xf` est l'activation **arrondie en binary16 puis réélargie**, parce
    /// que c'est ce que le noyau lit : ses entrées sont des `float4` de huit
    /// binary16. Utiliser l'activation f32 ici gonflerait l'erreur mesurée d'un
    /// écart qui n'est pas celui du bras.
    fn awq_quant_row(
        wrow: &[f64],
        xf: &[f64],
        d_in: usize,
        w_out: &mut [u32],
        z_out: &mut [u32],
        s_out: &mut [u16],
    ) -> f64 {
        let (ww, zw, sw) = awq_strides(d_in);
        let g = d_in / AWQ_GROUP;
        w_out[..ww].fill(0);
        z_out[..zw].fill(0);
        s_out[..sw].fill(0);
        let mut acc = 0.0f64;
        for gi in 0..g {
            let c0 = gi * AWQ_GROUP;
            let seg = &wrow[c0..c0 + AWQ_GROUP];
            let (lo, hi) = seg.iter().fold((f64::MAX, f64::MIN), |(a, b), &v| (a.min(v), b.max(v)));
            // Échelle et zéro asymétriques sur 4 bits, la convention d'AWQ :
            // `w ≈ scale·(q − zero)` avec `q ∈ [0, 15]`. Une échelle nulle
            // n'arrive que sur un groupe constant ; on la force non nulle pour
            // que la division existe, et le zéro absorbe alors la valeur.
            let mut scale = (hi - lo) / 15.0;
            if !(scale > 0.0) {
                scale = 1.0;
            }
            // L'échelle traverse un aller-retour binary16 AVANT de servir de
            // référence : c'est celle-là que le noyau lira, pas la f64.
            let scale = f16_to_f64(f16_bits(scale as f32));
            let zero = (-lo / scale).round().clamp(0.0, 15.0);
            s_out[gi] = f16_bits(scale as f32);
            let zq = zero as u32;
            z_out[gi / 8] |= (zq & 0xF) << (4 * (gi % 8));
            for (k, &w) in seg.iter().enumerate() {
                let q = (w / scale + zero).round().clamp(0.0, 15.0);
                let c = c0 + k;
                w_out[c / 8] |= ((q as u32) & 0xF) << (4 * (c % 8));
                acc += scale * (q - zero) * xf[c];
            }
        }
        acc
    }

    /// `awq_gemv_g128(inputs, weight, zeros, scaling_factors, outputs, IC, OC)`.
    ///
    /// La géométrie est **la leur**, recopiée de `gemv_forward_cuda` :
    /// `dim3 num_blocks(1, OC/4, B)` et `dim3 num_threads(32, 4)`, soit un warp
    /// par canal de sortie et quatre canaux par bloc. La changer ferait de ce
    /// bras une mesure de notre réglage, pas de leur noyau.
    #[allow(clippy::too_many_arguments)]
    fn launch_awq(
        cuda: &Cuda,
        f: &cudarc::driver::CudaFunction,
        xh: &cudarc::driver::CudaSlice<u16>,
        w: &cudarc::driver::CudaSlice<u32>,
        z: &cudarc::driver::CudaSlice<u32>,
        s: &cudarc::driver::CudaSlice<u16>,
        y: &mut cudarc::driver::CudaSlice<u16>,
        d_in: u32,
        d_out: u32,
    ) -> Result<(), String> {
        assert_eq!(d_out % 4, 0, "AWQ : OC/4 blocs, OC doit être multiple de 4");
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (1, d_out / 4, 1),
            block_dim: (32, 4, 1),
            shared_mem_bytes: 0,
        };
        let mut b = cuda.stream().launch_builder(f);
        b.arg(xh).arg(w).arg(z).arg(s).arg(y).arg(&d_in).arg(&d_out);
        unsafe { b.launch(cfg) }.map_err(|e| format!("awq_gemv_g128: {e}"))?;
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
        let (gcuh, gcu, golay_overridden) = load_golay_sources()?;
        let (segcu, seg_overridden) = load_planes_seg_source()?;
        let defines = format!("#define TILE_BLOCKS {TILE_BLOCKS}u\n");
        let parts = [
            defines.as_str(),
            base.parts[0].as_str(),
            base.parts[1].as_str(),
            pcuh.as_str(),
            pcu.as_str(),
            // After planes.cu: it uses `planes_dot` and `TILE_COLS`, and this
            // is the order `tests/host_planes.cpp` compiles them in.
            segcu.as_str(),
            p12cuh.as_str(),
            p12cu.as_str(),
            gcuh.as_str(),
            gcu.as_str(),
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
        if let Some(d) = &golay_overridden {
            println!("  ⚠️ SOURCES Golay70 SURCHARGÉES depuis {d}");
        }
        if let Some(d) = &seg_overridden {
            println!("  ⚠️ SOURCE Planes14 segmentée SURCHARGÉE depuis {d}");
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
        // The segmented kernels sit in the same translation unit as the five
        // measured ones, so their register reports are part of the same drift
        // detector: `tv_planes_seg` differs from `tv_planes` by one warp-uniform
        // load, and anything more than a register or two between them means the
        // compiler did something the source does not say.
        for name in [
            "tv_slot",
            "tv_slot_seg",
            "tv_f16",
            "tv_planes",
            "tv_planes_seg",
            "tv_planes12x",
            "tv_golay70",
        ] {
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
        // cache. With the E2 arm the lower bound is Golay70's 9-byte main
        // stream (its exceptions only add bytes), so size on 9, not 12.
        let one_pass: u64 = SHAPES
            .iter()
            .map(|&(_, o, i)| (o * (i / DIM)) as u64 * 9)
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

        // The Golay70 arm's two constant tables: the dedicated class table
        // (residue pairs / level values + mod-4 flags + coset flag) and the
        // canonical 4096-codeword table (16 KiB) the 12-bit rank resolves
        // through. Uploaded once, shared by every matrix.
        let golay = llvq_core::Golay::new();
        let g70cls = golay70_classes(&fd);
        let d_gtab = cuda.up_u32(&golay70_gpu_table(&g70cls))?;
        let d_cw = cuda.up_u32(&golay70_gpu_codewords(&golay))?;

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
            // The E2 arm: 9-byte main stream (exception blocks holed to the
            // origin) + exact exception records — pure table lookups plus a
            // Golay rank per block, no search.
            let g70 = golay70_transcode(&fd, &golay, &g70cls, &table, &s.indices, &s.gains);

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
                    let (g70, g70cls, golay) = (&g70, &g70cls, &golay);
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
                                // The E2 proof: main stream + exception
                                // records reconstruct the exact block, and
                                // the origin-holing is invisible here.
                                let (gpt, ggain) = golay70_decode_block(
                                    g70, g70cls, golay, table, row * nblocks + p,
                                );
                                assert_eq!(
                                    (pt, gain),
                                    (gpt, ggain),
                                    "{}: bloc {} — Golay70 ne reconstruit pas l'exact",
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

            // The E2 arm's three device arrays and its byte accounting:
            // 72 bits per block + 144 per exception + the same tail/rscale
            // terms as every LLVQ arm; upload paddings NOT billed — the
            // unified rule.
            let n_gexc = g70.exc_idx.len();
            let g70_bytes = g70.data.len() as u64
                + n_gexc as u64 * GOLAY70_EXC_BYTES as u64
                + (d_out * tail_w) as u64 * 4
                + d_out as u64 * 4;
            let gwords: Vec<u32> = pad12(g70.data);
            let gexcwords: Vec<u32> = pad12(g70.exc_data);
            let gexc_idx_up: Vec<u32> = if g70.exc_idx.is_empty() {
                vec![0]
            } else {
                g70.exc_idx
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
                gwords: cuda.up_u32(&gwords)?,
                gexc_idx: cuda.up_u32(&gexc_idx_up)?,
                gexc_words: cuda.up_u32(&gexcwords)?,
                n_gexc,
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
                g70_bytes,
                f16_bytes: (d_out * d_in * 2) as u64,
                y_ref,
                y16_ref,
                scale,
            })
        };

        println!(
            "\nConstruction, transcodage Slot32, Planes14, Planes12x (swap L = 5 → \
             L ≤ 4 inclus) et Golay70 (trous origine + exceptions E2), preuves de \
             bijection, d'overlay et de reconstruction exacte…"
        );
        let t0 = Instant::now();
        let mut mats = Vec::new();
        // Kept alive past `build` for the A4 fusion arm, which concatenates the
        // *indices* of q/k/v (and gate/up) before transcoding, not their
        // transcoded streams. Empty on the synthetic path, which is what turns
        // the fusion arm off there: `q_proj#0` and `q_proj#1` are different
        // repetitions, not a layer, and fusing them would be meaningless.
        // Costs ~1.8 GB of host RAM on the 4B — the same price `bin/matvec`
        // already pays for the Slot32 fusion arm.
        let mut srcs: Vec<Src> = Vec::new();
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
                    srcs.push(Src {
                        name: m.name.clone(),
                        d_out: m.d_out,
                        d_in: m.d_in,
                        indices: m.indices,
                        gains: m.gains,
                        centroids: [m.centroids[0] as f32, m.centroids[1] as f32],
                        rscale: m.row_scales.iter().map(|&v| v as f32).collect(),
                        tail: m.tail.iter().map(|&v| v as f32).collect(),
                    });
                }
                for s in &srcs {
                    mats.push(build(s)?);
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
        // ---- the fusion arm (item A4) ----
        //
        // q/k/v share an activation, and so do gate/up. Row-concatenating them
        // turns three grids of 512/128/128 blocks into one of 768 and removes
        // 108 launches a token (252 → 144). On **Slot32** that was measured at
        // 0.803 ms (`docs/mesures/fusion-qkv-cuda-2026-08-05.txt`); this arm
        // exists because that number does not transfer to Planes14 and cannot
        // be rescaled into it — see `kernels/planes_seg.cu`.
        //
        // Both layouts are fused here, and both are timed in the same process
        // and the same rounds. That is the whole methodological point: a
        // Planes14 delta compared against a Slot32 delta from another job would
        // be exactly the inter-process subtraction this repository has already
        // had to retract once.
        //
        // The concatenation is on the *indices*, before transcoding, so a fused
        // matrix is one ordinary stream — and `seg_concat` is proved on the
        // development machine by `tests/planes_segment_matches_unfused.rs`,
        // nine mutants deep.
        struct FusedMat {
            name: String,
            d_out: usize,
            nblocks: usize,
            tail_w: usize,
            words: cudarc::driver::CudaSlice<u32>,
            bases: cudarc::driver::CudaSlice<u32>,
            pwords: cudarc::driver::CudaSlice<u32>,
            gscale: cudarc::driver::CudaSlice<f32>,
            gs_off: cudarc::driver::CudaSlice<u32>,
            rscale: cudarc::driver::CudaSlice<f32>,
            tail: cudarc::driver::CudaSlice<f32>,
            slot_bytes: u64,
            planes_bytes: u64,
            /// Indices into `mats` of the matrices this one replaces, in row
            /// order — the fused output must equal their outputs concatenated.
            parts: Vec<usize>,
        }

        // 🕳️ **Le bras de fusion (A4) se construit APRÈS la table des cinq
        // bras, et non plus ici.** Il y était, et il déplaçait deux fois
        // l'objet que la table mesure — sans qu'aucune ligne de log le dise :
        //
        //   * `max_dout` chaînait les `d_out` fusionnés, donc `d_y` passait de
        //     9 728 à 19 456. Or `Planes12x` et `Golay70` font leur
        //     `memset_zeros(y)` sur **tout** le slice et **dans le
        //     chronomètre** (choix revendiqué, cf. `launch_planes12x`). Les
        //     deux bras à correction payaient donc ~9,8 Mo de plus par passe,
        //     ~+0,4 %, que dans le run qui a publié leurs chiffres — un biais
        //     systématique et unidirectionnel, sur exactement les deux bras
        //     dont on veut savoir s'ils ont dérivé.
        //   * ses ~2,9 Go de flux fusés restaient résidents pendant le
        //     chronométrage des cinq bras, portant l'occupation de ~15,3 à
        //     ~18,4 Go. La pression VRAM est le suspect nommé de la dispersion
        //     ×20 observée au passage de quatre à cinq bras.
        //
        // Aucun des deux n'est un bug : ce sont des effets de bord d'un lot
        // ajouté après coup (A4, commit 2d56cce, 2026-08-09). Mais ils
        // rendaient la table des cinq bras **incomparable au run qui l'a
        // publiée**, et c'est la seule chose que cette table doit garantir.
        let max_dout = mats.iter().map(|m| m.d_out).max().unwrap();
        let mut d_y = cuda.zeros_f32(max_dout)?;
        // Imprimé parce qu'il ne l'était pas : c'est ce silence qui a laissé
        // `d_y` doubler sans que personne ne puisse le lire dans un log.
        println!("  d_y : {max_dout} f32 — la table des cinq bras, et rien d'autre");
        println!(
            "  {} matrices, {:.2} Md de poids, en {:.0} s — bijection Planes14 vérifiée \
             bloc par bloc",
            mats.len(),
            n_weights as f64 / 1e9,
            t0.elapsed().as_secs_f64()
        );

        let f_slot = cuda.func("tv_slot")?;
        let f_slot_seg = cuda.func("tv_slot_seg")?;
        let f_planes = cuda.func("tv_planes")?;
        let f_planes_seg = cuda.func("tv_planes_seg")?;
        let f_planes12x = cuda.func("tv_planes12x")?;
        let f_golay70 = cuda.func("tv_golay70")?;
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
        let run_golay70 =
            |m: &Mat, y: &mut cudarc::driver::CudaSlice<f32>| -> Result<(), String> {
                launch_golay70(
                    &cuda, &f_golay70, &m.gwords, &m.gexc_idx, &m.gexc_words, &d_cw,
                    &d_gtab, &d_tab, &m.gscale, &m.rscale, &m.tail, &d_x, y,
                    m.nblocks as u32, m.tail_w as u32, m.n_gexc as u32, m.d_out as u32,
                    THREADS, shared,
                )
            };
        let run_f16 = |m: &Mat, y: &mut cudarc::driver::CudaSlice<f32>| -> Result<(), String> {
            cuda.launch_f16(&f_f16, &m.w16, &d_x, y, m.d_in as u32, m.d_out as u32, THREADS, shared)
        };
        let run_slot_seg =
            |fm: &FusedMat, y: &mut cudarc::driver::CudaSlice<f32>| -> Result<(), String> {
                cuda.launch_slot_seg(
                    &f_slot_seg, &fm.words, &fm.bases, &d_tab, &fm.gscale, &fm.gs_off,
                    &fm.rscale, &fm.tail, &d_x, y, fm.nblocks as u32, fm.tail_w as u32,
                    fm.d_out as u32, THREADS, shared,
                )
            };
        let run_planes_seg =
            |fm: &FusedMat, y: &mut cudarc::driver::CudaSlice<f32>| -> Result<(), String> {
                launch_planes_seg(
                    &cuda, &f_planes_seg, &fm.pwords, &d_tab, &fm.gscale, &fm.gs_off,
                    &fm.rscale, &fm.tail, &d_x, y, fm.nblocks as u32, fm.tail_w as u32,
                    fm.d_out as u32, THREADS, shared,
                )
            };

        println!("\nVérification de chaque ligne contre la référence f64…");
        let (mut ws, mut wp, mut wx, mut wg, mut wf) =
            (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
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

            // Same exact reference again: this passes only if the E2
            // correction pass restores every origin-holed exception block
            // exactly — the proof of the arm, before a single timing.
            run_golay70(m, &mut d_y)?;
            cuda.sync()?;
            let got = cuda.down_f32(&d_y)?;
            let e = worst_error(&got[..m.d_out], &m.y_ref, &m.scale);
            assert!(e < TOL, "{} / Golay70 : {e:.2e}·Σ|w·x|", m.name);
            wg = wg.max(e);

            run_f16(m, &mut d_y)?;
            cuda.sync()?;
            let got = cuda.down_f32(&d_y)?;
            let e = worst_error(&got[..m.d_out], &m.y16_ref, &m.scale);
            assert!(e < TOL, "{} / FP16 : {e:.2e}·Σ|w·x|", m.name);
            wf = wf.max(e);
        }
        let rows: usize = mats.iter().map(|m| m.d_out).sum();
        let total_exc: u64 = mats.iter().map(|m| m.n_exc as u64).sum();
        let total_gexc: u64 = mats.iter().map(|m| m.n_gexc as u64).sum();
        let total_blocks: u64 = mats.iter().map(|m| (m.d_out * m.nblocks) as u64).sum();
        println!(
            "  {rows} lignes, seuil {TOL:.0e} — pires erreurs Slot32 {ws:.1e}, \
             Planes14 {wp:.1e}, Planes12x {wx:.1e} (overlay exact), \
             Golay70 {wg:.1e} (E2 exact), FP16 {wf:.1e} ·Σ|w·x|"
        );
        println!(
            "  exceptions L = 5 : {total_exc} sur {total_blocks} blocs \
             ({:.4} %), corrigées dans le même lancement",
            total_exc as f64 * 100.0 / total_blocks as f64
        );
        println!(
            "  exceptions E2 (pair violant ou L = 5) : {total_gexc} sur {total_blocks} \
             blocs ({:.4} %), corrigées dans le même lancement",
            total_gexc as f64 * 100.0 / total_blocks as f64
        );

        // One pass = all matrices, one stream, in order — the layers' real
        // dependency. Wall clock around the pass plus a synchronize; the
        // five arms interleave inside each round. The Planes12x arm's pass
        // includes its per-matrix memset — part of what the layout costs.
        let mut times: [Vec<f64>; ARMS] =
            [Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        for rep in 0..ROUNDS {
            for (arm, t_arm) in times.iter_mut().enumerate() {
                let t = Instant::now();
                for m in &mats {
                    match arm {
                        0 => run_slot(m, &mut d_y)?,
                        1 => run_planes(m, &mut d_y)?,
                        2 => run_planes12x(m, &mut d_y)?,
                        3 => run_golay70(m, &mut d_y)?,
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
        let [t_slot, t_planes, t_p12, t_g70, t_f16] = times;
        let per_round = |num: &[f64], den: &[f64]| -> Vec<f64> {
            num.iter().zip(den).map(|(a, b)| a / b).collect()
        };
        let (s_lo, s_md, s_hi) = spread(t_slot.clone());
        let (p_lo, p_md, p_hi) = spread(t_planes.clone());
        let (x_lo, x_md, x_hi) = spread(t_p12.clone());
        let (g_lo, g_md, g_hi) = spread(t_g70.clone());
        let (f_lo, f_md, f_hi) = spread(t_f16.clone());
        let (rs_lo, rs_md, rs_hi) = spread(per_round(&t_f16, &t_slot));
        let (rp_lo, rp_md, rp_hi) = spread(per_round(&t_f16, &t_planes));
        let (rx_lo, rx_md, rx_hi) = spread(per_round(&t_f16, &t_p12));
        let (rg_lo, rg_md, rg_hi) = spread(per_round(&t_f16, &t_g70));
        let (sp_lo, sp_md, sp_hi) = spread(per_round(&t_slot, &t_planes));
        let (px_lo, px_md, px_hi) = spread(per_round(&t_planes, &t_p12));
        let (xg_lo, xg_md, xg_hi) = spread(per_round(&t_p12, &t_g70));
        let sb: u64 = mats.iter().map(|m| m.slot_bytes).sum();
        let pb: u64 = mats.iter().map(|m| m.planes_bytes).sum();
        let xb: u64 = mats.iter().map(|m| m.p12_bytes).sum();
        let gb: u64 = mats.iter().map(|m| m.g70_bytes).sum();
        let fb: u64 = mats.iter().map(|m| m.f16_bytes).sum();

        println!("\nUN TOKEN — {} matrices, un stream, cinq bras entrelacés", mats.len());
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
            ("LLVQ Golay70", (g_lo, g_md, g_hi), gb),
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
        println!("  Golay70   vs FP16     : {rg_md:.2}× [{rg_lo:.2}–{rg_hi:.2}]");
        println!(
            "  Planes14  vs Slot32   : {sp_md:.2}× [{sp_lo:.2}–{sp_hi:.2}]  (>1 = Planes14 \
             plus rapide, même contenu décodé)"
        );
        println!(
            "  Planes12x vs Planes14 : {px_md:.2}× [{px_lo:.2}–{px_hi:.2}]  (>1 = Planes12x \
             plus rapide ; memset + correction inclus, même y exact)"
        );
        println!(
            "  Golay70   vs Planes12x: {xg_md:.2}× [{xg_lo:.2}–{xg_hi:.2}]  (>1 = Golay70 \
             plus rapide ; memset + correction inclus, même y exact)"
        );
        println!("\n  source : {source}");
        let light = pb.min(xb).min(gb);
        println!(
            "  {:.0} Mo distincts par passe sur le bras le plus léger ({}), soit \
             {:.1}× la L2 lue.\n  Sous 1× on mesurerait le cache et pas la DRAM — le piège \
             qui a rendu\n  optimiste toute mesure LLVQ antérieure au 2026-07-31.",
            light as f64 / 1e6,
            if gb <= pb.min(xb) {
                "Golay70"
            } else if xb <= pb {
                "Planes12x"
            } else {
                "Planes14"
            },
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
             mesuré, ajouté aux cinq bras) : Slot32 {:.2}×, Planes14 {:.2}×, \
             Planes12x {:.2}×, Golay70 {:.2}×\n  au lieu de {rs_md:.2}× / {rp_md:.2}× / \
             {rx_md:.2}× / {rg_md:.2}×. Normes, activations, attention et\n  rotation ne \
             sont mesurées ni ici ni là.",
            389_070_848f64 / 1e6,
            head_s * 1e3,
            (f_lo + head_s) / (s_lo + head_s),
            (f_lo + head_s) / (p_lo + head_s),
            (f_lo + head_s) / (x_lo + head_s),
            (f_lo + head_s) / (g_lo + head_s)
        );
        println!(
            "\n  ⚠️ à ne JAMAIS comparer au chiffre Metal ligne à ligne, ni soustraire d'un\n  \
             run de bin/matvec : autres rounds, autre unité de traduction NVRTC. Chaque\n  \
             bras contre sa référence f64 ; les rapports se forment round par round,\n  \
             dans un même processus."
        );

        // ---- A4 : la fusion q+k+v et gate+up, justesse d'abord, coût ensuite ----
        //
        // Construite ICI, après la table des cinq bras : ni ses ~2,9 Go ni son
        // `d_out` doublé ne doivent exister pendant que la table se mesure
        // (voir le long commentaire au-dessus de `max_dout`).
        let mut fused: Vec<FusedMat> = Vec::new();
        if !srcs.is_empty() {
            let names: Vec<&str> = srcs.iter().map(|s| s.name.as_str()).collect();
            let groups = seg_groups(&names)?;
            let t_fuse = Instant::now();
            for (key, idx) in &groups {
                let sp: Vec<SegPart<'_>> = idx
                    .iter()
                    .map(|&i| {
                        let s = &srcs[i];
                        SegPart {
                            d_out: s.d_out,
                            d_in: s.d_in,
                            indices: &s.indices,
                            gains: &s.gains,
                            centroids: s.centroids,
                            rscale: &s.rscale,
                            tail: &s.tail,
                        }
                    })
                    .collect();
                let seg = seg_concat(&sp);
                // The twin of `build`'s guard: a fused matrix reads the same
                // shared activation, and nothing else would notice it reading
                // past the end of it.
                assert!(
                    seg.d_in <= x.len(),
                    "{key}: d_in {} dépasse l'activation",
                    seg.d_in
                );
                // Slot32 first, then Planes14 off it — the same two-step every
                // unfused matrix goes through, so the two arms differ in
                // geometry and in nothing else.
                let rt = transcode(&fd, &table, &seg.indices, &seg.gains, Layout::Slot32)
                    .map_err(|e| e.to_string())?;
                let pdata = planes14_from_slot32(&rt, &table);
                let slot_bytes = rt.data.len() as u64
                    + rt.bases.len() as u64 * 4
                    + (seg.d_out * seg.tail_w) as u64 * 4
                    + seg.d_out as u64 * 4;
                let planes_bytes = pdata.len() as u64
                    + (seg.d_out * seg.tail_w) as u64 * 4
                    + seg.d_out as u64 * 4;

                let mut sbytes = rt.data.clone();
                sbytes.extend_from_slice(&[0u8; 20]); // the five-word read of the last block
                while !sbytes.len().is_multiple_of(4) {
                    sbytes.push(0);
                }
                let swords: Vec<u32> = sbytes
                    .chunks_exact(4)
                    .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                let mut pbytes = pdata;
                pbytes.extend_from_slice(&[0u8; 4]); // the four-word read of the last block
                while !pbytes.len().is_multiple_of(4) {
                    pbytes.push(0);
                }
                let pwords: Vec<u32> = pbytes
                    .chunks_exact(4)
                    .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();

                fused.push(FusedMat {
                    name: key.clone(),
                    d_out: seg.d_out,
                    nblocks: seg.nblocks,
                    tail_w: seg.tail_w,
                    words: cuda.up_u32(&swords)?,
                    bases: cuda.up_u32(&rt.bases)?,
                    pwords: cuda.up_u32(&pwords)?,
                    gscale: cuda.up_f32(&seg.centroids)?,
                    gs_off: cuda.up_u32(&seg.gs_off)?,
                    rscale: cuda.up_f32(&seg.rscale)?,
                    tail: cuda
                        .up_f32(if seg.tail.is_empty() { &[0.0f32] } else { &seg.tail })?,
                    slot_bytes,
                    planes_bytes,
                    parts: idx.clone(),
                });
            }
            println!(
                "\n  fusion : {} groupes ({} matrices → {}), transcodés en {:.0} s",
                fused.len(),
                fused.iter().map(|f| f.parts.len()).sum::<usize>(),
                fused.len(),
                t_fuse.elapsed().as_secs_f64()
            );
        }

        // Un tampon de sortie propre à cette section : un bras fusionné écrit
        // jusqu'à 19 456 lignes, et c'est précisément ce qu'on refuse d'imposer
        // au `d_y` de la table. Les deux sections ne partagent plus rien.
        let a4_dout = fused
            .iter()
            .map(|f| f.d_out)
            .chain(std::iter::once(max_dout))
            .max()
            .expect("la chaîne porte au moins max_dout");
        drop(d_y);
        let mut d_y = cuda.zeros_f32(a4_dout)?;

        if !fused.is_empty() {
            // A fused row runs the same blocks, in the same order, with the
            // same centroids as the unfused row it came from — nothing is
            // reassociated — so the outputs must agree BIT FOR BIT. A tolerance
            // here would let a wrong `gs_off` through: swapping two segments'
            // centroids moves some rows by ~2× and leaves the rest untouched,
            // which a global epsilon on a different row still passes. The
            // Slot32 arm established that this equality is achievable on this
            // card and this compiler (921 600 rows, 2026-08-05); the Planes14
            // arm is the same claim about `tv_planes_seg` and is *not*
            // established until this block runs.
            println!("\n  Fusion (A4) — sortie contre les matrices non fusionnées");
            for fm in &fused {
                for layout in 0..2 {
                    if layout == 0 {
                        run_slot_seg(fm, &mut d_y)?
                    } else {
                        run_planes_seg(fm, &mut d_y)?
                    }
                    cuda.sync()?;
                    let got = cuda.down_f32(&d_y)?;
                    let mut at = 0usize;
                    for &pi in &fm.parts {
                        let m = &mats[pi];
                        if layout == 0 {
                            run_slot(m, &mut d_y)?
                        } else {
                            run_planes(m, &mut d_y)?
                        }
                        cuda.sync()?;
                        let want = cuda.down_f32(&d_y)?;
                        if got[at..at + m.d_out] != want[..m.d_out] {
                            let bad =
                                (0..m.d_out).find(|&r| got[at + r] != want[r]).unwrap_or(0);
                            return Err(format!(
                                "{} / {} / {} : ligne {bad} vaut {} fusionnée contre {} \
                                 séparée",
                                fm.name,
                                m.name,
                                if layout == 0 { "Slot32" } else { "Planes14" },
                                got[at + bad],
                                want[bad]
                            ));
                        }
                        at += m.d_out;
                    }
                }
            }
            println!(
                "  {} groupes, {} lignes — identiques AU BIT près, sur Slot32 ET sur Planes14",
                fused.len(),
                fused.iter().map(|f| f.d_out).sum::<usize>()
            );

            // Cost. Four arms, interleaved in every round, same process: LLVQ
            // against LLVQ on each layout, and the two layouts against each
            // other. No FP16 arm, deliberately — fusing the FP16 witness would
            // mean holding a second copy of 7.27 GB of f16 weights, so a
            // fused-LLVQ / unfused-FP16 ratio would credit the format for a
            // geometry change. Every number below is a DELTA, not a ratio.
            let mut tf: [Vec<f64>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
            for rep in 0..ROUNDS {
                for (arm, ta) in tf.iter_mut().enumerate() {
                    let tin = Instant::now();
                    match arm {
                        // 0/2: everything separate. 1/3: the fusible groups
                        // fused, o_proj and down_proj exactly as before.
                        0 => {
                            for m in &mats {
                                run_slot(m, &mut d_y)?;
                            }
                        }
                        1 => {
                            for fm in &fused {
                                run_slot_seg(fm, &mut d_y)?;
                            }
                            for (i, m) in mats.iter().enumerate() {
                                if !fused.iter().any(|f| f.parts.contains(&i)) {
                                    run_slot(m, &mut d_y)?;
                                }
                            }
                        }
                        2 => {
                            for m in &mats {
                                run_planes(m, &mut d_y)?;
                            }
                        }
                        _ => {
                            for fm in &fused {
                                run_planes_seg(fm, &mut d_y)?;
                            }
                            for (i, m) in mats.iter().enumerate() {
                                if !fused.iter().any(|f| f.parts.contains(&i)) {
                                    run_planes(m, &mut d_y)?;
                                }
                            }
                        }
                    }
                    cuda.sync()?;
                    let s = tin.elapsed().as_secs_f64();
                    if rep >= WARMUP {
                        ta.push(s);
                    }
                }
            }
            let [ts, tss, tp, tps] = tf;
            // Deltas formed ROUND BY ROUND, exactly as the ratios are: a
            // difference of two minima taken from rounds that never coexisted
            // is the mistake this repository documents.
            let ds: Vec<f64> = ts.iter().zip(&tss).map(|(a, b)| (a - b) * 1e3).collect();
            let dp: Vec<f64> = tp.iter().zip(&tps).map(|(a, b)| (a - b) * 1e3).collect();
            let rr: Vec<f64> = dp.iter().zip(&ds).map(|(a, b)| a / b).collect();
            let (_, s_sep, _) = spread(ts);
            let (_, s_fus, _) = spread(tss);
            let (_, p_sep, _) = spread(tp);
            let (_, p_fus, _) = spread(tps);
            let (ds_lo, ds_md, ds_hi) = spread(ds);
            let (dp_lo, dp_md, dp_hi) = spread(dp);
            let (rr_lo, rr_md, rr_hi) = spread(rr);

            let n_sep = mats.len();
            let n_fus = fused.len()
                + mats
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| !fused.iter().any(|f| f.parts.contains(i)))
                    .count();
            let unfused_bytes = |sel: fn(&Mat) -> u64| -> u64 {
                mats.iter().map(sel).sum::<u64>()
            };
            let fused_bytes = |fsel: fn(&FusedMat) -> u64, sel: fn(&Mat) -> u64| -> u64 {
                fused.iter().map(fsel).sum::<u64>()
                    + mats
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| !fused.iter().any(|f| f.parts.contains(i)))
                        .map(|(_, m)| sel(m))
                        .sum::<u64>()
            };
            let sb_sep = unfused_bytes(|m| m.slot_bytes);
            let sb_fus = fused_bytes(|f| f.slot_bytes, |m| m.slot_bytes);
            let pb_sep = unfused_bytes(|m| m.planes_bytes);
            let pb_fus = fused_bytes(|f| f.planes_bytes, |m| m.planes_bytes);

            println!(
                "\n  Coût — {ROUNDS} rounds, {WARMUP} jetés, QUATRE bras entrelacés, \
                 médianes\n  {}",
                "-".repeat(78)
            );
            println!(
                "  {:<34}{:>10.3} ms   {n_sep} lancements",
                "Slot32, matrices séparées",
                s_sep * 1e3
            );
            println!(
                "  {:<34}{:>10.3} ms   {n_fus} lancements",
                "Slot32, q+k+v et gate+up fusés",
                s_fus * 1e3
            );
            println!(
                "  {:<34}{:>10.3} ms   {n_sep} lancements",
                "Planes14, matrices séparées",
                p_sep * 1e3
            );
            println!(
                "  {:<34}{:>10.3} ms   {n_fus} lancements",
                "Planes14, q+k+v et gate+up fusés",
                p_fus * 1e3
            );
            println!("  {}", "-".repeat(78));
            println!(
                "  gain Slot32   : {ds_md:.3} ms [{ds_lo:.3}–{ds_hi:.3}]  ({:.1} %)",
                100.0 * ds_md / (s_sep * 1e3)
            );
            println!(
                "  gain Planes14 : {dp_md:.3} ms [{dp_lo:.3}–{dp_hi:.3}]  ({:.1} %)",
                100.0 * dp_md / (p_sep * 1e3)
            );
            println!(
                "  Planes14 / Slot32 sur le gain : {rr_md:.2}× [{rr_lo:.2}–{rr_hi:.2}]\n  \
                 ⚠️ rapport de deux DIFFÉRENCES : sa dispersion est celle des deux \
                 numérateurs\n  cumulée, donc bien plus large que celle des rapports du \
                 tableau ci-dessus. Le\n  lire comme un ordre de grandeur, jamais à deux \
                 décimales. Ce sont les deux\n  lignes « gain » qui portent le résultat."
            );
            println!(
                "  repère : 108 lancements en moins × 3,63 µs mesurés (a3-graph-2026-08-06)\n  \
                 = 0,392 ms, indépendants du layout. Ce qui dépasse est l'occupation."
            );
            println!(
                "  octets lus — Slot32   : {:.3} Go fusé contre {:.3} séparé ({:+.2} %)\n  \
                 octets lus — Planes14 : {:.3} Go fusé contre {:.3} séparé ({:+.2} %)",
                sb_fus as f64 / 1e9,
                sb_sep as f64 / 1e9,
                100.0 * (sb_fus as f64 - sb_sep as f64) / sb_sep as f64,
                pb_fus as f64 / 1e9,
                pb_sep as f64 / 1e9,
                100.0 * (pb_fus as f64 - pb_sep as f64) / pb_sep as f64
            );
            println!(
                "  Slot32 peut bouger : son stride est le plus large enregistrement d'un\n  \
                 groupe de 32, et la concaténation regroupe aux frontières de segment. \
                 Planes14\n  ne le peut pas — 14 octets par bloc, sans table de bases — et \
                 le +0,00 % ci-dessus\n  est donc une vérification, pas une mesure \
                 (tests/planes_segment_matches_unfused.rs).\n  Un gain d'octets serait un \
                 confondant, pas un bonus."
            );
            println!(
                "\n  ⚠️ CE BLOC NE PRODUIT AUCUN RAPPORT CONTRE LE FP16, et n'en autorise\n  \
                 aucun : le bras FP16 souffre lui aussi du sous-remplissage sur k/v et \
                 gagnerait\n  lui aussi à la fusion. Seuls les deux DELTAS LLVQ → LLVQ \
                 ci-dessus sont mesurés."
            );
        }
        Ok(())
    }
}
