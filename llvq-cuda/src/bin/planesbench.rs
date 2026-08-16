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
//! Since the v2 campaign the bench carries SEVEN arms: the six above (where
//! `tv_golay70` is the v2 decoder) plus `tv_golay70_v1`, the frozen witness
//! copy of the published Golay70 decode (`kernels/golay70_v1.cu`), which
//! shares the v2's device buffers — not one stored byte differs between the
//! two, so the witness arm costs zero VRAM.
//!
//! ## `LLVQ_BENCH_ARMS` — the arm selector (lot B, écart É1)
//!
//! Semicolon-separated phases of comma-separated arm names (see
//! `llvq_cuda::arms` for the whole contract). One process can therefore run
//! a control table and a full table back to back — the §4 rule of
//! `proofs/preregistration-2026-08-10.md` — with, during each phase, ONLY
//! that phase's buffers resident: a deselected arm builds no transcode, no
//! device buffer, no verification, no timing, no table row. The NVRTC
//! translation unit and the register report never change with the
//! selection; only buffers and dispatch do. After a multi-phase run the
//! bench prints `Δ_contrôle` — the largest relative drift of the common
//! arms between consecutive phases, the number §4's decision rule needs.
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
    use llvq_cuda::arms::{self, ArmSet};
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
    use llvq_cuda::TILE_BLOCKS;
    const THREADS: u32 = 256;
    const GSCALE: [f32; 2] = [0.625, 1.375];
    const TABLE_ENTRIES: usize = 512;
    const REC_WORDS: usize = 6;
    const TOL: f64 = 1e-5;
    /// Le seuil du bras concurrent, plus lâche d'un facteur ~100, et pour une
    /// raison de format et non de complaisance : `awq_gemv_g128` écrit sa
    /// sortie en **binary16** (`f2h`, dernière ligne de `awq_gemv.cu`) là où
    /// les cinq bras maison rendent du f32. Une sortie binary16 porte ~2⁻¹¹
    /// d'erreur relative par construction, donc exiger 1e-5 ici ferait échouer
    /// un noyau parfaitement juste. Sa RÉFÉRENCE, elle, reste exacte :
    /// `scale·(q − zero)` est ce que le noyau reconstruit au bit près.
    const AWQ_TOL: f64 = 1e-3;
    const ROUNDS: usize = 7;
    const WARMUP: usize = 2;
    // Arm order inside a round: `arms::ARM_NAMES`, the registration order —
    // Slot32, Planes14, Planes12x, Golay70 v1, FP16, AWQ, Golay70 v2.
    //
    // ⚠️ L'ordre est celui de l'ENREGISTREMENT, et un bras ajouté l'est
    // TOUJOURS EN DERNIER — insérer un bras au milieu changerait l'ordre de
    // dispatch des bras existants dans chaque round, donc l'objet que la
    // table publiée mesure. AWQ (2026-08-10) puis Golay70 v2 (2026-08-11)
    // suivent tous deux cette règle. Une sélection `LLVQ_BENCH_ARMS` ne
    // change jamais cet ordre : elle saute des bras, elle n'en déplace pas.

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
    /// Le bras concurrent. Il dépend de `h2f`/`f2h`, donc de `matvec.cu`,
    /// donc il est concaténé APRÈS lui — et en dernier, pour qu'ajouter un
    /// concurrent ne réordonne jamais les fragments des bras maison.
    const AWQ_CU_EMBED: &str = include_str!("../../kernels/awq_gemv.cu");
    /// Le bras témoin : la copie GELÉE du décodeur Golay70 publié (v1),
    /// symboles renommés, tampons partagés avec la v2. Concaténé après AWQ —
    /// le bras ajouté en dernier ne réordonne aucun fragment existant.
    const GOLAY_V1_CU_EMBED: &str = include_str!("../../kernels/golay70_v1.cu");

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
    /// Même contrat que les autres : sans la variable, le texte embarqué ;
    /// avec, le fichier du répertoire, annoncé bruyamment. C'est aussi le
    /// chemin par lequel un noyau qu'on ne peut pas vendoriser — QTIP est en
    /// GPL-3 — entrerait sans être commité.
    fn load_awq_source() -> Result<(String, Option<String>), String> {
        match std::env::var("LLVQ_KERNEL_DIR") {
            Err(_) => Ok((AWQ_CU_EMBED.to_string(), None)),
            Ok(dir) => {
                let cu =
                    std::fs::read_to_string(std::path::Path::new(&dir).join("awq_gemv.cu"))
                        .map_err(|e| format!("LLVQ_KERNEL_DIR={dir} : awq_gemv.cu : {e}"))?;
                Ok((cu, Some(dir)))
            }
        }
    }

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

    /// Même contrat pour la copie gelée v1 — surcharger le TÉMOIN est
    /// possible mais annoncé aussi bruyamment que les autres : un témoin
    /// silencieusement remplacé ne témoigne plus de rien.
    fn load_golay_v1_source() -> Result<(String, Option<String>), String> {
        match std::env::var("LLVQ_KERNEL_DIR") {
            Err(_) => Ok((GOLAY_V1_CU_EMBED.to_string(), None)),
            Ok(dir) => {
                let cu = std::fs::read_to_string(
                    std::path::Path::new(&dir).join("golay70_v1.cu"),
                )
                .map_err(|e| format!("LLVQ_KERNEL_DIR={dir} : golay70_v1.cu : {e}"))?;
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

    /// Un tampon dont le bras peut n'entrer qu'à une phase ultérieure : les
    /// octets sont préparés une seule fois à la construction (le travail
    /// hôte — références, transcodes — ne se refait pas entre phases), mais
    /// ils ne montent sur la carte qu'au début de la première phase qui
    /// chronomètre le bras. Pendant une phase, les FLUX résidents (les Go)
    /// sont donc exactement ceux des bras de cette phase — l'invariant de
    /// résidence que le contrôle du §4 du pré-enregistrement doit restituer.
    ///
    /// ⚠️ Périmètre exact, pour que l'énoncé ne sur-revendique pas : quatre
    /// tampons CONSTANTS suivent l'UNION des phases et non la phase
    /// courante — `d_gtab`/`d_cw` (~32 Kio, dès qu'un bras golay70 est
    /// sélectionné quelque part), `d_xh` (32 Kio) et `d_yh` (~19 Kio, dès
    /// qu'awq l'est). Ils ne passent pas par cet étage : ~83 Kio au total
    /// contre ~15 Go de flux, en deçà de toute résolution du banc, et
    /// déclarés ici plutôt que niés.
    enum Staged<T> {
        Host(Vec<T>),
        Dev(cudarc::driver::CudaSlice<T>),
    }

    impl<T> Staged<T> {
        /// Le tampon device — panique nominative si le bras n'a pas été
        /// téléversé : un lancement sur un bras non monté est un bug de
        /// séquencement des phases, jamais un cas à rattraper en silence.
        fn dev(&self) -> &cudarc::driver::CudaSlice<T> {
            match self {
                Staged::Dev(d) => d,
                Staged::Host(_) => panic!("bras non téléversé — bug de phase"),
            }
        }
    }

    fn stage_up_u32(cuda: &Cuda, s: &mut Staged<u32>) -> Result<(), String> {
        if let Staged::Host(v) = s {
            *s = Staged::Dev(cuda.up_u32(v)?);
        }
        Ok(())
    }

    fn stage_up_u16(cuda: &Cuda, s: &mut Staged<u16>) -> Result<(), String> {
        if let Staged::Host(v) = s {
            *s = Staged::Dev(cuda.up_u16(v)?);
        }
        Ok(())
    }

    // ---- un bras sélectionné = une Option construite, et rien d'autre ----
    //
    // Chaque bras porte ses tampons ET sa comptabilité d'octets : un bras
    // écarté n'a ni les uns ni l'autre, donc aucune ligne de table ne peut
    // citer un bras qui n'a pas tourné.

    struct SlotArm {
        words: Staged<u32>,
        bases: Staged<u32>,
        bytes: u64,
    }

    struct PlanesArm {
        pwords: Staged<u32>,
        bytes: u64,
    }

    struct P12Arm {
        words: Staged<u32>,
        exc_idx: Staged<u32>,
        exc_words: Staged<u32>,
        n_exc: usize,
        bytes: u64,
    }

    /// PARTAGÉ par les bras golay70v1 et golay70v2 : zéro octet de format ne
    /// les distingue, donc un seul jeu de tampons sert les deux — le bras
    /// témoin v1 ne coûte aucune VRAM.
    struct G70Arm {
        words: Staged<u32>,
        exc_idx: Staged<u32>,
        exc_words: Staged<u32>,
        n_exc: usize,
        bytes: u64,
    }

    // ---- le bras concurrent (AWQ w4g128) ----
    //
    // Trois tampons et sa propre référence : son contenu n'est celui
    // d'aucun autre bras, donc ni `y_ref` (le contenu LLVQ) ni `y16_ref`
    // (les mêmes poids en binary16) ne le décrivent. Même situation que le
    // bras FP16, même réponse.
    struct AwqArm {
        w: Staged<u32>,
        z: Staged<u32>,
        s: Staged<u16>,
        bytes: u64,
        y_ref: Vec<f64>,
        /// Le dénominateur d'erreur PROPRE au bras : `Σ|w·x|` sur ses poids à
        /// lui. Réutiliser `scale`, calculé sur les poids LLVQ, mesurerait
        /// l'erreur d'AWQ avec la règle d'un autre format.
        scale: Vec<f64>,
    }

    struct Mat {
        name: String,
        d_out: usize,
        d_in: usize,
        nblocks: usize,
        tail_w: usize,
        slot: Option<SlotArm>,
        planes: Option<PlanesArm>,
        p12: Option<P12Arm>,
        g70: Option<G70Arm>,
        awq: Option<AwqArm>,
        // Le témoin FP16 et les entrées partagées, toujours construits :
        // fp16 n'est pas désélectionnable (arms::parse_phases le refuse).
        gscale: cudarc::driver::CudaSlice<f32>,
        rscale: cudarc::driver::CudaSlice<f32>,
        tail: cudarc::driver::CudaSlice<f32>,
        w16: cudarc::driver::CudaSlice<u16>,
        f16_bytes: u64,
        y_ref: Vec<f64>,
        y16_ref: Vec<f64>,
        scale: Vec<f64>,
    }

    /// Les octets qu'un bras lit pour une matrice, dans la comptabilité du
    /// banc — 0 pour un bras non construit, qui n'a de toute façon aucune
    /// ligne de table où l'imprimer. v1 et v2 partagent la ligne golay70 :
    /// mêmes tampons, mêmes octets, par construction.
    fn arm_bytes(m: &Mat, a: usize) -> u64 {
        match a {
            arms::SLOT32 => m.slot.as_ref().map_or(0, |x| x.bytes),
            arms::PLANES14 => m.planes.as_ref().map_or(0, |x| x.bytes),
            arms::PLANES12X => m.p12.as_ref().map_or(0, |x| x.bytes),
            arms::GOLAY70V1 | arms::GOLAY70V2 => m.g70.as_ref().map_or(0, |x| x.bytes),
            arms::FP16 => m.f16_bytes,
            arms::AWQ => m.awq.as_ref().map_or(0, |x| x.bytes),
            _ => unreachable!("bras inconnu"),
        }
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
            // `!(scale > 0.0)` et non `scale <= 0.0` : la forme négée attrape
            // aussi un NaN (poids NaN en amont), que la forme positive
            // laisserait passer dans la division.
            #[allow(clippy::neg_cmp_op_on_partial_ord)]
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
        // La sélection de bras, AVANT toute construction : une valeur
        // invalide doit échouer ici, pas après vingt minutes de transcodage.
        // ⚠️ Elle ne touche PAS l'unité de traduction NVRTC ci-dessous — la
        // sélection change les tampons et le dispatch, jamais le texte
        // compilé ni le rapport de registres (le contraire referait É1).
        let arms_var = std::env::var("LLVQ_BENCH_ARMS").ok();
        let phases = arms::parse_phases(arms_var.as_deref())?;
        let union: ArmSet = {
            let mut u = ArmSet::empty();
            for p in &phases {
                for a in p.iter() {
                    u.insert(a);
                }
            }
            u
        };
        if arms_var.is_some() {
            println!("sélection LLVQ_BENCH_ARMS :");
            for (k, p) in phases.iter().enumerate() {
                println!("  phase {}/{} : {}", k + 1, phases.len(), p.label());
            }
        }
        let g70_needed = union.has(arms::GOLAY70V1) || union.has(arms::GOLAY70V2);

        // Parts concatenated in dependency order: llvq_planes.cuh needs
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
        let (awqcu, awq_overridden) = load_awq_source()?;
        let (gv1cu, golay_v1_overridden) = load_golay_v1_source()?;
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
            // Le concurrent en dernier : ajouter un bras ne doit jamais
            // réordonner les fragments des bras maison.
            awqcu.as_str(),
            // Puis le témoin v1 (2026-08-11), dernier arrivé — même règle.
            gv1cu.as_str(),
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
        if let Some(d) = &awq_overridden {
            println!("  ⚠️ SOURCE AWQ SURCHARGÉE depuis {d}");
        }
        if let Some(d) = &golay_overridden {
            println!("  ⚠️ SOURCES Golay70 SURCHARGÉES depuis {d}");
        }
        if let Some(d) = &golay_v1_overridden {
            println!("  ⚠️ SOURCE TÉMOIN Golay70 v1 SURCHARGÉE depuis {d}");
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
            "tv_golay70_v1",
            "awq_gemv_g128",
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
        // through. Uploaded once, shared by every matrix and by BOTH golay70
        // arms — and not at all when neither is selected. ⚠️ Conditionnels à
        // l'UNION des phases, pas à la phase courante : contrairement aux
        // flux (Staged), ces ~32 Kio résident dès le départ quand un bras
        // golay70 n'entre qu'en phase 2 — divulgué dans le commentaire de
        // `Staged`, avec les trois autres tampons de niveau union.
        // (`golay`/`g70cls`, côté hôte, servent aussi le transcode
        // conditionnel de `build`.)
        let golay = llvq_core::Golay::new();
        let g70cls = golay70_classes(&fd);
        let d_gtab = match g70_needed {
            true => Some(cuda.up_u32(&golay70_gpu_table(&g70cls))?),
            false => None,
        };
        let d_cw = match g70_needed {
            true => Some(cuda.up_u32(&golay70_gpu_codewords(&golay))?),
            false => None,
        };

        let mut rng = SplitMix64::new(0x6_D07);
        let x: Vec<f32> = (0..16384).map(|_| rng.next_gaussian() as f32).collect();
        let d_x = cuda.up_f32(&x)?;
        // L'activation telle que le noyau AWQ la lit : binary16, par float4 de
        // huit. Les bras LLVQ et FP16 lisent la f32 — c'est une différence de
        // format entre concurrents, pas un réglage, et elle se déclare. Elle
        // n'existe que si le bras concurrent tourne.
        let d_xh = match union.has(arms::AWQ) {
            true => {
                let xh: Vec<u16> = x.iter().map(|&v| f16_bits(v)).collect();
                Some(cuda.up_u16(&xh)?)
            }
            false => None,
        };

        // The L = 5 swap of the Planes12x arm needs the exact searcher the
        // artifact was encoded with; built once, shared by every transcode.
        let searcher = Searcher::new();

        let phase0 = phases[0];
        let build = |s: &Src| -> Result<Mat, String> {
            let (d_out, d_in) = (s.d_out, s.d_in);
            let nblocks = d_in / DIM;
            let tail_w = d_in % DIM;
            assert_eq!(d_out % 8, 0, "{}: CUDA lance des blocs entiers", s.name);
            assert!(d_in <= x.len(), "{}: d_in {d_in} dépasse l'activation", s.name);
            // The Slot32 host transcode is NOT conditional: it is the exact
            // content every LLVQ arm is proved against, and the reference
            // loop below decodes it row by row. What the selection spares is
            // its DEVICE buffers, not this.
            let rt = transcode(&fd, &table, &s.indices, &s.gains, Layout::Slot32)
                .map_err(|e| e.to_string())?;
            // The Planes14 stream: a bit-level bijection of the Slot32
            // content, proved block by block in the reference loop below.
            let planes_data = union
                .has(arms::PLANES14)
                .then(|| planes14_from_slot32(&rt, &table));
            // The M2 overlay: 12-byte main stream (L = 5 blocks swapped for
            // their best L ≤ 4 direction) + exact exception records. The
            // swap searches inside, threaded — the expensive build step, and
            // the selection's biggest saving when the arm is out.
            let p12 = match union.has(arms::PLANES12X) {
                true => Some(
                    transcode_planes12x(&fd, &table, &searcher, &s.indices, &s.gains)
                        .map_err(|e| e.to_string())?,
                ),
                false => None,
            };
            // The E2 arm: 9-byte main stream (exception blocks holed to the
            // origin) + exact exception records — pure table lookups plus a
            // Golay rank per block, no search. ONE stream for v1 and v2.
            let g70 = g70_needed
                .then(|| golay70_transcode(&fd, &golay, &g70cls, &table, &s.indices, &s.gains));

            let mut w16 = vec![0u16; d_out * d_in];
            let mut y_ref = vec![0.0f64; d_out];
            let mut y16_ref = vec![0.0f64; d_out];
            let mut scale = vec![0.0f64; d_out];
            // ---- les trois tampons du bras AWQ ----
            //
            // 🚨 Le tampon de poids porte une QUEUE D'ALLOCATION, et l'oublier
            // est une `illegal memory access` après vingt minutes de
            // transcodage. Le chargement `float4` des poids est
            // INCONDITIONNEL dans leur noyau (`awq_gemv.cu`), alors que la
            // garde `inputs_ptr_delta + ic_0 < IC/8` ne protège que l'entrée
            // et l'accumulation. L'indice u32 le plus haut atteint vaut donc
            // `(OC-1)*weight_w + NPG*128 - 1`, au-delà de `OC*weight_w` quand
            // `IC/8` n'est pas multiple de `NPG*128`. Les débordements des
            // lignes intérieures retombent dans la ligne suivante — seule la
            // dernière sort, d'où une queue unique et non un pas élargi.
            let (aww, awz, aws) = awq_strides(d_in);
            // ⚠️ La queue N'EST PAS allouée ici mais au téléversement. Si elle
            // l'était, `chunks_mut(chunk * aww)` pourrait rendre une tranche de
            // plus que `y_ref.chunks_mut(chunk)`, et `zip` s'arrête sur la plus
            // courte — donc des lignes entières seraient silencieusement non
            // quantifiées, et le bras passerait au vert sur un tampon à trous.
            let awq_tail = (awz * 128).saturating_sub(aww);
            let awq_on = union.has(arms::AWQ);
            let mut awq_w = awq_on.then(|| vec![0u32; d_out * aww]);
            let mut awq_z = awq_on.then(|| vec![0u32; d_out * awz]);
            let mut awq_s = awq_on.then(|| vec![0u16; d_out * aws]);
            let mut y_awq_ref = awq_on.then(|| vec![0.0f64; d_out]);
            let mut awq_scale = awq_on.then(|| vec![0.0f64; d_out]);
            // L'activation telle que le noyau AWQ la voit : binary16, lue par
            // `float4` de huit. La référence doit être calculée contre
            // celle-ci, pas contre la f32, sinon on facture au bras un écart
            // qui n'est pas le sien.
            let xh_f64: Option<Vec<f64>> = awq_on
                .then(|| x[..d_in].iter().map(|&v| f16_to_f64(f16_bits(v))).collect());
            let nthreads = std::thread::available_parallelism().map_or(8, |n| n.get());
            let chunk = d_out.div_ceil(nthreads);
            let n_chunks = d_out.div_ceil(chunk);
            // Les tranches par thread des tampons optionnels. ⚠️ Le piège que
            // cette forme évite : `chunks_mut` sur un Vec VIDE rend zéro
            // tranche, et le `zip` du pilote s'arrêterait à la plus courte —
            // toute la boucle de référence sauterait en silence. D'où des
            // `Vec<Option<&mut [T]>>` de longueur n_chunks EXACTEMENT, dans
            // tous les cas.
            fn opt_chunks<T>(
                v: &mut Option<Vec<T>>,
                chunk: usize,
                n: usize,
            ) -> Vec<Option<&mut [T]>> {
                match v.as_mut() {
                    Some(v) => {
                        let out: Vec<Option<&mut [T]>> =
                            v.chunks_mut(chunk).map(Some).collect();
                        assert_eq!(out.len(), n, "tranches optionnelles désalignées");
                        out
                    }
                    None => (0..n).map(|_| None).collect(),
                }
            }
            let awc_v = opt_chunks(&mut awq_w, chunk * aww, n_chunks);
            let azc_v = opt_chunks(&mut awq_z, chunk * awz, n_chunks);
            let asc_v = opt_chunks(&mut awq_s, chunk * aws, n_chunks);
            let yawc_v = opt_chunks(&mut y_awq_ref, chunk, n_chunks);
            let asqc_v = opt_chunks(&mut awq_scale, chunk, n_chunks);
            std::thread::scope(|sc| {
                for (ci, ((((((((w16c, yc), y16c), scc), awc), azc), asc), yawc), asqc)) in w16
                    .chunks_mut(chunk * d_in)
                    .zip(y_ref.chunks_mut(chunk))
                    .zip(y16_ref.chunks_mut(chunk))
                    .zip(scale.chunks_mut(chunk))
                    .zip(awc_v)
                    .zip(azc_v)
                    .zip(asc_v)
                    .zip(yawc_v)
                    .zip(asqc_v)
                    .enumerate()
                {
                    let (rt, table, x, src, planes) = (&rt, &table, &x, &s, &planes_data);
                    let p12 = &p12;
                    let (g70, g70cls, golay) = (&g70, &g70cls, &golay);
                    let xh = &xh_f64;
                    sc.spawn(move || {
                        let (mut awc, mut azc, mut asc, mut yawc, mut asqc) =
                            (awc, azc, asc, yawc, asqc);
                        let mut wrow = vec![0.0f64; d_in];
                        for lr in 0..yc.len() {
                            let row = ci * chunk + lr;
                            wrow.fill(0.0);
                            for p in 0..nblocks {
                                let (pt, gain) = rt.decode_block(table, row * nblocks + p);
                                // The bijection proof: every block of the
                                // Planes14 stream decodes to exactly the
                                // point and gain the Slot32 stream carries.
                                // Each proof runs iff its arm is selected —
                                // it is part of that arm's build, and the
                                // published five/six-arm runs all ran it.
                                if let Some(planes) = planes {
                                    let (ppt, pgain) =
                                        planes14_decode_block(planes, table, row * nblocks + p);
                                    assert_eq!(
                                        (pt, gain),
                                        (ppt, pgain),
                                        "{}: bloc {} — Planes14 n'est pas une bijection de \
                                         Slot32",
                                        src.name,
                                        row * nblocks + p
                                    );
                                }
                                // The overlay proof: main stream + exception
                                // records reconstruct the exact block — the
                                // approximation must be invisible here.
                                if let Some(p12) = p12 {
                                    let (xpt, xgain) =
                                        p12.decode_block(table, row * nblocks + p);
                                    assert_eq!(
                                        (pt, gain),
                                        (xpt, xgain),
                                        "{}: bloc {} — l'overlay Planes12x ne reconstruit pas \
                                         l'exact",
                                        src.name,
                                        row * nblocks + p
                                    );
                                }
                                // The E2 proof: main stream + exception
                                // records reconstruct the exact block, and
                                // the origin-holing is invisible here.
                                if let Some(g70) = g70 {
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
                                }
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
                            // Le bras concurrent, sur les MÊMES poids : w4g128
                            // appliqué à ce que LLVQ a reconstruit. Ce n'est
                            // pas le checkpoint AWQ de Qwen — c'est un w4g128
                            // fidèle en TEMPS (le noyau n'a aucune branche
                            // dépendante des données) et exact en RÉFÉRENCE,
                            // et il ne portera aucune phrase de qualité.
                            if let (Some(awc), Some(azc), Some(asc), Some(yawc), Some(asqc)) = (
                                awc.as_deref_mut(),
                                azc.as_deref_mut(),
                                asc.as_deref_mut(),
                                yawc.as_deref_mut(),
                                asqc.as_deref_mut(),
                            ) {
                                let xh = xh.as_ref().expect("xh_f64 construit avec le bras");
                                yawc[lr] = awq_quant_row(
                                    &wrow,
                                    xh,
                                    d_in,
                                    &mut awc[lr * aww..(lr + 1) * aww],
                                    &mut azc[lr * awz..(lr + 1) * awz],
                                    &mut asc[lr * aws..(lr + 1) * aws],
                                );
                                // Son dénominateur d'erreur, sur ses poids à lui.
                                let mut sa = 0.0f64;
                                for c in 0..d_in {
                                    sa += (wrow[c] * xh[c]).abs();
                                }
                                asqc[lr] = sa;
                            }
                        }
                    });
                }
            });

            // ---- l'étage : Dev pour la phase 1, Host pour les suivantes ----
            let up_or_hold_u32 = |v: Vec<u32>, in_p0: bool| -> Result<Staged<u32>, String> {
                Ok(if in_p0 { Staged::Dev(cuda.up_u32(&v)?) } else { Staged::Host(v) })
            };
            let up_or_hold_u16 = |v: Vec<u16>, in_p0: bool| -> Result<Staged<u16>, String> {
                Ok(if in_p0 { Staged::Dev(cuda.up_u16(&v)?) } else { Staged::Host(v) })
            };

            let slot = match union.has(arms::SLOT32) {
                true => {
                    let mut bytes = rt.data.clone();
                    bytes.extend_from_slice(&[0u8; 20]); // the five-word read of the last block
                    while !bytes.len().is_multiple_of(4) {
                        bytes.push(0);
                    }
                    let words: Vec<u32> = bytes
                        .chunks_exact(4)
                        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect();
                    let in0 = phase0.has(arms::SLOT32);
                    Some(SlotArm {
                        words: up_or_hold_u32(words, in0)?,
                        bases: up_or_hold_u32(rt.bases.clone(), in0)?,
                        bytes: rt.data.len() as u64
                            + rt.bases.len() as u64 * 4
                            + (d_out * tail_w) as u64 * 4
                            + d_out as u64 * 4,
                    })
                }
                false => None,
            };

            let planes = match planes_data {
                Some(pdata) => {
                    let bytes_acc = pdata.len() as u64
                        + (d_out * tail_w) as u64 * 4
                        + d_out as u64 * 4;
                    let mut pbytes = pdata;
                    // The last block's four-word window reaches at most 2
                    // bytes past the stream; pad 4 and align, mirroring the
                    // slot padding.
                    pbytes.extend_from_slice(&[0u8; 4]);
                    while !pbytes.len().is_multiple_of(4) {
                        pbytes.push(0);
                    }
                    let pwords: Vec<u32> = pbytes
                        .chunks_exact(4)
                        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect();
                    Some(PlanesArm {
                        pwords: up_or_hold_u32(pwords, phase0.has(arms::PLANES14))?,
                        bytes: bytes_acc,
                    })
                }
                None => None,
            };

            let pad12 = |mut b: Vec<u8>| -> Vec<u32> {
                b.extend_from_slice(&[0u8; 4]);
                while !b.len().is_multiple_of(4) {
                    b.push(0);
                }
                b.chunks_exact(4)
                    .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect()
            };

            // The M2 arm's three device arrays and its byte accounting:
            // main stream + exception indices + exception records + the
            // same tail/rscale terms as the other LLVQ arms. Upload paddings
            // are NOT billed: the slot arm ignores its 20-byte pad and the
            // planes arm its 4-byte pad, so billing ours here would mix two
            // byte accountings — the exact mistake the K-1 lot eliminated.
            let p12 = match p12 {
                Some(p12) => {
                    let n_exc = p12.exc_idx.len();
                    let bytes_acc = p12.data.len() as u64
                        + n_exc as u64 * 4
                        + p12.exc_data.len() as u64
                        + (d_out * tail_w) as u64 * 4
                        + d_out as u64 * 4;
                    let p12words: Vec<u32> = pad12(p12.data);
                    let excwords: Vec<u32> = pad12(p12.exc_data);
                    // cudarc refuses a zero-length upload; a matrix without
                    // any L = 5 block gets a one-word dummy the kernel never
                    // reads (n_exc == 0 spawns no correction CTA).
                    let exc_idx_up: Vec<u32> = if p12.exc_idx.is_empty() {
                        vec![0]
                    } else {
                        p12.exc_idx
                    };
                    let in0 = phase0.has(arms::PLANES12X);
                    Some(P12Arm {
                        words: up_or_hold_u32(p12words, in0)?,
                        exc_idx: up_or_hold_u32(exc_idx_up, in0)?,
                        exc_words: up_or_hold_u32(excwords, in0)?,
                        n_exc,
                        bytes: bytes_acc,
                    })
                }
                None => None,
            };

            // The E2 arm's three device arrays and its byte accounting:
            // 72 bits per block + 144 per exception + the same tail/rscale
            // terms as every LLVQ arm; upload paddings NOT billed — the
            // unified rule. One set of buffers for BOTH golay70 arms.
            let g70 = match g70 {
                Some(g70) => {
                    let n_gexc = g70.exc_idx.len();
                    let bytes_acc = g70.data.len() as u64
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
                    let in0 =
                        phase0.has(arms::GOLAY70V1) || phase0.has(arms::GOLAY70V2);
                    Some(G70Arm {
                        words: up_or_hold_u32(gwords, in0)?,
                        exc_idx: up_or_hold_u32(gexc_idx_up, in0)?,
                        exc_words: up_or_hold_u32(gexcwords, in0)?,
                        n_exc: n_gexc,
                        bytes: bytes_acc,
                    })
                }
                None => None,
            };

            let awq = match (awq_w, awq_z, awq_s, y_awq_ref, awq_scale) {
                (Some(awq_w), Some(awq_z), Some(awq_s), Some(y_awq), Some(a_scale)) => {
                    let in0 = phase0.has(arms::AWQ);
                    Some(AwqArm {
                        w: {
                            // La queue d'allocation, ajoutée ICI et nulle part
                            // ailleurs : le noyau lit jusqu'à `NPG*128` mots
                            // au-delà du début de la dernière ligne, et ces
                            // mots doivent exister. Ils ne sont PAS facturés —
                            // c'est une marge d'adressage, pas un octet que
                            // l'algorithme transporte, et la facturer
                            // gonflerait le bras concurrent d'un poids qu'il
                            // ne porte pas.
                            let mut wup = awq_w;
                            wup.resize(d_out * aww + awq_tail, 0);
                            up_or_hold_u32(wup, in0)?
                        },
                        z: up_or_hold_u32(awq_z, in0)?,
                        s: up_or_hold_u16(awq_s, in0)?,
                        bytes: awq_bytes(d_out, d_in),
                        y_ref: y_awq,
                        scale: a_scale,
                    })
                }
                _ => None,
            };

            Ok(Mat {
                name: s.name.clone(),
                d_out,
                d_in,
                nblocks,
                tail_w,
                slot,
                planes,
                p12,
                g70,
                awq,
                gscale: cuda.up_f32(&s.centroids)?,
                rscale: cuda.up_f32(&s.rscale)?,
                tail: cuda.up_f32(if s.tail.is_empty() { &[0.0f32] } else { &s.tail })?,
                w16: cuda.up_u16(&w16)?,
                f16_bytes: (d_out * d_in * 2) as u64,
                y_ref,
                y16_ref,
                scale,
            })
        };

        println!(
            "\nConstruction — transcodage Slot32 (référence, toujours){}{}{}{}, preuves \
             de reconstruction exacte des bras construits…",
            if union.has(arms::PLANES14) { ", Planes14" } else { "" },
            if union.has(arms::PLANES12X) {
                ", Planes12x (swap L = 5 → L ≤ 4 inclus)"
            } else {
                ""
            },
            if g70_needed { ", Golay70 (trous origine + exceptions E2)" } else { "" },
            if union.has(arms::AWQ) { ", AWQ w4g128" } else { "" },
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
        // Le bras AWQ écrit une sortie binary16 — c'est ce que fait leur
        // noyau. Un tampon séparé plutôt qu'une réinterprétation de `d_y` :
        // deux types, deux tampons, et rien qui puisse se recouvrir — et
        // pas de tampon du tout quand le bras n'est pas sélectionné.
        let mut d_yh = match union.has(arms::AWQ) {
            true => Some(cuda.zeros_u16(max_dout)?),
            false => None,
        };
        // Imprimé parce qu'il ne l'était pas : c'est ce silence qui a laissé
        // `d_y` doubler sans que personne ne puisse le lire dans un log.
        println!("  d_y : {max_dout} f32 — la table, et rien d'autre (la section A4 a le sien)");
        println!(
            "  {} matrices, {:.2} Md de poids, en {:.0} s — preuves bloc par bloc des \
             bras construits",
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
        let f_golay70_v1 = cuda.func("tv_golay70_v1")?;
        let f_f16 = cuda.func("tv_f16")?;
        let f_awq = cuda.func("awq_gemv_g128")?;
        let shared = (TILE_BLOCKS * DIM * 4) as u32;

        // Chaque fermeture n'est appelée que si son bras est dans la phase
        // courante ; le `expect` nominatif transforme toute erreur de
        // séquencement en panique lisible plutôt qu'en mesure d'un bras vide.
        let run_slot = |m: &Mat, y: &mut cudarc::driver::CudaSlice<f32>| -> Result<(), String> {
            let a = m.slot.as_ref().expect("bras slot32 non construit");
            cuda.launch_slot(
                &f_slot, a.words.dev(), a.bases.dev(), &d_tab, &m.gscale, &m.rscale,
                &m.tail, &d_x, y, m.nblocks as u32, m.tail_w as u32, m.d_out as u32,
                THREADS, shared,
            )
        };
        let run_planes = |m: &Mat, y: &mut cudarc::driver::CudaSlice<f32>| -> Result<(), String> {
            let a = m.planes.as_ref().expect("bras planes14 non construit");
            launch_planes(
                &cuda, &f_planes, a.pwords.dev(), &d_tab, &m.gscale, &m.rscale, &m.tail,
                &d_x, y, m.nblocks as u32, m.tail_w as u32, m.d_out as u32, THREADS, shared,
            )
        };
        let run_planes12x =
            |m: &Mat, y: &mut cudarc::driver::CudaSlice<f32>| -> Result<(), String> {
                let a = m.p12.as_ref().expect("bras planes12x non construit");
                launch_planes12x(
                    &cuda, &f_planes12x, a.words.dev(), a.exc_idx.dev(), a.exc_words.dev(),
                    &d_tab, &m.gscale, &m.rscale, &m.tail, &d_x, y, m.nblocks as u32,
                    m.tail_w as u32, a.n_exc as u32, m.d_out as u32, THREADS, shared,
                )
            };
        // v1 et v2 : mêmes tampons, même grille, même protocole memset —
        // seul le NOYAU change. C'est toute la comparaison.
        let run_g70 = |m: &Mat,
                       f: &cudarc::driver::CudaFunction,
                       y: &mut cudarc::driver::CudaSlice<f32>|
         -> Result<(), String> {
            let a = m.g70.as_ref().expect("bras golay70 non construit");
            launch_golay70(
                &cuda, f, a.words.dev(), a.exc_idx.dev(), a.exc_words.dev(),
                d_cw.as_ref().expect("tables golay70 non téléversées"),
                d_gtab.as_ref().expect("tables golay70 non téléversées"),
                &d_tab, &m.gscale, &m.rscale, &m.tail, &d_x, y,
                m.nblocks as u32, m.tail_w as u32, a.n_exc as u32, m.d_out as u32,
                THREADS, shared,
            )
        };
        let run_awq =
            |m: &Mat, y: &mut cudarc::driver::CudaSlice<u16>| -> Result<(), String> {
                let a = m.awq.as_ref().expect("bras awq non construit");
                launch_awq(
                    &cuda, &f_awq,
                    d_xh.as_ref().expect("activation binary16 non téléversée"),
                    a.w.dev(), a.z.dev(), a.s.dev(), y,
                    m.d_in as u32, m.d_out as u32,
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

        // ---- vérification f64, PAR BRAS — au plus tard avant la première
        // phase qui chronomètre le bras (la garde du §7 du
        // pré-enregistrement : jamais un chronométrage avant sa preuve).
        //
        // Les commentaires de fond des six vérifications d'origine tiennent
        // inchangés : même référence exacte pour tous les bras LLVQ (la
        // preuve de l'overlay Planes12x et de la correction E2 est justement
        // qu'ils l'atteignent), y16_ref pour le témoin FP16, et pour AWQ SA
        // référence, SON dénominateur et SA tolérance — plus lâche d'un
        // facteur ~100 parce que sa sortie est arrondie en binary16 par le
        // noyau (`f2h`), pas parce qu'on lui pardonne quoi que ce soit : sa
        // référence `scale·(q − zero)` reste exacte au bit près.
        let verify_arm = |a: usize,
                          mats: &[Mat],
                          d_y: &mut cudarc::driver::CudaSlice<f32>,
                          d_yh: &mut Option<cudarc::driver::CudaSlice<u16>>|
         -> Result<f64, String> {
            let mut worst = 0.0f64;
            for m in mats {
                let e = match a {
                    arms::AWQ => {
                        let yh = d_yh.as_mut().expect("d_yh du bras awq");
                        run_awq(m, yh)?;
                        cuda.sync()?;
                        let goth = cuda.down_u16(yh)?;
                        let gotf: Vec<f32> =
                            goth[..m.d_out].iter().map(|&h| f16_to_f64(h) as f32).collect();
                        let aw = m.awq.as_ref().expect("bras awq non construit");
                        let e = worst_error(&gotf, &aw.y_ref, &aw.scale);
                        assert!(e < AWQ_TOL, "{} / AWQ : {e:.2e}·Σ|w·x|", m.name);
                        e
                    }
                    _ => {
                        match a {
                            arms::SLOT32 => run_slot(m, d_y)?,
                            arms::PLANES14 => run_planes(m, d_y)?,
                            arms::PLANES12X => run_planes12x(m, d_y)?,
                            arms::GOLAY70V1 => run_g70(m, &f_golay70_v1, d_y)?,
                            arms::GOLAY70V2 => run_g70(m, &f_golay70, d_y)?,
                            arms::FP16 => run_f16(m, d_y)?,
                            _ => unreachable!("bras inconnu"),
                        }
                        cuda.sync()?;
                        let got = cuda.down_f32(d_y)?;
                        let want = if a == arms::FP16 { &m.y16_ref } else { &m.y_ref };
                        let e = worst_error(&got[..m.d_out], want, &m.scale);
                        assert!(
                            e < TOL,
                            "{} / {} : {e:.2e}·Σ|w·x|",
                            m.name,
                            arms::ARM_NAMES[a]
                        );
                        e
                    }
                };
                worst = worst.max(e);
            }
            Ok(worst)
        };

        // La vague de téléversement d'une phase : les tampons des bras qui
        // entrent montent sur la carte, ceux des bras déjà montés ne bougent
        // pas (stage_up_* est idempotent — v1 et v2 partagent le groupe
        // golay70, le second des deux ne re-téléverse rien).
        let upload_added = |mats: &mut [Mat], added: ArmSet| -> Result<(), String> {
            for m in mats.iter_mut() {
                if added.has(arms::SLOT32) {
                    let a = m.slot.as_mut().expect("bras slot32 non construit");
                    stage_up_u32(&cuda, &mut a.words)?;
                    stage_up_u32(&cuda, &mut a.bases)?;
                }
                if added.has(arms::PLANES14) {
                    let a = m.planes.as_mut().expect("bras planes14 non construit");
                    stage_up_u32(&cuda, &mut a.pwords)?;
                }
                if added.has(arms::PLANES12X) {
                    let a = m.p12.as_mut().expect("bras planes12x non construit");
                    stage_up_u32(&cuda, &mut a.words)?;
                    stage_up_u32(&cuda, &mut a.exc_idx)?;
                    stage_up_u32(&cuda, &mut a.exc_words)?;
                }
                if added.has(arms::GOLAY70V1) || added.has(arms::GOLAY70V2) {
                    let a = m.g70.as_mut().expect("bras golay70 non construit");
                    stage_up_u32(&cuda, &mut a.words)?;
                    stage_up_u32(&cuda, &mut a.exc_idx)?;
                    stage_up_u32(&cuda, &mut a.exc_words)?;
                }
                if added.has(arms::AWQ) {
                    let a = m.awq.as_mut().expect("bras awq non construit");
                    stage_up_u32(&cuda, &mut a.w)?;
                    stage_up_u32(&cuda, &mut a.z)?;
                    stage_up_u16(&cuda, &mut a.s)?;
                }
            }
            Ok(())
        };

        let rows: usize = mats.iter().map(|m| m.d_out).sum();
        let total_blocks: u64 = mats.iter().map(|m| (m.d_out * m.nblocks) as u64).sum();
        if union.has(arms::PLANES12X) {
            let total_exc: u64 = mats
                .iter()
                .map(|m| m.p12.as_ref().expect("bras planes12x non construit").n_exc as u64)
                .sum();
            println!(
                "  exceptions L = 5 : {total_exc} sur {total_blocks} blocs \
                 ({:.4} %), corrigées dans le même lancement",
                total_exc as f64 * 100.0 / total_blocks as f64
            );
        }
        if g70_needed {
            let total_gexc: u64 = mats
                .iter()
                .map(|m| m.g70.as_ref().expect("bras golay70 non construit").n_exc as u64)
                .sum();
            println!(
                "  exceptions E2 (pair violant ou L = 5) : {total_gexc} sur {total_blocks} \
                 blocs ({:.4} %), corrigées dans le même lancement",
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
        // L'ordre d'AFFICHAGE — cosmétique, distinct de l'ordre de dispatch :
        // le témoin d'abord, v2 sous v1, le concurrent en dernier — la forme
        // des tables publiées.
        const DISPLAY: [usize; arms::N_ARMS] = [
            arms::FP16,
            arms::SLOT32,
            arms::PLANES14,
            arms::PLANES12X,
            arms::GOLAY70V1,
            arms::GOLAY70V2,
            arms::AWQ,
        ];
        const ROW_NAMES: [&str; arms::N_ARMS] = [
            "LLVQ Slot32",
            "LLVQ Planes14",
            "LLVQ Planes12x",
            "LLVQ Golay70 v1",
            "FP16 (128 bits)",
            "AWQ w4g128",
            "LLVQ Golay70 v2",
        ];
        let n_phases = phases.len();
        let report = |mats: &[Mat],
                      pi: usize,
                      phase: ArmSet,
                      times: &[Vec<f64>; arms::N_ARMS]| {
            let bytes_of = |a: usize| -> u64 { mats.iter().map(|m| arm_bytes(m, a)).sum() };
            let t_f16 = &times[arms::FP16];
            let phase_tag = if n_phases > 1 {
                format!("  [phase {}/{} : {}]", pi + 1, n_phases, phase.label())
            } else {
                String::new()
            };
            println!(
                "\nUN TOKEN — {} matrices, un stream, {} bras entrelacés{}",
                mats.len(),
                phase.len(),
                phase_tag
            );
            println!(
                "  {ROUNDS} rounds, {WARMUP} jetés ; les rapports sont formés ROUND PAR ROUND"
            );
            println!("  {}", "-".repeat(80));
            println!(
                "  {:<22}{:>9}{:>9}{:>9}{:>9}{:>9}{:>9}",
                "format", "min ms", "méd ms", "max ms", "Go lus", "b/poids", "Go/s"
            );
            for a in DISPLAY {
                if !phase.has(a) {
                    continue;
                }
                let (lo, md, hi) = spread(times[a].clone());
                let b = bytes_of(a);
                println!(
                    "  {:<22}{:>9.3}{:>9.3}{:>9.3}{:>9.2}{:>9.3}{:>9.0}",
                    ROW_NAMES[a],
                    lo * 1e3,
                    md * 1e3,
                    hi * 1e3,
                    b as f64 / 1e9,
                    b as f64 * 8.0 / n_weights as f64,
                    b as f64 / lo / 1e9
                );
            }
            println!("  {}", "-".repeat(80));
            for a in DISPLAY {
                if a == arms::FP16 || !phase.has(a) {
                    continue;
                }
                let (lo, md, hi) = spread(per_round(t_f16, &times[a]));
                if a == arms::AWQ {
                    println!(
                        "  {:<16} vs FP16     : {md:.2}× [{lo:.2}–{hi:.2}]  \
                         (CONCURRENT — lit {:.3} b/poids ;\n  la grandeur comparable est \
                         les Go/s, pas ce rapport)",
                        ROW_NAMES[a],
                        bytes_of(a) as f64 * 8.0 / n_weights as f64
                    );
                } else {
                    println!(
                        "  {:<16} vs FP16     : {md:.2}× [{lo:.2}–{hi:.2}]",
                        ROW_NAMES[a]
                    );
                }
            }
            // Les rapports entre bras : `num` contre `den`, formés round par
            // round — >1 = `num` plus rapide. N'existent que si les DEUX
            // bras ont tourné dans cette phase.
            let pair = |num: usize, den: usize, note: &str| {
                if phase.has(num) && phase.has(den) {
                    let (lo, md, hi) = spread(per_round(&times[den], &times[num]));
                    println!(
                        "  {:<10} vs {:<10}: {md:.2}× [{lo:.2}–{hi:.2}]  {note}",
                        arms::ARM_NAMES[num],
                        arms::ARM_NAMES[den]
                    );
                }
            };
            pair(
                arms::PLANES14,
                arms::SLOT32,
                "(>1 = Planes14 plus rapide, même contenu décodé)",
            );
            pair(
                arms::PLANES12X,
                arms::PLANES14,
                "(>1 = Planes12x plus rapide ; memset + correction inclus, même y exact)",
            );
            pair(
                arms::GOLAY70V1,
                arms::PLANES12X,
                "(>1 = Golay70 v1 plus rapide ; memset + correction inclus, même y exact)",
            );
            pair(
                arms::GOLAY70V2,
                arms::GOLAY70V1,
                "(>1 = v2 plus rapide ; MÊMES tampons, même y exact — le rapport de la \
                 campagne v2)",
            );
            println!("\n  source : {source}");
            let light = DISPLAY
                .into_iter()
                .filter(|&a| {
                    a != arms::FP16 && a != arms::AWQ && phase.has(a)
                })
                .min_by_key(|&a| bytes_of(a));
            if let Some(la) = light {
                let light = bytes_of(la);
                println!(
                    "  {:.0} Mo distincts par passe sur le bras le plus léger ({}), soit \
                     {:.1}× la L2 lue.\n  Sous 1× on mesurerait le cache et pas la DRAM — le piège \
                     qui a rendu\n  optimiste toute mesure LLVQ antérieure au 2026-07-31.",
                    light as f64 / 1e6,
                    ROW_NAMES[la],
                    light as f64 / dev.l2_bytes as f64
                );
            }
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
            let fb = bytes_of(arms::FP16) as f64;
            let (f_lo, _, _) = spread(t_f16.clone());
            let head_s = head_bytes / (fb / f_lo);
            let with_head: Vec<String> = DISPLAY
                .into_iter()
                .filter(|&a| a != arms::FP16 && a != arms::AWQ && phase.has(a))
                .map(|a| {
                    let (lo, _, _) = spread(times[a].clone());
                    format!("{} {:.2}×", ROW_NAMES[a], (f_lo + head_s) / (lo + head_s))
                })
                .collect();
            if !with_head.is_empty() {
                println!(
                    "\n  Avec le lm_head f16 non quantifié ({:.0} M poids, {:.2} ms au débit FP16\n  \
                     mesuré, ajouté aux bras LLVQ) : {}.\n  Normes, activations, attention et \
                     rotation ne sont mesurées ni ici ni là.",
                    389_070_848f64 / 1e6,
                    head_s * 1e3,
                    with_head.join(", ")
                );
            }
            println!(
                "\n  ⚠️ à ne JAMAIS comparer au chiffre Metal ligne à ligne, ni soustraire d'un\n  \
                 run de bin/matvec : autres rounds, autre unité de traduction NVRTC. Chaque\n  \
                 bras contre sa référence f64 ; les rapports se forment round par round,\n  \
                 dans un même processus."
            );
        };

        // ---- les phases : téléversement, vérification, rounds, rapport ----
        let mut verified = [false; arms::N_ARMS];
        let mut phase_times: Vec<[Vec<f64>; arms::N_ARMS]> = Vec::new();
        for pi in 0..n_phases {
            let phase = phases[pi];
            if pi > 0 {
                let added = phase.minus(phases[pi - 1]);
                if !added.is_empty() {
                    println!(
                        "\n— phase {}/{n_phases} : téléversement des bras ajoutés ({}) —",
                        pi + 1,
                        added.label()
                    );
                    upload_added(&mut mats, added)?;
                }
            }
            let mut worsts: Vec<String> = Vec::new();
            for a in phase.iter() {
                if !verified[a] {
                    if worsts.is_empty() {
                        println!("\nVérification de chaque ligne contre la référence f64…");
                    }
                    let w = verify_arm(a, &mats, &mut d_y, &mut d_yh)?;
                    worsts.push(format!("{} {w:.1e}", arms::ARM_NAMES[a]));
                    verified[a] = true;
                }
            }
            if !worsts.is_empty() {
                println!(
                    "  {rows} lignes, seuil {TOL:.0e} (AWQ : {AWQ_TOL:.0e}, sortie \
                     binary16) — pires erreurs {} ·Σ|w·x|",
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
                                run_awq(m, d_yh.as_mut().expect("d_yh du bras awq"))?
                            }
                            arms::GOLAY70V2 => run_g70(m, &f_golay70, &mut d_y)?,
                            _ => unreachable!("bras inconnu"),
                        }
                    }
                    cuda.sync()?;
                    let s = t.elapsed().as_secs_f64();
                    if rep >= WARMUP {
                        t_arm.push(s);
                    }
                }
            }
            report(&mats, pi, phase, &times);
            phase_times.push(times);
        }

        // ---- Δ_contrôle : la dérive des bras communs entre phases ----
        //
        // Le chiffre que la règle §4 du pré-enregistrement du 2026-08-10
        // demande au job de rapporter, imprimé par le banc lui-même pour
        // qu'aucun rapport ne se fasse entre deux processus : R =
        // max(Δ_contrôle, demi-étendue intra-run du bras le plus dispersé) ;
        // |Δ| > 2R sépare, |Δ| < R est indiscernable à cette résolution,
        // entre les deux : non résolu, publié comme tel.
        if n_phases > 1 {
            println!("\nΔ_contrôle — dérive des bras communs entre phases consécutives");
            let med = |v: &[f64]| -> f64 {
                let mut s = v.to_vec();
                s.sort_by(f64::total_cmp);
                s[s.len() / 2]
            };
            for k in 1..n_phases {
                let (prev, cur) = (&phase_times[k - 1], &phase_times[k]);
                let common = phases[k - 1]; // ⊆ phases[k], garanti au parsing
                let mut delta_ctrl = 0.0f64;
                for a in common.iter() {
                    let (m0, m1) = (med(&prev[a]), med(&cur[a]));
                    let dms = (m1 - m0) / m0;
                    if a == arms::FP16 {
                        println!(
                            "  phases {k}→{} {:<10} méd {:>8.3} → {:>8.3} ms ({:+.2} %)",
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
                            "  phases {k}→{} {:<10} méd {:>8.3} → {:>8.3} ms ({:+.2} %), \
                             vs FP16 {:.3} → {:.3} ({:+.2} %)",
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
                    "  Δ_contrôle (max |Δ| des rapports vs FP16) = {:.2} % ; demi-étendue \
                     max intra-run = {:.2} % ({half_name}) ; R = {:.2} %",
                    delta_ctrl * 100.0,
                    half * 100.0,
                    r * 100.0
                );
            }
        }

        // ---- A4 : la fusion q+k+v et gate+up, justesse d'abord, coût ensuite ----
        //
        // Construite ICI, après la table des cinq bras : ni ses ~2,9 Go ni son
        // `d_out` doublé ne doivent exister pendant que la table se mesure
        // (voir le long commentaire au-dessus de `max_dout`).
        let mut fused: Vec<FusedMat> = Vec::new();
        // La section exige les DEUX bras qu'elle fusionne : sur une
        // sélection sans slot32 ou sans planes14, elle n'a ni flux à
        // concaténer ni bras séparé à qui comparer — elle saute, et le dit.
        let a4_on = union.has(arms::SLOT32) && union.has(arms::PLANES14);
        if !srcs.is_empty() && !a4_on {
            println!(
                "\n  section A4 (fusion) SAUTÉE : elle exige slot32 ET planes14 dans la \
                 sélection LLVQ_BENCH_ARMS"
            );
        }
        if !srcs.is_empty() && a4_on {
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
            let sb_sep = unfused_bytes(|m| arm_bytes(m, arms::SLOT32));
            let sb_fus = fused_bytes(|f| f.slot_bytes, |m| arm_bytes(m, arms::SLOT32));
            let pb_sep = unfused_bytes(|m| arm_bytes(m, arms::PLANES14));
            let pb_fus = fused_bytes(|f| f.planes_bytes, |m| arm_bytes(m, arms::PLANES14));

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
