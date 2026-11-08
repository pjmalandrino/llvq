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
//! ## `LLVQ_BENCH_ARMS` — the arm selector (lot B, deviation É1)
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
    llvq_cuda::check_removed_bench(|v| std::env::var(v).ok())?;
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
    use llvq_search::tetra::Tetra;
    use llvq_search::Searcher;
    use std::time::Instant;

    include!("../planes14_host.rs");
    include!("../golay70_host.rs");
    include!("../seg_host.rs");
    include!("../e1v_host.rs");

    const THREADS: u32 = 256;
    const GSCALE: [f32; 2] = [0.625, 1.375];
    const TABLE_ENTRIES: usize = 512;
    const REC_WORDS: usize = 6;
    const TOL: f64 = 1e-5;
    /// The competitor arm's threshold, looser by a factor of ~100, for a
    /// reason of format and not of indulgence: `awq_gemv_g128` writes its
    /// output in **binary16** (`f2h`, last line of `awq_gemv.cu`) where the
    /// five in-house arms return f32. A binary16 output carries ~2⁻¹¹ of
    /// relative error by construction, so demanding 1e-5 here would fail a
    /// perfectly correct kernel. Its REFERENCE stays exact: `scale·(q − zero)`
    /// is what the kernel reconstructs bit for bit.
    const AWQ_TOL: f64 = 1e-3;
    const ROUNDS: usize = 7;
    const WARMUP: usize = 2;
    // Arm order inside a round: `arms::ARM_NAMES`, the registration order —
    // Slot32, Planes14, Planes12x, Golay70 v1, FP16, AWQ, Golay70 v2.
    //
    // ⚠️ The order is the REGISTRATION order, and an added arm ALWAYS goes
    // LAST. Inserting an arm in the middle would change the dispatch order of
    // the existing arms inside every round, hence the object the published
    // table measures. AWQ (2026-08-10) then Golay70 v2 (2026-08-11) both
    // follow this rule. An `LLVQ_BENCH_ARMS` selection never changes this
    // order: it skips arms, it does not move them.

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
    /// The competitor arm. It depends on `h2f`/`f2h`, hence on `matvec.cu`,
    /// so it is concatenated AFTER it, and last, so that adding a competitor
    /// never reorders the fragments of the in-house arms.
    const AWQ_CU_EMBED: &str = include_str!("../../kernels/awq_gemv.cu");
    /// The witness arm: the FROZEN copy of the published Golay70 decoder
    /// (v1), symbols renamed, buffers shared with v2. Concatenated after AWQ,
    /// since the arm added last reorders no existing fragment.
    const GOLAY_V1_CU_EMBED: &str = include_str!("../../kernels/golay70_v1.cu");

    fn load_planes_sources() -> Result<(String, String, Option<String>), String> {
        match std::env::var("LLVQ_KERNEL_DIR") {
            Err(_) => Ok((PLANES_CUH_EMBED.to_string(), PLANES_CU_EMBED.to_string(), None)),
            Ok(dir) => {
                let rd = |n: &str| {
                    std::fs::read_to_string(std::path::Path::new(&dir).join(n))
                        .map_err(|e| format!("LLVQ_KERNEL_DIR={dir}: {n}: {e}"))
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
                        .map_err(|e| format!("LLVQ_KERNEL_DIR={dir}: {n}: {e}"))
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
                        .map_err(|e| format!("LLVQ_KERNEL_DIR={dir}: {n}: {e}"))
                };
                let cuh = rd("llvq_golay.cuh")?;
                let cu = rd("golay70.cu")?;
                Ok((cuh, cu, Some(dir)))
            }
        }
    }

    /// Same contract for the one segmented source — the A4 arm's kernel.
    /// Same contract as the others: without the variable, the embedded text;
    /// with it, the file from the directory, announced loudly. It is also the
    /// path by which a kernel we cannot vendor, QTIP being GPL-3, would enter
    /// without being committed.
    fn load_awq_source() -> Result<(String, Option<String>), String> {
        match std::env::var("LLVQ_KERNEL_DIR") {
            Err(_) => Ok((AWQ_CU_EMBED.to_string(), None)),
            Ok(dir) => {
                let cu =
                    std::fs::read_to_string(std::path::Path::new(&dir).join("awq_gemv.cu"))
                        .map_err(|e| format!("LLVQ_KERNEL_DIR={dir}: awq_gemv.cu: {e}"))?;
                Ok((cu, Some(dir)))
            }
        }
    }

    /// Our QTIP shims — committed, so always available, and version-locked to
    /// the binary that launches them.
    const QTIP_GLUE_EMBED: &str = include_str!("../../kernels/qtip_glue.cu");
    /// The typedefs the fetched kernel needs and NVRTC does not carry. Ours,
    /// and concatenated **before** the fetched half.
    const QTIP_PRELUDE_EMBED: &str = include_str!("../../kernels/qtip_prelude.cuh");

    /// The QTIP device half, and the one loader here that reads a **different**
    /// variable from every other.
    ///
    /// 🚨 `LLVQ_QTIP_DIR`, not `LLVQ_KERNEL_DIR`, and the distinction is not
    /// cosmetic. `LLVQ_KERNEL_DIR` means *override every kernel source from
    /// this directory*: every other loader in this file and in
    /// `load_sources_many` reads ITS OWN files from it and **fails hard** when
    /// one is missing. Pointing it at the QTIP fetch output would therefore
    /// break the whole bench on `matvec.cu: No such file or directory`. QTIP is
    /// not an override of anything — it is an ADDITION — so it gets a variable
    /// of its own, and the two compose instead of colliding.
    ///
    /// Only the device half is read from disk: the kernel is GPL v3 and this
    /// workspace is MIT OR Apache-2.0, so it is fetched at job time by
    /// `ops/fetch-qtip.sh` and never committed. The shims above are ours.
    /// `Ok(None)` means "no QTIP here", the normal state on any machine that
    /// has not run the script; the arm is then simply not dispatchable, which
    /// `arms::HAS_KERNEL` already says. What must never happen is a *silent*
    /// fallback to some other text, which is why there is nothing to fall
    /// back to.
    fn load_qtip_sources() -> Result<Option<(String, String)>, String> {
        let Ok(dir) = std::env::var("LLVQ_QTIP_DIR") else {
            return Ok(None);
        };
        let cuh = std::fs::read_to_string(std::path::Path::new(&dir).join("qtip_device.cuh"))
            .map_err(|e| {
                format!(
                    "LLVQ_QTIP_DIR={dir}: qtip_device.cuh: {e}, run ops/fetch-qtip.sh \
                     and point the variable at its output"
                )
            })?;
        // Prelude first — it declares the types the fetched half names.
        Ok(Some((format!("{QTIP_PRELUDE_EMBED}\n{cuh}"), QTIP_GLUE_EMBED.to_string())))
    }

    fn load_planes_seg_source() -> Result<(String, Option<String>), String> {
        match std::env::var("LLVQ_KERNEL_DIR") {
            Err(_) => Ok((PLANES_SEG_CU_EMBED.to_string(), None)),
            Ok(dir) => {
                let cu = std::fs::read_to_string(
                    std::path::Path::new(&dir).join("planes_seg.cu"),
                )
                .map_err(|e| format!("LLVQ_KERNEL_DIR={dir}: planes_seg.cu: {e}"))?;
                Ok((cu, Some(dir)))
            }
        }
    }

    /// Same contract for the frozen v1 copy. Overriding the WITNESS is
    /// allowed, but announced as loudly as the rest: a witness silently
    /// replaced attests to nothing.
    fn load_golay_v1_source() -> Result<(String, Option<String>), String> {
        match std::env::var("LLVQ_KERNEL_DIR") {
            Err(_) => Ok((GOLAY_V1_CU_EMBED.to_string(), None)),
            Ok(dir) => {
                let cu = std::fs::read_to_string(
                    std::path::Path::new(&dir).join("golay70_v1.cu"),
                )
                .map_err(|e| format!("LLVQ_KERNEL_DIR={dir}: golay70_v1.cu: {e}"))?;
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

    /// A buffer whose arm may enter only at a later phase: the bytes are
    /// prepared once at build time (the host work, references and transcodes,
    /// is not redone between phases), but they reach the card only at the
    /// start of the first phase that times the arm. During a phase the
    /// resident STREAMS (the GB) are therefore exactly those of that phase's
    /// arms, the residency invariant the §4 control of the preregistration
    /// must reproduce.
    ///
    /// ⚠️ Exact scope, so the claim does not overreach: four CONSTANT buffers
    /// follow the UNION of the phases and not the current phase.
    /// `d_gtab`/`d_cw` (~32 KiB, as soon as a golay70 arm is selected
    /// anywhere), `d_xh` (32 KiB) and `d_yh` (~19 KiB, as soon as awq is).
    /// They do not go through this stage: ~83 KiB in all against ~15 GB of
    /// stream, below any resolution of the bench, and declared here rather
    /// than denied.
    enum Staged<T> {
        Host(Vec<T>),
        Dev(cudarc::driver::CudaSlice<T>),
    }

    impl<T> Staged<T> {
        /// The device buffer. Named panic if the arm was not uploaded: a
        /// launch on an unstaged arm is a phase-sequencing bug, never a case
        /// to be recovered from in silence.
        fn dev(&self) -> &cudarc::driver::CudaSlice<T> {
            match self {
                Staged::Dev(d) => d,
                Staged::Host(_) => panic!("arm not uploaded, phase bug"),
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

    // ---- a selected arm = one Option built, and nothing else ----
    //
    // Every arm carries its buffers AND its byte accounting: a discarded arm
    // has neither, so no table row can cite an arm that did not run.

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

    /// SHARED by the golay70v1 and golay70v2 arms: not one format byte
    /// distinguishes them, so a single set of buffers serves both, and the v1
    /// witness arm costs no VRAM.
    struct G70Arm {
        words: Staged<u32>,
        exc_idx: Staged<u32>,
        exc_words: Staged<u32>,
        n_exc: usize,
        bytes: u64,
    }

    /// P1c. The row-aligned E1v stream, its bases table, and the number of
    /// blocks per ROW it was cut on, the same as the kernel's `nblocks`,
    /// carried here rather than recomputed at launch.
    struct E1vArm {
        data: Staged<u32>,
        bases: Staged<u32>,
        row_blocks: usize,
        bytes: u64,
    }

    // ---- the competitor arm (AWQ w4g128) ----
    //
    // Three buffers and its own reference: its content is no other arm's, so
    // neither `y_ref` (the LLVQ content) nor `y16_ref` (the same weights in
    // binary16) describes it. Same situation as the FP16 arm, same answer.
    struct AwqArm {
        w: Staged<u32>,
        z: Staged<u32>,
        s: Staged<u16>,
        bytes: u64,
        y_ref: Vec<f64>,
        /// The arm's OWN error denominator: `Σ|w·x|` over its own weights.
        /// Reusing `scale`, computed on the LLVQ weights, would measure AWQ's
        /// error by another format's rule.
        scale: Vec<f64>,
    }

    /// The 2-bit competitor. Its payload is **synthetic and is not a
    /// quantization of our weights** — a stronger caveat than the AWQ arm's,
    /// and it is a property of the code, not a shortcut. QTIP is a fixed-rate
    /// trellis: every bit pattern is a valid codeword, so a pseudo-random
    /// buffer is a legitimate input, but encoding *given* weights would need
    /// their Viterbi search. This arm therefore makes **no quality claim at
    /// all**; it measures time and nothing else.
    ///
    /// The timing stands regardless: the kernel has no data-dependent branch
    /// and its traffic is a function of the shapes alone.
    struct QtipArm {
        /// The trellis stream, two u16 per u32 in the order the kernel reads
        /// them (it casts the pointer to `const uint16_t*`).
        compressed: Staged<u32>,
        /// 512 half2 entries, flat as 1024 binary16 words.
        codebook: Staged<u16>,
        bytes: u64,
        /// Exact by construction: we know the bits, so we know the state, so we
        /// know the codebook entry and the sign.
        y_ref: Vec<f64>,
        /// `Σ|w·x|` over **its own** decoded weights — reusing the LLVQ scale
        /// would judge this arm by another format's rule.
        scale: Vec<f64>,
        /// The shim this shape resolves to, e.g. `qtip_mv_4096x2560`.
        kernel: String,
    }

    /// The served Tetra arm: the second file's stream and its own everything.
    ///
    /// It carries its own `gscale`, `rscale` and `tail` — a `Mat` already has
    /// all three, and they belong to the FIRST file. Same projection, same
    /// shape, different quantization: reusing the first file's would decode
    /// this stream against another file's scales, which is a silent wrong
    /// number and not a crash. Its reference and its error denominator are its
    /// own for the same reason the AWQ and QTIP arms' are.
    struct TetraArm {
        /// Little-endian, **row-strided** words. Not flat: a 6-byte record at
        /// `6·b` is not 4-aligned, so a row must start on a u32 boundary or
        /// every row after the first reads at a shifted phase.
        words: Staged<u32>,
        /// The stream's OWN stride, never recomputed at launch — the E1v
        /// argument, and it is asserted against `tetra48::stride_u32` below.
        stride_u32: u32,
        gscale: cudarc::driver::CudaSlice<f32>,
        rscale: cudarc::driver::CudaSlice<f32>,
        tail: cudarc::driver::CudaSlice<f32>,
        bytes: u64,
        /// The row padding, which `bytes` deliberately does NOT include.
        ///
        /// It is real VRAM — the stream allocated and uploaded IS the padded
        /// one — and it is not billed, exactly as the slot arm's 20-byte pad
        /// and the planes arm's 4-byte pad are not. Carried so the run can say
        /// how much VRAM it holds without billing it into a rate, which is the
        /// difference between the two accountings this repository refuses to
        /// mix.
        pad_bytes: u64,
        /// Weights this arm covers. NOT the matrix's: `v_proj` is int4 in the
        /// served file and has no lattice arm, so the Tetra arm's b/weight
        /// denominator is its own and every arm's column is now formed from
        /// the weights that arm actually read.
        weights: u64,
        y_ref: Vec<f64>,
        scale: Vec<f64>,
    }

    /// One matrix of the **second** file, already in the served stream.
    ///
    /// The indices and gains are dropped as soon as they are transcoded: the
    /// reference below decodes the packed words back through
    /// `Tetra48Blocks::decode_block`, which is the stronger check anyway — a
    /// mistake in the packing is then a disagreement between two routes and
    /// not an error both sides share. It also halves what this list costs:
    /// `srcs` already holds ~1.8 GB of host RAM for the first file.
    struct TetraSrc {
        d_out: usize,
        d_in: usize,
        blocks: llvq_artifact::tetra48::Tetra48Blocks,
        /// The **second** file's centroid pair, not the first's. Same
        /// projection, different quantization, different gain scale.
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
        slot: Option<SlotArm>,
        planes: Option<PlanesArm>,
        p12: Option<P12Arm>,
        g70: Option<G70Arm>,
        e1v: Option<E1vArm>,
        tetra: Option<TetraArm>,
        awq: Option<AwqArm>,
        qtip: Option<QtipArm>,
        // The FP16 witness and the shared inputs, always built: fp16 cannot
        // be deselected (arms::parse_phases refuses it).
        gscale: cudarc::driver::CudaSlice<f32>,
        rscale: cudarc::driver::CudaSlice<f32>,
        tail: cudarc::driver::CudaSlice<f32>,
        w16: cudarc::driver::CudaSlice<u16>,
        f16_bytes: u64,
        y_ref: Vec<f64>,
        y16_ref: Vec<f64>,
        /// The cublasf16 arm's reference: f16 weights × binary16 input, built
        /// only when the arm is.
        y16h_ref: Option<Vec<f64>>,
        scale: Vec<f64>,
    }

    /// The bytes an arm reads for one matrix, in the bench's accounting. 0
    /// for an arm that was not built, which has no table row to print it in
    /// anyway. v1 and v2 share the golay70 row: same buffers, same bytes, by
    /// construction.
    /// `1/sqrt(16 m)`, entry 0 the origin — reconstructed without a branch and
    /// without a division. `m ≤ 27` on this codebook, derived over the whole
    /// table by `llvq-bench/examples/tetrashell.rs`; 32 is that bound rounded
    /// up, and `llvq_tetra48.cuh` masks with it. Same table `bin/f1rankfloor`
    /// builds, same three lines, because it is the same kernel.
    fn invnorm_table() -> Vec<f32> {
        let mut t = vec![0.0f32; llvq_artifact::tetra48::TETRA48_SHELLS];
        for (m, e) in t.iter_mut().enumerate().skip(1) {
            *e = (1.0f64 / ((16 * m) as f64).sqrt()) as f32;
        }
        t
    }

    /// Bytes into little-endian u32, zero-padded — the rank table's prefix and
    /// suffix streams are `u8` and the upload path is `u32`.
    fn pack_u8(b: &[u8]) -> Vec<u32> {
        b.chunks(4)
            .map(|c| {
                let mut w = [0u8; 4];
                w[..c.len()].copy_from_slice(c);
                u32::from_le_bytes(w)
            })
            .collect()
    }

    /// The four rank-table buffers the F1 decoder walks, plus the two
    /// constants the served Tetra decode adds. Uploaded once for the run.
    /// A device pointer as the persistent sections take it.
    fn dptr<T>(s: &cudarc::driver::CudaSlice<T>, cuda: &Cuda) -> u64 {
        use cudarc::driver::DevicePtr;
        s.device_ptr(cuda.stream()).0
    }

    struct TetraTabs {
        rows: cudarc::driver::CudaSlice<u32>,
        prefixes: cudarc::driver::CudaSlice<u32>,
        branches: cudarc::driver::CudaSlice<u16>,
        suffixes: cudarc::driver::CudaSlice<u32>,
        invnorm: cudarc::driver::CudaSlice<f32>,
    }

    /// The weights an arm's bytes cover, for one matrix.
    ///
    /// 🕳️ It was `n_weights`, one accumulator for the whole table, and that
    /// was correct only while every arm read every matrix. The served Tetra
    /// arm does not: 36 of the 252 projections are `v_proj`, stored as int4
    /// g128 and read by another kernel. Dividing its bytes by all 252
    /// projections' weights understates its b/weight by the `v_proj` share —
    /// 94,371,840 of 3,633,315,840 weights, **2.597 %** (*computed*, from the
    /// seven shapes over 36 layers), which turns a true 2.1576 into a printed
    /// 2.1017. Both clear an F1d gate of 2.20, so it would not have flipped
    /// the verdict; it would have published a wrong number, which is worse for
    /// being survivable.
    ///
    /// Every arm now divides by what it actually read.
    fn arm_weights(m: &Mat, a: usize) -> u64 {
        match a {
            arms::TETRA48 => m.tetra.as_ref().map_or(0, |x| x.weights),
            _ => (m.d_out * m.d_in) as u64,
        }
    }

    fn arm_bytes(m: &Mat, a: usize) -> u64 {
        match a {
            arms::SLOT32 => m.slot.as_ref().map_or(0, |x| x.bytes),
            arms::PLANES14 => m.planes.as_ref().map_or(0, |x| x.bytes),
            arms::PLANES12X => m.p12.as_ref().map_or(0, |x| x.bytes),
            arms::GOLAY70V1 | arms::GOLAY70V2 => m.g70.as_ref().map_or(0, |x| x.bytes),
            arms::E1V => m.e1v.as_ref().map_or(0, |x| x.bytes),
            arms::TETRA48 => m.tetra.as_ref().map_or(0, |x| x.bytes),
            // The floor reads NO weight, which is its definition. What is
            // left is what every LLVQ arm uploads anyway: the f32 tail and
            // the row scales. Its table row will therefore show a near-zero
            // b/weight, and that is not an anomaly, it is the point.
            arms::NULLK => ((m.d_out * m.tail_w) as u64 + m.d_out as u64) * 4,
            arms::FP16 => m.f16_bytes,
            // Same bytes as the witness: cublasf16 reads the SAME w16.
            arms::CUBLASF16 => m.f16_bytes,
            arms::AWQ => m.awq.as_ref().map_or(0, |x| x.bytes),
            arms::QTIP => m.qtip.as_ref().map_or(0, |x| x.bytes),
            _ => unreachable!("unknown arm"),
        }
    }

    use llvq_cuda::stats::worst_error;

    use llvq_cuda::stats::spread;

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

    /// `tv_f1r_v3g(words, row_stride_u32, rows, prefixes, branches, suffixes,
    /// gscale, invnorm, rscale, tail, x, y, nblocks, tail_w)` — the served
    /// Tetra decode.
    ///
    /// Two things differ from every ball arm above, and both are the layout.
    /// The words are **row-strided**, so the kernel takes a stride where the
    /// others take nothing: a 6-byte record at `6·b` is not 4-aligned, and a
    /// row that did not start on a u32 boundary would read every block after
    /// the first at a shifted phase. And the four table pointers are the F1
    /// rank decoder's, not a `ClassRec` table: Tetra resolves a magnitude
    /// through a trellis rank, where the ball layouts index a class.
    #[allow(clippy::too_many_arguments)]
    fn launch_tetra48(
        cuda: &Cuda,
        f: &cudarc::driver::CudaFunction,
        words: &cudarc::driver::CudaSlice<u32>,
        row_stride_u32: u32,
        tabs: &TetraTabs,
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
        b.arg(words).arg(&row_stride_u32)
            .arg(&tabs.rows).arg(&tabs.prefixes).arg(&tabs.branches).arg(&tabs.suffixes)
            .arg(gscale).arg(&tabs.invnorm).arg(rscale).arg(tail)
            .arg(x).arg(y).arg(&nblocks).arg(&tail_w);
        unsafe { b.launch(cfg) }.map_err(|e| format!("tv_f1r_v3g: {e}"))?;
        Ok(())
    }

    /// `tv_tetra48_seg` — `launch_tetra48` with one pointer more.
    ///
    /// The only difference is `gs_off`, inserted right after `gscale`, because
    /// a row-concatenation of Tetra matrices carries one gain pair per segment
    /// and the kernel resolves which pair a row owns. Everything else is
    /// identical, deliberately: the words concatenate without re-encoding —
    /// segments sharing `d_in` share `nblocks` and therefore `row_stride_u32` —
    /// and `invnorm` is a property of the lattice rather than of a projection,
    /// so both are passed exactly as the unfused arm passes them.
    #[allow(clippy::too_many_arguments)]
    fn launch_tetra48_seg(
        cuda: &Cuda,
        f: &cudarc::driver::CudaFunction,
        words: &cudarc::driver::CudaSlice<u32>,
        row_stride_u32: u32,
        tabs: &TetraTabs,
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
        b.arg(words).arg(&row_stride_u32)
            .arg(&tabs.rows).arg(&tabs.prefixes).arg(&tabs.branches).arg(&tabs.suffixes)
            .arg(gscale).arg(gs_off).arg(&tabs.invnorm).arg(rscale).arg(tail)
            .arg(x).arg(y).arg(&nblocks).arg(&tail_w);
        unsafe { b.launch(cfg) }.map_err(|e| format!("tv_tetra48_seg: {e}"))?;
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

    /// `tv_nullk(rscale, tail, x, y, nblocks, tail_w)`: the floor. Same grid,
    /// same tiling, same epilogue, no weight buffer, so it has neither a
    /// `Staged` nor an arm struct, only a launch.
    #[allow(clippy::too_many_arguments)]
    fn launch_nullk(
        cuda: &Cuda,
        f: &cudarc::driver::CudaFunction,
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
        b.arg(rscale).arg(tail).arg(x).arg(y).arg(&nblocks).arg(&tail_w);
        unsafe { b.launch(cfg) }.map_err(|e| format!("tv_nullk: {e}"))?;
        Ok(())
    }

    /// `tv_e1v(data, bases, tab, pay, binom, golay, gscale, rscale, tail, x, y,
    /// nblocks, tail_w)`: the P1c arm. Same grid and same epilogue as
    /// `tv_planes`, a single launch, no correction region and no zeroed `y`.
    /// E1v has no exceptions, every row writes its own.
    ///
    /// ⚠️ `nblocks` is the number of blocks per ROW, and it is the value the
    /// stream was cut on. It comes from `E1vMat::row_blocks` and not from a
    /// second computation: two values would move every group boundary and the
    /// decode would go wrong from the second row on.
    #[allow(clippy::too_many_arguments)]
    fn launch_e1v(
        cuda: &Cuda,
        f: &cudarc::driver::CudaFunction,
        data: &cudarc::driver::CudaSlice<u32>,
        bases: &cudarc::driver::CudaSlice<u32>,
        tab: &cudarc::driver::CudaSlice<u32>,
        pay: &cudarc::driver::CudaSlice<u32>,
        binom: &cudarc::driver::CudaSlice<u32>,
        golay: &cudarc::driver::CudaSlice<u32>,
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
        b.arg(data).arg(bases).arg(tab).arg(pay).arg(binom).arg(golay).arg(gscale)
            .arg(rscale).arg(tail).arg(x).arg(y).arg(&nblocks).arg(&tail_w);
        unsafe { b.launch(cfg) }.map_err(|e| format!("tv_e1v: {e}"))?;
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

    // ================= the competitor arm: AWQ 4-bit, w4g128 =================
    //
    // A group of 128 input channels, one binary16 scale factor and one 4-bit
    // zero per group, four bits per weight. This is the configuration of the
    // official Qwen3 AWQ checkpoints, hence the only one this arm carries.
    const AWQ_GROUP: usize = 128;

    /// The three row strides `awq_gemv_g128` computes for itself. Copied here
    /// because the host must allocate and fill exactly what the kernel will
    /// read, and two of them are **padded**, which is neither cosmetic nor
    /// guessable from the shapes.
    fn awq_strides(d_in: usize) -> (usize, usize, usize) {
        let g = d_in / AWQ_GROUP; // actual groups
        let npg = g.div_ceil(8); // `num_groups_packed`
        (d_in / 8, npg, npg * 8) // weight_w, zeros_w, sf_w
    }

    /// Bytes an AWQ arm reads for one matrix, **in the bench's accounting**:
    /// the stream and nothing else, structural padding included because the
    /// kernel really does index it.
    ///
    /// No tail and no row scale here, unlike the LLVQ arms: w4g128 quantizes
    /// *every* column, it has no tail policy. That is a real difference
    /// between the formats, not a billing oversight, and it favours AWQ, so
    /// it is declared.
    fn awq_bytes(d_out: usize, d_in: usize) -> u64 {
        let (ww, zw, sw) = awq_strides(d_in);
        (d_out * ww * 4 + d_out * zw * 4 + d_out * sw * 2) as u64
    }

    /// Quantizes one row of weights into w4g128, packs it into the kernel's
    /// three buffers, and returns the EXACT f64 dot product of what the kernel
    /// will decode against the activation **as it sees it**.
    ///
    /// ## Why the reference is computed here and nowhere else
    ///
    /// The AWQ arm decodes a content that is no other arm's: neither `y_ref`
    /// (the LLVQ content) nor `y16_ref` (the same weights in binary16)
    /// describes it. It needs its own, on the FP16 arm's model, and that one
    /// is **exact by construction**: we know `q`, `scale` and `zero`, so
    /// `scale·(q − zero)` is the weight the kernel will reconstruct, bit for
    /// bit, with no approximation to bound.
    ///
    /// ⚠️ `xf` is the activation **rounded to binary16 then widened back**,
    /// because that is what the kernel reads: its inputs are `float4` of eight
    /// binary16. Using the f32 activation here would inflate the measured
    /// error by a deviation that is not the arm's.
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
            // Asymmetric 4-bit scale and zero, the AWQ convention:
            // `w ≈ scale·(q − zero)` with `q ∈ [0, 15]`. A zero scale happens
            // only on a constant group; we force it non-zero so the division
            // exists, and the zero then absorbs the value.
            let mut scale = (hi - lo) / 15.0;
            // `!(scale > 0.0)` and not `scale <= 0.0`: the negated form also
            // catches a NaN (a NaN weight upstream), which the positive form
            // would let through into the division.
            #[allow(clippy::neg_cmp_op_on_partial_ord)]
            if !(scale > 0.0) {
                scale = 1.0;
            }
            // The scale makes a binary16 round trip BEFORE serving as the
            // reference: that is the one the kernel will read, not the f64.
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

    /// `qtip_mv_<M>x<K>(out, compressed, x, codebook)` — the shim from
    /// `kernels/qtip_glue.cu`.
    ///
    /// The geometry is **theirs**, copied from `decompress_matvec_ptr`:
    /// `<<<128, 1024, 1 << (S + 5 + V + 1)>>>`. Changing it would make this arm
    /// a measurement of our tuning rather than of their kernel — the same rule
    /// the AWQ arm follows.
    ///
    /// ⚠️ The 64 KiB codebook is **dynamic** shared memory, above the 48 KiB
    /// per-block default, so `f` must have come from
    /// `Cuda::func_dynamic_shared(name, QTIP_SHARED_BYTES)` and not from
    /// `Cuda::func`. A function obtained the wrong way compiles and fails at
    /// launch.
    fn launch_qtip(
        cuda: &Cuda,
        f: &cudarc::driver::CudaFunction,
        y: &mut cudarc::driver::CudaSlice<f32>,
        compressed: &cudarc::driver::CudaSlice<u32>,
        xh: &cudarc::driver::CudaSlice<u16>,
        codebook: &cudarc::driver::CudaSlice<u16>,
    ) -> Result<(), String> {
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (128, 1, 1),
            block_dim: (1024, 1, 1),
            shared_mem_bytes: llvq_cuda::qtip_host::QTIP_SHARED_BYTES,
        };
        let mut b = cuda.stream().launch_builder(f);
        b.arg(y).arg(compressed).arg(xh).arg(codebook);
        unsafe { b.launch(cfg) }.map_err(|e| format!("qtip_mv: {e}"))?;
        Ok(())
    }

    /// The trellis stream as the kernel addresses it: it casts `compressed` to
    /// `const uint16_t*`, so u16 index `i` must sit at byte `2i`. On a
    /// little-endian device that is the low half of u32 word `i / 2`.
    ///
    /// A tile is 32 u16, so the length is always even and the last word is
    /// never half-filled — asserted rather than assumed.
    fn qtip_pack_u32(buf: &[u16]) -> Vec<u32> {
        assert!(buf.len().is_multiple_of(2), "QTIP buffer length must be even");
        buf.chunks_exact(2).map(|c| c[0] as u32 | ((c[1] as u32) << 16)).collect()
    }

    /// `awq_gemv_g128(inputs, weight, zeros, scaling_factors, outputs, IC, OC)`.
    ///
    /// The geometry is **theirs**, copied from `gemv_forward_cuda`:
    /// `dim3 num_blocks(1, OC/4, B)` and `dim3 num_threads(32, 4)`, one warp
    /// per output channel and four channels per block. Changing it would make
    /// this arm a measurement of our tuning, not of their kernel.
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
        assert_eq!(d_out % 4, 0, "AWQ: OC/4 blocks, OC must be a multiple of 4");
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
        // The arm selection, BEFORE any build: an invalid value must fail
        // here, not after twenty minutes of transcoding.
        // ⚠️ It does NOT touch the NVRTC translation unit below. The selection
        // changes the buffers and the dispatch, never the compiled text nor
        // the register report (the opposite would redo É1).
        let arms_var = std::env::var("LLVQ_BENCH_ARMS").ok();
        let mut phases = arms::parse_phases(arms_var.as_deref())?;
        // The Tetra arm needs a SECOND file, and `runnable()` cannot know
        // whether this invocation was given one: it says a kernel exists, not
        // that anything can feed it.
        //
        // A bare `planesbench <ball.llvq>` is the invocation every published
        // run used, so a DEFAULT selection is narrowed rather than refused —
        // loudly, on its own line, because an arm that vanishes without a word
        // is the failure this whole step is about. An EXPLICIT
        // `LLVQ_BENCH_ARMS=…,tetra48,…` is refused instead, further down: that
        // is a selection the operator made, and narrowing it would be making
        // a different one for them.
        let tetra_path = std::env::args().nth(2);
        if tetra_path.is_none() && arms_var.is_none() {
            for p in phases.iter_mut() {
                p.remove(arms::TETRA48);
            }
            println!(
                "no second file: the tetra48 arm is OUT of this run. It reads a Tetra file, \
                 which\n  cannot be the same file as the ball arms' — pass it as the second \
                 argument."
            );
        }
        let phases = phases;
        let union: ArmSet = {
            let mut u = ArmSet::empty();
            for p in &phases {
                for a in p.iter() {
                    u.insert(a);
                }
            }
            u
        };
        // 🚨 Every arm this binary can DISPATCH, and nothing else.
        //
        // `arms::HAS_KERNEL` says an arm's kernel exists **in the repository**;
        // it does not say this binary has a branch for it, and the two drifted
        // on 2026-09-09. `HAS_KERNEL[TETRA48]` went true when the arm entered
        // the registry, `parse_phases(None)` returns `ArmSet::runnable()`, and
        // for one day a bare `planesbench <model.llvq>` selected arm 17 and
        // reached `unreachable!("unknown arm")` — after the three minutes of
        // transcoding, on a rented card, with nothing on the development Mac
        // able to see it: this file is entirely `cfg(target_os = "linux")` and
        // the failure is a runtime panic, not a type error.
        //
        // So the list is stated here and checked in the first second. It is
        // hand-written because nothing can derive it from three `match` arms —
        // but a hand-written list that fails loud and early is worth a great
        // deal more than a derived one that fails late.
        for a in union.iter() {
            let dispatched = matches!(
                a,
                arms::SLOT32
                    | arms::PLANES14
                    | arms::PLANES12X
                    | arms::GOLAY70V1
                    | arms::GOLAY70V2
                    | arms::E1V
                    | arms::TETRA48
                    | arms::NULLK
                    | arms::FP16
                    | arms::CUBLASF16
                    | arms::AWQ
                    | arms::QTIP
            );
            if !dispatched {
                return Err(format!(
                    "planesbench cannot dispatch \"{}\" (arm {a}). It is registered in \
                     `arms.rs` and `HAS_KERNEL` says its kernel exists, but this binary has \
                     no branch for it — wire it, or take it out of the selection. Refused \
                     here rather than in `unreachable!` three minutes from now.",
                    arms::ARM_NAMES[a]
                ));
            }
        }
        if arms_var.is_some() {
            println!("LLVQ_BENCH_ARMS selection:");
            for (k, p) in phases.iter().enumerate() {
                println!("  phase {}/{}: {}", k + 1, phases.len(), p.label());
            }
        }
        // The tile, resolved HERE, before any transcoding:
        // it opens the card, and a refusal that fires after three minutes of
        // transcoding on a rented card is worth nothing. It has to precede the
        // source anyway — it is a `#define` in it.
        let tile = llvq_cuda::tile::resolve(llvq_cuda::gpu::probe_compute_cap()?);
        println!("{}", tile.provenance());

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
        // The Tetra fusion arm, OFF by default. Its bit-exact comparison is
        // fatal when it runs — that is the whole point of it — and on
        // 2026-09-20 it fired and took the ten-arm cost table down with it,
        // before the table was printed. An experimental arm must not be able
        // to do that to the table the bench exists for.
        //
        // Refused by name on any other value, like every flag in this file.
        let seg_tetra = match std::env::var("LLVQ_SEG_TETRA").as_deref() {
            Err(_) | Ok("0") => false,
            Ok("1") => true,
            Ok(v) => {
                return Err(format!(
                    "LLVQ_SEG_TETRA={v:?}: the values are 0 and 1, and unset is 0"
                ))
            }
        };
        // The persistent sections (llvq_cuda::persbench): L1's one-launch
        // Tetra pass, the row-loop microbenchmark, the driver cubins. Off
        // unless asked; read here, before anything is built.
        let pers_flags = llvq_cuda::persbench::Flags::from_env(seg_tetra, union.has(arms::TETRA48))?;
        if pers_flags.any() || pers_flags.dump_cubin.is_some() {
            println!(
                "Persistent sections: LLVQ_TETRA_OCC={} LLVQ_TROWS={} LLVQ_DUMP_CUBIN={}",
                u8::from(pers_flags.tetra_occ),
                u8::from(pers_flags.trows),
                pers_flags.dump_cubin.as_ref().map_or("unset".to_string(), |d| d.display().to_string())
            );
        }
        let (awqcu, awq_overridden) = load_awq_source()?;
        let (gv1cu, golay_v1_overridden) = load_golay_v1_source()?;
        // P1c. Same loader as the base pair, so `LLVQ_KERNEL_DIR` overrides it
        // the same way and the override is reported below like every other.
        let e1v = llvq_cuda::load_sources_many(&["llvq_e1v.cuh", "e1v.cu"])?;
        // The floor (P4 §2.5). It has no header of its own: `matvec.cu` is
        // enough, and it is concatenated after everything else like every
        // arrival.
        let nullk = llvq_cuda::load_sources_many(&["nullk.cu"])?;
        // QTIP (F2). Absent unless `ops/fetch-qtip.sh` has run and
        // `LLVQ_KERNEL_DIR` points at its output; the empty strings below then
        // contribute nothing to the translation unit — and, deliberately,
        // nothing to its hash either, so a machine without the fetch compiles
        // the SAME source as every published run.
        // The F1 family and the served Tetra decode (6.8a, 2026-09-10).
        // Always in the unit, selected or not: the unit
        // never varies with the selection, so every arm's sha256 moves
        // together or none does.
        //
        // The order is the caller's contract, stated identically in two files
        // (`tv_tetra48_h.cu:36-37`, `tetra48_v3g.cu:24-25`): llvq_slot.cuh,
        // matvec.cu, llvq_f1rank.cuh, llvq_f1rank_v3.cuh, llvq_tetra48.cuh,
        // then the arm. The first two are already `base.parts[0..1]` at the
        // head, so the four below go at the END of the unit — adding an arm
        // must never move a fragment of an arm that carries a published
        // number, and appending is the only placement that cannot.
        //
        // `f1rank.cu` and the v1/v2 variants are NOT here: `tetra48_v3g.cu`
        // names only the three headers, and `cuhcheck` assembles exactly that
        // unit.
        let f1 = llvq_cuda::load_sources_many(&[
            "llvq_f1rank.cuh",
            "llvq_f1rank_v3.cuh",
            "llvq_tetra48.cuh",
            "tetra48_v3g.cu",
            // A4 for Tetra, appended after the served arm for the rule that
            // governs every arrival: adding a kernel must never move a
            // fragment of an arm that carries a published number.
            "tetra48_seg.cu",
        ])?;
        let qtip_src = load_qtip_sources()?;
        let (qtip_cuh, qtip_glue) = match &qtip_src {
            Some((cuh, glue)) => (cuh.as_str(), glue.as_str()),
            None => ("", ""),
        };
        let defines = tile.define();
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
            // The competitor last: adding an arm must never reorder the
            // fragments of the in-house arms.
            awqcu.as_str(),
            // Then the v1 witness (2026-08-11), latest arrival, same rule.
            gv1cu.as_str(),
            // And E1v last (P1c, 2026-08-16): the same rule again, and
            // `e1v.cu` needs `matvec.cu` (warp_sum, TILE_BLOCKS), already
            // concatenated at the head.
            e1v.parts[0].as_str(),
            e1v.parts[1].as_str(),
            nullk.parts[0].as_str(),
            // QTIP last, and for the rule that has governed every arrival:
            // adding an arm must never reorder the fragments of the arms that
            // produced a published number. The device half first, then our
            // shims, which name it.
            qtip_cuh,
            qtip_glue,
            // Last of all, and after QTIP for the same rule QTIP arrived
            // under. Nothing before this line moves by one notch.
            f1.parts[0].as_str(),
            f1.parts[1].as_str(),
            f1.parts[2].as_str(),
            f1.parts[3].as_str(),
            f1.parts[4].as_str(),
        ];
        let src = KernelSource::new(&parts);
        // On the assembled text: the one place where what NVRTC compiles and
        // what this report prints are both visible.
        tile.assert_defined_in(&src.text);
        println!("NVRTC source: {} bytes, sha256 {}", src.text.len(), src.sha256);
        if let Some(d) = &base.overridden_from {
            println!("  WARNING: Slot32 SOURCES OVERRIDDEN from {d}");
        }
        // The four F1/Tetra fragments were the only ones whose override was
        // never announced. `load_sources_many` returns it and it was dropped:
        // every `tetra48` figure — b/weight, ms, GB/s, the ratio, the register
        // report on `tv_f1r_v3g` — would have been attributed to the committed
        // kernel while a substituted one produced them.
        if let Some(d) = &f1.overridden_from {
            println!("  WARNING: F1/Tetra48 SOURCES OVERRIDDEN from {d}");
        }
        if let Some(d) = &planes_overridden {
            println!("  WARNING: Planes14 SOURCES OVERRIDDEN from {d}");
        }
        if let Some(d) = &planes12_overridden {
            println!("  WARNING: Planes12x SOURCES OVERRIDDEN from {d}");
        }
        if let Some(d) = &awq_overridden {
            println!("  WARNING: AWQ SOURCE OVERRIDDEN from {d}");
        }
        if let Some(d) = &golay_overridden {
            println!("  WARNING: Golay70 SOURCES OVERRIDDEN from {d}");
        }
        if let Some(d) = &golay_v1_overridden {
            println!("  WARNING: Golay70 v1 WITNESS SOURCE OVERRIDDEN from {d}");
        }
        if let Some(d) = &seg_overridden {
            println!("  WARNING: segmented Planes14 SOURCE OVERRIDDEN from {d}");
        }
        if let Some(d) = &e1v.overridden_from {
            println!("  WARNING: E1v SOURCES OVERRIDDEN from {d}");
        }
        if let Some(d) = &nullk.overridden_from {
            println!("  WARNING: nullk SOURCE OVERRIDDEN from {d}");
        }
        match &qtip_src {
            Some(_) => println!(
                "  WARNING: QTIP KERNEL LOADED (GPL v3, not redistributed by this \
                 repository, see docs/qtip-provenance.md)"
            ),
            None => {
                if union.has(arms::QTIP) {
                    return Err("the qtip arm is selected but its kernel is absent: run \
                                ops/fetch-qtip.sh and point LLVQ_QTIP_DIR at it"
                        .to_string());
                }
            }
        }

        // Built before the module so a shape without a shim is a Rust-side
        // error message, never a failed symbol lookup on a rented card.
        let qtip_names: Vec<String> = match &qtip_src {
            None => Vec::new(),
            Some(_) => llvq_cuda::qtip_host::QTIP_SHAPES
                .iter()
                .map(|&(d_out, d_in)| {
                    llvq_cuda::qtip_host::kernel_name(d_out, d_in)
                        .expect("QTIP_SHAPES must name its own shims")
                })
                .collect(),
        };

        let cuda = Cuda::new(&src)?;
        let dev = cuda.device()?;
        println!(
            "\n{}: {} SM, L2 {:.1} MB (read), {} B of shared memory per block",
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
            // 🚨 `tv_e1v` is here for the `local_bytes != 0` line just below,
            // not for the table: the FLAT body was chosen over a body 24%
            // faster on Metal precisely so that nothing spills, and if this
            // kernel spills then that choice was wrong and the number measures
            // something else (E1v-CUDA preregistration §5.4).
            "tv_e1v",
            "tv_nullk",
            // The served Tetra decode. A kernel of OURS, so it falls into the
            // hard-stop branch below and a spill kills the run — which is what
            // F1d wants: the 6.4 kill read `num_regs ≤ 40` and
            // `local_bytes == 0` off this very function, and this is the only
            // evidence that NVRTC accepts it in a translation unit this much
            // larger than `f1rankfloor`'s.
            "tv_f1r_v3g",
        ]
        .iter()
        .copied()
        // The QTIP shims, when their source is present. They are reported for
        // the same reason `tv_e1v` is — the `local_bytes != 0` line below —
        // and for one more that is specific to this arm: **this report is the
        // only proof that NVRTC accepted the fetched kernel text at all**.
        // Everything about QTIP up to here was established by reading and by
        // host-side tests; whether libcu++-free inline PTX, `__shfl_sync` and
        // `__byte_perm` survive NVRTC is a question only a device compile
        // answers, and a successful lookup of these five names IS that answer.
        // They are listed even while `arms::HAS_KERNEL[QTIP]` is false,
        // deliberately: the translation unit never varies with arm selection
        // (`arms.rs`), so the compile can be proved a job before the arm is
        // allowed to run.
        .chain(qtip_names.iter().map(|s| s.as_str()))
        {
            let r = cuda.report(name)?;
            println!(
                "  {:<10} {:>3} registers, {} local bytes, sm_{}",
                r.name, r.num_regs, r.local_bytes, r.binary_version
            );
            if r.local_bytes != 0 {
                // 🚨 A spill in OUR kernel stops everything: it means a design
                // choice is wrong and the number would measure something
                // else. In a COMPETITOR's kernel ported as shipped it is a
                // FACT to report, not a defect to fix, their tuning at our
                // occupancy. Refusing to measure would be choosing not to
                // know.
                if qtip_names.iter().any(|q| q == name) {
                    println!(
                        "  WARNING: {name} SPILLS {} B, competitor arm ported as shipped, \
                         measured anyway and declared (F2 prereg §7 A1)",
                        r.local_bytes
                    );
                } else {
                    return Err(format!("{name}: {} spill bytes", r.local_bytes));
                }
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
        assert!(24 + table.worst_width_slot() <= 160, "five-word window exceeded");
        // The Planes14 bound: three bit-planes address 8 levels; the layout
        // is only bijective while every class has at most 5.
        assert!(
            (0..table.n_entries()).all(|e| table.record(e).len <= 5),
            "a class exceeds 5 levels: 3 planes are no longer enough"
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

        // The served Tetra decode's tables: the universal 16 KiB rank table,
        // the trellis's three streams, and `1/sqrt(16 m)`. Built by the
        // reference (`llvq_bench::f1`) and uploaded once — `bin/f1rankfloor`
        // builds them from the same four calls, and a second construction here
        // would be a second thing to keep in step.
        let tetra_tabs = match union.has(arms::TETRA48) {
            false => None,
            true => {
                let tr = llvq_bench::f1::Trellis::new();
                let table = llvq_bench::f1::rank::RankTable::build();
                assert_eq!(table.rows.len(), 4096, "the rank table is not 4096 rows");
                assert_eq!(
                    table.n0_mixed, 1240,
                    "RankTable::n0_mixed is {}, F1R_N0_MIXED in llvq_f1rank.cuh is 1240: \
                     the split of the mixed order differs",
                    table.n0_mixed
                );
                println!(
                    "tetra48: rank table {} B, prefixes 128 B, branches 2048 B, suffixes 128 B, \
                     invnorm {} B, N0 = {}",
                    table.rows.len() * 4,
                    llvq_artifact::tetra48::TETRA48_SHELLS * 4,
                    table.n0_mixed
                );
                Some(TetraTabs {
                    rows: cuda.up_u32(&table.rows)?,
                    prefixes: cuda.up_u32(&pack_u8(&llvq_bench::f1::rank::prefix_bytes(&tr)))?,
                    branches: cuda.up_u16(&llvq_bench::f1::rank::branch_words(&tr))?,
                    suffixes: cuda.up_u32(&pack_u8(&llvq_bench::f1::rank::suffix_bytes(&tr)))?,
                    invnorm: cuda.up_f32(&invnorm_table())?,
                })
            }
        };

        // The Golay70 arm's two constant tables: the dedicated class table
        // (residue pairs / level values + mod-4 flags + coset flag) and the
        // canonical 4096-codeword table (16 KiB) the 12-bit rank resolves
        // through. Uploaded once, shared by every matrix and by BOTH golay70
        // arms — and not at all when neither is selected. ⚠️ Conditional on
        // the UNION of the phases, not on the current phase: unlike the
        // streams (Staged), these ~32 KiB are resident from the start when a
        // golay70 arm enters only at phase 2. Disclosed in the `Staged`
        // comment, along with the three other union-level buffers.
        // (`golay`/`g70cls`, host side, also serve the conditional transcode
        // in `build`.)
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

        // The E1v arm's three constant tables, plus the canonical Golay
        // codewords. Conditional on the UNION like golay70's, and declared
        // here for the same reason: ~54 KiB of records, 2 KiB of widths,
        // 2.5 KiB of binomials and 16 KiB of codewords are resident from the
        // start when the arm enters only at phase 2.
        //
        // 🚨 The Golay words are those of `Golay::codewords()`, NOT golay70's
        // reworked table: `llvq_e1v.cuh` indexes `golay[golay_base + gi]` in
        // the format's canonical order, and `golay70_gpu_codewords` produces
        // another one. Two tables that look alike, and only one is right.
        let (d_e1vtab, d_e1vpay, d_e1vbinom, d_e1vcw) = match union.has(arms::E1V) {
            true => {
                let (tabw, pay, binom) = e1v_tables(&fd, &golay);
                (
                    Some(cuda.up_u32(&tabw)?),
                    Some(cuda.up_u32(&pay)?),
                    Some(cuda.up_u32(&binom)?),
                    Some(cuda.up_u32(golay.codewords())?),
                )
            }
            false => (None, None, None, None),
        };

        let mut rng = SplitMix64::new(0x6_D07);
        // 16,384 until 2026-10-04, which refused the 14B `down_proj` at
        // `d_in` 17,408 and made the 14B unbenchable (census-14b-base).
        //
        // Extending is the one edit here that cannot move a published number.
        // `SplitMix64` is sequential, so the first 16,384 draws are the same
        // values they have always been, and a matrix reads `x[0..d_in]`: every
        // arm at every `d_in` <= 16,384 sees a bit-identical activation.
        // Reseeding or shortening would not have that property.
        const XLEN: usize = 20480;
        let x: Vec<f32> = (0..XLEN).map(|_| rng.next_gaussian() as f32).collect();
        let d_x = cuda.up_f32(&x)?;
        // The activation as the AWQ kernel reads it: binary16, by float4 of
        // eight. The LLVQ and FP16 arms read the f32. That is a format
        // difference between competitors, not a tuning knob, and it is
        // declared. It exists only when an arm with a binary16 input runs:
        // AWQ, since 2026-08-18 cublasf16 (GemmEx requires A and B of the
        // same type, so x must be binary16 like the weights), and since
        // 2026-08-20 qtip, whose kernel reads `const half2* x`.
        //
        // 🚨 This list is a list of ARMS, not a list of two names: the QTIP
        // arm was wired on 2026-08-20 reading `d_xh` without being added
        // here, so the preregistration command, `fp16,qtip` without AWQ,
        // would have panicked on `expect` after the transcode, on a rented
        // card. Found by an adversarial review of the code, not by the
        // typecheck: a missing `Option` is a perfectly typed state.
        let d_xh = match union.has(arms::AWQ)
            || union.has(arms::CUBLASF16)
            || union.has(arms::QTIP)
        {
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
        // ---- the SECOND file, read before the first ----------------------
        //
        // Tetra and Planes14 cannot share a file: they are different code
        // alphabets, and the served Tetra file also carries its `v_proj` as
        // int4 g128. So the bench takes two paths and reads each format from
        // its own — every arm still in ONE process, which is what rule 5 is
        // about, and what F1d promises.
        //
        // Read FIRST, and entirely, for the reason the arm selection is parsed
        // in the first second: a bad path or an unreadable record must kill
        // the job before the three minutes of transcoding the first file
        // costs, not after.
        //
        // `read_record` is the only entry that walks a mixed file
        // (`format.rs:1167`). Every other one — `read_matrix_raw` included,
        // which is what this bench used for four months — refuses an int4
        // record AFTER `get_record_head` has consumed the head, and the body
        // has no length prefix a stranded reader could skip over. That is
        // deliberate (`format.rs:1899-1907`), and it is why the loop below
        // does not simply filter.
        let mut tetra: std::collections::HashMap<String, TetraSrc> =
            std::collections::HashMap::new();
        let mut tetra_int4 = 0usize;
        if let Some(path) = &tetra_path {
            let f = std::fs::File::open(path).map_err(|e| format!("open {path}: {e}"))?;
            let mut r = std::io::BufReader::new(f);
            let h = llvq_artifact::read_header(&mut r).map_err(|e| e.to_string())?;
            for _ in 0..h.matrices {
                match llvq_artifact::read_record(&mut r, h.version).map_err(|e| e.to_string())? {
                    llvq_artifact::Record::Int4(m) => {
                        // Counted, never timed. `v_proj` is stored weights, a
                        // second kernel's business (`tv_q4_h`), and the arm
                        // measured here is the lattice decode. What matters is
                        // that the count reaches the report: an arm covering
                        // 216 of 252 matrices is not comparable, launch for
                        // launch, to one covering 252.
                        tetra_int4 += 1;
                        let _ = m;
                    }
                    llvq_artifact::Record::Lattice(m) => {
                        if m.kind != llvq_artifact::CodeKind::Tetra {
                            return Err(format!(
                                "{path}: {} is a {} record. The second path is the \
                                 Tetra file; the first is the ball one, and swapping \
                                 them is the one mistake that would otherwise run",
                                m.name,
                                m.kind.name()
                            ));
                        }
                        assert_eq!(
                            m.centroids.len(),
                            2,
                            "{}: the tetra decode hard-codes 1 gain bit",
                            m.name
                        );
                        let nblocks = m.d_in / DIM;
                        let blocks = llvq_artifact::tetra48::transcode_tetra48(
                            &m.indices, &m.gains, m.d_out, nblocks,
                        )
                        .map_err(|e| e.to_string())?;
                        tetra.insert(
                            m.name.clone(),
                            TetraSrc {
                                d_out: m.d_out,
                                d_in: m.d_in,
                                blocks,
                                centroids: [m.centroids[0] as f32, m.centroids[1] as f32],
                                rscale: m.row_scales.iter().map(|&v| v as f32).collect(),
                                tail: m.tail.iter().map(|&v| v as f32).collect(),
                            },
                        );
                    }
                }
            }
            println!(
                "  {path} — {} matrices: {} Tetra, {tetra_int4} int4 g128 (counted, not timed)",
                h.matrices,
                tetra.len()
            );
        }
        if union.has(arms::TETRA48) && tetra.is_empty() {
            return Err("tetra48 is selected and no Tetra file was given. Pass the ball file \
                        as the first argument and the Tetra file as the SECOND — the two \
                        formats do not share a file. Or drop tetra48 from LLVQ_BENCH_ARMS."
                .to_string());
        }

        let build = |s: &Src| -> Result<Mat, String> {
            let (d_out, d_in) = (s.d_out, s.d_in);
            let nblocks = d_in / DIM;
            let tail_w = d_in % DIM;
            assert_eq!(d_out % 8, 0, "{}: CUDA launches whole blocks", s.name);
            assert!(
                d_in <= x.len(),
                "{}: d_in {d_in} overruns the activation of {} floats. Raise XLEN above \
                 and relaunch: extending the draw leaves every shorter matrix bit-identical",
                s.name,
                x.len()
            );
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
            // ---- the AWQ arm's three buffers ----
            //
            // 🚨 The weight buffer carries an ALLOCATION TAIL, and forgetting
            // it is an `illegal memory access` after twenty minutes of
            // transcoding. The `float4` load of the weights is UNCONDITIONAL
            // in their kernel (`awq_gemv.cu`), whereas the guard
            // `inputs_ptr_delta + ic_0 < IC/8` protects only the input and the
            // accumulation. The highest u32 index reached is therefore
            // `(OC-1)*weight_w + NPG*128 - 1`, past `OC*weight_w` when `IC/8`
            // is not a multiple of `NPG*128`. The overruns of the inner rows
            // fall back into the next row, only the last one goes out, hence a
            // single tail and not a widened stride.
            let (aww, awz, aws) = awq_strides(d_in);
            // ⚠️ The tail is NOT allocated here but at upload. If it were,
            // `chunks_mut(chunk * aww)` could return one slice more than
            // `y_ref.chunks_mut(chunk)`, and `zip` stops on the shorter one,
            // so whole rows would silently go unquantized and the arm would
            // turn green on a buffer full of holes.
            let awq_tail = (awz * 128).saturating_sub(aww);
            let awq_on = union.has(arms::AWQ);
            let mut awq_w = awq_on.then(|| vec![0u32; d_out * aww]);
            let mut awq_z = awq_on.then(|| vec![0u32; d_out * awz]);
            let mut awq_s = awq_on.then(|| vec![0u16; d_out * aws]);
            let mut y_awq_ref = awq_on.then(|| vec![0.0f64; d_out]);
            let mut awq_scale = awq_on.then(|| vec![0.0f64; d_out]);
            // The activation as the AWQ kernel, and cublasf16, sees it:
            // binary16. The reference must be computed against this one, not
            // against the f32, otherwise the arm is billed for a deviation
            // that is not its own.
            let cublas_on = union.has(arms::CUBLASF16);
            // Same rule, and the same 2026-08-20 oversight: the QTIP arm
            // forms its reference against the activation ROUNDED to binary16,
            // because that is the one its kernel reads. Without this input
            // there is no reference to form at all.
            let qtip_on = union.has(arms::QTIP);
            let xh_f64: Option<Vec<f64>> = (awq_on || cublas_on || qtip_on)
                .then(|| x[..d_in].iter().map(|&v| f16_to_f64(f16_bits(v))).collect());
            // The cublasf16 arm's reference: f16 weights × binary16 input,
            // accumulated in f64. Neither `y_ref` (exact weights) nor
            // `y16_ref` (f16 weights but f32 input) describes what GemmEx
            // computes.
            let mut y16h_ref = cublas_on.then(|| vec![0.0f64; d_out]);
            let nthreads = std::thread::available_parallelism().map_or(8, |n| n.get());
            let chunk = d_out.div_ceil(nthreads);
            let n_chunks = d_out.div_ceil(chunk);
            // The per-thread slices of the optional buffers. ⚠️ The trap this
            // shape avoids: `chunks_mut` on an EMPTY Vec returns zero slices,
            // and the driver's `zip` would stop at the shortest one, so the
            // whole reference loop would be skipped in silence. Hence
            // `Vec<Option<&mut [T]>>` of length n_chunks EXACTLY, in every
            // case.
            fn opt_chunks<T>(
                v: &mut Option<Vec<T>>,
                chunk: usize,
                n: usize,
            ) -> Vec<Option<&mut [T]>> {
                match v.as_mut() {
                    Some(v) => {
                        let out: Vec<Option<&mut [T]>> =
                            v.chunks_mut(chunk).map(Some).collect();
                        assert_eq!(out.len(), n, "optional slices misaligned");
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
            let y16hc_v = opt_chunks(&mut y16h_ref, chunk, n_chunks);
            std::thread::scope(|sc| {
                for (ci, (((((((((w16c, yc), y16c), scc), awc), azc), asc), yawc), asqc), y16hc)) in
                    w16.chunks_mut(chunk * d_in)
                        .zip(y_ref.chunks_mut(chunk))
                        .zip(y16_ref.chunks_mut(chunk))
                        .zip(scale.chunks_mut(chunk))
                        .zip(awc_v)
                        .zip(azc_v)
                        .zip(asc_v)
                        .zip(yawc_v)
                        .zip(asqc_v)
                        .zip(y16hc_v)
                        .enumerate()
                {
                    let (rt, table, x, src, planes) = (&rt, &table, &x, &s, &planes_data);
                    let p12 = &p12;
                    let (g70, g70cls, golay) = (&g70, &g70cls, &golay);
                    let xh = &xh_f64;
                    sc.spawn(move || {
                        let (mut awc, mut azc, mut asc, mut yawc, mut asqc, mut y16hc) =
                            (awc, azc, asc, yawc, asqc, y16hc);
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
                                        "{}: block {}, Planes14 is not a bijection of \
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
                                        "{}: block {}, the Planes12x overlay does not \
                                         reconstruct the exact block",
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
                                        "{}: block {}, Golay70 does not reconstruct the exact \
                                         block",
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
                            let (mut a, mut b, mut bh, mut ss) = (0.0, 0.0, 0.0, 0.0);
                            for c in 0..d_in {
                                let xv = x[c] as f64;
                                let wv = wrow[c];
                                let hb = f16_bits(wv as f32);
                                w16c[lr * d_in + c] = hb;
                                a += wv * xv;
                                b += f16_to_f64(hb) * xv;
                                // binary16 input: the sum GemmEx sees.
                                if let Some(xhv) = xh.as_deref() {
                                    bh += f16_to_f64(hb) * xhv[c];
                                }
                                ss += (wv * xv).abs();
                            }
                            yc[lr] = a;
                            y16c[lr] = b;
                            if let Some(y16hc) = y16hc.as_deref_mut() {
                                y16hc[lr] = bh;
                            }
                            scc[lr] = ss;
                            // The competitor arm, on the SAME weights: w4g128
                            // applied to what LLVQ reconstructed. This is not
                            // Qwen's AWQ checkpoint, it is a w4g128 faithful
                            // in TIME (the kernel has no data-dependent
                            // branch) and exact in REFERENCE, and it will
                            // carry no quality claim.
                            if let (Some(awc), Some(azc), Some(asc), Some(yawc), Some(asqc)) = (
                                awc.as_deref_mut(),
                                azc.as_deref_mut(),
                                asc.as_deref_mut(),
                                yawc.as_deref_mut(),
                                asqc.as_deref_mut(),
                            ) {
                                let xh = xh.as_ref().expect("xh_f64 built with the arm");
                                yawc[lr] = awq_quant_row(
                                    &wrow,
                                    xh,
                                    d_in,
                                    &mut awc[lr * aww..(lr + 1) * aww],
                                    &mut azc[lr * awz..(lr + 1) * awz],
                                    &mut asc[lr * aws..(lr + 1) * aws],
                                );
                                // Its error denominator, on its own weights.
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

            // ---- the stage: Dev for phase 1, Host for the ones after ----
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

            // P1c. The servable stream: `transcode_e1v_rows` through
            // `e1v_host.rs`, cut on `d_in / 24` blocks per row so that a group
            // never straddles a row, failing which no warp would read a single
            // group (X3). No search, no exception, a direct transcode.
            let e1v_arm = match union.has(arms::E1V) {
                true => {
                    let em = e1v_mat(&fd, &golay, &s.indices, &s.gains, d_in)?;
                    let in0 = phase0.has(arms::E1V);
                    Some(E1vArm {
                        row_blocks: em.row_blocks,
                        // The arm's bill: the stream and its bases table, plus
                        // the f32 tail and the row scales that EVERY LLVQ arm
                        // uploads, the `slot32` accounting twenty lines above,
                        // identical.
                        bytes: em.bytes + (d_out * tail_w) as u64 * 4 + d_out as u64 * 4,
                        data: up_or_hold_u32(em.data, in0)?,
                        bases: up_or_hold_u32(em.bases, in0)?,
                    })
                }
                false => None,
            };

            let awq = match (awq_w, awq_z, awq_s, y_awq_ref, awq_scale) {
                (Some(awq_w), Some(awq_z), Some(awq_s), Some(y_awq), Some(a_scale)) => {
                    let in0 = phase0.has(arms::AWQ);
                    Some(AwqArm {
                        w: {
                            // The allocation tail, added HERE and nowhere
                            // else: the kernel reads up to `NPG*128` words
                            // past the start of the last row, and those words
                            // must exist. They are NOT billed, being an
                            // addressing margin and not a byte the algorithm
                            // carries, and billing them would inflate the
                            // competitor arm by a weight it does not bear.
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

            // ---- the QTIP arm ----
            //
            // Built only when selected, like every other arm. Two things about
            // it are unlike the others and are stated here rather than left to
            // be inferred:
            //
            //  * the payload is pseudo-random, so the weights it decodes are
            //    nobody's weights. See `QtipArm`. The seed is derived from the
            //    shape so a matrix gets the same stream in every phase and in
            //    every process — a control run must reproduce the same object,
            //    and a clock-seeded buffer would silently break that.
            //  * the reference is exact, so this arm is held to OUR threshold
            //    (`TOL`), not to the looser `AWQ_TOL`. That looser bound exists
            //    because AWQ and cuBLAS write binary16 outputs; QTIP writes
            //    f32, so there is nothing to forgive.
            let qtip = match union.has(arms::QTIP) {
                true => {
                    use llvq_cuda::qtip_host as qh;
                    let kernel = qh::kernel_name(d_out, d_in).ok_or_else(|| {
                        format!(
                            "{}: QTIP has no shim for {d_out}x{d_in}, add the \
                             shape to kernels/qtip_glue.cu and to \
                             qtip_host::QTIP_SHAPES rather than guess a name",
                            s.name
                        )
                    })?;
                    // 🚨 Derived from the matrix NAME, not from its shape
                    // alone. One seed per shape gave 5 distinct payloads
                    // replicated 36 times: the 252 matrices would have
                    // produced 21,504 genuinely distinct rows instead of the
                    // 1,105,920 the verification claims to cover, and 36
                    // replicas of one content test it once. The name stays
                    // deterministic from run to run, so a control run
                    // reproduces the same object.
                    let mut seed = 0xF2_0000_0000u64 ^ ((d_out as u64) << 20) ^ d_in as u64;
                    for b in s.name.as_bytes() {
                        seed = seed.rotate_left(7) ^ u64::from(*b);
                    }
                    let buf = qh::pseudo_random_buffer(d_out, d_in, seed);
                    let tlut = qh::pseudo_random_tlut(seed ^ 0x5EED);
                    let xf = xh_f64.as_ref().expect("binary16 activation in f64 not built");
                    let mut y = vec![0.0f64; d_out];
                    let mut sc = vec![0.0f64; d_out];
                    for row in 0..d_out {
                        y[row] = qh::reference_row(&buf, d_in, row, xf, &tlut);
                        sc[row] = (0..d_in)
                            .map(|c| (qh::weight_at(&buf, d_in, row, c, &tlut) as f64 * xf[c]).abs())
                            .sum();
                    }
                    // 512 half2, flat as 1024 binary16 words, in the order the
                    // kernel indexes them: entry i is (x, y) at 2i and 2i+1.
                    let mut cb = vec![0u16; qh::QTIP_TLUT_LEN * 2];
                    for (i, &(a, b)) in tlut.iter().enumerate() {
                        cb[2 * i] = f16_bits(a);
                        cb[2 * i + 1] = f16_bits(b);
                    }
                    let in0 = phase0.has(arms::QTIP);
                    Some(QtipArm {
                        compressed: up_or_hold_u32(qtip_pack_u32(&buf), in0)?,
                        codebook: up_or_hold_u16(cb, in0)?,
                        bytes: qh::qtip_bytes(d_out, d_in),
                        y_ref: y,
                        scale: sc,
                        kernel,
                    })
                }
                false => None,
            };

            // ---- the served Tetra arm, from the SECOND file --------------
            //
            // Matched by NAME, not by position: the two files hold the same
            // 252 projections of the same model, but 36 of them are int4 in
            // the served one and have no lattice arm at all. A positional
            // match would pair `v_proj` of one file with `o_proj` of the other
            // and decode a stream against another projection's scales — which
            // produces numbers, not an error.
            let tetra_arm = match tetra.get(&s.name) {
                None => None,
                Some(t) => {
                    assert_eq!(
                        (t.d_out, t.d_in),
                        (d_out, d_in),
                        "{}: the two files disagree on the shape",
                        s.name
                    );
                    // The stream's own stride, checked against what the layout
                    // derives rather than trusted — the E1v argument
                    // (`:2185-2189`): a stride recomputed at launch is a second
                    // formula that can drift from the one that wrote the bytes.
                    assert_eq!(
                        t.blocks.stride_u32,
                        llvq_artifact::tetra48::stride_u32(nblocks),
                        "{}: the stream's row stride is not the one the layout derives",
                        s.name
                    );
                    let mut ty = vec![0.0f64; d_out];
                    let mut tsc = vec![0.0f64; d_out];
                    // Built once and shared: `Tetra::new()` builds the trio's
                    // tables, and one per row-chunk would dominate the build.
                    let td = Tetra::new();
                    std::thread::scope(|sc| {
                        for (ci, (yc, scc)) in
                            ty.chunks_mut(chunk).zip(tsc.chunks_mut(chunk)).enumerate()
                        {
                            let (blocks, t, x, td) = (&t.blocks, t, &x, &td);
                            let name = s.name.as_str();
                            sc.spawn(move || {
                                let mut wrow = vec![0.0f64; d_in];
                                for lr in 0..yc.len() {
                                    let row = ci * chunk + lr;
                                    wrow.fill(0.0);
                                    for b in 0..nblocks {
                                        // Decoded back OUT of the packed bytes,
                                        // never from the indices that produced
                                        // them — those were dropped at read.
                                        // A packing mistake is then a
                                        // disagreement between two routes.
                                        let (pt, gain) = blocks.decode_block(td, row, b);
                                        let n2: i64 =
                                            pt.iter().map(|&v| (v as i64) * (v as i64)).sum();
                                        if n2 == 0 {
                                            continue; // the origin, and its row is already zero
                                        }
                                        // `n2 = 16 m` on Λ₂₄, always. If it is
                                        // not, the decode is wrong and every
                                        // number after it is a fiction — the
                                        // kernel indexes `invnorm[n2 >> 4]`
                                        // and would silently take a neighbour.
                                        assert_eq!(
                                            n2 % 16,
                                            0,
                                            "{}: block {b} of row {row} has |y|² = {n2}, not a \
                                             multiple of 16",
                                            name
                                        );
                                        let k = t.centroids[gain as usize] as f64
                                            * t.rscale[row] as f64
                                            / (n2 as f64).sqrt();
                                        for (i, &v) in pt.iter().enumerate() {
                                            wrow[b * DIM + i] = v as f64 * k;
                                        }
                                    }
                                    for w in 0..tail_w {
                                        wrow[nblocks * DIM + w] = t.tail[row * tail_w + w] as f64;
                                    }
                                    let (mut a, mut ss) = (0.0f64, 0.0f64);
                                    for c in 0..d_in {
                                        let v = wrow[c] * x[c] as f64;
                                        a += v;
                                        ss += v.abs();
                                    }
                                    yc[lr] = a;
                                    scc[lr] = ss;
                                }
                            });
                        }
                    });
                    let words: Vec<u32> = t
                        .blocks
                        .data
                        .chunks_exact(4)
                        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect();
                    assert_eq!(
                        words.len(),
                        d_out * t.blocks.stride_u32,
                        "{}: the stream is not a whole number of row strides",
                        s.name
                    );
                    Some(TetraArm {
                        words: up_or_hold_u32(words, phase0.has(arms::TETRA48))?,
                        stride_u32: t.blocks.stride_u32 as u32,
                        gscale: cuda.up_f32(&t.centroids)?,
                        rscale: cuda.up_f32(&t.rscale)?,
                        tail: cuda.up_f32(if t.tail.is_empty() { &[0.0f32] } else { &t.tail })?,
                        // The stream PLUS the tail and the row scales, like
                        // every other arm here (`SlotArm` :2113, `PlanesArm`
                        // :2122, `P12Arm` :2164): this column is what the
                        // kernel READS for one matrix, not what the format
                        // spends on codes. It was the stream alone, which
                        // would have understated Tetra's b/weight against
                        // every arm it is printed beside.
                        //
                        // And the PADDED length, where Planes14 counts its
                        // unpadded stream — not an inconsistency, a difference
                        // between the two layouts. Planes14 is flat at a
                        // uniform 14-byte stride and its 4-byte pad is a read
                        // window past the end; Tetra pads EVERY ROW to a u32
                        // boundary and the kernel reads over that pad on every
                        // row. `Tetra48Blocks::bits_per_weight` says the same,
                        // and names the accounting error of 2026-07-31 that
                        // omitting it would repeat.
                        // The UNPADDED stream, which is this repository's
                        // stated convention and not a choice made here: the
                        // Planes14 arm ignores its 4-byte pad — "billing ours
                        // here would mix two byte accountings, the exact
                        // mistake the K-1 lot eliminated" — and
                        // `e1v_host.rs:35` says it outright, "the landing pad
                        // is not counted".
                        //
                        // Tetra's row pad is that same species. `stride_u32`
                        // rounds each row to EIGHT bytes and its own doc says
                        // why: "what covers the last block's two-word read
                        // window". On the 4B's shapes `6·nblocks` is already a
                        // multiple of 4 at nblocks 106 and 170, so none of
                        // those bytes is alignment — they are landing window.
                        //
                        // The decisive check is the column itself: Planes14's
                        // row reproduces ETAT's 4.8040. A column where five
                        // rows reproduce the record and the sixth does not has
                        // one wrong row, and billing the pad would print 2.159
                        // for a file the dossier publishes at 2.1498.
                        bytes: (llvq_artifact::tetra48::TETRA48_BYTES * d_out * nblocks) as u64
                            + (d_out * tail_w) as u64 * 4
                            + d_out as u64 * 4,
                        pad_bytes: t.blocks.data.len() as u64
                            - (llvq_artifact::tetra48::TETRA48_BYTES * d_out * nblocks) as u64,
                        weights: (d_out * d_in) as u64,
                        y_ref: ty,
                        scale: tsc,
                    })
                }
            };

            Ok(Mat {
                name: s.name.clone(),
                d_out,
                d_in,
                nblocks,
                tail_w,
                slot,
                planes,
                e1v: e1v_arm,
                tetra: tetra_arm,
                p12,
                g70,
                awq,
                qtip,
                gscale: cuda.up_f32(&s.centroids)?,
                rscale: cuda.up_f32(&s.rscale)?,
                tail: cuda.up_f32(if s.tail.is_empty() { &[0.0f32] } else { &s.tail })?,
                w16: cuda.up_u16(&w16)?,
                f16_bytes: (d_out * d_in * 2) as u64,
                y_ref,
                y16_ref,
                y16h_ref,
                scale,
            })
        };

        println!(
            "\nBuild: Slot32 transcode (reference, always){}{}{}{}{}, exact-reconstruction \
             proofs of the arms that were built…",
            if union.has(arms::TETRA48) {
                ", Tetra48 (SECOND FILE — its own reference, decoded back out of the \
                 packed words)"
            } else {
                ""
            },
            if union.has(arms::PLANES14) { ", Planes14" } else { "" },
            if union.has(arms::PLANES12X) {
                ", Planes12x (L = 5 → L ≤ 4 swap included)"
            } else {
                ""
            },
            if g70_needed { ", Golay70 (origin holes + E2 exceptions)" } else { "" },
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
                // A v5 Tetra file has no runtime layout yet: refused here by name, never read as a Ball.
                llvq_artifact::runtime::require_ball_kinds(h.kinds(), "planesbench").map_err(|e| e.to_string())?;
                println!("  {path} — {} matrices", h.matrices);
                // The provenance line every published table is read through.
                // With two files it has to name BOTH, and their record counts:
                // a reader who sees one path will read the whole table as one
                // file's, and the Tetra row is not.
                source = match &tetra_path {
                    None => format!("the published model ({path})"),
                    Some(t) => format!(
                        "TWO FILES — ball: {path} ({} records); tetra: {t} ({} Tetra + \
                         {tetra_int4} int4 g128). The arms do not all read the same bytes",
                        h.matrices,
                        tetra.len()
                    ),
                };
                for _ in 0..h.matrices {
                    let m = llvq_artifact::read_matrix_raw(&mut r, h.version).map_err(|e| e.to_string())?;
                    // The record's own kind, not only the header's declared set.
                    llvq_artifact::runtime::require_ball(m.kind, "planesbench")
                        .map_err(|e| e.to_string())?;
                    // Every decoder hard-codes one gain bit (`hdr >> 9`).
                    assert_eq!(
                        m.centroids.len(),
                        2,
                        "{}: the kernels hard-code 1 gain bit",
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
                // 🚨 Nothing checked that the two files' names agreed, and a
                // total mismatch is the quietest failure this bench can have:
                // the Tetra arm covers no matrix, its bytes and its weights
                // are both zero, `bpw` divides by `max(1)` and prints
                // **0.000 b/weight** — under the 2.20 gate, which then passes
                // on nothing — while `verify_arm` reports its best possible
                // error and the timed pass launches nothing at all.
                //
                // So the match is a fact of the run, printed, and a Tetra
                // record with no home is refused. On the pure Tetra file that
                // is 252 of 252; on the served mixed one, 216 of 252, the 36
                // `v_proj` being int4 and having no lattice arm.
                if !tetra.is_empty() {
                    let matched = mats.iter().filter(|m| m.tetra.is_some()).count();
                    println!(
                        "  tetra48: {matched} of {} matrices matched by name, from {} Tetra \
                         records",
                        mats.len(),
                        tetra.len()
                    );
                    if matched != tetra.len() {
                        return Err(format!(
                            "the two files do not name the same projections: {} Tetra records, \
                             {matched} of them found a matrix in the ball file. A record with no \
                             home would drop out of the timed pass in silence, and the b/weight \
                             column would divide by weights nobody read.",
                            tetra.len()
                        ));
                    }
                }
            }
            // ---- synthetic, real shapes, repeated past the L2 ----
            None => {
                source = format!("{reps} synthetic repetitions of the 7 shapes");
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
            /// The Tetra stream of the same group, present only when
            /// `tetra48` is in the selection. `Option` and not a sentinel: a
            /// group whose members are not all Tetra has no such stream, and
            /// an empty `CudaSlice` would launch and read garbage.
            ///
            /// It carries its own scales, like `TetraArm`. The `gscale`,
            /// `rscale` and `tail` above belong to the FIRST file; until
            /// 2026-11-04 the Tetra launch read them, on a stream transcoded
            /// from the first file's indices, and was compared against the
            /// second file's separate launches.
            tetra: Option<FusedTetra>,
            tstride: u32,
            tetra_bytes: u64,
            /// Indices into `mats` of the matrices this one replaces, in row
            /// order — the fused output must equal their outputs concatenated.
            parts: Vec<usize>,
        }

        /// The Tetra half of a fused group: the SECOND file's words, centroid
        /// pairs, `gs_off`, row scales and tail, concatenated by
        /// `seg_concat_tetra48` from the same members in the same order.
        struct FusedTetra {
            words: cudarc::driver::CudaSlice<u32>,
            gscale: cudarc::driver::CudaSlice<f32>,
            gs_off: cudarc::driver::CudaSlice<u32>,
            rscale: cudarc::driver::CudaSlice<f32>,
            tail: cudarc::driver::CudaSlice<f32>,
        }

        // 🕳️ **The fusion arm (A4) is built AFTER the five-arm table, and no
        // longer here.** It used to be here, and it moved the object the table
        // measures twice over, with no log line saying so:
        //
        //   * `max_dout` chained the fused `d_out`, so `d_y` went from 9,728
        //     to 19,456. But `Planes12x` and `Golay70` do their
        //     `memset_zeros(y)` over the **whole** slice and **inside the
        //     timer** (a claimed choice, see `launch_planes12x`). The two
        //     correction arms therefore paid ~9.8 MB more per pass, ~+0.4%,
        //     than in the run that published their numbers: a systematic and
        //     one-directional bias, on exactly the two arms we want to know
        //     whether they drifted.
        //   * its ~2.9 GB of fused streams stayed resident while the five arms
        //     were being timed, taking occupancy from ~15.3 to ~18.4 GB. VRAM
        //     pressure is the named suspect for the ×20 spread observed going
        //     from four arms to five.
        //
        // Neither is a bug: they are side effects of a lot added afterwards
        // (A4, commit 2d56cce, 2026-08-09). But they made the five-arm table
        // **incomparable with the run that published it**, and that is the one
        // thing this table has to guarantee.
        let max_dout = mats.iter().map(|m| m.d_out).max().unwrap();
        let mut d_y = cuda.zeros_f32(max_dout)?;
        // The AWQ arm writes a binary16 output, which is what their kernel
        // does. A separate buffer rather than a reinterpretation of `d_y`:
        // two types, two buffers, and nothing that can overlap, and no buffer
        // at all when the arm is not selected.
        let mut d_yh = match union.has(arms::AWQ) || union.has(arms::CUBLASF16) {
            true => Some(cuda.zeros_u16(max_dout)?),
            false => None,
        };
        // Printed because it was not: that silence is what let `d_y` double
        // with nobody able to read it in a log.
        println!("  d_y: {max_dout} f32, the table and nothing else (section A4 has its own)");
        println!(
            "  {} matrices, {:.2} G weights, in {:.0} s, block-by-block proofs of the \
             arms that were built",
            mats.len(),
            n_weights as f64 / 1e9,
            t0.elapsed().as_secs_f64()
        );

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
        let verify_arm = |a: usize,
                          mats: &[Mat],
                          d_y: &mut cudarc::driver::CudaSlice<f32>,
                          d_yh: &mut Option<cudarc::driver::CudaSlice<u16>>|
         -> Result<f64, String> {
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
        };

        // A phase's upload wave: the buffers of the entering arms go up to
        // the card, those of the arms already staged do not move (stage_up_*
        // is idempotent, v1 and v2 share the golay70 group, so the second of
        // the two re-uploads nothing).
        let upload_added = |mats: &mut [Mat], added: ArmSet| -> Result<(), String> {
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
        let report = |mats: &[Mat],
                      pi: usize,
                      phase: ArmSet,
                      times: &[Vec<f64>; arms::N_ARMS]| {
            let bytes_of = |a: usize| -> u64 { mats.iter().map(|m| arm_bytes(m, a)).sum() };
            // The arm's own denominator, never the table's. `n_weights` stays
            // what it always was — the model's weight count, printed once in
            // the header — but it is no longer what a b/weight column divides
            // by, because not every arm reads every matrix any more.
            let weights_of = |a: usize| -> u64 { mats.iter().map(|m| arm_weights(m, a)).sum() };
            let bpw = |a: usize| -> f64 { bytes_of(a) as f64 * 8.0 / weights_of(a).max(1) as f64 };
            // How many matrices an arm's timed pass actually launched. Equal
            // for every ball arm; 216 of 252 for the served Tetra one.
            let covered = |a: usize| -> usize {
                mats.iter().filter(|m| arm_weights(m, a) > 0).count()
            };
            // Rule 8: a number carries its accounting. This column and the
            // `rtbits` kernel rate differ by exactly this, and by nothing else.
            let tetra_pad: u64 = mats.iter().filter_map(|m| m.tetra.as_ref()).map(|t| t.pad_bytes).sum();
            let t_f16 = &times[arms::FP16];
            let phase_tag = if n_phases > 1 {
                format!("  [phase {}/{}: {}]", pi + 1, n_phases, phase.label())
            } else {
                String::new()
            };
            println!(
                "\nONE TOKEN — {} matrices, one stream, {} arms interleaved{}",
                mats.len(),
                phase.len(),
                phase_tag
            );
            println!(
                "  {ROUNDS} rounds, {WARMUP} discarded; the ratios are formed ROUND BY ROUND"
            );
            println!("  {}", "-".repeat(80));
            // "GB/s(min)", not "GB/s": the column divides by the MINIMUM time
            // — the most favourable reading, that of a peak throughput — and
            // the header did not say so (raised by the 2026-08-18 audit).
            // Declared rather than changed: the published journals carry this
            // accounting, and what was missing is the label, not the formula.
            println!(
                "  {:<22}{:>9}{:>9}{:>9}{:>9}{:>9}{:>10}",
                "format", "min ms", "med ms", "max ms", "GB read", "b/weight", "GB/s(min)"
            );
            // 🚨 The tile, ON the table. Its provenance is printed once, some
            // thirty lines and a three-minute transcode earlier, and F1d runs
            // THREE processes at three tiles into one journal: a reader
            // comparing two tables would have to scroll past a transcode to
            // learn which is which, and the numbers only differ by the thing
            // that is off-screen.
            println!("  {}", tile.provenance());
            for a in arms::DISPLAY_ORDER {
                if !phase.has(a) {
                    continue;
                }
                let (lo, md, hi) = spread(times[a].clone());
                let b = bytes_of(a);
                println!(
                    "  {:<22}{:>9.3}{:>9.3}{:>9.3}{:>9.2}{:>9.3}{:>9.0}",
                    arms::DISPLAY_NAMES[a],
                    lo * 1e3,
                    md * 1e3,
                    hi * 1e3,
                    b as f64 / 1e9,
                    bpw(a),
                    b as f64 / lo / 1e9
                );
            }
            println!("  {}", "-".repeat(80));
            if phase.has(arms::TETRA48) && tetra_pad > 0 {
                println!(
                    "  tetra48 also HOLDS {:.1} MB of row landing pad, NOT billed above — the \
                     convention\n  the slot arm's 20-byte pad and the planes arm's 4-byte pad \
                     already follow.\n  Real VRAM, not a rate: `stride_u32` rounds each row to \
                     eight bytes so the last\n  block's six-byte window cannot fault.",
                    tetra_pad as f64 / 1e6,
                );
            }
            for a in arms::DISPLAY_ORDER {
                if a == arms::FP16 || !phase.has(a) {
                    continue;
                }
                let (lo, md, hi) = spread(per_round(t_f16, &times[a]));
                // BOTH competitors carry the warning, and for the same
                // reason: their × against FP16 mechanically rewards whoever
                // reads the least. QTIP, at 2.0000 b/weight, is the extreme
                // case — its byte bound is 8.00× where ours is 3.33× — so a
                // bare ratio would be the most misleading one in the table.
                if a == arms::AWQ || a == arms::QTIP {
                    println!(
                        "  {:<16} vs FP16     : {md:.2}× [{lo:.2}–{hi:.2}]  \
                         (COMPETITOR — reads {:.3} b/weight;\n  the comparable quantity \
                         is the GB/s, not this ratio)",
                        arms::DISPLAY_NAMES[a],
                        bpw(a)
                    );
                } else {
                    println!(
                        "  {:<16} vs FP16     : {md:.2}× [{lo:.2}–{hi:.2}]",
                        arms::DISPLAY_NAMES[a]
                    );
                }
            }
            // The arm-against-arm ratios: `num` against `den`, formed round by
            // round — >1 = `num` faster. They exist only if BOTH arms ran in
            // this phase.
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
                "(>1 = Planes14 faster, same content decoded)",
            );
            pair(
                arms::PLANES12X,
                arms::PLANES14,
                "(>1 = Planes12x faster; memset + correction included, same exact y)",
            );
            pair(
                arms::GOLAY70V1,
                arms::PLANES12X,
                "(>1 = Golay70 v1 faster; memset + correction included, same exact y)",
            );
            pair(
                arms::GOLAY70V2,
                arms::GOLAY70V1,
                "(>1 = v2 faster; SAME buffers, same exact y — the ratio of the \
                 v2 campaign)",
            );
            // F1d's gate reads THIS line, and it is the one omission on the
            // printing path that would have been silent: every other consumer
            // of an arm picks it up from `DISPLAY_ORDER` automatically, and
            // `pair` is a closure called at four fixed sites that iterate
            // nothing. Without the fifth call the bench would build the Tetra
            // buffers, verify the arm against its own f64 reference, dispatch
            // it in every round, print its table row and its vs-FP16 line —
            // and never print the ratio the whole step exists to measure.
            if phase.has(arms::TETRA48) && phase.has(arms::PLANES14) {
                let (nt, np) = (covered(arms::TETRA48), covered(arms::PLANES14));
                pair(
                    arms::TETRA48,
                    arms::PLANES14,
                    match nt == np {
                        true => "(>1 = Tetra faster; DIFFERENT FILES, same 252 matrices — a layout ratio)",
                        false => "(>1 = Tetra faster — see the caveat below)",
                    },
                );
                // 🕳️ Printed unconditionally, and FALSE on the file F1d runs.
                // The prereg chose the PURE Tetra file precisely so the two
                // arms cover the same 252 matrices; a warning saying they do
                // not would travel into the dossier attached to the one line
                // the throughput gate reads, and a valid equal-support ratio
                // would be discounted for a reason that does not exist.
                if nt != np {
                    println!(
                        "    ⚠️ NOT the same support: Tetra launched over {nt} matrices, \
                         Planes14 over {np}.\n       The {} missing are int4 g128 in the Tetra \
                         file and are read by another kernel,\n       so this ratio is formed \
                         over unequal passes. b/weight is per arm and divides\n       by what \
                         that arm read.",
                        np.saturating_sub(nt)
                    );
                }
            }
            println!("\n  source: {source}");
            let light = arms::DISPLAY_ORDER
                .into_iter()
                .filter(|&a| {
                    // `bytes_of` is 0 for an arm whose buffers were not built
                    // — the synthetic path, or a run given no Tetra file — and
                    // 0 wins a `min_by_key` outright. The paragraph below then
                    // calls it "the lightest arm" and divides the L2 by it.
                    //
                    // 🕳️ And the FLOOR wins it whenever it is selected: `nullk`
                    // reads no weight by definition, so its bytes are the tail
                    // and the row scales alone — ~72 MB against Planes14's
                    // ~2.1 GB. The paragraph would then report 0.8× the L2 and
                    // condemn a run that measures DRAM perfectly well, on the
                    // arm that was built to read nothing. It is excluded for
                    // the reason FP16 and AWQ are: the sentence is about the
                    // LLVQ arms' traffic.
                    a != arms::FP16 && a != arms::AWQ && a != arms::NULLK
                        && phase.has(a) && bytes_of(a) > 0
                })
                .min_by_key(|&a| bytes_of(a));
            if let Some(la) = light {
                let light = bytes_of(la);
                println!(
                    "  {:.0} MB distinct per pass on the lightest arm ({}), which is \
                     {:.1}× the L2 read.\n  Under 1× we would be measuring the cache and not \
                     the DRAM — the trap that\n  made every LLVQ measurement before 2026-07-31 \
                     optimistic.",
                    light as f64 / 1e6,
                    arms::DISPLAY_NAMES[la],
                    light as f64 / dev.l2_bytes as f64
                );
            }
            if std::env::args().nth(1).is_none() {
                println!(
                    "  WARNING: SYNTHETIC blocks, drawn uniformly on the ball m ≤ 13. The class\n  \
                     mixture of a real artifact is not exercised, so the group strides of the\n  \
                     Slot32 arm and the byte traffic differ from the published model. This report\n  \
                     measures the KERNELS — pass the .llvq path as an argument for the model."
                );
            }
            // The tied lm_head, read once per token by every arm at the FP16
            // arm's measured rate — the constant that caps every ratio.
            let head_bytes = 389_070_848f64 * 2.0;
            let fb = bytes_of(arms::FP16) as f64;
            let (f_lo, _, _) = spread(t_f16.clone());
            let head_s = head_bytes / (fb / f_lo);
            let with_head: Vec<String> = arms::DISPLAY_ORDER
                .into_iter()
                .filter(|&a| a != arms::FP16 && a != arms::AWQ && phase.has(a))
                .map(|a| {
                    let (lo, _, _) = spread(times[a].clone());
                    format!("{} {:.2}×", arms::DISPLAY_NAMES[a], (f_lo + head_s) / (lo + head_s))
                })
                .collect();
            if !with_head.is_empty() {
                println!(
                    "\n  With the unquantized f16 lm_head ({:.0} M weights, {:.2} ms at the \
                     measured FP16\n  throughput, added to the LLVQ arms): {}.\n  Norms, \
                     activations, attention and rotation are measured neither here nor there.",
                    389_070_848f64 / 1e6,
                    head_s * 1e3,
                    with_head.join(", ")
                );
            }
            // 🚨 Hard rule 4 wants the same-head ratio BESIDE the raw one, and
            // hard rule 7 wants it formed round by round, never as a quotient
            // of two minima. The paragraph above gives neither for a PAIR: its
            // entries are per-arm × against FP16, and each is built from
            // `spread(...).0`, a minimum. For F1d's headline — the one line a
            // kill is read off — the two rules are met here explicitly.
            if phase.has(arms::TETRA48) && phase.has(arms::PLANES14) {
                // 🚨 TWO corrections, and this repository already uses the
                // phrase "same head" for the SECOND one. The F1 journals mean
                // *the launch floor removed* by it (`f1rankfloor` prints
                // "(v3g−nullk)/(planes14−nullk) … the SAME-HEAD ratio, launch
                // floor removed (rule 4)"), while this file's own paragraph
                // means *the f16 lm_head added*. They are different
                // quantities and they move in opposite directions: the floor
                // is common to both arms and pushes any ratio TOWARD 1, the
                // head is common too and pushes it toward 1 as well, but from
                // a different baseline. Printing one of them under the other's
                // name is how a dossier ends up comparing two things.
                //
                // So both are printed, each named for what it does, and both
                // formed round by round — never a quotient of two medians,
                // which is what a reader would otherwise compute off the table
                // above and which hard rule 7 forbids.
                let add = |v: &[f64], d: f64| -> Vec<f64> { v.iter().map(|t| t + d).collect() };
                let (lo, md, hi) = spread(per_round(
                    &add(&times[arms::PLANES14], head_s),
                    &add(&times[arms::TETRA48], head_s),
                ));
                println!(
                    "  tetra48 vs planes14, WITH THE F16 HEAD : {md:.2}× [{lo:.2}–{hi:.2}]  \
                     (the lm_head added to BOTH arms, round by round)"
                );
                if phase.has(arms::NULLK) {
                    // The F1 dossier's same-head ratio, the one `f1rankfloor`
                    // prints: the launch floor removed from BOTH arms, round
                    // by round, so this run's number and that bench's are the
                    // same function of the same measurement.
                    let sub = |v: &[f64]| -> Vec<f64> {
                        v.iter()
                            .zip(&times[arms::NULLK])
                            .map(|(t, f)| (t - f).max(f64::MIN_POSITIVE))
                            .collect()
                    };
                    let (lo, md, hi) = spread(per_round(
                        &sub(&times[arms::PLANES14]),
                        &sub(&times[arms::TETRA48]),
                    ));
                    println!(
                        "  tetra48 vs planes14, FLOOR REMOVED : {md:.2}× [{lo:.2}–{hi:.2}]  \
                         (nullk subtracted from BOTH, round by round — the F1 journals' \
                         \"same head\")"
                    );
                }
            }
            println!(
                "\n  WARNING: NEVER compare this line by line with the Metal figure, and never\n  \
                 subtract it from a bin/matvec run: other rounds, other NVRTC translation unit.\n  \
                 Every arm against its own f64 reference; the ratios form round by round,\n  \
                 inside one process."
            );
        };

        // ---- the phases: upload, verification, rounds, report ----
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
                    upload_added(&mut mats, added)?;
                }
            }
            let mut worsts: Vec<String> = Vec::new();
            for a in phase.iter() {
                if !verified[a] {
                    if worsts.is_empty() {
                        println!("\nVerifying every row against the f64 reference…");
                    }
                    let w = verify_arm(a, &mats, &mut d_y, &mut d_yh)?;
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
            report(&mats, pi, phase, &times);
            phase_times.push(times);
        }

        // ---- Δ_contrôle: the drift of the common arms between phases ----
        //
        // The number the §4 rule of the 2026-08-10 preregistration asks the job
        // to report, printed by the bench itself so that no ratio is ever taken
        // between two processes: R = max(Δ_contrôle, intra-run half-range of
        // the most dispersed arm); |Δ| > 2R separates, |Δ| < R is
        // indistinguishable at this resolution, and between the two: not
        // resolved, published as such.
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

        // ---- A4: the q+k+v and gate+up fusion, correctness first, cost after ----
        //
        // Built HERE, after the five-arm table: neither its ~2.9 GB nor its
        // doubled `d_out` may exist while the table is being measured (see the
        // long comment above `max_dout`).
        let mut fused: Vec<FusedMat> = Vec::new();
        // The section requires BOTH arms it fuses: on a selection without
        // slot32 or without planes14 it has neither a stream to concatenate nor
        // a separate arm to compare against — it skips, and says so.
        let a4_on = union.has(arms::SLOT32) && union.has(arms::PLANES14);
        if !srcs.is_empty() && !a4_on {
            println!(
                "\n  section A4 (fusion) SKIPPED: it requires slot32 AND planes14 in the \
                 LLVQ_BENCH_ARMS selection"
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
                    "{key}: d_in {} overruns the activation",
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
                    + seg.d_out as u64 * 4
                    // `gs_off`, as for the planes arm below.
                    + seg.d_out as u64 * 4;
                // 🕳️ `gs_off` was missing. The FUSED kernel reads one extra
                // u32 a row naming that row's centroid pair
                // (`kernels/planes_seg.cu`) and the unfused one does not — it
                // is the only byte fusion ADDS. Without it the ledger below
                // printed `+0.00%` and the sentence beside it called that a
                // verification.
                let planes_bytes = pdata.len() as u64
                    + (seg.d_out * seg.tail_w) as u64 * 4
                    + seg.d_out as u64 * 4
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

                // The Tetra stream of the same group, built from the SECOND
                // file. `seg` above is the FIRST file's concatenation: other
                // indices, other centroids, other row scales, other tail. The
                // separate Tetra launches read `tetra[name]`, so the fused one
                // must read the same members, concatenated in the same order
                // with the same `gs_off` rule. Until 2026-11-04 this block
                // transcoded `seg.indices` and launched on `seg`'s scales: the
                // check below then compared two different matrices, and failed
                // at row 0 on 2026-09-20.
                //
                // It costs no transcoding step of its own: each member is
                // already a row-strided stream, and rows sharing `d_in` share
                // `row_stride_u32`, so concatenating by rows is concatenating
                // the word arrays.
                //
                // Only built when the arm is selected. A group is skipped when
                // any member is missing from `tetra`, which is what the served
                // object's int4 `v_proj` does to a q+k+v group: the pair that
                // survives is q+k, and gate+up.
                let (ftetra, tstride, tetra_bytes) = match seg_tetra
                    && union.has(arms::TETRA48)
                    && idx.iter().all(|&i| tetra.contains_key(&srcs[i].name))
                {
                    false => (None, 0u32, 0u64),
                    true => {
                        let tp: Vec<SegTetraPart<'_>> = idx
                            .iter()
                            .map(|&i| {
                                let t = &tetra[&srcs[i].name];
                                SegTetraPart {
                                    d_out: t.d_out,
                                    d_in: t.d_in,
                                    blocks: &t.blocks,
                                    centroids: t.centroids,
                                    rscale: &t.rscale,
                                    tail: &t.tail,
                                }
                            })
                            .collect();
                        // The fused output is cut by the FIRST file's row
                        // counts (`mats[pi].d_out` below), so `seg` is passed
                        // for its boundaries only, and they are asserted.
                        // Proved on the Mac by
                        // `tests/tetra48_segment_matches_unfused.rs`.
                        let ts = seg_tetra48_group(key, &seg, &tp);
                        // The same ledger the two ball arms keep, with the same
                        // `gs_off` line: it is the only byte fusion ADDS.
                        let bytes = ts.words.len() as u64 * 4
                            + (ts.d_out * ts.tail_w) as u64 * 4
                            + ts.d_out as u64 * 4
                            + ts.d_out as u64 * 4;
                        let ft = FusedTetra {
                            words: cuda.up_u32(&ts.words)?,
                            gscale: cuda.up_f32(&ts.centroids)?,
                            gs_off: cuda.up_u32(&ts.gs_off)?,
                            rscale: cuda.up_f32(&ts.rscale)?,
                            tail: cuda
                                .up_f32(if ts.tail.is_empty() { &[0.0f32] } else { &ts.tail })?,
                        };
                        (Some(ft), ts.stride_u32 as u32, bytes)
                    }
                };

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
                    tetra: ftetra,
                    tstride,
                    tetra_bytes,
                    parts: idx.clone(),
                });
            }
            println!(
                "\n  fusion: {} groups ({} matrices → {}), transcoded in {:.0} s",
                fused.len(),
                fused.iter().map(|f| f.parts.len()).sum::<usize>(),
                fused.len(),
                t_fuse.elapsed().as_secs_f64()
            );
        }

        // An output buffer of this section's own: a fused arm writes up to
        // 19,456 rows, and that is exactly what we refuse to impose on the
        // table's `d_y`. The two sections no longer share anything.
        let a4_dout = fused
            .iter()
            .map(|f| f.d_out)
            .chain(std::iter::once(max_dout))
            .max()
            .expect("the chain carries at least max_dout");
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
            // card and this compiler (921,600 rows, 2026-08-05); the Planes14
            // arm is the same claim about `tv_planes_seg` and is *not*
            // established until this block runs.
            println!("\n  Fusion (A4) — output against the unfused matrices");
            // Counted and printed rather than assumed: on the served object
            // every q+k+v group is skipped, because `v_proj` is int4, and a
            // silent skip would read as a pass.
            let mut tetra_checked = 0usize;
            for fm in &fused {
                // Layout 2 is Tetra, appended for the rule every arrival
                // follows. It is the only one that can be ABSENT: a group whose
                // members are not all Tetra has no fused stream, and the
                // closure says so rather than launching on nothing.
                for layout in 0..3 {
                    match layout {
                        0 => run_slot_seg(fm, &mut d_y)?,
                        1 => run_planes_seg(fm, &mut d_y)?,
                        _ => {
                            if !run_tetra48_seg(fm, &mut d_y)? {
                                continue;
                            }
                            tetra_checked += 1;
                        }
                    }
                    cuda.sync()?;
                    let got = cuda.down_f32(&d_y)?;
                    let mut at = 0usize;
                    for &pi in &fm.parts {
                        let m = &mats[pi];
                        match layout {
                            0 => run_slot(m, &mut d_y)?,
                            1 => run_planes(m, &mut d_y)?,
                            _ => run_tetra48(m, &mut d_y)?,
                        }
                        cuda.sync()?;
                        let want = cuda.down_f32(&d_y)?;
                        if got[at..at + m.d_out] != want[..m.d_out] {
                            let bad =
                                (0..m.d_out).find(|&r| got[at + r] != want[r]).unwrap_or(0);
                            return Err(format!(
                                "{} / {} / {}: row {bad} is {} fused against {} \
                                 separate",
                                fm.name,
                                m.name,
                                match layout {
                                    0 => "Slot32",
                                    1 => "Planes14",
                                    _ => "Tetra48",
                                },
                                got[at + bad],
                                want[bad]
                            ));
                        }
                        at += m.d_out;
                    }
                }
            }
            println!(
                "  bit-exact against the unfused arms: {} groups on Slot32 and Planes14, \
                 {tetra_checked} on Tetra48",
                fused.len()
            );
            if tetra_checked == 0 {
                match seg_tetra {
                    false => println!(
                        "  Tetra fusion not requested (LLVQ_SEG_TETRA unset): \
                         `tv_tetra48_seg` is compiled and not exercised here"
                    ),
                    true => println!(
                        "  ⚠️ LLVQ_SEG_TETRA=1 and NO Tetra group was checked: no fused group \
                         has a Tetra stream, so `tv_tetra48_seg` is compiled and UNPROVEN"
                    ),
                }
            }
            println!(
                "  {} groups, {} rows — identical BIT FOR BIT, on Slot32 AND on Planes14",
                fused.len(),
                fused.iter().map(|f| f.d_out).sum::<usize>()
            );

            // Cost. Four arms, interleaved in every round, same process: LLVQ
            // against LLVQ on each layout, and the two layouts against each
            // other. No FP16 arm, deliberately — fusing the FP16 witness would
            // mean holding a second copy of 7.27 GB of f16 weights, so a
            // fused-LLVQ / unfused-FP16 ratio would credit the format for a
            // geometry change. Every number below is a DELTA, not a ratio.
            // Six series, not four: arms 4 and 5 are Tetra separate and Tetra
            // fused, appended after the historical four so no existing
            // denominator moves. They are timed in the SAME rounds, because a
            // delta of two minima from rounds that never coexisted is the
            // mistake this file documents.
            // ---- the persistent sections (L1, the row loops): built after
            // A4's proof, timed in A4's rounds ----
            // The codec outlives the section, whose f64 references borrow it.
            let codec = pers_flags.tetra_occ.then(llvq_search::tetra::Tetra::new);
            let mut pers_sec = None;
            if pers_flags.any() || pers_flags.dump_cubin.is_some() {
                let mut sec = llvq_cuda::persbench::Section::new(pers_flags.clone(), &cuda, &src.text, &tile.define())?;
                let tt = tetra_tabs.as_ref().expect("tetra48 tables not uploaded");
                let tables = llvq_cuda::persbench::Tables {
                    rows: dptr(&tt.rows, &cuda),
                    pref: dptr(&tt.prefixes, &cuda),
                    br: dptr(&tt.branches, &cuda),
                    suf: dptr(&tt.suffixes, &cuda),
                    invnorm: dptr(&tt.invnorm, &cuda),
                };
                let xp = dptr(&d_x, &cuda);
                // The f64 value of row r of one Tetra matrix, from the second
                // file's own records: a row that misses its owed equality is
                // judged against it.
                let f64_of = |name: &str| {
                    let src = &tetra[name];
                    let x = &x;
                    let codec = codec.as_ref().expect("built when a Tetra section is on");
                    move |r: usize| {
                        let tail: Vec<f64> = (0..src.d_in % DIM).map(|i| src.tail[r * (src.d_in % DIM) + i] as f64).collect();
                        llvq_cuda::pers::tetra_row_f64(
                            src.d_in / DIM,
                            |j| src.blocks.decode_block(codec, r, j),
                            src.centroids,
                            src.rscale[r],
                            &tail,
                            x,
                        )
                    }
                };
                if pers_flags.tetra_occ {
                    let zeros = cuda.up_u32(&vec![0u32; max_dout.max(19456)])?;
                    let mut sites = Vec::new();
                    for fm in &fused {
                        let Some(t) = fm.tetra.as_ref() else { continue };
                        run_tetra48_seg(fm, &mut d_y)?;
                        cuda.sync()?;
                        let reference = cuda.down_f32(&d_y)?[..fm.d_out].to_vec();
                        let parts: Vec<(String, usize)> = fm.parts.iter().map(|&pi| (mats[pi].name.clone(), mats[pi].d_out)).collect();
                        let rows_of: Vec<_> = parts.iter().map(|(n, d)| (Box::new(f64_of(n)) as Box<dyn Fn(usize) -> (f64, f64)>, *d)).collect();
                        sites.push(llvq_cuda::persbench::PersallSite {
                            name: fm.name.clone(),
                            words: dptr(&t.words, &cuda),
                            gscale: dptr(&t.gscale, &cuda),
                            gs_off: dptr(&t.gs_off, &cuda),
                            rscale: dptr(&t.rscale, &cuda),
                            tail: dptr(&t.tail, &cuda),
                            d_out: fm.d_out as u32,
                            nblocks: fm.nblocks as u32,
                            tail_w: fm.tail_w as u32,
                            stride_u32: fm.tstride,
                            reference,
                            f64_row: Box::new(move |mut r: usize| {
                                for (f, d) in &rows_of {
                                    if r < *d {
                                        return f(r);
                                    }
                                    r -= d;
                                }
                                unreachable!("row past the group")
                            }),
                        });
                    }
                    for (i, m) in mats.iter().enumerate() {
                        if fused.iter().any(|f| f.parts.contains(&i) && f.tetra.is_some()) {
                            continue;
                        }
                        let Some(a) = m.tetra.as_ref() else { continue };
                        run_tetra48(m, &mut d_y)?;
                        cuda.sync()?;
                        let reference = cuda.down_f32(&d_y)?[..m.d_out].to_vec();
                        sites.push(llvq_cuda::persbench::PersallSite {
                            name: m.name.clone(),
                            words: dptr(a.words.dev(), &cuda),
                            gscale: dptr(&a.gscale, &cuda),
                            gs_off: dptr(&zeros, &cuda),
                            rscale: dptr(&a.rscale, &cuda),
                            tail: dptr(&a.tail, &cuda),
                            d_out: m.d_out as u32,
                            nblocks: m.nblocks as u32,
                            tail_w: m.tail_w as u32,
                            stride_u32: a.stride_u32,
                            reference,
                            f64_row: Box::new(f64_of(&m.name)),
                        });
                    }
                    sec.persall_setup(sites, tables, xp, shared)?;
                    std::mem::forget(zeros); // read by every timed launch; freed with the process
                }
                if pers_flags.trows {
                    // The 252 matrices of the bare 4B file in served order:
                    // 36 layers, each q, k, v, o, gate, up, down.
                    const PROJS: [&str; 7] = [
                        "self_attn.q_proj",
                        "self_attn.k_proj",
                        "self_attn.v_proj",
                        "self_attn.o_proj",
                        "mlp.gate_proj",
                        "mlp.up_proj",
                        "mlp.down_proj",
                    ];
                    let mut sites = Vec::new();
                    for name in (0..36).flat_map(|l| PROJS.iter().map(move |p| format!("model.layers.{l}.{p}.weight"))) {
                        let m = mats
                            .iter()
                            .find(|m| m.name == name)
                            .ok_or_else(|| format!("LLVQ_TROWS: no matrix {name} in the bench"))?;
                        let a = m.tetra.as_ref().ok_or_else(|| format!("LLVQ_TROWS: {name} has no Tetra stream"))?;
                        let src = &tetra[&name];
                        sites.push(llvq_cuda::persbench::RowSite {
                            name: name.clone(),
                            words: dptr(a.words.dev(), &cuda),
                            gscale: dptr(&a.gscale, &cuda),
                            rscale: dptr(&a.rscale, &cuda),
                            // The file's binary16 tail, widened to f32 by the
                            // reader and narrowed back here: exact both ways.
                            tail_f16: src.tail.iter().map(|&v| llvq_cuda::f16_bits(v)).collect(),
                            d_out: m.d_out as u32,
                            d_in: m.d_in as u32,
                            stride_u32: a.stride_u32,
                        });
                    }
                    sec.trows_setup(&sites, tables, xp, x.len(), &tile.define())?;
                }
                pers_sec = Some((sec, tables, xp));
            }

            let mut tf: [Vec<f64>; 6] = [
                Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(),
            ];
            for rep in 0..ROUNDS {
                for (arm, ta) in tf.iter_mut().enumerate() {
                    // Arms 4 and 5 are the Tetra pair. Without a fused stream
                    // they would measure the same launches twice and print a
                    // delta of zero, which reads as "the fusion is worth
                    // nothing" rather than "the fusion did not run".
                    if arm >= 4 && !fused.iter().any(|f| f.tetra.is_some()) {
                        continue;
                    }
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
                        4 => {
                            for m in &mats {
                                run_tetra48(m, &mut d_y)?;
                            }
                        }
                        5 => {
                            // A group whose members are not all Tetra has no
                            // fused stream; its members run separately here, so
                            // arms 4 and 5 always cover the same matrices and
                            // their delta is the fusion and nothing else.
                            for fm in &fused {
                                if !run_tetra48_seg(fm, &mut d_y)? {
                                    for &pi in &fm.parts {
                                        run_tetra48(&mats[pi], &mut d_y)?;
                                    }
                                }
                            }
                            for (i, m) in mats.iter().enumerate() {
                                if !fused.iter().any(|f| f.parts.contains(&i)) {
                                    run_tetra48(m, &mut d_y)?;
                                }
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
                if let Some((sec, tables, xp)) = pers_sec.as_mut() {
                    sec.time_round(rep >= WARMUP, *tables, *xp)?;
                }
            }
            if let Some((sec, _, _)) = &pers_sec {
                sec.report(&tf[4], &tf[5]);
            }
            let [ts, tss, tp, tps, tt, tts] = tf;
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
            // The Tetra delta, formed round by round like the other two —
            // and ONLY when the arms ran. Skipping arms 4 and 5 leaves their
            // series empty, and `spread` indexes the median: on 2026-09-20
            // that panicked at the very end of a 25-minute job, after the
            // table it exists for had already been printed. A `continue` was
            // added without reading what consumed its result.
            let tetra_timed = !tt.is_empty() && !tts.is_empty();
            let (dt_lo, dt_md, dt_hi, t_sep, t_fus) = match tetra_timed {
                false => (0.0, 0.0, 0.0, 0.0, 0.0),
                true => {
                    let dt: Vec<f64> =
                        tt.iter().zip(&tts).map(|(a, b)| (a - b) * 1e3).collect();
                    let (_, ts_, _) = spread(tt);
                    let (_, tf_, _) = spread(tts);
                    let (lo, md, hi) = spread(dt);
                    (lo, md, hi, ts_, tf_)
                }
            };

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
            // The only byte fusion adds, on either layout: one u32 a fused row.
            let gs_off_bytes: u64 = fused.iter().map(|f| f.gs_off.len() as u64 * 4).sum();

            println!(
                "\n  Cost — {ROUNDS} rounds, {WARMUP} discarded, FOUR arms interleaved, \
                 medians\n  {}",
                "-".repeat(78)
            );
            println!(
                "  {:<34}{:>10.3} ms   {n_sep} launches",
                "Slot32, separate matrices",
                s_sep * 1e3
            );
            println!(
                "  {:<34}{:>10.3} ms   {n_fus} launches",
                "Slot32, q+k+v and gate+up fused",
                s_fus * 1e3
            );
            println!(
                "  {:<34}{:>10.3} ms   {n_sep} launches",
                "Planes14, separate matrices",
                p_sep * 1e3
            );
            println!(
                "  {:<34}{:>10.3} ms   {n_fus} launches",
                "Planes14, q+k+v and gate+up fused",
                p_fus * 1e3
            );
            println!("  {}", "-".repeat(78));
            // Tetra before the two historical gains, because its arm is the
            // one this section was extended for. Printed even when it is zero:
            // a fusion that fused nothing must say so, not disappear.
            let tb_sep = unfused_bytes(|m| arm_bytes(m, arms::TETRA48));
            let tb_fus = fused_bytes(|f| f.tetra_bytes, |m| arm_bytes(m, arms::TETRA48));
            if !tetra_timed {
                println!("  Tetra48 fusion: not timed (LLVQ_SEG_TETRA unset)");
            }
            if tetra_timed {
            println!(
                "  {:<34}{:>10.3} ms\n  {:<34}{:>10.3} ms",
                "Tetra48, separate matrices",
                t_sep * 1e3,
                "Tetra48, q+k and gate+up fused",
                t_fus * 1e3
            );
            println!(
                "  gain Tetra48  : {dt_md:.3} ms [{dt_lo:.3}–{dt_hi:.3}]  ({:.1}%)",
                if t_sep > 0.0 { dt_md / (t_sep * 1e3) * 100.0 } else { 0.0 }
            );
            println!(
                "  bytes read — Tetra48  : {:.3} GB fused against {:.3} separate ({:+.2}%)",
                tb_fus as f64 / 1e9,
                tb_sep as f64 / 1e9,
                if tb_sep > 0 {
                    (tb_fus as f64 / tb_sep as f64 - 1.0) * 100.0
                } else {
                    0.0
                }
            );
            }
            println!(
                "  gain Slot32   : {ds_md:.3} ms [{ds_lo:.3}–{ds_hi:.3}]  ({:.1}%)",
                100.0 * ds_md / (s_sep * 1e3)
            );
            println!(
                "  gain Planes14 : {dp_md:.3} ms [{dp_lo:.3}–{dp_hi:.3}]  ({:.1}%)",
                100.0 * dp_md / (p_sep * 1e3)
            );
            println!(
                "  Planes14 / Slot32 on the gain: {rr_md:.2}× [{rr_lo:.2}–{rr_hi:.2}]\n  \
                 WARNING: ratio of two DIFFERENCES. Its spread is that of the two \
                 numerators\n  combined, hence far wider than that of the ratios in the table \
                 above. Read it\n  as an order of magnitude, never to two decimals. The two \
                 \"gain\" lines are the\n  ones that carry the result."
            );
            println!(
                "  landmark: 108 fewer launches × 3.63 µs measured (a3-graph-2026-08-06)\n  \
                 = 0.392 ms, independent of the layout. What exceeds that is occupancy."
            );
            println!(
                "  bytes read — Slot32   : {:.3} GB fused against {:.3} separate ({:+.2}%)\n  \
                 bytes read — Planes14 : {:.3} GB fused against {:.3} separate ({:+.2}%)",
                sb_fus as f64 / 1e9,
                sb_sep as f64 / 1e9,
                100.0 * (sb_fus as f64 - sb_sep as f64) / sb_sep as f64,
                pb_fus as f64 / 1e9,
                pb_sep as f64 / 1e9,
                100.0 * (pb_fus as f64 - pb_sep as f64) / pb_sep as f64
            );
            println!(
                "  Slot32's STREAM can move: its stride is the widest record of a group of \
                 32,\n  and the concatenation regroups at segment boundaries. Planes14's \
                 cannot — 14\n  bytes per block, no bases table. What BOTH pay is `gs_off`, one \
                 u32 a fused\n  row naming that row's centroid pair: {:.1} MB over the 72 \
                 groups, and it is the\n  only byte fusion adds. It was missing from this ledger \
                 until 2026-09-10, which\n  printed +0.00% and called that a verification \
                 (tests/planes_segment_matches_unfused.rs).\n  Any gain BEYOND this would be a \
                 confounder, not a bonus.",
                (gs_off_bytes as f64) / 1e6
            );
            println!(
                "\n  WARNING: THIS BLOCK PRODUCES NO RATIO AGAINST FP16, and authorizes\n  \
                 none: the FP16 arm suffers the same underfill on k/v and would gain from\n  \
                 the fusion too. Only the two LLVQ → LLVQ DELTAS above are measured."
            );
        }
        Ok(())
    }
}
