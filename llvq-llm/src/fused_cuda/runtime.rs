//! `FusedRuntime`: the compiled module, the shared tables, and the inherent
//! methods every served projection goes through. One file because the struct
//! and its main `impl` are one unit; the additive entry points live beside it.

use super::*;

/// The module, the shared tables, and the device everything lives on.
pub struct FusedRuntime {
    pub(super) cuda: llvq_cuda::gpu::Cuda,
    /// The tile this module was **compiled** with, resolved from the card
    /// before the source was assembled.
    ///
    /// It lived here as a second `const TILE_BLOCKS = 128`, unlinked from
    /// `llvq_cuda::TILE_BLOCKS` — the defect that constant's own comment warns
    /// against, *"the Metal side carried it for months, TILE_BLOCKS defined
    /// twice, unlinked"*, reintroduced across the crate boundary. Held as one
    /// value because it feeds two things that must never disagree: the
    /// `#define` NVRTC compiled and the shared bytes every launch asks for.
    /// A launch whose shared parameter is smaller than the compiled tile
    /// overruns its staging area with no diagnostic.
    pub(super) tile: llvq_cuda::tile::Tile,
    f_rot: CudaFunction,
    /// `rot_apply` with the rows in the grid — one launch a chunk instead of
    /// one a row. Not an `Option`: `rotate.cu` is in every layout's source
    /// list, because every layout rotates, so the symbol is in every unit.
    f_rot_rows: CudaFunction,
    /// Whichever entry point [`matvec_kernel_name`] gave for the layout — one
    /// kernel per runtime, chosen with the layout, so a stream and a kernel
    /// of different layouts cannot meet.
    pub(super) f_matvec: CudaFunction,
    /// `tv_planes_seg_h` — present exactly when the layout can be segmented
    /// **and** the loader was asked to fuse. The `f_emb`/`g70_tabs` pattern: an
    /// `Option` whose `Some` is the *authorisation*, so no launch path can
    /// reach a function the source list never carried.
    f_matvec_seg: Option<CudaFunction>,
    pub(super) tab: CudaSlice<u32>,
    /// The Golay70 constant tables `(cwtab, gtab)` — the canonical 4096-word
    /// codeword table and the 512-entry `GolayClassRec` table — present
    /// exactly when the layout is `Golay70`, the `f_emb` pattern.
    pub(super) g70_tabs: Option<(CudaSlice<u32>, CudaSlice<u32>)>,
    /// The Tetra constant tables, present exactly when the layout is
    /// `Tetra48` — the `f_emb` pattern, where the `Some` is the
    /// *authorisation*: no launch path can reach a table the source list never
    /// carried.
    pub(super) tetra_tabs: Option<TetraTabs>,
    /// Bytes the prefill kernel stages, formed once so the launch and the
    /// check against the card are one number.
    pub(super) prefill_shared: u32,
    /// The `(rows, tile)` the prefill kernel was COMPILED at. The launch bound
    /// and the `#define`s come from this one value, so a runtime that staged
    /// one pair and launched another cannot exist.
    pub prefill: llvq_cuda::tile::Prefill,
    /// `tv_tetra48_rows_h` — present exactly when the layout carries a prefill
    /// kernel. The decode path never reaches it: at one row the one-row kernel
    /// is what runs, and every decode-time number stays attached to it.
    pub(super) f_matvec_rows: Option<CudaFunction>,
    /// `tv_q4_h` — present exactly when the file carried an int4 record, which
    /// is when its source was appended to the translation unit. The
    /// `f_emb`/`tetra_tabs` pattern: the `Some` is the authorisation.
    ///
    /// 🕳️ Keyed on the SOURCE being in the unit and never on the layout. The
    /// register report asked for `tv_planes_seg_h` because its condition was
    /// "the layout is not Slot32", and the Tetra unit does not carry that
    /// source: `named symbol not found`, on a rented card, after a full load
    /// (2026-09-10). int4 is orthogonal to the layout — that orthogonality is
    /// what makes a mixed file possible — so the layout could not answer this
    /// question even in principle.
    pub(super) f_int4: Option<CudaFunction>,
    /// `tv_q4_hh`, the same matvec reading an f16 activation. Present exactly
    /// when `f_int4` is: both entry points live in `tv_q4_h.cu`.
    pub(super) f_int4_half: Option<CudaFunction>,
    /// `tv_q4_ht` and `tv_q4_hht` (`tv_q4_tile.cu`): present exactly when the
    /// runtime holds its int4 records tiled ([`crate::fused::Int4Layout`]),
    /// which is when that source is in the unit. The `f_int4` pattern.
    pub(super) f_int4_tile: Option<(CudaFunction, CudaFunction)>,
    /// The kernels for widths past the card's shared-memory ceiling, from a
    /// translation unit of their own. `None` for every model whose widths all
    /// fit, which is every served model: their unit, its printed sha256 and
    /// their kernels do not move.
    pub(super) wide: Option<WideFns>,
    pub(super) rotations: HashMap<RotKey, RotBuffers>,
    /// The embedding kernels, `(gather, lm_head matvec)` — present exactly
    /// when the runtime was built with a quantized [`EmbedMode`], which is when
    /// their source was in the translation unit: `emb_q8.cu` under `Q8`,
    /// `emb_q4.cu` under `Q4`.
    pub(super) f_emb: Option<(CudaFunction, CudaFunction)>,
    /// The mode `f_emb` was built for. [`FusedRuntime::upload_embed`] refuses
    /// a table of any other width, since the kernels compiled one.
    pub(super) emode: EmbedMode,
    /// Dynamic shared memory the card allows one block **without asking**,
    /// read at startup — `tv_q8_h` stages the whole activation and must be
    /// refused past it. This is the *default* allowance and not the opt-in
    /// ceiling, because `tv_q8_h` is loaded through `func` and never opts in;
    /// the rotation, which does, is bounded in `new` instead and against both
    /// numbers (`llvq_cuda::shared`).
    shared_limit: usize,
    pub(super) device: candle_core::CudaDevice,
    /// Largest `d_in` any lattice projection or fused group takes. Read by
    /// [`Self::max_d_in`] only; the rotation's staging is decided per width.
    max_d_in: usize,
    /// The unit of `moe_tetra.cu`, compiled on first use ([`Self::moe_kernels`]).
    pub(super) moe: std::sync::OnceLock<MoeKernels>,
}

impl FusedRuntime {
    /// Upload a loaded model and compile the kernels onto candle's stream.
    ///
    /// `emode` decides whether the q8 embedding kernels join the translation
    /// unit; it must match how the caller intends to build the model, and is
    /// taken here rather than re-read from the environment so a runtime and
    /// its loader cannot resolve the variable twice differently. `fuse` is
    /// taken for exactly the same reason, and decides whether the segmented
    /// entry point is looked up at all.
    pub fn new(
        model: &FusedModel,
        device: &Device,
        emode: EmbedMode,
        fuse: FuseMode,
        int4_layout: crate::fused::Int4Layout,
    ) -> candle_core::Result<Loaded> {
        let dev = device.as_cuda_device()?.clone();
        let stream = dev.cuda_stream();
        // Before the source is assembled, not after: the tile is a `#define`
        // in that text, and `Cuda::device()` is a method on the compiled
        // module. The stream already carries its context, so no probe is
        // needed here — unlike the two benches, which build their source
        // before any context exists.
        let tile = llvq_cuda::tile::resolve(
            llvq_cuda::gpu::compute_cap_of(stream.context()).map_err(candle_core::Error::msg)?,
        );

        // The Slot32 translation unit is bit-identical to what shipped before
        // the layout switch existed — that arm is the comparison and the
        // fallback, and its register allocation must not move because a new
        // layout joined the build. Planes14 appends its three parts (in
        // planesbench's proven order: llvq_planes.cuh needs llvq_slot.cuh,
        // planes.cu needs matvec.cu) plus the half-storing entry point.
        let sources = llvq_cuda::load_sources_many(&["llvq_slot.cuh", "matvec.cu", "llvq_rot.cuh", "rotate.cu"])
            .map_err(candle_core::Error::msg)?;
        let planes = match model.layout {
            FusedLayout::Slot32 => None,
            layout => Some(load_planes_sources(layout).map_err(candle_core::Error::msg)?),
        };
        // The embedding pair of the mode, and only it: q8 and q4 are separate
        // files with separate names, so a q8 unit never carries the q4 text.
        let emb = match emode {
            EmbedMode::F16 => None,
            EmbedMode::Q8 => Some(load_emb_sources().map_err(candle_core::Error::msg)?),
            EmbedMode::Q4 => Some(crate::fused::load_emb_q4_sources().map_err(candle_core::Error::msg)?),
        };
        // Appended when the FILE carried an int4 record, never when the layout
        // suggests one: int4 is orthogonal to the lattice layout, which is the
        // whole reason a mixed file can be served at all. Its composition
        // contract is `llvq_slot.cuh` then `matvec.cu` — the first two parts of
        // every unit — so it goes last and nothing before it moves.
        let int4 = match model.int4.is_empty() {
            true => None,
            false => Some(crate::fused::load_int4_sources().map_err(candle_core::Error::msg)?),
        };
        // The prefill pair, resolved once and refused here if it does not fit
        // — before NVRTC sees a byte. `LLVQ_PREFILL` is a measurement mode:
        // it moves no bit of any answer, only how many passes over the weight
        // stream a prompt costs.
        let prefill = llvq_cuda::tile::Prefill::resolve().map_err(candle_core::Error::msg)?;
        let defines = format!("{}{}", tile.define(), prefill.defines());
        let mut parts: Vec<&str> = std::iter::once(defines.as_str())
            .chain(sources.parts.iter().map(String::as_str))
            .collect();
        if let Some((pp, overridden)) = &planes {
            parts.extend(pp.iter().map(String::as_str));
            if let Some(d) = overridden {
                eprintln!("WARNING: {} SOURCES OVERRIDDEN from {d}", model.layout.name());
            }
        }
        if let Some((es, overridden)) = &emb {
            parts.push(es.as_str());
            if let Some(d) = overridden {
                eprintln!("WARNING: emb_{} SOURCE OVERRIDDEN from {d}", emode.name());
            }
        }
        if let Some((cu, overridden)) = &int4 {
            parts.push(cu.as_str());
            if let Some(d) = overridden {
                eprintln!("WARNING: tv_q4_h SOURCE OVERRIDDEN from {d}");
            }
        }
        // The tiled readers, last, and only for a runtime that holds tiled
        // words: every other unit keeps its text and its printed sha256.
        let tiled = int4_layout == crate::fused::Int4Layout::Tiled;
        if tiled {
            if int4.is_none() {
                candle_core::bail!(
                    "int4 layout tiled on a file with no int4 record: the layout names 4E v9's int4 \
                     rows, and this file has none"
                );
            }
            if let Some((_, Some(d))) = &int4 {
                candle_core::bail!(
                    "int4 layout tiled with tv_q4_h.cu overridden from {d}: the tiled reader is embedded \
                     only, and a served config refuses LLVQ_KERNEL_DIR"
                );
            }
            parts.push(crate::fused::Q4_TILE_CU_EMBED);
        }
        let src = llvq_cuda::gpu::KernelSource::new(&parts);
        // On the assembled text, which is the only place where "what NVRTC
        // compiles" and "what this runtime reports" are both visible.
        tile.assert_defined_in(&src.text);
        // The five binaries of `llvq-cuda` print this and the served path never
        // did, while `fused::load_planes_sources` cites "the printed sha256" as
        // the justification of its all-or-nothing override policy. A lot that
        // ADDS a file to the served unit is the one where that stops being a
        // formality: without this line, a run with `LLVQ_KERNEL_DIR` set is
        // traceable by a directory name and nothing else.
        println!(
            "NVRTC source: {} bytes, sha256 {} ({} parts), {}, {}, target {}",
            src.text.len(),
            src.sha256,
            parts.len(),
            tile.provenance(),
            prefill.provenance(),
            // `LLVQ_NVRTC_ARCH` is the one variable below the served door that
            // the config deliberately does not refuse — it names the card, not
            // the object — so it is printed here, where a journal copies from.
            llvq_cuda::gpu::arch()
        );
        let cuda = llvq_cuda::gpu::Cuda::on_stream(stream, &src).map_err(candle_core::Error::msg)?;

        // Both shared-memory bounds of the card, read once, before anything is
        // sized against them: the int4 entry points below and the rotation
        // further down.
        let dev_report = cuda.device().map_err(candle_core::Error::msg)?;
        let (shared_def, shared_optin) = (
            dev_report.shared_per_block as usize,
            dev_report.shared_per_block_optin as usize,
        );
        // One size per int4 entry point: `tv_q4_h` / `tv_q4_hh` over the
        // F32-staged records, `tv_q4_hs` over the F16-staged ones. On every
        // served model no record is F16-staged, and the F32 size is the one
        // `fused::int4_shared_bytes` gave over all records, so nothing moves.
        let int4_entry = llvq_cuda::wide::int4_entry_bytes(
            model.int4.iter().map(|q| q.d_in),
            shared_def,
            shared_optin,
        )
        .map_err(candle_core::Error::msg)?;

        // The register report is a contract, not a diagnostic: the block
        // decoders keep their accumulators and `rot_mix` a KMAX-wide column
        // in registers, and a spill costs occupancy without changing a
        // result. Checked on the kernel this runtime will actually launch.
        let matvec_name = matvec_kernel_name(model.layout);
        let mut spill_checked = vec![matvec_name, "rot_apply", "rot_apply_rows"];
        // In the translation unit whenever ITS SOURCE is — read off the one
        // list that decides, never inferred from "the layout is not Slot32".
        //
        // 🕳️ It was `planes.is_some()`, which is true for every layout but
        // Slot32 — including `Tetra48`, whose list shares nothing with the
        // ball ones and carries no `tv_planes_seg_h.cu`. So the served Tetra
        // path compiled, loaded, reported 216 projections and 0.92 GB on the
        // card, and then died asking the driver for a symbol its own unit had
        // never contained: `no kernel tv_planes_seg_h: named symbol not found`
        // (2026-09-10, $0.03, the second card run of the served path). The
        // register report is a contract, and a contract that names a kernel
        // the build does not have is a crash rather than a check.
        if crate::fused::planes_source_names(model.layout).contains(&"tv_planes_seg_h.cu") {
            spill_checked.push("tv_planes_seg_h");
        }
        // Keyed on the mode that chose the source above, through the one
        // table `fused.rs` pins against the source text.
        if let Some((gather, head)) = crate::fused::emb_kernel_names(emode) {
            spill_checked.extend([gather, head]);
        }
        // 🚨 The prefill kernel, reported exactly when its source is in the
        // unit — and it is the arm where a spill is most likely and would cost
        // the most. `float acc[TETRA48_ROWS]` is one register a row: four is
        // nothing, sixteen is a sixth of a thread's 64-register budget at full
        // occupancy, and a spill there turns the accumulator into local memory
        // — which is DRAM, on the one kernel whose whole point is to touch
        // DRAM less. The report is what says so before a prompt is timed.
        if let Some(n) = crate::fused::rows_kernel_name(model.layout) {
            spill_checked.push(n);
        }
        // Same rule as the segmented kernel above: reported exactly when its
        // source is in the unit, which here is exactly when `int4` is `Some`.
        if int4.is_some() {
            spill_checked.push(crate::fused::INT4_KERNEL_NAME);
            spill_checked.push(crate::fused::INT4_HALF_KERNEL_NAME);
        }
        if tiled {
            spill_checked.push(crate::fused::INT4_TILE_KERNEL_NAME);
            spill_checked.push(crate::fused::INT4_TILE_HALF_KERNEL_NAME);
        }
        for name in spill_checked {
            let r = cuda.report(name).map_err(candle_core::Error::msg)?;
            if r.local_bytes != 0 {
                candle_core::bail!("{name}: {} bytes of spill", r.local_bytes);
            }
        }
        let f_matvec = cuda.func(matvec_name).map_err(candle_core::Error::msg)?;
        // The prefill entry point. `Some` is the authorisation, and it is
        // keyed on the LAYOUT naming one — which `fused.rs` pins to the source
        // list, so the pair cannot drift the way `tv_planes_seg_h` did.
        // Formed from the SAME tile the module was compiled at, and refused
        // against the card rather than assumed: this kernel is loaded through
        // `func` with no opt-in, so the default per-block allowance is the
        // bound. At the served tile of 128 four rows is exactly 49,152.
        // The VALUE here, where the tile is known; the CHECK against the card
        // ninety lines down, where `shared_limit` is read from the driver.
        // They were one statement and it named `shared_limit` before that line
        // existed — which this machine could not see, because the file needs
        // nvcc, and which the Space build reported in two errors.
        let prefill_shared = prefill.shared_bytes();
        let f_matvec_rows = match crate::fused::rows_kernel_name(model.layout) {
            Some(n) => Some(cuda.func(n).map_err(candle_core::Error::msg)?),
            None => None,
        };
        // Staged whole, `d_in` floats a block: an int4 `down_proj` at 14B
        // stages 69,632 B, past the 48 KiB default, so the opt-in is posed on
        // the function here, as for `rot_apply`. Under the default,
        // `func_dynamic_shared` is `func` and nothing changes.
        let (f_int4, f_int4_half) = match int4.is_some() {
            true => {
                // Both entry points stage `d_in` f32, whatever `x` arrives as,
                // and are sized over the records that stage in f32.
                let bytes = int4_entry.f32.unwrap_or(0);
                let f = |name: &str| {
                    cuda.func_dynamic_shared(name, bytes as u32)
                        .map_err(candle_core::Error::msg)
                };
                (
                    Some(f(crate::fused::INT4_KERNEL_NAME)?),
                    Some(f(crate::fused::INT4_HALF_KERNEL_NAME)?),
                )
            }
            false => (None, None),
        };
        // The tiled pair stages what the served pair stages, so the same
        // opt-in. Every tiled record is F32-staged (`upload_int4` refuses the
        // rest), so `int4_entry.f32` covers all of them.
        let f_int4_tile = match tiled {
            true => {
                let bytes = int4_entry.f32.unwrap_or(0) as u32;
                let f = |name: &str| cuda.func_dynamic_shared(name, bytes).map_err(candle_core::Error::msg);
                Some((
                    f(crate::fused::INT4_TILE_KERNEL_NAME)?,
                    f(crate::fused::INT4_TILE_HALF_KERNEL_NAME)?,
                ))
            }
            false => None,
        };
        // Looked up only when both the layout and the caller allow it: the
        // `Some` is the authorisation, so `forward_rotated_seg` has nothing to
        // fall back on rather than something to check.
        let f_matvec_seg = match (fuse, seg_kernel_name(model.layout)) {
            (FuseMode::On, Some(n)) => Some(cuda.func(n).map_err(candle_core::Error::msg)?),
            _ => None,
        };
        // `f_rot` is loaded further down, once the widest rotation is known:
        // staging past 48 KiB needs an opt-in posed on the *function*, and it
        // has to name the number of bytes. See the shared-memory block below.
        let f_emb = match crate::fused::emb_kernel_names(emode) {
            None => None,
            Some((gather, head)) => Some((
                cuda.func(gather).map_err(candle_core::Error::msg)?,
                cuda.func(head).map_err(candle_core::Error::msg)?,
            )),
        };

        // The 384-entry class table both sides of the format share, laid out
        // as the kernel's `ClassRec { float vals[5]; u32 len; }`.
        let fd = llvq_search::fastdec::FastDecoder::new();
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
        let tab = cuda.up_u32(&tab).map_err(candle_core::Error::msg)?;

        // The Golay70 constant tables, uploaded once and shared by every
        // matrix — built from the same `Golay70Table` derivation the
        // transcoder encoded against (`fused::golay70_gpu_class_table`), so
        // encoder and decoder cannot drift apart.
        // The Tetra tables, from the SAME `llvq_search::tetra::Tetra` the
        // encoder used — `crate::fused::tetra48_tables` owns the shapes and is
        // checked on a machine without a card.
        let tetra_tabs = match model.layout {
            FusedLayout::Tetra48 => {
                let tb = crate::fused::tetra48_tables(&llvq_search::tetra::Tetra::new());
                Some(TetraTabs {
                    rows: cuda.up_u32(&tb.rows).map_err(candle_core::Error::msg)?,
                    prefixes: cuda.up_u32(&tb.prefixes).map_err(candle_core::Error::msg)?,
                    branches: cuda.up_u16(&tb.branches).map_err(candle_core::Error::msg)?,
                    suffixes: cuda.up_u32(&tb.suffixes).map_err(candle_core::Error::msg)?,
                    invnorm: cuda.up_f32(&tb.invnorm).map_err(candle_core::Error::msg)?,
                })
            }
            _ => None,
        };

        let g70_tabs = match model.layout {
            FusedLayout::Golay70 => {
                let g70 = llvq_artifact::runtime::Golay70Table::new(&fd);
                let cw = cuda
                    .up_u32(&crate::fused::golay70_gpu_codewords(&g70))
                    .map_err(candle_core::Error::msg)?;
                let gt = cuda
                    .up_u32(&crate::fused::golay70_gpu_class_table(&fd, &g70))
                    .map_err(candle_core::Error::msg)?;
                Some((cw, gt))
            }
            _ => None,
        };

        // The DEFAULT allowance, and it stays the default on purpose: this
        // bound belongs to `tv_q8_h`, which stages `d` floats and is launched
        // through `func`, with no opt-in. Widening it here would loosen a
        // guard on a kernel that never asked the driver for anything.
        let shared_limit = dev_report.shared_per_block as usize;

        // The prefill staging, refused against the card rather than assumed:
        // that kernel is loaded through `func` with no opt-in, so the DEFAULT
        // per-block allowance is its bound — the same argument the rotation
        // makes just below. At the served tile of 128, four rows is exactly
        // 49,152 B, so this refuses nothing that exists today and names the
        // pair that would stop fitting.
        if crate::fused::rows_kernel_name(model.layout).is_some() && prefill_shared > shared_limit {
            candle_core::bail!(
                "the prefill kernel stages {prefill_shared} B ({} rows at tile {}), \
                 and the card allows {shared_limit}",
                prefill.rows,
                prefill.tile
            );
        }
        let prefill_shared = prefill_shared as u32;

        // The rotation is the one staging that can exceed the default — and
        // comparing it against `shared_limit` is what refused Qwen3-14B on
        // 2026-08-17 (69,632 o wanted, 49,152 offered by default, 101,376
        // available on request). Both bounds now, from `llvq_cuda::shared`,
        // which is where this arithmetic is testable: this file compiles
        // nowhere but inside an image build.
        //
        // A width `rot_plan` accepts keeps `rot_apply`; a width it refuses
        // (25,600 on a 99 KiB card) goes to `rot_apply_dev`, which stages in
        // device memory and returns the same bits. `wide::rot_kernel` is that
        // rule, and its tests pin every served width on `rot_apply`.
        let rot_kernel = |n: usize| llvq_cuda::wide::rot_kernel(n, shared_def, shared_optin);
        // Posed on the function, once, for the widest rotation this model
        // stages in SHARED memory — before any launch, and never per token.
        // Every served model stages all of its rotations there, so this is
        // the widest rotation, as before.
        let rot_bytes = model
            .rotations
            .values()
            .filter(|t| rot_kernel(t.n) != llvq_cuda::wide::RotKernel::Device)
            .map(|t| llvq_cuda::shared::rot_bytes(t.n))
            .max()
            .unwrap_or(0);
        let f_rot = cuda
            .func_dynamic_shared("rot_apply", rot_bytes as u32)
            .map_err(candle_core::Error::msg)?;
        // Same bound, and it is the same bound for a reason: the rows variant
        // puts one row in a block and the rows in the grid, so its shared
        // memory per block is one row's, unchanged at any chunk length.
        let f_rot_rows = cuda
            .func_dynamic_shared("rot_apply_rows", rot_bytes as u32)
            .map_err(candle_core::Error::msg)?;

        // The wide unit: compiled only when a width needs it, as a module of
        // its own on the same stream, so the served unit above is untouched.
        let mut device_widths: Vec<usize> = model
            .rotations
            .values()
            .filter(|t| rot_kernel(t.n) == llvq_cuda::wide::RotKernel::Device)
            .map(|t| t.n)
            .collect();
        device_widths.sort_unstable();
        device_widths.dedup();
        let wide = match (device_widths.is_empty(), int4_entry.f16) {
            (true, None) => None,
            (_, f16_bytes) => {
                let mut names = vec!["llvq_slot.cuh", "matvec.cu"];
                if !device_widths.is_empty() {
                    names.extend(["llvq_rot.cuh", "rotate_wide.cu"]);
                }
                let base = llvq_cuda::load_sources_many(&names).map_err(candle_core::Error::msg)?;
                let int4w = match f16_bytes {
                    Some(_) => Some(
                        crate::fused::load_int4_wide_sources().map_err(candle_core::Error::msg)?,
                    ),
                    None => None,
                };
                let mut wparts: Vec<&str> = std::iter::once(defines.as_str())
                    .chain(base.parts.iter().map(String::as_str))
                    .collect();
                if let Some((ss, _)) = &int4w {
                    wparts.extend(ss.iter().map(String::as_str));
                }
                for d in base.overridden_from.iter().chain(int4w.iter().flat_map(|(_, d)| d)) {
                    eprintln!("WARNING: wide unit SOURCES OVERRIDDEN from {d}");
                }
                let wsrc = llvq_cuda::gpu::KernelSource::new(&wparts);
                let f16_records = model
                    .int4
                    .iter()
                    .filter(|q| {
                        llvq_cuda::wide::int4_staging(q.d_in, shared_def, shared_optin)
                            == Some(llvq_cuda::wide::Int4Staging::F16)
                    })
                    .count();
                println!(
                    "NVRTC wide unit: {} bytes, sha256 {} ({} parts): rot_apply_dev for widths \
                     {device_widths:?}, tv_q4_hs for {f16_records} int4 record(s); the card \
                     offers {shared_def} B by default and {shared_optin} after opt-in",
                    wsrc.text.len(),
                    wsrc.sha256,
                    wparts.len(),
                );
                let wcuda = llvq_cuda::gpu::Cuda::on_stream(dev.cuda_stream(), &wsrc)
                    .map_err(candle_core::Error::msg)?;
                let mut checked: Vec<&str> = Vec::new();
                if !device_widths.is_empty() {
                    checked.extend(["rot_apply_dev", "rot_apply_rows_dev"]);
                }
                if f16_bytes.is_some() {
                    checked.push(crate::fused::INT4_WIDE_KERNEL_NAME);
                }
                for name in checked {
                    let r = wcuda.report(name).map_err(candle_core::Error::msg)?;
                    if r.local_bytes != 0 {
                        candle_core::bail!("{name}: {} bytes of spill", r.local_bytes);
                    }
                }
                let rot = match device_widths.is_empty() {
                    true => None,
                    false => Some((
                        wcuda.func("rot_apply_dev").map_err(candle_core::Error::msg)?,
                        wcuda.func("rot_apply_rows_dev").map_err(candle_core::Error::msg)?,
                    )),
                };
                // Its own opt-in, at the widest F16-staged record: 51,200 B at
                // 25,600 is past the 49,152 B default.
                let int4 = match f16_bytes {
                    Some(b) => Some(
                        wcuda
                            .func_dynamic_shared(crate::fused::INT4_WIDE_KERNEL_NAME, b as u32)
                            .map_err(candle_core::Error::msg)?,
                    ),
                    None => None,
                };
                Some(WideFns { rot, int4, _module: wcuda })
            }
        };

        let mut rotations = HashMap::new();
        for (&key, t) in &model.rotations {
            rotations.insert(key, upload_rotation(&cuda, t, rot_kernel(t.n))?);
        }

        // Over the groups as well: under fusion most of the model's activations
        // are a group's, and a bound taken over the lone projections alone
        // would be a bound over `o_proj` and `down_proj`.
        let max_d_in = model
            .matrices
            .iter()
            .map(|m| m.d_in)
            .chain(model.groups.iter().map(|g| g.d_in))
            .max()
            .unwrap_or(0);
        let mut projs = Vec::with_capacity(model.matrices.len());
        for m in &model.matrices {
            projs.push(upload_matrix(&cuda, m, model.layout)?);
        }
        let mut seg_projs = Vec::with_capacity(model.groups.len());
        for g in &model.groups {
            seg_projs.push(upload_group(&cuda, g, model.layout)?);
        }
        // `tv_q4_h` was loaded with its opt-in above, so its bound is the
        // ceiling, not the default `tv_q8_h` keeps in `shared_limit`. Each
        // record gets its staging here, F32 wherever it fits.
        let mut int4_projs = Vec::with_capacity(model.int4.len());
        for q in &model.int4 {
            int4_projs.push(upload_int4(&cuda, q, shared_def, shared_optin, int4_layout)?);
        }
        // On the first F32-staged record, which on every served model is the
        // first record.
        let first_f32 =
            int4_projs.iter().find(|p| p.staging == llvq_cuda::wide::Int4Staging::F32);
        match (&f_int4_tile, first_f32) {
            // Tiled: the f16 entry point against the f32 one on the tiled
            // words, as the served pair below; then the tiled reader against
            // the served one on a served copy of the same record, one record
            // of every width, so the tail of a width that has one is read too.
            (Some((fa, fb)), Some(p)) => {
                check_int4_half_parity(&cuda, fa, fb, p, crate::fused::INT4_TILE_KERNEL_NAME, crate::fused::INT4_TILE_HALF_KERNEL_NAME)
                    .map_err(candle_core::Error::msg)?;
                let f_hh = f_int4_half.as_ref().expect("a tiled runtime carries tv_q4_h.cu");
                let mut seen = Vec::new();
                for (q, p) in model.int4.iter().zip(&int4_projs) {
                    if !seen.contains(&q.d_in) {
                        seen.push(q.d_in);
                        check_int4_tile_parity(&cuda, f_hh, fb, q, p, shared_def, shared_optin)
                            .map_err(candle_core::Error::msg)?;
                    }
                }
            }
            _ => {
                if let (Some(fa), Some(fb), Some(p)) = (&f_int4, &f_int4_half, first_f32) {
                    check_int4_half_parity(&cuda, fa, fb, p, crate::fused::INT4_KERNEL_NAME, crate::fused::INT4_HALF_KERNEL_NAME)
                        .map_err(candle_core::Error::msg)?;
                }
            }
        }
        // A model with an F16-staged record: `tv_q4_hs` against `tv_q4_hh`,
        // bytes of every row, on a record where both fit. Never reached by a
        // served model.
        if let Some(hs) = wide.as_ref().and_then(|w| w.int4.as_ref()) {
            match (&f_int4_half, first_f32) {
                (Some(hh), Some(p)) => {
                    check_int4_wide_parity(&cuda, hh, hs, p).map_err(candle_core::Error::msg)?
                }
                _ => println!(
                    "int4: no F32-staged record to hold tv_q4_hs against tv_q4_hh at load; \
                     LLVQ_INT4_WIDE_CHECK holds it against the host on synthetic records"
                ),
            }
        }

        Ok((
            Self {
                cuda,
                tile,
                f_rot,
                f_rot_rows,
                f_matvec,
                f_matvec_seg,
                tab,
                tetra_tabs,
                prefill_shared,
                prefill,
                f_matvec_rows,
                f_int4,
                f_int4_half,
                f_int4_tile,
                wide,
                g70_tabs,
                rotations,
                f_emb,
                emode,
                shared_limit,
                device: dev,
                max_d_in,
                moe: std::sync::OnceLock::new(),
            },
            projs,
            seg_projs,
            int4_projs,
        ))
    }

    /// One rotation, through the kernel its upload chose: `rot_apply` exactly
    /// as before wherever it runs on this card, `rot_apply_dev` on a scratch
    /// of its own where it does not. The scratch is allocated per call, like
    /// the output beside it, so no two launches share one.
    pub(super) fn launch_rotation(
        &self,
        rot: &RotBuffers,
        all: &CudaSlice<f16>,
        xr: &mut CudaSlice<f32>,
        start: u32,
    ) -> candle_core::Result<()> {
        match rot.kernel {
            llvq_cuda::wide::RotKernel::Shared(_) => self.cuda.launch_rot(
                &self.f_rot, all, &rot.signbits, &rot.small, xr, rot.n, rot.m, rot.k, rot.inv,
                start, rot.threads,
            ),
            llvq_cuda::wide::RotKernel::Device => {
                let (f, _) = self.wide_rot(rot.n)?;
                let len = llvq_cuda::wide::rot_scratch_floats(
                    1,
                    rot.n as usize,
                    rot.m as usize,
                    rot.k as usize,
                );
                let mut scratch = unsafe { self.device.cuda_stream().alloc::<f32>(len) }
                    .map_err(|e| candle_core::Error::msg(format!("alloc rot scratch: {e}")))?;
                self.cuda.launch_rot_dev(
                    f, all, &rot.signbits, &rot.small, xr, &mut scratch, rot.n, rot.m, rot.k,
                    rot.inv, start, rot.threads,
                )
            }
        }
        .map_err(candle_core::Error::msg)
    }

    /// [`Self::launch_rotation`] for `n_rows` rows, one launch.
    pub(super) fn launch_rotation_rows(
        &self,
        rot: &RotBuffers,
        all: &CudaSlice<f16>,
        xr: &mut CudaSlice<f32>,
        start: u32,
        row_stride: u32,
        n_rows: u32,
    ) -> candle_core::Result<()> {
        match rot.kernel {
            llvq_cuda::wide::RotKernel::Shared(_) => self.cuda.launch_rot_rows(
                &self.f_rot_rows, all, &rot.signbits, &rot.small, xr, rot.n, rot.m, rot.k,
                rot.inv, start, row_stride, n_rows, rot.threads,
            ),
            llvq_cuda::wide::RotKernel::Device => {
                let (_, f) = self.wide_rot(rot.n)?;
                let len = llvq_cuda::wide::rot_scratch_floats(
                    n_rows as usize,
                    rot.n as usize,
                    rot.m as usize,
                    rot.k as usize,
                );
                let mut scratch = unsafe { self.device.cuda_stream().alloc::<f32>(len) }
                    .map_err(|e| candle_core::Error::msg(format!("alloc rot scratch: {e}")))?;
                self.cuda.launch_rot_rows_dev(
                    f, all, &rot.signbits, &rot.small, xr, &mut scratch, rot.n, rot.m, rot.k,
                    rot.inv, start, row_stride, n_rows, rot.threads,
                )
            }
        }
        .map_err(candle_core::Error::msg)
    }

    /// The device-staged rotation pair, or a refusal that names the width:
    /// a `Device` rotation on a runtime whose wide unit has no rotation is a
    /// loader bug, never a fallback.
    fn wide_rot(&self, n: u32) -> candle_core::Result<&(CudaFunction, CudaFunction)> {
        self.wide.as_ref().and_then(|w| w.rot.as_ref()).ok_or_else(|| {
            candle_core::Error::msg(format!(
                "rotation of width {n} is device-staged, and the runtime carries no \
                 rot_apply_dev: the wide unit is compiled only for the widths the load saw"
            ))
        })
    }

    /// `LLVQ_INT4_WIDE_CHECK`: the f16-staged int4 kernel against `tv_q4_hh`
    /// and against the host, bytes of every row, on this card.
    ///
    /// Compiles the wide int4 unit as a module of its own, whatever the model
    /// needs, then:
    ///   1. on the model's first F32-staged record, `tv_q4_hs` against the
    ///      runtime's own `tv_q4_hh`;
    ///   2. on each synthetic record of `fused::INT4_WIDE_CARD_CASES`,
    ///      `tv_q4_hs`'s output sha256 against the host's
    ///      (`fused::INT4_WIDE_FIXTURE`), and `tv_q4_hh` beside it wherever its
    ///      f32 staging fits the card (25,344 on a 99 KiB card, not 25,600).
    ///
    /// Every comparison is printed; any difference is an error, after all of
    /// them ran. No token is timed.
    pub(super) fn int4_wide_check(&self, projs: &[FusedInt4Proj]) -> Result<(), String> {
        let dev = self.cuda.device()?;
        let (def, optin) = (dev.shared_per_block as usize, dev.shared_per_block_optin as usize);
        let base = llvq_cuda::load_sources_many(&["llvq_slot.cuh", "matvec.cu"])?;
        let (int4w, overridden) = crate::fused::load_int4_wide_sources()?;
        let defines = format!("{}{}", self.tile.define(), self.prefill.defines());
        let parts: Vec<&str> = std::iter::once(defines.as_str())
            .chain(base.parts.iter().map(String::as_str))
            .chain(int4w.iter().map(String::as_str))
            .collect();
        for d in base.overridden_from.iter().chain(overridden.iter()) {
            println!("  WARNING: wide int4 SOURCES OVERRIDDEN from {d}");
        }
        let src = llvq_cuda::gpu::KernelSource::new(&parts);
        println!(
            "LLVQ_INT4_WIDE_CHECK: wide int4 unit {} bytes, sha256 {}; the card offers {def} B by \
             default and {optin} after opt-in; fixture sha256 {}",
            src.text.len(),
            src.sha256,
            llvq_cuda::sha256::sha256_hex(crate::fused::INT4_WIDE_FIXTURE.as_bytes())
        );
        let cuda = llvq_cuda::gpu::Cuda::on_stream(self.device.cuda_stream(), &src)?;
        let name = crate::fused::INT4_WIDE_KERNEL_NAME;
        let rep = cuda.report(name)?;
        println!("  {name}: {} registers, {} local B", rep.num_regs, rep.local_bytes);
        if rep.local_bytes != 0 {
            return Err(format!("{name}: {} bytes of spill", rep.local_bytes));
        }
        let mut failures: Vec<String> = Vec::new();
        let le = |v: &[u16]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };

        // 1. The model's first F32-staged record.
        let first = projs.iter().find(|p| p.staging == llvq_cuda::wide::Int4Staging::F32);
        match (first, self.f_int4_half.as_ref()) {
            (Some(p), Some(hh)) => {
                let hs = cuda.func_dynamic_shared(name, (p.d_in * 2) as u32)?;
                if let Err(e) = check_int4_wide_parity(&cuda, hh, &hs, p) {
                    println!("  {e}");
                    failures.push(e);
                }
            }
            _ => println!("  no F32-staged int4 record in this model: step 1 skipped"),
        }

        // 2. The synthetic records, against the host.
        let fixture = crate::fused::parse_int4_wide_fixture(crate::fused::INT4_WIDE_FIXTURE)?;
        for (d_out, d_in, seed, x_off) in crate::fused::INT4_WIDE_CARD_CASES {
            let want = fixture
                .get(&(d_out, d_in, seed, x_off))
                .ok_or_else(|| format!("no fixture line for {d_out}x{d_in} {seed:#x} {x_off}"))?;
            let (rec, x) = crate::fused::int4_synthetic(d_out, d_in, seed);
            let p = upload_int4(&cuda, &rec, def, optin, crate::fused::Int4Layout::Served).map_err(|e| e.to_string())?;
            let mut xb = vec![0x3c00u16; x_off];
            xb.extend(&x);
            let dx = cuda.up_u16(&xb)?;
            let hs = cuda.func_dynamic_shared(name, (d_in * 2) as u32)?;
            let mut y = cuda.up_u16(&vec![0xffffu16; d_out])?;
            launch_q4_hs(&cuda, &hs, &p, &dx, &mut y, x_off as u32, THREADS)?;
            let got = cuda.down_u16(&y)?;
            let h = llvq_cuda::sha256::sha256_hex(&le(&got));
            let hs_ok = &h == want;
            let mut line = format!(
                "  {d_out}x{d_in} seed {seed:#x} x_off {x_off}: tv_q4_hs sha256 {} the host's",
                if hs_ok { "EQUALS" } else { "DIFFERS FROM" }
            );
            if !hs_ok {
                failures.push(format!("{d_out}x{d_in} {seed:#x} {x_off}: tv_q4_hs {h} != {want}"));
            }
            let f32_fits = llvq_cuda::wide::Int4Staging::F32.bytes(d_in)
                <= llvq_cuda::shared::ceiling(def, optin);
            if f32_fits && x_off == 0 {
                let hh = cuda.func_dynamic_shared(
                    crate::fused::INT4_HALF_KERNEL_NAME,
                    (d_in * 4) as u32,
                )?;
                let mut y2 = cuda.up_u16(&vec![0u16; d_out])?;
                launch_q4_h(&cuda, &hh, &p, &dx, &mut y2, THREADS)?;
                let hh_out = cuda.down_u16(&y2)?;
                let same = hh_out == got;
                line.push_str(&format!(
                    "; tv_q4_hh {} it",
                    if same { "equals" } else { "DIFFERS FROM" }
                ));
                if !same {
                    failures.push(format!("{d_out}x{d_in} {seed:#x}: tv_q4_hh != tv_q4_hs"));
                }
            } else {
                line.push_str("; tv_q4_hh not run (its f32 staging does not fit, or x_off)");
            }
            println!("{line}");
        }
        if failures.is_empty() {
            println!("LLVQ_INT4_WIDE_CHECK: every comparison equal");
            Ok(())
        } else {
            Err(format!("LLVQ_INT4_WIDE_CHECK: {} failure(s): {}", failures.len(), failures.join("; ")))
        }
    }

    /// The widest activation any lattice projection or fused group takes;
    /// int4 records are not counted.
    ///
    /// Informative only: `new` compares it to nothing. Which rotation kernel
    /// a width gets, and whether it stages in shared or device memory, is
    /// decided per width by `llvq_cuda::wide::rot_kernel` at upload.
    pub fn max_d_in(&self) -> usize {
        self.max_d_in
    }

    /// One activation `[1, d_in]` in f16 → the same activation in the rotated
    /// basis, `[1, d_in]` in f32. Half of what `forward` used to do in one
    /// call; `crate::rotplan` says why it was split and why nothing is cached.
    /// The row loop lives in `model::group_forward` now, hence the one row.
    pub fn rotate(&self, proj: &FusedProj, x: &Tensor) -> candle_core::Result<Tensor> {
        let dims = x.dims();
        let d_in = *dims.last().expect("rank >= 1");
        if d_in != proj.d_in {
            candle_core::bail!("{} expects d_in={}, got {d_in}", proj.name, proj.d_in);
        }
        let rows: usize = dims[..dims.len() - 1].iter().product();
        if rows != 1 {
            candle_core::bail!(
                "{}: rotation requested for {rows} vectors. The row loop belongs to \
                 model::group_forward, which shares it across the projections of a group.",
                proj.name
            );
        }
        let x = x.to_dtype(DType::F16)?;
        let op = RotOp { rt: self, proj };
        x.apply_op1_no_bwd(&op)
    }

    /// [`Self::rotate`] for `n_rows` activations, ONE launch.
    ///
    /// `x` is `[n_rows, d_in]` f16 and contiguous — `model::row_block` narrows
    /// it out of one allocation, so its rows are `d_in` apart and no copy was
    /// made to put them there. The result is `[n_rows, d_in]` f32, contiguous,
    /// which is exactly what `forward_rotated_rows` reads.
    ///
    /// That last sentence is the whole point. Before this, a chunk was rotated
    /// a row at a time into `n_rows` separate allocations and then stacked with
    /// `Tensor::cat` — one device copy a row — to rebuild the very shape the
    /// matvec wanted. Now the shape comes out of the rotation.
    pub fn rotate_rows(
        &self,
        proj: &FusedProj,
        x: &Tensor,
        n_rows: usize,
    ) -> candle_core::Result<Tensor> {
        let dims = x.dims();
        let d_in = *dims.last().expect("rank >= 1");
        if d_in != proj.d_in {
            candle_core::bail!("{} expects d_in={}, got {d_in}", proj.name, proj.d_in);
        }
        let rows: usize = dims[..dims.len() - 1].iter().product();
        if rows != n_rows {
            candle_core::bail!("{}: {rows} rows of activation for {n_rows}", proj.name);
        }
        if n_rows == 0 {
            candle_core::bail!("{}: a rotation of zero rows", proj.name);
        }
        let x = x.to_dtype(DType::F16)?;
        let op = RotRowsOp { rt: self, proj, n_rows };
        x.apply_op1_no_bwd(&op)
    }

    /// `y = W' xr` for one activation already in the rotated basis. `xr` is
    /// [`Self::rotate`]'s f32 output; `out_dims` is the *caller's* shape, so
    /// the result keeps the caller's rank, exactly as a `Linear` would.
    /// `y = W x` for an int4 projection, from the activation in its **natural**
    /// basis — there is no rotated form, and asking for one would be asking for
    /// a basis the weights were never quantized in.
    ///
    /// The kernel accumulates against `x` in f32, "as in every projection
    /// kernel of this family", while the model carries f16. An f16 activation
    /// goes to `tv_q4_hh`, which widens it while staging: exact, so the same
    /// floats as a cast, and one launch fewer a call (84 a token on the sealed
    /// 4B). Any other dtype is widened here and goes to `tv_q4_h`. Neither
    /// narrows the activation: the reference this path is checked against,
    /// `RawTensor::to_f32`, is f32 arithmetic.
    pub fn forward_int4(
        &self,
        proj: &FusedInt4Proj,
        x: &Tensor,
        out_dims: &[usize],
    ) -> candle_core::Result<Tensor> {
        let out_shape = {
            let mut d = out_dims.to_vec();
            *d.last_mut().expect("rank >= 1") = proj.d_out;
            Shape::from(d)
        };
        // The kernel reads from the base pointer. The cast always handed it a
        // fresh tensor at offset zero; a row the prefill narrows out of a
        // block is still at an offset, so that case keeps the cast and
        // `tv_q4_h`, as before. A decode step's activation is fresh already.
        // (`Tensor::copy` would not do: it keeps the layout, offset included.)
        let c = x.contiguous()?;
        // A record staged in f16 (`tv_q4_hs`, past the card's ceiling for the
        // f32 staging) takes its f16 activation where it lies, offset
        // included. An f32 activation cannot reach it: narrowing would change
        // the arithmetic, and `tv_q4_h` does not fit this card at this width.
        if proj.staging == llvq_cuda::wide::Int4Staging::F16 {
            if c.dtype() != DType::F16 {
                candle_core::bail!(
                    "{}: an f32 activation at an int4 record staged in f16 (d_in {}). Narrowing \
                     it would change the arithmetic, and tv_q4_h's f32 staging does not fit \
                     this card at this width",
                    proj.name,
                    proj.d_in
                );
            }
            let op = FusedInt4Op { rt: self, proj, out_shape, path: Int4Path::Wide };
            return c.apply_op1_no_bwd(&op);
        }
        let (xs, half) = match (c.dtype(), c.layout().start_offset()) {
            (DType::F16, 0) => (c, true),
            _ => (c.to_dtype(DType::F32)?.contiguous()?, false),
        };
        let op = FusedInt4Op {
            rt: self,
            proj,
            out_shape,
            path: if half { Int4Path::Half } else { Int4Path::F32 },
        };
        xs.apply_op1_no_bwd(&op)
    }

    /// Whether this runtime's layout carries a kernel that takes several rows.
    pub fn has_rows_kernel(&self) -> bool {
        self.f_matvec_rows.is_some()
    }

    /// Rows this runtime's prefill kernel was COMPILED to take.
    ///
    /// `model::group_forward` chunks by it. Read from the runtime and not from
    /// `tile::PREFILL_ROWS`, because under `LLVQ_PREFILL` they differ — and a
    /// chunk of eight handed to a kernel compiled at four is exactly the
    /// silent corruption the pair exists to prevent.
    pub fn prefill_rows(&self) -> usize {
        self.prefill.rows
    }

    /// `y = W X` for `n_rows` rotated rows, one launch.
    ///
    /// `xr` is `[n_rows, d_in]` f32, contiguous, and based at offset zero —
    /// [`RotRowsOp`] allocates it, so all three hold by construction. It used
    /// to be built by `Tensor::cat` of `n_rows` single-row rotations, one
    /// device copy each; the rotation produces the shape now.
    pub fn forward_rotated_rows(
        &self,
        proj: &FusedProj,
        xr: &Tensor,
        n_rows: usize,
    ) -> candle_core::Result<Tensor> {
        if n_rows == 0 || n_rows > self.prefill.rows {
            candle_core::bail!(
                "{n_rows} rows for a kernel compiled at {}",
                self.prefill.rows
            );
        }
        let op = FusedRowsOp {
            rt: self,
            proj,
            n_rows,
            out_shape: Shape::from(vec![n_rows, proj.d_out]),
        };
        xr.apply_op1_no_bwd(&op)
    }

    pub fn forward_rotated(
        &self,
        proj: &FusedProj,
        xr: &Tensor,
        out_dims: &[usize],
    ) -> candle_core::Result<Tensor> {
        let out_shape = {
            let mut d = out_dims.to_vec();
            *d.last_mut().expect("rank >= 1") = proj.d_out;
            Shape::from(d)
        };
        let op = FusedOp {
            rt: self,
            proj,
            out_shape,
        };
        // No `to_dtype` on the result: the kernel already stored halves.
        xr.apply_op1_no_bwd(&op)
    }

    /// [`Self::rotate`] for a fused group — one `rot_apply`, one f32
    /// `[1, d_in]` result. The parts share one `d_in` and one rotation key by
    /// construction (`fused::segment_matrices` re-asserts both), so there is
    /// one rotation to do and no choice of which.
    pub fn rotate_group(
        &self,
        g: &FusedSegProj,
        x: &Tensor,
    ) -> candle_core::Result<Tensor> {
        let dims = x.dims();
        let d_in = *dims.last().expect("rank >= 1");
        if d_in != g.d_in {
            candle_core::bail!("{} expects d_in={}, got {d_in}", g.name, g.d_in);
        }
        let rows: usize = dims[..dims.len() - 1].iter().product();
        if rows != 1 {
            candle_core::bail!(
                "{}: rotation requested for {rows} vectors. The row loop belongs to \
                 model::group_forward, which shares it across the parts of the group.",
                g.name
            );
        }
        let x = x.to_dtype(DType::F16)?;
        let op = RotSegOp { rt: self, group: g };
        x.apply_op1_no_bwd(&op)
    }

    /// `y = W' xr` for one fused group. `out_dims` is the caller's shape; the
    /// last axis becomes the group's **total** width, which
    /// `model::group_forward` then narrows back into the parts.
    pub fn forward_rotated_seg(
        &self,
        g: &FusedSegProj,
        xr: &Tensor,
        out_dims: &[usize],
    ) -> candle_core::Result<Tensor> {
        // The only place the absent `CudaFunction` can surface, so it surfaces
        // as a message naming the group rather than as an `unwrap` in the
        // middle of a billed job.
        let f = self.f_matvec_seg.as_ref().ok_or_else(|| {
            candle_core::Error::msg(format!(
                "{}: fused group launched by a runtime built without tv_planes_seg_h \
                 (LLVQ_FUSE=0, or a layout that does not segment)",
                g.name
            ))
        })?;
        let out_shape = {
            let mut d = out_dims.to_vec();
            *d.last_mut().expect("rank >= 1") = g.d_out;
            Shape::from(d)
        };
        let op = FusedSegOp {
            rt: self,
            f,
            group: g,
            out_shape,
        };
        // No `to_dtype` on the result: the kernel already stored halves.
        xr.apply_op1_no_bwd(&op)
    }

    /// Upload a quantized g64 table, the embedding, for the runtime's two
    /// embedding kernels: int8 under [`EmbedMode::Q8`], int4 under
    /// [`EmbedMode::Q4`].
    ///
    /// Every assumption the kernels compile in is asserted here rather than
    /// trusted: the runtime's width, group 64, rows on word boundaries
    /// (`d % 4 == 0` at 8 bits, `d % 8 == 0` at 4), `vocab % 8 == 0` (whole
    /// warps, no bounds guard), and the staged activation within the card's
    /// shared memory.
    pub fn upload_embed(
        &self,
        t: &llvq_artifact::RawTensor,
    ) -> candle_core::Result<QuantEmbed> {
        let (Some(bits), Some((_, head))) = (self.emode.bits(), crate::fused::emb_kernel_names(self.emode))
        else {
            candle_core::bail!("runtime built without embedding kernels (LLVQ_EMBED=f16)");
        };
        if self.f_emb.is_none() {
            candle_core::bail!("runtime built without the {} kernels", self.emode.name());
        }
        let llvq_artifact::RawData::Quant(q) = &t.data else {
            candle_core::bail!("{}: not a quantized tensor", t.name);
        };
        if q.bits != bits || q.group != EMBED_GROUP {
            candle_core::bail!(
                "{}: int{} g{}, but the {} kernels hardcode int{bits} g{EMBED_GROUP}",
                t.name, q.bits, q.group, self.emode.name()
            );
        }
        if t.dims.len() != 2 {
            candle_core::bail!("{}: dims {:?}, an embedding is 2-D", t.name, t.dims);
        }
        let (vocab, d) = (t.dims[0], t.dims[1]);
        let per_word = 32 / bits as usize;
        if !d.is_multiple_of(per_word) {
            candle_core::bail!("{}: d={d} is not a multiple of {per_word}", t.name);
        }
        if !vocab.is_multiple_of(8) {
            candle_core::bail!("{}: vocab={vocab} is not a multiple of 8", t.name);
        }
        if d * 4 > self.shared_limit {
            candle_core::bail!(
                "{}: {head} asks for {} B of shared memory, the card offers {}",
                t.name, d * 4, self.shared_limit
            );
        }
        let gpr = d.div_ceil(EMBED_GROUP);
        if q.packed.len() != vocab * d / per_word * 4
            || q.scales.len() != vocab * gpr
            || q.biases.len() != q.scales.len()
        {
            candle_core::bail!(
                "{}: inconsistent payload ({} bytes, {} scales, {} biases for {vocab}×{d})",
                t.name, q.packed.len(), q.scales.len(), q.biases.len()
            );
        }
        let words: Vec<u32> = q
            .packed
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let bytes =
            q.packed.len() as u64 + (q.scales.len() as u64 + q.biases.len() as u64) * 2;
        Ok(QuantEmbed {
            vocab,
            d,
            gpr: gpr as u32,
            words: self.cuda.up_u32(&words).map_err(candle_core::Error::msg)?,
            scales: self.cuda.up_u16(&q.scales).map_err(candle_core::Error::msg)?,
            biases: self.cuda.up_u16(&q.biases).map_err(candle_core::Error::msg)?,
            bytes,
        })
    }

    /// Token ids `(.., l)` in u32 → embeddings `(.., l, d)` in f16, one
    /// gather launch for the whole call, rows dequantized on the device.
    pub fn embed(&self, q: &QuantEmbed, ids: &Tensor) -> candle_core::Result<Tensor> {
        let ids = ids.contiguous()?;
        let mut dims = ids.dims().to_vec();
        dims.push(q.d);
        let op = EmbedOp {
            rt: self,
            q,
            out_shape: Shape::from(dims),
        };
        ids.apply_op1_no_bwd(&op)
    }

    /// `logits = W · h` for hidden states `(.., d)` in f16, `W` the quantized
    /// table of [`EmbedMode`] (int4 in the served configs) — the tied
    /// `lm_head`, one matvec launch per row, all rows into one buffer.
    pub fn lm_head(&self, q: &QuantEmbed, h: &Tensor) -> candle_core::Result<Tensor> {
        let dims = h.dims();
        let d_in = *dims.last().expect("rank >= 1");
        if d_in != q.d {
            candle_core::bail!("lm_head {} expects d={}, got {d_in}", self.emode.name(), q.d);
        }
        let h = h.contiguous()?.to_dtype(DType::F16)?;
        let out_shape = {
            let mut d = dims.to_vec();
            *d.last_mut().expect("rank >= 1") = q.vocab;
            Shape::from(d)
        };
        let op = HeadOp {
            rt: self,
            q,
            out_shape,
        };
        h.apply_op1_no_bwd(&op)
    }
}
