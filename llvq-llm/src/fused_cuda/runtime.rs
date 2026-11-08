//! Building the runtime: the NVRTC unit, the uploads, the kernel lookups,
//! and the rotation helpers the custom ops launch through.

use super::*;

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
        // Keyed on the same answer the lookup uses. It carries one accumulator
        // a row on top of the one-row kernel's 40 registers, so a spill here
        // is the number that says PREFILL_ROWS is too high — and a spill is a
        // hard stop, not a diagnostic.
        if let Some(n) = crate::fused::rows_kernel_name(model.layout) {
            spill_checked.push(n);
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

    /// The widest activation any projection takes.
    ///
    /// The rotation kernel stages the whole vector in shared memory, so this
    /// is the bound a caller checks against the card before uploading
    /// anything — `new` refuses past it rather than corrupting.
    pub fn max_d_in(&self) -> usize {
        self.max_d_in
    }
}
