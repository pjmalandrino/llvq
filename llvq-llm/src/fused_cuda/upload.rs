//! Host to device: rotation tables, int4 records with their load-time
//! parity checks, lattice matrices and segmented groups.

use super::*;

pub(super) fn upload_rotation(
    cuda: &llvq_cuda::gpu::Cuda,
    t: &RotationTables,
    kernel: llvq_cuda::wide::RotKernel,
) -> candle_core::Result<RotBuffers> {
    Ok(RotBuffers {
        signbits: cuda.up_u32(&t.signbits).map_err(candle_core::Error::msg)?,
        small: cuda.up_f32(&t.small).map_err(candle_core::Error::msg)?,
        n: t.n as u32,
        m: t.m as u32,
        k: t.k as u32,
        inv: t.inv,
        // The widest block the driver allows, clamped to a warp at the bottom
        // so a narrow width still fills one.
        threads: (t.n as u32).next_power_of_two().clamp(32, 1024),
        kernel,
    })
}

/// `tv_tetra48_h(words, row_stride_u32, rows, prefixes, branches, suffixes,
/// gscale, invnorm, rscale, tail, x, y, nblocks, tail_w)`.
///
/// The grid is `launch_planes_h`'s, unchanged — one warp per row, eight rows a
/// block — so a residency comparison between the two served layouts is like
/// for like. What differs is the second argument: `row_stride_u32`, which
/// Planes14 does not have because it addresses blocks flat. Passing the wrong
/// stride there does not fail; it reads every row after the first at a shifted
/// phase and returns plausible, wrong weights, which is why it travels with
/// the buffer in [`DeviceStream::Tetra48`] rather than being re-derived here.
#[allow(clippy::too_many_arguments)]
/// One int4 record onto the card: three arrays, no transcode.
///
/// The kernel reads these bytes as they sit on disk — that is what
/// distinguishes a stored-weight format from a lattice one — so the only work
/// here is the byte-to-u32 view and four checks the kernel's own header asks
/// the host to make.
pub(super) fn upload_int4(
    cuda: &llvq_cuda::gpu::Cuda,
    q: &crate::fused::FusedInt4,
    shared_def: usize,
    shared_optin: usize,
    layout: crate::fused::Int4Layout,
) -> candle_core::Result<FusedInt4Proj> {
    // The four the kernel names, each because the arithmetic below it breaks
    // silently otherwise rather than faulting.
    if q.group != 128 {
        candle_core::bail!("{}: group {} — `tv_q4_h` hard-codes 128", q.name, q.group);
    }
    if !q.d_in.is_multiple_of(8) {
        candle_core::bail!(
            "{}: d_in {} is not a multiple of 8, so a row does not start on a u32 \\
             and every row after the first would read at a shifted nibble",
            q.name,
            q.d_in
        );
    }
    if !q.d_out.is_multiple_of(THREADS as usize / 32) {
        candle_core::bail!(
            "{}: d_out {} does not fill whole blocks of {} rows, and the kernel \\
             carries no bounds guard",
            q.name,
            q.d_out,
            THREADS / 32
        );
    }
    let gpr = q.d_in.div_ceil(q.group);
    if q.scales.len() != q.d_out * gpr || q.biases.len() != q.d_out * gpr {
        candle_core::bail!(
            "{}: {} scales and {} biases for {} rows × {gpr} groups",
            q.name,
            q.scales.len(),
            q.biases.len(),
            q.d_out
        );
    }
    if q.packed.len() != q.d_out * q.d_in / 2 {
        candle_core::bail!(
            "{}: {} packed bytes for {} × {} nibbles",
            q.name,
            q.packed.len(),
            q.d_out,
            q.d_in
        );
    }
    // The whole activation, not a tile, and `d_in` can be the intermediate
    // width (an int4 `down_proj`). F32 staging wherever it fits the card's
    // ceiling, which is the opt-in one the entry points were loaded with
    // (`llvq_cuda::wide::int4_entry_bytes`); F16 (`tv_q4_hs`) only past it.
    let shared_limit = llvq_cuda::shared::ceiling(shared_def, shared_optin);
    let staging = llvq_cuda::wide::int4_staging(q.d_in, shared_def, shared_optin)
        .unwrap_or(llvq_cuda::wide::Int4Staging::F16);
    let shared = staging.bytes(q.d_in);
    if shared > shared_limit {
        candle_core::bail!(
            "{}: staging {} values needs {shared} B of shared, the card allows {shared_limit}",
            q.name,
            q.d_in
        );
    }
    // Little-endian, so byte `b` of the stream is bits `8b` of the word and
    // nibble `i` stays at `4 · (i % 8)` — which is what `(p >> 4k) & 0xf`
    // reads. The disk order is low-nibble-first; a big-endian view here would
    // transpose every pair of columns and produce plausible, wrong weights.
    let words: Vec<u32> = q
        .packed
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    if words.len() * 4 != q.packed.len() {
        candle_core::bail!(
            "{}: {} packed bytes is not a whole number of u32 words",
            q.name,
            q.packed.len()
        );
    }
    // Tiled: the same words, permuted inside each row on the host, so the
    // card holds this record once, in the order 4E v9 and `tv_q4_hht` read.
    // Only F32-staged records: `tv_q4_hs` has no tiled form.
    let words = match layout {
        crate::fused::Int4Layout::Served => words,
        crate::fused::Int4Layout::Tiled => {
            if !crate::fused::int4_tileable(q.d_in) {
                candle_core::bail!(
                    "{}: d_in {} is not a multiple of 32, so its rows are not whole 16-byte loads \
                     and the tiled layout of 4E v9 cannot hold it",
                    q.name,
                    q.d_in
                );
            }
            if staging != llvq_cuda::wide::Int4Staging::F32 {
                candle_core::bail!(
                    "{}: d_in {} is staged in f16 on this card (tv_q4_hs), which has no tiled form; \
                     the tiled layout of 4E v9 is refused here",
                    q.name,
                    q.d_in
                );
            }
            llvq_cuda::pers::q4_tile_words(&words, q.d_out, q.d_in / 8)
                .map_err(|e| candle_core::Error::msg(format!("{}: {e}", q.name)))?
        }
    };
    Ok(FusedInt4Proj {
        name: q.name.clone(),
        d_out: q.d_out,
        d_in: q.d_in,
        gpr: gpr as u32,
        wq: cuda.up_u32(&words).map_err(candle_core::Error::msg)?,
        scales: cuda.up_u16(&q.scales).map_err(candle_core::Error::msg)?,
        biases: cuda.up_u16(&q.biases).map_err(candle_core::Error::msg)?,
        shared: shared as u32,
        staging,
        layout,
        bytes: q.bytes,
    })
}

/// `tv_q4_hh` against `tv_q4_h` on one real record, on the card, at load.
///
/// The f16 entry point exists only to save a cast launch, so its one claim is
/// that it returns `tv_q4_h`'s bits for the f32 widening of the same input.
/// No host test can run a warp kernel, so the claim is checked here, before
/// any token: a pseudo-random activation over several binades, both signs
/// and exact zeros, given to `tv_q4_hh` as f16 and to `tv_q4_h` as its exact
/// f32 widening. Any differing row stops the load and names the record.
pub(super) fn check_int4_half_parity(
    cuda: &llvq_cuda::gpu::Cuda,
    f_f32: &CudaFunction,
    f_f16: &CudaFunction,
    p: &FusedInt4Proj,
    n_f32: &str,
    n_f16: &str,
) -> Result<(), String> {
    let xh = parity_activation(p.d_in);
    let xf: Vec<f32> = xh.iter().map(|&b| f16::from_bits(b).to_f32()).collect();
    let dxh = cuda.up_u16(&xh)?;
    let dxf = cuda.up_f32(&xf)?;
    let mut ya = cuda.up_u16(&vec![0u16; p.d_out])?;
    let mut yb = cuda.up_u16(&vec![0xffffu16; p.d_out])?;
    launch_q4_h(cuda, f_f32, p, &dxf, &mut ya, THREADS)?;
    launch_q4_h(cuda, f_f16, p, &dxh, &mut yb, THREADS)?;
    let (a, b) = (cuda.down_u16(&ya)?, cuda.down_u16(&yb)?);
    if let Some(r) = (0..p.d_out).find(|&r| a[r] != b[r]) {
        return Err(format!(
            "{}: {n_f16} disagrees with {n_f32} at row {r} ({:#06x} against {:#06x}); \
             the f16 entry point is not the same matvec",
            p.name, b[r], a[r]
        ));
    }
    println!(
        "int4: {n_f16} equals {n_f32} bit for bit on {} ({} rows, d_in {})",
        p.name, p.d_out, p.d_in
    );
    Ok(())
}

/// `tv_q4_hht` on a tiled record against `tv_q4_hh` on the same record in
/// the file's order, on the card, at load: the claim of `tv_q4_tile.cu`
/// (only the addresses move) on a real record, every row. The served copy is
/// uploaded for the check and freed when it returns, so the card holds the
/// int4 bytes once from the first token on. Any differing row stops the load.
pub(super) fn check_int4_tile_parity(
    cuda: &llvq_cuda::gpu::Cuda,
    f_hh: &CudaFunction,
    f_hht: &CudaFunction,
    q: &crate::fused::FusedInt4,
    tiled: &FusedInt4Proj,
    shared_def: usize,
    shared_optin: usize,
) -> Result<(), String> {
    if tiled.layout != crate::fused::Int4Layout::Tiled {
        return Err(format!("{}: the tile check was handed a served-order record", tiled.name));
    }
    let served = upload_int4(cuda, q, shared_def, shared_optin, crate::fused::Int4Layout::Served)
        .map_err(|e| e.to_string())?;
    let xh = parity_activation(q.d_in);
    let dxh = cuda.up_u16(&xh)?;
    let mut ya = cuda.up_u16(&vec![0u16; q.d_out])?;
    let mut yb = cuda.up_u16(&vec![0xffffu16; q.d_out])?;
    launch_q4_h(cuda, f_hh, &served, &dxh, &mut ya, THREADS)?;
    launch_q4_h(cuda, f_hht, tiled, &dxh, &mut yb, THREADS)?;
    let (a, b) = (cuda.down_u16(&ya)?, cuda.down_u16(&yb)?);
    if let Some(r) = (0..q.d_out).find(|&r| a[r] != b[r]) {
        return Err(format!(
            "{}: tv_q4_hht on the tiled words disagrees with tv_q4_hh on the file's order at row {r} \
             ({:#06x} against {:#06x}); the tiled layout is not the same matvec",
            q.name, b[r], a[r]
        ));
    }
    println!(
        "int4: tv_q4_hht on the tiled words equals tv_q4_hh on the file's order bit for bit on {} \
         ({} rows, d_in {}); the file-order copy is freed",
        q.name, q.d_out, q.d_in
    );
    Ok(())
}

/// The activation the load-time int4 checks feed every entry point: a
/// pseudo-random draw over several binades, both signs and exact zeros, as f16
/// bits. One function, so the two checks below read the same vector.
fn parity_activation(d_in: usize) -> Vec<u16> {
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    (0..d_in)
        .map(|i| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let u = (state >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
            let v = if i % 17 == 0 { 0.0 } else { u * (1u32 << (i % 7)) as f32 };
            f16::from_f32(v).to_bits()
        })
        .collect()
}

/// `tv_q4_hs` against `tv_q4_hh` on one real record, on the card, at load.
///
/// Reached only by a model with an F16-staged record (none is served). The
/// record is an F32-staged one, where both kernels fit: `tv_q4_hh` with the
/// record's own `d_in * 4`, `tv_q4_hs` with `d_in * 2`. Any differing row
/// stops the load and names the record.
pub(super) fn check_int4_wide_parity(
    cuda: &llvq_cuda::gpu::Cuda,
    f_hh: &CudaFunction,
    f_hs: &CudaFunction,
    p: &FusedInt4Proj,
) -> Result<(), String> {
    let xh = parity_activation(p.d_in);
    let dxh = cuda.up_u16(&xh)?;
    let mut ya = cuda.up_u16(&vec![0u16; p.d_out])?;
    let mut yb = cuda.up_u16(&vec![0xffffu16; p.d_out])?;
    launch_q4_h(cuda, f_hh, p, &dxh, &mut ya, THREADS)?;
    launch_q4_hs(cuda, f_hs, p, &dxh, &mut yb, 0, THREADS)?;
    let (a, b) = (cuda.down_u16(&ya)?, cuda.down_u16(&yb)?);
    if let Some(r) = (0..p.d_out).find(|&r| a[r] != b[r]) {
        return Err(format!(
            "{}: tv_q4_hs disagrees with tv_q4_hh at row {r} ({:#06x} against {:#06x}); \
             the f16-staged entry point is not the same matvec",
            p.name, b[r], a[r]
        ));
    }
    println!(
        "int4: tv_q4_hs equals tv_q4_hh bit for bit on {} ({} rows, d_in {})",
        p.name, p.d_out, p.d_in
    );
    Ok(())
}

/// Measurement only, `false` everywhere else: run [`FusedRuntime::int4_wide_check`]
/// right after the load. `bin/fusedrun` sets it under `LLVQ_INT4_WIDE_CHECK=1`,
/// a measurement mode refused beside `LLVQ_CONFIG`.
pub(super) static INT4_WIDE_CHECK: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Set [`INT4_WIDE_CHECK`].
pub fn set_int4_wide_check(on: bool) {
    INT4_WIDE_CHECK.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub(super) fn upload_matrix(
    cuda: &llvq_cuda::gpu::Cuda,
    m: &FusedMatrix,
    layout: FusedLayout,
) -> candle_core::Result<FusedProj> {
    if !m.d_out.is_multiple_of(8) {
        // `tv_slot` has no bounds guard: a `return` before `__syncthreads()`
        // deadlocks, and it would break the full-warp mask the reduction
        // relies on. The grid is exact, so the host asserts instead.
        candle_core::bail!("{}: d_out={} is not a multiple of 8", m.name, m.d_out);
    }
    // A stream in the wrong layout would be read by the wrong kernel into
    // finite, plausible, wrong numbers — refused here, matrix by matrix,
    // rather than trusted to have been built consistently.
    let stream = match (&m.stream, layout) {
        (HostStream::Slot32 { words, bases }, FusedLayout::Slot32) => DeviceStream::Slot32 {
            words: cuda.up_u32(words).map_err(candle_core::Error::msg)?,
            bases: cuda.up_u32(bases).map_err(candle_core::Error::msg)?,
        },
        (HostStream::Tetra48 { words, stride_u32 }, FusedLayout::Tetra48) => {
            DeviceStream::Tetra48 {
                words: cuda.up_u32(words).map_err(candle_core::Error::msg)?,
                stride_u32: *stride_u32,
            }
        }
        (HostStream::Planes14 { words }, FusedLayout::Planes14) => DeviceStream::Planes14 {
            words: cuda.up_u32(words).map_err(candle_core::Error::msg)?,
        },
        (
            HostStream::Planes12x { words, exc_idx, exc_words, row_exc },
            FusedLayout::Planes12x,
        ) => {
            // cudarc refuses a zero-length upload. A matrix with no 5-level
            // block still needs a bound pointer, so it gets a one-word dummy
            // the kernel never dereferences: every row's slice is empty, so
            // `planes12x_row_correction`'s loop never runs. `exc_words` is
            // never empty — `pack_plane_bytes` appends the read-window pad —
            // and `row_exc` has `d_out + 1` entries, so only `exc_idx` needs
            // this.
            let dummy = [0u32];
            let idx: &[u32] = if exc_idx.is_empty() { &dummy } else { exc_idx };
            DeviceStream::Planes12x {
                words: cuda.up_u32(words).map_err(candle_core::Error::msg)?,
                exc_idx: cuda.up_u32(idx).map_err(candle_core::Error::msg)?,
                exc_words: cuda.up_u32(exc_words).map_err(candle_core::Error::msg)?,
                row_exc: cuda.up_u32(row_exc).map_err(candle_core::Error::msg)?,
            }
        }
        (
            HostStream::Golay70 { words, exc_idx, exc_words, row_exc },
            FusedLayout::Golay70,
        ) => {
            // Same zero-length rule as Planes12x: a matrix without any
            // exception block gets a one-word dummy the kernel never
            // dereferences — every row's slice is empty, so
            // `golay70_row_correction`'s loop never runs. `exc_words` is
            // never empty (`pack_plane_bytes` appends the read-window pad)
            // and `row_exc` has `d_out + 1` entries.
            let dummy = [0u32];
            let idx: &[u32] = if exc_idx.is_empty() { &dummy } else { exc_idx };
            DeviceStream::Golay70 {
                words: cuda.up_u32(words).map_err(candle_core::Error::msg)?,
                exc_idx: cuda.up_u32(idx).map_err(candle_core::Error::msg)?,
                exc_words: cuda.up_u32(exc_words).map_err(candle_core::Error::msg)?,
                row_exc: cuda.up_u32(row_exc).map_err(candle_core::Error::msg)?,
            }
        }
        _ => candle_core::bail!(
            "{}: host stream and runtime layout ({}) disagree",
            m.name,
            layout.name()
        ),
    };
    Ok(FusedProj {
        name: m.name.clone(),
        d_out: m.d_out,
        d_in: m.d_in,
        nblocks: m.nblocks as u32,
        tail_w: m.tail_w as u32,
        stream,
        gscale: cuda.up_f32(&m.gscale).map_err(candle_core::Error::msg)?,
        rscale: cuda.up_f32(&m.rscale).map_err(candle_core::Error::msg)?,
        // Binary16 bits, narrowed on the host by `fused::tail_f16_bits`.
        // cudarc refuses a zero-length upload, so a matrix whose `d_in` is a
        // multiple of 24 gets a one-element dummy the kernel never reads
        // (`tail_w == 0` makes `tail_dot_h`'s loop empty) — the same shape the
        // `Planes12x` exception index uses above.
        tail: cuda
            .up_u16(if m.tail.is_empty() { &[0u16] } else { &m.tail })
            .map_err(candle_core::Error::msg)?,
        rotation: m.rotation,
    })
}

/// Upload one fused group — the row concatenation, plus the offset table that
/// is the only thing it adds.
///
/// Every refusal below fires **before** a byte reaches the card, and every one
/// of them guards a failure that is finite, plausible and wrong rather than a
/// crash. The `gs_off` sweep in particular is the only guard possible on that
/// table: an entry past `gscale` would read arbitrary floats downstream of it
/// without ever leaving an arena allocator's allocation, so nothing on the card
/// would notice. It costs one pass over 25,600 `u32` a group, once, at load.
pub(super) fn upload_group(
    cuda: &llvq_cuda::gpu::Cuda,
    g: &FusedGroup,
    layout: FusedLayout,
) -> candle_core::Result<FusedSegProj> {
    if seg_kernel_name(layout).is_none() {
        candle_core::bail!(
            "{}: fused group on layout {}, only planes14 segments",
            g.key,
            layout.name()
        );
    }
    // Per part **and** on the total, the `fused::segment_matrices` rule: the
    // total alone would let an individually ragged part through on a lucky sum,
    // and the unfused control arm could then not be launched at all.
    if !g.d_out.is_multiple_of(8) {
        candle_core::bail!("{}: d_out={} is not a multiple of 8", g.key, g.d_out);
    }
    for p in &g.parts {
        if !p.d_out.is_multiple_of(8) {
            candle_core::bail!("{}: {} has d_out={}, not a multiple of 8", g.key, p.name, p.d_out);
        }
    }
    if g.gs_off.len() != g.d_out
        || g.gscale.len() != 2 * g.parts.len()
        || g.rscale.len() != g.d_out
        || g.tail.len() != g.d_out * g.tail_w
    {
        candle_core::bail!(
            "{}: {} gs_off, {} centroids, {} scales, {} tail values for {} rows \
             of {} and {} parts",
            g.key,
            g.gs_off.len(),
            g.gscale.len(),
            g.rscale.len(),
            g.tail.len(),
            g.d_out,
            g.tail_w,
            g.parts.len()
        );
    }
    if let Some(bad) = g.gs_off.iter().position(|&o| o as usize + 1 >= g.gscale.len()) {
        candle_core::bail!(
            "{}: gs_off[{bad}]={} outside the table of {} centroids",
            g.key,
            g.gs_off[bad],
            g.gscale.len()
        );
    }
    let HostStream::Planes14 { words } = &g.stream else {
        candle_core::bail!("{}: group stream that is not Planes14", g.key);
    };
    Ok(FusedSegProj {
        name: g.key.clone(),
        d_out: g.d_out,
        d_in: g.d_in,
        nblocks: g.nblocks as u32,
        tail_w: g.tail_w as u32,
        words: cuda.up_u32(words).map_err(candle_core::Error::msg)?,
        gscale: cuda.up_f32(&g.gscale).map_err(candle_core::Error::msg)?,
        gs_off: cuda.up_u32(&g.gs_off).map_err(candle_core::Error::msg)?,
        rscale: cuda.up_f32(&g.rscale).map_err(candle_core::Error::msg)?,
        // Same zero-length rule as `upload_matrix`: cudarc refuses an empty
        // upload, so a group whose `d_in` is a multiple of 24 gets a one-element
        // dummy the kernel never reads (`tail_w == 0` makes `tail_dot_h`'s loop
        // empty).
        tail: cuda
            .up_u16(if g.tail.is_empty() { &[0u16] } else { &g.tail })
            .map_err(candle_core::Error::msg)?,
        rotation: g.rotation,
        part_names: g.parts.iter().map(|p| p.name.clone()).collect(),
    })
}
