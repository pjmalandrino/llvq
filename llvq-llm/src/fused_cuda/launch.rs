//! The layout launchers: one function per kernel entry point.

use super::*;

/// The Planes14 twin of `Cuda::launch_slot_h` — `tv_slot_h`'s argument list
/// minus the bases array, which Planes14 does not have. Local to this crate
/// because `llvq-cuda` belongs to another lot; same grid (one warp per row,
/// whole blocks only, no bounds guard in the kernel), same generic `y` so
/// candle's `CudaSlice<half::f16>` can be handed over.
#[allow(clippy::too_many_arguments)]
pub(super) fn launch_planes_h<T: candle_core::cuda_backend::cudarc::driver::DeviceRepr>(
    cuda: &llvq_cuda::gpu::Cuda,
    f: &CudaFunction,
    words: &CudaSlice<u32>,
    tab: &CudaSlice<u32>,
    gscale: &CudaSlice<f32>,
    rscale: &CudaSlice<f32>,
    tail: &CudaSlice<u16>,
    x: &CudaSlice<f32>,
    y: &mut CudaSlice<T>,
    nblocks: u32,
    tail_w: u32,
    d_out: u32,
    threads: u32,
    shared: u32,
) -> Result<(), String> {
    assert_eq!(d_out % (threads / 32), 0, "rows must fill whole blocks");
    let cfg = LaunchConfig {
        grid_dim: (d_out * 32 / threads, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: shared,
    };
    let mut b = cuda.stream().launch_builder(f);
    b.arg(words).arg(tab).arg(gscale).arg(rscale).arg(tail).arg(x).arg(y)
        .arg(&nblocks).arg(&tail_w);
    unsafe { b.launch(cfg) }.map_err(|e| format!("tv_planes_h: {e}"))?;
    Ok(())
}

/// `tv_q4_hs(wq, scales, biases, x, y, d_in, gpr, x_off)`: the int4 matvec
/// with its activation staged as f16 bits, `d_in * 2` bytes of dynamic shared
/// memory whatever the record's own staging, `x` read from `x_off` elements.
/// Same grid as [`launch_q4_h`].
pub(super) fn launch_q4_hs<
    X: candle_core::cuda_backend::cudarc::driver::DeviceRepr,
    T: candle_core::cuda_backend::cudarc::driver::DeviceRepr,
>(
    cuda: &llvq_cuda::gpu::Cuda,
    f: &CudaFunction,
    p: &FusedInt4Proj,
    x: &CudaSlice<X>,
    y: &mut CudaSlice<T>,
    x_off: u32,
    threads: u32,
) -> Result<(), String> {
    let d_out = p.d_out as u32;
    assert_eq!(d_out % (threads / 32), 0, "rows must fill whole blocks");
    let d_in = p.d_in as u32;
    let cfg = LaunchConfig {
        grid_dim: (d_out * 32 / threads, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: llvq_cuda::wide::Int4Staging::F16.bytes(p.d_in) as u32,
    };
    let mut b = cuda.stream().launch_builder(f);
    b.arg(&p.wq)
        .arg(&p.scales)
        .arg(&p.biases)
        .arg(x)
        .arg(y)
        .arg(&d_in)
        .arg(&p.gpr)
        .arg(&x_off);
    unsafe { b.launch(cfg) }.map_err(|e| format!("tv_q4_hs: {e}"))?;
    Ok(())
}

/// `tv_q4_h(wq, scales, biases, x, y, d_in, gpr)` — the served int4 matvec,
/// or `tv_q4_hh` with the same arguments and `x` in f16; `f` decides which.
///
/// One warp a row and 256-thread blocks, like every projection kernel here,
/// and NO tile: the activation is staged whole, so `shared` comes off the
/// projection rather than off `Tile`.
pub(super) fn launch_q4_h<
    X: candle_core::cuda_backend::cudarc::driver::DeviceRepr,
    T: candle_core::cuda_backend::cudarc::driver::DeviceRepr,
>(
    cuda: &llvq_cuda::gpu::Cuda,
    f: &CudaFunction,
    p: &FusedInt4Proj,
    x: &CudaSlice<X>,
    y: &mut CudaSlice<T>,
    threads: u32,
) -> Result<(), String> {
    let d_out = p.d_out as u32;
    assert_eq!(d_out % (threads / 32), 0, "rows must fill whole blocks");
    let cfg = LaunchConfig {
        grid_dim: (d_out * 32 / threads, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: p.shared,
    };
    let d_in = p.d_in as u32;
    let mut b = cuda.stream().launch_builder(f);
    b.arg(&p.wq)
        .arg(&p.scales)
        .arg(&p.biases)
        .arg(x)
        .arg(y)
        .arg(&d_in)
        .arg(&p.gpr);
    unsafe { b.launch(cfg) }.map_err(|e| format!("tv_q4_h: {e}"))?;
    Ok(())
}

/// `tv_tetra48_rows_h(...)` — the prefill launch, `n_rows` activation rows.
///
/// Same grid as the one-row kernel: one warp an output row, eight rows a
/// block. What changes is the shared request — `PREFILL_ROWS` rows of tile
/// instead of one — and the three sizes the kernel needs to address a batch.
#[allow(clippy::too_many_arguments)]
pub(super) fn launch_tetra48_rows_h(
    cuda: &llvq_cuda::gpu::Cuda,
    f: &CudaFunction,
    words: &CudaSlice<u32>,
    stride_u32: u32,
    t: &TetraTabs,
    gscale: &CudaSlice<f32>,
    rscale: &CudaSlice<f32>,
    tail: &CudaSlice<u16>,
    x: &CudaSlice<f32>,
    y: &mut CudaSlice<f16>,
    nblocks: u32,
    tail_w: u32,
    n_rows: u32,
    d_in: u32,
    d_out: u32,
    threads: u32,
    shared: u32,
    // Rows the unit was COMPILED at — `FusedRuntime::prefill_rows`, never
    // `tile::PREFILL_ROWS`.
    rows: usize,
) -> Result<(), String> {
    assert_eq!(d_out % (threads / 32), 0, "rows must fill whole blocks");
    // 🕳️ This asserted against `tile::PREFILL_ROWS`, the SERVED constant, on a
    // launcher whose kernel is compiled at whatever the runtime resolved.
    // Under `LLVQ_PREFILL=8:64` the unit compiled at eight rows, the model
    // chunked at eight, and this line refused eight "for a kernel compiled at
    // 4" — a served constant vetoing a measurement of itself. Found on the
    // first card sweep, 2026-09-11, $0.06.
    //
    // `rows` is now the pair the CALLER compiled, so the two cannot disagree:
    // there is one row count in this launch, and it came from the same
    // `Prefill` as the `#define` and the `shared` below.
    assert!(
        n_rows > 0 && n_rows as usize <= rows,
        "{n_rows} rows for a kernel compiled at {rows}"
    );
    let cfg = LaunchConfig {
        grid_dim: (d_out * 32 / threads, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: shared,
    };
    let mut b = cuda.stream().launch_builder(f);
    b.arg(words)
        .arg(&stride_u32)
        .arg(&t.rows)
        .arg(&t.prefixes)
        .arg(&t.branches)
        .arg(&t.suffixes)
        .arg(gscale)
        .arg(&t.invnorm)
        .arg(rscale)
        .arg(tail)
        .arg(x)
        .arg(y)
        .arg(&nblocks)
        .arg(&tail_w)
        .arg(&n_rows)
        .arg(&d_in)
        .arg(&d_out);
    unsafe { b.launch(cfg) }.map_err(|e| format!("tv_tetra48_rows_h: {e}"))?;
    Ok(())
}

// Fifteen, and every one of them is a pointer the kernel takes: bundling them
// into a struct would put a second description of the argument list beside the
// `extern "C"` one, which is the drift this file spends its comments avoiding.
#[allow(clippy::too_many_arguments)]
pub(super) fn launch_tetra48_h<T: candle_core::cuda_backend::cudarc::driver::DeviceRepr>(
    cuda: &llvq_cuda::gpu::Cuda,
    f: &CudaFunction,
    words: &CudaSlice<u32>,
    stride_u32: u32,
    t: &TetraTabs,
    gscale: &CudaSlice<f32>,
    rscale: &CudaSlice<f32>,
    tail: &CudaSlice<u16>,
    x: &CudaSlice<f32>,
    y: &mut CudaSlice<T>,
    nblocks: u32,
    tail_w: u32,
    d_out: u32,
    threads: u32,
    shared: u32,
) -> Result<(), String> {
    assert_eq!(d_out % (threads / 32), 0, "rows must fill whole blocks");
    let cfg = LaunchConfig {
        grid_dim: (d_out * 32 / threads, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: shared,
    };
    let mut b = cuda.stream().launch_builder(f);
    b.arg(words)
        .arg(&stride_u32)
        .arg(&t.rows)
        .arg(&t.prefixes)
        .arg(&t.branches)
        .arg(&t.suffixes)
        .arg(gscale)
        .arg(&t.invnorm)
        .arg(rscale)
        .arg(tail)
        .arg(x)
        .arg(y)
        .arg(&nblocks)
        .arg(&tail_w);
    unsafe { b.launch(cfg) }.map_err(|e| format!("tv_tetra48_h: {e}"))?;
    Ok(())
}

/// The segmented twin of [`launch_planes_h`] — same grid, one extra array.
///
/// `gs_off` sits between `gscale` and `rscale`, which is `tv_planes_seg_h`'s
/// declaration order. Note what the types buy here, against the remark
/// [`launch_planes12x_h`] makes: `gscale` is `CudaSlice<f32>` and `gs_off` is
/// `CudaSlice<u32>`, so transposing *those two* does not compile. The pair that
/// still could is `gscale`/`rscale`, and that hazard predates this lot.
///
/// The grid is `tv_planes_h`'s, unchanged, over the **total** `d_out`: on the
/// published 4B, q+k+v becomes 768 CTAs where it was 512+128+128, and gate+up
/// 2432 where it was 1216+1216. Same CTAs, one launch.
#[allow(clippy::too_many_arguments)]
pub(super) fn launch_planes_seg_h<T: candle_core::cuda_backend::cudarc::driver::DeviceRepr>(
    cuda: &llvq_cuda::gpu::Cuda,
    f: &CudaFunction,
    words: &CudaSlice<u32>,
    tab: &CudaSlice<u32>,
    gscale: &CudaSlice<f32>,
    gs_off: &CudaSlice<u32>,
    rscale: &CudaSlice<f32>,
    tail: &CudaSlice<u16>,
    x: &CudaSlice<f32>,
    y: &mut CudaSlice<T>,
    nblocks: u32,
    tail_w: u32,
    d_out: u32,
    threads: u32,
    shared: u32,
) -> Result<(), String> {
    assert_eq!(d_out % (threads / 32), 0, "rows must fill whole blocks");
    let cfg = LaunchConfig {
        grid_dim: (d_out * 32 / threads, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: shared,
    };
    let mut b = cuda.stream().launch_builder(f);
    b.arg(words).arg(tab).arg(gscale).arg(gs_off).arg(rscale).arg(tail).arg(x).arg(y)
        .arg(&nblocks).arg(&tail_w);
    unsafe { b.launch(cfg) }.map_err(|e| format!("tv_planes_seg_h: {e}"))?;
    Ok(())
}

/// The Planes12x twin of [`launch_planes_h`] — same grid, four stream arrays
/// instead of one.
///
/// 🚨 **Read the grid, because it is the whole design decision.** It is
/// `tv_planes_h`'s grid, unchanged: `d_out·32/threads` CTAs, one warp per
/// output row, and *no* exception region. planesbench's `tv_planes12x` adds
/// `ceil(n_exc/8)` CTAs that `atomicAdd` into a `y` it memsets first; this
/// path instead hands each row its own slice of the exception table
/// (`row_exc`) so the corrections happen inside the row's warp. Consequences,
/// in the order they matter here:
///
///  * **`y` is not zeroed, and must not be.** `FusedOp::cuda_fwd` allocates
///    it uninitialised. That is sound for exactly the reason it is sound for
///    `tv_planes_h` — the grid is exact and every row is *stored*, never
///    accumulated into — and it stays sound here only because no CTA outside
///    a row's own warp writes that row. If anyone ever reintroduces an
///    exception region, the allocation upstream has to become a memset in the
///    same commit.
///  * **no atomic, so no `atomicAdd` on `__half`.** The accumulation is f32
///    from the first block to the last correction; the single narrowing to
///    binary16 is the final store, where candle would have narrowed anyway.
///    An `atomicAdd(__half*)` would have rounded every partial sum instead,
///    which is not the arithmetic the `Planes14` arm this replaces performs.
///
/// `arrays` is `[words, exc_idx, exc_words, row_exc]` in kernel order, taken
/// as one slice so a caller cannot silently transpose two `CudaSlice<u32>`
/// arguments of identical type — the one mistake here that compiles.
#[allow(clippy::too_many_arguments)]
pub(super) fn launch_planes12x_h<T: candle_core::cuda_backend::cudarc::driver::DeviceRepr>(
    cuda: &llvq_cuda::gpu::Cuda,
    f: &CudaFunction,
    arrays: &[&CudaSlice<u32>; 4],
    tab: &CudaSlice<u32>,
    gscale: &CudaSlice<f32>,
    rscale: &CudaSlice<f32>,
    tail: &CudaSlice<u16>,
    x: &CudaSlice<f32>,
    y: &mut CudaSlice<T>,
    nblocks: u32,
    tail_w: u32,
    d_out: u32,
    threads: u32,
    shared: u32,
) -> Result<(), String> {
    assert_eq!(d_out % (threads / 32), 0, "rows must fill whole blocks");
    let cfg = LaunchConfig {
        grid_dim: (d_out * 32 / threads, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: shared,
    };
    let mut b = cuda.stream().launch_builder(f);
    b.arg(arrays[0]).arg(arrays[1]).arg(arrays[2]).arg(arrays[3])
        .arg(tab).arg(gscale).arg(rscale).arg(tail).arg(x).arg(y)
        .arg(&nblocks).arg(&tail_w);
    unsafe { b.launch(cfg) }.map_err(|e| format!("tv_planes12x_h: {e}"))?;
    Ok(())
}

/// The Golay70 twin of [`launch_planes12x_h`] — same grid (one warp per
/// output row, no exception region, no memset, no atomic: the row-sliced
/// correction design of `tv_planes12x_h`, see its launcher's 🚨 note, which
/// holds here word for word), with the two Golay70 constant tables added
/// between the stream arrays and the shared class table, in the kernel's
/// argument order: `tv_golay70_h(words, exc_idx, exc_words, row_exc, cwtab,
/// gtab, tab, gscale, rscale, tail, x, y, nblocks, tail_w)`.
///
/// `arrays` is `[words, exc_idx, exc_words, row_exc]`, one slice for the same
/// reason as Planes12x: four `CudaSlice<u32>` of identical type, and a silent
/// transposition is the one mistake here that compiles.
#[allow(clippy::too_many_arguments)]
pub(super) fn launch_golay70_h<T: candle_core::cuda_backend::cudarc::driver::DeviceRepr>(
    cuda: &llvq_cuda::gpu::Cuda,
    f: &CudaFunction,
    arrays: &[&CudaSlice<u32>; 4],
    cwtab: &CudaSlice<u32>,
    gtab: &CudaSlice<u32>,
    tab: &CudaSlice<u32>,
    gscale: &CudaSlice<f32>,
    rscale: &CudaSlice<f32>,
    tail: &CudaSlice<u16>,
    x: &CudaSlice<f32>,
    y: &mut CudaSlice<T>,
    nblocks: u32,
    tail_w: u32,
    d_out: u32,
    threads: u32,
    shared: u32,
) -> Result<(), String> {
    assert_eq!(d_out % (threads / 32), 0, "rows must fill whole blocks");
    let cfg = LaunchConfig {
        grid_dim: (d_out * 32 / threads, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: shared,
    };
    let mut b = cuda.stream().launch_builder(f);
    b.arg(arrays[0]).arg(arrays[1]).arg(arrays[2]).arg(arrays[3])
        .arg(cwtab).arg(gtab)
        .arg(tab).arg(gscale).arg(rscale).arg(tail).arg(x).arg(y)
        .arg(&nblocks).arg(&tail_w);
    unsafe { b.launch(cfg) }.map_err(|e| format!("tv_golay70_h: {e}"))?;
    Ok(())
}
