//! The launch wrappers, one per kernel, with the AWQ and QTIP host helpers that fill their buffers. One file
//! because each wrapper holds the geometry and argument order its kernel expects, and nothing else.

use super::*;

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
pub(super) fn launch_planes(
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
pub(super) fn launch_tetra48(
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
pub(super) fn launch_tetra48_seg(
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
pub(super) fn launch_planes_seg(
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
pub(super) fn launch_planes12x(
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
pub(super) fn launch_nullk(
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
pub(super) fn launch_e1v(
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
pub(super) fn launch_golay70(
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
pub(super) fn awq_strides(d_in: usize) -> (usize, usize, usize) {
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
pub(super) fn awq_bytes(d_out: usize, d_in: usize) -> u64 {
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
pub(super) fn awq_quant_row(
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
pub(super) fn launch_qtip(
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
pub(super) fn qtip_pack_u32(buf: &[u16]) -> Vec<u32> {
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
pub(super) fn launch_awq(
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
