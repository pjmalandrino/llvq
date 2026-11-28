//! The device views 4E reads: tensor addresses, the Tetra and int4 views of a
//! projection, and the MLP arguments. One file because they are the free
//! functions the install path and the eager path share.

use candle_core::cuda_backend::cudarc::driver::DevicePtr;
use candle_core::{DType, Storage, Tensor};
use half::f16;
use llvq_cuda::pers;

use crate::fused_cuda::{CudaInt4, CudaLattice, TetraView};
use crate::model::Proj;

/// Is a projection the int4 one 4E reads (for the mode's refusals)?
pub fn is_int4(p: &Proj) -> bool {
    int4_of(p).is_some()
}

// ---- the device views 4E reads, moved here from route P's host (2026-11-08) ----

pub(super) fn tensor_addr(t: &Tensor, stream: &candle_core::cuda_backend::cudarc::driver::CudaStream) -> candle_core::Result<u64> {
    let (st, lay) = t.storage_and_layout();
    match &*st {
        Storage::Cuda(cs) => Ok(cs.as_cuda_slice::<f16>()?.device_ptr(stream).0 + 2 * lay.start_offset() as u64),
        _ => candle_core::bail!("4E: a tensor that is not on the card"),
    }
}

pub(super) fn tetra_of(p: &Proj) -> Option<(TetraView, &CudaLattice)> {
    let Proj::Lattice(d) = p else { return None };
    let c = d.as_any()?.downcast_ref::<CudaLattice>()?;
    Some((c.tetra_view()?, c))
}

pub(super) fn int4_of(p: &Proj) -> Option<&CudaInt4> {
    let Proj::Int4(d) = p else { return None };
    d.as_any()?.downcast_ref::<CudaInt4>()
}

pub(super) fn seg_tetra(v: &TetraView) -> pers::SegTetraHost {
    pers::SegTetraHost {
        words: v.words,
        rscale: v.rscale,
        tail: v.tail,
        gscale: v.gscale,
        stride_u32: v.stride_u32,
        nblocks: v.nblocks,
        tail_w: v.tail_w,
        d_out: v.d_out,
    }
}

/// One layer's `SegArgs`, the constants of `base` (SiLU table, scratch,
/// tables, slices, counters, widths) completed with this block's ln2, its
/// gate/up rotation and Tetra streams, and its Tetra or int4 down. Returns the
/// image, whether the down is Tetra, and an int4 down's word order (`None`
/// for a Tetra down).
pub(super) fn mlp_args(
    block: &crate::model::Block,
    frt: &crate::fused_cuda::FusedRuntime,
    base: &pers::SegArgsHost,
    stream: &candle_core::cuda_backend::cudarc::driver::CudaStream,
) -> anyhow::Result<(pers::SegArgsHost, bool, Option<crate::fused::Int4Layout>)> {
    let (hidden, inter) = (base.hidden, base.inter);
    let mut q4_layout = None;
    let (ln2_w, eps, _, gate, up, down) = block.mlp_parts();
    if ln2_w.dtype() != DType::F16 {
        anyhow::bail!("ln2 is {:?}, the segment reads binary16", ln2_w.dtype());
    }
    let (g, _) = tetra_of(gate).ok_or_else(|| anyhow::anyhow!("gate_proj is not a Tetra48 stream"))?;
    let (u, _) = tetra_of(up).ok_or_else(|| anyhow::anyhow!("up_proj is not a Tetra48 stream"))?;
    let key = g.rotation.ok_or_else(|| anyhow::anyhow!("gate_proj carries no rotation"))?;
    if u.rotation != Some(key) {
        anyhow::bail!("gate and up are not under one rotation (LLVQ_ROT_SHARE=1 is the served form)");
    }
    let rh = frt.rot_view(&key).ok_or_else(|| anyhow::anyhow!("no rotation buffers for gate/up"))?;
    if pers::rot_mk(rh.n).map_err(anyhow::Error::msg)? != (rh.m, rh.k) || rh.n != hidden {
        anyhow::bail!("the gate/up rotation is {}x{} at width {}", rh.m, rh.k, rh.n);
    }
    let mut a = base.clone();
    a.alpha = tensor_addr(ln2_w, stream)?;
    a.eps = eps as f32;
    a.signs_h = rh.signbits;
    a.small_h = rh.small;
    a.m_h = rh.m;
    a.k_h = rh.k;
    a.inv_h = rh.inv;
    a.gate = seg_tetra(&g);
    a.up = seg_tetra(&u);
    let tetra_down = if let Some((d, _)) = tetra_of(down) {
        let dk = d.rotation.ok_or_else(|| anyhow::anyhow!("down_proj carries no rotation"))?;
        let ri = frt.rot_view(&dk).ok_or_else(|| anyhow::anyhow!("no rotation buffers for down"))?;
        if ri.n != inter || pers::rot_mk(ri.n).map_err(anyhow::Error::msg)? != (ri.m, ri.k) {
            anyhow::bail!("the down rotation is {}x{} at width {}", ri.m, ri.k, ri.n);
        }
        a.down = seg_tetra(&d);
        a.signs_i = ri.signbits;
        a.small_i = ri.small;
        a.m_i = ri.m;
        a.k_i = ri.k;
        a.inv_i = ri.inv;
        true
    } else if let Some(q) = int4_of(down) {
        let v = q.int4_view();
        if v.d_in != inter || v.d_out != hidden || v.gpr != inter.div_ceil(128) {
            anyhow::bail!("the int4 down_proj is {}x{} with {} groups a row", v.d_out, v.d_in, v.gpr);
        }
        a.q4w = v.wq;
        a.q4s = v.scales;
        a.q4b = v.biases;
        a.gpr = v.gpr;
        q4_layout = Some(v.layout);
        false
    } else {
        anyhow::bail!("down_proj is neither a Tetra48 stream nor an int4 record");
    };
    Ok((a, tetra_down, q4_layout))
}
