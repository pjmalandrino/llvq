//! The decode-attention kernel inside candle, on CUDA.
//!
//! A thin adapter: everything that can be checked on the Mac lives in
//! `llvq_cuda::attn` (the launcher, its load-time check), which the
//! cross-target clippy type-checks. This file compiles nowhere but inside an
//! image build, so it does as little as it can: it finds the device pointers
//! of three candle tensors, reads the cache capacity off the view's stride,
//! and hands them over.
//!
//! The module is compiled onto candle's own stream, for the reason
//! `fused_cuda.rs` gives: ordering is then the stream's guarantee, not
//! something to remember. It is its own translation unit, so the projections'
//! unit keeps its bytes and its printed sha256.

use std::sync::Arc;

use candle_core::cuda_backend::cudarc::driver::CudaView;
use candle_core::{CudaStorage, Device, Layout, Shape};
use half::f16;

use crate::attn::{canon, Call};
use crate::device::AttnKernel;

/// The kernel for a CUDA device: compiled, reported, refused on a spill,
/// and checked against the CPU replay bit for bit before it is handed out.
pub fn attn_kernel(device: &Device, n_kv: usize, groups: usize) -> Result<Arc<dyn AttnKernel>, String> {
    let dev = device.as_cuda_device().map_err(|e| format!("attention kernel: {e}"))?.clone();
    let module = llvq_cuda::attn::AttnModule::on_stream(dev.cuda_stream())?;
    println!("{}", module.provenance());
    let dec = llvq_cuda::attn::AttnDec::new(module, n_kv, groups, canon::UNIT_SERVED)?;
    println!("{}", dec.report_line());
    if dec.report.local_bytes != 0 {
        return Err(format!(
            "{}: {} bytes of spill. The kernel keeps its states in registers, and a spill turns \
             them into DRAM traffic on the one kernel whose point is to read the cache once",
            dec.name, dec.report.local_bytes
        ));
    }
    let t0 = std::time::Instant::now();
    let line = dec.self_check()?;
    println!("{line} ({:.1} s)", t0.elapsed().as_secs_f64());
    Ok(Arc::new(CudaAttn { dec, c: canon::score_scale(canon::HD), device: dev }))
}

struct CudaAttn {
    dec: llvq_cuda::attn::AttnDec,
    c: f32,
    device: candle_core::CudaDevice,
}

impl AttnKernel for CudaAttn {
    fn name(&self) -> &str {
        self.dec.name
    }

    fn max_rows(&self) -> usize {
        canon::KMAX
    }

    fn attend(
        &self,
        q: &candle_core::Tensor,
        k: &candle_core::Tensor,
        v: &candle_core::Tensor,
    ) -> candle_core::Result<candle_core::Tensor> {
        let call = Call::of(q, k, v)?;
        if call.n_kv != self.dec.n_kv || call.groups != self.dec.groups {
            candle_core::bail!(
                "{}: built for {} KV heads x {} groups, called with {} x {}",
                self.dec.name,
                self.dec.n_kv,
                self.dec.groups,
                call.n_kv,
                call.groups
            );
        }
        q.apply_op3_no_bwd(k, v, &AttnOp { rt: self, call })
    }
}

struct AttnOp<'a> {
    rt: &'a CudaAttn,
    call: Call,
}

/// A view of a `Grow` buffer `[1, n_kv, len, hd]` whose head stride is the
/// capacity, as the device slice the kernel indexes and that capacity.
fn cache_view<'a>(
    s: &'a CudaStorage,
    l: &Layout,
    call: &Call,
) -> candle_core::Result<(CudaView<'a, f16>, usize)> {
    let st = l.stride();
    let hd = canon::HD;
    if st.len() != 4 || st[3] != 1 || st[2] != hd || !st[1].is_multiple_of(hd) {
        candle_core::bail!(
            "attention kernel: a cache view of stride {st:?}; the kernel reads rows of {hd} \
             contiguous values, heads a whole number of rows apart"
        );
    }
    let cap = st[1] / hd;
    let start = l.start_offset();
    if !start.is_multiple_of(8) {
        candle_core::bail!("attention kernel: a cache view at offset {start}, not 16-byte aligned");
    }
    let all = s.as_cuda_slice::<f16>()?;
    let end = start + call.n_kv * cap * hd;
    if end > all.len() || cap < call.len {
        candle_core::bail!(
            "attention kernel: a cache of capacity {cap} for {} positions, {} values held past offset {start}",
            call.len,
            all.len()
        );
    }
    Ok((all.slice(start..end), cap))
}

impl candle_core::CustomOp3 for AttnOp<'_> {
    fn name(&self) -> &'static str {
        "llvq-attn-dec"
    }

    fn cpu_fwd(
        &self,
        _: &candle_core::CpuStorage,
        _: &Layout,
        _: &candle_core::CpuStorage,
        _: &Layout,
        _: &candle_core::CpuStorage,
        _: &Layout,
    ) -> candle_core::Result<(candle_core::CpuStorage, Shape)> {
        candle_core::bail!("the CUDA attention kernel has no CPU path; the CPU uses crate::attn::CpuAttn")
    }

    fn cuda_fwd(
        &self,
        sq: &CudaStorage,
        lq: &Layout,
        sk: &CudaStorage,
        lk: &Layout,
        sv: &CudaStorage,
        lv: &Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        let c = &self.call;
        let (qs, qe) = lq
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::msg("attention kernel: q is not contiguous"))?;
        let q = sq.as_cuda_slice::<f16>()?.slice(qs..qe);
        let (kc, cap_k) = cache_view(sk, lk, c)?;
        let (vc, cap_v) = cache_view(sv, lv, c)?;
        if cap_k != cap_v {
            candle_core::bail!("attention kernel: K capacity {cap_k}, V capacity {cap_v}");
        }
        let n = c.n_heads * c.l * canon::HD;
        // Uninitialised on purpose: the combine writes every (head, row, dim).
        let mut out = unsafe { self.rt.device.cuda_stream().alloc::<f16>(n) }
            .map_err(|e| candle_core::Error::msg(format!("alloc attention out: {e}")))?;
        self.rt
            .dec
            .attend(&q, &kc, &vc, &mut out, c.pos0, c.l, cap_k, self.rt.c, None)
            .map_err(candle_core::Error::msg)?;
        Ok((
            CudaStorage::wrap_cuda_slice(out, self.rt.device.clone()),
            Shape::from((1, c.n_heads, c.l, canon::HD)),
        ))
    }
}
