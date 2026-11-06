//! The split-KV decode-attention kernel, as the model sees it: its canonical
//! arithmetic, its CPU fake, and the one function that builds the right
//! [`AttnKernel`] for a device.
//!
//! The arithmetic is `llvq-cuda/src/attn_ref.rs`, included here as text:
//! `llvq-cuda` is a Linux-only dependency of this crate, and the CPU fake must
//! run on the Mac. One file, two crates, nothing to keep in step by hand. The
//! card's kernel returns the same bits (`llvq-cuda/tests/attn_dec_matches_rust.rs`
//! on the Mac, `llvq_cuda::attn::AttnDec::self_check` at every load), so a host
//! test that runs [`CpuAttn`] runs the card's arithmetic.
//!
//! What this form is NOT: bit for bit with the grouped form it replaces on
//! decode rows. cuBLAS's reduction order is not the kernel's, and the grouped
//! form's scale and softmax run in f16. Serving it needs the operator's T8
//! decision (`docs/archive/debit-vague5-2026-11-02/attention.md` §7).

use std::sync::Arc;

use candle_core::{DType, Device, Result, Tensor};

use crate::device::AttnKernel;

/// The canonical arithmetic, `llvq_cuda::attn_ref` by text.
#[allow(clippy::all)]
pub mod canon {
    include!("../../llvq-cuda/src/attn_ref.rs");
}

/// The shapes one call hands the kernel, checked once for both backends.
///
/// `q` is `[1, n_heads, l, hd]`; `k` and `v` are `[1, n_kv, len, hd]` and
/// hold the `l` new rows at their end, so the rows sit at `len - l ..len`.
pub struct Call {
    pub n_heads: usize,
    pub n_kv: usize,
    pub groups: usize,
    pub l: usize,
    pub len: usize,
    pub pos0: usize,
}

impl Call {
    pub fn of(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Self> {
        for (name, t) in [("q", q), ("k", k), ("v", v)] {
            if t.dtype() != DType::F16 {
                candle_core::bail!(
                    "attention kernel: {name} is {:?}; the kernel reads f16 and nothing else, \
                     and a silent conversion would change the values it attends over",
                    t.dtype()
                );
            }
        }
        let (b, n_heads, l, hd) = q.dims4()?;
        let (bk, n_kv, len, hdk) = k.dims4()?;
        if v.dims4()? != (bk, n_kv, len, hdk) {
            candle_core::bail!("attention kernel: k {:?} and v {:?} differ", k.dims(), v.dims());
        }
        if b != 1 || bk != 1 {
            candle_core::bail!("attention kernel: batch {b}; the kernel serves batch 1");
        }
        if hd != canon::HD || hdk != canon::HD {
            candle_core::bail!("attention kernel: head dim {hd}/{hdk}; the kernel compiles {}", canon::HD);
        }
        if n_kv == 0 || n_heads % n_kv != 0 {
            candle_core::bail!("attention kernel: {n_heads} query heads over {n_kv} KV heads");
        }
        if !(1..=canon::KMAX).contains(&l) || l > len {
            candle_core::bail!("attention kernel: {l} rows over a cache of {len}, 1..={} rows a call", canon::KMAX);
        }
        Ok(Self { n_heads, n_kv, groups: n_heads / n_kv, l, len, pos0: len - l })
    }
}

/// The CPU fake: the canonical replay over host copies of the tensors.
///
/// Slow (one f32 pass per position and head, no SIMD) and exact: it returns
/// what the card returns. Built for the CPU device only.
pub struct CpuAttn {
    unit: usize,
    c: f32,
}

impl CpuAttn {
    pub fn new(unit: usize) -> std::result::Result<Self, String> {
        canon::check_unit(unit)?;
        Ok(Self { unit, c: canon::score_scale(canon::HD) })
    }
}

fn bits(t: &Tensor) -> Result<Vec<u16>> {
    Ok(t.contiguous()?.flatten_all()?.to_vec1::<half::f16>()?.into_iter().map(|h| h.to_bits()).collect())
}

impl AttnKernel for CpuAttn {
    fn name(&self) -> &str {
        "attn_dec (CPU replay)"
    }

    fn max_rows(&self) -> usize {
        canon::KMAX
    }

    fn attend(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let call = Call::of(q, k, v)?;
        let (qb, kb, vb) = (bits(q)?, bits(k)?, bits(v)?);
        let kv = canon::Kv { k: &kb, v: &vb, cap: call.len };
        let out = canon::attend(&qb, &kv, call.n_kv, call.groups, call.pos0, call.l, self.c, self.unit);
        let out: Vec<half::f16> = out.into_iter().map(half::f16::from_bits).collect();
        Tensor::from_vec(out, (1, call.n_heads, call.l, canon::HD), q.device())
    }
}

/// The kernel for `device`, sized for a model of `n_kv` KV heads and `groups`
/// query heads per KV head, at the served unit size.
///
/// CPU: the replay. CUDA: `attn_dec.cu` compiled onto candle's stream, its
/// driver report printed, a spill refused, and the card checked against the
/// replay bit for bit before it is handed out (`attn_cuda::attn_kernel`).
/// Metal: `attn_dec.metal`, the same arithmetic, checked against the replay
/// bit for bit before it is handed out (`attn_metal::attn_kernel`). Built only
/// where a caller asks for it: `Served::apply` still refuses the form on Metal
/// by name, and `bin/metalrun` reaches it as a measurement arm.
pub fn kernel_for(
    device: &Device,
    n_kv: usize,
    groups: usize,
    head_dim: usize,
) -> std::result::Result<Arc<dyn AttnKernel>, String> {
    if head_dim != canon::HD {
        return Err(format!("attention \"kernel\": head dim {head_dim}; the kernel compiles {}", canon::HD));
    }
    match device {
        Device::Cpu => Ok(Arc::new(CpuAttn::new(canon::UNIT_SERVED)?)),
        #[cfg(all(target_os = "linux", feature = "cuda"))]
        Device::Cuda(_) => crate::attn_cuda::attn_kernel(device, n_kv, groups),
        #[cfg(not(all(target_os = "linux", feature = "cuda")))]
        Device::Cuda(_) => {
            let _ = (n_kv, groups);
            Err("attention \"kernel\": this build carries no CUDA backend".to_string())
        }
        #[cfg(all(target_os = "macos", feature = "metal"))]
        Device::Metal(_) => crate::attn_metal::attn_kernel(device, n_kv, groups),
        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        Device::Metal(_) => {
            let _ = (n_kv, groups);
            Err("attention \"kernel\": this build carries no Metal backend".to_string())
        }
    }
}
