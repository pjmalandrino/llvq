//! The additive bias of a prompt's attention scores, computed on the card.
//!
//! [`super::layers`]'s `score_bias` builds three `[T, S]` tables on the host
//! (the relative index, the keep flag and the mask), uploads them, and runs a
//! gather, a product and a sum. Over a prompt that is `T x S` host entries a
//! visit and a copy of as many floats, growing with the square of the length.
//! This kernel writes the same `[heads, T, S]` tensor from the relative logits
//! alone.
//!
//! Exactness: each element is `rel[clamp(d)] * keep + mask` with one rounded
//! product and one rounded sum, the two operations candle runs in two
//! launches on the same operands. The values are meant to be the host
//! path's, bit for bit; a card run compares the logits.
//!
//! Off by default: [`set_on_device`] turns it on for a process.

#![cfg(all(target_os = "linux", feature = "cuda"))]

use std::sync::{Arc, Mutex};

use candle_core::cuda_backend::cudarc::driver::{CudaFunction, LaunchConfig, PushKernelArg};
use candle_core::{CpuStorage, CudaStorage, CustomOp1, DType, Device, Layout, Result, Shape, Tensor};

const SRC: &str = r#"
typedef unsigned int u32;
typedef unsigned long long u64;

// out[h][i][j] for query row i at position offset + i and key j at position j:
// the relative logit at the clamped distance, kept inside the extent, plus
// the causal mask and, when window > 0, the sliding window's.
extern "C" __global__ void attn_bias_rows(const float* __restrict__ rel,
                                          float* __restrict__ out,
                                          u32 heads, u32 t, u32 s, u32 extent, u32 offset, u32 window)
{
    u64 g = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    u64 per_head = (u64)t * s;
    if (g >= per_head * heads) { return; }
    u32 h = (u32)(g / per_head);
    u64 r = g - (u64)h * per_head;
    u32 i = (u32)(r / s);
    u32 j = (u32)(r - (u64)i * s);
    long long d = (long long)offset + (long long)i - (long long)j;
    u32 c = d < 0 ? 0u : (d >= (long long)extent ? extent - 1u : (u32)d);
    float keep = (d >= 0 && d < (long long)extent) ? 1.0f : 0.0f;
    bool visible = d >= 0 && (window == 0u || d < (long long)window);
    float mask = visible ? 0.0f : __int_as_float(0xff800000);
    out[g] = __fadd_rn(__fmul_rn(rel[((u64)h * t + i) * extent + c], keep), mask);
}
"#;

struct Kernel {
    _module: llvq_cuda::gpu::Cuda,
    f: CudaFunction,
}

static KERNEL: Mutex<Option<Arc<Kernel>>> = Mutex::new(None);

/// Compile the kernel on `dev` and use it for every prompt bias of this
/// process (`Some`), or go back to the host tables (`None`).
pub fn set_on_device(dev: Option<&Device>) -> Result<()> {
    let k = match dev {
        None => None,
        Some(Device::Cuda(d)) => {
            let src = llvq_cuda::gpu::KernelSource::new(&[SRC]);
            println!("NVRTC bias unit: {} bytes, sha256 {}", src.text.len(), src.sha256);
            let module = llvq_cuda::gpu::Cuda::on_stream(d.cuda_stream(), &src).map_err(candle_core::Error::msg)?;
            let f = module.func("attn_bias_rows").map_err(candle_core::Error::msg)?;
            Some(Arc::new(Kernel { _module: module, f }))
        }
        Some(_) => candle_core::bail!("the attention bias kernel wants a CUDA device"),
    };
    *KERNEL.lock().expect("no panic holds this lock") = k;
    Ok(())
}

/// The bias `[heads, T, S]` of `rel_logits` `[heads, T, extent]` on the card,
/// or `None` when the kernel is off or the logits live elsewhere. `offset`
/// and the key index count from the first key the slot holds
/// (`SlotCache::base`): only their difference enters, so a bounded sliding
/// slot passes its query position minus its base and the kernel is unchanged.
pub fn bias(rel_logits: &Tensor, offset: usize, s: usize, extent: usize, window: Option<usize>) -> Result<Option<Tensor>> {
    let Some(k) = KERNEL.lock().expect("no panic holds this lock").clone() else {
        return Ok(None);
    };
    let (heads, t, e) = rel_logits.dims3()?;
    if !rel_logits.device().is_cuda() || rel_logits.dtype() != DType::F32 || e != extent || heads * t * s == 0 {
        return Ok(None);
    }
    let words = [heads, t, s, extent, offset, window.unwrap_or(0)];
    if words.iter().any(|&w| w > u32::MAX as usize) || window == Some(0) {
        return Ok(None);
    }
    let op = Launch { k: &k, words: words.map(|w| w as u32), out: (heads, t, s) };
    rel_logits.contiguous()?.apply_op1_no_bwd(&op).map(Some)
}

struct Launch<'a> {
    k: &'a Kernel,
    words: [u32; 6],
    out: (usize, usize, usize),
}

impl CustomOp1 for Launch<'_> {
    fn name(&self) -> &'static str {
        "inkling-attn-bias"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("the attention bias kernel has no CPU path")
    }

    fn cuda_fwd(&self, storage: &CudaStorage, layout: &Layout) -> Result<(CudaStorage, Shape)> {
        let rel = storage.as_cuda_slice::<f32>()?;
        let (heads, t, s) = self.out;
        match layout.contiguous_offsets() {
            Some((0, end)) if end == heads * t * self.words[3] as usize => {}
            _ => candle_core::bail!("attention bias: the relative logits must be contiguous at offset 0"),
        }
        let dev = storage.device.clone();
        let stream = dev.cuda_stream();
        let n = heads * t * s;
        let mut out = unsafe { stream.alloc::<f32>(n) }
            .map_err(|e| candle_core::Error::msg(format!("attention bias: alloc of {n} floats: {e}")))?;
        let blocks = n.div_ceil(256);
        if blocks > i32::MAX as usize {
            candle_core::bail!("attention bias: {n} elements, past one launch");
        }
        let cfg = LaunchConfig { grid_dim: (blocks as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let mut b = stream.launch_builder(&self.k.f);
        b.arg(rel).arg(&mut out);
        for w in &self.words {
            b.arg(w);
        }
        crate::inkling::count_launch();
        unsafe { b.launch(cfg) }.map_err(|e| candle_core::Error::msg(format!("attn_bias_rows: {e}")))?;
        Ok((CudaStorage::wrap_cuda_slice(out, dev), Shape::from(self.out)))
    }
}
