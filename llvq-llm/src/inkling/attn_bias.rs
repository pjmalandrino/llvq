//! The additive bias of a prompt's attention scores, computed on the device.
//!
//! [`super::layers`]'s `score_bias` builds three `[T, S]` tables on the host
//! (the relative index, the keep flag and the mask), uploads them, and runs a
//! gather, a product and a sum. Over a prompt that is `T x S` host entries a
//! visit and a copy of as many floats, growing with the square of the length.
//! This kernel writes the same `[heads, T, S]` tensor from the relative logits
//! alone: `attn_bias_rows`, in CUDA (NVRTC) and in MSL.
//!
//! Exactness: each element is `rel[clamp(d)] * keep + mask` with one rounded
//! product and one rounded sum, the two operations candle runs in two
//! launches on the same operands. The values are the host path's, bit for
//! bit: on Metal the library is compiled with `fp contract(off)` and the safe
//! math mode, so the product is not fused into the sum, and the twin test
//! `tests/inkling_attn_bias.rs` compares the bits; a card run compares the
//! logits.
//!
//! Off by default: [`set_on_device`] turns it on for a process.

#![cfg(ink_glue)]

use std::sync::{Arc, Mutex};

#[cfg(all(target_os = "linux", feature = "cuda"))]
use candle_core::cuda_backend::cudarc::driver::{CudaFunction, LaunchConfig, PushKernelArg};
#[cfg(all(target_os = "linux", feature = "cuda"))]
use candle_core::CudaStorage;
#[cfg(all(target_os = "macos", feature = "metal"))]
use candle_core::{backend::BackendStorage, MetalStorage};
use candle_core::{CpuStorage, CustomOp1, DType, Device, Layout, Result, Shape, Tensor};

#[cfg(all(target_os = "linux", feature = "cuda"))]
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

/// The MSL twin of the CUDA kernel above: the same index arithmetic, one
/// thread an element, the product and the sum rounded apart.
#[cfg(all(target_os = "macos", feature = "metal"))]
const SRC_METAL: &str = r#"
#include <metal_stdlib>
using namespace metal;
#pragma METAL fp contract(off)

struct BiasArgs { uint heads; uint t; uint s; uint extent; uint offset; uint window; };

// out[h][i][j] for query row i at position offset + i and key j at position j:
// the relative logit at the clamped distance, kept inside the extent, plus
// the causal mask and, when window > 0, the sliding window's.
kernel void attn_bias_rows(device const float *rel [[buffer(0)]],
                           device float *out [[buffer(1)]],
                           constant BiasArgs &a [[buffer(2)]],
                           uint2 gp [[thread_position_in_grid]])
{
    // x: the key j, y: the row h * t + i. Two dimensions keep every index in 32 bits.
    uint j = gp.x;
    uint hi = gp.y;
    if (j >= a.s || hi >= a.heads * a.t) { return; }
    uint i = hi % a.t;
    long d = (long)a.offset + (long)i - (long)j;
    uint c = d < 0 ? 0u : (d >= (long)a.extent ? a.extent - 1u : (uint)d);
    float keep = (d >= 0 && d < (long)a.extent) ? 1.0f : 0.0f;
    bool visible = d >= 0 && (a.window == 0u || d < (long)a.window);
    float mask = visible ? 0.0f : -INFINITY;
    float p = rel[(ulong)hi * a.extent + c] * keep;
    out[(ulong)hi * a.s + j] = p + mask;
}
"#;

#[cfg(all(target_os = "linux", feature = "cuda"))]
struct Kernel {
    _module: llvq_cuda::gpu::Cuda,
    f: CudaFunction,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
struct Kernel {
    pipe: candle_metal_kernels::metal::ComputePipeline,
}

static KERNEL: Mutex<Option<Arc<Kernel>>> = Mutex::new(None);

/// Compile the kernel on `dev` and use it for every prompt bias of this
/// process (`Some`), or go back to the host tables (`None`).
pub fn set_on_device(dev: Option<&Device>) -> Result<()> {
    let k = match dev {
        None => None,
        Some(d) => Some(Arc::new(compile(d)?)),
    };
    *KERNEL.lock().expect("no panic holds this lock") = k;
    Ok(())
}

/// Whether the kernel is on for this process.
pub fn on_device() -> bool {
    KERNEL.lock().expect("no panic holds this lock").is_some()
}

#[cfg(all(target_os = "linux", feature = "cuda"))]
fn compile(dev: &Device) -> Result<Kernel> {
    let Device::Cuda(d) = dev else { candle_core::bail!("the attention bias kernel here wants a CUDA device") };
    let src = llvq_cuda::gpu::KernelSource::new(&[SRC]);
    println!("NVRTC bias unit: {} bytes, sha256 {}", src.text.len(), src.sha256);
    let module = llvq_cuda::gpu::Cuda::on_stream(d.cuda_stream(), &src).map_err(candle_core::Error::msg)?;
    let f = module.func("attn_bias_rows").map_err(candle_core::Error::msg)?;
    Ok(Kernel { _module: module, f })
}

#[cfg(all(target_os = "macos", feature = "metal"))]
fn compile(dev: &Device) -> Result<Kernel> {
    let Device::Metal(m) = dev else { candle_core::bail!("the attention bias kernel here wants a Metal device") };
    use objc2_metal::{MTLCompileOptions, MTLMathFloatingPointFunctions, MTLMathMode};
    let raw = m.device().clone();
    let built = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let opts = MTLCompileOptions::new();
        opts.setMathMode(MTLMathMode::Safe);
        opts.setMathFloatingPointFunctions(MTLMathFloatingPointFunctions::Precise);
        let lib = raw.new_library_with_source(SRC_METAL, Some(&opts))?;
        raw.new_compute_pipeline_state_with_function(&lib.get_function("attn_bias_rows", None)?)
    }));
    let pipe = match built {
        Ok(Ok(p)) => p,
        Ok(Err(e)) => candle_core::bail!("attn_bias_rows: Metal refused the source: {e:?}"),
        Err(panic) => {
            let what = panic
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_else(|| "no diagnostics".into());
            candle_core::bail!("attn_bias_rows: the MSL did not compile: {what}")
        }
    };
    Ok(Kernel { pipe })
}

/// The bias `[heads, T, S]` of `rel_logits` `[heads, T, extent]` on the
/// device, or `None` when the kernel is off or the logits live elsewhere.
/// `offset` and the key index count from the first key the slot holds
/// (`SlotCache::base`): only their difference enters, so a bounded sliding
/// slot passes its query position minus its base and the kernel is unchanged.
pub fn bias(rel_logits: &Tensor, offset: usize, s: usize, extent: usize, window: Option<usize>) -> Result<Option<Tensor>> {
    let Some(k) = KERNEL.lock().expect("no panic holds this lock").clone() else {
        return Ok(None);
    };
    bias_with(&k, rel_logits, offset, s, extent, window)
}

/// [`bias`] through a kernel compiled for the call, whatever the process
/// switch says: the twin test's entry point.
pub fn bias_once(dev: &Device, rel_logits: &Tensor, offset: usize, s: usize, extent: usize, window: Option<usize>) -> Result<Option<Tensor>> {
    bias_with(&compile(dev)?, rel_logits, offset, s, extent, window)
}

fn bias_with(
    k: &Kernel,
    rel_logits: &Tensor,
    offset: usize,
    s: usize,
    extent: usize,
    window: Option<usize>,
) -> Result<Option<Tensor>> {
    let (heads, t, e) = rel_logits.dims3()?;
    #[cfg(all(target_os = "linux", feature = "cuda"))]
    let here = rel_logits.device().is_cuda();
    #[cfg(all(target_os = "macos", feature = "metal"))]
    let here = rel_logits.device().is_metal();
    if !here || rel_logits.dtype() != DType::F32 || e != extent || heads * t * s == 0 {
        return Ok(None);
    }
    let words = [heads, t, s, extent, offset, window.unwrap_or(0)];
    if words.iter().any(|&w| w > u32::MAX as usize) || heads * t > u32::MAX as usize || window == Some(0) {
        return Ok(None);
    }
    let op = Launch { k, words: words.map(|w| w as u32), out: (heads, t, s) };
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

    #[cfg(all(target_os = "linux", feature = "cuda"))]
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

    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn metal_fwd(&self, storage: &MetalStorage, layout: &Layout) -> Result<(MetalStorage, Shape)> {
        use objc2_metal::{MTLResourceUsage, MTLSize};
        let (heads, t, s) = self.out;
        if storage.dtype() != DType::F32 || !layout.is_contiguous() {
            candle_core::bail!("attention bias: the relative logits must be contiguous f32");
        }
        let dev = storage.device().clone();
        let n = heads * t * s;
        let out = dev.new_buffer(n, DType::F32, "inkling-attn-bias")?;
        let enc = dev.command_encoder()?;
        enc.set_compute_pipeline_state(&self.k.pipe);
        enc.set_buffer(0, Some(storage.buffer()), layout.start_offset() * DType::F32.size_in_bytes());
        enc.use_resource(storage.buffer(), MTLResourceUsage::Read);
        enc.set_buffer(1, Some(&out), 0);
        enc.use_resource(&*out, MTLResourceUsage::Write);
        enc.set_bytes_directly(2, 4 * self.words.len(), self.words.as_ptr() as *const std::ffi::c_void);
        // 32 keys by 8 rows a threadgroup; the grid need not be a multiple of it, and the kernel guards its edge.
        enc.dispatch_threads(
            MTLSize { width: s, height: heads * t, depth: 1 },
            MTLSize { width: 32, height: 8, depth: 1 },
        );
        drop(enc);
        Ok((MetalStorage::new(out, dev, n, DType::F32), Shape::from(self.out)))
    }
}
