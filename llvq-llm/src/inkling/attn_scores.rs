//! The attention of one decoded row over the slot cache, in one Metal launch.
//!
//! Plan items D5 (grouped read) and D6 (bias and masks on the device), for
//! one query row. `AttnWeights::forward_chunked` builds, from the scores to
//! the weighted sum, a relative-bias index, a keep mask and a causal or window
//! mask on the host, uploads them, copies the keys and the values three times
//! for the query heads (`repeat_kv`), and runs a score product, a bias add, a
//! concatenation with the sinks, a softmax, a narrow and a value product.
//! [`AttnRow::forward`] does the same mathematics in f32 in one launch:
//!
//!   - one threadgroup a query head; head `h` reads key-value head
//!     `h / (heads / kv_heads)` where the cache holds it, through the strides
//!     of the view (no repeat, no transpose, no copy);
//!   - only the positions the row can see are read: `0..=pos` in a full layer,
//!     the last `window` of them in a sliding layer ([`visible`]);
//!   - the score of position `j` is `(q . k_j) * hd^-0.5 + bias`, where the
//!     bias is `rel[pos - j]` when `pos - j < extent` and 0 otherwise, as the
//!     keep mask of `score_bias` makes it;
//!   - the sink is one more logit with no value; softmax over the visible
//!     scores and the sink; the weighted sum of the values.
//!
//! Exactness: class B. The dot products run in a sequential fused
//! multiply-add over the head dimension, the max and the sum of the softmax in
//! a threadgroup tree, and the value sum in 32 simdgroup partials added in a
//! fixed order. candle runs its own GEMM and reduction orders, which differ on
//! the CPU and on Metal. The result is deterministic run to run. Positions
//! outside the window add exact zeros in the reference (`exp(-inf) = 0`, and
//! `0 * v = 0`), so skipping them changes no bit by itself.
//!
//! Bytes a launch reads ([`bytes_read`]): the query, `min(n, extent)` relative
//! logits, the sinks, and `n` keys and `n` values of `hd` floats for each of
//! the `heads` query heads, `n` the visible positions; plus the `n` scores a
//! head writes once and reads twice in its scratch. Each key-value head is
//! read by `heads / kv_heads` threadgroups; whether the second and third read
//! hit a cache is the device's business, so both counts are given.
//!
//! The library is compiled like `inkling::glue`: `MTLMathMode::Safe`, precise
//! functions, `fp contract(off)`, and `precise::` for `exp` and the division.
//! Same candle traps as `crate::fused_metal` (its header): the compile runs
//! under `catch_unwind`; every output word is written by the kernel.
#![cfg(ink_glue)]

use std::sync::RwLockReadGuard;

#[cfg(all(target_os = "macos", feature = "metal"))]
use candle_core::backend::BackendStorage;
#[cfg(all(target_os = "macos", feature = "metal"))]
use candle_core::MetalStorage;
use candle_core::{CpuStorage, CustomOp1, DType, Device, Layout, Result, Shape, Storage, Tensor};
#[cfg(all(target_os = "macos", feature = "metal"))]
use candle_metal_kernels::metal::{Buffer, ComputePipeline};
#[cfg(all(target_os = "macos", feature = "metal"))]
use objc2_metal::{MTLResourceUsage, MTLSize};

#[cfg(all(target_os = "linux", feature = "cuda"))]
use candle_core::cuda_backend::cudarc::driver::{CudaFunction, DevicePtr, LaunchConfig, PushKernelArg};
#[cfg(all(target_os = "linux", feature = "cuda"))]
use candle_core::CudaStorage;

#[cfg(all(target_os = "macos", feature = "metal"))]
type Pipe = ComputePipeline;
#[cfg(all(target_os = "linux", feature = "cuda"))]
type Pipe = CudaFunction;

#[cfg(all(target_os = "linux", feature = "cuda"))]
const SRC_CUDA: &str = include_str!("../../kernels/ink_scores.cu");

/// Threads a threadgroup (one query head).
const TG: usize = 1024;
/// Largest head dimension the kernel stages (4 columns a lane).
pub const MAX_HD: usize = 128;

#[cfg(all(target_os = "macos", feature = "metal"))]
const SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;
#pragma METAL fp contract(off)

#define NSG (ATTN_TG / 32u)
#define MAX_HD 128u
#define COLS (MAX_HD / 32u)

struct Args {
    uint nh; uint nkv; uint hd; uint extent;
    uint pos; uint lo; uint nvis;
    uint ks_h; uint ks_p; uint ks_d;
    uint vs_h; uint vs_p; uint vs_d;
    float scale;
};

static float tg_sum(float v, threadgroup float *scr, uint lane, uint sg) {
    float s = simd_sum(v);
    if (lane == 0) { scr[sg] = s; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float t = lane < NSG ? scr[lane] : 0.0f;
    t = simd_sum(t);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return t;
}

static float tg_max(float v, threadgroup float *scr, uint lane, uint sg) {
    float s = simd_max(v);
    if (lane == 0) { scr[sg] = s; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float t = lane < NSG ? scr[lane] : -INFINITY;
    t = simd_max(t);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return t;
}

// out: the result [nh, hd] | scratch [nh, nvis] (the scores, then their exponentials).
kernel void attn_row(
    device const float *q [[buffer(0)]],
    device const float *rel [[buffer(1)]],
    device const float *sinks [[buffer(2)]],
    device const float *kb [[buffer(3)]],
    device const float *vb [[buffer(4)]],
    device float *out [[buffer(5)]],
    constant Args &a [[buffer(6)]],
    uint h [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]])
{
    threadgroup float Q[MAX_HD];
    threadgroup float SCR[NSG];
    threadgroup float P[NSG * MAX_HD];
    const uint g = h / (a.nh / a.nkv);
    for (uint c = tid; c < a.hd; c += ATTN_TG) { Q[c] = q[h * a.hd + c]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float *E = out + a.nh * a.hd + h * a.nvis;
    device const float *kh = kb + g * a.ks_h;
    device const float *vh = vb + g * a.vs_h;
    device const float *rh = rel + h * a.extent;

    // Scores: one position a thread, the dot in order over the head dimension.
    float mloc = -INFINITY;
    for (uint i = tid; i < a.nvis; i += ATTN_TG) {
        uint j = a.lo + i;
        device const float *kp = kh + j * a.ks_p;
        float acc = 0.0f;
        for (uint c = 0; c < a.hd; c++) { acc = fma(Q[c], kp[c * a.ks_d], acc); }
        float s = acc * a.scale;
        uint d = a.pos - j;
        float b = d < a.extent ? rh[d] : 0.0f;
        s = s + b;
        E[i] = s;
        mloc = max(mloc, s);
    }
    const float sink = sinks[h];
    float m = tg_max(mloc, SCR, lane, sg);
    m = max(m, sink);

    // Exponentials and their sum, the sink's term last.
    float zloc = 0.0f;
    for (uint i = tid; i < a.nvis; i += ATTN_TG) {
        float e = precise::exp(E[i] - m);
        E[i] = e;
        zloc = zloc + e;
    }
    float z = tg_sum(zloc, SCR, lane, sg);
    z = z + precise::exp(sink - m);
    threadgroup_barrier(mem_flags::mem_device);

    // Weighted sum of the values: a simdgroup a position, a lane four columns.
    float acc[COLS];
    for (uint r = 0; r < COLS; r++) { acc[r] = 0.0f; }
    for (uint i = sg; i < a.nvis; i += NSG) {
        float p = precise::divide(E[i], z);
        device const float *vp = vh + (a.lo + i) * a.vs_p;
        for (uint r = 0; r < COLS; r++) {
            uint c = lane + 32u * r;
            if (c < a.hd) { acc[r] = fma(p, vp[c * a.vs_d], acc[r]); }
        }
    }
    for (uint r = 0; r < COLS; r++) {
        uint c = lane + 32u * r;
        if (c < a.hd) { P[sg * MAX_HD + c] = acc[r]; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint c = tid; c < a.hd; c += ATTN_TG) {
        float t = 0.0f;
        for (uint s = 0; s < NSG; s++) { t = t + P[s * MAX_HD + c]; }
        out[h * a.hd + c] = t;
    }
}
"#;

/// The positions a query row at `pos` sees in a cache: `(first, count)`.
/// Causal: `j <= pos`. Sliding: `pos - j < window` too (`score_bias`).
pub fn visible(pos: usize, window: Option<usize>) -> (usize, usize) {
    let lo = window.map_or(0, |w| (pos + 1).saturating_sub(w));
    (lo, pos + 1 - lo)
}

/// Bytes one launch reads and writes, for `n` visible positions: `(logical,
/// unique)`. Logical counts every threadgroup's reads of keys and values;
/// unique counts each key-value head once. The scratch traffic (`n` scores
/// written, read, rewritten and read again, a head) is in both.
pub fn bytes_read(heads: usize, kv_heads: usize, hd: usize, extent: usize, n: usize) -> (usize, usize) {
    let fixed = 4 * (heads * hd + heads * n.min(extent) + heads) + 4 * 4 * heads * n + 4 * heads * hd;
    let kv = |h: usize| 2 * 4 * h * n * hd;
    (fixed + kv(heads), fixed + kv(kv_heads))
}

/// What [`AttnRow::forward`] reads for one query row.
pub struct RowInputs<'a> {
    /// The query after its scale, clip and log scale, `[heads, hd]` (any
    /// shape of that many elements), contiguous f32 on the Metal device.
    pub q: &'a Tensor,
    /// The relative logits of the row, `[heads, extent]` (any leading shape
    /// of `heads` rows, the last dimension the extent), contiguous.
    pub rel: &'a Tensor,
    /// The sink logits, `[heads]`, contiguous.
    pub sinks: &'a Tensor,
    /// Keys `[kv_heads, held, hd]`, any strides: what `SlotCache::keys_values`
    /// returns, a view of the cache buffer, read in place.
    pub keys: &'a Tensor,
    /// Values `[kv_heads, held, hd]`, any strides, read in place.
    pub values: &'a Tensor,
    /// The row's position; `pos < held` (its own key is in the cache).
    pub pos: usize,
    /// The sliding window, or `None` for a full layer.
    pub window: Option<usize>,
}

/// The compiled kernel.
pub struct AttnRow {
    pipe: Pipe,
    #[cfg(all(target_os = "linux", feature = "cuda"))]
    _module: llvq_cuda::gpu::Cuda,
}

impl AttnRow {
    /// Compile the kernel for a CUDA device (`kernels/ink_scores.cu`, NVRTC).
    #[cfg(all(target_os = "linux", feature = "cuda"))]
    pub fn new(device: &Device) -> Result<Self> {
        let Device::Cuda(dev) = device else { candle_core::bail!("attn_row wants a CUDA device") };
        let define = format!("#define ATTN_TG {TG}u\n");
        let src = llvq_cuda::gpu::KernelSource::new(&[define.as_str(), SRC_CUDA]);
        println!("NVRTC scores unit: {} bytes, sha256 {}", src.text.len(), src.sha256);
        let module = llvq_cuda::gpu::Cuda::on_stream(dev.cuda_stream(), &src).map_err(candle_core::Error::msg)?;
        let pipe = module.func("attn_row").map_err(candle_core::Error::msg)?;
        Ok(Self { pipe, _module: module })
    }

    /// Compile the kernel for a Metal device.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn new(device: &Device) -> Result<Self> {
        let Device::Metal(dev) = device else { candle_core::bail!("attn_row wants a Metal device") };
        use objc2_metal::{MTLCompileOptions, MTLMathFloatingPointFunctions, MTLMathMode};
        let raw = dev.device().clone();
        let src = format!("#define ATTN_TG {TG}u\n{SRC}");
        let built = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let opts = MTLCompileOptions::new();
            opts.setMathMode(MTLMathMode::Safe);
            opts.setMathFloatingPointFunctions(MTLMathFloatingPointFunctions::Precise);
            let lib = raw.new_library_with_source(&src, Some(&opts))?;
            raw.new_compute_pipeline_state_with_function(&lib.get_function("attn_row", None)?)
        }));
        let pipe = match built {
            Ok(Ok(p)) => p,
            Ok(Err(e)) => candle_core::bail!("attn_row: Metal refused the source: {e:?}"),
            Err(panic) => {
                let what = panic
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_else(|| "no diagnostics".into());
                candle_core::bail!("attn_row: the MSL did not compile: {what}")
            }
        };
        let max = pipe.max_total_threads_per_threadgroup();
        if max < TG {
            candle_core::bail!("attn_row: the pipeline takes {max} threads a threadgroup, the kernel needs {TG}");
        }
        Ok(Self { pipe })
    }

    /// The attention result of the row before the output gate, `[1, heads * hd]`.
    pub fn forward(&self, x: &RowInputs) -> Result<Tensor> {
        let (nkv, held, hd) = x.keys.dims3()?;
        if x.values.dims() != [nkv, held, hd] {
            candle_core::bail!("attn_row: values {:?} for keys {:?}", x.values.dims(), x.keys.dims());
        }
        if hd == 0 || hd > MAX_HD {
            candle_core::bail!("attn_row: head dim {hd}, the kernel takes 1 to {MAX_HD}");
        }
        let extent = x.rel.dims().last().copied().unwrap_or(0);
        let nh = x.rel.elem_count().checked_div(extent).unwrap_or(0);
        if nkv == 0 || nh == 0 || nh % nkv != 0 {
            candle_core::bail!("attn_row: {nh} query heads over {nkv} key-value heads");
        }
        if x.q.elem_count() != nh * hd || x.sinks.elem_count() != nh || extent == 0 {
            candle_core::bail!(
                "attn_row: query {:?}, sinks {:?}, relative logits {:?} for {nh} heads of {hd}",
                x.q.dims(),
                x.sinks.dims(),
                x.rel.dims()
            );
        }
        if x.pos >= held {
            candle_core::bail!("attn_row: row position {} in a cache of {held} positions", x.pos);
        }
        if x.window == Some(0) {
            candle_core::bail!("attn_row: a window of 0 positions");
        }
        let (lo, nvis) = visible(x.pos, x.window);
        let ks = strides(x.keys, "keys")?;
        let vs = strides(x.values, "values")?;
        let args = [nh, nkv, hd, extent, x.pos, lo, nvis, ks[0], ks[1], ks[2], vs[0], vs[1], vs[2]]
            .iter()
            .map(|&v| u32::try_from(v).map_err(|_| candle_core::Error::Msg(format!("attn_row: argument {v} > u32"))))
            .collect::<Result<Vec<u32>>>()?;
        let mut words = args;
        words.push(((hd as f64).powf(-0.5) as f32).to_bits());
        let op = Launch { pipe: &self.pipe, x, words, nh, out_elems: nh * hd + nh * nvis };
        let out = x.q.apply_op1_no_bwd(&op)?;
        out.narrow(0, 0, nh * hd)?.reshape((1, nh * hd))
    }
}

/// The element strides of a rank-3 f32 view on the Metal device, checked to
/// fit the kernel's 32-bit indexing.
fn strides(t: &Tensor, what: &str) -> Result<[usize; 3]> {
    if t.dtype() != DType::F32 {
        candle_core::bail!("attn_row: {what} are {:?}, the kernel takes f32", t.dtype());
    }
    let layout = t.layout();
    let s = layout.stride();
    let d = layout.dims();
    let last: usize = layout.start_offset() + d.iter().zip(s).map(|(n, st)| (n - 1) * st).sum::<usize>();
    if last >= u32::MAX as usize {
        candle_core::bail!("attn_row: {what} reach element {last}, past 32-bit indexing");
    }
    Ok([s[0], s[1], s[2]])
}

struct Launch<'a> {
    pipe: &'a Pipe,
    x: &'a RowInputs<'a>,
    words: Vec<u32>,
    nh: usize,
    out_elems: usize,
}

struct Bound<'a> {
    guard: RwLockReadGuard<'a, Storage>,
    offset: usize,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
impl Bound<'_> {
    fn buffer(&self) -> &Buffer {
        match &*self.guard {
            Storage::Metal(m) => m.buffer(),
            _ => unreachable!("checked at bind"),
        }
    }
}

/// A tensor's storage and byte offset; `contiguous` asks for a dense layout.
fn bind<'a>(what: &str, t: &'a Tensor, contiguous: bool) -> Result<Bound<'a>> {
    let (guard, layout) = t.storage_and_layout();
    if t.dtype() != DType::F32 {
        candle_core::bail!("attn_row: {what} is {:?}, the kernel takes f32", t.dtype());
    }
    if contiguous && !layout.is_contiguous() {
        candle_core::bail!("attn_row: {what} is not contiguous");
    }
    #[cfg(all(target_os = "macos", feature = "metal"))]
    let here = matches!(&*guard, Storage::Metal(_));
    #[cfg(all(target_os = "linux", feature = "cuda"))]
    let here = matches!(&*guard, Storage::Cuda(_));
    if !here {
        candle_core::bail!("attn_row: {what} is not on the device of the kernel");
    }
    let offset = layout.start_offset() * DType::F32.size_in_bytes();
    Ok(Bound { guard, offset })
}

impl CustomOp1 for Launch<'_> {
    fn name(&self) -> &'static str {
        "inkling-attn-row"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("attn_row: GPU only; its CPU reference is AttnWeights::forward")
    }

    #[cfg(all(target_os = "linux", feature = "cuda"))]
    fn cuda_fwd(&self, storage: &CudaStorage, layout: &Layout) -> Result<(CudaStorage, Shape)> {
        if !layout.is_contiguous() {
            candle_core::bail!("attn_row: the query must be contiguous f32");
        }
        let dev = storage.device.clone();
        let stream = dev.cuda_stream();
        let bound = [
            bind("rel", self.x.rel, true)?,
            bind("sinks", self.x.sinks, true)?,
            bind("keys", self.x.keys, false)?,
            bind("values", self.x.values, false)?,
        ];
        let out = unsafe { stream.alloc::<f32>(self.out_elems) }
            .map_err(|e| candle_core::Error::msg(format!("attn_row: alloc of the output: {e}")))?;
        let mut ptrs: Vec<u64> =
            vec![storage.as_cuda_slice::<f32>()?.device_ptr(&stream).0 + 4 * layout.start_offset() as u64];
        for b in &bound {
            let Storage::Cuda(c) = &*b.guard else { unreachable!("checked at bind") };
            ptrs.push(c.as_cuda_slice::<f32>()?.device_ptr(&stream).0 + b.offset as u64);
        }
        ptrs.push(out.device_ptr(&stream).0);
        let cfg = LaunchConfig { grid_dim: (self.nh as u32, 1, 1), block_dim: (TG as u32, 1, 1), shared_mem_bytes: 0 };
        let mut b = stream.launch_builder(self.pipe);
        for p in &ptrs {
            b.arg(p);
        }
        for w in &self.words {
            b.arg(w);
        }
        let probe = crate::inkling::probe_begin(|| stream.synchronize().is_ok());
        crate::inkling::count_launch();
        unsafe { b.launch(cfg) }.map_err(|e| candle_core::Error::msg(format!("attn_row: {e}")))?;
        crate::inkling::probe_end(probe, "attn_row", || stream.synchronize().is_ok());
        drop(bound);
        Ok((CudaStorage::wrap_cuda_slice(out, dev), Shape::from(self.out_elems)))
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn metal_fwd(&self, storage: &MetalStorage, layout: &Layout) -> Result<(MetalStorage, Shape)> {
        if storage.dtype() != DType::F32 || !layout.is_contiguous() {
            candle_core::bail!("attn_row: the query must be contiguous f32");
        }
        let dev = storage.device().clone();
        let bound = [
            bind("rel", self.x.rel, true)?,
            bind("sinks", self.x.sinks, true)?,
            bind("keys", self.x.keys, false)?,
            bind("values", self.x.values, false)?,
        ];
        let out = dev.new_buffer(self.out_elems, DType::F32, "inkling-attn-row")?;
        let enc = dev.command_encoder()?;
        enc.set_compute_pipeline_state(self.pipe);
        enc.set_buffer(0, Some(storage.buffer()), layout.start_offset() * DType::F32.size_in_bytes());
        enc.use_resource(storage.buffer(), MTLResourceUsage::Read);
        for (i, b) in bound.iter().enumerate() {
            enc.set_buffer(i + 1, Some(b.buffer()), b.offset);
            enc.use_resource(b.buffer(), MTLResourceUsage::Read);
        }
        enc.set_buffer(5, Some(&out), 0);
        enc.use_resource(&*out, MTLResourceUsage::Write);
        enc.set_bytes_directly(6, 4 * self.words.len(), self.words.as_ptr() as *const std::ffi::c_void);
        enc.dispatch_thread_groups(
            MTLSize { width: self.nh, height: 1, depth: 1 },
            MTLSize { width: TG, height: 1, depth: 1 },
        );
        drop(enc);
        drop(bound);
        Ok((MetalStorage::new(out, dev, self.out_elems, DType::F32), Shape::from(self.out_elems)))
    }
}
