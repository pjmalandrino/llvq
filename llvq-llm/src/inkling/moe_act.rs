//! The activation of the fused one-row MoE, in one Metal launch.
//!
//! Between its gate/up launch and its down rotation, the fused MoE of
//! `resident::ResidentExperts` runs the softcapped SwiGLU of the reference
//! (lines 760-771) as candle operations on the routed rows:
//! `softcap(gate) * sigmoid(gate) * softcap(up)`, then a cast to f16 for the
//! rotation. That is eleven launches a call (two scalings and a tanh a
//! softcap, a sigmoid, two products, a cast), 48 calls a token.
//! [`MoeAct::forward`] does the same arithmetic in one launch and writes the
//! f16 rows the rotation reads.
//!
//! Exactness: meant to be candle's bits, and gated as such
//! (`tests/inkling_moe_act.rs`), not assumed. Each step is written as
//! candle's own kernel writes it (candle-metal-kernels 0.9.2):
//!
//!   - a scaling is `fma(x, mul, add)` with `add = 0` passed at run time
//!     (`affine.metal`), `mul` the f32 of `1 / cap` then of `cap`;
//!   - the tanh is `precise::tanh` (`unary.metal`);
//!   - the sigmoid is `1 / (1 + exp(-x))` (`unary.metal`), with the fast
//!     `exp` and the fast division candle's library is compiled with;
//!   - a product is one f32 multiply, in the reference's order,
//!     `(softcap(gate) * sigmoid(gate)) * softcap(up)`;
//!   - the cast is the device's f32 to f16 conversion.
//!
//! The library is compiled in the safe math mode, so the compiler keeps
//! every multiply where it is written; the fast functions candle gets from
//! its compile options are called by name here. If candle is told to compile
//! without fast math (`CANDLE_METAL_ENABLE_FAST_MATH=0`) its sigmoid changes
//! and this kernel no longer mirrors it: [`MoeAct::new`] then refuses, and
//! the caller keeps candle's operations.
//!
//! Same candle traps as `crate::fused_metal` (its header): the compile runs
//! under `catch_unwind`; every output word is written by the kernel.
#![cfg(all(target_os = "macos", feature = "metal"))]

use candle_core::backend::BackendStorage;
use candle_core::{CpuStorage, CustomOp1, DType, Device, Layout, MetalStorage, Result, Shape, Tensor};
use candle_metal_kernels::metal::ComputePipeline;
use objc2_metal::{MTLResourceUsage, MTLSize};

/// Threads a threadgroup of the elementwise grid.
const GROUP: usize = 256;

/// The kernels. `moe_act` is the one the MoE runs; `moe_act_probe` writes
/// the three factors apart, for the gate that compares each with candle's.
pub const SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct ActArgs { uint n; float inv_cap_g; float cap_g; float inv_cap_u; float cap_u; float zero; };

// softcap as candle's three launches run it: affine(1 / cap, 0), tanh, affine(cap, 0).
static float softcap(float x, float inv_cap, float cap, float zero) {
    float a = fma(x, inv_cap, zero);
    float t = precise::tanh(a);
    return fma(t, cap, zero);
}

// candle's sigmoid, recip(1 + exp(-x)), with the fast functions of its library.
static float sigm(float x) {
    float e = fast::exp(-x);
    float d = 1.0f + e;
    return fast::divide(1.0f, d);
}

// gu: gate [n] then up [n]. out: f16 [n].
kernel void moe_act(
    device const float *gu [[buffer(0)]],
    device half *out [[buffer(1)]],
    constant ActArgs &a [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= a.n) { return; }
    float g = gu[i];
    float u = gu[a.n + i];
    float sg = softcap(g, a.inv_cap_g, a.cap_g, a.zero);
    float s = sigm(g);
    float su = softcap(u, a.inv_cap_u, a.cap_u, a.zero);
    float p = sg * s;
    float act = p * su;
    out[i] = half(act);
}

// x: [n]. out: softcap_gate(x) [n], sigmoid(x) [n], softcap_up(x) [n].
kernel void moe_act_probe(
    device const float *x [[buffer(0)]],
    device float *out [[buffer(1)]],
    constant ActArgs &a [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= a.n) { return; }
    float v = x[i];
    out[i] = softcap(v, a.inv_cap_g, a.cap_g, a.zero);
    out[a.n + i] = sigm(v);
    out[2u * a.n + i] = softcap(v, a.inv_cap_u, a.cap_u, a.zero);
}
"#;

/// Whether candle compiles its Metal kernels with fast math, as
/// `candle_metal_kernels` reads it (on unless the variable says otherwise).
pub fn candle_fast_math() -> bool {
    match std::env::var("CANDLE_METAL_ENABLE_FAST_MATH") {
        Ok(v) => matches!(v.as_str(), "true" | "t" | "yes" | "y" | "1"),
        Err(_) => true,
    }
}

/// The compiled kernels.
pub struct MoeAct {
    act: ComputePipeline,
    probe: ComputePipeline,
}

impl MoeAct {
    /// Compile [`SRC`] for a Metal device. Refuses when candle is not
    /// compiling with fast math: the sigmoid here mirrors the fast one.
    pub fn new(device: &Device) -> Result<Self> {
        if !candle_fast_math() {
            candle_core::bail!("moe_act mirrors candle's fast-math sigmoid, and CANDLE_METAL_ENABLE_FAST_MATH turns it off");
        }
        Self::with_source(device, SRC)
    }

    /// Compile `src`, which must define `moe_act` and `moe_act_probe` as
    /// [`SRC`] does. For the gate that tries a variant of the source.
    pub fn with_source(device: &Device, src: &str) -> Result<Self> {
        let Device::Metal(dev) = device else { candle_core::bail!("moe_act wants a Metal device") };
        use objc2_metal::{MTLCompileOptions, MTLMathFloatingPointFunctions, MTLMathMode};
        let raw = dev.device().clone();
        let built = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let opts = MTLCompileOptions::new();
            opts.setMathMode(MTLMathMode::Safe);
            opts.setMathFloatingPointFunctions(MTLMathFloatingPointFunctions::Precise);
            let lib = raw.new_library_with_source(src, Some(&opts))?;
            ["moe_act", "moe_act_probe"]
                .iter()
                .map(|n| raw.new_compute_pipeline_state_with_function(&lib.get_function(n, None)?))
                .collect::<std::result::Result<Vec<_>, _>>()
        }));
        match built {
            Ok(Ok(p)) => {
                let mut p = p.into_iter();
                let mut next = || p.next().expect("two pipelines");
                Ok(Self { act: next(), probe: next() })
            }
            Ok(Err(e)) => candle_core::bail!("moe_act: Metal refused the source: {e:?}"),
            Err(panic) => {
                let what = panic
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_else(|| "no diagnostics".into());
                candle_core::bail!("moe_act: the MSL did not compile: {what}")
            }
        }
    }

    /// `softcap(gate, cap_gate) * sigmoid(gate) * softcap(up, cap_up)` as
    /// f16, for `gu` f32 `[2, rows, inter]` contiguous on the Metal device
    /// (gate rows, then up rows): `[rows, inter]`.
    pub fn forward(&self, gu: &Tensor, cap_gate: f64, cap_up: f64) -> Result<Tensor> {
        let (two, rows, inter) = gu.dims3()?;
        if two != 2 || rows * inter == 0 {
            candle_core::bail!("moe_act: gate and up as {:?}, want [2, rows, inter]", gu.dims());
        }
        let n = rows * inter;
        let op = Launch {
            name: "inkling-moe-act",
            pipe: &self.act,
            words: words(n, cap_gate, cap_up)?,
            n,
            out_elems: n,
            out_dtype: DType::F16,
            out_shape: Shape::from((rows, inter)),
        };
        gu.apply_op1_no_bwd(&op)
    }

    /// The three factors of [`Self::forward`] for the values `x` (f32, any
    /// shape, contiguous): `[3, n]` f32, the gate's softcap, the sigmoid and
    /// the up's softcap of each.
    pub fn probe(&self, x: &Tensor, cap_gate: f64, cap_up: f64) -> Result<Tensor> {
        let n = x.elem_count();
        if n == 0 {
            candle_core::bail!("moe_act: no value to probe");
        }
        let op = Launch {
            name: "inkling-moe-act-probe",
            pipe: &self.probe,
            words: words(n, cap_gate, cap_up)?,
            n,
            out_elems: 3 * n,
            out_dtype: DType::F32,
            out_shape: Shape::from((3, n)),
        };
        x.apply_op1_no_bwd(&op)
    }
}

/// `ActArgs`: n, 1 / cap_gate, cap_gate, 1 / cap_up, cap_up, zero. The
/// scalings narrow their f64 factor to f32 as candle's affine does.
fn words(n: usize, cap_gate: f64, cap_up: f64) -> Result<Vec<u32>> {
    let n = u32::try_from(n).map_err(|_| candle_core::Error::Msg(format!("moe_act: {n} values > u32")))?;
    let f = |v: f64| (v as f32).to_bits();
    Ok(vec![n, f(1.0 / cap_gate), f(cap_gate), f(1.0 / cap_up), f(cap_up), 0f32.to_bits()])
}

struct Launch<'a> {
    name: &'static str,
    pipe: &'a ComputePipeline,
    words: Vec<u32>,
    n: usize,
    out_elems: usize,
    out_dtype: DType,
    out_shape: Shape,
}

impl CustomOp1 for Launch<'_> {
    fn name(&self) -> &'static str {
        self.name
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("{}: Metal only; its reference is candle's operations", self.name)
    }

    fn metal_fwd(&self, storage: &MetalStorage, layout: &Layout) -> Result<(MetalStorage, Shape)> {
        if storage.dtype() != DType::F32 || !layout.is_contiguous() {
            candle_core::bail!("{}: the input must be contiguous f32", self.name);
        }
        let dev = storage.device().clone();
        let out = dev.new_buffer(self.out_elems, self.out_dtype, self.name)?;
        let enc = dev.command_encoder()?;
        enc.set_compute_pipeline_state(self.pipe);
        enc.set_buffer(0, Some(storage.buffer()), layout.start_offset() * DType::F32.size_in_bytes());
        enc.use_resource(storage.buffer(), MTLResourceUsage::Read);
        enc.set_buffer(1, Some(&out), 0);
        enc.use_resource(&*out, MTLResourceUsage::Write);
        enc.set_bytes_directly(2, 4 * self.words.len(), self.words.as_ptr() as *const std::ffi::c_void);
        enc.dispatch_threads(
            MTLSize { width: self.n, height: 1, depth: 1 },
            MTLSize { width: GROUP, height: 1, depth: 1 },
        );
        drop(enc);
        Ok((MetalStorage::new(out, dev, self.out_elems, self.out_dtype), self.out_shape.clone()))
    }
}
