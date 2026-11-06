//! The split-KV decode-attention kernel on Metal: `kernels/attn_dec.metal`
//! reached from candle, the twin of [`crate::attn_cuda`].
//!
//! Its arithmetic is the canonical one of [`crate::attn::canon`] (the text of
//! `llvq-cuda/src/attn_ref.rs`), so its output is the CPU replay's and the
//! card kernel's, bit for bit. `tests/attn_metal.rs` demands it at the cache
//! lengths that cross every tile, unit and split boundary, and
//! [`MetalAttn::new`] checks it again on a fixture before the kernel is handed
//! out, as the CUDA loader does.
//!
//! Two things differ from the matvec units, both deliberate:
//!   - the library is compiled with `MTLMathMode::Safe` and precise functions.
//!     The matvec units take Apple's fast-math defaults, under which a
//!     division may be approximate and the compiler may assume no infinity;
//!     this arithmetic divides once and tests `m == -inf`;
//!   - the combine is a second dispatch, not the CUDA kernel's last-CTA
//!     ticket. The mapping is bit-neutral, so the two kernels still agree.
//!
//! Serving this form on Metal is the operator's decision: `served.rs` refuses
//! attention "kernel" on Metal by name, and `bin/metalrun` reaches it as a
//! measurement arm only.

use std::sync::Arc;

use candle_core::backend::BackendStorage;
use candle_core::{CpuStorage, CustomOp3, DType, Layout, MetalStorage, Result, Shape, Tensor};
use candle_metal_kernels::metal::ComputePipeline;

use crate::attn::{canon, Call};
use crate::device::AttnKernel;

const SRC: &str = include_str!("../kernels/attn_dec.metal");
/// Threads of a split threadgroup: four groups of 16 lanes, one unit each.
const SPLIT_THREADS: usize = 64;
/// The unit size this kernel compiles (`ATTN_UNIT` in the MSL).
const UNIT: usize = 64;

/// `AttnArgs` of the MSL, field for field.
#[repr(C)]
#[derive(Clone, Copy)]
struct Args {
    n_heads: u32,
    groups: u32,
    rows: u32,
    pos0: u32,
    nsplit: u32,
    k_hs: u32,
    k_ps: u32,
    v_hs: u32,
    v_ps: u32,
    c: f32,
}

/// The two pipelines and the score scale.
pub struct MetalAttn {
    split: ComputePipeline,
    merge: ComputePipeline,
    c: f32,
}

fn compile(dev: &candle_core::MetalDevice) -> Result<(ComputePipeline, ComputePipeline)> {
    use objc2_metal::{MTLCompileOptions, MTLMathFloatingPointFunctions, MTLMathMode};
    let raw = dev.device().clone();
    let built = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let opts = MTLCompileOptions::new();
        opts.setMathMode(MTLMathMode::Safe);
        opts.setMathFloatingPointFunctions(MTLMathFloatingPointFunctions::Precise);
        let lib = raw.new_library_with_source(SRC, Some(&opts))?;
        let s = raw.new_compute_pipeline_state_with_function(&lib.get_function("attn_split_metal", None)?)?;
        let m = raw.new_compute_pipeline_state_with_function(&lib.get_function("attn_merge_metal", None)?)?;
        Ok::<_, candle_metal_kernels::MetalKernelError>((s, m))
    }));
    match built {
        Ok(Ok(p)) => Ok(p),
        Ok(Err(e)) => candle_core::bail!("attn_dec.metal: Metal refused the source: {e:?}"),
        Err(panic) => {
            let what = panic
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_else(|| "no diagnostics".into());
            candle_core::bail!("attn_dec.metal did not compile: {what}")
        }
    }
}

impl MetalAttn {
    /// Compile the kernel and check it against the CPU replay on a fixture of
    /// `n_kv` KV heads and `groups` query heads each: a decode row and a
    /// verify call of 8 rows over 777 positions, and a decode row over 64.
    /// A kernel that returns a different bit is refused, not served.
    pub fn new(device: &candle_core::Device, n_kv: usize, groups: usize) -> std::result::Result<Self, String> {
        let candle_core::Device::Metal(dev) = device else {
            return Err(format!("the Metal attention kernel wants a Metal device, got {device:?}"));
        };
        let (split, merge) = compile(dev).map_err(|e| e.to_string())?;
        let k = Self { split, merge, c: canon::score_scale(canon::HD) };
        k.self_check(device, n_kv, groups).map_err(|e| format!("attn_dec.metal self-check: {e}"))?;
        Ok(k)
    }

    fn self_check(&self, device: &candle_core::Device, n_kv: usize, groups: usize) -> Result<()> {
        for (len, rows, seed) in [(777usize, 8usize, 0x5EED_A77Eu64), (64, 1, 0x5EED_A77F)] {
            let want = reference(n_kv, groups, len, rows, seed);
            let got = self.attend_fixture(device, n_kv, groups, len, rows, seed)?;
            if let Some(i) = got.iter().zip(&want).position(|(a, b)| a != b) {
                candle_core::bail!(
                    "len {len}, rows {rows}: output {i} is {:#06x} against the replay's {:#06x}",
                    got[i],
                    want[i]
                );
            }
        }
        Ok(())
    }

    /// The kernel on the deterministic fixture of [`fixture`], as f16 bits.
    pub fn attend_fixture(
        &self,
        device: &candle_core::Device,
        n_kv: usize,
        groups: usize,
        len: usize,
        rows: usize,
        seed: u64,
    ) -> Result<Vec<u16>> {
        let (q, k, v) = fixture(device, n_kv, groups, len, rows, seed)?;
        bits(&self.attend(&q, &k, &v)?)
    }
}

fn bits(t: &Tensor) -> Result<Vec<u16>> {
    Ok(t.flatten_all()?.to_vec1::<half::f16>()?.into_iter().map(|h| h.to_bits()).collect())
}

/// Deterministic q, k, v (`canon::fill`), k and v as views of a cache with
/// spare capacity, as the Grow store hands them: `[1, n_kv, len, 128]` with a
/// head stride of `cap * 128`.
pub fn fixture(
    device: &candle_core::Device,
    n_kv: usize,
    groups: usize,
    len: usize,
    rows: usize,
    seed: u64,
) -> Result<(Tensor, Tensor, Tensor)> {
    let (hd, nh, cap) = (canon::HD, n_kv * groups, len + 37);
    let to = |b: Vec<u16>| b.into_iter().map(half::f16::from_bits).collect::<Vec<_>>();
    let q = Tensor::from_vec(to(canon::fill(nh * rows * hd, seed, 2.0)), (1, nh, rows, hd), device)?;
    let k = Tensor::from_vec(to(canon::fill(n_kv * cap * hd, seed ^ 0x6b, 2.0)), (1, n_kv, cap, hd), device)?;
    let v = Tensor::from_vec(to(canon::fill(n_kv * cap * hd, seed ^ 0x76, 1.0)), (1, n_kv, cap, hd), device)?;
    Ok((q, k.narrow(2, 0, len)?, v.narrow(2, 0, len)?))
}

/// The CPU replay of the same fixture.
pub fn reference(n_kv: usize, groups: usize, len: usize, rows: usize, seed: u64) -> Vec<u16> {
    let (q, k, v) = fixture(&candle_core::Device::Cpu, n_kv, groups, len, rows, seed).expect("host tensors");
    let host = |t: &Tensor| bits(&t.contiguous().expect("contiguous")).expect("bits");
    let (kb, vb) = (host(&k), host(&v));
    let kv = canon::Kv { k: &kb, v: &vb, cap: len };
    canon::attend(&host(&q), &kv, n_kv, groups, len - rows, rows, canon::score_scale(canon::HD), UNIT)
}

impl AttnKernel for MetalAttn {
    fn name(&self) -> &str {
        "attn_dec.metal"
    }

    fn max_rows(&self) -> usize {
        canon::KMAX
    }

    fn attend(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let call = Call::of(q, k, v)?;
        let q = q.contiguous()?;
        q.apply_op3_no_bwd(k, v, &AttnOp { kern: self, call })
    }
}

struct AttnOp<'a> {
    kern: &'a MetalAttn,
    call: Call,
}

/// The head and position strides of a `[1, n_kv, len, 128]` cache view, in
/// elements, and its start offset. The kernel reads dims contiguously.
fn view(name: &str, l: &Layout) -> Result<(usize, usize, usize)> {
    let st = l.stride();
    if st.len() != 4 || st[3] != 1 {
        candle_core::bail!("attention kernel: {name} strides {st:?}; the last dim must be contiguous");
    }
    Ok((st[1], st[2], l.start_offset()))
}

impl CustomOp3 for AttnOp<'_> {
    fn name(&self) -> &'static str {
        "attn-dec-metal"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("attn_dec.metal has no CPU arm; the CPU device takes attn::CpuAttn")
    }

    fn metal_fwd(
        &self,
        qs: &MetalStorage,
        ql: &Layout,
        ks: &MetalStorage,
        kl: &Layout,
        vs: &MetalStorage,
        vl: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        let c = &self.call;
        if !ql.is_contiguous() {
            candle_core::bail!("attention kernel: q must be contiguous");
        }
        let (k_hs, k_ps, k_off) = view("k", kl)?;
        let (v_hs, v_ps, v_off) = view("v", vl)?;
        let nsplit = c.len.div_ceil(canon::SPLIT);
        let args = Args {
            n_heads: c.n_heads as u32,
            groups: c.groups as u32,
            rows: c.l as u32,
            pos0: c.pos0 as u32,
            nsplit: nsplit as u32,
            k_hs: k_hs as u32,
            k_ps: k_ps as u32,
            v_hs: v_hs as u32,
            v_ps: v_ps as u32,
            c: self.kern.c,
        };
        let dev = qs.device().clone();
        let records = c.n_heads * c.l * nsplit * canon::PSTRIDE;
        let part = dev.new_buffer(records, DType::F32, "attn-dec-part")?;
        let out = dev.new_buffer(c.n_heads * c.l * canon::HD, DType::F16, "attn-dec-out")?;
        let h = DType::F16.size_in_bytes();
        use candle_metal_kernels::utils::set_param;
        {
            let enc = dev.command_encoder()?;
            enc.set_compute_pipeline_state(&self.kern.split);
            set_param(&enc, 0, (qs.buffer(), ql.start_offset() * h));
            set_param(&enc, 1, (ks.buffer(), k_off * h));
            set_param(&enc, 2, (vs.buffer(), v_off * h));
            set_param(&enc, 3, (&*part, 0usize));
            enc.set_bytes(4, &args);
            for b in [qs.buffer(), ks.buffer(), vs.buffer()] {
                enc.use_resource(b, objc2_metal::MTLResourceUsage::Read);
            }
            enc.use_resource(&*part, objc2_metal::MTLResourceUsage::Write);
            enc.dispatch_thread_groups(
                objc2_metal::MTLSize { width: nsplit, height: c.n_heads, depth: c.l },
                objc2_metal::MTLSize { width: SPLIT_THREADS, height: 1, depth: 1 },
            );
        }
        {
            let enc = dev.command_encoder()?;
            enc.set_compute_pipeline_state(&self.kern.merge);
            set_param(&enc, 0, (&*part, 0usize));
            set_param(&enc, 1, (&*out, 0usize));
            enc.set_bytes(2, &args);
            enc.use_resource(&*part, objc2_metal::MTLResourceUsage::Read);
            enc.use_resource(&*out, objc2_metal::MTLResourceUsage::Write);
            enc.dispatch_thread_groups(
                objc2_metal::MTLSize { width: c.n_heads, height: c.l, depth: 1 },
                objc2_metal::MTLSize { width: canon::LANES, height: 1, depth: 1 },
            );
        }
        Ok((
            MetalStorage::new(out, dev, c.n_heads * c.l * canon::HD, DType::F16),
            Shape::from(vec![1, c.n_heads, c.l, canon::HD]),
        ))
    }
}

/// The kernel for a Metal device, checked before it is handed out.
pub fn attn_kernel(
    device: &candle_core::Device,
    n_kv: usize,
    groups: usize,
) -> std::result::Result<Arc<dyn AttnKernel>, String> {
    Ok(Arc::new(MetalAttn::new(device, n_kv, groups)?))
}
