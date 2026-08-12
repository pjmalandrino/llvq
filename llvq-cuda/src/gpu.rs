//! cudarc plumbing: compile, load, upload, launch, and read back what the
//! card says about the kernel it just loaded.
//!
//! Four API traps are documented in `docs/archive/portage-noyau-cuda.md` §2.2, and
//! two of them are load-bearing here:
//!
//! * without an explicit `arch`, NVRTC compiles for `compute_75` by default,
//!   silently. The guard is not to trust the option but to read
//!   `binary_version()` back off the loaded function and assert it.
//! * `use_fast_math: Some(true)` only emits `--fmad=true`, which is NVRTC's
//!   default anyway. The field is a no-op in both directions; anything real
//!   goes through `options`.

use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig};
use cudarc::driver::sys::CUdevice_attribute as Attr;
use cudarc::driver::PushKernelArg;
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use std::sync::Arc;

/// Compute capability the image is built for, frozen in `ops/Dockerfile.cuda`
/// as `CUDA_COMPUTE_CAP=89`. An L40S reports 8.9; `l4x1`, `a100`, `a10g` and
/// `t4` cannot load code built for it, and `ops/run.py` refuses them before
/// billing anything.
pub const ARCH: &str = "compute_89";
const ARCH_BINARY_VERSION: i32 = 89;

/// What the card and the loaded kernel say about themselves.
///
/// Every field is *read*, never assumed. The repository has already retracted
/// one published figure — a "93 % of peak" built on a peak nobody measured —
/// and `llvq-metal` sets the precedent by reading the SIMD width rather than
/// hard-coding 32.
#[derive(Debug)]
pub struct DeviceReport {
    pub name: String,
    pub compute_cap: (i32, i32),
    pub sm_count: i32,
    /// Bytes. The cache-defeat argument depends on this, and third-party
    /// sources disagree on it for the L40S (48 MB for the SKU, 96 for a full
    /// AD102). NVIDIA does not publish it; the driver does.
    pub l2_bytes: i32,
    pub shared_per_block: i32,
    pub shared_per_sm: i32,
    pub clock_khz: i32,
    pub mem_clock_khz: i32,
    pub mem_bus_bits: i32,
}

impl DeviceReport {
    /// Peak DRAM bandwidth in GB/s, from the clock and bus width the driver
    /// reports. DDR, hence the factor 2.
    ///
    /// A *nameplate* number, not a measurement: ECC is on by default on an
    /// L40S and costs some of it. Any statement of the form "x % of peak"
    /// must use the floor kernel, never this.
    pub fn nameplate_bandwidth_gbs(&self) -> f64 {
        2.0 * (self.mem_clock_khz as f64) * 1e3 * (self.mem_bus_bits as f64 / 8.0) / 1e9
    }
}

/// Attributes of one loaded kernel.
#[derive(Debug)]
pub struct FnReport {
    pub name: String,
    pub num_regs: i32,
    /// Bytes of local memory. **Zero is the contract.** Anything else means
    /// `slot_dot`'s four accumulators became a dynamically indexed array and
    /// spilled onto the hottest path — the exact failure `#pragma unroll` is
    /// there to prevent, and one that no correctness test can see.
    pub local_bytes: i32,
    pub shared_bytes: i32,
    pub binary_version: i32,
    pub max_threads: i32,
}

/// The kernel text actually handed to NVRTC, and its fingerprint.
pub struct KernelSource {
    pub text: String,
    pub sha256: String,
}

impl KernelSource {
    /// Concatenate, in the order the compiler will see them.
    pub fn new(parts: &[&str]) -> Self {
        let text = parts.join("\n");
        let sha256 = sha256_hex(text.as_bytes());
        Self { text, sha256 }
    }
}

/// SHA-256, written out rather than pulled in.
///
/// `llvq-core`, `llvq-search` and `llvq-artifact` have no external
/// dependencies, and a benchmark crate that reads a sealed artifact should not
/// be the one to break that habit for forty lines of hashing.
fn sha256_hex(data: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let mut msg = data.to_vec();
    let bitlen = (data.len() as u64) * 8;
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bitlen.to_be_bytes());

    for chunk in msg.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (i, word) in w.iter_mut().enumerate().take(16) {
            let b = &chunk[i * 4..i * 4 + 4];
            *word = u32::from_be_bytes([b[0], b[1], b[2], b[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let mut v = h;
        for i in 0..64 {
            let s1 = v[4].rotate_right(6) ^ v[4].rotate_right(11) ^ v[4].rotate_right(25);
            let ch = (v[4] & v[5]) ^ ((!v[4]) & v[6]);
            let t1 = v[7]
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = v[0].rotate_right(2) ^ v[0].rotate_right(13) ^ v[0].rotate_right(22);
            let maj = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
            let t2 = s0.wrapping_add(maj);
            v = [
                t1.wrapping_add(t2),
                v[0],
                v[1],
                v[2],
                v[3].wrapping_add(t1),
                v[4],
                v[5],
                v[6],
            ];
        }
        for (hi, vi) in h.iter_mut().zip(v) {
            *hi = hi.wrapping_add(vi);
        }
    }
    h.iter().map(|x| format!("{x:08x}")).collect()
}

/// A context, a stream, and one compiled module.
pub struct Cuda {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
}

impl Cuda {
    /// Open device 0 and compile `src` for [`ARCH`].
    pub fn new(src: &KernelSource) -> Result<Self, String> {
        let ctx = CudaContext::new(0).map_err(|e| format!("no CUDA device: {e}"))?;
        let stream = ctx.default_stream();
        let opts = CompileOptions {
            // `arch` is `Option<&'static str>`; a value computed at runtime
            // would have to go through `options`, which NVRTC unrolls *after*
            // `--gpu-architecture=` and would therefore override it.
            arch: Some(ARCH),
            ..Default::default()
        };
        let ptx = compile_ptx_with_opts(&src.text, opts)
            .map_err(|e| format!("NVRTC refused the kernel source:\n{e}"))?;
        let module = ctx
            .load_module(ptx)
            .map_err(|e| format!("the driver refused the PTX: {e}"))?;
        Ok(Self { ctx, stream, module })
    }

    /// Open device 0 on a **fresh, non-default** stream.
    ///
    /// [`Self::new`] takes `ctx.default_stream()`, which is the legacy NULL
    /// stream — and the driver refuses to capture that one into a graph
    /// (`CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`). So a graph arm needs this,
    /// and switching streams **changes the object being measured**: the NULL
    /// stream has implicit synchronisation semantics against every other
    /// stream in the context, a fresh one does not.
    ///
    /// Hence the three-arm control the audit asks for — legacy, fresh, fresh +
    /// graph — in one job. Comparing a graph arm on a fresh stream against a
    /// published number taken on the legacy one would credit the graph with
    /// whatever the stream change is worth.
    pub fn new_on_fresh_stream(src: &KernelSource) -> Result<Self, String> {
        let ctx = CudaContext::new(0).map_err(|e| format!("no CUDA device: {e}"))?;
        // Event tracking OFF, and before a single allocation.
        //
        // Creating a second stream puts cudarc in *multi-stream mode*, where
        // every `arg()` on a `CudaSlice` pushes a `cuStreamWaitEvent` and a
        // `cuEventRecord` around the launch to order accesses for you
        // (`launch.rs:100-135`). Two consequences, both measured on the L40S on
        // 2026-08-06:
        //
        //  * it costs — the fresh-stream arm came out at 3.66 µs/launch against
        //    3.61 for the legacy one, slower, and those events are the only
        //    difference;
        //  * it makes a graph uncapturable — recording an event that is not
        //    part of the graph invalidates the capture, which surfaces as
        //    `CUDA_ERROR_STREAM_CAPTURE_INVALIDATED` at `end_capture` with the
        //    real cause already swallowed.
        //
        // Turning it off hands stream synchronisation back to us. That is safe
        // here and only here: this context drives **one** stream, so there is
        // no cross-stream ordering left to get wrong. Anything that later
        // creates a second stream on this context must revisit it.
        //
        // The call must precede every allocation — cudarc only skips the event
        // pair for slices *created after* it.
        // `unsafe` because cudarc can no longer prove accesses are ordered:
        // the caller takes that on. Here the proof is structural — one stream,
        // and the driver orders a single stream by issue.
        unsafe { ctx.disable_event_tracking() };
        let stream = ctx
            .new_stream()
            .map_err(|e| format!("new_stream: {e}"))?;
        Self::on_stream(stream, src)
    }

    /// Capture whatever `body` launches into a replayable graph.
    ///
    /// The capture is `Relaxed`: `ThreadLocal` would refuse any launch this
    /// thread did not make, and nothing here launches from elsewhere.
    ///
    /// Returns `None` when the driver captured nothing — which is what a
    /// legacy NULL stream produces rather than an error, and is exactly the
    /// silent failure this wrapper exists to surface.
    pub fn capture(
        &self,
        body: impl FnOnce() -> Result<(), String>,
    ) -> Result<cudarc::driver::CudaGraph, String> {
        self.stream
            .begin_capture(cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
            .map_err(|e| format!("begin_capture: {e}"))?;
        // A failure inside `body` leaves the stream in capture mode, so the
        // capture is closed on both paths before the error is returned.
        let r = body();
        let ended = self
            .stream
            .end_capture(cudarc::driver::sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH);
        // The body's error first. `end_capture` fails *because* of it —
        // `CUDA_ERROR_STREAM_CAPTURE_INVALIDATED` means "a previous operation
        // failed", and reporting that instead swallows the cause. Cost of
        // getting this wrong: one billed job that says nothing useful.
        r?;
        let g = ended.map_err(|e| format!("end_capture: {e}"))?;
        let g = g.ok_or_else(|| {
            "le driver n'a rien capturé — stream NULL legacy ? (cf. new_on_fresh_stream)"
                .to_string()
        })?;
        g.upload().map_err(|e| format!("graph upload: {e}"))?;
        Ok(g)
    }

    /// Compile `src` onto an **existing** stream — candle's, in practice.
    ///
    /// [`Self::new`] opens device 0 and takes `ctx.default_stream()`, which is
    /// the right thing for a bench that owns the card. Inside an inference
    /// runtime it is not: candle allocates its tensors on its own stream, and
    /// launching our kernels on a different one would order them against
    /// candle's work only by accident. Sharing the stream makes the ordering
    /// the stream's own guarantee rather than something we have to remember.
    ///
    /// (Both end up on the same primary context, so the pointers would have
    /// been valid either way — it is the *ordering* this buys, not validity.)
    pub fn on_stream(stream: Arc<CudaStream>, src: &KernelSource) -> Result<Self, String> {
        let ctx = stream.context().clone();
        let opts = CompileOptions {
            arch: Some(ARCH),
            ..Default::default()
        };
        let ptx = compile_ptx_with_opts(&src.text, opts)
            .map_err(|e| format!("NVRTC refused the kernel source:\n{e}"))?;
        let module = ctx
            .load_module(ptx)
            .map_err(|e| format!("the driver refused the PTX: {e}"))?;
        Ok(Self { ctx, stream, module })
    }

    pub fn device(&self) -> Result<DeviceReport, String> {
        let a = |x: Attr| self.ctx.attribute(x).map_err(|e| format!("attribute: {e}"));
        Ok(DeviceReport {
            name: self.ctx.name().map_err(|e| format!("device name: {e}"))?,
            compute_cap: (
                a(Attr::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?,
                a(Attr::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?,
            ),
            sm_count: a(Attr::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?,
            l2_bytes: a(Attr::CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE)?,
            shared_per_block: a(Attr::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)?,
            shared_per_sm: a(Attr::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR)?,
            clock_khz: a(Attr::CU_DEVICE_ATTRIBUTE_CLOCK_RATE)?,
            mem_clock_khz: a(Attr::CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE)?,
            mem_bus_bits: a(Attr::CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH)?,
        })
    }

    pub fn func(&self, name: &str) -> Result<CudaFunction, String> {
        self.module
            .load_function(name)
            .map_err(|e| format!("no kernel {name}: {e}"))
    }

    /// Read back what the driver made of a kernel, and refuse the two states
    /// that would invalidate every number taken afterwards.
    pub fn report(&self, name: &str) -> Result<FnReport, String> {
        let f = self.func(name)?;
        let g = |r: Result<i32, cudarc::driver::DriverError>| {
            r.map_err(|e| format!("{name}: {e}"))
        };
        let rep = FnReport {
            name: name.to_string(),
            num_regs: g(f.num_regs())?,
            local_bytes: g(f.local_size_bytes())?,
            shared_bytes: g(f.shared_size_bytes())?,
            binary_version: g(f.binary_version())?,
            max_threads: g(f.max_threads_per_block())?,
        };
        if rep.binary_version != ARCH_BINARY_VERSION {
            return Err(format!(
                "{name}: compiled for sm_{}, not sm_{ARCH_BINARY_VERSION}. NVRTC falls back to \
                 compute_75 when no architecture is given, silently — nothing measured on this \
                 module would describe the card.",
                rep.binary_version
            ));
        }
        Ok(rep)
    }

    pub fn up_u32(&self, v: &[u32]) -> Result<CudaSlice<u32>, String> {
        self.stream.clone_htod(v).map_err(|e| format!("H2D u32: {e}"))
    }

    pub fn up_f32(&self, v: &[f32]) -> Result<CudaSlice<f32>, String> {
        self.stream.clone_htod(v).map_err(|e| format!("H2D f32: {e}"))
    }

    pub fn up_u16(&self, v: &[u16]) -> Result<CudaSlice<u16>, String> {
        self.stream.clone_htod(v).map_err(|e| format!("H2D u16: {e}"))
    }

    pub fn zeros_u32(&self, n: usize) -> Result<CudaSlice<u32>, String> {
        self.stream.alloc_zeros(n).map_err(|e| format!("alloc u32: {e}"))
    }

    pub fn zeros_f32(&self, n: usize) -> Result<CudaSlice<f32>, String> {
        self.stream.alloc_zeros(n).map_err(|e| format!("alloc f32: {e}"))
    }

    /// Sortie en binary16, pour les noyaux qui en écrivent une.
    ///
    /// Aucun bras LLVQ n'en a besoin — ils écrivent tous `float* y`. Le bras
    /// AWQ, lui, écrit `unsigned short* outputs`, parce que c'est ce que fait
    /// le noyau amont et qu'on ne le modifie pas.
    pub fn zeros_u16(&self, n: usize) -> Result<CudaSlice<u16>, String> {
        self.stream.alloc_zeros(n).map_err(|e| format!("alloc u16: {e}"))
    }

    pub fn down_u32(&self, d: &CudaSlice<u32>) -> Result<Vec<u32>, String> {
        self.stream.clone_dtoh(d).map_err(|e| format!("D2H u32: {e}"))
    }

    pub fn down_f32(&self, d: &CudaSlice<f32>) -> Result<Vec<f32>, String> {
        self.stream.clone_dtoh(d).map_err(|e| format!("D2H f32: {e}"))
    }

    /// Le pendant de [`Self::zeros_u16`] : relire une sortie binary16.
    pub fn down_u16(&self, d: &CudaSlice<u16>) -> Result<Vec<u16>, String> {
        self.stream.clone_dtoh(d).map_err(|e| format!("D2H u16: {e}"))
    }

    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    pub fn sync(&self) -> Result<(), String> {
        self.stream.synchronize().map_err(|e| format!("sync: {e}"))
    }

    /// One-dimensional launch over `n` items with `block` threads per block.
    ///
    /// The kernels guard with `if (b >= n) return;` rather than relying on an
    /// exact division. That guard is safe *here* because these probes contain
    /// no `__syncthreads()` and no warp collective; in the fused matvec it
    /// would be both a deadlock and a broken full-warp mask, which is why the
    /// real kernel asserts `d_out % 8 == 0` instead.
    pub fn grid(n: usize, block: u32) -> LaunchConfig {
        LaunchConfig {
            grid_dim: ((n as u32).div_ceil(block), 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        }
    }
}

/// The read-only buffers every probe shares.
///
/// Bundled rather than passed one by one: the kernels take six and seven
/// arguments, and a positional list that long is exactly where a port swaps
/// two pointers of the same type and gets a plausible wrong answer.
pub struct Inputs<'a> {
    pub words: &'a CudaSlice<u32>,
    pub bases: &'a CudaSlice<u32>,
    pub tab: &'a CudaSlice<u32>,
    pub gscale: &'a CudaSlice<f32>,
    pub x: &'a CudaSlice<f32>,
    pub nblocks: u32,
}

/// `decode_probe(words, bases, tab, nblocks, slots_out, cls_out)`
pub fn launch_decode_probe(
    cuda: &Cuda,
    f: &CudaFunction,
    inp: &Inputs,
    slots: &mut CudaSlice<u32>,
    cls: &mut CudaSlice<u32>,
) -> Result<(), String> {
    let cfg = Cuda::grid(inp.nblocks as usize, 256);
    let mut b = cuda.stream().launch_builder(f);
    b.arg(inp.words)
        .arg(inp.bases)
        .arg(inp.tab)
        .arg(&inp.nblocks)
        .arg(slots)
        .arg(cls);
    unsafe { b.launch(cfg) }.map_err(|e| format!("decode_probe: {e}"))?;
    Ok(())
}

/// `dot_probe(words, bases, tab, gscale, x, nblocks, out)`
pub fn launch_dot_probe(
    cuda: &Cuda,
    f: &CudaFunction,
    inp: &Inputs,
    out: &mut CudaSlice<f32>,
) -> Result<(), String> {
    let cfg = Cuda::grid(inp.nblocks as usize, 256);
    let mut b = cuda.stream().launch_builder(f);
    b.arg(inp.words)
        .arg(inp.bases)
        .arg(inp.tab)
        .arg(inp.gscale)
        .arg(inp.x)
        .arg(&inp.nblocks)
        .arg(out);
    unsafe { b.launch(cfg) }.map_err(|e| format!("dot_probe: {e}"))?;
    Ok(())
}

/// One warp per output row, eight rows per block.
///
/// The grid is exact and there is no bounds guard in the kernel: a `return`
/// before `__syncthreads()` deadlocks, and it would break the full-warp mask
/// the reduction relies on. `d_out % 8 == 0` is asserted host-side instead.
impl Cuda {
    #[allow(clippy::too_many_arguments)]
    pub fn launch_slot(
        &self,
        f: &CudaFunction,
        words: &CudaSlice<u32>,
        bases: &CudaSlice<u32>,
        tab: &CudaSlice<u32>,
        gscale: &CudaSlice<f32>,
        rscale: &CudaSlice<f32>,
        tail: &CudaSlice<f32>,
        x: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        nblocks: u32,
        tail_w: u32,
        d_out: u32,
        threads: u32,
        shared: u32,
    ) -> Result<(), String> {
        let cfg = row_grid(d_out, threads, shared);
        let mut b = self.stream.launch_builder(f);
        b.arg(words).arg(bases).arg(tab).arg(gscale).arg(rscale).arg(tail).arg(x).arg(y)
            .arg(&nblocks).arg(&tail_w);
        unsafe { b.launch(cfg) }.map_err(|e| format!("tv_slot: {e}"))?;
        Ok(())
    }

    /// `tv_slot_h` — `tv_slot` storing f16.
    ///
    /// Diagnostics aside, this is the one an inference runtime calls: the
    /// model is f16 end to end, and a f32 result costs a conversion kernel per
    /// projection.
    ///
    /// `y` is generic over its element type for the same reason `launch_rot`'s
    /// input is: an inference runtime hands over candle's own
    /// `CudaSlice<half::f16>`, and this crate has no `half` dependency to name
    /// that type with. The kernel writes `unsigned short`, which is those bits.
    ///
    /// `tail` is `u16` for the same reason and the same encoding: since lot
    /// A7a the f16-storing entry points hold the `KeepExact` tail as binary16,
    /// which is what the dense arm they are diffed against holds it at. The
    /// *bench* arms — `launch_slot`, `launch_slot_seg`, and planesbench's —
    /// keep `f32`, because their published `b/poids noyau` bills 32 bits
    /// there. Two residencies, two accountings, and the type keeps a caller
    /// from handing one to the other.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_slot_h<T: cudarc::driver::DeviceRepr>(
        &self,
        f: &CudaFunction,
        words: &CudaSlice<u32>,
        bases: &CudaSlice<u32>,
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
        let cfg = row_grid(d_out, threads, shared);
        let mut b = self.stream.launch_builder(f);
        b.arg(words).arg(bases).arg(tab).arg(gscale).arg(rscale).arg(tail).arg(x).arg(y)
            .arg(&nblocks).arg(&tail_w);
        unsafe { b.launch(cfg) }.map_err(|e| format!("tv_slot_h: {e}"))?;
        Ok(())
    }

    /// `tv_slot_seg` — the same kernel over a row-concatenation of matrices
    /// that share an input, with one extra table naming each row's centroid
    /// pair.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_slot_seg(
        &self,
        f: &CudaFunction,
        words: &CudaSlice<u32>,
        bases: &CudaSlice<u32>,
        tab: &CudaSlice<u32>,
        gscale: &CudaSlice<f32>,
        gs_off: &CudaSlice<u32>,
        rscale: &CudaSlice<f32>,
        tail: &CudaSlice<f32>,
        x: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        nblocks: u32,
        tail_w: u32,
        d_out: u32,
        threads: u32,
        shared: u32,
    ) -> Result<(), String> {
        let cfg = row_grid(d_out, threads, shared);
        let mut b = self.stream.launch_builder(f);
        b.arg(words).arg(bases).arg(tab).arg(gscale).arg(gs_off).arg(rscale).arg(tail)
            .arg(x).arg(y).arg(&nblocks).arg(&tail_w);
        unsafe { b.launch(cfg) }.map_err(|e| format!("tv_slot_seg: {e}"))?;
        Ok(())
    }

    /// A timing event on this context.
    ///
    /// `new_event(None)` would create it with `CU_EVENT_DISABLE_TIMING`, which
    /// records fine and then fails at `elapsed_ms` — a mistake that costs a
    /// billed job to find, so the flag is passed explicitly here and nowhere
    /// else.
    pub fn new_event(&self) -> Result<cudarc::driver::CudaEvent, String> {
        self.ctx
            .new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|e| format!("event: {e}"))
    }

    /// One of the three floor probes — same shell as `tv_slot`, less work.
    ///
    /// Diagnostics only: none of them computes a matvec, so no ratio may be
    /// quoted against them.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_floor(
        &self,
        f: &CudaFunction,
        words: &CudaSlice<u32>,
        bases: &CudaSlice<u32>,
        tab: &CudaSlice<u32>,
        x: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        nblocks: u32,
        d_out: u32,
        threads: u32,
        shared: u32,
    ) -> Result<(), String> {
        let cfg = row_grid(d_out, threads, shared);
        let mut b = self.stream.launch_builder(f);
        b.arg(words).arg(bases).arg(tab).arg(x).arg(y).arg(&nblocks);
        unsafe { b.launch(cfg) }.map_err(|e| format!("floor probe: {e}"))?;
        Ok(())
    }

    /// `rot_apply` — one block, the whole activation in shared memory.
    ///
    /// The grid is fixed at one block by the kernel's design (a Walsh–Hadamard
    /// transform is `log₂ m` barriers, and CUDA has no barrier across blocks),
    /// so the only launch knob is the thread count. Shared memory is `n`
    /// floats, and the caller is what keeps it under the device limit — the
    /// kernel has no way to check and would simply corrupt.
    ///
    /// `xin` is generic over its element type so an inference runtime can hand
    /// over candle's own `CudaSlice<f16>` without a copy: the kernel reads it
    /// as `unsigned short` and widens with `cvt.f32.f16`, which is exactly
    /// what those bits are. A `u16` staging buffer would be the same bytes,
    /// one allocation and one device-to-device copy later.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_rot<T: cudarc::driver::DeviceRepr>(
        &self,
        f: &CudaFunction,
        xin: &CudaSlice<T>,
        signbits: &CudaSlice<u32>,
        small: &CudaSlice<f32>,
        xout: &mut CudaSlice<f32>,
        n: u32,
        m: u32,
        k: u32,
        inv: f32,
        x_off: u32,
        threads: u32,
    ) -> Result<(), String> {
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: n * 4,
        };
        let mut b = self.stream.launch_builder(f);
        b.arg(xin).arg(signbits).arg(small).arg(xout).arg(&n).arg(&m).arg(&k).arg(&inv)
            .arg(&x_off);
        unsafe { b.launch(cfg) }.map_err(|e| format!("rot_apply: {e}"))?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn launch_f16(
        &self,
        f: &CudaFunction,
        w: &CudaSlice<u16>,
        x: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        d_in: u32,
        d_out: u32,
        threads: u32,
        shared: u32,
    ) -> Result<(), String> {
        let cfg = row_grid(d_out, threads, shared);
        let mut b = self.stream.launch_builder(f);
        b.arg(w).arg(x).arg(y).arg(&d_in);
        unsafe { b.launch(cfg) }.map_err(|e| format!("tv_f16: {e}"))?;
        Ok(())
    }
}

fn row_grid(d_out: u32, threads: u32, shared: u32) -> LaunchConfig {
    assert_eq!(d_out % (threads / 32), 0, "rows must fill whole blocks");
    LaunchConfig {
        grid_dim: (d_out * 32 / threads, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: shared,
    }
}
