//! cudarc plumbing: compile, load, upload, launch, and read back what the
//! card says about the kernel it just loaded.
//!
//! Four API traps are documented in `docs/archive/portage-noyau-cuda.md` §2.2, and
//! two of them are load-bearing here:
//!
//! * without an explicit `arch`, NVRTC compiles for `compute_75` by default,
//!   silently. The guard is not to trust the option but to read
//!   `binary_version()` back off the loaded function and assert it — as an
//!   **order**, `binary >= compiled_for`, since PTX is forward-compatible and
//!   a card newer than the request JITs to its own sm. The predicate lives in
//!   `crate::arch_binary_ok`, outside this `cfg(linux)` module, so the dev
//!   machine can mutate it.
//! * `use_fast_math: Some(true)` only emits `--fmad=true`, which is NVRTC's
//!   default anyway. The field is a no-op in both directions; anything real
//!   goes through `options`.

use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig};
use cudarc::driver::sys::CUdevice_attribute as Attr;
use cudarc::driver::PushKernelArg;
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use std::sync::Arc;

/// Compute capability the NVRTC pass targets: `compute_89` (L40S, the only
/// card every published number ran on) unless `LLVQ_NVRTC_ARCH` overrides
/// it — added 2026-08-18 for the second-architecture point (F4 of the TACO
/// plan). The value is pinned once per process, feeds both the compile and
/// the `binary_version` assert below, and an unparseable value is refused
/// by name rather than silently defaulted — the `LLVQ_FUSED_LAYOUT` rule.
///
/// `CompileOptions::arch` is `Option<&'static str>`, so the override leaks
/// one small string per process, deliberately.
pub fn arch() -> &'static str {
    use std::sync::OnceLock;
    static ONCE: OnceLock<&'static str> = OnceLock::new();
    ONCE.get_or_init(|| match std::env::var("LLVQ_NVRTC_ARCH") {
        Ok(s) => {
            assert!(
                s.strip_prefix("compute_")
                    .and_then(|n| n.parse::<i32>().ok())
                    .is_some(),
                "LLVQ_NVRTC_ARCH={s}: expected `compute_NN` (compute_80, compute_89, …)"
            );
            Box::leak(s.into_boxed_str())
        }
        Err(_) => "compute_89",
    })
}

/// The sm the loaded function must report **at least** — derived from
/// [`arch()`], never a second constant that could drift from it.
fn arch_binary_version() -> i32 {
    arch()
        .strip_prefix("compute_")
        .expect("arch() guarantees the prefix")
        .parse()
        .expect("arch() guarantees the number")
}

/// The primary context, opened once and kept for the process.
///
/// `cuDevicePrimaryCtxRetain` is refcounted, so this and every `Cuda::new`
/// hold the *same* context rather than each creating one. Keeping it here
/// matters for [`probe_compute_cap`]: if the probe's handle were dropped
/// before a kernel was compiled it would be the last one, the driver would
/// destroy the context, and the next constructor would pay to build it again.
///
/// Never released, like the string [`arch`] leaks, and for the same reason —
/// it lives exactly as long as the process that needs it.
fn primary_ctx() -> Result<&'static Arc<CudaContext>, String> {
    use std::sync::OnceLock;
    static ONCE: OnceLock<Arc<CudaContext>> = OnceLock::new();
    if let Some(c) = ONCE.get() {
        return Ok(c);
    }
    let ctx = CudaContext::new(0).map_err(|e| format!("no CUDA device: {e}"))?;
    // A race here is harmless: `get_or_init` keeps one handle, the loser's
    // `Arc` drops and releases its own retain, and the refcount balances.
    Ok(ONCE.get_or_init(|| ctx))
}

/// The compute capability of a context's device. The single reader of those
/// two attributes: [`DeviceReport`] goes through it too, so a report and a
/// probe cannot disagree about the card they describe.
pub fn compute_cap_of(ctx: &CudaContext) -> Result<(i32, i32), String> {
    let a = |x: Attr| ctx.attribute(x).map_err(|e| format!("attribute: {e}"));
    Ok((
        a(Attr::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?,
        a(Attr::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?,
    ))
}

/// The card's capability, **before** any kernel source exists.
///
/// `Cuda::new` compiles NVRTC inside its constructor, and since 2026-09-10 the
/// tile is a `#define` in that source ([`crate::tile`]). So the capability has
/// to be readable before the text is assembled, which `Cuda::device()` — a
/// method on the already-compiled object — cannot do. A caller that resolves
/// its tile from this and then compiles is reading one card and describing the
/// same one; there is no second device to confuse it with, because
/// [`primary_ctx`] and every constructor share ordinal 0.
pub fn probe_compute_cap() -> Result<(i32, i32), String> {
    compute_cap_of(primary_ctx()?)
}

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
    /// Bytes one block may hold **after opting in** — 101,376 on an L40S
    /// against the 49,152 of `shared_per_block`. Read from
    /// `CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN`, never derived:
    /// the folklore is `shared_per_sm − 1024`, which happens to give the right
    /// answer on this card and is a guess about a reservation the driver owns.
    /// Same rule as `l2_bytes` above — the driver publishes it, so it is read.
    pub shared_per_block_optin: i32,
    pub shared_per_sm: i32,
    /// The two other residency limits an SM has, read for the same reason
    /// `shared_per_sm` is: the persistent arms of A3 size their grid to what
    /// the card holds resident (`crate::occ::residency`), and the 1,536
    /// threads / 65,536 registers of an L40S are a fixture in a test, never
    /// a constant in the launcher.
    pub max_threads_per_sm: i32,
    pub regs_per_sm: i32,
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
        let sha256 = crate::sha256::sha256_hex(text.as_bytes());
        Self { text, sha256 }
    }
}

/// NVRTC on `src` for [`arch()`], as [`Cuda::on_stream`] compiles it: no
/// context needed, so it may run on any thread.
pub fn compile_ptx(src: &KernelSource) -> Result<cudarc::nvrtc::Ptx, String> {
    let opts = CompileOptions {
        arch: Some(arch()),
        ..Default::default()
    };
    compile_ptx_with_opts(&src.text, opts).map_err(|e| format!("NVRTC refused the kernel source:\n{e}"))
}

/// A context, a stream, and one compiled module.
pub struct Cuda {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
}

impl Cuda {
    /// Open device 0 and compile `src` for [`arch()`].
    pub fn new(src: &KernelSource) -> Result<Self, String> {
        let ctx = CudaContext::new(0).map_err(|e| format!("no CUDA device: {e}"))?;
        let stream = ctx.default_stream();
        let opts = CompileOptions {
            // `arch` is `Option<&'static str>`; a value computed at runtime
            // would have to go through `options`, which NVRTC unrolls *after*
            // `--gpu-architecture=` and would therefore override it.
            arch: Some(arch()),
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
        capture_on(&self.stream, body)
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
        Self::on_stream_ptx(stream, compile_ptx(src)?)
    }

    /// [`Self::on_stream`] from PTX [`compile_ptx`] already made: the
    /// driver's JIT only. Lets a caller run NVRTC on several sources in
    /// parallel threads first (fusedrun's 4E load check).
    pub fn on_stream_ptx(stream: Arc<CudaStream>, ptx: cudarc::nvrtc::Ptx) -> Result<Self, String> {
        let ctx = stream.context().clone();
        let module = ctx
            .load_module(ptx)
            .map_err(|e| format!("the driver refused the PTX: {e}"))?;
        Ok(Self { ctx, stream, module })
    }

    pub fn device(&self) -> Result<DeviceReport, String> {
        let a = |x: Attr| self.ctx.attribute(x).map_err(|e| format!("attribute: {e}"));
        Ok(DeviceReport {
            name: self.ctx.name().map_err(|e| format!("device name: {e}"))?,
            compute_cap: compute_cap_of(&self.ctx)?,
            sm_count: a(Attr::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?,
            l2_bytes: a(Attr::CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE)?,
            shared_per_block: a(Attr::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)?,
            shared_per_block_optin: a(
                Attr::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
            )?,
            shared_per_sm: a(Attr::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR)?,
            max_threads_per_sm: a(Attr::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR)?,
            regs_per_sm: a(Attr::CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR)?,
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

    /// [`Self::func`], plus the opt-in a kernel needs to stage more than the
    /// default 48 KiB of dynamic shared memory.
    ///
    /// `bytes` is the **largest** allocation any later launch of this function
    /// will ask for. The driver refuses a launch whose `sharedMemBytes`
    /// exceeds the attribute (`CUDA_ERROR_INVALID_VALUE`), so under-declaring
    /// fails loudly rather than corrupting; over-declaring costs occupancy,
    /// which on a one-block kernel costs nothing at all.
    ///
    /// **Set on the function, once, at load.** It is a property of the loaded
    /// `CUfunction`, not of a launch: posing it per launch would be one driver
    /// call per token on a path whose measured problem is launch latency.
    ///
    /// Three states, and the third is the one that matters:
    ///
    ///  * under the default — nothing is asked of the driver, so a card with
    ///    no opt-in behaves exactly as before;
    ///  * over the default, under the ceiling — the attribute is set here and
    ///    the launch is legal;
    ///  * over the ceiling — **refused**, because there is no host-side
    ///    remedy and a launch past it is what corrupts.
    ///
    /// The arithmetic is `crate::shared`, which is portable and tested; this
    /// wrapper only reads the card and talks to the driver.
    pub fn func_dynamic_shared(&self, name: &str, bytes: u32) -> Result<CudaFunction, String> {
        let f = self.func(name)?;
        let dev = self.device()?;
        let (def, optin) = (dev.shared_per_block as usize, dev.shared_per_block_optin as usize);
        match crate::shared::plan(bytes as usize, def, optin) {
            None => Err(format!(
                "{name}: {bytes} bytes of dynamic shared memory requested, past both bounds: \
                 the card offers {def} by default and {} after opt-in.",
                crate::shared::ceiling(def, optin)
            )),
            Some(crate::shared::Fit::Default) => Ok(f),
            Some(crate::shared::Fit::OptIn) => {
                f.set_attribute(
                    cudarc::driver::sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    bytes as i32,
                )
                .map_err(|e| {
                    format!("{name}: opt-in for {bytes} shared bytes refused by the driver: {e}")
                })?;
                Ok(f)
            }
        }
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
        // The property is an ORDER, not an equality, and this guard tested the
        // equality until 2026-09-08.
        //
        // NVRTC emits PTX, and PTX is forward-compatible: the driver JITs a
        // `compute_NN` module for any sm ≥ NN. `ops/run.py` says exactly that
        // where it pins `MIN_COMPUTE_CAP` — *"the driver can JIT it for any
        // sm ≥ 89, and for nothing below"*. So a function loaded from
        // `compute_89` PTX on an sm_120 card reports `binary_version = 120`,
        // which is the mechanism working, not failing.
        //
        // The equality happened to hold everywhere it had been exercised —
        // `compute_89` on L40S (sm_89), `compute_80` on A100 (sm_80) in F4 —
        // because each run named its own card's sm. The first card newer than
        // its `LLVQ_NVRTC_ARCH` was an RTX PRO 6000 on 2026-09-08, and the
        // guard refused a run that was entirely correct. `compute_120` was not
        // the way out either: the image is CUDA 12.4 and sm_120 arrived in
        // 12.8, so its NVRTC cannot name that architecture at all.
        //
        // Nothing is lost by relaxing it. The failure the message names — a
        // silent fallback to `compute_75` — was never caught by the equality:
        // on a card of sm ≥ the request, `binary_version` reports the DEVICE's
        // sm whatever PTX it was JITted from, so both readings agree. What the
        // order does catch, and the equality also caught, is the real defect:
        // a module that ran BELOW the architecture it was compiled for, which
        // cannot describe the card the numbers are attributed to.
        //
        // The card's own name is printed beside every figure (`Device::name`),
        // so a reader still sees which silicon a number came from — and rule 5
        // forbids dividing a × across cards whether or not this guard fires.
        if !crate::arch_binary_ok(rep.binary_version, arch_binary_version()) {
            return Err(format!(
                "{name}: ran on sm_{}, BELOW the sm_{} it was compiled for. PTX is \
                 forward-compatible only, so nothing measured on this module describes \
                 the card.",
                rep.binary_version,
                arch_binary_version()
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

    /// The site table of A3's persistent-global arm: device pointers and
    /// packed `u32` pairs, eight `u64` a site (`crate::occ::site_words`).
    pub fn up_u64(&self, v: &[u64]) -> Result<CudaSlice<u64>, String> {
        self.stream.clone_htod(v).map_err(|e| format!("H2D u64: {e}"))
    }

    pub fn zeros_u32(&self, n: usize) -> Result<CudaSlice<u32>, String> {
        self.stream.alloc_zeros(n).map_err(|e| format!("alloc u32: {e}"))
    }

    pub fn zeros_f32(&self, n: usize) -> Result<CudaSlice<f32>, String> {
        self.stream.alloc_zeros(n).map_err(|e| format!("alloc f32: {e}"))
    }

    /// A binary16 output, for the kernels that write one.
    ///
    /// No LLVQ arm needs it, they all write `float* y`. The AWQ arm writes
    /// `unsigned short* outputs`, because that is what the upstream kernel
    /// does and we do not modify it.
    pub fn zeros_u16(&self, n: usize) -> Result<CudaSlice<u16>, String> {
        self.stream.alloc_zeros(n).map_err(|e| format!("alloc u16: {e}"))
    }

    pub fn down_u32(&self, d: &CudaSlice<u32>) -> Result<Vec<u32>, String> {
        self.stream.clone_dtoh(d).map_err(|e| format!("D2H u32: {e}"))
    }

    pub fn down_f32(&self, d: &CudaSlice<f32>) -> Result<Vec<f32>, String> {
        self.stream.clone_dtoh(d).map_err(|e| format!("D2H f32: {e}"))
    }

    /// The counterpart of [`Self::zeros_u16`]: read a binary16 output back.
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
    /// keep `f32`, because their published kernel `b/weight` bills 32 bits
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
    /// floats, and **the caller is still what keeps it under the device limit
    /// — the kernel has no way to check and would simply corrupt.**
    ///
    /// What moved on 2026-08-17 is the *bound*, not the responsibility. That
    /// limit is not one number: past the default 48 KiB a block gets, sm_70
    /// and later grant up to `MAX_SHARED_MEMORY_PER_BLOCK_OPTIN` (101,376 bytes on
    /// an L40S) **to a function that asked**. So the caller owes two things
    /// now: `crate::shared::rot_plan` to decide the width is legal, and
    /// [`Self::func_dynamic_shared`] to load `f` — which is where the opt-in
    /// is posed. Handing this a plain [`Self::func`] handle and an `n` over
    /// 12,288 gets the launch refused by the driver; handing it a width past
    /// the ceiling is what corrupts, and no check here would catch it.
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

    /// [`Self::launch_rot`] for `n_rows` activations, one launch.
    ///
    /// The grid carries the rows and the block carries one row, so the shared
    /// memory per block is what one row always needed — `n * 4` — and the
    /// bound `max_d_in` checks at load time holds at any `n_rows`.
    ///
    /// `grid.x` is `n_rows` exactly. The kernel has no bounds guard, so a
    /// grid wider than the output would overrun it; that is asserted here,
    /// once a call, rather than branched on by every thread.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_rot_rows<T: cudarc::driver::DeviceRepr>(
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
        row_stride: u32,
        n_rows: u32,
        threads: u32,
    ) -> Result<(), String> {
        if n_rows == 0 {
            return Err("rot_apply_rows: zero rows".to_string());
        }
        let want = (n as usize) * (n_rows as usize);
        if xout.len() < want {
            return Err(format!(
                "rot_apply_rows: {n_rows} rows of {n} need {want} floats, the output holds {}",
                xout.len()
            ));
        }
        let cfg = LaunchConfig {
            grid_dim: (n_rows, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: n * 4,
        };
        let mut b = self.stream.launch_builder(f);
        b.arg(xin).arg(signbits).arg(small).arg(xout).arg(&n).arg(&m).arg(&k).arg(&inv)
            .arg(&x_off).arg(&row_stride);
        unsafe { b.launch(cfg) }.map_err(|e| format!("rot_apply_rows: {e}"))?;
        Ok(())
    }

    /// [`Self::launch_rot`] for `rot_apply_dev` (`kernels/rotate_wide.cu`):
    /// the same transform staged in `scratch`, device memory, with **zero**
    /// bytes of dynamic shared memory. For widths `crate::shared::rot_plan`
    /// refuses; `crate::wide::rot_kernel` decides.
    ///
    /// `scratch` must hold `crate::wide::rot_scratch_floats(1, n, m, k)`
    /// floats and be a buffer of its own: checked here, once a call, since the
    /// kernel has no bound to check against.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_rot_dev<T: cudarc::driver::DeviceRepr>(
        &self,
        f: &CudaFunction,
        xin: &CudaSlice<T>,
        signbits: &CudaSlice<u32>,
        small: &CudaSlice<f32>,
        xout: &mut CudaSlice<f32>,
        scratch: &mut CudaSlice<f32>,
        n: u32,
        m: u32,
        k: u32,
        inv: f32,
        x_off: u32,
        threads: u32,
    ) -> Result<(), String> {
        let want = crate::wide::rot_scratch_floats(1, n as usize, m as usize, k as usize);
        if scratch.len() < want {
            return Err(format!(
                "rot_apply_dev: width {n} needs {want} floats of scratch, it holds {}",
                scratch.len()
            ));
        }
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = self.stream.launch_builder(f);
        b.arg(xin).arg(signbits).arg(small).arg(xout).arg(scratch).arg(&n).arg(&m).arg(&k)
            .arg(&inv).arg(&x_off);
        unsafe { b.launch(cfg) }.map_err(|e| format!("rot_apply_dev: {e}"))?;
        Ok(())
    }

    /// [`Self::launch_rot_rows`] for `rot_apply_rows_dev`: one row a block,
    /// row `r` staged at `scratch + r * n`, zero dynamic shared memory.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_rot_rows_dev<T: cudarc::driver::DeviceRepr>(
        &self,
        f: &CudaFunction,
        xin: &CudaSlice<T>,
        signbits: &CudaSlice<u32>,
        small: &CudaSlice<f32>,
        xout: &mut CudaSlice<f32>,
        scratch: &mut CudaSlice<f32>,
        n: u32,
        m: u32,
        k: u32,
        inv: f32,
        x_off: u32,
        row_stride: u32,
        n_rows: u32,
        threads: u32,
    ) -> Result<(), String> {
        if n_rows == 0 {
            return Err("rot_apply_rows_dev: zero rows".to_string());
        }
        let want = (n as usize) * (n_rows as usize);
        if xout.len() < want {
            return Err(format!(
                "rot_apply_rows_dev: {n_rows} rows of {n} need {want} floats, the output holds {}",
                xout.len()
            ));
        }
        let need =
            crate::wide::rot_scratch_floats(n_rows as usize, n as usize, m as usize, k as usize);
        if scratch.len() < need {
            return Err(format!(
                "rot_apply_rows_dev: {n_rows} rows of {n} need {need} floats of scratch, it \
                 holds {}",
                scratch.len()
            ));
        }
        let cfg = LaunchConfig {
            grid_dim: (n_rows, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = self.stream.launch_builder(f);
        b.arg(xin).arg(signbits).arg(small).arg(xout).arg(scratch).arg(&n).arg(&m).arg(&k)
            .arg(&inv).arg(&x_off).arg(&row_stride);
        unsafe { b.launch(cfg) }.map_err(|e| format!("rot_apply_rows_dev: {e}"))?;
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

/// Capture whatever `body` launches on `stream` into a replayable graph —
/// the free-standing form of [`Cuda::capture`], for callers that own a stream
/// without owning a [`Cuda`] (fusedrun's graph mode captures on CANDLE's
/// stream). Same contract, word for word: `Relaxed` mode, capture closed on
/// both error paths, the body's error reported first, `None` from the driver
/// surfaced as the legacy-NULL-stream message, `upload()` before return.
pub fn capture_on(
    stream: &Arc<CudaStream>,
    body: impl FnOnce() -> Result<(), String>,
) -> Result<cudarc::driver::CudaGraph, String> {
    stream
        .begin_capture(cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
        .map_err(|e| format!("begin_capture: {e}"))?;
    let r = body();
    let ended = stream
        .end_capture(cudarc::driver::sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH);
    r?;
    let g = ended.map_err(|e| format!("end_capture: {e}"))?;
    let g = g.ok_or_else(|| {
        "the driver captured nothing. Legacy NULL stream? (see new_on_fresh_stream)".to_string()
    })?;
    g.upload().map_err(|e| format!("graph upload: {e}"))?;
    Ok(g)
}

fn row_grid(d_out: u32, threads: u32, shared: u32) -> LaunchConfig {
    assert_eq!(d_out % (threads / 32), 0, "rows must fill whole blocks");
    LaunchConfig {
        grid_dim: (d_out * 32 / threads, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: shared,
    }
}
