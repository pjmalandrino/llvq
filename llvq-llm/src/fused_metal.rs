//! The Tetra kernel reached through candle, on Metal.
//!
//! The Metal twin of [`crate::fused_cuda`], and the adapter half of the port
//! in [`crate::device`]. The MSL lives in `kernels/llvq_tetra48.metal` and is
//! proved on its own by `llvq-metal/tests/tetra48_matches_rust.rs` and
//! `tetra48_matvec_matches_host.rs`, which drive it through metal-rs. This
//! module proves the BINDING: the same kernel, reached from a candle tensor.
//!
//! ## Why none of `llvq-metal` is reused
//!
//! `llvq-metal` pins `metal = "0.29"` (metal-rs). candle reaches Metal through
//! `objc2-metal`, and the two `Buffer` types are unrelated. A buffer made by
//! one cannot be bound by the other's encoder. Only the MSL text crosses.
//!
//! ## Five things about candle's Metal backend that cost a night if unknown
//!
//! Every one was measured on 2026-09-20, in a scratch crate, before this file
//! was written.
//!
//!   1. A bad MSL source **panics**; it does not return `Err`.
//!      `new_library_with_source` ends in `.unwrap()`. So the compile is
//!      wrapped in `catch_unwind` and the payload is re-raised as the error
//!      text, because `served::load_resolved` owes a refusal by name.
//!   2. Forgetting `set_threadgroup_memory_length` is **not** an error. The
//!      kernel dispatches, completes, and writes zeros. No fault, no warning.
//!   3. The allocator hands your buffers back out as scratch. `new_buffer`
//!      reuses anything at `Arc::strong_count == 1`, so the tables and the
//!      weight stream must be OWNED for the life of the model.
//!   4. Output buffers are pooled, rounded up to a power of two, and not
//!      zeroed. A row the kernel skips returns a previous tensor's bytes,
//!      which look like a plausible activation.
//!   5. candle's own `Kernels` cache cannot hold this library: `Source` is a
//!      closed enum over 15 `include_str!` constants. This module carries its
//!      own cache, and its key includes the injected tile, because a define
//!      that fails to land changes the staging and not the arithmetic.

use candle_core::backend::BackendStorage;
use candle_core::{CpuStorage, CustomOp1, DType, Layout, MetalStorage, Result, Shape, Tensor};
use candle_metal_kernels::metal::{Buffer, ComputePipeline};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// The coordinate order the quads come out in. Mirrors `TETRA48_ORDER` in the
/// shader and `TETRA48_ORDER` in `llvq_tetra48.cuh`.
const ORDER: [usize; llvq_core::DIM] = [
    0, 1, 2, 3, 4, 7, 10, 12, 6, 11, 13, 14, 16, 17, 18, 19, 5, 8, 9, 15, 20, 21, 22, 23,
];

/// Lanes in an Apple SIMD-group. The kernel gives one to a row.
const LANES: usize = 32;
/// Threads a threadgroup, so eight rows each. The kernel carries no
/// `row < d_out` guard, exactly like the CUDA one, so `d_out` must be a
/// multiple of the rows a group covers. Asserted at upload.
const GROUP: usize = 256;
/// Threadgroup memory an M3 Max offers, measured 2026-09-20 through
/// `MTLDevice.maxThreadgroupMemoryLength`.
///
/// A constant rather than a device query on purpose: the kernels are compiled
/// against it, so a machine that offered less would have to be refused rather
/// than quietly served a kernel that stages past its limit. Asking the device
/// and adapting would hide exactly that.
const THREADGROUP_LIMIT: usize = 32_768;

const SRC_TETRA: &str = include_str!("../kernels/llvq_tetra48.metal");
const SRC_ROT: &str = include_str!("../kernels/llvq_rot.metal");
const SRC_Q4: &str = include_str!("../kernels/tv_q4_h.metal");
const SRC_Q8: &str = include_str!("../kernels/emb_q8.metal");

/// Which file carries which entry point.
///
/// Four translation units, not one, for the reason `tetra48_seg.cu` gives on
/// the CUDA side: appending to a shipped unit changes its bytes and can move
/// the register allocation of a kernel no correctness test can see move.
fn source_of(name: &str) -> Result<&'static str> {
    match name {
        "tetra48_probe" | "tv_tetra48_metal" => Ok(SRC_TETRA),
        "rot_apply_metal" | "rot_apply_rows_metal" => Ok(SRC_ROT),
        "tv_q4_metal" => Ok(SRC_Q4),
        "emb_q8_gather_metal" | "tv_q8_metal" => Ok(SRC_Q8),
        other => candle_core::bail!("no Metal source carries {other}"),
    }
}

/// The pipeline cache, keyed on everything that changes the compiled code.
///
/// The tile is in the key and must stay there: it is injected as a text
/// prelude, and a define that fails to land changes how much threadgroup
/// memory the host must stage while leaving the arithmetic alone. That is the
/// failure a cache keyed on the function name alone produces, silently.
type Key = (&'static str, usize);

/// What the decoder needs, uploaded once and shared by every projection.
///
/// 18,688 bytes: rows 4,096 u32, branches 1,024 u16, prefixes 128 u8,
/// suffixes 128 u8. Held as `Arc<Buffer>` for the reason the header gives:
/// released, they become somebody else's scratch.
pub struct MetalTables {
    rows: Arc<Buffer>,
    prefixes: Arc<Buffer>,
    branches: Arc<Buffer>,
    suffixes: Arc<Buffer>,
    /// `1/sqrt(16 m)`, entry 0 the origin.
    invnorm: Arc<Buffer>,
    /// Host copies, kept only so `cpu_fwd` can serve as the oracle. `None` on
    /// a served model, where the CPU arm refuses instead.
    host: Option<HostTables>,
}

struct HostTables {
    rows: Vec<u32>,
    prefixes: Vec<u8>,
    branches: Vec<u16>,
    suffixes: Vec<u8>,
    invnorm: Vec<f32>,
}

/// One Tetra projection on the device.
pub struct MetalTetraProj {
    pub name: String,
    pub d_out: usize,
    pub d_in: usize,
    nblocks: usize,
    tail_w: usize,
    row_stride_u32: usize,
    words: Arc<Buffer>,
    gscale: Arc<Buffer>,
    rscale: Arc<Buffer>,
    tail: Arc<Buffer>,
    host: Option<HostProj>,
}

struct HostProj {
    words: Vec<u8>,
    gscale: Vec<f32>,
    rscale: Vec<f32>,
    tail: Vec<u16>,
}

/// The device, the tables and the compiled pipelines.
pub struct MetalRuntime {
    device: candle_core::MetalDevice,
    tables: MetalTables,
    tile: usize,
    pipelines: RwLock<HashMap<Key, ComputePipeline>>,
}

/// f16 bits widened exactly, which is what the shader's `float(half)` does.
///
/// A transcription of `h2f` in `llvq-cuda/kernels/matvec.cu:44-77`, branch for
/// branch, and it is a transcription because an earlier version here was
/// written from memory and was wrong on 4,094 of the 65,536 patterns: every
/// subnormal had its exponent one too low and its mantissa shifted one too
/// far, and there was no `exp == 31` arm at all, so infinities came back as
/// 65536 and NaN payloads were lost. The gate could not see it, because its
/// fixtures draw from a Gaussian and never produce one.
fn h2f(h: u16) -> f32 {
    let h = h as u32;
    let sign = (h & 0x8000) << 16;
    let exp = (h >> 10) & 0x1f;
    let mut man = h & 0x3ff;
    let bits = if exp == 0 {
        if man == 0 {
            sign
        } else {
            // Normalise: shift until the hidden bit appears, one exponent a shift.
            let mut e = 127 - 15 + 1;
            while man & 0x400 == 0 {
                man <<= 1;
                e -= 1;
            }
            sign | (e << 23) | ((man & 0x3ff) << 13)
        }
    } else if exp == 31 {
        // inf / NaN, payload kept.
        sign | 0x7f80_0000 | (man << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (man << 13)
    };
    f32::from_bits(bits)
}

impl MetalRuntime {
    /// Upload the decoder tables and take the device.
    ///
    /// `keep_host` keeps a host copy of everything, which is what lets
    /// [`TetraMatvec::cpu_fwd`] act as the oracle a test compares against. A
    /// served model passes `false` and the CPU arm then refuses by name
    /// instead of silently costing a model its memory twice.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &candle_core::Device,
        rows: &[u32],
        prefixes: &[u8],
        branches: &[u16],
        suffixes: &[u8],
        invnorm: &[f32],
        tile: usize,
        keep_host: bool,
    ) -> Result<Self> {
        let dev = match device {
            candle_core::Device::Metal(d) => d.clone(),
            other => candle_core::bail!("the Metal runtime wants a Metal device, got {other:?}"),
        };
        // 32..=256, not the CUDA contract's 32..=512. The staging is
        // `tile * 24 * 4` bytes against Apple's 32,768 B of threadgroup
        // memory, so 341 blocks is the ceiling and 256 the largest power of
        // two under it. 512 would ask for 49,152 B, pass a validator copied
        // from the wrong platform, and abort the process at dispatch.
        if !tile.is_power_of_two() || !(32..=256).contains(&tile) {
            candle_core::bail!(
                "tile {tile} is not a power of two in 32..=256. Apple stages \
                 {} B for it against a limit of 32768",
                tile * llvq_core::DIM * 4
            );
        }
        let tables = MetalTables {
            rows: dev.new_buffer_with_data(rows)?,
            prefixes: dev.new_buffer_with_data(prefixes)?,
            branches: dev.new_buffer_with_data(branches)?,
            suffixes: dev.new_buffer_with_data(suffixes)?,
            invnorm: dev.new_buffer_with_data(invnorm)?,
            host: keep_host.then(|| HostTables {
                rows: rows.to_vec(),
                prefixes: prefixes.to_vec(),
                branches: branches.to_vec(),
                suffixes: suffixes.to_vec(),
                invnorm: invnorm.to_vec(),
            }),
        };
        Ok(Self {
            device: dev,
            tables,
            tile,
            pipelines: RwLock::new(HashMap::new()),
        })
    }

    /// Upload one projection's stream and its scales.
    #[allow(clippy::too_many_arguments)]
    pub fn upload(
        &self,
        name: &str,
        d_out: usize,
        d_in: usize,
        nblocks: usize,
        row_stride_u32: usize,
        words: &[u8],
        gscale: &[f32; 2],
        rscale: &[f32],
        tail: &[u16],
    ) -> Result<MetalTetraProj> {
        let rows_per_group = GROUP / LANES;
        if !d_out.is_multiple_of(rows_per_group) {
            candle_core::bail!(
                "{name}: d_out {d_out} is not a multiple of {rows_per_group}. The kernel \
                 carries no row guard, so a partial threadgroup would compute and STORE \
                 rows past the output"
            );
        }
        let tail_w = d_in - nblocks * llvq_core::DIM;
        if rscale.len() != d_out {
            candle_core::bail!("{name}: {} row scales for {d_out} rows", rscale.len());
        }
        if tail.len() != d_out * tail_w {
            candle_core::bail!("{name}: {} tail values for {d_out}x{tail_w}", tail.len());
        }
        // The stream, which the kernel indexes as `words + row * row_stride_u32`
        // with no bound of its own. `rscale` and `tail` were checked and this
        // was not, which is the asymmetry an audit of 2026-09-21 named.
        let want_stride = llvq_artifact::tetra48::stride_u32(nblocks);
        if row_stride_u32 != want_stride {
            candle_core::bail!(
                "{name}: row stride {row_stride_u32} for {nblocks} blocks, which wants \
                 {want_stride}"
            );
        }
        // The stream is `d_out * row_stride_u32` words, plus AT MOST one guard
        // word. `f1r_load` reads `row[w]` and `row[w + 1]`, so the last block
        // of the last row reaches one word past the table; the writer pads for
        // it. An equality here refused the served 4B by four bytes.
        let want_bytes = d_out * row_stride_u32 * 4;
        if words.len() != want_bytes && words.len() != want_bytes + 4 {
            candle_core::bail!(
                "{name}: {} stream bytes for {d_out} rows of {row_stride_u32} words, which \
                 wants {want_bytes} or {want_bytes} plus one guard word",
                words.len()
            );
        }
        // Metal refuses a zero-length buffer and hands back a null pointer,
        // the same wall cudarc puts up. A `d_in` that is a multiple of 24 has
        // no tail, so it gets a one-element dummy the kernel never reads:
        // `tail_w == 0` makes the epilogue's loop empty.
        let dummy = [0u16];
        let tail_src = if tail.is_empty() { &dummy[..] } else { tail };
        Ok(MetalTetraProj {
            name: name.to_string(),
            d_out,
            d_in,
            nblocks,
            tail_w,
            row_stride_u32,
            words: self.device.new_buffer_with_data(words)?,
            gscale: self.device.new_buffer_with_data(gscale)?,
            rscale: self.device.new_buffer_with_data(rscale)?,
            tail: self.device.new_buffer_with_data(tail_src)?,
            host: self.tables.host.is_some().then(|| HostProj {
                words: words.to_vec(),
                gscale: gscale.to_vec(),
                rscale: rscale.to_vec(),
                tail: tail.to_vec(),
            }),
        })
    }

    /// The pipeline for `name` at this runtime's tile, compiled once.
    fn pipeline(&self, name: &'static str) -> Result<ComputePipeline> {
        let key: Key = (name, self.tile);
        if let Some(p) = self.pipelines.read().expect("no panic held this lock").get(&key) {
            return Ok(p.clone());
        }
        let src = format!("#define LLVQ_TILE_BLOCKS {}u\n{}", self.tile, source_of(name)?);
        let dev = self.device.device().clone();
        // A bad source PANICS rather than returning Err: both
        // `new_library_with_source` and `new_compute_pipeline_state_with_function`
        // end in `.unwrap()`. The panic payload carries Metal's own diagnostics,
        // so it is re-raised as the message instead of being swallowed.
        let built = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let lib = dev.new_library_with_source(&src, None)?;
            let func = lib.get_function(name, None)?;
            dev.new_compute_pipeline_state_with_function(&func)
        }));
        let pipe = match built {
            Ok(Ok(p)) => p,
            Ok(Err(e)) => candle_core::bail!("{name}: Metal refused the source: {e:?}"),
            Err(panic) => {
                let what = panic
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_else(|| "no diagnostics".into());
                candle_core::bail!("{name}: the MSL did not compile: {what}");
            }
        };
        self.pipelines
            .write()
            .expect("no panic held this lock")
            .insert(key, pipe.clone());
        Ok(pipe)
    }

    /// `y = W x` for one activation row, through the kernel.
    pub fn matvec(&self, proj: &MetalTetraProj, x: &Tensor) -> Result<Tensor> {
        let op = TetraMatvec { rt: self, proj };
        x.apply_op1_no_bwd(&op)
    }
}

/// One launch of `tv_tetra48_metal`, as a candle op.
struct TetraMatvec<'a> {
    rt: &'a MetalRuntime,
    proj: &'a MetalTetraProj,
}

impl TetraMatvec<'_> {
    /// What both arms owe the caller before either computes anything.
    ///
    /// Added after an audit of 2026-09-21 found `metal_fwd` checking dtype and
    /// contiguity and nothing else, while `cpu_fwd` checked only the dtype.
    /// The CUDA twin refuses all of this by name at `fused_cuda.rs:1491`, with
    /// a comment saying the missing check already cost one run.
    fn check(&self, dtype: DType, layout: &Layout) -> Result<()> {
        let p = self.proj;
        if dtype != DType::F32 {
            candle_core::bail!("{}: f32 only, got {dtype:?}", p.name);
        }
        if !layout.is_contiguous() {
            candle_core::bail!(
                "{}: the activation must be contiguous. The kernel indexes it linearly \
                 from one base and cannot honour a stride",
                p.name
            );
        }
        let dims = layout.dims();
        let d_in = *dims.last().unwrap_or(&0);
        if d_in != p.d_in {
            candle_core::bail!(
                "{}: activation of {d_in} values for d_in={}. A short one reads past the \
                 buffer and returns finite, plausible, wrong numbers",
                p.name,
                p.d_in
            );
        }
        // One vector a launch, like the CUDA twin. The output is allocated at
        // `d_out` elements, so accepting several rows would hand back a tensor
        // whose shape claims more than its storage holds.
        let rows: usize = dims[..dims.len().saturating_sub(1)].iter().product();
        if rows != 1 {
            candle_core::bail!(
                "{}: {rows} vectors at once. This kernel is a matvec and takes one; the \
                 row loop belongs to the caller",
                p.name
            );
        }
        Ok(())
    }
}

impl CustomOp1 for TetraMatvec<'_> {
    fn name(&self) -> &'static str {
        "tetra48-matvec"
    }

    /// The oracle, not a stub.
    ///
    /// `CustomOp1` requires a CPU arm, and making it the reference rather than
    /// a `bail!` is what lets a test drive the SAME op on both devices and
    /// demand equality. It reproduces the kernel's summation order exactly:
    /// 32 lanes striding the blocks, the shuffle-xor butterfly, and the
    /// multiply-then-add tail. A plain loop would need a tolerance, and a
    /// tolerance is the size of a real defect here.
    ///
    /// Refuses when the runtime was built without host copies, which is the
    /// served case: a model that kept them would pay for its weights twice.
    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        let (ht, hp) = match (&self.rt.tables.host, &self.proj.host) {
            (Some(t), Some(p)) => (t, p),
            _ => candle_core::bail!(
                "{}: this runtime was built without host copies, so its CPU arm is not \
                 available. It is the oracle of a test, never a fallback for a served model",
                self.proj.name
            ),
        };
        self.check(storage.dtype(), layout)?;
        let x = match storage {
            CpuStorage::F32(v) => &v[layout.start_offset()..],
            other => candle_core::bail!("{}: f32 only, got {:?}", self.proj.name, other.dtype()),
        };
        let p = self.proj;
        let tile = self.rt.tile;
        let mut y = vec![0f32; p.d_out];
        let ntiles = p.nblocks.div_ceil(tile);

        for (row, out) in y.iter_mut().enumerate() {
            let mut lanes = [0f32; LANES];
            for t in 0..ntiles {
                let jlo = t * tile;
                let jhi = (jlo + tile).min(p.nblocks);
                for (lane, acc) in lanes.iter_mut().enumerate() {
                    let mut j = jlo + lane;
                    while j < jhi {
                        *acc += host_block(ht, hp, p, row, j, x);
                        j += LANES;
                    }
                }
            }
            for k in [16usize, 8, 4, 2, 1] {
                let mut next = [0f32; LANES];
                for (l, n) in next.iter_mut().enumerate() {
                    *n = lanes[l] + lanes[l ^ k];
                }
                lanes = next;
            }
            let mut tv = 0f32;
            let xt = &x[p.nblocks * llvq_core::DIM..];
            // `mul_add` at both sites: the CUDA twin contracts there under
            // NVRTC's `--fmad=true`, and the shader now spells it out.
            for (i, xi) in xt.iter().enumerate().take(p.tail_w) {
                tv = h2f(hp.tail[row * p.tail_w + i]).mul_add(*xi, tv);
            }
            *out = lanes[0].mul_add(hp.rscale[row], tv);
        }
        let mut dims = layout.dims().to_vec();
        *dims.last_mut().expect("rank >= 1") = p.d_out;
        Ok((CpuStorage::F32(y), Shape::from(dims)))
    }

    fn metal_fwd(&self, storage: &MetalStorage, layout: &Layout) -> Result<(MetalStorage, Shape)> {
        self.check(storage.dtype(), layout)?;
        let p = self.proj;
        let dev = storage.device().clone();
        let pipe = self.rt.pipeline("tv_tetra48_metal")?;

        let out = dev.new_buffer(p.d_out, DType::F32, "tetra48-matvec")?;
        let enc = dev.command_encoder()?;
        enc.set_compute_pipeline_state(&pipe);

        // `start_offset()` is in ELEMENTS; the binding wants bytes.
        let x_off = layout.start_offset() * DType::F32.size_in_bytes();
        let stride = p.row_stride_u32 as u32;
        let nb = p.nblocks as u32;
        let tw = p.tail_w as u32;
        use candle_metal_kernels::utils::set_param;
        set_param(&enc, 0, (&*p.words, 0usize));
        set_param(&enc, 1, stride);
        set_param(&enc, 2, (&*self.rt.tables.rows, 0usize));
        set_param(&enc, 3, (&*self.rt.tables.prefixes, 0usize));
        set_param(&enc, 4, (&*self.rt.tables.branches, 0usize));
        set_param(&enc, 5, (&*self.rt.tables.suffixes, 0usize));
        set_param(&enc, 6, (&*p.gscale, 0usize));
        set_param(&enc, 7, (&*self.rt.tables.invnorm, 0usize));
        set_param(&enc, 8, (&*p.rscale, 0usize));
        set_param(&enc, 9, (&*p.tail, 0usize));
        set_param(&enc, 10, (storage.buffer(), x_off));
        set_param(&enc, 11, (&*out, 0usize));
        set_param(&enc, 12, nb);
        set_param(&enc, 13, tw);

        // Not optional, and not an error if omitted: the kernel would run to
        // completion and write zeros. The tile in the length and the tile in
        // the source are the same number by construction, because both come
        // from `self.rt.tile`.
        let staged = self.rt.tile * llvq_core::DIM * DType::F32.size_in_bytes();
        enc.set_threadgroup_memory_length(0, staged);

        for b in [
            &*p.words,
            &*self.rt.tables.rows,
            &*self.rt.tables.prefixes,
            &*self.rt.tables.branches,
            &*self.rt.tables.suffixes,
            &*p.gscale,
            &*self.rt.tables.invnorm,
            &*p.rscale,
            &*p.tail,
            storage.buffer(),
        ] {
            enc.use_resource(b, objc2_metal::MTLResourceUsage::Read);
        }
        enc.use_resource(&*out, objc2_metal::MTLResourceUsage::Write);

        let grid = objc2_metal::MTLSize {
            width: p.d_out * LANES,
            height: 1,
            depth: 1,
        };
        let group = objc2_metal::MTLSize {
            width: GROUP,
            height: 1,
            depth: 1,
        };
        enc.dispatch_threads(grid, group);

        let mut dims = layout.dims().to_vec();
        *dims.last_mut().expect("rank >= 1") = p.d_out;
        Ok((
            MetalStorage::new(out, dev, p.d_out, DType::F32),
            Shape::from(dims),
        ))
    }
}

/// One block's contribution, in the shader's order.
fn host_block(
    ht: &HostTables,
    hp: &HostProj,
    p: &MetalTetraProj,
    row: usize,
    j: usize,
    x: &[f32],
) -> f32 {
    let (lo, hi16) = host_load(&hp.words, p.row_stride_u32, row, j);
    let q = host_quads(ht, lo, hi16);
    let xb = &x[j * llvq_core::DIM..(j + 1) * llvq_core::DIM];
    let mut d = 0f32;
    for (i, quad) in q.iter().enumerate() {
        for k in 0..4 {
            let idx = ORDER[4 * i + k];
            let v = ((quad >> (8 * k)) & 0xff) as f32 - 128.0;
            d = v.mul_add(xb[idx], d);
        }
    }
    let n2: i32 = q
        .iter()
        .flat_map(|quad| (0..4).map(move |k| (((quad >> (8 * k)) & 0xff) as i32) - 128))
        .map(|v| v * v)
        .sum();
    let m = (((n2 as u32) >> 4) & 31) as usize;
    let g = ((hi16 >> 15) & 1) as usize;
    d * hp.gscale[g] * ht.invnorm[m]
}

/// `f1r_load`, on the host.
fn host_load(words: &[u8], row_stride_u32: usize, row: usize, j: usize) -> (u32, u32) {
    let base = row * row_stride_u32;
    let w = (3 * j) >> 1;
    let at = |i: usize| -> u32 {
        let o = (base + i) * 4;
        u32::from_le_bytes([words[o], words[o + 1], words[o + 2], words[o + 3]])
    };
    let w0 = at(w);
    let w1 = at(w + 1);
    match j & 1 {
        1 => ((w0 >> 16) | (w1 << 16), w1 >> 16),
        _ => (w0, w1 & 0xffff),
    }
}

/// CUDA's PRMT, default mode. See the shader's `prmt`.
fn prmt(a: u32, b: u32, s: u32) -> u32 {
    let mut out = 0u32;
    for i in 0..4 {
        let sel = (s >> (4 * i)) & 0xf;
        let idx = sel & 7;
        let src = if idx < 4 { a } else { b };
        let byte = (src >> (8 * (idx & 3))) & 0xff;
        let v = if sel & 8 != 0 {
            if byte & 0x80 != 0 {
                0xff
            } else {
                0
            }
        } else {
            byte
        };
        out |= v << (8 * i);
    }
    out
}

/// `f1r_v3_quads`, on the host.
fn host_quads(t: &HostTables, lo: u32, hi16: u32) -> [u32; 6] {
    const T: [[u32; 2]; 4] = [
        [0x887c8480, 0x90748c78],
        [0x79857d81, 0x718d7589],
        [0x7a867e82, 0x728e768a],
        [0x877b837f, 0x8f738b77],
    ];
    let p = lo & 1;
    let r = (lo >> 1) & 1;
    let s8 = (lo >> 2) & 63;
    let b1 = (lo >> 8) & 1;
    let i1 = (lo >> 9) & 0x7ff;
    let b2 = (lo >> 20) & 15;
    let i2 = ((lo >> 24) | ((hi16 & 7) << 8)) & 0x7ff;
    let b3 = (hi16 >> 3) & 1;
    let i3 = (hi16 >> 4) & 0x7ff;

    let c1 = t.prefixes[(2 * s8 + b1) as usize] as u32;
    let br = t.branches[(16 * s8 + b2) as usize] as u32;
    let c2 = br & 0xff;
    let s16 = (br >> 8) & 63;
    let c3 = t.suffixes[(2 * s16 + b3) as usize] as u32;

    let row1 = t.rows[(2048 * r + i1) as usize];
    let mid = i2 < 1240;
    let idx2 = if mid { i2 } else { (2048 - 1240) + i2 };
    let row2 = t.rows[idx2 as usize];
    let delta = u32::from(!mid);
    let r3 = (p ^ r ^ delta) & 1;
    let row3 = t.rows[(2048 * r3 + i3) as usize];

    let (c0lo, c0hi) = if p == 1 { (T[1][0], T[1][1]) } else { (T[0][0], T[0][1]) };
    let (c1lo, c1hi) = if p == 1 { (T[3][0], T[3][1]) } else { (T[2][0], T[2][1]) };
    let quad = |sel: u32, c4: u32| -> u32 {
        let m0 = prmt(c0lo, c0hi, sel);
        let m1 = prmt(c1lo, c1hi, sel);
        let k = ((c4.wrapping_mul(0x0020_4081)) & 0x0101_0101).wrapping_mul(0xff);
        (m0 & !k) | (m1 & k)
    };
    [
        quad(row1, c1 & 0xf),
        quad(row1 >> 16, c1 >> 4),
        quad(row2, c2 & 0xf),
        quad(row2 >> 16, c2 >> 4),
        quad(row3, c3 & 0xf),
        quad(row3 >> 16, c3 >> 4),
    ]
}



// ---------------------------------------------------------------------------
// The rotation.
//
// `prepare` owes this: the weights were quantized in a rotated basis, so the
// activation has to arrive in it. The kernel is a Walsh-Hadamard transform
// with a per-coordinate sign flip, then either a scale-out or a small mix.
//
// One thing differs from CUDA and it is not cosmetic. There the staging is a
// `__shared__` array, per block and free. Here it needs `n` floats a row, and
// `n` reaches 9,728 on the served 4B, which is 38,912 B against Apple's 32,768
// of threadgroup memory. So the scratch is a DEVICE buffer and every stage of
// the transform goes through global memory. Nothing about that cost is
// measured; it is the first thing to look at if the Metal throughput
// disappoints.
// ---------------------------------------------------------------------------

/// One rotation, uploaded. Shared by every projection that names its key.
pub struct MetalRotation {
    signbits: Arc<Buffer>,
    small: Arc<Buffer>,
    n: u32,
    m: u32,
    k: u32,
    inv: f32,
    /// Threads a threadgroup, the same rule the CUDA host uses.
    threads: usize,
}

impl MetalRuntime {
    /// Upload one rotation's tables.
    pub fn upload_rotation(&self, t: &crate::fused::RotationTables) -> Result<MetalRotation> {
        // Metal refuses a zero-length buffer. `small` is empty when k == 1,
        // which is the common case: `rot_mix` is not reached then and the
        // kernel never reads it.
        let dummy = [0f32];
        let small = if t.small.is_empty() { &dummy[..] } else { &t.small };
        Ok(MetalRotation {
            signbits: self.device.new_buffer_with_data(&t.signbits)?,
            small: self.device.new_buffer_with_data(small)?,
            n: t.n as u32,
            m: t.m as u32,
            k: t.k as u32,
            inv: t.inv,
            threads: (t.n as u32).next_power_of_two().clamp(32, 1024) as usize,
        })
    }

    /// `x` into the basis the weights were quantized in, one vector.
    pub fn rotate(&self, rot: &MetalRotation, name: &str, x: &Tensor) -> Result<Tensor> {
        let dims = x.dims();
        let rows: usize = dims[..dims.len() - 1].iter().product();
        if rows != 1 {
            candle_core::bail!(
                "{name}: rotation requested for {rows} vectors. The row loop belongs to \
                 the caller, which shares it across a group"
            );
        }
        let x = x.to_dtype(DType::F16)?;
        x.apply_op1_no_bwd(&RotOp { rt: self, rot, name: name.to_string(), rows: 1 })
    }

    /// The same for a chunk of rows, one threadgroup each.
    pub fn rotate_rows(&self, rot: &MetalRotation, name: &str, x: &Tensor, n_rows: usize) -> Result<Tensor> {
        let dims = x.dims();
        let rows: usize = dims[..dims.len() - 1].iter().product();
        if rows != n_rows {
            candle_core::bail!("{name}: {rows} rows of activation for {n_rows}");
        }
        if n_rows == 0 {
            candle_core::bail!("{name}: a rotation of zero rows");
        }
        let x = x.to_dtype(DType::F16)?;
        x.apply_op1_no_bwd(&RotOp { rt: self, rot, name: name.to_string(), rows: n_rows })
    }
}

struct RotOp<'a> {
    rt: &'a MetalRuntime,
    rot: &'a MetalRotation,
    name: String,
    rows: usize,
}

impl CustomOp1 for RotOp<'_> {
    fn name(&self) -> &'static str {
        "llvq-metal-rot"
    }

    /// No CPU path, and deliberately none.
    ///
    /// Unlike the matvec, this op has no oracle to be: the rotation is judged
    /// by `llvq-metal/tests/rot_matches_host.rs` against the CUDA text itself,
    /// executed. A second Rust transcription here would be a third place for
    /// the same mistake to live.
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("{}: the LLVQ rotation has no CPU path", self.name)
    }

    fn metal_fwd(&self, storage: &MetalStorage, layout: &Layout) -> Result<(MetalStorage, Shape)> {
        if storage.dtype() != DType::F16 {
            candle_core::bail!("{}: the rotation reads f16, got {:?}", self.name, storage.dtype());
        }
        let n = self.rot.n as usize;
        let (start, end) = layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::msg(format!("{}: non-contiguous activation", self.name)))?;
        if end - start != n * self.rows {
            candle_core::bail!(
                "{}: activation of {} values for {} rows of n={n}",
                self.name,
                end - start,
                self.rows
            );
        }
        let batched = self.rows > 1;
        let kname = if batched { "rot_apply_rows_metal" } else { "rot_apply_metal" };
        let pipe = self.rt.pipeline(kname)?;
        let dev = storage.device().clone();

        let out = dev.new_buffer(self.rows * n, DType::F32, "llvq-metal-rot")?;
        // The staging, in DEVICE memory. See the section header.
        let scratch = dev.new_buffer(self.rows * n, DType::F32, "llvq-metal-rot-scratch")?;

        let enc = dev.command_encoder()?;
        enc.set_compute_pipeline_state(&pipe);
        use candle_metal_kernels::utils::set_param;
        // `x_off` is in ELEMENTS here, not bytes: the kernel indexes
        // `xin[x_off + i]` on a ushort pointer. The Tetra matvec takes a byte
        // offset instead, because it binds at the offset. Two kernels, two
        // conventions, each matched to its own source.
        set_param(&enc, 0, (storage.buffer(), 0usize));
        set_param(&enc, 1, (&*self.rot.signbits, 0usize));
        set_param(&enc, 2, (&*self.rot.small, 0usize));
        set_param(&enc, 3, (&*out, 0usize));
        set_param(&enc, 4, (&*scratch, 0usize));
        set_param(&enc, 5, self.rot.n);
        set_param(&enc, 6, self.rot.m);
        set_param(&enc, 7, self.rot.k);
        set_param(&enc, 8, self.rot.inv);
        set_param(&enc, 9, start as u32);
        if batched {
            set_param(&enc, 10, n as u32);
        }
        for b in [storage.buffer(), &*self.rot.signbits, &*self.rot.small] {
            enc.use_resource(b, objc2_metal::MTLResourceUsage::Read);
        }
        for b in [&*out, &*scratch] {
            enc.use_resource(b, objc2_metal::MTLResourceUsage::Write);
        }
        let threads = self.rot.threads;
        enc.dispatch_threads(
            objc2_metal::MTLSize { width: threads * self.rows, height: 1, depth: 1 },
            objc2_metal::MTLSize { width: threads, height: 1, depth: 1 },
        );
        Ok((
            MetalStorage::new(out, dev, self.rows * n, DType::F32),
            Shape::from(vec![self.rows, n]),
        ))
    }
}

// ---------------------------------------------------------------------------
// The int4 arm: `v_proj` on the served mixed file, 36 of the 252 projections.
// ---------------------------------------------------------------------------

/// One projection served as affine int4 g128.
pub struct MetalInt4Proj {
    pub name: String,
    pub d_out: usize,
    pub d_in: usize,
    gpr: usize,
    wq: Arc<Buffer>,
    scales: Arc<Buffer>,
    biases: Arc<Buffer>,
}

impl MetalRuntime {
    /// Upload one int4 projection.
    #[allow(clippy::too_many_arguments)]
    pub fn upload_int4(
        &self,
        name: &str,
        d_out: usize,
        d_in: usize,
        gpr: usize,
        wq: &[u32],
        scales: &[u16],
        biases: &[u16],
    ) -> Result<MetalInt4Proj> {
        let rows_per_group = GROUP / LANES;
        if !d_out.is_multiple_of(rows_per_group) {
            candle_core::bail!(
                "{name}: d_out {d_out} is not a multiple of {rows_per_group}, and the kernel \
                 carries no row guard"
            );
        }
        if !d_in.is_multiple_of(8) {
            candle_core::bail!("{name}: d_in {d_in} is not a multiple of 8, so a row would \
                                start at a shifted nibble");
        }
        // The whole activation is staged, with no tile.
        let staged = d_in * 4;
        if staged > THREADGROUP_LIMIT {
            candle_core::bail!(
                "{name}: staging d_in={d_in} wants {staged} B of threadgroup memory against \
                 {THREADGROUP_LIMIT}"
            );
        }
        if wq.len() != d_out * d_in / 8 {
            candle_core::bail!("{name}: {} words for {d_out}x{d_in} int4", wq.len());
        }
        if scales.len() != d_out * gpr || biases.len() != d_out * gpr {
            candle_core::bail!("{name}: {} scales and {} biases for {d_out}x{gpr}",
                               scales.len(), biases.len());
        }
        Ok(MetalInt4Proj {
            name: name.to_string(),
            d_out,
            d_in,
            gpr,
            wq: self.device.new_buffer_with_data(wq)?,
            scales: self.device.new_buffer_with_data(scales)?,
            biases: self.device.new_buffer_with_data(biases)?,
        })
    }

    /// `y = W x`, one vector, through `tv_q4_metal`.
    pub fn matvec_int4(&self, proj: &MetalInt4Proj, x: &Tensor) -> Result<Tensor> {
        // Widened here, as `FusedRuntime::forward_int4` does. This arm reads
        // the CALLER's activation, which arrives in the model's f16, while the
        // kernel stages f32. `contiguous` because a narrowed row is a view.
        let x = x.to_dtype(DType::F32)?.contiguous()?;
        x.apply_op1_no_bwd(&Int4Op { rt: self, proj })
    }
}

struct Int4Op<'a> {
    rt: &'a MetalRuntime,
    proj: &'a MetalInt4Proj,
}

impl CustomOp1 for Int4Op<'_> {
    fn name(&self) -> &'static str {
        "llvq-metal-q4"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("{}: the int4 kernel has no CPU path", self.proj.name)
    }

    fn metal_fwd(&self, storage: &MetalStorage, layout: &Layout) -> Result<(MetalStorage, Shape)> {
        let p = self.proj;
        if storage.dtype() != DType::F32 {
            candle_core::bail!("{}: f32 only, got {:?}", p.name, storage.dtype());
        }
        let (start, end) = layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::msg(format!("{}: non-contiguous activation", p.name)))?;
        if end - start != p.d_in {
            candle_core::bail!("{}: activation of {} values for d_in={}", p.name, end - start, p.d_in);
        }
        let pipe = self.rt.pipeline("tv_q4_metal")?;
        let dev = storage.device().clone();
        let out = dev.new_buffer(p.d_out, DType::F32, "llvq-metal-q4")?;
        let enc = dev.command_encoder()?;
        enc.set_compute_pipeline_state(&pipe);
        use candle_metal_kernels::utils::set_param;
        set_param(&enc, 0, (&*p.wq, 0usize));
        set_param(&enc, 1, (&*p.scales, 0usize));
        set_param(&enc, 2, (&*p.biases, 0usize));
        // Bound AT the offset, so the kernel's `x[i]` starts at the row.
        set_param(&enc, 3, (storage.buffer(), start * DType::F32.size_in_bytes()));
        set_param(&enc, 4, (&*out, 0usize));
        set_param(&enc, 5, p.d_in as u32);
        set_param(&enc, 6, p.gpr as u32);
        enc.set_threadgroup_memory_length(0, p.d_in * DType::F32.size_in_bytes());
        for b in [&*p.wq, &*p.scales, &*p.biases, storage.buffer()] {
            enc.use_resource(b, objc2_metal::MTLResourceUsage::Read);
        }
        enc.use_resource(&*out, objc2_metal::MTLResourceUsage::Write);
        enc.dispatch_threads(
            objc2_metal::MTLSize { width: p.d_out * LANES, height: 1, depth: 1 },
            objc2_metal::MTLSize { width: GROUP, height: 1, depth: 1 },
        );
        let mut dims = layout.dims().to_vec();
        *dims.last_mut().expect("rank >= 1") = p.d_out;
        Ok((MetalStorage::new(out, dev, p.d_out, DType::F32), Shape::from(dims)))
    }
}

// ---------------------------------------------------------------------------
// The q8 embedding, and the lm_head that reads the same table.
//
// 413.3 MB of the served 4B's 1.39 GB, so it is not a detail. One table, two
// ends: `gather` for the token lookup and `project` for the logits.
// ---------------------------------------------------------------------------

/// The int8 g64 embedding table on the device.
pub struct MetalEmbedTable {
    pub name: String,
    pub vocab: usize,
    pub d: usize,
    gpr: usize,
    wq: Arc<Buffer>,
    scales: Arc<Buffer>,
    biases: Arc<Buffer>,
}

impl MetalRuntime {
    /// Upload one q8 table. A tied model uploads ONE and points both ends at it.
    #[allow(clippy::too_many_arguments)]
    pub fn upload_embed(
        &self,
        name: &str,
        vocab: usize,
        d: usize,
        gpr: usize,
        wq: &[u32],
        scales: &[u16],
        biases: &[u16],
    ) -> Result<Arc<MetalEmbedTable>> {
        if !d.is_multiple_of(4) {
            candle_core::bail!("{name}: d {d} is not a multiple of 4, so a row would start \
                                at a shifted byte");
        }
        if wq.len() != vocab * d / 4 {
            candle_core::bail!("{name}: {} words for {vocab}x{d} int8", wq.len());
        }
        if scales.len() != vocab * gpr || biases.len() != vocab * gpr {
            candle_core::bail!("{name}: {} scales and {} biases for {vocab}x{gpr}",
                               scales.len(), biases.len());
        }
        Ok(Arc::new(MetalEmbedTable {
            name: name.to_string(),
            vocab,
            d,
            gpr,
            wq: self.device.new_buffer_with_data(wq)?,
            scales: self.device.new_buffer_with_data(scales)?,
            biases: self.device.new_buffer_with_data(biases)?,
        }))
    }
}

/// The table plus the runtime that can launch it.
pub struct MetalEmbed {
    rt: Arc<MetalRuntime>,
    table: Arc<MetalEmbedTable>,
}

impl MetalEmbed {
    pub fn new(rt: Arc<MetalRuntime>, table: Arc<MetalEmbedTable>) -> Self {
        Self { rt, table }
    }

    /// The inner table, so the loader can compare two ends with `Arc::ptr_eq`.
    ///
    /// The tie is NOT observable on the `MetalEmbed`: a tied model wraps one
    /// table in two of them. It IS observable here, which is where
    /// `load_resolved` checks it against `tie_word_embeddings`, exactly as the
    /// CUDA loader does on its own buffers.
    pub fn table(&self) -> &Arc<MetalEmbedTable> {
        &self.table
    }
}

impl crate::device::QuantEmbedTable for MetalEmbed {
    fn gather(&self, ids: &Tensor) -> Result<Tensor> {
        let ids = ids.to_dtype(DType::U32)?.contiguous()?;
        ids.apply_op1_no_bwd(&GatherOp { rt: &self.rt, t: &self.table })?
            .to_dtype(DType::F16)
    }

    fn project(&self, h: &Tensor) -> Result<Tensor> {
        let dims = h.dims();
        let d = *dims.last().expect("rank >= 1");
        if d != self.table.d {
            candle_core::bail!("{}: hidden of {d} for d={}", self.table.name, self.table.d);
        }
        let h = h.to_dtype(DType::F16)?.contiguous()?;
        h.apply_op1_no_bwd(&HeadOp { rt: &self.rt, t: &self.table })?
            .to_dtype(DType::F16)
    }
}

struct GatherOp<'a> {
    rt: &'a MetalRuntime,
    t: &'a MetalEmbedTable,
}

impl CustomOp1 for GatherOp<'_> {
    fn name(&self) -> &'static str {
        "llvq-metal-q8-gather"
    }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("{}: the q8 gather has no CPU path", self.t.name)
    }
    fn metal_fwd(&self, storage: &MetalStorage, layout: &Layout) -> Result<(MetalStorage, Shape)> {
        let t = self.t;
        if storage.dtype() != DType::U32 {
            candle_core::bail!("{}: ids must be u32, got {:?}", t.name, storage.dtype());
        }
        let (start, end) = layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::msg(format!("{}: non-contiguous ids", t.name)))?;
        let n_ids = end - start;
        if n_ids == 0 {
            candle_core::bail!("{}: a gather of zero ids", t.name);
        }
        let pipe = self.rt.pipeline("emb_q8_gather_metal")?;
        let dev = storage.device().clone();
        let out = dev.new_buffer(n_ids * t.d, DType::F32, "llvq-metal-q8-gather")?;
        let enc = dev.command_encoder()?;
        enc.set_compute_pipeline_state(&pipe);
        use candle_metal_kernels::utils::set_param;
        set_param(&enc, 0, (&*t.wq, 0usize));
        set_param(&enc, 1, (&*t.scales, 0usize));
        set_param(&enc, 2, (&*t.biases, 0usize));
        set_param(&enc, 3, (storage.buffer(), 0usize));
        set_param(&enc, 4, (&*out, 0usize));
        set_param(&enc, 5, t.d as u32);
        set_param(&enc, 6, t.gpr as u32);
        // In ELEMENTS: the kernel reads `ids[ids_off + tok]`.
        set_param(&enc, 7, start as u32);
        for b in [&*t.wq, &*t.scales, &*t.biases, storage.buffer()] {
            enc.use_resource(b, objc2_metal::MTLResourceUsage::Read);
        }
        enc.use_resource(&*out, objc2_metal::MTLResourceUsage::Write);
        enc.dispatch_threads(
            objc2_metal::MTLSize { width: n_ids * GROUP, height: 1, depth: 1 },
            objc2_metal::MTLSize { width: GROUP, height: 1, depth: 1 },
        );
        let mut dims = layout.dims().to_vec();
        dims.push(t.d);
        Ok((MetalStorage::new(out, dev, n_ids * t.d, DType::F32), Shape::from(dims)))
    }
}

struct HeadOp<'a> {
    rt: &'a MetalRuntime,
    t: &'a MetalEmbedTable,
}

impl CustomOp1 for HeadOp<'_> {
    fn name(&self) -> &'static str {
        "llvq-metal-q8-head"
    }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("{}: the q8 head has no CPU path", self.t.name)
    }
    fn metal_fwd(&self, storage: &MetalStorage, layout: &Layout) -> Result<(MetalStorage, Shape)> {
        let t = self.t;
        if storage.dtype() != DType::F16 {
            candle_core::bail!("{}: the head reads f16, got {:?}", t.name, storage.dtype());
        }
        let (start, end) = layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::msg(format!("{}: non-contiguous hidden", t.name)))?;
        let rows = (end - start) / t.d;
        if rows * t.d != end - start {
            candle_core::bail!("{}: {} values is not a whole number of rows of {}",
                               t.name, end - start, t.d);
        }
        let staged = t.d * DType::F32.size_in_bytes();
        if staged > THREADGROUP_LIMIT {
            candle_core::bail!("{}: staging d={} wants {staged} B against {THREADGROUP_LIMIT}",
                               t.name, t.d);
        }
        let pipe = self.rt.pipeline("tv_q8_metal")?;
        let dev = storage.device().clone();
        let out = dev.new_buffer(rows * t.vocab, DType::F32, "llvq-metal-q8-head")?;
        use candle_metal_kernels::utils::set_param;
        // One launch a row, which is what the CUDA host does: the kernel
        // computes one vocabulary row per SIMD-group and takes one activation.
        for r in 0..rows {
            let enc = dev.command_encoder()?;
            enc.set_compute_pipeline_state(&pipe);
            set_param(&enc, 0, (&*t.wq, 0usize));
            set_param(&enc, 1, (&*t.scales, 0usize));
            set_param(&enc, 2, (&*t.biases, 0usize));
            set_param(&enc, 3, (storage.buffer(), 0usize));
            set_param(&enc, 4, (&*out, 0usize));
            set_param(&enc, 5, t.d as u32);
            set_param(&enc, 6, t.gpr as u32);
            set_param(&enc, 7, (start + r * t.d) as u32);
            set_param(&enc, 8, (r * t.vocab) as u32);
            enc.set_threadgroup_memory_length(0, staged);
            for b in [&*t.wq, &*t.scales, &*t.biases, storage.buffer()] {
                enc.use_resource(b, objc2_metal::MTLResourceUsage::Read);
            }
            enc.use_resource(&*out, objc2_metal::MTLResourceUsage::Write);
            enc.dispatch_threads(
                objc2_metal::MTLSize { width: t.vocab * LANES, height: 1, depth: 1 },
                objc2_metal::MTLSize { width: GROUP, height: 1, depth: 1 },
            );
        }
        let mut dims = layout.dims().to_vec();
        *dims.last_mut().expect("rank >= 1") = t.vocab;
        Ok((MetalStorage::new(out, dev, rows * t.vocab, DType::F32), Shape::from(dims)))
    }
}

// ---------------------------------------------------------------------------
// The port, implemented. From here `model.rs` cannot tell a card from a Mac.
// ---------------------------------------------------------------------------

/// One Tetra projection, its runtime and its rotation, as the model sees it.
pub struct MetalLattice {
    rt: Arc<MetalRuntime>,
    proj: MetalTetraProj,
    rot: Option<Arc<MetalRotation>>,
    key: Option<crate::fused::RotKey>,
}

impl crate::device::LatticeProj for MetalLattice {
    fn name(&self) -> &str {
        &self.proj.name
    }
    fn d_out(&self) -> usize {
        self.proj.d_out
    }
    fn d_in(&self) -> usize {
        self.proj.d_in
    }
    fn rotation(&self) -> Option<crate::fused::RotKey> {
        self.key
    }
    fn prepare(&self, x: &Tensor) -> Result<Tensor> {
        match &self.rot {
            Some(r) => self.rt.rotate(r, &self.proj.name, x),
            // An artifact without a rotation. The CUDA adapter refuses this
            // path by name rather than guess; so does this one.
            None => candle_core::bail!(
                "{}: artifact without rotation, path not covered",
                self.proj.name
            ),
        }
    }
    fn prepare_rows(&self, xs: &Tensor, rows: usize) -> Result<Tensor> {
        match &self.rot {
            Some(r) => self.rt.rotate_rows(r, &self.proj.name, xs, rows),
            None => candle_core::bail!(
                "{}: artifact without rotation, path not covered",
                self.proj.name
            ),
        }
    }
    fn matvec(&self, xr: &Tensor, _out_dims: &[usize]) -> Result<Tensor> {
        // Narrowed HERE, not in the kernel.
        //
        // The CUDA twin ends in `f2h` and stores f16; these kernels store f32,
        // which was a deliberate staging decision. The model runs in f16, so
        // the adapter narrows at its boundary instead. candle's conversion is
        // the `half` crate's, which is IEEE round-to-nearest-even, the same
        // rule `f2h` implements. So the two paths agree to the bit, and what
        // differs is only WHERE the rounding happens, plus one f32 buffer's
        // worth of traffic that a later lot can remove.
        self.rt.matvec(&self.proj, xr)?.to_dtype(DType::F16)
    }
    /// One row a launch. There is no Metal prefill kernel, so the caller fans
    /// a chunk out row by row: the same arithmetic, more launches.
    fn rows_per_launch(&self) -> usize {
        1
    }
}

/// One int4 projection, as the model sees it.
pub struct MetalInt4 {
    rt: Arc<MetalRuntime>,
    proj: MetalInt4Proj,
}

impl crate::device::Int4Proj for MetalInt4 {
    fn name(&self) -> &str {
        &self.proj.name
    }
    fn d_out(&self) -> usize {
        self.proj.d_out
    }
    fn d_in(&self) -> usize {
        self.proj.d_in
    }
    fn matvec(&self, x: &Tensor, _out_dims: &[usize]) -> Result<Tensor> {
        // See MetalLattice::matvec: narrowed at the boundary, not in the kernel.
        self.rt.matvec_int4(&self.proj, x)?.to_dtype(DType::F16)
    }
}

// ---------------------------------------------------------------------------
// The loader. One door, and after it `model.rs` names no backend.
// ---------------------------------------------------------------------------

impl MetalRuntime {
    /// A runtime from the portable table builder, which is where the CUDA
    /// loader gets the same numbers.
    ///
    /// `prefixes` and `suffixes` arrive packed into `u32` because that is the
    /// shape the CUDA upload path wants. The MSL reads them as `uchar*`, and
    /// on a little-endian machine those are the same bytes. Both sides are
    /// Apple silicon or x86, so this is a fact rather than an assumption.
    pub fn from_tables(
        device: &candle_core::Device,
        t: &crate::fused::Tetra48Tables,
        tile: usize,
    ) -> Result<Self> {
        let dev = match device {
            candle_core::Device::Metal(d) => d.clone(),
            other => candle_core::bail!("the Metal runtime wants a Metal device, got {other:?}"),
        };
        if !tile.is_power_of_two() || !(32..=256).contains(&tile) {
            candle_core::bail!("tile {tile} is not a power of two in 32..=256");
        }
        Ok(Self {
            tables: MetalTables {
                rows: dev.new_buffer_with_data(&t.rows)?,
                prefixes: dev.new_buffer_with_data(&t.prefixes)?,
                branches: dev.new_buffer_with_data(&t.branches)?,
                suffixes: dev.new_buffer_with_data(&t.suffixes)?,
                invnorm: dev.new_buffer_with_data(&t.invnorm)?,
                // A served model keeps no host copy: it would pay for the
                // tables twice and its CPU arm is not a fallback.
                host: None,
            },
            device: dev,
            tile,
            pipelines: RwLock::new(HashMap::new()),
        })
    }

    /// [`Self::upload`] from the row-strided `u32` stream the reader hands
    /// back, rather than from bytes.
    #[allow(clippy::too_many_arguments)]
    pub fn upload_stream(
        &self,
        name: &str,
        d_out: usize,
        d_in: usize,
        nblocks: usize,
        row_stride_u32: usize,
        words: &[u32],
        gscale: &[f32; 2],
        rscale: &[f32],
        tail: &[u16],
    ) -> Result<MetalTetraProj> {
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        self.upload(name, d_out, d_in, nblocks, row_stride_u32, &bytes, gscale, rscale, tail)
    }
}

/// The served object on Metal, loaded from a sealed artifact.
///
/// The twin of `fused_cuda::load_resolved`, and deliberately not a copy of it:
/// everything portable comes from [`crate::fused`], which the CUDA loader uses
/// too. What is written here is the device half, about a hundred lines.
#[allow(clippy::too_many_arguments)]
pub fn load_resolved(
    path: &str,
    device: &candle_core::Device,
    dtype: DType,
    layout: crate::fused::FusedLayout,
    emode: crate::fused::EmbedMode,
    share: crate::rotplan::RotShare,
    fuse: crate::fused::FuseMode,
    kv: crate::kvq::KvMode,
    origin: Option<&str>,
) -> Result<crate::fused::FusedSealed> {
    let from = |var: &str| origin.unwrap_or(var).to_string();
    if dtype != DType::F16 {
        candle_core::bail!("the Metal fused path serves f16, asked for {dtype:?}");
    }
    if layout != crate::fused::FusedLayout::Tetra48 {
        candle_core::bail!(
            "{}: the Metal path carries the Tetra48 kernels only, asked for {layout:?}",
            from("LLVQ_FUSED_LAYOUT")
        );
    }
    if fuse == crate::fused::FuseMode::On {
        candle_core::bail!(
            "{}: there is no segmented Metal kernel, so a fused group cannot launch",
            from("LLVQ_FUSE")
        );
    }
    crate::fused::check_fuse(layout, share, fuse).map_err(candle_core::Error::msg)?;

    let mut model = crate::fused::load_with(path, layout, fuse).map_err(candle_core::Error::msg)?;
    let rot_launches = crate::rotplan::rot_launches(share, &model.matrices, &model.groups);
    let matvec_launches = model.matrices.len() + model.int4.len() + model.groups.len();
    let quantized_weights = model.quantized_weights;
    let carried_weights = model.carried_weights;
    let (file_bytes, runtime_bytes) = (model.file_bytes, model.runtime_bytes);

    let config: candle_transformers::models::qwen3::Config =
        serde_json::from_slice(&model.config_json)
            .map_err(|e| candle_core::Error::msg(format!("{path}: config.json: {e}")))?;
    let tokenizer = tokenizers::Tokenizer::from_bytes(&model.tokenizer_json)
        .map_err(|e| candle_core::Error::msg(format!("{path}: tokenizer.json: {e}")))?;

    let embed_tables = match emode {
        crate::fused::EmbedMode::F16 => None,
        crate::fused::EmbedMode::Q8 => Some(
            crate::fused::take_embed_tables(&mut model.raw, config.tie_word_embeddings)
                .map_err(|e| candle_core::Error::msg(format!("{path}: {e}")))?,
        ),
    };

    // The tile: no measured row for Apple, so the shader's own default.
    // `tuile-l40s-2026-09-20` measured 64 on sm_89 and nothing here.
    let tetra = llvq_search::tetra::Tetra::new();
    let tables = crate::fused::tetra48_tables(&tetra);
    let rt = Arc::new(MetalRuntime::from_tables(device, &tables, 64)?);

    let mut rotations: HashMap<crate::fused::RotKey, Arc<MetalRotation>> = HashMap::new();
    for (&key, t) in &model.rotations {
        rotations.insert(key, Arc::new(rt.upload_rotation(t)?));
    }

    let mut by_site: HashMap<(usize, String), crate::model::Proj> = HashMap::new();
    for m in &model.matrices {
        let crate::fused::HostStream::Tetra48 { words, stride_u32 } = &m.stream else {
            candle_core::bail!("{}: not a Tetra48 stream on a Tetra48 load", m.name);
        };
        let proj = rt.upload_stream(
            &m.name, m.d_out, m.d_in, m.nblocks, *stride_u32 as usize, words, &m.gscale, &m.rscale, &m.tail,
        )?;
        let rot = match m.rotation {
            None => None,
            Some(k) => Some(
                rotations
                    .get(&k)
                    .ok_or_else(|| candle_core::Error::msg(format!("rotation {k:?} missing")))?
                    .clone(),
            ),
        };
        let (layer, name) = llvq_artifact::split_name(&m.name)
            .map_err(|e| candle_core::Error::msg(e.to_string()))?;
        by_site.insert(
            (layer, name),
            crate::model::Proj::Lattice(Arc::new(MetalLattice {
                rt: rt.clone(),
                proj,
                rot,
                key: m.rotation,
            })),
        );
    }

    for q in &model.int4 {
        let gpr = q.d_in / q.group;
        let wq: Vec<u32> = q
            .packed
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let proj = rt.upload_int4(&q.name, q.d_out, q.d_in, gpr, &wq, &q.scales, &q.biases)?;
        let (layer, name) = llvq_artifact::split_name(&q.name)
            .map_err(|e| candle_core::Error::msg(e.to_string()))?;
        by_site.insert(
            (layer, name),
            crate::model::Proj::Int4(Arc::new(MetalInt4 { rt: rt.clone(), proj })),
        );
    }
    finish(
        path, device, dtype, layout, emode, share, fuse, kv, origin, model, config, tokenizer,
        embed_tables, rt, by_site, rot_launches, matvec_launches, quantized_weights,
        carried_weights, file_bytes, runtime_bytes,
    )
}

/// The half of the load that names no device.
///
/// Carried tensors, the embedding, the `VarBuilder`, the model and the
/// bookkeeping. It is a separate function because it is the part a third
/// backend would reuse verbatim, and because `load_resolved` above is then
/// short enough to read as what it is: an upload loop.
#[allow(clippy::too_many_arguments)]
fn finish(
    path: &str,
    device: &candle_core::Device,
    dtype: DType,
    layout: crate::fused::FusedLayout,
    emode: crate::fused::EmbedMode,
    share: crate::rotplan::RotShare,
    fuse: crate::fused::FuseMode,
    kv: crate::kvq::KvMode,
    origin: Option<&str>,
    model: crate::fused::FusedModel,
    config: candle_transformers::models::qwen3::Config,
    tokenizer: tokenizers::Tokenizer,
    embed_tables: Option<crate::fused::EmbedTables>,
    rt: Arc<MetalRuntime>,
    mut by_site: HashMap<(usize, String), crate::model::Proj>,
    rot_launches: usize,
    matvec_launches: usize,
    quantized_weights: usize,
    carried_weights: usize,
    file_bytes: u64,
    runtime_bytes: u64,
) -> Result<crate::fused::FusedSealed> {
    let from = |var: &str| origin.unwrap_or(var).to_string();

    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    let mut carried_bytes = 0u64;
    for t in &model.raw {
        carried_bytes += t.len() as u64 * 2;
        tensors.insert(
            t.name.clone(),
            Tensor::from_vec(t.to_f32(), t.dims.clone(), device)?.to_dtype(dtype)?,
        );
    }

    let quant_embed = match &embed_tables {
        None => {
            let carried = crate::fused::carried_embed_tables(&model.raw);
            println!(
                "{}",
                crate::fused::EmbedReport::new(crate::fused::EmbedMode::F16, &carried)
                    .line(&from("LLVQ_EMBED"))
            );
            None
        }
        Some(tables) => {
            let to_upload = tables.buffers();
            let report = crate::fused::EmbedReport::new(crate::fused::EmbedMode::Q8, &to_upload);
            println!("{}", report.line(&from("LLVQ_EMBED")));
            let mut uploaded: Vec<Arc<MetalEmbedTable>> = Vec::with_capacity(to_upload.len());
            for t in &to_upload {
                let llvq_artifact::RawData::Quant(q) = &t.data else {
                    candle_core::bail!("{}: not a quantized tensor", t.name);
                };
                if q.bits != 8 {
                    candle_core::bail!("{}: int{} where the kernels want int8", t.name, q.bits);
                }
                if t.dims.len() != 2 {
                    candle_core::bail!("{}: dims {:?}, an embedding is 2-D", t.name, t.dims);
                }
                let (vocab, d) = (t.dims[0], t.dims[1]);
                let wq: Vec<u32> = q
                    .packed
                    .chunks_exact(4)
                    .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                uploaded.push(rt.upload_embed(
                    &t.name, vocab, d, d / q.group, &wq, &q.scales, &q.biases,
                )?);
                carried_bytes += (q.packed.len() + 2 * q.scales.len() + 2 * q.biases.len()) as u64;
            }
            Some((uploaded, tables.wiring()))
        }
    };
    println!(
        "total expected on the device: {:.2} GB (projections {:.2} + carried {:.2})",
        (runtime_bytes + carried_bytes) as f64 / 1e9,
        runtime_bytes as f64 / 1e9,
        carried_bytes as f64 / 1e9
    );

    // The tie, checked on the TABLES and not on the handles. Two `MetalEmbed`
    // values wrap one table when the ends are tied, so `Arc::ptr_eq` on them
    // is false either way. This is the same check `fused_cuda` makes on its
    // own buffers, and `device.rs` says so where the trait is declared.
    if let Some((bufs, (ie, ih))) = &quant_embed {
        let tied = Arc::ptr_eq(&bufs[*ie], &bufs[*ih]);
        if tied != config.tie_word_embeddings {
            candle_core::bail!(
                "inconsistent q8 wiring: embedding and lm_head {} the same table while \
                 tie_word_embeddings = {}",
                if tied { "share" } else { "do not share" },
                config.tie_word_embeddings
            );
        }
    }

    let vb = candle_nn::VarBuilder::from_tensors(tensors, dtype, device);
    let total_sites = by_site.len();
    let mut claimed = 0usize;
    let mut take = |layer: usize, name: &str| {
        by_site.remove(&(layer, name.to_string())).inspect(|_| {
            claimed += 1;
        })
    };
    let mut qwen = match &quant_embed {
        None => crate::model::Qwen3::new_with(&config, vb, &mut take, kv)?,
        Some((bufs, (ie, ih))) => crate::model::Qwen3::new_with_embed(
            &config,
            vb,
            &mut take,
            crate::model::Embed::Q8(Arc::new(MetalEmbed::new(rt.clone(), bufs[*ie].clone()))),
            crate::model::Head::Q8(Arc::new(MetalEmbed::new(rt.clone(), bufs[*ih].clone()))),
            kv,
        )?,
    };
    if claimed != total_sites {
        candle_core::bail!(
            "{path}: {claimed} of {total_sites} device projections were claimed by the model. \
             The rest would be served dense without saying so"
        );
    }
    qwen.set_rot_share(share);

    Ok(crate::fused::FusedSealed {
        model: qwen,
        tokenizer,
        config,
        layout,
        // No prefill kernel on Metal, and no measured tile: `rows` is 1 and
        // `tile` is what the shader was compiled at.
        prefill: crate::device::Prefill { rows: 1, tile: rt.tile, from_env: false },
        embed_mode: emode,
        rot_share: share,
        rot_launches,
        fuse,
        matvec_launches,
        quantized_weights,
        carried_weights,
        file_bytes,
        runtime_bytes,
        carried_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::h2f;

    /// The IEEE definition of binary16, in f64, from the fields.
    ///
    /// Not a second copy of `h2f`: this computes the VALUE from the exponent
    /// and the significand, where `h2f` assembles a bit pattern. A shared
    /// mistake would have to be a mistake about IEEE itself.
    fn ieee_half(h: u16) -> Option<f64> {
        let s = if h & 0x8000 != 0 { -1.0f64 } else { 1.0 };
        let e = ((h >> 10) & 0x1f) as i32;
        let m = (h & 0x3ff) as f64;
        match e {
            // Subnormal: no hidden bit, fixed exponent of -14.
            0 => Some(s * (m / 1024.0) * 2f64.powi(-14)),
            // inf / NaN, which the caller checks separately.
            31 => None,
            _ => Some(s * (1.0 + m / 1024.0) * 2f64.powi(e - 15)),
        }
    }

    /// All 65,536 patterns. The subnormal and inf arms were both wrong until
    /// an audit of 2026-09-21, and the gate's Gaussian fixtures never drew one.
    #[test]
    fn h2f_widens_every_binary16_pattern() {
        let mut checked = 0u32;
        for b in 0u32..=0xffff {
            let h = b as u16;
            let got = h2f(h);
            match ieee_half(h) {
                Some(want) => {
                    assert_eq!(
                        got,
                        want as f32,
                        "0x{b:04x}: h2f {got:e} against the IEEE value {want:e}"
                    );
                    checked += 1;
                }
                None => {
                    // exp == 31: infinity when the significand is zero, NaN otherwise.
                    if h & 0x3ff == 0 {
                        assert!(got.is_infinite(), "0x{b:04x} is an infinity");
                        assert_eq!(got.is_sign_negative(), h & 0x8000 != 0, "0x{b:04x} sign");
                    } else {
                        assert!(got.is_nan(), "0x{b:04x} is a NaN");
                    }
                    checked += 1;
                }
            }
        }
        assert_eq!(checked, 65_536, "every pattern is judged");
    }

    /// The three patterns the earlier version got wrong, named.
    #[test]
    fn the_patterns_an_earlier_h2f_got_wrong() {
        // The smallest positive subnormal: 2^-24.
        assert_eq!(h2f(0x0001), 2f32.powi(-24));
        // The largest subnormal: 1023 * 2^-24.
        assert_eq!(h2f(0x03ff), 1023.0 * 2f32.powi(-24));
        // Positive infinity, which an earlier version returned as 65536.
        assert!(h2f(0x7c00).is_infinite() && h2f(0x7c00) > 0.0);
    }
}
