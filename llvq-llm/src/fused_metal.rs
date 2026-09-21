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

const SOURCE: &str = include_str!("../kernels/llvq_tetra48.metal");

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
        let want_bytes = d_out * row_stride_u32 * 4;
        if words.len() != want_bytes {
            candle_core::bail!(
                "{name}: {} stream bytes for {d_out} rows of {row_stride_u32} words, which \
                 wants {want_bytes}",
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
        let src = format!("#define LLVQ_TILE_BLOCKS {}u\n{SOURCE}", self.tile);
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
