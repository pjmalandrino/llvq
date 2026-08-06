//! The fused projections, running inside candle.
//!
//! This is where the kernel stops being a bench and starts being inference.
//! It holds the encoded streams on the device — `Planes14` by default,
//! `Slot32` under `LLVQ_FUSED_LAYOUT=slot32` — and replaces one linear layer
//! with two launches:
//!
//! ```text
//! x (f16) ──rot_apply──▶ x' (f32, rotated basis) ──tv_planes_h / tv_slot_h──▶ y (f16)
//! ```
//!
//! ## Three things it does not do, on purpose
//!
//! * **It does not un-rotate.** The stored weights are `W' = W Qᵀ`, so the
//!   activation carries the `Q`. The identity is pinned on the CPU by
//!   `llvq-artifact/tests/fused_path_matches_dense.rs`; getting it backwards
//!   produces finite, plausible, wrong numbers and no error at all.
//! * **It does not handle more than one token per call.** `tv_slot` is a
//!   matrix–*vector* product: one warp per output row, one activation staged
//!   in shared memory. A prompt of `l` tokens loops `l` times, which is
//!   correct and slower than a GEMM would be — acceptable for generation
//!   (`l = 1` after the prompt), useless for scoring a 2048-token window.
//!   Perplexity therefore keeps the dense path, and `bin/ppl` is untouched.
//! * **It does not keep a dense copy as a fallback.** That would put the 8 GB
//!   back and forfeit the only gain that is not in dispute — 8.04 GB of f16
//!   weights against 3.28 with the projections encoded.
//!
//! ## Why the stream comes from candle
//!
//! `Cuda::on_stream` compiles our module onto candle's own stream. Both would
//! land on the same primary context either way, so the pointers were never in
//! question; what sharing buys is **ordering**. On a separate stream, a
//! `tv_slot` reading an activation candle has not finished writing is a race
//! that reproduces once in a hundred runs, on the card, in a billed job.

use std::collections::HashMap;
use std::sync::Mutex;

use candle_core::cuda_backend::cudarc::driver::{
    CudaFunction, CudaSlice, LaunchConfig, PushKernelArg,
};
use candle_core::{CudaStorage, DType, Device, Layout, Shape, Tensor};
use half::f16;

use crate::fused::{FusedLayout, FusedMatrix, FusedModel, HostStream, RotKey, RotationTables};

/// Threads per block for `tv_slot`: 256 = eight rows per block, the shape the
/// bench has always measured.
const THREADS: u32 = 256;
/// Blocks staged per tile — must equal the `TILE_BLOCKS` handed to NVRTC.
const TILE_BLOCKS: usize = 128;
/// Entries in the device class table. The class field is nine bits, so 512
/// are addressable while 384 exist; the tail stays the origin so a truncated
/// or corrupt index cannot address out of bounds.
const TABLE_ENTRIES: usize = 512;
const REC_WORDS: usize = 6;

/// The Planes14 kernel sources, embedded like every other kernel so a run is
/// reproducible from the binary alone. The two committed files are reused
/// verbatim from `llvq-cuda` (they belong to the C1 lot and are pinned by
/// planesbench); `tv_planes_h.cu` is this crate's own — the half-storing
/// entry point only the inference path needs.
const PLANES_CUH_EMBED: &str = include_str!("../../llvq-cuda/kernels/llvq_planes.cuh");
const PLANES_CU_EMBED: &str = include_str!("../../llvq-cuda/kernels/planes.cu");
const PLANES_H_CU_EMBED: &str = include_str!("../kernels/tv_planes_h.cu");

/// The three Planes14 parts, honouring `LLVQ_KERNEL_DIR` with the same
/// contract as `load_sources_many`: without the variable, the embedded text;
/// with it, **all** the files from the directory, disclosed loudly by the
/// caller — mixing embedded and overridden parts would make the printed
/// sha256 untraceable.
fn load_planes_sources() -> Result<([String; 3], Option<String>), String> {
    match std::env::var("LLVQ_KERNEL_DIR") {
        Err(_) => Ok((
            [
                PLANES_CUH_EMBED.to_string(),
                PLANES_CU_EMBED.to_string(),
                PLANES_H_CU_EMBED.to_string(),
            ],
            None,
        )),
        Ok(dir) => {
            let rd = |n: &str| {
                std::fs::read_to_string(std::path::Path::new(&dir).join(n))
                    .map_err(|e| format!("LLVQ_KERNEL_DIR={dir} : {n} : {e}"))
            };
            Ok((
                [rd("llvq_planes.cuh")?, rd("planes.cu")?, rd("tv_planes_h.cu")?],
                Some(dir),
            ))
        }
    }
}

/// The rotation tables, on the device.
struct RotBuffers {
    signbits: CudaSlice<u32>,
    small: CudaSlice<f32>,
    n: u32,
    m: u32,
    k: u32,
    inv: f32,
    /// One block, so the only knob is the thread count.
    threads: u32,
}

/// A matrix's payload on the device, mirroring [`HostStream`] variant for
/// variant. The `Planes14` arm carries **no bases slice at all** — reading or
/// launching with one on the planes path is a compile error, not a bug class.
enum DeviceStream {
    Slot32 {
        words: CudaSlice<u32>,
        bases: CudaSlice<u32>,
    },
    Planes14 {
        words: CudaSlice<u32>,
    },
}

/// One projection's weights, on the device.
pub struct FusedProj {
    pub name: String,
    pub d_out: usize,
    pub d_in: usize,
    nblocks: u32,
    tail_w: u32,
    stream: DeviceStream,
    gscale: CudaSlice<f32>,
    rscale: CudaSlice<f32>,
    tail: CudaSlice<f32>,
    rotation: Option<RotKey>,
}

/// The module, the shared tables, and the device everything lives on.
pub struct FusedRuntime {
    cuda: llvq_cuda::gpu::Cuda,
    f_rot: CudaFunction,
    /// `tv_planes_h` or `tv_slot_h`, whichever [`FusedLayout`] named — one
    /// kernel per runtime, chosen with the layout, so a stream and a kernel
    /// of different layouts cannot meet.
    f_matvec: CudaFunction,
    tab: CudaSlice<u32>,
    rotations: HashMap<RotKey, RotBuffers>,
    device: candle_core::CudaDevice,
    /// Largest `d_in` any projection takes — the staging bound the rotation
    /// kernel needs in shared memory.
    max_d_in: usize,
    /// One scratch buffer per distinct activation width, for the rotated
    /// activation that only ever lives between the two launches.
    ///
    /// The first version allocated it fresh on every projection: 252 `cudaMalloc`
    /// **and** 252 `memset` per token, for a buffer `rot_apply` overwrites in
    /// full and nothing ever reads before it does. Measured cost of that
    /// naivety: the fused path came out 4 % *slower* than dense despite the
    /// kernel itself being 1.9× faster.
    ///
    /// ## Why sharing one buffer across calls is safe
    ///
    /// Not because of the mutex — that only orders the Rust side, and the
    /// launches are asynchronous, so the lock is long released by the time
    /// either kernel runs. It is safe because **both launches go on the same
    /// stream**, which executes in issue order: call *n*'s `tv_slot` has
    /// finished reading the scratch before call *n+1*'s `rot_apply` starts
    /// writing it. Put the two kernels on different streams and this becomes
    /// a race that reproduces once in a hundred runs.
    scratch: Mutex<HashMap<usize, CudaSlice<f32>>>,
}

impl FusedRuntime {
    /// Upload a loaded model and compile the kernels onto candle's stream.
    pub fn new(model: &FusedModel, device: &Device) -> candle_core::Result<(Self, Vec<FusedProj>)> {
        let dev = device.as_cuda_device()?.clone();
        let stream = dev.cuda_stream();

        // The Slot32 translation unit is bit-identical to what shipped before
        // the layout switch existed — that arm is the comparison and the
        // fallback, and its register allocation must not move because a new
        // layout joined the build. Planes14 appends its three parts (in
        // planesbench's proven order: llvq_planes.cuh needs llvq_slot.cuh,
        // planes.cu needs matvec.cu) plus the half-storing entry point.
        let sources = llvq_cuda::load_sources_many(&["llvq_slot.cuh", "matvec.cu", "llvq_rot.cuh", "rotate.cu"])
            .map_err(candle_core::Error::msg)?;
        let planes = match model.layout {
            FusedLayout::Slot32 => None,
            FusedLayout::Planes14 => {
                Some(load_planes_sources().map_err(candle_core::Error::msg)?)
            }
        };
        let defines = format!("#define TILE_BLOCKS {TILE_BLOCKS}u\n");
        let mut parts: Vec<&str> = std::iter::once(defines.as_str())
            .chain(sources.parts.iter().map(String::as_str))
            .collect();
        if let Some((pp, overridden)) = &planes {
            parts.extend(pp.iter().map(String::as_str));
            if let Some(d) = overridden {
                eprintln!("⚠️ SOURCES Planes14 SURCHARGÉES depuis {d}");
            }
        }
        let src = llvq_cuda::gpu::KernelSource::new(&parts);
        let cuda = llvq_cuda::gpu::Cuda::on_stream(stream, &src).map_err(candle_core::Error::msg)?;

        // The register report is a contract, not a diagnostic: the block
        // decoders keep their accumulators and `rot_mix` a KMAX-wide column
        // in registers, and a spill costs occupancy without changing a
        // result. Checked on the kernel this runtime will actually launch.
        let matvec_name = match model.layout {
            FusedLayout::Planes14 => "tv_planes_h",
            FusedLayout::Slot32 => "tv_slot_h",
        };
        for name in [matvec_name, "rot_apply"] {
            let r = cuda.report(name).map_err(candle_core::Error::msg)?;
            if r.local_bytes != 0 {
                candle_core::bail!("{name} : {} octets de spill", r.local_bytes);
            }
        }
        let f_matvec = cuda.func(matvec_name).map_err(candle_core::Error::msg)?;
        let f_rot = cuda.func("rot_apply").map_err(candle_core::Error::msg)?;

        // The 384-entry class table both sides of the format share, laid out
        // as the kernel's `ClassRec { float vals[5]; u32 len; }`.
        let fd = llvq_search::fastdec::FastDecoder::new();
        let mut tab = vec![0u32; TABLE_ENTRIES * REC_WORDS];
        for e in 0..TABLE_ENTRIES {
            tab[e * REC_WORDS + REC_WORDS - 1] = 1;
        }
        for ci in 0..fd.n_classes() {
            let lv = fd.levels(ci);
            let norm = ((16 * lv.shell) as f64).sqrt();
            let base = (1 + ci) * REC_WORDS;
            for k in 0..lv.len {
                tab[base + k] = ((lv.values[k] as f64 / norm) as f32).to_bits();
            }
            tab[base + REC_WORDS - 1] = lv.len as u32;
        }
        let tab = cuda.up_u32(&tab).map_err(candle_core::Error::msg)?;

        let shared_limit = cuda.device().map_err(candle_core::Error::msg)?.shared_per_block as usize;
        let mut rotations = HashMap::new();
        for (&key, t) in &model.rotations {
            if t.n * 4 > shared_limit {
                candle_core::bail!(
                    "rotation de largeur {} : {} o de partagée demandés, la carte en offre {}. \
                     Le noyau à un bloc ne convient pas à cette largeur (cf. rotate.cu).",
                    t.n,
                    t.n * 4,
                    shared_limit
                );
            }
            rotations.insert(key, upload_rotation(&cuda, t)?);
        }

        let max_d_in = model.matrices.iter().map(|m| m.d_in).max().unwrap_or(0);
        let mut projs = Vec::with_capacity(model.matrices.len());
        for m in &model.matrices {
            projs.push(upload_matrix(&cuda, m, model.layout)?);
        }

        Ok((
            Self {
                cuda,
                f_rot,
                f_matvec,
                tab,
                rotations,
                device: dev,
                max_d_in,
                scratch: Mutex::new(HashMap::new()),
            },
            projs,
        ))
    }

    /// The widest activation any projection takes.
    ///
    /// The rotation kernel stages the whole vector in shared memory, so this
    /// is the bound a caller checks against the card before uploading
    /// anything — `new` refuses past it rather than corrupting.
    pub fn max_d_in(&self) -> usize {
        self.max_d_in
    }

    /// `y = W x` for one activation, through the fused path.
    ///
    /// `x` is `[.., d_in]` in f16; the result is `[.., d_out]` in f16, so the
    /// caller sees exactly what a `Linear` would have returned. The two
    /// conversions that costs are the price of a kernel that reads f16 and
    /// accumulates in f32 — the same arithmetic the dense path does, in the
    /// same order.
    pub fn forward(&self, proj: &FusedProj, x: &Tensor) -> candle_core::Result<Tensor> {
        let dims = x.dims();
        let d_in = *dims.last().expect("rank >= 1");
        if d_in != proj.d_in {
            candle_core::bail!("{} attend d_in={}, reçu {d_in}", proj.name, proj.d_in);
        }
        let rows: usize = dims[..dims.len() - 1].iter().product();

        // More than one vector: loop.
        //
        // This is not an optimization gap that can be papered over — `tv_slot`
        // is a matrix–*vector* product, one warp per output row with a single
        // activation staged in shared memory, so `l` tokens cost `l` launches.
        // The first version of this refused outright, which was wrong for a
        // reason the very first run found: **generation always begins by
        // running the whole prompt through**, so every `generate` call hits
        // `rows > 1` before it ever decodes a token.
        //
        // The cap survives that correction, because the danger it guarded
        // against is real: a 2048-token scoring window would issue half a
        // million launches and look like a hang. Prompts stay well under it,
        // and `bin/ppl` keeps the dense path.
        const MAX_ROWS: usize = 256;
        if rows > MAX_ROWS {
            candle_core::bail!(
                "{} : {rows} vecteurs d'un coup, au-delà de {MAX_ROWS}. Le noyau fusé est \
                 un matvec — il boucle, donc le coût est linéaire. La passe de score garde \
                 le chemin dense.",
                proj.name
            );
        }

        let x = x.contiguous()?.to_dtype(DType::F16)?;
        if rows > 1 {
            let flat = x.reshape((rows, d_in))?;
            let mut out = Vec::with_capacity(rows);
            for r in 0..rows {
                // `narrow` yields a view at an offset into the same buffer —
                // which is exactly why `rot_apply` takes `x_off`.
                out.push(self.forward(proj, &flat.narrow(0, r, 1)?)?);
            }
            let mut d = dims.to_vec();
            *d.last_mut().expect("rank >= 1") = proj.d_out;
            return Tensor::cat(&out, 0)?.reshape(d);
        }
        let out_shape = {
            let mut d = dims.to_vec();
            *d.last_mut().expect("rank >= 1") = proj.d_out;
            Shape::from(d)
        };
        let op = FusedOp {
            rt: self,
            proj,
            out_shape,
        };
        // No `to_dtype` here: the kernel already stored halves.
        x.apply_op1_no_bwd(&op)
    }

}

/// The `CustomOp1` candle needs to let us at the tensor's storage.
///
/// There is no public way to reach a `Tensor`'s device pointer other than this
/// trait, which is the right design: it keeps the layout in the picture. We
/// require contiguity rather than honouring arbitrary strides — the kernel
/// stages 24 consecutive floats per block and a strided activation would be
/// silently wrong.
struct FusedOp<'a> {
    rt: &'a FusedRuntime,
    proj: &'a FusedProj,
    out_shape: Shape,
}

impl candle_core::CustomOp1 for FusedOp<'_> {
    fn name(&self) -> &'static str {
        "llvq-fused-matvec"
    }

    fn cpu_fwd(
        &self,
        _: &candle_core::CpuStorage,
        _: &Layout,
    ) -> candle_core::Result<(candle_core::CpuStorage, Shape)> {
        candle_core::bail!("le noyau fusé LLVQ n'a pas de chemin CPU")
    }

    fn cuda_fwd(
        &self,
        storage: &CudaStorage,
        layout: &Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        let all = storage.as_cuda_slice::<f16>()?;
        // `(start, start + elem_count)` — a *range*, not `(offset, length)`.
        // Taking the second field for a length is silent at offset 0 and wrong
        // everywhere else, which is exactly how it survived the first run and
        // died on the second vector of the prompt.
        let (start, end) = layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::msg("activation non contiguë"))?;
        let len = end - start;
        if len != self.proj.d_in {
            candle_core::bail!("activation de {len} valeurs pour d_in={}", self.proj.d_in);
        }
        let rot = match self.proj.rotation {
            None => candle_core::bail!(
                "{} : artefact sans rotation — chemin non couvert, cf. fused_cuda.rs",
                self.proj.name
            ),
            Some(key) => self
                .rt
                .rotations
                .get(&key)
                .ok_or_else(|| candle_core::Error::msg(format!("rotation {key:?} absente")))?,
        };

        // f16, and uninitialised. Two things at once:
        //
        //  * `tv_slot_h` stores halves, so candle no longer needs a conversion
        //    kernel per projection — 252 launches a token, on a decode whose
        //    budget is half launch latency;
        //  * nothing zeroes it: the kernel writes `y[row]` for every row, the
        //    grid is exact and there is no bounds guard.
        let mut y = unsafe { self.rt.device.cuda_stream().alloc::<f16>(self.proj.d_out) }
            .map_err(|e| candle_core::Error::msg(format!("alloc y: {e}")))?;
        let shared = (TILE_BLOCKS * llvq_core::DIM * 4) as u32;

        // Both launches under one lock, so the scratch cannot be handed to a
        // second caller between them. See `FusedRuntime::scratch` for why the
        // *stream* is what makes the sharing sound, not this lock.
        {
            let mut pool = self
                .rt
                .scratch
                .lock()
                .map_err(|_| candle_core::Error::msg("scratch empoisonné"))?;
            let xr = match pool.entry(self.proj.d_in) {
                std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
                std::collections::hash_map::Entry::Vacant(e) => e.insert(
                    unsafe { self.rt.device.cuda_stream().alloc::<f32>(self.proj.d_in) }
                        .map_err(|er| candle_core::Error::msg(format!("alloc scratch: {er}")))?,
                ),
            };
            self.rt
                .cuda
                .launch_rot(
                    &self.rt.f_rot, all, &rot.signbits, &rot.small, xr, rot.n, rot.m, rot.k,
                    rot.inv, start as u32, rot.threads,
                )
                .map_err(candle_core::Error::msg)?;
            // One arm per layout. The Planes14 arm has no bases to pass —
            // the variant carries none — and the Slot32 arm is the exact
            // call that shipped before the switch existed.
            match &self.proj.stream {
                DeviceStream::Slot32 { words, bases } => self
                    .rt
                    .cuda
                    .launch_slot_h(
                        &self.rt.f_matvec,
                        words,
                        bases,
                        &self.rt.tab,
                        &self.proj.gscale,
                        &self.proj.rscale,
                        &self.proj.tail,
                        xr,
                        &mut y,
                        self.proj.nblocks,
                        self.proj.tail_w,
                        self.proj.d_out as u32,
                        THREADS,
                        shared,
                    )
                    .map_err(candle_core::Error::msg)?,
                DeviceStream::Planes14 { words } => launch_planes_h(
                    &self.rt.cuda,
                    &self.rt.f_matvec,
                    words,
                    &self.rt.tab,
                    &self.proj.gscale,
                    &self.proj.rscale,
                    &self.proj.tail,
                    xr,
                    &mut y,
                    self.proj.nblocks,
                    self.proj.tail_w,
                    self.proj.d_out as u32,
                    THREADS,
                    shared,
                )
                .map_err(candle_core::Error::msg)?,
            }
        }

        Ok((
            CudaStorage::wrap_cuda_slice(y, self.rt.device.clone()),
            self.out_shape.clone(),
        ))
    }
}

fn upload_rotation(
    cuda: &llvq_cuda::gpu::Cuda,
    t: &RotationTables,
) -> candle_core::Result<RotBuffers> {
    Ok(RotBuffers {
        signbits: cuda.up_u32(&t.signbits).map_err(candle_core::Error::msg)?,
        small: cuda.up_f32(&t.small).map_err(candle_core::Error::msg)?,
        n: t.n as u32,
        m: t.m as u32,
        k: t.k as u32,
        inv: t.inv,
        // The widest block the driver allows, clamped to a warp at the bottom
        // so a narrow width still fills one.
        threads: (t.n as u32).next_power_of_two().clamp(32, 1024),
    })
}

/// The Planes14 twin of `Cuda::launch_slot_h` — `tv_slot_h`'s argument list
/// minus the bases array, which Planes14 does not have. Local to this crate
/// because `llvq-cuda` belongs to another lot; same grid (one warp per row,
/// whole blocks only, no bounds guard in the kernel), same generic `y` so
/// candle's `CudaSlice<half::f16>` can be handed over.
#[allow(clippy::too_many_arguments)]
fn launch_planes_h<T: candle_core::cuda_backend::cudarc::driver::DeviceRepr>(
    cuda: &llvq_cuda::gpu::Cuda,
    f: &CudaFunction,
    words: &CudaSlice<u32>,
    tab: &CudaSlice<u32>,
    gscale: &CudaSlice<f32>,
    rscale: &CudaSlice<f32>,
    tail: &CudaSlice<f32>,
    x: &CudaSlice<f32>,
    y: &mut CudaSlice<T>,
    nblocks: u32,
    tail_w: u32,
    d_out: u32,
    threads: u32,
    shared: u32,
) -> Result<(), String> {
    assert_eq!(d_out % (threads / 32), 0, "rows must fill whole blocks");
    let cfg = LaunchConfig {
        grid_dim: (d_out * 32 / threads, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: shared,
    };
    let mut b = cuda.stream().launch_builder(f);
    b.arg(words).arg(tab).arg(gscale).arg(rscale).arg(tail).arg(x).arg(y)
        .arg(&nblocks).arg(&tail_w);
    unsafe { b.launch(cfg) }.map_err(|e| format!("tv_planes_h: {e}"))?;
    Ok(())
}

fn upload_matrix(
    cuda: &llvq_cuda::gpu::Cuda,
    m: &FusedMatrix,
    layout: FusedLayout,
) -> candle_core::Result<FusedProj> {
    if !m.d_out.is_multiple_of(8) {
        // `tv_slot` has no bounds guard: a `return` before `__syncthreads()`
        // deadlocks, and it would break the full-warp mask the reduction
        // relies on. The grid is exact, so the host asserts instead.
        candle_core::bail!("{} : d_out={} n'est pas un multiple de 8", m.name, m.d_out);
    }
    // A stream in the wrong layout would be read by the wrong kernel into
    // finite, plausible, wrong numbers — refused here, matrix by matrix,
    // rather than trusted to have been built consistently.
    let stream = match (&m.stream, layout) {
        (HostStream::Slot32 { words, bases }, FusedLayout::Slot32) => DeviceStream::Slot32 {
            words: cuda.up_u32(words).map_err(candle_core::Error::msg)?,
            bases: cuda.up_u32(bases).map_err(candle_core::Error::msg)?,
        },
        (HostStream::Planes14 { words }, FusedLayout::Planes14) => DeviceStream::Planes14 {
            words: cuda.up_u32(words).map_err(candle_core::Error::msg)?,
        },
        _ => candle_core::bail!(
            "{} : flux hôte et layout du runtime ({}) en désaccord",
            m.name,
            layout.name()
        ),
    };
    Ok(FusedProj {
        name: m.name.clone(),
        d_out: m.d_out,
        d_in: m.d_in,
        nblocks: m.nblocks as u32,
        tail_w: m.tail_w as u32,
        stream,
        gscale: cuda.up_f32(&m.gscale).map_err(candle_core::Error::msg)?,
        rscale: cuda.up_f32(&m.rscale).map_err(candle_core::Error::msg)?,
        tail: cuda
            .up_f32(if m.tail.is_empty() { &[0.0f32] } else { &m.tail })
            .map_err(candle_core::Error::msg)?,
        rotation: m.rotation,
    })
}

/// A model rebuilt from a sealed artifact **with its projections still
/// encoded**, plus what it took to do so.
pub struct FusedSealed {
    pub model: crate::model::Qwen3,
    pub tokenizer: tokenizers::Tokenizer,
    pub config: candle_transformers::models::qwen3::Config,
    /// The runtime layout the projections were transcoded to.
    pub layout: FusedLayout,
    pub quantized_weights: usize,
    pub carried_weights: usize,
    /// Size of the file on disk.
    pub file_bytes: u64,
    /// Bytes the projections occupy on the device — the number that decides
    /// whether a model fits, and the one a disk figure must never stand in for.
    pub runtime_bytes: u64,
}

/// Load a sealed artifact straight onto the fused path.
///
/// The counterpart of [`crate::sealed::load`], and the two must produce the
/// same logits — `bin/fusedrun` is what checks that. What differs is what
/// sits in VRAM: 8.04 GB of f16 there, `runtime_bytes` plus the embedding
/// here.
pub fn load(path: &str, device: &Device, dtype: DType) -> candle_core::Result<FusedSealed> {
    use std::sync::Arc;

    // The layout is resolved once, before any transcoding, and printed next
    // to the device bytes it decides — an A/B where the arm has to be
    // inferred from a byte count is not an A/B.
    let layout = FusedLayout::from_env().map_err(candle_core::Error::msg)?;
    let model = crate::fused::load(path, layout).map_err(candle_core::Error::msg)?;
    println!(
        "layout fusé : {} (LLVQ_FUSED_LAYOUT) — projections {:.2} Go sur la carte, \
         {:.3} b/poids",
        layout.name(),
        model.runtime_bytes as f64 / 1e9,
        model.runtime_bits_per_weight()
    );
    let (rt, projs) = FusedRuntime::new(&model, device)?;
    let rt = Arc::new(rt);

    // Index the uploaded projections by the pair `Block::new_with` asks for.
    let mut by_site: HashMap<(usize, String), Arc<FusedProj>> = HashMap::new();
    for p in projs {
        let (layer, proj) =
            llvq_artifact::split_name(&p.name).map_err(|e| candle_core::Error::msg(e.to_string()))?;
        by_site.insert((layer, proj), Arc::new(p));
    }

    // Everything not quantized — embedding and norms — as ordinary tensors.
    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    for (name, dims, vals) in &model.raw {
        tensors.insert(
            name.clone(),
            Tensor::from_slice(vals, dims.clone(), device)?.to_dtype(dtype)?,
        );
    }

    let config: candle_transformers::models::qwen3::Config =
        serde_json::from_slice(&model.config_json)
            .map_err(|e| candle_core::Error::msg(format!("config.json : {e}")))?;
    let tokenizer = tokenizers::Tokenizer::from_bytes(&model.tokenizer_json)
        .map_err(|e| candle_core::Error::msg(format!("tokenizer.json : {e}")))?;

    let vb = candle_nn::VarBuilder::from_tensors(tensors, dtype, device);
    // Every site the artifact carries must be claimed; anything left over
    // means a name the loader and the model disagree about, and the model
    // would silently fall back to a `VarBuilder` lookup that cannot succeed.
    let mut claimed = 0usize;
    let qwen = crate::model::Qwen3::new_with(&config, vb, &mut |layer, name| {
        by_site.get(&(layer, name.to_string())).map(|p| {
            claimed += 1;
            crate::model::Proj::Fused {
                rt: rt.clone(),
                proj: p.clone(),
            }
        })
    })?;
    if claimed != by_site.len() {
        candle_core::bail!(
            "{claimed} projections réclamées par le modèle sur {} portées par le fichier",
            by_site.len()
        );
    }

    Ok(FusedSealed {
        model: qwen,
        tokenizer,
        config,
        layout,
        quantized_weights: model.quantized_weights,
        carried_weights: model.carried_weights,
        file_bytes: model.file_bytes,
        runtime_bytes: model.runtime_bytes,
    })
}
