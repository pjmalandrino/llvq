//! The fused projections, running inside candle.
//!
//! This is where the kernel stops being a bench and starts being inference.
//! It holds the encoded streams on the device — `Planes14` by default,
//! `Planes12x` or `Slot32` under `LLVQ_FUSED_LAYOUT` — and replaces one
//! linear layer with two launches:
//!
//! ```text
//! x (f16) ──rot_apply──▶ x' (f32, rotated basis) ──tv_*_h──▶ y (f16)
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

use crate::fused::{
    load_planes_sources, matvec_kernel_name, EmbedMode, FusedLayout, FusedMatrix, FusedModel,
    HostStream, RotKey, RotationTables, EMBED_GROUP,
};

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

/// The q8 embedding kernels — appended only under `LLVQ_EMBED=q8`, so the
/// translation unit of both f16-embedding arms stays byte-identical to what
/// every published number compiled.
const EMB_Q8_CU_EMBED: &str = include_str!("../kernels/emb_q8.cu");

/// The q8 embedding kernel source, honouring `LLVQ_KERNEL_DIR` with the same
/// contract as [`crate::fused::load_planes_sources`].
fn load_emb_sources() -> Result<(String, Option<String>), String> {
    match std::env::var("LLVQ_KERNEL_DIR") {
        Err(_) => Ok((EMB_Q8_CU_EMBED.to_string(), None)),
        Ok(dir) => {
            let p = std::path::Path::new(&dir).join("emb_q8.cu");
            let s = std::fs::read_to_string(&p)
                .map_err(|e| format!("LLVQ_KERNEL_DIR={dir} : emb_q8.cu : {e}"))?;
            Ok((s, Some(dir)))
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
/// launching with one on the planes path is a compile error, not a bug class —
/// and, symmetrically, only the `Planes12x` arm can name an exception table,
/// so no other layout's launch can be handed one.
enum DeviceStream {
    Slot32 {
        words: CudaSlice<u32>,
        bases: CudaSlice<u32>,
    },
    Planes14 {
        words: CudaSlice<u32>,
    },
    Planes12x {
        words: CudaSlice<u32>,
        exc_idx: CudaSlice<u32>,
        exc_words: CudaSlice<u32>,
        row_exc: CudaSlice<u32>,
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
    /// The `KeepExact` tail as binary16 bits — the precision the dense arm
    /// holds these same columns at. See [`crate::fused::tail_f16_bits`], which
    /// owns the conversion, the argument and the accounting; `load` below
    /// refuses any dtype under which that argument would not hold.
    tail: CudaSlice<u16>,
    rotation: Option<RotKey>,
}

/// The module, the shared tables, and the device everything lives on.
pub struct FusedRuntime {
    cuda: llvq_cuda::gpu::Cuda,
    f_rot: CudaFunction,
    /// Whichever entry point [`matvec_kernel_name`] gave for the layout — one
    /// kernel per runtime, chosen with the layout, so a stream and a kernel
    /// of different layouts cannot meet.
    f_matvec: CudaFunction,
    tab: CudaSlice<u32>,
    rotations: HashMap<RotKey, RotBuffers>,
    /// The q8 embedding kernels, `(gather, lm_head matvec)` — present exactly
    /// when the runtime was built with [`EmbedMode::Q8`], which is when their
    /// source was in the translation unit.
    f_emb: Option<(CudaFunction, CudaFunction)>,
    /// Dynamic shared memory the card allows one block, read at startup —
    /// `tv_q8_h` stages the whole activation and must be refused past it.
    shared_limit: usize,
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
    ///
    /// `emode` decides whether the q8 embedding kernels join the translation
    /// unit; it must match how the caller intends to build the model, and is
    /// taken here rather than re-read from the environment so a runtime and
    /// its loader cannot resolve the variable twice differently.
    pub fn new(
        model: &FusedModel,
        device: &Device,
        emode: EmbedMode,
    ) -> candle_core::Result<(Self, Vec<FusedProj>)> {
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
            layout => Some(load_planes_sources(layout).map_err(candle_core::Error::msg)?),
        };
        let emb = match emode {
            EmbedMode::F16 => None,
            EmbedMode::Q8 => Some(load_emb_sources().map_err(candle_core::Error::msg)?),
        };
        let defines = format!("#define TILE_BLOCKS {TILE_BLOCKS}u\n");
        let mut parts: Vec<&str> = std::iter::once(defines.as_str())
            .chain(sources.parts.iter().map(String::as_str))
            .collect();
        if let Some((pp, overridden)) = &planes {
            parts.extend(pp.iter().map(String::as_str));
            if let Some(d) = overridden {
                eprintln!("⚠️ SOURCES {} SURCHARGÉES depuis {d}", model.layout.name());
            }
        }
        if let Some((es, overridden)) = &emb {
            parts.push(es.as_str());
            if let Some(d) = overridden {
                eprintln!("⚠️ SOURCE emb_q8 SURCHARGÉE depuis {d}");
            }
        }
        let src = llvq_cuda::gpu::KernelSource::new(&parts);
        let cuda = llvq_cuda::gpu::Cuda::on_stream(stream, &src).map_err(candle_core::Error::msg)?;

        // The register report is a contract, not a diagnostic: the block
        // decoders keep their accumulators and `rot_mix` a KMAX-wide column
        // in registers, and a spill costs occupancy without changing a
        // result. Checked on the kernel this runtime will actually launch.
        let matvec_name = matvec_kernel_name(model.layout);
        let mut spill_checked = vec![matvec_name, "rot_apply"];
        if emb.is_some() {
            spill_checked.extend(["emb_q8_gather", "tv_q8_h"]);
        }
        for name in spill_checked {
            let r = cuda.report(name).map_err(candle_core::Error::msg)?;
            if r.local_bytes != 0 {
                candle_core::bail!("{name} : {} octets de spill", r.local_bytes);
            }
        }
        let f_matvec = cuda.func(matvec_name).map_err(candle_core::Error::msg)?;
        let f_rot = cuda.func("rot_apply").map_err(candle_core::Error::msg)?;
        let f_emb = match emb {
            None => None,
            Some(_) => Some((
                cuda.func("emb_q8_gather").map_err(candle_core::Error::msg)?,
                cuda.func("tv_q8_h").map_err(candle_core::Error::msg)?,
            )),
        };

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
                f_emb,
                shared_limit,
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

    /// Upload an int8 g64 tensor — the embedding — for the two q8 kernels.
    ///
    /// Every assumption the kernels compile in is asserted here rather than
    /// trusted: 8 bits, group 64, `d % 4 == 0` (rows on word boundaries),
    /// `vocab % 8 == 0` (whole warps, no bounds guard), and the staged
    /// activation within the card's shared memory.
    pub fn upload_embed_q8(
        &self,
        t: &llvq_artifact::RawTensor,
    ) -> candle_core::Result<QuantEmbed> {
        if self.f_emb.is_none() {
            candle_core::bail!("runtime construit sans les noyaux q8 (LLVQ_EMBED=f16)");
        }
        let llvq_artifact::RawData::Quant(q) = &t.data else {
            candle_core::bail!("{} : pas un tenseur quantifié", t.name);
        };
        if q.bits != 8 || q.group != EMBED_GROUP {
            candle_core::bail!(
                "{} : int{} g{} — les noyaux codent int8 g{EMBED_GROUP} en dur",
                t.name, q.bits, q.group
            );
        }
        if t.dims.len() != 2 {
            candle_core::bail!("{} : dims {:?}, un embedding est 2-D", t.name, t.dims);
        }
        let (vocab, d) = (t.dims[0], t.dims[1]);
        if !d.is_multiple_of(4) {
            candle_core::bail!("{} : d={d} n'est pas un multiple de 4", t.name);
        }
        if !vocab.is_multiple_of(8) {
            candle_core::bail!("{} : vocab={vocab} n'est pas un multiple de 8", t.name);
        }
        if d * 4 > self.shared_limit {
            candle_core::bail!(
                "{} : {} o de partagée demandés par tv_q8_h, la carte en offre {}",
                t.name, d * 4, self.shared_limit
            );
        }
        let gpr = d.div_ceil(EMBED_GROUP);
        if q.packed.len() != vocab * d
            || q.scales.len() != vocab * gpr
            || q.biases.len() != q.scales.len()
        {
            candle_core::bail!(
                "{} : payload incohérent ({} octets, {} échelles, {} biais pour {vocab}×{d})",
                t.name, q.packed.len(), q.scales.len(), q.biases.len()
            );
        }
        let words: Vec<u32> = q
            .packed
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let bytes =
            q.packed.len() as u64 + (q.scales.len() as u64 + q.biases.len() as u64) * 2;
        Ok(QuantEmbed {
            vocab,
            d,
            gpr: gpr as u32,
            words: self.cuda.up_u32(&words).map_err(candle_core::Error::msg)?,
            scales: self.cuda.up_u16(&q.scales).map_err(candle_core::Error::msg)?,
            biases: self.cuda.up_u16(&q.biases).map_err(candle_core::Error::msg)?,
            bytes,
        })
    }

    /// Token ids `(.., l)` in u32 → embeddings `(.., l, d)` in f16, one
    /// gather launch for the whole call, rows dequantized on the device.
    pub fn embed(&self, q: &QuantEmbed, ids: &Tensor) -> candle_core::Result<Tensor> {
        let ids = ids.contiguous()?;
        let mut dims = ids.dims().to_vec();
        dims.push(q.d);
        let op = EmbedOp {
            rt: self,
            q,
            out_shape: Shape::from(dims),
        };
        ids.apply_op1_no_bwd(&op)
    }

    /// `logits = W_q8 · h` for hidden states `(.., d)` in f16 — the tied
    /// `lm_head`, one matvec launch per row, all rows into one buffer.
    pub fn lm_head(&self, q: &QuantEmbed, h: &Tensor) -> candle_core::Result<Tensor> {
        let dims = h.dims();
        let d_in = *dims.last().expect("rank >= 1");
        if d_in != q.d {
            candle_core::bail!("lm_head q8 attend d={}, reçu {d_in}", q.d);
        }
        let h = h.contiguous()?.to_dtype(DType::F16)?;
        let out_shape = {
            let mut d = dims.to_vec();
            *d.last_mut().expect("rank >= 1") = q.vocab;
            Shape::from(d)
        };
        let op = HeadOp {
            rt: self,
            q,
            out_shape,
        };
        h.apply_op1_no_bwd(&op)
    }
}

/// One int8 g64 embedding table, resident on the device.
///
/// When the model ties its two ends (Qwen3-4B) a single instance serves both
/// the gather at the input and the `lm_head` at the output — which is the
/// point: the −365 MB lot B validated exist only if no f16 copy is ever
/// materialized beside this. When they are untied (Qwen3-8B) there are two
/// instances, one per table, and `EmbedTables::wiring` says which is which.
pub struct QuantEmbed {
    pub vocab: usize,
    pub d: usize,
    gpr: u32,
    /// Packed int8 rows as `u32` words (`d % 4 == 0`, rows word-aligned).
    words: CudaSlice<u32>,
    scales: CudaSlice<u16>,
    biases: CudaSlice<u16>,
    /// Device bytes: payload + scales + biases.
    pub bytes: u64,
}

/// The gather: token ids in, f16 rows out.
struct EmbedOp<'a> {
    rt: &'a FusedRuntime,
    q: &'a QuantEmbed,
    out_shape: Shape,
}

impl candle_core::CustomOp1 for EmbedOp<'_> {
    fn name(&self) -> &'static str {
        "llvq-emb-q8-gather"
    }

    fn cpu_fwd(
        &self,
        _: &candle_core::CpuStorage,
        _: &Layout,
    ) -> candle_core::Result<(candle_core::CpuStorage, Shape)> {
        candle_core::bail!("le gather q8 n'a pas de chemin CPU")
    }

    fn cuda_fwd(
        &self,
        storage: &CudaStorage,
        layout: &Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        let ids = storage.as_cuda_slice::<u32>()?;
        let (start, end) = layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::msg("ids non contigus"))?;
        let ntok = end - start;
        if ntok == 0 {
            candle_core::bail!("gather q8 : zéro token");
        }
        let mut y = unsafe {
            self.rt
                .device
                .cuda_stream()
                .alloc::<f16>(ntok * self.q.d)
        }
        .map_err(|e| candle_core::Error::msg(format!("alloc emb: {e}")))?;
        let (f_gather, _) = self.rt.f_emb.as_ref().expect("checked at upload");
        let cfg = LaunchConfig {
            grid_dim: (ntok as u32, 1, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let (d, gpr, ids_off) = (self.q.d as u32, self.q.gpr, start as u32);
        let mut b = self.rt.cuda.stream().launch_builder(f_gather);
        b.arg(&self.q.words)
            .arg(&self.q.scales)
            .arg(&self.q.biases)
            .arg(ids)
            .arg(&mut y)
            .arg(&d)
            .arg(&gpr)
            .arg(&ids_off);
        unsafe { b.launch(cfg) }
            .map_err(|e| candle_core::Error::msg(format!("emb_q8_gather: {e}")))?;
        Ok((
            CudaStorage::wrap_cuda_slice(y, self.rt.device.clone()),
            self.out_shape.clone(),
        ))
    }
}

/// The tied `lm_head`: f16 hidden states in, f16 logits out.
struct HeadOp<'a> {
    rt: &'a FusedRuntime,
    q: &'a QuantEmbed,
    out_shape: Shape,
}

impl candle_core::CustomOp1 for HeadOp<'_> {
    fn name(&self) -> &'static str {
        "llvq-lmhead-q8"
    }

    fn cpu_fwd(
        &self,
        _: &candle_core::CpuStorage,
        _: &Layout,
    ) -> candle_core::Result<(candle_core::CpuStorage, Shape)> {
        candle_core::bail!("le lm_head q8 n'a pas de chemin CPU")
    }

    fn cuda_fwd(
        &self,
        storage: &CudaStorage,
        layout: &Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        let x = storage.as_cuda_slice::<f16>()?;
        let (start, end) = layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::msg("activation non contiguë"))?;
        let len = end - start;
        if len == 0 || !len.is_multiple_of(self.q.d) {
            candle_core::bail!("lm_head q8 : {len} valeurs pour d={}", self.q.d);
        }
        let rows = len / self.q.d;
        let mut y = unsafe {
            self.rt
                .device
                .cuda_stream()
                .alloc::<f16>(rows * self.q.vocab)
        }
        .map_err(|e| candle_core::Error::msg(format!("alloc logits: {e}")))?;
        let (_, f_head) = self.rt.f_emb.as_ref().expect("checked at upload");
        let cfg = LaunchConfig {
            grid_dim: (self.q.vocab as u32 * 32 / THREADS, 1, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: (self.q.d * 4) as u32,
        };
        let (d, gpr) = (self.q.d as u32, self.q.gpr);
        for r in 0..rows {
            let x_off = (start + r * self.q.d) as u32;
            let y_off = (r * self.q.vocab) as u32;
            let mut b = self.rt.cuda.stream().launch_builder(f_head);
            b.arg(&self.q.words)
                .arg(&self.q.scales)
                .arg(&self.q.biases)
                .arg(x)
                .arg(&mut y)
                .arg(&d)
                .arg(&gpr)
                .arg(&x_off)
                .arg(&y_off);
            unsafe { b.launch(cfg) }
                .map_err(|e| candle_core::Error::msg(format!("tv_q8_h: {e}")))?;
        }
        Ok((
            CudaStorage::wrap_cuda_slice(y, self.rt.device.clone()),
            self.out_shape.clone(),
        ))
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
                DeviceStream::Planes12x {
                    words,
                    exc_idx,
                    exc_words,
                    row_exc,
                } => launch_planes12x_h(
                    &self.rt.cuda,
                    &self.rt.f_matvec,
                    &[words, exc_idx, exc_words, row_exc],
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
    tail: &CudaSlice<u16>,
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

/// The Planes12x twin of [`launch_planes_h`] — same grid, four stream arrays
/// instead of one.
///
/// 🚨 **Read the grid, because it is the whole design decision.** It is
/// `tv_planes_h`'s grid, unchanged: `d_out·32/threads` CTAs, one warp per
/// output row, and *no* exception region. planesbench's `tv_planes12x` adds
/// `ceil(n_exc/8)` CTAs that `atomicAdd` into a `y` it memsets first; this
/// path instead hands each row its own slice of the exception table
/// (`row_exc`) so the corrections happen inside the row's warp. Consequences,
/// in the order they matter here:
///
///  * **`y` is not zeroed, and must not be.** `FusedOp::cuda_fwd` allocates
///    it uninitialised. That is sound for exactly the reason it is sound for
///    `tv_planes_h` — the grid is exact and every row is *stored*, never
///    accumulated into — and it stays sound here only because no CTA outside
///    a row's own warp writes that row. If anyone ever reintroduces an
///    exception region, the allocation upstream has to become a memset in the
///    same commit.
///  * **no atomic, so no `atomicAdd` on `__half`.** The accumulation is f32
///    from the first block to the last correction; the single narrowing to
///    binary16 is the final store, where candle would have narrowed anyway.
///    An `atomicAdd(__half*)` would have rounded every partial sum instead,
///    which is not the arithmetic the `Planes14` arm this replaces performs.
///
/// `arrays` is `[words, exc_idx, exc_words, row_exc]` in kernel order, taken
/// as one slice so a caller cannot silently transpose two `CudaSlice<u32>`
/// arguments of identical type — the one mistake here that compiles.
#[allow(clippy::too_many_arguments)]
fn launch_planes12x_h<T: candle_core::cuda_backend::cudarc::driver::DeviceRepr>(
    cuda: &llvq_cuda::gpu::Cuda,
    f: &CudaFunction,
    arrays: &[&CudaSlice<u32>; 4],
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
    assert_eq!(d_out % (threads / 32), 0, "rows must fill whole blocks");
    let cfg = LaunchConfig {
        grid_dim: (d_out * 32 / threads, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: shared,
    };
    let mut b = cuda.stream().launch_builder(f);
    b.arg(arrays[0]).arg(arrays[1]).arg(arrays[2]).arg(arrays[3])
        .arg(tab).arg(gscale).arg(rscale).arg(tail).arg(x).arg(y)
        .arg(&nblocks).arg(&tail_w);
    unsafe { b.launch(cfg) }.map_err(|e| format!("tv_planes12x_h: {e}"))?;
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
        (
            HostStream::Planes12x { words, exc_idx, exc_words, row_exc },
            FusedLayout::Planes12x,
        ) => {
            // cudarc refuses a zero-length upload. A matrix with no 5-level
            // block still needs a bound pointer, so it gets a one-word dummy
            // the kernel never dereferences: every row's slice is empty, so
            // `planes12x_row_correction`'s loop never runs. `exc_words` is
            // never empty — `pack_plane_bytes` appends the read-window pad —
            // and `row_exc` has `d_out + 1` entries, so only `exc_idx` needs
            // this.
            let dummy = [0u32];
            let idx: &[u32] = if exc_idx.is_empty() { &dummy } else { exc_idx };
            DeviceStream::Planes12x {
                words: cuda.up_u32(words).map_err(candle_core::Error::msg)?,
                exc_idx: cuda.up_u32(idx).map_err(candle_core::Error::msg)?,
                exc_words: cuda.up_u32(exc_words).map_err(candle_core::Error::msg)?,
                row_exc: cuda.up_u32(row_exc).map_err(candle_core::Error::msg)?,
            }
        }
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
        // Binary16 bits, narrowed on the host by `fused::tail_f16_bits`.
        // cudarc refuses a zero-length upload, so a matrix whose `d_in` is a
        // multiple of 24 gets a one-element dummy the kernel never reads
        // (`tail_w == 0` makes `tail_dot_h`'s loop empty) — the same shape the
        // `Planes12x` exception index uses above.
        tail: cuda
            .up_u16(if m.tail.is_empty() { &[0u16] } else { &m.tail })
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
    /// How the embedding and tied `lm_head` sit on the device.
    pub embed_mode: EmbedMode,
    pub quantized_weights: usize,
    pub carried_weights: usize,
    /// Size of the file on disk.
    pub file_bytes: u64,
    /// Bytes the projections occupy on the device — the number that decides
    /// whether a model fits, and the one a disk figure must never stand in for.
    pub runtime_bytes: u64,
    /// Bytes the carried tensors occupy on the device: `2 · carried_weights`
    /// under `LLVQ_EMBED=f16`, the int8 payload of **every** embedding table
    /// plus the f16 norms under `q8` — one table when the model ties its two
    /// ends, two when it unties them. `carried_weights · 2` must no longer
    /// stand in for this: that identity is exactly what q8 breaks, by −365 MB
    /// on the tied 4B and −1.17 GB on the untied 8B.
    pub carried_bytes: u64,
}

/// Load a sealed artifact straight onto the fused path.
///
/// The counterpart of [`crate::sealed::load`], and the two must produce the
/// same logits — `bin/fusedrun` is what checks that. What differs is what
/// sits in VRAM: 8.04 GB of f16 there, `runtime_bytes` plus the embedding
/// here.
///
/// ## `dtype` must be F16, and this now says so out loud
///
/// It always was, in fact: `FusedRuntime::forward` casts its input with
/// `to_dtype(DType::F16)` and the kernels *store* halves, so a model built at
/// F32 would have taken f16 tensors out of every projection. What changed on
/// 2026-08-09 is that the assumption acquired a consequence — the `KeepExact`
/// tail is now resident as binary16, and the argument that this costs nothing
/// (see [`crate::fused::tail_f16_bits`]) is *entirely* the fact that the dense
/// arm narrows the same columns to the run's dtype. At F32 that argument is
/// false and the tail would be the one place the fused path is coarser than
/// its reference. Refusing beats carrying a silently weaker claim.
pub fn load(path: &str, device: &Device, dtype: DType) -> candle_core::Result<FusedSealed> {
    use std::sync::Arc;

    if dtype != DType::F16 {
        candle_core::bail!(
            "chemin fusé demandé en {dtype:?} : il est f16 de bout en bout (activations \
             converties, noyaux stockant des demis, queue KeepExact résidente en binaire16 \
             pour s'aligner sur ce que `sealed::load` narrow au même dtype). Un autre dtype \
             ne rendrait pas un modèle plus précis, il rendrait une comparaison fausse."
        );
    }

    // The layout and the embedding mode are resolved once, before any
    // transcoding, and printed next to the device bytes they decide — an A/B
    // where the arm has to be inferred from a byte count is not an A/B.
    let layout = FusedLayout::from_env().map_err(candle_core::Error::msg)?;
    let emode = EmbedMode::from_env().map_err(candle_core::Error::msg)?;
    let mut model = crate::fused::load(path, layout).map_err(candle_core::Error::msg)?;
    // The accounting is named on the line, because since lot A7a this number
    // is deliberately *not* the bench's: the tail is resident at binary16
    // here and at f32 in `planesbench`/`rtbits`, worth 0.075 b/weight on the
    // 4B. A reader comparing 4.729 to a published 4.804 must be able to see
    // from the log itself that they are two residencies, not a regression.
    println!(
        "layout fusé : {} (LLVQ_FUSED_LAYOUT) — projections {:.2} Go sur la carte, \
         {:.3} b/poids (comptabilité INFÉRENCE : queue KeepExact en binaire16 ; \
         le banc facture la sienne en f32)",
        layout.name(),
        model.runtime_bytes as f64 / 1e9,
        model.runtime_bits_per_weight()
    );

    let config: candle_transformers::models::qwen3::Config =
        serde_json::from_slice(&model.config_json)
            .map_err(|e| candle_core::Error::msg(format!("config.json : {e}")))?;
    let tokenizer = tokenizers::Tokenizer::from_bytes(&model.tokenizer_json)
        .map_err(|e| candle_core::Error::msg(format!("tokenizer.json : {e}")))?;

    // Under q8, the embedding tables leave the carried list before any tensor
    // is built: nothing downstream may materialize an f16 copy of them. One
    // table when the ends are tied (4B), two when they are not (8B) — the
    // second is `lm_head.weight`, a different weight that gets a different
    // buffer. Both go through the exact `bin/embedq` arithmetic (same
    // function), or carry the file's own q8 bytes through untouched.
    let embed_tables = match emode {
        EmbedMode::F16 => None,
        EmbedMode::Q8 => Some(
            crate::fused::take_embed_tables(&mut model.raw, config.tie_word_embeddings)
                .map_err(|e| candle_core::Error::msg(format!("{path} : {e}")))?,
        ),
    };

    let (rt, projs) = FusedRuntime::new(&model, device, emode)?;
    let rt = Arc::new(rt);

    // Index the uploaded projections by the pair `Block::new_with` asks for.
    let mut by_site: HashMap<(usize, String), Arc<FusedProj>> = HashMap::new();
    for p in projs {
        let (layer, proj) =
            llvq_artifact::split_name(&p.name).map_err(|e| candle_core::Error::msg(e.to_string()))?;
        by_site.insert((layer, proj), Arc::new(p));
    }

    // Everything still carried — norms, and the embedding in f16 mode — as
    // ordinary tensors. Whatever encoding the file used, the device holds f16
    // here, so each costs `2 · len` bytes on the card.
    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    let mut carried_bytes = 0u64;
    for t in &model.raw {
        carried_bytes += t.len() as u64 * 2;
        tensors.insert(
            t.name.clone(),
            Tensor::from_vec(t.to_f32(), t.dims.clone(), device)?.to_dtype(dtype)?,
        );
    }

    // The announced footprint, mode by mode, before the first launch. Both
    // modes count **every** embedding table the device holds: with the ends
    // untied there are two, and reporting one of them beside a correct total
    // is how a footprint line ends up contradicting the one below it.
    let quant_embed = match &embed_tables {
        None => {
            let carried = crate::fused::carried_embed_tables(&model.raw);
            println!(
                "{}",
                crate::fused::EmbedReport::new(EmbedMode::F16, &carried).line()
            );
            None
        }
        Some(tables) => {
            let to_upload = tables.buffers();
            let report = crate::fused::EmbedReport::new(EmbedMode::Q8, &to_upload);
            println!("{}", report.line());
            let mut uploaded: Vec<Arc<QuantEmbed>> = Vec::with_capacity(to_upload.len());
            for (t, (_, packed, sb)) in to_upload.iter().zip(&report.tables) {
                let q = rt.upload_embed_q8(t)?;
                // A hard check, not a `debug_assert!`: release is the only
                // profile that runs on a card, and the whole point of this lot
                // is that the printed line cannot contradict the total below
                // it. Comparing what was announced to what was uploaded costs
                // nothing and closes exactly the defect being fixed.
                if q.bytes != *packed + *sb {
                    candle_core::bail!(
                        "table portée {} : {} octets annoncés, {} uploadés",
                        t.name, *packed + *sb, q.bytes
                    );
                }
                carried_bytes += q.bytes;
                uploaded.push(Arc::new(q));
            }
            Some((uploaded, tables.wiring()))
        }
    };
    println!(
        "total attendu sur la carte : {:.2} Go (projections {:.2} + portés {:.2})",
        (model.runtime_bytes + carried_bytes) as f64 / 1e9,
        model.runtime_bytes as f64 / 1e9,
        carried_bytes as f64 / 1e9
    );

    let vb = candle_nn::VarBuilder::from_tensors(tensors, dtype, device);
    // Every site the artifact carries must be claimed; anything left over
    // means a name the loader and the model disagree about, and the model
    // would silently fall back to a `VarBuilder` lookup that cannot succeed.
    let mut claimed = 0usize;
    let mut take = |layer: usize, name: &str| {
        by_site.get(&(layer, name.to_string())).map(|p| {
            claimed += 1;
            crate::model::Proj::Fused {
                rt: rt.clone(),
                proj: p.clone(),
            }
        })
    };
    // `(ie, ih)` is `(0, 0)` when the ends are tied — two clones of one `Arc`,
    // exactly what shipped — and `(0, 1)` when they are not. The choice is
    // `EmbedTables::wiring`'s, tested on any machine; here it is only indexed.
    // The two indexing expressions below are the only lines of this path no
    // test on a developer machine can reach, and a one-character slip there
    // (`ie` twice) yields a model that runs and lies. This check is their
    // self-verification, on the card, at zero cost.
    if let Some((bufs, (ie, ih))) = &quant_embed {
        if Arc::ptr_eq(&bufs[*ie], &bufs[*ih]) != config.tie_word_embeddings {
            candle_core::bail!(
                "câblage q8 incohérent : embedding et lm_head {} le même buffer \
                 alors que tie_word_embeddings = {}",
                if Arc::ptr_eq(&bufs[*ie], &bufs[*ih]) { "partagent" } else { "ne partagent pas" },
                config.tie_word_embeddings
            );
        }
    }
    let qwen = match &quant_embed {
        None => crate::model::Qwen3::new_with(&config, vb, &mut take)?,
        Some((bufs, (ie, ih))) => crate::model::Qwen3::new_with_embed(
            &config,
            vb,
            &mut take,
            crate::model::Embed::Q8 {
                rt: rt.clone(),
                q: bufs[*ie].clone(),
            },
            crate::model::Head::Q8 {
                rt: rt.clone(),
                q: bufs[*ih].clone(),
            },
        )?,
    };
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
        embed_mode: emode,
        quantized_weights: model.quantized_weights,
        // Counted at read time in `fused::load`, embedding included — the q8
        // extraction changes where those weights sit, not how many there are.
        carried_weights: model.carried_weights,
        file_bytes: model.file_bytes,
        runtime_bytes: model.runtime_bytes,
        carried_bytes,
    })
}
