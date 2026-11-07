//! The served object on the CPU: the sealed file loaded without decoding it.
//!
//! The third backend beside `fused_cuda` and `fused_metal`, for a machine with
//! neither card. Before it, such a machine took `sealed::load`, which rebuilds
//! every projection as f16 and holds about 8 GB for the 4B. Here the Tetra
//! words, the int4 records and the int4 embedding stay as the file stores
//! them, and [`crate::tetra_cpu`] reads them at every matvec.
//!
//! Everything portable comes from [`crate::fused`], as for the two device
//! loaders. What is written here is the adapter: four small types that put
//! the slices of `tetra_cpu` behind the traits of [`crate::device`].
//!
//! The model around the projections (norms, attention, the KV cache) runs on
//! candle's CPU backend at the dtype the caller asks for. Serve it at f32:
//! candle's f16 ops on the CPU moved one 4,096-token window by -0.47 % of
//! perplexity against the dense reference, f32 by -0.002 %, and f32 decodes
//! faster there (`docs/mesures/cpu-tetra-2026-11-07.txt`). `prepare` returns
//! the activation rotated AND put in trio order, as f32: only this module's
//! `matvec` reads it.

use crate::tetra_cpu::{self, Isa, Q4Matrix, Tables, TetraMatrix};
use candle_core::{DType, Device, Result, Tensor};
use llvq_quant::rotation::Rotation;
use std::collections::HashMap;
use std::sync::Arc;

/// Prompt rows one batched call decodes the weights once for.
pub const PREFILL_ROWS: usize = tetra_cpu::MAX_BATCH;

/// `x` as one f32 vector, whatever its dtype and layout.
fn host_f32(x: &Tensor) -> Result<Vec<f32>> {
    x.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()
}

/// One Tetra projection on the CPU.
pub struct CpuLattice {
    name: String,
    isa: Isa,
    tables: Arc<Tables>,
    m: TetraMatrix,
    rot: Option<Arc<Rotation>>,
    key: Option<crate::fused::RotKey>,
    /// The dtype `matvec` hands back: the model's.
    dtype: DType,
}

impl CpuLattice {
    pub fn new(
        name: String,
        tables: Arc<Tables>,
        m: TetraMatrix,
        rot: Option<Arc<Rotation>>,
        key: Option<crate::fused::RotKey>,
        dtype: DType,
    ) -> std::result::Result<Self, String> {
        m.check().map_err(|e| format!("{name}: {e}"))?;
        if let Some(r) = &rot {
            if r.dim() != m.d_in {
                return Err(format!("{name}: rotation of {} for d_in {}", r.dim(), m.d_in));
            }
        }
        Ok(Self { name, isa: Isa::detect(), tables, m, rot, key, dtype })
    }

    /// The same projection on another SIMD path: what the tests sweep.
    pub fn with_isa(mut self, isa: Isa) -> Self {
        self.isa = isa;
        self
    }

    pub fn bytes(&self) -> u64 {
        self.m.bytes()
    }

    /// `rows` activations of `d_in`, rotated and put in trio order.
    pub fn prepare_vec(&self, x: &[f32], rows: usize) -> Vec<f32> {
        let d = self.m.d_in;
        assert_eq!(x.len(), rows * d, "{}: {} values for {rows} rows of {d}", self.name, x.len());
        let mut out = vec![0f32; rows * d];
        let mut buf = vec![0f64; d];
        let mut rotated = vec![0f32; d];
        for (src, dst) in x.chunks(d).zip(out.chunks_mut(d)) {
            match &self.rot {
                Some(r) => {
                    for (b, &v) in buf.iter_mut().zip(src) {
                        *b = f64::from(v);
                    }
                    r.apply(&mut buf);
                    for (o, &v) in rotated.iter_mut().zip(&buf) {
                        *o = v as f32;
                    }
                    tetra_cpu::permute_blocks(&self.tables, &rotated, dst);
                }
                None => tetra_cpu::permute_blocks(&self.tables, src, dst),
            }
        }
        out
    }

    /// `y = W x` for `rows` prepared activations, `[rows, d_out]`.
    pub fn matvec_vec(&self, xp: &[f32], rows: usize) -> Vec<f32> {
        let mut y = vec![0f32; rows * self.m.d_out];
        match rows {
            1 => tetra_cpu::tetra_matvec(self.isa, &self.tables, &self.m, xp, &mut y),
            _ => tetra_cpu::tetra_matvec_rows(self.isa, &self.tables, &self.m, xp, rows, &mut y),
        }
        y
    }
}

impl crate::device::LatticeProj for CpuLattice {
    fn name(&self) -> &str {
        &self.name
    }
    fn d_out(&self) -> usize {
        self.m.d_out
    }
    fn d_in(&self) -> usize {
        self.m.d_in
    }
    fn rotation(&self) -> Option<crate::fused::RotKey> {
        self.key
    }
    fn prepare(&self, x: &Tensor) -> Result<Tensor> {
        let v = host_f32(x)?;
        Tensor::from_vec(self.prepare_vec(&v, 1), (1, self.m.d_in), &Device::Cpu)
    }
    fn prepare_rows(&self, xs: &Tensor, rows: usize) -> Result<Tensor> {
        let v = host_f32(xs)?;
        Tensor::from_vec(self.prepare_vec(&v, rows), (rows, self.m.d_in), &Device::Cpu)
    }
    fn matvec(&self, xr: &Tensor, out_dims: &[usize]) -> Result<Tensor> {
        let v = host_f32(xr)?;
        if v.len() != self.m.d_in {
            candle_core::bail!("{}: one row of {} for d_in {}", self.name, v.len(), self.m.d_in);
        }
        let mut dims = out_dims.to_vec();
        *dims.last_mut().expect("rank >= 1") = self.m.d_out;
        Tensor::from_vec(self.matvec_vec(&v, 1), dims, &Device::Cpu)?.to_dtype(self.dtype)
    }
    fn rows_per_launch(&self) -> usize {
        PREFILL_ROWS
    }
    fn matvec_rows(&self, xr: &Tensor, n_rows: usize) -> Result<Tensor> {
        let v = host_f32(xr)?;
        if v.len() != n_rows * self.m.d_in {
            candle_core::bail!("{}: {} values for {n_rows} rows of {}", self.name, v.len(), self.m.d_in);
        }
        Tensor::from_vec(self.matvec_vec(&v, n_rows), (n_rows, self.m.d_out), &Device::Cpu)?.to_dtype(self.dtype)
    }
}

/// One int4 g128 projection on the CPU, in the natural basis.
pub struct CpuInt4 {
    name: String,
    isa: Isa,
    m: Q4Matrix,
    dtype: DType,
}

impl CpuInt4 {
    pub fn new(name: String, m: Q4Matrix, dtype: DType) -> std::result::Result<Self, String> {
        m.check().map_err(|e| format!("{name}: {e}"))?;
        Ok(Self { name, isa: Isa::detect(), m, dtype })
    }

    pub fn with_isa(mut self, isa: Isa) -> Self {
        self.isa = isa;
        self
    }

    pub fn bytes(&self) -> u64 {
        self.m.bytes()
    }

    pub fn matvec_vec(&self, x: &[f32], rows: usize) -> Vec<f32> {
        let mut y = vec![0f32; rows * self.m.d_out];
        match rows {
            1 => tetra_cpu::q4_matvec(self.isa, &self.m, x, &mut y),
            _ => tetra_cpu::q4_matvec_rows(self.isa, &self.m, x, rows, &mut y),
        }
        y
    }
}

impl crate::device::Int4Proj for CpuInt4 {
    fn name(&self) -> &str {
        &self.name
    }
    fn d_out(&self) -> usize {
        self.m.d_out
    }
    fn d_in(&self) -> usize {
        self.m.d_in
    }
    fn matvec(&self, x: &Tensor, out_dims: &[usize]) -> Result<Tensor> {
        let v = host_f32(x)?;
        if v.len() != self.m.d_in {
            candle_core::bail!("{}: one row of {} for d_in {}", self.name, v.len(), self.m.d_in);
        }
        let mut dims = out_dims.to_vec();
        *dims.last_mut().expect("rank >= 1") = self.m.d_out;
        Tensor::from_vec(self.matvec_vec(&v, 1), dims, &Device::Cpu)?.to_dtype(self.dtype)
    }
    fn rows_per_launch(&self) -> usize {
        PREFILL_ROWS
    }
    fn matvec_rows(&self, x: &Tensor, n_rows: usize) -> Result<Tensor> {
        let v = host_f32(x)?;
        if v.len() != n_rows * self.m.d_in {
            candle_core::bail!("{}: {} values for {n_rows} rows of {}", self.name, v.len(), self.m.d_in);
        }
        Tensor::from_vec(self.matvec_vec(&v, n_rows), (n_rows, self.m.d_out), &Device::Cpu)?.to_dtype(self.dtype)
    }
}

/// The int4 embedding table: the gather at the input, the `lm_head` at the
/// output. A tied model holds one and hands it to both ends.
pub struct CpuEmbed {
    name: String,
    isa: Isa,
    m: Arc<Q4Matrix>,
    dtype: DType,
}

impl CpuEmbed {
    pub fn new(name: String, m: Arc<Q4Matrix>, dtype: DType) -> std::result::Result<Self, String> {
        m.check().map_err(|e| format!("{name}: {e}"))?;
        Ok(Self { name, isa: Isa::detect(), m, dtype })
    }

    pub fn with_isa(mut self, isa: Isa) -> Self {
        self.isa = isa;
        self
    }

    pub fn table(&self) -> &Arc<Q4Matrix> {
        &self.m
    }
}

impl crate::device::QuantEmbedTable for CpuEmbed {
    fn gather(&self, ids: &Tensor) -> Result<Tensor> {
        let d = self.m.d_in;
        let flat = ids.to_dtype(DType::U32)?.flatten_all()?.to_vec1::<u32>()?;
        let mut out = vec![0f32; flat.len() * d];
        for (&id, o) in flat.iter().zip(out.chunks_mut(d)) {
            if id as usize >= self.m.d_out {
                candle_core::bail!("{}: token {id} past a vocabulary of {}", self.name, self.m.d_out);
            }
            self.m.decode_row(id as usize, o);
        }
        let mut dims = ids.dims().to_vec();
        dims.push(d);
        Tensor::from_vec(out, dims, &Device::Cpu)?.to_dtype(self.dtype)
    }

    fn project(&self, h: &Tensor) -> Result<Tensor> {
        let dims = h.dims().to_vec();
        let d = *dims.last().expect("rank >= 1");
        if d != self.m.d_in {
            candle_core::bail!("{}: hidden of {d} for d={}", self.name, self.m.d_in);
        }
        let v = host_f32(h)?;
        let rows = v.len() / d;
        let mut y = vec![0f32; rows * self.m.d_out];
        match rows {
            1 => tetra_cpu::q4_matvec(self.isa, &self.m, &v, &mut y),
            _ => tetra_cpu::q4_matvec_rows(self.isa, &self.m, &v, rows, &mut y),
        }
        let mut out = dims;
        *out.last_mut().expect("rank >= 1") = self.m.d_out;
        Tensor::from_vec(y, out, &Device::Cpu)?.to_dtype(self.dtype)
    }
}

/// An int4 table as the embedding path reads it.
fn q4_of(t: &llvq_artifact::RawTensor) -> Result<Q4Matrix> {
    let llvq_artifact::RawData::Quant(q) = &t.data else {
        candle_core::bail!("{}: not a quantized tensor", t.name);
    };
    if q.bits != 4 || t.dims.len() != 2 {
        candle_core::bail!("{}: int{} of dims {:?}, the CPU path reads a 2-D int4 table", t.name, q.bits, t.dims);
    }
    let m = Q4Matrix {
        d_out: t.dims[0],
        d_in: t.dims[1],
        group: q.group,
        packed: q.packed.clone(),
        scales: q.scales.clone(),
        biases: q.biases.clone(),
    };
    m.check().map_err(|e| candle_core::Error::msg(format!("{}: {e}", t.name)))?;
    Ok(m)
}

/// The served object on the CPU, loaded from a sealed artifact.
///
/// The argument list is the device loaders'. The CPU path reads `tetra48`
/// with the file's int4 order, and no fused group; anything else is refused
/// by name. The embedding is read as int4 under `q4` and widened to a dense
/// table under `f16`; `q8` has no CPU kernel and is refused.
#[allow(clippy::too_many_arguments)]
pub fn load_resolved(
    path: &str,
    dtype: DType,
    layout: crate::fused::FusedLayout,
    emode: crate::fused::EmbedMode,
    share: crate::rotplan::RotShare,
    fuse: crate::fused::FuseMode,
    kv: crate::kvq::KvMode,
    int4: crate::fused::Int4Layout,
    origin: Option<&str>,
) -> Result<crate::fused::FusedSealed> {
    let from = |var: &str| origin.unwrap_or(var).to_string();
    // A bad `LLVQ_THREADS` is refused here, before the pool's first use.
    tetra_cpu::threads().map_err(candle_core::Error::msg)?;
    if !matches!(dtype, DType::F16 | DType::F32) {
        candle_core::bail!("the CPU path runs at f16 or f32, asked for {dtype:?}");
    }
    if layout != crate::fused::FusedLayout::Tetra48 {
        candle_core::bail!("{}: the CPU path reads tetra48 only, asked for {}", from("LLVQ_FUSED_LAYOUT"), layout.name());
    }
    if fuse == crate::fused::FuseMode::On {
        candle_core::bail!("{}: the CPU path has no fused group", from("LLVQ_FUSE"));
    }
    if int4 != crate::fused::Int4Layout::Served {
        candle_core::bail!("int4 layout {} on the CPU: the tiled order is 4E v9's, a CUDA kernel", int4.name());
    }
    if emode == crate::fused::EmbedMode::Q8 {
        candle_core::bail!("{}: q8 has no CPU kernel; the sealed files carry q4", from("LLVQ_EMBED"));
    }
    crate::fused::check_fuse(layout, share, fuse).map_err(candle_core::Error::msg)?;

    let mut model = crate::fused::load_with(path, layout, fuse).map_err(candle_core::Error::msg)?;
    let rot_launches = crate::rotplan::rot_launches(share, &model.matrices, &model.groups);
    let matvec_launches = model.matrices.len() + model.int4.len() + model.groups.len();
    let (quantized_weights, carried_weights) = (model.quantized_weights, model.carried_weights);
    let file_bytes = model.file_bytes;

    let config: candle_transformers::models::qwen3::Config = serde_json::from_slice(&model.config_json)
        .map_err(|e| candle_core::Error::msg(format!("{path}: config.json: {e}")))?;
    let tokenizer = tokenizers::Tokenizer::from_bytes(&model.tokenizer_json)
        .map_err(|e| candle_core::Error::msg(format!("{path}: tokenizer.json: {e}")))?;

    let tables = Arc::new(Tables::default());
    let mut rotations: HashMap<crate::fused::RotKey, Arc<Rotation>> = HashMap::new();
    for &(n, seed) in model.rotations.keys() {
        rotations.insert((n, seed), Arc::new(Rotation::new(n, seed)));
    }
    model.rotations.clear();

    let mut by_site: HashMap<(usize, String), crate::model::Proj> = HashMap::new();
    let mut runtime_bytes = 0u64;
    for m in std::mem::take(&mut model.matrices) {
        let crate::fused::HostStream::Tetra48 { words, stride_u32 } = m.stream else {
            candle_core::bail!("{}: not a Tetra48 stream on a Tetra48 load", m.name);
        };
        let rot = match m.rotation {
            None => None,
            Some(k) => Some(rotations.get(&k).ok_or_else(|| candle_core::Error::msg(format!("rotation {k:?} missing")))?.clone()),
        };
        let tm = TetraMatrix {
            d_out: m.d_out,
            d_in: m.d_in,
            nblocks: m.nblocks,
            tail_w: m.tail_w,
            stride: stride_u32 as usize,
            words,
            gscale: m.gscale,
            rscale: m.rscale,
            tail: m.tail,
        };
        let (layer, site) = llvq_artifact::split_name(&m.name).map_err(|e| candle_core::Error::msg(e.to_string()))?;
        let p = CpuLattice::new(m.name, tables.clone(), tm, rot, m.rotation, dtype).map_err(candle_core::Error::msg)?;
        runtime_bytes += p.bytes();
        by_site.insert((layer, site), crate::model::Proj::Lattice(Arc::new(p)));
    }
    for q in std::mem::take(&mut model.int4) {
        let (layer, site) = llvq_artifact::split_name(&q.name).map_err(|e| candle_core::Error::msg(e.to_string()))?;
        let m = Q4Matrix { d_out: q.d_out, d_in: q.d_in, group: q.group, packed: q.packed, scales: q.scales, biases: q.biases };
        let p = CpuInt4::new(q.name, m, dtype).map_err(candle_core::Error::msg)?;
        runtime_bytes += p.bytes();
        by_site.insert((layer, site), crate::model::Proj::Int4(Arc::new(p)));
    }

    let embed_tables = match emode.bits() {
        None => None,
        Some(bits) => Some(
            crate::fused::take_embed_tables_at(&mut model.raw, config.tie_word_embeddings, bits)
                .map_err(|e| candle_core::Error::msg(format!("{path}: {e}")))?,
        ),
    };

    let device = Device::Cpu;
    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    let mut carried_bytes = 0u64;
    for t in std::mem::take(&mut model.raw) {
        carried_bytes += t.len() as u64 * dtype.size_in_bytes() as u64;
        tensors.insert(t.name.clone(), Tensor::from_vec(t.to_f32(), t.dims.clone(), &device)?.to_dtype(dtype)?);
    }
    let quant_embed = match &embed_tables {
        None => None,
        Some(tables) => {
            let report = crate::fused::EmbedReport::new(emode, &tables.buffers());
            println!("{}", report.line(&from("LLVQ_EMBED")));
            let embed = Arc::new(q4_of(&tables.embed)?);
            let head = match &tables.head {
                None => embed.clone(),
                Some(h) => Arc::new(q4_of(h)?),
            };
            carried_bytes += embed.bytes();
            if !Arc::ptr_eq(&embed, &head) {
                carried_bytes += head.bytes();
            }
            // The tie, on the tables: one table exactly when the config ties.
            if Arc::ptr_eq(&embed, &head) != config.tie_word_embeddings {
                candle_core::bail!(
                    "inconsistent q4 wiring: embedding and lm_head {} the same table while tie_word_embeddings = {}",
                    if Arc::ptr_eq(&embed, &head) { "share" } else { "do not share" },
                    config.tie_word_embeddings
                );
            }
            Some((
                CpuEmbed::new(crate::fused::EMBED_NAME.into(), embed, dtype).map_err(candle_core::Error::msg)?,
                CpuEmbed::new(crate::fused::HEAD_NAME.into(), head, dtype).map_err(candle_core::Error::msg)?,
            ))
        }
    };
    println!(
        "cpu: {} SIMD on {} threads | tetra {} + int4 {} matrices | {:.2} GB in memory (projections {:.2} + carried {:.2})",
        Isa::detect().name(),
        tetra_cpu::pool().current_num_threads(),
        by_site.values().filter(|p| matches!(p, crate::model::Proj::Lattice(_))).count(),
        by_site.values().filter(|p| matches!(p, crate::model::Proj::Int4(_))).count(),
        (runtime_bytes + carried_bytes) as f64 / 1e9,
        runtime_bytes as f64 / 1e9,
        carried_bytes as f64 / 1e9
    );

    let vb = candle_nn::VarBuilder::from_tensors(tensors, dtype, &device);
    let total_sites = by_site.len();
    let mut claimed = 0usize;
    let mut take = |layer: usize, name: &str| {
        by_site.remove(&(layer, name.to_string())).inspect(|_| {
            claimed += 1;
        })
    };
    let mut qwen = match quant_embed {
        None => crate::model::Qwen3::new_with(&config, vb, &mut take, kv)?,
        Some((e, h)) => crate::model::Qwen3::new_with_embed(
            &config,
            vb,
            &mut take,
            crate::model::Embed::Q8(Arc::new(e)),
            crate::model::Head::Q8(Arc::new(h)),
            kv,
        )?,
    };
    if claimed != total_sites {
        candle_core::bail!(
            "{path}: {claimed} of {total_sites} CPU projections were claimed by the model. \
             The rest would be served dense without saying so"
        );
    }
    qwen.set_rot_share(share);

    Ok(crate::fused::FusedSealed {
        model: qwen,
        tokenizer,
        config,
        layout,
        prefill: crate::device::Prefill { rows: PREFILL_ROWS, tile: 0, from_env: false },
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
