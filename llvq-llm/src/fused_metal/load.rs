//! The loader: a runtime from the portable tables, and the one door that turns a resolved file into the
//! model's projections. One file because after it `model.rs` names no backend.

use super::*;

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
        check_threadgroup_memory(&dev)?;
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
                // `Tetra48Tables` packs the suffix bytes into `u32` for the
                // CUDA upload path; unpacked here because the composition is
                // indexed by byte.
                c3_of: dev.new_buffer_with_data(&compose_c3(
                    &t.branches,
                    &t.suffixes.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<u8>>(),
                ))?,
                // A served model keeps no host copy: it would pay for the
                // tables twice and its CPU arm is not a fallback.
                host: None,
            },
            kernel: kernel_from_env()?,
            q4_tile_cols: Q4_TILE_COLS,
            q4_prefetch: false,
            half_io: false,
            prefill_rows: 1,
            q4_rows: false,
            rot_tg: false,
            moe_down_sg: std::sync::atomic::AtomicBool::new(MOE_DOWN_SG_DEFAULT),
            device: dev,
            tile,
            pipelines: RwLock::new(HashMap::new()),
        })
    }

    /// Put this runtime on `tuning`: the Tetra matvec, the int4 tile and row
    /// loop. Refused by name before anything is set if the tuning is bad.
    /// Projections already uploaded keep the tile they were routed with.
    pub fn set_tuning(&mut self, tuning: MetalTuning) -> Result<()> {
        tuning.validate()?;
        self.kernel = tuning.tetra_kernel;
        self.q4_tile_cols = tuning.q4_tile_cols;
        self.q4_prefetch = tuning.q4_prefetch;
        self.half_io = tuning.half_io;
        self.prefill_rows = tuning.prefill_rows;
        self.q4_rows = tuning.q4_rows;
        self.rot_tg = tuning.rot_tg;
        Ok(())
    }

    /// The entry points this runtime has compiled, sorted: which kernels a
    /// run actually launched. The twins return the served bits by design, so
    /// no output can say which one ran; this can.
    pub fn compiled_pipelines(&self) -> Vec<&'static str> {
        let mut v: Vec<&'static str> =
            self.pipelines.read().expect("no panic held this lock").keys().map(|k| k.0).collect();
        v.sort_unstable();
        v.dedup();
        v
    }

    /// What this runtime launches, for a provenance line.
    pub fn tuning(&self) -> MetalTuning {
        MetalTuning {
            q4_tile_cols: self.q4_tile_cols,
            tetra_kernel: self.kernel,
            q4_prefetch: self.q4_prefetch,
            half_io: self.half_io,
            prefill_rows: self.prefill_rows,
            q4_rows: self.q4_rows,
            rot_tg: self.rot_tg,
        }
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
    load_resolved_tuned(path, device, dtype, layout, emode, share, fuse, kv, origin, None)
}

/// [`load_resolved`] with the runtime put on `tuning` before the first
/// upload. `None` is the served door, byte for byte: the Tetra matvec named
/// by `LLVQ_METAL_KERNEL` or the served one, the served int4 kernels, and
/// [`Q4_TILE_COLS`] for the records past the wall. A measurement harness
/// passes `Some` and names its arms itself.
#[allow(clippy::too_many_arguments)]
pub fn load_resolved_tuned(
    path: &str,
    device: &candle_core::Device,
    dtype: DType,
    layout: crate::fused::FusedLayout,
    emode: crate::fused::EmbedMode,
    share: crate::rotplan::RotShare,
    fuse: crate::fused::FuseMode,
    kv: crate::kvq::KvMode,
    origin: Option<&str>,
    tuning: Option<MetalTuning>,
) -> Result<crate::fused::FusedSealed> {
    let from = |var: &str| origin.unwrap_or(var).to_string();
    if let Some(t) = &tuning {
        t.validate()?;
    }
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
    crate::fused_cpu::check_token_ids(&tokenizer, &model.raw)
        .map_err(|e| candle_core::Error::msg(format!("{path}: {e}")))?;

    let embed_tables = match emode.bits() {
        None => None,
        Some(bits) => Some(
            crate::fused::take_embed_tables_at(&mut model.raw, config.tie_word_embeddings, bits)
                .map_err(|e| candle_core::Error::msg(format!("{path}: {e}")))?,
        ),
    };

    // The tile: no measured row for Apple, so the shader's own default.
    // `tuile-l40s-2026-09-20` measured 64 on sm_89 and nothing here.
    let tetra = llvq_search::tetra::Tetra::new();
    let tables = crate::fused::tetra48_tables(&tetra);
    let mut rt = MetalRuntime::from_tables(device, &tables, 64)?;
    if let Some(t) = tuning {
        rt.set_tuning(t)?;
    }
    let rt = Arc::new(rt);

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

    let (mut int4_records, mut int4_tiled) = (0usize, 0usize);
    for q in &model.int4 {
        let gpr = q.d_in / q.group;
        let wq: Vec<u32> = q
            .packed
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let proj = rt.upload_int4(&q.name, q.d_out, q.d_in, gpr, &wq, &q.scales, &q.biases)?;
        int4_records += 1;
        int4_tiled += usize::from(proj.tile_cols().is_some());
        let (layer, name) = llvq_artifact::split_name(&q.name)
            .map_err(|e| candle_core::Error::msg(e.to_string()))?;
        by_site.insert(
            (layer, name),
            crate::model::Proj::Int4(Arc::new(MetalInt4 { rt: rt.clone(), proj })),
        );
    }
    let t = rt.tuning();
    println!(
        "metal: tetra matvec {} | int4 {} records, {int4_tiled} through tv_q4_metal_tiled at {} columns a tile, {} \
         untiled | int4 row loop {} | io {} | prefill {} | rotation {}",
        t.tetra_kernel,
        int4_records,
        t.q4_tile_cols,
        int4_records - int4_tiled,
        if t.q4_prefetch { "prefetching (tv_q4_pf.metal)" } else { "served" },
        if t.half_io { "half (in-kernel casts, tv_q4_pf.metal and llvq_tetra48_rows.metal)" } else { "f32 (served)" },
        match t.prefill_rows {
            1 => "row by row (served)".to_string(),
            n => format!(
                "{n} rows a launch, Tetra (tv_tetra48_metal_lutdr){}",
                if t.q4_rows { " and int4 (tv_q4_metal_rows)" } else { ", int4 row by row" }
            ),
        },
        if t.rot_tg { "threadgroup up to 8,192 (rot_apply_tg_metal)" } else { "device (served)" },
    );
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
            let report = crate::fused::EmbedReport::new(emode, &to_upload);
            println!("{}", report.line(&from("LLVQ_EMBED")));
            let mut uploaded: Vec<Arc<MetalEmbedTable>> = Vec::with_capacity(to_upload.len());
            for t in &to_upload {
                let llvq_artifact::RawData::Quant(q) = &t.data else {
                    candle_core::bail!("{}: not a quantized tensor", t.name);
                };
                if Some(q.bits) != emode.bits() {
                    candle_core::bail!(
                        "{}: int{} where the {} kernels want int{:?}",
                        t.name, q.bits, emode.name(), emode.bits()
                    );
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
                    &t.name, q.bits, vocab, d, d.div_ceil(q.group), &wq, &q.scales, &q.biases,
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
        // `rows` is 1 on the served tuning, which carries no prefill kernel;
        // `tile` is what the shader was compiled at.
        prefill: crate::device::Prefill { rows: rt.prefill_rows(), tile: rt.tile, from_env: false },
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
