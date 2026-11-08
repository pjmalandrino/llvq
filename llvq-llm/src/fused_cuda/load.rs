//! Loading a sealed artifact onto the fused path.

use super::*;

/// Load a sealed artifact straight onto the fused path.
///
/// The counterpart of [`crate::sealed::load`], and the two must produce the
/// same logits — `bin/fusedrun` is what checks that. What differs is what
/// sits in VRAM: 8.04 GB of f16 there, `runtime_bytes` plus the embedding
/// here.
///
/// ## `dtype` must be F16, and this now says so out loud
///
/// It always was, in fact: [`FusedRuntime::rotate`] casts its input with
/// `to_dtype(DType::F16)` and the kernels *store* halves, so a model built at
/// F32 would have taken f16 tensors out of every projection. What changed on
/// 2026-08-09 is that the assumption acquired a consequence — the `KeepExact`
/// tail is now resident as binary16, and the argument that this costs nothing
/// (see [`crate::fused::tail_f16_bits`]) is *entirely* the fact that the dense
/// arm narrows the same columns to the run's dtype. At F32 that argument is
/// false and the tail would be the one place the fused path is coarser than
/// its reference. Refusing beats carrying a silently weaker claim.
/// [`load`] with the fusion mode named by the caller rather than read from the
/// environment — what lets `bin/fusedrun` run both arms in one process, each
/// dropped before the next loads, so the card holds one arm at a time and the
/// two share a card, a prompt and one NVRTC translation unit.
pub fn load_with(
    path: &str,
    device: &Device,
    dtype: DType,
    fuse: FuseMode,
) -> candle_core::Result<FusedSealed> {
    // The layout, the embedding mode and the hoist come from the environment
    // here — this is the measurement door, and every A/B in `docs/mesures/`
    // turns one of those variables between two processes.
    let layout = FusedLayout::from_env().map_err(candle_core::Error::msg)?;
    let emode = EmbedMode::from_env().map_err(candle_core::Error::msg)?;
    let share = crate::rotplan::RotShare::from_env().map_err(candle_core::Error::msg)?;
    // 🕳️ **F16 here by ALIGNMENT, not by default.** `bin/fusedrun` loads its
    // dense arm with `KvMode::F16` hardcoded (the dense arm of `bin/fusedrun`): its question
    // is the fused kernel, not the cache. The fused arm must take the same
    // one, or the comparison gains a second variable — which is what this
    // workstream spends its time forbidding. The served door takes the value
    // its config names, and answers a different question.
    load_resolved(path, device, dtype, layout, emode, share, fuse, crate::kvq::KvMode::F16, crate::fused::Int4Layout::Served, None)
}

/// [`load_with`] with the four choices already decided.
///
/// The served door. `crate::served::Served` reads them from a file and hands
/// them over, so nothing between that file and the transcoder consults the
/// environment — which is what makes the file the authority rather than a
/// suggestion that an unset variable could quietly outvote.
#[allow(clippy::too_many_arguments)]
pub fn load_resolved(
    path: &str,
    device: &Device,
    dtype: DType,
    layout: FusedLayout,
    emode: EmbedMode,
    share: crate::rotplan::RotShare,
    fuse: FuseMode,
    kv: crate::kvq::KvMode,
    // The order of the int4 words on the card: the file's, or 4E v9's tiled
    // one, alone, under a config whose rows are "v9" (`Served::int4_layout`).
    int4_layout: crate::fused::Int4Layout,
    // `Some("LLVQ_CONFIG")` from the served door, `None` from `load_with`. The
    // three choice lines below print it in place of the variable names, so a
    // served log does not attribute its choices to variables nobody set —
    // which is the ambiguity `crate::served` was written against. In bench
    // mode the variable names stay: they tell an A/B reader which one flipped.
    source: Option<&str>,
) -> candle_core::Result<FusedSealed> {
    use std::sync::Arc;
    let from = |var: &str| source.unwrap_or(var).to_string();

    if dtype != DType::F16 {
        candle_core::bail!(
            "fused path requested in {dtype:?}: it is f16 end to end (activations \
             converted, kernels storing halves, KeepExact tail resident in binary16 to \
             line up with what `sealed::load` narrows to the same dtype). Another dtype \
             would not make the model more accurate, it would make a comparison wrong."
        );
    }

    // cudarc creates two CUDA events with every allocation and waits on and
    // destroys them at every free, for ordering across streams (cudarc 0.19.8,
    // `CudaStream::alloc`, `Drop for CudaSlice`). The served path allocates
    // every op output fresh, about 1,450 a token on the sealed 4B, and runs
    // on ONE stream, candle's, where the driver orders work by issue: those
    // events order nothing and cost host time on every token. Turned off on
    // the served door only, before the first upload (cudarc skips the events
    // for slices created after the call), and only while the context holds a
    // single stream. The bench door keeps cudarc's default, so the measured
    // arms of `docs/mesures/` keep their protocol. Nothing on the device
    // changes, so no output can.
    if source.is_some() {
        let stream = device.as_cuda_device()?.cuda_stream();
        let ctx = stream.context();
        if ctx.is_in_multi_stream_mode() {
            println!("cudarc event tracking: kept on, the context holds several streams");
        } else {
            // Safe because one stream orders every access by issue; anything
            // that later creates a second stream on this context must revisit
            // it, as `llvq_cuda::gpu::Cuda::new_on_fresh_stream` says.
            unsafe { ctx.disable_event_tracking() };
            println!("cudarc event tracking: off (one stream, ordered by issue)");
        }
    }

    // The layout and the embedding mode arrive resolved and are printed next
    // to the device bytes they decide — an A/B where the arm has to be
    // inferred from a byte count is not an A/B.
    //
    // Both refusals before the 145 s transcode, not after it: a job that pays
    // for a load and then discovers its two variables are incompatible has
    // spent the money for nothing.
    crate::fused::check_fuse(layout, share, fuse).map_err(candle_core::Error::msg)?;
    let mut model =
        crate::fused::load_with(path, layout, fuse).map_err(candle_core::Error::msg)?;
    // The partition is checked inside `fused::load_with`; these are only its
    // counts, printed because `LLVQ_ROT_SHARE=1` reporting 252 launches — or
    // `LLVQ_FUSE=1` reporting 252 matvecs — would be a lot that did nothing
    // while looking green.
    let rot_launches = crate::rotplan::rot_launches(share, &model.matrices, &model.groups);
    let matvec_launches =
        crate::rotplan::matvec_launches_per_token(&model.matrices, &model.groups, model.int4.len());
    // Every projection the model holds, int4 included: the count sits beside a
    // launch count on the next two lines, and one that omitted 36 of 252 would
    // make the launches per token read as an inconsistency rather than as the
    // fact they are.
    let projections = model.matrices.len()
        + model.groups.iter().map(|g| g.parts.len()).sum::<usize>()
        + model.int4.len();
    println!(
        "shared rotation: {} ({}), {rot_launches} rot_launches/token \
         for {projections} projections",
        share.name(),
        from("LLVQ_ROT_SHARE")
    );
    println!(
        "projection fusion: {} ({}), {matvec_launches} matvec_launches/token \
         for {projections} projections ({} groups + {} lone + {} int4)",
        fuse.name(),
        from("LLVQ_FUSE"),
        model.groups.len(),
        model.matrices.len(),
        model.int4.len()
    );
    // The accounting is named on the line, because since lot A7a this number
    // is deliberately *not* the bench's: the tail is resident at binary16
    // here and at f32 in `planesbench`/`rtbits`, worth 0.075 b/weight on the
    // 4B. A reader comparing 4.729 to a published 4.804 must be able to see
    // from the log itself that they are two residencies, not a regression.
    println!(
        "fused layout: {} ({}), projections {:.2} GB on the card, \
         {:.3} b/weight (INFERENCE accounting: KeepExact tail in binary16; \
         the bench bills its own in f32)",
        layout.name(),
        from("LLVQ_FUSED_LAYOUT"),
        model.runtime_bytes as f64 / 1e9,
        model.runtime_bits_per_weight()
    );

    let config: candle_transformers::models::qwen3::Config =
        serde_json::from_slice(&model.config_json)
            .map_err(|e| candle_core::Error::msg(format!("config.json: {e}")))?;
    let tokenizer = tokenizers::Tokenizer::from_bytes(&model.tokenizer_json)
        .map_err(|e| candle_core::Error::msg(format!("tokenizer.json: {e}")))?;

    // Under q8, the embedding tables leave the carried list before any tensor
    // is built: nothing downstream may materialize an f16 copy of them. One
    // table when the ends are tied (4B), two when they are not (8B) — the
    // second is `lm_head.weight`, a different weight that gets a different
    // buffer. Both go through the exact `bin/embedq` arithmetic (same
    // function), or carry the file's own q8 bytes through untouched.
    let embed_tables = match emode.bits() {
        None => None,
        Some(bits) => Some(
            crate::fused::take_embed_tables_at(&mut model.raw, config.tie_word_embeddings, bits)
                .map_err(|e| candle_core::Error::msg(format!("{path}: {e}")))?,
        ),
    };

    let (rt, projs, seg_projs, int4_projs) = FusedRuntime::new(&model, device, emode, fuse, int4_layout)?;
    // Beside the card bytes the layout line printed: the tiled order is a
    // permutation inside each row, so the bytes, the file and the b/param do
    // not move; what moves is that 4E v9 needs no second copy.
    if !int4_projs.is_empty() {
        let int4_bytes: u64 = int4_projs.iter().map(|p| 4 * p.wq.len() as u64).sum();
        println!(
            "int4 layout: {} ({}), {} records, words {:.3} GB on the card, held once",
            int4_layout.name(),
            from("the bench door"),
            int4_projs.len(),
            int4_bytes as f64 / 1e9
        );
    }
    if INT4_WIDE_CHECK.load(std::sync::atomic::Ordering::Relaxed) {
        if source.is_some() {
            candle_core::bail!("LLVQ_INT4_WIDE_CHECK is a measurement mode: not on the served door");
        }
        rt.int4_wide_check(&int4_projs).map_err(candle_core::Error::msg)?;
    }
    let rt = Arc::new(rt);

    // Index every uploaded projection by the pair `Block::new_with` asks for —
    // from **both** sources. A lone projection yields one `Proj::Lattice`; a
    // group yields one `Proj::GroupPart` per part, all pointing at the *same*
    // `Arc<FusedSegProj>`, which is what `model::SegPlan::of` recognises with
    // `Arc::ptr_eq`. The `claimed != total_sites` check further down then also
    // catches a group one of whose parts the model never claimed.
    let mut by_site: HashMap<(usize, String), crate::model::Proj> = HashMap::new();
    for p in projs {
        let (layer, proj) =
            llvq_artifact::split_name(&p.name).map_err(|e| candle_core::Error::msg(e.to_string()))?;
        by_site.insert(
            (layer, proj),
            crate::model::Proj::Lattice(Arc::new(CudaLattice { rt: rt.clone(), proj: p })),
        );
    }
    // The int4 records, indexed by the same pair. They are never part of a
    // group: `fused::segment_matrices` reads `model.matrices`, which holds
    // lattice records only, so a `v_proj` served as int4 simply is not offered
    // to the fusion — and `model::group_forward` would refuse a mixed group
    // anyway, at `check_key`, because this projection's key is `None` and a
    // rotated group's is not.
    for q in int4_projs {
        let (layer, proj) = llvq_artifact::split_name(&q.name)
            .map_err(|e| candle_core::Error::msg(e.to_string()))?;
        by_site.insert(
            (layer, proj),
            crate::model::Proj::Int4(Arc::new(CudaInt4 { rt: rt.clone(), proj: q })),
        );
    }
    // The row order comes from `fused::segment_matrices`, which read it off
    // `Act::consumers()`; `model::SegPlan::of` re-derives it from these very
    // fields and refuses a group that does not tile. Two places, one table.
    for (g, group) in model.groups.iter().zip(seg_projs) {
        // OUTSIDE the loop, and the single sharpest line of the port. One
        // `Arc` per group, cloned into each part: `SegPlan::of` recognises a
        // group by `Arc::ptr_eq` on it. Built inside the loop instead, every
        // comparison goes false and the served path dies at the first token,
        // type-correct and card-only.
        let group: Arc<dyn crate::device::SegGroup> =
            Arc::new(CudaSeg { rt: rt.clone(), group });
        for part in &g.parts {
            by_site.insert(
                (part.layer, part.proj.clone()),
                crate::model::Proj::GroupPart {
                    group: group.clone(),
                    row0: part.row0,
                    d_out: part.d_out,
                    rank: part.rank,
                },
            );
        }
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
                crate::fused::EmbedReport::new(EmbedMode::F16, &carried).line(&from("LLVQ_EMBED"))
            );
            None
        }
        Some(tables) => {
            let to_upload = tables.buffers();
            let report = crate::fused::EmbedReport::new(emode, &to_upload);
            println!("{}", report.line(&from("LLVQ_EMBED")));
            let mut uploaded: Vec<Arc<QuantEmbed>> = Vec::with_capacity(to_upload.len());
            for (t, (_, packed, sb)) in to_upload.iter().zip(&report.tables) {
                let q = rt.upload_embed(t)?;
                // A hard check, not a `debug_assert!`: release is the only
                // profile that runs on a card, and the whole point of this lot
                // is that the printed line cannot contradict the total below
                // it. Comparing what was announced to what was uploaded costs
                // nothing and closes exactly the defect being fixed.
                if q.bytes != *packed + *sb {
                    candle_core::bail!(
                        "carried table {}: {} bytes announced, {} uploaded",
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
        "total expected on the card: {:.2} GB (projections {:.2} + carried {:.2})",
        (model.runtime_bytes + carried_bytes) as f64 / 1e9,
        model.runtime_bytes as f64 / 1e9,
        carried_bytes as f64 / 1e9
    );

    let vb = candle_nn::VarBuilder::from_tensors(tensors, dtype, device);
    // Every site the artifact carries must be claimed; anything left over
    // means a name the loader and the model disagree about, and the model
    // would silently fall back to a `VarBuilder` lookup that cannot succeed.
    //
    // `remove` rather than `get`: a `Proj` is not `Clone` (a `FusedSeg` shares
    // one `Arc` between its parts, and cloning the enum would be a second way
    // to say that), so the map hands each site over exactly once. `total_sites`
    // is therefore read **before** the model claims any of them.
    let total_sites = by_site.len();
    let mut claimed = 0usize;
    let mut take = |layer: usize, name: &str| {
        by_site.remove(&(layer, name.to_string())).inspect(|_| {
            claimed += 1;
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
                "inconsistent q8 wiring: embedding and lm_head {} the same buffer \
                 while tie_word_embeddings = {}",
                if Arc::ptr_eq(&bufs[*ie], &bufs[*ih]) { "share" } else { "do not share" },
                config.tie_word_embeddings
            );
        }
    }
    // The KV mode arrives as an argument. `load_with` pins it to F16 for the
    // alignment its own comment gives; the served door passes what its config
    // names. Either way it is decided above this line, not here.
    //
    // 🚨 These two calls stopped compiling when `KvMode` arrived (KV q8,
    // 2026-08-15): this file is under `cfg(cuda)`, so NO development machine
    // type-checks it, and the breakage only surfaced at the first image build,
    // 255 s of CI, on 2026-08-16. `--features cuda` is not covered by
    // `cargo clippy --all-targets` on a Mac, and it is the same class of blind
    // spot as the `planesbench` wiring of the same day. `ops/check-cuda.sh`
    // closes it in a third of a second.
    let mut qwen = match &quant_embed {
        None => crate::model::Qwen3::new_with(&config, vb, &mut take, kv)?,
        Some((bufs, (ie, ih))) => crate::model::Qwen3::new_with_embed(
            &config,
            vb,
            &mut take,
            crate::model::Embed::Q8(Arc::new(CudaEmbed {
                rt: rt.clone(),
                q: bufs[*ie].clone(),
            })),
            crate::model::Head::Q8(Arc::new(CudaEmbed {
                rt: rt.clone(),
                q: bufs[*ih].clone(),
            })),
            kv,
        )?,
    };
    if claimed != total_sites {
        candle_core::bail!(
            "{claimed} projections claimed by the model out of {total_sites} carried by the \
             file"
        );
    }
    // The one place `LLVQ_ROT_SHARE` reaches a model: every other `Qwen3` keeps
    // `RotShare::Off` and cannot be moved by an exported variable.
    qwen.set_rot_share(share);

    Ok(FusedSealed {
        model: qwen,
        tokenizer,
        config,
        layout,
        // From the runtime that compiled it, not from a second resolution.
        // The adapter's boundary: the driver's triple into the portable one.
        prefill: crate::device::Prefill {
            rows: rt.prefill.rows,
            tile: rt.prefill.tile,
            from_env: rt.prefill.from_env,
        },
        embed_mode: emode,
        rot_share: share,
        rot_launches,
        fuse,
        matvec_launches,
        quantized_weights: model.quantized_weights,
        // Counted at read time in `fused::load`, embedding included — the q8
        // extraction changes where those weights sit, not how many there are.
        carried_weights: model.carried_weights,
        file_bytes: model.file_bytes,
        runtime_bytes: model.runtime_bytes,
        carried_bytes,
    })
}
