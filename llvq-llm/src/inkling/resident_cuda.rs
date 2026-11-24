//! The routed experts of one Inkling 2 Mini layer, resident on an NVIDIA card
//! in their lattice codes, computed through the served Tetra kernel.
//!
//! The CUDA twin of `resident.rs` (Metal), with the same role and shapes as
//! [`super::layers::experts_forward`]: `x` `[T, H]` f32 and its [`Routing`] in,
//! the MoE output `[T, H]` f32 out, on the device of `x`. Nothing here is
//! dense. Each expert's three records are read and transcoded to `Tetra48` by
//! [`super::expert_codes::read_expert`], uploaded once by
//! [`FusedRuntime::upload_matrix_ext`], and its two rotations by
//! [`FusedRuntime::upload_rotation_ext`]. Those entry points are additive: the
//! kernels, the launches and the served Qwen3 paths of `fused_cuda` do not
//! move.
//!
//! Per expert that received tokens: gather its rows, rotate them into the
//! basis of `w13` (group 0), one Tetra matvec through gate and one through up,
//! the softcapped SwiGLU of the reference (lines 760-771), rotate into the
//! basis of `down` (group 1), one Tetra matvec through down, weight, and
//! `index_add` in ascending expert order like the reference (lines 787-807).
//! A run of rows goes through the rows kernel in chunks of
//! [`FusedRuntime::prefill_rows`]; each chunk is rotated on its own because
//! the rows kernel reads its activation from offset zero.
//!
//! What differs from the dense reconstruction, as on Metal: the rotation reads
//! binary16 (both the layer input and the SwiGLU output are narrowed once), the
//! kernel stores binary16, row scales and gain centroids are f32, and a row
//! sums in the kernel's order. What differs from Metal: the kernel's summation
//! order. Tokens equal to the Mac's resident run are therefore expected but
//! not guaranteed at a near-tie; `bin/inkrun` reports each step's top-2
//! margin so a divergence can be read against it.

use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use candle_core::{DType, Device, Result, Tensor};

use super::config::InklingConfig;
use super::expert_codes::{check_layer, layer_dir, read_expert, rotation_seed, tetra48_transcoder, verified_layers};
use super::layers::{MoeSource, Routing};
use super::{Inkling, LayerExperts};
use crate::fused::{EmbedMode, FuseMode, FusedLayout, FusedModel, Int4Layout, RotationTables};
use crate::fused_cuda::{ExtRotation, FusedProj, FusedRuntime, MoeExpertRefs, MoeKernels, MoeLayer, MOE_MAX_SELECTED};

/// Tokens one block of [`ResidentExperts::forward_rows_batched`] takes. The
/// pair tensors of a block hold `8 * BATCH_TOKENS` rows, so the peak of a
/// prompt's call is bounded by this count and not by the prompt.
const BATCH_TOKENS: usize = 1024;

fn err(e: anyhow::Error) -> candle_core::Error {
    candle_core::Error::msg(format!("{e:#}"))
}

/// One expert on the card.
pub struct ResidentExpert {
    pub gate: FusedProj,
    pub up: FusedProj,
    pub down: FusedProj,
    /// Group 0: the basis gate and up were quantized in (width `hidden`).
    pub rot_in: ExtRotation,
    /// Group 1: the basis down was quantized in (width `intermediate`).
    pub rot_down: ExtRotation,
}

/// What one layer's upload cost, in bytes and seconds (the fields of the
/// Metal module's `UploadStats`).
#[derive(Debug, Clone, Default)]
pub struct UploadStats {
    pub experts: usize,
    pub weights: u64,
    /// Bytes of the codes files read from disk.
    pub file_bytes: u64,
    /// Device bytes of the matrices as `FusedMatrix::bytes` counts them:
    /// Tetra48 words, row scales and tail (no tail here: widths are multiples
    /// of 24).
    pub matrix_bytes: u64,
    /// Rotation tables: sign bits and the padded odd block.
    pub rotation_bytes: u64,
    pub seconds_read: f64,
    pub seconds_transcode: f64,
    pub seconds_upload: f64,
}

impl UploadStats {
    pub fn uploaded_bytes(&self) -> u64 {
        self.matrix_bytes + self.rotation_bytes
    }
}

/// A `Tetra48` runtime for the experts of `cfg`, on the served tables.
///
/// `FusedRuntime::new` sizes the opt-in of `rot_apply` and the wide unit on
/// the rotation widths of the model it is given, so it is handed one rotation
/// of each expert width (`hidden` for gate and up, `intermediate` for down)
/// and no matrix. The tile is resolved from the card by the runtime itself.
pub fn served_runtime(dev: &Device, cfg: &InklingConfig) -> Result<Arc<FusedRuntime>> {
    let mut rotations = std::collections::HashMap::new();
    for (n, group) in [(cfg.hidden_size, 0u64), (cfg.intermediate_size, 1)] {
        let seed = rotation_seed(0, 0, group);
        rotations.insert((n, seed), RotationTables::build(n, seed).map_err(candle_core::Error::msg)?);
    }
    let model = FusedModel {
        layout: FusedLayout::Tetra48,
        matrices: vec![],
        groups: vec![],
        int4: vec![],
        rotations,
        raw: vec![],
        config_json: vec![],
        tokenizer_json: vec![],
        quantized_weights: 0,
        carried_weights: 0,
        file_bytes: 0,
        runtime_bytes: 0,
    };
    let (rt, projs, segs, int4) = FusedRuntime::new(&model, dev, EmbedMode::F16, FuseMode::Off, Int4Layout::Served)?;
    if !(projs.is_empty() && segs.is_empty() && int4.is_empty()) {
        candle_core::bail!("an expert runtime built from no matrix returned projections");
    }
    Ok(Arc::new(rt))
}

/// Every routed expert of one layer, resident on the card in lattice codes.
pub struct ResidentExperts {
    pub layer: usize,
    rt: Arc<FusedRuntime>,
    dev: Device,
    experts: Vec<ResidentExpert>,
    pub stats: UploadStats,
    /// LLVQ kernel launches (rotations and Tetra matvecs) issued so far.
    launches: AtomicUsize,
    /// The address table of the four-launch path (`moe_tetra.cu`); `None`
    /// when the layer's experts do not share one shape.
    moe: Option<MoeLayer>,
    /// Whether a one-row call takes the four-launch path. On by default.
    fused: std::sync::atomic::AtomicBool,
    /// Whether the four-launch path runs its activation in one launch
    /// (default) or through candle's operations, its reference.
    fused_act: std::sync::atomic::AtomicBool,
    /// Whether a prompt's rows take one launch a step for the whole layer
    /// ([`Self::forward_rows_batched`]) instead of one expert and one chunk
    /// at a time. Off by default: the reference of its gate.
    batch_rows: std::sync::atomic::AtomicBool,
    /// Whether [`Self::forward_rows_batched`] sums each token's weighted rows
    /// in one launch (`moe_wsum`, default) or through candle's multiply,
    /// gathers and adds, its reference.
    wsum_kernel: std::sync::atomic::AtomicBool,
}

impl ResidentExperts {
    /// Upload every expert of `layer` from `dir` (`D/layers/Lxx`). Refuses a
    /// layer without its `VERIFIED` file and any record that is not the slot
    /// `extq` writes (see [`read_expert`]).
    pub fn load(rt: Arc<FusedRuntime>, dev: &Device, dir: &Path, layer: usize, cfg: &InklingConfig) -> Result<Self> {
        check_layer(dir).map_err(err)?;
        let tr = tetra48_transcoder().map_err(err)?;
        let (h, inter) = (cfg.hidden_size, cfg.intermediate_size);
        let mut stats = UploadStats::default();
        let mut experts = Vec::with_capacity(cfg.n_routed_experts);
        for e in 0..cfg.n_routed_experts {
            let em = read_expert(&tr, dir, layer, e, cfg).map_err(err)?;
            stats.file_bytes += em.file_bytes;
            stats.seconds_read += em.seconds_read;
            stats.seconds_transcode += em.seconds_transcode;
            let t = Instant::now();
            let mut up_one = |m: &crate::fused::FusedMatrix| -> Result<FusedProj> {
                stats.matrix_bytes += m.bytes;
                stats.weights += (m.d_out * m.d_in) as u64;
                rt.upload_matrix_ext(m)
            };
            let gate = up_one(&em.gate)?;
            let up = up_one(&em.up)?;
            let down = up_one(&em.down)?;
            let rot_in = rt.upload_rotation_ext(h, rotation_seed(layer, e, 0))?;
            let rot_down = rt.upload_rotation_ext(inter, rotation_seed(layer, e, 1))?;
            stats.rotation_bytes += rot_in.bytes() + rot_down.bytes();
            stats.seconds_upload += t.elapsed().as_secs_f64();
            experts.push(ResidentExpert { gate, up, down, rot_in, rot_down });
        }
        stats.experts = experts.len();
        let refs: Vec<MoeExpertRefs<'_>> = experts
            .iter()
            .map(|x| MoeExpertRefs { gate: &x.gate, up: &x.up, down: &x.down, rot_in: &x.rot_in, rot_down: &x.rot_down })
            .collect();
        let moe = match rt.moe_kernels().and_then(|_| rt.moe_layer(&refs)) {
            Ok(m) => Some(m),
            Err(e) => {
                eprintln!("resident_cuda: layer {layer} keeps the one-expert path: {e}");
                None
            }
        };
        let fused = std::sync::atomic::AtomicBool::new(true);
        let fused_act = std::sync::atomic::AtomicBool::new(true);
        let batch_rows = std::sync::atomic::AtomicBool::new(false);
        let wsum_kernel = std::sync::atomic::AtomicBool::new(true);
        Ok(Self {
            layer,
            rt,
            dev: dev.clone(),
            experts,
            stats,
            launches: AtomicUsize::new(0),
            moe,
            fused,
            fused_act,
            batch_rows,
            wsum_kernel,
        })
    }

    pub fn count(&self) -> usize {
        self.experts.len()
    }

    /// LLVQ kernel launches issued since the last call, and reset.
    pub fn take_launches(&self) -> usize {
        self.launches.swap(0, Ordering::Relaxed)
    }

    fn bump(&self, n: usize) {
        self.launches.fetch_add(n, Ordering::Relaxed);
    }

    /// `x` `[k, d_in]` (one chunk) into the basis of `rot`: one launch.
    fn rotate(&self, rot: &ExtRotation, proj: &FusedProj, x: &Tensor) -> Result<Tensor> {
        let k = x.dim(0)?;
        self.bump(1);
        if k == 1 {
            self.rt.rotate_ext(rot, proj, x)
        } else {
            self.rt.rotate_ext_rows(rot, proj, x, k)
        }
    }

    /// `W x_r` for the `k` rotated rows of one chunk, f32 `[k, d_out]`: one launch.
    fn project(&self, proj: &FusedProj, xr: &Tensor) -> Result<Tensor> {
        let k = xr.dim(0)?;
        self.bump(1);
        let y = if k == 1 {
            self.rt.forward_rotated(proj, xr, &[1, proj.d_out])?
        } else {
            self.rt.forward_rotated_rows(proj, xr, k)?
        };
        y.to_dtype(DType::F32)
    }

    /// Rows a chunk takes: the rows kernel's compiled count, or one.
    fn chunk(&self) -> usize {
        if self.rt.has_rows_kernel() {
            self.rt.prefill_rows().max(1)
        } else {
            1
        }
    }

    /// `rot` then `projs`, chunk by chunk over the rows of `x`, each output
    /// concatenated back to `[n, d_out]`.
    fn rotate_project(&self, rot: &ExtRotation, projs: &[&FusedProj], x: &Tensor) -> Result<Vec<Tensor>> {
        let n = x.dim(0)?;
        let chunk = self.chunk();
        let mut parts: Vec<Vec<Tensor>> = vec![Vec::with_capacity(n.div_ceil(chunk)); projs.len()];
        let mut r = 0;
        while r < n {
            let k = chunk.min(n - r);
            let xc = x.narrow(0, r, k)?;
            let xr = self.rotate(rot, projs[0], &xc)?;
            for (p, out) in projs.iter().zip(parts.iter_mut()) {
                out.push(self.project(p, &xr)?);
            }
            r += k;
        }
        parts
            .into_iter()
            .map(|mut v| if v.len() == 1 { Ok(v.pop().expect("one part")) } else { Tensor::cat(&v, 0) })
            .collect()
    }

    /// Whether the prompt path sums each token's weighted rows in one launch
    /// (`true`, default) or through candle's operations, the reference of
    /// that launch's gate. Meant to be the same bits.
    pub fn set_wsum_kernel(&self, on: bool) {
        self.wsum_kernel.store(on, Ordering::Relaxed);
    }

    /// Whether a one-row call takes the four-launch path (default) or the
    /// one-expert path, its reference.
    pub fn set_fused(&self, on: bool) {
        self.fused.store(on, Ordering::Relaxed);
    }

    /// One row through the four launches of `moe_tetra.cu`: the rotation of
    /// the routed experts, gate and up, the activation (candle's operations,
    /// once for the eight experts), the down rotation, down with the weighted
    /// sum. The values are meant to be the one-expert path's, bit for bit;
    /// `set_fused(false)` is the reference that a card run compares it with.
    /// `None` when the path does not apply: the caller runs the other one.
    fn forward_one_row_fused(&self, x: &Tensor, routing: &Routing, cfg: &InklingConfig) -> Result<Option<Tensor>> {
        let Some(layer) = self.moe.as_ref() else {
            return Ok(None);
        };
        if !self.fused.load(Ordering::Relaxed) || x.dims() != [1, cfg.hidden_size] || routing.indices.len() != 1 {
            return Ok(None);
        }
        let mut order: Vec<(u32, f32)> =
            routing.indices[0].iter().copied().zip(routing.weights[0].iter().copied()).collect();
        order.sort_by_key(|p| p.0);
        if order.is_empty() || order.len() > MOE_MAX_SELECTED || order.windows(2).any(|p| p[0].0 == p[1].0) {
            return Ok(None);
        }
        if let Some(&(e, _)) = order.iter().find(|p| p.0 as usize >= self.experts.len()) {
            candle_core::bail!("token 0: expert {e} of a layer of {}", self.experts.len());
        }
        let mut sel: Vec<u32> = order.iter().map(|p| p.0).collect();
        sel.extend(order.iter().map(|p| p.1.to_bits()));
        let sel = Tensor::from_vec(sel, 2 * order.len(), &self.dev)?;
        self.four_launches(layer, x, &sel, cfg).map(Some)
    }

    /// The four launches on a routing already on the device: `sel` is `[2k]`
    /// u32, the routed ids ascending and distinct, then their weights' bits.
    fn four_launches(&self, layer: &MoeLayer, x: &Tensor, sel: &Tensor, cfg: &InklingConfig) -> Result<Tensor> {
        let k = self.rt.moe_kernels()?;
        let xh = x.to_device(&self.dev)?.to_dtype(DType::F16)?.contiguous()?;
        let xr = self.rt.moe_rotate(k, layer, false, &xh, sel)?;
        let gu = self.rt.moe_gate_up(k, layer, &xr, sel)?;
        // line 771: softcap(gate) * sigmoid(gate) * softcap(up)
        let act = if self.fused_act.load(Ordering::Relaxed) {
            self.rt.moe_activation(k, &gu, cfg.mlp_gate_softcap, cfg.mlp_up_softcap)?
        } else {
            let (g, u) = (gu.get(0)?, gu.get(1)?);
            softcap(&g, cfg.mlp_gate_softcap)?
                .mul(&candle_nn::ops::sigmoid(&g)?)?
                .mul(&softcap(&u, cfg.mlp_up_softcap)?)?
                .to_dtype(DType::F16)?
                .contiguous()?
        };
        let ar = self.rt.moe_rotate(k, layer, true, &act, sel)?;
        let y = self.rt.moe_down_sum(k, layer, &ar, sel)?;
        self.bump(4);
        y.to_device(x.device())
    }

    /// Several rows in one launch a step for the whole layer: every (token,
    /// expert) pair is a row, the pairs sorted by expert then token as the
    /// one-expert loop of [`Self::forward`] visits them. The rotation of the
    /// pairs, gate and up, candle's activation, the down rotation, down, then
    /// each token's weighted rows added in ascending expert order from zero.
    /// The same kernels' bodies on the same rows and the same order of sums as
    /// [`Self::forward`]: the values are meant to be its own, bit for bit.
    /// The tokens go in consecutive blocks of [`BATCH_TOKENS`] (see
    /// [`Self::rows_block`]); a row's value does not depend on the other rows
    /// of its launch, and a token's sum stays within its block, so the block
    /// size does not move a bit.
    /// `None` when the path does not apply: the caller runs the other one.
    fn forward_rows_batched(&self, x: &Tensor, routing: &Routing, cfg: &InklingConfig) -> Result<Option<Tensor>> {
        let Some(layer) = self.moe.as_ref() else {
            return Ok(None);
        };
        let t_rows = routing.indices.len();
        let k_sel = routing.indices.first().map_or(0, Vec::len);
        if !self.batch_rows.load(Ordering::Relaxed)
            || t_rows < 2
            || k_sel == 0
            || x.dims() != [t_rows, cfg.hidden_size]
            || routing.indices.iter().zip(&routing.weights).any(|(i, w)| i.len() != k_sel || w.len() != k_sel)
        {
            return Ok(None);
        }
        let k = self.rt.moe_kernels()?;
        if !self.rt.moe_has_rows(k) {
            return Ok(None);
        }
        // Each token's experts in ascending order, checked for the whole
        // prompt before the first launch.
        let mut owns: Vec<Vec<(u32, f32)>> = Vec::with_capacity(t_rows);
        for (t, (idx, w)) in routing.indices.iter().zip(&routing.weights).enumerate() {
            let mut own: Vec<(u32, f32)> = idx.iter().copied().zip(w.iter().copied()).collect();
            own.sort_by_key(|p| p.0);
            if own.windows(2).any(|p| p[0].0 == p[1].0) {
                return Ok(None);
            }
            if let Some(&(e, _)) = own.iter().find(|p| p.0 as usize >= self.experts.len()) {
                candle_core::bail!("token {t}: expert {e} of a layer of {}", self.experts.len());
            }
            owns.push(own);
        }
        let xd = x.to_device(&self.dev)?.to_dtype(DType::F32)?.contiguous()?;
        // Consecutive token ranges, in token order. A token's rows stay in
        // one block, so its sum is the same sum whatever the block size.
        let mut parts: Vec<Tensor> = Vec::with_capacity(t_rows.div_ceil(BATCH_TOKENS));
        let mut t0 = 0;
        while t0 < t_rows {
            let tb = BATCH_TOKENS.min(t_rows - t0);
            parts.push(self.rows_block(k, layer, &xd, &owns[t0..t0 + tb], t0, cfg)?);
            self.bump(4);
            t0 += tb;
        }
        drop(xd);
        let out = if parts.len() == 1 { parts.pop().expect("one block") } else { Tensor::cat(&parts, 0)? };
        drop(parts);
        out.to_device(x.device()).map(Some)
    }

    /// One block of [`Self::forward_rows_batched`]: the tokens `t0..t0 + owns.len()`
    /// of `xd` (f32 `[T, H]` on the card), `owns` their experts in ascending
    /// order. Returns f32 `[owns.len(), H]`. Each intermediate tensor is
    /// dropped once its last reader has run, so the peak follows the block.
    fn rows_block(
        &self,
        k: &MoeKernels,
        layer: &MoeLayer,
        xd: &Tensor,
        owns: &[Vec<(u32, f32)>],
        t0: usize,
        cfg: &InklingConfig,
    ) -> Result<Tensor> {
        let t_rows = owns.len();
        let k_sel = owns.first().map_or(0, Vec::len);
        // (expert, token in the block, weight, rank of the expert among the token's own, ascending)
        let mut pairs: Vec<(u32, u32, f32, u32)> = Vec::with_capacity(t_rows * k_sel);
        for (t, own) in owns.iter().enumerate() {
            for (rank, &(e, wt)) in own.iter().enumerate() {
                pairs.push((e, t as u32, wt, rank as u32));
            }
        }
        pairs.sort_by_key(|p| (p.0, p.1));
        let n = pairs.len();
        let rows = self.rt.prefill_rows().max(1);
        let mut chunks: Vec<u32> = vec![];
        let mut start = 0;
        while start < n {
            let e = pairs[start].0;
            let mut end = start;
            while end < n && end - start < rows && pairs[end].0 == e {
                end += 1;
            }
            chunks.extend([e, start as u32, (end - start) as u32]);
            start = end;
        }
        let n_chunks = chunks.len() / 3;
        let dev = &self.dev;
        let pair_expert = Tensor::from_vec(pairs.iter().map(|p| p.0).collect::<Vec<_>>(), n, dev)?;
        // Global token ids: the rows of `xd` this block reads.
        let pair_token = Tensor::from_vec(pairs.iter().map(|p| t0 as u32 + p.1).collect::<Vec<_>>(), n, dev)?;
        let chunks = Tensor::from_vec(chunks, 3 * n_chunks, dev)?;

        let xs = xd.index_select(&pair_token, 0)?.to_dtype(DType::F16)?.contiguous()?;
        drop(pair_token);
        let xr = self.rt.moe_rotate_pairs(k, layer, false, &xs, &pair_expert)?;
        drop(xs);
        let gu = self.rt.moe_matrix_pairs(k, layer, false, &xr, &chunks)?.to_dtype(DType::F32)?;
        drop(xr);
        let (g, u) = (gu.get(0)?, gu.get(1)?);
        // `g` and `u` are views: the storage goes when the last of the three does.
        drop(gu);
        // line 771: softcap(gate) * sigmoid(gate) * softcap(up)
        let act = softcap(&g, cfg.mlp_gate_softcap)?
            .mul(&candle_nn::ops::sigmoid(&g)?)?
            .mul(&softcap(&u, cfg.mlp_up_softcap)?)?
            .to_dtype(DType::F16)?
            .contiguous()?;
        drop((g, u));
        let ar = self.rt.moe_rotate_pairs(k, layer, true, &act, &pair_expert)?;
        drop((act, pair_expert));
        let yh = self.rt.moe_matrix_pairs(k, layer, true, &ar, &chunks)?;
        drop((ar, chunks));
        if self.wsum_kernel.load(Ordering::Relaxed) {
            // For token t and rank r at t * k_sel + r: its pair, then that pair's weight bits.
            let mut side = vec![0u32; 2 * t_rows * k_sel];
            for (p, &(_, t, wt, rank)) in pairs.iter().enumerate() {
                let at = t as usize * k_sel + rank as usize;
                side[at] = p as u32;
                side[t_rows * k_sel + at] = wt.to_bits();
            }
            let side = Tensor::from_vec(side, 2 * t_rows * k_sel, dev)?;
            return self.rt.moe_weighted_sum(k, &yh, &side, k_sel);
        }
        // The reference of `moe_wsum`: candle's multiply, gathers and adds.
        // by_rank[rank][token]: the pair holding that token's rank-th expert.
        let mut by_rank = vec![vec![0u32; t_rows]; k_sel];
        for (p, &(_, t, _, rank)) in pairs.iter().enumerate() {
            by_rank[rank as usize][t as usize] = p as u32;
        }
        let wts = Tensor::from_vec(pairs.iter().map(|p| p.2).collect::<Vec<_>>(), (n, 1), dev)?;
        let y = yh.get(0)?.to_dtype(DType::F32)?.broadcast_mul(&wts)?;
        drop((yh, wts));
        // zeros + the first expert's row, then the others, as `index_add` an expert at a time did.
        let mut out = Tensor::zeros((t_rows, xd.dim(1)?), DType::F32, dev)?;
        for sel in by_rank {
            let sel = Tensor::from_vec(sel, t_rows, dev)?;
            out = (out + y.index_select(&sel, 0)?)?;
        }
        Ok(out)
    }

    /// The MoE of this layer, see the module documentation.
    pub fn forward(&self, x: &Tensor, routing: &Routing, cfg: &InklingConfig) -> Result<Tensor> {
        if let Some(y) = self.forward_one_row_fused(x, routing, cfg)? {
            return Ok(y);
        }
        if let Some(y) = self.forward_rows_batched(x, routing, cfg)? {
            return Ok(y);
        }
        let inter = cfg.intermediate_size;
        let xd = x.to_device(&self.dev)?.to_dtype(DType::F32)?.contiguous()?;
        let mut out = xd.zeros_like()?;
        let mut per_expert: Vec<Vec<(u32, f32)>> = vec![vec![]; self.experts.len()];
        for (t, (idx, w)) in routing.indices.iter().zip(&routing.weights).enumerate() {
            for (&e, &wt) in idx.iter().zip(w) {
                let slot = per_expert.get_mut(e as usize).ok_or_else(|| {
                    candle_core::Error::msg(format!("token {t}: expert {e} of a layer of {}", self.experts.len()))
                })?;
                slot.push((t as u32, wt));
            }
        }
        for (e, toks) in per_expert.iter().enumerate() {
            if toks.is_empty() {
                continue;
            }
            let ex = &self.experts[e];
            let n = toks.len();
            let ids = Tensor::from_vec(toks.iter().map(|p| p.0).collect::<Vec<_>>(), n, &self.dev)?;
            let wts = Tensor::from_vec(toks.iter().map(|p| p.1).collect::<Vec<_>>(), (n, 1), &self.dev)?;
            let xs = xd.index_select(&ids, 0)?.contiguous()?;
            let gu = self.rotate_project(&ex.rot_in, &[&ex.gate, &ex.up], &xs)?;
            let (g, u) = (&gu[0], &gu[1]);
            if g.dims() != [n, inter] || u.dims() != [n, inter] {
                candle_core::bail!("expert {e}: gate {:?} and up {:?} for [{n}, {inter}]", g.dims(), u.dims());
            }
            // line 771: softcap(gate) * sigmoid(gate) * softcap(up)
            let act = softcap(g, cfg.mlp_gate_softcap)?
                .mul(&candle_nn::ops::sigmoid(g)?)?
                .mul(&softcap(u, cfg.mlp_up_softcap)?)?
                .contiguous()?;
            let y = self.rotate_project(&ex.rot_down, &[&ex.down], &act)?.remove(0).broadcast_mul(&wts)?;
            out = out.index_add(&ids, &y, 0)?;
        }
        out.to_device(x.device())
    }
}

/// `Inkling2LoopSoftcap`, lines 751-757.
fn softcap(x: &Tensor, cap: f64) -> Result<Tensor> {
    (x / cap)?.tanh()? * cap
}

/// The role and shapes of [`super::layers::experts_forward`], from the codes
/// resident in `src` instead of dense matrices.
pub fn experts_forward(x: &Tensor, routing: &Routing, src: &ResidentExperts, cfg: &InklingConfig) -> Result<Tensor> {
    src.forward(x, routing, cfg)
}

impl MoeSource for ResidentExperts {
    fn moe(&self, x: &Tensor, routing: &Routing, cfg: &InklingConfig) -> Result<Tensor> {
        experts_forward(x, routing, self, cfg)
    }

    fn take_launches(&self) -> usize {
        ResidentExperts::take_launches(self)
    }

    fn set_four_launches(&mut self, on: bool) {
        self.set_fused(on);
    }

    fn set_fused_act(&mut self, on: bool) {
        self.fused_act.store(on, Ordering::Relaxed);
    }

    fn set_batch_rows(&mut self, on: bool) {
        self.batch_rows.store(on, Ordering::Relaxed);
    }

    fn moe_routed_on_device(&self, x: &Tensor, sel: &Tensor, experts: usize, cfg: &InklingConfig) -> Result<Option<Tensor>> {
        let Some(layer) = self.moe.as_ref() else {
            return Ok(None);
        };
        let n = sel.elem_count();
        if !self.fused.load(Ordering::Relaxed)
            || x.dims() != [1, cfg.hidden_size]
            || experts != self.experts.len()
            || sel.dtype() != DType::U32
            || n == 0
            || !n.is_multiple_of(2)
            || n / 2 > MOE_MAX_SELECTED
        {
            return Ok(None);
        }
        self.four_launches(layer, x, sel, cfg).map(Some)
    }
}

/// Make resident on the card every layer of `layers_dir` (`D/layers`) whose
/// `VERIFIED` file exists; the others keep their native experts. Returns the
/// resident layers and what each upload cost. `model` must live on the CUDA
/// device of `rt`.
pub fn make_resident(
    model: &mut Inkling,
    layers_dir: &Path,
    rt: Arc<FusedRuntime>,
) -> Result<Vec<(usize, UploadStats)>> {
    let dev = model.device().clone();
    let cfg = model.cfg.clone();
    let mut out = vec![];
    for l in verified_layers(layers_dir, cfg.num_hidden_layers) {
        let r = ResidentExperts::load(rt.clone(), &dev, &layer_dir(layers_dir, l), l, &cfg)?;
        out.push((l, r.stats.clone()));
        model.set_experts(l, LayerExperts::Source(Box::new(r)));
    }
    Ok(out)
}
