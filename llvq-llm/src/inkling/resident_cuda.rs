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
use crate::fused_cuda::{ExtRotation, FusedProj, FusedRuntime};

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
        Ok(Self { layer, rt, dev: dev.clone(), experts, stats, launches: AtomicUsize::new(0) })
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

    /// The MoE of this layer, see the module documentation.
    pub fn forward(&self, x: &Tensor, routing: &Routing, cfg: &InklingConfig) -> Result<Tensor> {
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
