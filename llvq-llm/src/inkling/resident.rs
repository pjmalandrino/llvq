//! The routed experts of one Inkling 2 Mini layer, resident on the GPU in
//! their lattice codes, computed through the served Tetra kernel on Metal.
//!
//! Nothing here is dense. Each expert's three records (gate, up, down) are read
//! from `layers/Lxx/codes/eNNN.llvqrec`, transcoded to the served `tetra48`
//! stream by [`crate::fused::Transcoder`], and uploaded once by
//! [`MetalRuntime::upload_stream`], exactly as `fused_metal::load_resolved`
//! does for a sealed Qwen3 file. The two rotations of an expert are built by
//! [`RotationTables::build`] from the seed each record carries and uploaded by
//! [`MetalRuntime::upload_rotation`]. No kernel, shader or served path is
//! changed: this module only calls them.
//!
//! [`experts_forward`] has the role and the shapes of
//! [`super::layers::experts_forward`]: same `x` `[T, H]` f32, same
//! [`Routing`], same output `[T, H]` f32 on the device of `x`. Per expert that
//! received tokens: gather its rows, rotate them into the basis of `w13`
//! (group 0), one Tetra matvec per row through gate and one through up, the
//! softcapped SwiGLU of the reference (lines 760-771), rotate into the basis
//! of `down` (group 1), one Tetra matvec per row through down, weight, and
//! `index_add` in ascending expert order like the reference (lines 787-807).
//!
//! What the kernel path does differently from the dense reconstruction, and so
//! why the two are not bit-equal: the rotation reads its input in binary16
//! (`MetalRuntime::rotate`), so both the layer input and the SwiGLU output are
//! narrowed to f16 once; the row scales and the gain centroids are f32 where
//! the reconstruction applied them in f64; and the summation order of a row is
//! the kernel's 32-lane order in the rotated basis. `tests/inkling_resident.rs`
//! states the tolerance this allows and measures it.

use std::io::Read;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use candle_core::{DType, Device, Result, Tensor};

use super::config::InklingConfig;
use super::layers::{ExpertSource, MoeSource, Routing};
use crate::fused::{FusedLayout, HostStream, RotationTables, Transcoder};
use crate::fused_metal::{MetalRotation, MetalRuntime, MetalTetraProj};

/// The tile the served Metal door uses (`fused_metal::load_resolved_tuned`).
pub const SERVED_TILE: usize = 64;

/// Names of the three records of an expert, in file order.
const MATS: [&str; 3] = ["gate", "up", "down"];

/// The rotation seed `extq` gives an expert's input group: group 0 is the
/// input of the fused `w13` (gate and up), group 1 the input of `down`
/// (`W/llvq-llm/src/bin/extq.rs`, `fn seed`).
pub fn rotation_seed(layer: usize, expert: usize, group: u64) -> u64 {
    crate::calib::ROTATION_SEED ^ ((layer as u64) << 32) ^ ((expert as u64) << 20) ^ (group << 16)
}

/// The record name `extq` writes.
pub fn record_name(layer: usize, expert: usize, mat: &str) -> String {
    format!("model.layers.{layer}.mlp.experts.{expert}.{mat}_proj.weight")
}

/// One expert on the device.
pub struct ResidentExpert {
    pub gate: MetalTetraProj,
    pub up: MetalTetraProj,
    pub down: MetalTetraProj,
    /// Group 0: the basis gate and up were quantized in (width `hidden`).
    pub rot_in: MetalRotation,
    /// Group 1: the basis down was quantized in (width `intermediate`).
    pub rot_down: MetalRotation,
}

/// What one layer's upload cost, in bytes and seconds.
#[derive(Debug, Clone, Default)]
pub struct UploadStats {
    pub experts: usize,
    pub weights: u64,
    /// Bytes of the codes files read from disk.
    pub file_bytes: u64,
    /// Tetra48 words, as uploaded (row stride and guard word included).
    pub word_bytes: u64,
    /// Row scales (f32), gain centroids (2 f32) and tails (f16, or the
    /// one-element placeholder `upload` makes for a matrix without one).
    pub side_bytes: u64,
    /// Rotation tables: sign bits and the padded odd block.
    pub rotation_bytes: u64,
    pub seconds_read: f64,
    pub seconds_transcode: f64,
    pub seconds_upload: f64,
}

impl UploadStats {
    pub fn uploaded_bytes(&self) -> u64 {
        self.word_bytes + self.side_bytes + self.rotation_bytes
    }
}

/// Every routed expert of one layer, resident on the device in lattice codes.
pub struct ResidentExperts {
    pub layer: usize,
    rt: Arc<MetalRuntime>,
    dev: Device,
    experts: Vec<ResidentExpert>,
    pub stats: UploadStats,
    /// LLVQ kernel launches (rotations and Tetra matvecs) issued so far.
    launches: AtomicUsize,
}

/// A runtime on the served tables and tile, without host copies.
pub fn served_runtime(dev: &Device) -> Result<Arc<MetalRuntime>> {
    let tables = crate::fused::tetra48_tables(&llvq_search::tetra::Tetra::new());
    Ok(Arc::new(MetalRuntime::from_tables(dev, &tables, SERVED_TILE)?))
}

fn read_file(path: &Path) -> Result<Vec<u8>> {
    let mut f = std::fs::File::open(path)
        .map_err(|e| candle_core::Error::msg(format!("open {}: {e}", path.display())))?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf)?;
    Ok(buf)
}

impl ResidentExperts {
    /// Upload every expert of `layer` from `layer_dir` (`D/layers/Lxx`).
    ///
    /// Refuses a layer without its `VERIFIED` file, a record whose name,
    /// kind, shape or rotation seed is not the one `extq` writes for this
    /// slot, and a codes file with trailing bytes.
    pub fn load(
        rt: Arc<MetalRuntime>,
        dev: &Device,
        layer_dir: &Path,
        layer: usize,
        cfg: &InklingConfig,
    ) -> Result<Self> {
        let verified = layer_dir.join("VERIFIED");
        if !verified.exists() {
            candle_core::bail!("{}: no VERIFIED file, the layer is not complete", layer_dir.display());
        }
        let (h, inter, n) = (cfg.hidden_size, cfg.intermediate_size, cfg.n_routed_experts);
        let tr = Transcoder::for_kind(FusedLayout::Tetra48, llvq_artifact::CodeKind::Tetra)
            .map_err(candle_core::Error::msg)?;
        let mut stats = UploadStats::default();
        let mut experts = Vec::with_capacity(n);
        for e in 0..n {
            let path: PathBuf = layer_dir.join("codes").join(format!("e{e:03}.llvqrec"));
            let t = Instant::now();
            let bytes = read_file(&path)?;
            stats.file_bytes += bytes.len() as u64;
            let mut r = &bytes[..];
            let mut recs = Vec::with_capacity(3);
            for _ in MATS {
                let rec = llvq_artifact::read_record(&mut r, llvq_artifact::VERSION)
                    .map_err(|err| candle_core::Error::msg(format!("{}: {err}", path.display())))?;
                recs.push(rec);
            }
            if !r.is_empty() {
                candle_core::bail!("{}: {} trailing bytes after three records", path.display(), r.len());
            }
            stats.seconds_read += t.elapsed().as_secs_f64();

            let mut projs = Vec::with_capacity(3);
            let mut seeds = Vec::with_capacity(3);
            for (rec, (mat, (d_out, d_in, group))) in
                recs.into_iter().zip(MATS.iter().zip([(inter, h, 0u64), (inter, h, 0), (h, inter, 1)]))
            {
                let llvq_artifact::Record::Lattice(m) = rec else {
                    candle_core::bail!("{}: {mat} is not a lattice record", path.display());
                };
                let want = record_name(layer, e, mat);
                if m.name != want {
                    candle_core::bail!("{}: record {:?} where {want:?} belongs", path.display(), m.name);
                }
                if m.kind != llvq_artifact::CodeKind::Tetra {
                    candle_core::bail!("{want}: kind {:?}, the tetra48 layout reads Tetra", m.kind);
                }
                if (m.d_out, m.d_in) != (d_out, d_in) {
                    candle_core::bail!("{want}: {}x{}, expected {d_out}x{d_in}", m.d_out, m.d_in);
                }
                if m.centroids.len() != 2 {
                    candle_core::bail!("{want}: {} centroids, the kernel reads one gain bit", m.centroids.len());
                }
                let seed = rotation_seed(layer, e, group);
                if m.rotation_seed != Some(seed) {
                    candle_core::bail!("{want}: rotation seed {:?}, extq gives {seed}", m.rotation_seed);
                }
                seeds.push(seed);
                let nblocks = m.d_in / llvq_core::DIM;
                let t = Instant::now();
                let (stream, _) = tr.stream(&m.indices, &m.gains, m.d_out, nblocks).map_err(candle_core::Error::msg)?;
                let HostStream::Tetra48 { words, stride_u32 } = stream else {
                    candle_core::bail!("{want}: the transcoder returned a non-Tetra48 stream");
                };
                let gscale = [m.centroids[0] as f32, m.centroids[1] as f32];
                let rscale: Vec<f32> = m.row_scales.iter().map(|&v| v as f32).collect();
                let tail = crate::fused::tail_f16_bits(&m.tail);
                stats.seconds_transcode += t.elapsed().as_secs_f64();
                let t = Instant::now();
                let proj = rt.upload_stream(
                    &want, m.d_out, m.d_in, nblocks, stride_u32 as usize, &words, &gscale, &rscale, &tail,
                )?;
                stats.seconds_upload += t.elapsed().as_secs_f64();
                stats.word_bytes += words.len() as u64 * 4;
                stats.side_bytes += 8 + 4 * rscale.len() as u64 + 2 * tail.len().max(1) as u64;
                stats.weights += (m.d_out * m.d_in) as u64;
                projs.push(proj);
            }
            let mut rot = |n: usize, seed: u64| -> Result<MetalRotation> {
                let t = Instant::now();
                let tables = RotationTables::build(n, seed).map_err(candle_core::Error::msg)?;
                stats.seconds_transcode += t.elapsed().as_secs_f64();
                let t = Instant::now();
                let up = rt.upload_rotation(&tables)?;
                stats.seconds_upload += t.elapsed().as_secs_f64();
                stats.rotation_bytes += 4 * tables.signbits.len() as u64 + 4 * tables.small.len().max(1) as u64;
                Ok(up)
            };
            let rot_in = rot(h, seeds[0])?;
            let rot_down = rot(inter, seeds[2])?;
            let down = projs.pop().expect("three records");
            let up = projs.pop().expect("three records");
            let gate = projs.pop().expect("three records");
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

    /// `x` `[n, d]` into the basis of `rot`: one launch for one row or for all.
    fn rotate(&self, rot: &MetalRotation, name: &str, x: &Tensor) -> Result<Tensor> {
        let n = x.dim(0)?;
        self.bump(1);
        if n == 1 {
            self.rt.rotate(rot, name, x)
        } else {
            self.rt.rotate_rows(rot, name, x, n)
        }
    }

    /// `W x_r` for every row of the rotated `xr` `[n, d_in]`, as f32 `[n, d_out]`:
    /// one matvec a row on the served tuning, or chunks of `prefill_rows` rows
    /// through the rows kernel when the runtime was put on that tuning.
    fn project(&self, proj: &MetalTetraProj, xr: &Tensor) -> Result<Tensor> {
        let n = xr.dim(0)?;
        let chunk = self.rt.prefill_rows().max(1);
        let mut parts = Vec::with_capacity(n.div_ceil(chunk));
        let mut r = 0;
        while r < n {
            let k = chunk.min(n - r);
            let rows = xr.narrow(0, r, k)?;
            self.bump(1);
            let y = if k == 1 && chunk == 1 {
                self.rt.matvec(proj, &rows)?
            } else {
                self.rt.matvec_rows(proj, &rows.contiguous()?, k)?.to_dtype(DType::F32)?
            };
            parts.push(y);
            r += k;
        }
        if parts.len() == 1 {
            Ok(parts.pop().expect("one part"))
        } else {
            Tensor::cat(&parts, 0)
        }
    }

    /// The MoE of this layer, see [`experts_forward`].
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
            let xr = self.rotate(&ex.rot_in, &ex.gate.name, &xs)?;
            let g = self.project(&ex.gate, &xr)?;
            let u = self.project(&ex.up, &xr)?;
            if g.dims() != [n, inter] || u.dims() != [n, inter] {
                candle_core::bail!("expert {e}: gate {:?} and up {:?} for [{n}, {inter}]", g.dims(), u.dims());
            }
            // line 771: softcap(gate) * sigmoid(gate) * softcap(up)
            let act = softcap(&g, cfg.mlp_gate_softcap)?
                .mul(&candle_nn::ops::sigmoid(&g)?)?
                .mul(&softcap(&u, cfg.mlp_up_softcap)?)?
                .contiguous()?;
            let ar = self.rotate(&ex.rot_down, &ex.down.name, &act)?;
            let y = self.project(&ex.down, &ar)?.broadcast_mul(&wts)?;
            out = out.index_add(&ids, &y, 0)?;
        }
        out.to_device(x.device())
    }
}

/// `Inkling2LoopSoftcap`, lines 751-757 (`super::layers::softcap` is private).
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

/// The layers of `layers_dir` (`D/layers`) whose `Lxx/VERIFIED` exists, ascending.
pub fn verified_layers(layers_dir: &Path, n_layers: usize) -> Vec<usize> {
    (0..n_layers).filter(|l| layers_dir.join(format!("L{l:02}")).join("VERIFIED").exists()).collect()
}

/// The dense reconstructions `extq` writes beside the codes, as an
/// [`ExpertSource`]: `w13_hat` `[E, 2I, H]` (rows `0..I` gate, `I..2I` up,
/// already de-interleaved) and `w2_hat` `[E, H, I]`, raw little-endian, f32
/// in layers 6 to 17 and f16 elsewhere. One expert is read per call.
pub struct HatExperts {
    w13: std::fs::File,
    w2: std::fs::File,
    half: bool,
    experts: usize,
    hidden: usize,
    inter: usize,
}

impl HatExperts {
    /// Opens `w13_hat.f32` and `w2_hat.f32`, or their `.f16` twins, and
    /// checks their sizes against `cfg`.
    pub fn open(layer_dir: &Path, cfg: &InklingConfig) -> Result<Self> {
        let (h, i, n) = (cfg.hidden_size, cfg.intermediate_size, cfg.n_routed_experts);
        let half = !layer_dir.join("w13_hat.f32").exists();
        let ext = if half { "f16" } else { "f32" };
        let open = |name: &str, want: usize| -> Result<std::fs::File> {
            let p = layer_dir.join(format!("{name}.{ext}"));
            let f = std::fs::File::open(&p)
                .map_err(|e| candle_core::Error::msg(format!("open {}: {e}", p.display())))?;
            let len = f.metadata()?.len();
            let want = (want * if half { 2 } else { 4 }) as u64;
            if len != want {
                candle_core::bail!("{}: {len} bytes, expected {want}", p.display());
            }
            Ok(f)
        };
        Ok(Self {
            w13: open("w13_hat", n * 2 * i * h)?,
            w2: open("w2_hat", n * h * i)?,
            half,
            experts: n,
            hidden: h,
            inter: i,
        })
    }

    fn read(&self, f: &std::fs::File, e: usize, len: usize) -> Result<Vec<f32>> {
        let w = if self.half { 2 } else { 4 };
        let mut buf = vec![0u8; len * w];
        f.read_exact_at(&mut buf, (e * len * w) as u64)?;
        Ok(if self.half {
            buf.chunks_exact(2).map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32()).collect()
        } else {
            buf.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
        })
    }
}

impl ExpertSource for HatExperts {
    fn count(&self) -> usize {
        self.experts
    }
    fn expert(&self, e: usize, dev: &Device) -> Result<(Tensor, Tensor)> {
        let (h, i) = (self.hidden, self.inter);
        let gu = self.read(&self.w13, e, 2 * i * h)?;
        let down = self.read(&self.w2, e, h * i)?;
        Ok((Tensor::from_vec(gu, (2 * i, h), dev)?, Tensor::from_vec(down, (h, i), dev)?))
    }
}
