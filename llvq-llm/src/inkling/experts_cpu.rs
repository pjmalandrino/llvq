//! The experts of Inkling 2 Mini on the CPU, read from their lattice codes.
//!
//! The third implementor of [`MoeSource`] beside `resident::ResidentExperts`
//! (Metal) and `resident_cuda::ResidentExperts` (CUDA), for a machine with
//! neither card. Each expert's three Tetra records are read from
//! `layers/Lxx/codes/eNNN.llvqrec` by [`expert_codes::read_expert`] and kept
//! as the served `tetra48` words ([`CpuLattice`]); `tetra_cpu` decodes them
//! at every matvec. Under `LLVQ_CPU_WEIGHTS=nibbles` the codes are decoded
//! once at load into four bits a coordinate, the same outputs bit for bit.
//!
//! The math is [`super::layers::experts_forward`]'s, in the same order: per
//! routed expert in ascending index, the input rotated into the gate group's
//! basis once for gate and up, the softcapped SwiGLU in f32, the activation
//! rotated into the down group's basis, the down matvec, times the routing
//! weight, added to a sum that starts from zeros. A prompt groups the rows
//! routed to each expert and takes them in one batched matvec, then adds
//! each row's output to its token in the same ascending order, so one row
//! and the same row inside a prompt agree to rounding.
//!
//! Memory, *computed* for the real config (H = 2,304, I = 1,152, 128 experts,
//! 24 layers): the words are what the codes files hold, 2 bits a weight and
//! the row scales. The rotations are two a expert, built once at load:
//! `n` sign values in f64 and the odd block (`9 x 9` f64 at both widths),
//! 28.9 KB an expert, 3.7 MB a layer, 89 MB for the model. One [`Tables`]
//! is shared by every matrix.

use std::path::Path;
use std::sync::Arc;

use anyhow::Context;
use candle_core::{DType, Device, Result, Tensor};
use llvq_quant::rotation::Rotation;

use super::config::InklingConfig;
use super::expert_codes::{self, check_layer, layer_dir, read_expert, rotation_seed, tetra48_transcoder};
use super::layers::{MoeSource, Routing};
use super::{AttnKind, Inkling, LayerExperts};
use crate::device::LatticeProj;
use crate::fused::{FusedMatrix, HostStream};
use crate::fused_cpu::{phases, CpuLattice, CpuWeights};
use crate::tetra_cpu::{Tables, TetraMatrix};

/// Tokens of a prompt one forward call reads on the CPU stack
/// ([`Inkling::set_prefill_piece`]): the runtime's CUDA value.
pub const PROMPT_PIECE: usize = 512;
/// Query rows of one attention score tensor on the CPU stack
/// ([`Inkling::set_attn_chunk`]): the runtime's CUDA value.
pub const SCORE_ROWS: usize = 128;
/// Positions past its window a bounded sliding slot keeps on the CPU stack
/// ([`Inkling::set_kv_window`]): the runtime's CUDA value.
pub const KV_SLACK: usize = 1024;

/// One expert: gate and up read the same rotated input, down its own.
pub struct CpuExpert {
    pub gate: CpuLattice,
    pub up: CpuLattice,
    pub down: CpuLattice,
}

/// What loading one layer cost.
#[derive(Debug, Clone, Default)]
pub struct LoadStats {
    pub experts: usize,
    pub weights: u64,
    /// Bytes of the codes files read from disk.
    pub file_bytes: u64,
    /// Host bytes of the matrices: words (or nibbles), row scales, tails.
    pub matrix_bytes: u64,
    /// Host bytes of the rotations: signs and odd block, in f64.
    pub rotation_bytes: u64,
    /// Wall time of the layer's load, read and transcode included.
    pub seconds: f64,
}

/// Every expert of one layer on the CPU.
pub struct CpuExperts {
    layer: usize,
    hidden: usize,
    inter: usize,
    experts: Vec<CpuExpert>,
    pub stats: LoadStats,
}

/// Bytes a rotation of `n` holds: the signs and the odd block, in f64.
fn rotation_bytes(r: &Rotation) -> u64 {
    8 * (r.signs().len() + r.small().len()) as u64
}

/// One record read by `expert_codes` as the CPU holds it.
fn lattice_of(m: FusedMatrix, rot: Arc<Rotation>, tables: &Arc<Tables>, weights: CpuWeights) -> std::result::Result<CpuLattice, String> {
    let HostStream::Tetra48 { words, stride_u32 } = m.stream else {
        return Err(format!("{}: not a Tetra48 stream", m.name));
    };
    let key = m.rotation.ok_or_else(|| format!("{}: no rotation key, the experts are stored rotated", m.name))?;
    if key.0 != rot.dim() {
        return Err(format!("{}: rotation key {key:?} for a rotation of {}", m.name, rot.dim()));
    }
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
    let p = CpuLattice::new(m.name, tables.clone(), tm, Some(rot), Some(key), DType::F32)?;
    Ok(match weights {
        CpuWeights::Packed => p,
        CpuWeights::Nibbles => p.expanded(),
    })
}

impl CpuExperts {
    /// The experts of a layer from their projections, `experts[e]` expert
    /// `e`. Refuses a projection of the wrong shape, and gate and up that do
    /// not turn their input by the same rotation.
    pub fn new(layer: usize, experts: Vec<CpuExpert>, hidden: usize, inter: usize) -> std::result::Result<Self, String> {
        if experts.is_empty() {
            return Err(format!("layer {layer}: no expert"));
        }
        for (e, x) in experts.iter().enumerate() {
            for (what, p, d_out, d_in) in [("gate", &x.gate, inter, hidden), ("up", &x.up, inter, hidden), ("down", &x.down, hidden, inter)] {
                if (p.d_out(), p.d_in()) != (d_out, d_in) {
                    return Err(format!(
                        "layer {layer} expert {e}: {what} is {}x{}, expected {d_out}x{d_in}",
                        p.d_out(),
                        p.d_in()
                    ));
                }
            }
            if !x.gate.shares_rotation_with(&x.up) {
                return Err(format!("layer {layer} expert {e}: gate and up turn their input by two rotations"));
            }
        }
        let mut stats = LoadStats { experts: experts.len(), ..LoadStats::default() };
        for x in &experts {
            for p in [&x.gate, &x.up, &x.down] {
                stats.weights += (p.d_out() * p.d_in()) as u64;
                stats.matrix_bytes += p.bytes();
            }
        }
        Ok(Self { layer, hidden, inter, experts, stats })
    }

    /// Read, check and hold every expert of `layer` from `dir`
    /// (`D/layers/Lxx`), its `VERIFIED` file required. The experts are read
    /// in parallel on the CPU pool; `tables` is shared by every matrix.
    pub fn load(dir: &Path, layer: usize, cfg: &InklingConfig, tables: &Arc<Tables>, weights: CpuWeights) -> anyhow::Result<Self> {
        use rayon::prelude::*;
        let t = std::time::Instant::now();
        check_layer(dir)?;
        let tr = tetra48_transcoder()?;
        let (h, i) = (cfg.hidden_size, cfg.intermediate_size);
        let read = |e: usize| -> anyhow::Result<(CpuExpert, u64, u64)> {
            let em = read_expert(&tr, dir, layer, e, cfg)?;
            let rot_in = Arc::new(Rotation::new(h, rotation_seed(layer, e, 0)));
            let rot_down = Arc::new(Rotation::new(i, rotation_seed(layer, e, 1)));
            let rbytes = rotation_bytes(&rot_in) + rotation_bytes(&rot_down);
            let lat = |m, r: &Arc<Rotation>| lattice_of(m, r.clone(), tables, weights).map_err(anyhow::Error::msg);
            let x = CpuExpert { gate: lat(em.gate, &rot_in)?, up: lat(em.up, &rot_in)?, down: lat(em.down, &rot_down)? };
            Ok((x, em.file_bytes, rbytes))
        };
        let read: Vec<anyhow::Result<(CpuExpert, u64, u64)>> =
            crate::tetra_cpu::pool().install(|| (0..cfg.n_routed_experts).into_par_iter().map(read).collect());
        let (mut experts, mut file_bytes, mut rotation_bytes) = (Vec::with_capacity(read.len()), 0u64, 0u64);
        for r in read {
            let (x, f, rb) = r?;
            experts.push(x);
            file_bytes += f;
            rotation_bytes += rb;
        }
        let mut out = Self::new(layer, experts, h, i).map_err(anyhow::Error::msg)?;
        out.stats.file_bytes = file_bytes;
        out.stats.rotation_bytes = rotation_bytes;
        out.stats.seconds = t.elapsed().as_secs_f64();
        Ok(out)
    }

    pub fn count(&self) -> usize {
        self.experts.len()
    }

    pub fn layer(&self) -> usize {
        self.layer
    }

    /// The pairs `(expert, weight)` of each token, ascending in expert, the
    /// routing checked against this layer.
    fn checked(&self, routing: &Routing, t: usize) -> Result<Vec<Vec<(usize, f32)>>> {
        let l = self.layer;
        if routing.indices.len() != t || routing.weights.len() != t {
            candle_core::bail!(
                "layer {l} cpu experts: a routing of {} tokens ({} weight rows) for {t} rows",
                routing.indices.len(),
                routing.weights.len()
            );
        }
        let mut out = Vec::with_capacity(t);
        for (tok, (idx, w)) in routing.indices.iter().zip(&routing.weights).enumerate() {
            if idx.len() != w.len() {
                candle_core::bail!("layer {l} cpu experts: token {tok} has {} experts and {} weights", idx.len(), w.len());
            }
            let mut pairs: Vec<(usize, f32)> = idx.iter().map(|&e| e as usize).zip(w.iter().copied()).collect();
            pairs.sort_by_key(|p| p.0);
            for (k, &(e, _)) in pairs.iter().enumerate() {
                if e >= self.experts.len() {
                    candle_core::bail!("layer {l} cpu experts: token {tok} routes expert {e} of {}", self.experts.len());
                }
                if k > 0 && pairs[k - 1].0 == e {
                    candle_core::bail!("layer {l} cpu experts: token {tok} routes expert {e} twice");
                }
            }
            out.push(pairs);
        }
        Ok(out)
    }

    /// The MoE output of `rows` rows of `xs` (`[rows, H]`, all routed to
    /// expert `e`), unweighted: `[rows, H]`.
    ///
    /// One expert at a time, each matvec on the CPU pool. The loop over
    /// experts is serial in this version: a parallel one would share the pool
    /// with the matvecs inside it.
    fn expert_rows(&self, e: usize, xs: &[f32], rows: usize, cfg: &InklingConfig) -> Vec<f32> {
        let x = &self.experts[e];
        let xp = phases::timed(phases::ROTATE_KERNEL, || x.gate.prepare_vec(xs, rows));
        let g = x.gate.matvec_vec(&xp, rows);
        let u = x.up.matvec_vec(&xp, rows);
        let act = activation(&g, &u, cfg);
        let ap = phases::timed(phases::ROTATE_KERNEL, || x.down.prepare_vec(&act, rows));
        x.down.matvec_vec(&ap, rows)
    }
}

/// `softcap(g, gate_cap) * sigmoid(g) * softcap(u, up_cap)`, element by
/// element in f32, the products in the order of `experts_forward` (line 771
/// of the reference): `softcap(x, c) = tanh(x / c) * c`.
pub fn activation(g: &[f32], u: &[f32], cfg: &InklingConfig) -> Vec<f32> {
    let (gc, uc) = (cfg.mlp_gate_softcap as f32, cfg.mlp_up_softcap as f32);
    g.iter()
        .zip(u)
        .map(|(&g, &u)| {
            let sig = 1.0 / (1.0 + (-g).exp());
            (g / gc).tanh() * gc * sig * ((u / uc).tanh() * uc)
        })
        .collect()
}

impl MoeSource for CpuExperts {
    fn moe(&self, x: &Tensor, routing: &Routing, cfg: &InklingConfig) -> Result<Tensor> {
        phases::timed(phases::MOE, || {
            let l = self.layer;
            if !x.device().is_cpu() {
                candle_core::bail!("layer {l} cpu experts: the input is on {:?}, not the host", x.device());
            }
            if !matches!(x.dtype(), DType::F32 | DType::F16 | DType::BF16) {
                candle_core::bail!("layer {l} cpu experts: an input of {:?}, expected a float", x.dtype());
            }
            let (t, h) = match x.dims() {
                &[t, h] => (t, h),
                d => candle_core::bail!("layer {l} cpu experts: an input of dims {d:?}, expected [T, {}]", self.hidden),
            };
            if h != self.hidden || cfg.hidden_size != self.hidden || cfg.intermediate_size != self.inter {
                candle_core::bail!(
                    "layer {l} cpu experts: rows of {h} and a config of {}x{} for experts of {}x{}",
                    cfg.hidden_size,
                    cfg.intermediate_size,
                    self.hidden,
                    self.inter
                );
            }
            let routes = self.checked(routing, t)?;
            let xs = x.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
            let mut out = vec![0f32; t * h];
            if t == 1 {
                // One row: each routed expert on `x` itself, in ascending order.
                for &(e, w) in &routes[0] {
                    let y = self.expert_rows(e, &xs, 1, cfg);
                    for (o, v) in out.iter_mut().zip(&y) {
                        *o += v * w;
                    }
                }
            } else {
                // A prompt: the rows routed to each expert in one batched
                // matvec, then each added to its token, experts ascending.
                let mut per_expert: Vec<Vec<(usize, f32)>> = vec![vec![]; self.experts.len()];
                for (tok, pairs) in routes.iter().enumerate() {
                    for &(e, w) in pairs {
                        per_expert[e].push((tok, w));
                    }
                }
                let mut buf = Vec::new();
                for (e, toks) in per_expert.iter().enumerate() {
                    if toks.is_empty() {
                        continue;
                    }
                    buf.clear();
                    for &(tok, _) in toks {
                        buf.extend_from_slice(&xs[tok * h..(tok + 1) * h]);
                    }
                    let y = self.expert_rows(e, &buf, toks.len(), cfg);
                    for (&(tok, w), yr) in toks.iter().zip(y.chunks(h)) {
                        for (o, v) in out[tok * h..(tok + 1) * h].iter_mut().zip(yr) {
                            *o += v * w;
                        }
                    }
                }
            }
            Tensor::from_vec(out, (t, h), &Device::Cpu)
        })
    }
}

/// Hold on the CPU the experts of every layer of `layers_dir` (`D/layers`)
/// whose `VERIFIED` file exists; the others keep their native experts.
/// Returns the layers set and what each load cost. The model must live on
/// the CPU. `LLVQ_CPU_WEIGHTS` chooses packed words or nibbles.
pub fn make_resident(model: &mut Inkling, layers_dir: &Path) -> anyhow::Result<Vec<(usize, LoadStats)>> {
    anyhow::ensure!(model.device().is_cpu(), "the CPU experts serve a model on the CPU, this one is on {:?}", model.device());
    crate::tetra_cpu::threads().map_err(anyhow::Error::msg)?;
    let weights = CpuWeights::from_env().map_err(anyhow::Error::msg)?;
    let cfg = model.cfg.clone();
    let tables = Arc::new(Tables::default());
    let mut out = vec![];
    for l in expert_codes::verified_layers(layers_dir, cfg.num_hidden_layers) {
        let dir = layer_dir(layers_dir, l);
        let r = CpuExperts::load(&dir, l, &cfg, &tables, weights).with_context(|| format!("layer {l} on the CPU"))?;
        out.push((l, r.stats.clone()));
        model.set_experts(l, LayerExperts::Source(Box::new(r)));
    }
    Ok(out)
}

/// The served file set on the CPU, loaded as the runtime serves it: the
/// attention matrices in BF16 through the CPU kernels ([`AttnKind::Cpu`]),
/// the glue on candle in f32, the int4 head from
/// `tables/unembed.q4g64.llvqraw`, the int4 embedding from
/// `tables/embed.q4g64.llvqraw` kept packed, the experts of every verified
/// layer from their codes, and the prompt bounds of the runtime's CUDA stack.
/// Returns the model and the lines its load reports.
pub fn load_served(config: &Path, checkpoint: &Path, layers: &Path, tables_dir: &Path) -> anyhow::Result<(Inkling, Vec<String>)> {
    load_served_with(config, checkpoint, layers, tables_dir, AttnKind::Cpu)
}

/// [`load_served`] with the attention `attn`: [`AttnKind::Cpu`] is the
/// served one, [`AttnKind::Candle`] the f32 reference its gate reads.
pub fn load_served_with(
    config: &Path,
    checkpoint: &Path,
    layers: &Path,
    tables_dir: &Path,
    attn: AttnKind,
) -> anyhow::Result<(Inkling, Vec<String>)> {
    use super::{tables, LoadOptions};
    let dev = Device::Cpu;
    let mut lines = Vec::new();
    let opts = LoadOptions { int4_head: Some(tables_dir.join(tables::UNEMBED_FILE)) };
    let mut model = Inkling::load_with_attn(config, checkpoint, &dev, &opts, attn)?;
    let t = tables::load_embed(tables_dir, &model.cfg, true)?;
    lines.push(format!("tables: embed {}, unembed {}", &t.embed_sha256[..16], &t.unembed_sha256[..16]));
    let embed_bytes = t.embed.bytes();
    model.embed_table = Some(t.embed);
    let stats = make_resident(&mut model, layers)?;
    model.set_prefill_piece(PROMPT_PIECE);
    model.set_attn_chunk(SCORE_ROWS);
    model.set_kv_window(Some(KV_SLACK));
    let sum = |f: fn(&LoadStats) -> u64| stats.iter().map(|(_, s)| f(s)).sum::<u64>();
    let (attn_f32, attn_bf16) = model.attention_matrix_bytes();
    let head_bytes = model.out_proj.as_ref().map_or(0, |p| p.device_bytes());
    let gb = |b: u64| b as f64 / 1e9;
    lines.push(format!(
        "cpu: {} SIMD on {} threads | weights {} | experts {:.2} GB + rotations {:.3} GB on {} layers, read in {:.1} s",
        crate::tetra_cpu::Isa::detect().name(),
        crate::tetra_cpu::pool().current_num_threads(),
        CpuWeights::from_env().map_err(anyhow::Error::msg)?.name(),
        gb(sum(|s| s.matrix_bytes)),
        gb(sum(|s| s.rotation_bytes)),
        stats.len(),
        stats.iter().map(|(_, s)| s.seconds).sum::<f64>()
    ));
    lines.push(format!(
        "cpu: attention {} | BF16 {:.2} GB + f32 {:.2} GB, int4 embedding {:.2} GB, int4 head {:.2} GB",
        model.attn().name(),
        gb(attn_bf16 as u64),
        gb(attn_f32 as u64),
        gb(embed_bytes),
        gb(head_bytes)
    ));
    lines.push(format!(
        "prompt: pieces of {PROMPT_PIECE} tokens, scores in chunks of {SCORE_ROWS} rows, sliding slots bounded with a slack of {KV_SLACK}"
    ));
    Ok((model, lines))
}
