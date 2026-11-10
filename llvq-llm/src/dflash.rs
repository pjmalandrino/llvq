//! The DFlash block drafter of Qwen3-4B, `z-lab/Qwen3-4B-DFlash-b16` at a
//! pinned revision (MIT), in candle.
//!
//! Ported from `ops/dflash_alpha.py` (the MLX rewrite of the drafter's
//! modeling code that measured tau on the sealed 4B on 2026-11-09). Nothing
//! here names a device type: the same code runs on CPU, Metal and CUDA.
//!
//! ## The model
//!
//! Five Qwen3-shaped layers (hidden 2,560, 32 query heads and 8 KV heads of
//! 128, MLP 9,728, RMSNorm eps 1e-6, RoPE theta 1e6, rotate-half form), plus
//! `fc` [2,560 x 12,800] and `hidden_norm` over the target's states, and a
//! final `norm`. No embedding and no head: it borrows the target's.
//!
//! * Context. At pending position m the drafter reads the target's states at
//!   positions 0..m-1: the inputs of target layers 2, 10, 18, 26, 34 (HF
//!   `hidden_states[i + 1]` for `target_layer_ids` 1, 9, 17, 25, 33),
//!   concatenated in that order into [m, 12,800]. `c = hidden_norm(fc(taps))`
//!   is projected by each layer's `k_proj` and `v_proj` only (then `k_norm`
//!   and RoPE at the absolute position). Neither `input_layernorm` nor any
//!   query touches `c`.
//! * Block. The 16 rows `[x_m, MASK x 15]` (mask id 151669) at positions
//!   m..m+15, embedded by the target's table, go through the five layers.
//!   Each row's queries attend to every context key and to all 16 block keys,
//!   with no causal mask. The final norm of rows 1..15 goes through the
//!   target's head: the 15 logits rows propose x[m+1..m+15].
//!
//! ## What is cached between rounds
//!
//! The context keys and values depend on one position's taps and on its
//! position only, never on the block. [`DraftContext`] holds them, per layer
//! [8, cap, 128] for keys (already normed and rotated) and values. A round
//! costs:
//!
//! * [`DFlashDrafter::extend`] over the n newly accepted positions: `fc`,
//!   `hidden_norm`, and five `k_proj` + `v_proj` (no layer pass);
//! * [`DFlashDrafter::block_logits`]: five full layer passes over 16 rows,
//!   attention over m + 16 keys, the final norm and the head over 15 rows.
//!
//! The context only grows by accepted positions, whose target states are
//! final, so it never needs a rollback. [`DraftContext::truncate`] exists for
//! a caller that appended speculatively.
//!
//! ## Readouts
//!
//! [`DraftOut`] carries the greedy chain (argmax of each of the 15 rows), the
//! top 4 candidates a row with their log probabilities, and [`draft_tree`]
//! builds the tree of the reference (`tree_accepts` of `ops/dflash_alpha.py`,
//! DDTree's construction): a node is a rank path, scored by its summed log
//! probability, popped best first; the first N pops are closed under
//! ancestors.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context as _};
use candle_core::{DType, Device, Tensor, D};

/// The drafter repository and the pinned revision the reference measured.
pub const REPO: &str = "z-lab/Qwen3-4B-DFlash-b16";
pub const REVISION: &str = "b74e3a329c4d963783143b1e970d95b002be72bd";
/// Rows of a block: the last accepted token and 15 masks.
pub const BLOCK: usize = 16;
/// Drafted positions a block.
pub const DRAFTS: usize = BLOCK - 1;
pub const MASK_ID: u32 = 151669;
/// Target layers whose INPUT the drafter reads, in concatenation order.
pub const TAP_LAYERS: [usize; 5] = [2, 10, 18, 26, 34];
/// Candidates a position the tree reads.
pub const TREE_TOPK: usize = 4;

/// The drafter's shape, read from its `config.json` and checked against what
/// this port implements.
#[derive(Debug, Clone)]
pub struct DFlashConfig {
    pub hidden: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub intermediate: usize,
    pub layers: usize,
    pub eps: f64,
    pub theta: f64,
    pub vocab: usize,
}

impl DFlashConfig {
    pub fn from_json(text: &str) -> anyhow::Result<Self> {
        let v: serde_json::Value = serde_json::from_str(text)?;
        let u = |k: &str| -> anyhow::Result<usize> {
            v[k].as_u64().map(|x| x as usize).with_context(|| format!("config.json: no integer `{k}`"))
        };
        let f = |k: &str| -> anyhow::Result<f64> {
            v[k].as_f64().with_context(|| format!("config.json: no number `{k}`"))
        };
        let cfg = Self {
            hidden: u("hidden_size")?,
            heads: u("num_attention_heads")?,
            kv_heads: u("num_key_value_heads")?,
            head_dim: u("head_dim")?,
            intermediate: u("intermediate_size")?,
            layers: u("num_hidden_layers")?,
            eps: f("rms_norm_eps")?,
            theta: f("rope_theta")?,
            vocab: u("vocab_size")?,
        };
        if u("block_size")? != BLOCK {
            bail!("block_size {} is not {BLOCK}", u("block_size")?);
        }
        let d = &v["dflash_config"];
        if d["mask_token_id"].as_u64() != Some(MASK_ID as u64) {
            bail!("mask_token_id {:?} is not {MASK_ID}", d["mask_token_id"]);
        }
        let ids: Vec<u64> = d["target_layer_ids"]
            .as_array()
            .context("no target_layer_ids")?
            .iter()
            .filter_map(|x| x.as_u64())
            .collect();
        let want: Vec<u64> = TAP_LAYERS.iter().map(|&l| l as u64 - 1).collect();
        if ids != want {
            bail!("target_layer_ids {ids:?}, this port reads {want:?}");
        }
        if !v["rope_scaling"].is_null() || v["attention_bias"].as_bool() != Some(false) {
            bail!("rope_scaling or attention_bias set: not implemented");
        }
        if !cfg.heads.is_multiple_of(cfg.kv_heads) || !cfg.head_dim.is_multiple_of(2) {
            bail!("heads {} / kv {} / head_dim {}", cfg.heads, cfg.kv_heads, cfg.head_dim);
        }
        Ok(cfg)
    }
}

/// The snapshot directory of the pinned revision in the local Hugging Face
/// cache (`HF_HUB_CACHE`, else `HF_HOME/hub`, else `~/.cache/huggingface/hub`),
/// if its weights are there. Never downloads.
pub fn cached_snapshot() -> Option<PathBuf> {
    let hub = std::env::var_os("HF_HUB_CACHE").map(PathBuf::from).or_else(|| {
        std::env::var_os("HF_HOME")
            .map(|h| PathBuf::from(h).join("hub"))
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache/huggingface/hub")))
    })?;
    let dir = hub.join(format!("models--{}", REPO.replace('/', "--"))).join("snapshots").join(REVISION);
    dir.join("model.safetensors").exists().then_some(dir)
}

/// The target's tied embedding table and head, as the drafter needs them.
///
/// The drafter embeds 16 ids a round (two distinct: the last token and the
/// mask) and applies the head to 15 rows. A bench that holds the table int4
/// on the card implements this over its own kernels; [`DenseVocab`] is the
/// reference form.
pub trait TargetVocab {
    /// `ids` u32 [n] -> [n, hidden], any float dtype.
    fn embed(&self, ids: &Tensor) -> candle_core::Result<Tensor>;
    /// `hidden` [n, hidden] in the drafter's dtype -> logits [n, vocab], f32.
    fn logits(&self, hidden: &Tensor) -> candle_core::Result<Tensor>;
}

/// A dense tied table [vocab, hidden] on the drafter's device.
pub struct DenseVocab {
    pub table: Tensor,
}

impl TargetVocab for DenseVocab {
    fn embed(&self, ids: &Tensor) -> candle_core::Result<Tensor> {
        self.table.index_select(ids, 0)
    }
    fn logits(&self, hidden: &Tensor) -> candle_core::Result<Tensor> {
        hidden.to_dtype(self.table.dtype())?.matmul(&self.table.t()?)?.to_dtype(DType::F32)
    }
}

/// MLX affine int4 dequantization of the packed table (`hfpack` layout:
/// `qweight` u8 [V, H/2], low nibble first; `scales`, `biases` f16 [V, H/g]):
/// `w = q * scale + bias`, computed in f32 and rounded once to f16, as
/// `mx.dequantize` returns f16 for f16 scales (the reference then cast it to
/// f32).
pub fn dequant_q4(qweight: &[u8], scales: &[half::f16], biases: &[half::f16], hidden: usize) -> Vec<f32> {
    let rows = qweight.len() * 2 / hidden;
    let groups = scales.len() / rows;
    let g = hidden / groups;
    let mut out = vec![0f32; rows * hidden];
    for r in 0..rows {
        let q = &qweight[r * hidden / 2..(r + 1) * hidden / 2];
        for j in 0..hidden {
            let nib = (q[j / 2] >> ((j % 2) * 4)) & 0xf;
            let s = scales[r * groups + j / g].to_f32();
            let b = biases[r * groups + j / g].to_f32();
            out[r * hidden + j] = half::f16::from_f32(nib as f32 * s + b).to_f32();
        }
    }
    out
}

/// The packed bytes, scales and biases (binary16 bits), vocabulary and
/// hidden width of an int4 g64 table ([`sealed_q4_table`]).
pub type Q4Table = (Vec<u8>, Vec<u16>, Vec<u16>, usize, usize);

/// The sealed file's tied embedding table (`model.embed_tokens.weight`,
/// int4 g64): the packed bytes (two values a byte, the low nibble first, the
/// `hfpack` `qweight` layout), the scales and biases as binary16 bits, the
/// vocabulary and the hidden width. Reads the file through its records once;
/// loads no model.
pub fn sealed_q4_table(path: &Path) -> anyhow::Result<Q4Table> {
    use std::io::Read;
    let f = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut r = std::io::BufReader::with_capacity(1 << 20, f);
    let head = llvq_artifact::read_header(&mut r)?;
    anyhow::ensure!(head.is_self_contained(), "{} is not a sealed file", path.display());
    for _ in 0..head.matrices {
        llvq_artifact::read_record(&mut r, head.version)?;
    }
    let mut b4 = [0u8; 4];
    r.read_exact(&mut b4)?;
    for _ in 0..u32::from_le_bytes(b4) {
        let t = llvq_artifact::read_raw(&mut r, head.version)?;
        if t.name != "model.embed_tokens.weight" {
            continue;
        }
        let (vocab, hidden) = match t.dims[..] {
            [v, h] => (v, h),
            _ => anyhow::bail!("{}: dims {:?}", t.name, t.dims),
        };
        return match t.data {
            llvq_artifact::RawData::Quant(q) if q.bits == 4 && hidden % q.group == 0 => Ok((q.packed, q.scales, q.biases, vocab, hidden)),
            llvq_artifact::RawData::Quant(q) => anyhow::bail!("{}: int{} g{}, the drafter's table is int4", t.name, q.bits, q.group),
            llvq_artifact::RawData::F16(_) => anyhow::bail!("{}: f16, not the served int4 table", t.name),
        };
    }
    anyhow::bail!("{}: no model.embed_tokens.weight", path.display())
}

/// [`dequant_q4`] into binary16: the same value a weight, `f16(q * scale +
/// bias)` with the product and the sum in f32, without the f32 copy.
pub fn dequant_q4_f16(packed: &[u8], scales: &[u16], biases: &[u16], hidden: usize) -> Vec<half::f16> {
    let rows = packed.len() * 2 / hidden;
    let groups = scales.len() / rows;
    let g = hidden / groups;
    let mut out = Vec::with_capacity(rows * hidden);
    for r in 0..rows {
        let q = &packed[r * hidden / 2..(r + 1) * hidden / 2];
        for j in 0..hidden {
            let nib = (q[j / 2] >> ((j % 2) * 4)) & 0xf;
            let s = half::f16::from_bits(scales[r * groups + j / g]).to_f32();
            let b = half::f16::from_bits(biases[r * groups + j / g]).to_f32();
            out.push(half::f16::from_f32(nib as f32 * s + b));
        }
    }
    out
}

/// The sealed file's own tied table, dequantized once on the host to
/// binary16 and moved to `device` in `dtype`: the card's [`TargetVocab`]
/// (0.78 GB in f16 at the 4B), built from what the bench already holds.
pub fn vocab_from_sealed(path: &Path, device: &Device, dtype: DType) -> anyhow::Result<DenseVocab> {
    let (packed, scales, biases, vocab, hidden) = sealed_q4_table(path)?;
    let w = dequant_q4_f16(&packed, &scales, &biases, hidden);
    drop(packed);
    let table = Tensor::from_vec(w, (vocab, hidden), &Device::Cpu)?.to_dtype(dtype)?.to_device(device)?;
    Ok(DenseVocab { table })
}

/// The tied table of a packed directory (`hfpack` output, e.g.
/// `~/q4b-hf-2026-09-28`), dequantized to `dtype` on `device`. This is the
/// table the reference fed the drafter (the MLX door's int4 g64 table).
pub fn vocab_from_packed(dir: &Path, device: &Device, dtype: DType) -> anyhow::Result<DenseVocab> {
    let file = dir.join("model.safetensors");
    // SAFETY: a read-only mmap of a file nothing in this process writes.
    let st = unsafe { candle_core::safetensors::MmapedSafetensors::new(&file) }
        .with_context(|| format!("mmap {}", file.display()))?;
    let get = |k: &str| st.load(k, &Device::Cpu).with_context(|| format!("{}: no `{k}`", file.display()));
    let q = get("model.embed_tokens.qweight")?;
    let s = get("model.embed_tokens.scales")?;
    let b = get("model.embed_tokens.biases")?;
    let (v, half_h) = q.dims2()?;
    let hidden = half_h * 2;
    let w = dequant_q4(
        &q.flatten_all()?.to_vec1::<u8>()?,
        &s.flatten_all()?.to_vec1::<half::f16>()?,
        &b.flatten_all()?.to_vec1::<half::f16>()?,
        hidden,
    );
    let table = Tensor::from_vec(w, (v, hidden), &Device::Cpu)?.to_dtype(dtype)?.to_device(device)?;
    Ok(DenseVocab { table })
}

struct Layer {
    q: Tensor,
    k: Tensor,
    v: Tensor,
    o: Tensor,
    q_norm: Tensor,
    k_norm: Tensor,
    ln1: Tensor,
    ln2: Tensor,
    gate: Tensor,
    up: Tensor,
    down: Tensor,
}

/// `x` [n, in] times `w` [out, in] transposed.
fn lin(x: &Tensor, w: &Tensor) -> candle_core::Result<Tensor> {
    x.matmul(&w.t()?)
}

/// The drafter's weights on one device, in one dtype, except `fc` and
/// `hidden_norm`, always f32.
///
/// Why f32 there: the taps of target layer 34 reach 15,352 on the fixture's
/// chat prompt, and `fc` sums 12,800 of them; its output passed 65,504 on 5
/// values of 64 positions in binary16 (*measured* on Metal, 2026-11-10).
/// `hidden_norm` turns each such row into NaN, which the cached context
/// carries into every block: the f16 drafter proposed token 151551 fifteen
/// times on every round, and the card accepted 0.00 to 0.02 drafts a pass
/// (SP2). bf16 has the range; f16 does not. The output of `hidden_norm` is
/// bounded, so the rest runs in the drafter's dtype.
pub struct DFlashDrafter {
    pub cfg: DFlashConfig,
    fc: Tensor,
    hidden_norm: Tensor,
    norm: Tensor,
    layers: Vec<Layer>,
    inv_freq: Tensor,
    device: Device,
    dtype: DType,
}

/// The drafter's cached context: per layer the keys (normed, rotated) and
/// values of positions 0..len, [kv_heads, cap, head_dim].
pub struct DraftContext {
    k: Vec<Tensor>,
    v: Vec<Tensor>,
    len: usize,
    cap: usize,
}

impl DraftContext {
    /// Positions held: the next block starts at position `len()`.
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    /// Forget positions `n..`.
    pub fn truncate(&mut self, n: usize) {
        self.len = self.len.min(n);
    }
    /// Bytes the cache holds at its current capacity.
    pub fn bytes(&self) -> usize {
        self.k.iter().chain(&self.v).map(|t| t.elem_count() * t.dtype().size_in_bytes()).sum()
    }
}

/// One round's readouts.
#[derive(Debug, Clone)]
pub struct DraftOut {
    /// Argmax of each drafted row: the greedy chain of 15.
    pub chain: Vec<u32>,
    /// Per row the top [`TREE_TOPK`] ids, best first, and their log probabilities.
    pub top_ids: Vec<[u32; TREE_TOPK]>,
    pub top_logp: Vec<[f32; TREE_TOPK]>,
    /// Per row the logsumexp of the logits.
    pub lse: Vec<f32>,
}

impl DFlashDrafter {
    /// Load `config.json` and `model.safetensors` from `dir` (a snapshot
    /// directory, see [`cached_snapshot`]).
    pub fn load(dir: &Path, device: &Device, dtype: DType) -> anyhow::Result<Self> {
        let cfg = DFlashConfig::from_json(
            &std::fs::read_to_string(dir.join("config.json")).with_context(|| format!("{}", dir.display()))?,
        )?;
        let mut w: HashMap<String, Tensor> =
            candle_core::safetensors::load(dir.join("model.safetensors"), &Device::Cpu)?;
        let expected = 3 + 11 * cfg.layers;
        if w.len() != expected {
            bail!("{} tensors in the drafter, expected {expected}", w.len());
        }
        let mut take = |k: &str| -> anyhow::Result<Tensor> {
            let t = w.remove(k).with_context(|| format!("drafter: no `{k}`"))?;
            Ok(t.to_dtype(dtype)?.to_device(device)?)
        };
        let fc = take("fc.weight")?.to_dtype(DType::F32)?;
        if fc.dims() != [cfg.hidden, TAP_LAYERS.len() * cfg.hidden] {
            bail!("fc.weight {:?}", fc.dims());
        }
        let hidden_norm = take("hidden_norm.weight")?.to_dtype(DType::F32)?;
        let norm = take("norm.weight")?;
        let mut layers = Vec::with_capacity(cfg.layers);
        for i in 0..cfg.layers {
            let mut t = |n: &str| take(&format!("layers.{i}.{n}.weight"));
            layers.push(Layer {
                q: t("self_attn.q_proj")?,
                k: t("self_attn.k_proj")?,
                v: t("self_attn.v_proj")?,
                o: t("self_attn.o_proj")?,
                q_norm: t("self_attn.q_norm")?,
                k_norm: t("self_attn.k_norm")?,
                ln1: t("input_layernorm")?,
                ln2: t("post_attention_layernorm")?,
                gate: t("mlp.gate_proj")?,
                up: t("mlp.up_proj")?,
                down: t("mlp.down_proj")?,
            });
        }
        // As the reference: 1 / theta^(i / d) for even i, in f32.
        let half = cfg.head_dim / 2;
        let inv: Vec<f32> = (0..half)
            .map(|i| 1.0f32 / (cfg.theta as f32).powf((2 * i) as f32 / cfg.head_dim as f32))
            .collect();
        let inv_freq = Tensor::from_vec(inv, (1, half), device)?;
        Ok(Self { cfg, fc, hidden_norm, norm, layers, inv_freq, device: device.clone(), dtype })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Bytes of the drafter's weights in its dtype.
    pub fn weight_bytes(&self) -> usize {
        let mut n = self.norm.elem_count();
        for l in &self.layers {
            for t in [&l.q, &l.k, &l.v, &l.o, &l.q_norm, &l.k_norm, &l.ln1, &l.ln2, &l.gate, &l.up, &l.down] {
                n += t.elem_count();
            }
        }
        n * self.dtype.size_in_bytes() + (self.fc.elem_count() + self.hidden_norm.elem_count()) * 4
    }

    /// An empty context with room for `cap` positions (it grows by doubling).
    pub fn new_context(&self, cap: usize) -> anyhow::Result<DraftContext> {
        let cap = cap.max(BLOCK);
        let shape = (self.cfg.kv_heads, cap, self.cfg.head_dim);
        let z = || Tensor::zeros(shape, self.dtype, &self.device);
        let mut k = Vec::new();
        let mut v = Vec::new();
        for _ in 0..self.cfg.layers {
            k.push(z()?);
            v.push(z()?);
        }
        Ok(DraftContext { k, v, len: 0, cap })
    }

    fn rms(&self, x: &Tensor, w: &Tensor) -> candle_core::Result<Tensor> {
        candle_nn::ops::rms_norm(&x.contiguous()?, w, self.cfg.eps as f32)
    }

    /// cos and sin [n, head_dim / 2] at positions start..start+n.
    fn rope_tables(&self, start: usize, n: usize) -> candle_core::Result<(Tensor, Tensor)> {
        let pos: Vec<f32> = (start..start + n).map(|p| p as f32).collect();
        let pos = Tensor::from_vec(pos, (n, 1), &self.device)?;
        let ang = pos.broadcast_mul(&self.inv_freq)?;
        Ok((ang.cos()?.to_dtype(self.dtype)?, ang.sin()?.to_dtype(self.dtype)?))
    }

    /// `x` [heads, n, d] contiguous, rotated.
    fn rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> candle_core::Result<Tensor> {
        candle_nn::rotary_emb::rope(&x.unsqueeze(0)?, cos, sin)?.squeeze(0)
    }

    /// Append the target's tap states of the next `n` positions, [n, 5 x hidden]
    /// (layers 2, 10, 18, 26, 34 in that order), any float dtype.
    pub fn extend(&self, ctx: &mut DraftContext, taps: &Tensor) -> anyhow::Result<()> {
        let (n, w) = taps.dims2()?;
        if w != TAP_LAYERS.len() * self.cfg.hidden {
            bail!("taps width {w}, expected {}", TAP_LAYERS.len() * self.cfg.hidden);
        }
        if n == 0 {
            return Ok(());
        }
        let (kvh, dh) = (self.cfg.kv_heads, self.cfg.head_dim);
        if ctx.len + n > ctx.cap {
            let cap = (ctx.len + n).next_power_of_two().max(2 * ctx.cap);
            for t in ctx.k.iter_mut().chain(ctx.v.iter_mut()) {
                let g = Tensor::zeros((kvh, cap, dh), self.dtype, &self.device)?;
                if ctx.len > 0 {
                    g.slice_set(&t.narrow(1, 0, ctx.len)?.contiguous()?, 1, 0)?;
                }
                *t = g;
            }
            ctx.cap = cap;
        }
        // fc and hidden_norm in f32 whatever the dtype: see the struct's note
        let taps = taps.to_device(&self.device)?.to_dtype(DType::F32)?;
        let c = self.rms(&lin(&taps, &self.fc)?, &self.hidden_norm)?.to_dtype(self.dtype)?;
        let (cos, sin) = self.rope_tables(ctx.len, n)?;
        for (li, l) in self.layers.iter().enumerate() {
            let k = self.rms(&lin(&c, &l.k)?.reshape((n, kvh, dh))?, &l.k_norm)?;
            let k = Self::rope(&k.transpose(0, 1)?.contiguous()?, &cos, &sin)?;
            let v = lin(&c, &l.v)?.reshape((n, kvh, dh))?.transpose(0, 1)?.contiguous()?;
            ctx.k[li].slice_set(&k, 1, ctx.len)?;
            ctx.v[li].slice_set(&v, 1, ctx.len)?;
        }
        ctx.len += n;
        Ok(())
    }

    /// The final normed hidden states of the 15 drafted rows, [15, hidden],
    /// for the block `[last, MASK x 15]` at positions ctx.len()..ctx.len()+15.
    pub fn block_hidden(&self, ctx: &DraftContext, last: u32, vocab: &dyn TargetVocab) -> anyhow::Result<Tensor> {
        let cfg = &self.cfg;
        let (hq, kvh, dh) = (cfg.heads, cfg.kv_heads, cfg.head_dim);
        let rep = hq / kvh;
        let m = ctx.len;
        let mut ids = vec![MASK_ID; BLOCK];
        ids[0] = last;
        let ids = Tensor::from_vec(ids, BLOCK, &self.device)?;
        let mut h = vocab.embed(&ids)?.to_dtype(self.dtype)?;
        let (cos, sin) = self.rope_tables(m, BLOCK)?;
        let scale = (dh as f64).powf(-0.5);
        for (li, l) in self.layers.iter().enumerate() {
            let hn = self.rms(&h, &l.ln1)?;
            let q = self.rms(&lin(&hn, &l.q)?.reshape((BLOCK, hq, dh))?, &l.q_norm)?;
            let q = Self::rope(&q.transpose(0, 1)?.contiguous()?, &cos, &sin)?;
            // Query head j reads KV head j / rep: group the rep heads of a KV head as rows.
            let q = q.reshape((kvh, rep * BLOCK, dh))?;
            let kb = self.rms(&lin(&hn, &l.k)?.reshape((BLOCK, kvh, dh))?, &l.k_norm)?;
            let kb = Self::rope(&kb.transpose(0, 1)?.contiguous()?, &cos, &sin)?;
            let vb = lin(&hn, &l.v)?.reshape((BLOCK, kvh, dh))?.transpose(0, 1)?.contiguous()?;
            let sb = (q.matmul(&kb.t()?)? * scale)?;
            let att = if m == 0 {
                candle_nn::ops::softmax_last_dim(&sb)?.matmul(&vb)?
            } else {
                let kc = ctx.k[li].narrow(1, 0, m)?;
                let vc = ctx.v[li].narrow(1, 0, m)?;
                let sc = (q.matmul(&kc.t()?)? * scale)?;
                let p = candle_nn::ops::softmax_last_dim(&Tensor::cat(&[&sc, &sb], 2)?)?;
                let pc = p.narrow(2, 0, m)?.contiguous()?;
                let pb = p.narrow(2, m, BLOCK)?.contiguous()?;
                (pc.matmul(&vc)? + pb.matmul(&vb)?)?
            };
            let att = att.reshape((hq, BLOCK, dh))?.transpose(0, 1)?.reshape((BLOCK, hq * dh))?;
            h = (h + lin(&att, &l.o)?)?;
            let m2 = self.rms(&h, &l.ln2)?;
            let mlp = lin(&(candle_nn::ops::silu(&lin(&m2, &l.gate)?)? * lin(&m2, &l.up)?)?, &l.down)?;
            h = (h + mlp)?;
        }
        Ok(self.rms(&h, &self.norm)?.narrow(0, 1, DRAFTS)?)
    }

    /// The logits of the 15 drafted rows, [15, vocab] f32.
    pub fn block_logits(&self, ctx: &DraftContext, last: u32, vocab: &dyn TargetVocab) -> anyhow::Result<Tensor> {
        Ok(vocab.logits(&self.block_hidden(ctx, last, vocab)?)?)
    }

    /// One round: the logits of the block at ctx.len(), read out on the device.
    pub fn draft(&self, ctx: &DraftContext, last: u32, vocab: &dyn TargetVocab) -> anyhow::Result<DraftOut> {
        readout(&self.block_logits(ctx, last, vocab)?)
    }
}

/// Argmax, top [`TREE_TOPK`] and logsumexp of logits [rows, vocab], computed
/// on the logits' device; only rows x (2 x TOPK + 1) numbers leave it.
pub fn readout(logits: &Tensor) -> anyhow::Result<DraftOut> {
    let z = logits.to_dtype(DType::F32)?;
    let (rows, vocab) = z.dims2()?;
    let mx = z.max_keepdim(D::Minus1)?;
    let lse = (z.broadcast_sub(&mx)?.exp()?.sum_keepdim(D::Minus1)?.log()? + &mx)?;
    let ar = Tensor::arange(0u32, vocab as u32, z.device())?.unsqueeze(0)?;
    let ninf = Tensor::full(f32::NEG_INFINITY, (rows, vocab), z.device())?;
    let mut work = z.clone();
    let mut ids = Vec::with_capacity(TREE_TOPK);
    let mut vals = Vec::with_capacity(TREE_TOPK);
    for k in 0..TREE_TOPK {
        let i = work.argmax_keepdim(D::Minus1)?;
        vals.push(work.gather(&i, D::Minus1)?);
        if k + 1 < TREE_TOPK {
            work = ar.broadcast_eq(&i)?.where_cond(&ninf, &work)?;
        }
        ids.push(i);
    }
    let ids = Tensor::cat(&ids, 1)?.to_vec2::<u32>()?;
    let vals = Tensor::cat(&vals, 1)?.broadcast_sub(&lse)?.to_vec2::<f32>()?;
    let lse = lse.squeeze(1)?.to_vec1::<f32>()?;
    let mut out = DraftOut { chain: Vec::new(), top_ids: Vec::new(), top_logp: Vec::new(), lse };
    for r in 0..rows {
        let mut a = [0u32; TREE_TOPK];
        let mut b = [0f32; TREE_TOPK];
        a.copy_from_slice(&ids[r]);
        b.copy_from_slice(&vals[r]);
        out.chain.push(a[0]);
        out.top_ids.push(a);
        out.top_logp.push(b);
    }
    Ok(out)
}

/// f64 with a total order, for the heap.
struct Score(f64);
impl PartialEq for Score {
    fn eq(&self, o: &Self) -> bool {
        self.cmp(o).is_eq()
    }
}
impl Eq for Score {}
impl PartialOrd for Score {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Score {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&o.0)
    }
}

/// The tree of at most `nodes` nodes over one round's top candidates, in the
/// reference's pop order: `(token, parent)`, parent -1 for the root (the last
/// accepted token), else the index of an earlier node. Ties break on the rank
/// path, lexicographically, as Python's heap of `(-score, path)` tuples does.
pub fn draft_tree(out: &DraftOut, nodes: usize) -> Vec<(u32, i32)> {
    let depth = out.top_ids.len();
    let mut heap: BinaryHeap<Reverse<(Score, Vec<u8>)>> = BinaryHeap::new();
    for k in 0..TREE_TOPK {
        heap.push(Reverse((Score(-(out.top_logp[0][k] as f64)), vec![k as u8])));
    }
    let mut index: HashMap<Vec<u8>, i32> = HashMap::new();
    let mut tree = Vec::with_capacity(nodes);
    while tree.len() < nodes {
        let Some(Reverse((Score(s), path))) = heap.pop() else { break };
        let d = path.len();
        let parent = if d > 1 { index[&path[..d - 1]] } else { -1 };
        tree.push((out.top_ids[d - 1][path[d - 1] as usize], parent));
        index.insert(path.clone(), tree.len() as i32 - 1);
        if d < depth {
            for k in 0..TREE_TOPK {
                let mut p = path.clone();
                p.push(k as u8);
                heap.push(Reverse((Score(s - out.top_logp[d][k] as f64), p)));
            }
        }
    }
    tree
}

/// The equivalence of this port against the MLX reference, over a fixture
/// written by `ops/dflash_fixture.py`.
#[derive(Debug, Default, Clone)]
pub struct FixtureReport {
    pub rounds: usize,
    /// Rounds whose 15 argmax equal the reference's.
    pub chain_equal: usize,
    /// Rounds whose tree of 15 equals the reference's, token and parent.
    pub tree_equal: usize,
    /// Max over rounds of max |dz| / max |z_ref| over the reference's top 64 logits a row.
    pub logit_rel: f64,
    /// Max |d lse| / |lse_ref|.
    pub lse_rel: f64,
    /// Max over rounds of max |dh| / max |h_ref| over the final normed hidden states.
    pub hidden_rel: f64,
    /// Smallest top-1 minus top-2 logit gap of the reference over every row.
    pub min_gap: f64,
    /// Max |d| over the dequantized table rows the fixture holds.
    pub table_abs: f64,
}

impl std::fmt::Display for FixtureReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "rounds {}: chain equal {}/{}, tree equal {}/{}; logits rel {:.2e}, lse rel {:.2e}, hidden rel {:.2e}; \
             smallest reference top-1 gap {:.3}; table rows max abs {:.2e}",
            self.rounds,
            self.chain_equal,
            self.rounds,
            self.tree_equal,
            self.rounds,
            self.logit_rel,
            self.lse_rel,
            self.hidden_rel,
            self.min_gap,
            self.table_abs
        )
    }
}

/// Run every round of the fixture through `drafter`, with the context
/// extended incrementally round by round (the cache path, not one prefill).
pub fn check_fixture(path: &Path, drafter: &DFlashDrafter, vocab: &dyn TargetVocab) -> anyhow::Result<FixtureReport> {
    let fx = candle_core::safetensors::load(path, &Device::Cpu).with_context(|| format!("{}", path.display()))?;
    let get = |k: &str| fx.get(k).with_context(|| format!("fixture: no `{k}`"));
    let mut rep = FixtureReport { min_gap: f64::INFINITY, ..Default::default() };
    let rel = |a: &[f32], b: &[f32]| -> f64 {
        let num = a.iter().zip(b).map(|(x, y)| (x - y).abs() as f64).fold(0.0, f64::max);
        let den = b.iter().map(|y| y.abs() as f64).fold(0.0, f64::max);
        num / den.max(1e-30)
    };
    let tid = get("table.ids")?.to_dtype(DType::U32)?.to_device(drafter.device())?;
    let rows = vocab.embed(&tid)?.to_dtype(DType::F32)?.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
    let want = get("table.rows")?.flatten_all()?.to_vec1::<f32>()?;
    rep.table_abs = rows.iter().zip(&want).map(|(a, b)| (a - b).abs() as f64).fold(0.0, f64::max);
    let mut p = 0;
    while fx.contains_key(&format!("p{p}.ids")) {
        let ids = get(&format!("p{p}.ids"))?.to_dtype(DType::U32)?.to_vec1::<u32>()?;
        let taps = get(&format!("p{p}.taps"))?;
        let mut ctx = drafter.new_context(64)?;
        let mut r = 0;
        while fx.contains_key(&format!("p{p}.r{r}.m")) {
            let k = format!("p{p}.r{r}");
            let m = get(&format!("{k}.m"))?.to_dtype(DType::U32)?.to_vec1::<u32>()?[0] as usize;
            let len = ctx.len();
            drafter.extend(&mut ctx, &taps.narrow(0, len, m - len)?)?;
            let hidden = drafter.block_hidden(&ctx, ids[m], vocab)?;
            let z = vocab.logits(&hidden)?;
            let out = readout(&z)?;
            rep.rounds += 1;
            let h_ref = get(&format!("{k}.out"))?.flatten_all()?.to_vec1::<f32>()?;
            let h = hidden.to_dtype(DType::F32)?.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
            rep.hidden_rel = rep.hidden_rel.max(rel(&h, &h_ref));
            let top_ids = get(&format!("{k}.top_ids"))?.to_dtype(DType::U32)?.to_vec2::<u32>()?;
            let top = get(&format!("{k}.top_logits"))?.to_vec2::<f32>()?;
            let z = z.to_device(&Device::Cpu)?.to_vec2::<f32>()?;
            for (row, (ti, tz)) in top_ids.iter().zip(&top).enumerate() {
                let mine: Vec<f32> = ti.iter().map(|&i| z[row][i as usize]).collect();
                rep.logit_rel = rep.logit_rel.max(rel(&mine, tz));
                rep.min_gap = rep.min_gap.min((tz[0] - tz[1]) as f64);
            }
            let lse_ref = get(&format!("{k}.lse"))?.to_vec1::<f32>()?;
            rep.lse_rel = rep.lse_rel.max(rel(&out.lse, &lse_ref));
            let am = get(&format!("{k}.argmax"))?.to_dtype(DType::U32)?.to_vec1::<u32>()?;
            rep.chain_equal += usize::from(out.chain == am);
            let tt = get(&format!("{k}.tree_tok"))?.to_dtype(DType::U32)?.to_vec1::<u32>()?;
            let tp = get(&format!("{k}.tree_parent"))?.to_dtype(DType::I64)?.to_vec1::<i64>()?;
            let tree = draft_tree(&out, tt.len());
            let want: Vec<(u32, i32)> = tt.iter().zip(&tp).map(|(&t, &p)| (t, p as i32)).collect();
            rep.tree_equal += usize::from(tree == want);
            r += 1;
        }
        p += 1;
    }
    if rep.rounds == 0 {
        bail!("{}: no rounds", path.display());
    }
    Ok(rep)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tree_is_closed_under_ancestors_and_best_first() {
        // Row d's candidates: log probs -0.1, -2, -3, -4 at every depth, so the
        // best path is the chain of rank 0, and the first siblings come after.
        let rows = 15;
        let out = DraftOut {
            chain: (0..rows as u32).collect(),
            top_ids: (0..rows as u32).map(|d| [10 * d, 10 * d + 1, 10 * d + 2, 10 * d + 3]).collect(),
            top_logp: vec![[-0.1, -2.0, -3.0, -4.0]; rows],
            lse: vec![0.0; rows],
        };
        let t = draft_tree(&out, 15);
        assert_eq!(t.len(), 15);
        for (i, &(_, p)) in t.iter().enumerate() {
            assert!(p < i as i32);
        }
        // The chain of rank 0 to depth 15 scores -1.5 and beats any rank-1 node.
        for (i, &(tok, p)) in t.iter().enumerate() {
            assert_eq!(tok, 10 * i as u32);
            assert_eq!(p, i as i32 - 1);
        }
        let small = draft_tree(&out, 3);
        assert_eq!(small, vec![(0, -1), (10, 0), (20, 1)]);
    }

    #[test]
    fn dequant_low_nibble_first() {
        let q = [0x21u8, 0x43];
        let s = [half::f16::from_f32(0.5)];
        let b = [half::f16::from_f32(-1.0)];
        assert_eq!(dequant_q4(&q, &s, &b, 4), vec![-0.5, 0.0, 0.5, 1.0]);
    }
}
