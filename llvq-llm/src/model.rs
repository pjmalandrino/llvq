//! A Qwen3 forward pass we control, so the input of every linear layer is
//! observable.
//!
//! GPTQ needs `H = AᵀA/N` where `A` is the *input* to a linear layer. No
//! inference library exposes that: `candle`'s Qwen3 applies its `Linear`s
//! inline with no hook, and wrapping them from outside is not possible
//! because the intermediate tensors never leave the block. So the block
//! forward is written out here, with capture points on the four distinct
//! activations a Qwen3 block produces.
//!
//! **Four, not seven.** `q_proj`, `k_proj` and `v_proj` all consume the same
//! tensor (`input_layernorm(x)`), and `gate_proj`/`up_proj` both consume
//! `post_attention_layernorm(x)`. Only `o_proj` and `down_proj` have inputs
//! of their own. Accumulating one Hessian per *activation* rather than per
//! *matrix* is exact and saves 3/7 of the work.
//!
//! The implementation is validated against `candle_transformers`' own Qwen3
//! (see `bin/oracle.rs`): same weights, same window, logits must agree. That
//! reference is the only thing standing between a subtly wrong RoPE and
//! three weeks of chasing a perplexity that is off by 15 %.
//!
//! Derived from the architecture as implemented in `candle-transformers`
//! (MIT OR Apache-2.0), restructured for observability.

use candle_core::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::{Activation, Embedding, Linear, RmsNorm, VarBuilder};
use candle_transformers::models::qwen3::Config;

/// Which activation a capture callback is being handed.
///
/// Named after the tensor, not the matrix: `Attn` feeds `q_proj`, `k_proj`
/// and `v_proj` alike.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Act {
    /// `input_layernorm(x)` — input of q/k/v_proj. Width: `hidden_size`.
    Attn,
    /// The concatenated attention context — input of o_proj.
    /// Width: `head_dim · num_attention_heads`.
    AttnOut,
    /// `post_attention_layernorm(x)` — input of gate/up_proj.
    /// Width: `hidden_size`.
    Mlp,
    /// `act(gate(h)) · up(h)` — input of down_proj. Width: `intermediate_size`.
    MlpOut,
}

impl Act {
    pub const ALL: [Act; 4] = [Act::Attn, Act::AttnOut, Act::Mlp, Act::MlpOut];

    /// Stable index, so a rotation seed derived from it is reproducible
    /// across runs and across reorderings of `ALL`.
    pub fn index(&self) -> u64 {
        match self {
            Act::Attn => 0,
            Act::AttnOut => 1,
            Act::Mlp => 2,
            Act::MlpOut => 3,
        }
    }

    /// The matrices this activation feeds, as safetensors name suffixes.
    pub fn consumers(&self) -> &'static [&'static str] {
        match self {
            Act::Attn => &["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
            Act::AttnOut => &["self_attn.o_proj"],
            Act::Mlp => &["mlp.gate_proj", "mlp.up_proj"],
            Act::MlpOut => &["mlp.down_proj"],
        }
    }

    pub fn width(&self, cfg: &Config) -> usize {
        match self {
            Act::Attn | Act::Mlp => cfg.hidden_size,
            Act::AttnOut => cfg.head_dim * cfg.num_attention_heads,
            Act::MlpOut => cfg.intermediate_size,
        }
    }
}

/// Receives every linear-layer input as the forward pass produces it.
pub trait Capture {
    /// `x` is `(batch, seq, width)`; rows are samples for `AᵀA`.
    fn on_activation(&mut self, layer: usize, act: Act, x: &Tensor) -> Result<()>;
}

/// The do-nothing capture, for plain inference.
pub struct NoCapture;

impl Capture for NoCapture {
    fn on_activation(&mut self, _: usize, _: Act, _: &Tensor) -> Result<()> {
        Ok(())
    }
}

pub struct Rotary {
    sin: Tensor,
    cos: Tensor,
}

impl Rotary {
    pub fn new(dtype: DType, cfg: &Config, dev: &Device) -> Result<Self> {
        let dim = cfg.head_dim;
        let inv: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let n = inv.len();
        let inv = Tensor::from_vec(inv, (1, n), dev)?;
        let t = Tensor::arange(0u32, cfg.max_position_embeddings as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((cfg.max_position_embeddings, 1))?;
        let freqs = t.matmul(&inv)?;
        Ok(Self {
            sin: freqs.sin()?.to_dtype(dtype)?,
            cos: freqs.cos()?.to_dtype(dtype)?,
        })
    }

    fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let (_, _, l, _) = q.dims4()?;
        let cos = self.cos.narrow(0, offset, l)?;
        let sin = self.sin.narrow(0, offset, l)?;
        Ok((
            candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?,
            candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?,
        ))
    }
}

/// One transformer block, with its seven projections reachable by name.
pub struct Block {
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub o_proj: Linear,
    pub gate_proj: Linear,
    pub up_proj: Linear,
    pub down_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    ln1: RmsNorm,
    ln2: RmsNorm,
    act: Activation,
    n_heads: usize,
    n_kv: usize,
    head_dim: usize,
}

impl Block {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let (nh, nkv, hd) = (
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim,
        );
        let a = vb.pp("self_attn");
        let m = vb.pp("mlp");
        Ok(Self {
            q_proj: candle_nn::linear_no_bias(cfg.hidden_size, nh * hd, a.pp("q_proj"))?,
            k_proj: candle_nn::linear_no_bias(cfg.hidden_size, nkv * hd, a.pp("k_proj"))?,
            v_proj: candle_nn::linear_no_bias(cfg.hidden_size, nkv * hd, a.pp("v_proj"))?,
            o_proj: candle_nn::linear_no_bias(nh * hd, cfg.hidden_size, a.pp("o_proj"))?,
            gate_proj: candle_nn::linear_no_bias(
                cfg.hidden_size,
                cfg.intermediate_size,
                m.pp("gate_proj"),
            )?,
            up_proj: candle_nn::linear_no_bias(
                cfg.hidden_size,
                cfg.intermediate_size,
                m.pp("up_proj"),
            )?,
            down_proj: candle_nn::linear_no_bias(
                cfg.intermediate_size,
                cfg.hidden_size,
                m.pp("down_proj"),
            )?,
            q_norm: candle_nn::rms_norm(hd, cfg.rms_norm_eps, a.pp("q_norm"))?,
            k_norm: candle_nn::rms_norm(hd, cfg.rms_norm_eps, a.pp("k_norm"))?,
            ln1: candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?,
            ln2: candle_nn::rms_norm(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
            act: cfg.hidden_act,
            n_heads: nh,
            n_kv: nkv,
            head_dim: hd,
        })
    }

    /// Read-only view of a projection, by checkpoint name.
    pub fn linear(&self, name: &str) -> &Linear {
        match name {
            "self_attn.q_proj" => &self.q_proj,
            "self_attn.k_proj" => &self.k_proj,
            "self_attn.v_proj" => &self.v_proj,
            "self_attn.o_proj" => &self.o_proj,
            "mlp.gate_proj" => &self.gate_proj,
            "mlp.up_proj" => &self.up_proj,
            "mlp.down_proj" => &self.down_proj,
            other => panic!("unknown projection {other}"),
        }
    }

    /// The projection a safetensors name suffix refers to.
    ///
    /// The quantizer walks activations, not fields, so it needs to reach
    /// matrices by the name the checkpoint uses.
    pub fn linear_mut(&mut self, name: &str) -> &mut Linear {
        match name {
            "self_attn.q_proj" => &mut self.q_proj,
            "self_attn.k_proj" => &mut self.k_proj,
            "self_attn.v_proj" => &mut self.v_proj,
            "self_attn.o_proj" => &mut self.o_proj,
            "mlp.gate_proj" => &mut self.gate_proj,
            "mlp.up_proj" => &mut self.up_proj,
            "mlp.down_proj" => &mut self.down_proj,
            other => panic!("unknown projection {other}"),
        }
    }

    /// Full-sequence forward, no KV cache — scoring, not generation.
    pub fn forward(
        &self,
        x: &Tensor,
        rotary: &Rotary,
        mask: &Tensor,
        idx: usize,
        cap: &mut dyn Capture,
    ) -> Result<Tensor> {
        let (b, l, _) = x.dims3()?;

        let h = self.ln1.forward(x)?;
        cap.on_activation(idx, Act::Attn, &h)?;

        let q = self
            .q_proj
            .forward(&h)?
            .reshape((b, l, self.n_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = self
            .k_proj
            .forward(&h)?
            .reshape((b, l, self.n_kv, self.head_dim))?
            .transpose(1, 2)?;
        let v = self
            .v_proj
            .forward(&h)?
            .reshape((b, l, self.n_kv, self.head_dim))?
            .transpose(1, 2)?;

        // Per-head RMSNorm on q and k — specific to Qwen3, and the detail a
        // hand-written forward is most likely to drop.
        let q = self
            .q_norm
            .forward(&q.flatten(0, 2)?)?
            .reshape((b, self.n_heads, l, self.head_dim))?;
        let k = self
            .k_norm
            .forward(&k.flatten(0, 2)?)?
            .reshape((b, self.n_kv, l, self.head_dim))?;

        let (q, k) = rotary.apply(&q, &k, 0)?;
        let groups = self.n_heads / self.n_kv;
        let k = candle_transformers::utils::repeat_kv(k, groups)?.contiguous()?;
        let v = candle_transformers::utils::repeat_kv(v, groups)?.contiguous()?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?.broadcast_add(mask)?;
        let ctx = candle_nn::ops::softmax_last_dim(&scores)?.matmul(&v)?;
        let ctx = ctx
            .transpose(1, 2)?
            .reshape((b, l, self.n_heads * self.head_dim))?;
        cap.on_activation(idx, Act::AttnOut, &ctx)?;
        let x = (x + self.o_proj.forward(&ctx)?)?;

        let h = self.ln2.forward(&x)?;
        cap.on_activation(idx, Act::Mlp, &h)?;
        let gated = (self.gate_proj.forward(&h)?.apply(&self.act)? * self.up_proj.forward(&h)?)?;
        cap.on_activation(idx, Act::MlpOut, &gated)?;
        x + self.down_proj.forward(&gated)?
    }
}

/// Qwen3, loaded for scoring.
pub struct Qwen3 {
    cfg: Config,
    embed: Embedding,
    pub blocks: Vec<Block>,
    norm: RmsNorm,
    /// `lm_head`; Qwen3-0.6B ties it to the embedding matrix.
    head: Tensor,
    rotary: Rotary,
    device: Device,
    dtype: DType,
}

impl Qwen3 {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let embed = candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("model.embed_tokens"))?;
        let head = if cfg.tie_word_embeddings {
            embed.embeddings().clone()
        } else {
            vb.get((cfg.vocab_size, cfg.hidden_size), "lm_head.weight")?
        };
        let vb_l = vb.pp("model.layers");
        let blocks = (0..cfg.num_hidden_layers)
            .map(|i| Block::new(cfg, vb_l.pp(i)))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            cfg: cfg.clone(),
            embed,
            blocks,
            norm: candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("model.norm"))?,
            head,
            rotary: Rotary::new(vb.dtype(), cfg, vb.device())?,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    fn causal_mask(&self, b: usize, l: usize) -> Result<Tensor> {
        let m: Vec<f32> = (0..l)
            .flat_map(|i| (0..l).map(move |j| if j <= i { 0.0 } else { f32::NEG_INFINITY }))
            .collect();
        Tensor::from_slice(&m, (b, 1, l, l), &self.device)?.to_dtype(self.dtype)
    }

    /// Hidden states after the final norm, for every position.
    pub fn hidden(&self, input: &Tensor, cap: &mut dyn Capture) -> Result<Tensor> {
        let (b, l) = input.dims2()?;
        let mask = self.causal_mask(b, l)?;
        let mut h = self.embed.forward(input)?;
        for (i, blk) in self.blocks.iter().enumerate() {
            h = blk.forward(&h, &self.rotary, &mask, i, cap)?;
        }
        self.norm.forward(&h)
    }

    /// Logits for every position: `(batch, seq, vocab)`.
    pub fn logits(&self, input: &Tensor, cap: &mut dyn Capture) -> Result<Tensor> {
        self.hidden(input, cap)?
            .broadcast_matmul(&self.head.t()?)
    }

    /// Mean next-token negative log-likelihood over one window, in nats.
    ///
    /// Position `i` predicts token `i+1`, so a window of `l` tokens scores
    /// `l − 1` predictions.
    pub fn window_nll(&self, tokens: &[u32], cap: &mut dyn Capture) -> Result<(f64, usize)> {
        let l = tokens.len();
        assert!(l >= 2, "a window must hold at least two tokens");
        let input = Tensor::from_slice(tokens, (1, l), &self.device)?;
        let logits = self.logits(&input, cap)?.to_dtype(DType::F32)?;
        let logits = logits.i(0)?.narrow(0, 0, l - 1)?;
        let targets = Tensor::from_slice(&tokens[1..], l - 1, &self.device)?;
        let logp = candle_nn::ops::log_softmax(&logits, D::Minus1)?;
        let picked = logp.gather(&targets.reshape((l - 1, 1))?, 1)?;
        let total: f32 = picked.sum_all()?.to_scalar()?;
        Ok((-total as f64, l - 1))
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Greedy continuation of `prompt`, `max_new` tokens.
    ///
    /// No KV cache: each step re-runs the whole prefix. That is quadratic and
    /// entirely adequate for a probe of a few dozen tokens — and it reuses
    /// exactly the scoring path the perplexity numbers come from, so what is
    /// generated is what was measured.
    pub fn generate(
        &self,
        tokens: &[u32],
        max_new: usize,
        cap: &mut dyn Capture,
    ) -> Result<Vec<u32>> {
        let mut ids = tokens.to_vec();
        let mut out = Vec::with_capacity(max_new);
        for _ in 0..max_new {
            let input = Tensor::from_slice(&ids, (1, ids.len()), &self.device)?;
            let logits = self.logits(&input, cap)?.to_dtype(DType::F32)?;
            let last = logits.i((0, ids.len() - 1))?;
            let next = last.argmax(D::Minus1)?.to_scalar::<u32>()?;
            ids.push(next);
            out.push(next);
        }
        Ok(out)
    }

    pub fn rotary(&self) -> &Rotary {
        &self.rotary
    }

    /// Embed a token window — the input of block 0.
    pub fn embed_tokens(&self, input: &Tensor) -> Result<Tensor> {
        self.embed.forward(input)
    }

    /// Hidden states through the final norm, starting from block 0's input.
    /// Used to score a model whose blocks were replaced in place.
    pub fn head_from_hidden(&self, h: &Tensor) -> Result<Tensor> {
        self.norm.forward(h)?.broadcast_matmul(&self.head.t()?)
    }

    /// The causal mask matching a `(batch, seq, _)` hidden-state tensor.
    pub fn causal_mask_for(&self, h: &Tensor) -> Result<Tensor> {
        let (b, l, _) = h.dims3()?;
        self.causal_mask(b, l)
    }
}
