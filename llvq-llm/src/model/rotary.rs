//! The rotary position tables, one sine and one cosine row per position. A file of its own
//! because every block applies them and nothing else in the model depends on how they are built.

use candle_core::{DType, Device, Result, Tensor};
use candle_transformers::models::qwen3::Config;

pub struct Rotary {
    pub(super) sin: Tensor,
    pub(super) cos: Tensor,
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

    pub(super) fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let (_, _, l, _) = q.dims4()?;
        let cos = self.cos.narrow(0, offset, l)?;
        let sin = self.sin.narrow(0, offset, l)?;
        Ok((
            candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?,
            candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?,
        ))
    }
}
