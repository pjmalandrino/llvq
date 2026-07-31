//! Saving and reloading a quantized model.
//!
//! Without this, every experiment re-quantizes from scratch — 36 minutes on
//! Qwen3-0.6B — which makes it impossible to *look* at the model afterwards.
//! Quantize once, probe as often as you like.
//!
//! ## What this file is, and is not
//!
//! It stores the **reconstructions**: the floating-point values the lattice
//! decoder would produce, one per weight. That is what an evaluation needs
//! and it plugs straight into an unmodified model.
//!
//! It is **not** the compressed artifact. The real 2-bit format stores one
//! 48-bit index per 24 weights (the bijective indexing of gate G3) and is
//! ~8× smaller; producing it means writing the index side and a decoder, and
//! that belongs with the fused kernel of G6. Do not quote the size of this
//! file as a compression ratio.

use crate::model::{Act, Qwen3};
use candle_core::{DType, Device, Tensor};
use std::collections::HashMap;
use std::path::Path;

/// Name of a projection's weight, matching the checkpoint convention.
pub fn key(block: usize, name: &str) -> String {
    format!("model.layers.{block}.{name}.weight")
}

/// Every quantized projection, in `f16` — the reconstructions sit on top of a
/// 2-bit codebook, so half precision costs nothing meaningful and halves the
/// file.
pub fn save(model: &Qwen3, path: impl AsRef<Path>) -> anyhow::Result<()> {
    let mut out: HashMap<String, Tensor> = HashMap::new();
    for (b, block) in model.blocks.iter().enumerate() {
        for act in Act::ALL {
            for name in act.consumers() {
                let w = block.linear(name).weight().to_dtype(DType::F16)?;
                out.insert(key(b, name), w);
            }
        }
    }
    candle_core::safetensors::save(&out, path.as_ref())?;
    Ok(())
}

/// Overwrite the projections of `model` with a saved set.
pub fn load(model: &mut Qwen3, path: impl AsRef<Path>, device: &Device) -> anyhow::Result<usize> {
    let tensors = candle_core::safetensors::load(path.as_ref(), device)?;
    let mut n = 0;
    let dtype = model.dtype();
    for b in 0..model.blocks.len() {
        for act in Act::ALL {
            for name in act.consumers() {
                let k = key(b, name);
                let t = tensors
                    .get(&k)
                    .ok_or_else(|| anyhow::anyhow!("missing tensor {k}"))?
                    .to_dtype(dtype)?;
                let lin = model.blocks[b].linear_mut(name);
                anyhow::ensure!(
                    lin.weight().dims() == t.dims(),
                    "{k}: shape {:?} does not match the model's {:?}",
                    t.dims(),
                    lin.weight().dims()
                );
                *lin = candle_nn::Linear::new(t, None);
                n += 1;
            }
        }
    }
    Ok(n)
}
