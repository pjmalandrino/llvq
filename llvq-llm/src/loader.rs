//! Fetching and mapping a checkpoint.

use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen3::Config;
use std::path::PathBuf;

/// Everything needed to run a model, resolved from the Hugging Face cache
/// (downloading on first use).
pub struct Checkpoint {
    pub config: Config,
    pub weights: Vec<PathBuf>,
    pub tokenizer: PathBuf,
}

impl Checkpoint {
    pub fn fetch(repo: &str) -> anyhow::Result<Self> {
        let api = hf_hub::api::sync::Api::new()?;
        let r = api.model(repo.to_string());
        let config_path = r.get("config.json")?;
        let tokenizer = r.get("tokenizer.json")?;
        let config: Config = serde_json::from_slice(&std::fs::read(&config_path)?)?;

        // Small models ship one shard; larger ones an index listing many.
        let weights = match r.get("model.safetensors") {
            Ok(p) => vec![p],
            Err(_) => {
                let idx = r.get("model.safetensors.index.json")?;
                let json: serde_json::Value = serde_json::from_slice(&std::fs::read(idx)?)?;
                let map = json["weight_map"]
                    .as_object()
                    .ok_or_else(|| anyhow::anyhow!("malformed safetensors index"))?;
                let mut names: Vec<String> = map
                    .values()
                    .filter_map(|v| v.as_str().map(str::to_owned))
                    .collect();
                names.sort();
                names.dedup();
                names
                    .iter()
                    .map(|n| r.get(n).map_err(anyhow::Error::from))
                    .collect::<anyhow::Result<Vec<_>>>()?
            }
        };
        Ok(Self {
            config,
            weights,
            tokenizer,
        })
    }

    /// Map the weights.
    ///
    /// # Safety
    /// The checkpoint files must not be modified while the mapping is alive.
    pub fn var_builder(&self, dtype: DType, device: &Device) -> anyhow::Result<VarBuilder<'_>> {
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&self.weights, dtype, device)? };
        Ok(vb)
    }

    pub fn tokenizer(&self) -> anyhow::Result<tokenizers::Tokenizer> {
        tokenizers::Tokenizer::from_file(&self.tokenizer)
            .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))
    }
}
