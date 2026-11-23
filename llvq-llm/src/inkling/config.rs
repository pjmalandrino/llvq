//! The text config of Inkling 2 Mini, read from `config.json` (`D/textonly/`
//! or the `text_config` of the native file).
//!
//! The authority is `Inkling2LoopTextConfig` (transformers 046bdb8cc2,
//! `configuration_inkling2_loop.py`). Fields the reference code reads but the
//! file does not carry take the reference defaults. Fields the file carries
//! but the reference code ignores, or hard-codes to one behaviour, are
//! checked against that behaviour: a value the forward pass would silently
//! misread is refused at load.

use serde::Deserialize;

/// `loop_config` of the file, read by `Inkling2LoopTextConfig.__post_init__`
/// (configuration lines 135-140).
#[derive(Debug, Clone, Deserialize)]
pub struct LoopConfig {
    pub core_start: usize,
    pub core_end: usize,
    pub num_passes: usize,
    pub gate_rank: usize,
    #[serde(default)]
    pub is_varloop: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct InklingConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub vocab_size: usize,
    pub unpadded_vocab_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub swa_head_dim: usize,
    pub swa_num_attention_heads: usize,
    pub swa_num_key_value_heads: usize,
    pub sliding_window_size: usize,
    pub d_rel: usize,
    pub rel_extent: usize,
    pub log_scaling_n_floor: f64,
    pub log_scaling_alpha: f64,
    pub rms_norm_eps: f64,
    pub qk_l2_clip_threshold: f64,
    pub rel_l2_clip_threshold: f64,
    /// `conv_kernel_size` in the reference (attribute map, configuration line 88).
    pub sconv_kernel_size: usize,
    pub final_logit_softcapping: f64,
    pub n_layers_per_attn_res_block: usize,
    pub n_routed_experts: usize,
    pub num_experts_per_tok: usize,
    #[serde(default)]
    pub n_shared_experts: usize,
    /// `moe_intermediate_size` in the reference (attribute map, line 90).
    pub intermediate_size: usize,
    pub mlp_gate_softcap: f64,
    pub mlp_up_softcap: f64,
    /// `routed_scaling_factor` in the reference (attribute map, line 91).
    pub route_scale: f64,
    #[serde(default = "yes")]
    pub use_gate_bias: bool,
    #[serde(default = "sigmoid")]
    pub scoring_func: String,
    #[serde(default)]
    pub q_bias: bool,
    #[serde(default)]
    pub o_bias: bool,
    #[serde(default = "yes")]
    pub use_value_norm_no_gamma: bool,
    #[serde(default = "yes")]
    pub attn_res_normalize_values: bool,
    /// Absent from the file: the reference derives it (configuration lines
    /// 141-144), `"hybrid"` (full attention) when `(i + 1) % 6 == 0`.
    #[serde(default)]
    pub layer_types: Option<Vec<String>>,
    /// Carried by the file, **not read** by the reference, which derives the
    /// layer types from `(i + 1) % 6`. Checked against that derivation.
    #[serde(default)]
    pub local_layer_ids: Option<Vec<usize>>,
    pub loop_config: LoopConfig,
}

fn yes() -> bool {
    true
}

fn sigmoid() -> String {
    "sigmoid".into()
}

impl InklingConfig {
    pub fn from_file(path: &std::path::Path) -> anyhow::Result<Self> {
        let raw = std::fs::read(path).map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
        let mut v: serde_json::Value = serde_json::from_slice(&raw)?;
        // The native `config.json` nests the text model under `text_config`.
        if let Some(tc) = v.get("text_config") {
            v = tc.clone();
        }
        let mut cfg: Self = serde_json::from_value(v)?;
        cfg.finish()?;
        Ok(cfg)
    }

    /// Derive what the reference derives and refuse what this forward pass
    /// does not implement.
    fn finish(&mut self) -> anyhow::Result<()> {
        let n = self.num_hidden_layers;
        if self.layer_types.is_none() {
            self.layer_types = Some(
                (0..n)
                    .map(|i| if (i + 1) % 6 == 0 { "hybrid" } else { "hybrid_sliding" }.to_string())
                    .collect(),
            );
        }
        let types = self.layer_types.as_ref().expect("set above");
        anyhow::ensure!(types.len() == n, "layer_types has {} entries for {n} layers", types.len());
        for t in types {
            anyhow::ensure!(t == "hybrid" || t == "hybrid_sliding", "unknown layer type {t:?}");
        }
        if let Some(local) = &self.local_layer_ids {
            let derived: Vec<usize> = (0..n).filter(|&i| self.is_sliding(i)).collect();
            anyhow::ensure!(
                *local == derived,
                "local_layer_ids {local:?} disagree with the layer types the reference derives {derived:?}"
            );
        }
        anyhow::ensure!(self.scoring_func == "sigmoid", "scoring_func {:?}", self.scoring_func);
        anyhow::ensure!(self.n_shared_experts == 0, "shared experts are not implemented");
        anyhow::ensure!(self.use_gate_bias, "the reference always adds e_score_correction_bias");
        anyhow::ensure!(!self.q_bias && !self.o_bias, "the reference has no attention biases");
        anyhow::ensure!(
            self.use_value_norm_no_gamma && self.attn_res_normalize_values,
            "the reference hard-codes unweighted value and residual norms"
        );
        anyhow::ensure!(!self.loop_config.is_varloop, "variable loops are not implemented");
        let b = self.n_layers_per_attn_res_block;
        anyhow::ensure!(
            self.loop_config.core_start > 0
                && self.loop_config.core_start.is_multiple_of(b)
                && self.loop_config.core_end.is_multiple_of(b),
            "loop bounds must be multiples of the block size (configuration lines 146-159)"
        );
        anyhow::ensure!(self.num_attention_heads % self.num_key_value_heads == 0);
        anyhow::ensure!(self.swa_num_attention_heads % self.swa_num_key_value_heads == 0);
        Ok(())
    }

    pub fn is_sliding(&self, layer: usize) -> bool {
        self.layer_types.as_ref().expect("finished config")[layer] == "hybrid_sliding"
    }

    pub fn is_loop_layer(&self, layer: usize) -> bool {
        (self.loop_config.core_start..self.loop_config.core_end).contains(&layer)
    }

    /// `is_block_boundary` of every decoder layer (modeling line 861).
    pub fn is_block_boundary(&self, layer: usize) -> bool {
        layer.is_multiple_of(self.n_layers_per_attn_res_block)
    }

    /// (query heads, key-value heads, head dim) of `layer` (modeling lines 551-554).
    pub fn attn_shape(&self, layer: usize) -> (usize, usize, usize) {
        if self.is_sliding(layer) {
            (self.swa_num_attention_heads, self.swa_num_key_value_heads, self.swa_head_dim)
        } else {
            (self.num_attention_heads, self.num_key_value_heads, self.head_dim)
        }
    }

    /// The relative bias extent of `layer` (modeling line 557): the sliding
    /// window for sliding layers, `rel_extent` for full ones.
    pub fn layer_rel_extent(&self, layer: usize) -> usize {
        if self.is_sliding(layer) {
            self.sliding_window_size
        } else {
            self.rel_extent
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny() -> serde_json::Value {
        serde_json::json!({
            "hidden_size": 48, "num_hidden_layers": 12, "vocab_size": 64, "unpadded_vocab_size": 60,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
            "swa_head_dim": 8, "swa_num_attention_heads": 4, "swa_num_key_value_heads": 2,
            "sliding_window_size": 16, "d_rel": 4, "rel_extent": 16,
            "log_scaling_n_floor": 8192, "log_scaling_alpha": 0.1, "rms_norm_eps": 1e-6,
            "qk_l2_clip_threshold": 22.6, "rel_l2_clip_threshold": 8.0, "sconv_kernel_size": 4,
            "final_logit_softcapping": 30.0, "n_layers_per_attn_res_block": 6,
            "n_routed_experts": 8, "num_experts_per_tok": 2, "intermediate_size": 24,
            "mlp_gate_softcap": 4.0, "mlp_up_softcap": 25.0, "route_scale": 8.0,
            "local_layer_ids": [0, 1, 2, 3, 4, 6, 7, 8, 9, 10],
            "loop_config": {"core_start": 6, "core_end": 12, "num_passes": 3, "gate_rank": 4}
        })
    }

    #[test]
    fn layer_types_follow_the_reference_derivation() {
        let mut c: InklingConfig = serde_json::from_value(tiny()).unwrap();
        c.finish().unwrap();
        assert!(!c.is_sliding(5) && !c.is_sliding(11) && c.is_sliding(0) && c.is_sliding(6));
        assert!(c.is_block_boundary(0) && c.is_block_boundary(6) && !c.is_block_boundary(5));
        assert!(c.is_loop_layer(6) && !c.is_loop_layer(5) && !c.is_loop_layer(12));
    }

    #[test]
    fn local_layer_ids_that_disagree_are_refused() {
        let mut v = tiny();
        v["local_layer_ids"] = serde_json::json!([0, 1, 2, 3, 4, 5, 7, 8, 9, 10]);
        let mut c: InklingConfig = serde_json::from_value(v).unwrap();
        assert!(c.finish().is_err());
    }
}
