//! The minimal causal language model contract: a prompt forward that fills a
//! cache, one decode step on that cache, and the vocabulary size.
//!
//! `bin/ppl` and `bin/gsm8k` build [`crate::model::Qwen3`] directly. This
//! trait is the seam a second architecture (Inkling 2 Mini, `crate::inkling`)
//! plugs into without moving the Qwen3 paths: [`Qwen3`] implements it by
//! delegation to methods it already had, and nothing in `model.rs` changes.

use candle_core::{Result, Tensor};

use crate::model::{KvCache, Qwen3};

/// A causal language model that decodes token by token on its own cache.
pub trait CausalLm {
    /// Whatever the model keeps between steps (key-value caches, convolution
    /// states, ...). Opaque to the caller.
    type Cache;

    /// Rows of the output projection: the length of a logits vector.
    fn vocab_size(&self) -> usize;

    /// An empty cache, ready for [`Self::prefill`].
    fn new_cache(&self) -> Self::Cache;

    /// Run `tokens` from position 0 into `cache` and return the f32 logits of
    /// the last position, `[1, 1, vocab]`.
    fn prefill(&self, tokens: &[u32], cache: &mut Self::Cache) -> Result<Tensor>;

    /// Feed `token` at absolute position `offset` (the number of tokens the
    /// cache already holds) and return its f32 logits, `[1, 1, vocab]`.
    fn step(&self, token: u32, offset: usize, cache: &mut Self::Cache) -> Result<Tensor>;
}

impl CausalLm for Qwen3 {
    type Cache = Vec<KvCache>;

    fn vocab_size(&self) -> usize {
        self.config().vocab_size
    }

    fn new_cache(&self) -> Self::Cache {
        self.fresh_caches()
    }

    /// [`Qwen3::prefill_chunked`] in one chunk.
    fn prefill(&self, tokens: &[u32], cache: &mut Self::Cache) -> Result<Tensor> {
        self.prefill_chunked(tokens, tokens.len().max(1), cache)
    }

    /// [`Qwen3::step_logits`].
    fn step(&self, token: u32, offset: usize, cache: &mut Self::Cache) -> Result<Tensor> {
        self.step_logits(token, offset, cache)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::tests_support::tiny_model_kernel;

    /// Greedy decode through the trait only, generic over the model.
    fn greedy<M: CausalLm>(m: &M, prompt: &[u32], n: usize) -> Result<Vec<u32>> {
        let mut cache = m.new_cache();
        let mut logits = m.prefill(prompt, &mut cache)?;
        let mut out = Vec::new();
        for i in 0..n {
            let next = logits.flatten_all()?.argmax(0)?.to_scalar::<u32>()?;
            out.push(next);
            logits = m.step(next, prompt.len() + i, &mut cache)?;
        }
        Ok(out)
    }

    /// The trait is a pure delegation: the same logits, bit for bit, as the
    /// inherent methods it names, and the vocabulary of the config.
    #[test]
    fn qwen3_trait_delegates_bit_for_bit() -> Result<()> {
        let m = tiny_model_kernel();
        let prompt = [1u32, 5, 9, 2, 7];
        assert_eq!(CausalLm::vocab_size(&m), m.config().vocab_size);

        let mut a = CausalLm::new_cache(&m);
        let mut b = m.fresh_caches();
        let la = CausalLm::prefill(&m, &prompt, &mut a)?.flatten_all()?.to_vec1::<f32>()?;
        let lb = m.prefill_chunked(&prompt, prompt.len(), &mut b)?.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(la.len(), m.config().vocab_size);
        assert_eq!(la, lb, "prefill");
        let sa = CausalLm::step(&m, 3, prompt.len(), &mut a)?.flatten_all()?.to_vec1::<f32>()?;
        let sb = m.step_logits(3, prompt.len(), &mut b)?.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(sa, sb, "step");

        // and a generic caller runs on it
        assert_eq!(greedy(&m, &prompt, 4)?.len(), 4);
        Ok(())
    }
}
