//! The DFlash drafter behind the speculative loop's `Drafter` trait.
//!
//! `spec` holds a local copy of `MAX_DRAFT`, `DraftInput` and `Drafter` from
//! `llvq_cuda::spec_loop` (branch `claude/arbre-nuit-2026-11-10`, 5e65b12), so
//! this crate does not depend on `llvq-cuda` on a Mac; the merge points the
//! `use` below at `llvq_cuda::spec_loop` and deletes the copy.
//!
//! Conventions, the reference's (`ops/dflash_alpha.py`): with `m =
//! tokens.len() - 1`, the root `tokens[m]` sits at position m and the drafter
//! reads the target's taps at positions 0..m-1. The root's own taps are never
//! needed: they arrive with the next call, when the verify pass that moved the
//! root has run. So after the call's `new_taps` the drafter's context must
//! hold exactly m positions. If it does not (the verifier wrote no taps), the
//! context can never be realigned, since the missing positions are gone: the
//! adapter stops reading taps and returns no node from then on, the loop's
//! R = 1 decoding, counting each such round in [`DFlashSpec::skipped`].

use candle_core::{DType, Tensor};

use crate::dflash::{self, DFlashDrafter, DraftContext, TargetVocab, DRAFTS};

/// Local copy of `llvq_cuda::spec_loop`'s contract (5e65b12).
pub mod spec {
    pub const MAX_DRAFT: usize = 15;
    pub const TAPS: usize = 5;
    pub struct DraftInput<'a> {
        pub ctx_len: usize,
        pub tokens: &'a [u32],
        pub new_taps: &'a [u16],
        pub hidden: usize,
    }
    pub trait Drafter {
        fn draft(&mut self, input: &DraftInput) -> Vec<(u32, u8)>;
    }
}

use spec::{DraftInput, Drafter, MAX_DRAFT, TAPS};

/// What a round returns: the argmax chain of the first `k` drafts, or the
/// tree of `n` nodes drawn from the same pass.
#[derive(Debug, Clone, Copy)]
pub enum Shape {
    Chain(usize),
    Tree(usize),
}

pub struct DFlashSpec {
    pub drafter: DFlashDrafter,
    pub vocab: Box<dyn TargetVocab>,
    pub shape: Shape,
    ctx: DraftContext,
    /// Rounds that returned no node because the context did not reach the root.
    pub skipped: usize,
    dead: bool,
    /// The last error a round hit (the trait has no error path; the round then
    /// returns no node).
    pub last_error: Option<String>,
}

impl DFlashSpec {
    pub fn new(drafter: DFlashDrafter, vocab: Box<dyn TargetVocab>, shape: Shape) -> anyhow::Result<Self> {
        let ctx = drafter.new_context(4096)?;
        Ok(Self { drafter, vocab, shape, ctx, skipped: 0, dead: false, last_error: None })
    }

    fn round(&mut self, input: &DraftInput) -> anyhow::Result<Vec<(u32, u8)>> {
        let h = self.drafter.cfg.hidden;
        if input.hidden != h {
            anyhow::bail!("hidden {} against the drafter's {h}", input.hidden);
        }
        let w = TAPS * h;
        if !input.new_taps.len().is_multiple_of(w) {
            anyhow::bail!("new_taps holds {} values, not a multiple of {w}", input.new_taps.len());
        }
        if self.dead {
            self.skipped += 1;
            return Ok(Vec::new());
        }
        let n = input.new_taps.len() / w;
        if n > 0 {
            let bits: Vec<half::f16> = input.new_taps.iter().map(|&b| half::f16::from_bits(b)).collect();
            let taps = Tensor::from_vec(bits, (n, w), self.drafter.device())?.to_dtype(DType::F32)?;
            self.drafter.extend(&mut self.ctx, &taps)?;
        }
        let m = input.tokens.len().checked_sub(1).ok_or_else(|| anyhow::anyhow!("no root"))?;
        if self.ctx.len() != m {
            // Taps missing, or more than the root's predecessors: no aligned context.
            self.dead = true;
            self.skipped += 1;
            return Ok(Vec::new());
        }
        let out = self.drafter.draft(&self.ctx, input.tokens[m], self.vocab.as_ref())?;
        Ok(match self.shape {
            Shape::Chain(k) => {
                let k = k.min(DRAFTS).min(MAX_DRAFT);
                out.chain[..k].iter().enumerate().map(|(i, &t)| (t, i as u8)).collect()
            }
            Shape::Tree(nodes) => dflash::draft_tree(&out, nodes.min(MAX_DRAFT))
                .into_iter()
                .map(|(t, p)| (t, (p + 1) as u8))
                .collect(),
        })
    }
}

impl Drafter for DFlashSpec {
    fn draft(&mut self, input: &DraftInput) -> Vec<(u32, u8)> {
        match self.round(input) {
            Ok(v) => v,
            Err(e) => {
                self.last_error = Some(format!("{e:#}"));
                Vec::new()
            }
        }
    }
}
