//! The DFlash drafter behind the speculative loop's `Drafter` trait.
//!
//! The contract is `llvq_cuda::spec_loop`'s (`MAX_DRAFT`, `TAPS`,
//! `DraftInput`, `Drafter`), portable: `llvq-cuda` is a dependency of this
//! crate on the card (feature `cuda`) and on the Mac (no cudarc there).
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

use llvq_cuda::spec_loop::{DraftInput, Drafter, MAX_DRAFT, TAPS};

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
    /// When set, each round's stages are timed apart, the device synchronised
    /// between them: seconds a round in the four vectors below.
    pub timing: bool,
    pub t_extend: Vec<f64>,
    pub t_block: Vec<f64>,
    pub t_head: Vec<f64>,
    pub t_readout: Vec<f64>,
    /// Of `t_extend`, the host conversion and upload of `new_taps` (to the
    /// f32 tensor on the device, synchronised), timed when `timing` is set.
    pub t_upload: Vec<f64>,
    /// The readout of e3bbaee ([`dflash::readout_chunked`], a host hop
    /// mid-round) instead of [`dflash::DeviceReadout`], for comparison.
    pub chunked_readout: bool,
    readout: Option<dflash::DeviceReadout>,
}

impl DFlashSpec {
    pub fn new(drafter: DFlashDrafter, vocab: Box<dyn TargetVocab>, shape: Shape) -> anyhow::Result<Self> {
        let ctx = drafter.new_context(4096)?;
        Ok(Self {
            drafter,
            vocab,
            shape,
            ctx,
            skipped: 0,
            dead: false,
            last_error: None,
            timing: false,
            t_extend: Vec::new(),
            t_block: Vec::new(),
            t_head: Vec::new(),
            t_readout: Vec::new(),
            t_upload: Vec::new(),
            chunked_readout: false,
            readout: None,
        })
    }

    /// A fresh context and counters, for the next generation.
    pub fn reset(&mut self) -> anyhow::Result<()> {
        self.ctx = self.drafter.new_context(4096)?;
        self.skipped = 0;
        self.dead = false;
        self.last_error = None;
        self.t_extend.clear();
        self.t_block.clear();
        self.t_head.clear();
        self.t_readout.clear();
        self.t_upload.clear();
        Ok(())
    }

    fn sync(&self) -> anyhow::Result<()> {
        if self.timing {
            self.drafter.device().synchronize()?;
        }
        Ok(())
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
        self.sync()?;
        let t0 = std::time::Instant::now();
        if n > 0 {
            let bits: Vec<half::f16> = input.new_taps.iter().map(|&b| half::f16::from_bits(b)).collect();
            let taps = Tensor::from_vec(bits, (n, w), self.drafter.device())?.to_dtype(DType::F32)?;
            if self.timing {
                self.drafter.device().synchronize()?;
                self.t_upload.push(t0.elapsed().as_secs_f64());
            }
            self.drafter.extend(&mut self.ctx, &taps)?;
        }
        self.sync()?;
        if self.timing {
            self.t_extend.push(t0.elapsed().as_secs_f64());
        }
        let m = input.tokens.len().checked_sub(1).ok_or_else(|| anyhow::anyhow!("no root"))?;
        if self.ctx.len() != m {
            // Taps missing, or more than the root's predecessors: no aligned context.
            self.dead = true;
            self.skipped += 1;
            return Ok(Vec::new());
        }
        // the drafter's own draft (block, head, readout), its stages timed apart
        let t1 = std::time::Instant::now();
        let hidden = self.drafter.block_hidden(&self.ctx, input.tokens[m], self.vocab.as_ref())?;
        self.sync()?;
        let t2 = std::time::Instant::now();
        let logits = self.vocab.logits(&hidden)?;
        self.sync()?;
        let t3 = std::time::Instant::now();
        let out = if self.chunked_readout {
            dflash::readout_chunked(&logits)?
        } else {
            let (rows, vocab) = logits.dims2()?;
            if self.readout.is_none() {
                self.readout = Some(dflash::DeviceReadout::new(rows, vocab, logits.device())?);
            }
            self.readout.as_ref().expect("made above").run(&logits)?
        };
        if self.timing {
            self.t_block.push((t2 - t1).as_secs_f64());
            self.t_head.push((t3 - t2).as_secs_f64());
            self.t_readout.push(t3.elapsed().as_secs_f64());
        }
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
