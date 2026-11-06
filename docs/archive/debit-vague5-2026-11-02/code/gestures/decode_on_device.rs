// DRAFT, never compiled or run. Wave-5 deep dive "gestures", 2026-11-02, revised after review.
// Not part of the repository; a sketch for the operator to review.
//
// The token stays on the card, in steps that each move ONE mechanism, so the
// A/B ratios are single-mechanism ratios (METHODE §1: an A/B moves one
// mechanism only). The first draft routed the f16 arm through a ring copy
// (`slice_set`), which added a launch to the cast it removed, and left the
// readback of the key arm undefined; this version fixes both.
//
//   kind           what it does per token                                     vs its neighbour
//   Base           `generate` unchanged: head, f32 cast, candle argmax,       (today)
//                  to_scalar, from_slice of the id
//   F16            Base without the cast: candle argmax on the f16 logits,    F16 / Base = a1
//                  to_scalar, from_slice
//   Key            F16 with tok_argmax_h writing a ONE-slot ring instead of   Key / F16 = a2
//                  candle's argmax; to_scalar of the slot's rank-0 view,
//                  from_slice
//   OnDevKey       Key without the per-token readback and upload: the next    OnDevKey / Key = a3
//                  step embeds ring[pos]; one readback at the end
//   OnDevCandle    F16 without the per-token readback and upload: candle's    OnDevCandle / F16 = a3
//                  argmax output (keepdim) is embedded as the next input,     with candle's argmax
//                  no copy; one cat + readback at the end
//
// OnDev / Base moves three mechanisms and is reported only. OnDevCandle is the
// fallback of rule A2 (keep candle's kernel) measured in the same run, and the
// pair OnDevCandle / F16 against OnDevKey / Key says whether candle's per-call
// pageable upload of dims and strides (cuda_backend/mod.rs:333) holds a3 back.
//
// Same kernels, same order, same stream as `Qwen3::generate`
// (model.rs:2119-2155) up to the mechanism each kind names. Where it would
// live: llvq-llm/src/model.rs (generic, reads no environment), with the CUDA
// key kernel behind `KeyKernel` in llvq-llm/src/fused_cuda.rs.

use candle_core::{DType, IndexOp, Result, Tensor, D};

/// The key argmax as the model sees it: writes `ring[pos]` = the greedy token
/// of `logits` ([1, 1, vocab], f16), on the device, with candle's fast_argmax
/// tie order. Never reads to the host. CUDA: `tok_argmax_h` as a candle
/// `InplaceOp2` on the ring (no allocation, no upload). CPU tests: `HostKey`
/// below, the reference of argmax_key_ref.rs, so the plumbing of the Key and
/// OnDevKey kinds is pinned on the Mac too.
pub trait KeyKernel: Send + Sync {
    fn write(&self, logits: &Tensor, ring: &Tensor, pos: usize) -> Result<()>;
    /// Zero the sticky NaN flag before a generation.
    fn arm(&self) -> Result<()>;
    /// Any NaN met since `arm` (4-byte read; once per generation).
    fn nan_seen(&self) -> Result<bool>;
}

pub enum TokenLoop<'a> {
    F16,
    Key(&'a dyn KeyKernel),
    OnDevKey(&'a dyn KeyKernel),
    OnDevCandle,
}

impl Qwen3 {
    /// `logits_last` without the f32 cast: the argmax of the f16 logits is the
    /// argmax of their exact f32 widening (same fast_argmax template,
    /// reduce.cu:673 and :688). Callers that sample keep `logits_last`.
    pub fn logits_last_native(&self, h: &Tensor) -> Result<Tensor> {
        let l = h.dim(1)?;
        self.head.project(&h.narrow(1, l - 1, 1)?)
    }

    /// Greedy decode in one of the four non-base kinds. Base is `generate`
    /// itself, untouched, so the published loop is the reference of every arm.
    /// Same contract as `generate`: the same tokens, which
    /// `token_loops_match_generate` pins on every store and form, and the A/B
    /// pins on the card.
    pub fn generate_loop(&self, tokens: &[u32], max_new: usize, kind: &TokenLoop) -> Result<Vec<u32>> {
        if max_new == 0 {
            return Ok(Vec::new());
        }
        let mut caches = self.fresh_caches();
        // Key: one slot, read back every token. OnDevKey: one slot a token.
        let ring = match kind {
            TokenLoop::Key(k) => { k.arm()?; Some(Tensor::zeros(1, DType::U32, &self.device)?) }
            TokenLoop::OnDevKey(k) => { k.arm()?; Some(Tensor::zeros(max_new, DType::U32, &self.device)?) }
            _ => None,
        };
        let input = Tensor::from_slice(tokens, (1, tokens.len()), &self.device)?;
        let mut h = self.hidden_cached(&input, 0, &mut caches, &mut NoCapture)?;
        let mut offset = tokens.len();
        let mut out = Vec::with_capacity(max_new); // host ids (F16, Key)
        let mut kept = Vec::new(); // device ids (OnDevCandle)
        for pos in 0..max_new {
            let logits = self.logits_last_native(&h)?;
            // Each arm picks its token, then stops BEFORE building the next
            // input on the last token, as `generate` does (no extra upload).
            let next = match kind {
                TokenLoop::F16 => {
                    let id = logits.i((0, 0))?.argmax(D::Minus1)?.to_scalar::<u32>()?;
                    out.push(id);
                    if pos + 1 == max_new { break; }
                    Tensor::from_slice(&[id], (1, 1), &self.device)?
                }
                TokenLoop::Key(k) => {
                    let slot = ring.as_ref().expect("armed above");
                    k.write(&logits, slot, 0)?;
                    // rank-0 view: candle's to_scalar refuses any other rank
                    // (tensor.rs:662-669); it copies the 4-byte storage.
                    let id = slot.narrow(0, 0, 1)?.squeeze(0)?.to_scalar::<u32>()?;
                    out.push(id);
                    if pos + 1 == max_new { break; }
                    Tensor::from_slice(&[id], (1, 1), &self.device)?
                }
                TokenLoop::OnDevKey(k) => {
                    let r = ring.as_ref().expect("armed above");
                    k.write(&logits, r, pos)?;
                    if pos + 1 == max_new { break; }
                    // A view of ring[pos]: the gather reads the id where the
                    // kernel wrote it (EmbedOp honours the layout offset,
                    // fused_cuda.rs:1240-1271).
                    r.narrow(0, pos, 1)?.reshape((1, 1))?
                }
                TokenLoop::OnDevCandle => {
                    let t = logits.i((0, 0))?.argmax_keepdim(D::Minus1)?; // [1] u32 on the device
                    kept.push(t.clone());
                    if pos + 1 == max_new { break; }
                    t.reshape((1, 1))?
                }
            };
            h = self.hidden_cached(&next, offset, &mut caches, &mut NoCapture)?;
            offset += 1;
        }
        match kind {
            TokenLoop::F16 => Ok(out),
            TokenLoop::Key(k) => {
                if k.nan_seen()? {
                    candle_core::bail!("generate_loop: a NaN logit; the key argmax refuses to pick a token");
                }
                Ok(out)
            }
            // The one wait of the decode: a pageable D2H, which drains the stream.
            TokenLoop::OnDevKey(k) => {
                let v = ring.expect("armed above").to_vec1::<u32>()?;
                if k.nan_seen()? {
                    candle_core::bail!("generate_loop: a NaN logit; the key argmax refuses to pick a token");
                }
                Ok(v)
            }
            TokenLoop::OnDevCandle => Tensor::cat(&kept, 0)?.to_vec1::<u32>(),
        }
    }

    // The EOS form (bin/gsm8k, bin/chat), sketched, not written: lag one step.
    //   loop: write ring[pos]; enqueue step pos+1 from the ring view;
    //         async copy ring[pos] into a pinned 2-slot buffer + record an event;
    //         wait on the event of ring[pos-1] (complete or nearly: the device is
    //         one step ahead) and test it for EOS.
    //   on EOS at pos-1: step pos's work is wasted (one step a generation) and its
    //   KV position is dropped with `truncate(offset)`, O(1) on Grow and Prealloc,
    //   a prefix copy on Cat (model.rs:421-446). cudarc's `alloc_pinned` is
    //   write-combined (core.rs:1404-1427): fine for a 4-byte read; `as_slice`
    //   blocks on the event.
}

/// The CPU stand-in of the key kernel for host tests: the replay of
/// argmax_key_ref.rs on the logits (on the CPU device this is no readback).
#[cfg(test)]
struct HostKey;

#[cfg(test)]
impl KeyKernel for HostKey {
    fn write(&self, logits: &Tensor, ring: &Tensor, pos: usize) -> Result<()> {
        let bits: Vec<u16> = logits
            .i((0, 0))?
            .to_dtype(DType::F16)?
            .to_vec1::<half::f16>()?
            .iter()
            .map(|x| x.to_bits())
            .collect();
        // The tiny model's vocabulary is below 1,024, where the key's
        // precondition fails: the CPU stand-in falls back to the first index of
        // the maximum, which is what candle's CPU argmax returns.
        let id = if bits.len() >= 1024 {
            crate::argmax_key::key_argmax(&bits).0
        } else {
            logits.i((0, 0))?.argmax(D::Minus1)?.to_scalar::<u32>()?
        };
        ring.slice_set(&Tensor::new(&[id], ring.device())?, 0, pos)
    }
    fn arm(&self) -> Result<()> {
        Ok(())
    }
    fn nan_seen(&self) -> Result<bool> {
        Ok(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// On the CPU tiny model, every kind returns `generate`'s tokens on Cat,
    /// Grow and Prealloc, under the three attention forms, in f32 and f16, with
    /// groups of 2, 4 and 5. It pins the plumbing (ring, views, offsets, the
    /// dropped cast, the stop before the last upload), not the key, which
    /// argmax_key_ref.rs and the card check certify.
    #[test]
    fn token_loops_match_generate() {
        use crate::kvq::{AttnHeads, KvStore};
        let key = HostKey;
        for dtype in [DType::F32, DType::F16] {
            for (n_heads, n_kv) in [(4, 2), (8, 2), (10, 2)] {
                // (10, 2) is a group of 5, the 14B's shape: no test runs one today.
                let mut m = tiny_model_heads(dtype, n_heads, n_kv);
                let prompt = [3u32, 17, 42, 7];
                for store in [KvStore::Cat, KvStore::Grow(4), KvStore::Prealloc(32)] {
                    for heads in [AttnHeads::Expanded, AttnHeads::Shared, AttnHeads::Grouped] {
                        m.set_kv_store(store);
                        m.set_attn_heads(heads);
                        let want = m.generate(&prompt, 9, &mut NoCapture).unwrap();
                        for (name, kind) in [
                            ("f16", TokenLoop::F16),
                            ("key", TokenLoop::Key(&key)),
                            ("ondev-key", TokenLoop::OnDevKey(&key)),
                            ("ondev-candle", TokenLoop::OnDevCandle),
                        ] {
                            let got = m.generate_loop(&prompt, 9, &kind).unwrap();
                            assert_eq!(got, want, "{name} {dtype:?} {n_heads}/{n_kv} {store:?} {heads:?}");
                        }
                    }
                }
            }
        }
    }
    // Mutants, each must fail the test above: OnDevKey feeds ring.narrow(0, 0, 1)
    // (always the first token); drop `offset += 1`; OnDevKey writes ring[pos + 1];
    // break before the last write; OnDevCandle embeds kept[0].
}
