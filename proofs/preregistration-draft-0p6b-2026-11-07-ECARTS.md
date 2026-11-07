# Deviations from `preregistration-draft-0p6b-2026-11-07.md`

Written on 2026-11-07 after the run, beside the stamped prereg, which is not edited. Journal:
`docs/mesures/draft-0p6b-2026-11-07.txt`.

1. **The sealed file carries two embedding tables, 0.339 GB against 0.23 [0.21, 0.26]: miss.** The Qwen3-0.6B
   checkpoint ships `lm_head.weight` beside `model.embed_tokens.weight` although its config ties them. `seal` carries
   every tensor the encoder did not quantize, and `embedq` turned both into int4 g64 (87.5 MB each). The loader reads
   the embedding only (`fused::take_embed_tables_at` under `tie_word_embeddings = true`), so 0.0875 GB of the file is
   never read. Without it the file would be 0.252 GB (*computed*), inside the interval. The file was not re-sealed:
   the drafter's tokens are never served, and the card memory the gate reads is what counts.
2. **b/param.** `rtbits` counts 751.6 M parameters, both tables, and reads 3.389 b/param for the whole model (f16
   tail, the served accounting). Over the 596.1 M parameters the model reads, 3.10 (*computed*), inside
   [2.95, 3.25].
3. **Perplexity 40.30 against 26 [21, 34]: miss, above.** The f16 checkpoint reads 18.93 in the encoder's own f32
   evaluation and the encoded model 39.69, a factor 2.10 (*measured*, `smoke.txt`). The prediction leaned on the 4B,
   whose Tetra cost is 3 %; at 0.6B the cost is a factor two. It changes nothing in the targets' tokens, only the
   acceptance rate the card gate reads.
4. **Added, not preregistered:** the f16 checkpoint through `ppl` on the same protocol (Metal, f16, 12 x 4,096), to
   place the 40.30.
