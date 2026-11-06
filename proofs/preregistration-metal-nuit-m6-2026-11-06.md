# Preregistration. M6: the split-KV decode-attention kernel on Metal, its speed and its T8

Written on 2026-11-06 at night on branch `claude/metal-nuit-3c4257`, once the code it describes existed (commit and
binary in §2). To be timestamped before its first measurement, then never edited; departures go to
`preregistration-metal-nuit-2026-11-06-ECARTS.md` under an M6 heading. Cost **$0**, local, under the operator's go for
this night. No one can answer during the run: every choice is fixed here.

## 1. Why

On the L40S the 4B is served with `attn_dec.cu`, the split-KV decode-attention kernel, from position 0 (operator T8 of
2026-11-05), licensed by a perplexity band against the grouped form. Metal had no such kernel. This branch adds
`kernels/attn_dec.metal`, whose arithmetic is the canonical one (`attn_ref.rs`): its output equals the CPU replay's,
hence the card kernel's, bit for bit, on the fixtures of `tests/attn_metal.rs`, and it is checked again against the
replay every time it is built. This run reads its speed against the grouped form on the sealed 4B, and its perplexity
against the grouped form, the card's T8 pairing.

## 2. What runs

Binary `metalrun` sha256 `547def2c8dfc272c05304e6a53d9e06551ef44b3d6ca5c21a8d1be64091a35f9`, built from the commit that
carries this file. Mutation run of the kernel before the stamp: 10 mutants, 10 killed (journal).
By path, nice 10, quiet machine (no build from 15 minutes before the first run to the last).

1. `metalrun gate <sealed> configs/qwen3-4b-tetra-e4.json lutd,lutd+grp,lutd+ker`: projections and head, 253 of 253;
   and the kernel's own self-check at build.
2. S1: `metalrun time <sealed> <config> lutd+grp,lutd+ker 256 7` (5-token prompt).
3. S2: `metalrun time <sealed> <config> lutd+grp,lutd+ker 64 5 2048`.
4. S3: `metalrun time <sealed> <config> lutd+grp,lutd+ker 64 5 4096`.
5. T8: `metalrun ppl <sealed> <config> lutd 4096 12 ker`: the first 12 WikiText-2 test windows of 4,096, decode path,
   grouped and kernel (from position 0) in lockstep.

`lutd+ker` reads the kernel from position 0, as the card's served 4B config does.

## 3. Gates

1. The kernel builds and passes its self-check (the binary stops otherwise).
2. Bit gate 253 of 253 for both arms against `lutd`.
3. Every arm returns the same tokens every round.

## 4. Rules

- **Speed**, per context, `lutd+ker / lutd+grp` decode, round by round: wholly above 1.000, the kernel is faster
  there; straddling, a wash; wholly below, slower.
- **T8 band**, the card's: the pooled `kernel / grouped - 1` and every window's inside [-0.12 %, +0.12 %].
- **The NOT SERVED Metal config**: if the band holds and the kernel is faster or a wash at 4,096, the plan names the
  kernel form as the Metal candidate; otherwise the grouped form. Either way a config file marked NOT SERVED is
  written for it, and `served.rs` keeps its refusal: lifting it is the operator's decision.
- **Tokens**: the kernel is not bit for bit with the grouped form by design (online softmax in f32, another reduction
  order). Token agreement is reported, not gated.
- **Otherwise** (self-check fails, a gate fails, a hang): journaled, the grouped form stays the candidate.

GSM8K, the card T8's second criterion, does not run: 1,319 problems do not fit in the night on this Mac. The plan says
so.

## 5. Predictions, signed

*Estimated*.

| quantity | prediction | why |
|---|---|---|
| `ker / grp` at 5 tokens | x0.97 to x1.05 | one launch pair a layer against about six candle ops, at a cache of 5 positions |
| `ker / grp` at 2,048 | x0.95 to x1.25 | the grouped form already reads the cache in place; the kernel saves launches and the f16 score round trips |
| `ker / grp` at 4,096 | x0.95 to x1.35 | as above, more positions |
| T8 pooled `ker / grp - 1` | inside ±0.05 % | the card read +0.0095 % for grouped against expanded and passed the kernel's band |
| top-1 agreement | 99.5 to 99.95 % | the card's kernel T8 prediction was 99.8 % |
