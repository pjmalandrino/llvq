# Preregistration. M14: the rival on this Mac, mlx-lm's Qwen3-4B 4-bit, against our best Metal arm

Written on 2026-11-06 on branch `claude/metal-nuit-3c4257`. To be timestamped before its first measurement, then
never edited; departures go to `preregistration-metal-nuit-2026-11-06-ECARTS.md` under an M14 heading. Cost **$0**,
local; the operator approved the 2.28 GB download on 2026-11-06 (decision 3 of `docs/plan-metal-v2.md`).

## 1. Why

Step 0 of `docs/plan-metal-v2.md`: the bar a Mac user already has, before choosing between our own engine and MLX
(decision 2). A smoke run of `ops/mlx_bar.py`, one timed round, not quoted, read 103 tok/s of decode and a 3.2 s first
token at 4,096 positions, on the same prompt digest as `metalrun`.

## 2. What runs

Model `mlx-community/Qwen3-4B-4bit` at revision `4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25` (2,263,022,529 B of
safetensors), `ops/mlx_bar.py` from 11af4f2 (mlx-lm 0.28.3 by its PEP 723 header). Our side: `metalrun` sha256
`2bf7ea0779a69c147ccff5bf9821481b07f61dc62e8ff73ad57c47df6dcf06aa`, built from 11af4f2's parent tree (`+r` included), arm `lutd+h+grp+pq+r`. nice 10, one after the other:

1. O1: `metalrun time <sealed> configs/qwen3-4b-tetra-e4.json lutd+h+grp+pq+r 256 7`
2. X1: `uv run ops/mlx_bar.py <model> 256 7`
3. O2: `metalrun time <sealed> configs/qwen3-4b-tetra-e4.json lutd+h+grp+pq+r 64 5 4096`
4. X2: `uv run ops/mlx_bar.py <model> 64 5 4096`

## 3. Gates

1. Both sides print the same prompt digest at 4,096 tokens (`04e38430…`) and the 5-token prompt's ids.
2. Every round of a side returns that side's first round's tokens.

## 4. Rules (CLAUDE.md rules 4 to 6)

- No ratio across the two stacks is published (rule 5). Each side is reported in tok/s with its range, and in GB/s of
  the bytes it reads a token (weights, plus K and V at 4,096), *computed*, and in b/param for the whole model with its
  tables (rule 6): ours 2.7, mlx-community 4-bit about 4.5.
- The reading feeds decision 2: if mlx-lm decodes at more than 1.5 times our rate at both contexts, the plan's engine
  step (B) is re-costed against MLX as the engine, and the operator decides.

## 5. Predictions, signed

*Estimated*, informed by the smoke run for X2.

| quantity | prediction |
|---|---|
| O1 decode | 60 to 68 tok/s |
| X1 decode | 100 to 130 tok/s |
| O2 decode | 38 to 44 tok/s |
| X2 decode | 95 to 110 tok/s |
| X2 first token | 2.8 to 3.6 s |
