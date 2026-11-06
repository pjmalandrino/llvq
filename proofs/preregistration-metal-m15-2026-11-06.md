# Preregistration. M15: the sealed Tetra 4B on MLX, against mlx-lm's 4-bit, decode and prefill

Written on 2026-11-06 on branch `claude/metal-nuit-3c4257`. To be timestamped before its first measurement, then
never edited; departures go to `preregistration-metal-nuit-2026-11-06-ECARTS.md` under an M15 heading. Cost **$0**,
local, under the operator's go of 2026-11-06 (MLX as the engine, then "Préremplissage + rotation").

## 1. Why

`ops/mlx_tetra/llvq_mlx.py` at 54ffa22 serves the sealed 4B on MLX: our Tetra kernel (one row), the 8-row kernel, the
f16 decode plus MLX's GEMM for chunks of 9 rows or more, the rotation in one kernel, MLX's own layers for the int4
records, the embedding, the attention and the rest. Not bit for bit with the Rust engine (served under the M6
tolerance, decision 1 of `docs/plan-metal-v2.md`). `check.py` holds its kernels against each other: rotation within
1e-5, the 8-row kernel equal to the one-row kernel, the dense prefill within 7e-4 relative. Smoke runs, not quoted:
78.7 tok/s at 5 tokens, 3.3 s to read 4,096 tokens, peak 2.99 GB, and at 4,096 the Rust engine's 64 tokens.

## 2. What runs

The packed directory `~/q4b-hf-2026-09-28` (hfpack of the sealed 4B), `ops/mlx_tetra/llvq_mlx.py` and
`ops/mlx_bar.py` at 54ffa22, mlx-lm 0.28.3 by both headers; the rival at revision 4dcb3d10. nice 10, one after the
other, after 15 quiet minutes:

1. L1: `uv run ops/mlx_tetra/llvq_mlx.py <packed> time 256 7`
2. X1: `uv run ops/mlx_bar.py <rival> 256 7`
3. L2: `uv run ops/mlx_tetra/llvq_mlx.py <packed> time 64 5 4096`
4. X2: `uv run ops/mlx_bar.py <rival> 64 5 4096`

## 3. Gates

1. Prompt digests: 5d2eb20a at 5 tokens and 04e38430 at 4,096 on both sides.
2. Every round of a run returns its first round's tokens.
3. GPU under 20 % busy at the start of each run; otherwise the run is repeated once and the journal says so.

## 4. Rules

- Each side in tok/s with its range, GB/s of what it reads a token (computed), b/param for the whole model with its
  tables (ours 2.7, the rival about 4.5), and its peak memory. No ratio across the two stacks is published (rule 5).
- L2's tokens are compared with the Rust engine's at the same prompt (`7cbd553b…`, M8 to M14): equal or not, the
  journal says which; a difference is not a failure under the M6 tolerance.
- If L1 decodes at 75 tok/s or more and L2 reads its prompt in under 5 s, the MLX path becomes the bundle's Metal
  path (operator's go for the bundle's form).

## 5. Predictions, signed

*Estimated*, informed by the smoke runs.

| quantity | prediction |
|---|---|
| L1 decode | 74 to 84 tok/s |
| X1 decode | 115 to 130 tok/s |
| L2 decode | 55 to 70 tok/s |
| L2 first token | 2.9 to 3.8 s |
| L2 peak memory | 2.8 to 3.3 GB |
| X2 decode | 90 to 110 tok/s |
| X2 first token | 2.9 to 5.0 s |
