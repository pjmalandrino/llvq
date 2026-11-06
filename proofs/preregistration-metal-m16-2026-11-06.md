# Preregistration. M16: the tuned MLX path against mlx-lm's 4-bit, decode and prefill

Written on 2026-11-06 on branch `claude/metal-nuit-3c4257`. To be timestamped before its first measurement, then
never edited; departures go to `preregistration-metal-nuit-2026-11-06-ECARTS.md` under an M16 heading. Cost **$0**,
local, under the operator's go of 2026-11-06 ("Oui, mesurer").

## 1. Why

Since M15 (81.6 tok/s), `ops/mlx_tetra/llvq_mlx.py` at 285ad5c stacks q and k in one dispatch, decodes 2 rows a
simdgroup against one activation load, runs gate, up and SiLU(g) * u in one dispatch, rotates with every group in
parallel, and uses threadgroups of 64. Each change kept the 256 greedy tokens of the 5-token prompt (b4f07db4) and
check.py's bounds; the stacked q and k equal q and k alone bit for bit. Smoke runs read about 94 tok/s before the
machine drifted (journal). This file measures it the way M15 did.

## 2. What runs

As M15 §2, with the code at 285ad5c: L1 `llvq_mlx.py <packed> time 256 7`, X1 `mlx_bar.py <rival> 256 7`, L2
`llvq_mlx.py <packed> time 64 5 4096`, X2 `mlx_bar.py <rival> 64 5 4096`. Three idle minutes before L1, and each run
starts once the GPU reads under 20 %. nice 10.

## 3. Gates

1. Prompt digests 5d2eb20a and 04e38430 on both sides.
2. Every round of a run returns its first round's tokens; L1's tokens are b4f07db4.
3. Within a run, the slowest timed round is no more than 10 % under the median; otherwise the machine drifted, the
   run is repeated once after three idle minutes, and the journal says so.

## 4. Rules

As M15 §4: each side in tok/s with its range, GB/s, b/param (ours 2.7, the rival about 4.5) and peak memory; no
ratio across stacks published. The MLX path keeps its place as the bundle's Metal path if L1 is at or above M15's
81.6 tok/s.

## 5. Predictions, signed

*Estimated*, informed by the smoke runs.

| quantity | prediction |
|---|---|
| L1 decode | 88 to 100 tok/s |
| X1 decode | 120 to 132 tok/s |
| L2 decode | 62 to 78 tok/s |
| L2 first token | 3.0 to 3.8 s |
| L2 peak memory | 2.8 to 3.3 GB |
| X2 decode | 95 to 108 tok/s |
