# Preregistration. The batched decode under the served v9 configs: 4E for a lone request, the GEMM at two CTAs a SM, and llvq serve under concurrent requests

**Written on 2026-11-08.** Stamped with OpenTimestamps before the first measurement, then never edited; a deviation
goes beside it in `proofs/preregistration-batch-l40s-v3-2026-11-08-ECARTS.md`.

Operator go of 2026-11-08 (relayed): two fixes, (1) a lone request on 4E instead of the GEMM, (2) the GEMM's occupancy,
inside the $5 envelope. Spent: $2.40 (batch-l40s, batch-l40s-awq, batch-l40s-v2). Cost: **$1.29 expected, $1.95 at
worst** (*estimated*, section 7); the branch stays under $5 at worst ($4.35).

**Base commit:** the commit of `claude/batch-l40s` that carries this file and `ops/jobs/batch-l40s-v3.sh`.
**Image:** the Space sha built from it, passed as `IMAGE_SHA`; the launcher refuses another.

## 1. Why this run exists

- batch-l40s-v2-2026-11-08: the GEMM reads 105.5 tok/s at B = 1 on the 4B against 223.4 for the served 4E door, and its
  128-row entry holds one CTA a SM (82 KB of shared memory).
- (1) `configs/qwen3-*-tetra-e4-4e-v9-batch.json`: the served v9 configs with a batch field. The GEMM now reads 4E v9's
  tiled int4 words (host-proved against the loader's own permutation, 5 of 5 mutants killed), so one object on the card
  serves both paths. Under `llvq serve` with `LLVQ_BATCH`, a request with no other live decodes through `Session` on 4E
  v9 and, when another arrives, its caches are copied into a slot and it steps with the GEMM (the CPU server test checks
  the tokens of a request that joins mid-way).
- (2) Every GEMM entry under 48 KB of shared memory and 128 registers (`__launch_bounds__(256, 2)`), so two CTAs share
  a SM at every width: K chunks of 192 / 96 / 96 / 48 / 48 at 8 / 16 / 32 / 64 / 128 rows. NVRTC 12.4, sm_89 and
  sm_80: no local memory.
- The client's units: `ops/batch_serve_bench.py` posts B concurrent `/api/generate` requests to `llvq serve`.

## 2. What runs

One job, l40sx1, `ops/jobs/batch-l40s-v3.sh`, timeout 65 min. Oracle; the three sealed files, the three v9-batch
configs and the bench script at their bytes and sha256. Then for each size, 4B, 8B, 14B, under its v9-batch config:

1. (a) `batchrun check gemm`: on the tiled int4 words, the GEMM against the multi-row path, 1 to 128 rows and the two
   groups; the log must name the int4 layout tiled;
2. (b) `batchrun door 1,8,32,64,128 gemm 256 3`, `LLVQ_BATCH_REF` the size's served fingerprint (4B 61d0654532d35120,
   8B c817aeaf2f5d1116, 14B dc4f9ab9aa24c4d3);
3. (c) at 4B, `batchrun ppl 2048 16 gemm`;
4. (d) `llvq pull <tag> --from <sealed>`, `llvq serve` under `LLVQ_BATCH=128:1024`, then `batch_serve_bench.py --batches
   1,8,16,32,64,128 --tokens 256 --rounds 2`: raw prompt "The capital of France is", greedy.

## 3. Gates that void the run

1. The image builds; `oracle` prints MATCH; no `LLVQ_*` in the job environment.
2. The objects, the configs and the bench at their bytes and sha256.
3. Every `check` log names the int4 layout tiled, and every kernel loads at 0 B of local memory.
4. Every request of the client bench answers 256 tokens.

## 4. The verdicts this run reads

- **The tiled path (all sizes):** check under 1e-2; at 4B the band, pooled perplexity of gemm against rows inside
  +-0.12 %. Door rows equal to the fingerprint reported.
- **Fix (1):** the client bench at B = 1 reads the lone request's decode rate; it holds if it is within 10 % of the
  served 4E door of batch-l40s (223.4 / 137.1 / 87.1 tok/s), and its answer equals the B = 1 answer of every other B
  where the GEMM's rows equal batch 1.
- **Fix (2):** door aggregate at B = 32, 64 and 128 against batch-l40s-v2's; kept if the 4B at B = 128 gains and no B
  loses more than 5 %.
- **Against AWQ:** aggregate tok/s, tok/s a request and GB/s at each size and B against batch-l40s-awq-2026-11-08 (no x
  across stacks, rule 5).

## 5. Outcomes reported, never gated

Per size and B: door aggregate, range, tok/s a sequence; client aggregate, range, tok/s a request (server clock),
answers equal to the B = 1 answer; check per projection; the server's lines (`batch:`, the kernel report).

## 6. Signed predictions

*Estimated*.

| quantity | point | interval |
|---|---|---|
| client B = 1, tok/s a request, 4B / 8B / 14B | **215 / 132 / 84** | [200, 230] / [123, 142] / [78, 90] |
| door 4B aggregate at B = 32 / 64 / 128 | **2,700 / 4,100 / 5,300** | [2,350, 3,200] / [3,450, 4,900] / [4,300, 6,800] |
| door 8B / 14B at B = 128 | **4,200 / 2,600** | [3,400, 5,200] / [2,100, 3,300] |
| client 4B aggregate at B = 8 / 32 / 128 | **720 / 2,300 / 4,800** | [550, 850] / [1,800, 2,900] / [3,500, 6,500] |
| check, worst relative RMS | **4.3e-4** | [2e-4, 1e-3] |
| pooled ppl, 4B, gemm against rows | **within +-0.02 %** | [-0.05, +0.05] |

## 7. Cost

Image pull and 9 model loads about 12 min (each `llvq serve` loads once, at its first request); per size check, door,
serve bench about 8 / 10 / 13 min, the 4B's perplexity about 3. About 43 min at $1.80/h: **$1.29**; timeout 65 min,
**$1.95** at worst (*estimated*).
