# Preregistration. Batching baseline, AWQ side: Qwen3-4B AWQ w4g128 in vLLM at B = 1, 4, 8, 16, 32 on one L40S

**Written on 2026-11-07.** Stamped with OpenTimestamps before the first measurement, then never edited; a deviation
goes beside it in `proofs/preregistration-batch-baseline-2026-11-07-ECARTS.md`.

Operator brief of 2026-11-07, relayed by the session that opened the batching branch: measure the starting point
against AWQ, launch without asking again under $1. Cost: **$0.30 expected, $0.75 at worst** on l40sx1 (*estimated*,
§6). Project total before this run: about $278 (*computed*: $274.71 in `docs/data/jobs.csv` plus the three v9 doors in
flight, ~$3.3).

**Base commit:** the commit of `claude/batching-eca5f1` that carries this file, `ops/awq_batch.py` and
`ops/jobs/batch-awq.sh`. **Image:** `vllm/vllm-openai:v0.26.0`, digest `sha256:ffb2d59b…`, the image of the published
AWQ figures.

## 1. Why this run exists

- Every throughput figure of the project is batch 1. At batch 1 Tetra reads fewer bytes than AWQ and wins.
- At B sequences the weight stream is read once for B tokens, so the decode arithmetic decides, and AWQ's Marlin
  kernels run it on the tensor cores. No run has read AWQ's aggregate rate at B > 1 on our card.
- The Tetra side of the baseline (served path at B = 1, the existing multi-row kernel at B > 1) needs the batched
  serving path this branch writes. It runs in the Tetra job of this branch, under its own prereg, with the protocol of
  §2 (same prompt, same token count, same rounds). Two images cannot share one job.

## 2. What runs

One job, l40sx1, `ops/jobs/batch-awq.sh`, timeout 25 min. `ops/awq_batch.py` at the sha256 the launcher computes.

- `Qwen/Qwen3-4B-AWQ` at revision `74d4bd2b…`, the revision `ops/awq_dequant.py` validated; vLLM's default AWQ
  routing (Marlin on sm_89), dtype float16, prefix caching off and read back, `gpu_memory_utilization` 0.5.
- B copies of the door prompt, raw ids `[785, 6722, 315, 9625, 374]`, 256 greedy tokens each, `ignore_eos`.
- 2 discarded rounds, then 5 rounds; each round runs B = 1, 4, 8, 16, 32 once, in that order. Wall clock around one
  `generate` call, prefill included. Aggregate rate = B x 256 / elapsed.

## 3. Gates, each one voids the run

1. The script's sha256 inside the job equals the launcher's.
2. Prefix caching reads False after load.
3. Every call returns B sequences of exactly 256 tokens.

## 4. Outcomes reported, never gated

- Per B: aggregate tok/s, median and min-max over the 5 rounds; tok/s a sequence (median / B).
- Per B: GB/s = 2.67 GB (AWQ's weights on the card, `docs/ETAT.md` §3) x steps per second (median / B),
  *computed*. This is the quantity compared with Tetra (rule 5); no × is formed across the two stacks.
- Whether the B rows of a call are identical, and whether row 0 equals B = 1's row 0. Informative only: vLLM does not
  promise batch invariance.

## 5. Signed predictions

*Estimated*: B = 1 from the published 200.5 tok/s (128 tokens, ops/awq_speed.py); B > 1 from a memory-bound kernel
whose step time grows slowly until the Marlin tiles fill.

| B | aggregate tok/s, point | interval |
|---|---|---|
| 1 | **195** | [170, 215] |
| 4 | **700** | [550, 850] |
| 8 | **1,300** | [1,000, 1,650] |
| 16 | **2,300** | [1,700, 3,000] |
| 32 | **3,800** | [2,600, 5,200] |

## 6. Cost

Image pull ~10 GB and a 2.7 GB checkpoint, ~6 min; load and graph capture ~2 min; 7 rounds x 5 batch sizes, ~2 min.
About 10 min at $1.80/h: **$0.30**. Timeout 25 min: **$0.75** at worst (*estimated*).
