# Preregistration. The batched decode on one L40S: the GEMM's gate at 4B, and the bench at 4B, 8B and 14B against AWQ

**Written on 2026-11-08.** Stamped with OpenTimestamps before the first measurement, then never edited; a deviation
goes beside it in `proofs/preregistration-batch-l40s-2026-11-08-ECARTS.md`.

Operator brief of 2026-11-08 (relayed by the session that opened the branch): batching is the project's priority, "je
veux pouvoir régler le batch pour défoncer la L40S". No card budget was granted for this branch yet: the cost below is
announced, and nothing launches before the operator's go. Cost: **$2.31 expected, $4.05 at worst** on l40sx1, two jobs
(*estimated*, section 7). Branch total before these runs: $0.

**Base commit:** the commit of `claude/batch-l40s` that carries this file, `ops/jobs/batch-l40s.sh` and
`ops/jobs/batch-l40s-awq.sh`. **Image:** the Space sha built from it, passed as `IMAGE_SHA`; the launcher refuses
another. The AWQ job runs in `vllm/vllm-openai:v0.26.0` at its digest, as batch-awq-2026-11-07.

## 1. Why these runs exist

- The starting point is measured (batch-tetra-2026-11-07, batch-awq-2026-11-07, 4B, 256 tokens after the door
  prompt): the first tensor-core kernel (`mma`) reads 63.9 / 449.7 / 788.4 / 1,256.5 tok/s aggregate at B = 1 / 8 / 16
  / 32, 54.2 GB/s at B = 32; AWQ in vLLM reads 203.0 / 1,524.6 / 2,883.2 / 5,128.7, 427.9 GB/s at B = 32 (*measured*).
- This branch adds `kernels/tv_batch_gemm.cu` (`gemm`): the weight words of a K chunk copied into shared memory by
  cp.async two chunks ahead, each block decoded once per CTA into f16 weights with every scale folded in, one MMA pass
  with the activation rounded to f16, up to 128 rows, and the projections that read one activation (q, k, v; gate,
  up) in one launch. Host-proved against a double reference under the worst-case bound of its two f16 roundings, 17
  of 17 mutants killed (`tests/batch_gemm.rs`).
- The attention takes a pool of slots (`attn_slot_g4_u64`, `attn_slot_g5_u64`), host-proved bit for bit against the
  one-row entry. `llvq serve` batches continuously under `LLVQ_BATCH` through `configs/qwen3-*-tetra-e4-batch.json`.
- `gemm` is not bit for bit with batch 1 (sum order, two f16 roundings). Its licence is the T8 band of CLAUDE.md:
  paired decode-path perplexity inside +-0.12 %, and the GSM8K pilot.

## 2. What runs

Two jobs on l40sx1. Every Tetra command reads a config at its committed sha256.

**Job T, `ops/jobs/batch-l40s.sh`, timeout 90 min.** Oracle on Qwen3-0.6B (rule 10); the three sealed files at their
bytes and sha256 (`qwen3-4b-sealed.bin` 886391a8, `qwen3-8b-sealed-B.bin` 7bdb9a55, `qwen3-14b-sealed.bin` 61db37fe).
Then for each size, in the order 4B, 8B, 14B:

1. (a) the served door at batch 1, `fusedrun <sealed> 256` under the size's v9 config: its rate and its token
   fingerprint, the reference of (c);
2. (b) `batchrun check gemm` under the size's batch config: layer 0's seven projections and the head, then q, k, v and
   gate, up as the step launches them, at 1, 8, 13, 32, 64 and 128 rows, the GEMM against the multi-row path;
3. (c) `batchrun door 1,8,16,32,64,128 <arms> 256 3`: B copies of the door prompt fed through the batch, 256 tokens a
   sequence, one warm-up then 3 rounds; arms `gemm,mma` at 4B (`mma` up to 32, its widest entry), `gemm` at 8B and
   14B; `LLVQ_BATCH_REF` the fingerprint of (a);
4. (d) `batchrun ppl 2048 16 gemm`: the first 16 windows of 2,048 of WikiText-2 test, every position a decode row,
   rows against gemm (the 16-row entry). At 4B also `batchrun ppl 256 128 gemm`, the first 128 windows of 256 (the
   128-row entry, the other K chunk);
5. (e) at 4B only, `batchrun gsm8k 50 gemm 50`: the 50 pilot problems in one batch of 50 (the 64-row entry), each
   prompt prefilled on the batch-1 path, max 1,024 new tokens, against the served pilot dump (`served-v4`, sha256
   `a7203fa6…`, 40/50).

**Job A, `ops/jobs/batch-l40s-awq.sh`, timeout 45 min.** `ops/awq_batch.py --size 4b|8b|14b --batches
1,8,16,32,64,128`: the official AWQ w4g128 Qwen3 at the revisions `ops/awq_speed.py` pins, vLLM 0.26, f16, prefix
caching off, same prompt ids and token count, 2 warm-ups then 5 rounds.

## 3. Gates that void the run

1. The image builds; `oracle` prints MATCH; no `LLVQ_*` in the job environment.
2. The objects and the six configs at their bytes and sha256.
3. The 4B served door decodes fingerprint `61d0654532d35120`.
4. The GSM8K dump ends on fingerprint `94c7b98876288901` (the problem stream).
5. Every kernel loads with 0 B of local memory (`batch_cuda.rs` refuses otherwise).

## 4. The verdicts these runs read

- **The gate of the batched decode (4B), all three required:**
  1. `check gemm`: worst relative RMS under 1e-2 at every row count and every group (a wrong index reads near 1);
  2. T8 band: pooled paired perplexity of gemm against rows inside +-0.12 % at 16 x 2,048 and at 128 x 256;
  3. GSM8K pilot: correct count not below the served 40/50 by more than 2, reported with its discordant pairs.
  The door rows equal to the batch-1 fingerprint are reported, not gated: `gemm` does not owe them.
- **The kernel against the starting point:** gemm aggregate tok/s above mma at B = 8, 16 and 32 in the same job,
  medians with ranges not overlapping.
- **Against AWQ, at each size and each B:** aggregate tok/s and tok/s a sequence on both stacks, and GB/s = weights
  on the card x steps per second (Tetra 1.38 / 2.76 / 5.04 GB, AWQ 2.67 / 6.10 / 9.98 GB, `docs/ETAT.md`), *computed*.
  No x is formed across the two stacks (rule 5).
- **8B and 14B:** `check gemm` under 1e-2 and the T8 band at 16 x 2,048 license their batch configs; their door rows
  equal to their served fingerprint are reported.

## 5. Outcomes reported, never gated

- The served door at batch 1 per size (4E v9 rate and fingerprint).
- Per size, B and arm: aggregate tok/s, median and range, steps per second, tok/s a sequence.
- `check`'s relative RMS by projection, group and row count, and each entry's registers and shared bytes.
- Per window: perplexity under both arms and the paired difference; pooled; window mean and its 95 % interval.
- GSM8K: correct count, completions identical to the served dump, discordant pairs, wall time.
- AWQ per size and B: aggregate tok/s, range, tok/s a sequence, GB/s, whether its rows are equal.

## 6. Signed predictions

*Estimated*. A gemm step is the eager path's launches (about 20 a layer, host-bound near 4 to 5 ms on the 4B) plus
projections near the bandwidth up to B = 32, then the tensor cores (one f16 pass, about 1 TFLOP a step at B = 128 on
the 4B). Uncertain by a factor of two.

| quantity | point | interval |
|---|---|---|
| check gemm, worst relative RMS | **1e-3** | [1e-4, 5e-3] |
| 4B gemm aggregate tok/s at B = 1 / 8 / 16 / 32 / 64 / 128 | **110 / 800 / 1,450 / 2,600 / 4,300 / 5,800** | [70, 160] / [450, 1,300] / [800, 2,300] / [1,400, 4,000] / [2,300, 6,800] / [3,000, 10,000] |
| 4B mma, same job, B = 32 | **1,250** | [1,100, 1,400] |
| 8B gemm aggregate tok/s at B = 1 / 32 / 128 | **75 / 1,700 / 3,300** | [45, 110] / [900, 2,800] / [1,600, 6,000] |
| 14B gemm aggregate tok/s at B = 1 / 32 / 128 | **45 / 1,000 / 1,900** | [25, 70] / [500, 1,700] / [900, 3,500] |
| AWQ 4B aggregate tok/s at B = 64 / 128 | **8,500 / 13,000** | [6,000, 11,000] / [8,000, 18,000] |
| AWQ 8B / 14B at B = 1 | **123 / 78** | [100, 140] / [60, 90] |
| pooled ppl, gemm against rows, 4B, both widths | **+0.01 %** | [-0.06, +0.08] |
| GSM8K gemm: correct | **40/50** | [37, 43] |
| 4B door rows equal to the reference under gemm | **some, not all** | |

## 7. Cost

Job T: image pull and 14 model loads about 15 min; the 4B's five parts about 15 min, the 8B's four about 13, the
14B's four about 20 (the multi-row arm of the perplexity is most of it). About 55 min at $1.80/h: **$1.65**. Timeout
90 min: **$2.70** at worst. Job A: image pull (10 GB) and three loads about 15 min, the rounds about 7: about 22 min,
**$0.66**; timeout 45 min, **$1.35** at worst. Together **$2.31 expected, $4.05 at worst** (*estimated*). The Space
rebuild bills no card.
