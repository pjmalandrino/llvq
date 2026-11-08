# Preregistration. Batching on the sealed 4B: the starting point and the gate of the tensor-core kernel, one job

**Written on 2026-11-07.** Stamped with OpenTimestamps before the first measurement, then never edited; a deviation
goes beside it in `proofs/preregistration-batch-tetra-2026-11-07-ECARTS.md`.

Operator brief of 2026-11-07 (relayed by the session that opened the branch): launch the gate without asking again
while the branch total stays under $5. Cost: **$0.66 expected, $1.35 at worst** on l40sx1 (*estimated*, §7). Branch
total before this run: $0.08 (batch-awq, `docs/data/jobs.csv`).

**Base commit:** the commit of `claude/batching-eca5f1` that carries this file and `ops/jobs/batch-tetra.sh`.
**Image:** the Space sha built from it, passed as `IMAGE_SHA`; the launcher refuses another.

## 1. Why this run exists

- The AWQ side of the starting point is measured: 203 / 779 / 1,525 / 2,883 / 5,129 tok/s aggregate at B = 1 / 4 / 8
  / 16 / 32, vLLM, same prompt and token count (*measured*, batch-awq-2026-11-07,
  `proofs/preregistration-batch-baseline-2026-11-07.md`). Its §1 sends the Tetra side to this job.
- This branch adds a batched decode step (`Qwen3::batch_step`: B sequences, one cache slot each, one row each a step)
  with two kinds of projection. `rows`: the existing multi-row path (Tetra rows kernel at 4 rows a launch, int4 and
  head row by row), bit for bit with batch 1 by construction. `mma`: `kernels/tv_batch_mma.cu`, each block decoded
  once into shared memory as exact integers and multiplied for up to 32 rows on the tensor cores; the activation in
  two f16 halves; split K folded in a fixed order. The attention of both arms is `attn_dec_batch.cu`, the served
  kernel's body with one sequence per grid z, host-proved bit for bit against the one-row entry.
- `mma` is not bit for bit with batch 1 (sum order). Its licence is the T8 band of CLAUDE.md: paired perplexity
  inside +-0.12 %, and the GSM8K pilot.

## 2. What runs

One job, l40sx1, `ops/jobs/batch-tetra.sh`, timeout 45 min. The sealed 4B (`qwen3-4b-sealed.bin`, sha256
`886391a8…`) under `configs/qwen3-4b-tetra-e4-4e-v4.json` (the 4B's served config) at its committed sha256.

1. Oracle on Qwen3-0.6B (rule 10).
2. The served door at batch 1: `fusedrun <sealed> 256`.
3. `batchrun check`: layer 0's seven projections and the head, mma against rows, random activations, 1/8/16/32 rows.
4. `batchrun door 1,4,8,16,32 rows,mma 256 3`: B copies of the door prompt, the prompt fed through the batch, 256
   tokens a sequence, one warm-up then 3 rounds, every (B, arm) once a round; `LLVQ_BATCH_REF` the door fingerprint.
5. `batchrun ppl 2048 16`: the first 16 windows of 2,048 of WikiText-2 test, every position a decode row, both arms.
6. `batchrun gsm8k 16 rows 50`, then `... mma 50`: the 50 pilot problems, each prompt prefilled through the batch-1
   path and copied into its slot, the decode batched 16 at a time, max 1,024 new tokens.

## 3. Gates that void the run

1. The image builds; `oracle` prints MATCH; no `LLVQ_*` in the job environment.
2. The object and the config at their bytes and sha256.
3. The served door decodes fingerprint `61d0654532d35120`.
4. `batchrun check`: worst relative RMS under 1e-2 (a wrong index reads near 1).
5. Both GSM8K dumps end on fingerprint `94c7b98876288901`.

## 4. The verdicts this run reads

- **Serving path (rows arm)**: every row of every B equals `61d0654532d35120`, and the 50 pilot completions equal
  the served pilot dump (`served-v4`, sha256 `a7203fa6…`). Anything less is a defect of the batched path, not a
  tolerance.
- **The kernel gate (mma arm)**, all three required:
  1. aggregate tok/s above the rows arm at B = 8 and at B = 16 (median, ranges not overlapping);
  2. tokens: every door row equal to `61d0654532d35120`, or else the T8 band below;
  3. T8 band: pooled paired perplexity of mma against rows inside +-0.12 %, and the GSM8K pilot correct count not
     below the served 40/50 by more than 2 (reported with its discordant pairs; 50 problems separate nothing finer).
- **Against AWQ**: aggregate tok/s and GB/s at each B. GB/s = weights on the card x steps per second, 1.38 GB for
  Tetra (`docs/ETAT.md` §2), 2.67 GB for AWQ, *computed*. No x is formed across the two stacks (rule 5).

## 5. Outcomes reported, never gated

- The served door at batch 1 (4E and eager rates).
- Per B and arm: aggregate tok/s, median and range, steps per second, tok/s a sequence.
- `check`'s relative RMS by projection type and row count.
- Per window: perplexity under both arms and the paired difference; pooled; window mean and its 95 % interval.
- GSM8K: correct count, completions identical to the served dump, discordant pairs, wall time.

## 6. Signed predictions

*Estimated*. Rows: the eager step (7.9 ms, 126.8 tok/s at the door) plus the Tetra rows kernel at x1.2-1.35 a
launch of 4 rows, and the int4 records (84 of 252) and the head row by row. Mma: kernel time near batch-1 decode
cost for any B up to 32, plus per-sequence work that grows with B; uncertain by a factor of two.

| quantity | point | interval |
|---|---|---|
| served door, 4E, tok/s | **178** | [170, 186] |
| rows, aggregate tok/s at B = 1 / 4 / 8 / 16 / 32 | **120 / 250 / 290 / 300 / 300** | [95, 135] / [170, 360] / [180, 450] / [180, 500] / [180, 550] |
| mma, aggregate tok/s at B = 1 / 4 / 8 / 16 / 32 | **90 / 330 / 600 / 1,050 / 1,700** | [45, 135] / [150, 500] / [280, 1,000] / [450, 1,900] / [700, 3,400] |
| rows door rows equal to the reference | **all, every B** | |
| mma door rows equal to the reference | **not all** | |
| check, worst relative RMS | **3e-4** | [1e-5, 3e-3] |
| pooled ppl, mma against rows | **+0.00 %** | [-0.05, +0.05] |
| GSM8K rows: completions identical to served | **50/50** | |
| GSM8K mma: correct | **40/50** | [37, 43] |

## 7. Cost

Image pull and five model loads ~8 min; the door ~7 min; perplexity ~4 min; GSM8K ~4 min. About 22 min at $1.80/h:
**$0.66**. Timeout 45 min: **$1.35** at worst (*estimated*). The Space rebuild bills no card.
