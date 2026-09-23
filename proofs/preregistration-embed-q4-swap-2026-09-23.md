# Preregistration. The embedding pays for int4 tables: does the swap win MMLU?

**Written and committed on 2026-09-23, BEFORE the run. To be TIMESTAMPED (`ots stamp`) before
launch; the arm does not start on an unstamped file.**
Operator go for "step 1" given 2026-09-23. Cost: about 24 min on l40sx1, **$0.72**, timeout 1 h
(*estimated* from the two census arms of the same shape, jobs `6aae8c53` and `6aaf099f`,
24 min and $0.72 each). The `embedq` pass adds CPU minutes on the same card, not a second job.
Prerequisite outside the bill: the image republished with `embedq` in it (`ops/Dockerfile.cuda`,
this commit).

## 1. The claim on trial

The embedding is the one memory item the 4B can shrink without touching the kernel: q8 g64 to
q4 g64 frees **194.5 MB**, **0.3868 b/param** (*computed*, 388,956,160 weights at 8.5 then
4.5 b/weight). The swap spends it on int4 tables:

| item | b/param | kernel b/weight | MB |
|---|---|---|---|
| served object, DCLM + trained row scales | 2.7475 | 2.2044 | |
| embedding q8 g64 → q4 g64 | −0.3868 | 0 (outside the kernel) | −194.5 |
| `o_proj`, 36 layers, int4 g128 | +0.1971 | +0.2182 | +99.1 |
| `down_proj@12-23`, int4 g128 | +0.1560 | +0.1727 | +78.5 |
| **the package** | **2.7138** | **2.5953** | **−16.9** |

All *computed*: the int4 rows are the differences the 2026-09-19 journal
(`docs/mesures/dclm-down1223-2026-09-19.txt`) prints for the same types on the same codes. The
package is net negative on the file axis and costs 0.391 kernel b/weight, leaving 0.405 under
the triplet's b_max of 3.00.

The question is whether the package **wins on MMLU** against the object that reads **61.11**.
Its two halves have never been measured together, nor on this base:

- q4 embedding: **−0.35 pp**, on a `Planes14` base, under the 0.43 pp bar, so undetected
  (`docs/ROADMAP-QUALITY.md` row 4); +1.52 % perplexity, detected.
- `o_proj` int4: **+1.55 pp** on the C4 `tetra-q5` base (job `6aaab349`).
- `down_proj@12-23` int4: **+1.37 pp** on the DCLM base before row-scale training (job
  `6aae8c53`).

## 2. The arm

One file, one load, a census.

1. `embedq` on the scored object `/out/dclm-ft-2026-09-19/qwen3-4b-dclm-ft.bin`
   (sha256 `f8c1c903b753fe34...`), mode `q4`: every matrix record copied undecoded, the
   embedding rewritten int4 g64. Its matrix section must be byte-identical to the input's.
2. `mmlu` on the output, dense reconstruction, f16, flat, census, with
   `LLVQ_RESTORE_Q4=o_proj,down_proj@12-23` from `Qwen/Qwen3-4B`.

Paired with `docs/mesures/dclm-ft-mmlu-2026-09-20-brut/mmlu-4b-dclm-ft-FULL.csv` by `mmlupair`.

## 3. Signed prediction

| quantity | point | interval |
|---|---|---|
| paired gain over 61.11 | **+1.2 pp** | [+0.2, +2.2] |
| micro, full split | 62.3 | [61.3, 63.3] |

Reasoning, component by component. The embedding at −0.4 [−1.0, +0.2]: the only measurement is
on another base and undetected. The two tables transfer less than their measured gains, because
the row-scale training already repaired error that int4 would otherwise remove, and because
`LLVQ_RESTORE_Q4` discards the trained scales of the 48 matrices it replaces. The 2026-09-19
arm transferred 57 % of a C4 ceiling to DCLM; I take 60 to 70 % again, then 85 % of the sum for
the overlap of two types in the same residual stream: (0.9 + 1.0) × 0.85 ≈ +1.6, minus 0.4.

Named against me: a gain **above +3.0** (the raw sum of the three measured parts, 1.55 + 1.37
− 0.35 = 2.57, plus the bar) would mean the training did not overlap with int4 at all.

## 4. The decision rule

Census on both sides, constant codes: the 0.43 pp bar applies, and McNemar exact decides
significance.

| paired gain | reading | next |
|---|---|---|
| ≥ +0.43 and p < 0.05 | **The swap wins.** | Step 2: `EmbedMode::Q4`, the two q4 head kernels, then the int4 records transplanted into a real file (L37) and the served arm scored through the kernel |
| −0.43 to +0.43 | Neutral: 16.9 MB saved is not worth two kernels | Stop; record the pair |
| < −0.43 | The embedding costs more than the tables pay | Stop; the embedding stays q8 |

## 5. Controls

1. `oracle` first, hard rule 10.
2. `embedq` must print `216 lattice + 36 int4 records passed through undecoded`, one tensor
   requantized, `model.embed_tokens.weight`, 388,956,160 weights, and a file shorter than the
   input by 559.1 MB (f16 to int4 g64, *computed*: 777.9 → 218.8 MB).
3. The arm's log must declare **48 matrices, 676,331,520 weights** restored (36 × 2560 × 4096 +
   12 × 9728 × 2560).
4. 14,042 questions, plan fingerprint `a74a6d6213602979`.
5. sha256 of the input and output files in the journal.

## 6. What it will not establish

- The share of each half. One arm scores the package; the decomposition would take a second
  arm (tables alone on the f16 embedding, $0.72), not included.
- The q8 baseline. 61.11 was measured on the dense path with the file's **f16** embedding, so
  this pair measures f16 → q4 plus the tables, not the q8 → q4 of the served object. The q8 to
  f16 gap on this object is not measured.
- Nothing about serving it: the embedding is dequantized at load and the int4 tables are
  checkpoint tensors round-tripped through the shipped quantizer, dense. No q4 head kernel
  exists, and no int4 kernel has run on `o_proj`'s or `down_proj`'s shapes.
