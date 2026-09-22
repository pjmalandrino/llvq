# Preregistration. The input axis: RMSNorm weights trained beside the row scales

**DRAFT, not stamped.** To become binding it is renamed
`preregistration-dclm-rownorms-2026-09-22.md`, committed and `ots stamp`ed **before** the
training job starts; its sha256 then goes into the header of both job scripts. Any change after
stamping goes into a `-ECARTS.md` beside it.

Operator go: **not given yet.** Announced cost: training about $4.00 on l40sx1, scoring about
$0.72, so **about $4.72** (priced on `data/jobs.csv` rows 171 and 172, the same two jobs for the
control arm).

## 1. The claim on trial

The paper's fine-tuning is "learning the per-column scales" (`docs/llvq-paper-notes.md:107`).
The 61.11 arm trained one scale per output **row** and nothing on the input side; its own
prereg (`preregistration-tetranu-rowscales-2026-09-19.md` §6) left the axis open.

The RMSNorm weights are per-column scales on the input of q/k/v (`input_layernorm`) and gate/up
(`post_attention_layernorm`), applied before the rotation (`llvq-llm/src/model.rs:1369`), and
`model.norm` scales the columns of the tied language head. `bin/seal` carries all of them
verbatim in f16 (`llvq-llm/src/bin/seal.rs:11-14`). Training them costs **0 bits**: the fold
rewrites values the file already holds, and the byte count does not move.

  control     DCLM base + trained row scales             61.11 micro   2.7475 b/param
  **arm**     DCLM base + trained row scales + norms     **?**         **2.7475**

The two arms differ by `--mode` alone: same export (`/out/dclm-export-2026-09-19`), same
teacher, objective (KL, T = 1), corpus (DCLM-edu streaming, seed 0), wall budget (7,200 s),
learning rate (3e-4), card flavor (l40sx1). Trained values: 1,069,056 row scales in both arms,
plus 186,880 norm values in this one (2 × 36 × 2,560 + 2,560, *computed*).

## 2. What is already known, and what it does not say

- **On a toy**, the same mode leaves 2.4 times less held-out KL than row scales alone, on three
  seeds (`docs/mesures/row-norms-synthetique-2026-09-22.txt`). The toy code is 1 bit a weight
  in its direction, far cruder than Tetra, so that ratio does not transpose.
- **On the real object**, the row-scale KL plateaued: the median of the logged loss sits between
  0.218 and 0.228 from about step 3,500 to 9,450 (*computed* on
  `docs/mesures/dclm-rowscales-2026-09-19-brut/journal.jsonl`). The row degrees of freedom are
  saturated at this budget, which is why this arm adds degrees of freedom and not tokens.
- **The dissociation.** The 61.11 object reads perplexity ×1.0074 of f16 and MMLU nine points
  under it. A lower KL is not evidence of MMLU here, in either direction.

## 3. Signed prediction

| quantity | point | interval |
|---|---|---|
| paired gain over the 61.11 control | **+0.6 pp** | [−0.6, +2.0] |
| micro, full split | 61.7 | [60.5, 63.1] |

The point is small because the new degrees of freedom are diagonal like the row scales, 17 % of
their count, and shared by several matrices each, so the overlap should be large. The ceiling is
the paper's own fine-tuned line, 62.8, plus the draw noise the control arm itself carries.

## 4. Decision rule

Read on the dumps with `mmlupair`, census on both sides, fingerprint `a74a6d6213602979`, 14,042
questions. Δ is the paired micro difference, arm minus control; p is the exact McNemar.

| outcome | reading | action |
|---|---|---|
| Δ ≥ +1.0 and p < 0.01 | the input axis carries MMLU | the arm becomes the reference object; next, the `v_proj` int4 group scales, the last free parameter left untrained |
| +0.3 ≤ Δ < +1.0 and p < 0.01 | real and small | kept (0 bits); the axis does not reach the next level alone |
| \|Δ\| < 0.3, or p ≥ 0.01 | the natural-basis input axis adds nothing | closed; if the paper's "columns" are an axis, it is the rotated basis, L28's +0.0038 b/param variant |
| Δ ≤ −0.3 and p < 0.01 | KL on the norms costs MMLU | not kept; the dissociation of §2 now runs against us, recorded as such |

## 5. Controls, in order, each a stop condition

1. **Routing.** The probe's `first_loss` reads the control's **0.35267** within 1 %. Tau and
   sigma start at 1, so step one is the same model on the same batch. Outside: stop the job.
2. **Fold idempotence, on the real file.** An export of this run with every value replaced by
   1.0, folded by `rowscale` into `~/qwen3-4b-dclm.bin`, is byte-identical to it (`cmp`).
3. **Fold width.** The folded file is 1,794,564,765 bytes, the DCLM base's count, and
   `rowscale` prints 216 scaled, 36 int4 passed through, 73 norms of 73 named.
4. **Oracle.** `oracle` MATCH on cuda in the scoring job (hard rule 10).

## 6. Known defects carried

- **E8 of the control arm**: the trainer splits columns in the natural basis while the tail is
  stored rotated, bounded at 1.2e-4 relative, unrepaired. Same in both arms.
- **Step count.** It is derived from each job's own probe rate. The extra multiply per norm makes
  this arm slightly slower, so it may see slightly fewer tokens than the control's 19,470,336.
  Recorded, not corrected: §2 says the budget was not the constraint.
- **f16 rounding of the fold.** `w * tau` is rounded to f16 once; relative error at most 2^-11
  per value, the width the file has always held the norms in.
- **One seed, one draw.** A paired census removes the question sample, not the training draw.
  No run measures the spread of the training itself on this base.

## 7. Secondary, not gates

- Perplexity on the Mac, the protocol of `docs/fiche-4b.md` §5, $0. Reported beside the result,
  never used to read it (§2).
- The served path, `fusedrun` at the served flags (the `dclm-ft-fusedrun` job, ~$0.30), only if
  §4 keeps the arm.
