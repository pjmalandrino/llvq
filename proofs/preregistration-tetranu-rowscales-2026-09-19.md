# Preregistration. Training the row scales of bare Tetra, the first reference

**Written, committed and TIMESTAMPED on 2026-09-19, BEFORE the run.**
Operator go given 2026-09-19. Cost: training **$0** on the Mac, about 11 h.
Scoring is a separate arm and a separate go, about **$0.72** on l40sx1.

## 1. The claim on trial

Row 14 of `docs/ROADMAP-QUALITY.md`. The paper's own note, transcribed in
`docs/llvq-paper-notes.md`: "The fine-tuning here is no more than learning the per-column
scales (< 0.001 bit/weight, ~52M tokens). It is not end-to-end training." The paper reads
**+2.1 pp** on Qwen3-4B, our exact model.

  bare Tetra, C4-calibrated     54.64 micro   2.1498 kernel b/weight   2.7645 b/param
  **+ trained row scales**      **?**         **2.1498**               **2.7645**

The rate does not move. Nothing is added to the file. Only the values of 1,069,056
`row_scales` change, and `rowscale` folds them back in place.

## 2. Why bare Tetra and not the served base

`export` refuses a mixed Tetra + Int4G128 file: it has no int4 decode path and would write a
directory missing every matrix it refused. The served DCLM base is mixed. Bare Tetra is 252
lattice records and exports cleanly, so it is the subject that exists today.

This is a **reference**, not a candidate. It answers whether the lever moves MMLU at all, on
the arm where the lever is cleanest, before any int4 record is in the way.

## 3. What is trained, and what cannot be

Per-row multipliers on `row_scales`, initialized at 1.0, on 216 matrices. `v_proj` is excluded
by default because it is int4 on the served object, so the reference is read on the same 216
matrices a served run could fold.

The gradient never reaches the decoded directions. The Leech decoder is a table lookup and is
not differentiated. `ops/llvqtune` refuses a direction tensor that carries a gradient, and the
guard is covered by two tests and killed as a mutant.

## 4. The budget, and the gap to the paper

| quantity | this run | the paper |
|---|---|---|
| tokens | **about 7.4 M** | 52 M |
| ratio to the trained parameters | 7 : 1 | 47 : 1 |

Measured throughput on MPS, 2026-09-19: 280 tokens a second under cross entropy at seq 1024
batch 2, and seq 2048 batch 1 is four times worse at equal tokens because attention is
quadratic in the length. KL adds one teacher forward to three units of student work, so the
run is priced at about 210 tokens a second.

So this reference is trained on **one seventh** of the paper's data. A weak result will not
separate "the lever is weak" from "the data is short", and that ambiguity is accepted here
because the run costs nothing.

## 5. The protocol

  objective       KL(dense || quantized), temperature 1
  corpus          DCLM-edu, the paper's own, streaming, seed 0
  steps           3,600 at seq 1024 batch 2, so 7,372,800 tokens
  schedule        warmup 100, cosine to 0.1 of the peak, peak 3e-4
  optimizer       AdamW, no weight decay
  checkpoint      every 300 steps, so a crash costs at most 300 steps

The corpus **streams**, so every batch is text the run has not seen. The logged loss is
therefore a held-out loss already, and no separate validation split is carved out.

## 6. Signed prediction

| quantity | point | interval |
|---|---|---|
| gain over bare Tetra | **+1.0 pp** | [−0.5, +2.5] |
| micro, full split | 55.64 | [54.14, 57.14] |

The point sits below the paper's +2.1 for two reasons, and the interval includes zero and
negative values because of the second.

**One seventh of the data.** Scales are few, 1,069,056 against 4.02 B weights, so they should
fit fast, but 7 tokens a parameter is thin.

**The axis may not be the paper's.** The paper says "per-column scales". Our format holds one
scale per output **row**. Whether those are the same object in their convention is not
established anywhere in this repository. If they are not, this run trains a different
parameter than the one that measured +2.1, and the honest prior is much weaker.

## 7. What refutes what

- Loss falls and MMLU does not move: the objective is not MMLU's, which is the fourth
  dissociation this repository has recorded, and the lever is not free quality.
- Loss falls and MMLU falls: DCLM-edu KL pulls away from what MMLU rewards.
- MMLU rises by more than the interval: the axis question of §6 is settled in our favour and
  the full 52 M budget on a card is worth its $5 to $10.
- The run diverges: the schedule is wrong, not the lever, and it is rerun at a lower peak.

Scoring is one full-split arm, 14,042 questions, fingerprint `a74a6d6213602979`, compared
against the existing bare Tetra dump on the dumps and not on two printed lines.
