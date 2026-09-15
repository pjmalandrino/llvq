# Draft: Tetra gain choices and conditional continuation

This draft authorizes no model measurement. It is not timestamped.
The operator authorized implementation and software tests on 2026-09-13.

## 1. Question and scope

Does conditional Schur ranking choose the better continuation more often than Euclidean post-shape gain selection?
All choices share one Tetra direction, one compensated input, one factor and the same gain grid.
The future continuation always uses the source-norm policy.
This tests alignment with a quadratic proxy and reserved projection outputs, not MMLU.

The pilot uses checkpoint-prefix activations. It does not simulate sequentially quantized Transformer blocks.
A favorable result may justify a separate sequential-model experiment, never automatic adoption.

## 2. Fixed inputs

| field | proposed value |
|---|---|
| checkpoint | Qwen/Qwen3-0.6B, `c1899de289a04d12100db370d81485cdf75e47ca` |
| corpus snapshot | Salesforce/wikitext, `b08601e04326c79dfdd32d625aee71d232d685c3` |
| calibration | train split, 8 windows of 256 tokens per seed |
| reserved validation | validation split, 2 windows of 256 tokens per seed |
| seeds | 1 and 2; exact token IDs and offsets retained |
| text sampling | shuffled non-overlapping token chunks from the first 2,000,000 characters at most |
| Transformer layers | 0, 13, 27 |
| projections | `self_attn.q_proj`, `mlp.gate_proj`; v_proj excluded |
| rows | four evenly spaced indices over each projection's output rows |
| branch sites | three evenly spaced full blocks along each row |
| activations | original checkpoint, Metal, f32; backend oracle required |
| replay | CPU f64, fast-linalg; source-norm witness and continuations |
| rotation | input rotation, base seed `0x110feed`, existing site-derived seed |
| metric | shrinkage 1.0; damping 0.01 relative to mean diagonal |
| representation | Tetra, matrix-fitted two gains; fixed row scale; KeepExact tail |
| excluded | group scales, Design C, resume, extra directions, teacher targets, fine-tuning |

The preparation script materializes this table without model inference.
Freeze the source revision, working diff, binary SHA-256 and prepared input SHA-256 manifest before capture.
Timestamp this settled document and the manifest; never edit either afterwards.

## 3. Outcomes

For every shadow block, retain A/B/C choices, both gains and both local costs.
Only A advances the witness. For each branch site, retain both complete independent continuations.
Primary outcome: C minus B in mean regularized rollout selection regret, per projection, depth and seed.
Secondary outcome: the same difference for reserved-output selection regret.
Also report A, disagreements, absolute losses, global bounds, rollout excess and wall time.
Do not divide by a zero regret. Relative reductions are descriptive when the baseline is positive.

The snapshot sample is fixed before observing disagreements.
Each projection-depth-seed cell gets its own result; blocks do not supply independent error bars.
The calibration draws share a corpus and checkpoint. They are not independent model replications.

## 4. Signed prediction and interpretation

Prediction, unmeasured: C reduces mean rollout regret relative to B, with a modest 0–10% relative change.
No signed prediction is made for MMLU or reserved-output improvement.
This forecast is a working hypothesis, not an adoption threshold.

If C lowers rollout regret in both seeds, report which projection-depth cells agree or disagree.
If reserved-output regret also falls in both seeds, propose a separate sequential-model test.
If rollout regret falls but reserved-output regret rises, report proxy disagreement and stop expansion.
If the signs differ across seeds, report instability without selecting the favorable seed.
If no choices differ, report zero observed disagreement in this fixed sample.
Otherwise, report the pilot as inconclusive and return to the operator.

## 5. Budget and stopping

Compute resources: local Mac only; no rented job and no download.
Time proposal: 5–15 minutes total for both seeds, with a 30-minute ceiling (*estimated*, not measured).
The estimate includes checkpoint load, oracle, activation capture, retained-input writes and CPU replay.
It is deliberately conservative; no scaling from the toy correctness fixture is claimed.

There are 48 sampled rows and 144 branch sites across both seeds (*computed* from section 2).
The encoder budget is at most 7,968 calls (*computed*: 42 shadow calls plus 124 suffix calls per row).
Zero blocks may bypass direction search. No second search compares the two immediate gains.
Array storage is priced by `inspect`; peak resident memory is recorded by `/usr/bin/time -l`.
Stop on oracle failure, invalid input, corruption, non-finite loss, a violated bound or the time ceiling.
Keep partial outputs and the failure log. Do not change the sample or increase the ceiling automatically.

## 6. Retention and next authorization

Retain plans, token IDs, corpus fingerprints, code provenance, arrays, row traces and all raw timing logs.
Record actual duration and zero remote expenditure after the run.
Instructions: [local guide](../docs/tetra-schur-local.md).
Pending: operator approval of the checkpoint-prefix scope, sampling plan and total ceiling, then timestamping.
