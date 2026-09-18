# Tetra diagnostic: progress

What limits `Tetra`: its representation, its encoder, or their interaction with the model. The
target is a better trade of functional quality against served memory and speed, not a lower
geometric error and not crossing 60 MMLU. Opened 2026-09-17 on branch
`claude/tetra-quality-map-plan-t8ojq2`, which is never merged into `main` by operator decision.

Running cost: **$0**. No card, no job. About 15 min of Mac and 4.7 GB of disk.

| # | step | status | what it produced | cost |
|---|---|---|---|---|
| 1 | Reconcile the state | **done** 09-17 | [etat-reconcilie-2026-09-17](etat-reconcilie-2026-09-17.md). Four distinct rho named; the int4 budget redone in kernel b/weight, where every type fits | $0, 4 h |
| 2 | Fix the references and the accounting | **done** 09-18 | [references-comptabilite-2026-09-18](mesures/references-comptabilite-2026-09-18.txt). Both references fingerprinted, accounting read off the files; the served object's recorded sha256 was its prereg's | $0, 2 bucket reads |
| 3 | A reusable set of real dumps | **done** 09-18 | [tetra-diag-4b-2026-09-18](mesures/tetra-diag-4b-2026-09-18.txt). 30 cells, 120 rows at the 4B, three arms; the pilot's volume inflates held-out regret 2.8 to 19 times | $0, 5 min, 4.7 GB |
| 4 | Tetra against two geometric references | **blocked on code** | nothing. Three independent E8 codebooks do not exist in the repository | to be costed |
| 5 | The rho factorial, if it is a new experiment | **reframed by step 1** | rho_H cannot be varied alone: selection and compensation come from one `GptqFactor` | to be costed |
| 6a | `tetrahist` and a concrete compaction | **not started** | nothing exists under that name | to be costed |
| 6b | Q6a, distilling the format's free parameters | **not started** | row 17 of `ROADMAP-QUALITY`, +2 to +5 pp *estimated*, ~18 M parameters, zero bits | audit owed first |
| 7 | One ambitious branch | **waiting on 4** | the choice follows the diagnostics | n/a |

## What the next step should read

The v64 arm, not the low-volume one. At 2,048 calibration tokens every 4B Hessian in the set
is rank-deficient, down to 0.21 samples per dimension on `down_proj`, and the held-out regret
is 2.8 to 19 times its value at 16,384 tokens. The low-volume arm is the control that compares
to the 0.6B pilot, and nothing else.

The first cheap thing it makes possible: recompute the alternative-directions run of 2026-09-16
on the v64 arm. That run read its -8.763 % off the pilot's volume.

## Decisions waiting on the operator

1. **The two adoption gates.** o_proj needs a `kind = 2` writer and a native int4 kernel on
   2560 x 4096, which has only ever run on 1024 x 2560. `down_proj` needs the held-out
   confirmation o_proj received, and its budget objection is withdrawn.
2. **Step 4 without E8.** If a reference E8 implementation is not fundable, the geometric
   diagnostic reduces to `Tetra` against multi-shell Leech and loses its external comparison.
3. **PGMR's letters.** Filed as the gain centroid rescaling axis, which is the only lever in
   the record that gained perplexity and was abandoned. A second reading is possible: the
   pilot carries a `projected_gain` field, and the projection gain rule is a different lever
   that COST 4 % of perplexity.

## Debt noticed and left alone

Ten files of `docs/data/` predate this work and are declared nowhere. `configs/README.md`
carries 6 em dashes and `HISTORIQUE.md` 49, against the ban in `STYLE.md`. `ETAT.md` is dated
2026-09-06 and runs 560 lines against a target of 100.
