# Tetra diagnostic: progress

What limits `Tetra`: its representation, its encoder, or their interaction with the model. The
target is a better trade of functional quality against served memory and speed, not a lower
geometric error and not crossing 60 MMLU. Opened 2026-09-17 on branch
`claude/tetra-quality-map-plan-t8ojq2`, which is never merged into `main` by operator decision.

Running cost: **$1.42 spent** on the `down_proj` confirmation of 2026-09-18, everything else $0. About 1 h 30 of Mac and 4.8 GB of disk.

| # | step | status | what it produced | cost |
|---|---|---|---|---|
| 1 | Reconcile the state | **done** 09-17 | [etat-reconcilie-2026-09-17](etat-reconcilie-2026-09-17.md). Four distinct rho named; the int4 budget redone in kernel b/weight, where every type fits | $0, 4 h |
| 2 | Fix the references and the accounting | **done** 09-18 | [references-comptabilite-2026-09-18](mesures/references-comptabilite-2026-09-18.txt). Both references fingerprinted, accounting read off the files; the served object's recorded sha256 was its prereg's | $0, 2 bucket reads |
| 3 | A reusable set of real dumps | **done** 09-18 | [tetra-diag-4b-2026-09-18](mesures/tetra-diag-4b-2026-09-18.txt). 30 cells, 120 rows at the 4B, three arms; the pilot's volume inflates held-out regret 2.8 to 19 times | $0, 5 min, 4.7 GB |
| 4 | Tetra against two geometric references | **done** 09-18 | [arbitrage-e8](arbitrage-e8-2026-09-18.md) frames it; [etape0](mesures/e8-etape0-2026-09-18.txt) verifies G(E8) to +0.002 %; [etape1](mesures/e8-etape1-2026-09-18.txt) is void (E3) and returns the isotropic finding; [etape1bis](mesures/e8-etape1bis-2026-09-18.txt) settles it inside the loop: **J_B/J_A = 1.047** at 2.6 % more bits, 1.290 at 3.7 % fewer, **1.146 at equal rate** against a predicted 1.09 | $0, ~1 h of Mac |
| 5 | The rho factorial, if it is a new experiment | **reframed by step 1** | rho_H cannot be varied alone: selection and compensation come from one `GptqFactor` | to be costed |
| 6a | `tetrahist` and a concrete compaction | **not started** | nothing exists under that name | to be costed |
| 6b | Q6a, distilling the format's free parameters | **not started** | row 17 of `ROADMAP-QUALITY`, +2 to +5 pp *estimated*, ~18 M parameters, zero bits | audit owed first |
| 7 | One ambitious branch | **waiting on 4** | the choice follows the diagnostics | n/a |

## The MMLU line, 2026-09-18

`down_proj` at int4 is confirmed at **+4.05 pp** on 11,762 held-out questions, CI95
[+3.28; +4.83] ([downproj-int4-full](mesures/downproj-int4-full-2026-09-18.txt)). The served
object goes from 56.37 to **60.44 micro** on the full split, against the paper's LLVQ at 60.7.
The gap to the paper was −4.33 pp a week ago and is **−0.26**.

It was unblocked by step 1: the arm had been dropped on a budget computed in the wrong unit. In
kernel b/weight it is 2.2044 to 2.7226, under b_max, on one condition: a native int4 kernel for
9728 x 2560, which has never run.

| arm | kernel b/weight | b/param whole | MMLU micro | held-out gain |
|---|---|---|---|---|
| v, shipped | 2.2044 | 2.8126 | 56.37 | reference |
| v + o | 2.4226 | 3.0097 | 58.07 | +1.55 |
| v + down | 2.7226 | 3.2807 | **60.44** | **+4.05** |
| v + o + down | 2.9408 | 3.4778 | not measured | not measured |

Per bit: `down_proj` returns 7.82 pp per kernel b/weight against `o_proj`'s 7.10.

## What step 4 settled, and where it points

Inside the production loop, at equal bits, E8 cubed costs **1.146** of Tetra's Hessian-weighted
error (*computed* from two measured arms). The arbitration dossier predicted 1.09 from the ratio
of normalized second moments. The lattice is worth what theory says and no more, and that the
bound holds at 2 bits per dimension with shape-gain, a regime it was not derived for, was not
obvious before the run.

Against that, the compensation moves the same metric by a factor of **7.5**. So the leads that
touch H, the visit order and the calibration volume outrank the ones that touch the codebook,
and that reorders step 7.

E8 survives as a **simplification** candidate, not a quality one: 455 kB of table and a table
scan against 7,138 lines and a trellis, for 4.7 % of weighted error at 2.6 % more bits. Whether
that trade is worth taking is a kernel question this diagnostic does not answer.

## What the next step should read

The compensation, not the codebook. Stage 1 measured that the served residual costs **6.8 times
less** than a random residual of its own energy, while carrying 40 % more energy than a plain
E8 cubed encode at 3.7 % fewer bits. A factor of 6.8 sits in where the error goes, against a
bound of 1.09 on how much error the lattice makes. The lattice is the smaller question.

Two runs follow, both $0 and both on the existing dump set.

1. **Stage 1 proper**: arm B inside the same GPTQ loop, sharing the Schur factor, the gain
   levels and the row scale, so only the codebook differs. New prereg. Prediction already on
   the record: arm B's ratio to isotropic falls from 0.996 toward 0.148, and the residue is the
   lattice.
2. The alternative-directions run of 2026-09-16 recomputed on the v64 arm. It read its
   -8.763 % at 2,048 calibration tokens, where the held-out regret is 2.8 to 19 times its
   high-volume value.

And read the v64 arm, never the low-volume one: at 2,048 tokens every 4B Hessian in the set is
rank-deficient, down to 0.21 samples per dimension on `down_proj`.

## Decisions waiting on the operator

1. **The two adoption gates.** o_proj needs a `kind = 2` writer and a native int4 kernel on
   2560 x 4096, which has only ever run on 1024 x 2560. `down_proj` needs the held-out
   confirmation o_proj received, and its budget objection is withdrawn.
2. **Step 4, the E8 option.** Three options are costed in the dossier. Option A is ~1 day and
   ~300 lines in `llvq-bench`, because E8's norm-10 codebook is 455 kB and brute-forces, where
   Λ₂₄ needed 7,138 lines. It also verifies two constants the repository cites without
   transcribing.
3. **PGMR's letters.** Filed as the gain centroid rescaling axis, which is the only lever in
   the record that gained perplexity and was abandoned. A second reading is possible: the
   pilot carries a `projected_gain` field, and the projection gain rule is a different lever
   that COST 4 % of perplexity.

## Debt noticed and left alone

Ten files of `docs/data/` predate this work and are declared nowhere. `configs/README.md`
carries 6 em dashes and `HISTORIQUE.md` 49, against the ban in `STYLE.md`. `ETAT.md` is dated
2026-09-06 and runs 560 lines against a target of 100.
