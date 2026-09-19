# Tetra diagnostic: progress

What limits `Tetra`: its representation, its encoder, or their interaction with the model. The
target is a better trade of functional quality against served memory and speed, not a lower
geometric error and not crossing 60 MMLU. Opened 2026-09-17 on branch
`claude/tetra-quality-map-plan-t8ojq2`, which is never merged into `main` by operator decision.

Running cost: **$13.50 spent** on ten card jobs of 2026-09-18, everything else $0. About 10 h of Mac.

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

## The base, as of 2026-09-19

**The new reference is the DCLM-calibrated Q5 file**, operator decision of 2026-09-19:
`~/qwen3-4b-dclm.bin`, 1,794,564,765 bytes, sha256 `471f3988`, **57.95 micro** on the full
split at **2.2044 kernel b/weight and 2.8126 b/param**. It is not served: no `configs/` entry
names it and its codes were never published.

Every arm from now on is read against 57.95, not 56.37.

## The MMLU line, 2026-09-18

| arm | kernel b/weight | b/param | MMLU micro | source |
|---|---|---|---|---|
| f16, 4B | 16.000 | 16.000 | **70.14** | [f16-full](mesures/f16-full-2026-09-18.txt) |
| bare Tetra, 4B | 2.1498 | 2.7645 | 54.64 | [tetra-nu](mesures/tetra-nu-full-2026-09-18.txt) |
| **Q5 + DCLM, the new base** | **2.2044** | **2.8126** | **57.95** | [dclm-4b](mesures/dclm-4b-2026-09-18.txt) |
| Q5 + DCLM at 4x volume | 2.2044 | 2.8126 | 56.76 | [dclm-v4](mesures/dclm-v4-2026-09-18.txt) |
| Q5 served | 2.2044 | 2.8126 | 56.37 | census of 09-11 |
| + o_proj | 2.4226 | 3.0097 | 58.07 | [oproj](mesures/oproj-int4-full-2026-09-17.txt) |
| + down_proj | 2.7226 | 3.2807 | 60.44 | [downproj](mesures/downproj-int4-full-2026-09-18.txt) |
| **+ o + down** | **2.9408** | 3.4778 | **61.76** | [vod](mesures/vod-int4-full-2026-09-18.txt) |
| **bare Tetra + trained row scales** | **2.1498** | **2.7645** | **58.94** | [tetranu-rowscales](mesures/tetranu-rowscales-2026-09-19.txt) |
| bare Tetra, 8B | 2.1498 | 3.0672 | 63.85 | [vod-8b](mesures/vod-8b-2026-09-18.txt) |
| **8B + v + o + down** | **2.9260** | n/a | **68.03** | same |

Four things the afternoon settled.

1. **The gains add to 96.3 %** between types, and to **86.9 %** between slices of one type. A
   knapsack over matrices is a legitimate model, and it should carry the 86.9.
2. **`down_proj`'s gain is concentrated**: the middle twelve layers return +2.41 pp for a third
   of the bits, **13.95 pp per kernel b/weight against 7.82** for the whole type. The late
   twelve change 3.9 % of the model's answers.
3. **The lever fades with size**: 7.32 pp per b/weight at the 4B, **5.39 at the 8B**. Buying
   quality with bits has a decreasing return in scale, which is a problem at 14B and beyond.
4. **The f16 reference exists**, so every gap in this repository is now a same-population
   quantity: 13.77 pp and not 13.95.

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
