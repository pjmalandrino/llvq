# Tetra gain decision: revised plan

The submitted plan is executable in its first third and not measurable in its last third. This
document replaces it. It keeps the oracle on the two gains, cuts the propagation map and the
closed loop, and adds one lead the submitted plan does not carry.

Dated 2026-09-14, written against `7d62cff`. It is not a timestamped prereg and it authorises
no spend and no run.

## 1. What changed and why

The submitted plan prices its first stage from an unverifiable pilot figure and its last stage
from an MMLU endpoint the repository cannot resolve. Both prices are wrong in opposite
directions.

The instrument is roughly a hundred times cheaper than the plan assumes, because output rows
are independent in the GPTQ loop. The endpoint is unreachable, because every arm that
re-encodes carries 2.92 pp of MMLU noise (*measured*, [ROADMAP-QUALITY](ROADMAP-QUALITY.md)),
while the largest quality lever ever measured on this file is Q5 at +3.47 pp.

| Lot of the submitted plan | Verdict | Reason |
|---|---|---|
| 1, exact replay and isolation | Keep, reduced to one day | Per-row snapshot is a few kilobytes; four of its ten tests already exist |
| 2, two-gain oracle | Keep, it is the core | The gain decision is one bit and the direction does not depend on it |
| 3, regret and predictive power | Keep, on the layer proxy only | The proxy is exact and noiseless; the model endpoints are neither |
| 4, propagation map E/I/C | Cut | Weeks of hybrid forwards on a hypothesis that has not yet earned them |
| 5, decision to future effect | Cut | Same, and it depends on Lot 4 |
| 6, predictor | Defer | Two candidates and one bit leave almost nothing to learn |
| 7, closed loop | Defer behind row A | Its endpoint does not exist until the MMLU sampling plan changes |

One stage is added, stage 3 below, and it is the only part of this programme with a
first-order effect on reconstruction.

## 2. Corrections of fact

The submitted plan asks the agent to locate components that are absent, and does not use the
ones that are present.

| Claim in the submitted plan | State of the repository, 2026-09-14 |
|---|---|
| A direction-conditioned arm named `tetrapost` | No occurrence in any file |
| An existing Schur computation and Schur pilot | No occurrence in any `.rs`; the derivation exists in [hypothese-metrique-tetra](hypothese-metrique-tetra-2026-09-08.md) §"The metric needed at a block decision" |
| Pilot entries under `/Users/pjmalandrino/tetra-schur-pilot-2026-09-14` | Not reachable from a Linux session; not verified |
| 10.77 s of pilot cost | No trace; unusable as a basis under hard rule 8 |
| Blocks and KeepExact to be identified | Present, `llvq-quant/src/gptq.rs:260` |

Three assets the submitted plan renames instead of citing. The Schur scoring primitive is
`GptqFactor::solve_block`, which already computes the triangular solve whose squared norm is
the conditional cost. Its algebraic identity is already checked by
[checks.py](recherche-quantification-2026-09-08/checks.py). The A/B/C arm table already exists,
with the same three definitions, in [hypothese-metrique-tetra](hypothese-metrique-tetra-2026-09-08.md)
§"A staged discrimination protocol".

## 3. The structural fact that sets every cost

Output rows are independent inside `quantize_layer`, so a decision branch is the replay of one
row and not of a model.

Four properties make the snapshot small (*measured*, source read on `7d62cff`):

- the compensation touches only row `i` (`llvq-quant/src/gptq.rs:323-334`), and `solve_block`
  loops over rows without coupling them (`llvq-quant/src/linalg.rs`);
- row scales are frozen before the block loop (`llvq-quant/src/gptq.rs:251-253`);
- the factor `U` is read only, and no random number generator runs inside the loop; the only
  seed is `Rotation::new`, built before;
- the objective `tr(EHEᵀ)` is separable by row (`proxy_loss`, `llvq-quant/src/gptq.rs:501-520`).

`parallel_matches_serial_exactly` (`llvq-quant/tests/g5_gptq.rs:848`) already locks the first
property, which is why the submitted plan's tests 2, 3, 5 and 10 are mostly free.

The state at a decision boundary is therefore the row's remaining columns, its frozen scale and
the block cursor. The submitted plan's §3.1 lists nine items including solver state and random
state. Seven of them are constant across branches.

Cost consequence. The 4B encodes 151,388,160 blocks in 2 h 27 on the Mac (*computed* on the
3,633,315,840 projection weights of [fiche-4b](fiche-4b.md) and the 2 h 27 of
[ROADMAP-QUALITY](ROADMAP-QUALITY.md)), which is 17,164 blocks per second. A branch replays at
most 405 blocks, the length of a `down_proj` row, and 106 on every other matrix. Ten thousand
sites at two branches cost at most 8.1 million block encodes, so under 8 minutes of the same
machine (*computed*). The submitted plan's §13 warns against extrapolating a cost it never
had.

## 4. Stage 0: the measurement that decides the programme

Cost zero, no model run, no GPU, one day. It runs before any code for stages 1 to 4.

The current rule A picks the centroid nearest `‖x‖ / row_scale` (`llvq-quant/src/quantizer.rs:777`).
The Euclidean optimum over the two admissible gains is the centroid nearest `⟨x,u⟩ / row_scale`,
where `u` is the unit direction of the decoded Tetra point
([hypothese-metrique-tetra](hypothese-metrique-tetra-2026-09-08.md) §"Gain selection is
independently testable"). Since `⟨x,u⟩ ≤ ‖x‖`, rule B is a one-sided shift toward the lower
level and never a symmetric perturbation.

Two rules disagree exactly when the midpoint of the two centroids falls between `⟨x,u⟩` and
`‖x‖`. The width of that band is unknown and is measurable on data the repository already knows
how to produce.

Measure, on blocks dumped by `llvq-llm/examples/f1recdump.rs` re-run under `Codebook::Tetra`:

1. the disagreement rate between A and B, per matrix family and per column depth;
2. the distribution of `mid − ⟨x,u⟩` on the disagreed blocks, which bounds the per-block gain;
3. the same two quantities for rule C, using `solve_block` on the retained factor.

Decision rule, written before the measurement:

| Disagreement rate | Action |
|---|---|
| Under 1 % | Stop. Record the negative result and close the family, row 12 included |
| 1 % to 10 % | Continue to stage 1, with the measured rate as the basis for the site budget |
| Over 10 % | Continue, and raise stage 3 to first priority |

The submitted plan places this test in its Lot 3, after the whole harness is written. It
belongs first, because it can close the programme for one day of work.

## 5. Stage 1: the branch harness

One to two days, zero dollars, no model run.

Implement, inside `llvq-quant`, behind a feature or a test-only module so the served path keeps
`forbid(unsafe_code)` and its current behaviour:

- `RowSnapshot { cols: Vec<f64>, row_scale: f64, cursor: usize }`, taken at a block boundary;
- `replay(snapshot, factor, quantizer, forced_gain) -> RowOutcome`, which applies one forced
  gain at the cursor then continues to the end of the row with the unmodified policy;
- `RowOutcome { proxy: f64, codes: Vec<BlockCode>, gains: Vec<u32> }`, where `proxy` is the
  row's term of `tr(EHEᵀ)`.

Tests, all of them cheap and synthetic:

1. a replay with the policy's own gain reproduces the witness row bit for bit;
2. branch evaluation order does not change any outcome;
3. evaluating one branch mutates no other row and no shared factor;
4. a forced gain round-trips through `reconstruct_shape_gain`;
5. a row whose suffix is empty, and a row that ends on a `KeepExact` tail, are handled by the
   format's own rule and are excluded from the primary analysis;
6. any non-finite proxy fails loudly and is never ranked.

Tolerance: bit equality on the replay path, since the loop holds no random state and the
arithmetic is `f64` on one thread. A tolerance that is not bit equality means the harness is
wrong.

Gate G1. No result is interpretable until tests 1 to 3 pass and a deliberate mutation of the
replay breaks them.

## 6. Stage 2: three rules, one regret table

One day after stage 1, zero dollars.

Sites are drawn before any score is computed, stratified by matrix family and by column depth,
excluding blocks with no full block after them. The three rules are selectors over the same two
actions, so a site where two rules agree shares one replay.

For each site and each of the two gains, the harness reports the row proxy after continuation.
Regret is the difference to the better of the two, and it is exact. There is no confidence
interval to compute on a deterministic quantity, and none is reported.

Report, in this order: the number of sites, the number of sites where the two gains differ in
proxy, the disagreement rate of each rule against the oracle, the mean and the worst-case
regret of A, B and C, and the paired differences C−A, C−B and B−A in absolute proxy units
before any percentage.

Aggregation unit is the row, not the block. Blocks inside a row are sequentially dependent by
construction.

Gate G2. Continue only if C beats A on mean regret by a margin that survives being restated per
matrix family. A rule that wins on `down_proj` and loses on `v_proj` is a finding about
`down_proj`, not a policy.

## 7. Stage 3: the centroids, which is where the first-order effect is

This stage is absent from the submitted plan. It is the strongest lead the code audit produced.

`fit_gain_centroids` runs Lloyd-Max on `‖block‖ / row_scale` of the rotated weights, before the
GPTQ loop starts (`llvq-llm/src/calib.rs:840`). Two mismatches follow, both of them structural
and both zero bits.

The decision at encode time sees the compensated block, whose norm has drifted through the
error feedback of every earlier column. The centroids were fitted on the uncompensated
distribution. And under rule B or C the estimator that the decision compares against the
centroids is `⟨x,u⟩`, not `‖x‖`, so the two levels sit at the wrong place for the rule that
reads them.

The selection rule moves one block only when it sits near the crossover, which is a
second-order effect. The centroids move every block's reconstruction, which is first order.
Refitting them on the correct statistic costs one extra scan at encode time, changes no format
byte, and is a slice of row 17 of [ROADMAP-QUALITY](ROADMAP-QUALITY.md), rated +2 to +5 pp
*estimated*.

Measure on the stage 0 dump, at zero cost: the shift in the fitted centroids when the fit uses
the compensated blocks and `⟨x,u⟩`, and the resulting change in mean squared block error at
fixed direction and fixed rate.

## 8. Stage 4: one encoding arm, and what it can say

Gated on G2 and on stage 3. Two hours and a half of Mac per arm, zero dollars
([ROADMAP-QUALITY](ROADMAP-QUALITY.md)).

The endpoint is perplexity and held-out NLL, not MMLU. The reason is measured and it is on
file. `leech0c13` reads 19.6093 of perplexity against `Tetra`'s 16.1569, which is 21.5 % worse
at an identical rate, and its MMLU reads 54.67 against 53.49, CI95 [−1.54; +3.98] (*measured*,
[ETAT](ETAT.md) §5 nonies and [ROADMAP-QUALITY](ROADMAP-QUALITY.md) row 3). A 21.5 % move in
perplexity bought no MMLU point that the protocol can see.

A gain-selection change will move perplexity by single-digit percent at best. Claiming MMLU
from it requires row A of [ROADMAP-QUALITY](ROADMAP-QUALITY.md), the proportional sampling
plan, which takes the sampling error from 1.339 to 0.916 pp for half a day and zero dollars.
Until row A lands, an MMLU arm on this lead reports noise.

A prereg with an `ots` stamp comes before this stage and not before the earlier ones, since
stages 0 to 3 measure no fundamental criterion (hard rules 1 and 2,
[METHODE](METHODE.md) §1).

## 9. The three precedents, and why this lead is not one of them

The repository holds three cases where a better local proxy composed worse: design C at ×1.99
of perplexity, `group_scales` at 44.66 to 53.60, and gptq2 at 24.74 % of MMLU, which is chance
(*measured*, [ETAT](ETAT.md) §7). Row 12 of [ROADMAP-QUALITY](ROADMAP-QUALITY.md), the K-best
beam, is rated indeterminate and possibly negative for the same reason.

The three failures share a mechanism that rule C does not have. Each of them changed the
procedure around the decision: design C runs a free-magnitude loop then a post-hoc solve with
no compensation after the final snap, `group_scales` rescales blocks after their codes are
chosen, and gptq2 changes the target vector. Rule C changes which of two already-admissible
codes is written, and leaves the compensation loop, the row scale, the direction and the format
untouched.

Rule C is also the argmin of the objective the loop already optimises, under a continuous
relaxation of the suffix. Optimising the correct local objective is a different act from
optimising the wrong one harder. That argument predicts nothing about model quality, and it is
the reason the stage 4 gate is on perplexity rather than on the proxy.

## 10. Budget

| Stage | Wall time | Dollars | Machine | Prereg |
|---|---|---|---|---|
| 0, disagreement rate | 1 day | 0 | any CPU | none needed |
| 1, branch harness | 1 to 2 days | 0 | any CPU | none needed |
| 2, regret table | 1 day | 0 | any CPU | none needed |
| 3, centroid refit, measured | 0.5 day | 0 | any CPU | none needed |
| 4, one encoding arm | 2 h 27 of Mac per arm | 0 | Mac | stamped, before the first second |

Total before any decision that costs money: four to five days of development and zero dollars.
Stages 0 to 3 run on a laptop and need no checkpoint, no GPU and no artifact.

Stop conditions, written now: a disagreement rate under 1 % at stage 0, a regret that does not
survive per-family restatement at stage 2, or a centroid shift under the encoder's own
reproducibility at stage 3.

## 11. Artifacts

Repository conventions govern, not the submitted plan's parallel scheme. A journal goes in
`docs/mesures/` under its date, tabular results in `docs/data/` with an entry in
`docs/data/README.md`, and any stamped prereg in `proofs/`. No new manifest format is
introduced for a programme that writes no artifact and spends nothing.

Two files are enough for stages 0 to 3: one journal with the commands actually run and their
raw output, and one CSV of sites with their per-rule proxies and regrets.

## 12. What a positive result can claim, and what it cannot

A win at stage 2 establishes that rule C selects better codes under the objective GPTQ already
optimises. It establishes nothing about perplexity and nothing about MMLU.

A win at stage 4 establishes a perplexity movement on one model, one calibration set and one
seed. The repository's own dissociation makes that insufficient for a quality claim
([ROADMAP-QUALITY](ROADMAP-QUALITY.md) §"What the gain column shows").

The instrument itself is the durable output. Today, answering "does candidate rule X help?"
costs 2 h 27 of encoding plus an MMLU arm at 2.92 pp of noise, and the answer comes back
unresolved. After stage 1 it costs minutes of CPU and returns an exact regret. Row 12, row D
and any future candidate-selection question inherit that, whatever the verdict on rule C.
