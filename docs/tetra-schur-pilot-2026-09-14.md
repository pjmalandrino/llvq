# Tetra gain choices: local pilot

Schur lowers mean gain-selection regret in both seeds, with only five changed decisions among the 144 branch sites.
Counts and losses are *measured*; relative changes are *computed* from the [raw journal](mesures/tetra-schur-pilot-2026-09-14.txt).
This is a small favorable signal. Prediction of the discrete suffix remains unresolved.

## Results

Qwen3-0.6B uses unchanged checkpoint activations, with separate Wikitext train and validation splits.
A selects gain from the source norm. B uses the direction-conditioned Euclidean projection. C uses the conditional Schur metric.
Selection regret is the loss above the better of the same two complete continuations.
The table averages each seed's six projection-depth cells equally, as preregistered.

| seed | rollout regret B | rollout regret C | change | reserved-output regret B | reserved-output regret C | change |
|---|---|---|---|---|---|---|
| 1 | 0.00333505 | 0.00318050 | −4.63% | 0.02754167 | 0.02749131 | −0.183% |
| 2 | 0.000238763 | 0.0000947134 | −60.33% | 0.02404296 | 0.02210811 | −8.05% |

Schur differs from B on 54 of 2,016 shadow blocks (2.68%). Only five differences land on scheduled branch sites.
One projection-depth cell improves in seed 1. Three cells change in seed 2: two improve rollout regret and one worsens it.
The other eight cells are unchanged. No population confidence interval is inferred from these correlated blocks.

B alone is unstable against A: rollout regret falls 31.47% in seed 1 and rises 92.17% in seed 2.
The signed forecast of a 0–10% Schur improvement matches seed 1; seed 2 lies outside its predicted magnitude.
The reductions describe selection regret. MMLU and full-model quality were not evaluated.

## What the suffix evidence says

Three changed sites are the last full block; only a continuous KeepExact tail remains.
At these sites the Schur bound is attainable, so their improvement does not establish prediction of future discrete decisions.
The two interior changes occur in seed 2. Layer 27 gate_proj improves rollout loss; layer 13 q_proj worsens it.
Both improve reserved-output loss. Seed 1 contains no changed interior branch site.
This positional audit is descriptive and was added after the run; it changes no primary outcome.

The preregistered rule permits proposing a sequential-model experiment because both mean regrets fall in both seeds.
A broader interior-block diagnostic on retained inputs would first address the weak suffix evidence.
No additional run or policy adoption was made.

## Execution and retention

The complete pilot took 10.77 seconds, with zero remote expenditure (*measured*, same journal).
The prior 5–15 minute estimate was too high. Both Metal oracles match exactly; all stages completed inside the ceiling.
Independent dense algebra verified every one of the 288 continuations from retained arrays.
Maximum absolute error on full quadratic loss was 1.78e-14; on direct Schur scores, 8.88e-16.

[Protocol](../proofs/preregistration-tetra-schur-2026-09-14.md) and [input manifest](../proofs/tetra-schur-inputs-2026-09-14.sha256) were timestamped before capture.
The receipts matched their documents, with Bitcoin anchoring still pending at launch.
[Per-cell CSV](data/tetra-schur-pilot-2026-09-14.csv) and [audit JSON](data/tetra-schur-pilot-2026-09-14.json) accompany the journal.
Full inputs and traces remain in `/Users/pjmalandrino/tetra-schur-pilot-2026-09-14`, with an output SHA-256 manifest.
