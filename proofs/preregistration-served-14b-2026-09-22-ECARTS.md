# Deviations. Preregistration served-14b-2026-09-22

The prereg is stamped (sha256 `6488444e8c39b31d...`) and not edited.

## E1. The bench stage cannot run at 14B shapes

`planesbench` panicked before timing anything: `planesbench.rs:1917`,
"model.layers.0.mlp.down_proj.weight: d_in 17408 overruns the activation". The 14B's `down_proj`
reads 17,408 inputs and the bench's activation buffer is smaller. The stage is a bench-side limit,
not a fact about the object: the same file served 55.4 tok/s in the model minutes earlier in the
same job. The job's other three stages passed and their numbers stand (`stages.txt`: A ok, smoke
ok, bench FAILED rc=101, harness ok). The prereg's decision rule has no row for a stage that cannot
start; under "otherwise: not settled", the operator decides whether to pay the fix: a bench change,
an image rebuild ($0) and a rerun (~$0.90, *estimated*).

## E2. Arm A beat its interval by 4.2 pp

Predicted 68.3 [65.3, 71.2]; measured 72.53. The point carried the 8B base forward by the f16 gap
between the sizes; the 14B kept more than that rule allowed. The paired f16 − A also missed, +6.35
against +9.7 [+7.0, +12.5]. Both misses are in the object's favour. The running time, 65 min, fell
5 min under its interval.

## E3. The served decode job of this prereg has not run

`served-14b.sh` scores the trained file, which does not exist: the training exited 1 and the fold
is stopped by `preregistration-dclm-14b-rowscales-2026-09-22`'s decision rule, pending the
operator. The ceilings of this prereg's second job, $1.35, are not committed.

## E4. The two jobs of this prereg ran on different images from the reference census

Arms B and C ran on Space `a963a020`; arm A, the smoke and the harness ran on `af907416`, the
rebuild that adds `export` and `rowscale`. The prereg's harness control is what covers it, and it
passed: 2,280 of 2,280 identical picks and identical logits on the 8B base across the two images.
