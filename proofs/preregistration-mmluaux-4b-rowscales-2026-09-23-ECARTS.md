# Deviations. The MMLU-format row-scale arm

The prereg beside this file is timestamped and is not edited. This records what happened
instead.

## É1. The arm was never run

Job `6ab360af51992417dfcd5b92` was launched on l40sx1 on 2026-09-23 and **cancelled by the
operator while still in `SCHEDULING`**, before any hardware was allocated. Billed **$0.00**; the
output directory `mmluaux-4b-rowscales-2026-09-23/` is empty and no object was written. The
repository's running total is unchanged at $183.80.

Reason given: the operator's target is memory, not the training corpus — reducing the model's
footprint and paying for it with int4 tables. That is the reallocation lead, not this one.

So the signed prediction of §3 is **unarbitrated**. It is not wrong and it is not right: nothing
was measured. Anyone re-opening this lead inherits the prediction as written.

## É2. What stands, and costs nothing to keep

- `ops/mmlu_aux_overlap.py` and its result: 0 items of `test`, `dev` or `validation` share
  question and choices with `auxiliary_train` (*measured*, 2026-09-23, revision c30699e).
- `--corpus dclm|mmlu-aux|mix` in `ops/llvqtune`, 111 tests, 16 mutants killed, and the
  capacity guard that refuses a plan outrunning a finite corpus.
- The measured size of the split: about 24.75 M tokens at 247.9 a question, 0.79 of a
  9,507-step run at seed 0.
- `ops/jobs/mmluaux-4b-{rowscales,fold,ft-mmlu}.sh`, dry-run clean, unlaunched.
- `llvqtune-2026-09-23.tgz` in the bucket, 104,379 bytes, sha256 `9d3417f6…`.

Re-opening costs the same $4.72 and no new code.
