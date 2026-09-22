# Deviations. Preregistration census-8b-2026-09-21

The prereg is stamped (sha256 `a3c08d367fa78ed5...`) and not edited.

## E1. The f16 prediction missed its interval

Predicted 77.0 [75.5, 78.5]; measured 75.05. The point moved the 2,280-question sample (76.08)
up by half the shift the 8B `Tetra` showed between its sample and its census (2.24 pp). As
written, that recipe gives 77.2 for B and 74.1 for C (73.01); the prereg signed 77.0 and 74.5 and
does not record the step. The f16 checkpoint moved the other way, −1.03 pp. The harness control
rules out a harness change: the same weights give identical answers on the 2,280 shared
questions. The prereg already cited a precedent of that sign, the 4B f16 sample 0.18 pp over its
census (70.32 → 70.14), and took the sign from the 8B `Tetra` anyway.

## E2. The running time fell 0.7 min under the interval

Predicted 107 min [100, 115]; measured 99.3 min (5,959 s). Harmless: the cost was $2.98.

## E3. The launcher's bucket refusal was repaired after this job

The other launchers of the chain refused on `hf buckets ls` printing `(empty)` for a directory
that does not exist. This job's launcher checked for `mmlu-8b-` and was not affected. The fix is
in `ops/jobs/{bench-8b,served-8b,dclm-8b-rowscales,dclm-8b-ft-mmlu}.sh`.

## E4. A's b/param is computed, and it is not the accounting of the arm scored

The prereg labels A's 3.0683 *measured* by `rtbits`. It is *computed*: `rtbits` reads the file's
bytes but models the q8 embedding at 8.5 b/param. The file carries its embedding in f16 (16.0000
b/param, measured), and arm A scored it so (`config=none`): 4.2080 at the same accounting
(`docs/mesures/dclm-8b-2026-09-21-brut/rtbits.txt`). The journal gives both.

## E5. Two pairs published outside the prereg's list

The journal gives A against the bare 8B `Tetra` (+1.02 pp) and against the v+o+down int4 restore
(−3.16 pp). The prereg's published list does not hold them, and it names the bare 8B `Tetra`
under "not compared" as a format or corpus effect. The journal marks both descriptive only, not a
format effect.
