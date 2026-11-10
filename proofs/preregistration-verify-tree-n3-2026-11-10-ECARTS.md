# Deviations from prereg N3 (`preregistration-verify-tree-n3-2026-11-10.md`)

The prereg is not edited. A departure found after the measurement is written here with what it touches.

## Deviation 1: the job ended ERROR in its summary step, after the bench had finished

Written on 2026-11-10 at about 05:45 CET, after the job. Job `6af29bba9737d9a9e08eeaa5` ran `vtreebench` to its end
(`vtreebench exit 0`, every arm, both contexts, the full output in `vtree-4b.txt`). The n3 summary block of
`ops/jobs/verify-tree-n2.sh` then read `$STAGE`, which the job prefix did not carry, and `set -u` stopped the job
("bash: line 37: STAGE: unbound variable", `joblog-tail.txt`). The job is marked ERROR; 13 min billed, $0.40.

What it touches: the summary file stops after the phase tables, so the summary's own control-2 line and guard lines
were not written. The journal reads control 2, the guards, identity and every number from `vtree-4b.txt`, which the
bench had already closed. No measurement is lost or rerun. The script is fixed in the next commit (STAGE in the job
prefix); it has not run since.

No twin was needed: the job was RUNNING one minute after its submission.
