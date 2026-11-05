# Departures from proofs/preregistration-served-4e-2026-11-05.md

The prereg is stamped (integration/lot1 7f32775) and is not edited. This file records departures, written before
the run they concern.

## E1. The protocol may run inside the shaker job (written 2026-11-05, before any launch)

If the operator launches `ops/jobs/shaker-4e.sh` instead of `ops/jobs/served-4e.sh`, this prereg's protocol runs as
its part C. Two things differ, and nothing else:
- **The job script.** The shaker reads this protocol's preamble and body from `ops/jobs/served-4e.sh` when it is
  built: the variables, the embedded configs and their sha256, the comparison script, the pilot reference, then
  every command, gate and grep. It runs them unchanged in a subshell.
- **The output directory.** It is `shaker-4e-2026-11-05/C/` instead of `served-4e-2026-11-05/`. The job runs the
  oracle once more inside part C, as this protocol's body does.

Every gate, outcome, prediction and decision rule of the prereg applies as written. The 4E runtime of the served
config is row form 1 (v3), the form this prereg describes. The image also carries forms 2 and 3, which only the
shaker's part A builds.

If `ops/jobs/served-4e.sh` runs alone, E1 does not apply.
