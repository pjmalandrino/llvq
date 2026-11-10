# Deviations from prereg N4 (`preregistration-verify-tree-n4-2026-11-10.md`)

The prereg is not edited. Each deviation is written here before the measurement it affects.

## Deviation 1: a twin on l40sx4, the first to run is the measurement

Written on 2026-11-10 at about 06:29 CET, before any measurement, as the prereg's "Cost" section foresees: job
`6af2a8119737d9a9e08ef3a4` (l40sx1, submitted 06:13 CET) was still SCHEDULING after 15 minutes, never started, never
billed.

- A second copy of the same job is submitted on flavor l40sx4 (four L40S on the node, sm_89), `CUDA_VISIBLE_DEVICES=0`
  in the job prefix so the bench sees one device; `--any-flavor`, since `ops/run.py bench` whitelists l40sx1 alone.
  Same image (Space sha 3c83d70c, published from ca06294), same arms, same job body, its own output directory
  (`RETRY=1`: `verify-tree-n4-2026-11-10-r1/`), timeout 24 min.
- The first of the two to reach RUNNING is the measurement; the other is cancelled. If both are seen RUNNING at the
  same check, the l40sx4 one is cancelled.
- Cost of the twin at $8.30/h (*estimated*): about 19 min, $2.63; timeout 24 min, $3.32 at worst. With $1.08 spent
  before N4, the night stays under its cap of $10 even if both copies were billed to their timeouts ($0.90 + $3.32).
- What can move: the host side of the timing on another node; c_R is a ratio taken on one machine in the same rounds.
  Load and install (800 to 850 s expected) leave about 8 min of the twin's 24 for the measurement, against 15 for the
  bench's own 27-min limit on l40sx1: if the twin runs and its job timeout ends the bench first, the context it had
  not reached is missing, and that is written in the journal.
