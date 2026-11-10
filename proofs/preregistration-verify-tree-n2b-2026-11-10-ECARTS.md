# Deviations from prereg N2b (`preregistration-verify-tree-n2b-2026-11-10.md`)

The prereg is not edited. Each deviation is written here before the measurement it affects.

## Deviation 1: a twin on l40sx4, the first to run is the measurement

Written on 2026-11-10 at about 04:17 CET, before any measurement, as the prereg's "Cost" section foresees: job
`6af2890c9737d9a9e08eda49` (l40sx1, submitted 04:01 CET) was still SCHEDULING after 15 minutes, never started, never
billed.

- A second copy of the same job is submitted on flavor l40sx4 (four L40S on the node, sm_89), `CUDA_VISIBLE_DEVICES=0`
  in the job prefix so the bench sees one device; `--any-flavor`, since `ops/run.py bench` whitelists l40sx1 alone.
  Same image (Space sha 6047cb0a, published from aaf5d5c), same arms (`a3,a4`), same job body, its own output
  directory (`RETRY=1`: `verify-tree-n2b-2026-11-10-r1/`), timeout 15 min.
- The first of the two to reach RUNNING is the measurement; the other is cancelled. If both are seen RUNNING at the
  same check, the l40sx4 one is cancelled.
- Cost of the twin at $8.30/h (*estimated*): about 9 min, $1.25; timeout 15 min, $2.08 at worst, under the night cap
  ($0.42 spent before N2b).
- What can move: the host side of the timing on another node; c_R is a ratio taken on one machine in the same rounds.
