# Deviations from prereg SP2 (`preregistration-spec-loop-sp2-2026-11-10.md`)

The prereg is not edited. Each deviation is written here before the measurement it affects.

## Deviation 1: a twin on l40sx4, the first to run is the measurement

Written on 2026-11-10 at about 20:29 CET, before any measurement, as the prereg's "Cost" section foresees: job
`6af36ba99737d9a9e0915bf2` (l40sx1, submitted 20:08 CET) was still SCHEDULING after 20 minutes, never started, never
billed.

- A second copy of the same job is submitted on flavor l40sx4 (four L40S on the node, sm_89), `CUDA_VISIBLE_DEVICES=0`
  in the job prefix so the bench sees one device; `--any-flavor`, since `ops/run.py bench` whitelists l40sx1 alone.
  Same image (Space sha 158f9340, published from 2b6386f), same steps, same job body, its own output directory
  (`RETRY=1`: `spec-loop-sp2-2026-11-10-r1/`), timeout 40 min.
- The first of the two seen RUNNING is the measurement; the other is cancelled by hand and the cancellation confirmed.
  If both are seen RUNNING at the same check, the l40sx4 one is cancelled.
- Cost of the twin at $8.30/h (*estimated*): about 30 min, $4.15; timeout 40 min, $5.53 at worst. With $4.34 spent
  before SP2: if the l40sx4 copy runs to its estimate, the night stands at $8.49; if it ran to its timeout, $9.87, under
  the cap of $10. Only one copy is to run, and the l40sx1 copy is cancelled the moment the twin is seen RUNNING.
- What can move: the host side of the timing on another node; every ratio is taken on one machine in the same rounds.
