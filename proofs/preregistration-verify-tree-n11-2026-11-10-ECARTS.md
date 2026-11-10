# Deviations from prereg N11 (`preregistration-verify-tree-n11-2026-11-10.md`)

The prereg is not edited. Each deviation is written here before the measurement it affects.

## Deviation 1: a twin on l40sx4, the first to run is the measurement

Written on 2026-11-10 at about 16:55 CET, before any measurement, as the prereg's "Cost" section foresees: job
`6af33988ab34849891ac2c8f` (l40sx1, submitted 16:34 CET) was still SCHEDULING after 20 minutes, never started, never
billed.

- A second copy of the same job is submitted on flavor l40sx4 (four L40S on the node, sm_89), `CUDA_VISIBLE_DEVICES=0`
  in the job prefix so the bench sees one device; `--any-flavor`, since `ops/run.py bench` whitelists l40sx1 alone.
  Same image (Space sha ce2ee449, published from 5386da5), same arms, same job body, its own output directory
  (`RETRY=1`: `verify-tree-n11-2026-11-10-r1/`), timeout 34 min.
- The first of the two seen RUNNING is the measurement; the other is cancelled by hand and the cancellation confirmed.
  If both are seen RUNNING at the same check, the l40sx4 one is cancelled.
- Cost of the twin at $8.30/h (*estimated*): about 21 min, $2.91; timeout 34 min, $4.70 at worst. With $3.71 spent
  before N11 and at most $1.02 on the l40sx1 copy, the worst case if both were billed to their timeouts is $9.43, under
  the cap of $10; only one copy is to run.
- What can move: the host side of the timing on another node; c_R is a ratio taken on one machine in the same rounds.
