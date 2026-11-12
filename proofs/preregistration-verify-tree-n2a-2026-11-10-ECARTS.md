# Deviations from prereg N2a (`preregistration-verify-tree-n2a-2026-11-10.md`)

The prereg is not edited. Each deviation is written here before the measurement it affects.

## Deviation 1: a twin on l40sx4, the first to run is the measurement

Written on 2026-11-10 at about 03:09 CET, before any measurement, as the prereg's "Cost" section foresees: job
`6af2790c9737d9a9e08e97d7` (l40sx1, submitted 02:53 CET) was still SCHEDULING after 15 minutes, never started, never
billed.

- A second copy of the same job is submitted on flavor l40sx4 (four L40S on the node, sm_89), `CUDA_VISIBLE_DEVICES=0`
  in the job prefix so the bench sees one device; `--any-flavor`, since `ops/run.py bench` whitelists l40sx1 alone.
  Same image (Space sha e1360a47, code b981261), same arms (`a2,a3`), same job body, its own output directory
  (`RETRY=1`: `verify-tree-n2a-2026-11-10-r1/`), timeout 15 min.
- The first of the two to reach RUNNING is the measurement; the other is cancelled. If both are seen RUNNING at the
  same check, the l40sx4 one is cancelled.
- Cost of the twin at $8.30/h (*estimated*): about 9 min, $1.25; timeout 15 min, $2.08 at worst, under the night cap
  ($0.20 spent before N2a).
- What can move: the host side of the timing on another node; c_R is a ratio taken on one machine in the same rounds.

Outcome, written after the job: the l40sx1 job reached RUNNING first (03:13 CET, about 20 min after its submission);
the l40sx4 twin, still SCHEDULING, was cancelled at 03:13:25 and never ran ($0). The measurement ran on the prereg's
flavor, so deviation 1 touches no number.
