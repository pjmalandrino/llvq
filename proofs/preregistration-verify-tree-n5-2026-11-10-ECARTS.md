# Deviations from prereg N5 (`preregistration-verify-tree-n5-2026-11-10.md`)

The prereg is not edited. Each deviation is written here before the measurement it affects.

## Deviation 1: two copies submitted at once, l40sx1 and l40sx4; the first to run is the measurement

Written on 2026-11-10 at about 08:20 CET, before the submission and before any measurement. N4 waited about two hours
on both flavors without starting, so N5 is submitted on both at once instead of adding the twin after 15 minutes.

- l40sx1: `ops/jobs/verify-tree-n2.sh` as written, timeout 32 min.
- l40sx4 (four L40S on the node, sm_89): `CUDA_VISIBLE_DEVICES=0` in the job prefix so the bench sees one device;
  `--any-flavor`, since `ops/run.py bench` whitelists l40sx1 alone; its own output directory (`RETRY=1`:
  `verify-tree-n5-2026-11-10-r1/`); timeout 26 min.
- Same image (the Space sha recorded in the journal), same arms, same job body.
- The first of the two to reach RUNNING is the measurement; the other is cancelled at once. If both are seen RUNNING at
  the same check, the l40sx4 one is cancelled.
- Cost (*estimated*): l40sx1 about $0.57, $0.96 at worst; l40sx4 at $8.30/h about $2.63, $3.60 at worst.
- What can move: the host side of the timing on another node; c_R is a ratio taken on one machine in the same rounds.
  On l40sx4, load and install (800 to 850 s expected) leave about 12 min of the 26 for the bench; if the job timeout
  ends it first, the context it had not reached is missing, and the journal says so.
