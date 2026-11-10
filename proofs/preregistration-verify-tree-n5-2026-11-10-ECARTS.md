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

## Deviation 2: the race watcher was blind while the l40sx1 copy ran; the l40sx4 copy was cancelled by another hand

Written on 2026-11-10 at about 10:40 CET, after the job. The watcher's log has no check between 08:40:53 and 10:08 CET
(its 10:08 line reads both states empty): the Mac it ran on made no check in that window. The l40sx1 copy ran inside
it, from about 09:28 to 09:43 CET (the bucket's file times), and was seen COMPLETED at 10:18. The l40sx4 copy was
therefore not cancelled when the l40sx1 copy started, as deviation 1 required. It stayed SCHEDULING at every check
until 10:33 and was CANCELED at 10:34 CET, not by this watcher (its log only records "ended CANCELED"); it never ran
and cost $0.

What it touches: no number. One copy ran, on the prereg's flavor l40sx1, and the other never started, so the measurement
is the one deviation 1 names. The risk the watcher existed for (both copies billed) did not happen.
