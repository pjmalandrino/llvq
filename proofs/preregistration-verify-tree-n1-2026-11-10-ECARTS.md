# Deviations from prereg N1 (`preregistration-verify-tree-n1-2026-11-10.md`)

The prereg is not edited. Each deviation is written here before the measurement it affects.

## Deviation 1: a twin of the job on l40sx4, the first to start is the measurement

Written on 2026-11-10 at about 01:50 CET, before any measurement: job `6af25295ab34849891a95ee1` (l40sx1) had waited
about 102 minutes in the queue for an L40S, never started, never billed.

- A second copy of the same job is submitted on flavor **l40sx4**: the same card model (L40S, sm_89), four on the
  node. The bench drives one device; the job sets `CUDA_VISIBLE_DEVICES=0` in its prefix, so the other three are
  not visible to it and stay idle.
- Same image (Space `Pier-Jean/llvq-runner-cuda` at sha `9687c6fd`), same measured code (`88c544f`), same arms
  (`a0,a1,a2`), same job body (`ops/jobs/verify-tree-n1.sh`; the script only gains a flavor variable and passes
  `--any-flavor`, since `ops/run.py bench` whitelists l40sx1 alone). Its own output directory (`RETRY=1`:
  `verify-tree-n1-2026-11-10-r1/`).
- The first of the two jobs to reach RUNNING is the measurement; the other is cancelled before it runs. If both are
  seen RUNNING at the same check, the l40sx4 one is cancelled.
- Cost of the twin at $8.30/h (*estimated*, not in `ops/run.py`'s table): about 12 min, **$1.66**; timeout 18 min,
  **$2.49** at worst. Under the night cap of $10, with the l40sx1 job at $0 billed so far.
- What can move: the host side of the timing (CPU model and clocks of the node), which enters every c_R through
  the host wall. The same holds for the 4E token beside it, measured in the same rounds on the same node, so c_R is
  a ratio on one machine. The 4E token's signed interval [4,350, 4,450] us was set on an l40sx1 node; a value
  outside it on l40sx4 is read with this deviation in mind. The vtreebench timeout inside the body stays 20 min;
  the job timeout of 18 min ends it first.
