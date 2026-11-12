# Prereg N11: thread counts and plans around a10t384 (a10t384r16, a10t448, a10t320, a10t384 at a10's plan), sealed 4B

Written on 2026-11-10 at about 16:10 CET on branch `claude/arbre-nuit-2026-11-10`, once the code it describes existed
and passed its focused host checks (commit 816a713: a10t448, a10t320 and a10t384r16 equal a10 bit for bit on the twin).
Timestamped before the first measurement, then never edited; a departure goes to
`preregistration-verify-tree-n11-2026-11-10-ECARTS.md`.

Go: the operator, 2026-11-09 about 23:00 CET, in session "LLVQ pistes débit": « Compute HF ok tu as 10$ pour la nuit
comme tu le sens »; on 2026-11-10 the operator said to continue, then "go" (relayed by the orchestrating session). Cap:
**$10**, $3.71 spent before this job. Registry before this job: **$310.96**. Cost: about **$0.63** on l40sx1
(*estimated*, "Cost"), timeout 34 min, **$1.02** at worst. Measured code: commit `816a713` ("Tree verify: arms a10t448,
a10t320 and a10t384r16"); the image is published from a clean detached checkout of the commit that holds this file's
stamp, whose kernel, bench and runtime sources are 816a713's (only the job script and the proofs moved after it).

## Question

N10 found a10t384 (384 threads, plan [8, 10, 8, 20, 6]) the fastest arm (c_1 1.650, c_16 1.819), but its plan and its
thread count changed together. This job separates them and stacks the row cap N9 found useful:

- **a10t384@[8, 10, 8, 20, 6]**: N10's best arm, repeated;
- **a10t384r16@[8, 10, 8, 20, 6]**: the same with the 16-row cap; guard a10t384 at the same plan on the first 16 rows;
- **a10t448@[9, 11, 8, 22, 6]**: 448 threads (14 warps);
- **a10t320@[7, 8, 8, 16, 7]**: 320 threads (10 warps);
- **a10t384@[5, 14, 5, 28, 5]**: 384 threads at a10's plan (2 waves at 59.9 % fill on q/k/v), which isolates the thread
  count from the plan.

## Setup

One image, one job, `STAGE=n11 ARMS='a10t384@8.10.8.20.6,a10t384r16@8.10.8.20.6,a10t448@9.11.8.22.6,a10t320@7.8.8.16.7,
a10t384@5.14.5.28.5' ops/jobs/verify-tree-n2.sh` (one argument, no space), the bench's own timeout 31 min:

1. `oracle` on Qwen3-0.6B.
2. The sealed 4B at sha256 886391a8; `configs/qwen3-4b-tetra-e4-4e-v9.json` at its bytes.
3. `vtreebench`, one process: 4E v9 as the config builds it, then the tree pass beside it on the same tree-tiled copies,
   the five arms above; the bench adds a10 at four plans as untimed guard-only arms (6 NVRTC builds in all). At the chat
   prompt: `vt1` against 4E's greedy tokens at 64 positions and identity within each arm's row cap; 2 discarded and 5
   timed rounds, each a one-token 4E launch then every arm's `vt1`, `vt2`, `vt8`, `vt16`; `vt1` and `vt16` stamped,
   the phase tables; 4E stamped over two tokens. Then the rag prompt for the arm with the lowest chat c_16 only.

## Controls

Each voids the run.

1. `oracle` prints MATCH; the file and config at their sha256; no `LLVQ_*` in the job environment.
2. 4E's module is the served v9 module (252,757 bytes, sha256 1b422b62), and every one-token 4E launch makes the
   greedy token it made before.
3. Each arm's guard (a10 at its own plan; a10t384 at the same plan on the first 16 rows for a10t384r16) passes bit for
   bit. A failing guard voids its own arm only. A module that fails NVRTC, or that fits no CTA on a SM, voids its arm. If
   the image fails to build, nothing is launched.

Results the decision rule reads, not controls: each arm's identity, and its `vt1` argmax against 4E at 64 of 64.

## What gets published, and what does not get compared

Published: each module's registers, local and shared bytes from its install line; each arm's plan line as printed; each
arm's c_R for R = 1, 2, 8, 16, median and range formed round by round; the extra row; the phase tables, with the
row-phase walls and busy min / med / max of vt1 for the five arms. Not compared: any tok/s at a door, any acceptance, the
8B and 14B, another card.

## Decision rule

As N1's. Read on the chat context, on the best arm whose guard and identity hold and whose `vt1` makes 4E's argmax at
64 of 64.

| result | outcome |
|---|---|
| c_2 ≤ 1.40 and c_16 ≤ 1.60 | PASS: the reopening condition of `ETAT.md` §6 is met by measurement; reopening stays the operator's call |
| 1.60 < c_16 ≤ 2.35 (or c_16 ≤ 1.60 with c_2 > 1.40) | BETWEEN |
| c_16 > 2.35 | FAIL |

## Signed prediction

Chat context, sealed 4B, one L40S, the v9 config. Signed by the orchestrating session.

| quantity | point | interval |
|---|---|---|
| 4E token | 4,430 us | [4,380, 4,500] |
| a10t384@[8,10,8,20,6] c_1 | 1.65 | [1.62, 1.69] |
| a10t384@[8,10,8,20,6] c_16 | 1.82 | [1.78, 1.87] |
| a10t384r16@[8,10,8,20,6] c_1 | 1.58 | [1.50, 1.65] |
| a10t384r16@[8,10,8,20,6] c_2 | 1.59 | [1.51, 1.66] |
| a10t384r16@[8,10,8,20,6] c_16 | 1.77 | [1.68, 1.84] |
| a10t448@[9,11,8,22,6] c_1 | 1.68 | [1.58, 1.80] |
| a10t448@[9,11,8,22,6] c_16 | 1.85 | [1.74, 1.98] |
| a10t320@[7,8,8,16,7] c_1 | 1.68 | [1.56, 1.84] |
| a10t320@[7,8,8,16,7] c_16 | 1.84 | [1.72, 2.00] |
| a10t384@[5,14,5,28,5] c_1 | 1.74 | [1.62, 1.84] |
| a10t384@[5,14,5,28,5] c_16 | 1.92 | [1.80, 2.02] |
| a10t384@[5,14,5,28,5] q/k/v of vt1 (wall, us summed over the layers) | | [1,100, 1,500] |
| outcome | BETWEEN, on a10t384r16 | |

## What this will not establish

What the 720 to 736 B of local memory hold. Any thread count other than 320, 384, 448 and N10's 256 and 512. R = 4 and
32. Any card but the L40S. The 8B and 14B.

## Cost

N10 ran in 16 min billed, $0.48 (*measured*), three timed arms. This job compiles six modules and times five arms: about
21 min, $0.63 (*estimated*). Timeout 34 min: $1.02 at worst, under the cap. On l40sx1; if it is not RUNNING after 20
minutes, a twin on l40sx4 (one device visible) is added under a deviation written beside this file before it is
submitted, and the first to run is the measurement; the other is cancelled by hand once one copy is seen RUNNING or
COMPLETED, and the cancellation is confirmed.
