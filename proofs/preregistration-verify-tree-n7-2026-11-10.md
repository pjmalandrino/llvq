# Prereg N7: a11 (a10 with the L2 prefetch of the row phases' streams in its wait gaps) beside a10, sealed 4B

Written on 2026-11-10 at about 12:00 CET on branch `claude/arbre-nuit-2026-11-10`, once the code it describes existed
and passed its focused host checks (see Controls). Timestamped before the first measurement, then never edited; a
departure goes to `preregistration-verify-tree-n7-2026-11-10-ECARTS.md`.

Go: the operator, 2026-11-09 about 23:00 CET, in session "LLVQ pistes débit": « Compute HF ok tu as 10$ pour la nuit
comme tu le sens »; on 2026-11-10 the operator said to continue, then "go" (relayed by the orchestrating session). Cap:
**$10**, $1.84 spent before this job. Registry before this job: **$309.09**. Cost: about **$0.33** on l40sx1
(*estimated*, "Cost"), timeout 24 min, **$0.72** at worst. Measured code: commit `e1c963d` ("Tree verify a11: a10 with
the L2 prefetch of the row phases' streams in its wait gaps"); the image is published from a clean detached checkout of
the commit that holds this file's stamp, whose kernel, bench and runtime sources are e1c963d's (only the job script and
the proofs moved after it).

## Question

N6 (`docs/mesures/verify-tree-n6-2026-11-10.txt`) left a10 at P = [5, 14, 5, 28, 5] at c_1 1.747 and c_16 1.918, with
the four row phases of vt1 (q/k/v, o, gate/up, down) at about 5,360 us, and the CTAs idle in the waits around the
prologues and attention. a11 is a10 plus `prefetch.global.L2` of each CTA's own units' tree-tiled streams, issued only
inside wait gaps: q/k/v before ln1, o before attention, gate/up after attention, down during the gate/up fold's wait.
How much of the row phases does it take off? a7 (N3) put 4E's prefetch points in the pass and lost 1.13 tokens: this
job also says whether issuing in the gaps avoids that.

## Setup

One image, one job, `STAGE=n7 ARMS='a10@5.14.5.28.5,a11@5.14.5.28.5' ops/jobs/verify-tree-n2.sh`, the bench's own
timeout 21 min:

1. `oracle` on Qwen3-0.6B.
2. The sealed 4B at sha256 886391a8; `configs/qwen3-4b-tetra-e4-4e-v9.json` at its bytes.
3. `vtreebench`, one process: 4E v9 as the config builds it, then the tree pass beside it on the same tree-tiled
   copies, both arms at P:
   - **a10@P**: N6's arm;
   - **a11@P**: a11, a10@P its guard.
   At the chat prompt: `vt1` against 4E's greedy tokens at 64 positions and identity across R = 1, 2, 8, 16 and two
   shuffled slot orders; 2 discarded and 5 timed rounds, each a one-token 4E launch then every arm's `vt1`, `vt2`,
   `vt16`; `vt1` and `vt16` stamped, the phase tables with the fold rows; 4E stamped over two tokens. Then the rag
   prompt for the arm with the lowest chat c_16 only.

## Controls

Each voids the run.

1. `oracle` prints MATCH; the file and config at their sha256; no `LLVQ_*` in the job environment.
2. 4E's module is the served v9 module (252,757 bytes, sha256 1b422b62), and every one-token 4E launch makes the
   greedy token it made before.
3. a11's guard: a11@P's 32-row pass equals a10@P's bit for bit ("a11@5.14.5.28.5 guard against a10@5.14.5.28.5 ...
   PASSES"). If it fails, a11 is void and a10 stands. A module that fails NVRTC voids its arm.

Host gates at launch: the full `tree_verify_host` suite on c76c9ad (a10) passed, 26 of 26. a11's focused gates are green
(bit for bit with a10 at R = 1, 8, 32 under two plans and on the 8B's parts; the range mutant killed). No full suite ran
on e1c963d. a11's registers and local bytes are unknown before the card (Docker would not start for a ptxas check).

Results the decision rule reads, not controls: each arm's identity, and its `vt1` argmax against 4E at 64 of 64.

## What gets published, and what does not get compared

Published: each arm's c_R for R = 1, 2, 16, median and range formed round by round; the extra row; the phase tables of
`vt1` and `vt16`, a10 and a11 side by side with busy min / med / max of the row phases and of the waits where the
prefetch is issued; each module's bytes, sha256, registers, local and shared
bytes; any NVRTC error in full. Not compared: any tok/s at a door, any acceptance, the 8B and 14B, another card.

## Decision rule

As N1's. Read on the chat context, on the best arm whose guard (for a11) and identity hold and whose `vt1` makes 4E's
argmax at 64 of 64.

| result | outcome |
|---|---|
| c_2 ≤ 1.40 and c_16 ≤ 1.60 | PASS: the reopening condition of `ETAT.md` §6 is met by measurement; reopening stays the operator's call |
| 1.60 < c_16 ≤ 2.35 (or c_16 ≤ 1.60 with c_2 > 1.40) | BETWEEN |
| c_16 > 2.35 | FAIL |

## Signed prediction

Chat context, sealed 4B, one L40S, the v9 config, plan [5, 14, 5, 28, 5]. Signed by the orchestrating session, whose
note is that it was wrong on this lever once (a7), so the interval is wide on purpose.

| quantity | point | interval |
|---|---|---|
| 4E token | 4,450 us | [4,380, 4,500] |
| a10 c_1 | 1.747 | [1.71, 1.79] |
| a10 c_2 | 1.76 | [1.72, 1.80] |
| a10 c_16 | 1.92 | [1.87, 1.98] |
| a10 row phases of vt1 (q/k/v + o + gate/up + down, wall, us summed over the layers) | 5,360 | [5,250, 5,480] |
| a11 c_1 | 1.50 | [1.30, 1.80] |
| a11 c_2 | 1.51 | [1.31, 1.81] |
| a11 c_16 | 1.68 | [1.48, 1.98] |
| a11 row phases of vt1 | 4,300 | [3,400, 5,300] |
| a11 local memory, bytes a thread | at most 760 | |
| outcome | BETWEEN, on a11 | |

## What this will not establish

Which of a11's four prefetch points buys which part. Whether the prefetched lines survive in L2 until their phase on a
card with another L2 size. R = 4, 8 and 32. Any card but the L40S. The 8B and 14B.

## Cost

N6 ran in 11 min billed, $0.32 (*measured*), two modules and two arms. This job is the same size: about 11 min, $0.33
(*estimated*). Timeout 24 min: $0.72 at worst, under the cap. On l40sx1; if it is not RUNNING after 20
minutes, a twin on l40sx4 (TIMEOUT 20m, one device visible) is added under a deviation written beside this file before
it is submitted, and the first to run is the measurement; the other is cancelled by hand once one copy is seen RUNNING
or COMPLETED, and the cancellation is confirmed.
