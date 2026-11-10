# Prereg N6: a10 (a9 with 4E's prologue path and silu in the gate/up fold) beside a9, sealed 4B

Written on 2026-11-10 at about 11:05 CET on branch `claude/arbre-nuit-2026-11-10`, once the code it describes existed
and passed its focused host checks (see Controls). Timestamped before the first measurement, then never edited; a
departure goes to `preregistration-verify-tree-n6-2026-11-10-ECARTS.md`.

Go: the operator, 2026-11-09 about 23:00 CET, in session "LLVQ pistes débit": « Compute HF ok tu as 10$ pour la nuit
comme tu le sens »; on 2026-11-10 the operator said to continue, then "go" for this job (relayed by the orchestrating
session). Cap: **$10**, $1.52 spent before this job. Registry before this job: **$308.77**. Cost: about **$0.42** on
l40sx1 (*estimated*, "Cost"), timeout 24 min, **$0.72** at worst. Measured code: commit `c76c9ad` ("Tree verify a10: a9
with 4E's prologue path and silu in the gate/up fold"); the image is published from a clean detached checkout of the
commit that holds this file's stamp, whose kernel, bench and runtime sources are c76c9ad's (only the job script and the
proofs moved after it).

## Question

N5 (`docs/mesures/verify-tree-n5-2026-11-10.txt`) left a9 at P = [5, 14, 5, 28, 5] at c_1 1.966 and c_16 2.181, with
the prologues ln1, ln2 and dn_in at about 1,400 us of vt1, run by one CTA a row while the others wait. a10 is a9 with
4E's prologue path: the signs and rotation rows staged in shared memory during the preceding wait, 16-byte loads and
stores, the unrolled shared mix, and silu(gate) x up computed in the gate/up fold step. How much of the prologues and of
c_1 does it take off?

## Setup

One image, one job, `STAGE=n6 ARMS='a9@5.14.5.28.5,a10@5.14.5.28.5' ops/jobs/verify-tree-n2.sh`, the bench's own
timeout 21 min:

1. `oracle` on Qwen3-0.6B.
2. The sealed 4B at sha256 886391a8; `configs/qwen3-4b-tetra-e4-4e-v9.json` at its bytes.
3. `vtreebench`, one process: 4E v9 as the config builds it, then the tree pass beside it on the same tree-tiled
   copies, both arms at P:
   - **a9@P**: N5's arm;
   - **a10@P**: a10, a9@P its guard. Same phase names and wait count as a9.
   At the chat prompt: `vt1` against 4E's greedy tokens at 64 positions and identity across R = 1, 2, 8, 16 and two
   shuffled slot orders; 2 discarded and 5 timed rounds, each a one-token 4E launch then every arm's `vt1`, `vt2`,
   `vt16`; `vt1` and `vt16` stamped, the phase tables with the fold rows; 4E stamped over two tokens. Then the rag
   prompt for the arm with the lowest chat c_16 only.

## Controls

Each voids the run.

1. `oracle` prints MATCH; the file and config at their sha256; no `LLVQ_*` in the job environment.
2. 4E's module is the served v9 module (252,757 bytes, sha256 1b422b62), and every one-token 4E launch makes the
   greedy token it made before.
3. a10's guard: a10@P's 32-row pass equals a9@P's bit for bit ("a10@5.14.5.28.5 guard against a9@5.14.5.28.5 ...
   PASSES"). If it fails, a10 is void and a9 stands. A module that fails NVRTC voids its arm.

Host gates at launch: the focused host gates of a10 are green (bit for bit with a9 at R = 1, 2, 8, 16, 32 under two plans
and on the 8B's parts, 4 mutants of 4 killed). The full `tree_verify_host` suite on c76c9ad is still running when this
file is stamped; its result goes into the journal.

Results the decision rule reads, not controls: each arm's identity, and its `vt1` argmax against 4E at 64 of 64.

## What gets published, and what does not get compared

Published: each arm's c_R for R = 1, 2, 16, median and range formed round by round; the extra row; the phase tables of
`vt1` and `vt16`, a9 and a10 side by side with the fold rows; each module's bytes, sha256, registers, local and shared
bytes; any NVRTC error in full. Not compared: any tok/s at a door, any acceptance, the 8B and 14B, another card.

## Decision rule

As N1's. Read on the chat context, on the best arm whose guard (for a10) and identity hold and whose `vt1` makes 4E's
argmax at 64 of 64.

| result | outcome |
|---|---|
| c_2 ≤ 1.40 and c_16 ≤ 1.60 | PASS: the reopening condition of `ETAT.md` §6 is met by measurement; reopening stays the operator's call |
| 1.60 < c_16 ≤ 2.35 (or c_16 ≤ 1.60 with c_2 > 1.40) | BETWEEN |
| c_16 > 2.35 | FAIL |

## Signed prediction

Chat context, sealed 4B, one L40S, the v9 config, plan [5, 14, 5, 28, 5]. Signed by the orchestrating session.

| quantity | point | interval |
|---|---|---|
| 4E token | 4,435 us | [4,380, 4,480] |
| a9 c_1 | 1.966 | [1.93, 2.00] |
| a9 c_2 | 1.99 | [1.95, 2.03] |
| a9 c_16 | 2.18 | [2.12, 2.25] |
| a9 prologues of vt1 (ln1 + ln2 + dn_in, wall, us summed over the layers) | 1,403 | [1,300, 1,500] |
| a10 c_1 | 1.80 | [1.68, 1.93] |
| a10 c_2 | 1.82 | [1.70, 1.95] |
| a10 c_16 | 2.00 | [1.86, 2.15] |
| a10 prologues of vt1 | 600 | [420, 900] |
| a10 gate/up fold of vt1 (a9: 112 in N5) | | [110, 260] |
| outcome | BETWEEN, on a10 | |

## What this will not establish

Which of a10's changes (staged signs and rotation rows, vector loads and stores, the unrolled mix, silu in the fold)
buys which part. R = 4, 8 and 32. Any card but the L40S. The 8B and 14B.

## Cost

N5 ran in 15 min billed, $0.44 (*measured*), five modules and six arms. This job compiles two and times two: about 14
min, $0.42 (*estimated*). Timeout 24 min: $0.72 at worst, under the cap. On l40sx1; if it is not RUNNING after 20
minutes, a twin on l40sx4 (TIMEOUT 20m, one device visible) is added under a deviation written beside this file before
it is submitted, and the first to run is the measurement; the other is cancelled by hand once one copy is seen RUNNING
or COMPLETED, and the cancellation is confirmed.
