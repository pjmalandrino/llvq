# Prereg N8: where a row phase's time sits (a11i, a stamped a11), beside a10, sealed 4B

Written on 2026-11-10 at about 13:20 CET on branch `claude/arbre-nuit-2026-11-10`, once the code it describes existed
and passed its focused host checks (commit a6d86a2: a11i equals a11 bit for bit on the twin, stamped and unstamped).
Timestamped before the first measurement, then never edited; a departure goes to
`preregistration-verify-tree-n8-2026-11-10-ECARTS.md`.

Go: the operator, 2026-11-09 about 23:00 CET, in session "LLVQ pistes débit": « Compute HF ok tu as 10$ pour la nuit
comme tu le sens »; on 2026-11-10 the operator said to continue, then "go" (relayed by the orchestrating session). Cap:
**$10**, $2.21 spent before this job. Registry before this job: **$309.46**. Cost: about **$0.39** on l40sx1
(*estimated*, "Cost"), timeout 26 min, **$0.78** at worst. Measured code: commit `a6d86a2` ("Tree verify a11i: a11 with
fine stamps, and the wait classes for the a9 family"); the image is published from a clean detached checkout of the
commit that holds this file's stamp, whose kernel, bench and runtime sources are a6d86a2's (only the job script and the
proofs moved after it).

## Question

This job changes no kernel arithmetic and tests no repair: it is read for where a row phase's time sits. Three readings
of the row phases have now been wrong (a4, a7, a11). a11i is a11 with fine stamps on its units: per row phase, each
warp's loop, the staging, the close wait and the time to arrival. The bench also prints, for every a9-family arm, a
grid-wait table by class and each prologue's busy time on its owner CTAs.

a11i carries a11's prefetch, which N7 measured as neutral on the row phases (5,299 against 5,358 us at vt1) and costly
around them. a11i's unit parts are therefore read as a10's to within that difference, and its c_R is not a candidate.

## Setup

One image, one job, `STAGE=n8 ARMS='a10@5.14.5.28.5,a11i@5.14.5.28.5' ops/jobs/verify-tree-n2.sh`, the bench's own
timeout 23 min:

1. `oracle` on Qwen3-0.6B.
2. The sealed 4B at sha256 886391a8; `configs/qwen3-4b-tetra-e4-4e-v9.json` at its bytes.
3. `vtreebench`, one process: 4E v9 as the config builds it, then the tree pass beside it on the same tree-tiled
   copies, at P = [5, 14, 5, 28, 5]:
   - **a10@P**: N6's and N7's arm;
   - **a11i@P**: a11 with fine stamps; a11@P its guard, added by the bench as an untimed guard-only arm.
   At the chat prompt: `vt1` against 4E's greedy tokens at 64 positions and identity across R = 1, 2, 8, 16 and two
   shuffled slot orders; 2 discarded and 5 timed rounds of unstamped launches, each a one-token 4E launch then every
   arm's `vt1`, `vt2`, `vt16`; `vt1` and `vt16` stamped, the phase tables, the grid-wait tables by class and the
   prologue busy times of both arms, a11i's unit-parts table; 4E stamped over two tokens. Then the rag prompt for the
   arm with the lowest chat c_16 only.

## Controls

Each voids the run.

1. `oracle` prints MATCH; the file and config at their sha256; no `LLVQ_*` in the job environment.
2. 4E's module is the served v9 module (252,757 bytes, sha256 1b422b62), and every one-token 4E launch makes the
   greedy token it made before.
3. a11i's guard: a11i@P's 32-row pass equals a11@P's bit for bit ("a11i@5.14.5.28.5 guard against a11@5.14.5.28.5 ...
   PASSES"). If it fails, a11i's tables are void. A module that fails NVRTC voids its arm.

Results the decision rule reads, not controls: each arm's identity, and its `vt1` argmax against 4E at 64 of 64.

## What gets published, and what does not get compared

Published: each arm's c_R for R = 1, 2, 16, median and range formed round by round; the extra row; the phase tables of
`vt1` and `vt16`; the three new tables in full for `vt1` and `vt16` (a11i's unit parts, median [min-max] over the CTAs;
the grid waits by class for a10 and a11i; the prologue busy times on the owner CTAs); each module's bytes, sha256,
registers, local and shared bytes. Not compared: a11i's c_R as a candidate; any tok/s at a door, any acceptance, the 8B
and 14B, another card.

## Decision rule

As N1's. Read on the chat context, on the best arm whose guard and identity hold and whose `vt1` makes 4E's argmax at
64 of 64. a11i carries a11's prefetch and is not expected to be the best arm.

| result | outcome |
|---|---|
| c_2 ≤ 1.40 and c_16 ≤ 1.60 | PASS: the reopening condition of `ETAT.md` §6 is met by measurement; reopening stays the operator's call |
| 1.60 < c_16 ≤ 2.35 (or c_16 ≤ 1.60 with c_2 > 1.40) | BETWEEN |
| c_16 > 2.35 | FAIL |

## Signed prediction

Chat context, sealed 4B, one L40S, the v9 config, plan [5, 14, 5, 28, 5]. Signed by the orchestrating session.

| quantity | point | interval |
|---|---|---|
| 4E token | 4,455 us | [4,380, 4,500] |
| a10 c_1 | 1.756 | [1.72, 1.79] |
| a10 c_16 | 1.93 | [1.88, 1.98] |
| a11i c_1, unstamped | 1.97 | [1.90, 2.04] |
| a11i c_16, unstamped | 2.08 | [2.00, 2.16] |
| a11i unit parts, vt1, q/k/v: the slowest warp's loop over the fastest warp's loop, on the median CTA | | [1.15, 3.0] |
| a11i unit parts, vt1: staging as a share of the CTA's busy time | | under 15 % |
| a11i close wait and time to arrival | no value signed: finding them is the purpose | |
| outcome | BETWEEN, on a10 | |

## What this will not establish

Whether the 64-bit global atomics of a stamped launch distort the parts they time: the unstamped c_R of a11i against
a11's N7 values (c_1 1.967, c_16 2.077) is the only check. Any repair of the row phases. R = 4, 8 and 32. Any card but
the L40S. The 8B and 14B.

## Cost

N7 ran in 12 min billed, $0.37 (*measured*). This job is the same size, with more stamped launches: about 13 min, $0.39
(*estimated*). Timeout 26 min: $0.78 at worst, under the cap. On l40sx1; if it is not RUNNING after 20 minutes, a twin
on l40sx4 (one device visible) is added under a deviation written beside this file before it is submitted, and the first
to run is the measurement; the other is cancelled by hand once one copy is seen RUNNING or COMPLETED, and the
cancellation is confirmed.
