# Prereg N9: does a10 run faster with fewer accumulators (a10r16, a10r8)? Sealed 4B

Written on 2026-11-10 at about 14:15 CET on branch `claude/arbre-nuit-2026-11-10`, once the code it describes existed
and passed its focused host checks (commit baf5e57: a10r8 and a10r16 equal a10 bit for bit on the twin at the rows
they allow). Timestamped before the first measurement, then never edited; a departure goes to
`preregistration-verify-tree-n9-2026-11-10-ECARTS.md`.

Go: the operator, 2026-11-09 about 23:00 CET, in session "LLVQ pistes débit": « Compute HF ok tu as 10$ pour la nuit
comme tu le sens »; on 2026-11-10 the operator said to continue, then "go" (relayed by the orchestrating session). Cap:
**$10**, $2.71 spent before this job. Registry before this job: **$309.96**. Cost: about **$0.45** on l40sx1
(*estimated*, "Cost"), timeout 26 min, **$0.78** at worst. Measured code: commit `baf5e57` ("Tree verify: row-cap arms
a10r8 and a10r16, and vtreebench --res"); the image is published from a clean detached checkout of the commit that
holds this file's stamp, whose kernel, bench and runtime sources are baf5e57's (only the job script and the proofs moved
after it).

## Question

N8 found that code which only adds local memory slows the pass: a11i, a11 with stamps, ran 0.15 to 0.23 of a token
slower unstamped, at 824 B of local memory a thread against 736. a10 holds 16 accumulators a thread for up to 32 rows a
launch. a10r16 and a10r8 are a10 compiled for at most 16 and 8 rows a launch: 8 and 4 accumulators a thread, the same
arithmetic. If fewer accumulators lower the local memory, does the pass get faster? The hypothesis under test, stated
so it can fail: an arm whose local memory equals a10's has a10's c_R to within 0.02.

## Setup

One image, one job, `STAGE=n9 ARMS='a10@5.14.5.28.5,a10r16@5.14.5.28.5,a10r8@5.14.5.28.5'
ops/jobs/verify-tree-n2.sh` (one argument, no space), the bench's own timeout 23 min:

1. `oracle` on Qwen3-0.6B.
2. The sealed 4B at sha256 886391a8; `configs/qwen3-4b-tetra-e4-4e-v9.json` at its bytes.
3. `vtreebench --res a10,a10r16,a10r8,a3,a9`, its own timeout 3 min, into `res.txt`: each named arm's module through
   NVRTC, its registers, local and shared bytes, no model loaded. It has never compiled; its failure voids nothing,
   and the install lines of step 4 give the same three numbers for the timed arms.
4. `vtreebench`, one process: 4E v9 as the config builds it, then the tree pass beside it on the same tree-tiled
   copies, at P = [5, 14, 5, 28, 5]:
   - **a10@P**: N6 to N8's arm;
   - **a10r16@P**: a10 for at most 16 rows a launch; a10@P its guard on the common rows;
   - **a10r8@P**: a10 for at most 8 rows a launch; a10@P its guard on the common rows.
   At the chat prompt: `vt1` against 4E's greedy tokens at 64 positions and identity within each arm's row cap; 2
   discarded and 5 timed rounds, each a one-token 4E launch then every arm's `vt1`, `vt2`, `vt8`, `vt16` within its
   cap; `vt1` and `vt16` stamped (`vt8` for a10r8), the phase tables; 4E stamped over two tokens. Then the rag prompt
   for the arm with the lowest chat c_16 only.

## Controls

Each voids the run.

1. `oracle` prints MATCH; the file and config at their sha256; no `LLVQ_*` in the job environment.
2. 4E's module is the served v9 module (252,757 bytes, sha256 1b422b62), and every one-token 4E launch makes the
   greedy token it made before.
3. The row-cap arms' guards: each equals a10@P bit for bit on the rows it allows ("a10r16@... guard against a10@...
   PASSES", the same for a10r8). A failing guard voids its own arm only. A module that fails NVRTC voids its arm.

Results the decision rule reads, not controls: each arm's identity, and its `vt1` argmax against 4E at 64 of 64.

## What gets published, and what does not get compared

Published: the `--res` lines; each arm's c_R for R = 1, 2, 8, 16 within its cap, median and range formed round by
round; the extra row; the phase tables, with the row-phase walls of vt1 for the three arms; each module's bytes, sha256,
registers, local and shared bytes from its install line. Not compared: a10r8 at R = 16 (it does not allow it); any tok/s
at a door, any acceptance, the 8B and 14B, another card.

## Decision rule

As N1's, read on the chat context, on the best of the arms that allow 16 rows (a10@P, a10r16@P) whose guard and identity
hold and whose `vt1` makes 4E's argmax at 64 of 64. a10r8 is read at c_2 and c_8 only and cannot pass the rule.

| result | outcome |
|---|---|
| c_2 ≤ 1.40 and c_16 ≤ 1.60 | PASS: the reopening condition of `ETAT.md` §6 is met by measurement; reopening stays the operator's call |
| 1.60 < c_16 ≤ 2.35 (or c_16 ≤ 1.60 with c_2 > 1.40) | BETWEEN |
| c_16 > 2.35 | FAIL |

## Signed prediction

Chat context, sealed 4B, one L40S, the v9 config, plan [5, 14, 5, 28, 5]. Signed by the orchestrating session.

| quantity | point | interval |
|---|---|---|
| 4E token | 4,420 us | [4,350, 4,500] |
| a10 local memory, bytes a thread | 736 (measured three times) | |
| a10r16 local memory | 690 | [600, 736] |
| a10r8 local memory | 650 | [520, 736] |
| registers, all three arms | 128 | |
| a10 c_1 | 1.757 | [1.72, 1.79] |
| a10 c_8 | 1.84 | [1.80, 1.89] |
| a10 c_16 | 1.94 | [1.89, 1.99] |
| a10r16 c_1 | 1.70 | [1.55, 1.78] |
| a10r16 c_8 | 1.78 | [1.62, 1.87] |
| a10r16 c_16 | 1.87 | [1.70, 1.96] |
| a10r8 c_1 | 1.65 | [1.45, 1.78] |
| a10r8 c_8 | 1.73 | [1.52, 1.87] |
| an arm whose local memory equals a10's | its c_R equal to a10's within 0.02 | |
| outcome | BETWEEN | |

## What this will not establish

Whether the local memory is the cause of a time difference, rather than another change the row cap brings to the code
(the registers and the unrolling move with it). R = 4 and 32. Any card but the L40S. The 8B and 14B.

## Cost

N8 ran in 17 min billed, $0.50 (*measured*), with its long install. This job compiles three timed modules and runs the
compile-only step: about 15 min, $0.45 (*estimated*). Timeout 26 min: $0.78 at worst, under the cap. On l40sx1; if it
is not RUNNING after 20 minutes, a twin on l40sx4 (one device visible) is added under a deviation written beside this
file before it is submitted, and the first to run is the measurement; the other is cancelled by hand once one copy is
seen RUNNING or COMPLETED, and the cancellation is confirmed.
