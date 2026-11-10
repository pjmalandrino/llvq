# Prereg N10: a10 at 384 and 256 threads a CTA (a10t384, a10t256), sealed 4B

Written on 2026-11-10 at about 15:05 CET on branch `claude/arbre-nuit-2026-11-10`, once the code it describes existed
and passed its focused host checks (commit afd23b7: a10t384 and a10t256 equal a10 bit for bit on the twin). Timestamped
before the first measurement, then never edited; a departure goes to
`preregistration-verify-tree-n10-2026-11-10-ECARTS.md`.

Go: the operator, 2026-11-09 about 23:00 CET, in session "LLVQ pistes débit": « Compute HF ok tu as 10$ pour la nuit
comme tu le sens »; on 2026-11-10 the operator said to continue, then "go" (relayed by the orchestrating session). Cap:
**$10**, $3.23 spent before this job. Registry before this job: **$310.48**. Cost: about **$0.48** on l40sx1
(*estimated*, "Cost"), timeout 28 min, **$0.84** at worst. Measured code: commit `afd23b7` ("Tree verify: thread-count
arms a10t384 and a10t256"); the image is published from a clean detached checkout of the commit that holds this file's
stamp, whose kernel, bench and runtime sources are afd23b7's (only the job script and the proofs moved after it).

## Question

N9 found that capping the rows a launch carries (a10r16, a10r8) takes 0.07 to 0.09 of a token off c_1 while the local
memory the install reports does not move. a10 runs 512 threads a CTA at 128 registers a thread. a10t384 and a10t256 run
a10's unchanged pass at 384 and 256 threads a CTA (12 and 8 warps), one CTA a SM, so up to 170 and 255 registers a
thread; each has its own area, slice cap and shared size. Are the row phases bound by register pressure (fewer spills
gain) or by the loads in flight (fewer warps cost)? The job is read for that sign. If a thread arm has no local memory
and is not faster than a10 at c_1, register pressure is not the row phases' cost.

## Setup

One image, one job, `STAGE=n10 ARMS='a10@5.14.5.28.5,a10t384@8.10.8.20.6,a10t256@5.14.5.28.5'
ops/jobs/verify-tree-n2.sh` (one argument, no space), the bench's own timeout 25 min. No compile-only step.

1. `oracle` on Qwen3-0.6B.
2. The sealed 4B at sha256 886391a8; `configs/qwen3-4b-tetra-e4-4e-v9.json` at its bytes.
3. `vtreebench`, one process: 4E v9 as the config builds it, then the tree pass beside it on the same tree-tiled copies:
   - **a10@[5, 14, 5, 28, 5]**: N6 to N9's arm, 512 threads;
   - **a10t384@[8, 10, 8, 20, 6]**: 384 threads; its guard a10@[8, 10, 8, 20, 6], added by the bench untimed;
   - **a10t256@[5, 14, 5, 28, 5]**: 256 threads; its guard a10@[5, 14, 5, 28, 5].
   The bench also adds a9@[5, 14, 5, 28, 5] untimed, a10's own guard. At the chat prompt: `vt1` against 4E's greedy
   tokens at 64 positions and identity; 2 discarded and 5 timed rounds, each a one-token 4E launch then every arm's
   `vt1`, `vt2`, `vt8`, `vt16`; `vt1` and `vt16` stamped, the phase tables; 4E stamped over two tokens. Then the rag
   prompt for the arm with the lowest chat c_16 only.

## Controls

Each voids the run.

1. `oracle` prints MATCH; the file and config at their sha256; no `LLVQ_*` in the job environment.
2. 4E's module is the served v9 module (252,757 bytes, sha256 1b422b62), and every one-token 4E launch makes the
   greedy token it made before.
3. The thread arms' guards: each equals a10 at its own plan bit for bit ("a10t384@... guard against a10@... PASSES", the
   same for a10t256). A failing guard voids its own arm only. A module that fails NVRTC, or that fits no CTA on a SM,
   voids its arm.

Results the decision rule reads, not controls: each arm's identity, and its `vt1` argmax against 4E at 64 of 64.

## What gets published, and what does not get compared

Published: each module's registers, local and shared bytes from its install line; each arm's plan with its units, waves
and fill as printed; each arm's c_R for R = 1, 2, 8, 16, median and range formed round by round; the extra row; the
phase tables, with the row-phase walls and busy min / med / max of vt1 for the three arms. Not compared: a10t384 against
a10 at the same plan (it runs another plan; its guard is a10 at its own plan, untimed); any tok/s at a door, any
acceptance, the 8B and 14B, another card.

## Decision rule

As N1's. Read on the chat context, on the best arm whose guard and identity hold and whose `vt1` makes 4E's argmax at
64 of 64.

| result | outcome |
|---|---|
| c_2 ≤ 1.40 and c_16 ≤ 1.60 | PASS: the reopening condition of `ETAT.md` §6 is met by measurement; reopening stays the operator's call |
| 1.60 < c_16 ≤ 2.35 (or c_16 ≤ 1.60 with c_2 > 1.40) | BETWEEN |
| c_16 > 2.35 | FAIL |

## Signed prediction

Chat context, sealed 4B, one L40S, the v9 config. Signed by the orchestrating session, who states that the sign of the
thread arms is not known in advance.

| quantity | point | interval |
|---|---|---|
| 4E token | 4,450 us | [4,380, 4,500] |
| a10@[5,14,5,28,5] c_1 | 1.75 | [1.72, 1.79] |
| a10@[5,14,5,28,5] c_16 | 1.93 | [1.89, 1.98] |
| a10 registers, local memory | 128, 736 B | |
| a10t384@[8,10,8,20,6] registers | above 128, at most 170 | |
| a10t384 local memory | under 400 B | |
| a10t384 c_1 | 1.70 | [1.45, 2.10] |
| a10t384 c_16 | 1.90 | [1.60, 2.35] |
| a10t256@[5,14,5,28,5] registers | above 128, at most 255 | |
| a10t256 local memory | under 100 B | |
| a10t256 c_1 | 1.75 | [1.45, 2.30] |
| a10t256 c_16 | 1.95 | [1.60, 2.60] |
| outcome | BETWEEN | |

## What this will not establish

Which of the thread count, the plan (a10t384 runs [8, 10, 8, 20, 6]) or the register budget moves a thread arm's time,
beyond the sign the job is read for. R = 4 and 32. Any card but the L40S. The 8B and 14B.

## Cost

N9 ran in 17 min billed, $0.52 (*measured*). This job compiles three timed modules (and the guard-only arms' modules):
about 16 min, $0.48 (*estimated*). Timeout 28 min: $0.84 at worst, under the cap. On l40sx1; if it is not RUNNING after
20 minutes, a twin on l40sx4 (one device visible) is added under a deviation written beside this file before it is
submitted, and the first to run is the measurement; the other is cancelled by hand once one copy is seen RUNNING or
COMPLETED, and the cancellation is confirmed.
