# Prereg N5: the slice fold after the phase's grid wait (a9), and the protocol ladder, sealed 4B

Written on 2026-11-10 at about 08:20 CET on branch `claude/arbre-nuit-2026-11-10`, once the code it describes existed
and passed its host checks (commit 8ade3a5: a9 equals a3 bit for bit at R = 1, 2, 8, 16, 32 under both stream plans
and on the 8B's parts; 5 mutants of 5 killed). Timestamped before the first measurement, then never edited; a
departure goes to `preregistration-verify-tree-n5-2026-11-10-ECARTS.md`.

Go: the operator, 2026-11-09 about 23:00 CET, in session "LLVQ pistes débit": « Compute HF ok tu as 10$ pour la nuit
comme tu le sens »; on 2026-11-10 in the morning the operator said to continue (relayed by the orchestrating session).
Cap: **$10**, $1.08 spent before this job. Registry before this job: **$308.33**. Cost: two copies submitted at once
(deviation 1, written beside this file before the submission), about **$0.57** on l40sx1 (timeout 32 min, $0.96) or
**$2.63** on l40sx4 (timeout 26 min, $3.60) (*estimated*), the first to run being the measurement. Measured code:
commit `8ade3a5` ("Tree verify a9: the slice fold after the phase's grid wait, on a3"); the image is published from a
clean detached checkout of the commit that holds this file's stamp, whose kernel, bench and runtime sources are
8ade3a5's (only the job script and the proofs moved after it).

N5 supersedes N4 (`preregistration-verify-tree-n4-2026-11-10.md`), which never started: both its copies were cancelled
in the queue after about two hours (N4's ECARTS, deviation 2). N5 carries N4's ladder arms a3f, a3p and a3t and its a3 at
[5, 14, 5, 28, 5], under N4's predictions; a8 is not rerun.

## Question

N3 found the partial-ticket-fold protocol worth 0.22 of a token at R = 1 and 1.00 at R = 16 (a3 against the floor a3f).
a9 keeps a3's arithmetic and moves the fold out of the units: each slice writes its partial and stops, and after the
phase's grid wait every CTA folds a share of the (row, 4-row quarter) pairs, then one more wait (472 waits a pass at the
4B instead of 327). Does it remove the protocol's cost? And the ladder a3f, a3p (partials stored), a3t (plus fences and
tickets), a3 (plus the in-unit fold) names which piece costs the token.

## Setup

One image, one job, `STAGE=n5 ARMS='a3@5.14.5.28.5,a9@5.14.5.28.5,a9@11.14.7.28.5,a3f@5.14.5.28.5,a3p@5.14.5.28.5,
a3t@5.14.5.28.5' ops/jobs/verify-tree-n2.sh` (one argument, no space), the bench's own timeout 29 min:

1. `oracle` on Qwen3-0.6B.
2. The sealed 4B at sha256 886391a8; `configs/qwen3-4b-tetra-e4-4e-v9.json` at its bytes.
3. `vtreebench`, one process: 4E v9 as the config builds it, then the tree pass beside it on the same tree-tiled
   copies. P is the plan q/k/v 5, o 14, gate/up 5, down 28, head 5:
   - **a3@P**: the stream unit at P (valid arm);
   - **a9@P**: a9 at P, a3@P its guard (valid arm);
   - **a9@11.14.7.28.5**: a9 at the stream plan, a3@11.14.7.28.5 its guard, added by the bench as an untimed
     guard-only arm (valid arm);
   - **a3f@P**, **a3p@P**, **a3t@P**: FLOOR arms, timing only, values void by design, no identity, no guard, no vt1
     check, every line labelled "(floor arm, values void)".
   At the chat prompt: `vt1` against 4E's greedy tokens at 64 positions and identity across R = 1, 2, 8, 16 and two
   shuffled slot orders for the valid arms; 2 discarded and 5 timed rounds, each a one-token 4E launch then every arm's
   `vt1`, `vt2`, `vt16`; `vt1` and `vt16` stamped, the phase tables (a9's with a fold phase after each row phase); 4E
   stamped over two tokens. Then the rag prompt for the valid arm with the lowest chat c_16 only.

## Controls

Each voids the run.

1. `oracle` prints MATCH; the file and config at their sha256; no `LLVQ_*` in the job environment.
2. 4E's module is the served v9 module (252,757 bytes, sha256 1b422b62), and every one-token 4E launch makes the
   greedy token it made before.
3. a9's guards: a9@P against a3@P and a9@11.14.7.28.5 against a3@11.14.7.28.5, each a 32-row pass equal bit for bit
   ("a9... guard against a3... PASSES"); a failing guard voids its own arm only. An arm whose module fails NVRTC is
   void; the others still count.

Results the decision rule reads, not controls: each valid arm's identity, and its `vt1` argmax against 4E at 64 of 64.

## What gets published, and what does not get compared

Published: each arm's c_R for R = 1, 2, 16, median and range formed round by round; the extra row; the phase tables of
`vt1` and `vt16`, a3@P and a9@P side by side with a9's fold rows; the ladder a3f, a3p, a3t, a3 at R = 1 and 16 per row
phase; each module's bytes, sha256, registers, local and shared bytes; any NVRTC error in full. Not compared: the floor
arms as served numbers; any tok/s at a door, any acceptance, the 8B and 14B, another card.

## Decision rule

As N1's, read on valid arms only (a3@P, a9@P, a9@11.14.7.28.5: guard and identity hold, `vt1` at 64 of 64), on the chat
context, on the one with the lowest c_16. The floor arms are never read by the rule.

| result | outcome |
|---|---|
| c_2 ≤ 1.40 and c_16 ≤ 1.60 | PASS: the reopening condition of `ETAT.md` §6 is met by measurement; reopening stays the operator's call |
| 1.60 < c_16 ≤ 2.35 (or c_16 ≤ 1.60 with c_2 > 1.40) | BETWEEN |
| c_16 > 2.35 | FAIL |

## Signed prediction

Chat context, sealed 4B, one L40S, the v9 config; P = [5, 14, 5, 28, 5]. Signed by the orchestrating session; the
floor lines are N4's, carried over unchanged.

| quantity | point | interval |
|---|---|---|
| 4E token | 4,420 us | [4,350, 4,470] |
| a3@P c_1 | 2.03 | [1.97, 2.09] |
| a3@P c_16 | 2.57 | [2.48, 2.66] |
| a9@P c_1 | 1.95 | [1.85, 2.08] |
| a9@P c_2 | 1.96 | [1.85, 2.09] |
| a9@P c_16 | 2.10 | [1.92, 2.35] |
| a9@P extra row, in 4E tokens | 0.010 | [0.004, 0.025] |
| a9@11.14.7.28.5 c_1 | 2.00 | [1.88, 2.15] |
| a9@11.14.7.28.5 c_16 | 2.20 | [1.95, 2.50] |
| a3f@P c_1 (floor) | 1.84 | [1.76, 1.92] |
| a3f@P c_16 (floor) | 1.90 | [1.80, 2.02] |
| a3p@P c_1 (floor) | 1.87 | [1.78, 1.98] |
| a3p@P c_16 (floor) | 2.00 | [1.86, 2.25] |
| a3t@P c_1 (floor) | 1.93 | [1.82, 2.05] |
| a3t@P c_16 (floor) | 2.10 | [1.92, 2.45] |
| outcome | BETWEEN, on a9@P | |

## What this will not establish

The floor arms' values are void, so their times are floors for designs that drop parts of the protocol, not served
numbers. Which part of a3t's fences and tickets costs what (they ship together). Whether a9's extra 145 waits a pass
cost more on another card or at another size. R = 4, 8 and 32. Any card but the L40S. The 8B and 14B.

## Cost

N3 ran in 13 min billed, $0.40 (*measured*). This job compiles five tree-tiled modules and times six arms: about 19
min (*estimated*), of which about 800 to 850 s of load and install. l40sx1: about $0.57, timeout 32 min, $0.96 at
worst; l40sx4: about $2.63, timeout 26 min, $3.60 at worst. Only one copy runs; even both to their timeouts ($4.56)
keep the night under its cap.
