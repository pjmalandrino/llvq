# Prereg N4: the fold protocol taken apart (a3f, a3p, a3t, a3) and a lean fold (a8), sealed 4B

Written on 2026-11-10 at about 06:00 CET on branch `claude/arbre-nuit-2026-11-10`, once the code it describes existed
and passed its host checks (commit f53817c: a8 equals a3 bit for bit at R = 1, 8, 32 under both stream plans; a
quarter read one element off is killed; a3p and a3t run to their end with every row an argmax). Timestamped before the
first measurement, then never edited; a departure goes to `preregistration-verify-tree-n4-2026-11-10-ECARTS.md`.

Go: the operator, 2026-11-09 about 23:00 CET, in session "LLVQ pistes débit": « Compute HF ok tu as 10$ pour la nuit
comme tu le sens ». Night cap: **$10**, $1.08 spent before this job. Registry before this job: **$308.33**. Cost: about
**$0.57** on l40sx1 (*estimated*, "Cost"), timeout 30 min, **$0.90** at worst. Measured code: commit `f53817c` ("Tree
verify a8 (lean fold) and the protocol floor arms a3p, a3t, on a3"); the image is published from a clean detached
checkout of the commit that holds this file's stamp, whose kernel, bench and runtime sources are f53817c's (only the job
script and the proofs moved after it; the job script carries N3's STAGE fix, edaefcd).

## Question

N3 (`docs/mesures/verify-tree-n3-2026-11-10.txt`) found the partial-ticket-fold protocol worth 0.22 of a token at R = 1
and 1.00 at R = 16 (a3 against the floor arm a3f). Which piece costs it: the partial stores, the fences and tickets, or
the fold itself? Two more floor arms take the protocol apart: a3p writes every slice's partial and stops; a3t adds the
two fences and the ticket but does not fold. And a8 keeps a3's arithmetic with a lean fold (one 16-byte load per slice
quarter). The ladder a3f, a3p, a3t, a3 at R = 16 names the piece; a8 against a3 says what the lean fold buys.

## Setup

One image, one job, `STAGE=n4 ARMS='a3@5.14.5.28.5,a8@5.14.5.28.5,a3f@5.14.5.28.5,a3p@5.14.5.28.5,a3t@5.14.5.28.5,
a3@5.7.5.16.5' ops/jobs/verify-tree-n2.sh` (one argument, no space), the bench's own timeout 27 min:

1. `oracle` on Qwen3-0.6B.
2. The sealed 4B at sha256 886391a8; `configs/qwen3-4b-tetra-e4-4e-v9.json` at its bytes.
3. `vtreebench`, one process: 4E v9 as the config builds it, then the tree pass beside it on the same tree-tiled copies.
   P is the plan q/k/v 5, o 14, gate/up 5, down 28, head 5:
   - **a3@P**: the stream unit at P (valid arm);
   - **a8@P**: a3 with the lean fold, bit for bit with a3; a3@P its guard (valid arm);
   - **a3f@P**, **a3p@P**, **a3t@P**: FLOOR arms, timing only, values void by design, no identity, no guard, no vt1
     check, every line labelled "(floor arm, values void)". a3f stores slice 0 as if S were 1 and nothing else; a3p
     also writes every slice's partial; a3t also runs the two fences and the ticket, without folding;
   - **a3@5.7.5.16.5**: the stream unit at another plan (valid arm).
   At the chat prompt: `vt1` against 4E's greedy tokens at 64 positions and identity across R = 1, 2, 8, 16 and two
   shuffled slot orders for the valid arms; 2 discarded and 5 timed rounds, each a one-token 4E launch then every arm's
   `vt1`, `vt2`, `vt16`; `vt1` and `vt16` stamped, the phase tables; 4E stamped over two tokens. Then the rag prompt for
   the valid arm with the lowest chat c_16 only.

## Controls

Each voids the run.

1. `oracle` prints MATCH; the file and config at their sha256; no `LLVQ_*` in the job environment.
2. 4E's module is the served v9 module (252,757 bytes, sha256 1b422b62), and every one-token 4E launch makes the
   greedy token it made before.
3. a8's guard against a3 at P ("a8@5.14.5.28.5 guard against a3@5.14.5.28.5 ... PASSES"): if it fails, a8 is void.
   An arm whose module fails NVRTC is void; the others still count.

Results the decision rule reads, not controls: each valid arm's identity, and its `vt1` argmax against 4E at 64 of 64.

## What gets published, and what does not get compared

Published: each arm's c_R for R = 1, 2, 16, median and range formed round by round; the extra row; the phase tables of
`vt1` and `vt16` for all six arms; the ladder differences a3p - a3f, a3t - a3p, a3 - a3t at R = 1 and R = 16, per row
phase (q/k/v, o, gate/up, down, head); each module's bytes, sha256, registers, local and shared bytes; any NVRTC error
in full. Not compared: the floor arms as served numbers; any tok/s at a door, any acceptance, the 8B and 14B, another
card.

## Decision rule

As N1's, read on valid arms only (a3@P, a8@P, a3@5.7.5.16.5: guard and identity hold, `vt1` at 64 of 64), on the chat
context, on the one with the lowest c_16. The floor arms are never read by the rule.

| result | outcome |
|---|---|
| c_2 ≤ 1.40 and c_16 ≤ 1.60 | PASS: the reopening condition of `ETAT.md` §6 is met by measurement; reopening stays the operator's call |
| 1.60 < c_16 ≤ 2.35 (or c_16 ≤ 1.60 with c_2 > 1.40) | BETWEEN |
| c_16 > 2.35 | FAIL |

## Signed prediction

Chat context, sealed 4B, one L40S, the v9 config; P = plan [5, 14, 5, 28, 5]. Signed by the orchestrating session.

| quantity | point | interval |
|---|---|---|
| 4E token | 4,420 us | [4,350, 4,470] |
| a3@P c_1 | 2.03 | [1.97, 2.09] |
| a3@P c_16 | 2.57 | [2.48, 2.66] |
| a8@P c_1 | 2.02 | [1.94, 2.10] |
| a8@P c_16 | 2.40 | [2.15, 2.62] |
| a3f@P c_1 (floor) | 1.84 | [1.76, 1.92] |
| a3f@P c_16 (floor) | 1.90 | [1.80, 2.02] |
| a3p@P c_1 (floor) | 1.87 | [1.78, 1.98] |
| a3p@P c_16 (floor) | 2.00 | [1.86, 2.25] |
| a3t@P c_1 (floor) | 1.93 | [1.82, 2.05] |
| a3t@P c_16 (floor) | 2.10 | [1.92, 2.45] |
| a3@5.7.5.16.5 c_1 | 2.08 | [1.95, 2.25] |
| a3@5.7.5.16.5 c_16 | 2.55 | [2.35, 2.80] |
| outcome | FAIL, a8@P the closest valid arm | |

## What this will not establish

The floor arms' values are void, so their times are floors for designs that drop parts of the protocol, not served
numbers. Which part of a3t's fences and tickets costs what (they ship together). R = 4, 8 and 32. Any card but the L40S.
The 8B and 14B.

## Cost

N3 ran in 13 min billed, $0.40 (*measured*), four tree-tiled modules and five arms. This job compiles five and times
six: about 19 min, $0.57 (*estimated*), of which about 800 to 850 s of load and install. Timeout 30 min: $0.90 at
worst, under the night cap. On l40sx1; if it is not RUNNING after 15 minutes, a twin on l40sx4 (TIMEOUT 24m, one device
visible) is added under a deviation written beside this file before it is submitted, and the first to run is the
measurement.
