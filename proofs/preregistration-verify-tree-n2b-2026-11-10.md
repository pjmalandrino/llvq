# Prereg N2b: a4 (the stream unit, the fold batched, vector prologues) beside a3, on the sealed 4B

Written on 2026-11-10 at about 04:00 CET on branch `claude/arbre-nuit-2026-11-10`, once the code it describes existed
and passed its host checks (commit 50aaf6e: a4 equals a3 bit for bit at R = 1, 2, 8, 16, 32 under both stream plans
and on the 8B's parts, stamped equals unstamped, 5 mutants of 5 killed). Timestamped before the first measurement,
then never edited; a departure goes to `preregistration-verify-tree-n2b-2026-11-10-ECARTS.md`.

Go: the operator, 2026-11-09 about 23:00 CET, in session "LLVQ pistes débit": « Compute HF ok tu as 10$ pour la nuit
comme tu le sens ». Night cap: **$10**, $0.42 spent before this job. Registry before this job: **$307.67**. Cost: about
**$0.27** on l40sx1 (*estimated*, "Cost"), timeout 20 min, **$0.60** at worst. Measured code: commit `50aaf6e` ("Tree
verify a4: the stream unit with the fold batched, vector prologues, phase parts"); the image is published from a clean
detached checkout of the commit that holds this file's stamp, whose kernel, bench and runtime sources are 50aaf6e's
(only the job script, the journals and the proofs moved after it).

## Question

N2a (`docs/mesures/verify-tree-n2a-2026-11-10.txt`) left the stream unit a3 at c_1 2.035 and c_16 2.916, with about
1,650 us of vt1 in the prologues ln1, ln2 and dn_in that one CTA runs a row while the others wait, and an extra row of
0.059 of a token. a4 is a3 with the slice fold batched (F1: 8 independent loads before the adds, the same order) and the
prologues on 16-byte vectors (F2: 4E v9's batching, the same element arithmetic). How much of the fixed cost and of the
extra row do they take off? The F3 stamps split each row phase into setup, staging, loops and store.

## Setup

One image, one job, `STAGE=n2b ARMS=a3,a4 ops/jobs/verify-tree-n2.sh`:

1. `oracle` on Qwen3-0.6B.
2. The sealed 4B at sha256 886391a8; `configs/qwen3-4b-tetra-e4-4e-v9.json` at its bytes.
3. `vtreebench <sealed> a3,a4`, one process: 4E v9 as the config builds it, then the tree pass beside it with two arms
   on the same tree-tiled copies and the same stream plan, q/k/v 11, o 14, gate/up 7, down 28, head 5:
   - **a3**: the stream unit (`pers_4et_stream_*`), N2a's;
   - **a4**: a3 with F1, F2 and the F3 stamps (`pers_4et_a4_*`).
   At the chat prompt for both arms: `vt1` at 64 positions of 4E's greedy text against 4E's argmax; identity of every
   row across R = 1, 2, 8, 16 and two shuffled slot orders; 2 discarded and 5 timed rounds, each a one-token 4E launch
   then every arm's `vt1`, `vt2`, `vt8`, `vt16`; `vt1` and `vt16` stamped, 5 launches each, the phase table and, for
   a4, the F3 parts table; 4E stamped over two tokens. Then the rag prompt for the arm with the lowest chat c_16 only.

## Controls

Each voids the run.

1. `oracle` prints MATCH; the file and config at their sha256; no `LLVQ_*` in the job environment.
2. 4E's module is the served v9 module (252,757 bytes, sha256 1b422b62), and every one-token 4E launch makes the
   greedy token it made before.
3. a4's guard: a4's 32-row pass equals a3's bit for bit, logits and argmax, at the same slices ("a4 guard against a3
   ... PASSES"). If it fails, a4's numbers are void and a3 stands. a3's own guard against a2 is N2a's (a2 does not run
   here).

Results the decision rule reads, not controls: each arm's identity, and its `vt1` argmax against 4E at 64 of 64.

## What gets published, and what does not get compared

Published: each arm's c_R for R = 1, 2, 8, 16, median and range formed round by round; the extra row; the phase tables
of `vt1` and `vt16`, a3 and a4 side by side; a4's F3 parts tables in full; each module's bytes, sha256, registers,
local and shared bytes. If a4's module does not compile, the NVRTC error in full. Not compared: any tok/s at a door,
any acceptance, the 8B and 14B, another card.

## Decision rule

As N1's. Read on the chat context, on the best arm whose guard (for a4) and identity hold and whose `vt1` makes 4E's
argmax at 64 of 64.

| result | outcome |
|---|---|
| c_2 ≤ 1.40 and c_16 ≤ 1.60 | PASS: the reopening condition of `ETAT.md` §6 is met by measurement; reopening stays the operator's call |
| 1.60 < c_16 ≤ 2.35 (or c_16 ≤ 1.60 with c_2 > 1.40) | BETWEEN |
| c_16 > 2.35 | FAIL |

## Signed prediction

Chat context, sealed 4B, one L40S, the v9 config, both arms on the plan [11, 14, 7, 28, 5]. Signed by the
orchestrating session.

| quantity | point | interval |
|---|---|---|
| 4E token | 4,400 us | [4,350, 4,450] |
| a3 c_1 | 2.035 | [1.98, 2.09] |
| a3 c_16 | 2.92 | [2.84, 3.00] |
| a4 c_1 | 1.81 | [1.62, 2.00] |
| a4 c_2 | 1.82 | [1.62, 2.01] |
| a4 c_16 | 2.15 | [1.85, 2.50] |
| a4 extra row, in 4E tokens | 0.022 | [0.010, 0.038] |
| a3 prologues of vt1 (ln1 + ln2 + dn_in, wall, us summed over the layers) | 1,672 | [1,600, 1,750] |
| a4 prologues of vt1 | 650 | [450, 950] |
| outcome | BETWEEN, on a4 | |

## What this will not establish

Nothing on the card reports whether F2's vector path or its scalar fallback ran (they give the same bits); the
prologue times are the only witness. Which of F1 and F2 buys which part of a4's gain (they ship together). R = 4 and
32. Any card but the L40S.

## Cost

N2a ran in 7 min billed, $0.22 (*measured*), two arms. This job runs two. About 9 min, $0.27 (*estimated*). Timeout 20
min: $0.60 at worst, under the night cap. On l40sx1; if it is not RUNNING after 15 minutes, a twin on l40sx4 (TIMEOUT
15m, one device visible) is added under a deviation written beside this file before it is submitted, and the first to
run is the measurement.
