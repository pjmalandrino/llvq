# Prereg N2a: the stream unit beside the tiled pass, at the stream plan, on the sealed 4B

Written on 2026-11-10 at about 02:35 CET on branch `claude/arbre-nuit-2026-11-10`, once the code it describes existed
and passed its host checks (commit 384778a: stream equals tiled bit for bit at R = 1, 2, 8, 16, 32 under two plans
and on the 8B's parts, 6 mutants of 6 killed). Timestamped before the first measurement, then never edited; a
departure goes to `preregistration-verify-tree-n2a-2026-11-10-ECARTS.md`.

Go: the operator, 2026-11-09 about 23:00 CET, in session "LLVQ pistes débit": « Compute HF ok tu as 10$ pour la nuit
comme tu le sens ». Night cap: **$10**, $0.20 spent before this job. Registry before this job: **$307.45**. Cost: about
**$0.27** on l40sx1 (*estimated*, "Cost"), timeout 20 min, **$0.60** at worst. Measured code: commit `b981261` ("N2
job script takes the stage (n2, n2a, n2b), its arms, prereg and output directory"), which carries the stream unit of
384778a and 2c4c541; the image is published from a clean checkout of it.

Earlier submissions under the N2 name, before this prereg: job `6af27248ab34849891a9f1a1` (l40sx1, submitted 02:24
CET) started at once on the previous image (Space sha 9687c6fd, code 88c544f), whose bench refuses the arm list
`a2,a3,a4`: it ran the oracle and no bench (`vtreebench exit 1`), 1 min billed, $0.02. Job `6af2724aab34849891a9f1a8`
(l40sx4) was cancelled in the queue at 02:26 CET and never ran. Neither measured anything this prereg reads.

## Question

N1 (`docs/mesures/verify-tree-n1-2026-11-10.txt`) left a2 at c_1 2.121 and c_16 2.903, its one-row row phases at 6,880
us against about 3,450 us of the same work in 4E's token. The stream unit (a3) stages its slice of the activation once
and lets each warp stream its tile's words with one sub-step read ahead in registers, no `__syncthreads` or `cp.async`
inside the chunk loop. How much of the row phases does the chunk loop cost? This job also compiles the stream module
through NVRTC for the first time, before a4 is stacked on it.

## Setup

One image built from the measured commit, one job, `STAGE=n2a ARMS=a2,a3 ops/jobs/verify-tree-n2.sh`:

1. `oracle` on Qwen3-0.6B.
2. The sealed 4B at sha256 886391a8; `configs/qwen3-4b-tetra-e4-4e-v9.json` at its bytes.
3. `vtreebench <sealed> a2,a3`, one process: 4E v9 as the config builds it, then the tree pass beside it with two
   arms on the same tree-tiled copies and the same stream plan, q/k/v 11, o 14, gate/up 7, down 28, head 5
   (`pers::p4t_splits_stream`):
   - **a2**: the tiled kernel (N1's), at the stream plan;
   - **a3**: the stream unit (`pers_4et_stream.cu`, `P4T_STREAM`).
   At the chat prompt for both arms: `vt1` at 64 positions of 4E's greedy text against 4E's argmax; identity of every
   row across R = 1, 2, 8, 16 and two shuffled slot orders; 2 discarded and 5 timed rounds, each a one-token 4E launch
   then every arm's `vt1`, `vt2`, `vt8`, `vt16`; `vt1` and `vt16` stamped, 5 launches each, the phase table; 4E
   stamped over two tokens. Then the rag prompt for the arm with the lowest chat c_16 only.

## Controls

Each voids the run.

1. `oracle` prints MATCH; the file and config at their sha256; no `LLVQ_*` in the job environment.
2. 4E's module is the served v9 module (252,757 bytes, sha256 1b422b62), and every one-token 4E launch makes the
   greedy token it made before.
3. a3's guard: a3's 32-row pass equals a2's bit for bit, logits and argmax, at the same slices ("a3 guard against a2
   ... PASSES"). If it fails, a3's numbers are void and a2 stands.

Results the decision rule reads, not controls: each arm's identity, and its `vt1` argmax against 4E at 64 of 64.

## What gets published, and what does not get compared

Published: each arm's c_R for R = 1, 2, 8, 16, median and range formed round by round; the extra row; the phase tables
of `vt1` and `vt16`, a2 and a3 side by side; each module's bytes, sha256, registers, local and shared bytes. If the
stream module does not compile, the NVRTC error in full. Not compared: a2 at N1's plan (the plan differs: down 28 and
head 5 here), any tok/s at a door, any acceptance, the 8B and 14B, another card.

## Decision rule

As N1's. Read on the chat context, on the best arm whose guard (for a3) and identity hold and whose `vt1` makes 4E's
argmax at 64 of 64.

| result | outcome |
|---|---|
| c_2 ≤ 1.40 and c_16 ≤ 1.60 | PASS: the reopening condition of `ETAT.md` §6 is met by measurement; reopening stays the operator's call |
| 1.60 < c_16 ≤ 2.35 (or c_16 ≤ 1.60 with c_2 > 1.40) | BETWEEN |
| c_16 > 2.35 | FAIL |

The job is read for the difference between the two arms, phase by phase, and for the stream module's registers, local
and shared bytes.

## Signed prediction

Chat context, sealed 4B, one L40S, the v9 config, both arms on the plan [11, 14, 7, 28, 5]. Signed by the
orchestrating session.

| quantity | point | interval |
|---|---|---|
| 4E token | 4,397 us | [4,350, 4,450] |
| a2 c_1 | 2.15 | [2.05, 2.30] |
| a2 c_16 | 3.05 | [2.80, 3.40] |
| a3 c_1 | 2.05 | [1.85, 2.25] |
| a3 c_16 | 2.90 | [2.55, 3.35] |
| a2 row phases of vt1 (wall, us summed over the layers) | 6,950 | [6,700, 7,300] |
| a3 row phases of vt1 | 6,400 | [5,700, 7,000] |
| outcome | FAIL on both arms (c_16 above 2.35) | |

## Cost

N1 ran in 6 min billed, $0.18 (*measured*), three arms. This job runs two. About 9 min, $0.27 (*estimated*). Timeout
20 min: $0.60 at worst, under the night cap. On l40sx1; if it is not RUNNING after 15 minutes, a twin on l40sx4
(TIMEOUT 15m, one device visible) is added under a deviation written beside this file before it is submitted, and the
first to run is the measurement.
