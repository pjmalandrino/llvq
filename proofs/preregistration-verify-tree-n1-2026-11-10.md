# Prereg N1: the tree verify pass, balanced slices and tree-tiled weights, on the sealed 4B

Written in the night of 2026-11-09 to 2026-11-10 on branch `claude/arbre-nuit-2026-11-10`, once the code it describes
existed and passed its host checks. Timestamped before the first measurement, then never edited; a departure goes to
`preregistration-verify-tree-n1-2026-11-10-ECARTS.md`.

Go: the operator, 2026-11-09 about 23:00 CET, in session "LLVQ pistes débit": « Compute HF ok tu as 10$ pour la nuit
comme tu le sens ». Night cap: **$10**. Registry before this job: **$307.25**. Cost: about **$0.36** on l40sx1
(*estimated*, "Cost"), timeout 25 min, **$0.75** at worst. One job, not rerun if it fails without a new decision of
the orchestrating session under the night cap. Measured code: commit `88c544f` ("Tree verify N1: balanced slices and tree-tiled weights, three arms in one bench"); the image is built from it.

## Question

S1 (job 6af20dab, `docs/mesures/verify-tree-s1-2026-11-09.txt`) measured the tree pass at c_1 2.798 and c_16 3.092
4E tokens on chat, an extra row 0.0196. Reading `pers_4et.cu` (*computed*, not measured) names two causes of the fixed
cost: the q/k/v and gate/up phases deal 144 and 152 units on 142 CTAs, two waves at about half fill; and each CTA
copies 256 row streams of 4-byte `cp.async.ca` pieces, where 4E copies 16 contiguous streams. Does a split plan that
fills the grid (a1), and then tree-tiled weight copies read in 16-byte pieces (a2), bring the pass near 4E's token?

## Setup

One image built from the measured commit, one job, `ops/jobs/verify-tree-n1.sh`:

1. `oracle` on Qwen3-0.6B.
2. The sealed 4B at sha256 886391a8; `configs/qwen3-4b-tetra-e4-4e-v9.json` at its bytes.
3. `vtreebench <sealed> a0,a1,a2`, one process: 4E v9 as the config builds it, then the tree pass beside it with
   three arms. All three keep S0/S1's arithmetic (binary16 weights with folded scales, binary16 activation,
   `mma.m16n8k16`, f32 accumulation); only where the bytes sit and how units are dealt changes.
   - **a0**, reference: S1's kernel, S1's slices (q/k/v 6, o 14, gate/up 2, down 14, head 1).
   - **a1**, balanced: S1's kernel, slices q/k/v 11, o 14, gate/up 7, down 14, head 1. Units, waves and fill on 142
     CTAs (*computed*, `pers::p4t_plan`, printed by the install): q/k/v 264 units, 2 waves, 93.0 % (S1: 144, 2,
     50.7 %); o 140, 1, 98.6 %; gate/up 532, 4, 93.7 % (S1: 152, 2, 53.5 %); down 140, 1, 98.6 %; head 594, 5,
     83.7 %.
   - **a2**, balanced and tiled: a1's slices, the `P4T_TILED` module reading tree-tiled copies of every projection
     (16-row tile, 48-column sub-step, its rows' words contiguous), staged in 16-byte `cp.async.cg` pieces. The
     copies are built once on the card from the words 4E reads (`p4t_tile_words`); about 1.3 GB more on the card
     (*estimated*; the install prints it).
   At the chat prompt of `specrun`, for every arm: `vt1` at 64 positions of 4E's greedy text against 4E's argmax;
   identity of every row across R = 1, 2, 8, 16 and two shuffled slot orders against the arm's 32-row pass; 2
   discarded and 5 timed rounds, each a one-token 4E launch then every arm's `vt1`, `vt2`, `vt8`, `vt16`, host wall
   from launch to ring read; `vt1` and `vt16` stamped, 5 launches each, the phase table (wall, and the min, median and
   max over the CTAs of a CTA's busy time, summed over the layers); 4E stamped over two tokens. Then the rag prompt
   for the arm with the lowest chat c_16 only.

## Controls

Each voids the run.

1. `oracle` prints MATCH; the file and config at their sha256; no `LLVQ_*` in the job environment.
2. 4E's module is the served v9 module (252,757 bytes, sha256 1b422b62), and every one-token 4E launch makes the
   greedy token it made before (the bench stops otherwise).
3. a2's 32-row pass equals a1's bit for bit, logits and argmax (same slices, same arithmetic). If it does not, a2's
   numbers are void and a0 and a1 stand.

The a0 module's sha256 is not S1's `0701126a`: `pers_4et.cu` now carries the tiled branches as templates and the
phase stamps record a CTA's arrival at each wait as well as its leaving. Without `P4T_TILED` the module emits the
untiled symbols only, whose decode, staging and MMA are S1's line for line; an unstamped launch tests one more null
pointer a wait, and a stamped launch adds one `__syncthreads` a wait. The bench prints both modules' bytes, sha256, registers, local and shared bytes.

Results the decision rule reads, not controls: each arm's identity, and its `vt1` argmax against 4E at 64 of 64.

## What gets published, and what does not get compared

Published: each arm's c_R for R = 1, 2, 8, 16, median and range formed round by round; the extra row (c_16 - c_1) /
15; the phase tables of `vt1` and `vt16`; the modules' size, sha256, registers and local bytes; the tiled copies'
bytes. Not compared: any tok/s at a door, any acceptance, any perplexity, the 8B and 14B, another card, R = 32.

## Decision rule

Read on the chat context, on the best arm (lowest c_16) whose identity holds and whose `vt1` makes 4E's argmax at
64 of 64.

| result | outcome |
|---|---|
| c_2 ≤ 1.40 and c_16 ≤ 1.60 | PASS: the reopening condition of `ETAT.md` §6 is met by measurement; reopening stays the operator's call |
| 1.60 < c_16 ≤ 2.35 (or c_16 ≤ 1.60 with c_2 > 1.40) | BETWEEN: the next step is the stream unit (job N2) |
| c_16 > 2.35 | FAIL: tiling and balance do not explain the fixed cost; the phase tables say where to look |

2.35 is where DFlash's measured chat tau breaks even with greedy. Checked on three values: c_2 1.30 and c_16 1.50 is
PASS; c_16 2.00 is BETWEEN; c_16 2.50 is FAIL. No arm passing identity and `vt1` leaves the rule unread: the outcome is
then "no arm stands".

## Signed prediction

Chat context, sealed 4B, one L40S, the v9 config. Signed by the orchestrating session.

| quantity | point | interval |
|---|---|---|
| 4E token | 4,398 us | [4,350, 4,450] |
| a0 c_1 | 2.80 | [2.70, 2.90] |
| a0 c_16 | 3.09 | [2.98, 3.20] |
| a0 row phases (q/k/v, o, gate/up, down, head) in vt1's wall | 86 % | [78, 92] |
| a1 c_1 | 2.05 | [1.85, 2.40] |
| a1 c_16 | 2.40 | [2.10, 2.85] |
| a2 c_1 | 1.35 | [1.10, 1.90] |
| a2 c_2 | 1.37 | [1.12, 1.92] |
| a2 c_16 | 1.70 | [1.40, 2.30] |
| extra row, every arm, in 4E tokens | 0.020 | [0.012, 0.035] |
| outcome | BETWEEN | |

## Cost

S1 ran its job in 4 min billed (*measured*, $0.12, load 191 s). This job loads one model, builds the tiled copies once
and runs about 600 short launches. About 12 min, $0.36 (*estimated*). Timeout 25 min: $0.75 at worst, under the night
cap of $10.
