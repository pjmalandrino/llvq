# Prereg N3: three levers on the stream unit (the down's input, the L2 prefetch, the fold protocol), sealed 4B

Written on 2026-11-10 at about 05:00 CET on branch `claude/arbre-nuit-2026-11-10`, once the code it describes existed
and passed its host checks (commit 6238370: a6 equals a3 and a7 equals a6 bit for bit at R = 1, 8, 32 under both stream
plans; a deal that skips the last piece is killed; a3f runs to its end with every row an argmax). Timestamped before
the first measurement, then never edited; a departure goes to `preregistration-verify-tree-n3-2026-11-10-ECARTS.md`.

Go: the operator, 2026-11-09 about 23:00 CET, in session "LLVQ pistes débit": « Compute HF ok tu as 10$ pour la nuit
comme tu le sens ». Night cap: **$10**, $0.68 spent before this job. Registry before this job: **$307.93**. Cost: about
**$0.54** on l40sx1 (*estimated*, "Cost"), timeout 28 min, **$0.84** at worst. Measured code: commit `6238370` ("Tree
verify a6, a7 and the floor arm a3f, on a3"); the image is published from a clean detached checkout of the commit that
holds this file's stamp, whose kernel, bench and runtime sources are 6238370's (only the job script and the proofs
moved after it).

## Question

N2a and N2b left the stream unit a3 at c_1 2.03 and c_16 2.91, with about 890 us of vt1 in the down's input (dn_in),
which one CTA computes a row while the others wait, and a down phase of about 1,130 us. This job measures three levers
on a3: dealing an int4 down's input over the whole grid (a6), adding 4E's three L2 prefetch points (a7), and what the
partial-ticket-fold protocol costs in all (a3 against a3f, a floor arm that keeps the row loops and drops the protocol).

## Setup

One image, one job, `STAGE=n3 ARMS='a3,a6,a6@5.14.5.28.5,a7,a3f' ops/jobs/verify-tree-n2.sh`, the bench's own timeout
raised to 25 min:

1. `oracle` on Qwen3-0.6B.
2. The sealed 4B at sha256 886391a8; `configs/qwen3-4b-tetra-e4-4e-v9.json` at its bytes.
3. `vtreebench <sealed> a3,a6,a6@5.14.5.28.5,a7,a3f`, one process: 4E v9 as the config builds it, then the tree pass
   beside it on the same tree-tiled copies, plan q/k/v 11, o 14, gate/up 7, down 28, head 5 unless named:
   - **a3**: the stream unit (N2a's, N2b's);
   - **a6**: a3 with an int4 down's input dealt over the grid in 8-element pieces (a Tetra down keeps a3's path);
     a3 its guard;
   - **a6@5.14.5.28.5**: a6 at the plan q/k/v 5, o 14, gate/up 5, down 28, head 5; its guard is a3 at the same plan,
     added by the bench as an untimed guard-only arm;
   - **a7**: a6 with a5's three L2 prefetch points; a6 its guard;
   - **a3f**, a floor arm, not a served arm: a slice with sl > 0 stores nothing and slice 0 stores as if S were 1.
     Its values are void by design: no identity, no guard, no vt1 check; every line it prints says "a3f (floor arm,
     values void)".
   At the chat prompt for every arm: `vt1` at 64 positions of 4E's greedy text against 4E's argmax (not a3f);
   identity across R = 1, 2, 8, 16 and two shuffled slot orders (not a3f); 2 discarded and 5 timed rounds, each a
   one-token 4E launch then every arm's `vt1`, `vt2`, `vt16`; `vt1` and `vt16` stamped, the phase table; 4E stamped
   over two tokens. Then the rag prompt for the valid arm with the lowest chat c_16 only.

## Controls

Each voids the run.

1. `oracle` prints MATCH; the file and config at their sha256; no `LLVQ_*` in the job environment.
2. 4E's module is the served v9 module (252,757 bytes, sha256 1b422b62), and every one-token 4E launch makes the
   greedy token it made before.
3. Guards, each voiding its own arm only: a6 against a3, a6@5.14.5.28.5 against a3@5.14.5.28.5, a7 against a6, each a
   32-row pass equal bit for bit, logits and argmax ("... guard against ... PASSES"). An arm whose module fails NVRTC
   is void; the others still count.

Results the decision rule reads, not controls: each valid arm's identity, and its `vt1` argmax against 4E at 64 of 64.

## What gets published, and what does not get compared

Published: each arm's c_R for R = 1, 2, 16, median and range formed round by round; the extra row (c_16 - c_1) / 15;
the phase tables of `vt1` and `vt16`, all arms side by side; each module's bytes, sha256, registers, local and shared
bytes; any NVRTC error in full. Not compared: a3f against anything but as a timing floor; any tok/s at a door, any
acceptance, the 8B and 14B, another card, R = 8 in the timing (it no longer runs).

## Decision rule

As N1's, read on valid arms only (guard and identity hold, `vt1` makes 4E's argmax at 64 of 64), on the chat context,
on the one with the lowest c_16. a3f is never read by the rule.

| result | outcome |
|---|---|
| c_2 ≤ 1.40 and c_16 ≤ 1.60 | PASS: the reopening condition of `ETAT.md` §6 is met by measurement; reopening stays the operator's call |
| 1.60 < c_16 ≤ 2.35 (or c_16 ≤ 1.60 with c_2 > 1.40) | BETWEEN |
| c_16 > 2.35 | FAIL |

The job is read for the size of three levers: the down's input (a6 against a3), the prefetch (a7 against a6), and
what the partial-ticket-fold protocol costs in total (a3 against a3f).

## Signed prediction

Chat context, sealed 4B, one L40S, the v9 config; plan [11, 14, 7, 28, 5] unless the arm names another. Signed by the
orchestrating session.

| quantity | point | interval |
|---|---|---|
| 4E token | 4,400 us | [4,350, 4,450] |
| a3 c_1 | 2.03 | [1.98, 2.09] |
| a3 c_16 | 2.91 | [2.84, 3.00] |
| a3 dn_in of vt1 (wall, us summed over the layers) | 890 | [860, 920] |
| a3 down of vt1 | 1,128 | [1,050, 1,200] |
| a6 c_1 | 1.86 | [1.78, 2.00] |
| a6 c_16 | 2.74 | [2.62, 2.92] |
| a6 dn_in of vt1 | 150 | [60, 400] |
| a6@5.14.5.28.5 c_1 | 1.76 | [1.62, 1.95] |
| a6@5.14.5.28.5 c_16 | 2.70 | [2.45, 2.95] |
| a7 c_1 | 1.76 | [1.60, 1.90] |
| a7 c_16 | 2.64 | [2.45, 2.90] |
| a7 down of vt1 | 780 | [600, 1,000] |
| a3f c_1 (floor arm, timing only) | 1.60 | [1.35, 1.90] |
| a3f c_16 (floor arm, timing only) | 2.00 | [1.60, 2.45] |
| outcome | FAIL: no valid arm under c_16 2.35 | |

## What this will not establish

a3f's values are void, so its time is a floor for a design that keeps the row loops and drops the protocol, not a
served number. Which of a7's three prefetch points buys what. a6's lever on a Tetra down (those layers keep a3's
path). R = 4, 8 and 32. Any card but the L40S. The 8B and 14B.

## Cost

N2b ran in 9 min billed, $0.26 (*measured*), two arms. This job compiles four tree-tiled modules and times five arms:
about 18 min, $0.54 (*estimated*). Timeout 28 min: $0.84 at worst, under the night cap. On l40sx1; if it is not
RUNNING after 15 minutes, a twin on l40sx4 (TIMEOUT 22m, one device visible) is added under a deviation written beside
this file before it is submitted, and the first to run is the measurement.
