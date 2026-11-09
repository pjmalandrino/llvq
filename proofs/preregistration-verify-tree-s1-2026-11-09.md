# Prereg S1: the R-row tree verify pass beside 4E's token, on the sealed 4B

Written on 2026-11-09 on branch `claude/verif-arbre-e8b79b`, once the code it describes existed and passed its host
checks (S0, `docs/mesures/tree-verify-s0-host-2026-11-09.txt`). Timestamped before the first measurement, then never
edited; a departure goes to `preregistration-verify-tree-s1-2026-11-09-ECARTS.md`. It replaces the unstamped draft
`BROUILLON-preregistration-verify-tree-s1.md`.

Go: the operator, 2026-11-09, relayed by the parent session: « ok go pour la suite », after the S0 report. Cap in
force: **$0.60, S1 only, operator go of 2026-11-09**. One job, not rerun if it fails without a new go. Registry
before it: $307.13. Cost: about **$0.30** on l40sx1 (*estimated*, "Cost"), timeout 20 min, **$0.60** at worst.
Measured code: the commit that adds `bin/vtreebench` and `ops/jobs/verify-tree-s1.sh`, the one this file names in
its commit message.

## Question

What does a tree verify pass of R = 1, 2, 4, 8, 16, 32 rows cost, in one-token 4E v9 launches, when the rows share
each decoded weight block on the tensor cores and each row's prologue runs once on the grid? Today an extra row
costs 0.74 of a token at 4B (*measured*, spec-final-2026-11-08). The answer decides whether the tree verify goes on
to its quality gate (S2), and it reads the reopening condition of `docs/ETAT.md` §6.

## Setup

One image built from the measured commit, one job, `ops/jobs/verify-tree-s1.sh`:

1. `oracle` on Qwen3-0.6B.
2. The sealed 4B at sha256 886391a8; `configs/qwen3-4b-tetra-e4-4e-v9.json` at its bytes.
3. `vtreebench <sealed>`, one process: 4E v9 as the config builds it, then the tree pass installed beside it in its
   own module (`pers_4et.cu`, 512 threads, the d_in slices of `pers::p4t_splits` on this grid). At the chat and rag
   prompts of `specrun`, each context:
   - 4E decodes 64 greedy tokens;
   - `vt1` at each of those 64 positions, the token 4E read there, the caches 4E wrote;
   - a tree of 32 nodes (node i's parent (i - 1) / 4, depths 0 to 3), passes of its first R nodes for every R, then
     16 and 32 nodes in shuffled slot orders, logits on;
   - 2 discarded and 5 timed rounds, each a one-token 4E launch then `vt1` to `vt32` on the tree's first R nodes,
     host wall from launch to ring read (specrun's plain wall);
   - `vt1` and `vt16` stamped, 5 launches each (`pers::p4t_phases`), and 4E's stamped token.

The draft's row-loop arms (`v2`, `mma1`, `mma8`, `mma16`) are cut: they need a second kernel entry. β is not
measured; the phase stamps give the projection phases' share instead.

## Controls

Each voids the run.

1. `oracle` prints MATCH; the file and config at their sha256; no `LLVQ_*` in the job environment.
2. 4E's module is the served v9 module (252,757 bytes, sha256 1b422b62), and every one-token 4E launch makes the
   greedy token it made before (the bench stops otherwise).

Results the decision rule reads, not controls:

- identity: at both contexts, every row of every pass equals its node in the 32-row pass, logits and argmax byte for
  byte;
- `vt1` against 4E: the same argmax as 4E's next token at 63 of 64 positions or more, at both contexts.

## What gets published, and what does not get compared

Published: c_R = t(vt R) / t(4e), median and range formed round by round, at both contexts; the extra row (c_16 -
c_1) / 15 and (c_8 - c_1) / 7; c_2; the tree symbol's registers and local bytes (a spill is part of the cost
measured, so it voids nothing); the phase table of `vt1` and `vt16`; the tree module's size and sha256.
Not compared: any tok/s at a door, any acceptance, any perplexity (S2 and S3), the 8B and 14B, another card.

## Decision rule

c_16 and c_8 are the larger of the two contexts'.

| result | outcome |
|---|---|
| identity or the `vt1` check fails | stop; the pass is not one function of the row |
| both hold, c_16 ≤ 1.25 | PASS: S2 is proposed with its cost |
| both hold, c_16 > 1.25, and c_16 ≤ 1.60 or c_8 ≤ 1.48 | BETWEEN: the projection is recomputed at the measured c_R; the operator decides |
| both hold, c_16 > 1.60 and c_8 > 1.48 | FAIL: x1.3 on 4B chat is out of reach at the measured tokens a pass; the route stops |

Checked on three values: c_16 1.20 is PASS; c_16 1.40 is BETWEEN; c_16 1.70 with c_8 1.50 is FAIL. The bounds 1.60
and 1.48 are where 4B chat falls to x1.3 at the measured tokens a pass (2.28 at 16 rows and 2.12 at 8, 3 head steps
at 0.05 of a token, eagle-tree-2026-11-09). The reopening of `ETAT.md` §6 is read from the same numbers: an extra row
under 0.2 of a 4E token and c_2 under 1.4. It is reported; reopening stays the operator's call.

## Signed prediction

| quantity | point | interval |
|---|---|---|
| controls, identity, `vt1` check | hold | |
| c_1 | 1.09 | [0.95, 1.25] |
| c_2 | 1.10 | [0.96, 1.27] |
| c_8 | 1.18 | [0.99, 1.41] |
| c_16 | 1.27 | [1.03, 1.60] |
| c_32 | 1.56 | [1.14, 2.03] |
| extra row, R ≤ 16, in 4E tokens | 0.012 | [0.005, 0.023] |
| outcome | BETWEEN | |

Rationale (*estimated*, `docs/plan-verify-tree.md` section 2): the decode is the same, 24 FMAs a row a block become
one B fragment and one `mma.m16n8k16` per 8 rows, the prologues run once a row instead of once a row a CTA, and an
extra row adds its activation bytes from L2 and its attention. Known flaws: the batch GEMM's single row runs at half
4E's speed (batch-l40s-v2); this pass adds 3 grid waits a layer; nothing has compiled it with NVRTC yet.

## Cost

step1-micro ran 18 row-loop arms in 8 min 24 s billed, $0.25 (*measured*). This job loads one model and runs about
200 short launches. About 10 min, $0.30 (*estimated*). Timeout 20 min: $0.60 at worst, the cap in force.
