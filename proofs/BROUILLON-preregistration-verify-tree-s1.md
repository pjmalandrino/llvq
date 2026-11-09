# Prereg S1 (draft): the R-row tensor-core verify pass beside 4E's token, on the sealed 4B

Status: draft, NOT timestamped. It authorizes no measurement. The `BROUILLON-` prefix goes when the operator gives the
go and the code of stage S0 exists; the file is then renamed, committed and timestamped, and never edited again.
Cost: about $0.30 on l40sx1 (*estimated*, section "Cost"), hard cap $0.60 (timeout 20 min at $1.80/h). No spend cap is
in force on 2026-11-09; one is owed before this job. Registry before it: $307.13.
Measured code: the commit that carries stage S0 of `docs/plan-verify-tree.md` (to come).

## Question

What does a verify pass of R = 1, 2, 4, 8, 16, 32 rows cost, in one-token 4E v9 launches, when the rows share each
decoded weight block on tensor cores and each row's prologue runs once on the grid? Today an extra row costs 0.74 of
a token at 4B (*measured*, spec-final-2026-11-08). This decides whether the tree verify goes on to its quality gate
(S2), and it reads the reopening condition of `docs/ETAT.md` §6.

## Setup

One image from the S0 commit, one job, `ops/jobs/verify-tree-s1.sh` (to be written with the code):

1. `oracle` on Qwen3-0.6B.
2. The sealed 4B at sha256 886391a8; the served config `configs/qwen3-4b-tetra-e4-4e-v9.json` at its bytes.
3. `vtreebench <sealed> 1,2,4,8,16,32 62,362` (name to be fixed by S0): in one process and one context, 7 rounds, 2
   discarded, arms interleaved round by round:
   - `4e`: one 4E v9 token launch, the served module;
   - `vt<R>`: the tree verify pass of R rows at the same position, R = 1, 2, 4, 8, 16, 32, with the tree mask of an
     EAGLE-2 tree of R - 1 nodes taken from `eagle-tree-2026-11-09-brut` at R = 8 and 16, a chain elsewhere;
   - `v2`, `mma1`, `mma8`, `mma16`: the Tetra row loop alone over the 252 matrices of the 4B (step1-micro's banc),
     v2 the served loop, `mmaR` the verify's loop at R rows.
4. Phase stamps (`%globaltimer` at each grid wait, spec-final's method) on `vt1` and `vt16`.

## Controls

Each voids the run.

1. `oracle` prints MATCH; the files and config at their sha256; no `LLVQ_*` in the job environment.
2. `4e` makes the served door's greedy tokens at the house prompt (fingerprint 61d0654532d35120).
3. Every verify symbol loads with 0 B of local memory under the driver's ptxas.

Two checks are results, not controls, and the first row of the decision rule reads them:

- identity across R: row r's logits are bit for bit equal at every R and every slot permutation of the tree;
- `vt1` against `4e` at the same position: worst relative RMS of the logits at most 1e-2 (the batch GEMM's check),
  the same argmax on at least 63 of 64 positions.

## What gets published, and what does not get compared

Published: c_R = t(vt<R>) / t(4e), median and range formed round by round, at both contexts; the extra row
(c_16 - c_1) / 15 and (c_8 - c_1) / 7; c_2; registers and local bytes; the GB/s of `v2` and `mmaR` and the ratio
β = t(mma1) / t(v2); the phase shares of `vt1` and `vt16`.
Not compared: any tok/s at a door, any acceptance, any perplexity (S2 and S3), the 8B and 14B, another card.

## Decision rule

| result | outcome |
|---|---|
| identity or the `vt1` check fails | stop; the verify arithmetic is not one function of the row |
| identity and the `vt1` check hold, c_16 ≤ 1.25 | PASS: S2 is proposed with its cost |
| they hold, c_16 > 1.25, and c_16 ≤ 1.60 or c_8 ≤ 1.48 | BETWEEN: the projection is recomputed at the measured c_R; the operator decides |
| they hold, c_16 > 1.60 and c_8 > 1.48 | FAIL: x1.3 on 4B chat is out of reach at the measured tokens a pass; the route stops |

Checked on three values: c_16 1.20 is PASS; c_16 1.40 is BETWEEN; c_16 1.70 with c_8 1.50 is FAIL. The bounds 1.60
and 1.48 are where 4B chat falls to x1.3 at the measured tokens a pass (2.28 at 16 rows and 2.12 at 8, 3 head steps
at 0.05 of a token, eagle-tree-2026-11-09).
The reopening of `ETAT.md` §6 is read from the same numbers: an extra row under 0.2 of a 4E token (the EAGLE row)
and c_2 under 1.4 (the 0.6B row). It is reported, and reopening stays the operator's call.

## Signed prediction

| quantity | point | interval |
|---|---|---|
| controls, identity, `vt1` check | hold | |
| β = t(mma1) / t(v2) | 1.10 | [0.90, 1.30] |
| c_1 | 1.09 | [0.95, 1.25] |
| c_2 | 1.10 | [0.96, 1.27] |
| c_8 | 1.18 | [0.99, 1.41] |
| c_16 | 1.27 | [1.03, 1.60] |
| c_32 | 1.56 | [1.14, 2.03] |
| extra row, R ≤ 16, in 4E tokens | 0.012 | [0.005, 0.023] |
| outcome | BETWEEN | |

Rationale (*estimated*, `docs/plan-verify-tree.md` section 2): the decode is the same, 24 FMAs a row a block become
one B fragment and one `mma.m16n8k16` per 8 rows, the prologues run once a row instead of once a row a CTA, and an
extra row adds its activation bytes from L2 and its attention. Known flaw: the batch GEMM's single row runs at half
4E's speed (batch-l40s-v2), and nothing measured says an MMA row loop decodes at v2's 374 GB/s.

## Cost

step1-micro ran 18 row-loop arms in 8 min 24 s billed, $0.25 (*measured*). This job adds 13 short launches a round
and the phase stamps. About 10 min, $0.30 (*estimated*). Timeout 20 min: $0.60 at worst.
