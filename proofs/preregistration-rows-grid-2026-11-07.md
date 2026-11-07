# Preregistration. plan-300's row lead: do the grid barriers cost the rows, and do grid-wide row claims take it back?

**NOT STAMPED. NOT LAUNCHED.** Written on 2026-11-07 on branch `claude/route-300-toks-de97fe`, once the code it
describes existed. To be timestamped before the first measurement, then never edited; a departure goes to
`preregistration-rows-grid-2026-11-07-ECARTS.md`.

The operator asked on 2026-11-07 to focus on one lead, the weight rows inside 4E at their solo speed: cause first, then
the smallest bench that settles it, a 4E form only if the bench licenses one, at the bit. The card waits for a go.
Cost: about **$0.27** on l40sx1 (*estimated*, §8). Timeout 30 min: **$0.90** at worst (*computed* at $1.80/h). Spent
before it: $274.46 (*computed*, every branch's registry).

## 1. What the record already establishes ($0)

The request was framed as 260 GB/s in 4E against 379 alone, with 379 read at 32 warps a SM and 4E at 16. The
journals say otherwise on three points.

- **379 GB/s is at 16 warps a SM.** pers-rowloop ran v2 at one CTA of 512 threads a SM, 4E's own geometry: 379 GB/s
  [368-383] (*measured*). At 32 warps (`v2_o2`, two CTAs a SM, launch bound (512, 2), so at most 64 registers) the
  same loop reads ×1.076 over one CTA (*measured*, on the sites of d_in 4,096 or less).
- **260 GB/s is v2 and v3's figure, not today's.** pers-4e-v2 and v3 ran the stream loop (row form 1). Since v4 (row
  form 2) 4E runs v2's loop. On v6's stamps (step3-4e, whose stamp symbol does not spill), mean CTA work at 5 tokens:
  - gate/up rows: 0.467 GB in 1,199.3 µs, **390 GB/s**; at the slowest CTA of each layer, 1,335.4 µs, 350 GB/s;
  - every row phase, Tetra and int4: 1.18 GB in 2,390 µs, 493 GB/s. The int4 rows exceed any solo rate because 4E
    prefetches later phases' rows into L2 during the phases that leave DRAM idle (`pers_l2_prefetch`).
  (*computed*; bytes at 2.085 b/weight for the Tetra stream, pers-rowloop's 0.9467 GB over 3.633 G weights, and 4.25
  for int4 g128.)
- **On mean work the rows already run at their solo speed.** What they lose is at the grid barriers. Every row phase
  ends at one, so each layer waits for its slowest CTA. Per-layer slowest minus mean, summed over layers (v6, 5
  tokens): q/k/v rows 36.0 µs, gate/up 136.1, down rows 44.7, the o phase 81.4. That is **298 µs a token**, and the
  218 barrier hops add 0.771 µs each (step1-micro), 168 µs. Alone, the bench's sites run back to back with no grid
  barrier, so a CTA's slow site averages out with its fast ones.

So two hypotheses of the request fall on the record:
- **Occupancy** is worth ×1.076 at most, and 4E cannot reach it. Its CTA needs 57,984 to 85,952 B of shared memory,
  so two CTAs do not fit an SM's 100 KB. 1,024 threads at 64 registers would spill the 4E body: v7 spills 24 B at 128,
  and the 768-thread symbol spilled 176 B at 85.
- **Contention between CTAs** is the open one, as a spread at the barriers.

Two remain open, and this run measures them:
- **Registers.** Does the row loop alone need 128 a thread at 16 warps?
- **The spread's cure.** Is the spread something a fast CTA can take over, so that rows claimed across the grid
  recover it? Or is it fixed (per-SM clock, L2 distance), the same whichever rows a CTA takes?

## 2. What the code holds at the commit that carries this file

`pers_trows.cu`, three arms added to the row-loop microbenchmark (`LLVQ_TROWS=1`, `pers::TROWS_ARMS`):
- **gb**, `tv_trows_gb`: v2's rows, the static split (`lo + warp + k x 16`), and 4E's grid barrier (`pers_grid_wait`,
  the served one) after every site. Owes the bits.
- **gbc**, `tv_trows_gbc`: gb with each site's rows claimed across the whole grid from a cursor in global memory, one
  row a warp, the next claim issued before the current row decodes. Owes the bits: a row's bits do not depend on its
  warp. Its counters (barrier, abort flag, a cursor a site) are zeroed by the host before each launch, as gb's.
- **v2r64**: `tv_trows_v2_o2` (at most 64 registers) at one CTA a SM on every site.
- Each stamped launch now prints the CTAs' spread: over the sites, the sum of the slowest CTA's busy cycles over the
  mean's.

Host checks at $0 (`llvq-cuda/tests/pers_matches_served.rs`, under nice 10):
- `the_rows_grid_arms_are_the_served_rows`: gb and gbc give the served rows on five sites (d_in 2,560, 4,096, 9,728,
  128) at five grids and warp counts, with and without stamps. The harness checks that every CTA arrived at the
  barrier after every site.
- `rows_grid_mutants_are_killed`: 5 mutants caught (gbc's first and next claim, its bound, its per-site cursor, gb's
  barrier). gb's row choice has no mutant: without its branch it falls to the uniform loop, which owes the same bits.
- The library suite (126), the row-loop and step-1 tests, clippy on the Linux half of llvq-cuda.

## 3. What runs

One image built from the commit that carries this file. One job, l40sx1, `ops/jobs/rows-grid.sh`: the oracle, then
`planesbench` on the two banc-252 files with `LLVQ_TROWS=1`, as step1-micro ran it. Every arm of step 1 rides along.
The three new arms run in the same interleaved rounds.

## 4. Gates

Each voids the run:
1. The image builds; `oracle` prints MATCH.
2. The two files at their bytes; the Tetra file at `0adb7cfd`.
3. The references are the served unit (101,625 bytes, `6193094310bc…`).
4. v2, gb and gbc keep the served rows bit for bit; q4 keeps tv_q4_hh's.
5. No `LLVQ_*` variable in the job environment.

## 5. Outcomes reported, never gated

Each arm's ms, GB/s and speed over its base round by round (gb and v2r64 over v2, gbc over gb); registers and local
bytes; the stamped launch's cycles a step, warp balance and CTA spread.

## 6. Signed predictions

*Estimated*. The barrier term is 252 hops at 0.771 µs over v2's 2.50 ms; the spread term is v6's in-situ 1.11 on
gate/up. Nothing fitted to a number of this run.

| quantity | point | interval |
|---|---|---|
| v2, GB/s | **377** | [360, 395] |
| gb over v2, round by round | **x0.86** | [0.78, 0.95] |
| CTA spread, v2 | **1.10** | [1.03, 1.25] |
| CTA spread, gb | **1.10** | [1.03, 1.25] |
| CTA spread, gbc | **1.02** | [1.00, 1.06] |
| **gbc over gb, round by round** | **x1.06** | [0.97, 1.12] |
| v2r64 over v2, round by round | **x0.97** | [0.85, 1.02] |
| v2_o2 over v2_s (replication) | **x1.076** | [1.03, 1.12] |
| every symbol 0 B local | **yes** | |

## 7. Decision rules

- **A, identity.** §4.
- **B, the row lead (gate in served tok/s).** gbc over gb at its median, round by round, r:
  - r at or over **x1.05**: licenses v10, v9 with every row phase claimed across the grid. Its gain is projected as
    (1 − 1/r) × 2.26 ms, v6's slowest-CTA time in its three row phases (q/k/v 434.0, gate/up 1,335.4, down 494.2 µs).
    It is capped at the 0.30 ms of measured spread. At r = 1.05 that is 0.11 ms: v9's 238.6 tok/s in the measurement
    form becomes about 245, and its door about 227, **+5 tok/s served**, which is the gate.
  - r between x1.02 and x1.05: reported, no 4E form.
  - r under x1.02: dead. The spread is fixed per SM, the barriers' cost cannot be moved by assignment, and the rows'
    lead closes at the bit.
- **C, the barrier's cost alone.** gb over v2 says what the barriers take from the rows alone. With gb's CTA spread
  at or under 1.03, the cost is the hops, not the spread.
- **D, registers.** v2r64 over v2 at x0.98 or more: the row loop needs no more than 64 registers at 16 warps, and 4E's
  128 are spent elsewhere (the phases around it). Under x0.98: the loop uses its registers. Either way, occupancy is
  not reachable (§1), and no 4E form follows from D alone.
- **E, replication.** v2 inside [360, 395] and v2_o2 over v2_s inside [1.03, 1.12], or the round's numbers are
  reported against that miss.
- **G.** A miss is reported as a miss, with the term of §6 it refutes. No threshold moves after the clock.

## 8. Cost

*Estimated* on l40sx1 at $1.80/h. step1-micro billed 8 min 24 s ($0.25) for the same process. The three arms add a
few ms a round and three stamped launches. About 9 min, **$0.27**. Timeout 30 min: **$0.90** at worst.

## 9. What this run does not establish

- Any 4E tok/s: v10 is not written. A pass of rule B would license writing it and a card reading of its own.
- The spread at 8,192 positions, where attention dominates.
- The 8B and the 14B; any card but the L40S.
