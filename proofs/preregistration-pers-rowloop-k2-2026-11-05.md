# Preregistration. The k2 arm of the row-loop microbenchmark: one decode for two activations, the cost of a second token in a pass

**NOT STAMPED. NOT LAUNCHED.** Written on 2026-11-05 on branch `claude/serene-bohr-1edf69`, once the code it describes
existed, and before the result of the first row-loop job was read. To be timestamped before the first measurement,
then never edited; a departure goes to `preregistration-pers-rowloop-k2-2026-11-05-ECARTS.md`.

The launch is the operator's, through the integration session. Cost: about **$0.27** on l40sx1 (*estimated*, §8).
Timeout 30 min: **$0.90** at worst (*computed* at $1.80/h). Spent before it: see `docs/data/jobs.csv` at launch.

## 1. Why this run exists

A review of 2026-11-05 proposes speculative decoding on top of 4E: a draft head proposes tokens and one pass verifies
k of them. Its rate is tau, the tokens accepted a pass, times the passes a second. The pass must then carry k
activations through the row phase, each 24-weight block decoded once and applied to the k.

The operator answered yes to adding this arm to the row-loop microbenchmark, at no extra card time. The answer was
relayed by the throughput review session. The first row-loop job had already been stamped and launched without it
(`proofs/preregistration-pers-rowloop-2026-11-05.md`, stamped on integration/lot1 7f32775, job 6aec7403). So the
arm runs under this prereg, in a job of its own (`ops/jobs/pers-rowloop-k2.sh`, output
`pers-rowloop-k2-2026-11-05/`).

This run measures one quantity on fundamental criteria: whether each row of the two-activation path keeps the bits
of the one-activation path, and what a second token costs as a fraction of a pass. It builds nothing for
speculation. tau is measured apart, at $0, on the Mac (a public draft head for Qwen3-4B), before anything depends on
it.

## 2. What the code holds at the commit that carries this file

`llvq-cuda/kernels/pers_trows.cu`, arm `TROWS_K2`, symbol `tv_trows_k2`:
- V2's loop: one block a step, the next block's words in flight;
- each block decoded once (`f1r_v3_quads`), then two 24-FMA dots, one on the bench activation and one on a second
  activation (a seeded gaussian of 9,728 values), each in `TETRA48_ORDER`;
- each folded with `pers_fold` into its own accumulator, then `warp_sum`, then the epilogue on its own activation;
- the second activation staged after the widest first one, 96,896 shared bytes in all, one CTA a SM.

Its two outputs are checked against the served `tv_tetra48_h` on each activation, over the 252 matrices of the bare
Tetra file.

What the k2 code changed for the other arms, against the code of the first job (1b7282f):
- every `tv_trows_*` kernel takes one more argument, `x2`, which only k2 reads;
- the site table carries 10 words instead of 8 (k2's second outputs, 0 for every other arm).
- Their rows do not change: every exact arm still equals the served rows and their f32 values on the host, at 4
  grids. Their registers under the devel ptxas did not move (V2 68, V3 85, ilp2 94, v2u 67).
- The module text did change. So this run times V2 again in the same rounds, and k2's ratio is formed against that
  V2, never against job 6aec7403's.

Host checks at $0:
- `the_k2_arm_is_the_k1_arm_on_each_activation`. k2's rows on the first activation equal V2's, binary16 and f32 before
  `f2h`. On the second they equal the served rows on that activation. Checked at grids 3x96, 2x64, 4x128 and 5x32,
  with and without stamps.
- 4 mutants on k2, each caught: second fold, second activation, its staging, its tail.
- Devel SASS: k2's block loop is 273 instructions against V2's 241. The decode is once (LOP3 46, PRMT 36, IDP 6, as
  V2), and FFMA 50 and LDS.128 12 serve the two activations. 87 registers, no local memory.

## 3. What runs

One image built from the commit that carries this file. One job, l40sx1, `ops/jobs/pers-rowloop-k2.sh`.

1. **Oracle** on Qwen3-0.6B (rule 10).
2. The two banc-252 files checked by size, the Tetra file (`0adb7cfd…`) by sha256. Then `LLVQ_TILE_BLOCKS=64
   LLVQ_BENCH_ARMS=slot32,planes14,fp16,tetra48 LLVQ_SEG_TETRA=1 LLVQ_TROWS=1 LLVQ_DUMP_CUBIN=<dir> planesbench <ball>
   <tetra>`.
   - Every row-loop arm runs, k2 among them, each checked once, one stamped launch, then A4's 7 rounds, 2 discarded.
   - No one-launch pass.

## 4. Gates

Each voids the run:

1. The image builds; `oracle` prints MATCH.
2. The files at their bytes, the Tetra file at its sha256; `tetra48: 252 of 252 matrices matched by name`; `tile 64`.
3. The row loops' reference prints the served unit, 101,625 bytes, sha256 `6193094310bc…`.
4. V2 and V3 print every row identical bit for bit to `tv_tetra48_h`.
5. No `LLVQ_*` variable in the job environment.

## 5. Outcomes reported, never gated

- k2's two identity lines; its registers and local bytes under the driver's ptxas; its cubin.
- k2 and V2, ms and GB/s, median and range.
- f = t(k2) / t(V2) - 1, round by round, median and range.
- k2's stamps: cycles a step per d_in, balance, clock.
- The other arms' lines, read only as this run's own V2 context, never as a second reading of job 6aec7403's
  questions.

## 6. Signed predictions

*Estimated* from the devel SASS and the first job's design; nothing fitted to a number of either run.

| quantity | point | interval |
|---|---|---|
| k2 rows against the served rows, each activation | **identical** | |
| k2 loads without local memory (driver ptxas) | **yes** | |
| f, the second token as a fraction of a pass | **0.15** | [0.05, 0.60] |
| V2, GB/s | **250** | [190, 330] |

f's point is the static instruction ratio, 273 / 241 - 1 = 0.13, rounded up for the second activation's staging. Its
interval is wide because the first job exists precisely because no static model of these loops has held on the card.

## 7. Decision rules

- **A, identity.** Every k2 row equals the served row on its own activation, bit for bit, so the one-activation row
  on it. Then the multi-token row path keeps the served bits. Otherwise k2 is void, and no multi-token row path is
  proposed until the divergence is found on the host with the card's rows.
- **H, the cost.** f = t(k2) / t(V2) - 1, round by round.
  - A verify pass of k tokens costs about 1 + (k - 1) f passes in the row phase.
  - With tau measured apart, the served rate becomes about tau / (1 + (k - 1) f) times the one-token rate, before
    attention and the draft head.
  - f <= 0.25 keeps the door worth its complexity at tau >= 2.
  - f >= 0.60 closes it for the row phase as built.
  - In between: a point of the curve. The decision is the operator's, on tok/s served once tau is known.
- **Caveat, stated before the clock.** k2 holds 96,896 shared bytes, which leaves about 28 KB of L1 against about
  64 KB beside V2's 57,984. Part of f may be that, and this run does not separate it.
- **G.** A miss is reported as a miss, with the term of §6 it refutes. No threshold moves after the clock.

## 8. Cost

*Estimated* on l40sx1 at $1.80/h:

| step | time |
|---|---|
| container and image | 3 min |
| oracle | 1 to 2 min |
| planesbench: A4's arms and the row-loop module, no one-launch pass | about 4 min |

About 9 min, **$0.27**. Timeout 30 min: **$0.90** at worst.

## 9. What this run does not establish

- tau, the draft head, the tree attention, the KV rollback: none runs.
- Any served tok/s: no 4E pass carries two activations.
- k above 2, which needs the activations staged in tiles (beyond 99 KB of shared memory).
