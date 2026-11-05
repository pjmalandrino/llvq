# Preregistration. The "shaker": 4E v3, v4 and v5 with their inner stamps, and the row loops' k2 arm, in one card job

**NOT STAMPED. NOT LAUNCHED.** Written on 2026-11-05 on branch `claude/serene-bohr-1edf69`, once the code it describes
existed. To be timestamped before the first measurement, then never edited; a departure goes to
`preregistration-shaker-4e-2026-11-05-ECARTS.md`.

The launch is the operator's, through the integration session. Cost: about **$0.99** on l40sx1 for the whole job,
parts A, B and C (*estimated*, §8). Timeout 60 min: **$1.80** at worst (*computed* at $1.80/h). Spent before it: see
`docs/data/jobs.csv` at launch.

This prereg covers parts A and B. It replaces `proofs/preregistration-pers-rowloop-k2-2026-11-05.md` (never stamped,
never launched, removed with this file), whose arm is part B here. Part C runs the stamped protocol of
`proofs/preregistration-served-4e-2026-11-05.md` unchanged. Its departure (another job script and another output
directory) is written beside it, in `-ECARTS.md`.

## 1. Why this run exists

4E v3 decodes the sealed 4B at 179.2 tok/s after a 5-token prompt (*measured*, pers-4e-v3). The row loops alone ran at
379 GB/s against about 260 inside 4E (*measured*, pers-rowloop). Rule D of that run's prereg then showed where the
token goes (`docs/mesures/pers-4e-insitu-host-2026-11-05.txt`, *computed* from measured cycles):
- the rows predict 59 % of 4E's MLP;
- 1,660 us a token of fixed work sits around the rows, the same with either loop;
- 420 us are lost to rows split unevenly over 16 warps;
- v3's stream loop costs about 170 us against v2's.

This run measures three changes and one quantity, in one job:
- v4, v2's loop in v3's structure;
- v5, v4 with each CTA's last rows split over its warps;
- the inner stamps that name the fixed work;
- the cost of a second token in a pass (k2), the door to speculative decoding.

## 2. What the code holds at the commit that carries this file

- `llvq_pers.cuh`, `PERS_4E_ROWS`, the row form of 4E's 512-thread symbols, chosen when the module is built
  (`persist4e_cuda::install_rows`):
  - 1 = v3's stream. It is the default, and the served config's form.
  - 2 = v4: `pers_tetra_rows_static`, v2's row (one block a step, the next block's words in flight) on v3's static
    rows, with v3's mixed q/k/v split and short-context attention staging.
  - 3 = v5: form 2 with `pers_4e_rows_cta`. A CTA's rows modulo its 16 warps are split into (row, lane step) units
    dealt to every warp; each unit keeps `pers_fold`'s two factors in a 16 KB shared scratch. After a CTA barrier, the
    row's owner folds them in the served order, then `warp_sum` and the epilogue. For gate/up, the pair values wait in
    the scratch for their product after a barrier. A CTA with more than 512 pairs, or units past the scratch, runs
    form 2's path.
- The inner stamps: 15 events a layer instead of 10, in the order they happen, written by the stamp symbol only.
  - q/k/v: ln1 and its rotation done.
  - MLP: ln2 and its rotation done; gate/up done; the inner wait released; the down's input ready.
  - `pers::p4e_phase_report` reads them as 7 parts, each CTA work, which sum to their phase's mean work.
- `planesbench LLVQ_TROWS=1`: the row-loop arms of the first row-loop job, plus `k2` (one decode for two
  activations, each row owing the bits of the one-activation row on its own activation).

Host checks at $0 (`llvq-cuda/tests/pers_matches_served.rs`, `cargo test` under nice 10):
- v4: q/k/v, the token loop at two RoPE forms and at positions 270-271, and the stamped symbol's outputs and inner
  stamps, all equal to the served chain; 4 mutants caught.
- v5: the same at grids 3 to 7, and at 20 x 96, where gate/up's split is taken (CTAs of 486-487 pairs, 2 rows left
  over on 3 warps); 5 mutants caught.
- The inner stamps: the report's own test, and the parts summing to their phases on the host's stamps.
- k2: its rows equal the k = 1 rows on each activation at 4 grids; 4 mutants caught.
- devel ptxas 12.4 (indicative): `pers_4e_tokens_f1_t512` 120 / 122 / 121 registers for forms 1 / 2 / 3, and
  125 / 122 / 121 for the stamp symbols, all with 0 B stack.

## 3. What runs

One image built from the commit that carries this file. One job, l40sx1, `ops/jobs/shaker-4e.sh`. The sealed 4B
(`886391a8…`) and the two banc-252 files are checked by size, then sha256 for the sealed and Tetra files.

1. **Oracle** on Qwen3-0.6B (rule 10).
2. **Part A.** `LLVQ_FUSED_LAYOUT=tetra48 LLVQ_EMBED=q4 LLVQ_ROT_SHARE=1 LLVQ_FUSE=0 LLVQ_PERS_4E=64 fusedrun <sealed>
   64`.
   - Three runtimes at 512 threads, forms 1, 2, 3, each its own module.
   - At 5 tokens (measurement form, kernel on every row) and at 8,192 (the served config, inside the T8).
   - Per context: each form's tokens against the eager path's, 2 + 5 interleaved rounds, then one stamped launch a
     form.
3. **Part B.** `LLVQ_TILE_BLOCKS=64 LLVQ_BENCH_ARMS=slot32,planes14,fp16,tetra48 LLVQ_SEG_TETRA=1 LLVQ_TROWS=1
   LLVQ_DUMP_CUBIN=<dir> planesbench <ball> <tetra>`.
4. **Part C.** The served-4e protocol, as its prereg describes, under `C/`.

## 4. Gates

Each voids the run:

1. The image builds; `oracle` prints MATCH.
2. The files at their bytes and sha256.
3. Part A:
   - event tracking off;
   - a RoPE form equal to candle's;
   - **4E v3**'s tokens equal to the eager path's at both contexts.
4. Part B:
   - `tetra48: 252 of 252 matrices`, `tile 64`;
   - the reference is the served unit `6193094310bc…`;
   - v2's rows bit for bit.
5. No `LLVQ_*` variable in the job environment.

v4 or v5 moving a token, or not loading, voids only its own numbers. So does k2 moving a row. Part C's gates are its
own prereg's.

## 5. Outcomes reported, never gated

- Each form's registers and local bytes under the driver's ptxas.
- Per context:
  - each form's tok/s, median and range, and its ratio to the eager path round by round;
  - v4 over v3 and v5 over v4, round by round, formed from the same rounds.
- Each form's phase table and its 7 inner parts (CTA work, slowest and mean).
- Part B: every row-loop arm, as in the first row-loop job; k2's f = t(k2) / t(V2) - 1, round by round.

## 6. Signed predictions

*Estimated* from the measured row-loop cycles and the 4E v3 table (ops/pers_4e_insitu.py); nothing fitted to a number
of this run.

| quantity | point | interval |
|---|---|---|
| v3, v4, v5 tokens against the eager path, both contexts | **identical** | |
| v4 and v5 load without local memory (driver ptxas) | **yes** | |
| v3 at 5 tokens, tok/s | **179** | [176, 182] |
| v4 at 5 tokens, tok/s | **184** | [176, 194] |
| v5 at 5 tokens, tok/s | **195** | [183, 210] |
| v4 at 8,192, tok/s | **138.6** | [134, 144] |
| v5 at 8,192, tok/s | **144.8** | [138, 152] |
| v4 over v3, 5 tokens, round by round | **x1.03** | [1.00, 1.07] |
| v5 over v4, 5 tokens, round by round | **x1.06** | [1.01, 1.12] |
| inner part, ln1+rotation, us a token (v3, mean CTA) | **250** | [150, 450] |
| inner part, ln2+rotation, us a token | **250** | [150, 450] |
| inner part, MLP down input, us a token | **400** | [200, 800] |
| inner part, MLP inner wait, us a token | **60** | [20, 200] |
| inner part, MLP gate/up rows, us a token (v3) | **1,060** | [900, 1,400] |
| k2: rows against the k = 1 rows, each activation | **identical** | |
| k2: f, the second token as a fraction of a pass | **0.15** | [0.05, 0.60] |

The rotation and wait points use the isolated measurements of lot1 and L1 (rot_apply at 1,024 threads, a 768-thread
barrier); 4E runs 512 threads in every CTA, hence the wide intervals.

## 7. Decision rules

- **A, identity.** Each form whose tokens equal the eager path's at both contexts keeps the served bits on this card.
  A form that moves a token is void, and its first divergence goes to the host.
- **B, the forms.** v4 over v3 and v5 over v4, round by round.
  - A form faster than the one before it, with its whole range above x1.00, is proposed for serving: the served
    config's 4E would run it.
  - The served engine changes only on the operator's yes, after part C's reading, with the config's row form named.
  - Otherwise the form stays a measurement.
- **C, the inner parts.** Each part is a named cost.
  - The three rows parts (q/k/v rows, gate/up rows, down rows) are checked backward against the row-loop cycles
    times 4E's row counts. Within 15 %: the rows model holds in situ. Outside: the gap is a cost of its own, named
    with the part it sits in.
  - A non-row part at 300 us a token or more is a lever, proposed on its own measured number: the rotations (a faster
    per-CTA rotation, or one shared across the grid), the inner wait, the down's input.
  - No lever is proposed without its part's number.
- **H, k tokens a pass (k2).** As in the first row-loop work:
  - every k2 row equals the k = 1 row on its activation, or the arm is void;
  - f is reported round by round. A verify pass of k tokens costs about 1 + (k - 1) f passes in the row phase.
    With tau measured apart, the served rate is about tau / (1 + (k - 1) f) times the one-token rate, before
    attention and the draft head.
  - f <= 0.25 keeps the door worth it at tau >= 2; f >= 0.60 closes it for the row phase as built.
  - Caveat: k2 holds 96,896 shared bytes, which leaves about 28 KB of L1 against about 64 KB beside V2.
- **G.** A miss is reported as a miss, with the term of §6 it refutes. No threshold moves after the clock.

## 8. Cost

*Estimated* on l40sx1 at $1.80/h:

| step | time |
|---|---|
| container and image | 3 min |
| oracle | 1 to 2 min |
| part A (three 4E modules built, 2 contexts x 7 rounds x 4 decodes, 6 stamped launches; v3's own job ran 215 s with one form) | 6 min |
| part B (planesbench, A4's arms and the row loops, no one-launch pass) | 4.5 min |
| part C (the served-4e protocol, its own estimate less the container) | 18 min |

About 33 min, **$0.99**. Timeout 60 min: **$1.80** at worst.

## 9. What this run does not establish

- Serving v4 or v5: part C runs the served config's form, v3.
- tau, the draft head, a tree attention, a KV rollback.
- The 8B and the 14B; any card but the L40S.
