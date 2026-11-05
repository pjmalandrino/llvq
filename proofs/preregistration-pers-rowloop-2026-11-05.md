# Preregistration. 4E's Tetra row loops alone on the card: rates, cycles a step, warp balance, and which suspect binds

**NOT STAMPED. NOT LAUNCHED.** Written on 2026-11-05 on branch `claude/serene-bohr-1edf69`, once the code it describes
existed. To be timestamped before the first measurement, then never edited; a departure goes to
`preregistration-pers-rowloop-2026-11-05-ECARTS.md`.

The launch is the operator's, through the integration session. Cost: about **$0.30** on l40sx1 (*estimated*, §8).
Timeout 30 min: **$0.90** at worst (*computed* at $1.80/h). Spent before it: see `docs/data/jobs.csv` at launch.

## 1. Why this run exists

The target is the sealed 4B at 300 tok/s or more after a short prompt on the L40S. 4E v3 decodes at 179.2 tok/s
(*measured*, `docs/mesures/pers-4e-v3-2026-11-05.txt`). Its Tetra rows stream at about 256 GB/s on the MLP phase's
mean CTA work, and 300 needs about 610 (*computed*).

Two estimates in a row overpredicted the card by 35 to 50 %: v2 240 against 178.7, v3 273 against 179.2. The host
reading of v3's SASS (`docs/mesures/pers-4e-rowloop-host-2026-11-05.txt`) finds no limit inside the SM that explains
the rate. Issue, the dependency chain, the shared pipe and the words in flight each predict 900 GB/s or more
(*computed*). Two card readings on file already refute two models:
- the L1/shared pipe: tables in shared memory, x0.948;
- the per-row round trips: v3 removed them, and the rows did not move.

The coordinator's order is to calibrate before proposing any new 4E variant. This run measures the row loops
themselves, so the next estimate is fitted on measured rates, and it tests five suspects at once.

## 2. What the code holds at the commit that carries this file

Off by default; no served path moves. `planesbench` with `LLVQ_TROWS=1` builds its own NVRTC module, the persistent
unit followed by `llvq-cuda/kernels/pers_trows.cu`. 4E's module keeps its bytes.

The geometry is 4E's:
- one CTA of 512 threads a SM, forced by 4E's own 57,984 shared bytes;
- the decoder tables and the activation in shared memory, laid out as `pers_4e.cu` lays them out;
- each site's rows split over CTAs as 4E splits a phase, then over warps statically (`lo + warp + k x 16`);
- a CTA barrier between sites, no grid barrier.

The sites are the 252 matrices of the bare Tetra file (`qwen3-4b-tetra.bin`, sha256 `0adb7cfd…`) in served order.
The reference rows are the served `tv_tetra48_h`'s, from the served unit (digest `6193094310bc…`, a gate).

Arms:

| arm | what it is | owes the bits |
|---|---|---|
| v2 | 4E v2's row: `pers_tetra_row<24, false, true>`, then `pers_epilogue` | yes |
| v3 | 4E v3's stream: `pers_tetra_stream`, 4 blocks in flight across rows | yes |
| ilp2 | two blocks a step in one straight-line region, folded in the served order | yes |
| v2u | v2 with a trip count uniform over the warp; control of the next three | yes |
| l1w | v2u reading row 0's words for every row: the words never come from DRAM | no |
| bcast | v2u with every lane decoding lane 0's words: every table read is a broadcast | no |
| xbcast | v2u with every lane reading the step's lane-0 activation block: the LDS.128 are broadcasts | no |
| v2_s | v2 on the 216 sites of d_in 4,096 or less (all but the downs), one CTA a SM | yes |
| v2_o2 | the same at two CTAs a SM (32 warps), launch bound (512, 2) | yes |

Each arm is checked against the served rows once, then runs one stamped launch outside the rounds. The stamps hold
every warp's busy cycles and the CTA's per site, and `%clock64` and `%globaltimer` at the launch's ends. Then the arms
run in A4's 7 rounds, 2 discarded, each round timed on the host.

Host checks at $0:
- `the_row_loop_arms_are_the_served_rows`: the four exact arms give the served rows and their f32 values before
  `f2h` on 5 sites, at 4 grids, with and without stamps. The three diagnostics move rows.
- `row_loop_mutants_are_killed`: 8 mutants, all caught.
- Under the devel ptxas 12.4 (indicative): 63 to 94 registers, no local memory; v2_o2 at 63 (two CTAs need 64 or
  fewer). Never run on a card.

## 3. What runs

One image built from the commit that carries this file. One job, l40sx1, `ops/jobs/pers-rowloop.sh`.

1. **Oracle** on Qwen3-0.6B (rule 10).
2. The two banc-252 files are checked by size, and the Tetra file by sha256. Then `LLVQ_TILE_BLOCKS=64
   LLVQ_BENCH_ARMS=slot32,planes14,fp16,tetra48 LLVQ_SEG_TETRA=1 LLVQ_TETRA_OCC=1 LLVQ_TROWS=1 LLVQ_DUMP_CUBIN=<dir>
   planesbench <ball> <tetra>`. That is L1's command with the row loops added; the one-launch pass and its four
   variants are the run-to-run anchor.

## 4. Gates

Each voids the run:

1. The image builds; `oracle` prints MATCH.
2. The files at their bytes, the Tetra file at its sha256; `tetra48: 252 of 252 matrices matched by name`; `tile 64`.
3. The row loops' reference prints the served unit, 101,625 bytes, sha256 `6193094310bc…`.
4. v2 and v3 print every row identical bit for bit to `tv_tetra48_h`.
5. No `LLVQ_*` variable in the job environment.

Another arm that spills, does not fit, or moves a row voids only its own numbers.

## 5. Outcomes reported, never gated

- Each symbol's registers and local bytes under the driver's ptxas; the cubins.
- Per arm: ms and GB/s, median and range, on its bytes: words at the row stride, binary16 tails, row scales.
- Per arm, its speed over its base round by round:
  - ilp2 and v3 over v2;
  - l1w, bcast and xbcast over v2u;
  - v2_o2 over v2_s.
- From the stamps, per arm and per d_in (2,560, 4,096, 9,728): cycles a step (32 blocks of one row), in cycles and
  in us; the warp balance (mean busy warp time over the CTA's time); and the SM clock in MHz.
- tpersall and its variants as in L1 and v2.

## 6. Signed predictions

*Estimated* from the 4E v2 and v3 card tables and the host reading; nothing fitted to a number of this run. The
diagnostic arms have no mechanism behind them yet. Their points are a prior, and the decision rules do the work.

| quantity | point | interval |
|---|---|---|
| v2, v3, ilp2, v2u, v2_s, v2_o2 rows against tv_tetra48_h | **identical** | |
| every symbol loads without local memory (driver ptxas) | **yes** | |
| SM clock during the stamped launches, MHz | **2,500** | [2,100, 2,520] |
| warp balance at d_in 4,096 and at 9,728 (v2) | **0.563** (row counts) | [0.51, 0.62] |
| warp balance at d_in 2,560 (v2) | **0.805** (row counts) | [0.75, 0.86] |
| v2, cycles a step at d_in 2,560 | **3,200** | [2,200, 4,500] |
| v2, GB/s | **250** | [190, 330] |
| v3 over v2 | **x0.95** | [0.85, 1.03] |
| ilp2 over v2 | **x1.20** | [0.95, 1.70] |
| l1w over v2u | **x1.30** | [0.95, 2.50] |
| bcast over v2u | **x1.10** | [0.95, 1.60] |
| xbcast over v2u | **x1.10** | [0.95, 1.50] |
| v2_o2 over v2_s | **x1.30** | [1.00, 1.80] |
| tpersall, GB/s | **397** | [380, 410] |

## 7. Decision rules

- **A, identity.** An arm that owes the bits and moves a row is void, and the first divergence goes to the host with
  the card's rows. Gate 4 covers v2 and v3.
- **B, the stamps.**
  - The warp balance at d_in 4,096 and 9,728 lies within 0.05 of 0.563: a warp's rows take the same time
    whichever warp runs them. Then cycles a step read as the loop's cost.
  - Outside: rows slow each other unevenly inside a CTA. Cycles a step are then read per class with that caveat.
- **C, the bound.** Take each speed over its base:
  - l1w: the words' path from DRAM and L2;
  - bcast: the table gathers;
  - xbcast: the activation reads;
  - ilp2: the latency of a warp's decode chain;
  - v2_o2: the warps a SM.
  - Every one at 1.30 or more is a lever, ranked by speed.
  - None at 1.15 or more: none of the five binds. The next reading is a profiler (an `nsys` or `ncu` path on HF
    Jobs), and no 4E variant is proposed on a model.
- **D, calibration.** The next 4E estimate is built from three numbers of this run, and from no assumed Tetra rate:
  - the measured cycles a step of v2 and v3 per d_in;
  - the measured balance;
  - the measured clock.
  Its first check is backward. Take v3's cycles a step and 4E's own row counts per warp; they must give 4E v3's MLP
  mean work (2,975 us) within 10 %. If they do not, the in-situ phase has costs the bench does not carry: the SiLU
  epilogue, the grid barriers, the restaging. The gap is named before any forward estimate.
- **E, the "Λ-transposed read".** Each of its mechanisms is an arm:
  - bcast bounds pieces 1 and 4 (table reads made free);
  - xbcast bounds piece 3's broadcast activation;
  - ilp2 bounds piece 3's independent per-lane rows.
  - Its claimed 2-3x stands only if bcast x xbcast >= 2 or ilp2 >= 2. Otherwise the reading caps what it can buy.
- **F, v3's slower MLP.**
  - v3 over v2 at 0.97 or less: the stream loop's own cost explains it.
  - 0.99 or more: it does not, and the in-situ epilogue or the static split does.
- **G.** A miss is reported as a miss, with the term of §6 it refutes. No threshold moves after the clock.

## 8. Cost

*Estimated* on l40sx1 at $1.80/h:

| step | time |
|---|---|
| container and image | 3 min |
| oracle | 1 to 2 min |
| planesbench: L1's 270 s, plus the row-loop module and the served reference (about 1 min) | about 5.5 min |

About 10 min, **$0.30**. Timeout 30 min: **$0.90** at worst.

## 9. What this run does not establish

- Any 4E tok/s. No 4E variant runs.
- The in-situ costs the bench does not carry: the SiLU epilogue, the grid barriers, attention, norms, rotations.
- Any card but the L40S.
