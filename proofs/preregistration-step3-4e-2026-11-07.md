# Preregistration. plan-300 step 3: 4E v9 (K11, the int4 rows on tiled words), and the cause of v8's slower rows

**NOT STAMPED. NOT LAUNCHED.** Written on 2026-11-07 on branch `claude/route-300-toks-de97fe`, once the code it
describes existed. To be timestamped before the first measurement, then never edited; a departure goes to
`preregistration-step3-4e-2026-11-07-ECARTS.md`.

The operator's go was relayed on 2026-11-07: the next package of the route, at the bit, one job of about $1, to be
asked again above $1 or for a second job. Cost: about **$0.36** on l40sx1 (*estimated*, §8). Timeout 25 min:
**$0.75** at worst (*computed* at $1.80/h). Spent before it: $274.18 (*computed*: this branch's registry, $273.77,
plus the served-v6 door's $0.41 recorded on its branch).

## 1. Why this run exists, and which leads it carries

Step 2 (job 6aef3f69, `docs/mesures/step2-4e-2026-11-07.txt`) measured v8 at 227.9 tok/s at 5 prompt tokens, ×1.076
over v6 round by round, and 155.6 at 8,192, ×1.025. The candidates for the next package were read against the code
and the measured numbers. Ranked by served tok/s:

- **K11, int4 rows on tiled words: coded, v9.** The only remaining lead with a measured lever: step 1's q4t read
  ×1.338 over 4E's int4 row alone, at the same bits. 4E's int4 rows take about 0.697 ms a token in situ (pers-4e-v3).
- **v8's slower rows: one diagnostic arm, v6s.** v8's stamps put its rows 175 µs above v6's at 5 tokens. Two causes
  are open. (a) The carve-out: v8 asks for 85,952 B of shared memory, so the SM keeps 100 KB of its 128 KB as
  shared memory and L1 shrinks from about 64 KB to 28 KB. (b) The stamp symbol: v8's spills 48 B, and its stamped
  token costs 214 µs over the timed one, against 14 for v6. v6s, v6's module asking for v8's shared bytes, holds v6's
  code and bits under v8's carve-out, and isolates (a).
- **K3a's second half, the job spread: not coded.** At a short prompt a row holds one 256-position split, so attention
  runs one job a KV head, 8 jobs on 8 CTAs. Inside a job the units and the query-head pairs already run on separate
  warps. The critical path is one unit's tiles in order: up to 8 tiles of 8 positions, a chain the canonical
  arithmetic fixes. Spreading jobs over more CTAs does not shorten that chain, and a split over positions would
  change the bits. The slowest-over-mean ratio of step 2's gate counts the 134 idle CTAs, not a removable imbalance.
- **K3b, counters: not coded.** Step 1 found no counter 0.3 µs under the barrier (ctr8 0.274), so the counter form
  is not licensed.
- **K5, the certified head: not coded.** It adds about 0.6 MB to the card (+0.001 b/param) and takes 1 to 2 weeks.
  Its tokens stay the same, but it is not a package for one job.

## 2. What the code holds at the commit that carries this file

- **v9, `PERS_4E_ROWS` 7.** v8 (v7's staged constants and the attention ring), with every int4 row of the token
  loop read by `pers_q4_row_tile` (`llvq_pers.cuh`, after the scratch defines):
  - the v rows of q/k/v, the o rows, and the int4 downs of layers 12 to 23 go through `pers_q4_row_4e`;
  - `pers_q4_row_tile` is step 1's q4t. One 128-bit load gives lane l its words 128 t + l + 32 s, s = 0..3. Lane
    l still folds its words l, l + 32, ... in increasing order, eight products each, then the same `warp_sum`, so
    the bits are `pers_q4_row`'s. Words past the last full tile of 128 keep the served order.
  - At install, `persist4e_cuda::install_rows` relays each record 4E reads into a buffer of the same size with
    `pers_4e_q4_tile_relayout` (`pers_4e.cu`, layout `pers_q4_tile_index`). It refuses a record whose d_in is not a
    multiple of 32 or whose base is not 16-byte aligned. The served records stay as they are for the prefill and the
    eager path, so this measurement form holds the int4 bytes twice on the card, +0.41 GB. A served v9 would need
    the eager int4 readers on the same layout. That is not in this run.
- **v6s.** `install_rows_as(.., 4, Some(6))`: v6's module, launched with v8's dynamic shared bytes. Its label is
  "v6s".
- **Driver.** `pers::P4E_BENCH_FORMS` is `[v4, v6, v7, v8, v9]`, then v6s. Each rate is divided by every earlier
  form's, round by round. A v7, v8 or v9 divergence voids only that form; v4, v6 and v6s void the run. A module that
  does not install prints its registers and local bytes and the others go on.

Host checks at $0 (`llvq-cuda/tests/pers_matches_served.rs`, under nice 10):
- `the_4e_v9_int4_tiles_are_the_served_chain`: the token loop equal to the served chain at five geometries, both row
  variants, past position 256, two consecutive launches, and through the stamp symbol. The v rows (d_in 2,560: 2 tiles
  and a 64-word tail), the o rows (4,096: 4 tiles) and the int4 down (9,728: 9 tiles and a tail) all run the tiled
  reader.
- `the_4e_v9_mutants_are_killed`: 8 mutants caught. 5 on the reader and its layout (the index, the first and next
  tile, the column, the tail), 3 that put one site back on the served reader (v, o, down).
- The full llvq-cuda suite, `cargo clippy --all-targets`, the Linux half of llvq-cuda and of llvq-llm.

## 3. What runs

One image built from the commit that carries this file. One job, l40sx1, `ops/jobs/step3-4e.sh`. The sealed 4B
(`886391a8…`) is checked by size, then sha256.

1. **Oracle** on Qwen3-0.6B (rule 10).
2. `LLVQ_FUSED_LAYOUT=tetra48 LLVQ_EMBED=q4 LLVQ_ROT_SHARE=1 LLVQ_FUSE=0 LLVQ_PERS_4E=64 fusedrun <sealed> 64`.
   - Runtimes at 512 threads, v4, v6, v7, v8, v9, v6s, each its own module. v7 is expected not to load, as in step 2.
   - At 5 tokens (measurement form, kernel on every row) and at 8,192 (the served config, inside the T8).
   - Per context: each form's tokens against the eager path's, 2 + 5 interleaved rounds, then one stamped launch a
     form.

## 4. Gates

Each voids the run:

1. The image builds; `oracle` prints MATCH.
2. The sealed file at its bytes and sha256.
3. Event tracking off; a RoPE form equal to candle's.
4. **v4, v6 and v6s** tokens equal to the eager path's at both contexts.
5. No `LLVQ_*` variable in the job environment.

Each voids one form's numbers, not the run:

6. **v7, v8 or v9** tokens differ from the eager path's at a context, or the module does not install.

## 5. Outcomes reported, never gated

- Each form's registers, local bytes and shared bytes under the driver's ptxas.
- Per context: each form's tok/s, median and range; every pair's ratio round by round, among them v9 / v8, v9 / v6
  and v6s / v6.
- Each form's phase table and its 7 inner parts. A form whose stamp symbol spills is read on its timed rate only.

## 6. Signed predictions

*Estimated*. v9 takes K11's in-situ arithmetic at the midpoint of its high (0.697 ms at ×1.338: 0.176 ms) and low
(41 %: 0.072 ms) bounds. Nothing fitted to a number of this run.

| quantity | point | interval |
|---|---|---|
| v4, v6, v6s, v8, v9 tokens against the eager path, both contexts | **identical** | |
| v9 loads without local memory (driver ptxas, token symbol) | **yes** | |
| v6 at 5 tokens, tok/s | **211.8** | [208, 216] |
| v8 at 5 tokens, tok/s | **227.9** | [224, 231] |
| **v9 over v8, 5 tokens, round by round** | **x1.028** | [0.99, 1.045] |
| v9 at 5 tokens, tok/s | **234** | [225, 239] |
| v9 over v8, 8,192, round by round | **x1.019** | [0.99, 1.03] |
| v9 at 8,192, tok/s | **158.5** | [153, 161] |
| **v6s over v6, 5 tokens, round by round** | **x0.99** | [0.96, 1.01] |
| v6s over v6, 8,192, round by round | **x0.99** | [0.96, 1.01] |

## 7. Decision rules

- **A, identity.** §4. A v9 divergence is reported with its token and voids only v9.
- **B, K11.** v9 over v8 at 5 tokens, round by round:
  - whole range above x1.010: the lever holds in situ, and the eager int4 readers on the same layout are licensed
    as the next step toward serving it;
  - whole range under x1.010: it does not hold in situ, and the step-1 ratio was a DRAM-bound one 4E does not see.
- **C, the carve-out.** v6s over v6 at 5 tokens, round by round:
  - median at or under x0.98: the carve-out costs v6 at least 2 %, so it costs v8 and v9 too. A form under 64 KB
    of shared memory becomes a lead;
  - median at or above x0.995: the carve-out is not the cause, and v8's slower rows on its stamps are the stamp
    symbol's spill;
  - in between: not separated.
- **D, serving.** For v9: whole range above x1.00 over v8 at both contexts. Even then v9 is not proposed as is: its
  measurement form holds the int4 bytes twice. Serving v8 or v9, and any door job, is the operator's call.
- **E, the route.** No step 4 is chained by this run. The route's ceiling at the bit is re-read from its result.
- **G.** A miss is reported as a miss, with the term of §6 it refutes. No threshold moves after the clock.

## 8. Cost

*Estimated* on l40sx1 at $1.80/h. Step 2 billed 7 min for three timed forms:

| step | time |
|---|---|
| container and image | 2 to 3 min |
| oracle | 1 min |
| fusedrun: six modules built (v7's refused), 84 records relaid, 2 contexts x 7 rounds x 6 decodes, 10 stamped launches | 8 min |

About 12 min, **$0.36**. Timeout 25 min: **$0.75** at worst.

## 9. What this run does not establish

- v9 at the served door, or v9 with a single copy of the int4 bytes.
- The cause of v7's refusal beyond what its registers print.
- The 8B and the 14B; any card but the L40S.
