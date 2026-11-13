# Preregistration. Decode-speed research phase, R3: the first whole bodies on a flat, wide read, one card run of `rowbench`

Written on 2026-11-13 on branch `recherche/debit-2026-11-12`, once the code it describes existed. Timestamped before
the first measurement, then never edited; a departure goes to
`preregistration-recherche-debit-corps-2026-11-13-ECARTS.md`.

The launch is the operator's. Cost: about **$0.12** on l40sx1 (*estimated*, §9), after a Space rebuild that bills no
card. Cap: the job's timeout of 25 min, **$0.75** at worst (*computed* at $1.80/h). Spent before it: $0.20 of the
phase's $5 and $19.42 of the $25 cap (`docs/mesures/recherche-debit-r3-espace-2026-11-12.txt`, last line); see
`docs/data/jobs.csv` at launch.

## 1. The question

The second card run (`docs/mesures/recherche-debit-lecture-2026-11-12.txt`, job 6af5e9e7, L40S) measured the sealed
4B's Tetra words at 480 GB/s in the served read pattern with nothing decoded, and at 717 GB/s read as a flat stream of
aligned 128-bit loads with no side stream (the card's bare read: 751). The served row, decoding, reads 396. No decoder
has ever run on a flat, wide read.

This run asks, for a first population of real decoders built along the axes of the R3 note (the unit of work, the
load, the row's end, the sum, the layout): does any of them compute the served rows (their bits, or within a declared
bound of 1e-5) faster than the served row, on the sealed file's own records, by more than the job's own noise? It is
the search route of `docs/MODES.md`. Nothing here is expected to win; the second run gave a ceiling for the read, not a
rate for a decoder.

## 2. What the code holds at the commit that carries the bench

The commit is `11f9b31`. The bodies, the gate and the bench came at `ee621c9`. `11f9b31` writes the butterfly out so
that the 32 chains stay in registers, and gives the SASS count its two exclusions.

Nothing served moves: no `configs/` change, no served kernel text changes a byte (the served unit's pin
`pers.rs::the_served_unit_is_the_published_one`, 101,625 bytes, `6193094310bc…`, and 4E's pins stay green). Changed
since `70401d9`:

- `llvq-cuda/kernels/rowcand.cu`: the registry `ROWC_ARMS` takes a seventh field, SMEM, the shared bytes an arm owns
  beyond the row loop's layout. A registered struct is now a row program (`FORM ROWC_ROW`, as before) or a WHOLE BODY
  (`FORM ROWC_BODY`): a `slice()` that computes y for a site's rows of the CTA's slice with no row geometry imposed,
  called by `rowc_whole` after it stages the activation (once a site, once a gate/up pair). Seven bodies. `served_np`
  stays in the text and leaves the registry. The relay interface takes the row index and d_out.
- `llvq-cuda/src/rowcand.rs`: the seventh field, its checks.
- `llvq-cuda/src/bin/rowbench.rs`: each arm launched with its own shared bytes; the served arm timed again as each
  set's last run; the rule F1 to F3 and the `SCORE`, `PROBE` and `Q4READ` lines in place of rule D, `HYP` and `NET`;
  the timed list trimmed (§3).
- `llvq-cuda/tests/host_pers_rowcand.inc`: the arm's SMEM added to the host twin's shared bytes.
- `llvq-cuda/tests/rowcand_gate.rs`: the gate runs on the stride-6 site too; the body mutants.
- `ops/rowcand-sass.sh`: a store to shared memory no longer excludes a block loop (the flat bodies write a partial a
  run); a body whose words come from shared memory falls back to the shortest loop that reads shared memory.

### The candidates

A RUN is 8 consecutive blocks of a row: 48 bytes, 3 quads, at byte 48 g of the row. Every sealed row stride is whole
quads (160, 256 and 608 u32), so on the sealed file every run is read by three aligned 128-bit loads. A site whose
rows are not 16-byte aligned (only the gate's stride-6 site) is read by u32 loads, the same words; the bench prints
how many sealed records do (expected 0). The bodies that read runs stage the activation with one float of padding
every 8 blocks (block b at 24 b + b / 8), so that lanes on consecutive runs read 32 distinct banks.

The 1e-5 bound is on |f32 - served f32| over the rms of the site's served f32 rows, every row. Mac errors are over
5,568 rows (§2, stage 1), max and 0.99 quantile.

| arm | what it does | owes | Mac max, q99 | SMEM (B) | SASS a block |
|---|---|---|---|---|---|
| `acc2` (kept) | the served row with two accumulators a lane | bound 1e-5 | 3.753e-7, 2.024e-7 | 0 | 224 |
| `tr32` | a THREAD A ROW: the thread reads its row run after run by 3 aligned 128-bit loads, decodes each block (`pers_block`) into the chain of the served lane that folds it (block j on chain j mod 32), keeps the 32 chains in registers, sums them in the served butterfly's order, does its own tail, scale and store | the bits | 0, 0 | 256 | 251.9 |
| `tr32p` | `tr32` with the next run's 3 quads in flight while the current run decodes | the bits | 0, 0 | 256 | 253.5 |
| `endpass` | the served row (a warp a row, `pers_tetra_row`) with its row end moved out of the row loop: lane 0 writes the row's sum to shared memory; after a barrier, a thread a row does the tail, scale and store. Chunks of 2,048 rows | the bits | 0, 0 | 8,448 | 227 |
| `wq` | the served unit of work, a warp a row, on wide loads: lane l takes runs l, l + 32, ... of the row, 3 aligned 128-bit loads each, the next run in flight; one accumulator a lane, `warp_sum`, lane 0 does the row end | bound 1e-5 | 4.205e-7, 2.024e-7 | 256 | 258.8 |
| `flat` | the CTA's slice as one stream of runs across row boundaries: unit u is run u mod G of row c0 + u / G, a thread a unit, consecutive threads on consecutive runs; each unit's sum to its own shared slot; a barrier; a thread a row sums its G slots in run order and does the row end. Chunks of 2,048 / G rows | bound 1e-5 | 5.915e-7, 2.438e-7 | 8,448 | 258.5 |
| `tr32r` | `tr32p` on a RELAY layout rewritten at load at the same stride and bytes: rows in tiles of 32, quad k of row rr of a tile at (k R' + rr), R' the tile's rows, so a warp's 32 lanes read 512 contiguous bytes a load | the bits | 0, 0 | 256 | 253.6 |
| `flats` | `flat` with the chunk's words staged in shared memory by the flat read of the second run (thread t loads quads t, t + T, ...: consecutive 128-bit loads, no row boundary), the next chunk's quads held in registers (4 a thread) while the current one decodes from shared memory; chunks of min(1,024 / G, 2,048 / quads a row) rows | bound 1e-5 | 5.915e-7, 2.438e-7 | 37,120 | 239.1 |

Why `tr32`, `tr32p`, `tr32r` and `endpass` owe the bits. `endpass` runs the served row's own function and the served
epilogue on its value. The thread-a-row bodies keep the served association exactly: chain c folds blocks c, c + 32,
... in order, each with `pers_block`, from 0.0f; `warp_sum`'s xor butterfly on lane 0 is a[l] += a[l + k] for l < k,
k = 16 to 1, on one thread. After G runs the chains sit rotated by 8 G mod 32; they are not rotated back, because a
rotation by 8 maps the pairs {l, l + 16} and {l + 8, l + 24} of the first two steps onto each other and the addition
that joins them commutes (the gate checked the rotation back as an equivalent mutant, below).

Why `flat` and `flats` are race-free. Every slot of the chunk is written by the one thread that folds its unit
(u = threadIdx.x + k blockDim.x), read only after a barrier by the one thread of its row, and written again only after
the barrier that ends the gather. No atomic, no shuffle; the row's sum has a fixed order (run 0 to G - 1), so the
result does not depend on the schedule. `flat` and `flats` fold and gather the same way and give the same f32 rows.

What each allocates on the card. No candidate allocates VRAM beyond the bench's own result buffers (y in binary16, and
its f32 for the check), which every arm has. `tr32r` reads a copy of the words in its relay layout, the same size as
the words (d_out x stride u32); the bench holds both, a served form would replace the words. Shared memory a CTA:
the row loop's layout at d_in 9,728 (57,984 B) plus the arm's SMEM, at most 95,104 B (`flats`), under sm_89's 101,376.

The gate/up form. Every body has one: the pair's activation is staged once, then the CTA's slice of the gate site,
then the same slice of the up site. It is not 4E's interleave (gate row then up row on each warp); `sealed:gu4e`
describes it.

What the bodies cannot do well by construction (computed, not a prediction). A thread a row puts d_out / 142 rows on
a CTA: 7 to 69 threads of 512 on the sealed file, so `tr32`, `tr32p` and `tr32r` run on 1 to 3 warps a SM. `wq` puts
G = 14 runs of a 2,560-wide row on 14 of 32 lanes.

### Stage 1, run on the Mac at $0 (*measured* on the host twin, `cargo test -p llvq-cuda --test rowcand_gate`)

`every_registered_arm_keeps_what_it_owes`: every arm on both bodies (rows; gate/up pairs) and three grids (3 x 96,
2 x 64, 5 x 32), on the recorded sites of the first gate and the stride-6 site added in the second (its odd rows start
8 B off a 16-byte boundary): 5,568 row results over the six runs (three grids, both bodies). The served arm, `tr32`, `tr32p`, `endpass` and `tr32r`:
5,568 of 5,568 rows equal bit for bit (y and f32). The bound arms: the table's max and q99, every one under 1e-5.

`the_gate_kills_the_body_mutants`: eleven edits, each applied at exactly one site of a copy of `rowcand.cu`
(`mutant` asserts the count and a changed text), each refused on every arm that runs the edited code:

| mutant | the mechanism | arms, rows moved of 5,568 |
|---|---|---|
| `run_last_block` | the run fold drops the last block of every run | `wq`, `flat`, `flats`: 1,755 each |
| `tr_last_block` | the thread-a-row chains drop the last block of every run | `tr32`, `tr32p`, `tr32r`: 1,032 each |
| `quad_last` | the 128-bit reader drops the last quad of every row | `tr32` 5,280; `wq`, `flat`, `flats` 5,331 |
| `u32_tail` | the u32 reader (stride-6 rows) drops the word that holds the last block's high half | `tr32`, `tr32p` 288; `wq` 1,584; `flat`, `flats` 1,689 |
| `wrong_chain` | blocks 1 and 2 of every run folded on each other's chain | `tr32`, `tr32p`, `tr32r`: 792 each (max 2.4e-7: bits, not the bound) |
| `relay_last_unit` | the relay copy drops the last unit of every row | `tr32r`: 5,280 |
| `endpass_last_row` | the second pass skips the last row of every chunk | `endpass`: 190 |
| `skip_tail` | the row end drops the binary16 tail | `tr32`, `endpass`, `wq`, `flat`: 5,568 each |
| `wq_last_run` | the lanes drop the last run of every row | `wq`: 5,568 |
| `gather_boundary` | the gather takes each run's slot from the next one: a row's last run from the next row | `flat`, `flats`: 5,568 each |
| `stage_last_quad` | the stage drops the last quad of every chunk | `flats`: 1,576 |

Two edits were tried first and are EQUIVALENT, not survivors: the chains never rotated back (5,568 of 5,568 rows
equal), and a block folded on chain i + 1 mod 8 (the same). Every translation or xor of the chain index is a symmetry
of the butterfly; the rotation back was removed from the body, and `wrong_chain` is a swap, which is not.

Gates of the commit, on the Mac (*measured*, 2026-11-13): `cargo test -p llvq-cuda --no-fail-fast`, 34 suites, 291
passed, 0 failed (290 at the second run, plus `the_gate_kills_the_body_mutants`); `cargo clippy -p llvq-cuda
--all-targets` clean on the Mac and on `x86_64-unknown-linux-gnu`; the bench module (359,775 bytes, sha256
`587ddd48…`) compiles under devel nvcc 12.4 for sm_89, every `tv_rowc_*` kernel at 0 B of stack and 0 B of spill:
registers `served` 72, `acc2` 71, `endpass` 72, `wq` 88, `flat` 78, `flats` 91, `tr32` 102, `tr32p` 116, `tr32r` 117
(the `gu_` forms 4 to 20 more). The six kernels of the module that spill are the row-loop unit's `pers_4e_*`,
unchanged and not launched by `rowbench`.

SASS a block (`ops/rowcand-sass.sh`, devel count, to compare between arms only): the table's last column; `served`
226 and `pword` 24 as at the first run. The thread-a-row, lane-run and flat bodies count one run of 8 blocks a loop
iteration (BPI 8); their loops are about 2,015 to 2,070 instructions, 252 to 259 a block, 26 to 33 a block over the
served row: run addressing, the predicate on the last run and, for `tr32`, the 32 chains' rotation. `flats`' count is
its staged loop (the words from shared memory, three `LDS.128` a run), 239 a block. `endpass`'s block loop is the
served one, 227.

## 3. What runs

One image built from the commit that carries the bench. One job, l40sx1, `ops/jobs/recherche-debit-corps.sh`,
output `/out/recherche-debit-corps-2026-11-13`:

1. **Oracle** on Qwen3-0.6B (hard rule 10).
2. The sealed 4B (`sealed-4b-2026-09-23/qwen3-4b-sealed.bin`, 1,418,224,685 B, sha256 `886391a8…`) and the bare encode
   (`tetra-4b-2026-09-06/qwen3-4b-tetra.bin`, 1,770,529,149 B, sha256 `0adb7cfd…`) checked by size and sha256.
3. `rowbench <sealed> <bare> --rounds 15 --dump <out>/cubin`: no `LLVQ_*` variable; the tile is the card's row (64 on
   sm_89). 2 warmup rounds, 15 timed; every valid run once a round, in a fixed order; times on the host around launch
   and sync.

The sets: `sealed:d_in=2560`, `sealed:d_in=9728`, `sealed:all`, `sealed:gu4e`, `bare:all` (the bare encode's d_in sets
are dropped). The order inside a round, a set at a time: `served` FIRST, then `acc2`, `pword`, `tr32`, `tr32p`,
`endpass`, `wq`, `flat`, `tr32r`, `flats`, then `pw_all`, then on `bare:all` the journal controls `v2`, `v3`, `v2u`,
`nc`, `tile`, and `served_again` LAST (the `gu_` forms on `sealed:gu4e`). Then `q4`, `q4t` and `q4read` on
`sealed:int4:all`, then the 16 probe configurations. Dropped from the timed list, their code kept: `served_np`, the
other read diagnostics, the nine other journal controls, `q4hh` and `q4nc` (`tv_q4_hh` still gives the int4 reference
rows), the int4 d_in sets.

## 4. The score

A candidate's score on a set is its speed over `served`, round by round (t_served / t_arm), median [range], in the
same job. `sealed:all` decides; `sealed:d_in=2560`, `sealed:d_in=9728`, `sealed:gu4e` and `bare:all` describe. The
score is the rows' rate, words read and y written for the set's records; it is not tok/s.

## 5. The rule F1 to F3 (the R3 note)

- **F1.** On each set, `served_again` over `served`, round by round (a speed), must have a median inside
  [0.98, 1.02]. Its round-by-round range [lo, hi] is that set's noise floor. F1 on `sealed:all` decides the rule; on
  the other sets it describes, and a set whose own F1 fails prints its SCORE lines as a description.
- **F2.** A candidate counts as FASTER on a set only if its whole round-by-round range over `served` lies above hi;
  SLOWER if its whole range lies under lo; NOT SEPARATED otherwise. A refused candidate gets no rate.
- **F3.** On `bare:all`, `v3`, `v2u` and `nc` each read under 0.92 of `v2` (median), and `tile` inside
  [0.96, 1.06] of `v2`.
- The rule holds when F1 on `sealed:all` and the four F3 lines pass (`RULE PASS`). If it fails, every SCORE line is
  printed and read as a description, and says so.
- The probe's best configuration and `q4read` on `sealed:int4:all` are printed beside the first two runs' values
  (751.1 / 750.9 and 675.3 / 676.7). They gate nothing.

The rule is written after two failures (the second run's journal, RULE D) and is tested by this run.

## 6. Gates

Each voids the run:

1. The image builds; `oracle` prints MATCH.
2. Both files at their bytes and sha256. The references print the served unit, 101,625 bytes, sha256
   `6193094310bc…`; the bench module prints the digest the Mac builds (`rowcand.rs::tests::dump_rowcand_module`),
   359,775 bytes, `587ddd4857e92ea821d624f11dce4c67407999ddc80467841aa5eb10ac1049b6`.
3. The served arm prints every row bit for bit to `tv_tetra48_h` on sealed:all, sealed:gu4e and bare:all.
4. No `LLVQ_*` variable in the job environment.

Each voids only its own line: a candidate refused on a set (it misses what it owes, spills, or does not fit); a probe
configuration whose sum misses; `pw_all` if its read check misses.

## 7. What the run reports

- `ROUND` lines, medians with round-by-round ranges, speeds over bases, `TSV` (as before).
- Each candidate's check before timing, on each set: registers, local bytes, shared bytes, CTAs a SM, then y and f32
  equal counts, max and q99 relative error, PASS or REFUSED.
- `F1` a set, `F3` four lines, `RULE`.
- `SCORE` a candidate and a set: speed over served, median [range], what it owes, its max and q99 error, valid or
  REFUSED, faster / not separated / slower against the set's floor, decides or describes.
- `PROBE` and `Q4READ` beside the first two runs.
- The driver's cubin of the module (`cubin/rowbench.cubin`) and its text, for the card's own SASS count.

Labels: every rate *measured*; bytes *computed*; the SASS counts of §2 are the Mac's devel count, *measured*
offline, to compare between arms of one source only.

## 8. What would surprise

- A thread-a-row body (`tr32`, `tr32p`, `tr32r`) not slower than `served` on `sealed:all`: one to three warps a SM.
- `endpass` separated from `served` either way on `sealed:d_in=9,728`, where the second run found the side streams
  cost nothing.
- F1 failing on `sealed:all`: two timings of one kernel in one round would differ by more than 2 %.

## 9. Cost

*Estimated* from the last two runs of this bench (jobs 6af5ce28 and 6af5e9e7, `docs/data/jobs.csv`): 203 s running
each, $0.10, the oracle included. This run times 11 row arms on 5 sets (about 3.0 GB of words a round each at 0.1 to
0.7 TB/s, *computed*: the thread-a-row bodies are taken at 0.1), 5 controls and `pw_all`, 3 int4 arms and 16 probe
configurations: about 0.1 s a round, under 2 s over 17 rounds. About 4 min running, **$0.12**. Cap: 25 min, **$0.75**.

## 10. Signed predictions (main session)

Signed on 2026-11-13, before the stamp, from the two journals and the candidates' text; the hand-back's Mac gate was
rerun by the main session first (5,568 rows, the same figures, module `587ddd48…`). Scores are speeds over `served`,
round by round, median; rates are GB/s of words. Each line is scored in the journal as in or out.

| # | quantity | point | interval | why |
|---|---|---|---|---|
| 1 | F1 on sealed:all passes; the floor's width (hi - lo) | passes, 0.02 | 85 %; width under 0.04 | one kernel twice in a round; the first two runs' ranges are 3 % wide |
| 2 | F3, the four lines pass | passes | 85 % | v3, v2u, nc read 0.83 to 0.85 and tile 1.03 in the first run |
| 3 | `served` on sealed:all | 396 | [388, 404] | 395.6 in the second run |
| 4 | `tr32` on sealed:all | 0.20 | [0.10, 0.35] | 7 to 69 threads a CTA where the served row runs 512 |
| 5 | `tr32p` and `tr32r` over `tr32` | 1.05 each | [0.97, 1.20] | words in flight, contiguous loads; the body is bound by its few threads |
| 6 | `endpass` on sealed:all | 1.00 | [0.96, 1.05], not separated | the row end is moved, not removed |
| 7 | `wq` on sealed:all; on sealed:d_in=9728 | 0.62; 1.02 | [0.45, 0.80]; [0.90, 1.12] | 14 of 32 lanes work on a 2,560-wide row, all of them at 9,728 |
| 8 | `flat` on sealed:all | 1.10 | [0.95, 1.25]; FASTER under F2: 55 % | the read of pw_all (1.40 ms a GB against 2.08), the decode's 0.44 kept, the row ends kept, 14 % more instructions a block, a barrier a chunk |
| 9 | `flats` over `flat`, sealed:all | 0.92 | [0.75, 1.05] | one more copy through shared memory |
| 10 | `acc2` on sealed:all | 1.02 | [1.00, 1.04], not separated | 1.023 in the first run |
| 11 | no candidate is refused on the card on any set | holds | 70 % | `flats` at 95,104 B of shared memory and `tr32r`'s relay have never run |
| 12 | probe best; q4read | 751; 676 | +-6; +-8 | two runs agree to 0.2 and 1.4 |

The best candidate on sealed:all will be `flat`, and no bit-owing candidate will be FASTER. What would surprise beyond
§8: `flat` under 0.90 (the flat read would not survive a real decoder); `flat` above 1.30; `wq` above `flat` on
sealed:all.

## 11. Not established by this run

- Any tok/s. The score is the rows' rate alone, at 4E's geometry with no other traffic on the card; a candidate that
  wins is an idea only after a token-level measurement (R6).
- Why a candidate is fast or slow: the run times bodies, it does not attribute their time.
- Any card but the L40S. Hardware counters (none on HF Jobs). The SASS counts, which are offline.
