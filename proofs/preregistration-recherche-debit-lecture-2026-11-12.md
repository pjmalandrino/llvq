# Preregistration. Decode-speed research phase, R2 again: why the Tetra words read 493 GB/s, one card run of `rowbench`

Written on 2026-11-12 on branch `recherche/debit-2026-11-12`, once the code it describes existed. Timestamped before
the first measurement, then never edited; a departure goes to
`preregistration-recherche-debit-lecture-2026-11-12-ECARTS.md`.

The launch is the operator's. Cost: about **$0.15** on l40sx1 (*estimated*, §9), after a Space rebuild that bills no
card. Cap: the job's timeout of 25 min, **$0.75** at worst (*computed* at $1.80/h). Spent before it: $0.10 of the
phase's $5 and $19.32 of the $25 cap (`docs/mesures/recherche-debit-r1-recadrage-2026-11-12.txt` line 64); see
`docs/data/jobs.csv` at launch.

## 1. The question

The first card run (`docs/mesures/recherche-debit-banc-2026-11-12.txt`, job 6af5ce28) measured, on the L40S:

| what | GB/s, *measured* |
|---|---|
| the card, bare read probe, best of 16 configurations | 751.1 [748.0-752.5] |
| the sealed file's Tetra words, read and not decoded (`pword`) | 492.8 [487.3-500.7] |
| the sealed file's int4 bytes, read and not decoded (`q4read`) | 675.3 [670.4-684.0] |

Nothing in the repository explains why a loop that only loads the Tetra words streams 0.66 of the card. This run asks
how much of the 493 to 751 gap each hypothesis of the reframing note
(`docs/mesures/recherche-debit-r1-recadrage-2026-11-12.txt` lines 34 to 51) holds, and what is left when all of them
are removed at once. It is the explanation route of `docs/MODES.md`: diagnostic arms, no candidate, no gain projected.

## 2. What the code holds at the commit that carries the bench

The commit is `2cfb308`.

Nothing served moves: no `configs/` change, no served kernel text changes a byte (the served unit's pin
`pers.rs::the_served_unit_is_the_published_one`, 101,625 bytes, `6193094310bc…`, and 4E's pins stay green). Changed
since the first run (`2794bbc`):

- `llvq-cuda/kernels/rowcand.cu`: the read diagnostics, one line each in a second registry `ROWD_ARMS`, two kernels
  each, `tv_rowd_<arm>` (the rows body) and `tv_rowd_gu_<arm>` (4E's gate/up body). `ROWC_ARMS` and its four arms are
  unchanged, byte for byte.
- `llvq-cuda/src/rowcand.rs`: the registry's parser (`diag_arms`) and the check's sum (`words_sum`).
- `llvq-cuda/src/bin/rowbench.rs`: the diagnostics on every set, the check after every launch, the new rule D2, the
  `HYP` and `NET` lines.
- `llvq-cuda/tests/rowcand_gate.rs`, `tests/host_pers_rowcand.inc` (command `d`), `tests/host_pers.cpp` (one line):
  the Mac gate of the diagnostics.
- `ops/rowcand-sass.sh`: also counts the diagnostics' read loops.

### What `pword` is today (read in `rowcand.cu` and `llvq_pers.cuh` at `2794bbc`)

- The read: each row's 48-bit blocks by `f1r_load`, two u32 loads a block (one of them unaligned to the block), lane
  j takes blocks j, j + 32, ..., with the next block's two words in flight in registers. It folds `lo ^ (hi16 << 7)`
  by xor.
- **No L2 prefetch.** No `prefetch.global.L2` is in `pword`, nor in `served`. The `PREFETCH` template flag of
  `pers_tetra_row` is the next block in flight in registers, and `served_np` is the served row without it. The first
  journal's sentence "The L2 prefetch is worth 11 % to the rows (served_np x0.892)" names the register pipelining.
- The row: a warp a row, rows `lo + warp, lo + warp + 16, ...` of the CTA's slice `[d c / g, d (c + 1) / g)`; then
  `warp_sum` of `(float)(h & 0xff)`, five shuffles.
- The side streams, all kept: lane 0 runs the served epilogue, the binary16 tail (8 or 16 values against the staged
  activation), the f32 row scale, the binary16 y store (the f32 value only in the check launch, not when timed).
- The site: the activation is staged in shared memory at every site, `pers_stage_f32` with a barrier on each side. The
  decoder tables (18,816 B) are staged **once a launch**, not a site.

### The read diagnostics

Each arm changes one thing of its base. `pw_chk` is the base of H1 to H4: `pword` with its xor fold replaced by the
read check. `pword` itself is timed unchanged beside it, so `pw_chk` over `pword` reads the check's own cost.

| arm | hypothesis | base | the one change |
|---|---|---|---|
| `pw_chk` | | `pword` | the fold: the read check's u64 sum instead of the xor |
| `pw_noside` | H1 the side streams | `pw_chk` | no binary16 tail, no row scale read, no y store; the row's `warp_sum` kept |
| `pw_flat` | H2 the row | `pw_chk` | the CTA's slice of rows read as one flat stream of 48-bit units by its 512 threads (unit u on thread u mod 512, the next unit in flight), no row boundary, no per-row reduction; each row's epilogue kept, in its own loop after the read |
| `pw_u4` | H3 the load | `pw_chk` | each row's words by aligned 128-bit loads, quad q on lane q mod 32, the next quad in flight |
| `pw_nostage` | H4 the site | `pw_chk` | the activation staged once a launch, at the widest site, not at every site: no per-site barrier |
| `pw_grid` | H5 the schedule | `pw_flat` | the probe's schedule: every thread of the grid strides the whole site, unit u on thread u mod (142 x 512), no split of the site over the CTAs |
| `pw_all` | the stack | `pw_chk` | H1 to H5 together: the grid's stride, 128-bit loads, no per-site staging, no side stream |
| `pw_empty` | H0 the launch | | the same launch (grid, 512 threads, dynamic shared) over the same site list, reading only the sites' descriptors |

What had to be adapted from the note's wording:

- **H2** keeps the side streams in their own loop, so that it does not also remove what H1 removes.
- **H3** keeps the next load in flight, as `pword` does; a lane then holds 32 B in flight against 16. It reads the row
  padding too: 4 B of 640 at d_in 2,560, 2 of 2,432 at 9,728, 4 of 1,024 at 4,096 (*computed*).
- **H4**: the note says the site stages its activation and its tables; the tables are staged once a launch already,
  so the arm removes the per-site activation staging and its two barriers.
- **H5**: the note says every thread takes a contiguous share of the site. The probe's code does not: it is a grid
  stride, thread t reads elements t, t + S, t + 2 S, ... with S the grid's threads (`bwprobe.cu`, `bw_read_body`). The
  arm takes the probe's real schedule. A grid-wide stride has no rows, so the arm is built on `pw_flat`, and its kill
  line reads over `pw_flat`, where it changes one thing.
- **H0**: the empty launch does not stage the tables either.

**The read check.** A diagnostic owes no rows; it owes its check. Every arm but `pw_empty` adds every u32 it reads into
a u64 a thread (a 48-bit unit adds its bytes, each half-word at its weight in its u32), keeps it over the whole
launch, and writes it with one more u64 at the end. The host adds the slots and compares with its own sum of the
set's words, row padding included (the transcode writes it as zeros). `pw_empty` adds the d_out of every site it
visits; the host expects 142 x 512 times their sum. The bench checks before timing and **after every timed launch**;
a miss refuses the arm on that set. The slots cost every arm, `pw_empty` included, a write of 2 x 142 x 512 x 8 B =
1.16 MB a launch, 0.8 % of the smallest set's words (0.1494 GB, *computed*).

### Stage 1, run on the Mac at $0 (*measured* on the host twin, `cargo test -p llvq-cuda --test rowcand_gate`)

`every_read_diagnostic_reads_every_word`: each arm on both bodies (rows; gate/up pairs) and three grids (3 x 96,
2 x 64, 5 x 32), on the recorded sites of the first gate plus a site whose row stride (6 u32) is not a multiple of 4,
so that odd rows start 8 B off a 16-byte boundary: 90,240 words a pass of both bodies. All 8 arms: 6 checks of 6
equal to the host's sum.

`the_read_check_kills_a_skipped_word`: four edits, each applied at exactly one site of a copy of `rowcand.cu`
(`mutant` asserts the count and a changed text), each refused on every run of every arm that runs the edited reader:

| mutant | what it skips | arms | runs that miss |
|---|---|---|---|
| `row48_last_block` | the last 48-bit block of every row | `pw_chk`, `pw_noside`, `pw_nostage` | 6 of 6 each |
| `u4_last_quad` | the last 128-bit load of every stream | `pw_u4`, `pw_all` | 6 of 6 each |
| `u4_rest` | the words after the last whole quad | `pw_u4` | 6 of 6 |
| `flat48_rest` | the words after the last whole pair of units | `pw_flat`, `pw_grid` | 6 of 6 each |

Gates of the commit, on the Mac (*measured*, 2026-11-12): `cargo test -p llvq-cuda --no-fail-fast`, 34 suites,
290 passed, 0 failed (287 at the first run, plus the diagnostic registry's test and the two gate tests above);
`cargo clippy -p llvq-cuda --all-targets` clean on the Mac and on `x86_64-unknown-linux-gnu`; the bench module
(335,427 bytes, sha256 `6ca9543e…`) compiles under devel nvcc 12.4 for sm_89, every `tv_rowd_*` kernel at 0 B of
spill (12 to 78 registers). Read loops (`ops/rowcand-sass.sh`, devel count, to compare between arms only): `pword` 24
instructions a block, `pw_chk` 29, `pw_noside` and `pw_nostage` 29, `pw_flat` 37, `pw_grid` 35 a block; `pw_u4` 49
for four 128-bit loads (unrolled), `pw_all` 24 a 128-bit load. The check costs `pw_chk` 5 instructions a block over
`pword`.

## 3. What runs

One image built from the commit that carries the bench. One job, l40sx1, `ops/jobs/recherche-debit-lecture.sh`,
output `/out/recherche-debit-lecture-2026-11-12`:

1. **Oracle** on Qwen3-0.6B (hard rule 10).
2. The sealed 4B (`sealed-4b-2026-09-23/qwen3-4b-sealed.bin`, 1,418,224,685 B, sha256 `886391a8…`) and the bare encode
   (`tetra-4b-2026-09-06/qwen3-4b-tetra.bin`, 1,770,529,149 B, sha256 `0adb7cfd…`) checked by size and sha256.
3. `rowbench <sealed> <bare> --rounds 15 --dump <out>/cubin`: no `LLVQ_*` variable; the tile is the card's row (64 on
   sm_89). 2 warmup rounds, 15 timed; every valid run once a round, in a fixed order; times on the host around launch
   and sync.

Everything the first run timed is timed again in the same process: `served`, `served_np`, `acc2`, `pword` and their
`gu_` forms, the 14 journal controls, `q4hh` and the four int4 arms, the 16 probe configurations. The sets are the
first run's (prereg of the first run §4): `sealed:d_in=2560`, `sealed:d_in=9728`, `sealed:all`, `sealed:gu4e`,
`bare:d_in=2560`, `bare:d_in=4096`, `bare:d_in=9728`, `bare:all`. The eight diagnostics run on every one of them, the
`gu_` forms on `sealed:gu4e`.

## 4. The hypotheses and their kill lines

The decision set is **`sealed:all`** (the 168 Tetra records of the sealed file, 0.7157 GB of words). The other sets are
reported and decide nothing. A speed is the base's time over the arm's, round by round, its median over the 15
rounds (the `HYP` line, also over `pword`).

| hypothesis | arm | over | dead if |
|---|---|---|---|
| H1 the side streams | `pw_noside` | `pw_chk` | median under **x1.03** |
| H2 the row | `pw_flat` | `pw_chk` | median under **x1.03** |
| H3 the load | `pw_u4` | `pw_chk` | median under **x1.03** |
| H4 the site | `pw_nostage` | `pw_chk` | median under **x1.03** |
| H5 the schedule | `pw_grid` | `pw_flat` | median under **x1.03** |
| H0 the launch | `pw_empty` | | median time a launch under **20 us** |

The check's own cost, `pw_chk` over `pword`, is reported. If it lies outside [0.97, 1.03], the diagnostics are read
over `pw_chk` as written, and the journal says by how much the base moved.

The hypotheses are not exclusive: an arm that lives does not exclude another. What is left is `pw_all` against the
probe's 751 (this run's own probe best), reported in GB/s of words, raw and net of `pw_empty` (`NET`, the empty launch's
time subtracted round by round).

## 5. Rule D, the fidelity of the stand-in

- **D1**, unchanged. The served arm on bare:all, in the journals' accounting (0.9467 GB), reads a median inside
  **[368, 383] GB/s**.
- **D2**, new. Each known arm's median speed over its base, round by round, on bare:all, lies inside its journal's own
  round-by-round range **widened by 0.02 on each side**:

| arm | over | journal median [range] | source | window |
|---|---|---|---|---|
| v3 | v2 | 0.849 [0.839-0.849] | `pers-rowloop-2026-11-05.txt` line 26 | [0.819, 0.869] |
| ilp2 | v2 | 0.928 [0.905-0.930] | line 27 | [0.885, 0.950] |
| v2u | v2 | 0.825 [0.807-0.826] | line 28 | [0.787, 0.846] |
| l1w | v2u | 1.214 [1.210-1.221] | line 29 | [1.190, 1.241] |
| bcast | v2u | 0.987 [0.969-0.993] | line 30 | [0.949, 1.013] |
| xbcast | v2u | 0.995 [0.965-1.001] | line 31 | [0.945, 1.021] |
| v2_o2 | v2_s | 1.076 [1.073-1.080] | line 33 | [1.053, 1.100] |
| nc | v2 | 0.810 [0.787-0.829] | `step1-micro-2026-11-07.txt` line 27 | [0.767, 0.849] |
| nc256 | v2 | 0.845 [0.823-0.865] | line 28 | [0.803, 0.885] |
| cpa | v2 | 0.960 [0.944-0.987] | line 29 | [0.924, 1.007] |
| tile | v2 | 1.002 [0.982-1.024] | line 30 | [0.962, 1.044] |
| bal | v2 | 0.934 [0.916-0.956] | line 31 | [0.896, 0.976] |

Every control has a journal range; none falls back on the flat ±0.03. This rule was written after the first run
(reframing note lines 57 to 62). On that run's data its twelve ratios all lie inside these windows (*computed* from
`recherche-debit-banc-2026-11-12.txt` lines 23 to 25); the first run stays a description, and the rule is tested on
this run's data only.

If D1 or D2 fails, the `HYP` lines are reported as a description and no hypothesis is killed by this run.

## 6. Gates

Each voids the run:

1. The image builds; `oracle` prints MATCH.
2. Both files at their bytes and sha256. The references print the served unit, 101,625 bytes, sha256
   `6193094310bc…`; the bench module prints the digest the Mac builds (`rowcand.rs::tests::dump_rowcand_module`),
   335,427 bytes, `6ca9543e601df5a76dc30dfb12c7448e44105e2ea6ef7789109e304b33f257ec`.
3. The served arm prints every row bit for bit to `tv_tetra48_h` on sealed:all, sealed:gu4e and bare:all.
4. No `LLVQ_*` variable in the job environment.

Each voids only its own line: an arm or a probe configuration that is refused; a diagnostic whose check misses, before
timing or after any timed launch, on that set. A hypothesis whose arm is refused on `sealed:all` is neither dead nor
alive.

## 7. What the run reports

- Everything the first run reported (`ROUND`, medians with round-by-round ranges, speeds over bases, `FIDELITY`,
  `ORDER` with the new windows, `TSV`).
- The diagnostics' lines before timing: registers, local bytes, and "reads every word once" with the check's value.
- `HYP` lines, one an arm and a set: the speed over its base and over `pword`, median [range], and the kill verdict
  of §4; for `pw_empty`, its time a launch in us.
- `NET` lines: every row arm's and diagnostic's GB/s of words over (t - t_empty), round by round, median [range].
- The driver's cubin of the module (`cubin/rowbench.cubin`) and its text, for the offline SASS count
  (`ops/rowcand-sass.sh` now counts the diagnostics' read loops too).

Labels: every rate *measured*; bytes *computed*.

## 8. What would surprise

- `pw_chk` over `pword` outside [0.97, 1.03]: the check would not be free.
- `pw_all` well under 700 GB/s: the Tetra buffers themselves, read with the probe's own schedule and load, would not
  reach the card's rate, and the cause would lie outside every hypothesis here (the buffers' sizes, the site loop).
- No arm at x1.03 or more: the gap would have no mechanism among the five, and the route goes back to R2.

## 9. Cost

*Estimated* from the first run of this bench, job 6af5ce28 (`docs/data/jobs.csv` line 333): 203 s running, $0.10, the
oracle included. This run adds 16 kernels to the module (NVRTC compiles it twice, once for the dump) and 8
diagnostics on 8 sets: about 3.7 GB of words a round each, so about 30 GB and 60 ms a round at 0.5 TB/s, about 1 s over
the 17 rounds (*computed*), and a 1.2 MB read-back after each diagnostic launch. About 5 min running, **$0.15**. Cap:
25 min, **$0.75**.

## 10. Signed predictions (main session)

## 11. Not established by this run

- Any tok/s, any candidate's score: the diagnostics owe no rows.
- A mechanism inside an arm that lives: an arm says how much one change recovers, not why.
- Any card but the L40S. Hardware counters (none on HF Jobs). The SASS count, which is offline.
