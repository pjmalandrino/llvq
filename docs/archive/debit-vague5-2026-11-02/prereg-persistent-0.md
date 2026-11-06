DRAFT, not stamped, not committed.

# Preregistration. A one-launch Tetra pass at full occupancy, the fused Tetra geometry, the cost of a grid barrier on sm_89, and the value of a kernel boundary in the served 4B

**Draft written on 2026-11-02, read-only, for the persistent-kernel lead (`wave5-deep/persistent.md`, step L1). It was
added after an adversarial review, which asked for a cheaper first reading than the persistent arm set of
`prereg-persistent-1.md`. To be committed and TIMESTAMPED BEFORE the run, in `proofs/`, once the code it describes
exists.** Operator go: not given. Cost: about **$0.45** on l40sx1, *estimated* below (§8). Timeout 90 min, because
METHODE §6 asks it of any `planesbench` with five arms or more and this one times five, so **$2.70** at worst. Cap to
ask for this wave, which also covers `prereg-persistent-1.md` if its own preconditions are met: **$3**. Spent before
it: $248.84 over the 232 priced rows of `docs/data/jobs.csv`, 237 rows in all (*computed*). Three bench jobs of
2026-09-20 are missing from the registry (§9).

This step touches no setting frozen on 2026-08-05: its persistent arm is A3's `persall` pattern at today's occupancy,
with 256-thread CTAs, tables in L1 and the 24-float stride. It needs a go, a cap and a stamp (CLAUDE.md rules 1 and 2),
and no reading of decisions 8 or 9.

## 1. Why this run exists

The 4B decodes at 133.3 tok/s after a 5-token prompt (*measured*, `served-grouped-2026-11-02.txt`). Its 7.44 ms step
(*measured*, `gqa-ab-2026-11-01.txt`) streams 1.377 GB, which takes 1.59 ms at 864 GB/s and 2.05 ms at the best read
rate measured on this card, cuBLAS's 671 GB/s (*computed*; 671 GB/s *measured*, `banc-252-2026-10-06`). Two programs
compete to remove the rest. Route P runs one cooperative kernel per layer segment and keeps every bit. Route D fuses
operations in normal grids. Before two to three weeks of persistent-kernel code, five quantities are cheap to read:

- how fast a Tetra pass streams when one launch walks every site, at today's occupancy, with no dependency: the
  upper anchor of a persistent Tetra stream (U1);
- whether a re-gridded Tetra kernel reproduces the served kernel's rows bit for bit on the card (U3). This has never
  been observed. The one attempt, `tv_tetra48_seg` on 2026-09-20, compared two different matrices (§2);
- what fusing q+k+v and gate+up is worth on Tetra in the bench. This is route D's matvec reference, and the
  denominator the A3 grammar requires (`preregistration-a2-a3-geometrie-2026-08-31.md:117-122`);
- the latency of a grid barrier and of a flag hop on sm_89, and the overhead of a cooperative launch (U2 without skew,
  U5). No source publishes them for sm_89;
- what one kernel boundary costs inside the served 4B, by class (U7).

The run also writes the driver-JIT cubins of the bench unit and of the served unit to the bucket. Their SASS is read
off the card at $0. It settles the load widths and the contraction sites that the persistent design rests on
(`persistent.md` §3.7-3.8).

It is a measurement. No line of §7 adopts or kills anything; a kill stays the operator's (METHODE §1).

**Primary quantity**, on the served object: the decode time one added kernel boundary costs inside the served 4B, per
class, in µs, formed round by round (process 2). **Secondary quantities**, bench readings that inform and decide
nothing: the one-launch Tetra pass time and its stream rate, the Tetra fusion delta, row identity, the barrier, hop
and launch microbenchmarks.

## 2. What the code holds at the commit that carries this file

To be written, nothing of it exists today. The commit adds, off by default, so every published path runs as before:

- **The Tetra fusion stream from the Tetra file** (`code/persistent/seg_source_fix.rs`). At HEAD 06e1602, section A4
  of `planesbench` transcodes the fused Tetra stream from the first file's (ball) indices. It launches that stream with
  the ball file's centroids, `gs_off`, `rscale` and tail (`planesbench.rs:3876-3893`, `:3954-4004`, `:3060-3071`). It
  then compares the result with the unfused Tetra arm, which decodes the second file's own words with the second
  file's scales (`:2910-2919`, `:1854-1903`). On 2026-09-20 the two files were two encodes of the same projection. The
  check fired at the first row it read: `row 0 is -0.3080684 fused against -0.57255346 separate`, recorded in
  `llvq-cuda/tests/tetra48_segment_matches_unfused.rs:10-12`. That is a 46.2 % difference (*computed*), a different
  matrix and not a moved association. The fix builds the fused stream from the Tetra file's records. A host test with
  two different sources fails on HEAD and passes after the fix, and its mutant is killed.
- **The Tetra fusion check becomes an invalidation**, on A3's rule (`planesbench.rs:4377-4432`). An owed equality that
  misses, with the f64 reference of the Tetra file within 1e-5 of Σ|w·x|, is reported as a moved association and the
  arm is still timed. Above that, the Tetra fusion arm is invalidated and not timed, and the job goes on. Slot32 and
  Planes14 keep their fatal equality.
- **`tv_tetra48_persall`** (`code/persistent/tetra48_persall.cu`): A3's `tv_planes_persall` with `tetra48_dot`. The
  144 fused sites of a round in one launch, a grid of resident CTAs × 142, 8-row groups, the served tile 64. Behind
  `LLVQ_TETRA_OCC=1`, refused unless `LLVQ_SEG_TETRA=1` and `tetra48` are set. Correctness first, every row against
  `tv_tetra48_seg`; then `tetra48`, `tetra48seg` and `tpersall` interleaved in the same 7 rounds.
- **`LLVQ_PERS_MICRO=1`**: `tv_pers_bar`, `tv_pers_ring` and `tv_pers_empty` from `code/persistent/tetra48_pers.cu`
  §7, in their own NVRTC module, launched with cudarc's `launch_cooperative` (`launch.rs:243`). Each grid is checked
  first with `occupancy_max_active_blocks_per_multiprocessor` (`core.rs:2278`).
- **`LLVQ_DUMP_CUBIN=<dir>`**: the driver-JIT cubin of every module the bench loads, through `cuLinkComplete`
  (`code/persistent/culink_dump.rs`). The bench also assembles the served unit from its 11 parts, prints `served unit:
  101625 bytes, sha256 6193094310bc…` and dumps its cubin.
- **`LLVQ_BOUNDARY_AB=1`** in `bin/fusedrun` (`code/persistent/fusedrun_boundary_ab.rs`). It is refused beside
  `LLVQ_CONFIG`, beside the other measurement modes and beside `LLVQ_KV_PREALLOC`. It turns cudarc's events off itself
  after the single-stream check, sets the grouped form on `Grow(256)` and prints both from the model's getters. Its
  empty kernels come from a separate module on candle's stream.

Host tests on the Mac before the stamp: the seg-source test and its mutant, the `occ.rs` site words of the Tetra port,
and the boundary arms' row partitions.

## 3. What runs

One image built from the commit that carries this file. One job, l40sx1, three files read from the bucket and checked
by sha256: `ball-ref-2026-09-20/qwen3-4b-llvq.bin`, `tetra-4b-2026-09-06/qwen3-4b-tetra.bin` (the bare encode, 252
Tetra records, 0 int4) and `sealed-4b-2026-09-23/qwen3-4b-sealed.bin` (sha256 `886391a8…`). The job script is
`ops/jobs/pers-combined.sh` (`code/persistent/pers-combined.sh`).

1. **Oracle** on Qwen3-0.6B, rule 10.
2. **Process 1, the bench.** `LLVQ_TILE_BLOCKS=64 LLVQ_BENCH_ARMS=slot32,planes14,fp16,nullk,tetra48 LLVQ_SEG_TETRA=1
   LLVQ_TETRA_OCC=1 LLVQ_PERS_MICRO=1 LLVQ_DUMP_CUBIN=<out>/cubin planesbench <ball> <bare Tetra>`. Section A4 needs
   slot32 and planes14 (`planesbench.rs:3861`). There are 7 rounds, 2 discarded, and every arm runs in every round in
   registration order:
   - the table: five arms, a control against `banc-252` (the Slot32 row against `banc-t64-2026-09-20`);
   - A4: Slot32, Planes14 and Tetra, each separate (252 launches) and fused (144 launches). The Tetra pair is
     `tv_f1r_v3g` against `tv_tetra48_seg`;
   - the Tetra occupancy section: `tetra48` (252), `tetra48seg` (144), `tpersall` (1 launch);
   - microbenchmarks. Each is one cooperative launch of 10⁴ hops timed by events, 7 launches with 2 discarded: the
     grid barrier at 142 × 768, 284 × 512 and 852 × 256; a ring of 16 CTAs passing one counter; and 10⁴ back-to-back
     empty launches at 142 × 768, cooperative and normal.
3. **Process 2a, the served door**: `LLVQ_CONFIG=<grouped config> fusedrun <sealed> 16`. It runs for the
   `NVRTC source:` line it prints.
4. **Process 2b, the model.** `LLVQ_FUSED_LAYOUT=tetra48 LLVQ_EMBED=q4 LLVQ_ROT_SHARE=1 LLVQ_FUSE=0 LLVQ_BOUNDARY_AB=1
   fusedrun <sealed> 256`. One model under the grouped form on `Grow(256)`, with cudarc's events off. There are 2
   discarded and 5 timed rounds of 256-token generates from the 5-token prompt, and every arm runs in every round:

   | arm | added per decode token |
   |---|---|
   | `served` | nothing |
   | `split@wave` | gate_proj as 852 + 364 CTAs: 36 boundaries |
   | `split@half` | gate_proj as 608 + 608 CTAs: 36 boundaries |
   | `split-down` | the 24 Tetra down_proj as 160 + 160 CTAs: 24 boundaries |
   | `empty x1` | one 1-CTA empty kernel after each down_proj: 36 launches |
   | `empty x4` | four: 144 launches |
   | `emptygrid x1` | one 852-CTA empty grid after each down_proj: 36 |
   | `candle-tiny x1` | one candle affine on a 1-element tensor per layer: 36 |

   The cost of one added boundary is formed round by round as (t_arm − t_served) / boundaries, with t = elapsed / 256.
   All arms share the head, so the raw and same-head readings coincide.

## 4. Gates

Each voids the run:

1. The image builds, and `oracle` prints MATCH.
2. The three files are at their bytes and sha256 on the mount.
3. Process 1 prints `tetra48: 252 of 252 matrices matched by name`, `tile 64` and `served unit: 101625 bytes, sha256
   6193094310bc…`. Slot32 and Planes14 are bit-exact in A4, as on 2026-09-20 (921,600 rows).
4. Process 2b prints `int4: tv_q4_hh equals tv_q4_h bit for bit`, `cudarc event tracking: off (one stream, ordered by
   issue)` and `attention grouped on grow(256)`. Its `NVRTC source:` line equals process 2a's.
5. No `LLVQ_*` variable is in the job environment; each process passes its own on its command line.
6. Process 2b: any token that differs between arms or rounds voids process 2b, as in `run_trim_ab`.

Each invalidates one arm, never the job:

7. The Tetra fusion arm (against `tv_f1r_v3g`, per part) or `tpersall` (against `tv_tetra48_seg`, per site), if a row
   is outside 1e-5 of Σ|w·x| against the f64 reference of the Tetra file. A row that misses the owed equality but
   holds at 1e-5 is reported as a moved association, with its first row and both values, and is still timed.
   `tpersall` is checked only if the Tetra fusion arm is valid.
8. A kernel reporting local memory (spill) is not timed and is reported.
9. A microbenchmark whose grid fails the occupancy check, or whose launch aborts on the watchdog, is reported and not
   timed.

## 5. Outcomes reported, never gated

- Every arm's median ms and range. In each section, the ratios and deltas are formed round by round.
- The Tetra fusion delta, `tetra48` − `tetra48seg`, in ms and %.
- The one-launch reading: gain = (t(`tetra48seg`) − t(`tpersall`)) / t(`tetra48seg`), round by round, in A3's
  grammar. The stream rate B_pa = 0.9806 GB / t(`tpersall`), counting the f32 family's bytes (*computed* from the row
  formulas, `pers.rs::bare_file_bytes`; the bench prints 0.98 GB).
- Identity counts for the Tetra fusion and `tpersall`: rows identical, rows with a moved association, worst f64 error.
- Microbenchmark medians in µs a hop or a launch, with ranges.
- Process 2b: tok/s per arm, and µs per added boundary with ranges. Derived classes: drain and ramp = split@wave −
  empty x1; underfill = split@half − split@wave; dispatch = emptygrid x1 − empty x1; linearity of empty x4 against
  empty x1.
- The cubins, their sha256, and each function's register count from the loaded module. The SASS is read off the card
  afterwards (§7, rule F).
- The table, as a coherence check only: the bench unit changed, so it is not publishable.

## 6. Hypotheses and signed predictions

H1: one launch over the 144 fused Tetra sites at today's occupancy beats the fused 144-launch geometry by more than
10 %. H2: re-gridded Tetra rows (fused or one-launch) equal the per-matrix kernel's bit for bit on the card. H3: a grid
barrier on sm_89 costs less than a launch. H4: a kernel boundary inside the model costs more than a bare launch.

All points and intervals are *estimated*. No interval reaches below the physical floor of its bytes at 864 GB/s.

| quantity | point | interval |
|---|---|---|
| table: FP16 / nullk / Planes14 / Tetra (bench kernel), ms | 10.99 / 2.29 / 5.01 / 3.70 | each within ±1.5 % of `banc-252` sweep-64 |
| table: Slot32, ms | **5.73** | ±1.5 % of `banc-t64-2026-09-20` (5.726 and 5.732) |
| A4: Slot32 and Planes14 fused rows | **identical** | |
| A4: Tetra fused rows | **identical** | |
| Tetra fusion delta, ms | **0.55** | [0.35, 0.80] |
| `tetra48seg`, ms | **3.15** | [2.90, 3.40] |
| `tpersall`, ms | **2.00** | [1.60, 2.60] |
| `tpersall` rows against `tv_tetra48_seg` | **identical** | |
| one-launch gain over `tetra48seg`, % | **36** | [17, 52] |
| B_pa, GB/s | **490** | [377, 613] |
| barrier at 142 × 768 / 284 × 512 / 852 × 256, µs | **1.3 / 1.5 / 2.2** | [0.7, 2.5] / [0.8, 3.0] / [1.0, 4.5] |
| ring hop, 16 CTAs, µs | **0.8** | [0.3, 1.6] |
| normal empty launch, back to back, µs | **3.0** | [1.8, 5.0] |
| cooperative minus normal empty launch, µs | **+0.8** | [0.0, 3.0] |
| process 2b, `served` tok/s | **134** | [128, 140] |
| µs per boundary: split@wave / split@half / split-down | **5.0 / 6.5 / 5.0** | [3.0, 8.5] / [4.0, 10.0] / [3.0, 9.0] |
| µs per launch: empty x1, empty x4 | **2.5**, linear within ±25 % | [1.5, 3.8] |
| emptygrid x1 minus empty x1, µs | **0.8** | [0.3, 1.5] |
| candle-tiny x1, µs | **2.3** | [2.0, 2.8] |
| tokens, every arm and round | **identical** | |

Reasoning. The fusion delta takes Planes14's 0.569-0.593 ms (*measured*, 2026-09-20 and `a3`) as its order of
magnitude: 108 launches at about 3.6-5.3 µs, plus k/v underfill. `tpersall` uses an additive model on measured inputs
(*estimated*). A3's `persall` left 0.595 ms of warp-per-row geometry: 3.366 − (4.565 − 1.794) on Planes14 (*computed*).
Tetra at tile 64 runs 1.410 ms above its `nullk` floor: 3.701 − 2.291 (*computed*, `banc-252` sweep-64). Their sum is
2.0 ms. The interval is wide because Tetra is bound by its decoder, which persistence does not touch. The 0.98 GB pass
cannot run under 1.135 ms at 864 GB/s, nor under 1.461 ms at the 671 GB/s measured for cuBLAS (*computed*). The
barrier anchors are AlpinDale's 2.2 µs atomic barrier on an RTX 5090 and 1.29 µs for `grid.sync` on an H100, one
author each, both through the R1 fact-check. The launch anchors are 1.85 µs of host submission
(`attribution-cuda-2026-08-05`) and 3.63 µs a launch (`a3-graph-2026-08-06`). The boundary predictions are gap-2's.

## 7. Decision rules

The readings inform the operator. None closes a lead. Each rule ends with an otherwise.

- **A, identity.** If the Tetra fusion and `tpersall` rows are identical, a re-gridded Tetra kernel keeps the bits on
  this card, compiler and driver. That is the first observation of U3, in the f32 family. If a row moves within 1e-5,
  the association is pinned with explicit intrinsics before any route-P code, and the SASS of §5 says which site
  moved. Otherwise the arm is out, the reading uses the rest, and route D's `tv_tetra48_seg_h` waits for the cause.
- **B, one-launch anchor.** If B_pa ≥ 545 GB/s (`tpersall` ≤ 1.80 ms), L2 (`prereg-persistent-1.md`) is recommended,
  subject to decision 8. If B_pa < 377 GB/s (`tpersall` > 2.60 ms), route D comes first, and L2 runs only if the
  operator wants the 1-CTA-per-SM geometry tested anyway. Otherwise L2 is the operator's choice, and the SASS of rule
  F (the shared-memory load widths) weighs on it.
- **C, the A3 grammar.** If the one-launch gain over `tetra48seg` is ≥ 10 % over its whole range, removing launches
  beyond fusion is worth something on Tetra, as a bound. If its whole range is below 10 %, even an unconstrained
  persistent pass adds little over fusion, so route D is proposed and route P's L2 is put to the operator.
  Otherwise, a range that straddles 10 %, both stay open.
- **D, boundaries.** If drain and ramp (split@wave − empty x1) is ≥ 3 µs, route P is designed with fine counters and
  L2 prefetch at phase ends. If it is ≤ 1.5 µs, a barrier per phase is enough. Otherwise both stay open.
- **E, launches and barriers.** If a cooperative launch costs ≥ 3 µs more than a normal one, route P keeps one segment
  per layer, never two. If the 142-CTA barrier is ≥ 4 µs, no better than a launch, parking 4E is proposed, and route
  P stands only if rule B is in its upper branch. Otherwise no constraint follows.
- **F, the SASS** (read off the card at $0, after the job). If the 24 activation reads of `tv_tetra48_h` are 24
  `LDS.32` (8-way at stride 24), the persistent design must change its stride. The cap would be 358 GB/s
  (*computed*), and L2's stride-25 and trio arms become its main question. If they are 6 `LDS.128`, the cap does not
  exist, and L2 keeps those arms as controls with a prior of no effect. Otherwise, a mix, the cap is re-derived from
  the counted wavefronts. Separately, the SASS of the three contractible sites in `tv_tetra48_h`, `tv_f1r_v3g` and
  `tv_tetra48_seg` decides which intrinsics the persistent kernel spells.
- **G.** A prediction that misses is reported as a miss, with the model term it refutes. No threshold here moves after
  the clock.

## 8. Cost

*Estimated* on l40sx1 at $1.80/h, the median of 107 priced jobs of 5 min or more (*computed* from `jobs.csv`).
- Container and image pull: 3 min. Oracle: 1-2 min.
- Process 1:
  - transcoding: 148 s for planes14, fp16, nullk and tetra48 (*measured*, `banc-252`), with the Slot32 transcode always
    built;
  - the A4 fusion transcode: 77 s (*measured*, `banc-tetra-fuse-2026-09-20-brut`);
  - NVRTC units: about 1 min. Checks: about 1 min. Rounds and microbenchmarks: under 1 min;
  - about 7 min in all.
- Process 2: the served door's 16 tokens in under 1 min; the boundary A/B in about 3 min (8 arms × 7 rounds × about
  2 s).

In all, about 15 min: **$0.45**. Anchors: `banc-252` 35 min for $1.03, with its ten-arm build; the three
`banc-tetra-fuse` jobs of 2026-09-20 ran 26-27 min each with ten arms. Timeout 90 min: **$2.70** at worst. The image
build costs no dollar and about 17 minutes.

## 9. What this run does not establish

- The persistent geometry route P uses: one CTA per SM, 24 warps, tables in shared, waits. That is
  `prereg-persistent-1.md`.
- Identity in the f16 family against the served unit's own kernel: also `prereg-persistent-1.md`.
- Any end-to-end speed of route P or route D.
- The int4 rows' persistent rate: the bench holds no int4 record, by the operator's closure of the mixed arm (ETAT §6,
  2026-10-06).
- Anything at 8B, 14B, on sm_120 or on Metal.
- The registry gap. Jobs `6aaf9e00…` (COMPLETED), `6aafac81…` (ERROR, the failed Tetra check) and `6aafc951…` (ERROR,
  the t64 table) of 2026-09-20 ran 1,599, 1,602 and 1,556 s on l40sx1 (*measured*, `hf jobs ps -a`, read on 2026-11-02).
  None is in `docs/data/jobs.csv`: about $2.38 at $1.80/h (*computed*). That is for the operator to register; it
  moves the running total, not this run.

## 10. Deviations

Any departure from this file goes in `<this file>-ECARTS.md`, beside it, never inside it.
