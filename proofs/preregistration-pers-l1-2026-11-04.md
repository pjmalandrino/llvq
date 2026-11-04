# Preregistration. A one-launch Tetra pass at today's occupancy, the cost of a grid barrier on sm_89, and the value of a kernel boundary in the served 4B (step L1 of the persistent lead)

**NOT STAMPED. NOT LAUNCHED.** Written on 2026-11-04 on branch `claude/serene-bohr-1edf69`, from the draft
`docs/archive/debit-vague5-2026-11-02/prereg-persistent-0.md`, once the code it describes existed. To be timestamped
(OpenTimestamps) before the first measurement, then never edited; a departure goes to
`preregistration-pers-l1-2026-11-04-ECARTS.md`.

Operator go: not given. Cost: about **$0.36** on l40sx1 (*estimated*, §8). Timeout 90 min (METHODE §6: a `planesbench`
with five arms or more), so **$2.70** at worst. Cap asked: **$3** for this step and step L2
(`preregistration-pers-l2-2026-11-04.md`) together, whose expected costs sum to $0.66 (*estimated*); if this job runs
into its timeout, L2 does not launch without a new go. Spent before it: **$261.35** over the 246 priced rows of
`docs/data/jobs.csv`, 251 rows in all (*computed* on 2026-11-04; runs launched since then come first).

This step touches no setting frozen on 2026-08-05: its persistent arm runs 256-thread CTAs at today's occupancy, the
decoder tables in L1, the 24-float stride. It needs a go, a cap and a stamp (CLAUDE.md rules 1 and 2), and no reading
of decisions 8 or 9.

## 1. Why this run exists

The 4B decodes 133.3 tok/s after a short prompt (*measured*, `served-grouped-2026-11-02.txt`), a 7.50 ms step of which
the weight stream at 864 GB/s takes 1.59 ms (*computed*). Route P (one cooperative kernel per layer segment) and route
D (fusions in normal grids) compete for the rest. Before route P's two to three weeks of card work, five quantities are
cheap to read:

- the stream rate of a Tetra pass when one launch walks every fused site, at today's occupancy: U1's upper anchor;
- whether a re-gridded Tetra kernel keeps the A4 kernel's bits on the card (U3, f32 family);
- the latency of a grid barrier and of a flag hop on sm_89, and the overhead of a cooperative launch (U2 without skew,
  U5), which no source publishes for sm_89;
- what one kernel boundary costs inside the served 4B, by class (U7);
- the driver's own cubins, read off the card at $0 afterwards (load widths and contraction sites in SASS).

Two of the draft's questions are already answered and leave this run: the PTX probe (*measured*,
`ptx-probe-2026-11-04.txt`: six `ld.shared.v4.f32` a block and `fma.rn.f32` at the three contractible sites, so no
358 GB/s scalar cap at NVVM level) and the Tetra fusion check (*measured*, `seg-tetra-2026-11-04.txt`: 72 of 72 groups
bit for bit, fusion gain 0.763 ms [0.748-0.767]). The fusion is now this run's reference geometry, not its question.

It is a measurement. No rule of §7 adopts or kills anything; a kill stays the operator's (METHODE §1).

**Primary quantity**, on the served object: the decode time one added kernel boundary costs inside the served 4B, per
class, in µs, formed round by round (process 2b). **Secondary quantities**, bench readings that inform and decide
nothing: the one-launch pass time and its stream rate, row identity, the barrier, hop and launch microbenchmarks.

## 2. What the code holds at the commit that carries this file

Off by default; every published path runs as before, and the served unit keeps its 101,625 bytes and sha256
`6193094310bc18acdb0a0c554807ff45f6df09c1f6ab49e0577032874f107812` (pinned by
`llvq-cuda/src/pers.rs::the_served_unit_is_the_published_one`).

- **`LLVQ_TETRA_OCC=1`** in `planesbench` (`llvq-cuda/src/persbench.rs`): `tv_tetra48_persall`
  (`kernels/tetra48_persall.cu`), the 144 fused Tetra sites of a round (72 q+k+v and gate+up groups, 72 single o and
  down) in one launch, a grid of resident CTAs × SMs walking 8-row groups, the served tile 64, in its own NVRTC module.
  Refused unless `LLVQ_SEG_TETRA=1` and tetra48 are set. Every row is compared with the A4 kernel's row on the same
  site (`tv_tetra48_seg` for a group, `tv_f1r_v3g` for a single); then it is timed in A4's rounds.
- **`LLVQ_PERS_MICRO=1`**: `tv_pers_bar`, `tv_pers_ring`, `tv_pers_empty` (`kernels/tetra48_pers.cu`), launched
  cooperatively after the occupancy check.
- **`LLVQ_DUMP_CUBIN=<dir>`**: the driver's JIT (`cuLinkComplete`) of the bench unit and the persistent module, with
  their PTX, sizes and sha256.
- **`LLVQ_BOUNDARY_AB=1`** in `fusedrun` (`llvq-llm/src/boundary.rs`, `fused_cuda.rs`): eight arms, refused beside
  `LLVQ_CONFIG`, every other measurement mode, `LLVQ_KERNEL_DIR` and `LLVQ_KV_PREALLOC`. Events off before the load
  after the single-stream check; grouped attention on `Grow(256)`; empty kernels from their own module on candle's
  stream; every added boundary counted, an arm whose count is off voids the run.

Host checks at $0, at that commit: 18 tests of the kernel text against the served text on the Mac
(`llvq-cuda/tests/pers_matches_served.rs`, 30 mutants killed), 25 unit tests of `pers.rs`, 10 of `boundary.rs`, the
bench's served unit equal to the door's (`llvq-llm/tests/served_unit_copy.rs`), NVRTC 12.4 accepting the persistent
module for compute_89 with no spill in the devel ptxas (`ops/nvrtc-check.sh`, indicative).

## 3. What runs

One image built from the commit that carries this file (no new binary: the two lists of `ops/Dockerfile.cuda` and
`UPLOAD_ALLOW` do not move). One job, l40sx1, `ops/jobs/pers-l1.sh`. Three files read from the bucket, checked by size
then sha256: `ball-ref-2026-09-20/qwen3-4b-llvq.bin`, `tetra-4b-2026-09-06/qwen3-4b-tetra.bin` (252 Tetra records,
sha256 `0adb7cfd…`), `sealed-4b-2026-09-23/qwen3-4b-sealed.bin` (sha256 `886391a8…`).

1. **Oracle** on Qwen3-0.6B (rule 10).
2. **Process 1, the bench.** `LLVQ_TILE_BLOCKS=64 LLVQ_BENCH_ARMS=slot32,planes14,fp16,nullk,tetra48
   LLVQ_SEG_TETRA=1 LLVQ_TETRA_OCC=1 LLVQ_PERS_MICRO=1 LLVQ_DUMP_CUBIN=<out>/cubin planesbench <ball> <tetra>`. 7
   rounds, 2 discarded, every arm in every round: the five table arms, A4's six (Slot32, Planes14, Tetra, each separate
   and fused), then `tpersall`. Microbenchmarks: one cooperative launch of 10⁴ hops, 7 launches, 2 discarded: the grid
   barrier at 142 × 768, 284 × 512 and 852 × 256 (each after the occupancy check); a ring of 16 CTAs; 10⁴ back-to-back
   empty launches at 142 × 768, normal then cooperative.
3. **Process 2a, the served door**: `LLVQ_CONFIG=<grouped config> fusedrun <sealed> 16`, for its `NVRTC source:` line.
4. **Process 2b**: `LLVQ_FUSED_LAYOUT=tetra48 LLVQ_EMBED=q4 LLVQ_ROT_SHARE=1 LLVQ_FUSE=0 LLVQ_BOUNDARY_AB=1 fusedrun
   <sealed> 256`. 2 discarded and 5 timed rounds of 256-token generates from the fixed prompt, every arm in every round:

   | arm | added per decode token |
   |---|---|
   | `served` | nothing |
   | `split@wave` | each gate_proj as 852 + 364 CTAs: 36 boundaries |
   | `split@half` | each gate_proj as 608 + 608 CTAs: 36 |
   | `split-down` | each of the 24 Tetra down_proj as 160 + 160 CTAs: 24 |
   | `empty x1` | one 1-CTA empty kernel after each down_proj: 36 |
   | `empty x4` | four: 144 |
   | `emptygrid x1` | one 852-CTA empty grid after each down_proj: 36 |
   | `candle-tiny x1` | one candle affine on a 1-element tensor per layer: 36 |

   The cost of one added boundary is (t_arm − t_served) / boundaries, t = elapsed / 256, round by round.

## 4. Gates

Each voids the run:

1. The image builds, and `oracle` prints MATCH.
2. The three files are at their bytes and sha256 on the mount.
3. Process 1 prints `tetra48: 252 of 252 matrices matched by name` and `tile 64`; Slot32 and Planes14 are bit-exact in
   A4 (921,600 rows), and A4's Tetra check passes, as on 2026-11-04.
4. Process 2b prints `int4: tv_q4_hh equals tv_q4_h bit for bit`, `cudarc event tracking: off (one stream, ordered by
   issue)` and `attention grouped on grow(256)`; its `NVRTC source:` line equals process 2a's, and carries the sha256
   above.
5. No `LLVQ_*` variable in the job environment.
6. Process 2b: any token that differs between arms or rounds, or an arm whose counted boundaries are off.

Each invalidates one arm, never the job:

7. `tpersall`, if a row differs from the A4 kernel's and its f32 value lies outside 1e-5 of Σ|w·x| against the f64 row
   of the Tetra file. A row that differs within 1e-5 is a moved association: reported with its first row and both
   values, and the arm is still timed.
8. A kernel reporting local memory (spill) is not timed and is reported.
9. A microbenchmark whose grid fails the occupancy check, or whose launch aborts on the watchdog, is reported and not
   timed.

## 5. Outcomes reported, never gated

- Every arm's median ms and range; ratios and deltas formed round by round.
- `tpersall`: gain = (t(`tetra48seg`) − t(`tpersall`)) / t(`tetra48seg`), round by round, the A3 grammar; stream rate
  B_pa = 0.9806 GB / t(`tpersall`) (f32 family, *computed*, `pers.rs::bare_file_bytes`).
- Identity counts: rows identical, rows with a moved association, worst f64 error.
- Microbenchmark medians in µs a barrier, a hop, a launch, with ranges.
- Process 2b: tok/s per arm, µs per added boundary with ranges; derived classes: drain and ramp = split@wave − empty
  x1; underfill = split@half − split@wave; dispatch = emptygrid x1 − empty x1; linearity of empty x4 against x1.
- The cubins and their sha256, each kernel's registers. Their SASS is read off the card afterwards (rule F).
- The table, as a coherence check only: the bench unit is unchanged, the process is not the published one.

## 6. Hypotheses and signed predictions

H1: one launch over the 144 fused Tetra sites at today's occupancy beats the fused 144-launch geometry by more than
10 %. H2: re-gridded Tetra rows equal the A4 kernel's bit for bit on the card. H3: a grid barrier on sm_89 costs less
than a launch. H4: a kernel boundary inside the model costs more than a bare launch.

All points and intervals are *estimated*; derived intervals come from their components' bounds and no lower bound sits
under the floor of its bytes at 864 GB/s (`ops/pers_intervals.py`).

| quantity | point | interval |
|---|---|---|
| table: FP16 / nullk / Planes14 / Tetra (bench kernel) / Slot32, ms | 10.99 / 2.29 / 5.01 / 3.70 / 5.73 | each within ±1.5 % of `banc-252` sweep-64 (Slot32: `banc-t64`) |
| A4: `tetra48` / `tetra48seg`, ms | **3.79 / 3.03** | ±1.5 % of `seg-tetra-2026-11-04` |
| `tpersall`, ms | **2.00** | [1.55, 2.60] (floor 1.135) |
| `tpersall` rows against the A4 kernels | **identical** | |
| one-launch gain over `tetra48seg`, % | **34** | [13, 50] (derived) |
| B_pa, GB/s | **490** | [377, 633] (derived) |
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

Reasoning. `tpersall` uses an additive model on measured inputs: A3's `persall` left 0.595 ms of warp-per-row geometry
on Planes14 (3.366 − (4.565 − 1.794), *computed*), and Tetra at tile 64 runs 1.410 ms above its `nullk` floor
(3.701 − 2.291, *computed*, `banc-252` sweep-64): 2.0 ms. The interval is wide because Tetra is bound by its decoder,
which persistence does not touch. The barrier anchors are AlpinDale's 2.2 µs atomic barrier on an RTX 5090 and
1.29 µs for `grid.sync` on an H100, one author each. The launch anchors are 1.85 µs of host submission
(`attribution-cuda-2026-08-05`) and 3.63 µs a launch (`a3-graph-2026-08-06`). The boundary predictions are the wave-2
gap model's.

## 7. Decision rules

The readings inform the operator. None closes a lead. Each rule ends with an otherwise.

- **A, identity.** If `tpersall`'s rows are identical, a re-gridded Tetra kernel keeps the bits on this card, compiler
  and driver: the first observation of U3 outside A4's own grid. If rows move within 1e-5, the SASS of rule F says
  which site moved, and the persistent kernel spells it before any route-P card work. Otherwise the arm is out and the
  reading uses the rest.
- **B, one-launch anchor.** If B_pa ≥ 545 GB/s over its whole range (`tpersall` ≤ 1.80 ms), L2 is recommended, subject
  to decision 8. If B_pa < 377 GB/s over its whole range (`tpersall` > 2.60 ms), route D comes first and L2 runs only
  if the operator wants the one-CTA-per-SM geometry tested anyway. Otherwise L2 is the operator's choice.
- **C, the A3 grammar.** If the one-launch gain over `tetra48seg` is ≥ 10 % over its whole range, removing launches
  beyond fusion is worth something on Tetra, as a bound. If its whole range is below 10 %, route D is proposed and L2 is
  put to the operator. Otherwise both stay open.
- **D, boundaries.** If drain and ramp is ≥ 3 µs, route P is designed with fine counters and L2 prefetch at phase ends.
  If it is ≤ 1.5 µs, a barrier per phase is enough. Otherwise both stay open.
- **E, launches and barriers.** If a cooperative launch costs ≥ 3 µs more than a normal one, route P keeps one segment
  per layer, never two. If the 142-CTA barrier is ≥ 4 µs, parking 4E is proposed, and route P stands only if rule B is
  in its upper branch. Otherwise no constraint follows.
- **F, the SASS** (read off the card at $0 after the job, from `cubin/`). If `tv_tetra48_h`'s staged reads are not six
  `LDS.128` a block, the persistent design re-derives its stride cap from the counted wavefronts, and L2's stride arms
  become its main question. Otherwise they stay controls with a prior of no effect. Separately, an FFMA at each of the
  three sites confirms the persistent kernel's spelling; a split FMUL + FADD at a site means the driver's ptxas
  differs from NVVM's PTX there, and the persistent kernel is re-spelled before L2.
- **G.** A prediction that misses is reported as a miss, with the model term it refutes. No threshold moves after the
  clock.

## 8. Cost

*Estimated* on l40sx1 at $1.80/h. Container and image pull: 3 min. Oracle: 1-2 min. Process 1: the transcodes of five
arms and the A4 fusion (seg-tetra ran its four-arm bench in 270 s, *measured*), plus `nullk`, the persistent module's
compile and the microbenchmarks: about 6 min. Process 2: the served door's 16 tokens and 8 arms × 7 rounds × about 2 s
of generate: about 3 min with the loads. In all about 12 min: **$0.36**. Timeout 90 min: **$2.70** at worst. The
image build costs no dollar and about 17 minutes.

## 9. What this run does not establish

- Route P's own geometry: one CTA per SM, 24 warps, tables in shared, waits. That is step L2.
- Identity in the f16 family against the served kernel: step L2.
- Any end-to-end speed of route P or route D.
- The int4 rows' persistent rate: the bench holds no int4 record (ETAT §6, 2026-10-06).
- Anything at 8B, 14B, on sm_120 or on Metal.

## 10. Deviations from the draft, written before the stamp

- The A4 Tetra check stays fatal rather than becoming an invalidation: it passed 72 of 72 on 2026-11-04, so a failure
  now is a regression of the bench, which should stop it.
- The draft's L1a (PTX) and L1b (stream source) are done and measured; the A4 fusion is this run's reference.
- Every interval of §6 is derived or checked by `ops/pers_intervals.py`. The draft's one-launch gain [17, 52] could
  not be produced by any `tpersall` inside its own interval with `tetra48seg` in its own; it is [13, 50] now.
- The draft's `fusedrun_boundary_ab.rs` matched projection names with `ends_with("gate_proj")`, and every record name
  ends in `.weight`: it would have split nothing and timed eight copies of the served arm. The mode matches on the
  projection type and counts every added boundary.
