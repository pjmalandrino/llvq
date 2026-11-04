# Preregistration. The persistent Tetra pass in route P's geometry: stream rate, layout, prefetch, the exposed cost of a dependent wait, and bit identity against the served kernel (step L2 of the persistent lead)

**NOT STAMPED. NOT LAUNCHED.** Written on 2026-11-04 on branch `claude/serene-bohr-1edf69`, from the draft
`docs/archive/debit-vague5-2026-11-02/prereg-persistent-1.md`, once the code it describes existed. To be re-anchored on
step L1's measurements (§6), then timestamped before the first measurement and never edited; a departure goes to
`preregistration-pers-l2-2026-11-04-ECARTS.md`.

Operator go: not given. Cost: about **$0.30** on l40sx1 (*estimated*, §8). Timeout 90 min (METHODE §6), so **$2.70**
at worst. Cap: the **$3** asked with `preregistration-pers-l1-2026-11-04.md`, shared by the two jobs. Spent before it:
$261.35 over 246 priced rows on 2026-11-04 (*computed*), plus step L1.

## 0. Three preconditions, before the stamp

1. **Step L1 has run**, and its rule B is not in its lower branch, or the operator chooses this run anyway. L1's
   `tetra48seg`, `tpersall`, barrier latency and SASS re-anchor §6 before the stamp.
2. **Decision 8: the operator lifts four settings frozen on 2026-08-05.** Requested here, not assumed. None is an
   ETAT §6 row, so a go under rule 1 lifts them and rule 3 does not apply. This run needs all four:
   - CTAs of 512 threads and more ("512 threads/bloc rejeté", `audit-perf-noyau-cuda-2026-08-05.md:152`): every arm runs
     768 threads, `pt28_512` 512, `pt28_1024` 1,024;
   - the decoder tables in shared memory ("table en shared (régression)", `spec-lot-a-2026-08-05.md:131`, and
     `audit-perf-noyau-cuda-2026-08-05.md:151`): every MODE 3 arm copies 18,816 B of tables once a launch;
   - bank padding of the staged activation ("bancs/padding (réfuté)", `spec-lot-a-2026-08-05.md:131`): `psmem_s25`
     and the trio-28 arms. Padding was measured twice in normal grids and bought nothing there
     (`attribution-cuda-2026-08-05.txt:59-66`, `a3-occupation-banc-2026-09-01.txt:33`); it is reopened only as a
     question of the persistent geometry, with a prior of no effect since the PTX shows six 128-bit loads a block;
   - the decoder frozen until its SASS is read (`spec-lot-a-2026-08-05.md:132`): step L1's cubin dump meets the
     condition; lifting it stays the operator's call.
   The request, in one line for the operator: *lift the four settings above for the persistent bench arms of
   `planesbench` only, the served path keeping them all.*
3. **No reading of decision 9 is needed**: no R-row series, no batch above 1, no prefill.

## 1. Why this run exists

Route P runs one cooperative kernel per layer segment, one CTA per SM. Its rate rests on two unmeasured quantities:
the rate at which Tetra rows stream with 24 warps a SM instead of 48 (U1), and the exposed cost of one dependent
all-CTA wait, arrival skew included (U2). Its exactness rests on a third: whether a persistent row equals the served
kernel's row bit for bit on the card (U3). Step L1 anchors all three at today's occupancy. This run measures them in
route P's geometry, against the served unit's own `tv_tetra48_h` and against A4's fused geometry in the same rounds.

It is a measurement. No rule of §7 adopts or kills anything; route P's gate is end to end (step L3).

**Secondary quantities only**: the persistent pass times, B_T, h_exp and h_marg, the prologue cost, row identity.

## 2. What the code holds at the commit that carries this file

Off by default. `LLVQ_PERS_ARMS=<arms>` in `planesbench` (`llvq-cuda/src/persbench.rs`), refused unless
`LLVQ_SEG_TETRA=1` and tetra48 are set; a chained arm is refused without `psmem_t28` in the same rounds.

- Twelve arms (`kernels/tetra48_pers.cu`), each its own symbol over one template body, `__launch_bounds__(T, 1)`, one
  CTA a SM, launched cooperatively after the occupancy check, in their own NVRTC module:

  | arm | what it is |
  |---|---|
  | `pnull` | staging, claiming, `warp_sum`, stores; no word read |
  | `pword` | every word read and XOR-folded; no decode |
  | `pl1` | full decode, tables read in place through L1, stride 24 |
  | `psmem` | tables in shared, stride 24, natural order |
  | `psmem_s25` | `psmem` at stride 25: scalar loads with no conflict |
  | `psmem_t28` | trio-order staging at stride 28, six explicit `float4` loads a block |
  | `psmem_t28_u2` | `psmem_t28` with block j + 32's words loaded while block j decodes |
  | `pt28_512`, `pt28_1024` | `psmem_t28` at 16 and 32 warps a SM |
  | `pchain_w` | `psmem_t28`'s body plus the served 4B schedule: 145 grid-wide syncs |
  | `pchain_w2` | `pchain_w` with one more grid barrier after each sync: 290 |
  | `pchain_wp` | `pchain_w` plus per-CTA prologues: 72 norms and rotations at 2,560, 24 rotations at 9,728 |

- Every site ends with `__syncthreads()` before the claim cursor is set again, in every arm (the draft's race, killed
  by a mutant on the host). Each chained arm owns a counter region and an epoch (`pers.rs::Ledger`): the draft shared
  one counter array, so the second chained arm's waits were met before its own rows existed.
- **The reference `tetra48h`**: the served unit rebuilt from its 11 parts (`pers.rs::served_unit_text`, equal to the
  served door's assembly by `llvq-llm/tests/served_unit_copy.rs`), its digest printed, compiled as its own module.
  Tails uploaded in binary16, the exact narrowing of the reader's f32 widening of the file's binary16.
- **Identity machinery**: before an arm's identity launch the rows are filled with 0xFFFF and a debug buffer takes the
  f32 value before `f2h`. Every owed row is compared with `tetra48h` bit for bit; a row that differs is judged on its
  f32 value against the f64 row at 1e-5 of Σ|w·x|, its binary16 value at 1e-3 (`pers.rs::judge_row`). Timed launches
  pass null debug pointers in the same symbol.
- **Prologue checks** (`pchain_wp`): CTA 0's 72 normed vectors against `pers_ref_rmsnorm` (candle's `rmsnorm`
  transcribed in candle's own shape, same `rsqrtf`), its 96 rotated vectors against the served unit's `rot_apply`.

Host checks at $0: every arm on a two-layer schedule at the real widths, three launches and three CTAs, gives the served
rows bit for bit on y and on the f32 value before `f2h`, ends with the barrier counter where the ledger says, and
`pchain_wp`'s prologues equal candle's tree and `rot_apply`; one arm at 768 threads on a full layer
(`pers_matches_served.rs`). NVRTC 12.4 with the devel ptxas: no spill; 64 to 79 registers at 768 threads.

## 3. What runs

One image built from the commit that carries this file. One job, l40sx1, `ops/jobs/pers-l2.sh`, two files checked by
size then sha256: `ball-ref-2026-09-20/qwen3-4b-llvq.bin` and `tetra-4b-2026-09-06/qwen3-4b-tetra.bin`.

1. **Oracle** on Qwen3-0.6B (rule 10).
2. **The bench**, one process: `LLVQ_TILE_BLOCKS=64 LLVQ_BENCH_ARMS=slot32,planes14,fp16,nullk,tetra48
   LLVQ_SEG_TETRA=1 LLVQ_PERS_ARMS=pnull,pword,pl1,psmem,psmem_s25,psmem_t28,psmem_t28_u2,pt28_512,pt28_1024,pchain_w,pchain_w2,pchain_wp
   LLVQ_DUMP_CUBIN=<out>/cubin planesbench <ball> <tetra>`. 7 rounds, 2 discarded; every arm in every round in
   registration order: the five table arms, A4's six, then `tetra48h` and the twelve arms.

## 4. Gates

Each voids the run:

1. The image builds, and `oracle` prints MATCH.
2. The two files are at their bytes and sha256 on the mount.
3. The bench prints `tetra48: 252 of 252 matrices matched by name`, `tile 64` and `served unit: 101625 bytes, sha256
   6193094310bc…`; Slot32, Planes14 and Tetra are bit-exact in A4.
4. No `LLVQ_*` variable in the job environment.

Each invalidates one arm, never the job:

5. An owed arm whose differing row has its f32 value outside 1e-5 of Σ|w·x| against the f64 row, or its binary16 value
   outside 1e-3. Within both, a moved association: reported with its first row, both values and the error, and timed.
6. Any row left at 0xFFFF after the identity launch: rows were skipped.
7. `pchain_wp`, if a normed or rotated vector differs from its reference.
8. A launch that aborts on the watchdog, a grid that fails the occupancy check, or a kernel reporting local memory.
   `pnull` and `pword` owe no identity, only every row written.

## 5. Outcomes reported, never gated

- Every arm's median ms and range, and its ratio to `tetra48h`, round by round.
- B_T = 0.9467 GB / t for the decode arms, 0.9083 GB / t for `pword` (*computed* bytes).
- h_exp = (t(`pchain_w`) − t(`psmem_t28`)) / 145; h_marg = (t(`pchain_w2`) − t(`pchain_w`)) / 145; the skew share
  h_exp − h_marg. Round by round. 145 counts every grid-wide sync, the 48 counter waits included, since every CTA
  signals every waited site in the bench (the draft's divisor 97 overstated h by 1.49).
- Prologue cost t(`pchain_wp`) − t(`pchain_w`).
- Q = [t(`pchain_wp`) / t(`tetra48h`)] / [t(`tetra48seg`) / t(`tetra48`)], round by round.
- Registers and local bytes of every symbol; the persistent and served cubins.

## 6. Hypotheses and signed predictions

H1: in one launch with one CTA a SM, the Tetra pass runs under half of `tetra48h`'s 252-launch time. H2: a grid-wide
wait exposes less than a launch on sm_89. H3: persistent rows are the served kernel's bit for bit. H4: in the persistent
geometry, the staging layout and a register prefetch each move the pass by 5 % or more.

*Estimated*, to re-anchor on L1 before the stamp; derived intervals from `ops/pers_intervals.py`, every lower bound
above the floor of its bytes at 864 GB/s.

| quantity | point | interval |
|---|---|---|
| table and A4 | as L1 measured | ±1.5 % |
| `tetra48h`, ms | **3.75** | [3.55, 4.05] |
| `pnull`, ms | **0.30** | [0.15, 0.60] |
| `pword`, ms | **1.40** | [1.22, 1.75] (floor 1.051) |
| `pl1` / `psmem` / `psmem_s25` / `psmem_t28`, ms | **2.10 / 1.95 / 1.90 / 1.80** | each [1.45, 3.20] (floor 1.096) |
| `psmem_s25`, `psmem_t28` against `psmem` | **0.98** | [0.93, 1.02] |
| `psmem_t28_u2` / `psmem_t28` | **0.95** | [0.85, 1.02] |
| `pt28_512` / `psmem_t28` | **1.10** | [0.95, 1.30] |
| `pt28_1024` / `psmem_t28` | **0.97** | [0.85, 1.10], or a spill |
| h_exp, µs | **2.0** | [1.0, 4.0] |
| h_marg, µs | **1.3** | [0.6, 2.5] |
| prologue cost, ms | **0.15** | [0.08, 0.35] |
| `pchain_w`, ms | **2.09** | [1.60, 3.28] (derived) |
| `pchain_wp`, ms | **2.24** | [1.68, 3.63] (derived) |
| Q | **0.75** | [0.51, 1.29] (derived; the draft's 0.50 was under what its components allow) |
| identity, every owed arm | **identical** | |

Reasoning: as the draft's (§6 of `prereg-persistent-1.md`), with two changes. The layout arms' prior moved to no
effect: the PTX shows six 128-bit loads a block at stride 24, so the 358 GB/s scalar cap does not exist at NVVM level
(`ptx-probe-2026-11-04.txt`). And the derived quantities are computed from their components' intervals.

## 7. Decision rules

The readings inform the operator, who chooses the route. None closes a lead. Each rule ends with an otherwise.

- **A, identity.** If every owed row is identical, persistent rows keep the served bits on this card, compiler and
  driver. If rows move within the tolerances, the moved sites are read in both cubins' SASS and spelled before any
  route-P card work, and the arm's time stands. Otherwise the arm is out and the readings use the valid arms.
- **B, layout.** If `psmem_s25` or `psmem_t28` has its whole ratio range to `psmem` at or below 0.95, the faster is
  route P's layout. If both are at or above 0.95 over their whole ranges, stride 24 stays and padding needs no lift in
  route P. Otherwise `psmem_t28` stays the chained arms' base and the operator decides.
- **B', prefetch.** If `psmem_t28_u2` / `psmem_t28` is at or below 0.95 over its whole range, the stream is
  latency-bound at 24 warps and route P carries the prefetch. If at or above 0.98, no latency deficit shows. Otherwise
  an option.
- **C, route.** C1: route P is recommended, step A next, if the best valid decode arm without waits is at or under
  1.65 ms, h_exp at or under 2.5 µs, `pchain_wp` at or under 2.2 ms, and Q's whole range at or below 0.90 (A3's 10 %).
  C2: if not C1, but `pchain_wp` ≤ 2.8 ms, h_exp ≤ 4.0 µs and Q's whole range below 1.00, the operator chooses. C3:
  otherwise route D first, and the persistent design stays a curve point.
- **D, occupancy.** If `pt28_1024`'s whole range is at or below 0.95 without a spill, route P uses 1,024 threads; else
  if `pt28_512`'s is, 512; otherwise 768.
- **E, skew.** If h_exp − h_marg ≥ 1.0 µs, skew dominates the wait: route P balances slices by measured row time and
  keeps fine counters. Otherwise slices stay balanced by rows.
- **G.** A miss is reported as a miss, with the term it refutes. No threshold moves after the clock.

## 8. Cost

*Estimated* on l40sx1 at $1.80/h. Container and pull 3 min, oracle 1-2 min, the bench about 6 min (L1's process 1
plus the served unit's compile, 13 identity launches and their checks): about 10 min, **$0.30**. Timeout 90 min:
**$2.70** at worst.

## 9. What this run does not establish

- Any end-to-end speed of route P or 4E: step A's A/B (L3) is the first gate, its code is on this branch
  (`kernels/pers_segment.cu`, host-proven) and not yet wired in the model.
- The int4 rows' persistent rate (the bench holds no int4 record).
- The parity of the emulated candle operations with candle's own kernels: route P checks it at load.
- Anything at 8B, 14B, on sm_120 or on Metal.
