DRAFT, not stamped, not committed. Revised on 2026-11-02 after an adversarial review (§11).

# Preregistration. The persistent Tetra pass in route P's geometry: stream rate, layout, prefetch, the exposed cost of a dependent wait, and bit identity against the served kernel

**Draft written on 2026-11-02, read-only, for the persistent-kernel lead (`wave5-deep/persistent.md`, step L2). To be
committed and TIMESTAMPED BEFORE the run, in `proofs/`, once the code it describes exists.** Operator go: not given.
Cost: about **$0.36** on l40sx1, *estimated* below (§8). Timeout 90 min, because METHODE §6 asks it of any
`planesbench` with five arms or more and this one times five, so **$2.70** at worst. Cap: the **$3** asked with
`prereg-persistent-0.md`, shared by the two jobs. Spent before it: $248.84 over the 232 priced rows of
`docs/data/jobs.csv` (*computed*), plus whatever step L1 cost.

**Three preconditions, before the stamp:**

1. **Step L1 has run** (`prereg-persistent-0.md`), and its rule B is not in its lower branch, or the operator chooses
   this run anyway. Its measured values re-anchor §6 before the stamp: `tetra48seg`, `tpersall`, the barrier latency,
   and the SASS of the served kernel.
2. **The operator lifts four settings frozen on 2026-08-05** (report decision 8, "Par défaut : ils restent gelés").
   None of them is an ETAT §6 row, so a go under rule 1 lifts them and rule 3 does not apply:
   - CTAs of 512 threads and more: "512 threads/bloc rejeté", `audit-perf-noyau-cuda-2026-08-05.md:152`;
   - the decoder tables in shared memory: "table en shared (régression)", `spec-lot-a-2026-08-05.md:131`, and
     `audit-perf-noyau-cuda-2026-08-05.md:151`;
   - bank padding of the staged activation: "bancs/padding (réfuté)", `spec-lot-a-2026-08-05.md:131`. It was measured
     twice in normal grids: the 24 x reads cost 0.118 ms of a 5.82 ms Slot32 pass
     (`attribution-cuda-2026-08-05.txt:59-66, 120`), and `tv_planes_pad`, the stride-28 staging, gained −0.14 %
     [−0.52 ; −0.08] (`a3-occupation-banc-2026-09-01.txt:33, 112-117, 141-142`). This run reopens padding only as a
     question of the persistent geometry (§1);
   - the decoder frozen until its SASS is read: `spec-lot-a-2026-08-05.md:132`. Step L1's cubin dump meets the
     condition; lifting it stays the operator's decision.
3. **No reading of decision 9 is needed.** The R-row series of the first draft served leads that ROADMAP §4 holds,
   batch above 1 and prefill, so it left this job (§11).

## 1. Why this run exists

Route P runs one cooperative kernel per layer segment, one CTA per SM. Its decode rate rests on two unmeasured
quantities: the rate at which Tetra rows stream with 24 warps per SM instead of 48 (U1), and the exposed cost of one
dependent all-CTA wait, arrival skew included (U2). Its exactness rests on one more: whether a persistent row equals
the served kernel's row bit for bit on the card (U3). Step L1 gives anchors for all three at today's occupancy. This
run measures them in route P's own geometry, against the served unit's `tv_tetra48_h` and against the fused
normal-grid geometry of route D.

Padding is reopened here for one reason, specific to the persistent geometry. If the 24 activation reads of a Tetra
block are 32-bit loads at a 24-float stride, they conflict 8-way: 192 shared-memory wavefronts per warp and block,
against 24 without conflict (*computed*, `pers.rs::scalar_conflict_ways`). At 864 GB/s an SM would need 2.41
wavefronts a cycle for those reads alone, so the stream would be capped near 358 GB/s (*computed* at 2.52 GHz). That
is under every rate route P needs and far above today's 278 GB/s, which is why the normal-grid measurements above
saw no effect. If the reads are six 128-bit loads, the demand is 48 wavefronts and no cap exists. Step L1's SASS says
which (L1 rule F). This run measures the stride-25 and trio-28 layouts either way, and its predictions follow L1's
reading.

It is a measurement. No line of §7 adopts or kills anything; route P's gate is end to end (step L3), and a kill
stays the operator's.

**Secondary quantities only**, bench readings that inform the choice of route: the persistent pass times, the stream
rate B_T, the wait costs h_exp and h_marg, the prologue cost, and row identity.

## 2. What the code holds at the commit that carries this file

To be written, nothing of it exists today. Off by default:

- `llvq-cuda/kernels/tetra48_pers.cu` (`code/persistent/tetra48_pers.cu`): twelve persistent arms, each its own
  `extern "C"` symbol over one template body, `__launch_bounds__(T, 1)`, one CTA per SM, launched with
  `launch_cooperative` after the occupancy check, in a module of their own:

  | arm | symbol | what it is |
  |---|---|---|
  | `pnull` | `tv_pers_null` | staging, claiming, `warp_sum`, stores; no word read |
  | `pword` | `tv_pers_word` | every word read and XOR-folded; no decode |
  | `pl1` | `tv_pers_l1` | full decode, tables read in place through L1, stride 24 |
  | `psmem` | `tv_pers_smem` | tables copied to shared once a launch, stride 24, natural order |
  | `psmem_s25` | `tv_pers_smem_s25` | `psmem` at stride 25: scalar loads with no conflict |
  | `psmem_t28` | `tv_pers_smem_t28` | trio-order staging at stride 28, six explicit `float4` loads a block |
  | `psmem_t28_u2` | `tv_pers_smem_t28_u2` | `psmem_t28` with block j + 32's words loaded while block j decodes |
  | `pt28_512`, `pt28_1024` | `tv_pers_t28_512`, `tv_pers_t28_1024` | `psmem_t28` at 16 and 32 warps per SM |
  | `pchain_w` | `tv_pers_chain_w` | `psmem_t28`'s body plus the served 4B schedule: 145 grid-wide syncs |
  | `pchain_w2` | `tv_pers_chain_w2` | `pchain_w` with one more grid barrier after each sync: 290 |
  | `pchain_wp` | `tv_pers_chain_wp` | `pchain_w` plus per-CTA prologues: 72 norms and rotations at n = 2,560, 24 rotations at 9,728 |

  `pnull` and `pword` run at 768 threads with the trio-28 staging; every other arm runs at 768 threads unless named.
  Every site ends with `__syncthreads()` in every arm, before the claim cursor is reset. The served schedule lays 95
  barrier ordinals in the site list, plus 48 counter waits that every CTA's signal turns into grid-wide waits, plus 2
  trailing barriers: 145 (*computed*, `pers.rs::n_grid_syncs`). Only the 48 waited sites signal.
- `llvq-cuda/src/pers.rs` (`code/persistent/pers.rs`), host-tested on the Mac: the arm table, the site table and its
  validator, the slices, `bar_base` arithmetic (97 barrier calls a launch for `pchain_w`, 242 for `pchain_w2`), the
  staging maps and bank maps, the byte ledger, the shared budget (64,208 B for the trio-28 arms, under the 101,376 B
  opt-in).
- **The reference `tetra48h`**: the served unit assembled from its 11 parts, its digest printed (`served unit: 101625
  bytes, sha256 6193094310bc…` at HEAD 06e1602) and compiled as a module of its own, so its `tv_tetra48_h` is the
  served kernel's text. Its tail is uploaded in binary16, the exact narrowing of the bench's f32 widening of the file's
  binary16 tails.
- **The route-D reference `tetra48seg`**: section A4's `tv_tetra48_seg`, with its stream built from the Tetra file and
  its check non-fatal (step L1's code), beside `tetra48` (`tv_f1r_v3g`) in the same rounds.
- **Identity machinery**: before an arm's identity launch, `y_all` is filled with 0xFFFF and `dbg` points to an f32
  buffer. Every owed row is compared with `tetra48h` bit for bit; no row may remain 0xFFFF. A row that differs is
  judged on its f32 value against the f64 reference at 1e-5 of Σ|w·x|. Timed launches pass a null `dbg`, so the
  timed code and the checked code are the same symbol.
- **Prologue checks** (`pchain_wp`): CTA 0's normed vectors against the L0 host replay of candle's tree, and its rotated
  vectors against `rot_apply` from the served unit, bit for bit.
- `LLVQ_DUMP_CUBIN`, as in step L1, for the persistent module.

## 3. What runs

One image built from the commit that carries this file. One job, l40sx1, two files checked by sha256:
`ball-ref-2026-09-20/qwen3-4b-llvq.bin` and `tetra-4b-2026-09-06/qwen3-4b-tetra.bin`. The job script is
`ops/jobs/pers-arms.sh` (`code/persistent/pers-arms.sh`).

1. **Oracle** on Qwen3-0.6B, rule 10.
2. **The bench**, one process. `LLVQ_TILE_BLOCKS=64 LLVQ_BENCH_ARMS=slot32,planes14,fp16,nullk,tetra48
   LLVQ_SEG_TETRA=1 LLVQ_PERS_ARMS=pnull,pword,pl1,psmem,psmem_s25,psmem_t28,psmem_t28_u2,pt28_512,pt28_1024,pchain_w,pchain_w2,pchain_wp
   LLVQ_DUMP_CUBIN=<out>/cubin planesbench <ball> <bare Tetra>`. There are 7 rounds, 2 discarded. Every arm runs in
   every round in registration order: the five table arms, A4's six arms, then `tetra48h` and the twelve persistent
   arms. Ratios are formed round by round.

## 4. Gates

Each voids the run:

1. The image builds, and `oracle` prints MATCH.
2. The two files are at their bytes and sha256 on the mount.
3. The bench prints `tetra48: 252 of 252 matrices matched by name`, `tile 64` and the served unit's digest named in
   the stamped file. Slot32 and Planes14 are bit-exact in A4.
4. No `LLVQ_*` variable is in the job environment.

Each invalidates one arm, never the job:

5. An owed arm whose rows differ from `tetra48h` with an f32 value outside 1e-5 of Σ|w·x| against the f64 reference.
   A row that differs but holds at 1e-5 is reported as a moved association, with its first row, both binary16 values
   and the error, and the arm is still timed. The binary16 outputs are also reported against f64 at 1e-3, the bench's
   convention for binary16 outputs (`banc-252` sweep-64: "AWQ, cublasf16: 1e-3, binary16 output"; their worst errors
   were 5.7e-5 and 5.8e-5).
6. An arm with any row left at 0xFFFF after its identity launch: rows were skipped.
7. `pchain_wp`, if CTA 0's normed or rotated vectors differ from their references.
8. A launch that aborts on the watchdog, a grid that fails the occupancy check, or a kernel reporting local memory.
   `pnull` and `pword` owe no identity and are only required to run and to write every row.
9. `tetra48seg`, by step L1's rule; without it, the Q ratio of §5 is not formed and §7 rule C reads without it.

## 5. Outcomes reported, never gated

- Every arm's median ms and range, and its ratio to `tetra48h`, round by round.
- B_T = 0.9467 GB / t for the decode arms and 0.9083 GB / t for `pword`: the f16 family's bytes and the words alone
  (*computed*, `pers.rs::bare_file_bytes`).
- h_exp = (t(`pchain_w`) − t(`psmem_t28`)) / 145: the exposed cost of one grid-wide wait, skew included. h_marg =
  (t(`pchain_w2`) − t(`pchain_w`)) / 145: the cost of a barrier reached by CTAs already in step. The skew share is
  h_exp − h_marg. All are formed round by round.
- Prologue cost t(`pchain_wp`) − t(`pchain_w`).
- Q = [t(`pchain_wp`) / t(`tetra48h`)] / [t(`tetra48seg`) / t(`tetra48`)], round by round: the persistent gain beyond
  the fusion gain. Each family is set against its own eager reference: the persistent arms read binary16 tails
  (0.9467 GB), A4's arms f32 tails (0.9806 GB). Q leans toward route D, because `pchain_wp` carries prologue work that
  route D does in other kernels.
- Registers and local bytes of every symbol. The persistent module's cubin, read off the card afterwards.
- The table, as a coherence check only.

## 6. Hypotheses and signed predictions

H1: in one launch with one CTA per SM, the Tetra pass over the bare file runs under half of `tetra48h`'s 252-launch
time. H2: a grid-wide wait exposes less than a launch on sm_89. H3: persistent rows are the served kernel's bit for
bit. H4: in the persistent geometry, the staging layout and a register prefetch each move the pass by 5 % or more.

All points and intervals are *estimated*, drafts to be re-anchored on step L1's measurements before the stamp. Every
lower bound sits above the physical floor of its bytes at 864 GB/s (*computed*): 1.051 ms for `pword` and 1.096 ms for
the decode arms. Most also sit above the time at the best rate measured on this card, 671 GB/s: 1.354 and 1.411 ms.

| quantity | point | interval |
|---|---|---|
| table: FP16 / nullk / Planes14 / Tetra (bench kernel) / Slot32, ms | 10.99 / 2.29 / 5.01 / 3.70 / 5.73 | each within ±1.5 % of `banc-252` sweep-64 (Slot32: `banc-t64`) |
| `tetra48h`, ms | **3.75** | [3.55, 4.05] |
| `tetra48seg`, ms | **3.15** | [2.90, 3.40], or step L1's value ± 1.5 % |
| `pnull`, ms | **0.30** | [0.15, 0.60] |
| `pword`, ms | **1.40** | [1.22, 1.75] |
| `pl1`, ms | **2.10** | [1.50, 3.20] |
| `psmem`, ms | **1.95** | [1.45, 3.00] |
| `psmem_s25`, ms | **1.85** | [1.45, 2.80] |
| `psmem_t28`, ms | **1.80** | [1.45, 2.70] |
| `psmem_t28_u2` / `psmem_t28` | **0.95** | [0.85, 1.02] |
| `pt28_512` / `psmem_t28` | **1.10** | [0.95, 1.30] |
| `pt28_1024` / `psmem_t28` | **0.97** | [0.85, 1.10], or a spill |
| h_exp, µs | **2.0** | [1.0, 4.0] |
| h_marg, µs | **1.3** | [0.6, 2.5] |
| prologue cost, ms | **0.15** | [0.08, 0.35] |
| `pchain_w`, ms | **2.09** | [1.60, 3.30] |
| `pchain_wp`, ms | **2.24** | [1.70, 3.60] |
| Q | **0.70** | [0.50, 1.05] |
| identity, every owed arm | **identical** | |

Reasoning. `pword` streams 0.908 GB. A pure read stream with no decode is the one kernel here that may beat cuBLAS's
671 GB/s on this card; its interval spans 520-745 GB/s. The decode arms add the decoder that binds Tetra today
("il n'est plus limité par le trafic mais par le décodage", `docs/data/jobs.csv`, banc-tetra 2026-09-20) with half
the warps. Their center sits near the additive estimate of step L1's `tpersall` (2.0 ms on 0.98 GB, so about 1.93 ms
on 0.947 GB), minus what tables in shared and the staging layout buy. The prefetch prior follows the latency argument
of `persistent.md` U1: at 24 warps an SM holds about 4.6 KB in flight, against about 3.7 KB that 6.1 GB/s needs over
600 ns (*estimated*). h_marg follows the barrier microbenchmark; h_exp adds a fraction of a row time of skew per
wait. The chained arms are `psmem_t28` + 145 × h + prologues (*computed* from the estimates above).

## 7. Decision rules

The readings inform the operator, who chooses the route. None closes a lead. Each rule ends with an otherwise.

- **A, identity.** If every owed row is identical, persistent rows keep the served bits on this card, compiler and
  driver. If rows move within 1e-5, the moved sites are read in the persistent module's SASS and in the served one's
  (step L1). The persistent kernel spells them with the matching `__fmaf_rn` or `__fmul_rn` and `__fadd_rn` before any
  route-P code, and the arm's time stands. Otherwise the arm is out, and the readings below use the valid arms only.
- **B, layout.** B1: if `psmem_s25` or `psmem_t28` has the whole range of its ratio to `psmem` at or below 0.95, the
  faster of the two is route P's layout. B2: if both ratios have their whole range at or above 0.95, the 24-float
  stride stays, and padding needs no lift. Otherwise, ranges that straddle 0.95, `psmem_t28` stays as the chained arms'
  base and the operator decides whether the margin matters.
- **B', prefetch.** If `psmem_t28_u2` / `psmem_t28` has its whole range at or below 0.95, the stream is
  latency-bound at 24 warps and route P carries the prefetch. If its whole range is at or above 0.98, no latency
  deficit shows. Otherwise it stays an option.
- **C, route.** The best valid decode arm is the fastest arm of MODE 2 or 3 without waits.
  - C1: route P is recommended, step A next, and 4E is worth planning after Floor 3, if four conditions hold: the
    best valid decode arm is at or under 1.65 ms (B_T ≥ 574 GB/s), h_exp is at or under 2.5 µs, `pchain_wp` is at or
    under 2.2 ms, and Q has its whole range at or below 0.90. That last condition is A3's 10 % over the fused geometry
    (`preregistration-a2-a3-geometrie-2026-08-31.md:117-122`). At these edges, route P's model gives about 256 tok/s
    (*estimated*, `calc/persistent/route_p_model.py`).
  - C2: if not C1, but `pchain_wp` is at or under 2.8 ms, h_exp at or under 4.0 µs and Q's whole range below 1.00,
    route P or route D is the operator's choice. Route P's model is then re-derived from the measured B_T and h; at
    these edges it gives about 223 tok/s (*estimated*).
  - C3: otherwise route D comes first, the brick passes ahead of 4E, and the persistent design stays a curve point.
- **D, occupancy.** If `pt28_1024` has its whole ratio range at or below 0.95 without a spill, route P uses 1,024
  threads. Otherwise, if `pt28_512` has its whole range at or below 0.95, route P uses 512 threads. Otherwise 768
  threads.
- **E, skew.** If h_exp − h_marg is at or above 1.0 µs, arrival skew dominates the wait. Route P then balances its
  slices by measured row time rather than by row count, and keeps fine counters where it can. Otherwise barrier
  latency dominates, and slices stay balanced by rows.
- **G.** A prediction that misses is reported as a miss, with the model term it refutes. No threshold here moves
  after the clock.

## 8. Cost

*Estimated* on l40sx1 at $1.80/h (median of 107 priced jobs of 5 min or more, *computed* from `jobs.csv`).
- Container and image pull: 3 min. Oracle: 1-2 min.
- The bench:
  - transcoding: 148 s (*measured*, `banc-252`), plus the A4 fusion transcode, 77 s (*measured*,
    `banc-tetra-fuse-2026-09-20-brut`);
  - NVRTC units, the served unit and the persistent module: about 1.5 min;
  - identity and prologue checks: about 1 min. Rounds: under 1 min;
  - about 7 min in all.

In all, about 12 min: **$0.36**. Timeout 90 min: **$2.70** at worst. The image build costs no dollar and about 17
minutes.

## 9. What this run does not establish

- Any end-to-end speed of route P or 4E: none is built. Step A's A/B (L3) is the first gate.
- The int4 rows' persistent rate: the bench holds no int4 record, by the operator's closure of the mixed arm (ETAT §6,
  2026-10-06).
- The parity of the emulated candle operations with candle's own kernels: route P checks it at load. Here only the
  emulation is checked against its host replay.
- Anything at 8B, 14B, on sm_120 or on Metal.
- Why a layout effect appears or not. Without `nsys` or `ncu` counters, the shared-memory pipe is inferred from the
  SASS and the times, not observed.

## 10. Deviations

Any departure from this file goes in `<this file>-ECARTS.md`, beside it, never inside it.

## 11. What the review of 2026-11-02 changed in this draft

- The eager `tetra48h_t28` arm and its 0.85 prediction are gone. Padding was measured twice in normal grids and was
  worth nothing there (header, precondition 2). It stays here only as a persistent-geometry question, with a stride-25
  arm added for scalar loads.
- The R-row series is gone (decision 9).
- The fallback for a row that differs is the f32 value against f64 at 1e-5. A binary16 output is held to 1e-3.
- A race that let a warp claim the next site's rows in the waitless arms is closed in every arm. `y_all` is poisoned
  before every identity launch.
- h is divided by the 145 grid-wide syncs of the bench schedule, not by 97, and `pchain_w2` gives the skew-free slope.
  The C thresholds were re-derived for that estimator.
- The fused route-D geometry is in the same rounds, and rule C carries the Q ratio of the A3 grammar.
- Lower bounds respect the physical floor. B_T is no longer predicted above the best rate measured on this card.
- The timeout is 90 min, per METHODE §6. The cap is $3, shared with step L1.
- The frozen settings to lift are named in the header, bank padding included.
- `pchain_w` is `psmem_t28` plus waits by construction (`pers.rs::chained_arms_share_the_base_body`). Every arm is a
  symbol. The trailing barriers exist.
- A prefetch arm tests the latency mechanism behind U1.
- The microbenchmarks and the boundary A/B moved to step L1, the cheaper job.
