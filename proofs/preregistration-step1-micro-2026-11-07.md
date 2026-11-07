# Preregistration. plan-300 step 1: the row loop (K4), the int4 rows (K11) and the counters (K3b), alone on the card

**NOT STAMPED. NOT LAUNCHED.** Written on 2026-11-07 on branch `claude/route-300-toks-de97fe`, once the code it
describes existed. To be timestamped before the first measurement, then never edited; a departure goes to
`preregistration-step1-micro-2026-11-07-ECARTS.md`.

The launch is the operator's. Cost: about **$0.33** on l40sx1 (*estimated*, §8), after a Space rebuild that bills no
card. Timeout 30 min: **$0.90** at worst (*computed* at $1.80/h). Spent before it: see `docs/data/jobs.csv` at launch.

## 1. Why this run exists

The target is the sealed 4B at 300 tok/s at the served door after a short prompt on the L40S.
[`docs/plan-300.md`](../docs/plan-300.md) finds 300 out of reach on the low bound and reachable at the bit on the high
bound only with a row lever. Without one, the high bound stops at 279 to 290 tok/s (*estimated*, plan-300 §6). This
job is plan-300's step 1: it decides whether a row lever exists before any 4E variant is built on an uncalibrated
model (`docs/mesures/pers-4e-rowloop-host-2026-11-05.txt`).

Three questions, one job:
- **K4.** v2's Tetra row loop streams 379 GB/s alone (*measured*, `docs/mesures/pers-rowloop-2026-11-05.txt`). Its
  words served from L1 do not speed it (`l1w` 2.516 ms against v2's 2.499). Does any change of the load path, of
  the word layout, or of the row assignment speed it by ×1.15?
- **K11.** 4E's int4 rows stream about 587 GB/s inside 4E (*measured*, `pers-4e-v3`), against 700 for FlashInfer on
  the same card. Neither the served `tv_q4_hh` nor 4E's row has ever been timed alone. Does a load path or a tile
  layout that keeps every lane's words speed 4E's row by ×1.10?
- **K3b.** 4E crosses 218 grid barriers a token at 0.79 to 1.13 µs each (*measured*, `pers-l1-2026-11-04`). Is a
  counter hop cheaper than a barrier hop by 0.3 µs, at 4E's geometry?

## 2. What the code holds at the commit that carries this file

Off by default; no served path moves. `planesbench` with `LLVQ_TROWS=1` builds its own NVRTC module: the persistent
unit, then `llvq-cuda/kernels/pers_trows.cu`, `pers_q4rows.cu` and `pers_ctr.cu`. 4E's module keeps its bytes.

The geometry is 4E's: one CTA of 512 threads a SM (142 on the L40S), the decoder tables and the activation in shared
memory as `pers_4e.cu` lays them out, each site's rows split over CTAs as `[d c / g, d (c + 1) / g)` and over warps
statically, a CTA barrier between sites, no grid barrier.

### K4, the row loop

The sites are the 252 matrices of the bare Tetra file (`qwen3-4b-tetra.bin`, sha256 `0adb7cfd…`) in served order.
The reference rows are the served `tv_tetra48_h`'s, from the served unit (digest `6193094310bc…`, a gate).

| arm | what changes from v2 | owes the bits |
|---|---|---|
| v2 | nothing: `pers_tetra_row<24, false, true>`, 4E v4's and v6's loop. The base of every K4 ratio | yes |
| nc | the word loads through the read-only path, `ld.global.nc` | yes |
| nc256 | `ld.global.nc.L1::no_allocate.L2::256B`: no L1 allocation, a 256-byte L2 prefetch a request | yes |
| cpa | each warp stages its row two steps ahead (192 B a step) by `cp.async.cg` into a ring of three slots in shared memory, and decodes from there; 67,200 shared bytes | yes |
| tile | the words relaid once at setup so one 128-bit load gives a lane the low words of its blocks l, l+32, l+64, l+96 and one 64-bit load their high halves; the blocks past the last full tile of 128 in plane form. Same bytes a row at the 4B's widths, same lane-to-block map, same fold order | yes |
| bal | the rows of the CTA's slice claimed one at a time from the shared cursor instead of the static warp split | yes |

The nine arms of pers-rowloop run beside them unchanged (v3, ilp2, v2u, l1w, bcast, xbcast, v2_s, v2_o2, k2). v2
replicates pers-rowloop's 379 GB/s. `tile` reads a second copy of the words made for the bench; the setup prints its
GB against the served stream's (rule 6). No file and no served layout changes.

### K11, the int4 rows

84 synthetic records of the 4B's int4 shapes, in served order: v 1024 × 2560 in every layer, o 2560 × 4096 in every
layer, down 2560 × 9728 in layers 12 to 23. Group 128, binary16 scales and biases, 0.4094 GB (*computed*,
`pers::q4rows_bytes`). Values come from a seeded generator; they do not move the time, and identity is held on them.

| arm | what it is | owes the bits |
|---|---|---|
| q4hh | the served `tv_q4_hh`, 84 launches of 256 threads, as the eager path runs them. The reference rows and the reference time | |
| q4 | 4E's row, `pers_q4_row<true>`, one launch over the 84 sites. The base of every K11 ratio | yes |
| q4nc | q4 with the words through `ld.global.nc.L1::no_allocate.L2::256B` | yes |
| q4t | the words relaid once so one 128-bit load gives lane l its words 128t + l + 32s, s = 0..3; the words past the last full tile in served order. Same lane-to-word map, same fold order | yes |
| q4read | diagnostic: the same words by 128-bit loads, scales and biases by 16-bit loads, folded by xor | no |

### K3b, the counters

Each kernel is one cooperative launch of 10,000 hops at 142 × 512. The six run interleaved, 7 reps, 2 discarded, timed
on the host around launch and sync as pers-l1's microbenchmarks.

| kernel | one hop |
|---|---|
| bar | `pers_grid_wait` as 4E calls it: bar.sync, fence, atomicAdd, thread 0 polling with a `__nanosleep` back-off from 32 to 512 ns, fence, bar.sync. pers-l1's barrier, the replication anchor |
| tight | the same with no `__nanosleep` between polls |
| rel | arrival by `red.release.gpu.global.add` with no separate fence, polling by `ld.acquire.gpu` with no back-off, bar.sync |
| ctr8 | rel's arrival and poll on a counter that 8 CTAs signal: CTA c signals its group c / 8, waits on group c / 8 + 1 |
| split | rel's arrival, 2,500 cycles of independent work on every thread (about 1 µs), then the poll |
| work | the same 2,500 cycles and two bar.sync, no grid wait: split's control |

After the reps every counter must hold what the hops put there (`pers::ctr_expected`), or the kernel is void.

### Host checks at $0

- `the_step1_row_loop_arms_are_the_served_rows`: nc, nc256, cpa, tile and bal give the served rows and their f32
  values before `f2h` on 5 sites (d_in 128, 2,560, 4,096, 9,728), at 4 grids, with and without stamps.
- `the_int4_rows_alone_are_the_served_rows`: q4, q4nc and q4t give `tv_q4_hh`'s rows and f32 values on 4 sites at 4
  grids; q4read moves rows.
- `the_counter_kernels_count_every_hop`: every counter kernel runs its hops over two launches at three grids, the
  last group short, with the abort flag down.
- Mutants, each applied exactly once and killed: 11 on the K4 arms, 6 on the int4 arms, 4 on the counters.
- What the host cannot check: the asynchrony of `cp.async` (the host copies at once), the PTX of the loads, the
  memory ordering between SMs, any time.

## 3. What runs

One image built from the commit that carries this file. One job, l40sx1, `ops/jobs/step1-micro.sh`.

1. **Oracle** on Qwen3-0.6B (rule 10).
2. The two banc-252 files are checked by size, the Tetra file by sha256. Then `LLVQ_TILE_BLOCKS=64
   LLVQ_BENCH_ARMS=slot32,planes14,fp16,tetra48 LLVQ_SEG_TETRA=1 LLVQ_TETRA_OCC=1 LLVQ_TROWS=1 LLVQ_DUMP_CUBIN=<dir>
   planesbench <ball> <tetra>`: pers-rowloop's command. The K3b counters run once at setup, the K4 and K11 arms in
   A4's 7 rounds, 2 discarded.

## 4. Gates

Each voids the run:

1. The image builds; `oracle` prints MATCH.
2. The files at their bytes, the Tetra file at its sha256; `tetra48: 252 of 252 matrices matched by name`; `tile 64`.
3. The references print the served unit, 101,625 bytes, sha256 `6193094310bc…`.
4. v2 prints every row identical bit for bit to `tv_tetra48_h`; q4 prints every row identical bit for bit to
   `tv_q4_hh`.
5. `bar: counts OK`.
6. No `LLVQ_*` variable in the job environment.

Another arm that spills, does not fit, moves a row, aborts or miscounts voids only its own numbers.

## 5. Outcomes reported, never gated

- Each symbol's registers and local bytes under the driver's ptxas; the cubins.
- K4: per arm ms and GB/s, median and range, on words at the row stride, binary16 tails and row scales; speed over
  v2 round by round; the stamps (cycles a step, warp balance, clock) as in pers-rowloop. `tile`'s GB on card.
- K11: per arm ms and GB/s on 0.4094 GB; speed over q4hh and over q4, round by round.
- K3b: µs a hop, median and range; tight, rel and ctr8 under bar rep by rep; split over work rep by rep.

## 6. Signed predictions

*Estimated* from pers-rowloop, pers-4e-v3, pers-l1 and fi-decode-anchor; nothing fitted to a number of this run.

| quantity | point | interval |
|---|---|---|
| every arm that owes the bits, against its reference | **identical** | |
| every symbol loads without local memory (driver ptxas) | **yes** | |
| v2, GB/s | **379** | [360, 395] |
| nc over v2 | **×1.00** | [0.95, 1.08] |
| nc256 over v2 | **×1.02** | [0.95, 1.12] |
| cpa over v2 | **×0.97** | [0.85, 1.10] |
| tile over v2 | **×1.03** | [0.95, 1.15] |
| bal over v2 | **×1.00** | [0.97, 1.06] |
| some K4 arm at ×1.15 or more | **no** | |
| q4hh, GB/s | **450** | [350, 580] |
| q4, GB/s | **600** | [500, 700] |
| q4nc over q4 | **×1.03** | [0.97, 1.12] |
| q4t over q4 | **×1.06** | [0.97, 1.20] |
| q4read, GB/s | **720** | [600, 820] |
| bar, µs a hop | **0.95** | [0.75, 1.20] |
| tight, µs a hop | **0.75** | [0.50, 1.05] |
| rel, µs a hop | **0.70** | [0.45, 1.00] |
| ctr8, µs a hop | **0.60** | [0.35, 0.95] |
| split over work, µs a hop | **0.45** | [0.15, 0.90] |

Why bal sits at ×1.00: the d_in 4,096 and 9,728 sites hold 2,560 rows, 18 a CTA over 16 warps, so two warps run two
rows whatever assigns them. pers-rowloop's balance 0.626 and 0.637 lies near the 0.563 that row counts alone give
(*computed*). Dynamic claiming fixes uneven row times, not uneven row counts; splitting a row over warps is 4E v5,
closed in `docs/ETAT.md` §6.

## 7. Decision rules

- **A, identity.** An arm that owes the bits and moves a row is void, and its first divergence goes to the host with
  the card's rows. Gate 4 covers v2 and q4.
- **B, K4, the bar of plan-300.** For nc, nc256, cpa, tile and bal, the median of the round-by-round speed over v2:
  - at **×1.15** or more: a lever. It licenses the 4E port of plan-300 step 4, ranked by speed; 300 at the bit stays
    in play on the high bound.
  - from ×1.10 to ×1.15: reported, not a lever. No port on its strength alone.
  - under **×1.10**: dead for this route.
  - No arm at ×1.15: **300 tok/s at the bit is out on every bound plan-300 can anchor**. The route continues to steps
    2 and 3 toward 279 to 290 tok/s at most, and the K4 port of step 4 is dropped. The operator decides whether to go
    on.
- **C, K11.** q4nc or q4t over q4 at **×1.10** or more licenses the int4 port of step 4 (0.11 ms high, plan-300 §4).
  Under ×1.10, K11 is dead and step 4 carries no int4 port. q4 over q4hh is reported: the share of the eager int4
  path that 4E already recovered.
- **D, K3b.** tight, rel or ctr8 under bar by **0.3 µs** a hop or more, median of the rep-by-rep differences: the
  counter form of K3b is licensed for step 3, with the best of the three. Under 0.3 µs for all three: the barrier's
  mechanics are not the lever, K3b's high bound is re-estimated from the walls over mean alone before step 3 is
  costed. split over work under 0.3 µs: arrival and wait can be split in 4E to hide the hop behind independent work;
  that is reported, not gated.
- **E, replication.** v2 outside [360, 395] GB/s or bar outside [0.75, 1.20] µs: the card or the build differs from
  the runs they replicate. The ratios still stand, round by round; the absolute numbers are reported with the miss.
- **F.** A miss is reported as a miss, with the term of §6 it refutes. No threshold moves after the clock.

## 8. Cost

*Estimated* on l40sx1 at $1.80/h:

| step | time |
|---|---|
| container and image | 3 min |
| oracle | 1 to 2 min |
| planesbench: pers-rowloop's 497 s less its oracle and start, plus the five K4 arms, the K11 setup (0.41 GB generated and uploaded, 84 reference launches) and the counters | about 6 min |

About 11 min, **$0.33**. Timeout 30 min: **$0.90** at worst.

Before it, a Space rebuild from this commit (`uv run ops/run.py publish`): it bills no card, about 12 min of build.

## 9. What this run does not establish

- Any 4E tok/s. No 4E variant runs, and an arm alone that passes its bar still owes a 4E port and a door reading.
- The in-situ costs the bench does not carry: the epilogues, the grid barriers between phases, attention, norms,
  rotations.
- Real int4 records: the values are synthetic, the shapes and the bytes are the 4B's.
- Any card but the L40S.
