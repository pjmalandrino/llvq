# DRAFT, not stamped, not committed

# Preregistration. The rotation kernels and the f16-staged int4 kernel on the L40S: byte identity with the host at every width through 25,600, spill, and isolated latency

**Written on 2026-11-02 by a read-only session, for the operator's review; revised the same day after an adversarial
review (findings in `wave5-deep/bigmodel.md` §10). To be committed and TIMESTAMPED BEFORE the run.** No go given.
Cost: about **$0.30-0.39** on l40sx1 (*estimated*, §9), timeout 30 min, so $0.90 at worst. Cap proposed: **$1**,
which covers one retry. Spent before it: $248.84 over 237 rows of the registry (*computed*, `docs/data/jobs.csv`).

## 1. Why this run exists

- **The wall.** `rot_apply` stages the whole activation in f32 shared memory (`llvq-cuda/src/gpu.rs:813`): 102,400 B
  at the Qwen3-32B `down_proj` width of 25,600, against 101,376 B after opt-in on the L40S (*measured*,
  `fusedrun-14b-2026-08-17.txt`). `rot_apply_dev` (R0) runs the same helpers on a device scratch, as the Metal port
  does, and `rot_apply_rw` (R1w) is the fast form for the same widths. The int4 `down_proj` hits the same bound:
  `tv_q4_hh` stages d_in f32, and `tv_q4_hs` stages f16 bits instead, 51,200 B at 25,600.
- **The time.** The served 4B's 96 rotations sum to 1.04 ms of isolated latency (*computed* from
  `rotation-cuda-2026-08-05.txt` and the 96 sites of wave-1 U1). `rot_apply_r` (R1) does the same arithmetic with 5
  barriers instead of 10 or 11, no runtime division, and no wasted multiply-adds in the mix.
- **The blind spots.** Barriers are vacuous on the host at one thread, so the host test cannot tell a correct barrier
  from a missing one. The int4 host test executes `q4_deq` and a hand-written mirror of the accumulation, never the
  kernel (`llvq-llm/tests/host_tv_q4_h.cpp:15-19`). Only a card can close either, and this run compares bytes, not
  errors.

## 2. What the commit carrying this file must hold

Otherwise no job.

- The kernels R0, R1, R1w and their rows twins (draft: `wave5-deep/code/bigmodel/rotate_wide.cu`), in a translation
  unit of their own, `llvq_rot.cuh` then `rotate_wide.cu`, so the bytes and the printed sha256 of the served unit do
  not move. `tv_q4_hs` and `tv_q4_ht` (draft `tv_q4_wide.cu`) in a unit of their own for the same reason.
- The host test, extended: the shared stand-in raised from 16,384 to 32,768 floats (`host_rotate.cpp:45`); widths
  5,120, 6,144, 8,192, 17,408, 25,600 and 31,744 added to today's eight; byte equality of R0, R1 and R1w with
  `rot_apply`'s text at 1, 7, 32, 256 and 1,024 threads; mutants M1-M7 of `code/bigmodel/host_rotate_wide.cpp`, each
  killed and listed in the commit message.
- `llvq-cuda/tests/fixtures/rot-hashes.txt`: one line per (n, seed): the sha256 of the input bytes (f16 activation,
  sign words, padded `small`, `inv`) and the sha256 of `rot_apply`'s output executed on the host, both hashed by the
  portable `sha256_hex` that the card binary also uses.
- `rotbench` extended (draft: `code/bigmodel/rotbench_wide.rs`). It **compiles the fixture in**:
  `include_str!("../../tests/fixtures/rot-hashes.txt")` from `llvq-cuda/src/bin/rotbench.rs`. The runtime image holds
  binaries, `fetch-qtip.sh` and `configs/` only (`ops/Dockerfile.cuda:131-153, 158, 175`), so a path read at run time
  would not exist on the card. The build stage copies the whole uploaded tree (`:31`), and `llvq-*/**` is in
  `UPLOAD_ALLOW` (`ops/run.py:751`), so the file is there at compile time, and a missing fixture fails the free image
  build instead of a billed job. Under that condition only, no recipe changes: `rotbench` is already in both lists of
  `ops/Dockerfile.cuda` (`:87`, `:147`), and `fusedrun` too (`:53`, `:137`). A runtime `COPY` of the fixture is
  rejected: it would need the `COPY`, `UPLOAD_ALLOW` and the build list to move together, the trap that has bitten four
  times.
- `fusedrun` gains the measurement mode `LLVQ_INT4_WIDE_CHECK=1`, refused beside `LLVQ_CONFIG` like every measurement
  mode, which loads the int4 wide unit and runs the byte checks of §3 step 4.

## 3. What runs

One image built from that commit, one job on l40sx1 through `ops/run.py bench`.

1. **Oracle** on Qwen3-0.6B, rule 10.
2. **`rotbench`**, 12 widths: 2,560, 4,096, 5,120, 6,144, 8,192, 9,728, 12,288, 17,408, 25,600, 31,744, 3,072 and
   1,024. Four kernels: K0 `rot_apply` and R1 run where `rot_kernel()` answers `Shared`, all widths but 25,600 and
   31,744; R0 and R1w run everywhere. For each kernel and width:
   - **verification**: the input's sha256 against the fixture line, then the output's sha256 against it; the f64
     tolerances and the norm check of today's binary; rows twins at 1, 3 and 4 rows, each row byte-equal to the
     single-row output;
   - **timing**: 5 rounds; in each, every applicable kernel runs 20 discarded and 200 timed chained launches, in an
     order rotated by one each round.
3. The register and spill report of every kernel.
4. **`fusedrun`** at the bench door on the sealed 4B (`sealed-4b-2026-09-23/qwen3-4b-sealed.bin`), with
   `LLVQ_INT4_WIDE_CHECK=1`, `LLVQ_FUSED_LAYOUT=tetra48`, `LLVQ_EMBED=q4`, `LLVQ_ROT_SHARE=1`, `LLVQ_FUSE=0`. After load,
   bytes of every output row compared:
   - `tv_q4_hs` against `tv_q4_hh` on the model's first int4 record;
   - on a synthetic record of 256 rows × 25,344 (101,376 B in f32, the widest where all three fit): `tv_q4_hs`,
     `tv_q4_ht` and `tv_q4_hh`, pairwise;
   - on a synthetic record of 256 rows × 25,600: `tv_q4_hs` against `tv_q4_ht` (`tv_q4_hh` cannot run there on this card).
   No token is timed in this step.

**The opt-in, two handles per shared-staged kernel.** The opt-in belongs to the loaded function and may move its
carveout (`rotbench.rs:114-126`). Today's load path poses it once per model, for the widest rotation
(`fused_cuda.rs:768-783`): the served 4B poses none, since 9,728 × 4 = 38,912 B fits the default, and
`func_dynamic_shared` then returns `func` untouched (`gpu.rs:423-447`); the 14B poses 69,632 B for every width. So
K0 and R1 are each loaded twice, from two module loads of the same text: **handle A** with no opt-in, for every width
up to 12,288, the served 4B's object, the object of the 2026-08-05 run, and the object a 32B's narrow widths (5,120,
8,192) would get once 25,600 leaves the shared path; **handle B** at 69,632 B, for 5,120 and 17,408, the served 14B's
object and the object of the 2026-08-17 run. 5,120 runs on both. R0 asks for no dynamic shared memory, and R1w for
gb · m · 4 = 49,152 B at most, on the default bound: one handle each. The journal prints the handle beside every µs
column.

## 4. Gates, each one voids the run

1. The image builds, and `oracle` prints MATCH.
2. The header prints the L40S, sm_89, and both shared bounds: 49,152 and 101,376.
3. Every kernel reports 0 bytes of local memory.
4. At 25,600 and 31,744, `rot_kernel()` prints `Device` and K0 is not launched.
5. Every input sha256 equals its fixture line. A mismatch here is a drift between the host test's draw and the card's,
   not a kernel result, and no output comparison is read.
6. No `LLVQ_*` variable other than those named in §3 is set in the job environment, so NVRTC targets its default
   `compute_89`.

## 5. Outcomes reported, never gated

- For each kernel and width: output sha256 equal to the host's (yes or no); f64 relative and worst-coordinate errors;
  norm error; rows equality.
- For each kernel, width and handle: µs a launch, median and [min-max] over the 5 rounds.
- The ratios R1/K0 (same handle), R0/K0 (handle A at widths up to 12,288, handle B at 17,408) and R1w/R0, formed round
  by round, then their median and range. Never a quotient of medians (rule 7).
- Step 4: every int4 comparison, equal or not, with the first differing row if any.

## 6. Signed predictions

All *estimated*, except that K0 repeats *measured* values on the matching handle.

| quantity | point | interval |
|---|---|---|
| every output sha256 equal to the host's, and every int4 comparison equal | **yes, all** | |
| K0 µs, handle A, at 2,560 / 4,096 / 9,728 / 12,288 | **8.05 / 6.72 / 19.30 / 26.23**, 2026-08-05 | ±15 % |
| K0 µs, handle B, at 17,408 | **29.76**, 2026-08-17 | ±15 % |
| K0 µs at 5,120, handles A and B (never timed) | **11** | [8, 16] |
| K0 µs, handle A, at 6,144 / 8,192 (never timed) | **12 / 13** | [9, 16] / [9, 17] |
| R1 µs at 2,560 (A) | **4.6** | [3.8, 6.0] |
| R1 µs at 9,728 (A) | **6.3** | [4.8, 9.0] |
| R1 µs at 5,120 / 17,408 (B) | **4.8 / 8** | [3.9, 6.5] / [6, 12] |
| R1/K0 at 2,560 and 9,728 (A) | **0.57 and 0.33** | [0.47, 0.75] and [0.25, 0.47] |
| R0/K0 at the widths both run | **1.1** | [0.9, 1.5] |
| R0 µs at 25,600 | **45** | [25, 90] |
| R1w µs at 25,600 | **9** | [6, 15] |
| registers, R1 and R1w | **≤ 56** | no spill |

Reasoning. K0 walks log2(m) barrier stages on one SM, each dividing by a runtime `len`, then a mix padded to 32
terms: above the launch floor, the measured widths cost 0.15 to 0.21 ns per coordinate-stage when k > 1, and 0.07 at
4,096 where there is no mix (*computed* from the measured widths of 2026-08-05 and 2026-08-17). R1 removes the
division, halves the barriers and cuts the mix chain to 8, 24 or 32 terms. What remains is one load round trip, about
five barriers and a mix of n × KB multiply-adds on one SM. R0 keeps K0's instruction stream and moves its staging to
L1 and L2; with no dynamic shared memory requested, the carveout can leave most of the SM's 128 KB to L1. Metal
measured the device-staged form no slower than threadgroup staging (commit `31a5acb`), hence a ratio near 1. The
2026-08-17 run, on the opt-in object, read 7.90 / 6.70 / 18.92 / 26.15 at the four narrow widths (*measured*,
`docs/data/campagne-14b-vitesse-2/rotbench.txt`), within 2 % of 2026-08-05; the two handles keep the question out of
this run's ratios.

## 7. Decision rules

- **I, identity.** If every output sha256 equals the host's, R0, R1 and R1w are bit-exact on the card at every width
  they run, and no T8 question arises. R0 becomes the operator's wall-lift candidate, selected only where
  `rot_kernel()` says `Device`. If any kernel mismatches at any width, that kernel is not used. The mismatch is written
  in `-ECARTS.md` and diagnosed. There is no tolerance path. The same holds for `tv_q4_hs` and `tv_q4_ht` in step 4.
- **II, R1's latency.** This informs only: a bench never pronounces, and the end-to-end A/B decides (METHODE §1).
  Applied only if R1 passed rule I.
  - If R1/K0 has a median ≤ 0.80 at 2,560 and ≤ 0.70 at 9,728, with both ranges below 1.00, the A/B at 4B and 14B is
    written next (`wave5-deep/bigmodel.md` §5, V3).
  - If both medians are ≥ 0.95, proposed to the operator: R1 not pursued on the served path.
  - Otherwise, the operator decides whether V3 runs at 14B only, where the sites are wider.
- **III, the 32B path.** A design proposal that the operator confirms, applied only to kernels that passed rule I.
  - If R1w passed I and its median at 25,600 is ≤ 15 µs, R1w is proposed as the 32B path.
  - Otherwise, if R0 passed I and its median at 25,600 is ≤ 90 µs, R0 is proposed for a 32B's first light, and R1w is
    reworked.
  - Otherwise, R2 (multi-CTA, last-CTA mix) is designed before any 32B is served.
- **IV, the object, per handle.** If K0 on handle A at 2,560, 4,096, 9,728 or 12,288 differs from its 2026-08-05 value
  by more than 15 %, or K0 on handle B at 17,408 from its 2026-08-17 value, that handle's object moved (carveout,
  image, clocks). This run's within-run ratios stand, and no number is compared across runs.

## 8. Deviations

Any departure from this file goes in `<this file>-ECARTS.md`, beside it, never inside it.

## 9. Cost

*Estimated* on l40sx1 at $1.80/h: container and image pull 3 min, oracle 2 min, `rotbench` 3-5 min, `fusedrun` step 4
2-3 min with the 4B's load. The verification covers 4 kernels at 12 widths plus the rows twins, and the timing is
5 × 220 launches per kernel, width and handle, under 2 s of GPU in all. With the job's setup that makes 10-13 min,
**$0.30-0.39**. Timeout 30 min, **$0.90** at worst. Anchors: `gqa-ab-4b`, 10 billed min, $0.29; the rotation job of
2026-08-05, "facturé 0,01 $" in its journal header.

## 10. What this run does not establish

- The end-to-end decode, which is V3. Isolated latency on an empty stream is not in-situ time.
- Whether a faster rotation pays at all: the null-rotation bound (V0c) and the step-0 profile (V0b) say that.
- `tv_q4_hs` against `tv_q4_hh` at 25,600 itself: no 99 KiB card runs the f32-staged kernel there. The optional A100
  job V5 does.
- Any card other than the L40S, and anything about a 32B model. No 32B file exists.
