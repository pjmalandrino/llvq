# Preregistration. The 32B wall lift on the L40S: `rot_apply_dev` and `tv_q4_hs` return the host's bytes

**Written, committed and TIMESTAMPED on 2026-11-04, BEFORE the run.** Written from the draft
`docs/archive/debit-vague5-2026-11-02/prereg-bigmodel-2.md`, narrowed to the wall lift (§8 says what changed).
Operator go given on 2026-11-04 for three card runs, this one third ("ok pour les 3", relayed by the session that
ordered the work). The Space image is rebuilt from the commit that carries this file, after the two runs that go
first. Cost: **$0.30-0.39** on l40sx1 (*estimated*, §9). Timeout 20 min, so **$0.60** at worst (*computed*). No retry
without a new go. Spent before it: $260.89 over 250 rows of the registry (*computed*, `docs/data/jobs.csv`,
2026-11-04), before the two runs that precede this one.

## 1. Why this run exists

- **The wall.** `rot_apply` stages the activation in f32 shared memory: 102,400 B at the Qwen3-32B `down_proj` width,
  25,600, against 101,376 B after opt-in on the L40S (*measured*, `fusedrun-14b-2026-08-17.txt`). The int4
  `tv_q4_hh` stages the same 102,400 B. `rot_apply_dev` (R0, `llvq-cuda/kernels/rotate_wide.cu`) runs the same
  helpers on a device scratch. `tv_q4_hs` (`llvq-llm/kernels/tv_q4_wide.cu`) stages f16 bits, 51,200 B.
- **What the host already proved.** Both equal the kernel they replace byte for byte, at every width up to 31,744
  (*computed*, `llvq-cuda/tests/rotation_matches_rust.rs`, `llvq-llm/tests/proj_q4_wide.rs`; 20 mutants killed,
  listed in the commits).
- **The blind spots only a card closes.** A barrier is vacuous at one thread, so a missing `__syncthreads()` passes
  every host test. The int4 host test replaces `warp_sum` by a recorder. Register spills and the cost exist only on
  a card.

## 2. What the commit carrying this file holds

Otherwise no job.

- R0 and its rows twin in a unit of their own (`llvq_rot.cuh` then `rotate_wide.cu`), and `tv_q4_hs` in a unit of its
  own (`llvq_slot.cuh`, `matvec.cu`, `tv_q4_h.cu`, `tv_q4_wide.cu`). `llvq_rot.cuh`, `rotate.cu`, `tv_q4_h.cu` and
  `llvq_rot.metal` are byte-identical to `validation/vague1` (030747b, and a87f40a, merged in), so the served unit's
  printed sha256 does not move.
- The Linux half of `llvq-llm` (`--features cuda`, `fused_cuda.rs`, `fusedrun`) passes `cargo clippy --all-targets`
  for `x86_64-unknown-linux-gnu` with zero warnings, checked on the Mac by `ops/check-linux-on-mac.sh` (a stub
  `nvcc` writes empty PTX for candle's build script). A type error planted in `fused_cuda.rs`, then in `fusedrun.rs`,
  failed that check.
- `llvq_cuda::wide`: `rot_kernel` answers `Device` exactly where `shared::rot_plan` refuses; `int4_staging` and
  `int4_entry_bytes` give each int4 entry point its own size. Tests pin the served 4B, 8B and 14B on today's kernels
  and sizes on L40S, A100 and H200 bounds (`fused::tests::the_served_models_keep_todays_kernels_after_the_wall_lift`).
- `llvq-cuda/tests/fixtures/rot-hashes.txt`: per card width, the sha256 of the input bytes (f16 activation, sign
  words, KMAX × KMAX padded `small`, `inv`) and of `rot_apply`'s output on the host.
  `llvq-llm/tests/fixtures/q4-wide-hashes.txt`: the sha256 of `tv_q4_hs`'s output on three synthetic records. Both compiled in with `include_str!`, because the
  runtime image holds binaries, `fetch-qtip.sh` and `configs/` only; a missing file fails the free image build. Both
  sit under `llvq-*/**`, in `UPLOAD_ALLOW`. No binary is added, so neither list of `ops/Dockerfile.cuda` moves.
- `rotbench` extended, and `fusedrun` gains the measurement mode `LLVQ_INT4_WIDE_CHECK=1`, refused beside
  `LLVQ_CONFIG` and beside every other mode.
- The job script `ops/jobs/bigmodel-identity.sh`, which refuses to launch without this file's `.ots`.

## 3. What runs

One image built from that commit, one job on l40sx1 through `ops/run.py bench`, no `LLVQ_*` in the job env.

1. **Oracle** on Qwen3-0.6B, rule 10.
2. **`rotbench`**, 14 widths: 2,560, 4,096, 5,120, 6,144, 8,192, 9,728, 12,288, 17,408, 25,600, 31,744, 3,072, 1,024,
   96 and 24. Arms per width, from `llvq_cuda::wide::rot_kernel` on the card's bounds:
   - K0 `rot_apply` on **handle A** (a module with no attribute) at every width that fits the default; on **handle B**
     (a second module of the same text, opted in at 69,632 B) at 5,120 and 17,408; on **handle C** (its own module,
     opted in at n × 4) at any other width past the default that the opt-in holds, which on the L40S is none;
   - R0 `rot_apply_dev` at every width, no shared memory, one handle.

   Per arm and width: the input's sha256 against the fixture, then the output's; today's f64 tolerance and norm as a
   second net; the rows twins at 1, 3 and 4 rows, rows `n + 8` apart and distinct, each row against the single-row
   kernel read at that row's offset and row 0 against the fixture. Then 5 rounds; in each, every arm runs 20
   discarded and 200 timed chained launches, in an order rotated by one each round.
3. The register and spill report of `rot_apply`, `rot_apply_rows`, `rot_apply_dev`, `rot_apply_rows_dev`, `tv_q4_hs`.
4. **`fusedrun`** at the bench door on the sealed 4B (`sealed-4b-2026-09-23/qwen3-4b-sealed.bin`), with
   `LLVQ_INT4_WIDE_CHECK=1 LLVQ_FUSED_LAYOUT=tetra48 LLVQ_EMBED=q4 LLVQ_ROT_SHARE=1 LLVQ_FUSE=0`. After the load,
   bytes of every output row:
   - `tv_q4_hs` against the runtime's `tv_q4_hh` on the model's first int4 record;
   - on the synthetic record 256 × 25,344 (101,376 B in f32, the widest where both fit): `tv_q4_hs`'s output sha256
     against the host's, and `tv_q4_hh` against `tv_q4_hs`;
   - on 256 × 25,600 at `x_off` 0 and 37: `tv_q4_hs`'s output sha256 against the host's.

   No token is decoded or timed. The served load's own check, `tv_q4_hh` equals `tv_q4_h`, prints as always.

## 4. Gates, each one voids the run

1. The image builds, and `oracle` prints MATCH.
2. `rotbench`'s header names the L40S and both bounds, 49,152 and 101,376.
3. Every kernel of step 3 reports 0 bytes of local memory.
4. At 25,600 and 31,744 no K0 arm runs.
5. Every input sha256 equals its fixture line (`rotbench` stops on the first that does not). A mismatch is a drift
   between the host's draw and the card's, not a kernel result.
6. No `LLVQ_*` variable other than those named in §3 is set, so NVRTC targets its default `compute_89`.
7. The 4B loads at the bench door with the parity line `int4: tv_q4_hh equals tv_q4_h bit for bit`, as every load of
   it since 2026-10-30.

## 5. Outcomes reported, never gated

- For each arm and width: output sha256 equal to the host's, yes or no; rows twins equal, yes or no; f64 relative and
  worst-coordinate errors; norm error.
- For each arm and width: µs a launch, median and [min-max] over the 5 rounds, with the handle beside it.
- R0/K0 formed round by round on handle A up to 12,288 and handle B at 17,408, then median and range (rule 7). Never
  a quotient of medians.
- Step 4: every int4 comparison, equal or not, with the first differing row if any.

## 6. Signed predictions

All *estimated*, except that K0 repeats *measured* values on the matching handle.

| quantity | point | interval |
|---|---|---|
| every output and rows sha256 equal to the host's, every int4 comparison equal | **yes, all** | |
| K0 µs, handle A, at 2,560 / 4,096 / 9,728 / 12,288 | **8.05 / 6.72 / 19.30 / 26.23**, 2026-08-05 | ±15 % |
| K0 µs, handle B, at 17,408 | **29.76**, 2026-08-17 | ±15 % |
| K0 µs at 5,120, handles A and B (never timed) | **11** | [8, 16] |
| K0 µs, handle A, at 6,144 / 8,192 (never timed) | **12 / 13** | [9, 16] / [9, 17] |
| R0/K0 at every width where both run | **1.1** | [0.9, 1.5] |
| R0 µs at 25,600 | **45** | [25, 90] |
| R0 µs at 31,744 | **55** | [30, 110] |
| registers, R0 against `rot_apply` | **within 4** | no spill |
| registers, `tv_q4_hs` | **≤ 40** | no spill |

Reasoning. K0 walks log2(m) barrier stages on one SM, each dividing by a runtime `len`, then a mix padded to 32
terms. R0 keeps K0's instruction stream and moves its staging to L1 and L2; with no dynamic shared memory requested,
the carveout can leave most of the SM's 128 KB to L1, enough for the 100 KB working set at 25,600 if little else
competes. Metal measured its device-staged form no slower than threadgroup staging (commit `31a5acb`), hence a ratio
near 1. At 31,744 the mix runs k = 31 of its 32 terms on 1,024 columns, against 25 at 25,600, hence about 20 % more.
`tv_q4_hs` adds one conversion a column to `tv_q4_hh`'s loop and nothing that holds a register across it.

## 7. Decision rules

- **I, identity.** If every output sha256 equals the host's and every rows twin is equal, R0 is bit-exact on the card
  at every width it runs, and no T8 question arises. It becomes the operator's wall-lift candidate, selected only
  where `rot_kernel()` says `Device`. The same holds for `tv_q4_hs` with step 4. If a kernel mismatches at any width,
  that kernel is not used, the mismatch goes in `-ECARTS.md` beside this file and is diagnosed. There is no
  tolerance path.
- **II, the 32B path.** A design proposal that the operator confirms, applied only if R0 passed rule I. If R0's median
  at 25,600 is ≤ 90 µs, R0 is proposed for a 32B's first light (about 3 ms a token over its 64 `down_proj` sites,
  *estimated*). Otherwise R1w (`bigmodel.md` §3.4) or R2 is designed before any 32B is served. A bench never
  pronounces a kill.
- **III, the object, per handle.** If K0 on handle A at 2,560, 4,096, 9,728 or 12,288 differs from its 2026-08-05
  value by more than 15 %, or K0 on handle B at 17,408 from its 2026-08-17 value, that handle's object moved
  (carveout, image, clocks). This run's within-run ratios stand, and no number is compared across runs.

## 8. What changed from the draft of 2026-11-02

- **Scope.** R1, R1w and `tv_q4_ht` are out. R1 waits on the null-rotation bound V0c (`bigmodel.md` §5), and the wall
  needs none of the three. So the rule on R1's latency is gone, and the 32B rule names R0 only.
- **The int4 check at 25,600.** The draft compared `tv_q4_hs` against `tv_q4_ht` there. With no `tv_q4_ht`, it
  compares `tv_q4_hs` against the host's output hash, which is stronger: the host ran `tv_q4_hh` itself at 25,600.
- **Handle C** is named, for cards whose opt-in holds widths past 17,408 (the A100 at 25,600, step V5).
- **Rows twins** are held against the single-row kernel read at each row's offset, which also exercises `x_off` on
  the card.
- **Widths.** 96 and 24 stay, as `rotbench` has always checked them: 14 widths.
- **The load path.** Each int4 entry point gets its own size, `tv_q4_hs` its own opt-in, and an F16-staged record
  refuses an f32 activation by name instead of narrowing it (the three changes of `bigmodel.md` §3.6).

## 9. Cost

*Estimated* on l40sx1 at $1.80/h: container and image pull 3 min, oracle 2 min, `rotbench` 2-4 min (2 kernels at 14
widths, 5 × 220 launches each, under 2 s of GPU in all, plus 3 module loads), `fusedrun` step 4 3-4 min with the 4B's
load. With the job's setup that makes 10-13 min, **$0.30-0.39**. Timeout 20 min, **$0.60** at worst, which leaves
7 min over the high estimate. A timeout is an outcome: the raw output kept so far is read, and a relaunch needs a new
go. Anchor: `gqa-ab-4b`, 10 billed min, $0.29. The Space rebuild from the commit is free and comes first.

## 10. What this run does not establish

- Any end-to-end number, and whether a faster rotation pays (V0b, V0c, V3).
- `tv_q4_hs`'s speed: it is not timed. Whether one CTA of 8 warps an SM streams a 25,600-wide int4 `down_proj` well.
- `tv_q4_hs` against `tv_q4_hh` at 25,600 on a card: no 99 KiB card runs the f32 staging there. The optional A100 job
  V5 does, where `rotbench` runs K0 on handle C at 25,600 beside R0.
- Any card other than the L40S, and anything about a 32B model. No 32B file exists.
