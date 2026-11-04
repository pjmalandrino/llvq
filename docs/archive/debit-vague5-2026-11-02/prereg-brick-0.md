DRAFT, not stamped, not committed

# Preregistration. β0: the brick's launch geometry with a trivial decoder, timed before any reference is written

**Draft of 2026-11-02, to be committed and TIMESTAMPED BEFORE the run.** Operator go to launch: **not given**. Rulings
owed first: decision 7 (tensor cores, dp4a and ldmatrix in scope), decision 9 (the ROADMAP §4 hold on batch above 1
and prefill, since this job times multi-row kernels) and a spend cap (decision 2). The F₂ arm also needs decision 8's
reading of "décodage gelé jusqu'au SASS"; without it that arm is not built. Cost: about **$0.5-1.1** a try on l40sx1,
*estimated* below (§8), after about a week of code. Timeout 90 min (METHODE §6 asks it for five arms or more), so $2.70
at worst. Proposed cap: **$3**, which covers one retry. Spent before it: $248.84 over 237 rows of the registry
(*measured*, `docs/data/jobs.csv`). It is step S4 of `wave5-deep/brick.md` §5. It needs neither S3's result nor any
reference code.

## 1. Why this run exists

The brick's tile-major stream hands each warp 16 rows. Keeping today's 1,105,920 warps a pass needs a 16-way K split
per tile, inside the CTA or across CTAs with int64 atomics. On this card the two nearest moves lost at k = 1 on
`Planes14`: split-K with partials and a counter −1.87 % and −9.87 %, several rows a warp −10.45 % and −24.62 %
(*measured*, `a3-occupation-banc-2026-09-01.txt:34-40`).

Four numbers decide whether the brick's geometry can work, and nobody has them:
- its own floor at k = 1, which `docs/format-noyau.md` §6 forbids borrowing from `nullk`;
- the growth of that floor with k. Staging grows 0.24-0.92 GB a pass from k = 1 to k = 8 and int64 atomics by 31-124 M
  (*computed*, `wave5-deep/calc/brick/geometry_budget.out`), at L2 rates never measured on the L40S;
- its occupancy cost: 16-32 resident warps per SM against 48 for `tv_tetra48_h` (*computed* on estimated registers);
- the incumbent rows kernel's real t(k), which only a computation puts at 2.81 for k = 8 (audit T5, on f1e0).

These are geometry questions. A trivial decoder answers them after about a week of code. The reference, emulator and
real decoder cost 5.5-8.5 weeks more, and they are not spent if the geometry alone fails. The host emulator that S5
would build also models the CUTLASS fragment map, and a probe on the card checks that map first.

## 2. What the code would hold

Every arm is registered after the existing ones in `llvq-cuda/src/arms.rs`. The NVRTC unit never changes with the
selection, so the new kernels change the unit of every arm: every ratio is formed within this process, and banc-252's
published figures serve only as a coherence line.

- `tetra48h`: the served `tv_tetra48_h`, k = 1, embedded from `llvq-llm/kernels/tv_tetra48_h.cu` with its
  prerequisites (today only `llvq-llm/src/fused.rs:479` embeds it).
- `tetra48rows`: `tv_tetra48_rows_h` at (R, tile) = (2, 128), (4, 128), (8, 64), (16, 32), four entry points of one
  template.
- `t48gnull`, for two geometries of `brick.md` §3.4: G2 (64 rows × 4 K-groups, K split 4 across CTAs) at k = 1 and 8,
  and G4 (256 rows × 1 K-group, split 16) at k = 1, 8 and 16. Each builds the brick's grid, stages the int16 B records
  of k tokens, and runs the in-CTA and cross-CTA int64 combine with its self-cleaning last CTA, the epilogue and the
  f16 write. It reads no weight. As in `nullk`, the staged activation itself feeds the accumulators, so the compiler
  cannot empty the staging.
- `t48gword`: `t48gnull` plus the tile-major stream (lo32, hi16) folded into the partials, no decode.
- `t48gdp4a`: `t48gword` plus a trivial decoder, byte i of the word replicated into quad i and S = 1, on the dp4a back
  end, k = 1, 2, 4, 8.
- `t48gimma`: the same trivial decoder written to the swizzled warp tile, `ldmatrix.x4`, two int8 MMAs per k-step,
  k = 1, 2, 4, 8, and 16 on G4.
- `tetra48f2`, only under decision 8: `tv_tetra48_h` with the F₂ trellis, `TA[s8] ^ TB[b1 | b2<<1 | b3<<5]`, k = 1.
- Host side: the tile-major transcode of the bare file (bijection tests, odd block counts, the word-zero pad); the
  quantizer for k seeded Gaussian activation rows, untimed; a Rust integer replay of the trivial product and of the
  epilogue.
- `preflight`: a fragment probe. `ldmatrix.x4`, then `mma.m16n8k32` in its .s8.s8 and .s8.u8 forms, run on known
  inputs; all 32 lanes' fragments printed and compared in Rust with the CUTLASS map of `brick.md` §3.3.

On the Mac before the job (to be shown in the commit): the Rust replay equals a direct triple loop on small shapes; the
transcode's bijection tests pass; `CUDARC_CUDA_VERSION=12040 cargo clippy --target x86_64-unknown-linux-gnu -p
llvq-cuda --all-targets` is clean. No new binary is added, so the two lists of `ops/Dockerfile.cuda` and
`UPLOAD_ALLOW` do not move.

## 3. What runs

One image built from the commit. One job, l40sx1, with the two files of banc-252 (`ops/jobs/banc-252.sh:22-24`): the
ball file `ball-ref-2026-09-20/qwen3-4b-llvq.bin` and the bare Tetra file `tetra-4b-2026-09-06/qwen3-4b-tetra.bin`
(sha256 prefix `0adb7cfd`).

1. `preflight` with the fragment probe.
2. **Oracle** on Qwen3-0.6B, cuda (rule 10).
3. **The bench.** `LLVQ_BENCH_ARMS=fp16,nullk,tetra48,tetra48h,tetra48rows,t48gnull,t48gword,t48gdp4a,t48gimma`
   (plus `tetra48f2` under decision 8) `planesbench <ball> <tetra>` at tile 64. One pass is the 252 matrices launched
   back to back, as in banc-252. Every arm at every k in every round, registration order, 7 rounds with 2 discarded,
   ratios formed round by round.

## 4. Gates, each one voids the run

1. The image builds; the fragment probe matches the CUTLASS map on all 32 lanes for `ldmatrix.x4` and both MMA forms;
   `oracle` prints MATCH.
2. Both files at their bytes and digest on the mount; `252 of 252 matrices matched by name`.
3. **Identity.** Every `t48g*` arm that computes a product returns the Rust replay's int64 sums and f16 outputs bit for
   bit, for every matrix, every k and every round. Round r equals round 1, which a missed reset of the self-cleaning
   combine would break. `tetra48f2`, if built, equals `tetra48h` bit for bit.
4. The witnesses pass their own checks: `fp16` against f64 at 1e-5, `tetra48h` and `tetra48rows` against f64 at 1e-3
   (f16 outputs, the `AWQ_TOL` reason).
5. Every new kernel reads back 0 local bytes through `Cuda::report`; the header reads `tile 64`; no `LLVQ_*` variable
   other than the arm selection is set.

## 5. Outcomes reported, never gated

Medians and round ranges of every t(k). F(k) = t(`t48gnull`, k) and W(k) = t(k) − F(k), formed on the same grid. r(k)
for every arm and geometry. `t48gdp4a(1)` and `t48gimma(1)` over `tetra48h(1)`. `t48gnull(1)` beside `nullk`, a ratio of
two floors that is never subtracted (`format-noyau.md` §6). `tetra48rows(R)` over `tetra48h(1)`. `tetra48rows` row by
row against `tetra48h`: equal, or the largest |Δ|; this is the first card check of a rows kernel under `--fmad=true`.
`tetra48f2` over `tetra48h`. Register counts. Atomics a pass, computed from the shapes.

## 6. Signed predictions

| quantity | point | interval |
|---|---|---|
| `tetra48h`, one pass, ms | **3.85** | [3.6, 4.3] |
| `t48gnull(1) / nullk` | **0.95** | [0.70, 1.30] |
| `t48gdp4a(1) / tetra48h(1)`, better geometry | **0.97** | [0.85, 1.25] |
| `t48gimma(1) / tetra48h(1)`, better geometry | **1.05** | [0.88, 1.35] |
| `t48gimma` r(8), better geometry | **1.12** | [1.03, 1.40] |
| `t48gimma` r(16), G4 | **1.30** | [1.08, 1.80] |
| `t48gnull` r(8), better geometry | **1.10** | [1.03, 1.40] |
| G4 over G2, `t48gimma` at k = 8 | **1.10** | [0.80, 1.50] |
| `tetra48rows(8) / tetra48h(1)` | **2.3** | [1.7, 3.1] |
| `tetra48rows(16) / tetra48h(1)` | **3.5** | [2.5, 5.5] |
| `tetra48rows` row by row against `tetra48h` | **some rows differ** | |
| `tetra48f2 / tetra48h` | **0.97** | [0.90, 1.02] |
| identity, gate 3 | **every row** | |

All *estimated*. The k = 1 floor of the served geometry plus the DRAM time of 0.98 GB is 3.43 ms, 0.89 of
`tetra48h`'s 3.85 (*computed* on banc-252's 2.294 and 0.98, *measured*). The trivial decoder costs less than the real
one, which lowers the k = 1 points. A3's losses raise them. The growth with k follows the geometry table under assumed
L2 rates of 3-7 TB/s and 0.1-0.5 T int64 atomics/s (`calc/brick/geometry_budget.out`). The rows kernel's points come
from gap-3's per-row floor and audit T5's 2.81. The rows kernel's FMA chains are contractible under `--fmad=true`, so
some rows should differ from the one-row kernel.

## 7. Readings proposed to the operator

The bench informs and does not gate the lead; a kill is the operator's decision (METHODE §1).

- **Identity.** A gate-3 failure voids the timings; the defect is journaled and fixed before any rerun.

Write Y8 = `tetra48rows(8) / tetra48h(1)`, F8 = the better geometry's `t48gimma` r(8), and K1 = the better geometry's
`t48gdp4a(1) / tetra48h(1)`. Apply in order; the first rule that matches decides.

- **R1.** Y8 ≤ 1.30 and `tetra48rows` equal to `tetra48h` row by row: the incumbent already passes the audit's entry
  gate (t(k)/t(1) ≤ 1.3 up to k = 8, audit §5 step 6) with exact rows on the card. Proposed: stop the brick and take
  the incumbent rows path to decision 9.
- **R2.** Y8 ≤ 1.30 and some row differs: the incumbent is flat but not exact. The brick's case shrinks to exactness,
  and the operator weighs it.
- **C.** F8 > 1.30: the geometry fails the audit's entry gate before any decoder is written. Proposed: no reference
  work on these geometries; a persistent geometry (4E) or another one comes first.
- **B.** F8 ≤ 1.30 and K1 > 0.95: flat, but the room at k = 1 is spent before the real decoder and the quantizing
  epilogue (0.03-0.14 ms a token, *estimated*). The 0.95 is an *estimated* margin for those two. Before S5 the operator
  chooses between a k = 1 tax and running the brick at k ≥ 2 only, which makes verify ≠ decode and turns exact
  speculation off.
- **Otherwise, A.** S5 (reference, replay, emulator) can start on the better geometry, with the operator's go.

Reported beside any reading: the share of F8 that is floor (`t48gnull` r(8)), G4 against G2 (atomics against staging),
and the F₂ arm's ratio, which is bit-identical and could serve today's kernel on its own.

## 8. Cost

*Estimated* on l40sx1 at $1.80/h. Container and image pull 3 min, `preflight` 1 min, oracle 2 min, transcodes of the
selected arms 3-8 min (banc-252's four-arm build took 148 s, *measured*). The integer replay of the trivial product at
k ≤ 16 over the 252 matrices is about 58 G integer operations, 1-3 min on 8 vCPU; the f64 references of the witnesses
2-5 min; the timed rounds 2-4 min. In all 15-30 min, $0.45-0.90, and about $0.5-1.1 a try with the usual first-attempt
failure. Anchors: banc-252, 35 min, $1.03; A3, 237 s, $0.12 (`docs/data/jobs.csv`). Timeout 90 min, $2.70 at worst.
Cap $3.

## 9. Deviations

Any departure from this file goes in `<this file>-ECARTS.md`, beside it, never inside it.

## 10. What this run does not establish

- The real decoder's cost inside the geometry: bench β (`prereg-brick-2.md`).
- Quality: the int16 semantics' T8 (S3, S8).
- The quantizing epilogue's cost: bench β times it in `rotbench`.
- The two L2 rates themselves: the null arms read their combined effect, not each rate.
- sm_120 and sm_80, the 8B and 14B shapes, a persistent geometry.
