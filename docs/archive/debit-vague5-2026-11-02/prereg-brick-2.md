DRAFT, not stamped, not committed

# Preregistration. Bench β: the integer brick's t(k) with its real decoder, on the 252 matrices of the bare Tetra file

**Draft of 2026-11-02, revised the same day after review; to be committed and TIMESTAMPED BEFORE the run.** Operator
go to launch: **not given**. Rulings owed first: decision 7 (tensor cores, dp4a and ldmatrix in scope), decision 9 (the
ROADMAP §4 hold on batch above 1 and prefill) and a spend cap (decision 2). Cost: about **$0.8-1.5** a try on l40sx1,
*estimated* below (§8). Timeout 90 min, so $2.70 at worst. Proposed cap: **$3**, which covers one retry. Spent before
it: $248.84 over 237 rows of the registry (*measured*, `docs/data/jobs.csv`). It is step S6 of `wave5-deep/brick.md`
§5. Preconditions: `prereg-brick-1.md` fixed the activation width (rule A or B); β0 (`prereg-brick-0.md`) read A, or
the operator chose under its rule B; S5 is done (Rust reference, host replay, warp emulator, mutants). That is 6-9.5
weeks of code from the start of β0 (*estimated*), 9-19 at the verifiers' 1.5-2× factor.

## 1. Why this run exists

Exact speculative decoding multiplies only if verifying k tokens costs about one step: S = τ / (r(k) + d). No published
codebook or trellis kernel has a flat t(k). In CodeGEMM's Table 9, QuIP# and QTIP take 9.53 and 10.48 times their
batch-1 latency at batch 16, and CodeGEMM's own kernel 9.25 (*computed* on measured values, arXiv 2512.17970). β0
measured the incumbent rows kernel and the brick's geometry with a trivial decoder. This run puts the real decoder into
the geometry β0 chose and times t(k) on real words, in one process, against the served kernel and the incumbent rows
kernel. The quantizing epilogue is timed beside it, because the served path pays it 96 times a token.

The brick decodes each Tetra word once into signed int8 quads and feeds them to dp4a (k = 1) or to int8
`mma.m16n8k32` (k ≥ 2) against int16 activation planes, with exact integer block dots and an order-free int64
accumulation (`wave5-deep/brick.md` §3). Its models are not evidence. An operation count does not predict a time:
METHODE §7 records it from P1b on Metal (×1.002 *estimated*, ×2.17 *measured*), and on CUDA a source-level instruction
count missed by a factor of 2 on the August kernel (`docs/archive/spec-lot-a-2026-08-05.md:133-134`). On this card,
the geometry moves the brick makes lost at k = 1 on `Planes14` (*measured*, `a3-occupation-banc-2026-09-01.txt:34-40`).

## 2. What the code would hold

All arms are registered after the existing ones in `llvq-cuda/src/arms.rs`; the new kernels change the NVRTC unit of
every arm, so every ratio is formed within this process. The decoder tables are read through L1 in every arm, as in the
served kernel (`brick.md` §3.4); shared tables are a persistent kernel's question (4E, decision 8).

- `tetra48h` and `tetra48rows` (R = 2, 4, 8, 16 at tiles 128, 128, 64, 32), as in β0.
- `t48inull`: the null arm of the geometry β0 chose, k = 1, 8 and 16: the same grid, staging, combine with its
  self-cleaning last CTA, epilogue and write, no weight read.
- `t48idp4a`: the integer semantics on the dp4a back end, with the served v3 decoder and one XOR to signed quads,
  k = 1, 2, 4, 8.
- `t48imma`: the same decoder writing the swizzled warp tile, `ldmatrix.x4`, two int8 MMAs per k-step, k = 1, 2, 4, 8,
  16.
- `t48ispt`: the competing route, each lane decoding one section straight into its A fragment, no shared tile, k = 1
  and 8.
- In `rotbench`: `rot_apply` against `rot_apply_q` at k = 1, and `rot_apply_rows` against `rot_apply_rows_q` at k = 8
  and 16, at n = 2,560 and 9,728.
- The host side: the tile-major transcode, the quantizer for k seeded Gaussian activation rows (untimed), and the Rust
  reference of the semantics as the replay.

On the Mac, before the job (to be shown in the commit): the host C++ replay of every device function equals the Rust
reference bit for bit on the seven shapes and k = 1..16; the host warp emulator runs the fragment code of `t48imma`
and `t48ispt` and equals the reference, with the layout map that β0's probe checked on the card; the mutants of
`brick.md` §6 step 1 are each killed; the Linux half type-checks from the Mac. No SASS dump is asked for. No tool for
one exists in `ops/` or the crates, and NVRTC's compute_89 PTX is compiled by the driver's JIT (`gpu.rs:262-282`), so a
toolkit dump would not show the code that runs. Registers and spills are read back on the card (gate 5).

## 3. What runs

One image built from the commit. One job, l40sx1, the two files of banc-252 (`ops/jobs/banc-252.sh:22-24`).

1. **Oracle** on Qwen3-0.6B, cuda (rule 10).
2. **The bench.** `LLVQ_BENCH_ARMS=fp16,nullk,tetra48,tetra48h,tetra48rows,t48inull,t48idp4a,t48imma,t48ispt
   planesbench <ball> <tetra>` at tile 64. Every arm at every k in every round, registration order, 7 rounds with 2
   discarded, ratios formed round by round.
3. **The epilogue.** `rotbench` with the four quantizing arms, same rounds and discards.

## 4. Gates, each one voids the run

1. The image builds, and `oracle` prints MATCH.
2. Both files at their bytes and digest on the mount; `252 of 252 matrices matched by name`.
3. **Identity.** Every output row of `t48idp4a`, `t48imma` and `t48ispt`, at every k, equals the host replay bit for
   bit, on all 252 matrices. The rows of a k-token pass equal the first k rows of the 16-token pass. Every round equals
   round 1.
4. The witnesses pass their own checks: `fp16` against f64 at 1e-5; `tetra48h` and `tetra48rows` against f64 at 1e-3
   (f16 outputs, the `AWQ_TOL` reason); `rot_apply_q`'s planes and exponents equal the host quantizer applied to
   `rot_apply`'s output, bit for bit.
5. Every new kernel reads back 0 local bytes through `Cuda::report`; the header reads `tile 64`; no `LLVQ_*` variable
   other than the arm selection is set.

## 5. Outcomes reported, never gated

Medians and round ranges of every t(k); W(k) = t(k) − t(`t48inull`, k), on the same grid; the ratios of §6; register
counts against an expectation of ≤ 96 for `t48imma` at NT = 2; `tetra48rows` row by row against `tetra48h`, equal or
the largest |Δ|; the epilogue's cost per call and per token, which the journal adds to every k = 1 ratio (96 sites a
token at k = 1); for each brick arm the largest error against f64 of the quantized activation (expected: f16 rounding
only) and against f64 of the f32 activation (the quantization effect, reported against Σ|w·x|).

## 6. Signed predictions

| quantity | point | interval |
|---|---|---|
| `tetra48h`, one pass, ms | **3.85** | [3.6, 4.3] |
| `t48idp4a(1) / tetra48h(1)` | **1.02** | [0.88, 1.30] |
| `t48imma(1) / tetra48h(1)` | **1.10** | [0.90, 1.40] |
| `t48imma` r(2) / r(4) / r(8) / r(16) | **1.03 / 1.06 / 1.12 / 1.30** | [1.00, 1.15] / [1.01, 1.25] / [1.03, 1.35] / [1.08, 1.70] |
| `t48idp4a` r(2) / r(4) / r(8) | **1.15 / 1.35 / 1.8** | [1.05, 1.35] / [1.15, 1.7] / [1.4, 2.5] |
| `t48inull` r(8) | **1.10** | [1.03, 1.35] |
| `t48ispt(8) / t48imma(8)` | **1.15** | [0.85, 1.60] |
| quantizing epilogue, added per token at k = 1, ms | **0.08** | [0.03, 0.14] |
| identity, gate 3 | **every row** | |

All *estimated*. The k = 1 lower ends come from the floor-plus-work model (`brick.md` §3.5): nullk's floor plus the
DRAM time of 0.98 GB is 0.89 of `tetra48h`, and only a lower floor of the brick's own goes below it. The upper ends
carry A3's measured losses for split-K and occupancy. The growth with k follows the staging and atomics of the chosen
geometry under assumed L2 rates (`calc/brick/geometry_budget.out`). β0's measurements replace these priors where they
exist; any change is written in the deviations file before the run, never here.

## 7. Readings proposed to the operator

The bench informs and does not gate; the gates of the brick are the T8 control through the kernel with the served-door
tok/s (S8) and Floor 5's end-to-end speculative gate (METHODE §1). A kill is the operator's decision. Each reading
below covers its whole range.

- **Identity.** A differing row voids the timings; the defect is journaled and fixed before any rerun.
- **Flatness.** `t48imma` r(8) ≤ 1.30: the flat-verify bet holds at kernel level on sm_89, and the served integration
  and the k-row path are proposed. r(8) in (1.30, 1.60]: verification is proposed at k ≤ 4 if r(4) ≤ 1.20, and not
  proposed otherwise. Above 1.60: stopping the IMMA route for verification on sm_89 is proposed.
- **k = 1, the epilogue added.** The faster brick arm at ≤ 1.00 of `tetra48h`: the semantics switch costs plain decode
  nothing, and dp4a at k = 1 goes to S8, which decides with the served-door tok/s. Above 1.00: the tax is reported beside
  every speculation estimate, and the operator chooses before S8. One option pays the tax. The other runs the brick at
  k ≥ 2 only, which makes verify ≠ decode and turns exact speculation off.
- **Section per thread.** `t48ispt` below 0.95 of `t48imma` at k = 8: it becomes the default before integration;
  otherwise decode-to-shared stays.
- **L2.** `t48imma` r(16)/r(8) above 1.40: verification is capped at k = 8 on this card; otherwise k = 16 stays open.
- **Otherwise:** every other ratio goes to the journal, and no proposal is made beyond it.

## 8. Cost

*Estimated* on l40sx1 at $1.80/h. Container and image pull 3 min, oracle 2 min, load and transcodes of the selected
arms 4-8 min (banc-252's four-arm build took 148 s, *measured*), host replay of 16 tokens over 252 matrices with the
real decoder 3-8 min on 8 vCPU, f64 references 2-5 min, timed rounds 2-4 min, `rotbench` 2-3 min. In all 18-33 min,
$0.55-1.00, and $0.8-1.5 a try with the usual first-attempt failure. Anchor: banc-252, 35 min, $1.03
(`docs/data/jobs.csv:232`). Timeout 90 min, $2.70 at worst (METHODE §6 asks 90 min for five arms or more). Cap $3.

## 9. Deviations

Any departure from this file goes in `<this file>-ECARTS.md`, beside it, never inside it.

## 10. What this run does not establish

- The whole verify step: int4, the head and attention have no exact k-row path yet (S9).
- Quality: the T8 control through the kernel is S8's; `prereg-brick-1.md` only bounds it on the dense path.
- sm_120 and sm_80 (S7), the 8B and the 14B, a persistent geometry.
- The L2 rates of the L40S, which only `t48inull` and r(16)/r(8) read indirectly.
