# Preregistration. The split-KV attention kernel alone: identity, read rate, verify cost, unit size

**Finalized on 2026-11-04. NOT TIMESTAMPED, operator go to launch NOT GIVEN.** Stamp it with OpenTimestamps
before the first measurement (rule 2); after the stamp this file is never edited, a deviation goes beside it in
`proofs/preregistration-attn-bench-2026-11-04-ECARTS.md`. Cost: **$0.18 to $0.24** a run on l40sx1, *estimated*
in §8, timeout 25 min, so $0.75 at worst. Cap asked: **$1**, which covers a second run after one bit-neutral
revision through `LLVQ_KERNEL_DIR`. Spent before it: $256.88 over 248 rows of the registry, 5 rows unpriced
(*computed* from `docs/data/jobs.csv` on 2026-11-04).

It descends from the draft `docs/archive/debit-vague5-2026-11-02/prereg-attention-2a.md`. What changed from that
draft, and why, is in §10.

## 1. Why this run exists

The 4B decodes 77.5 tok/s at 8,192 positions, and its attention reads the KV cache at η 0.25-0.26 of 864 GB/s
(*computed*, `docs/archive/debit-vague5-2026-11-02/attention.md` §8 item 9). FlashInfer 0.6.14's batch-1 GQA
decode kernel reads the same geometry on the same card at η 0.810 at 8,192 (*measured*,
`docs/mesures/fi-decode-anchor-2026-11-04.txt`). Reading 1 of that run proposed writing our own kernel, readings
2 and 3 restricted its first version to CUDA cores and one CTA per split. That kernel now exists (§2). Its
arithmetic is ours and exact (explicit rounding, a polynomial exp2), which costs instructions FlashInfer does not
spend, so its read rate on Ada is unknown. So is the cost of a verify call of k rows against a decode call.

This bench measures both on the card before any model integration is timed, so a revision costs one mini-job
through `LLVQ_KERNEL_DIR` and no image rebuild. It also chooses one bit-relevant constant, the unit size, before
the arithmetic is frozen: no quality control has run on either unit size, so choosing between them by speed
biases nothing. It is a bench: it informs the build and gates nothing (METHODE §1). The in-situ A/B
(`prereg-attention-2`, to be finalized) decides speed, and the T8 control decides quality.

## 2. What the code holds, and what the Mac showed

All *measured* on the dev Mac (M-series, clang++ from the Xcode toolchain), commit that carries this file.

- `llvq-cuda/src/attn_ref.rs`: the canonical arithmetic replayed in Rust. 16 lanes of 8 dims, an fma chain from
  +0 per lane, the xor tree 8, 4, 2, 1, one product by c = f32(log2(e) / sqrt(128)) = 0x3e0293ee; tiles of 8
  positions, units of 64 (served) or 32, splits of 256 aligned to absolute positions, left folds; f16 output of
  acc / d; exp2 by a minimax degree-6 polynomial with p(0) = 1, within 1.18 ulp of 2^x on [-126, 0].
  `tests/attn_ref_props.rs`: within 4e-6 of an f64 softmax attention, relative to the row's magnitude, at L in
  {1, 7, 8, 9, 63, 64, 65, 255, 256, 257, 1000, 4096}, both unit sizes; rows invariant (a verify call of k rows
  equals k decode calls) at 13 positions and k = 1 to 8; splits absolute; the f16 conversions exact on every
  half and round-to-nearest-even on every midpoint.
- `llvq-cuda/kernels/attn_dec.cu`: one self-contained NVRTC unit, four entry points `attn_dec_g{4,5}_u{64,32}`.
  One CTA per (split, KV head), 128 threads, 8 groups of 16 lanes; tiles staged in shared memory with the next
  step's loads in flight in registers; unit states folded in order by the CTA; one ticket per launched CTA, the
  last folds the splits (planes_occ.cu's `occ_sk`). Static shared memory 16 KB at unit 64, 32 KB at unit 32.
- `llvq-cuda/tests/attn_dec_matches_rust.rs`: the kernel TEXT, compiled by clang++ and EXECUTED on
  `tests/host_cta.h` (one fiber per thread, real barriers, 16-lane shuffles that refuse a wrong mask, tickets,
  CTAs in a permuted order), equals the replay bit for bit on 124 launches, two of them under UBSan: rows ending at 1, 7, 8, 9, 63, 64, 65,
  255, 256, 257, 520 and verify calls straddling 256, G 4 and 5, unit 64 and 32, eager grids, grids one split
  wider than the capacity, three finishing orders, and calls where the kernel writes the fresh rows itself. The
  k rows of a verify call equal k decode calls on the kernel itself. `attn_ex2` equals `ex2_fixed` on 200,012
  inputs, edges included.
- Mutants (rule 10): 17 on the kernel text and 5 on the replay, each checked to apply at one site: 21 killed,
  1 equivalent (K5, `docs/mesures/attn-kernel-host-2026-11-04.txt`).
- `llvq-cuda/src/attn.rs`: the launcher and `AttnDec::self_check`, the load-time check of the served path: rows
  ending at 1, 255, 256, 257, 1000, 4096 with k in {1, 3, 8}, eager and wide grids, verify rows alone, fresh
  rows written by the kernel, the card against the replay bit for bit. Type-checked for Linux on the Mac
  (`CUDARC_CUDA_VERSION=12040 cargo clippy --target x86_64-unknown-linux-gnu -p llvq-cuda --all-targets`, zero
  warnings). Never run on a card.
- `bin/cuhcheck` parses the unit; `llvq_cuda::embedded_source("attn_dec.cu")` carries it.

What the Mac cannot show: NVRTC's reading of the text, the inline PTX conversions (`cvt.f32.f16`,
`cvt.rn.f16.f32`, replaced by `_Float16` on the host), memory ordering between CTAs that run at once, registers,
spills and speed.

## 3. What runs

One image built from the commit that carries this file and `bin/attnbench` (in both lists of
`ops/Dockerfile.cuda`). One job, l40sx1: `ops/jobs/attn-bench.sh`. No artifact is loaded.

1. **Oracle** on Qwen3-0.6B, rule 10.
2. **`attnbench 2048,4096,8192`**, which prints in order:
   1. the source line (`attn_dec.cu: embedded, sha256 …`) and the card;
   2. **load report** of the four entry points: registers, local bytes (spill), static shared bytes, sm;
   3. **identity, every variant**: the served load-time check (`self_check`), then at each L and k in
      {1, 2, 4, 8} (rows at L - k to L - 1, the last row of length L) the card against the CPU replay, an eager
      grid and a grid two splits wider than the capacity, all bit for bit; at k = 8 each row of the verify call
      against its own decode call;
   4. **timing**: G = 4 on 36 synthetic per-layer KV sets of the 4B geometry (8 KV heads, 128, f16,
      `attn_ref::fill`), G = 5 on 40 sets (the 14B's geometry). Each set is its own buffer, so a pass of 36 or
      40 eager launches on one stream reads 36 (or 40) x 4,096 x L bytes and reuses none, above the 96 MB L2 at
      every L. CUDA events around each pass, fresh stream, cudarc event tracking off. 2 discarded and 7 timed
      rounds; inside a round every (variant, L, k) runs once in one fixed order. Variants:
      - **V-64**: unit 64, the served arithmetic (`attn_dec_g4_u64`, `attn_dec_g5_u64`);
      - **V-32**: unit 32, a different canonical arithmetic with its own replay (`attn_dec_g4_u32`,
        `attn_dec_g5_u32`).

Derived, *computed*: η_k(L) = sets x 4,096 x L / (t x 864 GB/s); r_k(L) = t(k) / t(1); both formed round by
round, medians with ranges (rule 7).

## 4. Gates, each one voids the run

1. The image builds, and `oracle` prints MATCH.
2. The source line says `embedded`, and every entry point prints its report.
3. The identity lines and the timing table (48 rows) are printed in full.
4. No `LLVQ_*` variable is in the job environment, except `LLVQ_KERNEL_DIR` on an iteration, whose source sha256
   the source line prints.

## 5. Signed predictions

| quantity | point | interval |
|---|---|---|
| identity, every variant, every (L, k, G) | **pass** | |
| local bytes (spill), V-64 at G = 4 and 5 | **0** | |
| registers, V-64, G = 4 | **110** | [80, 168] |
| registers, V-32, G = 5 | **180** | [120, 255] |
| η_k(8,192), V-64, G = 4 | **0.62** | [0.35, 0.85] |
| η_k(2,048), V-64, G = 4 | **0.45** | [0.20, 0.75] |
| r_8(8,192), V-64, G = 4 | **2.2** | [1.2, 4.5] |
| r_8(2,048), V-64, G = 4 | **2.5** | [1.2, 6.0] |
| η_k(8,192) at G = 5 over G = 4, V-64 | **0.90** | [0.70, 1.05] |
| t(V-32) / t(V-64) at 2,048, k = 1, G = 4 | **0.95** | [0.75, 1.15] |

Reasoning, *estimated*. FlashInfer reached 0.810 at 8,192 with 16 lanes of 8 dims, splits of at least 256 and
cp.async staging. Ours stages with plain 16-byte loads one step ahead in registers, computes each exp2 in all 16
lanes with a seven-operation polynomial, and runs every score through a 4-level shuffle tree: about 95 warp
instructions per (row, position, KV head) against FlashInfer's leaner path (`calc/attention/attention_v2.py`
§4), with 3.6 warps an SM at 8,192 (128 units x 8 KV heads / 2 groups a warp / 142 SMs). Hence a centre below
FlashInfer's and a wide interval. Verify: this version processes a verify call's rows one after the other in the
same CTA and re-stages the tiles for each row, so r_8 sits well above the lead's 1.0-1.7 for a mapping that
shares the stage; the interval keeps 1.2, for an L2 that serves the re-reads nearly free. The known flaws: the
IPC per warp is a judgment, and the model ignores the combine and the tickets.

## 6. Decision rules

Readings for the build, evaluated in this order. None is a gate, none pronounces a kill (METHODE §1).

1. **Identity.** A variant that fails one identity check has a bug: it is excluded from every reading below and
   fixed before the in-situ A/B. If V-64 fails, no timing of this run is read; fix, then a deviation file and a
   rerun on the operator's go.
2. **Spills.** V-64 with local bytes above 0 at G = 4 or 5: the served path refuses to load it
   (`attn_cuda::attn_kernel`), so a register revision through `LLVQ_KERNEL_DIR` comes before anything else.
3. **The unit size, then the freeze.** UNIT = 32 is adopted if V-32's median k = 1 time at 2,048 is at least 5 %
   below V-64's, at most 2 % above it at 8,192, and V-32 has no spill. Otherwise UNIT = 64. The canonical
   constants (TT, UNIT, SPLIT, the tree, the exp2 coefficients) are then frozen in one commit, before
   `prereg-attention-2` is stamped.
4. **The read rate.** η_k(8,192) of the chosen variant at G = 4 at least 0.40: the in-situ A/B proceeds. Below
   0.40: one revision of the memory pipeline (stage depth, cp.async, CTAs per SM, the reduce-scatter schedule of
   the same tree), all bit-neutral, through `LLVQ_KERNEL_DIR`, before the A/B; the operator is told the
   projection that follows, a proposal, the kill being his. Otherwise (no variant passed identity): rule 1.
5. **Verify.** r_8(8,192) of the chosen variant at most 1.5: the speculation lead's profile keeps its slope.
   Above 1.5: a mapping that shares the stage across rows (one group a row, the lead's CTA-per-split k > 1 form)
   is built before the speculation floor is costed, and the slope is recomputed with the measured r_k.
   Otherwise: no verify figure is read.
6. **The 14B's cores.** η_k(8,192) at G = 5 over G = 4, chosen variant, at least 0.85: the 14B plans CUDA cores.
   Below 0.85: the 14B's choice goes to its own in-situ step, with this ratio and the FlashInfer tensor-core arms
   as input. Otherwise: no 14B figure is read.

## 7. Deviations

Any departure from this file after the stamp goes in `proofs/preregistration-attn-bench-2026-11-04-ECARTS.md`.

## 8. Cost

*Estimated* on l40sx1 at $1.80/h. Container and image pull 3 min, oracle 2 min. Identity: 4 load-time checks
(18 cases each, replays in parallel threads) and 48 bench cases whose replays reach 8 rows x 40 heads x 8,192
positions, about 0.3 s each on the host side; seconds of card. Timing: 48 configurations x 9 rounds of passes of
1 to 10 ms, under 5 s. Uploads: 40 x 2 sets of 16.8 MB. In all 6 to 8 min, **$0.18 to $0.24**. Anchor:
`fi-decode-anchor` ran 69 s of card for $0.03 after 713 s of queue, unbilled (`docs/data/jobs.csv`). Timeout
25 min, **$0.75** at worst.

## 9. What this run does not establish

- The decode step in the model: candle's launches around the kernel, the cache append, the q and k norms and
  RoPE. The in-situ A/B measures them.
- Quality. The kernel is not bit for bit with the grouped form (cuBLAS's order, and today's f16 softmax); its
  licence is a T8 control: paired perplexity within ±0.12 % at 4,096 and 8,192 and a task check, the operator's
  decision, prepared and never granted here.
- The prologue parity probe of the draft (RoPE and rmsnorm forms against candle): it belongs to the operator's
  branch and needs candle on the card, which this binary does not link.
- Any other card, and the int8 store, left out by the operator's decision of 2026-11-04.

## 10. Changes from the draft `prereg-attention-2a`

| draft | here | why |
|---|---|---|
| variants V-A, V-B, V-C, V-B32 | V-64 and V-32, CTA per split, allreduce tree | Reading 2 of the FlashInfer anchor (F = 0.822 ≥ 0.80): CTA-per-split mapping only. The reduce-scatter schedule is a bit-neutral revision of rule 4 |
| `LLVQ_ATTN_BENCH` mode of `fusedrun` | `bin/attnbench` in `llvq-cuda` | no artifact or candle needed; the binary type-checks on the Mac, `fusedrun` does not |
| a `pos_dev` word on the card | `pos0` as a kernel argument | nothing captures a graph (A2 stays out of the core); the wide-grid identity still exercises CTAs with no work |
| prologue parity probe, step 5 | dropped | §9 |
| predictions | recentred | the first version re-stages the tiles for each verify row and runs the allreduce tree; §5 |
| rule 2 (mapping choice) | rule 2 (spills) | one mapping is built |
