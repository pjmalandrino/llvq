DRAFT, not stamped, not committed

# Preregistration. The split-KV attention kernel alone: read rate, verify cost, mapping, registers

**Draft written on 2026-11-02, to be committed and TIMESTAMPED BEFORE the run.** Operator go to launch: **not given**.
It needs build steps 1 and 2 of attention.md §6: the CPU replay, the host probe of the kernel text, the CUDA kernel
with its launcher, and one measurement mode in `fusedrun`. It needs no model integration. Cost: about **$0.18 to
$0.24** a run on l40sx1, *estimated* below (§8), timeout 25 min, so $0.75 at worst. Cap asked: **$1**, which covers a
second run after one bit-neutral revision through `LLVQ_KERNEL_DIR`. Spent before it: $248.84 over 237 rows of the
registry, plus what `prereg-attention-1` spends.

## 1. Why this run exists

The lead's speed rests on two quantities no source gives for Ada: how fast its kernel reads the cache, and what a
verify call of k rows costs against a decode call. Both depend on choices that change no bit of the output: which
threads hold which heads, how the 16-lane score tree is scheduled, whether a CTA owns a split or a unit. A recount puts
the sketch's schedule at about 95 warp instructions per (row, position, KV head) and a reduce-scatter schedule of the
same tree at about 57 (*estimated*, `calc/attention/attention_v2.py`, section 4). With 2 to 7 warps per SM, that
difference decides whether k = 8 stays under the DRAM time.

This bench measures them on the card before any model integration, so a revision costs one mini-job through
`LLVQ_KERNEL_DIR` and no image rebuild. It also chooses one bit-relevant constant, the unit size, before the canonical
arithmetic is frozen: no quality control has run on any variant yet, so choosing among them by speed biases nothing.
It is a bench: it informs the build and gates nothing (METHODE §1). The in-situ A/B (`prereg-attention-2`) decides.

## 2. What the code holds, and what the Mac showed

To be filled at commit time with *measured* host facts: the CPU replay (`attn_ref`, the CPU fake of `AttnKernel`) and
its tests; the host probe (`llvq-llm/tests/host_attn_dec.cpp` with `attn_dec_host.rs`), which compiles the kernel text
with clang++ and diffs `ex2_fixed`, `dot8`, `tree16`, `tile_max_carry`, `tile_p`, `tile_accumulate`, `st_merge`,
`deq_r1` and `quant_codes` against the replay bit for bit; the mutants killed; the four entry-point variants of §3.

## 3. What runs

One image built from the commit that carries this file. One job, l40sx1. No artifact is loaded by the bench.
`ops/jobs/attn-bench.sh`.

1. **Oracle** on Qwen3-0.6B, rule 10.
2. **Load report.** For every entry point: `num_regs`, `local_bytes`, shared bytes and binary version, read from the
   driver (`llvq-cuda/src/gpu.rs`, `report`).
3. **Identity, every variant.** For each L in {2,048, 4,096, 8,192}, k in {1, 2, 4, 8} and G in {4, 5}: set 0's output
   against the CPU replay, bit for bit; the k rows of one verify call against k separate decode calls, bit for bit; a
   grid of `s_max` CTAs against a grid of `ceil((pos0 + k) / 256)` CTAs, bit for bit, which exercises the early exits
   that still draw their ticket.
4. **Timing.** `LLVQ_ATTN_BENCH=2048,4096,8192`, a measurement mode refused beside `LLVQ_CONFIG`. G = 4: 36 synthetic
   per-layer KV sets of the 4B geometry (8 KV heads, 128, f16, a fixed pseudo-random fill), 32 query heads. G = 5:
   40 sets, 40 query heads, the 14B's geometry. One pass reads 36 (or 40) × 4,096 × L bytes and reuses no set, above
   the 96 MB L2 at every length. A pass is 36 (or 40) eager launches on one stream, timed with CUDA events. 2 discarded
   and 7 timed rounds; inside a round, every (variant, L, k, G) runs in one fixed order. Variants:
   - **V-A**: CTA per split, allreduce schedule (the sketch, `code/attention/attn_dec_sketch.cu`);
   - **V-B**: CTA per split, reduce-scatter schedule;
   - **V-C**: CTA per unit, reduce-scatter, two-level tickets (per split, then per head);
   - **V-B32**: V-B with a unit of 32 positions, a different canonical arithmetic with its own CPU replay.
5. **Prologue parity probe**, for the operator's branch if it has not settled the question. RoPE in f16: for each of
   the two outputs, three forms (no fma, fma on the first product, fma on the second) against candle's `rope` on the
   card. rmsnorm on 128 columns: two forms (sum of squares by fma, by multiply then add) against candle's `rms_norm`.
   On 4,096 random rows, normal, plus rows holding subnormal and large values. It prints which form matches bit for bit.

Derived quantities, *computed*: η_k(L) = 36 × 4,096 × L / (t × 864 GB/s) at k = 1 (40 sets at G = 5); r_k(L) = t(k) /
t(1), formed round by round; medians with ranges (rule 7).

## 4. Gates, each one voids the run

1. The image builds, and `oracle` prints MATCH.
2. Every entry point loads and prints its report.
3. The identity table of step 3 and the timing table of step 4 are printed in full for every variant.
4. No `LLVQ_*` variable is in the job environment except `LLVQ_ATTN_BENCH`, and `LLVQ_KERNEL_DIR` on an iteration, whose
   source sha256 the loader prints.

## 5. Signed predictions

| quantity | point | interval |
|---|---|---|
| identity, every variant, every (L, k, G) | **pass** | |
| η_k(8,192), V-B, G = 4 | **0.70** | [0.45, 0.90] |
| η_k(2,048), V-B, G = 4 | **0.45** | [0.20, 0.75] |
| η_k(8,192), V-A / V-B, G = 4 | **0.95** | [0.75, 1.02] |
| r_8(8,192), V-A / V-B / V-C, G = 4 | **1.3 / 1.05 / 1.05** | [1.0, 1.9] / [1.0, 1.4] / [1.0, 1.3] |
| r_8(2,048), V-A / V-B / V-C, G = 4 | **3.0 / 2.0 / 1.2** | [1.5, 6.5] / [1.2, 3.6] / [1.0, 1.7] |
| η_k(8,192) at G = 5 over G = 4, V-B | **0.95** | [0.75, 1.02] |
| t(V-B32) / t(V-B) at 2,048, k = 1 | **0.92** | [0.75, 1.05] |
| registers, k = 1 path, G = 4 | **130** | [100, 200] |
| local bytes (spills), k = 1 path, G = 4 | **0** | |
| prologue: a RoPE form matches for each output, an rmsnorm form matches | **yes** | |

Reasoning, *estimated* (`calc/attention/attention_v2.py`). At k = 1 every unit runs in parallel, so the compute time of
a layer is one unit's chain: about 11 to 19 µs with V-A and 6 to 12 µs with V-B, against 46 to 65 µs of DRAM time at
8,192 and 11 to 16 µs at 2,048. At k = 8 a CTA per split puts four units in sequence on 2 to 7 warps per SM: 43 to
77 µs with V-A, 23 to 41 µs with V-B. A CTA per unit spreads the same work over 7 to 29 warps per SM. The known flaws:
the IPC per warp, 0.25 to 0.45, is a judgment, and the model ignores the combine and the tickets.

## 6. Decision rules

Readings for the build, evaluated in this order; none is a gate, none pronounces a kill (METHODE §1).

1. **Identity.** A variant that fails one identity check has a bug: it is excluded from every reading below and fixed
   before step 3. If every variant fails, no timing of this run is read; fix, then a deviation file and a rerun on the
   operator's go.
2. **The mapping for step 3.** Among the variants that pass, the one with the lowest sum over the three lengths of its
   median k = 1 pass time at G = 4. If that variant's r_8(2,048) exceeds 2.0 while another passing variant's is at
   most 1.5 with a k = 1 sum within 5 %, that other variant is taken. Otherwise the first choice stands.
3. **The unit size, then the freeze.** UNIT = 32 is adopted if V-B32's median k = 1 time at 2,048 is at least 5 % below
   V-B's and at most 2 % above it at 8,192. Otherwise UNIT = 64. The canonical constants (TT, UNIT, SPLIT, the tree,
   the exp2 coefficients) are then frozen in one commit, before `prereg-attention-2` is stamped.
4. **The read rate.** η_k(8,192) of the chosen variant at least 0.40: step 3 proceeds. Below 0.40: one revision of the
   memory pipeline (stage size and depth, CTAs per SM) through `LLVQ_KERNEL_DIR` before step 3, and the operator is
   told the projection that follows; that is a proposal, the kill is his. Otherwise (no variant passed identity): rule
   1 applies.
5. **Verify.** r_8(8,192) of the chosen variant at most 1.5: the speculation lead's P3 profile keeps its slope. Above
   1.5: that slope is recomputed with the measured r_k before the speculation floor is costed. Otherwise: no verify
   figure is read.
6. **The 14B's cores (an input to V9).** η_k(8,192) at G = 5 over G = 4, chosen variant, at least 0.85: the 14B plans
   CUDA cores. Below 0.85: the 14B's choice goes to V9 with this ratio and the FlashInfer tensor-core arms. Otherwise:
   no 14B figure is read.
7. **The prologue, for the branch.** One form matches candle for each RoPE output and one for rmsnorm: the branch's
   fused prologue can keep candle's bits on this card and driver, checked again at every load. No form matches: the
   branch keeps candle's ops or takes its fusion through a T8 control. Otherwise: not read.

## 7. Deviations

Any departure from this file goes in `proofs/preregistration-attn-bench-<date>-ECARTS.md`, beside it.

## 8. Cost

*Estimated* on l40sx1 at $1.80/h. Container and image pull 3 min, oracle 2 min. Identity: one CPU replay per (L, k,
G) and unit size, 48 in all, 1 to 2 s each on the host, then every variant compared against its own. Timing: 96 configurations, 9 rounds
of passes of 1 to 4 ms. The probe, seconds. In all 6 to 8 min, **$0.18 to $0.24** a run. Anchors: `gqa-ab` ran three
arms at four lengths in 575 s for $0.29 (`docs/data/jobs.csv`). Timeout 25 min, **$0.75** at worst.

## 9. What this run does not establish

- The decode step in the model: launches around the kernel, the prologue, the combine under the real stream. The
  in-situ A/B (`prereg-attention-2`) measures them.
- Quality: the kernel is not bit for bit with cuBLAS; its T8 control is V6.
- The int8 store, built in step 6.
- Any other card.
