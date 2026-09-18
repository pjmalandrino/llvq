# Preregistration. Is down_proj's gain concentrated by depth

**Written, committed and TIMESTAMPED on 2026-09-18, BEFORE the run.**
Operator go given 2026-09-18. Cost announced before the go: about 72 min on l40sx1, **$2.16**,
timeout capped at 2 h, worst case $3.60.

Not edited again. A fact it gets wrong goes beside it, in a `-ECARTS.md`.

## 1. The question, and why it is not a fourth confirmation

`down_proj` at int4 is confirmed at +4.05 pp for +0.5182 kernel b/weight, 36 matrices taken
because they share a name. An allocation is a knapsack over matrices, and nothing says the 31st
layer is worth what the 3rd is.

The free route to that ranking is closed. Two proxies were built and both failed on 2026-09-18:
the absolute Hessian-weighted error ranks by the calibration samples per dimension (r = +0.988
with it), and the scale-free ratio does not predict the measured gains (r = +0.571 on six
points, four of them exploratory) and is not a currency a knapsack can add.

So the benefit gets measured, and the only affordable grain is depth.

## 2. The configuration, frozen now

Three arms of the served Q5 file, `/out/tetra-q5-2026-09-09/qwen3-4b-tetra-q5.bin`, sha256
`ae31087a...2263`:

  A  `LLVQ_RESTORE_Q4=down_proj@0-11`
  B  `LLVQ_RESTORE_Q4=down_proj@12-23`
  C  `LLVQ_RESTORE_Q4=down_proj@24-35`

Full split, `LLVQ_MMLU_ALLOC=flat`, CUDA, a dump per question. Each slice is 12 matrices,
298,844,160 weights, **+0.17273 kernel b/weight**, one third of the full type's +0.5182.

**No witness arm, and the reason is the primary.** The question is which slice carries the
gain, which is a comparison between the three and not against the shipped file. The shipped
dump is committed and has reproduced **byte for byte across three jobs** (2026-09-17,
2026-09-18 twice, `cmp` clean each time), so the gains against it are read from it and the
$0.72 of a fourth arm buys nothing the primary needs.

## 3. The primary result, named before the numbers

The three paired gains against the committed shipped dump, full split, stratified bootstrap
without finite population correction, **and their spread**. The derived quantity that decides
the lead is the best slice's return per kernel b/weight.

## 4. Signed prediction

| quantity | point | interval |
|---|---|---|
| sum of the three gains | **+3.9 pp** | [+3.3, +4.6] |
| best slice | +1.9 pp | [+1.4, +3.0] |
| best slice divided by worst slice | 2.0 | [1.0, 5.0] |

The sum comes from additivity measured at 96 % on 2026-09-18 applied to the type's +4.05. The
spread has **no prior**: the repository holds no depth attribution for `down_proj`, and the
interval is wide because inventing a narrow one would be dishonest rather than bold.

Named against me: a sum above +4.6 would mean the slices are super-additive against a type that
contains them, which has no mechanism I can state.

## 5. The decision rule

`down_proj` entire returns **7.82 pp per kernel b/weight**. A slice costs 0.17273, so a slice
returning more than +1.35 pp already beats the full type per bit. That bar is nearly automatic
once the gains are unequal, so it is not the gate.

| best slice | Action |
|---|---|
| **>= +2.02 pp** (11.7 pp per b/weight, 1.5 times the full type) | The gain is concentrated. The subset lead is funded, and the next step is a finer grain on the winning slice |
| +1.35 to +2.02 | Concentrated, but not enough to pay for a finer search. Record the ranking, take the best slice if the budget is tight, and stop |
| < +1.35 with a spread under 1.5 | **The gain is uniform in depth.** The subset lead dies for `down_proj`, for $2.16, and the knapsack loses its cheapest dimension |

## 6. Controls

1. The three arms share one job, one card, one run fingerprint.
2. Each arm's restore count is read from its own log: 12 matrices, 298,844,160 weights. An arm
   that restored 36 or 0 voids itself, and the loader refuses a window that matches nothing.
3. 14,042 questions scored on all three arms.
4. The three dumps kept whole and committed.
5. `oracle` first on the backend, hard rule 10.
6. The layer window is new code, landed this day with five tests and five dead mutants
   (a backwards window, both exclusive bounds, a window ignored, an unreadable layer covered).

## 7. What it will not establish

- Nothing about serving any of it: dense reconstruction of dequantized int4, and no kernel has
  run on 9728 x 2560.
- Nothing about other types. `o_proj`, `gate_proj` and the rest may concentrate differently, and
  this run says nothing about them.
- Nothing about a grain finer than 12 layers, nor about non-contiguous subsets, which is the
  whole knapsack and not this arm.
- Nothing about perplexity, another model or another size.
