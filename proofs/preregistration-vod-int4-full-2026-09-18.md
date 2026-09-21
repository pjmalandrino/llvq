# Preregistration. v + o + down at int4 g128, Qwen3-4B

**Written, committed and TIMESTAMPED on 2026-09-18, BEFORE the run.**
Operator go given 2026-09-18. Cost announced before the go: about 47 min on l40sx1, **$1.42**,
timeout capped at 2 h.

Not edited again. A fact it gets wrong goes beside it, in a `-ECARTS.md`.

## 1. What this asks, and why it is not a third confirmation

`o_proj` and `down_proj` are each confirmed on held-out questions, +1.55 and +4.05 pp. Neither
result says anything about the two together: the gains are not additive, and the record has no
measurement of an int4 allocation over more than one restored type on the full split.

This arm is the combination, and the question is whether it earns its bits. It is not a
selection from a menu: both components are already confirmed, so there is no winner's curse to
discount here.

## 2. The configuration, frozen now

Reference arm: `/out/tetra-q5-2026-09-09/qwen3-4b-tetra-q5.bin`, sha256
`ae31087a4b72494d52394f2cf2070da37a845c0a7d4aded65d5bdb0bd18b2263`, no restoration. `v_proj`
is already int4 in that file, which is why this arm is "the three" with only two restorations.

Treatment arm: the same file, `LLVQ_RESTORE_Q4=o_proj,down_proj`, `LLVQ_MODEL=Qwen/Qwen3-4B`.
Both: `LLVQ_MMLU_ALLOC=flat`, full split, CUDA, a dump per question.

The budget, *computed* on 3,633,315,840 projection weights with `Tetra` at 2.1498 and int4 g128
at 4.250, the b/param column anchored on the 2.8126 `rtbits` measured on the served file:

  arm             int4 weights   kernel b/w native   dequantized   b/param whole
  v, shipped         94,371,840       2.2044            2.5095        2.8126
  v + o + down    1,368,391,680       **2.9408**        7.3661        3.4778

2.9408 sits under b_max = 3.00 with **0.0592 b/weight of margin**, on the condition every
confirmation so far carries: a native int4 kernel for each shape. It has run on `v_proj`'s
1024 x 2560 and on neither 2560 x 4096 nor 9728 x 2560.

## 3. The primary result, named before the numbers

The paired gain against the shipped arm on the **full 14,042-question split**, stratified
bootstrap without finite population correction.

Held-out is not the primary here and the reason is stated now: nothing was selected by this
arm, both components were confirmed on held-out sets of their own, so the full split is the
right population and the 2,280 carry no special status. The held-out subset is reported anyway,
for comparability with the two earlier confirmations.

Secondary: the gain per kernel b/weight, against `down_proj` alone.

## 4. Signed prediction

| quantity | point | interval |
|---|---|---|
| paired gain against shipped, full split | **+5.15 pp** | [+4.0, +6.0] |
| MMLU micro | 61.52 | [60.4, 62.4] |
| gain per kernel b/weight | 6.99 | [5.4, 8.1] |

The point comes from a multiplicative model, stated before the run so it can be wrong in
public. FP16 reads 70.32 on our protocol, the shipped object 56.37, so the gap is 13.95 pp.
`o_proj` closes 11.1 % of it and `down_proj` 29.0 %. If the two close independent fractions,
together they close `1 - (1-0.111)(1-0.290)` = 36.9 %, that is **+5.15 pp**.

Named against me: above **+5.60**, the arithmetic sum, the two are super-additive, which would
need explaining. Below +4.05 they interact negatively and the combination is worse than
`down_proj` alone.

## 5. The decision rule

| Full-split result | Action |
|---|---|
| >= +4.5 pp, interval excludes zero | The combination holds. It fits at 2.9408 with 0.0592 of margin, and the operator decides on adoption |
| +4.05 to +4.5 | Sub-additive. `o_proj`'s extra 0.2182 b/weight buys under 0.45 pp, and the per-bit column decides whether it stays in |
| < +4.05 | The two interact negatively: adding `o_proj` to `down_proj` costs quality. A finding, and the allocation is `down_proj` alone |

The per-bit gate, computed now: `down_proj` alone returns 7.82 pp per kernel b/weight. For the
combination to beat it the gain must exceed **+5.76 pp**. Below that, the three types are worse
value per bit than `down_proj` alone even when the total gain is larger.

## 6. Controls

1. Identical run fingerprint on both arms, question by question.
2. The reference arm must again be byte-identical to the committed dump. It has now reproduced
   twice, 2026-09-17 and 2026-09-18, `cmp` clean. A third failure to reproduce voids the run.
3. 14,042 questions scored on both arms; any shortfall voids the run.
4. Both dumps kept whole and committed.
5. `oracle` first on the backend, hard rule 10.

## 7. What it will not establish

- Nothing about serving it. The quality is measured by dense reconstruction of dequantized int4
  tensors. No file carries either type as int4 records, and no kernel has run on either shape.
  Without native kernels the arm reads 7.3661 kernel b/weight, far over b_max.
- Nothing about a third or fourth type. `q_proj`, `k_proj`, `gate_proj` and `up_proj` are not
  in this arm, and their exploration figures carry no multiplicity correction.
- Nothing about perplexity, another model or another size.
- Nothing about throughput: two more int4 projection types mean two more kernel shapes, and
  served speed is not measured here.
