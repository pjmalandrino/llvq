# Preregistration — confirming o_proj at int4 g128, Qwen3-4B

**Written, committed and TIMESTAMPED on 2026-09-16, BEFORE the confirmation run.**
Operator go given 2026-09-16.

🚨 **Not edited again.** A fact it gets wrong goes beside it, in a `-ECARTS.md`.

## 1. What selected this arm, and why that is not the result

An exploration of six types, each restored at int4 g128 on the served Q5 file,
scored on 2,280 questions (`docs/mesures/q5-alloc-int4-2026-09-16.txt`). Its
table carries **individual 95 % intervals with no correction for the six
comparisons**, so it selects a candidate and establishes nothing. `o_proj` was
picked on two grounds: a stratified Δ of +2.77 pp whose interval excluded zero,
and a surcharge of +0.1954 b/param that keeps the model under `b_max` = 3.00
(2.7645 → 2.9599).

`down_proj` measured higher, +3.79 pp, and is not chosen here because +0.4641
b/param puts the model at 3.23 and over budget. It is not dropped: restoring a
SUBSET of its layers is a later lead, and the gain is not assumed proportional
to the count of layers restored.

## 2. The configuration, frozen now

Reference arm: `/out/tetra-q5-2026-09-09/qwen3-4b-tetra-q5.bin`, no restoration.
Treatment arm: the same file, `LLVQ_RESTORE_Q4=o_proj`, `LLVQ_MODEL=Qwen/Qwen3-4B`.
Both: `LLVQ_MMLU_ALLOC=flat`, full split (no limit), CUDA, dump per question.

Nothing else moves. The reference artifact is byte-identical to the exploration's.

## 3. The primary result, named before the numbers

**The paired gain on the 11,762 questions that did NOT take part in the
selection**, with its interval. The 2,280 that did are reported separately and
are not the result.

The full 14,042 figure is published too, and it is stated on the same line that
it CONTAINS the 2,280 selection questions and is therefore not independent.

A nuance recorded now: those 11,762 are out-of-selection for this experiment,
not virgin in absolute terms — a full MMLU ran on 2026-09-11 and again on
2026-09-16 for another lead. No per-question answer from those runs entered the
choice of `o_proj`, which is what makes this confirmation meaningful.

## 4. Signed prediction

**+1.2 to +2.5 pp on the 11,762 held-out questions**, interval excluding zero.

Below the exploration's +2.77 on purpose: a bar selected as the best of six is
biased upward, and regression toward the mean is expected rather than feared.

Named against me: a point estimate at or above +2.77 would mean the selection
bias did not operate, which on six arms would itself need explaining. Below
+1.2, or an interval containing zero, and the candidate is not confirmed.

## 5. The decision rule

| Held-out result | Action |
|---|---|
| ≥ +1.2 pp, interval excludes zero | `o_proj` at int4 is a real gain within budget; the operator decides on adoption |
| positive, interval contains zero | Not confirmed. Report and stop; a third run on this lead needs its own case |
| ≤ 0 | The exploration selected noise. Record it against the six-arm table |

## 6. Controls

1. Identical run fingerprint on both arms, question by question.
2. The reference arm reproduces the exploration's shipped arm on the 2,280
   shared questions, pick for pick.
3. 14,042 questions scored on both arms; any shortfall voids the run.
4. Both dumps kept whole.

## 7. What it will not establish

Nothing about perplexity, which is not measured here. Nothing about other
allocations — `down_proj`, subsets of layers, combinations — which interact and
are not additive. Nothing about another model or another size. And no adoption:
the served encoder is an operator decision on a fundamental criterion.
