# Preregistration. Bare Tetra on the full split, the baseline of every "without int4" claim

**Written, committed and TIMESTAMPED on 2026-09-18, BEFORE the run.**
Operator go given 2026-09-18. Cost announced before the go: about 24 min on l40sx1, **$0.72**,
timeout capped at 1 h.

Not edited again. A fact it gets wrong goes beside it, in a `-ECARTS.md`.

## 1. Why it is owed, and it is the cheapest hole in the record

Bare `Tetra` at the 4B has **never been scored on the 14,042 questions**. Its only figure is
53.49 on the 2,280 sample (2026-09-06).

That figure is the baseline of the claim that matters most: reaching the paper's quality at the
paper's rate, 2.1498 kernel b/weight, **without int4 on any type**. Every arm measured on
2026-09-18 buys quality with bits; this one is the point they are bought from, and it is
currently an estimate.

Worse, the sample-to-full shift has no stable sign: it was +0.85 pp on the served object, −0.18
on f16 and +2.24 on the 8B. So a baseline read off the sample carries about two points of
uncertainty in an unknown direction.

## 2. The configuration

One arm: `/out/tetra-4b-2026-09-06/qwen3-4b-tetra.bin`, no restoration, 252 lattice records,
`LLVQ_MMLU_ALLOC=flat`, full split, CUDA, a dump per question. Nothing is compared inside the
job; the dump carries plan fingerprint `a74a6d6213602979` and pairs afterwards against every
committed census dump.

## 3. The primary result

The MMLU micro accuracy of bare `Tetra` at the 4B on the full split, with its macro beside it.

## 4. Signed prediction

| quantity | point | interval |
|---|---|---|
| micro, full split | **53.5** | [52.1, 54.8] |
| macro, full split | 55.6 | [54.0, 57.2] |

The point is the population-reweighted estimate of the committed 2,280 dump, which reproduces
the published 53.49 to the hundredth. The same method predicted f16 at 70.3 against a measured
70.14, and the served object at 55.52 against a measured 56.37. Its two errors so far are
−0.18 and +0.85.

The interval is the repository's flat-plan sampling error of 1.34.

Named against me: above 54.8 the sample under-estimates bare `Tetra` as it under-estimated the
8B, and the "without int4" baseline is better than the record says. Below 52.1 it
over-estimates, and every gap quoted against 53.49 is too small.

## 5. What it changes downstream

It is a denominator, and three live questions rest on it.

1. The gap to the paper's 60.7 **at comparable rate**, quoted today as 5.1 to 7.2 points
   depending on which of our numbers is used.
2. What the calibration has to deliver to reach 60 without int4: the arithmetic is
   `60.0 minus this number`.
3. The value of the dclm arms. The corpus gave +1.58 pp on the Q5 recipe, and whether that
   transfers to bare `Tetra` is a different question measured against this baseline.

## 6. Controls

1. 14,042 questions, plan fingerprint `a74a6d6213602979`.
2. The file is the one whose projections `rtbits` measured at 2.1498 kernel b/weight and
   2.7645 b/param, sha256 `0adb7cfd02ed7402`.
3. The dump kept whole and committed.
4. `oracle` first on the backend, hard rule 10.

## 7. What it will not establish

- Nothing about any calibration variant: this is the C4-calibrated encoding of 2026-09-06.
- Nothing about int4, which is the point: this arm has none.
- Nothing about perplexity, another model or another size.
