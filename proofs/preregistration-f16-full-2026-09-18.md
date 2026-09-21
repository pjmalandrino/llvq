# Preregistration. The f16 reference on the full MMLU split, Qwen3-4B

**Written, committed and TIMESTAMPED on 2026-09-18, BEFORE the run.**
Operator go given 2026-09-18. Cost announced before the go: about 24 min on l40sx1, **$0.72**,
timeout capped at 1 h.

Not edited again. A fact it gets wrong goes beside it, in a `-ECARTS.md`.

## 1. Why this is owed

Every statement of the form "we are N points below f16" in this repository divides a full-split
number by a sampled one. **No f16 has ever been scored on the 14,042 questions.** The 70.32 in
circulation is a 2,280-question estimate, and the census of 2026-09-11 showed that such an
estimate can sit 0.85 pp from the full value on the served object.

That mixing has already reached a signed prediction: the `v + o + down` prereg of today builds
its point estimate on a gap of 13.95 pp computed as 70.32 minus 56.37, two different
populations. This run removes the excuse.

## 2. The configuration, frozen now

One arm: `mmlu Qwen/Qwen3-4B cuda`, no limit, `LLVQ_MMLU_ALLOC=flat`, dtype f16, KV f16, dump
per question. The checkpoint is the pinned `1cfa9a7208912126459214e8b04321603b3df60c`, the same
revision every other arm reads.

One arm and not two: nothing is compared inside this job. The dump carries the plan fingerprint
`a74a6d6213602979`, which is what makes it pairable afterwards against the committed
shipped, `o_proj`, `down_proj` and `v + o + down` dumps.

## 3. The primary result, named before the numbers

The MMLU micro accuracy of Qwen3-4B in f16 on the full 14,042-question split, with its macro
beside it.

## 4. Signed prediction

| quantity | point | interval |
|---|---|---|
| f16 micro, full split | **70.3** | [69.0, 71.6] |
| f16 macro, full split | 71.5 | [69.5, 73.5] |

The point is the population-reweighted estimate of the existing 2,280 dump, recomputed today:
70.32. That recomputation reproduces the published 70.32 to the hundredth, and reproduces the
served object's published 55.52 to the hundredth as well, which is what licenses using it here.

The interval is the repository's own sampling error for a flat 2,280 plan, plus or minus 1.34.
An independent stratified variance computed today gives plus or minus 2.6; the narrower bar is
registered on purpose, because it is the more falsifiable of the two.

Named against me: the served object's sample sat 0.85 pp **below** its full value. If that
shift is a property of the plan rather than of the model, f16 should land near 71.2 rather than
70.3, and a result above 71.6 would say the sample under-estimates systematically and every
published gap in this repository is too small.

## 5. What it changes downstream

Nothing is decided by this number, and that is the point: it is a denominator. Once measured,
the gap of every quantized arm to f16 is a same-population quantity, and the `v + o + down`
prereg's 13.95 gets recomputed in that prereg's deviations.

## 6. Controls

1. The dump carries 14,042 questions and the plan fingerprint `a74a6d6213602979`.
2. The paper's Table 6 gives FP16 70.2 on this model. A result far from it is a protocol
   failure, not a model fact, and the binary says so in its own output.
3. `oracle` first on the backend, hard rule 10.
4. The dump is committed whole.

## 7. What it will not establish

- Nothing about any quantized arm on its own. This is one number.
- Nothing about another model, another size or another metric.
- Nothing about perplexity.
