# Preregistration. Does the int4 allocation transpose to the 8B

**Written, committed and TIMESTAMPED on 2026-09-18, BEFORE the run.**
Operator go given 2026-09-18. Cost announced before the go: about 95 min on l40sx1, **$2.85**,
timeout capped at 3 h, worst case $5.40.

Not edited again. A fact it gets wrong goes beside it, in a `-ECARTS.md`.

## 1. Why this is cheap, and why it is not obvious

The allocation needs no re-encoding: `LLVQ_RESTORE_Q4` reads the checkpoint and replaces the
matrices at load, on any sealed file. The `Tetra` 8B exists in the bucket
(`tetra-8b-2026-09-06/qwen3-8b-tetra.bin`, 4,324,244,913 bytes), so the 4B's best allocation
transposes for the price of two MMLU arms.

What is not obvious is whether it works. The 8B's own journal of 2026-09-06 is titled "the
memory holds, the quality does not": `Tetra` there reads **61.61** micro against `Planes14`'s
65.52, a loss of 3.91 pp, where the 4B lost 2.10. The format degrades with size, so the arm
could gain more (more error to remove) or less (a format in trouble does not get rescued by
four bits on three types).

The budget transposes almost exactly (*computed*, 6,945,767,424 projection weights, `Tetra`
2.1498, int4 g128 4.250):

  Tetra nu       2.1498   margin 0.8502
  v              2.1955   margin 0.8045
  v + o + down   **2.9260**   margin **0.0740**

against the 4B's 2.9408 and 0.0592.

## 2. The configuration, frozen now

Reference arm: `/out/tetra-8b-2026-09-06/qwen3-8b-tetra.bin`, no restoration. Bare `Tetra`,
252 lattice records, **v_proj is NOT int4 here**, unlike the served 4B.

Treatment arm: the same file, `LLVQ_RESTORE_Q4=v_proj,o_proj,down_proj`,
`LLVQ_MODEL=Qwen/Qwen3-8B`. Three types and not two, because the 8B has no Q5 mix to start
from: the v_proj step is inside this arm rather than in the baseline.

Both: `LLVQ_MMLU_ALLOC=flat`, full 14,042-question split, CUDA, a dump per question. No 8B has
ever been scored on the full split, so both arms are firsts and the witness cannot be
inherited.

## 3. The primary result, named before the numbers

The paired gain of the treatment over the reference, full split, stratified bootstrap without
finite population correction.

## 4. Signed prediction

| quantity | point | interval |
|---|---|---|
| paired gain, three types at int4 | **+7.0 pp** | [+3.0, +11.0] |
| reference arm, full split | 62.5 | [60.5, 64.5] |

The point transposes the 4B's chain: +3.47 for `v_proj` and +5.39 for the two others, 8.86
arithmetic, cut to 7.0 for sub-additivity and for the 8B's heavier FFN share. The interval is
wide and says so: the format's degradation with size has no measured direction on this lever.

The reference prediction is the 8B `Tetra`'s 61.61 sample micro plus the shift the 4B showed
from sample to full (+0.85), which is one precedent and not a law.

## 5. The decision rule

| Paired gain | Reading |
|---|---|
| >= +5.0 pp, interval excludes zero | The allocation transposes. The 8B becomes a candidate on the quality axis and the surcharge is 0.7762 kernel b/weight for it |
| +2.0 to +5.0 | It transposes weakly. The 4B was flattering, and the per-bit comparison against the 4B's 7.32 pp per kernel b/weight goes in the journal |
| < +2.0 | It does not transpose. The 4B result is a 4B result, and no claim about scale follows from it |

## 6. Controls

1. Both arms share one job, one card, one run fingerprint.
2. The treatment arm's log states its restore count: 108 matrices. An arm restoring a different
   count voids itself.
3. 14,042 questions scored on both arms.
4. Both dumps kept whole and committed. They are the first 8B dumps on the census plan.
5. `oracle` first on the backend, hard rule 10.

## 7. What it will not establish

- Nothing about serving it: dense reconstruction of dequantized int4, and no kernel has run on
  any 8B shape.
- Nothing about the 14B, which exists only as `Planes14` and has no `Tetra` encoding.
- Nothing about perplexity, which is not measured here.
- Nothing about the 4B's own numbers, which stand on their own job.
