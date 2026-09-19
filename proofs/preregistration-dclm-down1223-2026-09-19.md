# Preregistration. 60 MMLU under 3 b/param, or how much the two levers substitute

**Written, committed and TIMESTAMPED on 2026-09-19, BEFORE the run.**
Operator go given 2026-09-19. Cost: about 23 min on l40sx1, **$0.72**, timeout 1 h.

## 1. The claim on trial

  base, DCLM-calibrated Q5      57.95 micro   2.2044 kernel b/weight   2.8126 b/param
  **+ `down_proj@12-23`**       **?**         **2.3771**               **2.9686**

If it reaches 60.0 this is **60 MMLU under 3 b/param**, with one slice of twelve layers added
to a base that costs nothing extra to calibrate.

The two ingredients are measured, each on its own: the base on 14,042 questions, and the slice
at **+2.41 pp** [+1.79; +3.00] on the C4 base. This arm is not their sum, it is the test of
whether they add.

## 2. Why the sum is an upper bound and not a prediction

`LLVQ_RESTORE_Q4` round-trips the checkpoint tensor through the affine quantizer. **It never
reads a Hessian.** So the int4 arm is calibration-independent, and the only coupling between
the two levers is through the baseline: whatever error a better calibration already removed
from `down_proj` is error int4 no longer has to remove.

And `down_proj` is exactly where a corpus change should bite hardest. At 131,072 tokens it
carries **13.5 samples a dimension** against 51 for q, k, gate and up, the only one of the four
activations short of samples.

So **+2.41 is a ceiling**. The measurement is how far under it we land, and that number is the
substitution rate between calibration and bits.

## 3. Signed prediction

| quantity | point | interval |
|---|---|---|
| gain over the base | **+1.8 pp** | [+0.9, +2.6] |
| micro, full split | 59.75 | [58.85, 60.55] |
| substitution, 1 minus gain over 2.41 | 25 % | [−8 %, 63 %] |

The point takes three quarters of the slice's measured gain, on the reading that the corpus
already recovered a quarter of what `down_proj` was losing.

Named against me: **+2.41 or above** means the two levers do not substitute at all, which the
mechanism above says they must, and would need explaining. This arm is a **constant-file**
comparison, 0.43 pp of noise, so unlike the re-encoding arms of this day it can actually
resolve a point.

## 4. The decision rule

| gain | reading | next |
|---|---|---|
| >= +2.05 | **60 under 3 b/param is reached.** | seal the object: re-encode with `LLVQ_INT4_TYPES=v_proj,down_proj@12-23` so a real file carries the codes |
| +0.9 to +2.05 | Short of 60, and the substitution rate is the result | decide between adding `o_proj` at +0.2182 and stopping |
| < +0.9 | Strong substitution: the calibration already did most of what int4 was buying | the bits axis closes on this base, and that is worth more than the point |

## 5. Controls

1. Constant file: both arms are the same sealed object, `471f3988`, one restored at load.
2. The arm's log must declare **12 matrices, 298,844,160 weights**.
3. 14,042 questions, plan fingerprint `a74a6d6213602979`.
4. `oracle` first, hard rule 10.

## 6. What it will not establish

- Nothing about serving it: the quality is a dense reconstruction of a dequantized int4 tensor,
  and no kernel has run on 9728 x 2560. The 2.3771 assumes that kernel exists.
- Nothing about whether DCLM beats C4, which one prefix each cannot settle.
- Nothing about the other slices or the other types on this base.
