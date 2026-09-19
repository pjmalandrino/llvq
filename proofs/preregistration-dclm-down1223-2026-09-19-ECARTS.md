# Deviations from the dclm-down1223 prereg (2026-09-19)

The prereg `preregistration-dclm-down1223-2026-09-19.md` is timestamped and is not edited.

## E1. The bar is stated in the wrong accounting, and the closing verdict is void

The prereg is titled "60 MMLU under 3 b/param" and repeats that at lines 11 and 52. The
product triplet's bar is **b_max = 3.00 kernel b/weight** (`docs/ETAT.md:533`), not b/param.
`docs/ROADMAP-QUALITY.md:28` states the rule outright: "Mixing the two is the one subtraction
to refuse."

The journal then wrote "0.0314 b/param remain under the triplet's 3.00" and closed the axis on
it. That subtraction spans two accountings. Redone in the triplet's own unit:

  arm measured               2.3771 kernel b/weight
  margin before b_max        **0.6229**, twenty times the figure the journal used
  one down_proj layer        0.0144 kernel b/weight

So the two arms the journal killed as "over budget" are both inside it:

  + down_proj full           2.7226 kernel b/weight, under 3.00
  + o_proj                   2.5953 kernel b/weight, under 3.00
  + v + o + down             2.9408 kernel b/weight, under 3.00

**"No stack of int4 slices reaches 60 under b_max" is false**, and it was false when written:
`docs/avancement-diagnostic-tetra.md` already carried `+ down_proj` at 60.44 and `+ o + down`
at 61.76, both measured on 2026-09-18, the day before.

This is the same unit error `docs/etat-reconcilie-2026-09-17.md` was written to repair, by the
same author, two days earlier.

## E2. What actually blocks the axis, which is not arithmetic

Both 60+ arms are dense reconstructions of dequantized int4 tensors. Their own journals say so:

> "No kernel has run on 9728 x 2560. The 2.7226 assumes that kernel; without it the arm reads
> 5.9271 and is far over b_max." (`downproj-int4-full-2026-09-18.txt`)

> "Without native kernels the arm reads 7.3661 kernel b/weight." (`vod-int4-full-2026-09-18.txt`)

No file carries those types as int4 records. So the axis is not closed by the budget, it is
**blocked by kernels that do not exist**. That is a different fact and it calls for different
work: write the kernels, or drop the axis on a stated ground rather than an arithmetic one.

## E3. The sealing instruction of the decision table cannot be executed

Line 52 orders, on a result >= +2.05: "re-encode with `LLVQ_INT4_TYPES=v_proj,down_proj@12-23`".

`smoke.rs:707-721` validates every token of that variable against `sealed::PROJ_TYPES`, a flat
list of projection names. `down_proj@12-23` is not one of them and is refused by name. The
layer-window grammar added on 2026-09-19 went into `LLVQ_RESTORE_F16` and `LLVQ_RESTORE_Q4`
(`sealed.rs`), not into `LLVQ_INT4_TYPES`.

The arm can be measured and cannot be built. The gate was never reached, so nothing was sealed
on a broken instruction, but the instruction would not have run.

## E4. The result landed in a band whose verdict was not the one written

Measured +1.38 pp, 59.33 micro. The decision table's bands are at >= +2.05, and below. The
journal wrote a closure of the whole axis, which no band authorises.

## How these were found

By the audit workflow of 2026-09-19, 189 agents over ten lenses with adversarial refutation.
E1 survived three refuters unanimously. Not by the author, who had written the repair two days
earlier and repeated the error anyway.
