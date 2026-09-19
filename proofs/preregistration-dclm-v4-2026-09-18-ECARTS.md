# Deviations. Calibration volume prereg of 2026-09-18

The prereg is stamped (`1c8a94a3ea452b25ec8007d5...`, four calendars) and is not edited.

The protocol ran as section 2 froze it and the primary landed inside its interval, on the
negative side. Three deviations, none of which touches the primary's definition.

## E1. The card, and it cost three and a half times what was announced

Section 0 announced about 18 min on rtx-pro-6000 for **$0.83**. Billed: 55 min, **$2.51**.

The l40sx1 pool was saturated all day, so the arm was moved to rtx-pro-6000 under
`--any-flavor`. That card bills 55 min for the same 14,042-question MMLU the l40sx1 does in 23,
at 1.5 times the hourly rate. The reasoning that a faster card at a higher rate cancels out is
false for this workload: candle's CUDA path is not tuned for compute cap 120.

Two arms of 2026-09-18 paid it, this one and `tetra-nu-full`, for **$3.36 of overrun**. The rule
that follows: waiting for an l40sx1 beats starting at once on an rtx, for anything that scores
the full split.

## E2. The mechanism was named and is not tested

Section 2 predicted the effect would pass through `down_proj` alone, and gave the samples per
dimension to say why. The arm scores whole files and separates no type, so it neither confirms
nor refutes that. The journal says so rather than reading the sign of the whole as a verdict on
the mechanism.

## E3. A phase anomaly, recorded and not explained

`advance (pass 2)` took 0.7 s at one times the volume and 1,120 s at four times. A factor of
1,600 for a factor of 4 is not accounted for by the volume. Nothing in this run diagnoses it,
and it is written here so a later reader does not take the phase profile of the x1 encoding as
a model of the x4 one.
