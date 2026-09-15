# Deviations from the errmap prereg, 2026-09-15

## Prediction 1 is refuted

Predicted a majority of concave or flat directions; measured 13 of 196, 6.6 %
(*measured*, [journal](../docs/mesures/errmap-0.6b-2026-09-15.txt)). The
two-block pilot that suggested it ran on 8 calibration windows against 64 and
was a pathological regime. The trust region stays in the code — it is what keeps
the map defined on the 13 — but it is a guard rail, not the load-bearing part
the prereg thought it was.

## Prediction 2 holds, and its wording was wrong

q_proj and k_proj are not *exactly* zero: they are four orders of magnitude
below every other type, at the level of the f32 evaluation's own noise divided
by 2ε. The invariance is exact and the measurement of it cannot be. The prereg
asserted a property of the arithmetic where it meant a property of the model.

## Prediction 4 holds more strongly than stated

It predicted a difference from 1.02. The pooled post-hoc scale is 0.99456, which
differs in sign and not only in size.

## What the prereg did not foresee, and it is the finding

Nothing in the prereg anticipated that the corrections would point in opposite
directions: 76 matrices want dilating and 64 shrinking, out of the 140 that are
not scale invariant. The prereg was written as though there were a global bias
to find. There is not one, and that is why the sweep's single scalar bought
0.836 % while eight matrices moved individually bought 6.35 %.

## The defect the prereg should have caught

Probes and validation both read wikitext-2 test, so the map is fitted on the
evaluation set. The prereg specified the held-out *combination* and never
noticed the corpus was not held out at all. Every gain in the journal is
optimistic by an unknown amount, and the next run owes probes on one split and
validation on another.
