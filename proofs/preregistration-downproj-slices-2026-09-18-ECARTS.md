# Deviations. down_proj depth slices prereg of 2026-09-18

The prereg is stamped (`863fed682d3c5a6e4cdc76ba1f5efbfe5984360c9dcaf8dd1656bb6373c7c520`,
four calendars) and is not edited. This document reads beside it.

The protocol ran exactly as section 2 froze it. One deviation, in the third prediction.

## E1. The spread came out above its interval

Section 4 predicted the best slice divided by the worst at **2.0**, interval **[1.0, 5.0]**.
Measured: **2.41 / 0.48 = 5.02**, just above it.

The concentration is therefore stronger than the prereg dared claim. Two remarks, both against
the reading that this is a small miss.

First, the denominator is the one arm the run did not resolve: the 0-11 slice reads +0.48 with
an interval of [-0.17, +1.10], which contains zero. A ratio whose denominator is unresolved is
not a quantity to lean on, and the honest form of the same statement uses the two resolved
arms: **2.41 / 0.63 = 3.83**, inside the interval.

Second, the prereg said the interval was wide because "the repository holds no depth
attribution for `down_proj`". That was true and it remains the reason the interval was set from
nothing. Landing at its edge is what an interval set from nothing does.

The first two predictions landed inside: sum +3.52 against +3.9 [+3.3, +4.6], best slice +2.41
against +1.9 [+1.4, +3.0]. The gate at +2.02 was cleared, and the verdict does not turn on the
spread.
