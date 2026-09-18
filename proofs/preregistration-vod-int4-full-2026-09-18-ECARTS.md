# Deviations. v + o + down prereg of 2026-09-18

The prereg is stamped (`48c319fec8b463bc44741ec5a50f070d09ebffbe92d2f19788724a28a2c48320`,
four calendars) and is not edited. This document reads beside it.

The protocol ran as section 2 froze it, and all four registered quantities landed inside their
intervals. One deviation, and it is in the prediction's inputs rather than in its outcome.

## E1. The prediction's gap mixed two populations

Section 4 builds its point on "FP16 reads 70.32 on our protocol, the shipped object 56.37, so
the gap is 13.95 pp". Those two numbers come from different populations: **70.32 is a
2,280-question estimate and 56.37 is the full split.** No f16 had ever been scored on the
14,042 questions when the prereg was written.

It has now. The f16 full split reads **70.14**
(`docs/mesures/f16-full-2026-09-18.txt`), so the gap is **13.77 pp** rather than 13.95.
Recomputing the multiplicative model on it moves the point from +5.15 to **+5.22**, against a
measured +5.39. The prediction gets very slightly better, and the interval is unchanged.

The error is recorded because it is the class of mistake this repository polices, not because
it changed a verdict. A signed prediction whose inputs straddle two populations is a prediction
nobody can re-derive, and the f16 arm launched the same afternoon exists to remove that excuse
for good.
