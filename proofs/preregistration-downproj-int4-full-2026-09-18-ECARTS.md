# Deviations. down_proj prereg of 2026-09-18

The prereg is stamped (`155fcda3d1d8f47e54a23e564ad65a45552ec3e9a370211dd614551e9cc1326a`,
four calendars) and is not edited. This document reads beside it.

The protocol ran exactly as section 2 froze it. One deviation, and it is the prediction.

## E1. The signed prediction failed, in the direction the prereg named against itself

Section 4 predicted **+1.5 to +3.0 pp** on the held-out questions, point **+2.1**, and added:
"a point estimate at or above +3.79 would mean the selection bias did not operate on the
largest of six arms, which would itself need explaining."

Measured: **+4.05 pp**, CI95 [+3.28; +4.83]. Above the exploration's +3.79, and outside the
interval.

The explanation owed, and it is post-hoc. The winner's curse bites hardest when the true effect
sits near the noise floor. On the 2,280-question selection subset the paired SE is 1.19 pp, so
`o_proj`'s explored +2.77 stood 2.3 standard errors above zero and was half noise, which is why
it halved on confirmation. `down_proj`'s true effect is +4.05 with a held-out SE of 0.40: it
clears the noise by enough that being picked best of six barely inflated it.

One case each. It is not promoted to a rule, and the next confirmation of a selected arm should
still predict a shrink.

What this costs in credibility is worth stating plainly: a prediction that misses its interval
on the high side is still a miss, and the interval was set from one precedent rather than from
the arm's own standard error. A better rule would have started from the exploration's SE.
