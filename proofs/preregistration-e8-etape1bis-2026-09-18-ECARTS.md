# Deviations. E8 stage 1 bis prereg of 2026-09-18

The prereg is stamped (`8faa76613b95dd446f015d96636e5fbfaca81a788d3a0e09805c8b4af6551ec6`,
four calendars) and is not edited. This document reads beside it.

Both deviations concern the bit budget. Neither touches the primary's definition, and both
measured arms land inside the registered interval, so the verdict does not depend on them.

## E1. The prereg's own framing of the bit handicap was backwards

Section 2 says arm B at the norm-8 cap "runs on 3.7 % fewer bits, which makes a loss by arm B
conservative and a win by arm B qualified". The second half is right and the first is wrong.
Fewer bits make a **win** by arm B conclusive; they make a **loss** by arm B partly explained
by the deficit rather than by the codebook.

Arm B lost, so the handicap ran in the direction of the result. A second arm was run at the
norm-10 cap, 56,880 points, a 16-bit index, **2.0417 b/weight against Tetra's 1.9907**, where
arm B holds 2.6 % MORE bits. That arm is not in the prereg.

It changes the reading and not the verdict: `J_B/J_A` is 1.2896 at norm 8 and 1.0470 at norm
10, and the registered interval is [0.95, 1.30]. Both are inside it.

## E2. The equal-rate figure is interpolated, not measured

The journal reports `J_B/J_A = 1.146` at Tetra's exact 1.9907 b/weight. That is linear
interpolation between the two measured caps, 0.592 of the way from 1.9167 to 2.0417, and it is
labelled *computed* in the journal.

Measuring it directly needs a cap between shells 10 and 12, hence a partial shell and an
arbitrary ordering rule inside it. That was declined for the same reason it was declined in the
void run's E4: an arbitrary rule inside a shell is a free parameter nobody registered.
