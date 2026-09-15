# Deviations from the gain-scale prereg, 2026-09-15

The prereg is stamped and is not edited. This file is what it got wrong.

## The prediction is half right

§4 predicted the minimum strictly above 1.00, inside [1.01, 1.10], beating the
control by at least 2 %. Location right, size wrong: the minimum is at 1.02 and
worth **0.836 %** (*measured*,
[journal](../docs/mesures/gain-scale-0.6b-2026-09-15.txt)). Under half the floor
the prediction named.

## The decision rule lands on its middle line

§5: "Best point beats 1.00 by 0.5 % to 2 % — report it, and do not spend 4B time
on it until the map explains it." 0.836 % is that line. No 4B arm is run.

## The grid was not extended, and why

§3 allowed one neighbouring point if the best sat at an edge. The best is at
1.02, an interior point, so no extension is owed. Adding 1.01 or 1.03 would
refine a compromise optimum whose value the journal argues is the wrong thing to
refine — a global scalar cannot sit at twelve different per-matrix optima.

## The two-block knob-check was an artefact, as §1 allowed

§1 recorded 41.2339 at 1.00 and 24.4208 at 1.05 on two blocks, and called it a
knob-check rather than a result. On the real protocol 1.05 is worse than 1.00.
Writing it as a knob-check before the sweep is what keeps it from having been a
finding that later evaporated.

## What §7 anticipated and the journal now sharpens

§7 said a multiplier that helps is a symptom, not a mechanism. The sweep adds the
reason a global one cannot do better: between 1.05 and 1.08 perplexity moves 23 %
for three points of scale, so the objective has a cliff, and a single number
serving matrices with different optima is buying its 0.836 % as a compromise.
