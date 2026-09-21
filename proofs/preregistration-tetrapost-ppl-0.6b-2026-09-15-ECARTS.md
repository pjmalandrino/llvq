# Deviations from the tetrapost perplexity prereg, 2026-09-15

The prereg is stamped and is not edited. This file is what it got wrong.

## The signed prediction is refuted, in the opposite direction

§5 predicted B below A by 0.2 % to 1.2 % of perplexity, same sign on three
seeds. B is **above** A by **4.026 %** (41.8875 against 43.5739, *measured*,
[journal](../docs/mesures/tetrapost-ppl-0.6b-2026-09-15.txt)). Wrong sign, and
outside the interval by more than three times its width.

## The queue stopped where §4 said it would

§4: "A first pair inside ±0.5 %, or of the wrong sign, stops the queue there and
the remaining two seeds are not run." The first pair is of the wrong sign, so
seeds 1 and 2 were not run. 31 min of Mac spent of the 3 h authorised, $0.

## The doubt clause of §5 fired, and was discharged

§5: "any move beyond 3 % in either direction means the switch is not doing what
§1 says it does, and the measurement is to be doubted before the hypothesis."
4.026 % is beyond 3 %, so the clause fired. The four controls of §6 discharge it:
identical effective rate (2.1656 both), identical f32 baseline (19.5038 both,
and it is the M1 journal's own value), identical weight count (440,401,920 both),
one word of configuration different. The measurement stands; the hypothesis is
what falls.

## What §2 got right, and by how much it understated it

§2 named the pilot's unstable rollout regret as the signal arguing against the
prediction, and wrote that "the local gain is solid and its propagation is not".
That was correct and too mild. Propagation is not merely unstable: it is
reliably negative, and the journal measures why — the Euclidean rule shrinks
every reconstructed block by 2.9 % on average, one-sidedly, because its target
`⟨x,u⟩` never exceeds `‖x‖`.

Writing §2 before the run is what makes this a refuted prediction rather than a
surprise. The hypothesis had a named opponent and the opponent won.

## What no longer needs running

The 4B arm of the plan's §8, which this run gated. It is not run, and under §7
of the prereg nothing here licenses an MMLU arm either.
