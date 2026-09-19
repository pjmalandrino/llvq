# Deviations from the row-scale training prereg (2026-09-19)

The prereg `preregistration-tetranu-rowscales-2026-09-19.md` is timestamped and is not edited.
This file records what departed from it and what measured the departure.

## E1. The throughput in section 4 was wrong by a factor of three

The prereg priced the run at **210 tokens a second** and the budget at **about 11 h** for 3,600
steps. Both are wrong.

The 280 tokens a second it extrapolates from was measured under **cross entropy**, which only
gathers the target logit. The run uses **KL over the whole vocabulary**, and Qwen3's vocabulary
is 151,936. Worse, the extrapolation measured the student alone, without the teacher resident
and without the routed weights in the graph.

Measured on 2026-09-19, Qwen3-4B, seq 1024 batch 2, MPS, full configuration:

  weight() rebuild, KL f32      26.21 s a step    78 tokens a second
  linear() commuting, KL f32    32.53 s a step    63 tokens a second
  linear() commuting, KL f16    25.11 s a step    82 tokens a second

So 3,600 steps cost **about 26 h on the Mac**, not 11.

## E2. The run was started and stopped

Started 15:24:02 UTC, stopped on operator go after 13 minutes with fewer than 10 steps logged.
No sigma was written and no checkpoint was reached. Nothing from it is used anywhere.

## E3. Two explanations were proposed and both were refuted

Recorded because the refutations are the measurement, and because the first one is written into
the prereg's own section 4.

**The teacher forward.** Refuted. It costs 1.76 s against the student's 1.99 s, and the student
forward is unchanged with the teacher resident, 2.11 s against 1.99 s. There is no memory
pressure penalty from holding it.

**The rebuilt weights.** `RoutedLinear` materializes 3.52 G values, 7.05 GB in f16, into the
autograd graph at every step. Removing that entirely, by scaling the output instead of the
weight, made the step **slower**: 32.53 s against 26.21 s. The tail correction costs 216 extra
small matmuls whose launch overhead exceeds what the rebuild costs.

The commuting form is kept and is proven equal to the rebuilt form to 1e-12 in f64
(`tests/test_trainables.py::test_linear_agrees_with_building_the_weight`). It is **off by
default**, because the trade turns on the device and has never been read on a card.

## E4. The measurements themselves are unstable on MPS

The same quantity, the student forward at seq 1024 batch 2, read 1.99 s, 2.11 s, 5.88 s and
4.16 s across four probes on the same machine within the hour. That is a spread of three.

Consequence for the method: a step time read on MPS is not a number this repository should plan
against. The full-configuration figures above are the ones to use, because they were read on the
configuration that runs, and even they carry that spread.

## E5. What the protocol becomes

Unchanged: objective, corpus, seed, schedule, what is trained, the signed prediction of section
6 and the refutation table of section 7. Those do not depend on where the run executes.

Open: the venue and the step count. The prereg's 3,600 steps cost 26 h on the Mac. Deciding
between a longer Mac run, a shorter one, and a card is an operator call and is recorded here
once made.
