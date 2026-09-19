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

## E6. The run executed, and it is four times the preregistered budget

Job 6aaebe1652d0dbd7f1d6fcfb, l40sx1, created 16:53:42 UTC, completed 18:59:30.

  probe on the card            0.7437 s a step, so 2,754 tokens a second
  steps chosen from that       9,681
  tokens                       19,826,688, against the 7,372,800 of section 5
  training wall time           1.71 h, 0.6367 s a step measured over the run
  checkpoints written          48

The budget inverted as intended: the wall clock was the input and the token
count the output. The card ran 34 to 44 times faster than the Mac figures of
E1, which is why the budget bought 2.7 times the tokens the prereg named.

## E7. The training curve saturates, and the prediction is revised DOWN before scoring

  KL, first quarter of the run   0.2820
  KL, last quarter               0.2608
  fall                           7.5 %

By windows of 1,210 steps: 0.2929, 0.2710, 0.2699, 0.2664, 0.2570, 0.2561,
0.2621, 0.2595. Everything is acquired before step 1,210, which is 2.4 M
tokens. The remaining 17.4 M move nothing.

So section 6's worry, that one seventh of the paper's data could not separate a
weak lever from short data, is settled and not in the direction it feared: the
data was never the binding constraint. The paper's 52 M would not have changed
this curve.

The signed prediction of section 6 was **+1.0 pp [-0.5, +2.5]**. It is revised
**before the scoring arm is launched** to **+0.4 pp [-0.8, +1.6]**, on two
grounds visible without touching MMLU: the curve is flat after 12 % of the run,
and the trained sigma is centred on 1.0 (mean 0.99900).

Recording the revision here rather than editing the stamped prereg is hard rule
2. A prediction revised after the outcome would not be a prediction.

## E8. What was trained is not exactly what is folded back

`export` undoes the incoherence rotation (`export.rs:79`: "it rebuilds in f64,
undoes the incoherence rotation, and only then narrows"). The artifact's tail is
stored **in the rotated basis** (`format.rs:489`). So the trainer split its
columns at `d_in - d_in % 24` in the NATURAL basis, while `rowscale` leaves the
ROTATED tail unscaled.

The row scale itself is unaffected: an input rotation multiplies on the right
and does not mix rows, so a per-row scale is the same object in both bases. Only
the tail exclusion sits in the wrong basis.

  q_proj      16 of 2560 columns    0.62 %
  o_proj      16 of 4096            0.39 %
  down_proj    8 of 9728            0.08 %

With the measured spread of sigma, |1 - sigma| is about 0.02, so the two forms
differ by at most 1.2e-4 in relative terms and the dominant term is identical.
The arm is read as written, with this stated.

Found by the audit of 2026-09-19, not by the author.

## E9. The trained sigma has no global component

  mean 0.99900   sd 0.03463   p1 0.9057   p99 1.0858   min 0.3889   max 1.6205
  133,082 rows, 12.4 %, moved by more than 5 %

The global objective did not find the radial constant L01 ships. It kept the
mean at 1.0 and moved individual rows. Whatever L01's rho = 0.929 is, it is not
what a global fit converges to.

## E10. The scoring arm

  file      ~/qwen3-4b-tetra-ft.bin, folded by `rowscale` from ~/qwen3-4b-tetra.bin
  control   216 scaled, 36 untouched, 0 int4; same byte count; 6,953,069 bytes differ
  arm       full split, 14,042 questions, fingerprint a74a6d6213602979
  against   the bare Tetra dump at 54.64, paired on the dumps
  cost      about $0.72 on l40sx1
