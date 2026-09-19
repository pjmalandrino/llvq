# Three leads that cost no bits

Written 2026-09-19. The base is the DCLM-calibrated Q5 file, **57.95** micro on the full split
at 2.2044 kernel b/weight, sha256 `471f3988`. Every gain below is read against it.

Why these three and not the others: the session of 2026-09-18 bought 5.39 pp with 0.7364
b/weight, and measured that the lever fades with size (7.32 pp per b/weight at the 4B, 5.39 at
the 8B). Buying quality with bits has a decreasing return in scale. These three buy it with
none.

The running gate for all three: a re-encoding arm carries **2.92 pp** of draw noise against
0.43 at constant file. No single arm resolves a one-point effect, and every plan below says
what it does about that instead of pretending otherwise.

## Tracking

| # | lead | state | Mac | card | next action |
|---|---|---|---|---|---|
| A | Random window draws | **running** 2026-09-19 | 1 h 47 | $0.72 | encode `seed 1` at x1 |
| B | Row 5, `h_shrink` at the 4B | planned | 3 x 1 h 47 | $0 first | perplexity spread before any MMLU |
| C | Row C, intra-block sequencing | planned | 1 h 47 per arm | $0.72 | write the split capture |

---

## A. Random window draws, and why today's failure suggests it

**The observation that opened it.** Four times the calibration volume cost **1.20 pp**
(`dclm-v4-2026-09-18`), and perplexity agreed at 2.3 % worse. Both encodings drew their windows
as a **contiguous prefix from token 0** of one DCLM shard.

**The hypothesis.** More windows from a contiguous prefix add redundant text rather than
coverage. A Hessian averaged over more of the same thing is smoother, not better determined.
`LLVQ_CALIB_SEED` draws windows at random over the shard and has never been used on this axis.

**The design, a 2x2 already half filled.**

| | prefix | random |
|---|---|---|
| x1, 131,072 tokens | **57.95** measured | arm A1 |
| x4, 524,288 tokens | **56.76** measured | arm A2 |

A1 alone says whether the draw matters at the volume we serve. A2 says whether the volume's
failure was the prefix. Two encodings, 4 h 30 of Mac, two arms at $0.72.

**Prediction, signed.** A1 at **58.3** [55.4, 61.2], that is +0.35 pp over the prefix at the
same volume: the draw should matter little where the prefix is short enough not to be
redundant. A2 at **57.6** [54.7, 60.5], that is +0.85 over the prefix at four times: if the
redundancy reading is right, the random draw should lose less than the prefix did.

**The reading that would settle it.** If A2 minus A1 is positive where prefix x4 minus prefix
x1 was negative, the volume axis reopens with a different sampler and the a100 at 32 times
becomes worth its $5. If both differences are negative, the volume is dead whatever the
sampler.

**Order: A first**, because it needs no code, runs on the Mac for nothing, and follows directly
from a measured negative.

---

## B. Row 5, Hessian shrinkage at the 4B

**What is shipped and never used here.** `LLVQ_H_SHRINK` applies `H[i][j] *= rho` off the
diagonal, in the natural basis, before the rotation (`calib.rs:893`, applied at `:1052`). The
diagonal is left exact. The served 4B was encoded at **rho = 1**.

**What the 0.6B measured.** Median perplexity **-31 %** and the cross-seed range divided by
**6.7**, at rho in [0.5, 0.9] (`m1-hessienne-shrink-2026-09-02`). No MMLU figure anywhere.

**Why the 4B should show more, and it is the roadmap's own argument.** Shrinkage repairs an
under-determined off-diagonal. At 131,072 tokens the 4B has **13.5 samples a dimension on
`down_proj`** against 43.5 for the 0.6B. The matrix that most needs it is the one that carries
the most gain.

**The primary is the spread, not the mean.** A single rho arm cannot resolve a point against
2.92 pp of draw noise. But if shrinkage divides that noise by anything like 6.7, **every future
comparison in this project gets cheaper**, and that is worth more than the point estimate.

**The design, and it spends nothing on a card until it has to.**

  stage 1   three seeds at rho = 1 and three at rho = 0.7, perplexity only
            6 encodings, 10 h 40 of Mac, **$0**
            read: the spread of ppl across seeds, at each rho
  stage 2   only if the spread falls: one MMLU pair at the better rho
            2 arms, **$1.44**

**Prediction, signed, on stage 1.** The cross-seed ppl spread at rho = 0.7 is **at most half**
that at rho = 1. Named against me: an equal spread refutes the 0.6B's central finding at the
4B, and the row closes.

---

## C. Row C, intra-block sequencing

**The defect, and it is in the code rather than in a hypothesis.** `calib.rs:705-720` captures
the block's four Hessians in **one forward with the original weights**, then quantizes all
seven matrices. So:

  * `o_proj` is calibrated on an attention output its own q, k and v never quantized
  * `down_proj` is calibrated on an `act(gate) * up` that was never quantized

The module header at `calib.rs:1-13` denounces exactly this at the block level, and the code
commits it inside the block. **35 % of the file is calibrated on an input the served model never
sees.**

**The fix, four sub-passes a block instead of one.**

  1. capture `Attn` H, quantize q, k, v
  2. recompute attention with the quantized q, k, v
  3. capture `AttnOut` H, quantize o
  4. recompute the residual, capture `Mlp` H, quantize gate, up
  5. recompute `act(gate) * up`, capture `MlpOut` H, quantize down

**Cost.** The capture is 5.3 % of the encoding at x1 volume (337 s of 6,409). Four sub-passes
put it near 21 %, so the encoding goes from 1 h 47 to about 2 h 05. The code is the expense:
a restructuring of `quantize_model_capturing`, 150 to 250 lines, plus tests and mutants, behind
a flag so the published path stays bit-identical. **1 to 2 days.**

**Prediction, signed.** **+1.0 pp** [-1.9, +3.9] over the base. The point is deliberately
modest: the defect touches the calibration of two types out of seven, and the corpus change,
which touched all of them, was worth +1.58.

Named against me: a negative result would say that calibrating on an input the model never sees
is **better** than calibrating on the right one, which would be the most interesting failure of
the lot and would need its own explanation.

**Why it is third and not first.** It is the only one of the three that cannot start without
code, and the two ahead of it run on the Mac for nothing while it is written.
