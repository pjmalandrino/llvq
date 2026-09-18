# Preregistration. The calibration volume at four times, corpus held constant

**Written, committed and TIMESTAMPED on 2026-09-18, BEFORE the MMLU arm and before the
encoding it scores has finished.** Operator go given 2026-09-18 for the seal, the upload and
the arm, conditional on the encoding completing cleanly. Cost announced: about 24 min on
l40sx1, **$0.72**, timeout 1 h.

Not edited again. A fact it gets wrong goes beside it, in a `-ECARTS.md`.

## 1. The second factor

The corpus was measured on 2026-09-18 at constant volume: DCLM-edu against C4, same recipe,
**+1.58 pp** [+0.71; +2.49]. This arm measures the other factor, volume, with the corpus held
at DCLM-edu.

  arm              corpus   tokens      status
  served Q5        C4       131,072     56.37, measured
  dclm x1          dclm     131,072     57.95, measured
  **dclm x4**      dclm     **524,288** this arm

Everything else is identical across the three: `tetra1`, rotation seed `0x110feed`, `nogs`,
`h_shrink` 1, `gain_scale` 1, `LLVQ_INT4_TYPES=v_proj`, 2.2044 kernel b/weight.

A first attempt at eight times was killed after 105 min: the machine reached 0.8 GB free with
20.8 GB of swap and 1.4 TB of swapouts, and ran at 393 s a block against 178 at one times. Four
times runs at 265 s a block with no compression and no new swap. The volume ceiling on this
machine is memory, not time.

## 2. Why the effect should be concentrated, and where

Volume only helps a Hessian that is under-determined. Samples per dimension at 131,072 tokens:

  q, k, gate, up   n = 2,560   51.2    already determined
  o_proj           n = 4,096   32.0    determined
  **down_proj**    n = 9,728   **13.5**  the only marginal one

At four times, `down_proj` reaches **54**. So if the volume changes anything, it should change
it **through `down_proj`**, and the other five types should barely move. That is a mechanism,
not an intuition, and it is falsifiable: a gain that appears with no change in `down_proj`'s
contribution would refute it.

## 3. The primary result

The paired gain of `dclm x4` over `dclm x1` on the full split, stratified bootstrap without
finite population correction. The gain over the C4 served object is reported beside it, as the
total of both factors.

## 4. Signed prediction

| quantity | point | interval |
|---|---|---|
| gain over `dclm x1` | **+0.8 pp** | [−2.1, +3.7] |
| micro, full split | 58.75 | [55.9, 61.7] |
| total over the C4 object | +2.4 | [−0.5, +5.3] |

The point assumes the volume is worth about half the corpus at this factor: the corpus changed
every one of the 216 Tetra matrices, while the volume can only help the 36 whose Hessian was
short of samples.

The interval is the repository's measured re-encoding noise, **2.92 pp** between calibration
draws at the 4B, and it is wide for that reason alone.

Named against me: above +3.7 the volume is worth more than the corpus at a factor of four,
which would make the paper's factor of 95 the single largest lever in the record. Below −2.1,
more calibration makes the object worse, which has no mechanism I can state.

## 5. The readings, and what each one costs next

| gain over x1 | reading | next |
|---|---|---|
| >= +2.5 pp | The volume pays at four times | the a100-large at 32 times, ~$5, becomes the obvious spend |
| −1 to +2.5 | Indistinguishable from a draw at this factor | the 32 times arm is a gamble on the same noise; decide on the mechanism check below, not on this number |
| <= −1 | More calibration hurts | record it and stop the volume axis |

**The mechanism check, and it is the cheaper decision.** If the volume works through
`down_proj`, then restoring `down_proj` to int4 **on the x4 file** should gain less than the
+4.05 it gains on the C4 file. One arm, $0.72, and it answers whether the calibration has taken
int4's place, which is the question that decides whether bits can be given back.

## 6. Controls, checked before the arm is paid for

1. The encoding reports 36 blocks and exits cleanly.
2. The sealed file is 1,794,564,765 bytes, the byte count of the served object and of the
   dclm x1 file, and its sha256 differs from both.
3. The dump carries 14,042 questions and plan fingerprint `a74a6d6213602979`.
4. `oracle` first on the backend, hard rule 10.
5. The dump kept whole and committed.

## 7. What it will not establish

- No significance on an effect of about a point: one draw, 2.92 pp of noise. This registers a
  point and three readings, as the corpus arm did.
- Nothing about the paper's factor of 95, which needs 128 GB of activations and which no single
  card in the flavor list can hold.
- Nothing about bare `Tetra`: every arm in this chain carries `v_proj` at int4.
- Nothing about perplexity, another model or another size.
