# Deviations from the DCLM row-scale prereg (2026-09-19)

The prereg is timestamped and is not edited.

## E1. The run, and the free control on the new int4 export path

Job 6aaeeddc52d0dbd7f1d70916, l40sx1, launched 20:07 UTC.

  probe rate        0.7573 s a step
  steps chosen      9,507
  tokens            19,470,336

Section 5 named the probe's first KL as the check on the int4 export path added
the same day: bare Tetra read 0.4797 on the probe's first batch, and the DCLM
base is a better object, so materially above that would mean a broken export.

  bare Tetra, probe first batch, seed 0    0.4797
  DCLM base,  probe first batch, seed 0    0.3527

**26.5 % lower on identical text.** The export is sound, and this is also the
first direct measurement of how much closer to the dense model the DCLM
calibration gets, read on logits rather than on accuracy.

## E2. Section 2's argument is wrong, and the paper's own table refutes it

Section 2 argues that under `acc = a + c ln(SNR)` a multiplicative lever returns
the same points from any base, so a shortfall here could only come from overlap
between the two levers.

Table 6 of `docs/llvq-paper-notes.md` carries **five** no-FT / FT pairs on
Qwen3-4B, and the fine-tuning gain is a strongly decreasing function of the base:

  Quip#/E8P12       48.6 -> 52.9   +4.3
  LLVQ spherical    50.5 -> 54.9   +4.4
  QTIP (3INST)      57.4 -> 59.5   +2.1
  LLVQ sg-2bit      59.3 -> 60.9   +1.6
  LLVQ sg-0bit      60.7 -> 62.8   +2.1

  gain = 15.974 - 0.2364 x base,  r = -0.9564,  residual sd 0.452 pp

So the gain is NOT base-independent in the paper's own data. Either the law does
not apply to this lever, or the lever recovers a share of the gap to FP16 rather
than a fixed factor on the noise power. Section 2's reasoning is void either way.

## E3. A competing prediction, recorded BEFORE the result

The prereg signed **+3.5 pp [+1.5, +4.5]**, so 61.45. The regression of E2
predicts, at base 57.95:

  **+2.27 pp [+1.08, +3.47]**, so **60.22**

The prereg's prediction is NOT revised: the run was already launched when the
regression was computed, and moving a centre after launch on a new reading is
the practice this file exists to prevent. Both are recorded, both are scored,
and the result arbitrates between them.

For the record, the same regression scores the arm already measured: it predicts
+3.06 pp at base 54.64 where +4.30 was read, a residual of +1.24 pp, about 2.7
residual standard deviations. So "we doubled the paper's effect" is not what the
data says; the excess over their own trend is +1.24 pp.

One reading for that excess, unverified: we trained on DCLM-edu against a
C4-calibrated base, so part of what the scales bought is the calibration-corpus
repair the repository prices separately at +1.58 pp. The paper calibrates and
fine-tunes on the same corpus and cannot collect that term. If that reading is
right, the excess should NOT reappear here, because this base is already
DCLM-calibrated, and the arm lands nearer 60.2 than 61.5.

## E4. The comparison in the previous journal cites the wrong row

`docs/mesures/tetranu-rowscales-2026-09-19.txt` compares our +4.30 to the paper's
+2.1. That +2.1 is the **0-gain-bit** row. Our object carries a gain bit, so the
comparable row is **shape-gain 2 bit, 59.3 -> 60.9, which is +1.6**.
`ROADMAP-QUALITY.md` row 14 carries the same misattribution.

## How E2 to E4 were found

By a workflow of six lenses on 2026-09-19, whose adversarial verification stage
died on an account spend limit. The readings above are therefore **unverified by
the intended protocol**; the regression itself was recomputed by hand from the
transcribed table before being written here.
