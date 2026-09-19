# Preregistration. The trained row scales on the DCLM base, and the paper's own number

**Written, committed and TIMESTAMPED on 2026-09-19, BEFORE the run.**
Operator go given 2026-09-19. Training about $4 on l40sx1, scoring about $0.72.

## 1. The claim on trial

On bare Tetra, training the row scales was measured at **+4.30 pp** for zero bits, 54.64 to
58.94 (`docs/mesures/tetranu-rowscales-2026-09-19.txt`, McNemar p = 4.4e-40). That arm ran on
the C4-calibrated file, the worst base this project holds.

  DCLM base, Q5 served          57.95 micro   2.2044 kernel b/weight   2.8126 b/param
  **+ trained row scales**      **?**         **2.2044**               **2.8126**

The rate does not move. If the lever transfers, this object passes the paper's non-fine-tuned
LLVQ of **60.7** while costing fewer bits than our 59.33 arm.

## 2. Why the transfer is the question, and what the law says about it

Under `acc = 54.60 + 14.60 ln(SNR)` (`docs/mecanismes-perte-qualite-2026-09-12.md`), a lever
that divides the noise power by `f` returns `7.30 ln(f)` points **from any base**. The +4.30
corresponds to f = 1.802.

So the law rules out the lazy reading. "Our base is worse, so there was more to recover" does
**not** predict a smaller gain here: a multiplicative lever gives the same additive points
wherever it starts. Any shortfall has to come from **overlap** between the two levers, not from
headroom.

The precedent for overlap is measured. DCLM calibration and `down_proj@12-23` at int4 share
work, and only **57 %** of the int4 ceiling transferred once DCLM had run
(`docs/mesures/dclm-down1223-2026-09-19.txt`).

The case for low overlap here: `gptq.rs:245` fixes `row_scales` to the original row RMS
**before** the loop, whatever corpus the Hessian came from. A better Hessian moves where the
error is placed; it does not optimize the scales, because nothing does. The two levers act on
different objects.

## 3. Signed prediction

| quantity | point | interval |
|---|---|---|
| gain over the DCLM base | **+3.5 pp** | [+1.5, +4.5] |
| micro, full split | 61.45 | [59.45, 62.45] |
| noise power divided by | 1.61 | [1.23, 1.85] |

The point sits below full transfer and well above the int4 precedent, on the argument of §2
that the scales are untouched by calibration. The interval's floor is roughly the 57 %
precedent; its ceiling is full transfer.

The author's last signed prediction on this lever was wrong by a factor of ten, in the
pessimistic direction, and the deviation file records why. That is a reason to widen the
interval, not to move its centre.

## 4. What the arm is

  base        ~/qwen3-4b-dclm.bin, the DCLM-calibrated Q5 served object at 57.95
  export      `export` now reads its Int4G128 records, added 2026-09-19 with three
              tests and a mutation pass; 398 tensors, 36 from int4
  trained     the 216 lattice matrices only; `v_proj` is int4 and holds no `row_scales`
  objective   KL against dense Qwen3-4B, DCLM-edu streaming, seed 0
  budget      7,200 s of wall clock; the step count is read from a probe on the card
  fold        `rowscale`, which leaves the 36 int4 records untouched by construction

## 5. Known defects carried into this arm

**The basis split of E8.** The trainer splits columns at `d_in - d_in % 24` in the natural
basis while the artifact's tail is stored rotated. Bounded at 1.2e-4 relative and unrepaired.

**The training curve is not the measurement.** On bare Tetra the KL fell 7.5 % and MMLU moved
4.30 pp. The curve will be logged and must not be read as a result, in either direction.

**The first check is free and immediate.** The probe prints the KL at step one. On bare Tetra it
read about 0.44. The DCLM base is a better object, so a value materially above 0.44 means the
new int4 export path is wrong, and the run is stopped after six steps rather than after two
hours.

## 6. What refutes what

- Lands at or above 62: the two levers are independent and the object passes the paper's 60.7
  at 2.8126 b/param, under it on both axes.
- Lands near 60.4: the int4 overlap rate applies here too, and overlap is a property of the
  base rather than of the pair.
- Lands below 59.45: the interval is wrong and the bare-Tetra +4.30 was partly a property of a
  bad base, which the law of §2 says should not happen. That would be evidence against the law,
  and the law is load-bearing for the whole quality roadmap.
- Initial KL far above 0.44: the export is wrong, not the lever.

Scoring is one full-split arm, 14,042 questions, fingerprint `a74a6d6213602979`, paired on the
dumps against the DCLM base's own dump.
