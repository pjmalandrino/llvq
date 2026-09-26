# Preregistration. Row scales trained on task-format text instead of generic text

**Written, committed and TIMESTAMPED on 2026-09-23, BEFORE the run.**
Operator go given 2026-09-23, on the pure `mmlu-aux` arm, with the prediction of §3 signed as
written.
Training about $4.00 on l40sx1, scoring about $0.72. Nothing else is billed.

## 1. The claim on trial

The trained row scales took the DCLM-calibrated 4B from 57.95 to **61.11** for zero bits, and
took perplexity from 16.2509 to 12.3268 — **1.0074 times f16** (`docs/mesures/
dclm-rowscales-2026-09-20.txt`). The object now models generic text almost as well as the dense
model it came from, and still answers MMLU **nine points** below it.

That is the largest of the six dissociations this project has recorded, and it is what puts the
generic-text objective in question. This arm changes the training text and nothing else:

  base                     ~/qwen3-4b-dclm.bin, 57.95 micro, 2.7475 b/param
  + scales trained on DCLM      61.11        (the object of 2026-09-20)
  + scales trained on MMLU-format   **?**    same 2.7475, same 1,794,564,765 bytes

Same export, same teacher, same 9,507 steps, same 19,470,336 tokens, same seed 0, same
hyper-parameters, same fold. One flag: `--corpus mmlu-aux`.

## 2. It is not contamination, and that is measured

`cais/mmlu` ships `auxiliary_train`, 99,842 multiple-choice items, beside the three splits the
harness reads. The overlap was measured before any code was written (`uv run
ops/mmlu_aux_overlap.py`, 2026-09-23, revision c30699e):

  test        14,042 rows · question-only 14 (0.10 %) · question+choices **0**
  dev            285 rows · question-only  0 (0.00 %) · question+choices **0**
  validation   1,531 rows · question-only  1 (0.07 %) · question+choices **0**

All 14 are bare stems ("Which of the following is true?") under four unrelated options. The
adapter refuses `dev`, `validation` and `test` **by name**, and the mutant that removes that
guard is killed by the suite. The job re-runs the gate on the Mac before it launches.

`dev` matters most: the harness draws its five worked examples there, so a scale trained on it
would be trained on part of every scored prompt.

## 3. Signed prediction

| quantity | point | interval |
|---|---|---|
| paired delta against the 61.11 object | **+0.5 pp** | [−1.0, +2.0] |
| micro, full split | 61.6 | [60.1, 63.1] |
| perplexity, wikitext2, Metal, f32 | 14.0 | [12.2, 18.0] |

The point is small and the interval is wide, for reasons that pull opposite ways.

**Why small.** The trained object is 1,069,056 scalars, one per output row. A row scale cannot
carry task knowledge; it can only re-place quantization error across output channels. The best
placement under one English corpus and another English corpus should be close, and `gptq.rs:245`
fixes those scales to the row RMS before any corpus is read, so the corpus enters only through
the gradient's weighting of directions. The lever is a re-weighting of *where* error sits, not a
new degree of freedom.

**Why the interval is wide.** Changing calibration text alone has already moved perplexity 13.9 %
at full depth on this project (`ROADMAP-QUALITY` row 6), so the corpus is not inert. And the
author's last two signed predictions on this family were wrong by a factor of ten and by 0.89 pp.

**Why perplexity is predicted to rise.** 24.75 M tokens of multiple-choice prompts are far less
diverse than DCLM-edu. Scales fitted on them should model generic text worse than scales fitted
on generic text. A gain on MMLU *with* a perplexity rise is the expected shape; a gain on both
would be the surprise, and would say the two objectives were never in tension.

## 4. What the arm is

  base        ~/qwen3-4b-dclm.bin, 1,794,564,765 bytes, the sealed Q5 served object at 57.95
  export      /out/dclm-export-2026-09-19, 8,044,981,648 bytes, the very file the 61.11 object
              was trained from — re-used, not rebuilt, so the two arms share their input exactly
  trained     the 216 lattice matrices only; `v_proj` is int4 and holds no `row_scales`
  objective   KL against dense Qwen3-4B, unchanged
  corpus      `cais/mmlu` `auxiliary_train`, blocks in the harness's own format, six to a
              generic header, packed into 1,024-token sequences, seed 0
  steps       9,507, fixed, not budget-derived: the token count must equal the DCLM arm's
  fold        `rowscale`, which leaves the 36 int4 records untouched by construction
  scoring     one full-split census, 14,042 questions, fingerprint `a74a6d6213602979`, paired
              against the 61.11 object's own dump

## 5. Known defects carried into this arm

**The header names no subject.** The evaluation prefix reads "…questions (with answers) about
<subject>"; `auxiliary_train` ships no subject (every row's is empty, *measured*), so the
training header is the same sentence without the subject clause. This is the one respect in
which the training text is not the evaluation's text, and it is written here rather than
discovered in a result.

**The corpus is finite.** The split holds about 24.75 M tokens (*estimated* from 247.9 a
question, *measured* over 1,200 rows under the Qwen3-4B tokenizer) against the 19,470,336 this
run consumes: 0.79 of a pass at seed 0, no repetition, 27 % of margin. `check()` refuses the run
before a weight is loaded if the plan outruns the split. At seed 3 it would not fit, and that is
a refusal, not a silent short run.

**MAX_FIRST_KL is not set, deliberately.** The 0.44 and 0.50 thresholds the earlier arms carried
are first-KL values on DCLM text. This run reads its first KL on other text and the two are not
comparable. The export is the same file the 2026-09-19 run already validated, so that guard is
spent. `MAX_TRAIN_SECONDS=7600` still refuses a run that would die at the timeout.

**The basis split of E8**, unchanged and unrepaired: the trainer splits columns at
`d_in - d_in % 24` in the natural basis while the tail is stored rotated. Bounded at 1.2e-4
relative.

**The training curve is not the measurement**, and a KL read on this corpus cannot be compared to
the 0.2434 the DCLM arm logged. Different text, different KL. It will be logged and must not be
read as a result in either direction.

## 6. What refutes what

- **Lands at or above 63.1**: task-format training is a lever of its own, larger than the
  interval, and the survey's "nothing crosses 60 that is not a lottery" needs rewriting.
- **Lands between 61.5 and 63.1 with perplexity risen**: the expected shape. The corpus buys
  MMLU by spending generic-text quality, and the next question is the mix ratio, which
  `--corpus mix` already serves.
- **Lands within [60.7, 61.5]**: under the 0.43 pp paired bar this is *no detected difference*.
  The row scales are corpus-insensitive, which is what §3's "why small" argues, and the lead
  closes for one run's price. That is a real outcome, not a failed one.
- **Lands below 60.1**: task-format text is actively worse, and the diversity argument of §3
  beats the distribution-match argument. The mix arm would then be the only survivor.
- **A gain on MMLU *and* a fall in perplexity**: the two objectives were never in tension and
  something else is going on. That result would need its own investigation before publication.

## 7. Cost, and what is not being asked for

Training $4.00, scoring $0.72, **$4.72 total**, both on l40sx1. The fold and the perplexity arm
run on the Mac for $0. No re-encoding, no export, no new sealed base: every large object this arm
needs is already in the bucket (`hf buckets ls`, 2026-09-23).

Not in this prereg: the mix arm, a second seed, and the 8B. Each is another $4.72 and each waits
on this result.
