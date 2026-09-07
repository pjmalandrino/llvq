# Quality roadmap

The operator sanctioned this table on 2026-09-06. The adversarial pass of the same evening corrected
thirteen rows; what it changed is in its own section below. It replaces the quality axis of
[ROADMAP](ROADMAP.md) §2.3, which priced an arm at $7 on a `Planes14` base that is no longer the object.

Rows are ordered by **feasibility**, most feasible first. The gain column carries **MMLU points only**.
Where no MMLU figure exists anywhere, the cell says so and the perplexity figure goes to the comment.
That distinction is the point of the table: our gap to the paper is a perplexity/MMLU dissociation, so a
lead validated in perplexity proves nothing here.

## What the accounting rests on

Base: `Tetra` at the 4B, 2.7645 b/param, MMLU 53.49, perplexity 16.1569 (*measured*,
[tetra-4b-2026-09-06](mesures/tetra-4b-2026-09-06.txt)). The f16 checkpoint reads 70.32, which is the hard
ceiling. The margin before the product triplet's b_max is 0.7679 b/param.

An encoding of the 4B under `Tetra` costs **2 h 27 on the Mac and $0**. An MMLU arm costs **$0 on the Mac**:
Metal reproduces the L40S harness on the served file to within two questions out of 600, which cancel
(*measured*, 2026-09-06). The scarce resource is Mac hours, not dollars.

Noise: **0.43 pp** at constant file (every `LLVQ_RESTORE_*` arm), **2.92 pp** as soon as an arm re-encodes.
Gains do not add: measured sub-additivity runs 0.618 to 0.792 over seven isolated arms.

## The table

| # | Name | Gain, MMLU pp | Feasibility | Comment |
|---|---|---|---|---|
| 1 | Q5 served: `v_proj` in int4 g128 | **+3.47** *measured on `Tetra`*, CI95 [+1.42; +5.57] | **Done as a measurement** ($0.60, 2026-09-06). Serving it is the `kind = 2` writer, in progress, then `tv_q4_h` on a card | **70.4 MMLU points per b/param**, twenty times the rate of the whole attention. Costs +0.0493 b/param, 6.4% of the margin before b_max. `Tetra` with it reads 56.95 for 2.8138 b/param against the served format's 55.59 for 5.1619. Operator's decision of 2026-09-06: option B, code only, no re-encoding. No arm is served yet: `LLVQ_RESTORE_Q4` dequantizes to f16 before the matvec |
| 2 | `leech1c12` witness re-encoded | 0, it is a control | 0 days, 4 h Mac, $0 | Without it every delta measured against 55.59 carries the encoder drift of 2026-08-26: 87% of the indices of a re-encoded block differ. Declined on 2026-09-06; the cost travels with every citation of the −4.64% |
| 3 | `leech0c13` at the 4B | **+1.19**, CI95 [−1.54; +3.98] — *measured, not resolved* | **Done** (3 h 55 of Mac, $0.33, 2026-09-07) | **The codebook is exonerated.** It reads 19.6093 of perplexity against `Tetra`'s 16.1569, 21.5% worse, at an identical rate of 2.0702, and 54.67 of MMLU against 53.49, indistinguishable. Our three arms are indistinguishable in MMLU while their perplexities span a fifth. The 5.1-point gap to the paper is in the calibration volume, the corpus or the rotation, and rows 6 and 14 inherit it |
| 4 | Embedding in int4 g64 | **−0.35 pp** *measured*, under the 0.43 pp bar, so undetected rather than null | 0 days, the `q4b-e4.llvq` artifact exists, $0 | Negative memory cost: −0.4049 b/param, `Tetra` 2.7645 to **2.3596**. +1.52% of perplexity measured, and that one is detected. The embedding **is** the language head at the 4B (`tie_word_embeddings = true`), so degrading it hits the logits directly and serving it is a second decode kernel per token, not a knob. Measured on a `Planes14` base |
| 5 | Q1, Hessian shrinkage | **no MMLU figure** | 0 days, `LLVQ_H_SHRINK` shipped; 7 to 15 h Mac, $0 | Median perplexity −31% and cross-seed range divided by 6.7 (*measured*, 0.6B, 3 seeds, [m1-hessienne-shrink-2026-09-02](mesures/m1-hessienne-shrink-2026-09-02.txt)). The file predicts a **larger** effect at the 4B: 13.5 samples per dimension against 43.5. The only large internal lever never tried at the 4B, and the 4B `Tetra` ran at rho = 1 |
| 6 | Calibration volume and composition | 0 to +2 on STEM *estimated* | 0 days, 10 to 15 h Mac, $0 | Buried in perplexity on 3 blocks of the 0.6B, reopened on 2026-08-25: one arm moves **13.9% by changing the calibration text alone**, at full depth. Gated on MMLU sigma 2.92 over 2.0, not on price |
| 7 | Tail f32 to f16, accounting only | 0 | 1 to 2 days, $0 | **It frees nothing and pays for nothing.** The card has held the tail in f16 since 2026-08-09 (`TAIL_BYTES = 2`, `llvq-llm/src/fused.rs:737`), and `sealed::load` narrows it for `ppl` and `mmlu`, so 53.49 and 16.1569 already include it. What the correction buys is knowledge: the margin before b_max is **0.0747 b/weight larger** than published |
| 8 | Q4a, cross-layer equinorm | **0, structurally** | 1 to 2 days | Discarded by the adversarial pass. A block of 24 groups **rotated** coordinates: a diagonal s in the original basis becomes Q'diag(s)Q, dense, in the served basis, so it cannot equalize post-rotation block norms. The served Hadamard has already equalized them, kurtosis 3.01 *measured* on real blocks |
| 9 | Rotation seed, best of N | **0, refuted by our own journal** | 1 to 2 days | Seed 1 has the **worst** perplexity, 16.7425, and the **best** MMLU, 58.02; seed 3 has the best perplexity and a median MMLU (*measured*). A perplexity filter picks exactly the wrong seed. And publishing the max of N draws at sigma = 2.92 pp reports a selection artifact, not a gain |
| 10 | MagR | **no MMLU figure**; 0 to −5% of perplexity *estimated* | 3 to 5 days, ~150 lines in `llvq-quant`, $0, +10 min of encoding | The published −73% starts from a collapsed scalar GPTQ, 36.77 of perplexity for an f16 at 5.47. We sit at 1.32 times excess, where almost nothing is left to repair. Its infinity-norm objective is the statistic of an absmax scalar quantizer and has to be rewritten on shape-gain dispersion. Still the best-aligned lead structurally: no rotation, no permutation, no index touched |
| 11 | GPTAQ / GPTQv2 | **no MMLU figure at 2 bits**; −3 to −8% of perplexity *estimated* | 1 week, $0 on card, +10 to 40% of encoding | Their own gain decays from −56.4% to −7.5% as the base excess goes from 3.8 to 1.69 times. We are at 1.32, so their trend promises a few percent here, not 56. It changes the least-squares **target vector**, not the codebook: nothing to re-prove on the format side |
| 12 | Q3, K-best beam | **indeterminate, possibly negative** | 2 to 3 weeks, 2 h 27 times K on the Mac (~10 h at K = 4) | The beam optimizes the **local** proxy harder, and this file holds three precedents where a better local proxy composed worse: design C (perplexity times 1.99), `group_scales` (44.66 to 53.60), gptq2 (MMLU 24.74%, which is chance). The beam must range over (path, gain level), not (shell, point), so it is new code and not a port |
| 13 | Re-qualify the tail by salience | 0 to +2 *estimated* | days to weeks; **+0.0022 b/param** of permutation table, not zero, and it breaks format v1 | The served rotation destroys the notion of a salient column: the tail is the remainder modulo 24 in the **rotated** basis, where only the residual variation of the diagonal of Q'HQ survives. **That variation is measurable for $0 on the already-encoded artifact, and it decides the lead before any spend** |
| 14 | Learned column scales, the paper's fine-tuning | **+2.1** *measured by the paper on Qwen3-4B* | 2 to 3 weeks, no training loop exists here, ~$3 to $8 | The largest published gain at near-zero memory cost, and it is read on **our exact model**. Perplexity 17.05 to 9.26 as well. Same lever: +2.1 pp on QTIP, **+4.3 pp on QuIP#**. The decoder is **byte-identical** |
| 15 | E2E-QP, EfficientQAT scales only | **about 0 expected**, against the +1.15 pp announced | 2 to 3 weeks, $1 to $5 | They train one scale per **group of 64 weights**. We hold one per **row**, one per 3,285 weights on average: 51 times fewer degrees of freedom, and of another class, since a row scale is exactly a diagonal gain per output channel. The gradient still does not cross the decoder, which is why the lead stays on the list |
| 16 | Q4b full, 24x24 activation maps | not measured | 1 to 2 weeks, plus Q6c to rewrite first | +0.065 b/param. Internal anchor: radial bias, +3.69% of geometric overcost (*measured*, 0.6B, reproduced to the thousandth) |
| 17 | Q6a, distilling the format's free parameters | +2 to +5 *estimated* | weeks, ~$3 | About 18 M parameters already in the file: f16 tail, row scales, gain centroids, norms. Their values change, their widths do not. Zero bits |
| 18 | OWQ, weak columns in f16 | **no MMLU figure** | days for the encoder, weeks for the kernel; +0.0898 b/param | The mechanism rests on channel outliers **our input rotation exists to destroy**. Using it means extracting before rotating and rotating the residual: a third f16 tensor in the natural basis, a third launch, and a channel permutation that conflicts with row 13. Published at 3 bits, on OPT and LLaMA-1, without an incoherence rotation |
| 19 | GuidedQuant, output weighting | **no MMLU figure** | 2 to 4 weeks, no backward pass exists here; the spherical retraction proof of its Eq. 17 has to be redone | QTIP 6.82 to 6.11, −10.4%: **the only published measurement of this lever on a vector quantizer**, and it is in perplexity. Our spherical GPTQ assumes a metric where the block norm is preserved |
| 20 | Q6b, low-rank correction, EoRA or RILQ | +2 to +4 *estimated* | 1 week after Q6a; +0.113 b/param at r = 32 | Under `Planes14` it pushed to 5.41, above AWQ. Under `Tetra` it stays below 3.00: the lead becomes playable **only** because of `Tetra` |
| 21 | Block-AP, full EfficientQAT | **unknown here** | 3 to 6 weeks; **out of budget by a factor of one hundred** | Discarded by the adversarial pass. Its own decomposition credits the scales and zero-points (10.26 against 14.32 for weights alone), and `Tetra` has neither a zero-point nor a group scale. Its only addition above distilling scales is training the weights, which needs a Leech re-encoding inside the loop: 245 s per block, 2 h 27 per pass, thousands of steps |
| 22 | PV-tuning, Q6d, end-to-end KL | **0 transposable** *estimated* | 1 to 2 months of dev plus 384 to 1,536 GPU hours, $270 to $1,250 | Discarded by the adversarial pass. The excess divided by 1.20 to 2.11 is measured in perplexity on a **learned** codebook, AQLM, and the paper carries no MMLU. On a **fixed** lattice the P step has almost nothing to move: it reduces to the row scales and the rotation signs, so half the measured lever does not exist here |

## What the adversarial pass changed, 2026-09-06

The table was written from a survey of 124 leads. An adversarial pass then recomputed every memory cost on
the 4B accounting and tested every gain for transposability to our setting. It contested 48 of the rows it
read and kept 65 leads. Thirteen rows above carry its corrections. Four collapse to zero, and the reasons
are worth more than the rows:

- **Row 8, Q4a**, is zero *structurally*: a block of 24 groups rotated coordinates, so a diagonal in the
  original basis is dense in the served one.
- **Row 9, the rotation seed**, is refuted by our own journal: the seed with the worst perplexity has the
  best MMLU. A perplexity filter picks the wrong seed, and publishing the best of N at sigma = 2.92 pp
  reports a selection artifact.
- **Rows 21 and 22** are out of budget by a factor of one hundred and one thousand, and both credit their
  gain to parameters `Tetra` does not have: group scales and zero-points for Block-AP, a learned codebook
  for PV-tuning.

Two corrections travel beyond their row. **The tail f32 to f16 finances nothing**: the card has held it in
f16 since 2026-08-09, so the published 2.7645 already includes it and the margin before b_max is 0.0747
b/weight larger than stated. And **the paper's fine-tuning line is not 45.7% of perplexity recovered**: it
lands at 9.26 where the paper's own FP16 reads 12.41, so the 2-bit model beats its own f16 by 25%. That is
adaptation to 52 M tokens of the evaluation corpus. Only the **+2.1 pp of MMLU** transports, which is why
this table carries gains in MMLU points and nothing else.

One correction is a free measurement. **Row 13's deciding number costs $0 on the artifact we already have**:
the residual variation of the diagonal of Q'HQ says whether any salience survives the rotation, and it
decides the lead before a line is written.

## The instrument that gates half the table

**M3**: attention entropy per layer, plus an MMLU-STEM column in `mmlupair`. One to two days of dev, $0, and
it yields zero MMLU points by construction. It says whether a lead that wins in perplexity can win in MMLU.
Rows 10, 11, 15, 19 and 21 carry no MMLU figure at all, so M3 gates them. `bin/attnent` does not exist and
`mmlupair` has no STEM column (*measured*, grep, 2026-09-06).

## What the gain column shows

Three rows carry a measured MMLU figure: Q5 (+2.71 to +3.60, ours), the learned column scales (+2.1, the
paper's, on our model), and `leech0c13` (upper bound +5.1, unattributed). Everything else is estimated, or
measured in perplexity only.

The four best leads from the literature, MagR, GPTAQ, GuidedQuant and E2E-QP, are all perplexity-only. We
read 16.94 of perplexity for 55.59 of MMLU where the paper reads 17.05 for 60.7: better on perplexity, 5.1
points worse on MMLU. A perplexity gain is not evidence here.

## What the field does that we do not

Post-quantization fine-tuning. Every competitor in the paper's Table 6 has one and we have none. It is worth
2.1 to 4.3 MMLU points and 40 to 50% of perplexity, at under 0.001 b/weight. It is the only structural gap in
the survey, and it is measured on our exact model.

Then, in order: asymmetric calibration (GPTAQ, standard since 2025, one week of dev), output weighting
(GuidedQuant, the only measurement on a VQ), the K-best beam (AQLM names it a principal gain), and MagR (a
zero-cost preprocessing step that beats QuIP at 2 bits).

## The competitive landscape, and where the accounting is not comparable

The paper's Table 6, Qwen3-4B, plus our two arms ([llvq-paper-notes](llvq-paper-notes.md)):

| method | fine-tuned | perplexity | MMLU |
|---|---|---|---|
| FP16 | n/a | 12.41 | 70.2 |
| GPTQ + QuaRot | no | 280.7 | 26.3 |
| QuIP# E8P12 | no | 21.15 | 48.6 |
| QTIP 3INST | no | 17.04 | 57.4 |
| LLVQ shape-gain, 0 gain bit | no | 17.05 | 60.7 |
| QuIP# E8P12 | yes | 10.52 | 52.9 |
| QTIP 3INST | yes | 9.61 | 59.5 |
| LLVQ shape-gain, 0 gain bit | yes | 9.26 | 62.8 |
| ours, `Planes14` | no | 16.9422 | 55.59 |
| ours, `Tetra` | no | 16.1569 | 53.49 |

On memory, they exclude the embedding and the language head from their headline figure. QTIP, QuIP#, AQLM,
GPTQ and AWQ all do. Under our accounting, hard rule 6:

| | announced | b/param whole model, f16 embedding | b/param with our q8 embedding |
|---|---|---|---|
| QTIP | 2.00 | **3.355** | 2.629 |
| `Tetra` | — | **2.7645** | 2.7645 |

Against QTIP as shipped we are ahead by 0.59 b/param. At equal embedding it is ahead by 0.135. The
same-embedding figure is the one to publish, by the same reasoning as the same-head rule for speed ratios.

## Discarded, and why

| lead | reason |
|---|---|
| Dense per-site learned rotations, SpinQuant to the letter | prohibitive if stored: 36 times d squared in f16 |
| FlatQuant, AffineQuant | 3.41 MB of extra parameters, non-orthogonal transforms |
| AWQ 1% in f16 | +0.3205 b/weight; the method itself abandons it |
| `group_scales`, the paper's Algorithm 3 | +0.5995 b/param, and dead internally: perplexity 44.66 to 53.60 |
| ICQuant, coded index | +0.271 b/param |
| Channel permutation | breaks format v1 for an estimated small gain: the Hadamard has already flattened magnitudes |
| Output rotation | mean effect about 0 pp across the four families of the paper's Table 9 |

## Provenance

The survey is 124 leads from six sweeps of 2026-09-06: the repository, then five families of 2-bit
literature (transforms, rounding and calibration, post-quantization fine-tuning, codebook design, memory-cheap
corrections). Every b/param figure is recomputed on the 4B accounting of
[rtbits-planes-8b-2026-08-09](mesures/rtbits-planes-8b-2026-08-09.txt) section 3.
