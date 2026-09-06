# Quality roadmap

The operator sanctioned this table on 2026-09-06. It replaces the quality axis of
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
| 1 | Q5 / M2b replayed on `Tetra` | **+2.71 to +3.60** *measured* | 0 days of dev, 15 min, $0 on Mac | Two draws, CI above zero in both, paired at constant file. But the base is `Planes14`: nothing says it transposes from 53.49, and that is what this run settles. Serving the gain is 1 to 2 weeks more: the `kind = 2` writer does not exist and `tv_q4_h` has never run on a card |
| 2 | `leech1c12` witness re-encoded | 0, it is a control | 0 days, 4 h Mac, $0 | Without it every delta measured against 55.59 carries the encoder drift of 2026-08-26: 87% of the indices of a re-encoded block differ. Declined on 2026-09-06; the cost travels with every citation of the −4.64% |
| 3 | `leech0c13` at the 4B | up to **+5.1** if the whole gap comes from there, **0** otherwise | 0 days, 4 h Mac, $0 | The paper's 60.7 **is** this codebook. We serve `cap12 + 1 gain bit`, which appears in no LLM table of the paper. Measured at 0.6B only, and it is a coin flip: −9.56% of perplexity on draw 1, **+16.5% on draw 2**. Not transposable as-is to `Tetra`, which fixes one gain bit |
| 4 | Embedding in int4 g64 | never measured in MMLU | 0 days, the `q4b-e4.llvq` artifact exists, $0 | Negative memory cost: −0.4049 b/param, `Tetra` 2.7645 to **2.3596**. +1.52% of perplexity measured. The int4 gather kernel is missing to serve it |
| 5 | Q1, Hessian shrinkage | **no MMLU figure** | 0 days, `LLVQ_H_SHRINK` shipped; 7 to 15 h Mac, $0 | Median perplexity −31% and cross-seed range divided by 6.7 (*measured*, 0.6B, 3 seeds, [m1-hessienne-shrink-2026-09-02](mesures/m1-hessienne-shrink-2026-09-02.txt)). The file predicts a **larger** effect at the 4B: 13.5 samples per dimension against 43.5. The only large internal lever never tried at the 4B, and the 4B `Tetra` ran at rho = 1 |
| 6 | Calibration volume and composition | 0 to +2 on STEM *estimated* | 0 days, 10 to 15 h Mac, $0 | Buried in perplexity on 3 blocks of the 0.6B, reopened on 2026-08-25: one arm moves **13.9% by changing the calibration text alone**, at full depth. Gated on MMLU sigma 2.92 over 2.0, not on price |
| 7 | Tail f32 to f16 | 0 | 1 to 2 days, $0 | Returns **−0.0675 b/param** the card already does not pay: D1 prints 4.737 where `rtbits` bills 4.804. It pays for Q5 (+0.0493) with change left |
| 8 | Q4a, cross-layer equinorm | 0 to +1 *estimated* | 1 to 2 days plus a bit-exact invariance test | 1/s is absorbed by the 1,105,920 row scales. Nothing measured anywhere |
| 9 | Rotation seed, best of N | a few tenths *estimated* | 1 to 2 days plus 12 h Mac for N = 5, $0 | SpinQuant measures **13 points** between the best and the worst Hadamard rotation, but at W4A4, a different regime. Trap: the seed enters the artifact fingerprint |
| 10 | MagR | **no MMLU figure** | 3 to 5 days, ~150 lines in `llvq-quant`, $0, +10 min of encoding | OPTQ 36.77 to 9.94 at 7B (**−73% of perplexity**), and it **beats QuIP at 70B** (5.95 against 6.33) at 2 bits. Zero stored parameters, no inverse at inference. The best-aligned lead of the survey: no rotation, no permutation, no index touched |
| 11 | GPTAQ / GPTQv2 | **no MMLU figure at 2 bits** | 1 week, $0 on card, +10 to 40% of encoding | Llama-2-7B W2A16 with rotation: 20.7 to 9.02 (−56%). It changes the least-squares **target vector**, not the codebook: nothing to re-prove on the format side. About 20 lines for them, more here, since two forward passes must run in parallel |
| 12 | Q3, K-best beam | +0.5 to +1.5 *estimated* | 1 to 2 weeks, 2 h 27 times K on the Mac (~10 h at K = 4) | Under `Tetra` the K best paths are a **by-product of the trellis**, which they were not under Ball. The beam must range over (path, gain level), not (shell, point) |
| 13 | Re-qualify the tail by salience | **not measured, the hole in the file** | days to weeks, memory cost **exactly zero** | 16,957,440 weights kept before and after: the same spend, better placed. Reference point, OWQ: 60 to 83% of the gap closed for +0.01 b/weight. It breaks the index map, so the full suite runs before any commit |
| 14 | Learned column scales, the paper's fine-tuning | **+2.1** *measured by the paper on Qwen3-4B* | 2 to 3 weeks, no training loop exists here, ~$3 to $8 | The largest published gain at near-zero memory cost, and it is read on **our exact model**. Perplexity 17.05 to 9.26 as well. Same lever: +2.1 pp on QTIP, **+4.3 pp on QuIP#**. The decoder is **byte-identical** |
| 15 | E2E-QP, EfficientQAT scales only | **no MMLU figure**, +1.15 pp of 5-task zero-shot average | 2 to 3 weeks, $1 to $5 | The cheapest fine-tuning path: the gradient **does not cross the decoder**, since a served weight is a row scale times a constant decoded vector. Excess divided by 1.58, measured |
| 16 | Q4b full, 24x24 activation maps | not measured | 1 to 2 weeks, plus Q6c to rewrite first | +0.065 b/param. Internal anchor: radial bias, +3.69% of geometric overcost (*measured*, 0.6B, reproduced to the thousandth) |
| 17 | Q6a, distilling the format's free parameters | +2 to +5 *estimated* | weeks, ~$3 | About 18 M parameters already in the file: f16 tail, row scales, gain centroids, norms. Their values change, their widths do not. Zero bits |
| 18 | OWQ, weak columns in f16 | **no MMLU figure** | days for the encoder, weeks for the kernel; +0.0898 b/param | The best bits-per-quality ratio of the survey, **but at 3 bits, on OPT and LLaMA-1, and without an incoherence rotation**, and it is the rotation that makes our outliers. An arbitrary column cuts a Lambda-24 block in two |
| 19 | GuidedQuant, output weighting | **no MMLU figure** | 2 to 4 weeks, no backward pass exists here | QTIP 6.82 to 6.11, −10.4%: **the only published measurement of this lever on a vector quantizer**. Caveat: our spherical GPTQ assumes a metric where the block norm is preserved |
| 20 | Q6b, low-rank correction, EoRA or RILQ | +2 to +4 *estimated* | 1 week after Q6a; +0.113 b/param at r = 32 | Under `Planes14` it pushed to 5.41, above AWQ. Under `Tetra` it stays below 3.00: the lead becomes playable **only** because of `Tetra` |
| 21 | Block-AP, full EfficientQAT | **no MMLU figure** | 3 to 6 weeks, re-encoding and re-signing on every pass | Perplexity 8.53 against 10.26 for scales alone. The paper's lesson: training weights **without** the scales is worse than the scales alone, 14.32 against 10.26 |
| 22 | PV-tuning, Q6d, end-to-end KL | **no MMLU figure**, zero-shot deficit 7.29 to 3.45 | 1 to 2 months of dev plus 384 to 1,536 GPU hours | Excess divided by 2.11 at 7B, measured. The only lead whose projection reaches the target on its own, and the only one out of budget by an order of magnitude |

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
