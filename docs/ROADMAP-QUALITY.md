# Quality roadmap

The operator sanctioned this table on 2026-09-06. The adversarial pass of the same evening corrected
thirteen rows; what it changed is in its own section below. It replaces the quality axis of
[ROADMAP](ROADMAP.md) §2.3, which priced an arm at $7 on a `Planes14` base that is no longer the object.

**Folded on 2026-09-13**, on the operator's instruction, with the survey of 2026-09-12
([pistes-qualite-60](pistes-qualite-60-2026-09-12.md)): the accounting header is rewritten on the served
object, eleven cells carry a dated correction in place, and eighteen rows are added. The gain column of
row 6 moved — that column is the operator's, and this edit was asked for.

Rows are ordered by **feasibility**, most feasible first. The gain column carries **MMLU points only**.
Where no MMLU figure exists anywhere, the cell says so and the perplexity figure goes to the comment.
That distinction is the point of the table: our gap to the paper is a perplexity/MMLU dissociation, so a
lead validated in perplexity proves nothing here.

## What the accounting rests on

Rewritten 2026-09-13 on the served object. The 2026-09-06 wording, which rested on bare `Tetra`, is in
[HISTORIQUE](HISTORIQUE.md).

Base: the **served** 4B — 216 `Tetra` matrices plus 36 `v_proj` in int4 g128, q8 embedding — at
**2.8138 b/param whole model and 2.2044 kernel b/weight**, MMLU **56.37 on the full 14,042-question
split** (*measured*, dense arithmetic, [f1e-census-2026-09-11](mesures/f1e-census-2026-09-11.txt)); the
served kernel reads about 56.5 (*computed* from the −0.14 pp paired delta). The f16 checkpoint reads
70.32, the hard ceiling. The target is 60, so **+3.63 pp**.

The margin before the product triplet's b_max of 3.00 is **0.7956 kernel b/weight**. It is stated in
kernel b/weight and not in b/param because that is the accounting b_max is expressed in; the 0.7679
b/param this section carried until 2026-09-13 is the *bare-Tetra* margin, 0.8502 kernel b/weight
converted at 3,633,315,840 / 4,022,468,096 = 0.903255 (*computed*). Mixing the two is the one
subtraction to refuse: 3.00 − 2.8138 is not a margin, it spans two accountings.

A move to int4 g128 costs 2.10 kernel b/weight per weight moved, that is +0.021 kernel and +0.019
b/param per 1% of projection weights (*computed*).

An encoding of the 4B under `Tetra` costs **2 h 27 on the Mac and $0**. An MMLU arm on the **2,280-question
sample** costs $0 on the Mac; an arm on the **full split** does not — the census of 2026-09-11 cost $3.55
on an L40S, $0.70 per arm (*measured*, `data/jobs.csv`:147 and :149). Since the full split is what
resolves anything (below), Mac hours are no longer the only scarce resource.

Noise: **0.43 pp** at constant file (every `LLVQ_RESTORE_*` arm), **2.92 pp** as soon as an arm
re-encodes. Gains do not add: measured sub-additivity runs 0.618 to 0.792 over seven isolated arms, and
the one cross-family pair measured — Q5 with the calibration volume — composed worse, at **0.562**.

**What a sample can resolve, and it is less than this table assumed.** At 2,280 questions the paired
half-width on the published stratified micro is **2.67 pp** (*measured*,
[t3-genou-2026-09-12](mesures/t3-genou-2026-09-12.txt)), so no arm carrying 0.5 to 1.5 pp resolves its
own effect there — row A's proportional plan does not close it either, reaching [−0.11; +3.15] at best.
The full split has no sampling bar at all (±0.00) and a paired SE near 0.17 pp. Every "$0 on the Mac"
below therefore buys a **ranking**, not an interval.

## The table

| # | Name | Gain, MMLU pp | Feasibility | Comment |
|---|---|---|---|---|
| 1 | Q5 served: `v_proj` in int4 g128 | **+3.47** *measured on `Tetra`*, CI95 [+1.42; +5.57] | **Done as a measurement** ($0.60, 2026-09-06). Serving it is the `kind = 2` writer, in progress, then `tv_q4_h` on a card | **70.4 MMLU points per b/param**, twenty times the rate of the whole attention. Costs +0.0493 b/param, 6.4% of the margin before b_max. `Tetra` with it reads 56.95 for 2.8138 b/param against the served format's 55.59 for 5.1619 — but **that 56.95 is a `LLVQ_RESTORE_Q4` arm on the bare-Tetra file, not the re-encoded mixed object**, which reads 55.52 on the same questions; the +1.43 pp between them is not resolved (p = 0.2727, [t3-genou-2026-09-12](mesures/t3-genou-2026-09-12.txt)). Operator's decision of 2026-09-06: option B, code only, no re-encoding. No arm is served yet: `LLVQ_RESTORE_Q4` dequantizes to f16 before the matvec |
| 2 | `leech1c12` witness re-encoded | 0, it is a control | 0 days, 4 h Mac, $0 | Without it every delta measured against 55.59 carries the encoder drift of 2026-08-26: 87% of the indices of a re-encoded block differ. **Its premise has since moved**: the calibration shard went from 00000 to 00001 on 2026-08-01 (`corpus.rs`), so a witness re-encode is now a fresh draw against the published file and pairs cleanly only with the encodings of 2026-09-06 and 2026-09-09. Declined on 2026-09-06; the cost travels with every citation of the −4.64% |
| 3 | `leech0c13` at the 4B | **+1.19**, CI95 [−1.54; +3.98] — *measured, not resolved* | **Measured, and the verdict is suspended** (3 h 55 of Mac, $0.33, 2026-09-07) | **Measured in the wrong regime.** The paper's Table 10 gives no-gain-bit codebooks the win only under **Spherical GPTQ**; under Euclidean, which is what we ran, it gives them the loss (1B: 25.1 against 27.7 — ⚠️ neither figure is in `llvq-paper-notes.md`, so this citation is unsourced until the table is transcribed). This arm must be replayed after the spherical retraction lands. What held: It reads 19.6093 of perplexity against `Tetra`'s 16.1569, 21.5% worse, at an identical rate of 2.0702, and 54.67 of MMLU against 53.49, indistinguishable. Our three arms are indistinguishable in MMLU while their perplexities span a fifth. The 5.1-point gap to the paper is in the calibration volume, the corpus or the rotation, and rows 6 and 14 inherit it |
| 4 | Embedding in int4 g64 | **−0.35 pp** *measured*, under the 0.43 pp bar, so undetected rather than null | 0 days, the `q4b-e4.llvq` artifact exists, $0 | Negative memory cost: **−0.3868** b/param (8.5 to 4.5 b/weight on 388,956,160 embedding weights, *computed*; confirmed by `fiche-4b`'s e8 2.7961 against e4 2.4093). The −0.4049 this row carried until 2026-09-13 does not reproduce. +1.52% of perplexity measured, and that one is detected. The embedding **is** the language head at the 4B (`tie_word_embeddings = true`), so degrading it hits the logits directly and serving it is a second decode kernel per token, not a knob. Measured on a `Planes14` base |
| 5 | Q1, Hessian shrinkage | **no MMLU figure** | 0 days, `LLVQ_H_SHRINK` shipped; 7 to 15 h Mac, $0 | Median perplexity −31% and cross-seed range divided by 6.7 (*measured*, 0.6B, 3 seeds, [m1-hessienne-shrink-2026-09-02](mesures/m1-hessienne-shrink-2026-09-02.txt)). The file predicts a **larger** effect at the 4B: 13.5 samples per dimension against 43.5. The only large internal lever never tried at the 4B, and the 4B `Tetra` ran at rho = 1 |
| 6 | Calibration volume and composition | **+0.8 to +1.4** [−0.5; +3] *corrected 2026-09-13* | 0 days, 10 to 15 h Mac, $0 | Buried in perplexity on 3 blocks of the 0.6B, reopened on 2026-08-25: one arm moves **13.9% by changing the calibration text alone**, at full depth. Gated on MMLU sigma 2.92 over 2.0, not on price |
| 7 | Tail f32 to f16, accounting only | 0 | 1 to 2 days, $0 | **It frees nothing and pays for nothing.** The card has held the tail in f16 since 2026-08-09 (`TAIL_BYTES = 2`, `llvq-llm/src/fused.rs:737`), and `sealed::load` narrows it for `ppl` and `mmlu`, so 53.49 and 16.1569 already include it. What the correction buys is knowledge: the margin before b_max is **0.0747 b/weight larger** than published |
| 8 | Q4a, cross-layer equinorm | **0, structurally** | 1 to 2 days | Discarded by the adversarial pass. A block of 24 groups **rotated** coordinates: a diagonal s in the original basis becomes Q'diag(s)Q, dense, in the served basis, so it cannot equalize post-rotation block norms. The served Hadamard has already equalized them, kurtosis 3.01 — ⚠️ *measured* on **Qwen3-0.6B block 0 under a pure Hadamard (k = 1)**, which bounds nothing about the 4B's k = 5 and k = 19 Kronecker factors. The structural argument stands on its own; the kurtosis does not carry it |
| 9 | Rotation seed, best of N | **0, refuted — but on the wrong seed** | 1 to 2 days | Seed 1 has the **worst** perplexity, 16.7425, and the **best** MMLU, 58.02; seed 3 has the best perplexity and a median MMLU (*measured*) — ⚠️ these are **calibration** seeds (`LLVQ_CALIB_SEED`); the rotation seed has never been varied. The verdict stands, the label does not. A perplexity filter picks exactly the wrong seed. And publishing the max of N draws at sigma = 2.92 pp reports a selection artifact, not a gain |
| 10 | MagR | **no MMLU figure**; 0 to −5% of perplexity *estimated* | 3 to 5 days, ~150 lines in `llvq-quant`, $0, +10 min of encoding | The published −73% starts from a collapsed scalar GPTQ, 36.77 of perplexity for an f16 at 5.47. We sit at 1.32 times excess, where almost nothing is left to repair. Its infinity-norm objective is the statistic of an absmax scalar quantizer and has to be rewritten on shape-gain dispersion. Still the best-aligned lead structurally: no rotation, no permutation, no index touched |
| 11 | GPTAQ / GPTQv2 | **no MMLU figure at 2 bits**; −3 to −8% of perplexity *estimated* | 1 week, $0 on card, +10 to 40% of encoding | Their own gain decays from −56.4% to −7.5% as the base excess goes from 3.8 to 1.69 times. We are at 1.32, so their trend promises a few percent here, not 56. It changes the least-squares **target vector**, not the codebook: nothing to re-prove on the format side |
| 12 | Q3, K-best beam | **indeterminate, possibly negative** | 2 to 3 weeks, 2 h 27 times K on the Mac (~10 h at K = 4) | The beam optimizes the **local** proxy harder, and this file holds three precedents where a better local proxy composed worse: design C (perplexity times 1.99), `group_scales` (44.66 to 53.60), gptq2 (MMLU 24.74%, which is chance). The beam must range over (path, gain level), not (shell, point), so it is new code and not a port |
| 13 | Re-qualify the tail by salience | 0 to +2 *estimated* | days to weeks; **+0.0022 b/param** of permutation table, not zero, and it breaks format v1 | The served rotation destroys the notion of a salient column: the tail is the remainder modulo 24 in the **rotated** basis, where only the residual variation of the diagonal of Q'HQ survives. **That variation is measurable for $0 on the already-encoded artifact, and it decides the lead before any spend** |
| 14 | Learned row scales, the paper's fine-tuning | **DONE 2026-09-19/20: +4.30 pp on bare Tetra and +3.15 on the DCLM base**, *measured*, both at an unchanged rate ([tetranu](mesures/tetranu-rowscales-2026-09-19.txt), [dclm](mesures/dclm-rowscales-2026-09-20.txt)). The +2.1 this row used to carry is Table 6's **0-gain-bit** row; the row matching our object reads **+1.6**. And the paper's five no-FT/FT pairs make the gain fall with the base at r = -0.956, so a single number for this lever is not a quantity | Done: `ops/llvqtune` (hexagonal, 3 modes) plus `rowscale`; 1.71 h on l40sx1 and $4.70 an arm | The largest published gain at near-zero memory cost, and it is read on **our exact model**. Perplexity 17.05 to 9.26 as well. Same lever: +2.1 pp on QTIP, **+4.3 pp on QuIP#**. The decoder is **byte-identical** |
| 15 | E2E-QP, EfficientQAT scales only | **about 0 expected**, against the +1.15 pp announced | 2 to 3 weeks, $1 to $5 | They train one scale per **group of 64 weights**. We hold one per **row**, one per 3,285 weights on average: 51 times fewer degrees of freedom, and of another class, since a row scale is exactly a diagonal gain per output channel. The gradient still does not cross the decoder, which is why the lead stays on the list |
| 16 | Q4b full, 24x24 activation maps | not measured | 1 to 2 weeks, plus Q6c to rewrite first | +0.065 b/param. Internal anchor: radial bias, +3.69% of geometric overcost (*measured*, 0.6B, reproduced to the thousandth) |
| 17 | Q6a, distilling the format's free parameters | +2 to +5 *estimated* | weeks, ~$3 | About 18 M parameters already in the file: f16 tail, row scales, gain centroids, norms. Their values change, their widths do not. Zero bits |
| 18 | OWQ, weak columns in f16 | **no MMLU figure** | days for the encoder, weeks for the kernel; +0.0898 b/param | The mechanism rests on channel outliers **our input rotation exists to destroy**. Using it means extracting before rotating and rotating the residual: a third f16 tensor in the natural basis, a third launch, and a channel permutation that conflicts with row 13. Published at 3 bits, on OPT and LLaMA-1, without an incoherence rotation |
| 19 | GuidedQuant, output weighting | **no MMLU figure** | 2 to 4 weeks, no backward pass exists here | QTIP 6.82 to 6.11, −10.4%: **the only published measurement of this lever on a vector quantizer**, and it is in perplexity. Our spherical GPTQ assumes a metric where the block norm is preserved. The "retraction proof to redo" this row carried until 2026-09-13 is stale: Eq. 17 is a no-op under a coded gain (`ETAT` §4) |
| 20 | Q6b, low-rank correction, EoRA or RILQ | +2 to +4 *estimated* | 1 week after Q6a; **+0.263** b/param at r = 32 in f16 (+0.131 at r = 16) | Under `Planes14` it pushed to 5.41, above AWQ. Under `Tetra` it stays below 3.00: the lead becomes playable **only** because of `Tetra`. The +0.113 this row carried until 2026-09-13 is wrong — the shapes give 0.263 in f16, which `archive/ROADMAP-RECHERCHE.md`:186 already carried |
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

## Six rows added on 2026-09-07, from the paper re-read and the free hunt

The paper was re-read page by page against our 324 lines of notes, and a five-angle hunt looked for
free leads the survey had never listed. What follows is what survived an adversarial pruning, with
the two figures I recomputed myself on our own dumps.

| # | Name | Gain, MMLU pp | Feasibility | Comment |
|---|---|---|---|---|
| A | **Proportional MMLU sampling plan** | **0** — it moves the bar, not the score | half a day, $0 | The sampling error falls from **1.339 to 0.916 pp** at the same budget, a factor 1.46, or the same bar with **1,180 questions instead of 2,280** (*measured*, recomputed on `mmlu-4b-llvq.csv` with the repository's own stratified formula, which reproduces the published ±1.35). Three of our unresolved intervals — `Tetra` against `Planes14`, the radial correction, Q5 on V32 — would then exclude zero. Samples nest, since `select` shuffles by seed then truncates, so no existing dump is broken |
| B | **Output temperature** | unknown in MMLU, by construction invisible to an argmax | half a day plus one Metal run, $0 | Our logits carry a slope of **0.4660** against f16 where AWQ carries **0.9599** (*measured*, 9,120 centred logits over the paired dumps) — ⚠️ that 0.4660 is `Planes14`'s; the **served** file reads 0.4982, of which only the sd ratio (0.67) is temperature and the rest (0.75) is correlation, which no scalar removes. A positive scalar cannot move an argmax but it flattens a distribution, so **part of our ×1.32 perplexity excess is output calibration and not lost information**. One scalar folded into the final RMSNorm weight removes it at zero bits, and it may change the denominator of the four dissociations |
| C | **Intra-block sequencing** | unknown on both axes | 1 to 2 days, $0, encoding 2 h 27 to ~3 h 15 | `calib.rs:705-720` captures the block's four Hessians in one forward with the **original** weights, then quantizes all seven matrices. So `o_proj` is calibrated on an attention context its own q/k/v never quantized, and `down_proj` on an unquantized `act(gate)·up`. The module header at `calib.rs:1-13` denounces exactly this at the block level and the code commits it inside the block. 35% of the file is calibrated on an input the served model never sees |
| D | **Block-of-24 sweep order** | unknown | 1 day, $0 | `gptq.rs:255` sweeps left to right and `linalg.rs:68` factors without pivoting; neither order was ever justified by a measurement. This permutes the **order of visit**, not the contents of a block nor the order of records, so it costs no table and does not break format v1 — which is why the *Discarded* row on channel permutation does not apply |
| E | **Massive activations in H** | unknown | half a day to 1 day, $0 | `calib.rs:55-62` accumulates with no mask and no clipping. The internal hint is M1's own optimum at rho in [0.5; 0.9], which is what one would expect if a handful of rows of A dominated the covariance. Distinct from row 5, which regularizes **after** H is formed |
| F | **Cyclic option marginalisation** | at most +1 | one arm, ~$0.76 | The position bias is real and large: we under-pick B by 177 and over-pick C by 174 where f16 and AWQ are balanced (*measured*, same dumps) — ⚠️ those counts are `Planes14`'s; the **served** file over-picks A by 255 of 2,280. The per-letter additive fix is **dead** — oracle +1.05 pp, cross-validated **−0.51 pp** over ten folds — so the bias is question-dependent, not a global shift, and only the four-rotation average survives |

**D and E are settled by one free measurement**, the same one row 13 already asks for: the residual
spread of the diagonal of Q'HQ aggregated per block of 24, on the artifact we already have. One
measurement decides three rows.

**Row A is the most valuable line of the whole table and it yields no MMLU point.** It does not change
any number we have measured; it changes what we are allowed to say about them.

Two corrections the re-read forced elsewhere. **Removing the input rotation costs us 5.1 points**, not
the +2.5 an earlier reading claimed: the paper's no-rotation record of 37.4 belongs to *spherical
shaping*, and for our own shape-gain family Table 9 reads 34.9 with rotation against 29.8 without. And
the paper's default correction **is** Spherical GPTQ — rows without a qualifier are spherical, the
Euclidean ones say so — which is what suspends row 3.

## Eighteen rows added on 2026-09-13, from the survey of 2026-09-12

The survey is [pistes-qualite-60-2026-09-12](pistes-qualite-60-2026-09-12.md): 38 leads, four readers
over the repository, six literature sweeps, two adversarial verifiers a lead and a completeness critic.
76 agents, no run, no edit, $0. It corrected eleven cells of the table above — they carry their
correction in place, dated — and the eighteen rows below are the ones no row above covered.

Kept in the survey's own column order, and in the order of its decisive-measurement list rather than
this table's feasibility order: the instruments come first because they are what let the rest be read.
The survey's full 38 rows, with the leads that map onto rows 1 to 22 above, stay in that file.

| # | Lead | Chain | Gain, pp | Trade-off on the fundamental criteria | Complexity | Status in the repo | Decides it for the least |
|---|---|---|---|---|---|---|---|
| L36 | Capture-only pass with the served weights and dense H retained | instrument | **0** by construction; it decides **eight** rows (L02, L05, L14, L19, L23, L24, L25, L27) — the survey said ten, but L06's own cell asks for a paired re-encode and L15's for kurtosis on `f1recdump` blocks, and neither reads H (*measured*, [l36-capture-2026-09-13](mesures/l36-capture-2026-09-13.txt)) | None. One Mac hour once the code exists: pass 1 took 394.9 s on the 4B, `down_proj`'s H is 378 MB | **The capture half is done** (2026-09-13): `capture_model_hessians`, the `HessianSink` trait and the `hcapture` binary, built on `sealed::load` rather than `artifact2::load` because the sealed file is the served object itself. Its consumer, `llvq-bench --example hstats`, exists for **three** of the eight (L05, L14, L19 — the statistics a Hessian answers alone) and refuses a natural-basis dump by name. The five that need `ΔW`, hence the checkpoint, have no tool: `llvq-bench` cannot read one. Cost is in the **reduce**, not the capture: 126.3 s against 7.2 s, and the reduce does not scale with the calibration volume | Half built. `calib.rs` still drops dense H after factoring on the encoding path, by design; the capture path keeps it and streams it to a sink. Ten tests, five mutants, all five dead | It is the cheapest decisive measurement of the table |
| L01 | Ship the radial row-scale constant ρ = 0.929 on the served file | post-quant correction, 0 bits | **+1.23** [−0.50; +3.06], ***measured on the served base*** 2026-09-13 — **not resolved**, and on the unweighted count it is **+0.13 pp, p = 0.8944**: 112 questions won against 115 lost. The stratified gain is re-weighting — `professional law`, weight 10.9%, carries +0.819 of it. Bare Tetra read +1.65 [−0.18; +3.54] at p = 0.062, so the per-question signal *shrinks* on the served base | None: 0 bits, same kernel, no re-encode, 0.43 pp bar. Perplexity +9.5%, the fourth dissociation | **Done 2026-09-13**: `rhoapply` walks the served mixed file (216 lattice + 36 int4), passes int4 records through by kind and counts them, and its ρ = 1 idempotence control is byte-identical on that file. `v_proj` is never scaled **by construction** — an int4 record has no `row_scales` — and the tail never was. ρ recomputed on the served file: **0.929433** against the 0.929234 published on bare Tetra (*measured*, [l01-rhoapply-mixte-2026-09-13](mesures/l01-rhoapply-mixte-2026-09-13.txt)). What is left is the arm | Measured, not served; absent from the roadmap (M1 postdates it) | One full-split arm, $0.70 on L40S or ~50 min on the Mac; idempotence at ρ = 1 first |
| L37 | Record transplant into the served v5 file: every mixed-precision or post-hoc step as a constant-file arm | instrument | **0**; it moves the bar of L21, L26, L33, L34 and L35 from 2.92 pp to 0.43 (0 on the full split) and removes 2 h 27 of Mac per arm | None on the object | 1 day: a record walker over `read_record` and `write_record` with kind substitution; `rhoapply` refuses kinded files; `embedq` walks them by record since 2026-09-23 | Partial | $0: idempotence sha256 and bit-for-bit agreement with the restore path |
| L38 | Two more encodings of the served recipe at `LLVQ_CALIB_SEED` 1 and 2, scored on the full split | measurement design | **0**: the recipe's mean and sigma, published as a distribution, never as a best draw | None. ~4 h Mac and $1.40; a third seed +2 h and $0.70 | 0 days of code; operator go and a stamped prereg | Never run: seeded draws exist only for `Planes14` on the card | The encodings themselves |
| L22 | Protocol settlement: micro against macro (58.42), the f16 tie rule, full-split replays of f16 and `Planes14` | evaluation protocol | **0** [−0.13; +2.05], *computed*: the target moves by 2.05 only if the paper reports the unweighted mean; the repository adopted micro on 2026-08-01 | None. `Iterator::max_by` returns the last maximum, so exact f16 ties go to D: 0.16 pp on the kernel census dump, and the whole −0.14 kernel-versus-dense delta | 0.5 day; $1.86 for the two replays | Items new; row A done | $0: `mmlupair --intersect` of the census dump against the 19 flat dumps |
| L02 | Per-row scale refit in the Hessian metric, Algorithm 3 collapsed to one scalar a row | encoder objective | **0 to +0.3** [−1.3; +1], *estimated* (the identity-metric form equals the constant to 0.04 pp, M1b; the H metric may pull the scales back toward 1 and undo part of L01) | 0 bits. Post-hoc form: one capture pass with served weights, 20 to 40 min Mac; interleaved form ×K encodes. Counts once with L01 and L28's mask 1: one degree of freedom | 1 to 2 days beside `refine_group_scales` | Never tried in the H metric; new | $0 from L36: the distribution of the closed-form s_i. Near 1.0 with small spread closes the row axis on the constant; near 0.93 with structure earns one $0.70 arm |
| L25 | Per-output-row bias b = E[(W − Ŵ) x], closed form, post hoc | post-quant correction, bits | **+0.2 to +0.3** [−0.3; +1], *estimated* (DAC +0.33 MMLU on a scalar 2-bit Qwen3-8B) | +0.0044 b/param, +0.0049 kernel for an f16 field; one add per row | The eval arm costs nothing; 2 to 3 days for the field and the epilogue if it resolves | Never tried; new | $0 on the Mac: compute b on the candle side, apply it in a `RESTORE`-style arm, 0.43 pp bar |
| L23 | Super-weight rows of `down_proj` in f16 (early layers); sink-direction rank-1 correction on `v_proj` and `o_proj` | post-quant correction, bits | **+0.2 to +0.3** [0; +1.5], *estimated*, bimodal | +0.0004 b/param; −0.5 to −1.5% tok/s from extra launches | 1 day of diagnostics; 2 to 4 days for a per-row restore knob and a side-list record | Never tried; new. Row E is the input-side cousin | $0, one hour: one f16 forward for the per-layer maxima and the served reconstruction error along the sink direction |
| L14 | Block-of-24 sweep order: act-order at block granularity, min-pivot | encoder objective | **+0.1 to +0.2** [−0.3; +0.8], *estimated* (every act-order gain comes from un-rotated outlier models) | 0. Pivoting changes the factor: extend `both_factorizations_agree` | 1 day, `gptq.rs:255`, `linalg.rs:68` | Never tried; row D | $0: residual spread of diag(Q'HQ) per 24-block on the encoded artifact, shared with rows 13 and E |
| L33 | MSE-optimal clipping for the int4 `v_proj` groups in place of min/max RTN | mixed precision | **+0.15** [−0.1; +0.4], *estimated*, bounded by the same f16-minus-int4 gap | 0 bits, same bytes, same kernel; minutes of encoding; constant-file arm | 0.5 day: a clip-search sibling of `quantize_affine` in `embedquant.rs`; the pinned instrument `quantize_dequantize_q4` must not change | Never tried; new | $0: weight-space MSE at α = 1 against the best α per group on the 36 matrices, then the H-weighted error from L36 |
| L35 | Row scales at m = 2 to 4 segments a row, the middle of the Algorithm 3 ladder | post-quant correction, bits | **+0.1** [−0.5; +0.6], *estimated* (per-row freedom bought −0.04 over one constant; after the Hadamard the segments of a row are exchangeable to first order) | +0.0043 b/param and +0.0047 kernel per extra f16 scale a row (m = 4: +0.013 and +0.014, 1.8% of the margin); a segment multiply in the kernel | 0.5 day diagnostic; 2 to 4 days to serve | Never tried at any m between 1 and d_in/24; the M1 journal names it the column axis of chantier 14 | $0, half a day: fit m = 2 identity-metric scales on the served file, tail excluded; if 95% of rows sit within ±2% of 1, close |
| L27 | Sparse f16 outliers by weight salience (SqueezeLLM, SpQR) | mixed precision | **+0.1** [0; +0.4], *estimated* (perplexity only, 3 bits, scalar) | +0.144 kernel (18% of the margin), +0.130 b/param, a third launch; a worse rate than the int4 tier | 1 to 2 weeks, encoder and kernel | Never tried; row 18. A sparse overlay was measured as a kernel arm (`Planes12x`) | $0: share of Tr(ΔW H ΔWᵀ) in the top 0.45% weights |
| L08 | First-order error term in the compensation (FOEM): pull the running weights back toward the original during the sweep | encoder objective | **+0.4 to +0.5** [−0.5; +1.5], *estimated* (*reported* +2.3 MMLU at W3 scalar on Llama-3-8B, from a more collapsed base) | 0 bits, encoding unchanged, one new hyper-parameter β | 2 to 4 days: derive the 24-block form, ~80 lines in `gptq.rs:235-330`, mutant test | Never tried; new | 0.6B, 28 blocks, β in {0.05, 0.1, 0.2}, three seeds, read on the range |
| L09 | Teacher-decision token weighting of H (SchurQuant Eq. 19), forward only | Hessian | **0** [−1; +1], *estimated* (the paper reports a six-task zero-shot mean, no MMLU, and no isolation of the weighting; Eq. 19 carries a 1/r_i term the lead had dropped) | 0 bits; +36 forwards over the calibration set, 1 to 3 h Mac. Exclusive with L05's sink mask and L10: one weighting is served | 2 to 3 days | Never tried; new | $0: served-versus-f16 top-1 flip rate at prefix depths 9, 18, 27, 36 on 8 windows; already ≥ 14.7% at full depth, so only the depth profile is open |
| L10 | Output-gradient-weighted Hessian (GuidedQuant, KronQ, REAL-Q) through a one-off torch gradient export | Hessian | **0** [−0.5; +1], *estimated* (perplexity only; QTIP −10.4%, shrinking to −1.8% at 70B) | 0 bits; one torch backward over the calibration set; factor time × g | 5 to 10 days; backward in torch | Never tried; row 19 | Gradient traces per output row on 8 windows in torch; if rows differ by less than ×2, close at $0 |
| L29 | Output temperature, one scalar in the final RMSNorm | serving numerics | **0**, *computed* (argmax invariance) | None | 0.5 day | Row B | One Metal perplexity run; it reframes the perplexity gap, never MMLU |
| L31 | Instruments: M3 (STEM column, attention entropy), flips and KL to f16, recall-std, per-layer sensitivity map | instrument | **0** by construction | None | 0.5 day for the dump statistics, 1 to 2 days for the rest | Not built; M3 section | $0, half a day: the `mmlupair` columns on the existing dumps |
| L32 | Encoder numerics and the noise floor: the device confound (−1.18 pp on one pair), f64 accumulation, dead-column guard, draw counts | measurement design | **0**, a design fact | Sets the rule: three draws per re-encoding arm, or measure the variance reduction first | 0.5 day | New as a row | $0: encode block 0 twice with f32 against compensated accumulation and count differing indices |

**What crosses 60, and the honest answer is nothing here does.** The best stack the survey can build in
a month is **59.0 to 59.7** on the served draw, interval [57.5; 61.5]: the zero-bit constant-file leads
buy +0.8 to +1.1, the leads inside the margin +2.0 to +3.3, the re-encoding leads +1.0 to +1.5 more, and
sub-additivity of 0.62 to 0.79 takes the rest. The point estimate does not cross 60. What crosses 60 is
the **draw** — a fresh encoding of the served recipe, unchanged, passes it with probability near 11% at
sigma 2.92 — and row 9's rule holds: the best of N draws is a selection artefact, not a gain. The one
path to 60 that is not a lottery is the fine-tuning ladder, row 14's family, and it moves the comparison
to the fine-tuned column of the paper's Table 6.

**Two reservations travel with every figure above.** The sub-additivity was measured on f16 restore
arms, which overlap by construction; and the served recipe's deficit **doubles from the 4B to the 8B**,
so a 4B stack tuned to 60 says nothing about the classes the product triplet targets.

**Corrections owed elsewhere, and not made here.** `ROADMAP`:44 still counts the calibration volume
among "four leads stay closed" and `ROADMAP`:297 still says "the MMLU census is not run"; `ETAT` §7
still says the 4B scale-up never started. `HISTORIQUE` stops at 2026-09-07 and owes eleven entries
covering $42.14. Five rows of `data/jobs.csv` cite three journals that exist in no commit of any ref —
`volume-2026-09-07.txt` (three rows, $19.54), `q5-sur-v32-2026-09-07.txt` ($0.32) and
`m3-gptq-2026-08-30.txt` ($0.05) — and the first two carry the numbers row 6 above now rests on.

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
