# Preregistration. The sensitivity map: is FP16's sensitivity at depth 12 uneven in a basis known in advance

Written on 2026-11-15 on branch `recherche/qualite-n6-n7-2026-11-15`, before any job, by an executing agent and the
lead, who reviewed it and signs §12. Timestamped before the first reading, then never edited; a departure goes to
`preregistration-sensitivity-map-4b-2026-11-15-ECARTS.md`. The job waits for the operator's go; the phase has spent
$0.61 of its $10.

Script: `ops/sensitivity_map.py`, sha256 `94ffd6fe53b2b30828e157c60bb0e77c0495b95c4fe423ace40dd9ef89f5ae49`. It imports, unedited, `ops/patch_map.py` sha256
`a531bf4f14ab6517f0d3e61192a2e1068d9dbcafbd3a4d39abd706ebbc7e709f`, `ops/contrast_map.py` sha256
`936fa6cdb4b39b5d0d4d5aa1de1d265a3f11fe55d0c46e89a7dc60612fc94a20` and `ops/layer_vs_output.py` sha256
`c031c52f1294f3e044621bf7ed3e7f5089da0fb593f0ced5ca826fbb67596f12`. Job script: `ops/jobs/sensitivity-map-4b.sh`,
sha256 `63c4b7635181a2d11e30eb478662cd21278c642ae76c05093de89d79fe169bf2`. The job refuses to launch unless the stamped prereg names these sha256.

Cost: **$0.57 to $0.84** (*estimated*, §9), ceiling **$1.35** (timeout 45 min on `l40sx1`).

## 1. The question

R6 refuted the conjecture of R5: what a band of the stream gives back at depth 12 follows the noise energy it held,
flat over the variance bands (70.6, 79.3, 70.2 a unit, *computed* in R6). One way out was left: a sensitivity that is
uneven over directions in a basis that is not the variance order. Is it, on FP16 Qwen3-4B, in a basis estimated on
other text, so that an encoder could know it in advance? The reading is the give-back of a band per unit of the sealed
deviation's energy in that band, [1, 64] and [1, 512] over [513, 2560].

## 2. The objects

As patch-map-4b: FP16 `Qwen/Qwen3-4B@1cfa9a7208912126459214e8b04321603b3df60c`; the f16 export of the sealed 4B,
`sealed-4b-export-2026-09-23/model.safetensors`, 8,044,981,648 B; MMLU at `cais/mmlu@c30699e8`; the scored questions
are the first 570 of the round robin, contrast-map's, row for row. One torch process, f16 weights, image
`pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`, `l40sx1`. h(12) is the stream after block 11, as patch-map. A patch
writes into the sealed run's h(12) at every position; the sealed blocks 12 to 35, norm and head follow.

## 3. The three bases

All three are 2,560 × 2,560 orthonormal eigenbases, columns by eigenvalue descending, cut by rank in patch-map's
bands [1, 64], [65, 512], [513, 2560].

- `variance`: FP16's uncentred second moment of h(12) over every position of the 570 scored prompts. Patch-map's
  basis, recomputed by the same code in the same order.
- `fisher_text`: Σ g gᵀ, g ∈ R^2560 the gradient with respect to FP16's h(12) at one position of Σ_t log p(y_t), every
  position of a window, y_t **sampled** from FP16's own next-token distribution (seed 20261116, one label a position).
  256 non-overlapping windows of 512 tokens from token 0 of WikiText-2 test, the stream `layer_vs_output.fetch_tokens`
  pins (file sha256 and tokenizer sha256, 299,078 tokens). 131,072 position gradients from 256 backward passes.
- `fisher_mmlu`: Σ_k p_k s_k s_kᵀ, s_k the gradient with respect to h(12) at every position of log p_k, the
  four-option log-probability at the answer position, p FP16's own. That is the exact Fisher of the four-way answer
  (4 backward passes a question). On rows 570 to 1139 of the round robin, the 570 that follow the scored ones (10 more
  a subject). The job refuses to load the models if one shares a (subject, index) or a token hash with the scored set.

Why sampled labels. The Hessian of KL(FP16 ‖ perturbed) at FP16 is the Fisher under FP16's own distribution, the
quantity whose output-side factor YAQA estimates (attack-litterature B1, as summarised there; the paper was not
reopened). With sampled labels the cross-target terms of g gᵀ vanish in expectation (selftest, §12). Observed labels
give the empirical Fisher, which end-loss-gradient methods such as GuidedQuant use (abstract level only).

The Fisher an encoder can use is one d × d basis shared by every position, so the cross-position blocks of the full
(L·d) × (L·d) Fisher are dropped and the position blocks summed. Longer prompts and windows weigh more.

How the gradient is computed. FP16's h(12) in f16 under no_grad, the very stream the patch reads, cast to float32.
Blocks 12 to 35 and the head run in float32, f16 weights cast per call, matmul precision "highest" (TF32 refused).
Each block is checkpointed (non-reentrant) and attention runs on the math SDPA backend. Reasons: an f16 backward
underflows per-coordinate scores and can overflow at the head; float64 runs at 1/64 of the float32 rate on this card;
the math backend runs the same operations with and without autograd. Memory: math attention at the longest Fisher
prompt (~3,000 tokens) holds 32 × L² floats a layer, about 1.1 GB a tensor; unchecked over 24 layers it would pass
46 GB. Checkpointed, the peak is one block's recompute plus the resident 16.1 GB of weights, about 22 GB (*estimated*;
the job prints the peak).

## 4. The arms

| arm | what replaces the sealed run's h(12), every position |
|---|---|
| `fp16`, `control` | nothing: the two models |
| `d12` | the whole stream |
| `var_b1-64`, `var_b65-512`, `var_b513-2560` | one band of `variance`: h_s + (h_f − h_s) V_b V_bᵀ |
| `ft_b1-64`, `ft_b65-512`, `ft_b513-2560` | one band of `fisher_text` |
| `fm_b1-64`, `fm_b65-512`, `fm_b513-2560` | one band of `fisher_mmlu` |
| `ft_not1-64`, `fm_not1-64` | every band but the top one, of each Fisher basis |
| `var_b1-512`, `ft_b1-512`, `fm_b1-512` | the top two bands of each basis together, [1, 512] |

Beside every band: the energy of the sealed deviation δ = h_s(12) − h_f(12), every position of the 570 prompts, in
that band, as a share of the deviation's energy. Three conventions, all printed: `all`, every coordinate; `pos0`,
direction 1 of `variance` removed at position 0 only; `dir1`, direction 1 removed at every position. The mis-scale of
direction 1 (energy at position 0 and at every position, as shares of the whole deviation) is printed apart.

## 5. Gates

Each gate is printed as read; one red gate and nothing is read.
1. The three selftests GREEN in the job before a model loads (`contrast_map.py` 53 checks, `patch_map.py` 48,
   `sensitivity_map.py` 47).
2. The scored questions are contrast-map's row for row; the Fisher questions share none with them.
3. Rule D against the two Rust dumps, and contrast-map reproduced (419 and 378 right within 2, picks ≥ 0.99), as
   patch-map.
4. Patch-map reproduced: `d12` and the three `variance` bands give patch-map's picks on ≥ 0.99 of the questions and
   their s within 0.01 of patch-map's 0.390, 0.033, 0.144, 0.335 (`logits.tsv`, `shape.tsv`, pinned by sha256).
5. The variance control, on the two `variance` ratios of §7, point estimates. R_64 inside [0.5, 2]: its expectation
   is 1.03 (*computed* from contrast-map's depth-12 band slopes and noises, convention `dir1`; R6 reads 70.6 over
   70.2 = 1.01). R_512 inside [0.25, 2]: patch-map has no joint [1, 512] arm, so its expectation is a bracket,
   *computed* on patch-map's `logits.tsv` with the same energies, from 0.35 [0.02, 0.79] (s(`d12`) − s(b513-2560),
   the complement bound) to 1.14 [0.69, 1.62] (s(b1-64) + s(b65-512), the additive bound). The bands are
   super-additive (attack 5d), so the joint arm should fall between the two.
6. Exactness, bit for bit, first 57 prompts: the sealed h(12) patched into itself through the top and the bottom band
   of each basis changes nothing; `d12` equals FP16's blocks 0 to 11 then the sealed ones; FP16's logits in pass 2,
   after the gradient passes, equal pass 1's on all 570; the float32 logits of the gradient pass equal those of the
   same forward without autograd (8 windows, 8 questions).
7. Tolerances: three bands patched together give `d12`'s pick on ≥ 0.98 of 57 prompts, each basis; each basis has
   max |VᵀV − I| ≤ 1e-9; the three band energies sum to the deviation's energy computed in the canonical basis within
   1e-9 relative, every question, basis and convention; no gradient holds a non-finite value; the float32 path's pick
   equals FP16's f16 pick on ≥ 0.98 of the 570 Fisher questions.
8. The weights unchanged by sha256 after the loops.

## 6. The readings

The primary statistic is patch-map's: c_i, the cosine of an arm's four centred answer logits with FP16's;
s = (C − C_control) / (1 − C_control). It stays primary on patch-map's instrument check (Spearman 0.95 on these
dumps); the accuracy share is printed beside. Every interval is a 95 % paired bootstrap, 10,000 resamplings of the 570
questions, seed 20261115; the deviation energies are resampled with their questions.
- Every arm: C, s, accuracy and its difference to the control (b, c, McNemar), accuracy share, KL, log p of FP16's pick.
- Every basis: s of each band and of [1, 512]; the deviation energy share E of each band and of [1, 512], three
  conventions; s / E; the ratios R_64 = (s_[1,64] / E_[1,64]) / (s_bot / E_bot) and R_512 = (s_[1,512] / E_[1,512]) /
  (s_bot / E_bot), bottom = [513, 2560], under each convention and on the accuracy share; the cross term
  s_1 + s_2 + s_3 − s(`d12`) (attack 5d: a band's give-back is not its cost). For the Fisher bases, s of the
  complement arm and s(`d12`) minus it, the top band's marginal give-back.
- Spectra, *computed* on FP16 alone: trace share of each band, the arithmetic over geometric mean of the eigenvalues
  (the transform-coding gain; eigenvalues under 1e-12 of the largest left out and counted), the number of gradient
  samples, and the split-half overlap of the top-64 subspaces (windows, resp. questions, split by parity).
- Overlaps of the top-64 subspaces, ‖V_aᵀ V_b‖²_F / 64: `fisher_text` with `fisher_mmlu`, each with `variance`. At
  random it reads 0.025.

## 7. The kill rule

On `fisher_text`, the basis an encoder could have in advance: R_64 and R_512, convention `dir1`, and E_512, the share
of the deviation energy its [1, 512] holds. Read only if every gate is green and s(`d12`)'s interval lies above 0
(patch-map: [0.310, 0.463]).

| outcome | condition on the 95 % intervals | what follows |
|---|---|---|
| UNEVEN | R_64's or R_512's interval wholly above 3 | A unit of deviation energy costs over three times more in the directions running text marks as sensitive; the published two-sided objective has something to hold on to here, and its cost on this format is written and attacked before any encode. |
| FLAT | R_512's interval wholly inside [0.5, 2], R_64's not wholly above 3, and E_512 ≥ 0.15. If `fisher_text`'s [1, 512] holds under 0.15 of the deviation energy, FLAT cannot fire. | In the basis an encoder could have in advance, a unit of deviation energy costs the same in the most and the least sensitive directions; the lead proposes to close the axis at this rate. |
| NOT DECIDED | otherwise | The measurement does not separate the two; §8 says which side was reachable. |

UNEVEN outranks FLAT: a resolved R_64 above 3 reads UNEVEN whatever R_512 reads. Printed beside, no rule: both ratios
for `fisher_mmlu` and for `variance` (the latter is the gate-5 control), under every convention and on the accuracy
share. No outcome is a mechanism or a gain: a patch is a probe, and a Fisher is a local second-order quantity read
against a deviation that is not small.

## 8. Power

From patch-map's `shape.tsv`: a band share is known to ±0.045 to ±0.07 absolute (b1-64 [−0.011, +0.079], b65-512
[+0.096, +0.192], b513-2560 [+0.264, +0.403]), whatever its size. R6 read s(b513-2560) / s(b65-512) = 2.33 [1.72,
3.39]. The same R on the `variance` bands, with the deviation energies of gate 5 held fixed (`dir1`: E = 0.065,
0.252, 0.683), reads 1.03 [−0.36, 2.37] (*computed*, paired bootstrap on patch-map's logits): the top band holds too
little energy for its share to be resolved, so gate 5 reads a point, not an interval.
The two verdicts are not equally reachable (*computed*, approximate). Under flatness s ≈ 0.5 E (patch-map's three
bands read 0.50, 0.57, 0.49 a unit of `dir1` energy). FLAT needs s_top resolved to about ±35 %, s_top ≳ 0.13, hence
E_top ≳ 0.25 of the deviation energy in `fisher_text`'s top 64 directions. UNEVEN needs the lower end of s_top / E_top
above three times the bottom band's upper end (0.59), s_top ≳ 1.8 E_top + 0.045, e.g. 0.135 at E_top = 0.05: a top
band giving back about five times its energy-proportional share. R_64 alone could not read FLAT unless E_top ≳ 0.25.
That is why the rule reads FLAT on R_512. In the `variance` basis today [1, 512] holds 0.317 of the `dir1` deviation
energy (*computed*, contrast-map's bands), and the additive bound of its ratio reads 1.14 [0.69, 1.62], inside
[0.5, 2]: at that energy FLAT is reachable. In `fisher_text` the energy of [1, 512] is unknown before the job. Under
0.15 of the deviation energy, s_[1,512] would be about 0.075 or less under flatness, unresolved at ±0.05, and the rule
does not let FLAT fire (§7 table); UNEVEN stays reachable. The job prints E_512 beside the verdict.

## 9. Cost

Measured figures it rests on (patch-map-4b, job 6af9953a, `l40sx1`): load and one base hash 118 s; pass 1 285 s for
570 prompts with the 252-matrix error tap, which this job drops; pass 2 782 s for 570 prompts, i.e. a tapped control
and 454 untapped blocks a prompt, about 1.8 ms a block (*computed*); 1,225 s running in all.
Estimates: pass 1, one f16 forward a prompt, about 40 s. Text, 256 windows: a float32 forward of 24 blocks at 512
tokens is 2.5 TFLOP; checkpointing makes forward, recompute and backward about three of them, 8.2 TFLOP with the head,
0.15 to 0.25 s at 40 to 60 TFLOP/s of float32 plus the per-call casts, 40 to 65 s. Fisher, 570 questions at a mean of
605 tokens: one forward and four recompute-plus-backward passes, nine forward-equivalents of 2.9 TFLOP, plus the f16
forward for the pick check, 0.6 to 0.9 s, 350 to 510 s. Pass 2: 432 blocks a prompt (FP16, control, 15 arms of 24
blocks) at 1.8 ms plus three f64 projections, about 0.79 s, 450 s, and 25 s of checks. Setup about 4 min; analysis
about 1 min. Total 19 to 28 min billed, $0.57 to $0.84 at $1.80/h. Timeout 45 min, ceiling $1.35. Caps: the job
stops with exit 5 if pass 1 projects over 300 s after 57 prompts, the text over 300 s after 32 windows, the Fisher pass
over 900 s after 57 questions, or every loop together over 1,800 s after 114 prompts of pass 2.
Why 256 windows: 51 position gradients a dimension, two halves of 128 for the split-half check, about a minute of
card; 512 tokens sits near the scored prompts' median of 477 and keeps the full-vocabulary head at 0.6 GB.

## 10. What does not follow, whatever the outcome

- No MMLU point is gained or projected; no encoder changes. A band's give-back is not its cost (cross term printed).
- The Fisher bases are FP16's sensitivity; the arms read what the sealed blocks 12 to 35 do with FP16's band.
- A basis at depth 12 is not what an encoder of blocks 0 to 11 writes (attack 3, attack 6); folding a basis costs the
  tied head 0.026 to 0.435 b/param (attack 4). Neither is measured here.
- One size, one calibration draw, MMLU only, 570 questions easier than the full split; GSM8K is not read.
- WikiText-2 test is not the calibration text (dclm-edu); "other text" means not MMLU.

## 11. Choices this draft makes that the brief left open

1. Labels of `fisher_text` sampled from FP16 (the true Fisher), one a position, seed 20261116; not the observed token.
2. Text: WikiText-2 test as `layer_vs_output` pins it, the first 256 windows of 512; dclm-edu has no pinned reader in
   this image.
3. `fisher_mmlu` exact over the four options (4 backward passes), not sampled.
4. Position blocks summed, cross-position blocks dropped; no weighting by prompt length.
5. The gradient path of §3: float32 from the exact f16 h(12), checkpointed blocks, math attention, no TF32.
6. The rule's deviation energy is `dir1`, direction 1 removed at every position, not the brief's `pos0`, because
   only `dir1` has a computed expectation for the variance control (R6's "without direction 1"); `pos0` and `all`
   are printed beside every ratio.
7. Variance control: the point inside [0.5, 2], and patch-map reproduced (s within 0.01, picks ≥ 0.99).
8. Dropped from patch-map: the 252-matrix error taps and their gate, the massive, position, attention and branch
   arms, the lens at depth 25.
9. R's interval is a joint bootstrap of shares and energies; draws with a negative denominator are kept, which only
   widens it.
10. The two complement arms are included (they cost about 50 s).
11. The cosine share stays primary on patch-map's instrument check; the check is not rerun.
12. Amendment of the lead: the arms `var_b1-512`, `ft_b1-512`, `fm_b1-512` and the two-ratio rule of §7, which
    makes FLAT reachable when the top 64 directions hold little energy (§8). The FLAT floor E_512 ≥ 0.15 is written
    into the rule, so the table says when FLAT cannot fire. The variance control on R_512 is a bracket, because
    patch-map has no joint arm; its tolerance [0.25, 2] covers the bracket's two bounds.

## 12. Dated disclosure, and the signed predictions

What has run, as of writing, all on the Mac, no model loaded, nothing downloaded but the cached torch wheel:
- `uv run ops/sensitivity_map.py --selftest`: GREEN, 47 checks, tiny random models on CPU in float64, one float16
  check. They cover the gradient against Richardson-extrapolated finite differences (the forward's RMSNorm runs in
  float32, so a small step reads rounding: agreement 7.7e-4 and 6.5e-4, tolerance 2e-3); the sampled-label estimate
  against the exact Fisher of a window, cross-position blocks included (relative error 0.094 at 600 draws); a planted
  rank-2 head recovered by the text Fisher (top-2 overlap 1.000, against 0.130 for the variance basis); band patches
  in a non-variance basis; the energy shares; the ratio statistic on planted data (FLAT at 1, UNEVEN at 10, NOT
  DECIDED at 2.5); the two-ratio rule's three outcomes, the energy floor among them; the disjointness refusal.
- Twenty-eight mutants of `ops/sensitivity_map.py`, each at one counted site in a scratch copy, the worktree file
  never touched, the copy restored and its sha256 checked after each. Twenty-seven are killed by the check they
  aim at: the depth-12 cut a block early; the answer rows reversed; the text loss averaged; labels by argmax; scores
  of logits; every score k = 0's; the four-way Fisher weighted by p; the text Fisher from h(11); C_ff at depth 13;
  both Fisher halves in one; the windows in one half; the pos0 and dir1 conventions broken (three ways); the band
  matrix always band 1; every arm in the variance basis; the self-patch writing FP16; the hybrid a block late; the
  ratio's energies swapped; UNEVEN on the upper end; FLAT without its upper end; the cross term adding d12;
  disjointness ignoring hashes; the orthonormality error zeroed; AM/GM inverted; the overlap over all columns; the
  variance control without its upper end. The twenty-eighth, a checkpointed block detaching its input, makes the
  selftest raise before its verdict line, so the job's `selftest: GREEN` gate refuses it.
- After the amendment, five more mutants, each killed by its check: FLAT without the energy floor; UNEVEN read on
  [1, 64] only; FLAT read on [1, 64] instead of [1, 512]; the wide arm reduced to the top band; the [1, 512]
  control without its lower end.
- `DRY_RUN=1 bash ops/jobs/sensitivity-map-4b.sh`: the job script parses; nothing launched.
- Seen before writing: patch-map-4b's journal and raw files (its shares and logits), contrast-map's depth-12 bands,
  the R6 notes. Not seen: any Fisher, any deviation energy by band.
Not run: the `--run` path. It has never executed; the card is its first test.

### Signed predictions (main session)

1. Every gate is green on the first launch: 50 %. The gradient pass has never run on a card.
2. The job runs 23 min [17, 32].
3. The variance control holds, both ratios inside their gates: 85 %.
4. `fisher_text`'s spectrum is steep: its top 64 directions hold 0.60 [0.35, 0.85] of its trace.
5. The sealed deviation is spread, not concentrated there: `fisher_text`'s [1, 64] holds 0.04 [0.01, 0.12] of the
   deviation energy (convention `dir1`), its [1, 512] 0.25 [0.12, 0.40].
6. `fisher_text`: R_64 = 4 [1.5, 12]. The verdict reads UNEVEN: 55 %. NOT DECIDED: 35 %. FLAT: 10 %.
7. `fisher_mmlu`: R_64 is above `fisher_text`'s: 75 %; its interval lies above 3: 75 %.
8. The top-64 subspaces of `fisher_text` and `fisher_mmlu` overlap by 0.35 [0.15, 0.60].
9. The top-64 subspace of `fisher_text` overlaps the variance basis's top 64 by under 0.5.

The lead's reason for 6. Curvature spectra of networks are steep, and a deviation that is spread over directions
should then cost far more a unit of energy where the Fisher is large. The variance bands read flat today, which
fits a sensitivity that is uneven and unrelated to the variance order as well as one that is even. What the lead
does not know is whether the directions running text marks as sensitive are those the MMLU answer needs.
