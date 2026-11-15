# Preregistration. Where the sealed 4B loses the contrast of its answer logits: a map by depth, direction and matrix

**Written on 2026-11-14, BEFORE the job. NOT YET STAMPED.** The signed predictions (§8) are filled by the main
session, then the file is stamped (`ots stamp`) and committed before `ops/jobs/contrast-map-4b.sh` launches. Once
stamped, this file is not edited; a deviation goes into `-ECARTS.md` beside it.

Script: `ops/contrast_map.py`, sha256 `95bb1e55583e2d647f0eb1ad062725e45397ea56bcf42d935b1ff342a1635b76`; it imports `ops/layer_vs_output.py`, sha256
`c031c52f1294f3e044621bf7ed3e7f5089da0fb593f0ced5ca826fbb67596f12` (unchanged since 0079d04). Job script:
`ops/jobs/contrast-map-4b.sh`, sha256 `5b651331feaaaafbc98a5f187613110d5047007033683b823228df04775c25af`. Both at commit `397887e`.

Cost: **$0.54 to $0.63** (*estimated*, §7), ceiling **$0.99** (timeout 33 min on `l40sx1`). Inside the $10 of O13, $0 spent
before it. The operator's go of 2026-11-14 ("Ok go") is on a proposal that named this diagnostic at under $1.

## 1. The question

The sealed 4B's four MMLU answer logits regress on FP16's with slope 0.5817 and a noise of 1.4205 a logit (anomaly N6,
`docs/mesures/recherche-qualite2-r0-contraste-2026-11-14.txt`). Where in the network does that contrast go, and is
what is lost a coherent distortion or noise? No encoder, file or score is produced: this is a reading of two models
side by side.

Three rival hypotheses (the note's R3 draft):
- **Ha**, the dull one: slope and noise are one variable. Internal noise, then normalisation, shrink every direction
  of the residual stream by one factor.
- **Hb**: each quantized matrix passes less than the full coherent signal, more so in the quiet directions, and the
  loss multiplies along the depth while the noise adds.
- **Hc**: noise in q and k flattens the attention patterns and dilutes what attention retrieves.

## 2. The objects

- FP16: `Qwen/Qwen3-4B@1cfa9a7208912126459214e8b04321603b3df60c`, shards as pinned in `ops/layer_vs_output.py`.
- Sealed: the f16 export of `qwen3-4b-sealed.bin`, `sealed-4b-export-2026-09-23/model.safetensors`, 8,044,981,648 B,
  the file the 2026-11-13 layer-vs-output job read. It carries the trained row scales, the int4 records and the q4
  embedding as dense f16 tensors.
- Text: MMLU prompts as `bin/mmlu` builds them (ported from `llvq-llm/src/bin/mmlu.rs`, lines 313 to 323 and 765 to 820; `cais/mmlu@c30699e8356da336a370243923dbaf21066bb9fe`), questions of the full split ordered round-robin over the
  57 subjects. N is fixed from the time of the first 57 prompts under a 600 s budget, at least 570 and at most 2,280;
  about 800 are expected.
  Both models read the same N prompts, each on its own activations.
- One torch process, f16 weights, image `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`, `l40sx1`.

## 3. Controls, and rule D

1. The script's selftest reads GREEN in the job before the models load.
2. Rule D. On the N questions the script's four answer logits, each model with its own head, are set beside the
   committed dumps: `mmlu-4b-f16-FULL.csv` for FP16 and `mmlu-4b-abl-control-FULL.csv` for the sealed 4B (63.37 on the
   dense f16 path). GREEN if the picks agree on at least 98 % of the questions for each model and the slope fitted on
   the script's logits differs from the slope fitted on the dumps' logits, same questions, by at most 0.02. The floor
   is 98 and not 99 because two backends on one f16 file agree on 99.30 to 99.61 % of picks in the repository's own
   dumps (the code agent's count, 2,280 q), and at N near 800 a 99 % floor fails by chance with a correct port.
3. The relative error per matrix of reading R4 reproduces `Note1` of `ops/layer_vs_output.py` on the same tokens
   (selftest), so R4's third column is the quantity of the 2026-11-13 journal on other text.

If control 1 or 2 fails, nothing is read: the prompt port or the forward is wrong, and that is the finding.

## 4. The readings

Notation: h_f(l) and h_s(l) are the residual streams of FP16 and sealed after block l (0 = embeddings, 36 = last
block), each on its own activations; the answer position is the one `bin/mmlu` reads.

- **R1, lens slope by depth.** z = U_ans · final_norm(h(l)) with FP16's final norm weight and FP16's four answer rows
  for both streams. Slope and noise of sealed on FP16, centred, pooled, at every depth.
- **R2, stream slope by variance band**, depths 0, 4, ..., 36. In the eigenbasis of FP16's second moment over all
  token positions, eigenvalues descending, bands [1], [2, 8], [9, 64], [65, 512], [513, 2560]: the slope of the
  sealed coordinate on the FP16 coordinate and the noise ratio, over all positions and over the answer positions.
- **R3, linear predictability**, same depths and bands: the share of the sealed stream's energy explained by the
  best linear map of the FP16 stream, beside the share explained by the diagonal slope alone.
- **R4, coherent gain of each matrix**: a = 1 + ⟨(Ŵ − W) x, W x⟩ / ‖W x‖² and its noise ratio, pooled and by output
  row, on FP16 inputs and on sealed inputs, for the 252 projections.
- **R5, attention at the answer position**: entropy and largest weight by layer and head, both models.

## 5. What each hypothesis predicts, and the rules, written before the data

Primary numbers, all at depth 36 on the answer positions unless said otherwise:
- `s_loud` = R2's energy-weighted slope of band [9, 64]; `s_quiet` = of band [513, 2560].
- `a_med` = the median over the 252 matrices of R4's row-median coherent gain on FP16 inputs.
- `c` = the slope, across the 252 matrices, of (1 − a pooled) on the relative error S_ee / S_yy, FP16 inputs. A
  reconstruction at the MMSE point gives c = 1, a norm-preserving one c = 1/2, an unbiased one c = 0.
- `dH` = the mean over heads of R5's entropy, sealed minus FP16, in nats.

Rules:
- **Ha is supported, and Hb is dead**, if `s_loud − s_quiet` < 0.10: the loss does not depend on how loud a direction
  is.
- **Hb is supported** if all three hold: `s_loud − s_quiet` ≥ 0.15; `s_quiet` at depth 36 is at least 0.10 under
  `s_quiet` at depth 12; and `a_med` < 0.98. If the first two hold and `a_med` ≥ 0.99, the loss is by direction and
  by depth but no matrix shrinks on its own inputs: Hb is dead in its local form, and what remains is a compounding
  that the per-matrix reading does not see.
- **Hc is supported** if `dH` > 0.05 and at least 60 % of the heads have a higher entropy in the sealed model; **dead**
  if |`dH`| < 0.02.
- **Coherent or noise** (R3, depth 36, band [513, 2560], all positions): "mostly coherent" if the linear share is at
  least 0.5 and exceeds the diagonal share by at least 0.2; "mostly noise" if the linear share is under 0.3. The
  linear map is solved with a ridge of 1e-6 of the mean eigenvalue; where `linear.tsv` shows the ridge above the
  band's smallest eigenvalue, the rule is read on band [65, 512] and the quiet band is reported as not decided.
- Anything between two thresholds is reported as not decided. The hypotheses are not exclusive: Hb and Hc can both
  be supported.

The thresholds are judgement, fixed here so that the reading cannot move them. They are not confidence intervals:
the readings are sums over 10^5 to 10^6 tokens, and their sampling error is small beside the thresholds.

## 6. What does not follow, whatever the outcome

- No MMLU point is gained or projected. A supported Hb says where to look for a zero-bit correction; the correction
  is another prereg and another go.
- A slope under 1 in a direction is not a loss of information by itself: R3 exists to tell a remix from noise.
- One model size, one calibration draw, one text (MMLU prompts). Nothing here says the map holds at 8B or 14B.
- The lens of R1 reads early depths through the final readout; an early slope is a statement on that readout only.
- A higher contrast is not a higher score by itself. On Qwen3-4B, Wu et al. (arXiv 2608.06564, App. A.14, *reported*)
  protect nine layers by margin sensitivity: the surviving fraction of the margin reads 0.44 against 0.24 for a
  cost-matched control, and the flip rate 0.43 against 0.46. Here, restoring q, k and v raised the slope from 0.58 to
  0.70 for +0.55 MMLU (the R0 note). The phase's question is the ratio of contrast to noise.

## 7. Cost

- One job on `l40sx1` at $1.80/h. Setup as the 2026-11-13 job (inputs and load 104 s, hashes 25 s, *measured* there),
  the prompt loop under a 600 s budget at 0.7 to 0.8 s a question (*estimated* from the mean prompt length, 605
  tokens), eigendecompositions and report about 3 min (*estimated*): 18 to 21 min billed, $0.54 to $0.63.
- Timeout 33 min, ceiling $0.99. Queue unbilled. Spend of the phase before: $0 of $10.

## 8. Dated disclosure, and the signed predictions

What has run, as of writing, all on the Mac, no model loaded:
- `ops/contrast_map.py --selftest`: GREEN, 41 checks, on a tiny random model on CPU. It plants a matrix at 0.9 W plus
  noise and recovers a coherent gain of 0.9 with the `Note1` relative error; it recovers planted band slopes, noise
  ratios and linear shares on synthetic moments, with a pure rotation inside a band read as a low diagonal slope and
  a linear share of 1; it recovers a planted lens slope of 0.6.
- Six mutants, each at one counted site, each caught, each restored (sha256 checked): the sign of the cross term;
  the slope read from C_ss; the bands shifted by one; the lens under each model's own final norm; the answer position
  off by one; the ridge dropped on a singular moment.
- `--check-prompts`, from the local dataset cache: the port builds the 14,042 prompts of the full split and their
  token-id hashes equal the `qhash` column of both dumps on 28,084 of 28,084 rows; the census fingerprint is
  `a74a6d6213602979`, the dumps' own. On the card the run repeats the check before loading a model.
- `DRY_RUN=1` of the job script. One real launch attempt of the job script by the code agent, refused at its first
  guard (no `.ots`) before any network call: nothing was submitted.
- The R0 notes of the phase: the slope and noise of 24 dumps, and the literature half.
Not run: the `--run` path. It has never executed; the card is its first test. No reading of this prereg exists.

Known before the data:
- The sealed 4B's output slope, 0.5817, and noise, 1.4205 (N1 journal and the R0 note).
- The relative layer-output error by type on WikiText-2 (layer-vs-output journal, added section): up_proj 0.0337,
  gate_proj 0.0046, the others between.
- The published counter-fact of §6.

### Signed predictions (main session)

Signed by the main session on 2026-11-14, before the stamp and before the job. The lead's view: the MMSE identity
should show in R4 (a close to 1 minus the relative error), the stream's quiet directions should be attenuated more
than its loud ones, and attention should be only slightly flatter. The lead is least sure that the per-matrix shrink
is large enough to compound to 0.58.

| quantity | prediction | interval or alternatives |
|---|---|---|
| rule D | GREEN | about 85 % |
| N, questions read | 800 | [570, 1,200] |
| R1, lens slope at depth 36 | 0.60 | [0.52, 0.68] |
| R1, first depth where the lens slope is under 0.8 | 20 | [10, 32] |
| R2, pooled slope tr(C_sf)/tr(C_ff), depth 36, all positions | 0.97 | [0.92, 0.995] |
| `s_loud`, band [9, 64], depth 36, answer positions | 0.95 | [0.85, 0.995] |
| `s_quiet`, band [513, 2560], depth 36, answer positions | 0.75 | [0.50, 0.93] |
| `s_quiet` at depth 12, answer positions | 0.92 | [0.75, 0.99] |
| `a_med` | 0.985 | [0.96, 0.999] |
| R4, pooled coherent gain of up_proj as a type, FP16 inputs | 0.97 | [0.94, 0.995] |
| `c` | 0.9 | [0.5, 1.2] |
| `dH`, nats | +0.03 | [−0.02, +0.12] |
| share of heads with higher entropy in the sealed model | 60 % | [45, 80] |
| R3, linear share, band [513, 2560], depth 36 | 0.55 | [0.30, 0.80] |
| R3, diagonal share, same band and depth | 0.40 | [0.20, 0.65] |
| Ha supported (Hb dead) | no | about 20 % that it is |
| Hb supported, all three conditions | no | about 30 % that it is; a mixed reading about 50 % |
| Hc supported | no | about 35 % that it is |
