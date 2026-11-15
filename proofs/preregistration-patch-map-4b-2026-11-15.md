# Preregistration. The patch map: which part of FP16's stream gives back the sealed 4B's MMLU deficit

Written on 2026-11-15 on branch `recherche/qualite-n6-n7-2026-11-15`, before any job, by an executing agent and
the lead, whose review is folded in (amendments 1 to 3) and who signs §11. Timestamped before the first reading,
then never edited; a departure goes to `preregistration-patch-map-4b-2026-11-15-ECARTS.md`. After the stamp and
before the job, the instrument check of §6 runs at $0. The job, `ops/jobs/patch-map-4b.sh`, waits for the
operator's budget and go.

Script: `ops/patch_map.py`, sha256 `a531bf4f14ab6517f0d3e61192a2e1068d9dbcafbd3a4d39abd706ebbc7e709f`. It imports `ops/contrast_map.py`, sha256
`936fa6cdb4b39b5d0d4d5aa1de1d265a3f11fe55d0c46e89a7dc60612fc94a20` (887711c; its forward is unchanged since 397887e, the commit contrast-map-4b ran),
and `ops/layer_vs_output.py`, sha256 `c031c52f1294f3e044621bf7ed3e7f5089da0fb593f0ced5ca826fbb67596f12`. Job script:
`ops/jobs/patch-map-4b.sh`, sha256 `e12509ef26bec0db37fed77195af35f2c385120b8c11fbb21c624af5bfc12c36`. They are committed with this
file, and the job refuses to launch if the stamped prereg does not name these sha256.

Cost: **$0.42 to $0.84** (*estimated*, §8), ceiling **$1.20** (timeout 40 min on `l40sx1`). No budget is signed for it
yet; the frame (`docs/mesures/recherche-qualite3-r1-cadrage-2026-11-15.txt` §4) puts it after an operator's go.

## 1. The question

On 570 MMLU questions the sealed 4B answers 66.32 % where FP16 answers 73.51 % (contrast-map-4b, *measured*), a gap
of 7.19 points (*computed*). The deficit exists at depth 25, where the answer is written. Which part of the stream
before depth 25 carries it: which depth, which band of directions, which positions, which sublayer? The frame's
number is the share of the accuracy gap recovered when a named part of FP16's stream replaces the sealed one and the
sealed blocks run on. At 570 questions that share resolves to about ±0.51 (§7), so the rules are read on the cosine
share of §5, and the accuracy share is reported beside it. FP16's first 25 blocks recover 6.12 of 6.72 points on the
full split (depth-restore-4b, *measured*).

The five rivals of the frame (§2): H6-A, the quiet band is a symptom; H6-B, the matrices err most on answer-position
inputs; H6-C, massive activations; H6-D, single heads at layers 7 to 13 lose their focus; H6-E, the MLP of the first
blocks writes the band noise.

## 2. The objects

As contrast-map-4b: FP16 `Qwen/Qwen3-4B@1cfa9a7208912126459214e8b04321603b3df60c`; the f16 export of
`qwen3-4b-sealed.bin`, `sealed-4b-export-2026-09-23/model.safetensors`, 8,044,981,648 B; MMLU prompts as `bin/mmlu`
builds them at `cais/mmlu@c30699e8356da336a370243923dbaf21066bb9fe`; the first 570 of the round robin over the 57
subjects, checked row for row against contrast-map's `questions.tsv`. One torch process, f16 weights, image
`pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`, `l40sx1`.

Depths follow contrast-map: h(0) is the embedding output, h(d) the stream after block d. A patch at depth d writes
into the sealed run's h(d); the sealed blocks d to 35, the sealed final norm and the sealed head (its q4 embedding,
tied) follow. The lens is contrast-map's: FP16's final norm weight and FP16's four answer rows, read on h(25) at the
answer position (the last prompt token). The band basis is the eigenbasis of FP16's uncentred second moment of h(12)
over every position of the 570 prompts, descending, computed in a first pass.

## 3. The arms

| arm | where | positions | what replaces the sealed run's |
|---|---|---|---|
| `fp16`, `control` | none | none | the two models as they are |
| `d4`, `d8`, `d12`, `d16`, `d20` | h(4) to h(20) | every position | the whole stream |
| `d12_b1-64`, `d12_b65-512`, `d12_b513-2560` | h(12) | every position | a band: h_s + (h_f - h_s) V_b V_bᵀ |
| `d12_ans`, `d12_notans`, `d12_massive` | h(12) | answer alone; all but it; massive alone | the whole stream |
| `d12_b513-2560_ans` | h(12) | answer alone | band [513, 2560] |
| `attn_explicit` | blocks 7-13 | answer row | nothing: attention's answer row recomputed as softmax(q kᵀ / sqrt(d_h)) v in float64 |
| `attn_fp16_7-13` | blocks 7-13 | answer row | FP16's answer-row attention probabilities, times the sealed run's values |
| `attn_0-11` | blocks 0-11 | every position | the attention branch's output (o_proj) by FP16's at that block |
| `mlp_0-11` | blocks 0-11 | every position | the MLP branch's output (down_proj) by FP16's at that block |

The two branch arms start from the sealed embedding. FP16's branch outputs come from FP16's run of the same prompt.

Massive positions are fixed from FP16 alone: a position is massive if its FP16 h(8) norm exceeds 10 times the median
norm over the prompt's positions; the answer position is never massive. The job prints how many a prompt holds, their
token ids, position 0's share, how often the answer position would qualify, and the counts at multiples 5, 20 and 50.

Beside the arms, with no patch: each of the 252 matrices' relative error S_ee / S_yy (contrast-map's R4, products in
float32, sums in float64) on three masks of its inputs: every position, the answer position, every position but the
massive ones; on FP16 inputs (pass 1) and on the control's inputs (pass 2).

## 4. Gates

Every gate is printed as it is read. If one is red, the report says so and nothing below it is read.
1. Both selftests GREEN in the job before a model loads (`contrast_map.py`, 53 checks; `patch_map.py`, 48 checks).
2. The 570 questions are contrast-map's, row for row (subject, index, token-id hash).
3. Rule D, contrast-map's: picks of FP16 and of the control equal the Rust dumps' (`mmlu-4b-f16-FULL.csv`,
   `mmlu-4b-abl-control-FULL.csv`) on at least 98 %, and the control-on-FP16 slope fitted on the script's logits is
   within 0.02 of the slope fitted on the dumps'.
4. Contrast-map reproduced: FP16 reads 419 of 570 right (73.51 %) and the control 378 (66.32 %), each within 2
   questions, and each model's picks equal contrast-map's `control.tsv` script picks on at least 99 %.
5. Exactness, bit for bit, on the first 57 prompts. (a) Patching the control's own h(12) into itself, by the whole
   stream, by band [513, 2560], on the massive positions and on the answer position through the band path, changes
   neither h(12) nor the logits. (b) The whole-stream patch at every position at each depth d of 4, 8, 12, 16, 20
   equals FP16's embedding and blocks 0 to d-1 followed by the sealed blocks, at every depth after d and at the
   output, and its logits equal the main loop's. (c) FP16's h(0) with both FP16 branches at blocks 0 to 11 in the
   sealed run gives FP16's h(12) and `d12`'s logits. This is bit for bit, because the stream is FP16's own sum of the
   same tensors in the same order. (d) FP16's logits in pass 2 equal pass 1's on all 570 prompts.
6. Tolerance controls. The three bands patched together as one basis give the whole-stream arm's pick on at least
   98 % of the 57 prompts. The explicit attention path with no patch gives the control's pick on at least 98 % of the
   570. The largest differences are printed.
7. The 252 all-position relative errors equal contrast-map's `matrices.tsv` on both inputs to a relative 1e-3.
8. The bases are unchanged by sha256 after the loop.

Printed, not gated: FP16's and the control's lens accuracy at depth 25 against contrast-map's 72.28 % and 63.86 %;
the control's answer-position slope of band [513, 2560] at depth 12 against 0.4697; the depth-12 energy shares
against contrast-map's `stream_bands.tsv`.

## 5. The readings

The primary statistic of every arm X. Per question i, c_i(X) is the cosine between X's four output logits and FP16's,
each centred on its own mean over the four options. It is scale-free, so the last block's rescale does not enter; it
is 0 when a centred vector is all zeros, and those cases are counted. C(X) is the mean of c_i over the 570 questions,
and C(fp16) = 1. The share recovered is s(X) = (C(X) - C(control)) / (1 - C(control)). The same is read through the
lens at depth 25, on the four lens logits against FP16's.

Every interval is a 95 % paired bootstrap: 10,000 resamplings of the 570 questions, seed 20261115, percentile.
Reported beside the primary for every arm, with intervals and no rule: the accuracy share (acc_X - acc_control) /
(acc_fp16 - acc_control), underpowered at about ±0.51 of the gap (§7); the difference to the control with b, c, the
paired normal interval d ± 1.96 sqrt(b + c - (b - c)² / n) / n, and McNemar's exact p; the mean KL over the four
options from FP16 to the arm; the mean log-probability the arm gives FP16's pick; the share of picks changed; the
answer-position slope at depth 12 of each band, energy-weighted as contrast-map's R2, for every arm, the branch arms
included. Per type and per matrix: the answer-position and non-massive relative errors over the all-position one.
Per question: every arm's pick and lens pick, four output logits and four lens logits.

## 6. The instrument check, and the kill rules

**Instrument check, at $0, after the stamp and before the job.** `uv run ops/patch_map.py --instrument` reads eleven
full-split dumps by pinned sha256 in `docs/data/mmlu-dumps/`. The FP16 reference is `mmlu-4b-f16-FULL.csv` and the
control `mmlu-4b-abl-control-FULL.csv`. The nine restore arms are `mmlu-4b-abl-{qkv,o,gateup,down,attn,mlp,all}-FULL.csv`,
`mmlu-4b-depth-late-FULL.csv` and `mmlu-4b-depth-early-FULL.csv`. For each restore arm it computes s(X) and the
accuracy share on the 14,042 questions, and prints them side by side with the largest absolute difference. Rule: if
the Spearman rank correlation of the two columns over the nine arms is at least 0.9, the cosine share is the primary.
If not, the primary reverts to the accuracy share, and H6-A, H6-C (i) and H6-E are read as underpowered. The job
takes the outcome as `PRIMARY=cosine|accuracy`, and the report prints the other share's ratios beside, with no rule.

**Kill rules.** A rule reads KILLED only when the whole 95 % interval lies on the kill side, STANDS when it lies
wholly on the other side, NOT DECIDED otherwise. s is the primary share. A ratio over s(d12) is read only when
s(d12)'s own interval lies above 0. If that interval reaches 0, the whole-stream patch at depth 12 has not been shown
to recover anything, and the rule reads NOT DECIDED.

| rival | statistic | killed if | stands if | what follows |
|---|---|---|---|---|
| H6-A, the quiet band is a symptom | s(`d12_b513-2560`) / s(`d12`) | interval at or above 0.5 | interval under 0.5 | killed: the quiet band carries the deficit; the next step is a conjecture on the encoder's weighting, broken before any encode. Stands: the quiet band leaves N6; the phase looks at who writes the middle band |
| H6-B, a position mismatch of the matrices | per type, pooled over its 36 layers, FP16 inputs: relative error on answer-position inputs over its all-position value | all seven intervals inside [1/1.5, 1.5] | one interval wholly above 1.5 | killed: the four-times-harder hit at the answer position is made by propagation. Stands: the encoder's position-blind H is the first suspect |
| H6-C, massive activations | (i) s(`d12_massive`) / s(`d12`); (ii) pooled relative error of gate_proj 1, 2, 3 and down_proj 6 (the four under 0.002 in `matrices.tsv`), FP16 inputs, without the massive positions over with them | (i) under 0.1 and (ii) inside [1/2, 2] | (i) at or above 0.1, or (ii) wholly at or above 2 or at or under 1/2 | killed: no handful of rows holds the deficit. Stands: the idea is about a handful of rows, not a weighting |
| H6-D, single heads at 7 to 13 | (s_attn - s_base) / (1 - s_base), the depth-12 answer-position slope of band [513, 2560], base `attn_explicit` | interval under 0.2 | interval at or above 0.2 | killed: the band loss at depth 12 is not the heads' focus. Stands: the heads at 7 to 13 are read next by layer |
| H6-E, the early MLP writes the band noise | s(`mlp_0-11`) / s(`d12`) | interval under 0.5 | interval at or above 0.5 | stands: the MLP of the first twelve blocks writes most of what depth 12 costs; layers 4 to 11 are read next by layer |

H6-D's slope is Σ a / Σ e with a_i = Σ_k (y V)_ik (x V)_ik and e_i = Σ_k (x V)_ik² over the band, x FP16's and y the
arm's answer row of h(12). Printed beside, no rule: H6-D against the SDPA control; s(`attn_0-11`) / s(`d12`), which
need not sum with H6-E's ratio to 1. Descriptive, no rule: the depth profile d4 to d20, `d12_ans` against
`d12_notans`, `d12_b513-2560_ans`. No outcome is a mechanism by itself: a patch is a probe.

## 7. Power

On these 570 questions contrast-map-4b reads 79 questions FP16 gets right and the sealed 4B wrong, and 38 the reverse,
139 picks different (`control.tsv`, *computed*). The gap of +7.19 points has a paired standard error of 1.87 points and
a 95 % half-width of 3.67 points (*computed*): the gap itself is known to [+3.52, +10.86]. An arm against the control
with D discordant questions has a 95 % half-width of at most 1.96 sqrt(D) / 570: 3.7 points at D = 117, 2.7 at D = 60,
1.9 at D = 30 (*computed*). In shares of the gap, an arm at the FP16 discordance is resolved to about ±0.51, so ratio
rules on the accuracy share would mostly read NOT DECIDED. The cosine share uses every question's four logits, and
H6-B, H6-C (ii) and H6-D are sums over thousands of coordinates or tokens; their intervals come from the data.

## 8. Cost

Measured figures it rests on:
- contrast-map-4b (job 6af86d01, `l40sx1`): 918 s billed; load 95 s; one base hash pass 24 s; tokenization 12 s; the
  loop 705 s for 570 prompts, 1.24 s a prompt, with both models and every tap.
- lens-write-depth 8B (job 6af888e7, `l40sx1`): a plain f16 forward with one stream tap, 2,280 prompts in 160 s,
  0.070 s a prompt. The 4B has fewer weights; 0.070 s is used as the 4B's upper bound (*estimated*).

One patched forward of 570 prompts costs at most 570 × 0.070 = 40 s for 36 blocks, 27 s for a depth-12 arm (24
blocks) (*estimated*). Per prompt (*estimated*): pass 1 is one FP16 pass with the error tap and the C_ff tap, at most
0.62 s (half of contrast-map's tapped pair), 353 s for 570. Pass 2 is the control with the error tap (at most 0.62 s)
and 454 untapped blocks: FP16 36, depth profile 120, seven depth-12 arms 168, two attention arms 58, two branch arms
72. That is 12.6 forwards, at most 0.88 s, so pass 2 is at most 857 s. The controls on 57 prompts add 12.7 forwards a
prompt, 51 s. Setup about 4 min; analysis about 2 min. Total 14 to 28 min billed, $0.42 to $0.84 at $1.80/h. Timeout
40 min, ceiling $1.20. The script stops itself with exit 5 if pass 1 projects over 600 s after 57 prompts, or pass 1
plus pass 2 over 1,800 s after 114 prompts of pass 2; nothing is read then. The instrument check costs $0.

## 9. What does not follow, whatever the outcome

- No MMLU point is gained or projected; no encoder is changed. A share says where the deficit rides, not how to code it.
- A patch at depth 12 is read through 24 sealed blocks: it measures what the sealed blocks do with FP16's part.
- One size, one calibration draw, one text, 570 questions easier than the full split.
- The band basis is the all-position one; at the answer position band [513, 2560] holds 2.9 to 6.3 % of the energy
  against 0.6 to 1.4 % over all positions (R0 note, §8 point 5).
- A share in cosine is not a share in points. The instrument check asks only that the two rank the restores alike.
  A ratio of two cosine shares at 0.5 is not shown to be a ratio of two accuracy gains at 0.5; the accuracy shares
  are printed for that, with their wide intervals.

## 10. Choices this draft makes, as the lead accepted or amended them

1. Massive positions: FP16 h(8) norm above 10 times the prompt's median, answer position excluded (accepted).
2. H6-B two-sided, inside [1/1.5, 1.5] for every type, on FP16 inputs (accepted).
3. H6-C (ii) pools the four matrices under 0.002 and is two-sided, inside [1/2, 2] (accepted).
4. H6-D's deficit is the depth-12 answer-position slope of band [513, 2560], base the explicit path (accepted).
5. The FP16 attention patch writes FP16's probabilities only; the values are the sealed run's own (accepted).
6. The three-way reading by the whole 95 % interval (accepted).
7. The primary statistic is the cosine share of the four centred logits, subject to the instrument check; the
   accuracy share is reported (amendment 1).
8. H6-E enters with `attn_0-11` and `mlp_0-11`; branch outputs are replaced at every position, from the sealed
   embedding (amendment 2).
9. N stays at 570, contrast-map's, so that every gate has a reference (accepted).

## 11. Dated disclosure, and the signed predictions

What has run, as of writing, all on the Mac, no model loaded, no download but the cached torch wheel:
- `uv run ops/patch_map.py --selftest`: GREEN, 48 checks, tiny random models on CPU in float64. `uv run
  ops/contrast_map.py --selftest`: GREEN, 53 checks, the file unchanged.
- Twenty-one mutants of `ops/patch_map.py`, each at one counted site in a scratch copy, each caught, each restored
  with the sha256 checked: the resume loop skipping its first block; the band patch without "- h_s"; band columns
  shifted by one; the answer position allowed massive; the explicit softmax without its scale; the lens read at depth
  24; arms patched with the sealed stream; the answer mask on position 0; "all but the answer" as the answer; the
  paired interval without its (b - c)² / n term; three bands built from two; arms resumed a block late; massive norms
  read at depth 7; the FP16 attention arm writing its own probabilities; the answer row read one position early; the
  shape cosine without centring; branch arms stopping at block 10; the branch replacer ignoring the branch kind; the
  s(d12) guard never firing; ranks without tie averaging; the KL weighted by the arm instead of FP16.
- The instrument mode on built inputs only (a planted mixture alpha × FP16 + noise, its share growing with alpha;
  Spearman 1.0). It has not read the real dumps; it waits for the stamp. Only the header and trailer lines of the eleven
  dumps were read, to pin their sha256.
- The reference files of contrast-map read by the script's own reader: 570 questions, 419 and 378 right, the four
  matrices under 0.002 as named in §6.
- `DRY_RUN=1 bash ops/jobs/patch-map-4b.sh`: the job script parses; nothing launched.
Not run: the `--run` path. It has never executed; the card is its first test.

Seen by the lead before the stamp, and bearing on the instrument check: the pooled slope and residual sd of the nine
restore arms against FP16 (`recherche-qualite2-r0-contraste` and `recherche-qualite3-dumps`, M1), from which their
order by slope over noise is known. Not seen: any cosine share.

### Signed predictions (main session)

1. The instrument check passes, Spearman at least 0.9: 85 %. The largest absolute difference between the two shares
   is above 0.15, on `down`.
2. The job's gates all pass on the first launch: 65 %. It is the first execution of the `--run` path.
3. The job runs 20 min [14, 30].
4. Depth profile, s of the whole-stream patch: d4 0.10 [0.03, 0.25], d8 0.25 [0.12, 0.45], d12 0.45 [0.28, 0.65], d16
   0.60 [0.42, 0.78], d20 0.75 [0.58, 0.90]. It rises at every step.
5. H6-A: s(`d12_b513-2560`) / s(`d12`) = 0.55 [0.30, 0.80]. KILLED 30 %, STANDS 25 %, NOT DECIDED 45 %.
   s(`d12_b65-512`) / s(`d12`) = 0.40 [0.20, 0.60]. s(`d12_b1-64`) / s(`d12`) = 0.05 [-0.05, 0.20].
6. Positions: s(`d12_notans`) is above s(`d12_ans`): 80 %. s(`d12_ans`) / s(`d12`) = 0.15 [0.05, 0.35].
7. H6-C: (i) s(`d12_massive`) / s(`d12`) = 0.03 [-0.05, 0.15]; (ii) the four matrices' pooled error without the
   massive positions is at least twice the value with them: 75 %. H6-C reads STANDS, by (ii): 70 %.
8. H6-B: the median over the seven types of the answer-position error over the all-position one is 1.4 [1.0, 2.2].
   KILLED 40 %, STANDS 35 %.
9. H6-D: the share is 0.08 [-0.02, 0.25]. KILLED 60 %.
10. H6-E: s(`mlp_0-11`) / s(`d12`) = 0.75 [0.50, 0.95]. STANDS 60 %. s(`attn_0-11`) / s(`d12`) = 0.25 [0.05, 0.50].
11. The control's and FP16's accuracies reproduce contrast-map's 378 and 419 within 2 questions: 90 %.

The lead's reason for 5: at depth 12, over all positions, the noise energy of the stream sits mostly in band
[513, 2560] and then in [65, 512] (*computed* by the lead from report.txt line 61, share times noise ratio:
about 0.005 and 0.002 of the stream's energy, against 0.0004 for bands [2, 64]). If damage follows noise energy, the
quiet band carries about two thirds of what depth 12 costs. That is the reading the job tests.
