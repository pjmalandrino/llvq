# Preregistration. GSM8K on the 4B with the blocks after the answer write restored, then the blocks up to it

**Written on 2026-11-14, BEFORE the job. NOT YET STAMPED.** The signed predictions (§7) are filled by the main
session, then the file is stamped (`ots stamp`) and committed before `ops/jobs/gsm8k-depth-restore-4b.sh` launches.
Once stamped, this file is not edited; a deviation goes into `-ECARTS.md` beside it.

Scripts, at commit `a48190e`: `ops/hybrid_checkpoint.py`, sha256
`1e5b795c5eb5faf88297ef65aa77791889162ac5fb2c1600b14b9b527d76661f`; `ops/gsm8k_vllm_local.py`,
`daadd780d93aa0527943f0d2563c89f3a9c3104e59d5422c71d9549f792b59b4`, which imports `ops/gsm8k_vllm.py`,
`907ee931e0e2e2deda8cb9cd5e7c8bd7d9e1ed329329d9894a985b45c996c33b` (unchanged since the 2026-09-26 campaign), and
changes only where the tokenizer and the weights are read from; `ops/layer_vs_output.py`,
`c031c52f1294f3e044621bf7ed3e7f5089da0fb593f0ced5ca826fbb67596f12`, imported for its checkpoint pins. Job script:
`ops/jobs/gsm8k-depth-restore-4b.sh`, `a0d61b49f377405f33affb55e18542bb89a4fa6aac99c06c0aefd1c4e0a8f101`. Image
`vllm/vllm-openai:v0.26.0` at the digest `ops/jobs/gsm8k-vllm.sh` pins. The served dump the prompts are checked
against: `docs/mesures/gsm8k-wave1-2026-09-26-brut/run1-served/gsm8k-4b-served.jsonl`, sha256
`b8d06615bd6fbc781baf33b027cd82140fe6c83cc7197b4b3f5c54756289e165`.

Cost: **$0.39 to $0.57** (*estimated*, §6), ceiling **$1.05** (timeout 35 min on `l40sx1`). Inside the $10 of O13, $2.17 spent before it. The
operator's go of 2026-11-14 ("Ok") is on this check announced at $2.13.

## 1. The question

On MMLU, restoring layers 25 to 35 of the sealed 4B to f16 gains +0.43 [+0.11, +0.77] and layers 0 to 24 gain +6.12
(`docs/mesures/depth-restore-4b-2026-11-14.txt`, CERVEAU Q25). MMLU reads one token. Do the late blocks carry as
little of the loss when the model writes a worked answer?

On GSM8K the sealed 4B reads 82.49 through the served kernel and FP16 92.12 in vLLM, a gap of 9.63
(`docs/mesures/gsm8k-wave1-2026-09-26.txt`). By the even rate, the late blocks, 11 of 36, are worth 9.63 × 11/36 =
**+2.94**, and the early blocks 9.63 × 25/36 = +6.69 (*computed*).

## 2. The objects, and a path that is not the MMLU probe's

`bin/gsm8k` has no restore, and its dense path takes 157 min for one 4B arm. The restore is therefore made on files
and scored in vLLM, the engine of the FP16 reference of 2026-09-26.

- **control**: the f16 export of the sealed 4B, `sealed-4b-export-2026-09-23/model.safetensors`, 8,044,981,648 B,
  sha256 `2a018cbd…`, as it is.
- **late**: the export with the seven projection tensors of layers 25 to 35 replaced by those of
  `Qwen/Qwen3-4B@1cfa9a7208912126459214e8b04321603b3df60c` cast to f16: 77 matrices, 1,110,179,840 weights.
- **early**: the same for layers 0 to 24: 175 matrices, 2,523,136,000 weights.
- GSM8K as on 2026-09-26: the 1,319 problems of the test split at `740312ad`, zero-shot, Qwen3 chat template,
  reasoning block pre-filled empty, greedy, 1,024 new tokens at most, prompts handed to vLLM as token ids. One
  grader, `bin/gsm8kpair`, re-grades every row.

## 3. Controls

1. Each hybrid file is read back after it is written: every replaced tensor is bit-equal to the FP16 tensor cast to
   f16, every other tensor is bit-equal to the export's, the counts are those of §2, and no key is added or lost.
2. The prompts equal those of the 2026-09-26 served dump, problem by problem, before any generation.
3. Rule D, the path. The control arm is the sealed 4B's dense f16 reconstruction in another engine than the served
   kernel. It is paired with the served dump of 2026-09-26 (82.49). GREEN if the absolute paired difference is at
   most 2.5 points. If it is RED, the export in vLLM is not a stand-in for the served object on this task: the two
   restore arms are still paired with the control and reported, and the result is not carried to the served file.

## 4. The statistic

G_late and G_early: accuracy of each restore arm minus the control's, paired on the 1,319 problems by `gsm8kpair`,
with its 95 % interval.

## 5. Decision rules, written before the data

On G_late:
- **THE PATTERN HOLDS ON GENERATION** if the upper end of the interval is under +2.94, the even-rate figure.
- **IT DOES NOT** if the interval contains +2.94 or lies above it.

A pair on 1,319 problems has a half-width of 1.3 to 1.9 points (the 2026-09-26 journals), so the rule can only fire
for a late gain near +1 or under. G_early is read beside +6.69; the rate per billion restored weights of the two
windows is reported, with no rule on it.

What follows:
- If the pattern holds on both instruments, the idea that bits could follow the depth of the write is worth a cost
  note: it needs a tier under the present rate for the late blocks, which the format does not have.
- If it does not, the late blocks matter to generation, the MMLU reading does not transfer, and no allocation by
  depth is proposed.

One model, one calibration draw, one generation task, greedy.

## 6. Cost

- One job on `l40sx1` at $1.80/h: setup 2 to 3 min, the control's directory 1 min, the two hybrids 1.5 to 2.5 min
  each (16 GB hashed, 8 GB written and read back), three vLLM arms of 2.5 to 3.5 min (on 2026-09-26 start-up took
  about 80 s and generation 70 s): 13 to 19 min billed, $0.39 to $0.57 (*estimated*).
- Timeout 35 min, ceiling $1.05. Queue unbilled. Spend of the phase before: $2.17 of $10.

## 7. Dated disclosure, and the signed predictions

What has run, all on the Mac, no model loaded:
- `ops/hybrid_checkpoint.py --selftest`: GREEN, 30 checks. The windows are inclusive and 0-based and give 77 and 175
  matrices at the 4B's shapes; the cast from BF16 to f16 equals, on all 65,282 non-NaN patterns, a line-by-line port
  of the cast the Rust f16 restore of `bin/mmlu` uses; the read-back verification catches a wrong layer, a wrong or
  extra type, an altered tensor on either side, a skipped cast, a dropped or added key and a wrong count. Five
  mutants, each at one counted site, each caught.
- `ops/gsm8k_vllm_local.py --check` against the served dump of 2026-09-26: 1,319 of 1,319 prompts identical, run
  fingerprint `bfa9135ce4b55c75`. Its generation call was checked against a stand-in for the `vllm` module.
- `gsm8kpair` built once; run on the wave's FP16 vLLM dump against the served dump it reproduces the published +9.63
  [+7.69, +11.57], first argument minus second.
- `DRY_RUN=1` of the job script.
Not run: vLLM on a local directory, either hybrid build at full size, any generation. No GSM8K score of a depth
window exists. The control arm's tensor check is tautological, its file being a hard link to the export: what
identifies it is its sha256.

Known before the data: the MMLU restores of the two windows; the lens reading of the 4B, where the answer token of
MMLU is written at depth 25 of 36. Nothing on generation by depth.

### Signed predictions (main session)

Signed by the main session on 2026-11-14, before the stamp and before the job. The lead's view: a generated token
should be decided at about the same depth as the MMLU answer, so the late blocks should again carry less than their
share; the lead is less sure than on MMLU, because the late blocks may shape the exact digits a worked answer needs.

| quantity | prediction | interval or alternatives |
|---|---|---|
| control accuracy | 82.6 | [80.0, 85.0] |
| rule D | GREEN | about 85 % |
| G_late (pp) | +1.2 | [−0.8, +3.6] |
| G_early (pp) | +7.8 | [+5.5, +10.0] |
| outcome on G_late | THE PATTERN HOLDS | about 65 % |
| early arm accuracy | 90.4 | [88.0, 92.6] |
