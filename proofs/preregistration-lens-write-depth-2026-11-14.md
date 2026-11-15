# Preregistration. Where the original Qwen3-8B and Qwen3-14B write their MMLU answer: a reading of the original alone

**Written on 2026-11-14, BEFORE either job. NOT YET STAMPED.** The signed predictions (§6) are filled by the main
session, then the file is stamped (`ots stamp`) and committed before `ops/jobs/lens-write-depth.sh` launches a size.
Once stamped, this file is not edited; a deviation goes into `-ECARTS.md` beside it.

Script: `ops/contrast_map.py`, sha256 `936fa6cdb4b39b5d0d4d5aa1de1d265a3f11fe55d0c46e89a7dc60612fc94a20` (its
`--lens-only` mode); it imports `ops/layer_vs_output.py`, sha256
`c031c52f1294f3e044621bf7ed3e7f5089da0fb593f0ced5ca826fbb67596f12`. Job script: `ops/jobs/lens-write-depth.sh`,
sha256 `02d8be527b6e838177ea5db39f486fe1284aff5eae25e85553747981145b04ea`. Both at commit `887711c`.

Cost: **$0.69 to $0.90** for the two sizes (*estimated*, §5), ceilings **$0.75** (8B, 25 min) and **$0.90** (14B, 30
min) on `l40sx1`. Inside the $10 of O13: $0.46 spent, and a job under way under its own prereg with a ceiling of
$1.92. The operator's go of 2026-11-14 ("Ok go") names this reading at about $0.80.

## 1. The question

On the 4B, the original's lens accuracy goes from 28.3 % at depth 24 to 72.3 % at depth 25 of 36 (CERVEAU Q25, 570
questions): the answer is written in one block. That reading needs the original alone and no compression. Does the
same pattern exist in the 8B and the 14B, and at what depth?

The operator's reason: he wants a way to see, on a model and before any compression, whether it decides at one
place. This prereg reads that on two more sizes. Whether the reading places the loss of a compressed model is the
question of `preregistration-depth-restore-4b-2026-11-14.md`, not of this one.

## 2. The objects

- `Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218`, 36 layers, five shards, 16,381,516,776 B.
- `Qwen/Qwen3-14B@40c069824f4251a91eefaf281ebe4c544efd3e18`, 40 layers, eight shards, 29,536,665,640 B.
- Every shard is checked by size and sha256 in the job against the pins of `ORIGINALS` in the script.
- Text: the MMLU prompts of `bin/mmlu`, round robin over the 57 subjects; N from the time of the first 57 prompts
  under a 300 s budget, at least 570 and at most 2,280. One model, f16, one forward a question.
- The lens at depth l: the model's own final norm and its own four answer rows applied to the residual stream after
  block l, at the answer position. Depth 0 is the embedding output.

## 3. Controls

1. The script's selftest reads GREEN in the job before the model loads (53 checks on the Mac).
2. The prompts' token-id hashes equal the `qhash` column of that size's FP16 census dump before the model loads.
3. Rule D: the model's four answer logits against that size's dump (`mmlu-8b-f16-FULL.csv`, `mmlu-14b-f16-FULL.csv`),
   same questions. GREEN if the picks agree on at least 98 %. The 8B dump names no revision in its header: rule D is
   the proof that the pinned revision is the model of the dump.

If a control fails for a size, nothing is read for that size.

## 4. What is read, and the rules, written before the data

Per size, from `write.tsv` and `lens.tsv`:
- `d_write`: the depth just after the block whose rise of lens accuracy is the largest; `rise`: that rise in points.
- `total`: the output's accuracy on the N questions minus 25, the rise from chance.
- **ONE-BLOCK WRITE** if `rise` is at least half of `total`.
- **GRADUAL** if `rise` is under a quarter of `total`.
- **IN BETWEEN** otherwise.

Nothing is allocated, encoded or scored from this reading. It says where the original writes the answer of this
task into the directions of its four answer tokens. It does not say that the blocks after `d_write` matter less to
a compressed model: on the 4B that is under test in another job, and on these sizes it is not tested at all.

## 5. Cost

- 8B: setup 5 to 7 min, loop 300 s, report 1 min: 11 to 14 min, $0.33 to $0.42 (*estimated*). Timeout 25 min, $0.75.
- 14B: setup 6 to 9 min, loop 300 s, report 1 min: 12 to 16 min, $0.36 to $0.48 (*estimated*). Timeout 30 min, $0.90.
- The code agent's previous loop estimate was low by a factor of 1.6; the loop here is bounded by its budget, the
  setup is not. Queue unbilled.

## 6. Dated disclosure, and the signed predictions

What has run, on the Mac, no model loaded: the selftest, GREEN, 53 checks, with a planted one-block write found at
its block; nine mutants caught and restored; `--check-prompts` for the three sizes, 14,042 of 14,042 prompt hashes
equal to each dump, one tokenizer file at the three revisions, answer token ids 362, 425, 356, 422; `DRY_RUN=1` of the
job script. Not run: the `--lens-only` path on a card. No lens reading of the 8B or the 14B exists.

Known before the data: the 4B's reading (depth 25 of 36, a rise of 44 points in one block, FP16 column of the
contrast map). The full-split FP16 scores, 75.05 for the 8B and 78.88 for the 14B.

### Signed predictions (main session)

Signed by the main session on 2026-11-14, before the stamp and before either job. The lead's view: the same family
at two more sizes should show the same kind of write, at about the same relative depth; the lead has no evidence
for it beyond the 4B.

| quantity | prediction | interval or alternatives |
|---|---|---|
| 8B, rule D | GREEN | about 85 % |
| 8B, `d_write` of 36 | 25 | [20, 31] |
| 8B, `rise` (points) | 35 | [12, 52] |
| 8B, outcome | ONE-BLOCK WRITE | about 60 %; IN BETWEEN about 30 %, GRADUAL about 10 % |
| 8B, output accuracy on the N questions | 77.5 | [74.5, 80.5] |
| 8B, N | 2,280 | [1,100, 2,280] |
| 14B, rule D | GREEN | about 85 % |
| 14B, `d_write` of 40 | 28 | [22, 35] |
| 14B, `rise` (points) | 35 | [12, 55] |
| 14B, outcome | ONE-BLOCK WRITE | about 60 %; IN BETWEEN about 30 %, GRADUAL about 10 % |
| 14B, output accuracy on the N questions | 81 | [78, 84] |
| 14B, N | 1,200 | [570, 2,280] |
