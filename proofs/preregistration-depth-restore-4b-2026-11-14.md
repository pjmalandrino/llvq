# Preregistration. The sealed 4B with the blocks after the answer write restored, then the blocks up to it

**Written on 2026-11-14, BEFORE the job. NOT YET STAMPED.** The signed predictions (§7) are filled by the main
session, then the file is stamped (`ots stamp`) and committed before `ops/jobs/depth-restore-4b.sh` launches. Once
stamped, this file is not edited; a deviation goes into `-ECARTS.md` beside it.

Job script: `ops/jobs/depth-restore-4b.sh`, sha256 `8c3b57ec29d65eb5f87d513b597af7522a8bea103dcd6799d68dab81959b99e9`, at commit `ec79eab`. Image: the private Space
`Pier-Jean/llvq-runner-cuda-qualite` at `c0fb594fe50371bc0171de58167dfe7e1ed6f9e6`, built from `8107e19`.

Cost: **$1.38 to $1.50** (*estimated*, §6), ceiling **$1.92** (timeout 64 min on `l40sx1`). Inside the $10 of O13,
$0.46 spent before it. The operator's go of 2026-11-14 ("Ok go") is on these two probes at $1.40 and on a reading
of the 8B and 14B originals, which has its own prereg.

## 1. The question

The contrast map (`docs/mesures/contrast-map-4b-2026-11-14.txt`, CERVEAU Q25) reads, through FP16's final readout on
570 questions, that the sealed 4B's MMLU loss is complete when the answer is written, after block index 24 of 0 to
35, and that the eleven blocks after it do not change it. That is a reading of hidden states. Does a restore agree?

Two rival figures for the late blocks, layers 25 to 35, 11 of 36 blocks and 1,110,179,840 of the 3,633,315,840
projection weights:
- **The map**: restoring them to f16 gains about nothing.
- **The even rate** (Q22: per restored weight, the MLP's rate equals the rate of all projections): they are worth
  their share of the +6.72 of the whole restore, 6.72 × 11/36 = **+2.05** (*computed*). The early blocks, layers 0 to
  24, are worth 6.72 × 25/36 = +4.67 by the same rule.

The operator's reason for the probe: a reading of the original model alone costs cents and needs no compression. If
it places the loss as a restore does, it can be used on other models. This job is its one validation.

## 2. The objects

- The sealed 4B, `sealed-4b-2026-09-23/qwen3-4b-sealed.bin`, 1,418,224,685 B, sha256 `886391a8…`, checked in the job.
- Arm **late**: `LLVQ_RESTORE_F16` names the seven projection types at `@25-35` (inclusive, 0-based): 77 matrices,
  1,110,179,840 weights, from `Qwen/Qwen3-4B@1cfa9a7208912126459214e8b04321603b3df60c`.
- Arm **early**: the seven types at `@0-24`: 175 matrices, 2,523,136,000 weights.
- Both arms: `bin/mmlu`, full split, dense reconstruction, f16, kv f16, `config=none`, flat allocation, the path of
  the 2026-11-12 ablation, whose job body the script reuses. The late arm runs first.
- The control is not re-run: each arm is paired with `docs/data/mmlu-dumps/mmlu-4b-abl-control-FULL.csv`, the
  ablation's control arm, 63.3742 (8,899 of 14,042), whose picks equal the published dump's on 14,042 of 14,042.

## 3. Controls

1. `oracle` MATCH first on the card.
2. The sealed file's bytes and sha256 on the mount.
3. Each dump carries `dtype=f16`, `limit=census`, flat allocation, `config=none`, `arithmetic=dense reconstruction`,
   `kv=f16` and the end fingerprint `a74a6d6213602979` over 14,042 questions.
4. Each arm's log names the restore with its count, "(77 matrices, 1110179840 weights)" and "(175 matrices,
   2523136000 weights)"; the job stops otherwise.

A limit no control removes: the control arm ran on 2026-11-12 under the shared image at `a03a2bad`, these arms run
under the private image at `c0fb594f`. The two differ by the guided encoder path, which `mmlu` does not reach.
Re-running the control would cost $0.70.

## 4. The statistic

G_late and G_early: the paired MMLU difference of each arm minus the control, micro on the full split, by
`mmlupair --no-fpc` (the arm given first), with its paired-bootstrap 95 % interval.

## 5. Decision rules, written before the data

On G_late:
- **THE MAP IS SUPPORTED** if the upper end of the interval is under +1.03, half of the even-rate figure.
- **THE EVEN RATE IS SUPPORTED** if the interval contains +2.05 or lies above it.
- **NOT DECIDED** otherwise.

G_early is read beside its two figures, +4.67 by the even rate and the remainder of +6.72 by the map; it decides
nothing alone. The sum G_late + G_early is set beside +6.72 [+6.00, +7.45], the whole restore of 2026-11-12.

What follows:
- If the map is supported, Q25 gains a restore, Q22 gains a scope note (even by type, not by depth), and the lens
  reading of the original stands as an instrument on this model. That the late blocks can take fewer bits still does
  not follow: MMLU reads one token, and generation (GSM8K) may use those blocks. A depth-aware allocation is another
  idea, another prereg, and it is scored on both instruments.
- If the even rate is supported, the lens reading does not place the loss as a restore does, and it is not used to
  allocate anything.

One model, one calibration draw, one instrument.

## 6. Cost

- 22 to 24 min an arm on `l40sx1` (the seven restore arms of 2026-11-12 ran 21.7 to 23.3 min, *measured*), oracle
  and checkpoint fetch about 2 min: 46 to 50 min billed, $1.38 to $1.50 (*estimated*).
- Timeout 64 min, ceiling $1.92. Queue unbilled. Spend of the phase before: $0.46 of $10.

## 7. Dated disclosure, and the signed predictions

What has run: `DRY_RUN=1` of the job script, which parses; the two restore strings and their counts were computed
from the 4B's shapes (100,925,440 projection weights a block). The contrast map and its journal. Not run: either
arm. No score of a depth window of the sealed 4B exists, except down_proj alone at layers 12 to 23 on a base
(dclm-down1223, +1.38 over its base, not paired here).

Known before the data: on 570 questions, read through FP16's final readout, the sealed stream is 8.4 points under
FP16 at depth 25 and 7.2 under at depth 36 (Q25). The sealed file holds int4 records for v_proj at every layer and
for o_proj and down_proj at layers 12 to 23, so the late window restores lattice matrices and eleven int4 v_proj.

### Signed predictions (main session)

Signed by the main session on 2026-11-14, before the stamp and before the job. The lead's view: the map should
hold, with a small positive gain from the late blocks because restoring them also removes their noise on what the
earlier blocks wrote.

| quantity | prediction | interval or alternatives |
|---|---|---|
| G_late (pp) | +0.3 | [−0.5, +1.2] |
| G_early (pp) | +6.0 | [+5.0, +6.9] |
| G_late + G_early (pp) | +6.3 | [+5.3, +7.3] |
| outcome on G_late | THE MAP IS SUPPORTED | about 60 %; NOT DECIDED about 30 %, THE EVEN RATE about 10 % |
| late arm, MMLU micro | 63.7 | [62.9, 64.6] |
| early arm, MMLU micro | 69.4 | [68.4, 70.3] |
