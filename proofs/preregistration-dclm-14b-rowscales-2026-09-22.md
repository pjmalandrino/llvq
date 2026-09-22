# Preregistration. The trained row scales at 14B: training, fold, perplexity, census (2026-09-22)

**DRAFT, NOT STAMPED.** Written on 2026-09-22, before the 14B is encoded. To be TIMESTAMPED
(`ots stamp`) before the training job is launched, and ideally before arm A of the census is read,
so that the predictions below stay blind to the base. The commit that carries it follows on the
operator's go. Operator go, verbatim, 2026-09-22: "allé lance moi le 14B", on the card-mode plan.
That plan proposes a cap of $52 for the whole 14B chain. **TO CONFIRM before stamping: the go does
not restate the figure.**

Not edited again once stamped. A fact it gets wrong goes beside it, in a `-ECARTS.md`.

**Cost.** Training on h200 at $5.00/h: $10.42 central, **hard cap $15.00** (timeout 180 min). FT
census and the perplexity pair on l40sx1: $1.80 central, **hard cap $2.40** (timeout 80 min). Fold
on the Mac, $0. Ceilings of the 14B chain, from the launchers' timeouts: census B and C 4.05 +
encode 24.75 + arm A, smoke and bench 3.30 + training 15.00 + FT census 2.40 + served decode 1.35 =
**$50.85 ≤ $52**, $51.15 if `census-14b-base.sh` runs its harness stage (*computed*; the B/C and
encode ceilings are those of `census-14b-ref.sh` and `encode-14b.sh`, under their own preregs).
Central estimates are the launchers' (*estimated*). The cap does not cover a job billed past its
timeout: the platform has billed +18 and +28 min past one (`docs/data/jobs.csv`, q5-alloc-int4 and
volume-v32). At +28 min on each of the chain's seven jobs the ceiling is ~$59.1, the h200 alone
+$2.33 (*computed*).

## Question

Do the row scales, trained as at the 4B and the 8B, move the 14B base as they moved those? At the
4B: +3.15 pp of MMLU (57.95 → 61.11), perplexity ×1.328 → ×1.007 of f16. At the 8B: +3.29 pp
(64.87 → 68.16), perplexity ×1.198 → ×1.065. Both at an unchanged rate (*measured*,
`dclm-rowscales-2026-09-20.txt`, `dclm-8b-rowscales-2026-09-21.txt`). The 14B has never been
trained.

## Setup

The 8B recipe, `ops/jobs/dclm-8b-rowscales.sh`: KL at T = 1 against the dense bf16 teacher, full
vocabulary, seq 1024 × batch 2, AdamW lr 3e-4, warmup 100, cosine to 0.1, seed 0, the DCLM corpus
in the same order, checkpoint every 200, **STEPS = 9,507** (19,470,336 tokens), h200, image
`pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`, `transformers==5.17.0` and friends pinned,
`llvqtune-2026-09-21.tgz` (94,659 B, sha256 `0f5d9498001d029a...`), unchanged. Changed, and only
these:

- the model: student `dclm-14b-export-<OBJ_DATE>` (the f16 export of `qwen3-14b-dclm.bin`,
  29,536,665,800 B, *computed* by header emulation), teacher `Qwen/Qwen3-14B`. The launcher refuses
  if the Hub moved from `40c06982`, the revision the encode read;
- **PROBE = 20** steps instead of 6. At 8B the 6-step probe read 49 % slower than the loop
  (`-ECARTS` E1 of the 8B), and at 14B it would make the guard refuse a good run;
- **MAX_TRAIN_SECONDS = 9,500** (8B: 7,000); MAX_FIRST_KL = 0.50, unchanged;
- the export is staged by the launcher, and its `model.safetensors` is checked by sha256 against
  the line the encode job wrote, before the probe loads anything. At 8B, byte counts only;
- timeout 180 min.

Fold on the Mac with the 8B's `rowscale` binary, sha256 `d25136b9...` (built at 5d36d52), after
an all-ones control. FT census on l40sx1 on the image arm A ran on: `census-14b-base.sh` records
the Space sha, `dclm-14b-ft-mmlu.sh` refuses another. New at 14B: the perplexities of the trained
file and of the base are read on the card, in the FT census job, one after the other. The 8B read
them on Metal. At 14B the sealed f16 load is at least ~52 GB of Metal buffers on a 55.7 GB working
set (*computed*), and the Mac is not to be loaded.

Launchers: `ops/jobs/dclm-14b-rowscales.sh` (training), `fold-14b.sh` (Mac; its sha256 goes into
the journal), `ops/jobs/dclm-14b-ft-mmlu.sh` (census and perplexities). Bucket directories: the
training writes `dclm-14b-rowscales-<OBJ_DATE>/`, the fold uploads `dclm-14b-ft-<OBJ_DATE>/`, the FT
census writes `census-14b-ft-<launch date, UTC>/` (the launcher's default, `CENSUS_DATE` overrides;
the 8B prereg named none, its `-ECARTS` E3).

Preconditions, set by the other preregs of the chain and checked by the operator before the
training is launched (no launcher reads them): R in row 2 of the encode prereg's decision rule
(rows 1 and 3: the operator decides whether the h200 is paid; row 4: no further paid 14B job until
the operator chooses; row 5: no usable file), and arm A at or above 62.0, row 1 of
`preregistration-served-14b-2026-09-22.md`.

## Controls

1. Before the probe: the tarball's sha256 on the mount; the export's four files at 29,536,665,800,
   728, 11,422,654 and 64 B; the staged `model.safetensors` sha256 equal to the encode job's.
2. The pairing check passes and the probe closes, with a gauge.
3. Probe first KL ≤ 0.50.
4. The training exits 0: the KL improved.
5. `sigma.json`: 240 matrices, 2,048,000 values, all finite and positive, no `v_proj`.
6. All-ones control: `rowscale` reproduces the base byte for byte.
7. The fold: 240 scaled, 0 untouched, 40 int4 passed through; 2,048,000 row scales read; size
   unchanged; every differing byte before SIZE − 3,123,922,582, the carried tail.
8. FT census dump: the six header lines and the fingerprint `a74a6d6213602979`, as arm A's.
9. Both perplexities: f16, token fingerprint `3f1baca9033bf251`, finite.

## What gets published, and what does not get compared

Published: the probe and loop rates, the KL curve by eighths, the peak GPU memory, FT − base paired
on 14,042 questions (CI95, McNemar), FT against f16 and AWQ paired (arms B and C of the census),
the FT and base perplexities read in one job and their ratio, and the FT perplexity against f16's
7.9820 (*measured* on l40sx1, `campagne-14b-qualite-2026-08-10.txt:143`). The b/param is unchanged
by the fold and is given in both accountings: q8 embedding as served, and f16 embedding as the
census scores it (`config=none`; the 8B's E6).

Not compared: the 14B gain against the 4B and 8B gains as a law (one draw per size); the perplexity
read here against the encode job's reading of the base (another card); anything served.

## Decision rule

`G` is FT − base, paired, micro, on the census.

| result | reading | what follows |
|---|---|---|
| controls pass, G's CI95 above 0 | the lever transfers to the 14B | served decode, under its own prereg |
| controls pass, G's CI95 contains 0 | the lever does not resolve at 14B | served decode all the same (the file is the recipe's object); reported as such |
| controls pass, G's CI95 below 0 | training made the 14B worse | stop before the served decode; report ("gros problème") |
| the staged sha256 differs from the encode job's | the export on the bucket is not the one encoded | no training; re-upload or re-export, priced against the cap |
| train.sh exits 2, 3, 4 or 5, or the job dies | no sigma | stop; report; a relaunch is priced against what remains of the cap |
| training exits 1 (KL not improved) | no usable training | stop before the fold; report |
| control 5, 6 or 7 fails (sigma, all-ones control, fold bytes) | the FT file is not the recipe's | no upload, no FT census; report |
| otherwise | not settled | operator decision |

G = +3.2 falls in row 1, +0.4 with CI [−0.4, +1.2] in row 2, −1.0 resolved in row 3.

## Signed prediction

| quantity | point | interval |
|---|---|---|
| probe first KL | **0.30** | [0.18, 0.45] |
| probe rate on h200, 20 steps | **0.83 s/step** | [0.72, 1.00] |
| loop rate on h200 | **0.734 s/step** | [0.62, 0.90] |
| peak GPU memory allocated | **119.5 GB** | [108, 130] |
| KL, last eighth against first eighth | **−3 %** | [−7 %, −0.5 %] |
| G, paired, micro | **+3.2 pp** | [+1.7, +4.7] |
| FT perplexity f16, card | **8.50** | [8.05, 9.20] |
| FT against base perplexity, same job | **−16 %** | [−24 %, −7 %] |

**KL.** The first KL tracked the base's distance to its teacher: 0.3527 at R ×1.328 (4B), 0.2741
at ×1.198 (8B). The encode prereg signs R ×1.28 [×1.17, ×1.38] for the 14B, the card term
included. Linear in R between the two points, ×1.28 reads 0.32; the point is 0.30. The curve by
eighths went −6.6 % (4B) and −2.0 % (8B): more to repair, a steeper curve, so −3 %. Flaw: two
points per trend, and R is itself a prediction.

**Rate and memory.** Loop: 0.3994 s/step *measured* at 8B × 1.838, the 14B/8B FLOP ratio
(*computed*: 8 × matmul parameters × tokens, plus attention). Probe: the 8B's ~1.2 s of fixed
probe cost spread over 20 steps instead of 6. The top of the probe interval touches the guard:
above 0.9993 s/step, 9,507 steps project past MAX_TRAIN_SECONDS and train.sh exits 3 (row 5). Memory: the component model fitted to the 8B's
72.65 GB within 0.7 % (two bf16 models 59.07, non-routed gradients 3.53, rebuilt routed weights
26.00, activations 23.3-24.1, logits and KL 7.47; *computed*). Flaw: one h200 point, and the part
of the 8B residual that grows with size is not known.

**G.** Two measured points, +3.15 and +3.29, on bases 6.9 points apart: the prior that the gain
falls as the base rises failed at 8B (`-ECARTS` E4 of the 8B). The served prereg puts the 14B base
9.7 pp under f16, near the 8B's 10.2, so the point is the mean of the two measured, +3.2. The
interval keeps the 8B's width, ±1.5 ([+0.5, +3.5] around +2.0). Flaw: one calibration draw and one
training per size; σ between draws is 2.92 pp at the 4B.

**Perplexity.** The training kept 2.5 % of the base's log-excess over f16 at 4B (×1.328 to
×1.007) and 35 % at 8B (×1.198 to ×1.065). At R ×1.28, a middle share of ~25 % gives ×1.064 of
7.9820, 8.50; the base at ×1.28 reads 10.22, so the drop is −17 %, signed −16 % (4B −24.1 %, 8B
−11.1 %). Flaw: f16's 7.9820 is an August reading, before the device port, and R is the encode
prereg's prediction; the FT − base pair read in this job depends on neither.
