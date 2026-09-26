# Preregistration. The trained row scales at 8B: training, fold, perplexity, census (2026-09-21)

**Written on 2026-09-21 and TIMESTAMPED (`ots stamp`) BEFORE the training job is launched.** The
commit that carries it follows on the operator's go. Operator go and cap as in
`preregistration-census-8b-2026-09-21.md`: the whole 8B chain, $20, solo unless a big problem.

Not edited again. A fact it gets wrong goes beside it, in a `-ECARTS.md`.

**Cost.** Training on h200 at $5.00/h: $6.90 central, **hard cap $11.25** (timeout 2 h 15).
FT census on l40sx1: $1.02 central, **hard cap $1.80** (timeout 60 min). Fold and perplexity on
the Mac, $0. 8B chain before: census in flight, cap $4.05. After this prereg's two jobs, at
most $17.10 of $20 committed; the served checks keep $2.90 (*estimated* from the 4B jobs,
`jobs.csv:171-172`, scaled by the FLOP ratio 1.846 and an h200 speed factor of 2.3, which is a
vendor figure, not a measurement here).

## Question

Do the row scales, trained exactly as at the 4B, move the 8B base as they moved the 4B? At the
4B: +3.15 pp of MMLU on the DCLM base (57.95 → 61.11), perplexity ×1.328 → ×1.007 of f16, at an
unchanged rate (*measured*, `dclm-rowscales-2026-09-20.txt`). The 8B has never been trained.

## Setup

The 4B recipe, `ops/jobs/dclm-rowscales.sh`: KL at T = 1 against the dense bf16 teacher, full
vocabulary, seq 1024 × batch 2, AdamW lr 3e-4, warmup 100, cosine to 0.1, seed 0, the DCLM
corpus in the same order, checkpoint every 200. Changed, and only these:

- the model: student `dclm-8b-export-2026-09-21` (the f16 export of `qwen3-8b-dclm.bin`,
  16,381,516,776 B), teacher `Qwen/Qwen3-8B`;
- **STEPS = 9,507 fixed**, the 4B's count, 19,470,336 tokens (at the 4B it came from BUDGET /
  probe rate);
- the card: **h200** (the code as written peaks at 70-77 GB at 8B, *computed*; the L40S has 48);
- `transformers==5.17.0` and friends pinned (the 4B job installed them unpinned);
- `llvqtune` patched: `llvqtune-8b.patch` sha256 `d9177fc1...`, 80 tests green and 21 mutants
  killed; shipped as `llvqtune-2026-09-21.tgz`, sha256 `0f5d9498001d029a...`. The patch adds a
  mandatory TEACHER, a teacher/student pairing refusal, a STEPS override, a local staged copy of
  the export (the bucket mount is not mmapped), guards `MAX_TRAIN_SECONDS=7000` and
  `MAX_FIRST_KL=0.50` on the probe, and a CUDA memory gauge.

Launchers: `ops/jobs/dclm-8b-rowscales.sh` (training) and `ops/jobs/dclm-8b-ft-mmlu.sh`
(census, same image `a963a020` as the base census, refused if the Space moved). The fold and
the perplexity use the binaries hashed in `~/q8b-dclm-2026-09-21/bin-ft.sha256` and `bin.sha256`.

## Controls

1. The pairing check passes and the probe closes, with a gauge.
2. Probe first KL ≤ 0.50 (the 4B DCLM base read 0.3527).
3. The training exits 0: the KL improved (`__main__.py`, exit 1 otherwise).
4. `sigma.json`: 216 matrices, 1,363,968 values, all finite and positive.
5. All-ones control: `rowscale` reproduces the base byte for byte.
6. The fold: 216 scaled, 36 int4 passed through, size unchanged, differing bytes only in records.
7. FT census dump: the same six header lines and fingerprint `a74a6d6213602979` as the base's.
8. FT perplexity f16 on Metal, token fingerprint `3f1baca9033bf251`, finite.

## What gets published, and what does not get compared

Published: the probe and loop rates, the KL curve, the peak GPU memory, FT − base paired on
14,042 questions (CI95, McNemar), FT against f16 and AWQ paired, the FT perplexity, b/param
unchanged at 3.0683. Not compared: the 8B gain against the 4B gain as a law (one draw per size);
anything served.

## Decision rule

`G` is FT − base, paired, micro, on the census.

| result | reading | what follows |
|---|---|---|
| controls pass, G's CI95 above 0 | the lever transfers to the 8B | served checks, under their own prereg |
| controls pass, G's CI95 contains 0 | the lever does not resolve at 8B | served checks all the same (the file is the recipe's object); reported as such |
| controls pass, G's CI95 below 0 | training made the 8B worse | stop before any served job; report ("gros problème") |
| train.sh exits 2, 3, 4 or 5, or the job dies | no sigma | stop; report; a relaunch is priced against what remains of the $20 |
| training exits 1 (KL not improved) | no usable training | stop before the fold; report |
| otherwise | not settled | operator decision |

G = +2.0 falls in row 1, +0.3 with CI [−0.5, +1.1] in row 2, −1.0 resolved in row 3.

## Signed prediction

| quantity | point | interval |
|---|---|---|
| probe first KL | **0.29** | [0.20, 0.45] |
| loop rate on h200 | **0.52 s/step** | [0.40, 0.70] |
| peak GPU memory allocated | **76 GB** | [60, 95] |
| KL, last eighth against first eighth | **−3 %** | [−7 %, −0.5 %] |
| G, paired, micro | **+2.0 pp** | [+0.5, +3.5] |
| FT perplexity f16 | **9.15** | [9.00, 9.80] |

**KL and G.** The 4B DCLM base read 0.3527 first and −3.8 % over the run; the 8B base sits
nearer its teacher (×1.198 against ×1.328), so it starts lower and has less to repair. The 4B
paper's own Table 6 makes the gain a decreasing function of the base (r = −0.956 over five
pairs); +3.15 at a 57.95 base, less at a base near 66. Flaw: one pair per size, and the base
census is not known when this is written. **Perplexity.** The 4B went to ×1.007 of f16; ×1.018
of 8.99 is 9.15. **Rate and memory.** 0.648 s/step at 4B on L40S × 1.846 FLOPs ÷ 2.3; memory
from the component model of the costing (two bf16 models, non-routed gradients, rebuilt routed
weights, activations). Flaw: the h200 factor and the memory model were never measured here.
