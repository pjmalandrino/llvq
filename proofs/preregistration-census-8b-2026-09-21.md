# Preregistration. The 8B census, three arms: the DCLM base, f16, AWQ (2026-09-21)

**Written on 2026-09-21 and TIMESTAMPED (`ots stamp`) BEFORE the job is launched.** The commit
that carries it follows on the operator's go. Operator go and cap, verbatim, 2026-09-21: "tu
peux enchainer sur la suite du programme pour avoir le 8B au complet ... la totale", "Budget CAP
pour le total sur le 8B 20$", "enchaine solo jusqu'a avoir terminé le 8B total, sauf gros
problème".

Not edited again. A fact it gets wrong goes beside it, in a `-ECARTS.md`.

**Cost: $3.20 central, hard cap $4.05** (l40sx1 at $1.80/h, timeout 135 min of running time,
*estimated* from vod-8b: 2 arms, 4,131 s running, $2.07, `jobs.csv:162`). 8B chain spent before:
$0. After: at most $4.05 of $20. Image: Space `Pier-Jean/llvq-runner-cuda` at `a963a020`,
kept frozen for every 8B job. Launcher: `ops/jobs/census-8b.sh` (sha256 in the journal).

## Question

What does the 8B base file of step 1 score on the full MMLU split, and what do the f16 checkpoint
and AWQ w4 score on the same 14,042 questions? No 8B f16 or AWQ census exists: only
2,280-question samples, which read the 8B 2.24 pp low (`vod-8b-2026-09-18.txt`), and the paper
forbids subtracting a sample from a census. Arm A is also the base the trained arm pairs against.

## Setup

One job, one card, three arms in order, each a dense reconstruction in f16,
`LLVQ_MMLU_ALLOC=flat`, full split, `LLVQ_MMLU_DUMP` on each:

- **A**: `/out/dclm-8b-2026-09-21/qwen3-8b-dclm.bin`, sha256 `bcea0d5a7de2fbbf...`,
  4,364,205,777 B, checked on the mount against the Mac before scoring.
- **B**: `Qwen/Qwen3-8B`, downloaded to the container's local disk.
- **C**: `Pier-Jean/qwen3-8b-awq-deq`, same. C runs last: a timeout can only cut C.

`oracle Qwen/Qwen3-0.6B 64 cuda` first (hard rule 10). The Hub revisions of B and C and the
Space revision are printed on the Mac before launch.

Deviation declared before the job: the sealed base was uploaded to the bucket on 2026-09-21 in the evening, before
this prereg, although the step-1 prereg tied the upload to the first paid one. The file is
byte-checked and sha256-checked inside the job, so the order changes nothing measured.

## Controls

1. Oracle MATCH on CUDA.
2. The base on the mount: same byte count and same sha256 as `files.sha256` on the Mac.
3. Each dump: `dtype=f16`, `limit=census`, `alloc=flat ... 14042 questions`, `config=none`,
   `arithmetic=dense reconstruction`, `kv=f16`, end fingerprint `a74a6d6213602979`.
4. Harness across the device port: B and C against the 2,280-question dumps of the same weights
   (`mmlu-8b-f16.csv`, `mmlu-8b-awq.csv`), `mmlupair --intersect`: picks identical on at least
   99 % of the shared questions. Below that, the image moved the harness and the journal says so.

## What gets published, and what does not get compared

Published: micro and macro for A, B, C; the paired A−B, A−C and C−B with CI95 and McNemar; the
b/param of each arm, whole model, embedding included (A 3.0683, *measured* by `rtbits`).
Not compared: A against `Planes14` 8B or the bare 8B `Tetra` as a format or corpus effect (one
draw, σ = 2.92 pp between draws at the 4B, and three variables moved); any served speed.

## Decision rule

| result | reading | what follows |
|---|---|---|
| controls pass, A ≥ 58.0 | the base is a usable file | the row-scale training goes ahead, under its own prereg |
| controls pass, A < 58.0 | more than two draws under the bare 8B's 63.85: the file or the recipe is broken at 8B | stop the chain before any training; report ("gros problème") |
| A's dump passes, B or C fails or is cut | the base stands | training goes ahead; the missing arm is relaunched only if the cap allows, else declared empty |
| A's dump fails or the job dies before A | no base score | no training; diagnose; a relaunch costs from the cap |
| otherwise | not settled | operator decision |

A at 66.0 falls in row 1, 55.0 in row 2; a cut during C in row 3.

## Signed prediction

| quantity | point | interval |
|---|---|---|
| A, micro | **66.0** | [63.1, 68.9] |
| B, micro | **77.0** | [75.5, 78.5] |
| C, micro | **74.5** | [72.5, 76.5] |
| C − B, paired | **−2.5 pp** | [−4.0, −1.0] |
| running time | **107 min** | [100, 115] |

**A.** The bare 8B `Tetra` reads 63.85 on the full split (*measured*, `mmlu-8b-tetra-FULL.csv`).
At the 4B, full split, `v_proj` in int4 added 1.73 pp (54.64 → 56.37) and the DCLM corpus 1.58
(→ 57.95) (*measured*). At the 8B the int4 lever paid about 27 % less per bit (`vod-8b`). So
63.85 + 1.26 + ~1.2 ≈ 66.3, rounded down for the lower yield at size (*estimated*). The interval
is one draw, ±2.92 pp. Flaw: the 4B increments were measured on different draws and code.
**B, C.** The samples read 76.08 and 73.01 (`jobs.csv:40`). The 8B sample under-read the census
by 2.24 pp on `Tetra`, the 4B f16 by −0.18 pp (70.32 → 70.14): the correction depends on the
model, so the points take half the 8B's shift. Flaw: one precedent each.
