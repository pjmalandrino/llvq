# Preregistration. The 14B paper-2 object on a card: base census, served smoke, kernel bench, served decode (2026-09-22)

**Written on 2026-09-22 and TIMESTAMPED (`ots stamp`) BEFORE `census-14b-base.sh` is launched.**
The launcher refuses both jobs until the `.ots` exists. Operator go, verbatim, 2026-09-22: "allé
lance moi le 14B", on the card-mode plan, whose proposed cap for the 14B chain is $52. The
operator did not restate that figure in his own words; it is taken as accepted with the go, and
it was restated to him when the encode was launched. Spent on the 14B chain before these two
jobs: $18.17 (*measured*, `jobs.csv`: census-ref $3.14, encode seg1 $6.95, encode seg2 $8.08).

Not edited again once stamped. A fact it gets wrong goes beside it, in a `-ECARTS.md`.

**Scope.** Two jobs on l40sx1. (1) `ops/jobs/census-14b-base.sh`, every stage: arm A of the census,
the served smoke, the kernel bench, and the harness control when the Space moved since
`census-14b-ref.sh`. The census-ref prereg scores arms B and C only and leaves arm A to its own
(`preregistration-census-14b-ref-2026-09-22.md`, Question). (2) `ops/jobs/served-14b.sh`,
`PHASE=full DOOR=1`, on the trained file.

**Cost.** `census-14b-base.sh`: $2.13 central, **hard cap $3.30** (110 min); with the harness stage
$2.37, **hard cap $3.60** (120 min). Served decode: $0.66 central, **hard cap $1.35** (45 min). All
*estimated* in the launchers. Ceilings of the 14B chain: $50.85, or $51.15 with the harness stage,
≤ $52 (*computed*; the sum is in `preregistration-dclm-14b-rowscales-2026-09-22.md`, which also
prices a job billed past its timeout, outside the cap). Image: the Space as `census-14b-base.sh`
finds it, recorded, then frozen: `dclm-14b-ft-mmlu.sh` and `served-14b.sh` refuse any other.
Config: `configs/qwen3-14b-tetra-q5.json`, the 4B's served values on the 14B object; the jobs write
it from the command line and check its sha256.

## Question

What does the 14B base score on the full MMLU split? It is the arm the trained file pairs against,
and the gate before the training is paid. Does the Tetra kernel serve a Qwen3-14B file, give the
dense arm's tokens, and at what speed and VRAM? No Tetra file above 8B has run through
`tv_tetra48_h`. `rot_apply_rows`, the prefill rotation, has never run on a card at 17,408;
`rot_apply` ran exact there under Planes14 (`fusedrun-14b-2026-08-17.txt`,
`vague2-fusion-8b-14b-2026-08-31.txt`).

## Setup

`census-14b-base.sh`, one job, after its oracle and the base's bytes and sha256 against the encode
job's line. Each stage runs in its own shell; a failed stage does not skip the next.

1. **Arm A.** `/out/dclm-14b-<OBJ_DATE>/qwen3-14b-dclm.bin`, dense reconstruction in f16,
   `LLVQ_MMLU_ALLOC=flat`, full split, `LLVQ_MMLU_DUMP`. The embedding and the untied head stay f16
   (`config=none`), as at 8B.
2. **Served smoke**, on the base: the config written and checked; the prefill gate at 203 tokens
   through `LLVQ_CONFIG`; 32 served tokens, one arm.
3. **`planesbench`**: ball file first (`qwen3-14b-c12-3f21abde/qwen3-14b-llvq.bin`,
   6,506,354,741 B, sha256 `9df4d475...`, the only 14B Planes14 file), the base second. The fold
   changes row scales only (`rowscale.rs:17-18`), so the base presents the kernel the trained
   file's stream. Arms `fp16, planes14, nullk, tetra48`, tile unset (64 on sm_89), seven rounds,
   two dropped, ratios round by round.
4. **Harness control**, when the Space is not `a963a020`, the image `census-14b-ref.sh` defaults
   to: the 8B base `qwen3-8b-dclm.bin` (sha256 `bcea0d5a...`), which `census-8b` scored FULL on
   `a963a020`, scored again at `limit=40` on this image. Its picks are joined on the Mac with
   `docs/data/mmlu-dumps/mmlu-8b-dclm-FULL.csv` on (subject, index, qhash), 2,280 questions.

`served-14b.sh PHASE=full DOOR=1`, on **the trained file**, in this order: oracle; bytes and sha256
against the Mac's fold; the prefill gate at 203; the 57-question served door; `fusedrun` 256
tokens against the dense arm of the same process at the served flags (`tetra48`, q8 embedding,
`ROT_SHARE=1`, `FUSE=0`, KV f16), five rounds; the same at `LLVQ_EMBED=f16`, the same-head arm
(hard rule 4).

## Controls

1. Oracle MATCH in both jobs; each object's bytes and sha256 equal to the encode job's (base) or
   the Mac's (trained file).
2. Arm A's dump: `dtype=f16`, `limit=census`, `alloc=flat ... 14042 questions`, `config=none`,
   `arithmetic=dense reconstruction`, `kv=f16`, end fingerprint `a74a6d6213602979`.
3. The log says `tile 64 (served: measured optimum for sm_89)`, `160 rot_launches/token for 280
   projections` and `(0 groups + 240 lone + 40 int4)`.
4. The prefill gate: the same argmax over 203 tokens batched and one by one, on the base and on
   the trained file.
5. The door dump carries `# arithmetic=served kernel` and `# layout=tetra48`.
6. `planesbench`: 240 of 280 matrices matched by name; the f64 row check passes on FP16, Planes14
   and Tetra. `nullk` computes no product and is not compared (the 8B's E5).
7. Harness, when it runs: fingerprint `65dcd53655e8bfa5`, and picks identical on at least 99 % of
   the 2,280 joined questions. Below that, the image moved the harness: A − B and A − C are
   published with that line beside them. A − FT is not touched: both run on one image.

## What gets published, and what does not get compared

Published: arm A micro and macro; A − B and A − C paired on 14,042 questions with CI95 and McNemar
(B and C from `census-14b-ref.sh`); A's b/param in both accountings, 3.5272 as scored (f16
embedding) and 2.7371 as served (q8), both *computed* in the encode prereg until `rtbits` reads the
file. Served: tok/s median and range for the q8 arm, the f16 same-head arm and the dense arm; their
ratios as quotients of medians with the envelope fused lo over dense hi to fused hi over dense lo
(`fusedrun` loads one arm at a time, so rounds are never paired; the 8B's E4); GB on card; the
first divergence position; b/param whole model with both tables. `planesbench`: ms, GB read and
GB/s per arm, Tetra against FP16 and against Planes14 in the same process, stated over unequal
passes (240 matrices against 280). If the image still hard-codes the 4B head
(`planesbench.rs:3537, 3554`), the two `f16 lm_head` lines are struck.

Not compared: A against the Planes14 14B sample (72.12 on 2,280 questions; a sample is never
subtracted from a census, and the format, the corpus and the device all moved); A against the 4B
and 8B bases as a law of size (one draw per size, and this one is encoded on a card, a declared
confound in the encode prereg); anything across cards or against another process's numbers. The
served Planes14 14B (46.8 tok/s in 9.40 GB) is a reference, not a ratio. AWQ is not in the bench
(its buffers would put the device at 44.9 of 48.3 GB, *computed*).

## Decision rule

| result | reading | what follows |
|---|---|---|
| A's dump passes, A ≥ 62.0 | the base is a usable file | the row-scale training goes ahead, under its own prereg |
| A's dump passes, A < 62.0 | two draws under the point and under the 8B base's 64.87: the file or the recipe is broken at 14B | stop before any training; report ("gros problème") |
| A's stage fails or the job dies before A | no base score | no training; diagnose; a relaunch costs from the cap |
| smoke fails, A passes | the served path does not load or gate at 14B | training may go ahead on A; `served-14b.sh` is not launched until `PHASE=smoke` passes alone, if the cap still covers it |
| bench or harness fails, the rest passes | that line is missing at 14B | the chain goes on; the stage is rerun alone only on a go |
| served decode controls pass, first divergence absent or after token 5 | the kernel serves the 14B | publish; the chain closes |
| served decode controls pass, first divergence at token 5 or earlier | a defect in the served path at 14B | report ("gros problème"); no speed is published as the object's |
| otherwise | not settled | operator decision |

A at 68.0 falls in row 1, 60.5 in row 2. A first divergence at token 3 falls in row 7.

## Signed prediction

| quantity | point | interval |
|---|---|---|
| A, micro | **68.3** | [65.3, 71.2] |
| B − A, paired | **+9.7 pp** | [+7.0, +12.5] |
| `census-14b-base.sh` running time, harness stage off | **71 min** | [63, 86] |
| `census-14b-base.sh` running time, harness stage on | **79 min** | [70, 95] |
| harness, identical picks | **100 %** | ≥ 99 % |
| q8 arm, tok/s | **58** | [50, 66] |
| dense arm, tok/s | **17.0** | [16.5, 17.5] |
| f16 same-head arm against dense | **×1.60** | [×1.40, ×1.85] |
| GB on card, q8 arm | **5.06** | [4.95, 5.20] |
| GB on card, f16 same-head arm | **6.52** | [6.40, 6.65] |
| first divergence, q8 arm | **none in 256** | after token 32 |
| prefill gate, base and trained file | **same argmax** | |
| `planesbench` FP16 | **38.5 ms** | [36, 41] |
| `planesbench` Tetra against FP16 | **×4.7** | [×3.9, ×5.5] |
| `planesbench` Tetra against Planes14 | **×1.7** | [×1.5, ×1.9] |

**A.** The gap between f16 and the paper-2 base, both on the census: 12.19 pp at 4B (70.14 against
57.95) and 10.18 at 8B (75.05 against 64.87) (*measured*, `dclm-rowscales-2026-09-20.txt`,
`census-8b-2026-09-21.txt`). The same factor, 0.835, applied once more gives 8.50 at 14B. The card
adds 1.18 pp (one pair at 4B, the encode prereg's "The card, declared first"): 9.68. Under the
census-ref prereg's point for B, 77.94, that is 68.26 (*computed*). The interval is one draw,
±2.92 pp. The gate at 62.0 is the point less two draws, rounded down, and it sits under the 8B
base. Flaw: a two-point trend, a card term from one pair on another recipe, and B is itself a
prediction.

**Bench.** FP16 reads 26.42 GB over 280 matrices; at the 682 GB/s it held at 8B that is 38.7 ms
(*computed*). Tetra and Planes14 ms, fitted linear in weights through the tile-64 points of the 4B
(3.423 and 4.995 ms, `tuile-l40s-2026-09-20.txt`) and the 8B (4.969 and 8.033 ms,
`served-8b-2026-09-21.txt`), give 7.9 ms for Tetra over 13.00 G weights and 13.8 ms for Planes14
over 13.21 G: ×4.9 and ×1.74. The point on FP16 is shaded to ×4.7 for a two-point fit. Flaw: the
ratio rose from ×3.21 to ×4.11 between 4B and 8B for reasons the fit does not model, and tile 64
was measured at 4B shapes only.

**Decode.** The 8B's q8 arm ran 11.04 ms a token (90.6 tok/s). The served Planes14 went from 13.25
to 21.37 ms a token between 8B and 14B, ×1.61 (*measured*, ETAT §2 and
`vague2-fusion-8b-14b-2026-08-31.txt`); applied to Tetra, 17.8 ms, 56 tok/s. The bench fit adds
2.9 ms of Tetra kernel and scales the other 6.07 ms by depth and width (40/36 × 5120/4096 = 1.39):
16.4 ms, 61 tok/s. The point sits between. The dense arm read 17.0 tok/s at 14B, twice, and the 8B
dense arm did not move across the device port (26.5 in August and in September). **Same head.**
The f16 arm is slower than the q8 arm by a gap the bytes do not explain: 14.7 ms at 8B under Tetra,
14.66 and 18.75 ms at 8B and 14B under Planes14 (*measured*, B2). 17.2 + 18.75 = 35.95 ms, 27.8
tok/s, ×1.64 of 17.0. **Memory.** Projections ~3.41 GB: 2.058 b/weight at 14B (*computed*, the
encode prereg) plus the card-over-rtbits gap measured at 8B (2.100 against 2.0944), over 13.21 G
weights. q8 tables 1,653.1 MB (*measured* under Planes14 14B); two f16 tables 3.11 GB. Flaw: no
Tetra file above 8B has run through the kernel, and the head is untied at 1.56 GB a table in f16.

**Running time.** The launcher's sum: start, oracle and the base's sha256 5 min; arm A 2,824 s,
the 8B's 1,786 s × 1.58 (the 14B/8B sealed-arm sample ratio); smoke 4; bench 13 (*estimated*). At
the non-embedding weight ratio, 1.90, arm A takes 57 min and the job ~80. The harness stage adds
~8 min: 253 s of 8B scoring at `limit=40` (`campagne-8b-qualite-2026-08-08.txt`), a load and a
sha256 of 4.36 GB over the mount. It runs whenever the Space is not `a963a020`, which is the case
after the rebuild the encode needs, unless `REF_IMAGE_SHA` names the image `census-14b-ref.sh` ran
on. Flaw: both ratios come from the harness before the 2026-09-20 port.
