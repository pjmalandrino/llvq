# Preregistration. The 14B census, two reference arms: f16 and AWQ (2026-09-22)

**Written on 2026-09-22 and TIMESTAMPED (`ots stamp`) BEFORE the job is launched.** The
launcher refuses to run until the `.ots` exists. After the stamp it is not edited again. A fact it
gets wrong goes beside it, in a `-ECARTS.md`.

Operator go, verbatim, 2026-09-22: "allé lance moi le 14B", given in answer to the card-mode
costing ($38.3 central, $50.6 at the timeouts, a proposed cap of $52). The operator did not state a
cap in his own words; $52 is taken as accepted with the go, and it is restated to him before the
encode job, which commits most of the spend.

**Cost: $3.15 central, $4.05 at the timeout** (l40sx1 at $1.80/h, timeout 135 min of running
time, *estimated* from the 8B census and the 14B/8B time ratio of the 2,280-question runs;
derivation under the signed prediction). The timeout is not a hard cap: the platform once billed
28 min past it (volume-v32, 148 min on a 120-min timeout, `jobs.csv:121`), which would make this
job $4.89. 14B chain spent before: $0 if this is its first job, as the card-mode plan orders it;
the journal gives the real running total. After: at most $4.05 of $52, $4.89 with that overrun.
Project before: $186.99 (`served-8b-2026-09-21.txt`, after the five 8B jobs); after: at most
$191.04, $191.88 with the overrun (*computed*). Image: Space
`Pier-Jean/llvq-runner-cuda` at `a963a020` (commit `afaed1e`). Launcher:
`ops/jobs/census-14b-ref.sh` (sha256 in the journal).

## Question

What do the Qwen3-14B f16 checkpoint and its AWQ w4 g128 score on the full MMLU split, 14,042
questions? No 14B census exists: only 2,280-question samples, 78.97 and 78.21
(`campagne-14b-qualite-2026-08-10.txt:113,220`). At 8B the samples of these two arms moved by
−1.03 and +0.78 pp against their census, in opposite directions, and the paper forbids subtracting
a sample from a census. The arms of the 14B paper-2 object (the DCLM base and the row-scale-trained
file) will be scored on the full split and pair against these two.

## Setup

One job, one card, two arms, each a dense reconstruction in f16, `LLVQ_MMLU_ALLOC=flat`, full split,
`LLVQ_MMLU_DUMP` on each. The job does not wait for the 14B object.

- **B**: `Qwen/Qwen3-14B@40c069824f4251a91eefaf281ebe4c544efd3e18`, pinned in the command (the 8B
  census printed its revision instead). `40c06982` has been `main` since 2025-07-26, so the
  2026-08-10 sample read it too. Downloaded to `/scratch/hf`, the container's local disk.
- **C**: the dequantized AWQ checkpoint in the bucket at `qwen3-14b-awq-deq-1g/`, from
  `Qwen/Qwen3-14B-AWQ@31c69efc` over the base at `40c06982` (`ops/awq_dequant.py:185-187`). It is
  not on the Hub, unlike the 4B and 8B. The job copies 40 files by explicit list (33 shards,
  `model.safetensors.index.json`, `config.json`, four tokenizer files, `generation_config.json`) to
  `/scratch/awq-14b` with `cp`, then scores from there. The three stray
  `.shard-0003{0,1,2}.safetensors` stay out by construction, and so does `LICENSE` (11,544 B,
  not read by the loader): the directory holds 44 entries. Mmapping safetensors from the bucket
  mount killed job 6a796e08 with SIGBUS; the copy is the workaround measured on 2026-08-10 (job
  6a7971c3, 28 GB).
- Order: C is staged and checked first, then B is scored, then C. A failed copy is known in
  minutes and does not cost B; a timeout can only cut C. The copy is bounded, 600 s a file and
  20 min in all, so a stalled read on the mount skips C instead of holding B to the timeout.
- Output: the bucket directory `census-14b-ref-<launch date, UTC>/` (the launcher's default;
  `CENSUS_DATE` overrides it, and an existing directory is refused), dumps
  `mmlu-14b-{f16,awq}-FULL.csv`.

`oracle Qwen/Qwen3-0.6B 64 cuda` first (hard rule 10). The Space revision, the Hub revision of B
and its tokenizer hash are printed on the Mac before launch. `PREFLIGHT_ONLY=1` passed on
2026-09-22: the bucket listing matches the 40 sizes, the Space is at `a963a020`, `main` resolves to
`40c06982`, and the Hub's LFS sha256 of `tokenizer.json` there is `aeb13307a71acd8f...`, the same
file as the 4B and the 8B.

The job mounts `Pier-Jean/jobs-artifacts` writable. Two jobs writing the same bucket do not mount
(`ops/README.md:241-243`): the 14B encode job is launched after this one ends, or into another
bucket. If the Space is rebuilt before this job (for `export` and `rowscale`), the launcher's
`IMAGE_SHA` names the new revision and the journal says so; the dense scoring path
(`llvq-llm/src/{bin/mmlu.rs,loader.rs,model.rs,eval.rs}`) has no diff from `afaed1e` to `01dae9a`.

## Controls

1. Oracle MATCH on CUDA.
2. C staged, before any scoring of C. On the Mac: the live `hf buckets ls` matches the 40 names and
   byte counts of the launcher, whose 33 shards sum to 29,536,663,384 B. In the job: each copied
   file at its listed size; each shard's safetensors header ends its data exactly at the file's
   length; the headers hold 443 tensors, 14,768,307,200 weights and 29,536,614,400 tensor bytes,
   equal to `index.json` `total_size`; the index maps every tensor to the shard that holds it and
   names the 33 listed shards and no other; `config.json` has the 14B shapes (5120, 17408, 40
   layers, 40/8 heads, untied) and no `quantization_config`; `tokenizer.json` hashes to
   `aeb13307...`. The checker was run on a synthetic 14B-shaped checkpoint (pass) and on eleven
   mutants of it (all refused), and on the real local 8B snapshot, whose index `total_size` equals
   the tensor bytes its headers declare. The real files it will read were checked on 2026-09-22
   at $0 (`hf buckets cp <file> -`, read to stdout): `index.json` gives `total_size`
   29,536,614,400 and 443 tensors over exactly the 33 `model-000NN` shards; `config.json` gives
   5120, 17408, 40, 40/8, `head_dim` 128, vocabulary 151,936, untied, `torch_dtype` float16, no
   `quantization_config`. So a refusal on the card means the copy, not the checker's constants.
   A sha256 of the 33 local shards is recorded, not compared: no reference hash exists.
3. Each dump: `model=` naming the pinned revision (B) or `/scratch/awq-14b` (C), `dtype=f16`,
   `limit=census`, `alloc=flat ... 14042 questions`, `config=none`,
   `arithmetic=dense reconstruction`, `kv=f16`, end fingerprint `a74a6d6213602979`.
4. Harness across the device port, at 14B shapes: B and C against the 2,280-question dumps of the
   same weights (`docs/data/mmlu-dumps/mmlu-14b-{f16,awq}.csv`, job 6a7971c3, fingerprint
   `65dcd53655e8bfa5`, byte-identical to the bucket's `campagne-14b-qualite/`): identical `pick`
   on at least 99 % of the 2,280 shared questions. `mmlupair` does not count picks: it counts
   questions where one arm is right and the other wrong, and two different wrong picks are
   concordant for it. So the picks are counted by joining the two dumps on `subject,index`, with
   equal `qhash` required on every joined row, and the same join reports identical logits;
   `mmlupair --intersect` gives the discordant count beside it, as at 8B. Same bytes: B by
   revision; C because the bucket files carry mtimes of 06:14:18 to 06:18:38 UTC on 2026-08-10,
   before the AWQ sample dump at 06:49:16. Below 99 %, the image moved the harness and the journal
   says so.

## What gets published, and what does not get compared

Published: micro and macro for B and C; the paired B − C with CI95 and McNemar, stratified, no
finite-population correction (the full-split convention since 2026-09-17); each arm's shift from its
2,280-question sample, descriptive; the b/param of each arm, whole model, embedding included:
B 16.0000; C 5.4044 for the official packed `Qwen/Qwen3-14B-AWQ` (9,976,690,240 B,
`rtbits-14b-2026-08-17.txt:172`), both *computed*. The file scored as C is its f16 dequantization,
16 b/param, and the journal names both.

Not compared: C as a served AWQ (it is scored dense; no AWQ kernel, no speed); B or C against the
`Planes14` 14B sample (72.12, 2,280 questions), since a sample is never subtracted from a census;
any served speed.

## Decision rule

| result | reading | what follows |
|---|---|---|
| controls 1 to 4 pass on B and C | the 14B references stand | the base and FT arms of the 14B object pair against them, each under its own prereg |
| B passes; C's staging fails, or C is cut | B stands | C is relaunched alone (~$1.60, timeout 80m, worst $2.40, *estimated*) only if the cap allows; else C is declared empty |
| control 4 below 99 % on either arm | the image moved the harness at 14B shapes | the census stands as scored on this image; the object's arms are scored on the same Space revision, or, if the Space has been rebuilt since, their prereg adds its own harness control against B and C; the journal says so |
| B's dump fails, or the job dies before it | no f16 reference | diagnose; a relaunch costs from the cap |
| otherwise | not settled | operator decision |

A shard one byte short falls in row 2; 2,270 identical picks of 2,280 (99.56 %) in row 1; 2,250
(98.68 %) in row 3.

## Signed prediction

| quantity | point | interval |
|---|---|---|
| B, micro | **77.9** | [76.6, 81.3] |
| C, micro | **79.0** | [75.9, 80.6] |
| C − B, paired | **−0.8 pp** | [−2.2, +0.7] |
| harness control, identical picks | **100 %** on both | ≥ 99 % |
| running time | **105 min** | [95, 120] |

**B, C.** The recipe, written out step by step (deviation E1 of `census-8b`: the 8B prereg did not
record its step). Point = the 14B sample plus the shift the same arm showed at 8B between its sample
and its census: f16 76.08 → 75.05 (−1.03), AWQ 73.01 → 73.79 (+0.78) (`census-8b-2026-09-21.txt`).
78.97 − 1.03 = 77.94; 78.21 + 0.78 = 78.99 (*computed*). Interval = the 14B sample's own sampling
interval, ±1.96 standard errors. The `±` that `bin/mmlu` prints is one stratified standard error
with the finite-population correction (`mmlu.rs:390-415`): 1.19 for f16, 1.20 for AWQ. So the
interval covers the census value whether or not the 8B shift transfers: [76.64, 81.30] and
[75.86, 80.56], and it holds both points. Flaws: one precedent each. The two 8B shifts have
opposite signs on the same 2,280 questions, while the two weights disagree on only 7.1 % of the
census (1,002 discordant questions, McNemar 590 / 412). They behave like each arm's sampling error.
Nothing ties an arm's sampling error at 8B to the same arm's at 14B.

**C − B.** Carried arm by arm, the recipe gives +1.05: AWQ over f16. That lies outside the 14B
sample's own paired interval, [−2.17, +0.65] (`mmlu-appariee.csv`, f16 − AWQ +0.76, SE 0.71), and
against the 8B census (−1.27, CI95 [−1.70, −0.83]). So the pair is not signed as the difference of
the two points. It is signed on the 14B sample's paired difference, −0.76, with that pair's CI95.
The two readings cannot both land on their points; this is said before the job, not after it.
Flaw: at 8B the pair fell outside its sample's CI95 (f16 − AWQ 3.07 [1.61, 4.69], census 1.27):
one miss in one precedent.

**Harness.** At 8B, B and C matched their samples on 2,280 questions of 2,280 across the device
port (`census-8b-2026-09-21.txt`, control 4: 0 discordant). The same join, run on 2026-09-22 on
`mmlu-8b-{f16,awq}.csv` against `mmlu-8b-{f16,awq}-FULL.csv`, gives 2,280 identical picks of
2,280 and the four logits identical to the bit on every row, both arms (*measured*). Flaw: the 14B
has more layers and wider matrices, and has never been scored on the ported harness.

**Running time.** The 8B census scored f16 in 1,744 s and AWQ in 1,740 s
(`census-8b-2026-09-21-brut/out-{f16,awq}.txt`). The 14B/8B ratio on the 2,280-question runs:
387 / 248 = 1.56 for f16 and AWQ, 400 / 253 = 1.58 for the sealed arm
(`campagne-14b-qualite-2026-08-10.txt:99,206,312`; `campagne-8b-qualite-2026-08-08.txt:121,203,285`).
So 2,722 + 2,715 s = 90.6 min of scoring. Add ~14 min: start and oracle 2.5; staging C 4.5 (29.54
GB off the mount, plus sha256); B's download of 29.54 GB and load 6; C's load 1. Total 105 min
(*estimated*). Flaw: both ratios come from the harness before the 2026-09-20 port, on a sample.
The non-embedding weight ratio is 1.90 (13.21 G against 6.95 G, *computed*); at that ratio the
job takes ~124 min, inside the timeout and outside the interval.

## What it will not establish

- Nothing about the 14B paper-2 object. Its base and FT arms are scored in their own jobs.
- The object's arms pair with B and C across images if the Space is rebuilt in between. Control 4
  covers B and C only; the object's census needs its own harness control.
- That C scores what a served AWQ would. The dequantization passed its controls on 2026-08-10
  (`ops/README.md:234-238`). No AWQ kernel's arithmetic is measured here.
- A sample-to-census correction for other arms. At 8B three arms moved by −1.03, +0.78 and +2.24.
- Anything served, or any speed.
