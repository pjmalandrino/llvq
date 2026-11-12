# Preregistration. Where the sealed 4B loses its MMLU points: restore one projection group at a time

**Written on 2026-11-12, BEFORE the job. NOT YET STAMPED.** The signed predictions (§9) are filled by the main
session, then the file is stamped (`ots stamp`) and committed before `ops/jobs/ablation-qualite-4b.sh` launches.
Once stamped, this file is not edited; a deviation goes into `-ECARTS.md` beside it.

Job script: `ops/jobs/ablation-qualite-4b.sh`, sha256 `79168b5f2876ab416f02431bf9f548bca69d5d2001b429496c543ff89b8132d5`
at writing. If the script changes before the stamp, this line is updated before the stamp.

Cost: **~$5.07** for seven arms (*estimated*, §7), ceiling **$6.00** (timeout 200 min at $1.80/h on `l40sx1`).

## 1. The question

The sealed, served 4B (`qwen3-4b-sealed.bin`) reads 63.37 MMLU against 70.14 for FP16 on the full split, 14,042
questions: a paired gap of 6.77 pp, CI95 [6.05, 7.50] (*measured*, `docs/mesures/embed-q4-swap-2026-09-23.txt`). No
measurement says which projections carry it on this object.

The only attribution (M2, `docs/mesures/m2-attribution-4b-2026-09-02.txt`) ran on another object: the old `Planes14`
file, 55.59 against 70.32, all 252 matrices on the lattice, 2,280 questions. There the MLP group carried +10.78 of
14.73 and the seven single-type gains summed to 23.85, a ratio of 0.618. Since then the format changed (Tetra), the
calibration changed (DCLM-edu), row scales were trained, `v`, `o` and `down@12-23` became int4 records and the embedding
went to int4. The old split says nothing reliable about the new object.

The question: which projection group, restored to f16 in the sealed file, recovers how many of the 6.77 points?

## 2. The object and the path

- File: `hf://buckets/Pier-Jean/jobs-artifacts/sealed-4b-2026-09-23/qwen3-4b-sealed.bin`, 1,418,224,685 B, sha256
  `886391a8c03f66dc269cc65c3598c6627dbdcd259180aff36604ef10d37371b8`. It is the file on the Hub.
- What it holds (*measured*, `docs/mesures/sealed-4b-2026-09-23-brut/int4swap.txt`,
  `docs/mesures/dclm-rowscales-2026-09-20.txt`): Tetra lattice codes with trained row scales for `q`, `k`, `gate`,
  `up` (36 each) and `down` at layers 0-11 and 24-35 (24); int4 g128 records for `v` (36), `o` (36) and `down` at
  layers 12-23 (12); the embedding, tied to the head, in int4 g64. 168 lattice and 84 int4 records.
- Path, every arm: `bin/mmlu`, **dense reconstruction, dtype f16, kv f16, no `LLVQ_CONFIG`**, census plan
  (`LLVQ_MMLU_ALLOC=flat`, no limit), dumps with fingerprint `a74a6d6213602979`. This is the path of the published
  63.37 and of the FP16 census dump (`docs/data/mmlu-dumps/mmlu-4b-f16-FULL.csv`, job `6aad0dd9`). The restore arms
  cannot run on the kernel (`mmlu` refuses `LLVQ_RESTORE_*` beside `LLVQ_CONFIG`), so the control runs dense too and
  every pair is like for like.
- Restoration: `LLVQ_RESTORE_F16=<types>` with `LLVQ_MODEL=Qwen/Qwen3-4B@1cfa9a7208912126459214e8b04321603b3df60c`,
  the snapshot `int4swap` read. The named types are taken from the checkpoint at f16, every layer, lattice and int4
  records alike; everything else is the same bytes in every arm. Exactness of the swap: `decode_matrix` un-rotates, so
  the restored tensor drops into the same slot (M2 §1; M2's "all restored" reproduced the FP16 picks 2,280 of 2,280).
- Image: Space `Pier-Jean/llvq-runner-cuda` at `a03a2badec0dc876b2231f30a65abe220d80fa28`, built from commit
  `36230ff`. `mmlu.rs`, `sealed.rs`, `loader.rs`, `model.rs`, `eval.rs`, `corpus.rs`, `llvq-artifact` and
  `ops/Dockerfile.cuda` are identical between `36230ff` and the main of 2026-11-12. The script refuses another
  Space revision.

## 3. The arms

Weights *computed* from the shapes (36 layers, hidden 2560, q/o 4096, k/v 1024, intermediate 9728).

| arm | `LLVQ_RESTORE_F16` | matrices | weights | of which lattice / int4 |
|---|---|---|---|---|
| `control` | none | 0 | 0 | the sealed file as is |
| `qkv` | `q_proj,k_proj,v_proj` | 108 | 566,231,040 | 72 / 36 |
| `o` | `o_proj` | 36 | 377,487,360 | 0 / 36 |
| `gateup` | `gate_proj,up_proj` | 72 | 1,793,064,960 | 72 / 0 |
| `down` | `down_proj` | 36 | 896,532,480 | 24 / 12 |
| `attn` | `q_proj,k_proj,v_proj,o_proj` | 144 | 943,718,400 | 72 / 72 |
| `mlp` | `gate_proj,up_proj,down_proj` | 108 | 2,689,597,440 | 96 / 12 |

The four single arms partition the 252 projections. `attn` and `mlp` are the sums checks: they are run last and are
the first cut if the job runs long.

**Optional eighth arm, not in the default job:** `all` (`LLVQ_RESTORE_F16=all`, 252 matrices, 3,633,315,840 weights),
~24 min, ~$0.72 (*estimated*), launched with `WITH_ALL=1`. The binary cannot restore the embedding or the head: they are
not projection types. `all` is therefore FP16 except the int4 g64 embedding and tied head, so FP16 minus `all` is the
embedding's cost on this object, and `all` is the measured denominator of §5.2. With it the job is ~193 min, ~$5.79,
inside the $6.00 ceiling with 7 min of margin; a timeout would cut `all` alone. Its inclusion is decided before the
stamp and written on the line below, which must read exactly `ALL ARM: INCLUDED` or `ALL ARM: NOT INCLUDED`
(the script checks it against `WITH_ALL` and refuses the placeholder):

ALL ARM: INCLUDED

## 4. Controls. If one fails, no number is published

1. Oracle on CUDA prints MATCH (hard rule 10). Checked in the job.
2. The file on the mount is 1,418,224,685 B with the sha256 of §2. Checked in the job.
3. **The control reproduces the published picks**: against `embed-q4-swap-2026-09-23/mmlu-4b-embed-q4-swap-FULL.csv`
   (63.37, the file one step before sealing, int4 rebuilt at load), joined on subject, index and qhash, at least
   13,972 of 14,042 picks identical (99.5 %). The job stops before the first restore arm if not, so no money goes to
   arms whose base is wrong. Reported: the exact count and the control's micro (expected 63.37).
4. Every dump carries `dtype=f16`, `limit=census`, `alloc=flat, 100..1534 per subject, 14042 questions`,
   `config=none`, `arithmetic=dense reconstruction`, `kv=f16`, and the trailer `fingerprint=a74a6d6213602979
   questions=14042`. Checked in the job.
5. Every restore arm prints the matrix and weight counts of §3. Another count means another object was scored. Checked
   in the job.
6. The control's dump carries no restoration in its label. Checked in the job.

## 5. The statistic

### 5.1 Per arm

`gain(arm) = micro(arm) − micro(control)`, stratified micro, with its paired CI95 and McNemar from
`mmlupair mmlu-control-FULL.csv mmlu-<arm>-FULL.csv --no-fpc` (bootstrap stratified by subject, 10,000 draws, seed
`0xb0075eed`, no finite-population correction, as every census pair of `docs/ETAT.md` §3). `mmlupair` prints
Δ = control − arm; the gain is −Δ, bounds swapped. Each gain is also read as a share of 6.77.

The FP16 arm is not rerun: `docs/data/mmlu-dumps/mmlu-4b-f16-FULL.csv` (70.14, same fingerprint, same path, same
dtype) pairs with every dump of this job, `mmlupair mmlu-4b-f16-FULL.csv mmlu-<arm>-FULL.csv --no-fpc` gives what
each arm still lacks.

### 5.2 Additivity

- Within groups: `A_attn = gain(attn) / (gain(qkv) + gain(o))`, `A_mlp = gain(mlp) / (gain(gateup) + gain(down))`.
- Global: `S = gain(all) / (gain(qkv) + gain(o) + gain(gateup) + gain(down))`. Without the `all` arm, gain(all) is
  bounded above by FP16 − control = 6.77, and S is reported as an upper bound.
- Read against M2's 0.618, with one difference stated: M2's denominator summed seven single types, this one sums four
  groups, so part of the interaction M2 counted is inside each group here. A ratio above 0.618 is expected from that
  alone and is not evidence that the new object is more additive.
- The ratios are point values; their uncertainty is not bootstrapped jointly (the four arms share the control). Ratios
  whose numerator or denominator has a CI95 containing zero are not read.

## 6. Decision rules, written before the data

Half of the gap is 3.39 pp.

| reading | condition | next step |
|---|---|---|
| **concentrated** | exactly one of the four single arms has gain ≥ 3.39 pp and its CI95 lower bound ≥ 1.69 pp (a quarter of 6.77) | work on that group only, as below |
| **two carriers** | two single arms each ≥ 1.69 pp with lower bounds > 0.85, together ≥ 3.39, none ≥ 3.39 alone | the same, on the two groups, ranked by gain per b/param of fixing them |
| **diffuse** | no single arm ≥ 1.69 pp, or the largest gain is under twice the second and none ≥ 3.39 | a global objective: an encoder or fine-tuning objective on the whole model's output (KL on codes, global GPTQ objective), not a per-group allocation |
| **interacting** | S < 0.50 with the `all` arm, or `A_attn` or `A_mlp` outside [0.70, 1.30] | the single-arm gains are not a decomposition; the next step is global, as for diffuse, whatever the single arms say |

"Interacting" overrides the first two rows. The ranking of the single arms is published in every case.

What "that group" means for the next step depends on what the group is made of (§3):

- `gateup` (lattice only): the Leech encoder's objective on `gate`/`up`, for example an activation- or output-weighted
  objective restricted to them, priced in b/param for the whole model before any run.
- `o` (int4 only): not the lattice. The lever is the int4 records: group 64 instead of 128, or trained scales. Priced in
  b/param; the int4 kernels exist for `o`.
- `qkv` (72 lattice, 36 int4): ambiguous between lattice `q`/`k` and int4 `v`. One split arm (`q_proj,k_proj` against
  `v_proj`, ~$0.72 each) comes first, before any encoder work.
- `down` (24 lattice, 12 int4): one split arm by window (`down_proj@0-11,down_proj@24-35` against
  `down_proj@12-23`) comes first.

This is a measurement, not a gate: it cannot be red. It decides where the next quality step points.

## 7. Cost

- Per census arm, *estimated* from four measured full-split jobs on `l40sx1`: `6aad0dd9` (f16-full, one arm, 23 min),
  `6ab42404` (sealed-rownorms-mmlu, one arm and the oracle, 23 min), `6aaa4e4e` (errmap, two arms and the oracle, 45
  min), `6ab38781` (embed-q4-swap, one restore arm and one limit-40 arm, 29 min). The control ~23 min, each restore arm
  ~24 min (checkpoint fetch on the first), oracle and checks ~2 min.
- Seven arms: ~169 min, **~$5.07**. With `all`: ~193 min, ~$5.79.
- Ceiling: the job timeout, 200 min = **$6.00**. The arm order puts the sums checks and `all` last, so a timeout costs
  them first.
- Queue time is not billed. The job goes into `docs/data/jobs.csv` with its billed minutes.

## 8. What does not compare

- The single gains do not sum to 6.77, and are never presented as a decomposition of it (§5.2).
- One calibration draw: the ranking is this file's.
- Restoring an int4 record measures the int4 cost of that matrix, not the lattice's. The lattice / int4 column of §3
  is printed beside every gain.
- The served path (kernel, `kernel` attention, 4E) is not measured. It returns the eager tokens and its attention sits
  inside the ±0.12 % perplexity tolerance (`docs/ETAT.md` §4); this job says nothing about it beyond that.
- Nothing about speed, memory, perplexity, GSM8K or another size.

## 9. Dated disclosure, and the signed predictions

Known before the job:

- 63.37 (sealed, this path) and 70.14 (FP16, this path), paired gap 6.77 [6.05, 7.50]; AWQ 68.14.
- M2 on the old `Planes14` file (§1): gate +5.18, up +4.94, v +4.48, down +2.96, o +2.35, k +2.09, q +1.85; attention
  +6.90, MLP +10.78, all +14.73; ratio 0.618.
- `v_proj` int4 against lattice on the Tetra-nu object: +1.73 [+0.87, +2.59] (*measured*,
  `docs/mesures/tetra-nu-full-2026-09-18.txt`).
- The q4 embedding plus `o` and `down@12-23` int4, against the 61.11 file: +2.26 [+1.66, +2.87] (embed-q4-swap).
- Row scales: +3.15 at 4B for zero bits; a second training of the sealed file: +0.02 (sealed-rownorms).
- No restore arm has ever run on a Tetra file, at any size.

### Signed predictions (main session)

Signed by the main session on 2026-11-12, before the stamp and before any arm ran. Gain = arm minus control, in pp
of micro accuracy on 14,042 questions, with the interval I expect the point estimate to fall in.

| arm | predicted gain | interval |
|---|---|---|
| control | 63.37, picks equal to the reference on at least 13,972 | |
| qkv | +1.4 | [+0.6, +2.4] |
| o | +0.8 | [+0.2, +1.6] |
| gateup | +3.8 | [+2.5, +5.0] |
| down | +2.2 | [+1.2, +3.3] |
| attn | +2.0 | [+1.0, +3.0] |
| mlp | +5.0 | [+3.8, +6.0] |
| all | +6.3 | [+5.7, +6.8] |

- Reading of §6 expected: gate/up is the largest single carrier and sits near the 3.39 line; the MLP group carries
  more than half of 6.77; no single int4 group (`o`) passes 1.69.
- `A_attn` (attn over qkv plus o): 0.90, in [0.70, 1.10]. `A_mlp` (mlp over gateup plus down): 0.83, in [0.65, 1.00].
- S (all over the sum of the four single gains): 0.77, in [0.60, 0.95]. Sub-additive, less than M2's 0.618 says.
- What would surprise me, and so teach the most: a single int4 arm above +2, or S under 0.5.

Cost, operator's go of 2026-11-12: cap $7.00. The job is launched with `WITH_ALL=1` and `TIMEOUT=230m`, a ceiling of
$6.90, in place of the 200 min of §7. Spend under the $25 cap before this job: $13.96.
