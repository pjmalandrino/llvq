# Preregistration — E8 cubed against Tetra on the same blocks, stage 1

Written and stamped before the first block is encoded. Deviations go beside this file, in its
`-ECARTS` companion, never inside it.

Cost: $0, Mac only, no card, no job. Minutes of wall clock.

## 1. What this decides, and what selected it

`docs/arbitrage-e8-2026-09-18.md` bounds the lattice difference between E8 and Leech at 9.0 %
of MSE, and stage 0 measured the E8 half of that ratio to +0.002 %
(`docs/mesures/e8-etape0-2026-09-18.txt`). The bound is asymptotic and neither lattice runs in
that regime at 2 bits per dimension with shape-gain. Stage 1 asks whether a matched E8 cubed
code behaves on the model's real blocks as the bound predicts.

Nothing selected this arm from a set of candidates. It is the one comparison the dossier names.

## 2. The configuration, frozen now

Data: the v64 arm of the 4B diagnostic dump set, `capture-a-v64` and `capture-b-v64`, 10 cells
and 40 rows, manifest `docs/data/tetra-diag-4b-2026-09-18/outputs.sha256`. The v64 arm and not
the others, because at 2,048 calibration tokens every Hessian in the set is rank-deficient
(`docs/mesures/tetra-diag-4b-2026-09-18.txt`).

Blocks: every complete 24-dimensional block of `diagnostic.original` in each row. The tail of
`d_in mod 24` columns is excluded, as the served format excludes it. That is 6,204 blocks.

Arm A, Tetra: the served encoder's own answer, read from the dump's `witness` and
`witness_codes`. Nothing is re-encoded, so arm A cannot drift from what the object does.

Arm B, E8 cubed: the 24-dimensional block split into three consecutive 8-dimensional
sub-blocks, each quantized to the nearest direction of the norm-10 E8 codebook, 56,880 points.
One gain per 24-block, chosen from **the same two centroids the cell's bundle records**, and
the same `row_scale`. Only the direction codebook changes between the arms.

Rates, stated before the measurement: arm A is 1.9907 b/weight of stream (*measured*,
`references-comptabilite-2026-09-18`); arm B is 48 bits of index plus 1 gain bit per 24-block,
2.0417 b/weight (*computed*). **Arm B is given a 2.6 % bit advantage.** A loss by arm B is
therefore conservative, and a win by arm B is not conclusive.

## 3. The primary result, named before the numbers

The ratio of mean squared reconstruction error, arm B over arm A, pooled over the 6,204 blocks,
paired block by block. Reported per projection family as well as pooled.

Secondary, in this order: mean cos theta of each arm; and `J_local`, the Hessian-weighted
relative error `e' H e / (x' H x)`, with H the cell's own calibration Hessian.

## 4. Signed prediction

| quantity | point | interval |
|---|---|---|
| MSE ratio B/A, pooled | **1.09** | [1.00, 1.20] |
| mean cos theta, A minus B | +0.005 | [+0.002, +0.010] |
| `J_local` ratio B/A | no prediction | the normalized second moment says nothing about an H-weighted error |

The 1.09 is the 9.0 % of the dossier, transported to real blocks with no correction.

## 5. The decision rule

- Ratio inside [1.00, 1.20]: the lattice difference on real blocks is what the second moment
  predicts. E8 becomes a **simplification** candidate, and the next question is the kernel, not
  the quality.
- Ratio above 1.20: something beyond the lattice favours Leech on real blocks. That is a
  finding, and E8 is dropped as a simplification.
- Ratio below 1.00: E8 cubed wins while holding a 2.6 % bit advantage. Not a conclusion. It
  earns one rate-matched re-run at a cap between shells 10 and 12, and nothing is claimed
  before that re-run.

No outcome of this stage authorizes a served format, a kernel, or a quality statement.

## 6. Controls, all four run before the result is read

1. A zero block reconstructs to zero in both arms.
2. A block that already lies in the E8 cubed codebook reconstructs with zero error in arm B.
3. Arm A's error recomputed from the dump's `witness` agrees with the `shadow` entries the
   replay already recorded, to 1e-12.
4. The comparison is paired: same blocks, same order, one pass, no seed of its own.

## 7. What it will not establish

- No quality claim. `J_local` is a local activation metric over one projection, not the
  model's loss, not perplexity and not MMLU.
- Nothing about a kernel. No E8 decoder exists on a card, and the b/weight of arm B ignores
  what a real packing pays in addressing.
- Nothing about Leech's own second moment, which stays a citation.
- Three layers of 36 and four rows of thousands, one model, one seed of calibration.
