# Preregistration — E8 cubed against Tetra inside the same GPTQ loop

Written and stamped before the first cell is read. Deviations go beside this file, in its
`-ECARTS` companion, never inside it.

Cost: $0, Mac only, no card, no job. 12 to 15 min of wall clock, announced before the go and
measured on a synthetic Hessian of the same width (9.6 s for one row at n = 2560, including a
Cholesky that amortizes over the cell's four rows).

## 1. What this repairs

The run of 2026-09-18 (`docs/mesures/e8-etape1-2026-09-18.txt`) is void on its primary
question, by its own deviation E3: arm A was a sequential GPTQ witness and arm B a plain
per-block encode, so the ratio measured the compensation and not the codebook.

This removes that confound the only way that works. `llvq_quant::gptq::quantize_layer` runs
**both** arms, with the same `GptqFactor`, the same `GptqConfig`, the same gain centroids and
the same row scale. The only difference left is which `BlockQuantizer` the loop calls.

A wiring check ran on a synthetic Hessian before this file was written. It read no cell and no
number from it enters any result below.

## 2. The configuration, frozen now

Data: the v64 arm of the 4B diagnostic dump set, `capture-a-v64` and `capture-b-v64`, 10 cells
and 40 rows, rotated basis, manifest `docs/data/tetra-diag-4b-2026-09-18/outputs.sha256`. The
Hessian and the rows are read from the `.f64le` arrays; `n`, the centroids and the damping come
from each cell's `bundle.json`.

Loop: `block` 24, `retract` true, `group_scales` false, `design_c` false, `lambda` 1e-2,
`tail` KeepExact, `damping` 0.01 as the plan records it. Both quantizers return `None` from
`retraction_target`, so the retraction is a no-op for both.

Arm A: `TetraShapeGain` with the cell's centroids, the served rule.
Arm B: `E8Cubed` at the norm-8 cap, 26,640 points, a 15-bit index, 45 bits plus one gain bit,
**1.9167 b/weight against Tetra's measured 1.9907**. Arm B runs on 3.7 % fewer bits, which
makes a loss by arm B conservative and a win by arm B qualified.

## 3. The primary result, named before the numbers

`J_B / J_A`, the ratio of Hessian-weighted relative errors `e' H e / (x' H x)`, pooled over the
40 rows and reported per family. It is primary and not the unweighted error because the loop
minimizes the weighted one, and because the model pays the weighted one.

Secondary, in this order: the unweighted error ratio; each arm's ratio to its own isotropic
reference `|e|^2 tr(H) / n / (x' H x)`; and the mean cosine per arm.

## 4. Signed prediction

| quantity | point | interval |
|---|---|---|
| `J_B / J_A`, pooled | **1.09** | [0.95, 1.30] |
| unweighted error ratio B/A, pooled | 1.09 | [0.95, 1.30] |
| `\|J_B/iso_B - J_A/iso_A\|` | 0.00 | below 0.05 |
| mean cos A minus mean cos B | +0.005 | [0.000, +0.020] |

The 1.09 is the second-moment ratio of the arbitration dossier, transported with no correction.
The third line is the prediction the void run's journal already put on the record: once arm B
is inside the loop, its residual should stop being isotropic and land where arm A's lands.

## 5. The decision rule

- `J_B / J_A` inside [0.95, 1.30]: the lattice difference on real blocks is what the second
  moment predicts. E8 becomes a **simplification** candidate and the next question is the
  kernel, not the quality.
- Below 0.95: E8 cubed wins inside the loop on 3.7 % fewer bits. The lattice is not why Leech.
  That is a finding, and it earns one rate-matched re-run before any claim.
- Above 1.30: Leech's advantage on real blocks exceeds its second moment. That is a finding,
  and E8 is dropped as a simplification.

If the third prediction fails, that is, if the two ratios to isotropic still differ by more
than 0.05, then something other than compensation separates the arms and **no reading of the
primary is accepted**. The run would then be void for the same reason the last one was.

No outcome of this stage authorizes a served format, a kernel, or a quality statement.

## 6. Controls

1. Both arms pass through the same `quantize_layer` call site, by construction.
2. Arm A run twice on one row gives bit-identical weights: the loop is deterministic.
3. The isotropic reference is computed from `tr(H)/n` and the error energy, and is reported
   for both arms, so a reader can separate how much error from where it goes.
4. The E8 codebook carries stage 0's eight exact invariants plus four quantizer tests:
   a codebook block reconstructs exactly, a zero block stays zero, every output sits on a
   level sphere, and a wider shell cap never loses on angle.
5. `down_proj` is reported separately and read last: even on the v64 arm it carries 1.7
   calibration samples per dimension, so its Hessian is rank-deficient.

## 7. What it will not establish

- No quality claim. `J_local` is a local activation metric over one projection, not the model's
  loss, not perplexity, not MMLU.
- Nothing about a kernel. No E8 decoder exists on a card, and arm B's b/weight ignores what a
  real packing pays in addressing.
- Nothing about Leech's own second moment, which stays a citation.
- `Tetra` is a trellis-searched subset of the Leech lattice, not its Voronoi quantizer, so a
  result inside the interval confirms the bound for **the served format** and not for the
  lattice in general.
- Three layers of 36, four rows of thousands, one model, one calibration seed.
