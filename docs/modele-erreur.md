# The error model: what LLVQ optimizes, and what it should

Quantization error is not one quantity. This document separates the three that
LLVQ has been treating as one, gives the closed form of each, and reports where
they disagree — measured, on one model, on 2026-09-15.

It describes `llvq-llm/src/errmodel.rs` and `llvq-llm/src/bin/errmap.rs`. The
numbers are in [gain-scale-0.6b](mesures/gain-scale-0.6b-2026-09-15.txt) and
[tetrapost-ppl-0.6b](mesures/tetrapost-ppl-0.6b-2026-09-15.txt).

## 1. Three objectives, and why the difference is not academic

A quantizer replaces a weight matrix `W` by a reconstruction `Q`. Three
different things can be meant by "the error", and each has its own minimizer.

| | what it measures | minimized by |
|---|---|---|
| `‖W − Q‖²` | the weights | the nearest codebook point |
| `tr(E H Eᵀ)`, `E = W − Q` | the matrix's own output | **what GPTQ minimizes** |
| `L(model)` | the model's loss | what anyone actually wants |

The second is the standard objective of GPTQ and of every method descended from
it. It is exact, cheap, and it is **not** the third.

The gap is measurable, and on 2026-09-15 it was measured on one knob — a scalar
multiplying the fitted gain centroids of every matrix. Each of the three
objectives was minimized over that one parameter:

| objective | optimal multiplier |
|---|---|
| Euclidean error on the weights | 0.906 |
| `tr(E H Eᵀ)` | 0.999 |
| the model's perplexity | 1.02 |

The middle number is a closed form (§2) verified against direct evaluation to
`1.6e-14`, so the disagreement is not estimation noise. **A calibration rule
built on the layer objective reads "nothing to correct" and misses what the
model shows.**

## 2. The layer objective is an exact parabola in the scale

Let `Q` be the reconstruction at fixed codes and `s` a multiplier on the fitted
centroids. Because `reconstruct_shape_gain` is linear in the centroid,
`Q(s) = sQ` exactly, at fixed codes. Writing `H = U⁻¹U⁻ᵀ` for the factor GPTQ
already holds, and `X̂ = XU⁻¹`,

```
L_layer(s) = tr((W − sQ) H (W − sQ)ᵀ) = ‖Ŵ‖² − 2s⟨Ŵ, Q̂⟩ + s²‖Q̂‖²
```

which is a parabola with positive leading coefficient. Therefore

```
s* = ⟨Ŵ, Q̂⟩ / ‖Q̂‖²        and        L_layer(1) − L_layer(s*) = ‖Q̂‖² (s* − 1)²
```

Both are one pass over the matrix, with no re-encoding and no search. The second
identity is worth stating separately: the gain from rescaling is the energy of
the reconstruction in the `H` metric, times the squared distance of `s*` from
one. It is zero exactly when `s* = 1`.

**Verification.** On the pilot's own Hessians, the parabola reproduces direct
evaluation of `tr((W − sQ)H(W − sQ)ᵀ)` to a worst relative gap of `1.6e-14` over
`s ∈ {0.90, 0.98, 1.00, 1.02, 1.05, 1.10}`, and the gain identity to `4e-12`
(*measured*, 12 cells of Qwen3-0.6B). These are machine-precision agreements: the
derivation is checked, not merely believed.

**What it says.** Pooled over the 12 cells, `s* = 0.99888`; with the served
damping, `0.99864`; per cell, between `0.9962` and `1.0008`. The layer objective
is already at its optimum, to within a tenth of a percent, everywhere.

## 3. Why that answer is wrong, and the mechanism

The model's minimum is at 1.02, and moving there is worth 0.836 % of perplexity
(*measured*, four re-encoded arms). So the layer objective is at its own optimum
and the model is not at its.

The mechanism is measurable and was measured in the other direction first. The
Euclidean gain rule — which picks the centroid nearest `⟨x,u⟩` rather than `‖x‖`
— is the *local* optimum over the two admissible gains, and it lost 4.026 % of
perplexity. Its reconstruction places, on average, `0.97073` of each block's
norm against the served rule's `0.99336`: a systematic 2.93 % shrink, on every
block, because `⟨x,u⟩ = ‖x‖·cos θ ≤ ‖x‖`.

That is the whole mechanism. **Squared error does not distinguish a biased error
from a centred one of the same size, and a deep model does.** Decompose the
per-block error into a systematic part and a centred part:

```
E[‖e‖²] = ‖bias‖² + Var
```

Squared error charges both at the same rate. But the two propagate differently
through a stack of layers: centred errors are independent across blocks and
partly cancel when summed into an activation, while a bias is the *same*
direction in every block and adds coherently. Over `n` layers the centred part
grows like `√n` and the coherent part like `n`.

A local objective sees only `E[‖e‖²]` for its own matrix. It will therefore
trade bias for variance whenever that lowers the total — which is exactly what
the Euclidean rule does, and exactly what costs 4 %.

This is not a conjecture about the code. It is what the three measured optima
say: 0.906 minimizes weight error and is far too small, 0.999 minimizes layer
output error, 1.02 minimizes the model. The ordering is monotone in how much of
the stack the objective can see.

## 4. Modelling the endpoint directly

If the layer objective cannot be trusted, the endpoint has to be modelled. Let
`sₖ` scale the reconstruction of matrix `k`, `δₖ = sₖ − 1`, and `L` the model's
mean NLL. To second order around the run as it stands,

```
L(s) ≈ L(1) + Σₖ gₖ δₖ + ½ Σₖ hₖ δₖ² + Σ_{k<l} h_{kl} δₖ δₗ
```

`gₖ` and `hₖ` come from central differences at two evaluations per matrix:

```
gₖ = [L(+ε eₖ) − L(−ε eₖ)] / 2ε          hₖ = [L(+ε eₖ) − 2L(1) + L(−ε eₖ)] / ε²
```

Both are second-order accurate in `ε`, and **exact on a quadratic** — which is
what the module's first test asserts, so a discrepancy there is arithmetic and
not method.

Cross terms are not measured: there are `n(n−1)/2` of them and each costs an
evaluation. The surrogate therefore assumes the matrices act independently. That
assumption is not proved and is not provable here; it is *tested*, by predicting
a combination the probes never saw and measuring it (§6).

### The per-matrix optimum, and why it needs a trust region

The natural answer is `δₖ* = −gₖ/hₖ`, valid when `hₖ > 0`. On measured data most
curvatures are not positive: a quantized run does not sit at a minimum of its own
loss, and 9 of the first 14 probed directions came back concave or flat. For
those, `−g/h` is a maximum or does not exist.

The fix is to minimize the same quadratic over `δ ∈ [−T, T]`, which is defined
for every sign of curvature:

```
h > 0 :  δ* = clamp(−g/h, −T, T)
h < 0 :  δ* = −sign(g)·T      and, when g = 0, either endpoint — never 0,
                              which is the maximum of a concave arc
h = 0 :  δ* = −sign(g)·T,     and 0 when g = 0 as well
```

`T` is a small multiple of `ε`: past it, the parabola is being believed where
nothing was measured.

**Verification.** The closed form above is checked against a 20,001-point scan
of the same quadratic, for seven sign combinations of `(g, h)`. That test found a
real defect in the first implementation: at `g = 0, h < 0` it returned the
centre, which is the interval's worst point.

## 5. Directions the loss cannot see

Some matrices have `gₖ = hₖ = 0` exactly. In Qwen3 these are `q_proj` and
`k_proj`, and the reason is structural: Qwen3 applies an RMS norm per head to
`q` and `k`, and an RMS norm is invariant under scaling of its input. A scale
error on those matrices costs **exactly nothing**.

This is a calibration fact with a direct consequence: effort spent preserving
the scale of `q_proj` and `k_proj` is wasted, and the same effort spent on
`gate_proj` is not. A map that reported these as `NaN` would hide it.

## 6. What the model claims, and how it is caught being wrong

Three claims, in decreasing strength, each with its own test.

1. **Exactness on one matrix.** True by construction: with one matrix moved, the
   surrogate is the second-order Taylor expansion of a function of one variable.
   Nothing is asserted beyond the truncation.
2. **Additivity across matrices.** Assumed, not proved. Tested by moving the
   eight highest-ranked matrices together and measuring. On a two-block pilot
   the prediction was 3.56667 against a measured 3.51326 — 20 % of the move, and
   conservative.
3. **Usefulness as a ranking.** The weakest claim and the one that matters: a
   map that cannot tell an improvement from a regression ranks nothing. That is
   `Residual::agrees_in_sign`, and the error is reported as a fraction of the
   *predicted move*, never of the absolute loss, which is dominated by the model
   and would flatter any prediction.

## 7. What this is not

It is one model, 0.6 billion parameters, one calibration corpus, one evaluation
set. The decomposition of §3 is standard and its consequence here is measured
rather than derived: nothing above proves that coherent errors must dominate in
general, only that on this model the three objectives order as they do and that
a 2.93 % systematic shrink cost 4.026 %.

It models one family of perturbations — a scale per matrix. Bit allocation,
codebook choice and the number of gain levels are not continuous parameters and
do not fit this frame without further work.

And post-hoc scaling is not re-encoding. Scaling quantized weights after the
loop changes neither which level each block picked nor what later columns were
compensated against; scaling the centroids before it changes both. The map
measures the first. The gap between the two is a quantity this programme
measures rather than assumes, by comparing the map's pooled scale against the
sweep's re-encoded minimum of 1.02.
