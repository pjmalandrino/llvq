# Tetra error geometry: code audit and proposed discrimination

The code confirms a missing metric in discrete selection. It does not establish the cause of the quality loss.
This is an analytical audit of `noyau/tetra48`, dated 2026-09-08, without new model measurements.
The experiment outline below is not a timestamped preregistration or a launch decision.

## Evidence and corrections

Tetra scores 53.49 MMLU and 16.1569 perplexity, *measured* in the
[Tetra campaign](mesures/tetra-4b-2026-09-06.txt).
The pasted hypothesis assigns it Planes14's 55.59 and 16.9422.
Restoring Tetra's V projections to int4 yields +3.47 pp, CI95 [+1.42; +5.57],
*measured* in [Q5](mesures/q5-tetra-2026-09-06.txt).
The resulting model costs 2.8138 b/param, embedding included, *computed* in that journal.
The pasted +3.60 pp belongs to the earlier Planes14 intervention.

These interventions establish that restoring V helps the tested quantized model.
They do not compare projections at equal distortion or isolate radial error.
They restore every layer's V jointly, so they do not establish a layer ranking.
Int4 restoration also changes the quantization procedure and its coordinate basis.
Its measured gain cannot be attributed to the bit count alone.

Q, K and V share their input Hessian: [model.rs](../llvq-llm/src/model.rs), `Act::consumers`.
Their differing sensitivity cannot follow from different input spectra alone.
Their weight errors, normalization, head geometry and downstream operators can differ.

## The metric needed at a block decision

Let H be the rotated, regularized matrix actually supplied to GPTQ.
For a completed layer, its reconstruction objective is

\[
J=\operatorname{tr}(EHE^T),\qquad E=W-\widehat W.
\]

Without regularization, H is the uncentered activation second moment.
This quadratic equals average squared output error on those activations.
It is not merely an approximation of weight MSE.
It remains a surrogate for final language-model loss.

During encoding, let B denote the current block and R the remaining columns.
The prefix is fixed, and w is the current compensated target.
For a column-vector residual e on B, optimizing the continuous suffix gives

\[
\min_d\begin{bmatrix}e\\d\end{bmatrix}^{T}
\begin{bmatrix}H_{BB}&H_{BR}\\H_{RB}&H_{RR}\end{bmatrix}
\begin{bmatrix}e\\d\end{bmatrix}
=e^TS_Be,
\qquad S_B=H_{BB}-H_{BR}H_{RR}^{-1}H_{RB}.
\]

The optimum is d = −H_RR⁻¹H_RB e.
This is an exact continuous-relaxation result, not a guarantee about future discrete choices.
The final block has no suffix and uses S_B = H_BB.

The implementation already stores U such that H⁻¹ = UᵀU.
At each step, elimination of the prefix leaves the corresponding suffix factor.
For a row-vector residual, the conditional cost is

\[
eS_Be^T=\|eU_{BB}^{-1}\|_2^2,
\qquad S_B=U_{BB}^{-1}U_{BB}^{-T}.
\]

[GptqFactor::solve_block](../llvq-quant/src/linalg.rs) already computes that triangular solve.
It therefore supplies the local scoring primitive without retaining a second dense Hessian.
In [gptq.rs](../llvq-quant/src/gptq.rs), the solve occurs after `quant.quantize` chooses the code.
[TetraShapeGain::quantize](../llvq-quant/src/quantizer.rs) receives neither H nor U.
Its direction and gain selection cannot use this conditional metric.

Scalar rounding minimizes any positive scalar multiple of squared residual.
A vector block has a matrix-valued curvature, so that argument no longer applies.
This is a concrete optimizer gap, separate from downstream loss weighting.

SchurQuant derives the same suffix-conditioned objective for scalar affine groups.
Its grids and optimization algorithm differ from Tetra's fixed lattice.
Its gains are not predictions for this repository.
Source: [SchurQuant, sections 2–3](https://arxiv.org/html/2608.15567v1).

## Gain selection is independently testable

The current gain is nearest to the compensated block norm divided by the row scale.
For a fixed unit direction u, write the decoded candidate as a u, with a = s g.
In any positive-definite metric M,

\[
D(a)=(w-au)^TM(w-au)
=w^TMw-2a u^TMw+a^2u^TMu.
\]

The continuous optimum is a* = (uᵀMw)/(uᵀMu).
Both legal gains can instead be scored directly, preserving the stored representation.
For M = I, a* = uᵀw = ‖w‖ cos θ, not ‖w‖.
Thus the current norm-based rule need not minimize even Euclidean reconstruction error.
That is a design tradeoff: preserving magnitude may compose better through the network.

The smallest useful intervention retains the direction, row scale and codebook.
It compares the current norm rule with Euclidean gain selection and conditional-metric gain selection.
The Euclidean arm separates radial fitting from curvature weighting.
Including the baseline gain ensures non-increase of the chosen local objective at the same compensated target.
It guarantees neither better later choices nor better model quality.

With M = S_B, transformed w and u make both gain scores inexpensive quadratic evaluations.
Wall-clock overhead remains unmeasured.
Re-ranking several directions comes later, using the same candidate set for both metrics.
Otherwise candidate diversity and metric choice become confounded.

## V has a concrete normalization asymmetry

[Attention::forward_cached](../llvq-llm/src/model.rs) applies Q/K RMSNorm before attention.
V has no corresponding normalization.
For a positive scalar c, RMSNorm(cq) approximately equals RMSNorm(q), ignoring epsilon.
This cancellation applies to uniform scaling within a head, not arbitrary blockwise errors.
V retains such scaling before attention pooling and the output projection.

Holding attention probabilities P and the output projection fixed gives

\[
\delta Y=P\,\delta V\,W_O^T.
\]

This map mixes tokens and output channels.
The V projection's input Hessian contains neither P nor W_O.
Grouped-query attention also reuses values across query heads.
Reuse is not evidence of a fixed amplification factor.
Attention averaging can attenuate errors, depending on their correlations.
K is reused too, so reuse alone cannot explain V's rank.

A useful prediction is greater sensitivity to headwise radial perturbations on V than on Q/K.
Compare projection error before normalization, after normalization, and after the attention sublayer.
Separate coherent head scaling from angular error and nonuniform channel scaling.
If V remains unusually costly at matched post-normalization distortion, downstream geometry becomes more plausible.

## Three arguments that do not establish the mechanism

**Anisotropy alone is insufficient.** Orthogonal rotation preserves H's eigenvalues.
It can nevertheless flatten its diagonal and change each block's conditional geometry.
Gaussian-looking rotated weights do not imply isotropic activations.
Conversely, a broad full spectrum does not prove that S_B strongly changes candidate rankings.

The ratio eᵀHe/eᵀe is a Rayleigh quotient.
It varies with activation scale and error orientation, even without a quantizer defect.
Normalize it by tr(H)/dimension before comparing layers, and state which H is used.
More directly, measure candidate-ranking reversals and conditional regret.
Cross-block terms prevent summing isolated final-block distortions into full-layer reconstruction loss.

**Design C does not solve the constrained optimum.** It performs free-norm encoding,
a continuous scale solve, then snaps magnitudes back to the gain grid.
The final snap has no subsequent GPTQ compensation.
The regularized solve and the discrete projection optimize different problems.
Its failure cannot establish that the best representable Hessian solution hurts the model.
Measure objectives before the solve, after the solve, and after snapping to locate the damage.

**A fixed row scale is not a defect by itself.** It defines one decodable codebook throughout the sweep.
Changing it per block would require additional metadata.
Optimizing one shared row scale through repeated complete sweeps could preserve the format.
Its quality and encoding cost are unknown.

## A staged discrimination protocol

The first stage would inspect retained data before collecting activations.
A sealed artifact alone cannot recover its calibration Hessian or compensated targets.
[f1recdump](../llvq-llm/examples/f1recdump.rs) writes sampled targets, row scales and matrix/block identifiers.
Its output writer does not retain the Hessian, factor, or activations.
Those samples alone cannot establish conditional anisotropy.
Remote retention has not been inventoried for this audit; no replay cost is quoted.

Before a model run, verify the scoring identity against an independent tiny dense minimization.
Use anisotropic cases where Euclidean and conditional rankings reverse, plus an isotropic control.
Verify both gains reconstruct from their stored codes.
Mutation must break these checks before the instrument is qualified.

The proposed model arms are:

| Arm | Direction | Gain decision | Question |
|---|---|---|---|
| A | current Tetra | current norm rule | baseline |
| B | same selection rule | Euclidean score | radial fit |
| C | same selection rule | conditional score | curvature beyond radial fit |

Use identical calibration draws, damping, rotation, matrix order and row-scale rules.
Each arm follows its own compensated trajectory after the first differing choice.
Record counterfactual scores on identical targets separately from completed-arm comparisons.
For a later direction experiment, retain the baseline candidate in a shared candidate pool.

Record local conditional costs, gain occupancy, norm drift and completed-layer output reconstruction.
Evaluate reconstruction on held-out sequences as well as calibration sequences.
Report full-depth held-out NLL, logits and paired MMLU as distinct outcomes.
Use sequence or subject units for uncertainty; weight blocks are not independent model trials.
Choose layers and thresholds before examining their results.
Any magnitude-changing experiment requires the repository's full-depth check before promotion.

| Observation | Interpretation |
|---|---|
| Candidate rankings barely change | little opportunity in the tested pool; does not rule out better candidates |
| C beats B locally and on held-out reconstruction | supports conditional curvature as an optimizer improvement |
| Training reconstruction improves, held-out reconstruction worsens | supports calibration mismatch or overfitting |
| Held-out reconstruction improves, model NLL worsens | supports composition or surrogate mismatch |
| NLL improves, MMLU remains unresolved | no established task-quality gain |
| V radial errors survive where Q/K errors attenuate | supports the normalization mechanism |
| Otherwise | mechanism unresolved; no adoption claim |

A fixed-budget bit-allocation experiment is separate.
Restoring V adds information; it does not demonstrate a gain at constant memory.
That claim needs a named donor projection and paired evaluation of both changes together.

## Research priority

The strongest code-level lead is conditional-metric gain selection with unchanged representation.
The strongest V-specific mechanism is normalization asymmetry plus downstream mixing.
Calibration mismatch remains plausible: Hessians are captured before quantizing operations within each transformer block.
Shrinkage's existing benefit also argues against assuming that fitting calibration curvature harder must help.

End-loss guidance is an established research direction, including vector quantization in GuidedQuant.
It changes the objective and requires additional gradient information.
It should remain distinct from fixing discrete selection under the existing objective.
Source: [GuidedQuant](https://arxiv.org/html/2505.07004v4).

No present evidence assigns the dominant quality loss to any one of these mechanisms.
The staged comparisons above make those explanations compete through observable predictions.
