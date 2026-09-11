# Quantization papers relevant to LLVQ

This dossier examines the first three research directions selected for LLVQ: **KronQ**, **GPTQ-2D with BaKron**, and **REAL-Q**. It covers papers posted between 8 July and 30 August 2026, the current LLVQ implementation, and small exact algebra checks. The purpose is to decide what is worth importing into the Leech-lattice encoder and the fused decoder. It is a research review, not an implementation proposal accepted into the product.

The paper numbers below are labelled **reported** when they come from a paper, **measured** when they already exist in the LLVQ journals, and **computed** when they are derived here or by the accompanying script. No model run, GPU run, paid job, or format change was started for this review.

## Decision in one page

The papers do not point to one drop-in replacement for the current encoder. They separate three problems that should stay separate:

1. **Objective and calibration:** LLVQ's current Hessian is an input covariance captured before the operations inside a transformer block have been quantized. Refreshing that curvature between sublayers, and choosing each Leech gain with the conditional quadratic metric, is the most direct improvement. It reuses the existing factor and preserves the `.llvq` representation.
2. **Output-side sensitivity:** KronQ shows why a gradient covariance is useful for output rotations and bit allocation. Its main column-wise compensation does not use the full output covariance: the left factor cancels from the conditional update. A dense output metric would matter only if the candidate decision or order were changed. KronQ therefore supplies a calibration and preprocessing direction, not a full-G replacement for `GptqFactor::solve_block`.
3. **End-to-end correction:** REAL-Q has the strongest same-family evidence among the papers, including a reported Qwen3-8B W2A16 result. It is also the largest engineering change: an aggregated output Fisher, differentiable loss plumbing, a blockwise Adam step, and a two-block sliding graph. It is suitable for a controlled pilot after instrumentation, not for an unbounded rewrite.

GPTQ-2D and BaKron are useful as algorithmic references. GPTQ-2D proves an exact, work-efficient sweep for a fixed two-sided Kronecker metric; BaKron explores local Kronecker factors and reports model results. Neither paper establishes that a two-sided solver improves a Leech-vector quantizer at LLVQ's served rate. BaKron's headline **2.81 bits** is `log2(7)` for seven symmetric levels, not a packed, scale-inclusive `.llvq` storage measurement.

The recommended order is therefore:

| Priority | Work | What it answers | Why it is bounded |
|---|---|---|---|
| 1 | Conditional Tetra metric and refreshed intra-block factors | Does the existing objective select better Leech words and gains? | No format change; uses current `U` and sequential block loop |
| 2 | KronQ-style gradient traces and output rescaling as an offline diagnostic | Which projections are output-sensitive, and does BiIP change candidate statistics? | Keep rotation optional; do not put a dense `G` into the solver yet |
| 3 | REAL-Q one-block pilot at W2A16/W4A16 | Does an end-to-end gradient correct the residual after analytic quantization? | One model, held-out paired evaluation, explicit memory/time gate |
| 4 | GPTQ-2D/BaKron two-sided sweep | Does a fixed two-sided metric help enough to justify atomic Leech scheduling? | Requires a new ordering proof and a separate kernel path |

The existing LLVQ metric audit already contains the key conditional objective and the current implementation locations; this review treats that document as an input, not as a new result.[^1]

## Decision matrix on the fundamental criteria

The repository defines four product axes—**disk**, **VRAM**, **throughput**, and **quality**—plus three feasibility quantities: **loadable model class**, **encoding cost**, and **noise floor**.[^2] The table scores the transfer to LLVQ, not the paper's abstract contribution. “Reported” refers to a paper result; “computed” refers to the accounting or algebra in this dossier; “unknown” means that the paper does not measure the relevant LLVQ criterion.

| Paper | Gain and fundamental criteria affected | Potential cost and fundamental criteria affected | Feasibility for LLVQ |
|---|---|---|---|
| **KronQ** | **Quality:** reported W2 PPL gains from BiIP plus GPTAQ; output-aware trace allocation can spend bits where Q/K/V/V/O differ. **Disk/VRAM:** potentially neutral at fixed codebook rate; no demonstrated whole-model b/param reduction. **Throughput:** decoder can remain unchanged if rotations are folded into the served representation. | **Encoding cost:** one backward pass and full (H_G) storage; transient calibration memory reaches 1.17 GiB for Llama-3-8B MLPs in the paper's setup. **Throughput/VRAM:** unfurled orthogonal transforms add per-layer work and may require a new kernel path. **Noise floor:** extra Fisher/calibration sampling introduces a new source of drift. **Quality risk:** a diagonal (G) is a no-op in the current rowwise Tetra argmin; dense (G) requires joint decisions. | **High** for trace diagnostics and refreshed intra-block groups; **medium** for output rescaling; **low/medium** for full BiIP in the served kernel. Keep the first step offline and optional. |
| **GPTQ-2D** | **Encoding cost:** exact fixed-order sweep reduces the two-sided propagation work from quartic dense bookkeeping to `O(mn max(m,n))` after factorization. **Quality:** no gain is established; it preserves a chosen trajectory. **Disk/VRAM/throughput:** unchanged if the result is emitted in the existing format. | **Encoding cost/VRAM:** two-sided factors and buffers can be large; factorization remains cubic. **Quality/noise floor:** exactness assumes fixed factors and exact arithmetic; floating-point ties can diverge. **Loadable model class:** no change, but Leech words are atomic 24D decisions and do not align with scalar anti-diagonals. | **High** as a reference implementation and **low/medium** as a production Tetra sweep until the atomic ordering and compensation are proved. |
| **BaKron** | **Quality:** reported PPL improvements from local K-FAC/Shampoo on scalar seven-level quantization. **Encoding cost:** local factors can expose token- or sublayer-specific curvature. **Disk/VRAM/throughput:** no demonstrated served-format gain; the headline 2.81 bits is `log₂ 7`, not LLVQ whole-model storage. | **Encoding cost/VRAM:** local factorization and backprop variants are materially slower and heavier than GPTQ in the paper. **Quality/noise floor:** Base-model PPL and PIQA/Winogrande do not establish an MMLU gain or a Tetra gain; scalar clipping/codebook assumptions do not transfer directly. **Loadable model class:** richer local factors increase calibration pressure without changing served capacity. | **Medium** for a token-dependent MLP weighting diagnostic; **low** for direct adoption in the Leech encoder. Do not use its 2.81-bit number as an LLVQ rate. |
| **REAL-Q** | **Quality:** strongest reported signal: on Qwen3-8B W2A16, KL goes 0.626→0.499 versus GPTQ and average ten-task accuracy 41.06→49.28, at a nominally fixed rate. **Disk/VRAM/throughput:** the emitted code can stay unchanged, so no direct disk or decoder-throughput gain is required. **Loadable model class:** unchanged if Adam only updates trailing dense weights before commitment. | **Encoding cost:** repeated forward/backward passes and Adam states after every column packet; much higher than GPTQ. **VRAM/loadability:** aggregated output Fisher and a two-block autograd graph raise peak memory; this can become the limiting criterion before quality. **Noise floor:** stochastic mini-batches and learning-rate choices add variance; the paper reports no paired LLVQ confidence interval. **Quality risk:** the mean-field Fisher surrogate can reverse tokenwise rankings, and the Adam descent argument is empirical rather than proven. | **Medium** for a one-block oracle-gated pilot; **low** for immediate full-model production. It is the only one of the four that warrants a bounded quality experiment before a kernel redesign. |

The feasibility labels are deliberately asymmetric: a method can be easy to prototype as an offline diagnostic while being unsuitable for the served decoder. A gain in quality at fixed bits is the useful outcome for LLVQ; a paper's lower nominal bit count is not a gain until scales, metadata, embeddings, and the decoder are included in the whole-model b/param accounting.

## 1. KronQ: useful curvature, limited solver change

### What the paper actually changes

The latest available version is KronQ v2 (arXiv v1: 8 July 2026; v2: 8 August 2026). Version 2 corrects the asymmetric drift expansion from the first version, fixes the Kronecker orientation, changes the Fisher wording, and replaces the earlier two-sided bound with a one-sided expected-loss proposition. Any review that quotes the v1 bound as the current theorem is stale.

KronQ approximates the layer Hessian by a Kronecker product of an input covariance (H_X) and an output gradient covariance (H_G). With ΔW = W - Q and ΔX = X_teacher - X_student, its v2 local objective is

\[
J = \operatorname{tr}\!\left[H_G\left(\Delta W H_X\Delta W^T +
2W\Delta X\Delta X^T\Delta W^T\right)\right].
\]

Its BiIP preprocessing rescales rows and columns using diagonal entries of the two factors, then applies orthogonal Hadamard-style transforms. The paper also uses the trace product
\(\operatorname{tr}(H_G)\operatorname{tr}(H_X)\) to rank sublayers for mixed precision. The reported motivation is real: Q, K, and V share the same activation covariance while receiving different downstream gradients, so an activation-only score cannot distinguish them.

The critical detail for LLVQ is Proposition 1. Under the column-wise OBS/GPTAQ update, (H_G) cancels algebraically. The compensation after a column is quantized uses the input-side inverse and the asymmetric drift correction. The full output covariance is still used by BiIP and allocation, but it is not an extra dense factor in the column feedback. This is a positive engineering result: it prevents a needless `d_out × d_out` multiply in the inner loop.

The v2 ablation gives a more useful scale than the v1 headline. On reported WikiText-2 PPL for Llama-2-7B at the paper's low-bit setting, GPTQ is 10.18, GPTAQ is 8.19, input-only processing is 8.46, output-only is 180.43, and both sides are 8.19. On Llama-2-13B the corresponding GPTQ/GPTAQ values are 7.95/6.99; on Llama-3-8B they are 14.52/11.92. These are reported package comparisons without confidence intervals; they do not isolate a pure output-side causal effect. The paper's own factor-swap experiment is a useful caution: on Llama-2-7B W2, YAQA-B recalculated on the same WikiText-2 tokens reports 9.11 versus KronQ's 10.18, while on Llama-3.1-8B KronQ reports 14.32 versus 16.90 for that YAQA-B variant. Neither estimator dominates.

KronQ reports a transient calibration-memory cost for the full (H_G). For Llama-3-8B, its table gives 1.17 GiB for each MLP up/gate factor before release, falling to 0.40 GiB after release; down is 1.16 to 1.10 GiB. These are reported calibration figures with the paper's block size and dtype conventions, not LLVQ measurements. At inference it requires the reversion of both orthogonal transforms, with a stated \(\Theta(d_{in}\log d_{in}+d_{out}\log d_{out})\) overhead per layer.

### Why a diagonal output weight is not enough

For LLVQ's present row-wise candidate loop, a positive diagonal (G) gives

\[
J = \sum_i g_i\,e_i H e_i^T.
\]

Rows still choose independently; multiplying one row's objective by (g_i) does not change its argmin. The same cancellation applies to the conditional suffix update. A dense off-diagonal (G) is different: it couples rows and can change a joint code decision. The accompanying exact check uses

\[
G=\begin{bmatrix}1&0.9\\0.9&1\end{bmatrix},
\quad e_{00}=(0.49,0.49),\quad e_{01}=(0.49,-0.51).
\]

The independent choice has computed cost 0.91238, while the coupled candidate has computed cost 0.05038. That is a two-row toy counterexample, not an LLVQ benchmark. It says precisely what would have to change: a dense (G) requires joint candidate selection or an output rotation, not merely a new scalar weight in the existing row loop.

### Fit to the current LLVQ code

The current implementation already has the right one-sided mechanics. `llvq-quant/src/linalg.rs` factors (H^{-1}=U^TU), and `GptqFactor::solve_block` solves (XU_{QQ}=E) after a block has been quantized. `llvq-quant/src/gptq.rs` then applies the trailing update. The quantizer in `llvq-quant/src/quantizer.rs` receives no (H) or (U); it selects a scale-free Tetra direction and a norm-derived gain before the GPTQ correction.

That ordering leaves a concrete gap: the conditional metric is available only after the candidate has been selected. For a fixed unit direction (u), a gain (a) has quadratic optimum

\[
a^* = \frac{u^T M w}{u^T M u},
\]

where (M) is the current conditional metric. Euclidean norm matching is the special case (M=I) with an additional codebook restriction. The first implementation should score the existing legal gain candidates with (M), retain the old candidate when the score is worse, and log the score margin. It must freeze the row scale for the decision so that an accepted word does not silently change previously emitted words.

KronQ's code confirms this split. Its public implementation deletes the dense (G) after preprocessing and uses the usual input-side compensation. Its calibration pipeline has real intra-block groups (`[k,v,q]`, `[o]`, `[up,gate]`, `[down]`) and refreshes inputs between groups, plus a separate teacher cache for asymmetric correction. Its mixed-precision helper has hard-coded rank defaults and an illustrative average-bit formula; it does not provide an LLVQ-ready byte-budget allocator. It also computes dense (W^TW) or (WW^T) intermediates where a sum-of-squares would suffice.

For LLVQ, the immediate KronQ import is therefore: refreshed sublayer/group calibration, an optional output-side **diagnostic** and row rescaling, and trace-based allocation. The output rotation itself is a later compatibility question because Q/K share RoPE and RMSNorm paths, V does not, and arbitrary row rotations cannot be folded through all of those operations without changing the served kernel.

## 2. GPTQ-2D and BaKron: exact mechanics, separate modeling question

### GPTQ-2D

GPTQ-2D addresses the algorithmic problem \(\|A(Z-X)B\|_F^2\) for fixed nonsingular bases. It rounds one anti-diagonal at a time. The dense reference propagates each rounding error through the lower factor (L) and upper factor (U); the efficient algorithm stores (C=LE), pushes down through (L), and pushes right through (U). The authors prove that the lazy trajectory is identical to the dense trajectory in exact arithmetic, with a sweep cost (O(mn\max(m,n))) rather than (O(m^2n^2)), after the factorization cost.

The claim is narrower than “two-sided GPTQ is better.” The bases are fixed, the order is fixed, and there is no global-optimality claim. The paper explicitly separates the efficient sweep from the unresolved modeling question of how to estimate (A) and (B). A floating-point implementation can also diverge at a nearest-code tie even when the algebraic trajectory is equivalent.

The Leech encoder adds an additional constraint: one 24-dimensional lattice word is an atomic decision, while GPTQ-2D rounds scalar matrix entries on anti-diagonals. To import it, LLVQ would need an anti-diagonal schedule that never splits a codeword, a proof of the corresponding compensation map, and deterministic tie behavior. The current row-parallel path in `gptq.rs` deliberately assumes rows do not interact; a two-sided solver would invalidate that assumption. The right first use of GPTQ-2D is a small exact reference implementation on a synthetic block, not a replacement for the production encoder.

The accompanying script checks the fixed-order identity for 25 exact-rational matrices with dimensions 1 through 5 and unit-triangular test factors. It tests the propagation algorithm only; it does not claim that the result is a better LLVQ code assignment.

### BaKron

BaKron supplies the local-Kronecker modeling side. Its exact algebraic equivalence between naive, anti-diagonal, and recursive compensation is valuable for memory planning, but the finite-grid clipping and codebook behavior of Tetra are outside that equivalence. Its reported experiments use local K-FAC or Shampoo variants, with damping and a small symmetric scalar codebook. On Qwen3-4B-Base at the paper's setting, the reported GPTQ PPL is 14.72, MlpLocal-KFAC is 14.42, MlpLocal-Shampoo is 14.52, and Backprop-KFAC is 15.49. The reported times are 231 s for GPTQ, 443 s for MlpLocal-KFAC, and 941 s for Backprop-KFAC. These are the paper's PPL/time results, not LLVQ measurements, and they compare a Base checkpoint and PIQA/Winogrande rather than the current LLVQ MMLU protocol.

The paper's `2.81 bits` comes from seven symmetric levels, \(\log_2 7=2.80735\), computed here from the stated level count. It excludes the full packed representation question: scales, metadata, alignment, embeddings, and the decoder layout. It must not be compared directly to LLVQ's computed whole-model b/param number.

BaKron's useful LLVQ idea is not a static row multiplier. For an MLP, a local output-aware approximation produces token-dependent weights involving the downstream matrix and the activation derivative. A diagonal-only, token-independent row factor would be homogeneous and disappear from the current rowwise argmin; token-varying weights change the effective (H) and survive that no-op. This is a candidate for an offline diagnostic after the current conditional metric is in place. It needs a Qwen3-aware treatment of RMSNorm and attention paths before any claim about Q/K/V transfer.

## 3. REAL-Q: strongest empirical lead, largest integration cost

REAL-Q (30 August 2026) keeps an analytic GPTQ-style pass, then performs a dynamic correction after each 128-column block. It constructs a full block-output Fisher matrix

\[
F \approx T^{-1}\sum_t g_tg_t^T,
\qquad
L_{Fisher}=\tfrac12 T^{-1}\sum_t \Delta y_t^T F\Delta y_t,
\]

and updates the remaining unquantized columns with one Adam step. A sliding window blends the current and next transformer-block objectives, so each gradient graph spans at most two blocks. Already quantized words stay locked.

The paper is unusually explicit about the approximation. The exact second-order term is (E_t[\Delta y_t^T H_t\Delta y_t]); REAL-Q replaces it with (E_t[\Delta y_t^T E_t[H_t]\Delta y_t]). The error is a covariance between the token Hessian and the token perturbation outer product. A two-token exact calculation in the accompanying script reverses the ranking of two candidate errors under this mean-field replacement: computed exact costs are 4 and 100, while the mean-field costs are 202 and 50.5. This does not invalidate the surrogate; it marks the condition under which a held-out check is necessary.

The paper's theoretical descent statement is sufficient for an SGD step under a smooth true objective. It does not establish Adam descent: the paper itself says the preconditioned Adam direction is not covered by the displayed cosine inequality. The appendix language calling the condition “necessary” should be read as an imprecision; the main derivation is a sufficient condition.

### Evidence relevant to LLVQ

REAL-Q evaluates Qwen3 and Llama families. Its reported Qwen3-8B W2A16 table is the closest evidence to the current low-bit objective: the table reports KL in units of \(10^{-2}\), so GPTQ is 62.6 (=0.626), PPL 17.21, and average ten-task accuracy 41.06; GPTAQ is 60.6 (=0.606), 16.60, and 44.06; GuidedQuant is 58.8 (=0.588), 15.91, and 41.06; REAL-Q is 49.9 (=0.499), 14.94, and 49.28. The derived relative changes are approximately −20.3% KL versus GPTQ, −15.1% versus GuidedQuant, +8.22% accuracy versus GuidedQuant, and +5.22% versus GPTAQ. They are reported single-run comparisons without paired confidence intervals and use W2A16 group size 128 with 256×2048 WikiText-2 calibration, not the LLVQ Leech codebook or MMLU paired dumps.

On Qwen3-4B W4, the reported REAL-Q result is PPL 13.44 and average accuracy 62.91, versus GPTQ PPL 13.58 and 62.29, and GuidedQuant PPL 14.50 and 61.83. On Qwen3-32B W4, the reported average is lower for REAL-Q than GuidedQuant despite lower KL, which is evidence that downstream task accuracy is not a monotonic proxy for fidelity. The paper correctly cautions that post-trained Qwen references are not necessarily extrema of PPL or zero-shot accuracy.

### What LLVQ would have to add

The current `llvq-llm/src/calib.rs` already captures (X^TX), factors it, quantizes a group, decodes it, and forwards the updated block into the next block. It therefore refreshes **between** transformer blocks but not between sublayers within a block. It uses the current student activations for the input covariance and has no differentiable end-to-end loss path in the quantization loop; `window_nll` reduces to a host scalar.

A REAL-Q pilot would require:

* a tensor-valued NLL/Fisher loss that remains in the autograd graph, with an oracle gradient check on the unquantized model;
* an explicit teacher activation/output cache and a two-block forward graph;
* a 128-column schedule translated to 24-dimensional Leech groups, for example a documented 120/144 policy that never splits a word;
* frozen row scales and frozen emitted codes for the analytic prefix, with Adam operating only on trailing dense weights before their code is committed;
* logging of loss, KL, PPL, MMLU, peak memory, and wall time after every packet.

This is a mixed analytic/gradient PTQ procedure, not continuous QAT. Re-encoding an entire `.llvq` file after every Adam step would change the experiment and the accounting. The safe first gate is one Qwen3-4B layer or one transformer block with a teacher/student oracle, then one full paired pilot only if the graph and memory budgets pass.

## Cross-paper conclusions for LLVQ

The three directions improve different terms in the same error chain:

| Question | Best paper signal | LLVQ consequence |
|---|---|---|
| Are current candidate decisions using the right metric? | LLVQ audit plus KronQ's cancellation result | Score legal Tetra gains with the conditional (M); do not multiply the row loss by `diag(G)` |
| Can output sensitivity distinguish projections? | KronQ trace score and BiIP | Measure gradient traces and row anisotropy; keep output rotations optional until Qwen3 operator compatibility is proven |
| Can a two-sided factor be swept efficiently? | GPTQ-2D exact trajectory | Preserve as a reference path; atomic Leech ordering is a new proof obligation |
| Does local output structure matter? | BaKron local K-FAC/Shampoo results | Test token-dependent MLP weighting, not a homogeneous row scalar |
| Can end-to-end residuals be corrected? | REAL-Q Qwen3-8B W2A16 result | Run a bounded autograd pilot after the oracle and memory gates |

The decisive first measurement should compare three candidate selectors on the **same encoded representation**: current Euclidean gain, conditional-metric gain, and conditional-metric gain with the current candidate retained on a score regression. The evaluation should report calibration objective, held-out NLL/PPL, paired MMLU, code histogram, and the score margin. A result that changes b/param, row-scale convention, or decoder layout is a different experiment and must be accounted separately.

The second measurement should refresh (H) and the student activations at the existing intra-block group boundaries. This is lower risk than importing a dense output metric and directly tests the information-misalignment hypothesis shared by KronQ and REAL-Q. It also aligns with the current LLVQ roadmap's quality axis.

The third measurement, if the first two are positive, is a REAL-Q-style residual step on a single block. GPTQ-2D/BaKron should follow only if the conditional one-sided metric leaves a measured gap large enough to justify joint row coupling and a new Leech schedule.

## Reproducibility record

The exact checks can be rerun with:

```text
python3 docs/recherche-quantification-2026-09-08/checks.py
```

The current result is stored in `checks.json`: five checks pass, including 25 exact-rational unit-triangular GPTQ-2D dimensions, the Schur/trailing-factor identity, the KronQ suffix cancellation, the dense-output coupling counterexample, and the Fisher mean-field ranking reversal. These checks are deliberately small and do not replace an LLVQ model evaluation.

[^1]: LLVQ project, [Metric hypothesis for Tetra, 2026-09-08](../hypothese-metrique-tetra-2026-09-08.md), consulted as the existing analytical baseline.
[^2]: LLVQ project, [METHODE.md](../METHODE.md), §1, “fundamental criteria” and feasibility quantities.

## Sources

1. LLVQ project, [Metric hypothesis for Tetra, 2026-09-08](../hypothese-metrique-tetra-2026-09-08.md). Local analytical audit of (J=\operatorname{tr}(EHE^T)), the Schur complement, current factorization, and candidate-selection gap.
2. Lee et al., [KronQ: LLM Quantization via Kronecker-Factored Hessian, arXiv:2607.07964v2](https://arxiv.org/html/2607.07964v2). v2 posted 2026-08-08; official code at [Intelligent-Computing-Lab-Panda/KronQ](https://github.com/Intelligent-Computing-Lab-Panda/KronQ).
3. Birnick, [GPTQ-2D: Efficient Two-Sided Quantization, arXiv:2607.27042v1](https://arxiv.org/html/2607.27042). Posted 2026-07-29.
4. Birnick, [BaKron: Backpropagation-Free Kronecker-Factored Quantization, arXiv:2608.06291v1](https://arxiv.org/html/2608.06291). Posted 2026-08-06.
5. [REAL-Q: E2E LLM Quantization via Dynamic Gradient Descent, arXiv:2609.00049v1](https://arxiv.org/html/2609.00049). Posted 2026-08-30.
6. LLVQ project, [Roadmap quality axis](../ROADMAP-QUALITY.md), [state](../ETAT.md), and source files cited inline. These are the repository's measured/computed baseline and implementation contract.
