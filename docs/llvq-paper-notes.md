# Reading notes, arXiv:2603.11021v2 (LLVQ)

> *Leech Lattice Vector Quantization for Efficient LLM Compression*
> van der Ouderaa, van Baalen, Whatmough, Nagel (Qualcomm AI Research), v2 of
> 7 July 2026, 26 pages.
>
> Recorded on 2026-07-28 **by image rendering** of the PDF (see "How to read
> the PDF" below). Every number in this file was read on screen, not
> extracted by script.

## How to read the PDF (the trap, solved)

Text extraction from the PDF is corrupted: the font encoding is shifted, and
digits double up (`0.084` comes out as `0.1084`). **Never trust `pdftotext`
on this document.** Image rendering is perfect:

```python
import fitz  # pymupdf
d = fitz.open("2603.11021v2.pdf")
p = d[6]                                   # page 7, index 0
clip = fitz.Rect(p.rect.x0+60, p.rect.y0+395, p.rect.x0+330, p.rect.y0+450)
p.get_pixmap(matrix=fitz.Matrix(9, 9), clip=clip).save("table3.png")
```

A ×9 zoom on the area of a table makes it perfectly legible.

## Paper outline

| § | Content | Useful for |
|---|---|---|
| 2 | Λ₂₄, ball truncations, extended Golay code, classes and leaders | G1/G2, done |
| 3.1 | Search extended to `Λ₂₄(m)`, Euclidean and angular metrics | G2b, done |
| 3.2 | Hierarchical indexing scheme (shell → class → local symmetry) | G3, done |
| 3.3 | Spherical GPTQ = retraction on a product of spheres | **G5** |
| 4 | Gaussian source, SQNR and retention (Table 3) | G4, done |
| 5 | LLM results (Tables 4, 5, 6) | **G5, targets** |
| A | Dequantizer (index → vector) | G3/G6 |
| **B** | **Algorithm 1**, shape–gain plus Hessian corrections | **G5** |
| C | Fused CUDA kernel (Table 7) | G6 |
| D | Shape–gain and spherical codes from a lattice | G5 |
| E | Quantizer/dequantizer block diagrams (Fig. 2–5) | G5 |
| F | Closed-form optimal scales, Hessian corrections | **G5** |
| G | Single shell vs union of shells (Fig. 6) | G4/G6 |
| H | Spherical shaping vs shape–gain (Table 8) | **G4** |
| **I** | **Spherical GPTQ, Algorithms 2 and 3**, Hadamard ablation (Table 9) | **G5** |
| J | Llama-3.2 1B and 3B (Table 10) | G5 |

---

## Table 3 (§4), Gaussian retention at 2 bits/dim

| Method | Dim | MSE | SQNR (bits) | Ret (%) |
|---|---|---|---|---|
| Uniform | 1 | 0.15 | 1.37 | 69 |
| Lloyd-Max | 1 | 0.12 | 1.53 | 77 |
| E8 (cubic) | 8 | 0.103 | 1.64 | 82.0 |
| **LLVQ/Leech [spherical shaping]** | 24 | **0.084** | **1.79** | **89.4** |
| **LLVQ/Leech [shape-gain]** | 24 | **0.078** | **1.84** | **92.1** |
| Theoretical limit | n/a | 0.0625 | 2 | 100 |

Self-consistent: −½log₂(0.084) = 1.7885; 1.79/2 = 89.4%.

## Table 8 (Appendix H), the same comparison in detail

This is **the** table to compare with our G4, because it names the codebook.

| Method | Code | Bits/dim | MSE | SQNR | Ret (%) |
|---|---|---|---|---|---|
| Leech (spherical bounding) | `Λ₂₄(13)` | 2.0 | 0.084 | 1.787 | 89.37 |
| Leech (shape-gain) | `norm(Λ₂₄(13))` + 0 gain bits | 2.00000 + 0 | 0.085 | 1.782 | 89.12 |
| **Leech (shape-gain)** | `norm(Λ₂₄(12))` + 1 gain bit | 1.95833 + 0.04167 | **0.078** | **1.843** | **92.14** |
| Leech (shape-gain) | `norm(Λ₂₄(11))` + 2 gain bits | 1.91667 + 0.08333 | 0.080 | 1.825 | 91.24 |
| Leech (shape-gain) | `norm(Λ₂₄(10))` + 4 gain bits | 1.83333 + 0.16667 | 0.085 | 1.780 | 89.01 |

The paper's conclusions:
1. Shape–gain beats spherical shaping.
2. The "1/n of the bits to the gain" heuristic (2 bits for n = 24) is
   reasonable but **not optimal**: the empirical optimum is **1 bit**.
3. Recommendation: 1 or 2 gain bits per 24-vector at 2 bits/dim.

## Table 6 (§5.3), LLM targets, **our G5 reference**

Wikitext-2 at 4096 context, the authors' unified pipeline, 2 bits/weight.

**Qwen3-4B** (the smallest model in the paper that carries numbers):

| Method | FT | Wiki ↓ | MMLU ↑ | CSR ↑ |
|---|---|---|---|---|
| Baseline FP16 | n/a | 12.41 | 70.2 | 71.2 |
| GPTQ + Rotation (Quarot) | no | 280.7 | 26.3 | 43.6 |
| Quip#/E8P12 | no | 21.15 | 48.6 | 57.2 |
| QTIP (3INST) | no | 17.04 | 57.4 | 63.5 |
| LLVQ [spherical shaping] | no | **21.80** | 50.5 | 58.7 |
| LLVQ [shape-gain, 2 bit gain] | no | **15.54** | 59.3 | **64.1** |
| LLVQ [shape-gain, 0 bit gain] | no | 17.05 | **60.7** | 63.6 |
| Quip#/E8P12 | yes | 10.52 | 52.9 | 65.2 |
| QTIP (3INST) | yes | 9.61 | 59.5 | 66.9 |
| LLVQ [spherical shaping] | yes | 10.13 | 54.9 | 65.1 |
| LLVQ [shape-gain, 2 bit gain] | yes | 9.51 | 60.9 | 67.6 |
| LLVQ [shape-gain, 0 bit gain] | yes | **9.26** | **62.8** | 66.1 |

Qwen3-8B, without FT: baseline 8.99 / QTIP 11.17 / LLVQ sg-2bit 10.82 /
LLVQ sg-0bit **10.19**. With FT: LLVQ sg-0bit **7.59**.

> **Spherical shaping loses to QTIP** (21.80 vs 17.04 on Qwen3-4B without
> fine-tuning). Only **shape–gain** wins. The "fine-tuning" here is no more
> than learning the per-column scales (< 0.001 bit/weight, ~52M tokens). It
> is not end-to-end training.

---

## Algorithm 1 (Appendix B), shape–gain with Hessian corrections

```
Inputs: W ∈ R^{d_out × d_in}, X ∈ R^{N × d_in} (calibration),
        block size b = 24, direction quantizer Q_dir (Leech),
        optional gain quantizer Q_gain

 1  for each layer l:
 2      H  ← (1/N)·Xᵀ X
 3      U  ← chol(H⁻¹)ᵀ                      # H⁻¹ = Uᵀ U, U upper triangular
 4      W̃  ← W^(l)
 5      partition {1..d_in} into blocks Q_1..Q_m of size b (the last may be shorter)
 6      for t = 1..m:
 7          Q ← Q_t,   R ← ∪_{u>t} Q_u
 8          for each row i = 1..d_out (in parallel):
 9              v  ← W̃_{i,Q}
10              v̂  ← ‖v‖₂ · Q_dir(v/‖v‖₂)             # gain reset
11              option: v̂ ← Q_gain(‖v‖₂) · Q_dir(v/‖v‖₂)
12              Ŵ_{i,Q} ← v̂
13          end
14          E_{:,Q} ← W̃_{:,Q} − Ŵ_{:,Q}
15          W̃_{:,Q} ← Ŵ_{:,Q}
16          if R ≠ ∅:
17              W̃_{:,R} ← W̃_{:,R} − (E_{:,Q} U_QQ⁻¹) U_QR
18          end
19      end
20      W^(l) ← W̃
21  end
```

Points that matter:
- **Blocks of input channels**, left→right; the `d_out` rows are independent
  and parallelize.
- The error `E` is computed on the **current** `W̃` (already compensated),
  not on the original `W`. Line 14, the standard GPTQ convention.
- The line-17 correction is a right triangular solve in `U_QQ`, never an
  explicit inversion.

## Algorithm 3 (Appendix I.1), Spherical GPTQ plus group scales

This is the paper's **recommended configuration** (0 gain bits).

```
Inputs: W ∈ R^{k × d}, H SPD ∈ R^{d × d}, blocks Q_1..Q_m, quantizer Q, damping λ ≥ 0

 1  U ← chol(H⁻¹)ᵀ
 2  W̃ ← W
 3  for t = 1..m:
 4      Q ← Q_t,  R ← ∪_{u>t} Q_u
 5      W̃_{:,Q} ← (‖W̃_{:,Q}‖ / ‖Q(W̃_{:,Q})‖) · Q(W̃_{:,Q})       # retraction
 6      E_{:,Q} ← W_{:,Q} − W̃_{:,Q}
 7      if R ≠ ∅: W̃_{:,R} ← W̃_{:,R} − (E_{:,Q} U_QQ⁻¹) U_QR
10  end
    # final per-row scale refinement, in the Hessian metric
11  for i = 1..k:
12      M_i[p,q] ← W̃_{i,Q_p} H_{Q_p Q_q} W̃_{i,Q_q}ᵀ      ∀ p,q ∈ {1..m}
13      r_i[p]   ← W̃_{i,Q_p} H_{Q_p,:} W_{i,:}ᵀ           ∀ p
14      s_i      ← (M_i + λI)⁻¹ r_i
15      for p = 1..m: W̃_{i,Q_p} ← s_i[p] · W̃_{i,Q_p}
18  end
```

> **Two notation ambiguities to settle at implementation time.**
> (a) Line 5 writes the retraction with a Frobenius norm over the whole
> column block, while the text of Appendix I defines it **per row**
> (Eq. 17: `Ŵ_{i,B} = (‖W_{i,B}‖₂ / ‖W̃_{i,B}‖₂)·W̃_{i,B}`). The text is
> explicit: "quantization is performed row-wise on each row-block […] and we
> apply the same retraction per row". So **implement per row**.
> (b) Line 6 writes `E ← W − W̃` with the **original** `W`, while Algorithm 1
> (line 14) and Algorithm 2 (line 6) use the **compensated** `W̃`. The
> standard GPTQ convention is the one of Alg. 1/2; Algorithm 3 is probably a
> shorthand. **Follow Alg. 1.**

## Closed-form optimal scales (Appendix F.1)

The shape quantizer is **scale-invariant** (`q(sw) = q(w)` for `s > 0`), so
there is no line search over β. The scale that minimizes the reconstruction
error in weight space is the projection:

```
β* = argmin_β ‖w − β q‖²  =  (qᵀ w)/(qᵀ q)          with q = q(w)
```

and per block `β*_i = q(w_i)ᵀ w_i / (q(w_i)ᵀ q(w_i))`.

In activation space (local output error), with
`A := [q(w_1)x_1, …, q(w_G)x_G]` and `b := Wx`, the optimal per-group scales
are the least squares solution `β* = (AᵀA + λI)⁻¹ Aᵀ b`.

## Hessian corrections (Appendix F.2)

Standard local objective: `L = E‖ΔW x‖² = Tr(ΔW H ΔWᵀ)` with `H = E[xxᵀ]`,
approximated by `Ĥ = (1/N) XᵀX`. Partitioning into `Q` (quantized) and `R`
(remaining), the analytic correction is

```
Δw*_R = − H_RR⁻¹ H_RQ Δw_Q        ⟺        ΔW*_{:,R} = − ΔW_{:,Q} H_QQ⁻¹ H_QR
```

implemented without an inverse through the Cholesky of `H⁻¹` (the GPTQ form).

> The paper states the limitation itself: this local objective treats layers
> as decoupled and ignores inter-layer error propagation. More faithful
> curvature surrogates would do better but cost backward passes. **The gains
> from a better Hessian are orthogonal to LLVQ**, so out of scope for G5,
> where the comparison has to be made at equal correction.

## Hadamard ablation (Appendix I.2, Table 9, Llama-2 7B, without FT)

> **Transcribed in full by image rendering on 2026-08-04.** Llama-2 7B
> baseline: 5.12 / 45.7 / 70.4.

| Code | Correction | Hadamard | Wiki ↓ | MMLU ↑ | CSR ↑ |
|---|---|---|---|---|---|
| LLVQ [spherical shaping] | GPTQ | none | **191.90** | 24.0 | **53.5** |
| LLVQ [spherical shaping] | GPTQ | Input | 6.80 | 35.1 | 65.4 |
| LLVQ [spherical shaping] | GPTQ | Input+Output | 7.61 | 33.4 | 62.1 |
| LLVQ [sph. shaping] (forced ang.) | **Spherical GPTQ** | none | **6.90** | 37.4 | **63.8** |
| LLVQ [sph. shaping] (forced ang.) | Spherical GPTQ | Input | 6.70 | 35.1 | 65.4 |
| LLVQ [sph. shaping] (forced ang.) | Spherical GPTQ | Input+Output | 6.75 | 36.9 | 63.8 |
| LLVQ [shape-gain 2 bit] (forced eucl.) | GPTQ | none | 13.17 | **26.5** | **58.5** |
| LLVQ [shape-gain 2 bit] (forced eucl.) | GPTQ | Input | 7.28 | 34.1 | 62.8 |
| LLVQ [shape-gain 2 bit] (forced eucl.) | GPTQ | Input+Output | 7.31 | 35.3 | 62.8 |
| LLVQ [shape-gain 2 bit] | **Spherical GPTQ** | none | **7.27** | **29.8** | 61.5 |
| LLVQ [shape-gain 2 bit] | Spherical GPTQ | Input | 6.90 | 36.0 | 63.6 |
| LLVQ [shape-gain 2 bit] | Spherical GPTQ | Input+Output | 6.83 | 34.9 | 64.6 |

Values to reject wherever they turn up: Wiki 91.90 (a leading `1` lost by
text extraction), CSR 37.7 · 65.9 · 56.5, MMLU 26.3 · 29.3. The image
rendering of 2026-08-04 supersedes them.

**The most spectacular result of the paper**: without rotation, Euclidean
GPTQ collapses (**191.90** perplexity) where Spherical GPTQ holds (6.90).
Radial drift is the dominant failure mode, and norm preservation removes it.
Hence the *Hadamard-free* PTQ.

> Comparing 29.3, no rotation, with 34.9, Input+Output, gives +5.6 pp of
> MMLU. That figure covers the **whole** rotation, input stage included.
> With `Input` fixed, our configuration, the output stage is worth
> −1.7 · **+1.8** · **+1.2** · −1.1 pp over the four families.
> **Mean ≈ 0**. Output rotation therefore cannot explain our 4.8 pp deficit
> on MMLU.

The appendix's conclusions:
1. Spherical GPTQ improves on Euclidean GPTQ without touching the codebook.
2. It is **the more effective the lower the angular distortion of the code**,
   so particularly for LLVQ.
3. LLVQ wins under both corrections: the advantage comes from the
   *representation*, not from the correction heuristic.
4. **The bit budget slides toward the directions**: with a code of low
   angular distortion, the optimum moves from 2 gain bits (Euclidean GPTQ) to
   **0 gain bits** (Spherical GPTQ). All the capacity goes to the directions,
   the magnitudes being held by the spherical constraint in high precision
   during GPTQ, then by the closed-form update of Algorithm 3.

## Single shell vs union (Appendix G)

> **Re-read by image rendering on 2026-08-04**, and precision matters here:
> this is the section we are questioning the authors about.

**What they measure**: the angular distance to the nearest neighbor,
`D(x, q(x)) = arccos(xᵀq(x))/π`, on a **radially uniform** source
(normalized Gaussian), as a function of `log₂(N)/d`. Figure 6, violin plots.

- **Key finding 1, "Union of shells provide lowest angular distortion"**: the
  union gives "a slightly better **Gaussian rate–distortion curves**"
  compared with the individual shells. Exact wording of the last sentence:
  > "We therefore adopt this approach **in our method** and recommend doing
  > the same."
- **Key finding 2, "Single shell provides a simpler algorithm"**: the gap is
  **small**, and "from a hardware perspective, using a single shell offers
  significant advantages. In particular, a constant norm implies a fixed
  scaling between dot products, eliminating the need to rescale intermediate
  dot product results before aggregation (as in group-wise or block
  quantization), along with its associated complications."

> **Two points to get right when citing this section.**
> 1. Quote the sentence with "in our method" in it, and with the ellipsis. A
>    version that drops either one must not go to the authors.
> 2. Key finding 1 explicitly names the Gaussian rate–distortion curves.
>    That is our metric. Their claim therefore covers what we measure, and
>    our measurement (the union wins at equal rate) confirms them rather
>    than contradicting them.
>
> The hardware argument is theirs as well: Key finding 2 states it in full.
> The one question that really stays open: they name the hardware advantage
> and adopt the union anyway. Is that on the distortion curve alone, or did
> they measure the cost of rescaling inside a multi-shell kernel? That is
> what to ask them.

## Table 7 (Appendix C), fused CUDA kernel

| Shape | Kernel | Time |
|---|---|---|
| FP16 matvec | (4096×4096)·(4096×1) | 16.3 µs |
| FP16 matvec | (4096×4104)·(4104×1) | 17.69 µs |
| **LLVQ-FP16 (fused dequant + matvec)** | (4096×4104)·(4104×1) | **11.94 µs** (1.36× / 1.48×) |

The paper is specific: the kernel is limited to **a single shell (M = 3)**
"for simplicity", it is **slower than QTIP**, and the authors declare
low-level optimization "largely orthogonal" to their contribution.

## Dequantizer (Appendix A)

Global index → vector, in four steps: (1) identify the shell by searching the
table of cumulative counts `N(k) < I ≤ N(k+1)`; (2) identify the class from
the cumulative offsets of the shell; (3) unfold the local symmetries with
`r = I_class mod A`, `I' = ⌊I_class/A⌋`, `s = I' mod 2^B`, `I'' = ⌊I'/2^B⌋`,
where `r` selects the Golay refinement, `s` the sign pattern and `I''` the
permutation rank; (4) reconstruct from the leader.

No dependency between vectors, no wide memory access: trivially
parallelizable in blocks of 24. That is the argument for the GPU kernel.
