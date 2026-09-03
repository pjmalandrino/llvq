# LLVQ in Rust, implementation plan

> **Status, last reviewed 2026-08-08. This plan is executed up to and including Phase 6.** G1 to G6
> are green; Phase 6 is done on Metal **and** on CUDA, and its gate is passed: the multi-shell
> kernel m ≤ 12 works, exact row by row against the f64 reference on the model's 1,105,920 rows,
> and it delivers **2.03–2.09× FP16** on Metal, **2.14×** on L40S with the `Planes14` binary
> layout. It is **wired** into the model (`fused_cuda` + `bin/fusedrun`): 48.7 tok/s in 2.96 GB
> against 43.6 in 8.04. Point 1 of the gate ("reproduce the single-shell kernel at ≥ 1.36×") was
> skipped: we went straight to multi-shell.
> Full state: [`rapport-etat-2026-08-07.md`](archive/rapport-etat-2026-08-07.md).
> Caveat: the sentences "the multi-shell kernel does not exist / exists nowhere" describe the
> **published state of the art**, not ours, and that is still how they must be read. Here, it
> exists. The one item left in the plan is **step 5 of Phase 5**: the document-extraction business
> benchmark, never built.

> **Paper: [Leech Lattice Vector Quantization for Efficient LLM Compression, arXiv:2603.11021](https://arxiv.org/abs/2603.11021)** (v2, 2026-07-07)
> van der Ouderaa, van Baalen, Whatmough, Nagel (Qualcomm AI Research).

Version 2 of this plan, written after reading the full PDF. v1 relied on search-engine summaries
and contained three substantive errors, corrected here and flagged in §7.

Required references, in reading order:

1. **Adoul & Barth (1988)**, *Nearest neighbor algorithm for spherical codes from the Leech
   lattice*, IEEE Trans. Inf. Theory 34(5):1188–1202. **This is the base algorithm that LLVQ
   extends.** Without this paper, Phase 2 is infeasible.
2. Conway & Sloane, *Sphere Packings, Lattices and Groups* (3rd ed., 2013), ch. 10, construction
   of Λ₂₄ from the extended Golay code.
3. [QuIP#, arXiv:2402.04396](https://arxiv.org/abs/2402.04396): E8P codebook, baseline.
4. [QTIP, arXiv:2406.11235](https://arxiv.org/abs/2406.11235): the speed reference.
5. [GPTQ, arXiv:2210.17323](https://arxiv.org/abs/2210.17323): the Hessian correction loop.

---

## 1. What LLVQ does, precisely

### 1.1 The wall the paper gets past

VQ in dimension *d* at *b* bits/dim ⇒ 2^(bd) codewords. In dimension 24 at 2 bits/dim: 2⁴⁸
points, ~280 TB. Impossible to materialize. **That is exactly why QuIP# took E8 in
dimension 8**, whose E8P codebook fits in 2¹⁶ entries reduced to a 2⁸ table by symmetry. LLVQ
stores nothing: it computes the points from the structure of the extended Golay code.

### 1.2 Construction (§2.3 of the paper, Eq. 4–5)

`Λ₂₄ = (1/√8)·J`, with `J = J_even ∪ J_odd ⊂ Z²⁴`:

| | J_even | J_odd |
|---|---|---|
| (i) parity | `xᵢ ≡ 0 (mod 2)` | `xᵢ ≡ 1 (mod 2)` |
| (ii) Golay | `(x/2) mod 2 ∈ G₂₄` | `((x−1)/2) mod 2 ∈ G₂₄` |
| (iii) sum | `Σxᵢ ≡ 0 (mod 8)` | `Σxᵢ ≡ 4 (mod 8)` |

`G₂₄` = extended binary Golay code [24,12,8], 4096 words, Hamming weights ∈ {0,8,12,16,24}. With
the 1/√8 normalization, **the lattice is even and unimodular**, hence of covolume 1, which
validates the sizing computation of §3.

> Caveat: the (iii) congruences must be re-checked against Eq. (5) p. 4. The PDF text extraction
> garbles digits set in a mathematical font. The rest is certain.

### 1.3 Shells, classes, leaders (§2.2, §2.4)

`Shell(m) = {v ∈ Λ₂₄ : ‖v‖² = 2m}`, m ≥ 2. Table 1 of the paper:

| m | ‖v‖² | cardinality n(m) | cumulative N(m) | bits/dim |
|---|---|---|---|---|
| 2 | 4 | 196,560 | 196,560 | 0.75 |
| 3 | 6 | 16,773,120 | 16,969,680 | 1.042 |
| 4 | 8 | 398,034,000 | 415,003,680 | 1.208 |
| 5 | 10 | 4,629,381,120 | 5,044,384,800 | 1.375 |
| **13** | **26** | n/a | **280,974,212,784,720** | **2.000** |
| 19 | 38 | n/a | 23,546,209,100,646,960 | 2.292 |

The 2 bits/weight regime is the union of the shells up to m = 13, squared norm 26.

Inside a shell, the points group into **classes**: sets stable under coordinate permutation and
sign change, represented by a **leader** (the canonical multiset of absolute values). Table 2 of
the paper gives the classes of shells 2, 3 and 4. Cardinality of a class:

```
|class| = γ · 2^C · (24! / ∏ρ!) · (1 / ∏|q|!)
```
where γ = the number of admissible Golay words (**4096 for the odd classes**, γ ∈ {1, 759, 2576,
759, 1} for the even ones, depending on the required weight), 2^C the admissible signs, then the
multinomial permutation factors.

### 1.4 The four contributions

1. Hierarchical **bijective indexing** (§3.2): shell → class → local symmetries. The local
   symmetries decompose into (r) Golay refinement, (s) sign pattern, (I″) permutation rank, by
   successive divisions and modulos.
2. **Multi-shell search** (§3.1): Adoul–Barth handles one shell only, where dot-product ranking
   coincides with Euclidean ranking. As soon as several shells are unioned, the norms vary and
   the equivalence breaks. LLVQ adds two metrics, Euclidean (*spherical shaping*) and angular by
   cosine (*shape–gain*).
3. **Fused dequantization kernel** (Appendices A and C).
4. **Spherical GPTQ** (§3.3, Algorithm 1): the standard shape–gain rescaling, interleaved with
   GPTQ's Hessian error backpropagation, reads as a **retraction onto a product of spheres**.
   `ṽ = (‖v‖₂/‖av‖₂)·av`, and the GPTQ residuals are formed on `ṽ`.

### 1.5 The encoder / decoder asymmetry

| | When | Nature |
|---|---|---|
| **Encoder** (nearest neighbor) | offline, 1× per model | Adoul–Barth: leaders, Golay placements, sign patterns, dot-product ranking |
| **Decoder** (index → vector) | **every GEMM** | small static tables, integer div/mod, local combinatorial reconstruction |

The paper is explicit (Appendix A.5): "no dependency between vectors, no bulky memory access,
trivially parallelizable". That is the reason the project is viable.

---

## 2. The engineering opening, what the authors did not do

This is the most important point for deciding to go ahead, and the paper states it in black and
white (Appendix C):

Their CUDA kernel handles a single shell, M = 3, "for simplicity". The multi-shell kernel, the
one needed for the 2 bits/weight regime at m = 13, **does not exist**.

**Their kernel is slower than QTIP, and they say so**: *"we stress that this work does not aim to
make definitive claims about optimized runtime performance, since low-level kernel engineering
could plausibly improve our implementations further. These optimizations are largely orthogonal
to the main contribution."*

Table 7 of the paper, 4096×4096 matvec:

| Kernel | Time |
|---|---|
| FP16 matvec | 16.13 µs |
| FP16 matvec (4096×4104) | 17.169 µs |
| **LLVQ fused (dequant + matvec)** | **11.194 µs, 1.36–1.48× speedup over FP16** |

The authors deliver the best representation in the state of the art, with a single-shell
demonstration kernel they themselves declare unoptimized. The gap between the quality of the
representation and the quality of the implementation is the project.

No code published to date. Only existing repository: `dmnunez1993/llvq-paper-reproduction`
(notebook, 0 stars, dormant since 2026-06-02).

---

## 3. Sizing check

Since Λ₂₄ is unimodular, the number of points in a ball of radius R is `≈ V₂₄·R²⁴` with
`V₂₄ = π¹²/12! ≈ 1.930×10⁻³`. For 2⁴⁸ points: `R ≈ 5.19`, hence `‖v‖² ≈ 26.9`.

Table 1 of the paper gives 2.000 bits/dim at m = 13, where ‖v‖² = 26. The asymptotic estimate
and the exact count agree. This computation stays useful as a consistency test if the shell
implementation drifts.

---

## 4. Rust architecture

```
llvq/
├── llvq-core/       # Golay, Λ₂₄, shells, classes, leaders, indexing. #![no_std], 0 dependencies.
├── llvq-search/     # Adoul–Barth + multi-shell extension, Euclidean and angular metrics.
├── llvq-quant/      # shape–gain, Spherical GPTQ, Hessians.  → faer
├── llvq-kernels/    # fused CUDA kernel (cudarc) + CPU SIMD (pulp) + wgpu.
├── llvq-format/     # serialization, GGUF extension.
├── llvq-engine/     # mistral.rs / candle integration.
├── llvq-cli/        # quantize | eval | bench
└── llvq-bench/      # Gaussian source, perplexity, tok/s, VRAM
```

| Need | Choice | Reason |
|---|---|---|
| Dense algebra, Cholesky of H⁻¹ | **`faer`** | Pure Rust, no Fortran/BLAS dependency. Reproducible build. |
| CUDA GPU | **`cudarc`** + CUDA C kernel (NVRTC) | |
| Portable GPU | **`wgpu`** + WGSL (phase 8) | AMD, Intel Arc, Apple: the sovereignty argument. |
| CPU SIMD | **`pulp`** | AVX-512/AVX2/NEON without nightly. |
| Engine | **`mistral.rs`**, else `candle` | |
| Property tests | **`proptest`** | Bijectivity of the indexing (G3). |
| Micro-benchmark | **`criterion`** | |

The GPU kernel will not be in Rust. No Rust→GPU toolchain reaches the level of hand-written CUDA
when the target is QTIP. CUDA C driven by `cudarc`; everything else in Rust. A "100% Rust"
constraint means `wgpu`/WGSL, giving up tensor cores.

---

## 5. Phases and gates

### Phase 0, transcription · 2 to 3 days
Get hold of Adoul & Barth (1988). That is the real prerequisite, not the LLVQ paper. Transcribe
Table 1 (shells), Table 2 (classes and leaders), Eq. 4–5 (congruences), Algorithm 1.

> **Gate G0.** Adoul & Barth in hand and the search algorithm understood. Otherwise everything
> else is blocked: it is the project's only external prerequisite.

### Phase 1, mathematical core · 1 to 2 weeks
`llvq-core`: Golay `u32`, congruences, membership, enumeration by shell/class/leader.

> **Gate G1.** Public invariants, checkable without the paper.
>
> | Test | Expected |
> |---|---|
> | Golay words | 4096 |
> | Weight distribution | 1 / 759 / 2576 / 759 / 1 |
> | Minimum distance | 8 |
> | Minimum ‖v‖² of Λ₂₄ | 4 |
> | Kissing number | **196,560** |
> | Shell(3), Shell(4) | 16,773,120 · 398,034,000 |
> | Gram determinant | 1 |
> | Additive closure | over 10⁶ draws |
> | **Class cardinalities** | **must reproduce Table 2 of the paper** |

### Phase 2, nearest-neighbor search · 2 to 3 weeks
Single-shell Adoul–Barth, then the multi-shell extension with the two metrics.

> **Gate G2.** (1) On a single shell, exact agreement with exhaustive search over 10⁵ draws. The
> algorithm is *exact*, zero tolerance. (2) Multi-shell: the angular ranking must differ from the
> Euclidean ranking, and both must be correct separately. (3) ≥ 10⁵ blocks/s/core (70 billion
> weights ≈ 3×10⁹ blocks → ~15 min on 32 cores).

### Phase 3, bijective indexing · 1 to 2 weeks
Shell → class → (r, s, I″) hierarchy, linearization and delinearization by div/mod.

> **Gate G3, `proptest`.** Exact round trip in both directions over 10⁷ draws; injectivity
> checked exhaustively on Shell(2) (196,560 points, enumerable); every index in the budget decodes
> to a valid point. **A collision corrupts weights silently.** It is the worst failure mode, and
> it passes every perplexity test.

### Phase 4, validation on a Gaussian source · 3 days
The best gate in the project, and it needs no LLM. Quantize i.i.d. `N(0,1)` samples and compare
against Table 3 of the paper, at 2 bits/dim:

| Method | dim | MSE ↓ | SQNR (bits) ↑ | Retention ↑ |
|---|---|---|---|---|
| Uniform | 1 | 0.1151 | 1.377 | 69% |
| Lloyd–Max | 1 | 0.1121 | 1.537 | 77% |
| E8 (cubic) | 8 | 0.1103 | 1.648 | 82.10% |
| **LLVQ spherical shaping** | 24 | 0.1084 | 1.798 | 89.14% |
| **LLVQ shape–gain** | 24 | **0.1078** | **1.849** | **92.11%** |
| Theoretical limit | n/a | 0.0625 | 2 | 100% |

> **Correction (found in phase 4)**: the MSE and SQNR columns of this transcription are mutually
> inconsistent (−½log₂(0.1084) = 1.603 ≠ 1.798). The PDF text extraction had a shifted font
> encoding and the table digits are partly corrupted. The self-consistent anchor is the
> **retention** column (89.14% / 92.11%). Our implementation measures MSE 0.0775 / retention
> 92.23% (spherical, β tuned) at 1.9999 bits/dim, gate G4 reached. To be re-checked against the
> original PDF in Phase 5.

> **Gate G4.** Retention ≥ 89% in spherical shaping and ≥ 92% in shape–gain. Free analytical
> check: at 2 bits/dim, `MSE* = 2⁻²ᴿ = 0.0625` exactly. If your theoretical limit does not land on
> 0.0625, the measurement protocol is wrong before the quantizer is even involved. Three days to
> validate the whole core, before touching a model.

### Phase 5, Spherical GPTQ and LLM pipeline · 2 to 3 weeks
Algorithm 1 of the paper: blocks of b = 24 input channels, left to right, `H = (1/N)AᵀA`,
Cholesky of `H⁻¹`, rows in parallel, gain reset `ṽ = ‖v‖₂·Q_dir(v/‖v‖₂)`, residual propagated onto
the untreated columns.

Calibration: **6,100 DCLM-edu sequences** (same size as QuIP#). Optional finetuning: only the
input scales shared across rows, ~52M tokens, < 0.001 bpw of overhead.

**Small → large progression** (project decision): first **Qwen3-0.6B** as a smoke test of the
pipeline, then **Qwen3-4B**, the smallest model for which the paper publishes reference numbers
(Table 6), and only then the 7B/8B. Each step exists only to de-risk the next.

> **Gate G5, reproduction, on Qwen3-4B then Llama-2 7B and Llama-3 8B at 2 bpw.** LLVQ must beat
> QuIP#/E8P and QTIP on Wikitext-2 perplexity (context 4096), MMLU and CSR, in the unified
> pipeline of Table 6. PPL gap ≤ 0.05 → validated. > 0.2 or LLVQ does not beat QuIP# →
> **project exit point**.
>
> Add the document-extraction business benchmark here, not at the end, cf.
> [*The Illusion of Equivalency in Quantization*, arXiv:2607.08734](https://arxiv.org/abs/2607.08734).

### Phase 6, fused kernel · 3 to 4 weeks · *the core of the contribution*
Two distinct objectives, in this order:

1. **Reproduce** the authors' single-shell M = 3 kernel: ≥ 1.36× on the FP16 matvec.
2. **Go beyond**, which is where the added value sits:
   - a **multi-shell kernel**, which exists nowhere and which conditions the 2 bpw regime;
   - clearing the QTIP bar, which the authors did not try to reach.

> **Gate G6.** (1) ≥ 1.36× over FP16 in single-shell. (2) Multi-shell m ≤ 13 working and exact
> against the Rust reference decoder. (3) Against QTIP: at parity → objective reached; below →
> ship and document, a better representation at slightly lower throughput stays useful when the
> goal is to make the model *fit*.

### Phase 7, engine integration · 2 weeks
`mistral.rs`, serialization format, CLI.

> **Gate G7, on real hardware.** Peak VRAM, prefill and decode tok/s, perplexity, business
> benchmark. Binary question: does a model that did not fit now fit?

### Phase 8, portability · optional, 2 to 3 weeks
`wgpu`/WGSL and the CPU SIMD path. To arbitrate after G7.

---

## 6. Summary

| Phase | Duration | Gate | If it fails |
|---|---|---|---|
| 0, transcription + Adoul–Barth | 2–3 d | G0 | Blocking |
| 1, Golay + Λ₂₄ | 1–2 wk | G1: 196,560, Table 2 | Bug |
| 2, NN search | 2–3 wk | G2: exactness + throughput | Bug |
| 3, indexing | 1–2 wk | G3: bijectivity | Bug |
| 4, **Gaussian source** | **3 d** | **G4: retention 92.11%** | **Exit** |
| 5, Spherical GPTQ + LLM | 2–3 wk | G5: beats QuIP#/QTIP | **Exit** |
| 6, fused kernel | 3–4 wk | G6: 1.36× FP16, multi-shell | Ship and document |
| 7, integration | 2 wk | G7: end to end | n/a |
| 8, portability | 2–3 wk | n/a | Optional |

**Total: 12–16 weeks.** Two exit points, G4 and G5, both before the kernel investment. G4 in
particular costs three days and validates the whole mathematical core **without an LLM**.

---

## 7. What v1 of this plan got wrong

| v1 | Reality |
|---|---|
| "Conway–Sloane / Vardy–Be'ery style decoder through the hexacode" | **False.** LLVQ extends **Adoul & Barth (1988)**: leaders, Golay placements, sign patterns, dot-product ranking. A different algorithmic family. |
| "The remaining work is the production kernel" | Incomplete. The real gap is the **multi-shell kernel**, which does not exist: the authors stop at M = 3. |
| "Risk: kernel too slow" | Overstated. Their kernel already does 1.36–1.48× FP16. The bar is known, not hypothetical. |
| Vague validation steps | Replaced by **Table 3 (Gaussian source)**: three days, no LLM. |
| 11–15 weeks | 12–16, with a realistic split of Phase 6. |

Still true since v1: HARP and LLVQ are **substitutable, not complementary**. The paper confirms it
directly (Table 5). Spherical GPTQ strongly reduces the dependence on Hadamard rotations, and LLVQ
shape–gain stays competitive with no rotation at all. Improving the rotation upstream of a
24-dimensional quantizer buys little.

---

## 8. Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Adoul & Barth (1988) hard to obtain or to implement | **High** | Gate G0. A 1988 IEEE paper, 15 pages. This is the real critical path. |
| Qualcomm publishes its code | Medium | Four months of silence. And the multi-shell kernel + engine integration keeps its value. |
| Numbers not reproducible | Low | G4 then G5, before the kernel investment. |
| Not beating QTIP on speed | Medium | The authors do not beat it either. A documented fallback is acceptable. |
| Silent indexing collision | Low but **critical** | G3 through `proptest`, never relaxed. |
