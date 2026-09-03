# Inference cost reduction: survey and candidates to implement

**Survey date: July 2026**
**Goal: fit bigger models on local hardware (sovereignty), at constant quality.**

---

## 0. Method caveat

This survey was done from an environment whose network policy blocks `arxiv.org`,
`huggingface.co`, `openreview.net` and `semanticscholar.org` (403 on the egress proxy). Direct
consequence on the confidence level:

| Source | Verified? |
|---|---|
| Existence, name, date and description of the GitHub repositories | **Yes**, GitHub API queried directly |
| Titles, arXiv numbers, submission dates | Likely, from search results, not read on arXiv |
| Performance numbers (× speedup, perplexity, tok/s) | **Not verified**, taken from the abstracts, to be re-checked by reading the PDF |

**First move before any code: re-read the PDFs from an unfiltered machine** and confirm the numbers
in the section 3 table. The arXiv numbers given lead straight there.

---

## 1. Where the money actually is

On local single-user or small-batch inference (the "sovereignty" case), the bottleneck is almost
never compute: it is **memory and memory bandwidth**. Breakdown of the VRAM budget:

```
VRAM = model weights    +  KV cache  +  activations
        └─ dominant in    └─ dominant in    └─ negligible
           batch 1,          long context     outside prefill
           short context     (>32k tokens)
```

Hence four levers, in decreasing order of return for the goal "a bigger model on the same machine":

1. **Weight quantization**, the only lever that changes the class of model you can load. Going
   from 4 bits to reliable 3 bits cuts ~25% of the VRAM, so a 70B fits where only a 50B fit before.
2. **KV cache compression and sparsification**, decisive as soon as you target long context
   (document RAG, corpus analysis). This is the lever that is exploding in 2026.
3. **Smart MoE offload (CPU/GPU/NVMe)**, which lets you run models whose weights do *not fit at
   all* in VRAM. The most spectacular lever, and the most sensitive to systems engineering.
4. **Speculative decoding**, which improves latency, not memory capacity. Out of the main scope
   here, and the field is already well tooled (EAGLE-3, TorchSpec).

---

## 2. The trap to avoid: TurboQuant

TurboQuant (Google, ICLR 2026: 3-bit KV cache quantization, *data-oblivious*, no calibration) is
**the most publicized paper of the year** on the subject, and that is exactly why it is a bad
implementation target.

State observed on GitHub (July 2026):

| Repository | Stars | Created |
|---|---|---|
| `TheTom/turboquant_plus` | ~7,000 | 25 March 2026 |
| `0xSero/turboquant` (Triton kernels + vLLM) | ~1,700 | 25 March 2026 |
| `scrya-com/rotorquant` (claims to beat it) | ~1,040 | 26 March 2026 |
| `tonbistudio/turboquant-pytorch` | ~1,034 | 25 March 2026 |
| `mitkox/vllm-turboquant` | ~610 | 25 March 2026 |
| `AmesianX/TurboQuant` (llama.cpp port) | ~92 | 29 March 2026 |

The paper came out and **five serious reimplementations existed within a week**, plus an open
integration discussion on `ggml-org/llama.cpp` (#20969). There is no engineering value left to
create there: the slot is taken, and a competitor (`rotorquant`) already claims to do better.

**Transferable lesson**: the "freshly published paper" signal is not enough. The right filter is
*freshness × no implementation × real lever on VRAM*. The rest of this document applies that
filter.

---

## 3. Shortlist

Ranked by the ratio (value for sovereignty) / (engineering effort), taking into account the space
left open.

### 3.1 HARP: learned rotations replacing the fixed Hadamard

- **arXiv**: 2605.29843 (May 2026)
- **Title**: *HARP: Hadamard-Preconditioned Adaptive Rotation Processor for Extreme LLM Quantization*
- **Code**: `brain-lab-research/HARP`, **0 stars**, created 27 May 2026, last commit
  2 July 2026. Research code, not a product.

**The idea.** Every modern PTQ method (QuIP#, QTIP, QuaRot, SpinQuant) rests on *incoherence
processing*: the weights are multiplied by a randomized Hadamard transform (RHT) to spread the
outliers before quantization. That rotation is **fixed and blind**, the same for every layer,
every model, every quantizer. HARP replaces it with a rotation **learned on the calibration
data**, parameterized as a product of block-orthogonal "butterfly" stages (FFT structure), so it
is cheap to apply. It initializes on the RHT up to a permutation, so the worst case is matching
current performance.

**Why it is worth it.** This is lever no. 1 (weights), over the 2–4 bit range, on models from 1B
to 70B. The announced numbers (128 tok/s against 61 tok/s in FP16, a gain in perplexity and in
zero-shot accuracy against fixed RHT) are *to be verified*, but the logic is solid: a component
that was arbitrary until now becomes adaptive.

**The engineering work.** The paper gives the algorithm; everything downstream is missing.
1. Reproduce the rotation fit on a public model (Qwen3, Llama, Mistral) and check that it really
   beats RHT at equal bit budget.
2. Write the butterfly rotation kernel (Triton or CUDA). That is where the real value sits,
   because a badly implemented learned rotation cancels its own gain.
3. Serialize to the GGUF format and expose it through vLLM, otherwise nobody uses it.
4. Handle non-power-of-2 dimensions (mixed-radix schedules), a point the paper treats explicitly
   and that naive implementations miss.

**Two caveats, from cross-checking against §3.3.**

*On the baseline.* The headline number (128 tok/s against 61 tok/s in FP16) compares against FP16.
That is not the comparison that decides: fixed RHT is also much faster than FP16. The only
question that counts is **HARP against fixed RHT at equal bit budget**, in quality *and* in
throughput. A learned rotation costs more at run time than a Hadamard (parameters to load, a
transform that is harder to fuse). Check that first in the PDF.

*On the ceiling.* The LLVQ authors (§3.3) observe that high-dimensional vector quantization
**reduces the dependence on rotational preconditioning**: their rotation-free variant already
beats E8P with rotation. If that holds, HARP and LLVQ are partly **substitutable and not
complementary**: improving the rotation pays less the better the downstream quantizer is. HARP
keeps all its value ahead of a scalar or low-dimensional quantizer; ahead of Leech, the marginal
gain is uncertain.

**Effort**: 3–6 weeks. **Risk**: medium, the gains can melt away after real quantization, and the
ceiling above is real. **Open space**: yes, wide.

---

### 3.2 CoX-MoE: CPU (AMX) / GPU co-execution for MoE

- **arXiv**: 2605.17889 (May 2026)
- **Title**: *CoX-MoE: Coalesced Expert Execution for High-Throughput MoE Inference with AMX-Enabled CPU-GPU Co-Execution*
- **Code**: **no GitHub repository.** Search by exact name: 0 results.

**The idea.** A MoE model activates only a fraction of its experts per token, but all the weights
have to be somewhere. The classic approach keeps the hot experts in VRAM and the cold ones in RAM,
with a PCIe transfer on every miss, and the transfer dominates. CoX-MoE proposes to **compute the
cold experts directly on the CPU**, using the Intel AMX instructions (native matrix multiply on
Xeon Sapphire Rapids and later), and to *coalesce* expert executions to amortize the fixed costs.

**Why it is the best "sovereignty" candidate.** The typical French on-premise fleet is not an H100
cluster: it is a recent dual-Xeon with a lot of RAM and one or two mid-range cards. AMX is present
and massively under-used on that hardware. A runtime that can run the cold experts on the CPU
instead of pulling them back turns a general-purpose server into a MoE machine. That is exactly
the gap between "we cannot host this model" and "we host it".

**The engineering work.** Everything still has to be built, and it is systems work, not ML:
1. Expert GEMM kernel on AMX (`_tile_*` intrinsics, or through oneDNN) in INT8/BF16.
2. Hot/cold placement policy and coalescing of the expert batch.
3. Asynchronous GPU ↔ CPU pipeline to overlap compute and transfer.
4. Honest benchmark against what exists: `llama.cpp` with `--n-cpu-moe`, ktransformers, and the
   recent `JustVugg/colibri` (~20,000 stars, created 1 July 2026, which runs a 744B MoE on 25 GB
   of RAM by streaming the experts from disk).

**Effort**: 6–10 weeks, low-level skills required. **Risk**: low on feasibility, high on
reproducing the exact numbers. **Open space**: total.

> The immediate neighborhood, if CoX-MoE disappoints on reading: *Efficient CPU-GPU Collaborative
> Inference for MoE-based LLMs on Memory-Limited Systems* (arXiv 2512.16473, ASP-DAC 2026),
> *MoBiLE* (2510.12357, no repository found) and *Dynamic Expert Quantization* (2511.15015), which
> keeps the high-traffic experts in high precision and the rest in a low-precision fallback. Same
> lever, different angles.

---

### 3.3 Leech lattice vector quantization

- **arXiv**: 2603.11021 (March 2026), acronym **LLVQ**
- **Title**: *Leech Lattice Vector Quantization for Efficient LLM Compression*
- **Authors**: van der Ouderaa, van Baalen, Whatmough, Nagel, **Qualcomm AI Research**, the
  reference team on quantization. This is not an isolated paper.
- **Code**: no usable implementation. A single repository exists,
  `dmnunez1993/llvq-paper-reproduction` (Jupyter notebook, 0 stars, created 22 May 2026, last
  commit 2 June), a **dormant** reproduction attempt.

**The idea.** Scalar quantization loses by construction: quantizing each weight independently
ignores the structure of the vector. Lattice vector quantization uses the fact that in dimension
24 the Leech lattice is the proven optimal sphere packing, the theoretically best possible
codebook at that dimension.

**The lock the paper lifts.** Until now lattice VQ ran into a dilemma: either materialize the
codebook (at 2 bits/dim over 24 dims that would be 2⁴⁸ entries, impossible), or go down in
dimension. That is exactly why QuIP# picked **E8 in dimension 8** and not Leech: its E8P codebook
fits in 2¹⁶ entries, reduced to a table of 2⁸ by symmetry, hence in GPU shared memory. LLVQ
extends the search algorithm based on the extended Golay code to get (i) **indexing without
materializing the codebook**, (ii) an angular search over a union of lattice layers, (iii) a
**fully parallelizable dequantization kernel**. The three hard algorithmic pieces are therefore
handled in the paper.

**The announced result.** LLVQ is reported to beat QuIP#, QTIP and PVQ, the real state of the art,
not a convenient baseline. One point deserves attention: the shape–gain variant with spherical
GPTQ is reported to beat E8P **even without rotation**, the authors noting that high-dimensional
VQ *intrinsically reduces the dependence on rotational preconditioning*. See §3.1 for the
strategic consequence.

**The engineering work.** There is no Conway–Sloane to re-derive and no author intent to guess:
everything is specified. The work is downstream.
1. Production fused dequantization kernel (Triton/CUDA), at the level of what `cnygaard/glq` does
   for E8. This one decides: a slow kernel cancels the memory gain.
2. Serialization format and GGUF / vLLM integration.
3. Replay of the protocol against QTIP (`Cornell-RelaxML/qtip`) on *our* hardware.

**Effort**: 4–8 weeks. **Risk**: low on correctness (published and exact algorithm), real on
kernel throughput. **Open space**: yes. Four months after publication, nobody has shipped. The
barrier to entry is technical, therefore protective.

---

### 3.4 Attention Editing: GQA → MLA conversion on already post-trained models

- **arXiv**: 2604.05688 (April 2026)
- **Title**: *Attention Editing: A Versatile Framework for Cross-Architecture Attention Conversion*
- **Code**: not found.

**The idea.** MLA (Multi-head Latent Attention, DeepSeek's attention) compresses the KV cache by
an order of magnitude against GQA. But every open Western model (Llama, Qwen, Mistral) is on GQA.
TransMLA and MHA2MLA showed that conversion is possible *after the fact*, on base models only.
Attention Editing treats the target attention as a learned replaceable module, which extends the
conversion to models **already instruction-tuned or trained for reasoning**, the ones actually
deployed.

**Why it is strategic.** This is the only lever on the list that changes the *architecture* and
not the encoding. The KV cache gain stacks with weight quantization and is not capped by the same
wall. In practice: serving a sovereign instruction-tuned model with a 4× longer context at
constant VRAM.

**The engineering work.** A reproducible conversion pipeline on a target model (Mistral or Qwen
instruct), plus a serious non-regression evaluation. That is where this one is won or lost, because
the risk of damaging alignment or reasoning ability during the edit is real, and that is precisely
what the paper claims to solve.

**Effort**: 4–6 weeks, much of it evaluation. **Risk**: high (silent capability degradation).
**Open space**: yes.

> Neighbor to read at the same time: *GQLA / TransGQLA* (arXiv 2605.15250, May 2026), which aims at
> the GQA-efficiency / MLA-compression trade-off with hardware adaptation.

---

### 3.5 RaBitQCache: sparse attention with an adaptive budget

- **arXiv**: 2606.31519 (30 June 2026, ICML'26), **the freshest on the list**
- **Code**: `Sakuraaa0/RaBitQCache`, the official repository, ~14 stars.

**The idea.** Recent sparse attention methods retrieve the top-k tokens of the KV cache on a
*fixed* budget. RaBitQCache uses a randomized rotational binary quantization (RaBitQ, a technique
that comes from vector databases) to estimate the attention weights in binary-INT4 arithmetic. The
estimator is **unbiased with a proven error bound**, which allows **adaptive top-p** retrieval:
the token budget adjusts to the real sparsity of the attention instead of being guessed.

**Special status.** The official code exists, so this is not "awaiting implementation" in the
strict sense. But 14 stars, no engine integration, and it is ICML paper code. Reimplementation is
not the available work. The available work is the **port to vLLM / SGLang / llama.cpp**, where
there is nothing. And the upstream author has a direct interest in seeing that PR arrive.

**Effort**: 2–4 weeks (the shortest on the list, because it starts from code that runs).
**Risk**: low. **Open space**: yes on integration, no on the algorithm.

> Direct competition in the same slot, to be arbitrated on reading: *UNIQUE* (2605.27740,
> universal sparse top-k at KV page granularity), *Fluxion* (2605.07719, hybrid sparse with
> CPU-GPU parallelism), *HiLS* (2607.02980, July 2026, hierarchical attention extrapolating to
> 64× the training length), *OSCAR* (2605.17757, covariant rotation for 2-bit KV).

---

## 4. Decision table

| # | Paper | arXiv | Lever | Upstream code | Effort | Sovereignty impact |
|---|---|---|---|---|---|---|
| 3.2 | **CoX-MoE** | 2605.17889 | MoE offload | **None** | 6–10 wks | 5/5 |
| 3.3 | **Leech VQ (LLVQ)** | 2603.11021 | Weights | Dormant repro (0 stars) | 4–8 wks | 5/5 |
| 3.1 | **HARP** | 2605.29843 | Weights | Skeleton (0 stars) | 3–6 wks | 3/5 |
| 3.4 | **Attention Editing** | 2604.05688 | Architecture / KV | None | 4–6 wks | 4/5 |
| 3.5 | **RaBitQCache** | 2606.31519 | KV cache | Official (14 stars) | 2–4 wks | 3/5 |
| n/a | ~~TurboQuant~~ | n/a | KV cache | 6+ impls, 12k stars | n/a | **Saturated** |

---

## 5. Proposed sequencing

**Wave 0, one week, before any code.**
Build the test bench. Without it none of the five tracks is evaluable, and a real gain cannot be
told apart from a measurement artifact.
- Target hardware frozen and documented (the real fleet, not a rented H100).
- Metrics: peak VRAM, prefill tok/s, decode tok/s, perplexity **and** a business benchmark. For
  this context, a real document extraction task rather than an MMLU. The classic metrics hide
  regressions: that is exactly the point of *The Illusion of Equivalency in Quantization* (arXiv
  2607.08734, July 2026), which shows perplexity and accuracy staying stable while individual
  answers change a great deal. To read before defining the protocol.
- Frozen baselines: FP16, GGUF Q4_K_M, AWQ/GPTQ 4-bit.

**Wave 1, RaBitQCache (3.5).** The shortest path to a publishable result, on code that already
runs. It shakes down the test bench and establishes the credibility of the approach at minimal
risk.

**Wave 2, chosen by the team's profile:**
- low-level systems profile → **CoX-MoE (3.2)**, the most profitable bet;
- ML/quantization profile → **Leech VQ (3.3)** first, **HARP (3.1)** as fallback.

> 3.1 and 3.3 **do not compose** as well as advertised. LLVQ shows that high-dimensional VQ reduces
> the dependence on rotation, so the two tracks partly overlap. One of them has to be chosen.
> Stacking them and expecting the gains to add does not work.

**Wave 3, Attention Editing (3.4)**, once the evaluation protocol is strong enough to detect a
subtle degradation of capabilities. Doing it earlier means risking a wrong conclusion.

---

## 6. Sources

Papers:
- [HARP, 2605.29843](https://arxiv.org/abs/2605.29843)
- [CoX-MoE, 2605.17889](https://arxiv.org/pdf/2605.17889)
- [Leech Lattice VQ, 2603.11021](https://arxiv.org/pdf/2603.11021)
- [Attention Editing, 2604.05688](https://arxiv.org/pdf/2604.05688)
- [RaBitQCache, 2606.31519](https://arxiv.org/abs/2606.31519)
- [UNIQUE, 2605.27740](https://arxiv.org/abs/2605.27740)
- [Fluxion, 2605.07719](https://arxiv.org/abs/2605.07719)
- [HiLS, 2607.02980](https://arxiv.org/abs/2607.02980)
- [The Illusion of Equivalency in Quantization, 2607.08734](https://arxiv.org/abs/2607.08734)
- [GQLA, 2605.15250](https://arxiv.org/html/2605.15250v1)
- [OSCAR, 2605.17757](https://arxiv.org/pdf/2605.17757)
- [MoBiLE, 2510.12357](https://arxiv.org/html/2510.12357)
- [Dynamic Expert Quantization, 2511.15015](https://arxiv.org/abs/2511.15015)
- [CPU-GPU Collaborative MoE Inference, 2512.16473](https://arxiv.org/abs/2512.16473)
- [Token Sparse Attention, 2602.03216](https://arxiv.org/abs/2602.03216)
- [D2Quant, 2602.02546](https://arxiv.org/html/2602.02546v2)
- [QTIP (outgoing reference), 2406.11235](https://arxiv.org/abs/2406.11235)
- [QuIP# (E8P codebook), 2402.04396](https://arxiv.org/abs/2402.04396)
- [Grouped Lattice Vector Quantizers, 2510.20984](https://arxiv.org/pdf/2510.20984)

Repositories and ecosystem:
- [Sakuraaa0/RaBitQCache](https://github.com/Sakuraaa0/RaBitQCache)
- [brain-lab-research/HARP](https://github.com/brain-lab-research/HARP)
- [Cornell-RelaxML/qtip](https://github.com/Cornell-RelaxML/qtip)
- [JustVugg/colibri](https://github.com/JustVugg/colibri)
- [NVIDIA/kvpress](https://github.com/NVIDIA/kvpress)
- [ikawrakow/ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp)
- [TurboQuant discussion on llama.cpp](https://github.com/ggml-org/llama.cpp/discussions/20969)
