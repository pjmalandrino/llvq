<!--
Draft for a Hugging Face Community Article.
Publish at https://huggingface.co/new-blog (paste the markdown below the
metadata block; the title goes in the article's title field).

Suggested title : Two Bits per Weight, With the Bill Attached
Suggested tags  : quantization, llm, cuda, rust, inference
Do NOT mention any journal submission anywhere in the article.

Every number below is sourced from the paper (doi:10.5281/zenodo.22133607),
the README, or a measurement journal in docs/mesures/. If a number is edited,
re-check it against its source — the repo's rule is that every figure carries
its provenance.
-->

# Two Bits per Weight, With the Bill Attached

**TL;DR** — I spent a summer implementing
[Leech lattice vector quantization](https://arxiv.org/abs/2603.11021) for LLM
weights from scratch in Rust, including the fused CUDA decoder that, as far as
I can tell, existed nowhere — the original paper's kernel included. The result:
[Qwen3-4B running at 87 tok/s in 2.60 GB of VRAM](https://huggingface.co/Pier-Jean/Qwen3-4B-LLVQ-2bit),
same greedy tokens as the dense model up to a tie-break. The write-up is a
self-deposited preprint:
[**Unfolding the Leech Lattice: Fused Multi-Shell Decoding and VRAM Layouts for
2-Bit LLM Weights**](https://doi.org/10.5281/zenodo.22133606) — and this post
is the honest version of its story, the costs included, because most
quantization posts stop before that part.

⚠️ *The preprint is self-deposited and has not been peer reviewed. All code and
data are public: [pjmalandrino/llvq](https://github.com/pjmalandrino/llvq).*

## Why two bits at all

Bits per weight is the only lever that changes the *class* of model a machine
can hold. At 2 bits, a 70B goes from 140 GB to roughly 18 GB — it fits on a
24 GB card. If you care about running larger models on hardware you own, this
is the axis that matters, and the best reported quality at 2 bits comes from
quantizing weights in blocks of 24 onto the Leech lattice Λ₂₄
(van der Ouderaa et al., Qualcomm AI Research).

There was one problem with adopting that method: the CUDA kernel published
alongside it decodes a single shell of the lattice, for simplicity, and is
slower than its competitors. The codebook you actually need at 2 bits is a
*union* of shells — 301 equivalence classes, a 47-bit index naming one point
out of 1.1 × 10¹⁴. I could not find a decoder for it anywhere. So the
engineering question was open: **can that index be served at all, at matvec
speed?** That is what the paper builds and measures.

## What was built

- A Rust workspace where the mathematical core — lattice, exact
  nearest-neighbour search, bijective 48-bit indexing, spherical GPTQ — has
  **zero external dependencies** and forbids `unsafe`, so it can be audited
  end to end.
- A fused CUDA kernel that never decodes the combinatorial index inside the
  matrix-vector product. The codebook is unfolded **offline** into a VRAM
  layout (bit-planes, uniform 14-byte stride) that every lane reads with the
  same instruction sequence: no branch per class, no warp divergence.
- Correctness before speed, always: every output row is checked against a
  float64 reference before anything is timed (1,105,920 rows on the 4B), the
  file format's bijection is proven block by block on all 150,681,600 blocks,
  and the end-to-end check is that the fused engine emits the **same greedy
  tokens** as the dense f16 arm — it does, up to one documented tie-break at
  token 89.

## The result nobody advertises: file bits ≠ VRAM bits

The finding I care most about is not a speedup. It is that **the bit rate on
disk does not predict the bit rate in VRAM.** This format and QTIP both store
2.000 bits per weight in the file. In VRAM they differ by 2.4×: a codebook of
10¹⁴ points cannot sit in a lookup table, so it has to be unfolded into a
4.80-bit stream, while QTIP's 16-bit trellis state fits in 2 KiB of shared
memory.

Measured in one process, on the same shapes: **QTIP reads 2.40× fewer bytes
and runs 2.27× faster.** Both kernels sit at roughly the same fraction of
their memory-bandwidth bound (61% vs 65%), so the time gap *is* the traffic
gap — a consequence of codebook size, not of either implementation. If you
came here for "our kernel beats everything", this is the opposite: the
deployed trellis kernel wins the decode race, and the paper says so, with the
mechanism.

Two controls bound everything else. A null kernel that reads no weight bytes
at all is *slower* than QTIP — so it measures the launch geometry used here,
not the card's floor. And on an A100, **every lattice arm falls below FP16**:
the speedups in this post are an L40S/Ada result, not a general law.

## End to end

Qwen3-4B, everything measured on one L40S, one harness, identical token
fingerprints on every quality row. The 4-bit arm is Qwen's own official AWQ
checkpoint, not one I produced:

| | FP16 | AWQ 4-bit (official) | **LLVQ 2-bit + fused kernel** |
|---|---|---|---|
| Cold storage | 8.04 GB | 2.67 GB | **1.41–1.77 GB** |
| Card memory, whole model | 16.0 bits/param | 5.30 bits/param¹ | **5.162 bits/param (2.60 GB)** |
| Throughput | 43.5 tok/s | *not comparable*¹ | **87.0 tok/s** |
| WikiText-2 perplexity | 12.24 | 13.52 (×1.105) | 16.94 (×1.384) |
| MMLU (micro, 2,280 q) | 70.3% | 70.0% (−0.28 pp) | **55.6% (−14.7 pp)** |

¹ *In its own engine (vLLM). Cross-engine throughput comparisons mix up the
runtime and the format, so that cell stays empty on purpose: within its stack
the AWQ kernel is ×2.41 over vLLM's f16; within mine, the honest kernel-only
figure is below.*

Two readings of the speed number, and the second is the one that matters:

- **×2.0 raw** over my own dense arm — but that arm is handicapped: it pays a
  778 MB vocabulary copy per token through a `broadcast_matmul` path
  (reported upstream as
  [huggingface/candle#3871](https://github.com/huggingface/candle/issues/3871)).
- **With the output head held identical in both arms**, kernel and format
  together give **×1.11 at 4B, ×1.29 at 8B, ×1.41 at 14B**. That series is
  the measurement of the decoder and format themselves, and it is the one to
  quote.

Memory is the axis that genuinely flips: 5.162 bits/param whole-model is
**below the deployed AWQ 4-bit checkpoint** (5.30), and the same holds at 8B
and 14B (5.32 vs 5.96, 5.11 vs 5.40).

## The bill

At 4B, two bits cost **+38% perplexity and 14.7 MMLU points**, where 4-bit
quantization costs nothing measurable (−0.28 pp, inside its own error bar).
The damage is not uniform: abstract algebra and accounting fall to 25% —
exactly chance — while history, law and psychology hold above 80%. Two bits
destroy **reasoning** much more than **recall**, which is precisely what a
perplexity benchmark under-measures. Never accept perplexity alone as proof
of quality for a 2-bit model; this includes mine.

The gap shrinks with scale — the MMLU gap to 4-bit is 14.5 pp at 4B, 7.5 at
8B, 6.1 at 14B, each with a paired bootstrap CI — but three points do not
make a scaling law, and the paper claims none. The 32B point that would
actually settle it is not run.

## Method, because it is the actual point

Every performance claim in the paper was **pre-registered**: thresholds
written down and timestamped (OpenTimestamps) *before* the run, so a red
result kills a branch instead of being negotiated with after the fact.
Several were killed exactly that way, and they are all in the repo with their
journals: a 3.59 bits/weight layout rejected against a criterion written
before the measurement, a compute-bound decoder closed at 0.25× FP16, an
in-kernel index decoder buried on paper before costing a dollar. Every number carries its
provenance — *measured*, *computed*, or *estimated*, and in which accounting
— and the raw per-window data behind every confidence interval is committed.

The entire GPU campaign — every benchmark, every quantization run, every
quality measurement across three model sizes — came to **under $100** of
rented time.

## Try it, with the caveats attached

The model is at
[Pier-Jean/Qwen3-4B-LLVQ-2bit](https://huggingface.co/Pier-Jean/Qwen3-4B-LLVQ-2bit);
start from
[`LAUNCH_ME.md`](https://github.com/pjmalandrino/llvq/blob/main/LAUNCH_ME.md).

- It is **not** GGUF, AWQ or safetensors. `transformers`, `llama.cpp`, vLLM
  and TGI cannot read it; the only reader is `llvq-artifact`, in the repo.
- The memory win needs the fused path: `bin/fusedrun`, Linux + CUDA
  (measured on L40S/Ada; on A100 the lattice arms lose to FP16).
- The portable runner (`bin/run`) decodes every weight into memory: on that
  path the win is on disk only.
- All timings are batch-1 decode — no batching, no prefill claims.

## Citing

> Malandrino, P.-J. (2026). *Unfolding the Leech Lattice: Fused Multi-Shell
> Decoding and VRAM Layouts for 2-Bit LLM Weights.* Zenodo.
> [doi:10.5281/zenodo.22133606](https://doi.org/10.5281/zenodo.22133606)

The paper is CC BY 4.0; the code is MIT/Apache-2.0. The method implemented is
from [arXiv:2603.11021](https://arxiv.org/abs/2603.11021) (van der Ouderaa,
van Baalen, Whatmough, Nagel — Qualcomm AI Research); the fused multi-shell
decoder, the VRAM layouts, and all measurements here are this project's.
