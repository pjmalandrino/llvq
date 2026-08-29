<!--
Draft for a Hugging Face Community Article.
Publish at https://huggingface.co/new-blog (paste everything below this
comment block; the title goes in the article's title field).

Suggested title : Two bits per weight, and what it actually costs
Suggested tags  : quantization, llm, cuda, rust, inference
Do NOT mention any journal submission anywhere in the article.

Every number is sourced from the paper (doi:10.5281/zenodo.22133606), the
README, or a measurement journal in docs/mesures/. If a number changes,
check it against its source first.
-->

# Two bits per weight, and what it actually costs

I am not a researcher. I work at a software company in France, and a few weeks
ago I read a paper from Qualcomm AI Research about squeezing LLM weights down
to 2 bits using the Leech lattice, a 24 dimensional sphere packing. It reports
the best quality anyone has published at that rate.

I wanted to know if it survives contact with a real GPU, so I implemented it
from scratch in Rust. This post is what I found, including the parts that did
not go my way.

The short version: Qwen3-4B runs in 2.60 GB of VRAM at 87 tokens per second,
and it gives back the same greedy tokens as the fp16 model. It also loses 14.7
points of MMLU, which is a lot. Both halves are true and I think both are worth
publishing.

Model: [Pier-Jean/Qwen3-4B-LLVQ-2bit](https://huggingface.co/Pier-Jean/Qwen3-4B-LLVQ-2bit)
Code: [pjmalandrino/llvq](https://github.com/pjmalandrino/llvq)
Write up with a DOI: [10.5281/zenodo.22133606](https://doi.org/10.5281/zenodo.22133606)
(self deposited preprint, not peer reviewed)

## Why bother with 2 bits

Bits per weight is the only knob that changes which models fit on hardware you
own. At 2 bits a 70B model goes from 140 GB down to roughly 18 GB, so it fits
on a 24 GB card. That is the whole motivation.

There was a catch. The CUDA kernel published with the paper only decodes one
shell of the lattice, for simplicity, and the authors say it is slower than
QTIP. The codebook you actually need at 2 bits is a union of shells: 301
classes, and a 47 bit index that names one point out of 1.1e14. I could not
find a decoder for it anywhere, including in the original work. So the open
question was whether that index can be served at all, at matvec speed.

## What I built

The math core is a Rust workspace with no external dependencies at all, so you
can read it end to end: the lattice, exact nearest neighbour search, the
bijective index, and the GPTQ loop.

The CUDA kernel never decodes the combinatorial index inside the matrix vector
product. Instead the codebook is unrolled offline into a VRAM layout of bit
planes with a uniform 14 byte stride, so every lane reads it with the same
instruction sequence. No branch per class, no warp divergence.

I checked correctness before touching any timer. Every output row is compared
against a float64 reference, which is 1,105,920 rows on the 4B. The file format
bijection is proven block by block across all 150,681,600 blocks. And the end
to end test is the one that matters: the quantized model has to emit the same
greedy tokens as the dense one. It does, up to a single tie break at token 89.

## The thing I did not expect: file bits are not VRAM bits

This is the result I care about most, and it is not a speedup.

My format and QTIP both store exactly 2.000 bits per weight in the file. In
VRAM they differ by 2.4x. The reason is simple once you see it: a codebook with
1.1e14 points cannot sit in a lookup table, so it has to be unrolled into a
4.80 bit stream, while QTIP's 16 bit trellis state fits in 2 KB of shared
memory.

I ran both kernels in the same process, on the same shapes. QTIP reads 2.4x
fewer bytes and runs 2.27x faster. Both sit at roughly the same fraction of
their memory bandwidth bound, 61% against 65%, so the time gap is just the
traffic gap. That is a property of codebook size, not of either implementation.

If you came here expecting "my kernel beats everything", this is the opposite.
The deployed trellis kernel wins the decode race, and the mechanism is
understood.

Two other measurements bound the rest. A null kernel that reads no weight bytes
at all is still slower than QTIP, which tells me it measures the launch
geometry I use and not the card's floor. And on an A100, every lattice arm
falls below fp16, so the speedups here are an Ada result rather than a general
one.

## The numbers, end to end

Qwen3-4B, everything on one L40S, one harness, identical token fingerprints on
every quality row. The 4 bit arm is Qwen's own official AWQ checkpoint, not one
I made.

| | fp16 | AWQ 4 bit | LLVQ 2 bit (mine) |
|---|---|---|---|
| Disk | 8.04 GB | 2.67 GB | **1.41 GB** |
| Card memory | 16.0 bits/param | 5.30 bits/param | **5.162 bits/param (2.60 GB)** |
| Throughput | 43.5 tok/s | see note | **87.0 tok/s** |
| WikiText-2 ppl | 12.24 | 13.52 | 16.94 |
| MMLU | 70.3% | 70.0% | **55.6%** |

A note on the throughput column. Comparing across engines mixes up the runtime
with the format, so I left that cell empty on purpose. Within its own stack the
AWQ kernel is 2.41x over vLLM's fp16. Within mine, with the output head held
identical on both arms, kernel and format together give 1.11x at 4B, 1.29x at
8B and 1.41x at 14B. Those two numbers do not divide into each other and I will
not pretend they do.

Memory is the axis that actually flips. At 5.162 bits per param for the whole
model, embedding included, I am below the deployed AWQ checkpoint at 5.30, and
the same holds at 8B and 14B.

## The bill

Two bits cost 38% more perplexity and 14.7 points of MMLU at 4B. Four bit
quantization costs nothing measurable, 0.3 points, inside its own error bar.

The damage is not spread evenly. Abstract algebra and accounting fall to 25%,
which is exactly chance, while history, law and psychology stay above 80%. Two
bits hurt reasoning far more than recall, and recall is what a perplexity
benchmark mostly measures. So please do not accept perplexity alone as proof
that a 2 bit model is fine. That includes mine.

The gap does shrink with model size. The MMLU gap to 4 bit goes 14.5 points at
4B, 7.5 at 8B, 6.1 at 14B, each with a paired bootstrap confidence interval.
But three points do not make a scaling law and I am not claiming one.

## How I worked, since that is half the point

Every performance claim was written down and timestamped before the run, so a
red result kills a branch instead of getting argued with afterwards. Several
died exactly that way and they are all in the repo with their logs: a 3.59 bit
layout rejected against a threshold set in advance, a compute bound decoder
that came back at 0.25x fp16, an in kernel index decoder buried on paper before
it cost a dollar.

Every number carries a label saying whether it was measured, computed or
estimated, and in which accounting. The raw per window data behind every
confidence interval is committed.

The entire GPU campaign, every benchmark and quantization run and quality
measurement across three model sizes, came to under 100 dollars of rented time.

## If you want to try it

Start from [LAUNCH_ME.md](https://github.com/pjmalandrino/llvq/blob/main/LAUNCH_ME.md).
Please read these first.

It is not GGUF, AWQ or safetensors. transformers, llama.cpp, vLLM and TGI
cannot read it. The only reader is the Rust crate in the repo.

The memory win needs the fused CUDA path, so Linux and an NVIDIA card. It was
measured on an L40S. On an A100 the lattice arms lose to fp16.

There is a portable runner, but it decodes every weight into memory, so on that
path you only save disk space.

All timings are batch 1 decode. No batching numbers, no prefill claims. Long
prompts are limited right now because the fused path is a matvec and I have not
written the batched prefill yet.

## Citing

> Malandrino, P.-J. (2026). *Unfolding the Leech Lattice: Fused Multi-Shell
> Decoding and VRAM Layouts for 2-Bit LLM Weights.* Zenodo.
> [doi:10.5281/zenodo.22133606](https://doi.org/10.5281/zenodo.22133606)

The paper is CC BY 4.0 and the code is MIT/Apache-2.0. The method comes from
[arXiv:2603.11021](https://arxiv.org/abs/2603.11021) by van der Ouderaa, van
Baalen, Whatmough and Nagel at Qualcomm AI Research. The fused multi shell
decoder, the VRAM layouts and all the measurements here are mine.

If you have thoughts on whether this is worth porting somewhere people can
actually use it, I would like to hear them.
