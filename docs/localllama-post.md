<!--
Draft post for r/LocalLLaMA. Plain text, no markdown tables that break
on old.reddit. Post the HF blog link, not the PDF.
Do NOT mention any journal submission.
-->

TITLE:
I spent a month implementing 2-bit Leech lattice quantization in Rust. Qwen3-4B runs in 2.6 GB, and here is what it costs you.

BODY:

Hi all. I am not a researcher, I work at a software company in France. A few
weeks ago I read a paper from Qualcomm AI Research about quantizing LLM weights
onto the Leech lattice, which is a 24 dimensional sphere packing. It gets the
best reported quality at 2 bits per weight. I wanted to know if it actually
works outside a paper, so I implemented it from scratch in Rust.

It works, and I want to show both sides of it because most quantization posts
only show one.

What runs today:

Qwen3-4B, 2 bits per weight, on an L40S.
- 2.60 GB on the card, against 8.04 GB in fp16
- 1.41 GB on disk
- 87 tokens/s, against 43.5 for my own fp16 path
- same greedy tokens as the dense model up to one tie break at token 89

What it costs:

- WikiText-2 perplexity goes from 12.24 to 16.94, so about 38% worse
- MMLU goes from 70.3 to 55.6, so 14.7 points lost
- an official 4-bit AWQ of the same model loses 0.3 points, basically nothing

So at 4B, 4-bit wins on quality and it is not close. The gap does shrink with
size (14.7 points at 4B, 10.6 at 8B, 6.9 at 14B) but three points do not make
a scaling law and I am not claiming one.

Two other things I found that I did not expect:

1. Bits on disk do not predict bits in VRAM. My format and QTIP both store
exactly 2.000 bits per weight in the file. In VRAM they differ by 2.4x, because
a codebook with 10^14 points cannot fit in a lookup table and has to be
unrolled, while QTIP's trellis state fits in 2 KB of shared memory. I ran both
kernels in the same process on the same shapes: QTIP reads 2.4x fewer bytes and
runs 2.27x faster. That is a property of the codebook, not of my code.

2. On an A100 every lattice arm falls below fp16. The speedups above are an
Ada result, not a general one.

Honest warnings before anyone downloads it:

- it is NOT GGUF, AWQ or safetensors. llama.cpp, vLLM, TGI and transformers
  cannot read it. The only reader is the Rust crate in the repo.
- the memory win needs the fused CUDA kernel, so Linux and an NVIDIA card.
  There is a portable runner but it decodes everything into memory, so on that
  path you only save disk.
- prompts are limited right now, the fused path is a matvec and I have not
  written the batched prefill yet. Working on it.
- batch 1 decode only, no batching numbers, no prefill numbers.

Everything is public including the raw per window numbers behind every
confidence interval, the cost of every GPU job (the whole thing came to under
100 dollars of rented time), and the branches I killed along the way. I wrote
down thresholds before each run and timestamped them, so a red result kills a
branch instead of being argued with afterwards.

Model: https://huggingface.co/Pier-Jean/Qwen3-4B-LLVQ-2bit
Code: https://github.com/pjmalandrino/llvq
Write up with a DOI: https://doi.org/10.5281/zenodo.22133606
(self deposited preprint, not peer reviewed)

What I would like from you: is this interesting enough to be worth porting
somewhere people can actually use it? I am looking at a vLLM plugin first
because the CUDA kernel transfers almost as is. llama.cpp would need a CPU
decoder that does not exist yet. If you think one is more useful than the
other, or that 2 bits is just not worth it at these sizes, I would rather hear
it now.

Happy to answer anything about the lattice, the kernel or the measurements.
