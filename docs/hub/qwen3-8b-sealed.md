---
license: apache-2.0
base_model: Qwen/Qwen3-8B
base_model_relation: quantized
language:
  - en
library_name: llvq
inference: false
tags:
  - qwen3
  - quantization
  - 2-bit
  - leech-lattice
  - vector-quantization
  - llvq
  - tetra
---

# Qwen3-8B at 2.70 bits per parameter, one file

Qwen3-8B in a single 2.82 GB file that opens with no checkpoint and no network,
embedding and output head included. Most weight matrices are stored on the Leech
lattice Λ₂₄ with Tetra: 48 bits for a block of 24 weights, and one scale per row.
A fused CUDA kernel decodes and multiplies in one pass.

This file is read by the Rust engine of [LLVQ](https://github.com/pjmalandrino/llvq),
not by `transformers`. It is the 8B file that paper 2 measures, and its SHA-256 is
the one the paper publishes:
`7bdb9a5503518081b1f652a93215250898caae5cc1b6a2f33ee9fb6094565c13`.

## How good it is

| | this file | FP16 | AWQ w4g128 |
|---|---|---|---|
| bits per parameter, whole model | **2.70** | 16.00 | 5.96 |
| weight bytes on the GPU | **2.76 GB** | 16.38 GB | 6.10 GB |
| decode, batch 1, NVIDIA L40S | **95.0 tok/s** (our engine) | 46.3 (vLLM) | 123.2 (vLLM) |
| MMLU, 5-shot, 14,042 questions | **69.58** | 75.05 | 73.79 |
| GSM8K, zero-shot, 1,319 problems | **88.63** | 93.25 | 92.95 |
| WikiText-2 perplexity | **9.71** | 8.99 | 9.42 |

On the same questions, with 95 % intervals, this file is 5.48 MMLU points
[4.83, 6.10] and 4.62 GSM8K points [3.00, 6.25] below FP16, and 4.21 [3.54, 4.86]
and 4.32 [2.83, 5.81] below AWQ.

Each format runs in its own engine, so the speeds are not divided by one another.
GSM8K runs through the fused kernel. MMLU and perplexity run on the same weights
decoded to f16. The protocols are in paper 2, section 6.1.

## What is in the file

| part | how it is stored |
|---|---|
| 199 of the 252 projections | Tetra lattice codes: 48 bits per block of 24 weights, one scale per row, a small tail kept unquantized |
| `v_proj` of every layer, `down_proj` of layers 10 to 26 | int4, groups of 128 |
| embedding and output head | int4, groups of 64 |
| norms | f16 |
| `config.json`, tokenizer | copied byte for byte from the checkpoint |

The codes were fitted with GPTQ-style corrections on 131,072 tokens of DCLM-Edu.
The scale of each row was then retrained against the FP16 model, with the codes
frozen. MMLU after each step, on the full test set:

| step | MMLU |
|---|---|
| Tetra codes, int4 `v_proj` | 64.87 |
| + retrained row scales | 68.16 |
| + int4 `down_proj` 10 to 26, 4-bit embedding and head | 69.58 |

The B in the file name: two 8B files were built at about 2.7 bits per parameter.
This one, with int4 on `down_proj` only, scored 0.77 MMLU points [0.26, 1.28] above
the other and was kept under a rule fixed before measuring.

## Running it

```bash
git clone https://github.com/pjmalandrino/llvq && cd llvq
hf download Pier-Jean/Qwen3-8B-LLVQ-Tetra-sealed qwen3-8b-sealed-B.bin --local-dir .
# NVIDIA GPU, through the fused kernel
LLVQ_CONFIG=configs/qwen3-8b-tetra-e4.json \
  cargo run --release -p llvq-llm --features cuda --bin chat -- qwen3-8b-sealed-B.bin cuda
# GSM8K and MMLU, as measured above
LLVQ_CONFIG=configs/qwen3-8b-tetra-e4.json \
  cargo run --release -p llvq-llm --features cuda --bin gsm8k -- qwen3-8b-sealed-B.bin cuda
cargo run --release -p llvq-llm --features cuda --bin mmlu -- qwen3-8b-sealed-B.bin cuda
```

`qwen3-8b-tetra-e4.json`, in this repository, is the configuration the numbers
were taken under. On Apple silicon only the dense reconstruction runs: the Metal
fused path stops at an input width of 8,192, and `down_proj` is 12,288 wide.

## Limitations

- **Two GPUs.** The numbers above come from one NVIDIA L40S. On one NVIDIA A100,
  this file decodes 57.9 tok/s, below FP16 in vLLM (91.8), and with the
  embedding in f16 it is slower than our own dense f16 path. No consumer GPU was
  tested.
- **Batch 1, short context.** Nothing here measures several requests at once or
  long prompts.
- **One calibration draw.** The scores move with the calibration text, and the
  intervals above leave that spread out.
- **Choices made on MMLU test questions.** Which matrices went to int4 was chosen
  on MMLU, so the MMLU score carries a selection bias that was not measured.
  GSM8K and perplexity chose nothing.
- **One reader.** Only the Rust code of the repository reads and writes this file.

## Citation

- Paper 2, *Tetra: Serving Leech-Lattice Quantized LLMs at 2.7 Bits per
  Parameter*, in the repository under `paper2/`.
- Paper 1, the earlier layout: DOI
  [10.5281/zenodo.22133606](https://doi.org/10.5281/zenodo.22133606).
- The method: van der Ouderaa et al.,
  [arXiv:2603.11021](https://arxiv.org/abs/2603.11021).

The siblings: [Qwen3-4B](https://huggingface.co/Pier-Jean/Qwen3-4B-LLVQ-Tetra-sealed)
and [Qwen3-14B](https://huggingface.co/Pier-Jean/Qwen3-14B-LLVQ-Tetra-sealed).

## License and attribution

Apache 2.0, inherited from [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B).
The `LICENSE` file is Qwen's, unchanged.

**Modification made to the original work:** the weights of the 252 linear
projections of the transformer blocks are replaced by Leech-lattice codes (Tetra)
or by 4-bit integers, the scale of each row is retrained, and the embedding and
output head are stored in 4 bits. All other tensors are the originals, in f16.
Only the row scales are trained, never the codes. No architectural change.

The quantization code is at
[github.com/pjmalandrino/llvq](https://github.com/pjmalandrino/llvq)
(MIT OR Apache-2.0).
