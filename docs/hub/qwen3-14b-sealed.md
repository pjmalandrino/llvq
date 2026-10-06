---
license: apache-2.0
base_model: Qwen/Qwen3-14B
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

# Qwen3-14B at 2.73 bits per parameter, one file

Qwen3-14B in a single 5.09 GB file that opens with no checkpoint and no network,
embedding and output head included. Most weight matrices are stored on the Leech
lattice Λ₂₄ with Tetra: 48 bits for a block of 24 weights, and one scale per row.
A fused CUDA kernel decodes and multiplies in one pass.

This file is read by the Rust engine of [LLVQ](https://github.com/pjmalandrino/llvq),
not by `transformers`. It is the 14B file that paper 2 measures, and its SHA-256 is
the one the paper publishes:
`61db37fe7ce8e6a57c5c59f257319c796574981ecb5e450a02c73dd3c17e2ca0`.

## How good it is

| | this file | FP16 | AWQ w4g128 |
|---|---|---|---|
| bits per parameter, whole model | **2.73** | 16.00 | 5.40 |
| weight bytes on the GPU | **5.04 GB** | 29.54 GB | 9.98 GB |
| decode, batch 1, NVIDIA L40S | **57.2 tok/s** (our engine) | 25.8 (vLLM) | 77.8 (vLLM) |
| MMLU, 5-shot, 14,042 questions | **75.66** | 78.88 | 78.12 |
| GSM8K, zero-shot, 1,319 problems | **92.04** | 95.30 | 95.38 |
| WikiText-2 perplexity | **8.44** | 7.98 | 8.29 |

On the same questions, with 95 % intervals, this file is 3.22 MMLU points
[2.69, 3.75] and 3.26 GSM8K points [1.93, 4.59] below FP16, and 2.46 [1.92, 3.00]
and 3.34 [2.12, 4.55] below AWQ.

Each format runs in its own engine, so the speeds are not divided by one another.
GSM8K runs through the fused kernel. MMLU and perplexity run on the same weights
decoded to f16. The protocols are in paper 2, section 6.1.

## What is in the file

| part | how it is stored |
|---|---|
| 181 of the 280 projections | Tetra lattice codes: 48 bits per block of 24 weights, one scale per row, a small tail kept unquantized |
| `v_proj` and `o_proj` of every layer, `down_proj` of layers 10 to 28 | int4, groups of 128 |
| embedding and output head | int4, groups of 64 |
| norms | f16 |
| `config.json`, tokenizer | copied byte for byte from the checkpoint |

The codes were fitted with GPTQ-style corrections on 131,072 tokens of DCLM-Edu.
The scale of each row was then retrained against the FP16 model, with the codes
frozen. MMLU after each step, on the full test set:

| step | MMLU |
|---|---|
| Tetra codes, int4 `v_proj` | 72.53 |
| + retrained row scales | 74.20 |
| + int4 `o_proj` and `down_proj` 10 to 28, 4-bit embedding and head | 75.66 |

## Running it

```bash
git clone https://github.com/pjmalandrino/llvq && cd llvq
hf download Pier-Jean/Qwen3-14B-LLVQ-Tetra-sealed qwen3-14b-sealed.bin --local-dir .
# NVIDIA GPU, through the fused kernel
LLVQ_CONFIG=configs/qwen3-14b-tetra-e4.json \
  cargo run --release -p llvq-llm --features cuda --bin chat -- qwen3-14b-sealed.bin cuda
# GSM8K and MMLU, as measured above
LLVQ_CONFIG=configs/qwen3-14b-tetra-e4.json \
  cargo run --release -p llvq-llm --features cuda --bin gsm8k -- qwen3-14b-sealed.bin cuda
cargo run --release -p llvq-llm --features cuda --bin mmlu -- qwen3-14b-sealed.bin cuda
```

`qwen3-14b-tetra-e4.json`, in this repository, is the configuration the numbers
were taken under. On Apple silicon only the dense reconstruction runs: the Metal
fused path stops at an input width of 8,192, and `down_proj` is 17,408 wide.

## Limitations

- **Two GPUs.** The numbers above come from one NVIDIA L40S. On one NVIDIA A100,
  this file decodes 39.0 tok/s, below FP16 in vLLM (52.7), and with the
  embedding in f16 it is slower than our own dense f16 path. No consumer GPU was
  tested.
- **Greedy tokens differ at token 78.** Generating 256 tokens from one prompt,
  the fused kernel and the dense reconstruction first pick a different token at
  position 78. The cause was not measured.
- **Batch 1, short context.** Nothing here measures several requests at once or
  long prompts.
- **One calibration draw.** The scores move with the calibration text, and the
  intervals above leave that spread out.
- **Choices made on MMLU test questions.** The int4 window was fixed by a rule
  before measuring, but the rule came from choices made on MMLU at 4B, so the
  MMLU score carries a selection bias that was not measured. GSM8K and
  perplexity chose nothing.
- **One reader.** Only the Rust code of the repository reads and writes this file.

## Citation

- Paper 2, *Tetra: Serving Leech-Lattice Quantized LLMs at 2.7 Bits per
  Parameter*, in the repository under `paper2/`.
- Paper 1, the earlier layout: DOI
  [10.5281/zenodo.22133606](https://doi.org/10.5281/zenodo.22133606).
- The method: van der Ouderaa et al.,
  [arXiv:2603.11021](https://arxiv.org/abs/2603.11021).

The siblings: [Qwen3-4B](https://huggingface.co/Pier-Jean/Qwen3-4B-LLVQ-Tetra-sealed)
and [Qwen3-8B](https://huggingface.co/Pier-Jean/Qwen3-8B-LLVQ-Tetra-sealed).

## License and attribution

Apache 2.0, inherited from [Qwen/Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B).
The `LICENSE` file is Qwen's, unchanged.

**Modification made to the original work:** the weights of the 280 linear
projections of the transformer blocks are replaced by Leech-lattice codes (Tetra)
or by 4-bit integers, the scale of each row is retrained, and the embedding and
output head are stored in 4 bits. All other tensors are the originals, in f16.
Only the row scales are trained, never the codes. No architectural change.

The quantization code is at
[github.com/pjmalandrino/llvq](https://github.com/pjmalandrino/llvq)
(MIT OR Apache-2.0).
