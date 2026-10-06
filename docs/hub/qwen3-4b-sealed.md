---
license: apache-2.0
base_model: Qwen/Qwen3-4B
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

# Qwen3-4B at 2.73 bits per parameter, one file

Qwen3-4B in a single 1.42 GB file that opens with no checkpoint and no network,
embedding included. Most weight matrices are stored on the Leech lattice Λ₂₄ with
Tetra: 48 bits for a block of 24 weights, one scale per row. A fused CUDA kernel
decodes and multiplies in one pass, so the weights are never written out in full.

**This file is read by the Rust engine, not by `transformers`.** If you want
`transformers`, take
[Qwen3-4B-LLVQ-Tetra](https://huggingface.co/Pier-Jean/Qwen3-4B-LLVQ-Tetra)
instead: same weights, as safetensors, and it carries this file's digest so you
can check. This one is here because paper 2 publishes its SHA-256 and anyone
replaying our measurements needs the bytes.

It is worse than FP16 by a measured amount: 6.77 MMLU points and 9.63 GSM8K
points on the same questions. The numbers below come from one NVIDIA L40S.

## How good it is

| | this file | FP16 | AWQ w4g128 | IQ2_XXS |
|---|---|---|---|---|
| bits per parameter, whole model | **2.73** | 16.00 | 5.30 | 2.48 |
| weight bytes | **1.38 GB** | 8.04 GB | 2.67 GB | 1.25 GB |
| decode, batch 1, own engine | **113.8 tok/s** (ours) | 83.1 (vLLM) | 200.5 (vLLM) | 312.9 (llama.cpp) |
| MMLU, 5-shot, 14,042 questions | **63.37** | 70.14 | 68.14 | 39.78 |
| GSM8K, zero-shot, 1,319 problems | **82.49** | 92.12 | 89.01 | not scored |
| WikiText-2 perplexity | **12.58** | 12.24 | 13.52 | not scored |

Paired on the same questions, with 95 % intervals, it is 6.77 MMLU points
[6.05, 7.50] and 9.63 GSM8K points [7.69, 11.57] below FP16. Against AWQ, 4.76
[4.02, 5.49] and 6.52 [4.39, 8.65].

How the numbers were taken. Bits per parameter are counted by `rtbits` from the
file's records, at the widths the GPU holds. Weight bytes are this file's buffers
on the GPU and the other formats' weight files. Speeds are medians of five
rounds, 256 greedy tokens for this file and 128 for FP16 and AWQ, and IQ2_XXS is
the mean of five 128-token runs. Each format runs in its own engine, so dividing
one speed by another would say nothing: vLLM runs FP16 faster than our engine
does. AWQ's MMLU is read in our harness on its weights converted to f16.

## How the two scores were taken

GSM8K is scored through the kernel, which is the path you would actually run. The prompt
is zero-shot, in Qwen3's chat template with the thinking block left empty, and
asks for the answer in `\boxed{}`. Decoding is greedy, up to 1,024 tokens. FP16
and AWQ generate in vLLM from the same prompt tokens, and one grader scores all
three. To check that the engine does not move a score, the FP16 checkpoint also
ran through our dense path: 91.51 against 92.12 in vLLM, −0.61 points
[−1.27, +0.06]. This file makes 17.5 % errors on GSM8K where FP16 makes 7.9 %.

MMLU is scored on the dense reconstruction instead: the same weights decoded to
f16 and run through an ordinary forward pass. The answer is read from the logits
of the four answer letters, micro-averaged over the full test split. The 63.37
was scored on the file one step before sealing, with the same 4-bit matrices
rebuilt at load by the same quantizer. This file gives the same answers and the
same logits on 57 of 57 spot-checked questions, and on 50 GSM8K problems the
kernel and the dense reconstruction agree on all 50.

At 4B the model loses more points on GSM8K than on MMLU. At 8B and 14B the test
cannot tell the two losses apart: those files lose 4.62 and 3.26 GSM8K points to
FP16, against 5.48 and 3.22 on MMLU. Counted in errors rather than points, GSM8K
costs more at every size.

| sibling file | bits per parameter | MMLU | GSM8K | decode tok/s | weights |
|---|---|---|---|---|---|
| `qwen3-8b-sealed-B.bin` | 2.70 | 69.58 | 88.63 | 95.0 | 2.76 GB |
| `qwen3-14b-sealed.bin` | 2.73 | 75.66 | 92.04 | 57.2 | 5.04 GB |

Both are hosted, with the SHA-256 the paper publishes:
[Qwen3-8B-LLVQ-Tetra-sealed](https://huggingface.co/Pier-Jean/Qwen3-8B-LLVQ-Tetra-sealed)
(`7bdb9a55`) and
[Qwen3-14B-LLVQ-Tetra-sealed](https://huggingface.co/Pier-Jean/Qwen3-14B-LLVQ-Tetra-sealed)
(`61db37fe`).

## What is in the file

| part | how it is stored |
|---|---|
| 168 of the 252 projections | Tetra lattice codes: 48 bits per block of 24 weights, one scale per row, a small tail kept unquantized |
| `v_proj` of every layer, `o_proj` of every layer, `down_proj` of layers 12 to 23 | int4, groups of 128 |
| embedding, tied to the output head | int4, groups of 64 |
| norms and everything the quantizer does not touch | f16 |
| `config.json`, tokenizer | copied byte for byte from the checkpoint |

The codes were fitted with GPTQ-style corrections on 131,072 tokens of
DCLM-edu. The scale of each weight row was then retrained against the FP16
model, with the codes frozen. Each step is measured on the full MMLU test set:

| step | MMLU |
|---|---|
| Tetra codes, int4 `v_proj` | 57.95 |
| + retrained row scales | 61.11 |
| + int4 `o_proj` and `down_proj` 12 to 23, 4-bit embedding (these weights, int4 rebuilt at load) | 63.37 |

## Running it

```bash
git clone https://github.com/pjmalandrino/llvq && cd llvq
hf download Pier-Jean/Qwen3-4B-LLVQ-Tetra-sealed qwen3-4b-sealed.bin --local-dir .
# NVIDIA GPU, through the served kernel
LLVQ_CONFIG=configs/qwen3-4b-tetra-e4.json \
  cargo run --release -p llvq-llm --features cuda --bin chat -- qwen3-4b-sealed.bin cuda
# Apple silicon, dense reconstruction. The Rust Metal fused path still refuses
# this file, because tv_q4_metal stops at d_in 8192 and down_proj is 9728. The
# Python reader does not have that limit: see the safetensors repository.
cargo run --release -p llvq-llm --features metal --bin chat -- qwen3-4b-sealed.bin metal
# GSM8K and MMLU, as measured above
LLVQ_CONFIG=configs/qwen3-4b-tetra-e4.json LLVQ_GSM8K_DUMP=gsm8k.jsonl \
  cargo run --release -p llvq-llm --features cuda --bin gsm8k -- qwen3-4b-sealed.bin cuda
cargo run --release -p llvq-llm --features cuda --bin mmlu -- qwen3-4b-sealed.bin cuda
```

`qwen3-4b-tetra-e4.json`, in this repository, is the configuration the numbers
above were taken under: the Tetra layout, the 4-bit embedding, one rotation per
group of projections, an f16 KV cache. Running without it gives you a bench
rather than the served path.

## Limitations

- **Two GPUs.** The numbers above come from one NVIDIA L40S. On one NVIDIA A100,
  this file decodes 74.2 tok/s, below FP16 in vLLM (145.6), and with the
  embedding in f16 it is slower than our own dense f16 path. No consumer GPU was
  tested.
- **Choices made on MMLU test questions.** Which matrices went to int4 was chosen
  on MMLU, so the MMLU score carries a selection bias that was not measured.
  GSM8K and perplexity chose nothing.
- **Batch 1, short context.** Nothing here measures several requests at once,
  or prompts longer than about 1,400 tokens.
- **One calibration draw.** At 4B, three draws of calibration text, encoded
  with the earlier codebook, spread MMLU over 5.83 points on 2,280 questions
  (standard deviation 2.92). The gaps to FP16 and AWQ move with the draw like
  the absolute scores, and their intervals leave out this spread.
- **GSM8K is an easy test for this model family.** FP16 scores 92 to 95 %, and
  the problems have been public since 2021. Qwen3's thinking mode, which writes
  much longer chains, is not tested.
- **Two readers, and only one writer.** This repository's Rust code reads and
  writes the format with no external dependency; `llvq-tetra` reads it in Python
  for `transformers`. Nothing writes it outside Rust, and `save_pretrained`
  does not round-trip.
- **Not bit-reproducible across backends.** The calibration accumulates in f32
  on the accelerator, so encoding again elsewhere gives other codes.

## The same weights for `transformers`

[Pier-Jean/Qwen3-4B-LLVQ-Tetra](https://huggingface.co/Pier-Jean/Qwen3-4B-LLVQ-Tetra)
holds the same weights as safetensors that stay compressed, with a
`quantization_config` block describing every record. Its `llvq-digest.json`
records the artifact digest `886391a8c03f66dc` of THIS file, and a bit-for-bit
check rebuilds its 1,602 fields against the Rust decoder (*measured*,
`docs/mesures/hf-safetensors-4b-2026-09-28.txt`). Loading it gives the same 256
greedy tokens this file gives, on a CPU, on Metal and on an NVIDIA L4.

## Citation

- Paper 2, *Tetra: Serving Leech-Lattice Quantized LLMs at 2.7 Bits per
  Parameter*, in the repository under `paper2/`.
- Paper 1, the earlier layout: DOI
  [10.5281/zenodo.22133606](https://doi.org/10.5281/zenodo.22133606).
- The method: van der Ouderaa et al.,
  [arXiv:2603.11021](https://arxiv.org/abs/2603.11021).

Measurement logs, preregistrations and the job registry behind every number are
in the repository: `docs/mesures/gsm8k-wave1-2026-09-26.txt`,
`docs/mesures/gsm8k-wave2-2026-09-26.txt`, `docs/mesures/paper-table-2026-09-25.txt`,
`docs/mesures/embed-q4-swap-2026-09-23.txt`.

## License and attribution

Apache 2.0, inherited from [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B).
The `LICENSE` file is Qwen's, carried over unchanged.

**Modification made to the original work:** the weights of the 252 linear
projections of every transformer block are replaced by Leech-lattice codes
(Tetra) or by 4-bit integers, the scale of each row is retrained, and the tied
embedding is stored in 4 bits. All other tensors are the originals, in f16.
Only the row scales are trained, never the codes. No architectural change.

The quantization code is at
[github.com/pjmalandrino/llvq](https://github.com/pjmalandrino/llvq)
(MIT OR Apache-2.0).
