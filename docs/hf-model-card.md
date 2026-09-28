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

<!--
  STATUS, 2026-09-27.

  This card describes `qwen3-4b-sealed.bin`, the paper-2 object: 1,418,224,685 B,
  sha256 886391a8c03f66dc269cc65c3598c6627dbdcd259180aff36604ef10d37371b8.

  That file is not hosted yet. The card online at Pier-Jean/Qwen3-4B-LLVQ-2bit still
  describes the August file `qwen3-4b-llvq.bin` (Planes14, 1.771 GB, sha256
  9db213ef...c84b0), whose card this one replaces in the repository. Uploading the
  sealed file and replacing the online card are two operator decisions
  (docs/ROADMAP.md section 5). Until both are taken, this file and the Hub disagree,
  and the download line below does not work.
-->

# Qwen3-4B, LLVQ Tetra, 2.73 bits per parameter

Qwen3-4B stored at **2.73 bits per parameter over the whole model**, embedding
included, in **one 1.42 GB file** that opens with no checkpoint and no network.
Most weight matrices are coded on the Leech lattice Λ₂₄ with **Tetra**, a
codebook the GPU reads as stored: a fused CUDA kernel decodes and multiplies in
one pass. It follows the method of
[arXiv:2603.11021](https://arxiv.org/abs/2603.11021) (van der Ouderaa, van
Baalen, Whatmough, Nagel, 2026), in an independent Rust implementation.

> **A research artifact, not a drop-in model.** It is not GGUF, AWQ or
> safetensors. It does not load in `transformers`, llama.cpp or vLLM, and needs
> the Rust reader of [github.com/pjmalandrino/llvq](https://github.com/pjmalandrino/llvq).
> It loses **6.77 MMLU points and 9.63 GSM8K points** to FP16 on the same
> questions. Every number below comes from one NVIDIA L40S.

## Numbers

| | this file | FP16 | AWQ w4g128 | IQ2_XXS |
|---|---|---|---|---|
| bits per parameter, whole model | **2.73** | 16.00 | 5.30 | 2.48 |
| weight bytes | **1.38 GB** | 8.04 GB | 2.67 GB | 1.25 GB |
| decode, batch 1, own engine | **113.8 tok/s** (ours) | 83.1 (vLLM) | 200.5 (vLLM) | 312.9 (llama.cpp) |
| MMLU, 5-shot, 14,042 questions | **63.37** | 70.14 | 68.14 | 39.78 |
| GSM8K, zero-shot, 1,319 problems | **82.49** | 92.12 | 89.01 | not scored |

All *measured*. Bits per parameter are counted by `rtbits` from the file's records, at
the widths the GPU holds.
Weight bytes are this file's buffers on the GPU and the other formats' weight
files. Speeds are medians of five rounds, 256 greedy tokens for this file and
128 for FP16 and AWQ; IQ2_XXS is the mean of five 128-token repetitions. Each
format runs in its own engine, and speeds from two engines are never divided
by one another: vLLM runs FP16 faster than our engine does. The MMLU of AWQ is
read in our harness on its weights converted to f16.

Paired on the same questions, 95 % intervals:

| | below FP16 | below AWQ |
|---|---|---|
| MMLU | 6.77 [6.05, 7.50] | 4.76 [4.02, 5.49] |
| GSM8K | 9.63 [7.69, 11.57] | 6.52 [4.39, 8.65] |

## Quality

**GSM8K is scored through the served kernel**, the path a user runs. The prompt
is zero-shot, in Qwen3's chat template with the thinking block left empty, and
asks for the answer in `\boxed{}`. Decoding is greedy, up to 1,024 tokens. FP16
and AWQ generate in vLLM from the same prompt tokens, and one grader scores all
three. To check that the engine does not move a score, the FP16 checkpoint also
ran through our dense path: 91.51 against 92.12 in vLLM, −0.61 points
[−1.27, +0.06]. This file makes 17.5 % errors on GSM8K where FP16 makes 7.9 %.

**MMLU is scored on the dense reconstruction**: the same weights decoded to f16
and run through an ordinary forward pass. The answer is read from the logits of
the four answer letters, micro-averaged over the full test split. The 63.37
was scored on the file one step before sealing, with the same int4 matrices
rebuilt at load by the same quantizer; this file gives the same answers and
logits on 57 of 57 questions. On 50 GSM8K problems, the kernel and the dense
reconstruction give the same 50 answers.

In points, the model loses more on GSM8K than on MMLU at 4B. At 8B and 14B the
test cannot separate the two losses: the sibling files lose 4.62 and 3.26 GSM8K
points to FP16, against 5.48 and 3.22 on MMLU. Counted in errors, GSM8K costs
more at every size.

| sibling file | bits per parameter | MMLU | GSM8K | decode tok/s | weights |
|---|---|---|---|---|---|
| `qwen3-8b-sealed-B.bin` | 2.70 | 69.58 | 88.63 | 95.0 | 2.76 GB |
| `qwen3-14b-sealed.bin` | 2.73 | 75.66 | 92.04 | 57.2 | 5.04 GB |

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
hf download Pier-Jean/Qwen3-4B-LLVQ-2bit qwen3-4b-sealed.bin --local-dir .   # once hosted
# NVIDIA GPU, through the served kernel
LLVQ_CONFIG=configs/qwen3-4b-tetra-e4.json \
  cargo run --release -p llvq-llm --features cuda --bin chat -- qwen3-4b-sealed.bin cuda
# Apple silicon, dense reconstruction (the Metal fused path does not run this file)
cargo run --release -p llvq-llm --features metal --bin chat -- qwen3-4b-sealed.bin metal
# GSM8K and MMLU, as measured above
LLVQ_CONFIG=configs/qwen3-4b-tetra-e4.json LLVQ_GSM8K_DUMP=gsm8k.jsonl \
  cargo run --release -p llvq-llm --features cuda --bin gsm8k -- qwen3-4b-sealed.bin cuda
cargo run --release -p llvq-llm --features cuda --bin mmlu -- qwen3-4b-sealed.bin cuda
```

`configs/qwen3-4b-tetra-e4.json` is the served configuration: the Tetra layout,
the 4-bit embedding, one rotation per group of projections, an f16 KV cache.

## Limitations

- **One GPU.** Every number is on one NVIDIA L40S. On an A100, none of our
  earlier lattice kernels beat FP16.
- **Batch 1, short context.** Nothing here measures several requests at once,
  or prompts longer than about 1,400 tokens.
- **One calibration draw.** At 4B, three draws of calibration text, encoded
  with the earlier codebook, spread MMLU over 5.83 points on 2,280 questions
  (standard deviation 2.92). The gaps to FP16 and AWQ move with the draw like
  the absolute scores, and their intervals leave out this spread.
- **GSM8K is an easy test for this model family.** FP16 scores 92 to 95 %, and
  the problems have been public since 2021. Qwen3's thinking mode, which writes
  much longer chains, is not tested.
- **The format is read only by this repository's Rust code**, which has no
  external dependency. A reader in another language does not exist.
- **Not bit-reproducible across backends.** The calibration accumulates in f32
  on the accelerator, so encoding again elsewhere gives other codes.

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
