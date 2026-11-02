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

# Qwen3-4B at 2.73 bits per parameter, served as fast as we can serve it

The same 1.42 GB file as
[Qwen3-4B-LLVQ-Tetra-sealed](https://huggingface.co/Pier-Jean/Qwen3-4B-LLVQ-Tetra-sealed),
byte for byte, SHA-256 `886391a8`. Qwen3-4B with most weight matrices on the
Leech lattice Λ₂₄, 48 bits for a block of 24 weights, decoded and multiplied by
one fused CUDA kernel without ever being written out in full.

**This repository is the moving one.** That other repository is what paper 2
publishes the digest of, and its card and its files do not change: anyone
replaying the paper's measurements needs those exact bytes under those exact
numbers. This card is rewritten whenever a measurement replaces one of its
figures, which is why the two exist.

What moves here is not the weights. It is the configuration the kernel reads and
the speed that follows: this repository carries a second served configuration,
`qwen3-4b-tetra-e4-grouped.json`, which decodes the same file **1.10× faster
after a short prompt and 2.33× faster at 8,192 tokens of context**.

## Speed, on one NVIDIA L40S, batch 1

Both configurations, same file, same card, through the served door.

| prompt | `qwen3-4b-tetra-e4.json` | `qwen3-4b-tetra-e4-grouped.json` |
|---|---|---|
| 256 tokens after a 5-token prompt | 121.5 tok/s | **133.3 tok/s** |
| decode after 2,048 tokens of context | 82.7 tok/s | **113.8 tok/s** |
| decode after 4,096 | 61.4 tok/s | **98.0 tok/s** |
| decode after 8,192 | 33.2 tok/s | **77.5 tok/s** |
| weights on the card | 1.38 GB | 1.38 GB |

Medians of five rounds, one discarded, measured 2026-11-02
(`docs/mesures/served-grouped-2026-11-02.txt`). The grouped configuration
returns the **same 256-token text** as the other one from the same prompt.

The difference is how attention reads the KV cache. The shipped path
concatenates the whole history at every step and then copies each of the 8 key
and value heads to the 4 query heads that read it, so at 8,192 positions it
moves about 17 GB of cache per token against 1.38 GB of weights. The grouped
path stacks the query heads of one key head as the rows of one matmul and reads
each cached byte once, on a buffer grown in chunks and written in place.

Time to the first token is unchanged: 35.0 s against 34.8 at a prompt of 8,192.
That prefill, not the decode, is now what a long-context answer mostly waits
for, and it is the next thing we are working on.

## What the grouped configuration is licensed by, and what it is not

**It is not bit for bit.** Past five tokens of context it moves the logits:
cuBLAS rounds a batch of 8 key heads differently from 32 expanded ones, by up to
6.6e-2 on a logit. It is served under a bounded tolerance instead, and here is
the whole of that tolerance:

| control, grouped against shipped | result |
|---|---|
| WikiText-2 perplexity, prefill path, 12 windows of 4,096, paired | +0.0000 %, bit for bit |
| WikiText-2 perplexity, teacher-forced decode, paired | +0.0095 % [−0.0048, +0.0238] |
| band the two intervals had to fit inside | ±0.12 % |
| top-1 token agreement over 49,140 decode steps | 99.788 % |

Measured 2026-11-01 through the served kernel
(`docs/mesures/gqa-ppl-2026-11-01.txt`). One result is reported against us: a
few isolated decode steps move a single logit by up to 13.13, which nothing yet
explains.

**No task score has been run under the grouped configuration.** The MMLU 63.37
and GSM8K 82.49 this file is known for were measured under the shipped one. We
have no reason to expect them to move, and no measurement saying they do not.
If you need the scored object, use the other repository and its configuration.

## How good the file is, either way

| | this file | FP16 | AWQ w4g128 | IQ2_XXS |
|---|---|---|---|---|
| bits per parameter, whole model | **2.73** | 16.00 | 5.30 | 2.48 |
| weight bytes | **1.38 GB** | 8.04 GB | 2.67 GB | 1.25 GB |
| MMLU, 5-shot, 14,042 questions | **63.37** | 70.14 | 68.14 | 39.78 |
| GSM8K, zero-shot, 1,319 problems | **82.49** | 92.12 | 89.01 | not scored |
| WikiText-2 perplexity | **12.58** | 12.24 | 13.52 | not scored |

Paired on the same questions, with 95 % intervals, it is 6.77 MMLU points
[6.05, 7.50] and 9.63 GSM8K points [7.69, 11.57] below FP16. Against AWQ, 4.76
[4.02, 5.49] and 6.52 [4.39, 8.65]. Speeds in other engines are not divided by
ours: each format runs where it runs, and vLLM serves FP16 faster than our
engine does. The full account of how every one of these was taken is on the
[sealed repository's card](https://huggingface.co/Pier-Jean/Qwen3-4B-LLVQ-Tetra-sealed),
which does not move.

## Running it

```bash
git clone https://github.com/pjmalandrino/llvq && cd llvq
hf download Pier-Jean/Qwen3-4B-LLVQ-Tetra-served qwen3-4b-sealed.bin --local-dir .
# NVIDIA GPU, the fast configuration
LLVQ_CONFIG=configs/qwen3-4b-tetra-e4-grouped.json \
  cargo run --release -p llvq-llm --features cuda --bin chat -- qwen3-4b-sealed.bin cuda
# the shipped configuration, which every published score was taken under
LLVQ_CONFIG=configs/qwen3-4b-tetra-e4.json \
  cargo run --release -p llvq-llm --features cuda --bin chat -- qwen3-4b-sealed.bin cuda
```

Both configuration files are in this repository and in the source tree under
`configs/`. They differ in one field, `attention`, and in nothing else. An
absent `attention` field means the shipped form, so a configuration written
before this one existed serves exactly what it always served.

On Apple silicon the grouped form is **refused by name**: it has never been
measured on Metal, and the tolerance above was passed on an NVIDIA card. The
shipped configuration runs there, as a dense reconstruction.

## Limitations

- **One GPU, one size.** The speeds above are one NVIDIA L40S. The grouped form
  has never run on another card, on the 8B, on the 14B, or on Metal. On an A100
  this file already decodes below FP16 in vLLM under the shipped form.
- **No task score under the grouped form**, as said above.
- **Batch 1.** Nothing here measures several requests at once.
- **The prefill.** 35 s to the first token at 8,192 tokens of prompt, which the
  grouped form does not improve.
- **Choices made on MMLU test questions.** Which matrices went to int4 was
  chosen on MMLU, so that score carries an unmeasured selection bias. GSM8K and
  perplexity chose nothing.
- **One calibration draw.** At 4B three draws spread MMLU over 5.83 points
  (standard deviation 2.92), and the gaps to FP16 and AWQ move with the draw.
- **Not bit-reproducible across backends.** Calibration accumulates in f32 on
  the accelerator, so encoding again elsewhere gives other codes.

## The same weights for `transformers`

[Pier-Jean/Qwen3-4B-LLVQ-Tetra](https://huggingface.co/Pier-Jean/Qwen3-4B-LLVQ-Tetra)
holds these weights as safetensors that stay compressed, read by the Python
package `llvq-tetra`. It does not carry the grouped attention: that path is in
the Rust engine only.

## Citation

- Paper 2, *Tetra: Serving Leech-Lattice Quantized LLMs at 2.7 Bits per
  Parameter*, in the repository under `paper2/`. It publishes the digest of
  this file at the
  [sealed repository](https://huggingface.co/Pier-Jean/Qwen3-4B-LLVQ-Tetra-sealed);
  cite that one, not this one.
- Paper 1, the earlier layout: DOI
  [10.5281/zenodo.22133606](https://doi.org/10.5281/zenodo.22133606).
- The method: van der Ouderaa et al.,
  [arXiv:2603.11021](https://arxiv.org/abs/2603.11021).

Measurement logs, preregistrations and the job registry behind every number are
in the repository: `docs/mesures/served-grouped-2026-11-02.txt`,
`docs/mesures/gqa-ppl-2026-11-01.txt`, `docs/mesures/gqa-ab-2026-11-01.txt`,
`docs/mesures/trims-contexts-2026-10-30.txt`.

## License and attribution

Apache 2.0, inherited from [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B).
The `LICENSE` file is Qwen's, carried over unchanged.

**Modification made to the original work:** the weights of the 252 linear
projections of the transformer blocks are replaced by Leech-lattice codes
(Tetra) or by 4-bit integers, the scale of each row is retrained, and the tied
embedding is stored in 4 bits. All other tensors are the originals, in f16.
Only the row scales are trained, never the codes. No architectural change.

The quantization code is at
[github.com/pjmalandrino/llvq](https://github.com/pjmalandrino/llvq)
(MIT OR Apache-2.0).
