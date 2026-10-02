<!--
  STATUS, 2026-10-02.

  THIS FILE IS THE CARD OF Pier-Jean/Qwen3-4B-LLVQ-Tetra, byte for byte below
  this comment. Edit here, then re-upload. Its sibling is
  docs/hf-model-card.md, the card of -Tetra-sealed. Both lived outside the
  repository for an hour on 2026-10-02 and the rename to llvq-tetra broke the
  install line of this one within that hour, which is why neither lives outside
  any more.
-->

---
license: apache-2.0
base_model: Qwen/Qwen3-4B
base_model_relation: quantized
language:
  - en
library_name: transformers
pipeline_tag: text-generation
tags:
  - llvq
  - tetra
  - qwen3
  - quantization
  - 2-bit
  - leech-lattice
  - vector-quantization
---

# Qwen3-4B, LLVQ Tetra, 2.73 bits per parameter, loadable in `transformers`

Qwen3-4B stored at **2.73 bits per parameter over the whole model**, embedding
included, in **1.41 GB of safetensors that stay compressed in memory**. Most
weight matrices are coded on the Leech lattice Λ₂₄ with **Tetra**, a codebook a
GPU reads as stored: a fused kernel decodes and multiplies in one pass. It
follows the method of
[arXiv:2603.11021](https://arxiv.org/abs/2603.11021) (van der Ouderaa, van
Baalen, Whatmough, Nagel, 2026), in an independent Rust implementation.

This repository is the form `transformers` reads. The same object as a single
file for the Rust engine is at
[Pier-Jean/Qwen3-4B-LLVQ-Tetra-sealed](https://huggingface.co/Pier-Jean/Qwen3-4B-LLVQ-Tetra-sealed),
and both carry the artifact digest `886391a8c03f66dc`, so they are verifiably
the same weights.

> **A research artifact, not a drop-in model.** It needs the `llvq-tetra` reader
> below, which is not on PyPI yet. It loses **6.77 MMLU points and 9.63 GSM8K
> points** to FP16 on the same questions. `save_pretrained` does not round-trip.
> Qwen3 is the only architecture tried.

## Loading it

```bash
pip install git+https://github.com/pjmalandrino/llvq.git#subdirectory=llvq-tetra
```

```python
import llvq_tetra  # registers the method; without this import transformers skips it
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "Pier-Jean/Qwen3-4B-LLVQ-Tetra"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype="float32")

ids = tok("The capital of France is", return_tensors="pt")
print(tok.decode(model.generate(**ids, max_new_tokens=16, do_sample=False)[0]))
```

**The `import llvq_tetra` is load-bearing.** Without it `transformers` only warns,
"Unknown quantization type, got llvq ... we will skip the quantization", then
loads the file as if it were dense and fails on keys it cannot find.

Two arms. Unset, every record is **dequantized into a dense weight at load**:
8.05 GB in f16, no kernel, no compiler, and it runs on a CPU.

```bash
LLVQ_HF_FUSED=1   # the weights stay compressed and a fused kernel runs the matvec
```

With `LLVQ_HF_FUSED=1` on Apple silicon, all 252 projections stay compressed and
the loaded model holds **2.750 GB** on the device against 16.1 GB computed for
the dense f32 arm (*measured*). On CUDA, 168 of the 252 are fused and the 84 int4
records fall back to dense, because the CUDA binding carries the Tetra matvec and
not the int4 one yet; `llvq_tetra` prints the count rather than letting you assume.
The kernels compile at import, so that arm needs `ninja` and a compiler. The
dense arm needs neither.

## Numbers

| | this file | FP16 | AWQ w4g128 | IQ2_XXS |
|---|---|---|---|---|
| bits per parameter, whole model | **2.73** | 16.00 | 5.30 | 2.48 |
| weight bytes | **1.38 GB** | 8.04 GB | 2.67 GB | 1.25 GB |
| decode, batch 1, own engine | **113.8 tok/s** (ours) | 83.1 (vLLM) | 200.5 (vLLM) | 312.9 (llama.cpp) |
| MMLU, 5-shot, 14,042 questions | **63.37** | 70.14 | 68.14 | 39.78 |
| GSM8K, zero-shot, 1,319 problems | **82.49** | 92.12 | 89.01 | not scored |

All *measured*, and **none of them through this loader**. They are the Rust
engine's, on one NVIDIA L40S, and they are what the sealed file scores. What is
measured through this loader is token identity: 256 greedy tokens over four
prompts, identical to the engine's, on the CPU, on Metal with every projection
compressed, and on an NVIDIA L4. A quality table measured here does not exist
yet, and the limitation is listed below rather than hidden.

Paired on the same questions, 95 % intervals:

| | below FP16 | below AWQ |
|---|---|---|
| MMLU | 6.77 [6.05, 7.50] | 4.76 [4.02, 5.49] |
| GSM8K | 9.63 [7.69, 11.57] | 6.52 [4.39, 8.65] |

Speeds from two engines are never divided by one another: vLLM runs FP16 faster
than our engine does. The MMLU of AWQ is read in our harness on its weights
converted to f16.

## What is in the file

| part | how it is stored |
|---|---|
| 168 of the 252 projections | Tetra lattice codes: 48 bits per block of 24 weights, one scale per row, a small tail kept unquantized |
| `v_proj` and `o_proj` of every layer, `down_proj` of layers 12 to 23 | int4, groups of 128 |
| embedding, tied to the output head | int4, groups of 64 |
| norms and everything the quantizer does not touch | f16 |

`config.json` carries a `quantization_config` block with one descriptor per
record: kind, shape, block count, tail width, which rotation table it uses.
`llvq-digest.json` holds a SHA-256 per field of the sealed file, and
`llvq-dense-digest.json` one per reconstructed matrix, so a reader can check
itself against the Rust decoder field by field rather than trusting it.

The codes were fitted with GPTQ-style corrections on 131,072 tokens of
DCLM-edu. The scale of each weight row was then retrained against the FP16
model, with the codes frozen.

`tokenizer_config.json`, `vocab.json`, `merges.txt` and `generation_config.json`
are copied verbatim from
[Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) at revision
`1cfa9a7208912126459214e8b04321603b3df60c`. The packer does not carry them, so
without this copy the tokenizer would load with no chat template.

## Limitations

- **No quality number measured through this loader.** The MMLU and GSM8K above
  are the Rust engine's. This path is held to token identity against it, and
  that gate is known to be weak: a defect worth 8.79 % of a matrix row left 64
  greedy tokens untouched on two prompts of four.
- **One architecture.** `Qwen3ForCausalLM`. The code routes by record name and
  is not coupled to Qwen3, but nothing else has been tried.
- **`save_pretrained` does not round-trip.** `is_serializable()` returns false:
  saving would write dense weights, not this format. Writing the format needs
  the encoder, which is Rust.
- **The fused arms compile at import.** No precompiled kernel is published yet,
  so `LLVQ_HF_FUSED=1` needs `ninja` and Xcode command line tools, or `nvcc`.
- **int4 is not fused on CUDA.** 168 of 252 projections there, against 252 on
  Metal.
- **No batching in the fused arms.** The matvec takes one activation vector, so
  a prefill of T tokens is T dispatches per projection.
- **One GPU for every published speed.** One NVIDIA L40S. On an A100, none of
  our earlier lattice kernels beat FP16.
- **One calibration draw.** At 4B, three draws of calibration text spread MMLU
  over 5.83 points on 2,280 questions (standard deviation 2.92). The intervals
  above leave out that spread.
- **GSM8K is an easy test for this model family.** FP16 scores 92 to 95 %, and
  the problems have been public since 2021.
- **Not bit-reproducible across backends.** Calibration accumulates in f32 on
  the accelerator, so encoding again elsewhere gives other codes.

## Citation

- Paper 2, *Tetra: Serving Leech-Lattice Quantized LLMs at 2.7 Bits per
  Parameter*, in the repository under `paper2/`.
- Paper 1, the earlier layout: DOI
  [10.5281/zenodo.22133606](https://doi.org/10.5281/zenodo.22133606).
- The method: van der Ouderaa et al.,
  [arXiv:2603.11021](https://arxiv.org/abs/2603.11021).

Measurement logs, preregistrations and the job registry behind every number are
in [github.com/pjmalandrino/llvq](https://github.com/pjmalandrino/llvq):
`docs/mesures/gsm8k-wave1-2026-09-26.txt`,
`docs/mesures/gsm8k-wave2-2026-09-26.txt`,
`docs/mesures/paper-table-2026-09-25.txt`,
`docs/mesures/embed-q4-swap-2026-09-23.txt`, and for this loader
`docs/mesures/hf-safetensors-4b-2026-09-28.txt`,
`docs/mesures/hf-quantizer-4b-2026-09-30.txt`,
`docs/mesures/hf-metal-m2-4b-2026-09-30.txt`,
`docs/mesures/hf-cuda-4b-2026-09-30.txt`,
`docs/mesures/hf-cleanroom-4b-2026-10-01.txt`.

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
