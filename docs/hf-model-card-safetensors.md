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

# Qwen3-4B at 2.73 bits per parameter

Qwen3-4B in 1.4 GB instead of 8, embedding included, and it stays compressed in
memory if you want it to.

Most weight matrices are stored on the Leech lattice Λ₂₄ with Tetra: 48 bits for
a block of 24 weights, one scale per row. A kernel can decode and multiply in one
pass, so the weights never have to be written out in full.

This is the form `transformers` reads. The same weights as a single file for the
Rust engine are at
[Qwen3-4B-LLVQ-Tetra-sealed](https://huggingface.co/Pier-Jean/Qwen3-4B-LLVQ-Tetra-sealed),
and both carry the digest `886391a8c03f66dc`, so you can check they match.

## Using it

```bash
pip install llvq-tetra
```

```python
import llvq_tetra  # this import registers the method
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "Pier-Jean/Qwen3-4B-LLVQ-Tetra"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype="float32")

ids = tok("The capital of France is", return_tensors="pt")
print(tok.decode(model.generate(**ids, max_new_tokens=16)[0]))
```

**The `import llvq_tetra` matters, and more than it looks.** Without it
`transformers` prints a warning, loads the file as if it were not quantized,
reinitializes the 254 weights it then finds missing, and runs. It does not raise.
The logits look ordinary and the model is random. If you are benchmarking this
file, check that line before you trust a score.

`trust_remote_code=True` works too: the repository carries a small file that does
the import for you. It is there for tools you do not control, and it is the second
best option, because remote code is a thing you should be reluctant to run.

```python
model = AutoModelForCausalLM.from_pretrained(name, dtype="float32",
                                             trust_remote_code=True)
```

What neither option fixes: a caller that passes nothing and imports nothing still
gets the random model. `transformers` picks the model class from `model_type` and
never looks at the repository's own code, so the repository cannot refuse. The
only real fix is for the method to live in `transformers` itself, which is not
our decision to make.

By default the weights are decoded into dense tensors as the model loads. That
needs no GPU and no compiler, and gives you 8 GB in f16. Set `LLVQ_HF_FUSED=1`
and they stay compressed: on Apple silicon the loaded model holds 2.75 GB with
all 252 projections compressed, on NVIDIA 168 of them, the rest falling back to
dense because the CUDA side has the lattice kernel and not yet the 4-bit one. The
fused path compiles a kernel on first use, so it wants `ninja` and a compiler.

## How good it is

| | this file | FP16 | AWQ w4g128 | IQ2_XXS |
|---|---|---|---|---|
| bits per parameter, whole model | **2.73** | 16.00 | 5.30 | 2.48 |
| weight bytes | **1.38 GB** | 8.04 GB | 2.67 GB | 1.25 GB |
| MMLU, 5-shot, 14,042 questions | **63.37** | 70.14 | 68.14 | 39.78 |
| GSM8K, zero-shot, 1,319 problems | **82.49** | 92.12 | 89.01 | not scored |
| decode, batch 1, own engine | **113.8 tok/s** (ours) | 83.1 (vLLM) | 200.5 (vLLM) | 312.9 (llama.cpp) |

Paired on the same questions, with 95 % intervals, it is 6.77 MMLU points
[6.05, 7.50] and 9.63 GSM8K points [7.69, 11.57] below FP16. Against AWQ, 4.76
[4.02, 5.49] and 6.52 [4.39, 8.65].

Two things to read carefully. **None of those scores was measured through this
loader.** They are the Rust engine's, on one NVIDIA L40S, and they describe the
same weights. What is measured here is that this loader produces the same tokens
as that engine: 256 greedy tokens over four prompts, on CPU, on Metal with every
projection compressed, and on an NVIDIA L4. And the speeds come from four
different engines, so dividing one by another would say nothing: vLLM runs FP16
faster than our engine does.

## What is in the file

| part | stored as |
|---|---|
| 168 of the 252 projections | Tetra lattice codes, 48 bits per block of 24 weights, one scale per row, a short tail left alone |
| `v_proj` and `o_proj` everywhere, `down_proj` of layers 12 to 23 | 4-bit integers, groups of 128 |
| embedding, tied to the output head | 4-bit integers, groups of 64 |
| norms and the rest | f16, untouched |

`config.json` describes every record: kind, shape, block count, tail width, which
rotation table it uses. `llvq-digest.json` holds a SHA-256 per field and
`llvq-dense-digest.json` one per rebuilt matrix, so you can check the decoder
against the Rust one rather than trust it. `python -m llvq_tetra.checkdense <dir>`
does exactly that, 253 records on this model.

The codes were fitted with GPTQ-style corrections on 131,072 tokens of DCLM-edu,
then the scale of each row was retrained against the FP16 model with the codes
frozen.

`tokenizer_config.json`, `vocab.json`, `merges.txt` and `generation_config.json`
are copied as they are from [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B)
at revision `1cfa9a7208912126459214e8b04321603b3df60c`.

## Limitations

- No quality number measured through this loader. It is held to token identity
  against the Rust engine, and that is a weak test: a defect worth 8.79 % of a
  matrix row once left 64 greedy tokens untouched on two prompts out of four.
- Only Qwen3 has been tried.
- `save_pretrained` does not work. Writing the format needs the encoder, which is
  Rust.
- The fused path compiles a kernel on first use. No precompiled kernel is
  published yet.
- On NVIDIA, 168 of 252 projections are fused, against 252 on Metal.
- The fused kernels take one activation vector at a time, so a long prompt costs
  one dispatch per token per projection.
- Every published speed is from one NVIDIA L40S. On an A100, none of our earlier
  lattice kernels beat FP16.
- One calibration draw. At 4B, three draws spread MMLU over 5.83 points on 2,280
  questions, and the intervals above do not include that spread.
- GSM8K is easy for this family. FP16 scores 92 to 95 %, and the problems have
  been public since 2021.
- Encoding is not reproducible across backends: calibration accumulates in f32 on
  the accelerator, so the same text on another machine gives other codes.

## Credit and citation

The method is [arXiv:2603.11021](https://arxiv.org/abs/2603.11021), van der
Ouderaa, van Baalen, Whatmough and Nagel, 2026. This is an independent
implementation in Rust, with [llvq-tetra](https://pypi.org/project/llvq-tetra/)
as its Python reader.

Paper 2, *Tetra: Serving Leech-Lattice Quantized LLMs at 2.7 Bits per Parameter*,
is in the repository under `paper2/`. Paper 1 is at DOI
[10.5281/zenodo.22133606](https://doi.org/10.5281/zenodo.22133606).

The measurement logs and preregistrations behind every number above are in
[github.com/pjmalandrino/llvq](https://github.com/pjmalandrino/llvq), under
`docs/mesures/` and `proofs/`.

## License

Apache 2.0, inherited from [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B).
The `LICENSE` file is Qwen's, unchanged.

What was changed: the weights of the 252 linear projections in every transformer
block are replaced by Leech-lattice codes or by 4-bit integers, the scale of each
row is retrained, and the tied embedding is stored in 4 bits. Everything else is
the original, in f16. Only the row scales are trained, never the codes. No
architectural change.

The quantization code is at
[github.com/pjmalandrino/llvq](https://github.com/pjmalandrino/llvq), MIT or
Apache-2.0.
