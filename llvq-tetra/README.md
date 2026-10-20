# llvq-tetra

Load a Tetra-quantized model in `transformers`, and keep it compressed.

Tetra stores weights on the Leech lattice Λ₂₄: 48 bits for a block of 24 weights,
one scale per row. Qwen3-4B comes to 2.73 bits per parameter over the whole
model, embedding included, so 1.4 GB instead of 8.

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

Don't skip `import llvq_tetra`. Without it, `transformers` prints a warning,
ignores the quantization, fills the 254 weights it can't find with random
values, and runs anyway. It does not raise, and the logits look normal. If you
benchmark a model, check that line before you trust the score.

## Two ways to run it

By default the weights are decoded to ordinary dense tensors while the model
loads. That needs no GPU and no compiler, but the model then takes 8 GB in f16,
and 16 GB in float32, the dtype of the snippet above. The download is still
1.4 GB, which helps if disk or bandwidth is your limit rather than memory.

Set `LLVQ_HF_FUSED=1` and the weights stay compressed in memory. A kernel decodes
and multiplies in one pass. On Apple silicon the loaded 4B holds 2.75 GB on the
device, with all 252 projections compressed. On NVIDIA only 168 of them are
compressed and the rest fall back to dense, because the CUDA side has the
lattice kernel but not yet the 4-bit one. The package prints the count, so you
can see which case you are in.

The fused path compiles a kernel the first time you use it, so it needs `ninja`
and a compiler: Xcode command line tools on a Mac, `nvcc` on Linux. There is no
precompiled kernel yet.

## Before you use it

The model scores below FP16. On Qwen3-4B it loses 6.77 MMLU points and 9.63
GSM8K points on the same questions. Those scores were measured with the Rust
engine, and FP16 on GSM8K with vLLM, not with this package. This package is
only checked against that engine: it gives the same 256 greedy tokens over four
prompts, on CPU, on Metal and on an NVIDIA L4.

`save_pretrained` does not work. Writing the format needs the encoder, which is
in Rust. This package reads models, it cannot write them.

Only Qwen3 has been tested. The code goes by the record names in the file, not
by the architecture, so other models may work, but nobody has tried.

## Checking it yourself

A packed model ships with two digest files, so you can check the decoder instead
of trusting it.

```python
from llvq_tetra import PackedModel

with PackedModel("path/to/model") as m:
    w = m.dequantize("model.layers.0.self_attn.q_proj.weight")
```

`PackedModel` needs only numpy and safetensors. `python -m llvq_tetra.checkdense
<dir>` rebuilds every matrix and compares one SHA-256 per record with the digests
the Rust decoder wrote. On the published 4B that is 253 records, all equal.

## Models

- [Qwen3-4B-LLVQ-Tetra](https://huggingface.co/Pier-Jean/Qwen3-4B-LLVQ-Tetra),
  the one this package reads.
- [Qwen3-4B-LLVQ-Tetra-sealed](https://huggingface.co/Pier-Jean/Qwen3-4B-LLVQ-Tetra-sealed),
  the same weights as one file for the Rust engine.

## Credit

The method is [arXiv:2603.11021](https://arxiv.org/abs/2603.11021), van der
Ouderaa, van Baalen, Whatmough and Nagel, 2026. This is an independent
implementation in Rust, with this package as its Python reader. The code, the
measurement logs and the preregistrations are at
[github.com/pjmalandrino/llvq](https://github.com/pjmalandrino/llvq).

Apache-2.0 or MIT, your choice.
