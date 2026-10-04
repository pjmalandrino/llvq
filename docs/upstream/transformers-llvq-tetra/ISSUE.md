# Add LLVQ Tetra, a Leech-lattice vector quantization method, at 2.73 bits per parameter

<!--
DRAFT — not posted. When it is posted, this file becomes the verbatim archive of
the body, and stops being editable; see ../candle-broadcast-matmul/ISSUE.md for
the convention. Until then, edit freely.

Target: huggingface/transformers, as a feature request, per CONTRIBUTING.md which
asks for the motivation, the detail, a code snippet and the paper link. The
quantization contribution page (docs/source/en/quantization/contribute.md) lists
ten code steps and never mentions an issue, so this exists to ask whether the
method is wanted before the ten steps are taken.
-->

### Feature request

Support `quant_method: "llvq"` in tree: weights stored on the Leech lattice Λ₂₄,
48 bits per block of 24 weights with one scale per row, which puts Qwen3-4B at
**2.73 bits per parameter over the whole model**, embedding included.

A registered out-of-tree quantizer already works and is published, so this is a
request to adopt rather than a request to build. The code is at
[pjmalandrino/llvq](https://github.com/pjmalandrino/llvq) under `llvq-tetra/`,
MIT or Apache-2.0.

### Motivation

**The sizes.** Qwen3-4B in 1.4 GB of safetensors instead of 8, and the weights can
stay compressed in memory: on Apple silicon the loaded model holds 2.75 GB with
all 252 projections compressed, because a fused kernel decodes and multiplies in
one pass rather than materializing a dense matrix.

| | LLVQ Tetra | FP16 | AWQ w4g128 | IQ2_XXS |
|---|---|---|---|---|
| bits per parameter, whole model | **2.73** | 16.00 | 5.30 | 2.48 |
| weight bytes | **1.38 GB** | 8.04 GB | 2.67 GB | 1.25 GB |
| MMLU, 5-shot, 14,042 questions | **63.37** | 70.14 | 68.14 | 39.78 |
| GSM8K, zero-shot, 1,319 problems | **82.49** | 92.12 | 89.01 | not scored |

Paired on the same questions, 6.77 MMLU points [6.05, 7.50] and 9.63 GSM8K points
[7.69, 11.57] below FP16. Measured on one NVIDIA L40S, by the Rust engine, and
**not** through this loader; what is checked through the loader is token identity
against that engine. The gap is real and the card says so in those words.

**The precedent.** Vector quantization is already in tree: `aqlm`, `vptq`,
`higgs`, `spqr`. This is the same shape of contribution, on a different lattice.

**A problem on your side, which is the part I would most like your opinion on.**
`transformers` discovers quantization methods by import and not by entry point:
`quantizers/auto.py` has no such mechanism. So a method that is not in tree is
registered only by the user importing its package, and a caller who does not gets
this, on a published model, with no exception raised:

```
Unknown quantization type, got llvq - supported types are [...].
Hence, we will skip the quantization.
```

`from_pretrained` then reinitializes the 254 weights it finds missing and returns
a model that generates fluent nonsense. Measured on the published 4B: 254 missing
keys, 1119 unexpected, every weight at Qwen3's own init std of 0.0200, and a
forward pass whose logits sit in an ordinary range. A benchmark harness such as
`lm-eval` imports `transformers` and not our package, so it would score a randomly
initialized model and publish the number as the quantized model's.

I could not close this from the repository side. `auto_map` does not help, tried
on `AutoConfig` and on `AutoModelForCausalLM`: with a known `model_type` you
resolve a class from the type and never consult the map. An unresolvable
`model_type` does make it refuse, at the cost of the `qwen3` type string and of
`trust_remote_code=True` for everyone. In tree the problem does not exist, which
is the strongest argument I have for being in tree, and it is an argument about
`transformers` rather than about this method.

### Your contribution

A working implementation, a published model, and the code snippet below. I am
offering to do the ten steps of
`docs/source/en/quantization/contribute.md`, and asking first because an in-tree
method is a maintenance commitment for you and a wasted month for me if the answer
is no.

```python
# pip install llvq-tetra
import torch
import llvq_tetra  # registers the method; the issue above is about this line
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "Pier-Jean/Qwen3-4B-LLVQ-Tetra"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype="float16")

dev = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(dev)
text = tok.apply_chat_template(
    [{"role": "user", "content": "Explain a hash map to a beginner."}],
    tokenize=False, add_generation_prompt=True, enable_thinking=False,
)
ids = tok(text, return_tensors="pt").to(dev)
print(tok.decode(model.generate(**ids, max_new_tokens=120)[0]))
```

The quantizer is 282 lines against `HfQuantizer`, plus 152 for the modules and 52
for the weight conversion: it replaces `nn.Linear`
in `_process_model_before_weight_loading`, uses `get_weight_conversions()` with a
`WeightConverter` for the three-keys-to-one-parameter case of the 4-bit records,
and either materializes dense weights at load or arms a fused matvec.

**What is done, with its evidence.** Every claim below has a measurement log and a
timestamped preregistration in the repository, under `docs/mesures/` and
`proofs/`.

- The packed directory rebuilds **bit for bit** against the Rust decoder: 1,602
  fields, and 253 SHA-256 digests of the dequantized matrices.
- `from_pretrained` gives the same 256 greedy tokens as the engine, on CPU, on
  Metal with every projection compressed, and on an NVIDIA L4.
- Metal and CUDA kernels exist, reached as `torch.ops`. The Metal 4-bit matvec is
  bit-identical to the one the engine serves; the CUDA one is built by `nvcc` from
  the engine's own source rather than a copy.
- A 148 KB one-layer fixture in the test suite exercises `from_pretrained` with
  no download and no GPU.

**What is not done, and I would rather you heard it from me.**

- **No quality number measured through this loader.** The table above is the Rust
  engine's. The loader is held to token identity against it, which is a weak gate:
  a defect worth 8.79 % of a matrix row once left 64 greedy tokens untouched on
  two prompts of four.
- **The kernels compile at import**, so the fused path needs `ninja` and a
  compiler. No Kernel Hub build yet.
- **One architecture.** `Qwen3ForCausalLM`. The code routes on record names rather
  than on the architecture, so others may work, and none has been tried.
- **`is_serializable()` returns false.** `save_pretrained` would write dense
  weights. Writing the format needs the encoder, which is Rust.
- **The 4-bit matvec is not bound on CUDA**, so 168 of 252 projections are fused
  there against 252 on Metal.
- **Load is slow on the dense path**, 152 s for the 4B, because it rebuilds 252
  matrices before the first token. The fused path loads in 8 s and generates more
  slowly.

Happy to be told this belongs out of tree. In that case the one thing I would ask
is whether entry-point discovery for quantizers is something you would consider,
since it would close the silent-load problem for every out-of-tree method and not
only this one.

### References

- The method: van der Ouderaa, van Baalen, Whatmough, Nagel, *LLVQ*,
  [arXiv:2603.11021](https://arxiv.org/abs/2603.11021), 2026. This is an
  independent implementation, in Rust, with a Python reader.
- The engineering, in a paper under review: *Tetra: Serving Leech-Lattice
  Quantized LLMs at 2.7 Bits per Parameter*, in the repository under `paper2/`.
  Earlier layout at DOI
  [10.5281/zenodo.22133606](https://doi.org/10.5281/zenodo.22133606).
- Models: [Qwen3-4B-LLVQ-Tetra](https://huggingface.co/Pier-Jean/Qwen3-4B-LLVQ-Tetra)
  for safetensors, and `-Tetra-sealed` for the single file the Rust engine reads.
  Both carry the same artifact digest.

### System info

`transformers` 5.18.0, `torch` 2.14.1, Python 3.12, macOS 25.6 on an M3 Max, and
an NVIDIA L4 for the CUDA arm.
