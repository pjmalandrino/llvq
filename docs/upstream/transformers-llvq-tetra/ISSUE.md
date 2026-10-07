# Add LLVQ quantization (Leech lattice, 2.7 bits per parameter)

<!--
Draft, not posted. It goes out together with the PR. Before posting:
- put the PR number in "Your contribution", and check the PR does what that paragraph says;
- the snippet ran as written on 2026-10-07 (docs/mesures/hf-snippet-2026-10-07.txt);
- re-read each number against its journal in docs/mesures/.
Once posted, this file becomes the verbatim archive of the body and is not edited again,
as ../candle-broadcast-matmul/ISSUE.md is.
-->

### Feature request

I'd like to add LLVQ to transformers as a quantization method, `quant_method: "llvq"`.

LLVQ stores weights as points of the Leech lattice: 24 weights per 48-bit code, plus one scale per row. The method is
from van der Ouderaa et al. ([arXiv:2603.11021](https://arxiv.org/abs/2603.11021)). I wrote an independent
implementation in Rust, and a Python reader for transformers.

On Qwen3-4B, 168 of the 252 linear layers use the lattice. The other 84, and the embedding, are 4-bit groups. The whole
model comes to 2.73 bits per parameter, 1.38 GB of weights.

### Motivation

It is smaller than 4-bit, and it costs quality. Here is how much:

| Qwen3-4B | bits/param | weights | MMLU, 5-shot | GSM8K, 0-shot |
|---|---|---|---|---|
| FP16 | 16.00 | 8.04 GB | 70.14 | 92.12 |
| AWQ w4g128 | 5.30 | 2.67 GB | 68.14 | 89.01 |
| LLVQ | 2.73 | 1.38 GB | 63.37 | 82.49 |

MMLU is on all 14,042 questions and GSM8K on all 1,319 problems. Paired on the same questions, LLVQ is 6.8 MMLU points
and 9.6 GSM8K points below FP16. These numbers come from my Rust engine on an L40S, not from transformers. Through
transformers I checked that the model gives the same 256 greedy tokens as the engine, nothing more.

The second reason is a problem on the transformers side. A method that is not in tree only exists if the user imports
its package. If they don't, `from_pretrained` prints

```
Unknown quantization type, got llvq - supported types are [...]. Hence, we will skip the quantization.
```

then reinitializes the 254 weights it can't find and returns a model with random weights. No exception, and the logits
look normal. Anything that loads models by name without importing the package, lm-eval for example, would score that
random model.

`trust_remote_code=True` works around it, through an `auto_map` entry in the repo. A caller who passes nothing still
gets the random model, because `model_type: "qwen3"` picks the class directly. In tree, the problem goes away.

### Your contribution

PR #____ adds the method in tree, following `docs/source/en/quantization/contribute.md`: `LlvqConfig`, a quantizer,
the module swap, tests and a doc page. The layers and kernels live in the `llvq-tetra` package (PyPI, MIT or
Apache-2.0), the way `aqlm` and `vptq` rely on their own packages.

It already works out of tree:

```python
# pip install llvq-tetra
import llvq_tetra  # registers "llvq"
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "Pier-Jean/Qwen3-4B-LLVQ-Tetra"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype="float32")
ids = tok("The capital of France is", return_tensors="pt")
print(tok.decode(model.generate(**ids, max_new_tokens=16)[0]))
```

What I checked:

- the packed safetensors rebuild bit for bit against the Rust decoder;
- `from_pretrained` gives the engine's 256 greedy tokens on CPU, on Apple silicon and on an NVIDIA L4, and from a clean
  `pip install` with transformers 5.18.0 and 5.19.0;
- a 148 KB one-layer model in the tests runs `from_pretrained` with no download and no GPU.

What is missing:

- no quality number measured through transformers yet, only the token check;
- by default the weights are rebuilt dense at load, two and a half to five minutes for the 4B on CPU. The Metal and
  CUDA kernels that keep them compressed (2.75 GB allocated on Apple silicon) compile at import and need ninja. There
  are no prebuilt kernels yet;
- those kernels are matrix-vector, so the compressed path runs one token at a time and is slow on long prompts;
- on CUDA, the 4-bit layers are not fused yet: 168 layers of 252;
- only Qwen3 is tested;
- `save_pretrained` can't write the format. The encoder is offline, in Rust.

If you'd rather keep this out of tree, I understand. In that case, would you consider an entry point for out-of-tree
quantizers, or an error instead of a warning when `quant_method` is unknown? Either would fix the random-model problem
for every out-of-tree method, not just this one.

### References

- LLVQ: van der Ouderaa, van Baalen, Whatmough, Nagel, [arXiv:2603.11021](https://arxiv.org/abs/2603.11021).
- This implementation: [github.com/pjmalandrino/llvq](https://github.com/pjmalandrino/llvq). The write-up, *Tetra:
  Serving Leech-Lattice Quantized LLMs at 2.7 Bits per Parameter*, is attached to release
  [v0.0.2](https://github.com/pjmalandrino/llvq/releases/tag/v0.0.2).
- Models: [Qwen3-4B-LLVQ-Tetra](https://huggingface.co/Pier-Jean/Qwen3-4B-LLVQ-Tetra), the safetensors, and
  [Qwen3-4B-LLVQ-Tetra-sealed](https://huggingface.co/Pier-Jean/Qwen3-4B-LLVQ-Tetra-sealed), the file the Rust engine
  reads. Both carry the same artifact digest.
- Tested with torch 2.14.1, Python 3.12, macOS 26.6 on an M3 Max, and an NVIDIA L4 for CUDA.
