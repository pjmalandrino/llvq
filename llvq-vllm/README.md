# llvq-vllm

A prototype that serves a Tetra-quantized LLVQ model inside vLLM 0.26.0 as an out-of-tree quantization method. The
weights stay compressed on the card: the Tetra records, the int4 g128 records and the int4 g64 tied head are read by
the served kernels of this repository, never widened to f16. Nothing here has run on a card yet.

The design, the open questions and the gates are in [`docs/plan-vllm.md`](../docs/plan-vllm.md).

## Install, inside the vLLM image

```bash
pip install --no-deps ./llvq-tetra ./llvq-vllm
```

`--no-deps` keeps the image's torch 2.11.0. `llvq-tetra` declares torch>=2.14 for its `transformers` path, and a
resolver would replace the image's torch. The entry point `vllm.general_plugins: llvq` registers the method in every
process vLLM starts.

## Serve

```bash
vllm serve <packed directory> --dtype float16 --tokenizer Qwen/Qwen3-4B
```

The packed directory is what `bin/hfpack` writes, or the Hub repository
[Pier-Jean/Qwen3-4B-LLVQ-Tetra](https://huggingface.co/Pier-Jean/Qwen3-4B-LLVQ-Tetra). `--dtype float16` is required:
every kernel reads and writes f16, and the method refuses bf16 by name. Tensor parallelism is refused, one card only.

Two variables, both refused when out of range:

| variable | default | effect |
|---|---|---|
| `LLVQ_HF_TILE` | 64 | Tetra kernel tile, a compile-time define; 64 on sm_89, 32 on sm_120, 128 elsewhere |
| `LLVQ_VLLM_ROWS_MAX` | 16 | up to this many tokens a call, the row kernels; above, dequantize then GEMM |

## Test, on a Mac

```bash
cd llvq-vllm && uv run --group dev pytest
```

The tests run the plan, the buffers, the loaders and the forward of every layer of the one-layer fixture on CPU
references of the kernels, against the f64 reconstruction of `llvq-tetra`. vLLM itself is not installed: its adapter,
`llvq_vllm/vllm_glue.py`, is checked only for the fact that it fails by name without vLLM.
