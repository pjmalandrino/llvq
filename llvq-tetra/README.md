# llvq-tetra

Reads a Tetra-quantized LLVQ model in `transformers`, from a directory that stays
compressed on disk.

Stage 1 of [`docs/plan-transformers.md`](../docs/plan-transformers.md). The
directory is written by `cargo run -p llvq-llm --bin hfpack`, 1.42 GB for the
served Qwen3-4B against 8 GB for the f16 checkpoint `bin/export` produces.

This package is developed here and extracted at stage 5, with
`git subtree split -P llvq-tetra`. It is self-contained on purpose: its own
`pyproject.toml`, its own tests, and a fixture small enough that nothing here
needs the 1.4 GB object.

## The gate

Bit-exactness against the Rust decoder, not closeness:

```bash
cargo run --release -p llvq-llm --bin hfdense -- <sealed.llvq> <dir>/llvq-dense-digest.json
cd llvq-tetra && uv run python -m llvq_tetra.checkdense <dir>
```

`bin/hfdense` writes one SHA-256 per record of the f32 values
`llvq_artifact::decode_matrix` produces; `checkdense` rebuilds every one of them
from the safetensors alone. 253 of 253 on the served 4B, in 167 s
([journal](../docs/mesures/hf-quantizer-4b-2026-09-30.txt)).

## The tables

`llvq_tetra/data/tetra-tables.safetensors` is the Tetra map, 19 KB, dumped from
`llvq_search::tetra` by `cargo run -p llvq-llm --bin tetratables`. It ships with
the package and not with every model, because the map belongs to the codebook:
that is why a `.llvq` header carries a fingerprint and no table. The package
refuses a model whose fingerprint is not the one its tables were dumped under.

## What it does not do

No speed claim, at any stage of this plan. `transformers` is not a throughput
engine and no number it prints is divided against ours or vLLM's.

The loaded model is dense: each module dequantizes at load and frees its code
buffers. What is compressed is the file. The kernels that read the compressed
form on a card are stages 2 to 4.
