# LLVQ in Rust

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22133606.svg)](https://doi.org/10.5281/zenodo.22133606)

An independent Rust implementation of Leech-lattice vector quantization for LLM weights
([arXiv:2603.11021](https://arxiv.org/abs/2603.11021), Qualcomm AI Research, 2026). The core crates (lattice, search,
indexing, GPTQ, file format) have no external dependency. Fused CUDA and Metal kernels read the 2-bit codes inside
the model without ever building a dense matrix.

How it works, from the Golay code to the one-launch CUDA token: [`docs/how-it-works.md`](docs/how-it-works.md).

## Results

Three Qwen3 models, sealed at about 2.7 bits per parameter and served on one NVIDIA L40S.

| model | b/param | MMLU | GSM8K | WikiText-2 ppl | tok/s | GB on card |
|---|---|---|---|---|---|---|
| Qwen3-4B | 2.73 | 63.37 | 82.49 | 12.58 | 221.5 | 1.38 |
| Qwen3-8B | 2.70 | 69.58 | 88.63 | 9.71 | 142.5 | 2.76 |
| Qwen3-14B | 2.73 | 75.66 | 92.04 | 8.44 | 86.3 | 5.04 |

Every cell is *measured*; the journals are linked from [`docs/ETAT.md`](docs/ETAT.md) §2. MMLU is 5-shot on all 14,042
test questions. Speed is batch 1, greedy, 256 tokens after a 5-token prompt, median of five rounds. Bits per parameter
count the whole model, embedding and tables included.

The speeds come from the served form: one CUDA launch a token (4E) and our split-KV attention kernel. The scores were
measured under the expanded attention form of paper 2. The served form returns the eager engine's tokens, and its
attention stays within ±0.12 % of the expanded form's perplexity. Paper 2 reports 113.8, 95.0 and 57.2 tok/s, the
speeds of that earlier form. Throughput falls with context: at 8,192 tokens of prompt the 4B decodes 161.5 tok/s.

Against 4-bit AWQ on the same questions, our files score 4.76, 4.21 and 2.46 MMLU points lower, at 45 to 52 % of its
bits per parameter. At 4B we score 23.6 points above llama.cpp's IQ2_XXS for 0.25 more bits per parameter. The speed
is an L40S result: on an A100 our files decode slower than FP16 in vLLM ([`docs/ETAT.md`](docs/ETAT.md) §3 and §4).

## Run it

The sealed files are on the Hugging Face Hub. A different hash is a different file.

| model | repository | file | SHA-256 |
|---|---|---|---|
| Qwen3-4B | [Pier-Jean/Qwen3-4B-LLVQ-Tetra-sealed](https://huggingface.co/Pier-Jean/Qwen3-4B-LLVQ-Tetra-sealed) | `qwen3-4b-sealed.bin` | `886391a8…` |
| Qwen3-8B | [Pier-Jean/Qwen3-8B-LLVQ-Tetra-sealed](https://huggingface.co/Pier-Jean/Qwen3-8B-LLVQ-Tetra-sealed) | `qwen3-8b-sealed-B.bin` | `7bdb9a55…` |
| Qwen3-14B | [Pier-Jean/Qwen3-14B-LLVQ-Tetra-sealed](https://huggingface.co/Pier-Jean/Qwen3-14B-LLVQ-Tetra-sealed) | `qwen3-14b-sealed.bin` | `61db37fe…` |

The container is this repository's own format; no other runtime reads it. The easiest way in is the `llvq` runtime,
which speaks Ollama's and OpenAI's APIs ([`llvq-runtime/`](llvq-runtime/README.md)):

```bash
git clone https://github.com/pjmalandrino/llvq && cd llvq
cargo build --release -p llvq-runtime --features cuda    # Linux; --features metal on a Mac
target/release/llvq pull qwen3:4b                        # checked against its pinned sha256
target/release/llvq run qwen3:4b "Name three primary colours."
```

On CUDA the runtime serves the v9 configs, greedy only. On a Mac it runs the Rust Metal engine; the faster MLX door
for the 4B is `ops/llvq-mlx chat` ([`docs/plan-metal-debit.md`](docs/plan-metal-debit.md)).

Perplexity of the 4B (12.5834 on an L40S, token fingerprint `3f1baca9033bf251`):

```bash
LLVQ_DTYPE=f16 cargo run --release -p llvq-llm --features cuda --bin ppl -- 4096 12 cuda qwen3-4b-sealed.bin
```

### In `transformers`

The 4B is also published as compressed safetensors,
[Pier-Jean/Qwen3-4B-LLVQ-Tetra](https://huggingface.co/Pier-Jean/Qwen3-4B-LLVQ-Tetra), read by the
[`llvq-tetra`](https://pypi.org/project/llvq-tetra/) package ([`llvq-tetra/`](llvq-tetra/README.md)).

```bash
pip install llvq-tetra
```

```python
import llvq_tetra  # registers the method; without it, transformers loads a random model and does not raise
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "Pier-Jean/Qwen3-4B-LLVQ-Tetra"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype="float32")
```

It gives the same 256 greedy tokens as our engine (*measured*,
[hf-testpypi](docs/mesures/hf-testpypi-4b-2026-10-07.txt)).

### Paper 1's file

The earlier `Planes14` 4B of release `v0.0.1` stays at
[Pier-Jean/Qwen3-4B-LLVQ-2bit](https://huggingface.co/Pier-Jean/Qwen3-4B-LLVQ-2bit). Its numbers are in
[`docs/fiche-4b.md`](docs/fiche-4b.md), its runbook in [`LAUNCH_ME.md`](LAUNCH_ME.md), and the reviewer notes in
[`ARTIFACT-EVALUATION.md`](ARTIFACT-EVALUATION.md).

## Repository map

| crate | role |
|---|---|
| `llvq-core` | Golay code, Leech lattice, shells |
| `llvq-search` | nearest-point search, the v1 index, the Tetra word and encoder |
| `llvq-quant` | GPTQ and the block loop; `faer` behind `fast-linalg` |
| `llvq-artifact` | the `.llvq` container and its VRAM transcoding |
| `llvq-bench` | rate-distortion and decode-cost benches |
| `llvq-metal` | macOS GPU shaders and benches |
| `llvq-cuda` | CUDA kernels, 4E, benches (Linux) |
| `llvq-llm` | Qwen3 forward pass, served configs, perplexity, MMLU, GSM8K |
| `llvq-runtime` | the `llvq` binary: pull, run, serve |

The first five have no external dependency and forbid `unsafe`. Feature flags: `metal`, `cuda` (NVRTC at startup),
`fast-linalg` (needed in practice to quantize: the dependency-free path is 40 times slower, bit for bit the same).
Every binary and switch is listed in [`CLAUDE.md`](CLAUDE.md). `cargo test` is the fast loop.

## Documents

- [`docs/how-it-works.md`](docs/how-it-works.md): the maths, the lattice, the format and the kernel.
- [`docs/ETAT.md`](docs/ETAT.md): served configuration, numbers, open decisions.
- [`docs/ROADMAP.md`](docs/ROADMAP.md): what comes next.
- [`docs/METHODE.md`](docs/METHODE.md): the lab rules. [`docs/README.md`](docs/README.md): every other document.
- [`paper2/`](paper2/README.md): *Tetra: Serving Leech-Lattice Quantized LLMs at 2.7 Bits per Parameter*, version 2
  attached to release [`v0.0.2`](https://github.com/pjmalandrino/llvq/releases/tag/v0.0.2).

## Method

Every paid experiment is preregistered in [`proofs/`](proofs/) with its kill criterion and a signed prediction, before
the first measurement, and most preregistrations are stamped with OpenTimestamps. Deviations go in a companion
`-ECARTS.md`, wrong predictions included. Every number carries a label, *measured*, *computed* or *estimated*, and a
link to its journal in [`docs/mesures/`](docs/mesures/). Raw outputs are committed. Speeds from two engines are never
divided.

## Licence

Code: MIT OR Apache-2.0. The Qwen3 forward pass in `llvq-llm/src/model.rs` is derived from
[candle-transformers](https://github.com/huggingface/candle) (MIT OR Apache-2.0). The published models are Apache 2.0,
inherited from Qwen3. The QTIP kernel used in one bench is GPL v3 upstream and is not redistributed
([`docs/qtip-provenance.md`](docs/qtip-provenance.md)).
