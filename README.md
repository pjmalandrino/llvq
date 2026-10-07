# LLVQ in Rust

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22133606.svg)](https://doi.org/10.5281/zenodo.22133606)

An independent Rust implementation of Leech-lattice vector quantization for LLM weights
([arXiv:2603.11021](https://arxiv.org/abs/2603.11021), Qualcomm AI Research, 2026). The core crates (lattice, exact
search, indexing, GPTQ, file format) have no external dependency and can be read end to end. A fused CUDA kernel
serves the codes inside the model without dequantizing to a dense matrix.

## Results

Three Qwen3 models, sealed at about 2.7 bits per parameter, one served configuration, one NVIDIA L40S.

| model | b/param | MMLU | WikiText-2 ppl | tok/s | GB of weights |
|---|---|---|---|---|---|
| Qwen3-4B | 2.73 | 63.37 | 12.58 | 113.8 | 1.38 |
| Qwen3-8B | 2.70 | 69.58 | 9.71 | 95.0 | 2.76 |
| Qwen3-14B | 2.73 | 75.66 | 8.44 | 57.2 | 5.04 |

MMLU is 5-shot on all 14,042 test questions, micro average, scored on the dense reconstruction of the sealed file.
Perplexity is read the same way, on the raw WikiText-2 test split. Speed is batch 1, greedy, 256 tokens, median of
five rounds. Bits per parameter count the whole model, tables
included. Every cell is *measured*; the journals are in [`docs/ETAT.md`](docs/ETAT.md) §2.

Against 4-bit AWQ, on the same questions, our files score 4.76, 4.21 and 2.46 points lower, and use 45 to 52% of its
bits per parameter. Both gaps shrink as the model grows. At 4B we score 23.6 points above llama.cpp's IQ2_XXS for 0.25
more bits per parameter. The full comparison, with intervals, is in [`docs/ETAT.md`](docs/ETAT.md) §3. The speed is an
L40S result: on an A100 the files decode slower than FP16 in vLLM ([`docs/ETAT.md`](docs/ETAT.md) §4).

## How the bits come down

A block of 24 weights is one 48-bit code on the Leech lattice. The codebook has 1.1e14 points, so no kernel can hold a
lookup table for it. Our first layout, `Planes14`, expanded each code when loading the model and read **4.804 bits per
weight** from VRAM for 2 bits of code. `Tetra` reads the code as it is stored, through a 64-state trellis of the Golay
code and one 16 KiB table shared by every block: **2.150 bits per weight** (*computed*,
`docs/data/echelle-formats.csv`). Three further changes, which leave the lattice codes untouched, build the files
above: one trained scale per weight row, the matrices that lose the most stored as 4-bit integers, and the embedding
tables in 4 bits to pay for them.

The whole of it is written up in [`paper2/`](paper2/README.md), *Tetra: Serving Leech-Lattice Quantized LLMs at 2.7
Bits per Parameter*. Version 2 is attached as a PDF to release
[`v0.0.2`](https://github.com/pjmalandrino/llvq/releases/tag/v0.0.2).

## What you can download

The three sealed files are public on the Hugging Face Hub, each with its card, Qwen's license and its served
configuration. A different hash is a different file.

| model | repository | file | SHA-256 |
|---|---|---|---|
| Qwen3-4B | [Pier-Jean/Qwen3-4B-LLVQ-Tetra-sealed](https://huggingface.co/Pier-Jean/Qwen3-4B-LLVQ-Tetra-sealed) | `qwen3-4b-sealed.bin` | `886391a8…` |
| Qwen3-8B | [Pier-Jean/Qwen3-8B-LLVQ-Tetra-sealed](https://huggingface.co/Pier-Jean/Qwen3-8B-LLVQ-Tetra-sealed) | `qwen3-8b-sealed-B.bin` | `7bdb9a55…` |
| Qwen3-14B | [Pier-Jean/Qwen3-14B-LLVQ-Tetra-sealed](https://huggingface.co/Pier-Jean/Qwen3-14B-LLVQ-Tetra-sealed) | `qwen3-14b-sealed.bin` | `61db37fe…` |

The container is this repository's own: not GGUF, not AWQ, not safetensors, and no other runtime reads it.

```bash
git clone https://github.com/pjmalandrino/llvq && cd llvq
hf download Pier-Jean/Qwen3-4B-LLVQ-Tetra-sealed qwen3-4b-sealed.bin --local-dir .
shasum -a 256 qwen3-4b-sealed.bin

# Linux and CUDA: chat through the served kernel.
LLVQ_CONFIG=configs/qwen3-4b-tetra-e4.json \
  cargo run --release -p llvq-llm --features cuda --bin chat -- qwen3-4b-sealed.bin cuda

# Apple silicon: the dense reconstruction.
cargo run --release -p llvq-llm --features metal --bin chat -- qwen3-4b-sealed.bin metal

# Perplexity of that file (12.5834 on an L40S, token fingerprint 3f1baca9033bf251).
LLVQ_DTYPE=f16 cargo run --release -p llvq-llm --features cuda --bin ppl -- 4096 12 cuda qwen3-4b-sealed.bin

cargo test                                 # fast loop, minutes
```

The earlier `Planes14` object of release `v0.0.1` stays at
[Pier-Jean/Qwen3-4B-LLVQ-2bit](https://huggingface.co/Pier-Jean/Qwen3-4B-LLVQ-2bit), `qwen3-4b-llvq.bin`, sha256
`9db213ef9fa9d7d7000789a8a529ce9459ce9ba6002ef5a72fd5a1c05c1c84b0`. It reads 5.162 b/param and scores 55.59 on MMLU,
and its numbers are in [`docs/fiche-4b.md`](docs/fiche-4b.md). The archive tests read it, in tens of minutes:
`LLVQ_SEALED_ARTIFACT=$PWD/qwen3-4b-llvq.bin cargo test --release -- --include-ignored`.

Feature flags: `metal` (macOS GPU), `cuda` (Linux, NVRTC at startup), `fast-linalg` (`faer`, required in practice for
quantization: the dependency-free path is 40 times slower for a bit-identical result, *measured*,
[`smoke.rs`](llvq-llm/src/bin/smoke.rs)). The runtime switches and every binary are listed in
[`CLAUDE.md`](CLAUDE.md).

Requantizing the 4B takes 4.0 h on an M3 Max (*measured*, [`docs/fiche-4b.md`](docs/fiche-4b.md)), and the recipe no
longer reproduces the `Planes14` bytes of 2026-08. One block replayed on 2026-09-06 differs on the tail, the gains and
87% of the indices, most likely from a calibration-volume change of 2026-08-26 (*measured*,
`llvq-bench/examples/driftcheck.rs`). The three sealed files were all encoded after that date.

## Repository map

| crate | role | dependencies |
|---|---|---|
| `llvq-core` | Golay [24,12,8], Leech lattice, shells | none, `forbid(unsafe_code)` |
| `llvq-search` | exact nearest-neighbour search, 48-bit bijective index, packing | none, `forbid(unsafe_code)` |
| `llvq-quant` | GPTQ, dense algebra, quantizers | none by default; `faer` behind `fast-linalg` |
| `llvq-artifact` | the `.llvq` container: writer, reader, decoder | none, `forbid(unsafe_code)` |
| `llvq-bench` | rate-distortion, encoder throughput, decode cost | none, `forbid(unsafe_code)` |
| `llvq-metal` | macOS GPU micro-benches and the rank decoders | `metal` |
| `llvq-cuda` | the fused kernel, layouts, benches, Linux only | `cudarc` |
| `llvq-llm` | model loading, forward pass, calibration, perplexity, MMLU, served path | `candle` |

`unsafe` appears only at hardware boundaries (mmap, kernel launch, device reads) in the last three crates.

## Documents

- [`docs/ETAT.md`](docs/ETAT.md): served configuration, headline numbers, open decisions.
- [`docs/ROADMAP.md`](docs/ROADMAP.md): what comes next, with gates and costs.
- [`docs/HISTORIQUE.md`](docs/HISTORIQUE.md): the chronological thread, one entry per period.
- [`docs/METHODE.md`](docs/METHODE.md): the lab rules. [`docs/STYLE.md`](docs/STYLE.md): how these documents are written.
- [`docs/fiche-4b.md`](docs/fiche-4b.md): every number on the published file, with its provenance.
- [`docs/format-noyau.md`](docs/format-noyau.md): the VRAM layouts and the measurement traps.
- [`docs/hub/`](docs/hub/): the cards of the three files on the Hugging Face Hub.
- [`paper2/`](paper2/README.md): the second paper. [`ARTIFACT-EVALUATION.md`](ARTIFACT-EVALUATION.md): reviewer instructions.

## Method

Every paid experiment is preregistered in [`proofs/`](proofs/) with its kill criterion and a signed prediction,
before the first measurement, and most preregistrations are stamped with OpenTimestamps. A preregistration is never
edited; deviations go in a companion `-ECARTS.md`, including the predictions that were wrong. Every number carries a
provenance label, *measured*, *computed* or *estimated*, and a link to its journal. Raw outputs are committed. Speed is
published as a median with its range, and speeds from two engines are never divided.

As of 2026-10-06: 198 measurement journals and 72 raw-output directories in [`docs/mesures/`](docs/mesures/), 105
preregistrations, 96 of them stamped, and 58 deviation files in [`proofs/`](proofs/), 57 per-question MMLU dumps, and
221 priced jobs for $246.45 in [`docs/data/jobs.csv`](docs/data/jobs.csv).

## Licence

Code: MIT OR Apache-2.0. The Qwen3 forward pass in `llvq-llm/src/model.rs` is derived from
[candle-transformers](https://github.com/huggingface/candle) (MIT OR Apache-2.0). The published models are Apache 2.0,
inherited from Qwen3. The QTIP kernel used in one bench is GPL v3
upstream and is not redistributed here ([`docs/qtip-provenance.md`](docs/qtip-provenance.md)).
