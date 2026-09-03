# LLVQ in Rust

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22133606.svg)](https://doi.org/10.5281/zenodo.22133606)

## Overview

An independent Rust implementation of Leech-lattice vector quantization for
LLM weights ([arXiv:2603.11021](https://arxiv.org/abs/2603.11021), Qualcomm AI
Research, 2026). The core crates (lattice, exact search, indexing, GPTQ, file
format) have no external dependency and can be read end to end. A fused CUDA
kernel serves the 2-bit codes inside the model without dequantizing to a dense
matrix. The write-up is on Zenodo
([10.5281/zenodo.22133606](https://doi.org/10.5281/zenodo.22133606)).

## Results

| model | served tok/s | GB on card | bits/param | ppl, LLVQ / f16 | MMLU, LLVQ / f16 |
|---|---|---|---|---|---|
| Qwen3-4B | 100.6 [99.9–100.7] | 2.57 | 5.162 | 16.94 / 12.24 (×1.384) | 55.59 / 70.32 |
| Qwen3-8B | 75.5 [75.5–75.6] | 5.41 | 5.322 | 10.97 / 8.99 (×1.220) | 65.52 / 76.08 |
| Qwen3-14B | 46.8 [46.7–46.8] | 9.40 | 5.106 | 9.49 / 7.98 (×1.189) | 72.12 / 78.97 |

Three Qwen3 sizes, one served configuration (`Planes14` layout, int8
embedding, hoisted rotation, fused launches), one card (L40S). Every cell is
*measured*. Throughput is the median of 5 rounds with its range, 128 greedy
tokens, batch 1. "GB on card" is a host-side byte count. Bits per parameter
count the whole model, embedding included.

Sources: throughput and memory [d1](docs/mesures/d1-fusion-servie-2026-08-24.txt)
and [vague2](docs/mesures/vague2-fusion-8b-14b-2026-08-31.txt); bits/param
[rtbits-14b](docs/mesures/rtbits-14b-2026-08-17.txt); quality
[a4](docs/mesures/a4-campagne-2026-08-06.txt),
[8b](docs/mesures/campagne-8b-qualite-2026-08-08.txt),
[14b](docs/mesures/campagne-14b-qualite-2026-08-10.txt).
Perplexity is WikiText-2, context 4096, 12 windows, f16 on both arms, same
token fingerprint. MMLU is 5-shot, 2 280 questions, micro average.

The kernel-only speedup is the ratio at identical head: ×1.11, ×1.29, ×1.41
from 4B to 14B (*calculated* from the measured medians,
[b2-fusedrun-plages-2026-08-18.txt](docs/mesures/b2-fusedrun-plages-2026-08-18.txt)).
The raw ratio against our own dense path (×2.00 at 4B, *calculated*,
[b2](docs/mesures/b2-fusedrun-plages-2026-08-18.txt)) is inflated by a
defect of that path. That path copies the vocabulary every token.

At 4B the 4-bit format dominates on quality: the paired gap AWQ minus LLVQ is
14.45 pp [11.60; 17.27] (*measured*,
[mmlupair-4b-8b-2026-08-13.txt](docs/mesures/mmlupair-4b-8b-2026-08-13.txt)).
LLVQ beats IQ2_XXS by 16.20 pp [12.64; 19.72] at twice the served memory
(*measured*, [m3-iq2-metal-2026-08-30.txt](docs/mesures/m3-iq2-metal-2026-08-30.txt)).

| Qwen3-4B | bits/param | MMLU | ppl vs f16 |
|---|---|---|---|
| LLVQ 2-bit (this repository) | 5.162 | 55.59 | ×1.384 |
| AWQ w4 g128 (official Qwen) | 5.302 | 70.04 | ×1.105 |
| IQ2_XXS (llama.cpp, Metal) | 2.479 | 39.39 | ×2.629 |

MMLU and perplexity are *measured* ([a4](docs/mesures/a4-campagne-2026-08-06.txt),
[m3-iq2-metal-2026-08-30.txt](docs/mesures/m3-iq2-metal-2026-08-30.txt));
bits/param are *calculated* from measured bytes
([rtbits-planes-8b-2026-08-09.txt](docs/mesures/rtbits-planes-8b-2026-08-09.txt)).
These are the two formats a user would pick instead, on the same 2 280
questions. AWQ was scored on the same card and harness as LLVQ (L40S,
`bin/mmlu`). IQ2_XXS was scored on an M3 Max through llama.cpp and
`ops/gguf_mmlu.py`, with the same decision rule; its perplexity ratio is
intra-engine and the GGUF was not measured on CUDA.
The gap to AWQ is 7.49 pp at 8B and 6.09 pp [3.62; 8.52] at 14B (*measured*,
[mmlupair-14b](docs/mesures/mmlupair-14b-2026-08-17.txt)); three points are
not a scaling law.

The calibration draw moves perplexity by 5.2 % (sigma, n = 3) at 4B
(*calculated* from three measured runs, [f5-graines-4b-2026-08-19.txt](docs/mesures/f5-graines-4b-2026-08-19.txt)),
so each published quality number is one draw. Restoring `v_proj` alone (2.6 %
of the weights) to int4 returns 3.60 pp [1.47; 5.79] of MMLU (*measured*,
[m2b-v4bits-2026-09-02.txt](docs/mesures/m2b-v4bits-2026-09-02.txt)) for
0.013 bits/param less (*calculated*, same journal).
Whether to serve it is an open operator decision.

## The kernel and its cost

The lattice index is unfolded into bit planes in VRAM: 4.804 bits/weight are
read per 2.00 bits of code (*measured*,
[f2-p3-qtip-banc-2026-08-21.txt](docs/mesures/f2-p3-qtip-banc-2026-08-21.txt)).
A codebook of 1.1e14 points fits no lookup table.
`Planes14` decodes at 2.15× FP16 on L40S, at 428 GB/s (*measured*, same
journal). In the same bench and process, QTIP reads 2.40× fewer bytes
(*calculated* from measured bytes, same journal) and finishes the 252
projections 2.27× [2.27–2.28] faster (*calculated* from the two medians,
range widened outward, same journal; the bench does not print this ratio
per round). On A100
every lattice arm is below FP16 (0.79× for `Planes14`); the "decode at matvec
speed" claim holds on Ada only (*measured*,
[f4-a100-2026-08-18.txt](docs/mesures/f4-a100-2026-08-18.txt)).

## Quick start

The model is [Pier-Jean/Qwen3-4B-LLVQ-2bit](https://huggingface.co/Pier-Jean/Qwen3-4B-LLVQ-2bit):
one file, `qwen3-4b-llvq.bin`, 1 770 527 533 bytes, sha256
`9db213ef9fa9d7d7000789a8a529ce9459ce9ba6002ef5a72fd5a1c05c1c84b0`. A
different hash is a different file. The format is this repository's own
container: not GGUF, not AWQ, not safetensors, and no other runtime reads it.

```bash
git clone https://github.com/pjmalandrino/llvq && cd llvq
hf download Pier-Jean/Qwen3-4B-LLVQ-2bit qwen3-4b-llvq.bin --local-dir .
shasum -a 256 qwen3-4b-llvq.bin

# Any machine: decode to f16 and generate 24 tokens (dense path; peak RSS
# measured at 10 GB on CPU, 17.4 GB with Metal, docs/fiche-4b.md).
# Drop `--features metal` and pass `cpu` off a Mac.
cargo run --release -p llvq-llm --features metal --bin run -- qwen3-4b-llvq.bin metal 24

# Linux + CUDA: the fused kernel inside the model, A/B against the dense path,
# same greedy tokens. This is the served configuration.
LLVQ_EMBED=q8 LLVQ_ROT_SHARE=1 LLVQ_FUSE=1 \
  cargo run --release -p llvq-llm --features cuda --bin fusedrun -- qwen3-4b-llvq.bin 128

# Perplexity of the sealed file (expected 16.9415, tokens 3f1baca9033bf251).
LLVQ_DTYPE=f16 cargo run --release -p llvq-llm --features metal --bin ppl -- 4096 12 metal qwen3-4b-llvq.bin

cargo test                                 # fast loop (debug): heavy tests carry cfg_attr(debug_assertions, ignore)
LLVQ_SEALED_ARTIFACT=$PWD/qwen3-4b-llvq.bin cargo test --release -- --include-ignored   # tens of minutes
```

Feature flags: `metal` (macOS GPU), `cuda` (Linux, NVRTC at startup),
`fast-linalg` (`faer`, required in practice for quantization: the dependency-free
path is 40× slower for a bit-identical result, *measured*,
[`smoke.rs:1253-1265`](llvq-llm/src/bin/smoke.rs)). Runtime switches:
`LLVQ_FUSED_LAYOUT=planes14|planes12x|slot32|golay70`, `LLVQ_EMBED=f16|q8`,
`LLVQ_ROT_SHARE`, `LLVQ_FUSE`, `LLVQ_KV=f16|q8`, `LLVQ_RESTORE_F16` and
`LLVQ_RESTORE_Q4` (restore named projections in `mmlu` and `ppl`),
`LLVQ_H_SHRINK` (Hessian shrinkage in `smoke`). Requantizing the 4B takes
4.0 h on an M3 Max (*measured*, [docs/fiche-4b.md](docs/fiche-4b.md)); the
full recipe is in [LAUNCH_ME.md](LAUNCH_ME.md).
That recipe does not reproduce the bytes: the C4 calibration shard moved from `00000` to `00001` after the published run, and the container magic from `LVQ2` to `LVQ4` ([LAUNCH_ME.md](LAUNCH_ME.md)).
A re-run is expected to yield the published file's figures, 1.771 GB at 2.1595 bits/weight (*estimated*: the seal replay is listed as pending in [docs/fiche-4b.md](docs/fiche-4b.md)), in a file that will not be byte-identical; that replay has not been run.
`calib.rs` accumulates AᵀA in f32 on the accelerator, so a CUDA re-run gives different weights; that gap is not measured.

## Repository map

| crate | role | dependencies |
|---|---|---|
| `llvq-core` | Golay [24,12,8], Leech lattice, shells | none, `forbid(unsafe_code)` |
| `llvq-search` | exact nearest-neighbour search, 48-bit bijective index, packing | none, `forbid(unsafe_code)` |
| `llvq-quant` | spherical GPTQ, dense algebra, quantizers | none by default; `faer` behind `fast-linalg` |
| `llvq-artifact` | the `.llvq` container: writer, reader, decoder | none, `forbid(unsafe_code)` |
| `llvq-bench` | rate-distortion, encoder throughput, decode cost | none, `forbid(unsafe_code)` |
| `llvq-metal` | macOS GPU micro-benches and the rank decoders | `metal` |
| `llvq-cuda` | the fused kernel, layouts, benches (Linux only) | `cudarc` |
| `llvq-llm` | model loading, forward pass, calibration, ppl, MMLU, served path | `candle` |

`llvq-artifact` pulls 3 crates in total; `llvq-llm` pulls 261 (*measured*,
`cargo tree`, [audit-publication-2026-08-03.md](docs/archive/audit-publication-2026-08-03.md)).
`unsafe` appears only at hardware boundaries (mmap, kernel launch, device
reads) in the last three crates.

Documents:

- [docs/ETAT.md](docs/ETAT.md): served configuration, headline numbers, open decisions.
- [docs/HISTORIQUE.md](docs/HISTORIQUE.md): one entry per period, dated verdicts.
- [docs/ROADMAP.md](docs/ROADMAP.md): next experiments with gates and costs.
- [docs/METHODE.md](docs/METHODE.md): the lab rules.
- [docs/fiche-4b.md](docs/fiche-4b.md): provenance of every number on the published file.
- [docs/mesures/](docs/mesures/): 101 measurement journals and 10 raw-output directories (counted 2026-09-02).
- [docs/data/jobs.csv](docs/data/jobs.csv): 104 GPU jobs, 94.97 $ billed (5 rows carry no amount; counted 2026-09-02).
- [ARTIFACT-EVALUATION.md](ARTIFACT-EVALUATION.md): reviewer instructions.

## Method

Every paid experiment is preregistered in [proofs/](proofs/) with its kill
criterion and a signed prediction. The file is stamped with OpenTimestamps
before the first measurement (37 preregistrations and one protocol, 7 deviation notes,
31 stamps; counted 2026-09-02). A stamped file is never edited; deviations go in a
companion `-ECARTS.md`. Each number carries a
provenance label, *measured*, *calculated* or *estimated*, and a link to its
journal. Raw outputs (per-window NLL, MMLU picks) are committed. Speed is
published as a median with range. Ratios are formed round by round inside one
process when the bench prints them per round; otherwise they are quotients
of medians and labelled *calculated*.

## Licence

Code: MIT OR Apache-2.0. The Qwen3 forward pass in `llvq-llm/src/model.rs` is
derived from [candle-transformers](https://github.com/huggingface/candle)
(MIT OR Apache-2.0). The published model is Apache 2.0, inherited from
[Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B). The QTIP kernel used
in one bench is GPL v3 upstream and is not redistributed here
([docs/qtip-provenance.md](docs/qtip-provenance.md)).
