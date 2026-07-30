# LLVQ in Rust — Leech lattice vector quantization for LLM weights

An independent, from-scratch implementation of
[**Leech Lattice Vector Quantization for Efficient LLM Compression**](https://arxiv.org/abs/2603.11021)
(van der Ouderaa, van Baalen, Whatmough, Nagel — Qualcomm AI Research, 2026),
written to find out whether the method survives contact with industrial use.

The mathematical core — lattice, exact nearest-neighbour search, bijective
indexing, GPTQ — has **no external dependencies**, so it can be read end to
end. Only the model side pulls in `candle`.

> Working notes, non-obvious derivations and the full experimental history are
> in [`CLAUDE.md`](CLAUDE.md) (in French).

## Result

Qwen3-4B, 2 bits/weight, **no fine-tuning**, WikiText-2 perplexity at 4096
context. Calibration on C4 — out of domain with respect to the evaluation,
matching the paper's protocol.

| Method | Wiki ↓ | degradation |
|---|---|---|
| Baseline FP32 (paper: 12.41 — ours: **12.2336**) | — | — |
| Quip#/E8P12 *(paper)* | 21.15 | — |
| QTIP (3INST) *(paper)* | 17.04 | ×1.373 |
| LLVQ, 0 gain bits *(paper)* | 17.05 | ×1.374 |
| LLVQ, 2 gain bits *(paper, best without fine-tuning)* | 15.54 | ×1.252 |
| **This implementation** | **14.9104** | **×1.219** |

### Read this before quoting the number

* **Our rate is 2.1117 bits/weight, not 2.000** — 5.6 % more. About 0.1 bit of
  that is the tail policy: layer widths are not multiples of 24 (Qwen3-4B:
  2560 = 24·106 + 16) and we leave the remainder at full precision. The paper
  says only that the last block "may be smaller" and never says what
  quantizes it, so part of this gap is an unspecified detail on their side
  rather than an advantage on ours.
* **Evaluated on 12 windows, not the full 73.** Our FP32 baseline lands 1.4 %
  under the paper's, so our window subset is slightly easier.
* Two differences work **against** us: ~131 k calibration tokens against
  their 6 100 sequences (~100× less), and input-only incoherence rotation
  where they use *Input + Output*.

The defensible claim is that this reproduces the method and lands at the level
of the paper's best no-fine-tuning configuration. Not that it beats it.

## What is here

| Crate | Contents | Dependencies |
|---|---|---|
| `llvq-core` | Extended Golay code, Λ₂₄, shells | none, `forbid(unsafe)` |
| `llvq-search` | Exact NN search over shells m ≤ 13, bijective 48-bit index | none |
| `llvq-quant` | Spherical GPTQ (Alg. 1 & 3), dense linear algebra, incoherence rotation | none (`faer` optional) |
| `llvq-llm` | Observable Qwen3 forward, Hessians, perplexity, generation probe | `candle` |
| `llvq-bench` | Rate–distortion on a Gaussian source, encoder throughput | none |

The encoder runs at **1 469 blocks/s/core** (24 weights per block) after a
5.5× optimization pass; quantizing Qwen3-4B takes ~3.5 h on an M3 Max.

## What is *not* here

* **No 2-bit artifact.** The quantizer produces reconstructions; the 48-bit
  index format exists and is round-trip tested, but is not yet wired to a
  writer. The 1.74 GB figure is arithmetic, not a file.
* **No inference speedup.** Zero. The fused dequantize+matvec kernel is
  unwritten. The paper's own kernel handles a single shell "for simplicity"
  and is slower than QTIP; the multi-shell kernel the 2-bit regime needs does
  not exist anywhere.
* **No MMLU, no CSR**, and no domain-specific benchmark.

## An open question for the authors

Appendix G compares single Leech shells against unions and concludes that the
union gives better angular uniformity per bit — *"we therefore adopt this
approach and recommend doing the same"*.

Measuring rate–distortion retention instead, on an i.i.d. Gaussian source
(20 000 blocks, fixed seed, gain centroids fitted on a held-out train split):

| Code | bits/dim | MSE | Retention | Classes |
|---|---|---|---|---|
| union `norm(Λ₂₄(12))` + 1 gain bit *(paper's best, Table 8)* | 2.0000 | 0.078 | 92.14 % | 383 |
| **shell 12 only + 1 gain bit** | **1.9584** | 0.0805 | **92.81 %** | **79** |
| shell 13 only + 1 gain bit | 2.0113 | 0.0751 | 92.83 % | 82 |

A single shell appears to win on rate *and* retention, with 4.8× fewer classes
and — the point Appendix G itself raises — a **constant norm**, which removes
the rescaling of intermediate dot products in a fused kernel.

This is not a flat contradiction: the paper measures angular distance to the
nearest neighbour on a radially uniform source, which is a different quantity
from MSE retention after a gain quantizer. And it is one source, one harness,
one seed. **It has not been verified on real weights after the GPTQ loop**,
which is where it would actually matter. We would value being told what we are
missing.

## Reproducing

```bash
cargo test --release -- --include-ignored
```

```bash
cargo run --release -p llvq-bench --bin llvq-bench
```

```bash
LLVQ_MODEL=Qwen/Qwen3-4B LLVQ_CALIB=c4 cargo run --release -p llvq-llm --features metal,fast-linalg --bin smoke -- 64 2048 12 4096 metal nogs leech1 999 rot out.safetensors
```

## Notes on method

Every gate is held by tests designed to be *lethal*, and several were rewritten
after mutation testing showed they passed on broken code. Four defects found
that way are documented in `CLAUDE.md`, including one that produced a
perplexity of 1 327 613 and one that made a reported bit-rate wrong by 0.62
bits per weight. The common pattern: **an assertion that never exercises the
parameter it is supposed to cover.**

Reading notes on the paper — Algorithms 1 and 3 transcribed, two notational
ambiguities in Algorithm 3 resolved with justification, Tables 3/6/7/8/9 — are
in [`docs/llvq-paper-notes.md`](docs/llvq-paper-notes.md).

## Licence

MIT OR Apache-2.0. The Qwen3 forward pass in `llvq-llm/src/model.rs` is derived
from the architecture as implemented in
[`candle-transformers`](https://github.com/huggingface/candle) (MIT OR
Apache-2.0), restructured to make linear-layer inputs observable.
