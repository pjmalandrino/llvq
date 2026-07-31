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

Qwen3-4B, **no fine-tuning**, WikiText-2 perplexity at 4096 context.
Calibration on C4 — out of domain with respect to the evaluation, matching the
paper's protocol.

Every rate below is what the method *stores*. Ours is **weighed on a file**,
not computed: 981 MB on disk, and the file decodes back to the evaluated
weights bit for bit (3 633 315 840 of them).

| Method | Wiki ↓ | degradation | bits/weight |
|---|---|---|---|
| Baseline FP32 (paper: 12.41 — ours: **12.2336**) | — | — | 32 |
| Quip#/E8P12 *(paper)* | 21.15 | — | 2.000 |
| QTIP (3INST) *(paper)* | 17.04 | ×1.373 | 2.000 |
| LLVQ, 0 gain bits *(paper)* | 17.05 | ×1.374 | 2.000 |
| **This implementation** | **16.9617** | **×1.386** | **2.1696 weighed** |
| LLVQ, 2 gain bits *(paper, best without fine-tuning)* | 15.54 | ×1.252 | 2.000 |

Whole model, embedding included: **1.76 GB against 8.04 GB in FP16, ×4.57.**
The tied embedding stays at f16 and is 9.7 % of the model — which is why the
end-to-end ratio is ×4.57 and not the ×7.4 the linear layers alone suggest.

### Read this before quoting the number

* **We land just under QTIP, at 8.5 % more bits.** 16.96 against 17.04, but at
  2.1696 bits/weight rather than 2.000. That is a narrower claim than it looks.
* **We are 9 % above the paper's best configuration**, which reaches 15.54 at
  a true 2.000.
* **Where the extra 0.17 bit is**, all of it reducible: f32 tail columns
  (+0.075), f64 row scales (+0.015), and the tail policy itself (~0.05) —
  layer widths are not multiples of 24 and we leave the remainder exact. The
  paper says only that the last block "may be smaller" and never says what
  quantizes it.
* **Evaluated on 12 windows, not the full 73.** Our FP32 baseline lands 1.4 %
  under the paper's, so our window subset is slightly easier.
* Two differences work **against** us: ~131 k calibration tokens against their
  6 100 sequences (~100× less), and input-only incoherence rotation where they
  use *Input + Output*.

The defensible claim is that this reproduces the method and lands at QTIP's
level. Not that it beats the paper.

> **An earlier version of this file claimed 14.9104 at 2.1117 bits/weight.**
> The perplexity was real; the rate was not. The spherical retraction was
> cancelling the gain code, so the stored magnitude was a free float per block
> — 16 bits nothing charged for, and a true rate of 2.73. Two further
> accounting defects are described in
> [`docs/retraction-et-gain.md`](docs/retraction-et-gain.md). All three were
> found by trying to *write the file* rather than compute its size.

## What is here

| Crate | Contents | Dependencies |
|---|---|---|
| `llvq-core` | Extended Golay code, Λ₂₄, shells | none, `forbid(unsafe)` |
| `llvq-search` | Exact NN search over shells m ≤ 13, bijective 48-bit index, bit packing | none |
| `llvq-quant` | Spherical GPTQ (Alg. 1 & 3), dense linear algebra, incoherence rotation | none (`faer` optional) |
| `llvq-artifact` | The `.llvq` format: writer, reader, decoder | **none** |
| `llvq-llm` | Observable Qwen3 forward, Hessians, perplexity, generation | `candle` |
| `llvq-bench` | Rate–distortion, encoder throughput, decode cost | none |

`llvq-artifact` has no external dependencies **on purpose**: reading a
quantized model should not require a tensor runtime. Its whole dependency tree
is the three crates above it — against 690 transitive crates for `llvq-llm`.
Someone who wants to check what a `.llvq` contains, port the reader, or audit
the decoder before trusting a model can read it end to end.

The encoder runs at **1 469 blocks/s/core** (24 weights per block) after a
5.5× optimization pass; quantizing Qwen3-4B takes ~3.5 h on an M3 Max.

## What is *not* here

* **No inference speedup. Zero.** The fused dequantize+matvec kernel is
  unwritten, and the archive format is not usable in one: decoding a bijective
  index costs **827 ns/block** against 4.5 ns for a Golay codeword rebuilt by
  XOR — a factor of **183×**, measured (`cargo run --release -p llvq-bench
  --bin decbench`). A kernel needs a different, transcoded format. The paper is
  in the same position: its own CUDA kernel handles a single shell "for
  simplicity", is slower than QTIP, and its authors call low-level optimization
  "largely orthogonal" to their contribution. **The kernel the 2-bit regime
  needs does not exist anywhere.**
* **The artifact is not self-contained.** It carries the 252 linear
  projections; embeddings, RMSNorm weights and the config still come from the
  checkpoint.
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

Quantize Qwen3-4B and write the compressed artifact (~3.5 h on an M3 Max). The
run verifies the file by decoding it and demanding the evaluated weights back,
bit for bit:

```bash
LLVQ_MODEL=Qwen/Qwen3-4B LLVQ_CALIB=c4 LLVQ_ARTIFACT=q4b.llvq cargo run --release -p llvq-llm --features metal,fast-linalg --bin smoke -- 64 2048 12 4096 metal nogs leech1c12 999 rot
```

Load the artifact and generate from it:

```bash
LLVQ_MODEL=Qwen/Qwen3-4B cargo run --release -p llvq-llm --features metal --bin run -- q4b.llvq metal 24
```

What decoding costs, which is what gates the fused kernel:

```bash
cargo run --release -p llvq-bench --bin decbench
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
