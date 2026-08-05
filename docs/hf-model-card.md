---
license: apache-2.0
base_model: Qwen/Qwen3-4B
base_model_relation: quantized
language:
  - en
library_name: llvq
inference: false
tags:
  - qwen3
  - quantization
  - 2-bit
  - leech-lattice
  - vector-quantization
  - llvq
---

# Qwen3-4B — LLVQ 2-bit

Qwen3-4B quantized to **2.16 bits per weight** with Leech lattice vector
quantization, from an independent Rust implementation of
[**arXiv:2603.11021**](https://arxiv.org/abs/2603.11021) (van der Ouderaa,
van Baalen, Whatmough, Nagel — Qualcomm AI Research, 2026).

**One file, 1.771 GB, opens with no checkpoint, no cache and no network.**

> ⚠️ **This is a research artifact, not a drop-in model.** Two things to know
> before downloading. It is not GGUF, AWQ or safetensors: it does not load with
> `transformers`, `llama.cpp`, vLLM or TGI, and you need the Rust reader linked
> below. And it loses **14.3 points of MMLU** against its own FP16 baseline —
> reasoning tasks are hit hardest, some falling to chance. See *Quality*.

## Numbers

The file was written, read back, and its 3 633 315 840 projection weights
decode **bit for bit** to the weights that were evaluated.

| | |
|---|---|
| Size | **1.771 GB** against 8.045 GB in FP16 → **×4.54** |
| Rate | **2.1595 bits/weight** over the 3 633 315 840 projection weights |
| Rate, whole model | **3.5213 bits/parameter** (the f16 embedding is 9.7 % of it) |
| WikiText-2 perplexity, ctx 4096, f16, **on this file** | **16.9415** (f16 baseline 12.2361, ×1.385) |

The lattice code itself runs at **exactly 2.000 bits/weight over the
3 616 358 400 weights it encodes** — 47 index bits into the Λ₂₄(12) ball plus
one gain bit, packed into 6 bytes per block of 24 weights. The 980 770 752-byte
payload is 7 846 166 016 bits: 7 232 716 800 of lattice code, 542 638 080 of
tail columns kept exact in f32, 70 778 880 of per-row f64 scales (none of the
1 105 920 is representable in f32, so this is the price of the bit-exact decode
proof) and 32 256 of gain centroids. That is **8.5 % more than the lattice code
alone** — one payload under two exact denominators: 2.1595 bits/weight over the
3 633 315 840 projection weights, tail included (`bin/seal`, the figure quoted
above), or 2.1696 over the 3 616 358 400 the code actually encodes
(`bin/smoke`, the figure the GitHub README uses).

Composition: 252 quantized linear projections (0.981 GB) + 146 tensors the
quantizer does not touch, at f16 (0.778 GB — almost all of it the tied
embedding) + config and tokenizer (0.011 GB).

### Against published 2-bit methods

All figures without fine-tuning, no error bars on either side.

| Method | Wiki ↓ | bits/weight |
|---|---|---|
| Quip#/E8P12 | 21.15 | 2.000 |
| QTIP (3INST) | 17.04 | 2.000 |
| LLVQ, 0 gain bits *(paper)* | 17.05 | 2.000 |
| **This model** | **16.9617** *(f32, in-memory)* | 2.1595 |
| LLVQ, 2 gain bits *(paper's best)* | 15.54 | 2.000 |

**Raw perplexities across implementations are not comparable when the baselines
differ.** Ours is 12.2336 against the paper's 12.41. Normalised as excess
log-likelihood over each side's own baseline, this model is **3.1 % worse than
QTIP** on the f32 pair, **2.6 % worse** on the f16 pair measured on this file,
and 2.9 % worse than the paper's 0-gain-bit configuration — at 8.5 % more bits.
It is at QTIP's level, marginally worse. It is not state of the art, and an
earlier version of this card said it landed "just under QTIP", which was the
wrong reading of its own table.

## Quality

Perplexity says nothing about what a model can still *do*. 5-shot MMLU,
2 280 questions of the 14 042-question split at a fixed seed, measured **on this
exact file** through the project's own pipeline:

| | FP16 baseline | this model |
|---|---|---|
| MMLU (micro, population-weighted) | **70.42 ± 1.28** | **56.09 ± 1.36** |

**−14.33 points, 79.7 % retained.** The ± is a stratified standard error
covering sampling only — 1 σ, not a 95 % interval.

The damage is not uniform. Abstract algebra and professional accounting fall to
10/40 — indistinguishable from chance within a ±7 pp per-subject bar; European
history and international law hold at 33/40.
**Two-bit quantization damages reasoning far more than recall**, which is why
the perplexity above looks better than the model behaves. For reference, the
paper reports a 9.5-point drop on the same benchmark; we lose more, and we do
not currently know why. The leading untested candidate is calibration volume:
131 072 tokens against the paper's 6 100 sequences, whose length it does not
state. (Input-only versus Input + Output incoherence rotation looked like the
obvious suspect and is not: in the paper's own Table 9, adding the output stage
moves MMLU by −1.7 to +1.8 points across four configurations, mean ≈ 0.)

## Quantization recipe

**Algorithm 1 of the paper (shape–gain with gain reset) plus an input-side
incoherence rotation.** Angular search capped to the **Λ₂₄(12) ball** — the
union of shells 2..12, i.e. the paper's own `norm(Λ₂₄(12))` codebook — 47 index
bits plus one gain bit, per-row scale in f64, tail columns kept exact.
Calibration on **C4** — out of domain with respect to WikiText-2, as the
paper's calibration is (it uses DCLM-edu) — 64 windows of 2048 tokens
(131 072 tokens). 4 h on an M3 Max.

This is **not** the paper's Spherical GPTQ. With a finite gain codebook the
Eq. 17 retraction is a no-op — the quantizer has already placed the block on
the nearest level's sphere — and the closed-form group-scale refinement of
Algorithm 3 is disabled. An earlier version of this card described the recipe
as using spherical retraction; it does not.

## How to run it

```bash
git clone https://github.com/pjmalandrino/llvq && cd llvq
```
```bash
hf download Pier-Jean/Qwen3-4B-LLVQ-2bit qwen3-4b-llvq.bin --local-dir .
```
```bash
# Apple Silicon
cargo run --release -p llvq-llm --features metal --bin run -- qwen3-4b-llvq.bin metal 24
```
```bash
# CPU, anywhere
cargo run --release -p llvq-llm --bin run -- qwen3-4b-llvq.bin cpu 24
```

The cargo feature and the third argument are separate: asking for `metal`
without the feature is an error. Nothing else is required — no Hugging Face
cache, no network. Verified with an empty environment:

```bash
env -i HOME=/nonexistent PATH=/usr/bin:/bin ./target/release/run qwen3-4b-llvq.bin cpu 14
```

**Budget the RAM before you download.** The reader decodes every weight into
memory, so the resident model is 8.045 GB of f16 regardless of what the file
costs on disk. Measured peak RSS: **9.79 GB on CPU, 17.41 GB on Metal**. A
16 GB machine will swap on the Metal path. The size win here is on disk.

## Limitations

* **No inference speedup, and generation is slow.** The reader decodes weights
  into memory and then does an ordinary matvec, and the runner has **no KV
  cache** — it re-runs the whole prefix at every step, which is quadratic.
  Measured: **2.2 to 7.6 tok/s**, decreasing with length. A fused
  dequantize+matvec kernel **does exist** in this
  project (`llvq-metal`, `bin/thesis`: **2.03–2.09× FP16 over three invocations** over
  all 252 projections of this file, at 5.510 bits/weight in RAM; the ratio is
  formed **round by round** and reported as the median and range over the 5
  kept rounds — it is not the quotient of two best-of times; every output row
  verified against an f64 reference; log in
  `docs/mesures/k1-metal-2026-08-05.txt`) but it is
  **not wired into the runner**, so this file gains nothing from it today.
* **A plain 4-bit quantization beats this model on most axes.** On the same
  machine, `mlx_lm` q4 group-64 is 2.263 GB on disk, generates far faster, and
  its quality has never been measured against this one. This artifact wins on
  disk size — ×1.28, i.e. 22 % smaller — and that is the claim.
* **The format is not portable.** About 1 400 lines of dependency-free Rust
  (`llvq-artifact`) define the container, of which ~425 are the on-disk format
  itself — but *decoding* also needs `llvq-search` and `llvq-core` for the
  Leech index, some 6 500 dependency-free lines in all. A reader in another
  language is tractable, not trivial, and does not exist yet.
* **No commonsense-reasoning or task-specific evaluation**, and no error bar on
  perplexity: the only dispersion this pipeline has shown is ~7 % between two
  configurations that a test proves were the same quantizer — one observation,
  n = 2, cause unresolved, no σ.
* **Determinism is uneven.** The Leech encoder is exactly deterministic and
  pinned by a test, but the calibration Hessians accumulate `AᵀA` in f32 on the
  accelerator, so re-running the recipe on another backend does not reproduce
  these weights.
* **Evaluated on 12 windows**, not the full 73. The FP32 baseline lands 1.4 %
  under the paper's, so this window subset is slightly easier.
* **The published quantization command reproduces the method, not these
  bytes.** The calibration shard and the container format both moved after this
  file was written.

## License and attribution

Apache 2.0, inherited from
[Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B). The `LICENSE` file in
this repository is Qwen's, carried over unchanged.

**Modification made to the original work:** the 252 linear projection weight
tensors of every transformer block have been replaced by Leech lattice codes
(index + gain per 24 weights, with a per-row scale) and are reconstructed at
load time. All other tensors are the originals, converted to f16. No training,
no fine-tuning, no architectural change.

The quantization implementation is at
[github.com/pjmalandrino/llvq](https://github.com/pjmalandrino/llvq)
(MIT OR Apache-2.0).
