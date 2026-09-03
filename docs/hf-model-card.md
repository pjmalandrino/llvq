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

<!--
  STATUS, reviewed 2026-08-08.

  This card describes `qwen3-4b-llvq.bin`, the ONLINE artifact: f16 embedding,
  1.771 GB, sha256 9db213ef…c84b0. The speed and VRAM figures in the
  Limitations section are measured ON THESE BYTES, CUDA fused path, default
  Planes14 layout: 48.7 tok/s in 2.96 GB (5.89 b/param).
  Wrong since 2026-08-31: the single points in this comment (48.7;
  88.4-88.5) are replaced in the body by the B2 medians (48.3; 87.0), and
  the SERVED CONFIG v1 frozen that day (+ ROT_SHARE=1 FUSE=1) returns
  100.6 [99.9-100.7] in 2.57 GB, fusion measured in band at all three sizes.

  `LLVQ_EMBED=q8` (88.4-88.5 tok/s, 2.60 GB, 5.162 b/param) is a LOAD-TIME
  FLAG applied to the same bytes, not another file, and it is mentioned as
  such. The quality of that configuration, on the other hand, was measured on
  `q4b-e8.llvq` (1.406 GB, pre-baked int8 embedding), which is NOT published:
  bit-identical content verified, different bytes.

  2026-08-17, SETTLED: the published b/param is **5.162**, the `rtbits`
  verdict on the exact bytes (the journal itself writes "LE CHIFFRE 4B q8 À
  PUBLIER EST 5,162"). The **5.15** that stood here is the division of the
  "2.60 GB" shown by `nvidia-smi`, rounded: the real footprint is 2.595 GB.
  Caveat: the two numbers in that pair have two provenances and **cannot be
  derived from one another**. Do not divide the 2.60 to recover the b/param.

  THIS CARD IS A PUBLISHED SURFACE. The repository file is up to date; the
  card ONLINE on the Hub is not, until it is republished. Republishing is a
  separate outbound action and needs its own go. Until then, the repository
  and the published object diverge on this figure.

  USER DECISION PENDING, not to be taken by editing this file: should the HF
  artifact be republished as a pre-baked q8 variant? If so, the headline
  figures of this card change denominator (1.406 GB) and the sha256 above
  becomes wrong. Until that is settled, the card stays written against the
  f16 file actually online.
-->

# Qwen3-4B, LLVQ 2-bit

Qwen3-4B quantized to **2.16 bits per weight** with Leech lattice vector
quantization, from an independent Rust implementation of
[**arXiv:2603.11021**](https://arxiv.org/abs/2603.11021) (van der Ouderaa,
van Baalen, Whatmough, Nagel, Qualcomm AI Research, 2026).

**One file, 1.771 GB, opens with no checkpoint, no cache and no network.**

> **This is a research artifact, not a drop-in model.** Two things to know
> before downloading. It is not GGUF, AWQ or safetensors: it does not load with
> `transformers`, `llama.cpp`, vLLM or TGI, and you need the Rust reader linked
> below. And it loses **14.7 points of MMLU** against its own FP16 baseline
> (the CUDA campaign figure this card quotes throughout; the earlier Metal run
> gave 14.3). Reasoning tasks are hit hardest, some falling to chance. See
> *Quality*.

## Numbers

The file was written, read back, and its 3,633,315,840 projection weights
decode **bit for bit** to the weights that were evaluated.

| | |
|---|---|
| Size | **1.771 GB** against 8.045 GB in FP16 → **×4.54** |
| Rate | **2.1595 bits/weight** over the 3,633,315,840 projection weights |
| Rate, whole model | **3.5213 bits/parameter** (the f16 embedding is 9.7% of it) |
| WikiText-2 perplexity, ctx 4096, f16, **on this file** | **16.9415** (f16 baseline 12.2361, ×1.385) |

The lattice code itself runs at **exactly 2.000 bits/weight over the
3,616,358,400 weights it encodes**: 47 index bits into the Λ₂₄(12) ball plus
one gain bit, packed into 6 bytes per block of 24 weights. The 980,770,752-byte
payload is 7,846,166,016 bits: 7,232,716,800 of lattice code, 542,638,080 of
tail columns kept exact in f32, 70,778,880 of per-row f64 scales (none of the
1,105,920 is representable in f32, so this is the price of the bit-exact decode
proof) and 32,256 of gain centroids. That is **8.5% more than the lattice code
alone**, one payload under two exact denominators: 2.1595 bits/weight over the
3,633,315,840 projection weights, tail included (`bin/seal`, the figure quoted
above), or 2.1696 over the 3,616,358,400 the code actually encodes
(`bin/smoke`, the figure the GitHub README uses).

Composition: 252 quantized linear projections (0.981 GB) + 146 tensors the
quantizer does not touch, at f16 (0.778 GB, almost all of it the tied
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
differ.** Ours is 12.2336 against the paper's 12.41. Normalized as excess
log-likelihood over each side's own baseline, this model is **3.1% worse than
QTIP** on the f32 pair, **2.6% worse** on the f16 pair measured on this file,
and 2.9% worse than the paper's 0-gain-bit configuration, at 8.5% more bits.
It is at QTIP's level, marginally worse. It is not state of the art, and an
earlier version of this card said it landed "just under QTIP", which was the
wrong reading of its own table.

## Quality

Perplexity says nothing about what a model can still *do*. 5-shot MMLU,
2,280 questions of the 14,042-question split at a fixed seed, measured **on this
exact file** through the project's own pipeline:

| | FP16 baseline | this model |
|---|---|---|
| MMLU (micro), Metal / M3 Max | 70.42 ± 1.28 | 56.09 ± 1.36 |
| **MMLU (micro), CUDA / L40S**, the campaign figure | **70.32 ± 1.28** | **55.59 ± 1.35** |

**−14.33 points on Metal, −14.73 on CUDA; 79.7% and 79.1% retained.** The ±
is a stratified standard error covering sampling only, 1 σ, not a 95%
interval, and two of them do not subtract. For a *difference*, the paired test
below is the right instrument.

**The two rows disagree by 0.50 pp on what should be the same file, and we
do not know why.** The baseline moves by only 0.10 pp across the same backend
change, so this is five times the drift of its own control. It is not
verifiable by token fingerprint: the Metal run predates their printing. The
deltas are consistent either way and no conclusion here depends on the choice,
but it is an open provenance debt rather than a rounding difference. The
rest of this card quotes the CUDA row, because that is the one measured
alongside the AWQ arm on the same card with the same fingerprint.

The damage is not uniform. Abstract algebra and professional accounting fall to
10/40, indistinguishable from chance within a ±7 pp per-subject bar; European
history and international law hold at 33/40.
**Two-bit quantization damages reasoning far more than recall**, which is why
the perplexity above looks better than the model behaves. For reference, the
paper reports a 9.5-point drop on the same benchmark; we lose more, and we do
not currently know why. The leading untested candidate is calibration volume:
131,072 tokens against the paper's 6,100 sequences, whose length it does not
state. (Input-only versus Input + Output incoherence rotation looked like the
obvious suspect and is not: in the paper's own Table 9, adding the output stage
moves MMLU by −1.7 to +1.8 points across four configurations, mean ≈ 0.)

## Quantization recipe

**Algorithm 1 of the paper (shape–gain with gain reset) plus an input-side
incoherence rotation.** Angular search capped to the **Λ₂₄(12) ball**, the
union of shells 2..12, i.e. the paper's own `norm(Λ₂₄(12))` codebook, 47 index
bits plus one gain bit, per-row scale in f64, tail columns kept exact.
Calibration on **C4**, out of domain with respect to WikiText-2, as the
paper's calibration is (it uses DCLM-edu), 64 windows of 2048 tokens
(131,072 tokens). 4 h on an M3 Max.

This is **not** the paper's Spherical GPTQ. With a finite gain codebook the
Eq. 17 retraction is a no-op, because the quantizer has already placed the
block on the nearest level's sphere, and the closed-form group-scale
refinement of Algorithm 3 is disabled. An earlier version of this card
described the recipe as using spherical retraction; it does not.

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
without the feature is an error. Nothing else is required, no Hugging Face
cache, no network. Verified with an empty environment:

```bash
env -i HOME=/nonexistent PATH=/usr/bin:/bin ./target/release/run qwen3-4b-llvq.bin cpu 14
```

**Budget the RAM before you download.** `bin/run` decodes every weight into
memory, so the resident model is 8.045 GB of f16 regardless of what the file
costs on disk. Measured peak RSS: **9.79 GB on CPU, 17.41 GB on Metal**. A
16 GB machine will swap on the Metal path. On these two commands the size win
is on disk only.

The CUDA runner is the exception. It keeps the weights encoded and holds the
same model in 2.93 GB of card memory, 2.57 GB in the served configuration v1
(frozen 2026-08-31: `LLVQ_EMBED=q8 LLVQ_ROT_SHARE=1 LLVQ_FUSE=1`). It needs a
Linux host with an NVIDIA card:

```bash
LLVQ_EMBED=q8 LLVQ_ROT_SHARE=1 LLVQ_FUSE=1 \
cargo run --release -p llvq-llm --features cuda --bin fusedrun -- qwen3-4b-llvq.bin 128
```

*(The 2.96/2.60 GB pair quoted here until 2026-09-01 were the pre-B2 single
points; 2.93 and 2.56/2.57 are the B2 host byte counts, and the command above
is the frozen served configuration.)*

## Limitations

* **No speedup and no memory win on `bin/run`, whatever the backend.** The
  portable runner decodes every weight into memory and then does an ordinary
  matvec, so on CPU and on Metal this file costs 8.045 GB resident and buys
  only disk. It does have a KV cache (an earlier version of this card said it
  did not); on an L40S the sealed file generates at 42.7 tok/s through that
  path.
* **There is a fused path, and it is CUDA-only.** A fused
  dequantize + matvec kernel decodes the Leech blocks on the card without ever
  materializing f16 weights. It is wired into the model and driven by
  `bin/fusedrun` (Linux + `--features cuda`). On these exact bytes, L40S,
  128 tokens, default `Planes14` layout: **48.3 tok/s [48.1–48.3] in 2.93 GB
  of card memory against 43.5 [43.4–43.5] in 8.04 GB** for the dense arm.
  That is **×1.11 [1.11–1.11] in speed and ÷2.75 in memory**, 5.89 bits/param
  over the whole model, and the same greedy tokens up to a tie-break at
  token 89. *(This card carried the single points 48.7/×1.12 until
  2026-08-31; the B2 medians land within the inter-invocation dispersion.)*
  Setting `LLVQ_EMBED=q8` quantizes the tied embedding at load and takes the
  same bytes to **87.0 tok/s [86.8–87.0] in 2.56 GB** (5.162 bits/param,
  measured on the exact bytes). The **served configuration v1, frozen
  2026-08-31** (`+ LLVQ_ROT_SHARE=1 LLVQ_FUSE=1`, launch fusion measured in
  band at all three sizes) reaches **100.6 tok/s [99.9–100.7] in 2.57 GB**;
  that ×2.00 is mostly a replacement of an output head that recopies 778 MB
  of vocabulary per token, and **not** the Leech kernel, whose own
  contribution is the ×1.12. The two are never quoted apart. That copy is on
  *our* side: our dense arm calls `Tensor::broadcast_matmul`, whose
  rank-2-rhs path materializes the transposed weight every call. Models built on
  `candle_nn::Linear`, including candle's own, fold the batch dimensions and
  never pay it, so this is a trap in the primitive rather than a defect of
  candle's models ([reported
  upstream](https://github.com/huggingface/candle/issues/3871)). **On Apple
  silicon none of this applies**: `llvq-metal` is a benchmark
  (2.03–2.09× FP16 on the 252 projections, every output row verified against
  an f64 reference) with no runner behind it. Logs:
  `docs/mesures/planes14-fusedrun-2026-08-06.txt`,
  `docs/mesures/phases-2026-08-07.txt`,
  `docs/mesures/k1-metal-2026-08-05.txt`.
* **A 4-bit quantization beats this model on capabilities, and that is now
  measured rather than assumed.** Qwen's own AWQ 4-bit checkpoint, run through
  this project's harness on the same card with the same questions and the same
  token fingerprint, scores **70.04 ± 1.25 on MMLU against this file's 55.59 ±
  1.35**, and ×1.105 perplexity against ×1.384. On a paired, subject-stratified
  bootstrap over the same 2,280 questions, AWQ − f16 is **+0.27 pp, 95% CI
  [−1.63; +2.13]**. The interval contains zero, so the two are
  *indistinguishable under this protocol*, which is not the same as equal.
  Against this file the same test gives **+14.45 pp, 95% CI [+11.60; +17.27]**,
  resolved, and by a wide margin. This artifact wins disk size, and, with the
  fused path and an int8 embedding, card memory: **5.162 bits/param, measured
  by `rtbits` on the actual bytes, against 5.302 computed for AWQ in its own
  engine** (measured against computed: AWQ has never been loaded quantized in
  our harness). It loses quality, by 14 points.
  Logs: `docs/mesures/a4-campagne-2026-08-06.txt`,
  `docs/mesures/mmlupair-4b-8b-2026-08-13.txt`.
* **The format is not portable.** About 1,400 lines of dependency-free Rust
  (`llvq-artifact`) define the container, of which ~425 are the on-disk format
  itself, but *decoding* also needs `llvq-search` and `llvq-core` for the
  Leech index, some 6,500 dependency-free lines in all. A reader in another
  language is tractable, not trivial, and does not exist yet.
* **No commonsense-reasoning or task-specific evaluation, and no error bar on
  *this* perplexity.** A dispersion has since been measured, but on another
  object: three calibration seeds on a 3-block Qwen3-0.6B run give σ ≈ 0.15
  perplexity (≈ 0.7%) around ~20.66. That does not transfer to the 16.9415
  above, a different model, 3 blocks against 36, a different scale, and no σ
  has ever been measured on the full-model number. The older and cruder
  observation also stands: ~7% between two configurations that a test proves
  were the same quantizer, n = 2, cause unresolved.
* **Determinism is uneven.** The Leech encoder is exactly deterministic and
  pinned by a test, but the calibration Hessians accumulate `AᵀA` in f32 on the
  accelerator, so re-running the recipe on another backend does not reproduce
  these weights.
* **Evaluated on 12 windows**, not the full 73. The FP32 baseline lands 1.4%
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
