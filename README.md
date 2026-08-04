# LLVQ in Rust — Leech lattice vector quantization for LLM weights

An independent, from-scratch implementation of
[**Leech Lattice Vector Quantization for Efficient LLM Compression**](https://arxiv.org/abs/2603.11021)
(van der Ouderaa, van Baalen, Whatmough, Nagel — Qualcomm AI Research, 2026),
written to find out whether the method survives contact with industrial use.

The mathematical core — lattice, exact nearest-neighbour search, bijective
indexing, GPTQ — has **no external dependencies**, so it can be read end to
end. Only the model side pulls in `candle`.

> **Want to just run it?** → [`LAUNCH_ME.md`](LAUNCH_ME.md). The model is at
> [Pier-Jean/Qwen3-4B-LLVQ-2bit](https://huggingface.co/Pier-Jean/Qwen3-4B-LLVQ-2bit).
>
> ⚠️ **It is not GGUF, AWQ or safetensors, and the size win is on disk only.**
> `transformers`, `llama.cpp`, vLLM and TGI do not read this file — the only
> reader that exists is `llvq-artifact`, in this repository. The runner decodes
> every weight into memory, so it needs ~10 GB of RAM on CPU and ~17 GB on
> Metal. And the speed chapter below is Apple-only: `llvq-metal` is gated on
> macOS and there is no CUDA kernel.
>
> Every number below is tabulated with its provenance — the command that
> produces it, the object it scores, the dtype, the protocol, and whether it
> was measured, computed or assumed — in [`docs/fiche-4b.md`](docs/fiche-4b.md).
> Working notes and the full experimental history are in
> [`CLAUDE.md`](CLAUDE.md) (in French). ⚠️ That one is a lab notebook, not a
> specification: it still carries superseded figures this file retracts — among
> them a ×4.63 compression ratio computed on a 1.74 GB artifact that was never
> written. Where they disagree, this file and `docs/fiche-4b.md` win.

## Result

Qwen3-4B, **no fine-tuning**, WikiText-2 perplexity at 4096 context, 12
non-overlapping windows. Calibration on C4 — out of domain with respect to the
evaluation, as the paper's is — it calibrates on DCLM-edu, we use C4.

| Method | Wiki ↓ | degradation | bits/weight |
|---|---|---|---|
| Baseline FP32 (paper: 12.41 — ours: **12.2336**) | — | — | 32 |
| Quip#/E8P12 *(paper)* | 21.15 | — | 2.000 |
| QTIP (3INST) *(paper)* | 17.04 | ×1.373 | 2.000 |
| LLVQ, 0 gain bits *(paper)* | 17.05 | ×1.374 | 2.000 |
| **This implementation** | **16.9617** | **×1.3865** | **2.1696 weighed** |
| LLVQ, 2 gain bits *(paper, best without fine-tuning)* | 15.54 | ×1.252 | 2.000 |

**Raw perplexities are not comparable across implementations with different
baselines.** Normalised as excess log-likelihood over each implementation's own
baseline — the only cross-paper comparison that holds:

| | Δ nats/token | vs QTIP |
|---|---|---|
| **this implementation** | ln(16.9617) − ln(12.2336) = **0.3268** | **+3.1 %** |
| QTIP (17.04 / 12.41) | 0.3171 | — |
| LLVQ 0 gain bits (17.05 / 12.41) | 0.3176 | +0.2 % |

**We land at QTIP's level, marginally *worse* than it — 3.1 % more excess
log-likelihood — and 2.9 % worse than the paper's own 0-gain-bit configuration,
before paying for 8.5 % more bits.** An earlier version of this file said "just
under QTIP", which contradicted its own table.

The row above is measured in f32 on the model in memory, which is how the
paper-facing comparison was run. **Scored instead on the published file itself,
in f16, both arms print the same token fingerprint:**

```
qwen3-4b-llvq.bin [LLVQ 2-bit, sealed] — wikitext2, ctx 4096, 12 windows, dtype f16, tokens 3f1baca9033bf251
ppl = 16.9415
Qwen/Qwen3-4B [baseline]               — wikitext2, ctx 4096, 12 windows, dtype f16, tokens 3f1baca9033bf251
ppl = 12.2361
```

×1.3846, and 2.6 % *worse* than QTIP in excess log-likelihood. Same fingerprint
on both sides means the same 49 140 scored tokens, so the ratio means
something. This is the pair to quote: it is the one a reader can reproduce on
the bytes they download, with the two commands under *Reproducing*.

### The rate is weighed on a file, and it is exactly 2.000 in the code

981 MB of indices on disk, decoding back to the evaluated weights bit for bit
(3 633 315 840 of them). Two denominators circulate, both arithmetically exact:

| over | bits/weight | printed by |
|---|---|---|
| 3 616 358 400 quantized weights (tail excluded) | **2.1696** | `bin/smoke` |
| 3 633 315 840 projection weights (tail included) | **2.1595** | `bin/seal`, the model card |

The payload includes the tail, so 2.1595 is the homogeneous ratio; 2.1696 is
the conservative one and is what the comparison above uses. Over the **whole
model**, embedding included: **3.5213 bits/parameter**.

Where the 0.17 goes (denominator 3 616 358 400), closing to the 7th digit:

| | bits/weight |
|---|---|
| **lattice code — 150 681 600 blocks × 48 bits** | **2.000000** exactly |
| tail columns kept exact, stored **f32** | 0.150051 |
| row scales, stored **f64** | 0.019572 |
| gain centroids | 0.000009 |
| **total** | **2.169632** (+8.48 %) |
| *if tail and scales were f16* | *2.0799 (+4.0 %)* |

The lattice code runs at exactly 2.000 bits/weight — 47 index bits into the
Λ₂₄(12) ball plus 1 gain bit, packed into 6 bytes with no padding. The excess
is serialization, plus a tail policy the paper never specifies. Note that the
f64 row scales are **not** reducible for free: **none of the 1 105 920 is
representable in f32**, so f16 scales would forfeit the bit-exact decode proof.
The f32 tail is the real 0.075-bit reserve.

**The whole model is one file: 1.771 GB against 8.045 GB in FP16, ×4.54.**
It carries the quantized projections, every tensor the quantizer did not touch
(the tied embedding, at f16, is 9.7 % of the model), the config and the
tokenizer. It opens with no checkpoint, no Hugging Face cache and no network:

```bash
cargo run --release -p llvq-llm --features metal --bin run -- qwen3-4b-llvq.bin metal 24
```

**But not in 1.771 GB of RAM.** The reader decodes every weight into memory, so
the resident model is 8.045 GB of f16 whatever the file costs on disk: measured
peak RSS is **9.79 GB on CPU and 17.41 GB on Metal**. A 16 GB machine will swap
on the Metal path. Drop `--features metal` *and* change the third argument to
`cpu` for the portable path — the feature and the argument are separate, and
asking for `metal` without the feature is an error.

### MMLU — what the perplexity was hiding

Perplexity measures average surprise on running text. It says nothing about
what a model can still *do*. Measured here (5-shot Hendrycks, 2 280 questions
of the 14 042-question split sampled at a fixed seed, **through this project's
own pipeline on the shipped file** — not a dequantized checkpoint in someone
else's engine):

| | ours (micro, 1 gain bit) | paper (0 gain bits, their best MMLU) |
|---|---|---|
| FP16 baseline | **70.42 ± 1.28** | 70.2 |
| LLVQ 2-bit | **56.09 ± 1.36** | 60.7 |
| **drop** | **−14.33 pp** *(79.7 % retained)* | −9.5 pp *(86.5 % retained)* |

**Micro** is the paper's aggregation: a stratified estimator reweighting the 57
sampled subject rates by each subject's real population. The ± is a stratified
standard error with finite-population correction — **1 σ, not a 95 % interval**
— and it covers sampling only, not model, prompt or seed variance. The
unweighted macro average of the same run gives 72.85 and 57.59, a −15.26 pp
drop; those two numbers are **not comparable to the paper** and an earlier
version of this file published them as if they were.

**Our baseline reproduces the paper's to within +0.22 pp (0.17 σ).** That
validates the harness, and it means the quantized arm's shortfall cannot be
blamed on the protocol. We do not currently know what causes the remaining
4.8 pp. Candidates we have **not** measured: calibration volume (131 072
tokens against their 6 100 sequences, whose length the paper does not state),
the magnitude path (see *Naming*, below), and our 1-gain-bit configuration,
which has no counterpart in the paper's Table 6 — it reports 0 and 2 gain bits,
1.4 pp apart.

One candidate we can rule out from the paper itself: we use input-only
incoherence rotation where the paper's best configurations use *Input +
Output*, and that looked like the obvious suspect. It is not. In Table 9, going
from `Input` to `Input + Output` moves MMLU by −1.7, +1.8, +1.2 and −1.1 points
across the four LLVQ families — **mean ≈ 0**. The large jump in that ablation is
*no rotation* → *any rotation*, and we already have the input stage.

The per-subject profile makes the mechanism visible: abstract algebra and
professional accounting fall to 10/40 — chance, within a ±7 pp per-subject bar
— while European history and international law hold at 33/40. **Two-bit
quantization damages reasoning far more than recall** — and recall is what a
perplexity corpus mostly measures. A related but distinct decoupling is
reported in [arXiv:2607.08734](https://arxiv.org/abs/2607.08734), where
perplexity *and* accuracy stay flat while individual answers change; here
accuracy itself moves, which that work does not claim.

```bash
cargo run --release -p llvq-llm --features metal --bin mmlu -- qwen3-4b-llvq.bin metal 40
```

### Naming: this is not Spherical GPTQ

The shipped recipe is **Algorithm 1 (shape–gain with gain reset) plus an
input-side incoherence rotation**. It is not the Spherical GPTQ of Algorithm 3,
and earlier versions of this file said it was.

With a finite gain codebook, `quantize` has already placed the block on the
nearest level's sphere, so the Eq. 17 retraction has nothing left to do:
`retraction_target()` returns `None` and the rescale in `gptq.rs` is skipped
entirely. The second stage, the closed-form group-scale refinement, is
disabled in the published run. We have therefore **never exercised Eq. 17 as
written**, and we do not know whether that is a correct reading of the paper or
our own error.

Note also that the configuration line printed in the run logs
(`0 gain bits, spherical retraction, …`) is a **hard-coded string literal**. It
reflects neither the real gain-bit count (1) nor the state of the retraction.
Only the result line is trustworthy.

## Against 4-bit — the comparison that matters

Nobody deploys FP16 locally. The honest reference is an ordinary 4-bit
quantization, produced on the same machine from the same checkpoint
(`mlx_lm.convert -q --q-bits 4 --q-group-size 64`):

| | **LLVQ 2-bit** *(this model)* | **MLX q4** *(group 64)* | **FP16 baseline** |
|---|---|---|---|
| **Cold storage** | **1.771 GB** — 3.52 bits/param | **2.263 GB** — 4.50 bits/param | 8.045 GB — 16 bits/param *(computed)* |
| **Peak memory while generating** | **9.79 GB** (CPU) · **17.41 GB** (Metal) | *2.39 GB — MLX allocator peak, not an RSS; no trace kept* | *never measured* |
| **Generation throughput** | **2.2 – 7.6 tok/s** *(no KV cache — a floor)* | *129.8 tok/s — no trace kept* | *never measured* |
| **WikiText-2 perplexity** *(f16, ctx 4096, 12 windows)* | **16.9415** | *never measured* | **12.2361** |
| **MMLU** *(5-shot, micro, 2 280 q)* | **56.09 ± 1.36** | *never measured* | **70.42 ± 1.28** |

**Bold** = measured here, and a command under *Reproducing* reproduces it.
*Italic* = computed, unverified or absent, with the reason stated.

**Only one row is measured across all three columns — cold storage — and it is
the only one we win.** Four things to read with the table:

* **The two quality rows are strictly iso-conditions between LLVQ and FP16**:
  the same token fingerprint `3f1baca9033bf251` for perplexity, the same log,
  session and seed for MMLU. **The q4 column is empty**, and that is the
  comparison that would decide whether any of this is worth it.
* **The three memory cells are not the same quantity.** Ours is peak RSS; MLX's
  2.39 GB is `mx.get_peak_memory()`, an allocator high-water mark. They cannot
  be subtracted from one another.
* **Our throughput is a floor, not a regime.** `bin/run` has no KV cache and
  re-runs the whole prefix at every step, which is quadratic and documented as
  such in the code. The fused kernel below reaches 2.06–2.08× FP16 on the
  projections, but it is not wired in, so it does not belong in this table.
* MLX also quantizes the embedding, which we leave at f16. On projections alone
  the rate gap is 2.1595 against 4.5000, **×2.08**; the ×1.28 on disk is that
  gap diluted by the embedding.

**The empty column is the honest headline.** No perplexity and no MMLU has ever
been run on the 4-bit arm, here or anywhere else in this repo. Any claim that
4-bit loses "1-2 %" is unsupported, and we do not make it — but neither can we
claim to beat it on anything except disk.

The structural niche for 2-bit is the memory window where 4-bit does not fit
and we do: 12 % to 21 % wide, depending on the runtime layout. Whether that
window is worth anything at 70B is **untested** — no 70B has ever been
quantized here, and the KV cache (320 KiB/token in f16) is not budgeted in any
of our projections. Full analysis: [`docs/face-au-4-bits.md`](docs/face-au-4-bits.md).

## Read this before quoting the number

* **We are at QTIP's level, marginally worse.** 16.96 against 17.04 looks like
  a win and is not: our baseline is lower, and normalised on each side's own
  baseline we are 3.1 % worse.
* **The 0.08 perplexity margin is not defensible.** The only dispersion this
  pipeline has ever shown is ~7 % between two configurations that a test proves
  were the same quantizer. One observation, n = 2, cause unresolved, **no σ**.
  Any margin below that is noise.
* **We are 9 % above the paper's best configuration**, which reaches 15.54 at a
  true 2.000.
* **Evaluated on 12 windows, not the full 73.** Our FP32 baseline lands 1.4 %
  under the paper's, so our window subset is slightly easier.
* Two differences work **against** us: 131 072 calibration tokens against their
  6 100 sequences, and input-only incoherence rotation where they use
  *Input + Output*.

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
| `llvq-quant` | GPTQ (Alg. 1), dense linear algebra, incoherence rotation | none (`faer` optional) |
| `llvq-artifact` | The `.llvq` format: writer, reader, decoder | **none** |
| `llvq-llm` | Observable Qwen3 forward, Hessians, perplexity, generation | `candle` |
| `llvq-bench` | Rate–distortion, encoder throughput, decode cost | none |

`llvq-artifact` has no external dependencies **on purpose**: reading a
quantized model should not require a tensor runtime. Its whole dependency tree
is the three crates above it — against 261 distinct packages for `llvq-llm`
(291 with `metal,fast-linalg`). Someone who wants to check what a `.llvq`
contains, port the reader, or audit the decoder before trusting a model can
read it end to end.

The encoder runs at **1 469 blocks/s/core** (24 weights per block, 680 µs) after
a 5.2× optimization pass. Quantizing Qwen3-4B took **4.01 h** (14 447 s) on an
M3 Max with 16 threads.

## Speed: the fused kernel

The archive format is optimal in bits and unusable in a kernel — decoding a
bijective index costs **8.27 ns/block on a GPU**, 106× the floor. So the file
stays as it is, and a **transcode at load time** produces a kernel-shaped
layout. `Slot32` puts every field at a fixed offset —
`[class 9][gain 1][sign mask 24][slot masks 24×(L−1)]` — which turns the decode
into a fixed 24-slot loop with no divergence and no serial state.

Measured over **all 252 projection matrices** of the published model, one token,
one command buffer per format, cold by construction (2.50 GB and 7.27 GB of
distinct weights — three orders of magnitude past any system cache, so nothing
can be re-read), every one of the 1 105 920 output rows verified against an f64
CPU reference *before* timing:

| | ms/token | GB read | vs FP16 |
|---|---|---|---|
| FP16 (half4) | 21.7 – 22.7 | 7.27 | 1.00× |
| **LLVQ fused (Slot32)** | **10.5 – 11.0** | 2.50 | **2.06 – 2.08×** |

Ranges, not point values: two runs on the same machine differ by 4–5 % from
thermal drift, while the **ratio** moves by 0.8 %. What is reproducible to the
digit is the ratio, the 1 105 920 verified rows and the worst observed error,
**3.4·10⁻⁸ · Σ|wᵢxᵢ|**.

```bash
cargo run --release -p llvq-metal --bin thesis -- qwen3-4b-llvq.bin
```

**What the 2.07× is, and is not.** It times 252 fused matvecs on an M3 Max at
batch 1 in unified memory — not comparable to the paper's Table 7 on different
hardware, where a single-shell (M = 3) kernel reaches 1.36–1.48×. No fused
**multi-shell Leech** decoder appears to have been published before this one;
fused 2-bit kernels in general certainly have (QTIP, QuIP#, AQLM). That is a
negative claim, and it rests on one survey
([`docs/inference-cost-reduction-2026.md`](docs/inference-cost-reduction-2026.md))
carried out from a network that blocked arXiv and Hugging Face, and which says
so itself — **we would be glad to be pointed at prior art.** The comparison is
also against an FP16 kernel written by the same author, and has never been run
against MPS, MLX or Accelerate.

The FP16 arm holds `f16(w)` where `w` is the f64 reconstruction of the LLVQ
blocks, in the rotated basis — the same values as the quantized arm, to
rounding. Same shapes, same bytes, so the timing ratio holds; but this is a
**cost** baseline, not a quality one.

Every measured asymmetry in the harness runs **against** the LLVQ arm or is
negligible: FP16 is timed first, so thermal drift penalises LLVQ; submission
overhead is not subtracted; the LLVQ arm reads its tail in f32 where FP16 reads
f16; 9 buffer binds against 4. And any *common* additive term compresses the
ratio. **So 2.07× is a floor on the projection-arithmetic ratio, while the
1.88× below is a ceiling on anything end-to-end.**

It **excludes** attention, RMSNorm, SwiGLU, residuals, sampling, prefill, the
load-time transcode, and — the asymmetric one — **the incoherence rotation
applied to the activations, which only the quantized arm would pay**: 144
transforms per token, 0.2 % of the arithmetic, latency unmeasured because no
GPU implementation exists yet. Adding the f16 `lm_head` analytically (it is
never executed) brings the end-to-end ceiling to **1.88×**; the 78.2 tok/s that
follows from it is an upper bound, not a measurement of anything. **What the
shipped file actually generates today is 2.2–7.6 tok/s** — the kernel has no
caller in `bin/run`.

**What this costs in memory**, counted the way `bin/thesis` counts what it
reads — payload, addressing, f32 tail and f32 row scales, over every projection
weight:

| layout | bits/weight | loaded | speed | timed on |
|---|---|---|---|---|
| **Slot32** | **5.51** | **2.50 GB** | **2.06–2.08×** | the whole model |
| Flat32 | 4.68 | 2.12 GB | 0.90× | `gate_proj` only |
| Grouped32 | 3.50 | 1.59 GB | 0.68× | `gate_proj` only |

Fixed offsets are what buy the speed, and 5.51 b/w is *more* than an ordinary
4-bit format holds. Loading `Grouped32` gives up the speed for the space; both
come from the same file. Two caveats: only `Slot32` has ever been timed on the
whole model, so the 0.68× and 0.90× are single-layer figures under a different
harness; and `RuntimeBlocks::bits_per_weight()` reports a **narrower** metric
(payload and addressing over quantized weights only) that gives 5.38 for the
same `Slot32` — the two differ by convention, not by object.

An immediate reserve: the kernel reads the tail in **f32** where the FP16 arm
reads the same columns in f16 — 2.7 % of the LLVQ traffic, which would bring
`Slot32` to 5.44 b/w.

## What is *not* here

* **The kernel is not wired into the runner**, and the obstacle is not
  plumbing: `bin/run` has no KV cache, so it re-runs the whole prefix as a GEMM
  at every step, while `tv_slot` is a matvec. **The kernel has no caller.** The
  shipped model therefore gains no speed, and generates at 2.2–7.6 tok/s.
* **No CSR, and no domain-specific benchmark.** MMLU is measured above, on a
  2 280-question sample (16.2 % of the split), not the full suite.
* **No error bar on perplexity.** See *Read this before quoting the number*.
* **No measured quality for the 4-bit arm**, which leaves the most important
  column of the comparison empty.
* **The published command reproduces the method, not the bytes.** The C4
  calibration shard moved from `00000` to `00001` after the run, and the
  container format gained a magic bump; a re-run today produces a different,
  equally valid file. There is no CI.
* **The fused kernel implements exactly one point of the design space.** The
  `Slot32` shader reads a fixed 10-bit header — 9 bits of class, **1 gain
  bit** — and offsets every following field by it. The Rust transcoder is
  parameterised on the gain width; the shader is not. The paper's best
  no-fine-tuning configuration (2 gain bits) cannot be decoded by this kernel
  as written. The 2.07× is a result about this file's format, not about LLVQ
  layouts in general.
* **Determinism is uneven.** The Leech encoder is exactly deterministic and
  pinned by a test, but the calibration Hessians accumulate `AᵀA` in f32 on the
  accelerator, so re-running the recipe on another backend does not reproduce
  these weights.

## An open question for the authors

Appendix G compares single Leech shells against unions and adopts the union —
*"We therefore adopt this approach in our method and recommend doing the
same."*

We measured rate–distortion retention instead, on an i.i.d. Gaussian source
(20 000 blocks, fixed seed, gain centroids fitted by Lloyd–Max on a held-out
train split). Retention is `100·(−½·log₂ MSE)/rate`, and **every row below is
evaluated at the rate a file actually pays — whole bits, packed**:

| Code | bits/block | MSE | Retention | Classes |
|---|---|---|---|---|
| union `norm(Λ₂₄(12))` + 1 gain bit *(paper, Table 8)* | 48 | 0.078 *(as printed)* | 92.14 % *(paper's own, from its unrounded MSE; SQNR 1.843)* | 301 |
| **shell 12 only + 1 gain bit** | **48** | 0.0817 | **90.34 %** | **79** |
| shell 13 only + 1 gain bit | 49 | 0.0762 | 90.96 % | 82 |
| union `norm(Λ₂₄(13))` + 1 gain bit *(ours, same harness, same seed)* | 49 | **0.0725** | **92.72 %** | 383 |

`ceil(log₂ 70 486 236 999 360) = 47` for the single shell and
`ceil(log₂ 111 043 117 458 000) = 47` for the paper's ball — both cost 47 index
bits plus one gain bit, so rows one and two are matched at 48 bits per block.

**At matched rate the union is the better code, and we do not contest
Appendix G on distortion.** Rows three and four make the point strictly inside
one harness: the same 49 bits per block, 0.0725 against 0.0762, a 5 % MSE gap
in the union's favour. Row four is the unconstrained row of
`cargo run --release -p llvq-bench --bin lcap`.

> **An earlier version of this table claimed the opposite** — 92.24 % for the
> single shell against the paper's 92.14 %. That figure divided the same MSE by
> the *fractional* rate `log₂|Shell(12)|/24 = 1.9584`, which no file ever pays,
> and set it against a paper number quoted at 2.000. The rate column had been
> corrected; the retention column had not. It also credited the paper's
> `Λ₂₄(12)` with 383 equivalence classes — 383 is the count for `Λ₂₄(13)`;
> `Λ₂₄(12)` has 301.

**And this is a confirmation, not a finding.** Key finding 1 of Appendix G
already states that the union gives "slightly better *Gaussian
rate–distortion* curves" — which is exactly the quantity measured above, not
merely the angular nearest-neighbour distance of Figure 6. Our earlier table
contradicted a claim the paper had explicitly made; corrected, it agrees with
it. Key finding 2 likewise already states the hardware argument in full: that a
constant norm gives a fixed scaling between dot products and "eliminat[es] the
need to rescale intermediate dot product results before aggregation".

So the question we are left with is narrower, and it is genuinely a question.
Appendix G names the hardware advantage, calls the distortion difference
"small", and adopts the union anyway. **Was that decision taken on the
distortion curve alone, or did you measure what the rescaling costs in a
*multi-shell* fused kernel?** We ask because the trade we can measure is 79
equivalence classes against 301 and a fixed scale factor, for roughly 5 % more
MSE — and in the kernel we built, that rescaling is paid per shell.

Two things we cannot settle here. Our shipped file uses the **union** `Λ₂₄(12)`,
your own codebook, not the single shell; the shell has never been run through
the GPTQ loop on real weights, which is where a distortion gap of this size
would actually be decided. And this is one source, one seed.

## Reproducing

```bash
cargo test --release -- --include-ignored
```

```bash
cargo run --release -p llvq-bench --bin llvq-bench
```

Perplexity of the shipped file, and its baseline, in the same dtype and on the
same windows — the two arms are only comparable if the **token fingerprint**
printed on the result line matches:

```bash
LLVQ_DTYPE=f16 cargo run --release -p llvq-llm --features metal --bin ppl -- 4096 12 metal qwen3-4b-llvq.bin
```
```bash
LLVQ_MODEL=Qwen/Qwen3-4B LLVQ_DTYPE=f16 cargo run --release -p llvq-llm --features metal --bin ppl -- 4096 12 metal
```

Expect `ppl = 16.9415` and `ppl = 12.2361`, both with
`tokens 3f1baca9033bf251`. Scoring takes 165 s for the sealed file and 187 s
for the baseline; the sealed arm decodes 1.771 GB before its first window, so
budget ~6 min and ~4 min. On a machine with an empty Hugging Face cache the
baseline command also downloads the ~8 GB checkpoint first. **If the two
fingerprints differ, the ratio is meaningless** — that is what the line is
there for.

Quantize Qwen3-4B and write the compressed artifact (~4 h on an M3 Max). The
run verifies the file by decoding it and demanding the evaluated weights back,
bit for bit:

```bash
LLVQ_MODEL=Qwen/Qwen3-4B LLVQ_CALIB=c4 LLVQ_ARTIFACT=q4b.llvq cargo run --release -p llvq-llm --features metal,fast-linalg --bin smoke -- 64 2048 12 4096 metal nogs leech1c12 999 rot
```

One token of projections, LLVQ against FP16, on the whole model — with all 252
matrices verified before timing:

```bash
cargo run --release -p llvq-metal --bin thesis -- qwen3-4b-llvq.bin
```

What decoding costs in each candidate layout, which is what gated the kernel:

```bash
cargo run --release -p llvq-bench --bin decbench
cargo run --release -p llvq-metal --bin decreal -- qwen3-4b-llvq.bin
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

Determinism is uneven and worth knowing about: the Leech encoder is exactly
deterministic and pinned by a test, but the Hessians accumulate `AᵀA` in f32 on
the accelerator, so a third party on a different backend will not obtain the
same weights.

## Licence

MIT OR Apache-2.0. The Qwen3 forward pass in `llvq-llm/src/model.rs` is derived
from the architecture as implemented in
[`candle-transformers`](https://github.com/huggingface/candle) (MIT OR
Apache-2.0), restructured to make linear-layer inputs observable.
