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
> ⚠️ **It is not GGUF, AWQ or safetensors.** `transformers`, `llama.cpp`, vLLM
> and TGI do not read this file — the only reader that exists is
> `llvq-artifact`, in this repository. `bin/run`, the portable runner, decodes
> every weight into memory: on **that** path the size win is on disk only, and
> it needs ~10 GB of RAM on CPU and ~17 GB on Metal.
>
> **Since 2026-08-06 there is a second path, and it does win memory.** A CUDA
> kernel (`llvq-cuda`) keeps the weights encoded on the card; it is wired into
> the model and its caller is `bin/fusedrun` (Linux + `--features cuda`).
> Running the *published bytes* on an L40S: **2.96 GB on the card against 8.04,
> 48.7 tok/s against 43.6**, same greedy tokens as the dense arm up to a
> tie-break at token 89
> ([`docs/mesures/planes14-fusedrun-2026-08-06.txt`](docs/mesures/planes14-fusedrun-2026-08-06.txt)).
> On macOS the fused kernel is still a bench and nothing else: `llvq-metal` is
> gated on macOS and has no runner.
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

That is true of `bin/run`, on every backend. It is **not** true of the CUDA
fused runner, which keeps the weights encoded and holds the same model in
**2.96 GB** of card memory — see *Speed: the fused kernel*.

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
blamed on the protocol. **It has since reproduced itself too**: rerun four days
later on a different machine (L40S) in a different session, the same harness
gives 70.32 ± 1.28 and 55.59 ± 1.35 — 0.10 pp and 0.50 pp from the figures
above, both inside one σ. The table here keeps the original M3 Max run; the
*Against 4-bit* section uses the L40S one, because that is the run in which the
4-bit arm was scored under the same fingerprint.

We do not currently know what causes the remaining
4.8 pp, but the list of candidates is shorter than it was. **Two have been
tested and eliminated.** Calibration is *bounded*: deliberately calibrating on
the evaluation corpus itself — the maximum any volume, corpus or length choice
could ever buy — returns only −1.6 % of perplexity, closing 29 % of the gap,
and a ×13 volume increase returns −1.2 %. And the magnitude path (see *Naming*,
below) was implemented as the paper's Algorithm 3 reads to us, and **refuted at
full depth: ×1.99 perplexity** on a 28-block 0.6B run against the shipped
recipe — the second time in this project that a strictly better per-layer proxy
composed into a disaster over 28 layers
([`docs/archive/verdicts-lot-b-2026-08-06.md`](docs/archive/verdicts-lot-b-2026-08-06.md),
[`docs/archive/verdicts-nuit-2026-08-07.md`](docs/archive/verdicts-nuit-2026-08-07.md)).
**What moved instead was scale**, and it now has three points rather than two:
the MMLU drop against f16 reads −14.73 pp at 4B, −10.57 at 8B and −6.85 at 14B,
each one *paired on the same questions* with a 95 % interval excluding zero.
🚨 **This paragraph said "the perplexity excess flattens between the last two —
a knee, not a law". The knee was withdrawn on 2026-08-17 (morning) and half of
it was handed back the same evening — so any sentence about it must now NAME
ITS METRIC, because each bare form is half wrong.** On the **MMLU gap to
4-bit**, with all three AWQ − LLVQ gaps paired, the step-to-step drop tests as
**resolved from 4B to 8B (p = 0.0001)** and **unresolved from 8B to 14B
(p = 0.40)** — that slowdown is a property of the point estimates which the
error bars do not separate, and ⚠️ p = 0.40 does not prove equality either: on
that step the data are **silent**. What is tested there is the closing itself,
4B to 14B (−8.36 pp, p ≈ 1e-5). On **perplexity**, the slowdown **is resolved**:
paired window by window on the same 12 windows at all three sizes, the two steps
read ×0.881211 [0.856 ; 0.907] and ×0.974855 [0.959 ; 0.991], and their paired
difference is **−0.100992, 95 % CI [−0.137670 ; −0.064313], t = −6.06**
([`docs/mesures/ppl-appariee-4b-2026-08-17.txt`](docs/mesures/ppl-appariee-4b-2026-08-17.txt)).
**Two metrics, two verdicts — this is information, not a contradiction**:
perplexity is paired *across sizes* and weighs 49,140 scored tokens, MMLU
composes two independent 2,280-question campaigns, and 2-bit damages
**reasoning** far more than the **recall** a perplexity corpus mostly measures.
**Three points, not a law** — see *Against 4-bit*, which carries the figures and
the reserves.
An earlier version of this paragraph said "the gap to 4-bit halves", which was
true from 4B to 8B and is not the shape of the three-point curve. Candidates
still open and unmeasured:
calibration *composition* (the failure is concentrated in reasoning subjects,
which no corpus we use exercises), post-hoc compensation, per-column scale
fine-tuning, and our 1-gain-bit configuration,
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

## Against 4-bit — the comparison that matters, and we lose it

Nobody deploys FP16 locally. The honest reference is an ordinary 4-bit
quantization. **It was an empty column in this file until 2026-08-06; it is
measured now, and it does not go our way.**

Four arms, one L40S, one harness, the same windows and the same questions,
token fingerprints printed and identical on every quality row (perplexity
`3f1baca9033bf251`, MMLU `65dcd53655e8bfa5`). The 4-bit arm is **Qwen's own
official AWQ checkpoint**, not one we produced:

| | FP16 | **AWQ 4-bit** *(official)* | LLVQ 2-bit, dense | **LLVQ 2-bit + fused kernel** |
|---|---|---|---|---|
| **Cold storage** | 8.04 GB | 2.67 GB | **1.77 GB** | **1.41 GB**¹ |
| **Card memory** | 8.04 GB | *5.30 bits/param, in its own engine*² | 8.04 GB³ | **2.60 GB — 5.162 bits/param**⁵ |
| **Throughput** | 43.5 tok/s | *not comparable*² | 43.5 tok/s | **88.4 – 88.5 tok/s**⁴ |
| **WikiText-2 perplexity** | 12.2369 | **13.5207** *(×1.105)* | 16.9422 *(×1.384)* | **16.9358** *(×1.384)* |
| **MMLU** *(5-shot, micro, 2 280 q)* | 70.32 ± 1.28 | **70.04 ± 1.25** *(−0.28)* | 55.59 ± 1.35 | **55.70 ± 1.35** *(−14.6)* |

Sources: [`docs/campagne-finale-2026-08-07.md`](docs/campagne-finale-2026-08-07.md),
raw logs [`docs/mesures/a4-campagne-2026-08-06.txt`](docs/mesures/a4-campagne-2026-08-06.txt)
and [`docs/mesures/campagne-finale-bras4-2026-08-07.txt`](docs/mesures/campagne-finale-bras4-2026-08-07.txt).

¹ **Two files, one content.** Quality for column 4 is measured on
`q4b-e8.llvq` (1.406 GB, int8 embedding baked in); its speed and card memory
are measured on the published `qwen3-4b-llvq.bin` (1.770 GB) with
`LLVQ_EMBED=q8`, which quantizes the same embedding at load. Bit-identical
content, different bytes on disk.
² The AWQ has never run in *our* engine — loaded there it is dequantized to
f16, so any memory or speed number we could print for it would be meaningless.
Its 5.30 bits/param is its own engine's, whole model, embedding included.
³ Without the kernel the file decodes to f16 at load and runs exactly like
FP16 — that was the state of this project on 2026-08-04.
⁴ 88.4 in the final campaign, 88.5 in the integration run. Two protocols, one
object; the repo prints both rather than picking one.
⁵ **The two numbers in that cell do not divide into each other, and neither is
derived from the other.** 2.60 GB is what the card reports (`nvidia-smi`,
rounded to two digits); 5.162 bits/param is recomputed from the exact bytes
and the exact parameter count, whole model with embedding included, by
`rtbits` ([`docs/mesures/rtbits-planes-8b-2026-08-09.txt`](docs/mesures/rtbits-planes-8b-2026-08-09.txt),
which states the published 4B q8 figure is 5.162). The exact footprint is
2.595 GB, so dividing the displayed 2.60 GB gives **5.15** — the figure this
file published until 2026-08-17 and which is now labelled for what it is: a
quotation of the rounded card display, not a measurement of the object.

**Read the speed number with its companion.** The ×2.03 against the dense arm
is *not* the kernel alone: ~25 ms/token of it comes from replacing an output
head that recopies 778 MB of vocabulary per token. **That copy is ours, not
candle's.** Our dense arm calls `Tensor::broadcast_matmul`
([`llvq-llm/src/model.rs:553`](llvq-llm/src/model.rs)), whose rank-2-rhs path
materializes the transposed weight on every call (the `TODO` is in candle's
code). Models built on `candle_nn::Linear` fold the batch dimensions instead
and never pay it, so the trap is in the primitive, not in candle's models.
Reported upstream:
[huggingface/candle#3871](https://github.com/huggingface/candle/issues/3871).
**At identical head — f16 on both arms — the same measurement gives
×1.12** (48.7 against 43.6 tok/s), and that is the honest figure for what the
Leech kernel itself buys end to end
([`docs/mesures/phases-2026-08-07.txt`](docs/mesures/phases-2026-08-07.txt)).
Never quote the ×2.03 without the ×1.12.

**What the table says, in order.** We win cold storage (1.41–1.77 GB against
2.67) and, since Planes14 + int8 embedding, **card memory** (5.162 against 5.30
bits/param — the axis we were losing three days earlier). We lose quality, and
not marginally: **70.04 against 55.70 on MMLU, a 14.3-point gap**, while
4-bit is statistically indistinguishable from f16 (−0.28 pp, inside its own
±1.25). Speed does not compare honestly across two engines.

**On a 4B, 4-bit dominates us on capabilities and that is the verdict.** The
bet is scale, and it now has **three points — and three points are not a law.**
🕳️ *This sentence read "three points, and they show a knee, not a law" until
2026-08-17; on the MMLU gap the knee did not survive its own error bars, while
on perplexity it does — see the step tests below, and never state a knee without
naming which metric it belongs to.*
The LLVQ perplexity degradation reads ×1.3845 → ×1.2201 → ×1.1894 at 4B, 8B and
14B *(measured, same codebook, same calibration, same harness, same card, same
token fingerprints on all three)*; the excess over 1 therefore falls **42.8 %
from 4B to 8B and then only 13.9 % from 8B to 14B** *(computed from those three
ratios)*. ✅ **Both percentages now carry a paired interval: −42.8 %, 95 % CI
[−51.8 ; −33.5] and −13.9 %, 95 % CI [−22.8 ; −4.9], f16 reference** — and
their difference, the knee itself, excludes zero at t = −6.06.
🕳️ *This passage read "those two percentages are not comparable as evidence:
the first **cannot be given one at all** — the 4B campaign log is a summary, its
per-window NLLs were never kept." That was true for one day. The NLLs were in
the job's logs, which HF does not purge; `hf jobs logs` returned all 36 lines in
seconds, for $0, and the raw output is now committed
([`docs/mesures/a4-campagne-4b-ppl-BRUT-2026-08-06.txt`](docs/mesures/a4-campagne-4b-ppl-BRUT-2026-08-06.txt))
because that retention is neither documented nor guaranteed.* The −13.9 % is
still the **loosely bounded** one of the two, a factor 4.6 between its ends.
On the AWQ reference, the one that carries the
product argument, the same step reads **−1.58 %, 95 % CI [−3.14 ; −0.004]**,
t = 2.2063 against a 2.200985 threshold: it clears zero **by 0.005**. **Never
write that the gap closes significantly**. ⚠️ **And do not read −13.9 against
−1.58 as a ninefold difference: they are two parameterizations.** −13.9 % is
the fall of the **excess**, −1.58 % the fall of the **ratio** of perplexities,
which is the only form the journal publishes on the AWQ reference. Ratio
against ratio, the two references read **−2.51 % [−4.12 ; −0.88]** and
**−1.58 % [−3.14 ; −0.004]**
([`docs/mesures/ppl-appariee-8b-14b-2026-08-17.txt`](docs/mesures/ppl-appariee-8b-14b-2026-08-17.txt)).
The MMLU drop against f16 falls −14.73 → −10.57 → −6.85 pp, each one
**paired question by question** with a 95 % interval excluding zero — the 8B
term reads 10.57 here and 10.56 in the table further up because one is
`mmlupair`'s stratified estimate and the other the subtraction of two published
micro rates; same measurement, last digit only —
([`docs/mesures/mmlupair-4b-8b-2026-08-13.txt`](docs/mesures/mmlupair-4b-8b-2026-08-13.txt),
[`docs/mesures/campagne-14b-qualite-2026-08-10.txt`](docs/mesures/campagne-14b-qualite-2026-08-10.txt),
summary in [`docs/echelle-4b-8b-2026-08-08.md`](docs/echelle-4b-8b-2026-08-08.md)).
**Three points are not a scaling law any more than two were**, and this file
will not extrapolate one to 70B.

The gap to 4-bit reads **14.45 → 7.49 → 6.09 pp** — dense LLVQ arm on both
sides, which is why the first term reads 14.45 and not the 14.3 above. ✅ **All
three are now the same species of number**, each a *paired* AWQ − LLVQ estimate
with an interval: **+14.45 pp [+11.60 ; +17.27]** at 4B, **+7.49 pp
[+5.28 ; +9.70]** at 8B, **+6.09 pp [+3.62 ; +8.52]** at 14B (SE 1.25 pp, exact
McNemar p = 1.143e-11, 230/106 discordant pairs, stratified paired bootstrap,
10 000 draws, seed `0xb0075eed`, fingerprint `65dcd53655e8bfa5` on both sides —
[`docs/mesures/mmlupair-14b-2026-08-17.txt`](docs/mesures/mmlupair-14b-2026-08-17.txt)).
All three are resolved.

🕳️ **What this paragraph said until 2026-08-17, and why it was wrong.** It read:
"**Those three numbers are not the same species** […] the third is a bare
subtraction of two micro rates (78.21 − 72.12) — so **no AWQ − LLVQ pairing
exists at 14B**: no interval, no McNemar. Recovering one means rerunning the 14B
MMLU campaign, not recomputing something we hold. **Never quote 6.09 with an
interval.** […] the scratch directory is gone." The caution was right; the fact
was not. **The point estimate does not move by a hundredth** — this is not a new
number, it is the same one ceasing to be bare.

The dumps were never gone: the campaign job did not write to a machine, it wrote
to the **mounted bucket**, which exists precisely so job output outlives the
container. The 2026-08-16 check that declared them lost searched *the machine*.
They had been sitting in the bucket since 2026-08-10; recovering them cost
**579 kB of bandwidth and $0**, against a rebudgeted MMLU campaign. They are now
committed into [`docs/data/mmlu-dumps/`](docs/data/mmlu-dumps/), so the loss
cannot recur, and their authenticity was established *before* use: the three
stratified micro rates replay 78.97 / 78.21 / 72.12, and the already-published
`f16 − LLVQ` pair replays all four of its figures. **The standing rule this
bought: any output declared lost deserves an `hf buckets ls` before anyone
prices a re-run.**

🚨 **Widened the next day, because it happened twice and is therefore a pattern,
not an incident: exhaust the retention channels — `hf buckets ls`,
`hf jobs logs`, `hf jobs inspect` — before pricing a re-run.** On 2026-08-17 the
4B per-window NLLs were declared unrecoverable without a ~$0.25 card replay;
`hf jobs logs` returned all 36 of them in two seconds, for $0, and the raw
output is now committed
([`docs/mesures/a4-campagne-4b-ppl-BRUT-2026-08-06.txt`](docs/mesures/a4-campagne-4b-ppl-BRUT-2026-08-06.txt)).
Both times the verdict "lost" came from having looked in the wrong place, never
from a channel queried and empty, and both times money had been budgeted against
that absence. `hf jobs inspect` completes the set: it returns a past job's exact
command line. ⚠️ HF log retention is **neither documented nor guaranteed** —
which is why the raw output is committed rather than cited by job id.

🚨 **And the step-to-step test, which the homogeneous line makes possible for the
first time** (SEs composed in quadrature — *computed*; separate campaigns on
different models, so no cross-model pairing, which would be meaningless):

| step | drop in the gap | SE | z | p | verdict |
|---|---|---|---|---|---|
| 4B → 8B | 6.96 pp | 1.82 | 3.82 | 0.0001 | **resolved** |
| **8B → 14B** | **1.40 pp** | **1.68** | **0.83** | **0.40** | **unresolved** |
| 4B → 14B | 8.36 pp | 1.91 | 4.38 | ≈ 1e-5 | **resolved** |

**On MMLU, the first closing is real; the second is inside the noise.** ⚠️ And
p = 0.40 does not prove equality either — on that step the data are **silent**,
and both readings ("it is slowing" and "it keeps closing") remain compatible
with three points. **The perplexity verdict does not lift that silence**: a
second metric answering does not make the first one speak.

🚨 **And on perplexity the same slowdown *is* resolved, which is why every
sentence here names its metric.** Paired window by window on the same 12 windows
at all three sizes, the excess against f16 falls by a factor **×0.881211
[0.856 ; 0.907]** from 4B to 8B and **×0.974855 [0.959 ; 0.991]** from 8B to
14B; the difference of the two steps, paired, is **−0.100992 [−0.137670 ;
−0.064313], t = −6.06**, with 11 of 12 windows agreeing
([`docs/mesures/ppl-appariee-4b-2026-08-17.txt`](docs/mesures/ppl-appariee-4b-2026-08-17.txt),
[`docs/data/ppl-genou.csv`](docs/data/ppl-genou.csv)). The asymmetry has
mechanisms, not mysteries: perplexity is paired *across sizes* and weighs 49,140
scored tokens against 2,280 unpaired questions, and the two do not measure the
same thing — 2-bit damages **reasoning** far more than the **recall** a
perplexity corpus mostly probes. ⚠️ That interval carries corpus sampling only;
the **calibration** draw is absent at all three scales, and the knee compares
three artifacts each produced once.

What this strengthens is what the file already said: **no scaling law on three
points**, and the 32B point is what would settle it — the question it settles
being whether the *capability* curve flattens, which perplexity cannot answer
for it.

Three reserves this file will not smooth over.

* **"The 4-bit baseline starts paying" describes the 8B, not a trend.** It is
  the only scale where f16 − AWQ is resolved: **+3.07 pp [+1.61 ; +4.69]**. At
  4B it is unresolved (+0.27 [−1.63 ; +2.13]) and at 14B it is unresolved again
  (+0.76 [−0.65 ; +2.17]). Not monotone.
* **At 4B that verdict depends on the accounting.** The unweighted control
  *does* resolve f16 − AWQ (+1.97 [+0.92 ; +3.02]) where the stratified micro
  does not, and the disagreement is carried by `professional law`, 10.9 % of the
  population.
* 🕳️ **"None of these intervals tests the difference of differences between
  scales" — true when written, no longer the whole story.** `mmlupair` still
  pairs two arms on the same questions and never two model sizes, and
  non-overlapping intervals are still not a test. But since 2026-08-17 the
  step-to-step drop **is** tested, by composing the two campaign SEs in
  quadrature (table above): resolved 4B→8B, **unresolved 8B→14B**. That is a
  formal test, and it is the one that withdrew the knee **on this metric**.
  ⚠️ It says nothing about perplexity, which *is* paired across sizes — same 12
  windows, same text, same token fingerprint at all three scales — and where the
  knee is **resolved** (t = −6.06). Two metrics, two verdicts; a bare "the knee
  holds" or "the knee does not hold" is half wrong either way.

✅ **The memory reading reaches three points on 2026-08-17.**
🕳️ *This paragraph read "still at two points, not three — no whole-model
bits/param exists for the 14B, `rtbits` has never been run on a sealed 14B".
That was exact and is now obsolete: the sealed 14B artifact had never been
brought back after the campaign, but it was still in the bucket, and re-reading
it cost bandwidth only —*
[`docs/mesures/rtbits-14b-2026-08-17.txt`](docs/mesures/rtbits-14b-2026-08-17.txt).
The axis now holds **5.162 against AWQ's 5.302 at 4B (−2.6 %), 5.322 against
5.956 at 8B (−10.6 %), and 5.106 against 5.404 at 14B (−5.5 %)**
*(`Planes14` + int8 embedding; our figures **computed on measured bytes** with
the embedding **modelled** at 8.5 bits/param — the same status at all three
sizes — against AWQ safetensors bytes read from the Hub API, whole model with
embedding included, the only accounting in which the two compare)*. `params_total`
for the 14B is **14,768,307,200**, read from the sealed file and cross-checked
by the architecture's arithmetic. **We are under deployed 4-bit at all three
sizes.**

🚨 **The margin is not monotone and carries no trend.** It peaks at 8B and falls
back. The mechanism is not the method but the **embedding's share** — 9.7 % at
4B (tied heads), 15.2 % at 8B, 10.5 % at 14B — which AWQ leaves in f16 and we
move to int8. Three points, one mechanism, **no law**. ⚠️ Neither speed nor card
VRAM has ever been measured at 14B: no `fusedrun` has run at that width, so the
14B lacks the third instrument (the engine's own VRAM report) that
cross-checked the 4B and 8B cells. ✅ **Cold storage at 14B, on the other hand,
is settled and was already**: `qwen3-14b-llvq.bin` is **6,506,354,741 bytes =
6.506 GB**, *measured*, confirmed to the byte by `hf buckets ls` **and** by the
sealing job's log — two independent routes. Two cells open at 14B, not three.

⚠️ The AWQ references are **three different models**: 5.302 is the 4B, 5.956 the
8B, 5.404 the 14B — they are not one shared baseline. The table above now publishes **5.162**, the
`rtbits` verdict on the exact bytes — settled 2026-08-17. The **5.15** it
carried before is the same 4B object read off the rounded card display
(2.60 GB shown for an exact 2.595 GB); it is kept here labelled rather than
deleted, because a corrected claim says so.
🕳️ *This paragraph ended "The 14B figure is **missing**, not omitted for
brevity, and it is not to be extrapolated from the two that exist" — obsolete
on 2026-08-17, and it survived the correction three paragraphs above that
already publishes it.* The 14B figure is **5.106 against 5.404**, measured on
the sealed artifact recovered from the bucket
([`docs/mesures/rtbits-14b-2026-08-17.txt`](docs/mesures/rtbits-14b-2026-08-17.txt)).
What still must not be extrapolated is the **margin**: it is not monotone, and
nothing here licenses a fourth point.

<details><summary>The earlier MLX q4 comparison, on the Mac, kept for genealogy</summary>

Before the campaign above, the only 4-bit reference here was one produced
locally with `mlx_lm.convert -q --q-bits 4 --q-group-size 64`, on an M3 Max:
2.263 GB on disk (4.50 bits/param), 2.39 GB of MLX allocator peak (not an RSS),
129.8 tok/s end to end — none of it with a kept trace, and **its quality was
never measured**. That column stayed empty until Qwen's AWQ was run in our own
harness, which is what the table above does. The MLX figures are not comparable
to the L40S campaign and are not used anywhere in this file.

</details>

The structural niche for 2-bit is the memory window where 4-bit does not fit
and we do. Whether it is worth anything at 70B is **untested** — no 70B has
ever been quantized here, and the KV cache (320 KiB/token in f16) is not
budgeted in any of our projections. Full analysis:
[`docs/archive/face-au-4-bits.md`](docs/archive/face-au-4-bits.md).

## Read this before quoting the number

* **We are at QTIP's level, marginally worse.** 16.96 against 17.04 looks like
  a win and is not: our baseline is lower, and normalised on each side's own
  baseline we are 3.1 % worse.
* **The 0.08 perplexity margin is still not defensible, but the reason has
  improved.** There is now a measured dispersion — and it is measured on a
  *different object*. Three calibration seeds on a 3-block Qwen3-0.6B run give
  **σ ≈ 0.15 perplexity, ≈ 0.7 %**, around a quantized perplexity of ~20.66, so
  the working rule on such a run is that anything under ~1.5 % (2 σ) is noise
  ([`docs/archive/verdicts-lot-b-2026-08-06.md`](docs/archive/verdicts-lot-b-2026-08-06.md), §B1;
  n = 3, a coarse estimate and the project's first).
  **That σ does not transfer to the 16.9415 published here**: different model,
  3 blocks against 36, different perplexity scale. **No σ has ever been measured
  on the full-model number**, and a 0.5 % margin over QTIP sits well inside the
  only dispersion we can point at. The older and cruder observation still
  stands too: ~7 % between two configurations that a test proves were the same
  quantizer, n = 2, cause unresolved.
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
> [`docs/archive/retraction-et-gain.md`](docs/archive/retraction-et-gain.md). All three were
> found by trying to *write the file* rather than compute its size.

## What is here

| Crate | Contents | Dependencies |
|---|---|---|
| `llvq-core` | Extended Golay code, Λ₂₄, shells | none, `forbid(unsafe)` |
| `llvq-search` | Exact NN search over shells m ≤ 13, bijective 48-bit index, bit packing | none |
| `llvq-quant` | GPTQ (Alg. 1), dense linear algebra, incoherence rotation | none (`faer` optional) |
| `llvq-artifact` | The `.llvq` format: writer, reader, decoder | **none** |
| `llvq-metal` | Metal micro-benchmarks: decode cost, `bin/thesis` | macOS only |
| `llvq-cuda` | The CUDA kernels: `Slot32`, `Planes14`, `Planes12x`, `Golay70`, rotation | `cudarc`, Linux + NVIDIA only |
| `llvq-llm` | Observable Qwen3 forward, Hessians, perplexity, generation, fused runtime | `candle` |
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

| | ms/token (min – max) | GB read | bits/weight | vs FP16 |
|---|---|---|---|---|
| FP16 (half4) | 21.73 – 22.00 | 7.27 | 16.000 | 1.00× |
| **LLVQ fused (Slot32)** | **10.50 – 10.81** | 2.50 | 5.510 | **2.03× [2.03–2.10]** |

Ranges, not point values, and the ratio is formed **round by round** — 7 rounds,
2 discarded, both arms dispatched every round in the same order — then reported
as the median and range over the 5 kept rounds. It is never a quotient of two
best-of times, which would mix rounds that never coexisted. Log:
[`docs/mesures/k1-metal-2026-08-05.txt`](docs/mesures/k1-metal-2026-08-05.txt).

**The ratio drifts between processes, and the repo measured how much.** Three
consecutive invocations of the unmodified two-arm bench give 2.029×, 2.050×,
2.080×; three of the seven-arm bench give medians of 2.03×, 2.06×, 2.09×. The
bytes read, the bits/weight and the worst errors are identical to the digit in
all of them — **only the times move, and they move together on both arms**
([`docs/mesures/thesis-temoin-2026-08-04.txt`](docs/mesures/thesis-temoin-2026-08-04.txt)).
So the defensible statement is **2.03× to 2.09×**, a third decimal on this
ratio has no content, and an earlier version of this file claiming the ratio
moved by only 0.8 % was understating its own dispersion by a factor of four.
What *is* reproducible to the digit is the 1 105 920 verified rows and the worst
observed error, **3.4·10⁻⁸ · Σ|wᵢxᵢ|**.

```bash
cargo run --release -p llvq-metal --bin thesis -- qwen3-4b-llvq.bin
```

**What this ratio is, and is not.** It times 252 fused matvecs on an M3 Max at
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
ratio. **So 2.03–2.09× is a floor on the projection-arithmetic ratio, while the
1.88× below is a ceiling on anything end-to-end.**

It **excludes** attention, RMSNorm, SwiGLU, residuals, sampling, prefill and the
load-time transcode. It also excludes the asymmetric term — **the incoherence
rotation applied to the activations, which only the quantized arm pays**: 144
transforms per token, 0.2 % of the arithmetic. That one is no longer
unmeasured. A CUDA `rot_apply` exists, is verified against an f64 reference on
eight shapes (worst relative error 9.5·10⁻⁸) and costs 8.05 µs at n = 2560 in
isolation
([`docs/mesures/rotation-cuda-2026-08-05.txt`](docs/mesures/rotation-cuda-2026-08-05.txt));
it runs inside the fused path, one launch per projection ahead of the matvec, so
the end-to-end CUDA figures further down already pay for it. **There is still no
Metal implementation**, so the Metal ratio above does not.

Adding the f16 `lm_head` analytically (it is never executed in this bench)
brings the Metal end-to-end ceiling to **1.88×**; the 78.2 tok/s that follows
from it is an upper bound, not a measurement of anything. For what the shipped
file actually generates, see *End to end, on a card* below.

**What this costs in memory**, counted the way `bin/thesis` counts what it
reads — payload, addressing, f32 tail and f32 row scales, over every projection
weight. All three layouts are from the **same run, same protocol, same byte
accounting**, timed on the whole model:

| layout | bits/weight | loaded | vs FP16 |
|---|---|---|---|
| **Slot32** | **5.510** | **2.50 GB** | **2.03× [2.03–2.10]** |
| Flat32 | 5.256 | 2.39 GB | 0.91× [0.91–0.91] |
| Grouped32 | 3.498 | 1.59 GB | 0.69× [0.68–0.69] |

Source: [`docs/mesures/k1-metal-2026-08-05.txt`](docs/mesures/k1-metal-2026-08-05.txt),
seven arms, seven rounds, two discarded. **An earlier version of this file gave
Flat32 as 4.68 bits/weight over 2.12 GB and called the two slow rows
single-layer figures from a different harness.** Both were wrong: 4.68 came from
a different byte accounting than the 5.51 printed next to it — exactly the fault
this run existed to remove — and all seven arms were timed on the whole model,
interleaved, in one process.

The curve is brutally non-linear. Flat32 saves 0.254 bits/weight over Slot32 and
costs 2.27× the time; Grouped32 saves 2.012 and costs 3.01×. **Fixed offsets are
what buy the speed**, and 5.51 b/w is *more* than an ordinary 4-bit format holds
— which is why the next step was to take bits back *inside* a fixed-offset
layout rather than to change layout family.

Note also that `RuntimeBlocks::bits_per_weight()` reports a **narrower** metric
(payload and addressing over quantized weights only) that gives 5.38 for the
same `Slot32` — the two differ by convention, not by object. And the kernel
reads the tail in **f32** where the FP16 arm reads the same columns in f16 —
2.7 % of the LLVQ traffic, which would bring `Slot32` to 5.44 b/w.

### The layout scale on CUDA, and end to end on a card

`Slot32` is no longer the reference layout. **`Planes14`** replaces the one-hot
slot masks with binary bit-planes at a uniform 14-byte stride and no base table:
same decoded content, bit for bit, smaller *and* faster. Measured on an L40S,
**seven arms in one process** (six, then the same six plus one — the incumbents
move by at most 0.24 % between the two phases), ratios formed round by round
([`docs/mesures/golay70-v2-sept-bras-2026-08-11.txt`](docs/mesures/golay70-v2-sept-bras-2026-08-11.txt)):

| kernel | bits/weight | GB read | GB/s | of byte bound | vs FP16 |
|---|---|---|---|---|---|
| Slot32 | 5.510 | 2.50 | 429 | 65 % | 1.89× [1.88–1.89] |
| **Planes14** *(default)* | **4.804** | 2.18 | 427 | 65 % | **2.15× [2.15–2.16]** |
| Planes12x *(sparse overlay)* | **4.342** | 1.97 | 360 | 54 % | 2.00× [2.00–2.01] |
| Golay70 | 3.589 | 1.63 | 199 | 30 % | 1.34× [1.34–1.34] |
| Golay70, hoisted decode | 3.589 | 1.63 | 263 | 40 % | 1.77× [1.76–1.78] |
| **AWQ w4g128** *(competitor)* | 4.179 | 1.90 | 583 | **88 %** | 3.37× [3.36–3.38] |

🚨 **Read the "of byte bound" column, not the "vs FP16" one.** A ratio against
FP16 mechanically rewards whoever reads least; the comparable quantity is what
fraction of its own byte advantage a kernel converts into time, taking the
661 GB/s the FP16 control reaches on these shapes as the reference. A deployed
4-bit kernel converts 88 % where our best layout converts 65 % — **that gap is
the honest statement of what is left to do**, and it is not in the format.

⚠️ Deux asymétries de comptabilité, dans les deux sens. Notre queue en pleine
précision est facturée dans notre b/poids et AWQ n'en a pas — ça nous
défavorise sur cette colonne. Mais sur la colonne « of byte bound », qui porte
le résultat, elle nous **flatte** : queue et échelles de ligne retirées,
`Planes14` lirait 2,12 Go dans les mêmes 5,116 ms, soit 414 Go/s et **63 %**,
et l'écart au 4 bits serait de 25 points, pas 23.

Planes12x reaches 4.342 b/w at **exactly identical quality** — the capped blocks
are corrected by a sparse exception pass in the same launch, and every one of the
1 105 920 rows still matches the f64 reference.

**Golay70 is a negative result, attacked twice.** It stores a 12-bit *rank* of
the block's Golay codeword instead of a 24-bit sign mask — the kernel resolves
that rank through a resident 16 KiB codeword table; it does **not** re-encode
anything by XOR, and an earlier version of this section said it did. The format
result is real (3.589 b/w, reconstruction-exact on all 150 681 600 blocks); the
kernel result is not. Resolving the coset *per slot* left it ALU-bound at
199 GB/s and 1.34×, under the 1.6× criterion set before the run. Hoisting that
decode to a per-block prologue — **zero stored bytes changed**, identity proved
slot by slot and block by block — moved it to 263 GB/s and 1.77×, and that is
still under the replacement criterion (≥ 2.0× *and* ≥ 20 % whole-model memory
margin) which was committed and timestamped *before* the measurement
([`proofs/preregistration-2026-08-11.md`](proofs/preregistration-2026-08-11.md)).
Not adopted. The per-slot path is now identical to a layout we already ship, so
there is no obvious repair left.

**End to end, on a card.** `bin/fusedrun` loads the published artifact twice —
once dense, once with the projections left encoded — and requires the same
greedy tokens out. On an L40S, 128 tokens, `Planes14`:

| | dense f16 | fused |
|---|---|---|
| tok/s | 43.6 | **48.7 (×1.12)** |
| GB on the card | 8.04 | **2.96 (÷2.72)** |
| tokens | reference | identical to a tie-break at token 89 |

With the tied embedding also quantized to int8 at load (`LLVQ_EMBED=q8`) the
same run reaches **88.4–88.5 tok/s in 2.60 GB**. That ×2.03 is *not* the Leech
kernel: ~25 ms/token of it is *our own* dense `lm_head` path being replaced, the
`broadcast_matmul` copy described above, not something candle's models do.
**×1.12 is
the kernel's own end-to-end contribution and the two must be quoted together**
([`docs/mesures/planes14-fusedrun-2026-08-06.txt`](docs/mesures/planes14-fusedrun-2026-08-06.txt),
[`docs/mesures/phases-2026-08-07.txt`](docs/mesures/phases-2026-08-07.txt)).
Half a token's time is attention, norms and launch overhead, which the fused
path does not touch — that is what bounds ×1.12, and it is why the memory
column, not the speed column, is the result here.

## What is *not* here

* **No fused path on Apple silicon.** The kernel now has a caller — but only
  one, and only on CUDA: `bin/fusedrun`, gated on `linux` + `--features cuda`.
  `bin/run`, the portable runner used in every command in this file, still
  decodes to dense f16 at load on both CPU and Metal. So on a Mac the shipped
  model gains no memory and no speed from any of this. Two obstacles that used
  to be listed here are gone: `bin/run` has had a KV cache since commit
  `9c24d26`, and a GPU implementation of the incoherence rotation exists on
  CUDA.
* **No CSR, and no domain-specific benchmark.** MMLU is measured above, on a
  2 280-question sample (16.2 % of the split), not the full suite.
* **No error bar on the published perplexity.** A σ exists — three calibration
  seeds, but on a 3-block 0.6B run, which is a different object. See *Read this
  before quoting the number*.
* **No 4-bit arm inside the fused runner.** The 4-bit quality column is no
  longer empty (see *Against 4-bit*), but the AWQ is only ever loaded
  dequantized in our engine, so its memory and speed have never been measured
  here and the comparison on those two axes remains cross-engine.
* **The published command reproduces the method, not the bytes.** The C4
  calibration shard moved from `00000` to `00001` after the run, and the
  container format gained a magic bump; a re-run today produces a different,
  equally valid file. The repo's CI (since 2026-08-09: clippy, tests and the
  zero-dependency guard, on the 6 CPU crates — no GPU) checks the code, not
  the bytes: no automation reproduces the published artifact.
* **Every fused kernel implements exactly one point of the design space.** Both
  the `Slot32` shader and the `Planes14` one read a fixed 10-bit header — 9 bits
  of class, **1 gain bit** — and offset every following field by it. The Rust
  transcoder is parameterised on the gain width; the kernels are not, and
  `planes14_host.rs:113` asserts `gain_bits == 1` rather than degrade quietly.
  The paper's best no-fine-tuning configuration (2 gain bits) cannot be decoded
  by any of them as written. These ratios are results about this file's format,
  not about LLVQ layouts in general.
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

On an NVIDIA card (Linux only), the layout bench and the end-to-end A/B — the
second loads the same artifact twice, dense and fused, and requires the same
greedy tokens out. `LLVQ_FUSED_LAYOUT=slot32` and `LLVQ_EMBED=q8` are the two
switches the measurements above use:

```bash
cargo run --release -p llvq-cuda --bin planesbench -- qwen3-4b-llvq.bin
```
```bash
cargo run --release -p llvq-llm --features cuda --bin fusedrun -- qwen3-4b-llvq.bin 128
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
