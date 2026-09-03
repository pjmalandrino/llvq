# Test protocol, isolated stacks (2026-08-30)

> Written **before** any measurement. Each section says what is recorded, with
> which instrument, and above all **what the number is allowed to conclude**.

## 0. The choice of primary metric, and it is not the usual one

MMLU is the primary. Perplexity is the secondary. That is the reverse of common
practice, and it follows from the design.

1. **Perplexity does not cross stacks.** It depends on windowing conventions
   specific to each engine: our 12 windows of 4096 are not the sliding chunks of
   `llama-perplexity`. Its **level** only means something inside one machine;
   only its ratio to the f16 control leaves the machine.
2. **MMLU crosses.** It depends only on the tokenizer and on the logprob of 4
   answer tokens. Two engines that share the tokenizer return MMLU values
   comparable **in level**.
3. **And it is the metric that sees the damage.** §3ter has said so since
   2026-08-02: 2 bits damages **reasoning** far more than **recall**, and
   recall is mostly what a perplexity corpus measures. At 4B, ppl moves by
   ×1.384 while MMLU loses **14.73 pp**. Rule of the record, repeated here:
   *never present perplexity alone as evidence of quality.*

## 0bis. The backend axis, CUDA and Metal are two stacks

A backend is a stack, on the same footing as an engine. The record has already
measured that the × do not even divide between **two CUDA cards**: on A100 no
decoding arm beats FP16 (`Planes14` 0.79× against 2.14× on L40S), and batch G
settled that the ×1.78 **is** the clock ratio 2,520/1,410 MHz. Between CUDA and
Metal, the ban holds *a fortiori*.

⇒ **Each backend has its own f16 control**, and only ratios compare.

### What Metal can carry, and what it cannot

| | Metal |
|---|---|
| **quality** (P1, P2) | complete, and it is a **cross-backend control** |
| **memory** (P3) | complete |
| **LLVQ throughput** (P4) | **impossible**, there is no `fused_metal.rs` |
| **third-party arm throughput** | llama.cpp and MLX are served natively there |

The trap is structural. `llvq-llm` carries only `fused.rs` and `fused_cuda.rs`.
On Metal, LLVQ exists only as a micro-benchmark: `bin/thesis` measures **252
projections on one token**, where `mlx_lm.generate` runs a **whole model over
256 tokens**. Comparing them would be the "two denominators" fault of §7. On
Metal, LLVQ plays only on quality and on the micro-benchmark.

### C4, the cross-backend control, free

The **same sealed file** scored on Metal and on CUDA must return **the same
quality**: the same weights, decoded by two independent implementations.
The baseline has already shown it: **70.42 (Metal) → 70.32 (CUDA)**, 0.08 σ,
across a change of backend, card and dtype.

⇒ If the two backends diverge in quality on the same file, **that is a porting
bug**, not a result. This control costs nothing and it catches the most
expensive class of error in the project (cf. §5 of `CLAUDE.md`, the fourth
catch: *a transcription carries the guards of its original without carrying the
assumptions that made them sufficient*).

### C5, the Metal equivalent of F1: it does not exist, and it is a named debt

On CUDA, **F1** established that our in-house FP16 control reaches **1.024×** cuBLAS
on the 2-arm bench and **1.015×** on the 5-arm bench, on L40S, both ≤ 1.05. That
is what holds up *every* published "vs FP16" ratio.

On the Metal side, `docs/fiche-4b.md:438` names the angle and declares it
unaddressed: *"this baseline has never been confronted with MPS, MLX or
Accelerate: the 2.07× is a ratio against a kernel written by the same author.
That is the remaining hostile angle, unaddressed."*

⇒ **As long as C5 is not done, no Metal × gets published.** Confronting it with
MPS / MLX / Accelerate costs **$0** and is worth more than any extra arm of this
experiment.

## 1. P1, MMLU micro *(primary)*

- **Fixture**: 2,280 questions, 57 subjects, **micro** aggregation
  (`Σright/Σtotal`), never macro, cf. the warning in §3ter: swapping macro for
  micro is worth ~1 pp and hits the quantized arm harder than the control.
- **Mandatory output**: **per-question dump with `qhash`**, not an aggregate
  rate. That is what makes pairs formable after the fact: `mmlupair` made nine
  of them for $0 on 2026-08-17, from dumps dating from 2026-08-10.
- **Statistics**: **paired bootstrap stratified by subject**, 10,000 draws, seed
  `0xb0075eed`, plus exact McNemar. The sampling ± of a single arm is **not**
  the error bar of a difference.
- **Free cut, and the most informative**: MMLU **by subject group** (reasoning
  vs recall). The dump already carries it, it is an `awk`, not a run. That is
  where the mechanism shows: `abstract_algebra` and `accounting` fall to **25%,
  chance**, while history and law hold above 80%.

## 2. P2, perplexity *(secondary)*

- wikitext-2 test, ctx 4096, 12 windows, f16.
- **Per-window NLL kept.** `bin/ppl` prints them to 9 decimals **on stderr**, so
  they are lost without `2>`. §7 paid this lesson three times: a summary journal
  is an irreversible loss.
- **Published only as a ratio** to the f16 control of its own machine.
- Serves as a **bridge to the literature** (everyone publishes ppl), not as
  evidence of quality.

## 3. P3, memory

- **b/param whole model, embedding included.** Never a b/weight of projections
  against a b/param of a whole model: that is the serious fault of the batch A
  erratum.
- Comparable **directly** between machines: it is a byte count.
- Provenance note, and it plays **against** us: for a GGUF,
  `file bytes ÷ params` **is** the value, measured. For our arm, the embedding
  is *modeled* at 8.5 b/param. Label the two differently.

## 4. P4, throughput

- Median over **5 rounds**, 1 generation discarded, **with range**, never a
  single point. §7: milliseconds drift from one invocation to the next where
  bytes reproduce to the digit.
- **Always as a ratio to the f16 control of the same machine.** A bare tok/s
  does not get published.

## 5. The controls, without them nothing above holds

### C1, the f16 control of each machine must reproduce the known values

This is the most important and the cheapest control, and it is the cross-engine
analogue of the "identity control returns ×1.000 exactly" of §5.

At 4B, f16 must return **MMLU 70.32 ± 1.28** and **ppl 12.2369**.

If the f16 control of a machine does not reproduce these values, that machine is
broken and its quantized arm means nothing. No quantized arm is read before its
control is green. A control deviation is a harness failure, not a result.

### C2, the token fingerprint, identical everywhere

`65dcd53655e8bfa5` (MMLU) and `3f1baca9033bf251` (ppl). That is what **proves**
"same data" instead of declaring it. Two machines that show the same fingerprint
have read the same text, token for token. A machine that does not print its
fingerprint does not take part in the comparison.

### C3, the asymmetries, declared in advance and with the direction they push

| asymmetry | direction |
|---|---|
| our dense arm copies 778 MB of vocabulary per token (`Head::project` → `broadcast_matmul`) | **against us**, it sits in the denominator of our ratios, so we underestimate our lead |
| the vLLM 2-bit kernel is the ExLlamaV2 path, **not** Marlin (4 bits only) | **for us**, the GPTQ arm is handicapped on speed |
| `IQ2_XXS` is at **2.06 bpw** against our 2.0702 | near neutral, the tightest throughput comparison in the record |
| `Q2_K` is at ~2.6–3.0 bpw | **another throughput class, do not use it**, that would be the "two denominators" fault |

## 6. The reading bar, set in advance

- **Paired MMLU SE**: **0.43 pp** at constant file; **0.79 to 1.44 pp** between
  different models (*measured* on 2026-08-15; the 0.4–0.6 pp in circulation
  before were *estimated* and never computed). ⇒ **An MMLU gap below ~1.5 pp
  between two arms is not resolved.** Do not narrate it.
- **Calibration σ**: F5 returns **5.2%** of ppl at the published size, not the
  0.7% of batch B, which held for 3 blocks of Qwen3-0.6B and is wrong by a
  factor of ~7 here.

  **And it applies to this experiment, unlike the A/B tests at constant file.**
  The **GPTQ** and **IQ2_XXS** arms are **calibrated**, GPTQ on C4, IQ2 by
  importance matrix, so each is **ONE draw** of calibration windows, exactly
  like our own artifact. Consequence:

  - the **absolute level** of each calibrated arm is that of one draw, not privileged;
  - a difference between **two separately calibrated arms** mixes *method* and *draw*;
  - for a 14 pp gap this has no effect; **for a 3 pp gap between two 2-bit arms
    it is decisive**, and that is precisely the order of magnitude expected.

  ⇒ **No ranking between two 2-bit arms gets published on a single draw each.**
  Either the gap is large against 5.2%, or several seeds are needed, or we write
  that the data is silent.

## 7. The gates, in order

| gate | condition | if red |
|---|---|---|
| **G-a** | the f16 control of each machine reproduces 70.32 / 12.2369 | the machine is broken, its quantized arm is not read |
| **G-b** | the token fingerprint is identical on every machine | the machine does not take part in the comparison |
| **G-c** | the quantized arm returns a b/param consistent with its bytes | accounting error, cf. the batch A erratum |
| **G-d** | the same sealed file returns the **same quality** on Metal and on CUDA | porting bug, not a result (C4) |
| **G-e** | the FP16 **Metal** control has been confronted with MPS / MLX / Accelerate | **no Metal × gets published** while this is red (C5) |

## 8. What this experiment will not be able to say

- **No speed ratio between two machines.** Ever.
- **No end-to-end LLVQ throughput on Metal.** There is no `fused_metal.rs`: on
  that backend, LLVQ plays only on quality and on the micro-benchmark
  (`bin/thesis`, 252 projections on one token).
- **Nothing on QTIP**, out of scope for lack of a Qwen3 artifact.
- **No scaling law**: a single size (4B).
- **Nothing on reasoning in real use**: MMLU is multiple choice. The
  document-extraction business benchmark called for in §6 of Phase 5 **still**
  has neither verdict nor date, and this experiment does not replace it
  (cf. arXiv:2607.08734: perplexity and accuracy stay stable while individual
  answers change).
