# Isolated stacks: one kernel per machine, two backends (2026-08-30)

> **Status: PREPARED, NOT LAUNCHED.** Nothing in this folder has consumed a
> minute of card time. No job leaves without an explicit go from the operator and
> without the timestamp on the preregistration (§7 of `CLAUDE.md`).

## 1. The question

The record compares LLVQ to FP16 and to 4-bit AWQ. It has **never measured the
quality of a 2-bit competitor**: QTIP's 17.04 is quoted from Table 6 of the
paper, not measured by our harness, and F2 says so itself (pseudo-random
payload, "no quality claim can rest on this arm"). Every 2-bit quality verdict we
hold compares against a borrowed number.

This experiment answers it in the shape the operator asked for: **one machine per
arm, each with its native engine and its own kernel, the same data, then we shut
down**. It runs on **two backends**, **Metal** on the operator's machine and
**CUDA** on HF Jobs.

## 2. What the design allows and what it forbids

Separate machines never coexist. The 08-17 rule applies in full, and it is
*measured*: the vLLM f16 control gives **83.09 tok/s** where ours gives **43.6**
on the same card. The engine dominates the end-to-end gap; the weight decoder
does not.

**A backend is a stack just as much as an engine is.** The record measured that
the × do not divide even between **two CUDA cards**: on A100 no decoding arm
beats FP16 (`Planes14` 0.79× against 2.14× on L40S), and the ×1.78 **is** the
clock ratio. Between CUDA and Metal the ban holds *a fortiori*.

| quantity | comparable across machines? |
|---|---|
| **memory**, b/param over the whole model | yes, **directly**, a byte count with no engine in it |
| **MMLU** micro | yes, **directly**, it depends only on the tokenizer and 4 logprobs |
| **perplexity** | caveat: **only as a ratio to the f16 control of its own machine** |
| **tok/s** | caveat: **only as a ratio to the f16 control of its own machine** |

**The fix is structural**: every machine **also runs its own f16 control**, on
the same data. Each machine therefore returns a **ratio** rather than a bare
number. `ops/awq_speed.py` already has this shape.

## 3. The files

| file | contents |
|---|---|
| [`PROTOCOLE.md`](PROTOCOLE.md) | what we measure and **what each number is allowed to conclude** |
| [`MACHINES.md`](MACHINES.md) | one datasheet per machine, per backend |

## 4. Which arm runs where

| arm | CUDA (HF, paid) | Metal (Mac, $0) |
|---|---|---|
| **LLVQ 2-bit** | yes, **served**, `fusedrun` | caveat: **micro-benchmark + quality**, no `fused_metal.rs` |
| **AWQ 4-bit** | yes, served, vLLM | no engine |
| **GPTQ 2-bit** | caveat: artifact to produce | no engine |
| **`IQ2_XXS`** | caveat: served, llama.cpp CUDA | caveat: **served, llama.cpp Metal** |
| **MLX 2-bit** | no, does not exist outside Apple | caveat: **served, native**, `q_bits=2` verified |
| ~~QTIP~~ | no Qwen3 artifact | no |

**`IQ2_XXS` is the only arm that crosses both backends.** It is the bridge of the
experiment.

**QTIP is set aside, on a fact checked on 2026-08-30**: relaxml publishes Llama
only. Porting it would cost *estimated* $10–20 plus a risk of architecture
incompatibility. Its **speed** is already measured by F2, in the strongest form
there is (one process, interleaved arms, a legitimate division: 2.27×
[2.27–2.28]). Its **quality** is what is still missing.

## 5. The order, and it brings the cost down

**Everything that can be settled on Metal is settled before renting a card.**
That is what the project has done historically: "a free benchmark beats a rented
card for finding the measurement traps".

| phase | where | cost |
|---|---|---|
| **1**, C5: put the Metal FP16 control against MPS / MLX / Accelerate | Mac | **$0** |
| **2**, produce `IQ2_XXS` (imatrix + quantize) and **2-bit MLX** | Mac | **$0** |
| **3**, quality of LLVQ / MLX / `IQ2_XXS` on Metal, plus the C4 cross-backend control | Mac | **$0** |
| **4**, produce the 2-bit GPTQ artifact | HF | ~$0.5–1.0 |
| **5**, CUDA machines: GPTQ (M3) and `IQ2_XXS` (M4) | HF | ~$0.6 |
| | LLVQ (M1) and AWQ (M2): **already measured** | $0 |
| | **total** | **~$1.1–1.6** |

Provenance: *estimated* by analogy with the register: `awq-vllm-4b` $0.11 for 5
rounds, `campagne-8b-qualite` $1.01 for ppl+MMLU on three arms.

**Phase 1 is a precondition, not an option.** As long as C5 is red, no Metal ×
gets published. `docs/fiche-4b.md:438` has said so for a long time: *"the 2.07×
is a ratio against a kernel written by the same author. That is the remaining
hostile angle, unaddressed."* On CUDA, F1 settled the equivalent (1.024× and
1.015× against cuBLAS). On the Metal side, nothing.

## 6. What must be settled before launching

1. **MMLU fixture for llama.cpp.** Its `--multiple-choice` expects a different
   format. Either we build the fixture over the 2,280 questions with the same
   `qhash`, or `IQ2_XXS` carries **ppl only** on the first round.
2. **Scope.** How many arms, and whether MLX is in or out.
3. **The timestamp.** The preregistration is not written. It will be written once
   1 and 2 are settled, and timestamped **before the first millisecond**.

## 7. Two debts to clear while we are on the Mac, near-free

`docs/fiche-4b.md` marks the MLX q4 "129.8 tok/s" and "2.39 GB" as **SUSPECT**:
*no trace, no log, no script, no shell history*. Its §563 repairs them in **~2
min**. The q4 artifact is already local (`~/qwen3-4b-mlx-q4`, 2.1 GB) and
`mlx_lm 0.24.0` is installed.
