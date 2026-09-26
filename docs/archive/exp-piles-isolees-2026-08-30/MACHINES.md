# The machines: two backends, one datasheet per arm (2026-08-30)

> **Nothing is launched.** The commands are *indicative*: those of the machines
> marked ready come from past jobs, those marked to validate must be confirmed
> at launch time.
>
> Single model: **Qwen3-4B**. One size only, see `PROTOCOLE.md` §8.

## 0. The backend rule

**A backend is a stack.** The × do not divide between CUDA and Metal. That holds
*a fortiori*, since the record has already measured that they do not even
divide between **two CUDA cards**: on A100 no decoding arm beats FP16
(`Planes14` 0.79×), and batch G settled that the ×1.78 **is** the clock ratio
2,520/1,410 MHz.

Each backend therefore has **its own f16 control**, and the comparison is
between ratios.

## 1. Which arm runs where

| arm | CUDA (HF Jobs, paid) | Metal (your machine, $0) |
|---|---|---|
| **LLVQ 2-bit** | yes, **served**, `fusedrun` | caveat: **micro-benchmark only**, no `fused_metal.rs` |
| **AWQ 4-bit** | yes, served, vLLM | no engine |
| **GPTQ 2-bit** | caveat: served, vLLM, artifact to produce | no engine |
| **`IQ2_XXS`** | caveat: served, llama.cpp CUDA | caveat: **served, llama.cpp Metal** |
| **MLX 2-bit** | no, does not exist outside Apple | caveat: **served, native**, artifact to produce |
| ~~QTIP~~ | no Qwen3 artifact | no |

**`IQ2_XXS` is the only arm that crosses both backends.** It is the bridge of
the whole experiment: the only format for which we will be able to say "here is
its ratio to its f16 control here, and there".

**The Metal trap is structural.** LLVQ has no served path on Metal: `bin/thesis`
measures **252 projections on one token**, where `mlx_lm.generate` runs a
**whole model over 256 tokens**. Comparing them would be exactly the "two
denominators" fault of §7. On Metal, LLVQ plays only on **quality** and on the
**micro-benchmark**.

---

# Part A, CUDA on HF Jobs

## M1, LLVQ 2-bit, our kernel: ready

- **Artifact**: the sealed 4B, `Planes14` served, `LLVQ_EMBED=q8`.
- **f16 control**: our dense arm, same binary. Caveat: it is **handicapped**,
  `Head::project` → `broadcast_matmul` copies 778 MB of vocabulary per token.
  The asymmetry is **against us**. Hence the rule of the two formulations: the
  served × is never published alone, its **same-head** companion goes in the
  same table.
- **Published configuration**: `LLVQ_ROT_SHARE=0 LLVQ_FUSE=0`. Do not turn on
  the D1 fusion: the three-size tables rest on one identical configuration
  everywhere.

```bash
LLVQ_FUSED_LAYOUT=planes14 LLVQ_EMBED=q8 \
  cargo run --release -p llvq-llm --features cuda --bin fusedrun
```

**Known values**: 87.0 tok/s [86.8–87.0] in 2.56 GB · 5.162 b/param.

## M2, AWQ 4-bit, in vLLM: ready

- vLLM **0.26.0**, pinned image. Speed from `ops/awq_speed.py` (caveat: the
  script says so itself, **sequential rounds, not interleaved**). Quality
  from `ops/awq_dequant.py`, which brings it back to dense f16 in our harness,
  so the **token fingerprint is identical** to M1.
- **Known values**: 200.49 tok/s [200.39–200.61] against a vLLM f16 at
  **83.09** · ppl 13.5207 · MMLU 70.04 · 5.302 b/param.

**This is the control machine of the design.** It is the only arm measured on
both sides, in its own engine *and* in ours. If its two ratios agree, the
hypothesis "the ratio to the control transfers between stacks" becomes
**measured**. This control costs nothing more and is worth more than one extra
arm.

## M3, GPTQ 2-bit, in vLLM: to validate

- **Why**: the **market floor**, what a standard stack really delivers at 2
  bits. The arXiv:2505.02214 study concludes that at 2 bits on Qwen3, only the
  methods with calibration-based compensation hold up.
- **Artifact to produce**: GPTQModel, calibrated on **English C4 shard 1**, the
  shard that `llvq-llm/src/corpus.rs:187` reserves for calibration. **Same
  corpus as LLVQ**: it removes the confounder of the official AWQ, calibrated
  elsewhere.
- Caveat, **in our favor**: at 2 bits vLLM takes the ExLlamaV2 path, less
  optimized, since Marlin is 4 bits only.
- **Work**: `ops/awq_speed.py` already carries
  `ARMS: dict[str, tuple[str, str|None]]` (l. 143) and already greps `"gptq"`
  (l. 296). It takes **one entry**, no new code.

## M4, `IQ2_XXS`, llama.cpp CUDA: to validate

- **2.06 bpw** against our 2.0702, the tightest throughput comparison in the
  record. **`Q2_K` does not fit**: ~2.6–3.0 bpw, no codebook.
- **LUT counterfactual** of `docs/BACKLOG.md` §4.4: a codebook small enough to
  fit in a table, where our 1.1·10¹⁴ points force unfolding to 4.80 b/weight.
- **Artifact produced on Metal** (part B), then **the same GGUF** runs here.

---

# Part B, Metal on your machine, **$0**

> **Ordering principle**: everything that can be settled on Metal is settled
> **before** renting a card. That is what the project has done historically:
> "a free benchmark beats a rented card for finding the measurement traps".

## N0, the Metal equivalent of F1: **the most useful, and it does not exist**

`docs/fiche-4b.md:438` names the angle and declares it unaddressed:

> *"This baseline has moreover never been confronted with MPS, MLX or
> Accelerate: the 2.07× is a ratio against a kernel written by the same author.
> That is the remaining hostile angle, unaddressed."*

On CUDA, **F1 settled exactly that**: our in-house FP16 control is at
**1.024×** (2-arm bench) and **1.015×** (5-arm bench) of cuBLAS on L40S, both
≤ 1.05. That is what holds up *every* published "vs FP16" ratio.

**There is no Metal equivalent.** As long as it is missing, the 2.03–2.09× of
`bin/thesis` is a ratio against a kernel by the same author. Confronting our
Metal FP16 matvec with **MPS / MLX / Accelerate** costs **$0** and is worth
more than any extra arm of this experiment.

## N1, LLVQ on Metal: full quality plus micro-benchmark, to validate

- **Quality**: complete, through `sealed::load`, and it is a **cross-backend
  control**. The same artifact scored on Metal and on CUDA must return the same
  thing. Verified historically: baseline **70.42 (Metal) → 70.32 (CUDA)**, a gap
  of 0.08 σ.
- **Kernel**: `bin/thesis`, one token, 252 matrices, 7 rounds of which 2
  discarded, ratio formed round by round. **2.03–2.09× vs FP16**, 1,105,920
  rows checked against an f64 reference.
- **No end-to-end tok/s**: there is no `fused_metal.rs`.

```bash
cargo run --release -p llvq-metal --bin mslcheck   # the 7 MSL entry points, 3 s
cargo run --release -p llvq-metal --bin thesis     # LLVQ vs FP16, 252 projections
LLVQ_DTYPE=f16 cargo run --release -p llvq-llm --features metal \
  --bin ppl -- 4096 12 metal ~/qwen3-4b-llvq.bin 2> ppl-nll-metal.txt
```

## N2, `IQ2_XXS`, llama.cpp Metal: **the bridge**, to validate

- llama.cpp is **first-class on Metal**: it is the native macOS engine.
- **Local production, $0**: `llama-imatrix` on the same C4 shard 1, then
  `llama-quantize` to `IQ2_XXS`. The GGUF produced here then goes to M4.
- **Memory**: `GGUF bytes ÷ params` **is** the whole-model b/param, measured.
  Better provenance than our arm, whose embedding is *modeled* at 8.5 b/param,
  to be labelled as such.
- Caveat, **gap**: no MMLU harness. Its `--multiple-choice` expects a different
  format; replaying our 2,280 questions at the same `qhash` requires the
  fixture. **Pending decision**: build the fixture, or ppl alone on the first
  round.

## N3, MLX 2-bit, native Apple: to validate

- **Checked on 2026-08-30**: `mlx_lm.convert.convert(hf_path, mlx_path,
  quantize, q_group_size, q_bits, dtype, …, dequantize, quant_predicate)`.
  `q_bits` exists, **mlx_lm 0.24.0 is installed**, and the **q4 artifact is
  already local** (`~/qwen3-4b-mlx-q4`, 2.1 GB).
- **`dequantize` is a parameter**: MLX can return a dense f16, so the quality of
  an MLX artifact is scorable **in our harness**, at identical fingerprint. That
  is the path described by `fiche-4b.md:556`.
- Caveat, **asymmetry already documented and never stated**: **MLX quantizes the
  embedding TOO**, 253 tensors = 252 projections + `model.embed_tokens`
  (`fiche-4b.md:339`). The right comparator for our artifact is "q4 on the
  linears + f16 embedding", **not** the MLX file as it stands.

**Two debts to clear along the way, near-free.** `fiche-4b.md` marks the MLX q4
"129.8 tok/s" and "2.39 GB" as **SUSPECT**: *no trace, no log, no script, no
shell history*. Its §563 repairs them in **~2 min** with
`/usr/bin/time -l mlx_lm.generate … | tee`. To be done while we are on the
machine.

---

## Out of scope, QTIP

Checked on 2026-08-30: **relaxml publishes QTIP for the Llama family only**. No
Qwen3 checkpoint; porting it would cost *estimated* $10–20 plus a risk of
architecture incompatibility.

**Its speed is already measured, and better than this experiment would do it**:
F2 timed it in **a single process**, arms interleaved, `t(Planes14) ÷ t(QTIP)`
= **2.27× [2.27–2.28]**, division **licit**. Giving it its own machine would be
a step backwards. What it still lacks is its **quality**.
