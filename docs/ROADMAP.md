# Roadmap

What comes next, with its gate and its cost. State as of 2026-11-07. Where things stand is in [`ETAT.md`](ETAT.md),
the rules in [`METHODE.md`](METHODE.md). The quality leads of September are listed in
[`ROADMAP-QUALITY.md`](ROADMAP-QUALITY.md).

Rule 1 applies to every item: no run starts or stops, and no structural decision is taken, without an explicit go,
with the cost announced before and the running total after.

## 1. Starting point

The three sealed files are served by 4E v9 on CUDA at 221.5, 142.5 and 86.3 tok/s, and the 4B by MLX on the Mac
([`ETAT.md`](ETAT.md) §2). They score 4.76, 4.21 and 2.46 MMLU points below 4-bit AWQ for 45 to 52 % of its bits per
parameter.

An A/B that recalibrates is read against the draw's noise, 5.2 % of perplexity and 2.92 pp of MMLU. An A/B on one file
is read against 0.43 pp and 0.12 %.

## 2. Next, in order

### 2.1 Throughput: speculative decoding and batching

At the bit, the 4B tops out near 235 to 258 tok/s at the door (*estimated*, [`plan-300.md`](plan-300.md) §6b). Past
that, two levers change what a launch does rather than how fast it does it.

- **Speculative decoding: the verify gate is met, the memory is the price.** A tree pass verifies 16 rows for 1.73
  tokens of 4E, an extra row under 0.02 of a token (*measured*,
  [verify-tree-n11](mesures/verify-tree-n11-2026-11-10.txt)); the gate was 0.2. With the DFlash drafter the loop
  decodes GSM8K-200 at x2.30 and chat at x1.09 of 4E, for 5.41 GB against 2.01 (*measured*,
  [spec-loop-sp8](mesures/spec-loop-sp8-2026-11-11.txt), [spec-mem-m3](mesures/spec-mem-m3-2026-11-11.txt)).
  Operator, 2026-11-11: set aside as a fast profile, built when wanted; the default stays at 2.73 b/param.
- **Batching.** Decode several sequences per pass: the bytes are read once for all of them. vLLM's AWQ 4B reaches
  5,129 tok/s in aggregate at batch 32 on the L40S (*measured*, branch `claude/batching-eca5f1`, job 6aef873c).
  Our batched decode and its continuous batching in `llvq serve` are on main since 2026-11-09, licensed and not served by default. **Gate**: each sequence returns
  its batch-1 tokens, or a stated tolerance; aggregate tok/s against AWQ at the same batch.

### 2.2 Score the served object

**Gate**: the census through the kernel lands within the paired bar of the dense reconstruction, or the paper's
tables move. **Cost**: about $20 by the mixed route, $70 for the full census at three sizes (*estimated*).

Every MMLU we report is read on the dense reconstruction with f16 tables. The one census through the kernel, on an
older 4B object, read 55.66 against 55.52 dense, McNemar p = 0.25 (*measured*,
[f1e-census](mesures/f1e-census-2026-09-11.txt)). `ppl` does not read `LLVQ_CONFIG` yet, so code comes first.

### 2.3 A full GSM8K census under 4E

**Gate**: paired against the expanded census, as the grouped census was. **Cost**: about $2 at 4B (the grouped census
cost $2.02), more at 8B and 14B.

### 2.4 `down_proj` instead of `o_proj` in int4, at 4B and 14B

**Gate**: the paired gain against the shipped file clears 0.43 pp. **Cost**: about $2 a size, no encoding.

At 8B, `down_proj` of layers 10 to 26 scored 0.77 points better [0.26, 1.28] than `o_proj` plus `down_proj`, with
fewer bytes (*measured*, [sealed-8b-27](mesures/sealed-8b-27-2026-09-24.txt)). At 4B all of `down_proj` would go over
b_max, at 3.2286 b/param (*measured*, [q5-alloc-int4](mesures/q5-alloc-int4-2026-09-16.txt)), so the split needs
choosing.

### 2.5 A second calibration draw per size

**Gate**: the three levels move by less than 2.92 pp. **Cost**: one encode and one row-scale training a size; the 14B
took $15.35 and $9.78 (*measured*, `docs/data/jobs.csv`). The 4B and 8B encode on the Mac at $0.

### 2.6 A consumer GPU

**Gate**: written before the run. On the A100 the served files lost to vLLM's FP16 (*measured*,
[a100](mesures/a100-2026-10-06.txt)). A consumer card is GDDR like the L40S, with less compute. **Cost**: under the
A100 run's $3.04 (*estimated*).

### 2.7 An int4 KV cache

**Gate**: its own prereg, because it moves the logits: paired perplexity, a light MMLU and a task past 1,280 tokens.
**Cost**: $3 to $5 of card after 2 to 4 weeks of code (*estimated*). At 8,192 tokens it takes the cache from 1.21 to
about 0.34 GB, and it is the only lead that reaches 300 tok/s at that context ([`plan-300.md`](plan-300.md) §7).

### 2.8 Metal speed

The MLX door serves the 4B; its own speed is not measured yet, and the door holds q and k words twice (about 0.12 GB).
The Rust Metal path decodes 52.1 tok/s under a form `served.rs` accepts; its 65.5 tok/s arm uses the grouped form,
which Metal refuses. The MLX prototype decodes 93.9 (*measured*, [metal-nuit](mesures/metal-nuit-2026-11-06.txt)). The steps are in [`plan-metal-debit.md`](plan-metal-debit.md) §4.
**Cost**: $0, local.

### 2.9 `transformers` upstream

**Gate**: the maintainers accept or refuse the method in tree. **Cost**: $0. `llvq-tetra` 0.1.0 is on PyPI. The issue
and the PR go together, from a fork; the drafts are in `upstream/transformers-llvq-tetra/`
([`plan-transformers.md`](plan-transformers.md)).

## 3. Debt

- `[workspace.lints.rust] unsafe_code = "forbid"` on the five core crates. `#![forbid]` in a `lib.rs` does not cover
  integration tests.
- No host compilation of the `.cuh` files by `clang++` in CI.
- Two timestamps no longer attest their file (2026-08-10 and 08-11, rewritten by `01fdbe6`), and
  `f5-graines-4b-2026-08-19.v1-l4x4.md.ots` has no `.md`. Nothing repairs this.
- Nine preregistrations have no `.ots`: 2026-08-13, `p2`, `p3` and `p4` of 08-14, `e1v-cuda` of 08-15, `awq-vllm` and
  `fusedrun14b` of 08-17, `variance-calibration` of 08-26 and `gain-mr-4b` of 09-16.
- `llvq-cuda/kernels/llvq_tetra48.cuh` line 85 says `n2 ≤ 432`; the bound is 416. Fixing it moves the served unit's
  digest, so it waits for the next kernel change.
- The HF bucket has never been inventoried (69 files, 46.7 GB on 2026-08-17).
- [`ETAT.md`](ETAT.md) is maintained by hand.
- No tag points at the deposited commit `e21a8bb`; `v0.0.1` points at its child `16c9c8b`.

## 4. On hold

- **MoE.** Qwen3-30B-A3B chosen, gpt-oss ruled out. A policy for experts below full rank is missing (*measured*,
  [moe-routing](mesures/moe-routing-gptoss20b-2026-08-12.txt)). About $1.4 to open, $69 to serve (*estimated*).
- **The 32B.** About $62 and 11.4 h on `rtx-pro-6000x2` to encode (*estimated*). The `rot_apply` wall is lifted
  (*measured*, [lot1](mesures/lot1-4b-2026-11-04.txt)). A gate on the 14B-to-32B gap comes first.
- **Prefill by GEMM.** The 4B's first token at 8,192 takes 35 s. Batched prefill moves the logits, so it needs the T8
  band.

## 5. Decisions awaited

| decision | default if silent |
|---|---|
| a spend cap for the next campaign | no paid job |
| the served census, $20 or $70 | neither |
| the v9 config on the Hub's moving repository | not published |
| the Rust Metal door's grouped and kernel forms | refused |
| the upstream `transformers` issue and PR | not posted |
| the next venue for paper 2 | preprint only |
| a harder reasoning test, about $3 to $5 at 4B | not done |
| `ots upgrade` after each new stamp | stamps sit un-upgraded |
