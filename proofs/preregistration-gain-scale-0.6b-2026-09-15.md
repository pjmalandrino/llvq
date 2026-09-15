# Preregistration — is the served reconstruction biased small, and does undoing it help? Qwen3-0.6B

**Written, committed and TIMESTAMPED on 2026-09-15, BEFORE the sweep.**
Operator go given 2026-09-15.

🚨 **This file is not edited again**: the stamp attests these bytes at this date.
A fact it gets wrong is written *beside* it, in a `-ECARTS.md` (CLAUDE.md §7).

**What is measured**: `LLVQ_GAIN_SCALE`, added this day, which multiplies the
fitted gain centroids of every matrix. At 1.0 the multiply is skipped, so the
published path is bit-identical.

**Cost: $0, about 1 h 05 of Mac** (4 arms; *estimated* from the 15 and 16 min
the two arms of this morning took on this protocol). No other Metal job beside it.

---

## 1. Where this comes from

This morning's arm closed the opposite lead: the Euclidean gain rule, which
*shrinks* every block by 2.93 % on average, cost **+4.026 % of perplexity**
(*measured*, [journal](../docs/mesures/tetrapost-ppl-0.6b-2026-09-15.txt)). That
is a measurement of this knob, in the wrong direction and by accident.

The served rule is not unbiased either. On the same 2,016 compensated blocks it
places a mean amplitude of **0.99336** of the block norm — a residual 0.66 %
shrink. If perplexity is monotone in that bias near zero, removing it is worth
something, and it is **free**: the centroids are already fitted per matrix, so
scaling them changes no bit, no table, no kernel, no format. That is the shape of
lever the quality axis is looking for.

A two-block pilot run before writing this (8×2048 calibration, 4 evaluation
windows, blocks 0–1 only — *not* the protocol below) read 41.2339 at 1.00 and
24.4208 at 1.05. That is what widened the sweep from [1.000, 1.015] to the grid
of §3. It is a knob-check, not a result: different calibration, different
evaluation, 2 quantized blocks of 28.

## 2. The setup

Identical to this morning's arms, one variable: the multiplier.

```
LLVQ_MODEL=Qwen/Qwen3-0.6B LLVQ_GAIN_SCALE=<s> LLVQ_THREADS=12 nice -n 10 \
  cargo run --release -p llvq-llm --features metal,fast-linalg --bin smoke -- \
  64 2048 12 2048 metal nogs tetra 999 rot
```

Qwen3-0.6B, 28 blocks, calibration wikitext-2 train 64×2048 prefix, evaluation 12
windows ×2048, f32, Metal, nogs, rotation on, damping 1e-2, rho = 1.

## 3. The grid, fixed now

**s ∈ {1.00, 1.02, 1.05, 1.08}**, run in that order. 1.00 is the control.

If the best of the four is at an edge of the grid, the grid was wrong and one
neighbouring point may be added — that extension is declared here so that adding
it later is not a free choice, and any point added is reported as an extension.

## 4. Signed prediction

**The minimum sits strictly above 1.00, in [1.01, 1.10], and beats the control by
at least 2 % of perplexity.**

Named against me: if **1.00 is the best of the four**, the hypothesis is dead and
the two-block pilot was a regime artefact. If the curve is flat within ±0.5 %
across the whole grid, same verdict.

## 5. The decision rule, written before the first number

| Result | Action |
|---|---|
| A point beats 1.00 by ≥ 2 % and the curve has an interior minimum | Continue: a per-matrix bias map, then a 4B arm with its own stamped prereg |
| Best point beats 1.00 by 0.5 % to 2 % | Report it, and do not spend 4B time on it until the map explains it |
| 1.00 is best, or the spread is under 0.5 % | Close. The residual bias is not a lever |

## 6. Controls

1. **s = 1.00 must replay 41.8875 exactly**, this morning's `tetra` arm. Anything
   else means the knob touches the published path, and the sweep is void.
2. Effective rate 2.1656 b/weight on all four arms: the knob spends no bit.
3. f32 baseline 19.5038 on all four arms.
4. Raw logs kept whole, never summarized before commit.

## 7. What a positive result would and would not license

It licenses a per-matrix bias map and one 4B arm on perplexity. It licenses **no
MMLU claim** (sampling error 1.339 pp), and **no change to the served encoder**,
which is an operator decision on a fundamental criterion.

It would also not, by itself, explain anything. A multiplier that helps is a
symptom of a reconstruction that is biased small; the map of §5 is what would
turn the symptom into a mechanism.
