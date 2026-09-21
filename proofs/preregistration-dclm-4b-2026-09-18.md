# Preregistration. The paper's calibration corpus, at our volume

**Written, committed and TIMESTAMPED on 2026-09-18, BEFORE the MMLU arm.**
Operator go given 2026-09-18. Cost announced before the go: about 24 min on l40sx1, **$0.72**,
timeout capped at 1 h. The encoding itself is already done: 1 h 47 of Mac, $0.

Not edited again. A fact it gets wrong goes beside it, in a `-ECARTS.md`.

## 1. One variable

The served object was calibrated on **C4**. The paper calibrates on **DCLM-edu**, and
`llvq-llm/src/corpus.rs` states the hypothesis this arm tests: "DCLM-edu is web text filtered
for educational content, which is the domain MMLU examines, so the calibration corpus is a
candidate for that dissociation." The dissociation is 5.1 points.

The re-encoding done today changes the corpus and nothing else: `tetra1`, rotation seed
`0x110feed`, `nogs`, `h_shrink` 1, `gain_scale` 1, `LLVQ_INT4_TYPES=v_proj`, 64 x 2048 =
131,072 tokens. The same recipe that produced the served file, on a different corpus.

Sealed to 1,794,564,765 bytes, **the same byte count as the served object**, sha256
`471f39883b0baabc42b90c83...` against the served `ae31087a...`.

## 2. What one arm can and cannot say, stated before the number

This is a **re-encoding** comparison, not a constant-file one. The repository's measured noise
in that regime is **2.92 pp** between calibration draws at the 4B (job `6a8df156`, 2026-08-25),
against 0.43 pp at constant file.

So a single draw resolves only an effect of **8.1 pp**. It cannot confirm the +0.8 to +1.4 pp
that row 6 estimates, and it cannot refute it either. What it can do is see a large effect if
one exists, which is precisely what the `corpus.rs` hypothesis predicts.

This prereg therefore registers **no significance claim**. It registers a point estimate and
three readings.

## 3. The configuration

Reference: the committed `mmlu-q5-shipped-FULL.csv`, the C4-calibrated served object, which has
reproduced byte for byte across three jobs.
Treatment: `/out/dclm-4b-2026-09-18/qwen3-4b-dclm.bin`, no restoration, full split,
`LLVQ_MMLU_ALLOC=flat`, CUDA, a dump per question.

One arm. The reference is inherited rather than re-run, and unlike the earlier confirmations
the two arms are **different files**, so sharing a card would not make them constant-file
anyway.

## 4. Signed prediction

| quantity | point | interval |
|---|---|---|
| dclm micro, full split | **56.9** | [53.5, 60.3] |
| gain over the C4 object | +0.5 | [−2.9, +3.9] |

The point is the C4 object's 56.37 plus the perplexity signal, which is +0.46 % in dclm's
favour (16.2415 against 16.3161, same accounting, both Metal f32). A 0.46 % perplexity move is
worth well under a point of MMLU on every precedent in this repository, and two precedents have
it worth a **negative** MMLU move.

The interval is the 2.92 pp noise band, and it is wide on purpose.

Named against me: above 60.3 the corpus alone carries most of the 5.1-point dissociation, which
would be the largest free gain the project has found. Below 53.5 the paper's own corpus is
worse than C4 for us, which would need explaining.

## 5. The three readings

| result | reading |
|---|---|
| >= +3.9 pp | The corpus carries a large share of the gap. Fund two more draws per corpus at once |
| −2.9 to +3.9 | Indistinguishable from an encoding draw. **The corpus alone does not explain the gap**, and the hypothesis moves to the volume, where the factor is 95 and the cost is now measured at 9 h of Mac |
| <= −2.9 | The paper's corpus is worse here. Record it and look at what else differs |

## 6. Controls

1. The dclm file is byte-identical in size to the served object and differs in sha256.
2. 14,042 questions scored, plan fingerprint `a74a6d6213602979`.
3. The dump kept whole and committed.
4. `oracle` first on the backend, hard rule 10.

## 7. What it will not establish

- No significance either way on a +1 pp effect: one draw, 2.92 pp of noise.
- Nothing about the volume, which is the other half of row 6 and is not varied here.
- Nothing about perplexity beyond the 16.2415 already measured at encoding time.
- Nothing about the 8B or any other size.
