# Preregistration. Confirming down_proj at int4 g128, Qwen3-4B

**Written, committed and TIMESTAMPED on 2026-09-18, BEFORE the confirmation run.**
Operator go given 2026-09-18. Cost announced before the go: about 48 min on l40sx1,
**$1.44**, timeout capped at 2 h.

Not edited again. A fact it gets wrong goes beside it, in a `-ECARTS.md`.

## 1. What selected this arm, and why it was blocked until today

The same six-type exploration that selected `o_proj`
(`docs/mesures/q5-alloc-int4-2026-09-16.txt`). Its table carries individual 95 % intervals
over six comparisons with no multiplicity correction, so it selects a candidate and
establishes nothing. `down_proj` measured the **largest** gain of the six, **+3.79 pp**
[+1.52; +6.11].

It was not confirmed then because it read as over budget. That verdict was a unit error. The
exploration journal set whole-model b/param against a `b_max` defined in kernel b/weight, and
the `o_proj` journal of 2026-09-17 caught this for its own row and redid only that row. Redone
for `down_proj` (*computed*, `docs/etat-reconcilie-2026-09-17.md` section 3, on 3,633,315,840
projection weights with `Tetra` at 2.1498 and int4 g128 at 4.250):

  int4 served natively     2.2044 -> **2.7226** kernel b/weight, under 3.00
  int4 dequantized to f16  2.5095 -> 5.9271 kernel b/weight, far over

So the arm fits on the same condition `o_proj` fits on: a native int4 kernel for the shape.
That kernel has run on `v_proj`'s 1024 x 2560 and on no other. `down_proj` is 9728 x 2560,
896,532,480 weights, 2.37 times `o_proj`'s.

In whole-model b/param, the unit hard rule 6 asks for in comparisons, it is 2.7645 -> 3.2286.

## 2. The configuration, frozen now

Reference arm: `/out/tetra-q5-2026-09-09/qwen3-4b-tetra-q5.bin`, sha256
`ae31087a4b72494d52394f2cf2070da37a845c0a7d4aded65d5bdb0bd18b2263`, no restoration.
Treatment arm: the same file, `LLVQ_RESTORE_Q4=down_proj`, `LLVQ_MODEL=Qwen/Qwen3-4B`.
Both: `LLVQ_MMLU_ALLOC=flat`, full split, CUDA, a dump per question.

The reference arm is re-run rather than paired against the committed
`mmlu-q5-shipped-FULL.csv` of 2026-09-17. Pairing across jobs would save about $0.72 and would
compare two arms that never shared a card, which control 1 exists to prevent.

## 3. The primary result, named before the numbers

**The paired gain on the 11,762 questions that took no part in the selection**, with its
interval, stratified bootstrap without finite population correction, the reading the `o_proj`
run fixed in its E1 before the number existed.

The 2,280 selection questions are reported separately and are not the result. The full 14,042
figure is published too, on the same line as the statement that it CONTAINS the 2,280 and is
therefore not independent.

## 4. Signed prediction

**+1.5 to +3.0 pp on the 11,762 held-out questions**, point estimate **+2.1**, interval
excluding zero.

The reasoning, with its single precedent. `o_proj` explored at +2.77, reproduced +3.12 on the
2,280 selection questions, and confirmed **+1.55** held-out: the held-out gain was 0.50 of the
selection gain. `down_proj` explored at +3.79, so half of a similar reproduction lands near
+2.1. One precedent is not a law, which is why the interval is wide.

Named against me: a point estimate at or above +3.79 would mean the selection bias did not
operate on the largest of six arms, which would itself need explaining.

## 5. The decision rule

| Held-out result | Action |
|---|---|
| >= +1.5 pp, interval excludes zero | `down_proj` at int4 is a real gain that fits in kernel b/weight; the operator decides on adoption |
| positive, interval contains zero | Not confirmed. Report and stop |
| < +1.5 pp with an interval excluding zero | Confirmed but smaller than `o_proj` per bit: `down_proj` costs 2.37 times `o_proj`'s weights for that gain, and the comparison per bit goes in the journal |
| <= 0 | The exploration selected noise on its largest arm. Record it against the six-arm table |

## 6. Controls

1. Identical run fingerprint on both arms, question by question.
2. The reference arm reproduces the census of 2026-09-11 and the shipped arm of 2026-09-17,
   56.37 micro and 58.42 macro, pick for pick on the 14,042.
3. 14,042 questions scored on both arms; any shortfall voids the run.
4. Both dumps kept whole and committed.
5. `oracle` first on the backend, hard rule 10.

## 7. What it will not establish

- Nothing about serving it. The quality is measured by dense reconstruction of a dequantized
  int4 tensor, which prices four bits of information and is not the served path. No file
  carries `down_proj` as int4 records, and no kernel has run on its shape.
- Nothing about combining it with `o_proj`. The two are not additive, and v + o + down reads
  2.9408 kernel b/weight (*computed*), which fits but is not measured here.
- Nothing about perplexity, which is not measured.
- Nothing about another model or another size.
